"""Unit tests for VideoCollectionDataset and VideoCollectionDataLoader."""

import pytest
import torch
from pathlib import Path

from pvio.torch import VideoCollectionDataLoader, VideoCollectionDataset
from pvio.video_io import write_frames_to_video

from .test_utils import make_frames_with_stride


def test_extract_frame_number_success_and_errors():
    """Test frame number extraction from filenames."""
    # Simple successful extraction
    assert VideoCollectionDataset._extract_frame_number("img_001.png", r"(\d+)") == 1

    # Multiple matches -> error
    with pytest.raises(ValueError):
        VideoCollectionDataset._extract_frame_number("img_01_02.png", r"(\d+)")

    # Non-integer match -> error
    with pytest.raises(ValueError):
        VideoCollectionDataset._extract_frame_number("frame_x.png", r"(x)")


class DummyDataset(VideoCollectionDataset):
    """Dummy dataset for testing without actual video files."""

    def __init__(self):
        # Bypass parent initialization checks
        self.video_paths = []

    def assign_workers(
        self,
        n_frame_loading_workers: int,
        n_metadata_indexing_workers: int = -1,
        chunk_size: int = 1000,
    ):
        # No-op for tests to avoid expensive metadata indexing
        return

    def __iter__(self):
        # Yield three fake frame dicts
        for i in range(3):
            yield {
                "frame": torch.ones(3, 4, 5) * i,
                "video_path": f"video_{i}",
                "frame_idx": i,
            }


def test_collate_stacks_frames_correctly():
    """Test that the collate function stacks frames correctly."""
    ds = DummyDataset()
    loader = VideoCollectionDataLoader(ds, batch_size=3, num_workers=0)

    batch = next(iter(loader))
    assert "frames" in batch and "video_paths" in batch and "frame_indices" in batch
    assert batch["frames"].shape == (3, 3, 4, 5)

    # Check values
    assert torch.equal(batch["frames"][0], torch.zeros(3, 4, 5))
    assert torch.equal(batch["frames"][2], torch.ones(3, 4, 5) * 2)


def test_dataloader_init_type_and_kwargs_validation():
    """Test dataloader initialization validation."""
    # Wrong dataset type
    with pytest.raises(ValueError):
        VideoCollectionDataLoader(object(), batch_size=1)

    ds = DummyDataset()

    # Custom batch_sampler not allowed
    with pytest.raises(ValueError):
        VideoCollectionDataLoader(ds, batch_size=1, batch_sampler=[1])

    # Custom collate_fn not allowed
    with pytest.raises(ValueError):
        VideoCollectionDataLoader(ds, batch_size=1, collate_fn=lambda x: x)


def test_vcd_as_image_dirs_sorting_and_regex(tmp_path: Path):
    """Test image directory sorting with and without regex."""
    # Create files with unordered names
    d = tmp_path / "frames"
    d.mkdir()
    (d / "b.png").write_bytes(b"x")
    (d / "a.png").write_bytes(b"x")
    (d / "c.png").write_bytes(b"x")

    ds = VideoCollectionDataset([d], as_image_dirs=True)
    path_key = d.absolute().as_posix()
    files = ds.frame_sortings[path_key]
    assert [f.name for f in files] == ["a.png", "b.png", "c.png"]

    # Regex sorting
    d2 = tmp_path / "frames2"
    d2.mkdir()
    (d2 / "frame_2.png").write_bytes(b"x")
    (d2 / "frame_10.png").write_bytes(b"x")
    (d2 / "frame_1.png").write_bytes(b"x")

    ds2 = VideoCollectionDataset([d2], as_image_dirs=True, frame_sorting=r"(\d+)")
    path_key2 = d2.absolute().as_posix()
    files2 = ds2.frame_sortings[path_key2]
    assert [f.name for f in files2] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_assign_workers_creates_chunks(tmp_path: Path):
    """Test that videos are split into chunks correctly."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"

    write_frames_to_video(v1, make_frames_with_stride(25), fps=5.0)
    write_frames_to_video(v2, make_frames_with_stride(15), fps=5.0)

    ds = VideoCollectionDataset([v1, v2], as_image_dirs=False)
    ds.assign_workers(
        n_frame_loading_workers=2, n_metadata_indexing_workers=1, chunk_size=10
    )

    # Check n_frames_lookup uses absolute POSIX strings
    v1_key = v1.absolute().as_posix()
    v2_key = v2.absolute().as_posix()
    assert v1_key in ds.n_frames_lookup
    assert v2_key in ds.n_frames_lookup
    assert ds.n_frames_lookup[v1_key] == 25
    assert ds.n_frames_lookup[v2_key] == 15

    # Collect all chunks across workers
    all_chunks = []
    for worker_id in ds.worker_assignments:
        all_chunks.extend(ds.worker_assignments[worker_id])

    # Should have 3 chunks for v1 (0-10, 10-20, 20-25) and 2 for v2 (0-10, 10-15)
    assert len(all_chunks) == 5

    # Verify chunk structure
    v1_chunks = [c for c in all_chunks if c[0] == v1_key]
    v2_chunks = [c for c in all_chunks if c[0] == v2_key]

    assert len(v1_chunks) == 3
    assert len(v2_chunks) == 2

    # Check v1 chunks cover all frames
    v1_chunks_sorted = sorted(v1_chunks, key=lambda x: x[1])
    assert v1_chunks_sorted[0] == (v1_key, 0, 10)
    assert v1_chunks_sorted[1] == (v1_key, 10, 20)
    assert v1_chunks_sorted[2] == (v1_key, 20, 25)

    # Check v2 chunks
    v2_chunks_sorted = sorted(v2_chunks, key=lambda x: x[1])
    assert v2_chunks_sorted[0] == (v2_key, 0, 10)
    assert v2_chunks_sorted[1] == (v2_key, 10, 15)


def test_assign_workers_contiguous_chunks_per_worker(tmp_path: Path):
    """Test that workers get contiguous chunks to minimize seeking."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(40), fps=5.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)
    ds.assign_workers(
        n_frame_loading_workers=2, n_metadata_indexing_workers=1, chunk_size=10
    )

    # With 40 frames and chunk_size=10, we get 4 chunks
    # With 2 workers, each should get 2 contiguous chunks
    assert len(ds.worker_assignments) == 2
    worker_0_chunks = ds.worker_assignments[0]
    worker_1_chunks = ds.worker_assignments[1]

    assert len(worker_0_chunks) == 2
    assert len(worker_1_chunks) == 2

    # Verify contiguity - worker 0 should get first 2 chunks
    assert worker_0_chunks[0][1] < worker_0_chunks[1][1]  # Start indices increasing
    assert (
        worker_0_chunks[0][2] == worker_0_chunks[1][1]
    )  # End of chunk 0 = start of chunk 1


def test_chunks_handle_videos_shorter_than_chunk_size(tmp_path: Path):
    """Test that videos shorter than chunk_size work correctly."""
    v1 = tmp_path / "short.mp4"
    write_frames_to_video(v1, make_frames_with_stride(5), fps=5.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)
    ds.assign_workers(
        n_frame_loading_workers=2, n_metadata_indexing_workers=1, chunk_size=100
    )

    # Should create exactly 1 chunk covering all frames
    all_chunks = []
    for worker_chunks in ds.worker_assignments.values():
        all_chunks.extend(worker_chunks)

    assert len(all_chunks) == 1
    v1_key = v1.absolute().as_posix()
    assert all_chunks[0] == (v1_key, 0, 5)


def test_buffer_size_parameter():
    """Test that buffer_size parameter is accepted and stored."""
    ds = VideoCollectionDataset([], as_image_dirs=False, buffer_size=64)
    assert ds.buffer_size == 64

    # Default value
    ds2 = VideoCollectionDataset([], as_image_dirs=False)
    assert ds2.buffer_size == 32


def test_chunk_size_parameter_in_dataloader(tmp_path: Path):
    """Test that chunk_size is passed correctly to assign_workers."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(30), fps=5.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)

    # Create dataloader with custom chunk_size
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0, chunk_size=10)

    # Should have 3 chunks (0-10, 10-20, 20-30)
    all_chunks = []
    for worker_chunks in ds.worker_assignments.values():
        all_chunks.extend(worker_chunks)

    assert len(all_chunks) == 3


def test_image_dirs_with_chunks(tmp_path: Path):
    """Test that image directories work correctly with chunking."""
    d = tmp_path / "frames"
    d.mkdir()

    # Create 15 dummy image files
    for i in range(15):
        (d / f"frame_{i:03d}.png").write_bytes(b"x")

    ds = VideoCollectionDataset([d], as_image_dirs=True, frame_sorting=r"(\d+)")
    ds.assign_workers(n_frame_loading_workers=2, chunk_size=5)

    # Should create 3 chunks (0-5, 5-10, 10-15)
    all_chunks = []
    for worker_chunks in ds.worker_assignments.values():
        all_chunks.extend(worker_chunks)

    assert len(all_chunks) == 3

    # Verify chunks cover correct frame ranges
    d_key = d.absolute().as_posix()
    chunks_for_dir = [c for c in all_chunks if c[0] == d_key]
    chunks_sorted = sorted(chunks_for_dir, key=lambda x: x[1])

    assert chunks_sorted[0] == (d_key, 0, 5)
    assert chunks_sorted[1] == (d_key, 5, 10)
    assert chunks_sorted[2] == (d_key, 10, 15)


def test_multiple_videos_chunk_distribution(tmp_path: Path):
    """Test that chunks from multiple videos are distributed across workers."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    v3 = tmp_path / "v3.mp4"

    write_frames_to_video(v1, make_frames_with_stride(12), fps=5.0)
    write_frames_to_video(v2, make_frames_with_stride(12), fps=5.0)
    write_frames_to_video(v3, make_frames_with_stride(12), fps=5.0)

    ds = VideoCollectionDataset([v1, v2, v3], as_image_dirs=False)
    ds.assign_workers(
        n_frame_loading_workers=3, n_metadata_indexing_workers=1, chunk_size=10
    )

    # 3 videos * 2 chunks each = 6 chunks total
    all_chunks = []
    for worker_chunks in ds.worker_assignments.values():
        all_chunks.extend(worker_chunks)

    # 12 frames / 10 chunk_size = 2 chunks per video (0-10, 10-12)
    assert len(all_chunks) == 6

    # Verify all videos are represented
    v1_key = v1.absolute().as_posix()
    v2_key = v2.absolute().as_posix()
    v3_key = v3.absolute().as_posix()

    video_paths_in_chunks = {c[0] for c in all_chunks}
    assert video_paths_in_chunks == {v1_key, v2_key, v3_key}

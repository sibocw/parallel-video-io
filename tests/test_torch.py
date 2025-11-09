"""Unit tests for VideoCollectionDataset and VideoCollectionDataLoader."""

import pytest
import torch
from pathlib import Path

from pvio.torch import (
    VideoCollectionDataLoader,
    VideoCollectionDataset,
    SimpleVideoCollectionLoader,
)
from pvio.video_io import write_frames_to_video

from .test_utils import make_frames_with_stride


def test_extract_frame_number_success_and_errors():
    """Test frame number extraction from filenames."""
    from pvio.torch import _get_frame_idx_from_filename

    # Simple successful extraction
    assert _get_frame_idx_from_filename("img_001.png", r"(\d+)") == 1

    # Multiple matches -> error
    with pytest.raises(ValueError):
        _get_frame_idx_from_filename("img_01_02.png", r"(\d+)")

    # Non-integer match -> error
    with pytest.raises(ValueError):
        _get_frame_idx_from_filename("frame_x.png", r"(x)")


class DummyDataset(VideoCollectionDataset):
    """Dummy dataset for testing without actual video files."""

    def __init__(self):
        # Bypass parent initialization checks
        self.video_paths = []

    def assign_workers(self, n_loading_workers: int):
        # No-op for tests to avoid expensive metadata indexing
        self.worker_assignments = [[] for _ in range(n_loading_workers)]
        return

    def __iter__(self):
        # Yield three fake frame dicts
        for i in range(3):
            yield {
                "frame": torch.ones(3, 4, 5) * i,
                "video_idx": i,
                "frame_idx": i,
            }


def test_collate_stacks_frames_correctly():
    """Test that the collate function stacks frames correctly."""
    ds = DummyDataset()
    loader = VideoCollectionDataLoader(ds, batch_size=3, num_workers=0)

    batch = next(iter(loader))
    assert "frames" in batch and "video_indices" in batch and "frame_indices" in batch
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
    # The first (and only) directory's sorted files
    files = ds.sorted_frame_paths[0]
    assert [f.name for f in files] == ["a.png", "b.png", "c.png"]

    # Regex sorting
    d2 = tmp_path / "frames2"
    d2.mkdir()
    (d2 / "frame_2.png").write_bytes(b"x")
    (d2 / "frame_10.png").write_bytes(b"x")
    (d2 / "frame_1.png").write_bytes(b"x")

    ds2 = VideoCollectionDataset([d2], as_image_dirs=True, frame_sorting=r"(\d+)")
    # The first (and only) directory's sorted files
    files2 = ds2.sorted_frame_paths[0]
    assert [f.name for f in files2] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_assign_workers_creates_assignments(tmp_path: Path):
    """Test that videos are assigned to workers correctly."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"

    write_frames_to_video(v1, make_frames_with_stride(25), fps=5.0)
    write_frames_to_video(v2, make_frames_with_stride(15), fps=5.0)

    ds = VideoCollectionDataset([v1, v2], as_image_dirs=False)
    ds.assign_workers(n_loading_workers=2)

    # Check that frame counts are correct
    assert ds.n_frames_by_video == [25, 15]
    assert ds.n_frames_total == 40

    # Collect all assignments across workers
    all_assignments = []
    for worker_id in range(len(ds.worker_assignments)):
        all_assignments.extend(ds.worker_assignments[worker_id])

    # Verify all frames are covered
    total_frames_assigned = 0
    for video_idx, start_frame, end_frame in all_assignments:
        total_frames_assigned += end_frame - start_frame
        # Verify video indices are valid
        assert 0 <= video_idx < len(ds.video_paths)
        # Verify frame ranges are valid
        assert 0 <= start_frame < end_frame <= ds.n_frames_by_video[video_idx]

    assert total_frames_assigned == 40


def test_assign_workers_balanced_distribution(tmp_path: Path):
    """Test that workers get balanced frame assignments."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(600), fps=5.0)  # More frames

    # Use smaller min_frames_per_worker to allow more workers
    ds = VideoCollectionDataset([v1], as_image_dirs=False, min_frames_per_worker=50)
    ds.assign_workers(n_loading_workers=2)

    # With 600 frames and 2 workers, each should get 300 frames
    assert len(ds.worker_assignments) == 2

    frames_per_worker = []
    for worker_assignments in ds.worker_assignments:
        total_frames = 0
        for _, start_frame, end_frame in worker_assignments:
            total_frames += end_frame - start_frame
        frames_per_worker.append(total_frames)

    # Each worker should get exactly 300 frames
    assert frames_per_worker[0] == 300
    assert frames_per_worker[1] == 300
    assert sum(frames_per_worker) == 600


def test_assign_workers_handles_small_videos(tmp_path: Path):
    """Test that small videos work correctly."""
    v1 = tmp_path / "short.mp4"
    write_frames_to_video(v1, make_frames_with_stride(5), fps=5.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)
    ds.assign_workers(n_loading_workers=2)

    # With only 5 frames and min_frames_per_worker=300,
    # should reduce to 1 worker to meet minimum
    assert len(ds.worker_assignments) == 1

    # Collect all assignments
    all_assignments = []
    for worker_assignments in ds.worker_assignments:
        all_assignments.extend(worker_assignments)

    # Verify all 5 frames are covered
    total_frames = 0
    for _, start_frame, end_frame in all_assignments:
        total_frames += end_frame - start_frame

    assert total_frames == 5


def test_min_frames_per_worker_parameter():
    """Test that min_frames_per_worker parameter works correctly."""
    # Test with low min_frames_per_worker to allow more workers
    ds = VideoCollectionDataset([], as_image_dirs=False, min_frames_per_worker=10)
    assert ds.min_frames_per_worker == 10

    # Default value
    ds2 = VideoCollectionDataset([], as_image_dirs=False)
    assert ds2.min_frames_per_worker == 300


def test_buffer_size_parameter():
    """Test that buffer_size parameter is accepted and stored."""
    ds = VideoCollectionDataset([], as_image_dirs=False, buffer_size=128)
    assert ds.buffer_size == 128

    # Default value
    ds2 = VideoCollectionDataset([], as_image_dirs=False)
    assert ds2.buffer_size == 64


def test_dataloader_worker_assignment(tmp_path: Path):
    """Test that dataloader properly assigns workers."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(30), fps=5.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)

    # Create dataloader - should automatically call assign_workers
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0)

    # With num_workers=0, should have 1 worker
    assert len(ds.worker_assignments) == 1

    # All 30 frames should be assigned to the single worker
    total_frames = 0
    for _, start_frame, end_frame in ds.worker_assignments[0]:
        total_frames += end_frame - start_frame
    assert total_frames == 30


def test_image_dirs_with_workers(tmp_path: Path):
    """Test that image directories work correctly with worker assignment."""
    d = tmp_path / "frames"
    d.mkdir()

    # Create 15 dummy image files
    for i in range(15):
        (d / f"frame_{i:03d}.png").write_bytes(b"x")

    ds = VideoCollectionDataset([d], as_image_dirs=True, frame_sorting=r"(\d+)")
    ds.assign_workers(n_loading_workers=2)

    # Verify frame assignments
    all_assignments = []
    for worker_assignments in ds.worker_assignments:
        all_assignments.extend(worker_assignments)

    total_frames_assigned = 0
    for video_idx, start_frame, end_frame in all_assignments:
        total_frames_assigned += end_frame - start_frame
        # Only one directory (video_idx should be 0)
        assert video_idx == 0
        # Frame ranges should be valid
        assert 0 <= start_frame < end_frame <= 15

    assert total_frames_assigned == 15


def test_multiple_videos_worker_distribution(tmp_path: Path):
    """Test that frames from multiple videos are distributed across workers."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    v3 = tmp_path / "v3.mp4"

    write_frames_to_video(v1, make_frames_with_stride(12), fps=5.0)
    write_frames_to_video(v2, make_frames_with_stride(12), fps=5.0)
    write_frames_to_video(v3, make_frames_with_stride(12), fps=5.0)

    ds = VideoCollectionDataset([v1, v2, v3], as_image_dirs=False)
    ds.assign_workers(n_loading_workers=3)

    # 3 videos * 12 frames each = 36 frames total
    assert ds.n_frames_total == 36

    # Collect all assignments
    all_assignments = []
    for worker_assignments in ds.worker_assignments:
        all_assignments.extend(worker_assignments)

    # Verify all frames are covered
    total_frames_assigned = 0
    video_indices_seen = set()
    for video_idx, start_frame, end_frame in all_assignments:
        total_frames_assigned += end_frame - start_frame
        video_indices_seen.add(video_idx)
        # Verify valid video indices (0, 1, 2)
        assert video_idx in [0, 1, 2]

    assert total_frames_assigned == 36
    # All three videos should appear in assignments
    assert video_indices_seen == {0, 1, 2}


def test_simple_video_collection_loader_basic_functionality(tmp_path: Path):
    """Test that SimpleVideoCollectionLoader works as expected."""
    v1 = tmp_path / "video.mp4"
    v2 = tmp_path / "video2.mp4"

    write_frames_to_video(v1, make_frames_with_stride(15), fps=5.0)
    write_frames_to_video(v2, make_frames_with_stride(10), fps=5.0)

    # Test basic instantiation and usage
    loader = SimpleVideoCollectionLoader(
        [v1, v2],
        batch_size=5,
        num_workers=0,  # Use 0 for testing to avoid complexity
        min_frames_per_worker=5,
    )

    # Should inherit from VideoCollectionDataLoader
    assert isinstance(loader, VideoCollectionDataLoader)

    # Should have created a dataset internally
    assert isinstance(loader.dataset, VideoCollectionDataset)

    # Should be able to iterate
    all_frames = []
    all_video_indices = []
    all_frame_indices = []

    for batch in loader:
        assert "frames" in batch
        assert "video_indices" in batch
        assert "frame_indices" in batch

        all_frames.append(batch["frames"])
        all_video_indices.extend(batch["video_indices"])
        all_frame_indices.extend(batch["frame_indices"])

    # Should have gotten all 25 frames (15 + 10)
    assert len(all_video_indices) == 25
    assert len(all_frame_indices) == 25

    # Should have frames from both videos
    assert 0 in all_video_indices
    assert 1 in all_video_indices


def test_simple_video_collection_loader_with_parameters(tmp_path: Path):
    """Test SimpleVideoCollectionLoader with various parameters."""
    # Test with image directories
    d1 = tmp_path / "frames1"
    d1.mkdir()

    # Create some dummy images
    for i in range(8):
        (d1 / f"frame_{i:03d}.png").write_bytes(b"dummy")

    loader = SimpleVideoCollectionLoader(
        [d1],
        as_image_dirs=True,
        frame_sorting=r"(\d+)",
        batch_size=3,
        num_workers=0,
        buffer_size=16,
        min_frames_per_worker=2,
    )

    # Should work with image directories
    assert loader.dataset.as_image_dirs is True
    assert loader.dataset.frame_sorting == r"(\d+)"
    assert loader.dataset.buffer_size == 16
    assert loader.dataset.min_frames_per_worker == 2

    # Should have found 8 frames
    assert loader.dataset.n_frames_total == 8


def test_simple_video_collection_loader_worker_resolution():
    """Test that SimpleVideoCollectionLoader resolves worker counts correctly."""
    from multiprocessing import cpu_count

    # Test that -1 gets resolved to appropriate worker count
    loader = SimpleVideoCollectionLoader(
        [], num_workers=-1, min_frames_per_worker=1  # Empty paths for testing
    )

    # Should have resolved -1 to actual worker count
    assert loader.num_workers > 0
    assert loader.num_workers <= cpu_count()

    # Test that 0 gets resolved to 1
    loader2 = SimpleVideoCollectionLoader([], num_workers=0, min_frames_per_worker=1)

    assert loader2.num_workers == 1

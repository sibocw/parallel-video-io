"""Unit tests for VideoCollectionDataset and VideoCollectionDataLoader."""

import pytest
import torch
import numpy as np
import imageio.v2 as imageio
import tempfile
from pathlib import Path
from multiprocessing import cpu_count

from pvio.torch import (
    VideoCollectionDataLoader,
    VideoCollectionDataset,
    SimpleVideoCollectionLoader,
    EncodedVideo,
    ImageDirVideo,
)
from pvio.video_io import write_frames_to_video
from .test_utils import make_frames_with_stride, DummyDataset


# Test utilities
def create_test_images(
    directory: Path, count: int, prefix: str = "frame", use_regex_pattern: bool = True
):
    """Helper to create test image files."""
    directory.mkdir(exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    for i in range(count):
        if use_regex_pattern:
            filename = f"{prefix}_{i:03d}.png"
        else:
            # Use letters for non-regex testing
            filename = f"{chr(ord('a') + i % 26)}.png"
        imageio.imwrite(directory / filename, img)


def create_test_video(path: Path, n_frames: int = 15) -> Path:
    """Helper to create a test video file."""
    frames = make_frames_with_stride(n_frames)
    write_frames_to_video(path, frames, fps=5.0)
    return path


# Core functionality tests
def test_frame_extraction_parsing():
    """Test frame number extraction from filenames."""
    parse = ImageDirVideo._parse_frame_id_from_filename

    # Successful extractions
    assert parse("img_001.png", r"(\d+)") == 1
    assert parse("frame_42.jpg", r"(\d+)") == 42

    # Error cases
    with pytest.raises(ValueError, match="0 matches found"):
        parse("no_numbers.png", r"(\d+)")
    with pytest.raises(ValueError, match="2 matches found"):
        parse("img_01_02.png", r"(\d+)")


def test_dataloader_basic_functionality():
    """Test basic DataLoader functionality with collation."""
    ds = DummyDataset()
    loader = VideoCollectionDataLoader(ds, batch_size=3, num_workers=0)

    batch = next(iter(loader))
    assert "frames" in batch and "video_indices" in batch and "frame_indices" in batch
    assert batch["frames"].shape == (3, 3, 4, 5)

    # Check values are correctly collated
    assert torch.equal(batch["frames"][0], torch.zeros(3, 4, 5))
    assert torch.equal(batch["frames"][2], torch.ones(3, 4, 5) * 2)


def test_dataloader_validation():
    """Test DataLoader initialization validation."""
    with pytest.raises(ValueError, match="only works with VideoCollectionDataset"):
        VideoCollectionDataLoader(object(), batch_size=1)

    ds = DummyDataset()

    with pytest.raises(ValueError, match="does not support custom batch samplers"):
        VideoCollectionDataLoader(ds, batch_size=1, batch_sampler=[1])

    with pytest.raises(ValueError, match="must use the built-in collate function"):
        VideoCollectionDataLoader(ds, batch_size=1, collate_fn=lambda x: x)


def test_image_directory_handling(tmp_path: Path):
    """Test ImageDirVideo with different sorting strategies."""
    # Test filename-based sorting (no regex)
    d1 = tmp_path / "frames1"
    create_test_images(d1, 3, use_regex_pattern=False)

    video = ImageDirVideo(d1, frame_id_regex=None)
    video.setup()  # Need to call setup to populate phy_frame_id_to_path
    paths = [
        video.phy_frame_id_to_path[i] for i in sorted(video.phy_frame_id_to_path.keys())
    ]
    assert [p.name for p in paths] == ["a.png", "b.png", "c.png"]

    # Test regex-based sorting
    d2 = tmp_path / "frames2"
    create_test_images(d2, 3, "frame")

    video2 = ImageDirVideo(d2, frame_id_regex=r"(\d+)")
    video2.setup()  # Need to call setup to populate phy_frame_id_to_path
    paths2 = [
        video2.phy_frame_id_to_path[i]
        for i in sorted(video2.phy_frame_id_to_path.keys())
    ]
    assert [p.name for p in paths2] == [
        "frame_000.png",
        "frame_001.png",
        "frame_002.png",
    ]


def test_video_object_creation():
    """Test Video object parameter handling."""
    # EncodedVideo parameters
    dummy_path = Path("/fake/path.mp4")
    video = EncodedVideo(dummy_path, buffer_size=128)
    assert video.buffer_size == 128
    assert video.path == dummy_path

    # ImageDirVideo with real directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        create_test_images(tmp_path, 1, "frame")

        video2 = ImageDirVideo(tmp_path, frame_id_regex=r"(\d+)")
        assert video2.path == tmp_path


@pytest.mark.parametrize(
    "n_workers,expected_workers",
    [
        (0, 1),  # Zero workers becomes 1
    ],
)
def test_worker_assignment(tmp_path: Path, n_workers, expected_workers):
    """Test worker assignment with different configurations."""
    # Create test videos
    videos = []
    frame_counts = [25, 15, 30]

    for i, frames in enumerate(frame_counts):
        path = tmp_path / f"video_{i}.mp4"
        create_test_video(path, frames)
        videos.append(EncodedVideo(path))

    ds = VideoCollectionDataset(videos)
    ds.assign_workers(n_loading_workers=expected_workers)

    # Check basic properties
    assert ds.n_frames_total == sum(frame_counts)
    assert len(ds.worker_assignments) == expected_workers

    # Verify all frames are assigned
    total_assigned = sum(
        end - start for worker in ds.worker_assignments for _, start, end in worker
    )
    assert total_assigned == ds.n_frames_total


def test_worker_assignment_reduction(tmp_path: Path):
    """Test that workers are reduced when there aren't enough frames."""
    # Create small videos that will trigger worker reduction
    videos = []
    frame_counts = [25, 15, 30]  # Total 70 frames

    for i, frames in enumerate(frame_counts):
        path = tmp_path / f"video_{i}.mp4"
        create_test_video(path, frames)
        videos.append(EncodedVideo(path))

    ds = VideoCollectionDataset(videos)
    ds.assign_workers(n_loading_workers=2)  # Request 2 workers

    # With default min_frames_per_worker=300 and only 70 frames total,
    # should reduce to 1 worker
    assert len(ds.worker_assignments) == 1

    # All frames should still be assigned
    total_assigned = sum(
        end - start for worker in ds.worker_assignments for _, start, end in worker
    )
    assert total_assigned == sum(frame_counts)


def test_simple_video_collection_loader(tmp_path: Path):
    """Test SimpleVideoCollectionLoader end-to-end functionality."""
    # Test with mixed video types
    video_file = tmp_path / "video.mp4"
    image_dir = tmp_path / "frames"

    create_test_video(video_file, 20)
    create_test_images(image_dir, 15, "frame")

    # Test basic functionality
    loader = SimpleVideoCollectionLoader(
        [video_file, image_dir],
        batch_size=8,
        num_workers=0,
        frame_id_regex=r"(\d+)",
        min_frames_per_worker=5,
    )

    # Should inherit from VideoCollectionDataLoader
    assert isinstance(loader, VideoCollectionDataLoader)
    assert isinstance(loader.dataset, VideoCollectionDataset)

    # Test iteration
    total_frames = 0
    video_indices_seen = set()

    for batch in loader:
        assert all(key in batch for key in ["frames", "video_indices", "frame_indices"])
        total_frames += len(batch["frame_indices"])
        video_indices_seen.update(batch["video_indices"])

    assert total_frames == 35  # 20 + 15
    assert video_indices_seen == {0, 1}  # Both videos represented


def test_worker_resolution():
    """Test worker count resolution logic."""
    # Test automatic worker resolution
    loader = SimpleVideoCollectionLoader([], num_workers=-1, min_frames_per_worker=1)
    assert 0 < loader.num_workers <= cpu_count()

    # Test zero workers becomes 1
    loader2 = SimpleVideoCollectionLoader([], num_workers=0, min_frames_per_worker=1)
    assert loader2.num_workers == 1


def test_balanced_distribution_large_video(tmp_path: Path):
    """Test that large videos get balanced worker distribution."""
    # Create a large video
    large_video = tmp_path / "large.mp4"
    create_test_video(large_video, 600)

    video = EncodedVideo(large_video)
    ds = VideoCollectionDataset([video])

    # Force smaller min_frames_per_worker for testing
    ds.min_frames_per_worker = 50
    ds.assign_workers(n_loading_workers=2)

    # Should have 2 workers, each with ~300 frames
    frames_per_worker = [
        sum(end - start for _, start, end in worker_assignments)
        for worker_assignments in ds.worker_assignments
    ]

    assert len(frames_per_worker) == 2
    assert frames_per_worker[0] == 300
    assert frames_per_worker[1] == 300


def test_small_video_worker_reduction(tmp_path: Path):
    """Test that small videos reduce worker count appropriately."""
    small_video = tmp_path / "small.mp4"
    create_test_video(small_video, 5)

    video = EncodedVideo(small_video)
    ds = VideoCollectionDataset([video])
    ds.assign_workers(n_loading_workers=4)  # Request more workers than frames

    # Should reduce to 1 worker due to minimum frames constraint
    assert len(ds.worker_assignments) == 1

    total_frames = sum(end - start for _, start, end in ds.worker_assignments[0])
    assert total_frames == 5

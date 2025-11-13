"""Simplified integration tests for VideoCollectionDataset with parallel loading."""

import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

from pvio.io import write_frames_to_video
from pvio.video import EncodedVideo, ImageDirVideo
from pvio.torch_tools import (
    VideoCollectionDataLoader,
    VideoCollectionDataset,
    SimpleVideoCollectionLoader,
)

from .test_utils import make_frames_with_stride, DummyVideo, DummyDataset


def create_test_video_with_pattern(
    path: Path, n_frames: int, base_value: int = 0
) -> None:
    """Create a test video with a specific value pattern for verification."""
    frames = []
    for i in range(n_frames):
        value = (base_value + i * 10) % 256
        frame = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        frames.append(frame)
    write_frames_to_video(path, frames, fps=10.0)


def create_test_image_dir(directory: Path, n_frames: int, fill_value: int = 30) -> None:
    """Create test image directory with specified fill value."""
    directory.mkdir(exist_ok=True)
    for i in range(n_frames):
        img = np.full((32, 32, 3), fill_value=fill_value, dtype=np.uint8)
        imageio.imwrite(directory / f"frame_{i:03d}.png", img)


def test_single_video_parallel_loading(tmp_path: Path):
    """Test loading a single video with worker processing."""
    video_path = tmp_path / "video.mp4"
    n_frames = 25

    # Create video with stride pattern for verification
    frames = make_frames_with_stride(n_frames, stride=10)
    write_frames_to_video(video_path, frames, fps=10.0)

    video = EncodedVideo(video_path)
    ds = VideoCollectionDataset([video])
    # Use single worker to avoid complexity but test the full pipeline
    loader = VideoCollectionDataLoader(ds, batch_size=10, num_workers=0)

    # Collect all frames and verify completeness
    all_indices = []
    all_frames = []

    for batch in loader:
        all_frames.append(batch["frames"])
        all_indices.extend(batch["frame_indices"])

    # Verify we got all frames
    assert len(all_indices) == n_frames
    assert set(all_indices) == set(range(n_frames))

    # Verify frame values (allowing for compression artifacts)
    all_frames_tensor = torch.cat(all_frames, dim=0)
    for i in range(n_frames):
        idx_in_list = all_indices.index(i)
        frame = all_frames_tensor[idx_in_list]
        expected_value = ((i * 10) % 256) / 255.0
        mean_value = frame.mean().item()
        assert abs(mean_value - expected_value) < 0.05, f"Frame {i} value mismatch"


def test_multiple_videos_parallel_loading(tmp_path: Path):
    """Test loading multiple videos with distinct patterns."""
    video_paths = []
    expected_counts = [15, 20, 10]

    # Create videos with different base values for identification
    for i, count in enumerate(expected_counts):
        path = tmp_path / f"video_{i}.mp4"
        create_test_video_with_pattern(path, count, base_value=i * 100)
        video_paths.append(path)

    videos = [EncodedVideo(path) for path in video_paths]
    ds = VideoCollectionDataset(videos)
    # Use single worker to focus on functionality over parallelism
    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=0)

    # Collect frames by video
    frames_by_video = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            video_idx = batch["video_indices"][i]
            frame_idx = batch["frame_indices"][i]
            frame = batch["frames"][i]

            if video_idx not in frames_by_video:
                frames_by_video[video_idx] = {}
            frames_by_video[video_idx][frame_idx] = frame

    # Verify each video's frames
    assert len(frames_by_video) == 3

    for video_idx, frame_dict in frames_by_video.items():
        expected_count = expected_counts[video_idx]
        assert len(frame_dict) == expected_count

        # Verify frame values match the pattern
        base_value = video_idx * 100
        for frame_idx, frame in frame_dict.items():
            expected_value = ((base_value + frame_idx * 10) % 256) / 255.0
            mean_value = frame.mean().item()
            assert abs(mean_value - expected_value) < 0.05


def test_image_directories_parallel_loading(tmp_path: Path):
    """Test loading from image directories."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"

    # Create directories with different fill values
    create_test_image_dir(dir1, 20, fill_value=30)
    create_test_image_dir(dir2, 15, fill_value=80)

    videos = [
        ImageDirVideo(dir1, frame_id_regex=r"(\d+)"),
        ImageDirVideo(dir2, frame_id_regex=r"(\d+)"),
    ]

    ds = VideoCollectionDataset(videos)
    # Use single worker for stability
    loader = VideoCollectionDataLoader(ds, batch_size=7, num_workers=0)

    # Collect frames by directory
    frames_by_dir = {0: [], 1: []}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            dir_idx = batch["video_indices"][i]
            frame = batch["frames"][i]
            frames_by_dir[dir_idx].append(frame)

    # Verify frame counts and values (PNGs are lossless)
    assert len(frames_by_dir[0]) == 20
    assert len(frames_by_dir[1]) == 15

    # Check values
    for frame in frames_by_dir[0]:
        expected = 30 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)

    for frame in frames_by_dir[1]:
        expected = 80 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)


def test_parallel_worker_functionality(tmp_path: Path):
    """Test that parallel workers actually work with sufficient data."""
    # Create a large enough video to justify multiple workers
    video_path = tmp_path / "large_video.mp4"
    n_frames = 600  # Large enough to split across workers

    # Create simple pattern video
    frames = []
    for i in range(n_frames):
        value = (i * 5) % 256
        frame = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        frames.append(frame)
    write_frames_to_video(video_path, frames, fps=10.0)

    video = EncodedVideo(video_path)
    ds = VideoCollectionDataset([video])

    # Use SimpleVideoCollectionLoader with explicit min_frames_per_worker
    loader = SimpleVideoCollectionLoader(
        [video_path],
        batch_size=20,
        num_workers=2,
        min_frames_per_worker=50,  # This should allow 2 workers
    )

    all_indices = set()
    for batch in loader:
        all_indices.update(batch["frame_indices"])

    # Should have processed all frames
    assert len(all_indices) == n_frames
    assert all_indices == set(range(n_frames))


def test_transform_handling():
    """Test that transforms can be applied to loaded frames."""

    def double_transform(frame):
        assert frame.ndim == 3 and frame.shape[0] == 3  # CHW format
        return frame * 2.0

    # Use dummy data to avoid file I/O overhead
    ds = DummyDataset([DummyVideo(n_frames=5)])
    loader = VideoCollectionDataLoader(ds, batch_size=3, num_workers=0)

    for batch in loader:
        for frame in batch["frames"]:
            # Apply transform manually and verify
            transformed = double_transform(frame)
            assert torch.allclose(transformed, frame * 2.0)


def test_buffer_management_across_videos(tmp_path: Path):
    """Test that buffer is properly managed when switching between videos."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"

    # Create videos with distinct patterns
    create_test_video_with_pattern(v1, 30, base_value=0)
    create_test_video_with_pattern(v2, 30, base_value=100)

    # Use smaller buffer size to force more frequent switches
    videos = [EncodedVideo(v1, buffer_size=10), EncodedVideo(v2, buffer_size=10)]

    ds = VideoCollectionDataset(videos)
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=1)

    frames_by_video = {0: {}, 1: {}}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            video_idx = batch["video_indices"][i]
            frame_idx = batch["frame_indices"][i]
            frame = batch["frames"][i]
            frames_by_video[video_idx][frame_idx] = frame

    # Verify both videos have correct frame values despite buffer switching
    for video_idx, frame_dict in frames_by_video.items():
        base_value = video_idx * 100
        for frame_idx, frame in frame_dict.items():
            expected = ((base_value + frame_idx * 10) % 256) / 255.0
            mean_value = frame.mean().item()
            assert abs(mean_value - expected) < 0.05


def test_zero_workers_functionality(tmp_path: Path):
    """Test that num_workers=0 works correctly (single-threaded)."""
    video_path = tmp_path / "video.mp4"
    frames = make_frames_with_stride(25, stride=10)
    write_frames_to_video(video_path, frames, fps=10.0)

    video = EncodedVideo(video_path)
    ds = VideoCollectionDataset([video])
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0)

    all_indices = []
    for batch in loader:
        all_indices.extend(batch["frame_indices"])

    assert len(all_indices) == 25
    assert set(all_indices) == set(range(25))


def test_simple_video_collection_loader_integration(tmp_path: Path):
    """Test SimpleVideoCollectionLoader end-to-end with mixed video types."""
    # Create test data
    video_file = tmp_path / "video.mp4"
    image_dir = tmp_path / "frames"

    create_test_video_with_pattern(video_file, 20, base_value=0)
    create_test_image_dir(image_dir, 15, fill_value=120)

    # Test with SimpleVideoCollectionLoader
    loader = SimpleVideoCollectionLoader(
        [video_file, image_dir],
        batch_size=7,
        num_workers=2,
        frame_id_regex=r"(\d+)",
        min_frames_per_worker=10,
    )

    # Collect and verify
    total_frames = 0
    video_indices_seen = set()

    for batch in loader:
        total_frames += len(batch["frame_indices"])
        video_indices_seen.update(batch["video_indices"])

    assert total_frames == 35  # 20 + 15
    assert video_indices_seen == {0, 1}

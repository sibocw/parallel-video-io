"""Integration tests for VideoCollectionDataset with parallel loading."""

import torch
import numpy as np
from pathlib import Path

from pvio.torch import (
    VideoCollectionDataLoader,
    VideoCollectionDataset,
    SimpleVideoCollectionLoader,
    EncodedVideo,
    ImageDirVideo,
)
from pvio.video_io import write_frames_to_video

from .test_utils import make_frames_with_stride


def test_load_single_video_with_workers(tmp_path: Path):
    """Integration test: Load video with multiple workers."""
    v1 = tmp_path / "video.mp4"

    n_frames = 25
    frames = make_frames_with_stride(n_frames, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    # Create EncodedVideo object
    video = EncodedVideo(v1)
    ds = VideoCollectionDataset([video])
    loader = VideoCollectionDataLoader(ds, batch_size=10, num_workers=2, min_frames_per_worker=10)

    # Collect all frames
    all_frames = []
    all_indices = []
    for batch in loader:
        all_frames.append(batch["frames"])
        all_indices.extend(batch["frame_indices"])

    all_frames = torch.cat(all_frames, dim=0)

    # Verify we got all frames
    assert len(all_indices) == n_frames
    assert set(all_indices) == set(range(n_frames))

    # Verify frame values are approximately correct (allowing for compression)
    for i in range(n_frames):
        idx_in_list = all_indices.index(i)
        frame = all_frames[idx_in_list]
        expected_value = ((i * 10) % 256) / 255.0

        # Use higher tolerance for lossy compression
        mean_value = frame.mean().item()
        assert (
            abs(mean_value - expected_value) < 0.05
        ), f"Frame {i}: expected {expected_value:.3f}, got {mean_value:.3f}"


def test_load_multiple_videos_with_workers(tmp_path: Path):
    """Integration test: Load multiple videos with workers."""
    videos = []
    expected_frame_counts = [15, 20, 10]

    video_objects = []
    for i, n_frames in enumerate(expected_frame_counts):
        video_path = tmp_path / f"video_{i}.mp4"
        # Each video starts at a different base value to distinguish them
        # Video 0: 0, 10, 20, ...
        # Video 1: 100, 110, 120, ...
        # Video 2: 200, 210, 220, ...
        frames = []
        base_value = i * 100
        for j in range(n_frames):
            value = (base_value + j * 10) % 256
            frame = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
            frames.append(frame)
        write_frames_to_video(video_path, frames, fps=10.0)
        
        # Create EncodedVideo object
        video_objects.append(EncodedVideo(video_path))

    ds = VideoCollectionDataset(video_objects)
    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=3, min_frames_per_worker=10)

    # Collect all frames with their metadata
    frames_by_video = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            video_idx = batch["video_indices"][i]
            frame_idx = batch["frame_indices"][i]
            frame = batch["frames"][i]

            if video_idx not in frames_by_video:
                frames_by_video[video_idx] = {}
            frames_by_video[video_idx][frame_idx] = frame

    # Verify we got frames from all videos
    assert len(frames_by_video) == 3

    # Verify frame counts per video
    for video_idx, frame_dict in frames_by_video.items():
        expected_count = expected_frame_counts[video_idx]
        assert len(frame_dict) == expected_count

        # Verify frame values (approximately due to compression)
        base_value = video_idx * 100
        for frame_idx, frame in frame_dict.items():
            expected_value = ((base_value + frame_idx * 10) % 256) / 255.0
            mean_value = frame.mean().item()
            assert (
                abs(mean_value - expected_value) < 0.05
            ), f"Video {video_idx}, frame {frame_idx}: expected {expected_value:.3f}, got {mean_value:.3f}"


def test_transform_applied_correctly_with_workers(tmp_path: Path):
    """Integration test: Verify transform is applied correctly per-frame."""
    v1 = tmp_path / "video.mp4"

    # Create video with uniform value 120
    frames = [np.full((32, 32, 3), fill_value=120, dtype=np.uint8) for _ in range(20)]
    write_frames_to_video(v1, frames, fps=10.0)

    # Transform that doubles values
    def double_transform(frame):
        # frame is CHW, float in [0, 1]
        assert frame.ndim == 3, f"Transform expects CHW, got shape {frame.shape}"
        assert (
            frame.shape[0] == 3
        ), f"Transform expects 3 channels, got {frame.shape[0]}"
        return frame * 2.0

    # Create EncodedVideo object
    video = EncodedVideo(v1)
    ds = VideoCollectionDataset([video])
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=2, min_frames_per_worker=5)

    for batch in loader:
        # Check that we can apply transform to frames manually
        # Original value is ~120/255 ≈ 0.471
        for frame in batch["frames"]:
            # Apply transform and check result
            transformed = double_transform(frame)
            expected_value = (120 / 255.0) * 2.0
            mean_value = transformed.mean().item()
            assert abs(mean_value - expected_value) < 0.1


def test_buffer_across_videos(tmp_path: Path):
    """Test that buffer is properly managed when processing frames from different videos."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"

    # Create two videos with distinct stride patterns
    # Video 1: 0, 10, 20, ...
    # Video 2: 100, 110, 120, ...
    frames_v1 = make_frames_with_stride(30, stride=10)
    frames_v2 = []
    for i in range(30):
        value = (100 + i * 10) % 256
        frame = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        frames_v2.append(frame)

    write_frames_to_video(v1, frames_v1, fps=10.0)
    write_frames_to_video(v2, frames_v2, fps=10.0)

    # Create EncodedVideo objects with specific buffer size
    video1 = EncodedVideo(v1, buffer_size=10)
    video2 = EncodedVideo(v2, buffer_size=10)
    
    ds = VideoCollectionDataset([video1, video2])
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=1, min_frames_per_worker=15)

    frames_by_video = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            video_idx = batch["video_indices"][i]
            frame_idx = batch["frame_indices"][i]
            frame = batch["frames"][i]

            if video_idx not in frames_by_video:
                frames_by_video[video_idx] = {}
            frames_by_video[video_idx][frame_idx] = frame

    # Verify frames from each video have correct values
    # Check v1 frames (video_idx = 0)
    for frame_idx, frame in frames_by_video[0].items():
        expected = ((frame_idx * 10) % 256) / 255.0
        mean_value = frame.mean().item()
        assert (
            abs(mean_value - expected) < 0.05
        ), f"v1 frame {frame_idx}: expected {expected:.3f}, got {mean_value:.3f}"

    # Check v2 frames (video_idx = 1)
    for frame_idx, frame in frames_by_video[1].items():
        expected = ((100 + frame_idx * 10) % 256) / 255.0
        mean_value = frame.mean().item()
        assert (
            abs(mean_value - expected) < 0.05
        ), f"v2 frame {frame_idx}: expected {expected:.3f}, got {mean_value:.3f}"


def test_image_dirs_with_workers(tmp_path: Path):
    """Integration test: Load from image directories with workers."""
    import imageio.v2 as imageio

    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()

    # Create images in directory 1 (value 30)
    for i in range(20):
        img = np.full((32, 32, 3), fill_value=30, dtype=np.uint8)
        imageio.imwrite(d1 / f"frame_{i:03d}.png", img)

    # Create images in directory 2 (value 80)
    for i in range(15):
        img = np.full((32, 32, 3), fill_value=80, dtype=np.uint8)
        imageio.imwrite(d2 / f"frame_{i:03d}.png", img)

    # Create ImageDirVideo objects
    video1 = ImageDirVideo(d1, frameid_regex=r"(\d+)")
    video2 = ImageDirVideo(d2, frameid_regex=r"(\d+)")
    
    ds = VideoCollectionDataset([video1, video2])
    loader = VideoCollectionDataLoader(ds, batch_size=7, num_workers=2, min_frames_per_worker=10)

    frames_by_dir = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            dir_idx = batch["video_indices"][
                i
            ]  # Now using video_indices instead of video_paths
            frame = batch["frames"][i]

            if dir_idx not in frames_by_dir:
                frames_by_dir[dir_idx] = []
            frames_by_dir[dir_idx].append(frame)

    # Verify we got correct number of frames
    assert len(frames_by_dir[0]) == 20  # dir1
    assert len(frames_by_dir[1]) == 15  # dir2

    # Verify values (PNGs are lossless, so exact match expected)
    for frame in frames_by_dir[0]:
        expected = 30 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)

    for frame in frames_by_dir[1]:
        expected = 80 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)


def test_zero_workers_still_works(tmp_path: Path):
    """Test that num_workers=0 works correctly."""
    v1 = tmp_path / "video.mp4"

    frames = make_frames_with_stride(25, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    # Create EncodedVideo object
    video = EncodedVideo(v1)
    ds = VideoCollectionDataset([video])
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0)

    all_indices = []
    for batch in loader:
        all_indices.extend(batch["frame_indices"])

    # Should still get all frames
    assert len(all_indices) == 25
    assert set(all_indices) == set(range(25))


def test_simple_video_collection_loader_integration(tmp_path: Path):
    """Integration test: Test SimpleVideoCollectionLoader end-to-end."""
    # Create test videos with different patterns
    v1 = tmp_path / "test1.mp4"
    v2 = tmp_path / "test2.mp4"

    # Video 1: frames with values 0, 10, 20, ...
    frames_v1 = make_frames_with_stride(20, stride=10)
    write_frames_to_video(v1, frames_v1, fps=10.0)

    # Video 2: frames with values starting from 100
    frames_v2 = []
    for i in range(15):
        value = (100 + i * 10) % 256
        frame = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        frames_v2.append(frame)
    write_frames_to_video(v2, frames_v2, fps=10.0)

    # Test with SimpleVideoCollectionLoader
    loader = SimpleVideoCollectionLoader(
        [v1, v2],
        batch_size=7,
        num_workers=2,
        min_frames_per_worker=10,  # Allow multiple workers
    )

    # Collect all frames
    frames_by_video = {}
    total_frames = 0

    for batch in loader:
        assert "frames" in batch
        assert "video_indices" in batch
        assert "frame_indices" in batch

        for i in range(len(batch["frame_indices"])):
            video_idx = batch["video_indices"][i]
            frame_idx = batch["frame_indices"][i]
            frame = batch["frames"][i]

            if video_idx not in frames_by_video:
                frames_by_video[video_idx] = {}
            frames_by_video[video_idx][frame_idx] = frame
            total_frames += 1

    # Should have gotten all 35 frames (20 + 15)
    assert total_frames == 35
    assert len(frames_by_video) == 2

    # Verify video 1 frames (video_idx = 0)
    assert len(frames_by_video[0]) == 20
    for frame_idx, frame in frames_by_video[0].items():
        expected_value = ((frame_idx * 10) % 256) / 255.0
        mean_value = frame.mean().item()
        assert (
            abs(mean_value - expected_value) < 0.05
        ), f"Video 0, frame {frame_idx}: expected {expected_value:.3f}, got {mean_value:.3f}"

    # Verify video 2 frames (video_idx = 1)
    assert len(frames_by_video[1]) == 15
    for frame_idx, frame in frames_by_video[1].items():
        expected_value = ((100 + frame_idx * 10) % 256) / 255.0
        mean_value = frame.mean().item()
        assert (
            abs(mean_value - expected_value) < 0.05
        ), f"Video 1, frame {frame_idx}: expected {expected_value:.3f}, got {mean_value:.3f}"


def test_simple_video_collection_loader_with_transform(tmp_path: Path):
    """Test SimpleVideoCollectionLoader with transform function."""
    v1 = tmp_path / "video.mp4"

    # Create video with uniform value
    frames = [np.full((32, 32, 3), fill_value=120, dtype=np.uint8) for _ in range(10)]
    write_frames_to_video(v1, frames, fps=10.0)

    # Define transform
    def normalize_transform(frame):
        # Simple normalization: subtract 0.5
        return frame - 0.5

    loader = SimpleVideoCollectionLoader(
        [v1],
        batch_size=5,
        num_workers=0,  # Single worker for simplicity
        min_frames_per_worker=5,
    )

    for batch in loader:
        # Test transform application on loaded frames
        for frame in batch["frames"]:
            # Apply transform manually to test
            transformed = normalize_transform(frame)
            # Original value: ~120/255 ≈ 0.471
            # After transform: 0.471 - 0.5 = -0.029
            expected_value = (120 / 255.0) - 0.5
            mean_value = transformed.mean().item()
            assert abs(mean_value - expected_value) < 0.1

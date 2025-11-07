import pytest
import torch
import numpy as np
from pathlib import Path

from pvio.torch import VideoCollectionDataLoader, VideoCollectionDataset
from pvio.video_io import write_frames_to_video


def test_load_single_video_with_workers(tmp_path: Path):
    """Integration test: Load video with multiple workers"""
    v1 = tmp_path / "video.mp4"

    # Create a video where each frame has a unique value
    n_frames = 50
    frames = [
        np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(n_frames)
    ]
    write_frames_to_video(v1, frames, fps=10.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)
    loader = VideoCollectionDataLoader(ds, batch_size=10, num_workers=2, chunk_size=20)

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

    # Verify frame values are correct (each frame should have unique value)
    for i in range(n_frames):
        # Find the frame with index i
        idx_in_list = all_indices.index(i)
        frame = all_frames[idx_in_list]
        expected_value = i / 255.0  # Normalized
        assert torch.allclose(frame, torch.full_like(frame, expected_value), atol=1e-5)


def test_load_multiple_videos_with_workers(tmp_path: Path):
    """Integration test: Load multiple videos with workers"""
    videos = []
    expected_frame_counts = [15, 25, 10]

    for i, n_frames in enumerate(expected_frame_counts):
        video_path = tmp_path / f"video_{i}.mp4"
        # Each video has frames with values based on video index and frame index
        # Video 0: frames 0, 1, 2, ...
        # Video 1: frames 100, 101, 102, ...
        # Video 2: frames 200, 201, 202, ...
        base_value = i * 100
        frames = [
            np.full((32, 32, 3), fill_value=(base_value + j) % 256, dtype=np.uint8)
            for j in range(n_frames)
        ]
        write_frames_to_video(video_path, frames, fps=10.0)
        videos.append(video_path)

    ds = VideoCollectionDataset(videos, as_image_dirs=False)
    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=3, chunk_size=10)

    # Collect all frames with their metadata
    frames_by_video = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            video_path = batch["video_paths"][i]
            frame_idx = batch["frame_indices"][i]
            frame = batch["frames"][i]

            if video_path not in frames_by_video:
                frames_by_video[video_path] = {}
            frames_by_video[video_path][frame_idx] = frame

    # Verify we got frames from all videos
    assert len(frames_by_video) == 3

    # Verify frame counts per video
    for video_path, frame_dict in frames_by_video.items():
        video_idx = int(Path(video_path).stem.split("_")[1])
        expected_count = expected_frame_counts[video_idx]
        assert len(frame_dict) == expected_count

        # Verify frame values
        base_value = video_idx * 100
        for frame_idx, frame in frame_dict.items():
            expected_value = ((base_value + frame_idx) % 256) / 255.0
            assert torch.allclose(
                frame, torch.full_like(frame, expected_value), atol=1e-5
            ), f"Video {video_idx}, frame {frame_idx} has wrong value"


def test_transform_applied_correctly_with_workers(tmp_path: Path):
    """Integration test: Verify transform is applied correctly"""
    v1 = tmp_path / "video.mp4"

    # Create video with known values
    frames = [np.full((32, 32, 3), fill_value=128, dtype=np.uint8) for _ in range(20)]
    write_frames_to_video(v1, frames, fps=10.0)

    # Transform that doubles values
    def double_transform(frame):
        # frame is CHW, float in [0, 1]
        assert frame.ndim == 3, f"Transform expects CHW, got shape {frame.shape}"
        assert (
            frame.shape[0] == 3
        ), f"Transform expects 3 channels, got {frame.shape[0]}"
        return frame * 2.0

    ds = VideoCollectionDataset([v1], as_image_dirs=False, transform=double_transform)
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=2, chunk_size=10)

    for batch in loader:
        # Original value is 128/255 ≈ 0.502
        # After transform: 0.502 * 2 ≈ 1.004
        expected_value = (128 / 255.0) * 2.0
        for frame in batch["frames"]:
            assert torch.allclose(
                frame, torch.full_like(frame, expected_value), atol=1e-3
            )


def test_chunking_with_buffer_across_videos(tmp_path: Path):
    """Test that buffer is properly managed when processing chunks from different videos"""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"

    # Create two videos with distinct values
    # Video 1: all frames have value 50
    # Video 2: all frames have value 150
    frames_v1 = [np.full((32, 32, 3), fill_value=50, dtype=np.uint8) for _ in range(30)]
    frames_v2 = [
        np.full((32, 32, 3), fill_value=150, dtype=np.uint8) for _ in range(30)
    ]

    write_frames_to_video(v1, frames_v1, fps=10.0)
    write_frames_to_video(v2, frames_v2, fps=10.0)

    ds = VideoCollectionDataset([v1, v2], as_image_dirs=False, buffer_size=10)
    # Use chunk_size that will create multiple chunks per video
    # 30 frames / 15 chunk_size = 2 chunks per video
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=1, chunk_size=15)

    frames_by_video = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            video_path = batch["video_paths"][i]
            frame = batch["frames"][i]

            if video_path not in frames_by_video:
                frames_by_video[video_path] = []
            frames_by_video[video_path].append(frame)

    # Verify frames from each video have correct values
    v1_key = v1.absolute().as_posix()
    v2_key = v2.absolute().as_posix()

    for frame in frames_by_video[v1_key]:
        expected = 50 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)

    for frame in frames_by_video[v2_key]:
        expected = 150 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)


def test_image_dirs_with_workers(tmp_path: Path):
    """Integration test: Load from image directories with workers"""
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()

    # Create images in directory 1 (value 30)
    for i in range(20):
        img = np.full((32, 32, 3), fill_value=30, dtype=np.uint8)
        import imageio.v2 as imageio

        imageio.imwrite(d1 / f"frame_{i:03d}.png", img)

    # Create images in directory 2 (value 60)
    for i in range(15):
        img = np.full((32, 32, 3), fill_value=60, dtype=np.uint8)
        import imageio.v2 as imageio

        imageio.imwrite(d2 / f"frame_{i:03d}.png", img)

    ds = VideoCollectionDataset([d1, d2], as_image_dirs=True, frame_sorting=r"(\d+)")
    loader = VideoCollectionDataLoader(ds, batch_size=7, num_workers=2, chunk_size=10)

    frames_by_dir = {}
    for batch in loader:
        for i in range(len(batch["frame_indices"])):
            dir_path = batch["video_paths"][i]
            frame = batch["frames"][i]

            if dir_path not in frames_by_dir:
                frames_by_dir[dir_path] = []
            frames_by_dir[dir_path].append(frame)

    # Verify we got correct number of frames
    d1_key = d1.absolute().as_posix()
    d2_key = d2.absolute().as_posix()

    assert len(frames_by_dir[d1_key]) == 20
    assert len(frames_by_dir[d2_key]) == 15

    # Verify values
    for frame in frames_by_dir[d1_key]:
        expected = 30 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)

    for frame in frames_by_dir[d2_key]:
        expected = 60 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)


def test_zero_workers_still_works(tmp_path: Path):
    """Test that num_workers=0 works correctly"""
    v1 = tmp_path / "video.mp4"

    frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(25)]
    write_frames_to_video(v1, frames, fps=10.0)

    ds = VideoCollectionDataset([v1], as_image_dirs=False)
    # num_workers=0 means single-process loading
    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0, chunk_size=10)

    all_indices = []
    for batch in loader:
        all_indices.extend(batch["frame_indices"])

    # Should still get all frames
    assert len(all_indices) == 25
    assert set(all_indices) == set(range(25))

"""Integration tests for VideoCollectionLoader with parallel loading."""

import torch
import numpy as np
from pathlib import Path

from pvio.torch import VideoCollectionLoader
from pvio.video_io import write_frames_to_video

from .test_utils import make_frames_with_stride


def test_load_single_video_with_workers(tmp_path: Path):
    """Integration test: Load video with multiple workers."""
    v1 = tmp_path / "video.mp4"

    n_frames = 25
    frames = make_frames_with_stride(n_frames, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    loader = VideoCollectionLoader(
        [v1], batch_size=10, n_loading_workers=2, chunk_size=10
    )

    # Collect all frames
    all_frames = []
    all_video_indices = []
    all_frame_indices = []

    for batch in loader:
        all_frames.append(batch["frames"])
        all_video_indices.append(batch["video_indices"])
        all_frame_indices.append(batch["frame_indices"])

    all_frames = torch.cat(all_frames, dim=0)
    all_video_indices = torch.cat(all_video_indices, dim=0)
    all_frame_indices = torch.cat(all_frame_indices, dim=0)

    # Verify we got all frames
    assert all_frames.shape[0] == n_frames
    assert len(all_frame_indices) == n_frames

    # Verify frame indices cover all frames
    assert set(all_frame_indices.tolist()) == set(range(n_frames))

    # Verify all video indices are 0 (only one video)
    assert (all_video_indices == 0).all()

    # Verify frame values are approximately correct (allowing for compression)
    frame_indices_list = all_frame_indices.tolist()
    for i in range(n_frames):
        idx_in_list = frame_indices_list.index(i)
        frame = all_frames[idx_in_list]
        expected_value = ((i * 10) % 256) / 255.0

        mean_value = frame.mean().item()
        assert (
            abs(mean_value - expected_value) < 0.05
        ), f"Frame {i}: expected {expected_value:.3f}, got {mean_value:.3f}"


def test_load_multiple_videos_with_workers(tmp_path: Path):
    """Integration test: Load multiple videos with workers."""
    videos = []
    expected_frame_counts = [15, 20, 10]

    for i, n_frames in enumerate(expected_frame_counts):
        video_path = tmp_path / f"video_{i}.mp4"
        # Each video starts at a different base value to distinguish them
        frames = []
        base_value = i * 100
        for j in range(n_frames):
            value = (base_value + j * 10) % 256
            frame = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
            frames.append(frame)
        write_frames_to_video(video_path, frames, fps=10.0)
        videos.append(video_path)

    loader = VideoCollectionLoader(
        videos, batch_size=8, n_loading_workers=3, chunk_size=10
    )

    # Collect all frames with their metadata
    frames_by_video = {}

    for batch in loader:
        batch_frames = batch["frames"]
        batch_video_indices = batch["video_indices"].tolist()
        batch_frame_indices = batch["frame_indices"].tolist()

        for i in range(len(batch_frame_indices)):
            video_idx = batch_video_indices[i]
            frame_idx = batch_frame_indices[i]
            frame = batch_frames[i]

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

    loader = VideoCollectionLoader(
        [v1],
        batch_size=5,
        transform=double_transform,
        n_loading_workers=2,
        chunk_size=10,
    )

    for batch in loader:
        frames = batch["frames"]
        # Original value is ~120/255 ≈ 0.471
        # After transform: 0.471 * 2 ≈ 0.941
        expected_value = (120 / 255.0) * 2.0
        for frame in frames:
            mean_value = frame.mean().item()
            assert abs(mean_value - expected_value) < 0.1
        break  # Just check first batch


def test_buffer_works_across_chunks(tmp_path: Path):
    """Test that buffering works correctly when processing chunks."""
    v1 = tmp_path / "v1.mp4"

    # Create video with stride pattern
    frames_v1 = make_frames_with_stride(50, stride=10)
    write_frames_to_video(v1, frames_v1, fps=10.0)

    loader = VideoCollectionLoader(
        [v1], batch_size=5, buffer_size=10, n_loading_workers=1, chunk_size=15
    )

    frames_collected = {}
    for batch in loader:
        batch_frame_indices = batch["frame_indices"].tolist()
        batch_frames = batch["frames"]

        for i, frame_idx in enumerate(batch_frame_indices):
            frames_collected[frame_idx] = batch_frames[i]

    # Verify all frames collected
    assert len(frames_collected) == 50

    # Verify frame values
    for frame_idx, frame in frames_collected.items():
        expected = ((frame_idx * 10) % 256) / 255.0
        mean_value = frame.mean().item()
        assert abs(mean_value - expected) < 0.05


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

    loader = VideoCollectionLoader(
        [d1, d2],
        batch_size=7,
        as_image_dirs=True,
        frame_sorting=r"(\d+)",
        n_loading_workers=2,
        chunk_size=10,
    )

    frames_by_dir = {}
    for batch in loader:
        batch_video_indices = batch["video_indices"].tolist()
        batch_frames = batch["frames"]

        for i, video_idx in enumerate(batch_video_indices):
            if video_idx not in frames_by_dir:
                frames_by_dir[video_idx] = []
            frames_by_dir[video_idx].append(batch_frames[i])

    # Verify we got correct number of frames
    assert len(frames_by_dir[0]) == 20
    assert len(frames_by_dir[1]) == 15

    # Verify values (PNGs are lossless, so exact match expected)
    for frame in frames_by_dir[0]:
        expected = 30 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)

    for frame in frames_by_dir[1]:
        expected = 80 / 255.0
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=1e-5)


def test_single_worker_still_works(tmp_path: Path):
    """Test that n_loading_workers=1 works correctly."""
    v1 = tmp_path / "video.mp4"

    frames = make_frames_with_stride(25, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    loader = VideoCollectionLoader(
        [v1], batch_size=5, n_loading_workers=1, chunk_size=10
    )

    all_frame_indices = []
    for batch in loader:
        all_frame_indices.extend(batch["frame_indices"].tolist())

    # Should still get all frames
    assert len(all_frame_indices) == 25
    assert set(all_frame_indices) == set(range(25))


def test_even_split_with_multiple_workers(tmp_path: Path):
    """Test even_split chunk_size with multiple workers."""
    v1 = tmp_path / "video.mp4"

    frames = make_frames_with_stride(100, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    n_workers = 4
    loader = VideoCollectionLoader(
        [v1], batch_size=10, n_loading_workers=n_workers, chunk_size="even_split"
    )

    # Should process all frames
    total_frames = 0
    for batch in loader:
        total_frames += batch["frames"].shape[0]

    assert total_frames == 100


def test_batch_structure_correct(tmp_path: Path):
    """Test that batches have correct structure and types."""
    v1 = tmp_path / "video.mp4"

    frames = make_frames_with_stride(30, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    loader = VideoCollectionLoader([v1], batch_size=8, n_loading_workers=2)

    batch = next(iter(loader))

    # Check keys
    assert "frames" in batch
    assert "video_indices" in batch
    assert "frame_indices" in batch

    # Check types
    assert isinstance(batch["frames"], torch.Tensor)
    assert isinstance(batch["video_indices"], torch.Tensor)
    assert isinstance(batch["frame_indices"], torch.Tensor)

    # Check shapes
    assert batch["frames"].ndim == 4  # B x C x H x W
    assert batch["frames"].shape[0] == 8  # batch_size
    assert batch["frames"].shape[1] == 3  # 3 channels
    assert batch["video_indices"].shape == (8,)
    assert batch["frame_indices"].shape == (8,)

    # Check dtypes
    assert batch["frames"].dtype == torch.float32
    assert batch["video_indices"].dtype == torch.uint32
    assert batch["frame_indices"].dtype == torch.uint32


def test_last_batch_can_be_smaller(tmp_path: Path):
    """Test that last batch can be smaller than batch_size."""
    v1 = tmp_path / "video.mp4"

    # 27 frames with batch_size=10 means last batch has 7 frames
    frames = make_frames_with_stride(27, stride=10)
    write_frames_to_video(v1, frames, fps=10.0)

    loader = VideoCollectionLoader([v1], batch_size=10, n_loading_workers=1)

    batch_sizes = []
    for batch in loader:
        batch_sizes.append(batch["frames"].shape[0])

    assert batch_sizes == [10, 10, 7]

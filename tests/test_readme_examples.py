"""Tests that verify all examples from README.md work correctly."""

import numpy as np
import torch
from pathlib import Path

from pvio.video_io import (
    get_video_metadata,
    check_num_frames,
    read_frames_from_video,
    write_frames_to_video,
)
from pvio.torch import VideoCollectionDataset, VideoCollectionDataLoader

from .test_utils import make_frames_with_stride


def test_readme_example_reading_video_metadata(tmp_path: Path):
    """Test the 'Reading video metadata' example from README."""
    # Create a test video first
    example_video = tmp_path / "example.mp4"
    frames = make_frames_with_stride(30, stride=10)
    write_frames_to_video(example_video, frames, fps=10.0)

    # Example from README:
    # To get the number of frames in a video
    n_frames = check_num_frames(example_video)
    assert isinstance(n_frames, int)
    assert n_frames == 30

    # To get more information
    meta = get_video_metadata(example_video)
    assert isinstance(meta, dict)
    assert "n_frames" in meta
    assert "frame_size" in meta
    assert "fps" in meta
    assert meta["n_frames"] == 30
    assert meta["frame_size"] == (32, 32)


def test_readme_example_reading_video_frames(tmp_path: Path):
    """Test the 'Reading video frames' example from README."""
    # Create a test video first
    example_video = tmp_path / "example.mp4"
    test_frames = make_frames_with_stride(10, stride=10)
    write_frames_to_video(example_video, test_frames, fps=10.0)

    # Example from README:
    # You can read a whole video
    frames, fps = read_frames_from_video(example_video)
    assert isinstance(frames, list)
    assert len(frames) == 10
    assert fps is not None

    # ... or just some frames
    frames, fps = read_frames_from_video(example_video, frame_indices=[0, 5])
    assert isinstance(frames, list)
    assert len(frames) == 2
    assert fps is not None


def test_readme_example_writing_a_video(tmp_path: Path):
    """Test the 'Writing a video' example from README."""
    # Example from README (verbatim):
    # Create dummy 32x32 RGB frames (H, W, C)
    frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

    # Save them to file
    output_video = tmp_path / "example.mp4"
    write_frames_to_video(output_video, frames, fps=25.0)

    # Verify the video was created
    assert output_video.exists()

    # Verify we can read it back
    n_frames = check_num_frames(output_video)
    assert n_frames == 10


def test_readme_example_pytorch_basic(tmp_path: Path):
    """Test basic PyTorch dataset/dataloader example from README."""
    # Create test videos
    video1 = tmp_path / "video1.mp4"
    video2 = tmp_path / "video2.mp4"
    write_frames_to_video(video1, make_frames_with_stride(20, stride=10), fps=10.0)
    write_frames_to_video(video2, make_frames_with_stride(15, stride=10), fps=10.0)

    # Example from README:
    # Initialize Dataset from video files
    paths = [video1, video2]
    ds = VideoCollectionDataset(paths)

    # Wrap in the special DataLoader
    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=2)

    # Now you can iterate over all frames from all videos
    total_frames = 0
    for batch in loader:
        frames = batch["frames"]  # torch.Tensor: B x C x H x W
        video_paths = batch["video_paths"]  # list of str (absolute POSIX paths)
        frame_indices = batch["frame_indices"]  # list of int

        # Verify batch structure
        assert isinstance(frames, torch.Tensor)
        assert frames.ndim == 4  # B x C x H x W
        assert frames.shape[1] == 3  # C = 3 channels
        assert isinstance(video_paths, list)
        assert isinstance(frame_indices, list)
        assert len(video_paths) == len(frame_indices) == frames.shape[0]

        total_frames += frames.shape[0]

    # Verify we got all frames
    assert total_frames == 35  # 20 + 15


def test_readme_example_pytorch_image_dirs(tmp_path: Path):
    """Test PyTorch dataset with image directories example from README."""
    import imageio.v2 as imageio

    # Create test directories with frames
    dir1 = tmp_path / "frames_dir1"
    dir2 = tmp_path / "frames_dir2"
    dir1.mkdir()
    dir2.mkdir()

    # Create dummy frames
    for i in range(10):
        img = np.full((32, 32, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(dir1 / f"frame_{i:03d}.png", img)

    for i in range(8):
        img = np.full((32, 32, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(dir2 / f"frame_{i:03d}.png", img)

    # Example from README:
    paths = [dir1, dir2]
    ds = VideoCollectionDataset(paths, as_image_dirs=True)

    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=2)

    total_frames = 0
    for batch in loader:
        frames = batch["frames"]
        assert isinstance(frames, torch.Tensor)
        total_frames += frames.shape[0]

    assert total_frames == 18  # 10 + 8


def test_readme_example_pytorch_with_transform(tmp_path: Path):
    """Test PyTorch dataset with transform example from README."""
    # Create test video
    video = tmp_path / "video.mp4"
    write_frames_to_video(video, make_frames_with_stride(20, stride=10), fps=10.0)

    # Example from README:
    def my_transform(frame):
        return frame * 2.0  # example: double pixel values

    paths = [video]
    ds = VideoCollectionDataset(paths, transform=my_transform)

    loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=2)

    # Verify transform was applied
    for batch in loader:
        frames = batch["frames"]
        # Original frames have values ~0.0, 0.039, 0.078, ...
        # After doubling: ~0.0, 0.078, 0.156, ...
        # At least some values should be > 0.1 after transform
        assert frames.max() > 0.1
        break  # Just check first batch


def test_readme_example_pytorch_with_chunk_size(tmp_path: Path):
    """Test PyTorch dataloader with chunk_size parameter from README."""
    # Create test video
    video = tmp_path / "video.mp4"
    write_frames_to_video(video, make_frames_with_stride(100, stride=10), fps=10.0)

    # Example from README:
    paths = [video]
    ds = VideoCollectionDataset(paths)

    # You can control the temporal chunking behavior with chunk_size
    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=2, chunk_size=500)

    # Verify it works
    total_frames = 0
    for batch in loader:
        total_frames += batch["frames"].shape[0]

    assert total_frames == 100


def test_readme_example_pytorch_with_buffer_size(tmp_path: Path):
    """Test PyTorch dataset with buffer_size parameter from README."""
    # Create test video
    video = tmp_path / "video.mp4"
    write_frames_to_video(video, make_frames_with_stride(50, stride=10), fps=10.0)

    # Example from README:
    paths = [video]
    ds = VideoCollectionDataset(paths, buffer_size=64)

    # Verify buffer_size was set
    assert ds.buffer_size == 64

    loader = VideoCollectionDataLoader(ds, batch_size=10, num_workers=2)

    # Verify it works
    total_frames = 0
    for batch in loader:
        total_frames += batch["frames"].shape[0]

    assert total_frames == 50


def test_readme_example_pytorch_combined_parameters(tmp_path: Path):
    """Test PyTorch example with all parameters combined."""
    # Create test videos
    video1 = tmp_path / "video1.mp4"
    video2 = tmp_path / "video2.mp4"
    write_frames_to_video(video1, make_frames_with_stride(40, stride=10), fps=10.0)
    write_frames_to_video(video2, make_frames_with_stride(30, stride=10), fps=10.0)

    # Combine multiple README examples:
    def my_transform(frame):
        return frame * 2.0

    paths = [video1, video2]
    ds = VideoCollectionDataset(paths, transform=my_transform, buffer_size=64)

    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=2, chunk_size=500)

    # Verify everything works together
    total_frames = 0
    for batch in loader:
        frames = batch["frames"]
        video_paths = batch["video_paths"]
        frame_indices = batch["frame_indices"]

        assert isinstance(frames, torch.Tensor)
        assert isinstance(video_paths, list)
        assert isinstance(frame_indices, list)

        total_frames += frames.shape[0]

    assert total_frames == 70  # 40 + 30

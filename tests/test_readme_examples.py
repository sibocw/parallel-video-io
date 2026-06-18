# ruff: noqa
# fmt: off

"""Tests that verify all examples from docs/examples.md work correctly."""

import numpy as np
import torch
import imageio.v2 as imageio
from pathlib import Path

from pvio.io import (
    get_video_metadata,
    check_num_frames,
    read_frames_from_video,
    write_frames_to_video,
    write_image_paths_to_video,
)
from pvio.video import EncodedVideo, ImageDirVideo
from pvio.torch_tools import (
    VideoCollectionDataset,
    VideoCollectionDataLoader,
    SimpleVideoCollectionLoader,
)

from .test_utils import make_frames_with_stride


def test_readme_example_reading_video_metadata(tmp_path: Path):
    """Test the 'Reading video metadata' example from README."""
    # Create a test video first
    example_video = tmp_path / "example.mp4"
    frames = make_frames_with_stride(30, stride=10)
    write_frames_to_video(example_video, frames, fps=10.0)

    # Example from docs/examples.md:
    # Number of frames in a video
    n_frames = check_num_frames(example_video)
    print(n_frames)  # integer

    # Full metadata
    meta = get_video_metadata(example_video)
    print(meta.n_frames, meta.frame_size, meta.fps)  # VideoMetadata named tuple

    # Verify the example works
    assert isinstance(n_frames, int)
    assert n_frames == 30
    assert meta.n_frames == 30
    assert len(meta.frame_size) == 2
    assert meta.fps is None or isinstance(meta.fps, float)


def test_readme_example_reading_video_frames(tmp_path: Path):
    """Test the 'Reading video frames' example from README."""
    # Create a test video first
    example_video = tmp_path / "example.mp4"
    test_frames = make_frames_with_stride(10, stride=10)
    write_frames_to_video(example_video, test_frames, fps=10.0)

    # Example from docs/examples.md:
    # Read all frames
    frames, fps = read_frames_from_video(example_video)

    # ... or just specific frames
    frames, fps = read_frames_from_video(example_video, frame_indices=[0, 5])

    # Verify the examples work
    assert isinstance(frames, list)
    assert len(frames) == 2  # We read frames 0 and 5
    assert isinstance(fps, float)


def test_readme_example_writing_video(tmp_path: Path):
    """Test the 'Writing a video' example from docs/examples.md."""
    # Example from docs/examples.md:
    # Create dummy 32x32 RGB frames (H, W, C)
    frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

    example_video = tmp_path / "example.mp4"
    write_frames_to_video(example_video, frames, fps=25.0)

    assert example_video.exists()
    assert check_num_frames(example_video) == 10

    # Override encoding quality (quality=18 for near-lossless output)
    example_hq = tmp_path / "example_hq.mp4"
    write_frames_to_video(example_hq, frames, fps=25.0, quality=18)

    assert example_hq.exists()
    assert check_num_frames(example_hq) == 10

    # Force the CPU encoder with a preset and raw FFmpeg flags
    example_cpu = tmp_path / "example_cpu.mp4"
    write_frames_to_video(
        example_cpu,
        frames,
        fps=25.0,
        mode="cpu",
        preset="slow",
        extra_ffmpeg_params=["-level", "4.0"],
    )

    assert example_cpu.exists()
    assert check_num_frames(example_cpu) == 10


def test_readme_example_write_image_paths_to_video(tmp_path: Path):
    """Test the 'Combining image files into a video' example from docs/examples.md."""
    frames_dir = tmp_path / "frames_dir"
    frames_dir.mkdir()
    for i in range(10):
        img = np.full((32, 32, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(frames_dir / f"frame{i:03d}.png", img)

    # Example from docs/examples.md:
    # Frames are encoded in the order given — sort the paths if order matters.
    image_paths = sorted(frames_dir.glob("frame*.png"))
    example_video = tmp_path / "example.mp4"
    write_image_paths_to_video(example_video, image_paths, fps=25.0)

    assert example_video.exists()
    assert check_num_frames(example_video) == 10


def test_readme_example_pytorch_dataset_dataloader(tmp_path: Path):
    """Test the 'Using the PyTorch dataset and dataloader' example from docs/examples.md."""
    # Create test data
    video1_path = tmp_path / "video1.mp4"
    video2_path = tmp_path / "video2.mp4"
    frames_dir1 = tmp_path / "frames_dir1"
    frames_dir2 = tmp_path / "frames_dir2"

    write_frames_to_video(video1_path, make_frames_with_stride(20, stride=10), fps=10.0)
    write_frames_to_video(video2_path, make_frames_with_stride(15, stride=10), fps=10.0)

    frames_dir1.mkdir()
    frames_dir2.mkdir()

    for i in range(8):
        img = np.full((32, 32, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(frames_dir1 / f"frame_{i:03d}.png", img)
    for i in range(6):
        img = np.full((32, 32, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(frames_dir2 / f"frame_{i:03d}.png", img)

    # Example from docs/examples.md:
    # From video files
    video1 = EncodedVideo(video1_path)
    video2 = EncodedVideo(video2_path)
    ds = VideoCollectionDataset([video1, video2])

    # ... or from image-frame directories
    video3 = ImageDirVideo(frames_dir1)
    video4 = ImageDirVideo(frames_dir2, frame_id_regex=r"frame\D*(\d+)(?!\d)")
    ds = VideoCollectionDataset([video3, video4])

    # Apply a transform to each frame after loading
    def my_transform(frame):
        return frame * 2.0

    ds = VideoCollectionDataset([video1, video2], transform=my_transform)

    # Larger buffer_size = faster decoding at the cost of memory (default: 64)
    video_with_buffer = EncodedVideo(video1_path, buffer_size=128)
    ds = VideoCollectionDataset([video_with_buffer])

    # Wrap in a DataLoader — other DataLoader kwargs are forwarded as usual
    loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=0)

    for batch in loader:
        frames = batch["frames"]           # torch.Tensor: (B, C, H, W)
        video_indices = batch["video_indices"]  # list[int] — index into the videos list
        frame_indices = batch["frame_indices"]  # list[int] — virtual frame index

        assert isinstance(frames, torch.Tensor)
        assert frames.ndim == 4  # B x C x H x W
        assert isinstance(video_indices, list)
        assert isinstance(frame_indices, list)
        break  # Just test one batch


def test_readme_example_simple_video_collection_loader(tmp_path: Path):
    """Test the 'Using SimpleVideoCollectionLoader' example from docs/examples.md."""
    # Create test data
    video1_path = tmp_path / "video1.mp4"
    video2_path = tmp_path / "video2.mp4"
    dir1_path = tmp_path / "dir1"

    write_frames_to_video(video1_path, make_frames_with_stride(20, stride=10), fps=10.0)
    write_frames_to_video(video2_path, make_frames_with_stride(15, stride=10), fps=10.0)

    dir1_path.mkdir()
    for i in range(8):
        img = np.full((32, 32, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(dir1_path / f"frame_{i:03d}.png", img)

    def my_transform(frame):
        return frame * 2.0  # example: double pixel values

    # Example from docs/examples.md:
    # Mix paths to video files, image directories, and pre-built Video objects freely
    videos = [str(video1_path), str(dir1_path), EncodedVideo(video2_path)]

    loader = SimpleVideoCollectionLoader(
        videos,
        batch_size=8,
        num_workers=0,  # Use 0 workers for testing
        transform=my_transform,                    # optional
        buffer_size=64,                            # optional (for video files)
        frame_id_regex=r"frame\D*(\d+)(?!\d)",    # optional (for image directories)
    )

    for batch in loader:
        frames = batch["frames"]           # torch.Tensor: (B, C, H, W)
        video_indices = batch["video_indices"]  # list[int]
        frame_indices = batch["frame_indices"]  # list[int]

        assert isinstance(frames, torch.Tensor)
        assert frames.ndim == 4  # B x C x H x W
        assert isinstance(video_indices, list)
        assert isinstance(frame_indices, list)
        break  # Just test one batch

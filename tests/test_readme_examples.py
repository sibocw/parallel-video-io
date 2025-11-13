"""Tests that verify all examples from README.md work correctly."""

import numpy as np
import torch
import imageio.v2 as imageio
from pathlib import Path

from pvio.io import (
    get_video_metadata,
    check_num_frames,
    read_frames_from_video,
    write_frames_to_video,
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

    # Example from README:
    # To get the number of frames in a video
    n_frames = check_num_frames(example_video)
    print(n_frames)  # integer

    # To get more information
    meta = get_video_metadata(example_video)
    print(meta)  # dict containing the keys "n_frames", "frame_size", and "fps"

    # Verify the example works
    assert isinstance(n_frames, int)
    assert n_frames == 30
    assert isinstance(meta, dict)
    assert "n_frames" in meta
    assert "frame_size" in meta
    assert "fps" in meta


def test_readme_example_reading_video_frames(tmp_path: Path):
    """Test the 'Reading video frames' example from README."""
    # Create a test video first
    example_video = tmp_path / "example.mp4"
    test_frames = make_frames_with_stride(10, stride=10)
    write_frames_to_video(example_video, test_frames, fps=10.0)

    # Example from README:
    # You can read a whole video
    frames, fps = read_frames_from_video(example_video)

    # ... or just some frames
    frames, fps = read_frames_from_video(example_video, frame_indices=[0, 5])

    # Verify the examples work
    assert isinstance(frames, list)
    assert len(frames) == 2  # We read frames 0 and 5
    assert isinstance(fps, float)


def test_readme_example_writing_video(tmp_path: Path):
    """Test the 'Writing a video' example from README."""
    # Example from README:
    # Create dummy 32x32 RGB frames (H, W, C)
    frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

    # Save them to file
    example_video = tmp_path / "example.mp4"
    write_frames_to_video(example_video, frames, fps=25.0)

    # Verify the example works
    assert example_video.exists()
    n_frames = check_num_frames(example_video)
    assert n_frames == 10


def test_readme_example_pytorch_dataset_dataloader(tmp_path: Path):
    """Test the 'Using the PyTorch dataset and dataloader' example from README."""
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

    # Example from README:
    # Create Video objects for video files
    video1 = EncodedVideo(video1_path)
    video2 = EncodedVideo(video2_path)
    ds = VideoCollectionDataset([video1, video2])

    # ... or from directories containing individual frames as images
    video3 = ImageDirVideo(frames_dir1)
    video4 = ImageDirVideo(frames_dir2, frame_id_regex=r"frame\D*(\d+)(?!\d)")
    ds = VideoCollectionDataset([video3, video4])

    # You can optionally provide a transform function
    def my_transform(frame):
        return frame * 2.0  # example: double pixel values

    ds = VideoCollectionDataset([video1, video2], transform=my_transform)

    # You can set a buffer_size parameter when creating EncodedVideo objects.
    video_with_buffer = EncodedVideo(video1_path, buffer_size=128)
    ds = VideoCollectionDataset([video_with_buffer])

    # Wrap dataset in a DataLoader
    loader = VideoCollectionDataLoader(
        ds, batch_size=8, num_workers=0
    )  # Use 0 workers for testing

    # Now you can iterate over the entire dataset in batches through a single iterator
    for batch in loader:
        frames = batch["frames"]  # torch.Tensor: B x C x H x W
        video_indices = batch["video_indices"]  # list of int (video indices)
        frame_indices = batch["frame_indices"]  # list of int

        # Verify the batch structure matches README documentation
        assert isinstance(frames, torch.Tensor)
        assert frames.ndim == 4  # B x C x H x W
        assert isinstance(video_indices, list)
        assert isinstance(frame_indices, list)
        break  # Just test one batch


def test_readme_example_simple_video_collection_loader(tmp_path: Path):
    """Test the 'Using the SimpleVideoCollectionLoader' example from README."""
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

    # Example from README:
    # Video specification can be mixed: path to real videos, path to directories of images,
    # and pre-created Video objects are all allowed.
    videos = [str(video1_path), str(dir1_path), EncodedVideo(video2_path)]

    loader = SimpleVideoCollectionLoader(
        videos,
        batch_size=8,
        num_workers=0,  # Use 0 workers for testing
        transform=my_transform,  # optional
        buffer_size=64,  # optional (for video files)
        frame_id_regex=r"frame\D*(\d+)(?!\d)",  # optional (for image directories)
    )

    # Iterate over the entire dataset in batches through a single iterator
    for batch in loader:
        frames = batch["frames"]  # torch.Tensor: B x C x H x W
        video_indices = batch["video_indices"]  # list of int (video indices)
        frame_indices = batch["frame_indices"]  # list of int

        # Verify the batch structure matches README documentation
        assert isinstance(frames, torch.Tensor)
        assert frames.ndim == 4  # B x C x H x W
        assert isinstance(video_indices, list)
        assert isinstance(frame_indices, list)
        break  # Just test one batch

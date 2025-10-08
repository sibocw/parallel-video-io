# parallel-video-io

Tools for reading and writing videos and for efficient frame-level loading with PyTorch.

This repository provides small, focused utilities around video I/O and a PyTorch-friendly iterable dataset + dataloader that make it easy to stream frames from many videos or directories of image frames in parallel.

## Key features
- Read frames from videos (random access or sequential) using imageio/ffmpeg.
- Write sequences of numpy frames to H.264 MP4 files with sane defaults.
- PyTorch-compatible `VideoCollectionDataset` and `VideoCollectionDataLoader` that provide a simple iterator that uses multiple processes to load data from different videos under the hood. This is especially handy for running trained deep learning models on many videos in production.

## Table of contents
- [Installation](#installation)
- [Quick examples](#quick-examples)
	- [Reading video metadata](#reading-video-metadata)
	- [Reading video frames](#reading-video-frames)
	- [Writing a video](#writing-a-video)
	- [Using the PyTorch dataset and dataloader](#using-the-pytorch-dataset-and-dataloader)
- [Testing](#testing)
- [Notes & troubleshooting](#notes--troubleshooting)

## Installation

This project targets Python >= 3.10. The library's runtime dependencies are listed in `pyproject.toml` (torch, imageio, imageio-ffmpeg, torchcodec, joblib, tqdm, numpy, pytest).

If you're using pip in a development environment, install editable with:

```bash
pip install -e .
```

Or with Poetry:

```bash
poetry install
```

You can include this package as a dependency for your project by including the following in your `pyproject.toml`:

```toml
[project]
# ... other stuff
dependencies = [
    # ...
    "parallel-video-io @ git+https://github.com/sibocw/parallel-video-io.git",
]
```

Make sure `ffmpeg` is available on your `$PATH` (required by imageio-ffmpeg).

## Quick examples

These examples use NumPy arrays for frames in (height, width, channels) order and uint8 dtype.

### Reading video metadata

```python
from pvio.video_io import get_video_metadata, check_num_frames

# To get the number of frames in a video
n_frames = check_num_frames("example.mp4")
print(n_frames)  # this is an integer frame count

# To get more information
# Note that this function actually caches these information in a JSON file. To control
# whether you want to save the cache file or disregard existing cache files, set the
# `cache_metadata` (default True) and `use_cached_metadata` (default True) arguments.
meta = get_video_metadata("example.mp4")
print(meta)  # meta is a dictionary containing the keys "n_frames", "frame_size", "fps"
```

### Reading video frames

```python
from pvio.video_io import read_frames_from_video

# You can read a whole video
frames, fps = read_frames_from_video("example.mp4")

# ... or just some frames
frames, fps = read_frames_from_video("example.mp4", frame_indices=[0, 5])
```

### Writing a video

```python
import numpy as np
from pvio.video_io import write_frames_to_video

# Create dummy 32x32 RGB frames (H, W, C)
frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

# Save them to file
# There are more complex video writing parameters that can be tuned - see the docstring
# for details.
write_frames_to_video("example.mp4", frames, fps=25.0)
```

Notes: the writer verifies that all frames share the same (height, width). FFmpeg can
automatically resize frames to meet codec alignment requirements; for deterministic
results, use dimensions divisible by 16.

### Using the PyTorch dataset and dataloader

The `VideoCollectionDataset` iterates frames either from video files or from directories containing individual image frames.

```python
from pvio.torch import VideoCollectionDataset, VideoCollectionDataLoader

# Initialize Dataset from video files
paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
ds = VideoCollectionDataset(paths)
# ... or from directories containing individual frames as images
paths = ["/path/to/frames_dir1", "/path/to/frames_dir2"]
# To control sorting of frame files within each dir, use the `frame_sorting` argument
# (see docstring for details)
ds = VideoCollectionDataset(paths, as_image_dirs=True)

# Wrap in the special DataLoader
# (you can add other DataLoader keyword arguments if you wish)
loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=4)

# Now you can iterate over all frames from all videos in a single iterator. Behind the
# scenes, these frames are fetched in parallel (each worker handles one video at a time)
for batch in loader:
	frames = batch["frames"]  # torch.Tensor: B x C x H x W
	video_paths = batch["video_paths"]  # list of Path or str, depending on input
	frame_indices = batch["frame_indices"]  # list of int
```

When loading from video files (as_image_dirs=False), the dataset uses `torchcodec`'s `VideoDecoder` to decode frames and `get_video_metadata` to build per-video frame counts; you may want to enable caching if you index many large files.

## Testing

The test suite uses pytest. Run it from the repository root:

```bash
pytest tests
```

There are a few tests that write small MP4 files using imageio/ffmpeg; ensure `ffmpeg` is available in the environment where tests run.

## Notes & troubleshooting

- FFmpeg macroblock constraints: some ffmpeg builds require frame dimensions to be divisible by 16. If you see a warning about `macro_block_size=16` and unexpected resizing, choose frame sizes divisible by 16 in production pipelines.
- If you plan to decode many large videos, enabling metadata caching (the package writes a `.metadata.json` next to each video when `get_video_metadata` is called) will speed up repeated indexing.
- The PyTorch loader expects the dataset passed to `VideoCollectionDataLoader` to be an instance of `VideoCollectionDataset` and enforces the built-in collate function.

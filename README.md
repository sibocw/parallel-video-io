# parallel-video-io

Tools for reading and writing videos and for efficient frame-level loading with PyTorch.

This repository provides small, focused utilities around video I/O and a PyTorch-friendly
iterable dataset + dataloader that make it easy to stream frames from many videos or
directories of image frames in parallel.

## Key features
- Read frames from videos (random access or sequential) using imageio/ffmpeg.
- Write sequences of numpy frames to H.264 MP4 files with sane defaults.
- PyTorch-compatible `VideoCollectionDataset` and `VideoCollectionDataLoader` that provide a simple iterator that uses multiple processes to load data from different videos under the hood. This is especially handy for running trained deep learning models on many videos in production.

# parallel-video-io

Tools for reading and writing videos and for efficient frame-level loading with PyTorch.

This repository provides small, focused utilities around video I/O and a PyTorch-friendly
iterable dataset + dataloader that make it easy to stream frames from many videos or
directories of image frames in parallel.

## Key features
- Read frames from videos (random access or sequential) using imageio/ffmpeg.
- Write sequences of numpy frames to H.264 MP4 files with sane defaults.
- PyTorch-compatible `VideoCollectionDataset` and `VideoCollectionDataLoader` that provide a simple iterator that uses multiple processes to load data from different videos under the hood.

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

This project targets Python >= 3.10. The library's runtime dependencies are listed in
`pyproject.toml` (torch, imageio, imageio-ffmpeg, torchcodec, joblib, tqdm, numpy, pytest).

If you're using pip in a development environment, install editable with:

```bash
pip install -e .
```

Or with Poetry:

```bash
poetry install
```

Make sure `ffmpeg` is available on your `$PATH` (required by imageio-ffmpeg).

## Quick examples

These examples use NumPy arrays for frames in (height, width, channels) order and uint8 dtype.

### Reading video metadata

```python
from pvio.video_io import get_video_metadata
from pathlib import Path

video = Path("example.mp4")
meta = get_video_metadata(video)
print(meta)
```

### Reading video frames

```python
from pvio.video_io import read_frames_from_video
from pathlib import Path

video = Path("example.mp4")
frames, fps = read_frames_from_video(video, frame_indices=[0, 5])
print(len(frames), fps)
```

### Writing a video

```python
from pathlib import Path
import numpy as np
from pvio.video_io import write_frames_to_video

# Create 32x32 RGB frames (H, W, C)
frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

out = Path("example.mp4")
write_frames_to_video(out, frames, fps=25.0)
```

Notes: the writer verifies that all frames share the same (height, width). FFmpeg can
automatically resize frames to meet codec alignment requirements; for deterministic
results, use dimensions divisible by 16 (e.g., 32, 64).

### Using the PyTorch dataset and dataloader

The `VideoCollectionDataset` iterates frames either from video files or from directories
containing individual image frames.

```python
from pathlib import Path
from pvio.torch import VideoCollectionDataset, VideoCollectionDataLoader

# Example: treat directories as image-frame collections
paths = ["/path/to/frames_dir1", "/path/to/frames_dir2"]
ds = VideoCollectionDataset(paths, as_image_dirs=True)

# Wrap in the special DataLoader which enforces the built-in collate and assigns workers
loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=4)

for batch in loader:
		# batch is a dict with keys: frames (torch.Tensor: B x C x H x W),
		# video_paths (list[str]), frame_indices (list[int])
		frames = batch["frames"]
		# process frames...
		break
```

When loading from video files (as_image_dirs=False), the dataset uses `torchcodec`'s
`VideoDecoder` to decode frames and `get_video_metadata` to build per-video frame counts;
you may want to enable caching if you index many large files.

## Testing

The test suite uses pytest. Run it from the repository root:

```bash
pytest -q
```

There are a few tests that write small MP4 files using imageio/ffmpeg; ensure `ffmpeg` is available in the environment where tests run.

## Notes & troubleshooting

- FFmpeg macroblock constraints: some ffmpeg builds require frame dimensions to be
	divisible by 16. If you see a warning about `macro_block_size=16` and unexpected
	resizing, choose frame sizes divisible by 16 in production pipelines.
- If you plan to decode many large videos, enabling metadata caching (the package
	writes a `.metadata.json` next to each video when `get_video_metadata` is called)
	will speed up repeated indexing.
- The PyTorch loader expects the dataset passed to `VideoCollectionDataLoader` to be an
	instance of `VideoCollectionDataset` and enforces the built-in collate function.

# parallel-video-io

Tools for reading and writing videos and for efficient frame-level loading with PyTorch.

This repository provides small, focused utilities around video I/O and a PyTorch-friendly iterable dataset + dataloader that make it easy to stream frames from many videos or directories of image frames in parallel.

## Key features
- Read frames from videos (random access or sequential) using imageio/ffmpeg.
- Write sequences of numpy frames to H.264 MP4 files with sane defaults.
- PyTorch-compatible `VideoCollectionDataset` and `VideoCollectionDataLoader` that provide a simple iterator that uses multiple processes to load data from different videos under the hood. Videos and frames are distributed across workers for efficient parallel loading.
- `SimpleVideoCollectionLoader`: an even easier API that combines dataset and dataloader creation in one step.

## Table of contents
- [Installation](#installation)
- [Quick examples](#quick-examples)
	- [Reading video metadata](#reading-video-metadata)
	- [Reading video frames](#reading-video-frames)
	- [Writing a video](#writing-a-video)
	- [Using the PyTorch dataset and dataloader](#using-the-pytorch-dataset-and-dataloader)
	- [Using the simplified dataloader](#using-the-simplevideocollectionloader)
- [Testing](#testing)
- [Notes & troubleshooting](#notes--troubleshooting)

## Installation

Install from PyPI:
```bash
pip install parallel-video-io
```

To clone a copy and install in editable mode:

```bash
git clone git@github.com:sibocw/parallel-video-io.git
cd parallel-video-io

# with pip
pip install -e . --config-settings editable_mode=compat

# ... or with Poetry
poetry install
```

Make sure `ffmpeg` is available on your `$PATH` (required by imageio-ffmpeg).

## Quick examples

These examples use NumPy arrays for frames in (height, width, channels) order and uint8 dtype.

### Reading video metadata

```python
from pvio.video_io import get_video_metadata, check_num_frames

# To get the number of frames in a video
n_frames = check_num_frames("example.mp4")
print(n_frames)  # integer

# To get more information
# This function actually caches these information in a JSON file. To control whether you
# want to use caching, modify the `cache_metadata` and `use_cached_metadata` arguments.
meta = get_video_metadata("example.mp4")
print(meta)  # dict containing the keys "n_frames", "frame_size", and "fps"
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
# More complex video writing parameters are available - see the docstring for details
write_frames_to_video("example.mp4", frames, fps=25.0)
```

Notes: the writer verifies that all frames share the same (height, width). FFmpeg can
automatically resize frames to meet codec alignment requirements; for deterministic
results, use dimensions divisible by 16.

### Using the PyTorch dataset and dataloader

The `VideoCollectionDataset` iterates frames either from video files or from directories containing individual image frames. Then, you can use `VideoCollectionDataLoader` to load frames in parallel. This can be very handy for inference pipelines of neural networks that independently process all frames in a video. [TorchCodec](https://meta-pytorch.org/torchcodec) is used under the hood.

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

# You can optionally provide a transform function that will be applied to each frame
# after loading (applied to CHW float tensors in [0, 1])
paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
def my_transform(frame):
    return frame * 2.0  # example: double pixel values
ds = VideoCollectionDataset(paths, transform=my_transform)

# You can also set a buffer_size parameter. This is the number of frames to decode at
# once when reading videos (default 64). Larger buffer size = faster loading at the cost
# of memory usage.
ds = VideoCollectionDataset(paths, buffer_size=64)

# Wrap in the special DataLoader
# (you can add other torch.utils.data.DataLoader keyword arguments if you wish)
loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=4)

# Now you can iterate over all frames from all videos in a single iterator. Behind the
# scenes, frames are distributed across workers for efficient parallel loading
for batch in loader:
    frames = batch["frames"]  # torch.Tensor: B x C x H x W
    video_indices = batch["video_indices"]  # list of int (video indices)
    frame_indices = batch["frame_indices"]  # list of int
```

### Using the SimpleVideoCollectionLoader

If you don't mind breaking the standard `Dataset` + `DataLoader` pattern with `torch.utils.data`, you can use `SimpleVideoCollectionLoader` which combines dataset and dataloader creation:

```python
from pvio.torch import SimpleVideoCollectionLoader

# All VideoCollectionDataset parameters, plus DataLoader parameters in one step
loader = SimpleVideoCollectionLoader(
    ["/path/to/video1.mp4", "/path/to/video2.mp4"],
    batch_size=8,
    num_workers=4,
    transform=my_transform,  # optional
    buffer_size=64,  # optional
)

for batch in loader:
    frames = batch["frames"]  # torch.Tensor: B x C x H x W
    video_indices = batch["video_indices"]  # list of int (video indices)
    frame_indices = batch["frame_indices"]  # list of int
```

When loading from video files (as_image_dirs=False), the dataset uses `torchcodec`'s `VideoDecoder` to decode frames and `get_video_metadata` to build per-video frame counts; you may want to enable caching if you index many large files.

## Testing

The test suite uses pytest. Run it from the repository root:

```bash
pytest tests
```

The tests are organized into:
- `test_video_io.py` - Tests for video I/O functions
- `test_torch.py` - Unit tests for VideoCollectionDataset
- `test_integration.py` - Integration tests with parallel loading

There are a few tests that write small MP4 files using imageio/ffmpeg; ensure `ffmpeg` is available in the environment where tests run.


## Notes & troubleshooting

- FFmpeg macroblock constraints: some ffmpeg builds require frame dimensions to be divisible by 16. If you see a warning about `macro_block_size=16` and unexpected resizing, choose frame sizes divisible by 16 in production pipelines.
- If you plan to decode many large videos, enabling metadata caching will speed up repeated indexing (the package writes a `.metadata.json` for each video under the same directory when `get_video_metadata` is called).
- The PyTorch loader expects the dataset passed to `VideoCollectionDataLoader` to be an instance of `VideoCollectionDataset` and enforces the built-in collate function.
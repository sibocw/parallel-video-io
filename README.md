# parallel-video-io

Tools for reading and writing videos and for efficient frame-level loading with PyTorch.

This repository provides small, focused utilities around video I/O and a PyTorch-friendly iterable dataset + dataloader that make it easy to stream frames from many videos or directories of image frames in parallel.

## Key features
- Read frames from videos (random access or sequential) using imageio/ffmpeg.
- Write sequences of numpy frames to H.264 MP4 files with sane defaults.
- PyTorch-compatible `VideoCollectionDataset` and `VideoCollectionDataLoader` that provide a simple iterator that uses multiple processes to load data from different videos under the hood.
- `SimpleVideoCollectionLoader`: an even easier API that combines dataset and dataloader creation in one step.

## Table of contents
- [Installation](#installation)
- [Quick examples](#quick-examples)
	- [Reading video metadata](#reading-video-metadata)
	- [Reading video frames](#reading-video-frames)
	- [Writing a video](#writing-a-video)
	- [Using the PyTorch dataset and dataloader](#using-the-pytorch-dataset-and-dataloader)
	- [Using simplified dataloader](#using-simplevideocollectionloader)
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
from pvio.torch import EncodedVideo  # for "real" videos (e.g. MP4 files)
from pvio.torch import ImageDirVideo  # for directories containing individual images
from pvio.torch import VideoCollectionDataset, VideoCollectionDataLoader

# Create Video objects for video files
video1 = EncodedVideo("path/to/video1.mp4")
video2 = EncodedVideo("path/to/video2.mp4")
ds = VideoCollectionDataset([video1, video2])

# ... or from directories containing individual frames as images
video3 = ImageDirVideo("path/to/frames_dir1")
# (hint: you can use a custom regular expression to control how frame IDs are parsed)
video4 = ImageDirVideo("path/to/frames_dir2", frameid_regex=r"frame\D*(\d+)(?!\d)")
ds = VideoCollectionDataset([video3, video4])

# You can optionally provide a transform function that will be applied to each frame
# after loading (frames already in float tensor format, ranged [0, 1], in CHW format)
# (hint: these can also be from torchvision.transforms)
def my_transform(frame):
    return frame * 2.0  # example: double pixel values
ds = VideoCollectionDataset([video1, video2], transform=my_transform)

# You can set a buffer_size parameter when creating EncodedVideo objects. 
# This is the number of frames to decode at once (default 64).
# Larger buffer size = faster loading at the cost of memory usage.
video_with_buffer = EncodedVideo("path/to/video.mp4", buffer_size=128)
ds = VideoCollectionDataset([video_with_buffer])

# Wrap dataset in a DataLoader
# (you can supply other torch.utils.data.DataLoader keyword arguments if you wish)
loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=4)

# Now you can iterate over the entire dataset in batches through a single interator
# Behind the scenes, frames are distributed across workers for efficient loading
for batch in loader:
    frames = batch["frames"]  # torch.Tensor: B x C x H x W
    video_indices = batch["video_indices"]  # list of int (video indices)
    frame_indices = batch["frame_indices"]  # list of int
```

### Using SimpleVideoCollectionLoader

If you don't mind breaking the standard `Dataset` + `DataLoader` pattern with `torch.utils.data`, you can use `SimpleVideoCollectionLoader`, which combines dataset and dataloader creation. This dataloader can also automatically create the appropriate Video objects from paths:

```python
from pvio.torch import SimpleVideoCollectionLoader

# Video specification can be mixed: path to real videos, path to directories of images,
# and pre-created Video objects are all allowed.
videos = ["path/to/video1.mp4", "path/to/dir1/", EncodedVideo("path/to/video2.mp4")]

# Supply all Video backend parameters, VideoCollectionDataset parameters, and DataLoader
# parameters in one call.
loader = SimpleVideoCollectionLoader(
    videos,
    batch_size=8,
    num_workers=4,
    transform=my_transform,  # optional
    buffer_size=64,  # optional (for video files)
    frameid_regex=r"frame\D*(\d+)(?!\d)",  # optional (for image directories)
)

# Iterate over the entire dataset in batches through a single interator
for batch in loader:
    frames = batch["frames"]  # torch.Tensor: B x C x H x W
    video_indices = batch["video_indices"]  # list of int (video indices)
    frame_indices = batch["frame_indices"]  # list of int
```

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
- If you have a non-standard data format, you can implement your own backend by creating a subclass of `pvio.torch.Video`.
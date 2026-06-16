# Examples

These examples use NumPy arrays with shape `(height, width, channels)` and `uint8` dtype.

## Reading video metadata

```python
from pvio.io import get_video_metadata, check_num_frames

# Number of frames in a video
n_frames = check_num_frames("example.mp4")
print(n_frames)  # integer

# Full metadata — results are cached to a JSON file alongside the video.
# Control caching with the `cache_metadata` and `use_cached_metadata` arguments.
meta = get_video_metadata("example.mp4")
print(meta)  # dict with keys "n_frames", "frame_size", and "fps"
```

## Reading video frames

```python
from pvio.io import read_frames_from_video

# Read all frames
frames, fps = read_frames_from_video("example.mp4")

# ... or just specific frames
frames, fps = read_frames_from_video("example.mp4", frame_indices=[0, 5])
```

## Writing a video

```python
import numpy as np
from pvio.io import write_frames_to_video

# Create dummy 32×32 RGB frames (H, W, C)
frames = [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(10)]

write_frames_to_video("example.mp4", frames, fps=25.0)

# Override encoding quality — lower CRF = higher quality, larger file.
# Default is CRF 20 (conservative vs FFmpeg's 23); use 18 for lossless-like output.
write_frames_to_video(
    "example_hq.mp4", frames, fps=25.0,
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-profile:v", "high", "-level", "4.0"],
)
```

!!! note "FFmpeg macroblock alignment"
    The writer verifies that all frames share the same `(height, width)`. Some FFmpeg
    builds require frame dimensions to be divisible by 16; use such dimensions to avoid
    unexpected automatic resizing.

## Using the PyTorch dataset and dataloader

`VideoCollectionDataset` iterates frames from video files or directories of image frames.
Wrap it in `VideoCollectionDataLoader` to load in parallel across workers — useful for
inference pipelines that process every frame independently.
[TorchCodec](https://pytorch.org/torchcodec) is used for decoding.

```python
from pvio.video import EncodedVideo   # for encoded video files (e.g. MP4)
from pvio.video import ImageDirVideo  # for directories of individual frame images
from pvio.torch_tools import VideoCollectionDataset, VideoCollectionDataLoader

# From video files
video1 = EncodedVideo("path/to/video1.mp4")
video2 = EncodedVideo("path/to/video2.mp4")
ds = VideoCollectionDataset([video1, video2])

# ... or from image-frame directories
video3 = ImageDirVideo("path/to/frames_dir1")
# Use a custom regex to control how frame indices are parsed from filenames
video4 = ImageDirVideo("path/to/frames_dir2", frame_id_regex=r"frame\D*(\d+)(?!\d)")
ds = VideoCollectionDataset([video3, video4])

# Apply a transform to each frame after loading
# (frames arrive as CHW float32 tensors in [0, 1])
def my_transform(frame):
    return frame * 2.0

ds = VideoCollectionDataset([video1, video2], transform=my_transform)

# Larger buffer_size = faster decoding at the cost of memory (default: 64)
video_with_buffer = EncodedVideo("path/to/video.mp4", buffer_size=128)
ds = VideoCollectionDataset([video_with_buffer])

# Wrap in a DataLoader — other DataLoader kwargs are forwarded as usual
loader = VideoCollectionDataLoader(ds, batch_size=8, num_workers=4)

for batch in loader:
    frames = batch["frames"]           # torch.Tensor: (B, C, H, W)
    video_indices = batch["video_indices"]  # list[int] — index into the videos list
    frame_indices = batch["frame_indices"]  # list[int] — virtual frame index
```

## Using SimpleVideoCollectionLoader

`SimpleVideoCollectionLoader` combines dataset and dataloader creation in one call and
automatically constructs the right `Video` object from a path:

```python
from pvio.torch_tools import SimpleVideoCollectionLoader
from pvio.video import EncodedVideo

# Mix paths to video files, image directories, and pre-built Video objects freely
videos = ["path/to/video1.mp4", "path/to/frames_dir/", EncodedVideo("path/to/video2.mp4")]

loader = SimpleVideoCollectionLoader(
    videos,
    batch_size=8,
    num_workers=4,
    transform=my_transform,                      # optional
    buffer_size=64,                              # optional (for video files)
    frame_id_regex=r"frame\D*(\d+)(?!\d)",       # optional (for image directories)
)

for batch in loader:
    frames = batch["frames"]           # torch.Tensor: (B, C, H, W)
    video_indices = batch["video_indices"]  # list[int]
    frame_indices = batch["frame_indices"]  # list[int]
```

import sys

if sys.platform != "linux":
    raise RuntimeError(
        f"parallel-video-io only supports Linux. Current platform: {sys.platform!r}."
    )

from . import io
from . import torch_tools
from . import video

from .io import (
    read_frames_from_video,
    write_frames_to_video,
    check_num_frames,
    get_video_metadata,
)
from .video import Video, EncodedVideo, ImageDirVideo
from .torch_tools import (
    VideoCollectionDataset,
    VideoCollectionDataLoader,
    SimpleVideoCollectionLoader,
)

__all__ = [
    "io",
    "torch_tools",
    "video",
    "read_frames_from_video",
    "write_frames_to_video",
    "check_num_frames",
    "get_video_metadata",
    "Video",
    "EncodedVideo",
    "ImageDirVideo",
    "VideoCollectionDataset",
    "VideoCollectionDataLoader",
    "SimpleVideoCollectionLoader",
]

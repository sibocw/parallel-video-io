import sys

if sys.platform != "linux":
    raise RuntimeError(
        f"parallel-video-io only supports Linux. Current platform: {sys.platform!r}."
    )

# Curate a flat public API. The submodules (pvio.io, pvio.video, pvio.torch_tools)
# remain importable directly (e.g. `from pvio.io import ...`) but are intentionally
# not re-exported as bare names here, so there is one obvious way to reach each
# symbol and `from pvio import io` does not shadow the stdlib `io` module.
from .io import (
    read_frames_from_video,
    write_frames_to_video,
    check_num_frames,
    get_video_metadata,
    VideoMetadata,
)
from .video import Video, EncodedVideo, ImageDirVideo
from .torch_tools import (
    VideoCollectionDataset,
    VideoCollectionDataLoader,
    SimpleVideoCollectionLoader,
)

__all__ = [
    "read_frames_from_video",
    "write_frames_to_video",
    "check_num_frames",
    "get_video_metadata",
    "VideoMetadata",
    "Video",
    "EncodedVideo",
    "ImageDirVideo",
    "VideoCollectionDataset",
    "VideoCollectionDataLoader",
    "SimpleVideoCollectionLoader",
]

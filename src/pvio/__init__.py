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
    write_image_paths_to_video,
    check_num_frames,
    get_video_metadata,
    VideoMetadata,
)
_LAZY_IMPORTS: dict[str, str] = {
    "Video": ".video",
    "EncodedVideo": ".video",
    "ImageDirVideo": ".video",
    "VideoCollectionDataset": ".torch_tools",
    "VideoCollectionDataLoader": ".torch_tools",
    "SimpleVideoCollectionLoader": ".torch_tools",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], package=__name__)
        value = getattr(mod, name)
        globals()[name] = value  # cache to avoid repeated lookups
        return value
    raise AttributeError(f"module 'pvio' has no attribute {name!r}")

__all__ = [
    "read_frames_from_video",
    "write_frames_to_video",
    "write_image_paths_to_video",
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

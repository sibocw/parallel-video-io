"""Tests for the curated top-level public API (pvio.__init__)."""

import pvio


EXPECTED_PUBLIC_NAMES = {
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
}


def test_all_lists_exactly_the_curated_symbols():
    """__all__ exposes the flat API and does not re-export bare submodules."""
    assert set(pvio.__all__) == EXPECTED_PUBLIC_NAMES
    # Submodules are intentionally absent from the curated flat surface.
    for submodule in ("io", "video", "torch_tools"):
        assert submodule not in pvio.__all__


def test_curated_symbols_are_importable_from_package():
    """Every name in __all__ is actually present on the package."""
    for name in pvio.__all__:
        assert hasattr(pvio, name), f"{name} missing from pvio"


def test_submodules_remain_importable_directly():
    """Direct submodule imports keep working even though they aren't re-exported."""
    from pvio.io import read_frames_from_video  # noqa: F401
    from pvio.video import EncodedVideo  # noqa: F401
    from pvio.torch_tools import SimpleVideoCollectionLoader  # noqa: F401

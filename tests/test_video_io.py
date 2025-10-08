import pytest
import numpy as np
from pathlib import Path

from pvio.video_io import (
    write_frames_to_video,
    read_frames_from_video,
    get_video_metadata,
)


def _make_dummy_frames(n=10, h=16, w=12, channels=3):
    # Create deterministic frames with a simple pattern
    frames = []
    for i in range(n):
        arr = np.zeros((h, w, channels), dtype=np.uint8)
        arr[..., 0] = i  # vary red channel to identify frames
        frames.append(arr)
    return frames


def test_write_and_read_dummy_video(tmp_path: Path):
    # use dimensions divisible by 16 to avoid ffmpeg auto-resize
    frames = _make_dummy_frames(n=5, h=32, w=32, channels=3)
    out = tmp_path / "out.mp4"

    # write
    write_frames_to_video(out, frames, fps=10.0)

    assert out.exists()

    # read back first frame and metadata
    read_frames, fps = read_frames_from_video(out, frame_indices=[0])
    assert fps is not None
    assert len(read_frames) == 1
    assert read_frames[0].shape[:2] == (32, 32)

    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta["n_frames"] >= 1
    assert meta["frame_size"] == (32, 32)


def test_write_frames_mismatched_sizes_raises(tmp_path: Path):
    frames = _make_dummy_frames(n=3, h=20, w=20)
    # change second frame shape
    frames[1] = np.zeros((10, 10, 3), dtype=np.uint8)
    out = tmp_path / "bad.mp4"
    with pytest.raises(ValueError):
        write_frames_to_video(out, frames, fps=5.0)

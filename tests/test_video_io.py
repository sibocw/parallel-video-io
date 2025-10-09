import pytest
import numpy as np
from pathlib import Path

from pvio.video_io import (
    write_frames_to_video,
    read_frames_from_video,
    get_video_metadata,
    check_num_frames,
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


def test_get_video_metadata_uses_cache(tmp_path: Path):
    # prepare dummy video and metadata
    frames = _make_dummy_frames(n=2, h=32, w=32, channels=3)
    out = tmp_path / "cache.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    meta_file = out.with_suffix(".metadata.json")
    # write fake metadata
    fake_meta = {"n_frames": 123, "frame_size": [32, 32], "fps": 12.0}
    import json

    with open(meta_file, "w") as f:
        json.dump(fake_meta, f)

    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta["n_frames"] == 123
    assert meta["frame_size"] == (32, 32)
    assert meta["fps"] == 12.0


def test_get_video_metadata_corrupted_cache_raises(tmp_path: Path):
    frames = _make_dummy_frames(n=1, h=32, w=32, channels=3)
    out = tmp_path / "corrupt.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    meta_file = out.with_suffix(".metadata.json")
    with open(meta_file, "w") as f:
        f.write("not a json")

    with pytest.raises(Exception):
        # implementation currently prints and re-raises
        get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)


def test_check_num_frames_on_invalid_file_raises(tmp_path: Path):
    bad = tmp_path / "not_a_video.txt"
    bad.write_text("hello")
    with pytest.raises(RuntimeError):
        check_num_frames(bad)


def test_write_frames_to_video_empty_raises(tmp_path: Path):
    out = tmp_path / "empty.mp4"
    with pytest.raises(ValueError):
        write_frames_to_video(out, [], fps=10.0)

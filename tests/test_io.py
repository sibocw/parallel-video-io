"""Tests for io module."""

import pytest
import numpy as np
import json
from pathlib import Path

from pvio.io import (
    write_frames_to_video,
    read_frames_from_video,
    get_video_metadata,
    check_num_frames,
)

from .test_utils import make_simple_frames


def test_write_and_read_dummy_video(tmp_path: Path):
    """Test basic video writing and reading."""
    frames = make_simple_frames(n=5, h=32, w=32, channels=3)
    out = tmp_path / "out.mp4"

    # Write video
    write_frames_to_video(out, frames, fps=10.0)
    assert out.exists()

    # Read back first frame and metadata
    read_frames, fps = read_frames_from_video(out, frame_indices=[0])
    assert fps is not None
    assert len(read_frames) == 1
    assert read_frames[0].shape[:2] == (32, 32)

    # Check metadata
    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta["n_frames"] >= 1
    assert meta["frame_size"] == (32, 32)


def test_write_frames_mismatched_sizes_raises(tmp_path: Path):
    """Test that mismatched frame sizes raise an error."""
    frames = make_simple_frames(n=3, h=20, w=20)
    frames[1] = np.zeros((10, 10, 3), dtype=np.uint8)  # Wrong size
    out = tmp_path / "bad.mp4"

    with pytest.raises(ValueError):
        write_frames_to_video(out, frames, fps=5.0)


def test_get_video_metadata_uses_cache(tmp_path: Path):
    """Test that cached metadata is used when available."""
    frames = make_simple_frames(n=2, h=32, w=32, channels=3)
    out = tmp_path / "cache.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    # Write fake metadata
    meta_file = out.with_suffix(".metadata.json")
    fake_meta = {"n_frames": 123, "frame_size": [32, 32], "fps": 12.0}
    with open(meta_file, "w") as f:
        json.dump(fake_meta, f)

    # Should use cached metadata
    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta["n_frames"] == 123
    assert meta["frame_size"] == (32, 32)
    assert meta["fps"] == 12.0


def test_get_video_metadata_corrupted_cache_raises(tmp_path: Path):
    """Test that corrupted cache raises an error."""
    frames = make_simple_frames(n=1, h=32, w=32, channels=3)
    out = tmp_path / "corrupt.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    # Write invalid JSON
    meta_file = out.with_suffix(".metadata.json")
    with open(meta_file, "w") as f:
        f.write("not a json")

    with pytest.raises(Exception):
        get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)


def test_check_num_frames_on_invalid_file_raises(tmp_path: Path):
    """Test that invalid files raise an error."""
    bad = tmp_path / "not_a_video.txt"
    bad.write_text("hello")

    with pytest.raises(RuntimeError):
        check_num_frames(bad)


def test_write_frames_to_video_empty_raises(tmp_path: Path):
    """Test that empty frame list raises an error."""
    out = tmp_path / "empty.mp4"

    with pytest.raises(ValueError):
        write_frames_to_video(out, [], fps=10.0)

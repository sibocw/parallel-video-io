"""Tests for io module."""

import pytest
import numpy as np
import json
import av
import imageio.v2 as imageio
from pathlib import Path

from pvio.io import (
    write_frames_to_video,
    write_image_paths_to_video,
    read_frames_from_video,
    get_video_metadata,
    check_num_frames,
    _file_signature,
)

from .test_utils import make_simple_frames


def _write_frame_images(
    dir_path: Path, frames: list[np.ndarray], ext: str = "png"
) -> list[Path]:
    """Write *frames* to ``dir_path`` as numbered images and return their paths."""
    paths = []
    for i, frame in enumerate(frames):
        p = dir_path / f"frame{i:04d}.{ext}"
        imageio.imwrite(p, frame)
        paths.append(p)
    return paths


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
    assert meta.n_frames >= 1
    assert meta.frame_size == (32, 32)


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

    # Write fake metadata at the correct cache path (name + suffix, not replacing
    # suffix). A matching (size, mtime) signature makes the cache trusted, so the
    # fake values are used.
    meta_file = out.parent / (out.name + ".metadata.json")
    size, mtime_ns = _file_signature(out)
    fake_meta = {
        "n_frames": 123,
        "frame_size": [32, 32],
        "fps": 12.0,
        "size": size,
        "mtime_ns": mtime_ns,
    }
    with open(meta_file, "w") as f:
        json.dump(fake_meta, f)

    # Should use cached metadata
    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta.n_frames == 123
    assert meta.frame_size == (32, 32)
    assert meta.fps == 12.0


def test_get_video_metadata_corrupted_cache_raises(tmp_path: Path):
    """Test that corrupted cache raises an error."""
    frames = make_simple_frames(n=1, h=32, w=32, channels=3)
    out = tmp_path / "corrupt.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    # Write invalid JSON at the correct cache path
    meta_file = out.parent / (out.name + ".metadata.json")
    with open(meta_file, "w") as f:
        f.write("not a json")

    with pytest.raises(Exception):
        get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)


def test_metadata_cache_path_appends_not_replaces(tmp_path: Path):
    """Bug 4: cache path must append the suffix, not replace the video extension.

    Two videos with the same stem but different extensions must not share a cache file.
    """
    frames = make_simple_frames(n=2, h=32, w=32)
    mp4 = tmp_path / "clip.mp4"
    avi = tmp_path / "clip.avi"
    write_frames_to_video(mp4, frames, fps=5.0)
    write_frames_to_video(avi, frames, fps=5.0)

    # Cache metadata for both
    get_video_metadata(mp4, cache_metadata=True, use_cached_metadata=False)
    get_video_metadata(avi, cache_metadata=True, use_cached_metadata=False)

    # Each video must produce a distinct cache file
    mp4_cache = tmp_path / "clip.mp4.metadata.json"
    avi_cache = tmp_path / "clip.avi.metadata.json"
    assert mp4_cache.exists(), f"Expected cache file {mp4_cache}"
    assert avi_cache.exists(), f"Expected cache file {avi_cache}"
    # The old buggy path (replacing .mp4) must NOT exist
    assert not (tmp_path / "clip.metadata.json").exists()


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


# ---------------------------------------------------------------------------
# Comprehensive tests
# ---------------------------------------------------------------------------


def test_read_frames_all_frames(tmp_path: Path):
    """read_frames_from_video with no indices returns all frames."""
    frames = make_simple_frames(n=8, h=16, w=16)
    out = tmp_path / "all.mp4"
    write_frames_to_video(out, frames, fps=10.0)
    read_back, fps = read_frames_from_video(out)
    assert len(read_back) == 8
    assert fps is not None


def test_read_frames_specific_indices(tmp_path: Path):
    """read_frames_from_video with explicit indices returns only those frames."""
    frames = make_simple_frames(n=10, h=16, w=16)
    out = tmp_path / "sel.mp4"
    write_frames_to_video(out, frames, fps=10.0)
    selected, _ = read_frames_from_video(out, frame_indices=[0, 3, 7])
    assert len(selected) == 3


def _write_ffv1_16bit_rgba(
    path: Path, frames: list[np.ndarray], fps: float = 10.0
) -> None:
    """Write *frames* (uint16 ``(H, W, 4)`` RGBA) as a lossless 16-bit FFV1 MKV.

    Mirrors the kind of high-bit-depth source (FFV1 ``gbrap16le``) that exposed
    the 8-bit-truncation bug; encoded losslessly so the bytes round-trip exactly.
    """
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=int(fps))
        stream.width, stream.height = w, h
        stream.pix_fmt = "gbrap16le"
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgba64le")
            for packet in stream.encode(frame.reformat(format="gbrap16le")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_read_frames_preserves_16bit_native_channels(tmp_path: Path):
    """A 16-bit RGBA source round-trips as uint16 (H, W, 4) with exact values.

    Regression for the bug where imageio decoded high-bit-depth lossless video to
    8-bit RGB, keeping only the high byte so small 16-bit values collapsed to
    0/1/2 and the alpha channel was dropped.
    """
    h, w, n = 8, 12, 4
    src = []
    for i in range(n):
        arr = np.zeros((h, w, 4), dtype=np.uint16)
        arr[..., 0] = 300 + i  # small values: would become 1 under >>8
        arr[..., 1] = 301 + i
        arr[..., 2] = 302 + i
        arr[..., 3] = 303 + i  # alpha carries real data
        src.append(arr)

    out = tmp_path / "raw16.mkv"
    _write_ffv1_16bit_rgba(out, src, fps=10.0)

    # Specific indices (seek path) recover native dtype, channel count, and values.
    selected, fps = read_frames_from_video(out, frame_indices=[0, 2])
    assert fps == 10.0
    assert selected[0].dtype == np.uint16
    assert selected[0].shape == (h, w, 4)
    assert np.array_equal(selected[0], src[0])
    assert np.array_equal(selected[1], src[2])

    # All-frames (sequential path) recovers every frame exactly and in order.
    all_frames, _ = read_frames_from_video(out)
    assert len(all_frames) == n
    assert all(np.array_equal(a, b) for a, b in zip(all_frames, src))


def test_read_frames_out_of_range_raises(tmp_path: Path):
    """Requesting a frame past the end raises IndexError."""
    frames = make_simple_frames(n=4, h=16, w=16)
    out = tmp_path / "short.mp4"
    write_frames_to_video(out, frames, fps=10.0)
    with pytest.raises(IndexError):
        read_frames_from_video(out, frame_indices=[99])


def test_read_frames_indexed_correct_across_access_patterns(tmp_path: Path):
    """Sparse, clustered, and reverse index reads all return the right frames.

    Exercises both branches of the adaptive indexed read (seek to far frames,
    decode forward between near ones) on an all-intra source whose container
    only marks the first packet as a keyframe -- the case that made a naive
    "decode everything" strategy quadratically slow. Each frame is tagged with
    its index in pixel [0, 0] so the mapping can be checked exactly.
    """
    n = 24
    src = []
    for i in range(n):
        arr = np.zeros((8, 12, 4), dtype=np.uint16)
        arr[..., :] = 100 + i  # distinct per-frame value
        arr[0, 0, 0] = 1000 + i  # unambiguous index tag
        src.append(arr)
    out = tmp_path / "intra24.mkv"
    _write_ffv1_16bit_rgba(out, src, fps=10.0)

    for indices in ([0, 11, 23], [5, 6, 7, 8], [23, 0, 12], [9, 9, 0]):
        got, _ = read_frames_from_video(out, frame_indices=indices)
        assert len(got) == len(indices)
        for arr, idx in zip(got, indices):
            assert arr[0, 0, 0] == 1000 + idx, f"wrong frame for index {idx}"
            assert np.array_equal(arr, src[idx])


def test_write_frames_creates_file(tmp_path: Path):
    """write_frames_to_video creates a file at the given path."""
    out = tmp_path / "created.mp4"
    write_frames_to_video(out, make_simple_frames(n=3, h=16, w=16), fps=5.0)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_frames_cpu_mode_custom_quality(tmp_path: Path):
    """write_frames_to_video accepts mode='cpu' with explicit quality/preset."""
    out = tmp_path / "custom.mp4"
    write_frames_to_video(
        out,
        make_simple_frames(n=3, h=16, w=16),
        fps=5.0,
        mode="cpu",
        quality=23,
        preset="ultrafast",
    )
    assert out.is_file()


def test_write_frames_invalid_mode_raises(tmp_path: Path):
    """An unknown mode is rejected."""
    with pytest.raises(ValueError):
        write_frames_to_video(
            tmp_path / "x.mp4", make_simple_frames(n=2, h=16, w=16), fps=5.0, mode="tpu"
        )


def test_get_video_metadata_cache_bypass(tmp_path: Path):
    """use_cached_metadata=False re-reads from disk even if a cache file exists."""
    frames = make_simple_frames(n=3, h=16, w=16)
    out = tmp_path / "bypass.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    # Plant a fake cache with wrong n_frames
    cache = out.parent / (out.name + ".metadata.json")
    with open(cache, "w") as f:
        json.dump({"n_frames": 999, "frame_size": [16, 16], "fps": 5.0}, f)

    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert meta.n_frames != 999


def test_get_video_metadata_no_cache_written(tmp_path: Path):
    """cache_metadata=False must not write a cache file."""
    frames = make_simple_frames(n=2, h=16, w=16)
    out = tmp_path / "nocache.mp4"
    write_frames_to_video(out, frames, fps=5.0)

    get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)

    cache = out.parent / (out.name + ".metadata.json")
    assert not cache.exists()


def test_get_video_metadata_returns_correct_types(tmp_path: Path):
    """get_video_metadata returns dict with correctly-typed values."""
    frames = make_simple_frames(n=4, h=32, w=48)
    out = tmp_path / "types.mp4"
    write_frames_to_video(out, frames, fps=24.0)
    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert isinstance(meta.n_frames, int)
    assert isinstance(meta.frame_size, tuple)
    assert len(meta.frame_size) == 2
    assert meta.fps is None or isinstance(meta.fps, (int, float))


def test_check_num_frames_matches_metadata(tmp_path: Path):
    """check_num_frames and get_video_metadata agree on frame count."""
    n = 7
    frames = make_simple_frames(n=n, h=16, w=16)
    out = tmp_path / "count.mp4"
    write_frames_to_video(out, frames, fps=5.0)
    assert (
        check_num_frames(out)
        == get_video_metadata(
            out, cache_metadata=False, use_cached_metadata=False
        ).n_frames
    )


def test_write_image_paths_and_read_back(tmp_path: Path):
    """write_image_paths_to_video combines on-disk images into a video."""
    frames = make_simple_frames(n=6, h=32, w=32, channels=3)
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    paths = _write_frame_images(img_dir, frames)

    out = tmp_path / "from_paths.mp4"
    write_image_paths_to_video(out, paths, fps=10.0)
    assert out.is_file() and out.stat().st_size > 0

    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert meta.n_frames == 6
    assert meta.frame_size == (32, 32)


def test_write_image_paths_accepts_str_paths(tmp_path: Path):
    """String (not just Path) image paths are accepted."""
    frames = make_simple_frames(n=3, h=16, w=16)
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    paths = [str(p) for p in _write_frame_images(img_dir, frames)]

    out = tmp_path / "str_paths.mp4"
    write_image_paths_to_video(out, paths, fps=5.0, mode="cpu")
    assert out.is_file()


def test_write_image_paths_empty_raises(tmp_path: Path):
    """An empty image path list raises."""
    with pytest.raises(ValueError):
        write_image_paths_to_video(tmp_path / "x.mp4", [], fps=10.0)


def test_write_image_paths_invalid_mode_raises(tmp_path: Path):
    """An unknown mode is rejected before any image is read."""
    with pytest.raises(ValueError):
        write_image_paths_to_video(tmp_path / "x.mp4", ["a.png"], fps=10.0, mode="tpu")


def test_write_image_paths_mismatched_sizes_raises(tmp_path: Path):
    """Images with differing dimensions raise a ValueError."""
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    imageio.imwrite(img_dir / "frame0000.png", np.zeros((16, 16, 3), dtype=np.uint8))
    imageio.imwrite(img_dir / "frame0001.png", np.zeros((20, 20, 3), dtype=np.uint8))
    paths = [img_dir / "frame0000.png", img_dir / "frame0001.png"]

    with pytest.raises(ValueError):
        write_image_paths_to_video(tmp_path / "bad.mp4", paths, fps=5.0)


def test_get_video_metadata_cache_written(tmp_path: Path):
    """cache_metadata=True writes a parseable JSON cache file."""
    frames = make_simple_frames(n=2, h=16, w=16)
    out = tmp_path / "write_cache.mp4"
    write_frames_to_video(out, frames, fps=5.0)
    get_video_metadata(out, cache_metadata=True, use_cached_metadata=False)

    cache = out.parent / (out.name + ".metadata.json")
    assert cache.exists()
    with open(cache) as f:
        data = json.load(f)
    assert "n_frames" in data and "frame_size" in data and "fps" in data
    st = out.stat()
    assert data["size"] == st.st_size and data["mtime_ns"] == st.st_mtime_ns


def test_get_video_metadata_invalidates_cache_on_modification(tmp_path: Path):
    """A changed video (size, mtime) signature invalidates the cache and forces a re-read."""
    out = tmp_path / "modified.mp4"
    write_frames_to_video(out, make_simple_frames(n=3, h=16, w=16), fps=5.0)

    # Prime the cache, then overwrite the video with a different number of frames.
    first = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert first.n_frames == 3
    write_frames_to_video(out, make_simple_frames(n=7, h=16, w=16), fps=5.0)

    # The stale cache must be invalidated and the new frame count returned.
    second = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert second.n_frames == 7

    # The cache file must have been rewritten with the new video's signature.
    cache = out.parent / (out.name + ".metadata.json")
    with open(cache) as f:
        data = json.load(f)
    assert data["n_frames"] == 7
    st = out.stat()
    assert data["size"] == st.st_size and data["mtime_ns"] == st.st_mtime_ns


def test_get_video_metadata_cache_stores_signature(tmp_path: Path):
    """The cache records the cheap (size, mtime_ns) staleness guard."""
    out = tmp_path / "sig.mp4"
    write_frames_to_video(out, make_simple_frames(n=2, h=16, w=16), fps=5.0)
    get_video_metadata(out, cache_metadata=True, use_cached_metadata=False)

    cache = out.parent / (out.name + ".metadata.json")
    with open(cache) as f:
        data = json.load(f)
    st = out.stat()
    assert data["size"] == st.st_size
    assert data["mtime_ns"] == st.st_mtime_ns


def test_get_video_metadata_unchanged_file_skips_reread(tmp_path: Path, monkeypatch):
    """An unchanged (size, mtime) cache hit must not re-probe the video."""
    import pvio.io as pvio_io

    out = tmp_path / "fast.mp4"
    write_frames_to_video(out, make_simple_frames(n=3, h=16, w=16), fps=5.0)
    get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)  # prime

    def _boom(*args, **kwargs):
        raise AssertionError("video must not be re-probed on the stat fast path")

    monkeypatch.setattr(pvio_io, "_probe_video", _boom)
    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta.n_frames == 3


def test_get_video_metadata_touch_reread_stays_correct(tmp_path: Path):
    """A touched video (changed mtime) is re-probed and still returns correct metadata."""
    import os

    out = tmp_path / "touched.mp4"
    write_frames_to_video(out, make_simple_frames(n=5, h=16, w=16), fps=5.0)
    first = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)

    # Bump mtime without changing the bytes (e.g. a copy or `touch`). The signature
    # no longer matches, so the cache is invalidated and the metadata re-read --
    # which is cheap and yields the same (correct) answer.
    st = out.stat()
    os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    second = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert second.n_frames == first.n_frames == 5


def test_get_video_metadata_legacy_cache_without_signature_reread(tmp_path: Path):
    """A cache file lacking a (size, mtime) signature (older format) is re-read, not trusted."""
    out = tmp_path / "legacy.mp4"
    write_frames_to_video(out, make_simple_frames(n=4, h=16, w=16), fps=5.0)

    cache = out.parent / (out.name + ".metadata.json")
    with open(cache, "w") as f:
        json.dump({"n_frames": 999, "frame_size": [16, 16], "fps": 5.0}, f)

    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta.n_frames == 4

    # The cache is upgraded in place to include the (size, mtime) signature.
    with open(cache) as f:
        data = json.load(f)
    st = out.stat()
    assert data["size"] == st.st_size and data["mtime_ns"] == st.st_mtime_ns

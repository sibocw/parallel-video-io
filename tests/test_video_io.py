import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import av

from pvio.video_io import (
    read_frames_from_video,
    write_frames_to_video,
    check_num_frames,
    get_video_metadata,
    _InputVideo,
)

# ---------------------------
# Helpers & fixtures
# ---------------------------


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def tmpdir(tmp_path: Path) -> Path:
    return tmp_path


def _encode_frame_number(frame_idx: int) -> tuple[int, int, int]:
    """
    Encode frame number into RGB values using intervals of 10.
    Range: 0-999 frames (R: 0-9, G: 0-9, B: 0-9 -> RR*100 + GG*10 + BB)
    Using multiples of 10 to be robust against lossy encoding.
    """
    assert 0 <= frame_idx < 1000, "Frame index must be 0-999"
    r = (frame_idx // 100) * 10
    g = ((frame_idx % 100) // 10) * 10
    b = (frame_idx % 10) * 10
    return (r, g, b)


def _decode_frame_number(rgb: tuple[int, int, int]) -> int:
    """Decode frame number from RGB values (with tolerance for lossy encoding)."""
    r = round(rgb[0] / 10) * 10
    g = round(rgb[1] / 10) * 10
    b = round(rgb[2] / 10) * 10
    return (r // 10) * 100 + (g // 10) * 10 + (b // 10)


def _solid_frame(h: int, w: int, rgb):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = np.array(rgb, dtype=np.uint8)
    return frame


def _make_cfr_video(path: Path, n: int = 32, size=(64, 48), fps=24.0):
    """Create a small CFR video where each frame encodes its index in RGB values.
    Size must be divisible by 16 to avoid imageio auto-resizing.
    """
    # Ensure size is divisible by 16
    assert size[0] % 16 == 0 and size[1] % 16 == 0, f"Size {size} must be divisible by 16"
    
    colors = [_encode_frame_number(i) for i in range(n)]
    frames = [_solid_frame(size[1], size[0], c) for c in colors]
    write_frames_to_video(path, frames, fps=fps)
    return frames, fps


def _assert_frame_matches(img: np.ndarray, expected_frame_idx: int, tolerance: int = 5):
    """Assert that a frame matches the expected frame index (with tolerance for lossy encoding)."""
    # Sample a pixel from the center
    h, w = img.shape[:2]
    center_pixel = tuple(img[h // 2, w // 2])
    decoded_idx = _decode_frame_number(center_pixel)
    
    assert decoded_idx == expected_frame_idx, (
        f"Frame mismatch: expected frame {expected_frame_idx}, "
        f"decoded frame {decoded_idx} from pixel {center_pixel}"
    )


# ---------------------------
# Smoke tests for writer
# ---------------------------


def test_write_and_read_all_frames(tmpdir):
    out = tmpdir / "basic.mp4"
    frames, fps = _make_cfr_video(out, n=16, size=(80, 64), fps=30.0)

    imgs, got_fps = read_frames_from_video(out, frame_indices=None, return_fps=True)
    assert len(imgs) == len(frames)
    assert abs(got_fps - fps) < 1e-6

    # spot-check frame indices (first, middle, last)
    for idx in (0, len(frames) // 2, len(frames) - 1):
        _assert_frame_matches(imgs[idx], idx)


def test_writer_empty_raises(tmpdir):
    out = tmpdir / "empty.mp4"
    with pytest.raises(ValueError, match="No frames provided"):
        write_frames_to_video(out, [], fps=24.0)


def test_writer_mismatched_sizes_raises(tmpdir):
    out = tmpdir / "mismatch.mp4"
    f1 = _solid_frame(32, 32, (10, 20, 30))
    f2 = _solid_frame(33, 32, (10, 20, 30))
    with pytest.raises(ValueError, match="All frames must have the same dimensions"):
        write_frames_to_video(out, [f1, f2], fps=24.0)


def test_writer_log_interval(tmpdir, caplog):
    import logging
    
    out = tmpdir / "logged.mp4"
    frames, _ = _make_cfr_video(out, n=10, size=(64, 48), fps=24.0)
    
    # Write again with log interval
    out2 = tmpdir / "logged2.mp4"
    with caplog.at_level(logging.INFO):
        write_frames_to_video(out2, frames, fps=24.0, log_interval=3)
    
    # Check that we got log messages
    assert any("Written frame" in record.message for record in caplog.records)


# ---------------------------
# Read selected frames (seek vs sequential)
# ---------------------------


@pytest.mark.parametrize("threshold", [1, 5, 10])
def test_read_selected_frames(tmpdir, threshold):
    out = tmpdir / f"sel_{threshold}.mp4"
    frames, fps = _make_cfr_video(out, n=40, size=(64, 48), fps=25.0)

    # choose indices with both small and large gaps
    idxs = [0, 1, 2, 7, 8, 9, 20, 21, 22, 35, 39, 39]  # duplicates included
    imgs = read_frames_from_video(
        out, frame_indices=idxs, max_sequential_decodes=threshold
    )
    assert len(imgs) == len(idxs)
    for req, img in zip(idxs, imgs):
        _assert_frame_matches(img, req)


def test_read_selected_frames_unordered(tmpdir):
    """Test that frames are returned in the order requested, not sorted order."""
    out = tmpdir / "unordered.mp4"
    frames, _ = _make_cfr_video(out, n=20, size=(32, 32), fps=24.0)

    # Request frames in non-sorted order
    idxs = [15, 2, 8, 19, 0, 5]
    imgs = read_frames_from_video(out, frame_indices=idxs, max_sequential_decodes=5)
    
    assert len(imgs) == len(idxs)
    for req_idx, img in zip(idxs, imgs):
        _assert_frame_matches(img, req_idx)


def test_read_single_frame(tmpdir):
    out = tmpdir / "single.mp4"
    frames, _ = _make_cfr_video(out, n=10, size=(32, 32), fps=24.0)
    
    imgs = read_frames_from_video(out, frame_indices=[5])
    assert len(imgs) == 1
    _assert_frame_matches(imgs[0], 5)


def test_read_last_frame(tmpdir):
    out = tmpdir / "last.mp4"
    frames, _ = _make_cfr_video(out, n=10, size=(32, 32), fps=24.0)
    
    imgs = read_frames_from_video(out, frame_indices=[9])
    assert len(imgs) == 1
    _assert_frame_matches(imgs[0], 9)


def test_read_duplicate_frames(tmpdir):
    out = tmpdir / "dup.mp4"
    frames, _ = _make_cfr_video(out, n=10, size=(32, 32), fps=24.0)
    
    # Request same frame multiple times
    idxs = [3, 3, 3, 5, 5]
    imgs = read_frames_from_video(out, frame_indices=idxs, max_sequential_decodes=5)
    
    assert len(imgs) == len(idxs)
    for i, req_idx in enumerate(idxs):
        _assert_frame_matches(imgs[i], req_idx)


def test_read_selected_frames_raises_on_negative(tmpdir):
    out = tmpdir / "neg.mp4"
    _make_cfr_video(out, n=8, size=(32, 32), fps=24.0)
    with pytest.raises(IndexError, match="Negative frame index"):
        read_frames_from_video(out, frame_indices=[-1, 0, 1])


def test_read_selected_frames_raises_on_oob(tmpdir):
    out = tmpdir / "oob.mp4"
    _make_cfr_video(out, n=8, size=(32, 32), fps=24.0)
    with pytest.raises(IndexError, match="beyond end of video"):
        read_frames_from_video(out, frame_indices=[0, 7, 8])  # 8 is OOB


def test_read_selected_empty_list_ok(tmpdir, caplog):
    import logging
    
    out = tmpdir / "empty_req.mp4"
    _make_cfr_video(out, n=8, size=(64, 48))
    
    with caplog.at_level(logging.WARNING):
        imgs = read_frames_from_video(out, frame_indices=[], max_sequential_decodes=3)
    
    assert imgs == []
    assert any("zero frames" in record.message.lower() for record in caplog.records)


def test_read_contiguous_sequence(tmpdir):
    """Test reading a contiguous sequence of frames (should use sequential decode)."""
    out = tmpdir / "contiguous.mp4"
    frames, _ = _make_cfr_video(out, n=30, size=(32, 32), fps=24.0)
    
    # Read a contiguous block
    idxs = list(range(10, 20))
    imgs = read_frames_from_video(out, frame_indices=idxs, max_sequential_decodes=10)
    
    assert len(imgs) == len(idxs)
    for i, idx in enumerate(idxs):
        _assert_frame_matches(imgs[i], idx)


def test_read_with_large_gaps(tmpdir):
    """Test reading frames with gaps larger than threshold (should use seek)."""
    out = tmpdir / "gaps.mp4"
    frames, _ = _make_cfr_video(out, n=100, size=(32, 32), fps=24.0)
    
    # Large gaps between requested frames
    idxs = [0, 50, 99]
    imgs = read_frames_from_video(out, frame_indices=idxs, max_sequential_decodes=10)
    
    assert len(imgs) == len(idxs)
    for i, idx in enumerate(idxs):
        _assert_frame_matches(imgs[i], idx)


# ---------------------------
# _InputVideo invariants & integer timeline
# ---------------------------


def test_inputvideo_invariants_and_mappings(tmpdir):
    out = tmpdir / "inv.mp4"
    frames, fps = _make_cfr_video(out, n=12, size=(64, 48), fps=30.0)

    with _InputVideo(out) as vid:
        # zero/one stream, integer ticks enforced by constructor
        assert vid.n_frames == 12
        assert math.isclose(vid.get_fps(), fps, rel_tol=0, abs_tol=1e-6)
        assert isinstance(vid.ticks_per_frame, int) and vid.ticks_per_frame > 0

        # frame<->pts mapping roundtrip on a few points
        for i in (0, 1, 5, 11):
            pts = vid.frame_idx_to_pts(i)
            assert vid.pts_to_frame_idx(pts) == i

        # precise seek returns the correct pixels
        img0 = vid._exact_seek_and_read(0).to_ndarray(format="rgb24")
        img5 = vid._exact_seek_and_read(5).to_ndarray(format="rgb24")
        _assert_frame_matches(img0, 0)
        _assert_frame_matches(img5, 5)


def test_inputvideo_get_fps_returns_fraction(tmpdir):
    out = tmpdir / "frac.mp4"
    _make_cfr_video(out, n=8, size=(64, 48), fps=24.0)
    
    with _InputVideo(out) as vid:
        fps_float = vid.get_fps(as_float=True)
        fps_frac = vid.get_fps(as_float=False)
        
        assert isinstance(fps_float, float)
        assert isinstance(fps_frac, Fraction)
        assert abs(float(fps_frac) - fps_float) < 1e-6


def test_inputvideo_context_manager_closes(tmpdir):
    out = tmpdir / "ctx.mp4"
    _make_cfr_video(out, n=8, size=(64, 48), fps=24.0)
    
    vid = _InputVideo(out)
    assert not vid._closed
    
    vid.close()
    assert vid._closed
    
    # Should be idempotent
    vid.close()
    assert vid._closed


def test_inputvideo_frame_idx_to_pts_negative_raises(tmpdir):
    out = tmpdir / "neg_pts.mp4"
    _make_cfr_video(out, n=8, size=(64, 48), fps=24.0)
    
    with _InputVideo(out) as vid:
        with pytest.raises(IndexError, match="Negative frame index"):
            vid.frame_idx_to_pts(-1)


def test_inputvideo_pts_to_frame_idx_negative_raises(tmpdir):
    out = tmpdir / "neg_idx.mp4"
    _make_cfr_video(out, n=8, size=(64, 48), fps=24.0)
    
    with _InputVideo(out) as vid:
        with pytest.raises(IndexError, match="PTS before stream start"):
            vid.pts_to_frame_idx(-1)


def test_inputvideo_pts_to_frame_idx_unaligned_raises(tmpdir):
    out = tmpdir / "unaligned.mp4"
    _make_cfr_video(out, n=8, size=(64, 48), fps=24.0)
    
    with _InputVideo(out) as vid:
        # Get a valid PTS and add 1 to make it unaligned
        valid_pts = vid.frame_idx_to_pts(5)
        with pytest.raises(ValueError, match="not aligned to frame grid"):
            vid.pts_to_frame_idx(valid_pts + 1)


# ---------------------------
# Metadata helpers
# ---------------------------


def test_check_num_frames_matches(tmpdir):
    out = tmpdir / "meta.mp4"
    frames, _ = _make_cfr_video(out, n=15, size=(64, 48))
    n = check_num_frames(out)
    assert n == len(frames)


def test_check_num_frames_nonexistent_raises(tmpdir):
    out = tmpdir / "nonexistent.mp4"
    with pytest.raises(RuntimeError, match="Failed to open video file"):
        check_num_frames(out)


def test_get_video_metadata_basic(tmpdir):
    out = tmpdir / "meta_basic.mp4"
    frames, fps = _make_cfr_video(out, n=9, size=(64, 48), fps=23.0)
    
    meta = get_video_metadata(out, cache_metadata=False)
    assert meta["n_frames"] == 9
    assert meta["frame_size"] == (48, 64)  # (height, width)
    assert abs(meta["fps"] - 23.0) < 1e-6


def test_get_video_metadata_caches(tmpdir):
    out = tmpdir / "cache.mp4"
    frames, fps = _make_cfr_video(out, n=9, size=(64, 48), fps=23.0)
    
    # First call without using cache
    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=False)
    assert meta["n_frames"] == 9
    assert meta["frame_size"] == (48, 64)
    assert abs(meta["fps"] - 23.0) < 1e-6

    # Cache file should exist
    cache_path = out.with_suffix(".metadata.json")
    assert cache_path.exists()
    
    # Mutate cache to ensure it is read
    fake = {"n_frames": 123, "frame_size": [1, 2], "fps": 99.0}
    cache_path.write_text(json.dumps(fake))
    
    # Second call should use cache
    meta2 = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta2 == {"n_frames": 123, "frame_size": (1, 2), "fps": 99.0}


def test_get_video_metadata_corrupted_cache_raises(tmpdir):
    out = tmpdir / "corrupt.mp4"
    _make_cfr_video(out, n=5, size=(64, 48), fps=24.0)
    
    # Create corrupted cache
    cache_path = out.with_suffix(".metadata.json")
    cache_path.write_text("not valid json {{{")
    
    with pytest.raises(Exception):  # JSONDecodeError
        get_video_metadata(out, cache_metadata=False, use_cached_metadata=True)


def test_get_video_metadata_custom_suffix(tmpdir):
    out = tmpdir / "custom_suffix.mp4"
    _make_cfr_video(out, n=5, size=(64, 48), fps=24.0)
    
    custom_suffix = ".my_metadata.json"
    meta = get_video_metadata(
        out, 
        cache_metadata=True, 
        use_cached_metadata=False,
        metadata_suffix=custom_suffix
    )
    
    cache_path = out.with_suffix(custom_suffix)
    assert cache_path.exists()


# ---------------------------
# Error cases: constructor validation
# ---------------------------


def test_init_raises_if_multiple_streams(tmpdir):
    """Add a dummy audio stream and ensure _InputVideo refuses."""
    out = tmpdir / "multi.mkv"
    # Create container with video + audio
    container = av.open(out.as_posix(), mode="w")
    v = container.add_stream("libx264", rate=24)
    v.width, v.height, v.pix_fmt = 32, 32, "yuv420p"
    a = container.add_stream("aac")  # dummy audio stream
    
    # Encode one black frame
    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
    for pkt in v.encode(frame):
        container.mux(pkt)
    for pkt in v.encode():
        container.mux(pkt)
    container.close()

    with pytest.raises(ValueError, match="Expected exactly one stream"):
        _InputVideo(out)


def test_init_raises_if_no_streams(tmpdir):
    """Ensure _InputVideo refuses a container with no streams."""
    out = tmpdir / "no_streams.mkv"
    container = av.open(out.as_posix(), mode="w")
    # Add a dummy stream so the file is valid, then we'll test with an invalid file
    container.close()
    
    # The file won't exist or will be invalid
    # Instead, create a simple text file that's not a valid video
    out.write_text("not a video file")
    
    with pytest.raises((ValueError, av.error.InvalidDataError)):
        _InputVideo(out)


# ---------------------------
# Error cases: runtime decode issues
# ---------------------------


def test_exact_seek_frame_not_found(tmpdir):
    """Test that seeking beyond the video length raises appropriate error."""
    out = tmpdir / "short.mp4"
    _make_cfr_video(out, n=5, size=(64, 48), fps=24.0)
    
    with _InputVideo(out) as vid:
        # Try to seek to frame beyond the end
        with pytest.raises(IndexError, match="not found before EOF"):
            vid._exact_seek_and_read(100)


def test_read_frames_first_frame_missing(tmpdir):
    """Test error when first requested frame doesn't exist."""
    out = tmpdir / "missing_first.mp4"
    _make_cfr_video(out, n=5, size=(64, 48), fps=24.0)
    
    with pytest.raises(IndexError, match="beyond end of video"):
        read_frames_from_video(out, frame_indices=[10])


def test_read_frames_middle_frame_missing(tmpdir):
    """Test error when a middle requested frame doesn't exist."""
    out = tmpdir / "missing_middle.mp4"
    _make_cfr_video(out, n=10, size=(64, 48), fps=24.0)
    
    with pytest.raises(IndexError, match="beyond end of video"):
        read_frames_from_video(out, frame_indices=[2, 15, 5])


# ---------------------------
# Edge cases and special scenarios
# ---------------------------


def test_read_all_frames_with_explicit_indices(tmpdir):
    """Test reading all frames by explicitly listing all indices."""
    out = tmpdir / "all_explicit.mp4"
    frames, _ = _make_cfr_video(out, n=10, size=(64, 48), fps=24.0)
    
    all_indices = list(range(10))
    imgs = read_frames_from_video(out, frame_indices=all_indices)
    
    assert len(imgs) == len(frames)
    for i in range(len(imgs)):
        _assert_frame_matches(imgs[i], i)


def test_different_frame_sizes(tmpdir):
    """Test videos with different frame dimensions (all divisible by 16)."""
    sizes = [(32, 32), (64, 48), (128, 96), (320, 240)]
    
    for size in sizes:
        out = tmpdir / f"size_{size[0]}x{size[1]}.mp4"
        frames, _ = _make_cfr_video(out, n=5, size=size, fps=24.0)
        
        imgs = read_frames_from_video(out)
        assert len(imgs) == len(frames)
        assert imgs[0].shape[:2] == (size[1], size[0])


def test_different_fps_values(tmpdir):
    """Test videos with different FPS values."""
    fps_values = [15.0, 24.0, 30.0, 60.0]
    
    for fps in fps_values:
        out = tmpdir / f"fps_{fps}.mp4"
        frames, _ = _make_cfr_video(out, n=10, size=(64, 48), fps=fps)
        
        imgs, got_fps = read_frames_from_video(out, return_fps=True)
        assert len(imgs) == len(frames)
        assert abs(got_fps - fps) < 0.1  # Allow small tolerance


def test_very_short_video(tmpdir):
    """Test handling of single-frame video."""
    out = tmpdir / "single_frame.mp4"
    frames, _ = _make_cfr_video(out, n=1, size=(64, 48), fps=24.0)
    
    imgs = read_frames_from_video(out)
    assert len(imgs) == 1
    _assert_frame_matches(imgs[0], 0)


def test_pathlib_and_str_paths(tmpdir):
    """Test that both Path and str arguments work."""
    out = tmpdir / "path_test.mp4"
    frames, _ = _make_cfr_video(out, n=5, size=(64, 48), fps=24.0)
    
    # Test with Path
    imgs1 = read_frames_from_video(out)
    
    # Test with str
    imgs2 = read_frames_from_video(str(out))
    
    assert len(imgs1) == len(imgs2) == len(frames)
    for i in range(len(frames)):
        _assert_frame_matches(imgs1[i], i)
        _assert_frame_matches(imgs2[i], i)


def test_return_fps_flag(tmpdir):
    """Test the return_fps parameter works correctly."""
    out = tmpdir / "return_fps.mp4"
    _make_cfr_video(out, n=5, size=(64, 48), fps=25.0)
    
    # Without return_fps
    result1 = read_frames_from_video(out, return_fps=False)
    assert isinstance(result1, list)
    
    # With return_fps
    result2 = read_frames_from_video(out, return_fps=True)
    assert isinstance(result2, tuple)
    assert len(result2) == 2
    frames, fps = result2
    assert isinstance(frames, list)
    assert isinstance(fps, float)
    
    
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test_read_all_frames_with_explicit_indices(Path())
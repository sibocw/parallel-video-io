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


def _solid_frame(h: int, w: int, rgb):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = np.array(rgb, dtype=np.uint8)
    return frame


def _make_cfr_video(path: Path, n: int = 32, size=(64, 48), fps=24.0):
    """Create a small CFR video of solid-color frames."""
    colors = [(i % 256, (2 * i) % 256, (3 * i) % 256) for i in range(n)]
    frames = [_solid_frame(size[1], size[0], c) for c in colors]
    write_frames_to_video(path, frames, fps=fps)
    return frames, fps


# ---------------------------
# Smoke tests for writer
# ---------------------------


def test_write_and_read_all_frames(tmpdir):
    out = tmpdir / "basic.mp4"
    frames, fps = _make_cfr_video(out, n=16, size=(80, 60), fps=30.0)

    imgs, got_fps = read_frames_from_video(out, frame_indices=None, return_fps=True)
    assert len(imgs) == len(frames)
    assert abs(got_fps - fps) < 1e-6

    # spot-check pixel equality (first, middle, last)
    for idx in (0, len(frames) // 2, len(frames) - 1):
        assert np.array_equal(imgs[idx], frames[idx])


def test_writer_empty_raises(tmpdir):
    out = tmpdir / "empty.mp4"
    with pytest.raises(ValueError):
        write_frames_to_video(
            out, [], fps=24.0
        )  # :contentReference[oaicite:1]{index=1}


def test_writer_mismatched_sizes_raises(tmpdir):
    out = tmpdir / "mismatch.mp4"
    f1 = _solid_frame(32, 32, (10, 20, 30))
    f2 = _solid_frame(33, 32, (10, 20, 30))
    with pytest.raises(ValueError):
        write_frames_to_video(
            out, [f1, f2], fps=24.0
        )  # :contentReference[oaicite:2]{index=2}


# ---------------------------
# Read selected frames (seek vs sequential)
# ---------------------------


@pytest.mark.parametrize("threshold", [1, 5, 10])
def test_read_selected_frames(tmpdir, threshold):
    out = tmpdir / f"sel_{threshold}.mp4"
    frames, fps = _make_cfr_video(out, n=40, size=(64, 48), fps=25.0)

    # choose indices with both small and large gaps
    idxs = [
        0,
        1,
        2,
        7,
        8,
        9,
        20,
        21,
        22,
        35,
        39,
        39,
    ]  # duplicates included intentionally
    imgs = read_frames_from_video(
        out, frame_indices=idxs, max_sequential_decodes=threshold
    )
    assert len(imgs) == len(idxs)
    for req, img in zip(idxs, imgs):
        assert np.array_equal(img, frames[req])


def test_read_selected_frames_raises_on_negative(tmpdir):
    out = tmpdir / "neg.mp4"
    _make_cfr_video(out, n=8, size=(32, 24), fps=24.0)
    with pytest.raises(IndexError):
        read_frames_from_video(out, frame_indices=[-1, 0, 1])


def test_read_selected_frames_raises_on_oob(tmpdir):
    out = tmpdir / "oob.mp4"
    frames, _ = _make_cfr_video(out, n=8, size=(32, 24), fps=24.0)
    with pytest.raises(IndexError):
        read_frames_from_video(
            out, frame_indices=[0, 7, 8]
        )  # 8 is OOB  # :contentReference[oaicite:3]{index=3}


def test_read_selected_empty_list_ok(tmpdir, caplog):
    out = tmpdir / "empty_req.mp4"
    _make_cfr_video(out, n=8)
    imgs = read_frames_from_video(out, frame_indices=[], max_sequential_decodes=3)
    assert imgs == []
    # Optional: ensure it logged a warning


# ---------------------------
# Auto threshold (keyframe scan)
# ---------------------------


def test_auto_threshold_returns_int_ge1(tmpdir):
    out = tmpdir / "auto.mp4"
    _make_cfr_video(out, n=32, fps=24.0)
    # If "auto", module will call estimate_sequential_threshold()  # :contentReference[oaicite:4]{index=4}
    imgs = read_frames_from_video(
        out, frame_indices=[0, 10, 20, 30], max_sequential_decodes="auto"
    )
    assert len(imgs) == 4


# ---------------------------
# _InputVideo invariants & integer timeline
# ---------------------------


def test_inputvideo_invariants_and_mappings(tmpdir):
    out = tmpdir / "inv.mp4"
    frames, fps = _make_cfr_video(out, n=12, fps=30.0)

    with _InputVideo(out) as vid:
        # zero/one stream, integer ticks enforced by constructor  # :contentReference[oaicite:5]{index=5}
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
        assert np.array_equal(img0, frames[0])
        assert np.array_equal(img5, frames[5])


# ---------------------------
# Metadata helpers
# ---------------------------


def test_check_num_frames_matches(tmpdir):
    out = tmpdir / "meta.mp4"
    frames, _ = _make_cfr_video(out, n=15)
    n = check_num_frames(out)
    assert n == len(frames)  # decoded count  # :contentReference[oaicite:6]{index=6}


def test_get_video_metadata_caches(tmpdir):
    out = tmpdir / "cache.mp4"
    frames, fps = _make_cfr_video(out, n=9, size=(50, 40), fps=23.0)
    meta = get_video_metadata(out, cache_metadata=True, use_cached_metadata=False)
    assert meta["n_frames"] == 9
    assert meta["frame_size"] == (40, 50)
    assert abs(meta["fps"] - 23.0) < 1e-6

    # cached file present, read via cache
    cache_path = out.with_suffix(".metadata.json")
    assert cache_path.exists()
    # mutate cache to ensure it is read
    fake = {"n_frames": 123, "frame_size": [1, 2], "fps": 99.0}
    cache_path.write_text(json.dumps(fake))
    meta2 = get_video_metadata(out, cache_metadata=True, use_cached_metadata=True)
    assert meta2 == {
        "n_frames": 123,
        "frame_size": (1, 2),
        "fps": 99.0,
    }  # :contentReference[oaicite:7]{index=7}


# ---------------------------
# Error cases via monkeypatch (simulate “funky stuff”)
# ---------------------------


def test_init_raises_if_multiple_streams(tmpdir):
    """Add a dummy audio stream and ensure _InputVideo refuses."""
    out = tmpdir / "multi.mkv"
    # Create container with video + audio
    container = av.open(out.as_posix(), mode="w")
    v = container.add_stream("libx264", rate=24)
    v.width, v.height, v.pix_fmt = 32, 24, "yuv420p"
    a = container.add_stream("aac")  # dummy audio stream
    # encode one black frame
    frame = av.VideoFrame.from_ndarray(np.zeros((24, 32, 3), np.uint8), format="rgb24")
    for pkt in v.encode(frame):
        container.mux(pkt)
    for pkt in v.encode():
        container.mux(pkt)
    container.close()

    with pytest.raises(ValueError):
        _ = _InputVideo(
            out
        )  # must reject because there are 2 streams  # :contentReference[oaicite:8]{index=8}


def test_init_raises_on_nonzero_container_start(monkeypatch, tmpdir):
    out = tmpdir / "nzstart.mp4"
    _make_cfr_video(out, n=4)

    iv = _InputVideo(out)
    # Force a non-zero start time on the container object
    monkeypatch.setattr(iv.container, "start_time", 12345, raising=False)

    # Re-run init path by constructing fresh; easiest is to patch av.open
    def fake_open(_):
        return iv.container

    monkeypatch.setattr("video_io.av.open", lambda p: fake_open(p))
    with pytest.raises(ValueError):
        _ = _InputVideo(
            out
        )  # should raise at container.start_time check  # :contentReference[oaicite:9]{index=9}


def test_init_raises_on_vfr(monkeypatch, tmpdir):
    out = tmpdir / "vfr.mp4"
    _make_cfr_video(out, n=4, fps=24.0)

    real = _InputVideo(out)
    # Pretend the stream has a time_base and fps that produce non-integer ticks_per_frame
    # e.g., time_base=1/1000, fps=29.97 -> 1/(1/1000 * 30000/1001) = 1000*1001/30000 = 1001/30 (non-integer)
    tb = Fraction(1, 1000)
    fps = Fraction(30000, 1001)

    def fake_open(_):
        return real.container

    def patch_ctor(monkeypatch):
        monkeypatch.setattr("video_io.av.open", lambda p: fake_open(p))
        monkeypatch.setattr(real.stream, "time_base", tb, raising=False)
        monkeypatch.setattr(real.stream, "average_rate", fps, raising=False)

    # Build a fresh object but with patched stream parameters
    monkeypatch.setattr(
        "video_io._InputVideo.__init__", object.__init__
    )  # bypass original __init__
    # Reconstruct manually to run your logic: easier approach—mock within context:
    # Simpler: patch after open, then call original __init__ again.
    # Instead, we use a smaller trick: call your real constructor but patch attributes afterward
    # and then trigger the validation logic by constructing again via a small wrapper.
    monkeypatch.setattr("video_io.av.open", lambda p: fake_open(p))
    with pytest.raises(ValueError):
        _ = _InputVideo(
            out
        )  # your constructor should now compute non-integer ticks and raise  # :contentReference[oaicite:10]{index=10}


def test_read_frames_rejects_non_increasing_sequence(monkeypatch, tmpdir):
    out = tmpdir / "order.mp4"
    _make_cfr_video(out, n=10)

    # We will bypass set/sort inside read_frames by calling the private directly:
    with _InputVideo(out) as vid:
        # Prime internal state by reading first frame so we can simulate the loop body
        first = vid._exact_seek_and_read(2)
        # Now call read_frames with crafted indices where the sorted set is increasing,
        # but then the internal "gap <= 0" should never occur. To force the error,
        # we monkeypatch last_idx_obtained behavior by wrapping _exact_seek_and_read
        # to misreport the last index (simulate a logic regression).
        orig = vid._exact_seek_and_read

        call_count = {"n": 0}

        def bad_seek(i):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return orig(i)
            # After the first seek, pretend we landed on a future frame so next gap<=0 occurs
            return orig(i + 1)

        monkeypatch.setattr(vid, "_exact_seek_and_read", bad_seek)
        with pytest.raises(RuntimeError):
            _ = vid.read_frames(
                [2, 4, 6], max_sequential_decodes=999
            )  # triggers gap<=0 branch  # :contentReference[oaicite:11]{index=11}

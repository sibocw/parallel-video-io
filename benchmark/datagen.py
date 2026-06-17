"""Synthesise reproducible test videos with ffmpeg.

Each frame carries a bright vertical bar whose horizontal position encodes the
frame index. The bar survives lossy compression well, so a decoded frame can be
matched back to its index (see :func:`recover_index` / :func:`expected_column`),
which is how the random-access benchmark checks *seek correctness*. A fixed
textured background plus mild per-frame noise gives the encoder realistic work
so decode speed and file sizes are representative rather than degenerate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import DATA_DIR, VideoSpec

_BAR_WIDTH = 8
_BG_VALUE = 40
_BAR_VALUE = 235

_ENCODER = {"h264": "libx264", "hevc": "libx265"}


def _bar_step(spec: VideoSpec) -> float:
    """Pixels of horizontal travel per frame for the index-encoding bar."""
    usable = spec.width - _BAR_WIDTH
    return usable / max(1, spec.n_frames - 1)


def expected_column(spec: VideoSpec, index: int) -> float:
    """Centre column where frame ``index``'s bar should appear."""
    return index * _bar_step(spec) + _BAR_WIDTH / 2.0


def recover_index(spec: VideoSpec, frame_hwc: np.ndarray) -> float:
    """Recover the (fractional) frame index from a decoded frame's bar position."""
    gray = frame_hwc.astype(np.float32).mean(axis=2)  # H, W
    col_profile = gray.mean(axis=0)  # W
    centre = float(np.argmax(col_profile))
    return (centre - _BAR_WIDTH / 2.0) / _bar_step(spec)


def _make_frame(spec: VideoSpec, index: int, background: np.ndarray) -> np.ndarray:
    frame = background.copy()
    # Per-frame noise so inter-frame deltas are non-trivial for the encoder.
    rng = np.random.default_rng(1000 + index)
    noise = rng.integers(-6, 7, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Moving index bar.
    x0 = int(round(index * _bar_step(spec)))
    frame[:, x0 : x0 + _BAR_WIDTH, :] = _BAR_VALUE
    return frame


def _background(spec: VideoSpec) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(spec.name)) % (2**32))
    # Low-frequency texture: upsample a small random field so it compresses into
    # the keyframe but still has spatial detail.
    small = rng.integers(0, 80, size=(spec.height // 16 + 1, spec.width // 16 + 1, 3))
    small = small.astype(np.uint8)
    bg = np.kron(small, np.ones((16, 16, 1), dtype=np.uint8))[
        : spec.height, : spec.width, :
    ]
    return np.clip(bg.astype(np.int16) + _BG_VALUE, 0, 255).astype(np.uint8)


def _ffmpeg_encode(spec: VideoSpec, frames_iter) -> None:
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{spec.width}x{spec.height}",
        "-r",
        str(spec.fps),
        "-i",
        "-",
        "-c:v",
        _ENCODER[spec.codec],
        "-g",
        str(spec.gop),
        "-keyint_min",
        str(spec.gop),
        "-crf",
        str(spec.crf),
        "-pix_fmt",
        "yuv420p",
        str(spec.path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {spec.name}")


def generate_video(spec: VideoSpec, *, force: bool = False) -> Path:
    """Create ``spec``'s mp4 if missing. Returns its path."""
    if spec.path.exists() and not force:
        return spec.path
    bg = _background(spec)
    _ffmpeg_encode(spec, (_make_frame(spec, i, bg) for i in range(spec.n_frames)))
    return spec.path


def generate_frames_dir(spec: VideoSpec, *, force: bool = False) -> Path:
    """Dump frames as PNGs into ``spec.frames_dir`` (for ImageDir-style backends)."""
    import imageio.v2 as imageio

    out = spec.frames_dir
    if out.exists() and any(out.iterdir()) and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    bg = _background(spec)
    for i in range(spec.n_frames):
        imageio.imwrite(out / f"frame_{i:05d}.png", _make_frame(spec, i, bg))
    return out


def ensure_videos(specs: list[VideoSpec]) -> None:
    for spec in specs:
        generate_video(spec)


def raw_frames(spec: VideoSpec) -> np.ndarray:
    """Return the ground-truth frames (N, H, W, 3) uint8 for write benchmarks."""
    bg = _background(spec)
    return np.stack([_make_frame(spec, i, bg) for i in range(spec.n_frames)])


if __name__ == "__main__":
    from .config import READ_VIDEOS, collection_videos

    print(f"Generating into {DATA_DIR}")
    for s in READ_VIDEOS + collection_videos():
        p = generate_video(s)
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")

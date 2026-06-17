"""Task 4: write NumPy frames -> MP4. Encode speed, file size, and quality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import config, datagen
from .backends.write import WRITE_BACKENDS
from .common import Result, best_of, timer

# Dedicated (small) specs for writing so we can hold the raw frames in memory
# and decode them back for quality scoring. Dimensions are multiples of 16:
# PVIO's imageio/FFmpeg writer pads to the macroblock size (16), which would
# otherwise change the frame size and make a like-for-like quality comparison
# impossible. (That padding behaviour is itself worth knowing — see README.)
WRITE_SPECS = [
    config.VideoSpec("write_sd", 480, 848, 150, config.FPS, "h264", gop=30),
    config.VideoSpec("write_hd", 720, 1280, 150, config.FPS, "h264", gop=30),
]
_CRF = 20
_N_QUALITY_FRAMES = 16  # evenly spaced frames scored for PSNR/SSIM


def _decode_back(path: str, indices: list[int]) -> np.ndarray:
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(path, seek_mode="exact")
    batch = dec.get_frames_at(indices).data  # NCHW uint8
    return batch.permute(0, 2, 3, 1).cpu().numpy()


def _quality(source: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # Crop to common dimensions in case a writer changed the frame size
    # (e.g. macroblock padding).
    h = min(source.shape[1], decoded.shape[1])
    w = min(source.shape[2], decoded.shape[2])
    source, decoded = source[:, :h, :w], decoded[:, :h, :w]
    psnrs, ssims = [], []
    for s, d in zip(source, decoded):
        psnrs.append(peak_signal_noise_ratio(s, d, data_range=255))
        ssims.append(structural_similarity(s, d, channel_axis=2, data_range=255))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def _bench_one(backend, spec, frames) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            "write",
            backend.name,
            "cpu",
            spec.name,
            float("nan"),
            "frames/s",
            error=reason,
        )
    fps = spec.fps
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.mp4")

            def run() -> float:
                with timer() as t:
                    backend.write(frames, fps, out, _CRF)
                return len(frames) / t[0]

            enc_fps = best_of(run, config.N_REPEATS)
            size_mb = Path(out).stat().st_size / 1e6

            idx = (
                np.linspace(0, len(frames) - 1, _N_QUALITY_FRAMES).astype(int).tolist()
            )
            decoded = _decode_back(out, idx)
            psnr, ssim = _quality(frames[idx], decoded)

        return Result(
            "write",
            backend.name,
            "cpu",
            spec.name,
            enc_fps,
            "frames/s",
            extra={
                "file_size_mb": round(size_mb, 2),
                "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4),
                "crf_controlled": backend.crf_controlled,
                "resolution": f"{spec.height}x{spec.width}",
            },
        )
    except Exception as e:
        return Result(
            "write",
            backend.name,
            "cpu",
            spec.name,
            float("nan"),
            "frames/s",
            error=f"{type(e).__name__}: {e}",
        )


def run() -> list[Result]:
    results: list[Result] = []
    for spec in WRITE_SPECS:
        frames = datagen.raw_frames(spec)
        for backend in WRITE_BACKENDS:
            print(f"  [write] {spec.name:12s} {backend.name}")
            results.append(_bench_one(backend, spec, frames))
    return results


if __name__ == "__main__":
    for r in run():
        if r.error:
            print(f"{r.backend:10s} {r.workload:10s} SKIP/ERR: {r.error}")
        else:
            print(
                f"{r.backend:10s} {r.workload:10s} {r.metric_main:7.1f} fps  {r.extra}"
            )

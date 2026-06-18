"""Task 1: encoding — merge NumPy frames into an H.264 MP4.

Metrics: encode throughput (frames/s) and compression ratio, defined as
(sum of per-frame JPEG bytes) / (encoded video bytes) — i.e. how much smaller
the video is than a folder of per-frame JPEGs at the same quality. PSNR/SSIM
are recorded for every point so the speed/size numbers can be read at matched
image quality.

A single result set is produced: a quality sweep over each encoder
(``encode_pareto``) that traces its speed-vs-compression frontier. Encoders are
*not* compared at a shared quality number — CRF (libx264) and QP (NVENC) are
different operating points — so the runner derives a matched-PSNR comparison
from this sweep instead (see ``run_all`` / ``plots``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import config
from .backends.encode import ENCODE_BACKENDS
from .common import (
    Result,
    best_of,
    decode_back,
    jpeg_baseline_bytes,
    quality_psnr_ssim,
    timer,
)
from .datagen import raw_frames

_N_QUALITY_FRAMES = 16  # evenly spaced frames scored for PSNR/SSIM


def _bench_point(backend, spec, frames, quality, jpeg_bytes, task) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            task,
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "frames/s",
            extra={"quality_param": quality},
            error=reason,
        )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.mp4")

            def run() -> float:
                with timer() as t:
                    backend.encode(frames, spec.fps, out, quality)
                return len(frames) / t[0]

            enc_fps = best_of(run, config.N_REPEATS)
            size_bytes = Path(out).stat().st_size
            compression_ratio = jpeg_bytes / size_bytes if size_bytes else float("nan")

            idx = (
                np.linspace(0, len(frames) - 1, _N_QUALITY_FRAMES).astype(int).tolist()
            )
            psnr, ssim = quality_psnr_ssim(frames[idx], decode_back(out, idx))

        return Result(
            task,
            backend.name,
            backend.device,
            spec.name,
            enc_fps,
            "frames/s",
            extra={
                "quality_param": backend.quality_display(quality)
                if backend.tunable
                else None,
                "file_size_mb": round(size_bytes / 1e6, 2),
                "compression_ratio": round(compression_ratio, 1),
                "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4),
                "resolution": f"{spec.height}x{spec.width}",
            },
        )
    except Exception as e:
        return Result(
            task,
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "frames/s",
            extra={"quality_param": quality},
            error=f"{type(e).__name__}: {e}",
        )


def run() -> list[Result]:
    results: list[Result] = []
    for spec in config.ENCODE_SPECS:
        frames = raw_frames(spec)
        jpeg_bytes = jpeg_baseline_bytes(frames, config.JPEG_QUALITY)

        # Quality sweep -> speed-vs-compression frontier per encoder. A matched-
        # PSNR comparison is derived from these points downstream.
        for backend in ENCODE_BACKENDS:
            if not backend.tunable:
                continue
            for q in config.quality_sweep():
                print(f"  [pareto] {spec.name:8s} {backend.name:10s} q={q}")
                results.append(
                    _bench_point(backend, spec, frames, q, jpeg_bytes, "encode_pareto")
                )
    return results


if __name__ == "__main__":
    for r in run():
        if r.error:
            print(f"{r.task:13s} {r.backend:12s} {r.workload:8s} SKIP/ERR: {r.error}")
        else:
            print(
                f"{r.task:13s} {r.backend:12s} {r.workload:8s} {r.metric_main:7.1f} fps {r.extra}"
            )

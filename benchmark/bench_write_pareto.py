"""Speed-vs-compression Pareto sweep for the tunable PVIO encoders.

The plain write benchmark (``bench_write``) reports one operating point per
backend. That shows *where* each encoder sits but not its frontier: encode
throughput, file size, and quality all move together as you turn the quality
knob. Here we sweep that knob — CRF for CPU libx264, constant-QP for GPU NVENC —
so each method traces a curve in (compression ratio, throughput) space.

Plotting throughput against compression ratio gives the Pareto front: up and to
the right is better (faster *and* smaller). Because the two codecs reach
different quality at the same compression, each point also carries its PSNR/SSIM
so the comparison stays honest (compare at matched quality, not matched knob).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import config
from .bench_write import _N_QUALITY_FRAMES, _decode_back, _quality
from .common import Result, best_of, timer
from .datagen import raw_frames

# Quality levels swept per method. libx264 CRF and NVENC constant-QP share the
# 0-51 H.264 quantiser scale (lower = higher quality / larger file), so the same
# values give comparable — though not identical — quality across the two.
_QUALITY_SWEEP = [16, 20, 24, 28, 34]


def _methods():
    """(label, codec, params-builder, available?) for each tunable encoder."""
    from pvio import _accel

    def libx264_params(q: int) -> list[str]:
        return ["-crf", str(q), "-preset", "slow", "-profile:v", "high"]

    def nvenc_params(q: int) -> list[str]:
        return [
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "constqp",
            "-qp",
            str(q),
            "-profile:v",
            "high",
        ]

    methods = [("pvio_libx264", "libx264", libx264_params, True, "")]
    nvenc_ok = _accel.cuda_available() and _accel.nvenc_ffmpeg_exe() is not None
    reason = "" if nvenc_ok else "no CUDA + NVENC-capable ffmpeg"
    methods.append(("pvio_nvenc", "h264_nvenc", nvenc_params, nvenc_ok, reason))
    return methods


def _bench_point(label, codec, params, spec, frames, quality) -> Result:
    from pvio.io import write_frames_to_video

    fps = spec.fps
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "out.mp4")

            def run() -> float:
                with timer() as t:
                    write_frames_to_video(
                        out, list(frames), fps=fps, codec=codec, ffmpeg_params=params
                    )
                return len(frames) / t[0]

            enc_fps = best_of(run, config.N_REPEATS)
            size_bytes = Path(out).stat().st_size
            compression_ratio = int(frames.size) / size_bytes if size_bytes else np.nan

            idx = (
                np.linspace(0, len(frames) - 1, _N_QUALITY_FRAMES).astype(int).tolist()
            )
            psnr, ssim = _quality(frames[idx], _decode_back(out, idx))

        return Result(
            "write_pareto",
            label,
            "cuda" if "nvenc" in codec else "cpu",
            spec.name,
            enc_fps,
            "frames/s",
            extra={
                "quality_param": quality,
                "file_size_mb": round(size_bytes / 1e6, 2),
                "compression_ratio": round(compression_ratio, 1),
                "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4),
                "resolution": f"{spec.height}x{spec.width}",
            },
        )
    except Exception as e:
        return Result(
            "write_pareto",
            label,
            "cuda" if "nvenc" in codec else "cpu",
            spec.name,
            float("nan"),
            "frames/s",
            extra={"quality_param": quality},
            error=f"{type(e).__name__}: {e}",
        )


def run() -> list[Result]:
    from .bench_write import WRITE_SPECS

    results: list[Result] = []
    for spec in WRITE_SPECS:
        frames = raw_frames(spec)
        for label, codec, params_fn, ok, reason in _methods():
            if not ok:
                results.append(
                    Result(
                        "write_pareto",
                        label,
                        "cuda" if "nvenc" in codec else "cpu",
                        spec.name,
                        float("nan"),
                        "frames/s",
                        error=reason,
                    )
                )
                continue
            for q in _QUALITY_SWEEP:
                print(f"  [write_pareto] {spec.name:12s} {label:14s} q={q}")
                results.append(
                    _bench_point(label, codec, params_fn(q), spec, frames, q)
                )
    return results


if __name__ == "__main__":
    for r in run():
        tag = f"{r.backend:14s} {r.workload:10s}"
        if r.error:
            print(f"{tag} q={r.extra.get('quality_param')} SKIP/ERR: {r.error}")
        else:
            print(f"{tag} {r.metric_main:7.1f} fps  {r.extra}")

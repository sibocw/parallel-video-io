"""Task 3: sequential frame access — streaming decode of every frame in order.

Metric: throughput (frames/s). Frames are streamed one at a time and never
accumulated, so the measurement reflects decode speed, not buffering, and
high-resolution videos do not exhaust memory.
"""

from __future__ import annotations

from . import config, datagen
from .backends.decode import DECODE_BACKENDS
from .common import PeakMemSampler, Result, best_of, timer


def _bench_sequential(backend, spec) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            "sequential",
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "frames/s",
            error=reason,
        )
    path = str(spec.path)
    try:
        backend.sequential(path, spec.n_frames)  # warm-up (caches, codec init)

        def run() -> float:
            with PeakMemSampler() as mem, timer() as t:
                n = backend.sequential(path, spec.n_frames)
            run._mem = mem  # type: ignore[attr-defined]
            return n / t[0]

        fps = best_of(run, config.N_REPEATS)
        mem = run._mem  # type: ignore[attr-defined]
        return Result(
            "sequential",
            backend.name,
            backend.device,
            spec.name,
            fps,
            "frames/s",
            extra={
                "peak_rss_mb": round(mem.peak_rss_mb, 1),
                "peak_cuda_mb": round(mem.peak_cuda_mb, 1),
                "resolution": f"{spec.height}x{spec.width}",
            },
        )
    except Exception as e:
        return Result(
            "sequential",
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "frames/s",
            error=f"{type(e).__name__}: {e}",
        )


def run() -> list[Result]:
    datagen.ensure_videos(config.DECODE_VIDEOS)
    results: list[Result] = []
    for spec in config.DECODE_VIDEOS:
        for backend in DECODE_BACKENDS:
            if not backend.supports_sequential:
                continue
            print(f"  [sequential] {spec.name:10s} {backend.name}")
            results.append(_bench_sequential(backend, spec))
    return results


if __name__ == "__main__":
    for r in run():
        print(r.flat())

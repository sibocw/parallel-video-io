"""Task 1 (sequential decode throughput) and Task 2 (random-access seek
latency + correctness)."""

from __future__ import annotations

import numpy as np

from . import config, datagen
from .backends.read import READ_BACKENDS
from .common import PeakMemSampler, Result, best_of, timer

# A recovered index within this many frames of the request counts as a correct
# seek (datagen recovery error is < 0.6 frame; a wrong keyframe lands frames off).
_CORRECT_TOL = 1.0


def _bench_sequential(backend, spec) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            "read_sequential",
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
            "read_sequential",
            backend.name,
            backend.device,
            spec.name,
            fps,
            "frames/s",
            extra={
                "peak_rss_mb": round(mem.peak_rss_mb, 1),
                "peak_cuda_mb": round(mem.peak_cuda_mb, 1),
                "resolution": f"{spec.height}x{spec.width}",
                "codec": spec.codec,
                "gop": spec.gop,
            },
        )
    except Exception as e:
        return Result(
            "read_sequential",
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "frames/s",
            error=f"{type(e).__name__}: {e}",
        )


def _bench_random(backend, spec, indices) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            "read_random",
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "ms/frame",
            error=reason,
        )
    path = str(spec.path)
    try:
        backend.random(path, indices[:4])  # warm-up

        def run() -> float:
            with timer() as t:
                frames = backend.random(path, indices)
            run._frames = frames  # type: ignore[attr-defined]
            # throughput (frames/s) so best_of picks the fastest run
            return len(indices) / t[0]

        fps = best_of(run, config.N_REPEATS)
        frames = run._frames  # type: ignore[attr-defined]
        ms_per_frame = 1000.0 / fps

        # Seek correctness: recover each frame's encoded index.
        errors = [
            abs(datagen.recover_index(spec, frames[i]) - idx)
            for i, idx in enumerate(indices)
        ]
        max_err = float(np.max(errors))
        correct = bool(max_err <= _CORRECT_TOL)
        return Result(
            "read_random",
            backend.name,
            backend.device,
            spec.name,
            ms_per_frame,
            "ms/frame",
            extra={
                "frames_per_s": round(fps, 1),
                "seek_correct": correct,
                "max_seek_err_frames": round(max_err, 2),
                "resolution": f"{spec.height}x{spec.width}",
                "codec": spec.codec,
                "gop": spec.gop,
            },
        )
    except Exception as e:
        return Result(
            "read_random",
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "ms/frame",
            error=f"{type(e).__name__}: {e}",
        )


def run(specs: list[config.VideoSpec] | None = None) -> list[Result]:
    specs = specs or config.READ_VIDEOS
    datagen.ensure_videos(specs)
    results: list[Result] = []
    for spec in specs:
        rng = np.random.default_rng(0)
        indices = sorted(
            rng.choice(
                spec.n_frames,
                size=min(config.N_RANDOM_READS, spec.n_frames),
                replace=False,
            ).tolist()
        )
        for backend in READ_BACKENDS:
            print(f"  [seq]  {spec.name:18s} {backend.name}")
            results.append(_bench_sequential(backend, spec))
            print(f"  [rand] {spec.name:18s} {backend.name}")
            results.append(_bench_random(backend, spec, indices))
    return results


if __name__ == "__main__":
    for r in run():
        print(r.flat())

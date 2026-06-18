"""Task 2: random frame access — precise (frame-accurate) seek decoding.

Metric: throughput (frames/s) when fetching a fixed set of random frame
indices. We also verify *seek correctness exactly*: each decoded frame carries
a binary barcode of its index (see datagen), recovered as an exact integer, so
the decoded frame must equal the requested one with zero tolerance. A backend
that is fast but lands on a neighbouring or keyframe-aligned frame decodes to
the wrong index and is flagged, not credited.
"""

from __future__ import annotations

import numpy as np

from . import config, datagen
from .backends.decode import DECODE_BACKENDS
from .common import Result, best_of, timer


def _bench_random(backend, spec, indices) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            "random",
            backend.name,
            backend.device,
            spec.name,
            float("nan"),
            "frames/s",
            error=reason,
        )
    path = str(spec.path)
    try:
        backend.random(path, indices[:4])  # warm-up

        def run() -> float:
            with timer() as t:
                frames = backend.random(path, indices)
            run._frames = frames  # type: ignore[attr-defined]
            return len(indices) / t[0]

        fps = best_of(run, config.N_REPEATS)
        frames = run._frames  # type: ignore[attr-defined]

        recovered = [
            datagen.recover_index(spec, frames[i]) for i in range(len(indices))
        ]
        n_wrong = sum(int(r != idx) for r, idx in zip(recovered, indices))
        return Result(
            "random",
            backend.name,
            backend.device,
            spec.name,
            fps,
            "frames/s",
            extra={
                "ms_per_frame": round(1000.0 / fps, 2),
                "seek_correct": bool(n_wrong == 0),
                "n_wrong_frames": n_wrong,
                "n_frames_checked": len(indices),
                "resolution": f"{spec.height}x{spec.width}",
            },
        )
    except Exception as e:
        return Result(
            "random",
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
        rng = np.random.default_rng(0)
        indices = sorted(
            rng.choice(
                spec.n_frames,
                size=min(config.N_RANDOM_READS, spec.n_frames),
                replace=False,
            ).tolist()
        )
        for backend in DECODE_BACKENDS:
            if not backend.supports_random:
                continue
            print(f"  [random] {spec.name:10s} {backend.name}")
            results.append(_bench_random(backend, spec, indices))
    return results


if __name__ == "__main__":
    for r in run():
        print(r.flat())

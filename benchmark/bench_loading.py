"""Task 3: parallel multi-video streaming throughput (the headline use case).

Streams every frame of a collection of videos through each loading backend,
delivering frames to the GPU where a tiny consumer model runs, and reports
aggregate frames/s, peak host RSS and peak CUDA memory, swept over worker count.
"""

from __future__ import annotations

import os

import torch

from . import config, datagen
from .backends.loaders import LOADING_BACKENDS
from .common import PeakMemSampler, Result, timer


def _bench_one(backend, paths, num_workers, batch_size, device) -> Result:
    ok, reason = backend.available()
    if not ok:
        return Result(
            "loading",
            backend.name,
            device,
            "collection",
            float("nan"),
            "frames/s",
            extra={"num_workers": num_workers},
            error=reason,
        )
    try:
        with PeakMemSampler() as mem, timer() as t:
            n = backend.run(paths, num_workers, batch_size, device)
        fps = n / t[0]
        return Result(
            "loading",
            backend.name,
            device,
            "collection",
            fps,
            "frames/s",
            extra={
                "num_workers": num_workers,
                "n_frames": n,
                "seconds": round(t[0], 2),
                "peak_rss_mb": round(mem.peak_rss_mb, 1),
                "peak_cuda_mb": round(mem.peak_cuda_mb, 1),
                "n_videos": len(paths),
                "batch_size": batch_size,
            },
        )
    except Exception as e:
        return Result(
            "loading",
            backend.name,
            device,
            "collection",
            float("nan"),
            "frames/s",
            extra={"num_workers": num_workers},
            error=f"{type(e).__name__}: {e}",
        )


def run() -> list[Result]:
    specs = config.collection_videos()
    datagen.ensure_videos(specs)
    paths = [str(s.path) for s in specs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_cpu = os.cpu_count() or 1
    sweep = [w for w in config.worker_sweep() if w <= n_cpu]

    results: list[Result] = []
    for backend in LOADING_BACKENDS:
        for nw in sweep:
            print(f"  [load] {backend.name:18s} workers={nw}")
            results.append(
                _bench_one(backend, paths, nw, config.LOADING_BATCH_SIZE, device)
            )
    return results


if __name__ == "__main__":
    for r in run():
        if r.error:
            print(f"{r.backend:18s} w={r.extra['num_workers']} SKIP/ERR: {r.error}")
        else:
            print(
                f"{r.backend:18s} w={r.extra['num_workers']} "
                f"{r.metric_main:8.1f} fps  {r.extra}"
            )

"""Shared utilities: timing, memory sampling, result records, environment capture."""

from __future__ import annotations

import contextlib
import gc
import json
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class Result:
    """One benchmark measurement row. Serialised to CSV/JSON by the runner."""

    task: str  # "read_sequential", "read_random", "write", "loading", "loc"
    backend: str  # e.g. "pvio", "torchcodec_cuda", "decord_cpu", "dali_gpu"
    device: str  # "cpu" or "cuda"
    workload: str  # e.g. video spec name, or "collection"
    metric_main: float  # primary number (frames/s, or for loc: code lines)
    metric_unit: str  # "frames/s", "ms/frame", "lines", ...
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # set if the backend failed/was skipped

    def flat(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        for k, v in extra.items():
            d[f"x_{k}"] = v
        return d


@contextlib.contextmanager
def timer():
    """Wall-clock timer. Yields a one-element list; result lands in ``out[0]``."""
    out = [0.0]
    start = time.perf_counter()
    try:
        yield out
    finally:
        out[0] = time.perf_counter() - start


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PeakMemSampler:
    """Background sampler for peak resident set size (RSS) across this process
    tree, plus peak CUDA memory via torch's allocator stats.

    RSS is sampled in a thread (covers DataLoader worker subprocesses via the
    OS, but Python-side we read this process; worker RSS is captured by reading
    /proc children). CUDA peak is read from torch.cuda.max_memory_allocated.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0

    def _read_tree_rss_kb(self) -> int:
        # Sum RSS of this process and all descendants from /proc.
        import os

        pids = [os.getpid()]
        total = 0
        seen = set()
        # Build a children map once per sample (cheap enough at 20 Hz).
        try:
            proc = Path("/proc")
            children: dict[int, list[int]] = {}
            for p in proc.iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    ppid = int((p / "stat").read_text().split()[3])
                except (OSError, ValueError, IndexError):
                    continue
                children.setdefault(ppid, []).append(int(p.name))
            stack = list(pids)
            while stack:
                pid = stack.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    for line in (proc / str(pid) / "status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                            break
                except OSError:
                    pass
                stack.extend(children.get(pid, []))
        except OSError:
            pass
        return total

    def _loop(self):
        while not self._stop.is_set():
            rss_mb = self._read_tree_rss_kb() / 1024.0
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            self._stop.wait(self.interval)

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.peak_rss_mb = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_cuda_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0


def best_of(fn: Callable[[], float], repeats: int) -> float:
    """Run ``fn`` ``repeats`` times, return the best (max) throughput.

    ``fn`` must return a throughput-like number where higher is better.
    """
    best = float("-inf")
    for _ in range(repeats):
        best = max(best, fn())
    return best


def capture_environment() -> dict[str, Any]:
    """Snapshot of hardware/software for reproducibility, stored next to results."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    for mod in ("torchcodec", "av", "cv2", "decord"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:
            env[mod] = None
    try:
        import nvidia.dali as dali

        env["dali"] = dali.__version__
    except Exception:
        env["dali"] = None
    try:
        ff = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        env["ffmpeg"] = ff.stdout.splitlines()[0]
    except Exception:
        env["ffmpeg"] = None
    env["cpu_count"] = __import__("os").cpu_count()
    return env


def save_environment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capture_environment(), indent=2))

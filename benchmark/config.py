"""Central configuration for the benchmark suite.

All knobs live here so the harness can be scaled from a quick smoke run to a
full sweep without touching the benchmark logic. Override any value via the
matching ``PVIO_BENCH_*`` environment variable (see :func:`_env`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository-root-relative locations.
BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SNIPPETS_DIR = BENCH_DIR / "snippets"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PVIO_BENCH_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass(frozen=True)
class VideoSpec:
    """A synthetic test video: how it is generated and encoded."""

    name: str
    height: int
    width: int
    n_frames: int
    fps: int
    codec: str  # "h264" or "hevc"
    gop: int  # keyframe interval (group-of-pictures size)
    crf: int = 20

    @property
    def path(self) -> Path:
        return DATA_DIR / f"{self.name}.mp4"

    @property
    def frames_dir(self) -> Path:
        return DATA_DIR / f"{self.name}_frames"


# Resolution presets (height, width).
_RES = {
    "sd": (480, 854),
    "hd": (720, 1280),
    "fhd": (1080, 1920),
    "uhd": (2160, 3840),
}

# Number of frames per single-video test file. Small by default so the suite
# runs in minutes; bump via PVIO_BENCH_NFRAMES for a heavier sweep.
NFRAMES = _env_int("NFRAMES", 600)
FPS = _env_int("FPS", 30)

# Single-file read/write test matrix: resolution x codec x GOP.
READ_VIDEOS: list[VideoSpec] = [
    VideoSpec("sd_h264_g30", *_RES["sd"], NFRAMES, FPS, "h264", gop=30),
    VideoSpec("hd_h264_g30", *_RES["hd"], NFRAMES, FPS, "h264", gop=30),
    VideoSpec("hd_h264_g120", *_RES["hd"], NFRAMES, FPS, "h264", gop=120),
    VideoSpec("fhd_h264_g30", *_RES["fhd"], NFRAMES, FPS, "h264", gop=30),
    VideoSpec("fhd_hevc_g30", *_RES["fhd"], NFRAMES, FPS, "hevc", gop=30),
    VideoSpec("uhd_h264_g30", *_RES["uhd"], NFRAMES, FPS, "h264", gop=30),
]

# The parallel-loading benchmark (PVIO's headline use case) streams frames from
# many videos at once. Keep these small per-file but numerous.
N_COLLECTION_VIDEOS = _env_int("N_COLLECTION_VIDEOS", 32)
COLLECTION_NFRAMES = _env_int("COLLECTION_NFRAMES", 300)
COLLECTION_RES = _RES[_env("COLLECTION_RES", "hd")]
COLLECTION_CODEC = _env("COLLECTION_CODEC", "h264")
COLLECTION_GOP = _env_int("COLLECTION_GOP", 30)


def collection_videos() -> list[VideoSpec]:
    """Specs for the multi-video collection used by the loading benchmark.

    Frame counts are deliberately *uneven* (cycling between ~1/3 and full
    length). This exercises PVIO's frame-level load balancing against the
    common "shard whole videos across workers" baseline, which becomes
    imbalanced when videos differ in length.
    """
    h, w = COLLECTION_RES
    lo = max(30, COLLECTION_NFRAMES // 3)
    specs = []
    for i in range(N_COLLECTION_VIDEOS):
        frac = (i % 4) / 3.0  # 0, 1/3, 2/3, 1, repeating
        n = int(lo + frac * (COLLECTION_NFRAMES - lo))
        specs.append(
            VideoSpec(
                f"coll_{i:03d}", h, w, n, FPS, COLLECTION_CODEC, gop=COLLECTION_GOP
            )
        )
    return specs


# Random-access read benchmark: how many random frames to fetch per video.
N_RANDOM_READS = _env_int("N_RANDOM_READS", 100)


# Worker counts swept in the loading benchmark. Capped to available cores at run
# time. "0" means PVIO's single-process path.
def worker_sweep() -> list[int]:
    raw = _env("WORKER_SWEEP", "1,2,4,8")
    return [int(x) for x in raw.split(",") if x.strip()]


# Batch size for the loading benchmark.
LOADING_BATCH_SIZE = _env_int("LOADING_BATCH_SIZE", 16)

# Number of timed repeats for the (fast) read/write micro-benchmarks.
N_REPEATS = _env_int("N_REPEATS", 3)


@dataclass
class RunConfig:
    """Top-level switches for an end-to-end run."""

    read: bool = True
    write: bool = True
    loading: bool = True
    loc: bool = True
    seed: int = field(default_factory=lambda: _env_int("SEED", 0))

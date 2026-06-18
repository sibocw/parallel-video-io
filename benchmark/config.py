"""Central configuration for the consolidated benchmark suite.

Three tasks — encoding (merge frames -> video), random access (precise seek),
and sequential access — are measured across the video libraries. All knobs live
here and can be overridden via the matching ``PVIO_BENCH_*`` environment
variable, so the suite scales from a quick smoke run to a full sweep.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repository-root-relative locations.
BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


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
    codec: str  # "h264" (only encoder wired up in datagen)
    gop: int  # keyframe interval (group-of-pictures size)
    crf: int = 20  # CRF passed to ffmpeg when generating this source video

    @property
    def path(self) -> Path:
        return DATA_DIR / f"{self.name}.mp4"


# Resolution presets (height, width). Encode specs use multiples of 16 so the
# FFmpeg/imageio writer does not macroblock-pad and change the frame size.
_RES = {
    "sd": (480, 854),
    "hd": (720, 1280),
    "fhd": (1080, 1920),
}

FPS = _env_int("FPS", 30)
NFRAMES = _env_int("NFRAMES", 300)

# Decode test matrix (random + sequential access), generated to disk.
DECODE_VIDEOS: list[VideoSpec] = [
    VideoSpec("sd_h264", *_RES["sd"], NFRAMES, FPS, "h264", gop=30),
    VideoSpec("hd_h264", *_RES["hd"], NFRAMES, FPS, "h264", gop=30),
    VideoSpec("fhd_h264", *_RES["fhd"], NFRAMES, FPS, "h264", gop=30),
]

# Encode test specs (frames held in memory so they can be re-decoded for quality
# scoring). Dimensions are multiples of 16.
_ENCODE_NFRAMES = _env_int("ENCODE_NFRAMES", 150)
ENCODE_SPECS: list[VideoSpec] = [
    VideoSpec("enc_sd", 480, 848, _ENCODE_NFRAMES, FPS, "h264", gop=30),
    VideoSpec("enc_hd", 720, 1280, _ENCODE_NFRAMES, FPS, "h264", gop=30),
]


# Quality sweep used to trace each encoder's speed-vs-compression frontier. The
# values are on the 0-51 H.264 quantiser scale, applied as libx264's CRF or
# NVENC's QP per encoder. CRF and QP are *not* equivalent operating points, so
# encoders are never compared at a shared quality number — they are compared at
# matched PSNR, interpolated from this sweep (see MATCH_PSNR).
def quality_sweep() -> list[int]:
    raw = _env("QUALITY_SWEEP", "17,19,21,23,25")
    return [int(x) for x in raw.split(",") if x.strip()]


# Target PSNR (dB) for the matched-quality encode comparison. 0 (default) picks,
# per workload, the midpoint of the PSNR range reachable by every encoder, so
# all backends are compared at the same image quality regardless of CRF/QP.
MATCH_PSNR = float(_env("MATCH_PSNR", "0"))


# Per-frame JPEG quality used as the compression-ratio baseline: compression
# ratio = (sum of per-frame JPEG bytes) / (encoded video bytes), i.e. how much
# smaller the video is than storing each frame as a high-quality JPEG.
JPEG_QUALITY = _env_int("JPEG_QUALITY", 95)

# Random-access benchmark: how many random frames to fetch per video.
N_RANDOM_READS = _env_int("N_RANDOM_READS", 100)

# Number of timed repeats; the best (fastest) run is reported.
N_REPEATS = _env_int("N_REPEATS", 3)

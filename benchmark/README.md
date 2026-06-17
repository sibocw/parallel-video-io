# PVIO benchmark suite

Benchmarks [parallel-video-io](../README.md) (PVIO) against other video IO
libraries across three tasks — **encoding** (merge frames into a video),
**random access** (precise seek decoding), and **sequential access** — on four
metrics: lines of user code, throughput, compression ratio, and the encode
speed-vs-compression Pareto front. GPU paths (NVENC encode, NVDEC decode,
on-GPU DALI) are used where available.

## Install

```bash
uv sync --extra benchmark
```

This pulls in the comparison libraries via the `benchmark` optional-dependency
group in [`pyproject.toml`](../pyproject.toml): PyAV, OpenCV, the
[eva-decord](https://pypi.org/project/eva-decord/) Decord fork, NVIDIA DALI,
scikit-image (PSNR/SSIM), pandas, and matplotlib. FFmpeg must be on `PATH`.

## Run

```bash
uv run python -m benchmark.run_all            # full suite + figures
uv run python -m benchmark.run_all --quick    # small, fast smoke run
uv run python -m benchmark.run_all --only loc # one task group
uv run python -m benchmark.run_all --no-figures
```

Outputs land in `benchmark/results/`:

- `results.csv` — every measurement (one row per backend × workload).
- `SUMMARY.md` — pivoted markdown tables.
- `environment.json` — hardware/library versions for reproducibility.
- `figures/*.png` — the charts (also regenerable via `python -m benchmark.plots`).

Synthetic test videos are generated once into `benchmark/data/` (git-ignored)
and reused. Tune the workload with `PVIO_BENCH_*` environment variables — see
[`config.py`](config.py).

## What is measured

| Task | Metric(s) | Backends |
|------|-----------|----------|
| **Encoding** (frames → video) | encode frames/s, compression ratio, PSNR/SSIM, Pareto front | pvio_cpu (libx264), pvio_gpu (NVENC), pyav, opencv |
| **Random access** (precise seek) | frames/s **and seek correctness** | decord (seek_accurate), opencv, pvio cpu/gpu, pyav, torchcodec cpu/cuda |
| **Sequential access** | frames/s, peak RAM/GPU mem | dali, decord, opencv, pvio cpu/gpu, pyav, torchcodec cpu/cuda |
| **Lines of code** | SLOC of idiomatic user code (ruff-formatted, lint-clean) | per task, per library |

### Metric definitions and fairness

- **Throughput** is frames/s (higher is better) on every task, taking the best
  of `N_REPEATS` timed runs after a warm-up.
- **Compression ratio** is `(sum of per-frame JPEG bytes) / (encoded video
  bytes)` — i.e. how much smaller the video is than storing each frame as a
  JPEG (quality `JPEG_QUALITY`, default 95). This is a meaningful, bounded
  baseline; the raw-RGB baseline produced uninterpretable ratios in the
  thousands. Encoders are compared at the same effective quality (CRF for
  libx264, QP for NVENC, default 20), with PSNR/SSIM recorded so size can be
  read at matched quality.
- **Pareto front** sweeps the quality knob for each tunable encoder, tracing a
  curve in (compression ratio, throughput) space; up and to the right is
  better. NVENC's throughput is roughly flat (hardware-fixed) while libx264
  speeds up as quality drops, so the curves cross.
- **Seek correctness** is real, not assumed: each synthetic frame carries a
  bright vertical bar whose position encodes its index. After a random read we
  recover the index from the decoded pixels and check it matches the request
  (tolerance ≈ 1 frame). A library that silently returns the nearest keyframe
  is caught here; Decord is driven via `seek_accurate` for frame accuracy.
- **LOC** counts source lines (non-blank, non-comment) of the minimal idiomatic
  snippet for each library in [`snippets/`](snippets), after `ruff format`, and
  requires `ruff check` to pass so the code is genuinely reasonable style.

## Backend notes

- **PVIO** appears as **CPU** and **GPU** for both encode (libx264 / NVENC via
  `write_frames_to_video(mode=...)`) and decode (`EncodedVideo(device=...)`).
- **TorchCodec** is benchmarked raw on both **CPU** and **CUDA/NVDEC**. It is
  PVIO's own decode backend, so `pvio_cpu` vs `torchcodec_cpu` (and `pvio_gpu`
  vs `torchcodec_cuda`) shows the wrapper's overhead.
- **DALI** decodes on the GPU; its video reader is a sequential pipeline, so it
  appears in the sequential task only (no precise random seek).
- **Decord**: the PyPI `eva-decord` wheel is **CPU-only**; random access uses
  `seek_accurate` for frame-accurate seeks.
- **Encoding frame sizes** are multiples of 16: `write_frames_to_video` goes
  through imageio/FFmpeg, which macroblock-pads otherwise, which would change
  the frame size and break the like-for-like quality comparison.

## Layout

```
benchmark/
  config.py           # all knobs (env-overridable)
  common.py           # timing, mem sampling, Result, JPEG baseline, quality scoring
  datagen.py          # synthesise test videos (index-encoded frames)
  backends/           # encode.py, decode.py — one class per backend
  bench_encode.py     # task 1: encoding (+ Pareto sweep)
  bench_random.py     # task 2: precise random access
  bench_sequential.py # task 3: sequential access
  loc.py              # lines-of-code metric
  snippets/           # idiomatic user code per task/library (counted by loc.py)
  plots.py            # figures from results.csv
  run_all.py          # orchestrator
```

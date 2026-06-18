# PVIO benchmark suite

Benchmarks [parallel-video-io](../README.md) (PVIO) against other video IO
libraries across three tasks — **encoding** (merge frames into a video),
**random access** (precise seek decoding), and **sequential access**. Each task
reports throughput (frames/s); encoding adds compression ratio with PSNR/SSIM
(so encoders can be compared at matched image quality) and random access adds
**exact seek correctness**. GPU paths (NVENC encode, NVDEC decode, on-GPU DALI)
are used where available.

## Latest results

Pre-run results are in [`results/SUMMARY.md`](results/SUMMARY.md) (pivoted
markdown tables) and [`results/figures/`](results/figures/) (HTML charts).
Re-generate at any time following the steps below.

## Prerequisites

```bash
uv sync --extra benchmark
```

This pulls in PyAV, OpenCV, [eva-decord](https://pypi.org/project/eva-decord/),
scikit-image, pandas, and plotly via the `benchmark` optional-dependency group
in [`pyproject.toml`](../pyproject.toml). TorchCodec is already a core PVIO
dependency. FFmpeg must be on `PATH`.

GPU backends (PVIO GPU, TorchCodec CUDA, DALI) require an NVIDIA GPU with CUDA
and the matching NVENC/NVDEC drivers. They are automatically skipped when no GPU
is available. **NVIDIA DALI** is GPU-only and ships from NVIDIA's own package
index, so it is not part of the `benchmark` extra; install it separately (e.g.
`pip install nvidia-dali-cuda120`) to include the DALI sequential-decode
backend.

## Running the benchmark

### Full suite

```bash
uv run python -m benchmark.run_all
```

This runs all three tasks (encode, random access, sequential access), writes
results to `benchmark/results/`, and regenerates the interactive HTML figures.

### Quick smoke run

```bash
uv run python -m benchmark.run_all --quick
```

Reduces frame counts, repeats, and quality-sweep points to finish in a few
minutes. Useful for checking that all backends work before committing to a full
run.

### Partial runs

```bash
uv run python -m benchmark.run_all --only encode
uv run python -m benchmark.run_all --only random sequential
uv run python -m benchmark.run_all --no-figures   # skip figure generation
```

### Regenerate figures only

If `benchmark/results/results.csv` already exists and you only changed
`plots.py`:

```bash
uv run python -m benchmark.plots
```

## Output files

| File | Description |
|------|-------------|
| `results/results.csv` | Every measurement — one row per backend × workload × task |
| `results/SUMMARY.md` | Pivoted markdown tables (embedded in the docs) |
| `results/environment.json` | Hardware and library versions for reproducibility |
| `results/figures/*.html` | Interactive Plotly figures (embedded in the docs) |

Synthetic test videos are generated once into `benchmark/data/` (git-ignored)
and reused across runs.

## Fixing stale test videos

If you see errors like `KeyError: 122` or `Invalid frame index=122` in the
random-access task, your `benchmark/data/` videos were generated with a
different `PVIO_BENCH_NFRAMES` value than the current config. Delete and
regenerate:

```bash
rm -rf benchmark/data/
uv run python -m benchmark.run_all
```

The datagen step runs automatically at the start of any benchmark run and
creates videos matching the current `NFRAMES` setting (default: 300 frames).

## Tuning the workload

All knobs can be overridden via environment variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `PVIO_BENCH_NFRAMES` | `300` | Frames per decode test video |
| `PVIO_BENCH_ENCODE_NFRAMES` | `150` | Frames per encode test clip |
| `PVIO_BENCH_FPS` | `30` | Frame rate of generated videos |
| `PVIO_BENCH_N_REPEATS` | `3` | Timed repetitions (best is reported) |
| `PVIO_BENCH_N_RANDOM_READS` | `100` | Random frames fetched per video |
| `PVIO_BENCH_QUALITY_SWEEP` | `17,19,21,23,25` | CRF/QP values swept per encoder |
| `PVIO_BENCH_MATCH_PSNR` | `0` (auto) | Target PSNR (dB) for the matched-quality encode comparison; `0` uses the per-workload overlap midpoint |
| `PVIO_BENCH_JPEG_QUALITY` | `95` | JPEG quality for compression-ratio baseline |

## What is measured

| Task | Metric(s) | Backends |
|------|-----------|----------|
| **Encoding** (frames → video) | encode frames/s, compression ratio, PSNR/SSIM, matched-PSNR point | pvio_cpu (libx264), pvio_gpu (NVENC), pyav, opencv (MJPEG) |
| **Random access** (precise seek) | frames/s **and exact seek correctness** | decord, opencv, pvio cpu/gpu, pyav, torchcodec cpu/cuda |
| **Sequential access** | frames/s | dali, decord, opencv, pvio cpu/gpu, pyav, torchcodec cpu/cuda |

### Metric definitions and fairness

- **Throughput** is frames/s (higher is better) on every task, taking the best
  of `N_REPEATS` timed runs after a warm-up pass.
- **Compression ratio** is `(sum of per-frame JPEG bytes) / (encoded video
  bytes)` — how much smaller the video is than storing each frame as a JPEG
  (quality `JPEG_QUALITY`, default 95).
- **Matched-quality comparison.** Encoders are *not* compared at a shared
  quality number: libx264's CRF and NVENC's QP sit on the same 0–51 scale but
  are different operating points (CRF 20 ≠ QP 20). Instead each encoder's
  quality is swept and its throughput/compression are interpolated to a common
  **PSNR** (`MATCH_PSNR`, default auto), so all encoders are read at equal image
  quality.
- **Preset is held fixed, not swept.** `write_frames_to_video` exposes *two*
  encode knobs — `quality` (CRF/QP) and `preset` (the encoder's
  speed-vs-compression effort level, e.g. libx264 `ultrafast`…`placebo` or NVENC
  `p1`…`p7`). The benchmark sweeps only `quality`; `preset` is pinned to each
  encoder's default (`slow` for libx264, `p7` for NVENC) and the PyAV libx264
  baseline is pinned to the matching `slow`. This is deliberate: the goal is a
  single speed-vs-compression frontier per encoder compared at matched PSNR, so
  preset is held constant across the CPU encoders to isolate codec/encoder
  efficiency rather than confound it with a second free variable. Preset remains
  a real tuning axis in the library API (`write_frames_to_video(..., preset=)`)
  — it is simply not part of this comparison. To re-run the whole suite at a
  different effort level, change the pinned preset in `backends/encode.py`.
- **Seek correctness** is verified *exactly*: each synthetic frame carries a
  binary barcode of its index (macroblock-sized black/white blocks that survive
  H.264 intact). After a random read, the index is decoded from the pixels and
  must equal the requested index — zero tolerance. A library that silently
  returns the nearest keyframe decodes to the wrong index and is flagged.
- **Single-threaded decode.** All CPU decoders are pinned to one FFmpeg thread.
  Out of the box OpenCV and Decord decode multithreaded (grabbing every core)
  while TorchCodec/PVIO and PyAV decode single-threaded, so comparing defaults
  measures *how many cores each library grabs* rather than decode efficiency —
  with all cores the FFmpeg-based decoders converge to within ~15% of each
  other. Pinning to one thread isolates per-core efficiency and matches PVIO's
  operating regime: many videos decoded in parallel across DataLoader workers,
  where one thread per decode avoids core oversubscription. (GPU decoders —
  NVDEC, DALI — are unaffected.)

## Backend notes

- **PVIO** appears as **CPU** and **GPU** for both encode (libx264 / NVENC via
  `write_frames_to_video(mode=...)`) and decode (`EncodedVideo(device=...)`).
- **TorchCodec** is benchmarked raw on both **CPU** and **CUDA/NVDEC**. It is
  PVIO's own decode backend, and raw TorchCodec uses `seek_mode="exact"` to
  match what PVIO does internally, so `pvio_cpu` vs `torchcodec_cpu` (and
  `pvio_gpu` vs `torchcodec_cuda`) isolates the wrapper's overhead rather than a
  seek-mode difference.
- **DALI** decodes on the GPU; its video reader is a sequential pipeline, so it
  appears in the sequential task only (no precise random seek).
- **Decord**: the PyPI `eva-decord` wheel is CPU-only; random access uses
  `seek_accurate` for frame-accurate seeks.
- **Encoding frame sizes** are multiples of 16: `write_frames_to_video` goes
  through imageio/FFmpeg, which macroblock-pads otherwise, changing the frame
  size and breaking the like-for-like quality comparison.

## Layout

```
benchmark/
  config.py           # all knobs (env-overridable)
  common.py           # timing, Result, JPEG baseline, quality scoring
  datagen.py          # synthesise test videos (barcode-indexed frames)
  analysis.py         # derived metrics (matched-PSNR comparison)
  backends/           # encode.py, decode.py — one class per backend
  bench_encode.py     # task 1: encoding quality sweep
  bench_random.py     # task 2: precise random access
  bench_sequential.py # task 3: sequential access
  plots.py            # figures from results.csv
  run_all.py          # orchestrator
```

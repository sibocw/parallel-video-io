# PVIO benchmark suite

Benchmarks [parallel-video-io](../README.md) (PVIO) against other video IO
libraries across reading, writing, parallel data loading, and **lines of
user-written code**. GPU paths (NVDEC decode, on-GPU DALI pipeline) are used
where available.

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
| **Sequential decode** | frames/s, peak RAM/GPU mem | pvio (EncodedVideo), pvio_simple (imageio), torchcodec cpu/cuda, decord, pyav, opencv |
| **Random-access seek** | ms/frame **and seek correctness** | same as above |
| **Parallel loading** (headline) | frames/s vs `num_workers`, peak RAM/GPU mem | pvio (`VideoCollectionDataLoader`), torchcodec_naive (DIY shard-by-video), dali_gpu |
| **Write** | encode frames/s, file size, PSNR/SSIM at matched CRF | pvio (imageio), pyav, opencv |
| **Lines of code** | SLOC of idiomatic user code (ruff-formatted, lint-clean) | per task, per library |

### How tasks are made fair

- **Seek correctness** is real, not assumed: each synthetic frame carries a
  bright vertical bar whose position encodes its index. After a random read we
  recover the index from the decoded pixels and check it matches the request
  (tolerance ≈ 1 frame; recovery error is < 0.6 frame). A library that silently
  returns the nearest keyframe is caught here.
- **Uneven video lengths** in the loading collection deliberately stress
  PVIO's frame-level load balancing against the common "shard whole videos
  across workers" approach, which goes imbalanced when clip lengths differ.
- **Matched CRF** for the write quality/size comparison (PVIO and PyAV at the
  same CRF; OpenCV's `VideoWriter` can't set CRF, so it is flagged
  `crf_controlled=false`).
- **LOC** counts source lines (non-blank, non-comment) of the minimal idiomatic
  snippet for each library in [`snippets/`](snippets), after `ruff format`, and
  requires `ruff check` to pass so the code is genuinely reasonable style.

## Backend notes

- **TorchCodec** is benchmarked raw on both **CPU** and **CUDA/NVDEC**. It is
  also PVIO's own decode backend, so `pvio` vs `torchcodec_cpu` shows the
  wrapper's overhead. `torchvision.io.VideoReader` is intentionally omitted: it
  is deprecated in favour of TorchCodec (already covered) and ships no cu130
  wheel.
- **Decord**: the PyPI `eva-decord` wheel is **CPU-only** (not built with
  CUDA), so the GPU path is reported as unavailable.
- **PVIO writer gotcha**: `write_frames_to_video` goes through imageio/FFmpeg,
  which pads frame dimensions up to a multiple of 16 (the macroblock size). The
  write benchmark uses dimensions divisible by 16 so quality is compared
  like-for-like; with arbitrary sizes PVIO's output dimensions can differ from
  the input.

## Layout

```
benchmark/
  config.py         # all knobs (env-overridable)
  common.py         # timing, peak RAM/GPU sampling, Result record, env capture
  datagen.py        # synthesise test videos (index-encoded frames)
  backends/         # read.py, write.py, loaders.py — one class per backend
  bench_read.py     # sequential + random tasks
  bench_loading.py  # parallel multi-video loading task
  bench_write.py    # write task
  loc.py            # lines-of-code metric
  snippets/         # idiomatic user code per task/library (counted by loc.py)
  plots.py          # figures from results.csv
  run_all.py        # orchestrator
```

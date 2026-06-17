"""Render figures from the benchmark result CSV.

Reads ``benchmark/results/results.csv`` (or an in-memory DataFrame) and writes
PNGs to ``benchmark/results/figures``. Robust to partially-missing data:
each figure is skipped if its inputs are empty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES_DIR, RESULTS_DIR  # noqa: E402


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that produced a finite measurement (drop skipped/errored backends)."""
    return df[df["error"].isna() & df["metric_main"].notna()]


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _grouped_bar(ax, df, index, columns, values):
    pivot = df.pivot_table(index=index, columns=columns, values=values)
    pivot.plot.bar(ax=ax)
    ax.set_xlabel(index)
    ax.grid(axis="y", alpha=0.3)
    return pivot


def plot_read_sequential(df, out: list[Path]):
    sub = _ok(df[df["task"] == "read_sequential"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bar(ax, sub, "workload", "backend", "metric_main")
    ax.set_ylabel("frames / s (higher is better)")
    ax.set_title("Sequential decode throughput")
    ax.legend(title="backend", fontsize=8, ncol=2)
    out.append(_save(fig, "read_sequential_throughput.png"))


def plot_read_random(df, out: list[Path]):
    sub = _ok(df[df["task"] == "read_random"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bar(ax, sub, "workload", "backend", "metric_main")
    ax.set_ylabel("ms / frame (lower is better)")
    ax.set_title("Random-access seek latency (all seeks verified correct)")
    ax.legend(title="backend", fontsize=8, ncol=2)
    out.append(_save(fig, "read_random_latency.png"))


def plot_loading(df, out: list[Path]):
    sub = _ok(df[df["task"] == "loading"]).copy()
    if sub.empty:
        return
    sub["num_workers"] = sub["x_num_workers"].astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for backend, g in sub.groupby("backend"):
        g = g.sort_values("num_workers")
        ax1.plot(g["num_workers"], g["metric_main"], marker="o", label=backend)
    ax1.set_xlabel("num_workers")
    ax1.set_ylabel("frames / s (higher is better)")
    ax1.set_title("Parallel multi-video loading throughput")
    ax1.grid(alpha=0.3)
    ax1.legend(title="backend")

    # Peak host memory at the largest worker count.
    max_w = sub["num_workers"].max()
    mem = sub[sub["num_workers"] == max_w]
    ax2.bar(mem["backend"], mem["x_peak_rss_mb"])
    ax2.set_ylabel("peak host RSS (MB)")
    ax2.set_title(f"Peak memory @ {max_w} workers")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.3)
    out.append(_save(fig, "loading_throughput_and_memory.png"))


def plot_write(df, out: list[Path]):
    sub = _ok(df[df["task"] == "write"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    _grouped_bar(axes[0], sub, "workload", "backend", "metric_main")
    axes[0].set_ylabel("frames / s")
    axes[0].set_title("Encode throughput (higher is better)")
    _grouped_bar(axes[1], sub, "workload", "backend", "x_file_size_mb")
    axes[1].set_ylabel("file size (MB)")
    axes[1].set_title("Output size (lower is better)")
    _grouped_bar(axes[2], sub, "workload", "backend", "x_compression_ratio")
    axes[2].set_ylabel("compression ratio (raw / encoded)")
    axes[2].set_title("Compression ratio (higher is better)")
    _grouped_bar(axes[3], sub, "workload", "backend", "x_ssim")
    axes[3].set_ylabel("SSIM vs source")
    axes[3].set_title("Reconstruction quality (higher is better)")
    for ax in axes:
        ax.legend(title="backend", fontsize=8)
    out.append(_save(fig, "write_speed_size_quality.png"))


def plot_loc(df, out: list[Path]):
    sub = df[df["task"] == "loc"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bar(ax, sub, "workload", "backend", "metric_main")
    ax.set_ylabel("source lines of code (lower is better)")
    ax.set_title("User code required per task (ruff-formatted, lint-clean)")
    ax.legend(title="backend", fontsize=8, ncol=2)
    out.append(_save(fig, "lines_of_code.png"))


def generate(df: pd.DataFrame | None = None) -> list[Path]:
    if df is None:
        csv = RESULTS_DIR / "results.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found; run the benchmark first.")
        df = pd.read_csv(csv)
    if "error" not in df:
        df["error"] = pd.NA
    out: list[Path] = []
    plot_read_sequential(df, out)
    plot_read_random(df, out)
    plot_loading(df, out)
    plot_write(df, out)
    plot_loc(df, out)
    return out


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p}")

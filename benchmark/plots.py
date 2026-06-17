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


def plot_write_pareto(df, out: list[Path]):
    """Throughput vs compression-ratio Pareto curves, one line per encoder.

    Each method is swept across quality settings (libx264 CRF / NVENC QP), so it
    traces a frontier rather than sitting at a single point. Up and to the right
    is better (faster *and* smaller). Each marker is annotated with its PSNR so
    the curves can be read at matched quality, not just matched compression.
    """
    sub = _ok(df[df["task"] == "write_pareto"]).copy()
    if sub.empty:
        return
    workloads = sorted(sub["workload"].unique())
    fig, axes = plt.subplots(1, len(workloads), figsize=(7 * len(workloads), 5.5))
    if len(workloads) == 1:
        axes = [axes]
    for ax, workload in zip(axes, workloads):
        wl = sub[sub["workload"] == workload]
        for backend, g in wl.groupby("backend"):
            g = g.sort_values("x_compression_ratio")
            ax.plot(
                g["x_compression_ratio"],
                g["metric_main"],
                marker="o",
                label=backend,
            )
            for _, row in g.iterrows():
                ax.annotate(
                    f"{row['x_psnr_db']:.1f} dB",
                    (row["x_compression_ratio"], row["metric_main"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )
        ax.set_xlabel("compression ratio (raw / encoded) →")
        ax.set_ylabel("encode throughput (frames / s) →")
        ax.set_title(f"Speed vs compression Pareto — {workload}")
        ax.grid(alpha=0.3)
        ax.legend(title="encoder")
    out.append(_save(fig, "write_pareto.png"))


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
    plot_write_pareto(df, out)
    plot_loc(df, out)
    return out


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p}")

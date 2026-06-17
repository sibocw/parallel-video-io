"""Render figures from the consolidated benchmark result CSV.

Reads ``benchmark/results/results.csv`` (or an in-memory DataFrame) and writes
PNGs to ``benchmark/results/figures``. Each figure is skipped if its inputs are
empty, so partial runs still produce what they can.
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
    ax.legend(title=columns, fontsize=8, ncol=2)
    return pivot


def plot_encode(df, out: list[Path]):
    sub = _ok(df[df["task"] == "encode"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _grouped_bar(axes[0], sub, "workload", "backend", "metric_main")
    axes[0].set_ylabel("frames / s (higher is better)")
    axes[0].set_title("Encode throughput")
    _grouped_bar(axes[1], sub, "workload", "backend", "x_compression_ratio")
    axes[1].set_ylabel("JPEG-folder / video size (higher is better)")
    axes[1].set_title("Compression ratio vs per-frame JPEGs")
    out.append(_save(fig, "encode_throughput_compression.png"))


def plot_encode_pareto(df, out: list[Path]):
    sub = _ok(df[df["task"] == "encode_pareto"]).copy()
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
                g["x_compression_ratio"], g["metric_main"], marker="o", label=backend
            )
            for _, row in g.iterrows():
                ax.annotate(
                    f"{row['x_psnr_db']:.1f} dB",
                    (row["x_compression_ratio"], row["metric_main"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )
        ax.set_xlabel("compression ratio (JPEG folder / video) →")
        ax.set_ylabel("encode throughput (frames / s) →")
        ax.set_title(f"Speed vs compression Pareto — {workload}")
        ax.grid(alpha=0.3)
        ax.legend(title="encoder")
    out.append(_save(fig, "encode_pareto.png"))


def plot_random(df, out: list[Path]):
    sub = _ok(df[df["task"] == "random"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bar(ax, sub, "workload", "backend", "metric_main")
    ax.set_ylabel("frames / s (higher is better)")
    ax.set_title("Random-access throughput (precise seek; all seeks verified)")
    out.append(_save(fig, "random_throughput.png"))


def plot_sequential(df, out: list[Path]):
    sub = _ok(df[df["task"] == "sequential"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bar(ax, sub, "workload", "backend", "metric_main")
    ax.set_ylabel("frames / s (higher is better)")
    ax.set_title("Sequential decode throughput")
    out.append(_save(fig, "sequential_throughput.png"))


def plot_loc(df, out: list[Path]):
    sub = df[df["task"] == "loc"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bar(ax, sub, "workload", "backend", "metric_main")
    ax.set_ylabel("source lines of code (lower is better)")
    ax.set_title("User code per task (ruff-formatted, lint-clean)")
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
    plot_encode(df, out)
    plot_encode_pareto(df, out)
    plot_random(df, out)
    plot_sequential(df, out)
    plot_loc(df, out)
    return out


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p}")

"""Render figures from the consolidated benchmark result CSV.

Reads ``benchmark/results/results.csv`` (or an in-memory DataFrame) and writes
interactive HTML figures to ``benchmark/results/figures``. Figures are embedded
in the docs site via pymdownx.snippets; plotly.js is loaded from CDN.

Figures produced:

* ``encode_quality.html`` — throughput and compression vs PSNR (the quality
  sweep), per encoder.
* ``encode_matched.html`` — throughput and compression at matched PSNR (a fair
  single operating point, interpolated from the sweep).
* ``decode_throughput.html`` — sequential and random-access throughput across
  all decode workloads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis
from .config import FIGURES_DIR, MATCH_PSNR, RESULTS_DIR

_QUALITY_PARAM_LABEL = {"pvio_gpu": "QP", "opencv": "JPEG Q"}  # all others use CRF

_WORKLOAD_LABEL: dict[str, str] = {
    "enc_sd": "Encoding SD (848×480)",
    "enc_hd": "Encoding HD (1280×720)",
    "sd_h264": "SD (854×480)",
    "hd_h264": "HD (1280×720)",
    "fhd_h264": "Full HD (1920×1080)",
}

_BACKEND_LABEL: dict[str, str] = {
    "pvio_cpu":       "PVIO (CPU)",
    "pvio_gpu":       "PVIO (GPU)",
    "pyav":           "PyAV",
    "pyav_cpu":       "PyAV (CPU)",
    "opencv":         "OpenCV",
    "opencv_cpu":     "OpenCV (CPU)",
    "decord_cpu":     "Decord (CPU)",
    "torchcodec_cpu": "TorchCodec (CPU)",
    "torchcodec_cuda":"TorchCodec (CUDA)",
    "dali_gpu":       "DALI (GPU)",
}

# GPU-based backends (NVENC/NVDEC/DALI). All others are CPU.
_IS_GPU = frozenset({"pvio_gpu", "torchcodec_cuda", "dali_gpu"})

# Two-tone color scheme: PVIO anchor colors, uniform shade for all other CPU/GPU.
_CPU_DARK  = "#0c2e6e"   # PVIO CPU — very dark blue
_CPU_LIGHT = "#90c4e4"   # all other CPU — light blue
_GPU_DARK  = "#7a0042"   # PVIO GPU — dark magenta
_GPU_LIGHT = "#f0a0c4"   # all other GPU — light pink

_BACKEND_COLOR: dict[str, str] = {
    "pvio_cpu":        _CPU_DARK,
    "pvio_gpu":        _GPU_DARK,
    "pyav":            _CPU_LIGHT,
    "pyav_cpu":        _CPU_LIGHT,
    "opencv":          _CPU_LIGHT,
    "opencv_cpu":      _CPU_LIGHT,
    "decord_cpu":      _CPU_LIGHT,
    "torchcodec_cpu":  _CPU_LIGHT,
    "torchcodec_cuda": _GPU_LIGHT,
    "dali_gpu":        _GPU_LIGHT,
}


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that produced a finite measurement (drop skipped/errored backends)."""
    return df[df["error"].isna() & df["metric_main"].notna()]


def _pvio_first(backend: str) -> tuple:
    return (0 if backend.startswith("pvio") else 1, backend)


def _decode_order(backend: str) -> tuple:
    """CPU backends first (pvio first within each group), then GPU."""
    return (1 if backend in _IS_GPU else 0, 0 if backend.startswith("pvio") else 1, backend)


def _wl(name: str) -> str:
    return _WORKLOAD_LABEL.get(name, name)


def _be(name: str) -> str:
    return _BACKEND_LABEL.get(name, name)


def _color(backend: str) -> str:
    return _BACKEND_COLOR.get(backend, "#888888")


def _decode_workloads(df: pd.DataFrame) -> list[str]:
    """Decode workloads present in the data, ordered SD, HD, Full HD."""
    seen = set()
    for task in ("sequential", "random"):
        seen |= set(_ok(df[df["task"] == task])["workload"].unique())
    order = {"sd_h264": 0, "hd_h264": 1, "fhd_h264": 2}
    return sorted(seen, key=lambda w: (order.get(w, 99), w))


# Legend group keys and display names for the bar charts.
# All non-PVIO CPU backends share one legend entry; same for GPU.
def _legend_group(backend: str) -> str:
    if backend == "pvio_cpu":
        return "pvio_cpu"
    if backend == "pvio_gpu":
        return "pvio_gpu"
    return "other_gpu" if backend in _IS_GPU else "other_cpu"


_LEGEND_GROUP_NAME: dict[str, str] = {
    "pvio_cpu":  "PVIO (CPU)",
    "pvio_gpu":  "PVIO (GPU)",
    "other_cpu": "Other CPU",
    "other_gpu": "Other GPU",
}


def _save(fig: go.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=False)
    return path


def _bar_panel(
    fig,
    panel: pd.DataFrame,
    value_col: str,
    row: int,
    col: int,
    n_cols: int,
    all_backends: list[str],
    shown_lg: set[str],
    unit: str,
) -> None:
    """Draw one grouped-bar panel (one bar per backend) with CPU/GPU separator.

    Shared by the decode-throughput and memory figures; legend entries are
    grouped (PVIO CPU/GPU + Other CPU/GPU) and shown once across the figure.
    """
    backends = [b for b in all_backends if not panel[panel["backend"] == b].empty]
    n_cpu = sum(1 for b in backends if b not in _IS_GPU)
    for backend in backends:
        g = panel[panel["backend"] == backend]
        lg = _legend_group(backend)
        fig.add_trace(
            go.Bar(
                name=_LEGEND_GROUP_NAME[lg],
                x=[_be(backend)],
                y=[float(g[value_col].iloc[0])],
                marker_color=_color(backend),
                legendgroup=lg,
                showlegend=(lg not in shown_lg),
                hovertemplate=f"<b>{_be(backend)}</b><br>%{{y:.1f}} {unit}<extra></extra>",
            ),
            row=row, col=col,
        )
        shown_lg.add(lg)
    # Dashed separator between CPU and GPU groups when both are present.
    if 0 < n_cpu < len(backends):
        panel_idx = (row - 1) * n_cols + col
        xax = "x" if panel_idx == 1 else f"x{panel_idx}"
        yax = "y" if panel_idx == 1 else f"y{panel_idx}"
        fig.add_shape(
            type="line", x0=n_cpu - 0.5, x1=n_cpu - 0.5, y0=0, y1=1,
            xref=xax, yref=f"{yax} domain",
            line=dict(color="rgba(100,100,100,0.35)", width=1, dash="dot"),
        )


def plot_encode_quality(df, out: list[Path]):
    sub = _ok(df[df["task"] == "encode_pareto"]).copy()
    if sub.empty:
        return
    workloads = sorted(sub["workload"].unique(), key=lambda w: (0 if "sd" in w else 1, w))
    backends = sorted(sub["backend"].unique(), key=_pvio_first)
    metrics = [
        ("metric_main",        "Throughput (frames/s)"),
        ("x_compression_ratio","Compression ratio (JPEG folder / video)"),
    ]
    n_metrics = len(metrics)
    subplot_titles = [
        f"{_wl(wl)} — {label.split('(')[0].strip()}"
        for wl in workloads
        for _, label in metrics
    ]
    fig = make_subplots(rows=len(workloads) * n_metrics, cols=1, subplot_titles=subplot_titles)
    for row_idx, workload in enumerate(workloads, start=1):
        wl_display = _wl(workload)
        for backend in backends:
            g = (
                sub[(sub["workload"] == workload) & (sub["backend"] == backend)]
                .sort_values("x_psnr_db")
            )
            if g.empty:
                continue
            param = _QUALITY_PARAM_LABEL.get(backend, "CRF")
            display = _be(backend)
            color = _color(backend)
            cd = list(zip(g["x_quality_param"].astype(int), [wl_display] * len(g)))
            for metric_idx, (y_col, _) in enumerate(metrics, start=1):
                sub_row = (row_idx - 1) * n_metrics + metric_idx
                is_fps = y_col == "metric_main"
                fig.add_trace(
                    go.Scatter(
                        x=g["x_psnr_db"],
                        y=g[y_col],
                        mode="lines+markers",
                        name=display,
                        legendgroup=backend,
                        showlegend=(row_idx == 1 and metric_idx == 1),
                        line=dict(color=color),
                        marker=dict(color=color),
                        customdata=cd,
                        hovertemplate=(
                            f"<b>{display}</b><br>"
                            "PSNR: %{x:.1f} dB<br>"
                            f"Param: {param}=%{{customdata[0]}}<br>"
                            + ("Throughput: %{y:.0f} fps" if is_fps else "Compression ratio: %{y:.1f}×")
                            + "<extra></extra>"
                        ),
                    ),
                    row=sub_row, col=1,
                )
        for metric_idx, (_, y_label) in enumerate(metrics, start=1):
            sub_row = (row_idx - 1) * n_metrics + metric_idx
            fig.update_xaxes(title_text="PSNR (dB)", row=sub_row, col=1)
            fig.update_yaxes(title_text=y_label, row=sub_row, col=1)
    fig.update_layout(
        title="Encode throughput and compression ratio vs quality (PSNR)",
        hovermode="closest",
        height=350 * len(workloads) * n_metrics,
        legend_title_text="Backend",
    )
    out.append(_save(fig, "encode_quality.html"))


def plot_encode_matched(df, out: list[Path]):
    """Throughput and compression at matched PSNR — a fair single operating point."""
    matched = analysis.matched_psnr(df, MATCH_PSNR)
    if matched.empty:
        return
    workloads = sorted(matched["workload"].unique(), key=lambda w: (0 if "sd" in w else 1, w))
    all_backends = sorted(matched["backend"].unique(), key=_decode_order)
    metrics = [
        ("metric_main",         "Throughput (frames/s)", "fps"),
        ("x_compression_ratio", "Compression ratio (×)", "×"),
    ]
    n_metrics = len(metrics)
    subplot_titles = [
        f"{_wl(wl)} — {label.split('(')[0].strip()}"
        for wl in workloads
        for _, label, _ in metrics
    ]
    fig = make_subplots(rows=len(workloads) * n_metrics, cols=1, subplot_titles=subplot_titles)
    shown_lg: set[str] = set()
    for row_idx, workload in enumerate(workloads, start=1):
        panel = matched[matched["workload"] == workload]
        backends = [b for b in all_backends if not panel[panel["backend"] == b].empty]
        n_cpu = sum(1 for b in backends if b not in _IS_GPU)
        for metric_idx, (col_name, label, unit) in enumerate(metrics, start=1):
            sub_row = (row_idx - 1) * n_metrics + metric_idx
            for backend in backends:
                g = panel[panel["backend"] == backend]
                lg = _legend_group(backend)
                matched_ok = bool(g["x_psnr_matched"].iloc[0])
                eff = float(g["x_psnr_db"].iloc[0])
                fig.add_trace(
                    go.Bar(
                        name=_LEGEND_GROUP_NAME[lg],
                        x=[_be(backend)],
                        y=[float(g[col_name].iloc[0])],
                        marker_color=_color(backend),
                        legendgroup=lg,
                        showlegend=(lg not in shown_lg),
                        hovertemplate=(
                            f"<b>{_be(backend)}</b><br>%{{y:.1f}} {unit}<br>"
                            f"PSNR {eff:.1f} dB"
                            + ("" if matched_ok else " (off-target — knob did not reach it)")
                            + "<extra></extra>"
                        ),
                    ),
                    row=sub_row, col=1,
                )
                shown_lg.add(lg)
            if 0 < n_cpu < len(backends):
                xax = "x" if sub_row == 1 else f"x{sub_row}"
                yax = "y" if sub_row == 1 else f"y{sub_row}"
                fig.add_shape(
                    type="line", x0=n_cpu - 0.5, x1=n_cpu - 0.5, y0=0, y1=1,
                    xref=xax, yref=f"{yax} domain",
                    line=dict(color="rgba(100,100,100,0.35)", width=1, dash="dot"),
                )
            fig.update_yaxes(title_text=label, row=sub_row, col=1)
    tgt = matched["x_target_psnr"].iloc[0]
    label = f"{MATCH_PSNR:.0f} dB" if MATCH_PSNR > 0 else f"≈{tgt:.0f} dB (auto)"
    fig.update_layout(
        title=f"Encode at matched image quality (target PSNR {label})  ·  blue = CPU, pink = GPU",
        barmode="group",
        height=350 * len(workloads) * n_metrics,
        legend_title_text="Backend",
    )
    out.append(_save(fig, "encode_matched.html"))


def plot_decode(df, out: list[Path]):
    TASKS = [
        ("sequential", "Sequential"),
        ("random",     "Precise random-access"),
    ]
    workloads = _decode_workloads(df)
    if not workloads:
        return
    n_tasks = len(TASKS)
    subplot_titles = [
        f"{_wl(wl)} — {task_label}"
        for wl in workloads
        for _, task_label in TASKS
    ]
    n_rows = len(workloads) * n_tasks
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=subplot_titles)

    all_backends = sorted(
        {b for tk, _ in TASKS for b in _ok(df[df["task"] == tk])["backend"].unique()},
        key=_decode_order,
    )
    shown_lg: set[str] = set()
    for row_idx, workload in enumerate(workloads, start=1):
        for task_idx, (task_key, _) in enumerate(TASKS, start=1):
            sub_row = (row_idx - 1) * n_tasks + task_idx
            sub = _ok(df[df["task"] == task_key])
            panel = sub[sub["workload"] == workload]
            _bar_panel(fig, panel, "metric_main", sub_row, 1, 1,
                       all_backends, shown_lg, "fps")
            fig.update_yaxes(title_text="Throughput (frames/s)", row=sub_row, col=1)

    fig.update_layout(
        title="Decode throughput  (blue = CPU · pink = GPU)",
        hovermode="closest",
        height=320 * n_rows,
        legend_title_text="Backend",
    )
    out.append(_save(fig, "decode_throughput.html"))


def generate(df: pd.DataFrame | None = None) -> list[Path]:
    if df is None:
        csv = RESULTS_DIR / "results.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found; run the benchmark first.")
        df = pd.read_csv(csv)
    if "error" not in df:
        df["error"] = pd.NA
    out: list[Path] = []
    plot_encode_quality(df, out)
    plot_encode_matched(df, out)
    plot_decode(df, out)
    return out


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p}")

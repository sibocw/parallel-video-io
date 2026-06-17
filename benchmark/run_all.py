"""Run the full benchmark suite and write results, a markdown summary, and figures.

Usage::

    uv run python -m benchmark.run_all                 # everything
    uv run python -m benchmark.run_all --only loc read # subset of tasks
    uv run python -m benchmark.run_all --quick         # small, fast smoke run
    uv run python -m benchmark.run_all --no-figures

Results land in ``benchmark/results/`` (``results.csv``, ``environment.json``,
``SUMMARY.md``) and ``benchmark/results/figures/``.
"""

from __future__ import annotations

import argparse
import os

ALL_TASKS = ("read", "loading", "write", "write_pareto", "loc")


def _apply_quick_defaults() -> None:
    """Shrink the workload for a fast smoke run (only sets unset env vars)."""
    defaults = {
        "PVIO_BENCH_NFRAMES": "120",
        "PVIO_BENCH_N_COLLECTION_VIDEOS": "6",
        "PVIO_BENCH_COLLECTION_NFRAMES": "120",
        "PVIO_BENCH_COLLECTION_RES": "sd",
        "PVIO_BENCH_WORKER_SWEEP": "1,4",
        "PVIO_BENCH_N_REPEATS": "1",
        "PVIO_BENCH_N_RANDOM_READS": "50",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def _render_md_table(df) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([head, sep, *rows])


def _write_summary(df, path) -> None:
    ok = df[df["error"].isna() & df["metric_main"].notna()]
    lines = ["# Benchmark summary", ""]

    def section(title, body):
        lines.extend([f"## {title}", "", body, ""])

    if (s := ok[ok["task"] == "read_sequential"]).shape[0]:
        p = s.pivot_table(
            index="workload", columns="backend", values="metric_main"
        ).round(0)
        section(
            "Sequential decode (frames/s, higher better)",
            _render_md_table(p.reset_index()),
        )
    if (s := ok[ok["task"] == "read_random"]).shape[0]:
        p = s.pivot_table(
            index="workload", columns="backend", values="metric_main"
        ).round(2)
        section(
            "Random-access seek (ms/frame, lower better)",
            _render_md_table(p.reset_index()),
        )
        corr = s.groupby("backend")["x_seek_correct"].all()
        section("Seek correctness (all videos)", _render_md_table(corr.reset_index()))
    if (s := ok[ok["task"] == "loading"]).shape[0]:
        p = s.pivot_table(
            index="x_num_workers", columns="backend", values="metric_main"
        ).round(0)
        section(
            "Parallel loading (frames/s vs workers, higher better)",
            _render_md_table(p.reset_index()),
        )
    if (s := ok[ok["task"] == "write"]).shape[0]:
        cols = [
            "workload",
            "backend",
            "metric_main",
            "x_file_size_mb",
            "x_compression_ratio",
            "x_psnr_db",
            "x_ssim",
        ]
        section(
            "Write (speed / size / compression / quality)",
            _render_md_table(s[cols].round(3)),
        )
    if (s := ok[ok["task"] == "write_pareto"]).shape[0]:
        cols = [
            "workload",
            "backend",
            "x_quality_param",
            "metric_main",
            "x_compression_ratio",
            "x_psnr_db",
            "x_ssim",
        ]
        s = s.sort_values(["workload", "backend", "x_quality_param"])
        section(
            "Write Pareto sweep (throughput vs compression, per quality level)",
            _render_md_table(s[cols].round(3)),
        )
    if (s := df[df["task"] == "loc"]).shape[0]:
        p = s.pivot_table(index="workload", columns="backend", values="metric_main")
        section(
            "Lines of user code (SLOC, lower better)", _render_md_table(p.reset_index())
        )

    errs = df[df["error"].notna()]
    if errs.shape[0]:
        cols = ["task", "backend", "workload", "error"]
        section("Skipped / errored", _render_md_table(errs[cols]))

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=ALL_TASKS, default=list(ALL_TASKS))
    parser.add_argument("--quick", action="store_true", help="small, fast smoke run")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if args.quick:
        _apply_quick_defaults()

    # Import after env defaults so config picks them up.
    import pandas as pd

    from . import config
    from .common import Result, save_environment

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_environment(config.RESULTS_DIR / "environment.json")

    results: list[Result] = []
    if "read" in args.only:
        from . import bench_read

        print("== read ==")
        results += bench_read.run()
    if "loading" in args.only:
        from . import bench_loading

        print("== loading ==")
        results += bench_loading.run()
    if "write" in args.only:
        from . import bench_write

        print("== write ==")
        results += bench_write.run()
    if "write_pareto" in args.only:
        from . import bench_write_pareto

        print("== write_pareto ==")
        results += bench_write_pareto.run()
    if "loc" in args.only:
        from . import loc

        print("== loc ==")
        results += loc.run()

    df = pd.DataFrame([r.flat() for r in results])
    csv_path = config.RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df)} rows)")

    summary_path = config.RESULTS_DIR / "SUMMARY.md"
    _write_summary(df, summary_path)
    print(f"Wrote {summary_path}")

    if not args.no_figures:
        from . import plots

        figs = plots.generate(df)
        print(f"Wrote {len(figs)} figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()

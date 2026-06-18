"""Run the consolidated benchmark suite and write results, a summary, and figures.

Three tasks (encoding, random access, sequential access) are measured across the
video libraries on throughput, compression ratio (with PSNR/SSIM for matched-
quality comparison), and exact seek correctness.

Usage::

    uv run python -m benchmark.run_all                  # everything
    uv run python -m benchmark.run_all --only encode    # subset of tasks
    uv run python -m benchmark.run_all --quick          # small, fast smoke run
    uv run python -m benchmark.run_all --no-figures

Results land in ``benchmark/results/`` (``results.csv``, ``environment.json``,
``SUMMARY.md``) and ``benchmark/results/figures/``.
"""

from __future__ import annotations

import argparse
import os

ALL_TASKS = ("encode", "random", "sequential")


def _apply_quick_defaults() -> None:
    """Shrink the workload for a fast smoke run (only sets unset env vars)."""
    defaults = {
        "PVIO_BENCH_NFRAMES": "120",
        "PVIO_BENCH_ENCODE_NFRAMES": "120",
        "PVIO_BENCH_N_REPEATS": "1",
        "PVIO_BENCH_N_RANDOM_READS": "50",
        "PVIO_BENCH_QUALITY_SWEEP": "17,21,25",
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
    from . import analysis, config

    ok = df[df["error"].isna() & df["metric_main"].notna()]
    lines = ["## Benchmark result tables", ""]

    def section(title, body):
        indented = "\n".join("    " + line for line in body.split("\n"))
        lines.extend([f'??? info "{title}"', "", indented, ""])

    matched = analysis.matched_psnr(df, config.MATCH_PSNR)
    if matched.shape[0]:
        cols = [
            "workload",
            "backend",
            "x_target_psnr",
            "x_psnr_db",
            "x_psnr_matched",
            "metric_main",
            "x_compression_ratio",
            "x_file_size_mb",
        ]
        matched = matched.sort_values(["workload", "backend"])
        section(
            "Encoding at matched PSNR (frames/s and compression at equal image "
            "quality, interpolated from the sweep; x_psnr_matched=False means the "
            "encoder's sweep did not reach the target and the nearest point is shown)",
            _render_md_table(matched[cols]),
        )
    if (s := ok[ok["task"] == "encode_pareto"]).shape[0]:
        cols = [
            "workload",
            "backend",
            "x_quality_param",
            "metric_main",
            "x_compression_ratio",
            "x_psnr_db",
        ]
        s = s.sort_values(["workload", "backend", "x_quality_param"])
        section(
            "Encoding Pareto sweep (throughput vs compression per quality level)",
            _render_md_table(s[cols].round(3)),
        )
    if (s := ok[ok["task"] == "random"]).shape[0]:
        p = s.pivot_table(
            index="workload", columns="backend", values="metric_main"
        ).round(0)
        section(
            "Random access (frames/s, higher better)", _render_md_table(p.reset_index())
        )
        corr = s.groupby("backend")["x_seek_correct"].all()
        section("Seek correctness (all videos)", _render_md_table(corr.reset_index()))
    if (s := ok[ok["task"] == "sequential"]).shape[0]:
        p = s.pivot_table(
            index="workload", columns="backend", values="metric_main"
        ).round(0)
        section(
            "Sequential access (frames/s, higher better)",
            _render_md_table(p.reset_index()),
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
    if "encode" in args.only:
        from . import bench_encode

        print("== encode ==")
        results += bench_encode.run()
    if "random" in args.only:
        from . import bench_random

        print("== random ==")
        results += bench_random.run()
    if "sequential" in args.only:
        from . import bench_sequential

        print("== sequential ==")
        results += bench_sequential.run()
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

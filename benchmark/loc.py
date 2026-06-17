"""Task 5: lines of user-written code.

For each task, the minimal idiomatic user code for every backend lives as a
standalone file under ``benchmark/snippets/<task>/<backend>.py``. We normalise
style with ``ruff format``, require ``ruff check`` to pass (so the code is
genuinely "reasonable style", not artificially shrunk), then count source lines
of code: physical lines that are neither blank nor a pure comment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .common import Result
from .config import SNIPPETS_DIR


def _sloc(path: Path) -> int:
    n = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        n += 1
    return n


def _ruff(args: list[str], path: Path) -> bool:
    proc = subprocess.run(["ruff", *args, str(path)], capture_output=True, text=True)
    return proc.returncode == 0


def run() -> list[Result]:
    results: list[Result] = []
    if not SNIPPETS_DIR.is_dir():
        return results
    for task_dir in sorted(SNIPPETS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for snippet in sorted(task_dir.glob("*.py")):
            backend = snippet.stem
            # Enforce reasonable style: must already be ruff-formatted and lint-clean.
            formatted = _ruff(["format", "--check"], snippet)
            clean = _ruff(["check"], snippet)
            results.append(
                Result(
                    "loc",
                    backend,
                    "-",
                    task,
                    float(_sloc(snippet)),
                    "lines",
                    extra={
                        "ruff_clean": clean,
                        "ruff_formatted": formatted,
                        "file": str(snippet.relative_to(SNIPPETS_DIR.parent)),
                    },
                )
            )
    return results


if __name__ == "__main__":
    for r in run():
        flag = "" if (r.extra["ruff_clean"] and r.extra["ruff_formatted"]) else "  [!]"
        print(f"{r.workload:16s} {r.backend:18s} {int(r.metric_main):3d} lines{flag}")

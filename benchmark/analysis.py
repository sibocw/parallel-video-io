"""Derived metrics computed from the raw benchmark results.

The encoders cannot be compared at a shared quality number — libx264's CRF and
NVENC's QP are different operating points on the same 0-51 scale. So the fair
single-number comparison is at *matched image quality*: interpolate each
encoder's quality sweep to a common PSNR and read off its throughput and
compression there. :func:`matched_psnr` does exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# An encoder counts as "tunable in practice" only if sweeping its quality knob
# actually moves PSNR by at least this much; otherwise (e.g. OpenCV MJPEG whose
# quality knob is a no-op in many builds) it is a single fixed operating point
# and must not constrain where the matched-PSNR target is placed.
_VARYING_DB = 0.3


def _finite(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["error"].isna() & df["metric_main"].notna()]


def _loginterp(x: float, xp, fp) -> float:
    """Interpolate a strictly-positive quantity in log space.

    File size, compression ratio and throughput vary roughly geometrically with
    quality (bitrate is ~exponential in PSNR), so the curve is far closer to a
    straight line in log space. Linear interpolation across the steep raw curve
    would otherwise amplify a small PSNR offset into a large apparent gap.
    """
    return float(np.exp(np.interp(x, xp, np.log(np.maximum(fp, 1e-12)))))


def matched_psnr(df: pd.DataFrame, target: float = 0.0) -> pd.DataFrame:
    """Compare encoders at matched PSNR, interpolated from the quality sweep.

    Args:
        df: the full results frame (uses ``task == "encode_pareto"`` rows).
        target: target PSNR in dB. ``<= 0`` picks, per workload, the midpoint of
            the PSNR range reachable by the encoders that actually vary, so the
            comparison sits where their sweeps overlap.

    Returns:
        One row per (workload, backend) with throughput, compression ratio and
        file size at the matched PSNR. Each row also carries ``x_target_psnr``
        (the shared target), ``x_psnr_db`` (the backend's *effective* PSNR at
        the reported point, which differs if its sweep does not reach the
        target) and ``x_psnr_matched`` (within 0.5 dB of the target). Empty if
        there are no sweep rows.
    """
    sub = _finite(df[df["task"] == "encode_pareto"])
    if sub.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for workload, wl in sub.groupby("workload"):
        groups = {b: g.sort_values("x_psnr_db") for b, g in wl.groupby("backend")}
        ranges = {
            b: (g["x_psnr_db"].min(), g["x_psnr_db"].max()) for b, g in groups.items()
        }
        # Place the target using only encoders whose sweep meaningfully varies.
        varying = {b for b, (lo, hi) in ranges.items() if hi - lo >= _VARYING_DB}
        basis = varying or set(groups)
        lo = max(ranges[b][0] for b in basis)
        hi = min(ranges[b][1] for b in basis)
        if lo <= hi:
            tgt = float(np.clip(target, lo, hi)) if target > 0 else (lo + hi) / 2.0
        else:  # varying encoders' sweeps don't overlap — use the median midpoint
            tgt = float(np.median([(ranges[b][0] + ranges[b][1]) / 2 for b in basis]))

        for backend, g in groups.items():
            xp = g["x_psnr_db"].to_numpy()
            eff = float(np.clip(tgt, xp.min(), xp.max()))  # nearest reachable PSNR
            rows.append(
                {
                    "workload": workload,
                    "backend": backend,
                    "x_target_psnr": round(tgt, 2),
                    "x_psnr_db": round(eff, 2),
                    "x_psnr_matched": bool(abs(eff - tgt) <= 0.5),
                    "metric_main": round(_loginterp(tgt, xp, g["metric_main"]), 1),
                    "x_compression_ratio": round(
                        _loginterp(tgt, xp, g["x_compression_ratio"]), 1
                    ),
                    "x_file_size_mb": round(
                        _loginterp(tgt, xp, g["x_file_size_mb"]), 2
                    ),
                }
            )
    return pd.DataFrame(rows)

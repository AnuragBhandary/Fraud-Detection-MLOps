"""Population stability index (PSI) and a weekly drift + performance report.

PSI compares how a feature is distributed in a window against the training period:
``sum((cur - ref) * ln(cur / ref))`` over bins cut at the reference deciles, with missing values
as their own bin. The usual reading: < 0.1 stable, 0.1-0.2 watch, > 0.2 investigate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from fraudops.config import DAY_SECONDS, TARGET, TIME

EPS = 1e-4


def _bins(ref: np.ndarray, n_bins: int) -> np.ndarray:
    finite = ref[~np.isnan(ref)]
    if finite.size == 0:
        return np.array([])
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.asarray(edges)


def _shares(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    missing = np.isnan(x)
    idx = np.searchsorted(edges, x[~missing], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    return np.asarray(counts / max(len(x), 1))


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    edges = _bins(ref, n_bins)
    r = np.clip(_shares(ref, edges), EPS, None)
    c = np.clip(_shares(cur, edges), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def weekly_report(
    ref: pl.DataFrame,
    cur: pl.DataFrame,
    features: Sequence[str],
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
) -> pl.DataFrame:
    """One row per 7-day window of ``cur``: score PSI, fraud rate, PR-AUC, worst feature PSIs."""
    from fraudops.metrics import pr_auc

    week = (cur[TIME].to_numpy() // DAY_SECONDS) // 7
    ref_cols = {f: ref[f].cast(pl.Float64).to_numpy() for f in features}
    rows = []
    for w in np.unique(week):
        mask = week == w
        if mask.sum() < 500:
            continue
        sub = cur.filter(pl.Series(mask))
        y = sub[TARGET].to_numpy()
        feat_psi = {f: psi(ref_cols[f], sub[f].cast(pl.Float64).to_numpy()) for f in features}
        worst = sorted(feat_psi.items(), key=lambda kv: -kv[1])[:3]
        rows.append(
            {
                "week": int(w),
                "transactions": int(mask.sum()),
                "fraud_rate_pct": round(100 * float(y.mean()), 2),
                "pr_auc": round(pr_auc(y, cur_scores[mask]), 4) if y.any() else None,
                "score_psi": round(psi(ref_scores, cur_scores[mask]), 4),
                "features_psi_over_0.2": sum(v > 0.2 for v in feat_psi.values()),
                "worst_features": ", ".join(f"{k} {v:.2f}" for k, v in worst),
            }
        )
    return pl.DataFrame(rows)

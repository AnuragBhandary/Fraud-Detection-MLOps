"""Ranking metrics, the dollar cost of a decision rule, and bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from fraudops.config import COSTS, Costs

Array = np.ndarray


def pr_auc(y: Array, p: Array) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y: Array, p: Array) -> float:
    return float(roc_auc_score(y, p))


def recall_at_fpr(y: Array, p: Array, fpr: float = 0.01) -> float:
    """Share of fraud caught when at most ``fpr`` of good transactions are flagged."""
    f, t, _ = roc_curve(y, p)
    return float(np.max(t[f <= fpr]))


def ranking_metrics(y: Array, p: Array) -> dict[str, float]:
    return {
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
    }


def decision_cost(y: Array, flag: Array, amount: Array, costs: Costs = COSTS) -> float:
    """Reviews cost ``costs.review`` each; every unflagged fraud is charged back."""
    missed = (y == 1) & ~flag
    return float(costs.review * flag.sum() + (amount[missed] + costs.chargeback_fee).sum())


def cost_report(y: Array, flag: Array, amount: Array, costs: Costs = COSTS) -> dict[str, float]:
    no_model = decision_cost(y, np.zeros_like(flag, dtype=bool), amount, costs)
    cost = decision_cost(y, flag, amount, costs)
    fraud_dollars = amount[y == 1].sum()
    caught_dollars = amount[(y == 1) & flag].sum()
    return {
        "cost": cost,
        "cost_no_model": no_model,
        "saved_pct": 100.0 * (1 - cost / no_model),
        "flag_rate_pct": 100.0 * float(flag.mean()),
        "precision": float(y[flag].mean()) if flag.any() else 0.0,
        "recall": float(flag[y == 1].mean()),
        "fraud_dollars_caught_pct": 100.0 * float(caught_dollars / fraud_dollars),
    }


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    cost: float


def best_threshold(y: Array, p: Array, amount: Array, costs: Costs = COSTS) -> ThresholdChoice:
    """Score threshold with the lowest total cost (exact: tries every distinct score)."""
    order = np.argsort(-p, kind="stable")
    ys, ps, amts = y[order], p[order], amount[order]
    fraud_loss = np.where(ys == 1, amts + costs.chargeback_fee, 0.0)
    # Flagging the top i rows: i reviews plus every fraud loss below row i.
    total_loss = fraud_loss.sum()
    caught = np.concatenate([[0.0], np.cumsum(fraud_loss)])
    flagged = np.arange(len(ys) + 1)
    cost = costs.review * flagged + (total_loss - caught)
    # Only cut between distinct scores (ties must share a decision).
    valid = np.concatenate([[True], ps[:-1] != ps[1:], [True]])
    cost = np.where(valid, cost, np.inf)
    i = int(np.argmin(cost))
    thr = float(np.inf) if i == 0 else float(ps[i - 1])
    return ThresholdChoice(threshold=thr, cost=float(cost[i]))


def expected_cost_flags(p: Array, amount: Array, costs: Costs = COSTS) -> Array:
    """Flag when the expected chargeback beats the review cost (needs calibrated ``p``)."""
    return np.asarray(p * (amount + costs.chargeback_fee) > costs.review)


def bootstrap_ci(
    stat: Callable[[Array], float],
    n: int,
    reps: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of ``stat(idx)`` over row resamples."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(reps)])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

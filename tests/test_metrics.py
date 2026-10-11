from __future__ import annotations

import numpy as np

from fraudops.config import Costs
from fraudops.drift import psi
from fraudops.metrics import (
    best_threshold,
    bootstrap_ci,
    cost_report,
    decision_cost,
    expected_cost_flags,
    recall_at_fpr,
)

COSTS = Costs(review=10.0, chargeback_fee=20.0)


def test_decision_cost_by_hand() -> None:
    y = np.array([1, 1, 0, 0])
    amt = np.array([100.0, 5.0, 50.0, 50.0])
    flag = np.array([True, False, True, False])
    # 2 reviews ($20) + missed fraud of $5 (+ $20 fee)
    assert decision_cost(y, flag, amt, COSTS) == 45.0
    rep = cost_report(y, flag, amt, COSTS)
    assert rep["cost_no_model"] == 145.0  # (100 + 20) + (5 + 20)
    assert rep["precision"] == 0.5 and rep["recall"] == 0.5


def test_best_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.1).astype(int)
    p = np.round(rng.random(400) * 0.5 + 0.4 * y, 2)  # rounded -> many ties
    amt = rng.lognormal(4, 1, 400)
    choice = best_threshold(y, p, amt, COSTS)
    brute = min(
        [decision_cost(y, p >= t, amt, COSTS) for t in np.unique(p)]
        + [decision_cost(y, np.zeros(400, bool), amt, COSTS)]
    )
    assert np.isclose(choice.cost, brute)
    assert np.isclose(decision_cost(y, p >= choice.threshold, amt, COSTS), brute)


def test_flag_nothing_when_reviews_never_pay() -> None:
    y = np.array([0, 0, 1])
    choice = best_threshold(y, np.array([0.9, 0.8, 0.1]), np.array([1.0, 1.0, 1.0]), Costs(500, 0))
    assert choice.threshold == np.inf


def test_expected_cost_rule() -> None:
    flags = expected_cost_flags(np.array([0.1, 0.1, 0.5]), np.array([50.0, 200.0, 1.0]), COSTS)
    # 0.1 * 70 = 7 < 10; 0.1 * 220 = 22 > 10; 0.5 * 21 = 10.5 > 10
    assert flags.tolist() == [False, True, True]


def test_recall_at_fpr() -> None:
    y = np.array([0] * 100 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.45, 1, 10)])
    assert recall_at_fpr(y, p, 0.0) == 0.9  # one fraud scores below the top negative


def test_bootstrap_ci_covers_mean() -> None:
    x = np.random.default_rng(0).normal(5, 1, 2000)
    lo, hi = bootstrap_ci(lambda idx: float(x[idx].mean()), len(x), reps=200)
    assert lo < 5 < hi and hi - lo < 0.2


def test_psi() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(ref, rng.normal(1, 1, 20_000)) > 0.5
    with_nans = np.where(rng.random(20_000) < 0.5, np.nan, rng.normal(0, 1, 20_000))
    assert psi(ref, with_nans) > 1  # a column going half-missing is drift too
    assert psi(np.full(10, np.nan), np.full(10, np.nan)) == 0.0

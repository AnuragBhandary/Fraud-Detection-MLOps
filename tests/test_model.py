from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fraudops.model import FraudModel, bundle, predict, prepare, train_lgbm, train_logistic, tune
from fraudops.split import time_split


@pytest.fixture(scope="module")
def trained(features: pl.DataFrame) -> tuple[FraudModel, object]:
    s = time_split(features)
    m = prepare(s.train, s.val, s.test)
    booster = train_lgbm(m, max_rounds=200, patience=20)
    return bundle(m, booster, version="t"), m


def test_time_split_is_ordered(features: pl.DataFrame) -> None:
    s = time_split(features)
    assert s.train["TransactionDT"].max() < s.val["TransactionDT"].min()  # type: ignore[operator]
    assert s.val["TransactionDT"].max() < s.test["TransactionDT"].min()  # type: ignore[operator]
    assert s.train.height + s.val.height + s.test.height == features.height


def test_model_beats_chance_and_calibrates(trained: tuple[FraudModel, object]) -> None:
    model, m = trained
    p = model.probability(m.X_test)  # type: ignore[attr-defined]
    y = m.y_test  # type: ignore[attr-defined]
    from fraudops.metrics import pr_auc

    assert pr_auc(y, model.raw_score(m.X_test)) > 2 * y.mean()  # type: ignore[attr-defined]
    assert p.min() >= 0 and p.max() <= 1
    # Isotonic calibration: average predicted close to observed rate.
    assert abs(p.mean() - y.mean()) < 0.03


def test_contributions_add_up_to_the_margin(trained: tuple[FraudModel, object]) -> None:
    model, m = trained
    X = m.X_test[:50]  # type: ignore[attr-defined]
    contrib = model.contributions(X)
    margin = np.log(model.raw_score(X) / (1 - model.raw_score(X)))
    assert np.allclose(contrib.sum(axis=1), margin, atol=1e-6)


def test_native_treeshap_equals_shap_library(trained: tuple[FraudModel, object]) -> None:
    shap = pytest.importorskip("shap")
    model, m = trained
    X = m.X_test[:40]  # type: ignore[attr-defined]
    ours = model.contributions(X)[:, :-1]
    theirs = shap.TreeExplainer(model.booster).shap_values(X)
    theirs = theirs[1] if isinstance(theirs, list) else theirs
    assert np.allclose(ours, theirs, atol=1e-5)


def test_reasons_are_ranked(trained: tuple[FraudModel, object]) -> None:
    model, m = trained
    reasons = model.reasons(m.X_test[0], k=4)  # type: ignore[attr-defined]
    assert len(reasons) == 4
    mags = [abs(r["contribution"]) for r in reasons]
    assert mags == sorted(mags, reverse=True)


def test_bundle_pickles_without_live_booster(trained: tuple[FraudModel, object]) -> None:
    import pickle

    model, m = trained
    _ = model.booster
    clone = pickle.loads(pickle.dumps(model))
    assert clone._booster is None
    X = m.X_val[:10]  # type: ignore[attr-defined]
    assert np.array_equal(clone.probability(X), model.probability(X))


def test_logistic_and_tuning_run(features: pl.DataFrame) -> None:
    s = time_split(features)
    m = prepare(s.train, s.val, s.test, drop=("V1",))
    assert "V1" not in m.features
    lr = train_logistic(m)
    assert lr.predict_proba(m.X_val).shape == (len(m.y_val), 2)
    study = tune(m, n_trials=3)
    assert len(study.trials) == 3
    assert predict(train_lgbm(m, study.best_params, max_rounds=50), m.X_val).shape == m.y_val.shape

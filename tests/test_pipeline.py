"""The CLI steps end to end on synthetic data (each writes docs/results/<step>.json)."""

from __future__ import annotations

import json

import polars as pl

from fraudops import pipeline
from fraudops.cli import main
from fraudops.config import RESULTS_DIR


def test_full_pipeline(features: pl.DataFrame, transactions: pl.DataFrame) -> None:
    main(["train", "--trials", "3"])
    train = json.loads((RESULTS_DIR / "train.json").read_text())
    assert train["registry"]["promoted_to_champion"] in (True, False)
    d = train["decisions"]
    assert d["flag_nothing"]["saved_pct"] == 0
    assert d["expected_cost_rule"]["cost"] <= d["flag_nothing"]["cost"]
    assert len(train["shap_top20"]) == 20

    main(["build-store"])
    assert json.loads((RESULTS_DIR / "store.json").read_text())["replayed"] > 0

    main(["drift"])
    drift = json.loads((RESULTS_DIR / "drift.json").read_text())
    assert drift["weekly"] and drift["features_total"] == train["features"]

    main(["bench", "-n", "200"])
    bench = json.loads((RESULTS_DIR / "bench.json").read_text())
    assert bench["parity_rows_differing"] == 0


def test_leakage_step(features: pl.DataFrame, transactions: pl.DataFrame) -> None:
    out = pipeline.leakage()
    assert set(out) == {
        "past_only_features/time_split",
        "past_only_features/random_split",
        "future_peeking_features/time_split",
        "future_peeking_features/random_split",
    }


def test_describe(features: pl.DataFrame) -> None:
    d = pipeline.describe(features)
    assert d["rows"] == sum(p["rows"] for p in d["split"].values())

"""Point the whole package at a throwaway home before anything imports it."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_HOME = Path(tempfile.mkdtemp(prefix="fraudops-test-"))
os.environ["FRAUDOPS_HOME"] = str(_HOME)
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_HOME / 'mlflow.db'}"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ.setdefault("FRAUDOPS_THREADS", "2")

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from fraudops.config import FEATURES_PARQUET, TRANSACTIONS_PARQUET  # noqa: E402
from fraudops.features import build_features  # noqa: E402
from fraudops.synth import make_transactions  # noqa: E402


@pytest.fixture(scope="session")
def transactions() -> pl.DataFrame:
    df = make_transactions(n_customers=2500, days=120, seed=1)
    TRANSACTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(TRANSACTIONS_PARQUET)
    return df


@pytest.fixture(scope="session")
def features(transactions: pl.DataFrame) -> pl.DataFrame:
    f = build_features(transactions)
    f.write_parquet(FEATURES_PARQUET)
    return f

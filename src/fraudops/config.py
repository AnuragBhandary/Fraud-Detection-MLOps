"""Paths, column groups and the business numbers every other module shares."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("FRAUDOPS_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSACTIONS_PARQUET = DATA_DIR / "transactions.parquet"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "docs" / "results"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MODEL_NAME = "fraud-detector"
CHAMPION_ALIAS = "champion"

# Laptop-friendly: every model and BLAS call stays on four threads.
N_THREADS = int(os.environ.get("FRAUDOPS_THREADS", "4"))
SEED = 7

ID = "TransactionID"
TARGET = "isFraud"
TIME = "TransactionDT"  # seconds since an unpublished reference instant
AMOUNT = "TransactionAmt"
DAY_SECONDS = 86_400

CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *[f"M{i}" for i in range(1, 10)],
    "DeviceType",
    "DeviceInfo",
    *[f"id_{i}" for i in range(12, 39)],
]


@dataclass(frozen=True)
class Costs:
    """What one decision costs the issuer, in US dollars (the dataset's currency).

    A flagged transaction goes to a manual review (analyst time plus friction for a good customer).
    A missed fraud is charged back: the issuer loses the amount plus a dispute fee.
    """

    review: float = 10.0
    chargeback_fee: float = 20.0

    def missed(self, amount: float) -> float:
        return amount + self.chargeback_fee


COSTS = Costs()

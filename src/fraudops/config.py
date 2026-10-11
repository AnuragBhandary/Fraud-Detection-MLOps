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
# A fraud label (chargeback) is assumed known this many days after the transaction; before that,
# the transaction's outcome is unknown to the model. Sensitivity: `fraudops label-delay`.
LABEL_DELAY_DAYS = 30


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

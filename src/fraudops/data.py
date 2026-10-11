"""Raw Kaggle CSVs -> one typed Parquet file.

Only the labelled half of IEEE-CIS is used (``train_*``): the competition's test labels were never
published, so the project carves its own validation and test periods out of it by time
(:mod:`fraudops.split`).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from fraudops.config import AMOUNT, ID, RAW_DIR, TARGET, TIME, TRANSACTIONS_PARQUET


def _csv(raw_dir: Path, name: str) -> Path:
    """Return the CSV, unzipping it first if Kaggle delivered ``name.zip``."""
    csv = raw_dir / name
    if not csv.exists():
        archive = raw_dir / f"{name}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"{csv} (or .zip) missing: run `make data` first")
        with zipfile.ZipFile(archive) as z:
            z.extract(name, raw_dir)
    return csv


def load_raw(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    txn = pl.read_csv(_csv(raw_dir, "train_transaction.csv"), infer_schema_length=20_000)
    ident = pl.read_csv(_csv(raw_dir, "train_identity.csv"), infer_schema_length=20_000)
    df = txn.join(ident, on=ID, how="left")
    # Halve memory: measurements to float32; keep the amount in float64 (money, and history sums).
    floats = [c for c, t in df.schema.items() if t == pl.Float64 and c != AMOUNT]
    return df.with_columns(
        pl.col(floats).cast(pl.Float32),
        pl.col(ID).cast(pl.Int64),
        pl.col(TIME).cast(pl.Int64),
        pl.col(TARGET).cast(pl.Int8),
    ).sort([TIME, ID])


def write_parquet(raw_dir: Path = RAW_DIR, out: Path = TRANSACTIONS_PARQUET) -> pl.DataFrame:
    df = load_raw(raw_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return df


def read_transactions(path: Path = TRANSACTIONS_PARQUET) -> pl.DataFrame:
    return pl.read_parquet(path)

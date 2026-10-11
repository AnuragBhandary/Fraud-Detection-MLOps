from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from fraudops.data import load_raw, write_parquet


def test_load_raw_joins_and_types(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "TransactionID": [2, 1],
            "isFraud": [0, 1],
            "TransactionDT": [20, 10],
            "TransactionAmt": [1.5, 2.25],
            "V1": [1.0, None],
            "card4": ["visa", None],
        }
    ).write_csv(tmp_path / "train_transaction.csv")
    pl.DataFrame({"TransactionID": [1], "id_01": [-5.0], "DeviceType": ["mobile"]}).write_csv(
        tmp_path / "ident.csv"
    )
    with zipfile.ZipFile(tmp_path / "train_identity.csv.zip", "w") as z:
        z.write(tmp_path / "ident.csv", "train_identity.csv")
    df = load_raw(tmp_path)
    assert df["TransactionID"].to_list() == [1, 2]  # stream order
    assert df.schema["V1"] == pl.Float32 and df.schema["TransactionAmt"] == pl.Float64
    assert df["DeviceType"].to_list() == ["mobile", None]
    out = tmp_path / "t.parquet"
    write_parquet(tmp_path, out)
    assert pl.read_parquet(out).height == 2


def test_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)

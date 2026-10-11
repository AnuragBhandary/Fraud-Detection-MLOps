"""Out-of-time split: train on the past, tune on the next period, report on the one after.

A random split puts tomorrow's transactions of the same card in the training set, which makes
the model look better than it will ever be in production. ``fraudops leakage`` measures by how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fraudops.config import DAY_SECONDS, TIME


@dataclass(frozen=True)
class TimeSplit:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    val_start_day: int
    test_start_day: int


def time_split(df: pl.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> TimeSplit:
    """Cut on whole days so one day never straddles two sets."""
    day = pl.col(TIME) // DAY_SECONDS
    days = df.select(day.alias("d"))["d"]
    val_start = int(days.quantile(1 - val_frac - test_frac, interpolation="nearest") or 0)
    test_start = int(days.quantile(1 - test_frac, interpolation="nearest") or 0)
    return TimeSplit(
        train=df.filter(day < val_start),
        val=df.filter((day >= val_start) & (day < test_start)),
        test=df.filter(day >= test_start),
        val_start_day=val_start,
        test_start_day=test_start,
    )

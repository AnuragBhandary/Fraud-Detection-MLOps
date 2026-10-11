"""Offline feature engineering.

Three kinds of features:

* **Row features** that only read the transaction itself (hour, cents, log amount, account
  start day). Identical offline and online by construction.
* **History features** per entity (a card, and a reconstructed customer ``uid``): how many
  transactions came before, their mean and spread of amount, time since the previous one and how
  many arrived in the last 24 hours. "Before" means strictly earlier in the stream order
  ``(TransactionDT, TransactionID)``, so a row never sees itself or anything after it. The
  online store in :mod:`fraudops.online` computes the same numbers one transaction at a time and
  a test replays the stream through it to prove they match.
* **Known-fraud features** per entity: how many of its earlier transactions have a *matured*
  label (older than ``LABEL_DELAY_DAYS``) and how many of those were fraud. A chargeback takes
  weeks to arrive, so a label is only usable once it is that old; younger transactions count as
  unknown, never as legitimate.
* **Category codes** fitted on the training period only (:class:`CategoryEncoder`).

``history_features(mode="global")`` exists only for the leakage experiment: it computes the same
aggregates over the whole dataset, future included, the way many public Kaggle notebooks do.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from fraudops.config import AMOUNT, DAY_SECONDS, ID, LABEL_DELAY_DAYS, TARGET, TIME

HISTORY_KEYS = ("card1", "uid")
HISTORY_STATS = (
    "n_prior",
    "amt_mean_prior",
    "amt_std_prior",
    "amt_ratio",
    "secs_since_prev",
    "n_24h",
)
LABEL_STATS = ("labels_matured", "fraud_known", "fraud_rate_known")
# account_start is the account's start *day*: a calendar date in disguise (PSI 1.8 train vs test).
# It is kept to build the uid but is not a model input; the known-fraud features carry what it
# was standing in for (see docs/DESIGN.md section 3).
NOT_FEATURES = {ID, TARGET, TIME, "day", "uid", "account_start"}


def history_columns(keys: Iterable[str] = HISTORY_KEYS) -> list[str]:
    return [f"{k}_{s}" for k in keys for s in HISTORY_STATS]


def label_columns(keys: Iterable[str] = HISTORY_KEYS) -> list[str]:
    return [f"{k}_{s}" for k in keys for s in LABEL_STATS]


def row_features(df: pl.DataFrame) -> pl.DataFrame:
    """Columns derived from a single transaction (no other rows involved)."""
    day = pl.col(TIME) // DAY_SECONDS
    out = df.with_columns(
        day=day,
        hour=((pl.col(TIME) // 3600) % 24).cast(pl.Int8),
        dow=(day % 7).cast(pl.Int8),
        amt_log=pl.col(AMOUNT).log1p().cast(pl.Float32),
        amt_cents=((pl.col(AMOUNT) * 100).round(0) % 100).cast(pl.Float32),
        # D1 = days since the card was first seen, so day - D1 is a stable account start day.
        account_start=(day - pl.col("D1")).cast(pl.Float32),
    )
    return out.with_columns(
        uid=pl.when(pl.col("addr1").is_not_null() & pl.col("account_start").is_not_null())
        .then(
            pl.concat_str(
                [
                    pl.col("card1").cast(pl.Int64).cast(pl.String),
                    pl.col("addr1").cast(pl.Int64).cast(pl.String),
                    pl.col("account_start").cast(pl.Int64).cast(pl.String),
                ],
                separator="_",
            )
        )
        .otherwise(None)
    )


def _key_codes(values: pl.Series) -> np.ndarray:
    """Dense int codes for a key column; -1 for null."""
    codes = values.cast(pl.String).cast(pl.Categorical).to_physical().cast(pl.Int64)
    return codes.fill_null(-1).to_numpy()


def _count_last_24h(key: np.ndarray, t: np.ndarray, pos_in_stream: np.ndarray) -> np.ndarray:
    """For every row: earlier rows (stream order) of the same key with time > t - 1 day."""
    order = np.lexsort((pos_in_stream, key))  # by key, then stream position
    k_sorted = key[order]
    t_sorted = t[order]
    combined = k_sorted.astype(np.int64) * (1 << 40) + t_sorted
    lower = np.searchsorted(combined, k_sorted * (1 << 40) + t_sorted - DAY_SECONDS, side="right")
    counts = np.arange(len(order)) - lower
    out = np.empty(len(order), dtype=np.int64)
    out[order] = counts
    return out


def history_features(
    df: pl.DataFrame,
    keys: Sequence[str] = HISTORY_KEYS,
    mode: Literal["prior", "global"] = "prior",
) -> pl.DataFrame:
    """Add per-entity history features. ``df`` must already have :func:`row_features`."""
    df = df.sort([TIME, ID]).with_row_index("_pos")
    amt = pl.col(AMOUNT).cast(pl.Float64)
    t = df[TIME].cast(pl.Int64).to_numpy()
    pos = df["_pos"].to_numpy()
    for k in keys:
        valid = pl.col(k).is_not_null()
        if mode == "prior":
            n = pl.int_range(pl.len()).over(k)
            s = amt.cum_sum().shift(1).over(k)
            ss = (amt * amt).cum_sum().shift(1).over(k)
            prev_t = pl.col(TIME).shift(1).over(k)
        else:  # whole dataset, future included: the leak public notebooks often have
            n = pl.len().over(k) - 1
            s = amt.sum().over(k) - amt
            ss = (amt * amt).sum().over(k) - amt * amt
            prev_t = pl.col(TIME).shift(1).over(k)
        df = df.with_columns(_n=n, _s=s, _ss=ss, _prev=prev_t)
        mean = pl.col("_s") / pl.col("_n")
        var = (pl.col("_ss") / pl.col("_n") - mean * mean).clip(lower_bound=0.0)
        codes = _key_codes(df[k])
        n_24h = _count_last_24h(codes, t, pos)
        if mode == "global":  # also the rows in the NEXT 24 hours (negated time runs backwards)
            n_24h = n_24h + _count_last_24h(codes, -t, -pos)
        df = df.with_columns(
            pl.when(valid).then(pl.col("_n")).cast(pl.Float32).alias(f"{k}_n_prior"),
            pl.when(valid & (pl.col("_n") > 0))
            .then(mean)
            .cast(pl.Float32)
            .alias(f"{k}_amt_mean_prior"),
            pl.when(valid & (pl.col("_n") > 1))
            .then(var.sqrt())
            .cast(pl.Float32)
            .alias(f"{k}_amt_std_prior"),
            pl.when(valid & (pl.col("_n") > 0))
            .then(amt / mean)
            .cast(pl.Float32)
            .alias(f"{k}_amt_ratio"),
            pl.when(valid)
            .then(pl.col(TIME) - pl.col("_prev"))
            .cast(pl.Float32)
            .alias(f"{k}_secs_since_prev"),
            pl.when(valid).then(pl.Series(n_24h)).cast(pl.Float32).alias(f"{k}_n_24h"),
        )
    return df.drop(["_pos", "_n", "_s", "_ss", "_prev"])


def label_features(
    df: pl.DataFrame,
    keys: Sequence[str] = HISTORY_KEYS,
    delay_days: int = LABEL_DELAY_DAYS,
) -> pl.DataFrame:
    """Per entity: earlier transactions whose label has matured, and how many were fraud.

    A transaction at time ``s`` has a known label at time ``t`` when ``s <= t - delay``.
    """
    df = df.sort([TIME, ID])
    t = df[TIME].cast(pl.Int64).to_numpy()
    y = df[TARGET].cast(pl.Int64).to_numpy()
    pos = np.arange(len(df))
    for k in keys:
        codes = _key_codes(df[k])
        order = np.lexsort((pos, codes))
        k_sorted = codes[order].astype(np.int64)
        combined = k_sorted * (1 << 40) + t[order]
        group_start = np.searchsorted(combined, k_sorted * (1 << 40) - (1 << 39), side="left")
        cutoff = np.searchsorted(
            combined, k_sorted * (1 << 40) + t[order] - delay_days * DAY_SECONDS, side="right"
        )
        cum_fraud = np.concatenate([[0], np.cumsum(y[order])])
        matured = np.empty(len(df), dtype=np.float64)
        fraud = np.empty(len(df), dtype=np.float64)
        matured[order] = cutoff - group_start
        fraud[order] = cum_fraud[cutoff] - cum_fraud[group_start]
        valid = codes != -1
        df = df.with_columns(
            pl.Series(f"{k}_labels_matured", np.where(valid, matured, np.nan)).fill_nan(None),
            pl.Series(f"{k}_fraud_known", np.where(valid, fraud, np.nan)).fill_nan(None),
        ).with_columns(
            pl.when(pl.col(f"{k}_labels_matured") > 0)
            .then(pl.col(f"{k}_fraud_known") / pl.col(f"{k}_labels_matured"))
            .alias(f"{k}_fraud_rate_known"),
        )
        df = df.with_columns(pl.col(label_columns([k])).cast(pl.Float32))
    return df


def build_features(
    df: pl.DataFrame,
    mode: Literal["prior", "global"] = "prior",
    delay_days: int = LABEL_DELAY_DAYS,
) -> pl.DataFrame:
    return label_features(history_features(row_features(df), mode=mode), delay_days=delay_days)


@dataclass
class CategoryEncoder:
    """String columns -> integer codes learnt on the training period.

    Values seen fewer than ``min_count`` times share one "rare" code, as do values never seen in
    training. Missing stays missing (LightGBM has its own branch for it).
    """

    min_count: int = 20
    mapping: dict[str, dict[str, int]] = field(default_factory=dict)

    def fit(self, df: pl.DataFrame) -> CategoryEncoder:
        self.mapping = {}
        for col in df.columns:
            if df.schema[col] != pl.String or col in NOT_FEATURES:
                continue
            counts = df[col].drop_nulls().value_counts()
            frequent = counts.filter(pl.col("count") >= self.min_count).sort(col)[col].to_list()
            self.mapping[col] = {v: i for i, v in enumerate(frequent)}
        return self

    def rare_code(self, col: str) -> int:
        return len(self.mapping[col])

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for col, m in self.mapping.items():
            if col not in df.columns:
                continue
            exprs.append(
                pl.col(col)
                .replace_strict(m, default=self.rare_code(col), return_dtype=pl.Int32)
                .alias(col)
            )
            # replace_strict maps null to the default too; put missing back.
            exprs[-1] = pl.when(pl.col(col).is_null()).then(None).otherwise(exprs[-1]).alias(col)
        return df.with_columns(exprs)

    def encode_value(self, col: str, value: str | None) -> int | None:
        if value is None:
            return None
        return self.mapping[col].get(value, self.rare_code(col))


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES]

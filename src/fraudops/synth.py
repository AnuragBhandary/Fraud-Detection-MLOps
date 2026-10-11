"""A small synthetic dataset with the IEEE-CIS schema, for tests and CI.

The real data is behind Kaggle's competition rules and cannot be committed or downloaded in CI.
This generator keeps the columns the pipeline touches and plants a few signals (risky product,
risky email domain, card-testing bursts, new accounts) so the models have something to learn.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fraudops.config import AMOUNT, DAY_SECONDS, ID, TARGET, TIME


def make_transactions(n_customers: int = 600, days: int = 90, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    products = ["W", "C", "H", "R", "S"]
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", None]
    for _ in range(n_customers):
        card1 = int(rng.integers(1000, 1000 + n_customers // 3))
        addr1 = float(rng.integers(100, 140)) if rng.random() > 0.1 else None
        start = int(rng.integers(-400, days))
        fraudster = rng.random() < 0.06
        n_txn = int(rng.integers(1, 25))
        t = max(start, 0) * DAY_SECONDS + int(rng.integers(0, DAY_SECONDS))
        domain = domains[int(rng.integers(0, len(domains)))]
        if fraudster and rng.random() < 0.5:
            domain = "anonymous.com"
        device = "mobile" if rng.random() < 0.4 else "desktop"
        for _ in range(n_txn):
            gap = rng.exponential(600 if fraudster else 4 * DAY_SECONDS)
            t += int(gap) + 1
            if t >= days * DAY_SECONDS:
                break
            day = t // DAY_SECONDS
            amount = float(np.round(rng.lognormal(4.5 if fraudster else 4.0, 1.0), 2))
            product = products[int(rng.integers(0, 5))] if not fraudster else "C"
            p = 0.02 + (0.55 if fraudster else 0.0) + (0.05 if product == "C" else 0.0)
            rows.append(
                {
                    TIME: t + 86_400,
                    AMOUNT: amount,
                    "ProductCD": product,
                    "card1": card1,
                    "card2": float(rng.integers(100, 600)),
                    "card4": "visa" if card1 % 2 else "mastercard",
                    "card6": "credit" if card1 % 3 == 0 else "debit",
                    "addr1": addr1,
                    "addr2": 87.0 if addr1 is not None else None,
                    "dist1": float(rng.integers(0, 300)) if rng.random() < 0.4 else None,
                    "P_emaildomain": domain,
                    "R_emaildomain": None if rng.random() < 0.7 else domain,
                    "C1": float(rng.poisson(1 + 4 * fraudster)),
                    "C2": float(rng.poisson(1)),
                    "D1": float(day + 1 - start) if day + 1 - start >= 0 else 0.0,
                    "D15": float(rng.integers(0, 600)) if rng.random() < 0.8 else None,
                    "M4": ["M0", "M1", "M2", None][int(rng.integers(0, 4))],
                    "M6": "T" if rng.random() < 0.5 else "F",
                    "V1": float(rng.normal(1.0, 0.2)),
                    "V2": float(rng.normal(2.0 * fraudster, 1.0)),
                    "id_01": float(rng.normal(-5, 3)) if rng.random() < 0.3 else None,
                    "id_31": "chrome" if rng.random() < 0.6 else "safari",
                    "DeviceType": device if rng.random() < 0.3 else None,
                    "DeviceInfo": ["Windows", "iOS Device", "MacOS", None][int(rng.integers(0, 4))],
                    TARGET: int(rng.random() < p),
                }
            )
    df = pl.DataFrame(rows, infer_schema_length=None).sort(TIME)
    return df.with_row_index(ID, offset=2_987_000).with_columns(pl.col(ID).cast(pl.Int64))

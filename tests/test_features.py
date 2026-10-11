from __future__ import annotations

import numpy as np
import polars as pl

from fraudops.config import AMOUNT, DAY_SECONDS, ID, TIME
from fraudops.features import CategoryEncoder, build_features, history_columns


def _tiny() -> pl.DataFrame:
    rows = [  # card 1 three times (two within a day), card 2 once, then card 1 again later
        (1, 100, 10.0, 1, 200.0, 5.0, 1),
        (2, 200, 30.0, 1, 200.0, 5.0, 0),
        (3, 250, 7.0, 2, None, 3.0, 0),
        (4, 200 + DAY_SECONDS, 50.0, 1, 200.0, 6.0, 1),
        (5, 300 + DAY_SECONDS, 20.0, 1, 200.0, 6.0, 0),
    ]
    return pl.DataFrame(
        rows, schema=[ID, TIME, AMOUNT, "card1", "addr1", "D1", "isFraud"], orient="row"
    ).with_columns(pl.col(TIME).cast(pl.Int64))


def test_prior_features_hand_computed() -> None:
    f = build_features(_tiny()).sort(ID)
    assert f["card1_n_prior"].to_list() == [0, 1, 0, 2, 3]
    assert f["card1_amt_mean_prior"].to_list()[:2] == [None, 10.0]
    assert f["card1_amt_mean_prior"][4] == np.float32(30.0)  # (10 + 30 + 50) / 3
    assert f["card1_amt_ratio"][1] == 3.0
    assert f["card1_secs_since_prev"].to_list() == [None, 100, None, DAY_SECONDS, 100]
    # 24h window is (t - 1 day, t): row 4 at 200+day sees only row 2 (t=200 is exactly a day old).
    assert f["card1_n_24h"].to_list() == [0, 1, 0, 0, 1]
    # std needs two earlier rows: row 4 sees 10 and 30 -> population std 10.
    assert f["card1_amt_std_prior"][3] == 10.0


def test_known_fraud_waits_for_the_label_delay() -> None:
    f = build_features(_tiny(), delay_days=1).sort(ID)
    # Rows 4 and 5 (a day later): rows 1 and 2 are a day old, so known (one fraud). Row 5 does
    # not know row 4 (fraud, 100 s old): unknown, and not counted as legitimate either.
    assert f["card1_labels_matured"].to_list() == [0, 0, 0, 2, 2]
    assert f["card1_fraud_known"].to_list() == [0, 0, 0, 1, 1]
    assert f["card1_fraud_rate_known"].to_list() == [None, None, None, 0.5, 0.5]
    assert f["uid_labels_matured"][2] is None  # no uid, no features
    with_30 = build_features(_tiny()).sort(ID)
    assert with_30["card1_labels_matured"].to_list() == [0, 0, 0, 0, 0]


def test_uid_needs_address_and_account_start() -> None:
    f = build_features(_tiny()).sort(ID)
    assert f["uid"][2] is None
    assert f["uid_n_prior"][2] is None
    # day 0 - D1 5 = -5 and day 1 - D1 6 = -5: the same account across days.
    assert f["uid"][0] == f["uid"][4] == "1_200_-5"
    assert f["uid_n_prior"][4] == 3


def test_future_rows_cannot_change_past_features(transactions: pl.DataFrame) -> None:
    cutoff = int(transactions[TIME].quantile(0.6) or 0)
    base = build_features(transactions).filter(pl.col(TIME) < cutoff).sort(ID)
    tampered = transactions.with_columns(
        pl.when(pl.col(TIME) >= cutoff).then(pl.col(AMOUNT) * 50).otherwise(pl.col(AMOUNT))
    )
    after = build_features(tampered).filter(pl.col(TIME) < cutoff).sort(ID)
    for c in history_columns():
        assert base[c].equals(after[c], null_equal=True), c


def test_global_mode_peeks_into_the_future(transactions: pl.DataFrame) -> None:
    cutoff = int(transactions[TIME].quantile(0.6) or 0)
    tampered = transactions.with_columns(
        pl.when(pl.col(TIME) >= cutoff).then(pl.col(AMOUNT) * 50).otherwise(pl.col(AMOUNT))
    )
    a = build_features(transactions, mode="global").filter(pl.col(TIME) < cutoff).sort(ID)
    b = build_features(tampered, mode="global").filter(pl.col(TIME) < cutoff).sort(ID)
    assert not a["card1_amt_mean_prior"].equals(b["card1_amt_mean_prior"], null_equal=True)


def test_n_24h_matches_brute_force(features: pl.DataFrame) -> None:
    f = features.sort([TIME, ID])
    t, card = f[TIME].to_numpy(), f["card1"].to_numpy()
    got = f["card1_n_24h"].to_numpy()
    for i in np.random.default_rng(0).choice(len(f), 300, replace=False):
        earlier = slice(0, i)
        want = np.sum((card[earlier] == card[i]) & (t[earlier] > t[i] - DAY_SECONDS))
        assert got[i] == want


def test_category_encoder() -> None:
    train = pl.DataFrame({"c": ["a"] * 30 + ["b"] * 5 + [None]})
    enc = CategoryEncoder(min_count=20).fit(train)
    assert enc.mapping == {"c": {"a": 0}}
    out = enc.transform(pl.DataFrame({"c": ["a", "b", "zzz", None]}))
    assert out["c"].to_list() == [0, 1, 1, None]
    assert enc.encode_value("c", "a") == 0
    assert enc.encode_value("c", "never-seen") == 1
    assert enc.encode_value("c", None) is None

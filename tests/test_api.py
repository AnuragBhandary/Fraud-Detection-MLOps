from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient

from fraudops.api import create_app
from fraudops.config import ID, TIME
from fraudops.model import bundle, prepare, train_lgbm
from fraudops.online import OnlineFeatureStore
from fraudops.split import time_split


@pytest.fixture(scope="module")
def client(features: pl.DataFrame) -> TestClient:
    s = time_split(features)
    m = prepare(s.train, s.val, s.test)
    model = bundle(m, train_lgbm(m, max_rounds=100, patience=10), version="7")
    return TestClient(create_app(model, OnlineFeatureStore()))


def _txn(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "TransactionID": 1,
        "TransactionDT": 90_000,
        "TransactionAmt": 49.99,
        "ProductCD": "C",
        "card1": 4242,
        "addr1": 120.0,
        "D1": 3.0,
        "P_emaildomain": "anonymous.com",
    }
    return base | kw


def test_health_and_model(client: TestClient) -> None:
    assert client.get("/health").json()["model_version"] == "7"
    info = client.get("/v1/model").json()
    assert info["features"] > 20 and info["review_cost_usd"] == 10.0


def test_score_returns_decision_and_reasons(client: TestClient) -> None:
    r = client.post("/v1/score", json=_txn())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"review", "approve"}
    assert len(body["reasons"]) == 5
    assert float(r.headers["X-Process-Ms"]) > 0


def test_history_accumulates_between_requests(client: TestClient) -> None:
    store: OnlineFeatureStore = client.app.state.store  # type: ignore[attr-defined]
    before = store.state["card1"].get(777)
    assert before is None
    client.post("/v1/score", json=_txn(TransactionID=2, card1=777, TransactionDT=100_000))
    client.post("/v1/score", json=_txn(TransactionID=3, card1=777, TransactionDT=100_060))
    assert store.state["card1"][777].n == 2


def test_explain_off_and_unknown_fields(client: TestClient) -> None:
    body = client.post(
        "/v1/score", params={"explain": False}, json=_txn(TransactionID=4, brand_new_col="x")
    ).json()
    assert body["reasons"] is None


def test_validation(client: TestClient) -> None:
    assert client.post("/v1/score", json=_txn(TransactionAmt=0)).status_code == 422
    bad = _txn()
    del bad["TransactionDT"]
    assert client.post("/v1/score", json=bad).status_code == 422


def test_api_matches_offline_batch(features: pl.DataFrame) -> None:
    """Score the test period through the API from a store warmed on train+val."""
    import numpy as np

    from fraudops.model import to_matrix

    s = time_split(features)
    m = prepare(s.train, s.val, s.test)
    model = bundle(m, train_lgbm(m, max_rounds=100, patience=10))
    store = OnlineFeatureStore()
    store.warm(list(pl.concat([s.train, s.val]).sort([TIME, ID]).iter_rows(named=True)))
    c = TestClient(create_app(model, store))
    test = s.test.sort([TIME, ID]).head(300)
    offline = model.probability(to_matrix(model.encoder.transform(test), model.features))
    raw_cols = [
        c
        for c in test.columns
        if not c.startswith(("card1_", "uid"))
        and c not in {"hour", "dow", "amt_log", "amt_cents", "account_start", "day", "isFraud"}
    ]
    online = []
    for row in test.select(raw_cols).iter_rows(named=True):
        body = {k: v for k, v in row.items() if v is not None}
        online.append(c.post("/v1/score", json=body).json()["fraud_probability"])
    assert np.allclose(online, np.round(offline, 6), atol=1e-9)

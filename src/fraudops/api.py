"""Real-time scoring API.

``POST /v1/score`` takes one raw transaction (the IEEE-CIS columns), computes its history
features from the online store, scores it with the champion model and returns the calibrated
probability, the decision (review or approve), the expected loss, and the top SHAP reasons.
The transaction is then added to the store, so the next one from the same card sees it.

``POST /v1/chargeback`` reports a transaction as fraud. It feeds the known-fraud features once the
transaction is older than the label delay (until then its outcome is unknown, as in production).
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from fraudops.config import AMOUNT, ARTIFACTS_DIR
from fraudops.model import FraudModel
from fraudops.online import OnlineFeatureStore


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionDT: int = Field(ge=0)
    TransactionAmt: float = Field(gt=0)


class Chargeback(BaseModel):
    TransactionID: int


class Reason(BaseModel):
    feature: str
    value: float | None
    contribution: float


class Score(BaseModel):
    transaction_id: int
    fraud_probability: float
    decision: str
    expected_loss_usd: float
    model_version: str
    reasons: list[Reason] | None = None


def vectorize(model: FraudModel, values: dict[str, Any]) -> np.ndarray:
    x = np.full(len(model.features), np.nan, dtype=np.float32)
    for i, f in enumerate(model.features):
        v = values.get(f)
        if f in model.encoder.mapping:
            code = model.encoder.encode_value(f, None if v is None else str(v))
            if code is not None:
                x[i] = code
        elif v is not None and not (isinstance(v, float) and math.isnan(v)):
            x[i] = float(v)
    return x


def score_one(
    model: FraudModel, store: OnlineFeatureStore, txn: dict[str, Any], explain: bool = True
) -> Score:
    values = {**txn, **store.process(txn)}
    x = vectorize(model, values)
    p = float(model.probability(x.reshape(1, -1), threads=1)[0])
    amount = float(txn[AMOUNT])
    flag = bool(model.decide(np.array([p]), np.array([amount]))[0])
    return Score(
        transaction_id=int(txn["TransactionID"]),
        fraud_probability=round(p, 6),
        decision="review" if flag else "approve",
        expected_loss_usd=round(p * model.costs.missed(amount), 2),
        model_version=model.version,
        reasons=[Reason(**r) for r in model.reasons(x)] if explain else None,
    )


def create_app(model: FraudModel | None = None, store: OnlineFeatureStore | None = None) -> FastAPI:
    app = FastAPI(title="fraudops", version="1.0.0")

    if model is None:
        from fraudops.registry import load_champion

        model = load_champion()
    if store is None:
        snapshot = Path(os.environ.get("FRAUDOPS_STORE", ARTIFACTS_DIR / "feature_store.pkl"))
        store = OnlineFeatureStore.load(snapshot) if snapshot.exists() else OnlineFeatureStore()
    app.state.model = model
    app.state.store = store

    @app.middleware("http")
    async def timing(request: Request, call_next: Any) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        response.headers["X-Process-Ms"] = f"{(time.perf_counter() - start) * 1000:.3f}"
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_version": model.version, "entities": store.entities()}

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        return {
            "version": model.version,
            "features": len(model.features),
            "review_cost_usd": model.costs.review,
            "chargeback_fee_usd": model.costs.chargeback_fee,
        }

    # Sync handler: FastAPI runs it in a worker thread, so scoring does not stall the event loop.
    @app.post("/v1/score", response_model=Score)
    def score(txn: Transaction, explain: bool = True) -> Score:
        return score_one(model, store, txn.model_dump(), explain=explain)

    @app.post("/v1/chargeback", status_code=202)
    def chargeback(cb: Chargeback) -> dict[str, int]:
        store.report_chargeback(cb.TransactionID)
        return {"accepted": cb.TransactionID}

    return app

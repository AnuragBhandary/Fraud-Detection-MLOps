"""Serving benchmark: replay test-period transactions through the real HTTP API.

Starts uvicorn in a subprocess with the champion model and the store snapshot (state as of the
end of validation), then sends test transactions one at a time in stream order. Reports client
latency percentiles, the server's own processing time, and parity: every API probability is
compared with the offline batch prediction for the same row. Fraud rows are reported through
``/v1/chargeback`` after scoring, which is how the known-fraud features learn about them.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import time
from typing import Any

import httpx
import numpy as np
import polars as pl

from fraudops.config import ARTIFACTS_DIR, FEATURES_PARQUET, ID, TIME, TRANSACTIONS_PARQUET
from fraudops.model import FraudModel, to_matrix
from fraudops.split import time_split


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in row.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }


def run(n: int = 5000, warmup: int = 200, explain: bool = True) -> dict[str, Any]:
    from fraudops.registry import load_champion

    model: FraudModel = load_champion()  # the same artifact the server loads
    raw = pl.read_parquet(TRANSACTIONS_PARQUET)
    test_raw = time_split(raw).test.sort([TIME, ID]).head(n + warmup)
    feats = pl.read_parquet(FEATURES_PARQUET)
    test_feats = time_split(feats).test.sort([TIME, ID]).head(n + warmup)
    offline = model.probability(to_matrix(model.encoder.transform(test_feats), model.features))

    port = _free_port()
    env = {**os.environ, "FRAUDOPS_STORE": str(ARTIFACTS_DIR / "feature_store.pkl")}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "fraudops.serve:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=url, timeout=10) as client:
            for _ in range(300):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.2)
            lat, server_ms, probs, decisions = [], [], [], []
            for row in test_raw.iter_rows(named=True):
                body = _clean(row)
                is_fraud = body.pop("isFraud", 0)
                t0 = time.perf_counter()
                r = client.post("/v1/score", params={"explain": explain}, json=body)
                lat.append((time.perf_counter() - t0) * 1000)
                r.raise_for_status()
                server_ms.append(float(r.headers["X-Process-Ms"]))
                out = r.json()
                probs.append(out["fraud_probability"])
                decisions.append(out["decision"])
                if is_fraud:  # the issuer later learns the truth; it counts once matured
                    client.post("/v1/chargeback", json={"TransactionID": row["TransactionID"]})
    finally:
        server.terminate()
        server.wait(10)

    lat_a = np.array(lat[warmup:])
    srv = np.array(server_ms[warmup:])
    diff = np.abs(np.array(probs) - np.round(offline, 6))
    result = {
        "requests": len(lat_a),
        "explain": explain,
        "client_ms": {
            q: round(float(np.percentile(lat_a, int(q[1:]))), 2) for q in ("p50", "p95", "p99")
        },
        "server_ms": {
            q: round(float(np.percentile(srv, int(q[1:]))), 2) for q in ("p50", "p95", "p99")
        },
        "parity_max_abs_diff": float(diff.max()),
        "parity_rows_differing": int((diff > 1e-6).sum()),
        "review_rate_pct": round(100 * float(np.mean(np.array(decisions) == "review")), 2),
    }
    print(result)
    return result

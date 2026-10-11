from __future__ import annotations

import polars as pl

from fraudops import registry
from fraudops.model import bundle, prepare, train_lgbm
from fraudops.split import time_split


def test_champion_promotion(features: pl.DataFrame) -> None:
    import mlflow

    client = registry.setup()
    s = time_split(features)
    m = prepare(s.train, s.val, s.test)
    model = bundle(m, train_lgbm(m, max_rounds=30, patience=10))
    with mlflow.start_run(run_name="a"):
        v1 = registry.log_model(model, 0.30)
    assert registry.promote_if_better(client, v1, 0.30)
    with mlflow.start_run(run_name="b"):
        v2 = registry.log_model(model, 0.25)
    assert not registry.promote_if_better(client, v2, 0.25)  # worse: stays challenger
    assert registry.champion_score(client) == 0.30
    with mlflow.start_run(run_name="c"):
        v3 = registry.log_model(model, 0.35)
    assert registry.promote_if_better(client, v3, 0.35)
    loaded = registry.load_champion()
    assert loaded.version == v3
    assert loaded.features == model.features
    # The pyfunc face works too.
    pyfunc = mlflow.pyfunc.load_model(f"models:/{registry.MODEL_NAME}@champion")
    X = m.X_val[:5]
    import numpy as np
    import pandas as pd

    out = pyfunc.predict(pd.DataFrame(X, columns=m.features))
    assert np.allclose(out, model.probability(X))

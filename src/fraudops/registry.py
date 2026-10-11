"""MLflow: experiment tracking, the model registry and champion/challenger promotion.

The deployable unit is a pyfunc model wrapping a pickled :class:`fraudops.model.FraudModel`
(encoder + feature order + booster + calibrator + decision rule), so the API loads exactly what
was evaluated. A new version only takes the ``champion`` alias if it beats the current champion
on the validation period; otherwise it stays registered as a challenger.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient

from fraudops.config import CHAMPION_ALIAS, MLFLOW_URI, MODEL_NAME, ROOT
from fraudops.model import FraudModel

EXPERIMENT = "fraud-detection"


class FraudPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
    """Pyfunc face of the bundle: feature matrix in, calibrated fraud probability out."""

    def load_context(self, context: Any) -> None:
        with open(context.artifacts["bundle"], "rb") as f:
            self.model: FraudModel = pickle.load(f)

    # Left unannotated on purpose: MLflow infers (and complains about) a signature from hints.
    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        X = np.asarray(model_input[self.model.features], dtype=np.float32)
        return self.model.probability(X)


def setup(uri: str = MLFLOW_URI) -> MlflowClient:
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    if client.get_experiment_by_name(EXPERIMENT) is None:
        # Keep artifacts next to the database, not in whatever directory the command ran from.
        client.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return client


def log_model(model: FraudModel, val_pr_auc: float, tags: dict[str, str] | None = None) -> str:
    """Log the bundle inside the active run and register it. Returns the new version number."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.pkl"
        with path.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        warnings.filterwarnings("ignore", message="Passing a Python object as `python_model`")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudPyfunc(),
            artifacts={"bundle": str(path)},
            registered_model_name=MODEL_NAME,
            pip_requirements=["lightgbm", "scikit-learn", "numpy", "polars"],
        )
    client = MlflowClient()
    version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL_NAME, version, "val_pr_auc", f"{val_pr_auc:.6f}")
    for k, v in (tags or {}).items():
        client.set_model_version_tag(MODEL_NAME, version, k, v)
    return version


def champion_score(client: MlflowClient) -> float | None:
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    score = mv.tags.get("val_pr_auc")
    return float(score) if score is not None else None


def promote_if_better(client: MlflowClient, version: str, val_pr_auc: float) -> bool:
    current = champion_score(client)
    if current is not None and val_pr_auc <= current:
        client.set_model_version_tag(MODEL_NAME, version, "status", "challenger")
        return False
    try:
        previous = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version
        client.set_model_version_tag(MODEL_NAME, previous, "status", "retired")
    except mlflow.exceptions.MlflowException:
        pass  # first model: nothing to retire
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    client.set_model_version_tag(MODEL_NAME, version, "status", "champion")
    return True


def load_champion(uri: str = MLFLOW_URI) -> FraudModel:
    mlflow.set_tracking_uri(uri)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    model: FraudModel = pyfunc.unwrap_python_model().model
    if model.version == "dev":
        client = MlflowClient(uri)
        model.version = str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version)
    return model

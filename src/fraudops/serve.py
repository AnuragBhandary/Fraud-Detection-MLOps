"""ASGI entry point: ``uvicorn fraudops.serve:app``.

Loads the registry champion (or a pickled bundle from ``FRAUDOPS_MODEL``) and the store snapshot.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from fraudops.api import create_app
from fraudops.model import FraudModel


def _model() -> FraudModel | None:
    path = os.environ.get("FRAUDOPS_MODEL")
    if not path:
        return None  # create_app falls back to the registry champion
    with Path(path).open("rb") as f:
        model: FraudModel = pickle.load(f)
    return model


app = create_app(_model())

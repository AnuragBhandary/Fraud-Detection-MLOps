"""Training: logistic baseline, LightGBM, Optuna search, calibration, and the deployable bundle."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from fraudops.config import AMOUNT, COSTS, N_THREADS, SEED, TARGET, Costs
from fraudops.features import CategoryEncoder, feature_names
from fraudops.metrics import expected_cost_flags, pr_auc

BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "average_precision",
    "num_threads": N_THREADS,
    "seed": SEED,
    "verbosity": -1,
}


@dataclass
class Matrices:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    amt_val: np.ndarray
    amt_test: np.ndarray
    features: list[str]
    categorical: list[str]
    encoder: CategoryEncoder


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in features]).to_numpy()


def prepare(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    drop: tuple[str, ...] = (),
) -> Matrices:
    enc = CategoryEncoder().fit(train)
    tr, va, te = enc.transform(train), enc.transform(val), enc.transform(test)
    feats = [c for c in feature_names(tr) if c not in drop]
    return Matrices(
        X_train=to_matrix(tr, feats),
        y_train=tr[TARGET].to_numpy(),
        X_val=to_matrix(va, feats),
        y_val=va[TARGET].to_numpy(),
        X_test=to_matrix(te, feats),
        y_test=te[TARGET].to_numpy(),
        amt_val=va[AMOUNT].to_numpy(),
        amt_test=te[AMOUNT].to_numpy(),
        features=feats,
        categorical=[c for c in feats if c in enc.mapping],
        encoder=enc,
    )


def train_lgbm(
    m: Matrices,
    params: dict[str, Any] | None = None,
    max_rounds: int = 6000,
    patience: int = 100,
    callbacks: list[Callable[..., None]] | None = None,
) -> lgb.Booster:
    p = {**BASE_PARAMS, **(params or {})}
    dtrain = lgb.Dataset(
        m.X_train, m.y_train, feature_name=m.features, categorical_feature=m.categorical
    )
    dval = lgb.Dataset(m.X_val, m.y_val, reference=dtrain)
    return lgb.train(
        p,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(patience, verbose=False), *(callbacks or [])],
    )


def predict(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(X, num_iteration=booster.best_iteration))


def train_logistic(m: Matrices) -> Pipeline:
    """Linear baseline on the non-Vesta columns: rank-scaled numerics, one-hot categoricals.

    Rank scaling (QuantileTransformer) rather than StandardScaler: counts and time deltas in later
    months fall far outside the training range, and a linear model extrapolates them without
    limit. With StandardScaler the test PR-AUC halved (docs/RESULTS.md).
    """
    numeric = [
        i for i, c in enumerate(m.features) if c not in m.categorical and not c.startswith("V")
    ]
    cats = [i for i, c in enumerate(m.features) if c in m.categorical]
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    QuantileTransformer(
                        output_distribution="normal", subsample=100_000, random_state=SEED
                    ),
                ),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=-1),
                    OneHotEncoder(handle_unknown="ignore", max_categories=30),
                ),
                cats,
            ),
        ]
    )
    model = Pipeline(
        [("pre", pre), ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED))]
    )
    model.fit(m.X_train, m.y_train)
    return model


def _pruning_callback(trial: optuna.Trial, every: int = 25) -> Callable[..., None]:
    def cb(env: lgb.callback.CallbackEnv) -> None:
        if env.iteration % every:
            return
        score = (env.evaluation_result_list or [])[0][2]
        trial.report(score, env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned

    return cb


def tune(m: Matrices, n_trials: int = 30, timeout_s: float | None = None) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 512, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
        start = time.perf_counter()
        b = train_lgbm(m, params, callbacks=[_pruning_callback(trial)])
        trial.set_user_attr("best_iteration", b.best_iteration)
        trial.set_user_attr("seconds", time.perf_counter() - start)
        return pr_auc(m.y_val, predict(b, m.X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )
    study.enqueue_trial(
        {
            k: BASE_PARAMS[k]
            for k in (
                "num_leaves",
                "min_child_samples",
                "feature_fraction",
                "bagging_fraction",
                "lambda_l2",
            )
        }
        | {"min_gain_to_split": 0.0}
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    return study


@dataclass
class FraudModel:
    """Everything serving needs: encoder, feature order, booster, calibration and the rule."""

    encoder: CategoryEncoder
    features: list[str]
    booster_str: str
    calibrator: IsotonicRegression
    costs: Costs = field(default_factory=lambda: COSTS)
    version: str = "dev"
    _booster: lgb.Booster | None = field(default=None, repr=False)

    @property
    def booster(self) -> lgb.Booster:
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.booster_str)
        return self._booster

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_booster"] = None
        return state

    def raw_score(self, X: np.ndarray, threads: int = N_THREADS) -> np.ndarray:
        return np.asarray(self.booster.predict(X, num_threads=threads))

    def probability(self, X: np.ndarray, threads: int = N_THREADS) -> np.ndarray:
        return np.asarray(self.calibrator.predict(self.raw_score(X, threads)))

    def decide(self, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
        return expected_cost_flags(p, amount, self.costs)

    def contributions(self, X: np.ndarray, threads: int = 1) -> np.ndarray:
        """TreeSHAP values in log-odds, one column per feature plus the bias (last)."""
        return np.asarray(self.booster.predict(X, pred_contrib=True, num_threads=threads))

    def reasons(self, x: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        contrib = self.contributions(x.reshape(1, -1))[0, :-1]
        top = np.argsort(-np.abs(contrib))[:k]
        return [
            {
                "feature": self.features[i],
                "value": None if np.isnan(x[i]) else float(x[i]),
                "contribution": round(float(contrib[i]), 4),
            }
            for i in top
        ]


def fit_calibrator(raw_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, y_val)


def bundle(m: Matrices, booster: lgb.Booster, version: str = "dev") -> FraudModel:
    # Freeze the early-stopped iteration count into the model text.
    text = booster.model_to_string(num_iteration=booster.best_iteration)
    model = FraudModel(
        encoder=m.encoder,
        features=m.features,
        booster_str=text,
        calibrator=IsotonicRegression(),
        version=version,
    )
    model.calibrator = fit_calibrator(model.raw_score(m.X_val), m.y_val)
    return model

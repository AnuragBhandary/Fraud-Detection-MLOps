"""The experiment steps behind the CLI. Each writes its numbers to ``docs/results/<step>.json``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fraudops.config import (
    AMOUNT,
    ARTIFACTS_DIR,
    FEATURES_PARQUET,
    ID,
    RESULTS_DIR,
    SEED,
    TARGET,
    TIME,
    TRANSACTIONS_PARQUET,
)
from fraudops.features import build_features, history_columns
from fraudops.metrics import (
    best_threshold,
    bootstrap_ci,
    cost_report,
    expected_cost_flags,
    pr_auc,
    ranking_metrics,
)
from fraudops.model import (
    BASE_PARAMS,
    FraudModel,
    Matrices,
    bundle,
    predict,
    prepare,
    train_lgbm,
    train_logistic,
    tune,
)
from fraudops.split import time_split


def save_results(name: str, payload: dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path


def _round(d: dict[str, float], nd: int = 4) -> dict[str, float]:
    return {k: round(float(v), nd) for k, v in d.items()}


def make_features(
    transactions: Path = TRANSACTIONS_PARQUET, out: Path = FEATURES_PARQUET
) -> pl.DataFrame:
    df = pl.read_parquet(transactions)
    start = time.perf_counter()
    feats = build_features(df)
    feats.write_parquet(out)
    print(f"features: {feats.shape} in {time.perf_counter() - start:.1f}s -> {out}")
    return feats


def describe(df: pl.DataFrame) -> dict[str, Any]:
    def stats(d: pl.DataFrame) -> dict[str, Any]:
        row = d.select(
            rows=pl.len(),
            fraud_rate_pct=(100 * pl.col(TARGET).mean()).round(3),
            first_day=pl.col(TIME).min() // 86400,
            last_day=pl.col(TIME).max() // 86400,
            fraud_dollars_pct=(
                100 * pl.col(AMOUNT).filter(pl.col(TARGET) == 1).sum() / pl.col(AMOUNT).sum()
            ).round(2),
        )
        return row.to_dicts()[0]

    split = time_split(df)
    return {
        **stats(df),
        "columns": df.width,
        "split": {
            n: stats(d)
            for n, d in (("train", split.train), ("val", split.val), ("test", split.test))
        },
    }


def _policies(
    model: FraudModel, m: Matrices, raw_val: np.ndarray, raw_test: np.ndarray
) -> dict[str, Any]:
    """Compare decision rules on the test period (all chosen on validation)."""
    p_val = model.calibrator.predict(raw_val)
    p_test = model.calibrator.predict(raw_test)
    thr = best_threshold(m.y_val, raw_val, m.amt_val)
    out: dict[str, Any] = {
        "flag_nothing": cost_report(m.y_test, np.zeros_like(m.y_test, dtype=bool), m.amt_test),
        "raw_score_0.5": cost_report(m.y_test, raw_test >= 0.5, m.amt_test),
        "best_val_threshold": {
            "threshold": round(thr.threshold, 5),
            **cost_report(m.y_test, raw_test >= thr.threshold, m.amt_test),
        },
        "expected_cost_rule": cost_report(
            m.y_test, expected_cost_flags(p_test, m.amt_test, model.costs), m.amt_test
        ),
    }
    # Calibration check: predicted vs observed fraud by decile of calibrated probability.
    bins = np.quantile(p_test, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(bins, p_test, side="right") - 1, 0, 9)
    out["calibration_test"] = [
        {
            "decile": int(b + 1),
            "mean_predicted": round(float(p_test[idx == b].mean()), 4),
            "observed": round(float(m.y_test[idx == b].mean()), 4),
        }
        for b in range(10)
        if (idx == b).any()
    ]
    out["val_expected_cost_rule"] = cost_report(
        m.y_val, expected_cost_flags(p_val, m.amt_val, model.costs), m.amt_val
    )
    return {
        k: _round(v) if isinstance(v, dict) and "threshold" not in v else v for k, v in out.items()
    }


def train(
    n_trials: int = 30, features_path: Path = FEATURES_PARQUET, register: bool = True
) -> dict[str, Any]:
    from fraudops import registry

    df = pl.read_parquet(features_path)
    split = time_split(df)
    m = prepare(split.train, split.val, split.test)
    results: dict[str, Any] = {"data": describe(df), "features": len(m.features)}
    if register:
        client = registry.setup()
        import mlflow

    timings: dict[str, float] = {}

    def run(name: str, params: dict[str, Any]) -> Any:
        return mlflow.start_run(run_name=name, tags={"step": name}) if register else _null()

    # 1. Logistic regression baseline.
    t0 = time.perf_counter()
    lr = train_logistic(m)
    lr_val = lr.predict_proba(m.X_val)[:, 1]
    lr_test = lr.predict_proba(m.X_test)[:, 1]
    timings["logistic_s"] = time.perf_counter() - t0
    results["logistic"] = {
        "val": _round(ranking_metrics(m.y_val, lr_val)),
        "test": _round(ranking_metrics(m.y_test, lr_test)),
    }
    with run("logistic", {}):
        if register:
            mlflow.log_metrics({f"val_{k}": v for k, v in results["logistic"]["val"].items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in results["logistic"]["test"].items()})
    print("logistic", results["logistic"])

    # 2. LightGBM with hand-picked defaults.
    t0 = time.perf_counter()
    base = train_lgbm(m)
    timings["lgbm_base_s"] = time.perf_counter() - t0
    base_val, base_test = predict(base, m.X_val), predict(base, m.X_test)
    results["lgbm_base"] = {
        "best_iteration": base.best_iteration,
        "val": _round(ranking_metrics(m.y_val, base_val)),
        "test": _round(ranking_metrics(m.y_test, base_test)),
    }
    with run("lgbm_base", BASE_PARAMS):
        if register:
            mlflow.log_params(BASE_PARAMS)
            mlflow.log_metrics({f"val_{k}": v for k, v in results["lgbm_base"]["val"].items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in results["lgbm_base"]["test"].items()})
    print("lgbm_base", results["lgbm_base"])

    # 3. Ablation: same model without the history features.
    hist = set(history_columns())
    m_nohist = prepare(split.train, split.val, split.test, drop=tuple(hist))
    nohist = train_lgbm(m_nohist)
    results["lgbm_no_history"] = {
        "val": _round(ranking_metrics(m.y_val, predict(nohist, m_nohist.X_val))),
        "test": _round(ranking_metrics(m.y_test, predict(nohist, m_nohist.X_test))),
    }
    del m_nohist
    print("lgbm_no_history", results["lgbm_no_history"])

    # 4. Optuna search (validation PR-AUC, early stopping, median pruning).
    t0 = time.perf_counter()
    study = tune(m, n_trials=n_trials)
    timings["optuna_s"] = time.perf_counter() - t0
    trials = [t for t in study.trials if t.value is not None]
    results["optuna"] = {
        "trials": len(study.trials),
        "completed": sum(t.state.name == "COMPLETE" for t in study.trials),
        "pruned": sum(t.state.name == "PRUNED" for t in study.trials),
        "best_val_pr_auc": round(study.best_value, 4),
        "default_val_pr_auc": round(float(trials[0].value or 0.0), 4) if trials else None,
        "best_params": study.best_params,
    }
    print("optuna", results["optuna"])

    # 5. Final model = best trial's params.
    t0 = time.perf_counter()
    best = train_lgbm(m, study.best_params)
    timings["lgbm_tuned_s"] = time.perf_counter() - t0
    model = bundle(m, best)
    raw_val, raw_test = model.raw_score(m.X_val), model.raw_score(m.X_test)
    results["lgbm_tuned"] = {
        "best_iteration": best.best_iteration,
        "val": _round(ranking_metrics(m.y_val, raw_val)),
        "test": _round(ranking_metrics(m.y_test, raw_test)),
    }
    print("lgbm_tuned", results["lgbm_tuned"])

    # Is the gain over the baseline real? Paired bootstrap over test rows.
    def diff(idx: np.ndarray) -> float:
        return pr_auc(m.y_test[idx], raw_test[idx]) - pr_auc(m.y_test[idx], lr_test[idx])

    lo, hi = bootstrap_ci(diff, len(m.y_test), reps=300, seed=SEED)
    results["tuned_minus_logistic_test_pr_auc_ci95"] = [round(lo, 4), round(hi, 4)]

    results["decisions"] = _policies(model, m, raw_val, raw_test)

    # Global importance from TreeSHAP on a test sample.
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(m.X_test), size=min(20_000, len(m.X_test)), replace=False)
    contrib = model.contributions(m.X_test[sample], threads=4)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    top = np.argsort(-mean_abs)[:20]
    results["shap_top20"] = [
        {"feature": model.features[i], "mean_abs_log_odds": round(float(mean_abs[i]), 4)}
        for i in top
    ]
    results["timings_s"] = _round(timings, 1)

    if register:
        with mlflow.start_run(run_name="lgbm_tuned", tags={"step": "lgbm_tuned"}):
            mlflow.log_params(study.best_params | {"best_iteration": best.best_iteration})
            mlflow.log_metrics({f"val_{k}": v for k, v in results["lgbm_tuned"]["val"].items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in results["lgbm_tuned"]["test"].items()})
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in results["decisions"]["expected_cost_rule"].items()}
            )
            mlflow.log_dict(results, "results.json")
            version = registry.log_model(
                model, results["lgbm_tuned"]["val"]["pr_auc"], tags={"trials": str(n_trials)}
            )
            promoted = registry.promote_if_better(
                client, version, results["lgbm_tuned"]["val"]["pr_auc"]
            )
        results["registry"] = {"version": version, "promoted_to_champion": promoted}
        print("registry", results["registry"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    import pickle

    with (ARTIFACTS_DIR / "model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_results("train", results)
    return results


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: object) -> None:
        return None


def leakage(
    transactions: Path = TRANSACTIONS_PARQUET, features_path: Path = FEATURES_PARQUET
) -> dict[str, Any]:
    """How much do a random split and future-peeking aggregates inflate the scores?

    Same LightGBM settings everywhere; every number is PR-AUC on the held-out part. "Honest" is the
    production-like setup: past-only features, train before / evaluate after.
    """
    honest = pl.read_parquet(features_path)
    leaky = build_features(pl.read_parquet(transactions), mode="global")
    out: dict[str, Any] = {}

    def fit_eval(df: pl.DataFrame, random_split: bool) -> dict[str, float]:
        if random_split:
            shuffled = df.sample(fraction=1.0, shuffle=True, seed=SEED)
            n = df.height
            tr = shuffled[: int(0.7 * n)]
            va = shuffled[int(0.7 * n) : int(0.85 * n)]
            te = shuffled[int(0.85 * n) :]
        else:
            s = time_split(df)
            tr, va, te = s.train, s.val, s.test
        m = prepare(tr, va, te)
        b = train_lgbm(m)
        return {
            "test_pr_auc": round(pr_auc(m.y_test, predict(b, m.X_test)), 4),
            "rounds": b.best_iteration,
        }

    for feats_name, df in (("past_only_features", honest), ("future_peeking_features", leaky)):
        for split_name, rnd in (("time_split", False), ("random_split", True)):
            key = f"{feats_name}/{split_name}"
            out[key] = fit_eval(df, rnd)
            print(key, out[key])
    save_results("leakage", out)
    return out


def build_store(features_path: Path = FEATURES_PARQUET, out: Path | None = None) -> dict[str, Any]:
    """Replay every transaction before the test period into the online store and snapshot it."""
    from fraudops.online import OnlineFeatureStore

    out = out or ARTIFACTS_DIR / "feature_store.pkl"
    df = pl.read_parquet(features_path, columns=[ID, TIME, AMOUNT, "card1", "addr1", "D1"])
    split = time_split(df)
    history = pl.concat([split.train, split.val]).sort([TIME, ID])
    store = OnlineFeatureStore()
    start = time.perf_counter()
    store.warm(list(history.iter_rows(named=True)))
    elapsed = time.perf_counter() - start
    store.save(out)
    info = {
        "replayed": history.height,
        "seconds": round(elapsed, 1),
        "entities": store.entities(),
        "snapshot_mb": round(out.stat().st_size / 1e6, 1),
    }
    print(info)
    return info


def drift(features_path: Path = FEATURES_PARQUET, top_k: int = 30) -> dict[str, Any]:
    import pickle

    from fraudops.drift import psi, weekly_report

    with (ARTIFACTS_DIR / "model.pkl").open("rb") as f:
        model: FraudModel = pickle.load(f)
    df = pl.read_parquet(features_path)
    split = time_split(df)
    enc = model.encoder
    tr = enc.transform(split.train)
    later = enc.transform(pl.concat([split.val, split.test]))
    from fraudops.model import to_matrix

    X_tr, X_later = to_matrix(tr, model.features), to_matrix(later, model.features)
    s_tr, s_later = model.raw_score(X_tr), model.raw_score(X_later)
    # Monitor the features the model leans on most.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr), size=min(20_000, len(X_tr)), replace=False)
    imp = np.abs(model.contributions(X_tr[idx], threads=4)[:, :-1]).mean(axis=0)
    watched = [model.features[i] for i in np.argsort(-imp)[:top_k]]
    report = weekly_report(tr, later, watched, s_tr, s_later)
    all_psi = {f: round(psi(X_tr[:, i], X_later[:, i]), 4) for i, f in enumerate(model.features)}
    worst = sorted(all_psi.items(), key=lambda kv: -kv[1])[:15]
    out = {
        "watched_features": watched,
        "weekly": report.to_dicts(),
        "worst_features_val_test_vs_train": dict(worst),
        "features_psi_over_0.2": sum(v > 0.2 for v in all_psi.values()),
        "features_total": len(all_psi),
    }
    save_results("drift", out)
    print(report)
    return out

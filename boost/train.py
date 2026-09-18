"""LightGBM wrapper. The only file that imports the boosting library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from boost.types import Fold, Task


@dataclass
class Matrix:
    X: np.ndarray
    names: list[str]
    categorical: list[int]
    encoders: dict[str, dict[str, int]]


def encode_matrix(df: pl.DataFrame, features: list[str]) -> Matrix:
    """Numeric -> float64 with NaN for null. String -> integer codes, null -> -1 (missing)."""
    cols: list[np.ndarray] = []
    cat_idx: list[int] = []
    encoders: dict[str, dict[str, int]] = {}
    for i, name in enumerate(features):
        s = df[name]
        if s.dtype in (pl.Utf8, pl.Categorical, pl.Enum):
            vals = s.cast(pl.Utf8)
            levels = sorted(v for v in vals.unique().to_list() if v is not None)
            enc = {v: k for k, v in enumerate(levels)}
            codes = np.array([enc.get(v, -1) if v is not None else -1 for v in vals.to_list()], dtype=np.float64)
            cols.append(codes)
            cat_idx.append(i)
            encoders[name] = enc
        elif s.dtype == pl.Boolean:
            cols.append(s.cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float64))
        else:
            cols.append(s.cast(pl.Float64).to_numpy().astype(np.float64))
    X = np.column_stack(cols) if cols else np.zeros((df.height, 0))
    return Matrix(X=X, names=list(features), categorical=cat_idx, encoders=encoders)


def _objective(task: Task) -> dict[str, Any]:
    if task.type == "binary":
        return {"objective": "binary", "metric": "binary_logloss"}
    if task.type == "multiclass":
        return {"objective": "multiclass", "num_class": task.n_classes, "metric": "multi_logloss"}
    if task.type == "poisson":
        return {"objective": "poisson", "metric": "poisson"}
    return {"objective": "regression", "metric": "l2"}


@dataclass
class FitResult:
    booster: lgb.Booster
    best_iteration: int
    curve: pl.DataFrame          # iteration, train, valid
    importance: pl.DataFrame     # feature, gain, split
    valid_pred: np.ndarray       # predictions on the valid rows from the early-stopped model
    test_pred: np.ndarray        # predictions on the test rows (refit model if refit: true)
    refit: bool


def _predict(booster: lgb.Booster, X: np.ndarray, task: Task, num_iteration: int | None, offset: np.ndarray | None = None) -> np.ndarray:
    """With an offset the booster's raw score is a correction; add it back before the link."""
    if offset is None:
        p = np.asarray(booster.predict(X, num_iteration=num_iteration), dtype=np.float64)
        if task.type == "binary":
            p = np.column_stack([1.0 - p, p])
        return p
    raw = np.asarray(booster.predict(X, num_iteration=num_iteration, raw_score=True), dtype=np.float64) + offset
    if task.type == "multiclass":
        raw = raw - raw.max(axis=1, keepdims=True)
        e = np.exp(raw)
        return e / e.sum(axis=1, keepdims=True)
    if task.type == "binary":
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p, p])
    if task.type == "poisson":
        return np.exp(raw)
    return raw


def offset_from_baseline(B: np.ndarray, task: Task) -> np.ndarray:
    """Baseline probabilities/values -> init_score the booster starts from (log-odds or log-rate)."""
    eps = 1e-6
    if task.type == "multiclass":
        P = np.clip(B, eps, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        return np.log(P)
    if task.type == "binary":
        p = np.clip(B[:, -1] if B.ndim == 2 else B, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
    if task.type == "poisson":
        return np.log(np.clip(B[:, 0] if B.ndim == 2 else B, eps, None))
    return (B[:, 0] if B.ndim == 2 else B).astype(np.float64)


def fit_fold(m: Matrix, y: np.ndarray, fold: Fold, task: Task, model_cfg: dict[str, Any], seed_offset: int = 0, offset: np.ndarray | None = None) -> FitResult:
    """offset: per-row init_score (n or n x k). The booster learns a correction on top of it."""
    mcfg = dict(model_cfg.get("model") or {})
    params: dict[str, Any] = {"verbosity": -1, "seed": 7, **_objective(task), **dict(mcfg.get("params") or {})}
    params["seed"] = int(params.get("seed", 7)) + seed_offset
    rounds = int(mcfg.get("rounds", 2000))
    patience = int(mcfg.get("early_stopping", 100))
    refit = bool(mcfg.get("refit", True))

    def init(idx: np.ndarray) -> np.ndarray | None:
        return None if offset is None else offset[idx]

    dtrain = lgb.Dataset(m.X[fold.train], label=y[fold.train], init_score=init(fold.train), feature_name=m.names, categorical_feature=m.categorical, free_raw_data=False)
    dvalid = lgb.Dataset(m.X[fold.valid], label=y[fold.valid], init_score=init(fold.valid), reference=dtrain, free_raw_data=False)
    evals: dict[str, Any] = {}
    callbacks = [lgb.record_evaluation(evals)]
    if patience > 0:
        callbacks.append(lgb.early_stopping(patience, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dtrain, dvalid], valid_names=["train", "valid"], callbacks=callbacks)
    best = int(booster.best_iteration or rounds)
    metric = params["metric"]
    tr = evals["train"][metric]
    va = evals["valid"][metric]
    curve = pl.DataFrame({"iteration": np.arange(1, len(tr) + 1), "train": tr, "valid": va})

    valid_pred = _predict(booster, m.X[fold.valid], task, best, init(fold.valid))
    final = booster
    if refit and fold.test.size:
        both = np.concatenate([fold.train, fold.valid])
        dboth = lgb.Dataset(m.X[both], label=y[both], init_score=init(both), feature_name=m.names, categorical_feature=m.categorical, free_raw_data=False)
        final = lgb.train(params, dboth, num_boost_round=best)
    test_pred = _predict(final, m.X[fold.test], task, best if not refit else None, init(fold.test)) if fold.test.size else np.zeros((0, task.n_classes))

    gain = final.feature_importance(importance_type="gain")
    split = final.feature_importance(importance_type="split")
    importance = pl.DataFrame({"feature": m.names, "gain": gain.astype(np.float64), "split": split.astype(np.int64)}).sort("gain", descending=True)
    return FitResult(booster=final, best_iteration=best, curve=curve, importance=importance, valid_pred=valid_pred, test_pred=test_pred, refit=refit)

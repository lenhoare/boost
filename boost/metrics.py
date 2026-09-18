"""Proper scores and diagnostics. Nothing here knows what a row is."""

from __future__ import annotations

import numpy as np
import polars as pl

from boost.types import Task

EPS = 1e-12


def logloss_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(P[np.arange(len(y)), y], EPS, 1.0)
    return -np.log(p)


def brier_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return ((P - onehot) ** 2).sum(axis=1)


def classification_summary(P: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {}
    return {
        "n": float(len(y)),
        "logloss": float(logloss_rows(P, y).mean()),
        "brier": float(brier_rows(P, y).mean()),
        "accuracy": float((P.argmax(axis=1) == y).mean()),
    }


def regression_summary(pred: np.ndarray, y: np.ndarray, poisson: bool = False) -> dict[str, float]:
    if len(y) == 0:
        return {}
    out = {
        "n": float(len(y)),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
    }
    if poisson:
        mu = np.clip(pred, EPS, None)
        dev = 2.0 * (np.where(y > 0, y * np.log(np.clip(y, EPS, None) / mu), 0.0) - (y - mu))
        out["poisson_deviance"] = float(dev.mean())
    return out


def summary(task: Task, pred: pl.DataFrame, baseline: list[str] | None) -> dict[str, float]:
    """Metrics on one frame of predictions (already filtered to a split)."""
    if pred.height == 0:
        return {}
    if task.is_classification:
        P = pred.select(task.prob_columns).to_numpy()
        y = pred["_y"].to_numpy().astype(np.int64)
        out = classification_summary(P, y)
        if baseline:
            B = pred.select(baseline).to_numpy()
            ok = np.isfinite(B).all(axis=1)
            if ok.any():
                B = B[ok] / B[ok].sum(axis=1, keepdims=True)
                out["baseline_n"] = float(ok.sum())
                out["baseline_logloss"] = float(logloss_rows(B, y[ok]).mean())
                out["model_logloss_on_baseline_rows"] = float(logloss_rows(P[ok], y[ok]).mean())
                out["model_minus_baseline"] = out["model_logloss_on_baseline_rows"] - out["baseline_logloss"]
        return out
    p = pred["pred"].to_numpy()
    y = pred["_y"].to_numpy().astype(np.float64)
    out = regression_summary(p, y, poisson=task.type == "poisson")
    if baseline:
        b = pred[baseline[0]].to_numpy().astype(np.float64)
        ok = np.isfinite(b)
        if ok.any():
            base = regression_summary(b[ok], y[ok], poisson=task.type == "poisson")
            out["baseline_rmse"] = base["rmse"]
    return out


def calibration_table(task: Task, pred: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    """Per class: predicted-probability bins vs observed frequency."""
    P = pred.select(task.prob_columns).to_numpy()
    y = pred["_y"].to_numpy().astype(np.int64)
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for k, name in enumerate(task.class_names):
        pk = P[:, k]
        hit = (y == k).astype(np.float64)
        which = np.clip(np.digitize(pk, edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            m = which == b
            if not m.any():
                continue
            rows.append({
                "class": name,
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(m.sum()),
                "mean_pred": float(pk[m].mean()),
                "observed": float(hit[m].mean()),
                "gap": float(hit[m].mean() - pk[m].mean()),
            })
    return pl.DataFrame(rows)


def period_table(task: Task, pred: pl.DataFrame, period_col: str, baseline: list[str] | None) -> pl.DataFrame:
    rows = []
    for period in pred[period_col].unique(maintain_order=True).to_list():
        part = pred.filter(pl.col(period_col) == period)
        s = summary(task, part, baseline)
        row = {"period": str(period), "n": int(s.get("n", 0))}
        if task.is_classification:
            row.update({"logloss": s.get("logloss"), "brier": s.get("brier"), "accuracy": s.get("accuracy")})
            if "baseline_logloss" in s:
                row.update({"baseline_logloss": s["baseline_logloss"], "model_minus_baseline": s["model_minus_baseline"]})
        else:
            row.update({"rmse": s.get("rmse"), "mae": s.get("mae")})
            if "poisson_deviance" in s:
                row["poisson_deviance"] = s["poisson_deviance"]
        rows.append(row)
    return pl.DataFrame(rows)


def class_balance(task: Task, pred: pl.DataFrame) -> pl.DataFrame:
    P = pred.select(task.prob_columns).to_numpy()
    y = pred["_y"].to_numpy().astype(np.int64)
    rows = []
    for k, name in enumerate(task.class_names):
        rows.append({
            "class": name,
            "observed_rate": float((y == k).mean()),
            "mean_pred": float(P[:, k].mean()),
            "picked_rate": float((P.argmax(axis=1) == k).mean()),
        })
    return pl.DataFrame(rows)

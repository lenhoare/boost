"""Train one model config on one project and save a run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from boost.data import build_features, select_features
from boost.metrics import summary
from boost.project import Project
from boost.split import make_folds
from boost.store import new_run_dir, write_meta
from boost.train import encode_matrix, fit_fold, offset_from_baseline
from boost.types import Task


def _unique(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(s for s in seq if s))


def train(project: Project, model_name: str, tag: str | None = None, log=print) -> Path:
    t0 = time.time()
    frame, all_features = build_features(project)
    mcfg = project.model_cfg(model_name)
    task = Task.from_cfg(mcfg["task"])
    baseline = [str(c) for c in (mcfg.get("baseline") or [])]

    df = frame
    rows_expr = mcfg.get("rows")
    if rows_expr:
        df = df.filter(pl.sql_expr(str(rows_expr)))
    df = df.filter(pl.col(task.target).is_not_null())
    if df.height == 0:
        raise ValueError("no rows left after rows filter / target null drop")

    features = select_features(all_features, mcfg.get("features"))
    if task.target in features:
        features = [f for f in features if f != task.target]
        log(f"note: dropped target column {task.target!r} from features")
    overlap = [f for f in features if f in baseline]
    if overlap:
        log(f"note: baseline columns are also features: {overlap} (the model can see the bar it is measured against)")
    missing_base = [c for c in baseline if c not in df.columns]
    if missing_base:
        raise KeyError(f"baseline columns not in frame: {missing_base}")

    offset = None
    if mcfg.get("offset"):
        if not baseline:
            raise ValueError("offset: true needs baseline columns")
        B = df.select(baseline).to_numpy().astype(np.float64)
        ok = np.isfinite(B).all(axis=1)
        if not ok.all():
            log(f"note: offset needs the baseline on every row; dropping {int((~ok).sum())} rows without it")
            df = df.filter(pl.Series(ok))
            B = B[ok]
        offset = offset_from_baseline(B, task)
        log("offset: booster starts from the baseline and learns a correction to it")

    folds = make_folds(df, mcfg.get("split") or {}, project.order_col, project.period_col)
    m = encode_matrix(df, features)
    y = task.encode(df[task.target])

    passthrough = [c for c in (mcfg.get("passthrough") or []) if c in df.columns]
    keep = _unique(project.id_cols + [project.period_col or "", project.order_col or ""] + baseline + passthrough + [task.target])
    keep = [c for c in keep if c in df.columns]

    log(f"{project.name}/{model_name}  task {task.type} on {task.target}  rows {df.height}  features {len(features)}  folds {len(folds)}  split {mcfg.get('split', {}).get('strategy', 'holdout')}")

    pieces: list[pl.DataFrame] = []
    curves: list[pl.DataFrame] = []
    importances: list[pl.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []
    last_booster = None
    for i, fold in enumerate(folds):
        res = fit_fold(m, y, fold, task, mcfg, offset=offset)
        last_booster = res.booster
        for split, idx, pred in (("valid", fold.valid, res.valid_pred), ("test", fold.test, res.test_pred)):
            if idx.size == 0:
                continue
            part = df[idx].select(keep)
            extra: dict[str, Any] = {"fold": [fold.name] * idx.size, "split": [split] * idx.size, "_y": y[idx]}
            if task.is_classification:
                for k, col in enumerate(task.prob_columns):
                    extra[col] = pred[:, k]
            else:
                extra["pred"] = pred
            pieces.append(part.with_columns([pl.Series(k, v) for k, v in extra.items()]))
        curves.append(res.curve.with_columns(pl.lit(fold.name).alias("fold")))
        importances.append(res.importance.with_columns(pl.lit(fold.name).alias("fold")))
        fm = {"fold": fold.name, "n_train": int(fold.train.size), "n_valid": int(fold.valid.size), "n_test": int(fold.test.size), "best_iteration": res.best_iteration, "valid_loss": float(res.curve["valid"][res.best_iteration - 1]) if res.best_iteration <= res.curve.height else None}
        fold_meta.append(fm)
        log(f"  fold {fold.name:>10}  train {fm['n_train']:5d}  valid {fm['n_valid']:5d}  test {fm['n_test']:5d}  best_iter {res.best_iteration:4d}  valid_loss {fm['valid_loss']:.4f}")

    pred = pl.concat(pieces, how="vertical_relaxed")
    curve = pl.concat(curves, how="vertical_relaxed")
    importance = (
        pl.concat(importances, how="vertical_relaxed")
        .group_by("feature")
        .agg(pl.col("gain").mean(), pl.col("split").mean(), pl.len().alias("folds"))
        .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_share"))
        .sort("gain", descending=True)
    )

    metrics = {split: summary(task, pred.filter(pl.col("split") == split), baseline) for split in ("valid", "test")}
    run_dir = new_run_dir(project, model_name)
    pred.write_parquet(run_dir / "predictions.parquet")
    curve.write_parquet(run_dir / "curve.parquet")
    importance.write_parquet(run_dir / "importance.parquet")
    (run_dir / "features.json").write_text(json.dumps({"features": features, "categorical": [m.names[i] for i in m.categorical]}, indent=2))
    if last_booster is not None:
        last_booster.save_model(str(run_dir / "model.txt"))
    meta = {
        "project": project.name,
        "model": model_name,
        "tag": tag,
        "task": {"type": task.type, "target": task.target, "classes": task.class_names or None},
        "baseline": baseline,
        "offset": bool(offset is not None),
        "rows": int(df.height),
        "n_features": len(features),
        "config": mcfg,
        "folds": fold_meta,
        "metrics": metrics,
        "seconds": round(time.time() - t0, 1),
    }
    write_meta(run_dir, meta)
    te = metrics.get("test") or {}
    line = f"test logloss {te.get('logloss', float('nan')):.4f}" if task.is_classification else f"test rmse {te.get('rmse', float('nan')):.4f}"
    if "baseline_logloss" in te:
        line += f"  baseline {te['baseline_logloss']:.4f}  model-baseline {te['model_minus_baseline']:+.4f}"
    log(f"{line}  ({meta['seconds']}s)")
    log(f"run {run_dir}")
    return run_dir

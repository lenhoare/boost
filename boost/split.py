"""Row splits. Core knows time order and periods; nothing else."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from boost.types import Fold


def make_folds(
    df: pl.DataFrame,
    cfg: dict[str, Any],
    order_col: str | None,
    period_col: str | None,
) -> list[Fold]:
    strategy = str(cfg.get("strategy") or "holdout")
    if strategy == "holdout":
        return [_holdout(df, cfg)]
    if strategy == "walk_forward":
        if not period_col:
            raise ValueError("walk_forward needs roles.period in project.yaml")
        return _walk_forward(df, cfg, period_col)
    if strategy == "random":
        return [_random(df, cfg)]
    raise ValueError(f"unknown split.strategy {strategy!r}")


def _holdout(df: pl.DataFrame, cfg: dict[str, Any]) -> Fold:
    """Rows are already in order. Last (1-train_fraction) is test; the tail of train is valid."""
    n = df.height
    train_frac = float(cfg.get("train_fraction", 0.75))
    valid_frac = float(cfg.get("valid_fraction", 0.15))
    cut = int(round(n * train_frac))
    cut = max(2, min(n - 1, cut))
    v = int(round(cut * valid_frac))
    v = max(1, min(cut - 1, v)) if valid_frac > 0 else 0
    idx = np.arange(n)
    return Fold(name="holdout", train=idx[: cut - v], valid=idx[cut - v : cut], test=idx[cut:])


def _random(df: pl.DataFrame, cfg: dict[str, Any]) -> Fold:
    n = df.height
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    idx = rng.permutation(n)
    train_frac = float(cfg.get("train_fraction", 0.75))
    valid_frac = float(cfg.get("valid_fraction", 0.15))
    cut = int(round(n * train_frac))
    v = int(round(cut * valid_frac))
    return Fold(name="random", train=idx[: cut - v], valid=idx[cut - v : cut], test=idx[cut:])


def _walk_forward(df: pl.DataFrame, cfg: dict[str, Any], period_col: str) -> list[Fold]:
    """One fold per period after the first `min_periods`: train on everything before it.

    valid = the most recent `valid_periods` periods of that training window.
    Predictions across folds together give an out-of-sample series over most of the data.
    """
    min_periods = int(cfg.get("min_periods", 3))
    valid_periods = int(cfg.get("valid_periods", 1))
    periods = df[period_col].to_list()
    uniq = list(dict.fromkeys(periods))  # first-appearance order == time order
    pos = {p: i for i, p in enumerate(uniq)}
    period_idx = np.array([pos[p] for p in periods])
    folds: list[Fold] = []
    for k in range(min_periods, len(uniq)):
        test = np.flatnonzero(period_idx == k)
        valid = np.flatnonzero((period_idx >= k - valid_periods) & (period_idx < k))
        train = np.flatnonzero(period_idx < k - valid_periods)
        if train.size == 0 or valid.size == 0 or test.size == 0:
            continue
        folds.append(Fold(name=str(uniq[k]), train=train, valid=valid, test=test))
    if not folds:
        raise ValueError("walk_forward produced no folds; lower min_periods")
    return folds

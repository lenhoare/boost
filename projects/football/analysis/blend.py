"""Does a forecast add anything to a bookmaker price? Walk-forward two-weight blend.

    p  ∝  book^a × forecast^b       (log-linear pool, then renormalised)

a and b are fitted before each season on all earlier seasons (min_train rows),
then applied to that season. If the forecast carries information the price
lacks, b is reliably positive and the blend beats the price out of sample.
Only two numbers are learned, so unlike boosting it cannot overfit much.

Usage (from the repo root, after `boost -p football features`):
    .venv/bin/python projects/football/analysis/blend.py [--forecast dc] [--from 2014-15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from boost.metrics import logloss_rows, paired  # noqa: E402

BOOKS = {"b365_pre": "book_b365", "pinnacle_pre": "book_ps", "avg_pre": "book_avg", "pinnacle_close": "book_psc",
         "b365_close": "book_b365c", "avg_close": "book_avgc", "betfair_close": "book_bfec"}


def _pool(logs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    z = sum(wi * li for wi, li in zip(w, logs))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def walk_forward_blend(frame: pl.DataFrame, book: str, forecast: str, start: str, min_train: int = 300) -> dict:
    y = np.array(["HDA".index(v) for v in frame["ftr"].to_list()])
    seasons = frame["season"].to_numpy()
    B = frame.select([f"{book}_{k}" for k in "hda"]).to_numpy().astype(float)
    F = frame.select([f"{forecast}_{k}" for k in "hda"]).to_numpy().astype(float)
    ok = np.isfinite(B).all(1) & np.isfinite(F).all(1)
    B = np.where(ok[:, None], B / np.where(ok, B.sum(1), 1.0)[:, None], 1 / 3)
    LB, LF = np.log(B), np.log(np.where(ok[:, None], F, 1 / 3))
    pred = np.full((len(y), 3), np.nan)
    weights = []
    for s in sorted(set(seasons)):
        if s < start:
            continue
        tr, te = ok & (seasons < s), ok & (seasons == s)
        if tr.sum() < min_train or not te.any():
            continue
        w = minimize(lambda w: logloss_rows(_pool([LB[tr], LF[tr]], w), y[tr]).mean(), [1.0, 0.0], method="Nelder-Mead").x
        weights.append({"season": s, "w_book": float(w[0]), "w_forecast": float(w[1])})
        pred[te] = _pool([LB[te], LF[te]], w)
    m = np.isfinite(pred).all(1)
    lm, lb = logloss_rows(pred[m], y[m]), logloss_rows(B[m], y[m])
    return {"n": int(m.sum()), "blend": float(lm.mean()), "book": float(lb.mean()), **paired(lm, lb), "weights": weights}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default="dc", help="column prefix with _h/_d/_a probabilities (dc, dcx, ...)")
    ap.add_argument("--from", dest="start", default="2014-15", help="first season scored")
    args = ap.parse_args()
    frame = pl.read_parquet(ROOT / "projects/football/data/features.parquet").sort(["date", "match_id"])
    print(f"forecast {args.forecast}: walk-forward blend with each price, seasons from {args.start}")
    print(f"{'price':15s} {'n':>5s} {'blend':>7s} {'price':>7s} {'diff':>8s} {'se':>7s} {'z':>5s}  mean w_forecast")
    for name, col in BOOKS.items():
        r = walk_forward_blend(frame, col, args.forecast, args.start)
        if not r["weights"]:
            continue
        wf = np.mean([w["w_forecast"] for w in r["weights"]])
        print(f"{name:15s} {r['n']:5d} {r['blend']:7.4f} {r['book']:7.4f} {r['diff']:+8.4f} {r['diff_se']:7.4f} {r['diff_z']:+5.1f}  {wf:+.3f}")


if __name__ == "__main__":
    main()

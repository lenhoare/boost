"""Can a model that is worse on average still pick winning bets? (step 6)

Logloss asks "would I make a better bookmaker?". A punter only needs the model to be right where
it disagrees with the price and chooses to bet. Two tests, walk-forward forecasts only:

1. Disagreement calibration. For every match and outcome, q = model probability, m = market
   probability (Pinnacle pre-match, power de-vig; market average where Pinnacle is missing).
   Regress (result - m) on (q - m) through the origin:
       slope 0 -> where they disagree the market is right; slope 1 -> the model is right.
   Overall, by size of disagreement, and in pre-chosen segments (outcome, odds band, early season,
   promoted club involved).

2. Bet selection. Back each outcome whose model EV = q x O - 1 clears a threshold, at Bet365,
   best-UK or Pinnacle pre-match prices. CLV against Pinnacle's close (power de-vig), ROI with a
   bootstrap interval.

Models: dc (Dixon-Coles), dcx (DC half xG), dcf (DC + Elo + pre-match xG, step 5 logit, no prices).

Usage (repo root):  .venv/bin/python projects/football/analysis/bet_select.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects/football/analysis"))
from exchange_eval import devig  # noqa: E402
import feature_blend as fb  # noqa: E402

FIRST = "2014-15"


def slope(d: np.ndarray, r: np.ndarray) -> tuple[float, float, int]:
    """OLS through the origin of r on d, with a match-clustered-ish (plain) SE."""
    ok = np.isfinite(d) & np.isfinite(r)
    d, r = d[ok], r[ok]
    if d.size < 30:
        return np.nan, np.nan, int(d.size)
    b = (d * r).sum() / (d * d).sum()
    res = r - b * d
    se = np.sqrt((d * d * res * res).sum()) / (d * d).sum()   # heteroskedasticity-robust
    return float(b), float(se), int(d.size)


def dcf_forecast(f: pl.DataFrame) -> np.ndarray:
    y = np.array(["HDA".index(v) for v in f["ftr"]])
    seasons = f["season"].to_numpy()
    feats = fb.feature_matrix(f)
    LD = fb.logs(f, "dc")
    X = np.column_stack([feats["elo"], feats["xg_pre"]])
    pred, _ = fb.walk_forward(f, [LD], X, y, seasons, FIRST, np.isfinite(LD).all(1))
    return pred


def bets(q, O, close, y, thr, mask, rng):
    ev = q * O - 1.0
    ok = mask[:, None] & np.isfinite(ev) & np.isfinite(close)
    take = ok & (ev > thr)
    n = int(take.sum())
    if n < 20:
        return f"bets {n:5d}"
    rows, cols = np.nonzero(take)
    o, pc = O[rows, cols], close[rows, cols]
    pnl = np.where(y[rows] == cols, o - 1.0, -1.0)
    clv = pc * o - 1.0
    boot = lambda x: np.quantile(x[rng.integers(0, x.size, (3000, x.size))].mean(1), [0.025, 0.975])  # noqa: E731
    (a, b), (c, d) = boot(clv), boot(pnl)
    return f"bets {n:5d}  odds {o.mean():5.2f}  CLV {clv.mean():+.3f} [{a:+.3f},{b:+.3f}]  ROI {pnl.mean():+.3f} [{c:+.3f},{d:+.3f}]"


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    f = pl.read_parquet(fb.FEATURES).sort(["date", "match_id"])
    y = np.array(["HDA".index(v) for v in f["ftr"]])
    seasons = f["season"].to_numpy()
    O = lambda s: f.select([f"{s}_{k}" for k in "hda"]).to_numpy().astype(np.float64)  # noqa: E731
    PS, AVG, PSC = devig(O("odds_ps"), "power"), devig(O("odds_avg"), "power"), devig(O("odds_psc"), "power")
    M = np.where(np.isfinite(PS).all(1, keepdims=True), PS, AVG)
    Y = np.eye(3)[y]
    models = {"dc": f.select(["dc_h", "dc_d", "dc_a"]).to_numpy(), "dcx": f.select(["dcx_h", "dcx_d", "dcx_a"]).to_numpy(),
              "dcf": dcf_forecast(f)}
    test = (seasons >= FIRST) & np.isfinite(M).all(1)
    rng = np.random.default_rng(11)

    early = f["days_into_season"].to_numpy() < 60
    promoted = (f["h_newcomer"].to_numpy() + f["a_newcomer"].to_numpy()) > 0
    seg_rows = {"early season (<60 days)": early, "rest of season": ~early,
                "promoted club involved": promoted, "no promoted club": ~promoted}

    print(f"Test seasons {FIRST} on, {int(test.sum())} matches. Market = Pinnacle pre (power de-vig), else average pre.\n")
    print("1. Where model and market disagree, who is right?  slope of (result - m) on (q - m): 0 = market, 1 = model")
    for name, Q in models.items():
        ok = test & np.isfinite(Q).all(1)
        D, R = (Q - M)[ok], (Y - M)[ok]
        b, se, n = slope(D.ravel(), R.ravel())
        print(f"\n   {name}: all outcomes  slope {b:+.3f} ± {se:.3f}  (n {n})   mean |q - m| {np.abs(D).mean():.3f}")
        absd = np.abs(D)
        for lo, hi in ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 1.0)):
            m = (absd >= lo) & (absd < hi)
            b, se, n = slope(D[m], R[m])
            hit = (R[m] * np.sign(D[m])).mean()
            print(f"      |q-m| {lo:.2f}-{hi:.2f}  slope {b:+.3f} ± {se:.3f}  n {n:5d}   mean (result - m) in the model's direction {hit:+.4f} vs model claims {absd[m].mean():.4f}")
        for k, lab in enumerate(("home", "draw", "away")):
            b, se, n = slope(D[:, k], R[:, k])
            print(f"      {lab:5s}      slope {b:+.3f} ± {se:.3f}")
        mk = M[ok]
        for lo, hi, lab in ((0.5, 1.0, "favourites (m>0.5)"), (0.25, 0.5, "middle (0.25-0.5)"), (0.0, 0.25, "outsiders (m<0.25)")):
            m = (mk >= lo) & (mk < hi)
            b, se, n = slope(D[m], R[m])
            print(f"      {lab:20s} slope {b:+.3f} ± {se:.3f}  n {n}")
        for lab, rows in seg_rows.items():
            m = rows[ok]
            b, se, n = slope(D[m].ravel(), R[m].ravel())
            print(f"      {lab:24s} slope {b:+.3f} ± {se:.3f}")

    print("\n2. Bets: every outcome with model EV > threshold. CLV vs Pinnacle close (power); ROI realised")
    for name, Q in models.items():
        for book, stem in (("Bet365 pre", "odds_b365"), ("best UK pre", "odds_bestuk"), ("Pinnacle pre", "odds_ps")):
            Ob = O(stem)
            for thr in (0.0, 0.05, 0.10, 0.20):
                print(f"   {name:4s} {book:12s} EV>{thr:.2f}  {bets(Q, Ob, PSC, y, thr, test, rng)}")
        print()
    print("   reference: best UK pre with fair = Pinnacle pre (step 3 rule), EV>0.02")
    print(f"   {'':4s} {'best UK pre':12s} EV>0.02  {bets(PS, O('odds_bestuk'), PSC, y, 0.02, test, rng)}")


if __name__ == "__main__":
    main()

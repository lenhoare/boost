"""Value betting at the Betfair exchange pre-match price, net of commission (2024-25 on).

Exchanges do not restrict winning accounts, so an edge here is one that can keep being used.
A winning back bet at decimal odds O pays (O - 1) x (1 - commission); a loser costs the stake.

    net EV = p (O - 1)(1 - c) - (1 - p)

For each fair-price source (Pinnacle pre-match, market-average pre-match, and the line model's
predicted Pinnacle close where it exists) back the side with the best net EV when it clears a
threshold. CLV = expected net profit if the closing price is the truth, against the exchange's own
close and against Pinnacle's close. ROI is what happened. Both de-vig methods are reported,
because the proportional one overstates longshots.

football-data's exchange price is a back price; lay prices are not supplied, so only backing is tested.

Usage (repo root):  .venv/bin/python projects/football/analysis/exchange_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "projects/football/data/features.parquet"
RUNS = ROOT / "projects/football/runs"


def devig(O: np.ndarray, method: str) -> np.ndarray:
    q = 1.0 / O
    if method == "prop":
        return q / q.sum(axis=1, keepdims=True)
    P = np.full(O.shape, np.nan)
    for i, row in enumerate(q):
        if np.isfinite(row).all():
            k = brentq(lambda k: (row ** k).sum() - 1.0, 0.3, 3.0)
            P[i] = row ** k
    return P


def line_pred(f: pl.DataFrame) -> np.ndarray | None:
    """Predicted log moves from the latest line_move_{h,d,a} runs, aligned to f (NaN where absent)."""
    out = np.full((f.height, 3), np.nan)
    ids = f["match_id"].to_numpy()
    for i, k in enumerate("hda"):
        runs = sorted(p for p in RUNS.glob(f"*-line_move_{k}") if (p / "predictions.parquet").exists())
        if not runs:
            return None
        p = pl.read_parquet(runs[-1] / "predictions.parquet").filter(pl.col("split") == "test")
        m = dict(zip(p["match_id"].to_list(), p["pred"].to_list()))
        out[:, i] = [m.get(int(x), np.nan) for x in ids]
    return out


def ledger(fair, O, close, y, c, thr, rng):
    ok = np.isfinite(fair).all(1) & np.isfinite(O).all(1) & np.isfinite(close).all(1)
    ev = fair * (O - 1.0) * (1.0 - c) - (1.0 - fair)
    ev[~ok] = -9.0
    idx = np.arange(len(y))
    side = ev.argmax(1)
    take = ok & (ev[idx, side] > thr)
    n = int(take.sum())
    if n < 2:
        return {"bets": n, "rows": int(ok.sum())}
    o, pc = O[idx, side][take], close[idx, side][take]
    pnl = np.where((y == side)[take], (o - 1.0) * (1.0 - c), -1.0)
    clv = pc * (o - 1.0) * (1.0 - c) - (1.0 - pc)
    boot = lambda x: np.quantile(x[rng.integers(0, x.size, (4000, x.size))].mean(1), [0.025, 0.975])  # noqa: E731
    return {"bets": n, "rows": int(ok.sum()), "odds": float(o.mean()), "clv": float(clv.mean()), "clv_ci": boot(clv),
            "roi": float(pnl.mean()), "roi_ci": boot(pnl)}


def main() -> None:
    f = pl.read_parquet(FEATURES).filter(pl.col("odds_bfe_h").is_not_null() & pl.col("odds_bfec_h").is_not_null()).sort(["date", "match_id"])
    y = np.array(["HDA".index(v) for v in f["ftr"].to_list()])
    O = lambda s: f.select([f"{s}_{k}" for k in "hda"]).to_numpy().astype(np.float64)  # noqa: E731
    rng = np.random.default_rng(7)
    bfe = O("odds_bfe")
    over = (1.0 / bfe).sum(1)
    print(f"Betfair exchange pre-match: {f.height} matches ({f['season'].min()} .. {f['season'].max()}), back-price book sums to {over.mean():.4f} on average")
    print(f"Pinnacle pre-match on {int(np.isfinite(O('odds_ps')).all(1).sum())} of them\n")
    lp = line_pred(f)
    for method in ("prop", "power"):
        PS, AVG, PSC, BFEC = devig(O("odds_ps"), method), devig(O("odds_avg"), method), devig(O("odds_psc"), method), devig(O("odds_bfec"), method)
        fairs = {"pinnacle_pre": PS, "average_pre": AVG}
        if lp is not None:
            P = PS * np.exp(lp)
            fairs["predicted_close"] = P / P.sum(1, keepdims=True)
        print(f"== de-vig {method}")
        print(f"   {'fair':16s} {'comm':>4s} {'thr':>4s} {'bets':>5s} {'odds':>5s}  {'CLV vs exchange close':>26s}  {'CLV vs Pinnacle close':>26s}  {'ROI':>24s}")
        for name, fair in fairs.items():
            for c in (0.02, 0.05):
                for thr in (0.0, 0.02):
                    a = ledger(fair, bfe, BFEC, y, c, thr, rng)
                    b = ledger(fair, bfe, PSC, y, c, thr, rng)
                    if a["bets"] < 2:
                        print(f"   {name:16s} {c:4.0%} {thr:4.2f} {a['bets']:5d}"); continue
                    pin = f"{b['clv']:+.3f} [{b['clv_ci'][0]:+.3f},{b['clv_ci'][1]:+.3f}] n{b['bets']:4d}" if b["bets"] >= 2 else "n/a"
                    print(f"   {name:16s} {c:4.0%} {thr:4.2f} {a['bets']:5d} {a['odds']:5.2f}  {a['clv']:+.3f} [{a['clv_ci'][0]:+.3f},{a['clv_ci'][1]:+.3f}]       {pin:>26s}  {a['roi']:+.3f} [{a['roi_ci'][0]:+.3f},{a['roi_ci'][1]:+.3f}]")
        print()


if __name__ == "__main__":
    sys.exit(main())

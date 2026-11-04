"""Evaluate the line model: the three line_move_{h,d,a} runs combined into a predicted Pinnacle close.

    predicted close  ∝  Pinnacle pre-match probability × exp(predicted log move)

Questions, all on walk-forward test seasons (2014-15 on, Pinnacle seasons only):
  1. Moves: out-of-sample R² against "no move", and against a simple linear rule
     (move = b × (market average − Pinnacle), refitted each season on earlier seasons).
  2. Outcomes: logloss of Pinnacle pre-match, predicted close and the actual close.
  3. Bets: back the best-EV side at a pre-match price when EV > threshold, with EV computed from
     either Pinnacle pre-match (no model) or the predicted close. Closing-line value (CLV) is the
     expected profit if the actual Pinnacle close is the truth; ROI is what really happened.

Usage (repo root, after training line_move_h, line_move_d, line_move_a):
    .venv/bin/python projects/football/analysis/line_eval.py [--runs latest]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from boost.metrics import logloss_rows, paired  # noqa: E402

RUNS = ROOT / "projects/football/runs"
FEATURES = ROOT / "projects/football/data/features.parquet"
ERAS = (("2014-15", "2017-18"), ("2018-19", "2021-22"), ("2022-23", "2026-27"))


def latest_run(model: str) -> Path:
    runs = sorted(p for p in RUNS.glob(f"*-{model}") if (p / "predictions.parquet").exists())
    if not runs:
        raise FileNotFoundError(f"no run for {model}; train it first")
    return runs[-1]


def load() -> tuple[pl.DataFrame, list[str]]:
    used = []
    preds = None
    for k in "hda":
        run = latest_run(f"line_move_{k}")
        used.append(run.name)
        p = pl.read_parquet(run / "predictions.parquet").filter(pl.col("split") == "test").select(["match_id", pl.col("pred").alias(f"pred_{k}")])
        preds = p if preds is None else preds.join(p, on="match_id", how="inner")
    f = pl.read_parquet(FEATURES)
    return f.join(preds, on="match_id", how="inner").sort(["date", "match_id"]), used


def probs(f: pl.DataFrame, stem: str) -> np.ndarray:
    O = f.select([f"{stem}_{k}" for k in "hda"]).to_numpy().astype(np.float64)
    q = 1.0 / O
    return q / q.sum(axis=1, keepdims=True)


def linear_rule(f: pl.DataFrame) -> np.ndarray:
    """Walk-forward per-outcome OLS: move_k = a_k + b_k × ln_avg_k, fitted on earlier test-era seasons
    plus the two training seasons before them."""
    all_f = pl.read_parquet(FEATURES).filter(pl.col("mv_ps_h").is_not_null() & pl.col("ln_ps_h").is_not_null())
    out = np.full((f.height, 3), np.nan)
    seasons = f["season"].to_numpy()
    for s in sorted(set(seasons)):
        tr = all_f.filter(pl.col("season") < s)
        m = seasons == s
        for i, k in enumerate("hda"):
            x, y = tr[f"ln_avg_{k}"].to_numpy(), tr[f"mv_ps_{k}"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            b, a = np.polyfit(x[ok], y[ok], 1)
            out[m, i] = a + b * f.filter(pl.Series(m))[f"ln_avg_{k}"].to_numpy()
    return out


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(1.0 - ((y[ok] - pred[ok]) ** 2).sum() / (y[ok] ** 2).sum())


def bets(fair: np.ndarray, odds: np.ndarray, close: np.ndarray, y: np.ndarray, thr: float, mask: np.ndarray, rng) -> dict:
    ok = mask & np.isfinite(fair).all(1) & np.isfinite(odds).all(1) & np.isfinite(close).all(1)
    ev = fair * odds - 1.0
    ev[~ok] = -9.0
    side = ev.argmax(1)
    idx = np.arange(len(y))
    take = ok & (ev[idx, side] > thr)
    n = int(take.sum())
    if n < 2:
        return {"bets": n}
    o = odds[idx, side][take]
    pnl = np.where((y == side)[take], o - 1.0, -1.0)
    clv = close[idx, side][take] * o - 1.0
    boot = lambda x: np.quantile(x[rng.integers(0, x.size, (2000, x.size))].mean(1), [0.025, 0.975])  # noqa: E731
    (rlo, rhi), (clo, chi) = boot(pnl), boot(clv)
    return {"bets": n, "share": n / ok.sum(), "avg_odds": float(o.mean()), "clv": float(clv.mean()), "clv_lo": clo, "clv_hi": chi,
            "roi": float(pnl.mean()), "roi_lo": rlo, "roi_hi": rhi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the results here")
    args = ap.parse_args()
    f, used = load()
    rng = np.random.default_rng(7)
    y = np.array(["HDA".index(v) for v in f["ftr"].to_list()])
    seasons = f["season"].to_numpy()
    PS, PSC = probs(f, "odds_ps"), probs(f, "odds_psc")
    move = f.select([f"mv_ps_{k}" for k in "hda"]).to_numpy()
    pm = f.select([f"pred_{k}" for k in "hda"]).to_numpy()
    lin = linear_rule(f)
    PRED = PS * np.exp(pm)
    PRED /= PRED.sum(1, keepdims=True)
    LIN = PS * np.exp(lin)
    LIN /= LIN.sum(1, keepdims=True)
    res: dict = {"runs": used, "n": int(f.height)}

    print(f"line model runs: {', '.join(used)}   test matches {f.height} ({seasons.min()} .. {seasons.max()})\n")
    print("1. Moves (log close - log pre, Pinnacle): out-of-sample R² vs 'no move'")
    res["r2"] = {}
    for i, k in enumerate("hda"):
        res["r2"][k] = {"booster": r2(pm[:, i], move[:, i]), "linear": r2(lin[:, i], move[:, i])}
        print(f"   {k}: booster {res['r2'][k]['booster']:+.3f}   linear rule {res['r2'][k]['linear']:+.3f}")

    print("\n2. Outcomes: logloss (lower is better), paired against Pinnacle pre-match")
    l_pre = logloss_rows(PS, y)
    res["logloss"] = {}
    for name, P in (("pinnacle_pre", PS), ("linear_close", LIN), ("predicted_close", PRED), ("pinnacle_close", PSC)):
        lm = logloss_rows(P, y)
        d = paired(lm, l_pre) if name != "pinnacle_pre" else {}
        res["logloss"][name] = {"logloss": float(lm.mean()), **d}
        extra = f"  diff {d['diff']:+.4f} ± {d['diff_se']:.4f} (z {d['diff_z']:+.1f})" if d else ""
        print(f"   {name:16s} {lm.mean():.4f}{extra}")
    gap = res["logloss"]["pinnacle_pre"]["logloss"] - res["logloss"]["pinnacle_close"]["logloss"]
    got = res["logloss"]["pinnacle_pre"]["logloss"] - res["logloss"]["predicted_close"]["logloss"]
    print(f"   predicted close recovers {got / gap:.0%} of the pre-to-close improvement")

    print("\n3. Bets on the best-EV side if EV > threshold. CLV vs Pinnacle close; 95% bootstrap intervals")
    res["bets"] = []
    everything = np.ones(len(y), bool)
    for book, stem in (("Pinnacle pre", "odds_ps"), ("best UK pre", "odds_bestuk"), ("Bet365 pre", "odds_b365")):
        O = f.select([f"{stem}_{k}" for k in "hda"]).to_numpy().astype(np.float64)
        for thr in (0.0, 0.02, 0.05):
            for fair_name, fair in (("pinnacle_pre", PS), ("predicted_close", PRED)):
                if book == "Pinnacle pre" and fair_name == "pinnacle_pre":
                    continue  # a de-vigged price never shows value at its own odds
                b = bets(fair, O, PSC, y, thr, everything, rng)
                res["bets"].append({"book": book, "fair": fair_name, "thr": thr, **b})
                if b["bets"] < 2:
                    print(f"   {book:13s} fair={fair_name:16s} thr {thr:.2f}  bets {b['bets']}")
                    continue
                print(f"   {book:13s} fair={fair_name:16s} thr {thr:.2f}  bets {b['bets']:5d} ({b['share']:4.0%})  odds {b['avg_odds']:5.2f}"
                      f"  CLV {b['clv']:+.3f} [{b['clv_lo']:+.3f},{b['clv_hi']:+.3f}]  ROI {b['roi']:+.3f} [{b['roi_lo']:+.3f},{b['roi_hi']:+.3f}]")

    print("\n   by era, fair = predicted close, EV > 0.02")
    for lo, hi in ERAS:
        m = (seasons >= lo) & (seasons <= hi)
        for book, stem in (("Pinnacle pre", "odds_ps"), ("best UK pre", "odds_bestuk")):
            O = f.select([f"{stem}_{k}" for k in "hda"]).to_numpy().astype(np.float64)
            b = bets(PRED, O, PSC, y, 0.02, m, rng)
            res["bets"].append({"book": book, "fair": "predicted_close", "thr": 0.02, "era": f"{lo}..{hi}", **b})
            if b["bets"] < 2:
                print(f"   {lo}..{hi} {book:13s} bets {b['bets']}")
                continue
            print(f"   {lo}..{hi} {book:13s} bets {b['bets']:5d}  CLV {b['clv']:+.3f} [{b['clv_lo']:+.3f},{b['clv_hi']:+.3f}]  ROI {b['roi']:+.3f} [{b['roi_lo']:+.3f},{b['roi_hi']:+.3f}]")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()

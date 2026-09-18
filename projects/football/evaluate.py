"""Betting evaluation hook: is the model's edge over the book real, and is it bettable?

Runs on test rows only. Needs p_H/p_D/p_A, the de-vigged book columns and the
raw odds passed through by the model config.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from boost.report import PALETTE, save_fig, style
from boost.types import RunContext, Section

SIDES = (("H", "odds_home", "book_h"), ("D", "odds_draw", "book_d"), ("A", "odds_away", "book_a"))


def _arrays(pred: pl.DataFrame) -> dict[str, np.ndarray]:
    P = pred.select(["p_H", "p_D", "p_A"]).to_numpy()
    O = pred.select([o for _, o, _ in SIDES]).to_numpy().astype(np.float64)
    B = pred.select([b for _, _, b in SIDES]).to_numpy().astype(np.float64)
    y = pred["_y"].to_numpy().astype(np.int64)
    ok = np.isfinite(O).all(axis=1) & (O > 1.01).all(axis=1) & np.isfinite(B).all(axis=1)
    return {"P": P, "O": O, "B": B, "y": y, "ok": ok}


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    return float((peak[1:] - equity).max())


def _bootstrap_ci(profits: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    if profits.size < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, profits.size, size=(n, profits.size))
    means = profits[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _bets(a: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    """One flat-stake bet per match on the side with the highest EV, if EV > threshold."""
    P, O, y, ok = a["P"], a["O"], a["y"], a["ok"]
    ev = P * O - 1.0
    ev[~ok] = -np.inf
    side = ev.argmax(axis=1)
    best = ev[np.arange(len(y)), side]
    take = ok & (best > threshold)
    odds = O[np.arange(len(y)), side]
    won = y == side
    profit = np.where(won, odds - 1.0, -1.0)
    return {"take": take, "side": side, "odds": odds, "won": won, "profit": profit, "ev": best, "p": P[np.arange(len(y)), side]}


def _strategy_table(a: dict[str, np.ndarray], thresholds: list[float], n_boot: int, rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    for thr in thresholds:
        b = _bets(a, thr)
        t = b["take"]
        n = int(t.sum())
        if n == 0:
            rows.append({"ev_threshold": thr, "bets": 0}); continue
        pr = b["profit"][t]
        lo, hi = _bootstrap_ci(pr, n_boot, rng)
        rows.append({
            "ev_threshold": thr, "bets": n, "matches": int(a["ok"].sum()), "hit_rate": float(b["won"][t].mean()),
            "avg_odds": float(b["odds"][t].mean()), "avg_model_p": float(b["p"][t].mean()),
            "profit": float(pr.sum()), "roi": float(pr.mean()), "roi_ci_lo": lo, "roi_ci_hi": hi,
            "max_drawdown": _max_drawdown(np.cumsum(pr)),
            "bets_H": int((b["side"][t] == 0).sum()), "bets_D": int((b["side"][t] == 1).sum()), "bets_A": int((b["side"][t] == 2).sum()),
        })
    return pl.DataFrame(rows)


def _blind_table(a: dict[str, np.ndarray]) -> pl.DataFrame:
    """What you get with no model: back every home / draw / away, or always the book favourite."""
    O, B, y, ok = a["O"], a["B"], a["y"], a["ok"]
    rows = []
    for k, (name, _, _) in enumerate(SIDES):
        pr = np.where(y[ok] == k, O[ok, k] - 1.0, -1.0)
        rows.append({"strategy": f"always {name}", "bets": int(ok.sum()), "roi": float(pr.mean()), "hit_rate": float((y[ok] == k).mean())})
    fav = B.argmax(axis=1)
    pr = np.where(y[ok] == fav[ok], O[ok, fav[ok]] - 1.0, -1.0)
    rows.append({"strategy": "book favourite", "bets": int(ok.sum()), "roi": float(pr.mean()), "hit_rate": float((y[ok] == fav[ok]).mean())})
    return pl.DataFrame(rows)


def _by_group(a: dict[str, np.ndarray], groups: np.ndarray, threshold: float, label: str) -> pl.DataFrame:
    b = _bets(a, threshold)
    rows = []
    for g in list(dict.fromkeys(groups.tolist())):
        m = (groups == g) & b["take"]
        n = int(m.sum())
        row: dict[str, Any] = {label: g, "matches": int(((groups == g) & a["ok"]).sum()), "bets": n}
        if n:
            pr = b["profit"][m]
            row.update({"hit_rate": float(b["won"][m].mean()), "avg_odds": float(b["odds"][m].mean()), "profit": float(pr.sum()), "roi": float(pr.mean())})
        rows.append(row)
    return pl.DataFrame(rows)


def _by_side(a: dict[str, np.ndarray], threshold: float) -> pl.DataFrame:
    b = _bets(a, threshold)
    rows = []
    for k, (name, _, _) in enumerate(SIDES):
        m = b["take"] & (b["side"] == k)
        n = int(m.sum())
        row: dict[str, Any] = {"side": name, "bets": n}
        if n:
            pr = b["profit"][m]
            row.update({"hit_rate": float(b["won"][m].mean()), "avg_odds": float(b["odds"][m].mean()), "avg_model_p": float(b["p"][m].mean()), "avg_book_p": float(1.0 / b["odds"][m]).mean() if False else float((1.0 / b["odds"][m]).mean()), "profit": float(pr.sum()), "roi": float(pr.mean())})
        rows.append(row)
    return pl.DataFrame(rows)


def _edge_reliability(a: dict[str, np.ndarray]) -> pl.DataFrame:
    """Bin every (match, side) by model p minus book p. Does the observed rate follow the model or the book?"""
    P, B, y, ok = a["P"], a["B"], a["y"], a["ok"]
    rows = []
    edges = [-1.0, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 1.0]
    diff = (P - B)[ok]
    hit = np.stack([(y[ok] == k) for k in range(3)], axis=1).astype(float)
    p, b = P[ok], B[ok]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (diff >= lo) & (diff < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "model_minus_book": f"{lo:+.2f} to {hi:+.2f}", "n": n,
            "model_p": float(p[m].mean()), "book_p": float(b[m].mean()), "observed": float(hit[m].mean()),
            "closer_to": "model" if abs(hit[m].mean() - p[m].mean()) < abs(hit[m].mean() - b[m].mean()) else "book",
        })
    return pl.DataFrame(rows)


def _kelly(a: dict[str, np.ndarray], fraction: float, threshold: float = 0.0) -> tuple[np.ndarray, dict[str, float]]:
    """Fractional Kelly on the best-EV side, bankroll starting at 1.0. Returns log-equity path."""
    b = _bets(a, threshold)
    P, O = b["p"], b["odds"]
    f = np.clip((P * O - 1.0) / (O - 1.0), 0.0, 1.0) * fraction
    f[~b["take"]] = 0.0
    growth = np.where(b["won"], 1.0 + f * (O - 1.0), 1.0 - f)
    growth[~b["take"]] = 1.0
    equity = np.cumprod(growth)
    stats = {"final_bankroll": float(equity[-1]) if equity.size else 1.0, "bets": int(b["take"].sum()), "avg_stake_frac": float(f[b["take"]].mean()) if b["take"].any() else 0.0, "max_drawdown_frac": float(1.0 - (equity / np.maximum.accumulate(equity)).min()) if equity.size else 0.0}
    return equity, stats


def evaluate(ctx: RunContext) -> list[Section]:
    pred = ctx.predictions.filter(pl.col("split") == "test")
    needed = ["p_H", "p_D", "p_A", "odds_home", "odds_draw", "odds_away", "book_h", "book_d", "book_a"]
    if any(c not in pred.columns for c in needed) or pred.height == 0:
        return [Section("Betting", {}, [], ["Betting analysis needs a 1X2 model with odds passed through (passthrough: [odds_home, odds_draw, odds_away])."])]
    style()
    ecfg = dict((ctx.project.cfg.get("evaluation") or {}))
    thresholds = [float(t) for t in ecfg.get("ev_thresholds", [0.0, 0.05, 0.10])]
    kelly_frac = float(ecfg.get("kelly_fraction", 0.25))
    n_boot = int(ecfg.get("bootstrap", 2000))
    rng = np.random.default_rng(7)
    a = _arrays(pred)
    mid = thresholds[1] if len(thresholds) > 1 else thresholds[0]

    sections: list[Section] = []
    strat = _strategy_table(a, thresholds, n_boot, rng)
    blind = _blind_table(a)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, thr in enumerate(thresholds):
        b = _bets(a, thr)
        if b["take"].any():
            eq = np.cumsum(np.where(b["take"], b["profit"], 0.0))
            ax.plot(np.arange(len(eq)), eq, color=PALETTE[i % len(PALETTE)], lw=1.2, label=f"EV > {thr:.2f} ({int(b['take'].sum())} bets)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("test match (time order)"); ax.set_ylabel("units"); ax.set_title("Flat-stake equity, 1 unit per bet"); ax.legend()
    sections.append(Section("Betting: flat stakes", {"By EV threshold": strat, "No-model baselines": blind}, [save_fig(fig, ctx.run_dir, "equity.png")], [
        "One bet per match on the side with the highest expected value (model p × odds − 1), when that EV clears the threshold. Stake 1 unit.",
        "`roi` is mean profit per unit staked; `roi_ci_lo/hi` is a 95% bootstrap interval over the bets placed. An interval that includes zero is not evidence of an edge.",
        "Odds are closing prices. Betting at these prices assumes you can get the close, which for an exchange is optimistic.",
    ]))

    sections.append(Section(f"Betting: breakdown at EV > {mid:.2f}", {
        "By side": _by_side(a, mid),
        "By season": _by_group(a, pred["season"].to_numpy().astype(str), mid, "season") if "season" in pred.columns else pl.DataFrame(),
        "By book source": _by_group(a, pred["odds_source"].to_numpy().astype(str), mid, "odds_source") if "odds_source" in pred.columns else pl.DataFrame(),
        "By odds band": _by_group(a, np.array([f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(*_odds_band(_bets(a, mid)["odds"]))]), mid, "odds_band"),
    }, [], []))

    edge = _edge_reliability(a)
    equity, kstats = _kelly(a, kelly_frac)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(equity, color=PALETTE[0], lw=1.2)
    ax.axhline(1.0, color="#999", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("test match"); ax.set_ylabel("bankroll (start 1)"); ax.set_title(f"{kelly_frac:.2f} Kelly on any positive EV")
    sections.append(Section("Betting: is the edge information?", {
        "Edge reliability": edge,
        "Fractional Kelly": pl.DataFrame([kstats]),
    }, [save_fig(fig, ctx.run_dir, "kelly.png")], [
        "Edge reliability bins every (match, side) by model probability minus book probability. If `observed` tracks `model_p` in the outer bins, the disagreement carries information. If it tracks `book_p`, the model is noise around the book.",
    ]))
    return sections


def _odds_band(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 1000.0])
    idx = np.clip(np.digitize(odds, edges) - 1, 0, len(edges) - 2)
    return edges[idx], edges[idx + 1]

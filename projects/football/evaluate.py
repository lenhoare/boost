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

# Betting terms for the run page tooltips. Core supplies the generic ones.
GLOSSARY = {
    "H": "Home win.",
    "D": "Draw.",
    "A": "Away win.",
    "ev_threshold": "Minimum expected value to place a bet. Expected value is model probability times decimal odds minus 1: the profit per unit staked if the model's probability were exactly right. 0 means bet whenever the model thinks the price is fair or better.",
    "bets": "Number of bets placed (at most one per match, on the side with the highest expected value).",
    "matches": "Matches available to bet on, i.e. with valid prices.",
    "hit_rate": "Fraction of bets that won. On its own it means little: backing long shots wins rarely but pays a lot.",
    "avg_odds": "Average decimal odds of the bets placed. Decimal odds of 3.0 return 3 units for a 1 unit stake if the bet wins, so a profit of 2.",
    "avg_model_p": "Average probability the model gave to the sides it backed.",
    "avg_book_p": "Average probability implied by the odds on the sides backed (1 divided by the odds).",
    "profit": "Total units won or lost, with 1 unit staked per bet.",
    "roi": "Return on investment: profit divided by total stake, i.e. average profit per unit bet. -0.05 means you lose 5p per pound staked.",
    "roi_ci_lo": "Lower end of a 95% bootstrap confidence interval for roi. Resamples the bets many times to see how much the roi could move by luck.",
    "roi_ci_hi": "Upper end of that interval. If the interval contains zero, the result is consistent with no edge at all.",
    "max_drawdown": "The largest fall in units from a running peak to a later trough. How much you would have been down at the worst moment.",
    "bets_H": "Bets placed on home wins.",
    "bets_D": "Bets placed on draws.",
    "bets_A": "Bets placed on away wins.",
    "strategy": "A rule for what to bet on with no model at all, as a sanity check.",
    "always H": "Back the home side in every match at the closing price.",
    "always D": "Back the draw in every match.",
    "always A": "Back the away side in every match.",
    "book favourite": "Back whichever side the bookmaker makes most likely.",
    "side": "Which outcome was backed.",
    "odds_band": "Bets grouped by the decimal odds taken.",
    "odds_source": "Which bookmaker or exchange the closing prices come from. Bet365 prices carry a 3 to 5% margin; the Betfair exchange under 1%.",
    "model_minus_book": "The model's probability minus the bookmaker's, for one side of one match, grouped into bands.",
    "model_p": "Average model probability in the band.",
    "book_p": "Average bookmaker probability in the band (margin removed).",
    "closer_to": "Whether the observed frequency in this band sits nearer the model's number or the bookmaker's. If it is always 'book', the model's disagreements with the market are noise.",
    "final_bankroll": "Where a bankroll that started at 1.0 ended after staking a fraction of it on every positive-value bet.",
    "avg_stake_frac": "Average fraction of the bankroll staked per bet.",
    "max_drawdown_frac": "Largest fractional fall of the bankroll from a peak. 0.5 means it halved at some point.",
    "season": "Premier League season, August to May.",
}


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
        "This section asks the practical question: if you had bet 1 unit whenever the model thought the price was good, what would have happened? Test rows only, in time order.",
        "Expected value of a bet is model probability times decimal odds minus 1. It is the profit per unit you would expect if the model's probability were exactly right. The model bets on the side with the highest expected value in each match, if that clears the threshold.",
        "Odds are closing prices. Betting at these prices assumes you can get the close, which for an exchange is optimistic.",
    ], {
        "By EV threshold": "One row per threshold. `roi` is the average profit per unit staked and is the number that matters. `roi_ci_lo` and `roi_ci_hi` bound how much luck could move it; if that range includes zero the result is consistent with no edge.",
        "No-model baselines": "What blind rules return at the same prices. The bookmaker's margin makes all of these negative. A model has to beat these, not zero, to be adding anything.",
    }))

    sections.append(Section(f"Betting: breakdown at EV > {mid:.2f}", {
        "By side": _by_side(a, mid),
        "By season": _by_group(a, pred["season"].to_numpy().astype(str), mid, "season") if "season" in pred.columns else pl.DataFrame(),
        "By book source": _by_group(a, pred["odds_source"].to_numpy().astype(str), mid, "odds_source") if "odds_source" in pred.columns else pl.DataFrame(),
        "By odds band": _by_group(a, np.array([f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(*_odds_band(_bets(a, mid)["odds"]))]), mid, "odds_band"),
    }, [], [f"The EV > {mid:.2f} strategy from the table above, cut different ways. Small `bets` counts make `roi` unreliable."], {
        "By side": "Is the model's supposed edge concentrated on home wins, draws or away wins? Draws and away wins are longer prices, so a few winners move roi a lot.",
        "By season": "Does the edge hold up season after season, or was it one good year? A real edge is consistent in sign, not necessarily in size.",
        "By book source": "Bet365 prices carry a bigger margin than the Betfair exchange, so the same model can look worse in the exchange era simply because the bar is higher.",
        "By odds band": "Bets grouped by the price taken. Models often look profitable only at long odds, where a handful of winners dominates and the confidence interval is widest.",
    }))

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
        "The betting ledger can be swamped by luck. This section asks the underlying question directly: when the model and the bookmaker disagree, who is right?",
    ], {
        "Edge reliability": "Every (match, side) pair is put in a band by how far the model's probability sits from the bookmaker's. Within each band, `observed` is how often that side actually happened. If observed follows `model_p` in the outer bands, the model's disagreements carry information. If it follows `book_p`, they are noise.",
        "Fractional Kelly": "A staking rule that bets a fraction of the bankroll proportional to the expected value. It compounds, so a real edge grows the bankroll and a fake one shrinks it fast. `final_bankroll` below 1 means a loss.",
    }))
    return sections


def _odds_band(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 1000.0])
    idx = np.clip(np.digitize(odds, edges) - 1, 0, len(edges) - 2)
    return edges[idx], edges[idx + 1]

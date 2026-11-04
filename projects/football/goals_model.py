"""Dixon-Coles goals model, refitted walk-forward before every match date.

For each match date D the model is fitted on all matches strictly before D,
each weighted by exp(-days_ago / tau) with tau = half_life / ln 2, and then
predicts every match on D. Nothing from D or later is used.

Model (log-linear Poisson, Dixon & Coles 1997):
    home goals ~ Poisson(exp(c + home + att[h] - dfn[a]))
    away goals ~ Poisson(exp(c        + att[a] - dfn[h]))
with the Dixon-Coles low-score correction tau(x, y; rho) applied to the score
matrix. att and dfn are shrunk towards a prior (ridge, weight `ridge`):
0 for clubs in the league last season, and a fitted shared "newcomer" mean for
clubs that were not (promoted, or first season in the data). That gives
promoted sides a sensible starting point instead of league average.

Target: goals, or a blend `xg_weight` * xG + (1 - xg_weight) * goals where xG
exists (the Poisson log-likelihood is fine with non-integer targets). rho is
fitted on goals only, in a second step given the fitted means.

Outputs per match: H/D/A probabilities, expected goals each side, P(over 2.5),
P(both teams score), and the score-matrix mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass
class DCConfig:
    half_life_days: float = 365.0
    ridge: float = 2.0            # prior weight, in units of "effective matches"
    xg_weight: float = 0.0        # 0 = goals only; 0.5 = half xG
    window_days: float = 4 * 365  # ignore matches older than this (weights are tiny anyway)
    prefix: str = "dc"

    @classmethod
    def from_cfg(cls, d: dict[str, Any] | None, prefix: str) -> "DCConfig":
        d = dict(d or {})
        return cls(
            half_life_days=float(d.get("half_life_days", 365.0)),
            ridge=float(d.get("ridge", 2.0)),
            xg_weight=float(d.get("xg_weight", 0.0)),
            window_days=float(d.get("window_days", 4 * 365)),
            prefix=prefix,
        )


def _tau(x: np.ndarray, y: np.ndarray, mh: np.ndarray, ma: np.ndarray, rho: float) -> np.ndarray:
    t = np.ones_like(mh)
    t = np.where((x == 0) & (y == 0), 1.0 - mh * ma * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + mh * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + ma * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


class _Fit:
    """Weighted ridge Poisson fit with analytic gradient. Parameters:
    [c, home, newcomer_att, newcomer_dfn, att[0..T-1], dfn[0..T-1]]."""

    def __init__(self, n_teams: int):
        self.T = n_teams
        self.theta = np.zeros(4 + 2 * n_teams)

    def fit(self, hi, ai, yh, ya, w, newcomer, ridge):
        T = self.T
        wsum = w.sum()

        def f(theta):
            c, home, n_att, n_dfn = theta[:4]
            att, dfn = theta[4:4 + T], theta[4 + T:]
            eh = c + home + att[hi] - dfn[ai]
            ea = c + att[ai] - dfn[hi]
            mh, ma = np.exp(eh), np.exp(ea)
            # negative weighted log-likelihood (constant terms dropped), per unit weight
            nll = -(w * (yh * eh - mh + ya * ea - ma)).sum() / wsum
            ra = att - n_att * newcomer
            rd = dfn - n_dfn * newcomer
            pen = ridge / wsum * 0.5 * ((ra ** 2).sum() + (rd ** 2).sum())
            rh = w * (yh - mh) / wsum      # d(-nll)/d eh
            rA = w * (ya - ma) / wsum
            g = np.zeros_like(theta)
            g[0] = -(rh.sum() + rA.sum())
            g[1] = -rh.sum()
            g_att = -(np.bincount(hi, rh, T) + np.bincount(ai, rA, T))
            g_dfn = np.bincount(ai, rh, T) + np.bincount(hi, rA, T)
            g_att += ridge / wsum * ra
            g_dfn += ridge / wsum * rd
            g[2] = -ridge / wsum * (ra * newcomer).sum()
            g[3] = -ridge / wsum * (rd * newcomer).sum()
            g[4:4 + T], g[4 + T:] = g_att, g_dfn
            return nll + pen, g

        res = minimize(f, self.theta, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.theta = res.x
        return self

    def means(self, hi, ai):
        T = self.T
        c, home = self.theta[:2]
        att, dfn = self.theta[4:4 + T], self.theta[4 + T:]
        return np.exp(c + home + att[hi] - dfn[ai]), np.exp(c + att[ai] - dfn[hi])


def _fit_rho(xh, xa, mh, ma, w) -> float:
    low = (xh <= 1) & (xa <= 1)
    if not low.any():
        return 0.0
    x, y, h, a, ww = xh[low], xa[low], mh[low], ma[low], w[low]

    def nll(rho):
        t = _tau(x, y, h, a, rho)
        if (t <= 0).any():
            return 1e9
        return -(ww * np.log(t)).sum()

    lim = float(min(1.0 / max((h * a).max(), 1e-9), 1.0 / max(h.max(), a.max())))
    r = minimize_scalar(nll, bounds=(-0.99 * lim, 0.99 * lim), method="bounded")
    return float(r.x)


def score_probs(mh: np.ndarray, ma: np.ndarray, rho: float) -> dict[str, np.ndarray]:
    """Score matrix per match -> outcome probabilities."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], mh[:, None])        # n x G
    pa = poisson.pmf(g[None, :], ma[:, None])
    M = ph[:, :, None] * pa[:, None, :]              # n x G x G (home, away)
    M[:, 0, 0] *= 1.0 - mh * ma * rho
    M[:, 0, 1] *= 1.0 + mh * rho
    M[:, 1, 0] *= 1.0 + ma * rho
    M[:, 1, 1] *= 1.0 - rho
    M = np.clip(M, 0.0, None)
    M /= M.sum(axis=(1, 2), keepdims=True)
    tot = g[:, None] + g[None, :]
    home = np.tril(np.ones((len(g), len(g))), -1).astype(bool)   # home > away
    away = home.T
    return {
        "h": M[:, home].sum(axis=1), "d": M[:, np.eye(len(g), dtype=bool)].sum(axis=1), "a": M[:, away].sum(axis=1),
        "over25": M[:, tot > 2.5].sum(axis=1),
        "btts": M[:, 1:, 1:].sum(axis=(1, 2)),
    }


def forecasts(df: pl.DataFrame, cfg: DCConfig) -> pl.DataFrame:
    """Walk-forward forecasts for every row of df (needs match_id, date, season, home_team,
    away_team, home_goals, away_goals; home_xg/away_xg if xg_weight > 0)."""
    d = df.select(["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals",
                   *[c for c in ("home_xg", "away_xg") if c in df.columns]]).sort(["date", "match_id"])
    teams = sorted(set(d["home_team"].to_list()) | set(d["away_team"].to_list()))
    tix = {t: i for i, t in enumerate(teams)}
    hi = np.array([tix[t] for t in d["home_team"].to_list()])
    ai = np.array([tix[t] for t in d["away_team"].to_list()])
    gh = d["home_goals"].cast(pl.Float64).to_numpy()
    ga = d["away_goals"].cast(pl.Float64).to_numpy()
    yh, ya = gh.copy(), ga.copy()
    if cfg.xg_weight > 0 and "home_xg" in d.columns:
        xh = d["home_xg"].cast(pl.Float64).to_numpy()
        xa = d["away_xg"].cast(pl.Float64).to_numpy()
        ok = np.isfinite(xh) & np.isfinite(xa)
        yh = np.where(ok, cfg.xg_weight * xh + (1 - cfg.xg_weight) * gh, gh)
        ya = np.where(ok, cfg.xg_weight * xa + (1 - cfg.xg_weight) * ga, ga)
    days = (d["date"] - d["date"].min()).dt.total_days().to_numpy().astype(np.float64)
    seasons = d["season"].to_list()
    season_order = sorted(set(seasons))
    in_season: dict[str, set[int]] = {s: set() for s in season_order}
    for s, h, a in zip(seasons, hi, ai):
        in_season[s].add(h); in_season[s].add(a)

    n = d.height
    out = {k: np.full(n, np.nan) for k in ("h", "d", "a", "xg_home", "xg_away", "over25", "btts", "rho")}
    fitter = _Fit(len(teams))
    k_decay = np.log(2.0) / cfg.half_life_days
    uniq_days = np.unique(days)
    played = np.isfinite(gh) & np.isfinite(ga)
    for day in uniq_days:
        today = np.flatnonzero(days == day)
        past = np.flatnonzero((days < day) & (days >= day - cfg.window_days) & played)
        if past.size < 200:      # first ~half season: not enough to fit
            continue
        s = seasons[today[0]]
        si = season_order.index(s)
        prev = in_season[season_order[si - 1]] if si > 0 else set()
        newcomer = np.array([0.0 if t in prev else 1.0 for t in range(len(teams))])
        w = np.exp(-k_decay * (day - days[past]))
        fitter.fit(hi[past], ai[past], yh[past], ya[past], w, newcomer, cfg.ridge)
        mh_p, ma_p = fitter.means(hi[past], ai[past])
        rho = _fit_rho(gh[past], ga[past], mh_p, ma_p, w)
        mh, ma = fitter.means(hi[today], ai[today])
        pr = score_probs(mh, ma, rho)
        for k in ("h", "d", "a", "over25", "btts"):
            out[k][today] = pr[k]
        out["xg_home"][today], out["xg_away"][today], out["rho"][today] = mh, ma, rho
    p = cfg.prefix
    return pl.DataFrame({
        "match_id": d["match_id"],
        f"{p}_h": out["h"], f"{p}_d": out["d"], f"{p}_a": out["a"],
        f"{p}_xg_home": out["xg_home"], f"{p}_xg_away": out["xg_away"],
        f"{p}_over25": out["over25"], f"{p}_btts": out["btts"], f"{p}_rho": out["rho"],
    }).fill_nan(None)   # no fit yet (first ~200 matches) -> null, not NaN

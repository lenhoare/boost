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

Options (all off by default, which reproduces the original model):
  home_half_life_days  home advantage gets its own, faster time decay: it is set before each fit
                       to log(sum w hg / sum w ag) over past matches with that half-life, and held
                       fixed in the main fit. Lets it fall quickly when, say, crowds disappear.
  team_home            per-team home advantage on top, shrunk to 0 with weight ridge_home.
  refit_days           refit only when this many days have passed since the last fit (or a new
                       season starts); dates in between use the last fit, which saw only matches
                       before its own date, so nothing leaks. 7 is about 5x faster than daily.
  draw_inflation       multiply every draw score by (1 + delta) and renormalise; delta is fitted
                       with rho in the second step (diagonal-inflated Dixon-Coles).

Outputs per match: H/D/A probabilities, expected goals each side, P(over 2.5),
P(both teams score).
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
    home_half_life_days: float = 0.0   # > 0: home advantage with its own decay (see module doc)
    team_home: bool = False
    ridge_home: float = 20.0
    draw_inflation: bool = False
    refit_days: float = 0.0       # 0: refit before every match date; 7: at most weekly (see forecasts)
    prefix: str = "dc"

    @classmethod
    def from_cfg(cls, d: dict[str, Any] | None, prefix: str) -> "DCConfig":
        d = dict(d or {})
        return cls(
            half_life_days=float(d.get("half_life_days", 365.0)),
            ridge=float(d.get("ridge", 2.0)),
            xg_weight=float(d.get("xg_weight", 0.0)),
            window_days=float(d.get("window_days", 4 * 365)),
            home_half_life_days=float(d.get("home_half_life_days", 0.0)),
            team_home=bool(d.get("team_home", False)),
            ridge_home=float(d.get("ridge_home", 20.0)),
            draw_inflation=bool(d.get("draw_inflation", False)),
            refit_days=float(d.get("refit_days", 0.0)),
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
    [c, home, newcomer_att, newcomer_dfn, att[0..T-1], dfn[0..T-1], ha[0..T-1]].
    home_fixed: hold `home` at this value. team_home: fit ha (per-team home advantage)."""

    def __init__(self, n_teams: int):
        self.T = n_teams
        self.theta = np.zeros(4 + 3 * n_teams)

    def fit(self, hi, ai, yh, ya, w, newcomer, ridge, home_fixed=None, team_home=False, ridge_home=20.0):
        T = self.T
        wsum = w.sum()
        if home_fixed is not None:
            self.theta[1] = home_fixed
        if not team_home:
            self.theta[4 + 2 * T:] = 0.0

        def f(theta):
            c, home, n_att, n_dfn = theta[:4]
            att, dfn, ha = theta[4:4 + T], theta[4 + T:4 + 2 * T], theta[4 + 2 * T:]
            eh = c + home + ha[hi] + att[hi] - dfn[ai]
            ea = c + att[ai] - dfn[hi]
            mh, ma = np.exp(eh), np.exp(ea)
            # negative weighted log-likelihood (constant terms dropped), per unit weight
            nll = -(w * (yh * eh - mh + ya * ea - ma)).sum() / wsum
            ra = att - n_att * newcomer
            rd = dfn - n_dfn * newcomer
            pen = ridge / wsum * 0.5 * ((ra ** 2).sum() + (rd ** 2).sum()) + ridge_home / wsum * 0.5 * (ha ** 2).sum()
            rh = w * (yh - mh) / wsum      # d(-nll)/d eh
            rA = w * (ya - ma) / wsum
            g = np.zeros_like(theta)
            g[0] = -(rh.sum() + rA.sum())
            g[1] = 0.0 if home_fixed is not None else -rh.sum()
            g_att = -(np.bincount(hi, rh, T) + np.bincount(ai, rA, T))
            g_dfn = np.bincount(ai, rh, T) + np.bincount(hi, rA, T)
            g_att += ridge / wsum * ra
            g_dfn += ridge / wsum * rd
            g[2] = -ridge / wsum * (ra * newcomer).sum()
            g[3] = -ridge / wsum * (rd * newcomer).sum()
            g[4:4 + T], g[4 + T:4 + 2 * T] = g_att, g_dfn
            if team_home:
                g[4 + 2 * T:] = -np.bincount(hi, rh, T) + ridge_home / wsum * ha
            return nll + pen, g

        res = minimize(f, self.theta, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.theta = res.x
        return self

    def means(self, hi, ai):
        T = self.T
        c, home = self.theta[:2]
        att, dfn, ha = self.theta[4:4 + T], self.theta[4 + T:4 + 2 * T], self.theta[4 + 2 * T:]
        return np.exp(c + home + ha[hi] + att[hi] - dfn[ai]), np.exp(c + att[ai] - dfn[hi])


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


def _fit_rho_delta(xh, xa, mh, ma, w) -> tuple[float, float]:
    """Fit rho and the draw inflation delta jointly on the observed scores, given the means.

    P(x, y) = Pois(x; mh) Pois(y; ma) tau(x, y; rho) (1 + delta [x == y]) / (1 + delta D)
    where D = sum_x P_tau(x, x) is the draw probability before inflation (tau keeps the total at 1).
    """
    g = np.arange(MAX_GOALS + 1)
    ph, pa = poisson.pmf(g[None, :], mh[:, None]), poisson.pmf(g[None, :], ma[:, None])
    diag = (ph * pa).sum(axis=1)
    p00, p11 = ph[:, 0] * pa[:, 0], ph[:, 1] * pa[:, 1]
    draw = (xh == xa).astype(np.float64)
    low = (xh <= 1) & (xa <= 1)

    def nll(v):
        rho, delta = v
        if delta <= -0.99:
            return 1e9
        t = _tau(xh, xa, mh, ma, rho)
        if (t[low] <= 0).any():
            return 1e9
        D = diag - p00 * mh * ma * rho - p11 * rho
        return -(w * (np.log(t) + np.log1p(delta * draw) - np.log1p(delta * D))).sum()

    r = minimize(nll, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
    return float(r.x[0]), float(r.x[1])


def score_probs(mh: np.ndarray, ma: np.ndarray, rho: float, delta: float = 0.0) -> dict[str, np.ndarray]:
    """Score matrix per match -> outcome probabilities."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], mh[:, None])        # n x G
    pa = poisson.pmf(g[None, :], ma[:, None])
    M = ph[:, :, None] * pa[:, None, :]              # n x G x G (home, away)
    M[:, 0, 0] *= 1.0 - mh * ma * rho
    M[:, 0, 1] *= 1.0 + mh * rho
    M[:, 1, 0] *= 1.0 + ma * rho
    M[:, 1, 1] *= 1.0 - rho
    eye = np.eye(len(g), dtype=bool)
    if delta:
        M[:, eye] *= 1.0 + delta
    M = np.clip(M, 0.0, None)
    M /= M.sum(axis=(1, 2), keepdims=True)
    tot = g[:, None] + g[None, :]
    home = np.tril(np.ones((len(g), len(g))), -1).astype(bool)   # home > away
    away = home.T
    return {
        "h": M[:, home].sum(axis=1), "d": M[:, eye].sum(axis=1), "a": M[:, away].sum(axis=1),
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
    k_home = np.log(2.0) / cfg.home_half_life_days if cfg.home_half_life_days > 0 else None
    window = max(cfg.window_days, 2.5 * cfg.half_life_days)   # long half-lives need a longer look-back
    uniq_days = np.unique(days)
    played = np.isfinite(gh) & np.isfinite(ga)
    last_fit, last_season, rho, delta = None, None, 0.0, 0.0
    for day in uniq_days:
        today = np.flatnonzero(days == day)
        s = seasons[today[0]]
        if last_fit is not None and s == last_season and day < last_fit + cfg.refit_days:
            mh, ma = fitter.means(hi[today], ai[today])   # reuse the last fit
            pr = score_probs(mh, ma, rho, delta)
            for k in ("h", "d", "a", "over25", "btts"):
                out[k][today] = pr[k]
            out["xg_home"][today], out["xg_away"][today], out["rho"][today] = mh, ma, rho
            continue
        past = np.flatnonzero((days < day) & (days >= day - window) & played)
        if past.size < 200:      # first ~half season: not enough to fit
            continue
        last_fit, last_season = day, s
        si = season_order.index(s)
        prev = in_season[season_order[si - 1]] if si > 0 else set()
        newcomer = np.array([0.0 if t in prev else 1.0 for t in range(len(teams))])
        w = np.exp(-k_decay * (day - days[past]))
        home_fixed = None
        if k_home is not None:
            wh = np.exp(-k_home * (day - days[past]))
            home_fixed = float(np.log((wh * yh[past]).sum() / (wh * ya[past]).sum()))
        fitter.fit(hi[past], ai[past], yh[past], ya[past], w, newcomer, cfg.ridge,
                   home_fixed=home_fixed, team_home=cfg.team_home, ridge_home=cfg.ridge_home)
        mh_p, ma_p = fitter.means(hi[past], ai[past])
        if cfg.draw_inflation:
            rho, delta = _fit_rho_delta(gh[past], ga[past], mh_p, ma_p, w)
        else:
            rho, delta = _fit_rho(gh[past], ga[past], mh_p, ma_p, w), 0.0
        mh, ma = fitter.means(hi[today], ai[today])
        pr = score_probs(mh, ma, rho, delta)
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

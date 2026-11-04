"""Pre-match features for one row per match.

Rules: every feature is computed from matches strictly before this one, or
from information published before kick-off (book prices, referee, calendar).
Same-match statistics (xG, shots, possession...) enter only through lagged
team histories. Prefixes: h_ home side, a_ away side, d_ home minus away.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# per-team-match stats we build histories from: name -> (home column, away column)
STATS = {
    "gf": ("home_goals", "away_goals"),
    "ga": ("away_goals", "home_goals"),
    "xgf": ("home_xg", "away_xg"),
    "xga": ("away_xg", "home_xg"),
    "shf": ("home_shots", "away_shots"),
    "sha": ("away_shots", "home_shots"),
    "sotf": ("home_shots_on_target", "away_shots_on_target"),
    "sota": ("away_shots_on_target", "home_shots_on_target"),
    "poss": ("home_possession", "away_possession"),
    "cf": ("home_corners", "away_corners"),
    "ca": ("away_corners", "home_corners"),
}
FORM_WINDOWS = (5, 10)
EWM_STATS = ("pts", "gf", "ga", "xgf", "xga", "sotf", "sota")
VENUE_STATS = ("pts", "gf", "ga")


def _long(df: pl.DataFrame) -> pl.DataFrame:
    """Two rows per match: one per team, with stats from that team's point of view."""
    base = ["match_id", "date", "season"]
    home = df.select(base + [
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"), pl.lit(True).alias("is_home"),
        *[pl.col(h).cast(pl.Float64).alias(k) for k, (h, _) in STATS.items()],
    ])
    away = df.select(base + [
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"), pl.lit(False).alias("is_home"),
        *[pl.col(a).cast(pl.Float64).alias(k) for k, (_, a) in STATS.items()],
    ])
    long = pl.concat([home, away]).sort(["date", "match_id", "is_home"], descending=[False, False, True])
    return long.with_columns([
        pl.when(pl.col("gf") > pl.col("ga")).then(3.0).when(pl.col("gf") == pl.col("ga")).then(1.0).otherwise(0.0).alias("pts"),
        (pl.col("gf") - pl.col("ga")).alias("gd"),
    ])


def _team_history(long: pl.DataFrame, half_life: float) -> pl.DataFrame:
    """Lagged rolling / season-to-date / venue-specific aggregates per team."""
    stats = ["pts", "gd", *STATS.keys()]
    exprs: list[pl.Expr] = []
    for s in stats:
        lag = pl.col(s).shift(1).over("team")
        for w in FORM_WINDOWS:
            exprs.append(lag.rolling_mean(window_size=w, min_samples=1).over("team").alias(f"form{w}_{s}"))
    for s in EWM_STATS:
        exprs.append(pl.col(s).shift(1).ewm_mean(half_life=half_life, ignore_nulls=True).over("team").alias(f"ewm_{s}"))
    exprs += [
        pl.int_range(pl.len()).over(["team", "season"]).alias("std_played"),
        pl.col("pts").cum_sum().shift(1).fill_null(0.0).over(["team", "season"]).alias("std_pts"),
        pl.col("gf").cum_sum().shift(1).fill_null(0.0).over(["team", "season"]).alias("std_gf_sum"),
        pl.col("ga").cum_sum().shift(1).fill_null(0.0).over(["team", "season"]).alias("std_ga_sum"),
        pl.int_range(pl.len()).over("team").alias("career_played"),
        (pl.col("date") - pl.col("date").shift(1).over("team")).dt.total_days().alias("rest_days"),
    ]
    long = long.with_columns(exprs)
    long = long.with_columns([
        pl.when(pl.col("std_played") > 0).then(pl.col("std_pts") / pl.col("std_played")).alias("std_ppg"),
        pl.when(pl.col("std_played") > 0).then(pl.col("std_gf_sum") / pl.col("std_played")).alias("std_gf"),
        pl.when(pl.col("std_played") > 0).then(pl.col("std_ga_sum") / pl.col("std_played")).alias("std_ga"),
        pl.when(pl.col("std_played") > 0).then((pl.col("std_gf_sum") - pl.col("std_ga_sum")) / pl.col("std_played")).alias("std_gd"),
    ])
    # venue-specific form: a home side's last 5 at home, an away side's last 5 away
    venue = (
        long.select(["match_id", "team", "is_home", *VENUE_STATS])
        .sort(["team", "is_home", "match_id"])
        .with_columns([
            pl.col(s).shift(1).rolling_mean(window_size=5, min_samples=1).over(["team", "is_home"]).alias(f"venue5_{s}")
            for s in VENUE_STATS
        ])
        .with_columns(pl.int_range(pl.len()).over(["team", "is_home"]).alias("venue_played"))
        .select(["match_id", "team", *[f"venue5_{s}" for s in VENUE_STATS], "venue_played"])
    )
    long = long.join(venue, on=["match_id", "team"], how="left")
    # previous season summary
    season_order = {s: i for i, s in enumerate(sorted(long["season"].unique().to_list()))}
    prev = (
        long.group_by(["team", "season"]).agg([pl.col("pts").mean().alias("prev_ppg"), pl.col("gd").mean().alias("prev_gd"), pl.len().alias("prev_played")])
        .with_columns(pl.col("season").replace_strict(season_order, return_dtype=pl.Int64).alias("season_idx"))
    )
    long = long.with_columns(pl.col("season").replace_strict(season_order, return_dtype=pl.Int64).alias("season_idx"))
    long = long.join(
        prev.select(["team", (pl.col("season_idx") + 1).alias("season_idx"), "prev_ppg", "prev_gd", "prev_played"]),
        on=["team", "season_idx"], how="left",
    ).with_columns(pl.col("prev_played").is_null().cast(pl.Int8).alias("newcomer"))
    return long


def _sequential(df: pl.DataFrame, k: float, home_adv: float, k_goals: float = 0.05) -> pl.DataFrame:
    """One pass in date order: Elo, standings, congestion, h2h, last-match surprise.

    Also six venue-specific ratings per team (added 2026-09-18, see newfeaturenotes.md):
      res_home / res_away  Elo from home results only / away results only (start 1500, step k)
      att_home / def_home  goals scored / conceded at home, log-rate units (start 0, step k_goals)
      att_away / def_away  goals scored / conceded away
    Expected home goals = running league average x exp(att_home[h] - def_away[a]); the
    attack rating rises and the defence rating falls by k_goals x (actual - expected).
    """
    n = df.height
    rating: dict[str, float] = defaultdict(lambda: 1500.0)
    res_home: dict[str, float] = defaultdict(lambda: 1500.0)
    res_away: dict[str, float] = defaultdict(lambda: 1500.0)
    att_home: dict[str, float] = defaultdict(float)
    def_home: dict[str, float] = defaultdict(float)
    att_away: dict[str, float] = defaultdict(float)
    def_away: dict[str, float] = defaultdict(float)
    goals_seen = 0
    sum_hg, sum_ag = 0.0, 0.0
    recent: dict[str, deque] = defaultdict(deque)
    last_resid: dict[str, float] = {}
    last_gd: dict[str, float] = {}
    table_pts: dict[str, float] = {}
    table_gd: dict[str, float] = {}
    table_played: dict[str, int] = {}
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=3))
    current_season = None
    out = {name: np.full(n, np.nan) for name in (
        "h_elo", "a_elo", "elo_exp_home", "h_rank", "a_rank", "h_cong14", "a_cong14",
        "h2h_gd3", "h2h_n", "h_last_resid", "a_last_resid", "h_last_gd", "a_last_gd", "n_teams_season",
        "h_elo_res_home", "a_elo_res_away", "elo_venue_exp_home",
        "h_elo_att_home", "h_elo_def_home", "a_elo_att_away", "a_elo_def_away",
        "elo_xg_home", "elo_xg_away",
    )}
    rows = df.select(["season", "date", "home_team", "away_team", "home_goals", "away_goals"]).iter_rows()
    for i, (season, date, h, a, hg, ag) in enumerate(rows):
        if season != current_season:
            current_season = season
            table_pts, table_gd, table_played = {}, {}, {}
        for t in (h, a):
            table_pts.setdefault(t, 0.0); table_gd.setdefault(t, 0.0); table_played.setdefault(t, 0)
        rh, ra = rating[h], rating[a]
        exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + home_adv)) / 400.0))
        out["h_elo"][i], out["a_elo"][i], out["elo_exp_home"][i] = rh, ra, exp_h
        # venue-specific ratings, read before this match
        vh, va = res_home[h], res_away[a]
        venue_exp = 1.0 / (1.0 + 10 ** ((va - vh) / 400.0))
        out["h_elo_res_home"][i], out["a_elo_res_away"][i], out["elo_venue_exp_home"][i] = vh, va, venue_exp
        avg_hg = sum_hg / goals_seen if goals_seen else 1.5
        avg_ag = sum_ag / goals_seen if goals_seen else 1.2
        xg_h = avg_hg * np.exp(att_home[h] - def_away[a])
        xg_a = avg_ag * np.exp(att_away[a] - def_home[h])
        out["h_elo_att_home"][i], out["h_elo_def_home"][i] = att_home[h], def_home[h]
        out["a_elo_att_away"][i], out["a_elo_def_away"][i] = att_away[a], def_away[a]
        out["elo_xg_home"][i], out["elo_xg_away"][i] = xg_h, xg_a
        standings = sorted(table_pts, key=lambda t: (table_pts[t], table_gd[t]), reverse=True)
        rank = {t: r + 1 for r, t in enumerate(standings)}
        out["h_rank"][i], out["a_rank"][i] = rank[h], rank[a]
        out["n_teams_season"][i] = len(standings)
        for t, key in ((h, "h_cong14"), (a, "a_cong14")):
            q = recent[t]
            while q and (date - q[0]).days > 14:
                q.popleft()
            out[key][i] = len(q)
        key = (h, a)
        hist = h2h[key]
        if hist:
            out["h2h_gd3"][i] = float(np.mean(hist))
        out["h2h_n"][i] = len(hist)
        out["h_last_resid"][i] = last_resid.get(h, np.nan)
        out["a_last_resid"][i] = last_resid.get(a, np.nan)
        out["h_last_gd"][i] = last_gd.get(h, np.nan)
        out["a_last_gd"][i] = last_gd.get(a, np.nan)

        # update with this match's result
        if hg is None or ag is None:
            continue
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        margin = 1.0 + np.log1p(abs(hg - ag))  # goal-margin multiplier, mild
        rating[h] = rh + k * margin * (actual - exp_h)
        rating[a] = ra + k * margin * ((1.0 - actual) - (1.0 - exp_h))
        last_resid[h], last_resid[a] = actual - exp_h, (1.0 - actual) - (1.0 - exp_h)
        res_home[h] = vh + k * (actual - venue_exp)
        res_away[a] = va + k * ((1.0 - actual) - (1.0 - venue_exp))
        att_home[h] += k_goals * (hg - xg_h)
        def_away[a] -= k_goals * (hg - xg_h)
        att_away[a] += k_goals * (ag - xg_a)
        def_home[h] -= k_goals * (ag - xg_a)
        goals_seen += 1
        sum_hg += hg
        sum_ag += ag
        last_gd[h], last_gd[a] = float(hg - ag), float(ag - hg)
        ph, pa = (3, 0) if hg > ag else ((0, 3) if hg < ag else (1, 1))
        table_pts[h] += ph; table_pts[a] += pa
        table_gd[h] += hg - ag; table_gd[a] += ag - hg
        table_played[h] += 1; table_played[a] += 1
        recent[h].append(date); recent[a].append(date)
        h2h[(h, a)].append(hg - ag)
        h2h[(a, h)].append(ag - hg)
    return pl.DataFrame({"match_id": df["match_id"], **{k_: pl.Series(v) for k_, v in out.items()}})


def _goals_model():
    """goals_model.py sits next to this file; load it by path (projects are not packages)."""
    spec = importlib.util.spec_from_file_location("football_goals_model", Path(__file__).with_name("goals_model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def _dixon_coles(df: pl.DataFrame, fcfg: dict[str, Any]) -> pl.DataFrame:
    """Walk-forward Dixon-Coles forecasts: dc_* from goals, dcx_* from a goals/xG blend."""
    gm = _goals_model()
    gcfg = dict(fcfg.get("goals_model") or {})
    out = None
    for prefix, xg_w in (("dc", 0.0), ("dcx", float(gcfg.get("xg_weight", 0.5)))):
        c = gm.DCConfig.from_cfg({**gcfg, "xg_weight": xg_w}, prefix)
        fc = gm.forecasts(df, c).drop(f"{prefix}_rho")
        out = fc if out is None else out.join(fc, on="match_id")
    return out


LINE_REFS = ("avg", "b365", "bestuk", "max", "bfe")   # pre-match prices compared with Pinnacle's


def _line(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Line-movement inputs and targets. Inputs use only prices collected at the same moment as
    Pinnacle's pre-match price (football-data's Friday/Tuesday snapshot) plus our own forecasts.

    ln_ps_{h,d,a}        log of Pinnacle's pre-match de-vigged probability (the level)
    ln_<ref>_{h,d,a}     log(ref probability) - log(Pinnacle probability), refs: market average,
                         Bet365, best UK price, best overall price, Betfair exchange, Dixon-Coles
    ln_sd_{h,d,a}        spread (sd of log probability) across individual pre-match books
    ln_nbooks            how many individual books priced the match
    ln_*_overround       margins: Pinnacle, Bet365, market average
    ln_kickoff_hour      kick-off hour where known (2019-20 on)
    Targets (never features): mv_ps_{h,d,a} = log(Pinnacle close prob) - log(Pinnacle pre prob);
    mv_zero = 0, the "no move" forecast.
    """
    def probs(stem: str) -> list[pl.Expr]:
        cols = [pl.col(f"{stem}_{k}") for k in "hda"]
        tot = sum(1.0 / c for c in cols)
        return [(1.0 / c) / tot for c in cols]

    exprs: list[pl.Expr] = []
    names: list[str] = []
    ps = probs("odds_ps")
    for k, e in zip("hda", ps):
        exprs.append(e.log().alias(f"ln_ps_{k}")); names.append(f"ln_ps_{k}")
    for ref in LINE_REFS:
        for k, e, base in zip("hda", probs(f"odds_{ref}"), ps):
            exprs.append((e.log() - base.log()).alias(f"ln_{ref}_{k}")); names.append(f"ln_{ref}_{k}")
    for k, base in zip("hda", ps):
        exprs.append((pl.col(f"dc_{k}").log() - base.log()).alias(f"ln_dc_{k}")); names.append(f"ln_dc_{k}")
    for ref in ("ps", "b365", "avg"):
        exprs.append(pl.col(f"book_{ref}_overround").alias(f"ln_{ref}_overround")); names.append(f"ln_{ref}_overround")
    exprs.append(pl.col("kickoff").str.slice(0, 2).cast(pl.Int32, strict=False).alias("ln_kickoff_hour")); names.append("ln_kickoff_hour")
    for k, e in zip("hda", probs("odds_psc")):
        exprs.append((e.log() - ps["hda".index(k)].log()).alias(f"mv_ps_{k}"))
    exprs.append(pl.lit(0.0).alias("mv_zero"))
    out = df.with_columns(exprs)

    # dispersion across individual pre-match books (closing books end in "c"; exclude the reference and exchanges)
    pre = sorted({c[3:-2] for c in df.columns if c.startswith("bk_") and c.endswith("_h")})
    pre = [b for b in pre if not b.endswith("c") and b not in ("ps", "bfe")]
    L = np.full((df.height, len(pre), 3), np.nan)
    for j, b in enumerate(pre):
        O = df.select([f"bk_{b}_{k}" for k in "hda"]).to_numpy().astype(np.float64)
        q = 1.0 / O
        L[:, j, :] = np.log(q / q.sum(axis=1, keepdims=True))
    n = np.isfinite(L[:, :, 0]).sum(axis=1)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(L, axis=1)
    sd[n < 2] = np.nan
    out = out.with_columns([pl.Series(f"ln_sd_{k}", sd[:, i]).fill_nan(None) for i, k in enumerate("hda")]
                           + [pl.Series("ln_nbooks", n.astype(np.int64))])
    names += ["ln_sd_h", "ln_sd_d", "ln_sd_a", "ln_nbooks"]
    return out, names


def build(df: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    fcfg = dict(cfg.get("features") or {})
    k = float(fcfg.get("elo_k", 20)); home_adv = float(fcfg.get("elo_home_adv", 60)); hl = float(fcfg.get("ewm_half_life", 8))
    k_goals = float(fcfg.get("elo_goals_k", 0.05))

    long = _team_history(_long(df), hl)
    side_cols = [c for c in long.columns if c.startswith(("form", "ewm_", "std_", "venue", "prev_", "career_")) or c in ("rest_days", "newcomer")]
    side_cols = [c for c in side_cols if c not in ("std_pts", "std_gf_sum", "std_ga_sum", "prev_played")]
    home_side = long.filter(pl.col("is_home")).select(["match_id", *[pl.col(c).alias(f"h_{c}") for c in side_cols]])
    away_side = long.filter(~pl.col("is_home")).select(["match_id", *[pl.col(c).alias(f"a_{c}") for c in side_cols]])
    seq = _sequential(df, k, home_adv, k_goals)

    dc = _dixon_coles(df, fcfg)
    out = df.join(home_side, on="match_id", how="left").join(away_side, on="match_id", how="left").join(seq, on="match_id", how="left").join(dc, on="match_id", how="left")
    season_start = out.group_by("season").agg(pl.col("date").min().alias("season_first"))
    out = out.join(season_start, on="season", how="left").with_columns([
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"),
        (pl.col("date") - pl.col("season_first")).dt.total_days().alias("days_into_season"),
        (pl.col("home_xg_pre") - pl.col("away_xg_pre")).alias("d_xg_pre"),
        (pl.col("home_ppg_pre") - pl.col("away_ppg_pre")).alias("d_ppg_pre"),
        (pl.col("h_elo") - pl.col("a_elo")).alias("d_elo"),
        (pl.col("h_rank") - pl.col("a_rank")).alias("d_rank"),
        (pl.col("h_cong14") - pl.col("a_cong14")).alias("d_cong14"),
        (pl.col("h_rest_days") - pl.col("a_rest_days")).alias("d_rest_days"),
        (pl.col("h_prev_ppg") - pl.col("a_prev_ppg")).alias("d_prev_ppg"),
        (pl.col("h_std_ppg") - pl.col("a_std_ppg")).alias("d_std_ppg"),
        (pl.col("h_std_gd") - pl.col("a_std_gd")).alias("d_std_gd"),
        (pl.col("h_form5_pts") - pl.col("a_form5_pts")).alias("d_form5_pts"),
        (pl.col("h_form10_pts") - pl.col("a_form10_pts")).alias("d_form10_pts"),
        (pl.col("h_ewm_pts") - pl.col("a_ewm_pts")).alias("d_ewm_pts"),
        (pl.col("h_ewm_xgf") - pl.col("a_ewm_xga")).alias("d_attack_h"),   # home attack vs away defence
        (pl.col("a_ewm_xgf") - pl.col("h_ewm_xga")).alias("d_attack_a"),   # away attack vs home defence
        (pl.col("h_ewm_gf") - pl.col("a_ewm_gf")).alias("d_ewm_gf"),
        (pl.col("h_ewm_ga") - pl.col("a_ewm_ga")).alias("d_ewm_ga"),
        (pl.col("h_venue5_pts") - pl.col("a_venue5_pts")).alias("d_venue5_pts"),
        (pl.col("h_elo_res_home") - pl.col("a_elo_res_away")).alias("d_elo_res_venue"),
        (pl.col("h_elo_att_home") - pl.col("a_elo_def_away")).alias("d_elo_attack_h"),
        (pl.col("a_elo_att_away") - pl.col("h_elo_def_home")).alias("d_elo_attack_a"),
        (pl.col("elo_xg_home") - pl.col("elo_xg_away")).alias("d_elo_xg"),
        (pl.col("dc_xg_home") - pl.col("dc_xg_away")).alias("d_dc_xg"),
        (pl.col("dcx_xg_home") - pl.col("dcx_xg_away")).alias("d_dcx_xg"),
        (pl.col("h_rank") <= 4).cast(pl.Int8).alias("h_top4"),
        (pl.col("a_rank") <= 4).cast(pl.Int8).alias("a_top4"),
        pl.when(pl.col("odds_source").str.contains("betfair")).then(1).otherwise(0).cast(pl.Int8).alias("book_is_exchange"),
    ]).drop("season_first")
    out, line_names = _line(out)

    features = (
        [c for c in out.columns if c.startswith(("h_", "a_", "d_"))]
        + ["home_xg_pre", "away_xg_pre", "home_ppg_pre", "away_ppg_pre", "elo_exp_home", "h2h_gd3", "h2h_n",
           "elo_venue_exp_home", "elo_xg_home", "elo_xg_away",
           *[f"{p}_{k}" for p in ("dc", "dcx") for k in ("h", "d", "a", "xg_home", "xg_away", "over25", "btts")],
           "n_teams_season", "month", "weekday", "days_into_season",
           "home_team", "away_team", "referee",
           "book_h", "book_d", "book_a", "book_overround", "book_is_exchange",
           *line_names]
    )
    features = list(dict.fromkeys(features))
    return out, features


# ---------------------------------------------------------------------------
# Plain-English descriptions. Core calls describe(name) for tooltips and the
# "Features used" section. Names are built from a few parts, so most are
# derived; the rest are listed.

_STAT = {
    "pts": "points taken (3 win, 1 draw, 0 loss)",
    "gd": "goal difference",
    "gf": "goals scored",
    "ga": "goals conceded",
    "xgf": "expected goals (xG) created",
    "xga": "expected goals (xG) conceded",
    "shf": "shots taken",
    "sha": "shots faced",
    "sotf": "shots on target taken",
    "sota": "shots on target faced",
    "poss": "possession share (%)",
    "cf": "corners won",
    "ca": "corners conceded",
}
_SIDE = {"h": "Home side's", "a": "Away side's"}
_FIXED = {
    "elo": "Elo rating before this match. Everyone starts at 1500; it rises with wins and falls with losses, moving more for surprising results and bigger margins.",
    "rank": "league position before this match (1 = top). Early in a season this is noisy.",
    "cong14": "matches played in the previous 14 days. High values mean a congested fixture list.",
    "rest_days": "days since their previous match.",
    "last_resid": "how far their previous result beat or fell short of what Elo expected, from +1 (won when expected to lose) to -1 (lost when expected to win). Captures a one-off surprise that may carry over.",
    "last_gd": "goal difference in their previous match.",
    "top4": "1 if they were in the top four before this match, else 0.",
    "std_played": "matches played so far this season.",
    "std_ppg": "points per game so far this season.",
    "std_gf": "goals scored per game so far this season.",
    "std_ga": "goals conceded per game so far this season.",
    "std_gd": "goal difference per game so far this season.",
    "prev_ppg": "points per game over the whole of last season. Missing if they were not in the league last season.",
    "prev_gd": "goal difference per game over the whole of last season. Missing if not in the league last season.",
    "newcomer": "1 if there is no previous season for them in the data (promoted, or the first season we have), else 0.",
    "career_played": "matches for them in the dataset before this one.",
    "venue_played": "matches before this one at this venue type (home games for the home side, away games for the away side).",
}
_WHOLE = {
    "elo_exp_home": "Elo's own probability of a home win, from the two ratings plus a 60-point home advantage.",
    "h_elo_res_home": "Home side's Elo built only from its home results. Starts at 1500 and updates like the overall Elo, but away games never touch it.",
    "a_elo_res_away": "Away side's Elo built only from its away results. Starts at 1500; home games never touch it.",
    "elo_venue_exp_home": "Probability of a home win implied by the two venue-specific result Elos (home side's home rating against away side's away rating). No separate home-advantage term is needed because the split already carries it.",
    "h_elo_att_home": "Home side's goals-scored rating at home, in log units: 0 is league average, +0.1 means about 10% more goals than average. Rises when they score more than expected at home.",
    "h_elo_def_home": "Home side's goals-conceded rating at home, log units. Higher is a better defence: it rises when they concede fewer than expected at home.",
    "a_elo_att_away": "Away side's goals-scored rating away from home, log units; 0 is league average.",
    "a_elo_def_away": "Away side's goals-conceded rating away from home, log units; higher is a better defence.",
    "elo_xg_home": "Expected home goals from the goals ratings: running league average for home goals times exp(home attack rating minus away defence rating).",
    "elo_xg_away": "Expected away goals from the goals ratings: running league average for away goals times exp(away attack rating minus home defence rating).",
    "d_elo_res_venue": "Home side's home-results Elo minus away side's away-results Elo.",
    "d_elo_attack_h": "Home attack against away defence in the goals ratings: home side's home attack rating minus away side's away defence rating. Positive means more home goals than average are expected.",
    "d_elo_attack_a": "Away attack against home defence in the goals ratings: away side's away attack rating minus home side's home defence rating.",
    "d_elo_xg": "Expected goals from the ratings, home minus away.",
    **{f"{p}_{k}": f"{label} {text}" for p, label in (
        ("dc", "Dixon-Coles goals model (fitted on goals):"),
        ("dcx", "Dixon-Coles goals model (fitted on a 50/50 blend of goals and xG where xG exists):"),
    ) for k, text in (
        ("h", "probability of a home win. Refitted before every match date on all earlier matches, recent ones weighted more (half-life 500 days)."),
        ("d", "probability of a draw, including the Dixon-Coles low-score correction."),
        ("a", "probability of an away win."),
        ("xg_home", "expected home goals: exp(constant + home advantage + home attack - away defence)."),
        ("xg_away", "expected away goals: exp(constant + away attack - home defence)."),
        ("over25", "probability of three or more goals."),
        ("btts", "probability that both teams score."),
    )},
    **{f"ln_ps_{k}": f"Log of Pinnacle's pre-match {o} probability (margin removed). The starting point of the line model." for k, o in (("h", "home-win"), ("d", "draw"), ("a", "away-win"))},
    **{f"ln_{r}_{k}": f"{rl} {o} probability minus Pinnacle's, both pre-match, in log units. Positive: {rl.split(' (')[0].lower()} rates the {o.split('-')[0]} side higher than Pinnacle does."
       for r, rl in (("avg", "Market-average"), ("b365", "Bet365"), ("bestuk", "Best UK price (implied)"), ("max", "Best overall price (implied)"), ("bfe", "Betfair exchange"), ("dc", "Dixon-Coles"))
       for k, o in (("h", "home-win"), ("d", "draw"), ("a", "away-win"))},
    **{f"ln_sd_{k}": f"Disagreement between bookmakers on the {o} before kick-off: standard deviation of their log probabilities." for k, o in (("h", "home win"), ("d", "draw"), ("a", "away win"))},
    "ln_nbooks": "Number of individual bookmakers that priced the match in the pre-match snapshot.",
    "ln_ps_overround": "Pinnacle's pre-match margin (sum of implied probabilities).",
    "ln_b365_overround": "Bet365's pre-match margin.",
    "ln_avg_overround": "Market-average pre-match margin.",
    "ln_kickoff_hour": "Kick-off hour (UK time), 2019-20 on.",
    "d_dc_xg": "Dixon-Coles (goals) expected goals, home minus away.",
    "d_dcx_xg": "Dixon-Coles (goals and xG blend) expected goals, home minus away.",
    "h2h_gd3": "Average goal difference from the home side's point of view over the last three meetings between these two teams, any venue. Missing if they have not met in the data.",
    "h2h_n": "How many previous meetings (up to three) feed h2h_gd3.",
    "n_teams_season": "Number of teams seen so far this season; 20 once every team has played.",
    "month": "Calendar month of the match (1 to 12).",
    "weekday": "Day of the week (1 = Monday to 7 = Sunday).",
    "days_into_season": "Days since the first match of this season.",
    "home_xg_pre": "Pre-match expected-goals forecast for the home side, supplied with the data by its source's own model. Missing before 2016-17.",
    "away_xg_pre": "Pre-match expected-goals forecast for the away side, supplied with the data. Missing before 2016-17.",
    "home_ppg_pre": "Pre-match points-per-game figure for the home side, supplied with the data. Its exact definition is unknown; weaker than our own season-to-date figure.",
    "away_ppg_pre": "Pre-match points-per-game figure for the away side, supplied with the data. Definition unknown.",
    "d_xg_pre": "Supplied pre-match xG forecast, home minus away.",
    "d_ppg_pre": "Supplied pre-match points per game, home minus away.",
    "d_attack_h": "Home attack against away defence: home side's weighted xG created minus away side's weighted xG conceded.",
    "d_attack_a": "Away attack against home defence: away side's weighted xG created minus home side's weighted xG conceded.",
    "home_team": "Home team name, as a category.",
    "away_team": "Away team name, as a category.",
    "referee": "Match referee, as a category. Missing on about a third of rows.",
    "book_h": "Bookmaker's closing probability of a home win with the margin removed. Excluded from history-only models.",
    "book_d": "Bookmaker's closing probability of a draw, margin removed.",
    "book_a": "Bookmaker's closing probability of an away win, margin removed.",
    "book_overround": "Sum of the bookmaker's three implied probabilities before removing the margin; 1.05 means a 5% margin.",
    "book_is_exchange": "1 if the prices come from the Betfair exchange (2024-25 on), 0 for Bet365.",
}


def _body(body: str) -> str | None:
    if body in _FIXED:
        return _FIXED[body]
    if body.startswith("form"):
        win, _, stat = body[4:].partition("_")
        if stat in _STAT and win.isdigit():
            return f"average {_STAT[stat]} over their last {win} matches (any venue), before this one."
    if body.startswith("ewm_"):
        stat = body[4:]
        if stat in _STAT:
            return f"weighted average {_STAT[stat]} over all their past matches, recent ones counting more (half-life 8 matches)."
    if body.startswith("venue5_"):
        stat = body[7:]
        if stat in _STAT:
            return f"average {_STAT[stat]} over their last 5 matches at this venue type (home games for the home side, away games for the away side)."
    return None


def describe(name: str) -> str | None:
    """Plain-English meaning of one feature column, or None if unknown."""
    if name in _WHOLE:
        return _WHOLE[name]
    side, _, body = name.partition("_")
    if side in _SIDE:
        text = _body(body)
        return f"{_SIDE[side]} {text}" if text else None
    if side == "d":
        text = _body(body)
        if text:
            first = text.split(". ")[0].rstrip(".")
            return f"Home minus away: {first[0].lower() + first[1:]}, for the home side minus the same for the away side."
    return None

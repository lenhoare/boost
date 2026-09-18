"""Pre-match features for one row per match.

Rules: every feature is computed from matches strictly before this one, or
from information published before kick-off (book prices, referee, calendar).
Same-match statistics (xG, shots, possession...) enter only through lagged
team histories. Prefixes: h_ home side, a_ away side, d_ home minus away.
"""

from __future__ import annotations

from collections import defaultdict, deque
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


def _sequential(df: pl.DataFrame, k: float, home_adv: float) -> pl.DataFrame:
    """One pass in date order: Elo, standings, congestion, h2h, last-match surprise."""
    n = df.height
    rating: dict[str, float] = defaultdict(lambda: 1500.0)
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
        last_gd[h], last_gd[a] = float(hg - ag), float(ag - hg)
        ph, pa = (3, 0) if hg > ag else ((0, 3) if hg < ag else (1, 1))
        table_pts[h] += ph; table_pts[a] += pa
        table_gd[h] += hg - ag; table_gd[a] += ag - hg
        table_played[h] += 1; table_played[a] += 1
        recent[h].append(date); recent[a].append(date)
        h2h[(h, a)].append(hg - ag)
        h2h[(a, h)].append(ag - hg)
    return pl.DataFrame({"match_id": df["match_id"], **{k_: pl.Series(v) for k_, v in out.items()}})


def build(df: pl.DataFrame, cfg: dict[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    fcfg = dict(cfg.get("features") or {})
    k = float(fcfg.get("elo_k", 20)); home_adv = float(fcfg.get("elo_home_adv", 60)); hl = float(fcfg.get("ewm_half_life", 8))

    long = _team_history(_long(df), hl)
    side_cols = [c for c in long.columns if c.startswith(("form", "ewm_", "std_", "venue", "prev_", "career_")) or c in ("rest_days", "newcomer")]
    side_cols = [c for c in side_cols if c not in ("std_pts", "std_gf_sum", "std_ga_sum", "prev_played")]
    home_side = long.filter(pl.col("is_home")).select(["match_id", *[pl.col(c).alias(f"h_{c}") for c in side_cols]])
    away_side = long.filter(~pl.col("is_home")).select(["match_id", *[pl.col(c).alias(f"a_{c}") for c in side_cols]])
    seq = _sequential(df, k, home_adv)

    out = df.join(home_side, on="match_id", how="left").join(away_side, on="match_id", how="left").join(seq, on="match_id", how="left")
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
        (pl.col("h_rank") <= 4).cast(pl.Int8).alias("h_top4"),
        (pl.col("a_rank") <= 4).cast(pl.Int8).alias("a_top4"),
        pl.when(pl.col("odds_source").str.contains("betfair")).then(1).otherwise(0).cast(pl.Int8).alias("book_is_exchange"),
    ]).drop("season_first")

    features = (
        [c for c in out.columns if c.startswith(("h_", "a_", "d_"))]
        + ["home_xg_pre", "away_xg_pre", "home_ppg_pre", "away_ppg_pre", "elo_exp_home", "h2h_gd3", "h2h_n",
           "n_teams_season", "month", "weekday", "days_into_season",
           "home_team", "away_team", "referee",
           "book_h", "book_d", "book_a", "book_overround", "book_is_exchange"]
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

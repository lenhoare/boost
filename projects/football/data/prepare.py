"""combined.csv -> prepared.parquet. Cleaning, standard names, book probabilities, leak audit.

Everything in the prepared frame is either known before kick-off, or is a
match outcome/statistic that the feature recipe will only ever use lagged.
The audit printed at the end records what was found and what was dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Post-match stats that the source encodes as -1 when not collected.
NEG1_STATS = [
    "home_corners", "away_corners", "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_shots_off_target", "away_shots_off_target", "home_fouls", "away_fouls",
    "home_possession", "away_possession",
]
# Zero here means "not available", never a real value.
ZERO_IS_NULL = ["home_xg", "away_xg", "home_prematch_xg", "away_prematch_xg"]


def _parse_minutes(raw: str | None) -> list[int]:
    """'12,45'6,90'3' -> [12, 45, 90]. Stoppage time folds into 45 / 90."""
    if raw is None:
        return []
    out: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        head = tok.split("'", 1)[0].replace("’", "")
        try:
            out.append(int(head))
        except ValueError:
            continue
    return out


def _timing_columns(df: pl.DataFrame) -> pl.DataFrame:
    hm = [_parse_minutes(x) for x in df["home_goal_timings"].to_list()]
    am = [_parse_minutes(x) for x in df["away_goal_timings"].to_list()]
    home_ht = [sum(1 for m in ms if m <= 45) for ms in hm]
    away_ht = [sum(1 for m in ms if m <= 45) for ms in am]
    first_min: list[int | None] = []
    first_side: list[str | None] = []
    for h, a in zip(hm, am):
        cands = [(m, "H") for m in h] + [(m, "A") for m in a]
        if cands:
            m, s = min(cands)
            first_min.append(m)
            first_side.append(s)
        else:
            first_min.append(None)
            first_side.append(None)
    return df.with_columns([
        pl.Series("home_goals_ht", home_ht, dtype=pl.Int64),
        pl.Series("away_goals_ht", away_ht, dtype=pl.Int64),
        pl.Series("first_goal_minute", first_min, dtype=pl.Int64),
        pl.Series("first_goal_side", first_side, dtype=pl.Utf8),
    ])


def _audit_ppg(df: pl.DataFrame) -> list[str]:
    """Is home_ppg the season-final figure (a leak) and is *_prematch_ppg genuinely season-to-date?"""
    notes: list[str] = []
    const = (
        df.group_by(["season", "home_team"]).agg(pl.col("home_ppg").n_unique().alias("nu"))
        .filter(pl.col("nu") > 1).height
    )
    total = df.group_by(["season", "home_team"]).len().height
    notes.append(f"home_ppg is constant within team-season for {total - const}/{total} team-seasons -> season-final figure, a leak, dropped")

    # Recompute season-to-date ppg ourselves and compare with the supplied prematch figure.
    long = pl.concat([
        df.select(["match_id", "season", "date", pl.col("home_team").alias("team"),
                   pl.when(pl.col("home_goals") > pl.col("away_goals")).then(3).when(pl.col("home_goals") == pl.col("away_goals")).then(1).otherwise(0).alias("pts"),
                   pl.col("home_prematch_ppg").alias("given")]),
        df.select(["match_id", "season", "date", pl.col("away_team").alias("team"),
                   pl.when(pl.col("away_goals") > pl.col("home_goals")).then(3).when(pl.col("home_goals") == pl.col("away_goals")).then(1).otherwise(0).alias("pts"),
                   pl.col("away_prematch_ppg").alias("given")]),
    ]).sort(["date", "match_id"])
    long = long.with_columns([
        pl.col("pts").cum_sum().shift(1).fill_null(0).over(["team", "season"]).alias("cum_pts"),
        pl.int_range(pl.len()).over(["team", "season"]).alias("played"),
    ]).with_columns((pl.col("cum_pts") / pl.col("played")).alias("ours"))
    cmp = long.filter(pl.col("played") > 0).filter(pl.col("given").is_not_null())
    diff = (cmp["ours"] - cmp["given"]).abs()
    notes.append(f"supplied prematch_ppg vs our season-to-date ppg: median abs diff {diff.median():.3f}, within 0.05 on {(diff < 0.05).mean():.1%} of rows")
    return notes


def prepare(project_dir: Path, cfg: dict[str, Any]) -> pl.DataFrame:
    src = project_dir / str((cfg.get("data") or {}).get("source") or "data/combined.csv")
    df = pl.read_csv(src, infer_schema_length=20000, null_values=["N/A", ""])
    n_raw = df.height

    # dates arrive as dd/mm/yy and dd/mm/yyyy
    df = df.with_columns(
        pl.when(pl.col("date").str.len_chars() == 10)
        .then(pl.col("date").str.strptime(pl.Date, "%d/%m/%Y", strict=False))
        .otherwise(pl.col("date").str.strptime(pl.Date, "%d/%m/%y", strict=False))
        .alias("date")
    )
    if df["date"].null_count():
        raise ValueError(f"{df['date'].null_count()} unparseable dates")

    df = df.rename({"fthg": "home_goals", "ftag": "away_goals"})
    df = df.with_columns([
        *[pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c) for c in NEG1_STATS if c in df.columns],
        *[pl.when(pl.col(c) <= 0).then(None).otherwise(pl.col(c)).alias(c) for c in ZERO_IS_NULL if c in df.columns],
        pl.col("attendance").cast(pl.Int64, strict=False),
    ])
    df = df.sort(["date", "home_team", "away_team"]).with_row_index("match_id")

    # book: de-vigged 1X2 probabilities. odds > 1.01 guards dummy prices.
    inv = [(1.0 / pl.col(c)).alias(f"inv_{c}") for c in ("odds_home", "odds_draw", "odds_away")]
    df = df.with_columns(inv)
    tot = pl.col("inv_odds_home") + pl.col("inv_odds_draw") + pl.col("inv_odds_away")
    ok = (pl.col("odds_home") > 1.01) & (pl.col("odds_draw") > 1.01) & (pl.col("odds_away") > 1.01)
    df = df.with_columns([
        pl.when(ok).then(pl.col("inv_odds_home") / tot).alias("book_h"),
        pl.when(ok).then(pl.col("inv_odds_draw") / tot).alias("book_d"),
        pl.when(ok).then(pl.col("inv_odds_away") / tot).alias("book_a"),
        pl.when(ok).then(tot).alias("book_overround"),
    ]).drop(["inv_odds_home", "inv_odds_draw", "inv_odds_away"])

    df = _timing_columns(df)
    df = df.with_columns([
        (pl.col("home_goals") - pl.col("away_goals")).alias("goal_diff"),
        (pl.col("home_goals") + pl.col("away_goals")).alias("total_goals"),
        pl.col("season").str.slice(0, 4).cast(pl.Int64).alias("season_start"),
    ])

    notes = _audit_ppg(df)
    df = df.drop(["home_ppg", "away_ppg"])
    df = df.rename({"home_prematch_ppg": "home_ppg_pre", "away_prematch_ppg": "away_ppg_pre",
                    "home_prematch_xg": "home_xg_pre", "away_prematch_xg": "away_xg_pre"})

    # audit printout
    print(f"prepare: {n_raw} rows, {df['season'].n_unique()} seasons, {df['date'].min()} .. {df['date'].max()}")
    print(f"  results: {dict(zip(*np.unique(df['ftr'].to_numpy(), return_counts=True)))}")
    for n in notes:
        print(f"  {n}")
    cov = df.group_by("season").agg([
        pl.len().alias("n"),
        pl.col("odds_source").first(),
        pl.col("book_overround").mean().round(4).alias("overround"),
        pl.col("home_xg").is_not_null().mean().round(2).alias("xg"),
        pl.col("home_xg_pre").is_not_null().mean().round(2).alias("xg_pre"),
        pl.col("home_shots").is_not_null().mean().round(2).alias("shots"),
        pl.col("home_possession").is_not_null().mean().round(2).alias("poss"),
        pl.col("home_corners").is_not_null().mean().round(2).alias("corners"),
    ]).sort("season")
    with pl.Config(tbl_rows=40, tbl_cols=20):
        print(cov)
    return df

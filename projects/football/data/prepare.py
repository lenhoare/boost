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

# football-data.co.uk price sets: our name -> candidate column stems, first present wins.
# A stem plus H/D/A gives the 1X2 columns. "Bb" stems are the pre-2019-20 Betbrain names.
# Pre-match prices are collected Friday afternoon (weekend) / Tuesday (midweek); "C" = closing.
FD_1X2 = {
    "b365": ["B365"], "b365c": ["B365C"],
    "ps": ["PS"], "psc": ["PSC"],
    "avg": ["Avg", "BbAv"], "avgc": ["AvgC"],
    "max": ["Max", "BbMx"], "maxc": ["MaxC"],
    "bfe": ["BFE"], "bfec": ["BFEC"],
}
# over/under 2.5 goals: our name -> candidate prefixes; prefix + ">2.5" / "<2.5"
FD_OU = {
    "b365": ["B365"], "b365c": ["B365C"], "ps": ["P"], "psc": ["PC"],
    "avg": ["Avg", "BbAv"], "avgc": ["AvgC"], "max": ["Max", "BbMx"], "maxc": ["MaxC"],
    "bfe": ["BFE"], "bfec": ["BFEC"],
}
# Asian handicap: line columns, then price sets; prefix + "AHH" / "AHA"
FD_AH_LINE = {"ah_line": ["AHh", "BbAHh"], "ah_line_c": ["AHCh"]}
FD_AH = {
    "b365": ["B365"], "b365c": ["B365C"], "ps": ["P"], "psc": ["PC"],
    "avg": ["Avg", "BbAv"], "avgc": ["AvgC"], "max": ["Max", "BbMx"], "maxc": ["MaxC"],
    "bfe": ["BFE"], "bfec": ["BFEC"],
}
# Mainstream UK-licensed bookmakers a bettor could open accounts with. "Best UK price" is the
# max over whichever of these the season's file has. Excluded on purpose: PS (Pinnacle, the
# sharp reference), BFE (exchange, commission), 1XB (offshore), Max/Avg (aggregates).
UK_BOOKS = ["B365", "BW", "WH", "VC", "LB", "IW", "BV", "BMGM", "BFD", "BF", "SKB", "PP", "GB", "SJ", "BS", "SB"]

# football-data team name -> ours, where lowercase-and-underscore is not enough
FD_TEAMS = {"Man United": "man_utd", "Nott'm Forest": "nottingham_forest", "Sheffield United": "sheffield_utd"}


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


def _num(d: pl.DataFrame, col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64, strict=False) if col in d.columns else pl.lit(None, dtype=pl.Float64)


def _first(d: pl.DataFrame, cands: list[str], suffix: str) -> str | None:
    for c in cands:
        if c + suffix in d.columns and d[c + suffix].is_not_null().any():
            return c + suffix
    return None


def _read_football_data(folder: Path) -> pl.DataFrame:
    """Every E0_*.csv -> one frame keyed on (date, home_team, away_team) with canonical odds columns.

    Raw prices: odds_<set>_{h,d,a}, ou_<set>_{over,under}, ah_<set>_{h,a}, plus ah_line / ah_line_c
    (home handicap, e.g. -0.75). Kick-off time as kickoff where supplied (2019-20 on).
    """
    frames = []
    for f in sorted(folder.glob("E0_*.csv")):
        d = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True)
        d = d.filter(pl.col("HomeTeam").is_not_null() & pl.col("Date").is_not_null())
        cols: list[pl.Expr] = [
            pl.when(pl.col("Date").str.len_chars() == 10)
            .then(pl.col("Date").str.strptime(pl.Date, "%d/%m/%Y", strict=False))
            .otherwise(pl.col("Date").str.strptime(pl.Date, "%d/%m/%y", strict=False)).alias("date"),
            pl.col("HomeTeam").alias("fd_home"), pl.col("AwayTeam").alias("fd_away"),
            (pl.col("Time") if "Time" in d.columns else pl.lit(None, dtype=pl.Utf8)).alias("kickoff"),
        ]
        for name, stems in FD_1X2.items():
            stem = next((s for s in stems if _first(d, [s], "H")), None)
            for side in "HDA":
                cols.append((_num(d, stem + side) if stem else pl.lit(None, dtype=pl.Float64)).alias(f"odds_{name}_{side.lower()}"))
        for name, pre in FD_OU.items():
            c_over, c_under = _first(d, pre, ">2.5"), _first(d, pre, "<2.5")
            cols.append((_num(d, c_over) if c_over else pl.lit(None, dtype=pl.Float64)).alias(f"ou_{name}_over"))
            cols.append((_num(d, c_under) if c_under else pl.lit(None, dtype=pl.Float64)).alias(f"ou_{name}_under"))
        for name, cands in FD_AH_LINE.items():
            c = next((x for x in cands if x in d.columns), None)
            cols.append((_num(d, c) if c else pl.lit(None, dtype=pl.Float64)).alias(name))
        for name, pre in FD_AH.items():
            ch, ca = _first(d, pre, "AHH"), _first(d, pre, "AHA")
            cols.append((_num(d, ch) if ch else pl.lit(None, dtype=pl.Float64)).alias(f"ah_{name}_h"))
            cols.append((_num(d, ca) if ca else pl.lit(None, dtype=pl.Float64)).alias(f"ah_{name}_a"))
        # every individual bookmaker's 1X2 triple, pre-match and closing: bk_<code>_{h,d,a}
        books = sorted({c[:-1] for c in d.columns if len(c) > 1 and c.endswith("H") and c[:-1] + "D" in d.columns and c[:-1] + "A" in d.columns
                        and not c.startswith(("Bb", "Max", "Avg")) and c[:-1].isalnum()})
        for b in books:
            for side in "HDA":
                cols.append(_num(d, b + side).alias(f"bk_{b.lower()}_{side.lower()}"))
        frames.append(d.select(cols))
    fd = pl.concat(frames, how="diagonal_relaxed")   # bookmaker columns differ by season
    fix = lambda c: pl.col(c).replace_strict(FD_TEAMS, default=pl.col(c).str.to_lowercase().str.replace_all(" ", "_"))  # noqa: E731
    fd = fd.with_columns(fix("fd_home").alias("home_team"), fix("fd_away").alias("away_team")).drop(["fd_home", "fd_away"])
    # dummy / missing prices: anything at or below 1.01 is not a real price
    price = [c for c in fd.columns if c.startswith(("odds_", "ou_", "ah_", "bk_")) and not c.startswith("ah_line")]
    fd = fd.with_columns([pl.when(pl.col(c) > 1.01).then(pl.col(c)).alias(c) for c in price])
    # best price over mainstream UK books, pre-match and closing, and how many books fed it
    for suffix, tag in (("", "bestuk"), ("c", "bestukc")):
        names = [b.lower() + suffix for b in UK_BOOKS if f"bk_{b.lower() + suffix}_h" in fd.columns]
        for side in "hda":
            fd = fd.with_columns(pl.max_horizontal([pl.col(f"bk_{n}_{side}") for n in names]).alias(f"odds_{tag}_{side}"))
        fd = fd.with_columns(pl.sum_horizontal([pl.col(f"bk_{n}_h").is_not_null().cast(pl.Int8) for n in names]).alias(f"n_{tag}"))
    if fd.select(["date", "home_team", "away_team"]).is_duplicated().any():
        raise ValueError("duplicate (date, home, away) in football-data files")
    return fd


def _devig(df: pl.DataFrame) -> pl.DataFrame:
    """Proportional de-vig for every 1X2 set -> book_<set>_{h,d,a} and book_<set>_overround.

    Max (best price across books) is not one book's market, so it is left as raw odds only.
    Over/under: p_<set>_over (de-vigged). Asian handicap prices stay raw (pushes and quarter lines).
    """
    exprs: list[pl.Expr] = []
    for name in FD_1X2:
        if name.startswith("max"):
            continue
        h, d_, a = (pl.col(f"odds_{name}_{s}") for s in "hda")
        tot = 1.0 / h + 1.0 / d_ + 1.0 / a
        exprs += [((1.0 / h) / tot).alias(f"book_{name}_h"), ((1.0 / d_) / tot).alias(f"book_{name}_d"),
                  ((1.0 / a) / tot).alias(f"book_{name}_a"), tot.alias(f"book_{name}_overround")]
    for name in FD_OU:
        if name.startswith("max"):
            continue
        o, u = pl.col(f"ou_{name}_over"), pl.col(f"ou_{name}_under")
        exprs.append(((1.0 / o) / (1.0 / o + 1.0 / u)).alias(f"p_{name}_over25"))
    return df.with_columns(exprs)


def _merge_football_data(df: pl.DataFrame, folder: Path) -> tuple[pl.DataFrame, list[str]]:
    notes: list[str] = []
    if not folder.is_dir():
        return df, [f"no football-data folder at {folder}; extra price sets skipped"]
    fd = _read_football_data(folder)
    out = df.join(fd, on=["date", "home_team", "away_team"], how="left")
    miss = out.filter(pl.col("odds_b365_h").is_null() & pl.col("odds_avg_h").is_null() & pl.col("odds_bfe_h").is_null())
    if miss.height:
        raise ValueError(f"{miss.height} combined.csv rows found no football-data match, e.g. {miss.select(['date', 'home_team', 'away_team']).head(5).to_dicts()}")
    notes.append(f"football-data merged: {out.height}/{df.height} rows matched on (date, home, away)")
    # consistency: our single price set must equal the football-data column its odds_source names
    src_col = {"bet365": "odds_b365_h", "bet365_close": "odds_b365c_h", "betfair_exchange_close": "odds_bfec_h", "betfair_exchange": "odds_bfe_h"}
    for src, col in src_col.items():
        sub = out.filter(pl.col("odds_source") == src)
        if sub.height:
            same = ((sub["odds_home"] - sub[col]).abs() < 1e-6).mean()
            notes.append(f"  odds_source={src}: odds_home equals {col} on {same:.1%} of {sub.height} rows")
    return _devig(out), notes


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

    fd_dir = project_dir / str((cfg.get("data") or {}).get("football_data") or "data/football-data")
    df, fd_notes = _merge_football_data(df, fd_dir)

    df = _timing_columns(df)
    df = df.with_columns([
        (pl.col("home_goals") - pl.col("away_goals")).alias("goal_diff"),
        (pl.col("home_goals") + pl.col("away_goals")).alias("total_goals"),
        pl.col("season").str.slice(0, 4).cast(pl.Int64).alias("season_start"),
    ])

    notes = fd_notes + _audit_ppg(df)
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

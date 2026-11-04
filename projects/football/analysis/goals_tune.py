"""Tune the Dixon-Coles goals model honestly: a fixed grid, nested walk-forward selection.

Every grid variant produces walk-forward forecasts (each match predicted from earlier matches
only). For each test season s, the variant with the lowest logloss over seasons before s is
chosen and used for s, so the reported score includes the cost of choosing. The grid was fixed
before any results were seen (see measurement.md, step 4).

Also reported: one-at-a-time effects (change one ingredient of the current model), the best
fixed variant in hindsight (optimistic, for reference only), and a dispersion check
(variance / mean of goals given the model's means; Poisson is 1).

Usage (repo root):
    .venv/bin/python projects/football/analysis/goals_tune.py [--workers 4] [--refresh]
Forecasts are cached in projects/football/data/goals_grid.parquet (gitignored).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects/football"))
from boost.metrics import logloss_rows, paired  # noqa: E402

PREPARED = ROOT / "projects/football/data/prepared.parquet"
CACHE = ROOT / "projects/football/data/goals_grid.parquet"
FIRST_TEST = "2014-15"
FIRST_SCORED = "2011-12"   # forecasts exist from part-way through 2010-11; selection uses 2011-12 on

GRID = {
    "half_life_days": [250, 500, 1000],
    "ridge": [2.0, 5.0, 10.0],
    "home": ["global", "dynamic", "dynamic_team"],
    "xg_weight": [0.0, 0.25, 0.5],
    "draw_inflation": [False, True],
}
CURRENT = {"half_life_days": 500, "ridge": 5.0, "home": "global", "xg_weight": 0.0, "draw_inflation": False}


def key(v: dict) -> str:
    return f"hl{v['half_life_days']}_r{v['ridge']:g}_{v['home']}_xg{v['xg_weight']:g}_{'di' if v['draw_inflation'] else 'rho'}"


def variants() -> list[dict]:
    names = list(GRID)
    return [dict(zip(names, combo)) for combo in itertools.product(*GRID.values())]


def _config(v: dict):
    import goals_model as gm
    return gm.DCConfig(
        half_life_days=float(v["half_life_days"]), ridge=float(v["ridge"]), xg_weight=float(v["xg_weight"]),
        home_half_life_days=120.0 if v["home"] != "global" else 0.0, team_home=v["home"] == "dynamic_team",
        draw_inflation=bool(v["draw_inflation"]), prefix="g",
    )


def _run(v: dict) -> tuple[str, pl.DataFrame]:
    import goals_model as gm
    df = pl.read_parquet(PREPARED)
    fc = gm.forecasts(df, _config(v))
    k = key(v)
    return k, fc.select(["match_id", *[pl.col(f"g_{s}").alias(f"{k}__{s}") for s in ("h", "d", "a", "xg_home", "xg_away")]])


def build_cache(workers: int, refresh: bool) -> pl.DataFrame:
    have = pl.read_parquet(CACHE) if CACHE.exists() and not refresh else None
    todo = [v for v in variants() if have is None or f"{key(v)}__h" not in have.columns]
    if todo:
        print(f"fitting {len(todo)} variants on {workers} workers ...", flush=True)
        frames = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (k, fc) in enumerate(ex.map(_run, todo), 1):
                frames.append(fc)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
        out = have if have is not None else frames[0].select("match_id")
        for fc in frames:
            out = out.join(fc, on="match_id", how="left")
        out.write_parquet(CACHE)
        have = out
    return have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    grid = build_cache(args.workers, args.refresh)
    base = pl.read_parquet(PREPARED).select(["match_id", "season", "ftr", "home_goals", "away_goals"])
    df = base.join(grid, on="match_id").filter(pl.col("season") >= FIRST_SCORED).sort("match_id")
    y = np.array(["HDA".index(v) for v in df["ftr"].to_list()])
    seasons = df["season"].to_numpy()
    vs = variants()
    keys = [key(v) for v in vs]
    P = {k: df.select([f"{k}__{s}" for s in "hda"]).to_numpy() for k in keys}
    L = {k: logloss_rows(P[k], y) for k in keys}
    ok = np.all([np.isfinite(L[k]) for k in keys], axis=0)
    test_seasons = sorted(s for s in set(seasons) if s >= FIRST_TEST)
    test = ok & (seasons >= FIRST_TEST)
    cur = key(CURRENT)

    # nested selection
    nested = np.full(len(y), np.nan)
    chosen = []
    for s in test_seasons:
        prior = ok & (seasons < s)
        best = min(keys, key=lambda k: L[k][prior].mean())
        chosen.append((s, best))
        m = ok & (seasons == s)
        nested[m] = L[best][m]
    print(f"\nTest seasons {test_seasons[0]} .. {test_seasons[-1]}, {int(test.sum())} matches. Current model: {cur}\n")
    print(f"current model          {L[cur][test].mean():.4f}")
    d = paired(nested[test], L[cur][test])
    print(f"nested selection       {nested[test].mean():.4f}   vs current {d['diff']:+.4f} ± {d['diff_se']:.4f} (z {d['diff_z']:+.1f})   <- the honest number")
    hind = min(keys, key=lambda k: L[k][test].mean())
    d = paired(L[hind][test], L[cur][test])
    print(f"best in hindsight      {L[hind][test].mean():.4f}   vs current {d['diff']:+.4f} ± {d['diff_se']:.4f}   ({hind}; optimistic, chosen on the test seasons)")

    print("\nChosen variant per test season (from earlier seasons only):")
    for s, k in chosen:
        m = ok & (seasons == s)
        print(f"  {s}  {k:36s}  {L[k][m].mean():.4f}  current {L[cur][m].mean():.4f}  diff {L[k][m].mean() - L[cur][m].mean():+.4f}")

    print("\nOne ingredient changed from the current model (test seasons):")
    for name, options in GRID.items():
        for opt in options:
            if opt == CURRENT[name]:
                continue
            k = key({**CURRENT, name: opt})
            d = paired(L[k][test], L[cur][test])
            print(f"  {name:15s} = {str(opt):13s} {L[k][test].mean():.4f}  diff {d['diff']:+.4f} ± {d['diff_se']:.4f} (z {d['diff_z']:+.1f})")

    print("\nBest variant with each ingredient fixed (test seasons; hindsight within the group):")
    for name, options in GRID.items():
        row = []
        for opt in options:
            ks = [key(v) for v in vs if v[name] == opt]
            row.append(f"{opt}: {min(L[k][test].mean() for k in ks):.4f}")
        print(f"  {name:15s} " + "   ".join(row))

    # dispersion: is Poisson too narrow? variance / mean of goals given the current model's means
    xh, xa = df["home_goals"].to_numpy().astype(float), df["away_goals"].to_numpy().astype(float)
    mh, ma = df[f"{cur}__xg_home"].to_numpy(), df[f"{cur}__xg_away"].to_numpy()
    m = test & np.isfinite(mh)
    disp = ((xh[m] - mh[m]) ** 2 / mh[m]).mean(), ((xa[m] - ma[m]) ** 2 / ma[m]).mean()
    print(f"\nDispersion (Pearson, Poisson = 1): home goals {disp[0]:.3f}, away goals {disp[1]:.3f}")


if __name__ == "__main__":
    main()

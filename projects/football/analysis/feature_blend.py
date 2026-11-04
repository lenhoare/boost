"""Few-weight correction of the Dixon-Coles forecast with a handful of features (step 5).

    z_k = a · log dc_k  [+ g · log book_k]  + c_k + Σ_j b_jk x_j      k ∈ {H, A}; draw is the reference (z_D = a log dc_D [+ g log book_D])
    p   = softmax(z)

Features (home minus away, standardised on the training rows, missing -> 0 with no separate flag):
  xg_pre   FootyStats pre-match xG difference (2016-17 on)
  elo      Elo rating difference
  rest     rest-days difference, clipped to [-7, 7]
  promoted newcomer flag difference (in the league last season or not)

Fitted walk-forward: before each season on all earlier seasons, small L2 penalty on the b's.
Each feature is compared against "recalibrated DC" (same model, no features), so a gain is the
feature's, not the recalibration's. Then the same with a bookmaker price as a second offset: does
anything here add to the price?

Usage (repo root):  .venv/bin/python projects/football/analysis/feature_blend.py [--from 2014-15] [--l2 1]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from boost.metrics import logloss_rows, paired  # noqa: E402

FEATURES = ROOT / "projects/football/data/features.parquet"
L2 = 1.0   # penalty on feature weights, per 1000 training rows' worth of loss


def feature_matrix(f: pl.DataFrame) -> dict[str, np.ndarray]:
    col = lambda c: f[c].cast(pl.Float64).to_numpy()  # noqa: E731
    return {
        "xg_pre": col("home_xg_pre") - col("away_xg_pre"),
        "elo": col("h_elo") - col("a_elo"),
        "rest": np.clip(col("h_rest_days") - col("a_rest_days"), -7, 7),
        "promoted": col("h_newcomer") - col("a_newcomer"),
    }


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _design(offsets, X):
    """theta = [offset weights..., c_H, c_A, b_H (J), b_A (J)]"""
    n_o, J = len(offsets), X.shape[1]

    def probs(theta):
        z = sum(theta[i] * offsets[i] for i in range(n_o))
        c_h, c_a = theta[n_o], theta[n_o + 1]
        b_h, b_a = theta[n_o + 2:n_o + 2 + J], theta[n_o + 2 + J:]
        z = z.copy()
        z[:, 0] += c_h + X @ b_h
        z[:, 2] += c_a + X @ b_a
        return _softmax(z)
    return probs, n_o, J


def fit_predict(offsets_tr, X_tr, y_tr, offsets_te, X_te, init):
    probs_tr, n_o, J = _design(offsets_tr, X_tr)
    n = len(y_tr)

    def loss(theta):
        p = probs_tr(theta)
        ll = -np.log(np.clip(p[np.arange(n), y_tr], 1e-15, None)).mean()
        return ll + L2 / 1000.0 * 0.5 * (theta[n_o + 2:] ** 2).sum()

    theta = minimize(loss, init(n_o, J), method="L-BFGS-B").x
    probs_te, _, _ = _design(offsets_te, X_te)
    return probs_te(theta), theta


def walk_forward(f, offsets, X, y, seasons, start, ok):
    pred = np.full((len(y), 3), np.nan)
    thetas = []
    init = lambda n_o, J: np.r_[np.ones(n_o) / n_o, 0.0, 0.0, np.zeros(2 * J)]  # noqa: E731
    for s in sorted(set(seasons)):
        if s < start:
            continue
        tr, te = ok & (seasons < s), ok & (seasons == s)
        if tr.sum() < 500 or not te.any():
            continue
        Xs = X.copy()
        if Xs.shape[1]:
            mu = np.nanmean(Xs[tr], 0)
            sd = np.nanstd(Xs[tr], 0)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            Xs = np.nan_to_num((Xs - mu) / sd, nan=0.0)
        p, th = fit_predict([o[tr] for o in offsets], Xs[tr], y[tr], [o[te] for o in offsets], Xs[te], init)
        pred[te] = p
        thetas.append((s, th))
    return pred, thetas


def logs(f, stem):
    P = f.select([f"{stem}_{k}" for k in "hda"]).to_numpy().astype(np.float64)
    P = P / P.sum(1, keepdims=True)
    return np.log(P)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2014-15")
    ap.add_argument("--l2", type=float, default=L2, help="penalty on feature weights (10 and 100 tried too; see measurement.md)")
    args = ap.parse_args()
    globals()["L2"] = args.l2
    warnings.simplefilter("ignore", RuntimeWarning)   # nanmean over seasons with no xg_pre yet
    f = pl.read_parquet(FEATURES).sort(["date", "match_id"])
    y = np.array(["HDA".index(v) for v in f["ftr"].to_list()])
    seasons = f["season"].to_numpy()
    feats = feature_matrix(f)
    names = list(feats)
    LD = logs(f, "dc")
    ok_dc = np.isfinite(LD).all(1)
    test = (seasons >= args.start)

    def run(offs, cols, ok):
        X = np.column_stack([feats[c] for c in cols]) if cols else np.zeros((len(y), 0))
        return walk_forward(f, offs, X, y, seasons, args.start, ok)

    def score(pred, ref, mask):
        m = mask & np.isfinite(pred).all(1) & np.isfinite(ref).all(1)
        a, b = logloss_rows(pred[m], y[m]), logloss_rows(ref[m], y[m])
        return a.mean(), paired(a, b), int(m.sum())

    print(f"features: {', '.join(names)}; walk-forward from {args.start}\n")
    print("A. Dixon-Coles plus features (no prices)")
    dc_raw = np.exp(LD)
    base, _ = run([LD], [], ok_dc)
    ll, d, n = score(base, dc_raw, test & ok_dc)
    print(f"   {'raw DC':22s} {score(dc_raw, dc_raw, test & ok_dc)[0]:.4f}   n {n}")
    print(f"   {'recalibrated DC':22s} {ll:.4f}   vs raw {d['diff']:+.4f} ± {d['diff_se']:.4f}")
    for cols in [[c] for c in names] + [names]:
        pred, th = run([LD], cols, ok_dc)
        ll, d, _ = score(pred, base, test & ok_dc)
        label = "+ " + " + ".join(cols) if len(cols) == 1 else "+ all four"
        print(f"   {label:22s} {ll:.4f}   vs recalibrated DC {d['diff']:+.4f} ± {d['diff_se']:.4f} (z {d['diff_z']:+.1f})")
    # weights of the full model in the last season, for interpretation (standardised units)
    _, th = run([LD], names, ok_dc)
    s_last, t = th[-1]
    J = len(names)
    print(f"   weights fitted for {s_last} (per sd; home / away logit vs draw): "
          + ", ".join(f"{c} {t[3 + j]:+.3f}/{t[3 + J + j]:+.3f}" for j, c in enumerate(names)) + f"; DC weight {t[0]:.3f}")

    print("\nB. Does it add to a price? price + DC (+ features), vs the price alone (recalibrated the same way)")
    for name, stem in (("b365_pre", "book_b365"), ("pinnacle_pre", "book_ps"), ("avg_pre", "book_avg")):
        LB = logs(f, stem)
        ok = ok_dc & np.isfinite(LB).all(1)
        p_book, _ = run([LB], [], ok)
        p_dc, _ = run([LB, LD], [], ok)
        p_all, _ = run([LB, LD], names, ok)
        start_ok = ok & test & np.isfinite(p_book).all(1)
        l0, _, n = score(p_book, p_book, start_ok)
        l1, d1, _ = score(p_dc, p_book, start_ok)
        l2, d2, _ = score(p_all, p_book, start_ok)
        print(f"   {name:14s} n {n:5d}  price {l0:.4f}   +DC {l1:.4f} ({d1['diff']:+.4f} ± {d1['diff_se']:.4f})"
              f"   +DC+features {l2:.4f} ({d2['diff']:+.4f} ± {d2['diff_se']:.4f}, z {d2['diff_z']:+.1f})")


if __name__ == "__main__":
    main()

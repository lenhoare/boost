"""Prepared data and features. Both are cached as parquet inside the project."""

from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import polars as pl

from boost.project import Project


def prepare(project: Project) -> Path:
    mod = project.module("data")
    if not hasattr(mod, "prepare"):
        raise TypeError("data module must define prepare(project_dir, cfg) -> pl.DataFrame")
    df = mod.prepare(project.dir, project.cfg)
    if not isinstance(df, pl.DataFrame):
        raise TypeError("data.prepare must return a polars DataFrame")
    df = _ordered(project, df)
    out = project.prepared_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def _ordered(project: Project, df: pl.DataFrame) -> pl.DataFrame:
    order = project.order_col
    if order and order in df.columns:
        return df.sort(order, maintain_order=True)
    return df


def load_prepared(project: Project) -> pl.DataFrame:
    path = project.prepared_path
    if not path.exists():
        prepare(project)
    return _ordered(project, pl.read_parquet(path))


def _features_meta_path(project: Project) -> Path:
    return project.features_path.with_suffix(".json")


def _cache_fresh(project: Project) -> bool:
    cache = project.features_path
    meta = _features_meta_path(project)
    if not cache.exists() or not meta.exists():
        return False
    stamp = cache.stat().st_mtime
    sources = [project.prepared_path]
    rel = (project.cfg.get("features") or {}).get("module")
    if rel:
        sources.append(project.path(str(rel)))
    return all(s.exists() and s.stat().st_mtime <= stamp for s in sources)


def build_features(project: Project, force: bool = False) -> tuple[pl.DataFrame, list[str]]:
    """Prepared frame + project features. Returns (frame, feature column names)."""
    if not force and _cache_fresh(project):
        frame = pl.read_parquet(project.features_path)
        names = json.loads(_features_meta_path(project).read_text())["features"]
        return frame, list(names)

    base = load_prepared(project)
    if not project.has_module("features"):
        names = [c for c in base.columns]
        frame = base
    else:
        mod = project.module("features")
        if not hasattr(mod, "build"):
            raise TypeError("features module must define build(df, cfg) -> (df, feature_names)")
        out = mod.build(base, project.cfg)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError("features.build must return (DataFrame, list[str])")
        frame, names = out
        names = [str(n) for n in names]
        missing = [n for n in names if n not in frame.columns]
        if missing:
            raise KeyError(f"features.build listed columns not in frame: {missing}")
        if frame.height != base.height:
            raise ValueError("features.build must not add or drop rows")
    frame = _ordered(project, frame)
    project.features_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(project.features_path)
    _features_meta_path(project).write_text(json.dumps({"features": names}, indent=2))
    return frame, names


def select_features(all_names: list[str], spec: dict[str, Any] | None) -> list[str]:
    """include/exclude glob lists from a model config, applied in order."""
    spec = dict(spec or {})
    include = spec.get("include") or ["*"]
    exclude = spec.get("exclude") or []
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    chosen = [n for n in all_names if any(fnmatch(n, pat) for pat in include)]
    chosen = [n for n in chosen if not any(fnmatch(n, pat) for pat in exclude)]
    if not chosen:
        raise ValueError(f"feature selection left nothing. include={include} exclude={exclude}")
    return chosen

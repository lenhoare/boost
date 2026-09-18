"""Runs are folders. Old runs are never touched."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from boost.project import Project


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def new_run_dir(project: Project, model_name: str) -> Path:
    root = project.runs_dir
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}-{model_name}"
    run_dir = root / base
    k = 1
    while run_dir.exists():
        k += 1
        run_dir = root / f"{base}-{k}"
    run_dir.mkdir()
    return run_dir


def write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    meta = dict(meta)
    meta.setdefault("run_id", run_dir.name)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    (run_dir / "meta.json").write_text(json.dumps(_json_safe(meta), indent=2))


def load_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "meta.json").read_text())


def list_runs(project: Project) -> list[Path]:
    root = project.runs_dir
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())


def resolve_run(project: Project, run: str | None) -> Path:
    runs = list_runs(project)
    if not run or run == "latest":
        if not runs:
            raise FileNotFoundError(f"no runs in {project.runs_dir}")
        return runs[-1]
    p = Path(run)
    if p.is_dir() and (p / "meta.json").exists():
        return p
    for r in runs:
        if r.name == run or r.name.startswith(run):
            return r
    raise FileNotFoundError(f"run not found: {run}")


def load_predictions(run_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(run_dir / "predictions.parquet")

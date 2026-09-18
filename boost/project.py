from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Project:
    """projects/<name>/project.yaml plus the python files it points at."""

    def __init__(self, path: Path) -> None:
        self.dir = path.resolve()
        cfg_path = self.dir / "project.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no project.yaml in {self.dir}")
        self.cfg = load_yaml(cfg_path)
        self.name = str(self.cfg.get("name") or self.dir.name)
        self._modules: dict[str, ModuleType] = {}

    @classmethod
    def find(cls, name_or_path: str) -> "Project":
        p = Path(name_or_path)
        if p.is_dir() and (p / "project.yaml").exists():
            return cls(p)
        candidate = repo_root() / "projects" / name_or_path
        if candidate.is_dir():
            return cls(candidate)
        raise FileNotFoundError(f"no project {name_or_path!r} (looked in {candidate})")

    # --- config blocks -------------------------------------------------
    @property
    def roles(self) -> dict[str, Any]:
        return dict(self.cfg.get("roles") or {})

    @property
    def order_col(self) -> str | None:
        v = self.roles.get("order")
        return str(v) if v else None

    @property
    def period_col(self) -> str | None:
        v = self.roles.get("period")
        return str(v) if v else None

    @property
    def id_cols(self) -> list[str]:
        return [str(c) for c in (self.roles.get("id") or [])]

    # --- files ---------------------------------------------------------
    def path(self, rel: str) -> Path:
        return self.dir / rel

    @property
    def prepared_path(self) -> Path:
        return self.path(str((self.cfg.get("data") or {}).get("prepared") or "data/prepared.parquet"))

    @property
    def features_path(self) -> Path:
        return self.path(str((self.cfg.get("features") or {}).get("cache") or "data/features.parquet"))

    @property
    def runs_dir(self) -> Path:
        return self.dir / "runs"

    @property
    def models_dir(self) -> Path:
        return self.dir / "models"

    def module(self, block: str) -> ModuleType:
        """Load the python file named by cfg[block]['module'] (cached)."""
        if block in self._modules:
            return self._modules[block]
        section = self.cfg.get(block) or {}
        rel = section.get("module")
        if not rel:
            raise FileNotFoundError(f"project.yaml has no {block}.module")
        mod = load_module(self.path(str(rel)), f"boost_project_{self.name}_{block}")
        self._modules[block] = mod
        return mod

    def has_module(self, block: str) -> bool:
        section = self.cfg.get(block) or {}
        rel = section.get("module")
        return bool(rel) and self.path(str(rel)).exists()

    def model_names(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(p.stem for p in self.models_dir.glob("*.yaml"))

    def model_cfg(self, name: str) -> dict[str, Any]:
        path = self.models_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no model config {path}. available: {self.model_names()}")
        cfg = load_yaml(path)
        cfg.setdefault("name", name)
        return cfg

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

TASK_TYPES = ("binary", "multiclass", "regression", "poisson")


@dataclass(frozen=True)
class Task:
    """What is being predicted. `classes` are label values in the target column."""

    type: str
    target: str
    classes: tuple[str, ...] | None = None

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "Task":
        t = str(cfg["type"])
        if t not in TASK_TYPES:
            raise ValueError(f"task.type must be one of {TASK_TYPES}, got {t!r}")
        classes = cfg.get("classes")
        if t == "multiclass" and not classes:
            raise ValueError("multiclass task needs task.classes")
        if t == "binary" and classes and len(classes) != 2:
            raise ValueError("binary task.classes must have exactly two entries [negative, positive]")
        return cls(type=t, target=str(cfg["target"]), classes=tuple(str(c) for c in classes) if classes else None)

    @property
    def is_classification(self) -> bool:
        return self.type in ("binary", "multiclass")

    @property
    def n_classes(self) -> int:
        if self.type == "multiclass":
            return len(self.classes or ())
        if self.type == "binary":
            return 2
        return 1

    @property
    def class_names(self) -> list[str]:
        if self.type == "multiclass":
            return list(self.classes or ())
        if self.type == "binary":
            return list(self.classes) if self.classes else ["0", "1"]
        return []

    @property
    def prob_columns(self) -> list[str]:
        return [f"p_{c}" for c in self.class_names]

    def encode(self, series: pl.Series) -> np.ndarray:
        """Target column -> numeric array the library can fit."""
        if self.type == "multiclass":
            mapping = {c: i for i, c in enumerate(self.class_names)}
            vals = series.cast(pl.Utf8).to_list()
            bad = sorted({v for v in vals if v not in mapping})
            if bad:
                raise ValueError(f"{self.target}: values not in task.classes: {bad[:10]}")
            return np.array([mapping[v] for v in vals], dtype=np.int64)
        if self.type == "binary":
            if self.classes:
                mapping = {self.classes[0]: 0, self.classes[1]: 1}
                vals = series.cast(pl.Utf8).to_list()
                bad = sorted({v for v in vals if v not in mapping})
                if bad:
                    raise ValueError(f"{self.target}: values not in task.classes: {bad[:10]}")
                return np.array([mapping[v] for v in vals], dtype=np.int64)
            arr = series.cast(pl.Float64).to_numpy()
            if not np.all(np.isin(arr, (0.0, 1.0))):
                raise ValueError(f"{self.target}: binary target must be 0/1 or declare task.classes")
            return arr.astype(np.int64)
        return series.cast(pl.Float64).to_numpy()


@dataclass
class Fold:
    """Row indices into the model frame. valid is for early stopping; test is scored."""

    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


@dataclass
class Section:
    """A block of a run report. Produced by core and by project evaluation hooks."""

    title: str
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)  # file names inside run_dir
    notes: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    """What an evaluation hook gets to see."""

    project: Any
    model_cfg: dict[str, Any]
    task: Task
    run_dir: Path
    predictions: pl.DataFrame
    meta: dict[str, Any]

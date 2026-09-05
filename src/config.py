"""Typed access to ``config/settings.yaml``.

Alpha, replication counts, sample sizes, skew, variance ratios and seeds live in the YAML file,
never in the modules that use them. Only ``profile`` is overridable from the environment
(``HTA_PROFILE=ci``), because that is the one knob CI needs to turn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

ProfileName = Literal["full", "ci"]
PopulationKind = Literal["normal", "lognormal", "heavy_tailed", "contaminated"]


class Profile(BaseModel):
    """How many replications, shuffles, resamples and posterior draws to run."""

    n_reps: int = Field(gt=0)
    n_perm: int = Field(gt=0)
    n_boot: int = Field(gt=0)
    n_posterior: int = Field(gt=0)


class PopulationSpec(BaseModel):
    """True population parameters for one kind of population.

    ``mean`` and ``sd`` are always the *population* mean and standard deviation, so that an
    effect expressed in Cohen's d units means the same thing for every kind.
    """

    mean: float
    sd: float = Field(gt=0)
    df: float = Field(default=3.0, gt=2.0)
    contamination: float = Field(default=0.1, ge=0.0, lt=1.0)
    outlier_sd_mult: float = Field(default=10.0, gt=0)


class Grid(BaseModel):
    """The knobs the notebooks sweep."""

    effect_sizes: list[float]
    sample_sizes: list[int]
    sd_ratios: list[float]
    n_small: int = Field(gt=1)
    n_large: int = Field(gt=1)
    arena_effect_d: float


class NotebookSettings(BaseModel):
    """Knobs used only by the notebooks."""

    power_curve_reps: int = Field(default=2000, gt=0)
    multiple_comparisons_k: int = Field(default=20, gt=1)
    demo_seed: int = 7


class Settings(BaseSettings):
    """The resolved project configuration."""

    model_config = SettingsConfigDict(env_prefix="HTA_", extra="ignore")

    profile: ProfileName = "full"
    seed: int = 20250905
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    profiles: dict[str, Profile]
    populations: dict[str, PopulationSpec]
    grid: Grid
    notebooks: NotebookSettings = NotebookSettings()

    @property
    def reps(self) -> Profile:
        """Replication counts for the active profile."""
        return self.profiles[self.profile]

    def population(self, kind: PopulationKind) -> PopulationSpec:
        """Population defaults for ``kind``."""
        try:
            return self.populations[kind]
        except KeyError as exc:  # pragma: no cover - configuration error
            known = ", ".join(sorted(self.populations))
            raise KeyError(f"unknown population kind {kind!r}; known kinds: {known}") from exc


@lru_cache(maxsize=4)
def get_settings(path: Path | None = None) -> Settings:
    """Load and cache the project settings.

    The YAML file supplies every field except ``profile``, which is left to the ``HTA_PROFILE``
    environment variable (pydantic-settings gives init arguments priority over the environment,
    so anything present in the YAML wins).
    """
    config_path = path or CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Settings(**data)


def _jsonable(value: Any) -> Any:
    """Recursively coerce the numpy and pandas types the notebooks produce into plain JSON.

    Non-finite floats become ``null``: the arena legitimately carries NaN in ``target`` for
    power rows, and ``json.dumps`` would otherwise emit a bare ``NaN``, which is not valid JSON
    and which most readers of these files will choke on.
    """
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_results(name: str, payload: Mapping[str, Any]) -> str:
    """Write one notebook's numbers and claims to ``outputs/results/<name>.json``.

    Every number quoted in the article comes out of one of these files. Nothing that appears in
    prose should exist only inside a notebook's output cell, where it cannot be checked against
    the code that produced it.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)

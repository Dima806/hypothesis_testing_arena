"""Shared fixtures and the Monte Carlo tolerance rule for the pytest suite.

Every simulation assertion in this suite is deterministic (fixed seeds) *and* given a tolerance
derived from the binomial standard error rather than from a number that happened to pass once.
``FAST_REPS`` is small enough to keep ``make test`` in the tens of seconds and large enough that
the effects being asserted are many standard errors wide.
"""

from __future__ import annotations

import math

import pytest

from src.config import Settings, get_settings
from src.simulation.scenarios import Scenario, base_scenarios

FAST_REPS = 800
"""Replications per simulation assertion. mc_se at p = 0.05 is 0.0077."""

FAST_PERM = 199
"""Shuffles/resamples per resampling test in the suite. Minimum attainable p is 0.005."""

SEED = 20250905


def mc_tolerance(rate: float, n_reps: int, n_se: float = 4.0) -> float:
    """Monte Carlo tolerance for a rejection rate: ``n_se`` binomial standard errors.

    Four rather than three: several of these assertions sweep seven scenarios at once, and at
    three standard errors a seven-way sweep would raise a false alarm roughly one run in fifty.
    """
    return n_se * math.sqrt(rate * (1.0 - rate) / n_reps)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def scenarios(settings: Settings) -> dict[str, Scenario]:
    """Base scenarios by name, still at ``effect_d = 0``."""
    return {scenario.name: scenario for scenario in base_scenarios(settings)}

"""Scoring: how often each test rejects, and whether that rate is the promised one.

This module holds the vocabulary the four evaluation modules share - the replication runner and
the rate result - the same way ``src.tests`` holds :class:`~src.tests.TestResult`.

Two generators, deliberately
----------------------------
:func:`run_replications` uses a **data** generator and a separate **test** generator. The data
stream depends only on the scenario, so every contender in the arena sees byte-identical
samples; the test stream depends on the scenario and the test, so the resampling tests get
independent randomness without perturbing the data. That makes the arena a paired comparison,
which removes a large chunk of Monte Carlo noise from every head-to-head statement in the
project.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.simulation.scenarios import Scenario
from src.tests import FloatArray, TestResult

TestFn = Callable[..., TestResult]
Metric = Literal["fpr", "power"]


def derive_seed(master_seed: int, *keys: int) -> int:
    """Derive a reproducible child seed from the master seed and integer keys."""
    sequence = np.random.SeedSequence([master_seed, *keys])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def name_key(name: str) -> int:
    """Stable integer key for a string, so seeds do not depend on ``hash()`` randomization."""
    key = 0
    for char in name.encode("utf-8"):
        key = (key * 131 + char) % 2_147_483_647
    return key


@dataclass(frozen=True)
class Replications:
    """The raw outcome of repeating one test on one scenario ``n_reps`` times."""

    test: str
    scenario: str
    rejects: FloatArray
    p_values: FloatArray
    effects: FloatArray
    alpha: float
    data_seed: int
    test_seed: int

    @property
    def n_reps(self) -> int:
        return int(self.rejects.size)

    @property
    def reject_rate(self) -> float:
        return float(self.rejects.mean())

    @property
    def has_p_values(self) -> bool:
        """False for the Bayesian contender, which reports a posterior probability instead."""
        return bool(np.isfinite(self.p_values).any())

    @property
    def valid_p_values(self) -> FloatArray:
        return np.asarray(self.p_values[np.isfinite(self.p_values)], dtype=np.float64)


@dataclass(frozen=True)
class RateResult:
    """A rejection rate with its Monte Carlo uncertainty.

    Never quote ``rate`` without ``mc_se``: at 1 000 replications a rate of 0.054 against a
    nominal 0.05 is noise, not inflation.
    """

    test: str
    scenario: str
    metric: Metric
    rate: float
    n_reps: int
    alpha: float

    @property
    def mc_se(self) -> float:
        """Monte Carlo standard error of the rate, ``sqrt(p (1 - p) / n)``."""
        return math.sqrt(self.rate * (1.0 - self.rate) / self.n_reps)

    @property
    def ci(self) -> tuple[float, float]:
        """Approximate 95% Monte Carlo interval for the rate."""
        half = 1.96 * self.mc_se
        return (max(0.0, self.rate - half), min(1.0, self.rate + half))

    def deviation_in_se(self, target: float) -> float:
        """How many Monte Carlo standard errors the rate sits from ``target``."""
        return (self.rate - target) / self.mc_se if self.mc_se > 0 else 0.0


def run_replications(
    test_fn: TestFn,
    scenario: Scenario,
    *,
    n_reps: int,
    alpha: float = 0.05,
    data_seed: int = 0,
    test_seed: int = 1,
    test_name: str | None = None,
) -> Replications:
    """Repeat ``test_fn`` on fresh draws from ``scenario`` ``n_reps`` times."""
    data_rng = np.random.default_rng(data_seed)
    test_rng = np.random.default_rng(test_seed)

    rejects = np.empty(n_reps, dtype=np.float64)
    p_values = np.full(n_reps, np.nan, dtype=np.float64)
    effects = np.empty(n_reps, dtype=np.float64)
    name = test_name or "unknown"

    for i in range(n_reps):
        sample_a, sample_b = scenario.draw(data_rng)
        result = test_fn(sample_a, sample_b, alpha=alpha, rng=test_rng)
        rejects[i] = float(result.reject)
        effects[i] = result.effect
        if result.p_value is not None:
            p_values[i] = result.p_value
        if test_name is None:
            name = result.name

    return Replications(
        test=name,
        scenario=scenario.name,
        rejects=rejects,
        p_values=p_values,
        effects=effects,
        alpha=alpha,
        data_seed=data_seed,
        test_seed=test_seed,
    )

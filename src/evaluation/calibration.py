"""p-value calibration: under the null, p-values must be Uniform(0, 1).

This is the deepest check in the project, and the one that turns "the t-test is unreliable here"
into something you can see. A valid test rejects at exactly alpha *for every* alpha, which is
the same statement as "the p-values are uniform". When the assumptions break, the p-value
distribution bends away from the diagonal, and the direction of the bend says which failure you
have: bowed above the diagonal means too many small p-values (phantom discoveries), bowed below
means the test is conservative and is quietly spending its power.

The Bayesian contender has no p-value; asking for its calibration raises rather than inventing
one for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.evaluation import TestFn, run_replications
from src.simulation.scenarios import Scenario
from src.tests import FloatArray


@dataclass(frozen=True)
class CalibrationResult:
    """p-values from repeated runs under the null, and how far they are from uniform."""

    test: str
    scenario: str
    p_values: FloatArray
    ks_statistic: float
    ks_p_value: float
    alpha: float

    @property
    def n_reps(self) -> int:
        return int(self.p_values.size)

    @property
    def rejection_rate(self) -> float:
        """The empirical size at ``alpha`` - the same number as the false-positive rate."""
        return float(np.mean(self.p_values <= self.alpha))

    def is_uniform(self, *, level: float = 0.01) -> bool:
        """Does a Kolmogorov-Smirnov test fail to reject uniformity at ``level``?"""
        return self.ks_p_value > level


def pvalue_calibration(
    test_fn: TestFn,
    scenario: Scenario,
    *,
    n_reps: int,
    alpha: float = 0.05,
    data_seed: int = 0,
    test_seed: int = 1,
    test_name: str | None = None,
) -> CalibrationResult:
    """Collect null p-values from ``test_fn`` and test them against Uniform(0, 1)."""
    if not scenario.is_null:
        raise ValueError(
            f"scenario {scenario.name!r} has a true effect; p-values are only expected to be "
            "uniform under the null"
        )
    reps = run_replications(
        test_fn,
        scenario,
        n_reps=n_reps,
        alpha=alpha,
        data_seed=data_seed,
        test_seed=test_seed,
        test_name=test_name,
    )
    if not reps.has_p_values:
        raise ValueError(
            f"test {reps.test!r} does not produce p-values; calibrate its posterior "
            "probabilities separately rather than inventing a p-value for it"
        )
    p_values = reps.valid_p_values
    ks = stats.kstest(p_values, "uniform")
    return CalibrationResult(
        test=reps.test,
        scenario=reps.scenario,
        p_values=p_values,
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
        alpha=alpha,
    )


def uniform_pp_points(p_values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return ``(theoretical, empirical)`` coordinates for a PP-plot against Uniform(0, 1).

    A perfectly calibrated test lies on the diagonal.
    """
    ordered = np.sort(np.asarray(p_values, dtype=np.float64))
    theoretical = (np.arange(1, ordered.size + 1) - 0.5) / ordered.size
    return np.asarray(theoretical, dtype=np.float64), ordered


def rejection_rate_curve(
    p_values: FloatArray, levels: FloatArray | None = None
) -> tuple[FloatArray, FloatArray]:
    """Empirical rejection rate as a function of the nominal level.

    The honest version of the calibration plot: a valid test traces the diagonal, so any gap is
    read directly as "the level you asked for versus the level you got".
    """
    grid = levels if levels is not None else np.linspace(0.001, 0.2, 200)
    grid = np.asarray(grid, dtype=np.float64)
    observed = np.array([float(np.mean(p_values <= level)) for level in grid])
    return grid, observed

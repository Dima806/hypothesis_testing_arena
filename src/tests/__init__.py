"""The six group-difference tests, plus the vocabulary they share.

This package holds *statistical* tests, not pytest cases; the pytest suite lives in the
top-level ``tests/`` directory and never collects from here.

Conventions enforced across all six modules
-------------------------------------------
* Samples are passed as ``(a, b)`` where ``a`` is control/group 1 and ``b`` is treatment/group 2.
* The effect is **always** ``b - a``. ``alternative="greater"`` always means "b exceeds a".
* Every test returns a :class:`TestResult`, so the arena can score all six identically.
* Variances always use ``ddof=1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
Alternative = Literal["two-sided", "less", "greater"]

__all__ = [
    "Alternative",
    "FloatArray",
    "Statistic",
    "TestResult",
    "as_samples",
    "mean_difference",
    "median_difference",
    "p_value_from_t",
    "trimmed_mean_difference",
    "welch_t_statistic",
]


@dataclass(frozen=True)
class TestResult:
    """The uniform verdict returned by all six tests.

    Attributes
    ----------
    name:
        Registry name of the test ("student", "welch", ...).
    statistic:
        The test statistic (t, U, a studentized permutation statistic, ...). ``None`` when the
        contender has no test statistic.
    p_value:
        ``None`` for the Bayesian contender, which reports a posterior probability instead. A
        p-value of exactly zero is never returned by the resampling tests.
    effect:
        Point estimate of ``b - a`` on the scale of the data.
    ci:
        Interval estimate of ``b - a`` at level ``1 - alpha``, or ``None``.
    reject:
        The decision the arena scores. Every contender must define it; the definition is
        documented in the module that produces it.
    extra:
        Test-specific extras (degrees of freedom, posterior probability, resample counts, ...).
    """

    name: str
    statistic: float | None
    p_value: float | None
    effect: float
    ci: tuple[float, float] | None
    alpha: float
    reject: bool
    extra: dict[str, float] = field(default_factory=dict)

    # Keeps pytest from trying to collect this as a test class when a test module imports it.
    __test__ = False


class Statistic(Protocol):
    """A two-sample statistic that reduces over the last axis.

    Implementations must accept ``(..., n_a)`` and ``(..., n_b)`` arrays so the resampling tests
    can evaluate thousands of shuffles or resamples in a single vectorized call.
    """

    def __call__(self, a: FloatArray, b: FloatArray) -> FloatArray: ...


def as_samples(a: npt.ArrayLike, b: npt.ArrayLike, *, min_size: int = 2) -> tuple[FloatArray, ...]:
    """Coerce two samples to 1-D float arrays and reject degenerate input."""
    arr_a = np.asarray(a, dtype=np.float64).ravel()
    arr_b = np.asarray(b, dtype=np.float64).ravel()
    if arr_a.size < min_size or arr_b.size < min_size:
        raise ValueError(f"each sample needs at least {min_size} observations")
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        raise ValueError("samples must be finite")
    return arr_a, arr_b


def mean_difference(a: FloatArray, b: FloatArray) -> FloatArray:
    """``mean(b) - mean(a)``, reducing over the last axis."""
    return np.asarray(b.mean(axis=-1) - a.mean(axis=-1), dtype=np.float64)


def median_difference(a: FloatArray, b: FloatArray) -> FloatArray:
    """``median(b) - median(a)``, reducing over the last axis."""
    return np.asarray(np.median(b, axis=-1) - np.median(a, axis=-1), dtype=np.float64)


def trimmed_mean_difference(a: FloatArray, b: FloatArray, proportion: float = 0.2) -> FloatArray:
    """Difference of symmetrically trimmed means, reducing over the last axis."""
    trimmed_a = stats.trim_mean(a, proportion, axis=-1)
    trimmed_b = stats.trim_mean(b, proportion, axis=-1)
    return np.asarray(trimmed_b - trimmed_a, dtype=np.float64)


def welch_t_statistic(a: FloatArray, b: FloatArray) -> FloatArray:
    """The Welch t statistic, reducing over the last axis.

    This is the default statistic for the permutation test. Studentizing matters: a permutation
    test on the raw mean difference is exact only under full exchangeability (identical
    distributions), so it does *not* hold its false-positive rate when the two groups have
    unequal variances and unequal sample sizes. Studentizing restores asymptotic validity there
    (Janssen 1997; Chung and Romano 2013), which is what lets the permutation test hold alpha
    across every scenario in the arena.
    """
    n_a = a.shape[-1]
    n_b = b.shape[-1]
    var_a = a.var(axis=-1, ddof=1)
    var_b = b.var(axis=-1, ddof=1)
    standard_error = np.sqrt(var_a / n_a + var_b / n_b)
    standard_error = np.where(standard_error == 0.0, np.finfo(np.float64).tiny, standard_error)
    return np.asarray((b.mean(axis=-1) - a.mean(axis=-1)) / standard_error, dtype=np.float64)


def p_value_from_t(t_stat: float, df: float, alternative: Alternative) -> float:
    """Convert a t statistic to a p-value under the given alternative (``b - a``)."""
    if alternative == "two-sided":
        return float(2.0 * stats.t.sf(abs(t_stat), df))
    if alternative == "greater":
        return float(stats.t.sf(t_stat, df))
    return float(stats.t.cdf(t_stat, df))

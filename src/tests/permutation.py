"""The permutation test, from scratch.

Assumes nothing about the shape of either distribution. Under the null the group labels are
arbitrary, so shuffling them and recomputing the statistic builds the null distribution
directly, out of the data actually in hand.

Two design decisions worth knowing
----------------------------------
1. **The default statistic is the Welch t, not the raw mean difference.** A permutation test on
   the mean difference is exact only under full exchangeability (identical distributions). With
   unequal variances *and* unequal sample sizes it does not hold its false-positive rate when
   testing equal means. Studentizing restores asymptotic validity there, and it is what lets the
   permutation test hold alpha across the whole arena. ``effect_fn`` is separate so the reported
   effect stays on the scale of the data.
2. **The p-value uses the add-one correction**, ``p = (1 + #{as extreme}) / (n_perm + 1)``. It
   can never be zero: with ``n_perm`` shuffles the smallest attainable p-value is
   ``1 / (n_perm + 1)``, and pretending otherwise claims more evidence than was gathered.

Works for any statistic, which is the other reason to reach for it: swap in a median or a
trimmed mean and nothing else changes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng

from src.tests import (
    Alternative,
    FloatArray,
    Statistic,
    TestResult,
    as_samples,
    mean_difference,
    welch_t_statistic,
)

MAX_ELEMENTS = 2_000_000
"""Chunk size for the shuffle matrix, in array elements. 2e6 float64 is 16 MB."""


def permutation_null_distribution(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    statistic: Statistic = welch_t_statistic,
    n_perm: int = 999,
    rng: Generator | None = None,
    max_elements: int = MAX_ELEMENTS,
) -> FloatArray:
    """Return ``n_perm`` draws from the permutation null distribution of ``statistic``.

    Exposed on its own because the null distribution is the figure: notebook 03 and the
    Streamlit app plot it with the observed statistic marked in it.
    """
    sample_a, sample_b = as_samples(a, b)
    generator = rng if rng is not None else default_rng()
    pooled = np.concatenate([sample_a, sample_b])
    n_a = sample_a.size
    n_total = pooled.size

    chunk = max(1, min(n_perm, max_elements // n_total))
    draws: list[FloatArray] = []
    drawn = 0
    while drawn < n_perm:
        size = min(chunk, n_perm - drawn)
        shuffled = generator.permuted(np.tile(pooled, (size, 1)), axis=1)
        draws.append(np.atleast_1d(statistic(shuffled[:, :n_a], shuffled[:, n_a:])))
        drawn += size
    return np.concatenate(draws)


def _count_at_least_as_extreme(null: FloatArray, observed: float, alternative: Alternative) -> int:
    """Count null draws at least as extreme as ``observed``, with a float-safety tolerance."""
    tolerance = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        return int(np.count_nonzero(np.abs(null) >= abs(observed) - tolerance))
    if alternative == "greater":
        return int(np.count_nonzero(null >= observed - tolerance))
    return int(np.count_nonzero(null <= observed + tolerance))


def permutation_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    statistic: Statistic = welch_t_statistic,
    effect_fn: Statistic = mean_difference,
    n_perm: int = 999,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    rng: Generator | None = None,
    max_elements: int = MAX_ELEMENTS,
) -> TestResult:
    """Run a permutation test by shuffling the group labels ``n_perm`` times.

    ``statistic`` builds the null distribution (default: the Welch t, see the module docstring);
    ``effect_fn`` is the reported point estimate of ``b - a`` (default: the mean difference).
    ``reject`` is ``p_value <= alpha``. No confidence interval is produced - a permutation test
    is a test, not an interval estimator; use the bootstrap for that.
    """
    sample_a, sample_b = as_samples(a, b)
    generator = rng if rng is not None else default_rng()

    observed = float(np.asarray(statistic(sample_a, sample_b)).item())
    null = permutation_null_distribution(
        sample_a,
        sample_b,
        statistic=statistic,
        n_perm=n_perm,
        rng=generator,
        max_elements=max_elements,
    )
    count = _count_at_least_as_extreme(null, observed, alternative)
    p_value = (1.0 + count) / (n_perm + 1.0)

    return TestResult(
        name="permutation",
        statistic=observed,
        p_value=p_value,
        effect=float(np.asarray(effect_fn(sample_a, sample_b)).item()),
        ci=None,
        alpha=alpha,
        reject=p_value <= alpha,
        extra={
            "n_perm": float(n_perm),
            "n_at_least_as_extreme": float(count),
            "min_attainable_p": 1.0 / (n_perm + 1.0),
            "null_sd": float(null.std(ddof=1)),
        },
    )

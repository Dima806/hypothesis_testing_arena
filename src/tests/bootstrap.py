"""The bootstrap difference test, from scratch.

Resample each group with replacement, recompute the difference, and read the uncertainty of the
difference straight off the resampling distribution. Assumes nothing about the shape. Because
each group is resampled separately, unequal variances are handled for free.

Two modes, and the difference matters
-------------------------------------
* ``null_shift=False`` (default): report the percentile confidence interval of the difference
  and reject when it excludes zero. This is a **CI-inversion decision, not a p-value** - it is
  the "how big is the difference and how sure are we" framing, which is the honest reason to
  prefer the bootstrap in the first place.
* ``null_shift=True``: additionally build a null distribution by shifting group ``b`` down by
  the observed effect, so the null is true by construction, then resample from the shifted data
  and count how often the resampled statistic is as extreme as the observed one. This yields an
  actual p-value, so it is what the arena registers - the other five contenders are compared on
  p-values and calibration, and bootstrap needs one to join that comparison.

The shift is applied to ``b`` alone, which zeroes any translation-equivariant statistic (mean
difference, median difference, trimmed-mean difference).

Why the null-shift p-value is studentized by default
----------------------------------------------------
Comparing raw mean differences to a null-shifted resampling distribution runs above alpha on
small unbalanced samples - measured at 0.083 against a nominal 0.05 with n = 100 vs 15 and a
4x variance ratio. Comparing Welch t statistics instead brings that back to 0.047, the same fix
and the same reason as the studentized permutation test. The cost is real and is stated rather
than hidden: on heavily contaminated data the studentized version becomes *conservative*
(measured at 0.015), which spends power. That is the failure direction to prefer in a project
about phantom discoveries, but it is still a failure. Pass ``studentize=False`` to see the
inflated version, and note that studentizing assumes a mean-based ``statistic``.
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


def _resample(sample: FloatArray, n_boot: int, rng: Generator) -> FloatArray:
    """Draw ``n_boot`` bootstrap resamples of ``sample`` as an ``(n_boot, n)`` array."""
    indices = rng.integers(0, sample.size, size=(n_boot, sample.size))
    return np.asarray(sample[indices], dtype=np.float64)


def bootstrap_distribution(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    statistic: Statistic = mean_difference,
    n_boot: int = 999,
    rng: Generator | None = None,
) -> FloatArray:
    """Return the bootstrap resampling distribution of ``statistic``."""
    sample_a, sample_b = as_samples(a, b)
    generator = rng if rng is not None else default_rng()
    return np.atleast_1d(
        statistic(_resample(sample_a, n_boot, generator), _resample(sample_b, n_boot, generator))
    )


def bootstrap_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    statistic: Statistic = mean_difference,
    n_boot: int = 999,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    rng: Generator | None = None,
    null_shift: bool = False,
    studentize: bool = True,
) -> TestResult:
    """Bootstrap the difference between two groups.

    With ``null_shift=False`` the decision comes from the percentile interval; ``p_value`` is
    ``None``. With ``null_shift=True`` the decision comes from a bootstrap p-value computed
    under a null-shifted distribution, and the interval is still reported. ``studentize`` only
    affects that p-value; see the module docstring for why it defaults to on.
    """
    sample_a, sample_b = as_samples(a, b)
    generator = rng if rng is not None else default_rng()

    observed = float(np.asarray(statistic(sample_a, sample_b)).item())
    boot = np.atleast_1d(
        statistic(_resample(sample_a, n_boot, generator), _resample(sample_b, n_boot, generator))
    )

    if alternative == "two-sided":
        lower, upper = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        ci = (float(lower), float(upper))
        ci_rejects = ci[0] > 0.0 or ci[1] < 0.0
    elif alternative == "greater":
        ci = (float(np.percentile(boot, 100.0 * alpha)), float(np.inf))
        ci_rejects = ci[0] > 0.0
    else:
        ci = (float(-np.inf), float(np.percentile(boot, 100.0 * (1.0 - alpha))))
        ci_rejects = ci[1] < 0.0

    extra = {
        "n_boot": float(n_boot),
        "bootstrap_sd": float(boot.std(ddof=1)),
        "ci_excludes_zero": float(ci_rejects),
    }

    p_value: float | None = None
    reject = ci_rejects
    if null_shift:
        resampled_a = _resample(sample_a, n_boot, generator)
        resampled_b = _resample(sample_b - observed, n_boot, generator)
        if studentize:
            null = np.atleast_1d(welch_t_statistic(resampled_a, resampled_b))
            reference = float(np.asarray(welch_t_statistic(sample_a, sample_b)).item())
        else:
            null = np.atleast_1d(statistic(resampled_a, resampled_b))
            reference = observed

        if alternative == "two-sided":
            count = int(np.count_nonzero(np.abs(null) >= abs(reference)))
        elif alternative == "greater":
            count = int(np.count_nonzero(null >= reference))
        else:
            count = int(np.count_nonzero(null <= reference))
        p_value = (1.0 + count) / (n_boot + 1.0)
        reject = p_value <= alpha
        extra["min_attainable_p"] = 1.0 / (n_boot + 1.0)
        extra["studentized"] = float(studentize)

    return TestResult(
        name="bootstrap",
        statistic=observed,
        p_value=p_value,
        effect=observed,
        ci=ci,
        alpha=alpha,
        reject=reject,
        extra=extra,
    )

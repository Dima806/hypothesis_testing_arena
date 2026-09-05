"""Bayesian estimation of the difference in means, from scratch.

Answers the question stakeholders actually ask - "how likely is it that b beats a, and by how
much" - instead of "how surprising would this data be if nothing were going on".

Model
-----
Independent reference (Jeffreys) priors on each group's mean and variance give a closed-form
marginal posterior for each mean:

``mu | data  ~  t_{n-1}(mean, s / sqrt(n))``

so the posterior for the difference is obtained by drawing from both and subtracting. Exact, no
sampler, no PyMC, and fast enough to run inside a ten-thousand-replication loop.

Reading the output honestly
---------------------------
* ``p_value`` is ``None``. This contender does not produce one, and manufacturing one would
  defeat the point.
* ``reject`` is defined as *the credible interval of the difference excludes the ROPE* (by
  default the ROPE is the single point zero). It is a decision rule, so the arena can score it,
  but scoring a Bayesian rule by its long-run false-positive rate is a frequentist audit of a
  Bayesian procedure - label it that way in any write-up.
* With these priors the two-group problem is the Behrens-Fisher problem, so this contender is
  expected to behave much like Welch. That is a result, not a bug.

A Kruschke-style BEST model with a Student-t likelihood (robust to outliers, needs a sampler) is
a documented extension, not the default.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng

from src.tests import Alternative, FloatArray, TestResult, as_samples


def posterior_mean_draws(sample: FloatArray, n_draws: int, rng: Generator) -> FloatArray:
    """Draw from the marginal posterior of the mean under a Jeffreys prior."""
    n = sample.size
    scale = float(sample.std(ddof=1)) / np.sqrt(n)
    return np.asarray(sample.mean() + scale * rng.standard_t(n - 1, size=n_draws))


def posterior_difference(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_posterior: int = 4000,
    rng: Generator | None = None,
) -> FloatArray:
    """Return posterior draws of ``mu_b - mu_a``."""
    sample_a, sample_b = as_samples(a, b)
    generator = rng if rng is not None else default_rng()
    return posterior_mean_draws(sample_b, n_posterior, generator) - posterior_mean_draws(
        sample_a, n_posterior, generator
    )


def bayesian_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    n_posterior: int = 4000,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    rope: tuple[float, float] | None = None,
    rng: Generator | None = None,
) -> TestResult:
    """Estimate ``mu_b - mu_a`` and report the probability that b exceeds a.

    ``extra["prob_effect"]`` is P(mu_b > mu_a). ``ci`` is the equal-tailed ``1 - alpha``
    credible interval of the difference.
    """
    sample_a, sample_b = as_samples(a, b)
    generator = rng if rng is not None else default_rng()
    draws = posterior_mean_draws(sample_b, n_posterior, generator) - posterior_mean_draws(
        sample_a, n_posterior, generator
    )

    prob_effect = float(np.mean(draws > 0.0))
    low, high = rope if rope is not None else (0.0, 0.0)

    if alternative == "two-sided":
        lower, upper = np.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        ci = (float(lower), float(upper))
        reject = ci[0] > high or ci[1] < low
    elif alternative == "greater":
        ci = (float(np.percentile(draws, 100.0 * alpha)), float(np.inf))
        reject = ci[0] > high
    else:
        ci = (float(-np.inf), float(np.percentile(draws, 100.0 * (1.0 - alpha))))
        reject = ci[1] < low

    prob_rope = float(np.mean((draws >= low) & (draws <= high))) if rope is not None else 0.0
    return TestResult(
        name="bayesian",
        statistic=None,
        p_value=None,
        effect=float(sample_b.mean() - sample_a.mean()),
        ci=ci,
        alpha=alpha,
        reject=reject,
        extra={
            "prob_effect": prob_effect,
            "posterior_mean": float(draws.mean()),
            "posterior_sd": float(draws.std(ddof=1)),
            "prob_rope": prob_rope,
            "n_posterior": float(n_posterior),
        },
    )

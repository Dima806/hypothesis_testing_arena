"""Welch's two-sample t-test, from scratch.

Assumptions
-----------
1. Both groups are normally distributed.
2. Nothing about the variances.

Dropping the equal-variance assumption costs a slightly more awkward degrees-of-freedom formula
and essentially no power when the variances *are* equal. That is why Welch, not Student, is the
sensible default (Delacre, Lakens and Leys 2017), and why this project treats the choice of
Student over Welch as a free, avoidable error.

Formula
-------
``se = sqrt(s_a^2/n_a + s_b^2/n_b)``, ``t = (mean_b - mean_a) / se`` and the
Welch-Satterthwaite degrees of freedom
``df = se^4 / ((s_a^2/n_a)^2/(n_a - 1) + (s_b^2/n_b)^2/(n_b - 1))``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import stats

from src.tests import Alternative, TestResult, as_samples, p_value_from_t


def welch_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    rng: Generator | None = None,
) -> TestResult:
    """Run the unequal-variance two-sample t-test.

    ``reject`` is ``p_value <= alpha``. The confidence interval is always the two-sided
    ``1 - alpha`` interval for ``mean_b - mean_a``. ``rng`` is accepted and ignored.
    """
    del rng  # deterministic test; the parameter keeps the signature uniform
    sample_a, sample_b = as_samples(a, b)
    n_a = sample_a.size
    n_b = sample_b.size

    mean_a = float(sample_a.mean())
    mean_b = float(sample_b.mean())
    term_a = float(sample_a.var(ddof=1)) / n_a
    term_b = float(sample_b.var(ddof=1)) / n_b

    standard_error = float(np.sqrt(term_a + term_b))
    denominator = term_a**2 / (n_a - 1) + term_b**2 / (n_b - 1)
    df = float((term_a + term_b) ** 2 / denominator) if denominator > 0 else float(n_a + n_b - 2)

    effect = mean_b - mean_a
    if standard_error == 0.0:
        t_stat = 0.0 if effect == 0.0 else np.inf * np.sign(effect)
    else:
        t_stat = effect / standard_error

    p_value = p_value_from_t(float(t_stat), df, alternative)
    half_width = float(stats.t.ppf(1.0 - alpha / 2.0, df)) * standard_error

    return TestResult(
        name="welch",
        statistic=float(t_stat),
        p_value=p_value,
        effect=effect,
        ci=(effect - half_width, effect + half_width),
        alpha=alpha,
        reject=p_value <= alpha,
        extra={"df": df, "standard_error": standard_error},
    )

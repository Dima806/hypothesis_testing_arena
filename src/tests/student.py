"""Student's two-sample t-test, from scratch.

Assumptions
-----------
1. Both groups are normally distributed.
2. **The two groups have equal variances.**

Assumption 2 is the one almost nobody checks, and the one this project is built to break. When
the group with the larger variance is also the smaller group, the pooled variance underestimates
the standard error of the difference, the t statistic is too large, and the false-positive rate
climbs above alpha. With equal sample sizes the two errors cancel and Student is close to Welch,
which is why the failure is invisible in textbook examples.

Formula
-------
``s_p^2 = ((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2)``, ``df = n_a + n_b - 2``,
``t = (mean_b - mean_a) / sqrt(s_p^2 (1/n_a + 1/n_b))``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import stats

from src.tests import Alternative, TestResult, as_samples, p_value_from_t


def student_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    rng: Generator | None = None,
) -> TestResult:
    """Run the equal-variance two-sample t-test.

    ``reject`` is ``p_value <= alpha``. The confidence interval is always the two-sided
    ``1 - alpha`` interval for ``mean_b - mean_a``, whatever the alternative.

    ``rng`` is accepted and ignored; the test is deterministic. It exists so that every
    contender shares one call signature.
    """
    del rng  # deterministic test; the parameter keeps the signature uniform
    sample_a, sample_b = as_samples(a, b)
    n_a = sample_a.size
    n_b = sample_b.size

    mean_a = float(sample_a.mean())
    mean_b = float(sample_b.mean())
    var_a = float(sample_a.var(ddof=1))
    var_b = float(sample_b.var(ddof=1))

    df = float(n_a + n_b - 2)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
    standard_error = float(np.sqrt(pooled_var * (1.0 / n_a + 1.0 / n_b)))

    effect = mean_b - mean_a
    if standard_error == 0.0:
        t_stat = 0.0 if effect == 0.0 else np.inf * np.sign(effect)
    else:
        t_stat = effect / standard_error

    p_value = p_value_from_t(float(t_stat), df, alternative)
    half_width = float(stats.t.ppf(1.0 - alpha / 2.0, df)) * standard_error

    return TestResult(
        name="student",
        statistic=float(t_stat),
        p_value=p_value,
        effect=effect,
        ci=(effect - half_width, effect + half_width),
        alpha=alpha,
        reject=p_value <= alpha,
        extra={"df": df, "standard_error": standard_error, "pooled_var": pooled_var},
    )

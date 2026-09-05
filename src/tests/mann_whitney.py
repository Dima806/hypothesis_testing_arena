"""The Mann-Whitney U test (normal approximation), from scratch.

**Estimand caveat, stated everywhere this test appears.** Mann-Whitney does not test a
difference in means. It tests stochastic dominance: whether a random draw from one group tends
to exceed a random draw from the other. On symmetric distributions with a pure location shift
the two questions coincide, which is why the distinction is easy to miss; under skew or unequal
spread they come apart, and a Mann-Whitney rejection is *not* evidence that the means differ.
The arena reports its mean difference for comparability, but scores it on its own hypothesis.

Assumptions
-----------
Independent observations. No distributional shape is assumed. Under the null of *identical*
distributions the test is exact; it is not generally valid as a test of equal medians when the
two distributions have different shapes or spreads.

Formula
-------
Average ranks over the pooled sample, ``U_b = R_b - n_b (n_b + 1) / 2``,
``mu = n_a n_b / 2`` and the tie-corrected variance
``sigma^2 = (n_a n_b / 12) [(n + 1) - sum(t^3 - t) / (n (n - 1))]``, with a continuity
correction of 0.5. This mirrors ``scipy.stats.mannwhitneyu(method="asymptotic")``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import stats

from src.tests import Alternative, TestResult, as_samples


def mann_whitney_test(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    rng: Generator | None = None,
    use_continuity: bool = True,
) -> TestResult:
    """Run the Mann-Whitney U test using the tie-corrected normal approximation.

    ``statistic`` is ``U`` for group ``b``. ``reject`` is ``p_value <= alpha``. ``extra``
    carries ``prob_superiority`` = P(b > a) + P(tie)/2, the estimand this test is actually
    about, and the rank-biserial correlation. ``rng`` is accepted and ignored.

    No confidence interval is reported: an interval for the mean difference would misrepresent
    what the test estimates.
    """
    del rng  # deterministic test; the parameter keeps the signature uniform
    sample_a, sample_b = as_samples(a, b, min_size=1)
    n_a = sample_a.size
    n_b = sample_b.size
    n_total = n_a + n_b

    ranks = stats.rankdata(np.concatenate([sample_a, sample_b]))
    u_b = float(ranks[n_a:].sum() - n_b * (n_b + 1) / 2.0)
    u_a = float(n_a * n_b - u_b)

    _, tie_counts = np.unique(np.concatenate([sample_a, sample_b]), return_counts=True)
    tie_term = float(np.sum(tie_counts.astype(np.float64) ** 3 - tie_counts))
    mu = n_a * n_b / 2.0
    variance = (n_a * n_b / 12.0) * ((n_total + 1) - tie_term / (n_total * (n_total - 1)))
    sigma = float(np.sqrt(variance))

    # scipy takes U for the first argument under "greater"; our first argument is b.
    u_stat = {"greater": u_b, "less": u_a, "two-sided": max(u_a, u_b)}[alternative]
    numerator = u_stat - mu
    if use_continuity:
        numerator -= 0.5
    z_stat = numerator / sigma if sigma > 0 else 0.0

    survival = float(stats.norm.sf(z_stat))
    p_value = min(2.0 * survival, 1.0) if alternative == "two-sided" else survival

    prob_superiority = u_b / (n_a * n_b)
    return TestResult(
        name="mann_whitney",
        statistic=u_b,
        p_value=p_value,
        effect=float(sample_b.mean() - sample_a.mean()),
        ci=None,
        alpha=alpha,
        reject=p_value <= alpha,
        extra={
            "z": float(z_stat),
            "prob_superiority": prob_superiority,
            "rank_biserial": 2.0 * prob_superiority - 1.0,
            "median_difference": float(np.median(sample_b) - np.median(sample_a)),
        },
    )

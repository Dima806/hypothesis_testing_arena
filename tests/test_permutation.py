"""The permutation test must hold its false-positive rate everywhere it claims to.

This is the project's central positive claim, so it is asserted rather than asserted-in-prose.
The one scenario where it does not hold - extreme skew concentrated in the smaller group - is
asserted too, as a limit rather than as a success.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.evaluation import derive_seed, name_key
from src.evaluation.calibration import pvalue_calibration
from src.evaluation.false_positive import false_positive_rate, is_calibrated
from src.tests import mean_difference, trimmed_mean_difference
from src.tests.permutation import permutation_null_distribution, permutation_test
from tests.conftest import FAST_PERM, FAST_REPS, SEED, mc_tolerance

WELL_BEHAVED = [
    "normal_equal_var",
    "normal_unequal_var_equal_n",
    "normal_unequal_var_unequal_n",
    "skewed_small_n",
    "skewed_unequal_n",
    "heavy_tailed",
    "contaminated_outliers",
]


def _permutation(**kwargs):
    from functools import partial

    return partial(permutation_test, n_perm=FAST_PERM, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("scenario_name", WELL_BEHAVED)
def test_permutation_holds_alpha_across_scenarios(scenarios, scenario_name):
    """Normal, skewed, heavy-tailed, contaminated, balanced or not: alpha is alpha."""
    scenario = scenarios[scenario_name]
    result = false_positive_rate(
        _permutation(),
        scenario,
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key(scenario.name)),
        test_seed=derive_seed(SEED, name_key(scenario.name), name_key("permutation")),
        test_name="permutation",
    )
    tolerance = mc_tolerance(0.05, FAST_REPS)
    assert abs(result.rate - 0.05) <= tolerance, (
        f"{scenario_name}: permutation false-positive rate {result.rate:.4f} "
        f"is {result.deviation_in_se(0.05):.1f} Monte Carlo SEs from 0.05"
    )
    assert is_calibrated(result, n_se=4.0)


@pytest.mark.slow
def test_permutation_does_not_hold_alpha_under_extreme_skew(scenarios):
    """The honest limit. No mean-based test survives skewness 14 in the smaller group.

    Asserted so that the claim in the notebooks stays "permutation holds alpha across the
    conditions in the grid *except this one*", not "permutation always works".
    """
    scenario = scenarios["skewed_extreme_unequal"]
    result = false_positive_rate(
        _permutation(),
        scenario,
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key(scenario.name)),
        test_seed=derive_seed(SEED, name_key(scenario.name), name_key("permutation")),
        test_name="permutation",
    )
    assert result.rate > 0.08, result.rate


@pytest.mark.slow
def test_permutation_p_values_are_uniform_under_the_null(scenarios):
    """Holding alpha at one level is weaker than being uniform at every level."""
    scenario = scenarios["skewed_small_n"]
    calibration = pvalue_calibration(
        _permutation(),
        scenario,
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key("uniformity")),
        test_seed=derive_seed(SEED, name_key("uniformity"), name_key("permutation")),
        test_name="permutation",
    )
    # A discrete p-value grid of 200 values will not pass a strict KS test against a continuous
    # uniform, so the level is deliberately loose; the rejection rate check above is the sharp
    # one, and the mean of a Uniform(0, 1) is the shape check.
    assert calibration.ks_statistic < 0.1, calibration.ks_statistic
    assert float(np.mean(calibration.p_values)) == pytest.approx(0.5, abs=0.05)


def test_permutation_p_value_uses_the_add_one_correction():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.0, 1.0, 25), rng.normal(0.0, 1.0, 25)
    result = permutation_test(a, b, n_perm=99, rng=np.random.default_rng(1))
    assert result.p_value >= 1.0 / 100.0
    assert result.extra["min_attainable_p"] == pytest.approx(1.0 / 100.0)
    # The count is an integer number of shuffles out of n_perm.
    assert 0 <= result.extra["n_at_least_as_extreme"] <= 99


def test_permutation_is_exact_when_the_labels_are_genuinely_exchangeable():
    """Under identical distributions, the shuffle *is* the null - no approximation involved."""
    rng = np.random.default_rng(20250905)
    test_rng = np.random.default_rng(1)
    rejections = 0
    trials = 400
    for _ in range(trials):
        pooled = rng.normal(0.0, 1.0, 24)
        result = permutation_test(pooled[:12], pooled[12:], n_perm=FAST_PERM, rng=test_rng)
        rejections += int(result.reject)
    rate = rejections / trials
    assert abs(rate - 0.05) <= mc_tolerance(0.05, trials), rate


def test_studentizing_is_what_makes_the_permutation_test_robust(scenarios):
    """The design decision behind the default statistic, asserted rather than asserted-in-prose.

    On unequal variances with unequal group sizes, permuting raw mean differences does not hold
    alpha; permuting Welch t statistics does.
    """
    scenario = scenarios["normal_unequal_var_unequal_n"]
    common = {
        "n_reps": FAST_REPS,
        "alpha": 0.05,
        "data_seed": derive_seed(SEED, name_key("studentize")),
        "test_seed": derive_seed(SEED, name_key("studentize"), name_key("permutation")),
        "test_name": "permutation",
    }
    raw = false_positive_rate(_permutation(statistic=mean_difference), scenario, **common)
    studentized = false_positive_rate(_permutation(), scenario, **common)

    assert raw.rate > 0.10, raw.rate
    assert abs(studentized.rate - 0.05) <= mc_tolerance(0.05, FAST_REPS), studentized.rate


def test_permutation_null_distribution_is_centred_and_the_right_size():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0.0, 1.0, 30), rng.normal(0.0, 1.0, 30)
    null = permutation_null_distribution(
        a, b, statistic=mean_difference, n_perm=2000, rng=np.random.default_rng(1)
    )
    assert null.size == 2000
    assert float(null.mean()) == pytest.approx(0.0, abs=0.05)


def test_permutation_null_is_chunked_without_changing_the_answer():
    """Memory chunking is an implementation detail and must not move the p-value."""
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.0, 1.0, 40), rng.normal(0.5, 1.0, 40)
    whole = permutation_test(a, b, n_perm=500, rng=np.random.default_rng(7))
    chunked = permutation_test(a, b, n_perm=500, rng=np.random.default_rng(7), max_elements=400)
    assert whole.p_value == pytest.approx(chunked.p_value, abs=0.05)


def test_permutation_agrees_with_scipy_on_a_small_case():
    """Independent implementation check against scipy's own permutation machinery."""
    rng = np.random.default_rng(6)
    a, b = rng.normal(0.0, 1.0, 12), rng.normal(1.0, 1.0, 12)

    def statistic(x, y, axis=-1):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    theirs = stats.permutation_test(
        (a, b),
        statistic,
        permutation_type="independent",
        n_resamples=5000,
        random_state=np.random.default_rng(1),
        alternative="two-sided",
    )
    ours = permutation_test(
        a, b, statistic=mean_difference, n_perm=5000, rng=np.random.default_rng(1)
    )
    assert ours.p_value == pytest.approx(float(theirs.pvalue), abs=0.02)


def test_a_robust_statistic_changes_the_answer_on_contaminated_data():
    """The practical payoff of "any statistic": swap the mean out and the outliers stop mattering."""
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 1.0, 40)
    b = rng.normal(0.8, 1.0, 40)
    b[:3] = -60.0  # three logging errors, enough to hide a real effect from the mean

    on_the_mean = permutation_test(
        a,
        b,
        statistic=mean_difference,
        effect_fn=mean_difference,
        n_perm=999,
        rng=np.random.default_rng(1),
    )
    on_a_trimmed_mean = permutation_test(
        a,
        b,
        statistic=trimmed_mean_difference,
        effect_fn=trimmed_mean_difference,
        n_perm=999,
        rng=np.random.default_rng(1),
    )
    assert not on_the_mean.reject
    assert on_a_trimmed_mean.reject

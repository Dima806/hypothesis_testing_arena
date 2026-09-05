"""The showpiece, asserted: the Student t-test manufactures discoveries under unequal variance.

Also asserted, and just as important: the condition under which it does *not*. Student is close
to fine when the sample sizes are equal, which is exactly why the failure is invisible in
textbook examples and why this project has to simulate the unbalanced case to see it.
"""

from __future__ import annotations

import pytest

from src.evaluation import derive_seed, name_key
from src.evaluation.calibration import pvalue_calibration
from src.evaluation.false_positive import false_positive_rate, is_calibrated, is_inflated
from tests.conftest import FAST_PERM, FAST_REPS, SEED, mc_tolerance


def _rate(test_name, scenario, registry, *, tag=""):
    return false_positive_rate(
        registry[test_name],
        scenario,
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key(scenario.name + tag)),
        test_seed=derive_seed(SEED, name_key(scenario.name + tag), name_key(test_name)),
        test_name=test_name,
    )


@pytest.fixture(scope="module")
def registry(request):
    """The six contenders at test-suite resampling budgets."""
    from functools import partial

    from src.tests.bayesian import bayesian_test
    from src.tests.bootstrap import bootstrap_test
    from src.tests.mann_whitney import mann_whitney_test
    from src.tests.permutation import permutation_test
    from src.tests.student import student_test
    from src.tests.welch import welch_test

    del request
    return {
        "student": student_test,
        "welch": welch_test,
        "mann_whitney": mann_whitney_test,
        "permutation": partial(permutation_test, n_perm=FAST_PERM),
        "bootstrap": partial(bootstrap_test, n_boot=FAST_PERM, null_shift=True),
        "bayesian": partial(bayesian_test, n_posterior=1000),
    }


@pytest.mark.slow
def test_student_inflates_type_one_error_under_unequal_variance(scenarios, registry):
    """PRD success criterion: Student's false-positive rate is well above the nominal alpha.

    The scenario is n = 100 against n = 15 with the larger spread in the *smaller* group. The
    pooled variance is dominated by the large group, so it underestimates the standard error of
    the difference and every borderline result is pushed over the line.
    """
    scenario = scenarios["normal_unequal_var_unequal_n"]
    result = _rate("student", scenario, registry)
    assert result.rate > 0.15, (
        f"Student's false-positive rate was {result.rate:.4f}; the showpiece claim of this "
        "project is that it lands far above 0.05 here"
    )
    assert is_inflated(result, n_se=4.0)


@pytest.mark.slow
def test_welch_fixes_what_student_breaks(scenarios, registry):
    """PRD success criterion: Welch's false-positive rate is at the nominal alpha there."""
    scenario = scenarios["normal_unequal_var_unequal_n"]
    student = _rate("student", scenario, registry)
    welch = _rate("welch", scenario, registry)

    assert abs(welch.rate - 0.05) <= mc_tolerance(0.05, FAST_REPS), welch.rate
    assert is_calibrated(welch, n_se=4.0)
    assert welch.rate < student.rate / 3.0


@pytest.mark.slow
def test_the_assumption_free_tests_also_hold_alpha_there(scenarios, registry):
    scenario = scenarios["normal_unequal_var_unequal_n"]
    for name in ("permutation", "bootstrap", "bayesian"):
        result = _rate(name, scenario, registry)
        assert is_calibrated(result, n_se=4.0), f"{name}: {result.rate:.4f}"


@pytest.mark.slow
def test_student_is_fine_when_the_sample_sizes_are_equal(scenarios, registry):
    """The honesty check. Do not strawman the t-test.

    Same 4x variance ratio, but balanced groups: Student is close to nominal. Any claim that
    "unequal variance breaks the t-test" without this caveat is overstated.
    """
    scenario = scenarios["normal_unequal_var_equal_n"]
    result = _rate("student", scenario, registry)
    assert is_calibrated(result, n_se=4.0), result.rate


@pytest.mark.slow
def test_student_is_fine_on_clean_normal_data(scenarios, registry):
    """Its home ground, where it is the right tool and this project says so."""
    result = _rate("student", scenarios["normal_equal_var"], registry)
    assert is_calibrated(result, n_se=4.0), result.rate


@pytest.mark.slow
def test_student_p_values_are_not_uniform_when_its_assumptions_break(scenarios, registry):
    """The proof figure of notebook 02, as an assertion.

    A valid test's null p-values are Uniform(0, 1). Student's are not, and the Kolmogorov-
    Smirnov statistic against uniform separates the two cases by an order of magnitude.
    """
    broken = pvalue_calibration(
        registry["student"],
        scenarios["normal_unequal_var_unequal_n"],
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key("calibration-broken")),
        test_seed=derive_seed(SEED, name_key("calibration-broken"), name_key("student")),
        test_name="student",
    )
    fine = pvalue_calibration(
        registry["student"],
        scenarios["normal_equal_var"],
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key("calibration-fine")),
        test_seed=derive_seed(SEED, name_key("calibration-fine"), name_key("student")),
        test_name="student",
    )

    assert not broken.is_uniform(level=0.01), broken.ks_p_value
    assert fine.is_uniform(level=0.01), fine.ks_p_value
    assert broken.ks_statistic > 5 * fine.ks_statistic


@pytest.mark.slow
def test_mann_whitney_rejects_under_unequal_spread_because_it_asks_a_different_question(
    scenarios, registry
):
    """Not a bug in Mann-Whitney, and the project must not report it as one.

    With equal means but unequal variances the two distributions are genuinely different, so a
    test of stochastic equality is *right* to reject. Reading that as "the means differ" is the
    error, which is why the estimand caveat is stated everywhere this test appears.
    """
    result = _rate("mann_whitney", scenarios["normal_unequal_var_equal_n"], registry)
    assert result.rate > 0.05


@pytest.mark.slow
def test_no_contender_holds_alpha_under_extreme_skew(scenarios, registry):
    """The limit of the whole project, asserted so no notebook can quietly overclaim."""
    scenario = scenarios["skewed_extreme_unequal"]
    rates = {name: _rate(name, scenario, registry).rate for name in registry}
    assert all(rate > 0.05 for rate in rates.values()), rates
    # The assumption-free tests are still the least-bad option here, which is the honest claim.
    assert rates["permutation"] < rates["student"]

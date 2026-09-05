"""Power: does the test catch a real effect, and is its power worth anything?

Note what this file does *not* assert. The PRD predicted that under skew and small n the
Student t-test would lose power to the permutation test. Measured, that turned out to be false:
on lognormal data at n = 15 per group, Student, Welch and a permutation test on the same mean
difference all land within a point of each other, because they are all asking about the mean and
the mean is what skew ruins. The real result is better and is what is asserted here: the
permutation test's advantage under skew comes from being free to test a *different statistic*,
and swapping the mean for a trimmed mean recovers a large chunk of power at the same alpha.
"""

from __future__ import annotations

from functools import partial

import pytest

from src.evaluation import derive_seed, name_key
from src.evaluation.false_positive import false_positive_rate, is_calibrated
from src.evaluation.power import power, power_curve
from src.simulation.scenarios import make_scenario
from src.tests import trimmed_mean_difference
from src.tests.mann_whitney import mann_whitney_test
from src.tests.permutation import permutation_test
from src.tests.student import student_test
from src.tests.welch import welch_test
from tests.conftest import FAST_PERM, FAST_REPS, SEED, mc_tolerance

PERMUTATION_ON_THE_MEAN = partial(permutation_test, n_perm=FAST_PERM)
PERMUTATION_ON_A_TRIMMED_MEAN = partial(
    permutation_test,
    n_perm=FAST_PERM,
    statistic=trimmed_mean_difference,
    effect_fn=trimmed_mean_difference,
)


def _power(test_name, test_fn, scenario, *, tag=""):
    return power(
        test_fn,
        scenario,
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key(scenario.name + tag)),
        test_seed=derive_seed(SEED, name_key(scenario.name + tag), name_key(test_name)),
        test_name=test_name,
    )


def _fpr(test_name, test_fn, scenario, *, tag=""):
    return false_positive_rate(
        test_fn,
        scenario,
        n_reps=FAST_REPS,
        alpha=0.05,
        data_seed=derive_seed(SEED, name_key(scenario.name + tag)),
        test_seed=derive_seed(SEED, name_key(scenario.name + tag), name_key(test_name)),
        test_name=test_name,
    )


@pytest.mark.slow
def test_the_t_test_wins_on_its_home_ground(settings):
    """Clean, normal, equal-variance, balanced data: the t-test is the right tool.

    Asserted so the project cannot be accused of strawmanning it. Student is at least as
    powerful as the assumption-free alternatives here, by a margin too small to matter.
    """
    scenario = make_scenario(
        "home_ground", "normal", n_a=25, n_b=25, effect_d=0.8, settings=settings
    )
    student = _power("student", student_test, scenario)
    permutation = _power("permutation", PERMUTATION_ON_THE_MEAN, scenario)
    mwu = _power("mann_whitney", mann_whitney_test, scenario)

    assert student.rate >= permutation.rate - mc_tolerance(student.rate, FAST_REPS)
    assert student.rate >= mwu.rate - mc_tolerance(student.rate, FAST_REPS)
    assert student.rate > 0.5


@pytest.mark.slow
def test_under_skew_a_robust_permutation_test_beats_the_t_test(settings):
    """The honest version of the PRD's power claim, at matched false-positive rates.

    Both tests hold alpha on this data, so the power gap is a real gain and not the artifact of
    a broken test rejecting more often.
    """
    effect = make_scenario(
        "skew_power", "lognormal", n_a=15, n_b=15, effect_d=0.8, settings=settings
    )
    null = make_scenario("skew_power", "lognormal", n_a=15, n_b=15, settings=settings)

    student = _power("student", student_test, effect)
    robust = _power("permutation", PERMUTATION_ON_A_TRIMMED_MEAN, effect)

    assert is_calibrated(_fpr("student", student_test, null), n_se=4.0)
    assert is_calibrated(_fpr("permutation", PERMUTATION_ON_A_TRIMMED_MEAN, null), n_se=4.0)

    gap = robust.rate - student.rate
    assert gap > 0.05, (
        f"trimmed-mean permutation power {robust.rate:.3f} vs Student {student.rate:.3f}"
    )


@pytest.mark.slow
def test_permutation_on_the_mean_does_not_beat_the_t_test_under_skew(settings):
    """The claim this project does *not* make, pinned so nobody re-adds it.

    Same reference distribution problem, same statistic: essentially identical power. If this
    test starts failing, the narrative in notebook 05 needs rewriting, not the assertion.
    """
    scenario = make_scenario(
        "skew_mean", "lognormal", n_a=15, n_b=15, effect_d=0.8, settings=settings
    )
    student = _power("student", student_test, scenario)
    permutation = _power("permutation", PERMUTATION_ON_THE_MEAN, scenario)
    assert abs(permutation.rate - student.rate) < 0.05


@pytest.mark.slow
def test_students_extra_power_under_unequal_variance_is_bought_with_false_positives(
    scenarios, settings
):
    """The most important lesson in the file: power without alpha control is worthless.

    Student looks dramatically more powerful than Welch in the unbalanced unequal-variance
    scenario. It is not more powerful; it simply rejects more often, whether or not anything is
    there, and the same behaviour produced a false-positive rate several times alpha on the
    matching null.
    """
    base = scenarios["normal_unequal_var_unequal_n"]
    effect = base.with_effect(settings.grid.arena_effect_d)

    student_power = _power("student", student_test, effect)
    welch_power = _power("welch", welch_test, effect)
    student_fpr = _fpr("student", student_test, base)

    assert student_power.rate > welch_power.rate
    assert student_fpr.rate > 0.15
    # The "advantage" is an artifact: it disappears once the rejection rate under the null is
    # accounted for.
    assert student_power.rate - student_fpr.rate < welch_power.rate + 0.15


@pytest.mark.slow
def test_power_grows_with_sample_size(settings):
    """Notebook 05's message: a non-significant result may just be too little data."""
    scenarios = [
        make_scenario(f"n{n}", "normal", n_a=n, n_b=n, effect_d=0.5, settings=settings)
        for n in (10, 30, 80)
    ]
    frame = power_curve({"welch": welch_test}, scenarios, n_reps=FAST_REPS, alpha=0.05, seed=SEED)
    powers = frame.sort_values("n_a")["power"].tolist()
    assert powers == sorted(powers), powers
    assert powers[-1] > powers[0] + 0.3


@pytest.mark.slow
def test_cohens_d_is_not_comparable_across_distribution_shapes(settings):
    """A caution the notebooks have to state, because the measurement contradicts the intuition.

    The intuitive claim - "skew costs power, so you need more data" - is false at a fixed
    Cohen's d, and measurably so: on a lognormal with skewness 4, Welch has *more* power than on
    a normal population with the same d and the same n. The reason is that d divides by a
    standard deviation the long right tail has inflated, so the same d corresponds to a much
    larger separation of the bulk of the two distributions.

    What skew actually costs is validity, not power: see the calibration assertions in
    test_false_positive.py. Comparing power across the rows of the arena that use different
    population kinds is therefore not meaningful; comparing tests within a row is.
    """
    normal = make_scenario("n", "normal", n_a=30, n_b=30, effect_d=0.5, settings=settings)
    skewed = make_scenario("s", "lognormal", n_a=30, n_b=30, effect_d=0.5, settings=settings)
    assert skewed.population_a.skewness == pytest.approx(4.0)
    assert _power("welch", welch_test, skewed).rate > _power("welch", welch_test, normal).rate

"""The from-scratch tests must reproduce scipy exactly on clean data.

If these fail, every downstream finding in the project is an implementation bug rather than a
statistical one, so this file runs first in spirit: the Student t-test is only allowed to be
shown failing once it has been shown to be correct.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from scipy import stats

from src.evaluation.comparison import build_registry, check_assumptions, compare_groups
from src.simulation.populations import (
    ContaminatedNormalPopulation,
    HeavyTailedPopulation,
    LogNormalPopulation,
    NormalPopulation,
)
from src.tests import TestResult, mean_difference, welch_t_statistic
from src.tests.bayesian import bayesian_test
from src.tests.bootstrap import bootstrap_test
from src.tests.mann_whitney import mann_whitney_test
from src.tests.permutation import permutation_test
from src.tests.student import student_test
from src.tests.welch import welch_test

TOLERANCE = 1e-9
ALTERNATIVES = ["two-sided", "greater", "less"]


@pytest.fixture
def clean_samples() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20250905)
    return rng.normal(0.0, 1.0, 40), rng.normal(0.4, 1.0, 55)


# --------------------------------------------------------------------------------------------
# Agreement with scipy. Note the argument order: our convention is that the effect is b - a,
# so the matching scipy call is scipy(b, a), not scipy(a, b).
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_student_matches_scipy(clean_samples, alternative):
    a, b = clean_samples
    ours = student_test(a, b, alternative=alternative)
    theirs = stats.ttest_ind(b, a, equal_var=True, alternative=alternative)
    assert ours.statistic == pytest.approx(theirs.statistic, abs=TOLERANCE)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=TOLERANCE)
    assert ours.extra["df"] == pytest.approx(theirs.df, abs=TOLERANCE)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_welch_matches_scipy(clean_samples, alternative):
    a, b = clean_samples
    ours = welch_test(a, b, alternative=alternative)
    theirs = stats.ttest_ind(b, a, equal_var=False, alternative=alternative)
    assert ours.statistic == pytest.approx(theirs.statistic, abs=TOLERANCE)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=TOLERANCE)
    assert ours.extra["df"] == pytest.approx(theirs.df, abs=TOLERANCE)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mann_whitney_matches_scipy(clean_samples, alternative):
    a, b = clean_samples
    ours = mann_whitney_test(a, b, alternative=alternative)
    theirs = stats.mannwhitneyu(b, a, method="asymptotic", alternative=alternative)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=TOLERANCE)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mann_whitney_matches_scipy_with_ties(alternative):
    """Ties are where a hand-rolled rank test usually diverges from scipy."""
    rng = np.random.default_rng(7)
    a = np.round(rng.normal(0.0, 1.0, 30))
    b = np.round(rng.normal(0.6, 1.0, 30))
    assert len(np.unique(np.concatenate([a, b]))) < a.size + b.size, "expected ties"
    ours = mann_whitney_test(a, b, alternative=alternative)
    theirs = stats.mannwhitneyu(b, a, method="asymptotic", alternative=alternative)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=TOLERANCE)


def test_student_and_welch_agree_when_variances_and_sizes_are_equal(clean_samples):
    """The two t-tests coincide at equal n and equal variance. This is why the failure hides."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.0, 1.0, 40), rng.normal(0.0, 1.0, 40)
    assert student_test(a, b).p_value == pytest.approx(welch_test(a, b).p_value, rel=1e-2)


def test_welch_t_statistic_matches_the_welch_test(clean_samples):
    a, b = clean_samples
    assert float(welch_t_statistic(a, b)) == pytest.approx(welch_test(a, b).statistic, abs=1e-12)


# --------------------------------------------------------------------------------------------
# Contract shared by all six contenders.
# --------------------------------------------------------------------------------------------


def test_every_contender_returns_the_same_shape_of_result(clean_samples):
    a, b = clean_samples
    rng = np.random.default_rng(11)
    for name, test_fn in build_registry().items():
        result = test_fn(a, b, alpha=0.05, rng=rng)
        assert isinstance(result, TestResult)
        assert result.name == name
        assert result.alpha == 0.05
        assert isinstance(result.reject, bool | np.bool_)
        assert result.effect == pytest.approx(float(mean_difference(a, b)), rel=1e-6)
        if result.p_value is not None:
            assert 0.0 < result.p_value <= 1.0
            assert result.reject == (result.p_value <= 0.05)


def test_effect_sign_follows_the_b_minus_a_convention():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.0, 1.0, 50), rng.normal(2.0, 1.0, 50)
    for test_fn in build_registry().values():
        assert test_fn(a, b, rng=np.random.default_rng(1)).effect > 0
        assert test_fn(b, a, rng=np.random.default_rng(1)).effect < 0


def test_resampling_tests_never_report_a_zero_p_value():
    """The add-one correction: with B shuffles the smallest honest p-value is 1 / (B + 1)."""
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.0, 1.0, 40), rng.normal(50.0, 1.0, 40)  # absurdly separated
    perm = permutation_test(a, b, n_perm=199, rng=np.random.default_rng(1))
    boot = bootstrap_test(a, b, n_boot=199, null_shift=True, rng=np.random.default_rng(1))
    assert perm.p_value == pytest.approx(1.0 / 200.0)
    assert boot.p_value == pytest.approx(1.0 / 200.0)


def test_permutation_accepts_any_statistic(clean_samples):
    """ "Works for any statistic" is a claim the project makes; this is the claim."""
    from src.tests import median_difference, trimmed_mean_difference

    a, b = clean_samples
    for statistic in (mean_difference, median_difference, trimmed_mean_difference):
        result = permutation_test(
            a,
            b,
            statistic=statistic,
            effect_fn=statistic,
            n_perm=199,
            rng=np.random.default_rng(4),
        )
        assert 0.0 < result.p_value <= 1.0


def test_bootstrap_ci_contains_the_true_difference_on_skewed_data():
    """PRD success criterion: percentile CI coverage at the nominal level on skewed data."""
    population_a = LogNormalPopulation(mean=10.0, sd=10.0)
    population_b = LogNormalPopulation(mean=13.0, sd=10.0)
    rng = np.random.default_rng(20250905)
    covered = 0
    trials = 400
    for _ in range(trials):
        a = population_a.rvs(60, rng)
        b = population_b.rvs(60, rng)
        result = bootstrap_test(a, b, n_boot=299, rng=rng)
        assert result.ci is not None
        covered += int(result.ci[0] <= 3.0 <= result.ci[1])
    coverage = covered / trials
    # Nominal 0.95; the percentile interval is known to under-cover somewhat on skewed data at
    # this sample size, so the assertion is "close to nominal", not "at least nominal".
    assert 0.90 <= coverage <= 0.99, coverage


def test_bayesian_probability_tracks_the_true_effect_direction():
    rng = np.random.default_rng(9)
    a = rng.normal(0.0, 1.0, 60)
    higher = rng.normal(1.5, 1.0, 60)
    lower = rng.normal(-1.5, 1.0, 60)

    assert bayesian_test(a, higher, rng=np.random.default_rng(1)).extra["prob_effect"] > 0.99
    assert bayesian_test(a, lower, rng=np.random.default_rng(1)).extra["prob_effect"] < 0.01


def test_bayesian_probability_averages_to_a_half_under_the_null():
    """A single null draw can legitimately give P(b > a) = 0.9; the *average* must be 0.5.

    This is the honest calibration check for a contender that has no p-value.
    """
    data_rng = np.random.default_rng(20250905)
    test_rng = np.random.default_rng(1)
    probabilities = [
        bayesian_test(
            data_rng.normal(0.0, 1.0, 40),
            data_rng.normal(0.0, 1.0, 40),
            n_posterior=1000,
            rng=test_rng,
        ).extra["prob_effect"]
        for _ in range(300)
    ]
    assert float(np.mean(probabilities)) == pytest.approx(0.5, abs=0.05)


def test_bayesian_reports_no_p_value(clean_samples):
    a, b = clean_samples
    assert bayesian_test(a, b, rng=np.random.default_rng(1)).p_value is None


def test_tests_reject_degenerate_input():
    with pytest.raises(ValueError, match="at least 2"):
        student_test([1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        welch_test([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------------------------
# Populations: the ground truth has to actually be the truth.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "population",
    [
        NormalPopulation(mean=3.0, sd=2.0),
        LogNormalPopulation(mean=10.0, sd=10.0),
        HeavyTailedPopulation(mean=3.0, sd=2.0, df=5.0),
        ContaminatedNormalPopulation(mean=3.0, sd=2.0),
    ],
    ids=["normal", "lognormal", "heavy_tailed", "contaminated"],
)
def test_population_mean_sd_and_median_are_the_stated_ones(population):
    """Every population is parameterized by its true mean and sd; verify by brute force."""
    rng = np.random.default_rng(20250905)
    draws = population.rvs(400_000, rng)
    assert float(draws.mean()) == pytest.approx(population.mean, abs=0.05 * population.sd)
    assert float(draws.std(ddof=1)) == pytest.approx(population.sd, rel=0.05)
    assert float(np.median(draws)) == pytest.approx(population.median, abs=0.05 * population.sd)


def test_a_pure_location_shift_moves_the_mean_and_the_median_together():
    """Effect scenarios are built as location shifts, so the shape does not move with them."""
    from src.simulation.scenarios import make_scenario

    skewed = make_scenario("s", "lognormal", n_a=30, n_b=30, effect_d=0.8)
    assert skewed.true_mean_difference == pytest.approx(skewed.true_median_difference)
    assert skewed.population_a.skewness == pytest.approx(skewed.population_b.skewness)
    assert skewed.skewness == pytest.approx(4.0)


def test_unequal_spread_on_skewed_data_separates_the_mean_from_the_median():
    """Why every arena row carries both truths.

    With different spreads on a skewed population the two groups have equal means and
    *different* medians, so Mann-Whitney is right to reject while the mean difference is zero.
    Scoring that as a Mann-Whitney false positive would be the project's own error.
    """
    from src.simulation.scenarios import make_scenario

    scenario = make_scenario("s", "lognormal", n_a=100, n_b=15, sd_ratio=2.0)
    assert scenario.is_null
    assert scenario.true_mean_difference == 0.0
    assert scenario.true_median_difference < -1.0
    assert scenario.population_b.skewness > 3 * scenario.population_a.skewness


# --------------------------------------------------------------------------------------------
# compare_groups: the artifact notebook 06 ships.
# --------------------------------------------------------------------------------------------


def test_compare_groups_warns_when_the_t_test_assumptions_break():
    rng = np.random.default_rng(1)
    wide = rng.normal(0.0, 6.0, 40)
    narrow = rng.normal(0.0, 1.0, 40)

    # Student is warned about unequal spread; Welch is not, because Welch is the fix for it.
    with pytest.warns(UserWarning, match="unequal spread"):
        compare_groups(wide, narrow, test="student")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_groups(wide, narrow, test="welch")

    tiny_a, tiny_b = rng.normal(0.0, 1.0, 12), rng.normal(0.0, 1.0, 12)
    with pytest.warns(UserWarning, match="small sample"):
        compare_groups(tiny_a, tiny_b, test="welch")

    skewed = LogNormalPopulation(mean=10.0, sd=20.0).rvs(120, rng)
    with pytest.warns(UserWarning, match="skewed data"):
        compare_groups(skewed, skewed + 1.0, test="student")


def test_compare_groups_stays_quiet_on_clean_data_and_for_assumption_free_tests():
    rng = np.random.default_rng(6)
    a, b = rng.normal(0.0, 1.0, 200), rng.normal(0.0, 1.0, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_groups(a, b, test="welch")
        compare_groups(a, b, test="permutation", rng=np.random.default_rng(1))


def test_compare_groups_defaults_to_the_assumption_free_test():
    rng = np.random.default_rng(8)
    a, b = rng.normal(0.0, 1.0, 30), rng.normal(0.0, 1.0, 30)
    assert compare_groups(a, b, rng=np.random.default_rng(1)).name == "permutation"


def test_compare_groups_rejects_an_unknown_test():
    with pytest.raises(ValueError, match="unknown test"):
        compare_groups([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], test="anova")


def test_the_arena_cache_is_keyed_on_content_not_on_a_filename(tmp_path, monkeypatch):
    """A stale cache must not be able to masquerade as a fresh result."""
    import src.evaluation.comparison as comparison
    from src.evaluation.comparison import arena_cache_key, run_arena
    from src.simulation.scenarios import make_scenario

    monkeypatch.setattr(comparison, "CACHE_DIR", tmp_path)
    grid = [make_scenario("cache_probe", "normal", n_a=10, n_b=10)]
    tests = {"welch": welch_test}

    first = run_arena(grid, tests=tests, n_reps=20, alpha=0.05, seed=1)
    assert len(list(tmp_path.glob("arena_*.csv"))) == 1
    second = run_arena(grid, tests=tests, n_reps=20, alpha=0.05, seed=1)
    # Bit-identical, not merely close: a cached arena must be interchangeable with a fresh one.
    assert second.equals(first)
    assert second.dtypes.equals(first.dtypes)

    # Anything that changes the result changes the key.
    settings = __import__("src.config", fromlist=["get_settings"]).get_settings()
    base = dict(n_reps=20, alpha=0.05, seed=1, settings=settings)
    key = arena_cache_key(grid, ["welch"], **base)
    assert arena_cache_key(grid, ["welch"], **{**base, "seed": 2}) != key
    assert arena_cache_key(grid, ["welch"], **{**base, "n_reps": 40}) != key
    assert arena_cache_key(grid, ["welch"], **{**base, "alpha": 0.01}) != key
    assert arena_cache_key(grid, ["welch", "student"], **base) != key
    assert arena_cache_key([grid[0].with_effect(0.5)], ["welch"], **base) != key


def test_save_results_writes_strict_json(tmp_path, monkeypatch):
    """The notebooks' numbers must land on disk as valid, readable JSON.

    NaN is the case that matters: the arena legitimately carries it in ``target`` for power
    rows, and json.dumps would otherwise emit a bare NaN, which is not valid JSON.
    """
    import json

    import pandas as pd

    import src.config as config_module
    from src.config import save_results

    monkeypatch.setattr(config_module, "RESULTS_DIR", tmp_path)
    payload = {
        "scalar": np.float64(1.5),
        "not_a_number": float("nan"),
        "array": np.arange(3),
        "frame": pd.DataFrame({"a": [1, 2], "b": [float("nan"), 0.5]}),
        "nested": {"tuple": (1, 2), "path": tmp_path},
        "claims": ["a sentence"],
    }
    path = save_results("probe", payload)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert "NaN" not in text
    restored = json.loads(text)  # strict: would raise on NaN/Infinity
    assert restored["scalar"] == 1.5
    assert restored["not_a_number"] is None
    assert restored["array"] == [0, 1, 2]
    assert restored["frame"] == [{"a": 1, "b": None}, {"a": 2, "b": 0.5}]
    assert restored["nested"]["tuple"] == [1, 2]


def test_check_assumptions_flags_the_conditions_it_should():
    rng = np.random.default_rng(12)
    clean = check_assumptions(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
    assert clean.clean, clean.messages

    skewed = LogNormalPopulation(mean=10.0, sd=20.0).rvs(500, rng)
    dirty = check_assumptions(skewed, rng.normal(10, 1, 500))
    assert any("skewed" in message for message in dirty.messages)
    assert any("unequal spread" in message for message in dirty.messages)

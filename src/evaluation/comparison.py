"""The arena: six tests across the scenario grid, scored on the same simulated data.

The output is always a tidy long frame, one row per ``(scenario, test)``. Wide tables are for
display only; do not compute on them.

What the arena is careful about
-------------------------------
* Every test sees **identical** samples within a scenario (see :mod:`src.evaluation`), so a
  head-to-head difference is about the tests.
* Every rate is reported with its Monte Carlo standard error.
* The true mean difference *and* the true median difference travel with every row, because
  Mann-Whitney is not estimating the first of those.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import Generator
from scipy import stats

from src.config import CACHE_DIR, Settings, get_settings
from src.evaluation import RateResult, TestFn, derive_seed, name_key, run_replications
from src.simulation.scenarios import Scenario, default_scenarios
from src.tests import TestResult, as_samples
from src.tests.bayesian import bayesian_test
from src.tests.bootstrap import bootstrap_test
from src.tests.mann_whitney import mann_whitney_test
from src.tests.permutation import permutation_test
from src.tests.student import student_test
from src.tests.welch import welch_test

ARENA_TESTS: tuple[str, ...] = (
    "student",
    "welch",
    "mann_whitney",
    "permutation",
    "bootstrap",
    "bayesian",
)

ARENA_COLUMNS: tuple[str, ...] = (
    "scenario",
    "kind",
    "test",
    "metric",
    "reject_rate",
    "mc_se",
    "target",
    "n_a",
    "n_b",
    "alpha",
    "effect_d",
    "sd_ratio",
    "true_mean_difference",
    "true_median_difference",
    "skewness",
    "reps",
    "data_seed",
    "test_seed",
)

ARENA_DTYPES: dict[str, str] = {
    # The four label columns are deliberately absent: pandas' inferred string dtype differs
    # between 2.x (object) and 3.x (str), and declaring either one would make the cached frame
    # mismatch the computed one on the other version. Inference matches on both.
    "reject_rate": "float64",
    "mc_se": "float64",
    "target": "float64",
    "n_a": "int64",
    "n_b": "int64",
    "alpha": "float64",
    "effect_d": "float64",
    "sd_ratio": "float64",
    "true_mean_difference": "float64",
    "true_median_difference": "float64",
    "skewness": "float64",
    "reps": "int64",
    "data_seed": "int64",
    "test_seed": "int64",
}
"""Declared on read so a cached arena comes back with the dtypes it was written with.

Without this, a column that happens to hold only whole numbers (``sd_ratio`` of 1.0, a
``skewness`` of 0.0) is inferred as int64 on the way back in, and the cached frame stops being
interchangeable with a freshly computed one.
"""


def build_registry(settings: Settings | None = None) -> dict[str, TestFn]:
    """The six contenders, with their resampling budgets bound from the active profile.

    ``bootstrap`` is registered with ``null_shift=True``: the arena scores decisions and
    :mod:`src.evaluation.calibration` needs p-values, so the bootstrap joins on the same footing
    as the other five. Its CI-inversion mode (the default of
    :func:`~src.tests.bootstrap.bootstrap_test`) is the one to use interactively.
    """
    resolved = settings or get_settings()
    reps = resolved.reps
    return {
        "student": student_test,
        "welch": welch_test,
        "mann_whitney": mann_whitney_test,
        "permutation": partial(permutation_test, n_perm=reps.n_perm),
        "bootstrap": partial(bootstrap_test, n_boot=reps.n_boot, null_shift=True),
        "bayesian": partial(bayesian_test, n_posterior=reps.n_posterior),
    }


def score(
    test_fn: TestFn,
    scenario: Scenario,
    *,
    n_reps: int,
    alpha: float,
    data_seed: int,
    test_seed: int,
    test_name: str,
) -> RateResult:
    """Score one ``(scenario, test)`` cell, choosing the metric from the scenario's truth."""
    reps = run_replications(
        test_fn,
        scenario,
        n_reps=n_reps,
        alpha=alpha,
        data_seed=data_seed,
        test_seed=test_seed,
        test_name=test_name,
    )
    return RateResult(
        test=test_name,
        scenario=scenario.name,
        metric="fpr" if scenario.is_null else "power",
        rate=reps.reject_rate,
        n_reps=reps.n_reps,
        alpha=alpha,
    )


def arena_cache_key(
    grid: Sequence[Scenario],
    test_names: Sequence[str],
    *,
    n_reps: int,
    alpha: float,
    seed: int,
    settings: Settings,
) -> str:
    """Hash of everything that determines the arena's output.

    Keyed on the *content* of the run, not on a filename: change a scenario, a replication
    count, the alpha, the seed or the resampling budgets and the key changes, so a stale cache
    cannot masquerade as a fresh result.
    """
    payload = {
        "scenarios": [asdict(scenario) for scenario in grid],
        "tests": list(test_names),
        "n_reps": n_reps,
        "alpha": alpha,
        "seed": seed,
        "budgets": settings.reps.model_dump(),
        "columns": list(ARENA_COLUMNS),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_arena(
    scenarios: Sequence[Scenario] | None = None,
    *,
    tests: dict[str, TestFn] | None = None,
    settings: Settings | None = None,
    n_reps: int | None = None,
    alpha: float | None = None,
    seed: int | None = None,
    progress: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """Run every test against every scenario and return the tidy scoreboard.

    ``target`` is what the row *should* be: alpha for a null row (an exact target), and NaN for a
    power row (higher is better, but there is no single right answer).

    The full grid at 10 000 replications takes about nine and a half minutes on a 2-CPU
    Codespace, so results are cached under ``outputs/cache/`` against a hash of the run's
    content (see :func:`arena_cache_key`). Pass ``cache=False`` to force a recompute, and pass
    it when you supply a custom ``tests`` registry, whose resampling budgets the key cannot see.
    """
    resolved = settings or get_settings()
    grid = list(scenarios) if scenarios is not None else default_scenarios(resolved)
    registry = tests if tests is not None else build_registry(resolved)
    reps_count = n_reps if n_reps is not None else resolved.reps.n_reps
    level = alpha if alpha is not None else resolved.alpha
    master_seed = seed if seed is not None else resolved.seed

    cache_path = CACHE_DIR / (
        "arena_"
        + arena_cache_key(
            grid,
            list(registry),
            n_reps=reps_count,
            alpha=level,
            seed=master_seed,
            settings=resolved,
        )
        + ".csv"
    )
    if cache and cache_path.exists():
        if progress:
            print(f"  loaded from cache: {cache_path.name}")  # noqa: T201 - notebook progress
        # float_precision="round_trip" is required: pandas' default C float parser is fast and
        # inexact, and silently returns values a few ulps off what was written.
        cached = pd.read_csv(cache_path, float_precision="round_trip")
        return cached.astype(ARENA_DTYPES)

    rows: list[dict[str, object]] = []
    for scenario in grid:
        data_seed = derive_seed(master_seed, name_key(scenario.name))
        for test_name, test_fn in registry.items():
            if progress:
                print(f"  {scenario.name:<38} {test_name}")  # noqa: T201 - notebook progress
            test_seed = derive_seed(master_seed, name_key(scenario.name), name_key(test_name))
            result = score(
                test_fn,
                scenario,
                n_reps=reps_count,
                alpha=level,
                data_seed=data_seed,
                test_seed=test_seed,
                test_name=test_name,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "kind": scenario.kind,
                    "test": test_name,
                    "metric": result.metric,
                    "reject_rate": result.rate,
                    "mc_se": result.mc_se,
                    "target": level if scenario.is_null else float("nan"),
                    "n_a": scenario.n_a,
                    "n_b": scenario.n_b,
                    "alpha": level,
                    "effect_d": scenario.effect_d,
                    "sd_ratio": scenario.sd_ratio,
                    "true_mean_difference": scenario.true_mean_difference,
                    "true_median_difference": scenario.true_median_difference,
                    "skewness": scenario.skewness,
                    "reps": reps_count,
                    "data_seed": data_seed,
                    "test_seed": test_seed,
                }
            )

    arena = pd.DataFrame(rows, columns=list(ARENA_COLUMNS))
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # %.17g round-trips float64 exactly; pandas' default formatting silently drops the last
        # couple of digits, which would make a cached arena differ from a recomputed one.
        arena.to_csv(cache_path, index=False, float_format="%.17g")
    return arena


def scoreboard(arena: pd.DataFrame, metric: str = "fpr") -> pd.DataFrame:
    """Pivot the tidy frame to scenarios x tests. Display only - do not compute on this."""
    subset = arena[arena["metric"] == metric]
    wide = subset.pivot(index="scenario", columns="test", values="reject_rate")
    ordered = [name for name in ARENA_TESTS if name in wide.columns]
    return wide[ordered]


def verdict(arena: pd.DataFrame, *, n_se: float = 3.0) -> pd.DataFrame:
    """Per-test summary: worst null miss, mean power, and how many null cells were off.

    ``max_abs_deviation_se`` is the largest distance from alpha across the null scenarios, in
    Monte Carlo standard errors. A test that holds alpha everywhere keeps this small.

    Inflated and conservative cells are counted separately, because they are not the same
    failure. An inflated cell invents discoveries that are not there; a conservative one only
    spends power. Both are miscalibration, one is much more expensive than the other, and a
    single "cells off" count would hide the difference.
    """
    nulls = arena[arena["metric"] == "fpr"].copy()
    nulls["deviation_se"] = (nulls["reject_rate"] - nulls["alpha"]) / nulls["mc_se"]
    powers = arena[arena["metric"] == "power"]

    summary = pd.DataFrame(
        {
            "worst_fpr": nulls.groupby("test")["reject_rate"].max(),
            "max_abs_deviation_se": nulls.groupby("test")["deviation_se"].apply(
                lambda s: s.abs().max()
            ),
            "n_inflated": nulls.groupby("test")["deviation_se"].apply(
                lambda s: int((s > n_se).sum())
            ),
            "n_conservative": nulls.groupby("test")["deviation_se"].apply(
                lambda s: int((s < -n_se).sum())
            ),
            "mean_power": powers.groupby("test")["reject_rate"].mean(),
        }
    )
    ordered = [name for name in ARENA_TESTS if name in summary.index]
    return summary.loc[ordered]


# --------------------------------------------------------------------------------------------
# The practical artifact: run a test and say out loud when its assumptions do not hold.
# --------------------------------------------------------------------------------------------

SMALL_SAMPLE = 30
"""Below this, the central limit theorem is not doing the work people assume it is."""


@dataclass(frozen=True)
class AssumptionReport:
    """What the data says about the assumptions the parametric tests make."""

    n_a: int
    n_b: int
    skewness_a: float
    skewness_b: float
    sd_ratio: float
    normality_p_a: float
    normality_p_b: float
    outlier_fraction: float
    messages: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.messages


def check_assumptions(
    a: npt.ArrayLike, b: npt.ArrayLike, *, normality_level: float = 0.05
) -> AssumptionReport:
    """Flag the conditions that make a parametric test untrustworthy.

    The checks are deliberately blunt, and the direction of their error matters. A normality
    test on 15 observations has almost no power, so a passing p-value is not evidence of
    normality - which is exactly why the sample-size check fires independently. Absence of a
    warning is not a licence; it is the absence of an obvious problem.
    """
    sample_a, sample_b = as_samples(a, b)
    n_a, n_b = sample_a.size, sample_b.size
    sd_a, sd_b = float(sample_a.std(ddof=1)), float(sample_b.std(ddof=1))
    ratio = max(sd_a, sd_b) / min(sd_a, sd_b) if min(sd_a, sd_b) > 0 else float("inf")
    skew_a, skew_b = float(stats.skew(sample_a)), float(stats.skew(sample_b))

    def normality_p(sample: npt.NDArray[np.float64]) -> float:
        if sample.size < 3:
            return float("nan")
        return float(stats.shapiro(sample).pvalue)

    p_a, p_b = normality_p(sample_a), normality_p(sample_b)

    def outlier_fraction(sample: npt.NDArray[np.float64]) -> float:
        """Fraction beyond 3 IQRs of the sample's own quartiles.

        Computed per group on purpose. Pooling the two groups first would let a legitimate
        difference in spread masquerade as outliers: the wider group's ordinary tail falls
        outside the pooled fences, and the report would blame contamination for what is
        really unequal variance - which has a different fix.
        """
        q1, q3 = np.percentile(sample, [25, 75])
        iqr = float(q3 - q1)
        if iqr <= 0:
            return 0.0
        return float(np.mean((sample < q1 - 3 * iqr) | (sample > q3 + 3 * iqr)))

    outliers = max(outlier_fraction(sample_a), outlier_fraction(sample_b))

    messages: list[str] = []
    if min(n_a, n_b) < SMALL_SAMPLE:
        messages.append(
            f"small sample (n = {n_a}, {n_b}): the t-test's normal approximation is doing work "
            "it cannot support, and a normality test here has almost no power"
        )
    if ratio > 2.0:
        messages.append(
            f"unequal spread (sd ratio {ratio:.2f}): never use the Student t-test here; "
            "Welch, permutation or bootstrap"
        )
    if max(abs(skew_a), abs(skew_b)) > 1.0:
        messages.append(
            f"skewed data (skewness {skew_a:.2f}, {skew_b:.2f}): the mean may be the wrong "
            "summary; consider a permutation test on a median or trimmed mean"
        )
    if min(p_a, p_b) < normality_level:
        messages.append(
            f"normality rejected (Shapiro-Wilk p = {min(p_a, p_b):.4g}): prefer an "
            "assumption-free test"
        )
    if outliers > 0.0:
        messages.append(
            f"{outliers:.1%} of observations are far outliers: they inflate the variance "
            "estimate and drain the t-test's power"
        )

    return AssumptionReport(
        n_a=n_a,
        n_b=n_b,
        skewness_a=skew_a,
        skewness_b=skew_b,
        sd_ratio=ratio,
        normality_p_a=p_a,
        normality_p_b=p_b,
        outlier_fraction=outliers,
        messages=messages,
    )


def compare_groups(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    test: str = "permutation",
    alpha: float = 0.05,
    rng: Generator | None = None,
    warn: bool = True,
    settings: Settings | None = None,
) -> TestResult:
    """Compare two groups with the named test, warning when the data breaks its assumptions.

    The default is the permutation test, not a t-test, for the reason the whole project exists:
    when in doubt the assumption-free option costs only compute.

    Warnings are raised for Student and Welch, the two contenders whose validity depends on the
    shape of the data. ``warn=False`` silences them; the report is still available from
    :func:`check_assumptions`.
    """
    registry = build_registry(settings)
    if test not in registry:
        known = ", ".join(registry)
        raise ValueError(f"unknown test {test!r}; choose one of: {known}")

    if warn and test in {"student", "welch"}:
        report = check_assumptions(a, b)
        for message in report.messages:
            if test == "welch" and message.startswith("unequal spread"):
                continue  # Welch is the fix for that one
            warnings.warn(f"{test} t-test: {message}", UserWarning, stacklevel=2)

    return registry[test](a, b, alpha=alpha, rng=rng)

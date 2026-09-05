"""Power under a real effect: does the test catch what is actually there?

The second way a test fails. A non-significant result is not "no effect"; it may be "not enough
data", or a test whose assumptions failed in the direction that costs power rather than the one
that costs honesty.

Power is only interpretable next to the false-positive rate. A test that rejects constantly has
high power and no value, which is why :mod:`src.evaluation.comparison` always reports both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.evaluation import RateResult, TestFn, derive_seed, name_key, run_replications
from src.simulation.scenarios import Scenario
from src.tests import FloatArray


def power(
    test_fn: TestFn,
    scenario: Scenario,
    *,
    n_reps: int,
    alpha: float = 0.05,
    data_seed: int = 0,
    test_seed: int = 1,
    test_name: str | None = None,
) -> RateResult:
    """Measure the rejection rate of ``test_fn`` on a scenario with a true effect."""
    if scenario.is_null:
        raise ValueError(
            f"scenario {scenario.name!r} has no true effect; power is only defined "
            "when the alternative is true"
        )
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
        test=reps.test,
        scenario=reps.scenario,
        metric="power",
        rate=reps.reject_rate,
        n_reps=reps.n_reps,
        alpha=alpha,
    )


@dataclass(frozen=True)
class FamilyWiseResult:
    """What happens when you run ``k`` tests and none of them has anything to find."""

    k: int
    trials: int
    alpha: float
    any_significant_rate: float
    mean_significant: float
    p_values: FloatArray

    @property
    def expected_rate(self) -> float:
        """``1 - (1 - alpha)^k``, the rate if the k tests were independent."""
        return 1.0 - (1.0 - self.alpha) ** self.k


def family_wise_error_rate(
    test_fn: TestFn,
    scenario: Scenario,
    *,
    k: int,
    trials: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> FamilyWiseResult:
    """Run ``k`` independent null comparisons ``trials`` times and count the false alarms.

    The multiple-comparisons trap, measured rather than asserted: each individual test is
    perfectly calibrated at ``alpha``, and the chance that *at least one* of ``k`` of them fires
    is nevertheless ``1 - (1 - alpha)^k``. Testing twenty metrics on an experiment that changed
    nothing produces a "significant" one about two times in three.
    """
    if not scenario.is_null:
        raise ValueError("the multiple-comparisons trap is about tests with nothing to find")
    reps = run_replications(
        test_fn,
        scenario,
        n_reps=k * trials,
        alpha=alpha,
        data_seed=derive_seed(seed, name_key(scenario.name), k),
        test_seed=derive_seed(seed, name_key(scenario.name), k, 1),
        test_name="family",
    )
    hits = reps.rejects.reshape(trials, k)
    return FamilyWiseResult(
        k=k,
        trials=trials,
        alpha=alpha,
        any_significant_rate=float((hits.sum(axis=1) > 0).mean()),
        mean_significant=float(hits.sum(axis=1).mean()),
        p_values=reps.p_values[:k],
    )


def power_curve(
    tests: dict[str, TestFn],
    scenarios: Sequence[Scenario],
    *,
    n_reps: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Power for every ``(scenario, test)`` pair, as a tidy long frame.

    Every test sees identical data within a scenario, so differences between rows of the same
    scenario are differences between tests, not between draws.
    """
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        data_seed = derive_seed(seed, name_key(scenario.name))
        for test_name, test_fn in tests.items():
            test_seed = derive_seed(seed, name_key(scenario.name), name_key(test_name))
            result = power(
                test_fn,
                scenario,
                n_reps=n_reps,
                alpha=alpha,
                data_seed=data_seed,
                test_seed=test_seed,
                test_name=test_name,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "kind": scenario.kind,
                    "test": test_name,
                    "n_a": scenario.n_a,
                    "n_b": scenario.n_b,
                    "effect_d": scenario.effect_d,
                    "sd_ratio": scenario.sd_ratio,
                    "reps": n_reps,
                    "power": result.rate,
                    "mc_se": result.mc_se,
                }
            )
    return pd.DataFrame(rows)

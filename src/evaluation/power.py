"""Power under a real effect: does the test catch what is actually there?

The second way a test fails. A non-significant result is not "no effect"; it may be "not enough
data", or a test whose assumptions failed in the direction that costs power rather than the one
that costs honesty.

Power is only interpretable next to the false-positive rate. A test that rejects constantly has
high power and no value, which is why :mod:`src.evaluation.comparison` always reports both.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.evaluation import RateResult, TestFn, derive_seed, name_key, run_replications
from src.simulation.scenarios import Scenario


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

"""Type I error under the null: does the test deliver the alpha it promises?

A test that rejects 5% of the time when nothing is going on is doing its job. A test that
rejects 15% of the time is manufacturing discoveries, and every one of those "significant"
results is a phantom. This is the first of the two ways a test can fail.
"""

from __future__ import annotations

from src.evaluation import RateResult, TestFn, run_replications
from src.simulation.scenarios import Scenario


def false_positive_rate(
    test_fn: TestFn,
    scenario: Scenario,
    *,
    n_reps: int,
    alpha: float = 0.05,
    data_seed: int = 0,
    test_seed: int = 1,
    test_name: str | None = None,
) -> RateResult:
    """Measure the rejection rate of ``test_fn`` on a scenario where the null is true."""
    if not scenario.is_null:
        raise ValueError(
            f"scenario {scenario.name!r} has a true effect of {scenario.delta}; "
            "a false-positive rate is only defined under the null"
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
        metric="fpr",
        rate=reps.reject_rate,
        n_reps=reps.n_reps,
        alpha=alpha,
    )


def is_inflated(result: RateResult, *, n_se: float = 3.0) -> bool:
    """Is the false-positive rate above alpha by more than ``n_se`` Monte Carlo SEs?

    The Monte Carlo margin is the point: "0.054 versus 0.05" is not inflation at any replication
    count this project can afford.
    """
    return result.deviation_in_se(result.alpha) > n_se


def is_calibrated(result: RateResult, *, n_se: float = 3.0) -> bool:
    """Is the false-positive rate within ``n_se`` Monte Carlo SEs of alpha, in either direction?"""
    return abs(result.deviation_in_se(result.alpha)) <= n_se

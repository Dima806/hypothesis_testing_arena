"""Interactive companion to the notebooks.

Four things you cannot get from a static figure: break the assumptions yourself and watch the
false-positive rate move, see all six tests disagree on the same data, watch p-values fail to be
uniform, and trade sample size against power.

Everything here calls the same functions the notebooks call. Replication counts are deliberately
small so a slider stays responsive; the published numbers come from ``make notebooks``, not from
this app, and the app says so wherever it shows a rate.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.evaluation import derive_seed, name_key  # noqa: E402
from src.evaluation.calibration import pvalue_calibration  # noqa: E402
from src.evaluation.false_positive import false_positive_rate  # noqa: E402
from src.evaluation.power import power  # noqa: E402
from src.simulation.scenarios import Scenario, make_scenario  # noqa: E402
from src.tests.bayesian import bayesian_test  # noqa: E402
from src.tests.bootstrap import bootstrap_test  # noqa: E402
from src.tests.mann_whitney import mann_whitney_test  # noqa: E402
from src.tests.permutation import permutation_null_distribution, permutation_test  # noqa: E402
from src.tests.student import student_test  # noqa: E402
from src.tests.welch import welch_test  # noqa: E402
from src.visualisation import (  # noqa: E402
    label,
    plot_null_distribution,
    plot_pvalue_calibration,
    plot_samples,
)

st.set_page_config(page_title="Hypothesis Testing Arena", page_icon="⚖️", layout="wide")

LIVE_REPS = 400
LIVE_PERM = 199
KINDS = {
    "normal (the t-test's home)": "normal",
    "right-skewed (revenue, durations)": "lognormal",
    "heavy-tailed (outliers)": "heavy_tailed",
    "contaminated (10% wild values)": "contaminated",
}


def registry(n_perm: int = LIVE_PERM, n_posterior: int = 1000) -> dict:
    return {
        "student": student_test,
        "welch": welch_test,
        "mann_whitney": mann_whitney_test,
        "permutation": partial(permutation_test, n_perm=n_perm),
        "bootstrap": partial(bootstrap_test, n_boot=n_perm, null_shift=True),
        "bayesian": partial(bayesian_test, n_posterior=n_posterior),
    }


def scenario_controls(key: str, *, effect: bool = False) -> Scenario:
    """Shared sidebar-style controls; returns the scenario they describe."""
    settings = get_settings()
    columns = st.columns(5 if effect else 4)
    kind_label = columns[0].selectbox("population", list(KINDS), key=f"{key}-kind")
    n_a = columns[1].slider("n (group a)", 5, 200, 100, key=f"{key}-na")
    n_b = columns[2].slider("n (group b)", 5, 200, 15, key=f"{key}-nb")
    sd_ratio = columns[3].select_slider(
        "sd ratio (b / a)", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=4.0, key=f"{key}-ratio"
    )
    effect_d = (
        columns[4].slider("true effect (Cohen's d)", 0.0, 1.5, 0.5, 0.1, key=f"{key}-d")
        if effect
        else 0.0
    )
    return make_scenario(
        key,
        KINDS[kind_label],
        n_a=n_a,
        n_b=n_b,
        effect_d=float(effect_d),
        sd_ratio=float(sd_ratio),
        settings=settings,
    )


@st.cache_data(show_spinner=False)
def cached_rates(scenario_key: tuple, *, n_reps: int, alpha: float, metric: str) -> pd.DataFrame:
    """Rejection rate per test for a scenario, keyed by its hashable description."""
    scenario = Scenario(*scenario_key)
    rows = []
    for name, test_fn in registry().items():
        measure = false_positive_rate if metric == "fpr" else power
        result = measure(
            test_fn,
            scenario,
            n_reps=n_reps,
            alpha=alpha,
            data_seed=derive_seed(20250905, name_key(scenario.name)),
            test_seed=derive_seed(20250905, name_key(scenario.name), name_key(name)),
            test_name=name,
        )
        rows.append(
            {
                "test": label(name),
                "rate": result.rate,
                "mc_se": result.mc_se,
                "±2 SE": f"{result.rate:.3f} ± {2 * result.mc_se:.3f}",
            }
        )
    return pd.DataFrame(rows)


def as_key(scenario: Scenario) -> tuple:
    return (
        scenario.name,
        scenario.kind,
        scenario.n_a,
        scenario.n_b,
        scenario.mean_a,
        scenario.sd_a,
        scenario.effect_d,
        scenario.sd_ratio,
        scenario.df,
        scenario.contamination,
        scenario.outlier_sd_mult,
    )


st.title("⚖️ The t-test you were taught assumes a bell curve your data does not have")
st.caption(
    "Six ways to answer 'are these two groups different', judged by simulation against a known "
    f"truth. Rates here use {LIVE_REPS} replications so the sliders stay responsive — read them "
    "with their Monte Carlo error, and take the published numbers from the notebooks."
)

breaker, race, calibration_tab, power_tab = st.tabs(
    ["1 · Assumption breaker", "2 · Test race", "3 · Null calibration", "4 · Power explorer"]
)

# --------------------------------------------------------------------------------------------
with breaker:
    st.subheader("Break the assumptions and watch the false-positive rate climb")
    st.markdown(
        "Both groups are drawn from populations with **exactly the same mean**, so every "
        "rejection below is a phantom discovery. A calibrated test rejects 5% of the time. "
        "Try `sd ratio = 4` with `n(a) = 100` and `n(b) = 15`."
    )
    scenario = scenario_controls("breaker")
    alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="breaker-alpha")

    with st.spinner(f"running {LIVE_REPS} simulations per test…"):
        rates = cached_rates(as_key(scenario), n_reps=LIVE_REPS, alpha=alpha, metric="fpr")

    chart = rates.set_index("test")[["rate"]]
    st.bar_chart(chart, height=280)
    worst = rates.loc[rates["rate"].idxmax()]
    if worst["rate"] > alpha + 3 * worst["mc_se"]:
        st.error(
            f"**{worst['test']}** rejected {worst['rate']:.1%} of the time when nothing was "
            f"going on — {worst['rate'] / alpha:.1f}× the {alpha:.0%} it promises."
        )
    else:
        st.success("Every test is holding its false-positive rate on this data.")
    st.dataframe(rates[["test", "±2 SE"]], hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------
with race:
    st.subheader("One dataset, six verdicts")
    scenario = scenario_controls("race", effect=True)
    seed = st.number_input("draw", min_value=0, max_value=9999, value=0, key="race-seed")

    sample_a, sample_b = scenario.draw(np.random.default_rng(int(seed)))
    left, right = st.columns([3, 4])
    with left:
        st.pyplot(plot_samples(sample_a, sample_b, title="The data you actually have").figure)
        st.caption(
            f"true mean difference **{scenario.true_mean_difference:.2f}**, "
            f"true median difference **{scenario.true_median_difference:.2f}**, "
            f"skewness **{scenario.skewness:.1f}**"
        )

    rows = []
    for name, test_fn in registry().items():
        result = test_fn(sample_a, sample_b, alpha=0.05, rng=np.random.default_rng(1))
        rows.append(
            {
                "test": label(name),
                "verdict": "significant" if result.reject else "not significant",
                "p-value": "—" if result.p_value is None else f"{result.p_value:.4f}",
                "effect (b − a)": f"{result.effect:.3f}",
                "95% interval": "—"
                if result.ci is None
                else f"[{result.ci[0]:.2f}, {result.ci[1]:.2f}]",
                "note": "P(b > a) = {:.3f}".format(result.extra["prob_effect"])
                if "prob_effect" in result.extra
                else ("tests stochastic dominance, not means" if name == "mann_whitney" else ""),
            }
        )
    verdicts = pd.DataFrame(rows)
    with right:
        st.dataframe(verdicts, hide_index=True, width="stretch")
        if verdicts["verdict"].nunique() > 1:
            st.warning(
                "**The six tests disagree on this data.** That is the whole point: the answer "
                "you report depends on a choice most people never make consciously."
            )
        else:
            st.info("All six agree here. Change the sd ratio or shrink a group to break them.")

    st.markdown("**The permutation null, built by shuffling the labels of this exact dataset**")
    null = permutation_null_distribution(
        sample_a, sample_b, n_perm=2000, rng=np.random.default_rng(1)
    )
    observed = permutation_test(sample_a, sample_b, n_perm=999, rng=np.random.default_rng(1))
    st.pyplot(
        plot_null_distribution(null, observed.statistic or 0.0, p_value=observed.p_value).figure
    )

# --------------------------------------------------------------------------------------------
with calibration_tab:
    st.subheader("Under the null, p-values must be uniform")
    st.markdown(
        "A valid test rejects at exactly α *for every* α, which is the same statement as "
        "'the p-values are uniform'. A curve above the diagonal means too many small p-values."
    )
    scenario = scenario_controls("calib")
    chosen = st.multiselect(
        "tests (the Bayesian contender has no p-value and cannot appear here)",
        ["student", "welch", "mann_whitney", "permutation", "bootstrap"],
        default=["student", "welch", "permutation"],
        key="calib-tests",
    )
    if chosen:
        with st.spinner(f"running {LIVE_REPS} simulations per test…"):
            results = [
                pvalue_calibration(
                    registry()[name],
                    scenario,
                    n_reps=LIVE_REPS,
                    alpha=0.05,
                    data_seed=derive_seed(20250905, name_key(scenario.name)),
                    test_seed=derive_seed(20250905, name_key(scenario.name), name_key(name)),
                    test_name=name,
                )
                for name in chosen
            ]
        st.pyplot(plot_pvalue_calibration(results).figure)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "test": label(r.test),
                        "rejection rate at α = 0.05": f"{r.rejection_rate:.3f}",
                        "KS distance from uniform": f"{r.ks_statistic:.3f}",
                        "uniform?": "yes" if r.is_uniform() else "no — invalid here",
                    }
                    for r in results
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    plt.close("all")

# --------------------------------------------------------------------------------------------
with power_tab:
    st.subheader("How often does each test catch a real effect?")
    st.markdown(
        "Power is only meaningful for a test that holds its false-positive rate. Check tab 1 "
        "for the same settings before believing a number here — a test that rejects constantly "
        "has excellent power and no value."
    )
    scenario = scenario_controls("power", effect=True)
    if scenario.is_null:
        st.info("Set a non-zero effect size to measure power.")
    else:
        with st.spinner(f"running {LIVE_REPS} simulations per test…"):
            rates = cached_rates(as_key(scenario), n_reps=LIVE_REPS, alpha=0.05, metric="power")
        st.bar_chart(rates.set_index("test")[["rate"]], height=280)
        st.dataframe(rates[["test", "±2 SE"]], hide_index=True, width="stretch")
        st.caption(
            f"True effect: {scenario.true_mean_difference:.2f} in means, "
            f"{scenario.true_median_difference:.2f} in medians. Cohen's d is not comparable "
            "across population shapes — a skewed population's standard deviation is inflated "
            "by its tail."
        )

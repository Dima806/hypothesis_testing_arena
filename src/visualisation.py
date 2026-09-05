"""Every figure in the project.

Notebooks and the Streamlit app call these functions; they never build axes themselves. One
test, one colour, in every figure - a reader who learns the palette in notebook 02 can read the
arena heatmap in notebook 04 without a legend lookup.

Any figure that shows a rate estimated from simulation shows the alpha reference line and the
Monte Carlo error, because a rate without its uncertainty invites the reader to over-read a
gap that is noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import FIGURES_DIR
from src.evaluation.calibration import CalibrationResult, rejection_rate_curve, uniform_pp_points
from src.evaluation.comparison import ARENA_TESTS
from src.tests import FloatArray

TEST_COLOURS: dict[str, str] = {
    "student": "#d62728",  # the cautionary protagonist, in red
    "welch": "#ff7f0e",
    "mann_whitney": "#9467bd",
    "permutation": "#2ca02c",  # the recommended general-purpose tool, in green
    "bootstrap": "#1f77b4",
    "bayesian": "#8c564b",
}

TEST_LABELS: dict[str, str] = {
    "student": "Student t",
    "welch": "Welch t",
    "mann_whitney": "Mann-Whitney U",
    "permutation": "Permutation",
    "bootstrap": "Bootstrap",
    "bayesian": "Bayesian",
}


def _as_figure(ax: Axes) -> Figure:
    """matplotlib types ``ax.figure`` as ``Figure | SubFigure``; ours is always a Figure."""
    return cast(Figure, ax.figure)


def colour(test: str) -> str:
    """Colour for a test, stable across the whole project."""
    return TEST_COLOURS.get(test, "#7f7f7f")


def label(test: str) -> str:
    """Display label for a test."""
    return TEST_LABELS.get(test, test)


def _ordered_tests(present: Sequence[str]) -> list[str]:
    known = [name for name in ARENA_TESTS if name in set(present)]
    return known + [name for name in present if name not in set(ARENA_TESTS)]


def save_figure(fig: Figure, name: str, *, dpi: int = 150) -> str:
    """Save a figure to ``outputs/figures/`` under a stable filename and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)


def plot_false_positive_rates(
    arena: pd.DataFrame, *, alpha: float = 0.05, ax: Axes | None = None
) -> Figure:
    """Grouped bars of the false-positive rate per scenario, with the alpha line.

    Error bars are +/- 2 Monte Carlo standard errors. A bar whose error bar straddles the alpha
    line is not evidence of anything.
    """
    nulls = arena[arena["metric"] == "fpr"]
    scenarios = list(dict.fromkeys(nulls["scenario"]))
    tests = _ordered_tests(list(dict.fromkeys(nulls["test"])))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(tests), 1)
    positions = np.arange(len(scenarios))

    for index, test in enumerate(tests):
        subset = nulls[nulls["test"] == test].set_index("scenario")
        rates = [float(subset.loc[name, "reject_rate"]) for name in scenarios]
        errors = [2.0 * float(subset.loc[name, "mc_se"]) for name in scenarios]
        ax.bar(
            positions + (index - (len(tests) - 1) / 2) * width,
            rates,
            width=width,
            yerr=errors,
            capsize=2,
            color=colour(test),
            label=label(test),
        )

    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.2, label=f"nominal α = {alpha}")
    # Groups are centred on their tick, which holds for any number of tests (the arena shows
    # six; notebook 01 shows two). The explicit limits keep a margin when there is only one
    # scenario, where matplotlib would otherwise let the bars fill the whole axes.
    ax.set_xticks(positions)
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    ax.set_xticklabels([s.replace("::null", "") for s in scenarios], rotation=30, ha="right")
    ax.set_ylabel("false-positive rate")
    ax.set_title("Type I error under the null: everything above the dashed line is a phantom")
    ax.legend(ncol=4, fontsize=8)
    return _as_figure(ax)


def plot_power(arena: pd.DataFrame, *, ax: Axes | None = None) -> Figure:
    """Grouped bars of power per scenario.

    Power is only meaningful for a test that holds its false-positive rate; read this next to
    :func:`plot_false_positive_rates`, never on its own.
    """
    effects = arena[arena["metric"] == "power"]
    scenarios = list(dict.fromkeys(effects["scenario"]))
    tests = _ordered_tests(list(dict.fromkeys(effects["test"])))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(tests), 1)
    positions = np.arange(len(scenarios))

    for index, test in enumerate(tests):
        subset = effects[effects["test"] == test].set_index("scenario")
        rates = [float(subset.loc[name, "reject_rate"]) for name in scenarios]
        errors = [2.0 * float(subset.loc[name, "mc_se"]) for name in scenarios]
        ax.bar(
            positions + (index - (len(tests) - 1) / 2) * width,
            rates,
            width=width,
            yerr=errors,
            capsize=2,
            color=colour(test),
            label=label(test),
        )

    # Groups are centred on their tick, which holds for any number of tests (the arena shows
    # six; notebook 01 shows two). The explicit limits keep a margin when there is only one
    # scenario, where matplotlib would otherwise let the bars fill the whole axes.
    ax.set_xticks(positions)
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    ax.set_xticklabels([s.replace("::effect", "") for s in scenarios], rotation=30, ha="right")
    ax.set_ylabel("power")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Power under a real effect (only trustworthy where α is held)")
    ax.legend(ncol=4, fontsize=8)
    return _as_figure(ax)


def plot_pvalue_calibration(
    results: Sequence[CalibrationResult], *, alpha: float = 0.05, ax: Axes | None = None
) -> Figure:
    """PP-plot of null p-values against Uniform(0, 1). The proof figure of notebook 02.

    A valid test lies on the diagonal. Above the diagonal means too many small p-values, so the
    test is manufacturing discoveries; below means it is conservative and spending power.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for result in results:
        theoretical, empirical = uniform_pp_points(result.p_values)
        ax.plot(
            theoretical,
            empirical,
            color=colour(result.test),
            label=f"{label(result.test)} (KS = {result.ks_statistic:.3f})",
            linewidth=1.8,
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, label="uniform (valid)")
    ax.axvline(alpha, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xlabel("theoretical quantile of Uniform(0, 1)")
    ax.set_ylabel("observed p-value quantile")
    ax.set_title("Under the null, p-values must be uniform")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    return _as_figure(ax)


def plot_rejection_rate_curve(
    results: Sequence[CalibrationResult], *, ax: Axes | None = None
) -> Figure:
    """The level you asked for versus the level you got, read straight off the diagonal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for result in results:
        levels, observed = rejection_rate_curve(result.p_values)
        ax.plot(levels, observed, color=colour(result.test), label=label(result.test), lw=1.8)
    limit = 0.2
    ax.plot([0, limit], [0, limit], color="black", ls="--", lw=1.0, label="nominal = actual")
    ax.set_xlabel("nominal level")
    ax.set_ylabel("actual rejection rate")
    ax.set_title("Nominal versus actual false-positive rate")
    ax.legend(fontsize=8)
    return _as_figure(ax)


def plot_null_distribution(
    null: FloatArray,
    observed: float,
    *,
    p_value: float | None = None,
    ax: Axes | None = None,
    title: str = "The permutation null, built by shuffling the labels",
) -> Figure:
    """The permutation null distribution with the observed statistic marked in it.

    This is the whole idea of the permutation test in one picture: the null was not assumed, it
    was constructed from the data.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(null, bins=60, color="#c7c7c7", edgecolor="white")
    ax.axvline(observed, color=colour("permutation"), lw=2.0, label=f"observed = {observed:.3f}")
    ax.axvline(-observed, color=colour("permutation"), lw=1.0, ls=":")
    if p_value is not None:
        ax.set_title(f"{title}  (p = {p_value:.4f})")
    else:
        ax.set_title(title)
    ax.set_xlabel("statistic under shuffled labels")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    return _as_figure(ax)


def plot_arena_heatmap(
    arena: pd.DataFrame, *, metric: str = "fpr", alpha: float = 0.05, ax: Axes | None = None
) -> Figure:
    """Scenarios x tests as a heatmap.

    For ``metric="fpr"`` the colour is the *distance from alpha*, so calibrated cells are pale
    and phantom-discovery cells are loud; for ``metric="power"`` it is the rate itself.
    """
    subset = arena[arena["metric"] == metric]
    wide = subset.pivot(index="scenario", columns="test", values="reject_rate")
    wide = wide[[name for name in ARENA_TESTS if name in wide.columns]]

    values = wide.to_numpy(dtype=float)
    if metric == "fpr":
        shown = values - alpha
        cmap, vmin, vmax = "RdBu_r", -max(abs(shown).max(), 1e-9), max(abs(shown).max(), 1e-9)
    else:
        shown, cmap, vmin, vmax = values, "viridis", 0.0, 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.55 * len(wide) + 2.5))
    image = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels([label(c) for c in wide.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels([str(name).split("::")[0] for name in wide.index])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.3f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(image, ax=ax, label="rate − α" if metric == "fpr" else "power")
    ax.set_title(
        "False-positive rate minus α (blue = conservative, red = phantom discoveries)"
        if metric == "fpr"
        else "Power"
    )
    return _as_figure(ax)


def plot_power_curve(
    frame: pd.DataFrame, *, x: str = "n_a", ax: Axes | None = None, title: str | None = None
) -> Figure:
    """Power against a swept knob, one line per test, with Monte Carlo bands.

    ``frame`` is the tidy output of :func:`~src.evaluation.power.power_curve`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for test in _ordered_tests(list(dict.fromkeys(frame["test"]))):
        subset = frame[frame["test"] == test].sort_values(x)
        ax.plot(
            subset[x], subset["power"], marker="o", color=colour(test), label=label(test), lw=1.8
        )
        ax.fill_between(
            subset[x],
            subset["power"] - 2 * subset["mc_se"],
            subset["power"] + 2 * subset["mc_se"],
            color=colour(test),
            alpha=0.15,
        )
    ax.axhline(0.8, color="grey", ls=":", lw=1.0, label="conventional 80% target")
    ax.set_xlabel(
        {"n_a": "sample size per group", "effect_d": "true effect (Cohen's d)"}.get(x, x)
    )
    ax.set_ylabel("power")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title or "Power against sample size")
    ax.legend(fontsize=8)
    return _as_figure(ax)


def plot_bootstrap_distribution(
    boot: FloatArray,
    observed: float,
    ci: tuple[float, float],
    *,
    ax: Axes | None = None,
    title: str = "The bootstrap answers 'how big, and how sure'",
) -> Figure:
    """The resampling distribution of the difference with its percentile interval marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(boot, bins=60, color="#c7c7c7", edgecolor="white")
    ax.axvline(observed, color=colour("bootstrap"), lw=2.0, label=f"observed = {observed:.3f}")
    ax.axvline(ci[0], color=colour("bootstrap"), ls="--", lw=1.4)
    ax.axvline(ci[1], color=colour("bootstrap"), ls="--", lw=1.4, label="95% percentile interval")
    ax.axvline(0.0, color="black", lw=1.2, ls=":", label="no difference")
    ax.set_xlabel("resampled difference (b − a)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return _as_figure(ax)


def plot_multiple_comparisons(
    p_values: FloatArray, *, alpha: float = 0.05, ax: Axes | None = None
) -> Figure:
    """Twenty null tests, and the one that came up 'significant' anyway."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(1, p_values.size + 1)
    hit = p_values <= alpha
    ax.bar(
        positions,
        p_values,
        color=np.where(hit, "#d62728", "#c7c7c7"),
        edgecolor="white",
    )
    ax.axhline(alpha, color="black", ls="--", lw=1.2, label=f"α = {alpha}")
    ax.set_xlabel("metric tested (all of them null)")
    ax.set_ylabel("p-value")
    ax.set_title(
        f"{int(hit.sum())} of {p_values.size} null metrics came up 'significant' by chance"
    )
    ax.legend(fontsize=9)
    return _as_figure(ax)


def plot_decision_flowchart(*, ax: Axes | None = None) -> Figure:
    """The whole project as one picture: which test, and why."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, text, face, width=3.4, height=0.95, size=9):
        ax.add_patch(
            plt.Rectangle(  # type: ignore[attr-defined]
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor=face,
                edgecolor="#444444",
                linewidth=1.2,
                zorder=2,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=3, wrap=True)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.2},
            zorder=1,
        )
        if text:
            ax.text(
                (x1 + x2) / 2 + 0.15,
                (y1 + y2) / 2,
                text,
                fontsize=8,
                style="italic",
                color="#333333",
            )

    box(5, 9.3, "Comparing two groups", "#e8e8e8", width=4.0)
    box(5, 7.8, "Comparing something\nother than a mean?", "#fff3cd", width=4.0)
    box(1.9, 6.2, "Permutation\nor bootstrap\n(any statistic)", colour("permutation"), width=3.0)
    box(6.6, 6.2, "Skewed, heavy-tailed,\noutliers, or small n?", "#fff3cd", width=4.0)
    box(1.9, 4.4, "Permutation\nor bootstrap\n(assume nothing)", colour("permutation"), width=3.0)
    box(6.6, 4.4, "Unequal variances?", "#fff3cd", width=3.4)
    box(4.3, 2.8, "Welch t-test\n(never Student)", colour("welch"), width=3.0)
    box(8.3, 2.8, "Student is fine —\nbut Welch is free", colour("student"), width=3.2)
    box(
        5,
        1.1,
        "Want P(effect) instead of a p-value?  →  Bayesian estimation",
        "#d5e8f7",
        width=8.0,
    )

    arrow(5, 8.83, 5, 8.28)
    arrow(3.0, 7.8, 1.9, 6.68, "yes")
    arrow(7.0, 7.8, 6.6, 6.68, "no")
    arrow(4.6, 6.2, 3.4, 4.88, "yes")
    arrow(6.6, 5.73, 6.6, 4.88, "no")
    arrow(5.4, 4.13, 4.3, 3.28, "yes")
    arrow(7.8, 4.13, 8.3, 3.28, "no")

    ax.set_title(
        "Which test? (and the universal rule: never report a p-value from a test\n"
        "whose assumptions you did not check)",
        fontsize=11,
    )
    return _as_figure(ax)


def plot_samples(
    a: FloatArray, b: FloatArray, *, ax: Axes | None = None, title: str = "The two groups"
) -> Figure:
    """Overlaid histograms of the two groups with their means marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins=40)
    ax.hist(a, bins=bins, alpha=0.6, color="#4c72b0", label=f"a (n = {a.size})")
    ax.hist(b, bins=bins, alpha=0.6, color="#dd8452", label=f"b (n = {b.size})")
    ax.axvline(float(a.mean()), color="#4c72b0", ls="--", lw=1.5)
    ax.axvline(float(b.mean()), color="#dd8452", ls="--", lw=1.5)
    ax.set_title(title)
    ax.legend(fontsize=9)
    return _as_figure(ax)

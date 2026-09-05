"""Null and effect scenarios carrying a known ground truth.

A scenario fixes everything about a simulated comparison: the shape of both populations, the two
sample sizes, the variance ratio and the true effect. It records the true difference in **means**
*and* the true difference in **medians**, because under skew those are not the same number and
Mann-Whitney is not answering the same question as the t-tests. Without both recorded, the arena
would convict Mann-Whitney of an error it did not make.

The effect is specified in Cohen's d units relative to group a's standard deviation, so
``delta = effect_d * sd_a`` is the true difference in population means, exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy.random import Generator

from src.config import PopulationKind, PopulationSpec, Settings, get_settings
from src.simulation.populations import Population, ShiftedPopulation, build_population
from src.tests import FloatArray


@dataclass(frozen=True)
class Scenario:
    """One simulated comparison with a known truth.

    Attributes
    ----------
    effect_d:
        True effect in units of ``sd_a``. Zero means the null is true.
    sd_ratio:
        ``sd_b / sd_a``. Values above 1 with unequal sample sizes are what break Student.
    """

    name: str
    kind: PopulationKind
    n_a: int
    n_b: int
    mean_a: float
    sd_a: float
    effect_d: float = 0.0
    sd_ratio: float = 1.0
    df: float = 3.0
    contamination: float = 0.1
    outlier_sd_mult: float = 10.0

    @property
    def delta(self) -> float:
        """True difference in population means, ``mean_b - mean_a``."""
        return self.effect_d * self.sd_a

    @property
    def sd_b(self) -> float:
        return self.sd_a * self.sd_ratio

    @property
    def mean_b(self) -> float:
        return self.mean_a + self.delta

    @property
    def is_null(self) -> bool:
        """True when the population means are genuinely equal."""
        return self.delta == 0.0

    @property
    def population_a(self) -> Population:
        return self._population(self.mean_a, self.sd_a)

    @property
    def population_b(self) -> Population:
        """Group b: group a's shape at ``sd_b``, moved sideways by ``delta``.

        Building the effect as a pure location shift rather than as a higher target mean keeps
        the treated group's skewness equal to the control group's. See
        :class:`~src.simulation.populations.ShiftedPopulation` for why that is not a detail.
        """
        base = self._population(self.mean_a, self.sd_b)
        return ShiftedPopulation(base=base, shift=self.delta) if self.delta else base

    @property
    def true_mean_difference(self) -> float:
        """Exactly ``delta``, by construction."""
        return self.delta

    @property
    def true_median_difference(self) -> float:
        """The estimand Mann-Whitney is closer to.

        Equal to ``delta`` under a pure location shift, and different from it as soon as the two
        groups have different spreads on a skewed population - which is exactly the case where
        reading a Mann-Whitney rejection as a statement about means goes wrong.
        """
        return self.population_b.median - self.population_a.median

    @property
    def skewness(self) -> float:
        return self.population_a.skewness

    def _population(self, mean: float, sd: float) -> Population:
        spec = PopulationSpec(
            mean=mean,
            sd=sd,
            df=self.df,
            contamination=self.contamination,
            outlier_sd_mult=self.outlier_sd_mult,
        )
        return build_population(self.kind, mean=mean, sd=sd, spec=spec)

    def draw(self, rng: Generator) -> tuple[FloatArray, FloatArray]:
        """Draw one pair of samples ``(a, b)``."""
        return self.population_a.rvs(self.n_a, rng), self.population_b.rvs(self.n_b, rng)

    def with_effect(self, effect_d: float, *, suffix: str | None = None) -> Scenario:
        """Return a copy with a different true effect, keeping every other knob fixed."""
        label = suffix if suffix is not None else ("null" if effect_d == 0.0 else "effect")
        base = self.name.rsplit("::", maxsplit=1)[0]
        return replace(self, name=f"{base}::{label}", effect_d=effect_d)


def make_scenario(
    name: str,
    kind: PopulationKind,
    *,
    n_a: int,
    n_b: int,
    effect_d: float = 0.0,
    sd_ratio: float = 1.0,
    settings: Settings | None = None,
) -> Scenario:
    """Build a scenario, taking the population defaults from ``config/settings.yaml``."""
    resolved = settings or get_settings()
    spec = resolved.population(kind)
    return Scenario(
        name=name,
        kind=kind,
        n_a=n_a,
        n_b=n_b,
        mean_a=spec.mean,
        sd_a=spec.sd,
        effect_d=effect_d,
        sd_ratio=sd_ratio,
        df=spec.df,
        contamination=spec.contamination,
        outlier_sd_mult=spec.outlier_sd_mult,
    )


def base_scenarios(settings: Settings | None = None) -> list[Scenario]:
    """The seven conditions the arena sweeps, each still at ``effect_d = 0``.

    Ordered from the t-test's home ground outwards to the conditions real data actually has.
    """
    resolved = settings or get_settings()
    grid = resolved.grid
    return [
        make_scenario(
            "normal_equal_var",
            "normal",
            n_a=grid.n_large // 2,
            n_b=grid.n_large // 2,
            settings=resolved,
        ),
        # Unequal variance with EQUAL n: the honesty check. Student is close to fine here,
        # which is why the failure stays invisible in textbook examples.
        make_scenario(
            "normal_unequal_var_equal_n",
            "normal",
            n_a=grid.n_large // 2,
            n_b=grid.n_large // 2,
            sd_ratio=4.0,
            settings=resolved,
        ),
        # Unequal variance with the LARGER spread in the SMALLER group: the showpiece. The
        # pooled variance underestimates the standard error and Student's false-positive rate
        # climbs well above alpha.
        make_scenario(
            "normal_unequal_var_unequal_n",
            "normal",
            n_a=grid.n_large,
            n_b=grid.n_small,
            sd_ratio=4.0,
            settings=resolved,
        ),
        make_scenario(
            "skewed_small_n",
            "lognormal",
            n_a=grid.n_small,
            n_b=grid.n_small,
            settings=resolved,
        ),
        # Skewed and unbalanced, the shape of most real business metrics. Permutation is exact
        # here (both groups share a distribution under the null) while Welch, which trusts a
        # normal approximation the data does not support, runs above alpha.
        make_scenario(
            "skewed_unequal_n",
            "lognormal",
            n_a=grid.n_large,
            n_b=grid.n_small,
            settings=resolved,
        ),
        # The honest limit case: extreme skew (group b's skewness is 14) in the smaller group.
        # No mean-based test holds alpha here, the assumption-free ones included. Kept in the
        # grid on purpose - a project about overclaimed tests should not overclaim its own.
        make_scenario(
            "skewed_extreme_unequal",
            "lognormal",
            n_a=grid.n_large,
            n_b=grid.n_small,
            sd_ratio=2.0,
            settings=resolved,
        ),
        make_scenario(
            "heavy_tailed",
            "heavy_tailed",
            n_a=grid.n_large // 2,
            n_b=grid.n_large // 2,
            settings=resolved,
        ),
        make_scenario(
            "contaminated_outliers",
            "contaminated",
            n_a=grid.n_large // 2,
            n_b=grid.n_large // 2,
            settings=resolved,
        ),
    ]


def default_scenarios(settings: Settings | None = None) -> list[Scenario]:
    """The full arena grid: every base condition as a null pair and an effect pair."""
    resolved = settings or get_settings()
    effect_d = resolved.grid.arena_effect_d
    scenarios: list[Scenario] = []
    for base in base_scenarios(resolved):
        scenarios.append(base.with_effect(0.0))
        scenarios.append(base.with_effect(effect_d))
    return scenarios


def null_scenarios(settings: Settings | None = None) -> list[Scenario]:
    """Just the null half of the grid, for false-positive and calibration work."""
    return [s for s in default_scenarios(settings) if s.is_null]


def effect_scenarios(settings: Settings | None = None) -> list[Scenario]:
    """Just the effect half of the grid, for power work."""
    return [s for s in default_scenarios(settings) if not s.is_null]

"""Population generators, each parameterized by its true mean and standard deviation.

Every population takes ``(mean, sd)`` as the *population* mean and standard deviation, whatever
its shape. That is what makes the arena fair: an effect of 0.8 standard deviations is the same
size whether the data is normal, lognormal, heavy-tailed or contaminated, so power differences
across kinds are about the tests, not about an accidentally larger effect.

Each population also exposes its true **median**, because the mean-based tests and Mann-Whitney
target different estimands and the difference between the two is exactly what skew creates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.random import Generator

from src.config import PopulationKind, PopulationSpec
from src.tests import FloatArray


class Population(Protocol):
    """A distribution with a known mean, standard deviation and median.

    Declared with read-only properties rather than plain attributes so that the frozen
    dataclasses below satisfy it: a mutable protocol attribute would require a settable one.
    """

    @property
    def name(self) -> str:
        """Population kind, used for labels."""
        ...

    @property
    def mean(self) -> float:
        """The true population mean."""
        ...

    @property
    def sd(self) -> float:
        """The true population standard deviation."""
        ...

    @property
    def median(self) -> float:
        """The true population median."""
        ...

    @property
    def skewness(self) -> float:
        """The true population skewness (0.0 for the symmetric kinds)."""
        ...

    def rvs(self, n: int, rng: Generator) -> FloatArray:
        """Draw ``n`` observations."""
        ...


@dataclass(frozen=True)
class NormalPopulation:
    """The t-test's home ground."""

    mean: float
    sd: float
    name: str = "normal"

    @property
    def median(self) -> float:
        return self.mean

    @property
    def skewness(self) -> float:
        return 0.0

    def rvs(self, n: int, rng: Generator) -> FloatArray:
        return np.asarray(rng.normal(self.mean, self.sd, size=n), dtype=np.float64)


@dataclass(frozen=True)
class LogNormalPopulation:
    """Right-skewed, the shape of revenue, session length and basket size.

    Parameterized by the target mean and standard deviation: with ``cv = sd / mean``,
    ``sigma^2 = log(1 + cv^2)`` and ``mu = log(mean) - sigma^2 / 2``. Skewness is then
    ``(cv^2 + 3) cv``, so the coefficient of variation *is* the skew knob.
    """

    mean: float
    sd: float
    name: str = "lognormal"

    def __post_init__(self) -> None:
        if self.mean <= 0.0:
            raise ValueError("a lognormal population needs a strictly positive mean")

    @property
    def cv(self) -> float:
        return self.sd / self.mean

    @property
    def sigma(self) -> float:
        return math.sqrt(math.log1p(self.cv**2))

    @property
    def mu(self) -> float:
        return math.log(self.mean) - self.sigma**2 / 2.0

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    @property
    def skewness(self) -> float:
        return (self.cv**2 + 3.0) * self.cv

    def rvs(self, n: int, rng: Generator) -> FloatArray:
        return np.asarray(rng.lognormal(self.mu, self.sigma, size=n), dtype=np.float64)


@dataclass(frozen=True)
class HeavyTailedPopulation:
    """A scaled Student-t: symmetric, but with tails that produce real outliers.

    ``df > 2`` keeps the variance finite; ``df <= 4`` leaves the kurtosis infinite, which is
    what makes the sample standard deviation - and therefore the t statistic - erratic.
    """

    mean: float
    sd: float
    df: float = 3.0
    name: str = "heavy_tailed"

    def __post_init__(self) -> None:
        if self.df <= 2.0:
            raise ValueError("df must exceed 2 for the population variance to exist")

    @property
    def scale(self) -> float:
        return self.sd / math.sqrt(self.df / (self.df - 2.0))

    @property
    def median(self) -> float:
        return self.mean

    @property
    def skewness(self) -> float:
        return 0.0

    def rvs(self, n: int, rng: Generator) -> FloatArray:
        return np.asarray(self.mean + self.scale * rng.standard_t(self.df, size=n))


@dataclass(frozen=True)
class ContaminatedNormalPopulation:
    """Mostly normal, with a minority of draws from a much wider component.

    The everyday version of "outliers": a clean process plus a small fraction of logging errors,
    bots or whales. ``sd`` is the standard deviation of the *mixture*, so the base component is
    scaled down accordingly.
    """

    mean: float
    sd: float
    contamination: float = 0.1
    outlier_sd_mult: float = 10.0
    name: str = "contaminated"

    def __post_init__(self) -> None:
        if not 0.0 <= self.contamination < 1.0:
            raise ValueError("contamination must lie in [0, 1)")

    @property
    def base_sd(self) -> float:
        inflation = (1.0 - self.contamination) + self.contamination * self.outlier_sd_mult**2
        return self.sd / math.sqrt(inflation)

    @property
    def median(self) -> float:
        return self.mean

    @property
    def skewness(self) -> float:
        return 0.0

    def rvs(self, n: int, rng: Generator) -> FloatArray:
        is_outlier = rng.random(n) < self.contamination
        scale = np.where(is_outlier, self.base_sd * self.outlier_sd_mult, self.base_sd)
        return np.asarray(rng.normal(self.mean, scale), dtype=np.float64)


@dataclass(frozen=True)
class ShiftedPopulation:
    """A population moved sideways by a constant, leaving its shape untouched.

    This is how effect scenarios are built, and the choice matters. The obvious alternative -
    raising the target mean and holding the standard deviation fixed - silently changes the
    shape of a skewed population: for a lognormal with mean 10 and sd 10, adding an effect of
    0.8 sd drops the treated group's coefficient of variation from 1.00 to 0.56, so it is
    materially *less* skewed than the control group. The arena would then be measuring each
    test's response to a shape change as well as to a mean difference, and the measured power
    on skewed data came out *higher* than on normal data purely because of it.

    A pure location shift keeps the mean difference, the median difference and the skewness of
    both groups all exactly where they were intended, so every contender is aimed at the same
    target.
    """

    base: Population
    shift: float

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def mean(self) -> float:
        return self.base.mean + self.shift

    @property
    def sd(self) -> float:
        return self.base.sd

    @property
    def median(self) -> float:
        return self.base.median + self.shift

    @property
    def skewness(self) -> float:
        return self.base.skewness

    def rvs(self, n: int, rng: Generator) -> FloatArray:
        return np.asarray(self.base.rvs(n, rng) + self.shift, dtype=np.float64)


def build_population(
    kind: PopulationKind, mean: float, sd: float, spec: PopulationSpec | None = None
) -> Population:
    """Construct a population of ``kind`` with the given true mean and standard deviation.

    ``spec`` supplies the shape parameters (degrees of freedom, contamination rate) that are not
    determined by the mean and standard deviation; it comes from ``config/settings.yaml``.
    """
    if kind == "normal":
        return NormalPopulation(mean=mean, sd=sd)
    if kind == "lognormal":
        return LogNormalPopulation(mean=mean, sd=sd)
    if kind == "heavy_tailed":
        df = spec.df if spec is not None else 3.0
        return HeavyTailedPopulation(mean=mean, sd=sd, df=df)
    if kind == "contaminated":
        contamination = spec.contamination if spec is not None else 0.1
        multiplier = spec.outlier_sd_mult if spec is not None else 10.0
        return ContaminatedNormalPopulation(
            mean=mean, sd=sd, contamination=contamination, outlier_sd_mult=multiplier
        )
    raise ValueError(f"unknown population kind: {kind!r}")

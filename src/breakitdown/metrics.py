"""Metric definitions.

A metric turns the rows of a ``MetricTable`` into one number per flight and knows how to test the
difference between two flights.

==========  ===============================================  =================================
Kind        Value per flight                                 Test
==========  ===============================================  =================================
Count       mean over units of each unit's total             Welch t-test
Binary      share of units with any positive value (or a     Welch t-test on the 0/1 indicator
            total of at least ``min_sum``)
Ratio       sum(numerator) / sum(denominator) over units     delta method, z-test
MeanRatio   mean over units of numerator / denominator       Welch t-test on the unit ratios
Quantile    p-th quantile of the row values (events)         unit-clustered asymptotic z-test
==========  ===============================================  =================================

Units without rows count as zero for Count, Binary and Ratio, so the denominator of a mean is
always the flight's population, never just the units that happened to have rows. MeanRatio is the
exception: a unit's ratio exists only when its denominator is not zero, so it averages over those
units and reports them as its own unit count.

Each kind owns everything specific to it, which keeps adding a kind (or capping, or a
variance-reduction method) a local change:

1. ``unit_aggregations``: Spark aggregates rolling a unit's rows up to one value per unit.
2. ``group_aggregations``: Spark aggregates reducing unit values to sufficient statistics per
   flight x segment x day.
3. ``estimate``: numpy code turning sufficient statistics into a point estimate and its variance.

Quantile metrics need the raw rows, so their Spark side lives in ``aggregate.py``; they still own
their estimator and sample-size rule.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pyspark.sql import Column
from pyspark.sql import functions as F

from . import stats
from ._common import quote

StatGetter = Callable[..., np.ndarray]
"""``stat(name, fill=0.0)`` returns one sufficient statistic as a float array; NaN -> ``fill``."""

__all__ = ["Direction", "QuantileMethod", "Metric", "UnitMetric", "Count", "Binary", "Ratio", "MeanRatio", "Quantile"]


class Direction(str, enum.Enum):
    """Which direction of change is an improvement for a metric."""

    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class QuantileMethod(str, enum.Enum):
    """How a quantile and the density at that quantile are estimated.

    REGULAR
        Spark ``percentile_approx`` at p and at p +/- ``bandwidth``. Cheap; suited to continuous
        values. With discrete values the quantiles at p +/- bandwidth often tie, which leaves the
        density (and so the test) undefined.
    DENSITY
        Exact histogram of the values, with the CDF interpolated linearly inside an interval of
        +/- ``resolution`` / 2 around each distinct value. Gives a usable density for integer-like
        metrics. Cost grows with the number of distinct values, so round or clip (``lower`` /
        ``upper``) high-cardinality metrics.
    """

    REGULAR = "regular"
    DENSITY = "density"


@dataclass(frozen=True)
class MetricOutput:
    """One result row family of a metric. A quantile metric has one per probability."""

    prefix: str
    quantile: Optional[float] = None


class Metric:
    """Base class of all metric kinds. Concrete kinds are frozen dataclasses."""

    kind: ClassVar[str] = ""
    name: str
    label: str
    direction: Direction

    @property
    def display_name(self) -> str:
        return self.label or self.name

    @property
    def input_columns(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def outputs(self, key: str) -> List[MetricOutput]:
        return [MetricOutput(key)]

    def estimate(self, stat: StatGetter, n: np.ndarray) -> stats.Estimate:
        raise NotImplementedError

    def has_min_sample(self, control: stats.Estimate, treatment: stats.Estimate) -> np.ndarray:
        """Rule-of-thumb sample sizes under which the normal approximation is not trusted."""
        raise NotImplementedError


class UnitMetric(Metric):
    """A metric computed from per-unit values: Count, Binary and Ratio."""

    def unit_aggregations(self, key: str) -> Dict[str, Column]:
        raise NotImplementedError

    def group_aggregations(self, key: str) -> Dict[str, Column]:
        raise NotImplementedError


def _number(col: str) -> Column:
    return F.col(quote(col)).cast("double")


def _setup(metric: Metric, default_name: str) -> None:
    if not metric.name:
        object.__setattr__(metric, "name", default_name)
    object.__setattr__(metric, "direction", Direction(metric.direction))


@dataclass(frozen=True)
class Count(UnitMetric):
    """Mean over units of each unit's total ``col``, e.g. clicks per user.

    Rows are summed per unit (per unit and day for daily results), so the table can be at any grain
    at or below unit x day. Units without rows count as zero.
    """

    col: str
    name: str = ""
    label: str = ""
    direction: Union[Direction, str] = Direction.UP

    kind: ClassVar[str] = "count"
    min_units: ClassVar[int] = 200

    def __post_init__(self) -> None:
        _setup(self, self.col)

    @property
    def input_columns(self) -> Tuple[str, ...]:
        return (self.col,)

    def unit_aggregations(self, key: str) -> Dict[str, Column]:
        return {key: F.sum(_number(self.col))}

    def group_aggregations(self, key: str) -> Dict[str, Column]:
        x = F.coalesce(F.col(key), F.lit(0.0))
        return {f"{key}__sum": F.sum(x), f"{key}__sumsq": F.sum(x * x)}

    def estimate(self, stat: StatGetter, n: np.ndarray) -> stats.Estimate:
        return stats.mean_estimate(n, stat("sum"), stat("sumsq"))

    def has_min_sample(self, control: stats.Estimate, treatment: stats.Estimate) -> np.ndarray:
        return (control.n >= self.min_units) & (treatment.n >= self.min_units)


@dataclass(frozen=True)
class Binary(UnitMetric):
    """Share of units with any positive ``col`` value, e.g. the share of users who clicked.

    With ``min_sum``, a unit counts when its total ``col`` is at least ``min_sum`` instead: on a
    user x day table with a 0/1 ``active`` column, ``Binary("active", min_sum=2)`` is the share of
    users active on two or more days.
    """

    col: str
    name: str = ""
    label: str = ""
    direction: Union[Direction, str] = Direction.UP
    min_sum: Optional[float] = None

    kind: ClassVar[str] = "binary"
    min_events: ClassVar[int] = 10

    def __post_init__(self) -> None:
        _setup(self, self.col)
        if self.min_sum is not None and not float(self.min_sum) > 0:
            raise ValueError("min_sum must be positive: units without rows never count.")

    @property
    def input_columns(self) -> Tuple[str, ...]:
        return (self.col,)

    def unit_aggregations(self, key: str) -> Dict[str, Column]:
        if self.min_sum is None:
            return {key: F.max(F.when(_number(self.col) > 0, F.lit(1.0)).otherwise(F.lit(0.0)))}
        return {key: F.when(F.sum(_number(self.col)) >= float(self.min_sum), F.lit(1.0)).otherwise(F.lit(0.0))}

    def group_aggregations(self, key: str) -> Dict[str, Column]:
        return {f"{key}__hits": F.sum(F.coalesce(F.col(key), F.lit(0.0)))}

    def estimate(self, stat: StatGetter, n: np.ndarray) -> stats.Estimate:
        hits = stat("hits")
        return stats.mean_estimate(n, hits, hits, events=hits)

    def has_min_sample(self, control: stats.Estimate, treatment: stats.Estimate) -> np.ndarray:
        m = self.min_events
        return (
            (control.events >= m)
            & (treatment.events >= m)
            & (control.n - control.events >= m)
            & (treatment.n - treatment.events >= m)
        )


def _component(value: Union[str, Count, Binary], role: str) -> Union[Count, Binary]:
    if isinstance(value, str):
        return Count(value)
    if isinstance(value, (Count, Binary)):
        return value
    raise TypeError(f"Ratio {role} must be a column name, Count or Binary; got {value!r}.")


@dataclass(frozen=True)
class Ratio(UnitMetric):
    """``sum(numerator) / sum(denominator)`` over units, e.g. clicks per session.

    ``numerator`` and ``denominator`` are column names (summed per unit, like Count) or Count/Binary
    metrics (a Binary component contributes the unit's 0/1 indicator). The variance uses the delta
    method with units as the independent observations.
    """

    numerator: Union[str, Count, Binary]
    denominator: Union[str, Count, Binary]
    name: str = ""
    label: str = ""
    direction: Union[Direction, str] = Direction.UP

    kind: ClassVar[str] = "ratio"
    min_units: ClassVar[int] = 100

    def __post_init__(self) -> None:
        numerator = _component(self.numerator, "numerator")
        denominator = _component(self.denominator, "denominator")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        _setup(self, f"{numerator.col}_per_{denominator.col}")

    @property
    def input_columns(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.numerator.input_columns + self.denominator.input_columns))

    def unit_aggregations(self, key: str) -> Dict[str, Column]:
        return {
            **self.numerator.unit_aggregations(f"{key}__n"),
            **self.denominator.unit_aggregations(f"{key}__d"),
        }

    def group_aggregations(self, key: str) -> Dict[str, Column]:
        num = F.coalesce(F.col(f"{key}__n"), F.lit(0.0))
        den = F.coalesce(F.col(f"{key}__d"), F.lit(0.0))
        return {
            f"{key}__num": F.sum(num),
            f"{key}__num2": F.sum(num * num),
            f"{key}__den": F.sum(den),
            f"{key}__den2": F.sum(den * den),
            f"{key}__numden": F.sum(num * den),
        }

    def estimate(self, stat: StatGetter, n: np.ndarray) -> stats.Estimate:
        return stats.ratio_estimate(n, stat("num"), stat("num2"), stat("den"), stat("den2"), stat("numden"))

    def has_min_sample(self, control: stats.Estimate, treatment: stats.Estimate) -> np.ndarray:
        return (control.n >= self.min_units) & (treatment.n >= self.min_units)


@dataclass(frozen=True)
class MeanRatio(UnitMetric):
    """Mean over units of each unit's ``numerator / denominator``, e.g. each user's clicks per session,
    averaged over users.

    ``numerator`` and ``denominator`` are summed per unit, as in ``Ratio``. Only units whose
    denominator is not zero have a ratio, so the mean, its Welch t-test and the reported unit count
    cover those units. ``Ratio`` divides the totals instead, which weighs units by their denominator.
    """

    numerator: Union[str, Count, Binary]
    denominator: Union[str, Count, Binary]
    name: str = ""
    label: str = ""
    direction: Union[Direction, str] = Direction.UP

    kind: ClassVar[str] = "mean_ratio"
    min_units: ClassVar[int] = 200

    def __post_init__(self) -> None:
        numerator = _component(self.numerator, "numerator")
        denominator = _component(self.denominator, "denominator")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        _setup(self, f"mean_{numerator.col}_per_{denominator.col}")

    @property
    def input_columns(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.numerator.input_columns + self.denominator.input_columns))

    def unit_aggregations(self, key: str) -> Dict[str, Column]:
        return {
            **self.numerator.unit_aggregations(f"{key}__n"),
            **self.denominator.unit_aggregations(f"{key}__d"),
        }

    def group_aggregations(self, key: str) -> Dict[str, Column]:
        num = F.coalesce(F.col(f"{key}__n"), F.lit(0.0))
        den = F.coalesce(F.col(f"{key}__d"), F.lit(0.0))
        ratio = F.when(den != 0, num / den)  # NULL without a denominator: left out of all three sums
        return {f"{key}__units": F.count(ratio), f"{key}__sum": F.sum(ratio), f"{key}__sumsq": F.sum(ratio * ratio)}

    def estimate(self, stat: StatGetter, n: np.ndarray) -> stats.Estimate:
        return stats.mean_estimate(stat("units"), stat("sum"), stat("sumsq"))

    def has_min_sample(self, control: stats.Estimate, treatment: stats.Estimate) -> np.ndarray:
        return (control.n >= self.min_units) & (treatment.n >= self.min_units)


def _probability(value: Union[float, int, str]) -> float:
    p = float(value)
    if 1 < p < 100:
        p /= 100.0
    if not 0 < p < 1:
        raise ValueError(f"Quantile {value!r} must be in (0, 1) or a percentage in (1, 100).")
    return p


@dataclass(frozen=True)
class Quantile(Metric):
    """Quantiles of ``col`` over all rows of a flight, e.g. p50/p90 request latency.

    Every row is one observation, so the table is usually at event grain. Rows of one unit are
    correlated; the variance treats units as clusters. Rows where ``col`` is NULL or NaN are ignored.

    Parameters
    ----------
    quantiles:
        Probabilities in (0, 1); percentages such as ``50`` or ``"90"`` are accepted too.
    method:
        ``"regular"`` or ``"density"``; see ``QuantileMethod``.
    bandwidth:
        Half-width h, in probability, of the density estimate ``2h / (Q(p+h) - Q(p-h))``.
        The default is 0.001 (0.1 percentage points).
    lower, upper:
        Optional clipping bounds applied to values before anything else.
    resolution:
        DENSITY method only: the measurement resolution of the values (1 for integer counts).
    accuracy:
        REGULAR method only: ``percentile_approx`` accuracy; relative rank error is 1 / accuracy.
    """

    col: str
    quantiles: Sequence[Union[float, int, str]] = (0.5, 0.9)
    method: Union[QuantileMethod, str] = QuantileMethod.REGULAR
    bandwidth: float = 0.001
    lower: Optional[float] = None
    upper: Optional[float] = None
    resolution: float = 1.0
    accuracy: int = 10_000
    name: str = ""
    label: str = ""
    direction: Union[Direction, str] = Direction.UP

    kind: ClassVar[str] = "quantile"
    min_units: ClassVar[int] = 1000
    min_events: ClassVar[int] = 2000

    def __post_init__(self) -> None:
        _setup(self, self.col)
        object.__setattr__(self, "method", QuantileMethod(self.method))
        probabilities = tuple(_probability(q) for q in self.quantiles)
        if not probabilities:
            raise ValueError("Quantile needs at least one probability.")
        if len(set(probabilities)) != len(probabilities):
            raise ValueError(f"Duplicate quantiles in {self.quantiles!r}.")
        if not 0 < self.bandwidth < 0.5:
            raise ValueError("bandwidth must be in (0, 0.5).")
        for p in probabilities:
            if not (0 < p - self.bandwidth and p + self.bandwidth < 1):
                raise ValueError(f"Quantile {p} +/- bandwidth {self.bandwidth} must stay inside (0, 1).")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower must not exceed upper.")
        object.__setattr__(self, "quantiles", probabilities)

    @property
    def input_columns(self) -> Tuple[str, ...]:
        return (self.col,)

    def outputs(self, key: str) -> List[MetricOutput]:
        return [MetricOutput(f"{key}__p{i}", p) for i, p in enumerate(self.quantiles)]

    def value(self) -> Column:
        """The row value used for the quantile: ``col`` as a double, clipped to [lower, upper]."""
        value = _number(self.col)
        if self.lower is not None:
            value = F.greatest(value, F.lit(float(self.lower)))
        if self.upper is not None:
            value = F.least(value, F.lit(float(self.upper)))
        return value

    def estimate(self, stat: StatGetter, n: np.ndarray) -> stats.Estimate:
        return stats.quantile_estimate(
            q=stat("q", fill=np.nan),
            density=stat("density", fill=np.nan),
            events=stat("events"),
            events_sq=stat("events2"),
            above=stat("above"),
            above_sq=stat("above2"),
            above_events=stat("above_events"),
            n=n,
        )

    def has_min_sample(self, control: stats.Estimate, treatment: stats.Estimate) -> np.ndarray:
        return (
            (control.n >= self.min_units)
            & (treatment.n >= self.min_units)
            & (control.events >= self.min_events)
            & (treatment.events >= self.min_events)
        )

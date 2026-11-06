"""Inputs of an analysis: MetricTable (rows plus metric metadata) and the opt-in FlightMapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from pyspark.sql import DataFrame, SparkSession

from ._common import SegmentsLike, normalize_segments, segment_columns
from .columns import Columns
from .metrics import Metric, Quantile, UnitMetric

Source = Union[str, DataFrame]
"""A SQL query, a table or view name (``db.table``), or a Spark DataFrame."""

_TABLE_NAME = re.compile(r"^[\w`.\-]+$")


def load_source(source: Source, spark: SparkSession) -> DataFrame:
    if isinstance(source, DataFrame):
        return source
    if isinstance(source, str):
        text = source.strip().rstrip(";").strip()
        return spark.table(text) if _TABLE_NAME.match(text) else spark.sql(text)
    raise TypeError(f"A source must be a SQL string, a table name or a DataFrame; got {type(source).__name__}.")


@dataclass
class MetricTable:
    """A source of rows and the metrics computed from them.

    Pre-joined mode (default): every row carries the unit id, its flight, the event time and the
    experiment's user segment columns. The units present in the table (within the date range and
    the experiment's flights) are the population.

    Mapping mode (``Experiment(mapping=...)``): rows need only the unit id and the event time; the
    flight and the user segments come from the mapping, and the mapping is the population.

    Grain: Count, Binary and Ratio roll rows up per unit, so any grain at or below unit x day works.
    Quantile treats each row as one observation.

    Parameters
    ----------
    source:
        SQL query, table/view name or DataFrame. Do not pre-filter dates; breakitdown filters ``time``
        to the experiment's range (and Spark prunes partitions on it).
    metrics:
        Metric objects. Names must be unique across all tables of one analysis.
    name:
        Label of the table in results. Defaults to the table name, or to the metric names.
    metric_segments:
        Breakdowns of the metric itself, e.g. ``["media_type", ("media_type", "os")]``. A metric
        segment splits a unit's rows, so one unit can contribute to several segment values; the
        denominator stays the whole flight. Columns must be in the table.
    columns:
        Column names, if they differ from the experiment's ``Columns``.
    inner_join:
        Mapping mode only: keep only units with rows in this table, instead of counting every mapped
        unit (units without rows as zeros).
    daily:
        Set False to skip daily results for this table.
    """

    source: Source
    metrics: Sequence[Metric]
    name: str = ""
    metric_segments: SegmentsLike = None
    columns: Optional[Columns] = None
    inner_join: bool = False
    daily: bool = True

    def __post_init__(self) -> None:
        metrics = (self.metrics,) if isinstance(self.metrics, Metric) else tuple(self.metrics)
        if not metrics:
            raise ValueError("A MetricTable needs at least one metric.")
        for metric in metrics:
            if not isinstance(metric, Metric):
                raise TypeError(f"Expected Count, Binary, Ratio or Quantile objects, got {metric!r}.")
        duplicates = sorted({m.name for m in metrics if [x.name for x in metrics].count(m.name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate metric names {duplicates}; give each metric a unique name=.")
        self.metrics = metrics
        self.metric_segments = normalize_segments(self.metric_segments, "metric segment")
        overlap = set(self.metric_segment_columns) & set(self.input_columns)
        if overlap:
            raise ValueError(f"Columns {sorted(overlap)} are used both as metric inputs and as metric segments.")
        if not self.name:
            if isinstance(self.source, str) and _TABLE_NAME.match(self.source.strip()):
                self.name = self.source.strip()
            else:
                self.name = "_".join(m.name for m in metrics)[:100]

    @property
    def unit_metrics(self) -> Tuple[UnitMetric, ...]:
        return tuple(m for m in self.metrics if isinstance(m, UnitMetric))

    @property
    def quantile_metrics(self) -> Tuple[Quantile, ...]:
        return tuple(m for m in self.metrics if isinstance(m, Quantile))

    @property
    def input_columns(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(col for metric in self.metrics for col in metric.input_columns))

    @property
    def metric_segment_columns(self) -> Tuple[str, ...]:
        return segment_columns(self.metric_segments)


@dataclass
class FlightMapping:
    """Opt-in: the unit -> flight assignment, joined to every MetricTable of the analysis.

    Use it when metric tables do not carry the flight, or when units without any rows must count
    as zeros. Expected columns: ``columns.unit``, ``columns.flight``, optionally
    ``columns.exposure``, and the experiment's user segment columns. Several rows per unit are fine:
    they are reduced to one row per unit and flight, with the earliest exposure and the largest
    non-null value of each segment column. Units exposed after the end date are left out.
    """

    source: Source
    columns: Optional[Columns] = None

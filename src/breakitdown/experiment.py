"""Experiment: the main entry point of breakitdown."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import aggregate, compare
from ._common import DateLike, Period, SegmentsLike, normalize_segments, quote, segment_columns, to_date
from .aggregate import CUMULATIVE, DAILY, FLIGHT, UNIT
from .columns import Columns
from .metrics import Metric, Quantile, UnitMetric
from .results import ExperimentResults
from .tables import FlightMapping, MetricTable, load_source

log = logging.getLogger("breakitdown")

MetricTables = Union[MetricTable, Iterable[MetricTable]]


class Experiment:
    """An A/B experiment: its flights, its date range and how its units are segmented.

    The experiment reads ``MetricTable``\\ s and compares every treatment flight with the control.

    Two ways to know which unit is in which flight:

    * Pre-joined (default): each metric table already carries the flight (and user segment columns)
      on every row. The units present in a table are its population. No join is run.
    * Flight mapping (opt-in): pass ``mapping=FlightMapping(...)``. One mapping is joined to every
      metric table; mapped units without rows count as zeros.

    Example::

        exp = Experiment(
            control="control",
            treatments=["t1", "t2"],
            start="2024-03-01",
            end="2024-03-14",
            user_segments=["os", ("os", "country")],
        )
        results = exp.analyze([
            MetricTable(
                "SELECT * FROM events.app_events",
                metrics=[Count("clicks"), Binary("purchases", name="buyers"), Ratio("clicks", "sessions")],
                metric_segments=["media_type"],
            ),
            MetricTable("events.latency", metrics=[Quantile("latency_ms", [0.5, 0.9], method="density")]),
        ])
        results.overall()

    Parameters
    ----------
    control, treatments:
        Flight values (compared as strings).
    start, end:
        Inclusive date range of the analysis, as dates or ``YYYY-MM-DD`` / ``YYYYMMDD`` strings.
    user_segments:
        Breakdowns of units, e.g. ``["os", ("os", "country")]`` or ``"os|country"``. A unit is in
        exactly one value of each segment. When a pre-joined unit's rows disagree, the value on its
        earliest row is used, so adding a breakdown never changes the overall results.
    mapping:
        Optional ``FlightMapping``; switches every table to mapping mode.
    columns:
        Column names shared by all tables (``Columns()`` defaults: ruid, flight, event_time).
    expected_traffic:
        Optional ``{flight: share}`` for sample-ratio-mismatch tests, e.g. ``{"control": 1, "t1": 1}``.
    exclude_multi_flight_units:
        Leave out units seen in more than one of the experiment's flights, say in both control and
        treatment logs (the default). With False they count in each flight. ``validate()`` reports
        how many there are either way.
    spark:
        SparkSession; defaults to the one of a DataFrame source, else the active session.
    """

    def __init__(
        self,
        control: str,
        treatments: Union[str, Sequence[str]],
        start: DateLike,
        end: DateLike,
        *,
        user_segments: SegmentsLike = None,
        mapping: Optional[FlightMapping] = None,
        columns: Optional[Columns] = None,
        expected_traffic: Optional[Mapping[str, float]] = None,
        exclude_multi_flight_units: bool = True,
        spark: Optional[SparkSession] = None,
    ):
        self.control = str(control)
        treatment_list = [treatments] if isinstance(treatments, str) else list(treatments)
        self.treatments = tuple(str(t) for t in treatment_list)
        if not self.treatments:
            raise ValueError("At least one treatment flight is required.")
        if len(set(self.flights)) != len(self.flights):
            raise ValueError(f"Flights must be distinct, got {self.flights}.")
        self.period = Period(to_date(start), to_date(end))
        self.user_segments = normalize_segments(user_segments, "user segment")
        if mapping is not None and not isinstance(mapping, FlightMapping):
            raise TypeError(f"mapping must be a FlightMapping, got {type(mapping).__name__}.")
        self.mapping = mapping
        self.columns = columns or Columns()
        self.expected_traffic: Optional[Dict[str, float]] = None
        if expected_traffic is not None:
            traffic = {str(k): float(v) for k, v in dict(expected_traffic).items()}
            missing = [f for f in self.flights if f not in traffic]
            if missing:
                raise ValueError(f"expected_traffic has no share for flights {missing}.")
            if any(traffic[f] <= 0 for f in self.flights):
                raise ValueError("expected_traffic shares must be positive.")
            self.expected_traffic = {f: traffic[f] for f in self.flights}
        self.exclude_multi_flight_units = bool(exclude_multi_flight_units)
        self._spark = spark

    @property
    def flights(self) -> Tuple[str, ...]:
        """Control first, then the treatments."""
        return (self.control, *self.treatments)

    @property
    def start(self):
        return self.period.start

    @property
    def end(self):
        return self.period.end

    def __repr__(self) -> str:
        mode = "flight mapping" if self.mapping is not None else "pre-joined"
        return (f"Experiment(control={self.control!r}, treatments={list(self.treatments)!r}, "
                f"{self.start}..{self.end}, user_segments={[list(s) for s in self.user_segments]!r}, {mode})")

    # -------------------------------------------------------------------------------- analysis

    def analyze(
        self,
        metric_tables: MetricTables,
        *,
        daily: bool = True,
        alpha: float = 0.05,
        enforce_min_sample: bool = True,
        hte_alpha: float = 0.001,
        hte_min_units: float = 10_000,
        hte_scale: str = "absolute",
        cache: bool = True,
    ) -> ExperimentResults:
        """Compare every treatment with control on every metric, overall and by segment.

        Parameters
        ----------
        metric_tables:
            One MetricTable or several. Metric names must be unique across tables.
        daily:
            Also compute each day on its own (denominator: units exposed by that day).
        alpha:
            Significance level of the confidence intervals and of ``significant``.
        enforce_min_sample:
            Blank out confidence intervals and p-values where a metric's rule-of-thumb minimum sample
            is not met (``sufficient_sample`` is False). Point estimates are kept.
        hte_alpha, hte_min_units, hte_scale:
            Heterogeneous-treatment-effect test across user segment values: a segment value takes
            part when both flights have at least ``hte_min_units`` units; HTE is flagged when the
            p-value is below ``hte_alpha``. ``hte_scale`` is ``"absolute"`` (differences) or
            ``"relative"`` (percentage differences).
        cache:
            Persist each table's prepared rows (and the mapping) while it is analysed.
        """
        tables = self._check_tables(metric_tables)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if hte_scale not in ("absolute", "relative"):
            raise ValueError(f"hte_scale must be 'absolute' or 'relative', got {hte_scale!r}.")
        spark = self._session(tables)
        comparisons: List[pd.DataFrame] = []
        populations: List[pd.DataFrame] = []
        raw: Dict[str, pd.DataFrame] = {}
        mapping_population = None
        try:
            if self.mapping is not None:
                population, has_timestamps = self._mapping_population(spark)
                mapping_population = (population.persist() if cache else population, has_timestamps)
            for table in tables:
                self._analyze_table(
                    table, spark, mapping_population, daily, alpha, enforce_min_sample, cache,
                    comparisons, populations, raw,
                )
        finally:
            if cache and mapping_population is not None:
                mapping_population[0].unpersist()

        user_order = {"|".join(spec): i for i, spec in enumerate(self.user_segments)}
        metric_segment_order: Dict[str, int] = {}
        for table in tables:
            for spec in table.metric_segments:
                metric_segment_order.setdefault("|".join(spec), len(metric_segment_order))
        result = compare.finalize(
            pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame(),
            alpha=alpha,
            hte_alpha=hte_alpha,
            hte_min_units=hte_min_units,
            hte_scale=hte_scale,
            metric_order={(t.name, m.name): i for i, (t, m) in enumerate((t, m) for t in tables for m in t.metrics)},
            user_segment_order=user_order,
            metric_segment_order=metric_segment_order,
            treatment_order={f: i for i, f in enumerate(self.treatments)},
        )
        settings = dict(
            start=self.start, end=self.end, flights=self.flights, daily=daily, alpha=alpha,
            enforce_min_sample=enforce_min_sample, hte_alpha=hte_alpha, hte_min_units=hte_min_units,
            hte_scale=hte_scale, mode="mapping" if self.mapping is not None else "pre-joined",
        )
        return ExperimentResults(
            result,
            pd.concat(populations, ignore_index=True) if populations else pd.DataFrame(),
            raw,
            self.flights,
            self.expected_traffic,
            settings,
        )

    def _analyze_table(
        self,
        table: MetricTable,
        spark: SparkSession,
        mapping_population: Optional[Tuple[DataFrame, bool]],
        daily: bool,
        alpha: float,
        enforce_min_sample: bool,
        cache: bool,
        comparisons: List[pd.DataFrame],
        populations: List[pd.DataFrame],
        raw: Dict[str, pd.DataFrame],
    ) -> None:
        log.info("breakitdown: analysing %s (%d metrics)", table.name, len(table.metrics))
        prepared = self._prepare(table, spark, mapping_population)
        cached: List[DataFrame] = []
        if cache:
            prepared = aggregate.Prepared(prepared.rows.persist(), prepared.population.persist())
            cached = [prepared.rows, prepared.population]
        try:
            granularities = (CUMULATIVE, DAILY) if daily and table.daily else (CUMULATIVE,)
            counts = aggregate.population_counts(prepared.population, self.period, self.user_segments, DAILY in granularities)
            if counts.empty:
                log.warning("breakitdown: %s has no units in flights %s between %s and %s.",
                            table.name, self.flights, self.start, self.end)
            populations.append(counts.assign(metric_table=table.name))
            keyed = [(f"__m{i}", metric) for i, metric in enumerate(table.metrics)]
            jobs: List[Tuple[List[Tuple[str, Metric]], DataFrame]] = []
            unit_metrics = [(k, m) for k, m in keyed if isinstance(m, UnitMetric)]
            if unit_metrics:
                jobs.append((unit_metrics, aggregate.unit_metric_stats(
                    prepared, unit_metrics, self.period, self.user_segments, table.metric_segments, granularities)))
            for key, metric in keyed:
                if isinstance(metric, Quantile):
                    jobs.append(([(key, metric)], aggregate.quantile_stats(
                        prepared, key, metric, self.period, self.user_segments, table.metric_segments, granularities)))
            frames = []
            for keyed_part, job in jobs:
                sufficient_stats = aggregate.collect_pandas(job)
                frames.append(_readable(sufficient_stats, keyed_part))
                comparisons.append(compare.compare_table(
                    table.name, keyed_part, sufficient_stats, counts, self.flights,
                    alpha=alpha, enforce_min_sample=enforce_min_sample, expected_traffic=self.expected_traffic,
                ))
            raw[table.name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        finally:
            for df in cached:
                df.unpersist()

    def _prepare(
        self,
        table: MetricTable,
        spark: SparkSession,
        mapping_population: Optional[Tuple[DataFrame, bool]],
    ) -> aggregate.Prepared:
        df = load_source(table.source, spark)
        columns = table.columns or self.columns
        user_cols = segment_columns(self.user_segments)
        where = f"MetricTable {table.name!r}"
        if mapping_population is None:
            return aggregate.prepare_prejoined(
                df, columns, self.period, self.flights, user_cols, table.metric_segment_columns,
                table.input_columns, where, self.exclude_multi_flight_units,
            )
        population, has_timestamps = mapping_population
        return aggregate.prepare_joined(
            df, columns, self.period, population, has_timestamps, user_cols, table.metric_segment_columns,
            table.input_columns, table.inner_join, where,
        )

    def _mapping_population(self, spark: SparkSession) -> Tuple[DataFrame, bool]:
        assert self.mapping is not None
        df = load_source(self.mapping.source, spark)
        return aggregate.prepare_mapping(
            df, self.mapping.columns or self.columns, self.period, self.flights,
            segment_columns(self.user_segments), "FlightMapping", self.exclude_multi_flight_units,
        )

    def _check_tables(self, metric_tables: MetricTables) -> List[MetricTable]:
        tables = [metric_tables] if isinstance(metric_tables, MetricTable) else list(metric_tables)
        if not tables:
            raise ValueError("Pass at least one MetricTable.")
        for table in tables:
            if not isinstance(table, MetricTable):
                raise TypeError(f"Expected MetricTable objects, got {table!r}.")
        repeated = sorted(n for n, c in Counter(t.name for t in tables).items() if c > 1)
        if repeated:
            raise ValueError(f"Duplicate MetricTable names {repeated}; set name= on each table.")
        repeated = sorted(n for n, c in Counter(m.name for t in tables for m in t.metrics).items() if c > 1)
        if repeated:
            raise ValueError(f"Metric names must be unique across tables; repeated: {repeated}.")
        user_cols = set(segment_columns(self.user_segments))
        for table in tables:
            clash = user_cols & (set(table.metric_segment_columns) | set(table.input_columns))
            if clash:
                raise ValueError(
                    f"MetricTable {table.name!r}: columns {sorted(clash)} are user segments and also "
                    "metric inputs or metric segments."
                )
            if table.inner_join and self.mapping is None:
                log.warning("breakitdown: inner_join on %r has no effect without a FlightMapping.", table.name)
        return tables

    def _session(self, tables: Sequence[MetricTable]) -> SparkSession:
        if self._spark is not None:
            return self._spark
        sources = [t.source for t in tables] + ([self.mapping.source] if self.mapping is not None else [])
        for source in sources:
            if isinstance(source, DataFrame):
                return source.sparkSession
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

    # ------------------------------------------------------------------------------ validation

    def validate(self, table: MetricTable) -> pd.DataFrame:
        """Data checks for one MetricTable, as a small table of (check, value, ok).

        Pre-joined mode checks the table: units in the date range, units seen in more than one
        flight (left out of the analysis unless ``exclude_multi_flight_units`` is False), units whose
        user segment values change between rows (the analysis uses the value on their earliest row),
        and flights with no units. Mapping mode runs the same checks on the mapping and counts metric
        rows whose unit is not mapped.
        """
        spark = self._session([table])
        user_cols = segment_columns(self.user_segments)
        checks: List[Tuple[str, Any, Optional[bool]]] = []
        if self.mapping is None:
            rows = aggregate.prejoined_rows(
                load_source(table.source, spark), table.columns or self.columns, self.period, self.flights,
                user_cols, table.metric_segment_columns, table.input_columns, f"MetricTable {table.name!r}",
            )
            checks.append(("rows in date range", rows.count(), None))
        else:
            mapping_columns = self.mapping.columns or self.columns
            rows = aggregate.mapping_rows(
                load_source(self.mapping.source, spark), mapping_columns, None, self.flights, user_cols, "FlightMapping"
            )
            columns = table.columns or self.columns
            metric_df = load_source(table.source, spark)
            aggregate.require_columns(metric_df, [columns.unit, columns.time], f"MetricTable {table.name!r}")
            day = aggregate.day_column(metric_df, columns.time, columns.time_format)
            in_range = metric_df.where(day.between(F.lit(self.start), F.lit(self.end))).select(
                F.col(quote(columns.unit)).alias(UNIT)
            )
            unmatched = in_range.join(rows.select(UNIT).distinct(), on=UNIT, how="left_anti")
            checks.append(("metric rows in date range", in_range.count(), None))
            checks.append(("metric rows whose unit is not in the mapping", unmatched.count(), None))
        per_unit = rows.groupBy(UNIT).agg(
            F.countDistinct(FLIGHT).alias("__flights"),
            *[F.countDistinct(F.col(quote(c))).alias(f"__s{i}") for i, c in enumerate(user_cols)],
        )
        summary = per_unit.agg(
            F.count(F.lit(1)).alias("units"),
            F.sum(F.when(F.col("__flights") > 1, 1).otherwise(0)).alias("multi"),
            *[F.sum(F.when(F.col(f"__s{i}") > 1, 1).otherwise(0)).alias(f"s{i}") for i in range(len(user_cols))],
        ).first()
        units = int(summary["units"] or 0)
        multi = int(summary["multi"] or 0)
        checks.append(("units", units, units > 0))
        # Left out of the analysis by default, so then just for information; counted twice, a problem.
        checks.append(("units in more than one flight", multi, None if self.exclude_multi_flight_units else multi == 0))
        for i, col in enumerate(user_cols):
            inconsistent = int(summary[f"s{i}"] or 0)
            checks.append((f"units with more than one {col!r} value", inconsistent, inconsistent == 0))
        observed = {r[0] for r in rows.select(FLIGHT).distinct().collect()}
        missing = [f for f in self.flights if f not in observed]
        checks.append(("flights without units", ", ".join(missing) or "none", not missing))
        return pd.DataFrame(checks, columns=["check", "value", "ok"])


def _readable(sufficient_stats: pd.DataFrame, keyed_metrics: Sequence[Tuple[str, Metric]]) -> pd.DataFrame:
    """Rename internal ``__m<i>`` statistic prefixes to metric names."""
    renames = {}
    for key, metric in keyed_metrics:
        for col in sufficient_stats.columns:
            if col.startswith(f"{key}__"):
                renames[col] = metric.name + col[len(key):]
    return sufficient_stats.rename(columns=renames)

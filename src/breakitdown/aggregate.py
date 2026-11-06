"""Spark side of the analysis: prepare rows, derive the population, reduce to sufficient statistics.

Per MetricTable, everything stays lazy until a single collect per job::

    source --> rows        one schema: __unit, __flight, __day, segment columns, metric inputs
           +-> population  one row per unit and flight: user segments and first exposure day

    rows --> unit-level rollup (unit [x day] [x metric segment]) --> sufficient statistics
    rows --> quantile per group --> per-unit row and above-quantile counts --> sufficient statistics
    population --> units per flight x user segment [x day]: the denominators

Sufficient statistics are sums over units *with* rows; the denominators come from the population,
so units without rows count as zeros without ever being materialised.

Pre-joined mode builds both ``rows`` and ``population`` from the metric table. Mapping mode joins
the metric rows to the mapping and takes the population from the mapping.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ._common import NULL_LABEL, OVERALL, SEPARATOR, Period, SegmentSpec, quote, segment_columns, segment_name
from .columns import Columns
from .metrics import Quantile, QuantileMethod, UnitMetric

UNIT = "__unit"
FLIGHT = "__flight"
DAY = "__day"
TS = "__ts"
EXPOSURE_DAY = "__exposure_day"
ORDER = "__order"  # a row's time, for finding each unit's earliest row
EXPOSURE_TS = "__exposure_ts"
FIRST_DAY = "__first_day"
VALUE = "__value"

CUMULATIVE = "cumulative"
DAILY = "daily"

_TIMESTAMP_TYPES = tuple(t for t in (T.TimestampType, getattr(T, "TimestampNTZType", None)) if t is not None)
_INTEGRAL_TYPES = (T.ByteType, T.ShortType, T.IntegerType, T.LongType)


@dataclass
class Prepared:
    rows: DataFrame
    """UNIT, FLIGHT, DAY, user segment columns, metric segment columns, metric input columns."""
    population: DataFrame
    """UNIT, FLIGHT, FIRST_DAY, user segment columns: one row per unit and flight."""


# --------------------------------------------------------------------------------------------- helpers


def _cols(names: Iterable[str]) -> List[Column]:
    return [F.col(quote(name)) for name in names]


def _union(parts: Sequence[DataFrame]) -> DataFrame:
    return functools.reduce(lambda a, b: a.unionByName(b), parts)


def collect_pandas(df: DataFrame) -> pd.DataFrame:
    """Collect a small aggregated DataFrame into pandas.

    A plain ``collect()`` rather than ``toPandas()``: results are small, and this keeps column types
    the same whatever the session's Arrow settings (and avoids ``toPandas``' distutils import, which
    fails on Python 3.12 with PySpark 3.5).
    """
    return pd.DataFrame.from_records([tuple(r) for r in df.collect()], columns=df.columns)


def _field_type(df: DataFrame, name: str) -> T.DataType:
    fields = df.schema.fields
    for field in fields:
        if field.name == name:
            return field.dataType
    for field in fields:
        if field.name.lower() == name.lower():
            return field.dataType
    raise KeyError(name)


def require_columns(df: DataFrame, columns: Iterable[str], where: str) -> None:
    available = {c.lower() for c in df.columns}
    missing = [c for c in dict.fromkeys(columns) if c.lower() not in available]
    if missing:
        raise ValueError(f"{where} is missing column(s) {missing}. Available columns: {df.columns}")


def day_column(df: DataFrame, name: str, time_format: Optional[str]) -> Column:
    """Calendar day of a DATE, TIMESTAMP, string or yyyyMMdd-integer column."""
    dtype = _field_type(df, name)
    col = F.col(quote(name))
    if isinstance(dtype, T.DateType):
        return col
    if isinstance(dtype, _TIMESTAMP_TYPES):
        return F.to_date(col)
    if isinstance(dtype, T.StringType):
        return F.to_date(col, time_format) if time_format else F.to_date(col)
    if isinstance(dtype, _INTEGRAL_TYPES):
        return F.to_date(col.cast("string"), time_format or "yyyyMMdd")
    raise TypeError(f"Column {name!r} has type {dtype.simpleString()}; expected date, timestamp, string or integer.")


def is_timestamp(df: DataFrame, name: str) -> bool:
    return isinstance(_field_type(df, name), _TIMESTAMP_TYPES)


def _segment(col: str) -> Column:
    """A segment column as a non-null string: NULL becomes its own ``(null)`` segment."""
    return F.coalesce(F.col(quote(col)).cast("string"), F.lit(NULL_LABEL)).alias(col)


def _in_period(day: Column, period: Period) -> Column:
    return day.between(F.lit(period.start), F.lit(period.end))


# ----------------------------------------------------------------------------------- preparation


def prejoined_rows(
    df: DataFrame,
    columns: Columns,
    period: Period,
    flights: Sequence[str],
    user_cols: Sequence[str],
    metric_segment_cols: Sequence[str],
    inputs: Sequence[str],
    where: str,
) -> DataFrame:
    """Rows of a pre-joined table in the date range and the experiment's flights, with internal names.

    User segment columns keep their raw values (NULLs included); ``prepare_prejoined`` reduces them to
    one value per unit. ORDER (the row's time) and, with an exposure column, EXPOSURE_DAY ride along.
    """
    exposure = columns.exposure
    require_columns(
        df,
        [columns.unit, columns.flight, columns.time, *([exposure] if exposure else []), *user_cols,
         *metric_segment_cols, *inputs],
        where,
    )
    day = day_column(df, columns.time, columns.time_format)
    flight = F.col(quote(columns.flight)).cast("string")
    keep = _in_period(day, period) & flight.isin(list(flights)) & F.col(quote(columns.unit)).isNotNull()
    order = F.col(quote(columns.time)) if is_timestamp(df, columns.time) else day
    selects = [F.col(quote(columns.unit)).alias(UNIT), flight.alias(FLIGHT), day.alias(DAY), order.alias(ORDER)]
    if exposure:
        exposure_day = day_column(df, exposure, columns.time_format)
        if is_timestamp(df, columns.time) and is_timestamp(df, exposure):
            after = F.col(quote(columns.time)) >= F.col(quote(exposure))
        else:
            after = day >= exposure_day
        keep = keep & (F.col(quote(exposure)).isNull() | after)
        selects.append(exposure_day.alias(EXPOSURE_DAY))
    selects += [F.col(quote(c)).cast("string").alias(c) for c in user_cols]
    selects += [_segment(c) for c in metric_segment_cols]
    selects += [F.col(quote(c)).alias(c) for c in inputs]
    return df.where(keep).select(*selects)


def _earliest_value(col: str) -> Column:
    """A unit's value of ``col`` on its earliest row, as one aggregate (no sort, no window).

    Ties at the same time go to the smallest value, so the result is deterministic. NULLs are skipped,
    and the unit lands in the ``(null)`` segment only when every row is NULL.
    """
    value = F.col(quote(col))
    first = F.min(F.when(value.isNotNull(), F.struct(F.col(ORDER).alias("t"), value.alias("v"))))
    return F.coalesce(first.getField("v"), F.lit(NULL_LABEL)).alias(col)


def prepare_prejoined(
    df: DataFrame,
    columns: Columns,
    period: Period,
    flights: Sequence[str],
    user_cols: Sequence[str],
    metric_segment_cols: Sequence[str],
    inputs: Sequence[str],
    where: str,
) -> Prepared:
    """Rows and population of a table that already carries flight and user segments on every row.

    Every unit gets exactly one value per user segment column, taken from its earliest row (the value
    closest to exposure). Rows then carry that value, so a breakdown never splits a unit and adding one
    never changes the overall results.
    """
    raw = prejoined_rows(df, columns, period, flights, user_cols, metric_segment_cols, inputs, where)
    first_day = F.coalesce(F.min(EXPOSURE_DAY), F.min(DAY)) if columns.exposure else F.min(DAY)
    population = (
        raw.groupBy(UNIT, FLIGHT)
        .agg(first_day.alias(FIRST_DAY), *[_earliest_value(c) for c in user_cols])
        .withColumn(FIRST_DAY, F.greatest(F.col(FIRST_DAY), F.lit(period.start)))
    )
    rows = raw.drop(ORDER, *([EXPOSURE_DAY] if columns.exposure else []), *user_cols)
    if user_cols:
        rows = rows.join(population.select(*_cols((UNIT, FLIGHT, *user_cols))), on=[UNIT, FLIGHT], how="inner")
    return Prepared(rows, population)


def mapping_rows(df: DataFrame, columns: Columns, flights: Sequence[str], user_cols: Sequence[str], where: str) -> DataFrame:
    """The mapping's rows for the experiment's flights, with internal column names."""
    exposure = columns.exposure
    require_columns(df, [columns.unit, columns.flight, *([exposure] if exposure else []), *user_cols], where)
    flight = F.col(quote(columns.flight)).cast("string")
    selects = [F.col(quote(columns.unit)).alias(UNIT), flight.alias(FLIGHT)]
    selects += [F.col(quote(c)).cast("string").alias(c) for c in user_cols]
    if exposure:
        selects.append(day_column(df, exposure, columns.time_format).alias(EXPOSURE_DAY))
        if is_timestamp(df, exposure):
            selects.append(F.col(quote(exposure)).alias(EXPOSURE_TS))
    keep = flight.isin(list(flights)) & F.col(quote(columns.unit)).isNotNull()
    return df.where(keep).select(*selects)


def prepare_mapping(
    df: DataFrame,
    columns: Columns,
    period: Period,
    flights: Sequence[str],
    user_cols: Sequence[str],
    where: str,
) -> Tuple[DataFrame, bool]:
    """Population from a flight mapping. Returns (population, has exact exposure timestamps)."""
    rows = mapping_rows(df, columns, flights, user_cols, where)
    has_day = EXPOSURE_DAY in rows.columns
    precise = EXPOSURE_TS in rows.columns
    aggs = [F.max(F.col(quote(c))).alias(c) for c in user_cols]
    aggs.append((F.min(EXPOSURE_DAY) if has_day else F.lit(None).cast("date")).alias(FIRST_DAY))
    if precise:
        aggs.append(F.min(EXPOSURE_TS).alias(EXPOSURE_TS))
    population = rows.groupBy(UNIT, FLIGHT).agg(*aggs)
    population = population.where(F.col(FIRST_DAY).isNull() | (F.col(FIRST_DAY) <= F.lit(period.end)))
    population = population.withColumn(
        FIRST_DAY, F.greatest(F.coalesce(F.col(FIRST_DAY), F.lit(period.start)), F.lit(period.start))
    )
    for c in user_cols:
        population = population.withColumn(c, F.coalesce(F.col(quote(c)), F.lit(NULL_LABEL)))
    return population, precise


def prepare_joined(
    df: DataFrame,
    columns: Columns,
    period: Period,
    mapping_population: DataFrame,
    mapping_has_timestamps: bool,
    user_cols: Sequence[str],
    metric_segment_cols: Sequence[str],
    inputs: Sequence[str],
    inner_join: bool,
    where: str,
) -> Prepared:
    """Rows of a metric table joined to the mapping; the mapping (or its units with rows) is the population."""
    require_columns(df, [columns.unit, columns.time, *metric_segment_cols, *inputs], where)
    day = day_column(df, columns.time, columns.time_format)
    precise = mapping_has_timestamps and is_timestamp(df, columns.time)
    selects = [F.col(quote(columns.unit)).alias(UNIT), day.alias(DAY)]
    if precise:
        selects.append(F.col(quote(columns.time)).alias(TS))
    selects += [_segment(c) for c in metric_segment_cols]
    selects += [F.col(quote(c)).alias(c) for c in inputs]
    metric_rows = df.where(_in_period(day, period) & F.col(quote(columns.unit)).isNotNull()).select(*selects)
    mapped = mapping_population.select(*_cols((UNIT, FLIGHT, FIRST_DAY, *user_cols, *([EXPOSURE_TS] if precise else []))))
    joined = metric_rows.join(mapped, on=UNIT, how="inner")
    if precise:
        after = F.col(EXPOSURE_TS).isNull() | (F.col(TS) >= F.col(EXPOSURE_TS))
    else:
        after = F.col(DAY) >= F.col(FIRST_DAY)
    rows = joined.where(after).select(*_cols((UNIT, FLIGHT, DAY, *user_cols, *metric_segment_cols, *inputs)))
    population = mapping_population.select(*_cols((UNIT, FLIGHT, FIRST_DAY, *user_cols)))
    if inner_join:
        population = population.join(rows.select(UNIT, FLIGHT).distinct(), on=[UNIT, FLIGHT], how="left_semi")
    return Prepared(rows, population)


# ------------------------------------------------------------------------------- sufficient statistics


def _label(
    df: DataFrame,
    period: Period,
    granularity: str,
    user_spec: SegmentSpec,
    metric_spec: SegmentSpec,
    stat_names: Sequence[str],
) -> DataFrame:
    """Put a grouped result into the common schema so all groupings can be unioned and collected once."""

    def value(spec: SegmentSpec) -> Column:
        return F.concat_ws(SEPARATOR, *_cols(spec)) if spec else F.lit(OVERALL)

    day = F.col(DAY) if granularity == DAILY else F.lit(None).cast("date")
    return df.select(
        F.lit(period.name).alias("period"),
        F.lit(granularity).alias("granularity"),
        day.alias("day"),
        F.lit(segment_name(user_spec)).alias("user_segment"),
        value(user_spec).alias("user_segment_value"),
        F.lit(segment_name(metric_spec)).alias("metric_segment"),
        value(metric_spec).alias("metric_segment_value"),
        F.col(FLIGHT).alias("flight"),
        F.col("__units_with_data").alias("units_with_data"),
        *_cols(stat_names),
    )


def unit_metric_stats(
    prepared: Prepared,
    keyed_metrics: Sequence[Tuple[str, UnitMetric]],
    period: Period,
    user_specs: Sequence[SegmentSpec],
    metric_specs: Sequence[SegmentSpec],
    granularities: Sequence[str],
) -> DataFrame:
    """Sufficient statistics of all Count/Binary/Ratio metrics of a table, for every grouping."""
    user_cols = segment_columns(user_specs)
    unit_aggs: Dict[str, Column] = {}
    group_aggs: Dict[str, Column] = {}
    for key, metric in keyed_metrics:
        unit_aggs.update(metric.unit_aggregations(key))
        group_aggs.update(metric.group_aggregations(key))
    parts = []
    for metric_spec in ((),) + tuple(metric_specs):
        for granularity in granularities:
            day = (DAY,) if granularity == DAILY else ()
            # User segment columns are unit attributes, so keeping them all in the unit key does not
            # change its grain; every user-segment grouping below reuses this one rollup.
            units = prepared.rows.groupBy(*_cols((UNIT, FLIGHT, *user_cols, *metric_spec, *day))).agg(
                *[expr.alias(name) for name, expr in unit_aggs.items()]
            )
            for user_spec in ((),) + tuple(user_specs):
                grouped = units.groupBy(*_cols((FLIGHT, *user_spec, *metric_spec, *day))).agg(
                    F.count(F.lit(1)).alias("__units_with_data"),
                    *[expr.alias(name) for name, expr in group_aggs.items()],
                )
                parts.append(_label(grouped, period, granularity, user_spec, metric_spec, list(group_aggs)))
    return _union(parts)


def _approx_quantiles(events: DataFrame, keys: Sequence[str], metric: Quantile) -> DataFrame:
    """REGULAR method: percentile_approx at p and p +/- bandwidth."""
    h = metric.bandwidth
    probabilities = [x for p in metric.quantiles for x in (p - h, p, p + h)]
    table = events.groupBy(*_cols(keys)).agg(
        F.percentile_approx(VALUE, probabilities, metric.accuracy).alias("__qs")
    )
    selects = _cols(keys)
    for i in range(len(metric.quantiles)):
        low, mid, high = (F.col("__qs")[3 * i + j] for j in range(3))
        selects += [mid.alias(f"__q{i}"), _density(low, high, h).alias(f"__f{i}")]
    return table.select(*selects)


def _interpolated_quantiles(events: DataFrame, keys: Sequence[str], metric: Quantile) -> DataFrame:
    """DENSITY method: exact histogram with the CDF interpolated between neighbouring values.

    Each distinct value v spreads over [start, end]: halfway to its neighbours, but at most
    ``resolution`` away (so at most resolution / 2 on each side); the smallest and the largest value
    do not spread outward. The quantile for probability f is found in the first value whose
    cumulative count reaches ``f * total`` and interpolated linearly inside its interval.
    """
    h, r = metric.bandwidth, metric.resolution
    targets = [(3 * i + j, float(p + d)) for i, p in enumerate(metric.quantiles) for j, d in enumerate((-h, 0.0, h))]
    target_df = events.sparkSession.createDataFrame(targets, "__tid INT, __target DOUBLE")
    by_value = Window.partitionBy(*_cols(keys)).orderBy(VALUE)
    whole = Window.partitionBy(*_cols(keys))
    v = F.col(VALUE)
    previous, following = F.lag(v).over(by_value), F.lead(v).over(by_value)
    histogram = events.groupBy(*_cols(keys), v).agg(F.count(F.lit(1)).alias("__c")).select(
        *_cols(keys),
        "__c",
        F.sum("__c").over(by_value.rowsBetween(Window.unboundedPreceding, Window.currentRow)).alias("__cum"),
        F.sum("__c").over(whole).alias("__total"),
        ((F.when(previous.isNull(), v).otherwise(F.greatest(previous, v - r)) + v) / 2).alias("__start"),
        ((F.when(following.isNull(), v).otherwise(F.least(following, v + r)) + v) / 2).alias("__end"),
    )
    reached = histogram.crossJoin(F.broadcast(target_df)).where(F.col("__cum") >= F.col("__target") * F.col("__total"))
    first = reached.groupBy(*_cols(keys), "__tid").agg(
        F.min(F.struct("__cum", "__c", "__start", "__end", "__total", "__target")).alias("__s")
    )
    s = F.col("__s")
    quantile = s["__start"] + (s["__target"] * s["__total"] - (s["__cum"] - s["__c"])) / s["__c"] * (
        s["__end"] - s["__start"]
    )
    first = first.select(*_cols(keys), "__tid", quantile.alias("__x"))

    def at(tid: int) -> Column:
        return F.max(F.when(F.col("__tid") == tid, F.col("__x")))

    aggs = []
    for i in range(len(metric.quantiles)):
        aggs += [at(3 * i + 1).alias(f"__q{i}"), _density(at(3 * i), at(3 * i + 2), h).alias(f"__f{i}")]
    return first.groupBy(*_cols(keys)).agg(*aggs)


def _density(low: Column, high: Column, bandwidth: float) -> Column:
    """2h / (Q(p+h) - Q(p-h)); NULL when the two quantiles tie and the density is undefined."""
    width = high - low
    return F.when(width > 0, F.lit(2 * bandwidth) / width)


def quantile_stats(
    prepared: Prepared,
    key: str,
    metric: Quantile,
    period: Period,
    user_specs: Sequence[SegmentSpec],
    metric_specs: Sequence[SegmentSpec],
    granularities: Sequence[str],
) -> DataFrame:
    """Sufficient statistics of one Quantile metric, for every grouping."""
    user_cols = segment_columns(user_specs)
    metric_cols = segment_columns(metric_specs)
    raw = F.col(quote(metric.col))
    events = prepared.rows.where(raw.isNotNull() & ~F.isnan(raw.cast("double"))).select(
        *_cols((UNIT, FLIGHT, DAY, *user_cols, *metric_cols)), metric.value().alias(VALUE)
    )
    find_quantiles = _interpolated_quantiles if metric.method == QuantileMethod.DENSITY else _approx_quantiles
    outputs = metric.outputs(key)
    parts = []
    for metric_spec in ((),) + tuple(metric_specs):
        for granularity in granularities:
            day = (DAY,) if granularity == DAILY else ()
            for user_spec in ((),) + tuple(user_specs):
                keys = (FLIGHT, *user_spec, *metric_spec, *day)
                quantiles = find_quantiles(events, keys, metric)
                joined = events.join(F.broadcast(quantiles), on=list(keys), how="inner")
                unit_aggs = [F.count(F.lit(1)).alias("__n")]
                for i in range(len(outputs)):
                    above = F.when(F.col(VALUE) > F.col(f"__q{i}"), F.lit(1.0)).otherwise(F.lit(0.0))
                    unit_aggs += [F.sum(above).alias(f"__z{i}"), F.max(f"__q{i}").alias(f"__q{i}"),
                                  F.max(f"__f{i}").alias(f"__f{i}")]
                units = joined.groupBy(*_cols((UNIT, *keys))).agg(*unit_aggs)
                n = F.col("__n").cast("double")
                group_aggs: Dict[str, Column] = {}
                for i, output in enumerate(outputs):
                    z = F.col(f"__z{i}")
                    p = output.prefix
                    group_aggs.update({
                        f"{p}__q": F.max(f"__q{i}"),
                        f"{p}__density": F.max(f"__f{i}"),
                        f"{p}__events": F.sum(n),
                        f"{p}__events2": F.sum(n * n),
                        f"{p}__above": F.sum(z),
                        f"{p}__above2": F.sum(z * z),
                        f"{p}__above_events": F.sum(z * n),
                    })
                grouped = units.groupBy(*_cols(keys)).agg(
                    F.count(F.lit(1)).alias("__units_with_data"),
                    *[expr.alias(name) for name, expr in group_aggs.items()],
                )
                parts.append(_label(grouped, period, granularity, user_spec, metric_spec, list(group_aggs)))
    return _union(parts)


# ------------------------------------------------------------------------------------- denominators


def population_counts(
    population: DataFrame,
    period: Period,
    user_specs: Sequence[SegmentSpec],
    daily: bool,
) -> pd.DataFrame:
    """Units per flight and user segment: overall (cumulative) and exposed-by-day (daily)."""
    parts = []
    for spec in ((),) + tuple(user_specs):
        value = F.concat_ws(SEPARATOR, *_cols(spec)) if spec else F.lit(OVERALL)
        parts.append(
            population.groupBy(*_cols((FLIGHT, *spec, FIRST_DAY)))
            .agg(F.count(F.lit(1)).alias("n"))
            .select(
                F.lit(segment_name(spec)).alias("user_segment"),
                value.alias("user_segment_value"),
                F.col(FLIGHT).alias("flight"),
                F.col(FIRST_DAY).alias("first_day"),
                "n",
            )
        )
    return expand_counts(collect_pandas(_union(parts)), period, daily)


def expand_counts(by_first_day: pd.DataFrame, period: Period, daily: bool) -> pd.DataFrame:
    """Turn unit counts by first exposure day into cumulative and per-day denominators."""
    keys = ["user_segment", "user_segment_value", "flight"]
    columns = ["period", "granularity", "day", *keys, "n"]
    if by_first_day.empty:
        return pd.DataFrame(columns=columns)
    counts = by_first_day.assign(n=by_first_day["n"].astype(float), first_day=pd.to_datetime(by_first_day["first_day"]))
    cumulative = counts.groupby(keys, as_index=False)["n"].sum().assign(granularity=CUMULATIVE, day=pd.NaT)
    frames = [cumulative]
    if daily:
        days = pd.DatetimeIndex(pd.to_datetime(period.days), name="day")
        wide = counts.pivot_table(index=keys, columns="first_day", values="n", aggfunc="sum", fill_value=0.0)
        wide = wide.reindex(columns=days, fill_value=0.0).cumsum(axis=1)
        wide.columns.name = "day"
        frames.append(wide.reset_index().melt(id_vars=keys, var_name="day", value_name="n").assign(granularity=DAILY))
    out = pd.concat(frames, ignore_index=True).assign(period=period.name)
    out["day"] = pd.to_datetime(out["day"])
    return out[columns]

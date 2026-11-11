# breakitdown

**Break it down: A/B tests on Spark, sliced any way you like.**

breakitdown compares the flights (variants) of an A/B test on data in Spark. It breaks every result
down by user segment (os, country, or any combination), by metric segment (say, clicks on photos vs
clicks on videos) and by day, all through one small Python API. It works on the tables you already
have: if they carry the flight on every row, nothing needs to be joined.

* **Any breakdown, no SQL.** Name user segments on the experiment and metric segments on a table.
  Every combination is computed and tested, each with the right denominator.
* **Four metric kinds.** Per-user counts, binary rates, ratios and quantiles, all with unit-level
  (clustered) variances.
* **Tidy output.** One pandas DataFrame with one row per comparison, ready for notebooks and dashboards.
* **Built for Spark.** Each table is reduced to sufficient statistics in Spark. Only those reach the
  driver.

## Install

```bash
pip install breakitdown            # on a Spark cluster: uses the cluster's PySpark
pip install "breakitdown[spark]"   # anywhere else: installs PySpark too
```

Requires Python 3.9+ and PySpark 3.3+, and is tested with PySpark 3.5 and 4.3. Running Spark locally
needs Java 17.

## Quick start

The Spark table `analytics.app_events` has one row per user, day and media type. Its `flight` column
holds each row's flight assignment, here `control`, `t1` or `t2`. Any grain at or below user × day works,
and the flight column can also list several flights: see
[One table for many experiments](#one-table-for-many-experiments).

| ruid | flight  | event_time | os      | country | media_type | clicks | sessions |
|------|---------|------------|---------|---------|------------|-------:|---------:|
| u1   | control | 2024-03-01 | ios     | US      | photo      |      3 |        2 |
| u1   | control | 2024-03-01 | ios     | US      | video      |      1 |        0 |
| u2   | t1      | 2024-03-02 | android | GB      | photo      |      0 |        1 |

```python
from breakitdown import Binary, Count, Experiment, MetricTable, Ratio

exp = Experiment(control="control", treatments=["t1", "t2"], start="2024-03-01", end="2024-03-14")

events = MetricTable("analytics.app_events", [
    Count("clicks"),                                         # mean clicks per user
    Binary("clicks", name="clickers"),                       # share of users with any click
    Ratio("clicks", "sessions", name="clicks_per_session"),  # sum of clicks / sum of sessions
])

results = exp.analyze(events)
results.overall()
```

```text
            metric treatment n_control n_treatment value_control value_treatment pct_diff pct_diff_ci_lower pct_diff_ci_upper p_value outcome
            clicks        t1      1003        1020         1.299           1.523   +17.20             +6.15            +28.25  0.0010  better
            clicks        t2      1003         977         1.299           1.294    -0.41            -10.05             +9.23  0.9334    flat
          clickers        t1      1003        1020        0.6271          0.6696    +6.78             -0.10            +13.65  0.0454  better
          clickers        t2      1003         977        0.6271          0.6469    +3.15             -3.72            +10.02  0.3609    flat
clicks_per_session        t1      1003        1020        0.1493          0.1822   +22.02            +12.90            +31.14 <0.0001  better
clicks_per_session        t2      1003         977        0.1493          0.1524    +2.08             -5.91            +10.08  0.6059    flat
```

* **Rows:** each row compares one treatment with control. CIs are 95% (`analyze(alpha=...)`).
  The [Methods](#methods) section explains the `clickers` t1 row, whose relative CI just includes 0
  while p < 0.05.
* **Everything else:** `results.table` holds every comparison, including segments and days. In a
  notebook, `results` renders as a colour-coded scorecard.
* **Column names:** the defaults are `ruid` (randomization unit), `flight` and `event_time`; see
  [Input contract](#input-contract).

## Examples

Each example adds one idea to the quick start.

### Break it down: user segments and metric segments

```python
exp = Experiment("control", ["t1", "t2"], "2024-03-01", "2024-03-14",
                 user_segments=["os", ("os", "country")])
events = MetricTable("analytics.app_events", [Count("clicks")], metric_segments=["media_type"])
results = exp.analyze(events)

results.by_user_segment("os")            # one row per os value and treatment
results.by_metric_segment("media_type")  # clicks on photos and on videos, per user in the flight
results.hte()                            # does the effect differ across os values?
results.daily("clicks")                  # each day on its own
```

* **User segments** split *users*: each user has one `os`.
* **Metric segments** split a metric's *rows*: one user can click both photos and videos. The
  denominator stays every user in the flight, so the segment values of a Count add up to its overall
  value.
* **Combinations:** every user segment is crossed with every metric segment.
* **Daily results:** each day's denominator is the users exposed by that day.

### Quantiles, labels and the desired direction

```python
from breakitdown import Quantile

latency = MetricTable(
    "analytics.request_latency",  # one row per request
    [Quantile("latency_ms", [0.5, 0.9], method="density", label="Latency", direction="down")],
)
results = exp.analyze([events, latency])
```

* `direction` (`"up"`, `"down"` or `"neutral"`) decides whether a significant change is `better` or
  `worse` in `outcome`.
* Every metric has a `name` (a unique id, defaulting to the column) and a `label` (display text).
* Quantiles pool all rows of a flight, with users as clusters. `method="density"` suits
  integer-valued metrics; see [Methods](#methods).

### Thresholds and per-user ratios

```python
from breakitdown import MeanRatio

engagement = MetricTable("analytics.app_events", [
    Binary("sessions", name="returning_users", min_sum=2),                # users with 2+ sessions
    MeanRatio("clicks", "sessions", name="clicks_per_session_per_user"),  # each user's own ratio, averaged
], name="engagement")
results = exp.analyze([events, engagement])
```

* `Binary(col, min_sum=k)` counts a user whose `col` adds up to at least k, instead of any positive
  value. With a 0/1 column on a user × day table, `min_sum=2` is the share of users active on two or
  more days.
* `MeanRatio(numerator, denominator)` averages each user's own ratio, over the users whose
  denominator isn't zero, and tests it with Welch's t-test. `Ratio` divides the totals instead, so
  heavy users weigh more.

### One table for many experiments

Many teams log one table for all their experiments and tag every row with each flight its user is in,
as a comma-separated list. Set `flight_separator` and each experiment reads the rows that list one of
its own flights, skipping the rest. Flight names must then be unique across experiments.

| ruid | flight           | event_time | clicks |
|------|------------------|------------|-------:|
| u1   | search_c,feed_t1 | 2024-03-01 |      3 |
| u2   | search_t1,feed_c | 2024-03-01 |      0 |

```python
from breakitdown import Columns

shared = MetricTable("analytics.shared_events", [Count("clicks")], columns=Columns(flight_separator=","))
search = Experiment(control="search_c", treatments=["search_t1"], start="2024-03-01", end="2024-03-14")
feed = Experiment(control="feed_c", treatments=["feed_t1"], start="2024-03-01", end="2024-03-14")
search.analyze(shared).overall()
feed.analyze(shared).overall()
```

An array column is read as a list without the setting. Rows that name none of the experiment's flights
are dropped with a cheap text test before any splitting. Without a separator, breakitdown expects one
flight per row and matches it exactly, so Spark can skip files and partitions that hold other flights.

### Joining a flight mapping (opt-in)

Use this when metric tables don't carry the flight, or when users without rows must count as zeros:

```python
from breakitdown import Columns, FlightMapping

exp = Experiment(
    "control", ["t1", "t2"], "2024-03-01", "2024-03-14",
    mapping=FlightMapping("analytics.flight_assignment", Columns(exposure="exposure_time")),
)
purchases = MetricTable("analytics.purchases", [Count("revenue"), Binary("revenue", name="buyers")])
results = exp.analyze(purchases)
```

`analytics.purchases` needs only `ruid`, `event_time` and `revenue`. Every mapped user counts:
mapped users with no purchases count as zero revenue. Rows before a user's `exposure_time` are dropped.

The mapping is read up to `end`: users exposed later are dropped before any grouping, with a filter
Spark can push down to the source. Everyone exposed earlier stays in, even before `start`, so a window
that starts late still covers the whole experiment. To read the mapping your own way, give it a SQL
query, say to keep only users exposed since the window began:

```python
recent = Experiment("control", ["t1", "t2"], "2024-03-08", "2024-03-14", mapping=FlightMapping(
    "SELECT * FROM analytics.flight_assignment WHERE exposure_time >= '2024-03-08'",
    Columns(exposure="exposure_time"),
))
recent.analyze(purchases).overall()
```

### Other column names and date formats

```python
exp = Experiment(
    "control", ["t1"], "2024-03-01", "2024-03-14",
    columns=Columns(unit="user_id", flight="variant", time="ds", time_format="yyyyMMdd"),
)
results = exp.analyze(MetricTable("analytics.daily_user_stats", [Count("clicks")]))
```

### SQL sources, data checks and sample ratio mismatch

```python
exp = Experiment("control", ["t1", "t2"], "2024-03-01", "2024-03-14",
                 expected_traffic={"control": 1, "t1": 1, "t2": 1})
ios = MetricTable("SELECT * FROM analytics.app_events WHERE os = 'ios'", [Count("clicks")], name="ios_events")

exp.validate(ios)  # users in two flights, segment values that change, flights without users
results = exp.analyze(ios)
results.srm()      # one test per control/treatment pair
```

A source can be a table name, a SQL query or a Spark DataFrame. Don't pre-filter dates: breakitdown
filters the time column itself, and Spark prunes partitions on it.

## API at a glance

```text
Experiment(control, treatments, start, end, *, user_segments=None, mapping=None, columns=None,
           expected_traffic=None, exclude_multi_flight_units=True, spark=None)
    .analyze(metric_tables, *, daily=True, alpha=0.05, enforce_min_sample=True,
             hte_alpha=0.001, hte_min_units=10_000, hte_scale="absolute", cache=True) -> ExperimentResults
    .validate(metric_table) -> pandas.DataFrame

MetricTable(source, metrics, name="", metric_segments=None, columns=None, inner_join=False, daily=True)
FlightMapping(source, columns=None)
Columns(unit="ruid", flight="flight", time="event_time", exposure=None, time_format=None, flight_separator=None)

Count(col, name="", label="", direction="up")
Binary(col, name="", label="", direction="up", min_sum=None)
Ratio(numerator, denominator, name="", label="", direction="up")     # numerator/denominator: column, Count or Binary
MeanRatio(numerator, denominator, name="", label="", direction="up") # the same inputs; mean of per-unit ratios
Quantile(col, quantiles=(0.5, 0.9), method="regular", bandwidth=0.001, lower=None, upper=None,
         resolution=1.0, accuracy=10_000, name="", label="", direction="up")
```

`ExperimentResults` views. Those with a `granularity` argument accept `"cumulative"` (the default)
or `"daily"`.

| Call | Returns |
|------|---------|
| `overall()` | Comparisons without a breakdown |
| `by_user_segment(segment)` | User-segment rows; with a segment such as `"os"` or `("os", "country")`, adds those columns back |
| `by_metric_segment(segment)` | Metric-segment rows, with `segment_share_control` and `segment_contribution` |
| `daily(metric)` | Daily rows; filter with `user_segment=`, `user_segment_value=`, `metric_segment=`, `metric_segment_value=` |
| `hte()` | One row per heterogeneity test, most significant first |
| `srm()` | One row per control/treatment pair and segment value (needs `expected_traffic`) |
| `table` | Every comparison |
| `population` | Denominators: users per table, flight and segment value (and day) |
| `sufficient_stats` | Raw per-flight sums collected from Spark, per table |

## Methods

The randomization unit (`ruid`) is the independent observation. Each metric is first reduced to one
value per unit: its total over the date range for cumulative results, and over the day for daily
results. Units in the population without rows contribute zeros, except in MeanRatio, which has a
ratio only for units whose denominator isn't zero. Sample variances divide by n − 1.

| Metric | Estimate per flight | Variance V of the estimate | Test and CI quantile c |
|--------|---------------------|----------------------------|------------------------|
| `Count(y)` | ȳ = Σ y_u / n | s²_y / n | Welch t; c = t(1 − α/2, ν) |
| `Binary(y)` | p̂ = #{u : y_u > 0} / n; with `min_sum=k`, #{u : y_u ≥ k} / n | p̂ (1 − p̂) / (n − 1) | Welch t; c = t(1 − α/2, ν) |
| `Ratio(y, x)` | R = Σ y_u / Σ x_u | (s²_y − 2R s_xy + R² s²_x) / (n x̄²), the delta method | z; c = z(1 − α/2) |
| `MeanRatio(y, x)` | r̄ = Σ r_u / n_x, with r_u = y_u / x_u over the n_x units with x_u ≠ 0 | s²_r / n_x | Welch t; c = t(1 − α/2, ν) |
| `Quantile(v, p)` | Q = p-quantile of v over all rows | Σ_u (a_u − r m_u)² / (Σ_u m_u)² / f̂(Q)² | z; c = z(1 − α/2) |

For treatment T and control C, with estimates θ and variances V:

* **Absolute difference:** Δ = θ_T − θ_C, with CI Δ ± c √(V_T + V_C). `p_value` is the two-sided test
  of Δ = 0 on the same statistic, so this CI excludes 0 exactly when p < α. The Welch degrees of
  freedom are ν = (V_T + V_C)² / (V_T² / (n_T − 1) + V_C² / (n_C − 1)).
* **Relative difference:** `pct_diff` = 100 (θ_T / θ_C − 1), with a delta-method CI
  100 [(θ_T / θ_C − 1) ± c √(V_T / θ_C² + θ_T² V_C / θ_C⁴)], which includes the control's uncertainty.
  `p_value` tests the absolute difference, so near α the two can disagree. In the quick start,
  `clickers` for t1 has p = 0.045 and a relative CI of [−0.10%, +13.65%].
* **Quantiles:**
  * m_u is unit u's number of rows, a_u its rows above Q, and r = Σa / Σm.
  * The density is f̂(Q) = 2h / (Q_{p+h} − Q_{p−h}) with `bandwidth` h = 0.001.
  * `method="regular"` reads Q and Q_{p±h} from `percentile_approx` (`accuracy` 10,000).
  * `method="density"` reads them from an exact histogram whose CDF is interpolated linearly within
    ±`resolution`/2 of each distinct value. That keeps f̂ finite for integer-valued metrics. Its cost
    grows with the number of distinct values, so round or clip (`lower`/`upper`) high-cardinality
    metrics.
* **Minimum samples, per flight:**
  * Count: n ≥ 200.
  * Binary: at least 10 units with a hit and 10 without.
  * Ratio: n ≥ 100.
  * MeanRatio: n_x ≥ 200.
  * Quantile: n ≥ 1,000 and at least 2,000 rows.

  Below these, `sufficient_sample` is False, and the CIs and p-value are left empty unless you pass
  `enforce_min_sample=False`.
* **Multiple testing:** `p_value_bh` is Benjamini–Hochberg across all metrics and quantiles of one
  comparison cell, meaning the same treatment, day and segment values.
* **Sample ratio mismatch:** `srm_p_value` tests each control/treatment pair on its own, as a
  chi-square goodness of fit (1 dof) of the pair's unit counts against `expected_traffic`.
* **Heterogeneous effects:** a Cochran's Q test that Δ is equal across the values of a user segment
  (χ² with k − 1 dof, using V_T + V_C).
  * A value takes part when both flights have at least `hte_min_units` units. `(null)` never takes
    part.
  * HTE is flagged when p < `hte_alpha`.
  * `hte_scale="relative"` tests `pct_diff` instead of Δ.
  * `hte_pattern` summarises which tested values are significant on their own at 0.01.
* **Metric segments** (Count and Binary metrics only):
  * `segment_share_control` is the segment's control value over the sum across segment values.
  * `segment_contribution` is the segment's Δ over the sum of Δ.

## Input contract

### Columns

| `Columns` field | Default | Meaning |
|-----------------|---------|---------|
| `unit` | `ruid` | Randomization unit id |
| `flight` | `flight` | The unit's flight, compared as a string. A row can also list several flights: a string (set `flight_separator`) or an array. |
| `flight_separator` | `None` | Set it, e.g. to `","`, when a string `flight` column lists several flights; spaces around each are ignored. `None`: one flight per row, matched exactly, so Spark can skip other flights' data. |
| `time` | `event_time` | DATE, TIMESTAMP, string or `yyyyMMdd` integer. Filtered to `start..end` and bucketed by calendar day in the Spark session time zone. |
| `exposure` | `None` | Optional first-exposure time, with the same types as `time` |
| `time_format` | `None` | Spark datetime pattern for string or integer times; strings default to `yyyy-MM-dd` |

Set the columns once on the `Experiment`. A table with a different schema passes its own
`MetricTable(columns=...)` or `FlightMapping(source, columns)`.

### Pre-joined tables vs a flight mapping

|                              | Pre-joined (default) | `FlightMapping` (opt-in) |
|------------------------------|----------------------|--------------------------|
| Metric table columns         | unit, flight, time, user segment columns, metric columns | unit, time, metric columns |
| Population (the denominator) | units with rows in this table | mapped units exposed by `end`. With `MetricTable(inner_join=True)`: only those with rows in this table. |
| Rows before exposure         | dropped when `exposure` is set | dropped when the mapping has `exposure` |
| Daily denominator on day d   | units whose exposure (or else first row) is on or before d | units exposed on or before d; all units if there is no `exposure` |

* **Exposure cut-off:** exact when both `time` and `exposure` are timestamps, by day otherwise.
* **Several mapping rows per unit:** reduced to one row per unit and flight, keeping the earliest
  exposure and the largest non-null segment value.
* **Unit in two flights** (in both control and treatment logs, say, or on a row that lists both): left
  out of the population and the rows. `Experiment(exclude_multi_flight_units=False)` counts it in
  each flight instead. `exp.validate(table)` reports how many there are.
* **Mapping dates:** rows exposed after `end` are dropped before grouping. Everyone exposed earlier
  stays in, even before `start`. To read the mapping another way, make its source a SQL query.

### Grain and segments

* **Grain:** Count and Ratio sum rows per unit, and Binary takes "any row > 0". So a table can be at
  any grain at or below unit × day. Quantile treats every row as one observation.
* **Segment syntax:** both kinds of segment accept `"os"`, `("os", "country")`, `"os|country"`, or
  a list of these. Multi-column values read like `ios|US`.
* **NULL values:** these form their own `(null)` segment.
* **User segments:** each unit gets exactly one value per column. In a pre-joined table whose rows
  disagree for a unit, the value on the unit's earliest row is used. Ties go to the smallest value, and
  NULL counts only if every row is NULL. A breakdown therefore never changes the overall results.
  `exp.validate(table)` counts the units whose values change.

## Roadmap

Planned. The code already leaves a place for each:

| Feature | Where it plugs in |
|---------|-------------------|
| Retro A/A | `Period` (`_common.py`): run the same pipeline over the pre-experiment range as period `"aa"`. Results already carry a `period` column. |
| CUPED | Join each unit's pre-period value in the unit rollup (`aggregate.unit_metric_stats`). Add Σx, Σx², Σxy to the metric's `group_aggregations`. Add an estimator in `stats.py` that adjusts θ and V before `compare`. |
| Capping / winsorization | `Count(cap=...)`: wrap the expression in `Count.unit_aggregations` with `least(..., cap)`. Per-day caps become a `when` on the day. |
| Plots (forest, daily trend) | A `plots.py` over the tidy `results.table`. |
| Ordinal HTE "trend" pattern | Mark a user segment as ordinal, then extend `compare._hte_pattern`. |
| Sequential testing | Another inference function next to `stats.compare`. |
| Very wide breakdowns | Replace the per-grouping aggregations plus union in `aggregate.py` with `GROUPING SETS`. |

## Design

```text
source ─► rows (unit, flight, day, segments, metric inputs)    pre-joined: rows and population from the table
       └► population (one row per unit and flight)              mapping: rows ⋈ mapping, population from the mapping

rows ─► unit rollup (unit [× day] [× metric segment]) ─► sufficient statistics per flight × segment [× day]
rows ─► quantiles per group ─► per-unit row and above-quantile counts ─► sufficient statistics
population ─► units per flight × user segment [× exposure day]   (denominators; units without rows count as zeros)
                                  │
                    pandas/numpy: estimates ─► compare ─► BH, HTE, contributions ─► ExperimentResults
```

* **Spark:** all groupings of a table are unioned and collected once. The unit-level shuffle is
  reused across user-segment groupings.
* **Statistics:** `stats.py` is plain numpy, vectorised over all groups.
* **Modules** (in `src/breakitdown/`):
  * `experiment.py`: `Experiment`
  * `tables.py`: `MetricTable`, `FlightMapping`
  * `metrics.py`: the metric classes
  * `columns.py`: `Columns`
  * `aggregate.py`: Spark
  * `compare.py`: pandas
  * `stats.py`: numpy/scipy
  * `results.py`: `ExperimentResults`

## Development

From a clone of the repository:

```bash
pip install -e ".[test]"
python -m pytest
```

The tests need Java 17 for local Spark.

* `tests/test_stats.py` checks every estimator against an independent computation on raw unit-level
  data: SciPy's Welch test, the linearised ratio variance and the Wald contrast test.
* The Spark tests run end to end on local Spark and compare every number with an independent
  pandas/SciPy implementation (`tests/reference.py`).
* `tests/test_readme.py` runs every Python block of this README against synthetic tables.

## License

Apache License 2.0; see [LICENSE](LICENSE).

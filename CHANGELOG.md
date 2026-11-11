# Changelog

## 0.3.0 (2026-11-10)

Added:

- `MeanRatio(numerator, denominator)`: the mean over users of each user's own numerator / denominator,
  over the users whose denominator isn't zero, tested with Welch's t-test. Its unit count is those
  users. `Ratio` stays the ratio of the totals.
- `Binary(col, min_sum=k)`: a user counts when `col` adds up to at least k, e.g. users active on two
  or more days. Without `min_sum`, `Binary` is unchanged (any positive value).
- `Experiment(exclude_multi_flight_units=...)`, see below.

Changed:

- Users seen in more than one of an experiment's flights, say in both control and treatment logs, are
  now left out of the population and the rows. Results change only for experiments that have such
  users, and `validate()` reports how many. `exclude_multi_flight_units=False` counts them in each
  flight, as before.
- A flight mapping is now filtered to exposures up to `end` before it's grouped, with a filter Spark
  can push down to the source, which matters for very large mapping tables. Users exposed before
  `start` still count. One edge case changes: a user whose mapping rows have no exposure time, apart
  from ones after `end`, now counts as exposed from the start, like a user with no exposure time at all.
- The README shows how to pass a SQL query as the mapping, to read it your own way.

## 0.2.0 (2026-11-10)

Added: one table for many experiments. A row's flight column can list every flight its user is in,
as many teams log it.

- Set `Columns(flight_separator=",")`, or any separator, for a string column. An array column is read
  as a list on its own. Spaces around each flight are ignored.
- Each experiment reads the rows that list one of its own flights and skips the rest. A row that lists
  two of them counts in both, and `Experiment.validate()` reports it.
- Rows that name none of the experiment's flights are dropped with a cheap text test before splitting.
- One flight per row stays the default: an exact match that Spark pushes down to the source, so it
  still skips other flights' files and partitions.

Also: the package metadata links to the author's homepage, https://maxhe.org/.

## 0.1.1 (2026-11-06)

Fixed: in pre-joined tables, a user whose segment value changed between rows (say, `os` recorded per
event for someone with two devices) was split into one unit per value. That also changed the overall
results, not just the breakdown.

- Every user now gets exactly one value per user segment column: the value on their earliest row, the
  one closest to exposure.
- Ties at the same time go to the smallest value. NULLs are skipped unless every row is NULL.
- Adding a breakdown never changes the overall numbers, and segment sizes add up to the overall size.
- `Experiment.validate()` still counts users whose values change.

## 0.1.0 (2026-11-06)

First release.

- `Experiment` and `MetricTable`: A/B analysis on Spark.
  - Tables that carry the flight on every row work out of the box.
  - Joining one `FlightMapping` to many metric tables is opt-in.
- Metrics: `Count`, `Binary`, `Ratio` and `Quantile`. Quantiles have `regular` and `density` methods.
- User segments and metric segments, in any combination, with cumulative and daily results.
- Inference:
  - Welch t-tests.
  - Delta-method ratio and relative-difference intervals.
  - Unit-clustered quantile variance.
  - Benjamini–Hochberg adjustment.
  - Pairwise sample-ratio-mismatch tests.
  - Cochran's Q test for heterogeneous effects.

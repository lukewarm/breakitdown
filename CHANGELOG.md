# Changelog

## Unreleased

- The package metadata links to the author's homepage, https://maxhe.org/.

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

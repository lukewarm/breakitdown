# Changelog

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

"""Specs and argument validation: no Spark job runs here."""

import datetime as dt

import pandas as pd
import pytest

from breakitdown import Binary, Columns, Count, Direction, Experiment, FlightMapping, MetricTable, Quantile, QuantileMethod, Ratio
from breakitdown._common import normalize_segments, to_date
from breakitdown.compare import compare_table


def test_segment_specs():
    assert normalize_segments(None) == ()
    assert normalize_segments("os") == (("os",),)
    assert normalize_segments("os|country") == (("os", "country"),)
    assert normalize_segments(["os", ("os", "country"), "os"]) == (("os",), ("os", "country"))
    with pytest.raises(ValueError):
        normalize_segments([("os", "os")])
    with pytest.raises(ValueError):
        normalize_segments([()])


def test_dates():
    assert to_date("2024-03-01") == to_date("20240301") == to_date(dt.datetime(2024, 3, 1, 5)) == dt.date(2024, 3, 1)
    with pytest.raises(ValueError):
        to_date("March 1")


def test_metric_defaults_and_validation():
    assert Count("clicks").name == "clicks" and Count("clicks").direction is Direction.UP
    assert Count("latency", direction="down").direction is Direction.DOWN
    ratio = Ratio("clicks", Binary("sessions"))
    assert ratio.name == "clicks_per_sessions" and isinstance(ratio.numerator, Count)
    assert ratio.input_columns == ("clicks", "sessions")
    q = Quantile("latency_ms", [50, "90", 0.99], method="density")
    assert q.quantiles == (0.5, 0.9, 0.99) and q.method is QuantileMethod.DENSITY
    with pytest.raises(ValueError):
        Quantile("x", [0.9995])  # p + bandwidth leaves (0, 1)
    with pytest.raises(ValueError):
        Quantile("x", [0.5, 50])
    with pytest.raises(TypeError):
        Ratio("clicks", Quantile("x"))


def test_metric_table_validation():
    table = MetricTable("db.events", [Count("clicks")])
    assert table.name == "db.events"
    assert MetricTable("SELECT 1", [Count("a"), Binary("b")]).name == "a_b"
    with pytest.raises(ValueError, match="Duplicate metric names"):
        MetricTable("db.events", [Count("clicks"), Binary("clicks")])
    with pytest.raises(ValueError, match="metric segments"):
        MetricTable("db.events", [Count("clicks")], metric_segments=["clicks"])


def test_experiment_validation():
    exp = Experiment("c", "t", "20240301", "20240307", user_segments="os")
    assert exp.flights == ("c", "t") and exp.user_segments == (("os",),)
    with pytest.raises(ValueError):
        Experiment("c", ["c"], "2024-03-01", "2024-03-07")
    with pytest.raises(ValueError):
        Experiment("c", ["t"], "2024-03-07", "2024-03-01")
    with pytest.raises(ValueError, match="no share"):
        Experiment("c", ["t"], "2024-03-01", "2024-03-07", expected_traffic={"c": 1})
    with pytest.raises(ValueError, match="user segments"):
        exp._check_tables([MetricTable("db.e", [Count("os")])])
    with pytest.raises(ValueError, match="unique across tables"):
        exp._check_tables([MetricTable("db.a", [Count("x")]), MetricTable("db.b", [Count("x")])])
    with pytest.raises(TypeError):
        Experiment("c", "t", "2024-03-01", "2024-03-07", mapping="db.mapping")
    assert FlightMapping("db.mapping", Columns(exposure="exposure_time")).columns.exposure == "exposure_time"


def _stats_frame(rows):
    base = dict(period="ab", granularity="cumulative", day=None, user_segment="overall",
                user_segment_value="overall", metric_segment="overall", metric_segment_value="overall")
    return pd.DataFrame([{**base, **row} for row in rows])


def _counts(n):
    return pd.DataFrame([dict(period="ab", granularity="cumulative", day=pd.NaT, user_segment="overall",
                              user_segment_value="overall", flight=f, n=v) for f, v in n.items()])


def test_compare_fills_flights_without_rows_and_enforces_min_sample():
    # t2 has units but none of them has rows: its sums are zero, not missing.
    stats_df = _stats_frame([
        dict(flight="c", units_with_data=300, __m0__sum=600.0, __m0__sumsq=2000.0),
        dict(flight="t1", units_with_data=290, __m0__sum=660.0, __m0__sumsq=2300.0),
    ])
    out = compare_table("events", [("__m0", Count("clicks"))], stats_df, _counts({"c": 400, "t1": 400, "t2": 150}),
                        ["c", "t1", "t2"], alpha=0.05, enforce_min_sample=True, expected_traffic=None)
    t1, t2 = out.set_index("treatment").loc["t1"], out.set_index("treatment").loc["t2"]
    assert t1.value_control == pytest.approx(1.5) and t1.value_treatment == pytest.approx(1.65)
    assert t1.sufficient_sample and t1.p_value > 0
    assert t2.value_treatment == 0 and t2.n_treatment == 150
    assert not t2.sufficient_sample and pd.isna(t2.p_value) and pd.isna(t2.pct_diff_ci_lower)
    assert t2.pct_diff == pytest.approx(-100)  # point estimates are kept

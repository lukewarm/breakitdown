"""Sources, column conventions and formats, exposure in pre-joined tables, validate(), regular quantiles."""

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from breakitdown import Columns, Count, Experiment, MetricTable, Quantile
from . import reference as ref
from .reference import check, row


@pytest.fixture(scope="module")
def small(spark):
    records = ref.event_records(seed=3, units=600)
    return spark.createDataFrame(records, ref.EVENT_SCHEMA), ref.to_pandas(records)


def test_sql_and_table_name_sources_match_a_dataframe(small):
    sdf, _ = small
    sdf.createOrReplaceTempView("breakitdown_test_events")
    experiment = Experiment("control", ["t1", "t2"], ref.START, ref.END)
    out = experiment.analyze([
        MetricTable(sdf, [Count("clicks")], name="dataframe"),
        MetricTable("breakitdown_test_events", [Count("clicks", name="clicks_table")]),
        MetricTable("SELECT * FROM breakitdown_test_events WHERE clicks >= 0", [Count("clicks", name="clicks_sql")],
                    name="sql"),
    ], daily=False, enforce_min_sample=False).overall()
    expected = row(out, metric="clicks", treatment="t1")
    for name in ("clicks_table", "clicks_sql"):
        check(row(out, metric=name, treatment="t1"),
              {c: expected[c] for c in ("n_control", "value_control", "value_treatment", "p_value")})


def test_custom_column_names_and_date_formats(small):
    sdf, pdf = small
    renamed = sdf.select(
        F.col("ruid").alias("user_id"),
        F.col("flight").alias("variant"),
        F.date_format("event_time", "yyyyMMdd").alias("ds"),
        "clicks",
    )
    population = pdf[["ruid", "flight"]].drop_duplicates()
    expected = ref.welch(ref.unit_values(pdf, population, "clicks"), "t1")

    as_string = Experiment("control", ["t1", "t2"], "20240301", "20240305",
                           columns=Columns(unit="user_id", flight="variant", time="ds", time_format="yyyyMMdd"))
    results = as_string.analyze(MetricTable(renamed, [Count("clicks")]), enforce_min_sample=False)
    check(row(results.overall(), metric="clicks", treatment="t1"), expected)
    assert results.daily("clicks")["day"].nunique() == ref.DAYS

    as_integer = Experiment("control", ["t1", "t2"], "20240301", "20240305",
                            columns=Columns(unit="user_id", flight="variant", time="ds"))
    results = as_integer.analyze(MetricTable(renamed.withColumn("ds", F.col("ds").cast("int")), [Count("clicks")]),
                                 daily=False, enforce_min_sample=False)
    check(row(results.overall(), metric="clicks", treatment="t1"), expected)


def test_exposure_column_in_a_prejoined_table(spark):
    rows = [
        ("a", "control", ref._at(0, 100), ref._at(1, 0), 5),  # before the unit's exposure: dropped
        ("a", "control", ref._at(2, 100), ref._at(1, 0), 1),
        ("b", "t1", ref._at(3, 100), ref._at(0, 50), 2),
        ("c", "t1", ref._at(-1, 100), ref._at(-2, 0), 7),  # outside the date range: c is not in the population
    ]
    df = spark.createDataFrame(rows, "ruid string, flight string, event_time timestamp, exposure_time timestamp, clicks int")
    experiment = Experiment("control", ["t1"], ref.START, ref.END, columns=Columns(exposure="exposure_time"))
    results = experiment.analyze(MetricTable(df, [Count("clicks")]), enforce_min_sample=False)
    overall = row(results.overall(), metric="clicks", treatment="t1")
    assert (overall.n_control, overall.value_control, overall.n_treatment, overall.value_treatment) == (1, 1, 1, 2)
    # On day 2, b (exposed on day 0) is in the denominator with zero clicks.
    day2 = row(results.daily("clicks"), day=pd.Timestamp(ref.START) + pd.Timedelta(days=2), treatment="t1")
    assert (day2.n_control, day2.value_control, day2.n_treatment, day2.value_treatment) == (1, 1, 1, 0)


def test_regular_quantile_matches_numpy(spark):
    rng = np.random.default_rng(9)
    records = []
    for i in range(6000):
        unit = int(rng.integers(0, 900))
        records.append(dict(ruid=f"u{unit}", flight=ref.FLIGHTS[unit % 3],
                            event_time=ref._at(int(rng.integers(0, ref.DAYS)), 60),
                            value=float(rng.exponential(100.0)) * (0.9 if unit % 3 == 1 else 1.0)))
    sdf = spark.createDataFrame(records, "ruid string, flight string, event_time timestamp, value double")
    experiment = Experiment("control", ["t1"], ref.START, ref.END)  # t2 rows are ignored
    results = experiment.analyze(MetricTable(sdf, [Quantile("value", [0.5, 0.9], accuracy=1_000_000)]),
                                 daily=False, enforce_min_sample=False)
    pdf = pd.DataFrame(records)
    pdf = pdf[pdf["flight"] != "t2"]
    n_units = pdf.groupby("flight")["ruid"].nunique().to_dict()
    for p in (0.5, 0.9):
        expected = ref.quantile(pdf, n_units, "value", p, "t1", method="regular")
        check(row(results.overall(), metric="value", quantile=p, treatment="t1"),
              {k: expected[k] for k in ("value_control", "value_treatment", "n_control", "p_value")})


def test_validate_reports_data_problems(spark):
    rows = [
        ("a", "control", "ios", ref._at(0, 10), 1),
        ("a", "t1", "ios", ref._at(1, 10), 1),  # one unit in two flights
        ("b", "control", "ios", ref._at(0, 10), 1),
        ("b", "control", "android", ref._at(1, 10), 1),  # segment value changes
        ("c", "control", None, ref._at(2, 10), 1),
    ]
    df = spark.createDataFrame(rows, "ruid string, flight string, os string, event_time timestamp, clicks int")
    experiment = Experiment("control", ["t1", "t2"], ref.START, ref.END, user_segments="os")
    checks = experiment.validate(MetricTable(df, [Count("clicks")])).set_index("check")
    assert checks.loc["units", "value"] == 3
    assert checks.loc["units in more than one flight", "value"] == 1
    assert checks.loc["units with more than one 'os' value", "value"] == 1
    assert checks.loc["flights without units", "value"] == "t2" and not checks.loc["flights without units", "ok"]


def test_missing_columns_are_reported_before_any_job(small):
    sdf, _ = small
    experiment = Experiment("control", ["t1"], ref.START, ref.END, user_segments="platform")
    with pytest.raises(ValueError, match=r"missing column\(s\) \['platform'\]"):
        experiment.analyze(MetricTable(sdf, [Count("clicks")]))

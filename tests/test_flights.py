"""Reading flights: one per row (an exact match) or a list, units in two flights, and the mapping's dates.

Every layout of the same assignments must give the same results as one flight per row.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from breakitdown import Columns, Count, Experiment, FlightMapping, MetricTable, aggregate
from breakitdown._common import Period

START, END = "2024-03-01", "2024-03-07"
COMPARED = ["metric", "treatment", "n_control", "n_treatment", "value_control", "value_treatment", "p_value"]
EXPERIMENTS = {"search": ("search_c", "search_t1"), "feed": ("feed_c", "feed_t1")}


@pytest.fixture(scope="module")
def assignments():
    """Users in two independent experiments, search and feed, with a week of daily clicks."""
    rng = np.random.default_rng(7)
    users = []
    for u in range(600):
        search, feed = str(rng.choice(["search_c", "search_t1"])), str(rng.choice(["feed_c", "feed_t1"]))
        days = [(dt.date(2024, 3, 1) + dt.timedelta(days=d), int(rng.poisson(1.3 if search == "search_t1" else 1.0)))
                for d in range(7)]
        users.append((f"u{u}", search, feed, days))
    return users


def events(spark, assignments, flight, flight_type="string"):
    """The click table, with each row's flight column made by ``flight(ruid, search, feed)``."""
    rows = [(ruid, flight(ruid, search, feed), day, clicks)
            for ruid, search, feed, days in assignments for day, clicks in days]
    return spark.createDataFrame(rows, f"ruid string, flight {flight_type}, event_time date, clicks int")


def overall(experiment, df):
    return experiment.analyze(MetricTable(df, [Count("clicks")]), enforce_min_sample=False).overall()[COMPARED]


@pytest.fixture(scope="module")
def expected(spark, assignments):
    """Each experiment on its own table with one flight per row: the layout breakitdown always read."""
    tables = {
        "search": events(spark, assignments, lambda ruid, search, feed: search),
        "feed": events(spark, assignments, lambda ruid, search, feed: feed),
    }
    return {name: overall(Experiment(c, [t], START, END), tables[name]) for name, (c, t) in EXPERIMENTS.items()}


LAYOUTS = {
    "comma-separated": (lambda ruid, search, feed: f"{search},{feed}", "string", ","),
    "spaces around names": (lambda ruid, search, feed: f" {feed} , {search} ", "string", ","),
    "a regex character": (lambda ruid, search, feed: f"{search}|{feed}", "string", "|"),
    "several characters": (lambda ruid, search, feed: f"{search}::{feed}", "string", "::"),
    "array": (lambda ruid, search, feed: [search, feed], "array<string>", None),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_listed_flights_match_one_flight_per_row(spark, assignments, expected, layout):
    flight, flight_type, separator = LAYOUTS[layout]
    shared = events(spark, assignments, flight, flight_type)
    for name, (control, treatment) in EXPERIMENTS.items():
        experiment = Experiment(control, [treatment], START, END, columns=Columns(flight_separator=separator))
        pd.testing.assert_frame_equal(overall(experiment, shared), expected[name])


def test_by_default_each_value_is_one_flight(spark, assignments, expected):
    with_commas = events(spark, assignments, lambda ruid, search, feed: search.replace("_", ","))
    result = overall(Experiment("search,c", ["search,t1"], START, END), with_commas).assign(treatment="search_t1")
    pd.testing.assert_frame_equal(result, expected["search"])


def test_one_flight_per_row_is_pushed_down_to_the_source(spark, assignments, tmp_path):
    path = str(tmp_path / "events")
    events(spark, assignments, lambda ruid, search, feed: search).write.parquet(path)
    df = spark.read.parquet(path)
    keep, flight = aggregate.experiment_flight(df, Columns(), ["search_c", "search_t1"])
    assert "In(flight" in pushed_filters(df.where(keep).select(flight))


def pushed_filters(df):
    """The filters Spark hands to the file source, from the physical plan."""
    plan = df._jdf.queryExecution().executedPlan().toString()
    return plan.split("PushedFilters:")[1].split("ReadSchema")[0]


def test_flight_mapping_can_list_flights(spark, assignments, expected):
    clicks = spark.createDataFrame(
        [(ruid, day, c) for ruid, _, _, days in assignments for day, c in days], "ruid string, event_time date, clicks int"
    )
    mapping = spark.createDataFrame([(ruid, f"{s},{f}") for ruid, s, f, _ in assignments], "ruid string, flight string")
    experiment = Experiment("search_c", ["search_t1"], START, END,
                            mapping=FlightMapping(mapping, Columns(flight_separator=",")))
    pd.testing.assert_frame_equal(overall(experiment, clicks), expected["search"])


def test_a_unit_in_two_flights_is_left_out_unless_asked(spark, assignments):
    both = events(spark, assignments,
                  lambda ruid, search, feed: "search_c,search_t1" if ruid == "u0" else f"{search},{feed}")
    columns = Columns(flight_separator=",")
    left_out = Experiment("search_c", ["search_t1"], START, END, columns=columns)
    checks = left_out.validate(MetricTable(both, [Count("clicks")])).set_index("check")
    assert checks.loc["units in more than one flight", "value"] == 1
    assert checks.loc["units in more than one flight", "ok"] is None  # left out, so just for information
    result = overall(left_out, both).iloc[0]
    assert result["n_control"] + result["n_treatment"] == len(assignments) - 1
    in_both = Experiment("search_c", ["search_t1"], START, END, columns=columns, exclude_multi_flight_units=False)
    result = overall(in_both, both).iloc[0]
    assert result["n_control"] + result["n_treatment"] == len(assignments) + 1


def test_a_mapped_unit_in_two_flights_is_left_out(spark, assignments, expected):
    clicks = spark.createDataFrame(
        [(ruid, day, c) for ruid, _, _, days in assignments for day, c in days], "ruid string, event_time date, clicks int"
    )
    rows = [(ruid, s) for ruid, s, _, _ in assignments] + [("u0", "search_t1" if assignments[0][1] == "search_c" else "search_c")]
    mapping = spark.createDataFrame(rows, "ruid string, flight string")
    result = overall(Experiment("search_c", ["search_t1"], START, END, mapping=FlightMapping(mapping)), clicks).iloc[0]
    assert result["n_control"] + result["n_treatment"] == len(assignments) - 1


EXPOSURES = {
    "timestamp": ("timestamp", lambda day, hour: dt.datetime.combine(day, dt.time(hour), tzinfo=dt.timezone.utc)),
    "date": ("date", lambda day, hour: day),
    "yyyyMMdd integer": ("int", lambda day, hour: int(day.strftime("%Y%m%d"))),
    "ISO string": ("string", lambda day, hour: day.isoformat()),
}


@pytest.mark.parametrize("kind", EXPOSURES)
def test_mapping_keeps_every_exposure_up_to_the_end_date(spark, kind):
    sql_type, value = EXPOSURES[kind]
    rows = [
        ("before", "control", value(dt.date(2024, 2, 20), 9)),  # before the start date: still in
        ("last", "t1", value(dt.date(2024, 3, 7), 23)),         # on the end date: in
        ("after", "t1", value(dt.date(2024, 3, 8), 0)),         # after it: out
        ("unknown", "control", None),                           # no exposure time: in
    ]
    mapping = spark.createDataFrame(rows, f"ruid string, flight string, exposure_time {sql_type}")
    kept = aggregate.mapping_rows(mapping, Columns(exposure="exposure_time"), Period(dt.date(2024, 3, 1), dt.date(2024, 3, 7)),
                                  ["control", "t1"], [], "FlightMapping")
    assert sorted(r[0] for r in kept.select(aggregate.UNIT).collect()) == ["before", "last", "unknown"]


def test_mapping_end_date_filter_reaches_the_source(spark, tmp_path):
    path = str(tmp_path / "mapping")
    spark.createDataFrame([("a", "control", dt.datetime(2024, 3, 2, tzinfo=dt.timezone.utc))],
                          "ruid string, flight string, exposure_time timestamp").write.parquet(path)
    rows = aggregate.mapping_rows(spark.read.parquet(path), Columns(exposure="exposure_time"),
                                  Period(dt.date(2024, 3, 1), dt.date(2024, 3, 7)), ["control", "t1"], [], "FlightMapping")
    assert "LessThan(exposure_time" in pushed_filters(rows)

"""Flight columns: one flight per row (an exact match), or a list of flights in a string or an array.

Every layout of the same assignments must give the same results as one flight per row.
"""

import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest

from breakitdown import Columns, Count, Experiment, FlightMapping, MetricTable, aggregate

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
    plan = df.where(keep).select(flight)._jdf.queryExecution().executedPlan().toString()
    assert re.search(r"PushedFilters: \[[^\]]*In\(flight", plan), plan


def test_flight_mapping_can_list_flights(spark, assignments, expected):
    clicks = spark.createDataFrame(
        [(ruid, day, c) for ruid, _, _, days in assignments for day, c in days], "ruid string, event_time date, clicks int"
    )
    mapping = spark.createDataFrame([(ruid, f"{s},{f}") for ruid, s, f, _ in assignments], "ruid string, flight string")
    experiment = Experiment("search_c", ["search_t1"], START, END,
                            mapping=FlightMapping(mapping, Columns(flight_separator=",")))
    pd.testing.assert_frame_equal(overall(experiment, clicks), expected["search"])


def test_a_row_listing_two_flights_of_one_experiment_counts_in_both(spark, assignments):
    both = events(spark, assignments,
                  lambda ruid, search, feed: "search_c,search_t1" if ruid == "u0" else f"{search},{feed}")
    experiment = Experiment("search_c", ["search_t1"], START, END, columns=Columns(flight_separator=","))
    checks = experiment.validate(MetricTable(both, [Count("clicks")])).set_index("check")
    assert checks.loc["units in more than one flight", "value"] == 1
    result = overall(experiment, both).iloc[0]
    assert result["n_control"] + result["n_treatment"] == len(assignments) + 1

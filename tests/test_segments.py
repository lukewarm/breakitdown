"""User segments in pre-joined tables: every unit gets exactly one value per segment column."""

import random

import pandas as pd
import pytest

from breakitdown import Count, Experiment, MetricTable
from . import reference as ref

SCHEMA = "ruid string, flight string, os string, event_time timestamp, clicks int"


@pytest.fixture(scope="module")
def changing_os(spark):
    rows = [
        ("a", "control", "ios", ref._at(0, 100), 1),
        ("a", "control", "android", ref._at(1, 100), 2),  # earliest row wins: ios
        ("b", "control", None, ref._at(0, 100), 3),
        ("b", "control", "android", ref._at(2, 100), 4),  # NULLs are skipped: android
        ("c", "control", "ios", ref._at(1, 100), 5),
        ("c", "control", "android", ref._at(1, 100), 6),  # same instant: smallest value, android
        ("d", "control", None, ref._at(0, 100), 7),  # NULL on every row: (null)
        ("e", "t1", "android", ref._at(0, 100), 1),
        ("e", "t1", "ios", ref._at(3, 100), 1),  # earliest row wins: android
    ]
    return spark.createDataFrame(rows, SCHEMA)


def test_each_unit_takes_the_value_on_its_earliest_row(changing_os):
    experiment = Experiment("control", ["t1"], ref.START, ref.END, user_segments="os")
    results = experiment.analyze(MetricTable(changing_os, [Count("clicks")]), daily=False, enforce_min_sample=False)
    population = results.population
    counts = population[population["user_segment"] == "os"].set_index(["flight", "user_segment_value"])["n"]
    assert counts.to_dict() == {("control", "android"): 2, ("control", "ios"): 1, ("control", "(null)"): 1,
                                ("t1", "android"): 1}
    # Unit c (android) keeps both of its rows: 5 + 6 clicks, and b keeps its NULL-os row: 3 + 4.
    android = results.by_user_segment("os").set_index("os").loc["android"]
    assert android.value_control == pytest.approx((3 + 4 + 5 + 6) / 2)
    checks = experiment.validate(MetricTable(changing_os, [Count("clicks")])).set_index("check")
    assert checks.loc["units with more than one 'os' value", "value"] == 3  # a, c and e; b's NULL is not a value


def test_a_breakdown_never_changes_the_overall_result(spark):
    rng = random.Random(3)
    rows = [(f"u{u}", ["control", "t1"][u % 2], rng.choice(["ios", "android"]), ref._at(d, (60 * u) % 86400),
             rng.randint(0, 3 + u % 2)) for u in range(600) for d in range(3)]
    table = MetricTable(spark.createDataFrame(rows, SCHEMA), [Count("clicks")])
    plain = Experiment("control", "t1", ref.START, ref.END).analyze(table, enforce_min_sample=False)
    segmented = Experiment("control", "t1", ref.START, ref.END, user_segments="os").analyze(table, enforce_min_sample=False)

    before, after = plain.overall().iloc[0], segmented.overall().iloc[0]
    for column in ("n_control", "n_treatment", "value_control", "value_treatment", "p_value"):
        assert after[column] == pytest.approx(before[column], rel=1e-12), column

    # Units are never split: segment sizes add up to the overall size, cumulatively and on every day.
    by_os = segmented.by_user_segment("os")
    assert by_os["n_control"].sum() == after.n_control and by_os["n_treatment"].sum() == after.n_treatment
    daily = segmented.table[segmented.table["granularity"] == "daily"]
    per_day = daily[daily["metric_segment"] == "overall"].groupby(["day", "user_segment"])["n_control"].sum().unstack()
    pd.testing.assert_series_equal(per_day["os"], per_day["overall"], check_names=False)

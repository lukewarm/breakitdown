"""Mapping mode: one FlightMapping joined to several metric tables."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from breakitdown import Columns, Count, Experiment, FlightMapping, MetricTable
from . import reference as ref
from .reference import check, row

MAPPING_SCHEMA = "ruid string, flight string, os string, exposure_time timestamp"
EVENT_SCHEMA = "ruid string, event_time timestamp, clicks int"
DAY_EVENT_SCHEMA = "ruid string, event_time date, clicks int"


def _records(seed=5, units=1500):
    rng = np.random.default_rng(seed)
    mapping, events = [], []
    for u in range(units):
        flight = ref.FLIGHTS[int(rng.integers(0, 3))]
        # Exposure from the day before the range to the day after it (those units are left out).
        exposure = ref._at(int(rng.integers(-1, ref.DAYS + 1)), int(rng.integers(0, 86400)))
        mapping.append(dict(ruid=f"u{u}", flight=flight, os=str(rng.choice(["ios", "android"])), exposure_time=exposure))
        if rng.random() < 0.2:
            continue  # mapped units without any rows count as zeros
        for _ in range(int(rng.poisson(6))):
            events.append(dict(
                ruid=f"u{u}",
                event_time=ref._at(int(rng.integers(-1, ref.DAYS + 1)), int(rng.integers(0, 86400))),
                clicks=int(rng.poisson(1.2 if flight == "t1" else 1.0)),
            ))
    events.append(dict(ruid="stranger", event_time=ref._at(1, 10), clicks=3))  # not in the mapping
    return mapping, events


@pytest.fixture(scope="module")
def data(spark):
    mapping, events = _records()
    day_events = [dict(e, event_time=e["event_time"].date()) for e in events]
    frames = dict(
        mapping=spark.createDataFrame(mapping, MAPPING_SCHEMA),
        events=spark.createDataFrame(events, EVENT_SCHEMA),
        day_events=spark.createDataFrame(day_events, DAY_EVENT_SCHEMA),
    )
    m = pd.DataFrame(mapping)
    m["exposure_day"] = m["exposure_time"].map(lambda t: t.date())
    population = m[m["exposure_day"] <= ref.END].copy()
    population["first_day"] = population["exposure_day"].clip(lower=ref.START)
    e = pd.DataFrame(events)
    e["day"] = e["event_time"].map(lambda t: t.date())
    e = e.merge(population[["ruid", "flight", "exposure_time", "exposure_day"]], on="ruid")
    e = e[(e["day"] >= ref.START) & (e["day"] <= ref.END)]
    return frames, population, e


@pytest.fixture(scope="module")
def experiment(data):
    frames, _, _ = data
    return Experiment(
        "control", ["t1", "t2"], ref.START, ref.END, user_segments="os",
        mapping=FlightMapping(frames["mapping"], Columns(exposure="exposure_time")),
    )


@pytest.fixture(scope="module")
def results(data, experiment):
    frames, _, _ = data
    return experiment.analyze([
        MetricTable(frames["events"], [Count("clicks")], name="timestamps"),
        MetricTable(frames["events"], [Count("clicks", name="clicks_inner")], name="timestamps_inner", inner_join=True),
        MetricTable(frames["day_events"], [Count("clicks", name="clicks_by_day")], name="dates"),
    ], enforce_min_sample=False)


def test_mapped_units_without_rows_count_as_zero(results, data):
    _, population, e = data
    after_exposure = e[e["event_time"] >= e["exposure_time"]]  # both timestamps: exact cut-off
    units = population[["ruid", "flight"]]
    for treatment in ("t1", "t2"):
        check(row(results.overall(), metric="clicks", treatment=treatment),
              ref.welch(ref.unit_values(after_exposure, units, "clicks"), treatment))


def test_inner_join_keeps_only_units_with_rows(results, data):
    _, population, e = data
    after_exposure = e[e["event_time"] >= e["exposure_time"]]
    active = population[population["ruid"].isin(after_exposure["ruid"])][["ruid", "flight"]]
    check(row(results.overall(), metric="clicks_inner", treatment="t1"),
          ref.welch(ref.unit_values(after_exposure, active, "clicks"), "t1"))


def test_date_grain_table_cuts_off_by_day(results, data):
    _, population, e = data
    same_day_or_later = e[e["day"] >= e["exposure_day"]]
    check(row(results.overall(), metric="clicks_by_day", treatment="t1"),
          ref.welch(ref.unit_values(same_day_or_later, population[["ruid", "flight"]], "clicks"), "t1"))


def test_daily_and_segments_use_the_mapping(results, data):
    _, population, e = data
    after_exposure = e[e["event_time"] >= e["exposure_time"]]
    day = ref.START + dt.timedelta(days=1)
    exposed = population[population["first_day"] <= day][["ruid", "flight"]]
    check(row(results.daily("clicks"), day=pd.Timestamp(day), treatment="t1"),
          ref.welch(ref.unit_values(after_exposure[after_exposure["day"] == day], exposed, "clicks"), "t1"))
    ios = population[population["os"] == "ios"][["ruid", "flight"]]
    check(row(results.by_user_segment("os"), metric="clicks", os="ios", treatment="t1"),
          ref.welch(ref.unit_values(after_exposure, ios, "clicks"), "t1"))
    assert results.settings["mode"] == "mapping"


def test_validate_in_mapping_mode(experiment, data):
    frames, _, _ = data
    checks = experiment.validate(MetricTable(frames["events"], [Count("clicks")])).set_index("check")
    assert checks.loc["metric rows whose unit is not in the mapping", "value"] == 1
    assert checks.loc["units in more than one flight", "value"] == 0
    assert checks.loc["flights without units", "ok"]

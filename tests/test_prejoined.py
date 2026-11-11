"""End to end on Spark in pre-joined mode, checked against an independent pandas/scipy reference."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from breakitdown import Binary, Count, Experiment, MeanRatio, MetricTable, Quantile, Ratio, stats
from . import reference as ref
from .reference import check, row


@pytest.fixture(scope="module")
def events(spark):
    records = ref.event_records()
    return spark.createDataFrame(records, ref.EVENT_SCHEMA), ref.to_pandas(records)


@pytest.fixture(scope="module")
def population(events):
    _, pdf = events
    return pdf.groupby(["ruid", "flight", "os", "country"])["day"].min().rename("first_day").reset_index()


@pytest.fixture(scope="module")
def results(events):
    sdf, _ = events
    experiment = Experiment(
        "control", ["t1", "t2"], ref.START, ref.END,
        user_segments=["os", ("os", "country")],
        expected_traffic={"control": 1, "t1": 1, "t2": 1},
    )
    table = MetricTable(
        sdf,
        [
            Count("clicks"),
            Binary("clicks", name="clickers"),
            Ratio("clicks", "session_start", name="clicks_per_session"),
            Binary("clicks", name="clicks_3plus", min_sum=3),
            MeanRatio("session_start", "clicks", name="mean_sessions_per_click"),  # 71 units have no clicks
            Quantile("latency_ms", [0.5, 0.9], method="density", name="latency", direction="down"),
        ],
        name="events",
        metric_segments=["media_type"],
    )
    return experiment.analyze(table, hte_min_units=50, enforce_min_sample=False)


@pytest.mark.parametrize("treatment", ["t1", "t2"])
def test_overall_count_binary_ratio(results, events, population, treatment):
    _, pdf = events
    overall = results.overall()
    clicks = ref.unit_values(pdf, population, "clicks")
    check(row(overall, metric="clicks", treatment=treatment), ref.welch(clicks, treatment))
    clickers = ref.unit_values(pdf, population, "clicks", how="any")
    check(row(overall, metric="clickers", treatment=treatment), ref.welch(clickers, treatment))
    sessions = ref.unit_values(pdf, population, "session_start")
    check(row(overall, metric="clicks_per_session", treatment=treatment), ref.ratio(clicks, sessions, treatment))


@pytest.mark.parametrize("treatment", ["t1", "t2"])
def test_overall_binary_threshold_and_mean_ratio(results, events, population, treatment):
    _, pdf = events
    overall = results.overall()
    clicks = ref.unit_values(pdf, population, "clicks")
    three_plus = clicks.assign(v=(clicks["v"] >= 3).astype(float))
    check(row(overall, metric="clicks_3plus", treatment=treatment), ref.welch(three_plus, treatment))
    sessions = ref.unit_values(pdf, population, "session_start")
    both = sessions.merge(clicks, on=["ruid", "flight"], suffixes=("_n", "_d"))
    with_clicks = both[both["v_d"] != 0]  # a unit's ratio exists only with a non-zero denominator
    assert len(with_clicks) < len(both)  # so the metric has its own unit count
    ratios = with_clicks.assign(v=with_clicks["v_n"] / with_clicks["v_d"])
    check(row(overall, metric="mean_sessions_per_click", treatment=treatment), ref.welch(ratios, treatment))


@pytest.mark.parametrize("p", [0.5, 0.9])
def test_overall_quantile_density_method(results, events, population, p):
    _, pdf = events
    n_units = population.groupby("flight").size().to_dict()
    expected = ref.quantile(pdf, n_units, "latency_ms", p, "t2")
    actual = row(results.overall(), metric="latency", quantile=p, treatment="t2")
    check(actual, {k: expected[k] for k in ("value_control", "value_treatment", "n_control", "p_value")})
    assert actual.outcome == "better"  # latency went down and its desired direction is down


def test_daily_uses_units_exposed_by_that_day(results, events, population):
    _, pdf = events
    day = ref.START + dt.timedelta(days=2)
    exposed = population[population["first_day"] <= day]
    clicks = ref.unit_values(pdf[pdf["day"] == day], exposed, "clicks")
    check(row(results.daily("clicks"), day=pd.Timestamp(day), treatment="t1"), ref.welch(clicks, "t1"))
    assert results.daily("clicks")["day"].nunique() == ref.DAYS


def test_user_segments(results, events, population):
    _, pdf = events
    ios = ref.unit_values(pdf, population[population["os"] == "ios"], "clicks")
    by_os = results.by_user_segment("os")
    check(row(by_os, metric="clicks", os="ios", treatment="t1"), ref.welch(ios, "t1"))
    assert set(by_os["os"]) == {"ios", "android", "(null)"}
    two_way = results.by_user_segment(("os", "country"))
    assert len(two_way[two_way["metric"] == "clicks"]) == 3 * 2 * 2  # os x country x treatments
    ios_gb = population[(population["os"] == "ios") & (population["country"] == "GB")]
    check(row(two_way, metric="clicks", os="ios", country="GB", treatment="t1"),
          ref.welch(ref.unit_values(pdf, ios_gb, "clicks"), "t1"))


def test_metric_segments_keep_the_flight_denominator(results, events, population):
    _, pdf = events
    video = ref.unit_values(pdf[pdf["media_type"] == "video"], population, "clicks")
    by_media = results.by_metric_segment("media_type")
    check(row(by_media, metric="clicks", media_type="video", treatment="t1"), ref.welch(video, "t1"))
    video_clickers = ref.unit_values(pdf[pdf["media_type"] == "video"], population, "clicks", how="any")
    check(row(by_media, metric="clickers", media_type="video", treatment="t1"), ref.welch(video_clickers, "t1"))
    clicks = by_media[(by_media["metric"] == "clicks") & (by_media["treatment"] == "t1")]
    assert clicks["segment_share_control"].sum() == pytest.approx(1.0)
    assert clicks["segment_contribution"].sum() == pytest.approx(1.0)
    overall = row(results.overall(), metric="clicks", treatment="t1")
    assert clicks["value_control"].sum() == pytest.approx(overall.value_control)


def test_hte_detects_the_ios_only_effect(results):
    hte = results.hte()
    t1 = row(hte, metric="clicks", treatment="t1", user_segment="os", metric_segment="overall")
    assert t1.hte_detected and t1.segments_tested == 2  # "(null)" never takes part
    assert t1.hte_pattern == "only some segments affected"
    t2 = row(hte, metric="clicks", treatment="t2", user_segment="os", metric_segment="overall")
    assert not t2.hte_detected
    by_os = results.by_user_segment("os")
    tested = by_os[(by_os["metric"] == "clicks") & (by_os["treatment"] == "t1") & by_os["hte_tested"]]
    _, _, p_value = stats.heterogeneity_test(tested["diff"], tested["std_error"] ** 2)
    assert t1.hte_p_value == pytest.approx(p_value)


def test_srm_bh_and_sample_size_flags(results):
    overall = results.overall()
    clicks = row(overall, metric="clicks", treatment="t1")
    assert clicks.srm_p_value == pytest.approx(stats.srm_p_value([clicks.n_control], [clicks.n_treatment], 1, 1)[0])
    srm = results.srm()
    assert len(srm[srm["user_segment"] == "overall"]) == 2  # one test per control/treatment pair
    pair = row(srm, metric_table="events", user_segment="overall", treatment="t1")
    assert (pair.n_control, pair.n_treatment) == (clicks.n_control, clicks.n_treatment)
    assert pair.p_value == pytest.approx(clicks.srm_p_value) and pair.expected_treatment_share == 0.5
    ios = row(srm, user_segment="os", user_segment_value="ios", treatment="t2")
    by_os = row(results.by_user_segment("os"), metric="clicks", os="ios", treatment="t2")
    assert ios.p_value == pytest.approx(by_os.srm_p_value)
    family = overall[overall["treatment"] == "t1"]
    assert len(family) == 7  # 5 unit metrics + 2 quantiles
    np.testing.assert_allclose(family["p_value_bh"], stats.bh_adjust(family["p_value"]))
    null_os = row(results.by_user_segment("os"), metric="clicks", os="(null)", treatment="t1")
    assert not null_os.sufficient_sample and not np.isnan(null_os.p_value)  # flagged, not blanked here


def test_display_and_raw_statistics(results):
    html = results.to_html()
    assert "clicks_per_session" in html and "latency p50" in html and "latency p90" in html
    raw = results.sufficient_stats["events"]
    assert {"clicks__sum", "clicks__sumsq", "clickers__hits", "latency__p0__q", "latency__p1__density"} <= set(raw.columns)
    assert repr(results).startswith("<ExperimentResults: ")
    assert results.settings["mode"] == "pre-joined"

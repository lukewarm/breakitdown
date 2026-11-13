"""How often an analysis reads its sources: each once, however the results are broken down."""

import pytest

from breakitdown import Binary, Count, Experiment, FlightMapping, MetricTable, Quantile
from . import reference as ref


def counted(spark, records, schema):
    """``records`` as a DataFrame, and an accumulator that counts every row read from it."""
    reads = spark.sparkContext.accumulator(0)

    def read(rows):
        for r in rows:
            reads.add(1)
            yield r

    return spark.createDataFrame(spark.sparkContext.parallelize(records, 2).mapPartitions(read), schema), reads


@pytest.mark.parametrize("user_segments", [[], ["os", ("os", "country")]])
@pytest.mark.parametrize("exclude", [True, False])
def test_a_pre_joined_table_is_read_once(spark, user_segments, exclude):
    records = ref.event_records(units=600)
    events, reads = counted(spark, records, ref.EVENT_SCHEMA)
    experiment = Experiment("control", ["t1", "t2"], ref.START, ref.END, user_segments=user_segments,
                            exclude_multi_flight_units=exclude)
    metrics = [Count("clicks"), Binary("clicks", name="clickers"), Quantile("latency_ms", [0.5])]
    experiment.analyze(MetricTable(events, metrics, metric_segments=["media_type"]), enforce_min_sample=False)
    assert reads.value == len(records)


def test_a_mapping_and_each_metric_table_are_read_once(spark):
    records = ref.event_records(units=600)
    assignments = list({r["ruid"]: (r["ruid"], r["flight"], r["os"]) for r in records}.values())
    mapping, mapping_reads = counted(spark, assignments, "ruid string, flight string, os string")
    events, event_reads = counted(spark, records, ref.EVENT_SCHEMA)
    experiment = Experiment("control", ["t1", "t2"], ref.START, ref.END, user_segments=["os"],
                            mapping=FlightMapping(mapping))
    experiment.analyze([MetricTable(events, [Count("clicks")], name="clicks"),
                        MetricTable(events, [Quantile("latency_ms", [0.5])], name="latency")], enforce_min_sample=False)
    assert mapping_reads.value == len(assignments)
    assert event_reads.value == 2 * len(records)

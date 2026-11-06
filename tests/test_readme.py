"""Runs every ```python block of README.md, in order and in one namespace, against synthetic tables.

Keeps the README's examples working: if an example breaks, this test names the block.
"""

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
START = dt.date(2024, 3, 1)
DAYS = 14
UTC = dt.timezone.utc


def readme_python_blocks():
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)


def build_readme_tables(spark, users=3000, seed=0):
    """The ``analytics.*`` tables the README examples read. t1 lifts photo clicks by 20%; t2 cuts latency."""
    rng = np.random.default_rng(seed)
    app_events, latency, purchases, assignment, daily_stats = [], [], [], [], []
    for u in range(users):
        ruid = f"u{u}"
        flight = str(rng.choice(["control", "t1", "t2"]))
        os_, country = str(rng.choice(["ios", "android"])), str(rng.choice(["US", "GB", "DE"], p=[0.5, 0.3, 0.2]))
        first = int(rng.integers(0, DAYS))
        exposure = dt.datetime.combine(START + dt.timedelta(days=first), dt.time(), tzinfo=UTC)
        assignment.append((ruid, flight, exposure + dt.timedelta(seconds=int(rng.integers(0, 3600))), os_, country))
        for d in range(first, DAYS):
            if d > first and rng.random() < 0.5:
                continue
            day = START + dt.timedelta(days=d)
            sessions = 1 + int(rng.poisson(1.0))
            photo = int(rng.poisson(0.12 * sessions * (1.2 if flight == "t1" else 1.0)))
            app_events.append((ruid, flight, day, os_, country, "photo", photo, sessions))
            clicks = photo
            if rng.random() < 0.4:
                video = int(rng.poisson(0.08 * sessions))
                app_events.append((ruid, flight, day, os_, country, "video", video, 0))
                clicks += video
            daily_stats.append((ruid, flight, day.strftime("%Y%m%d"), clicks))
            midnight = dt.datetime.combine(day, dt.time(), tzinfo=UTC)
            for _ in range(sessions):
                latency.append((ruid, flight, midnight + dt.timedelta(seconds=int(rng.integers(0, 86400))), os_, country,
                                max(1, int(rng.poisson(110 if flight == "t2" else 120)))))
            if rng.random() < 0.08:
                purchases.append((ruid, midnight + dt.timedelta(hours=20), round(float(rng.exponential(20.0)), 2)))

    spark.sql("CREATE DATABASE IF NOT EXISTS analytics")
    tables = {
        "app_events": (app_events, "ruid string, flight string, event_time date, os string, country string, "
                                   "media_type string, clicks int, sessions int"),
        "request_latency": (latency, "ruid string, flight string, event_time timestamp, os string, country string, "
                                     "latency_ms int"),
        "purchases": (purchases, "ruid string, event_time timestamp, revenue double"),
        "flight_assignment": (assignment, "ruid string, flight string, exposure_time timestamp, os string, country string"),
        "daily_user_stats": (daily_stats, "user_id string, variant string, ds string, clicks int"),
    }
    for name, (rows, schema) in tables.items():
        spark.createDataFrame(rows, schema).write.mode("overwrite").saveAsTable(f"analytics.{name}")


def test_readme_examples_run(spark):
    build_readme_tables(spark)
    blocks = readme_python_blocks()
    assert len(blocks) >= 5
    namespace = {"spark": spark}
    for number, block in enumerate(blocks, start=1):
        try:
            exec(compile(block, f"<README.md python block {number}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report which block broke
            pytest.fail(f"README python block {number} failed: {exc!r}\n\n{block}")
    assert not namespace["results"].overall().empty

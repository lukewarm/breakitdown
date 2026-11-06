import pytest


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("breakitdown-tests")
            .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("spark-warehouse")))
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    except Exception as exc:  # e.g. no Java runtime available
        pytest.skip(f"Spark is not available: {exc}")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()

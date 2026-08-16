"""Pytest fixtures for testing."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("WeatherPipelineTests")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    spark.stop()

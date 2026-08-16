"""Spark session management."""

import logging
from contextlib import contextmanager
from typing import Generator

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "WeatherPredictionPipeline") -> SparkSession:
    """Create and configure a Spark session.

    Args:
        app_name: Name for the Spark application

    Returns:
        Configured SparkSession
    """
    logger.info(f"Creating Spark session: {app_name}")
    spark = SparkSession.builder.appName(app_name).getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")

    return spark


@contextmanager
def spark_session_scope(
    app_name: str = "WeatherPredictionPipeline",
) -> Generator[SparkSession, None, None]:
    """Context manager for Spark session lifecycle.

    Usage:
        with spark_session_scope() as spark:
            df = spark.read.parquet(...)
    """
    spark = create_spark_session(app_name)
    try:
        yield spark
    finally:
        logger.info("Stopping Spark session")
        spark.stop()

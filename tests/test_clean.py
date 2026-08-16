"""Test data cleaning."""

import pytest
from pyspark.sql import Row

from weather_pipeline.clean import drop_null_target, validate_no_nulls


def test_drop_null_target(spark):
    """Test that null target rows are dropped."""
    data = [
        Row(city="A", min_temp_c=10.0),
        Row(city="B", min_temp_c=None),
        Row(city="C", min_temp_c=5.0),
        Row(city="D", min_temp_c=None),
    ]

    df = spark.createDataFrame(data)
    result = drop_null_target(df, target_col="min_temp_c")

    assert result.count() == 2

    cities = [row.city for row in result.collect()]
    assert "A" in cities
    assert "C" in cities
    assert "B" not in cities
    assert "D" not in cities


def test_validate_no_nulls(spark, caplog):
    """Test that null validation logs correctly."""
    data = [
        Row(city="A", temp=10.0),
        Row(city="B", temp=5.0),
    ]

    df = spark.createDataFrame(data)

    # Should log "no nulls found"
    result = validate_no_nulls(df, stage="test_stage")

    assert result.count() == 2
    assert "No nulls found" in caplog.text or "0 total nulls" in caplog.text

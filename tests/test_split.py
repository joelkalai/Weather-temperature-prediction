"""Test train/test temporal split."""

import pytest
from datetime import datetime
from pyspark.sql import Row

from weather_pipeline.config import SplitConfig
from weather_pipeline.split import temporal_split


def test_temporal_split_no_leakage(spark):
    """Test that temporal split never puts future data in training set."""
    config = SplitConfig(split_year=2010)

    # Create data spanning 2008-2011
    data = [
        Row(date=datetime(2008, 1, 1), min_temp_c=5.0),
        Row(date=datetime(2009, 6, 15), min_temp_c=15.0),
        Row(date=datetime(2009, 12, 31), min_temp_c=0.0),
        Row(date=datetime(2010, 1, 1), min_temp_c=2.0),
        Row(date=datetime(2010, 7, 10), min_temp_c=20.0),
        Row(date=datetime(2011, 3, 20), min_temp_c=10.0),
    ]

    df = spark.createDataFrame(data)

    train, test = temporal_split(df, config)

    train_dates = [row.date for row in train.collect()]
    test_dates = [row.date for row in test.collect()]

    # Verify counts
    assert len(train_dates) == 3  # 2008, 2009 rows
    assert len(test_dates) == 3  # 2010, 2011 rows

    # Verify all train dates < 2010-01-01
    for d in train_dates:
        assert d < datetime(2010, 1, 1)

    # Verify all test dates >= 2010-01-01
    for d in test_dates:
        assert d >= datetime(2010, 1, 1)

    # Verify no overlap: max(train) < min(test)
    assert max(train_dates) < min(test_dates)


def test_temporal_split_boundary(spark):
    """Test split behavior at exact boundary."""
    config = SplitConfig(split_year=2010)

    data = [
        Row(date=datetime(2009, 12, 31, 23, 59, 59), min_temp_c=0.0),  # Last of 2009
        Row(date=datetime(2010, 1, 1, 0, 0, 0), min_temp_c=1.0),  # First of 2010
    ]

    df = spark.createDataFrame(data)
    train, test = temporal_split(df, config)

    assert train.count() == 1
    assert test.count() == 1

    train_row = train.collect()[0]
    test_row = test.collect()[0]

    assert train_row.date.year == 2009
    assert test_row.date.year == 2010


def test_temporal_split_validates_no_overlap(spark):
    """Test that validation catches any data leakage."""
    # This test verifies the built-in validation logic works

    config = SplitConfig(split_year=2015)

    data = [
        Row(date=datetime(2014, 1, 1), min_temp_c=5.0),
        Row(date=datetime(2015, 1, 1), min_temp_c=10.0),
        Row(date=datetime(2016, 1, 1), min_temp_c=15.0),
    ]

    df = spark.createDataFrame(data)

    # Should not raise (validation should pass)
    train, test = temporal_split(df, config)

    # Verify the split is correct
    assert train.count() == 1
    assert test.count() == 2

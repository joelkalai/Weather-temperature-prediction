"""Test feature engineering to ensure no data leakage."""

import pytest
from datetime import datetime, timedelta
from pyspark.sql import Row

from weather_pipeline.config import FeatureConfig
from weather_pipeline.features import (
    add_lagged_features,
    add_rolling_features,
    add_temporal_features,
    add_trend_features,
)


def test_temporal_features(spark):
    """Test that temporal features are correctly extracted."""
    data = [
        Row(date=datetime(2020, 3, 15), min_temp_c=5.0),
        Row(date=datetime(2020, 12, 31), min_temp_c=0.0),
    ]
    df = spark.createDataFrame(data)

    result = add_temporal_features(df)

    rows = result.collect()
    assert rows[0].year == 2020
    assert rows[0].month == 3
    assert rows[0].day == 15

    assert rows[1].year == 2020
    assert rows[1].month == 12
    assert rows[1].day == 31


def test_lagged_features_no_same_day_leakage(spark):
    """Test that lagged features only use strictly prior data.

    For row N, prev_day_min_temp should be the min_temp_c from row N-1.
    """
    config = FeatureConfig()

    # Create simple time series for one city
    dates = [datetime(2020, 1, i) for i in range(1, 6)]
    temps = [10.0, 12.0, 8.0, 15.0, 11.0]

    data = [
        Row(
            station_id="TEST",
            city_name="TestCity",
            date=d,
            min_temp_c=t,
            avg_temp_c=t + 2,
            max_temp_c=t + 5,
        )
        for d, t in zip(dates, temps)
    ]

    df = spark.createDataFrame(data)
    result = add_lagged_features(df, config)

    rows = sorted(result.collect(), key=lambda r: r.date)

    # First row should have null lagged features
    assert rows[0].prev_day_min_temp is None

    # Second row (Jan 2) should have prev_day_min_temp = Jan 1's min_temp_c = 10.0
    assert rows[1].prev_day_min_temp == 10.0
    assert rows[1].prev_day_avg_temp == 12.0  # Jan 1's avg
    assert rows[1].prev_day_max_temp == 15.0  # Jan 1's max

    # Third row (Jan 3) should have prev_day_min_temp = Jan 2's min_temp_c = 12.0
    assert rows[2].prev_day_min_temp == 12.0

    # Verify NO row has same-day leakage (prev_day_min_temp != min_temp_c for same row)
    for row in rows[1:]:
        assert row.prev_day_min_temp != row.min_temp_c


def test_rolling_features_exclude_current_row(spark):
    """Test that rolling windows exclude the current row (no same-day leakage).

    For a 7-day rolling average on row N, the window should be days N-7 to N-1,
    NOT including day N.
    """
    config = FeatureConfig(rolling_window_7d=3)  # Use 3-day for simpler testing

    # Create a simple series: 1, 2, 3, 4, 5, 6, 7, 8
    dates = [datetime(2020, 1, i) for i in range(1, 9)]
    temps = list(range(1, 9))

    data = [
        Row(station_id="TEST", city_name="TestCity", date=d, min_temp_c=float(t))
        for d, t in zip(dates, temps)
    ]

    df = spark.createDataFrame(data)
    result = add_rolling_features(df, config)

    rows = sorted(result.collect(), key=lambda r: r.date)

    # First 3 rows should have null (not enough history)
    assert rows[0].rolling_7d_min_temp is None
    assert rows[1].rolling_7d_min_temp is None
    assert rows[2].rolling_7d_min_temp is None

    # Row 4 (min_temp_c=4): rolling avg should be mean of [1, 2, 3] = 2.0
    assert rows[3].rolling_7d_min_temp == pytest.approx(2.0)
    assert rows[3].min_temp_c == 4.0  # Current value NOT included

    # Row 5 (min_temp_c=5): rolling avg should be mean of [2, 3, 4] = 3.0
    assert rows[4].rolling_7d_min_temp == pytest.approx(3.0)
    assert rows[4].min_temp_c == 5.0

    # Verify: rolling average is ALWAYS < current value in this increasing series
    # (if current was included, rolling avg would be higher)
    for row in rows[3:]:
        assert row.rolling_7d_min_temp < row.min_temp_c


def test_trend_features(spark):
    """Test that temperature trend is correctly computed."""
    config = FeatureConfig(trend_lag=2)

    dates = [datetime(2020, 1, i) for i in range(1, 6)]
    temps = [10.0, 12.0, 8.0, 15.0, 11.0]

    data = [
        Row(
            station_id="TEST",
            city_name="TestCity",
            date=d,
            min_temp_c=t,
            avg_temp_c=t,
            max_temp_c=t,
        )
        for d, t in zip(dates, temps)
    ]

    df = spark.createDataFrame(data)

    # First add lagged features (required for trend calculation)
    df = add_lagged_features(df, config)
    result = add_trend_features(df, config)

    rows = sorted(result.collect(), key=lambda r: r.date)

    # First two rows: not enough data
    assert rows[0].temp_trend is None
    assert rows[1].temp_trend is None

    # Third row (Jan 3, min=8):
    #   prev_day_min_temp = 12 (Jan 2)
    #   lag(min_temp_c, 2) = 10 (Jan 1)
    #   trend = 12 - 10 = 2.0
    assert rows[2].temp_trend == pytest.approx(2.0)


def test_feature_pipeline_produces_only_historical_features(spark):
    """Integration test: ensure full feature pipeline uses only historical data.

    This is a critical test to prevent data leakage. Every feature must be
    derivable from strictly prior rows.
    """
    config = FeatureConfig()

    # Create a longer series to test properly
    dates = [datetime(2020, 1, i) for i in range(1, 50)]
    temps = [10.0 + i * 0.5 for i in range(len(dates))]

    data = [
        Row(
            station_id="TEST",
            city_name="TestCity",
            date=d,
            min_temp_c=t,
            avg_temp_c=t + 2,
            max_temp_c=t + 5,
        )
        for d, t in zip(dates, temps)
    ]

    df = spark.createDataFrame(data)

    # Run feature pipeline
    df = add_temporal_features(df)
    df = add_lagged_features(df, config)
    df = add_rolling_features(df, config)
    df = add_trend_features(df, config)

    # Filter to rows with complete features (after warm-up period)
    df_complete = df.filter(
        df.prev_day_min_temp.isNotNull()
        & df.rolling_7d_min_temp.isNotNull()
        & df.rolling_30d_min_temp.isNotNull()
        & df.temp_trend.isNotNull()
    )

    rows = df_complete.collect()

    # For each row, verify that NO feature equals the current min_temp_c
    # (that would indicate same-day leakage)
    for row in rows:
        assert row.prev_day_min_temp != row.min_temp_c
        assert row.rolling_7d_min_temp != row.min_temp_c
        assert row.rolling_30d_min_temp != row.min_temp_c

        # Rolling averages should be based on history only
        # In an increasing series, they should lag behind current value
        assert row.rolling_7d_min_temp < row.min_temp_c
        assert row.rolling_30d_min_temp < row.min_temp_c

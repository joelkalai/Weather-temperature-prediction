"""Feature engineering: lagged temps, rolling windows, temporal features.

CRITICAL: All features must be derivable from strictly prior data only.
No same-day features (avg_temp_c, max_temp_c) are used to avoid data leakage.
"""

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import FeatureConfig

logger = logging.getLogger(__name__)


def add_temporal_features(df: DataFrame) -> DataFrame:
    """Add year, month, day features from date column.

    Args:
        df: DataFrame with 'date' column

    Returns:
        DataFrame with year, month, day columns added
    """
    logger.info("Adding temporal features: year, month, day")

    df = df.withColumn("year", F.year("date"))
    df = df.withColumn("month", F.month("date"))
    df = df.withColumn("day", F.dayofmonth("date"))

    return df


def add_lagged_features(df: DataFrame, config: FeatureConfig) -> DataFrame:
    """Add lagged temperature features using window functions.

    Creates:
    - prev_day_min_temp: Previous day's minimum temperature
    - prev_day_avg_temp: Previous day's average temperature
    - prev_day_max_temp: Previous day's maximum temperature

    Window is partitioned by (station_id, city_name) and ordered by date.

    Args:
        df: DataFrame with temperature columns
        config: Feature configuration with lag parameters

    Returns:
        DataFrame with lagged features added
    """
    logger.info(f"Adding lagged features (lag={config.lag_days} day)")

    # Window spec: partition by city, order by date
    window_spec = Window.partitionBy("station_id", "city_name").orderBy("date")

    df = df.withColumn(
        "prev_day_min_temp", F.lag("min_temp_c", config.lag_days).over(window_spec)
    )
    df = df.withColumn(
        "prev_day_avg_temp", F.lag("avg_temp_c", config.lag_days).over(window_spec)
    )
    df = df.withColumn(
        "prev_day_max_temp", F.lag("max_temp_c", config.lag_days).over(window_spec)
    )

    return df


def add_rolling_features(df: DataFrame, config: FeatureConfig) -> DataFrame:
    """Add rolling window statistics.

    Creates:
    - rolling_7d_min_temp: 7-day rolling average of min_temp_c
    - rolling_30d_min_temp: 30-day rolling average of min_temp_c

    Rolling windows look ONLY at strictly prior rows (rowsBetween(-N, -1)).

    Args:
        df: DataFrame with min_temp_c column
        config: Feature configuration with window sizes

    Returns:
        DataFrame with rolling features added
    """
    logger.info(
        f"Adding rolling features (windows: {config.rolling_window_7d}d, {config.rolling_window_30d}d)"
    )

    # Window spec for rolling averages: partition by city, order by date
    # rowsBetween(-N, -1) means N days back to yesterday (excludes current row)
    window_7d = (
        Window.partitionBy("station_id", "city_name")
        .orderBy("date")
        .rowsBetween(-config.rolling_window_7d, -1)
    )
    window_30d = (
        Window.partitionBy("station_id", "city_name")
        .orderBy("date")
        .rowsBetween(-config.rolling_window_30d, -1)
    )

    df = df.withColumn("rolling_7d_min_temp", F.avg("min_temp_c").over(window_7d))
    df = df.withColumn("rolling_30d_min_temp", F.avg("min_temp_c").over(window_30d))

    return df


def add_trend_features(df: DataFrame, config: FeatureConfig) -> DataFrame:
    """Add temperature trend indicators.

    Creates:
    - temp_trend: Difference between yesterday's min temp and the day before

    Args:
        df: DataFrame with prev_day_min_temp already computed
        config: Feature configuration with trend lag

    Returns:
        DataFrame with trend features added
    """
    logger.info(f"Adding trend features (lag={config.trend_lag})")

    window_spec = Window.partitionBy("station_id", "city_name").orderBy("date")

    df = df.withColumn(
        "temp_trend",
        F.col("prev_day_min_temp") - F.lag("min_temp_c", config.trend_lag).over(window_spec),
    )

    return df


def select_final_features(df: DataFrame) -> DataFrame:
    """Select final feature set, excluding same-day leakage features.

    IMPORTANT: avg_temp_c and max_temp_c are NOT included as features
    because they are measured on the same day as the target (min_temp_c).
    Only historically available features are kept.

    Args:
        df: DataFrame with all engineered features

    Returns:
        DataFrame with final feature columns only
    """
    logger.info("Selecting final feature set (no same-day leakage)")

    final_cols = [
        "city_name",
        "date",
        "season",
        "latitude",
        "longitude",
        "region",
        "continent",
        "year",
        "month",
        "day",
        "prev_day_min_temp",
        "prev_day_avg_temp",
        "prev_day_max_temp",
        "rolling_7d_min_temp",
        "rolling_30d_min_temp",
        "temp_trend",
        "min_temp_c",  # target
    ]

    df = df.select(final_cols)

    return df


def drop_incomplete_windows(df: DataFrame) -> DataFrame:
    """Drop rows where lagged/rolling features are null.

    This removes the first ~30 rows per city where window functions
    don't have enough history.

    Args:
        df: DataFrame with engineered features

    Returns:
        DataFrame with incomplete window rows removed
    """
    logger.info("Dropping rows with incomplete windows (null lagged/rolling features)")

    initial_count = df.count()

    df = df.filter(
        F.col("prev_day_min_temp").isNotNull()
        & F.col("rolling_7d_min_temp").isNotNull()
        & F.col("rolling_30d_min_temp").isNotNull()
        & F.col("temp_trend").isNotNull()
    )

    # Also drop any remaining nulls
    df = df.na.drop()

    final_count = df.count()
    dropped = initial_count - final_count

    logger.info(f"Dropped {dropped} rows with incomplete windows ({initial_count} → {final_count})")

    return df


def engineer_features(df: DataFrame, config: FeatureConfig) -> DataFrame:
    """Run complete feature engineering pipeline.

    Args:
        df: Cleaned DataFrame
        config: Feature configuration

    Returns:
        DataFrame with all features engineered, incomplete windows dropped
    """
    df = add_temporal_features(df)
    df = add_lagged_features(df, config)
    df = add_rolling_features(df, config)
    df = add_trend_features(df, config)
    df = select_final_features(df)
    df = drop_incomplete_windows(df)

    return df

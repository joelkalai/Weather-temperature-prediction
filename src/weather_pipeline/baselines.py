"""Naive persistence baselines for honest model comparison.

These baselines predict tomorrow's minimum temperature using simple heuristics:
- Persistence (min): Tomorrow's min = today's min
- Persistence (avg): Tomorrow's min = today's avg

The GBT model should be compared against these baselines to validate
that it provides meaningful improvement over trivial predictions.
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def persistence_baseline_min(df: DataFrame) -> DataFrame:
    """Naive persistence baseline: predict tomorrow's min_temp_c as today's min_temp_c.

    Uses prev_day_min_temp as the prediction, since that's today's min temperature.

    Args:
        df: DataFrame with prev_day_min_temp and min_temp_c

    Returns:
        DataFrame with 'prediction' column added (= prev_day_min_temp)
    """
    logger.info("Creating persistence baseline (min): prediction = prev_day_min_temp")

    df = df.withColumn("prediction", F.col("prev_day_min_temp"))

    return df


def persistence_baseline_avg(df: DataFrame) -> DataFrame:
    """Naive persistence baseline: predict tomorrow's min_temp_c as today's avg_temp_c.

    Uses prev_day_avg_temp as the prediction, since that's today's average temperature.

    Args:
        df: DataFrame with prev_day_avg_temp and min_temp_c

    Returns:
        DataFrame with 'prediction' column added (= prev_day_avg_temp)
    """
    logger.info("Creating persistence baseline (avg): prediction = prev_day_avg_temp")

    df = df.withColumn("prediction", F.col("prev_day_avg_temp"))

    return df


def rolling_mean_baseline(df: DataFrame) -> DataFrame:
    """Baseline using 7-day rolling mean as prediction.

    Args:
        df: DataFrame with rolling_7d_min_temp and min_temp_c

    Returns:
        DataFrame with 'prediction' column added (= rolling_7d_min_temp)
    """
    logger.info("Creating rolling mean baseline: prediction = rolling_7d_min_temp")

    df = df.withColumn("prediction", F.col("rolling_7d_min_temp"))

    return df

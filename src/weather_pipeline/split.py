"""Train/test temporal split to prevent data leakage."""

import logging
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import SplitConfig

logger = logging.getLogger(__name__)


def temporal_split(
    df: DataFrame, config: SplitConfig
) -> Tuple[DataFrame, DataFrame]:
    """Split data into train/test by date threshold.

    Train: all data before split_year
    Test: all data >= split_year

    This ensures no future data leaks into training.

    Args:
        df: DataFrame with 'date' column
        config: Split configuration with split_year

    Returns:
        Tuple of (train_df, test_df)
    """
    split_date = f"{config.split_year}-01-01"
    logger.info(f"Splitting data: train < {split_date}, test >= {split_date}")

    train_df = df.filter(F.col("date") < split_date)
    test_df = df.filter(F.col("date") >= split_date)

    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count

    logger.info(
        f"Train: {train_count} ({train_count/total*100:.1f}%), "
        f"Test: {test_count} ({test_count/total*100:.1f}%)"
    )

    # Validation: ensure no overlap
    min_test_date = test_df.agg(F.min("date")).collect()[0][0]
    max_train_date = train_df.agg(F.max("date")).collect()[0][0]

    if max_train_date >= min_test_date:
        raise ValueError(
            f"Data leakage detected: max train date ({max_train_date}) "
            f">= min test date ({min_test_date})"
        )

    logger.info(f"Validated: max train date = {max_train_date}, min test date = {min_test_date}")

    return train_df, test_df

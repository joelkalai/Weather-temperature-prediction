"""Data cleaning: null handling, deduplication, schema validation."""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def drop_null_target(df: DataFrame, target_col: str = "min_temp_c") -> DataFrame:
    """Drop records where the target variable is null.

    Args:
        df: Input DataFrame
        target_col: Name of the target column (default: min_temp_c)

    Returns:
        DataFrame with null target rows removed
    """
    initial_count = df.count()
    df_clean = df.filter(F.col(target_col).isNotNull())
    final_count = df_clean.count()

    dropped = initial_count - final_count
    logger.info(
        f"Dropped {dropped} rows with null {target_col} ({initial_count} → {final_count})"
    )

    return df_clean


def validate_no_nulls(df: DataFrame, stage: str = "final") -> DataFrame:
    """Validate that no nulls remain in the DataFrame.

    Logs null counts for all columns. Raises warning if nulls are found.

    Args:
        df: DataFrame to validate
        stage: Description of the current stage (for logging)

    Returns:
        Same DataFrame (for chaining)
    """
    logger.info(f"Validating no nulls at stage: {stage}")

    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).collect()[0].asDict()

    total_nulls = sum(null_counts.values())

    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} total nulls at stage '{stage}':")
        for col, count in null_counts.items():
            if count > 0:
                logger.warning(f"  {col}: {count} nulls")
    else:
        logger.info(f"No nulls found at stage '{stage}'")

    return df


def clean_data(df: DataFrame) -> DataFrame:
    """Run complete data cleaning pipeline.

    Args:
        df: Raw ingested DataFrame

    Returns:
        Cleaned DataFrame with nulls removed
    """
    df = drop_null_target(df, target_col="min_temp_c")
    df = validate_no_nulls(df, stage="after_cleaning")

    return df

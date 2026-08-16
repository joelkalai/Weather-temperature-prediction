"""Model evaluation: compute metrics and feature importance."""

import logging
from typing import Dict, List

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def compute_metrics(
    df: DataFrame, metrics: List[str] = None
) -> Dict[str, float]:
    """Compute regression metrics on predictions.

    Args:
        df: DataFrame with 'min_temp_c' (label) and 'prediction' columns
        metrics: List of metric names (default: ["mae", "rmse", "r2"])

    Returns:
        Dictionary mapping metric names to values
    """
    if metrics is None:
        metrics = ["mae", "rmse", "r2"]

    results = {}

    for metric in metrics:
        evaluator = RegressionEvaluator(
            labelCol="min_temp_c", predictionCol="prediction", metricName=metric
        )
        value = evaluator.evaluate(df)
        results[metric] = value

    return results


def log_metrics(model_name: str, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> None:
    """Log metrics in a formatted table.

    Args:
        model_name: Name of the model
        train_metrics: Dictionary of training metrics
        test_metrics: Dictionary of test metrics
    """
    logger.info(f"\n=== {model_name} Evaluation ===")
    logger.info(f"{'Metric':<6s} {'Train':>10s} {'Test':>10s}")
    logger.info("-" * 28)

    for metric in train_metrics.keys():
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        logger.info(f"{metric.upper():<6s} {train_val:>10.4f} {test_val:>10.4f}")


def extract_feature_importance(model, num_cols: List[str]) -> List[tuple]:
    """Extract and sort feature importances from a GBT model.

    Args:
        model: Trained PipelineModel with GBT as final stage
        num_cols: List of numerical feature column names (in order)

    Returns:
        List of (feature_name, importance) tuples, sorted by importance descending
    """
    # GBT is the last stage in the pipeline
    gbt = model.stages[-1]

    importances = gbt.featureImportances.toArray()

    # First N importances correspond to numerical features
    # (categorical features are one-hot encoded and come after)
    num_importances = importances[: len(num_cols)]

    feature_importance = list(zip(num_cols, num_importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def log_feature_importance(feature_importance: List[tuple], top_n: int = 15) -> None:
    """Log top N features by importance.

    Args:
        feature_importance: List of (feature_name, importance) tuples
        top_n: Number of top features to display
    """
    logger.info(f"\n=== Top {top_n} Feature Importances ===")

    for name, importance in feature_importance[:top_n]:
        bar = "█" * int(importance * 50)
        logger.info(f"{name:25s} {importance:6.4f} {bar}")

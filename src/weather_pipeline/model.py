"""ML pipeline assembly: feature preprocessing + GBT regression with cross-validation."""

import logging
from typing import List

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from .config import ModelConfig

logger = logging.getLogger(__name__)


def build_preprocessing_stages(
    cat_cols: List[str], num_cols: List[str]
) -> List:
    """Build preprocessing pipeline stages.

    Pipeline:
    1. StringIndexer for each categorical column
    2. OneHotEncoder for indexed categories
    3. VectorAssembler for numerical features
    4. StandardScaler for numerical features
    5. Final VectorAssembler combining scaled numerical + encoded categorical

    Args:
        cat_cols: List of categorical column names
        num_cols: List of numerical column names

    Returns:
        List of preprocessing stages
    """
    logger.info(f"Building preprocessing stages: {len(cat_cols)} categorical, {len(num_cols)} numerical")

    # Stage 1: StringIndexer for each categorical column
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep")
        for c in cat_cols
    ]

    # Stage 2: OneHotEncoder for indexed columns
    encoder = OneHotEncoder(
        inputCols=[c + "_index" for c in cat_cols],
        outputCols=[c + "_ohe" for c in cat_cols],
    )

    # Stage 3: Assemble numerical features
    num_assembler = VectorAssembler(inputCols=num_cols, outputCol="num_features")

    # Stage 4: Scale numerical features
    scaler = StandardScaler(
        inputCol="num_features", outputCol="scaled_num_features"
    )

    # Stage 5: Combine all features
    final_assembler = VectorAssembler(
        inputCols=["scaled_num_features"] + [c + "_ohe" for c in cat_cols],
        outputCol="features",
    )

    return indexers + [encoder, num_assembler, scaler, final_assembler]


def get_feature_columns() -> tuple[List[str], List[str]]:
    """Define categorical and numerical feature columns.

    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    cat_cols = ["city_name", "season", "region", "continent"]

    # Only historically available features - no same-day data
    num_cols = [
        "latitude",
        "longitude",
        "year",
        "month",
        "day",
        "prev_day_min_temp",
        "prev_day_avg_temp",
        "prev_day_max_temp",
        "rolling_7d_min_temp",
        "rolling_30d_min_temp",
        "temp_trend",
    ]

    return cat_cols, num_cols


def build_cross_validator(config: ModelConfig) -> CrossValidator:
    """Build cross-validator with hyperparameter grid.

    Args:
        config: Model configuration with tuning parameters

    Returns:
        Configured CrossValidator
    """
    cat_cols, num_cols = get_feature_columns()
    preprocessing_stages = build_preprocessing_stages(cat_cols, num_cols)

    # GBT regressor (hyperparameters will be tuned)
    gbt = GBTRegressor(featuresCol="features", labelCol="min_temp_c")

    # Complete pipeline
    pipeline = Pipeline(stages=preprocessing_stages + [gbt])

    # Hyperparameter grid
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, config.max_iter_grid)
        .addGrid(gbt.maxDepth, config.max_depth_grid)
        .addGrid(gbt.stepSize, config.step_size_grid)
        .build()
    )

    total_combinations = len(param_grid)
    logger.info(f"Hyperparameter grid: {total_combinations} combinations")
    logger.info(f"  maxIter: {config.max_iter_grid}")
    logger.info(f"  maxDepth: {config.max_depth_grid}")
    logger.info(f"  stepSize: {config.step_size_grid}")

    # Evaluator
    evaluator = RegressionEvaluator(
        labelCol="min_temp_c", predictionCol="prediction", metricName=config.cv_metric
    )

    # Cross-validator
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=config.num_folds,
        parallelism=config.parallelism,
    )

    logger.info(
        f"Cross-validator: {config.num_folds} folds, parallelism={config.parallelism}, "
        f"metric={config.cv_metric}"
    )

    return cv


def log_cv_results(cv_model, gbt_param, param_grid) -> None:
    """Log cross-validation results for all parameter combinations.

    Args:
        cv_model: Fitted CrossValidatorModel
        gbt_param: GBTRegressor parameter object (for extracting params)
        param_grid: Parameter grid used in CV
    """
    logger.info("Cross-validation results:")

    avg_metrics = cv_model.avgMetrics

    for i, (params, metric) in enumerate(zip(param_grid, avg_metrics)):
        max_iter = params.get(gbt_param.maxIter)
        max_depth = params.get(gbt_param.maxDepth)
        step_size = params.get(gbt_param.stepSize)

        logger.info(
            f"  [{i+1}] maxIter={max_iter:3d}, maxDepth={max_depth:2d}, "
            f"stepSize={step_size:.2f} → metric={metric:.4f}"
        )


def log_best_params(best_model) -> dict:
    """Extract and log best hyperparameters from the trained model.

    Args:
        best_model: Best model from CrossValidatorModel

    Returns:
        Dictionary of best hyperparameters
    """
    # GBT is the last stage in the pipeline
    best_gbt = best_model.stages[-1]

    best_params = {
        "maxIter": best_gbt.getMaxIter(),
        "maxDepth": best_gbt.getMaxDepth(),
        "stepSize": best_gbt.getStepSize(),
    }

    logger.info("Best hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    return best_params

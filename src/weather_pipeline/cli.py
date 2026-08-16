"""Command-line interface for the weather prediction pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from . import baselines, clean, evaluate, features, ingest, model, split
from .config import PipelineConfig
from .session import spark_session_scope

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        PipelineConfig instance
    """
    logger.info(f"Loading config from {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return PipelineConfig.from_dict(config_dict)


def run_pipeline(config: PipelineConfig, stage: str = "all") -> None:
    """Run the complete or partial weather prediction pipeline.

    Args:
        config: Pipeline configuration
        stage: Which stage to run ("all", "ingest", "features", "train", "evaluate")
    """
    with spark_session_scope() as spark:
        if stage in ("all", "ingest"):
            logger.info("=" * 80)
            logger.info("STAGE: Data Ingestion")
            logger.info("=" * 80)
            data = ingest.ingest_data(spark, config.data)

            logger.info("=" * 80)
            logger.info("STAGE: Data Cleaning")
            logger.info("=" * 80)
            data = clean.clean_data(data)

        if stage in ("all", "features"):
            if stage == "features":
                # Load data if not running full pipeline
                logger.info("Loading previously ingested data...")
                # This would require saving intermediate results
                raise NotImplementedError(
                    "Running 'features' stage alone requires intermediate data. Use 'all' for now."
                )

            logger.info("=" * 80)
            logger.info("STAGE: Feature Engineering")
            logger.info("=" * 80)
            data = features.engineer_features(data, config.features)

            logger.info("=" * 80)
            logger.info("STAGE: Train/Test Split")
            logger.info("=" * 80)
            train_data, test_data = split.temporal_split(data, config.split)

        if stage in ("all", "train"):
            if stage == "train":
                raise NotImplementedError(
                    "Running 'train' stage alone requires intermediate data. Use 'all' for now."
                )

            logger.info("=" * 80)
            logger.info("STAGE: Model Training (Cross-Validation)")
            logger.info("=" * 80)

            cv = model.build_cross_validator(config.model)

            logger.info("Starting cross-validation... (this may take a while)")
            cv_model = cv.fit(train_data)
            logger.info("Cross-validation complete!")

            # Log results
            gbt_param = cv.getEstimator().getStages()[-1]
            model.log_cv_results(cv_model, gbt_param, cv.getEstimatorParamMaps())

            # Extract best model
            best_model = cv_model.bestModel
            best_params = model.log_best_params(best_model)

        if stage in ("all", "evaluate"):
            if stage == "evaluate":
                raise NotImplementedError(
                    "Running 'evaluate' stage alone requires intermediate data. Use 'all' for now."
                )

            logger.info("=" * 80)
            logger.info("STAGE: Model Evaluation")
            logger.info("=" * 80)

            # Generate predictions
            train_preds = best_model.transform(train_data)
            test_preds = best_model.transform(test_data)

            # Evaluate GBT model
            train_metrics = evaluate.compute_metrics(train_preds)
            test_metrics = evaluate.compute_metrics(test_preds)
            evaluate.log_metrics("GBT", train_metrics, test_metrics)

            # Baselines
            logger.info("\n" + "=" * 80)
            logger.info("STAGE: Baseline Comparisons")
            logger.info("=" * 80)

            # Persistence baseline (min)
            train_baseline_min = baselines.persistence_baseline_min(train_data)
            test_baseline_min = baselines.persistence_baseline_min(test_data)
            train_baseline_min_metrics = evaluate.compute_metrics(train_baseline_min)
            test_baseline_min_metrics = evaluate.compute_metrics(test_baseline_min)
            evaluate.log_metrics(
                "Persistence (min)", train_baseline_min_metrics, test_baseline_min_metrics
            )

            # Persistence baseline (avg)
            train_baseline_avg = baselines.persistence_baseline_avg(train_data)
            test_baseline_avg = baselines.persistence_baseline_avg(test_data)
            train_baseline_avg_metrics = evaluate.compute_metrics(train_baseline_avg)
            test_baseline_avg_metrics = evaluate.compute_metrics(test_baseline_avg)
            evaluate.log_metrics(
                "Persistence (avg)", train_baseline_avg_metrics, test_baseline_avg_metrics
            )

            # Rolling mean baseline
            train_baseline_roll = baselines.rolling_mean_baseline(train_data)
            test_baseline_roll = baselines.rolling_mean_baseline(test_data)
            train_baseline_roll_metrics = evaluate.compute_metrics(train_baseline_roll)
            test_baseline_roll_metrics = evaluate.compute_metrics(test_baseline_roll)
            evaluate.log_metrics(
                "Rolling 7d mean", train_baseline_roll_metrics, test_baseline_roll_metrics
            )

            # Feature importance
            logger.info("\n" + "=" * 80)
            logger.info("STAGE: Feature Importance")
            logger.info("=" * 80)

            _, num_cols = model.get_feature_columns()
            feature_importance = evaluate.extract_feature_importance(best_model, num_cols)
            evaluate.log_feature_importance(feature_importance)

            # Save results
            logger.info("\n" + "=" * 80)
            logger.info("STAGE: Save Results")
            logger.info("=" * 80)

            results = {
                "best_params": best_params,
                "models": {
                    "GBT": {"train": train_metrics, "test": test_metrics},
                    "Persistence_min": {
                        "train": train_baseline_min_metrics,
                        "test": test_baseline_min_metrics,
                    },
                    "Persistence_avg": {
                        "train": train_baseline_avg_metrics,
                        "test": test_baseline_avg_metrics,
                    },
                    "Rolling_7d_mean": {
                        "train": train_baseline_roll_metrics,
                        "test": test_baseline_roll_metrics,
                    },
                },
                "feature_importance": [
                    {"feature": name, "importance": float(imp)}
                    for name, imp in feature_importance
                ],
            }

            output_dir = Path(config.data.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "metrics.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

            # Print comparison table
            logger.info("\n" + "=" * 80)
            logger.info("FINAL COMPARISON (Test Set)")
            logger.info("=" * 80)
            logger.info(f"{'Model':<20s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s}")
            logger.info("-" * 52)

            for model_name, metrics_dict in results["models"].items():
                test_m = metrics_dict["test"]
                logger.info(
                    f"{model_name.replace('_', ' '):<20s} "
                    f"{test_m['mae']:>10.4f} "
                    f"{test_m['rmse']:>10.4f} "
                    f"{test_m['r2']:>10.4f}"
                )

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline complete!")
    logger.info("=" * 80)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Weather temperature prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "ingest", "features", "train", "evaluate"],
        default="all",
        help="Which stage to run (default: all)",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_pipeline(config, stage=args.stage)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

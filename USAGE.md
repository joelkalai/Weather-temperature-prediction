# Usage Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Update data paths in config
# Edit configs/default.yaml and set your data paths

# 3. Run the complete pipeline
python -m weather_pipeline.cli --config configs/default.yaml --stage all

# 4. View results
cat results/metrics.json
```

## Using the Makefile

```bash
# Install production dependencies
make install

# Install development dependencies (includes testing tools)
make install-dev

# Run the complete pipeline
make run

# Run tests (requires Java 11 or 17 for PySpark)
make test

# Lint code
make lint

# Format code
make format

# Type checking
make type-check

# Clean up generated files
make clean
```

## Running Specific Stages

The pipeline supports running individual stages:

```bash
# Run everything
python -m weather_pipeline.cli --stage all

# Run only data ingestion and cleaning
python -m weather_pipeline.cli --stage ingest

# Note: Individual stages (features, train, evaluate) are not yet implemented
# for standalone execution. Use --stage all for now.
```

## Customizing Configuration

Edit `configs/default.yaml`:

```yaml
data:
  raw_weather_path: "data/daily_weather.parquet"  # Update this
  cities_path: "data/cities.csv"                  # Update this
  countries_path: "data/countries.csv"            # Update this
  top_n_cities: 10                                # Change number of cities

features:
  lag_days: 1                    # Lag for previous day features
  rolling_window_7d: 7           # 7-day rolling window
  rolling_window_30d: 30         # 30-day rolling window

split:
  split_year: 2010              # Train < 2010, Test >= 2010

model:
  num_folds: 3                  # Cross-validation folds
  max_iter_grid: [50, 100]      # Hyperparameter grid
  max_depth_grid: [5, 7, 10]
  step_size_grid: [0.05, 0.1]
```

## Understanding the Output

The pipeline generates `results/metrics.json` with:

```json
{
  "best_params": {
    "maxIter": 100,
    "maxDepth": 7,
    "stepSize": 0.1
  },
  "models": {
    "GBT": {
      "train": {"mae": 1.67, "rmse": 2.19, "r2": 0.94},
      "test": {"mae": 1.74, "rmse": 2.24, "r2": 0.93}
    },
    "Persistence_min": {
      "train": {...},
      "test": {...}
    },
    ...
  },
  "feature_importance": [
    {"feature": "prev_day_avg_temp", "importance": 0.872},
    ...
  ]
}
```

## Exploring Results

Use the exploration notebook:

```bash
jupyter notebook notebooks/exploration.ipynb
```

This notebook:
- Loads metrics.json
- Visualizes model comparison
- Plots feature importance
- Provides example code for data exploration

## Running Tests

Tests are written with pytest and cover:
- Feature engineering (no same-day leakage)
- Temporal split validation
- Data cleaning logic

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ -v --cov=src/weather_pipeline --cov-report=term-missing
```

**Note**: Tests require Java 11 or 17 for PySpark. If you have Java 21+, you may encounter compatibility issues.

## Troubleshooting

### Java Version Issues

If you see `java.lang.UnsupportedOperationException: getSubject is not supported`:

- PySpark requires Java 11 or 17
- Check your Java version: `java -version`
- Install Java 11: https://adoptium.net/
- Set JAVA_HOME if needed

### Memory Issues

If Spark runs out of memory:

```bash
export PYSPARK_SUBMIT_ARGS="--driver-memory 4g --executor-memory 4g pyspark-shell"
python -m weather_pipeline.cli --config configs/default.yaml
```

### Data Path Issues

If the pipeline can't find your data:

1. Check that paths in `configs/default.yaml` are absolute or relative to project root
2. Verify files exist: `ls -lh data/`

## Development Workflow

1. Make changes to source code
2. Run linter: `make lint`
3. Run tests: `make test`
4. Format code: `make format`
5. Run pipeline: `make run`
6. Commit changes

## CI/CD

GitHub Actions runs on every push:
- Linting with ruff
- Type checking with mypy
- Tests with pytest

See `.github/workflows/ci.yml` for configuration.

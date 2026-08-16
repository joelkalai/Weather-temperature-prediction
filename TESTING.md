# Testing Strategy

## Test Coverage

The test suite covers critical pipeline logic to prevent data leakage and ensure correctness.

### Feature Engineering Tests (`tests/test_features.py`)

**Critical**: These tests verify that NO same-day data leaks into predictions.

1. **test_temporal_features**
   - Verifies year, month, day are correctly extracted from date column

2. **test_lagged_features_no_same_day_leakage**
   - For row N, `prev_day_min_temp` must equal row N-1's `min_temp_c`
   - Asserts NO row has `prev_day_min_temp == min_temp_c` (same-day leakage)

3. **test_rolling_features_exclude_current_row**
   - Rolling windows use `rowsBetween(-N, -1)` to exclude current row
   - For an increasing series, verifies `rolling_avg < current_value` always

4. **test_trend_features**
   - Trend = yesterday's min - day-before-yesterday's min
   - Verifies correct window lag computation

5. **test_feature_pipeline_produces_only_historical_features**
   - **Integration test**: Runs full feature pipeline
   - Asserts EVERY feature value ≠ current target value
   - In an increasing series, rolling averages lag behind current value

### Temporal Split Tests (`tests/test_split.py`)

**Critical**: These tests ensure train/test split prevents temporal leakage.

1. **test_temporal_split_no_leakage**
   - All train dates < split_year
   - All test dates >= split_year
   - max(train dates) < min(test dates)

2. **test_temporal_split_boundary**
   - Tests exact boundary behavior (last second of split_year-1 vs first second of split_year)

3. **test_temporal_split_validates_no_overlap**
   - Validates built-in overlap detection raises errors if misconfigured

### Data Cleaning Tests (`tests/test_clean.py`)

1. **test_drop_null_target**
   - Verifies rows with null min_temp_c are dropped

2. **test_validate_no_nulls**
   - Verifies null validation logging works correctly

## Why These Tests Matter

### Preventing Data Leakage

The #1 mistake in time-series ML is using future data to predict the past. Our tests catch:

- **Same-day features**: Using today's average to predict today's minimum
- **Future-looking windows**: Including current row in rolling averages
- **Temporal test contamination**: Training on data after the test set

### Correctness Guarantees

These tests act as **executable documentation** of pipeline assumptions:

```python
# If this test passes, you have proof that:
assert row.prev_day_min_temp != row.min_temp_c  # No same-day leakage
assert row.rolling_7d_min_temp < row.min_temp_c  # Window excludes current
assert max(train_dates) < min(test_dates)        # No temporal leakage
```

## Running Tests Locally

### Requirements

- Python 3.10+
- **Java 11 or 17** (PySpark dependency)
- Check: `java -version`

### Run All Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
PYTHONPATH=src pytest tests/ -v

# With coverage
PYTHONPATH=src pytest tests/ -v --cov=src/weather_pipeline --cov-report=term-missing
```

### Expected Output (if Java is compatible)

```
tests/test_features.py::test_temporal_features PASSED
tests/test_features.py::test_lagged_features_no_same_day_leakage PASSED
tests/test_features.py::test_rolling_features_exclude_current_row PASSED
tests/test_features.py::test_trend_features PASSED
tests/test_features.py::test_feature_pipeline_produces_only_historical_features PASSED
tests/test_split.py::test_temporal_split_no_leakage PASSED
tests/test_split.py::test_temporal_split_boundary PASSED
tests/test_split.py::test_temporal_split_validates_no_overlap PASSED
tests/test_clean.py::test_drop_null_target PASSED
tests/test_clean.py::test_validate_no_nulls PASSED

========== 10 passed in 3.45s ==========
```

## Known Issues

### Java Version Incompatibility

**Error**: `java.lang.UnsupportedOperationException: getSubject is not supported`

**Cause**: PySpark 3.5.0 is incompatible with Java 21+

**Solution**:
1. Install Java 11 or 17 from [Adoptium](https://adoptium.net/)
2. Set `JAVA_HOME` to Java 11/17 installation
3. Re-run tests

**Why this happens**:
- macOS ships with Java 21 by default
- PySpark uses deprecated Java security APIs removed in Java 21
- This is a known upstream issue, not a problem with our code

### Workarounds

If you can't install Java 11/17, you can still verify logic by:

1. **Code Review**: Read the test files to understand what they validate
2. **Manual Testing**: Run the pipeline on sample data and inspect output
3. **CI**: GitHub Actions uses Java 17 and will run tests on push

## Test Philosophy

**Test the hard parts, trust the libraries.**

We test:
- ✅ Window function logic (custom, error-prone)
- ✅ Temporal split validation (custom, critical)
- ✅ Feature leakage prevention (custom, subtle bugs)

We don't test:
- ❌ Spark built-ins (lag, avg, etc.) — trust the library
- ❌ Model training — covered by cross-validation metrics
- ❌ File I/O — trust Spark DataFrameReader

## Adding New Tests

When adding features, add tests that verify:

1. **Causality**: New features only use strictly prior data
2. **Nulls**: New features handle missing data correctly
3. **Edge cases**: First/last rows, single-city data, etc.

Example:

```python
def test_new_feature_no_leakage(spark):
    """Verify new feature doesn't leak same-day data."""
    data = [...]  # Create test data
    df = spark.createDataFrame(data)

    result = add_new_feature(df)

    for row in result.collect():
        # New feature must not equal target for same row
        assert row.new_feature != row.min_temp_c
```

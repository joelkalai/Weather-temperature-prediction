# Weather Temperature Prediction with PySpark

A scalable machine learning pipeline for predicting daily minimum temperatures across capital cities using Apache Spark MLlib with hyperparameter tuning.

## Overview

This project builds an end-to-end weather prediction system that:
- Processes **500K+ weather records** from 10 capital cities worldwide
- Engineers **time-series features** including lagged temperatures and rolling averages
- Performs **hyperparameter tuning** with 3-fold cross-validation across 12 parameter combinations
- Trains a **Gradient Boosted Tree regression model** using PySpark MLlib
- Achieves prediction of next-day minimum temperatures using only historically available data

## Key Features

- **Scalable Data Processing**: Built on PySpark to handle large-scale weather datasets
- **Time-Series Feature Engineering**:
  - Previous day temperatures (min, avg, max)
  - 7-day and 30-day rolling averages
  - Temperature trend indicators
- **Hyperparameter Tuning**: Cross-validation with grid search over:
  - `maxIter`: [50, 100]
  - `maxDepth`: [5, 7, 10]
  - `stepSize`: [0.05, 0.1]
- **Feature Importance Analysis**: Identifies which features contribute most to predictions
- **Production-Ready Pipeline**: End-to-end ML pipeline with preprocessing, feature encoding, and model training
- **Proper Train/Test Split**: Temporal split (pre-2010 vs 2010+) to prevent data leakage

## Dataset

Weather data from 10 capital cities with the most historical records:
- Brussels, Vienna, Stockholm, Zagreb, Dublin
- Kiev, Tashkent, Vilnius, Vaduz, Tbilisi

**Features used for prediction:**
| Feature | Description |
|---------|-------------|
| `prev_day_min_temp` | Previous day's minimum temperature |
| `prev_day_avg_temp` | Previous day's average temperature |
| `prev_day_max_temp` | Previous day's maximum temperature |
| `rolling_7d_min_temp` | 7-day rolling average of minimum temperature |
| `rolling_30d_min_temp` | 30-day rolling average of minimum temperature |
| `temp_trend` | Temperature change direction |
| `latitude`, `longitude` | Geographic coordinates |
| `year`, `month`, `day` | Temporal features |
| `season`, `city_name`, `region`, `continent` | Categorical features |

## Tech Stack

- **Apache Spark / PySpark** - Distributed data processing
- **Spark MLlib** - Machine learning pipeline & cross-validation
- **Gradient Boosted Trees** - Regression model
- **Python** - Programming language

## Project Structure

```
weather-temperature-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── weather_prediction_pipeline.ipynb  # Main analysis notebook
├── src/                                    # Modular source code
├── data/
│   └── sample/                             # Sample data for testing
└── results/                                # Model outputs and visualizations
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. Open the notebook in Jupyter or Google Colab:
   ```bash
   jupyter notebook notebooks/weather_prediction_pipeline.ipynb
   ```

2. Update the data paths to point to your local data files

3. Run all cells to:
   - Load and preprocess weather data
   - Engineer time-series features
   - Run cross-validation to find optimal hyperparameters
   - Train the GBT regression model with best parameters
   - Evaluate on held-out test data
   - Analyze feature importance

## Results

| Metric | Training | Testing |
|--------|----------|---------|
| MAE    | 1.67°C   | 1.74°C  |
| RMSE   | 2.19°C   | 2.24°C  |
| R²     | 0.940    | 0.927   |

**Best Hyperparameters Found:**
- `maxIter`: 100
- `maxDepth`: 7
- `stepSize`: 0.1

**Top Features by Importance:**
| Feature | Importance |
|---------|------------|
| `prev_day_avg_temp` | 87.2% |
| `prev_day_min_temp` | 1.8% |
| `year` | 1.6% |
| `prev_day_max_temp` | 1.4% |
| `rolling_7d_min_temp` | 1.3% |

## Methodology

### Data Preparation
1. Filter to capital cities by joining weather stations with country metadata
2. Select top 10 cities by data availability
3. Handle missing values

### Feature Engineering
1. Extract temporal features (year, month, day)
2. Create lagged features using window functions
3. Calculate rolling averages (7-day, 30-day windows)
4. Compute temperature trend indicators
5. **Avoid data leakage**: Only use historically available data (no same-day features)

### Model Pipeline
1. **StringIndexer** → Index categorical columns
2. **OneHotEncoder** → Encode indexed categories
3. **VectorAssembler** → Combine numerical features
4. **StandardScaler** → Normalize numerical features
5. **GBTRegressor** → Train gradient boosted tree model

### Hyperparameter Tuning
- **Method**: 3-fold Cross-Validation with Grid Search
- **Parameter Grid**: 12 combinations (2 × 3 × 2)
- **Evaluation Metric**: Mean Absolute Error (MAE)
- **Total Model Fits**: 36 (12 combinations × 3 folds)

### Evaluation
- Temporal train/test split (data before 2010 for training, 2010+ for testing)
- Metrics: MAE, RMSE, R²

## Future Improvements

- [x] ~~Add hyperparameter tuning with CrossValidator~~
- [x] ~~Add feature importance analysis~~
- [ ] Compare multiple models (Linear Regression, Random Forest)
- [ ] Include visualization of predictions vs actuals
- [ ] Deploy as a REST API for real-time predictions

## License

MIT License

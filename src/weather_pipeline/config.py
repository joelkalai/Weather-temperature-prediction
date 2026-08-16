"""Configuration for the weather prediction pipeline."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data paths and ingestion parameters."""

    raw_weather_path: str
    cities_path: str
    countries_path: str
    output_path: str = "results"
    top_n_cities: int = 10


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    lag_days: int = 1
    rolling_window_7d: int = 7
    rolling_window_30d: int = 30
    trend_lag: int = 2


@dataclass
class SplitConfig:
    """Train/test split parameters."""

    split_year: int = 2010  # Data before this year is training, >= is test


@dataclass
class ModelConfig:
    """Model hyperparameters and tuning grid."""

    # Cross-validation
    num_folds: int = 3
    parallelism: int = 2

    # Hyperparameter grid for GBT
    max_iter_grid: List[int] = field(default_factory=lambda: [50, 100])
    max_depth_grid: List[int] = field(default_factory=lambda: [5, 7, 10])
    step_size_grid: List[float] = field(default_factory=lambda: [0.05, 0.1])

    # Evaluation metric for cross-validation
    cv_metric: str = "mae"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            features=FeatureConfig(**config_dict.get("features", {})),
            split=SplitConfig(**config_dict.get("split", {})),
            model=ModelConfig(**config_dict.get("model", {})),
        )

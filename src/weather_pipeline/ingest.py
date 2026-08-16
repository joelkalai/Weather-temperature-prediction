"""Data ingestion: load raw weather, cities, countries data and filter to top N capitals."""

import logging
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import DataConfig

logger = logging.getLogger(__name__)


def load_raw_data(
    spark: SparkSession, config: DataConfig
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Load raw weather, cities, and countries data.

    Args:
        spark: SparkSession instance
        config: Data configuration with file paths

    Returns:
        Tuple of (daily_weather, cities, countries) DataFrames
    """
    logger.info("Loading raw data files")

    daily_weather = spark.read.parquet(config.raw_weather_path)
    logger.info(f"Loaded daily_weather: {daily_weather.count()} rows")

    cities = spark.read.csv(config.cities_path, header=True, inferSchema=True)
    logger.info(f"Loaded cities: {cities.count()} rows")

    countries = spark.read.csv(config.countries_path, header=True, inferSchema=True)
    logger.info(f"Loaded countries: {countries.count()} rows")

    return daily_weather, cities, countries


def identify_capital_stations(
    cities: DataFrame, countries: DataFrame
) -> DataFrame:
    """Identify weather stations in capital cities.

    Joins cities with countries where city_name matches capital AND iso3 matches.

    Args:
        cities: Cities DataFrame with station_id, city_name, iso3
        countries: Countries DataFrame with capital, iso3

    Returns:
        DataFrame with (station_id, city_name) for capital cities
    """
    logger.info("Identifying capital city stations")

    capital_stations = (
        cities.join(
            countries,
            (cities.iso3 == countries.iso3) & (cities.city_name == countries.capital),
            "inner",
        )
        .select(cities.station_id, cities.city_name)
        .distinct()
    )

    count = capital_stations.count()
    logger.info(f"Found {count} capital city stations")

    return capital_stations


def filter_to_top_n_capitals(
    daily_weather: DataFrame,
    capital_stations: DataFrame,
    cities: DataFrame,
    countries: DataFrame,
    top_n: int,
) -> DataFrame:
    """Filter weather data to top N capital cities by record count.

    Args:
        daily_weather: Raw weather DataFrame
        capital_stations: Capital stations DataFrame from identify_capital_stations
        cities: Cities DataFrame for metadata
        countries: Countries DataFrame for metadata
        top_n: Number of top cities to keep

    Returns:
        DataFrame with weather data for top N capitals, enriched with geographic metadata
    """
    logger.info(f"Filtering to top {top_n} capital cities by record count")

    # Filter daily_weather to capital cities only
    capital_weather = daily_weather.join(
        capital_stations, ["station_id", "city_name"], "inner"
    )

    # Find top N by record count
    top_n_cities = (
        capital_weather.groupBy("station_id", "city_name")
        .count()
        .orderBy(F.col("count").desc())
        .limit(top_n)
    )

    logger.info(f"Top {top_n} cities:")
    top_n_cities.show(truncate=False)

    # Filter to top N
    top_n_data = capital_weather.join(
        top_n_cities.select("station_id", "city_name"), ["station_id", "city_name"], "inner"
    )

    # Join with cities to get latitude, longitude, iso3
    cities_info = cities.select(
        "station_id", "city_name", "latitude", "longitude", "iso3"
    )
    top_n_data = top_n_data.join(cities_info, ["station_id", "city_name"], "inner")

    # Join with countries to get region, continent
    countries_info = countries.select("iso3", "region", "continent")
    top_n_data = top_n_data.join(countries_info, "iso3", "inner")

    count = top_n_data.count()
    logger.info(f"Top {top_n} cities data: {count} rows")

    return top_n_data


def ingest_data(spark: SparkSession, config: DataConfig) -> DataFrame:
    """Run complete data ingestion pipeline.

    Args:
        spark: SparkSession instance
        config: Data configuration

    Returns:
        DataFrame with weather data for top N capital cities with geographic metadata
    """
    daily_weather, cities, countries = load_raw_data(spark, config)
    capital_stations = identify_capital_stations(cities, countries)
    top_n_data = filter_to_top_n_capitals(
        daily_weather, capital_stations, cities, countries, config.top_n_cities
    )

    return top_n_data

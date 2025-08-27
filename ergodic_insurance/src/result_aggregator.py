"""Advanced result aggregation framework for Monte Carlo simulations.

This module provides comprehensive aggregation capabilities for simulation results,
supporting hierarchical aggregation, time-series analysis, and memory-efficient
processing of large datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""

    percentiles: List[float] = field(default_factory=lambda: [1, 5, 10, 25, 50, 75, 90, 95, 99])
    calculate_moments: bool = True
    calculate_distribution_fit: bool = False
    chunk_size: int = 10_000
    cache_results: bool = True
    precision: int = 6


class BaseAggregator(ABC):
    """Abstract base class for result aggregation.

    Provides common functionality for all aggregation types.
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        """Initialize aggregator with configuration.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def aggregate(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform aggregation on data.

        Args:
            data: Input data array

        Returns:
            Dictionary of aggregated statistics
        """
        raise NotImplementedError("Subclasses must implement aggregate method")

    def _get_cache_key(self, data_hash: str, operation: str) -> str:
        """Generate cache key for operation."""
        return f"{data_hash}_{operation}"

    def _cache_result(self, key: str, value: Any) -> Any:
        """Cache and return a result."""
        if self.config.cache_results:
            self._cache[key] = value
        return value

    def _round_value(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Round value to configured precision."""
        if isinstance(value, np.ndarray):
            return np.round(value, self.config.precision)
        return round(value, self.config.precision)


class ResultAggregator(BaseAggregator):
    """Main aggregator for simulation results.

    Provides comprehensive aggregation of Monte Carlo simulation results
    with support for custom aggregation functions.
    """

    def __init__(
        self,
        config: Optional[AggregationConfig] = None,
        custom_functions: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize result aggregator.

        Args:
            config: Aggregation configuration
            custom_functions: Dictionary of custom aggregation functions
        """
        super().__init__(config)
        self.custom_functions = custom_functions or {}

    def aggregate(self, data: np.ndarray) -> Dict[str, Any]:
        """Aggregate simulation results.

        Args:
            data: Array of simulation results

        Returns:
            Dictionary containing all aggregated statistics
        """
        results: Dict[str, Any] = {}

        # Basic statistics
        results["count"] = len(data)
        results["mean"] = self._round_value(np.mean(data))
        results["std"] = self._round_value(np.std(data))
        results["min"] = self._round_value(np.min(data))
        results["max"] = self._round_value(np.max(data))

        # Percentiles
        if self.config.percentiles:
            percentile_values = np.percentile(data, self.config.percentiles)
            results["percentiles"] = {
                f"p{int(p)}": self._round_value(val)
                for p, val in zip(self.config.percentiles, percentile_values)
            }

        # Statistical moments
        if self.config.calculate_moments:
            results["moments"] = self._calculate_moments(data)

        # Distribution fitting
        if self.config.calculate_distribution_fit:
            results["distribution_fit"] = self._fit_distributions(data)

        # Custom aggregations
        for name, func in self.custom_functions.items():
            try:
                results[f"custom_{name}"] = self._round_value(func(data))
            except (ValueError, TypeError, AttributeError) as e:
                results[f"custom_{name}_error"] = str(e)

        return results

    def _calculate_moments(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate statistical moments.

        Args:
            data: Input data array

        Returns:
            Dictionary of statistical moments
        """
        # Suppress precision warnings for nearly identical data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Precision loss occurred")
            skewness = stats.skew(data, nan_policy="omit")
            kurtosis = stats.kurtosis(data, nan_policy="omit")

        return {
            "variance": self._round_value(np.var(data)),
            "skewness": self._round_value(skewness),
            "kurtosis": self._round_value(kurtosis),
            "coefficient_variation": self._round_value(
                np.std(data) / np.mean(data) if np.mean(data) != 0 else np.nan
            ),
        }

    def _fit_distributions(self, data: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Fit distributions to data.

        Args:
            data: Input data array

        Returns:
            Dictionary of fitted distribution parameters
        """
        distributions = {}

        # Fit normal distribution
        try:
            mu, sigma = stats.norm.fit(data)
            distributions["normal"] = {
                "mu": self._round_value(mu),
                "sigma": self._round_value(sigma),
                "ks_statistic": self._round_value(stats.kstest(data, "norm", (mu, sigma))[0]),
            }
        except (ValueError, TypeError):
            pass

        # Fit lognormal distribution
        try:
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            distributions["lognormal"] = {
                "shape": self._round_value(shape),
                "scale": self._round_value(scale),
                "ks_statistic": self._round_value(
                    stats.kstest(data, "lognorm", (shape, loc, scale))[0]
                ),
            }
        except (ValueError, TypeError):
            pass

        return distributions


class TimeSeriesAggregator(BaseAggregator):
    """Aggregator for time-series data.

    Supports annual, cumulative, and rolling window aggregations.
    """

    def __init__(self, config: Optional[AggregationConfig] = None, window_size: int = 12):
        """Initialize time-series aggregator.

        Args:
            config: Aggregation configuration
            window_size: Size of rolling window for aggregation
        """
        super().__init__(config)
        self.window_size = window_size

    def aggregate(self, data: np.ndarray) -> Dict[str, Any]:
        """Aggregate time-series data.

        Args:
            data: 2D array where rows are time periods and columns are simulations

        Returns:
            Dictionary of time-series aggregations
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_periods, n_simulations = data.shape
        results: Dict[str, Any] = {}

        # Period-wise statistics
        results["period_mean"] = self._round_value(np.mean(data, axis=1))
        results["period_std"] = self._round_value(np.std(data, axis=1))
        results["period_min"] = self._round_value(np.min(data, axis=1))
        results["period_max"] = self._round_value(np.max(data, axis=1))

        # Cumulative statistics
        cumulative_data = np.cumsum(data, axis=0)
        results["cumulative_mean"] = self._round_value(np.mean(cumulative_data, axis=1))
        results["cumulative_std"] = self._round_value(np.std(cumulative_data, axis=1))

        # Rolling window statistics
        if n_periods >= self.window_size:
            results["rolling_stats"] = self._calculate_rolling_stats(data)

        # Growth rates
        if n_periods > 1:
            # Handle division by zero in growth rate calculation
            with np.errstate(divide="ignore", invalid="ignore"):
                growth_rates = (data[1:] / data[:-1] - 1) * 100
                growth_rates = np.where(np.isfinite(growth_rates), growth_rates, 0)
            results["growth_rate_mean"] = self._round_value(np.mean(growth_rates, axis=1))
            results["growth_rate_std"] = self._round_value(np.std(growth_rates, axis=1))

        # Autocorrelation
        results["autocorrelation"] = self._calculate_autocorrelation(data)

        return results

    def _calculate_rolling_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate rolling window statistics.

        Args:
            data: Time-series data

        Returns:
            Dictionary of rolling statistics
        """
        n_periods, n_simulations = data.shape
        n_windows = n_periods - self.window_size + 1

        rolling_mean = np.zeros((n_windows, n_simulations))
        rolling_std = np.zeros((n_windows, n_simulations))

        for i in range(n_windows):
            window_data = data[i : i + self.window_size]
            rolling_mean[i] = np.mean(window_data, axis=0)
            rolling_std[i] = np.std(window_data, axis=0)

        return {
            "mean": self._round_value(np.mean(rolling_mean, axis=1)),
            "std": self._round_value(np.mean(rolling_std, axis=1)),
            "volatility": self._round_value(np.std(rolling_mean, axis=1)),
        }

    def _calculate_autocorrelation(self, data: np.ndarray, max_lag: int = 5) -> Dict[str, Any]:
        """Calculate autocorrelation for different lags.

        Args:
            data: Time-series data
            max_lag: Maximum lag to calculate

        Returns:
            Dictionary of autocorrelations by lag
        """
        n_periods = data.shape[0]
        autocorr: Dict[str, Any] = {}

        for lag in range(1, min(max_lag + 1, n_periods)):
            if n_periods > lag:
                correlation = np.corrcoef(data[:-lag].flatten(), data[lag:].flatten())[0, 1]
                autocorr[f"lag_{lag}"] = self._round_value(correlation)

        return autocorr


class PercentileTracker:
    """Efficient percentile tracking for streaming data.

    Uses approximate algorithms for memory-efficient percentile calculation
    on large datasets.
    """

    def __init__(self, percentiles: List[float], max_samples: int = 100_000):
        """Initialize percentile tracker.

        Args:
            percentiles: List of percentiles to track
            max_samples: Maximum samples to keep in memory
        """
        self.percentiles = sorted(percentiles)
        self.max_samples = max_samples
        self.samples: List[float] = []
        self.total_count = 0
        self.reservoir_full = False

    def update(self, values: np.ndarray) -> None:
        """Update tracker with new values.

        Args:
            values: New values to add
        """
        for value in values:
            self.total_count += 1

            if len(self.samples) < self.max_samples:
                self.samples.append(value)
            else:
                # Reservoir sampling for memory efficiency
                if not self.reservoir_full:
                    self.samples = sorted(self.samples)
                    self.reservoir_full = True

                # Randomly replace samples
                idx = np.random.randint(self.total_count)
                if idx < self.max_samples:
                    self.samples[idx] = value

    def get_percentiles(self) -> Dict[str, float]:
        """Get current percentile estimates.

        Returns:
            Dictionary of percentile values
        """
        if not self.samples:
            return {}

        sorted_samples = np.sort(self.samples)
        result = {}

        for p in self.percentiles:
            idx = int(len(sorted_samples) * p / 100)
            idx = min(idx, len(sorted_samples) - 1)
            result[f"p{int(p)}"] = float(sorted_samples[idx])

        return result

    def reset(self) -> None:
        """Reset tracker state."""
        self.samples.clear()
        self.total_count = 0
        self.reservoir_full = False


class ResultExporter:
    """Export aggregated results to various formats."""

    @staticmethod
    def to_csv(results: Dict[str, Any], filepath: Path, index_label: str = "metric") -> None:
        """Export results to CSV file.

        Args:
            results: Aggregated results dictionary
            filepath: Output file path
            index_label: Label for index column
        """
        # Flatten nested dictionaries
        flat_results = ResultExporter._flatten_dict(results)

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(flat_results, orient="index", columns=["value"])
        df.index.name = index_label

        # Save to CSV
        df.to_csv(filepath)

    @staticmethod
    def to_json(results: Dict[str, Any], filepath: Path, indent: int = 2) -> None:
        """Export results to JSON file.

        Args:
            results: Aggregated results dictionary
            filepath: Output file path
            indent: JSON indentation level
        """
        # Convert numpy arrays to lists for JSON serialization
        json_results = ResultExporter._prepare_for_json(results)

        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=indent)

    @staticmethod
    def to_hdf5(results: Dict[str, Any], filepath: Path, compression: str = "gzip") -> None:
        """Export results to HDF5 file.

        Args:
            results: Aggregated results dictionary
            filepath: Output file path
            compression: Compression algorithm to use
        """
        with h5py.File(filepath, "w") as hf:
            ResultExporter._write_to_hdf5(hf, results, compression=compression)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for keys

        Returns:
            Flattened dictionary
        """
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ResultExporter._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def _prepare_for_json(obj: Any) -> Any:
        """Prepare object for JSON serialization.

        Args:
            obj: Object to prepare

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: ResultExporter._prepare_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ResultExporter._prepare_for_json(item) for item in obj]
        return obj

    @staticmethod
    def _write_to_hdf5(group: h5py.Group, data: Dict[str, Any], compression: str = "gzip") -> None:
        """Recursively write data to HDF5 group.

        Args:
            group: HDF5 group to write to
            data: Data dictionary to write
            compression: Compression algorithm
        """
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                ResultExporter._write_to_hdf5(subgroup, value, compression)
            elif isinstance(value, (np.ndarray, list)):
                group.create_dataset(key, data=np.asarray(value), compression=compression)
            else:
                group.attrs[key] = value


class HierarchicalAggregator:
    """Aggregator for hierarchical data structures.

    Supports multi-level aggregation across different dimensions
    (e.g., scenario -> year -> simulation).
    """

    def __init__(self, levels: List[str], config: Optional[AggregationConfig] = None):
        """Initialize hierarchical aggregator.

        Args:
            levels: List of aggregation levels in order
            config: Aggregation configuration
        """
        self.levels = levels
        self.config = config or AggregationConfig()
        self.aggregator = ResultAggregator(config)

    def aggregate_hierarchy(self, data: Dict[str, Any], level: int = 0) -> Dict[str, Any]:
        """Recursively aggregate hierarchical data.

        Args:
            data: Hierarchical data dictionary
            level: Current level in hierarchy

        Returns:
            Aggregated results at all levels
        """
        if level >= len(self.levels):
            # Leaf level - aggregate the actual data
            if isinstance(data, dict):
                # Data is a dict but we've reached the end of levels
                return data
            if isinstance(data, np.ndarray):  # type: ignore[unreachable]
                return self.aggregator.aggregate(data)
            # Default case for any other type
            return data

        current_level = self.levels[level]
        results = {"level": current_level, "items": {}}

        # Aggregate each item at this level
        for key, value in data.items():
            results["items"][key] = self.aggregate_hierarchy(value, level + 1)  # type: ignore[index]

        # Add summary across all items at this level
        if results["items"]:
            results["summary"] = self._summarize_level(results["items"])  # type: ignore[arg-type]

        return results

    def _summarize_level(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics across items at a level.

        Args:
            items: Dictionary of items to summarize

        Returns:
            Summary statistics
        """
        summary = {}

        # Collect all numeric values
        numeric_fields = set()
        for item in items.values():
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (int, float, np.number)):
                        numeric_fields.add(key)

        # Calculate summary for each numeric field
        for field in numeric_fields:
            values = []
            for item in items.values():
                if isinstance(item, dict) and field in item:
                    values.append(item[field])

            if values:
                summary[field] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        return summary

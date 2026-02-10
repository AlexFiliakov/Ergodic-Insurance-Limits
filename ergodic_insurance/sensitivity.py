"""Comprehensive sensitivity analysis tools for insurance optimization.

This module provides tools for analyzing how changes in key parameters affect
optimization results, including one-at-a-time (OAT) analysis, tornado diagrams,
and two-way sensitivity analysis with efficient caching.

Example:
    Basic sensitivity analysis for a single parameter::

        from ergodic_insurance.sensitivity import SensitivityAnalyzer
        from ergodic_insurance.business_optimizer import BusinessOptimizer
        from ergodic_insurance.manufacturer import WidgetManufacturer

        # Setup optimizer
        manufacturer = WidgetManufacturer(initial_assets=10_000_000)
        optimizer = BusinessOptimizer(manufacturer)

        # Run sensitivity analysis
        analyzer = SensitivityAnalyzer(base_config, optimizer)
        result = analyzer.analyze_parameter(
            "frequency",
            param_range=(3, 8),
            n_points=11
        )

        # Generate tornado diagram
        tornado_data = analyzer.create_tornado_diagram(
            parameters=["frequency", "severity_mean", "premium_rate"],
            metric="optimal_roe"
        )

.. versionchanged:: 0.7.0
    Replaced bare ``print()`` warning calls with ``logging.warning()``.
    See :issue:`382`.

Author: Alex Filiakov
Date: 2025-01-29
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .safe_pickle import safe_dump, safe_load

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis for a single parameter.

    Attributes:
        parameter: Name of the parameter being analyzed
        baseline_value: Original value of the parameter
        variations: Array of parameter values tested
        metrics: Dictionary of metric arrays for each variation
        parameter_path: Nested path to parameter (e.g., "manufacturer.base_operating_margin")
        units: Optional units for the parameter (e.g., "percentage", "dollars")
    """

    parameter: str
    baseline_value: float
    variations: np.ndarray
    metrics: Dict[str, np.ndarray]
    parameter_path: Optional[str] = None
    units: Optional[str] = None

    def calculate_impact(self, metric: str) -> float:
        """Calculate standardized impact on a specific metric.

        The impact is calculated as the elasticity of the metric with respect
        to the parameter, normalized by the baseline values.

        Args:
            metric: Name of the metric to calculate impact for

        Returns:
            Standardized impact coefficient (elasticity)

        Raises:
            KeyError: If metric not found in results
        """
        if metric not in self.metrics:
            raise KeyError(f"Metric '{metric}' not found in results")

        baseline_idx = len(self.variations) // 2
        baseline_metric = self.metrics[metric][baseline_idx]

        # Avoid division by zero
        if baseline_metric == 0:
            return 0.0

        # Calculate range of outcomes
        metric_range = self.metrics[metric].max() - self.metrics[metric].min()
        param_range = self.variations.max() - self.variations.min()

        # Avoid division by zero for parameter range
        if param_range == 0 or self.baseline_value == 0:
            return 0.0

        # Standardized sensitivity (elasticity)
        return float(
            (metric_range / abs(baseline_metric)) / (param_range / abs(self.baseline_value))
        )

    def get_metric_bounds(self, metric: str) -> Tuple[float, float]:
        """Get the minimum and maximum values for a metric.

        Args:
            metric: Name of the metric

        Returns:
            Tuple of (min_value, max_value)

        Raises:
            KeyError: If metric not found in results
        """
        if metric not in self.metrics:
            raise KeyError(f"Metric '{metric}' not found in results")
        return float(self.metrics[metric].min()), float(self.metrics[metric].max())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with variations and all metrics
        """
        data = {"parameter_value": self.variations}
        data.update(self.metrics)
        return pd.DataFrame(data)


@dataclass
class TwoWaySensitivityResult:
    """Results from two-way sensitivity analysis.

    Attributes:
        parameter1: Name of first parameter
        parameter2: Name of second parameter
        values1: Array of values for first parameter
        values2: Array of values for second parameter
        metric_grid: 2D array of metric values [len(values1), len(values2)]
        metric_name: Name of the metric analyzed
    """

    parameter1: str
    parameter2: str
    values1: np.ndarray
    values2: np.ndarray
    metric_grid: np.ndarray
    metric_name: str

    def find_optimal_region(self, target_value: float, tolerance: float = 0.05) -> np.ndarray:
        """Find parameter combinations that achieve target metric value.

        Args:
            target_value: Target value for the metric
            tolerance: Relative tolerance for matching (default 5%)

        Returns:
            Boolean mask array indicating satisfactory regions
        """
        lower_bound = target_value * (1 - tolerance)
        upper_bound = target_value * (1 + tolerance)
        return (self.metric_grid >= lower_bound) & (self.metric_grid <= upper_bound)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easier manipulation.

        Returns:
            DataFrame with multi-index for parameters and metric values
        """
        # Create meshgrid for parameter combinations
        p1_grid, p2_grid = np.meshgrid(self.values1, self.values2, indexing="ij")

        # Flatten arrays for DataFrame
        data = {
            self.parameter1: p1_grid.flatten(),
            self.parameter2: p2_grid.flatten(),
            self.metric_name: self.metric_grid.flatten(),
        }

        return pd.DataFrame(data)


class SensitivityAnalyzer:
    """Comprehensive sensitivity analysis tools for optimization.

    This class provides methods for analyzing how parameter changes affect
    optimization outcomes, with built-in caching for efficiency.

    Attributes:
        base_config: Base configuration dictionary
        optimizer: Optimizer object with an optimize() method
        results_cache: Cache for optimization results
        cache_dir: Directory for persistent cache storage
    """

    def __init__(
        self, base_config: Dict[str, Any], optimizer: Any, cache_dir: Optional[Path] = None
    ):
        """Initialize sensitivity analyzer.

        Args:
            base_config: Base configuration dictionary for optimization
            optimizer: Object with optimize(config) method returning results
            cache_dir: Optional directory for persistent caching
        """
        self.base_config = base_config.copy()
        self.optimizer = optimizer
        self.results_cache: Dict[str, Any] = {}
        self.cache_dir = cache_dir

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key for configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string for the configuration
        """
        # Sort keys for consistent hashing
        sorted_config = dict(sorted(config.items()))
        config_str = str(sorted_config)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached result if available.

        Args:
            cache_key: Cache key for the result

        Returns:
            Cached result or None if not found
        """
        # Check in-memory cache first
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        # Check persistent cache if configured
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        result = safe_load(f)
                        self.results_cache[cache_key] = result
                        return result
                except Exception:  # pylint: disable=broad-exception-caught
                    # If cache loading fails, continue without it
                    pass

        return None

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Store result in cache.

        Args:
            cache_key: Cache key for the result
            result: Result to cache
        """
        # Store in memory
        self.results_cache[cache_key] = result

        # Store persistently if configured
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    safe_dump(result, f)
            except Exception:  # pylint: disable=broad-exception-caught
                # If caching fails, continue without it
                pass

    def _update_nested_config(
        self, config: Dict[str, Any], param_path: str, value: Any
    ) -> Dict[str, Any]:
        """Update a nested parameter in configuration.

        Args:
            config: Configuration dictionary
            param_path: Dot-separated path to parameter
            value: New value for parameter

        Returns:
            Updated configuration dictionary
        """
        import copy

        config_copy = copy.deepcopy(config)
        parts = param_path.split(".")

        # Navigate to the nested location
        current = config_copy
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Convert to dict if needed
                current[part] = {"value": current[part]}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value
        return config_copy

    def analyze_parameter(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        param_name: str,
        param_range: Optional[Tuple[float, float]] = None,
        n_points: int = 11,
        param_path: Optional[str] = None,
        relative_range: float = 0.3,
    ) -> SensitivityResult:
        """Analyze sensitivity to a single parameter.

        Args:
            param_name: Name of parameter to analyze
            param_range: (min, max) range for parameter values
            n_points: Number of points to evaluate
            param_path: Nested path to parameter (e.g., "manufacturer.tax_rate")
            relative_range: If param_range not provided, use ±relative_range from baseline

        Returns:
            SensitivityResult with analysis results

        Raises:
            KeyError: If parameter not found in base configuration
        """
        # Determine parameter path
        if param_path is None:
            param_path = param_name

        # Get baseline value
        baseline: Any
        if "." in param_path:
            # Handle nested parameters
            parts = param_path.split(".")
            baseline = self.base_config
            for part in parts:
                if part not in baseline:
                    raise KeyError(f"Parameter '{param_path}' not found in configuration")
                baseline = baseline[part]
        else:
            if param_name not in self.base_config:
                raise KeyError(f"Parameter '{param_name}' not found in configuration")
            baseline = self.base_config[param_name]

        # Determine parameter range
        if param_range is None:
            # Ensure baseline is numeric
            try:
                baseline_float = float(baseline)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Parameter '{param_name}' has non-numeric baseline value: {baseline}"
                ) from exc
            min_val = baseline_float * (1 - relative_range)
            max_val = baseline_float * (1 + relative_range)
            param_range = (min_val, max_val)

        min_val, max_val = param_range
        variations = np.linspace(min_val, max_val, n_points)

        # Initialize metrics storage
        metrics: Dict[str, List[float]] = {
            "optimal_roe": [],
            "bankruptcy_risk": [],
            "optimal_retention": [],
            "total_premium": [],
            "growth_rate": [],
            "capital_efficiency": [],
        }

        # Run optimization for each variation
        for value in variations:
            # Update configuration
            if "." in param_path:
                config = self._update_nested_config(self.base_config, param_path, value)
            else:
                config = self.base_config.copy()
                config[param_name] = value

            # Get result (with caching)
            cache_key = self._get_cache_key(config)
            result = self._get_cached_result(cache_key)

            if result is None:
                # Run optimization
                result = self.optimizer.optimize(config)
                self._cache_result(cache_key, result)

            # Extract metrics
            # Handle different result structures
            if hasattr(result, "optimal_strategy"):
                strategy = result.optimal_strategy
                metrics["optimal_roe"].append(strategy.expected_roe)
                metrics["bankruptcy_risk"].append(strategy.bankruptcy_risk)
                metrics["growth_rate"].append(strategy.growth_rate)
                metrics["capital_efficiency"].append(strategy.capital_efficiency)

                # Handle retention/deductible
                if hasattr(strategy, "deductible"):
                    metrics["optimal_retention"].append(strategy.deductible)
                else:
                    metrics["optimal_retention"].append(0.0)

                # Handle premium
                if hasattr(strategy, "premium_rate"):
                    metrics["total_premium"].append(strategy.premium_rate)
                else:
                    metrics["total_premium"].append(0.0)
            else:
                # Fallback for simpler result structures
                metrics["optimal_roe"].append(getattr(result, "roe", 0.0))
                metrics["bankruptcy_risk"].append(getattr(result, "ruin_prob", 0.0))
                metrics["optimal_retention"].append(getattr(result, "retention", 0.0))
                metrics["total_premium"].append(getattr(result, "premium", 0.0))
                metrics["growth_rate"].append(getattr(result, "growth_rate", 0.0))
                metrics["capital_efficiency"].append(getattr(result, "capital_efficiency", 0.0))

        # Convert metrics to arrays
        metrics_arrays: Dict[str, np.ndarray] = {}
        for key, values in metrics.items():
            metrics_arrays[key] = np.array(values)

        return SensitivityResult(
            parameter=param_name,
            baseline_value=float(
                baseline
            ),  # baseline is guaranteed to be numeric from earlier check
            variations=variations,
            metrics=metrics_arrays,
            parameter_path=param_path,
        )

    def create_tornado_diagram(
        self,
        parameters: List[Union[str, Tuple[str, str]]],
        metric: str = "optimal_roe",
        relative_range: float = 0.3,
        n_points: int = 11,
    ) -> pd.DataFrame:
        """Create tornado diagram data for parameter impacts.

        Args:
            parameters: List of parameter names or (name, path) tuples
            metric: Metric to analyze
            relative_range: Relative range for parameter variations
            n_points: Number of points for analysis

        Returns:
            DataFrame sorted by impact magnitude with columns:
            - parameter: Parameter name
            - impact: Absolute impact value
            - direction: "positive" or "negative"
            - low_value: Metric value at parameter minimum
            - high_value: Metric value at parameter maximum
            - baseline: Metric value at baseline
            - baseline_param: Baseline parameter value
        """
        impacts = []

        for param in parameters:
            # Handle both string and tuple inputs
            if isinstance(param, tuple):
                param_name, param_path = param
            else:
                param_name = param
                param_path = param

            try:
                # Analyze sensitivity
                result = self.analyze_parameter(
                    param_name,
                    param_path=param_path,
                    relative_range=relative_range,
                    n_points=n_points,
                )

                # Calculate impact
                impact = result.calculate_impact(metric)

                # Get metric bounds
                low_val, high_val = result.get_metric_bounds(metric)
                baseline_idx = len(result.variations) // 2
                baseline_metric = result.metrics[metric][baseline_idx]

                # Store for tornado diagram
                impacts.append(
                    {
                        "parameter": param_name,
                        "impact": abs(impact),
                        "direction": "positive" if impact > 0 else "negative",
                        "low_value": low_val,
                        "high_value": high_val,
                        "baseline": baseline_metric,
                        "baseline_param": result.baseline_value,
                        "range_width": high_val - low_val,
                    }
                )

            except (KeyError, Exception) as e:  # pylint: disable=broad-exception-caught
                # Skip parameters that cause errors
                logger.warning("Could not analyze parameter '%s': %s", param_name, e)
                continue

        # Create DataFrame and sort by impact
        df = pd.DataFrame(impacts)
        if not df.empty:
            df = df.sort_values("impact", ascending=False)

        return df

    def analyze_two_way(  # pylint: disable=too-many-locals,too-many-branches
        self,
        param1: Union[str, Tuple[str, str]],
        param2: Union[str, Tuple[str, str]],
        param1_range: Optional[Tuple[float, float]] = None,
        param2_range: Optional[Tuple[float, float]] = None,
        n_points1: int = 10,
        n_points2: int = 10,
        metric: str = "optimal_roe",
        relative_range: float = 0.3,
    ) -> TwoWaySensitivityResult:
        """Perform two-way sensitivity analysis.

        Args:
            param1: First parameter name or (name, path) tuple
            param2: Second parameter name or (name, path) tuple
            param1_range: Range for first parameter
            param2_range: Range for second parameter
            n_points1: Number of points for first parameter
            n_points2: Number of points for second parameter
            metric: Metric to analyze
            relative_range: Relative range if explicit ranges not provided

        Returns:
            TwoWaySensitivityResult with grid of metric values
        """
        # Parse parameter specifications
        if isinstance(param1, tuple):
            param1_name, param1_path = param1
        else:
            param1_name = param1_path = param1

        if isinstance(param2, tuple):
            param2_name, param2_path = param2
        else:
            param2_name = param2_path = param2

        # Get baseline values
        baseline1 = self._get_param_value(param1_path)
        baseline2 = self._get_param_value(param2_path)

        # Determine ranges
        if param1_range is None:
            param1_range = (baseline1 * (1 - relative_range), baseline1 * (1 + relative_range))
        if param2_range is None:
            param2_range = (baseline2 * (1 - relative_range), baseline2 * (1 + relative_range))

        # Create parameter grids
        values1 = np.linspace(param1_range[0], param1_range[1], n_points1)
        values2 = np.linspace(param2_range[0], param2_range[1], n_points2)

        # Initialize result grid
        metric_grid = np.zeros((len(values1), len(values2)))

        # Run optimization for each combination
        for i, val1 in enumerate(values1):
            for j, val2 in enumerate(values2):
                # Update configuration
                config = self.base_config.copy()

                if "." in param1_path:
                    config = self._update_nested_config(config, param1_path, val1)
                else:
                    config[param1_name] = val1

                if "." in param2_path:
                    config = self._update_nested_config(config, param2_path, val2)
                else:
                    config[param2_name] = val2

                # Get result (with caching)
                cache_key = self._get_cache_key(config)
                result = self._get_cached_result(cache_key)

                if result is None:
                    result = self.optimizer.optimize(config)
                    self._cache_result(cache_key, result)

                # Extract metric value
                metric_value = self._extract_metric(result, metric)
                metric_grid[i, j] = metric_value

        return TwoWaySensitivityResult(
            parameter1=param1_name,
            parameter2=param2_name,
            values1=values1,
            values2=values2,
            metric_grid=metric_grid,
            metric_name=metric,
        )

    def _get_param_value(self, param_path: str) -> Any:
        """Get parameter value from configuration.

        Args:
            param_path: Dot-separated path to parameter

        Returns:
            Parameter value

        Raises:
            KeyError: If parameter not found
        """
        if "." in param_path:
            parts = param_path.split(".")
            value = self.base_config
            for part in parts:
                if part not in value:
                    raise KeyError(f"Parameter '{param_path}' not found")
                value = value[part]
            return value
        if param_path not in self.base_config:
            raise KeyError(f"Parameter '{param_path}' not found")
        return self.base_config[param_path]

    def _extract_metric(self, result: Any, metric: str) -> float:
        """Extract metric value from optimization result.

        Args:
            result: Optimization result object
            metric: Name of metric to extract

        Returns:
            Metric value
        """
        # Try different result structures
        if hasattr(result, "optimal_strategy"):
            strategy = result.optimal_strategy

            # Map metrics to strategy attributes
            strategy_metric_map = {
                "optimal_roe": lambda s: float(s.expected_roe),
                "bankruptcy_risk": lambda s: float(s.bankruptcy_risk),
                "growth_rate": lambda s: float(s.growth_rate),
                "capital_efficiency": lambda s: float(s.capital_efficiency),
                "optimal_retention": lambda s: getattr(s, "deductible", 0.0),
                "total_premium": lambda s: getattr(s, "premium_rate", 0.0),
            }

            if metric in strategy_metric_map:
                return strategy_metric_map[metric](strategy)

        # Fallback to direct attribute access
        metric_map = {
            "optimal_roe": "roe",
            "bankruptcy_risk": "ruin_prob",
            "optimal_retention": "retention",
            "total_premium": "premium",
            "growth_rate": "growth_rate",
            "capital_efficiency": "capital_efficiency",
        }

        attr_name = metric_map.get(metric, metric)
        return getattr(result, attr_name, 0.0)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.results_cache.clear()

        # Clear persistent cache if configured
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

    def analyze_parameter_group(
        self,
        parameter_group: Dict[str, Tuple[float, float]],
        n_points: int = 11,
        metric: str = "optimal_roe",
    ) -> Dict[str, SensitivityResult]:
        """Analyze sensitivity for a group of parameters.

        Args:
            parameter_group: Dictionary of parameter names to (min, max) ranges
            n_points: Number of points for each parameter
            metric: Primary metric for analysis

        Returns:
            Dictionary of parameter names to SensitivityResult objects
        """
        results = {}

        for param_name, param_range in parameter_group.items():
            try:
                result = self.analyze_parameter(
                    param_name, param_range=param_range, n_points=n_points
                )
                results[param_name] = result
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("Could not analyze '%s': %s", param_name, e)

        return results

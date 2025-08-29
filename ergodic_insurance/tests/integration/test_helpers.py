"""Helper utilities for integration testing.

This module provides utility functions for complex assertions,
data validation, and test scenario generation.
"""

from contextlib import contextmanager
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ergodic_insurance.src.simulation import SimulationResults

# ============================================================================
# Timing and Performance Utilities
# ============================================================================


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations.

    Args:
        name: Name of the operation being timed.

    Yields:
        Dict to store timing results.

    Example:
        >>> with timer("Simulation") as t:
        ...     run_simulation()
        >>> print(f"Elapsed: {t['elapsed']:.2f}s")
    """
    result: Dict[str, Union[float, str]] = {}
    start = time.time()
    try:
        yield result
    finally:
        elapsed = time.time() - start
        result["elapsed"] = elapsed
        result["name"] = name


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    n_runs: int = 3,
    warmup: int = 1,
) -> Dict[str, Any]:
    """Benchmark a function's performance.

    Args:
        func: Function to benchmark.
        args: Positional arguments.
        kwargs: Keyword arguments.
        n_runs: Number of benchmark runs.
        warmup: Number of warmup runs.

    Returns:
        Dict with timing statistics.
    """
    if kwargs is None:
        kwargs = {}

    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "min": min(times),
        "max": max(times),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "median": float(np.median(times)),
        "runs": n_runs,
        "last_result": result,
    }


# ============================================================================
# Data Validation Utilities
# ============================================================================


def validate_trajectory(  # pylint: disable=too-many-return-statements
    trajectory: np.ndarray,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_negative: bool = False,
    check_monotonic: Optional[str] = None,
) -> bool:
    """Validate a trajectory array.

    Args:
        trajectory: Array to validate.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        allow_negative: Whether negative values are allowed.
        check_monotonic: Check if 'increasing' or 'decreasing'.

    Returns:
        True if valid, False otherwise.
    """
    # Check for NaN or Inf
    if not np.all(np.isfinite(trajectory)):
        return False

    # Check bounds
    if min_value is not None and np.any(trajectory < min_value):
        return False
    if max_value is not None and np.any(trajectory > max_value):
        return False

    # Check negativity
    if not allow_negative and np.any(trajectory < 0):
        return False

    # Check monotonicity
    if check_monotonic == "increasing":
        if not np.all(np.diff(trajectory) >= -1e-10):  # Allow small numerical errors
            return False
    elif check_monotonic == "decreasing":
        if not np.all(np.diff(trajectory) <= 1e-10):
            return False

    return True


def validate_correlation(
    series1: np.ndarray,
    series2: np.ndarray,
    expected_corr: float,
    tolerance: float = 0.1,
) -> bool:
    """Validate correlation between two series.

    Args:
        series1: First data series.
        series2: Second data series.
        expected_corr: Expected correlation coefficient.
        tolerance: Acceptable deviation from expected.

    Returns:
        True if correlation is within tolerance.
    """
    if len(series1) != len(series2):
        return False

    actual_corr = np.corrcoef(series1, series2)[0, 1]
    return bool(abs(actual_corr - expected_corr) <= tolerance)


def validate_distribution(
    data: np.ndarray,
    distribution: str,
    params: Dict[str, float],
    significance: float = 0.05,
) -> Tuple[bool, float]:
    """Validate that data follows expected distribution.

    Args:
        data: Sample data.
        distribution: Distribution name ('normal', 'lognormal', 'exponential').
        params: Distribution parameters.
        significance: Significance level for test.

    Returns:
        Tuple of (passes_test, p_value).
    """
    from scipy import stats

    if distribution == "normal":
        statistic, p_value = stats.normaltest(data)
    elif distribution == "lognormal":
        log_data = np.log(data[data > 0])
        statistic, p_value = stats.normaltest(log_data)
    elif distribution == "exponential":
        # Use Kolmogorov-Smirnov test
        scale = params.get("scale", 1.0)
        statistic, p_value = stats.kstest(
            data,
            lambda x: stats.expon.cdf(x, scale=scale),
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return p_value > significance, p_value


# ============================================================================
# Scenario Comparison Utilities
# ============================================================================


def compare_scenarios(
    baseline: SimulationResults,
    alternative: SimulationResults,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare two simulation scenarios.

    Args:
        baseline: Baseline scenario results.
        alternative: Alternative scenario results.
        metrics: Metrics to compare (default: equity, assets, revenue).

    Returns:
        Dict with comparison statistics for each metric.
    """
    if metrics is None:
        metrics = ["equity", "assets", "revenue"]

    comparisons = {}

    for metric in metrics:
        if hasattr(baseline, metric) and hasattr(alternative, metric):
            base_values = getattr(baseline, metric)
            alt_values = getattr(alternative, metric)

            # Terminal value comparison
            base_terminal = base_values[-1] if len(base_values) > 0 else 0
            alt_terminal = alt_values[-1] if len(alt_values) > 0 else 0

            # Growth rate comparison (CAGR)
            n_years = len(base_values)
            base_initial = base_values[0] if len(base_values) > 0 else 1
            alt_initial = alt_values[0] if len(alt_values) > 0 else 1

            base_cagr = (base_terminal / base_initial) ** (1 / n_years) - 1 if n_years > 0 else 0
            alt_cagr = (alt_terminal / alt_initial) ** (1 / n_years) - 1 if n_years > 0 else 0

            comparisons[metric] = {
                "baseline_terminal": base_terminal,
                "alternative_terminal": alt_terminal,
                "terminal_difference": alt_terminal - base_terminal,
                "terminal_ratio": alt_terminal / base_terminal if base_terminal != 0 else np.inf,
                "baseline_cagr": base_cagr,
                "alternative_cagr": alt_cagr,
                "cagr_difference": alt_cagr - base_cagr,
                "baseline_mean": np.mean(base_values),
                "alternative_mean": np.mean(alt_values),
                "baseline_std": np.std(base_values),
                "alternative_std": np.std(alt_values),
                "volatility_ratio": np.std(alt_values) / np.std(base_values)
                if np.std(base_values) > 0
                else np.inf,
            }

    return comparisons


def calculate_ergodic_metrics(
    trajectories: np.ndarray,
    metric: str = "growth_rate",
) -> Dict[str, float]:
    """Calculate ergodic theory metrics from trajectories.

    Args:
        trajectories: Array of shape (n_paths, n_time).
        metric: Metric to calculate ('growth_rate', 'survival', 'volatility').

    Returns:
        Dict with ergodic metrics.
    """
    n_paths, n_time = trajectories.shape

    results = {}

    if metric == "growth_rate":
        # Time-average growth for each path
        time_averages = []
        for path in trajectories:
            if path[0] > 0 and path[-1] > 0:
                growth = np.log(path[-1] / path[0]) / (n_time - 1)
                time_averages.append(growth)

        # Ensemble average growth
        ensemble_growth = np.mean(
            [
                np.log(trajectories[:, t].mean() / trajectories[:, 0].mean())
                for t in range(1, n_time)
                if trajectories[:, t].mean() > 0 and trajectories[:, 0].mean() > 0
            ]
        )

        results["time_average_mean"] = np.mean(time_averages) if time_averages else -np.inf
        results["time_average_std"] = np.std(time_averages) if time_averages else 0
        results["ensemble_average"] = ensemble_growth
        results["ergodic_difference"] = results["time_average_mean"] - results["ensemble_average"]

    elif metric == "survival":
        # Survival rate (paths that don't go to zero)
        min_threshold = trajectories[:, 0].mean() * 0.01  # 1% of initial
        survived = np.sum(trajectories[:, -1] > min_threshold)
        results["survival_rate"] = survived / n_paths
        results["ruin_rate"] = 1 - results["survival_rate"]

    elif metric == "volatility":
        # Path volatility
        path_vols = []
        for path in trajectories:
            if len(path) > 1:
                returns = np.diff(np.log(path + 1e-10))
                path_vols.append(np.std(returns))

        results["mean_volatility"] = np.mean(path_vols)
        results["volatility_of_volatility"] = np.std(path_vols)

    return results


# ============================================================================
# Data Generation Utilities
# ============================================================================


def generate_correlated_series(
    n_points: int,
    correlation: float,
    mean1: float = 0,
    std1: float = 1,
    mean2: float = 0,
    std2: float = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two correlated time series.

    Args:
        n_points: Number of data points.
        correlation: Target correlation coefficient.
        mean1, std1: Mean and std of first series.
        mean2, std2: Mean and std of second series.
        seed: Random seed.

    Returns:
        Tuple of two correlated series.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate independent normal variables
    z1 = np.random.normal(0, 1, n_points)
    z2 = np.random.normal(0, 1, n_points)

    # Create correlation
    x1 = z1
    x2 = correlation * z1 + np.sqrt(1 - correlation**2) * z2

    # Scale and shift
    series1 = mean1 + std1 * x1
    series2 = mean2 + std2 * x2

    return series1, series2


def generate_regime_switching_series(
    n_points: int,
    regimes: List[Dict[str, float]],
    transition_probs: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate regime-switching time series.

    Args:
        n_points: Number of data points.
        regimes: List of regime parameters (mean, std).
        transition_probs: Transition probability matrix.
        seed: Random seed.

    Returns:
        Tuple of (series, regime_indicators).
    """
    if seed is not None:
        np.random.seed(seed)

    n_regimes = len(regimes)
    current_regime = 0

    series = []
    regime_history = []

    for _ in range(n_points):
        # Generate value for current regime
        regime_params = regimes[current_regime]
        value = np.random.normal(
            regime_params["mean"],
            regime_params["std"],
        )
        series.append(value)
        regime_history.append(current_regime)

        # Transition to next regime
        probs = transition_probs[current_regime]
        current_regime = np.random.choice(n_regimes, p=probs)

    return np.array(series), np.array(regime_history)


# ============================================================================
# Assertion Utilities
# ============================================================================


def assert_convergence(
    series: np.ndarray,
    target: float,
    tolerance: float = 0.01,
    window: int = 100,
) -> None:
    """Assert that a series converges to target value.

    Args:
        series: Time series to check.
        target: Target convergence value.
        tolerance: Acceptable deviation from target.
        window: Window size for convergence check.

    Raises:
        AssertionError: If series doesn't converge.
    """
    window = min(window, len(series))

    final_mean = np.mean(series[-window:])
    deviation = abs(final_mean - target)

    assert deviation <= tolerance, (
        f"Series did not converge to {target:.4f} "
        f"(final mean: {final_mean:.4f}, deviation: {deviation:.4f})"
    )


def assert_stationarity(
    series: np.ndarray,
    significance: float = 0.05,
) -> None:
    """Assert that a time series is stationary.

    Args:
        series: Time series to test.
        significance: Significance level for test.

    Raises:
        AssertionError: If series is not stationary.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series)
    p_value = result[1]

    assert p_value < significance, f"Series is not stationary (p-value: {p_value:.4f})"


def assert_data_consistency(
    data1: Any,
    data2: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that two datasets are consistent.

    Args:
        data1: First dataset.
        data2: Second dataset.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If data is inconsistent.
    """
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        np.testing.assert_allclose(data1, data2, rtol=rtol, atol=atol)
    elif isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
        pd.testing.assert_frame_equal(data1, data2, rtol=rtol, atol=atol)
    elif isinstance(data1, dict) and isinstance(data2, dict):
        assert set(data1.keys()) == set(data2.keys()), "Different keys"
        for key in data1:
            assert_data_consistency(data1[key], data2[key], rtol, atol)
    else:
        assert data1 == data2, f"Data mismatch: {data1} != {data2}"


# ============================================================================
# Integration Test Scenarios
# ============================================================================


def create_stress_scenario(
    base_config: Dict[str, Any],
    stress_factor: float = 2.0,
) -> Dict[str, Any]:
    """Create a stress test scenario configuration.

    Args:
        base_config: Base configuration dict.
        stress_factor: Multiplier for stress parameters.

    Returns:
        Stressed configuration.
    """
    stressed = base_config.copy()

    # Increase loss frequency and severity
    if "claim_frequency" in stressed:
        stressed["claim_frequency"] *= stress_factor
    if "claim_severity_mean" in stressed:
        stressed["claim_severity_mean"] *= stress_factor

    # Reduce margins
    if "operating_margin" in stressed:
        stressed["operating_margin"] /= stress_factor

    # Increase volatility
    if "revenue_volatility" in stressed:
        stressed["revenue_volatility"] *= stress_factor

    return stressed


def create_recovery_scenario(
    crisis_duration: int = 2,
    recovery_duration: int = 5,
    crisis_severity: float = 0.5,
) -> Dict[str, Any]:
    """Create a crisis and recovery scenario.

    Args:
        crisis_duration: Years of crisis.
        recovery_duration: Years to recover.
        crisis_severity: Reduction factor during crisis.

    Returns:
        Scenario configuration.
    """
    return {
        "phases": [
            {
                "name": "pre_crisis",
                "duration": 2,
                "revenue_multiplier": 1.0,
                "loss_multiplier": 1.0,
            },
            {
                "name": "crisis",
                "duration": crisis_duration,
                "revenue_multiplier": crisis_severity,
                "loss_multiplier": 2.0,
            },
            {
                "name": "recovery",
                "duration": recovery_duration,
                "revenue_multiplier": 0.8,
                "loss_multiplier": 1.2,
            },
            {
                "name": "post_recovery",
                "duration": 3,
                "revenue_multiplier": 1.1,
                "loss_multiplier": 0.9,
            },
        ]
    }

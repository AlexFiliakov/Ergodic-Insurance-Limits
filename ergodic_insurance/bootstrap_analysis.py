"""Bootstrap confidence interval analysis for simulation results.

This module provides comprehensive bootstrap analysis capabilities for
statistical significance testing and confidence interval calculation.
Supports both percentile and BCa (bias-corrected and accelerated) methods
with parallel processing for performance optimization.

Example:
    >>> import numpy as np
    >>> from bootstrap_analysis import BootstrapAnalyzer

    >>> # Create sample data
    >>> data = np.random.normal(100, 15, 1000)
    >>> analyzer = BootstrapAnalyzer(n_bootstrap=10000, seed=42)

    >>> # Calculate confidence interval for mean
    >>> ci = analyzer.confidence_interval(data, np.mean)
    >>> print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    >>> # Parallel bootstrap for faster computation
    >>> ci_parallel = analyzer.confidence_interval(
    ...     data, np.mean, method='bca', parallel=True
    ... )

Attributes:
    DEFAULT_N_BOOTSTRAP (int): Default number of bootstrap iterations (10000).
    DEFAULT_CONFIDENCE (float): Default confidence level (0.95).
    DEFAULT_N_WORKERS (int): Default number of parallel workers (4).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import stats
from tqdm import tqdm


def _bootstrap_worker(args: Tuple[np.ndarray, Any, int, int, Optional[int]]) -> List[float]:
    """Worker function for parallel bootstrap (module-level for pickling).

    Args:
        args: Tuple of (data, statistic, start_idx, n_samples, seed)

    Returns:
        List of bootstrap statistics.
    """
    data, statistic, start_idx, n_samples, seed = args

    # Handle common statistics as strings for better pickling support
    statistic_func: Callable[[np.ndarray], float]
    if isinstance(statistic, str):
        if statistic == "mean":
            statistic_func = np.mean
        elif statistic == "median":
            statistic_func = np.median
        elif statistic == "std":
            statistic_func = np.std
        elif statistic == "var":
            statistic_func = np.var
        else:
            raise ValueError(f"Unknown statistic string: {statistic}")
    else:
        statistic_func = statistic

    worker_rng = np.random.RandomState(seed)
    n = len(data)
    results = []

    for _ in range(n_samples):
        indices = worker_rng.choice(n, size=n, replace=True)
        bootstrap_sample = data[indices]
        results.append(statistic_func(bootstrap_sample))

    return results


@dataclass
class BootstrapResult:
    """Container for bootstrap analysis results."""

    statistic: float
    confidence_level: float
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    method: str
    n_bootstrap: int
    bias: Optional[float] = None
    acceleration: Optional[float] = None
    converged: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Generate human-readable summary of bootstrap results.

        Returns:
            Formatted string with key bootstrap statistics.
        """
        summary = [
            "Bootstrap Analysis Results",
            f"{'=' * 40}",
            f"Original Statistic: {self.statistic:.6f}",
            f"Method: {self.method}",
            f"Bootstrap Iterations: {self.n_bootstrap:,}",
            f"{self.confidence_level:.1%} Confidence Interval:",
            f"  [{self.confidence_interval[0]:.6f}, {self.confidence_interval[1]:.6f}]",
            f"CI Width: {self.confidence_interval[1] - self.confidence_interval[0]:.6f}",
            f"Bootstrap Mean: {np.mean(self.bootstrap_distribution):.6f}",
            f"Bootstrap Std: {np.std(self.bootstrap_distribution):.6f}",
        ]

        if self.bias is not None:
            summary.append(f"Bias: {self.bias:.6f}")
        if self.acceleration is not None:
            summary.append(f"Acceleration: {self.acceleration:.6f}")

        summary.append(f"Converged: {'Yes' if self.converged else 'No'}")

        return "\n".join(summary)


class BootstrapAnalyzer:
    """Main class for bootstrap confidence interval analysis.

    Provides methods for calculating bootstrap confidence intervals using
    various methods including percentile and BCa. Supports parallel processing
    for improved performance with large datasets.

    Args:
        n_bootstrap: Number of bootstrap iterations (default 10000).
        confidence_level: Confidence level for intervals (default 0.95).
        seed: Random seed for reproducibility (default None).
        n_workers: Number of parallel workers (default 4).
        show_progress: Whether to show progress bar (default True).

    Example:
        >>> analyzer = BootstrapAnalyzer(n_bootstrap=5000, confidence_level=0.99)
        >>> data = np.random.exponential(2, 1000)
        >>> result = analyzer.confidence_interval(data, np.median)
        >>> print(result.summary())
    """

    DEFAULT_N_BOOTSTRAP = 10000
    DEFAULT_CONFIDENCE = 0.95
    DEFAULT_N_WORKERS = 4

    def __init__(
        self,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        confidence_level: float = DEFAULT_CONFIDENCE,
        seed: Optional[int] = None,
        n_workers: int = DEFAULT_N_WORKERS,
        show_progress: bool = True,
    ):
        """Initialize bootstrap analyzer with specified parameters."""
        if not 0 < confidence_level < 1:
            raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")
        if n_bootstrap < 100:
            warnings.warn(
                f"Low n_bootstrap ({n_bootstrap}) may produce unstable results", UserWarning
            )

        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed
        self.n_workers = n_workers
        self.show_progress = show_progress
        self.rng = np.random.RandomState(seed)

    def bootstrap_sample(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate bootstrap samples and compute statistics.

        Args:
            data: Input data array.
            statistic: Function to compute on each bootstrap sample.
            n_samples: Number of bootstrap samples to generate.

        Returns:
            Array of bootstrap statistics.
        """
        n = len(data)
        bootstrap_stats = np.zeros(n_samples)

        for i in range(n_samples):
            indices = self.rng.choice(n, size=n, replace=True)
            bootstrap_data = data[indices]
            bootstrap_stats[i] = statistic(bootstrap_data)

        return bootstrap_stats

    def confidence_interval(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        confidence_level: Optional[float] = None,
        method: str = "percentile",
        parallel: bool = False,
    ) -> BootstrapResult:
        """Calculate bootstrap confidence interval for a statistic.

        Args:
            data: Input data array.
            statistic: Function to compute the statistic of interest.
            confidence_level: Confidence level (uses default if None).
            method: 'percentile' or 'bca' (bias-corrected and accelerated).
            parallel: Whether to use parallel processing.

        Returns:
            BootstrapResult containing confidence interval and diagnostics.

        Raises:
            ValueError: If method is not 'percentile' or 'bca'.
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if method not in ["percentile", "bca"]:
            raise ValueError(f"Method must be 'percentile' or 'bca', got {method}")

        # Compute original statistic
        original_stat = statistic(data)

        # Generate bootstrap distribution
        if parallel and self.n_workers > 1:
            try:
                bootstrap_dist = self._parallel_bootstrap(data, statistic)
            except (ValueError, RuntimeError, TypeError) as e:
                # Fall back to sequential if parallel fails
                if self.show_progress:
                    print(f"Parallel bootstrap failed: {e}. Falling back to sequential.")
                bootstrap_dist = self._sequential_bootstrap(data, statistic)
        else:
            bootstrap_dist = self._sequential_bootstrap(data, statistic)

        # Calculate confidence interval
        if method == "percentile":
            ci = self._percentile_interval(bootstrap_dist, confidence_level)
            bias = None
            acceleration = None
        else:  # method == 'bca'
            ci, bias, acceleration = self._bca_interval(
                data, statistic, bootstrap_dist, confidence_level
            )

        # Check convergence
        converged = self._check_convergence(bootstrap_dist)

        return BootstrapResult(
            statistic=original_stat,
            confidence_level=confidence_level,
            confidence_interval=ci,
            bootstrap_distribution=bootstrap_dist,
            method=method,
            n_bootstrap=self.n_bootstrap,
            bias=bias,
            acceleration=acceleration,
            converged=converged,
        )

    def _sequential_bootstrap(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Perform sequential bootstrap sampling.

        Args:
            data: Input data array.
            statistic: Function to compute on each sample.

        Returns:
            Array of bootstrap statistics.
        """
        n = len(data)
        bootstrap_dist = np.zeros(self.n_bootstrap)

        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Bootstrap sampling", unit="sample")

        for i in iterator:
            indices = self.rng.choice(n, size=n, replace=True)
            bootstrap_sample = data[indices]
            bootstrap_dist[i] = statistic(bootstrap_sample)

        return bootstrap_dist

    def _submit_bootstrap_jobs(
        self,
        executor: ProcessPoolExecutor,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        chunks: List[np.ndarray],
    ) -> List[Tuple[int, Any]]:
        """Submit bootstrap jobs to executor.

        Args:
            executor: Process pool executor
            data: Input data array
            statistic: Function to compute on each sample
            chunks: Array chunks for parallel processing

        Returns:
            List of (start_index, future) tuples
        """
        # Check if statistic is a common numpy function and convert to string
        statistic_to_pass: Union[str, Callable[[np.ndarray], float]] = statistic
        if statistic is np.mean:
            statistic_to_pass = "mean"
        elif statistic is np.median:
            statistic_to_pass = "median"
        elif statistic is np.std:
            statistic_to_pass = "std"
        elif statistic is np.var:
            statistic_to_pass = "var"

        futures = []
        for i, chunk in enumerate(chunks):
            if len(chunk) > 0:
                worker_seed = self.seed + i if self.seed else None
                args = (data, statistic_to_pass, chunk[0], len(chunk), worker_seed)
                future = executor.submit(_bootstrap_worker, args)
                futures.append((chunk[0], future))
        return futures

    def _parallel_bootstrap(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Perform parallel bootstrap sampling using multiprocessing.

        Args:
            data: Input data array.
            statistic: Function to compute on each sample.

        Returns:
            Array of bootstrap statistics.
        """
        chunks = np.array_split(np.arange(self.n_bootstrap), self.n_workers)
        bootstrap_dist = np.zeros(self.n_bootstrap)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = self._submit_bootstrap_jobs(executor, data, statistic, chunks)

            if self.show_progress:
                pbar = tqdm(total=self.n_bootstrap, desc="Parallel bootstrap", unit="sample")

            for start_idx, future in futures:
                results = future.result()
                bootstrap_dist[start_idx : start_idx + len(results)] = results

                if self.show_progress:
                    pbar.update(len(results))

            if self.show_progress:
                pbar.close()

        return bootstrap_dist

    def _percentile_interval(
        self,
        bootstrap_dist: np.ndarray,
        confidence_level: float,
    ) -> Tuple[float, float]:
        """Calculate percentile bootstrap confidence interval.

        Args:
            bootstrap_dist: Array of bootstrap statistics.
            confidence_level: Confidence level for the interval.

        Returns:
            Tuple of (lower, upper) confidence bounds.
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(bootstrap_dist, lower_percentile)
        upper = np.percentile(bootstrap_dist, upper_percentile)

        return (float(lower), float(upper))

    def _calculate_bias(self, bootstrap_dist: np.ndarray, original_stat: float) -> float:
        """Calculate bias correction for BCa interval.

        Args:
            bootstrap_dist: Array of bootstrap statistics.
            original_stat: Original statistic value.

        Returns:
            Bias correction value.
        """
        return float(stats.norm.ppf(np.mean(bootstrap_dist < original_stat)))

    def _calculate_acceleration(
        self, data: np.ndarray, statistic: Callable[[np.ndarray], float]
    ) -> float:
        """Calculate acceleration parameter using jackknife.

        Args:
            data: Original data array.
            statistic: Function to compute the statistic.

        Returns:
            Acceleration parameter.
        """
        n = len(data)
        jackknife_stats = np.zeros(n)

        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)

        jackknife_mean = np.mean(jackknife_stats)
        num = np.sum((jackknife_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)

        return float(num / den) if den != 0 else 0.0

    def _bca_interval(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        bootstrap_dist: np.ndarray,
        confidence_level: float,
    ) -> Tuple[Tuple[float, float], float, float]:
        """Calculate BCa (bias-corrected and accelerated) confidence interval.

        BCa intervals correct for both bias and skewness in the bootstrap
        distribution, providing more accurate coverage than percentile intervals.

        Args:
            data: Original data array.
            statistic: Function to compute the statistic.
            bootstrap_dist: Array of bootstrap statistics.
            confidence_level: Confidence level for the interval.

        Returns:
            Tuple of ((lower, upper), bias, acceleration).
        """
        original_stat = statistic(data)
        bias = self._calculate_bias(bootstrap_dist, original_stat)
        acceleration = self._calculate_acceleration(data, statistic)

        # Adjusted percentiles
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1alpha = stats.norm.ppf(1 - alpha / 2)

        a1 = stats.norm.cdf(bias + (bias + z_alpha) / (1 - acceleration * (bias + z_alpha)))
        a2 = stats.norm.cdf(bias + (bias + z_1alpha) / (1 - acceleration * (bias + z_1alpha)))

        # Calculate adjusted confidence interval
        lower = np.percentile(bootstrap_dist, a1 * 100)
        upper = np.percentile(bootstrap_dist, a2 * 100)

        return ((float(lower), float(upper)), float(bias), float(acceleration))

    def _check_convergence(
        self,
        bootstrap_dist: np.ndarray,
        stability_threshold: float = 0.01,
    ) -> bool:
        """Check if bootstrap distribution has converged.

        Convergence is assessed by checking the stability of percentiles
        across different portions of the bootstrap distribution.

        Args:
            bootstrap_dist: Array of bootstrap statistics.
            stability_threshold: Maximum allowed relative change for convergence.

        Returns:
            True if converged, False otherwise.
        """
        if len(bootstrap_dist) < 1000:
            return True  # Too few samples to assess convergence

        # Split distribution into halves and check stability
        mid = len(bootstrap_dist) // 2
        first_half = bootstrap_dist[:mid]
        second_half = bootstrap_dist[mid:]

        # Check key percentiles
        percentiles = [2.5, 25, 50, 75, 97.5]

        for p in percentiles:
            val1 = np.percentile(first_half, p)
            val2 = np.percentile(second_half, p)

            if val1 != 0:
                relative_change = abs((val2 - val1) / val1)
                if relative_change > stability_threshold:
                    return False

        return True

    def compare_statistics(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        comparison: str = "difference",
    ) -> BootstrapResult:
        """Compare statistics between two datasets using bootstrap.

        Args:
            data1: First dataset.
            data2: Second dataset.
            statistic: Function to compute on each dataset.
            comparison: Type of comparison ('difference' or 'ratio').

        Returns:
            BootstrapResult for the comparison statistic.

        Raises:
            ValueError: If comparison type is not supported.
        """
        if comparison not in ["difference", "ratio"]:
            raise ValueError(f"Comparison must be 'difference' or 'ratio', got {comparison}")

        def comparison_statistic(indices: np.ndarray) -> float:
            """Compute comparison statistic for bootstrap sample."""
            # Resample both datasets
            n1, n2 = len(data1), len(data2)
            idx1 = self.rng.choice(n1, size=n1, replace=True)
            idx2 = self.rng.choice(n2, size=n2, replace=True)

            stat1 = statistic(data1[idx1])
            stat2 = statistic(data2[idx2])

            if comparison == "difference":
                return stat1 - stat2
            # ratio
            if stat2 == 0:
                return np.nan
            return stat1 / stat2

        # Create combined index array for resampling
        combined_indices = np.arange(len(data1) + len(data2))

        # Generate bootstrap distribution of comparison statistic
        bootstrap_dist = np.zeros(self.n_bootstrap)

        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Bootstrap {comparison}", unit="sample")

        for i in iterator:
            bootstrap_dist[i] = comparison_statistic(combined_indices)

        # Remove NaN values if any (from ratio comparison)
        bootstrap_dist = bootstrap_dist[~np.isnan(bootstrap_dist)]

        # Calculate confidence interval
        ci = self._percentile_interval(bootstrap_dist, self.confidence_level)

        # Original comparison statistic
        orig_stat1 = statistic(data1)
        orig_stat2 = statistic(data2)

        if comparison == "difference":
            original_comparison = orig_stat1 - orig_stat2
        else:
            original_comparison = orig_stat1 / orig_stat2 if orig_stat2 != 0 else np.nan

        return BootstrapResult(
            statistic=original_comparison,
            confidence_level=self.confidence_level,
            confidence_interval=ci,
            bootstrap_distribution=bootstrap_dist,
            method="percentile",
            n_bootstrap=len(bootstrap_dist),
            converged=self._check_convergence(bootstrap_dist),
            metadata={
                "comparison_type": comparison,
                "stat1": orig_stat1,
                "stat2": orig_stat2,
            },
        )


def bootstrap_confidence_interval(
    data: Union[np.ndarray, List[float]],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    method: str = "percentile",
    seed: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Convenience function for simple bootstrap confidence interval calculation.

    Args:
        data: Input data (array or list).
        statistic: Function to compute statistic (default: mean).
        confidence_level: Confidence level (default: 0.95).
        n_bootstrap: Number of bootstrap iterations (default: 10000).
        method: 'percentile' or 'bca' (default: 'percentile').
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (original_statistic, (lower_bound, upper_bound)).

    Example:
        >>> data = np.random.normal(100, 15, 1000)
        >>> stat, ci = bootstrap_confidence_interval(data, np.median)
        >>> print(f"Median: {stat:.2f}, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    analyzer = BootstrapAnalyzer(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        show_progress=False,
    )

    result = analyzer.confidence_interval(data, statistic, method=method)

    return result.statistic, result.confidence_interval

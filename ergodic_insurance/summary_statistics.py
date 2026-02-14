"""Comprehensive summary statistics and report generation for simulation results.

This module provides statistical analysis tools, distribution fitting utilities,
and formatted report generation for Monte Carlo simulation results.
"""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import io
import math
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def format_quantile_key(q: float) -> str:
    """Format a quantile value as a dictionary key using per-mille resolution.

    Uses per-mille (parts per thousand) to avoid key collisions for
    sub-percentile quantiles that are critical for insurance risk metrics.

    Args:
        q: Quantile value in range [0, 1].

    Returns:
        Formatted key string, e.g. ``q0250`` for the 25th percentile,
        ``q0005`` for the 0.5th percentile, ``q0001`` for the 0.1th percentile.
    """
    return f"q{round(q * 1000):04d}"


@dataclass
class StatisticalSummary:
    """Complete statistical summary of simulation results."""

    basic_stats: Dict[str, float]
    distribution_params: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    hypothesis_tests: Dict[str, Dict[str, float]]
    extreme_values: Dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary to pandas DataFrame.

        Returns:
            DataFrame with all summary statistics
        """
        rows = []

        # Basic statistics
        for stat, value in self.basic_stats.items():
            rows.append({"category": "basic", "metric": stat, "value": value})

        # Distribution parameters
        for dist, params in self.distribution_params.items():
            for param, value in params.items():
                rows.append({"category": f"distribution_{dist}", "metric": param, "value": value})

        # Confidence intervals
        for metric, (lower, upper) in self.confidence_intervals.items():
            rows.append(
                {"category": "confidence_interval", "metric": f"{metric}_lower", "value": lower}
            )
            rows.append(
                {"category": "confidence_interval", "metric": f"{metric}_upper", "value": upper}
            )

        # Hypothesis tests
        for test, results in self.hypothesis_tests.items():
            for metric, value in results.items():
                rows.append({"category": f"test_{test}", "metric": metric, "value": value})

        # Extreme values
        for metric, value in self.extreme_values.items():
            rows.append({"category": "extreme", "metric": metric, "value": value})

        return pd.DataFrame(rows)


class SummaryStatistics:
    """Calculate comprehensive summary statistics for simulation results."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 1000,
        seed: Optional[int] = None,
        assume_iid: bool = True,
    ):
        """Initialize summary statistics calculator.

        Args:
            confidence_level: Confidence level for intervals
            bootstrap_iterations: Number of bootstrap iterations
            seed: Optional random seed for reproducibility
            assume_iid: If True (default), compute stderr as std/sqrt(n),
                assuming independent observations.  If False, estimate the
                effective sample size (ESS) via batch means and compute
                stderr as std(ddof=1)/sqrt(ESS).  Use False for
                autocorrelated data such as MCMC output or time-series
                simulations where observations are not independent.
        """
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self._rng = np.random.default_rng(seed)
        self.assume_iid = assume_iid

    def calculate_summary(
        self, data: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> StatisticalSummary:
        """Calculate complete statistical summary.

        Args:
            data: Input data array
            weights: Optional weights for weighted statistics

        Returns:
            Complete statistical summary
        """
        # Handle empty data
        if len(data) == 0:
            return StatisticalSummary(
                basic_stats=self._calculate_basic_stats(data, weights),
                distribution_params={},
                confidence_intervals={},
                hypothesis_tests={},
                extreme_values={},
            )

        # Basic statistics
        basic_stats = self._calculate_basic_stats(data, weights)

        # Fit distributions
        distribution_params = self._fit_distributions(data)

        # Bootstrap confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(data)

        # Hypothesis tests
        hypothesis_tests = self._perform_hypothesis_tests(data)

        # Extreme value statistics
        extreme_values = self._calculate_extreme_values(data)

        return StatisticalSummary(
            basic_stats=basic_stats,
            distribution_params=distribution_params,
            confidence_intervals=confidence_intervals,
            hypothesis_tests=hypothesis_tests,
            extreme_values=extreme_values,
        )

    def _safe_skew_kurtosis(self, data: np.ndarray, stat_type: str) -> float:
        """Calculate skewness or kurtosis with warning suppression."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Precision loss occurred")
            if stat_type == "skew":
                return float(stats.skew(data, nan_policy="omit"))
            return float(stats.kurtosis(data, nan_policy="omit"))

    @staticmethod
    def _estimate_ess_batch_means(data: np.ndarray) -> float:
        """Estimate effective sample size using the batch means method.

        Splits *data* into floor(sqrt(n)) non-overlapping batches, computes
        the variance of the batch means, and derives ESS as::

            ESS = n * Var(chain) / (batch_size * Var(batch_means))

        The result is clamped to [1, n].  For short arrays (n <= 10) the
        method returns n (i.e. assumes IID).

        Args:
            data: 1-D array of samples (must be non-empty).

        Returns:
            Estimated effective sample size.
        """
        n = len(data)
        if n <= 10:
            return float(n)

        n_batches = int(np.sqrt(n))
        if n_batches < 2:
            return float(n)

        batch_size = n // n_batches
        usable = n_batches * batch_size
        batches = data[:usable].reshape(n_batches, batch_size)
        batch_means = batches.mean(axis=1)

        var_bm = np.var(batch_means, ddof=1)
        var_chain = np.var(data[:usable], ddof=1)

        if var_bm <= 0 or var_chain <= 0:
            return float(n)

        ess = usable * var_chain / (batch_size * var_bm)
        return float(max(1.0, min(ess, n)))

    def _calculate_basic_stats(
        self, data: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate basic descriptive statistics.

        Args:
            data: Input data
            weights: Optional weights

        Returns:
            Dictionary of basic statistics
        """
        # Handle empty data
        if len(data) == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "iqr": 0.0,
                "cv": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "stderr": 0.0,
            }

        if weights is None:
            if self.assume_iid:
                stderr = float(np.std(data) / np.sqrt(len(data)))
            else:
                ess = self._estimate_ess_batch_means(data)
                stderr = float(np.std(data, ddof=1) / np.sqrt(ess))

            return {
                "count": len(data),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "variance": float(np.var(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "range": float(np.max(data) - np.min(data)),
                "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
                "cv": float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else np.inf,
                "skewness": float(self._safe_skew_kurtosis(data, "skew")),
                "kurtosis": float(self._safe_skew_kurtosis(data, "kurtosis")),
                "stderr": stderr,
            }

        mean = np.average(data, weights=weights)
        variance = np.average((data - mean) ** 2, weights=weights)
        std = np.sqrt(variance)

        return {
            "count": len(data),
            "mean": float(mean),
            "median": float(self._weighted_percentile(data, weights, 50)),
            "std": float(std),
            "variance": float(variance),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.max(data) - np.min(data)),
            "iqr": float(
                self._weighted_percentile(data, weights, 75)
                - self._weighted_percentile(data, weights, 25)
            ),
            "cv": float(std / mean) if mean != 0 else np.inf,
            "effective_sample_size": float(np.sum(weights) ** 2 / np.sum(weights**2)),
        }

    def _weighted_percentile(
        self, data: np.ndarray, weights: np.ndarray, percentile: float
    ) -> float:
        """Calculate weighted percentile.

        Args:
            data: Data values
            weights: Weights
            percentile: Percentile to calculate

        Returns:
            Weighted percentile value
        """
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        cutoff = percentile / 100.0 * cumsum[-1]

        return float(sorted_data[np.searchsorted(cumsum, cutoff)])

    def _fit_distributions(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Fit various distributions to data.

        Args:
            data: Input data

        Returns:
            Dictionary of fitted distribution parameters
        """
        results = {}

        # Normal distribution
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                mu, sigma = stats.norm.fit(data)
                ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.norm.cdf(x, mu, sigma))
            results["normal"] = {
                "mu": float(mu),
                "sigma": float(sigma),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "aic": float(self._calculate_aic(data, stats.norm, mu, sigma)),
            }
        except (ValueError, TypeError, RuntimeError):
            pass

        # Log-normal distribution
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                shape, loc, scale = stats.lognorm.fit(data, floc=0)
                ks_stat, ks_pvalue = stats.kstest(
                    data, lambda x: stats.lognorm.cdf(x, shape, loc, scale)
                )
            results["lognormal"] = {
                "shape": float(shape),
                "location": float(loc),
                "scale": float(scale),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "aic": float(
                    self._calculate_aic(data, stats.lognorm, shape, loc, scale, n_free_params=2)
                ),
            }
        except (ValueError, TypeError, RuntimeError):
            pass

        # Gamma distribution
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha, loc, scale = stats.gamma.fit(data, floc=0)
                ks_stat, ks_pvalue = stats.kstest(
                    data, lambda x: stats.gamma.cdf(x, alpha, loc, scale)
                )
            results["gamma"] = {
                "alpha": float(alpha),
                "location": float(loc),
                "scale": float(scale),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "aic": float(
                    self._calculate_aic(data, stats.gamma, alpha, loc, scale, n_free_params=2)
                ),
            }
        except (ValueError, TypeError, RuntimeError):
            pass

        # Exponential distribution
        try:
            loc, scale = stats.expon.fit(data, floc=0)
            ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.expon.cdf(x, loc, scale))
            results["exponential"] = {
                "location": float(loc),
                "scale": float(scale),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "aic": float(self._calculate_aic(data, stats.expon, loc, scale, n_free_params=1)),
            }
        except (ValueError, TypeError, RuntimeError):
            pass

        return results

    def _calculate_aic(
        self,
        data: np.ndarray,
        distribution: stats.rv_continuous,
        *params,
        n_free_params: Optional[int] = None,
    ) -> float:
        """Calculate corrected Akaike Information Criterion (AICc) for distribution fit.

        Uses AICc (Burnham & Anderson, 2002) which adds a finite-sample correction
        to the standard AIC. AICc converges to AIC as n -> infinity, so there is no
        downside to using it unconditionally.

        Args:
            data: Data points
            distribution: Scipy distribution object
            params: Distribution parameters (including any fixed parameters
                needed for evaluation)
            n_free_params: Number of free (estimated) parameters. If None,
                defaults to len(params). Pass explicitly when some parameters
                are fixed during fitting (e.g., floc=0).

        Returns:
            AICc value
        """
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        k = n_free_params if n_free_params is not None else len(params)
        n = len(data)
        aic = 2 * k - 2 * log_likelihood
        # AICc correction (Burnham & Anderson, 2002; Hurvich & Tsai, 1989)
        if n - k - 1 > 0:
            aic += 2 * k * (k + 1) / (n - k - 1)
        return float(aic)

    def _calculate_confidence_intervals(self, data: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals.

        Args:
            data: Input data

        Returns:
            Dictionary of confidence intervals
        """
        n_samples = len(data)
        alpha = 1 - self.confidence_level

        # Bootstrap samples
        means = []
        medians = []
        stds = []

        for _ in range(self.bootstrap_iterations):
            sample = self._rng.choice(data, size=n_samples, replace=True)
            means.append(np.mean(sample))
            medians.append(np.median(sample))
            stds.append(np.std(sample))

        # Calculate percentile confidence intervals
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return {
            "mean": (
                float(np.percentile(means, lower_percentile)),
                float(np.percentile(means, upper_percentile)),
            ),
            "median": (
                float(np.percentile(medians, lower_percentile)),
                float(np.percentile(medians, upper_percentile)),
            ),
            "std": (
                float(np.percentile(stds, lower_percentile)),
                float(np.percentile(stds, upper_percentile)),
            ),
        }

    def _perform_hypothesis_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Perform various hypothesis tests on data.

        Args:
            data: Input data

        Returns:
            Dictionary of test results
        """
        results = {}

        # Normality tests
        # Shapiro test requires at least 3 samples
        if len(data) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(
                data[: min(5000, len(data))]
            )  # Limit sample size
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Jarque-Bera test needs sufficient samples
        if len(data) >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                jarque_bera_stat, jarque_bera_p = stats.jarque_bera(data)
        else:
            jarque_bera_stat, jarque_bera_p = np.nan, np.nan

        results["normality"] = {
            "shapiro_statistic": float(shapiro_stat),
            "shapiro_pvalue": float(shapiro_p),
            "jarque_bera_statistic": float(jarque_bera_stat),
            "jarque_bera_pvalue": float(jarque_bera_p),
        }

        # One-sample t-test (test if mean is different from 0)
        if len(data) >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stat, t_p = stats.ttest_1samp(data, 0)
        else:
            t_stat, t_p = np.nan, np.nan
        results["t_test"] = {"statistic": float(t_stat), "pvalue": float(t_p)}

        # Autocorrelation test (Ljung-Box test approximation)
        if len(data) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(data[:-1], data[1:])
                lag1_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            results["autocorrelation"] = {
                "lag1_correlation": float(lag1_corr),
                "significant": float(abs(lag1_corr) > 2 / np.sqrt(len(data))),
            }

        return results

    def _calculate_extreme_values(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate extreme value statistics.

        Args:
            data: Input data

        Returns:
            Dictionary of extreme value statistics
        """
        percentiles = [0.1, 1, 5, 95, 99, 99.9]
        extreme_stats = {}

        for p in percentiles:
            extreme_stats[f"percentile_{p}"] = float(np.percentile(data, p))

        # Tail indices
        threshold_lower = np.percentile(data, 5)
        threshold_upper = np.percentile(data, 95)

        lower_tail = data[data <= threshold_lower]
        upper_tail = data[data >= threshold_upper]

        if len(lower_tail) > 1 and np.mean(lower_tail) != 0:
            extreme_stats["lower_tail_index"] = float(np.std(lower_tail) / np.mean(lower_tail))

        if len(upper_tail) > 1 and np.mean(upper_tail) != 0:
            extreme_stats["upper_tail_index"] = float(np.std(upper_tail) / np.mean(upper_tail))

        # Expected shortfall (CVaR) - average of losses in the upper tail
        expected_shortfall = np.mean(data[data >= threshold_upper])
        extreme_stats["expected_shortfall_95%"] = float(expected_shortfall)

        return extreme_stats


class TDigest:
    """T-digest data structure for streaming quantile estimation.

    Implements the merging digest variant from Dunning & Ertl (2019).
    Provides accurate quantile estimates, especially at the tails,
    with bounded memory usage proportional to the compression parameter.

    The t-digest maintains a sorted set of centroids (mean, weight) that
    adaptively cluster data points. Clusters near the tails (q->0 or q->1)
    are kept small for precision, while clusters near the median can be larger.

    Args:
        compression: Controls accuracy vs memory tradeoff. Higher values
            give more accuracy but use more memory. Typical range: 100-300.
            Default 200 gives ~0.2-1% error at median, ~0.005-0.05% at q01/q99.

    References:
        Dunning, T. & Ertl, O. (2019). "Computing Extremely Accurate Quantiles
        Using t-Digests." arXiv:1902.04023.
    """

    def __init__(self, compression: float = 200):
        self.compression = compression
        self._means: np.ndarray = np.array([], dtype=np.float64)
        self._weights: np.ndarray = np.array([], dtype=np.float64)
        self._buffer: List[float] = []
        self._buffer_capacity = max(int(compression * 5), 500)
        self._total_weight = 0.0
        self._min_val = float("inf")
        self._max_val = float("-inf")
        self._count = 0
        self._merge_direction = True  # Alternate merge direction for better accuracy

    def update(self, value: float) -> None:
        """Add a single observation to the digest.

        Args:
            value: The value to add.

        Raises:
            ValueError: If the value is NaN or infinity.
        """
        if not math.isfinite(value):
            raise ValueError(f"TDigest does not accept non-finite values: {value}")
        self._buffer.append(value)
        self._min_val = min(self._min_val, value)
        self._max_val = max(self._max_val, value)
        self._count += 1
        if len(self._buffer) >= self._buffer_capacity:
            self._flush()

    def update_batch(self, values: np.ndarray) -> None:
        """Add an array of observations to the digest.

        Args:
            values: Array of values to add.

        Raises:
            ValueError: If any value is NaN or infinity.
        """
        if len(values) == 0:
            return
        flat = values.ravel()
        if not np.all(np.isfinite(flat)):
            raise ValueError("TDigest does not accept non-finite values (NaN or inf)")
        self._buffer.extend(flat.tolist())
        min_v = float(np.min(flat))
        max_v = float(np.max(flat))
        self._min_val = min(self._min_val, min_v)
        self._max_val = max(self._max_val, max_v)
        self._count += len(flat)
        if len(self._buffer) >= self._buffer_capacity:
            self._flush()

    def merge(self, other: "TDigest") -> None:
        """Merge another t-digest into this one.

        After merging, this digest contains the combined information from both
        digests. The other digest is not modified.

        Args:
            other: Another TDigest to merge into this one.
        """
        # Read other's centroids and buffer without mutating it (#335)
        other_means = other._means.copy()
        other_weights = other._weights.copy()
        if other._buffer:
            buf = np.array(other._buffer, dtype=np.float64)
            if len(other_means) > 0:
                other_means = np.concatenate([other_means, buf])
                other_weights = np.concatenate([other_weights, np.ones(len(buf), dtype=np.float64)])
            else:
                other_means = buf
                other_weights = np.ones(len(buf), dtype=np.float64)

        if len(other_means) == 0 and other._count == 0:
            return

        self._flush()

        if len(self._means) == 0 and self._count == 0:
            # Bootstrap self from the other's data by merging centroids
            self._min_val = other._min_val
            self._max_val = other._max_val
            self._count = other._count
            self._merge_centroids(other_means, other_weights)
            return

        all_means = np.concatenate([self._means, other_means])
        all_weights = np.concatenate([self._weights, other_weights])
        self._count += other._count
        self._min_val = min(self._min_val, other._min_val)
        self._max_val = max(self._max_val, other._max_val)
        self._merge_centroids(all_means, all_weights)

    def quantile(self, q: float) -> float:
        """Estimate a single quantile.

        Args:
            q: Quantile to estimate, in range [0, 1].

        Returns:
            Estimated value at the given quantile.

        Raises:
            ValueError: If the digest is empty.
        """
        self._flush()

        if len(self._means) == 0:
            raise ValueError("Cannot compute quantile of empty digest")

        if q <= 0:
            return self._min_val
        if q >= 1:
            return self._max_val

        if len(self._means) == 1:
            return float(self._means[0])

        total = self._total_weight
        target = q * total

        # Compute centroid weight centers (cumulative weight at center of each centroid)
        cum = np.cumsum(self._weights)
        centers = cum - self._weights / 2.0

        # Left tail: target before center of first centroid
        if target <= centers[0]:
            if centers[0] > 0:
                return float(self._min_val + (self._means[0] - self._min_val) * target / centers[0])
            return float(self._min_val)

        # Right tail: target after center of last centroid
        if target >= centers[-1]:
            remaining = total - centers[-1]
            if remaining > 0:
                return float(
                    self._means[-1]
                    + (self._max_val - self._means[-1]) * (target - centers[-1]) / remaining
                )
            return float(self._max_val)

        # Interior: interpolate between adjacent centroid centers
        idx = int(np.searchsorted(centers, target, side="right")) - 1
        idx = max(0, min(idx, len(centers) - 2))

        left_center = centers[idx]
        right_center = centers[idx + 1]

        if right_center > left_center:
            t = (target - left_center) / (right_center - left_center)
            return float(self._means[idx] + t * (self._means[idx + 1] - self._means[idx]))
        return float(self._means[idx])

    def quantiles(self, qs: List[float]) -> Dict[str, float]:
        """Estimate multiple quantiles.

        Args:
            qs: List of quantiles to estimate, each in range [0, 1].

        Returns:
            Dictionary mapping per-mille quantile keys (e.g. ``q0250``
            for the 25th percentile) to estimated values.
        """
        results = {}
        for q in sorted(qs):
            key = format_quantile_key(q)
            results[key] = self.quantile(q)
        return results

    def cdf(self, value: float) -> float:
        """Estimate the cumulative distribution function at a value.

        Args:
            value: The value at which to estimate the CDF.

        Returns:
            Estimated probability P(X <= value).

        Raises:
            ValueError: If the digest is empty.
        """
        self._flush()

        if len(self._means) == 0:
            raise ValueError("Cannot compute CDF of empty digest")

        if value <= self._min_val:
            return 0.0
        if value >= self._max_val:
            return 1.0

        if len(self._means) == 1:
            # Single centroid: linear interpolation between min and max
            if self._max_val > self._min_val:
                return (value - self._min_val) / (self._max_val - self._min_val)
            return 0.5

        total = self._total_weight
        cum = np.cumsum(self._weights)
        centers = cum - self._weights / 2.0

        # Before first centroid mean
        if value <= self._means[0]:
            if self._means[0] > self._min_val:
                t = (value - self._min_val) / (self._means[0] - self._min_val)
                return float(t * centers[0] / total)
            return float(centers[0] / total)

        # After last centroid mean
        if value >= self._means[-1]:
            if self._max_val > self._means[-1]:
                t = (value - self._means[-1]) / (self._max_val - self._means[-1])
                return float((centers[-1] + t * (total - centers[-1])) / total)
            return float(centers[-1] / total)

        # Interior: find bracketing centroids
        idx = int(np.searchsorted(self._means, value, side="right")) - 1
        idx = max(0, min(idx, len(self._means) - 2))

        left_mean = self._means[idx]
        right_mean = self._means[idx + 1]

        if right_mean > left_mean:
            t = (value - left_mean) / (right_mean - left_mean)
            weight_pos = centers[idx] + t * (centers[idx + 1] - centers[idx])
            return float(weight_pos / total)
        return float(centers[idx] / total)

    @property
    def centroid_count(self) -> int:
        """Return the number of centroids currently stored."""
        self._flush()
        return len(self._means)

    def __len__(self) -> int:
        """Return the total count of observations added."""
        return self._count

    def _flush(self) -> None:
        """Merge buffered values into centroids."""
        if not self._buffer:
            return

        buf = np.array(self._buffer, dtype=np.float64)
        self._buffer = []

        if len(self._means) > 0:
            all_means = np.concatenate([self._means, buf])
            all_weights = np.concatenate([self._weights, np.ones(len(buf), dtype=np.float64)])
        else:
            all_means = buf
            all_weights = np.ones(len(buf), dtype=np.float64)

        self._merge_centroids(all_means, all_weights)

    def _merge_centroids(self, means: np.ndarray, weights: np.ndarray) -> None:
        """Core merge step: sort centroids and merge under scale function constraints.

        Alternates merge direction (left-to-right / right-to-left) for balanced accuracy.
        """
        total = float(np.sum(weights))

        # Sort by mean
        order = np.argsort(means, kind="mergesort")
        means = means[order]
        weights = weights[order]

        # Alternate merge direction for better accuracy at both tails
        if not self._merge_direction:
            means = means[::-1]
            weights = weights[::-1]
        self._merge_direction = not self._merge_direction

        result_m: List[float] = []
        result_w: List[float] = []

        cum_weight = 0.0
        cur_m = float(means[0])
        cur_w = float(weights[0])

        for i in range(1, len(means)):
            proposed_w = cur_w + float(weights[i])
            q_left = cum_weight / total
            q_right = (cum_weight + proposed_w) / total

            # Scale function constraint: k(q_right) - k(q_left) <= 1
            if self._k(q_right) - self._k(q_left) <= 1.0:
                # Merge into current centroid
                cur_m = (cur_m * cur_w + float(means[i]) * float(weights[i])) / proposed_w
                cur_w = proposed_w
            else:
                # Emit current centroid, start new one
                result_m.append(cur_m)
                result_w.append(cur_w)
                cum_weight += cur_w
                cur_m = float(means[i])
                cur_w = float(weights[i])

        # Emit last centroid
        result_m.append(cur_m)
        result_w.append(cur_w)

        new_means = np.array(result_m, dtype=np.float64)
        new_weights = np.array(result_w, dtype=np.float64)

        # If we merged right-to-left, reverse back to sorted order
        if self._merge_direction:  # We already flipped the flag, so check current state
            new_order = np.argsort(new_means, kind="mergesort")
            new_means = new_means[new_order]
            new_weights = new_weights[new_order]

        self._means = new_means
        self._weights = new_weights
        self._total_weight = total

    def _k(self, q: float) -> float:
        """Scale function k1 from Dunning & Ertl (2019).

        k1(q) = (delta / (2*pi)) * arcsin(2*q - 1)

        This scale function provides highest precision at the tails
        (q near 0 or 1) where insurance risk metrics (VaR, TVaR) are computed.
        """
        q = max(1e-15, min(q, 1 - 1e-15))
        return (self.compression / (2.0 * math.pi)) * math.asin(2.0 * q - 1.0)


class QuantileCalculator:
    """Efficient quantile calculation for large datasets."""

    def __init__(self, quantiles: Optional[List[float]] = None, seed: Optional[int] = None):
        """Initialize quantile calculator.

        Args:
            quantiles: List of quantiles to calculate (0-1 range)
            seed: Optional random seed for reproducibility
        """
        if quantiles is None:
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        self.quantiles = sorted(quantiles)
        self._rng = np.random.default_rng(seed)

    @lru_cache(maxsize=128)
    def calculate_quantiles(self, data_hash: int, method: str = "linear") -> Dict[str, float]:
        """Calculate quantiles with caching.

        Args:
            data_hash: Hash of data array for caching
            method: Interpolation method

        Returns:
            Dictionary of quantile values
        """
        # This is a placeholder - actual data needs to be passed separately
        # due to hashing limitations
        return {}

    def calculate(self, data: np.ndarray, method: str = "linear") -> Dict[str, float]:
        """Calculate quantiles for data.

        Args:
            data: Input data array
            method: Interpolation method ('linear', 'nearest', 'lower', 'higher', 'midpoint')

        Returns:
            Dictionary of quantile values
        """
        results = {}

        # Use numpy's percentile function for compatibility
        quantile_values = np.percentile(data, [q * 100 for q in self.quantiles])

        for q, value in zip(self.quantiles, quantile_values):
            results[format_quantile_key(q)] = float(value)

        return results

    def streaming_quantiles(
        self, data_stream: np.ndarray, compression: float = 200
    ) -> Dict[str, float]:
        """Calculate quantiles for streaming data using the t-digest algorithm.

        Uses the t-digest merging digest algorithm (Dunning & Ertl, 2019) for
        streaming quantile estimation with bounded memory and high accuracy,
        especially at tail quantiles relevant to insurance risk metrics.

        Args:
            data_stream: Streaming data array
            compression: Controls accuracy vs memory tradeoff. Higher values
                give more accuracy but use more memory. Typical range: 100-300.
                Default 200 gives ~0.2-1% error at median, ~0.005-0.05% at
                q01/q99. Passed directly to TDigest.

        Returns:
            Dictionary of approximate quantile values
        """
        if len(data_stream) <= int(compression * 5):
            return self.calculate(data_stream)

        digest = TDigest(compression=compression)
        digest.update_batch(data_stream)
        return digest.quantiles(self.quantiles)


class DistributionFitter:
    """Fit and compare multiple probability distributions to data."""

    DISTRIBUTIONS = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "gamma": stats.gamma,
        "exponential": stats.expon,
        "weibull": stats.weibull_min,
        "beta": stats.beta,
        "pareto": stats.pareto,
        "uniform": stats.uniform,
    }

    def __init__(self):
        """Initialize distribution fitter."""
        self.fitted_params = {}
        self.goodness_of_fit = {}

    def fit_all(self, data: np.ndarray, distributions: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit multiple distributions and compare goodness of fit.

        Args:
            data: Input data
            distributions: List of distributions to fit (None for all)

        Returns:
            DataFrame comparing distribution fits
        """
        if distributions is None:
            distributions = list(self.DISTRIBUTIONS.keys())

        results = []

        for dist_name in distributions:
            if dist_name not in self.DISTRIBUTIONS:
                continue

            dist = self.DISTRIBUTIONS[dist_name]

            try:
                # Fit distribution
                params = dist.fit(data)
                self.fitted_params[dist_name] = params

                # Calculate goodness of fit metrics
                ks_stat, ks_p = stats.kstest(
                    data, lambda x, dist=dist, params=params: dist.cdf(x, *params)
                )
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * log_likelihood
                # AICc correction (Burnham & Anderson, 2002)
                if n - k - 1 > 0:
                    aic += 2 * k * (k + 1) / (n - k - 1)
                bic = k * np.log(n) - 2 * log_likelihood

                results.append(
                    {
                        "distribution": dist_name,
                        "n_params": len(params),
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_p,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                    }
                )

                self.goodness_of_fit[dist_name] = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_p,
                    "aic": aic,
                    "bic": bic,
                }

            except (ValueError, TypeError, RuntimeError) as e:
                results.append({"distribution": dist_name, "error": str(e)})

        # Create DataFrame and sort by AIC
        df = pd.DataFrame(results)
        if "aic" in df.columns:
            df = df.sort_values("aic")

        return df

    def get_best_distribution(self, criterion: str = "aic") -> Tuple[str, Dict[str, float]]:
        """Get the best-fitting distribution based on criterion.

        Args:
            criterion: Selection criterion ('aic', 'bic', 'ks_pvalue')

        Returns:
            Tuple of (distribution name, parameters)
        """
        if not self.goodness_of_fit:
            raise ValueError("No distributions fitted yet")

        if criterion == "ks_pvalue":
            # Higher p-value is better
            best_dist = max(
                self.goodness_of_fit.items(), key=lambda x: x[1].get(criterion, -np.inf)
            )[0]
        else:
            # Lower AIC/BIC is better
            best_dist = min(
                self.goodness_of_fit.items(), key=lambda x: x[1].get(criterion, np.inf)
            )[0]

        return best_dist, self.fitted_params[best_dist]

    def generate_qq_plot_data(
        self, data: np.ndarray, distribution: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data for Q-Q plot.

        Args:
            data: Original data
            distribution: Distribution name

        Returns:
            Tuple of (theoretical quantiles, sample quantiles)
        """
        if distribution not in self.fitted_params:
            raise ValueError(f"Distribution {distribution} not fitted")

        params = self.fitted_params[distribution]
        dist = self.DISTRIBUTIONS[distribution]

        # Calculate quantiles
        n = len(data)
        theoretical_quantiles = np.array(
            [dist.ppf((i - 0.5) / n, *params) for i in range(1, n + 1)]
        )
        sample_quantiles = np.sort(data)

        return theoretical_quantiles, sample_quantiles


class SummaryReportGenerator:
    """Generate formatted summary reports for simulation results."""

    def __init__(self, style: str = "markdown"):
        """Initialize report generator.

        Args:
            style: Report style ('markdown', 'html', 'latex')
        """
        self.style = style

    def generate_report(
        self,
        summary: StatisticalSummary,
        title: str = "Simulation Results Summary",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate formatted report.

        Args:
            summary: Statistical summary object
            title: Report title
            metadata: Additional metadata to include

        Returns:
            Formatted report string
        """
        if self.style == "markdown":
            return self._generate_markdown_report(summary, title, metadata)
        if self.style == "html":
            return self._generate_html_report(summary, title, metadata)
        if self.style == "latex":
            return self._generate_latex_report(summary, title, metadata)
        raise ValueError(f"Unsupported style: {self.style}")

    def _generate_markdown_report(
        self, summary: StatisticalSummary, title: str, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate Markdown report.

        Args:
            summary: Statistical summary
            title: Report title
            metadata: Additional metadata

        Returns:
            Markdown formatted report
        """
        report = io.StringIO()

        # Title and metadata
        report.write(f"# {title}\n\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if metadata:
            report.write("## Metadata\n\n")
            for key, value in metadata.items():
                report.write(f"- **{key}**: {value}\n")
            report.write("\n")

        # Basic statistics
        report.write("## Basic Statistics\n\n")
        report.write("| Metric | Value |\n")
        report.write("|--------|-------|\n")
        for metric, value in summary.basic_stats.items():
            report.write(f"| {metric} | {value:.6f} |\n")
        report.write("\n")

        # Distribution fits
        if summary.distribution_params:
            report.write("## Distribution Fitting\n\n")
            for dist, params in summary.distribution_params.items():
                report.write(f"### {dist.title()} Distribution\n\n")
                for param, value in params.items():
                    report.write(f"- {param}: {value:.6f}\n")
                report.write("\n")

        # Confidence intervals
        report.write("## Confidence Intervals\n\n")
        report.write("| Metric | Lower | Upper |\n")
        report.write("|--------|-------|-------|\n")
        for metric, (lower, upper) in summary.confidence_intervals.items():
            report.write(f"| {metric} | {lower:.6f} | {upper:.6f} |\n")
        report.write("\n")

        # Hypothesis tests
        if summary.hypothesis_tests:
            report.write("## Hypothesis Tests\n\n")
            for test, results in summary.hypothesis_tests.items():
                report.write(f"### {test.replace('_', ' ').title()}\n\n")
                for metric, value in results.items():
                    report.write(f"- {metric}: {value:.6f}\n")
                report.write("\n")

        # Extreme values
        report.write("## Extreme Value Statistics\n\n")
        report.write("| Metric | Value |\n")
        report.write("|--------|-------|\n")
        for metric, value in summary.extreme_values.items():
            report.write(f"| {metric} | {value:.6f} |\n")

        return report.getvalue()

    def _generate_html_report(
        self, summary: StatisticalSummary, title: str, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate HTML report.

        Args:
            summary: Statistical summary
            title: Report title
            metadata: Additional metadata

        Returns:
            HTML formatted report
        """
        df = summary.to_dataframe()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metadata {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        if metadata:
            html += '<div class="metadata"><h2>Metadata</h2><ul>'
            for key, value in metadata.items():
                html += f"<li><strong>{key}</strong>: {value}</li>"
            html += "</ul></div>"

        html += df.to_html(index=False, classes="results-table")
        html += "</body></html>"

        return str(html)

    def _generate_latex_report(
        self, summary: StatisticalSummary, title: str, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate LaTeX report.

        Args:
            summary: Statistical summary
            title: Report title
            metadata: Additional metadata

        Returns:
            LaTeX formatted report
        """
        df = summary.to_dataframe()

        latex = f"""
\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\title{{{title}}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        if metadata:
            latex += "\\section{Metadata}\n\\begin{itemize}\n"
            for key, value in metadata.items():
                latex += f"\\item \\textbf{{{key}}}: {value}\n"
            latex += "\\end{itemize}\n"

        latex += "\\section{Results}\n"
        latex += df.to_latex(index=False, longtable=True)
        latex += "\\end{document}"

        return str(latex)

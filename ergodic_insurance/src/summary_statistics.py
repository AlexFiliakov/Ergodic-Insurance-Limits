"""Comprehensive summary statistics and report generation for simulation results.

This module provides statistical analysis tools, distribution fitting utilities,
and formatted report generation for Monte Carlo simulation results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import io
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


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

    def __init__(self, confidence_level: float = 0.95, bootstrap_iterations: int = 1000):
        """Initialize summary statistics calculator.

        Args:
            confidence_level: Confidence level for intervals
            bootstrap_iterations: Number of bootstrap iterations
        """
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations

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
        if weights is None:
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
                "stderr": float(np.std(data) / np.sqrt(len(data))),
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
                "aic": float(self._calculate_aic(data, stats.lognorm, shape, loc, scale)),
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
                "aic": float(self._calculate_aic(data, stats.gamma, alpha, loc, scale)),
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
                "aic": float(self._calculate_aic(data, stats.expon, loc, scale)),
            }
        except (ValueError, TypeError, RuntimeError):
            pass

        return results

    def _calculate_aic(self, data: np.ndarray, distribution: stats.rv_continuous, *params) -> float:
        """Calculate Akaike Information Criterion for distribution fit.

        Args:
            data: Data points
            distribution: Scipy distribution object
            params: Distribution parameters

        Returns:
            AIC value
        """
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        n_params = len(params)
        return float(2 * n_params - 2 * log_likelihood)

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
            sample = np.random.choice(data, size=n_samples, replace=True)
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

        if len(lower_tail) > 1:
            extreme_stats["lower_tail_index"] = float(np.std(lower_tail) / np.mean(lower_tail))

        if len(upper_tail) > 1:
            extreme_stats["upper_tail_index"] = float(np.std(upper_tail) / np.mean(upper_tail))

        # Expected shortfall (CVaR)
        var_95 = np.percentile(data, 5)
        expected_shortfall = np.mean(data[data <= var_95])
        extreme_stats["expected_shortfall_5%"] = float(expected_shortfall)

        return extreme_stats


class QuantileCalculator:
    """Efficient quantile calculation for large datasets."""

    def __init__(self, quantiles: Optional[List[float]] = None):
        """Initialize quantile calculator.

        Args:
            quantiles: List of quantiles to calculate (0-1 range)
        """
        if quantiles is None:
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        self.quantiles = sorted(quantiles)

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
            results[f"q{int(q*100):03d}"] = float(value)

        return results

    def streaming_quantiles(
        self, data_stream: np.ndarray, buffer_size: int = 10000
    ) -> Dict[str, float]:
        """Calculate quantiles for streaming data.

        Uses P-square algorithm for online quantile estimation.

        Args:
            data_stream: Streaming data array
            buffer_size: Size of buffer for approximation

        Returns:
            Dictionary of approximate quantile values
        """
        # Simplified reservoir sampling for quantile approximation
        if len(data_stream) <= buffer_size:
            return self.calculate(data_stream)

        # Reservoir sampling
        reservoir = data_stream[:buffer_size].copy()

        for i in range(buffer_size, len(data_stream)):
            j = np.random.randint(0, i + 1)
            if j < buffer_size:
                reservoir[j] = data_stream[i]

        return self.calculate(reservoir)


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
                aic = 2 * len(params) - 2 * log_likelihood
                bic = len(params) * np.log(len(data)) - 2 * log_likelihood

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

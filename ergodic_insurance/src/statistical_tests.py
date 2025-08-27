"""Statistical hypothesis testing utilities for simulation results.

This module provides bootstrap-based hypothesis testing functions for
comparing strategies, validating performance differences, and assessing
statistical significance of simulation outcomes.

Example:
    >>> from statistical_tests import test_difference_in_means
    >>> import numpy as np

    >>> # Compare two strategies
    >>> strategy_a_returns = np.random.normal(0.08, 0.02, 1000)
    >>> strategy_b_returns = np.random.normal(0.10, 0.03, 1000)

    >>> result = test_difference_in_means(
    ...     strategy_a_returns,
    ...     strategy_b_returns,
    ...     alternative='less'
    ... )
    >>> print(f"P-value: {result.p_value:.4f}")
    >>> print(f"Strategy B is better: {result.reject_null}")

Attributes:
    DEFAULT_N_BOOTSTRAP (int): Default bootstrap iterations for tests (10000).
    DEFAULT_ALPHA (float): Default significance level (0.05).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import stats

from .bootstrap_analysis import BootstrapAnalyzer, BootstrapResult


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results.

    Attributes:
        test_statistic: Computed test statistic value.
        p_value: P-value from the test.
        reject_null: Whether to reject null hypothesis at given alpha.
        confidence_interval: Confidence interval for the test statistic.
        null_hypothesis: Description of null hypothesis.
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater').
        alpha: Significance level used.
        method: Testing method used.
        bootstrap_distribution: Bootstrap distribution of test statistic.
        metadata: Additional test information.
    """

    test_statistic: float
    p_value: float
    reject_null: bool
    confidence_interval: Tuple[float, float]
    null_hypothesis: str
    alternative: str
    alpha: float
    method: str
    bootstrap_distribution: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Generate human-readable summary of test results.

        Returns:
            Formatted string with test results and interpretation.
        """
        summary = [
            f"Hypothesis Test Results",
            f"{'=' * 40}",
            f"Null Hypothesis: {self.null_hypothesis}",
            f"Alternative: {self.alternative}",
            f"Test Statistic: {self.test_statistic:.6f}",
            f"P-value: {self.p_value:.4f}",
            f"Significance Level: {self.alpha:.3f}",
            f"Reject Null: {'Yes' if self.reject_null else 'No'}",
            f"{(1-self.alpha):.1%} Confidence Interval:",
            f"  [{self.confidence_interval[0]:.6f}, {self.confidence_interval[1]:.6f}]",
            f"Method: {self.method}",
        ]

        # Add interpretation
        if self.reject_null:
            summary.append(f"\nConclusion: Significant difference detected (p < {self.alpha})")
        else:
            summary.append(f"\nConclusion: No significant difference (p >= {self.alpha})")

        return "\n".join(summary)


def difference_in_means_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> HypothesisTestResult:
    """Test difference in means between two samples using bootstrap.

    Tests the null hypothesis that the means of two populations are equal
    against various alternatives using bootstrap resampling.

    Args:
        sample1: First sample array.
        sample2: Second sample array.
        alternative: Type of alternative hypothesis:
            - 'two-sided': means are different
            - 'less': mean1 < mean2
            - 'greater': mean1 > mean2
        alpha: Significance level (default 0.05).
        n_bootstrap: Number of bootstrap iterations (default 10000).
        seed: Random seed for reproducibility.

    Returns:
        HypothesisTestResult containing test statistics and decision.

    Raises:
        ValueError: If alternative is not valid.

    Example:
        >>> # Test if Strategy A has lower returns than Strategy B
        >>> result = test_difference_in_means(
        ...     returns_a, returns_b, alternative='less'
        ... )
        >>> if result.reject_null:
        ...     print("Strategy B significantly outperforms Strategy A")
    """
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(
            f"Alternative must be 'two-sided', 'less', or 'greater', got {alternative}"
        )

    # Calculate observed difference
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    observed_diff = mean1 - mean2

    # Combine samples for permutation test
    combined = np.concatenate([sample1, sample2])
    n1, n2 = len(sample1), len(sample2)

    # Bootstrap under null hypothesis (permutation)
    rng = np.random.RandomState(seed)
    bootstrap_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Permute combined sample
        permuted = rng.permutation(combined)

        # Split into two groups
        perm_sample1 = permuted[:n1]
        perm_sample2 = permuted[n1:]

        # Calculate difference
        bootstrap_diffs[i] = np.mean(perm_sample1) - np.mean(perm_sample2)

    # Calculate p-value based on alternative
    if alternative == "two-sided":
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    elif alternative == "less":
        p_value = np.mean(bootstrap_diffs <= observed_diff)
    else:  # greater
        p_value = np.mean(bootstrap_diffs >= observed_diff)

    # Calculate confidence interval for difference
    analyzer = BootstrapAnalyzer(
        n_bootstrap=n_bootstrap, confidence_level=1 - alpha, seed=seed, show_progress=False
    )

    def diff_statistic(indices: np.ndarray) -> float:
        """Calculate difference in means for bootstrap sample."""
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)
        return np.mean(sample1[idx1]) - np.mean(sample2[idx2])

    # Get confidence interval for actual difference
    ci_result = analyzer.confidence_interval(
        np.arange(n1 + n2),  # Dummy array for indexing
        lambda x: observed_diff,  # Return observed for original
    )

    # Recalculate proper CI
    bootstrap_actual_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)
        bootstrap_actual_diffs[i] = np.mean(sample1[idx1]) - np.mean(sample2[idx2])

    ci = np.percentile(bootstrap_actual_diffs, [(alpha / 2) * 100, (1 - alpha / 2) * 100])

    return HypothesisTestResult(
        test_statistic=observed_diff,
        p_value=float(p_value),
        reject_null=p_value < alpha,
        confidence_interval=(float(ci[0]), float(ci[1])),
        null_hypothesis="mean1 = mean2",
        alternative=alternative,
        alpha=alpha,
        method="bootstrap permutation test",
        bootstrap_distribution=bootstrap_diffs,
        metadata={
            "mean1": mean1,
            "mean2": mean2,
            "n1": n1,
            "n2": n2,
            "n_bootstrap": n_bootstrap,
        },
    )


def ratio_of_metrics_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    null_ratio: float = 1.0,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> HypothesisTestResult:
    """Test ratio of metrics between two samples using bootstrap.

    Tests whether the ratio of a statistic (e.g., mean, median) between
    two samples equals a specified value (typically 1.0).

    Args:
        sample1: First sample array.
        sample2: Second sample array.
        statistic: Function to compute on each sample (default: mean).
        null_ratio: Null hypothesis ratio value (default: 1.0).
        alternative: Alternative hypothesis type.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        HypothesisTestResult for the ratio test.

    Example:
        >>> # Test if ROE ratio differs from 1.0
        >>> result = test_ratio_of_metrics(
        ...     roe_strategy_a,
        ...     roe_strategy_b,
        ...     statistic=np.median,
        ...     null_ratio=1.0
        ... )
    """
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(f"Alternative must be 'two-sided', 'less', or 'greater'")

    # Calculate observed ratio
    stat1 = statistic(sample1)
    stat2 = statistic(sample2)

    if stat2 == 0:
        warnings.warn("Denominator statistic is zero, cannot compute ratio")
        return HypothesisTestResult(
            test_statistic=np.inf,
            p_value=1.0,
            reject_null=False,
            confidence_interval=(np.nan, np.nan),
            null_hypothesis=f"ratio = {null_ratio}",
            alternative=alternative,
            alpha=alpha,
            method="bootstrap ratio test",
        )

    observed_ratio = stat1 / stat2

    # Bootstrap distribution of ratio
    n1, n2 = len(sample1), len(sample2)
    rng = np.random.RandomState(seed)
    bootstrap_ratios = []

    for i in range(n_bootstrap):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)

        boot_stat1 = statistic(sample1[idx1])
        boot_stat2 = statistic(sample2[idx2])

        if boot_stat2 != 0:
            bootstrap_ratios.append(boot_stat1 / boot_stat2)

    bootstrap_ratios = np.array(bootstrap_ratios)

    # Center around null ratio for hypothesis test
    centered_ratios = bootstrap_ratios - np.mean(bootstrap_ratios) + null_ratio

    # Calculate p-value
    if alternative == "two-sided":
        p_value = np.mean(
            np.abs(centered_ratios - null_ratio) >= np.abs(observed_ratio - null_ratio)
        )
    elif alternative == "less":
        p_value = np.mean(centered_ratios <= observed_ratio)
    else:  # greater
        p_value = np.mean(centered_ratios >= observed_ratio)

    # Confidence interval for ratio
    ci = np.percentile(bootstrap_ratios, [(alpha / 2) * 100, (1 - alpha / 2) * 100])

    return HypothesisTestResult(
        test_statistic=observed_ratio,
        p_value=float(p_value),
        reject_null=p_value < alpha,
        confidence_interval=(float(ci[0]), float(ci[1])),
        null_hypothesis=f"ratio = {null_ratio}",
        alternative=alternative,
        alpha=alpha,
        method="bootstrap ratio test",
        bootstrap_distribution=bootstrap_ratios,
        metadata={
            "stat1": stat1,
            "stat2": stat2,
            "null_ratio": null_ratio,
            "n_valid_bootstraps": len(bootstrap_ratios),
        },
    )


def paired_comparison_test(
    paired_differences: np.ndarray,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> HypothesisTestResult:
    """Test paired differences using bootstrap.

    Tests whether paired differences (e.g., from matched scenarios)
    have a mean equal to a specified value (typically 0).

    Args:
        paired_differences: Array of paired differences.
        null_value: Null hypothesis value for mean difference (default: 0).
        alternative: Alternative hypothesis type.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        HypothesisTestResult for the paired test.

    Example:
        >>> # Test if insurance improves outcomes
        >>> differences = outcomes_with_insurance - outcomes_without_insurance
        >>> result = paired_comparison_test(differences, alternative='greater')
    """
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(f"Alternative must be 'two-sided', 'less', or 'greater'")

    # Calculate observed mean difference
    observed_mean = np.mean(paired_differences)
    n = len(paired_differences)

    # Center differences around null value for bootstrap
    centered_diffs = paired_differences - observed_mean + null_value

    # Bootstrap distribution under null
    rng = np.random.RandomState(seed)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means[i] = np.mean(centered_diffs[indices])

    # Calculate p-value
    if alternative == "two-sided":
        p_value = np.mean(
            np.abs(bootstrap_means - null_value) >= np.abs(observed_mean - null_value)
        )
    elif alternative == "less":
        p_value = np.mean(bootstrap_means <= observed_mean)
    else:  # greater
        p_value = np.mean(bootstrap_means >= observed_mean)

    # Confidence interval for mean difference
    bootstrap_actual_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_actual_means[i] = np.mean(paired_differences[indices])

    ci = np.percentile(bootstrap_actual_means, [(alpha / 2) * 100, (1 - alpha / 2) * 100])

    return HypothesisTestResult(
        test_statistic=observed_mean,
        p_value=float(p_value),
        reject_null=p_value < alpha,
        confidence_interval=(float(ci[0]), float(ci[1])),
        null_hypothesis=f"mean difference = {null_value}",
        alternative=alternative,
        alpha=alpha,
        method="bootstrap paired test",
        bootstrap_distribution=bootstrap_means,
        metadata={
            "n_pairs": n,
            "std_difference": np.std(paired_differences),
            "null_value": null_value,
        },
    )


def bootstrap_hypothesis_test(
    data: np.ndarray,
    null_hypothesis: Callable[[np.ndarray], float],
    test_statistic: Callable[[np.ndarray], float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> HypothesisTestResult:
    """General bootstrap hypothesis testing framework.

    Allows testing of custom hypotheses using any test statistic.

    Args:
        data: Input data array.
        null_hypothesis: Function that transforms data to satisfy null.
        test_statistic: Function to compute test statistic.
        alternative: Alternative hypothesis type.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        HypothesisTestResult for the custom test.

    Example:
        >>> # Test if variance exceeds threshold
        >>> def null_transform(x):
        ...     return x * np.sqrt(threshold_var / np.var(x))
        >>> result = bootstrap_hypothesis_test(
        ...     data, null_transform, np.var, alternative='greater'
        ... )
    """
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(f"Alternative must be 'two-sided', 'less', or 'greater'")

    # Calculate observed test statistic
    observed_stat = test_statistic(data)

    # Transform data to satisfy null hypothesis
    null_data = null_hypothesis(data)

    # Bootstrap distribution under null
    n = len(null_data)
    rng = np.random.RandomState(seed)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_sample = null_data[indices]
        bootstrap_stats[i] = test_statistic(bootstrap_sample)

    # Calculate p-value
    if alternative == "two-sided":
        null_stat = test_statistic(null_data)
        p_value = np.mean(np.abs(bootstrap_stats - null_stat) >= np.abs(observed_stat - null_stat))
    elif alternative == "less":
        p_value = np.mean(bootstrap_stats <= observed_stat)
    else:  # greater
        p_value = np.mean(bootstrap_stats >= observed_stat)

    # Confidence interval for test statistic
    bootstrap_actual_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(len(data), size=len(data), replace=True)
        bootstrap_actual_stats[i] = test_statistic(data[indices])

    ci = np.percentile(bootstrap_actual_stats, [(alpha / 2) * 100, (1 - alpha / 2) * 100])

    return HypothesisTestResult(
        test_statistic=observed_stat,
        p_value=float(p_value),
        reject_null=p_value < alpha,
        confidence_interval=(float(ci[0]), float(ci[1])),
        null_hypothesis="Custom null hypothesis",
        alternative=alternative,
        alpha=alpha,
        method="bootstrap hypothesis test",
        bootstrap_distribution=bootstrap_stats,
    )


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multiple comparison correction to p-values.

    Adjusts p-values when multiple hypothesis tests are performed
    to control family-wise error rate or false discovery rate.

    Args:
        p_values: List of p-values from multiple tests.
        method: Correction method:
            - 'bonferroni': Bonferroni correction
            - 'holm': Holm-Bonferroni method
            - 'fdr': Benjamini-Hochberg FDR
        alpha: Overall significance level.

    Returns:
        Tuple of (adjusted_p_values, reject_decisions).

    Example:
        >>> p_vals = [0.01, 0.04, 0.03, 0.20]
        >>> adj_p, reject = multiple_comparison_correction(p_vals)
        >>> print(f"Significant tests: {np.sum(reject)}")
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)

    if method == "bonferroni":
        # Simple Bonferroni correction
        adjusted_p = np.minimum(p_values * n_tests, 1.0)
        reject = adjusted_p < alpha

    elif method == "holm":
        # Holm-Bonferroni method
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted_p = np.zeros(n_tests)
        reject = np.zeros(n_tests, dtype=bool)

        for i in range(n_tests):
            adj_p = sorted_p[i] * (n_tests - i)
            adjusted_p[sorted_idx[i]] = min(adj_p, 1.0)

            if adj_p < alpha:
                reject[sorted_idx[i]] = True
            else:
                break  # Stop testing once we fail to reject

    elif method == "fdr":
        # Benjamini-Hochberg FDR control
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted_p = np.zeros(n_tests)
        reject = np.zeros(n_tests, dtype=bool)

        # Find largest i where P(i) <= (i/m) * alpha
        threshold_found = False
        for i in range(n_tests - 1, -1, -1):
            threshold = ((i + 1) / n_tests) * alpha
            if sorted_p[i] <= threshold:
                threshold_found = True
                # Reject all hypotheses up to i
                for j in range(i + 1):
                    reject[sorted_idx[j]] = True
                break

        # Adjust p-values
        for i in range(n_tests):
            adj_p = sorted_p[i] * n_tests / (i + 1)
            adjusted_p[sorted_idx[i]] = min(adj_p, 1.0)

    else:
        raise ValueError(f"Method must be 'bonferroni', 'holm', or 'fdr', got {method}")

    return adjusted_p, reject

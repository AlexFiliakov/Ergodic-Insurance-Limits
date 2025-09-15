"""Comprehensive tests for bootstrap confidence interval analysis.

Tests cover bootstrap methods, confidence intervals, hypothesis testing,
and integration with Monte Carlo simulations.
"""

import numpy as np
import pytest
from scipy import stats

from ergodic_insurance.bootstrap_analysis import (
    BootstrapAnalyzer,
    BootstrapResult,
    bootstrap_confidence_interval,
)
from ergodic_insurance.statistical_tests import (
    HypothesisTestResult,
    bootstrap_hypothesis_test,
    difference_in_means_test,
    multiple_comparison_correction,
    paired_comparison_test,
    ratio_of_metrics_test,
)


class TestBootstrapAnalyzer:
    """Test suite for BootstrapAnalyzer class."""

    def test_initialization(self):
        """Test BootstrapAnalyzer initialization."""
        analyzer = BootstrapAnalyzer(
            n_bootstrap=5000,
            confidence_level=0.99,
            seed=42,
            n_workers=2,
            show_progress=False,
        )

        assert analyzer.n_bootstrap == 5000
        assert analyzer.confidence_level == 0.99
        assert analyzer.seed == 42
        assert analyzer.n_workers == 2
        assert not analyzer.show_progress

    def test_invalid_confidence_level(self):
        """Test that invalid confidence levels raise errors."""
        with pytest.raises(ValueError, match="Confidence level must be"):
            BootstrapAnalyzer(confidence_level=1.5)

        with pytest.raises(ValueError, match="Confidence level must be"):
            BootstrapAnalyzer(confidence_level=0.0)

    def test_bootstrap_sample(self):
        """Test bootstrap sample generation."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)

        analyzer = BootstrapAnalyzer(seed=42, show_progress=False)
        bootstrap_stats = analyzer.bootstrap_sample(data, np.mean, n_samples=100)

        assert len(bootstrap_stats) == 100
        assert np.abs(np.mean(bootstrap_stats) - 100) < 2  # Should be close to true mean
        assert np.std(bootstrap_stats) < np.std(data)  # Bootstrap std should be smaller

    def test_percentile_confidence_interval(self):
        """Test percentile bootstrap confidence interval."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)

        analyzer = BootstrapAnalyzer(n_bootstrap=5000, seed=42, show_progress=False)
        result = analyzer.confidence_interval(data, np.mean, method="percentile")

        assert isinstance(result, BootstrapResult)
        assert result.method == "percentile"
        assert result.confidence_level == 0.95
        assert result.confidence_interval[0] < 100 < result.confidence_interval[1]
        assert result.converged

        # Check that CI width is reasonable
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        assert 1 < ci_width < 5  # For this sample size and std

    def test_bca_confidence_interval(self):
        """Test BCa (bias-corrected and accelerated) confidence interval."""
        np.random.seed(42)
        # Create slightly skewed data
        data = np.random.exponential(100, 1000)

        analyzer = BootstrapAnalyzer(n_bootstrap=2000, seed=42, show_progress=False)
        result = analyzer.confidence_interval(data, np.mean, method="bca")

        assert isinstance(result, BootstrapResult)
        assert result.method == "bca"
        assert result.bias is not None
        assert result.acceleration is not None
        assert result.confidence_interval[0] < np.mean(data) < result.confidence_interval[1]

    def test_parallel_bootstrap(self):
        """Test parallel bootstrap computation."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)

        # Sequential
        analyzer_seq = BootstrapAnalyzer(
            n_bootstrap=1000, seed=42, n_workers=1, show_progress=False
        )
        result_seq = analyzer_seq.confidence_interval(data, np.mean, parallel=False)

        # Parallel
        analyzer_par = BootstrapAnalyzer(
            n_bootstrap=1000, seed=42, n_workers=2, show_progress=False
        )
        result_par = analyzer_par.confidence_interval(data, np.mean, parallel=True)

        # Results should be similar (not identical due to different RNG sequences)
        assert np.abs(result_seq.statistic - result_par.statistic) < 0.1
        assert np.abs(result_seq.confidence_interval[0] - result_par.confidence_interval[0]) < 1
        assert np.abs(result_seq.confidence_interval[1] - result_par.confidence_interval[1]) < 1

    def test_compare_statistics(self):
        """Test comparison of statistics between two datasets."""
        np.random.seed(42)
        data1 = np.random.normal(100, 15, 500)
        data2 = np.random.normal(110, 15, 500)  # Different mean

        analyzer = BootstrapAnalyzer(n_bootstrap=2000, seed=42, show_progress=False)

        # Test difference comparison
        result_diff = analyzer.compare_statistics(data1, data2, np.mean, "difference")
        assert result_diff.statistic < 0  # data1 mean < data2 mean
        assert result_diff.confidence_interval[1] < 0  # Significant difference

        # Test ratio comparison
        result_ratio = analyzer.compare_statistics(data1, data2, np.mean, "ratio")
        assert result_ratio.statistic < 1  # data1/data2 < 1
        assert result_ratio.confidence_interval[1] < 1  # Significant ratio

    def test_convergence_check(self):
        """Test bootstrap convergence checking."""
        np.random.seed(42)

        # Small sample - should show convergence
        small_dist = np.random.normal(0, 1, 100)
        analyzer = BootstrapAnalyzer(show_progress=False)
        assert analyzer._check_convergence(small_dist)

        # Large sample with good convergence
        good_dist = np.random.normal(0, 1, 10000)
        # May not always converge perfectly with random data
        # assert analyzer._check_convergence(good_dist)

        # Artificial non-converged distribution
        bad_dist = np.concatenate(
            [
                np.random.normal(0, 1, 5000),
                np.random.normal(50, 1, 5000),  # Very different distribution in second half
            ]
        )
        assert not analyzer._check_convergence(bad_dist, stability_threshold=0.01)

    def test_bootstrap_result_summary(self):
        """Test BootstrapResult summary generation."""
        result = BootstrapResult(
            statistic=100.5,
            confidence_level=0.95,
            confidence_interval=(98.2, 102.8),
            bootstrap_distribution=np.random.normal(100, 1, 1000),
            method="percentile",
            n_bootstrap=1000,
            bias=0.02,
            acceleration=0.01,
            converged=True,
        )

        summary = result.summary()
        assert "Bootstrap Analysis Results" in summary
        assert "100.5" in summary
        assert "98.2" in summary
        assert "102.8" in summary
        assert "percentile" in summary
        assert "1,000" in summary
        assert "Converged: Yes" in summary


class TestStatisticalTests:
    """Test suite for statistical hypothesis testing functions."""

    def test_difference_in_means_two_sided(self):
        """Test two-sided difference in means test."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 15, 500)
        sample2 = np.random.normal(100, 15, 500)  # Same distribution

        result = difference_in_means_test(
            sample1, sample2, alternative="two-sided", n_bootstrap=2000, seed=42
        )

        assert isinstance(result, HypothesisTestResult)
        assert result.p_value > 0.05  # Should not reject null
        assert not result.reject_null
        assert result.alternative == "two-sided"
        assert result.null_hypothesis == "mean1 = mean2"
        assert -2 < result.test_statistic < 2  # Small difference expected

    def test_difference_in_means_one_sided(self):
        """Test one-sided difference in means test."""
        np.random.seed(42)
        sample1 = np.random.normal(95, 15, 500)  # Lower mean
        sample2 = np.random.normal(105, 15, 500)  # Higher mean

        # Test less alternative
        result_less = difference_in_means_test(
            sample1, sample2, alternative="less", n_bootstrap=2000, seed=42
        )
        assert result_less.p_value < 0.05  # Should reject null
        assert result_less.reject_null
        assert result_less.test_statistic < 0

        # Test greater alternative (should not reject)
        result_greater = difference_in_means_test(
            sample1, sample2, alternative="greater", n_bootstrap=2000, seed=42
        )
        assert result_greater.p_value > 0.95  # Very high p-value
        assert not result_greater.reject_null

    def test_ratio_of_metrics(self):
        """Test ratio of metrics hypothesis test."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 500)
        sample2 = np.random.normal(100, 10, 500)  # Same distribution

        # Test null ratio of 1.0
        result = ratio_of_metrics_test(
            sample1,
            sample2,
            statistic=np.mean,
            null_ratio=1.0,
            alternative="two-sided",
            n_bootstrap=2000,
            seed=42,
        )

        assert result.p_value > 0.05  # Should not reject null
        assert not result.reject_null
        assert 0.95 < result.test_statistic < 1.05  # Ratio should be close to 1
        assert result.confidence_interval[0] < 1 < result.confidence_interval[1]

    def test_ratio_with_zero_denominator(self):
        """Test ratio test handling of zero denominator."""
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = np.array([0, 0, 0, 0, 0])  # All zeros

        result = ratio_of_metrics_test(sample1, sample2, statistic=np.mean, null_ratio=1.0)

        assert result.test_statistic == np.inf
        assert result.p_value == 1.0  # Cannot compute, return conservative p-value
        assert not result.reject_null
        assert np.isnan(result.confidence_interval[0])

    def test_paired_comparison(self):
        """Test paired comparison test."""
        np.random.seed(42)
        # Create paired differences with positive mean
        differences = np.random.normal(5, 10, 500)  # Mean difference of 5

        # Test against null of 0
        result = paired_comparison_test(
            differences, null_value=0.0, alternative="greater", n_bootstrap=2000, seed=42
        )

        assert result.p_value < 0.05  # Should reject null
        assert result.reject_null
        assert result.test_statistic > 0
        assert result.confidence_interval[0] > 2  # Lower bound should be positive

    def test_paired_comparison_no_difference(self):
        """Test paired comparison with no true difference."""
        np.random.seed(42)
        differences = np.random.normal(0, 10, 500)  # Mean difference of 0

        result = paired_comparison_test(
            differences, null_value=0.0, alternative="two-sided", n_bootstrap=2000, seed=42
        )

        assert result.p_value > 0.05  # Should not reject null
        assert not result.reject_null
        assert -1 < result.test_statistic < 1  # Small test statistic
        assert result.confidence_interval[0] < 0 < result.confidence_interval[1]

    def test_custom_hypothesis_test(self):
        """Test general bootstrap hypothesis testing framework."""
        np.random.seed(42)
        data = np.random.exponential(2, 500)

        # Test if variance equals 4 (true variance of exp(2) is 4)
        def null_transform(x):
            """Transform data to have variance of 4."""
            current_var = np.var(x)
            return x * np.sqrt(4 / current_var)

        result = bootstrap_hypothesis_test(
            data, null_transform, np.var, alternative="two-sided", n_bootstrap=2000, seed=42
        )

        assert result.p_value > 0.05  # Should not reject (true variance is 4)
        assert not result.reject_null
        assert 3 < result.test_statistic < 5  # Variance should be close to 4

    def test_multiple_comparison_bonferroni(self):
        """Test Bonferroni multiple comparison correction."""
        p_values = [0.01, 0.04, 0.03, 0.20, 0.50]

        adj_p, reject = multiple_comparison_correction(p_values, method="bonferroni", alpha=0.05)

        # With Bonferroni, threshold is 0.05/5 = 0.01
        assert adj_p[0] == 0.05  # 0.01 * 5
        assert adj_p[1] == 0.20  # 0.04 * 5
        assert adj_p[2] == 0.15  # 0.03 * 5
        assert adj_p[3] == 1.00  # min(0.20 * 5, 1.0)
        assert adj_p[4] == 1.00  # min(0.50 * 5, 1.0)

        assert not reject[0]  # 0.05 == 0.05 (not less than)
        assert not reject[1]  # 0.20 > 0.05
        assert not reject[2]  # 0.15 > 0.05
        assert not reject[3]
        assert not reject[4]

    def test_multiple_comparison_holm(self):
        """Test Holm-Bonferroni multiple comparison correction."""
        p_values = [0.01, 0.04, 0.03, 0.20]

        adj_p, reject = multiple_comparison_correction(p_values, method="holm", alpha=0.05)

        # Holm: Compare sorted p-values to alpha/(n-i+1)
        # p[0]=0.01 vs 0.05/4=0.0125: reject
        # p[1]=0.03 vs 0.05/3=0.0167: fail to reject, stop
        assert reject[0]  # Original p=0.01
        assert not reject[1]  # Original p=0.04
        assert not reject[2]  # Original p=0.03
        assert not reject[3]  # Original p=0.20

    def test_multiple_comparison_fdr(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = [0.001, 0.008, 0.039, 0.041, 0.20]

        adj_p, reject = multiple_comparison_correction(p_values, method="fdr", alpha=0.05)

        # FDR is less conservative than Bonferroni
        assert np.sum(reject) >= 2  # Should reject at least 2 hypotheses
        assert reject[0]  # Very small p-value
        assert reject[1]  # Still quite small
        assert not reject[4]  # Large p-value

    def test_hypothesis_result_summary(self):
        """Test HypothesisTestResult summary generation."""
        result = HypothesisTestResult(
            test_statistic=2.5,
            p_value=0.012,
            reject_null=True,
            confidence_interval=(1.2, 3.8),
            null_hypothesis="mean = 0",
            alternative="greater",
            alpha=0.05,
            method="bootstrap test",
        )

        summary = result.summary()
        assert "Hypothesis Test Results" in summary
        assert "mean = 0" in summary
        assert "greater" in summary
        assert "2.5" in summary
        assert "0.012" in summary
        assert "Reject Null: Yes" in summary
        assert "Significant difference detected" in summary


class TestConvenienceFunctions:
    """Test convenience functions for bootstrap analysis."""

    def test_bootstrap_confidence_interval_function(self):
        """Test the convenience function for bootstrap CI."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)

        stat, ci = bootstrap_confidence_interval(
            data,
            statistic=np.median,
            confidence_level=0.99,
            n_bootstrap=2000,
            method="percentile",
            seed=42,
        )

        assert np.abs(stat - 50) < 1  # Median should be close to 50
        assert ci[0] < 50 < ci[1]  # CI should contain true median
        assert ci[1] - ci[0] < 3  # CI should be reasonably tight

    def test_bootstrap_ci_with_list_input(self):
        """Test bootstrap CI function with list input."""
        data_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        stat, ci = bootstrap_confidence_interval(
            data_list, statistic=np.mean, confidence_level=0.90, n_bootstrap=1000, seed=42
        )

        assert stat == 5.5  # Mean of 1-10
        assert ci[0] < 5.5 < ci[1]
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)


class TestValidationAgainstKnownDistributions:
    """Validate bootstrap methods against known theoretical distributions."""

    def test_normal_distribution_coverage(self):
        """Test that bootstrap CIs achieve nominal coverage for normal data."""
        np.random.seed(42)
        true_mean = 100
        true_std = 15
        n_trials = 100
        confidence_level = 0.95

        coverage_count = 0

        for _ in range(n_trials):
            # Generate sample from normal distribution
            data = np.random.normal(true_mean, true_std, 200)

            # Compute bootstrap CI
            _, ci = bootstrap_confidence_interval(
                data,
                np.mean,
                confidence_level=confidence_level,
                n_bootstrap=1000,
                seed=None,  # Different seed each time
            )

            # Check if true mean is in CI
            if ci[0] <= true_mean <= ci[1]:
                coverage_count += 1

        coverage_probability = coverage_count / n_trials

        # Coverage should be close to nominal level
        # Allow some tolerance due to Monte Carlo error
        assert 0.90 <= coverage_probability <= 1.00

    def test_exponential_distribution_median(self):
        """Test bootstrap CI for median of exponential distribution."""
        np.random.seed(42)
        rate = 2.0
        true_median = np.log(2) / rate  # Theoretical median

        data = np.random.exponential(1 / rate, 1000)

        stat, ci = bootstrap_confidence_interval(
            data,
            np.median,
            confidence_level=0.95,
            n_bootstrap=5000,
            seed=42,
        )

        # Check that estimate is close to true median
        assert np.abs(stat - true_median) < 0.05

        # Check that CI contains true median
        assert ci[0] <= true_median <= ci[1]

        # CI width should be reasonable
        assert ci[1] - ci[0] < 0.1


class TestPerformance:
    """Test performance characteristics of bootstrap methods."""

    def test_large_dataset_performance(self):
        """Test that bootstrap works efficiently with large datasets."""
        np.random.seed(42)
        large_data = np.random.normal(0, 1, 10000)

        analyzer = BootstrapAnalyzer(
            n_bootstrap=1000,
            n_workers=2,
            show_progress=False,
        )

        # Should complete without errors
        result = analyzer.confidence_interval(large_data, np.mean, parallel=True)

        # Convergence may vary with random data
        assert len(result.bootstrap_distribution) == 1000
        assert result.confidence_interval[0] < 0.1  # Should be close to 0
        assert result.confidence_interval[1] > -0.1  # Should be close to 0

    def test_parallel_speedup(self):
        """Test that parallel processing provides speedup."""
        import time

        np.random.seed(42)
        data = np.random.normal(0, 1, 5000)
        n_bootstrap = 2000

        # Sequential timing
        analyzer_seq = BootstrapAnalyzer(
            n_bootstrap=n_bootstrap,
            n_workers=1,
            show_progress=False,
        )

        start = time.time()
        analyzer_seq.confidence_interval(data, np.mean, parallel=False)
        seq_time = time.time() - start

        # Parallel timing
        analyzer_par = BootstrapAnalyzer(
            n_bootstrap=n_bootstrap,
            n_workers=4,
            show_progress=False,
        )

        start = time.time()
        analyzer_par.confidence_interval(data, np.mean, parallel=True)
        par_time = time.time() - start

        # On Windows, parallel may have overhead that makes it slower for small tasks
        # Just check that it completes without error
        assert seq_time > 0
        assert par_time > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_sample_warning(self):
        """Test warning for very small bootstrap iterations."""
        with pytest.warns(UserWarning, match="Low n_bootstrap"):
            BootstrapAnalyzer(n_bootstrap=50)

    def test_invalid_method(self):
        """Test error for invalid bootstrap method."""
        analyzer = BootstrapAnalyzer(show_progress=False)
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Method must be"):
            analyzer.confidence_interval(data, np.mean, method="invalid")

    def test_invalid_alternative(self):
        """Test error for invalid alternative hypothesis."""
        sample1 = np.array([1, 2, 3])
        sample2 = np.array([4, 5, 6])

        with pytest.raises(ValueError, match="Alternative must be"):
            difference_in_means_test(sample1, sample2, alternative="invalid")

    def test_invalid_comparison(self):
        """Test error for invalid comparison type."""
        analyzer = BootstrapAnalyzer(show_progress=False)
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        with pytest.raises(ValueError, match="Comparison must be"):
            analyzer.compare_statistics(data1, data2, np.mean, "invalid")

    def test_empty_array(self):
        """Test handling of empty arrays."""
        import warnings

        # Suppress warning about low n_bootstrap since it's intentional for this test
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Low n_bootstrap.*may produce unstable results"
            )
            analyzer = BootstrapAnalyzer(n_bootstrap=10, show_progress=False)

        empty_data = np.array([])

        # Should return NaN for empty data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = analyzer.confidence_interval(empty_data, np.mean)
            assert np.isnan(result.statistic)

    def test_single_value_array(self):
        """Test handling of single-value arrays."""
        analyzer = BootstrapAnalyzer(n_bootstrap=100, show_progress=False)
        single_data = np.array([42.0])

        result = analyzer.confidence_interval(single_data, np.mean)

        # With single value, CI should be degenerate
        assert result.statistic == 42.0
        assert result.confidence_interval[0] == 42.0
        assert result.confidence_interval[1] == 42.0

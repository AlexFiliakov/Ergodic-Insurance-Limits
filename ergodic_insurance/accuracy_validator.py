"""Numerical accuracy validation for Monte Carlo simulations.

This module provides tools to validate the numerical accuracy of optimized
Monte Carlo simulations against reference implementations, ensuring that
performance optimizations don't compromise result quality.

Key features:
    - High-precision reference implementations
    - Statistical validation of distributions
    - Edge case and boundary condition testing
    - Accuracy comparison metrics
    - Detailed validation reports

Example:
    >>> from accuracy_validator import AccuracyValidator
    >>> import numpy as np
    >>>
    >>> validator = AccuracyValidator()
    >>>
    >>> # Compare optimized vs reference implementation
    >>> optimized_results = np.random.normal(0.08, 0.02, 10000)
    >>> reference_results = np.random.normal(0.08, 0.02, 10000)
    >>>
    >>> validation = validator.compare_implementations(
    ...     optimized_results, reference_results
    ... )
    >>> print(f"Accuracy: {validation.accuracy_score:.4f}")

Google-style docstrings are used throughout for Sphinx documentation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import kstwobign


@dataclass
class ValidationResult:
    """Results from accuracy validation."""

    accuracy_score: float  #: Overall accuracy score (0-1)
    mean_error: float = 0.0  #: Mean absolute error
    max_error: float = 0.0  #: Maximum absolute error
    relative_error: float = 0.0  #: Mean relative error
    ks_statistic: float = 0.0  #: Kolmogorov-Smirnov test statistic
    ks_pvalue: float = 0.0  #: Kolmogorov-Smirnov test p-value
    passed_tests: List[str] = field(default_factory=list)  #: List of passed validation tests
    failed_tests: List[str] = field(default_factory=list)  #: List of failed validation tests
    edge_cases: Dict[str, bool] = field(default_factory=dict)  #: Results from edge case testing

    def is_valid(self, tolerance: float = 0.01) -> bool:
        """Check if validation passes within tolerance.

        Args:
            tolerance: Maximum acceptable relative error.

        Returns:
            True if validation passes.
        """
        return (
            self.relative_error < tolerance
            and len(self.failed_tests) == 0
            and self.accuracy_score > 0.99
        )

    def summary(self) -> str:
        """Generate validation summary.

        Returns:
            Formatted summary string.
        """
        summary = f"Accuracy Validation Summary\n{'='*50}\n"
        summary += f"Accuracy Score: {self.accuracy_score:.4f}\n"
        summary += f"Mean Error: {self.mean_error:.6f}\n"
        summary += f"Max Error: {self.max_error:.6f}\n"
        summary += f"Relative Error: {self.relative_error:.2%}\n"
        summary += f"KS Test: statistic={self.ks_statistic:.4f}, p-value={self.ks_pvalue:.4f}\n"

        summary += "\nValidation Tests:\n"
        summary += f"  Passed: {len(self.passed_tests)}\n"
        summary += f"  Failed: {len(self.failed_tests)}\n"

        if self.failed_tests:
            summary += "\nFailed Tests:\n"
            for test in self.failed_tests[:5]:
                summary += f"  - {test}\n"

        if self.edge_cases:
            summary += "\nEdge Cases:\n"
            for case, passed in self.edge_cases.items():
                status = "✓" if passed else "✗"
                summary += f"  {status} {case}\n"

        return summary


class ReferenceImplementations:
    """High-precision reference implementations for validation.

    These implementations prioritize accuracy over speed and serve
    as the ground truth for validation.
    """

    @staticmethod
    def calculate_growth_rate_precise(
        final_assets: float, initial_assets: float, n_years: float
    ) -> float:
        """Calculate growth rate with high precision.

        Args:
            final_assets: Final asset value.
            initial_assets: Initial asset value.
            n_years: Number of years.

        Returns:
            Precise growth rate.
        """
        if final_assets <= 0 or initial_assets <= 0:
            return -np.inf

        # Use high precision logarithm (float128 not available on Windows, using float64)
        ratio = np.float64(final_assets) / np.float64(initial_assets)
        if ratio <= 0:
            return -np.inf

        return float(np.log(ratio) / n_years)

    @staticmethod
    def apply_insurance_precise(
        loss: float, attachment: float, limit: float
    ) -> Tuple[float, float]:
        """Apply insurance with precise calculations.

        Args:
            loss: Loss amount.
            attachment: Insurance attachment point.
            limit: Insurance limit.

        Returns:
            Tuple of (retained_loss, recovered_amount).
        """
        # Use high precision for edge cases (float128 not available on Windows, using float64)
        loss = np.float64(loss)
        attachment = np.float64(attachment)
        limit = np.float64(limit)

        if loss <= attachment:
            return float(loss), 0.0

        excess = loss - attachment
        recovery = min(float(excess), float(limit))
        retained = loss - recovery

        return float(retained), float(recovery)

    @staticmethod
    def calculate_var_precise(losses: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk with high precision.

        Args:
            losses: Array of loss amounts.
            confidence: Confidence level (e.g., 0.95).

        Returns:
            VaR at specified confidence level.
        """
        if len(losses) == 0:
            return 0.0

        # Use high precision sorting (float128 not available on Windows, using float64)
        sorted_losses = np.sort(losses.astype(np.float64))
        index = int(np.ceil(confidence * len(sorted_losses))) - 1
        index = max(0, min(index, len(sorted_losses) - 1))

        return float(sorted_losses[index])

    @staticmethod
    def calculate_tvar_precise(losses: np.ndarray, confidence: float) -> float:
        """Calculate Tail Value at Risk with high precision.

        Args:
            losses: Array of loss amounts.
            confidence: Confidence level (e.g., 0.95).

        Returns:
            TVaR at specified confidence level.
        """
        if len(losses) == 0:
            return 0.0

        var = ReferenceImplementations.calculate_var_precise(losses, confidence)
        tail_losses = losses[losses >= var]

        if len(tail_losses) == 0:
            return float(var)

        return float(np.mean(tail_losses.astype(np.float64)))

    @staticmethod
    def calculate_ruin_probability_precise(paths: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate ruin probability with high precision.

        Args:
            paths: Array of asset paths.
            threshold: Ruin threshold.

        Returns:
            Probability of ruin.
        """
        if len(paths) == 0:
            return 0.0

        # Check each path precisely
        ruin_count = 0
        for path in paths:
            min_value = np.min(path.astype(np.float64))
            if min_value <= threshold:
                ruin_count += 1

        return ruin_count / len(paths)


class StatisticalValidation:
    """Statistical tests for distribution validation."""

    @staticmethod
    def compare_distributions(data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Compare two distributions statistically.

        Args:
            data1: First dataset.
            data2: Second dataset.

        Returns:
            Dictionary of statistical test results.
        """
        import warnings

        results = {}

        # Kolmogorov-Smirnov test - use asymptotic method for small samples or edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Exact calculation unsuccessful.*")
            # Force asymptotic method for very small samples or when data has many duplicates
            if (
                len(data1) < 10
                or len(data2) < 10
                or len(np.unique(data1)) < 3
                or len(np.unique(data2)) < 3
            ):
                ks_stat, ks_p = stats.ks_2samp(data1, data2, method="asymp")
            else:
                ks_stat, ks_p = stats.ks_2samp(data1, data2)
        results["ks_statistic"] = ks_stat
        results["ks_pvalue"] = ks_p
        results["ks_passes"] = bool(ks_p > 0.05)

        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        results["mw_statistic"] = mw_stat
        results["mw_pvalue"] = mw_p
        results["mw_passes"] = bool(mw_p > 0.05)

        # Compare moments
        results["mean_diff"] = abs(np.mean(data1) - np.mean(data2))
        results["std_diff"] = abs(np.std(data1) - np.std(data2))
        results["skew_diff"] = abs(stats.skew(data1) - stats.skew(data2))
        results["kurtosis_diff"] = abs(stats.kurtosis(data1) - stats.kurtosis(data2))

        # Quantile comparison
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        q1 = np.percentile(data1, [q * 100 for q in quantiles])
        q2 = np.percentile(data2, [q * 100 for q in quantiles])
        results["quantile_errors"] = {
            f"q{int(q*100)}": abs(q1[i] - q2[i]) for i, q in enumerate(quantiles)
        }

        return results

    @staticmethod
    def validate_statistical_properties(
        data: np.ndarray, expected_mean: float, expected_std: float, tolerance: float = 0.05
    ) -> Dict[str, bool]:
        """Validate statistical properties of data.

        Args:
            data: Data to validate.
            expected_mean: Expected mean value.
            expected_std: Expected standard deviation.
            tolerance: Relative tolerance for validation.

        Returns:
            Dictionary of validation results.
        """
        actual_mean = np.mean(data)
        actual_std = np.std(data)

        validations = {
            "mean_valid": bool(
                abs(actual_mean - expected_mean) / (abs(expected_mean) + 1e-10) < tolerance
            ),
            "std_valid": bool(
                abs(actual_std - expected_std) / (abs(expected_std) + 1e-10) < tolerance
            ),
            "normality_test": bool(stats.normaltest(data).pvalue > 0.05)
            if len(data) > 20
            else True,
            "no_outliers": bool(np.sum(np.abs(stats.zscore(data)) > 4) / len(data) < 0.01),
        }

        return validations


class EdgeCaseTester:
    """Test edge cases and boundary conditions."""

    @staticmethod
    def test_extreme_values() -> Dict[str, bool]:
        """Test handling of extreme values.

        Returns:
            Dictionary of test results.
        """
        tests = {}
        ref = ReferenceImplementations()

        # Test zero values
        tests["zero_initial_assets"] = bool(
            ref.calculate_growth_rate_precise(100, 0, 10) == -np.inf
        )
        tests["zero_final_assets"] = bool(ref.calculate_growth_rate_precise(0, 100, 10) == -np.inf)

        # Test infinity
        tests["infinite_loss"] = bool(ref.apply_insurance_precise(np.inf, 1000, 10000)[0] == np.inf)

        # Test negative values
        tests["negative_assets"] = bool(ref.calculate_growth_rate_precise(-100, 100, 10) == -np.inf)

        # Test very large numbers
        large_num = 1e308  # Near float64 max
        retained, recovered = ref.apply_insurance_precise(large_num, 1000, 10000)
        tests["large_number_handling"] = bool(retained > 0 and not np.isnan(retained))

        # Test very small numbers
        small_num = 1e-308  # Near float64 min
        tests["small_number_handling"] = bool(
            ref.apply_insurance_precise(small_num, 0, 1)[1] == small_num
        )

        return tests

    @staticmethod
    def test_boundary_conditions() -> Dict[str, bool]:
        """Test boundary conditions.

        Returns:
            Dictionary of test results.
        """
        tests = {}
        ref = ReferenceImplementations()

        # Test insurance boundaries
        loss = 5000
        attachment = 1000
        limit = 3000
        retained, recovered = ref.apply_insurance_precise(loss, attachment, limit)
        tests["insurance_limit_boundary"] = bool(abs(recovered - limit) < 1e-10)

        # Test exact attachment point
        retained2, recovered2 = ref.apply_insurance_precise(attachment, attachment, limit)
        tests["exact_attachment"] = bool(recovered2 == 0)

        # Test loss below attachment
        retained3, recovered3 = ref.apply_insurance_precise(500, attachment, limit)
        tests["below_attachment"] = bool(recovered3 == 0 and retained3 == 500)

        # Test empty arrays
        tests["empty_var"] = bool(ref.calculate_var_precise(np.array([]), 0.95) == 0)
        tests["empty_tvar"] = bool(ref.calculate_tvar_precise(np.array([]), 0.95) == 0)
        tests["empty_ruin"] = bool(ref.calculate_ruin_probability_precise(np.array([]), 0) == 0)

        return tests


class AccuracyValidator:
    """Main accuracy validation engine.

    Provides comprehensive validation of numerical accuracy for
    Monte Carlo simulations.
    """

    def __init__(self, tolerance: float = 0.01):
        """Initialize accuracy validator.

        Args:
            tolerance: Maximum acceptable relative error.
        """
        self.tolerance = tolerance
        self.reference = ReferenceImplementations()
        self.statistical = StatisticalValidation()
        self.edge_tester = EdgeCaseTester()

    def compare_implementations(
        self,
        optimized_results: np.ndarray,
        reference_results: np.ndarray,
        test_name: str = "Implementation Comparison",
    ) -> ValidationResult:
        """Compare optimized implementation against reference.

        Args:
            optimized_results: Results from optimized implementation.
            reference_results: Results from reference implementation.
            test_name: Name of the test being performed.

        Returns:
            ValidationResult with comparison metrics.
        """
        passed_tests = []
        failed_tests = []

        # Calculate errors
        abs_errors = np.abs(optimized_results - reference_results)
        mean_error = np.mean(abs_errors)
        max_error = np.max(abs_errors)

        # Relative error (avoid division by zero)
        nonzero_mask = reference_results != 0
        if np.any(nonzero_mask):
            relative_errors = abs_errors[nonzero_mask] / np.abs(reference_results[nonzero_mask])
            relative_error = np.mean(relative_errors)
        else:
            relative_error = 0.0

        # Statistical comparison
        stat_results = self.statistical.compare_distributions(optimized_results, reference_results)

        # Build test results
        if relative_error < self.tolerance:
            passed_tests.append(f"{test_name}: Relative error within tolerance")
        else:
            failed_tests.append(
                f"{test_name}: Relative error {relative_error:.4f} exceeds tolerance"
            )

        if stat_results["ks_passes"]:
            passed_tests.append("Kolmogorov-Smirnov test passed")
        else:
            failed_tests.append(
                f"Kolmogorov-Smirnov test failed (p={stat_results['ks_pvalue']:.4f})"
            )

        if stat_results["mean_diff"] < self.tolerance * abs(np.mean(reference_results)):
            passed_tests.append("Mean comparison passed")
        else:
            failed_tests.append(f"Mean difference too large: {stat_results['mean_diff']:.6f}")

        # Calculate accuracy score
        accuracy_score = 1.0 - min(relative_error, 1.0)

        return ValidationResult(
            accuracy_score=accuracy_score,
            mean_error=mean_error,
            max_error=max_error,
            relative_error=relative_error,
            ks_statistic=stat_results["ks_statistic"],
            ks_pvalue=stat_results["ks_pvalue"],
            passed_tests=passed_tests,
            failed_tests=failed_tests,
        )

    def validate_growth_rates(
        self, optimized_func: Callable, test_cases: Optional[List[Tuple]] = None
    ) -> ValidationResult:
        """Validate growth rate calculations.

        Args:
            optimized_func: Optimized growth rate function.
            test_cases: List of (final, initial, years) test cases.

        Returns:
            ValidationResult for growth rate calculations.
        """
        if test_cases is None:
            # Generate default test cases
            rng = np.random.default_rng(42)
            test_cases = [(rng.uniform(5e6, 20e6), 10e6, 10) for _ in range(1000)]

        optimized_results = []
        reference_results = []

        for final, initial, years in test_cases:
            opt_result = optimized_func(final, initial, years)
            ref_result = self.reference.calculate_growth_rate_precise(final, initial, years)

            optimized_results.append(opt_result)
            reference_results.append(ref_result)

        return self.compare_implementations(
            np.array(optimized_results), np.array(reference_results), "Growth Rate Calculation"
        )

    def validate_insurance_calculations(
        self, optimized_func: Callable, test_cases: Optional[List[Tuple]] = None
    ) -> ValidationResult:
        """Validate insurance calculations.

        Args:
            optimized_func: Optimized insurance function.
            test_cases: List of (loss, attachment, limit) test cases.

        Returns:
            ValidationResult for insurance calculations.
        """
        if test_cases is None:
            # Generate test cases
            rng = np.random.default_rng(42)
            test_cases = [(rng.exponential(100000), 50000, 500000) for _ in range(1000)]

        optimized_retained = []
        reference_retained = []

        for loss, attachment, limit in test_cases:
            opt_retained, _ = optimized_func(loss, attachment, limit)
            ref_retained, _ = self.reference.apply_insurance_precise(loss, attachment, limit)

            optimized_retained.append(opt_retained)
            reference_retained.append(ref_retained)

        return self.compare_implementations(
            np.array(optimized_retained), np.array(reference_retained), "Insurance Calculation"
        )

    def validate_risk_metrics(
        self,
        optimized_var: Callable,
        optimized_tvar: Callable,
        test_data: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """Validate risk metric calculations.

        Args:
            optimized_var: Optimized VaR function.
            optimized_tvar: Optimized TVaR function.
            test_data: Test loss data.

        Returns:
            ValidationResult for risk metrics.
        """
        if test_data is None:
            rng = np.random.default_rng(42)
            test_data = rng.lognormal(12, 1.5, 10000)

        confidence_levels = [0.9, 0.95, 0.99]
        passed_tests = []
        failed_tests = []

        total_error = 0
        test_count = 0

        for confidence in confidence_levels:
            # VaR validation
            opt_var = optimized_var(test_data, confidence)
            ref_var = self.reference.calculate_var_precise(test_data, confidence)
            var_error = abs(opt_var - ref_var) / (abs(ref_var) + 1e-10)

            if var_error < self.tolerance:
                passed_tests.append(f"VaR@{confidence:.0%} validation passed")
            else:
                failed_tests.append(f"VaR@{confidence:.0%} error: {var_error:.4f}")

            total_error += var_error
            test_count += 1

            # TVaR validation
            opt_tvar = optimized_tvar(test_data, confidence)
            ref_tvar = self.reference.calculate_tvar_precise(test_data, confidence)
            tvar_error = abs(opt_tvar - ref_tvar) / (abs(ref_tvar) + 1e-10)

            if tvar_error < self.tolerance:
                passed_tests.append(f"TVaR@{confidence:.0%} validation passed")
            else:
                failed_tests.append(f"TVaR@{confidence:.0%} error: {tvar_error:.4f}")

            total_error += tvar_error
            test_count += 1

        avg_error = total_error / test_count
        accuracy_score = 1.0 - min(avg_error, 1.0)

        return ValidationResult(
            accuracy_score=accuracy_score,
            relative_error=avg_error,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
        )

    def run_full_validation(self) -> ValidationResult:
        """Run comprehensive validation suite.

        Returns:
            Complete ValidationResult.
        """
        passed_tests = []
        failed_tests = []

        # Test edge cases
        edge_results = self.edge_tester.test_extreme_values()
        for test_name, passed in edge_results.items():
            if passed:
                passed_tests.append(f"Edge case: {test_name}")
            else:
                failed_tests.append(f"Edge case: {test_name}")

        # Test boundary conditions
        boundary_results = self.edge_tester.test_boundary_conditions()
        for test_name, passed in boundary_results.items():
            if passed:
                passed_tests.append(f"Boundary: {test_name}")
            else:
                failed_tests.append(f"Boundary: {test_name}")

        # Calculate overall accuracy
        total_tests = len(passed_tests) + len(failed_tests)
        accuracy_score = len(passed_tests) / total_tests if total_tests > 0 else 0

        return ValidationResult(
            accuracy_score=accuracy_score,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            edge_cases={**edge_results, **boundary_results},
        )

    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """Generate comprehensive validation report.

        Args:
            results: List of validation results.

        Returns:
            Formatted validation report.
        """
        report = "ACCURACY VALIDATION REPORT\n" + "=" * 60 + "\n\n"

        for i, result in enumerate(results, 1):
            report += f"Test #{i}\n" + "-" * 30 + "\n"
            report += result.summary() + "\n\n"

        # Overall summary
        avg_accuracy = np.mean([r.accuracy_score for r in results])
        total_passed = sum(len(r.passed_tests) for r in results)
        total_failed = sum(len(r.failed_tests) for r in results)

        report += "OVERALL SUMMARY\n" + "=" * 60 + "\n"
        report += f"Average Accuracy Score: {avg_accuracy:.4f}\n"
        report += f"Total Tests Passed: {total_passed}\n"
        report += f"Total Tests Failed: {total_failed}\n"
        report += f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%\n"

        if avg_accuracy >= 0.99 and total_failed == 0:
            report += "\n✓ VALIDATION PASSED: All accuracy requirements met\n"
        else:
            report += "\n✗ VALIDATION FAILED: Accuracy requirements not met\n"

        return report


if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(42)

    # Create validator
    validator = AccuracyValidator(tolerance=0.01)

    # Generate test data
    optimized = rng.normal(0.08, 0.02, 10000)
    reference = optimized + rng.normal(0, 0.0001, 10000)  # Small perturbation

    # Run validation
    result = validator.compare_implementations(optimized, reference)
    print(result.summary())

    # Run full validation suite
    full_result = validator.run_full_validation()
    print("\n" + full_result.summary())

    # Generate report
    report = validator.generate_validation_report([result, full_result])
    print("\n" + report)

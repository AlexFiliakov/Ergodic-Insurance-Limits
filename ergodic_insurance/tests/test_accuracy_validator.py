"""Comprehensive tests for the accuracy_validator module."""

import numpy as np
import pytest
from scipy import stats

from ergodic_insurance.accuracy_validator import (
    AccuracyValidator,
    EdgeCaseTester,
    ReferenceImplementations,
    StatisticalValidation,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult(accuracy_score=0.95)
        assert result.accuracy_score == 0.95
        assert result.mean_error == 0.0
        assert result.max_error == 0.0
        assert result.relative_error == 0.0
        assert result.ks_statistic == 0.0
        assert result.ks_pvalue == 0.0
        assert result.passed_tests == []
        assert result.failed_tests == []
        assert result.edge_cases == {}

    def test_is_valid_success(self):
        """Test is_valid method when validation passes."""
        result = ValidationResult(
            accuracy_score=0.995, relative_error=0.005, passed_tests=["test1"], failed_tests=[]
        )
        assert result.is_valid(tolerance=0.01) is True

    def test_is_valid_failure_relative_error(self):
        """Test is_valid method when relative error exceeds tolerance."""
        result = ValidationResult(
            accuracy_score=0.995, relative_error=0.02, passed_tests=["test1"], failed_tests=[]
        )
        assert result.is_valid(tolerance=0.01) is False

    def test_is_valid_failure_failed_tests(self):
        """Test is_valid method when there are failed tests."""
        result = ValidationResult(
            accuracy_score=0.995,
            relative_error=0.005,
            passed_tests=["test1"],
            failed_tests=["test2"],
        )
        assert result.is_valid(tolerance=0.01) is False

    def test_is_valid_failure_low_accuracy(self):
        """Test is_valid method when accuracy score is too low."""
        result = ValidationResult(
            accuracy_score=0.98, relative_error=0.005, passed_tests=["test1"], failed_tests=[]
        )
        assert result.is_valid(tolerance=0.01) is False

    def test_summary(self):
        """Test summary generation."""
        result = ValidationResult(
            accuracy_score=0.95,
            mean_error=0.001,
            max_error=0.01,
            relative_error=0.02,
            ks_statistic=0.05,
            ks_pvalue=0.1,
            passed_tests=["test1", "test2"],
            failed_tests=["test3"],
            edge_cases={"edge1": True, "edge2": False},
        )
        summary = result.summary()
        assert "Accuracy Score: 0.9500" in summary
        assert "Mean Error: 0.001000" in summary
        assert "Max Error: 0.010000" in summary
        assert "Relative Error: 2.00%" in summary
        assert "KS Test: statistic=0.0500, p-value=0.1000" in summary
        assert "Passed: 2" in summary
        assert "Failed: 1" in summary
        assert "test3" in summary
        assert "✓ edge1" in summary
        assert "✗ edge2" in summary

    def test_summary_many_failed_tests(self):
        """Test summary with more than 5 failed tests."""
        failed_tests = [f"test{i}" for i in range(10)]
        result = ValidationResult(accuracy_score=0.5, failed_tests=failed_tests)
        summary = result.summary()
        # Should only show first 5 failed tests
        assert "test0" in summary
        assert "test4" in summary
        assert "test5" not in summary


class TestReferenceImplementations:
    """Test ReferenceImplementations class."""

    def test_calculate_growth_rate_precise_normal(self):
        """Test precise growth rate calculation for normal cases."""
        ref = ReferenceImplementations()
        rate = ref.calculate_growth_rate_precise(20000000, 10000000, 10)
        expected = np.log(2.0) / 10
        assert np.isclose(rate, expected, rtol=1e-10)

    def test_calculate_growth_rate_precise_zero_final(self):
        """Test growth rate calculation with zero final assets."""
        ref = ReferenceImplementations()
        rate = ref.calculate_growth_rate_precise(0, 10000000, 10)
        assert rate == -np.inf

    def test_calculate_growth_rate_precise_zero_initial(self):
        """Test growth rate calculation with zero initial assets."""
        ref = ReferenceImplementations()
        rate = ref.calculate_growth_rate_precise(10000000, 0, 10)
        assert rate == -np.inf

    def test_calculate_growth_rate_precise_negative_assets(self):
        """Test growth rate calculation with negative assets."""
        ref = ReferenceImplementations()
        rate = ref.calculate_growth_rate_precise(-10000000, 10000000, 10)
        assert rate == -np.inf
        rate = ref.calculate_growth_rate_precise(10000000, -10000000, 10)
        assert rate == -np.inf

    def test_apply_insurance_precise_below_attachment(self):
        """Test insurance application below attachment point."""
        ref = ReferenceImplementations()
        retained, recovered = ref.apply_insurance_precise(1000, 2000, 5000)
        assert retained == 1000
        assert recovered == 0

    def test_apply_insurance_precise_at_attachment(self):
        """Test insurance application at attachment point."""
        ref = ReferenceImplementations()
        retained, recovered = ref.apply_insurance_precise(2000, 2000, 5000)
        assert retained == 2000
        assert recovered == 0

    def test_apply_insurance_precise_within_limit(self):
        """Test insurance application within limit."""
        ref = ReferenceImplementations()
        retained, recovered = ref.apply_insurance_precise(5000, 2000, 5000)
        assert retained == 2000
        assert recovered == 3000

    def test_apply_insurance_precise_exceeds_limit(self):
        """Test insurance application exceeding limit."""
        ref = ReferenceImplementations()
        retained, recovered = ref.apply_insurance_precise(10000, 2000, 5000)
        assert retained == 5000
        assert recovered == 5000

    def test_calculate_var_precise_empty(self):
        """Test VaR calculation with empty array."""
        ref = ReferenceImplementations()
        var = ref.calculate_var_precise(np.array([]), 0.95)
        assert var == 0.0

    def test_calculate_var_precise_normal(self):
        """Test VaR calculation with normal data."""
        ref = ReferenceImplementations()
        losses = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        var = ref.calculate_var_precise(losses, 0.9)
        assert var == 900

    def test_calculate_var_precise_edge_cases(self):
        """Test VaR calculation with edge confidence levels."""
        ref = ReferenceImplementations()
        losses = np.array([100, 200, 300, 400, 500])
        var = ref.calculate_var_precise(losses, 0.0)
        assert var == 100
        var = ref.calculate_var_precise(losses, 1.0)
        assert var == 500

    def test_calculate_tvar_precise_empty(self):
        """Test TVaR calculation with empty array."""
        ref = ReferenceImplementations()
        tvar = ref.calculate_tvar_precise(np.array([]), 0.95)
        assert tvar == 0.0

    def test_calculate_tvar_precise_normal(self):
        """Test TVaR calculation with normal data."""
        ref = ReferenceImplementations()
        losses = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        tvar = ref.calculate_tvar_precise(losses, 0.9)
        expected = np.mean([900, 1000])
        assert np.isclose(tvar, expected)

    def test_calculate_tvar_precise_all_below_var(self):
        """Test TVaR when all losses are below VaR."""
        ref = ReferenceImplementations()
        losses = np.array([100, 100, 100, 100, 100])
        tvar = ref.calculate_tvar_precise(losses, 0.95)
        assert tvar == 100

    def test_calculate_ruin_probability_precise_empty(self):
        """Test ruin probability with empty paths."""
        ref = ReferenceImplementations()
        prob = ref.calculate_ruin_probability_precise(np.array([]), 0.0)
        assert prob == 0.0

    def test_calculate_ruin_probability_precise_no_ruin(self):
        """Test ruin probability with no ruin events."""
        ref = ReferenceImplementations()
        paths = np.array([[100, 110, 120, 130], [100, 105, 115, 125], [100, 102, 108, 118]])
        prob = ref.calculate_ruin_probability_precise(paths, 50)
        assert prob == 0.0

    def test_calculate_ruin_probability_precise_with_ruin(self):
        """Test ruin probability with some ruin events."""
        ref = ReferenceImplementations()
        paths = np.array(
            [
                [100, 110, 120, 130],
                [100, 50, 60, 70],  # Ruin
                [100, 102, 108, 118],
                [100, 0, 10, 20],  # Ruin
            ]
        )
        prob = ref.calculate_ruin_probability_precise(paths, 75)
        assert prob == 0.5


class TestStatisticalValidation:
    """Test StatisticalValidation class."""

    def test_compare_distributions_identical(self):
        """Test comparing identical distributions."""
        val = StatisticalValidation()
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        results = val.compare_distributions(data, data)
        assert results["ks_statistic"] == 0.0
        assert results["ks_pvalue"] == 1.0
        assert results["ks_passes"] is True
        assert results["mean_diff"] < 1e-10
        assert results["std_diff"] < 1e-10

    def test_compare_distributions_different(self):
        """Test comparing different distributions."""
        val = StatisticalValidation()
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(2, 1, 1000)
        results = val.compare_distributions(data1, data2)
        assert results["ks_statistic"] > 0.5
        assert results["ks_pvalue"] < 0.01
        assert results["ks_passes"] is False
        assert results["mean_diff"] > 1.5

    def test_compare_distributions_quantiles(self):
        """Test quantile comparison."""
        val = StatisticalValidation()
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 2, 1000)
        results = val.compare_distributions(data1, data2)
        assert "quantile_errors" in results
        assert "q0010" in results["quantile_errors"]
        assert "q0990" in results["quantile_errors"]
        # Higher quantiles should have larger errors due to different std
        assert results["quantile_errors"]["q0990"] > results["quantile_errors"]["q0500"]

    def test_validate_statistical_properties_valid(self):
        """Test statistical property validation with valid data."""
        val = StatisticalValidation()
        np.random.seed(42)
        data = np.random.normal(10, 2, 10000)
        validations = val.validate_statistical_properties(data, 10, 2, tolerance=0.1)
        assert validations["mean_valid"] is True
        assert validations["std_valid"] is True
        assert validations["no_outliers"] is True

    def test_validate_statistical_properties_invalid_mean(self):
        """Test statistical property validation with invalid mean."""
        val = StatisticalValidation()
        np.random.seed(42)
        data = np.random.normal(15, 2, 10000)
        validations = val.validate_statistical_properties(data, 10, 2, tolerance=0.05)
        assert validations["mean_valid"] is False
        assert validations["std_valid"] is True

    def test_validate_statistical_properties_outliers(self):
        """Test statistical property validation with outliers."""
        val = StatisticalValidation()
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        # Add outliers - need >1% of data to be outliers (z-score > 4)
        # Adding 15 outliers to 1000 points = 1.5% outlier rate
        outliers = [
            100,
            -100,
            150,
            -150,
            200,
            -200,
            120,
            -120,
            180,
            -180,
            140,
            -140,
            160,
            -160,
            130,
        ]
        data = np.append(data, outliers)
        validations = val.validate_statistical_properties(data, 10, 2, tolerance=0.1)
        assert validations["no_outliers"] is False

    def test_validate_statistical_properties_small_sample(self):
        """Test statistical property validation with small sample."""
        val = StatisticalValidation()
        data = np.array([9.8, 10.1, 9.9, 10.2, 10.0])
        validations = val.validate_statistical_properties(data, 10, 0.15, tolerance=0.1)
        # Normality test should be skipped for small samples
        assert validations["normality_test"] is True


class TestEdgeCaseTester:
    """Test EdgeCaseTester class."""

    def test_test_extreme_values(self):
        """Test extreme value handling."""
        tester = EdgeCaseTester()
        tests = tester.test_extreme_values()
        assert tests["zero_initial_assets"] is True
        assert tests["zero_final_assets"] is True
        assert tests["infinite_loss"] is True
        assert tests["negative_assets"] is True
        assert tests["large_number_handling"] is True
        assert tests["small_number_handling"] is True

    def test_test_boundary_conditions(self):
        """Test boundary condition handling."""
        tester = EdgeCaseTester()
        tests = tester.test_boundary_conditions()
        assert tests["insurance_limit_boundary"] is True
        assert tests["exact_attachment"] is True
        assert tests["below_attachment"] is True
        assert tests["empty_var"] is True
        assert tests["empty_tvar"] is True
        assert tests["empty_ruin"] is True


class TestAccuracyValidator:
    """Test AccuracyValidator main class."""

    def test_initialization(self):
        """Test AccuracyValidator initialization."""
        validator = AccuracyValidator(tolerance=0.02)
        assert validator.tolerance == 0.02
        assert validator.reference is not None
        assert validator.statistical is not None
        assert validator.edge_tester is not None

    def test_compare_implementations_identical(self):
        """Test comparing identical implementations."""
        validator = AccuracyValidator()
        np.random.seed(42)
        data = np.random.normal(100, 10, 1000)
        result = validator.compare_implementations(data, data, "Test")
        assert result.accuracy_score == 1.0
        assert result.mean_error == 0.0
        assert result.max_error == 0.0
        assert result.relative_error == 0.0
        assert len(result.passed_tests) > 0
        assert len(result.failed_tests) == 0

    def test_compare_implementations_different(self):
        """Test comparing different implementations."""
        validator = AccuracyValidator(tolerance=0.01)
        np.random.seed(42)
        optimized = np.random.normal(100, 10, 1000)
        reference = optimized * 1.05  # 5% error
        result = validator.compare_implementations(optimized, reference, "Test")
        assert result.accuracy_score < 1.0
        assert result.mean_error > 0
        assert result.max_error > 0
        assert result.relative_error > 0.04
        assert len(result.failed_tests) > 0

    def test_compare_implementations_with_zeros(self):
        """Test comparing implementations with zero values."""
        validator = AccuracyValidator()
        optimized = np.array([0, 0, 0, 100, 200])
        reference = np.array([0, 0, 0, 102, 198])
        result = validator.compare_implementations(optimized, reference)
        assert result.relative_error > 0

    def test_validate_growth_rates(self):
        """Test growth rate validation."""
        validator = AccuracyValidator()

        def optimized_func(final, initial, years):
            return np.log(final / initial) / years

        test_cases = [(20000000, 10000000, 10), (15000000, 10000000, 5)]
        result = validator.validate_growth_rates(optimized_func, test_cases)
        assert result.accuracy_score > 0.99
        assert len(result.passed_tests) > 0

    def test_validate_growth_rates_default(self):
        """Test growth rate validation with default test cases."""
        validator = AccuracyValidator()

        def optimized_func(final, initial, years):
            return np.log(final / initial) / years

        result = validator.validate_growth_rates(optimized_func)
        assert result.accuracy_score > 0.99

    def test_validate_insurance_calculations(self):
        """Test insurance calculation validation."""
        validator = AccuracyValidator()

        def optimized_func(loss, attachment, limit):
            if loss <= attachment:
                return loss, 0
            excess = loss - attachment
            recovery = min(excess, limit)
            retained = loss - recovery
            return retained, recovery

        test_cases = [(100000, 50000, 500000), (25000, 50000, 500000)]
        result = validator.validate_insurance_calculations(optimized_func, test_cases)
        assert result.accuracy_score > 0.99
        assert len(result.passed_tests) > 0

    def test_validate_risk_metrics(self):
        """Test risk metric validation."""
        validator = AccuracyValidator()

        def optimized_var(data, confidence):
            return np.percentile(data, confidence * 100)

        def optimized_tvar(data, confidence):
            var = np.percentile(data, confidence * 100)
            return np.mean(data[data >= var])

        np.random.seed(42)
        test_data = np.random.lognormal(12, 1.5, 1000)
        result = validator.validate_risk_metrics(optimized_var, optimized_tvar, test_data)
        assert result.accuracy_score > 0.98
        assert result.relative_error < 0.02

    def test_validate_risk_metrics_default(self):
        """Test risk metric validation with default data."""
        validator = AccuracyValidator()

        def optimized_var(data, confidence):
            return np.percentile(data, confidence * 100)

        def optimized_tvar(data, confidence):
            var = np.percentile(data, confidence * 100)
            return np.mean(data[data >= var])

        result = validator.validate_risk_metrics(optimized_var, optimized_tvar)
        assert result.accuracy_score > 0.98

    def test_run_full_validation(self):
        """Test full validation suite."""
        validator = AccuracyValidator()
        result = validator.run_full_validation()
        assert result.accuracy_score >= 0.0
        assert len(result.passed_tests) > 0
        assert isinstance(result.edge_cases, dict)

    def test_generate_validation_report(self):
        """Test validation report generation."""
        validator = AccuracyValidator()
        results = [
            ValidationResult(accuracy_score=0.99, passed_tests=["test1", "test2"], failed_tests=[]),
            ValidationResult(accuracy_score=0.95, passed_tests=["test3"], failed_tests=["test4"]),
        ]
        report = validator.generate_validation_report(results)
        assert "ACCURACY VALIDATION REPORT" in report
        assert "Test #1" in report
        assert "Test #2" in report
        assert "OVERALL SUMMARY" in report
        assert "Average Accuracy Score" in report
        assert "Total Tests Passed: 3" in report
        assert "Total Tests Failed: 1" in report
        assert "Success Rate: 75.0%" in report

    def test_generate_validation_report_all_passed(self):
        """Test report generation when all tests pass."""
        validator = AccuracyValidator()
        results = [
            ValidationResult(accuracy_score=0.995, passed_tests=["test1", "test2"], failed_tests=[])
        ]
        report = validator.generate_validation_report(results)
        assert "✓ VALIDATION PASSED" in report

    def test_generate_validation_report_some_failed(self):
        """Test report generation when some tests fail."""
        validator = AccuracyValidator()
        results = [
            ValidationResult(accuracy_score=0.95, passed_tests=["test1"], failed_tests=["test2"])
        ]
        report = validator.generate_validation_report(results)
        assert "✗ VALIDATION FAILED" in report

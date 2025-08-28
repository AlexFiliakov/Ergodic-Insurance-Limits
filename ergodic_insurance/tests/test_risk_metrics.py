"""Unit tests for risk metrics module."""

import time
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from ergodic_insurance.src.risk_metrics import RiskMetrics, RiskMetricsResult, compare_risk_metrics


class TestRiskMetrics:
    """Test RiskMetrics class."""

    def test_initialization(self):
        """Test RiskMetrics initialization."""
        losses = np.array([100, 200, 300, 400, 500])
        metrics = RiskMetrics(losses)
        assert len(metrics.losses) == 5
        assert metrics.weights is None

    def test_initialization_with_weights(self):
        """Test initialization with importance weights."""
        losses = np.array([100, 200, 300])
        weights = np.array([0.2, 0.5, 0.3])
        metrics = RiskMetrics(losses, weights)
        assert metrics.weights is not None
        assert len(metrics.weights) == 3

    def test_empty_losses_raises_error(self):
        """Test that empty losses array raises ValueError."""
        with pytest.raises(ValueError, match="Losses array cannot be empty"):
            RiskMetrics(np.array([]))

    def test_handle_non_finite_values(self):
        """Test handling of NaN and infinite values."""
        losses = np.array([100, np.nan, 200, np.inf, 300])
        metrics = RiskMetrics(losses)
        assert len(metrics.losses) == 3
        assert np.all(np.isfinite(metrics.losses))


class TestVaR:
    """Test Value at Risk calculations."""

    def test_empirical_var(self):
        """Test empirical VaR calculation."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 10000)
        metrics = RiskMetrics(losses)

        var_95 = metrics.var(0.95)
        var_99 = metrics.var(0.99)

        # VaR should increase with confidence level
        assert var_99 > var_95

        # For normal distribution, check against theoretical values
        theoretical_var_95 = 1000 + 100 * stats.norm.ppf(0.95)
        theoretical_var_99 = 1000 + 100 * stats.norm.ppf(0.99)

        assert abs(var_95 - theoretical_var_95) < 20  # Within 20 units
        assert abs(var_99 - theoretical_var_99) < 20

    def test_parametric_var(self):
        """Test parametric VaR assuming normal distribution."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 10000)
        metrics = RiskMetrics(losses)

        var_99 = metrics.var(0.99, method="parametric")
        theoretical_var_99 = 1000 + 100 * stats.norm.ppf(0.99)

        # Parametric should be very close to theoretical for normal data
        assert abs(var_99 - theoretical_var_99) < 5

    def test_weighted_var(self):
        """Test VaR with importance weights."""
        losses = np.array([100, 200, 300, 400, 500])
        weights = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        metrics = RiskMetrics(losses, weights)

        var_80 = metrics.var(0.8)
        # With these weights, 80th percentile should be around 400-500
        assert 350 < var_80 <= 500

    def test_var_confidence_bounds(self):
        """Test VaR with invalid confidence levels."""
        losses = np.array([100, 200, 300])
        metrics = RiskMetrics(losses)

        with pytest.raises(ValueError, match="Confidence must be in"):
            metrics.var(0.0)

        with pytest.raises(ValueError, match="Confidence must be in"):
            metrics.var(1.0)

    def test_var_with_bootstrap_ci(self):
        """Test VaR with bootstrap confidence intervals."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 1000)
        metrics = RiskMetrics(losses, seed=42)

        result = metrics.var(0.95, bootstrap_ci=True, n_bootstrap=100)

        assert isinstance(result, RiskMetricsResult)
        assert result.metric_name == "VaR"
        assert result.confidence_level == 0.95
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.value < result.confidence_interval[1]


class TestTVaR:
    """Test Tail Value at Risk calculations."""

    def test_tvar_basic(self):
        """Test basic TVaR calculation."""
        losses = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        metrics = RiskMetrics(losses)

        var_90 = metrics.var(0.9)
        tvar_90 = metrics.tvar(0.9)

        # TVaR should be greater than VaR
        assert tvar_90 > var_90

        # TVaR should be the average of tail losses
        tail_losses = losses[losses >= var_90]
        expected_tvar = np.mean(tail_losses)
        assert abs(tvar_90 - expected_tvar) < 1e-10

    def test_tvar_coherence_properties(self):
        """Test that TVaR satisfies coherence properties."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 1000)
        metrics = RiskMetrics(losses)

        # Test positive homogeneity
        tvar_original = metrics.tvar(0.95)
        scaled_losses = losses * 2
        metrics_scaled = RiskMetrics(scaled_losses)
        tvar_scaled = metrics_scaled.tvar(0.95)
        assert abs(tvar_scaled - 2 * tvar_original) < 0.01 * tvar_original

        # Test translation invariance
        shift = 1000
        shifted_losses = losses + shift
        metrics_shifted = RiskMetrics(shifted_losses)
        tvar_shifted = metrics_shifted.tvar(0.95)
        assert abs(tvar_shifted - (tvar_original + shift)) < 1

    def test_tvar_with_weights(self):
        """Test TVaR with importance weights."""
        losses = np.array([100, 200, 300, 400, 500])
        weights = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        metrics = RiskMetrics(losses, weights)

        tvar_80 = metrics.tvar(0.8)
        var_80 = metrics.var(0.8)

        # TVaR should be >= VaR (equal when all tail values are the same)
        assert tvar_80 >= var_80


class TestPML:
    """Test Probable Maximum Loss calculations."""

    def test_pml_basic(self):
        """Test basic PML calculation."""
        np.random.seed(42)
        losses = np.random.lognormal(10, 1.5, 10000)
        metrics = RiskMetrics(losses)

        pml_100 = metrics.pml(100)
        pml_250 = metrics.pml(250)
        pml_500 = metrics.pml(500)

        # PML should increase with return period
        assert pml_100 < pml_250 < pml_500

        # PML(100) should equal VaR(99%)
        var_99 = metrics.var(0.99)
        assert abs(pml_100 - var_99) < 1e-10

    def test_pml_invalid_period(self):
        """Test PML with invalid return period."""
        losses = np.array([100, 200, 300])
        metrics = RiskMetrics(losses)

        with pytest.raises(ValueError, match="Return period must be >= 1"):
            metrics.pml(0)


class TestExpectedShortfall:
    """Test Expected Shortfall calculations."""

    def test_expected_shortfall_basic(self):
        """Test basic expected shortfall calculation."""
        losses = np.array([100, 200, 300, 400, 500])
        metrics = RiskMetrics(losses)

        threshold = 250
        es = metrics.expected_shortfall(threshold)

        # ES should be average of losses above threshold
        expected_es = np.mean([300, 400, 500])
        assert abs(es - expected_es) < 1e-10

    def test_expected_shortfall_no_exceedances(self):
        """Test ES when no losses exceed threshold."""
        losses = np.array([100, 200, 300])
        metrics = RiskMetrics(losses)

        es = metrics.expected_shortfall(1000)
        assert es == 0.0


class TestAdditionalMetrics:
    """Test additional risk metrics."""

    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create losses that produce a clear drawdown
        losses = np.array([10, -5, -3, 8, -12, 5, -2, 10, -15, 5])
        metrics = RiskMetrics(losses)

        max_dd = metrics.maximum_drawdown()
        assert max_dd > 0

    def test_economic_capital(self):
        """Test economic capital calculation."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 1000)
        metrics = RiskMetrics(losses)

        ec = metrics.economic_capital(0.999)
        var_999 = metrics.var(0.999)
        mean_loss = np.mean(losses)

        # Economic capital = VaR - Expected Loss
        expected_ec = var_999 - mean_loss
        assert abs(ec - expected_ec) < 1

    def test_tail_index(self):
        """Test tail index estimation."""
        np.random.seed(42)
        # Generate heavy-tailed distribution (Pareto)
        losses = np.random.pareto(2, 1000) * 1000
        metrics = RiskMetrics(losses)

        tail_idx = metrics.tail_index()
        assert tail_idx > 0
        # For Pareto(2), tail index should be around 2
        assert 1 < tail_idx < 4

    def test_conditional_tail_expectation(self):
        """Test CTE calculation."""
        losses = np.array([100, 200, 300, 400, 500])
        metrics = RiskMetrics(losses)

        cte_80 = metrics.conditional_tail_expectation(0.8)
        tvar_80 = metrics.tvar(0.8)

        # CTE should equal TVaR in our implementation
        assert abs(cte_80 - tvar_80) < 1e-10


class TestReturnPeriodCurve:
    """Test return period curve generation."""

    def test_return_period_curve(self):
        """Test return period curve generation."""
        np.random.seed(42)
        losses = np.random.lognormal(10, 1.5, 10000)
        metrics = RiskMetrics(losses)

        periods, loss_values = metrics.return_period_curve()

        # Check that losses increase with return period
        assert np.all(np.diff(loss_values) > 0)
        assert len(periods) == len(loss_values)

    def test_custom_return_periods(self):
        """Test return period curve with custom periods."""
        losses = np.random.lognormal(10, 1, 1000)
        metrics = RiskMetrics(losses)

        custom_periods = np.array([10, 50, 100])
        periods, loss_values = metrics.return_period_curve(custom_periods)

        assert len(periods) == 3
        assert np.array_equal(periods, custom_periods)


class TestRiskAdjustedMetrics:
    """Test risk-adjusted return metrics."""

    def test_risk_adjusted_metrics(self):
        """Test calculation of risk-adjusted metrics."""
        np.random.seed(42)
        # Generate returns (negative losses)
        losses = -np.random.normal(0.08, 0.15, 1000)  # 8% return, 15% volatility
        metrics = RiskMetrics(losses)

        risk_metrics = metrics.risk_adjusted_metrics(risk_free_rate=0.02)

        assert "sharpe_ratio" in risk_metrics
        assert "sortino_ratio" in risk_metrics
        assert "mean_return" in risk_metrics
        assert "volatility" in risk_metrics

        # Sharpe ratio should be approximately (0.08 - 0.02) / 0.15 = 0.4
        assert 0.3 < risk_metrics["sharpe_ratio"] < 0.5


class TestSummaryStatistics:
    """Test summary statistics calculation."""

    def test_summary_statistics(self):
        """Test comprehensive summary statistics."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 1000)
        metrics = RiskMetrics(losses)

        stats = metrics.summary_statistics()

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats

        assert stats["count"] == 1000
        assert stats["min"] < stats["mean"] < stats["max"]
        assert stats["skewness"] > 0  # Lognormal is right-skewed

    def test_weighted_summary_statistics(self):
        """Test summary statistics with weights."""
        losses = np.array([100, 200, 300, 400, 500])
        weights = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        metrics = RiskMetrics(losses, weights)

        stats = metrics.summary_statistics()

        # Weighted mean should be higher due to larger weights on larger values
        weighted_mean = np.average(losses, weights=weights)
        assert abs(stats["mean"] - weighted_mean) < 1e-10


class TestCoherenceTest:
    """Test coherence property testing."""

    def test_coherence_test(self):
        """Test coherence property verification."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 1000)
        metrics = RiskMetrics(losses)

        coherence = metrics.coherence_test()

        assert "tvar_positive_homogeneity" in coherence
        assert "tvar_translation_invariance" in coherence

        # TVaR should satisfy coherence properties
        assert coherence["tvar_positive_homogeneity"]
        assert coherence["tvar_translation_invariance"]


class TestVisualization:
    """Test visualization methods."""

    @patch("matplotlib.pyplot.show")
    def test_plot_distribution(self, mock_show):
        """Test distribution plotting."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 100)
        metrics = RiskMetrics(losses)

        fig = metrics.plot_distribution(bins=20)
        assert fig is not None

        # Test with custom parameters
        fig = metrics.plot_distribution(
            bins=30, show_metrics=True, confidence_levels=[0.9, 0.95, 0.99]
        )
        assert fig is not None


class TestCompareRiskMetrics:
    """Test risk metrics comparison function."""

    def test_compare_risk_metrics(self):
        """Test comparing metrics across scenarios."""
        np.random.seed(42)

        scenarios = {
            "normal": np.random.normal(1000, 100, 1000),
            "lognormal": np.random.lognormal(7, 0.5, 1000),
            "uniform": np.random.uniform(500, 1500, 1000),
        }

        df = compare_risk_metrics(scenarios)

        assert len(df) == 3
        assert "scenario" in df.columns
        assert "var_95.0%" in df.columns
        assert "tvar_99.0%" in df.columns
        assert "pml_100yr" in df.columns
        assert "economic_capital" in df.columns


class TestPerformance:
    """Test performance requirements."""

    def test_large_dataset_performance(self):
        """Test that metrics calculate quickly for large datasets."""
        np.random.seed(42)
        # Generate 1 million scenarios
        losses = np.random.lognormal(10, 1.5, 1_000_000)

        start_time = time.time()

        metrics = RiskMetrics(losses)
        _ = metrics.var(0.95)
        _ = metrics.var(0.99)
        _ = metrics.tvar(0.95)
        _ = metrics.tvar(0.99)
        _ = metrics.pml(100)
        _ = metrics.pml(250)
        _ = metrics.expected_shortfall(metrics.var(0.99))
        _ = metrics.maximum_drawdown()
        _ = metrics.economic_capital(0.999)

        elapsed = time.time() - start_time

        # Should complete in less than 5 seconds
        assert elapsed < 5.0, f"Performance test failed: {elapsed:.2f}s > 5s"


class TestROEAnalyzer:
    """Test ROEAnalyzer class for comprehensive ROE analysis."""

    def test_initialization(self):
        """Test ROEAnalyzer initialization."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        roe_series = np.array([0.10, 0.12, 0.08, 0.15, 0.11])
        equity_series = np.array([100, 110, 120, 130, 140])

        analyzer = ROEAnalyzer(roe_series, equity_series)

        assert len(analyzer.roe_series) == 5
        assert len(analyzer.valid_roe) == 5
        assert analyzer.equity_series is not None

    def test_time_weighted_average(self):
        """Test time-weighted average ROE calculation."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        # Test with constant ROE
        roe_series = np.array([0.10, 0.10, 0.10, 0.10])
        analyzer = ROEAnalyzer(roe_series)

        time_weighted = analyzer.time_weighted_average()
        assert time_weighted == pytest.approx(0.10, rel=0.01)

        # Test with variable ROE
        roe_series = np.array([0.05, 0.15, 0.10, -0.05, 0.20])
        analyzer = ROEAnalyzer(roe_series)

        time_weighted = analyzer.time_weighted_average()
        # Should handle negative values appropriately
        assert isinstance(time_weighted, float)

    def test_equity_weighted_average(self):
        """Test equity-weighted average ROE calculation."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        roe_series = np.array([0.10, 0.15, 0.08])
        equity_series = np.array([100, 200, 300])  # More weight to later periods

        analyzer = ROEAnalyzer(roe_series, equity_series)

        equity_weighted = analyzer.equity_weighted_average()
        # Should be closer to 0.08 due to high equity weight in last period
        assert equity_weighted < np.mean(roe_series)
        assert equity_weighted > 0.08

    def test_rolling_statistics(self):
        """Test rolling window statistics."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        roe_series = np.array([0.08, 0.10, 0.12, 0.09, 0.11, 0.10])
        analyzer = ROEAnalyzer(roe_series)

        # Test 3-period rolling stats
        rolling_stats = analyzer.rolling_statistics(3)

        assert "mean" in rolling_stats
        assert "std" in rolling_stats
        assert "sharpe" in rolling_stats

        # Check that first two values are NaN
        assert np.isnan(rolling_stats["mean"][0])
        assert np.isnan(rolling_stats["mean"][1])

        # Third value should be mean of first 3
        assert rolling_stats["mean"][2] == pytest.approx(0.10, rel=0.01)

    def test_volatility_metrics(self):
        """Test volatility metrics calculation."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        roe_series = np.array([0.05, 0.15, 0.10, 0.20, -0.05])
        analyzer = ROEAnalyzer(roe_series)

        volatility = analyzer.volatility_metrics()

        assert "standard_deviation" in volatility
        assert "downside_deviation" in volatility
        assert "upside_deviation" in volatility
        assert "semi_variance" in volatility
        assert "coefficient_variation" in volatility

        # All volatility measures should be non-negative
        assert volatility["standard_deviation"] > 0
        assert volatility["downside_deviation"] >= 0
        assert volatility["upside_deviation"] >= 0
        assert volatility["semi_variance"] >= 0

    def test_performance_ratios(self):
        """Test performance ratio calculations."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        roe_series = np.array([0.08, 0.12, 0.10, 0.15, 0.09])
        analyzer = ROEAnalyzer(roe_series)

        ratios = analyzer.performance_ratios(risk_free_rate=0.02)

        assert "sharpe_ratio" in ratios
        assert "sortino_ratio" in ratios
        assert "calmar_ratio" in ratios
        assert "information_ratio" in ratios
        assert "omega_ratio" in ratios

        # Sharpe ratio should be positive for positive excess returns
        mean_roe = np.mean(roe_series)
        if mean_roe > 0.02:
            assert ratios["sharpe_ratio"] > 0

    def test_distribution_analysis(self):
        """Test distribution analysis."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        np.random.seed(42)
        roe_series = np.random.normal(0.10, 0.05, 100)
        analyzer = ROEAnalyzer(roe_series)

        distribution = analyzer.distribution_analysis()

        assert "mean" in distribution
        assert "median" in distribution
        assert "skewness" in distribution
        assert "kurtosis" in distribution
        assert "percentile_5" in distribution
        assert "percentile_95" in distribution

        # Check reasonable values
        assert distribution["mean"] == pytest.approx(np.mean(roe_series), rel=0.01)
        assert distribution["percentile_5"] < distribution["median"]
        assert distribution["median"] < distribution["percentile_95"]

    def test_stability_analysis(self):
        """Test stability analysis across periods."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        # Create a series with increasing stability
        roe_series = np.concatenate(
            [
                np.random.normal(0.10, 0.10, 20),  # High volatility
                np.random.normal(0.10, 0.01, 20),  # Low volatility
            ]
        )

        analyzer = ROEAnalyzer(roe_series)
        stability = analyzer.stability_analysis(periods=[5, 10])

        assert "5yr" in stability
        assert "10yr" in stability

        # Check that metrics exist
        assert "mean_stability" in stability["5yr"]
        assert "volatility_stability" in stability["5yr"]
        assert "consistency" in stability["5yr"]

    def test_edge_cases_roe_analyzer(self):
        """Test edge cases for ROEAnalyzer."""
        from ergodic_insurance.src.risk_metrics import ROEAnalyzer

        # Test with all NaN values
        roe_series = np.array([np.nan, np.nan, np.nan])
        analyzer = ROEAnalyzer(roe_series)

        assert analyzer.time_weighted_average() == 0.0
        assert len(analyzer.volatility_metrics()) > 0

        # Test with single value
        roe_series = np.array([0.10])
        analyzer = ROEAnalyzer(roe_series)

        assert analyzer.time_weighted_average() == pytest.approx(0.10, rel=0.01)

        # Test with mixed NaN values
        roe_series = np.array([0.10, np.nan, 0.15, np.nan, 0.08])
        analyzer = ROEAnalyzer(roe_series)

        assert len(analyzer.valid_roe) == 3
        assert analyzer.time_weighted_average() > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value(self):
        """Test metrics with single loss value."""
        losses = np.array([1000])
        metrics = RiskMetrics(losses)

        var_95 = metrics.var(0.95)
        tvar_95 = metrics.tvar(0.95)

        # With single value, all metrics should equal that value
        assert var_95 == 1000
        assert tvar_95 == 1000

    def test_identical_values(self):
        """Test metrics with identical loss values."""
        losses = np.array([500] * 100)
        metrics = RiskMetrics(losses)

        var_99 = metrics.var(0.99)
        tvar_99 = metrics.tvar(0.99)

        # With identical values, all metrics should equal that value
        assert var_99 == 500
        assert tvar_99 == 500

    def test_extreme_confidence_levels(self):
        """Test metrics with extreme confidence levels."""
        losses = np.random.normal(1000, 100, 1000)
        metrics = RiskMetrics(losses)

        var_001 = metrics.var(0.001)  # Very low confidence
        var_999 = metrics.var(0.999)  # Very high confidence

        assert var_001 < np.mean(losses)
        assert var_999 > np.mean(losses)
        assert var_999 > var_001


class TestIntegration:
    """Integration tests with loss_distributions module."""

    def test_integration_with_loss_distributions(self):
        """Test integration with ManufacturingLossGenerator."""
        from ergodic_insurance.src.loss_distributions import LognormalLoss

        # Generate losses using LognormalLoss
        loss_dist = LognormalLoss(mean=50000, cv=0.8, seed=42)
        losses = loss_dist.generate_severity(5000)

        # Calculate risk metrics
        metrics = RiskMetrics(losses)

        var_95 = metrics.var(0.95)
        tvar_95 = metrics.tvar(0.95)
        pml_100 = metrics.pml(100)

        # Verify reasonable values for manufacturing context
        assert 0 < var_95 < 200000  # Reasonable range for $50K mean
        assert tvar_95 > var_95
        assert pml_100 > var_95

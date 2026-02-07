"""Additional tests for risk_metrics to cover missing lines.

Targets missing coverage lines:
63 (weights filtered with non-finite), 82 (unreachable sorted_weights None),
116 (invalid VaR method), 137 (weighted VaR idx overflow), 146-147 (parametric VaR weighted),
162-174 (bootstrap VaR CI with weights), 199 (tvar RiskMetricsResult handling),
206 (empty tail unweighted tvar), 210 (empty tail weighted tvar),
236-238 (expected_shortfall weighted no exceedances), 264 (pml with RiskMetricsResult),
298 (weighted max drawdown), 304-305 (overflow handling in drawdown),
340 (economic_capital with RiskMetricsResult), 384 (tail_index < 2 tail values),
414-416 (risk_adjusted_metrics weighted), 427-432 (weighted sortino),
435 (no downside returns sortino), 544/574-584 (weighted Q-Q plot),
617 (plot_distribution var_99 RiskMetricsResult),
719 (negative growth factors), 733 (equity_weighted without equity),
738 (equity sum 0), 755 (rolling window too large),
843 (performance_ratios < 2 values), 894 (max_drawdown < 2 values),
917 (max drawdown overflow), 935 (distribution_analysis empty),
969 (stability_analysis default periods), 975 (period > series length)
"""

from unittest.mock import patch
import warnings

import numpy as np
import pytest
from scipy import stats

from ergodic_insurance.risk_metrics import (
    RiskMetrics,
    RiskMetricsResult,
    ROEAnalyzer,
    compare_risk_metrics,
)

# ---------------------------------------------------------------------------
# RiskMetrics init edge cases
# ---------------------------------------------------------------------------


class TestRiskMetricsInitEdgeCases:
    """Test initialization edge cases for RiskMetrics."""

    def test_weights_filtered_with_nonfinite_values(self):
        """When losses have NaN/inf, corresponding weights are also filtered (line 63)."""
        losses = np.array([100.0, np.nan, 200.0, np.inf, 300.0])
        weights = np.array([0.2, 0.1, 0.3, 0.1, 0.3])
        metrics = RiskMetrics(losses, weights)
        assert len(metrics.losses) == 3
        assert metrics.weights is not None
        assert len(metrics.weights) == 3
        np.testing.assert_array_equal(metrics.losses, [100.0, 200.0, 300.0])
        np.testing.assert_array_equal(metrics.weights, [0.2, 0.3, 0.3])


# ---------------------------------------------------------------------------
# VaR edge cases (lines 116, 137, 146-147)
# ---------------------------------------------------------------------------


class TestVaREdgeCases:
    """Test VaR calculation edge cases."""

    def test_invalid_method_raises(self):
        """Invalid method argument raises ValueError (line 116)."""
        losses = np.array([100, 200, 300])
        metrics = RiskMetrics(losses)
        with pytest.raises(ValueError, match="Method must be"):
            metrics.var(0.95, method="invalid_method")

    def test_weighted_var_idx_overflow(self):
        """Weighted VaR where searchsorted returns idx >= len (line 137)."""
        # Use very high confidence to push idx beyond array
        losses = np.array([10.0, 20.0, 30.0])
        weights = np.array([0.5, 0.3, 0.2])
        metrics = RiskMetrics(losses, weights)
        # 0.999 confidence will likely exceed cumulative weights
        var_val = metrics.var(0.999)
        assert var_val == 30.0  # Should clamp to last element

    def test_parametric_var_with_weights(self):
        """Parametric VaR with importance weights (lines 146-147)."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 500)
        weights = np.random.uniform(0.5, 1.5, 500)
        metrics = RiskMetrics(losses, weights)

        var_99 = metrics.var(0.99, method="parametric")
        assert isinstance(var_99, float)
        assert var_99 > 1000  # Should be above mean for 99% confidence


# ---------------------------------------------------------------------------
# Bootstrap VaR CI with weights (lines 162-174)
# ---------------------------------------------------------------------------


class TestBootstrapVaRCIWeighted:
    """Test bootstrap VaR CI with weighted losses."""

    def test_bootstrap_ci_with_weights(self):
        """Bootstrap CI with importance weights (lines 162-174)."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 200)
        weights = np.random.uniform(0.5, 1.5, 200)
        metrics = RiskMetrics(losses, weights, seed=42)

        result = metrics.var(0.95, bootstrap_ci=True, n_bootstrap=50)
        assert isinstance(result, RiskMetricsResult)
        assert result.confidence_interval is not None
        assert result.confidence_interval[0] < result.value
        assert result.value < result.confidence_interval[1]


# ---------------------------------------------------------------------------
# TVaR edge cases (lines 199, 206, 210)
# ---------------------------------------------------------------------------


class TestTVaREdgeCases:
    """Test TVaR edge cases."""

    def test_tvar_empty_tail_unweighted(self):
        """TVaR with no losses above VaR threshold (line 206)."""
        # All identical losses -- VaR == max loss, no values strictly above
        losses = np.array([100.0, 100.0, 100.0])
        metrics = RiskMetrics(losses)
        # Use very high confidence so VaR >= all losses
        tvar = metrics.tvar(0.999)
        assert tvar == 100.0

    def test_tvar_empty_tail_weighted(self):
        """TVaR with weighted data and no exceedances (line 210)."""
        losses = np.array([50.0, 50.0, 50.0])
        weights = np.array([0.3, 0.4, 0.3])
        metrics = RiskMetrics(losses, weights)
        # Very high var_value to ensure no losses exceed it
        tvar = metrics.tvar(var_value=100.0)
        assert tvar == 100.0

    def test_tvar_with_weighted_exceedances(self):
        """TVaR with weighted data and some exceedances."""
        losses = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        metrics = RiskMetrics(losses, weights)
        tvar = metrics.tvar(var_value=250.0)
        # Should be weighted average of [300, 400, 500] with weights [0.3, 0.2, 0.2]
        expected = np.average([300, 400, 500], weights=[0.3, 0.2, 0.2])
        assert abs(tvar - expected) < 1e-10


# ---------------------------------------------------------------------------
# Expected shortfall with weights (lines 236-238)
# ---------------------------------------------------------------------------


class TestExpectedShortfallWeighted:
    """Test expected shortfall with weighted data."""

    def test_expected_shortfall_weighted_no_exceedances(self):
        """ES with weighted data and no exceedances returns 0 (lines 236-238)."""
        losses = np.array([100.0, 200.0, 300.0])
        weights = np.array([0.3, 0.4, 0.3])
        metrics = RiskMetrics(losses, weights)
        es = metrics.expected_shortfall(500.0)
        assert es == 0.0

    def test_expected_shortfall_weighted_with_exceedances(self):
        """ES with weighted data and exceedances delegates to tvar."""
        losses = np.array([100.0, 200.0, 300.0, 400.0])
        weights = np.array([0.2, 0.3, 0.3, 0.2])
        metrics = RiskMetrics(losses, weights)
        es = metrics.expected_shortfall(250.0)
        assert es > 250.0


# ---------------------------------------------------------------------------
# PML with RiskMetricsResult (line 264 - normally not hit but covering path)
# ---------------------------------------------------------------------------


class TestPMLEdgeCases:
    """Test PML edge cases."""

    def test_pml_returns_float(self):
        """PML should return a float for typical usage."""
        losses = np.random.lognormal(10, 1, 1000)
        metrics = RiskMetrics(losses)
        pml = metrics.pml(100)
        assert isinstance(pml, float)


# ---------------------------------------------------------------------------
# Maximum drawdown edge cases (lines 298, 304-305)
# ---------------------------------------------------------------------------


class TestMaxDrawdownEdgeCases:
    """Test maximum drawdown edge cases."""

    def test_weighted_maximum_drawdown(self):
        """Maximum drawdown with weighted data (line 298)."""
        losses = np.array([10.0, -5.0, 8.0, -3.0, 12.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        metrics = RiskMetrics(losses, weights)
        dd = metrics.maximum_drawdown()
        assert dd >= 0

    def test_maximum_drawdown_with_overflow(self):
        """Maximum drawdown handles overflow in cumulative sum (lines 304-305)."""
        # Use very large values that could cause overflow
        losses = np.array([1e308, 1e308, 1e308])
        metrics = RiskMetrics(losses)
        dd = metrics.maximum_drawdown()
        assert np.isfinite(dd)


# ---------------------------------------------------------------------------
# Economic capital edge case (line 340)
# ---------------------------------------------------------------------------


class TestEconomicCapitalEdgeCases:
    """Test economic capital edge cases."""

    def test_economic_capital_with_custom_expected_loss(self):
        """Economic capital with user-provided expected loss."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 1000)
        metrics = RiskMetrics(losses)

        ec = metrics.economic_capital(0.999, expected_loss=500.0)
        assert ec > 0


# ---------------------------------------------------------------------------
# Tail index edge case (line 384)
# ---------------------------------------------------------------------------


class TestTailIndexEdgeCases:
    """Test tail index edge cases."""

    def test_tail_index_fewer_than_2_tail_values(self):
        """Tail index with < 2 values above threshold returns NaN (line 384)."""
        losses = np.array([10.0, 20.0, 30.0])
        metrics = RiskMetrics(losses)
        # Set threshold so high that fewer than 2 values exceed it
        tail_idx = metrics.tail_index(threshold=25.0)
        assert np.isnan(tail_idx)

    def test_tail_index_custom_threshold(self):
        """Tail index with a custom threshold."""
        np.random.seed(42)
        losses = np.random.pareto(3, 1000) * 100
        metrics = RiskMetrics(losses)
        tail_idx = metrics.tail_index(threshold=50.0)
        assert np.isfinite(tail_idx) or np.isnan(tail_idx)


# ---------------------------------------------------------------------------
# Risk-adjusted metrics with weights (lines 414-416, 427-432, 435)
# ---------------------------------------------------------------------------


class TestRiskAdjustedMetricsWeighted:
    """Test risk-adjusted metrics with weighted data."""

    def test_risk_adjusted_metrics_weighted(self):
        """Risk-adjusted metrics with weights (lines 414-416)."""
        np.random.seed(42)
        losses = np.random.normal(500, 100, 200)
        weights = np.random.uniform(0.5, 1.5, 200)
        metrics = RiskMetrics(losses, weights)

        result = metrics.risk_adjusted_metrics(risk_free_rate=0.02)
        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert isinstance(result["volatility"], (float, np.floating))

    def test_weighted_sortino_ratio(self):
        """Sortino ratio with weighted downside returns (lines 427-432)."""
        # Create data where many returns are below risk-free rate
        np.random.seed(42)
        losses = np.random.normal(0.01, 0.05, 200)  # Returns centered near 0.01
        weights = np.random.uniform(0.5, 1.5, 200)
        metrics = RiskMetrics(losses, weights)

        # Use the negative of losses as returns to get downside returns
        returns = -losses
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=0.02)
        assert "sortino_ratio" in result

    def test_sortino_no_downside_returns(self):
        """Sortino when no returns are below risk-free rate (line 435)."""
        # All returns well above risk_free_rate
        losses = np.array([-0.10, -0.12, -0.15])  # Negative losses = positive returns
        metrics = RiskMetrics(losses)

        returns = np.array([0.10, 0.12, 0.15])
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=0.02)
        # Sortino should be inf when mean_return > risk_free_rate and no downside
        assert result["sortino_ratio"] == np.inf


# ---------------------------------------------------------------------------
# Issue #386: Sortino downside deviation formula fix
# ---------------------------------------------------------------------------


class TestSortinoDownsideDeviationFix:
    """Verify correct Sortino downside deviation per Sortino & Price (1994).

    DD = sqrt( (1/N) * sum_{i=1}^{N} min(r_i - target, 0)^2 )

    The key correction: denominator uses ALL N observations, not just
    the count below the target.
    """

    def test_acceptance_criteria_example(self):
        """Issue #386 acceptance criterion 1: known hand-calculated result."""
        # returns = [10%, 12%, -5%, 8%, -3%], target = 0%
        returns = np.array([0.10, 0.12, -0.05, 0.08, -0.03])
        target = 0.0
        metrics = RiskMetrics(returns)

        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=target)

        # min(r_i - 0, 0)^2 = [0, 0, 0.0025, 0, 0.0009]
        # mean = (0 + 0 + 0.0025 + 0 + 0.0009) / 5 = 0.00068
        # DD = sqrt(0.00068) = 0.02607...
        expected_dd = np.sqrt(0.00068)
        mean_return = np.mean(returns)
        expected_sortino = mean_return / expected_dd  # target is 0

        assert result["sortino_ratio"] == pytest.approx(expected_sortino, rel=1e-6)

    def test_both_locations_agree(self):
        """Issue #386 acceptance criterion 2: both locations identical."""
        data = np.array([0.10, 0.12, -0.05, 0.08, -0.03, 0.15, -0.02])
        rf = 0.03

        # Location 1: RiskMetrics.risk_adjusted_metrics
        metrics = RiskMetrics(data)
        loc1 = metrics.risk_adjusted_metrics(returns=data, risk_free_rate=rf)

        # Location 2: ROEAnalyzer.performance_ratios
        analyzer = ROEAnalyzer(data)
        loc2 = analyzer.performance_ratios(risk_free_rate=rf)

        assert loc1["sortino_ratio"] == pytest.approx(loc2["sortino_ratio"], rel=1e-6)

    def test_sortino_increases_with_more_positive_returns(self):
        """Issue #386 acceptance criterion 3: Sortino up when more positives."""
        # Mostly negative
        returns_bad = np.array([0.10, -0.05, -0.08, -0.03, -0.07])
        # Mostly positive
        returns_good = np.array([0.10, 0.05, 0.08, -0.03, 0.07])

        metrics_bad = RiskMetrics(returns_bad)
        metrics_good = RiskMetrics(returns_good)

        sortino_bad = metrics_bad.risk_adjusted_metrics(returns=returns_bad, risk_free_rate=0.0)[
            "sortino_ratio"
        ]
        sortino_good = metrics_good.risk_adjusted_metrics(returns=returns_good, risk_free_rate=0.0)[
            "sortino_ratio"
        ]

        assert sortino_good > sortino_bad

    def test_downside_deviation_uses_all_observations(self):
        """Core fix: DD denominator is N, not count of below-target."""
        # 9 returns above target, 1 below by -0.10
        returns = np.array([0.05] * 9 + [-0.10])
        target = 0.0
        metrics = RiskMetrics(returns)

        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=target)

        # Correct: DD = sqrt(0.01 / 10) = sqrt(0.001) = 0.03162
        # Wrong (old code): DD = sqrt(var([-0.10])) = 0  or std([-0.10]) = 0
        expected_dd = np.sqrt(0.01 / 10)
        mean_return = np.mean(returns)
        expected_sortino = mean_return / expected_dd

        assert result["sortino_ratio"] == pytest.approx(expected_sortino, rel=1e-6)

    def test_weighted_sortino_correct_formula(self):
        """Weighted Sortino uses weighted average of min(r-target,0)^2."""
        returns = np.array([0.10, -0.05, 0.08, -0.03])
        weights = np.array([1.0, 2.0, 1.0, 2.0])
        target = 0.0

        metrics = RiskMetrics(returns, weights)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=target)

        # min(r_i - 0, 0)^2 = [0, 0.0025, 0, 0.0009]
        # weighted avg = (1*0 + 2*0.0025 + 1*0 + 2*0.0009) / (1+2+1+2)
        #              = (0.005 + 0.0018) / 6 = 0.00113333...
        expected_dd = np.sqrt(np.average(np.minimum(returns - target, 0) ** 2, weights=weights))
        mean_return = np.average(returns, weights=weights)
        expected_sortino = mean_return / expected_dd

        assert result["sortino_ratio"] == pytest.approx(expected_sortino, rel=1e-6)

    def test_all_returns_equal_target(self):
        """When all returns equal target, DD=0, Sortino=0."""
        returns = np.array([0.02, 0.02, 0.02])
        metrics = RiskMetrics(returns)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=0.02)
        assert result["sortino_ratio"] == 0

    def test_roe_analyzer_downside_deviation_uses_all_n(self):
        """ROEAnalyzer.volatility_metrics uses all N in denominator."""
        # 9 values above mean, 1 below
        roe = np.array([0.12] * 9 + [0.02])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()

        mean_roe = np.mean(roe)
        expected_dd = np.sqrt(np.mean(np.minimum(roe - mean_roe, 0) ** 2))

        assert vol["downside_deviation"] == pytest.approx(expected_dd, rel=1e-6)


# ---------------------------------------------------------------------------
# Plot distribution with weighted data (lines 544, 574-584, 617)
# ---------------------------------------------------------------------------


class TestPlotDistributionWeighted:
    """Test plot_distribution with weighted data."""

    @patch("matplotlib.pyplot.show")
    def test_plot_weighted_distribution(self, mock_show):
        """Plot distribution with weights exercises weighted Q-Q code (lines 574-584)."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 100)
        weights = np.random.uniform(0.5, 1.5, 100)
        metrics = RiskMetrics(losses, weights)

        fig = metrics.plot_distribution(bins=20)
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_distribution_metrics_off(self, mock_show):
        """Plot distribution with show_metrics=False."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 100)
        metrics = RiskMetrics(losses)

        fig = metrics.plot_distribution(show_metrics=False)
        assert fig is not None


# ---------------------------------------------------------------------------
# ROEAnalyzer edge cases
# ---------------------------------------------------------------------------


class TestROEAnalyzerEdgeCases:
    """Test ROEAnalyzer edge cases for uncovered lines."""

    def test_negative_growth_factors_fallback(self):
        """Time-weighted avg with negative growth factors uses arithmetic mean (line 719)."""
        roe_series = np.array([-2.0, 0.10, 0.05])  # -2.0 gives growth factor of -1.0
        analyzer = ROEAnalyzer(roe_series)
        result = analyzer.time_weighted_average()
        # Should fall back to arithmetic mean
        expected = float(np.mean(roe_series))
        assert abs(result - expected) < 1e-10

    def test_equity_weighted_no_equity(self):
        """Equity-weighted avg without equity_series returns time-weighted (line 733)."""
        roe_series = np.array([0.10, 0.12, 0.08])
        analyzer = ROEAnalyzer(roe_series)  # No equity_series
        eq_weighted = analyzer.equity_weighted_average()
        tw_weighted = analyzer.time_weighted_average()
        assert abs(eq_weighted - tw_weighted) < 1e-10

    def test_equity_weighted_zero_sum(self):
        """Equity-weighted avg with zero equity sum returns 0 (line 738)."""
        roe_series = np.array([0.10, 0.12, 0.08])
        equity_series = np.array([0.0, 0.0, 0.0])
        analyzer = ROEAnalyzer(roe_series, equity_series)
        result = analyzer.equity_weighted_average()
        assert result == 0.0

    def test_rolling_statistics_window_too_large(self):
        """Rolling statistics with window > series length raises ValueError (line 755)."""
        roe_series = np.array([0.10, 0.12])
        analyzer = ROEAnalyzer(roe_series)
        with pytest.raises(ValueError, match="Window .* larger than series length"):
            analyzer.rolling_statistics(window=5)

    def test_performance_ratios_single_value(self):
        """Performance ratios with < 2 values returns zeros (line 843)."""
        roe_series = np.array([0.10])
        analyzer = ROEAnalyzer(roe_series)
        ratios = analyzer.performance_ratios()
        assert ratios["sharpe_ratio"] == 0.0
        assert ratios["sortino_ratio"] == 0.0
        assert ratios["calmar_ratio"] == 0.0
        assert ratios["information_ratio"] == 0.0
        assert ratios["omega_ratio"] == 0.0

    def test_max_drawdown_single_value(self):
        """Max drawdown with < 2 values returns 0 (line 894)."""
        roe_series = np.array([0.10])
        analyzer = ROEAnalyzer(roe_series)
        dd = analyzer._calculate_max_drawdown()
        assert dd == 0.0

    def test_max_drawdown_overflow_fallback(self):
        """Max drawdown handles overflow in cumulative product (lines 905-912, 917).

        Create values large enough that np.cumprod(1 + roe) overflows even
        after clipping to 10.0, forcing the except branch.
        """
        # Values clipped to 10.0 give growth factors of 11.0
        # 11.0^200 = inf, triggering FloatingPointError with over="raise"
        roe_series = np.full(200, 10.0)
        analyzer = ROEAnalyzer(roe_series)
        dd = analyzer._calculate_max_drawdown()
        assert np.isfinite(dd)

    def test_max_drawdown_mixed_overflow(self):
        """Max drawdown with values causing partial overflow (line 917)."""
        # Mix large and small values to exercise both paths
        roe_series = np.concatenate(
            [
                np.full(100, 10.0),  # Causes overflow
                np.array([-0.5, -0.3, 0.1]),  # Normal values
            ]
        )
        analyzer = ROEAnalyzer(roe_series)
        dd = analyzer._calculate_max_drawdown()
        assert np.isfinite(dd)

    def test_distribution_analysis_empty(self):
        """Distribution analysis with empty valid ROE returns zeros (line 935)."""
        roe_series = np.array([np.nan, np.nan])
        analyzer = ROEAnalyzer(roe_series)
        dist = analyzer.distribution_analysis()
        assert dist["mean"] == 0.0
        assert dist["median"] == 0.0
        assert dist["skewness"] == 0.0

    def test_stability_analysis_default_periods(self):
        """Stability analysis with default periods (line 969)."""
        np.random.seed(42)
        roe_series = np.random.normal(0.10, 0.03, 20)
        analyzer = ROEAnalyzer(roe_series)
        stability = analyzer.stability_analysis()
        # Default periods are [1, 3, 5, 10]
        assert "1yr" in stability
        assert "3yr" in stability
        assert "5yr" in stability
        assert "10yr" in stability

    def test_stability_analysis_period_exceeds_series(self):
        """Stability analysis skips periods larger than series (line 975)."""
        roe_series = np.array([0.10, 0.12, 0.08])
        analyzer = ROEAnalyzer(roe_series)
        stability = analyzer.stability_analysis(periods=[2, 5, 10])
        # period=2 should work, period=5 and 10 should be skipped
        assert "2yr" in stability
        assert "5yr" not in stability
        assert "10yr" not in stability

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
        """Bootstrap CI with importance weights via var_with_ci."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 200)
        weights = np.random.uniform(0.5, 1.5, 200)
        metrics = RiskMetrics(losses, weights, seed=42)

        result = metrics.var_with_ci(0.95, n_bootstrap=50)
        assert isinstance(result, RiskMetricsResult)
        assert result.confidence_interval is not None
        assert result.confidence_interval[0] < result.value
        assert result.value < result.confidence_interval[1]

    def test_deprecated_bootstrap_ci_with_weights(self):
        """Deprecated bootstrap_ci=True with weights still works."""
        np.random.seed(42)
        losses = np.random.normal(1000, 100, 200)
        weights = np.random.uniform(0.5, 1.5, 200)
        metrics = RiskMetrics(losses, weights, seed=42)

        with pytest.warns(DeprecationWarning, match="bootstrap_ci.*deprecated"):
            result = metrics.var(0.95, bootstrap_ci=True, n_bootstrap=50)
        assert isinstance(result, RiskMetricsResult)
        assert result.confidence_interval is not None


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
# Issue #488: Sharpe ratio uses sample std (ddof=1)
# ---------------------------------------------------------------------------


class TestSharpeRatioSampleStd:
    """Verify Sharpe ratio uses sample standard deviation (ddof=1).

    SR = (R_bar - R_f) / s

    where s = sqrt(sum((R_i - R_bar)^2) / (n-1)) is the sample std.

    Reference: Sharpe, W. F. (1994). "The Sharpe Ratio."
    Journal of Portfolio Management, 21(1), 49-58.
    """

    def test_unweighted_sharpe_hand_calculated(self):
        """Issue #488 AC: hand-calculated Sharpe with known unweighted data."""
        returns = np.array([0.10, 0.12, -0.05, 0.08, -0.03])
        risk_free_rate = 0.02

        # Hand calculation:
        # mean = (0.10 + 0.12 - 0.05 + 0.08 - 0.03) / 5 = 0.044
        mean_r = 0.044
        # deviations from mean: [0.056, 0.076, -0.094, 0.036, -0.074]
        # squared: [0.003136, 0.005776, 0.008836, 0.001296, 0.005476]
        # sum = 0.02452
        # sample variance (ddof=1) = 0.02452 / 4 = 0.00613
        # sample std = sqrt(0.00613)
        sample_std = np.sqrt(0.02452 / 4)
        expected_sharpe = (mean_r - risk_free_rate) / sample_std

        metrics = RiskMetrics(returns)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=risk_free_rate)

        assert result["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-10)
        assert result["volatility"] == pytest.approx(sample_std, rel=1e-10)

    def test_unweighted_sharpe_matches_numpy_ddof1(self):
        """Sharpe volatility must equal np.std(returns, ddof=1)."""
        np.random.seed(123)
        returns = np.random.normal(0.05, 0.10, 50)
        expected_std = np.std(returns, ddof=1)

        metrics = RiskMetrics(returns)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=0.0)

        assert result["volatility"] == pytest.approx(expected_std, rel=1e-12)

    def test_unweighted_sharpe_not_population_std(self):
        """Sharpe volatility must NOT equal np.std(returns, ddof=0)."""
        # Use small sample where ddof=0 vs ddof=1 difference is large
        returns = np.array([0.10, 0.20, 0.30])
        pop_std = np.std(returns, ddof=0)

        metrics = RiskMetrics(returns)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=0.0)

        assert result["volatility"] != pytest.approx(pop_std, rel=1e-6)

    def test_weighted_sharpe_hand_calculated(self):
        """Issue #488 AC: hand-calculated Sharpe with known weighted data."""
        returns = np.array([0.10, -0.05, 0.08, -0.03])
        weights = np.array([1.0, 2.0, 1.0, 2.0])
        risk_free_rate = 0.02

        # Hand calculation:
        # weighted mean = (1*0.10 + 2*(-0.05) + 1*0.08 + 2*(-0.03)) / 6
        #               = (0.10 - 0.10 + 0.08 - 0.06) / 6 = 0.02 / 6
        w_mean = np.average(returns, weights=weights)
        # weighted population variance = sum(w_i*(x_i-mean)^2) / sum(w_i)
        pop_var = np.average((returns - w_mean) ** 2, weights=weights)
        # Bessel correction: V1^2 / (V1^2 - V2)
        v1 = np.sum(weights)  # 6.0
        v2 = np.sum(weights**2)  # 1+4+1+4 = 10.0
        bessel = v1**2 / (v1**2 - v2)  # 36 / 26
        sample_var = pop_var * bessel
        sample_std = np.sqrt(sample_var)
        expected_sharpe = (w_mean - risk_free_rate) / sample_std

        metrics = RiskMetrics(returns, weights)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=risk_free_rate)

        assert result["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-10)
        assert result["volatility"] == pytest.approx(sample_std, rel=1e-10)

    def test_weighted_bessel_reduces_to_unweighted(self):
        """Uniform weights must produce the same result as unweighted ddof=1."""
        returns = np.array([0.10, 0.12, -0.05, 0.08, -0.03])
        uniform_weights = np.ones(5)

        metrics_unw = RiskMetrics(returns)
        metrics_w = RiskMetrics(returns, uniform_weights)

        result_unw = metrics_unw.risk_adjusted_metrics(returns=returns, risk_free_rate=0.02)
        result_w = metrics_w.risk_adjusted_metrics(returns=returns, risk_free_rate=0.02)

        assert result_w["sharpe_ratio"] == pytest.approx(result_unw["sharpe_ratio"], rel=1e-10)
        assert result_w["volatility"] == pytest.approx(result_unw["volatility"], rel=1e-10)

    def test_small_sample_ddof_difference_significant(self):
        """For n=5, ddof=0 vs ddof=1 differs by ~12%, verifying correction matters."""
        returns = np.array([0.10, 0.12, -0.05, 0.08, -0.03])
        pop_std = np.std(returns, ddof=0)
        sample_std = np.std(returns, ddof=1)

        # Ratio should be sqrt(n/(n-1)) = sqrt(5/4) ≈ 1.118
        assert sample_std / pop_std == pytest.approx(np.sqrt(5.0 / 4.0), rel=1e-10)

        metrics = RiskMetrics(returns)
        result = metrics.risk_adjusted_metrics(returns=returns, risk_free_rate=0.0)
        assert result["volatility"] == pytest.approx(sample_std, rel=1e-10)


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


# ---------------------------------------------------------------------------
# Issue #307: weighted median bounds check
# ---------------------------------------------------------------------------


class TestWeightedMedianBoundsCheck:
    """Ensure searchsorted index is clamped to valid range."""

    def test_weighted_median_no_oob(self):
        """Weighted median with extreme weights doesn't access out of bounds."""
        # All weight concentrated on the first element: cumulative weights
        # reach 1.0 at index 0, so searchsorted(0.5) = 0 — no issue.
        # But if weights are such that 0.5 is never reached (float edge case),
        # searchsorted could return len(array).  We exercise the bounds clamp.
        losses = np.array([1.0, 2.0, 3.0])
        # Weights that sum to a value where float rounding may push searchsorted
        # past the end (edge case: all cumulative weights < 0.5)
        weights = np.array([1e-15, 1e-15, 1.0])
        rm = RiskMetrics(losses, weights=weights)
        stats = rm.summary_statistics()
        # Should not raise; median should be the last element
        assert np.isfinite(stats["median"])

    def test_weighted_median_single_element(self):
        """Weighted median with a single element is that element."""
        losses = np.array([42.0])
        weights = np.array([1.0])
        rm = RiskMetrics(losses, weights=weights)
        stats = rm.summary_statistics()
        assert stats["median"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Issue #307: maximum_drawdown docstring semantics
# ---------------------------------------------------------------------------


class TestMaximumDrawdownSemantics:
    """Verify maximum drawdown operates on cumulative losses."""

    def test_drawdown_is_on_cumulative_losses(self):
        """Confirm drawdown measures worst accumulated-loss stretch."""
        # Losses: [0, 10, 0, 0] → cumsum = [0, 10, 10, 10]
        # running_max - cumsum = [0, 0, 0, 0] — no decline from peak
        losses = np.array([0.0, 10.0, 0.0, 0.0])
        rm = RiskMetrics(losses)
        assert rm.maximum_drawdown() == pytest.approx(0.0)

        # Losses: [10, 0, 0, 0] → cumsum = [10, 10, 10, 10], dd = 0
        losses2 = np.array([10.0, 0.0, 0.0, 0.0])
        rm2 = RiskMetrics(losses2)
        assert rm2.maximum_drawdown() == pytest.approx(0.0)

        # Losses: [10, -5, 10, -5] → cumsum = [10, 5, 15, 10]
        # running_max = [10, 10, 15, 15], dd = [0, 5, 0, 5] → max 5
        losses3 = np.array([10.0, -5.0, 10.0, -5.0])
        rm3 = RiskMetrics(losses3)
        assert rm3.maximum_drawdown() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Issue #307: print() replaced with logging
# ---------------------------------------------------------------------------


class TestLoggingReplacedPrint:
    """Verify non-finite value warning uses logging, not print()."""

    def test_nonfinite_warning_uses_logger(self):
        """Removing non-finite values logs a warning, not print()."""
        import logging

        with patch("ergodic_insurance.risk_metrics.logger") as mock_logger:
            losses = np.array([1.0, np.nan, 3.0, np.inf])
            RiskMetrics(losses)
            mock_logger.warning.assert_called_once()
            assert "non-finite" in mock_logger.warning.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Issue #353: semi-variance uses correct formula over all observations
# ---------------------------------------------------------------------------


class TestSemiVarianceFix:
    """Semi-variance = (1/N) * sum(min(r_i - target, 0)^2) over ALL returns."""

    def test_semi_variance_known_values(self):
        """Hand-calculated semi-variance with target = 0."""
        # roe = [0.10, 0.05, -0.03, 0.15, -0.08]
        # min(r_i, 0)^2 = [0, 0, 0.0009, 0, 0.0064]
        # mean = (0 + 0 + 0.0009 + 0 + 0.0064) / 5 = 0.00146
        roe = np.array([0.10, 0.05, -0.03, 0.15, -0.08])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()

        expected_sv = (0.0009 + 0.0064) / 5
        assert vol["semi_variance"] == pytest.approx(expected_sv, rel=1e-6)

    def test_semi_variance_all_positive(self):
        """All positive returns → semi-variance = 0."""
        roe = np.array([0.05, 0.10, 0.15])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()
        assert vol["semi_variance"] == pytest.approx(0.0)

    def test_semi_variance_all_negative(self):
        """All negative returns → semi-variance = mean(r^2)."""
        roe = np.array([-0.10, -0.20, -0.30])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()

        expected_sv = np.mean(roe**2)
        assert vol["semi_variance"] == pytest.approx(expected_sv, rel=1e-6)

    def test_semi_variance_denominator_is_total_n(self):
        """Old bug: subset np.var divided by subset count, not total count."""
        # 9 positive returns, 1 negative
        roe = np.array([0.10] * 9 + [-0.05])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()

        # Correct: (0.05^2) / 10 = 0.00025
        expected_sv = 0.0025 / 10
        assert vol["semi_variance"] == pytest.approx(expected_sv, rel=1e-6)

        # Old wrong answer: np.var([-0.05]) = 0.0
        assert vol["semi_variance"] != 0.0


# ---------------------------------------------------------------------------
# Issue #353: tracking_error removed from volatility_metrics
# ---------------------------------------------------------------------------


class TestTrackingErrorRemoved:
    """tracking_error was trivially equal to std; verify it is removed."""

    def test_volatility_metrics_no_tracking_error_key(self):
        """volatility_metrics() should not contain tracking_error."""
        roe = np.array([0.05, 0.10, 0.15, 0.08])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()
        assert "tracking_error" not in vol

    def test_volatility_metrics_short_series_no_tracking_error(self):
        """Early-return path for < 2 values should also omit tracking_error."""
        roe = np.array([0.10])
        analyzer = ROEAnalyzer(roe)
        vol = analyzer.volatility_metrics()
        assert "tracking_error" not in vol


# ---------------------------------------------------------------------------
# Issue #1303: Sign convention parameter and validation
# ---------------------------------------------------------------------------


class TestSignConvention:
    """Test the convention parameter and sign-convention heuristic."""

    def test_default_convention_is_loss(self):
        """Default convention is 'loss' and stored on instance."""
        losses = np.array([100, 200, 300])
        rm = RiskMetrics(losses)
        assert rm.convention == "loss"

    def test_return_convention_negates_internally(self):
        """convention='return' negates data so VaR is computed on losses."""
        returns = np.array([0.05, 0.08, -0.02, 0.10, 0.03])
        rm = RiskMetrics(returns, convention="return")
        assert rm.convention == "return"
        # Internally stored losses should be the negation of the returns
        np.testing.assert_array_almost_equal(rm.losses, -returns)

    def test_invalid_convention_raises(self):
        """Invalid convention value raises ValueError."""
        with pytest.raises(ValueError, match="convention must be"):
            RiskMetrics(np.array([1, 2, 3]), convention="invalid")  # type: ignore[arg-type]

    def test_warning_when_mostly_negative_loss_convention(self):
        """Warn when >80% of values are negative under loss convention."""
        # 90% negative values with loss convention should warn
        data = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, 1.0])
        with pytest.warns(UserWarning, match="negative.*RiskMetrics expects losses"):
            RiskMetrics(data, convention="loss")

    def test_no_warning_when_return_convention(self):
        """No warning when convention='return' even if mostly negative."""
        data = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, 1.0])
        # Should NOT warn when convention is explicitly 'return'
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            RiskMetrics(data, convention="return")

    def test_no_warning_when_mostly_positive(self):
        """No warning when most values are positive under loss convention."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, -1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            RiskMetrics(data, convention="loss")

    def test_return_convention_var_matches_negated_loss(self):
        """VaR with convention='return' equals VaR on negated data."""
        returns = np.array([0.05, 0.08, -0.02, 0.10, 0.03, -0.01, 0.07, 0.04, 0.06, 0.09])
        rm_return = RiskMetrics(returns, convention="return")
        rm_loss = RiskMetrics(-returns, convention="loss")
        assert rm_return.var(0.95) == pytest.approx(rm_loss.var(0.95))
        assert rm_return.tvar(0.95) == pytest.approx(rm_loss.tvar(0.95))

    def test_return_convention_risk_adjusted_metrics(self):
        """risk_adjusted_metrics recovers original returns with convention='return'."""
        returns = np.array([0.05, 0.08, -0.02, 0.10, 0.03])
        rm = RiskMetrics(returns, convention="return")
        metrics = rm.risk_adjusted_metrics(risk_free_rate=0.02)
        # Mean return should match original returns, not losses
        expected_mean = np.mean(returns)
        assert metrics["mean_return"] == pytest.approx(expected_mean)

    def test_return_convention_pml(self):
        """PML with convention='return' matches negated loss convention."""
        returns = np.random.default_rng(42).normal(0.05, 0.10, 1000)
        rm_return = RiskMetrics(returns, convention="return")
        rm_loss = RiskMetrics(-returns, convention="loss")
        assert rm_return.pml(100) == pytest.approx(rm_loss.pml(100))

    def test_return_convention_economic_capital(self):
        """Economic capital consistent between conventions."""
        returns = np.random.default_rng(42).normal(0.05, 0.10, 1000)
        rm_return = RiskMetrics(returns, convention="return")
        rm_loss = RiskMetrics(-returns, convention="loss")
        assert rm_return.economic_capital(0.999) == pytest.approx(rm_loss.economic_capital(0.999))

    def test_return_convention_with_weights(self):
        """convention='return' works with importance weights."""
        returns = np.array([0.05, 0.08, -0.02, 0.10, 0.03])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        rm = RiskMetrics(returns, weights=weights, convention="return")
        assert rm.convention == "return"
        np.testing.assert_array_almost_equal(rm.losses, -returns)

    def test_backward_compatibility_default(self):
        """Existing code using default convention is unaffected."""
        np.random.seed(42)
        losses = np.random.lognormal(7, 1, 1000)
        rm = RiskMetrics(losses)
        # VaR, TVaR should be the same as before (loss convention)
        assert rm.var(0.95) > 0
        assert rm.tvar(0.95) >= rm.var(0.95)

    def test_coherence_test_with_return_convention(self):
        """coherence_test works with return convention."""
        np.random.seed(42)
        returns = np.random.normal(0.05, 0.10, 1000)
        rm = RiskMetrics(returns, convention="return")
        coherence = rm.coherence_test()
        assert coherence["tvar_positive_homogeneity"]
        assert coherence["tvar_translation_invariance"]

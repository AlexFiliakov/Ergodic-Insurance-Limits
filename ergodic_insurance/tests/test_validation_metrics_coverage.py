"""Additional tests for validation_metrics to cover missing lines.

Targets missing coverage lines:
143-157 (StrategyPerformance.to_dataframe), 200 (growth_rate without n_years),
214 (max_drawdown = 0.0 for single return), 248 (stability = 0.0 for <= 2 returns),
277-288 (calculate_rolling_metrics), 353 (PerformanceTargets max_drawdown failure),
358 (PerformanceTargets min_growth_rate failure)
"""

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.validation_metrics import (
    MetricCalculator,
    PerformanceTargets,
    StrategyPerformance,
    ValidationMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metrics():
    """Create a ValidationMetrics instance for reuse."""
    return ValidationMetrics(
        roe=0.10,
        ruin_probability=0.02,
        growth_rate=0.08,
        volatility=0.15,
        sharpe_ratio=0.53,
        max_drawdown=0.20,
        var_95=-0.05,
        cvar_95=-0.08,
        win_rate=0.60,
        profit_factor=1.5,
        recovery_time=3.0,
        stability=0.85,
    )


@pytest.fixture
def calculator():
    """Create a MetricCalculator instance."""
    return MetricCalculator(risk_free_rate=0.02)


# ---------------------------------------------------------------------------
# StrategyPerformance.to_dataframe (lines 143-157)
# ---------------------------------------------------------------------------


class TestStrategyPerformanceToDataframe:
    """Test the to_dataframe method of StrategyPerformance."""

    def test_to_dataframe_both_periods(self, sample_metrics):
        """DataFrame should have two rows when both in-sample and out-sample are set."""
        perf = StrategyPerformance(
            strategy_name="test_strategy",
            in_sample_metrics=sample_metrics,
            out_sample_metrics=sample_metrics,
        )
        df = perf.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "in_sample" in df["period"].values
        assert "out_sample" in df["period"].values
        assert "strategy" in df.columns
        assert all(df["strategy"] == "test_strategy")

    def test_to_dataframe_in_sample_only(self, sample_metrics):
        """DataFrame should have one row for in-sample only."""
        perf = StrategyPerformance(
            strategy_name="in_only",
            in_sample_metrics=sample_metrics,
        )
        df = perf.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["period"] == "in_sample"

    def test_to_dataframe_out_sample_only(self, sample_metrics):
        """DataFrame should have one row for out-sample only."""
        perf = StrategyPerformance(
            strategy_name="out_only",
            out_sample_metrics=sample_metrics,
        )
        df = perf.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["period"] == "out_sample"

    def test_to_dataframe_no_metrics(self):
        """DataFrame should be empty when no metrics are set."""
        perf = StrategyPerformance(strategy_name="empty")
        df = perf.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# MetricCalculator.calculate_metrics edge cases (lines 200, 214, 248)
# ---------------------------------------------------------------------------


class TestCalculateMetricsEdgeCases:
    """Test MetricCalculator.calculate_metrics for uncovered branches."""

    def test_growth_rate_without_n_years(self, calculator):
        """When final_assets is provided but n_years is None (line 200)."""
        returns = np.array([0.05, 0.06, 0.07, 0.08])
        final_assets = np.array([10_500_000, 10_600_000, 10_700_000, 10_800_000])
        metrics = calculator.calculate_metrics(
            returns=returns,
            final_assets=final_assets,
            initial_assets=10_000_000,
            n_years=None,
        )
        # growth_rate should be mean(final_assets / initial_assets - 1)
        expected_growth = float(np.mean(final_assets / 10_000_000 - 1))
        assert abs(metrics.growth_rate - expected_growth) < 1e-10

    def test_single_return_max_drawdown_zero(self, calculator):
        """Single return value should yield max_drawdown = 0 (line 214)."""
        returns = np.array([0.05])
        metrics = calculator.calculate_metrics(returns=returns)
        assert metrics.max_drawdown == 0.0

    def test_two_returns_stability_zero(self, calculator):
        """Two or fewer returns should yield stability = 0 (line 248)."""
        returns = np.array([0.05, 0.06])
        metrics = calculator.calculate_metrics(returns=returns)
        assert metrics.stability == 0.0

    def test_single_return_stability_zero(self, calculator):
        """Single return should also yield stability = 0."""
        returns = np.array([0.10])
        metrics = calculator.calculate_metrics(returns=returns)
        assert metrics.stability == 0.0


# ---------------------------------------------------------------------------
# MetricCalculator.calculate_rolling_metrics (lines 277-288)
# ---------------------------------------------------------------------------


class TestCalculateRollingMetrics:
    """Test the calculate_rolling_metrics method."""

    def test_basic_rolling_metrics(self, calculator):
        """Rolling metrics should produce correct number of windows (lines 277-288)."""
        np.random.seed(42)
        returns = np.random.normal(0.08, 0.02, 50)
        window_size = 10

        df = calculator.calculate_rolling_metrics(returns, window_size=window_size)

        assert isinstance(df, pd.DataFrame)
        expected_windows = len(returns) - window_size + 1
        assert len(df) == expected_windows
        assert "roe" in df.columns
        assert "sharpe_ratio" in df.columns
        assert "window_start" in df.columns
        assert "window_end" in df.columns

    def test_rolling_metrics_window_start_end(self, calculator):
        """Verify window_start and window_end values are correct."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.01, 20)
        df = calculator.calculate_rolling_metrics(returns, window_size=5)

        assert df.iloc[0]["window_start"] == 0
        assert df.iloc[0]["window_end"] == 5
        assert df.iloc[-1]["window_start"] == 15
        assert df.iloc[-1]["window_end"] == 20

    def test_rolling_metrics_single_window(self, calculator):
        """When window_size == len(returns), should get exactly 1 row."""
        returns = np.array([0.05, 0.06, 0.07])
        df = calculator.calculate_rolling_metrics(returns, window_size=3)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# PerformanceTargets.evaluate edge cases (lines 353, 358)
# ---------------------------------------------------------------------------


class TestPerformanceTargetsEdgeCases:
    """Test PerformanceTargets.evaluate for max_drawdown and min_growth_rate."""

    def test_max_drawdown_failure(self, sample_metrics):
        """Metrics exceeding max_drawdown target should fail (line 353)."""
        targets = PerformanceTargets(max_drawdown=0.10)
        passes, failures = targets.evaluate(sample_metrics)
        assert not passes
        assert len(failures) == 1
        assert "Max drawdown" in failures[0]

    def test_min_growth_rate_failure(self):
        """Metrics below min_growth_rate target should fail (line 358)."""
        metrics = ValidationMetrics(
            roe=0.05,
            ruin_probability=0.01,
            growth_rate=0.03,
            volatility=0.10,
        )
        targets = PerformanceTargets(min_growth_rate=0.05)
        passes, failures = targets.evaluate(metrics)
        assert not passes
        assert len(failures) == 1
        assert "Growth rate" in failures[0]

    def test_all_targets_pass(self, sample_metrics):
        """All generous targets should pass."""
        targets = PerformanceTargets(
            min_roe=0.05,
            max_ruin_probability=0.10,
            min_sharpe_ratio=0.30,
            max_drawdown=0.50,
            min_growth_rate=0.01,
        )
        passes, failures = targets.evaluate(sample_metrics)
        assert passes
        assert len(failures) == 0

    def test_multiple_failures(self):
        """Multiple targets can fail simultaneously."""
        metrics = ValidationMetrics(
            roe=0.01,
            ruin_probability=0.50,
            growth_rate=0.01,
            volatility=0.30,
            sharpe_ratio=-0.5,
            max_drawdown=0.80,
        )
        targets = PerformanceTargets(
            min_roe=0.05,
            max_ruin_probability=0.10,
            min_sharpe_ratio=0.5,
            max_drawdown=0.20,
            min_growth_rate=0.05,
        )
        passes, failures = targets.evaluate(metrics)
        assert not passes
        assert len(failures) == 5

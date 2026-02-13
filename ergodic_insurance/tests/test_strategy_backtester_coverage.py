"""Tests targeting specific uncovered lines in strategy_backtester.py.

This test file systematically covers the missing lines identified by coverage analysis:
- Line 103: InsuranceStrategy.get_description() base method
- Lines 267-394: OptimizedStaticStrategy.optimize_limits() with objective/constraint/minimize
- Lines 482, 486-527: AdaptiveStrategy.update() logic for all three branches
- Lines 634-635: StrategyBacktester.test_strategy() cache hit path
- Lines 641-646: StrategyBacktester.test_strategy() OptimizedStaticStrategy special handling
- Lines 778-794: StrategyBacktester._calculate_metrics() from SimulationResults
"""

from typing import Dict
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloResults
from ergodic_insurance.simulation import SimulationResults
from ergodic_insurance.strategy_backtester import (
    AdaptiveStrategy,
    BacktestResult,
    ConservativeFixedStrategy,
    InsuranceStrategy,
    NoInsuranceStrategy,
    OptimizedStaticStrategy,
    StrategyBacktester,
)
from ergodic_insurance.validation_metrics import MetricCalculator, ValidationMetrics

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manufacturer_config():
    """Standard ManufacturerConfig for tests."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        base_operating_margin=0.10,
        tax_rate=0.25,
        retention_ratio=0.80,
    )


@pytest.fixture
def manufacturer(manufacturer_config):
    """Standard WidgetManufacturer for tests."""
    return WidgetManufacturer(manufacturer_config)


@pytest.fixture
def sim_config():
    """Lightweight MonteCarloConfig for backtesting."""
    return MonteCarloConfig(
        n_simulations=10,
        n_years=5,
        parallel=False,
        seed=42,
        progress_bar=False,
        use_enhanced_parallel=False,
        enable_advanced_aggregation=False,
    )


@pytest.fixture
def mock_mc_results():
    """Build a plausible MonteCarloResults mock."""
    mock = MagicMock(spec=MonteCarloResults)
    mock.growth_rates = np.array([0.05, 0.08, 0.03, 0.10, 0.07, -0.02, 0.06, 0.04, 0.09, 0.01])
    mock.final_assets = np.array(
        [
            11_000_000,
            12_500_000,
            9_800_000,
            13_000_000,
            10_500_000,
            8_000_000,
            11_500_000,
            10_200_000,
            12_000_000,
            9_500_000,
        ]
    )
    mock.ruin_probability = {"5": 0.02, "10": 0.05}
    mock.metrics = {"mean_roe": 0.08, "median_roe": 0.07}
    return mock


@pytest.fixture
def mock_validation_metrics():
    """A ValidationMetrics object for result comparison."""
    return ValidationMetrics(
        roe=0.08,
        ruin_probability=0.02,
        growth_rate=0.06,
        volatility=0.03,
        sharpe_ratio=2.0,
        max_drawdown=0.05,
    )


# ===========================================================================
# Line 103: InsuranceStrategy.get_description()
# ===========================================================================


class TestInsuranceStrategyGetDescription:
    """Cover base class get_description() which returns '{name} strategy'."""

    def test_get_description_returns_name_with_suffix(self):
        """get_description should return '<name> strategy' for any concrete subclass."""
        strategy = NoInsuranceStrategy()
        desc = strategy.get_description()
        assert desc == "No Insurance strategy"

    def test_get_description_conservative(self):
        """Conservative strategy also inherits the base get_description."""
        strategy = ConservativeFixedStrategy()
        desc = strategy.get_description()
        assert desc == "Conservative Fixed strategy"

    def test_get_description_adaptive(self):
        """Adaptive strategy also inherits the base get_description."""
        strategy = AdaptiveStrategy()
        desc = strategy.get_description()
        assert desc == "Adaptive strategy"


# ===========================================================================
# Lines 267-394: OptimizedStaticStrategy.optimize_limits()
# ===========================================================================


class TestOptimizedStaticStrategyOptimizeLimits:
    """Cover the optimize_limits method including objective, constraint, and scipy.minimize.

    The optimize_limits method uses a shared ``_run_simulation`` helper with
    memoization so that objective and constraint share MC results for the same
    params.  We only need to patch the module-level MonteCarloEngine import.
    """

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_optimize_limits_success(self, mock_mc_cls, manufacturer, mock_mc_results):
        """When scipy.optimize.minimize succeeds, optimized_params should be set from result.x."""
        mock_engine_instance = MagicMock()
        mock_mc_results.metrics = {"mean_roe": 0.12}
        mock_mc_results.ruin_probability = {"5": 0.005}
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance

        strategy = OptimizedStaticStrategy(target_roe=0.15, max_ruin_prob=0.01)

        mock_opt_result = MagicMock()
        mock_opt_result.success = True
        mock_opt_result.x = np.array([150_000.0, 4_000_000.0, 15_000_000.0])

        with patch("scipy.optimize.minimize", return_value=mock_opt_result):
            strategy.optimize_limits(manufacturer, MagicMock())

        assert strategy.optimized_params is not None
        assert strategy.optimized_params["deductible"] == pytest.approx(150_000.0)
        assert strategy.optimized_params["primary_limit"] == pytest.approx(4_000_000.0)
        assert strategy.optimized_params["excess_limit"] == pytest.approx(15_000_000.0)

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_optimize_limits_failure_falls_back_to_defaults(
        self, mock_mc_cls, manufacturer, mock_mc_results
    ):
        """When scipy.optimize.minimize fails, should fall back to conservative defaults."""
        mock_engine_instance = MagicMock()
        mock_mc_results.metrics = {"mean_roe": 0.05}
        mock_mc_results.ruin_probability = {"5": 0.08}
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance

        strategy = OptimizedStaticStrategy(target_roe=0.15, max_ruin_prob=0.01)

        mock_opt_result = MagicMock()
        mock_opt_result.success = False
        mock_opt_result.x = np.array([200_000.0, 3_000_000.0, 10_000_000.0])

        with patch("scipy.optimize.minimize", return_value=mock_opt_result):
            strategy.optimize_limits(manufacturer, MagicMock())

        # Should have fallen back to defaults
        assert strategy.optimized_params is not None
        assert strategy.optimized_params["deductible"] == 100_000
        assert strategy.optimized_params["primary_limit"] == 5_000_000
        assert strategy.optimized_params["excess_limit"] == 20_000_000

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_optimize_limits_objective_calls_mc_engine(self, mock_mc_cls, manufacturer):
        """The objective and constraint share MC results via memoization."""
        mock_engine_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.metrics = {"mean_roe": 0.10}
        mock_results.ruin_probability = {"5": 0.003}
        mock_engine_instance.run.return_value = mock_results
        mock_mc_cls.return_value = mock_engine_instance

        strategy = OptimizedStaticStrategy(target_roe=0.15, max_ruin_prob=0.02)

        def fake_minimize(func, x0, method, bounds, constraints, options):
            constraint_fun = constraints[0]["fun"]

            # Call the objective then the constraint with the same params
            func(x0)
            constraint_fun(x0)

            result = MagicMock()
            result.success = True
            result.x = np.array(x0)
            return result

        with patch("scipy.optimize.minimize", side_effect=fake_minimize):
            strategy.optimize_limits(manufacturer, MagicMock())

        # With memoization, same params should only trigger one engine.run() call
        assert mock_engine_instance.run.call_count == 1

        # Params should be set from the "successful" result
        assert strategy.optimized_params is not None
        assert strategy.optimized_params["deductible"] == pytest.approx(100_000)
        assert strategy.optimized_params["primary_limit"] == pytest.approx(5_000_000)

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_optimize_limits_different_params_not_cached(self, mock_mc_cls, manufacturer):
        """Different params should trigger separate MC engine runs."""
        mock_engine_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.metrics = {"mean_roe": 0.10}
        mock_results.ruin_probability = {"5": 0.003}
        mock_engine_instance.run.return_value = mock_results
        mock_mc_cls.return_value = mock_engine_instance

        strategy = OptimizedStaticStrategy(target_roe=0.15, max_ruin_prob=0.02)

        def fake_minimize(func, x0, method, bounds, constraints, options):
            constraint_fun = constraints[0]["fun"]

            # Call with two different param sets
            func(x0)
            constraint_fun(x0)  # same params — should hit cache
            func([200000, 6000000, 25000000])  # different params — should miss cache

            result = MagicMock()
            result.success = True
            result.x = np.array(x0)
            return result

        with patch("scipy.optimize.minimize", side_effect=fake_minimize):
            strategy.optimize_limits(manufacturer, MagicMock())

        # Two distinct param sets = two engine.run() calls
        assert mock_engine_instance.run.call_count == 2

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_optimize_limits_ruin_constraint_with_empty_ruin_probability(
        self, mock_mc_cls, manufacturer
    ):
        """Test the ruin_constraint when ruin_probability dict is empty."""
        mock_engine_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.metrics = {"mean_roe": 0.10}
        mock_results.ruin_probability = {}  # empty dict
        mock_engine_instance.run.return_value = mock_results
        mock_mc_cls.return_value = mock_engine_instance

        strategy = OptimizedStaticStrategy(target_roe=0.15, max_ruin_prob=0.05)

        captured_constraint_val = None

        def fake_minimize(func, x0, method, bounds, constraints, options):
            nonlocal captured_constraint_val
            constraint_fun = constraints[0]["fun"]
            # Exercise constraint with empty ruin_probability
            captured_constraint_val = constraint_fun(x0)
            result = MagicMock()
            result.success = True
            result.x = np.array(x0)
            return result

        with patch("scipy.optimize.minimize", side_effect=fake_minimize):
            strategy.optimize_limits(manufacturer, MagicMock())

        # With empty ruin_probability, default is 0.0, so constraint = 0.05 - 0.0
        assert captured_constraint_val == pytest.approx(0.05)
        assert strategy.optimized_params is not None

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_optimize_limits_objective_returns_negative_roe(self, mock_mc_cls, manufacturer):
        """The objective function should return -mean_roe, and handle missing key gracefully."""
        mock_engine_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.metrics = {"mean_roe": 0.15}
        mock_results.ruin_probability = {"5": 0.01}
        mock_engine_instance.run.return_value = mock_results
        mock_mc_cls.return_value = mock_engine_instance

        strategy = OptimizedStaticStrategy()

        objective_return_values = []

        def fake_minimize(func, x0, method, bounds, constraints, options):
            val = func(x0)
            objective_return_values.append(val)
            result = MagicMock()
            result.success = True
            result.x = np.array(x0)
            return result

        with patch("scipy.optimize.minimize", side_effect=fake_minimize):
            strategy.optimize_limits(manufacturer, MagicMock())

        # The objective returns -mean_roe, so it should be -0.15
        assert len(objective_return_values) == 1
        assert objective_return_values[0] == pytest.approx(-0.15)


# ===========================================================================
# OptimizedStaticStrategy.get_insurance_program() — lines 396-430
# ===========================================================================


class TestOptimizedStaticStrategyGetInsuranceProgram:
    """Cover get_insurance_program with and without optimized_params."""

    def test_get_program_without_optimization_uses_defaults(self, manufacturer):
        """When optimized_params is None, defaults should be assigned."""
        strategy = OptimizedStaticStrategy()
        assert strategy.optimized_params is None

        program = strategy.get_insurance_program(manufacturer)

        # After call, defaults should have been set
        assert strategy.optimized_params is not None
        assert strategy.optimized_params["deductible"] == 100_000  # type: ignore[unreachable]
        assert strategy.optimized_params["primary_limit"] == 5_000_000
        assert strategy.optimized_params["excess_limit"] == 20_000_000

        # The program should have two layers
        assert isinstance(program, InsuranceProgram)
        assert len(program.layers) == 2

    def test_get_program_with_custom_optimized_params(self, manufacturer):
        """When optimized_params is already set, those values should be used."""
        strategy = OptimizedStaticStrategy()
        strategy.optimized_params = {
            "deductible": 200_000,
            "primary_limit": 3_000_000,
            "excess_limit": 12_000_000,
        }

        program = strategy.get_insurance_program(manufacturer)

        assert isinstance(program, InsuranceProgram)
        assert len(program.layers) == 2

        # Verify the first layer uses the custom deductible as attachment point
        layer0 = program.layers[0]
        assert layer0.attachment_point == pytest.approx(200_000)
        # Limit = primary_limit - deductible = 3_000_000 - 200_000 = 2_800_000
        assert layer0.limit == pytest.approx(2_800_000)

        # Second layer
        layer1 = program.layers[1]
        assert layer1.attachment_point == pytest.approx(3_000_000)
        assert layer1.limit == pytest.approx(12_000_000)

    def test_get_program_deductible_propagated(self, manufacturer):
        """The program deductible should match optimized_params['deductible']."""
        strategy = OptimizedStaticStrategy()
        strategy.optimized_params = {
            "deductible": 75_000,
            "primary_limit": 2_000_000,
            "excess_limit": 8_000_000,
        }

        program = strategy.get_insurance_program(manufacturer)
        assert program is not None
        assert program.deductible == pytest.approx(75_000)


# ===========================================================================
# Lines 468-537: AdaptiveStrategy.update()
# ===========================================================================


class TestAdaptiveStrategyUpdate:
    """Cover update() with all three branches: ratio > 1.5, ratio < 0.5, and middle."""

    def test_update_first_year_no_adaptation(self):
        """With only one year of history, no adaptation should occur."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
        )

        losses = np.array([50_000, 30_000])
        recoveries = np.array([40_000, 20_000])

        strategy.update(losses, recoveries, year=0)

        # Only 1 entry in history, need >= 2 for adaptation
        assert len(strategy.loss_history) == 1
        assert strategy.loss_history[0] == pytest.approx(80_000)
        # No adaptation yet, params unchanged
        assert strategy.current_deductible == 100_000
        assert strategy.current_primary == 3_000_000
        assert strategy.current_excess == 10_000_000
        assert len(strategy.adaptation_history) == 0

    def test_update_high_ratio_increases_coverage(self):
        """When recent losses >> average, ratio > 1.5 triggers increased coverage."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
            adjustment_factor=0.2,
        )

        # Year 0: low losses -> sets baseline
        strategy.update(np.array([50_000]), np.array([40_000]), year=0)
        assert len(strategy.adaptation_history) == 0

        # Year 1: very high losses -> ratio = 500_000 / mean(50_000, 500_000) > 1.5
        strategy.update(np.array([500_000]), np.array([200_000]), year=1)

        assert len(strategy.loss_history) == 2
        assert len(strategy.adaptation_history) == 1

        record = strategy.adaptation_history[0]
        avg = np.mean([50_000, 500_000])
        expected_ratio = 500_000 / avg
        assert record["ratio"] == pytest.approx(expected_ratio)
        assert expected_ratio > 1.5

        # Coverage should have increased: primary > base, excess > base
        assert strategy.current_primary > 3_000_000
        # Deductible should have decreased
        assert strategy.current_deductible < 100_000

    def test_update_low_ratio_decreases_coverage(self):
        """When recent losses << average, ratio < 0.5 triggers decreased coverage."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
            adjustment_factor=0.2,
        )

        # Year 0: moderate losses
        strategy.update(np.array([200_000]), np.array([100_000]), year=0)

        # Year 1: very low losses -> ratio = 10_000 / mean(200_000, 10_000) < 0.5
        strategy.update(np.array([10_000]), np.array([5_000]), year=1)

        assert len(strategy.adaptation_history) == 1
        record = strategy.adaptation_history[0]
        avg = np.mean([200_000, 10_000])
        expected_ratio = 10_000 / avg
        assert record["ratio"] == pytest.approx(expected_ratio)
        assert expected_ratio < 0.5

        # Coverage should have decreased
        assert strategy.current_primary < 3_000_000
        assert strategy.current_excess < 10_000_000
        # Deductible should have increased
        assert strategy.current_deductible > 100_000

    def test_update_moderate_ratio_gradual_return(self):
        """When ratio is between 0.5 and 1.5, parameters gradually return to base."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
            adjustment_factor=0.2,
        )

        # Manually set current parameters away from base
        strategy.current_primary = 4_000_000
        strategy.current_excess = 12_000_000
        strategy.current_deductible = 80_000

        # Year 0: moderate losses
        strategy.update(np.array([100_000]), np.array([80_000]), year=0)

        # Year 1: similar losses -> ratio ~ 1.0, which is between 0.5 and 1.5
        strategy.update(np.array([100_000]), np.array([80_000]), year=1)

        assert len(strategy.adaptation_history) == 1
        record = strategy.adaptation_history[0]
        assert 0.5 <= record["ratio"] <= 1.5

        # Should move toward base: current_primary = 0.9 * 4_000_000 + 0.1 * 3_000_000
        expected_primary = 0.9 * 4_000_000 + 0.1 * 3_000_000
        assert strategy.current_primary == pytest.approx(expected_primary)

        expected_excess = 0.9 * 12_000_000 + 0.1 * 10_000_000
        assert strategy.current_excess == pytest.approx(expected_excess)

        expected_ded = 0.9 * 80_000 + 0.1 * 100_000
        assert strategy.current_deductible == pytest.approx(expected_ded)

    def test_update_zero_average_losses_uses_ratio_one(self):
        """When average losses is 0, ratio should default to 1.0 (moderate branch)."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
        )

        # Two years of zero losses
        strategy.update(np.array([0.0]), np.array([0.0]), year=0)
        strategy.update(np.array([0.0]), np.array([0.0]), year=1)

        assert len(strategy.adaptation_history) == 1
        record = strategy.adaptation_history[0]
        assert record["ratio"] == pytest.approx(1.0)

    def test_update_trims_history_beyond_window(self):
        """When loss_history exceeds adaptation_window, old entries are trimmed (line 482)."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
            adaptation_window=3,
        )

        # Add 4 years of history (window = 3)
        for year in range(4):
            losses = np.array([100_000 * (year + 1)])
            strategy.update(losses, np.array([0.0]), year=year)

        # Should only keep last 3 entries
        assert len(strategy.loss_history) == 3
        assert strategy.loss_history[0] == pytest.approx(200_000)  # year 1
        assert strategy.loss_history[1] == pytest.approx(300_000)  # year 2
        assert strategy.loss_history[2] == pytest.approx(400_000)  # year 3

    def test_update_high_ratio_caps_at_double(self):
        """When adjustment is very large, coverage caps at 2x base."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
            adjustment_factor=0.5,
        )

        # Year 0: small losses
        strategy.update(np.array([1_000]), np.array([500]), year=0)
        # Year 1: enormous losses -> very high ratio
        strategy.update(np.array([100_000_000]), np.array([50_000_000]), year=1)

        # Primary should be capped at 2x base = 6_000_000
        assert strategy.current_primary <= 3_000_000 * 2
        # Excess capped at 2x base = 20_000_000
        assert strategy.current_excess <= 10_000_000 * 2
        # Deductible floored at 0.5x base = 50_000
        assert strategy.current_deductible >= 100_000 * 0.5

    def test_update_low_ratio_floors_at_half(self):
        """When adjustment is very large downward, coverage floors at 0.5x base."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
            adjustment_factor=0.5,
        )

        # Year 0: large losses
        strategy.update(np.array([10_000_000]), np.array([5_000_000]), year=0)
        # Year 1: tiny losses -> very low ratio
        strategy.update(np.array([1]), np.array([0]), year=1)

        # Primary should be floored at 0.5x base = 1_500_000
        assert strategy.current_primary >= 3_000_000 * 0.5
        # Excess floored at 0.5x base = 5_000_000
        assert strategy.current_excess >= 10_000_000 * 0.5
        # Deductible capped at 2x base = 200_000
        assert strategy.current_deductible <= 100_000 * 2


# ===========================================================================
# AdaptiveStrategy.get_insurance_program() — lines 539-565
# ===========================================================================


class TestAdaptiveStrategyGetInsuranceProgram:
    """Cover get_insurance_program returning a program with adapted parameters."""

    def test_get_program_returns_program_with_current_params(self, manufacturer):
        """The returned program should use current_deductible, current_primary, current_excess."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
        )

        program = strategy.get_insurance_program(manufacturer)

        assert isinstance(program, InsuranceProgram)
        assert len(program.layers) == 2
        assert program.deductible == pytest.approx(100_000)

        # First layer: attachment=deductible, limit=primary-deductible
        layer0 = program.layers[0]
        assert layer0.attachment_point == pytest.approx(100_000)
        assert layer0.limit == pytest.approx(3_000_000 - 100_000)

        # Second layer
        layer1 = program.layers[1]
        assert layer1.attachment_point == pytest.approx(3_000_000)
        assert layer1.limit == pytest.approx(10_000_000)

    def test_get_program_after_adaptation_reflects_changes(self, manufacturer):
        """After update() adjusts params, get_insurance_program should reflect them."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
        )

        # Force a high-loss adaptation
        strategy.update(np.array([10_000]), np.array([5_000]), year=0)
        strategy.update(np.array([500_000]), np.array([200_000]), year=1)

        program = strategy.get_insurance_program(manufacturer)
        assert program is not None
        # The layer parameters should differ from base after adaptation
        assert program.deductible == pytest.approx(strategy.current_deductible)
        layer0 = program.layers[0]
        assert layer0.attachment_point == pytest.approx(strategy.current_deductible)
        assert layer0.limit == pytest.approx(strategy.current_primary - strategy.current_deductible)


# ===========================================================================
# AdaptiveStrategy.reset() — lines 567-573
# ===========================================================================


class TestAdaptiveStrategyReset:
    """Cover reset() restoring base parameters and clearing history."""

    def test_reset_restores_base_parameters(self):
        """After reset, all parameters should return to base values."""
        strategy = AdaptiveStrategy(
            base_deductible=100_000,
            base_primary=3_000_000,
            base_excess=10_000_000,
        )

        # Modify state
        strategy.current_deductible = 50_000
        strategy.current_primary = 5_000_000
        strategy.current_excess = 15_000_000
        strategy.loss_history = [100_000, 200_000, 300_000]
        strategy.adaptation_history.append({"year": 0, "ratio": 1.5})

        strategy.reset()

        assert strategy.current_deductible == 100_000
        assert strategy.current_primary == 3_000_000
        assert strategy.current_excess == 10_000_000
        assert len(strategy.loss_history) == 0
        assert len(strategy.adaptation_history) == 0


# ===========================================================================
# Lines 634-635: StrategyBacktester.test_strategy() cache hit
# ===========================================================================


class TestStrategyBacktesterCacheHit:
    """Cover the cache-hit path at lines 634-635."""

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_cache_hit_returns_cached_result(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """Second call with same strategy+config should return cached result without re-running."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        backtester = StrategyBacktester()
        strategy = ConservativeFixedStrategy()

        # First call - should run simulation
        result1 = backtester.test_strategy(strategy, manufacturer, sim_config, use_cache=True)
        assert mock_engine_instance.run.call_count == 1

        # Second call - should use cache
        result2 = backtester.test_strategy(strategy, manufacturer, sim_config, use_cache=True)
        # MC engine should NOT have been called again
        assert mock_engine_instance.run.call_count == 1

        # Both results should be the same object
        assert result1 is result2

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_cache_disabled_always_reruns(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """With use_cache=False, simulation should always run."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        backtester = StrategyBacktester()
        strategy = ConservativeFixedStrategy()

        backtester.test_strategy(strategy, manufacturer, sim_config, use_cache=False)
        backtester.test_strategy(strategy, manufacturer, sim_config, use_cache=False)
        assert mock_engine_instance.run.call_count == 2


# ===========================================================================
# Lines 641-646: OptimizedStaticStrategy special handling in test_strategy
# ===========================================================================


class TestStrategyBacktesterOptimizedHandling:
    """Cover the special handling for OptimizedStaticStrategy in test_strategy."""

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_optimized_strategy_triggers_optimization_when_no_params(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """When OptimizedStaticStrategy has no optimized_params and simulation_engine exists,
        optimize_limits should be called."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        sim_engine = MagicMock()
        backtester = StrategyBacktester(simulation_engine=sim_engine)

        strategy = OptimizedStaticStrategy()
        assert strategy.optimized_params is None

        with patch.object(strategy, "optimize_limits") as mock_optimize:
            result = backtester.test_strategy(strategy, manufacturer, sim_config)
            mock_optimize.assert_called_once_with(manufacturer, sim_engine)

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_optimized_strategy_no_engine_logs_warning(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """When simulation_engine is None, optimization should be skipped with a warning."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        # No simulation engine
        backtester = StrategyBacktester(simulation_engine=None)

        strategy = OptimizedStaticStrategy()
        assert strategy.optimized_params is None

        import logging

        with patch("ergodic_insurance.strategy_backtester.logger") as mock_logger:
            result = backtester.test_strategy(strategy, manufacturer, sim_config)
            # Should log warning about no simulation engine
            mock_logger.warning.assert_called_once_with(
                "No simulation engine available for optimization"
            )

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_optimized_strategy_with_existing_params_skips_optimization(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """When OptimizedStaticStrategy already has optimized_params, optimize_limits should not run."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        sim_engine = MagicMock()
        backtester = StrategyBacktester(simulation_engine=sim_engine)

        strategy = OptimizedStaticStrategy()
        strategy.optimized_params = {
            "deductible": 100_000,
            "primary_limit": 5_000_000,
            "excess_limit": 20_000_000,
        }

        with patch.object(strategy, "optimize_limits") as mock_optimize:
            result = backtester.test_strategy(strategy, manufacturer, sim_config)
            mock_optimize.assert_not_called()


# ===========================================================================
# Lines 778-794: StrategyBacktester._calculate_metrics() from SimulationResults
# ===========================================================================


class TestStrategyBacktesterCalculateMetrics:
    """Cover _calculate_metrics() which processes a SimulationResults (not MC)."""

    def test_calculate_metrics_surviving_simulation(self):
        """Test metrics calculation with a simulation that survived (no insolvency)."""
        backtester = StrategyBacktester()

        n_years = 10
        sim_results = SimulationResults(
            years=np.arange(n_years),
            assets=np.linspace(10_000_000, 15_000_000, n_years),
            equity=np.linspace(5_000_000, 8_000_000, n_years),
            roe=np.array([0.08, 0.10, 0.07, 0.09, 0.11, 0.06, 0.12, 0.08, 0.10, 0.09]),
            revenue=np.full(n_years, 5_000_000),
            net_income=np.full(n_years, 400_000),
            claim_counts=np.ones(n_years, dtype=int),
            claim_amounts=np.full(n_years, 100_000),
            insolvency_year=None,
        )

        metrics = backtester._calculate_metrics(sim_results, n_years)

        assert isinstance(metrics, ValidationMetrics)
        # ROE should be the mean of the roe array
        expected_roe = float(np.mean(sim_results.roe))
        assert metrics.roe == pytest.approx(expected_roe, rel=1e-3)
        # Ruin probability should be 0 since insolvency_year is None
        assert metrics.ruin_probability == pytest.approx(0.0)
        # Growth rate should be computed from final_assets / initial_assets
        assert metrics.growth_rate != 0.0  # Should have non-zero growth

    def test_calculate_metrics_insolvent_simulation(self):
        """Test metrics calculation with a simulation that went insolvent."""
        backtester = StrategyBacktester()

        n_years = 5
        sim_results = SimulationResults(
            years=np.arange(n_years),
            assets=np.array([10_000_000, 8_000_000, 5_000_000, 1_000_000, 0]),
            equity=np.array([5_000_000, 3_000_000, 1_000_000, -500_000, -1_000_000]),
            roe=np.array([-0.20, -0.30, -0.40, -0.50, -1.0]),
            revenue=np.full(n_years, 5_000_000),
            net_income=np.array([-1_000_000, -2_000_000, -3_000_000, -4_000_000, -5_000_000]),
            claim_counts=np.array([5, 8, 12, 15, 20]),
            claim_amounts=np.array([2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000]),
            insolvency_year=4,
        )

        metrics = backtester._calculate_metrics(sim_results, n_years)

        assert isinstance(metrics, ValidationMetrics)
        # Ruin probability should be 1.0 since insolvency_year is set
        assert metrics.ruin_probability == pytest.approx(1.0)
        # ROE should be negative
        assert metrics.roe < 0

    def test_calculate_metrics_uses_final_asset_value(self):
        """The final_assets passed to calculate_metrics should be [simulation_results.assets[-1]]."""
        backtester = StrategyBacktester()

        # Mock the metric_calculator to verify what gets passed
        mock_calculator = MagicMock(spec=MetricCalculator)
        mock_metrics = ValidationMetrics(
            roe=0.1, ruin_probability=0.0, growth_rate=0.05, volatility=0.02
        )
        mock_calculator.calculate_metrics.return_value = mock_metrics
        backtester.metric_calculator = mock_calculator

        n_years = 3
        sim_results = SimulationResults(
            years=np.arange(n_years),
            assets=np.array([10_000_000, 11_000_000, 12_000_000]),
            equity=np.array([5_000_000, 5_500_000, 6_000_000]),
            roe=np.array([0.08, 0.10, 0.09]),
            revenue=np.full(n_years, 5_000_000),
            net_income=np.full(n_years, 400_000),
            claim_counts=np.ones(n_years, dtype=int),
            claim_amounts=np.full(n_years, 50_000),
            insolvency_year=None,
        )

        metrics = backtester._calculate_metrics(sim_results, n_years)

        # Verify calculate_metrics was called with the correct arguments
        mock_calculator.calculate_metrics.assert_called_once()
        call_kwargs = mock_calculator.calculate_metrics.call_args

        # final_assets should be [assets[-1]] = [12_000_000]
        np.testing.assert_array_equal(
            call_kwargs.kwargs.get("final_assets", call_kwargs[1].get("final_assets")),
            np.array([12_000_000]),
        )
        # n_years should be passed through
        assert call_kwargs.kwargs.get("n_years", call_kwargs[1].get("n_years")) == 3

        # Ruin probability should be 0.0 (no insolvency), overriding whatever calculator set
        assert metrics.ruin_probability == pytest.approx(0.0)


# ===========================================================================
# BacktestResult dataclass
# ===========================================================================


class TestBacktestResult:
    """Verify BacktestResult dataclass holds expected fields."""

    def test_backtest_result_creation(self, mock_mc_results, sim_config, mock_validation_metrics):
        """BacktestResult can be created with all required fields."""
        result = BacktestResult(
            strategy_name="Test Strategy",
            simulation_results=mock_mc_results,
            metrics=mock_validation_metrics,
            execution_time=1.5,
            config=sim_config,
        )

        assert result.strategy_name == "Test Strategy"
        assert result.execution_time == pytest.approx(1.5)
        assert result.metrics.roe == pytest.approx(0.08)
        assert result.config.n_simulations == 10


# ===========================================================================
# Integration-like tests (still mocked, but end-to-end flow)
# ===========================================================================


class TestStrategyBacktesterTestStrategy:
    """Cover test_strategy end-to-end flow including NoInsurance path."""

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_no_insurance_strategy_creates_empty_program(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """NoInsuranceStrategy returns None, which should create empty InsuranceProgram."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        backtester = StrategyBacktester()
        strategy = NoInsuranceStrategy()

        result = backtester.test_strategy(strategy, manufacturer, sim_config)

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "No Insurance"
        assert result.execution_time >= 0

        # Verify MC engine was created (the empty program path was taken)
        mock_mc_cls.assert_called_once()

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    @patch("ergodic_insurance.strategy_backtester.ManufacturingLossGenerator")
    def test_test_strategy_result_has_valid_metrics(
        self, mock_loss_gen_cls, mock_mc_cls, manufacturer, sim_config, mock_mc_results
    ):
        """The result should contain valid ValidationMetrics."""
        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = mock_mc_results
        mock_mc_cls.return_value = mock_engine_instance
        mock_loss_gen_cls.return_value = MagicMock()

        backtester = StrategyBacktester()
        strategy = ConservativeFixedStrategy()

        result = backtester.test_strategy(strategy, manufacturer, sim_config)

        assert isinstance(result.metrics, ValidationMetrics)
        # ROE should be computed from growth_rates
        assert isinstance(result.metrics.roe, float)
        assert isinstance(result.metrics.volatility, float)

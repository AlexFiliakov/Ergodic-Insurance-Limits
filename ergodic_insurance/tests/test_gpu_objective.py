"""Tests for GPU-accelerated objective evaluation (Issue #966).

Tests cover:
- GPUBatchObjective: vectorized objective evaluation matching CPU scalar results
- GPUObjectiveWrapper: scipy.optimize compatibility, caching, gradient computation
- GPUMultiStartScreener: batch starting point screening
- GPUDifferentialEvolution: GPU-native DE optimizer
- BusinessOptimizer GPU integration: maximize_roe_gpu method
- Pareto frontier GPU acceleration
- CPU fallback behavior when GPU is unavailable

Since:
    Version 0.11.0 (Issue #966)
"""

from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.business_optimizer import (
    BusinessConstraints,
    BusinessOptimizer,
    OptimalStrategy,
)
from ergodic_insurance.config.optimizer import BusinessOptimizerConfig
from ergodic_insurance.gpu_backend import GPUConfig
from ergodic_insurance.gpu_objective import (
    GPUBatchObjective,
    GPUDifferentialEvolution,
    GPUMultiStartScreener,
    GPUObjectiveWrapper,
)
from ergodic_insurance.manufacturer import WidgetManufacturer


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizer_config():
    """Default optimizer config."""
    return BusinessOptimizerConfig()


@pytest.fixture
def gpu_config_disabled():
    """GPU config with GPU disabled (CPU fallback)."""
    return GPUConfig(enabled=False)


@pytest.fixture
def batch_objective(optimizer_config, gpu_config_disabled):
    """GPUBatchObjective using CPU backend (no CuPy needed)."""
    return GPUBatchObjective(
        equity=4_000_000,
        total_assets=10_000_000,
        revenue=5_000_000,
        optimizer_config=optimizer_config,
        gpu_config=gpu_config_disabled,
    )


@pytest.fixture
def mock_manufacturer():
    """Mock manufacturer for BusinessOptimizer tests."""
    manufacturer = Mock(spec=WidgetManufacturer)
    manufacturer.total_assets = 10_000_000
    manufacturer.equity = 4_000_000
    manufacturer.liabilities = 6_000_000
    manufacturer.revenue = 5_000_000
    manufacturer.operating_income = 500_000
    manufacturer.cash = 2_000_000
    manufacturer.config = Mock()
    manufacturer.calculate_revenue = Mock(return_value=5_000_000)
    return manufacturer


# ---------------------------------------------------------------------------
#  GPUBatchObjective tests
# ---------------------------------------------------------------------------


class TestGPUBatchObjective:
    """Test vectorized batch evaluation matches scalar CPU implementations."""

    def test_construction(self, batch_objective):
        """Test GPUBatchObjective can be constructed."""
        assert batch_objective.equity == 4_000_000
        assert batch_objective.total_assets == 10_000_000
        assert batch_objective.revenue == 5_000_000
        assert batch_objective.use_gpu is False  # No CuPy in test env

    def test_evaluate_batch_roe_shape(self, batch_objective):
        """Test ROE batch evaluation returns correct shape."""
        param_sets = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.03],
            [10_000_000, 200_000, 0.015],
        ])
        result = batch_objective.evaluate_batch_roe(param_sets, time_horizon=5)
        assert result.shape == (3,)
        assert result.dtype == np.float64

    def test_evaluate_batch_roe_single(self, batch_objective):
        """Test single-element batch."""
        param_sets = np.array([[5_000_000, 100_000, 0.02]])
        result = batch_objective.evaluate_batch_roe(param_sets, time_horizon=5)
        assert result.shape == (1,)

    def test_evaluate_batch_roe_matches_scalar(self, batch_objective):
        """Test batch ROE matches scalar BusinessOptimizer._simulate_roe.

        Note: The scalar version uses np.random.default_rng (new Generator API)
        while the batch version uses np.random.seed (legacy API), so the MC noise
        differs slightly. Both converge to the same expected value; we use a
        tolerance consistent with MC standard error (~0.01 for 100 sims).
        """
        from ergodic_insurance.business_optimizer import BusinessOptimizer

        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 4_000_000
        manufacturer.calculate_revenue = Mock(return_value=5_000_000)
        manufacturer.config = Mock()

        optimizer = BusinessOptimizer(manufacturer)

        params = [5_000_000, 100_000, 0.02]
        scalar_roe = optimizer._simulate_roe(
            coverage_limit=params[0],
            deductible=params[1],
            premium_rate=params[2],
            time_horizon=5,
            n_simulations=100,
        )

        batch_roe = batch_objective.evaluate_batch_roe(
            np.array([params]), time_horizon=5, n_simulations=100
        )

        # MC noise uses different RNG streams, so tolerance is ~O(noise_std/sqrt(n))
        assert abs(batch_roe[0] - scalar_roe) < 0.02

    def test_evaluate_batch_bankruptcy_risk_shape(self, batch_objective):
        """Test bankruptcy risk returns correct shape and bounds."""
        param_sets = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.03],
        ])
        result = batch_objective.evaluate_batch_bankruptcy_risk(param_sets, time_horizon=5)
        assert result.shape == (2,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_evaluate_batch_bankruptcy_risk_matches_scalar(self, batch_objective):
        """Test batch bankruptcy risk matches scalar implementation."""
        from ergodic_insurance.business_optimizer import BusinessOptimizer

        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 4_000_000
        manufacturer.calculate_revenue = Mock(return_value=5_000_000)
        manufacturer.config = Mock()

        optimizer = BusinessOptimizer(manufacturer)

        params = [5_000_000, 100_000, 0.02]
        scalar_risk = optimizer._estimate_bankruptcy_risk(
            coverage_limit=params[0],
            deductible=params[1],
            premium_rate=params[2],
            time_horizon=10,
        )

        batch_risk = batch_objective.evaluate_batch_bankruptcy_risk(
            np.array([params]), time_horizon=10
        )

        assert abs(batch_risk[0] - scalar_risk) < 1e-10

    def test_evaluate_batch_growth_rate_shape(self, batch_objective):
        """Test growth rate returns correct shape and non-negative values."""
        param_sets = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.03],
        ])
        result = batch_objective.evaluate_batch_growth_rate(param_sets, time_horizon=5)
        assert result.shape == (2,)
        assert np.all(result >= 0)

    def test_evaluate_batch_growth_rate_matches_scalar(self, batch_objective):
        """Test batch growth rate matches scalar implementation."""
        from ergodic_insurance.business_optimizer import BusinessOptimizer

        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 4_000_000
        manufacturer.calculate_revenue = Mock(return_value=5_000_000)
        manufacturer.config = Mock()

        optimizer = BusinessOptimizer(manufacturer)

        params = [5_000_000, 100_000, 0.02]
        scalar_growth = optimizer._estimate_growth_rate(
            coverage_limit=params[0],
            deductible=params[1],
            premium_rate=params[2],
            time_horizon=10,
        )

        batch_growth = batch_objective.evaluate_batch_growth_rate(
            np.array([params]), time_horizon=10
        )

        assert abs(batch_growth[0] - scalar_growth) < 1e-10

    def test_evaluate_batch_growth_rate_metrics(self, batch_objective):
        """Test growth rate with different metrics produces different values."""
        params = np.array([[5_000_000, 100_000, 0.02]])
        revenue_growth = batch_objective.evaluate_batch_growth_rate(
            params, time_horizon=5, metric="revenue"
        )
        asset_growth = batch_objective.evaluate_batch_growth_rate(
            params, time_horizon=5, metric="assets"
        )
        equity_growth = batch_objective.evaluate_batch_growth_rate(
            params, time_horizon=5, metric="equity"
        )
        # Different metrics should produce different values
        assert revenue_growth[0] != asset_growth[0]
        assert revenue_growth[0] != equity_growth[0]

    def test_evaluate_batch_capital_efficiency_shape(self, batch_objective):
        """Test capital efficiency returns correct shape."""
        param_sets = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.03],
        ])
        result = batch_objective.evaluate_batch_capital_efficiency(param_sets)
        assert result.shape == (2,)
        assert np.all(result >= 0)

    def test_evaluate_batch_capital_efficiency_matches_scalar(self, batch_objective):
        """Test batch capital efficiency matches scalar implementation."""
        from ergodic_insurance.business_optimizer import BusinessOptimizer

        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 4_000_000
        manufacturer.calculate_revenue = Mock(return_value=5_000_000)
        manufacturer.config = Mock()

        optimizer = BusinessOptimizer(manufacturer)

        params = [5_000_000, 100_000, 0.02]
        scalar_eff = optimizer._calculate_capital_efficiency(
            coverage_limit=params[0],
            deductible=params[1],
            premium_rate=params[2],
        )

        batch_eff = batch_objective.evaluate_batch_capital_efficiency(
            np.array([params])
        )

        assert abs(batch_eff[0] - scalar_eff) < 1e-10

    def test_deductible_ratio_edge_cases(self, batch_objective):
        """Test deductible ratio handles edge cases."""
        xp = batch_objective.xp

        # Zero coverage limit
        ratio = batch_objective._deductible_ratio(
            xp.array([100_000.0]), xp.array([0.0])
        )
        assert float(ratio[0]) == 0.0

        # Deductible exceeds coverage
        ratio = batch_objective._deductible_ratio(
            xp.array([10_000_000.0]), xp.array([5_000_000.0])
        )
        assert float(ratio[0]) == 1.0  # clamped

        # Normal case
        ratio = batch_objective._deductible_ratio(
            xp.array([100_000.0]), xp.array([5_000_000.0])
        )
        assert abs(float(ratio[0]) - 0.02) < 1e-10

    def test_multiple_param_sets_varied(self, batch_objective):
        """Test that different parameters produce different results."""
        param_sets = np.array([
            [5_000_000, 10_000, 0.02],
            [5_000_000, 500_000, 0.02],
        ])
        roe = batch_objective.evaluate_batch_roe(param_sets, time_horizon=5)
        assert roe[0] != roe[1]  # Different deductibles should give different ROE


# ---------------------------------------------------------------------------
#  GPUObjectiveWrapper tests
# ---------------------------------------------------------------------------


class TestGPUObjectiveWrapper:
    """Test scipy-compatible objective wrapper."""

    def test_call_returns_scalar(self, batch_objective):
        """Test __call__ returns a float scalar."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="roe", time_horizon=5
        )
        x = np.array([5_000_000, 100_000, 0.02])
        result = wrapper(x)
        assert isinstance(result, float)

    def test_call_negates_maximization(self, batch_objective):
        """Test that maximization objectives are negated for scipy."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="roe", time_horizon=5
        )
        x = np.array([5_000_000, 100_000, 0.02])
        raw = wrapper.evaluate_single(x)
        negated = wrapper(x)
        assert negated == -raw

    def test_call_no_negation_minimization(self, batch_objective):
        """Test that minimization objectives are not negated."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="bankruptcy_risk", time_horizon=5
        )
        x = np.array([5_000_000, 100_000, 0.02])
        raw = wrapper.evaluate_single(x)
        result = wrapper(x)
        assert result == raw

    def test_caching(self, batch_objective):
        """Test that repeated calls with same input use cache."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="roe", time_horizon=5, cache_enabled=True
        )
        x = np.array([5_000_000, 100_000, 0.02])
        result1 = wrapper(x)
        result2 = wrapper(x)
        assert result1 == result2

    def test_cache_clear(self, batch_objective):
        """Test cache clearing."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="roe", time_horizon=5
        )
        x = np.array([5_000_000, 100_000, 0.02])
        wrapper(x)
        assert len(wrapper._cache) > 0
        wrapper.clear_cache()
        assert len(wrapper._cache) == 0

    def test_gradient_shape(self, batch_objective):
        """Test gradient returns correct shape."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="roe", time_horizon=5
        )
        x = np.array([5_000_000, 100_000, 0.02])
        grad = wrapper.gradient(x)
        assert grad.shape == (3,)
        assert grad.dtype == np.float64

    def test_gradient_finite_difference(self, batch_objective):
        """Test gradient is a valid finite difference approximation.

        Uses bankruptcy_risk (deterministic, no MC noise) to avoid RNG
        stream mismatches between batched and sequential evaluations.
        """
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="bankruptcy_risk", time_horizon=5,
            cache_enabled=False,
        )
        x = np.array([5_000_000, 100_000, 0.02])
        grad = wrapper.gradient(x, h=1.0)

        # Verify gradient manually for first parameter
        h = 1.0
        f_plus = wrapper(x + np.array([h, 0, 0]))
        f_minus = wrapper(x - np.array([h, 0, 0]))
        expected_grad_0 = (f_plus - f_minus) / (2 * h)
        assert abs(grad[0] - expected_grad_0) < 1e-6

    def test_unknown_objective_raises(self, batch_objective):
        """Test unknown objective name raises ValueError."""
        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="unknown", time_horizon=5
        )
        x = np.array([5_000_000, 100_000, 0.02])
        with pytest.raises(ValueError, match="Unknown objective"):
            wrapper(x)

    def test_all_objective_names(self, batch_objective):
        """Test all supported objective names work."""
        x = np.array([5_000_000, 100_000, 0.02])
        for name in ["roe", "bankruptcy_risk", "growth_rate", "capital_efficiency"]:
            wrapper = GPUObjectiveWrapper(
                batch_objective, objective_name=name, time_horizon=5
            )
            result = wrapper(x)
            assert isinstance(result, float)
            assert np.isfinite(result)


# ---------------------------------------------------------------------------
#  GPUMultiStartScreener tests
# ---------------------------------------------------------------------------


class TestGPUMultiStartScreener:
    """Test GPU-accelerated starting point screening."""

    def test_screen_returns_correct_count(self, batch_objective):
        """Test screening returns requested number of points."""
        screener = GPUMultiStartScreener(
            batch_objective, objective_name="roe", time_horizon=5
        )
        rng = np.random.default_rng(42)
        starts = rng.uniform(
            [1e6, 0, 0.001], [20e6, 1e6, 0.10], size=(20, 3)
        )
        top = screener.screen_starting_points(starts, top_k=5)
        assert len(top) == 5

    def test_screen_top_k_exceeds_total(self, batch_objective):
        """Test requesting more points than available."""
        screener = GPUMultiStartScreener(
            batch_objective, objective_name="roe", time_horizon=5
        )
        starts = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.03],
        ])
        top = screener.screen_starting_points(starts, top_k=10)
        assert len(top) == 2  # Only 2 available

    def test_screen_roe_sorts_descending(self, batch_objective):
        """Test maximization objectives sort best-first (descending)."""
        screener = GPUMultiStartScreener(
            batch_objective, objective_name="roe", time_horizon=5
        )
        starts = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.01],
            [2_000_000, 500_000, 0.05],
        ])
        top = screener.screen_starting_points(starts, top_k=3)
        # Evaluate to verify ordering
        values = batch_objective.evaluate_batch_roe(np.array(top), time_horizon=5)
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_screen_risk_sorts_ascending(self, batch_objective):
        """Test minimization objectives sort best-first (ascending)."""
        screener = GPUMultiStartScreener(
            batch_objective, objective_name="bankruptcy_risk", time_horizon=5
        )
        starts = np.array([
            [5_000_000, 100_000, 0.02],
            [8_000_000, 50_000, 0.01],
            [2_000_000, 500_000, 0.05],
        ])
        top = screener.screen_starting_points(starts, top_k=3)
        values = batch_objective.evaluate_batch_bankruptcy_risk(
            np.array(top), time_horizon=5
        )
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


# ---------------------------------------------------------------------------
#  GPUDifferentialEvolution tests
# ---------------------------------------------------------------------------


class TestGPUDifferentialEvolution:
    """Test GPU-native differential evolution optimizer."""

    def test_optimize_returns_result(self, batch_objective):
        """Test DE optimization returns a result object."""
        bounds = [
            (1e6, 20e6),
            (0, 1e6),
            (0.001, 0.10),
        ]
        de = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="roe", time_horizon=5, seed=42,
        )
        result = de.optimize(pop_size=10, n_generations=5)
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "nit")
        assert hasattr(result, "nfev")
        assert hasattr(result, "success")
        assert result.success is True
        assert result.x.shape == (3,)

    def test_optimize_respects_bounds(self, batch_objective):
        """Test DE result is within bounds."""
        bounds = [
            (1e6, 20e6),
            (0, 1e6),
            (0.001, 0.10),
        ]
        de = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="roe", time_horizon=5, seed=42,
        )
        result = de.optimize(pop_size=10, n_generations=10)
        for i, (lo, hi) in enumerate(bounds):
            assert result.x[i] >= lo - 1e-6
            assert result.x[i] <= hi + 1e-6

    def test_optimize_population_output(self, batch_objective):
        """Test DE returns final population."""
        bounds = [
            (1e6, 20e6),
            (0, 1e6),
            (0.001, 0.10),
        ]
        de = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="roe", time_horizon=5, seed=42,
        )
        result = de.optimize(pop_size=15, n_generations=3)
        assert result.population.shape == (15, 3)
        assert result.population_values.shape == (15,)

    def test_optimize_nfev_tracking(self, batch_objective):
        """Test function evaluation counting."""
        bounds = [(1e6, 20e6), (0, 1e6), (0.001, 0.10)]
        pop_size = 10
        n_gens = 5
        de = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="roe", time_horizon=5, seed=42,
        )
        result = de.optimize(pop_size=pop_size, n_generations=n_gens)
        # Initial eval + one per generation
        expected_nfev = pop_size + n_gens * pop_size
        assert result.nfev == expected_nfev

    def test_optimize_minimization_objective(self, batch_objective):
        """Test DE with minimization objective (bankruptcy_risk)."""
        bounds = [(1e6, 20e6), (0, 1e6), (0.001, 0.10)]
        de = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="bankruptcy_risk", time_horizon=5, seed=42,
        )
        result = de.optimize(pop_size=10, n_generations=5)
        assert result.fun >= 0  # Risk should be non-negative
        assert result.fun <= 1  # Risk should be bounded


# ---------------------------------------------------------------------------
#  BusinessOptimizer GPU integration tests
# ---------------------------------------------------------------------------


class TestBusinessOptimizerGPU:
    """Test GPU integration in BusinessOptimizer."""

    def test_init_with_gpu_config(self, mock_manufacturer):
        """Test BusinessOptimizer accepts gpu_config parameter."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)
        assert optimizer.gpu_config is gpu_config
        assert optimizer._gpu_batch_objective is None

    def test_init_without_gpu_config(self, mock_manufacturer):
        """Test BusinessOptimizer works without gpu_config (default)."""
        optimizer = BusinessOptimizer(mock_manufacturer)
        assert optimizer.gpu_config is None

    def test_get_gpu_batch_objective_with_config(self, mock_manufacturer):
        """Test lazy initialization of GPU batch objective."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)
        batch_obj = optimizer._get_gpu_batch_objective()
        assert batch_obj is not None
        assert isinstance(batch_obj, GPUBatchObjective)

    def test_get_gpu_batch_objective_without_config(self, mock_manufacturer):
        """Test GPU batch objective is None when no config provided."""
        optimizer = BusinessOptimizer(mock_manufacturer)
        batch_obj = optimizer._get_gpu_batch_objective()
        assert batch_obj is None

    def test_with_manufacturer_preserves_gpu_config(self, mock_manufacturer):
        """Test with_manufacturer passes gpu_config to clone."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)

        new_manufacturer = Mock(spec=WidgetManufacturer)
        new_manufacturer.total_assets = 20_000_000
        new_manufacturer.equity = 8_000_000
        new_manufacturer.calculate_revenue = Mock(return_value=10_000_000)

        cloned = optimizer.with_manufacturer(new_manufacturer)
        assert cloned.gpu_config is gpu_config

    def test_maximize_roe_gpu_scipy_method(self, mock_manufacturer):
        """Test GPU-accelerated ROE maximization with scipy method."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)
        constraints = BusinessConstraints()

        strategy = optimizer.maximize_roe_gpu(
            constraints=constraints,
            time_horizon=5,
            n_simulations=10,
            method="scipy",
        )

        assert isinstance(strategy, OptimalStrategy)
        assert strategy.coverage_limit > 0
        assert strategy.deductible >= 0
        assert 0 < strategy.premium_rate <= 0.10

    def test_maximize_roe_gpu_de_method(self, mock_manufacturer):
        """Test GPU-accelerated ROE maximization with DE method."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)
        constraints = BusinessConstraints()

        strategy = optimizer.maximize_roe_gpu(
            constraints=constraints,
            time_horizon=5,
            n_simulations=10,
            method="de",
            de_pop_size=10,
            de_generations=5,
        )

        assert isinstance(strategy, OptimalStrategy)
        assert strategy.coverage_limit > 0

    def test_maximize_roe_gpu_multi_start_method(self, mock_manufacturer):
        """Test GPU-accelerated ROE maximization with multi-start method."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)
        constraints = BusinessConstraints()

        strategy = optimizer.maximize_roe_gpu(
            constraints=constraints,
            time_horizon=5,
            n_simulations=10,
            method="multi_start",
            n_starts=5,
            top_k=3,
        )

        assert isinstance(strategy, OptimalStrategy)
        assert strategy.coverage_limit > 0

    def test_maximize_roe_gpu_fallback_no_config(self, mock_manufacturer):
        """Test fallback to CPU when no gpu_config."""
        optimizer = BusinessOptimizer(mock_manufacturer)
        constraints = BusinessConstraints()

        strategy = optimizer.maximize_roe_gpu(
            constraints=constraints,
            time_horizon=5,
            n_simulations=10,
        )

        # Should still return a valid result via CPU fallback
        assert isinstance(strategy, OptimalStrategy)
        assert strategy.coverage_limit > 0

    def test_maximize_roe_gpu_produces_recommendations(self, mock_manufacturer):
        """Test that GPU optimization produces recommendations."""
        gpu_config = GPUConfig(enabled=False)
        optimizer = BusinessOptimizer(mock_manufacturer, gpu_config=gpu_config)
        constraints = BusinessConstraints()

        strategy = optimizer.maximize_roe_gpu(
            constraints=constraints, time_horizon=5, n_simulations=10
        )

        assert len(strategy.recommendations) > 0
        assert all(isinstance(r, str) for r in strategy.recommendations)


# ---------------------------------------------------------------------------
#  Pareto frontier GPU tests
# ---------------------------------------------------------------------------


class TestParetoFrontierGPU:
    """Test GPU-accelerated Pareto frontier operations."""

    def test_pareto_frontier_accepts_gpu_config(self):
        """Test ParetoFrontier constructor accepts gpu_config."""
        from ergodic_insurance.pareto_frontier import (
            Objective,
            ObjectiveType,
            ParetoFrontier,
        )

        objectives = [
            Objective(name="ROE", type=ObjectiveType.MAXIMIZE),
            Objective(name="risk", type=ObjectiveType.MINIMIZE),
        ]

        def dummy_objective(x):
            return {"ROE": float(x[0]), "risk": float(x[1])}

        bounds = [(0, 1), (0, 1)]
        gpu_config = GPUConfig(enabled=False)

        frontier = ParetoFrontier(
            objectives=objectives,
            objective_function=dummy_objective,
            bounds=bounds,
            gpu_config=gpu_config,
        )

        assert frontier.gpu_config is gpu_config
        assert frontier._use_gpu is False

    def test_filter_dominated_points_gpu_path(self):
        """Test dominated point filtering with GPU config."""
        from ergodic_insurance.pareto_frontier import (
            Objective,
            ObjectiveType,
            ParetoFrontier,
            ParetoPoint,
        )

        objectives = [
            Objective(name="ROE", type=ObjectiveType.MAXIMIZE),
            Objective(name="risk", type=ObjectiveType.MINIMIZE),
        ]

        gpu_config = GPUConfig(enabled=False)
        frontier = ParetoFrontier(
            objectives=objectives,
            objective_function=lambda x: {},
            bounds=[(0, 1)],
            gpu_config=gpu_config,
        )

        points = [
            ParetoPoint(objectives={"ROE": 0.2, "risk": 0.05}, decision_variables=np.array([1])),
            ParetoPoint(objectives={"ROE": 0.15, "risk": 0.08}, decision_variables=np.array([2])),
            ParetoPoint(objectives={"ROE": 0.25, "risk": 0.03}, decision_variables=np.array([3])),
        ]

        filtered = frontier._filter_dominated_points(points)

        # Point 3 dominates point 1 (higher ROE, lower risk)
        # Point 3 dominates point 2 (higher ROE, lower risk)
        assert len(filtered) == 1
        assert filtered[0].objectives["ROE"] == 0.25


# ---------------------------------------------------------------------------
#  Optimization module GPU tests
# ---------------------------------------------------------------------------


class TestOptimizationGPU:
    """Test GPU features in optimization module."""

    def test_multi_start_accepts_gpu_config(self):
        """Test MultiStartOptimizer accepts gpu_config."""
        from scipy.optimize import Bounds

        from ergodic_insurance.optimization import MultiStartOptimizer

        gpu_config = GPUConfig(enabled=False)
        optimizer = MultiStartOptimizer(
            objective_fn=lambda x: np.sum(x**2),
            bounds=Bounds([0, 0], [1, 1]),
            gpu_config=gpu_config,
        )
        assert optimizer.gpu_config is gpu_config

    def test_screen_starting_points_gpu(self):
        """Test GPU-accelerated starting point screening."""
        from scipy.optimize import Bounds

        from ergodic_insurance.optimization import MultiStartOptimizer

        gpu_config = GPUConfig(enabled=False)
        optimizer = MultiStartOptimizer(
            objective_fn=lambda x: np.sum(x**2),
            bounds=Bounds([0, 0], [1, 1]),
            gpu_config=gpu_config,
        )

        starts = [
            np.array([0.9, 0.9]),
            np.array([0.1, 0.1]),
            np.array([0.5, 0.5]),
        ]
        top = optimizer.screen_starting_points_gpu(starts, top_k=2)
        assert len(top) == 2
        # Best for x^2 is smallest magnitude
        assert np.allclose(top[0], [0.1, 0.1])

    def test_create_optimizer_passes_gpu_config(self):
        """Test factory passes gpu_config to MultiStartOptimizer."""
        from scipy.optimize import Bounds

        from ergodic_insurance.optimization import create_optimizer

        gpu_config = GPUConfig(enabled=False)
        optimizer = create_optimizer(
            method="multi-start",
            objective_fn=lambda x: np.sum(x**2),
            bounds=Bounds([0, 0], [1, 1]),
            gpu_config=gpu_config,
        )
        assert optimizer.gpu_config is gpu_config


# ---------------------------------------------------------------------------
#  CPU fallback / edge case tests
# ---------------------------------------------------------------------------


class TestCPUFallback:
    """Test that everything works correctly without GPU."""

    def test_batch_objective_cpu_only(self, optimizer_config):
        """Test GPUBatchObjective works without any GPU config."""
        batch_obj = GPUBatchObjective(
            equity=4_000_000,
            total_assets=10_000_000,
            revenue=5_000_000,
            optimizer_config=optimizer_config,
            gpu_config=None,
        )
        assert batch_obj.use_gpu is False

        params = np.array([[5_000_000, 100_000, 0.02]])
        result = batch_obj.evaluate_batch_roe(params, time_horizon=5)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_de_deterministic_with_seed(self, batch_objective):
        """Test DE produces deterministic results with same seed."""
        bounds = [(1e6, 20e6), (0, 1e6), (0.001, 0.10)]

        de1 = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="roe", time_horizon=5, seed=123,
        )
        result1 = de1.optimize(pop_size=10, n_generations=3)

        de2 = GPUDifferentialEvolution(
            batch_objective, bounds=bounds,
            objective_name="roe", time_horizon=5, seed=123,
        )
        result2 = de2.optimize(pop_size=10, n_generations=3)

        np.testing.assert_array_almost_equal(result1.x, result2.x)
        assert abs(result1.fun - result2.fun) < 1e-10

    def test_wrapper_with_scipy_minimize(self, batch_objective):
        """Test GPUObjectiveWrapper works with scipy.optimize.minimize."""
        from scipy import optimize as sp_optimize

        wrapper = GPUObjectiveWrapper(
            batch_objective, objective_name="roe", time_horizon=5,
            n_simulations=10,
        )

        x0 = np.array([8_000_000, 100_000, 0.02])
        bounds = [(1e6, 20e6), (0, 1e6), (0.001, 0.10)]

        result = sp_optimize.minimize(
            wrapper,
            x0,
            method="SLSQP",
            jac=wrapper.gradient,
            bounds=bounds,
            options={"maxiter": 20, "ftol": 1e-4},
        )

        assert result.x.shape == (3,)
        assert np.all(np.isfinite(result.x))

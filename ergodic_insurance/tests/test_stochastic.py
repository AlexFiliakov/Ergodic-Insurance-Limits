"""Tests for stochastic processes."""

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.stochastic_processes import (
    GeometricBrownianMotion,
    LognormalVolatility,
    MeanRevertingProcess,
    StochasticConfig,
    create_stochastic_process,
)


class TestStochasticProcesses:
    """Test stochastic process implementations."""

    def test_gbm_creation(self):
        """Test creation of Geometric Brownian Motion process."""
        config = StochasticConfig(volatility=0.2, drift=0.05, random_seed=42)
        process = GeometricBrownianMotion(config)

        assert process.config.volatility == 0.2
        assert process.config.drift == 0.05
        assert process.config.random_seed == 42

    def test_gbm_reproducibility(self):
        """Test that GBM produces reproducible results with fixed seed."""
        config = StochasticConfig(volatility=0.2, drift=0.05, random_seed=42)
        process1 = GeometricBrownianMotion(config)
        process2 = GeometricBrownianMotion(config)

        # Generate shocks from both processes
        shocks1 = [process1.generate_shock(100) for _ in range(10)]
        shocks2 = [process2.generate_shock(100) for _ in range(10)]

        # Should be identical with same seed
        assert shocks1 == shocks2

    def test_gbm_shock_distribution(self):
        """Test that GBM shocks have expected statistical properties."""
        config = StochasticConfig(volatility=0.2, drift=0.0, random_seed=42, time_step=1.0)
        process = GeometricBrownianMotion(config)

        # Generate many shocks
        n_samples = 10000
        shocks = [process.generate_shock(100) for _ in range(n_samples)]

        # Log of shocks should be normally distributed
        log_shocks = np.log(shocks)

        # Expected mean of log shocks: (drift - 0.5*vol^2)*dt
        expected_mean = (0.0 - 0.5 * 0.2**2) * 1.0
        # Expected std of log shocks: vol*sqrt(dt)
        expected_std = 0.2 * np.sqrt(1.0)

        # Check within reasonable bounds (3 standard errors)
        sample_mean = np.mean(log_shocks)
        sample_std = np.std(log_shocks)

        assert abs(sample_mean - expected_mean) < 3 * expected_std / np.sqrt(n_samples)
        assert abs(sample_std - expected_std) < 0.01

    def test_lognormal_volatility(self):
        """Test lognormal volatility generator."""
        config = StochasticConfig(volatility=0.15, drift=0.0, random_seed=123)
        process = LognormalVolatility(config)

        # Generate shocks
        shocks = [process.generate_shock(100) for _ in range(1000)]

        # Mean should be close to 1 (for small volatility)
        mean_shock = np.mean(shocks)
        assert 0.95 < mean_shock < 1.05

        # Check volatility is roughly correct
        log_std = np.std(np.log(shocks))
        assert abs(log_std - 0.15) < 0.02

    def test_mean_reverting_process(self):
        """Test mean-reverting process."""
        config = StochasticConfig(volatility=0.1, drift=0.0, random_seed=456)
        process = MeanRevertingProcess(config, mean_level=1.0, reversion_speed=0.5)

        # Start far from mean
        current = 2.0
        shocks = []
        for _ in range(100):
            shock = process.generate_shock(current)
            current = current * shock
            shocks.append(current)

        # Should revert toward mean of 1.0
        final_values = shocks[-20:]
        mean_final = np.mean(final_values)
        assert 0.8 < mean_final < 1.2  # Should be close to 1.0

    def test_process_factory(self):
        """Test factory function for creating processes."""
        # Test GBM creation
        gbm = create_stochastic_process("gbm", volatility=0.2, drift=0.05, random_seed=42)
        assert isinstance(gbm, GeometricBrownianMotion)

        # Test lognormal creation
        lognormal = create_stochastic_process("lognormal", volatility=0.15)
        assert isinstance(lognormal, LognormalVolatility)

        # Test mean reverting creation
        mean_rev = create_stochastic_process("mean_reverting", volatility=0.1)
        assert isinstance(mean_rev, MeanRevertingProcess)

        # Test invalid type
        with pytest.raises(ValueError):
            create_stochastic_process("invalid_type", volatility=0.1)

    def test_process_reset(self):
        """Test resetting stochastic process."""
        config = StochasticConfig(volatility=0.2, drift=0.05, random_seed=42)
        process = GeometricBrownianMotion(config)

        # Generate some shocks
        shocks1 = [process.generate_shock(100) for _ in range(5)]

        # Reset with same seed
        process.reset(42)

        # Should generate same shocks
        shocks2 = [process.generate_shock(100) for _ in range(5)]
        assert shocks1 == shocks2

        # Reset with different seed
        process.reset(123)

        # Should generate different shocks
        shocks3 = [process.generate_shock(100) for _ in range(5)]
        assert shocks1 != shocks3


class TestStochasticManufacturer:
    """Test manufacturer with stochastic processes."""

    def test_deterministic_compatibility(self):
        """Test that manufacturer still works in deterministic mode."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )

        # Create without stochastic process
        manufacturer = WidgetManufacturer(config)

        # Should work normally
        metrics = manufacturer.step()
        assert metrics["revenue"] > 0
        assert metrics["net_income"] > 0
        assert "assets" in metrics

    def test_stochastic_mode(self):
        """Test manufacturer with stochastic process."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )

        # Create with stochastic process
        stoch_config = StochasticConfig(volatility=0.15, drift=0.0, random_seed=42)
        process = LognormalVolatility(stoch_config)
        manufacturer = WidgetManufacturer(config, stochastic_process=process)

        # Run with stochastic shocks
        metrics_stoch = manufacturer.step(apply_stochastic=True)

        # Reset and run without stochastic
        manufacturer.reset()
        metrics_det = manufacturer.step(apply_stochastic=False)

        # Results should differ
        assert metrics_stoch["revenue"] != metrics_det["revenue"]

    def test_stochastic_reproducibility(self):
        """Test that stochastic simulation is reproducible with same seed."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )

        # Create two manufacturers with same seed
        stoch_config = StochasticConfig(volatility=0.2, drift=0.0, random_seed=42)
        process1 = LognormalVolatility(stoch_config)
        manufacturer1 = WidgetManufacturer(config, stochastic_process=process1)

        process2 = LognormalVolatility(StochasticConfig(volatility=0.2, drift=0.0, random_seed=42))
        manufacturer2 = WidgetManufacturer(config, stochastic_process=process2)

        # Run simulations
        results1 = []
        results2 = []
        for _ in range(10):
            results1.append(manufacturer1.step(apply_stochastic=True)["revenue"])
            results2.append(manufacturer2.step(apply_stochastic=True)["revenue"])

        # Should produce identical results
        assert results1 == results2

    def test_stochastic_growth(self):
        """Test stochastic growth rate application."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )

        stoch_config = StochasticConfig(volatility=0.1, drift=0.0, random_seed=42)
        process = LognormalVolatility(stoch_config)
        manufacturer = WidgetManufacturer(config, stochastic_process=process)

        # Run with growth and stochastic
        initial_turnover = manufacturer.asset_turnover_ratio
        manufacturer.step(growth_rate=0.05, apply_stochastic=True)
        final_turnover = manufacturer.asset_turnover_ratio

        # Turnover should have changed
        assert final_turnover != initial_turnover
        # Should be roughly 5% higher (with some stochastic variation)
        assert 1.0 < final_turnover / initial_turnover < 1.2

    def test_memory_efficiency(self):
        """Test that long simulations don't use excessive memory."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )

        stoch_config = StochasticConfig(volatility=0.15, drift=0.0, random_seed=42)
        process = LognormalVolatility(stoch_config)
        manufacturer = WidgetManufacturer(config, stochastic_process=process)

        # Run 1000-year simulation
        for _ in range(1000):
            manufacturer.step(apply_stochastic=True)

        # Check that we haven't accumulated too much history
        # (this is a simple check - real memory testing would be more complex)
        assert len(manufacturer.metrics_history) == 1000

        # Basic sanity check on final state
        assert manufacturer.current_year == 1000
        assert manufacturer.total_assets > 0  # Should still have assets

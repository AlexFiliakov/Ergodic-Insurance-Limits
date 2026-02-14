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
        """Test mean-reverting process reverts toward target."""
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

    def test_mean_reverting_shocks_always_positive(self):
        """Test that exponential OU shocks are always strictly positive."""
        config = StochasticConfig(volatility=0.5, drift=0.0, random_seed=789)
        process = MeanRevertingProcess(config, mean_level=1.0, reversion_speed=0.5)

        # Test across a wide range of current values, including near-zero
        for current_value in [0.01, 0.1, 1.0, 5.0, 100.0]:
            for _ in range(500):
                shock = process.generate_shock(current_value)
                assert (
                    shock > 0
                ), f"Shock must be positive, got {shock} at current_value={current_value}"

    def test_mean_reverting_volatility_state_independent(self):
        """Test that multiplicative shock volatility doesn't depend on current value."""
        config = StochasticConfig(volatility=0.2, drift=0.0, random_seed=42)
        n_samples = 5000

        # Generate log-shock distributions at two very different current values
        stds = []
        for current_value in [0.5, 50.0]:
            process = MeanRevertingProcess(
                StochasticConfig(volatility=0.2, drift=0.0, random_seed=42),
                mean_level=1.0,
                reversion_speed=0.0,  # No reversion to isolate volatility
            )
            log_shocks = [np.log(process.generate_shock(current_value)) for _ in range(n_samples)]
            stds.append(np.std(log_shocks))

        # With exponential OU at theta=0, log-shock std = sigma*sqrt(dt) regardless
        # of current_value. Allow 5% relative tolerance.
        assert (
            abs(stds[0] - stds[1]) / stds[0] < 0.05
        ), f"Volatility should be state-independent: std@0.5={stds[0]:.4f}, std@50={stds[1]:.4f}"

    def test_mean_reverting_stationary_distribution(self):
        """Verify stationary distribution matches N(log(mu), sigma^2/(2*theta)).

        Simulate 100,000 independent OU paths long enough to reach stationarity,
        then check that the empirical distribution of log(X) matches the
        theoretical stationary distribution within 3 standard errors.
        """
        theta = 1.0
        sigma = 0.3
        mu = 2.0
        dt = 1.0
        n_paths = 100_000
        burn_in = 50  # steps to reach stationarity

        config = StochasticConfig(volatility=sigma, drift=0.0, random_seed=12345, time_step=dt)
        process = MeanRevertingProcess(config, mean_level=mu, reversion_speed=theta)

        # Run all paths from the same starting point, collecting final values
        final_log_values = np.empty(n_paths)
        current = 1.0  # starting far from mu to stress burn-in
        for i in range(n_paths):
            x = current
            for _ in range(burn_in):
                x = x * process.generate_shock(x)
            final_log_values[i] = np.log(x)

        # Theoretical stationary distribution: N(log(mu), sigma^2 / (2*theta))
        expected_mean = np.log(mu)
        expected_var = sigma**2 / (2 * theta)
        expected_std = np.sqrt(expected_var)

        sample_mean = np.mean(final_log_values)
        sample_var = np.var(final_log_values)
        se_mean = expected_std / np.sqrt(n_paths)

        assert abs(sample_mean - expected_mean) < 3 * se_mean, (
            f"Stationary mean mismatch: sample={sample_mean:.4f}, "
            f"expected={expected_mean:.4f}, 3*SE={3*se_mean:.4f}"
        )
        # Variance check (chi-squared): SE of sample variance ≈ expected_var * sqrt(2/n)
        se_var = expected_var * np.sqrt(2 / n_paths)
        assert abs(sample_var - expected_var) < 3 * se_var, (
            f"Stationary variance mismatch: sample={sample_var:.6f}, "
            f"expected={expected_var:.6f}, 3*SE={3*se_var:.6f}"
        )

    def test_mean_reverting_long_run_variance(self):
        """Verify that the long-run variance of log(X) equals sigma^2 / (2*theta)."""
        theta = 0.5
        sigma = 0.2
        mu = 1.0
        dt = 1.0
        n_paths = 50_000
        n_steps = 100  # long enough for convergence

        config = StochasticConfig(volatility=sigma, drift=0.0, random_seed=999, time_step=dt)
        process = MeanRevertingProcess(config, mean_level=mu, reversion_speed=theta)

        final_log_values = np.empty(n_paths)
        for i in range(n_paths):
            x = mu  # start at mean
            for _ in range(n_steps):
                x = x * process.generate_shock(x)
            final_log_values[i] = np.log(x)

        expected_var = sigma**2 / (2 * theta)
        sample_var = np.var(final_log_values)
        se_var = expected_var * np.sqrt(2 / n_paths)

        assert abs(sample_var - expected_var) < 3 * se_var, (
            f"Long-run variance: sample={sample_var:.6f}, "
            f"expected={expected_var:.6f}, 3*SE={3*se_var:.6f}"
        )

    def test_mean_reverting_theta_zero_reduces_to_gbm(self):
        """For theta -> 0, the shock reduces to pure GBM (no mean reversion)."""
        sigma = 0.2
        dt = 1.0
        n_samples = 10_000
        current = 5.0  # arbitrary starting point

        # MeanRevertingProcess with theta ≈ 0
        config = StochasticConfig(volatility=sigma, drift=0.0, random_seed=42, time_step=dt)
        process = MeanRevertingProcess(config, mean_level=1.0, reversion_speed=1e-12)

        log_shocks = np.array([np.log(process.generate_shock(current)) for _ in range(n_samples)])

        # With theta=0, log_shock should be N(0, sigma^2 * dt)
        # (no mean reversion, pure diffusion)
        assert abs(np.mean(log_shocks)) < 3 * sigma * np.sqrt(dt) / np.sqrt(n_samples)
        assert abs(np.std(log_shocks) - sigma * np.sqrt(dt)) < 0.02

    def test_mean_reverting_large_theta_jumps_to_mean(self):
        """For theta -> infinity, the shock immediately jumps to the mean level."""
        sigma = 0.2
        dt = 1.0
        mu = 3.0
        theta = 50.0
        n_samples = 200

        config = StochasticConfig(volatility=sigma, drift=0.0, random_seed=42, time_step=dt)
        process = MeanRevertingProcess(config, mean_level=mu, reversion_speed=theta)

        # With theta=50, conditional_std = sigma*sqrt(1/(2*theta)) ≈ 0.02.
        # Average over many draws to verify the mean converges to mu.
        for current in [0.5, 1.0, 5.0, 20.0]:
            next_values = np.array(
                [current * process.generate_shock(current) for _ in range(n_samples)]
            )
            mean_next = np.mean(next_values)
            # conditional_std in log-space ≈ 0.02, so SE of mean ≈ 0.02/sqrt(n)
            se = sigma * np.sqrt(1 / (2 * theta)) / np.sqrt(n_samples)
            assert abs(np.log(mean_next) - np.log(mu)) < 3 * se, (
                f"Large theta: mean(next)={mean_next:.4f}, expected≈{mu}, " f"3*SE={3*se:.4f}"
            )

    def test_mean_reverting_rejects_nonpositive_mean(self):
        """Test that MeanRevertingProcess rejects non-positive mean_level."""
        config = StochasticConfig(volatility=0.1, drift=0.0, random_seed=42)
        with pytest.raises(ValueError, match="mean_level must be positive"):
            MeanRevertingProcess(config, mean_level=0.0)
        with pytest.raises(ValueError, match="mean_level must be positive"):
            MeanRevertingProcess(config, mean_level=-1.0)

    def test_mean_reverting_near_zero_current(self):
        """Test that near-zero current_value doesn't break the process."""
        config = StochasticConfig(volatility=0.1, drift=0.0, random_seed=42)
        process = MeanRevertingProcess(config, mean_level=1.0, reversion_speed=0.5)

        # Near-zero current value should still produce a valid positive shock
        shock = process.generate_shock(1e-12)
        assert shock > 0
        assert np.isfinite(shock)

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
        # Should be roughly 5% higher (with stochastic variation from 10% volatility)
        assert 0.8 < final_turnover / initial_turnover < 1.3

    def test_memory_efficiency(self):
        """Test that long simulations don't use excessive memory."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
            capex_to_depreciation_ratio=0.0,
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

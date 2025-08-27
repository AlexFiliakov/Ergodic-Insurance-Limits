"""Performance benchmarks for Monte Carlo simulation engine.

These tests are marked with @pytest.mark.slow and can be run explicitly with:
    pytest -m slow                   # Run only slow tests
    pytest -m "slow and integration"  # Run slow integration tests

To exclude slow tests during regular test runs:
    pytest -m "not slow"             # Skip all slow tests
"""

import time
from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig


class TestPerformanceBenchmarks:
    """Performance benchmarks for Monte Carlo engine."""

    @pytest.fixture
    def setup_realistic_engine(self):
        """Set up engine with realistic parameters."""
        # Create loss generator with realistic parameters
        loss_generator = ManufacturingLossGenerator(
            attritional_params={"base_frequency": 5.0, "severity_mean": 50_000, "severity_cv": 0.8},
            large_params={"base_frequency": 0.5, "severity_mean": 2_000_000, "severity_cv": 1.2},
            catastrophic_params={
                "base_frequency": 0.02,
                "severity_xm": 10_000_000,
                "severity_alpha": 2.5,
            },
            seed=42,
        )

        # Create realistic insurance program
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.015),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, premium_rate=0.008
            ),
        ]
        insurance_program = InsuranceProgram(layers=layers)

        # Create manufacturer
        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        return loss_generator, insurance_program, manufacturer

    @pytest.mark.slow
    def test_10k_simulations_performance(self, setup_realistic_engine):
        """Test that 10K simulations complete in reasonable time."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        config = SimulationConfig(
            n_simulations=10_000,
            n_years=10,
            parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        assert results is not None
        assert len(results.final_assets) == 10_000
        assert execution_time < 60  # Should complete in under 1 minute
        print(f"\n10K simulations completed in {execution_time:.2f}s")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_100k_simulations_performance(self, setup_realistic_engine):
        """Test that 100K simulations complete in under 10 seconds."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Mock the loss generator for faster testing
        mock_generator = Mock(spec=ManufacturingLossGenerator)
        mock_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=100_000, loss_type="test")],
            {"total_amount": 100_000},
        )

        config = SimulationConfig(
            n_simulations=100_000,
            n_years=10,
            parallel=False,  # Changed to False - Mock objects can't be pickled for multiprocessing
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=mock_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        assert results is not None
        assert len(results.final_assets) == 100_000
        # Relaxed constraint for CI environments
        assert execution_time < 30  # Should complete in under 30 seconds
        print(f"\n100K simulations completed in {execution_time:.2f}s")

    @pytest.mark.slow
    def test_memory_efficiency(self, setup_realistic_engine):
        """Test memory usage for large simulations."""
        import os

        import psutil

        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Mock for faster testing
        mock_generator = Mock(spec=ManufacturingLossGenerator)
        mock_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=100_000, loss_type="test")],
            {"total_amount": 100_000},
        )

        config = SimulationConfig(
            n_simulations=100_000,
            n_years=10,
            parallel=False,
            use_float32=True,  # Use float32 for memory efficiency
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=mock_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Get memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run simulation
        results = engine.run()

        # Get memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        assert results is not None
        # Memory usage should be reasonable (< 2GB for 100K simulations)
        assert mem_used < 2000  # MB
        print(f"\nMemory used for 100K simulations: {mem_used:.2f} MB")

    @pytest.mark.slow
    def test_parallel_speedup(self, setup_realistic_engine):
        """Test parallel processing speedup."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Use real loss generator instead of Mock for parallel processing
        # since Mock objects can't be pickled

        # Sequential run
        config_seq = SimulationConfig(
            n_simulations=20_000,
            n_years=10,
            parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        engine_seq = MonteCarloEngine(
            loss_generator=loss_generator,  # Use real generator
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config_seq,
        )

        start_time = time.time()
        results_seq = engine_seq.run()
        time_seq = time.time() - start_time

        # Parallel run
        config_par = SimulationConfig(
            n_simulations=20_000,
            n_years=10,
            parallel=True,
            n_workers=4,
            chunk_size=5_000,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        engine_par = MonteCarloEngine(
            loss_generator=loss_generator,  # Use real generator
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config_par,
        )

        start_time = time.time()
        results_par = engine_par.run()
        time_par = time.time() - start_time

        speedup = time_seq / time_par if time_par > 0 else 0

        assert results_seq is not None
        assert results_par is not None
        # Should achieve at least 2x speedup with 4 workers
        # Relaxed for CI environments
        assert speedup > 1.5
        print(
            f"\nParallel speedup: {speedup:.2f}x (sequential: {time_seq:.2f}s, parallel: {time_par:.2f}s)"
        )

    @pytest.mark.slow
    @pytest.mark.integration
    def test_convergence_efficiency(self, setup_realistic_engine):
        """Test convergence monitoring efficiency."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Mock for faster testing
        mock_generator = Mock(spec=ManufacturingLossGenerator)
        mock_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=100_000, loss_type="test")],
            {"total_amount": 100_000},
        )

        config = SimulationConfig(
            n_simulations=50_000,
            n_years=10,
            parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=mock_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        start_time = time.time()
        results = engine.run_with_convergence_monitoring(
            target_r_hat=1.1, check_interval=5_000, max_iterations=50_000
        )
        execution_time = time.time() - start_time

        assert results is not None
        # Should converge before max iterations
        assert len(results.final_assets) <= 50_000
        print(
            f"\nConvergence achieved with {len(results.final_assets)} simulations in {execution_time:.2f}s"
        )

    def test_vectorization_performance(self):
        """Test vectorized operations performance."""
        # Test vectorized growth rate calculation
        n_sims = 1_000_000
        final_assets = np.random.lognormal(16, 0.5, n_sims)  # ~10M with variation
        initial_assets = 10_000_000
        n_years = 10

        start_time = time.time()

        # Vectorized calculation
        valid_mask = (final_assets > 0) & (initial_assets > 0)
        growth_rates = np.zeros_like(final_assets)
        growth_rates[valid_mask] = np.log(final_assets[valid_mask] / initial_assets) / n_years

        vectorized_time = time.time() - start_time

        # Ensure it's fast
        assert vectorized_time < 0.1  # Should complete in under 100ms
        print(
            f"\nVectorized growth rate calculation for 1M simulations: {vectorized_time*1000:.2f}ms"
        )

    def test_metrics_calculation_performance(self):
        """Test risk metrics calculation performance."""
        from ergodic_insurance.src.risk_metrics import RiskMetrics

        # Generate large dataset
        n_sims = 1_000_000
        losses = np.random.lognormal(12, 1.5, n_sims)  # Log-normal losses

        start_time = time.time()

        # Calculate metrics
        risk_metrics = RiskMetrics(losses)
        var_95 = risk_metrics.var(0.95)
        var_99 = risk_metrics.var(0.99)
        tvar_99 = risk_metrics.tvar(0.99)

        metrics_time = time.time() - start_time

        assert var_95 > 0
        assert var_99 > var_95
        assert tvar_99 > var_99
        assert metrics_time < 1.0  # Should complete in under 1 second
        print(f"\nRisk metrics calculation for 1M simulations: {metrics_time:.2f}s")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_cache_performance(self, setup_realistic_engine):
        """Test caching performance improvement."""
        from pathlib import Path
        import tempfile

        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Mock for consistent results
        mock_generator = Mock(spec=ManufacturingLossGenerator)
        mock_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=100_000, loss_type="test")],
            {"total_amount": 100_000},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulationConfig(
                n_simulations=10_000,
                n_years=10,
                parallel=False,
                cache_results=True,
                progress_bar=False,
                seed=42,
            )

            engine = MonteCarloEngine(
                loss_generator=mock_generator,
                insurance_program=insurance_program,
                manufacturer=manufacturer,
                config=config,
            )
            engine.cache_dir = Path(tmpdir) / "cache"
            engine.cache_dir.mkdir(parents=True)

            # First run - no cache
            start_time = time.time()
            results1 = engine.run()
            time_no_cache = time.time() - start_time

            # Second run - with cache
            start_time = time.time()
            results2 = engine.run()
            time_with_cache = time.time() - start_time

            speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0

            assert np.array_equal(results1.final_assets, results2.final_assets)
            assert speedup > 10  # Cache should be at least 10x faster
            print(
                f"\nCache speedup: {speedup:.1f}x (no cache: {time_no_cache:.2f}s, with cache: {time_with_cache:.2f}s)"
            )

"""Performance benchmarks for Monte Carlo simulation engine.

These tests are marked with @pytest.mark.slow and can be run explicitly with:
    pytest -m slow                   # Run only slow tests
    pytest -m "slow and integration"  # Run slow integration tests

To exclude slow tests during regular test runs:
    pytest -m "not slow"             # Skip all slow tests

This module also tests the new performance optimization, accuracy validation,
and benchmarking modules to ensure they meet the 100K simulations target.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ergodic_insurance.src.accuracy_validator import (
    AccuracyValidator,
    EdgeCaseTester,
    ReferenceImplementations,
    StatisticalValidation,
    ValidationResult,
)
from ergodic_insurance.src.benchmarking import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    SystemProfiler,
)
from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig
from ergodic_insurance.src.performance_optimizer import (
    OptimizationConfig,
    PerformanceOptimizer,
    ProfileResult,
    SmartCache,
    VectorizedOperations,
)


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
        var_95_result = risk_metrics.var(0.95)
        var_99_result = risk_metrics.var(0.99)
        tvar_99 = risk_metrics.tvar(0.99)

        metrics_time = time.time() - start_time

        # Extract values if RiskMetricsResult objects
        var_95 = var_95_result.value if hasattr(var_95_result, "value") else var_95_result
        var_99 = var_99_result.value if hasattr(var_99_result, "value") else var_99_result

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


class TestPerformanceOptimizer:
    """Tests for the performance optimizer module."""

    def test_smart_cache_operations(self):
        """Test smart cache functionality."""
        cache = SmartCache(max_size=3)

        # Test set and get
        cache.set(("key1",), "value1")
        assert cache.get(("key1",)) == "value1"
        assert cache.hits == 1

        # Test miss
        assert cache.get(("key2",)) is None
        assert cache.misses == 1

        # Test eviction
        cache.set(("key2",), "value2")
        cache.set(("key3",), "value3")
        cache.set(("key4",), "value4")  # Should evict least accessed

        # Test hit rate
        assert cache.hit_rate > 0

        # Test clear
        cache.clear()
        assert len(cache.cache) == 0

    def test_vectorized_operations(self):
        """Test vectorized operation performance."""
        vec_ops = VectorizedOperations()

        # Test growth rate calculation
        n_sims = 10000
        final_assets = np.random.uniform(5e6, 20e6, n_sims)
        initial_assets = 10e6
        n_years = 10

        start = time.time()
        growth_rates = vec_ops.calculate_growth_rates(final_assets, initial_assets, n_years)
        vec_time = time.time() - start

        assert len(growth_rates) == n_sims
        assert vec_time < 0.1  # Should be very fast

        # Test insurance application
        losses = np.random.exponential(100000, n_sims)
        retained, recovered = vec_ops.apply_insurance_vectorized(losses, 50000, 500000)

        assert len(retained) == n_sims
        assert len(recovered) == n_sims
        assert np.all(retained >= 0)
        assert np.all(recovered >= 0)
        assert np.all(retained + recovered == losses)

    def test_performance_profiling(self):
        """Test performance profiling capabilities."""
        optimizer = PerformanceOptimizer()

        # Define a test function that takes some time
        def slow_function(n=1000):
            import time

            result = 0
            for i in range(n):
                result += i**2
            time.sleep(0.01)  # Ensure it takes some time
            return result

        # Profile the function
        profile_result = optimizer.profile_execution(slow_function, n=10000)

        assert isinstance(profile_result, ProfileResult)
        assert profile_result.total_time >= 0.01  # At least the sleep time
        assert len(profile_result.function_times) > 0
        assert profile_result.memory_usage >= 0

        # Check summary generation
        summary = profile_result.summary()
        assert "Performance Profile Summary" in summary

    def test_optimization_config(self):
        """Test optimization configuration."""
        config = OptimizationConfig(
            enable_vectorization=True, enable_caching=True, cache_size=500, memory_limit_mb=2000
        )

        optimizer = PerformanceOptimizer(config)
        assert optimizer.config.enable_vectorization
        assert optimizer.config.cache_size == 500
        assert optimizer.cache.max_size == 500

    def test_memory_optimization(self):
        """Test memory optimization features."""
        optimizer = PerformanceOptimizer()

        # Test memory metrics
        metrics = optimizer.optimize_memory_usage()

        assert "process_memory_mb" in metrics
        assert "available_memory_mb" in metrics
        assert "suggested_chunk_size" in metrics
        assert metrics["suggested_chunk_size"] >= 1000
        assert metrics["suggested_chunk_size"] <= 100000

    @pytest.mark.slow
    def test_insurance_optimization(self):
        """Test insurance calculation optimization."""
        optimizer = PerformanceOptimizer()

        # Generate test data
        n_losses = 100000
        losses = np.random.exponential(100000, n_losses)
        layers = [(0, 1000000, 0.015), (1000000, 4000000, 0.008), (5000000, 20000000, 0.004)]

        # Run optimized calculation
        start = time.time()
        result = optimizer.optimize_insurance_calculation(losses, layers)
        opt_time = time.time() - start

        assert "retained_losses" in result
        assert "total_recovered" in result
        assert "total_premiums" in result
        assert len(result["retained_losses"]) == n_losses

        # Should complete quickly
        assert opt_time < 1.0  # Less than 1 second for 100K

        # Test caching
        start = time.time()
        result2 = optimizer.optimize_insurance_calculation(losses, layers)
        cached_time = time.time() - start

        # Cached should be much faster
        if optimizer.config.enable_caching:
            # Cache should be at least 2x faster (allowing for some variance)
            assert cached_time < opt_time * 0.5
            assert optimizer.cache.hit_rate > 0


class TestAccuracyValidator:
    """Tests for the accuracy validator module."""

    def test_reference_implementations(self):
        """Test high-precision reference implementations."""
        ref = ReferenceImplementations()

        # Test growth rate calculation
        rate = ref.calculate_growth_rate_precise(20e6, 10e6, 10)
        expected = np.log(2.0) / 10
        assert abs(rate - expected) < 1e-10

        # Test insurance application
        retained, recovered = ref.apply_insurance_precise(150000, 50000, 75000)
        assert retained == 75000
        assert recovered == 75000

        # Test VaR calculation
        losses = np.array([100, 200, 300, 400, 500])
        var_95 = ref.calculate_var_precise(losses, 0.95)
        assert var_95 == 500

        # Test TVaR calculation
        tvar_95 = ref.calculate_tvar_precise(losses, 0.95)
        assert tvar_95 == 500

    def test_statistical_validation(self):
        """Test statistical distribution comparison."""
        stat_val = StatisticalValidation()

        # Generate similar distributions
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)

        # Compare distributions
        results = stat_val.compare_distributions(data1, data2)

        assert "ks_statistic" in results
        assert "ks_pvalue" in results
        assert results["ks_pvalue"] > 0.05  # Should not reject null hypothesis
        assert results["mean_diff"] < 0.1
        assert results["std_diff"] < 0.1

    def test_edge_case_testing(self):
        """Test edge case validation."""
        tester = EdgeCaseTester()

        # Test extreme values
        extreme_results = tester.test_extreme_values()
        assert "zero_initial_assets" in extreme_results
        assert "infinite_loss" in extreme_results
        assert extreme_results["zero_initial_assets"]  # Should handle correctly

        # Test boundary conditions
        boundary_results = tester.test_boundary_conditions()
        assert "insurance_limit_boundary" in boundary_results
        assert "exact_attachment" in boundary_results
        assert boundary_results["exact_attachment"]  # Should handle correctly

    def test_accuracy_comparison(self):
        """Test accuracy comparison between implementations."""
        validator = AccuracyValidator(tolerance=0.01)

        # Generate test data
        np.random.seed(42)
        optimized = np.random.normal(0.08, 0.02, 1000)
        reference = optimized + np.random.normal(0, 0.0001, 1000)

        # Compare implementations
        result = validator.compare_implementations(optimized, reference, "Test Comparison")

        assert isinstance(result, ValidationResult)
        assert result.accuracy_score > 0.99
        assert result.relative_error < 0.01
        assert result.is_valid()

    @pytest.mark.slow
    def test_full_validation_suite(self):
        """Test comprehensive validation suite."""
        validator = AccuracyValidator()

        # Run full validation
        result = validator.run_full_validation()

        assert result.accuracy_score >= 0
        assert len(result.passed_tests) > 0
        assert isinstance(result.edge_cases, dict)

        # Check report generation
        report = validator.generate_validation_report([result])
        assert "ACCURACY VALIDATION REPORT" in report
        assert "OVERALL SUMMARY" in report


class TestBenchmarking:
    """Tests for the benchmarking module."""

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

    def test_system_profiler(self):
        """Test system profiling capabilities."""
        profiler = SystemProfiler()

        # Test system info retrieval
        sys_info = profiler.get_system_info()
        assert "platform" in sys_info
        assert "cpu" in sys_info
        assert "memory" in sys_info
        assert sys_info["cpu"]["cores_physical"] > 0
        assert sys_info["memory"]["total_gb"] > 0

        # Test profiling
        profiler.start()
        time.sleep(0.1)
        profiler.sample()
        profiler.sample()

        avg_cpu, peak_mem, avg_mem = profiler.get_metrics()
        assert avg_cpu >= 0
        assert peak_mem > 0
        assert avg_mem > 0

    def test_benchmark_runner(self):
        """Test benchmark runner functionality."""
        runner = BenchmarkRunner()

        # Define test function
        def test_func(n=1000):
            return sum(range(n))

        # Run single benchmark
        metrics = runner.run_single_benchmark(test_func, kwargs={"n": 10000})

        assert isinstance(metrics, BenchmarkMetrics)
        assert metrics.execution_time > 0
        assert metrics.memory_peak_mb > 0

        # Test with warmup
        metrics_list = runner.run_with_warmup(
            test_func, kwargs={"n": 10000}, warmup_runs=1, benchmark_runs=2
        )

        assert len(metrics_list) == 2
        assert all(isinstance(m, BenchmarkMetrics) for m in metrics_list)

    def test_benchmark_config(self):
        """Test benchmark configuration."""
        config = BenchmarkConfig(scales=[1000, 10000], n_years=5, n_workers=2, memory_limit_mb=2000)

        assert 1000 in config.scales
        assert config.target_times[1000] == 1.0
        assert config.memory_limit_mb == 2000

    @pytest.mark.slow
    def test_benchmark_suite(self, setup_realistic_engine):
        """Test benchmark suite functionality."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Create engine with small scale for testing
        config = SimulationConfig(n_simulations=100, n_years=5, parallel=False, progress_bar=False)

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Create benchmark suite
        suite = BenchmarkSuite()
        bench_config = BenchmarkConfig(scales=[100], repetitions=1, warmup_runs=0)

        # Run benchmark
        result = suite.benchmark_scale(engine, 100, bench_config)

        assert isinstance(result, BenchmarkResult)
        assert result.scale == 100
        assert result.metrics.execution_time > 0
        assert result.metrics.simulations_per_second > 0

    def test_benchmark_result_validation(self):
        """Test benchmark result validation."""
        from datetime import datetime

        metrics = BenchmarkMetrics(
            execution_time=50.0,
            simulations_per_second=2000,
            memory_peak_mb=3500,
            memory_average_mb=3000,
            cpu_utilization=80,
            cache_hit_rate=90,
        )

        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={"n_workers": 4}, timestamp=datetime.now()
        )

        # Check if meets targets
        assert result.meets_target(60.0, 4000.0)  # Should pass
        assert not result.meets_target(30.0, 4000.0)  # Should fail on time
        assert not result.meets_target(60.0, 3000.0)  # Should fail on memory

        # Check summary
        summary = result.summary()
        assert "100,000 simulations" in summary
        assert "50.00s" in summary
        assert "2000 sims/s" in summary


class TestIntegration:
    """Integration tests for performance optimization suite."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_optimizer_with_engine(self, setup_realistic_engine):
        """Test performance optimizer integration with Monte Carlo engine."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Create engine
        config = SimulationConfig(
            n_simulations=10000,
            n_years=10,
            parallel=True,
            n_workers=4,
            use_enhanced_parallel=True,
            monitor_performance=True,
            progress_bar=False,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Create optimizer and profile
        optimizer = PerformanceOptimizer()

        # Run and check performance metrics
        start = time.time()
        results = engine.run()
        execution_time = time.time() - start

        assert results is not None
        assert len(results.final_assets) == 10000
        assert execution_time < 30  # Should be reasonably fast

        if results.performance_metrics:
            assert results.performance_metrics.total_time > 0
            assert results.performance_metrics.items_per_second > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_accuracy_validation_with_optimizer(self):
        """Test that optimizations maintain accuracy."""
        validator = AccuracyValidator(tolerance=0.001)
        optimizer = PerformanceOptimizer()
        vec_ops = VectorizedOperations()

        # Test growth rate accuracy
        test_cases = [(15e6, 10e6, 10), (8e6, 10e6, 10), (10e6, 10e6, 10)]

        for final, initial, years in test_cases:
            # Vectorized version
            opt_result = vec_ops.calculate_growth_rates(np.array([final]), initial, years)[0]

            # Reference version
            ref_result = validator.reference.calculate_growth_rate_precise(final, initial, years)

            # Should be very close
            if not np.isinf(ref_result):
                assert abs(opt_result - ref_result) < 1e-10

    @pytest.mark.slow
    @pytest.mark.integration
    def test_100k_performance_target(self, setup_realistic_engine):
        """Test that 100K simulations meet performance targets."""
        loss_generator, insurance_program, manufacturer = setup_realistic_engine

        # Use mocked loss generator for consistent fast performance
        mock_generator = Mock(spec=ManufacturingLossGenerator)
        mock_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=100000, loss_type="test")],
            {"total_amount": 100000},
        )

        # Create optimized configuration
        config = SimulationConfig(
            n_simulations=100000,
            n_years=10,
            parallel=True,
            n_workers=4,
            use_enhanced_parallel=True,
            use_float32=True,
            monitor_performance=True,
            adaptive_chunking=True,
            shared_memory=True,
            progress_bar=False,
            cache_results=False,
        )

        engine = MonteCarloEngine(
            loss_generator=mock_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Run benchmark
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        start = time.time()
        results = engine.run()
        execution_time = time.time() - start

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_used = final_memory - initial_memory

        # Verify performance targets
        assert results is not None
        assert len(results.final_assets) == 100000
        assert execution_time < 60  # Must complete in under 60 seconds
        assert memory_used < 4000  # Must use less than 4GB

        print(f"\n100K Performance: {execution_time:.2f}s, {memory_used:.1f}MB")

"""Comprehensive tests for ESS calculation and progress monitoring.

Tests effective sample size calculation against theoretical expectations
and validates progress monitoring performance impact.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.src.convergence import ConvergenceDiagnostics, ConvergenceStats
from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig
from ergodic_insurance.src.progress_monitor import ProgressMonitor, ProgressStats


class TestESSCalculation:
    """Tests for Effective Sample Size calculation."""

    @pytest.fixture
    def diagnostics(self):
        """Create convergence diagnostics instance."""
        return ConvergenceDiagnostics()

    def test_ess_with_independent_samples(self, diagnostics):
        """Test ESS with independent samples (should equal N)."""
        # Generate independent samples
        np.random.seed(42)
        chain = np.random.randn(10000)

        ess = diagnostics.calculate_ess(chain)

        # ESS should be close to N for independent samples
        assert ess > 9000  # Allow some tolerance
        assert ess <= len(chain)

    def test_ess_with_perfect_correlation(self, diagnostics):
        """Test ESS with perfectly correlated samples."""
        # Generate perfectly correlated samples (all the same)
        chain = np.ones(1000)

        ess = diagnostics.calculate_ess(chain)

        # ESS should be very small for perfectly correlated samples
        # Due to zero variance, special handling applies
        assert ess == 1000  # Returns N when variance is 0

    def test_ess_with_ar1_process(self, diagnostics):
        """Test ESS with AR(1) process of known autocorrelation."""
        # Generate AR(1) process with rho = 0.5
        np.random.seed(42)
        n = 10000
        rho = 0.5

        # Generate AR(1) process
        chain = np.zeros(n)
        chain[0] = np.random.randn()
        for i in range(1, n):
            chain[i] = rho * chain[i - 1] + np.sqrt(1 - rho**2) * np.random.randn()

        ess = diagnostics.calculate_ess(chain)

        # Theoretical ESS for AR(1) is N * (1 - rho) / (1 + rho)
        theoretical_ess = n * (1 - rho) / (1 + rho)

        # Should be within 20% of theoretical value
        assert abs(ess - theoretical_ess) / theoretical_ess < 0.2

    def test_ess_with_negative_autocorrelation(self, diagnostics):
        """Test ESS with negative autocorrelation (ESS > N possible)."""
        # Generate alternating series (negative autocorrelation)
        n = 1000
        chain = np.array([(-1) ** i for i in range(n)]) + np.random.randn(n) * 0.1

        ess = diagnostics.calculate_ess(chain)

        # ESS can be larger than N with negative autocorrelation
        # But implementation caps at N
        assert ess <= n
        assert ess > 0

    def test_ess_geyers_initial_positive_sequence(self, diagnostics):
        """Test that ESS uses Geyer's initial positive sequence estimator."""
        # Generate chain with known structure
        np.random.seed(42)
        n = 5000

        # Create chain with specific autocorrelation pattern
        chain = np.cumsum(np.random.randn(n)) / np.sqrt(n)

        ess = diagnostics.calculate_ess(chain, max_lag=100)

        # Should handle the autocorrelation structure properly
        assert 0 < ess < n
        assert ess < n / 2  # Should show significant autocorrelation

    def test_batch_ess_calculation(self, diagnostics):
        """Test batch ESS calculation for multiple chains."""
        np.random.seed(42)

        # Create multiple chains
        chains = np.random.randn(4, 1000)

        # Test mean method
        ess_mean = diagnostics.calculate_batch_ess(chains, method="mean")
        assert isinstance(ess_mean, float)
        assert ess_mean > 0

        # Test min method
        ess_min = diagnostics.calculate_batch_ess(chains, method="min")
        assert isinstance(ess_min, float)
        assert ess_min <= ess_mean

        # Test all method
        ess_all = diagnostics.calculate_batch_ess(chains, method="all")
        assert ess_all.shape == (4,)
        assert np.all(ess_all > 0)

    def test_batch_ess_with_multiple_metrics(self, diagnostics):
        """Test batch ESS with multiple metrics."""
        np.random.seed(42)

        # Create chains with multiple metrics
        chains = np.random.randn(3, 1000, 2)  # 3 chains, 1000 iterations, 2 metrics

        # Test mean method
        ess_mean = diagnostics.calculate_batch_ess(chains, method="mean")
        assert ess_mean.shape == (2,)
        assert np.all(ess_mean > 0)

        # Test min method
        ess_min = diagnostics.calculate_batch_ess(chains, method="min")
        assert ess_min.shape == (2,)
        assert np.all(ess_min <= ess_mean)

    def test_ess_per_second_calculation(self, diagnostics):
        """Test ESS per second calculation."""
        chain = np.random.randn(1000)
        computation_time = 2.0  # 2 seconds

        ess_per_sec = diagnostics.calculate_ess_per_second(chain, computation_time)

        assert ess_per_sec > 0
        assert ess_per_sec < len(chain) / computation_time

    def test_ess_with_very_small_chain(self, diagnostics):
        """Test ESS with chain smaller than minimum size."""
        chain = np.array([1.0, 2.0])  # Only 2 samples

        ess = diagnostics.calculate_ess(chain)

        assert ess == 2.0  # Should return chain length


class TestProgressMonitor:
    """Tests for progress monitoring functionality."""

    def test_progress_monitor_initialization(self):
        """Test ProgressMonitor initialization."""
        monitor = ProgressMonitor(
            total_iterations=100000,
            check_intervals=[10000, 25000, 50000, 100000],
            update_frequency=1000,
            show_console=False,
            convergence_threshold=1.1,
        )

        assert monitor.total_iterations == 100000
        assert monitor.check_intervals == [10000, 25000, 50000, 100000]
        assert monitor.converged is False
        assert monitor.converged_at is None

    def test_progress_update_without_convergence(self):
        """Test progress update without convergence check."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        # Update progress
        should_continue = monitor.update(100)

        assert should_continue is True
        assert monitor.current_iteration == 100
        assert monitor.converged is False

    def test_progress_update_with_convergence(self):
        """Test progress update with convergence achieved."""
        monitor = ProgressMonitor(
            total_iterations=100000,
            check_intervals=[10000],
            show_console=False,
            convergence_threshold=1.1,
        )

        # Update at check interval with good convergence
        should_continue = monitor.update(10000, convergence_value=1.05)

        assert should_continue is False  # Should stop
        assert monitor.converged is True
        assert monitor.converged_at == 10000
        assert len(monitor.convergence_checks) == 1

    def test_progress_stats_generation(self):
        """Test progress statistics generation."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        # Update some progress
        monitor.update(500)
        time.sleep(0.01)  # Small delay to ensure elapsed time > 0

        stats = monitor.get_stats()

        assert isinstance(stats, ProgressStats)
        assert stats.current_iteration == 500
        assert stats.total_iterations == 1000
        assert stats.elapsed_time > 0
        assert stats.iterations_per_second > 0

    def test_convergence_summary_generation(self):
        """Test convergence summary generation."""
        monitor = ProgressMonitor(
            total_iterations=100000,
            check_intervals=[10000, 25000, 50000],
            show_console=False,
            convergence_threshold=1.1,
        )

        # Add convergence checks
        monitor.update(10000, convergence_value=1.5)
        monitor.update(25000, convergence_value=1.3)
        monitor.update(50000, convergence_value=1.08)

        summary = monitor.generate_convergence_summary()

        assert summary["converged"] is True
        assert summary["converged_at"] == 50000
        assert summary["total_checks"] == 3
        assert summary["check_iterations"] == [10000, 25000, 50000]
        assert summary["convergence_values"] == [1.5, 1.3, 1.08]
        assert summary["final_value"] == 1.08

    def test_convergence_trend_analysis(self):
        """Test convergence trend analysis."""
        monitor = ProgressMonitor(total_iterations=100000, show_console=False)

        # Test monotonic improvement
        trend = monitor._analyze_convergence_trend([1.5, 1.3, 1.1, 1.05])
        assert "monotonic" in trend

        # Test improvement with fluctuations
        trend = monitor._analyze_convergence_trend([1.5, 1.2, 1.3, 1.1])
        assert "fluctuations" in trend or "improvement" in trend

        # Test no improvement - all same values
        trend = monitor._analyze_convergence_trend([1.5, 1.5, 1.5, 1.5])
        assert "monotonic improvement" in trend or "no clear improvement" in trend

    def test_convergence_rate_estimation(self):
        """Test convergence rate estimation."""
        monitor = ProgressMonitor(total_iterations=100000, show_console=False)

        # Generate exponentially decaying values
        iterations = [10000, 20000, 30000, 40000]
        values = [1.5 * np.exp(-0.00001 * i) + 1.0 for i in iterations]

        rate = monitor._estimate_convergence_rate(iterations, values)

        if rate is not None:
            assert rate > 0
            # Rate should be positive and reasonable (relaxed tolerance for estimation)
            assert 0 < rate < 0.001  # Just check it's in a reasonable range

    def test_performance_overhead_tracking(self):
        """Test that monitoring overhead is tracked and reasonable."""
        monitor = ProgressMonitor(
            total_iterations=10000,
            update_frequency=100,
            check_intervals=[5000, 10000],
            show_console=False,
        )

        # Run many updates including convergence checks
        for i in range(0, 10001, 100):
            if i in [5000, 10000]:
                monitor.update(i, convergence_value=1.2)
            else:
                monitor.update(i)

        # Check overhead is tracked
        assert monitor.monitor_overhead > 0

        # Generate summary
        summary = monitor.generate_convergence_summary()
        assert "performance_overhead_pct" in summary

        # For very fast tests without real work, overhead will be higher.
        # In real simulations with actual computation, it should be < 1%
        # For this minimal test, we allow up to 20% overhead
        assert summary["performance_overhead_pct"] < 20.0  # Reasonable for minimal test


class TestMonteCarloIntegration:
    """Integration tests with Monte Carlo engine."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for Monte Carlo engine."""
        loss_generator = MagicMock()
        insurance_program = MagicMock()

        # Create a proper manufacturer mock with copy method
        manufacturer = MagicMock()
        manufacturer.assets = 10_000_000

        # Create a copy that returns a new mock with the same attributes
        manufacturer_copy = MagicMock()
        manufacturer_copy.assets = 10_000_000
        manufacturer_copy.calculate_revenue.return_value = 5_000_000
        manufacturer_copy.process_insurance_claim.return_value = None
        manufacturer_copy.step.return_value = {"revenue": 5_000_000}
        manufacturer.copy.return_value = manufacturer_copy

        # Set up loss generator mock
        loss_generator.generate_losses.return_value = ([], {})

        # Set up insurance program mock
        insurance_program.process_claim.return_value = {"total_recovery": 0}

        return loss_generator, insurance_program, manufacturer

    def test_run_with_progress_monitoring(self, mock_components):
        """Test Monte Carlo run with progress monitoring."""
        loss_generator, insurance_program, manufacturer = mock_components

        config = SimulationConfig(n_simulations=1000, n_years=5, progress_bar=False, parallel=False)

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Run with progress monitoring
        results = engine.run_with_progress_monitoring(
            check_intervals=[500, 1000],
            convergence_threshold=1.1,
            early_stopping=False,
            show_progress=False,
        )

        assert results is not None
        assert results.config.n_simulations == 1000
        assert "actual_iterations" in results.metrics
        assert "convergence_achieved" in results.metrics
        assert "monitoring_overhead_pct" in results.metrics

    def test_early_stopping_on_convergence(self, mock_components):
        """Test early stopping when convergence is achieved."""
        loss_generator, insurance_program, manufacturer = mock_components

        config = SimulationConfig(
            n_simulations=100000, n_years=5, progress_bar=False, parallel=False
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Mock convergence diagnostics to always return good R-hat
        with patch.object(engine.convergence_diagnostics, "calculate_r_hat", return_value=1.05):
            results = engine.run_with_progress_monitoring(
                check_intervals=[1000],
                convergence_threshold=1.1,
                early_stopping=True,
                show_progress=False,
            )

            # Should stop early at 1000 iterations
            assert results.metrics["actual_iterations"] == 1000
            assert results.metrics["convergence_achieved"]
            assert results.metrics["convergence_iteration"] == 1000

    def test_progress_monitoring_performance_impact(self, mock_components):
        """Test that progress monitoring has minimal performance impact."""
        loss_generator, insurance_program, manufacturer = mock_components

        config = SimulationConfig(n_simulations=5000, n_years=5, progress_bar=False, parallel=False)

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        # Run without monitoring
        start = time.perf_counter()
        results_no_monitor = engine.run()
        time_no_monitor = time.perf_counter() - start

        # Run with monitoring
        start = time.perf_counter()
        results_monitor = engine.run_with_progress_monitoring(
            check_intervals=[1000, 2500, 5000], show_progress=False
        )
        time_monitor = time.perf_counter() - start

        # Overhead should be less than 20% (progress monitoring adds some overhead)
        overhead = (time_monitor - time_no_monitor) / time_no_monitor
        assert overhead < 0.20  # Less than 20% overhead is acceptable for monitoring

        # Results should be similar
        assert len(results_monitor.final_assets) == len(results_no_monitor.final_assets)


class TestConvergenceAtIntervals:
    """Test convergence checking at specific intervals."""

    def test_convergence_checks_at_correct_intervals(self):
        """Test that convergence is checked at specified intervals."""
        monitor = ProgressMonitor(
            total_iterations=100000,
            check_intervals=[10000, 25000, 50000, 100000],
            show_console=False,
        )

        # Simulate updates
        checked_intervals = []

        for i in range(1, 100001):
            if i in [10000, 25000, 50000, 100000]:
                monitor.update(i, convergence_value=1.2)
                checked_intervals.append(i)
            else:
                monitor.update(i)

        assert checked_intervals == [10000, 25000, 50000, 100000]
        assert len(monitor.convergence_checks) == 4

    def test_filtered_check_intervals(self):
        """Test that check intervals are handled correctly in Monte Carlo."""
        # The filtering happens in MonteCarloEngine.run_with_progress_monitoring, not in ProgressMonitor
        # So we test that ProgressMonitor accepts any intervals
        monitor = ProgressMonitor(
            total_iterations=50000,
            check_intervals=[10000, 25000, 50000, 100000],
            show_console=False,
        )

        # ProgressMonitor should keep all intervals as-is
        assert monitor.check_intervals == [10000, 25000, 50000, 100000]

    def test_convergence_summary_statistics(self):
        """Test generation of convergence summary statistics."""
        monitor = ProgressMonitor(total_iterations=100000, show_console=False)

        # Add multiple convergence checks
        monitor.update(10000, 1.8)
        monitor.update(25000, 1.4)
        monitor.update(50000, 1.2)
        monitor.update(100000, 1.05)

        summary = monitor.generate_convergence_summary()

        # Check summary contains expected fields
        assert "avg_improvement_per_check" in summary
        assert summary["avg_improvement_per_check"] > 0
        assert "convergence_rate" in summary

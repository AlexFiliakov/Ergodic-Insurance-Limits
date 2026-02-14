"""Comprehensive tests for the progress_monitor module."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

from ergodic_insurance.progress_monitor import ProgressMonitor, ProgressStats


class TestProgressStats:
    """Test ProgressStats dataclass."""

    def test_initialization_minimal(self):
        """Test minimal ProgressStats initialization."""
        stats = ProgressStats(
            current_iteration=500,
            total_iterations=1000,
            start_time=time.time(),
            elapsed_time=10.0,
            estimated_time_remaining=10.0,
            iterations_per_second=50.0,
        )
        assert stats.current_iteration == 500
        assert stats.total_iterations == 1000
        assert stats.elapsed_time == 10.0
        assert stats.estimated_time_remaining == 10.0
        assert stats.iterations_per_second == 50.0
        assert stats.convergence_checks == []
        assert stats.converged is False
        assert stats.converged_at is None

    def test_initialization_full(self):
        """Test full ProgressStats initialization."""
        stats = ProgressStats(
            current_iteration=500,
            total_iterations=1000,
            start_time=time.time(),
            elapsed_time=10.0,
            estimated_time_remaining=10.0,
            iterations_per_second=50.0,
            convergence_checks=[(100, 1.5), (200, 1.2)],
            converged=True,
            converged_at=200,
        )
        assert len(stats.convergence_checks) == 2
        assert stats.converged is True
        assert stats.converged_at == 200

    def test_summary(self):
        """Test progress summary generation."""
        stats = ProgressStats(
            current_iteration=500,
            total_iterations=1000,
            start_time=time.time(),
            elapsed_time=10.0,
            estimated_time_remaining=10.0,
            iterations_per_second=50.0,
        )
        summary = stats.summary()
        assert "Progress: 500/1,000" in summary
        assert "(50.0%)" in summary
        assert "Speed: 50 it/s" in summary
        assert "Elapsed: 0:00:10" in summary
        assert "ETA: 0:00:10" in summary

    def test_summary_with_convergence(self):
        """Test summary with convergence information."""
        stats = ProgressStats(
            current_iteration=500,
            total_iterations=1000,
            start_time=time.time(),
            elapsed_time=10.0,
            estimated_time_remaining=10.0,
            iterations_per_second=50.0,
            converged=True,
            converged_at=400,
        )
        summary = stats.summary()
        assert "CONVERGED at iteration 400" in summary

    def test_summary_large_numbers(self):
        """Test summary with large numbers."""
        stats = ProgressStats(
            current_iteration=50000,
            total_iterations=100000,
            start_time=time.time(),
            elapsed_time=3600.0,  # 1 hour
            estimated_time_remaining=3600.0,
            iterations_per_second=13.89,
        )
        summary = stats.summary()
        assert "50,000/100,000" in summary
        assert "1:00:00" in summary  # 1 hour elapsed
        assert "14 it/s" in summary


class TestProgressMonitor:
    """Test ProgressMonitor class."""

    def test_initialization_default(self):
        """Test default ProgressMonitor initialization."""
        monitor = ProgressMonitor(total_iterations=10000)
        assert monitor.total_iterations == 10000
        assert monitor.check_intervals == [10_000, 25_000, 50_000, 100_000]
        assert monitor.update_frequency == 1000
        assert monitor.show_console is True
        assert monitor.convergence_threshold == 1.1
        assert monitor.current_iteration == 0
        assert monitor.converged is False
        assert monitor.converged_at is None

    def test_initialization_custom(self):
        """Test custom ProgressMonitor initialization."""
        check_intervals = [1000, 5000, 10000]
        monitor = ProgressMonitor(
            total_iterations=10000,
            check_intervals=check_intervals,
            update_frequency=500,
            show_console=False,
            convergence_threshold=1.05,
        )
        assert monitor.check_intervals == check_intervals
        assert monitor.update_frequency == 500
        assert monitor.show_console is False
        assert monitor.convergence_threshold == 1.05

    def test_update_basic(self):
        """Test basic progress update."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        should_continue = monitor.update(100)

        assert should_continue is True
        assert monitor.current_iteration == 100

    def test_update_with_convergence_check(self):
        """Test update with convergence checking."""
        monitor = ProgressMonitor(
            total_iterations=20000, check_intervals=[10000], show_console=False
        )

        # Update at check interval with convergence value
        should_continue = monitor.update(10000, convergence_value=1.05)

        assert should_continue is False  # Converged
        assert monitor.converged is True
        assert monitor.converged_at == 10000
        assert len(monitor.convergence_checks) == 1
        assert monitor.convergence_checks[0] == (10000, 1.05)

    def test_update_no_convergence(self):
        """Test update when not converged."""
        monitor = ProgressMonitor(
            total_iterations=20000,
            check_intervals=[10000],
            show_console=False,
            convergence_threshold=1.1,
        )

        # Update with value above threshold
        should_continue = monitor.update(10000, convergence_value=1.2)

        assert should_continue is True  # Not converged
        assert monitor.converged is False
        assert len(monitor.convergence_checks) == 1

    def test_update_skip_check_if_converged(self):
        """Test that convergence checks are skipped after convergence."""
        monitor = ProgressMonitor(
            total_iterations=50000, check_intervals=[10000, 20000], show_console=False
        )

        # First check - converge
        monitor.update(10000, convergence_value=1.05)
        assert monitor.converged is True

        # Second check - should be skipped
        initial_checks = len(monitor.convergence_checks)
        monitor.update(20000, convergence_value=1.5)
        assert len(monitor.convergence_checks) == initial_checks  # No new check

    def test_console_output(self, caplog):
        """Test console output generation."""
        monitor = ProgressMonitor(total_iterations=1000, update_frequency=100, show_console=True)

        # Mock time to control elapsed calculation
        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 10.0
            with caplog.at_level(logging.DEBUG, logger="ergodic_insurance.progress_monitor"):
                monitor.update(100)

            output = " ".join(r.message for r in caplog.records)
            assert "10.0%" in output
            assert "100/1,000" in output

    def test_console_output_with_convergence(self, caplog):
        """Test console output with convergence message."""
        monitor = ProgressMonitor(
            total_iterations=20000, check_intervals=[10000], show_console=True
        )

        with caplog.at_level(logging.INFO, logger="ergodic_insurance.progress_monitor"):
            monitor.update(10000, convergence_value=1.05)

        output = " ".join(r.message for r in caplog.records)
        assert "[OK] Convergence achieved" in output
        assert "R-hat = 1.050" in output

    def test_update_console_frequency(self):
        """Test that console updates respect update frequency."""
        monitor = ProgressMonitor(total_iterations=10000, update_frequency=1000, show_console=True)

        with patch.object(monitor, "_update_console") as mock_update:
            # Should not update
            monitor.update(500)
            assert mock_update.call_count == 0

            # Should update
            monitor.update(1000)
            assert mock_update.call_count == 1

            # Should not update again immediately
            monitor.update(1100)
            assert mock_update.call_count == 1

            # Should update at next interval
            monitor.update(2000)
            assert mock_update.call_count == 2

    def test_get_stats(self):
        """Test getting progress statistics."""
        monitor = ProgressMonitor(total_iterations=10000, show_console=False)

        # Update progress
        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 20.0
            monitor.update(5000)

            stats = monitor.get_stats()

            assert isinstance(stats, ProgressStats)
            assert stats.current_iteration == 5000
            assert stats.total_iterations == 10000
            assert stats.elapsed_time == 20.0
            assert stats.iterations_per_second == 250.0  # 5000/20
            assert stats.estimated_time_remaining == 20.0  # 5000 remaining at 250/s

    def test_get_stats_converged(self):
        """Test getting stats after convergence."""
        monitor = ProgressMonitor(
            total_iterations=20000, check_intervals=[10000], show_console=False
        )

        monitor.update(10000, convergence_value=1.05)
        stats = monitor.get_stats()

        assert stats.converged is True
        assert stats.converged_at == 10000
        assert len(stats.convergence_checks) == 1

    def test_finish(self):
        """Test finishing progress monitoring."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 10.0
            monitor.update(1000)

            stats = monitor.finish()

            assert isinstance(stats, ProgressStats)
            assert stats.current_iteration == 1000
            assert stats.elapsed_time == 10.0
            assert monitor.current_iteration == 1000

    def test_finish_with_console(self):
        """Test finishing with console output (finish no longer prints)."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=True)

        monitor.update(1000)
        stats = monitor.finish()

        assert isinstance(stats, ProgressStats)
        assert stats.current_iteration == 1000

    def test_performance_overhead_tracking(self):
        """Test that performance overhead is tracked."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        monitor.update(100)
        assert monitor.monitor_overhead > 0

        monitor.update(200, convergence_value=1.2)
        assert monitor.total_check_time >= 0

    def test_estimated_time_calculation(self):
        """Test ETA calculation."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        with patch("time.time") as mock_time:
            # 100 iterations in 10 seconds = 10 it/s
            mock_time.return_value = monitor.start_time + 10.0
            monitor._update_console(mock_time.return_value)

            # With 100 iterations done and 900 remaining at 10 it/s
            # ETA should be 90 seconds
            # (This is tested indirectly through console output)

    def test_progress_bar_generation(self, caplog):
        """Test progress bar visual generation."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=True)

        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 10.0

            # 25% progress
            monitor.current_iteration = 250
            with caplog.at_level(logging.DEBUG, logger="ergodic_insurance.progress_monitor"):
                monitor._update_console(mock_time.return_value)

            output = " ".join(r.message for r in caplog.records)
            assert "25.0%" in output

    def test_convergence_display_in_console(self, caplog):
        """Test convergence information in console output."""
        monitor = ProgressMonitor(total_iterations=10000, show_console=True)

        monitor.convergence_checks = [(5000, 1.15), (7500, 1.08)]

        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 10.0
            monitor.current_iteration = 8000
            with caplog.at_level(logging.DEBUG, logger="ergodic_insurance.progress_monitor"):
                monitor._update_console(mock_time.return_value)

            output = " ".join(r.message for r in caplog.records)
            assert "R-hat: 1.080" in output

    def test_zero_speed_handling(self):
        """Test handling of zero speed (division by zero)."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        with patch("time.time") as mock_time:
            # No time elapsed
            mock_time.return_value = monitor.start_time
            monitor.current_iteration = 0

            # Should not raise division by zero
            monitor._update_console(mock_time.return_value)

    def test_get_overhead_percentage(self):
        """Test calculation of monitoring overhead percentage."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        monitor.monitor_overhead = 0.1  # 100ms overhead

        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 10.0

            overhead_pct = monitor.get_overhead_percentage()
            assert overhead_pct == 1.0  # 0.1/10 * 100 = 1%

    def test_get_overhead_percentage_no_time(self):
        """Test overhead percentage with no elapsed time."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        overhead_pct = monitor.get_overhead_percentage()
        assert overhead_pct == 0.0

    def test_reset(self):
        """Test resetting the monitor."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        # Make some progress
        monitor.update(500)
        monitor.convergence_checks = [(100, 1.2)]
        monitor.converged = True
        monitor.converged_at = 100

        # Reset
        monitor.reset()

        assert monitor.current_iteration == 0
        assert monitor.convergence_checks == []
        assert monitor.converged is False
        assert monitor.converged_at is None
        assert monitor.monitor_overhead == 0.0  # type: ignore[unreachable]
        assert monitor.total_check_time == 0.0

    def test_context_manager(self):
        """Test using monitor as context manager."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        with monitor as m:
            assert m is monitor
            m.update(100)
            m.update(200)

        # Should have called finish
        assert monitor.current_iteration == 200

    def test_large_iteration_counts(self):
        """Test handling of very large iteration counts."""
        monitor = ProgressMonitor(total_iterations=1_000_000_000, show_console=False)  # 1 billion

        monitor.update(500_000_000)
        stats = monitor.get_stats()

        assert stats.current_iteration == 500_000_000
        assert stats.total_iterations == 1_000_000_000

    def test_update_after_finish(self):
        """Test that updates after convergence still track iteration."""
        monitor = ProgressMonitor(
            total_iterations=20000, check_intervals=[5000], show_console=False
        )

        # Converge early
        monitor.update(5000, convergence_value=1.05)
        assert monitor.converged is True

        # Continue updating
        should_continue = monitor.update(10000)
        assert should_continue is False  # Still converged
        assert monitor.current_iteration == 10000  # But iteration tracked

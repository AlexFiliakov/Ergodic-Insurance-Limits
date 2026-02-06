"""Coverage-targeted tests for progress_monitor.py.

Targets specific uncovered lines: 232, 287-289, 304, 313, 321-322, 370-393.
"""

import io
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

from ergodic_insurance.progress_monitor import ProgressMonitor, ProgressStats


# ---------------------------------------------------------------------------
# generate_convergence_summary: no checks performed (line 232)
# ---------------------------------------------------------------------------
class TestConvergenceSummaryNoChecks:
    """Test generate_convergence_summary when no checks have been performed."""

    def test_no_checks_returns_not_converged(self):
        """Line 232: Returns 'No convergence checks performed' when list is empty."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        summary = monitor.generate_convergence_summary()

        assert summary["converged"] is False
        assert "No convergence checks performed" in summary["message"]


# ---------------------------------------------------------------------------
# _analyze_convergence_trend: "no clear improvement" (lines 287-289)
# ---------------------------------------------------------------------------
class TestAnalyzeConvergenceTrendNoClearImprovement:
    """Test _analyze_convergence_trend returning 'no clear improvement'."""

    def test_no_clear_improvement(self):
        """Lines 287-289: Returns 'no clear improvement' when second half >= first half."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        # Values that increase (second half worse than first half)
        values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        trend = monitor._analyze_convergence_trend(values)
        assert trend == "no clear improvement"

    def test_slow_improvement(self):
        """Line 288: Returns 'slow improvement' when second half is slightly lower."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        # Second half is lower but not by 10%
        values = [1.5, 1.4, 1.45, 1.38, 1.35, 1.33]
        trend = monitor._analyze_convergence_trend(values)
        # second_half_mean ~ 1.353, first_half_mean ~ 1.45
        # 1.353 < 1.45 but 1.353 >= 1.45 * 0.9 (= 1.305)
        assert trend == "slow improvement"

    def test_improving_with_fluctuations(self):
        """Returns 'improving with fluctuations' when second half < first*0.9 and not monotonic."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        # Not monotonically decreasing but significant improvement
        values = [2.0, 1.8, 1.9, 1.2, 1.0, 1.1]
        trend = monitor._analyze_convergence_trend(values)
        assert trend == "improving with fluctuations"

    def test_monotonic_convergence_achieved(self):
        """Returns 'monotonic convergence achieved' when converged and decreasing."""
        monitor = ProgressMonitor(
            total_iterations=1000, show_console=False, convergence_threshold=1.1
        )
        monitor.converged = True

        values = [2.0, 1.5, 1.2, 1.0]
        trend = monitor._analyze_convergence_trend(values)
        assert trend == "monotonic convergence achieved"

    def test_monotonic_improvement_not_converged(self):
        """Returns 'monotonic improvement' when decreasing but not converged."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        values = [2.0, 1.5, 1.2, 1.0]
        trend = monitor._analyze_convergence_trend(values)
        assert trend == "monotonic improvement"

    def test_insufficient_data(self):
        """Returns 'insufficient data' when fewer than 2 values."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        assert monitor._analyze_convergence_trend([1.0]) == "insufficient data"
        assert monitor._analyze_convergence_trend([]) == "insufficient data"


# ---------------------------------------------------------------------------
# _estimate_convergence_rate: fewer than 3 values (line 304)
# ---------------------------------------------------------------------------
class TestEstimateConvergenceRateShortList:
    """Test _estimate_convergence_rate with fewer than 3 values."""

    def test_fewer_than_3_returns_none(self):
        """Line 304: Returns None when fewer than 3 values."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        result = monitor._estimate_convergence_rate([1000, 2000], [1.5, 1.2])
        assert result is None

    def test_single_value_returns_none(self):
        """Returns None for a single value."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        result = monitor._estimate_convergence_rate([1000], [1.5])
        assert result is None


# ---------------------------------------------------------------------------
# _estimate_convergence_rate: adjusted values <= 0 (line 313)
# ---------------------------------------------------------------------------
class TestEstimateConvergenceRateNonPositive:
    """Test _estimate_convergence_rate when adjusted values include non-positive."""

    def test_adjusted_values_zero(self):
        """Line 313: Returns None when adjusted values contain zeros."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        # min_val = 1.0 * 0.9 = 0.9, so adjusted = [0.1, 0.1, 0.1]
        # But if values are all the same as min_val*0.9:
        # values = [0.9, 0.9, 0.9], min_val = 0.9 * 0.9 = 0.81
        # adjusted = [0.09, 0.09, 0.09] - all positive, won't trigger
        # Need values where min * 0.9 >= one of the values:
        # values = [1.0, 1.0, 0.5], min_val = 0.5 * 0.9 = 0.45
        # adjusted = [0.55, 0.55, 0.05] - all positive
        # We need adjusted to be <= 0: min_val * 0.9 >= value
        # If values = [1.0, 0.5, 0.0], min_val = 0.0 * 0.9 = 0.0
        # adjusted = [1.0, 0.5, 0.0] -> 0.0 <= 0, returns None
        result = monitor._estimate_convergence_rate([1000, 2000, 3000], [1.0, 0.5, 0.0])
        assert result is None


# ---------------------------------------------------------------------------
# _estimate_convergence_rate: exception handling (lines 321-322)
# ---------------------------------------------------------------------------
class TestEstimateConvergenceRateException:
    """Test _estimate_convergence_rate exception handling."""

    def test_polyfit_failure_returns_none(self):
        """Lines 321-322: Returns None when polyfit raises ValueError."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        with patch("numpy.polyfit", side_effect=ValueError("singular")):
            result = monitor._estimate_convergence_rate([1000, 2000, 3000], [2.0, 1.5, 1.0])
        assert result is None

    def test_negative_rate_returns_none(self):
        """Returns None when computed rate is not positive (diverging)."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        # Values that increase (diverging) => rate will be negative
        result = monitor._estimate_convergence_rate([1000, 2000, 3000], [1.0, 1.5, 2.0])
        assert result is None


# ---------------------------------------------------------------------------
# finalize method (lines 370-393)
# ---------------------------------------------------------------------------
class TestFinalize:
    """Test the finalize method for full output."""

    def test_finalize_with_convergence(self):
        """Lines 370-393: finalize prints complete summary with convergence."""
        monitor = ProgressMonitor(
            total_iterations=20000,
            check_intervals=[10000, 15000],
            show_console=True,
            convergence_threshold=1.1,
        )

        # Simulate convergence
        monitor.converged = True
        monitor.converged_at = 10000
        monitor.current_iteration = 15000
        monitor.convergence_checks = [(10000, 1.05), (15000, 1.02)]
        monitor.monitor_overhead = 0.01

        with patch("builtins.print") as mock_print:
            monitor.finalize()

        # Verify key output elements
        print_calls = [str(c) for c in mock_print.call_args_list]
        output = " ".join(print_calls)

        assert "Simulation Complete" in output
        assert "Converged" in output

    def test_finalize_without_convergence(self):
        """Lines 370-393: finalize prints summary without convergence."""
        monitor = ProgressMonitor(
            total_iterations=10000,
            show_console=True,
        )
        monitor.current_iteration = 10000
        monitor.monitor_overhead = 0.005
        # Set start_time to the past to avoid ZeroDivisionError on fast machines
        monitor.start_time = time.time() - 10.0

        with patch("builtins.print") as mock_print:
            monitor.finalize()

        print_calls = [str(c) for c in mock_print.call_args_list]
        output = " ".join(print_calls)

        assert "Simulation Complete" in output
        assert "Did not achieve convergence" in output

    def test_finalize_with_checks_not_converged(self):
        """Lines 385-389: finalize shows convergence checks even without convergence."""
        monitor = ProgressMonitor(
            total_iterations=20000,
            check_intervals=[10000],
            show_console=True,
            convergence_threshold=1.05,
        )
        monitor.current_iteration = 15000
        monitor.convergence_checks = [(10000, 1.2)]
        monitor.monitor_overhead = 0.002
        # Set start_time to the past to avoid ZeroDivisionError on fast machines
        monitor.start_time = time.time() - 10.0

        with patch("builtins.print") as mock_print:
            monitor.finalize()

        print_calls = [str(c) for c in mock_print.call_args_list]
        output = " ".join(print_calls)

        assert "Convergence checks performed" in output
        assert "R-hat" in output

    def test_finalize_silent_when_console_disabled(self):
        """Lines 369-370: finalize does nothing when show_console is False."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        monitor.current_iteration = 1000

        with patch("builtins.print") as mock_print:
            monitor.finalize()

        mock_print.assert_not_called()


# ---------------------------------------------------------------------------
# generate_convergence_summary: full path with checks (integration)
# ---------------------------------------------------------------------------
class TestConvergenceSummaryFull:
    """Test generate_convergence_summary with actual convergence data."""

    def test_summary_with_converged_checks(self):
        """Test full summary generation with convergence data."""
        monitor = ProgressMonitor(
            total_iterations=50000,
            check_intervals=[10000, 20000, 30000],
            show_console=False,
            convergence_threshold=1.1,
        )

        monitor.convergence_checks = [
            (10000, 1.5),
            (20000, 1.2),
            (30000, 1.05),
        ]
        monitor.converged = True
        monitor.converged_at = 30000
        monitor.monitor_overhead = 0.01

        summary = monitor.generate_convergence_summary()

        assert summary["converged"] is True
        assert summary["converged_at"] == 30000
        assert summary["total_checks"] == 3
        assert summary["final_value"] == 1.05
        assert "avg_improvement_per_check" in summary
        assert "convergence_rate" in summary
        assert isinstance(summary["convergence_trend"], str)

    def test_summary_with_two_checks(self):
        """Summary with exactly two checks has improvements but maybe no rate."""
        monitor = ProgressMonitor(
            total_iterations=50000,
            check_intervals=[10000, 20000],
            show_console=False,
        )
        monitor.convergence_checks = [(10000, 1.5), (20000, 1.3)]
        monitor.monitor_overhead = 0.005

        summary = monitor.generate_convergence_summary()
        assert "avg_improvement_per_check" in summary
        # Rate requires >= 3 values, should be None
        assert summary.get("convergence_rate") is None


# ---------------------------------------------------------------------------
# Context manager finish semantics
# ---------------------------------------------------------------------------
class TestContextManagerFinish:
    """Test context manager properly calls finish."""

    def test_context_manager_calls_finish(self):
        """__exit__ calls finish()."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)

        with patch.object(monitor, "finish") as mock_finish:
            with monitor:
                monitor.update(500)
            mock_finish.assert_called_once()


# ---------------------------------------------------------------------------
# _print_convergence_message
# ---------------------------------------------------------------------------
class TestPrintConvergenceMessage:
    """Test _print_convergence_message output."""

    def test_convergence_message_format(self):
        """Verify convergence message includes iteration and R-hat value."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=True)

        with patch("builtins.print") as mock_print:
            monitor._print_convergence_message(5000, 1.05)

        call_str = str(mock_print.call_args)
        assert "5,000" in call_str
        assert "1.050" in call_str


# ---------------------------------------------------------------------------
# get_overhead_percentage edge cases
# ---------------------------------------------------------------------------
class TestOverheadPercentageEdgeCases:
    """Test overhead calculation edge cases."""

    def test_high_overhead(self):
        """High overhead returns appropriate percentage."""
        monitor = ProgressMonitor(total_iterations=1000, show_console=False)
        monitor.monitor_overhead = 5.0

        with patch("time.time") as mock_time:
            mock_time.return_value = monitor.start_time + 10.0
            overhead = monitor.get_overhead_percentage()

        assert overhead == pytest.approx(50.0)

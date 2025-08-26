"""Lightweight progress monitoring for Monte Carlo simulations.

This module provides efficient progress tracking with minimal performance overhead,
including ETA estimation, convergence summaries, and console output.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProgressStats:
    """Statistics for progress monitoring.

    Attributes:
        current_iteration: Current iteration number
        total_iterations: Total planned iterations
        start_time: Simulation start time
        elapsed_time: Elapsed time in seconds
        estimated_time_remaining: Estimated time remaining in seconds
        iterations_per_second: Current processing speed
        convergence_checks: List of convergence check results
        converged: Whether convergence has been achieved
        converged_at: Iteration where convergence was achieved
    """

    current_iteration: int
    total_iterations: int
    start_time: float
    elapsed_time: float
    estimated_time_remaining: float
    iterations_per_second: float
    convergence_checks: List[Tuple[int, float]] = field(default_factory=list)
    converged: bool = False
    converged_at: Optional[int] = None

    def summary(self) -> str:
        """Generate progress summary."""
        progress_pct = (self.current_iteration / self.total_iterations) * 100
        eta = timedelta(seconds=int(self.estimated_time_remaining))
        elapsed = timedelta(seconds=int(self.elapsed_time))

        summary = (
            f"Progress: {self.current_iteration:,}/{self.total_iterations:,} "
            f"({progress_pct:.1f}%) | "
            f"Speed: {self.iterations_per_second:.0f} it/s | "
            f"Elapsed: {elapsed} | "
            f"ETA: {eta}"
        )

        if self.converged:
            summary += f" | CONVERGED at iteration {self.converged_at:,}"

        return summary


class ProgressMonitor:
    """Lightweight progress monitor for Monte Carlo simulations.

    Provides real-time progress tracking with minimal performance overhead (<1%).
    Includes ETA estimation, convergence monitoring, and console output.
    """

    def __init__(
        self,
        total_iterations: int,
        check_intervals: Optional[List[int]] = None,
        update_frequency: int = 1000,
        show_console: bool = True,
        convergence_threshold: float = 1.1,
    ):
        """Initialize progress monitor.

        Args:
            total_iterations: Total number of iterations to run
            check_intervals: Iterations at which to check convergence
            update_frequency: Update console every N iterations
            show_console: Whether to show console output
            convergence_threshold: R-hat threshold for convergence
        """
        self.total_iterations = total_iterations
        self.check_intervals = check_intervals or [10_000, 25_000, 50_000, 100_000]
        self.update_frequency = update_frequency
        self.show_console = show_console
        self.convergence_threshold = convergence_threshold

        # State tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_iteration = 0
        self.current_iteration = 0

        # Performance tracking
        self.iteration_times: List[float] = []
        self.convergence_checks: List[Tuple[int, float]] = []
        self.converged = False
        self.converged_at: Optional[int] = None

        # Performance impact tracking
        self.monitor_overhead = 0.0
        self.total_check_time = 0.0

    def update(self, iteration: int, convergence_value: Optional[float] = None) -> bool:
        """Update progress and check for convergence.

        Args:
            iteration: Current iteration number
            convergence_value: Optional convergence metric (e.g., R-hat)

        Returns:
            True if should continue, False if converged and should stop
        """
        monitor_start = time.perf_counter()

        self.current_iteration = iteration
        current_time = time.time()

        # Check if we should perform convergence check
        should_check_convergence = (
            convergence_value is not None
            and iteration in self.check_intervals
            and not self.converged
        )

        if should_check_convergence:
            check_start = time.perf_counter()
            self.convergence_checks.append((iteration, convergence_value))  # type: ignore

            if convergence_value is not None and convergence_value < self.convergence_threshold:
                self.converged = True
                self.converged_at = iteration

                if self.show_console:
                    self._print_convergence_message(iteration, convergence_value)

            self.total_check_time += time.perf_counter() - check_start

        # Update console if needed
        should_update_console = (
            self.show_console and (iteration - self.last_update_iteration) >= self.update_frequency
        )

        if should_update_console:
            self._update_console(current_time)
            self.last_update_time = current_time
            self.last_update_iteration = iteration

        # Track overhead
        self.monitor_overhead += time.perf_counter() - monitor_start

        # Return False if converged (should stop)
        return not self.converged

    def _update_console(self, current_time: float) -> None:
        """Update console with progress information."""
        elapsed = current_time - self.start_time

        # Calculate speed
        if elapsed > 0:
            speed = self.current_iteration / elapsed
        else:
            speed = 0

        # Estimate remaining time
        if speed > 0 and self.current_iteration > 0:
            remaining_iterations = self.total_iterations - self.current_iteration
            eta = remaining_iterations / speed
        else:
            eta = 0

        # Create progress bar
        progress_pct = (self.current_iteration / self.total_iterations) * 100
        bar_width = 40
        filled = int(bar_width * self.current_iteration / self.total_iterations)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Format time strings
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        eta_str = str(timedelta(seconds=int(eta)))

        # Build status line
        status = (
            f"\r[{bar}] {progress_pct:5.1f}% | "
            f"{self.current_iteration:,}/{self.total_iterations:,} | "
            f"{speed:.0f} it/s | "
            f"Elapsed: {elapsed_str} | "
            f"ETA: {eta_str}"
        )

        # Add convergence info if available
        if self.convergence_checks:
            last_check = self.convergence_checks[-1]
            status += f" | R-hat: {last_check[1]:.3f}"

        print(status, end="", flush=True)

    def _print_convergence_message(self, iteration: int, convergence_value: float) -> None:
        """Print convergence achievement message."""
        print(
            f"\n✓ Convergence achieved at iteration {iteration:,} (R-hat = {convergence_value:.3f})"
        )

    def get_stats(self) -> ProgressStats:
        """Get current progress statistics.

        Returns:
            ProgressStats object with current metrics
        """
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Calculate speed and ETA
        if elapsed > 0:
            speed = self.current_iteration / elapsed
            remaining_iterations = self.total_iterations - self.current_iteration
            eta = remaining_iterations / speed if speed > 0 else 0
        else:
            speed = 0
            eta = 0

        return ProgressStats(
            current_iteration=self.current_iteration,
            total_iterations=self.total_iterations,
            start_time=self.start_time,
            elapsed_time=elapsed,
            estimated_time_remaining=eta,
            iterations_per_second=speed,
            convergence_checks=self.convergence_checks.copy(),
            converged=self.converged,
            converged_at=self.converged_at,
        )

    def generate_convergence_summary(self) -> Dict[str, Any]:
        """Generate detailed convergence summary.

        Returns:
            Dictionary with convergence analysis results
        """
        if not self.convergence_checks:
            return {"converged": False, "message": "No convergence checks performed"}

        iterations, values = zip(*self.convergence_checks)

        elapsed = time.time() - self.start_time
        overhead_pct = (self.monitor_overhead / elapsed * 100) if elapsed > 0 else 0.0

        summary = {
            "converged": self.converged,
            "converged_at": self.converged_at,
            "total_checks": len(self.convergence_checks),
            "check_iterations": list(iterations),
            "convergence_values": list(values),
            "final_value": values[-1],
            "convergence_trend": self._analyze_convergence_trend(list(values)),
            "performance_overhead_pct": overhead_pct,
        }

        # Add convergence rate if we have enough data
        if len(values) >= 2:
            # Calculate rate of convergence improvement
            improvements = [values[i] - values[i + 1] for i in range(len(values) - 1)]
            summary["avg_improvement_per_check"] = np.mean(improvements)
            summary["convergence_rate"] = self._estimate_convergence_rate(
                list(iterations), list(values)
            )

        return summary

    def _analyze_convergence_trend(self, values: List[float]) -> str:
        """Analyze convergence trend from values.

        Args:
            values: List of convergence metric values

        Returns:
            String description of trend
        """
        if len(values) < 2:
            return "insufficient data"

        # Check if monotonically decreasing
        is_decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

        if is_decreasing:
            if self.converged:
                return "monotonic convergence achieved"
            else:
                return "monotonic improvement"
        else:
            # Check overall trend
            first_half = np.mean(values[: len(values) // 2])
            second_half = np.mean(values[len(values) // 2 :])

            if second_half < first_half * 0.9:
                return "improving with fluctuations"
            elif second_half < first_half:
                return "slow improvement"
            else:
                return "no clear improvement"

    def _estimate_convergence_rate(
        self, iterations: List[int], values: List[float]
    ) -> Optional[float]:
        """Estimate convergence rate using exponential fit.

        Args:
            iterations: List of iteration numbers
            values: List of convergence values

        Returns:
            Estimated convergence rate (decay parameter) or None
        """
        if len(values) < 3:
            return None

        try:
            # Fit exponential decay: value = a * exp(-rate * iteration) + b
            # Use log-linear regression on (value - min_value)
            min_val = min(values) * 0.9  # Slightly below minimum
            adjusted_values = [v - min_val for v in values]

            if any(v <= 0 for v in adjusted_values):
                return None

            log_values = np.log(adjusted_values)
            coeffs = np.polyfit(iterations, log_values, 1)
            rate = -coeffs[0]  # Negative of slope gives decay rate

            return float(rate) if rate > 0 else None

        except (ValueError, RuntimeWarning):
            return None

    def finalize(self) -> None:
        """Finalize progress monitoring and print summary."""
        if self.show_console:
            print()  # New line after progress bar

            stats = self.get_stats()
            print(f"\n{'='*60}")
            print("Simulation Complete")
            print(f"{'='*60}")
            print(f"Total iterations: {stats.current_iteration:,}")
            print(f"Total time: {timedelta(seconds=int(stats.elapsed_time))}")
            print(f"Average speed: {stats.iterations_per_second:.0f} iterations/second")

            if self.converged:
                print(f"✓ Converged at iteration {self.converged_at:,}")
            else:
                print("✗ Did not achieve convergence")

            if self.convergence_checks:
                print(f"\nConvergence checks performed: {len(self.convergence_checks)}")
                for iter_num, value in self.convergence_checks:
                    status = "✓" if value < self.convergence_threshold else "✗"
                    print(f"  {status} Iteration {iter_num:,}: R-hat = {value:.3f}")

            # Performance overhead
            overhead_pct = (self.monitor_overhead / (time.time() - self.start_time)) * 100
            print(f"\nMonitoring overhead: {overhead_pct:.2f}%")

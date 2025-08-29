"""Adaptive stopping criteria for Monte Carlo simulations.

This module implements adaptive stopping rules based on convergence diagnostics,
allowing simulations to terminate early when convergence criteria are met.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import stats


class StoppingRule(Enum):
    """Enumeration of available stopping rules."""

    R_HAT = "r_hat"
    ESS = "ess"
    RELATIVE_CHANGE = "relative_change"
    MCSE = "mcse"
    GEWEKE = "geweke"
    HEIDELBERGER = "heidelberger"
    COMBINED = "combined"
    CUSTOM = "custom"


@dataclass
class StoppingCriteria:
    """Configuration for stopping criteria.

    Attributes:
        rule: Type of stopping rule to apply
        r_hat_threshold: Maximum R-hat for convergence
        min_ess: Minimum effective sample size
        relative_tolerance: Relative change tolerance
        mcse_relative_threshold: Maximum relative MCSE
        min_iterations: Minimum iterations before checking
        max_iterations: Maximum iterations allowed
        check_interval: Check convergence every N iterations
        patience: Number of consecutive checks before stopping
        confidence_level: Confidence level for statistical tests
    """

    rule: StoppingRule = StoppingRule.COMBINED
    r_hat_threshold: float = 1.05
    min_ess: int = 1000
    relative_tolerance: float = 0.01
    mcse_relative_threshold: float = 0.05
    min_iterations: int = 1000
    max_iterations: int = 100000
    check_interval: int = 100
    patience: int = 3
    confidence_level: float = 0.95

    def __post_init__(self):
        """Validate criteria after initialization."""
        if self.r_hat_threshold <= 1.0:
            raise ValueError("R-hat threshold must be > 1.0")
        if self.min_ess < 100:
            warnings.warn("Very low ESS threshold may lead to poor estimates")
        if self.min_iterations < 100:
            warnings.warn("Very low minimum iterations may lead to premature stopping")


@dataclass
class ConvergenceStatus:
    """Container for convergence status information.

    Attributes:
        converged: Whether convergence criteria are met
        iteration: Current iteration number
        reason: Reason for convergence or non-convergence
        diagnostics: Dictionary of diagnostic values
        should_stop: Whether to stop the simulation
        estimated_remaining: Estimated iterations to convergence
    """

    converged: bool
    iteration: int
    reason: str
    diagnostics: Dict[str, float]
    should_stop: bool
    estimated_remaining: Optional[int] = None

    def __str__(self) -> str:
        """String representation of convergence status."""
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return f"ConvergenceStatus({status} at iteration {self.iteration}): " f"{self.reason}"


class AdaptiveStoppingMonitor:
    """Monitor for adaptive stopping based on convergence criteria.

    Provides sophisticated adaptive stopping with multiple criteria,
    burn-in detection, and convergence rate estimation.
    """

    def __init__(
        self, criteria: Optional[StoppingCriteria] = None, custom_rule: Optional[Callable] = None
    ):
        """Initialize adaptive stopping monitor.

        Args:
            criteria: Stopping criteria configuration
            custom_rule: Custom stopping rule function
        """
        self.criteria = criteria or StoppingCriteria()
        self.custom_rule = custom_rule

        # History tracking
        self.r_hat_history: List[float] = []
        self.ess_history: List[float] = []
        self.mean_history: List[float] = []
        self.variance_history: List[float] = []
        self.iteration_history: List[int] = []

        # Convergence tracking
        self.consecutive_convergence = 0
        self.burn_in_detected = False
        self.burn_in_iteration = 0

        # Rate estimation
        self.convergence_rate: Optional[float] = None
        self.estimated_total_iterations: Optional[int] = None

    def check_convergence(
        self, iteration: int, chains: np.ndarray, diagnostics: Optional[Dict[str, float]] = None
    ) -> ConvergenceStatus:
        """Check if convergence criteria are met.

        Args:
            iteration: Current iteration number
            chains: Array of chain values
            diagnostics: Pre-calculated diagnostics (optional)

        Returns:
            ConvergenceStatus object with convergence information
        """
        # Don't check before minimum iterations
        if iteration < self.criteria.min_iterations:
            return ConvergenceStatus(
                converged=False,
                iteration=iteration,
                reason=f"Below minimum iterations ({self.criteria.min_iterations})",
                diagnostics={},
                should_stop=False,
                estimated_remaining=self.criteria.min_iterations - iteration,
            )

        # Check if at maximum iterations
        if iteration >= self.criteria.max_iterations:
            return ConvergenceStatus(
                converged=False,
                iteration=iteration,
                reason=f"Maximum iterations reached ({self.criteria.max_iterations})",
                diagnostics=diagnostics or {},
                should_stop=True,
            )

        # Only check at intervals for efficiency
        if iteration % self.criteria.check_interval != 0:
            return ConvergenceStatus(
                converged=False,
                iteration=iteration,
                reason="Not at check interval",
                diagnostics={},
                should_stop=False,
            )

        # Calculate diagnostics if not provided
        if diagnostics is None:
            diagnostics = self._calculate_diagnostics(chains)

        # Update history
        self._update_history(iteration, diagnostics)

        # Detect burn-in if not already done
        if not self.burn_in_detected:
            self._detect_burn_in(chains, iteration)

        # Check stopping rule
        converged, reason = self._check_stopping_rule(diagnostics)

        # Update consecutive convergence counter
        if converged:
            self.consecutive_convergence += 1
        else:
            self.consecutive_convergence = 0

        # Determine if should stop (need patience consecutive convergences)
        should_stop = converged and self.consecutive_convergence >= self.criteria.patience

        # Estimate remaining iterations
        estimated_remaining = self._estimate_remaining_iterations(iteration, converged, diagnostics)

        return ConvergenceStatus(
            converged=converged,
            iteration=iteration,
            reason=reason,
            diagnostics=diagnostics,
            should_stop=should_stop,
            estimated_remaining=estimated_remaining,
        )

    def detect_adaptive_burn_in(self, chains: np.ndarray, method: str = "geweke") -> int:
        """Detect burn-in period adaptively.

        Args:
            chains: Array of chain values
            method: Method for burn-in detection

        Returns:
            Estimated burn-in period
        """
        if chains.ndim == 1:
            chains = chains.reshape(1, -1)

        n_chains, n_iterations = chains.shape[:2]

        if method == "geweke":
            # Use Geweke test to find burn-in
            burn_in_estimates = []

            for chain_idx in range(n_chains):
                chain = chains[chain_idx].flatten()

                # Test different potential burn-in points
                test_points = np.linspace(0, n_iterations // 2, 20).astype(int)

                for test_point in test_points[1:]:  # Skip 0
                    # Test stationarity after this point
                    test_chain = chain[test_point:]

                    if len(test_chain) < 100:
                        continue

                    # Geweke test
                    z_score, p_value = self._geweke_test(test_chain)

                    if p_value > 0.05:  # Stationary
                        burn_in_estimates.append(test_point)
                        break

            if burn_in_estimates:
                return int(np.median(burn_in_estimates))
            return int(n_iterations // 10)  # Default fallback

        elif method == "variance":  # pylint: disable=no-else-return
            # Detect when variance stabilizes
            window_size = max(10, n_iterations // 100)
            variances = []

            for i in range(window_size, n_iterations - window_size):
                window_var = np.var(chains[:, i - window_size : i + window_size])
                variances.append(window_var)

            if len(variances) > 0:
                # Find where variance change rate drops
                var_change = np.abs(np.diff(variances))
                threshold = np.percentile(var_change, 10)

                stable_points = np.where(var_change < threshold)[0]
                if len(stable_points) > 0:
                    return int(stable_points[0] + window_size)

            return int(n_iterations // 10)  # Default fallback

        else:
            raise ValueError(f"Unknown burn-in detection method: {method}")

    def estimate_convergence_rate(
        self, diagnostic_history: List[float], target_value: float = 1.0
    ) -> Tuple[float, int]:
        """Estimate convergence rate and iterations to target.

        Args:
            diagnostic_history: History of diagnostic values
            target_value: Target value for convergence

        Returns:
            Tuple of (convergence_rate, estimated_iterations_to_target)
        """
        if len(diagnostic_history) < 3:
            return 0.0, -1

        # Fit exponential decay model
        iterations = np.arange(len(diagnostic_history))
        values = np.array(diagnostic_history)

        # Transform for linear regression
        # Assuming: value = a * exp(-rate * iteration) + target
        # log(value - target) = log(a) - rate * iteration

        values_shifted = values - target_value
        positive_mask = values_shifted > 0

        if np.sum(positive_mask) < 2:
            return 0.0, -1

        log_values = np.log(values_shifted[positive_mask])
        iterations_masked = iterations[positive_mask]

        # Linear regression
        if len(iterations_masked) >= 2:
            slope, intercept = np.polyfit(iterations_masked, log_values, 1)
            rate = -slope

            if rate > 0:
                # Estimate iterations to reach target (within 1% of target)
                current_value = values[-1]
                if abs(current_value - target_value) > 0.01 * abs(target_value):
                    iterations_to_target = int(np.log(0.01) / (-rate))
                else:
                    iterations_to_target = 0

                return rate, iterations_to_target

        return 0.0, -1

    def get_stopping_summary(self) -> Dict[str, Any]:
        """Get summary of stopping monitor state.

        Returns:
            Dictionary with monitor summary information
        """
        summary = {
            "iterations_checked": len(self.iteration_history),
            "consecutive_convergence": self.consecutive_convergence,
            "burn_in_detected": self.burn_in_detected,
            "burn_in_iteration": self.burn_in_iteration,
            "convergence_rate": self.convergence_rate,
            "estimated_total_iterations": self.estimated_total_iterations,
            "criteria": {
                "rule": self.criteria.rule.value,
                "r_hat_threshold": self.criteria.r_hat_threshold,
                "min_ess": self.criteria.min_ess,
                "patience": self.criteria.patience,
            },
        }

        # Add latest diagnostics if available
        if self.r_hat_history:
            summary["latest_r_hat"] = self.r_hat_history[-1]
        if self.ess_history:
            summary["latest_ess"] = self.ess_history[-1]
        if self.mean_history:
            summary["latest_mean"] = self.mean_history[-1]

        return summary

    # Private helper methods

    def _calculate_diagnostics(self, chains: np.ndarray) -> Dict[str, float]:
        """Calculate convergence diagnostics from chains."""
        if chains.ndim == 1:
            chains = chains.reshape(1, -1)

        diagnostics = {}

        # R-hat calculation (if multiple chains)
        if chains.shape[0] > 1:
            diagnostics["r_hat"] = self._calculate_r_hat(chains)
        else:
            diagnostics["r_hat"] = 1.0

        # ESS calculation
        pooled_chain = chains.flatten()
        diagnostics["ess"] = self._calculate_ess(pooled_chain)

        # Mean and variance
        diagnostics["mean"] = np.mean(pooled_chain)
        diagnostics["variance"] = np.var(pooled_chain, ddof=1)

        # MCSE
        if diagnostics["ess"] > 0:
            diagnostics["mcse"] = np.sqrt(diagnostics["variance"] / diagnostics["ess"])
            diagnostics["mcse_relative"] = (
                diagnostics["mcse"] / abs(diagnostics["mean"])
                if diagnostics["mean"] != 0
                else np.inf
            )
        else:
            diagnostics["mcse"] = np.inf
            diagnostics["mcse_relative"] = np.inf

        return diagnostics

    def _calculate_r_hat(self, chains: np.ndarray) -> float:
        """Calculate Gelman-Rubin R-hat statistic."""
        n_chains, n_iterations = chains.shape[:2]

        # Between-chain variance
        chain_means = np.mean(chains, axis=1)
        grand_mean = np.mean(chain_means)
        between_var = n_iterations * np.var(chain_means, ddof=1)

        # Within-chain variance
        within_vars = np.var(chains, axis=1, ddof=1)
        within_var = np.mean(within_vars)

        # Calculate R-hat
        var_est = ((n_iterations - 1) * within_var + between_var) / n_iterations
        r_hat = np.sqrt(var_est / within_var) if within_var > 0 else np.inf

        return float(r_hat)

    def _calculate_ess(self, chain: np.ndarray) -> float:
        """Calculate effective sample size."""
        n = len(chain)

        if n < 4:
            return float(n)

        # Calculate autocorrelations
        max_lag = min(n // 4, 1000)
        acf = self._calculate_acf(chain, max_lag)

        # Find first negative autocorrelation
        first_negative = np.where(acf < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = len(acf)

        # Sum autocorrelations (Geyer's method)
        sum_acf = 1.0
        for i in range(1, cutoff, 2):
            if i + 1 < cutoff:
                pair_sum = acf[i] + acf[i + 1]
                if pair_sum > 0:
                    sum_acf += 2 * pair_sum
                else:
                    break

        ess = n / max(sum_acf, 1)
        return float(min(ess, n))

    def _calculate_acf(self, chain: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(chain)
        chain_centered = chain - np.mean(chain)
        c0 = np.dot(chain_centered, chain_centered) / n

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, min(max_lag + 1, n)):
            c_lag = np.dot(chain_centered[:-lag], chain_centered[lag:]) / n
            acf[lag] = c_lag / c0 if c0 > 0 else 0

        return acf

    def _geweke_test(
        self, chain: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5
    ) -> Tuple[float, float]:
        """Perform Geweke convergence test."""
        n = len(chain)
        n_first = int(n * first_frac)
        n_last = int(n * last_frac)

        first_portion = chain[:n_first]
        last_portion = chain[-n_last:]

        mean_first = np.mean(first_portion)
        mean_last = np.mean(last_portion)

        var_first = np.var(first_portion, ddof=1) / n_first
        var_last = np.var(last_portion, ddof=1) / n_last

        z_score = (mean_first - mean_last) / np.sqrt(var_first + var_last)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return z_score, p_value

    def _update_history(self, iteration: int, diagnostics: Dict[str, float]):
        """Update diagnostic history."""
        self.iteration_history.append(iteration)

        if "r_hat" in diagnostics:
            self.r_hat_history.append(diagnostics["r_hat"])
        if "ess" in diagnostics:
            self.ess_history.append(diagnostics["ess"])
        if "mean" in diagnostics:
            self.mean_history.append(diagnostics["mean"])
        if "variance" in diagnostics:
            self.variance_history.append(diagnostics["variance"])

    def _detect_burn_in(self, chains: np.ndarray, iteration: int):
        """Detect burn-in period."""
        if iteration < 500:  # Too early to detect
            return

        # Use adaptive burn-in detection
        burn_in = self.detect_adaptive_burn_in(chains, method="geweke")

        if burn_in < iteration // 2:  # Reasonable burn-in found
            self.burn_in_detected = True
            self.burn_in_iteration = burn_in

    def _check_stopping_rule(self, diagnostics: Dict[str, float]) -> Tuple[bool, str]:  # pylint: disable=too-many-branches
        """Check if stopping rule is satisfied."""
        if self.criteria.rule == StoppingRule.R_HAT:
            r_hat = diagnostics.get("r_hat", np.inf)
            converged = r_hat < self.criteria.r_hat_threshold
            reason = f"R-hat = {r_hat:.4f} (threshold: {self.criteria.r_hat_threshold})"

        elif self.criteria.rule == StoppingRule.ESS:
            ess = diagnostics.get("ess", 0)
            converged = ess >= self.criteria.min_ess
            reason = f"ESS = {ess:.0f} (minimum: {self.criteria.min_ess})"

        elif self.criteria.rule == StoppingRule.MCSE:
            mcse_rel = diagnostics.get("mcse_relative", np.inf)
            converged = mcse_rel < self.criteria.mcse_relative_threshold
            reason = f"Relative MCSE = {mcse_rel:.4f} (threshold: {self.criteria.mcse_relative_threshold})"

        elif self.criteria.rule == StoppingRule.RELATIVE_CHANGE:
            converged = False
            reason = "Relative change not yet implemented"

            if len(self.mean_history) >= 2:
                recent_mean = self.mean_history[-1]
                previous_mean = self.mean_history[-2]
                if previous_mean != 0:
                    rel_change = abs(recent_mean - previous_mean) / abs(previous_mean)
                    converged = rel_change < self.criteria.relative_tolerance
                    reason = f"Relative change = {rel_change:.4f} (tolerance: {self.criteria.relative_tolerance})"

        elif self.criteria.rule == StoppingRule.COMBINED:
            # Check all criteria
            checks = []

            r_hat = diagnostics.get("r_hat", np.inf)
            r_hat_ok = r_hat < self.criteria.r_hat_threshold
            checks.append((r_hat_ok, f"R-hat={r_hat:.3f}"))

            ess = diagnostics.get("ess", 0)
            ess_ok = ess >= self.criteria.min_ess
            checks.append((ess_ok, f"ESS={ess:.0f}"))

            mcse_rel = diagnostics.get("mcse_relative", np.inf)
            mcse_ok = mcse_rel < self.criteria.mcse_relative_threshold
            checks.append((mcse_ok, f"MCSE_rel={mcse_rel:.3f}"))

            converged = all(check[0] for check in checks)
            failed_checks = [check[1] for check in checks if not check[0]]

            if converged:
                reason = "All criteria met: " + ", ".join(check[1] for check in checks)
            else:
                reason = "Failed: " + ", ".join(failed_checks)

        elif self.criteria.rule == StoppingRule.CUSTOM:
            if self.custom_rule is not None:
                converged, reason = self.custom_rule(diagnostics)
            else:
                converged = False
                reason = "No custom rule provided"

        else:
            converged = False
            reason = f"Unknown stopping rule: {self.criteria.rule}"

        return converged, reason

    def _estimate_remaining_iterations(
        self, current_iteration: int, converged: bool, diagnostics: Dict[str, float]
    ) -> Optional[int]:
        """Estimate remaining iterations to convergence."""
        if converged:
            return 0

        # Use R-hat history for estimation if available
        if len(self.r_hat_history) >= 3:
            rate, iterations_to_target = self.estimate_convergence_rate(
                self.r_hat_history, target_value=self.criteria.r_hat_threshold
            )

            if iterations_to_target > 0:
                self.convergence_rate = rate
                self.estimated_total_iterations = current_iteration + iterations_to_target
                return iterations_to_target

        # Fallback: use simple linear extrapolation
        if len(self.ess_history) >= 2:
            current_ess = self.ess_history[-1]
            previous_ess = self.ess_history[-2]
            ess_rate = (current_ess - previous_ess) / self.criteria.check_interval

            if ess_rate > 0:
                ess_needed = self.criteria.min_ess - current_ess
                iterations_needed = int(ess_needed / ess_rate)
                return max(0, iterations_needed)

        return None

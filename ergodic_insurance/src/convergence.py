"""Convergence diagnostics for Monte Carlo simulations.

This module provides tools for assessing convergence of Monte Carlo simulations
including Gelman-Rubin R-hat, effective sample size, and Monte Carlo standard error.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class ConvergenceStats:
    """Container for convergence statistics.

    Attributes:
        r_hat: Gelman-Rubin convergence statistic (should be < 1.1)
        ess: Effective sample size
        mcse: Monte Carlo standard error
        converged: Whether convergence criteria are met
        n_iterations: Number of iterations analyzed
        autocorrelation: Lag-1 autocorrelation
    """

    r_hat: float
    ess: float
    mcse: float
    converged: bool
    n_iterations: int
    autocorrelation: float

    def __str__(self) -> str:
        """String representation of convergence stats."""
        return (
            f"ConvergenceStats(r_hat={self.r_hat:.3f}, "
            f"ess={self.ess:.0f}, mcse={self.mcse:.4f}, "
            f"converged={self.converged})"
        )


class ConvergenceDiagnostics:
    """Convergence diagnostics for Monte Carlo simulations.

    Provides methods for assessing convergence using multiple chains
    and calculating effective sample sizes.
    """

    def __init__(
        self,
        r_hat_threshold: float = 1.1,
        min_ess: int = 1000,
        relative_mcse_threshold: float = 0.05,
    ):
        """Initialize convergence diagnostics.

        Args:
            r_hat_threshold: Maximum R-hat for convergence (default 1.1)
            min_ess: Minimum effective sample size (default 1000)
            relative_mcse_threshold: Maximum relative MCSE (default 0.05)
        """
        self.r_hat_threshold = r_hat_threshold
        self.min_ess = min_ess
        self.relative_mcse_threshold = relative_mcse_threshold

    def calculate_r_hat(self, chains: np.ndarray) -> float:
        """Calculate Gelman-Rubin R-hat statistic.

        Args:
            chains: Array of shape (n_chains, n_iterations) or (n_chains, n_iterations, n_metrics)

        Returns:
            R-hat statistic (values close to 1 indicate convergence)
        """
        if chains.ndim == 2:
            n_chains, n_iterations = chains.shape
        elif chains.ndim == 3:
            n_chains, n_iterations, n_metrics = chains.shape
            # Calculate R-hat for each metric and return maximum
            r_hats = [self.calculate_r_hat(chains[:, :, i]) for i in range(n_metrics)]
            return max(r_hats)
        else:
            raise ValueError("Chains must be 2D or 3D array")

        if n_chains < 2:
            raise ValueError("Need at least 2 chains for R-hat calculation")

        # Calculate between-chain variance
        chain_means = np.mean(chains, axis=1)
        grand_mean = np.mean(chain_means)
        between_var = n_iterations * np.var(chain_means, ddof=1)

        # Calculate within-chain variance
        within_vars = np.var(chains, axis=1, ddof=1)
        within_var = np.mean(within_vars)

        # Calculate pooled variance estimate
        var_est = ((n_iterations - 1) * within_var + between_var) / n_iterations

        # Calculate R-hat
        r_hat = np.sqrt(var_est / within_var) if within_var > 0 else np.inf

        return float(r_hat)

    def calculate_ess(self, chain: np.ndarray, max_lag: Optional[int] = None) -> float:
        """Calculate effective sample size using autocorrelation.

        Uses the formula: ESS = N / (1 + 2 * sum(autocorrelations))
        where the sum is truncated at the first negative autocorrelation.

        Args:
            chain: 1D array of samples
            max_lag: Maximum lag for autocorrelation calculation

        Returns:
            Effective sample size
        """
        n = len(chain)

        if n < 4:
            return float(n)

        if max_lag is None:
            max_lag = min(n // 4, 1000)

        # Calculate autocorrelations
        autocorr = self._calculate_autocorrelation(chain, max_lag)

        # Find first negative autocorrelation (Geyer's initial monotone sequence)
        first_negative = np.where(autocorr < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = len(autocorr)

        # Apply Geyer's initial positive sequence estimator
        # Sum pairs of autocorrelations and stop when sum becomes negative
        sum_autocorr = 1.0  # Start with lag 0 (always 1)
        for i in range(1, cutoff, 2):
            if i + 1 < cutoff:
                pair_sum = autocorr[i] + autocorr[i + 1]
                if pair_sum > 0:
                    sum_autocorr += 2 * pair_sum
                else:
                    break
            else:
                # Handle odd final term
                if autocorr[i] > 0:
                    sum_autocorr += 2 * autocorr[i]

        # Calculate ESS
        ess = n / max(sum_autocorr, 1)

        return float(min(ess, n))  # ESS cannot exceed actual sample size

    def calculate_batch_ess(
        self, chains: np.ndarray, method: str = "mean"
    ) -> Union[float, np.ndarray]:
        """Calculate ESS for multiple chains or metrics.

        Args:
            chains: Array of shape (n_chains, n_iterations) or (n_chains, n_iterations, n_metrics)
            method: How to combine ESS across chains ('mean', 'min', 'all')

        Returns:
            Combined ESS value(s)
        """
        if chains.ndim == 2:
            # Multiple chains, single metric
            ess_values = [self.calculate_ess(chain) for chain in chains]
        elif chains.ndim == 3:
            # Multiple chains, multiple metrics
            n_chains, n_iterations, n_metrics = chains.shape
            ess_values = []
            for m in range(n_metrics):
                metric_ess = [self.calculate_ess(chains[c, :, m]) for c in range(n_chains)]
                ess_values.append(metric_ess)  # type: ignore
        else:
            raise ValueError("Chains must be 2D or 3D array")

        # Process based on method
        if method == "mean":
            if chains.ndim == 2:
                return float(np.mean(ess_values))
            else:
                return np.array([np.mean(metric_ess) for metric_ess in ess_values])
        elif method == "min":
            if chains.ndim == 2:
                return float(np.min(ess_values))
            else:
                return np.array([np.min(metric_ess) for metric_ess in ess_values])
        elif method == "all":
            return np.array(ess_values)
        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_ess_per_second(self, chain: np.ndarray, computation_time: float) -> float:
        """Calculate ESS per second of computation.

        Useful for comparing efficiency of different sampling methods.

        Args:
            chain: 1D array of samples
            computation_time: Time in seconds taken to generate the chain

        Returns:
            ESS per second
        """
        ess = self.calculate_ess(chain)
        return ess / computation_time if computation_time > 0 else 0.0

    def calculate_mcse(self, chain: np.ndarray, ess: Optional[float] = None) -> float:
        """Calculate Monte Carlo standard error.

        Args:
            chain: 1D array of samples
            ess: Effective sample size (calculated if not provided)

        Returns:
            Monte Carlo standard error
        """
        if ess is None:
            ess = self.calculate_ess(chain)

        # Calculate standard error using ESS
        std_dev = np.std(chain, ddof=1)
        mcse = std_dev / np.sqrt(ess)

        return float(mcse)

    def check_convergence(
        self, chains: Union[np.ndarray, List[np.ndarray]], metric_names: Optional[List[str]] = None
    ) -> Dict[str, ConvergenceStats]:
        """Check convergence for multiple chains and metrics.

        Args:
            chains: Array of shape (n_chains, n_iterations, n_metrics) or list of chains
            metric_names: Names of metrics (optional)

        Returns:
            Dictionary mapping metric names to convergence statistics
        """
        # Convert list to array if needed
        if isinstance(chains, list):
            chains = np.array(chains)

        # Handle different array shapes
        if chains.ndim == 1:
            chains = chains.reshape(1, -1, 1)
        elif chains.ndim == 2:
            if chains.shape[0] < chains.shape[1]:
                # Assume shape is (n_chains, n_iterations)
                chains = chains.reshape(chains.shape[0], chains.shape[1], 1)
            else:
                # Assume shape is (n_iterations, n_metrics)
                chains = chains.T.reshape(chains.shape[1], chains.shape[0], 1)

        n_chains, n_iterations, n_metrics = chains.shape

        if metric_names is None:
            metric_names = [f"metric_{i}" for i in range(n_metrics)]

        results = {}

        for i, name in enumerate(metric_names):
            metric_chains = chains[:, :, i]

            # Calculate R-hat
            r_hat = self.calculate_r_hat(metric_chains) if n_chains > 1 else 1.0

            # Calculate ESS and MCSE for combined chain
            combined_chain = metric_chains.flatten()
            ess = self.calculate_ess(combined_chain)
            mcse = self.calculate_mcse(combined_chain, ess)

            # Calculate autocorrelation
            autocorr = self._calculate_autocorrelation(combined_chain, 1)[0]

            # Check convergence criteria
            mean_val = np.mean(combined_chain)
            relative_mcse = mcse / abs(mean_val) if mean_val != 0 else np.inf

            converged = (
                r_hat < self.r_hat_threshold
                and ess >= self.min_ess
                and relative_mcse < self.relative_mcse_threshold
            )

            results[name] = ConvergenceStats(
                r_hat=r_hat,
                ess=ess,
                mcse=mcse,
                converged=converged,
                n_iterations=n_iterations * n_chains,
                autocorrelation=autocorr,
            )

        return results

    def _calculate_autocorrelation(self, chain: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function.

        Args:
            chain: 1D array of samples
            max_lag: Maximum lag

        Returns:
            Array of autocorrelations for lags 0 to max_lag
        """
        n = len(chain)
        chain = chain - np.mean(chain)
        c0 = np.dot(chain, chain) / n

        autocorr = np.zeros(max_lag + 1)
        autocorr[0] = 1.0

        for lag in range(1, min(max_lag + 1, n)):
            c_lag = np.dot(chain[:-lag], chain[lag:]) / n
            autocorr[lag] = c_lag / c0 if c0 > 0 else 0

        return autocorr

    def geweke_test(
        self, chain: np.ndarray, first_fraction: float = 0.1, last_fraction: float = 0.5
    ) -> Tuple[float, float]:
        """Perform Geweke convergence test.

        Compares means of first and last portions of chain.

        Args:
            chain: 1D array of samples
            first_fraction: Fraction of chain to use for first portion
            last_fraction: Fraction of chain to use for last portion

        Returns:
            Tuple of (z-score, p-value)
        """
        n = len(chain)
        n_first = int(n * first_fraction)
        n_last = int(n * last_fraction)

        first_portion = chain[:n_first]
        last_portion = chain[-n_last:]

        # Calculate means and spectral density estimates
        mean_first = np.mean(first_portion)
        mean_last = np.mean(last_portion)

        # Simple variance estimates (could use spectral density for more accuracy)
        var_first = np.var(first_portion, ddof=1) / n_first
        var_last = np.var(last_portion, ddof=1) / n_last

        # Calculate z-score
        z_score = (mean_first - mean_last) / np.sqrt(var_first + var_last)

        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return z_score, p_value

    def heidelberger_welch_test(
        self, chain: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, Union[bool, float]]:
        """Perform Heidelberger-Welch stationarity and halfwidth tests.

        Args:
            chain: 1D array of samples
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        n = len(chain)

        # Stationarity test using Cramer-von Mises
        # Simplified version - checks if mean is stable
        window_size = n // 10
        means = []

        for i in range(window_size, n - window_size):
            window_mean = np.mean(chain[i - window_size : i + window_size])
            means.append(window_mean)

        # Check if means are stable
        mean_variance = np.var(means)
        overall_variance = np.var(chain)
        stationarity_ratio = mean_variance / overall_variance if overall_variance > 0 else np.inf

        stationary = stationarity_ratio < 0.1  # Heuristic threshold

        # Halfwidth test
        mean_estimate = np.mean(chain)
        mcse = self.calculate_mcse(chain)
        halfwidth = 1.96 * mcse  # 95% confidence interval halfwidth
        relative_halfwidth = halfwidth / abs(mean_estimate) if mean_estimate != 0 else np.inf

        halfwidth_passed = relative_halfwidth < 0.1  # 10% relative precision

        return {
            "stationary": stationary,
            "stationarity_ratio": stationarity_ratio,
            "halfwidth_passed": halfwidth_passed,
            "relative_halfwidth": relative_halfwidth,
            "mean": mean_estimate,
            "mcse": mcse,
        }

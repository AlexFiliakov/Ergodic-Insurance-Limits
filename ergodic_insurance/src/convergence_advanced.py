"""Advanced convergence diagnostics for Monte Carlo simulations.

This module extends basic convergence diagnostics with advanced features including
autocorrelation analysis, spectral density estimation, and sophisticated ESS calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import fft, signal, stats
from scipy.optimize import minimize_scalar


@dataclass
class SpectralDiagnostics:
    """Container for spectral analysis results.

    Attributes:
        spectral_density: Array of spectral density values
        frequencies: Array of frequency values
        integrated_autocorr_time: Integrated autocorrelation time
        effective_sample_size: ESS from spectral analysis
    """

    spectral_density: np.ndarray
    frequencies: np.ndarray
    integrated_autocorr_time: float
    effective_sample_size: float

    def __str__(self) -> str:
        """String representation of spectral diagnostics."""
        return (
            f"SpectralDiagnostics(tau={self.integrated_autocorr_time:.2f}, "
            f"ess={self.effective_sample_size:.0f})"
        )


@dataclass
class AutocorrelationAnalysis:
    """Container for autocorrelation analysis results.

    Attributes:
        acf_values: Autocorrelation function values
        lags: Lag values
        integrated_time: Integrated autocorrelation time
        initial_monotone_sequence: Length of initial monotone sequence
        initial_positive_sequence: Length of initial positive sequence
    """

    acf_values: np.ndarray
    lags: np.ndarray
    integrated_time: float
    initial_monotone_sequence: int
    initial_positive_sequence: int

    def __str__(self) -> str:
        """String representation of autocorrelation analysis."""
        return (
            f"AutocorrelationAnalysis(tau={self.integrated_time:.2f}, "
            f"monotone={self.initial_monotone_sequence}, "
            f"positive={self.initial_positive_sequence})"
        )


class AdvancedConvergenceDiagnostics:
    """Advanced convergence diagnostics for Monte Carlo simulations.

    Provides sophisticated methods for assessing convergence including
    spectral density estimation, multiple ESS calculation methods, and
    advanced autocorrelation analysis.
    """

    def __init__(self, fft_size: Optional[int] = None):
        """Initialize advanced convergence diagnostics.

        Args:
            fft_size: Size for FFT calculations (None for automatic)
        """
        self.fft_size = fft_size

    def calculate_autocorrelation_full(
        self, chain: np.ndarray, max_lag: Optional[int] = None, method: str = "fft"
    ) -> AutocorrelationAnalysis:
        """Calculate comprehensive autocorrelation analysis.

        Args:
            chain: 1D array of samples
            max_lag: Maximum lag for autocorrelation (None for automatic)
            method: Method for calculation ("fft", "direct", or "biased")

        Returns:
            AutocorrelationAnalysis object with detailed results
        """
        n = len(chain)

        if max_lag is None:
            max_lag = min(n // 4, 1000)

        # Calculate ACF using specified method
        if method == "fft":
            acf_values = self._acf_fft(chain, max_lag)
        elif method == "direct":
            acf_values = self._acf_direct(chain, max_lag)
        elif method == "biased":
            acf_values = self._acf_biased(chain, max_lag)
        else:
            raise ValueError(f"Unknown ACF method: {method}")

        lags = np.arange(max_lag + 1)

        # Find initial monotone sequence (Geyer, 1992)
        monotone_length = self._find_initial_monotone(acf_values)

        # Find initial positive sequence
        positive_length = self._find_initial_positive(acf_values)

        # Calculate integrated autocorrelation time
        tau = self._calculate_integrated_time(acf_values, monotone_length)

        return AutocorrelationAnalysis(
            acf_values=acf_values,
            lags=lags,
            integrated_time=tau,
            initial_monotone_sequence=monotone_length,
            initial_positive_sequence=positive_length,
        )

    def calculate_spectral_density(
        self, chain: np.ndarray, method: str = "welch", nperseg: Optional[int] = None
    ) -> SpectralDiagnostics:
        """Calculate spectral density and related diagnostics.

        Args:
            chain: 1D array of samples
            method: Method for spectral estimation ("welch", "periodogram", "multitaper")
            nperseg: Length of each segment for Welch's method

        Returns:
            SpectralDiagnostics object with spectral analysis results
        """
        n = len(chain)

        # Center the chain
        chain_centered = chain - np.mean(chain)

        if method == "welch":
            # Welch's method for spectral density estimation
            if nperseg is None:
                nperseg = min(n // 8, 256)
            frequencies, psd = signal.welch(
                chain_centered, fs=1.0, nperseg=nperseg, scaling="density"
            )
        elif method == "periodogram":
            # Simple periodogram
            frequencies, psd = signal.periodogram(chain_centered, fs=1.0, scaling="density")
        elif method == "multitaper":
            # Multitaper method (more robust but computationally intensive)
            # Using DPSS windows
            NW = 4  # Time-bandwidth product
            frequencies, psd = self._multitaper_psd(chain_centered, NW=NW)
        else:
            raise ValueError(f"Unknown spectral method: {method}")

        # Calculate integrated autocorrelation time from spectral density
        # tau = S(0) / (2 * sigma^2) where S(0) is spectral density at zero frequency
        variance = np.var(chain, ddof=1)
        if len(psd) > 0 and variance > 0:
            # Extrapolate to zero frequency if needed
            s_zero = psd[0] if frequencies[0] == 0 else self._extrapolate_to_zero(frequencies, psd)
            tau = s_zero / (2 * variance)
        else:
            tau = 1.0

        # Calculate ESS from integrated autocorrelation time
        ess = n / tau if tau > 0 else float(n)

        return SpectralDiagnostics(
            spectral_density=psd,
            frequencies=frequencies,
            integrated_autocorr_time=tau,
            effective_sample_size=min(ess, n),
        )

    def calculate_ess_batch_means(
        self, chain: np.ndarray, batch_size: Optional[int] = None, n_batches: Optional[int] = None
    ) -> float:
        """Calculate ESS using batch means method.

        Args:
            chain: 1D array of samples
            batch_size: Size of each batch (calculated if None)
            n_batches: Number of batches (calculated if None)

        Returns:
            Effective sample size estimate
        """
        n = len(chain)

        # Determine batch parameters
        if batch_size is not None and n_batches is not None:
            if batch_size * n_batches > n:
                raise ValueError("batch_size * n_batches exceeds chain length")
        elif batch_size is not None:
            n_batches = n // batch_size
        elif n_batches is not None:
            batch_size = n // n_batches
        else:
            # Use rule of thumb: batch_size = sqrt(n)
            batch_size = int(np.sqrt(n))
            n_batches = n // batch_size

        if n_batches < 2:
            return float(n)  # Not enough batches

        # Calculate batch means
        batch_means = np.zeros(n_batches)
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_means[i] = np.mean(chain[start_idx:end_idx])

        # Calculate variance of batch means
        var_batch_means = np.var(batch_means, ddof=1)

        # Calculate overall variance
        var_chain = np.var(chain[: n_batches * batch_size], ddof=1)

        # ESS estimate
        if var_batch_means > 0:
            ess = var_chain / var_batch_means * n_batches
        else:
            ess = float(n)

        return float(min(ess, n))

    def calculate_ess_overlapping_batch(
        self, chain: np.ndarray, batch_size: Optional[int] = None
    ) -> float:
        """Calculate ESS using overlapping batch means (more efficient).

        Args:
            chain: 1D array of samples
            batch_size: Size of each batch (calculated if None)

        Returns:
            Effective sample size estimate
        """
        n = len(chain)

        if batch_size is None:
            # Optimal batch size from Flegal & Jones (2010)
            batch_size = int(n ** (1 / 3))

        if batch_size >= n:
            return float(n)

        # Calculate overlapping batch means
        n_batches = n - batch_size + 1
        batch_means = np.zeros(n_batches)

        # Efficient calculation using cumsum
        cumsum = np.cumsum(np.insert(chain, 0, 0))
        batch_sums = cumsum[batch_size:] - cumsum[:-batch_size]
        batch_means = batch_sums / batch_size

        # Calculate batch variance
        var_batch = np.var(batch_means, ddof=1)

        # Calculate overall variance
        var_chain = np.var(chain, ddof=1)

        # ESS estimate with finite sample correction
        if var_batch > 0:
            # Correction factor for overlapping batches
            correction = batch_size / n_batches
            ess = var_chain / (var_batch * correction)
        else:
            ess = float(n)

        return float(min(ess, n))

    def heidelberger_welch_advanced(
        self,
        chain: np.ndarray,
        alpha: float = 0.05,
        eps: float = 0.1,
        pvalue_threshold: float = 0.05,
    ) -> Dict[str, Union[bool, float, int]]:
        """Advanced Heidelberger-Welch stationarity test.

        Args:
            chain: 1D array of samples
            alpha: Significance level for confidence intervals
            eps: Relative precision for halfwidth test
            pvalue_threshold: P-value threshold for stationarity

        Returns:
            Dictionary with detailed test results
        """
        n = len(chain)

        # Perform stationarity test using Cramer-von Mises statistic
        stationary, start_iter, pvalue = self._stationarity_test(chain, pvalue_threshold)

        if stationary:
            # Use stationary portion for further analysis
            stationary_chain = chain[start_iter:]

            # Calculate mean and MCSE
            mean_est = np.mean(stationary_chain)

            # Use spectral density at zero for variance estimation
            spec_diag = self.calculate_spectral_density(stationary_chain, method="welch")
            var_est = (
                spec_diag.spectral_density[0]
                if len(spec_diag.spectral_density) > 0
                else np.var(stationary_chain)
            )

            # MCSE using spectral method
            mcse = np.sqrt(var_est / len(stationary_chain))

            # Halfwidth test
            z_score = stats.norm.ppf(1 - alpha / 2)
            halfwidth = z_score * mcse
            relative_halfwidth = halfwidth / abs(mean_est) if mean_est != 0 else np.inf
            halfwidth_passed = relative_halfwidth < eps

            # Calculate integrated autocorrelation time
            tau = spec_diag.integrated_autocorr_time

        else:
            # Non-stationary chain
            stationary_chain = chain
            mean_est = np.mean(chain)
            mcse = np.std(chain) / np.sqrt(n)
            halfwidth = 1.96 * mcse
            relative_halfwidth = np.inf
            halfwidth_passed = False
            tau = np.nan

        return {
            "stationary": stationary,
            "start_iteration": start_iter,
            "pvalue": pvalue,
            "halfwidth_passed": halfwidth_passed,
            "relative_halfwidth": relative_halfwidth,
            "mean": mean_est,
            "mcse": mcse,
            "halfwidth": halfwidth,
            "integrated_autocorr_time": tau,
            "n_usable": len(stationary_chain),
            "n_discarded": start_iter,
        }

    def raftery_lewis_diagnostic(
        self, chain: np.ndarray, q: float = 0.025, r: float = 0.005, s: float = 0.95
    ) -> Dict[str, float]:
        """Raftery-Lewis diagnostic for required chain length.

        Args:
            chain: 1D array of samples
            q: Quantile of interest
            r: Desired accuracy
            s: Probability of achieving accuracy

        Returns:
            Dictionary with diagnostic results
        """
        n = len(chain)

        # Dichotomize the chain
        cutoff = np.quantile(chain, q)
        binary_chain = (chain <= cutoff).astype(int)

        # Calculate transition probabilities
        transitions = np.zeros((2, 2))
        for i in range(n - 1):
            transitions[binary_chain[i], binary_chain[i + 1]] += 1

        # Normalize to get probabilities
        for i in range(2):
            row_sum = transitions[i].sum()
            if row_sum > 0:
                transitions[i] /= row_sum

        # Calculate required burn-in and total iterations
        # Using simplified Raftery-Lewis formulae
        alpha = transitions[0, 1] if transitions[0, 0] < 1 else 0
        beta = transitions[1, 0] if transitions[1, 1] < 1 else 0

        if alpha > 0 and beta > 0:
            # Stationary distribution
            pi = alpha / (alpha + beta)

            # Required precision
            phi = stats.norm.ppf((s + 1) / 2)

            # Burn-in requirement
            m_burn = np.log((alpha + beta) * r / max(alpha, beta)) / np.log(abs(1 - alpha - beta))
            m_burn = max(1, int(np.ceil(m_burn)))

            # Total iterations required
            n_min = (q * (1 - q) * phi**2) / r**2

            # Thinning requirement
            k_thin = 1 + 2 * (alpha + beta) / (alpha * beta)
            k_thin = max(1, int(np.ceil(k_thin)))

            # Total iterations with thinning
            n_total = m_burn + n_min * k_thin

            # Dependence factor
            i_stat = n_total / n_min if n_min > 0 else np.inf

        else:
            # Chain appears to be stuck
            m_burn = n
            n_min = n * 10
            k_thin = 1
            n_total = n * 10
            i_stat = np.inf

        return {
            "burn_in": m_burn,
            "n_min": n_min,
            "thinning": k_thin,
            "n_total": n_total,
            "dependence_factor": i_stat,
            "n_current": n,
            "sufficient": n >= n_total,
        }

    # Private helper methods

    def _acf_fft(self, chain: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate ACF using FFT (fast for long chains)."""
        n = len(chain)

        # Handle edge cases
        if n == 0:
            return np.array([np.nan])
        if n == 1:
            return np.array([1.0])

        chain_centered = chain - np.mean(chain)

        # Check for zero variance
        if np.var(chain_centered) == 0:
            return np.ones(min(max_lag + 1, n))

        # Pad for FFT
        c_padded = np.concatenate([chain_centered, np.zeros(n)])

        # FFT-based autocorrelation
        f = fft.fft(c_padded)
        acf_full = fft.ifft(f * np.conj(f))[:n].real

        # Normalize (avoid division by zero)
        if acf_full[0] != 0:
            acf_full = acf_full / acf_full[0]
        else:
            acf_full = np.ones_like(acf_full)

        return np.array(acf_full[: min(max_lag + 1, n)])

    def _acf_direct(self, chain: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate ACF using direct method (unbiased)."""
        n = len(chain)
        chain_centered = chain - np.mean(chain)
        c0 = np.dot(chain_centered, chain_centered) / n

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, min(max_lag + 1, n)):
            c_lag = np.dot(chain_centered[:-lag], chain_centered[lag:]) / (n - lag)
            acf[lag] = c_lag / c0 if c0 > 0 else 0

        return acf

    def _acf_biased(self, chain: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate ACF using biased estimator (denominator n)."""
        n = len(chain)
        chain_centered = chain - np.mean(chain)
        c0 = np.dot(chain_centered, chain_centered) / n

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, min(max_lag + 1, n)):
            c_lag = np.dot(chain_centered[:-lag], chain_centered[lag:]) / n
            acf[lag] = c_lag / c0 if c0 > 0 else 0

        return acf

    def _find_initial_monotone(self, acf: np.ndarray) -> int:
        """Find initial monotone sequence in ACF (Geyer, 1992)."""
        n = len(acf)

        # Look at sums of consecutive pairs
        for i in range(1, n - 1, 2):
            if acf[i] + acf[i + 1] < 0:
                return i - 1
            if i > 1 and acf[i - 1] + acf[i] < acf[i] + acf[i + 1]:
                return i - 1

        return n - 1

    def _find_initial_positive(self, acf: np.ndarray) -> int:
        """Find initial positive sequence in ACF."""
        for i in range(1, len(acf)):
            if acf[i] < 0:
                return i - 1
        return len(acf) - 1

    def _calculate_integrated_time(self, acf: np.ndarray, cutoff: int) -> float:
        """Calculate integrated autocorrelation time."""
        # Use Geyer's initial positive sequence estimator
        tau = 1.0  # Start with lag 0

        for i in range(1, cutoff, 2):
            if i + 1 < cutoff:
                pair_sum = acf[i] + acf[i + 1]
                if pair_sum > 0:
                    tau += pair_sum
                else:
                    break
            else:
                if acf[i] > 0:
                    tau += acf[i]

        return tau

    def _multitaper_psd(
        self, data_signal: np.ndarray, NW: float = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate PSD using multitaper method."""
        n = len(data_signal)

        # Generate DPSS tapers
        from scipy.signal.windows import dpss

        tapers, concentrations = dpss(n, NW, NW * 2 - 1, return_ratios=True)

        # Calculate periodogram for each taper
        psds = []
        for taper in tapers:
            windowed = data_signal * taper
            freqs, psd = signal.periodogram(windowed, fs=1.0)
            psds.append(psd)

        # Average PSDs weighted by concentration ratios
        psd_avg = np.average(psds, axis=0, weights=concentrations)

        return freqs, psd_avg

    def _extrapolate_to_zero(self, frequencies: np.ndarray, psd: np.ndarray) -> float:
        """Extrapolate spectral density to zero frequency."""
        # Use first few points to extrapolate
        n_points = min(5, len(frequencies))
        if n_points < 2:
            return float(psd[0])

        # Linear extrapolation in log-log space if possible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                log_freq = np.log(frequencies[:n_points] + 1e-10)
                log_psd = np.log(psd[:n_points] + 1e-10)
                slope, intercept = np.polyfit(log_freq, log_psd, 1)
                return float(np.exp(intercept))
            except Exception:  # pylint: disable=broad-exception-caught
                # Fall back to simple extrapolation
                return float(psd[0])

    def _stationarity_test(  # pylint: disable=too-many-locals
        self, chain: np.ndarray, pvalue_threshold: float
    ) -> Tuple[bool, int, float]:
        """Perform Cramer-von Mises stationarity test."""
        n = len(chain)

        # Test different starting points
        test_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        for frac in test_fractions:
            start_idx = int(n * frac)
            if start_idx >= n - 10:  # Need at least 10 samples
                continue

            test_chain = chain[start_idx:]

            # Simplified stationarity test using batches
            n_test = len(test_chain)
            batch_size = max(10, n_test // 20)
            n_batches = n_test // batch_size

            if n_batches < 2:
                continue

            batch_means = []
            for i in range(n_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch_means.append(np.mean(test_chain[batch_start:batch_end]))

            # Test if batch means are stationary
            # Using simple ANOVA-like test
            overall_mean = np.mean(batch_means)
            within_var = np.var(batch_means)

            # Between batch variance (time trend test)
            time_indices = np.arange(n_batches)
            correlation = np.corrcoef(time_indices, batch_means)[0, 1]

            # Simple p-value based on correlation
            t_stat = (
                correlation * np.sqrt(n_batches - 2) / np.sqrt(1 - correlation**2)
                if abs(correlation) < 1
                else 0
            )
            pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n_batches - 2))

            if pvalue > pvalue_threshold:
                return True, start_idx, pvalue

        # Failed all tests
        return False, 0, 0.0

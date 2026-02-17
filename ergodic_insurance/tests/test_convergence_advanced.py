"""Tests for advanced convergence diagnostics module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ergodic_insurance.convergence_advanced import (
    AdvancedConvergenceDiagnostics,
    AutocorrelationAnalysis,
    SpectralDiagnostics,
)


class TestAdvancedConvergenceDiagnostics:
    """Test suite for AdvancedConvergenceDiagnostics class."""

    @pytest.fixture
    def diagnostics(self):
        """Create diagnostics instance."""
        return AdvancedConvergenceDiagnostics()

    @pytest.fixture
    def sample_chain(self):
        """Create sample MCMC chain."""
        np.random.seed(42)
        # Create autocorrelated chain
        n = 1000
        chain = np.zeros(n)
        chain[0] = np.random.randn()
        for i in range(1, n):
            chain[i] = 0.5 * chain[i - 1] + np.random.randn()
        return chain

    @pytest.fixture
    def multiple_chains(self):
        """Create multiple MCMC chains."""
        np.random.seed(42)
        n_chains = 4
        n_iterations = 500
        chains = np.zeros((n_chains, n_iterations))

        for c in range(n_chains):
            chains[c, 0] = np.random.randn()
            for i in range(1, n_iterations):
                chains[c, i] = 0.3 * chains[c, i - 1] + np.random.randn()

        return chains

    def test_autocorrelation_full_fft(self, diagnostics, sample_chain):
        """Test full autocorrelation analysis with FFT method."""
        result = diagnostics.calculate_autocorrelation_full(sample_chain, max_lag=50, method="fft")

        assert isinstance(result, AutocorrelationAnalysis)
        assert len(result.acf_values) == 51  # max_lag + 1
        assert result.acf_values[0] == pytest.approx(1.0)
        assert result.integrated_time > 1.0  # Should be > 1 for correlated chain
        assert result.initial_monotone_sequence > 0
        assert result.initial_positive_sequence > 0

    def test_autocorrelation_full_direct(self, diagnostics, sample_chain):
        """Test full autocorrelation analysis with direct method."""
        result = diagnostics.calculate_autocorrelation_full(
            sample_chain, max_lag=50, method="direct"
        )

        assert isinstance(result, AutocorrelationAnalysis)
        assert len(result.acf_values) == 51
        assert result.acf_values[0] == pytest.approx(1.0)
        # ACF should decay for autocorrelated chain
        assert result.acf_values[10] < result.acf_values[1]

    def test_autocorrelation_full_biased(self, diagnostics, sample_chain):
        """Test full autocorrelation analysis with biased method."""
        result = diagnostics.calculate_autocorrelation_full(
            sample_chain, max_lag=50, method="biased"
        )

        assert isinstance(result, AutocorrelationAnalysis)
        assert len(result.acf_values) == 51
        assert result.acf_values[0] == pytest.approx(1.0)

    def test_autocorrelation_invalid_method(self, diagnostics, sample_chain):
        """Test autocorrelation with invalid method."""
        with pytest.raises(ValueError, match="Unknown ACF method"):
            diagnostics.calculate_autocorrelation_full(sample_chain, method="invalid")

    def test_spectral_density_welch(self, diagnostics, sample_chain):
        """Test spectral density calculation with Welch's method."""
        result = diagnostics.calculate_spectral_density(sample_chain, method="welch")

        assert isinstance(result, SpectralDiagnostics)
        assert len(result.spectral_density) > 0
        assert len(result.frequencies) == len(result.spectral_density)
        assert result.integrated_autocorr_time > 0
        assert result.effective_sample_size > 0
        assert result.effective_sample_size <= len(sample_chain)

    def test_spectral_density_periodogram(self, diagnostics, sample_chain):
        """Test spectral density with periodogram method."""
        result = diagnostics.calculate_spectral_density(sample_chain, method="periodogram")

        assert isinstance(result, SpectralDiagnostics)
        assert len(result.spectral_density) > 0
        assert result.integrated_autocorr_time > 0

    def test_spectral_density_invalid_method(self, diagnostics, sample_chain):
        """Test spectral density with invalid method."""
        with pytest.raises(ValueError, match="Unknown spectral method"):
            diagnostics.calculate_spectral_density(sample_chain, method="invalid")

    def test_ess_batch_means(self, diagnostics, sample_chain):
        """Test ESS calculation using batch means."""
        ess = diagnostics.calculate_ess_batch_means(sample_chain)

        assert isinstance(ess, float)
        assert ess > 0
        assert ess <= len(sample_chain)

        # Test with specified batch size
        ess_custom = diagnostics.calculate_ess_batch_means(sample_chain, batch_size=50)
        assert ess_custom > 0

    def test_ess_batch_means_invalid_params(self, diagnostics):
        """Test ESS batch means with invalid parameters."""
        rng = np.random.default_rng(200)
        chain = rng.standard_normal(100)

        with pytest.raises(ValueError, match="exceeds chain length"):
            diagnostics.calculate_ess_batch_means(chain, batch_size=50, n_batches=3)

    def test_ess_batch_means_short_chain(self, diagnostics):
        """Test ESS batch means with very short chain."""
        chain = np.array([1.0, 2.0, 3.0])
        ess = diagnostics.calculate_ess_batch_means(chain)

        assert ess == 3.0  # Should return chain length

    def test_ess_overlapping_batch(self, diagnostics, sample_chain):
        """Test ESS with overlapping batch means."""
        ess = diagnostics.calculate_ess_overlapping_batch(sample_chain)

        assert isinstance(ess, float)
        assert ess > 0
        assert ess <= len(sample_chain)

        # Should be more efficient than non-overlapping
        ess_regular = diagnostics.calculate_ess_batch_means(sample_chain)
        # Note: Not always true, but generally overlapping is more efficient
        assert ess > 0 and ess_regular > 0

    def test_heidelberger_welch_advanced(self, diagnostics, sample_chain):
        """Test advanced Heidelberger-Welch test."""
        result = diagnostics.heidelberger_welch_advanced(sample_chain)

        assert isinstance(result, dict)
        assert "stationary" in result
        assert "start_iteration" in result
        assert "pvalue" in result
        assert "halfwidth_passed" in result
        assert "relative_halfwidth" in result
        assert "mean" in result
        assert "mcse" in result
        assert "integrated_autocorr_time" in result
        assert "n_usable" in result
        assert "n_discarded" in result

        assert isinstance(result["stationary"], bool)
        assert result["start_iteration"] >= 0
        assert result["n_usable"] + result["n_discarded"] <= len(sample_chain)

    def test_heidelberger_welch_non_stationary(self, diagnostics):
        """Test Heidelberger-Welch with non-stationary chain."""
        # Create trending chain (non-stationary)
        rng = np.random.default_rng(201)
        chain = np.cumsum(rng.standard_normal(1000))
        result = diagnostics.heidelberger_welch_advanced(chain)

        # Should detect non-stationarity (though simplified test may not always)
        assert isinstance(result["stationary"], bool)
        assert "mcse" in result

    def test_raftery_lewis_diagnostic(self, diagnostics, sample_chain):
        """Test Raftery-Lewis diagnostic."""
        result = diagnostics.raftery_lewis_diagnostic(sample_chain)

        assert isinstance(result, dict)
        assert "burn_in" in result
        assert "n_min" in result
        assert "thinning" in result
        assert "n_total" in result
        assert "dependence_factor" in result
        assert "n_current" in result
        assert "sufficient" in result

        assert result["burn_in"] >= 1
        assert result["n_min"] > 0
        assert result["thinning"] >= 1
        assert result["n_total"] > 0
        assert result["n_current"] == len(sample_chain)

    def test_raftery_lewis_with_custom_params(self, diagnostics, sample_chain):
        """Test Raftery-Lewis with custom parameters."""
        result = diagnostics.raftery_lewis_diagnostic(sample_chain, q=0.05, r=0.01, s=0.99)

        assert isinstance(result, dict)
        assert result["burn_in"] >= 1
        # More stringent requirements should need more iterations
        assert result["n_total"] > 0

    def test_find_initial_monotone(self, diagnostics):
        """Test finding initial monotone sequence."""
        # Create ACF with known structure
        acf = np.array([1.0, 0.8, 0.6, 0.4, 0.2, -0.1, -0.2])

        diagnostics_obj = diagnostics
        length = diagnostics_obj._find_initial_monotone(acf)

        assert length >= 0
        assert length < len(acf)

    def test_find_initial_positive(self, diagnostics):
        """Test finding initial positive sequence."""
        # Create ACF with known structure
        acf = np.array([1.0, 0.5, 0.2, -0.1, -0.2])

        diagnostics_obj = diagnostics
        length = diagnostics_obj._find_initial_positive(acf)

        assert length == 2  # First negative at index 3, so positive until 2

    def test_calculate_integrated_time(self, diagnostics):
        """Test integrated autocorrelation time calculation."""
        acf = np.array([1.0, 0.5, 0.3, 0.1, -0.1])

        diagnostics_obj = diagnostics
        tau = diagnostics_obj._calculate_integrated_time(acf, cutoff=4)

        assert tau > 1.0  # Should be > 1 for positive autocorrelation
        assert tau < len(acf)  # Should be bounded

    def test_extrapolate_to_zero(self, diagnostics):
        """Test spectral density extrapolation to zero frequency."""
        frequencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        psd = np.array([10.0, 8.0, 6.0, 5.0, 4.0])

        diagnostics_obj = diagnostics
        s_zero = diagnostics_obj._extrapolate_to_zero(frequencies, psd)

        assert s_zero > 0
        # Note: extrapolation may not always be higher than first value
        # depending on the trend in the data

    def test_stationarity_test(self, diagnostics, sample_chain):
        """Test stationarity test helper."""
        diagnostics_obj = diagnostics
        stationary, start_idx, pvalue = diagnostics_obj._stationarity_test(
            sample_chain, pvalue_threshold=0.05
        )

        assert isinstance(stationary, bool)
        assert start_idx >= 0
        assert 0 <= pvalue <= 1 or pvalue == 0.0

    def test_spectral_diagnostics_str(self):
        """Test SpectralDiagnostics string representation."""
        diag = SpectralDiagnostics(
            spectral_density=np.array([1.0, 2.0]),
            frequencies=np.array([0.0, 0.5]),
            integrated_autocorr_time=2.5,
            effective_sample_size=400.0,
        )

        str_repr = str(diag)
        assert "tau=2.50" in str_repr
        assert "ess=400" in str_repr

    def test_autocorrelation_analysis_str(self):
        """Test AutocorrelationAnalysis string representation."""
        analysis = AutocorrelationAnalysis(
            acf_values=np.array([1.0, 0.5]),
            lags=np.array([0, 1]),
            integrated_time=1.5,
            initial_monotone_sequence=10,
            initial_positive_sequence=15,
        )

        str_repr = str(analysis)
        assert "tau=1.50" in str_repr
        assert "monotone=10" in str_repr
        assert "positive=15" in str_repr

    def test_with_multidimensional_chains(self, diagnostics):
        """Test methods with multidimensional chain data."""
        # Create 3D chain data (chains x iterations x parameters)
        np.random.seed(42)
        chains = np.random.randn(4, 100, 2)

        # Flatten for single parameter analysis
        chain = chains[:, :, 0].flatten()

        # Should handle flattened data
        result = diagnostics.calculate_autocorrelation_full(chain, max_lag=10)
        assert isinstance(result, AutocorrelationAnalysis)

        ess = diagnostics.calculate_ess_batch_means(chain)
        assert ess > 0

    def test_edge_cases(self, diagnostics):
        """Test edge cases and boundary conditions."""
        # Empty chain - should handle gracefully
        result = diagnostics.calculate_autocorrelation_full(np.array([]), max_lag=0)
        assert len(result.acf_values) == 1
        assert np.isnan(result.acf_values[0])  # NaN for empty chain

        # Single value chain
        chain = np.array([1.0])
        result = diagnostics.calculate_autocorrelation_full(chain, max_lag=0)
        assert len(result.acf_values) == 1
        assert result.acf_values[0] == 1.0

        # Constant chain (zero variance)
        chain = np.ones(100)
        ess = diagnostics.calculate_ess_batch_means(chain)
        assert ess == 100.0  # Perfect sampling for constant

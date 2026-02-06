"""Coverage-targeted tests for convergence_advanced.py.

Targets specific uncovered lines: 146-147, 159, 193, 200, 242, 265,
405-409, 437, 450, 515, 526-543, 550, 560-562, 576, 586.
"""

from unittest.mock import patch

import numpy as np
import pytest

from ergodic_insurance.convergence_advanced import (
    AdvancedConvergenceDiagnostics,
    AutocorrelationAnalysis,
    SpectralDiagnostics,
)


@pytest.fixture
def diagnostics():
    """Create an AdvancedConvergenceDiagnostics instance."""
    return AdvancedConvergenceDiagnostics()


@pytest.fixture
def autocorrelated_chain():
    """Create an autocorrelated chain for testing."""
    rng = np.random.default_rng(42)
    n = 1000
    chain = np.zeros(n)
    chain[0] = rng.standard_normal()
    for i in range(1, n):
        chain[i] = 0.5 * chain[i - 1] + rng.standard_normal()
    return chain


# ---------------------------------------------------------------------------
# Spectral density: multitaper method (lines 146-147)
# ---------------------------------------------------------------------------
class TestSpectralDensityMultitaper:
    """Test calculate_spectral_density with multitaper method."""

    def test_multitaper_method(self, diagnostics, autocorrelated_chain):
        """Lines 146-147: multitaper method in calculate_spectral_density."""
        result = diagnostics.calculate_spectral_density(autocorrelated_chain, method="multitaper")

        assert isinstance(result, SpectralDiagnostics)
        assert len(result.spectral_density) > 0
        assert len(result.frequencies) == len(result.spectral_density)
        assert result.integrated_autocorr_time > 0
        assert result.effective_sample_size > 0
        assert result.effective_sample_size <= len(autocorrelated_chain)


# ---------------------------------------------------------------------------
# Spectral density: zero variance tau fallback (line 159)
# ---------------------------------------------------------------------------
class TestSpectralDensityZeroVariance:
    """Test spectral density with zero variance chain."""

    def test_tau_fallback_for_zero_variance(self, diagnostics):
        """Line 159: tau defaults to 1.0 when variance is zero."""
        chain = np.ones(100)  # Zero variance
        result = diagnostics.calculate_spectral_density(chain, method="welch")

        # With zero variance, tau should default to 1.0
        assert result.integrated_autocorr_time == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ESS batch means: n_batches parameter (line 193)
# ---------------------------------------------------------------------------
class TestESSBatchMeansNBatches:
    """Test ESS batch means with n_batches parameter."""

    def test_batch_size_from_n_batches(self, diagnostics, autocorrelated_chain):
        """Line 193: batch_size calculated from n_batches."""
        ess = diagnostics.calculate_ess_batch_means(autocorrelated_chain, n_batches=10)
        assert isinstance(ess, float)
        assert ess > 0
        assert ess <= len(autocorrelated_chain)

    def test_n_batches_less_than_2(self, diagnostics):
        """Line 200: Returns chain length when n_batches < 2."""
        chain = np.array([1.0, 2.0, 3.0])
        # batch_size=3, n_batches=1 means we can only have 1 batch
        ess = diagnostics.calculate_ess_batch_means(chain, batch_size=3)
        assert ess == float(len(chain))


# ---------------------------------------------------------------------------
# ESS overlapping batch: batch_size >= n (line 242), var_batch == 0 (line 265)
# ---------------------------------------------------------------------------
class TestESSOverlappingBatchEdgeCases:
    """Test ESS overlapping batch means edge cases."""

    def test_batch_size_exceeds_chain_length(self, diagnostics):
        """Line 242: Returns chain length when batch_size >= n."""
        chain = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ess = diagnostics.calculate_ess_overlapping_batch(chain, batch_size=10)
        assert ess == float(len(chain))

    def test_zero_batch_variance(self, diagnostics):
        """Line 265: Returns chain length when batch variance is zero."""
        # Constant chain => zero variance in batch means
        chain = np.ones(100)
        ess = diagnostics.calculate_ess_overlapping_batch(chain, batch_size=10)
        assert ess == float(len(chain))


# ---------------------------------------------------------------------------
# Raftery-Lewis: stuck chain (lines 405-409)
# ---------------------------------------------------------------------------
class TestRafteryLewisStuckChain:
    """Test Raftery-Lewis diagnostic with a stuck chain."""

    def test_stuck_chain(self, diagnostics):
        """Lines 405-409: Chain that appears stuck (alpha=0 or beta=0)."""
        # All values the same: transitions will have alpha=0, beta=0
        chain = np.ones(200)
        result = diagnostics.raftery_lewis_diagnostic(chain)

        assert result["burn_in"] == len(chain)
        assert result["n_total"] == len(chain) * 10
        assert result["dependence_factor"] == np.inf


# ---------------------------------------------------------------------------
# ACF FFT: zero variance chain (line 437)
# ---------------------------------------------------------------------------
class TestACFFftZeroVariance:
    """Test _acf_fft with zero variance chain."""

    def test_zero_variance_returns_ones(self, diagnostics):
        """Line 437: Returns array of ones for zero variance chain."""
        chain = np.ones(50)
        acf = diagnostics._acf_fft(chain, max_lag=10)
        np.testing.assert_array_equal(acf, np.ones(11))


# ---------------------------------------------------------------------------
# ACF FFT: acf_full[0] == 0 (line 450)
# ---------------------------------------------------------------------------
class TestACFFftZeroAutocorrelation:
    """Test _acf_fft when acf_full[0] is zero."""

    def test_acf_zero_at_origin(self, diagnostics):
        """Line 450: Returns ones_like when acf_full[0] is zero."""
        # This is hard to trigger naturally - patch the FFT result
        chain = np.array([1.0, -1.0, 1.0, -1.0, 1.0])

        with patch("ergodic_insurance.convergence_advanced.fft") as mock_fft:
            # Make the IFFT return an array where first element is 0
            mock_fft.fft.return_value = np.zeros(10, dtype=complex)
            mock_fft.ifft.return_value = np.zeros(10, dtype=complex)

            acf = diagnostics._acf_fft(chain, max_lag=3)
            # When acf_full[0] == 0, should return ones
            np.testing.assert_array_equal(acf, np.ones(4))


# ---------------------------------------------------------------------------
# _calculate_integrated_time: pair_sum <= 0 break (line 515)
# ---------------------------------------------------------------------------
class TestCalculateIntegratedTimeBreak:
    """Test _calculate_integrated_time stopping when pair_sum <= 0."""

    def test_early_break_on_negative_pair_sum(self, diagnostics):
        """Line 515: Loop breaks when pair_sum becomes non-positive."""
        # ACF that starts positive then goes negative
        acf = np.array([1.0, 0.5, 0.3, -0.2, -0.4, -0.5, -0.6])
        tau = diagnostics._calculate_integrated_time(acf, cutoff=7)
        # Should include pairs (0.5, 0.3) but stop at (-0.2, -0.4)
        assert tau == pytest.approx(1.0 + 0.5 + 0.3, abs=0.01)


# ---------------------------------------------------------------------------
# _multitaper_psd (lines 526-543)
# ---------------------------------------------------------------------------
class TestMultitaperPSD:
    """Test _multitaper_psd method directly."""

    def test_multitaper_returns_frequencies_and_psd(self, diagnostics):
        """Lines 526-543: _multitaper_psd returns valid frequencies and PSD."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(256)

        freqs, psd = diagnostics._multitaper_psd(data, NW=4)

        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(psd >= 0)  # PSD should be non-negative

    def test_multitaper_short_signal(self, diagnostics):
        """_multitaper_psd works with shorter signals."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(64)

        freqs, psd = diagnostics._multitaper_psd(data, NW=3)
        assert len(freqs) > 0
        assert len(psd) == len(freqs)


# ---------------------------------------------------------------------------
# _extrapolate_to_zero: fewer than 2 points (line 550)
# ---------------------------------------------------------------------------
class TestExtrapolateToZeroFewPoints:
    """Test _extrapolate_to_zero with fewer than 2 points."""

    def test_single_point_returns_psd0(self, diagnostics):
        """Line 550: Returns psd[0] when n_points < 2."""
        frequencies = np.array([0.1])
        psd = np.array([5.0])

        result = diagnostics._extrapolate_to_zero(frequencies, psd)
        assert result == 5.0


# ---------------------------------------------------------------------------
# _extrapolate_to_zero: exception fallback (lines 560-562)
# ---------------------------------------------------------------------------
class TestExtrapolateToZeroException:
    """Test _extrapolate_to_zero exception handling."""

    def test_polyfit_failure_fallback(self, diagnostics):
        """Lines 560-562: Falls back to psd[0] when polyfit raises an exception."""
        frequencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        psd = np.array([10.0, 8.0, 6.0, 5.0, 4.0])

        # Patch np.polyfit to raise an exception
        with patch("numpy.polyfit", side_effect=np.linalg.LinAlgError("singular")):
            result = diagnostics._extrapolate_to_zero(frequencies, psd)
        # Should fall back to psd[0]
        assert result == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _stationarity_test: start_idx >= n - 10 (line 576)
# ---------------------------------------------------------------------------
class TestStationarityTestShortChain:
    """Test _stationarity_test edge cases."""

    def test_very_short_chain_skips_fractions(self, diagnostics):
        """Line 576: Fractions where start_idx >= n-10 are skipped."""
        # With 15 samples, frac=0.5 gives start_idx=7, which means
        # we have only 8 samples remaining. But n-10=5, so 7 >= 5
        # This exercises the continue on line 576.
        chain = np.random.default_rng(42).standard_normal(15)
        stationary, start_idx, pvalue = diagnostics._stationarity_test(chain, 0.05)
        assert isinstance(stationary, bool)

    def test_n_batches_less_than_2_skips(self, diagnostics):
        """Line 586: Skips when n_batches < 2 in stationarity test."""
        # Very short chain where batch computation yields < 2 batches
        chain = np.random.default_rng(42).standard_normal(20)
        stationary, start_idx, pvalue = diagnostics._stationarity_test(chain, 0.05)
        assert isinstance(stationary, bool)


# ---------------------------------------------------------------------------
# Heidelberger-Welch: non-stationary result (tested indirectly but ensuring
# the non-stationary branch is exercised)
# ---------------------------------------------------------------------------
class TestHeidelbergerWelchNonStationary:
    """Ensure non-stationary branch is fully covered."""

    def test_non_stationary_chain_properties(self, diagnostics):
        """Test all return values when chain is non-stationary."""
        # Strong trend makes chain non-stationary
        chain = np.cumsum(np.ones(200))
        result = diagnostics.heidelberger_welch_advanced(chain, pvalue_threshold=0.99)

        assert "stationary" in result
        assert "mcse" in result
        assert "halfwidth" in result
        assert result["n_usable"] + result["n_discarded"] <= len(chain) + 1


# ---------------------------------------------------------------------------
# Integration test: full pipeline coverage
# ---------------------------------------------------------------------------
class TestFullPipelineConvergence:
    """Integration test exercising multiple methods together."""

    def test_well_converged_chain(self, diagnostics):
        """Test all diagnostics on a well-behaved chain."""
        rng = np.random.default_rng(42)
        chain = rng.standard_normal(2000)

        # ACF
        acf = diagnostics.calculate_autocorrelation_full(chain, max_lag=50)
        assert acf.acf_values[0] == pytest.approx(1.0)

        # Spectral - welch
        spec = diagnostics.calculate_spectral_density(chain, method="welch")
        assert spec.effective_sample_size > 100

        # ESS batch
        ess_batch = diagnostics.calculate_ess_batch_means(chain)
        assert ess_batch > 100

        # ESS overlapping
        ess_overlap = diagnostics.calculate_ess_overlapping_batch(chain)
        assert ess_overlap > 100

        # Heidelberger-Welch
        hw = diagnostics.heidelberger_welch_advanced(chain)
        assert isinstance(hw["stationary"], bool)
        # A well-converged constant chain should be detected as stationary
        assert hw["stationary"] is True

        # Raftery-Lewis
        rl = diagnostics.raftery_lewis_diagnostic(chain)
        assert rl["n_current"] == 2000


# ---------------------------------------------------------------------------
# ESS batch means: both batch_size and n_batches provided
# ---------------------------------------------------------------------------
class TestESSBatchMeansBothParams:
    """Test ESS batch means when both batch_size and n_batches are provided."""

    def test_valid_both_params(self, diagnostics, autocorrelated_chain):
        """batch_size and n_batches both specified and valid."""
        ess = diagnostics.calculate_ess_batch_means(
            autocorrelated_chain, batch_size=50, n_batches=10
        )
        assert ess > 0
        assert ess <= len(autocorrelated_chain)

    def test_exceeds_chain_length(self, diagnostics):
        """Raises when batch_size * n_batches > chain length."""
        chain = np.random.default_rng(42).standard_normal(100)
        with pytest.raises(ValueError, match="exceeds chain length"):
            diagnostics.calculate_ess_batch_means(chain, batch_size=50, n_batches=3)


# ---------------------------------------------------------------------------
# _calculate_integrated_time: odd cutoff, single tail ACF value (line 517-518)
# ---------------------------------------------------------------------------
class TestCalculateIntegratedTimeOddCutoff:
    """Test _calculate_integrated_time with odd cutoff."""

    def test_odd_cutoff_adds_single_acf(self, diagnostics):
        """Line 517-518: Adds single acf[i] when i+1 >= cutoff."""
        # cutoff=3, so pairs checked: i=1 (acf[1]+acf[2]), then i=3 >= cutoff => done
        # But with cutoff=4: pairs at i=1 (acf[1]+acf[2]), then i=3: i+1=4 >= cutoff
        # so we enter the else branch and add acf[3] alone
        acf = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        tau = diagnostics._calculate_integrated_time(acf, cutoff=4)
        # tau = 1.0 (lag0) + 0.8 + 0.6 (pair at i=1) + 0.4 (single at i=3)
        assert tau == pytest.approx(1.0 + 0.8 + 0.6 + 0.4, abs=0.01)

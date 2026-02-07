"""Regression tests for spectral density ESS fix (issue #476).

The integrated autocorrelation time (tau) was underestimated because:
1. The formula divided by ``2 * variance`` instead of ``variance``.
2. ``signal.welch`` per-segment detrending (``detrend='constant'``) suppressed
   the PSD near zero frequency for autocorrelated data.

scipy.signal.welch returns a one-sided PSD whose DC component is NOT doubled,
so the correct relation is:

    tau = PSD_one_sided(0) / Var(X)

and welch must be called with ``detrend=False`` when the chain has already been
globally centered.

These tests verify:
1. AR(1) chain with known tau: spectral ESS matches analytical ESS
2. Heidelberger-Welch advanced test still produces correct results
3. The spectral density convention (one-sided PSD) is respected
"""

import numpy as np
import pytest

from ergodic_insurance.convergence_advanced import AdvancedConvergenceDiagnostics


@pytest.fixture
def adv():
    return AdvancedConvergenceDiagnostics()


def _make_ar1_chain(n: int, rho: float, seed: int = 42) -> np.ndarray:
    """Generate an AR(1) chain: X_t = rho * X_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    chain = np.zeros(n)
    chain[0] = rng.standard_normal()
    for i in range(1, n):
        chain[i] = rho * chain[i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal()
    return chain


class TestSpectralTauFormula:
    """Verify the corrected tau = S(0) / variance formula."""

    def test_ar1_spectral_ess_matches_analytical(self, adv):
        """For AR(1) with parameter rho, the analytical tau is (1+rho)/(1-rho).

        The spectral ESS should be within reasonable tolerance of n / tau_analytical.
        With the old bug (factor of 2 + per-segment detrending), ess_spectral
        would be ~4-6x too high and fail this check.
        """
        n = 10_000
        rho = 0.7
        chain = _make_ar1_chain(n, rho, seed=123)

        # Analytical values for AR(1)
        tau_analytical = (1 + rho) / (1 - rho)  # ~5.667
        ess_analytical = n / tau_analytical  # ~1765

        result = adv.calculate_spectral_density(chain, method="welch")
        ess_spectral = result.effective_sample_size

        # Welch spectral estimation has variance, so allow 50% tolerance.
        # With the old bug, ess_spectral would be ~4-6x too high.
        ratio = ess_spectral / ess_analytical
        assert 0.5 < ratio < 1.5, (
            f"Spectral ESS ({ess_spectral:.0f}) is too far from analytical "
            f"ESS ({ess_analytical:.0f}); ratio={ratio:.2f}. "
            f"If ratio >> 2.0, the old bugs (factor-of-2 or detrend) are present."
        )

    def test_ar1_tau_not_half_of_analytical(self, adv):
        """Directly check that estimated tau is NOT half the analytical value.

        This is the most direct regression test for issue #476.
        """
        n = 10_000
        rho = 0.5
        chain = _make_ar1_chain(n, rho, seed=456)

        tau_analytical = (1 + rho) / (1 - rho)  # 3.0

        result = adv.calculate_spectral_density(chain, method="welch")
        tau_spectral = result.integrated_autocorr_time

        # tau_spectral should be in the right ballpark of tau_analytical (~3.0).
        # With the old bug, tau_spectral was ~0.36 (ratio ~0.12).
        ratio = tau_spectral / tau_analytical
        assert ratio > 0.5, (
            f"tau_spectral ({tau_spectral:.2f}) is less than 50% of "
            f"tau_analytical ({tau_analytical:.2f}); ratio={ratio:.2f}. "
            f"If ratio < 0.3, the old bugs are likely still present."
        )
        assert ratio < 2.0, (
            f"tau_spectral ({tau_spectral:.2f}) is more than 200% of "
            f"tau_analytical ({tau_analytical:.2f}); ratio={ratio:.2f}."
        )

    def test_white_noise_tau_near_one(self, adv):
        """For iid samples, tau should be close to 1.0."""
        rng = np.random.default_rng(789)
        chain = rng.standard_normal(5000)

        result = adv.calculate_spectral_density(chain, method="welch")
        tau = result.integrated_autocorr_time

        # For white noise, tau ~ 1.0. With old bug, tau ~ 0.5.
        assert 0.5 < tau < 2.0, (
            f"White noise tau should be ~1.0, got {tau:.2f}. "
            f"If tau ~0.5, the factor-of-2 bug is still present."
        )

    def test_higher_autocorrelation_gives_higher_tau(self, adv):
        """Chains with stronger autocorrelation should yield higher tau."""
        n = 5000
        chain_low = _make_ar1_chain(n, rho=0.3, seed=100)
        chain_high = _make_ar1_chain(n, rho=0.7, seed=100)

        tau_low = adv.calculate_spectral_density(chain_low, method="welch").integrated_autocorr_time
        tau_high = adv.calculate_spectral_density(
            chain_high, method="welch"
        ).integrated_autocorr_time

        assert tau_high > tau_low, (
            f"Higher rho should give higher tau, but got "
            f"tau(rho=0.3)={tau_low:.2f} >= tau(rho=0.7)={tau_high:.2f}"
        )


class TestHeidelbergerWelchWithCorrectedTau:
    """Verify Heidelberger-Welch still works correctly after the tau fix."""

    def test_stationary_chain_detected(self, adv):
        """A well-converged chain should still be detected as stationary."""
        rng = np.random.default_rng(42)
        chain = rng.standard_normal(3000)

        result = adv.heidelberger_welch_advanced(chain)
        assert result["stationary"] is True

    def test_nonstationary_chain_detected(self, adv):
        """A chain with a trend should be detected as non-stationary."""
        rng = np.random.default_rng(42)
        n = 2000
        # Strong linear trend dominates the signal
        chain = np.linspace(0, 10, n) + rng.standard_normal(n) * 0.1

        result = adv.heidelberger_welch_advanced(chain)
        assert result["stationary"] is False

    def test_heidelberger_tau_uses_corrected_formula(self, adv):
        """The tau reported by Heidelberger-Welch should reflect the fix."""
        n = 5000
        rho = 0.5
        chain = _make_ar1_chain(n, rho, seed=999)

        result = adv.heidelberger_welch_advanced(chain)
        if result["stationary"]:
            tau = result["integrated_autocorr_time"]
            tau_analytical = (1 + rho) / (1 - rho)  # 3.0
            ratio = tau / tau_analytical
            # Should NOT be ~0.12 (old bug with factor-of-2 + detrend)
            assert ratio > 0.4, (
                f"H-W tau ({tau:.2f}) looks like old halved value; "
                f"ratio to analytical ({tau_analytical:.2f}) = {ratio:.2f}"
            )

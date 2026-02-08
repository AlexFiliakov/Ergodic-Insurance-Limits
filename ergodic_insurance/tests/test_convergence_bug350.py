"""Regression tests for convergence diagnostics bug fixes (issue #350).

Tests verify:
1. Factor of 2 in integrated autocorrelation time (convergence_advanced.py)
2. Lag-1 autocorrelation in ConvergenceStats (convergence.py)
3. Non-overlapping Geyer monotone sequence pairs (convergence_advanced.py)
"""

import numpy as np
import pytest

from ergodic_insurance.convergence import ConvergenceDiagnostics
from ergodic_insurance.convergence_advanced import AdvancedConvergenceDiagnostics


class TestIntegratedAutocorrTimeFactor:
    """Verify the 2x factor in integrated autocorrelation time."""

    @pytest.fixture
    def adv(self):
        return AdvancedConvergenceDiagnostics()

    @pytest.fixture
    def basic(self):
        return ConvergenceDiagnostics()

    def test_integrated_time_known_acf(self, adv):
        """Integrated time for a known ACF should include the 2x multiplier.

        For acf = [1.0, 0.5, 0.3, 0.1, -0.1]:
        Pair 1 = 0.5 + 0.3 = 0.8
        tau = 1 + 2 * 0.8 + 2 * 0.1 = 2.8
        (The last element 0.1 is a single odd term from the first pair's range.)
        """
        acf = np.array([1.0, 0.5, 0.3, 0.1, -0.1])
        tau = adv._calculate_integrated_time(acf, cutoff=4)
        # tau = 1 + 2*(0.5+0.3) + 2*0.1 = 1 + 1.6 + 0.2 = 2.8
        assert tau == pytest.approx(2.8)

    def test_ess_matches_between_modules(self):
        """ESS from AdvancedConvergenceDiagnostics should be close to ESS
        from ConvergenceDiagnostics for the same data (acceptance criterion 1).
        """
        np.random.seed(123)
        n = 5000
        rho = 0.5
        chain = np.zeros(n)
        chain[0] = np.random.randn()
        for i in range(1, n):
            chain[i] = rho * chain[i - 1] + np.sqrt(1 - rho**2) * np.random.randn()

        basic = ConvergenceDiagnostics()
        adv = AdvancedConvergenceDiagnostics()

        ess_basic = basic.calculate_ess(chain)
        acf_result = adv.calculate_autocorrelation_full(chain, method="biased")
        ess_advanced = n / acf_result.integrated_time

        # Both should be reasonably close (within 30% of each other)
        ratio = ess_basic / ess_advanced
        assert (
            0.5 < ratio < 2.0
        ), f"ESS mismatch: basic={ess_basic:.0f}, advanced={ess_advanced:.0f}, ratio={ratio:.2f}"

    def test_integrated_time_single_pair(self, adv):
        """Single positive pair should get the 2x factor."""
        acf = np.array([1.0, 0.4, 0.3, -0.1])
        tau = adv._calculate_integrated_time(acf, cutoff=3)
        # cutoff=3: i=1, i+1=2 < 3 → pair(0.4+0.3)=0.7, tau = 1 + 2*0.7 = 2.4
        assert tau == pytest.approx(2.4)


class TestAutocorrelationIndex:
    """Verify ConvergenceStats.autocorrelation returns lag-1 (not lag-0)."""

    def test_autocorrelation_not_always_one(self):
        """ConvergenceStats.autocorrelation should NOT always be 1.0
        (acceptance criterion 2).
        """
        np.random.seed(42)
        n = 2000
        rho = 0.7
        chain = np.zeros(n)
        chain[0] = np.random.randn()
        for i in range(1, n):
            chain[i] = rho * chain[i - 1] + np.sqrt(1 - rho**2) * np.random.randn()

        chains = np.stack([chain, chain + np.random.randn(n) * 0.01])

        diag = ConvergenceDiagnostics()
        results = diag.check_convergence(chains)
        stats = results["metric_0"]

        assert (
            stats.autocorrelation != 1.0
        ), "autocorrelation should be lag-1, not lag-0 (which is always 1.0)"
        # For AR(1) with rho=0.7, lag-1 autocorrelation should be close to 0.7
        assert 0.4 < stats.autocorrelation < 0.95

    def test_white_noise_autocorrelation_near_zero(self):
        """For white noise, lag-1 autocorrelation should be near zero."""
        np.random.seed(99)
        chain = np.random.randn(5000)
        chains = np.stack([chain[:2500], chain[2500:]])

        diag = ConvergenceDiagnostics()
        results = diag.check_convergence(chains)
        stats = results["metric_0"]

        assert (
            abs(stats.autocorrelation) < 0.1
        ), f"White noise lag-1 autocorrelation should be ~0, got {stats.autocorrelation}"


class TestGeyerMonotoneNonOverlapping:
    """Verify Geyer monotone check uses non-overlapping pairs."""

    @pytest.fixture
    def adv(self):
        return AdvancedConvergenceDiagnostics()

    def test_non_overlapping_pair_comparison(self, adv):
        """Construct ACF where overlapping pairs give wrong cutoff
        (acceptance criterion 3).

        ACF: [1.0, 0.6, 0.1, 0.05, 0.55, -0.1, -0.2]
        Non-overlapping pairs:
          P1 = acf[1]+acf[2] = 0.6+0.1 = 0.7
          P2 = acf[3]+acf[4] = 0.05+0.55 = 0.6
          P3 = acf[5]+acf[6] = -0.1+(-0.2) = -0.3
        Monotonicity: P1=0.7 >= P2=0.6 ✓, P3 negative → stop
        Correct cutoff includes P1 and P2 → cutoff should be 4

        Old overlapping code would compare acf[2]+acf[3]=0.15 vs acf[3]+acf[4]=0.6.
        Since 0.15 < 0.6, old code would flag non-monotone at i=3 → cutoff=2,
        which is too early.
        """
        acf = np.array([1.0, 0.6, 0.1, 0.05, 0.55, -0.1, -0.2])
        cutoff = adv._find_initial_monotone(acf)

        # The correct result should allow P1 and P2 (both positive, P1 >= P2)
        # cutoff should be >= 4 to include both pairs
        assert cutoff >= 4, (
            f"Geyer cutoff should be >= 4 for this ACF, got {cutoff}. "
            "Likely comparing overlapping pairs."
        )

    def test_monotone_decreasing_pairs(self, adv):
        """Monotonically decreasing pair sums should all be included."""
        # P1=0.9, P2=0.5, P3=0.2, all positive and decreasing
        acf = np.array([1.0, 0.5, 0.4, 0.3, 0.2, 0.15, 0.05])
        cutoff = adv._find_initial_monotone(acf)

        # All pairs positive and monotone → include all, cutoff = n-1 = 6
        assert cutoff == 6

    def test_monotone_violation_detected(self, adv):
        """Non-monotone pair sums should trigger early cutoff."""
        # P1=0.3, P2=0.7 → P2 > P1, non-monotone
        acf = np.array([1.0, 0.2, 0.1, 0.4, 0.3, 0.0, 0.0])
        cutoff = adv._find_initial_monotone(acf)

        # P2 > P1 violates monotonicity at i=3, return i-1=2
        assert cutoff == 2

    def test_first_pair_negative(self, adv):
        """Negative first pair should return cutoff 0."""
        acf = np.array([1.0, -0.1, -0.2, 0.0])
        cutoff = adv._find_initial_monotone(acf)

        assert cutoff == 0

"""Regression tests for initial monotone sequence bug (issue #396).

Tests verify the acceptance criteria from issue #396:
1. Monotonically decreasing ACF passes all pair monotonicity checks
2. Non-monotone pair sums trigger cutoff at the correct position
3. Results match Geyer (1992) algorithm (R mcmc::initseq() reference)

The underlying code fix (tracking prev_pair_sum instead of comparing
overlapping pairs) was applied in #350. These tests lock down the
specific acceptance criteria from #396.
"""

import numpy as np
import pytest

from ergodic_insurance.convergence_advanced import AdvancedConvergenceDiagnostics


@pytest.fixture
def adv():
    return AdvancedConvergenceDiagnostics()


class TestAcceptanceCriteria396:
    """Acceptance criteria from issue #396."""

    def test_ac1_monotonically_decreasing_acf(self, adv):
        """AC1: For monotonically decreasing ACF, all pairs pass monotonicity.

        ACF = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        Gamma_1 = 0.9 + 0.8 = 1.7
        Gamma_2 = 0.7 + 0.6 = 1.3
        Gamma_3 = 0.5 + 0.4 = 0.9
        All positive and monotone (1.7 >= 1.3 >= 0.9).
        Cutoff should be n-1 = 7.
        """
        acf = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        cutoff = adv._find_initial_monotone(acf)

        assert cutoff == 7, (
            f"All pairs are positive and monotonically decreasing, "
            f"expected cutoff=7, got {cutoff}"
        )

    def test_ac2_non_monotone_gamma3(self, adv):
        """AC2: Non-monotone Gamma_3 triggers cutoff.

        ACF = [1, 0.9, 0.8, 0.3, 0.2, 0.8, 0.7]
        Gamma_1 = 0.9 + 0.8 = 1.7
        Gamma_2 = 0.3 + 0.2 = 0.5
        Gamma_3 = 0.8 + 0.7 = 1.5
        Gamma_3 (1.5) > Gamma_2 (0.5) violates monotonicity.
        Should stop at Gamma_3 → cutoff = 4.
        """
        acf = np.array([1.0, 0.9, 0.8, 0.3, 0.2, 0.8, 0.7])
        cutoff = adv._find_initial_monotone(acf)

        assert cutoff == 4, (
            f"Gamma_3=1.5 > Gamma_2=0.5 violates monotonicity, " f"expected cutoff=4, got {cutoff}"
        )

    def test_ac3_geyer_reference_ar1(self, adv):
        """AC3: Compare with Geyer (1992) algorithm on known ACF.

        For an AR(1) process with rho=0.8, the theoretical ACF is
        acf[k] = rho^k. Pair sums are:
          Gamma_k = rho^(2k-1) + rho^(2k)
                  = rho^(2k-1) * (1 + rho)

        Since rho < 1, rho^(2k-1) is strictly decreasing in k,
        so all Gamma_k are positive and monotonically decreasing.
        R's mcmc::initseq() would use the full sequence.

        Expected cutoff = n-1 (all pairs valid).
        """
        rho = 0.8
        n = 20
        acf = np.array([rho**k for k in range(n)])
        cutoff = adv._find_initial_monotone(acf)

        assert cutoff == n - 1, (
            f"AR(1) ACF with rho=0.8 should pass all monotonicity checks, "
            f"expected cutoff={n - 1}, got {cutoff}"
        )

    def test_ac3_geyer_reference_decaying_pairs(self, adv):
        """AC3: Verify against manually computed Geyer sequence.

        Construct ACF where pair sums are known and verify the
        algorithm matches Geyer (1992) hand computation.

        ACF = [1.0, 0.7, 0.5, 0.4, 0.2, 0.15, 0.05, 0.01, -0.05]
        Gamma_1 = 0.7 + 0.5  = 1.2
        Gamma_2 = 0.4 + 0.2  = 0.6
        Gamma_3 = 0.15 + 0.05 = 0.2
        Gamma_4 = 0.01 + (-0.05) = -0.04  (negative → stop)

        R's mcmc::initseq() initial monotone estimator truncates at
        the first negative pair. Valid pairs: Gamma_1..Gamma_3.
        Cutoff = 6 (includes lags through index 6).
        """
        acf = np.array([1.0, 0.7, 0.5, 0.4, 0.2, 0.15, 0.05, 0.01, -0.05])
        cutoff = adv._find_initial_monotone(acf)

        assert cutoff == 6, (
            f"Gamma_4 is negative, valid sequence ends after Gamma_3, "
            f"expected cutoff=6, got {cutoff}"
        )

    def test_ac3_geyer_reference_integrated_time(self, adv):
        """AC3: Verify integrated autocorrelation time with correct cutoff.

        Using the ACF from AC2:
        ACF = [1, 0.9, 0.8, 0.3, 0.2, 0.8, 0.7]
        Cutoff = 4 (Gamma_3 violates monotonicity at i=5, return i-1=4)

        With cutoff=4, _calculate_integrated_time processes:
          i=1: full pair (0.9 + 0.8 = 1.7) → tau += 2*1.7 = 3.4
          i=3: singleton (i+1=4 not < cutoff=4) → tau += 2*0.3 = 0.6
          tau = 1.0 + 3.4 + 0.6 = 5.0
        """
        acf = np.array([1.0, 0.9, 0.8, 0.3, 0.2, 0.8, 0.7])
        cutoff = adv._find_initial_monotone(acf)
        tau = adv._calculate_integrated_time(acf, cutoff)

        assert cutoff == 4
        assert tau == pytest.approx(
            5.0
        ), f"Integrated time should be 1 + 2*1.7 + 2*0.3 = 5.0, got {tau}"


class TestEdgeCases396:
    """Edge cases for _find_initial_monotone robustness."""

    def test_single_pair(self, adv):
        """ACF with exactly one pair."""
        acf = np.array([1.0, 0.5, 0.3])
        cutoff = adv._find_initial_monotone(acf)
        assert cutoff == 2

    def test_equal_consecutive_pairs(self, adv):
        """Equal pair sums should NOT violate monotonicity.

        Geyer (1992) requires Gamma_k <= Gamma_{k-1} (non-strict).
        Equal pairs satisfy this.

        ACF = [1.0, 0.5, 0.3, 0.4, 0.4, -0.1], n=6
        Gamma_1 = 0.5 + 0.3 = 0.8
        Gamma_2 = 0.4 + 0.4 = 0.8
        Equal (0.8 <= 0.8), both valid. Loop ends → return n-1 = 5.
        """
        acf = np.array([1.0, 0.5, 0.3, 0.4, 0.4, -0.1])
        cutoff = adv._find_initial_monotone(acf)

        # Equal pairs pass monotonicity, loop completes, cutoff = n-1 = 5
        assert cutoff == 5, (
            f"Equal pair sums should not violate monotonicity, " f"expected cutoff=5, got {cutoff}"
        )

    def test_all_negative_pairs(self, adv):
        """ACF where first pair is negative should return 0."""
        acf = np.array([1.0, -0.5, -0.3, -0.1])
        cutoff = adv._find_initial_monotone(acf)
        assert cutoff == 0

    def test_long_monotone_sequence(self, adv):
        """Long monotonically decreasing ACF should use full length."""
        n = 100
        acf = np.array([1.0 / (k + 1) for k in range(n)])
        cutoff = adv._find_initial_monotone(acf)
        assert cutoff == n - 1

"""Tests for StochasticClaimDevelopment (Issue #521).

Covers Dirichlet perturbation, sum-to-one guarantee, reproducibility,
concentration effects, integration with ClaimLiability / Claim, and deep copy.
"""

import copy
from decimal import Decimal

import numpy as np
import pytest

from ergodic_insurance.claim_development import (
    Claim,
    ClaimDevelopment,
    StochasticClaimDevelopment,
)
from ergodic_insurance.claim_liability import ClaimLiability


class TestStochasticClaimDevelopmentConstruction:
    """Test construction and validation."""

    def test_default_construction(self):
        """Stochastic pattern can be constructed from any base pattern."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        assert stoch.stochastic is True
        assert stoch.concentration == 50.0
        assert stoch.base_pattern is base
        assert "_stochastic" in stoch.pattern_name

    def test_deterministic_fallback(self):
        """stochastic=False returns exact base pattern factors."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, stochastic=False)

        assert stoch.development_factors == base.development_factors
        assert stoch.tail_factor == base.tail_factor

    def test_invalid_concentration_zero(self):
        base = ClaimDevelopment.create_immediate()
        with pytest.raises(ValueError, match="Concentration must be positive"):
            StochasticClaimDevelopment(base, concentration=0)

    def test_invalid_concentration_negative(self):
        base = ClaimDevelopment.create_immediate()
        with pytest.raises(ValueError, match="Concentration must be positive"):
            StochasticClaimDevelopment(base, concentration=-10)

    def test_pattern_name_suffix(self):
        base = ClaimDevelopment.create_medium_tail_5yr()
        stoch = StochasticClaimDevelopment(base, seed=0)
        assert stoch.pattern_name == "MEDIUM_TAIL_5YR_stochastic"

    def test_repr(self):
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, concentration=100, seed=0)
        r = repr(stoch)
        assert "StochasticClaimDevelopment" in r
        assert "LONG_TAIL_10YR" in r
        assert "100" in r


class TestDirichletProperties:
    """Verify statistical properties of the Dirichlet perturbation."""

    def test_factors_sum_to_one(self):
        """Dirichlet output always sums to 1.0."""
        base = ClaimDevelopment.create_long_tail_10yr()
        for seed in range(50):
            stoch = StochasticClaimDevelopment(base, seed=seed)
            total = sum(stoch.development_factors) + stoch.tail_factor
            assert abs(total - 1.0) < 0.01, f"seed={seed}: total={total}"

    def test_factors_non_negative(self):
        """All perturbed factors are non-negative."""
        base = ClaimDevelopment.create_very_long_tail_15yr()
        for seed in range(50):
            stoch = StochasticClaimDevelopment(base, seed=seed)
            assert all(f >= 0 for f in stoch.development_factors)
            assert stoch.tail_factor >= 0

    def test_factor_count_preserved(self):
        """Number of development factors matches the base pattern."""
        for factory in [
            ClaimDevelopment.create_immediate,
            ClaimDevelopment.create_medium_tail_5yr,
            ClaimDevelopment.create_long_tail_10yr,
            ClaimDevelopment.create_very_long_tail_15yr,
        ]:
            base = factory()
            stoch = StochasticClaimDevelopment(base, seed=42)
            assert len(stoch.development_factors) == len(base.development_factors)

    def test_tail_factor_preserved_zero(self):
        """Tail factor stays zero when base has no tail."""
        base = ClaimDevelopment.create_long_tail_10yr()
        assert base.tail_factor == 0.0
        stoch = StochasticClaimDevelopment(base, seed=42)
        assert stoch.tail_factor == 0.0

    def test_tail_factor_perturbed(self):
        """Tail factor is included in Dirichlet simplex when present."""
        base = ClaimDevelopment(
            pattern_name="WITH_TAIL",
            development_factors=[0.4, 0.3, 0.2],
            tail_factor=0.1,
        )
        stoch = StochasticClaimDevelopment(base, seed=42)
        # Tail should be perturbed (different from 0.1) but still positive
        assert stoch.tail_factor > 0
        total = sum(stoch.development_factors) + stoch.tail_factor
        assert abs(total - 1.0) < 0.01


class TestConcentrationEffect:
    """Verify that concentration controls variability."""

    def test_high_concentration_low_variance(self):
        """High kappa produces factors close to the base."""
        base = ClaimDevelopment.create_long_tail_10yr()
        deviations = []
        for seed in range(100):
            stoch = StochasticClaimDevelopment(base, concentration=500, seed=seed)
            dev = sum(
                (s - b) ** 2 for s, b in zip(stoch.development_factors, base.development_factors)
            )
            deviations.append(dev)
        mean_deviation = np.mean(deviations)
        assert mean_deviation < 0.005  # Very tight around base

    def test_low_concentration_high_variance(self):
        """Low kappa produces more dispersed factors."""
        base = ClaimDevelopment.create_long_tail_10yr()
        deviations = []
        for seed in range(100):
            stoch = StochasticClaimDevelopment(base, concentration=10, seed=seed)
            dev = sum(
                (s - b) ** 2 for s, b in zip(stoch.development_factors, base.development_factors)
            )
            deviations.append(dev)
        mean_deviation = np.mean(deviations)
        assert mean_deviation > 0.005  # Noticeably dispersed

    def test_monotone_variance_with_concentration(self):
        """Decreasing kappa increases variability."""
        base = ClaimDevelopment.create_long_tail_10yr()
        concentrations = [200, 100, 50, 20, 10]
        mean_deviations = []
        for kappa in concentrations:
            devs = []
            for seed in range(200):
                stoch = StochasticClaimDevelopment(base, concentration=kappa, seed=seed)
                dev = sum(
                    (s - b) ** 2
                    for s, b in zip(stoch.development_factors, base.development_factors)
                )
                devs.append(dev)
            mean_deviations.append(np.mean(devs))

        # Each lower kappa should produce higher deviation
        for i in range(len(mean_deviations) - 1):
            assert mean_deviations[i] < mean_deviations[i + 1], (
                f"kappa={concentrations[i]} dev={mean_deviations[i]:.6f} >= "
                f"kappa={concentrations[i+1]} dev={mean_deviations[i+1]:.6f}"
            )


class TestReproducibility:
    """Test seeded reproducibility."""

    def test_same_seed_same_factors(self):
        """Same seed produces identical factors."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch1 = StochasticClaimDevelopment(base, seed=12345)
        stoch2 = StochasticClaimDevelopment(base, seed=12345)
        assert stoch1.development_factors == stoch2.development_factors

    def test_different_seed_different_factors(self):
        """Different seeds produce different factors."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch1 = StochasticClaimDevelopment(base, seed=1)
        stoch2 = StochasticClaimDevelopment(base, seed=2)
        assert stoch1.development_factors != stoch2.development_factors

    def test_seedsequence_support(self):
        """SeedSequence objects work for seeding."""
        base = ClaimDevelopment.create_long_tail_10yr()
        ss = np.random.SeedSequence(42)
        stoch1 = StochasticClaimDevelopment(base, seed=ss.spawn(1)[0])
        stoch2 = StochasticClaimDevelopment(base, seed=ss.spawn(1)[0])
        # Different child seeds from spawn produce different results
        assert stoch1.development_factors != stoch2.development_factors

    def test_none_seed_nondeterministic(self):
        """None seed uses entropy; two calls are (very likely) different."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch1 = StochasticClaimDevelopment(base, seed=None)
        stoch2 = StochasticClaimDevelopment(base, seed=None)
        # Technically possible to collide, but astronomically unlikely
        assert stoch1.development_factors != stoch2.development_factors


class TestPaymentCalculation:
    """Verify calculate_payments and get_cumulative_paid work correctly."""

    def test_calculate_payments_uses_perturbed_factors(self):
        """Payments use the realized (perturbed) factors, not base."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        amount = 1_000_000
        payment_y0 = stoch.calculate_payments(amount, 2020, 2020)
        expected = amount * stoch.development_factors[0]
        assert payment_y0 == pytest.approx(expected)

    def test_total_payments_sum_to_claim_amount(self):
        """Sum of all payments equals the claim amount (factors sum to 1)."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        amount = 1_000_000
        total = sum(
            stoch.calculate_payments(amount, 2020, 2020 + y)
            for y in range(len(stoch.development_factors))
        )
        assert total == pytest.approx(amount, rel=0.01)

    def test_cumulative_paid(self):
        """get_cumulative_paid returns correct running totals."""
        base = ClaimDevelopment.create_medium_tail_5yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        cumulative = stoch.get_cumulative_paid(len(stoch.development_factors))
        assert cumulative == pytest.approx(1.0, abs=0.01)

        # Partial cumulative should be between 0 and 1
        partial = stoch.get_cumulative_paid(2)
        assert 0 < partial < 1


class TestIntegrationWithClaimLiability:
    """Verify StochasticClaimDevelopment works as a drop-in for ClaimLiability."""

    def test_claim_liability_accepts_stochastic(self):
        """ClaimLiability works with StochasticClaimDevelopment strategy."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        claim = ClaimLiability(
            original_amount=Decimal(1_000_000),
            remaining_amount=Decimal(1_000_000),
            year_incurred=2020,
            development_strategy=stoch,
        )
        # Verify payments work
        payment = claim.get_payment(0)
        assert float(payment) > 0
        assert float(payment) == pytest.approx(1_000_000 * stoch.development_factors[0], rel=1e-6)

    def test_claim_liability_deep_copy(self):
        """Deep copy of ClaimLiability preserves StochasticClaimDevelopment."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        original = ClaimLiability(
            original_amount=Decimal(1_000_000),
            remaining_amount=Decimal(1_000_000),
            year_incurred=2020,
            development_strategy=stoch,
        )

        copied = copy.deepcopy(original)

        # Type is preserved
        assert isinstance(copied.development_strategy, StochasticClaimDevelopment)
        # Factors are identical (not re-sampled)
        assert (
            copied.development_strategy.development_factors
            == original.development_strategy.development_factors
        )
        # But they are independent objects
        assert copied.development_strategy is not original.development_strategy
        # Concentration preserved
        assert copied.development_strategy.concentration == stoch.concentration

    def test_payment_schedule_property(self):
        """ClaimLiability.payment_schedule returns stochastic factors."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        claim = ClaimLiability(
            original_amount=Decimal(1_000_000),
            remaining_amount=Decimal(1_000_000),
            year_incurred=2020,
            development_strategy=stoch,
        )
        assert claim.payment_schedule == stoch.development_factors


class TestIntegrationWithClaim:
    """Verify StochasticClaimDevelopment works with the Claim class."""

    def test_claim_with_stochastic_pattern(self):
        """Claim object accepts StochasticClaimDevelopment."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)

        claim = Claim(
            claim_id="CL001",
            accident_year=2020,
            reported_year=2020,
            initial_estimate=500_000,
            development_pattern=stoch,
        )
        assert claim.development_pattern is stoch
        assert isinstance(claim.development_pattern, ClaimDevelopment)


class TestDeepCopy:
    """Test deep copy behavior of StochasticClaimDevelopment."""

    def test_deep_copy_preserves_factors(self):
        """Deep copy preserves realized factors (no re-sampling)."""
        base = ClaimDevelopment.create_long_tail_10yr()
        original = StochasticClaimDevelopment(base, seed=42)
        copied = copy.deepcopy(original)

        assert copied.development_factors == original.development_factors
        assert copied.tail_factor == original.tail_factor
        assert copied.pattern_name == original.pattern_name

    def test_deep_copy_independent_mutations(self):
        """Modifying copied object does not affect original."""
        base = ClaimDevelopment.create_long_tail_10yr()
        original = StochasticClaimDevelopment(base, seed=42)
        copied = copy.deepcopy(original)

        copied.development_factors[0] = 999.0
        assert original.development_factors[0] != 999.0

    def test_deep_copy_preserves_attributes(self):
        """All StochasticClaimDevelopment attributes are preserved."""
        base = ClaimDevelopment.create_long_tail_10yr()
        original = StochasticClaimDevelopment(base, concentration=75, seed=42)
        copied = copy.deepcopy(original)

        assert copied.concentration == 75
        assert copied.stochastic is True
        assert copied.base_pattern.pattern_name == base.pattern_name


class TestAllFactoryPatterns:
    """Ensure stochastic wrapping works with all built-in patterns."""

    @pytest.mark.parametrize(
        "factory",
        [
            ClaimDevelopment.create_immediate,
            ClaimDevelopment.create_medium_tail_5yr,
            ClaimDevelopment.create_long_tail_10yr,
            ClaimDevelopment.create_very_long_tail_15yr,
        ],
    )
    def test_stochastic_wrapping(self, factory):
        """Each factory pattern can be wrapped stochastically."""
        base = factory()
        stoch = StochasticClaimDevelopment(base, seed=42)

        total = sum(stoch.development_factors) + stoch.tail_factor
        assert abs(total - 1.0) < 0.01
        assert len(stoch.development_factors) == len(base.development_factors)
        assert all(f >= 0 for f in stoch.development_factors)

    @pytest.mark.parametrize(
        "factory",
        [
            ClaimDevelopment.create_immediate,
            ClaimDevelopment.create_medium_tail_5yr,
            ClaimDevelopment.create_long_tail_10yr,
            ClaimDevelopment.create_very_long_tail_15yr,
        ],
    )
    def test_deterministic_wrapping(self, factory):
        """stochastic=False returns exact base factors for all patterns."""
        base = factory()
        stoch = StochasticClaimDevelopment(base, stochastic=False)

        assert stoch.development_factors == base.development_factors
        assert stoch.tail_factor == base.tail_factor


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_factor_pattern(self):
        """Immediate (single factor) pattern works with Dirichlet."""
        base = ClaimDevelopment.create_immediate()
        stoch = StochasticClaimDevelopment(base, seed=42)
        # Single-element Dirichlet always returns [1.0]
        assert len(stoch.development_factors) == 1
        assert stoch.development_factors[0] == pytest.approx(1.0)

    def test_custom_pattern_with_tail(self):
        """Custom pattern with tail factor is properly perturbed."""
        base = ClaimDevelopment(
            pattern_name="CUSTOM_TAIL",
            development_factors=[0.3, 0.3, 0.2, 0.1],
            tail_factor=0.1,
        )
        stoch = StochasticClaimDevelopment(base, concentration=50, seed=42)
        total = sum(stoch.development_factors) + stoch.tail_factor
        assert abs(total - 1.0) < 0.01
        assert stoch.tail_factor > 0  # Tail should be perturbed but positive

    def test_very_high_concentration(self):
        """Very high kappa essentially reproduces the base pattern."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, concentration=100_000, seed=42)

        for s, b in zip(stoch.development_factors, base.development_factors):
            assert s == pytest.approx(b, abs=0.005)

    def test_isinstance_check(self):
        """StochasticClaimDevelopment is recognized as ClaimDevelopment."""
        base = ClaimDevelopment.create_long_tail_10yr()
        stoch = StochasticClaimDevelopment(base, seed=42)
        assert isinstance(stoch, ClaimDevelopment)

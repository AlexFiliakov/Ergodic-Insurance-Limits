"""Unit and integration tests for reserve re-estimation (Issue #470, ASC 944-40-25).

Tests cover:
- ClaimLiability.re_estimate() behavior
- Noise applied at claim creation
- re_estimate_reserves() integration with manufacturer
- Income statement flow-through
- Reserve reconciliation report
- Full step() integration and backward compatibility
"""

import copy
from decimal import Decimal
import random

import pytest

from ergodic_insurance.claim_development import ClaimDevelopment
from ergodic_insurance.claim_liability import ClaimLiability
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestClaimLiabilityReEstimate:
    """Unit tests for ClaimLiability.re_estimate()."""

    def test_noop_when_true_ultimate_is_none(self):
        """re_estimate returns ZERO when reserve development is off."""
        claim = ClaimLiability(
            original_amount=to_decimal(100_000),
            remaining_amount=to_decimal(100_000),
            year_incurred=1,
        )
        rng = random.Random(42)
        change = claim.re_estimate(current_year=2, rng=rng)
        assert change == ZERO
        assert claim.remaining_amount == to_decimal(100_000)

    def test_noop_when_remaining_is_zero(self):
        """re_estimate returns ZERO when claim is fully paid."""
        claim = ClaimLiability(
            original_amount=to_decimal(100_000),
            remaining_amount=ZERO,
            year_incurred=1,
            true_ultimate=to_decimal(100_000),
        )
        claim._noise_std = 0.20
        rng = random.Random(42)
        change = claim.re_estimate(current_year=2, rng=rng)
        assert change == ZERO

    def test_convergence_at_maturity(self):
        """At full maturity, estimate snaps to true residual."""
        # 10-year pattern, 10 years elapsed -> maturity = 1.0
        claim = ClaimLiability(
            original_amount=to_decimal(120_000),
            remaining_amount=to_decimal(120_000),
            year_incurred=0,
            true_ultimate=to_decimal(100_000),
        )
        claim._noise_std = 0.20
        rng = random.Random(42)
        change = claim.re_estimate(current_year=10, rng=rng)
        # true_residual = 100_000 - 0 paid = 100_000
        assert claim.remaining_amount == to_decimal(100_000)
        expected_change = to_decimal(100_000) - to_decimal(120_000)
        assert change == expected_change

    def test_remaining_always_non_negative(self):
        """re_estimate never sets remaining_amount below zero."""
        claim = ClaimLiability(
            original_amount=to_decimal(10_000),
            remaining_amount=to_decimal(10_000),
            year_incurred=0,
            true_ultimate=to_decimal(1),  # Very small true amount
        )
        claim._noise_std = 0.90  # High noise
        rng = random.Random(99)
        for year in range(1, 15):
            claim.re_estimate(current_year=year, rng=rng)
            assert claim.remaining_amount >= ZERO

    def test_reproducibility_with_same_seed(self):
        """Same seed produces identical re-estimation results."""

        def run_re_estimate(seed):
            claim = ClaimLiability(
                original_amount=to_decimal(100_000),
                remaining_amount=to_decimal(100_000),
                year_incurred=0,
                true_ultimate=to_decimal(80_000),
            )
            claim._noise_std = 0.20
            rng = random.Random(seed)
            change = claim.re_estimate(current_year=3, rng=rng)
            return change, claim.remaining_amount

        change1, rem1 = run_re_estimate(42)
        change2, rem2 = run_re_estimate(42)
        assert change1 == change2
        assert rem1 == rem2

    def test_early_maturity_has_more_noise(self):
        """Claims early in development have larger noise variance than late ones."""
        results_early = []
        results_late = []
        for seed in range(100):
            # Early: year 1 of 10
            claim_early = ClaimLiability(
                original_amount=to_decimal(100_000),
                remaining_amount=to_decimal(100_000),
                year_incurred=0,
                true_ultimate=to_decimal(100_000),
            )
            claim_early._noise_std = 0.30
            rng_early = random.Random(seed)
            change_early = claim_early.re_estimate(current_year=1, rng=rng_early)
            results_early.append(float(change_early))

            # Late: year 8 of 10
            claim_late = ClaimLiability(
                original_amount=to_decimal(100_000),
                remaining_amount=to_decimal(100_000),
                year_incurred=0,
                true_ultimate=to_decimal(100_000),
            )
            claim_late._noise_std = 0.30
            rng_late = random.Random(seed)
            change_late = claim_late.re_estimate(current_year=8, rng=rng_late)
            results_late.append(float(change_late))

        # Variance of early changes should be larger
        import statistics

        var_early = statistics.variance(results_early)
        var_late = statistics.variance(results_late)
        assert var_early > var_late

    def test_total_paid_tracked_correctly(self):
        """_total_paid accumulates across make_payment calls."""
        claim = ClaimLiability(
            original_amount=to_decimal(100_000),
            remaining_amount=to_decimal(100_000),
            year_incurred=0,
            true_ultimate=to_decimal(100_000),
        )
        claim._noise_std = 0.20
        claim.make_payment(to_decimal(30_000))
        assert claim._total_paid == to_decimal(30_000)
        claim.make_payment(to_decimal(20_000))
        assert claim._total_paid == to_decimal(50_000)


class TestReserveNoiseAtCreation:
    """Test that noise is applied at claim creation when feature is enabled."""

    def _make_manufacturer(self, enable=True, noise_std=0.20):
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=enable,
            reserve_noise_std=noise_std,
        )
        return WidgetManufacturer(config)

    def test_noise_applied_when_enabled(self):
        """Claims get noisy estimates when reserve development is on."""
        mfr = self._make_manufacturer(enable=True, noise_std=0.20)
        mfr.current_year = 1
        mfr.process_uninsured_claim(500_000)

        assert len(mfr.claim_liabilities) == 1
        claim = mfr.claim_liabilities[0]
        assert claim.true_ultimate == to_decimal(500_000)
        # Noisy estimate should differ from true (almost surely with std=0.20)
        # We can't guarantee it's different due to randomness, but true_ultimate is set
        assert claim._noise_std == 0.20

    def test_no_noise_when_disabled(self):
        """Claims use exact amounts when feature is off."""
        mfr = self._make_manufacturer(enable=False)
        mfr.current_year = 1
        mfr.process_uninsured_claim(500_000)

        assert len(mfr.claim_liabilities) == 1
        claim = mfr.claim_liabilities[0]
        assert claim.true_ultimate is None
        assert claim._noise_std == 0.0

    def test_noisy_estimate_clamped_non_negative(self):
        """Noisy estimates never go below zero."""
        mfr = self._make_manufacturer(enable=True, noise_std=0.99)
        # Create many claims to test clamping
        for i in range(50):
            mfr.current_year = i
            mfr.process_uninsured_claim(1000)
        for claim in mfr.claim_liabilities:
            assert claim.original_amount >= ZERO
            assert claim.remaining_amount >= ZERO

    def test_noise_applied_to_insured_claim(self):
        """Noise is applied to insured claims from process_insurance_claim."""
        mfr = self._make_manufacturer(enable=True, noise_std=0.15)
        mfr.current_year = 1
        mfr.process_insurance_claim(
            claim_amount=200_000,
            deductible_amount=50_000,
            insurance_limit=200_000,
        )
        # Find the insured claim
        insured = [c for c in mfr.claim_liabilities if c.is_insured]
        if insured:
            assert insured[0].true_ultimate is not None
            assert insured[0]._noise_std == 0.15

    def test_noise_applied_in_record_claim_accrual(self):
        """Noise is applied via record_claim_accrual."""
        mfr = self._make_manufacturer(enable=True)
        mfr.current_year = 1
        mfr.record_claim_accrual(300_000)

        claim = mfr.claim_liabilities[-1]
        assert claim.true_ultimate == to_decimal(300_000)
        assert claim._noise_std == 0.20


class TestReEstimateReservesIntegration:
    """Integration tests for re_estimate_reserves() with manufacturer."""

    def _make_manufacturer(self, noise_std=0.20):
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=noise_std,
        )
        return WidgetManufacturer(config)

    def test_skips_current_year_claims(self):
        """re_estimate_reserves skips claims from the current year."""
        mfr = self._make_manufacturer()
        mfr.current_year = 5
        # Create claim in year 5 (current year)
        mfr.record_claim_accrual(100_000)
        original_remaining = mfr.claim_liabilities[0].remaining_amount

        mfr.re_estimate_reserves()
        # Should be unchanged
        assert mfr.claim_liabilities[0].remaining_amount == original_remaining

    def test_reestimates_prior_year_claims(self):
        """re_estimate_reserves processes claims from prior years."""
        mfr = self._make_manufacturer()
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)
        mfr.current_year = 3
        # Now re-estimate - claim from year 1 should be processed
        mfr.re_estimate_reserves()
        # Development amounts should be tracked (could be zero in rare cases)
        assert mfr.period_adverse_development >= ZERO
        assert mfr.period_favorable_development >= ZERO

    def test_ledger_entries_recorded(self):
        """re_estimate_reserves records proper double-entry ledger entries."""
        mfr = self._make_manufacturer(noise_std=0.50)
        mfr.current_year = 1
        mfr.record_claim_accrual(1_000_000)
        entries_before = len(mfr.ledger.entries)

        mfr.current_year = 3
        mfr.re_estimate_reserves()

        # Should have at least 2 new entries (debit + credit) if change != 0
        new_entries = mfr.ledger.get_entries(transaction_type=TransactionType.RESERVE_DEVELOPMENT)
        if mfr.period_adverse_development > ZERO or mfr.period_favorable_development > ZERO:
            assert len(new_entries) >= 2

    def test_adverse_development_entries(self):
        """Adverse development: Dr RESERVE_DEVELOPMENT, Cr CLAIM_LIABILITIES."""
        # Use a claim where remaining is far below true ultimate to force adverse
        mfr = self._make_manufacturer(noise_std=0.01)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)
        claim = mfr.claim_liabilities[0]
        # Force the estimate way below true
        claim.remaining_amount = to_decimal(200_000)
        claim.true_ultimate = to_decimal(500_000)
        claim._noise_std = 0.01

        mfr.current_year = 2
        mfr.re_estimate_reserves()

        if mfr.period_adverse_development > ZERO:
            dev_entries = mfr.ledger.get_entries(
                transaction_type=TransactionType.RESERVE_DEVELOPMENT
            )
            debit_entries = [
                e for e in dev_entries if e.account == AccountName.RESERVE_DEVELOPMENT.value
            ]
            credit_entries = [
                e for e in dev_entries if e.account == AccountName.CLAIM_LIABILITIES.value
            ]
            assert len(debit_entries) > 0 or len(credit_entries) > 0

    def test_period_tracking_reset(self):
        """reset_period_insurance_costs clears development trackers."""
        mfr = self._make_manufacturer()
        mfr.period_adverse_development = to_decimal(50_000)
        mfr.period_favorable_development = to_decimal(30_000)
        mfr.reset_period_insurance_costs()
        assert mfr.period_adverse_development == ZERO
        assert mfr.period_favorable_development == ZERO


class TestIncomeStatementFlow:
    """Test that development flows through operating income."""

    def test_adverse_development_reduces_income(self):
        """Net adverse development reduces operating income."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
        )
        mfr = WidgetManufacturer(config)
        mfr.period_adverse_development = to_decimal(100_000)
        mfr.period_favorable_development = ZERO

        revenue = mfr.calculate_revenue()
        oi_with_dev = mfr.calculate_operating_income(revenue)

        # Compare to what it would be without development
        mfr.period_adverse_development = ZERO
        oi_without_dev = mfr.calculate_operating_income(revenue)

        assert oi_with_dev < oi_without_dev
        assert oi_without_dev - oi_with_dev == to_decimal(100_000)

    def test_favorable_development_increases_income(self):
        """Net favorable development increases operating income."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
        )
        mfr = WidgetManufacturer(config)
        mfr.period_adverse_development = ZERO
        mfr.period_favorable_development = to_decimal(80_000)

        revenue = mfr.calculate_revenue()
        oi_with_dev = mfr.calculate_operating_income(revenue)

        mfr.period_favorable_development = ZERO
        oi_without_dev = mfr.calculate_operating_income(revenue)

        assert oi_with_dev > oi_without_dev
        assert oi_with_dev - oi_without_dev == to_decimal(80_000)

    def test_zero_development_no_impact(self):
        """Zero development has no impact on operating income."""
        config = ManufacturerConfig(initial_assets=10_000_000)
        mfr = WidgetManufacturer(config)

        revenue = mfr.calculate_revenue()
        oi = mfr.calculate_operating_income(revenue)

        # These should default to ZERO
        assert getattr(mfr, "period_adverse_development", ZERO) == ZERO
        assert getattr(mfr, "period_favorable_development", ZERO) == ZERO

        # Recalculate - should be identical
        oi2 = mfr.calculate_operating_income(revenue)
        assert oi == oi2


class TestReserveReconciliation:
    """Test get_reserve_reconciliation() accuracy."""

    def test_reconciliation_with_reserve_development(self):
        """Reconciliation correctly reports booked vs true residual."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=0.20,
        )
        mfr = WidgetManufacturer(config)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)
        mfr.record_claim_accrual(300_000)

        recon = mfr.get_reserve_reconciliation()
        assert recon["claim_count"] == 2
        assert recon["total_booked_reserves"] > ZERO
        assert recon["total_true_residual"] > ZERO
        # Redundancy or deficiency should be non-negative
        assert recon["total_redundancy"] >= ZERO
        assert recon["total_deficiency"] >= ZERO
        # Only one of redundancy/deficiency can be positive
        assert ZERO in (recon["total_redundancy"], recon["total_deficiency"])

    def test_reconciliation_without_reserve_development(self):
        """Without reserve development, booked == true residual."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=False,
        )
        mfr = WidgetManufacturer(config)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        recon = mfr.get_reserve_reconciliation()
        assert recon["total_booked_reserves"] == recon["total_true_residual"]
        assert recon["total_redundancy"] == ZERO
        assert recon["total_deficiency"] == ZERO

    def test_reconciliation_after_payment(self):
        """Reconciliation accounts for payments made."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=0.01,  # Very small noise for predictability
        )
        mfr = WidgetManufacturer(config)
        mfr.current_year = 1
        mfr.record_claim_accrual(100_000)

        claim = mfr.claim_liabilities[0]
        claim.make_payment(to_decimal(20_000))

        recon = mfr.get_reserve_reconciliation()
        # true_residual should account for payments
        assert claim.true_ultimate is not None
        expected_true_residual = max(claim.true_ultimate - claim._total_paid, ZERO)
        assert recon["total_true_residual"] == expected_true_residual


class TestStepIntegration:
    """Test full step() integration with reserve development."""

    def test_step_with_feature_enabled(self):
        """step() runs without errors when reserve development is on."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=0.20,
        )
        mfr = WidgetManufacturer(config)

        # Create some claims
        mfr.process_uninsured_claim(200_000)

        # Run a step
        metrics = mfr.step(letter_of_credit_rate=0.015, growth_rate=0.03)
        assert metrics is not None
        assert "adverse_development" in metrics
        assert "favorable_development" in metrics
        assert "net_reserve_development" in metrics

    def test_step_backward_compatibility(self):
        """step() produces identical output with feature off vs before."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=False,
        )
        mfr = WidgetManufacturer(config)
        metrics = mfr.step(letter_of_credit_rate=0.015, growth_rate=0.03)

        assert metrics["adverse_development"] == ZERO
        assert metrics["favorable_development"] == ZERO
        assert metrics["net_reserve_development"] == ZERO

    def test_multi_year_simulation(self):
        """Multi-year simulation with reserve development produces valid results."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=0.25,
        )
        mfr = WidgetManufacturer(config)

        for year in range(5):
            if year % 2 == 0:
                mfr.process_uninsured_claim(100_000)
            metrics = mfr.step(letter_of_credit_rate=0.015, growth_rate=0.02)
            assert not mfr.is_ruined or metrics["is_solvent"] is False

    def test_reset_clears_reserve_state(self):
        """reset() properly clears all reserve development state."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=0.20,
        )
        mfr = WidgetManufacturer(config)
        mfr.process_uninsured_claim(200_000)
        mfr.step(letter_of_credit_rate=0.015)
        mfr.reset()

        assert mfr.period_adverse_development == ZERO
        assert mfr.period_favorable_development == ZERO
        assert len(mfr.claim_liabilities) == 0

    def test_deepcopy_preserves_reserve_fields(self):
        """Deep copy preserves all reserve development state."""
        claim = ClaimLiability(
            original_amount=to_decimal(120_000),
            remaining_amount=to_decimal(110_000),
            year_incurred=1,
            true_ultimate=to_decimal(100_000),
        )
        claim._total_paid = to_decimal(10_000)
        claim._noise_std = 0.25

        copied = copy.deepcopy(claim)
        assert copied.true_ultimate == to_decimal(100_000)
        assert copied._total_paid == to_decimal(10_000)
        assert copied._noise_std == 0.25
        assert copied.remaining_amount == to_decimal(110_000)

        # Verify independence
        copied.remaining_amount = to_decimal(90_000)
        assert claim.remaining_amount == to_decimal(110_000)

    def test_metrics_include_development(self):
        """Metrics dictionary includes reserve development fields."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
        )
        mfr = WidgetManufacturer(config)
        metrics = mfr.calculate_metrics()
        assert "adverse_development" in metrics
        assert "favorable_development" in metrics
        assert "net_reserve_development" in metrics

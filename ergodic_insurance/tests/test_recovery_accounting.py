"""Tests for insurance claim recovery accounting."""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_program import (
    EnhancedInsuranceLayer,
    InsuranceProgram,
)
from ergodic_insurance.ledger import AccountName
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestRecoveryAccounting:
    """Test insurance claim recovery accounting functionality."""

    manufacturer: WidgetManufacturer

    def setup_method(self):
        """Set up test fixtures."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        self.manufacturer = WidgetManufacturer(config)

    def test_insurance_claim_creates_receivable(self):
        """Test that insurance claim coverage creates a receivable."""
        claim_amount = 5_000_000
        deductible = 1_000_000
        insurance_limit = 10_000_000

        # Process claim
        company_payment, insurance_payment = self.manufacturer.process_insurance_claim(
            claim_amount, deductible, insurance_limit
        )

        assert company_payment == 1_000_000  # Deductible
        assert insurance_payment == 4_000_000  # Remainder

        # Check receivable created
        receivables = self.manufacturer.insurance_accounting.get_total_receivables()
        assert receivables == 4_000_000

    def test_receive_insurance_recovery_payment(self):
        """Test receiving insurance recovery payment."""
        # Process claim first
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)
        initial_cash = self.manufacturer.cash

        # Receive partial payment
        result = self.manufacturer.receive_insurance_recovery(2_000_000)

        assert result["cash_received"] == 2_000_000
        assert result["remaining_receivables"] == 2_000_000
        assert self.manufacturer.cash == initial_cash + 2_000_000

    def test_multiple_claim_recoveries(self):
        """Test tracking multiple claim recoveries."""
        # Process multiple claims
        self.manufacturer.process_insurance_claim(3_000_000, 500_000, 10_000_000)
        self.manufacturer.process_insurance_claim(2_000_000, 500_000, 10_000_000)

        # Total insurance recoveries
        expected_recoveries = (3_000_000 - 500_000) + (2_000_000 - 500_000)
        actual_recoveries = self.manufacturer.insurance_accounting.get_total_receivables()

        assert actual_recoveries == expected_recoveries

    def test_full_recovery_payment(self):
        """Test receiving full recovery amount."""
        # Process claim
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)

        # Receive full insurance payment
        result = self.manufacturer.receive_insurance_recovery(4_000_000)

        assert result["cash_received"] == 4_000_000
        assert result["remaining_receivables"] == 0

    def test_overpayment_handling(self):
        """Test handling overpayment on recovery."""
        # Process claim
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)

        # Try to receive more than outstanding
        result = self.manufacturer.receive_insurance_recovery(5_000_000)

        # Should only receive up to outstanding amount
        assert result["cash_received"] == 4_000_000
        assert result["remaining_receivables"] == 0

    def test_claim_with_no_insurance_coverage(self):
        """Test claim with amount below deductible (no insurance coverage)."""
        claim_amount = 500_000
        deductible = 1_000_000

        # Process claim below deductible
        company_payment, insurance_payment = self.manufacturer.process_insurance_claim(
            claim_amount, deductible, 10_000_000
        )

        assert company_payment == 500_000
        assert insurance_payment == 0

        # No receivable should be created
        receivables = self.manufacturer.insurance_accounting.get_total_receivables()
        assert receivables == 0

    def test_claim_exceeding_limit(self):
        """Test claim exceeding insurance limit."""
        claim_amount = 15_000_000
        deductible = 1_000_000
        insurance_limit = 10_000_000

        # Process claim exceeding limit
        company_payment, insurance_payment = self.manufacturer.process_insurance_claim(
            claim_amount, deductible, insurance_limit
        )

        # Company pays deductible + excess
        assert company_payment == 5_000_000  # 1M deductible + 4M excess
        assert insurance_payment == 10_000_000  # Capped at limit

        # Receivable should be for insurance limit
        receivables = self.manufacturer.insurance_accounting.get_total_receivables()
        assert receivables == 10_000_000

    def test_sequential_recovery_payments(self):
        """Test receiving recovery payments over time."""
        # Process claim
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)
        initial_cash = self.manufacturer.cash

        # Receive payments in installments
        self.manufacturer.receive_insurance_recovery(1_000_000)
        self.manufacturer.receive_insurance_recovery(1_500_000)
        self.manufacturer.receive_insurance_recovery(1_500_000)

        # Check total received
        assert self.manufacturer.cash == initial_cash + 4_000_000
        assert self.manufacturer.insurance_accounting.get_total_receivables() == 0

    def test_recovery_with_claim_id_tracking(self):
        """Test recovery tracking with specific claim IDs."""
        # Process multiple claims (they get auto-generated IDs)
        self.manufacturer.current_year = 2024
        self.manufacturer.process_insurance_claim(3_000_000, 500_000, 10_000_000)

        self.manufacturer.current_year = 2025
        self.manufacturer.process_insurance_claim(2_000_000, 500_000, 10_000_000)

        # Check we have two recoveries with different IDs
        recoveries = self.manufacturer.insurance_accounting.recoveries
        assert len(recoveries) == 2
        assert recoveries[0].claim_id != recoveries[1].claim_id

        # Receive payment for specific claim
        claim_id = recoveries[0].claim_id
        result = self.manufacturer.receive_insurance_recovery(1_000_000, claim_id)

        assert result["cash_received"] == 1_000_000
        # First claim reduced, second unchanged
        assert recoveries[0].outstanding == 1_500_000
        assert recoveries[1].outstanding == 1_500_000

    def test_recovery_affects_cash_flow(self):
        """Test that recovery payments affect cash flow statement."""
        # Process claim creating receivable
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)

        # Track initial position
        initial_cash = self.manufacturer.cash

        # Receive recovery
        self.manufacturer.receive_insurance_recovery(2_000_000)

        # Cash should increase
        assert self.manufacturer.cash == initial_cash + 2_000_000

    def test_no_recovery_for_uninsured_claim(self):
        """Test that uninsured claims don't create receivables."""
        # Process uninsured claim
        self.manufacturer.process_uninsured_claim(1_000_000, immediate_payment=False)

        # No receivable should be created
        receivables = self.manufacturer.insurance_accounting.get_total_receivables()
        assert receivables == 0

    def test_recovery_summary(self):
        """Test getting recovery summary information."""
        # Process multiple claims
        self.manufacturer.process_insurance_claim(3_000_000, 500_000, 10_000_000)
        self.manufacturer.process_insurance_claim(2_000_000, 500_000, 10_000_000)

        # Receive partial payment
        self.manufacturer.receive_insurance_recovery(1_000_000)

        # Get summary
        summary = self.manufacturer.insurance_accounting.get_summary()

        assert summary["total_receivables"] == 3_000_000  # 4M total - 1M received
        assert summary["recovery_count"] == 2


class TestInsuranceReceivableLedgerIntegrity:
    """Test that insurance receivables are properly recorded in the general ledger (Issue #625)."""

    manufacturer: WidgetManufacturer

    def setup_method(self):
        """Set up test fixtures."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        self.manufacturer = WidgetManufacturer(config)

    def _get_ledger_balance(self, account: AccountName) -> Decimal:
        """Get ledger balance for an account."""
        return self.manufacturer.ledger.get_balance(account)

    def test_claim_debits_insurance_receivables_in_ledger(self):
        """Verify INSURANCE_RECEIVABLES is debited when a claim receivable is created."""
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)

        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        assert ledger_balance == 4_000_000

    def test_ledger_matches_insurance_accounting_receivables(self):
        """Verify ledger INSURANCE_RECEIVABLES balance matches InsuranceAccounting total."""
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)

        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        accounting_balance = self.manufacturer.insurance_accounting.get_total_receivables()
        assert ledger_balance == accounting_balance

    def test_receivable_balance_always_non_negative(self):
        """Verify INSURANCE_RECEIVABLES balance is always >= 0 (non-negative asset)."""
        # Create receivable
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)
        assert self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES) >= 0

        # Partial recovery
        self.manufacturer.receive_insurance_recovery(2_000_000)
        assert self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES) >= 0

        # Full recovery
        self.manufacturer.receive_insurance_recovery(2_000_000)
        assert self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES) >= 0

    def test_full_lifecycle_receivable_zeroes_out(self):
        """Process claim with insurance -> verify receivable > 0 -> receive recovery -> verify == 0."""
        # Step 1: Process claim
        company_payment, insurance_payment = self.manufacturer.process_insurance_claim(
            5_000_000, 1_000_000, 10_000_000
        )
        assert insurance_payment == 4_000_000

        # Step 2: Verify receivable > 0 in both systems
        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        assert ledger_balance > 0
        assert ledger_balance == 4_000_000

        # Step 3: Receive full recovery
        self.manufacturer.receive_insurance_recovery(4_000_000)

        # Step 4: Verify receivable == 0 in both systems
        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        accounting_balance = self.manufacturer.insurance_accounting.get_total_receivables()
        assert ledger_balance == 0
        assert accounting_balance == 0

    def test_balance_sheet_includes_outstanding_receivables(self):
        """Verify total_assets includes outstanding insurance receivables."""
        assets_before = self.manufacturer.total_assets

        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)

        # The receivable (4M) adds to assets; the INSURANCE_LOSS debit/credit net out
        # within the income statement, so total_assets should reflect the receivable.
        receivable = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        assert receivable == 4_000_000

    def test_no_receivable_ledger_entry_when_no_insurance(self):
        """Verify no INSURANCE_RECEIVABLES entry when claim is below deductible."""
        self.manufacturer.process_insurance_claim(500_000, 1_000_000, 10_000_000)

        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        assert ledger_balance == 0

    def test_multiple_claims_accumulate_in_ledger(self):
        """Verify multiple claims accumulate correctly in the ledger."""
        self.manufacturer.process_insurance_claim(3_000_000, 500_000, 10_000_000)
        self.manufacturer.process_insurance_claim(2_000_000, 500_000, 10_000_000)

        expected = (3_000_000 - 500_000) + (2_000_000 - 500_000)  # 4M
        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        accounting_balance = self.manufacturer.insurance_accounting.get_total_receivables()

        assert ledger_balance == expected
        assert ledger_balance == accounting_balance

    def test_partial_recovery_reduces_ledger_balance(self):
        """Verify partial recovery correctly reduces ledger balance."""
        self.manufacturer.process_insurance_claim(5_000_000, 1_000_000, 10_000_000)
        self.manufacturer.receive_insurance_recovery(1_500_000)

        ledger_balance = self._get_ledger_balance(AccountName.INSURANCE_RECEIVABLES)
        accounting_balance = self.manufacturer.insurance_accounting.get_total_receivables()

        assert ledger_balance == 2_500_000
        assert ledger_balance == accounting_balance


class TestInsuranceClaimNetEconomicEffect:
    """End-to-end economic-impact tests for insured vs uninsured claims.

    Compares period-end equity change against the expected impact under
    proper GAAP recognition (gross loss recognized, recovery offsets it,
    net = deductible).  Catches double-counting bugs in operating-income
    calculation (e.g., recording loss = deductible while crediting full
    recovery as revenue, which produces phantom income from claims).

    Configuration uses tax_rate=0 and retention_ratio=1.0 to remove tax
    and dividend noise from the equity-delta arithmetic.
    """

    @pytest.fixture
    def cfg(self) -> ManufacturerConfig:
        return ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=2.0,
            base_operating_margin=0.15,
            tax_rate=0.0,
            retention_ratio=1.0,
            ppe_ratio=0.25,
            working_capital_facility_limit=2_000_000,
        )

    @pytest.fixture
    def manufacturer(self, cfg: ManufacturerConfig) -> WidgetManufacturer:
        return WidgetManufacturer(cfg)

    @staticmethod
    def _baseline_op_income(cfg: ManufacturerConfig) -> float:
        return (
            float(cfg.initial_assets)
            * float(cfg.asset_turnover_ratio)
            * float(cfg.base_operating_margin)
        )

    def test_no_claim_baseline(self, manufacturer: WidgetManufacturer, cfg: ManufacturerConfig):
        """One step with no claim should change equity by ~op_income (= +$1.5M)."""
        eq0 = float(manufacturer.equity)
        manufacturer.step(letter_of_credit_rate=0.015, apply_stochastic=False)
        delta = float(manufacturer.equity) - eq0
        assert delta == pytest.approx(self._baseline_op_income(cfg), rel=0.01)

    def test_uninsured_claim_costs_full_claim_plus_lae(
        self, manufacturer: WidgetManufacturer, cfg: ManufacturerConfig
    ):
        """An uninsured $3M claim should cost equity ~$3.36M ($3M + 12% LAE)
        relative to the no-claim baseline.
        """
        manufacturer.process_uninsured_claim(3_000_000)
        manufacturer.step(letter_of_credit_rate=0.015, apply_stochastic=False)
        delta = float(manufacturer.equity) - float(cfg.initial_assets)
        net_cost = self._baseline_op_income(cfg) - delta
        lae_ratio = float(getattr(cfg, "lae_ratio", 0.12))
        assert net_cost == pytest.approx(3_000_000 * (1 + lae_ratio), rel=0.02)

    def test_insured_claim_costs_only_deductible_plus_lae(
        self, manufacturer: WidgetManufacturer, cfg: ManufacturerConfig
    ):
        """An insured $3M claim with $100K SIR and sufficient tower limit should
        cost equity ~$112K ($100K SIR + 12% LAE on company portion) relative
        to baseline.  This catches the recovery-double-counting bug — current
        behavior shows the company GAINING ~$3.7M from a $3M claim because the
        gross loss is never recognized while the recovery is credited as
        revenue (see manufacturer_income.py:calculate_operating_income).
        """
        layers = [
            EnhancedInsuranceLayer(attachment_point=100_000, limit=4_900_000, base_premium_rate=0.0)
        ]
        program = InsuranceProgram(layers=layers, deductible=100_000, name="test")
        result = program.process_claim(3_000_000)
        manufacturer.process_insurance_claim(
            claim_amount=3_000_000,
            deductible_amount=result.deductible_paid,
            insurance_recovery=result.insurance_recovery,
            record_period_loss=True,
        )
        manufacturer.step(letter_of_credit_rate=0.015, apply_stochastic=False)
        delta = float(manufacturer.equity) - float(cfg.initial_assets)
        net_cost = self._baseline_op_income(cfg) - delta
        lae_ratio = float(getattr(cfg, "lae_ratio", 0.12))
        # Absolute slack ($20K) absorbs LoC interest on collateral & Decimal rounding.
        assert net_cost == pytest.approx(100_000 * (1 + lae_ratio), abs=20_000), (
            f"Insured claim should cost ~${100_000 * (1 + lae_ratio):,.0f}, "
            f"got ${net_cost:,.0f}.  A negative or very large positive value "
            f"indicates phantom income from recovery being recognized without "
            f"offsetting gross-loss expense."
        )

    def test_insured_pl_equals_uninsured_pl_plus_net_recovery(self, cfg: ManufacturerConfig):
        """Equity delta(insured) - Equity delta(uninsured) should equal the
        insurance recovery, grossed up for the LAE saved on the recovered
        portion: $2.9M * (1 + 12%) = $3.248M.

        Cross-checks the two single-case tests against each other — if both
        are off by a consistent offset (e.g., uniform tax effect), this test
        still constrains their difference.
        """
        # Uninsured baseline
        mfr_u = WidgetManufacturer(cfg)
        mfr_u.process_uninsured_claim(3_000_000)
        mfr_u.step(letter_of_credit_rate=0.015, apply_stochastic=False)
        delta_u = float(mfr_u.equity) - float(cfg.initial_assets)

        # Insured with full coverage above $100K SIR
        mfr_i = WidgetManufacturer(cfg)
        layers = [
            EnhancedInsuranceLayer(attachment_point=100_000, limit=4_900_000, base_premium_rate=0.0)
        ]
        program = InsuranceProgram(layers=layers, deductible=100_000, name="test")
        result = program.process_claim(3_000_000)
        mfr_i.process_insurance_claim(
            claim_amount=3_000_000,
            deductible_amount=result.deductible_paid,
            insurance_recovery=result.insurance_recovery,
            record_period_loss=True,
        )
        mfr_i.step(letter_of_credit_rate=0.015, apply_stochastic=False)
        delta_i = float(mfr_i.equity) - float(cfg.initial_assets)

        diff = delta_i - delta_u
        lae_ratio = float(getattr(cfg, "lae_ratio", 0.12))
        expected_benefit = (3_000_000 - 100_000) * (1 + lae_ratio)
        assert diff == pytest.approx(expected_benefit, rel=0.05)

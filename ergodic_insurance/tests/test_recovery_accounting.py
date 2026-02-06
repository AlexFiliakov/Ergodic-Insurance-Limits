"""Tests for insurance claim recovery accounting."""

import pytest

from ergodic_insurance.config import ManufacturerConfig
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

"""Tests for insurance accounting module."""

import pytest

from ergodic_insurance.insurance_accounting import InsuranceAccounting, InsuranceRecovery


class TestInsuranceAccounting:
    """Test insurance accounting functionality."""

    def test_pay_annual_premium(self):
        """Test annual premium payment recording."""
        accounting = InsuranceAccounting()

        # Pay annual premium
        result = accounting.pay_annual_premium(1_200_000)

        assert result["cash_outflow"] == 1_200_000
        assert result["prepaid_asset"] == 1_200_000
        assert result["monthly_expense"] == 100_000  # 1.2M / 12

        assert accounting.prepaid_insurance == 1_200_000
        assert accounting.annual_premium == 1_200_000
        assert accounting.monthly_expense == 100_000
        assert accounting.current_month == 0

    def test_record_monthly_expense(self):
        """Test monthly expense amortization."""
        accounting = InsuranceAccounting()
        accounting.pay_annual_premium(1_200_000)

        # Record first month
        result = accounting.record_monthly_expense()
        assert result["insurance_expense"] == 100_000
        assert result["prepaid_reduction"] == 100_000
        assert result["remaining_prepaid"] == 1_100_000
        assert accounting.current_month == 1

        # Record second month
        result = accounting.record_monthly_expense()
        assert result["insurance_expense"] == 100_000
        assert result["remaining_prepaid"] == 1_000_000
        assert accounting.current_month == 2

    def test_full_year_amortization(self):
        """Test complete 12-month amortization cycle."""
        accounting = InsuranceAccounting()
        accounting.pay_annual_premium(1_200_000)

        total_expense = 0.0
        for month in range(12):
            result = accounting.record_monthly_expense()
            total_expense += result["insurance_expense"]

        # All premium should be expensed
        assert total_expense == 1_200_000
        assert accounting.prepaid_insurance == 0
        assert accounting.current_month == 12

    def test_partial_month_handling(self):
        """Test handling of partial months at period end."""
        accounting = InsuranceAccounting()
        accounting.pay_annual_premium(1_000_000)  # Not evenly divisible by 12

        total_expense = 0.0
        for month in range(12):
            result = accounting.record_monthly_expense()
            total_expense += result["insurance_expense"]

        # Should handle rounding appropriately
        assert abs(total_expense - 1_000_000) < 0.01
        assert accounting.prepaid_insurance < 0.01

    def test_record_claim_recovery(self):
        """Test recording insurance claim recoveries."""
        accounting = InsuranceAccounting()

        # Record recovery
        result = accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        assert result["insurance_receivable"] == 500_000
        assert result["total_receivables"] == 500_000
        assert len(accounting.recoveries) == 1

        recovery = accounting.recoveries[0]
        assert recovery.amount == 500_000
        assert recovery.claim_id == "CLAIM_001"
        assert recovery.year_approved == 2024
        assert recovery.outstanding == 500_000

    def test_multiple_recoveries(self):
        """Test tracking multiple claim recoveries."""
        accounting = InsuranceAccounting()

        # Record multiple recoveries
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)
        accounting.record_claim_recovery(300_000, "CLAIM_002", 2024)
        accounting.record_claim_recovery(200_000, "CLAIM_003", 2024)

        assert accounting.get_total_receivables() == 1_000_000
        assert len(accounting.recoveries) == 3

    def test_receive_recovery_payment(self):
        """Test receiving insurance recovery payments."""
        accounting = InsuranceAccounting()

        # Record recovery
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        # Receive partial payment
        result = accounting.receive_recovery_payment(200_000, "CLAIM_001")

        assert result["cash_received"] == 200_000
        assert result["receivable_reduction"] == 200_000
        assert result["remaining_receivables"] == 300_000

        recovery = accounting.recoveries[0]
        assert recovery.amount_received == 200_000
        assert recovery.outstanding == 300_000

    def test_receive_full_recovery(self):
        """Test receiving full recovery payment."""
        accounting = InsuranceAccounting()

        # Record recovery
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        # Receive full payment
        result = accounting.receive_recovery_payment(500_000, "CLAIM_001")

        assert result["cash_received"] == 500_000
        assert result["remaining_receivables"] == 0

        recovery = accounting.recoveries[0]
        assert recovery.outstanding == 0

    def test_overpayment_handling(self):
        """Test handling overpayment on recovery."""
        accounting = InsuranceAccounting()

        # Record recovery
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        # Try to receive more than outstanding
        result = accounting.receive_recovery_payment(600_000, "CLAIM_001")

        # Should only apply up to outstanding amount
        assert result["cash_received"] == 500_000
        assert result["remaining_receivables"] == 0

    def test_auto_claim_id_generation(self):
        """Test automatic claim ID generation."""
        accounting = InsuranceAccounting()

        # Record recovery without ID
        result = accounting.record_claim_recovery(500_000, year=2024)

        assert len(accounting.recoveries) == 1
        assert accounting.recoveries[0].claim_id == "CLAIM_2024_1"

    def test_get_amortization_schedule(self):
        """Test generating amortization schedule."""
        accounting = InsuranceAccounting()
        accounting.pay_annual_premium(1_200_000)

        # Record 3 months
        for _ in range(3):
            accounting.record_monthly_expense()

        # Get remaining schedule
        schedule = accounting.get_amortization_schedule()

        assert len(schedule) == 9  # 9 months remaining
        assert schedule[0]["month"] == 4
        assert schedule[0]["expense"] == 100_000
        assert schedule[-1]["month"] == 12
        assert schedule[-1]["remaining_prepaid"] == 0

    def test_reset_for_new_period(self):
        """Test resetting for new coverage period."""
        accounting = InsuranceAccounting()

        # Set up with premium and recoveries
        accounting.pay_annual_premium(1_200_000)
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        # Reset for new period
        accounting.reset_for_new_period()

        # Premium data should be reset
        assert accounting.prepaid_insurance == 0
        assert accounting.annual_premium == 0
        assert accounting.monthly_expense == 0
        assert accounting.current_month == 0

        # Recoveries should be preserved
        assert len(accounting.recoveries) == 1
        assert accounting.get_total_receivables() == 500_000

    def test_get_summary(self):
        """Test getting accounting summary."""
        accounting = InsuranceAccounting()

        # Set up accounting
        accounting.pay_annual_premium(1_200_000)
        for _ in range(3):
            accounting.record_monthly_expense()
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        summary = accounting.get_summary()

        assert summary["prepaid_insurance"] == 900_000
        assert summary["monthly_expense"] == 100_000
        assert summary["annual_premium"] == 1_200_000
        assert summary["months_elapsed"] == 3
        assert summary["months_remaining"] == 9
        assert summary["total_receivables"] == 500_000
        assert summary["recovery_count"] == 1

    def test_negative_premium_validation(self):
        """Test validation of negative premium amounts."""
        accounting = InsuranceAccounting()

        with pytest.raises(ValueError, match="Premium amount must be non-negative"):
            accounting.pay_annual_premium(-100_000)

    def test_negative_recovery_validation(self):
        """Test validation of negative recovery amounts."""
        accounting = InsuranceAccounting()

        with pytest.raises(ValueError, match="Recovery amount must be non-negative"):
            accounting.record_claim_recovery(-100_000)

    def test_invalid_payment_validation(self):
        """Test validation of invalid payment amounts."""
        accounting = InsuranceAccounting()
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        with pytest.raises(ValueError, match="Payment amount must be positive"):
            accounting.receive_recovery_payment(-100_000)

    def test_payment_without_receivables(self):
        """Test payment when no receivables exist."""
        accounting = InsuranceAccounting()

        with pytest.raises(ValueError, match="No outstanding recoveries"):
            accounting.receive_recovery_payment(100_000)

    def test_payment_invalid_claim_id(self):
        """Test payment with invalid claim ID."""
        accounting = InsuranceAccounting()
        accounting.record_claim_recovery(500_000, "CLAIM_001", 2024)

        with pytest.raises(ValueError, match="No recovery found with ID"):
            accounting.receive_recovery_payment(100_000, "INVALID_ID")

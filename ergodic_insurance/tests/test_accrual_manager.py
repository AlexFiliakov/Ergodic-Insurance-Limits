"""Unit tests for AccrualManager."""

import pytest

from ergodic_insurance.accrual_manager import (
    AccrualItem,
    AccrualManager,
    AccrualType,
    PaymentSchedule,
)


class TestAccrualManager:
    """Test AccrualManager functionality."""

    def test_initialization(self):
        """Test AccrualManager initialization."""
        manager = AccrualManager()
        assert manager.current_period == 0
        assert manager.get_total_accrued_expenses() == 0
        assert manager.get_total_accrued_revenues() == 0

    def test_record_expense_accrual_immediate(self):
        """Test recording an expense with immediate payment schedule."""
        manager = AccrualManager()
        accrual = manager.record_expense_accrual(
            item_type=AccrualType.WAGES,
            amount=10000.0,
            payment_schedule=PaymentSchedule.IMMEDIATE,
            description="Monthly wages",
        )

        assert accrual.amount == 10000.0
        assert accrual.item_type == AccrualType.WAGES
        assert accrual.payment_schedule == PaymentSchedule.IMMEDIATE
        assert accrual.payment_dates == [0]
        assert manager.get_total_accrued_expenses() == 10000.0

    def test_record_expense_accrual_quarterly(self):
        """Test recording an expense with quarterly payment schedule."""
        manager = AccrualManager()
        manager.current_period = 1  # February
        accrual = manager.record_expense_accrual(
            item_type=AccrualType.TAXES,
            amount=40000.0,
            payment_schedule=PaymentSchedule.QUARTERLY,
            description="Annual taxes",
        )

        # Should schedule payments for months 3, 5, 8, 11 (0-indexed)
        assert accrual.payment_dates == [3, 5, 8, 11]
        assert accrual.amount == 40000.0

    def test_record_expense_accrual_custom(self):
        """Test recording an expense with custom payment schedule."""
        manager = AccrualManager()
        payment_dates = [6, 12, 18]
        accrual = manager.record_expense_accrual(
            item_type=AccrualType.INSURANCE_CLAIMS,
            amount=100000.0,
            payment_schedule=PaymentSchedule.CUSTOM,
            payment_dates=payment_dates,
            description="Large claim settlement",
        )

        assert accrual.payment_dates == payment_dates
        assert accrual.amount == 100000.0

    def test_record_expense_accrual_custom_no_dates_raises(self):
        """Test that custom schedule without dates raises error."""
        manager = AccrualManager()
        with pytest.raises(ValueError, match="Custom schedule requires payment_dates"):
            manager.record_expense_accrual(
                item_type=AccrualType.OTHER, amount=5000.0, payment_schedule=PaymentSchedule.CUSTOM
            )

    def test_record_revenue_accrual(self):
        """Test recording accrued revenue."""
        manager = AccrualManager()
        collection_dates = [3, 6]
        accrual = manager.record_revenue_accrual(
            amount=20000.0, collection_dates=collection_dates, description="Service revenue"
        )

        assert accrual.item_type == AccrualType.REVENUE
        assert accrual.amount == 20000.0
        assert accrual.payment_dates == collection_dates
        assert manager.get_total_accrued_revenues() == 20000.0

    def test_process_payment_single_accrual(self):
        """Test processing payment against single accrual."""
        manager = AccrualManager()
        manager.record_expense_accrual(
            item_type=AccrualType.WAGES, amount=10000.0, payment_schedule=PaymentSchedule.IMMEDIATE
        )

        payments = manager.process_payment(AccrualType.WAGES, 5000.0)

        assert len(payments) == 1
        assert payments[0][1] == 5000.0  # Amount applied
        assert manager.get_total_accrued_expenses() == 5000.0  # Remaining

    def test_process_payment_multiple_accruals_fifo(self):
        """Test FIFO payment processing across multiple accruals."""
        manager = AccrualManager()

        # Create multiple wage accruals
        manager.record_expense_accrual(AccrualType.WAGES, 5000.0)
        manager.advance_period()
        manager.record_expense_accrual(AccrualType.WAGES, 7000.0)
        manager.advance_period()
        manager.record_expense_accrual(AccrualType.WAGES, 3000.0)

        # Process payment that spans multiple accruals
        payments = manager.process_payment(AccrualType.WAGES, 10000.0)

        assert len(payments) == 2
        assert payments[0][1] == 5000.0  # First accrual fully paid
        assert payments[1][1] == 5000.0  # Second accrual partially paid
        assert manager.get_total_accrued_expenses() == 5000.0  # Remaining

    def test_process_payment_overpayment(self):
        """Test processing payment larger than accruals."""
        manager = AccrualManager()
        manager.record_expense_accrual(AccrualType.TAXES, 5000.0)

        payments = manager.process_payment(AccrualType.TAXES, 8000.0)

        assert len(payments) == 1
        assert payments[0][1] == 5000.0
        assert manager.get_total_accrued_expenses() == 0

    def test_get_quarterly_tax_schedule(self):
        """Test quarterly tax payment schedule generation."""
        manager = AccrualManager()
        manager.current_period = 2  # March

        schedule = manager.get_quarterly_tax_schedule(40000.0)

        assert len(schedule) == 4
        assert all(amount == 10000.0 for _, amount in schedule)
        assert [period for period, _ in schedule] == [3, 5, 8, 11]

    def test_get_claim_payment_schedule_default(self):
        """Test claim payment schedule with default pattern."""
        manager = AccrualManager()
        manager.current_period = 0

        schedule = manager.get_claim_payment_schedule(100000.0)

        assert len(schedule) == 4
        expected_amounts = [40000.0, 30000.0, 20000.0, 10000.0]
        for i, (period, amount) in enumerate(schedule):
            assert period == i * 12
            assert amount == expected_amounts[i]

    def test_get_claim_payment_schedule_custom(self):
        """Test claim payment schedule with custom pattern."""
        manager = AccrualManager()
        custom_pattern = [0.5, 0.3, 0.2]

        schedule = manager.get_claim_payment_schedule(100000.0, custom_pattern)

        assert len(schedule) == 3
        expected_amounts = [50000.0, 30000.0, 20000.0]
        for i, (_, amount) in enumerate(schedule):
            assert amount == expected_amounts[i]

    def test_get_accruals_by_type(self):
        """Test retrieving accruals by type."""
        manager = AccrualManager()
        manager.record_expense_accrual(AccrualType.WAGES, 5000.0)
        manager.record_expense_accrual(AccrualType.WAGES, 3000.0)
        manager.record_expense_accrual(AccrualType.TAXES, 2000.0)
        manager.record_revenue_accrual(10000.0)

        wage_accruals = manager.get_accruals_by_type(AccrualType.WAGES)
        assert len(wage_accruals) == 2
        assert sum(a.amount for a in wage_accruals) == 8000.0

        tax_accruals = manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 1
        assert tax_accruals[0].amount == 2000.0

        revenue_accruals = manager.get_accruals_by_type(AccrualType.REVENUE)
        assert len(revenue_accruals) == 1
        assert revenue_accruals[0].amount == 10000.0

    def test_get_payments_due(self):
        """Test getting payments due in specific period."""
        manager = AccrualManager()

        # Create accruals with different payment schedules
        manager.record_expense_accrual(AccrualType.WAGES, 12000.0, PaymentSchedule.QUARTERLY)
        manager.record_expense_accrual(AccrualType.TAXES, 4000.0, PaymentSchedule.ANNUAL)

        # Check payments due in month 3 (first quarterly payment)
        payments_due = manager.get_payments_due(period=3)
        assert AccrualType.WAGES in payments_due
        assert payments_due[AccrualType.WAGES] == 3000.0  # 12000 / 4

        # Check payments due in month 12 (annual payment)
        payments_due = manager.get_payments_due(period=12)
        assert AccrualType.TAXES in payments_due
        assert payments_due[AccrualType.TAXES] == 4000.0

    def test_advance_period(self):
        """Test period advancement."""
        manager = AccrualManager()
        assert manager.current_period == 0

        manager.advance_period()
        assert manager.current_period == 1

        manager.advance_period(5)
        assert manager.current_period == 6

    def test_get_balance_sheet_items(self):
        """Test balance sheet item generation."""
        manager = AccrualManager()

        # Create various accruals
        manager.record_expense_accrual(AccrualType.WAGES, 5000.0)
        manager.record_expense_accrual(AccrualType.TAXES, 10000.0)
        manager.record_expense_accrual(AccrualType.INTEREST, 2000.0)
        manager.record_revenue_accrual(8000.0)

        balance_sheet = manager.get_balance_sheet_items()

        assert balance_sheet["accrued_expenses"] == 17000.0
        assert balance_sheet["accrued_revenues"] == 8000.0
        assert balance_sheet["accrued_wages"] == 5000.0
        assert balance_sheet["accrued_taxes"] == 10000.0
        assert balance_sheet["accrued_interest"] == 2000.0

    def test_clear_fully_paid(self):
        """Test clearing fully paid accruals."""
        manager = AccrualManager()

        # Create accruals
        manager.record_expense_accrual(AccrualType.WAGES, 5000.0)
        manager.record_expense_accrual(AccrualType.WAGES, 3000.0)
        manager.record_revenue_accrual(2000.0)

        # Fully pay first wage accrual and revenue
        manager.process_payment(AccrualType.WAGES, 5000.0)
        manager.process_payment(AccrualType.REVENUE, 2000.0)

        # Clear fully paid
        manager.clear_fully_paid()

        wage_accruals = manager.get_accruals_by_type(AccrualType.WAGES)
        assert len(wage_accruals) == 1
        assert wage_accruals[0].amount == 3000.0

        revenue_accruals = manager.get_accruals_by_type(AccrualType.REVENUE)
        assert len(revenue_accruals) == 0

    def test_accrual_item_properties(self):
        """Test AccrualItem properties."""
        item = AccrualItem(
            item_type=AccrualType.WAGES,
            amount=10000.0,
            period_incurred=0,
            payment_schedule=PaymentSchedule.IMMEDIATE,
        )

        assert item.remaining_balance == 10000.0
        assert not item.is_fully_paid

        item.amounts_paid.append(5000.0)
        assert item.remaining_balance == 5000.0
        assert not item.is_fully_paid

        item.amounts_paid.append(5000.0)
        assert item.remaining_balance == 0.0
        assert item.is_fully_paid

    def test_multiple_period_operations(self):
        """Test operations across multiple periods."""
        manager = AccrualManager()

        # Month 0: Record annual tax liability
        manager.record_expense_accrual(AccrualType.TAXES, 40000.0, PaymentSchedule.QUARTERLY)

        # Month 3: First quarterly payment
        manager.current_period = 3
        payments_due = manager.get_payments_due()
        assert AccrualType.TAXES in payments_due
        assert payments_due[AccrualType.TAXES] == 10000.0

        # Process the payment
        manager.process_payment(AccrualType.TAXES, 10000.0)
        assert manager.get_total_accrued_expenses() == 30000.0

        # Month 5: Second quarterly payment
        manager.current_period = 5
        payments_due = manager.get_payments_due()
        assert payments_due[AccrualType.TAXES] == 10000.0

        manager.process_payment(AccrualType.TAXES, 10000.0)
        assert manager.get_total_accrued_expenses() == 20000.0

    def test_concurrent_accruals_same_type(self):
        """Test handling multiple concurrent accruals of same type."""
        manager = AccrualManager()

        # Create multiple tax accruals
        manager.record_expense_accrual(
            AccrualType.TAXES, 20000.0, PaymentSchedule.QUARTERLY, description="Federal taxes"
        )
        manager.record_expense_accrual(
            AccrualType.TAXES, 8000.0, PaymentSchedule.QUARTERLY, description="State taxes"
        )

        # Both should be tracked
        tax_accruals = manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 2
        assert sum(a.amount for a in tax_accruals) == 28000.0

        # Payments should apply FIFO
        manager.process_payment(AccrualType.TAXES, 15000.0)
        assert tax_accruals[0].remaining_balance == 5000.0
        assert tax_accruals[1].remaining_balance == 8000.0

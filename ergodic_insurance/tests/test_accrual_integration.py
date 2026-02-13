"""Integration tests for accrual management within the manufacturer."""

import pytest

from ergodic_insurance.accrual_manager import AccrualType, PaymentSchedule
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestAccrualIntegration:
    """Test accrual management integration with manufacturer."""

    @pytest.fixture
    def config(self):
        """Create test manufacturer configuration."""
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            nol_carryforward_enabled=False,
            lae_ratio=0.0,
        )

    @pytest.fixture
    def manufacturer(self, config):
        """Create test manufacturer instance."""
        return WidgetManufacturer(config)

    def test_quarterly_tax_accrual_and_payment(self, manufacturer):
        """Test quarterly tax accrual and payment schedule."""
        # Generate income and accrue taxes
        metrics = manufacturer.step(time_resolution="annual")

        # Check that taxes were accrued
        assert metrics["accrued_taxes"] > 0
        initial_accrued_taxes = metrics["accrued_taxes"]

        # Simulate quarterly payments
        tax_payments_made = False
        for month in range(12):
            cash_before_step = manufacturer.cash
            month_metrics = manufacturer.step(time_resolution="monthly")

            # Check if tax payment was made (accrued taxes decreased)
            if month_metrics["accrued_taxes"] < initial_accrued_taxes:
                tax_payments_made = True
                initial_accrued_taxes = month_metrics["accrued_taxes"]  # Update for next comparison

        # Ensure at least one tax payment was made during the year
        assert tax_payments_made, "No tax payments were made during the year"

    def test_wage_accrual_immediate_payment(self, manufacturer):
        """Test wage accrual with immediate payment."""
        wage_amount = to_decimal(50000)

        # Record wage accrual
        manufacturer.record_wage_accrual(wage_amount, PaymentSchedule.IMMEDIATE)

        # Check accrual was recorded in AccrualManager (single source of truth - issue #238)
        assert manufacturer.accrual_manager.get_total_accrued_expenses() == wage_amount

        # Process payment in same period
        initial_cash = manufacturer.cash
        manufacturer.process_accrued_payments("annual")

        # Check payment was processed
        assert manufacturer.cash == initial_cash - wage_amount
        assert manufacturer.accrual_manager.get_total_accrued_expenses() == 0

    def test_claim_accrual_multi_year_payment(self, manufacturer):
        """Test insurance claim with multi-year payment schedule.

        Note: record_claim_accrual() now creates ClaimLiability objects
        (the single source of truth) instead of AccrualManager items.
        Claims are paid via pay_claim_liabilities(). See GitHub issue #213.
        """
        claim_amount = to_decimal(1_000_000)
        development_pattern = [0.5, 0.3, 0.2]  # 3-year payment pattern

        # Record claim accrual - now creates ClaimLiability
        manufacturer.record_claim_accrual(claim_amount, development_pattern)

        # Check ClaimLiability was created (single source of truth)
        assert len(manufacturer.claim_liabilities) == 1
        assert manufacturer.claim_liabilities[0].original_amount == claim_amount
        assert manufacturer.total_claim_liabilities == claim_amount

        # Simulate 3 years of payments using pay_claim_liabilities
        total_paid = to_decimal(0)

        for year in range(3):
            # Advance to next year
            manufacturer.current_year = year

            # Pay claims via ClaimLiability system (source of truth)
            paid_this_year = manufacturer.pay_claim_liabilities()
            total_paid += paid_this_year

        # Verify total payments match claim amount
        assert abs(total_paid - claim_amount) < to_decimal("0.01")
        # Verify claim is fully paid
        assert manufacturer.total_claim_liabilities < to_decimal("0.01")

    def test_accrual_in_metrics_history(self, manufacturer):
        """Test that accrual details appear in metrics history."""
        # Record various accruals with non-immediate payment schedules
        manufacturer.record_wage_accrual(to_decimal(25000), PaymentSchedule.ANNUAL)
        manufacturer.accrual_manager.record_expense_accrual(
            item_type=AccrualType.INTEREST,
            amount=to_decimal(5000),
            payment_schedule=PaymentSchedule.ANNUAL,
        )

        # Run a simulation step
        metrics = manufacturer.step()

        # Check metrics include accrual breakdown
        assert "accrued_wages" in metrics
        assert metrics["accrued_wages"] == to_decimal(25000)
        assert "accrued_interest" in metrics
        assert metrics["accrued_interest"] == to_decimal(5000)
        assert "accrued_taxes" in metrics
        assert metrics["accrued_taxes"] > 0  # Should have tax accrual from step

    def test_accrual_reset(self, manufacturer):
        """Test that accruals are properly reset."""
        # Create accruals
        manufacturer.record_wage_accrual(to_decimal(10000))
        manufacturer.accrual_manager.record_expense_accrual(
            AccrualType.TAXES, to_decimal(20000), PaymentSchedule.QUARTERLY
        )

        # Verify accruals exist
        assert manufacturer.accrual_manager.get_total_accrued_expenses() > 0

        # Reset manufacturer
        manufacturer.reset()

        # Verify accruals cleared (AccrualManager is single source of truth - issue #238)
        assert manufacturer.accrual_manager.get_total_accrued_expenses() == 0

    def test_monthly_resolution_accrual_sync(self, manufacturer):
        """Test accrual manager syncs with monthly resolution."""
        # Run monthly steps
        for month in range(6):
            manufacturer.step(time_resolution="monthly")

        # Check accrual manager period is synchronized
        # After 6 steps, we've processed periods 0-5, and current_period should be 5
        # The manufacturer time has advanced to month 6 after the increments
        assert manufacturer.current_month == 6
        assert manufacturer.accrual_manager.current_period == 5  # Last processed period

    def test_accrual_impact_on_cash_flow(self, manufacturer):
        """Test that accruals affect cash flow timing."""
        # Run first year to generate tax liability
        metrics_year1 = manufacturer.step(time_resolution="annual")

        net_income_year1 = metrics_year1["net_income"]
        accrued_taxes_year1 = metrics_year1["accrued_taxes"]

        # If profitable, should have accrued taxes
        if net_income_year1 > 0:
            assert accrued_taxes_year1 > 0

            # Track cash through quarterly payments
            cash_before = manufacturer.cash

            # Simulate second year with quarterly tax payments
            for quarter in range(4):
                # Each quarter is 3 months
                for month in range(3):
                    manufacturer.step(time_resolution="monthly")

            # Cash should have decreased by tax payments
            cash_after = manufacturer.cash
            # Note: Cash change includes operations, so we check accruals cleared
            remaining_accrued = manufacturer.accrual_manager.get_accruals_by_type(AccrualType.TAXES)

            # Previous year's taxes should be mostly paid
            unpaid_from_year1 = sum(
                a.remaining_balance for a in remaining_accrued if a.period_incurred == 0
            )
            assert unpaid_from_year1 < accrued_taxes_year1 * to_decimal(
                "0.1"
            )  # Less than 10% unpaid

    def test_accrual_with_insolvency(self, manufacturer):
        """Test accrual behavior when company becomes insolvent."""
        # Create significant accruals
        manufacturer.record_wage_accrual(to_decimal(100000))
        manufacturer.accrual_manager.record_expense_accrual(
            AccrualType.TAXES, to_decimal(50000), PaymentSchedule.QUARTERLY
        )

        # Force insolvency
        manufacturer.is_ruined = True

        # Process payments should not crash
        paid = manufacturer.process_accrued_payments("annual")

        # Payments should still process if cash available
        if manufacturer.cash > 0:
            assert paid >= 0

    def test_accrual_fifo_payment_order(self, manufacturer):
        """Test that accrual payments follow FIFO order."""
        # Create multiple wage accruals in sequence
        manufacturer.record_wage_accrual(to_decimal(10000))
        manufacturer.accrual_manager.advance_period()
        manufacturer.record_wage_accrual(to_decimal(20000))
        manufacturer.accrual_manager.advance_period()
        manufacturer.record_wage_accrual(to_decimal(30000))

        # Process partial payment
        manufacturer.accrual_manager.process_payment(AccrualType.WAGES, to_decimal(15000))

        # Check FIFO: first accrual fully paid, second partially
        wage_accruals = manufacturer.accrual_manager.get_accruals_by_type(AccrualType.WAGES)
        assert wage_accruals[0].is_fully_paid
        assert wage_accruals[1].remaining_balance == to_decimal(15000)
        assert wage_accruals[2].remaining_balance == to_decimal(30000)

    def test_discharge_reverses_liability_from_ledger(self, manufacturer):
        """Test that discharged accruals create a reversal ledger entry (Issue #1063).

        Per ASC 405-20, when a liability is extinguished due to limited liability,
        the ledger must record Dr ACCRUED_XXX / Cr RETAINED_EARNINGS.
        """
        # Create a wage accrual and its matching ledger entry
        wage_amount = to_decimal(100_000)
        manufacturer.record_wage_accrual(wage_amount, PaymentSchedule.IMMEDIATE)

        # Manually record the liability in the ledger (Dr WAGE_EXPENSE / Cr ACCRUED_WAGES)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.WAGE_EXPENSE,
            credit_account=AccountName.ACCRUED_WAGES,
            amount=wage_amount,
            transaction_type="accrual",
            description="Wage accrual for test",
        )

        accrued_wages_before = manufacturer.ledger.get_balance(AccountName.ACCRUED_WAGES)
        assert accrued_wages_before == wage_amount

        # Process with max_payable < total due → partial discharge
        max_payable = to_decimal(40_000)
        manufacturer.process_accrued_payments("annual", max_payable=max_payable)

        # The unpayable amount (60,000) should be discharged from the ledger
        accrued_wages_after = manufacturer.ledger.get_balance(AccountName.ACCRUED_WAGES)
        assert (
            accrued_wages_after == ZERO
        ), f"ACCRUED_WAGES should be zero after full discharge, got {accrued_wages_after}"

        # RETAINED_EARNINGS should absorb the discharged amount
        re_balance = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        assert re_balance >= to_decimal(
            60_000
        ), f"RETAINED_EARNINGS should include discharged amount, got {re_balance}"

        # Trial balance must remain balanced
        assert manufacturer.ledger.verify_balance()

    def test_discharge_clears_accrual_manager(self, manufacturer):
        """Test that discharged accruals are also cleared from AccrualManager (Issue #1063)."""
        wage_amount = to_decimal(50_000)
        manufacturer.record_wage_accrual(wage_amount, PaymentSchedule.IMMEDIATE)

        # Process with zero payable → full discharge
        manufacturer.process_accrued_payments("annual", max_payable=to_decimal(0))

        # AccrualManager should have no remaining balance
        remaining = manufacturer.accrual_manager.get_total_accrued_expenses()
        assert (
            remaining == ZERO
        ), f"AccrualManager should have zero remaining after discharge, got {remaining}"

    def test_tax_accrual_creates_ledger_entry(self, manufacturer):
        """Test that tax accrual creates Dr TAX_EXPENSE / Cr ACCRUED_TAXES (Issue #1081)."""
        # Run a step to generate income and accrue taxes
        metrics = manufacturer.step(time_resolution="annual")

        if metrics["net_income"] > 0:
            # TAX_EXPENSE should now be in the ledger
            tax_expense_balance = manufacturer.ledger.get_balance(AccountName.TAX_EXPENSE)
            assert (
                tax_expense_balance > ZERO
            ), "TAX_EXPENSE ledger balance should be positive after profitable year"

            # ACCRUED_TAXES in ledger should match AccrualManager
            accrued_taxes_ledger = manufacturer.ledger.get_balance(AccountName.ACCRUED_TAXES)
            accrued_taxes_manager = sum(
                a.remaining_balance
                for a in manufacturer.accrual_manager.get_accruals_by_type(AccrualType.TAXES)
                if not a.is_fully_paid
            )
            assert accrued_taxes_ledger == accrued_taxes_manager, (
                f"Ledger ACCRUED_TAXES ({accrued_taxes_ledger}) should match "
                f"AccrualManager ({accrued_taxes_manager})"
            )

            # Trial balance must remain balanced
            assert manufacturer.ledger.verify_balance()

    def test_tax_accrual_and_payment_ledger_flow(self, manufacturer):
        """Test full tax lifecycle: accrual → payment → ledger stays balanced (Issue #1081)."""
        # Year 1: Generate income, accrue taxes
        metrics_y1 = manufacturer.step(time_resolution="annual")
        accrued_taxes_y1 = metrics_y1["accrued_taxes"]

        if accrued_taxes_y1 > 0:
            # Verify accrual is in ledger
            assert manufacturer.ledger.get_balance(AccountName.ACCRUED_TAXES) > ZERO

            # Year 2: Monthly steps should pay quarterly taxes
            for month in range(12):
                manufacturer.step(time_resolution="monthly")

            # After payment, ACCRUED_TAXES should have decreased
            # (new accruals may have been added too)
            assert manufacturer.ledger.verify_balance()

    def test_discharge_all_accrual_types(self, manufacturer):
        """Test discharge works for taxes, wages, and interest (Issue #1063)."""
        # Record accruals for multiple types
        manufacturer.record_wage_accrual(to_decimal(10_000), PaymentSchedule.IMMEDIATE)
        manufacturer.accrual_manager.record_expense_accrual(
            AccrualType.INTEREST,
            to_decimal(5_000),
            PaymentSchedule.IMMEDIATE,
        )
        manufacturer.accrual_manager.record_expense_accrual(
            AccrualType.TAXES,
            to_decimal(20_000),
            PaymentSchedule.IMMEDIATE,
        )

        # Record matching ledger liabilities
        for account, amount in [
            (AccountName.ACCRUED_WAGES, to_decimal(10_000)),
            (AccountName.ACCRUED_INTEREST, to_decimal(5_000)),
            (AccountName.ACCRUED_TAXES, to_decimal(20_000)),
        ]:
            manufacturer.ledger.record_double_entry(
                date=manufacturer.current_year,
                debit_account=AccountName.OPERATING_EXPENSES,
                credit_account=account,
                amount=amount,
                transaction_type="accrual",
                description="Test accrual",
            )

        # Discharge all (zero payable)
        manufacturer.process_accrued_payments("annual", max_payable=to_decimal(0))

        # All accrued liability accounts should be zero
        assert manufacturer.ledger.get_balance(AccountName.ACCRUED_WAGES) == ZERO
        assert manufacturer.ledger.get_balance(AccountName.ACCRUED_INTEREST) == ZERO
        assert manufacturer.ledger.get_balance(AccountName.ACCRUED_TAXES) == ZERO

        # Trial balance must remain balanced
        assert manufacturer.ledger.verify_balance()

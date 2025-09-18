"""Integration tests for accrual management within the manufacturer."""

import pytest

from ergodic_insurance.accrual_manager import AccrualType, PaymentSchedule
from ergodic_insurance.config import ManufacturerConfig
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
        )

    @pytest.fixture
    def manufacturer(self, config):
        """Create test manufacturer instance."""
        return WidgetManufacturer(config)

    def test_quarterly_tax_accrual_and_payment(self, manufacturer):
        """Test quarterly tax accrual and payment schedule."""
        # Generate income and accrue taxes
        metrics = manufacturer.step(working_capital_pct=0.2, time_resolution="annual")

        # Check that taxes were accrued
        assert metrics["accrued_taxes"] > 0
        initial_accrued_taxes = metrics["accrued_taxes"]

        # Simulate quarterly payments
        initial_cash = manufacturer.cash
        for month in range(12):
            month_metrics = manufacturer.step(working_capital_pct=0.2, time_resolution="monthly")

            # Check for quarterly tax payments (months 3, 5, 8, 11)
            if month in [3, 5, 8, 11]:
                # Should have processed a tax payment
                assert month_metrics["accrued_taxes"] < initial_accrued_taxes
                assert manufacturer.cash < initial_cash  # Cash should decrease

    def test_wage_accrual_immediate_payment(self, manufacturer):
        """Test wage accrual with immediate payment."""
        wage_amount = 50000.0

        # Record wage accrual
        manufacturer.record_wage_accrual(wage_amount, PaymentSchedule.IMMEDIATE)

        # Check accrual was recorded
        assert manufacturer.accrued_expenses == wage_amount

        # Process payment in same period
        initial_cash = manufacturer.cash
        manufacturer.process_accrued_payments("annual")

        # Check payment was processed
        assert manufacturer.cash == initial_cash - wage_amount
        assert manufacturer.accrual_manager.get_total_accrued_expenses() == 0

    def test_claim_accrual_multi_year_payment(self, manufacturer):
        """Test insurance claim with multi-year payment schedule."""
        claim_amount = 1_000_000.0
        development_pattern = [0.5, 0.3, 0.2]  # 3-year payment pattern

        # Record claim accrual
        manufacturer.record_claim_accrual(claim_amount, development_pattern)

        # Check accrual was recorded
        total_accrued = manufacturer.accrual_manager.get_total_accrued_expenses()
        assert total_accrued >= claim_amount

        # Simulate 3 years of payments
        initial_cash = manufacturer.cash
        total_paid = 0.0

        for year in range(3):
            # Advance to next year
            manufacturer.current_year = year
            manufacturer.accrual_manager.current_period = year

            # Process payments
            paid_this_year = manufacturer.process_accrued_payments("annual")
            total_paid += paid_this_year

        # Verify total payments match claim amount
        assert abs(total_paid - claim_amount) < 0.01  # Floating point tolerance

    def test_accrual_in_metrics_history(self, manufacturer):
        """Test that accrual details appear in metrics history."""
        # Record various accruals
        manufacturer.record_wage_accrual(25000.0)
        manufacturer.accrual_manager.record_expense_accrual(
            item_type=AccrualType.INTEREST, amount=5000.0, payment_schedule=PaymentSchedule.ANNUAL
        )

        # Run a simulation step
        metrics = manufacturer.step(working_capital_pct=0.2)

        # Check metrics include accrual breakdown
        assert "accrued_wages" in metrics
        assert metrics["accrued_wages"] == 25000.0
        assert "accrued_interest" in metrics
        assert metrics["accrued_interest"] == 5000.0
        assert "accrued_taxes" in metrics
        assert metrics["accrued_taxes"] > 0  # Should have tax accrual from step

    def test_accrual_reset(self, manufacturer):
        """Test that accruals are properly reset."""
        # Create accruals
        manufacturer.record_wage_accrual(10000.0)
        manufacturer.accrual_manager.record_expense_accrual(
            AccrualType.TAXES, 20000.0, PaymentSchedule.QUARTERLY
        )

        # Verify accruals exist
        assert manufacturer.accrual_manager.get_total_accrued_expenses() > 0

        # Reset manufacturer
        manufacturer.reset()

        # Verify accruals cleared
        assert manufacturer.accrual_manager.get_total_accrued_expenses() == 0
        assert manufacturer.accrued_expenses == 0

    def test_monthly_resolution_accrual_sync(self, manufacturer):
        """Test accrual manager syncs with monthly resolution."""
        # Run monthly steps
        for month in range(6):
            manufacturer.step(working_capital_pct=0.2, time_resolution="monthly")

        # Check accrual manager period is synchronized
        expected_period = manufacturer.current_year * 12 + manufacturer.current_month
        assert manufacturer.accrual_manager.current_period == expected_period

    def test_accrual_impact_on_cash_flow(self, manufacturer):
        """Test that accruals affect cash flow timing."""
        # Run first year to generate tax liability
        metrics_year1 = manufacturer.step(working_capital_pct=0.2, time_resolution="annual")

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
                    manufacturer.step(working_capital_pct=0.2, time_resolution="monthly")

            # Cash should have decreased by tax payments
            cash_after = manufacturer.cash
            # Note: Cash change includes operations, so we check accruals cleared
            remaining_accrued = manufacturer.accrual_manager.get_accruals_by_type(AccrualType.TAXES)

            # Previous year's taxes should be mostly paid
            unpaid_from_year1 = sum(
                a.remaining_balance for a in remaining_accrued if a.period_incurred == 0
            )
            assert unpaid_from_year1 < accrued_taxes_year1 * 0.1  # Less than 10% unpaid

    def test_accrual_with_insolvency(self, manufacturer):
        """Test accrual behavior when company becomes insolvent."""
        # Create significant accruals
        manufacturer.record_wage_accrual(100000.0)
        manufacturer.accrual_manager.record_expense_accrual(
            AccrualType.TAXES, 50000.0, PaymentSchedule.QUARTERLY
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
        manufacturer.record_wage_accrual(10000.0)
        manufacturer.accrual_manager.advance_period()
        manufacturer.record_wage_accrual(20000.0)
        manufacturer.accrual_manager.advance_period()
        manufacturer.record_wage_accrual(30000.0)

        # Process partial payment
        manufacturer.accrual_manager.process_payment(AccrualType.WAGES, 15000.0)

        # Check FIFO: first accrual fully paid, second partially
        wage_accruals = manufacturer.accrual_manager.get_accruals_by_type(AccrualType.WAGES)
        assert wage_accruals[0].is_fully_paid
        assert wage_accruals[1].remaining_balance == 15000.0
        assert wage_accruals[2].remaining_balance == 30000.0

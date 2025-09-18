"""Tests for insurance premium amortization logic."""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_accounting import InsuranceAccounting
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestPremiumAmortization:
    """Test insurance premium amortization functionality."""

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

    def test_annual_premium_payment_creates_prepaid_asset(self):
        """Test that annual premium payment creates a prepaid asset."""
        initial_cash = self.manufacturer.cash
        premium_amount = 1_200_000

        # Pay annual premium
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        # Check prepaid asset created
        assert self.manufacturer.prepaid_insurance == premium_amount
        assert self.manufacturer.cash == initial_cash - premium_amount

        # Check insurance accounting module
        assert self.manufacturer.insurance_accounting.prepaid_insurance == premium_amount
        assert self.manufacturer.insurance_accounting.monthly_expense == 100_000

    def test_monthly_amortization_reduces_prepaid(self):
        """Test monthly amortization reduces prepaid asset."""
        premium_amount = 1_200_000
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        # Amortize one month
        expense = self.manufacturer.amortize_prepaid_insurance(1)

        assert expense == 100_000  # 1.2M / 12
        assert self.manufacturer.prepaid_insurance == 1_100_000
        assert self.manufacturer.period_insurance_premiums == 100_000

    def test_full_year_amortization_exhausts_prepaid(self):
        """Test that 12 months of amortization exhausts prepaid asset."""
        premium_amount = 1_200_000
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        total_expense = 0.0
        for month in range(12):
            expense = self.manufacturer.amortize_prepaid_insurance(1)
            total_expense += expense

        assert total_expense == premium_amount
        assert self.manufacturer.prepaid_insurance == 0
        assert self.manufacturer.insurance_accounting.prepaid_insurance == 0

    def test_amortization_schedule_calculation(self):
        """Test amortization schedule generation."""
        premium_amount = 1_200_000
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        # Amortize 3 months
        for _ in range(3):
            self.manufacturer.amortize_prepaid_insurance(1)

        # Get remaining schedule
        schedule = self.manufacturer.insurance_accounting.get_amortization_schedule()

        assert len(schedule) == 9  # 9 months remaining
        assert schedule[0]["expense"] == 100_000
        assert schedule[-1]["remaining_prepaid"] == 0

    def test_direct_monthly_premium_no_prepaid(self):
        """Test that monthly premium expense recording doesn't create prepaid asset.

        When is_annual=False, this records an expense for tax purposes but doesn't
        represent an actual cash payment. The cash reduction happens through the
        net income calculation during the step() method.
        """
        initial_cash = self.manufacturer.cash
        monthly_premium = 100_000

        # Record monthly premium expense (not a cash payment)
        self.manufacturer.record_insurance_premium(monthly_premium, is_annual=False)

        # Check no prepaid asset created
        assert self.manufacturer.prepaid_insurance == 0
        # Cash is not immediately reduced (expense flows through net income)
        assert self.manufacturer.cash == initial_cash
        # Expense is recorded for tax purposes
        assert self.manufacturer.period_insurance_premiums == monthly_premium

    def test_multiple_annual_periods(self):
        """Test handling multiple annual premium periods."""
        # First year
        self.manufacturer.record_insurance_premium(1_200_000, is_annual=True)

        # Amortize full year
        for _ in range(12):
            self.manufacturer.amortize_prepaid_insurance(1)

        assert self.manufacturer.prepaid_insurance == 0

        # Second year
        self.manufacturer.insurance_accounting.reset_for_new_period()
        self.manufacturer.record_insurance_premium(1_500_000, is_annual=True)

        assert self.manufacturer.prepaid_insurance == 1_500_000
        assert self.manufacturer.insurance_accounting.monthly_expense == 125_000

    def test_partial_month_amortization(self):
        """Test amortization with non-evenly divisible annual premium."""
        # Use premium not evenly divisible by 12
        premium_amount = 1_000_000
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        total_expense = 0.0
        for _ in range(12):
            expense = self.manufacturer.amortize_prepaid_insurance(1)
            total_expense += expense

        # Should handle rounding appropriately
        assert abs(total_expense - premium_amount) < 1.0
        assert self.manufacturer.prepaid_insurance < 1.0

    def test_amortization_with_no_prepaid(self):
        """Test amortization when no prepaid exists."""
        # Don't pay any premium
        expense = self.manufacturer.amortize_prepaid_insurance(1)

        assert expense == 0
        assert self.manufacturer.prepaid_insurance == 0
        assert self.manufacturer.period_insurance_premiums == 0

    def test_multi_month_amortization(self):
        """Test amortizing multiple months at once."""
        premium_amount = 1_200_000
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        # Amortize 3 months at once
        total_expense = self.manufacturer.amortize_prepaid_insurance(3)

        assert total_expense == 300_000  # 100K * 3
        assert self.manufacturer.prepaid_insurance == 900_000
        assert self.manufacturer.period_insurance_premiums == 300_000

    def test_premium_affects_balance_sheet(self):
        """Test that premium payments affect balance sheet correctly."""
        initial_assets = self.manufacturer.total_assets
        premium_amount = 1_200_000

        # Pay annual premium
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        # Total assets should be unchanged (cash becomes prepaid asset)
        assert self.manufacturer.total_assets == initial_assets

        # Current assets composition should change
        assert self.manufacturer.prepaid_insurance == premium_amount

    def test_amortization_affects_income(self):
        """Test that amortization affects operating income."""
        premium_amount = 1_200_000
        self.manufacturer.record_insurance_premium(premium_amount, is_annual=True)

        # Reset period costs
        self.manufacturer.reset_period_insurance_costs()

        # Amortize one month
        self.manufacturer.amortize_prepaid_insurance(1)

        # Calculate operating income
        revenue = self.manufacturer.calculate_revenue()
        operating_income = self.manufacturer.calculate_operating_income(revenue)

        # Operating income should be reduced by monthly expense
        base_income = revenue * self.manufacturer.base_operating_margin
        expected_income = base_income - 100_000  # Monthly premium expense

        assert abs(operating_income - expected_income) < 1.0

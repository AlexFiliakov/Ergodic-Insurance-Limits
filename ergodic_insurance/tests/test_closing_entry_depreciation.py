"""Test that closing entries correctly handle depreciation (Issue #1213).

The core bug: period_cash_expenses was computed as
  base_expenses - depreciation_expense
where base_expenses = revenue * (1 - base_operating_margin) already includes
depreciation embedded in the margin.  Subtracting depreciation_expense again
double-counted depreciation in the decomposition.

The fix closes temporary accounts via net_income directly, computing
cash outflow = revenue - net_income - depreciation so that:
  - RE change = net_income
  - Cash change = net_income + depreciation (indirect-method OCF, ASC 230-10-28)
  - All temporary accounts are zeroed after closing
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def manufacturer():
    """Create a manufacturer with standard config for closing entry tests."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,  # Retain all earnings to simplify verification
        capex_to_depreciation_ratio=0.0,  # No capex to isolate depreciation
    )
    return WidgetManufacturer(config)


class TestClosingEntryDepreciation:
    """Verify closing entries produce correct RE and CASH changes (Issue #1213)."""

    def test_re_change_equals_net_income_with_closing_entries(self, manufacturer):
        """RE should change by exactly net_income after closing entries."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        net_income = to_decimal(400_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(1_000_000)

        # Record revenue in ledger (as step() does)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )

        # Record depreciation (as record_depreciation() does)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        # Call closing entries via update_balance_sheet
        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re
        assert re_change == net_income, (
            f"RE should change by net_income ({net_income}), "
            f"got {re_change} (initial={initial_re}, final={final_re})"
        )

    def test_cash_change_equals_net_income_plus_depreciation(self, manufacturer):
        """Cash should change by net_income + depreciation (indirect-method OCF)."""
        initial_cash = manufacturer.cash
        net_income = to_decimal(400_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(1_000_000)

        # Record revenue
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )

        # Record depreciation
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        expected_cash = initial_cash + net_income + depreciation
        assert manufacturer.cash == expected_cash, (
            f"Cash should be initial + net_income + depreciation = "
            f"{initial_cash} + {net_income} + {depreciation} = {expected_cash}, "
            f"got {manufacturer.cash}"
        )

    def test_temporary_accounts_zeroed_after_closing(self, manufacturer):
        """SALES_REVENUE and DEPRECIATION_EXPENSE should be zero after closing."""
        net_income = to_decimal(400_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(1_000_000)

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        # Temporary accounts should be zeroed
        sales_bal = manufacturer.ledger.get_balance(AccountName.SALES_REVENUE)
        dep_bal = manufacturer.ledger.get_balance(AccountName.DEPRECIATION_EXPENSE)
        assert sales_bal == ZERO, f"SALES_REVENUE should be zero after closing, got {sales_bal}"
        assert dep_bal == ZERO, f"DEPRECIATION_EXPENSE should be zero after closing, got {dep_bal}"

    def test_no_operating_expenses_recorded_in_step(self, manufacturer):
        """step() should NOT record OPERATING_EXPENSES entries (Issue #1213).

        The old code recorded Dr OPERATING_EXPENSES / Cr CASH with an
        incorrectly computed period_cash_expenses that double-counted
        depreciation.  The fix removes this entry entirely.
        """
        # Run one step
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # OPERATING_EXPENSES should have zero balance (no entries recorded)
        op_exp_balance = manufacturer.ledger.get_balance(AccountName.OPERATING_EXPENSES)
        assert op_exp_balance == ZERO, (
            f"OPERATING_EXPENSES should not be used after Issue #1213 fix, "
            f"got balance {op_exp_balance}"
        )

    def test_step_produces_correct_re_change(self, manufacturer):
        """Full step() should produce RE change equal to retained net income."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)

        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re

        # With retention_ratio=1.0, RE change should equal net income
        expected_re_change = to_decimal(metrics["net_income"])
        assert (
            re_change == expected_re_change
        ), f"RE change ({re_change}) should equal net_income ({expected_re_change})"

    def test_step_closing_entries_do_not_use_operating_expenses(self, manufacturer):
        """After step(), OPERATING_EXPENSES should have zero balance.

        The old code recorded Dr OPERATING_EXPENSES / Cr CASH with an
        incorrectly decomposed amount. After the fix, closing entries
        bypass OPERATING_EXPENSES entirely and use net_income directly.
        """
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # Verify no OPERATING_EXPENSES entries were created by step()
        op_exp_balance = manufacturer.ledger.get_balance(AccountName.OPERATING_EXPENSES)
        assert (
            op_exp_balance == ZERO
        ), f"OPERATING_EXPENSES should have zero balance after step(), got {op_exp_balance}"

        # Verify all temporary accounts are zeroed after closing
        sales_bal = manufacturer.ledger.get_balance(AccountName.SALES_REVENUE)
        dep_bal = manufacturer.ledger.get_balance(AccountName.DEPRECIATION_EXPENSE)
        assert sales_bal == ZERO, f"SALES_REVENUE not zero after closing: {sales_bal}"
        assert dep_bal == ZERO, f"DEPRECIATION_EXPENSE not zero after closing: {dep_bal}"

    def test_loss_scenario_closing_entries(self, manufacturer):
        """Closing entries should work correctly when net_income is negative."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        initial_cash = manufacturer.cash
        net_income = to_decimal(-200_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(500_000)

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        # RE should change by net_income (negative = loss absorption)
        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re
        assert (
            re_change == net_income
        ), f"RE change ({re_change}) should equal net_income ({net_income}) in loss scenario"

        # Cash change = net_income + depreciation = -200K + 100K = -100K
        expected_cash = initial_cash + net_income + depreciation
        assert (
            manufacturer.cash == expected_cash
        ), f"Cash should be {expected_cash} in loss scenario, got {manufacturer.cash}"

"""Test that COGS and OPEX are recorded as explicit ledger entries (Issue #1326).

The fix records Dr COST_OF_GOODS_SOLD / Cr CASH and Dr OPERATING_EXPENSES / Cr CASH
during step(), then closes these temporary accounts to RETAINED_EARNINGS in the
closing entries.  Only the cash-consuming portions are recorded (depreciation is
already recorded via Dr DEPRECIATION_EXPENSE / Cr ACCUMULATED_DEPRECIATION).

Key invariants after step():
- Revenue - COGS - OPEX - Depreciation ≈ Operating Income
- Cash change = net_income + depreciation (indirect-method OCF, ASC 230-10-28)
- RE change = net_income (after dividends = 0 with retention_ratio = 1.0)
- All temporary accounts (SALES_REVENUE, COGS, OPEX, DEPRECIATION_EXPENSE) are zeroed
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.config.manufacturer import ExpenseRatioConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, EntryType, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def manufacturer():
    """Manufacturer with standard config for COGS/OPEX tests."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,  # Retain all earnings to simplify verification
        capex_to_depreciation_ratio=0.0,  # No capex to isolate depreciation
    )
    return WidgetManufacturer(config)


@pytest.fixture
def manufacturer_with_ratios():
    """Manufacturer with explicit expense_ratios config."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,
        capex_to_depreciation_ratio=0.0,
        expense_ratios=ExpenseRatioConfig(
            gross_margin_ratio=0.15,
            sga_expense_ratio=0.07,
            manufacturing_depreciation_allocation=0.7,
            admin_depreciation_allocation=0.3,
        ),
    )
    return WidgetManufacturer(config)


def _debit_entries(ledger, account):
    """Get debit entries for a given account."""
    return [e for e in ledger.get_entries(account=account) if e.entry_type == EntryType.DEBIT]


def _credit_entries(ledger, account):
    """Get credit entries for a given account."""
    return [e for e in ledger.get_entries(account=account) if e.entry_type == EntryType.CREDIT]


class TestCOGSOPEXLedgerEntries:
    """Verify COGS and OPEX are recorded as explicit ledger entries (Issue #1326)."""

    def test_cogs_recorded_after_step(self, manufacturer):
        """step() should record COGS debit entries in the ledger."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        cogs_debits = _debit_entries(manufacturer.ledger, AccountName.COST_OF_GOODS_SOLD)
        assert len(cogs_debits) > 0, "No COGS debit entries recorded by step()"

    def test_opex_recorded_after_step(self, manufacturer):
        """step() should record OPEX debit entries in the ledger."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        opex_debits = _debit_entries(manufacturer.ledger, AccountName.OPERATING_EXPENSES)
        assert len(opex_debits) > 0, "No OPEX debit entries recorded by step()"

    def test_cogs_opex_zeroed_after_closing(self, manufacturer):
        """COGS and OPEX should be zeroed after closing entries in step()."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        cogs_bal = manufacturer.ledger.get_balance(AccountName.COST_OF_GOODS_SOLD)
        opex_bal = manufacturer.ledger.get_balance(AccountName.OPERATING_EXPENSES)
        assert cogs_bal == ZERO, f"COGS should be zero after closing, got {cogs_bal}"
        assert opex_bal == ZERO, f"OPEX should be zero after closing, got {opex_bal}"

    def test_cash_change_with_closing_entries(self, manufacturer):
        """Cash change = net_income + depreciation when using closing entries directly.

        This tests the revenue/COGS/OPEX/closing subsystem in isolation
        via update_balance_sheet() (step() has additional cash flows from
        accruals, claims, etc.).
        """
        initial_cash = manufacturer.cash
        net_income = to_decimal(400_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(1_000_000)
        cogs = to_decimal(300_000)
        opex = to_decimal(50_000)

        # Record revenue (Dr CASH / Cr SALES_REVENUE)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )
        # Record depreciation (Dr DEP_EXP / Cr ACCUM_DEP)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )
        # Record COGS (Dr COGS / Cr CASH)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.COST_OF_GOODS_SOLD,
            credit_account=AccountName.CASH,
            amount=cogs,
            transaction_type=TransactionType.EXPENSE,
            description="Test COGS",
        )
        # Record OPEX (Dr OPEX / Cr CASH)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=opex,
            transaction_type=TransactionType.EXPENSE,
            description="Test OPEX",
        )

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
            cogs_expense=cogs,
            opex_expense=opex,
        )

        expected_cash = initial_cash + net_income + depreciation
        assert manufacturer.cash == expected_cash, (
            f"Cash should be initial + net_income + depreciation = "
            f"{initial_cash} + {net_income} + {depreciation} = {expected_cash}, "
            f"got {manufacturer.cash}"
        )

    def test_re_change_equals_net_income(self, manufacturer):
        """RE should change by net_income (with retention_ratio=1.0)."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)

        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        net_income = to_decimal(metrics["net_income"])
        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re

        assert (
            re_change == net_income
        ), f"RE change ({re_change}) should equal net_income ({net_income})"

    def test_income_statement_reconciliation(self, manufacturer):
        """Revenue - COGS - OPEX - Depreciation should approximate Operating Income."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # Get amounts from ledger entries using proper API
        revenue_credits = _credit_entries(manufacturer.ledger, AccountName.SALES_REVENUE)
        cogs_debits = _debit_entries(manufacturer.ledger, AccountName.COST_OF_GOODS_SOLD)
        opex_debits = _debit_entries(manufacturer.ledger, AccountName.OPERATING_EXPENSES)
        dep_debits = _debit_entries(manufacturer.ledger, AccountName.DEPRECIATION_EXPENSE)

        total_revenue = sum(e.amount for e in revenue_credits)
        total_cogs = sum(e.amount for e in cogs_debits)
        total_opex = sum(e.amount for e in opex_debits)
        total_dep = sum(e.amount for e in dep_debits)

        # Revenue - COGS - OPEX - Depreciation = Operating Income
        operating_income = total_revenue - total_cogs - total_opex - total_dep

        # Operating income should be base_operating_margin * revenue
        expected_oi = total_revenue * to_decimal(manufacturer.config.base_operating_margin)

        # Allow small tolerance for decimal arithmetic
        tolerance = total_revenue * to_decimal("0.001")
        assert abs(operating_income - expected_oi) <= tolerance, (
            f"Operating income mismatch: "
            f"Revenue({total_revenue}) - COGS({total_cogs}) - OPEX({total_opex}) "
            f"- Depreciation({total_dep}) = {operating_income}, "
            f"expected ≈ {expected_oi}"
        )

    def test_custom_expense_ratios(self, manufacturer_with_ratios):
        """Custom expense_ratios should produce correct COGS/OPEX split."""
        manufacturer = manufacturer_with_ratios
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # Get amounts from ledger entries
        cogs_debits = _debit_entries(manufacturer.ledger, AccountName.COST_OF_GOODS_SOLD)
        opex_debits = _debit_entries(manufacturer.ledger, AccountName.OPERATING_EXPENSES)
        revenue_credits = _credit_entries(manufacturer.ledger, AccountName.SALES_REVENUE)
        dep_debits = _debit_entries(manufacturer.ledger, AccountName.DEPRECIATION_EXPENSE)

        total_revenue = sum(e.amount for e in revenue_credits)
        total_cogs = sum(e.amount for e in cogs_debits)
        total_opex = sum(e.amount for e in opex_debits)
        total_dep = sum(e.amount for e in dep_debits)

        # With gross_margin_ratio=0.15, cogs_ratio=0.85, sga=0.07
        # mfg_dep_alloc=0.7, admin_dep_alloc=0.3
        expected_cogs_cash = max(
            ZERO,
            total_revenue * to_decimal("0.85") - total_dep * to_decimal("0.7"),
        )
        expected_opex_cash = max(
            ZERO,
            total_revenue * to_decimal("0.07") - total_dep * to_decimal("0.3"),
        )

        assert (
            total_cogs == expected_cogs_cash
        ), f"COGS mismatch: got {total_cogs}, expected {expected_cogs_cash}"
        assert (
            total_opex == expected_opex_cash
        ), f"OPEX mismatch: got {total_opex}, expected {expected_opex_cash}"

    def test_ledger_balance_after_step(self, manufacturer):
        """Ledger should balance after step() with COGS/OPEX entries."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        assert (
            manufacturer.ledger.verify_balance()
        ), "Ledger should balance after step() with COGS/OPEX entries"

    def test_backward_compat_zero_cogs_opex(self, manufacturer):
        """update_balance_sheet() works without COGS/OPEX params (backward compat)."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
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

        # Call without cogs_expense/opex_expense (backward compat)
        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        # RE should still change by net_income
        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re
        assert re_change == net_income, (
            f"RE change ({re_change}) should equal net_income ({net_income}) "
            f"even without COGS/OPEX params"
        )

    def test_cogs_opex_amounts_non_negative(self, manufacturer):
        """COGS and OPEX cash amounts should never be negative."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        cogs_debits = _debit_entries(manufacturer.ledger, AccountName.COST_OF_GOODS_SOLD)
        opex_debits = _debit_entries(manufacturer.ledger, AccountName.OPERATING_EXPENSES)

        for e in cogs_debits:
            assert e.amount >= ZERO, f"COGS amount should be non-negative, got {e.amount}"
        for e in opex_debits:
            assert e.amount >= ZERO, f"OPEX amount should be non-negative, got {e.amount}"

    def test_default_ratios_without_expense_config(self, manufacturer):
        """When no expense_ratios config, default ratios should be used."""
        assert getattr(manufacturer.config, "expense_ratios", None) is None

        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        cogs_debits = _debit_entries(manufacturer.ledger, AccountName.COST_OF_GOODS_SOLD)
        opex_debits = _debit_entries(manufacturer.ledger, AccountName.OPERATING_EXPENSES)
        assert len(cogs_debits) > 0, "COGS should be recorded with default ratios"
        assert len(opex_debits) > 0, "OPEX should be recorded with default ratios"

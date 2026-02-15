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
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
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

        # Issue #1302: Revenue goes to AR (accrual basis)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )

        # Simulate cash collection (DSO=0 for simplicity — all collected)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.ACCOUNTS_RECEIVABLE,
            amount=revenue,
            transaction_type=TransactionType.COLLECTION,
            description="Test collection",
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
        """All temporary accounts should be zero after closing (Issue #1297)."""
        from ergodic_insurance.ledger import CHART_OF_ACCOUNTS, AccountType

        net_income = to_decimal(400_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(1_000_000)
        cogs = to_decimal(300_000)
        opex = to_decimal(50_000)
        insurance_exp = to_decimal(30_000)
        tax_exp = to_decimal(20_000)

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
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

        # Issue #1326: Record COGS and OPEX entries
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.COST_OF_GOODS_SOLD,
            credit_account=AccountName.CASH,
            amount=cogs,
            transaction_type=TransactionType.EXPENSE,
            description="Test COGS",
        )
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=opex,
            transaction_type=TransactionType.EXPENSE,
            description="Test OPEX",
        )

        # Issue #1297: Record additional expense accounts that were previously unclosed
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_EXPENSE,
            credit_account=AccountName.PREPAID_INSURANCE,
            amount=insurance_exp,
            transaction_type=TransactionType.EXPENSE,
            description="Test insurance amortization",
        )
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.TAX_EXPENSE,
            credit_account=AccountName.ACCRUED_TAXES,
            amount=tax_exp,
            transaction_type=TransactionType.EXPENSE,
            description="Test tax accrual",
        )

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
            cogs_expense=cogs,
            opex_expense=opex,
        )

        # Issue #1297: ALL temporary accounts should be zeroed
        for account, acct_type in CHART_OF_ACCOUNTS.items():
            if acct_type in (AccountType.REVENUE, AccountType.EXPENSE):
                balance = manufacturer.ledger.get_balance(account)
                assert (
                    balance == ZERO
                ), f"{account.value} should be zero after closing, got {balance}"

    def test_cogs_opex_zeroed_after_closing_in_step(self, manufacturer):
        """COGS and OPEX should be zeroed after closing entries in step() (Issue #1326).

        step() now records COGS and OPEX as explicit Dr EXPENSE / Cr CASH
        entries, and closing entries zero them out.
        """
        # Run one step
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # All temporary expense accounts should have zero balance after closing
        cogs_bal = manufacturer.ledger.get_balance(AccountName.COST_OF_GOODS_SOLD)
        opex_bal = manufacturer.ledger.get_balance(AccountName.OPERATING_EXPENSES)
        assert cogs_bal == ZERO, f"COST_OF_GOODS_SOLD should be zero after closing, got {cogs_bal}"
        assert opex_bal == ZERO, f"OPERATING_EXPENSES should be zero after closing, got {opex_bal}"

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

    def test_temporary_accounts_zeroed_after_step(self, manufacturer):
        """After step(), ALL temporary accounts should have zero balance (Issue #1297).

        step() records revenue, COGS, OPEX, depreciation, and potentially
        insurance/tax entries, then closing entries zero all temporary accounts.
        """
        from ergodic_insurance.ledger import CHART_OF_ACCOUNTS, AccountType

        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # Issue #1297: Verify EVERY temporary account is zeroed after closing
        for account, acct_type in CHART_OF_ACCOUNTS.items():
            if acct_type in (AccountType.REVENUE, AccountType.EXPENSE):
                balance = manufacturer.ledger.get_balance(account)
                assert (
                    balance == ZERO
                ), f"{account.value} ({acct_type.value}) not zero after closing: {balance}"

    def test_loss_scenario_closing_entries(self, manufacturer):
        """Closing entries should work correctly when net_income is negative."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        initial_cash = manufacturer.cash
        net_income = to_decimal(-200_000)
        depreciation = to_decimal(100_000)
        revenue = to_decimal(500_000)

        # Issue #1302: Revenue to AR (accrual basis)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Test revenue",
        )

        # Simulate cash collection (all collected)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.ACCOUNTS_RECEIVABLE,
            amount=revenue,
            transaction_type=TransactionType.COLLECTION,
            description="Test collection",
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


class TestGAAPClosingAllTemporaryAccounts:
    """Verify all temporary accounts are closed per GAAP (Issue #1297).

    The core bug: _record_closing_entries only closed SALES_REVENUE,
    DEPRECIATION_EXPENSE, COGS, and OPEX to retained earnings.  All other
    income statement accounts (INSURANCE_EXPENSE, INSURANCE_LOSS, TAX_EXPENSE,
    INTEREST_EXPENSE, LAE_EXPENSE, RESERVE_DEVELOPMENT, INTEREST_INCOME,
    INSURANCE_RECOVERY, etc.) were never closed, causing their balances to
    persist across periods.
    """

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with standard config for GAAP closing tests."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
            capex_to_depreciation_ratio=0.0,
        )
        return WidgetManufacturer(config)

    def test_all_temporary_accounts_zeroed_with_multiple_expense_types(self, manufacturer):
        """Every revenue and expense account must be zero after closing.

        Records entries to multiple previously-unclosed accounts and verifies
        closing entries zero them all.
        """
        from ergodic_insurance.ledger import CHART_OF_ACCOUNTS, AccountType

        revenue = to_decimal(1_000_000)
        depreciation = to_decimal(100_000)
        cogs = to_decimal(300_000)
        opex = to_decimal(50_000)
        insurance_exp = to_decimal(80_000)
        insurance_loss = to_decimal(40_000)
        tax_exp = to_decimal(75_000)
        lae_exp = to_decimal(10_000)
        interest_income = to_decimal(5_000)

        # Revenue
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Sales",
        )

        # Depreciation (non-cash)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Depreciation",
        )

        # COGS and OPEX (cash)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.COST_OF_GOODS_SOLD,
            credit_account=AccountName.CASH,
            amount=cogs,
            transaction_type=TransactionType.EXPENSE,
            description="COGS",
        )
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=opex,
            transaction_type=TransactionType.EXPENSE,
            description="OPEX",
        )

        # Insurance expense (amortization of prepaid)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_EXPENSE,
            credit_account=AccountName.PREPAID_INSURANCE,
            amount=insurance_exp,
            transaction_type=TransactionType.EXPENSE,
            description="Insurance amortization",
        )

        # Insurance loss (claim liability)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_LOSS,
            credit_account=AccountName.CLAIM_LIABILITIES,
            amount=insurance_loss,
            transaction_type=TransactionType.INSURANCE_CLAIM,
            description="Claim loss",
        )

        # Tax expense (accrual)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.TAX_EXPENSE,
            credit_account=AccountName.ACCRUED_TAXES,
            amount=tax_exp,
            transaction_type=TransactionType.EXPENSE,
            description="Tax accrual",
        )

        # LAE expense
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.LAE_EXPENSE,
            credit_account=AccountName.CLAIM_LIABILITIES,
            amount=lae_exp,
            transaction_type=TransactionType.INSURANCE_CLAIM,
            description="LAE",
        )

        # Interest income (revenue)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.INTEREST_INCOME,
            amount=interest_income,
            transaction_type=TransactionType.REVENUE,
            description="Interest earned",
        )

        # net_income = revenue + interest_income - all expenses
        net_income = (
            revenue
            + interest_income
            - depreciation
            - cogs
            - opex
            - insurance_exp
            - tax_exp
            - insurance_loss
            - lae_exp
        )

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
            cogs_expense=cogs,
            opex_expense=opex,
        )

        # Every temporary account must be zero
        for account, acct_type in CHART_OF_ACCOUNTS.items():
            if acct_type in (AccountType.REVENUE, AccountType.EXPENSE):
                balance = manufacturer.ledger.get_balance(account)
                assert balance == ZERO, (
                    f"{account.value} ({acct_type.value}) should be zero "
                    f"after closing, got {balance}"
                )

    def test_re_equals_net_income_with_all_accounts(self, manufacturer):
        """RE should change by exactly net_income even with many temp accounts."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)

        revenue = to_decimal(1_000_000)
        depreciation = to_decimal(100_000)
        insurance_exp = to_decimal(80_000)
        tax_exp = to_decimal(75_000)
        interest_income = to_decimal(5_000)

        # Revenue
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Sales",
        )
        # Collect cash
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.ACCOUNTS_RECEIVABLE,
            amount=revenue,
            transaction_type=TransactionType.COLLECTION,
            description="Collection",
        )
        # Depreciation
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Depreciation",
        )
        # Insurance expense
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_EXPENSE,
            credit_account=AccountName.PREPAID_INSURANCE,
            amount=insurance_exp,
            transaction_type=TransactionType.EXPENSE,
            description="Insurance",
        )
        # Tax expense
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.TAX_EXPENSE,
            credit_account=AccountName.ACCRUED_TAXES,
            amount=tax_exp,
            transaction_type=TransactionType.EXPENSE,
            description="Tax",
        )
        # Interest income
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.INTEREST_INCOME,
            amount=interest_income,
            transaction_type=TransactionType.REVENUE,
            description="Interest",
        )

        net_income = revenue + interest_income - depreciation - insurance_exp - tax_exp

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re
        assert (
            re_change == net_income
        ), f"RE change ({re_change}) should equal net_income ({net_income})"

    def test_favorable_reserve_development_closes_correctly(self, manufacturer):
        """Favorable reserve development (credit balance on expense account)
        should be properly closed to RE.
        """
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)

        revenue = to_decimal(1_000_000)
        depreciation = to_decimal(100_000)
        favorable_dev = to_decimal(20_000)

        # Revenue
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.SALES_REVENUE,
            amount=revenue,
            transaction_type=TransactionType.REVENUE,
            description="Sales",
        )
        # Depreciation
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Depreciation",
        )
        # Favorable reserve development: Dr CLAIM_LIABILITIES / Cr RESERVE_DEVELOPMENT
        # This gives RESERVE_DEVELOPMENT a CREDIT balance (negative for expense account)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CLAIM_LIABILITIES,
            credit_account=AccountName.RESERVE_DEVELOPMENT,
            amount=favorable_dev,
            transaction_type=TransactionType.RESERVE_DEVELOPMENT,
            description="Favorable reserve development",
        )

        # net_income includes favorable development as a gain
        net_income = revenue - depreciation + favorable_dev

        manufacturer.update_balance_sheet(
            net_income,
            depreciation_expense=depreciation,
            period_revenue=revenue,
        )

        # RESERVE_DEVELOPMENT should be zero (credit balance was closed)
        rd_bal = manufacturer.ledger.get_balance(AccountName.RESERVE_DEVELOPMENT)
        assert rd_bal == ZERO, f"RESERVE_DEVELOPMENT should be zero after closing, got {rd_bal}"

        # RE should change by net_income
        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re
        assert (
            re_change == net_income
        ), f"RE change ({re_change}) should equal net_income ({net_income})"

    def test_multi_period_accounts_reset_each_period(self, manufacturer):
        """Temporary accounts should not accumulate across periods (Issue #1297).

        This was the core bug: get_balance() returned cumulative multi-period
        balances instead of single-period amounts.
        """
        # Run period 1
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # After period 1: all temp accounts should be zero
        ins_bal_1 = manufacturer.ledger.get_balance(AccountName.INSURANCE_EXPENSE)
        tax_bal_1 = manufacturer.ledger.get_balance(AccountName.TAX_EXPENSE)
        assert ins_bal_1 == ZERO, f"Period 1: INSURANCE_EXPENSE not zero: {ins_bal_1}"
        assert tax_bal_1 == ZERO, f"Period 1: TAX_EXPENSE not zero: {tax_bal_1}"

        # Run period 2
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # After period 2: still zero — should NOT accumulate
        ins_bal_2 = manufacturer.ledger.get_balance(AccountName.INSURANCE_EXPENSE)
        tax_bal_2 = manufacturer.ledger.get_balance(AccountName.TAX_EXPENSE)
        assert ins_bal_2 == ZERO, (
            f"Period 2: INSURANCE_EXPENSE accumulated ({ins_bal_2}), "
            f"should be zero after closing"
        )
        assert tax_bal_2 == ZERO, (
            f"Period 2: TAX_EXPENSE accumulated ({tax_bal_2}), " f"should be zero after closing"
        )

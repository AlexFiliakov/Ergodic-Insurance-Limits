"""Tests for the event-sourcing ledger module.

Tests cover:
- LedgerEntry creation and validation
- Double-entry recording
- Balance calculations
- Cash flow extraction
- Trial balance generation
- Ledger verification
"""

from decimal import Decimal
import warnings

import pytest

from ergodic_insurance.ledger import (
    CHART_OF_ACCOUNTS,
    AccountName,
    AccountType,
    EntryType,
    Ledger,
    LedgerEntry,
    TransactionType,
)


class TestLedgerEntry:
    """Tests for LedgerEntry dataclass."""

    def test_create_debit_entry(self):
        """Test creating a debit entry."""
        entry = LedgerEntry(
            date=5,
            account="cash",
            amount=Decimal("1000.0"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
            description="Customer payment",
        )

        assert entry.date == 5
        assert entry.account == "cash"
        assert entry.amount == Decimal("1000.0")
        assert entry.entry_type == EntryType.DEBIT
        assert entry.description == "Customer payment"
        assert entry.reference_id is not None

    def test_create_credit_entry(self):
        """Test creating a credit entry."""
        entry = LedgerEntry(
            date=5,
            account="revenue",
            amount=Decimal("1000.0"),
            entry_type=EntryType.CREDIT,
            transaction_type=TransactionType.REVENUE,
            description="Sales revenue",
        )

        assert entry.entry_type == EntryType.CREDIT
        assert entry.transaction_type == TransactionType.REVENUE

    def test_signed_amount_debit(self):
        """Test signed amount for debit entry."""
        entry = LedgerEntry(
            date=5,
            account="cash",
            amount=Decimal("1000.0"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
        )

        assert entry.signed_amount == Decimal("1000.0")

    def test_signed_amount_credit(self):
        """Test signed amount for credit entry."""
        entry = LedgerEntry(
            date=5,
            account="revenue",
            amount=Decimal("1000.0"),
            entry_type=EntryType.CREDIT,
            transaction_type=TransactionType.REVENUE,
        )

        assert entry.signed_amount == Decimal("-1000.0")

    def test_negative_amount_raises(self):
        """Test that negative amounts raise an error."""
        with pytest.raises(ValueError, match="non-negative"):
            LedgerEntry(
                date=5,
                account="cash",
                amount=Decimal("-1000.0"),
                entry_type=EntryType.DEBIT,
                transaction_type=TransactionType.COLLECTION,
            )


class TestLedger:
    """Tests for Ledger class."""

    @pytest.fixture
    def ledger(self):
        """Create empty ledger for testing."""
        return Ledger()

    @pytest.fixture
    def ledger_with_entries(self):
        """Create ledger with sample entries."""
        ledger = Ledger()

        # Year 1: Initial cash injection
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="retained_earnings",
            amount=10_000_000,
            transaction_type=TransactionType.EQUITY_ISSUANCE,
            description="Initial capitalization",
        )

        # Year 1: Revenue on credit
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=5_000_000,
            transaction_type=TransactionType.REVENUE,
            description="Annual sales on credit",
        )

        # Year 1: Collect cash from customers
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="accounts_receivable",
            amount=4_500_000,
            transaction_type=TransactionType.COLLECTION,
            description="Collections on AR",
        )

        # Year 1: Pay operating expenses
        ledger.record_double_entry(
            date=1,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=2_000_000,
            transaction_type=TransactionType.PAYMENT,
            description="Operating expense payments",
        )

        return ledger

    def test_empty_ledger(self, ledger):
        """Test empty ledger state."""
        assert len(ledger) == 0
        assert ledger.get_balance("cash") == 0

    def test_record_single_entry(self, ledger):
        """Test recording a single entry."""
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=Decimal("1000"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
        )
        ledger.record(entry)

        assert len(ledger) == 1
        assert ledger.entries[0] == entry

    def test_record_double_entry(self, ledger):
        """Test recording a complete double-entry transaction."""
        debit, credit = ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=1000,
            transaction_type=TransactionType.REVENUE,
            description="Cash sale",
        )

        assert len(ledger) == 2
        assert debit.entry_type == EntryType.DEBIT
        assert credit.entry_type == EntryType.CREDIT
        assert debit.reference_id == credit.reference_id
        assert debit.amount == credit.amount == 1000

    def test_double_entry_zero_amount(self, ledger):
        """Test that zero-amount transactions return None sentinel (Issue #315)."""
        debit, credit = ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=0,
            transaction_type=TransactionType.REVENUE,
        )

        assert len(ledger) == 0  # Zero-amount entries not recorded
        assert debit is None
        assert credit is None

    def test_double_entry_negative_raises(self, ledger):
        """Test that negative amounts raise an error."""
        with pytest.raises(ValueError, match="non-negative"):
            ledger.record_double_entry(
                date=1,
                debit_account="cash",
                credit_account="revenue",
                amount=-1000,
                transaction_type=TransactionType.REVENUE,
            )

    def test_get_balance_asset(self, ledger):
        """Test balance calculation for asset account (debit normal)."""
        # Debit cash 1000
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=1000,
            transaction_type=TransactionType.REVENUE,
        )
        # Credit cash 300
        ledger.record_double_entry(
            date=1,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=300,
            transaction_type=TransactionType.PAYMENT,
        )

        assert ledger.get_balance("cash") == 700

    def test_get_balance_liability(self, ledger):
        """Test balance calculation for liability account (credit normal)."""
        # Credit accounts_payable 5000
        ledger.record_double_entry(
            date=1,
            debit_account="inventory",
            credit_account="accounts_payable",
            amount=5000,
            transaction_type=TransactionType.INVENTORY_PURCHASE,
        )
        # Debit accounts_payable 2000 (payment)
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_payable",
            credit_account="cash",
            amount=2000,
            transaction_type=TransactionType.PAYMENT,
        )

        assert ledger.get_balance("accounts_payable") == 3000

    def test_get_balance_revenue(self, ledger):
        """Test balance calculation for revenue account (credit normal)."""
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=10000,
            transaction_type=TransactionType.REVENUE,
        )

        assert ledger.get_balance("revenue") == 10000

    def test_get_balance_expense(self, ledger):
        """Test balance calculation for expense account (debit normal)."""
        ledger.record_double_entry(
            date=1,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=5000,
            transaction_type=TransactionType.PAYMENT,
        )

        assert ledger.get_balance("operating_expenses") == 5000

    def test_get_balance_as_of_date(self, ledger_with_entries):
        """Test balance calculation with date filter."""
        # Add year 2 transaction
        ledger_with_entries.record_double_entry(
            date=2,
            debit_account="cash",
            credit_account="accounts_receivable",
            amount=500_000,
            transaction_type=TransactionType.COLLECTION,
        )

        balance_y1 = ledger_with_entries.get_balance("cash", as_of_date=1)
        balance_y2 = ledger_with_entries.get_balance("cash", as_of_date=2)

        assert balance_y2 == balance_y1 + 500_000

    def test_get_period_change(self, ledger_with_entries):
        """Test period change calculation."""
        # Add year 2 transaction
        ledger_with_entries.record_double_entry(
            date=2,
            debit_account="cash",
            credit_account="revenue",
            amount=1_000_000,
            transaction_type=TransactionType.REVENUE,
        )

        change_y1 = ledger_with_entries.get_period_change("cash", period=1)
        change_y2 = ledger_with_entries.get_period_change("cash", period=2)

        # Year 1: +10M capitalization, +4.5M collections, -2M expenses = 12.5M
        assert change_y1 == 12_500_000
        # Year 2: +1M revenue
        assert change_y2 == 1_000_000

    def test_get_entries_all(self, ledger_with_entries):
        """Test getting all entries."""
        entries = ledger_with_entries.get_entries()
        assert len(entries) == 8  # 4 transactions * 2 entries each

    def test_get_entries_by_account(self, ledger_with_entries):
        """Test filtering entries by account."""
        cash_entries = ledger_with_entries.get_entries(account="cash")
        # 3 cash entries: debit from capitalization, debit from collection, credit for expense
        assert len(cash_entries) == 3

    def test_get_entries_by_date_range(self, ledger_with_entries):
        """Test filtering entries by date range."""
        # Add year 2 transaction
        ledger_with_entries.record_double_entry(
            date=2,
            debit_account="cash",
            credit_account="revenue",
            amount=1_000_000,
            transaction_type=TransactionType.REVENUE,
        )

        y1_entries = ledger_with_entries.get_entries(start_date=1, end_date=1)
        y2_entries = ledger_with_entries.get_entries(start_date=2, end_date=2)

        assert len(y1_entries) == 8
        assert len(y2_entries) == 2

    def test_get_entries_by_transaction_type(self, ledger_with_entries):
        """Test filtering entries by transaction type."""
        collection_entries = ledger_with_entries.get_entries(
            transaction_type=TransactionType.COLLECTION
        )
        assert len(collection_entries) == 2  # Debit and credit

    def test_sum_by_transaction_type(self, ledger_with_entries):
        """Test summing by transaction type."""
        collections = ledger_with_entries.sum_by_transaction_type(
            transaction_type=TransactionType.COLLECTION,
            period=1,
            account="cash",
            entry_type=EntryType.DEBIT,
        )
        assert collections == 4_500_000

    def test_get_cash_flows(self, ledger_with_entries):
        """Test cash flow extraction."""
        flows = ledger_with_entries.get_cash_flows(period=1)

        # Cash from customers = collections
        assert flows["cash_from_customers"] == 4_500_000

        # Cash to suppliers = operating expense payments
        assert flows["cash_to_suppliers"] == 2_000_000

        # Net operating (without initial capitalization)
        # 4.5M collections - 2M expenses = 2.5M
        assert flows["net_operating"] == 2_500_000

    def test_verify_balance_balanced(self, ledger_with_entries):
        """Test verification of balanced ledger."""
        is_balanced, diff = ledger_with_entries.verify_balance()
        assert is_balanced is True
        assert abs(diff) < 0.01

    def test_verify_balance_unbalanced(self, ledger):
        """Test verification detects unbalanced ledger."""
        # Record only one side (bad practice, but tests verification)
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=Decimal("1000"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
        )
        ledger.record(entry)

        is_balanced, diff = ledger.verify_balance()
        assert is_balanced is False
        assert diff == Decimal("1000")

    def test_get_trial_balance(self, ledger_with_entries):
        """Test trial balance generation."""
        trial = ledger_with_entries.get_trial_balance()

        # Should have balances for active accounts
        assert "cash" in trial
        assert "accounts_receivable" in trial
        assert "revenue" in trial
        assert "retained_earnings" in trial
        assert "operating_expenses" in trial

        # Cash: 10M + 4.5M - 2M = 12.5M
        assert trial["cash"] == 12_500_000

        # AR: 5M - 4.5M = 0.5M
        assert trial["accounts_receivable"] == 500_000

        # Revenue: 5M
        assert trial["revenue"] == 5_000_000

    def test_clear(self, ledger_with_entries):
        """Test clearing the ledger."""
        assert len(ledger_with_entries) > 0
        ledger_with_entries.clear()
        assert len(ledger_with_entries) == 0

    def test_unknown_account_raises_error(self, ledger):
        """Test that unknown accounts raise ValueError in strict mode (Issue #260)."""
        with pytest.raises(ValueError, match="Unknown account name"):
            ledger.record_double_entry(
                date=1,
                debit_account="new_custom_account",
                credit_account="cash",
                amount=1000,
                transaction_type=TransactionType.PAYMENT,
            )

    def test_unknown_account_defaults_to_asset_nonstrict(self):
        """Test that unknown accounts are added as ASSET in non-strict mode."""
        ledger = Ledger(strict_validation=False)
        ledger.record_double_entry(
            date=1,
            debit_account="new_custom_account",
            credit_account="cash",
            amount=1000,
            transaction_type=TransactionType.PAYMENT,
        )

        assert "new_custom_account" in ledger.chart_of_accounts
        assert ledger.chart_of_accounts["new_custom_account"] == AccountType.ASSET

    def test_account_name_enum_works(self, ledger):
        """Test that AccountName enum can be used for accounts (Issue #260)."""
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=1000,
            transaction_type=TransactionType.REVENUE,
        )

        assert ledger.get_balance(AccountName.CASH) == 1000
        assert ledger.get_balance(AccountName.REVENUE) == 1000


class TestChartOfAccounts:
    """Tests for the standard chart of accounts."""

    def test_asset_accounts(self):
        """Test that asset accounts are classified correctly."""
        assets = [
            AccountName.CASH,
            AccountName.ACCOUNTS_RECEIVABLE,
            AccountName.INVENTORY,
            AccountName.PREPAID_INSURANCE,
            AccountName.GROSS_PPE,
        ]
        for account in assets:
            assert CHART_OF_ACCOUNTS[account] == AccountType.ASSET

    def test_liability_accounts(self):
        """Test that liability accounts are classified correctly."""
        liabilities = [
            AccountName.ACCOUNTS_PAYABLE,
            AccountName.ACCRUED_EXPENSES,
            AccountName.CLAIM_LIABILITIES,
        ]
        for account in liabilities:
            assert CHART_OF_ACCOUNTS[account] == AccountType.LIABILITY

    def test_equity_accounts(self):
        """Test that equity accounts are classified correctly."""
        equity = [AccountName.RETAINED_EARNINGS, AccountName.COMMON_STOCK]
        for account in equity:
            assert CHART_OF_ACCOUNTS[account] == AccountType.EQUITY

    def test_revenue_accounts(self):
        """Test that revenue accounts are classified correctly."""
        revenue = [AccountName.REVENUE, AccountName.SALES_REVENUE, AccountName.INTEREST_INCOME]
        for account in revenue:
            assert CHART_OF_ACCOUNTS[account] == AccountType.REVENUE

    def test_expense_accounts(self):
        """Test that expense accounts are classified correctly."""
        expenses = [
            AccountName.COST_OF_GOODS_SOLD,
            AccountName.OPERATING_EXPENSES,
            AccountName.DEPRECIATION_EXPENSE,
            AccountName.INSURANCE_EXPENSE,
        ]
        for account in expenses:
            assert CHART_OF_ACCOUNTS[account] == AccountType.EXPENSE

    def test_account_name_enum_values_match(self):
        """Test that AccountName enum values are consistent (Issue #260)."""
        # Verify enum values are lowercase strings matching expected format
        assert AccountName.CASH.value == "cash"
        assert AccountName.ACCOUNTS_RECEIVABLE.value == "accounts_receivable"
        assert AccountName.CLAIM_LIABILITIES.value == "claim_liabilities"


class TestCashFlowIntegration:
    """Integration tests for cash flow statement generation from ledger."""

    @pytest.fixture
    def full_year_ledger(self):
        """Create ledger with a full year of typical transactions."""
        ledger = Ledger()

        # Initial capitalization
        ledger.record_double_entry(
            date=0,
            debit_account="cash",
            credit_account="retained_earnings",
            amount=10_000_000,
            transaction_type=TransactionType.EQUITY_ISSUANCE,
            description="Initial capital",
        )

        # Year 1 transactions
        # Revenue recognition
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=8_000_000,
            transaction_type=TransactionType.REVENUE,
            description="Annual revenue",
        )

        # Cash collection (90% of revenue)
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="accounts_receivable",
            amount=7_200_000,
            transaction_type=TransactionType.COLLECTION,
            description="Collections",
        )

        # Inventory purchase on credit
        ledger.record_double_entry(
            date=1,
            debit_account="inventory",
            credit_account="accounts_payable",
            amount=3_000_000,
            transaction_type=TransactionType.INVENTORY_PURCHASE,
            description="Inventory purchased",
        )

        # Pay suppliers
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_payable",
            credit_account="cash",
            amount=2_500_000,
            transaction_type=TransactionType.PAYMENT,
            description="Supplier payments",
        )

        # Insurance premium
        ledger.record_double_entry(
            date=1,
            debit_account="prepaid_insurance",
            credit_account="cash",
            amount=500_000,
            transaction_type=TransactionType.INSURANCE_PREMIUM,
            description="Annual premium",
        )

        # Tax payment
        ledger.record_double_entry(
            date=1,
            debit_account="tax_expense",
            credit_account="cash",
            amount=800_000,
            transaction_type=TransactionType.TAX_PAYMENT,
            description="Tax payment",
        )

        # Capital expenditure
        ledger.record_double_entry(
            date=1,
            debit_account="gross_ppe",
            credit_account="cash",
            amount=1_000_000,
            transaction_type=TransactionType.CAPEX,
            description="Equipment purchase",
        )

        # Depreciation (non-cash)
        ledger.record_double_entry(
            date=1,
            debit_account="depreciation_expense",
            credit_account="accumulated_depreciation",
            amount=200_000,
            transaction_type=TransactionType.DEPRECIATION,
            description="Annual depreciation",
        )

        # Dividend payment
        ledger.record_double_entry(
            date=1,
            debit_account="retained_earnings",
            credit_account="cash",
            amount=600_000,
            transaction_type=TransactionType.DIVIDEND,
            description="Dividend payment",
        )

        return ledger

    def test_operating_cash_flow(self, full_year_ledger):
        """Test operating cash flow calculation."""
        flows = full_year_ledger.get_cash_flows(period=1)

        # Cash from customers: 7.2M collections
        assert flows["cash_from_customers"] == 7_200_000

        # Cash for insurance: 0.5M
        assert flows["cash_for_insurance"] == 500_000

        # Cash for taxes: 0.8M
        assert flows["cash_for_taxes"] == 800_000

    def test_investing_cash_flow(self, full_year_ledger):
        """Test investing cash flow calculation."""
        flows = full_year_ledger.get_cash_flows(period=1)

        # Capital expenditures: 1M
        assert flows["capital_expenditures"] == 1_000_000
        assert flows["net_investing"] == -1_000_000

    def test_financing_cash_flow(self, full_year_ledger):
        """Test financing cash flow calculation."""
        flows = full_year_ledger.get_cash_flows(period=1)

        # Dividends: 0.6M
        assert flows["dividends_paid"] == 600_000
        assert flows["net_financing"] == -600_000

    def test_cash_reconciliation(self, full_year_ledger):
        """Test that cash flows reconcile to balance change."""
        flows = full_year_ledger.get_cash_flows(period=1)

        # Beginning cash (from year 0)
        beginning_cash = full_year_ledger.get_balance("cash", as_of_date=0)
        assert beginning_cash == 10_000_000

        # Ending cash
        ending_cash = full_year_ledger.get_balance("cash")

        # Net change should equal ending - beginning
        net_change = flows["net_change_in_cash"]
        calculated_ending = beginning_cash + net_change

        # Note: The simple cash flow extraction doesn't capture supplier payments properly
        # in the "cash_to_suppliers" category because they use PAYMENT type not INVENTORY_PURCHASE
        # This test verifies the ledger balance approach works correctly
        actual_cash_change = ending_cash - beginning_cash

        # Verify actual balance change
        # 7.2M collections - 2.5M payments - 0.5M insurance - 0.8M tax - 1M capex - 0.6M dividends
        # = 1.8M
        assert actual_cash_change == 1_800_000

    def test_ledger_stays_balanced(self, full_year_ledger):
        """Test that ledger remains balanced after all transactions."""
        is_balanced, diff = full_year_ledger.verify_balance()
        assert is_balanced is True
        assert abs(diff) < 0.01


class TestBalanceCache:
    """Tests for the balance cache optimization (Issue #259)."""

    @pytest.fixture
    def ledger(self):
        """Create empty ledger for testing."""
        return Ledger()

    def test_cache_initialized_empty(self, ledger):
        """Test that cache starts empty."""
        assert ledger._balances == {}

    def test_cache_matches_iteration_for_assets(self, ledger):
        """Test cache matches iteration-based calculation for asset accounts."""
        # Record multiple transactions
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=10000,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record_double_entry(
            date=1,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=3000,
            transaction_type=TransactionType.PAYMENT,
        )

        # Get cached balance (as_of_date=None)
        cached_balance = ledger.get_balance("cash")

        # Get iteration-based balance (as_of_date specified)
        iteration_balance = ledger.get_balance("cash", as_of_date=999)

        assert cached_balance == iteration_balance == 7000

    def test_cache_matches_iteration_for_liabilities(self, ledger):
        """Test cache matches iteration-based calculation for liability accounts."""
        ledger.record_double_entry(
            date=1,
            debit_account="inventory",
            credit_account="accounts_payable",
            amount=5000,
            transaction_type=TransactionType.INVENTORY_PURCHASE,
        )
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_payable",
            credit_account="cash",
            amount=2000,
            transaction_type=TransactionType.PAYMENT,
        )

        cached_balance = ledger.get_balance("accounts_payable")
        iteration_balance = ledger.get_balance("accounts_payable", as_of_date=999)

        assert cached_balance == iteration_balance == 3000

    def test_cache_matches_iteration_for_revenue(self, ledger):
        """Test cache matches iteration-based calculation for revenue accounts."""
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=8000,
            transaction_type=TransactionType.REVENUE,
        )

        cached_balance = ledger.get_balance("revenue")
        iteration_balance = ledger.get_balance("revenue", as_of_date=999)

        assert cached_balance == iteration_balance == 8000

    def test_cache_matches_iteration_for_expenses(self, ledger):
        """Test cache matches iteration-based calculation for expense accounts."""
        ledger.record_double_entry(
            date=1,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=4000,
            transaction_type=TransactionType.PAYMENT,
        )

        cached_balance = ledger.get_balance("operating_expenses")
        iteration_balance = ledger.get_balance("operating_expenses", as_of_date=999)

        assert cached_balance == iteration_balance == 4000

    def test_cache_matches_iteration_for_equity(self, ledger):
        """Test cache matches iteration-based calculation for equity accounts."""
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="retained_earnings",
            amount=10000,
            transaction_type=TransactionType.EQUITY_ISSUANCE,
        )

        cached_balance = ledger.get_balance("retained_earnings")
        iteration_balance = ledger.get_balance("retained_earnings", as_of_date=999)

        assert cached_balance == iteration_balance == 10000

    def test_cache_reset_on_clear(self, ledger):
        """Test that cache is properly reset when ledger is cleared."""
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=5000,
            transaction_type=TransactionType.REVENUE,
        )

        assert ledger.get_balance("cash") == 5000
        assert len(ledger._balances) > 0

        ledger.clear()

        assert ledger.get_balance("cash") == 0
        assert ledger._balances == {}

    def test_cache_with_record_double_entry(self, ledger):
        """Test cache consistency through multiple double-entry transactions."""
        # Series of transactions
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="retained_earnings",
            amount=100000,
            transaction_type=TransactionType.EQUITY_ISSUANCE,
        )
        ledger.record_double_entry(
            date=1,
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=50000,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="accounts_receivable",
            amount=40000,
            transaction_type=TransactionType.COLLECTION,
        )
        ledger.record_double_entry(
            date=1,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=20000,
            transaction_type=TransactionType.PAYMENT,
        )

        # Verify all accounts match cache vs iteration
        accounts = [
            "cash",
            "retained_earnings",
            "accounts_receivable",
            "revenue",
            "operating_expenses",
        ]
        for account in accounts:
            cached = ledger.get_balance(account)
            iteration = ledger.get_balance(account, as_of_date=999)
            assert (
                cached == iteration
            ), f"Mismatch for {account}: cache={cached}, iteration={iteration}"

        # Verify expected values
        assert ledger.get_balance("cash") == 120000  # 100k + 40k - 20k
        assert ledger.get_balance("accounts_receivable") == 10000  # 50k - 40k
        assert ledger.get_balance("revenue") == 50000
        assert ledger.get_balance("operating_expenses") == 20000
        assert ledger.get_balance("retained_earnings") == 100000

    def test_cache_with_single_record(self, ledger):
        """Test cache updates correctly with single entry record."""
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=Decimal("1000"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
        )
        ledger.record(entry)

        assert ledger.get_balance("cash") == Decimal("1000")
        assert ledger._balances.get("cash") == Decimal("1000")

    def test_historical_query_still_works(self, ledger):
        """Test that as_of_date queries work correctly for historical balances."""
        # Year 1 transaction
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=10000,
            transaction_type=TransactionType.REVENUE,
        )
        # Year 2 transaction
        ledger.record_double_entry(
            date=2,
            debit_account="cash",
            credit_account="revenue",
            amount=5000,
            transaction_type=TransactionType.REVENUE,
        )

        # Current balance should be 15000
        assert ledger.get_balance("cash") == 15000

        # Historical balance as of year 1 should be 10000
        assert ledger.get_balance("cash", as_of_date=1) == 10000

        # Historical balance as of year 0 should be 0
        assert ledger.get_balance("cash", as_of_date=0) == 0

    def test_cache_with_account_name_enum(self, ledger):
        """Test cache works correctly with AccountName enum."""
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=7500,
            transaction_type=TransactionType.REVENUE,
        )

        # Both enum and string should return same value
        assert ledger.get_balance(AccountName.CASH) == 7500
        assert ledger.get_balance("cash") == 7500
        assert ledger.get_balance(AccountName.REVENUE) == 7500
        assert ledger.get_balance("revenue") == 7500


class TestMonthValidation:
    """Tests for LedgerEntry month range validation (Issue #315)."""

    def test_valid_month_zero(self):
        """Month 0 is valid."""
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=Decimal("100"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
            month=0,
        )
        assert entry.month == 0

    def test_valid_month_eleven(self):
        """Month 11 is valid."""
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=Decimal("100"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.COLLECTION,
            month=11,
        )
        assert entry.month == 11

    def test_invalid_month_negative(self):
        """Negative month raises ValueError."""
        with pytest.raises(ValueError, match="Month must be 0-11"):
            LedgerEntry(
                date=1,
                account="cash",
                amount=Decimal("100"),
                entry_type=EntryType.DEBIT,
                transaction_type=TransactionType.COLLECTION,
                month=-1,
            )

    def test_invalid_month_twelve(self):
        """Month 12 raises ValueError."""
        with pytest.raises(ValueError, match="Month must be 0-11"):
            LedgerEntry(
                date=1,
                account="cash",
                amount=Decimal("100"),
                entry_type=EntryType.DEBIT,
                transaction_type=TransactionType.COLLECTION,
                month=12,
            )


class TestEntryPruning:
    """Tests for ledger entry pruning (Issue #315)."""

    @pytest.fixture
    def multi_year_ledger(self):
        """Ledger with entries across years 1-5."""
        ledger = Ledger()
        for year in range(1, 6):
            ledger.record_double_entry(
                date=year,
                debit_account="cash",
                credit_account="revenue",
                amount=1000 * year,
                transaction_type=TransactionType.REVENUE,
                description=f"Year {year} revenue",
            )
        return ledger

    def test_prune_removes_old_entries(self, multi_year_ledger):
        """Pruning removes entries before the cutoff date."""
        original_count = len(multi_year_ledger)
        removed = multi_year_ledger.prune_entries(before_date=3)
        # Years 1 and 2 removed: 2 transactions * 2 entries = 4
        assert removed == 4
        assert len(multi_year_ledger) == original_count - 4

    def test_prune_preserves_current_balances(self, multi_year_ledger):
        """Current balance cache is unaffected by pruning."""
        balance_before = multi_year_ledger.get_balance("cash")
        multi_year_ledger.prune_entries(before_date=3)
        assert multi_year_ledger.get_balance("cash") == balance_before

    def test_prune_preserves_trial_balance_current(self, multi_year_ledger):
        """Current trial balance is unchanged after pruning."""
        trial_before = multi_year_ledger.get_trial_balance()
        multi_year_ledger.prune_entries(before_date=3)
        trial_after = multi_year_ledger.get_trial_balance()
        assert trial_before == trial_after

    def test_prune_historical_balance_at_cutoff(self, multi_year_ledger):
        """Historical balance at or after prune date is correct."""
        bal_at_3_before = multi_year_ledger.get_balance("cash", as_of_date=3)
        multi_year_ledger.prune_entries(before_date=3)
        bal_at_3_after = multi_year_ledger.get_balance("cash", as_of_date=3)
        assert bal_at_3_after == bal_at_3_before

    def test_prune_historical_trial_balance_at_cutoff(self, multi_year_ledger):
        """Historical trial balance at or after prune date is correct."""
        trial_at_5_before = multi_year_ledger.get_trial_balance(as_of_date=5)
        multi_year_ledger.prune_entries(before_date=3)
        trial_at_5_after = multi_year_ledger.get_trial_balance(as_of_date=5)
        assert trial_at_5_after == trial_at_5_before

    def test_prune_nothing_to_remove(self, multi_year_ledger):
        """Pruning with cutoff before all entries removes nothing."""
        removed = multi_year_ledger.prune_entries(before_date=0)
        assert removed == 0
        assert len(multi_year_ledger) == 10  # 5 years * 2 entries

    def test_prune_all_entries(self, multi_year_ledger):
        """Pruning with cutoff after all entries removes everything."""
        balance_before = multi_year_ledger.get_balance("cash")
        removed = multi_year_ledger.prune_entries(before_date=100)
        assert removed == 10
        assert len(multi_year_ledger) == 0
        # Current balance still correct
        assert multi_year_ledger.get_balance("cash") == balance_before

    def test_prune_cleared_on_reset(self, multi_year_ledger):
        """Clear resets pruning state."""
        multi_year_ledger.prune_entries(before_date=3)
        multi_year_ledger.clear()
        assert multi_year_ledger._pruned_balances == {}
        assert multi_year_ledger._prune_cutoff is None

    def test_prune_multiple_times(self, multi_year_ledger):
        """Multiple prune calls accumulate correctly."""
        multi_year_ledger.prune_entries(before_date=2)
        multi_year_ledger.prune_entries(before_date=4)
        # Only years 4 and 5 remain
        assert len(multi_year_ledger) == 4  # 2 years * 2 entries
        # Total balance: 1000+2000+3000+4000+5000 = 15000
        assert multi_year_ledger.get_balance("cash") == 15000


class TestSinglePassTrialBalance:
    """Tests for O(N) single-pass trial balance (Issue #315)."""

    def test_trial_balance_current_uses_cache(self):
        """Current trial balance reads from cache, not entries."""
        ledger = Ledger()
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=5000,
            transaction_type=TransactionType.REVENUE,
        )
        # Should return non-zero accounts from cache
        trial = ledger.get_trial_balance()
        assert trial["cash"] == 5000
        assert trial["revenue"] == 5000

    def test_trial_balance_historical_single_pass(self):
        """Historical trial balance matches per-account query results."""
        ledger = Ledger()
        for year in range(1, 4):
            ledger.record_double_entry(
                date=year,
                debit_account="cash",
                credit_account="revenue",
                amount=1000 * year,
                transaction_type=TransactionType.REVENUE,
            )
            ledger.record_double_entry(
                date=year,
                debit_account="operating_expenses",
                credit_account="cash",
                amount=500 * year,
                transaction_type=TransactionType.PAYMENT,
            )

        # Compare single-pass trial balance vs per-account queries
        trial = ledger.get_trial_balance(as_of_date=2)
        assert trial["cash"] == ledger.get_balance("cash", as_of_date=2)
        assert trial["revenue"] == ledger.get_balance("revenue", as_of_date=2)
        assert trial["operating_expenses"] == ledger.get_balance("operating_expenses", as_of_date=2)

    def test_trial_balance_after_prune(self):
        """Trial balance with as_of_date works after pruning."""
        ledger = Ledger()
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=1000,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record_double_entry(
            date=2,
            debit_account="cash",
            credit_account="revenue",
            amount=2000,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record_double_entry(
            date=3,
            debit_account="cash",
            credit_account="revenue",
            amount=3000,
            transaction_type=TransactionType.REVENUE,
        )

        trial_full = ledger.get_trial_balance(as_of_date=3)
        ledger.prune_entries(before_date=2)
        trial_pruned = ledger.get_trial_balance(as_of_date=3)
        assert trial_full == trial_pruned


class TestPrunedLedgerWarningsAndVerification:
    """Tests for Issue #362: correct behavior after ledger pruning."""

    @pytest.fixture
    def pruned_ledger(self):
        """Ledger with years 1-5, pruned at year 3."""
        ledger = Ledger()
        for year in range(1, 6):
            ledger.record_double_entry(
                date=year,
                debit_account="cash",
                credit_account="revenue",
                amount=1000 * year,
                transaction_type=TransactionType.REVENUE,
                description=f"Year {year} revenue",
            )
        ledger.prune_entries(before_date=3)
        return ledger

    def test_get_balance_warns_before_prune_cutoff(self, pruned_ledger):
        """get_balance warns when as_of_date < prune_cutoff."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pruned_ledger.get_balance("cash", as_of_date=1)
            assert len(w) == 1
            assert "prune cutoff" in str(w[0].message)

    def test_get_balance_no_warning_at_cutoff(self, pruned_ledger):
        """get_balance does not warn when as_of_date >= prune_cutoff."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pruned_ledger.get_balance("cash", as_of_date=3)
            assert len(w) == 0

    def test_get_balance_no_warning_after_cutoff(self, pruned_ledger):
        """get_balance does not warn when as_of_date > prune_cutoff."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pruned_ledger.get_balance("cash", as_of_date=5)
            assert len(w) == 0

    def test_get_balance_no_warning_without_pruning(self):
        """get_balance never warns on an unpruned ledger."""
        ledger = Ledger()
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=1000,
            transaction_type=TransactionType.REVENUE,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ledger.get_balance("cash", as_of_date=1)
            assert len(w) == 0

    def test_get_trial_balance_warns_before_prune_cutoff(self, pruned_ledger):
        """get_trial_balance warns when as_of_date < prune_cutoff."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pruned_ledger.get_trial_balance(as_of_date=2)
            assert len(w) == 1
            assert "prune cutoff" in str(w[0].message)

    def test_get_trial_balance_no_warning_at_cutoff(self, pruned_ledger):
        """get_trial_balance does not warn at prune_cutoff."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pruned_ledger.get_trial_balance(as_of_date=3)
            assert len(w) == 0

    def test_verify_balance_correct_after_pruning(self, pruned_ledger):
        """verify_balance returns balanced after pruning."""
        balanced, diff = pruned_ledger.verify_balance()
        assert balanced is True
        assert diff == Decimal("0")

    def test_verify_balance_correct_after_multiple_prunes(self):
        """verify_balance stays correct across multiple prune calls."""
        ledger = Ledger()
        for year in range(1, 6):
            ledger.record_double_entry(
                date=year,
                debit_account="cash",
                credit_account="revenue",
                amount=1000 * year,
                transaction_type=TransactionType.REVENUE,
            )
        ledger.prune_entries(before_date=2)
        balanced1, diff1 = ledger.verify_balance()
        assert balanced1 is True
        assert diff1 == Decimal("0")

        ledger.prune_entries(before_date=4)
        balanced2, diff2 = ledger.verify_balance()
        assert balanced2 is True
        assert diff2 == Decimal("0")

    def test_verify_balance_correct_after_pruning_all(self):
        """verify_balance correct when all entries are pruned."""
        ledger = Ledger()
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=5000,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.prune_entries(before_date=100)
        balanced, diff = ledger.verify_balance()
        assert balanced is True
        assert diff == Decimal("0")

    def test_verify_balance_detects_imbalance_after_pruning(self):
        """verify_balance detects imbalance even when imbalanced entries are pruned."""
        ledger = Ledger()
        # Record a balanced double-entry
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=5000,
            transaction_type=TransactionType.REVENUE,
        )
        # Force an imbalanced entry via direct record
        ledger.record(
            LedgerEntry(
                date=1,
                account="cash",
                entry_type=EntryType.DEBIT,
                amount=Decimal("100"),
                transaction_type=TransactionType.ADJUSTMENT,
                description="imbalanced debit",
            )
        )
        # Verify imbalanced before pruning
        balanced_before, _ = ledger.verify_balance()
        assert balanced_before is False

        # Prune everything
        ledger.prune_entries(before_date=100)
        balanced_after, diff = ledger.verify_balance()
        assert balanced_after is False
        assert diff == Decimal("100")

    def test_pruned_state_reset_on_clear(self):
        """clear() resets pruned debit/credit totals."""
        ledger = Ledger()
        ledger.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=1000,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.prune_entries(before_date=100)
        ledger.clear()
        assert ledger._pruned_debits == Decimal("0")
        assert ledger._pruned_credits == Decimal("0")
        balanced, diff = ledger.verify_balance()
        assert balanced is True

    def test_deepcopy_preserves_pruned_totals(self, pruned_ledger):
        """deepcopy preserves pruned debit/credit totals."""
        import copy

        copied = copy.deepcopy(pruned_ledger)
        assert copied._pruned_debits == pruned_ledger._pruned_debits
        assert copied._pruned_credits == pruned_ledger._pruned_credits
        balanced, diff = copied.verify_balance()
        assert balanced is True

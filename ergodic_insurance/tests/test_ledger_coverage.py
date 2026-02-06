"""Coverage tests for ledger.py targeting specific uncovered lines.

Missing lines: 236, 362, 451-460, 662, 671-674, 766, 958, 1003, 1008
"""

from decimal import Decimal

import pytest

from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import (
    AccountName,
    AccountType,
    EntryType,
    Ledger,
    LedgerEntry,
    TransactionType,
    _resolve_account_name,
)


class TestLedgerEntryNonDecimalAmount:
    """Tests for LedgerEntry.__post_init__ non-Decimal conversion (line 236)."""

    def test_float_amount_converted_to_decimal(self):
        """Line 236: Float amount is converted to Decimal on construction."""
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=1000.50,  # type: ignore[arg-type]
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.REVENUE,
        )
        assert isinstance(entry.amount, Decimal)

    def test_int_amount_converted_to_decimal(self):
        """Line 236: Integer amount is converted to Decimal on construction."""
        entry = LedgerEntry(
            date=1,
            account="cash",
            amount=5000,  # type: ignore[arg-type]
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.REVENUE,
        )
        assert isinstance(entry.amount, Decimal)

    def test_negative_amount_raises(self):
        """Negative amount raises ValueError after conversion."""
        with pytest.raises(ValueError, match="non-negative"):
            LedgerEntry(
                date=1,
                account="cash",
                amount=-100.0,  # type: ignore[arg-type]
                entry_type=EntryType.DEBIT,
                transaction_type=TransactionType.REVENUE,
            )


class TestResolveAccountNameUnknown:
    """Tests for _resolve_account_name unknown account with suggestions (line 362)."""

    def test_unknown_account_with_similar_suggestion(self):
        """Line 362: Unknown account raises ValueError with suggestions."""
        with pytest.raises(ValueError, match="Did you mean"):
            _resolve_account_name("cas")  # Similar to "cash"

    def test_unknown_account_no_match(self):
        """Unknown account with no similar names gives generic error."""
        with pytest.raises(ValueError, match="Unknown account name"):
            _resolve_account_name("zzz_nonexistent_account_xyz")

    def test_valid_enum_resolves(self):
        """AccountName enum resolves properly."""
        result = _resolve_account_name(AccountName.CASH)
        assert result == "cash"

    def test_valid_string_resolves(self):
        """Valid string account name resolves properly."""
        result = _resolve_account_name("cash")
        assert result == "cash"


class TestLedgerRecordStrictValidation:
    """Tests for Ledger.record strict validation with suggestions (lines 451-460)."""

    def test_strict_unknown_account_raises_with_suggestions(self):
        """Lines 451-460: Strict mode rejects unknown account with suggestions."""
        ledger = Ledger(strict_validation=True)
        entry = LedgerEntry(
            date=1,
            account="cas",  # Typo similar to "cash"
            amount=Decimal("1000"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.REVENUE,
        )
        with pytest.raises(ValueError, match="Did you mean"):
            ledger.record(entry)

    def test_strict_unknown_account_no_similar(self):
        """Strict mode rejects totally unknown account."""
        ledger = Ledger(strict_validation=True)
        entry = LedgerEntry(
            date=1,
            account="zzz_unknown_xyz",
            amount=Decimal("1000"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.REVENUE,
        )
        with pytest.raises(ValueError, match="Unknown account name"):
            ledger.record(entry)

    def test_non_strict_allows_unknown_account(self):
        """Non-strict mode adds unknown account as ASSET."""
        ledger = Ledger(strict_validation=False)
        entry = LedgerEntry(
            date=1,
            account="custom_account",
            amount=Decimal("1000"),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record(entry)
        assert "custom_account" in ledger.chart_of_accounts
        assert ledger.chart_of_accounts["custom_account"] == AccountType.ASSET


class TestGetPeriodChangeMonthFilter:
    """Tests for get_period_change month filter (line 662)."""

    def test_month_filter_applied(self):
        """Line 662: Month filter restricts to specific month."""
        ledger = Ledger(strict_validation=True)
        # Record two entries in same period, different months
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("1000"),
            transaction_type=TransactionType.REVENUE,
            description="Month 0 revenue",
            month=0,
        )
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("2000"),
            transaction_type=TransactionType.REVENUE,
            description="Month 3 revenue",
            month=3,
        )

        # Get change for specific month
        change_m0 = ledger.get_period_change(AccountName.CASH, period=1, month=0)
        assert change_m0 == Decimal("1000")

        change_m3 = ledger.get_period_change(AccountName.CASH, period=1, month=3)
        assert change_m3 == Decimal("2000")

        # Total without month filter should include both
        change_all = ledger.get_period_change(AccountName.CASH, period=1)
        assert change_all == Decimal("3000")


class TestGetPeriodChangeCreditNormal:
    """Tests for get_period_change credit-normal account handling (lines 671-674)."""

    def test_credit_normal_account_change(self):
        """Lines 671-674: Credit-normal account (LIABILITY) sign handling."""
        ledger = Ledger(strict_validation=True)

        # Record a credit entry to a liability account
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.ACCOUNTS_PAYABLE,
            amount=Decimal("5000"),
            transaction_type=TransactionType.EXPENSE,
            description="Increase accounts payable",
        )

        # Credit-normal: credit increases, debit decreases
        change = ledger.get_period_change(AccountName.ACCOUNTS_PAYABLE, period=1)
        assert change == Decimal("5000")

        # Now record a debit to accounts_payable (payment)
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.ACCOUNTS_PAYABLE,
            credit_account=AccountName.CASH,
            amount=Decimal("2000"),
            transaction_type=TransactionType.PAYMENT,
            description="Pay down accounts payable",
        )

        change = ledger.get_period_change(AccountName.ACCOUNTS_PAYABLE, period=1)
        # 5000 credit - 2000 debit = 3000
        assert change == Decimal("3000")


class TestSumByTransactionTypeEntryFilter:
    """Tests for sum_by_transaction_type entry_type filter (line 766)."""

    def test_entry_type_filter(self):
        """Line 766: entry_type parameter filters entries."""
        ledger = Ledger(strict_validation=True)

        # Record a double entry that creates both debit and credit entries
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("3000"),
            transaction_type=TransactionType.REVENUE,
            description="Revenue collection",
        )

        # Sum only debit entries for REVENUE transaction type
        debit_sum = ledger.sum_by_transaction_type(
            TransactionType.REVENUE, period=1, entry_type=EntryType.DEBIT
        )
        assert debit_sum == Decimal("3000")

        # Sum only credit entries for REVENUE transaction type
        credit_sum = ledger.sum_by_transaction_type(
            TransactionType.REVENUE, period=1, entry_type=EntryType.CREDIT
        )
        assert credit_sum == Decimal("3000")

    def test_entry_type_filter_with_account(self):
        """Filter by both entry_type and account."""
        ledger = Ledger(strict_validation=True)

        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("5000"),
            transaction_type=TransactionType.REVENUE,
        )

        # Sum debits for cash account only
        debit_cash = ledger.sum_by_transaction_type(
            TransactionType.REVENUE,
            period=1,
            account=AccountName.CASH,
            entry_type=EntryType.DEBIT,
        )
        assert debit_cash == Decimal("5000")

        # Sum credits for cash account (should be zero - credit went to revenue)
        credit_cash = ledger.sum_by_transaction_type(
            TransactionType.REVENUE,
            period=1,
            account=AccountName.CASH,
            entry_type=EntryType.CREDIT,
        )
        assert credit_cash == ZERO


class TestGetTrialBalanceCreditNormalDebit:
    """Tests for get_trial_balance credit-normal account with debit (line 958)."""

    def test_credit_normal_account_with_debit_entry(self):
        """Line 958: Credit-normal account accumulates debit as subtraction."""
        ledger = Ledger(strict_validation=True)

        # Create a liability account with credit entry
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.ACCOUNTS_PAYABLE,
            amount=Decimal("10000"),
            transaction_type=TransactionType.EXPENSE,
        )
        # Now add a debit entry to the liability (partial payment)
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.ACCOUNTS_PAYABLE,
            credit_account=AccountName.CASH,
            amount=Decimal("3000"),
            transaction_type=TransactionType.PAYMENT,
        )

        trial_balance = ledger.get_trial_balance(as_of_date=1)
        # Accounts payable: 10000 credit - 3000 debit = 7000
        assert trial_balance["accounts_payable"] == Decimal("7000")

    def test_credit_normal_equity_with_debit(self):
        """Credit-normal equity account with debit entry."""
        ledger = Ledger(strict_validation=True)

        # Equity credit
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=Decimal("50000"),
            transaction_type=TransactionType.ADJUSTMENT,
        )
        # Equity debit (e.g., dividend declaration)
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.RETAINED_EARNINGS,
            credit_account=AccountName.CASH,
            amount=Decimal("10000"),
            transaction_type=TransactionType.DIVIDEND,
        )

        trial_balance = ledger.get_trial_balance(as_of_date=1)
        assert trial_balance["retained_earnings"] == Decimal("40000")


class TestPruneEntriesCreditNormal:
    """Tests for prune_entries credit-normal snapshot accumulation (lines 1003, 1008)."""

    def test_prune_credit_normal_account(self):
        """Lines 1003, 1008: Pruning correctly snapshots credit-normal accounts."""
        ledger = Ledger(strict_validation=True)

        # Period 1: Liability created
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.ACCOUNTS_PAYABLE,
            amount=Decimal("8000"),
            transaction_type=TransactionType.EXPENSE,
        )
        # Period 1: Partial payment
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.ACCOUNTS_PAYABLE,
            credit_account=AccountName.CASH,
            amount=Decimal("3000"),
            transaction_type=TransactionType.PAYMENT,
        )
        # Period 2: More expenses
        ledger.record_double_entry(
            date=2,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.ACCOUNTS_PAYABLE,
            amount=Decimal("5000"),
            transaction_type=TransactionType.EXPENSE,
        )

        # Get balance before prune for verification
        balance_before = ledger.get_balance(AccountName.ACCOUNTS_PAYABLE, as_of_date=2)

        # Prune entries from period 1
        removed = ledger.prune_entries(before_date=2)
        assert removed > 0

        # Snapshot should preserve the accumulated balance from period 1
        assert "accounts_payable" in ledger._pruned_balances

        # Total balance should still be correct after pruning
        balance_after = ledger.get_balance(AccountName.ACCOUNTS_PAYABLE, as_of_date=2)
        assert balance_after == balance_before

    def test_prune_with_credit_entry_for_credit_normal(self):
        """Line 1008: Credit entries for credit-normal accounts accumulate positively."""
        ledger = Ledger(strict_validation=True)

        # Revenue is credit-normal
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("20000"),
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record_double_entry(
            date=2,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("15000"),
            transaction_type=TransactionType.REVENUE,
        )

        # Prune period 1
        removed = ledger.prune_entries(before_date=2)
        assert removed > 0

        # Revenue snapshot from period 1 should be positive (credit-normal, credit entry)
        revenue_snapshot = ledger._pruned_balances.get("revenue", ZERO)
        assert revenue_snapshot == Decimal("20000")

    def test_prune_preserves_later_entries(self):
        """Entries at or after prune date are kept."""
        ledger = Ledger(strict_validation=True)

        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("1000"),
            transaction_type=TransactionType.REVENUE,
        )
        ledger.record_double_entry(
            date=3,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=Decimal("2000"),
            transaction_type=TransactionType.REVENUE,
        )

        removed = ledger.prune_entries(before_date=2)
        assert removed == 2  # Two entries from date=1 (debit cash, credit revenue)
        # Remaining entries should be from date=3
        assert all(e.date == 3 for e in ledger.entries)

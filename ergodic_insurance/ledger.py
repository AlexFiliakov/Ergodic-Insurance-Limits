"""Event-sourcing ledger for financial transactions.

This module implements a simple ledger system that tracks individual financial
transactions using double-entry accounting. This provides transaction-level
detail that is lost when using only point-in-time metrics snapshots.

The ledger enables:
- Perfect reconciliation between financial statements
- Direct method cash flow statement generation
- Audit trail for all financial changes
- Understanding of WHY balances changed (e.g., "was this AR change a
  write-off or a payment?")

Example:
    Record a sale on credit::

        ledger = Ledger()
        ledger.record_double_entry(
            date=5,  # Year 5
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=1_000_000,
            description="Annual sales on credit"
        )

    Generate cash flows for a period::

        operating_cash_flows = ledger.get_cash_flows(period=5)
        print(f"Cash from customers: ${operating_cash_flows['cash_from_customers']:,.0f}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import uuid

from .decimal_utils import ZERO, is_zero, quantize_currency, to_decimal


class AccountType(Enum):
    """Classification of accounts per GAAP chart of accounts.

    Attributes:
        ASSET: Resources owned by the company (debit normal balance)
        LIABILITY: Obligations owed to others (credit normal balance)
        EQUITY: Owner's residual interest (credit normal balance)
        REVENUE: Income from operations (credit normal balance)
        EXPENSE: Costs of operations (debit normal balance)
    """

    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"


class EntryType(Enum):
    """Type of ledger entry - debit or credit.

    In double-entry accounting:
    - DEBIT increases assets and expenses, decreases liabilities and equity
    - CREDIT decreases assets and expenses, increases liabilities and equity
    """

    DEBIT = "debit"
    CREDIT = "credit"


class TransactionType(Enum):
    """Classification of transaction for cash flow statement mapping.

    These types enable automatic classification into operating, investing,
    or financing activities for cash flow statement generation.
    """

    # Operating Activities
    REVENUE = "revenue"
    COLLECTION = "collection"  # Cash collection from AR
    EXPENSE = "expense"
    PAYMENT = "payment"  # Cash payment for expenses/AP
    INVENTORY_PURCHASE = "inventory_purchase"
    INVENTORY_SALE = "inventory_sale"  # COGS recognition
    INSURANCE_PREMIUM = "insurance_premium"
    INSURANCE_CLAIM = "insurance_claim"
    TAX_ACCRUAL = "tax_accrual"
    TAX_PAYMENT = "tax_payment"
    DEPRECIATION = "depreciation"
    WORKING_CAPITAL = "working_capital"

    # Investing Activities
    CAPEX = "capex"  # Capital expenditure
    ASSET_SALE = "asset_sale"

    # Financing Activities
    DIVIDEND = "dividend"
    EQUITY_ISSUANCE = "equity_issuance"
    DEBT_ISSUANCE = "debt_issuance"
    DEBT_REPAYMENT = "debt_repayment"

    # Non-cash
    ADJUSTMENT = "adjustment"
    ACCRUAL = "accrual"


@dataclass
class LedgerEntry:
    """A single entry in the accounting ledger.

    Each entry represents one side of a double-entry transaction.
    A complete transaction always has matching debits and credits.

    Attributes:
        date: Period (year) when the transaction occurred
        account: Name of the account affected (e.g., "cash", "accounts_receivable")
        amount: Dollar amount of the entry (always positive)
        entry_type: DEBIT or CREDIT
        transaction_type: Classification for cash flow mapping
        description: Human-readable description of the transaction
        reference_id: UUID linking both sides of a double-entry transaction
        timestamp: Actual datetime when entry was recorded (for audit)
        month: Optional month within the year (0-11)
    """

    date: int  # Year/period
    account: str
    amount: Decimal
    entry_type: EntryType
    transaction_type: TransactionType
    description: str = ""
    reference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    month: int = 0  # Month within year (0-11)

    def __post_init__(self) -> None:
        """Validate entry after initialization."""
        # Convert amount to Decimal if not already (runtime check for backwards compatibility)
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", to_decimal(self.amount))  # type: ignore[unreachable]
        if self.amount < ZERO:
            raise ValueError(f"Ledger entry amount must be non-negative, got {self.amount}")

    @property
    def signed_amount(self) -> Decimal:
        """Return amount with sign based on entry type.

        For balance calculations:
        - Assets/Expenses: Debit positive, Credit negative
        - Liabilities/Equity/Revenue: Credit positive, Debit negative

        This method returns the raw signed amount for debits (+) and credits (-).
        The Ledger class handles account type normalization.
        """
        if self.entry_type == EntryType.DEBIT:
            return self.amount
        else:
            return -self.amount


# Standard chart of accounts with their types
CHART_OF_ACCOUNTS: Dict[str, AccountType] = {
    # Assets (debit normal balance)
    "cash": AccountType.ASSET,
    "accounts_receivable": AccountType.ASSET,
    "inventory": AccountType.ASSET,
    "prepaid_insurance": AccountType.ASSET,
    "insurance_receivables": AccountType.ASSET,
    "gross_ppe": AccountType.ASSET,
    "accumulated_depreciation": AccountType.ASSET,  # Contra-asset
    "restricted_cash": AccountType.ASSET,
    "collateral": AccountType.ASSET,
    # Liabilities (credit normal balance)
    "accounts_payable": AccountType.LIABILITY,
    "accrued_expenses": AccountType.LIABILITY,
    "accrued_wages": AccountType.LIABILITY,
    "accrued_taxes": AccountType.LIABILITY,
    "accrued_interest": AccountType.LIABILITY,
    "claim_liabilities": AccountType.LIABILITY,
    "unearned_revenue": AccountType.LIABILITY,
    # Equity (credit normal balance)
    "retained_earnings": AccountType.EQUITY,
    "common_stock": AccountType.EQUITY,
    "dividends": AccountType.EQUITY,  # Contra-equity
    # Revenue (credit normal balance)
    "revenue": AccountType.REVENUE,
    "sales_revenue": AccountType.REVENUE,
    "interest_income": AccountType.REVENUE,
    "insurance_recovery": AccountType.REVENUE,
    # Expenses (debit normal balance)
    "cost_of_goods_sold": AccountType.EXPENSE,
    "operating_expenses": AccountType.EXPENSE,
    "depreciation_expense": AccountType.EXPENSE,
    "insurance_expense": AccountType.EXPENSE,
    "insurance_loss": AccountType.EXPENSE,
    "tax_expense": AccountType.EXPENSE,
    "interest_expense": AccountType.EXPENSE,
    "collateral_expense": AccountType.EXPENSE,
    "wage_expense": AccountType.EXPENSE,
}


class Ledger:
    """Double-entry accounting ledger for event sourcing.

    The Ledger tracks all financial transactions at the entry level,
    enabling perfect reconciliation and direct method cash flow generation.

    Attributes:
        entries: List of all ledger entries
        chart_of_accounts: Mapping of account names to their types
    """

    def __init__(self) -> None:
        """Initialize an empty ledger."""
        self.entries: List[LedgerEntry] = []
        self.chart_of_accounts: Dict[str, AccountType] = CHART_OF_ACCOUNTS.copy()

    def record(self, entry: LedgerEntry) -> None:
        """Record a single ledger entry.

        Args:
            entry: The LedgerEntry to add to the ledger

        Note:
            Prefer using record_double_entry() for complete transactions
            to ensure debits always equal credits.
        """
        # Add account to chart if not present
        if entry.account not in self.chart_of_accounts:
            # Default to ASSET for unknown accounts
            self.chart_of_accounts[entry.account] = AccountType.ASSET

        self.entries.append(entry)

    def record_double_entry(
        self,
        date: int,
        debit_account: str,
        credit_account: str,
        amount: Union[Decimal, float, int],
        transaction_type: TransactionType,
        description: str = "",
        month: int = 0,
    ) -> Tuple[LedgerEntry, LedgerEntry]:
        """Record a complete double-entry transaction.

        Creates matching debit and credit entries with the same reference_id.

        Args:
            date: Period (year) of the transaction
            debit_account: Account to debit (increase assets/expenses)
            credit_account: Account to credit (increase liabilities/equity/revenue)
            amount: Dollar amount of the transaction (converted to Decimal)
            transaction_type: Classification for cash flow mapping
            description: Human-readable description
            month: Optional month within the year (0-11)

        Returns:
            Tuple of (debit_entry, credit_entry)

        Raises:
            ValueError: If amount is negative

        Example:
            Record a cash sale::

                debit, credit = ledger.record_double_entry(
                    date=5,
                    debit_account="cash",
                    credit_account="revenue",
                    amount=500_000,
                    transaction_type=TransactionType.REVENUE,
                    description="Cash sales"
                )
        """
        # Convert to Decimal for precise calculations
        amount = to_decimal(amount)

        if amount < ZERO:
            raise ValueError(f"Transaction amount must be non-negative, got {amount}")

        if amount == ZERO:
            # Skip zero-amount transactions
            return (
                LedgerEntry(
                    date=date,
                    account=debit_account,
                    amount=ZERO,
                    entry_type=EntryType.DEBIT,
                    transaction_type=transaction_type,
                    description=description,
                    month=month,
                ),
                LedgerEntry(
                    date=date,
                    account=credit_account,
                    amount=ZERO,
                    entry_type=EntryType.CREDIT,
                    transaction_type=transaction_type,
                    description=description,
                    month=month,
                ),
            )

        # Generate shared reference ID
        ref_id = str(uuid.uuid4())
        timestamp = datetime.now()

        debit_entry = LedgerEntry(
            date=date,
            account=debit_account,
            amount=amount,
            entry_type=EntryType.DEBIT,
            transaction_type=transaction_type,
            description=description,
            reference_id=ref_id,
            timestamp=timestamp,
            month=month,
        )

        credit_entry = LedgerEntry(
            date=date,
            account=credit_account,
            amount=amount,
            entry_type=EntryType.CREDIT,
            transaction_type=transaction_type,
            description=description,
            reference_id=ref_id,
            timestamp=timestamp,
            month=month,
        )

        self.record(debit_entry)
        self.record(credit_entry)

        return debit_entry, credit_entry

    def get_balance(self, account: str, as_of_date: Optional[int] = None) -> Decimal:
        """Calculate the balance for an account.

        Args:
            account: Name of the account
            as_of_date: Optional period to calculate balance as of (inclusive)

        Returns:
            Current balance of the account as Decimal, properly signed based on
            account type:
            - Assets/Expenses: positive = debit balance
            - Liabilities/Equity/Revenue: positive = credit balance

        Example:
            Get current cash balance::

                cash = ledger.get_balance("cash")
                print(f"Cash: ${cash:,.0f}")
        """
        account_type = self.chart_of_accounts.get(account, AccountType.ASSET)

        total = ZERO
        for entry in self.entries:
            if entry.account != account:
                continue
            if as_of_date is not None and entry.date > as_of_date:
                continue

            # Calculate based on normal balance
            if account_type in (AccountType.ASSET, AccountType.EXPENSE):
                # Debit-normal accounts: debit increases, credit decreases
                if entry.entry_type == EntryType.DEBIT:
                    total += entry.amount
                else:
                    total -= entry.amount
            else:
                # Credit-normal accounts: credit increases, debit decreases
                if entry.entry_type == EntryType.CREDIT:
                    total += entry.amount
                else:
                    total -= entry.amount

        return total

    def get_period_change(self, account: str, period: int, month: Optional[int] = None) -> Decimal:
        """Calculate the change in account balance for a specific period.

        Args:
            account: Name of the account
            period: Year/period to calculate change for
            month: Optional specific month within the period

        Returns:
            Net change in account balance during the period as Decimal
        """
        account_type = self.chart_of_accounts.get(account, AccountType.ASSET)

        total = ZERO
        for entry in self.entries:
            if entry.account != account:
                continue
            if entry.date != period:
                continue
            if month is not None and entry.month != month:
                continue

            # Calculate based on normal balance
            if account_type in (AccountType.ASSET, AccountType.EXPENSE):
                if entry.entry_type == EntryType.DEBIT:
                    total += entry.amount
                else:
                    total -= entry.amount
            else:
                if entry.entry_type == EntryType.CREDIT:
                    total += entry.amount
                else:
                    total -= entry.amount

        return total

    def get_entries(
        self,
        account: Optional[str] = None,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        transaction_type: Optional[TransactionType] = None,
    ) -> List[LedgerEntry]:
        """Query ledger entries with optional filters.

        Args:
            account: Filter by account name
            start_date: Filter by minimum period (inclusive)
            end_date: Filter by maximum period (inclusive)
            transaction_type: Filter by transaction classification

        Returns:
            List of matching LedgerEntry objects

        Example:
            Get all cash entries for year 5::

                cash_entries = ledger.get_entries(
                    account="cash",
                    start_date=5,
                    end_date=5
                )
        """
        results = []

        for entry in self.entries:
            # Apply filters
            if account is not None and entry.account != account:
                continue
            if start_date is not None and entry.date < start_date:
                continue
            if end_date is not None and entry.date > end_date:
                continue
            if transaction_type is not None and entry.transaction_type != transaction_type:
                continue

            results.append(entry)

        return results

    def sum_by_transaction_type(
        self,
        transaction_type: TransactionType,
        period: int,
        account: Optional[str] = None,
        entry_type: Optional[EntryType] = None,
    ) -> Decimal:
        """Sum entries by transaction type for cash flow extraction.

        Args:
            transaction_type: Classification to sum
            period: Year/period to sum
            account: Optional account filter
            entry_type: Optional debit/credit filter

        Returns:
            Sum of matching entries as Decimal (absolute value)

        Example:
            Get total collections for year 5::

                collections = ledger.sum_by_transaction_type(
                    transaction_type=TransactionType.COLLECTION,
                    period=5,
                    account="cash",
                    entry_type=EntryType.DEBIT
                )
        """
        total = ZERO

        for entry in self.entries:
            if entry.transaction_type != transaction_type:
                continue
            if entry.date != period:
                continue
            if account is not None and entry.account != account:
                continue
            if entry_type is not None and entry.entry_type != entry_type:
                continue

            total += entry.amount

        return total

    def get_cash_flows(self, period: int) -> Dict[str, Decimal]:
        """Extract cash flows for direct method cash flow statement.

        Sums all cash-affecting transactions by category for the specified period.

        Args:
            period: Year/period to extract cash flows for

        Returns:
            Dictionary with cash flow categories as Decimal values:
            - cash_from_customers: Collections on AR + cash sales
            - cash_to_suppliers: Inventory + expense payments
            - cash_for_insurance: Premium payments
            - cash_for_taxes: Tax payments
            - cash_for_wages: Wage payments
            - capital_expenditures: PP&E purchases
            - dividends_paid: Dividend payments
            - net_operating: Total operating cash flow
            - net_investing: Total investing cash flow
            - net_financing: Total financing cash flow

        Example:
            Generate direct method cash flow::

                flows = ledger.get_cash_flows(period=5)
                print(f"Operating: ${flows['net_operating']:,.0f}")
                print(f"Investing: ${flows['net_investing']:,.0f}")
                print(f"Financing: ${flows['net_financing']:,.0f}")
        """
        flows: Dict[str, Decimal] = {}

        # Cash receipts (debits to cash)
        flows["cash_from_customers"] = self.sum_by_transaction_type(
            TransactionType.COLLECTION, period, "cash", EntryType.DEBIT
        ) + self.sum_by_transaction_type(TransactionType.REVENUE, period, "cash", EntryType.DEBIT)

        flows["cash_from_insurance"] = self.sum_by_transaction_type(
            TransactionType.INSURANCE_CLAIM, period, "cash", EntryType.DEBIT
        )

        # Cash payments (credits to cash)
        flows["cash_to_suppliers"] = self.sum_by_transaction_type(
            TransactionType.PAYMENT, period, "cash", EntryType.CREDIT
        ) + self.sum_by_transaction_type(
            TransactionType.INVENTORY_PURCHASE, period, "cash", EntryType.CREDIT
        )

        flows["cash_for_insurance"] = self.sum_by_transaction_type(
            TransactionType.INSURANCE_PREMIUM, period, "cash", EntryType.CREDIT
        )

        flows["cash_for_taxes"] = self.sum_by_transaction_type(
            TransactionType.TAX_PAYMENT, period, "cash", EntryType.CREDIT
        )

        flows["cash_for_wages"] = self.sum_by_transaction_type(
            TransactionType.PAYMENT, period, "cash", EntryType.CREDIT
        )

        # Investing activities
        flows["capital_expenditures"] = self.sum_by_transaction_type(
            TransactionType.CAPEX, period, "cash", EntryType.CREDIT
        )

        flows["asset_sales"] = self.sum_by_transaction_type(
            TransactionType.ASSET_SALE, period, "cash", EntryType.DEBIT
        )

        # Financing activities
        flows["dividends_paid"] = self.sum_by_transaction_type(
            TransactionType.DIVIDEND, period, "cash", EntryType.CREDIT
        )

        flows["equity_issuance"] = self.sum_by_transaction_type(
            TransactionType.EQUITY_ISSUANCE, period, "cash", EntryType.DEBIT
        )

        # Calculate totals
        flows["net_operating"] = (
            flows["cash_from_customers"]
            + flows["cash_from_insurance"]
            - flows["cash_to_suppliers"]
            - flows["cash_for_insurance"]
            - flows["cash_for_taxes"]
        )

        flows["net_investing"] = flows["asset_sales"] - flows["capital_expenditures"]

        flows["net_financing"] = flows["equity_issuance"] - flows["dividends_paid"]

        flows["net_change_in_cash"] = (
            flows["net_operating"] + flows["net_investing"] + flows["net_financing"]
        )

        return flows

    def verify_balance(self) -> Tuple[bool, Decimal]:
        """Verify that debits equal credits (accounting equation).

        Returns:
            Tuple of (is_balanced, difference)
            - is_balanced: True if debits exactly equal credits (using Decimal precision)
            - difference: Total debits minus total credits as Decimal

        Example:
            Check ledger integrity::

                balanced, diff = ledger.verify_balance()
                if not balanced:
                    print(f"Warning: Ledger out of balance by ${diff:,.2f}")
        """
        total_debits = sum(
            (e.amount for e in self.entries if e.entry_type == EntryType.DEBIT),
            ZERO,
        )
        total_credits = sum(
            (e.amount for e in self.entries if e.entry_type == EntryType.CREDIT),
            ZERO,
        )

        difference = total_debits - total_credits
        # With Decimal precision, we can use exact comparison
        is_balanced = difference == ZERO

        return is_balanced, difference

    def get_trial_balance(self, as_of_date: Optional[int] = None) -> Dict[str, Decimal]:
        """Generate a trial balance showing all account balances.

        Args:
            as_of_date: Optional period to generate balance as of

        Returns:
            Dictionary mapping account names to their balances as Decimal

        Example:
            Review all balances::

                trial = ledger.get_trial_balance()
                for account, balance in trial.items():
                    print(f"{account}: ${balance:,.0f}")
        """
        accounts = set(e.account for e in self.entries)

        trial_balance = {}
        for account in sorted(accounts):
            balance = self.get_balance(account, as_of_date)
            if balance != ZERO:  # Skip zero balances (exact comparison with Decimal)
                trial_balance[account] = balance

        return trial_balance

    def clear(self) -> None:
        """Clear all entries from the ledger.

        Useful for resetting the ledger during simulation reset.
        """
        self.entries.clear()

    def __len__(self) -> int:
        """Return the number of entries in the ledger."""
        return len(self.entries)

    def __repr__(self) -> str:
        """Return string representation of the ledger."""
        return f"Ledger(entries={len(self.entries)})"

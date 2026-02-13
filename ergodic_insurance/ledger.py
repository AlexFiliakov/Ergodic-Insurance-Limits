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
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from .decimal_utils import ZERO, is_float_mode, to_decimal

# Lightweight monotonic counter for transaction reference IDs.
# Replaces uuid4() to avoid syscall overhead in the Monte Carlo inner loop.
_entry_counter = itertools.count()


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


class AccountName(Enum):
    """Standard account names for the chart of accounts.

    Using this enum instead of raw strings prevents typos that would
    silently result in zero balances on financial statements. See Issue #260.

    Account names are grouped by their AccountType:

    Assets (debit normal balance):
        CASH, ACCOUNTS_RECEIVABLE, INVENTORY, PREPAID_INSURANCE,
        INSURANCE_RECEIVABLES, GROSS_PPE, ACCUMULATED_DEPRECIATION,
        RESTRICTED_CASH, COLLATERAL, DEFERRED_TAX_ASSET

    Liabilities (credit normal balance):
        ACCOUNTS_PAYABLE, ACCRUED_EXPENSES, ACCRUED_WAGES, ACCRUED_TAXES,
        ACCRUED_INTEREST, CLAIM_LIABILITIES, SHORT_TERM_BORROWINGS,
        UNEARNED_REVENUE

    Equity (credit normal balance):
        RETAINED_EARNINGS, COMMON_STOCK, DIVIDENDS

    Revenue (credit normal balance):
        REVENUE, SALES_REVENUE, INTEREST_INCOME, INSURANCE_RECOVERY

    Expenses (debit normal balance):
        COST_OF_GOODS_SOLD, OPERATING_EXPENSES, DEPRECIATION_EXPENSE,
        INSURANCE_EXPENSE, INSURANCE_LOSS, LAE_EXPENSE, TAX_EXPENSE,
        INTEREST_EXPENSE, COLLATERAL_EXPENSE, WAGE_EXPENSE

    Example:
        Use AccountName instead of strings to prevent typos::

            from ergodic_insurance.ledger import AccountName, Ledger

            ledger = Ledger()
            ledger.record_double_entry(
                date=5,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,  # Safe
                credit_account=AccountName.REVENUE,
                amount=1_000_000,
                transaction_type=TransactionType.REVENUE,
            )

            # This would be a compile/lint error:
            # debit_account=AccountName.ACCOUNT_RECEIVABLE  # Typo caught!
    """

    # Assets (debit normal balance)
    CASH = "cash"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    INVENTORY = "inventory"
    PREPAID_INSURANCE = "prepaid_insurance"
    INSURANCE_RECEIVABLES = "insurance_receivables"
    GROSS_PPE = "gross_ppe"
    ACCUMULATED_DEPRECIATION = "accumulated_depreciation"
    RESTRICTED_CASH = "restricted_cash"
    COLLATERAL = "collateral"  # Deprecated: tracked via RESTRICTED_CASH (Issue #302/#319)
    DEFERRED_TAX_ASSET = "deferred_tax_asset"  # DTA from NOL carryforward per ASC 740
    DTA_VALUATION_ALLOWANCE = "dta_valuation_allowance"  # Contra-asset per ASC 740-10-30-5

    # Liabilities (credit normal balance)
    ACCOUNTS_PAYABLE = "accounts_payable"
    ACCRUED_EXPENSES = "accrued_expenses"
    ACCRUED_WAGES = "accrued_wages"
    ACCRUED_TAXES = "accrued_taxes"
    ACCRUED_INTEREST = "accrued_interest"
    CLAIM_LIABILITIES = "claim_liabilities"
    SHORT_TERM_BORROWINGS = "short_term_borrowings"  # Working capital facility per ASC 470-10
    DEFERRED_TAX_LIABILITY = "deferred_tax_liability"  # DTL from depreciation timing per ASC 740
    UNEARNED_REVENUE = "unearned_revenue"

    # Equity (credit normal balance)
    RETAINED_EARNINGS = "retained_earnings"
    COMMON_STOCK = "common_stock"
    DIVIDENDS = "dividends"

    # Revenue (credit normal balance)
    REVENUE = "revenue"
    SALES_REVENUE = "sales_revenue"
    INTEREST_INCOME = "interest_income"
    INSURANCE_RECOVERY = "insurance_recovery"

    # Expenses (debit normal balance)
    COST_OF_GOODS_SOLD = "cost_of_goods_sold"
    OPERATING_EXPENSES = "operating_expenses"
    DEPRECIATION_EXPENSE = "depreciation_expense"
    INSURANCE_EXPENSE = "insurance_expense"
    INSURANCE_LOSS = "insurance_loss"
    TAX_EXPENSE = "tax_expense"
    INTEREST_EXPENSE = "interest_expense"
    COLLATERAL_EXPENSE = "collateral_expense"
    WAGE_EXPENSE = "wage_expense"
    LAE_EXPENSE = "lae_expense"  # Loss adjustment expenses per ASC 944-40
    RESERVE_DEVELOPMENT = "reserve_development"


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
    WAGE_PAYMENT = "wage_payment"  # Cash payment for wages
    INTEREST_PAYMENT = "interest_payment"  # Cash payment for interest
    INVENTORY_PURCHASE = "inventory_purchase"
    INVENTORY_SALE = "inventory_sale"  # COGS recognition
    INSURANCE_PREMIUM = "insurance_premium"
    INSURANCE_CLAIM = "insurance_claim"
    TAX_ACCRUAL = "tax_accrual"
    TAX_PAYMENT = "tax_payment"
    DTA_ADJUSTMENT = "dta_adjustment"  # Deferred tax asset recognition/reversal
    DTL_ADJUSTMENT = "dtl_adjustment"  # Deferred tax liability recognition/reversal
    RESERVE_DEVELOPMENT = "reserve_development"  # Reserve re-estimation per ASC 944-40-25
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
    WRITE_OFF = "write_off"  # Writing off bad debts or losses
    REVALUATION = "revaluation"  # Asset value adjustments
    LIQUIDATION = "liquidation"  # Bankruptcy/emergency liquidation
    TRANSFER = "transfer"  # Internal asset transfers (e.g., cash to restricted)
    RETAINED_EARNINGS = "retained_earnings"  # Internal equity allocation


@dataclass(slots=True)
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
        reference_id: Lightweight ID linking both sides of a double-entry transaction
        timestamp: Datetime when entry was recorded (None in simulation hot path)
        month: Optional month within the year (0-11)
    """

    date: int  # Year/period
    account: str
    amount: Decimal
    entry_type: EntryType
    transaction_type: TransactionType
    description: str = ""
    reference_id: str = field(default_factory=lambda: f"txn_{next(_entry_counter)}")
    timestamp: Optional[datetime] = None
    month: int = 0  # Month within year (0-11)

    def __post_init__(self) -> None:
        """Validate entry after initialization."""
        # Convert amount to the mode-appropriate type (Issue #1142).
        # In float mode, accept float as-is; in Decimal mode, convert
        # non-Decimal values to Decimal.
        if is_float_mode():
            if not isinstance(self.amount, (Decimal, float)):
                object.__setattr__(self, "amount", to_decimal(self.amount))  # type: ignore[unreachable]
        elif not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", to_decimal(self.amount))  # type: ignore[unreachable]
        if self.amount < ZERO:
            raise ValueError(f"Ledger entry amount must be non-negative, got {self.amount}")
        if not 0 <= self.month <= 11:
            raise ValueError(f"Month must be 0-11, got {self.month}")

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
        return -self.amount

    def __deepcopy__(self, memo: Dict[int, Any]) -> "LedgerEntry":
        """Create a deep copy of this ledger entry.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this LedgerEntry
        """
        import copy

        return LedgerEntry(
            date=self.date,
            account=self.account,
            amount=copy.deepcopy(self.amount, memo),
            entry_type=self.entry_type,
            transaction_type=self.transaction_type,
            description=self.description,
            reference_id=self.reference_id,
            timestamp=self.timestamp,
            month=self.month,
        )


# Standard chart of accounts with their types
# Uses AccountName enum for type safety (Issue #260)
CHART_OF_ACCOUNTS: Dict[AccountName, AccountType] = {
    # Assets (debit normal balance)
    AccountName.CASH: AccountType.ASSET,
    AccountName.ACCOUNTS_RECEIVABLE: AccountType.ASSET,
    AccountName.INVENTORY: AccountType.ASSET,
    AccountName.PREPAID_INSURANCE: AccountType.ASSET,
    AccountName.INSURANCE_RECEIVABLES: AccountType.ASSET,
    AccountName.GROSS_PPE: AccountType.ASSET,
    AccountName.ACCUMULATED_DEPRECIATION: AccountType.ASSET,  # Contra-asset
    AccountName.RESTRICTED_CASH: AccountType.ASSET,
    AccountName.COLLATERAL: AccountType.ASSET,  # Deprecated: tracked via RESTRICTED_CASH (#302/#319)
    AccountName.DEFERRED_TAX_ASSET: AccountType.ASSET,
    AccountName.DTA_VALUATION_ALLOWANCE: AccountType.ASSET,  # Contra-asset (ASC 740-10-30-5)
    # Liabilities (credit normal balance)
    AccountName.ACCOUNTS_PAYABLE: AccountType.LIABILITY,
    AccountName.ACCRUED_EXPENSES: AccountType.LIABILITY,
    AccountName.ACCRUED_WAGES: AccountType.LIABILITY,
    AccountName.ACCRUED_TAXES: AccountType.LIABILITY,
    AccountName.ACCRUED_INTEREST: AccountType.LIABILITY,
    AccountName.CLAIM_LIABILITIES: AccountType.LIABILITY,
    AccountName.SHORT_TERM_BORROWINGS: AccountType.LIABILITY,  # Working capital facility (ASC 470-10)
    AccountName.DEFERRED_TAX_LIABILITY: AccountType.LIABILITY,
    AccountName.UNEARNED_REVENUE: AccountType.LIABILITY,
    # Equity (credit normal balance)
    AccountName.RETAINED_EARNINGS: AccountType.EQUITY,
    AccountName.COMMON_STOCK: AccountType.EQUITY,
    AccountName.DIVIDENDS: AccountType.EQUITY,  # Contra-equity
    # Revenue (credit normal balance)
    AccountName.REVENUE: AccountType.REVENUE,
    AccountName.SALES_REVENUE: AccountType.REVENUE,
    AccountName.INTEREST_INCOME: AccountType.REVENUE,
    AccountName.INSURANCE_RECOVERY: AccountType.REVENUE,
    # Expenses (debit normal balance)
    AccountName.COST_OF_GOODS_SOLD: AccountType.EXPENSE,
    AccountName.OPERATING_EXPENSES: AccountType.EXPENSE,
    AccountName.DEPRECIATION_EXPENSE: AccountType.EXPENSE,
    AccountName.INSURANCE_EXPENSE: AccountType.EXPENSE,
    AccountName.INSURANCE_LOSS: AccountType.EXPENSE,
    AccountName.TAX_EXPENSE: AccountType.EXPENSE,
    AccountName.INTEREST_EXPENSE: AccountType.EXPENSE,
    AccountName.COLLATERAL_EXPENSE: AccountType.EXPENSE,
    AccountName.WAGE_EXPENSE: AccountType.EXPENSE,
    AccountName.LAE_EXPENSE: AccountType.EXPENSE,
    AccountName.RESERVE_DEVELOPMENT: AccountType.EXPENSE,
}

# String-keyed version for backward compatibility and internal lookups
# This allows get_balance("cash") to work while we transition
_CHART_OF_ACCOUNTS_BY_STRING: Dict[str, AccountType] = {
    name.value: account_type for name, account_type in CHART_OF_ACCOUNTS.items()
}

# Set of valid account name strings for fast validation
_VALID_ACCOUNT_NAMES: set[str] = {name.value for name in AccountName}


def _resolve_account_name(account: Union[AccountName, str]) -> str:
    """Resolve an account identifier to its string name with validation.

    Args:
        account: Either an AccountName enum member or a string account name

    Returns:
        The string name of the account

    Raises:
        ValueError: If the account name is not in the chart of accounts

    Example:
        >>> _resolve_account_name(AccountName.CASH)
        'cash'
        >>> _resolve_account_name("cash")
        'cash'
        >>> _resolve_account_name("typo_account")  # Raises ValueError
    """
    if isinstance(account, AccountName):
        return account.value

    # String path - validate against known accounts
    if account not in _VALID_ACCOUNT_NAMES:
        # Provide helpful error message with suggestions
        similar = [name for name in _VALID_ACCOUNT_NAMES if account in name or name in account]
        error_msg = f"Unknown account name: '{account}'. "
        if similar:
            error_msg += f"Did you mean one of: {similar}? "
        error_msg += "Use AccountName enum to prevent typos (Issue #260)."
        raise ValueError(error_msg)

    return account


class Ledger:
    """Double-entry accounting ledger for event sourcing.

    The Ledger tracks all financial transactions at the entry level,
    enabling perfect reconciliation and direct method cash flow generation.

    Attributes:
        entries: List of all ledger entries
        chart_of_accounts: Mapping of account names to their types

    Thread Safety:
        This class is **not** thread-safe.  Concurrent reads are safe, but
        concurrent writes (``record``, ``record_double_entry``,
        ``prune_entries``, ``clear``) or a mix of reads and writes require
        external synchronisation (e.g. a ``threading.Lock``).  Each
        simulation trial should use its own ``Ledger`` instance.
    """

    def __init__(self, strict_validation: bool = True, simulation_mode: bool = False) -> None:
        """Initialize an empty ledger.

        Args:
            strict_validation: If True (default), unknown account names
                raise ValueError. If False, unknown accounts are added
                as ASSET type (backward compatible behavior). The strict
                mode is recommended to catch typos early (Issue #260).
            simulation_mode: If True, only maintain the ``_balances`` cache
                without storing individual entries (Issue #1146).  This
                drastically reduces memory in Monte Carlo hot loops where
                only final balances matter.
        """
        self._simulation_mode = simulation_mode
        self.entries: List[LedgerEntry] = []
        self.chart_of_accounts: Dict[str, AccountType] = _CHART_OF_ACCOUNTS_BY_STRING.copy()
        self._strict_validation = strict_validation
        # Running balance cache for O(1) current balance queries (Issue #259)
        self._balances: Dict[str, Decimal] = {}
        # Snapshot of balances at the prune point (Issue #315)
        self._pruned_balances: Dict[str, Decimal] = {}
        self._prune_cutoff: Optional[int] = None
        # Aggregate debit/credit totals for pruned entries (Issue #362)
        # Use to_decimal(0) so the type adapts to float mode (Issue #1142)
        self._pruned_debits = to_decimal(0)
        self._pruned_credits = to_decimal(0)

    def _update_balance_cache(self, entry: LedgerEntry) -> None:
        """Update running balance cache after recording an entry.

        This maintains O(1) balance lookups for current balances (Issue #259).

        Args:
            entry: The LedgerEntry that was just recorded
        """
        account = entry.account
        account_type = self.chart_of_accounts.get(account, AccountType.ASSET)

        if account not in self._balances:
            self._balances[account] = to_decimal(0)

        # Apply entry based on account type and entry type
        if account_type in (AccountType.ASSET, AccountType.EXPENSE):
            # Debit-normal accounts: debit increases, credit decreases
            if entry.entry_type == EntryType.DEBIT:
                self._balances[account] += entry.amount
            else:
                self._balances[account] -= entry.amount
        else:
            # Credit-normal accounts: credit increases, debit decreases
            if entry.entry_type == EntryType.CREDIT:
                self._balances[account] += entry.amount
            else:
                self._balances[account] -= entry.amount

    def record(self, entry: LedgerEntry) -> None:
        """Record a single ledger entry.

        Args:
            entry: The LedgerEntry to add to the ledger

        Raises:
            ValueError: If strict_validation is True and the account name
                is not in the chart of accounts.

        Note:
            Prefer using record_double_entry() for complete transactions
            to ensure debits always equal credits.
        """
        # Validate account name
        if entry.account not in self.chart_of_accounts:
            if self._strict_validation:
                # Provide helpful error message with suggestions
                similar = [
                    name
                    for name in self.chart_of_accounts.keys()
                    if entry.account in name or name in entry.account
                ]
                error_msg = f"Unknown account name: '{entry.account}'. "
                if similar:
                    error_msg += f"Did you mean one of: {similar}? "
                error_msg += "Use AccountName enum to prevent typos (Issue #260)."
                raise ValueError(error_msg)
            # Backward compatible: add unknown account as ASSET
            self.chart_of_accounts[entry.account] = AccountType.ASSET

        if not self._simulation_mode:
            self.entries.append(entry)
        self._update_balance_cache(entry)

    def record_double_entry(
        self,
        date: int,
        debit_account: Union[AccountName, str],
        credit_account: Union[AccountName, str],
        amount: Union[Decimal, float, int],
        transaction_type: TransactionType,
        description: str = "",
        month: int = 0,
    ) -> Tuple[Optional[LedgerEntry], Optional[LedgerEntry]]:
        """Record a complete double-entry transaction.

        Creates matching debit and credit entries with the same reference_id.

        Args:
            date: Period (year) of the transaction
            debit_account: Account to debit (increase assets/expenses).
                Can be AccountName enum (recommended) or string.
            credit_account: Account to credit (increase liabilities/equity/revenue).
                Can be AccountName enum (recommended) or string.
            amount: Dollar amount of the transaction (converted to Decimal)
            transaction_type: Classification for cash flow mapping
            description: Human-readable description
            month: Optional month within the year (0-11)

        Returns:
            Tuple of (debit_entry, credit_entry), or (None, None) for
            zero-amount transactions (Issue #315).

        Raises:
            ValueError: If amount is negative, or if account names are invalid
                (when strict_validation is True)

        Example:
            Record a cash sale using AccountName enum (recommended)::

                debit, credit = ledger.record_double_entry(
                    date=5,
                    debit_account=AccountName.CASH,
                    credit_account=AccountName.REVENUE,
                    amount=500_000,
                    transaction_type=TransactionType.REVENUE,
                    description="Cash sales"
                )

            String account names still work but are validated::

                debit, credit = ledger.record_double_entry(
                    date=5,
                    debit_account="cash",  # Validated against chart
                    credit_account="revenue",
                    amount=500_000,
                    transaction_type=TransactionType.REVENUE,
                )
        """
        # Resolve account names - in strict mode, validates against known accounts
        # In non-strict mode, allows any account name (backward compatible)
        if self._strict_validation:
            debit_account_str = _resolve_account_name(debit_account)
            credit_account_str = _resolve_account_name(credit_account)
        else:
            # Non-strict: just convert to string, validation happens in record()
            debit_account_str = (
                debit_account.value if isinstance(debit_account, AccountName) else debit_account
            )
            credit_account_str = (
                credit_account.value if isinstance(credit_account, AccountName) else credit_account
            )

        # Convert to Decimal for precise calculations
        amount = to_decimal(amount)

        if amount < ZERO:
            raise ValueError(f"Transaction amount must be non-negative, got {amount}")

        if amount == ZERO:
            # Return None sentinel for zero-amount transactions (Issue #315)
            return (None, None)

        # Shared reference ID for both sides of the double-entry
        ref_id = f"txn_{next(_entry_counter)}"

        debit_entry = LedgerEntry(
            date=date,
            account=debit_account_str,
            amount=amount,
            entry_type=EntryType.DEBIT,
            transaction_type=transaction_type,
            description=description,
            reference_id=ref_id,
            month=month,
        )

        credit_entry = LedgerEntry(
            date=date,
            account=credit_account_str,
            amount=amount,
            entry_type=EntryType.CREDIT,
            transaction_type=transaction_type,
            description=description,
            reference_id=ref_id,
            month=month,
        )

        self.record(debit_entry)
        self.record(credit_entry)

        return debit_entry, credit_entry

    def get_balance(
        self, account: Union[AccountName, str], as_of_date: Optional[int] = None
    ) -> Decimal:
        """Calculate the balance for an account.

        Args:
            account: Name of the account (AccountName enum recommended, string accepted)
            as_of_date: Optional period to calculate balance as of (inclusive).
                When None, returns from cache in O(1). When specified, iterates
                through entries (O(N) for historical queries).

        Returns:
            Current balance of the account as Decimal, properly signed based on
            account type:
            - Assets/Expenses: positive = debit balance
            - Liabilities/Equity/Revenue: positive = credit balance

        Example:
            Get current cash balance::

                cash = ledger.get_balance(AccountName.CASH)
                print(f"Cash: ${cash:,.0f}")

                # String also works (validated)
                cash = ledger.get_balance("cash")
        """
        account_str = _resolve_account_name(account)

        # O(1) lookup for current balance (Issue #259)
        # Use to_decimal(0) so the default adapts to float mode (Issue #1142)
        if as_of_date is None:
            return self._balances.get(account_str, to_decimal(0))

        # Warn when querying dates in the pruned range (Issue #362)
        if self._prune_cutoff is not None and as_of_date < self._prune_cutoff:
            warnings.warn(
                f"as_of_date {as_of_date} is before prune cutoff "
                f"{self._prune_cutoff}; returned balance reflects the "
                f"prune-point snapshot, not the true historical balance",
                stacklevel=2,
            )

        # Historical query: iterate through entries (less frequent use case)
        account_type = self.chart_of_accounts.get(account_str, AccountType.ASSET)

        # Start from pruned snapshot if entries have been pruned (Issue #315)
        total = self._pruned_balances.get(account_str, to_decimal(0))
        for entry in self.entries:
            if entry.account != account_str:
                continue
            if entry.date > as_of_date:
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

    def get_period_change(
        self, account: Union[AccountName, str], period: int, month: Optional[int] = None
    ) -> Decimal:
        """Calculate the change in account balance for a specific period.

        Args:
            account: Name of the account (AccountName enum recommended, string accepted)
            period: Year/period to calculate change for
            month: Optional specific month within the period

        Returns:
            Net change in account balance during the period as Decimal
        """
        account_str = _resolve_account_name(account)
        account_type = self.chart_of_accounts.get(account_str, AccountType.ASSET)

        total = to_decimal(0)
        for entry in self.entries:
            if entry.account != account_str:
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
        account: Optional[Union[AccountName, str]] = None,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        transaction_type: Optional[TransactionType] = None,
    ) -> List[LedgerEntry]:
        """Query ledger entries with optional filters.

        Args:
            account: Filter by account name (AccountName enum or string)
            start_date: Filter by minimum period (inclusive)
            end_date: Filter by maximum period (inclusive)
            transaction_type: Filter by transaction classification

        Returns:
            List of matching LedgerEntry objects

        Example:
            Get all cash entries for year 5::

                cash_entries = ledger.get_entries(
                    account=AccountName.CASH,
                    start_date=5,
                    end_date=5
                )
        """
        # Resolve account name if provided
        account_str = _resolve_account_name(account) if account is not None else None

        results = []

        for entry in self.entries:
            # Apply filters
            if account_str is not None and entry.account != account_str:
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
        account: Optional[Union[AccountName, str]] = None,
        entry_type: Optional[EntryType] = None,
    ) -> Decimal:
        """Sum entries by transaction type for cash flow extraction.

        Args:
            transaction_type: Classification to sum
            period: Year/period to sum
            account: Optional account filter (AccountName enum or string)
            entry_type: Optional debit/credit filter

        Returns:
            Sum of matching entries as Decimal (absolute value)

        Example:
            Get total collections for year 5::

                collections = ledger.sum_by_transaction_type(
                    transaction_type=TransactionType.COLLECTION,
                    period=5,
                    account=AccountName.CASH,
                    entry_type=EntryType.DEBIT
                )
        """
        # Resolve account name if provided
        account_str = _resolve_account_name(account) if account is not None else None

        total = to_decimal(0)

        for entry in self.entries:
            if entry.transaction_type != transaction_type:
                continue
            if entry.date != period:
                continue
            if account_str is not None and entry.account != account_str:
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
            - cash_for_claim_losses: Claim-related asset reduction payments
            - cash_for_taxes: Tax payments
            - cash_for_wages: Wage payments
            - cash_for_interest: Interest payments
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

        flows["cash_for_claim_losses"] = self.sum_by_transaction_type(
            TransactionType.INSURANCE_CLAIM, period, "cash", EntryType.CREDIT
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
            TransactionType.WAGE_PAYMENT, period, "cash", EntryType.CREDIT
        )

        flows["cash_for_interest"] = self.sum_by_transaction_type(
            TransactionType.INTEREST_PAYMENT, period, "cash", EntryType.CREDIT
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

        # Calculate totals (Issue #319: include wages and interest in operating)
        # Issue #379: include claim loss payments in operating outflows
        flows["net_operating"] = (
            flows["cash_from_customers"]
            + flows["cash_from_insurance"]
            - flows["cash_to_suppliers"]
            - flows["cash_for_insurance"]
            - flows["cash_for_claim_losses"]
            - flows["cash_for_taxes"]
            - flows["cash_for_wages"]
            - flows["cash_for_interest"]
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
                    warnings.warn(
                        f"Ledger out of balance by ${diff:,.2f}",
                        stacklevel=2,
                    )
        """
        total_debits = self._pruned_debits + sum(
            (e.amount for e in self.entries if e.entry_type == EntryType.DEBIT),
            to_decimal(0),
        )
        total_credits = self._pruned_credits + sum(
            (e.amount for e in self.entries if e.entry_type == EntryType.CREDIT),
            to_decimal(0),
        )

        difference = total_debits - total_credits
        # With Decimal precision, we can use exact comparison
        is_balanced = difference == ZERO

        return is_balanced, difference

    def get_trial_balance(self, as_of_date: Optional[int] = None) -> Dict[str, Decimal]:
        """Generate a trial balance showing all account balances.

        When ``as_of_date`` is None, reads directly from the O(1) balance
        cache.  When a date is specified, performs a single O(N) pass over
        all entries instead of the previous O(N*M) approach (Issue #315).

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
        if as_of_date is None:
            # O(1): read directly from cached balances
            return {
                account: balance
                for account, balance in sorted(self._balances.items())
                if balance != ZERO
            }

        # Warn when querying dates in the pruned range (Issue #362)
        if self._prune_cutoff is not None and as_of_date < self._prune_cutoff:
            warnings.warn(
                f"as_of_date {as_of_date} is before prune cutoff "
                f"{self._prune_cutoff}; returned balances reflect the "
                f"prune-point snapshot, not true historical balances",
                stacklevel=2,
            )

        # O(N) single-pass: accumulate per-account balances in one iteration
        # Include any pruned snapshot balances as starting points
        totals: Dict[str, Decimal] = {}
        if self._pruned_balances:
            for account, snap_balance in self._pruned_balances.items():
                totals[account] = snap_balance

        for entry in self.entries:
            if entry.date > as_of_date:
                continue

            account = entry.account
            if account not in totals:
                totals[account] = to_decimal(0)

            account_type = self.chart_of_accounts.get(account, AccountType.ASSET)
            if account_type in (AccountType.ASSET, AccountType.EXPENSE):
                if entry.entry_type == EntryType.DEBIT:
                    totals[account] += entry.amount
                else:
                    totals[account] -= entry.amount
            else:
                if entry.entry_type == EntryType.CREDIT:
                    totals[account] += entry.amount
                else:
                    totals[account] -= entry.amount

        return {account: balance for account, balance in sorted(totals.items()) if balance != ZERO}

    def prune_entries(self, before_date: int) -> int:
        """Discard entries older than *before_date* to bound memory (Issue #315).

        Before discarding, a per-account balance snapshot is computed so
        that ``get_balance(account, as_of_date)`` and ``get_trial_balance``
        still return correct values for dates >= the prune point.

        Entries with ``date < before_date`` are removed.  The current
        balance cache (``_balances``) is unaffected because it already
        holds the cumulative totals.

        Args:
            before_date: Entries with ``date`` strictly less than this
                value are pruned.

        Returns:
            Number of entries removed.

        Note:
            After pruning, historical queries for dates prior to
            ``before_date`` will reflect the snapshot balance at the prune
            boundary, not the true historical balance at that earlier date.
        """
        # Build snapshot of balances for entries that will be removed
        snapshot: Dict[str, Decimal] = {}
        if self._pruned_balances:
            snapshot = dict(self._pruned_balances)

        # Track aggregate debit/credit totals for verify_balance (Issue #362)
        pruned_debits = self._pruned_debits
        pruned_credits = self._pruned_credits

        kept: List[LedgerEntry] = []
        removed = 0
        for entry in self.entries:
            if entry.date < before_date:
                # Accumulate into snapshot
                account = entry.account
                if account not in snapshot:
                    snapshot[account] = to_decimal(0)
                account_type = self.chart_of_accounts.get(account, AccountType.ASSET)
                if account_type in (AccountType.ASSET, AccountType.EXPENSE):
                    if entry.entry_type == EntryType.DEBIT:
                        snapshot[account] += entry.amount
                    else:
                        snapshot[account] -= entry.amount
                else:
                    if entry.entry_type == EntryType.CREDIT:
                        snapshot[account] += entry.amount
                    else:
                        snapshot[account] -= entry.amount
                # Track raw debit/credit for verify_balance
                if entry.entry_type == EntryType.DEBIT:
                    pruned_debits += entry.amount
                else:
                    pruned_credits += entry.amount
                removed += 1
            else:
                kept.append(entry)

        self.entries = kept
        self._pruned_balances = snapshot
        self._prune_cutoff = before_date
        self._pruned_debits = pruned_debits
        self._pruned_credits = pruned_credits
        return removed

    def clear(self) -> None:
        """Clear all entries from the ledger.

        Useful for resetting the ledger during simulation reset.
        Also resets the balance cache (Issue #259) and pruning state (Issue #315).
        """
        self.entries.clear()
        self._balances.clear()
        self._pruned_balances.clear()
        self._prune_cutoff = None
        self._pruned_debits = to_decimal(0)
        self._pruned_credits = to_decimal(0)

    def __len__(self) -> int:
        """Return the number of entries in the ledger."""
        return len(self.entries)

    def __repr__(self) -> str:
        """Return string representation of the ledger."""
        return f"Ledger(entries={len(self.entries)})"

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Ledger":
        """Create a deep copy of this ledger.

        Preserves all entries and the balance cache for O(1) balance queries.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this Ledger with all entries and cached balances
        """
        import copy

        # Create new instance without calling __init__ to avoid reinitializing
        result = Ledger.__new__(Ledger)
        memo[id(self)] = result

        # Deep copy entries
        result.entries = copy.deepcopy(self.entries, memo)

        # Copy chart of accounts (shallow copy is fine - values are enums)
        result.chart_of_accounts = self.chart_of_accounts.copy()

        # Copy validation and simulation mode settings
        result._strict_validation = self._strict_validation
        result._simulation_mode = self._simulation_mode

        # Deep copy balance cache
        result._balances = copy.deepcopy(self._balances, memo)

        # Deep copy pruning state (Issue #315, #362)
        result._pruned_balances = copy.deepcopy(self._pruned_balances, memo)
        result._prune_cutoff = self._prune_cutoff
        result._pruned_debits = self._pruned_debits
        result._pruned_credits = self._pruned_credits

        return result

"""Accrual and timing management for financial operations.

This module provides functionality to track timing differences between
cash movements and accounting recognition, following GAAP principles.

Uses Decimal for all currency amounts to prevent floating-point precision errors.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .decimal_utils import ZERO, to_decimal


class AccrualType(Enum):
    """Types of accrued items."""

    WAGES = "wages"
    INTEREST = "interest"
    TAXES = "taxes"
    INSURANCE_CLAIMS = "insurance_claims"
    REVENUE = "revenue"
    OTHER = "other"


class PaymentSchedule(Enum):
    """Payment schedule types."""

    IMMEDIATE = "immediate"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


@dataclass
class AccrualItem:
    """Individual accrual item with tracking information.

    Uses Decimal for all currency amounts to ensure precise calculations.
    """

    item_type: AccrualType
    amount: Decimal
    period_incurred: int  # Month or year when expense/revenue incurred
    payment_schedule: PaymentSchedule
    payment_dates: List[int] = field(default_factory=list)
    amounts_paid: List[Decimal] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        """Convert amounts to Decimal if needed (runtime check for backwards compatibility)."""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", to_decimal(self.amount))  # type: ignore[unreachable]
        # Convert any float amounts in amounts_paid to Decimal
        converted_paid = [
            to_decimal(a) if not isinstance(a, Decimal) else a for a in self.amounts_paid
        ]
        object.__setattr__(self, "amounts_paid", converted_paid)

    @property
    def remaining_balance(self) -> Decimal:
        """Calculate remaining unpaid balance."""
        # Convert any float values that may have been added after construction
        paid_total = sum((to_decimal(a) for a in self.amounts_paid), ZERO)
        return self.amount - paid_total

    @property
    def is_fully_paid(self) -> bool:
        """Check if accrual has been fully paid."""
        # With Decimal precision, we can use exact comparison
        return self.remaining_balance == ZERO

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AccrualItem":
        """Create a deep copy of this accrual item.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this AccrualItem
        """
        import copy

        return AccrualItem(
            item_type=self.item_type,
            amount=copy.deepcopy(self.amount, memo),
            period_incurred=self.period_incurred,
            payment_schedule=self.payment_schedule,
            payment_dates=copy.deepcopy(self.payment_dates, memo),
            amounts_paid=copy.deepcopy(self.amounts_paid, memo),
            description=self.description,
        )


class AccrualManager:
    """Manages accruals and timing differences for financial operations.

    Tracks accrued expenses and revenues with various payment schedules,
    particularly focusing on quarterly tax payments and multi-year claim
    settlements. Uses FIFO approach for payment matching.
    """

    def __init__(self, fiscal_year_end: int = 12):
        """Initialize the accrual manager.

        Args:
            fiscal_year_end: Month of fiscal year end (1-12). Default is 12
                (December) for calendar year alignment.
        """
        self.fiscal_year_end = fiscal_year_end
        self._fiscal_year_start = (fiscal_year_end % 12) + 1
        self.accrued_expenses: Dict[AccrualType, List[AccrualItem]] = {
            accrual_type: [] for accrual_type in AccrualType
        }
        self.accrued_revenues: List[AccrualItem] = []
        self.current_period: int = 0

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AccrualManager":
        """Create a deep copy of this accrual manager.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this AccrualManager with all accruals
        """
        import copy

        result = AccrualManager(fiscal_year_end=self.fiscal_year_end)
        memo[id(self)] = result

        # Deep copy all accrued expenses
        result.accrued_expenses = {
            accrual_type: copy.deepcopy(items, memo)
            for accrual_type, items in self.accrued_expenses.items()
        }

        # Deep copy accrued revenues
        result.accrued_revenues = copy.deepcopy(self.accrued_revenues, memo)

        # Copy current period
        result.current_period = self.current_period

        return result

    def _get_fiscal_payment_periods(self) -> List[int]:
        """Compute absolute payment periods for quarterly tax payments.

        IRS rules: payments are due on the 15th day of the 4th, 6th, 9th, and
        12th months of the corporation's tax year.

        Returns:
            List of 4 absolute period numbers for quarterly tax payments.
        """
        # 0-indexed start month of fiscal year
        fiscal_start_month = self.fiscal_year_end % 12

        current_month = self.current_period % 12
        current_year = self.current_period // 12

        # Find the start of the fiscal year containing current_period
        if fiscal_start_month == 0 or current_month >= fiscal_start_month:
            fy_start_period = current_year * 12 + fiscal_start_month
        else:
            fy_start_period = (current_year - 1) * 12 + fiscal_start_month

        # Payment offsets from fiscal year start (4th, 6th, 9th, 12th months)
        return [fy_start_period + offset for offset in [3, 5, 8, 11]]

    def record_expense_accrual(
        self,
        item_type: AccrualType,
        amount: Union[Decimal, float, int],
        payment_schedule: PaymentSchedule = PaymentSchedule.IMMEDIATE,
        payment_dates: Optional[List[int]] = None,
        description: str = "",
    ) -> AccrualItem:
        """Record an accrued expense.

        Args:
            item_type: Type of expense being accrued
            amount: Total amount to be accrued (converted to Decimal)
            payment_schedule: Schedule for payments
            payment_dates: Custom payment dates if schedule is CUSTOM
            description: Optional description of the accrual

        Returns:
            The created AccrualItem
        """
        amount = to_decimal(amount)
        if payment_schedule == PaymentSchedule.CUSTOM and not payment_dates:
            raise ValueError("Custom schedule requires payment_dates")

        # Generate payment dates based on schedule
        if payment_schedule == PaymentSchedule.QUARTERLY:
            # Quarterly tax payments on 15th of 4th, 6th, 9th, 12th fiscal months
            payment_dates = self._get_fiscal_payment_periods()
        elif payment_schedule == PaymentSchedule.ANNUAL:
            payment_dates = [self.current_period + 12]
        elif payment_schedule == PaymentSchedule.IMMEDIATE:
            payment_dates = [self.current_period]

        accrual = AccrualItem(
            item_type=item_type,
            amount=amount,
            period_incurred=self.current_period,
            payment_schedule=payment_schedule,
            payment_dates=payment_dates or [],
            description=description,
        )

        self.accrued_expenses[item_type].append(accrual)
        return accrual

    def record_revenue_accrual(
        self,
        amount: Union[Decimal, float, int],
        collection_dates: Optional[List[int]] = None,
        description: str = "",
    ) -> AccrualItem:
        """Record accrued revenue not yet collected.

        Args:
            amount: Amount of revenue accrued (converted to Decimal)
            collection_dates: Expected collection dates
            description: Optional description

        Returns:
            The created AccrualItem
        """
        amount = to_decimal(amount)
        accrual = AccrualItem(
            item_type=AccrualType.REVENUE,
            amount=amount,
            period_incurred=self.current_period,
            payment_schedule=(
                PaymentSchedule.CUSTOM if collection_dates else PaymentSchedule.IMMEDIATE
            ),
            payment_dates=collection_dates or [self.current_period],
            description=description,
        )

        self.accrued_revenues.append(accrual)
        return accrual

    def process_payment(
        self,
        item_type: AccrualType,
        amount: Union[Decimal, float, int],
        period: Optional[int] = None,
    ) -> List[Tuple[AccrualItem, Decimal]]:
        """Process a payment against accrued items using FIFO.

        Args:
            item_type: Type of accrual being paid
            amount: Payment amount (converted to Decimal)
            period: Period when payment is made (defaults to current)

        Returns:
            List of (AccrualItem, amount_applied) tuples with Decimal amounts
        """
        amount = to_decimal(amount)

        if period is None:
            period = self.current_period

        if item_type == AccrualType.REVENUE:
            accruals = self.accrued_revenues
        else:
            accruals = self.accrued_expenses[item_type]

        remaining_payment = amount
        payments_applied: List[Tuple[AccrualItem, Decimal]] = []

        # Apply payment to accruals in FIFO order
        for accrual in accruals:
            if accrual.is_fully_paid or remaining_payment <= ZERO:
                continue

            amount_to_apply = min(remaining_payment, accrual.remaining_balance)
            accrual.amounts_paid.append(amount_to_apply)
            payments_applied.append((accrual, amount_to_apply))
            remaining_payment -= amount_to_apply

        return payments_applied

    def get_quarterly_tax_schedule(
        self, annual_tax: Union[Decimal, float, int]
    ) -> List[Tuple[int, Decimal]]:
        """Calculate quarterly tax payment schedule.

        Args:
            annual_tax: Total annual tax liability (converted to Decimal)

        Returns:
            List of (period, amount) tuples for quarterly payments (Decimal amounts)
        """
        annual_tax = to_decimal(annual_tax)
        quarterly_amount = annual_tax / Decimal(4)
        payment_periods = self._get_fiscal_payment_periods()

        return [(period, quarterly_amount) for period in payment_periods]

    def get_claim_payment_schedule(
        self,
        claim_amount: Union[Decimal, float, int],
        development_pattern: Optional[List[Union[Decimal, float]]] = None,
    ) -> List[Tuple[int, Decimal]]:
        """Calculate insurance claim payment schedule over multiple years.

        Args:
            claim_amount: Total claim amount (converted to Decimal)
            development_pattern: Percentage paid each year (defaults to standard pattern)

        Returns:
            List of (period, amount) tuples for claim payments (Decimal amounts)
        """
        claim_amount = to_decimal(claim_amount)

        if development_pattern is None:
            # Standard claim development pattern
            development_pattern = [
                Decimal("0.4"),
                Decimal("0.3"),
                Decimal("0.2"),
                Decimal("0.1"),
            ]  # 40%, 30%, 20%, 10%

        schedule: List[Tuple[int, Decimal]] = []
        for year, percentage in enumerate(development_pattern):
            period = self.current_period + (year * 12)
            amount = claim_amount * to_decimal(percentage)
            schedule.append((period, amount))

        return schedule

    def get_total_accrued_expenses(self) -> Decimal:
        """Get total outstanding accrued expenses as Decimal."""
        total = ZERO
        for expense_list in self.accrued_expenses.values():
            for accrual in expense_list:
                if not accrual.is_fully_paid:
                    total += accrual.remaining_balance
        return total

    def get_total_accrued_revenues(self) -> Decimal:
        """Get total outstanding accrued revenues as Decimal."""
        return sum(
            (
                accrual.remaining_balance
                for accrual in self.accrued_revenues
                if not accrual.is_fully_paid
            ),
            ZERO,
        )

    def get_accruals_by_type(self, item_type: AccrualType) -> List[AccrualItem]:
        """Get all accruals of a specific type.

        Args:
            item_type: Type of accrual to retrieve

        Returns:
            List of accruals of the specified type
        """
        if item_type == AccrualType.REVENUE:
            return self.accrued_revenues
        return self.accrued_expenses[item_type]

    def get_payments_due(self, period: Optional[int] = None) -> Dict[AccrualType, Decimal]:
        """Get payments due in a specific period.

        Args:
            period: Period to check (defaults to current)

        Returns:
            Dictionary of payment amounts by type (Decimal values)
        """
        if period is None:
            period = self.current_period

        payments_due: Dict[AccrualType, Decimal] = {}

        # Check expense accruals
        for expense_type, accruals in self.accrued_expenses.items():
            amount_due = ZERO
            for accrual in accruals:
                if not accrual.is_fully_paid:
                    # Count how many payment dates are due (including past-due)
                    due_payments = 0
                    paid_payments = len(accrual.amounts_paid)

                    for payment_date in accrual.payment_dates:
                        if payment_date <= period:
                            due_payments += 1

                    # Calculate how many payments still need to be made
                    unpaid_due = due_payments - paid_payments
                    if unpaid_due > 0:
                        # Calculate proportional payment amount
                        total_periods = len(accrual.payment_dates)
                        amount_per_payment = accrual.amount / Decimal(total_periods)
                        amount_due += amount_per_payment * Decimal(unpaid_due)

            if amount_due > ZERO:
                payments_due[expense_type] = amount_due

        return payments_due

    def advance_period(self, periods: int = 1):
        """Advance the current period.

        Args:
            periods: Number of periods to advance
        """
        self.current_period += periods

    def get_balance_sheet_items(self) -> Dict[str, Decimal]:
        """Get accrual items for balance sheet reporting.

        Returns:
            Dictionary with balance sheet line items (Decimal values)
        """
        return {
            "accrued_expenses": self.get_total_accrued_expenses(),
            "accrued_revenues": self.get_total_accrued_revenues(),
            "accrued_wages": sum(
                (
                    a.remaining_balance
                    for a in self.accrued_expenses[AccrualType.WAGES]
                    if not a.is_fully_paid
                ),
                ZERO,
            ),
            "accrued_taxes": sum(
                (
                    a.remaining_balance
                    for a in self.accrued_expenses[AccrualType.TAXES]
                    if not a.is_fully_paid
                ),
                ZERO,
            ),
            "accrued_interest": sum(
                (
                    a.remaining_balance
                    for a in self.accrued_expenses[AccrualType.INTEREST]
                    if not a.is_fully_paid
                ),
                ZERO,
            ),
        }

    def clear_fully_paid(self):
        """Remove fully paid accruals to maintain performance."""
        # Clean up expense accruals
        for expense_type in self.accrued_expenses:
            self.accrued_expenses[expense_type] = [
                accrual
                for accrual in self.accrued_expenses[expense_type]
                if not accrual.is_fully_paid
            ]

        # Clean up revenue accruals
        self.accrued_revenues = [
            accrual for accrual in self.accrued_revenues if not accrual.is_fully_paid
        ]

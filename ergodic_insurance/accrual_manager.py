"""Accrual and timing management for financial operations.

This module provides functionality to track timing differences between
cash movements and accounting recognition, following GAAP principles.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


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
    """Individual accrual item with tracking information."""

    item_type: AccrualType
    amount: float
    period_incurred: int  # Month or year when expense/revenue incurred
    payment_schedule: PaymentSchedule
    payment_dates: List[int] = field(default_factory=list)
    amounts_paid: List[float] = field(default_factory=list)
    description: str = ""

    @property
    def remaining_balance(self) -> float:
        """Calculate remaining unpaid balance."""
        return self.amount - sum(self.amounts_paid)

    @property
    def is_fully_paid(self) -> bool:
        """Check if accrual has been fully paid."""
        return abs(self.remaining_balance) < 0.01  # Floating point tolerance


class AccrualManager:
    """Manages accruals and timing differences for financial operations.

    Tracks accrued expenses and revenues with various payment schedules,
    particularly focusing on quarterly tax payments and multi-year claim
    settlements. Uses FIFO approach for payment matching.
    """

    def __init__(self):
        """Initialize the accrual manager."""
        self.accrued_expenses: Dict[AccrualType, List[AccrualItem]] = {
            accrual_type: [] for accrual_type in AccrualType
        }
        self.accrued_revenues: List[AccrualItem] = []
        self.current_period: int = 0

    def record_expense_accrual(
        self,
        item_type: AccrualType,
        amount: float,
        payment_schedule: PaymentSchedule = PaymentSchedule.IMMEDIATE,
        payment_dates: Optional[List[int]] = None,
        description: str = "",
    ) -> AccrualItem:
        """Record an accrued expense.

        Args:
            item_type: Type of expense being accrued
            amount: Total amount to be accrued
            payment_schedule: Schedule for payments
            payment_dates: Custom payment dates if schedule is CUSTOM
            description: Optional description of the accrual

        Returns:
            The created AccrualItem
        """
        if payment_schedule == PaymentSchedule.CUSTOM and not payment_dates:
            raise ValueError("Custom schedule requires payment_dates")

        # Generate payment dates based on schedule
        if payment_schedule == PaymentSchedule.QUARTERLY:
            # Quarterly tax payments on 15th of 4th, 6th, 9th, 12th months
            base_year = self.current_period // 12
            payment_dates = [base_year * 12 + month for month in [3, 5, 8, 11]]
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
        self, amount: float, collection_dates: Optional[List[int]] = None, description: str = ""
    ) -> AccrualItem:
        """Record accrued revenue not yet collected.

        Args:
            amount: Amount of revenue accrued
            collection_dates: Expected collection dates
            description: Optional description

        Returns:
            The created AccrualItem
        """
        accrual = AccrualItem(
            item_type=AccrualType.REVENUE,
            amount=amount,
            period_incurred=self.current_period,
            payment_schedule=PaymentSchedule.CUSTOM
            if collection_dates
            else PaymentSchedule.IMMEDIATE,
            payment_dates=collection_dates or [self.current_period],
            description=description,
        )

        self.accrued_revenues.append(accrual)
        return accrual

    def process_payment(
        self, item_type: AccrualType, amount: float, period: Optional[int] = None
    ) -> List[Tuple[AccrualItem, float]]:
        """Process a payment against accrued items using FIFO.

        Args:
            item_type: Type of accrual being paid
            amount: Payment amount
            period: Period when payment is made (defaults to current)

        Returns:
            List of (AccrualItem, amount_applied) tuples
        """
        if period is None:
            period = self.current_period

        if item_type == AccrualType.REVENUE:
            accruals = self.accrued_revenues
        else:
            accruals = self.accrued_expenses[item_type]

        remaining_payment = amount
        payments_applied = []

        # Apply payment to accruals in FIFO order
        for accrual in accruals:
            if accrual.is_fully_paid or remaining_payment <= 0:
                continue

            amount_to_apply = min(remaining_payment, accrual.remaining_balance)
            accrual.amounts_paid.append(amount_to_apply)
            payments_applied.append((accrual, amount_to_apply))
            remaining_payment -= amount_to_apply

        return payments_applied

    def get_quarterly_tax_schedule(self, annual_tax: float) -> List[Tuple[int, float]]:
        """Calculate quarterly tax payment schedule.

        Args:
            annual_tax: Total annual tax liability

        Returns:
            List of (period, amount) tuples for quarterly payments
        """
        quarterly_amount = annual_tax / 4
        base_year = self.current_period // 12

        return [
            (base_year * 12 + 3, quarterly_amount),  # April 15
            (base_year * 12 + 5, quarterly_amount),  # June 15
            (base_year * 12 + 8, quarterly_amount),  # September 15
            (base_year * 12 + 11, quarterly_amount),  # December 15
        ]

    def get_claim_payment_schedule(
        self, claim_amount: float, development_pattern: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """Calculate insurance claim payment schedule over multiple years.

        Args:
            claim_amount: Total claim amount
            development_pattern: Percentage paid each year (defaults to standard pattern)

        Returns:
            List of (period, amount) tuples for claim payments
        """
        if development_pattern is None:
            # Standard claim development pattern
            development_pattern = [0.4, 0.3, 0.2, 0.1]  # 40%, 30%, 20%, 10%

        schedule = []
        for year, percentage in enumerate(development_pattern):
            period = self.current_period + (year * 12)
            amount = claim_amount * percentage
            schedule.append((period, amount))

        return schedule

    def get_total_accrued_expenses(self) -> float:
        """Get total outstanding accrued expenses."""
        total = 0.0
        for expense_list in self.accrued_expenses.values():
            for accrual in expense_list:
                if not accrual.is_fully_paid:
                    total += accrual.remaining_balance
        return total

    def get_total_accrued_revenues(self) -> float:
        """Get total outstanding accrued revenues."""
        return sum(
            accrual.remaining_balance
            for accrual in self.accrued_revenues
            if not accrual.is_fully_paid
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

    def get_payments_due(self, period: Optional[int] = None) -> Dict[AccrualType, float]:
        """Get payments due in a specific period.

        Args:
            period: Period to check (defaults to current)

        Returns:
            Dictionary of payment amounts by type
        """
        if period is None:
            period = self.current_period

        payments_due = {}

        # Check expense accruals
        for expense_type, accruals in self.accrued_expenses.items():
            amount_due = 0.0
            for accrual in accruals:
                if not accrual.is_fully_paid and period in accrual.payment_dates:
                    # Calculate proportional payment for this period
                    total_periods = len(accrual.payment_dates)
                    amount_due += accrual.amount / total_periods

            if amount_due > 0:
                payments_due[expense_type] = amount_due

        return payments_due

    def advance_period(self, periods: int = 1):
        """Advance the current period.

        Args:
            periods: Number of periods to advance
        """
        self.current_period += periods

    def get_balance_sheet_items(self) -> Dict[str, float]:
        """Get accrual items for balance sheet reporting.

        Returns:
            Dictionary with balance sheet line items
        """
        return {
            "accrued_expenses": self.get_total_accrued_expenses(),
            "accrued_revenues": self.get_total_accrued_revenues(),
            "accrued_wages": sum(
                a.remaining_balance
                for a in self.accrued_expenses[AccrualType.WAGES]
                if not a.is_fully_paid
            ),
            "accrued_taxes": sum(
                a.remaining_balance
                for a in self.accrued_expenses[AccrualType.TAXES]
                if not a.is_fully_paid
            ),
            "accrued_interest": sum(
                a.remaining_balance
                for a in self.accrued_expenses[AccrualType.INTEREST]
                if not a.is_fully_paid
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

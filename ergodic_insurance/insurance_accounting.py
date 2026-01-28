"""Insurance premium accounting module.

This module provides proper insurance premium accounting with prepaid asset tracking
and systematic monthly amortization following GAAP principles.

Uses Decimal for all currency amounts to prevent floating-point precision errors
in iterative calculations.
"""

from dataclasses import dataclass, field
from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional, Union

from .decimal_utils import ZERO, quantize_currency, to_decimal

logger = logging.getLogger(__name__)


@dataclass
class InsuranceRecovery:
    """Represents an insurance claim recovery receivable.

    Attributes:
        amount: Recovery amount approved by insurance (Decimal)
        claim_id: Unique identifier for the claim
        year_approved: Year when recovery was approved
        amount_received: Amount received to date (Decimal)
    """

    amount: Decimal
    claim_id: str
    year_approved: int
    amount_received: Decimal = field(default_factory=lambda: ZERO)

    def __post_init__(self) -> None:
        """Convert amounts to Decimal if needed (runtime check for backwards compatibility)."""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", to_decimal(self.amount))  # type: ignore[unreachable]
        if not isinstance(self.amount_received, Decimal):
            object.__setattr__(self, "amount_received", to_decimal(self.amount_received))  # type: ignore[unreachable]

    @property
    def outstanding(self) -> Decimal:
        """Calculate outstanding receivable amount."""
        return self.amount - self.amount_received

    def __deepcopy__(self, memo: Dict[int, Any]) -> "InsuranceRecovery":
        """Create a deep copy of this insurance recovery.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this InsuranceRecovery
        """
        import copy

        return InsuranceRecovery(
            amount=copy.deepcopy(self.amount, memo),
            claim_id=self.claim_id,
            year_approved=self.year_approved,
            amount_received=copy.deepcopy(self.amount_received, memo),
        )


@dataclass
class InsuranceAccounting:
    """Manages insurance premium accounting with proper GAAP treatment.

    This class tracks annual insurance premium payments as prepaid assets
    and amortizes them monthly over the coverage period using straight-line
    amortization. It also tracks insurance claim recoveries separately from
    claim liabilities.

    All currency amounts use Decimal for precise financial calculations.

    Attributes:
        prepaid_insurance: Current prepaid insurance asset balance (Decimal)
        monthly_expense: Calculated monthly insurance expense (Decimal)
        annual_premium: Total annual premium amount (Decimal)
        months_in_period: Number of months in coverage period (default 12)
        current_month: Current month in coverage period
        recoveries: List of insurance recoveries receivable
    """

    prepaid_insurance: Decimal = field(default_factory=lambda: ZERO)
    monthly_expense: Decimal = field(default_factory=lambda: ZERO)
    annual_premium: Decimal = field(default_factory=lambda: ZERO)
    months_in_period: int = 12
    current_month: int = 0
    recoveries: List[InsuranceRecovery] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert amounts to Decimal if needed (runtime check for backwards compatibility)."""
        if not isinstance(self.prepaid_insurance, Decimal):
            self.prepaid_insurance = to_decimal(self.prepaid_insurance)  # type: ignore[unreachable]
        if not isinstance(self.monthly_expense, Decimal):
            self.monthly_expense = to_decimal(self.monthly_expense)  # type: ignore[unreachable]
        if not isinstance(self.annual_premium, Decimal):
            self.annual_premium = to_decimal(self.annual_premium)  # type: ignore[unreachable]

    def __deepcopy__(self, memo: Dict[int, Any]) -> "InsuranceAccounting":
        """Create a deep copy of this insurance accounting instance.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this InsuranceAccounting with all recoveries
        """
        import copy

        return InsuranceAccounting(
            prepaid_insurance=copy.deepcopy(self.prepaid_insurance, memo),
            monthly_expense=copy.deepcopy(self.monthly_expense, memo),
            annual_premium=copy.deepcopy(self.annual_premium, memo),
            months_in_period=self.months_in_period,
            current_month=self.current_month,
            recoveries=copy.deepcopy(self.recoveries, memo),
        )

    def pay_annual_premium(self, premium_amount: Union[Decimal, float, int]) -> Dict[str, Decimal]:
        """Record annual premium payment at start of coverage period.

        Args:
            premium_amount: Annual premium amount to pay (converted to Decimal)

        Returns:
            Dictionary with transaction details as Decimal:
                - cash_outflow: Cash paid for premium
                - prepaid_asset: Prepaid insurance asset created
                - monthly_expense: Calculated monthly expense
        """
        premium_amount = to_decimal(premium_amount)

        if premium_amount < ZERO:
            raise ValueError("Premium amount must be non-negative")

        self.annual_premium = premium_amount
        self.prepaid_insurance = premium_amount
        # Quantize monthly expense to cents to avoid accumulation errors
        self.monthly_expense = quantize_currency(premium_amount / Decimal(self.months_in_period))
        self.current_month = 0

        logger.info(f"Paid annual premium: ${premium_amount:,.2f}")
        logger.debug(f"Monthly expense will be: ${self.monthly_expense:,.2f}")

        return {
            "cash_outflow": premium_amount,
            "prepaid_asset": premium_amount,
            "monthly_expense": self.monthly_expense,
        }

    def record_monthly_expense(self) -> Dict[str, Decimal]:
        """Amortize monthly insurance expense from prepaid asset.

        Records one month of insurance expense by reducing the prepaid
        asset and recognizing the expense. Uses straight-line amortization
        over the coverage period.

        Returns:
            Dictionary with transaction details as Decimal:
                - insurance_expense: Monthly expense recognized
                - prepaid_reduction: Reduction in prepaid asset
                - remaining_prepaid: Remaining prepaid balance
        """
        # Calculate expense for this month (handle partial months at end)
        expense = min(self.monthly_expense, self.prepaid_insurance)

        # Reduce prepaid asset
        self.prepaid_insurance -= expense
        self.current_month += 1

        logger.debug(
            f"Month {self.current_month}: Expense ${expense:,.2f}, "
            f"Remaining prepaid ${self.prepaid_insurance:,.2f}"
        )

        return {
            "insurance_expense": expense,
            "prepaid_reduction": expense,
            "remaining_prepaid": self.prepaid_insurance,
        }

    def record_claim_recovery(
        self,
        recovery_amount: Union[Decimal, float, int],
        claim_id: Optional[str] = None,
        year: int = 0,
    ) -> Dict[str, Decimal]:
        """Record insurance claim recovery as receivable.

        Args:
            recovery_amount: Amount approved for recovery from insurance (converted to Decimal)
            claim_id: Optional unique identifier for the claim
            year: Year when recovery was approved

        Returns:
            Dictionary with recovery details as Decimal:
                - insurance_receivable: New receivable amount
                - total_receivables: Total outstanding receivables
        """
        recovery_amount = to_decimal(recovery_amount)

        if recovery_amount < ZERO:
            raise ValueError("Recovery amount must be non-negative")

        # Generate claim ID if not provided
        if claim_id is None:
            claim_id = f"CLAIM_{year}_{len(self.recoveries) + 1}"

        recovery = InsuranceRecovery(amount=recovery_amount, claim_id=claim_id, year_approved=year)
        self.recoveries.append(recovery)

        logger.info(f"Recorded insurance recovery: ${recovery_amount:,.2f} (ID: {claim_id})")

        return {
            "insurance_receivable": recovery_amount,
            "total_receivables": self.get_total_receivables(),
        }

    def receive_recovery_payment(
        self, amount: Union[Decimal, float, int], claim_id: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """Record receipt of insurance recovery payment.

        Args:
            amount: Amount received from insurance (converted to Decimal)
            claim_id: Optional claim ID to apply payment to

        Returns:
            Dictionary with payment details as Decimal:
                - cash_received: Cash inflow amount
                - receivable_reduction: Reduction in receivables
                - remaining_receivables: Total remaining receivables
        """
        amount = to_decimal(amount)

        if amount <= ZERO:
            raise ValueError("Payment amount must be positive")

        # Apply to specific claim or oldest outstanding
        if claim_id:
            recovery = next((r for r in self.recoveries if r.claim_id == claim_id), None)
            if not recovery:
                raise ValueError(f"No recovery found with ID {claim_id}")
        else:
            # Apply to oldest outstanding recovery
            outstanding_recoveries = [r for r in self.recoveries if r.outstanding > ZERO]
            if not outstanding_recoveries:
                raise ValueError("No outstanding recoveries to apply payment to")
            recovery = outstanding_recoveries[0]

        # Apply payment (up to outstanding amount)
        applied_amount = min(amount, recovery.outstanding)
        recovery.amount_received += applied_amount

        logger.info(f"Received recovery payment: ${applied_amount:,.2f} for {recovery.claim_id}")

        return {
            "cash_received": applied_amount,
            "receivable_reduction": applied_amount,
            "remaining_receivables": self.get_total_receivables(),
        }

    def get_total_receivables(self) -> Decimal:
        """Calculate total outstanding insurance receivables.

        Returns:
            Total amount of outstanding insurance receivables as Decimal
        """
        return sum((r.outstanding for r in self.recoveries), ZERO)

    def get_amortization_schedule(self) -> List[Dict[str, Union[int, Decimal]]]:
        """Generate remaining amortization schedule.

        Returns:
            List of monthly amortization entries remaining (amounts as Decimal)
        """
        schedule: List[Dict[str, Union[int, Decimal]]] = []
        remaining = self.prepaid_insurance
        months_left = self.months_in_period - self.current_month

        for month in range(months_left):
            month_expense = min(self.monthly_expense, remaining)
            remaining -= month_expense
            schedule.append(
                {
                    "month": self.current_month + month + 1,
                    "expense": month_expense,
                    "remaining_prepaid": remaining,
                }
            )

        return schedule

    def reset_for_new_period(self) -> None:
        """Reset accounting for a new coverage period.

        Clears current period data while preserving recoveries.
        """
        self.prepaid_insurance = ZERO
        self.monthly_expense = ZERO
        self.annual_premium = ZERO
        self.current_month = 0
        # Keep recoveries as they span multiple periods

    def get_summary(self) -> Dict[str, Union[int, Decimal]]:
        """Get summary of current insurance accounting status.

        Returns:
            Dictionary with key accounting metrics (amounts as Decimal)
        """
        return {
            "prepaid_insurance": self.prepaid_insurance,
            "monthly_expense": self.monthly_expense,
            "annual_premium": self.annual_premium,
            "months_elapsed": self.current_month,
            "months_remaining": self.months_in_period - self.current_month,
            "total_receivables": self.get_total_receivables(),
            "recovery_count": len(self.recoveries),
        }

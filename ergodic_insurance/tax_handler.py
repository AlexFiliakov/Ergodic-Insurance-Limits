"""Tax calculation, accrual, and payment logic.

This module contains the TaxHandler dataclass, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305).
"""

from dataclasses import dataclass
from decimal import Decimal
import logging
from typing import Union

try:
    from ergodic_insurance.accrual_manager import AccrualManager, AccrualType, PaymentSchedule
    from ergodic_insurance.decimal_utils import ZERO, to_decimal
except ImportError:
    try:
        from .accrual_manager import AccrualManager, AccrualType, PaymentSchedule
        from .decimal_utils import ZERO, to_decimal
    except ImportError:
        from accrual_manager import (  # type: ignore[no-redef]
            AccrualManager,
            AccrualType,
            PaymentSchedule,
        )
        from decimal_utils import ZERO, to_decimal  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass
class TaxHandler:
    """Consolidates tax calculation, accrual, and payment logic.

    This class centralizes all tax-related operations to provide clear documentation
    and prevent confusion about the tax calculation flow. The design explicitly
    addresses concerns about potential circular dependencies in the tax logic.

    Tax Flow Sequence (IMPORTANT - No Circular Dependency):
    --------------------------------------------------------
    The tax calculation follows a strict sequential flow that prevents circularity:

    1. **Read Current State**: At the start of tax calculation, we read the current
       equity value. This equity includes any PREVIOUSLY accrued taxes (from prior
       periods) but NOT the tax we are about to calculate.

    2. **Calculate Tax**: Based on income_before_tax and tax_rate, we calculate
       the theoretical tax liability.

    3. **Apply Limited Liability Cap**: If equity is insufficient to support the
       full tax liability, we cap the accrual at available equity. This protects
       against creating liabilities the company cannot support.

    4. **Record Accrual**: ONLY AFTER the equity check and cap calculation do we
       record the new tax accrual to the AccrualManager. This means the equity
       value read in step 1 was NOT affected by the accrual we're recording.

    5. **Future Payment**: The accrued tax becomes a liability on the balance sheet
       and will be paid in future periods via process_accrued_payments().

    Why This Is Not Circular:
    -------------------------
    - Equity_t is read BEFORE Tax_t is accrued
    - Tax_t is recorded AFTER Equity_t check
    - Equity_t+1 will include Tax_t, but Tax_t+1 calculation will read Equity_t+1
    - Each period's tax is based on that period's pre-accrual equity

    This is analogous to how real companies operate: they determine tax liability
    based on their financial position, then record it. The liability affects
    future equity, but doesn't retroactively change the calculation.

    Integration Points:
    -------------------
    - `calculate_net_income()`: Calls calculate_and_accrue_tax() to handle taxes
    - `process_accrued_payments()`: Pays previously accrued taxes
    - `AccrualManager`: Stores tax accruals as liabilities
    - `total_liabilities`: Includes accrued taxes from AccrualManager

    Attributes:
        tax_rate: Corporate tax rate (0.0 to 1.0)
        accrual_manager: Reference to the AccrualManager for recording accruals

    Example:
        Within WidgetManufacturer.calculate_net_income()::

            tax_handler = TaxHandler(self.tax_rate, self.accrual_manager)
            actual_tax, capped = tax_handler.calculate_and_accrue_tax(
                income_before_tax=1_000_000,
                current_equity=5_000_000,
                use_accrual=True,
                time_resolution="annual",
                current_year=2024,
                current_month=0
            )
            # actual_tax is the expense to deduct from net income
            # capped indicates if limited liability was applied

    See Also:
        :meth:`WidgetManufacturer.calculate_net_income`: Uses this handler
        :meth:`WidgetManufacturer.process_accrued_payments`: Pays accrued taxes
        :class:`AccrualManager`: Tracks tax liabilities
    """

    tax_rate: float
    accrual_manager: "AccrualManager"

    def calculate_tax_liability(self, income_before_tax: Union[Decimal, float]) -> Decimal:
        """Calculate theoretical tax liability from pre-tax income.

        This is a pure calculation with no side effects. Taxes are only
        applied to positive income; losses generate no tax benefit.

        Args:
            income_before_tax: Pre-tax income in dollars

        Returns:
            Theoretical tax liability (>=0). Returns ZERO for negative income.
        """
        income = to_decimal(income_before_tax)
        return max(ZERO, income * to_decimal(self.tax_rate))

    def apply_limited_liability_cap(
        self, tax_amount: Union[Decimal, float], current_equity: Union[Decimal, float]
    ) -> tuple[Decimal, bool]:
        """Apply limited liability protection to cap tax accrual at equity.

        Companies cannot accrue more liabilities than their equity can support.
        This method caps the tax accrual at the current equity value.

        Args:
            tax_amount: Calculated tax liability
            current_equity: Current shareholder equity

        Returns:
            Tuple of (capped_amount, was_capped):
            - capped_amount: Tax amount after applying equity cap
            - was_capped: True if the cap was applied (original > capped)
        """
        tax = to_decimal(tax_amount)
        equity = to_decimal(current_equity)
        if equity <= ZERO:
            return ZERO, tax > ZERO

        capped_amount = min(tax, equity)
        was_capped = capped_amount < tax

        return capped_amount, was_capped

    def record_tax_accrual(
        self,
        amount: Union[Decimal, float],
        time_resolution: str,
        current_year: int,
        current_month: int,
        description: str = "",
    ) -> None:
        """Record tax accrual to the AccrualManager.

        This method records the tax liability and sets up the payment schedule.
        For annual resolution, taxes are accrued with quarterly payment dates
        in the following year. For monthly resolution, quarterly taxes are
        accrued at quarter-end for more immediate payment.

        Args:
            amount: Tax amount to accrue
            time_resolution: "annual" or "monthly"
            current_year: Current simulation year
            current_month: Current simulation month (0-11)
            description: Optional description for the accrual
        """
        amount_decimal = to_decimal(amount)
        if amount_decimal <= ZERO:
            return

        if time_resolution == "annual":
            # Annual taxes are paid quarterly in the NEXT year
            next_year_base = (current_year + 1) * 12
            payment_dates = [next_year_base + month for month in [3, 5, 8, 11]]

            self.accrual_manager.record_expense_accrual(
                item_type=AccrualType.TAXES,
                amount=amount_decimal,
                payment_schedule=PaymentSchedule.CUSTOM,
                payment_dates=payment_dates,
                description=description or f"Year {current_year} tax liability",
            )
        else:
            # Monthly mode: use default quarterly schedule
            self.accrual_manager.record_expense_accrual(
                item_type=AccrualType.TAXES,
                amount=amount_decimal,
                payment_schedule=PaymentSchedule.QUARTERLY,
                description=description,
            )

    def calculate_and_accrue_tax(
        self,
        income_before_tax: Union[Decimal, float],
        current_equity: Union[Decimal, float],
        use_accrual: bool,
        time_resolution: str,
        current_year: int,
        current_month: int,
    ) -> tuple[Decimal, bool]:
        """Calculate tax and optionally accrue it - the main entry point.

        This method orchestrates the complete tax calculation flow:
        1. Calculate theoretical tax
        2. Apply limited liability cap based on current equity
        3. Record accrual if enabled and timing is appropriate

        IMPORTANT: The current_equity parameter should be the equity value
        BEFORE this tax is accrued. This ensures no circular dependency.

        Args:
            income_before_tax: Pre-tax income in dollars
            current_equity: Current equity (BEFORE this tax accrual)
            use_accrual: Whether to use accrual accounting
            time_resolution: "annual" or "monthly"
            current_year: Current simulation year
            current_month: Current simulation month (0-11)

        Returns:
            Tuple of (actual_tax_expense, was_capped):
            - actual_tax_expense: Tax expense for net income calculation
            - was_capped: True if limited liability cap was applied

        Example:
            actual_tax, capped = handler.calculate_and_accrue_tax(
                income_before_tax=1_000_000,
                current_equity=5_000_000,
                use_accrual=True,
                time_resolution="annual",
                current_year=2024,
                current_month=0
            )
        """
        # Step 1: Calculate theoretical tax
        theoretical_tax = self.calculate_tax_liability(income_before_tax)

        if theoretical_tax <= ZERO:
            return ZERO, False

        # Step 2: Determine if this period should accrue taxes
        should_accrue = False
        description = ""

        if use_accrual:
            if time_resolution == "annual":
                should_accrue = True
                description = f"Year {current_year} tax liability"
            elif time_resolution == "monthly" and current_month in [2, 5, 8, 11]:
                should_accrue = True
                quarter = (current_month // 3) + 1
                description = f"Year {current_year} Q{quarter} tax liability"

        if not should_accrue:
            # No accrual needed - return full tax as expense
            return theoretical_tax, False

        # Step 3: Apply limited liability cap
        # This is where we read current_equity to cap the accrual
        capped_tax, was_capped = self.apply_limited_liability_cap(theoretical_tax, current_equity)

        if was_capped:
            logger.warning(
                f"LIMITED LIABILITY: Cannot accrue full tax liability of ${theoretical_tax:,.2f}. "
                f"Equity only ${current_equity:,.2f}. Accruing ${capped_tax:,.2f}."
            )

        # Step 4: Record the accrual (AFTER equity check)
        self.record_tax_accrual(
            amount=capped_tax,
            time_resolution=time_resolution,
            current_year=current_year,
            current_month=current_month,
            description=description,
        )

        return capped_tax, was_capped

"""Tax calculation, accrual, and payment logic.

This module contains the TaxHandler dataclass, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305).

Issue #365: Added NOL carryforward tracking per ASC 740 / IRC §172.
Issue #464: Added DTA valuation allowance per ASC 740-10-30-5.
"""

from dataclasses import dataclass, field
from decimal import Decimal
import logging
from typing import Dict, Union

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
       the theoretical tax liability, applying any available NOL carryforward
       per IRC §172 (Issue #365).

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

    NOL Carryforward (Issue #365):
    ------------------------------
    Per ASC 740-10-25-3 and IRC §172, net operating losses create deferred tax
    assets that offset future tax liabilities. Post-TCJA rules apply:
    - NOLs carry forward indefinitely (no carryback)
    - NOL deduction limited to 80% of taxable income per IRC §172(a)(2)
    - DTA = cumulative unused NOL x enacted tax rate

    Valuation Allowance (Issue #464):
    ---------------------------------
    Per ASC 740-10-30-5, a valuation allowance reduces the DTA when it is
    "more likely than not" (>50% probability) that some or all of the DTA
    will not be realized. This implementation uses consecutive loss years as
    negative evidence:
    - < 3 consecutive loss years: 0% allowance
    - 3 consecutive loss years: 50% allowance
    - 4 consecutive loss years: 75% allowance
    - 5+ consecutive loss years: 100% allowance
    The allowance reverses when the company returns to profitability.

    Integration Points:
    -------------------
    - ``calculate_net_income()``: Calls calculate_and_accrue_tax() to handle taxes
    - ``process_accrued_payments()``: Pays previously accrued taxes
    - ``AccrualManager``: Stores tax accruals as liabilities
    - ``total_liabilities``: Includes accrued taxes from AccrualManager

    TCJA Applicability (Issue #808):
    --------------------------------
    Per IRC §172(a)(2) as amended by the Tax Cuts and Jobs Act (TCJA):
    - NOLs from tax years beginning **after December 31, 2017**: Limited to 80%
      of taxable income (``nol_limitation_pct``), but carry forward indefinitely.
    - NOLs from tax years beginning **before January 1, 2018**: No percentage
      limitation (100% deduction allowed), but subject to 20-year carryforward
      expiration (expiration not modeled here).

    Since simulations typically start in recent years, the post-TCJA 80% limitation
    is applied by default (``apply_tcja_limitation=True``). Set to ``False`` to model
    pre-2018 NOL rules where the full NOL can offset 100% of taxable income.

    Attributes:
        tax_rate: Corporate tax rate (0.0 to 1.0)
        accrual_manager: Reference to the AccrualManager for recording accruals
        nol_carryforward: Cumulative unused NOL per IRC §172
        nol_limitation_pct: 80% limitation per IRC §172(a)(2), post-TCJA
        apply_tcja_limitation: When True (default), applies 80% NOL deduction
            limitation per IRC §172(a)(2). Set to False for pre-TCJA modeling
            where NOLs can offset 100% of taxable income.
        consecutive_loss_years: Count of consecutive loss years for valuation allowance

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
    nol_carryforward: Decimal = field(default_factory=lambda: Decimal("0"))
    nol_limitation_pct: float = 0.80
    apply_tcja_limitation: bool = True
    tax_accumulated_depreciation: Decimal = field(default_factory=lambda: Decimal("0"))
    consecutive_loss_years: int = 0

    # Valuation allowance thresholds per ASC 740-10-30-5 (Issue #464)
    _VA_THRESHOLD: int = 3  # Consecutive loss years to trigger allowance
    _VA_RATES: Dict[int, Decimal] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize graduated valuation allowance rate schedule."""
        self._VA_RATES = {3: Decimal("0.50"), 4: Decimal("0.75")}
        # 5+ years: 100% (handled by default in valuation_allowance_rate)

    @property
    def deferred_tax_asset(self) -> Decimal:
        """Gross deferred tax asset from NOL carryforward per ASC 740-10-25-3.

        DTA = cumulative unused NOL x enacted tax rate.
        This is the gross DTA before any valuation allowance.
        """
        return self.nol_carryforward * to_decimal(self.tax_rate)

    @property
    def valuation_allowance_rate(self) -> Decimal:
        """Valuation allowance rate based on consecutive loss years.

        Per ASC 740-10-30-5, graduated based on negative evidence:
        - < 3 years: 0% (sufficient positive evidence)
        - 3 years: 50% (more likely than not partial non-realization)
        - 4 years: 75% (significant doubt about realization)
        - 5+ years: 100% (full allowance, sustained losses)
        """
        if self.consecutive_loss_years < self._VA_THRESHOLD:
            return ZERO
        rate: Decimal = self._VA_RATES.get(self.consecutive_loss_years, to_decimal("1.00"))
        return rate

    @property
    def valuation_allowance(self) -> Decimal:
        """Valuation allowance amount per ASC 740-10-30-5.

        Reduces the gross DTA when realization is not more likely than not.
        """
        return self.deferred_tax_asset * self.valuation_allowance_rate

    @property
    def net_deferred_tax_asset(self) -> Decimal:
        """Net DTA after valuation allowance per ASC 740-10-30-5.

        Net DTA = Gross DTA - Valuation Allowance.
        This is the amount reported on the balance sheet.
        """
        return self.deferred_tax_asset - self.valuation_allowance

    def calculate_tax_liability(
        self, income_before_tax: Union[Decimal, float]
    ) -> tuple[Decimal, Decimal]:
        """Calculate tax liability with NOL carryforward per IRC §172.

        For negative income: accumulates the loss into nol_carryforward (no tax due).
        For positive income: applies available NOL carryforward subject to the
        80% limitation per IRC §172(a)(2) before computing tax.

        Args:
            income_before_tax: Pre-tax income (may be negative).

        Returns:
            Tuple of (tax_liability, nol_utilized):
            - tax_liability: Tax due after NOL offset (>= 0).
            - nol_utilized: Amount of NOL consumed this period.
        """
        income = to_decimal(income_before_tax)

        if income <= ZERO:
            # Loss year: accumulate NOL per IRC §172(b)(1)(A)(ii)
            self.nol_carryforward += abs(income)
            # Track consecutive loss years for valuation allowance (Issue #464)
            self.consecutive_loss_years += 1
            return ZERO, ZERO

        # Profit year: reset consecutive loss counter (Issue #464)
        self.consecutive_loss_years = 0

        if self.nol_carryforward <= ZERO:
            # No NOL available — standard tax
            return max(ZERO, income * to_decimal(self.tax_rate)), ZERO

        # Apply NOL deduction limitation per IRC §172(a)(2):
        # Post-TCJA (apply_tcja_limitation=True): limited to nol_limitation_pct (80%)
        # Pre-TCJA (apply_tcja_limitation=False): 100% of taxable income (no limit)
        if self.apply_tcja_limitation:
            max_nol_deduction = income * to_decimal(self.nol_limitation_pct)
        else:
            max_nol_deduction = income  # Pre-TCJA: no percentage limitation
        nol_utilized = min(self.nol_carryforward, max_nol_deduction)

        taxable_income = income - nol_utilized
        self.nol_carryforward -= nol_utilized

        tax = max(ZERO, taxable_income * to_decimal(self.tax_rate))
        return tax, nol_utilized

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
    ) -> tuple[Decimal, bool, Decimal]:
        """Calculate tax and optionally accrue it - the main entry point.

        This method orchestrates the complete tax calculation flow:
        1. Calculate theoretical tax (with NOL offset per IRC §172)
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
            Tuple of (actual_tax_expense, was_capped, nol_utilized):
            - actual_tax_expense: Tax expense for net income calculation
            - was_capped: True if limited liability cap was applied
            - nol_utilized: Amount of NOL consumed this period

        Example:
            actual_tax, capped, nol_used = handler.calculate_and_accrue_tax(
                income_before_tax=1_000_000,
                current_equity=5_000_000,
                use_accrual=True,
                time_resolution="annual",
                current_year=2024,
                current_month=0
            )
        """
        # Step 1: Calculate theoretical tax (with NOL offset)
        theoretical_tax, nol_utilized = self.calculate_tax_liability(income_before_tax)

        if nol_utilized > ZERO:
            logger.info(
                f"NOL utilization: ${nol_utilized:,.2f} offset against income. "
                f"Remaining NOL carryforward: ${self.nol_carryforward:,.2f}"
            )

        if theoretical_tax <= ZERO:
            return ZERO, False, nol_utilized

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
            return theoretical_tax, False, nol_utilized

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

        return capped_tax, was_capped, nol_utilized

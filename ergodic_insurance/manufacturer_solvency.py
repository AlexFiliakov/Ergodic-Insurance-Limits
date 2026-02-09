# mypy: disable-error-code="attr-defined, has-type, no-any-return"
"""Solvency mixin for WidgetManufacturer.

This module contains the SolvencyMixin class, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305). It provides solvency checking,
insolvency handling, liquidity constraints, and minimum cash estimation methods.
"""

from decimal import Decimal
import logging
from typing import List, Optional, Tuple

try:
    from ergodic_insurance.decimal_utils import ZERO, to_decimal
except ImportError:
    try:
        from .decimal_utils import ZERO, to_decimal
    except ImportError:
        from decimal_utils import ZERO, to_decimal  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class SolvencyMixin:
    """Mixin providing solvency checking and insolvency handling methods.

    This mixin expects the host class to have:
        - self.equity: Decimal (property from BalanceSheetMixin)
        - self.total_assets: Decimal (property from BalanceSheetMixin)
        - self.cash: Decimal (property from BalanceSheetMixin)
        - self.restricted_assets: Decimal (property from BalanceSheetMixin)
        - self.total_claim_liabilities: Decimal (property from ClaimProcessingMixin)
        - self.claim_liabilities: List[ClaimLiability]
        - self.config: ManufacturerConfig instance
        - self.current_year: int
        - self.current_month: int
        - self.is_ruined: bool
        - self.ruin_month: Optional[int]
        - self.ledger: Ledger instance
        - self.base_operating_margin: float
        - self.tax_rate: float
        - self.calculate_revenue(): method from IncomeCalculationMixin
        - self._record_liquidation(): method from BalanceSheetMixin
    """

    # Declare ruin_month type to match WidgetManufacturer.__init__ (Optional[int])
    ruin_month: Optional[int]

    def handle_insolvency(self) -> None:
        """Handle insolvency by enforcing zero equity floor and freezing operations.

        Implements standard bankruptcy accounting with limited liability:
        - Sets equity floor at $0 (never negative)
        - Marks company as insolvent (is_ruined = True)
        - Keeps unpayable liabilities on the books
        - Freezes all further business operations
        """
        if self.equity < ZERO:
            logger.info(
                f"Equity is negative (${self.equity:,.2f}) — limited liability applies. "
                f"Creditors absorb ${-self.equity:,.2f} in losses."
            )

        if not self.is_ruined:
            self.is_ruined = True
            total_liabilities = self.total_claim_liabilities
            pre_liquidation_assets = self.total_assets

            logger.warning(
                f"INSOLVENCY: Company is now insolvent. "
                f"Equity: ${self.equity:,.2f}, "
                f"Assets: ${self.total_assets:,.2f}, "
                f"Liabilities: ${total_liabilities:,.2f}, "
                f"Unpayable debt: ${max(0, total_liabilities - self.total_assets):,.2f}"
            )

            insolvency_tolerance = to_decimal(self.config.insolvency_tolerance)
            if self.equity > insolvency_tolerance:
                liquidation_loss = self.cash - insolvency_tolerance
                if liquidation_loss > ZERO:
                    self._record_liquidation(
                        amount=liquidation_loss,
                        description="Bankruptcy liquidation costs and asset haircuts",
                    )
                    logger.info(
                        f"LIQUIDATION: Assets reduced from ${pre_liquidation_assets:,.2f} "
                        f"to ${self.total_assets:,.2f} due to bankruptcy liquidation costs"
                    )

    def check_solvency(self) -> bool:
        """Check if the company is solvent and update ruin status.

        Returns:
            bool: True if company is solvent, False if insolvent.
        """
        # Per ASC 470-10 (Issue #496), negative cash represents a draw on the
        # working capital facility and is reclassified as short-term borrowings.
        # It is NOT an insolvency signal — solvency is determined by equity below.
        if self.cash < ZERO:
            logger.info(
                f"Working capital facility in use: cash balance ${self.cash:,.2f}. "
                f"Reclassified as ${-self.cash:,.2f} short-term borrowings (ASC 470-10)."
            )

        # Use operational equity for solvency — add back valuation allowance
        # since it's a non-cash accounting adjustment that doesn't affect the
        # company's ability to continue operations (Issue #464)
        va = getattr(self, "dta_valuation_allowance", ZERO)
        operational_equity = self.equity + va
        if operational_equity <= ZERO:
            self.handle_insolvency()
            return False

        # Payment insolvency check
        if self.claim_liabilities:
            current_year_payments: Decimal = ZERO
            for claim_item in self.claim_liabilities:
                years_since = self.current_year - claim_item.year_incurred
                scheduled_payment = claim_item.get_payment(years_since)
                current_year_payments += scheduled_payment

            if current_year_payments > ZERO:
                current_revenue = self.calculate_revenue()
                payment_burden_ratio = (
                    current_year_payments / current_revenue
                    if current_revenue > ZERO
                    else to_decimal(float("inf"))
                )

                if payment_burden_ratio > to_decimal(0.80):
                    if not self.is_ruined:
                        self.is_ruined = True
                        logger.warning(
                            f"Company became insolvent - unsustainable payment burden: "
                            f"${current_year_payments:,.0f} payments vs ${current_revenue:,.0f} revenue "
                            f"({payment_burden_ratio:.1%} burden ratio)"
                        )
                    return False

        return True

    def estimate_minimum_cash_point(self, time_resolution: str = "annual") -> Tuple[Decimal, int]:
        """Estimate the minimum cash point within the current period.

        Args:
            time_resolution: Time step resolution ("annual" or "monthly").

        Returns:
            Tuple[Decimal, int]: (min_cash, min_month)
        """
        if time_resolution == "monthly":
            return self.cash, self.current_month

        premium_payment_month = getattr(self.config, "premium_payment_month", 0)
        revenue_pattern = getattr(self.config, "revenue_pattern", "uniform")

        annual_revenue = self.calculate_revenue()
        annual_premium = getattr(self, "period_insurance_premiums", ZERO)

        operating_margin = to_decimal(self.base_operating_margin)
        estimated_annual_income = annual_revenue * operating_margin
        annual_tax = estimated_annual_income * to_decimal(self.tax_rate)
        quarterly_tax = annual_tax / to_decimal(4)

        monthly_revenues = self._get_monthly_revenue_distribution(annual_revenue, revenue_pattern)

        cash_balance = self.cash
        min_cash = cash_balance
        min_month = 0

        tax_months = [3, 5, 8, 11]

        for month in range(12):
            if month == premium_payment_month:
                cash_balance -= annual_premium
            if month in tax_months:
                cash_balance -= quarterly_tax

            cash_balance += monthly_revenues[month]

            if cash_balance < min_cash:
                min_cash = cash_balance
                min_month = month

        return min_cash, min_month

    def _get_monthly_revenue_distribution(
        self, annual_revenue: Decimal, pattern: str
    ) -> List[Decimal]:
        """Get monthly revenue distribution based on configured pattern.

        Args:
            annual_revenue: Total annual revenue to distribute.
            pattern: Distribution pattern ("uniform", "seasonal", "back_loaded").

        Returns:
            List[Decimal]: Monthly revenues (12 elements, one per month).
        """
        if pattern == "uniform":
            monthly = annual_revenue / to_decimal(12)
            return [monthly] * 12
        if pattern == "seasonal":
            q1_q3_monthly = (annual_revenue * to_decimal(0.60)) / to_decimal(9)
            q4_monthly = (annual_revenue * to_decimal(0.40)) / to_decimal(3)
            return [q1_q3_monthly] * 9 + [q4_monthly] * 3
        if pattern == "back_loaded":
            h1_monthly = (annual_revenue * to_decimal(0.40)) / to_decimal(6)
            h2_monthly = (annual_revenue * to_decimal(0.60)) / to_decimal(6)
            return [h1_monthly] * 6 + [h2_monthly] * 6
        # Default: uniform distribution
        monthly = annual_revenue / to_decimal(12)
        return [monthly] * 12

    def check_liquidity_constraints(self, time_resolution: str = "annual") -> bool:
        """Check if the company maintains positive liquidity throughout the period.

        Args:
            time_resolution: Time step resolution ("annual" or "monthly").

        Returns:
            bool: True if liquidity constraints are met, False if mid-year insolvency.
        """
        if not getattr(self.config, "check_intra_period_liquidity", True):
            return True

        if self.is_ruined:
            return False

        min_cash, min_month = self.estimate_minimum_cash_point(time_resolution)

        if min_cash < ZERO:
            self.is_ruined = True
            self.ruin_month = min_month
            logger.warning(
                f"MID-YEAR INSOLVENCY: Company would become insolvent in month {min_month} "
                f"with estimated cash of ${min_cash:,.2f}. Year {self.current_year}, "
                f"premium payment month: {getattr(self.config, 'premium_payment_month', 0)}"
            )
            return False

        return True

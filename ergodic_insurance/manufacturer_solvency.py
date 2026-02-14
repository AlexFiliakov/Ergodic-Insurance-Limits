# mypy: disable-error-code="attr-defined, has-type, no-any-return"
"""Solvency mixin for WidgetManufacturer.

This module contains the SolvencyMixin class, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305). It provides solvency checking,
insolvency handling, liquidity constraints, and minimum cash estimation methods.

Going concern assessment follows ASC 205-40 multi-factor approach (Issue #489).
"""

from decimal import Decimal
import logging
from typing import Dict, List, Optional, Tuple

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
        - self.total_liabilities: Decimal (property from BalanceSheetMixin)
        - self.cash: Decimal (property from BalanceSheetMixin)
        - self.accounts_receivable: Decimal (property from BalanceSheetMixin)
        - self.inventory: Decimal (property from BalanceSheetMixin)
        - self.prepaid_insurance: Decimal (property from BalanceSheetMixin)
        - self.accounts_payable: Decimal (property from BalanceSheetMixin)
        - self.short_term_borrowings: Decimal (property from BalanceSheetMixin)
        - self.deferred_tax_liability: Decimal (property from BalanceSheetMixin)
        - self.restricted_assets: Decimal (property from BalanceSheetMixin)
        - self.gross_ppe: Decimal (property from BalanceSheetMixin)
        - self.accumulated_depreciation: Decimal (property from BalanceSheetMixin)
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

    @property
    def solvency_equity(self) -> Decimal:
        """Operational equity used for all solvency and going-concern assessments.

        Returns book equity with the DTA valuation allowance added back.
        The valuation allowance (ASC 740-10-30-5) is a non-cash accounting
        adjustment that reduces the deferred tax asset when realization is
        uncertain. It does not impair the company's ability to continue
        operations, so it is excluded from solvency determinations.

        Per ASC 205-40-50-7, going concern assessment must use a single,
        consistent equity definition throughout. This property is the single
        source of truth for equity in check_solvency(), compute_z_prime_score(),
        _assess_going_concern_indicators(), and calculate_metrics() (Issue #1311).

        Returns:
            Decimal: Operational equity (book equity + valuation allowance).
        """
        va = getattr(self, "dta_valuation_allowance", to_decimal(0))
        return self.equity + va

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

    def _assess_going_concern_indicators(self) -> List[Dict]:
        """Assess going concern indicators per ASC 205-40 (Issue #489).

        Evaluates four financial indicators against configurable thresholds
        to determine whether substantial doubt exists about the entity's
        ability to continue as a going concern.

        Returns:
            List of dicts, each with keys: name, value, threshold, breached.
        """
        indicators: List[Dict] = []
        revenue = self.calculate_revenue()

        # 1. Current Ratio = Current Assets / Current Liabilities
        reported_cash = max(self.cash, to_decimal(0))
        current_assets = (
            reported_cash + self.accounts_receivable + self.inventory + self.prepaid_insurance
        )
        # Current liabilities = total_liabilities - long-term claims - DTL
        claim_total = sum(
            (liability.remaining_amount for liability in self.claim_liabilities), to_decimal(0)
        )
        dtl = getattr(self, "deferred_tax_liability", to_decimal(0))
        current_liabilities = self.total_liabilities - claim_total - dtl
        if current_liabilities > ZERO:
            current_ratio = current_assets / current_liabilities
            threshold = to_decimal(self.config.going_concern_min_current_ratio)
            indicators.append(
                {
                    "name": "Current Ratio",
                    "value": current_ratio,
                    "threshold": threshold,
                    "breached": current_ratio < threshold,
                }
            )
        else:
            # No current liabilities — indicator not breached
            indicators.append(
                {
                    "name": "Current Ratio",
                    "value": to_decimal("Inf"),
                    "threshold": to_decimal(self.config.going_concern_min_current_ratio),
                    "breached": False,
                }
            )

        # 2. DSCR = Operating Income / Debt Service (current year claim payments)
        current_year_payments: Decimal = to_decimal(0)
        for claim_item in self.claim_liabilities:
            years_since = self.current_year - claim_item.year_incurred
            current_year_payments += claim_item.get_payment(years_since)

        if current_year_payments > ZERO:
            operating_income = revenue * to_decimal(self.base_operating_margin)
            dscr = (
                operating_income / current_year_payments if current_year_payments > ZERO else ZERO
            )
            threshold = to_decimal(self.config.going_concern_min_dscr)
            indicators.append(
                {
                    "name": "DSCR",
                    "value": dscr,
                    "threshold": threshold,
                    "breached": dscr < threshold,
                }
            )
        else:
            # No debt service obligations — indicator not breached
            indicators.append(
                {
                    "name": "DSCR",
                    "value": to_decimal("Inf"),
                    "threshold": to_decimal(self.config.going_concern_min_dscr),
                    "breached": False,
                }
            )

        # 3. Equity Ratio = Solvency Equity / Total Assets (Issue #1311)
        if self.total_assets > ZERO:
            equity_ratio = self.solvency_equity / self.total_assets
            threshold = to_decimal(self.config.going_concern_min_equity_ratio)
            indicators.append(
                {
                    "name": "Equity Ratio",
                    "value": equity_ratio,
                    "threshold": threshold,
                    "breached": equity_ratio < threshold,
                }
            )
        else:
            indicators.append(
                {
                    "name": "Equity Ratio",
                    "value": ZERO,
                    "threshold": to_decimal(self.config.going_concern_min_equity_ratio),
                    "breached": True,
                }
            )

        # 4. Cash Runway = Cash / Monthly Operating Expenses
        monthly_opex = (
            revenue * (to_decimal(1) - to_decimal(self.base_operating_margin)) / to_decimal(12)
        )
        if monthly_opex > ZERO:
            cash_runway = self.cash / monthly_opex
            threshold = to_decimal(self.config.going_concern_min_cash_runway_months)
            indicators.append(
                {
                    "name": "Cash Runway",
                    "value": cash_runway,
                    "threshold": threshold,
                    "breached": cash_runway < threshold,
                }
            )
        else:
            # No operating expenses — indicator not breached
            indicators.append(
                {
                    "name": "Cash Runway",
                    "value": to_decimal("Inf"),
                    "threshold": to_decimal(self.config.going_concern_min_cash_runway_months),
                    "breached": False,
                }
            )

        return indicators

    def compute_z_prime_score(self) -> Decimal:
        """Compute Altman Z-prime Score (private company variant) as a diagnostic metric.

        Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.42*X4 + 0.998*X5

        Where:
            X1 = Working Capital / Total Assets
            X2 = Solvency Equity / Total Assets (retained earnings proxy, Issue #1311)
            X3 = EBIT / Total Assets
            X4 = Solvency Equity / Total Liabilities (Issue #1311)
            X5 = Sales / Total Assets

        Returns:
            Decimal: Z-prime score. < 1.23 = distress, 1.23-2.90 = grey, > 2.90 = safe.
        """
        total_assets = self.total_assets
        if total_assets <= ZERO:
            return to_decimal(0)

        total_liabilities = self.total_liabilities
        revenue = self.calculate_revenue()

        # Current assets and current liabilities for working capital
        reported_cash = max(self.cash, to_decimal(0))
        current_assets = (
            reported_cash + self.accounts_receivable + self.inventory + self.prepaid_insurance
        )
        claim_total = sum(
            (liability.remaining_amount for liability in self.claim_liabilities), to_decimal(0)
        )
        dtl = getattr(self, "deferred_tax_liability", to_decimal(0))
        current_liabilities = total_liabilities - claim_total - dtl

        working_capital = current_assets - current_liabilities
        ebit = revenue * to_decimal(self.base_operating_margin)

        x1 = working_capital / total_assets
        x2 = self.solvency_equity / total_assets  # Retained earnings proxy (Issue #1311)
        x3 = ebit / total_assets
        x4 = (
            self.solvency_equity / total_liabilities if total_liabilities > ZERO else to_decimal(10)
        )  # Issue #1311
        x5 = revenue / total_assets

        z_prime = (
            to_decimal("0.717") * x1
            + to_decimal("0.847") * x2
            + to_decimal("3.107") * x3
            + to_decimal("0.42") * x4
            + to_decimal("0.998") * x5
        )
        return z_prime

    def check_solvency(self) -> bool:
        """Check if the company is solvent using ASC 205-40 going concern assessment.

        Implements a two-tier assessment (Issue #489):

        Tier 1 (Hard Stops): Non-configurable checks for unambiguous insolvency.
            - Balance sheet insolvency: operational equity <= 0

        Tier 2 (Multi-Factor): Configurable going concern indicators per ASC 205-40.
            Insolvency triggers when N or more indicators are simultaneously breached.
            - Current Ratio < threshold (default 1.0)
            - DSCR < threshold (default 1.0)
            - Equity Ratio < threshold (default 5%)
            - Cash Runway < threshold (default 3 months)

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

        # --- Tier 1: Hard stops (non-configurable) ---

        # Use solvency_equity (operational equity) — see property docstring
        # for GAAP justification (ASC 205-40-50-7, Issue #464, #1311)
        if self.solvency_equity <= ZERO:
            self.handle_insolvency()
            return False

        # --- Tier 2: Multi-factor going concern assessment (ASC 205-40, Issue #489) ---

        indicators = self._assess_going_concern_indicators()
        breached = [ind for ind in indicators if ind["breached"]]
        breached_count = len(breached)
        min_required = self.config.going_concern_min_indicators_breached

        # Log indicator detail for diagnostics
        if breached_count > 0:
            detail_parts = []
            for ind in indicators:
                status = "BREACH" if ind["breached"] else "OK"
                # Handle special Inf values for display
                if isinstance(ind["value"], Decimal):
                    try:
                        val_str = f"{ind['value']:.2f}"
                    except Exception:
                        val_str = str(ind["value"])
                else:
                    val_str = str(ind["value"])
                detail_parts.append(
                    f"{ind['name']}: {val_str} (min {ind['threshold']:.2f}) [{status}]"
                )
            detail = ", ".join(detail_parts)

            # Compute Z-prime score as diagnostic when any indicators are breached
            z_prime = self.compute_z_prime_score()
            z_zone = (
                "DISTRESS"
                if z_prime < to_decimal("1.23")
                else "GREY" if z_prime < to_decimal("2.90") else "SAFE"
            )

            logger.info(
                f"Going concern assessment: {breached_count}/{len(indicators)} "
                f"indicators breached (trigger at {min_required}). {detail}. "
                f"Z-prime: {z_prime:.2f} [{z_zone}]"
            )

        if breached_count >= min_required:
            if not self.is_ruined:
                self.is_ruined = True
                breached_names = [ind["name"] for ind in breached]
                logger.warning(
                    f"GOING CONCERN: {breached_count} of {len(indicators)} indicators "
                    f"breached (threshold: {min_required}). "
                    f"Breached: {', '.join(breached_names)}. "
                    f"Company deemed insolvent per ASC 205-40 multi-factor assessment."
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
        annual_premium = getattr(self, "period_insurance_premiums", to_decimal(0))

        operating_margin = to_decimal(self.base_operating_margin)
        estimated_annual_income = annual_revenue * operating_margin

        # Account for NOL carryforward in tax estimate per IRC §172 (Issue #689)
        tax_handler = getattr(self, "tax_handler", None)
        if (
            tax_handler is not None
            and hasattr(tax_handler, "nol_carryforward")
            and tax_handler.nol_carryforward > ZERO
            and estimated_annual_income > ZERO
        ):
            nol_limit_pct = to_decimal(getattr(tax_handler, "nol_limitation_pct", 0.80))
            max_nol_deduction = estimated_annual_income * nol_limit_pct
            nol_deduction = min(tax_handler.nol_carryforward, max_nol_deduction)
            taxable_income = estimated_annual_income - nol_deduction
        else:
            taxable_income = max(to_decimal(0), estimated_annual_income)

        annual_tax = taxable_income * to_decimal(self.tax_rate)
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

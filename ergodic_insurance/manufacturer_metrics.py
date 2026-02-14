# mypy: disable-error-code="attr-defined, has-type, no-any-return"
"""Metrics calculation mixin for WidgetManufacturer.

This module contains the MetricsCalculationMixin class, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305). It provides the calculate_metrics
method for comprehensive financial metric generation.
"""

from decimal import Decimal
import logging
from typing import Optional, Union

try:
    from ergodic_insurance.decimal_utils import ONE, ZERO, MetricsDict, to_decimal
except ImportError:
    try:
        from .decimal_utils import ONE, ZERO, MetricsDict, to_decimal
    except ImportError:
        from decimal_utils import ONE, ZERO, MetricsDict, to_decimal  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class MetricsCalculationMixin:
    """Mixin providing financial metrics calculation methods.

    This mixin expects the host class to have:
        - Balance sheet properties from BalanceSheetMixin
        - Income calculation methods from IncomeCalculationMixin
        - Claim properties from ClaimProcessingMixin
        - self.config: ManufacturerConfig instance
        - self.is_ruined: bool
        - self.accrual_manager: AccrualManager instance
        - self.period_insurance_premiums: Decimal
        - self.period_insurance_losses: Decimal
        - self._last_dividends_paid: Decimal
        - self.base_operating_margin: float
    """

    def calculate_metrics(
        self,
        period_revenue: Optional[Union[Decimal, float]] = None,
        letter_of_credit_rate: Union[Decimal, float] = 0.015,
    ) -> MetricsDict:
        """Calculate comprehensive financial metrics for analysis.

        Args:
            period_revenue (Optional[float]): Actual revenue earned during the period.
            letter_of_credit_rate (float): Annual interest rate for letter of credit.

        Returns:
            Dict[str, float]: Comprehensive metrics dictionary.
        """
        metrics: MetricsDict = {}

        # Basic balance sheet metrics
        metrics["assets"] = self.total_assets
        metrics["collateral"] = self.collateral
        metrics["restricted_assets"] = self.restricted_assets
        metrics["available_assets"] = self.available_assets
        # Report solvency equity — consistent with check_solvency() and
        # compute_z_prime_score() per ASC 205-40-50-7 (Issue #464, #1311)
        metrics["equity"] = self.solvency_equity
        metrics["net_assets"] = self.net_assets
        metrics["claim_liabilities"] = self.total_claim_liabilities

        # Current/non-current claim split from development schedules (Issue #466, ASC 450)
        current_claims = to_decimal(0)
        for claim in self.claim_liabilities:
            years_since = self.current_year - claim.year_incurred
            next_year_payment = claim.get_payment(years_since + 1)
            current_claims += min(next_year_payment, claim.remaining_amount)
        metrics["current_claim_liabilities"] = current_claims
        metrics["non_current_claim_liabilities"] = self.total_claim_liabilities - current_claims

        metrics["is_solvent"] = not self.is_ruined

        # Enhanced balance sheet components
        # Per ASC 210-10-45-1 (Issue #496), negative cash is reclassified as
        # short-term borrowings for presentation purposes.
        metrics["cash"] = max(self.cash, to_decimal(0))
        metrics["accounts_receivable"] = self.accounts_receivable
        metrics["inventory"] = self.inventory
        metrics["prepaid_insurance"] = self.prepaid_insurance
        metrics["insurance_receivables"] = self.insurance_receivables  # Issue #814
        metrics["accounts_payable"] = self.accounts_payable
        metrics["short_term_borrowings"] = self.short_term_borrowings

        # Accrual breakdown from AccrualManager
        accrual_items = self.accrual_manager.get_balance_sheet_items()
        metrics["accrued_expenses"] = accrual_items.get("accrued_expenses", to_decimal(0))

        metrics["gross_ppe"] = self.gross_ppe
        metrics["accumulated_depreciation"] = self.accumulated_depreciation
        metrics["net_ppe"] = self.net_ppe

        # Detailed accrual breakdown
        metrics["accrued_wages"] = accrual_items.get("accrued_wages", to_decimal(0))
        metrics["accrued_taxes"] = accrual_items.get("accrued_taxes", to_decimal(0))
        metrics["accrued_interest"] = accrual_items.get("accrued_interest", to_decimal(0))
        metrics["accrued_revenues"] = accrual_items.get("accrued_revenues", to_decimal(0))

        # Calculate operating metrics
        revenue = (
            to_decimal(period_revenue) if period_revenue is not None else self.calculate_revenue()
        )
        annual_depreciation = (
            self.gross_ppe / to_decimal(10) if self.gross_ppe > ZERO else to_decimal(0)
        )
        operating_income = self.calculate_operating_income(revenue)
        collateral_costs = self.calculate_collateral_costs(letter_of_credit_rate, "annual")

        # Use cached net income from step() to avoid double-firing tax calculation,
        # which would mutate NOL state and record duplicate DTA journal entries (Issue #617)
        cached = getattr(self, "_period_net_income", None)
        if cached is not None:
            net_income = cached
        else:
            logger.warning(
                "calculate_metrics() called without prior step() — "
                "computing net income directly (tax state will be mutated)"
            )
            net_income = self.calculate_net_income(
                operating_income,
                collateral_costs,
                use_accrual=False,
            )

        metrics["revenue"] = revenue
        metrics["operating_income"] = operating_income
        metrics["net_income"] = net_income
        metrics["interest_expense"] = collateral_costs

        # Insurance expenses
        metrics["insurance_premiums"] = self.period_insurance_premiums
        metrics["insurance_losses"] = self.period_insurance_losses
        period_lae: Decimal = getattr(self, "period_insurance_lae", to_decimal(0))
        metrics["insurance_lae"] = period_lae
        metrics["total_insurance_costs"] = (
            self.period_insurance_premiums + self.period_insurance_losses + period_lae
        )

        # Reserve development metrics (Issue #470)
        adverse_dev: Decimal = getattr(self, "period_adverse_development", to_decimal(0))
        favorable_dev: Decimal = getattr(self, "period_favorable_development", to_decimal(0))
        metrics["adverse_development"] = adverse_dev
        metrics["favorable_development"] = favorable_dev
        metrics["net_reserve_development"] = adverse_dev - favorable_dev

        # Deferred tax balances (Issue #367, ASC 740; Issue #464, ASC 740-10-30-5)
        metrics["deferred_tax_asset"] = self.deferred_tax_asset
        metrics["dta_valuation_allowance"] = self.dta_valuation_allowance
        metrics["deferred_tax_liability"] = self.deferred_tax_liability

        # Dividends and depreciation
        metrics["dividends_paid"] = self._last_dividends_paid
        metrics["depreciation_expense"] = annual_depreciation

        # COGS and SG&A breakdown (Issue #255)
        expense_ratios = getattr(self.config, "expense_ratios", None)

        if expense_ratios is not None:
            gross_margin_ratio = to_decimal(expense_ratios.gross_margin_ratio)
            sga_expense_ratio = to_decimal(expense_ratios.sga_expense_ratio)
            mfg_depreciation_alloc = to_decimal(
                expense_ratios.manufacturing_depreciation_allocation
            )
            admin_depreciation_alloc = to_decimal(expense_ratios.admin_depreciation_allocation)
            direct_materials_ratio = to_decimal(expense_ratios.direct_materials_ratio)
            direct_labor_ratio = to_decimal(expense_ratios.direct_labor_ratio)
            manufacturing_overhead_ratio = to_decimal(expense_ratios.manufacturing_overhead_ratio)
            selling_expense_ratio = to_decimal(expense_ratios.selling_expense_ratio)
            general_admin_ratio = to_decimal(expense_ratios.general_admin_ratio)
        else:
            gross_margin_ratio = to_decimal(0.15)
            sga_expense_ratio = to_decimal(0.07)
            mfg_depreciation_alloc = to_decimal(0.7)
            admin_depreciation_alloc = to_decimal(0.3)
            direct_materials_ratio = to_decimal(0.4)
            direct_labor_ratio = to_decimal(0.3)
            manufacturing_overhead_ratio = to_decimal(0.3)
            selling_expense_ratio = to_decimal(0.4)
            general_admin_ratio = to_decimal(0.6)

        # Calculate COGS breakdown
        cogs_ratio = to_decimal(1) - gross_margin_ratio
        base_cogs = revenue * cogs_ratio
        mfg_depreciation = annual_depreciation * mfg_depreciation_alloc
        cogs_before_depreciation = base_cogs - mfg_depreciation

        metrics["direct_materials"] = cogs_before_depreciation * direct_materials_ratio
        metrics["direct_labor"] = cogs_before_depreciation * direct_labor_ratio
        metrics["manufacturing_overhead"] = cogs_before_depreciation * manufacturing_overhead_ratio
        metrics["mfg_depreciation"] = mfg_depreciation
        metrics["total_cogs"] = base_cogs

        # Calculate SG&A breakdown
        base_sga = revenue * sga_expense_ratio
        admin_depreciation = annual_depreciation * admin_depreciation_alloc
        sga_before_depreciation = base_sga - admin_depreciation

        metrics["selling_expenses"] = sga_before_depreciation * selling_expense_ratio
        metrics["general_admin_expenses"] = sga_before_depreciation * general_admin_ratio
        metrics["admin_depreciation"] = admin_depreciation
        metrics["total_sga"] = base_sga

        # Store expense ratios
        metrics["gross_margin_ratio"] = gross_margin_ratio
        metrics["sga_expense_ratio"] = sga_expense_ratio

        # Financial ratios
        metrics["asset_turnover"] = (
            revenue / self.total_assets if self.total_assets > ZERO else to_decimal(0)
        )

        base_margin = to_decimal(self.base_operating_margin)
        actual_margin = operating_income / revenue if revenue > ZERO else to_decimal(0)
        metrics["base_operating_margin"] = base_margin
        metrics["actual_operating_margin"] = actual_margin
        metrics["insurance_impact_on_margin"] = base_margin - actual_margin

        MIN_EQUITY_THRESHOLD = to_decimal(100)

        metrics["roe"] = (
            net_income / self.equity if self.equity > MIN_EQUITY_THRESHOLD else to_decimal(0)
        )
        metrics["roa"] = (
            net_income / self.total_assets if self.total_assets > ZERO else to_decimal(0)
        )

        # Leverage metrics
        metrics["collateral_to_equity"] = (
            self.collateral / self.equity if self.equity > ZERO else to_decimal(0)
        )
        metrics["collateral_to_assets"] = (
            self.collateral / self.total_assets if self.total_assets > ZERO else to_decimal(0)
        )

        return metrics

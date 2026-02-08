# mypy: disable-error-code="attr-defined, has-type, no-any-return"
"""Income calculation mixin for WidgetManufacturer.

This module contains the IncomeCalculationMixin class, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305). It provides revenue, operating
income, collateral costs, and net income calculation methods.
"""

from decimal import Decimal
import logging
from typing import Union

try:
    from ergodic_insurance.decimal_utils import ZERO, to_decimal
    from ergodic_insurance.ledger import AccountName, TransactionType
    from ergodic_insurance.tax_handler import TaxHandler
except ImportError:
    try:
        from .decimal_utils import ZERO, to_decimal
        from .ledger import AccountName, TransactionType
        from .tax_handler import TaxHandler
    except ImportError:
        from decimal_utils import ZERO, to_decimal  # type: ignore[no-redef]
        from ledger import AccountName, TransactionType  # type: ignore[no-redef]
        from tax_handler import TaxHandler  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class IncomeCalculationMixin:
    """Mixin providing income statement calculation methods.

    This mixin expects the host class to have:
        - self.total_assets: Decimal (property from BalanceSheetMixin)
        - self.asset_turnover_ratio: float
        - self.base_operating_margin: float
        - self.tax_rate: float
        - self.stochastic_process: Optional[StochasticProcess]
        - self.period_insurance_premiums: Decimal
        - self.period_insurance_losses: Decimal
        - self.collateral: Decimal (property from BalanceSheetMixin)
        - self.equity: Decimal (property from BalanceSheetMixin)
        - self.accrual_manager: AccrualManager instance
        - self.tax_handler: TaxHandler instance (persistent, Issue #365)
        - self.ledger: Ledger instance
        - self.current_year: int
        - self.current_month: int
        - self._nol_carryforward_enabled: bool
    """

    def calculate_revenue(self, apply_stochastic: bool = False) -> Decimal:
        """Calculate revenue based on available assets and turnover ratio.

        Args:
            apply_stochastic (bool): Whether to apply stochastic shock to revenue.

        Returns:
            float: Annual revenue in dollars. Always non-negative.
        """
        available_assets = max(ZERO, self.total_assets)
        revenue = available_assets * to_decimal(self.asset_turnover_ratio)

        if apply_stochastic and self.stochastic_process is not None:
            shock = self.stochastic_process.generate_shock(float(revenue))
            revenue *= to_decimal(shock)
            logger.debug(f"Applied stochastic shock: {shock:.4f}")

        logger.debug(f"Revenue calculated: ${revenue:,.2f} from assets ${self.total_assets:,.2f}")
        return revenue

    def calculate_operating_income(self, revenue: Union[Decimal, float]) -> Decimal:
        """Calculate operating income including insurance as operating expenses.

        Depreciation is NOT subtracted here because it is already embedded in the
        COGS/SGA expense ratios derived from base_operating_margin.  Since
        Revenue = COGS + SGA + base_operating_income (where COGS and SGA each
        include their allocated depreciation), subtracting depreciation again
        would double-count it (Issue #475).

        Args:
            revenue: Annual revenue in dollars.

        Returns:
            Decimal: Operating income in dollars after insurance costs.
        """
        revenue_decimal = to_decimal(revenue)

        base_operating_income = revenue_decimal * to_decimal(self.base_operating_margin)

        # Net reserve development: adverse increases expense, favorable decreases it
        net_reserve_development = getattr(self, "period_adverse_development", ZERO) - getattr(
            self, "period_favorable_development", ZERO
        )

        actual_operating_income = (
            base_operating_income
            - self.period_insurance_premiums
            - self.period_insurance_losses
            - net_reserve_development
        )

        logger.debug(
            f"Operating income: ${actual_operating_income:,.2f} "
            f"(base: ${base_operating_income:,.2f}, insurance: "
            f"${self.period_insurance_premiums + self.period_insurance_losses:,.2f})"
        )
        return actual_operating_income

    def calculate_collateral_costs(
        self,
        letter_of_credit_rate: Union[Decimal, float] = 0.015,
        time_period: str = "annual",
    ) -> Decimal:
        """Calculate costs for letter of credit collateral.

        Args:
            letter_of_credit_rate (float): Annual interest rate for letter of credit.
            time_period (str): "annual" or "monthly".

        Returns:
            Decimal: Collateral costs for the specified period in dollars.
        """
        if time_period == "monthly":
            period_rate = to_decimal(letter_of_credit_rate / 12)
        else:
            period_rate = to_decimal(letter_of_credit_rate)

        collateral_costs = self.collateral * period_rate
        if collateral_costs > ZERO:
            logger.debug(
                f"Collateral costs ({time_period}): ${collateral_costs:,.2f} on ${self.collateral:,.2f} collateral"
            )
        return collateral_costs

    def calculate_net_income(
        self,
        operating_income: Union[Decimal, float],
        collateral_costs: Union[Decimal, float],
        use_accrual: bool = True,
        time_resolution: str = "annual",
    ) -> Decimal:
        """Calculate net income after collateral costs and taxes.

        Insurance costs (premiums and losses) are already deducted in
        calculate_operating_income() and must NOT be passed here to avoid
        double-counting (Issue #374).

        Args:
            operating_income: Operating income after insurance costs (from
                calculate_operating_income).
            collateral_costs: Financing costs for letter of credit collateral.
            use_accrual: Whether to use accrual accounting for taxes.
            time_resolution: Time resolution for tax accrual calculation.

        Returns:
            Decimal: Net income after all expenses and taxes.
        """
        # Convert inputs to Decimal
        operating_income_decimal = to_decimal(operating_income)
        collateral_costs_decimal = to_decimal(collateral_costs)

        income_before_tax = operating_income_decimal - collateral_costs_decimal

        # Capture DTA before tax calculation for journal entry delta (Issue #365)
        old_dta = self.tax_handler.deferred_tax_asset if self._nol_carryforward_enabled else ZERO

        # Compute theoretical tax for logging (no side effects)
        theoretical_tax_for_log = max(
            ZERO, to_decimal(income_before_tax) * to_decimal(self.tax_rate)
        )

        # Read equity BEFORE tax accrual (critical for non-circular flow)
        current_equity = self.equity

        # Use persistent TaxHandler — calculate_and_accrue_tax calls
        # calculate_tax_liability internally, which modifies NOL state (Issue #365)
        actual_tax_expense, was_capped, nol_utilized = self.tax_handler.calculate_and_accrue_tax(
            income_before_tax=income_before_tax,
            current_equity=current_equity,
            use_accrual=use_accrual,
            time_resolution=time_resolution,
            current_year=self.current_year,
            current_month=self.current_month,
        )

        net_income = income_before_tax - actual_tax_expense

        # Record DTA journal entries for NOL changes (Issue #365)
        if self._nol_carryforward_enabled:
            new_dta = self.tax_handler.deferred_tax_asset
            dta_change = new_dta - old_dta
            if dta_change > ZERO:
                # DTA increased (loss year or additional losses):
                # Dr DEFERRED_TAX_ASSET, Cr TAX_EXPENSE (tax benefit)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.DEFERRED_TAX_ASSET,
                    credit_account=AccountName.TAX_EXPENSE,
                    amount=dta_change,
                    transaction_type=TransactionType.DTA_ADJUSTMENT,
                    description=f"Year {self.current_year} DTA recognition from NOL carryforward",
                    month=self.current_month,
                )
            elif dta_change < ZERO:
                # DTA decreased (NOL utilized):
                # Dr TAX_EXPENSE, Cr DEFERRED_TAX_ASSET (DTA reversal)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.TAX_EXPENSE,
                    credit_account=AccountName.DEFERRED_TAX_ASSET,
                    amount=abs(dta_change),
                    transaction_type=TransactionType.DTA_ADJUSTMENT,
                    description=f"Year {self.current_year} DTA reversal from NOL utilization",
                    month=self.current_month,
                )

        # Enhanced profit waterfall logging
        logger.info("===== PROFIT WATERFALL =====")
        logger.info(f"Operating Income:        ${operating_income_decimal:,.2f}")
        if collateral_costs_decimal > ZERO:
            logger.info(f"  - Collateral Costs:    ${collateral_costs_decimal:,.2f}")
        logger.info(f"Income Before Tax:       ${income_before_tax:,.2f}")
        if nol_utilized > ZERO:
            logger.info(f"  - NOL Utilized:        ${nol_utilized:,.2f}")
            logger.info(f"  Taxable Income:        ${income_before_tax - nol_utilized:,.2f}")
        logger.info(f"  - Taxes (@{self.tax_rate:.1%}):      ${actual_tax_expense:,.2f}")
        if was_capped:
            logger.info(
                f"    (Capped from ${theoretical_tax_for_log:,.2f} due to limited liability)"
            )
        if use_accrual and actual_tax_expense > 0:
            logger.info("    (Accrued for quarterly payment)")
        if self._nol_carryforward_enabled and self.tax_handler.nol_carryforward > ZERO:
            logger.info(
                f"  NOL Carryforward:      ${self.tax_handler.nol_carryforward:,.2f} "
                f"(DTA: ${self.tax_handler.deferred_tax_asset:,.2f})"
            )
        logger.info(f"NET INCOME:              ${net_income:,.2f}")
        logger.info("============================")

        # Validation assertion
        epsilon = to_decimal("0.000000001")
        if collateral_costs_decimal > epsilon:
            assert (
                net_income <= operating_income_decimal + epsilon
            ), f"Net income ({net_income}) should be less than or equal to operating income ({operating_income_decimal}) when costs exist"

        return net_income

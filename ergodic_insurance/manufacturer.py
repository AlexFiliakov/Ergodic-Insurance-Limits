# pylint: disable=too-many-lines
"""Widget manufacturer financial model implementation.

This module implements the core financial model for a widget manufacturing
company, providing comprehensive balance sheet management, insurance claim
processing, and stochastic modeling capabilities. It serves as the central
component of the ergodic insurance optimization framework.

The manufacturer model simulates realistic business operations including:
    - Asset-based revenue generation with configurable turnover ratios
    - Operating income calculations with industry-standard margins
    - Multi-layer insurance claim processing with deductibles and limits
    - Letter of credit collateral management for claim liabilities
    - Actuarial claim payment schedules over multiple years
    - Dynamic balance sheet evolution with growth and volatility
    - Integration with sophisticated stochastic processes
    - Comprehensive financial metrics and ratio analysis

Key Components:
    - :class:`WidgetManufacturer`: Main financial model class
    - :class:`ClaimLiability`: Actuarial claim payment tracking (re-exported)
    - :class:`TaxHandler`: Tax calculation and accrual (re-exported)

Examples:
    Basic manufacturer setup and simulation::

        from ergodic_insurance.config import ManufacturerConfig
        from ergodic_insurance.manufacturer import WidgetManufacturer

        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7
        )

        manufacturer = WidgetManufacturer(config)

        metrics = manufacturer.step(
            letter_of_credit_rate=0.015,
            growth_rate=0.05
        )

        print(f"ROE: {metrics['roe']:.1%}")
"""

import copy as copy_module
from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional, Union

try:
    from ergodic_insurance.accrual_manager import AccrualManager, AccrualType, PaymentSchedule
    from ergodic_insurance.config import ManufacturerConfig
    from ergodic_insurance.decimal_utils import ONE, ZERO, MetricsDict, to_decimal
    from ergodic_insurance.insurance_accounting import InsuranceAccounting
    from ergodic_insurance.ledger import AccountName, Ledger, TransactionType
    from ergodic_insurance.stochastic_processes import StochasticProcess
except ImportError:
    try:
        from .accrual_manager import AccrualManager, AccrualType, PaymentSchedule
        from .config import ManufacturerConfig
        from .decimal_utils import ONE, ZERO, MetricsDict, to_decimal
        from .insurance_accounting import InsuranceAccounting
        from .ledger import AccountName, Ledger, TransactionType
        from .stochastic_processes import StochasticProcess
    except ImportError:
        from accrual_manager import (  # type: ignore[no-redef]
            AccrualManager,
            AccrualType,
            PaymentSchedule,
        )
        from config import ManufacturerConfig  # type: ignore[no-redef]
        from decimal_utils import ONE, ZERO, MetricsDict, to_decimal  # type: ignore[no-redef]
        from insurance_accounting import InsuranceAccounting  # type: ignore[no-redef]
        from ledger import AccountName, Ledger, TransactionType  # type: ignore[no-redef]
        from stochastic_processes import StochasticProcess  # type: ignore[no-redef]

# Re-exports for backward compatibility (Issue #305)
from ergodic_insurance.claim_liability import (  # noqa: F401  pylint: disable=ungrouped-imports
    ClaimLiability,
)

# Import mixins
from ergodic_insurance.manufacturer_balance_sheet import (  # pylint: disable=ungrouped-imports
    BalanceSheetMixin,
)
from ergodic_insurance.manufacturer_claims import ClaimProcessingMixin
from ergodic_insurance.manufacturer_income import IncomeCalculationMixin
from ergodic_insurance.manufacturer_metrics import MetricsCalculationMixin
from ergodic_insurance.manufacturer_solvency import SolvencyMixin
from ergodic_insurance.tax_handler import TaxHandler  # noqa: F401  pylint: disable=unused-import

logger = logging.getLogger(__name__)


class WidgetManufacturer(
    BalanceSheetMixin,
    ClaimProcessingMixin,
    IncomeCalculationMixin,
    SolvencyMixin,
    MetricsCalculationMixin,
):
    """Financial model for a widget manufacturing company.

    This class models the complete financial operations of a manufacturing
    company including revenue generation, claim processing, collateral
    management, and balance sheet evolution over time.

    The manufacturer maintains a balance sheet with assets, equity, and tracks
    insurance-related collateral. It can process insurance claims with multi-year
    payment schedules and manages working capital requirements.

    Attributes:
        config: Manufacturing configuration parameters
        stochastic_process: Optional stochastic process for revenue volatility
        assets: Current total assets
        collateral: Letter of credit collateral for insurance claims
        restricted_assets: Assets restricted as collateral
        equity: Current equity (assets minus liabilities)
        year: Current simulation year
        outstanding_liabilities: List of active claim liabilities
        metrics_history: Historical metrics for each simulation period
        bankruptcy: Whether the company has gone bankrupt
        bankruptcy_year: Year when bankruptcy occurred (if applicable)

    Example:
        Running a multi-year simulation::

            manufacturer = WidgetManufacturer(config)

            for year in range(10):
                losses, _ = loss_generator.generate_losses(duration=1, revenue=revenue)
                for loss in losses:
                    manufacturer.process_insurance_claim(
                        loss.amount, deductible, limit
                    )
                metrics = manufacturer.step(letter_of_credit_rate=0.015)
                print(f"Year {year}: ROE={metrics['roe']:.1%}")
    """

    def __init__(
        self, config: ManufacturerConfig, stochastic_process: Optional[StochasticProcess] = None
    ):
        """Initialize manufacturer with configuration parameters.

        Args:
            config (ManufacturerConfig): Manufacturing configuration parameters.
            stochastic_process (Optional[StochasticProcess]): Optional stochastic
                process for adding revenue volatility. Defaults to None.
        """
        self.config = config
        self.stochastic_process = stochastic_process

        # Initialize the event-sourcing ledger FIRST
        self.ledger = Ledger()

        # Track original prepaid premium for amortization calculation
        self._original_prepaid_premium: Decimal = ZERO

        # Insurance accounting module
        self.insurance_accounting = InsuranceAccounting()

        # Accrual management for timing differences
        self.accrual_manager = AccrualManager()

        # Operating parameters
        self.asset_turnover_ratio = config.asset_turnover_ratio
        self.base_operating_margin = config.base_operating_margin
        self.tax_rate = config.tax_rate
        self.retention_ratio = config.retention_ratio

        # Claim tracking
        self.claim_liabilities: List[ClaimLiability] = []
        self.current_year = 0
        self.current_month = 0

        # Insurance cost tracking for tax purposes
        self.period_insurance_premiums: Decimal = ZERO
        self.period_insurance_losses: Decimal = ZERO

        # Track actual dividends paid
        self._last_dividends_paid: Decimal = ZERO

        # Solvency tracking
        self.is_ruined = False
        self.ruin_month: Optional[int] = None

        # Metrics tracking
        self.metrics_history: List[MetricsDict] = []

        # Store initial values for base comparisons
        self._initial_assets: Decimal = to_decimal(config.initial_assets)
        self._initial_equity: Decimal = to_decimal(config.initial_assets)

        # Compute initial balance sheet values
        # Type ignore: ppe_ratio is guaranteed non-None after model_validator
        initial_gross_ppe: Decimal = to_decimal(config.initial_assets * config.ppe_ratio)  # type: ignore
        initial_accumulated_depreciation: Decimal = ZERO

        # Current Assets - initialize working capital to steady state
        initial_revenue = to_decimal(config.initial_assets * config.asset_turnover_ratio)
        initial_cogs = initial_revenue * to_decimal(1 - config.base_operating_margin)

        initial_accounts_receivable: Decimal = initial_revenue * to_decimal(45 / 365)
        initial_inventory: Decimal = initial_cogs * to_decimal(60 / 365)
        initial_prepaid_insurance: Decimal = ZERO

        working_capital_assets = initial_accounts_receivable + initial_inventory
        initial_cash: Decimal = to_decimal(config.initial_assets * (1 - config.ppe_ratio)) - working_capital_assets  # type: ignore

        initial_accounts_payable: Decimal = ZERO
        initial_collateral: Decimal = ZERO
        initial_restricted_assets: Decimal = ZERO

        # Record all initial balances to ledger
        self._record_initial_balances(
            cash=initial_cash,
            accounts_receivable=initial_accounts_receivable,
            inventory=initial_inventory,
            prepaid_insurance=initial_prepaid_insurance,
            gross_ppe=initial_gross_ppe,
            accumulated_depreciation=initial_accumulated_depreciation,
            accounts_payable=initial_accounts_payable,
            collateral=initial_collateral,
            restricted_assets=initial_restricted_assets,
        )

    def _record_initial_balances(
        self,
        cash: Decimal,
        accounts_receivable: Decimal,
        inventory: Decimal,
        prepaid_insurance: Decimal,
        gross_ppe: Decimal,
        accumulated_depreciation: Decimal,
        accounts_payable: Decimal,
        collateral: Decimal,
        restricted_assets: Decimal,
    ) -> None:
        """Record initial balance sheet entries in the ledger.

        This establishes the opening balances for all accounts at year 0.
        Uses equity (retained_earnings) as the balancing entry.

        Args:
            cash: Initial cash position
            accounts_receivable: Initial accounts receivable
            inventory: Initial inventory
            prepaid_insurance: Initial prepaid insurance
            gross_ppe: Initial gross property, plant & equipment
            accumulated_depreciation: Initial accumulated depreciation
            accounts_payable: Initial accounts payable
            collateral: Initial letter of credit collateral
            restricted_assets: Initial restricted assets
        """
        if cash > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=cash,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial cash position",
            )

        if accounts_receivable > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=accounts_receivable,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial accounts receivable",
            )

        if inventory > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.INVENTORY,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=inventory,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial inventory",
            )

        if prepaid_insurance > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.PREPAID_INSURANCE,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=prepaid_insurance,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial prepaid insurance",
            )

        if gross_ppe > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.GROSS_PPE,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=gross_ppe,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial gross PP&E",
            )

        if accumulated_depreciation > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.ACCUMULATED_DEPRECIATION,
                amount=accumulated_depreciation,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial accumulated depreciation",
            )

        if accounts_payable > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.ACCOUNTS_PAYABLE,
                amount=accounts_payable,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial accounts payable",
            )

        if restricted_assets > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RESTRICTED_CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=restricted_assets,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial restricted assets",
            )

    # Properties for FinancialStateProvider protocol
    @property
    def current_revenue(self) -> Decimal:
        """Get current revenue based on current assets and turnover ratio."""
        return self.calculate_revenue()

    @property
    def current_assets(self) -> Decimal:
        """Get current total assets."""
        return self.total_assets

    @property
    def current_equity(self) -> Decimal:
        """Get current equity value."""
        return self.equity

    @property
    def base_revenue(self) -> Decimal:
        """Get base (initial) revenue for comparison."""
        return self._initial_assets * to_decimal(self.config.asset_turnover_ratio)

    @property
    def base_assets(self) -> Decimal:
        """Get base (initial) assets for comparison."""
        return self._initial_assets

    @property
    def base_equity(self) -> Decimal:
        """Get base (initial) equity for comparison."""
        return self._initial_equity

    # ========================================================================
    # Serialization support
    # ========================================================================

    def __deepcopy__(self, memo: Dict[int, Any]) -> "WidgetManufacturer":
        """Create a deep copy preserving all state for Monte Carlo forking.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this WidgetManufacturer with all state preserved
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            setattr(result, key, copy_module.deepcopy(value, memo))

        return result

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling (required for Windows multiprocessing).

        Returns:
            Dictionary of all instance attributes
        """
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from pickle (required for Windows multiprocessing).

        Args:
            state: Dictionary of instance attributes to restore
        """
        self.__dict__.update(state)

    # ========================================================================
    # Accrual coordination
    # ========================================================================

    def process_accrued_payments(
        self, time_resolution: str = "annual", max_payable: Optional[Union[Decimal, float]] = None
    ) -> Decimal:
        """Process due accrual payments for the current period.

        Args:
            time_resolution: "annual" or "monthly" for determining current period
            max_payable: Optional maximum amount that can be paid.

        Returns:
            Total cash payments made for accruals in this period
        """
        if time_resolution == "monthly":
            period = self.current_year * 12 + self.current_month
        else:
            period = self.current_year * 12

        self.accrual_manager.current_period = period

        payments_due = self.accrual_manager.get_payments_due(period)

        total_due = sum((to_decimal(v) for v in payments_due.values()), ZERO)
        if max_payable is not None:
            max_total_payable: Decimal = min(total_due, to_decimal(max_payable))
        else:
            current_equity = self.equity
            max_total_payable = min(total_due, current_equity) if current_equity > ZERO else ZERO

        if total_due > max_total_payable:
            logger.warning(
                f"LIMITED LIABILITY: Capping total accrued payments. "
                f"Due: ${total_due:,.2f}, Payable: ${max_total_payable:,.2f}"
            )

        payment_ratio: Decimal = max_total_payable / total_due if total_due > ZERO else ZERO

        total_paid: Decimal = ZERO
        for accrual_type, amount_due in payments_due.items():
            payable_amount: Decimal = to_decimal(amount_due) * payment_ratio
            unpayable_amount = to_decimal(amount_due) - payable_amount

            if payable_amount > ZERO:
                self.accrual_manager.process_payment(accrual_type, payable_amount, period)

                total_paid += payable_amount

                if accrual_type == AccrualType.TAXES:
                    trans_type = TransactionType.TAX_PAYMENT
                    debit_account = AccountName.ACCRUED_TAXES
                elif accrual_type == AccrualType.WAGES:
                    trans_type = TransactionType.WAGE_PAYMENT
                    debit_account = AccountName.ACCRUED_WAGES
                elif accrual_type == AccrualType.INTEREST:
                    trans_type = TransactionType.INTEREST_PAYMENT
                    debit_account = AccountName.ACCRUED_INTEREST
                else:
                    trans_type = TransactionType.PAYMENT
                    debit_account = AccountName.ACCRUED_EXPENSES

                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=debit_account,
                    credit_account=AccountName.CASH,
                    amount=payable_amount,
                    transaction_type=trans_type,
                    description=f"Accrued {accrual_type.value} payment",
                    month=self.current_month,
                )

                logger.debug(f"Paid accrued {accrual_type.value}: ${payable_amount:,.2f}")

            if unpayable_amount > ZERO:
                logger.warning(
                    f"LIMITED LIABILITY: Discharged ${unpayable_amount:,.2f} of unpayable {accrual_type.value} from liabilities"
                )

        if total_paid > ZERO:
            logger.info(f"Total accrual payments this period: ${total_paid:,.2f}")

        return total_paid

    def record_wage_accrual(
        self, amount: float, payment_schedule: PaymentSchedule = PaymentSchedule.IMMEDIATE
    ) -> None:
        """Record accrued wages to be paid later.

        Args:
            amount: Wage amount to accrue
            payment_schedule: When wages will be paid
        """
        self.accrual_manager.record_expense_accrual(
            item_type=AccrualType.WAGES,
            amount=amount,
            payment_schedule=payment_schedule,
            description=f"Period {self.current_year} wages",
        )

    # ========================================================================
    # Simulation orchestration
    # ========================================================================

    def _handle_insolvent_step(self, time_resolution: str) -> MetricsDict:
        """Handle a simulation step when the company is already insolvent.

        Args:
            time_resolution: "annual" or "monthly" for simulation step.

        Returns:
            Dictionary of metrics for this time step.
        """
        logger.warning("Company is already insolvent, skipping step")
        metrics = self.calculate_metrics()
        metrics["year"] = self.current_year
        metrics["month"] = self.current_month if time_resolution == "monthly" else 0
        self._increment_time(time_resolution)
        return metrics

    def _increment_time(self, time_resolution: str) -> None:
        """Increment the current time based on resolution.

        Args:
            time_resolution: "annual" or "monthly" for simulation step.
        """
        if time_resolution == "monthly":
            self.current_month += 1
            if self.current_month >= 12:
                self.current_month = 0
                self.current_year += 1
        else:
            self.current_year += 1

    def step(
        self,
        letter_of_credit_rate: Union[Decimal, float] = 0.015,
        growth_rate: Union[Decimal, float] = 0.0,
        time_resolution: str = "annual",
        apply_stochastic: bool = False,
    ) -> MetricsDict:
        """Execute one time step of the financial model simulation.

        Args:
            letter_of_credit_rate (float): Annual interest rate for letter of credit.
            growth_rate (float): Revenue growth rate for the period.
            time_resolution (str): "annual" or "monthly".
            apply_stochastic (bool): Whether to apply stochastic shocks.

        Returns:
            Dict[str, float]: Comprehensive financial metrics dictionary.
        """
        # Check if already ruined
        if self.is_ruined:
            return self._handle_insolvent_step(time_resolution)

        # Check for potential mid-year insolvency (Issue #279)
        if not self.check_liquidity_constraints(time_resolution):
            return self._handle_insolvent_step(time_resolution)

        # Store initial revenue for working capital calculation in monthly mode
        if time_resolution == "monthly" and self.current_month == 0:
            self._annual_revenue_for_wc = self.calculate_revenue(apply_stochastic)

        # Calculate financial performance
        revenue = self.calculate_revenue(apply_stochastic)

        # Calculate working capital components BEFORE payment coordination
        if time_resolution == "annual":
            self.calculate_working_capital_components(revenue)
        elif time_resolution == "monthly":
            if hasattr(self, "_annual_revenue_for_wc"):
                self.calculate_working_capital_components(self._annual_revenue_for_wc)
            else:
                annual_revenue = self.total_assets * to_decimal(self.asset_turnover_ratio)
                self.calculate_working_capital_components(annual_revenue)

        # COORDINATED LIMITED LIABILITY ENFORCEMENT
        if time_resolution == "monthly":
            period = self.current_year * 12 + self.current_month
        else:
            period = self.current_year * 12

        # Calculate total accrual payments due
        self.accrual_manager.current_period = period
        accrual_payments_due = self.accrual_manager.get_payments_due(period)
        total_accrual_due: Decimal = sum(
            (to_decimal(v) for v in accrual_payments_due.values()), ZERO
        )

        # Calculate total claim payments scheduled
        total_claim_due: Decimal = ZERO
        if time_resolution == "annual" or self.current_month == 0:
            for claim_item in self.claim_liabilities:
                years_since = self.current_year - claim_item.year_incurred
                scheduled_payment = claim_item.get_payment(years_since)
                total_claim_due += scheduled_payment

        # Cap TOTAL payments at available liquid resources
        total_payments_due = total_accrual_due + total_claim_due
        available_liquidity = self.cash + self.restricted_assets
        max_total_payable: Decimal = (
            min(total_payments_due, available_liquidity) if available_liquidity > ZERO else ZERO
        )

        # Allocate capped amount proportionally
        if total_payments_due > ZERO:
            allocation_ratio = max_total_payable / total_payments_due
            max_accrual_payable: Decimal = total_accrual_due * allocation_ratio
            max_claim_payable: Decimal = total_claim_due * allocation_ratio
        else:
            max_accrual_payable = ZERO
            max_claim_payable = ZERO

        if total_payments_due > max_total_payable:
            logger.warning(
                f"LIQUIDITY CONSTRAINT: Total payments due ${total_payments_due:,.2f} "
                f"exceeds available liquidity ${available_liquidity:,.2f} "
                f"(cash: ${self.cash:,.2f}, restricted: ${self.restricted_assets:,.2f}). "
                f"Capping at ${max_total_payable:,.2f} "
                f"(Accruals: ${max_accrual_payable:,.2f}, Claims: ${max_claim_payable:,.2f})"
            )

        # Process accrual payments with coordinated cap
        self.process_accrued_payments(time_resolution, max_payable=max_accrual_payable)

        # Pay scheduled claim liabilities with coordinated cap
        if time_resolution == "annual" or self.current_month == 0:
            self.pay_claim_liabilities(max_payable=max_claim_payable)

        # Calculate depreciation expense for the period
        if time_resolution == "annual":
            depreciation_expense = self.record_depreciation(useful_life_years=10)
        elif time_resolution == "monthly":
            depreciation_expense = self.record_depreciation(useful_life_years=10 * 12)
        else:
            depreciation_expense = ZERO

        # Calculate operating income including depreciation
        operating_income = self.calculate_operating_income(revenue, depreciation_expense)

        # Calculate collateral costs
        if time_resolution == "monthly":
            collateral_costs = self.calculate_collateral_costs(letter_of_credit_rate, "monthly")
            revenue = revenue / 12
            operating_income = operating_income / 12
        else:
            collateral_costs = self.calculate_collateral_costs(letter_of_credit_rate, "annual")

        # Calculate net income
        net_income = self.calculate_net_income(
            operating_income,
            collateral_costs,
            0,
            0,
            use_accrual=True,
            time_resolution=time_resolution,
        )

        # Update balance sheet with retained earnings
        self.update_balance_sheet(net_income, growth_rate)

        # Amortize prepaid insurance if applicable
        if time_resolution == "monthly":
            self.amortize_prepaid_insurance(months=1)

        # Apply revenue growth
        self._apply_growth(growth_rate, time_resolution, apply_stochastic)

        # Check solvency
        self.check_solvency()

        # Verify accounting equation (Issue #319)
        self._verify_accounting_equation()

        # Calculate and store metrics
        metrics = self.calculate_metrics(
            period_revenue=revenue, letter_of_credit_rate=letter_of_credit_rate
        )
        metrics["year"] = self.current_year
        metrics["month"] = self.current_month if time_resolution == "monthly" else 0
        self.metrics_history.append(metrics)

        # Increment time
        self._increment_time(time_resolution)

        # Reset period insurance costs for next period
        self.reset_period_insurance_costs()

        return metrics

    def reset(self) -> None:
        """Reset the manufacturer to initial state for new simulation.

        This method restores all financial parameters to their configured
        initial values and clears historical data, enabling fresh simulation
        runs from the same starting point.

        Bug Fixes (Issue #305):
        - FIX 1: Uses config.ppe_ratio directly instead of recalculating from margins
        - FIX 2: Initializes AR/Inventory to steady-state (matching __init__) instead of zero
        """
        # Reset operating parameters
        self.asset_turnover_ratio = self.config.asset_turnover_ratio
        self.claim_liabilities = []
        self.current_year = 0
        self.current_month = 0
        self.is_ruined = False
        self.ruin_month = None
        self.metrics_history = []

        # Reset period insurance cost tracking
        self.period_insurance_premiums = ZERO
        self.period_insurance_losses = ZERO

        # Reset dividend tracking
        self._last_dividends_paid = ZERO

        # Reset initial values (for exposure bases)
        initial_assets = to_decimal(self.config.initial_assets)
        self._initial_assets = initial_assets
        self._initial_equity = initial_assets

        # Reset accrual manager
        self.accrual_manager = AccrualManager()

        # Track original prepaid premium for amortization calculation
        self._original_prepaid_premium = ZERO

        # FIX 1 (Issue #305): Use config.ppe_ratio directly, same as __init__()
        # Previously, reset() recalculated ppe_ratio from margin thresholds,
        # which could diverge from the config value used in __init__().
        # Type ignore: ppe_ratio is guaranteed non-None after model_validator
        ppe_ratio = to_decimal(self.config.ppe_ratio)

        initial_gross_ppe: Decimal = initial_assets * ppe_ratio
        initial_accumulated_depreciation: Decimal = ZERO

        # FIX 2 (Issue #305): Initialize AR/Inventory to steady-state, same as __init__()
        # Previously, reset() set AR/Inventory to zero, causing Year 1 "warm-up" distortion
        initial_revenue = to_decimal(self.config.initial_assets * self.config.asset_turnover_ratio)
        initial_cogs = initial_revenue * to_decimal(1 - self.config.base_operating_margin)

        initial_accounts_receivable: Decimal = initial_revenue * to_decimal(45 / 365)
        initial_inventory: Decimal = initial_cogs * to_decimal(60 / 365)
        initial_prepaid_insurance: Decimal = ZERO
        initial_accounts_payable: Decimal = ZERO
        initial_collateral: Decimal = ZERO
        initial_restricted_assets: Decimal = ZERO

        # Adjust cash to fund working capital assets (same as __init__)
        working_capital_assets = initial_accounts_receivable + initial_inventory
        initial_cash: Decimal = initial_assets * (ONE - ppe_ratio) - working_capital_assets

        # Reset ledger FIRST (single source of truth)
        self.ledger = Ledger()
        self._record_initial_balances(
            cash=initial_cash,
            accounts_receivable=initial_accounts_receivable,
            inventory=initial_inventory,
            prepaid_insurance=initial_prepaid_insurance,
            gross_ppe=initial_gross_ppe,
            accumulated_depreciation=initial_accumulated_depreciation,
            accounts_payable=initial_accounts_payable,
            collateral=initial_collateral,
            restricted_assets=initial_restricted_assets,
        )

        # Reset stochastic process if present
        if self.stochastic_process is not None:
            self.stochastic_process.reset()

        logger.info("Manufacturer reset to initial state")

    def copy(self) -> "WidgetManufacturer":
        """Create a deep copy of the manufacturer for parallel simulations.

        Returns:
            WidgetManufacturer: A new manufacturer instance with same configuration.
        """
        new_manufacturer = WidgetManufacturer(
            config=self.config,
            stochastic_process=(
                copy_module.deepcopy(self.stochastic_process) if self.stochastic_process else None
            ),
        )

        logger.debug("Created copy of manufacturer")
        return new_manufacturer

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
    - :class:`ClaimLiability`: Actuarial claim payment tracking
    - Integration with :mod:`~ergodic_insurance.config` for parameter management
    - Integration with :mod:`~ergodic_insurance.stochastic_processes` for uncertainty

Examples:
    Basic manufacturer setup and simulation::

        from ergodic_insurance.config_v2 import ManufacturerConfig
        from ergodic_insurance.manufacturer import WidgetManufacturer

        # Configure manufacturer with realistic parameters
        config = ManufacturerConfig(
            initial_assets=10_000_000,          # $10M starting assets
            asset_turnover_ratio=0.8,           # 0.8x asset turnover
            base_operating_margin=0.08,              # 8% operating margin
            tax_rate=0.25,                      # 25% corporate tax rate
            retention_ratio=0.7                 # Retain 70% of earnings
        )

        # Create manufacturer instance
        manufacturer = WidgetManufacturer(config)

        # Process insurance claim
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=5_000_000,      # $5M total claim
            deductible=1_000_000,        # $1M deductible
            insurance_limit=10_000_000   # $10M coverage limit
        )
        print(f"Company pays: ${company_payment:,.2f}")
        print(f"Insurance covers: ${insurance_payment:,.2f}")

        # Run annual business operations
        metrics = manufacturer.step(
            working_capital_pct=0.2,         # 20% working capital
            letter_of_credit_rate=0.015,     # 1.5% LoC rate
            growth_rate=0.05                 # 5% annual growth
        )

        print(f"ROE: {metrics['roe']:.1%}")
        print(f"Assets: ${metrics['assets']:,.2f}")
        print(f"Equity: ${metrics['equity']:,.2f}")

    Multi-year simulation with claims::

        # Initialize for long-term simulation
        manufacturer = WidgetManufacturer(config)

        for year in range(20):
            # Process annual claims (example: 1 large claim every 5 years)
            if year % 5 == 0 and year > 0:
                manufacturer.process_insurance_claim(
                    claim_amount=8_000_000,
                    deductible=2_000_000,
                    insurance_limit=15_000_000
                )
                print(f"Year {year}: Large claim processed")

            # Run annual operations
            metrics = manufacturer.step(
                working_capital_pct=0.18,
                growth_rate=0.03
            )

            # Check for insolvency
            if not metrics['is_solvent']:
                print(f"Bankruptcy in year {year}")
                break

            print(f"Year {year}: ROE={metrics['roe']:.1%}, "
                  f"Assets=${metrics['assets']:,.0f}")

    Stochastic modeling with revenue volatility::

        from ergodic_insurance.stochastic_processes import (
            GeometricBrownianMotion, LognormalVolatility
        )

        # Create stochastic process for revenue volatility
        gbm = GeometricBrownianMotion(
            drift=0.03,        # 3% expected growth
            volatility=0.15,   # 15% annual volatility
            dt=1.0             # Annual time steps
        )

        # Initialize manufacturer with stochastic capability
        stochastic_manufacturer = WidgetManufacturer(config, gbm)

        # Run simulation with stochastic revenue
        results = []
        for simulation in range(1000):  # Monte Carlo
            stochastic_manufacturer.reset()

            for year in range(10):
                metrics = stochastic_manufacturer.step(
                    working_capital_pct=0.2,
                    apply_stochastic=True  # Enable volatility
                )

                if not metrics['is_solvent']:
                    break

            results.append(stochastic_manufacturer.equity)

        # Analyze results
        import numpy as np
        final_equity = np.array(results)
        print(f"Mean final equity: ${np.mean(final_equity):,.0f}")
        print(f"Std deviation: ${np.std(final_equity):,.0f}")
        print(f"Bankruptcy rate: {np.mean(final_equity <= 0):.1%}")

    Integration with claim development patterns::

        from ergodic_insurance.claim_development import (
            ClaimDevelopment, WorkersCompensation
        )

        # Enhanced claim processing with actuarial patterns
        wc_pattern = WorkersCompensation()

        company_paid, insurance_paid, claim_obj = manufacturer.process_insurance_claim_with_development(
            claim_amount=3_000_000,
            deductible=500_000,
            insurance_limit=5_000_000,
            development_pattern=wc_pattern,
            claim_type="workers_compensation"
        )

        if claim_obj:
            print(f"Claim created with ID: {claim_obj.claim_id}")
            print(f"Development pattern: {claim_obj.development_pattern.pattern_name}")

Financial Theory Background:
    The manufacturer model implements several key financial concepts:

    **Asset-Based Revenue Model**: Revenue = Assets × Turnover Ratio, accounting
    for working capital requirements that tie up assets.

    **Multi-Period Claim Liabilities**: Insurance claims follow actuarial payment
    patterns over multiple years, requiring collateral management and cash flow
    planning.

    **Stochastic Business Dynamics**: Integration with geometric Brownian motion
    and other stochastic processes enables realistic volatility modeling.

    **Solvency Monitoring**: Continuous tracking of equity position with
    bankruptcy detection when liabilities exceed assets.

Performance Considerations:
    - The model is optimized for Monte Carlo simulations with 1000+ iterations
    - Memory efficient claim liability tracking automatically removes paid claims
    - Configurable time resolution (annual/monthly) for different analysis needs
    - Reset and copy methods enable parallel simulation scenarios

Integration Points:
    - :mod:`~ergodic_insurance.config_v2`: Parameter configuration and validation
    - :mod:`~ergodic_insurance.stochastic_processes`: Uncertainty modeling
    - :mod:`~ergodic_insurance.claim_generator`: Automated claim generation
    - :mod:`~ergodic_insurance.insurance_program`: Multi-layer insurance structures
    - :mod:`~ergodic_insurance.simulation`: High-level simulation orchestration

See Also:
    :class:`~ergodic_insurance.config_v2.ManufacturerConfig`: Configuration parameters
    :mod:`~ergodic_insurance.stochastic_processes`: Stochastic modeling options
    :mod:`~ergodic_insurance.claim_development`: Advanced actuarial patterns
    :mod:`~ergodic_insurance.simulation`: Simulation framework integration

Note:
    This module forms the financial foundation for ergodic insurance optimization.
    All business logic assumes a debt-free balance sheet where equity equals
    net assets, simplifying the model while maintaining realistic cash flows.
"""

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

try:
    # Try absolute import first (for installed package)
    from ergodic_insurance.accrual_manager import AccrualManager, AccrualType, PaymentSchedule
    from ergodic_insurance.config import ManufacturerConfig
    from ergodic_insurance.insurance_accounting import InsuranceAccounting
    from ergodic_insurance.stochastic_processes import StochasticProcess
except ImportError:
    try:
        # Try relative import (for package context)
        from .accrual_manager import AccrualManager, AccrualType, PaymentSchedule
        from .config import ManufacturerConfig
        from .insurance_accounting import InsuranceAccounting
        from .stochastic_processes import StochasticProcess
    except ImportError:
        # Fall back to direct import (for notebooks/scripts)
        from accrual_manager import (  # type: ignore[no-redef]
            AccrualManager,
            AccrualType,
            PaymentSchedule,
        )
        from config import ManufacturerConfig  # type: ignore[no-redef]
        from insurance_accounting import InsuranceAccounting  # type: ignore[no-redef]
        from stochastic_processes import StochasticProcess  # type: ignore[no-redef]

# Optional import for claim development integration
if TYPE_CHECKING:
    from .claim_development import Claim, ClaimDevelopment

logger = logging.getLogger(__name__)


@dataclass
class ClaimLiability:
    """Represents an outstanding insurance claim liability with payment schedule.

    This class tracks insurance claims that require multi-year payment
    schedules and manages the collateral required to support them. It follows
    standard actuarial claim development patterns for realistic cash flow modeling.

    The default payment schedule follows a typical long-tail liability pattern
    where claims are paid over 10 years, with higher percentages in early years
    and gradually decreasing payments over time. This pattern is commonly observed
    in general liability and workers' compensation claims.

    Attributes:
        original_amount (float): The original claim amount at inception.
        remaining_amount (float): The unpaid balance of the claim.
        year_incurred (int): The year when the claim was first incurred.
        is_insured (bool): Whether this claim involves insurance coverage.
            True for insured claims (company deductible), False for uninsured claims.
        payment_schedule (List[float]): Payment percentages by year since inception.
            Default follows a standard long-tail pattern:
            - Years 1-3: Front-loaded payments (10%, 20%, 20%)
            - Years 4-6: Moderate payments (15%, 10%, 8%)
            - Years 7-10: Tail payments (7%, 5%, 3%, 2%)

    Examples:
        Create and track a claim liability::

            # Create a $1M claim liability
            claim = ClaimLiability(
                original_amount=1_000_000,
                remaining_amount=1_000_000,
                year_incurred=2023
            )

            # Get payment due in year 2 (1 year after incurred)
            payment_due = claim.get_payment(1)  # Returns 200,000 (20%)

            # Make the payment
            actual_payment = claim.make_payment(payment_due)
            print(f"Remaining: ${claim.remaining_amount:,.2f}")

        Custom payment schedule::

            # Create claim with custom 3-year schedule
            custom_claim = ClaimLiability(
                original_amount=500_000,
                remaining_amount=500_000,
                year_incurred=2023,
                payment_schedule=[0.5, 0.3, 0.2]  # 50%, 30%, 20%
            )

    Note:
        The payment schedule percentages should sum to 1.0 for full claim payout.
        Payments beyond the schedule length return 0.0.

    See Also:
        :class:`~ergodic_insurance.claim_development.ClaimDevelopment`: For more
        sophisticated claim development patterns.
        :class:`~ergodic_insurance.claim_development.Claim`: For comprehensive
        claim tracking with reserving.
    """

    original_amount: float
    remaining_amount: float
    year_incurred: int
    is_insured: bool = True  # Default to insured for backward compatibility
    payment_schedule: List[float] = field(
        default_factory=lambda: [
            0.10,  # Year 1: 10%
            0.20,  # Year 2: 20%
            0.20,  # Year 3: 20%
            0.15,  # Year 4: 15%
            0.10,  # Year 5: 10%
            0.08,  # Year 6: 8%
            0.07,  # Year 7: 7%
            0.05,  # Year 8: 5%
            0.03,  # Year 9: 3%
            0.02,  # Year 10: 2%
        ]
    )

    def get_payment(self, years_since_incurred: int) -> float:
        """Calculate payment due for a given year after claim incurred.

        This method returns the scheduled payment amount based on the claim's
        development pattern. It handles boundary conditions gracefully, returning
        zero for negative years or years beyond the payment schedule.

        Args:
            years_since_incurred (int): Number of years since the claim was first
                incurred. Must be >= 0. Year 0 represents the year of incurrence.

        Returns:
            float: Payment amount due for this year in absolute dollars. Returns
                0.0 if years_since_incurred is negative or beyond the schedule.

        Examples:
            Calculate payments over time::

                claim = ClaimLiability(1_000_000, 1_000_000, 2020)

                # Year of incurrence (2020)
                payment_0 = claim.get_payment(0)  # $100,000 (10%)

                # One year later (2021)
                payment_1 = claim.get_payment(1)  # $200,000 (20%)

                # Beyond schedule
                payment_20 = claim.get_payment(20)  # $0 (beyond 10-year schedule)

        Note:
            The method multiplies the original amount by the percentage from the
            payment schedule. It does not check against remaining balance.
        """
        if years_since_incurred < 0 or years_since_incurred >= len(self.payment_schedule):
            return 0.0
        return self.original_amount * self.payment_schedule[years_since_incurred]

    def make_payment(self, amount: float) -> float:
        """Make a payment against the liability and update remaining balance.

        This method processes a payment against the claim liability, reducing
        the remaining amount. If the requested payment exceeds the remaining
        balance, only the remaining amount is paid.

        Args:
            amount (float): Requested payment amount in dollars. Should be >= 0.
                Negative amounts are treated as zero payment.

        Returns:
            float: Actual payment made in dollars. May be less than requested
                if insufficient liability remains. Returns 0.0 if amount <= 0
                or remaining_amount <= 0.

        Examples:
            Make payments and track remaining balance::

                claim = ClaimLiability(1_000_000, 1_000_000, 2020)

                # Make first scheduled payment
                actual = claim.make_payment(100_000)  # Returns 100_000
                assert claim.remaining_amount == 900_000

                # Try to overpay
                actual = claim.make_payment(2_000_000)  # Returns 900_000
                assert claim.remaining_amount == 0

                # No more payments possible
                actual = claim.make_payment(50_000)  # Returns 0

        Warning:
            This method modifies the claim's remaining_amount. Use with caution
            in concurrent scenarios.
        """
        payment = min(amount, self.remaining_amount)
        self.remaining_amount -= payment
        return payment


class WidgetManufacturer:
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
                # Generate claims for this year
                claims = claim_generator.generate_claims()

                # Process claims through insurance
                for claim in claims:
                    manufacturer.process_insurance_claim(
                        claim.amount, deductible, limit
                    )

                # Run annual business operations
                metrics = manufacturer.step(
                    working_capital_pct=0.2,
                    letter_of_credit_rate=0.015
                )

                print(f"Year {year}: ROE={metrics['roe']:.1%}")
    """

    def __init__(
        self, config: ManufacturerConfig, stochastic_process: Optional[StochasticProcess] = None
    ):
        """Initialize manufacturer with configuration parameters.

        Sets up the manufacturer's initial financial state based on the provided
        configuration. All balance sheet items start with their configured values,
        and the manufacturer begins in a solvent state with no outstanding claims.

        Args:
            config (ManufacturerConfig): Manufacturing configuration parameters
                including initial assets, margins, tax rates, and financial ratios.
                See :class:`~ergodic_insurance.config.ManufacturerConfig` for
                complete parameter descriptions.
            stochastic_process (Optional[StochasticProcess]): Optional stochastic
                process for adding revenue volatility. If provided, enables
                stochastic shocks to revenue calculations when apply_stochastic=True
                in simulation methods. Defaults to None for deterministic modeling.

        Examples:
            Basic deterministic setup::

                config = ManufacturerConfig(
                    initial_assets=10_000_000,
                    base_operating_margin=0.08,
                    tax_rate=0.25
                )
                manufacturer = WidgetManufacturer(config)

            With stochastic revenue modeling::

                from ergodic_insurance.stochastic_processes import (
                    GeometricBrownianMotion
                )

                gbm = GeometricBrownianMotion(
                    drift=0.05, volatility=0.15, dt=1.0
                )
                manufacturer = WidgetManufacturer(config, gbm)

        See Also:
            :class:`~ergodic_insurance.config.ManufacturerConfig`: Configuration
            parameters and validation.
            :class:`~ergodic_insurance.stochastic_processes.StochasticProcess`:
            Base class for stochastic modeling.
        """
        self.config = config
        self.stochastic_process = stochastic_process

        # Balance sheet items - removed self.assets and self.equity
        # as they are now calculated properties
        self.collateral = 0.0  # Letter of credit collateral for claims
        self.restricted_assets = 0.0  # Assets restricted as collateral

        # Enhanced balance sheet components for GAAP compliance
        # Fixed Assets (allocate first to determine cash)
        # Use PPE ratio from config (defaults based on operating margin if not specified)
        # Type ignore: ppe_ratio is guaranteed non-None after model_validator
        self.gross_ppe = config.initial_assets * config.ppe_ratio  # type: ignore
        self.accumulated_depreciation = 0.0  # Will accumulate over time

        # Current Assets
        self.cash = config.initial_assets * (1 - config.ppe_ratio)  # type: ignore
        self.accounts_receivable = 0.0  # Based on DSO
        self.inventory = 0.0  # Based on DIO
        self.prepaid_insurance = 0.0  # Annual premiums paid in advance

        # Current Liabilities
        self.accounts_payable = 0.0  # Based on DPO
        self.accrued_expenses = 0.0  # Other accrued items

        # Track original prepaid premium for amortization calculation
        self._original_prepaid_premium = 0.0

        # Insurance accounting module
        self.insurance_accounting = InsuranceAccounting()

        # Accrual management for timing differences
        self.accrual_manager = AccrualManager()

        # Operating parameters
        self.asset_turnover_ratio = config.asset_turnover_ratio
        self.base_operating_margin = config.base_operating_margin
        self.tax_rate = config.tax_rate
        self.retention_ratio = config.retention_ratio

        # Claim tracking - initialize early for property calculations
        self.claim_liabilities: List[ClaimLiability] = []
        self.current_year = 0
        self.current_month = 0  # Track months for monthly LoC payments

        # Insurance cost tracking for tax purposes
        self.period_insurance_premiums = 0.0  # Premiums paid this period
        self.period_insurance_losses = 0.0  # Losses paid this period (deductibles)

        # Solvency tracking
        self.is_ruined = False

        # Metrics tracking
        self.metrics_history: List[Dict[str, float]] = []

        # Store initial values for base comparisons (for exposure bases)
        self._initial_assets = config.initial_assets
        self._initial_equity = config.initial_assets

    # Properties for FinancialStateProvider protocol
    @property
    def current_revenue(self) -> float:
        """Get current revenue based on current assets and turnover ratio."""
        return self.calculate_revenue()

    @property
    def current_assets(self) -> float:
        """Get current total assets."""
        return self.total_assets

    @property
    def current_equity(self) -> float:
        """Get current equity value."""
        return self.equity

    @property
    def base_revenue(self) -> float:
        """Get base (initial) revenue for comparison."""
        return self._initial_assets * self.config.asset_turnover_ratio

    @property
    def base_assets(self) -> float:
        """Get base (initial) assets for comparison."""
        return self._initial_assets

    @property
    def base_equity(self) -> float:
        """Get base (initial) equity for comparison."""
        return self._initial_equity

    @property
    def total_assets(self) -> float:
        """Calculate total assets from all asset components.

        Total assets include all current and non-current assets following
        the accounting equation: Assets = Liabilities + Equity.

        Returns:
            float: Total assets in dollars, sum of all asset components.
        """
        # Current assets
        current = self.cash + self.accounts_receivable + self.inventory + self.prepaid_insurance
        # Non-current assets
        net_ppe = self.gross_ppe - self.accumulated_depreciation
        # Total
        return current + net_ppe + self.restricted_assets

    @total_assets.setter
    def total_assets(self, value: float) -> None:
        """Set total assets by proportionally adjusting all asset components.

        This setter maintains the relative proportions of all asset components
        when changing the total asset value. If current assets are zero,
        it sets cash to the full value.

        Args:
            value: New total asset value in dollars.
        """
        current_total = self.total_assets

        # Handle zero or negative values
        if value <= 0:
            self.cash = 0
            self.accounts_receivable = 0
            self.inventory = 0
            self.prepaid_insurance = 0
            self.gross_ppe = 0
            self.restricted_assets = 0
            return

        # If current total is zero, put everything in cash
        if current_total <= 0:
            self.cash = value
            self.accounts_receivable = 0
            self.inventory = 0
            self.prepaid_insurance = 0
            self.gross_ppe = 0
            self.restricted_assets = 0
            return

        # Calculate adjustment ratio
        ratio = value / current_total

        # Adjust all asset components proportionally
        self.cash *= ratio
        self.accounts_receivable *= ratio
        self.inventory *= ratio
        self.prepaid_insurance *= ratio
        self.gross_ppe *= ratio
        self.accumulated_depreciation *= ratio
        self.restricted_assets *= ratio

    @property
    def total_liabilities(self) -> float:
        """Calculate total liabilities from all liability components.

        Total liabilities include current liabilities and long-term claim liabilities.

        Returns:
            float: Total liabilities in dollars, sum of all liability components.
        """
        # Get accrued expenses from accrual manager (includes taxes, wages, etc.)
        accrual_items = self.accrual_manager.get_balance_sheet_items()
        total_accrued_expenses = accrual_items.get("accrued_expenses", 0)
        # Also include any legacy accrued_expenses not in the manager
        total_accrued = max(self.accrued_expenses, total_accrued_expenses)

        # Current liabilities
        current_liabilities = self.accounts_payable + total_accrued

        # Long-term liabilities (claim liabilities)
        claim_liability_total = sum(
            liability.remaining_amount for liability in self.claim_liabilities
        )

        return current_liabilities + claim_liability_total

    @property
    def equity(self) -> float:
        """Calculate equity using the accounting equation.

        Equity is derived as Assets - Liabilities, ensuring the accounting
        equation always balances: Assets = Liabilities + Equity.

        Returns:
            float: Shareholder equity in dollars.
        """
        return self.total_assets - self.total_liabilities

    @property
    def net_assets(self) -> float:
        """Calculate net assets (total assets minus restricted assets).

        Net assets represent the portion of total assets that are available
        for operational use. Restricted assets are those pledged as collateral
        for insurance claims and cannot be used for general business purposes.

        Returns:
            float: Net assets in dollars. Always non-negative as restricted
                assets cannot exceed total assets.

        Examples:
            Track net assets after claims::

                manufacturer = WidgetManufacturer(config)
                initial_net = manufacturer.net_assets  # $10,000,000

                # Process claim requiring $2M collateral
                manufacturer.process_insurance_claim(
                    claim_amount=5_000_000,
                    deductible=1_000_000,
                    insurance_limit=10_000_000
                )

                reduced_net = manufacturer.net_assets  # $8,000,000

        See Also:
            :attr:`available_assets`: Alias for this property.
            :attr:`restricted_assets`: Assets pledged as collateral.
        """
        return float(self.total_assets - self.restricted_assets)

    def record_insurance_premium(self, premium_amount: float, is_annual: bool = False) -> None:
        """Record insurance premium payment with proper GAAP prepaid asset treatment.

        This method records insurance premium payments either as prepaid assets
        (for annual premiums) or as direct expenses (for monthly premiums).
        Annual premiums are recorded as prepaid assets and amortized monthly.

        For backward compatibility, defaults to direct expense (is_annual=False).

        Args:
            premium_amount (float): Premium amount paid in the current period.
                Must be >= 0.
            is_annual (bool): Whether this is an annual premium payment (default False).
                If True, creates prepaid asset. If False, records as direct expense.

        Examples:
            Record annual premium payment::

                # Pay annual insurance premium
                annual_premium = 250_000
                manufacturer.record_insurance_premium(annual_premium, is_annual=True)

                # Premium creates prepaid asset, monthly expense will be amortized

        Side Effects:
            - For annual premiums: Creates prepaid asset and sets monthly amortization
            - For monthly premiums: Records direct period expense
            - Updates cash position for premium payment

        Note:
            Annual premiums should be paid at the start of the coverage period.
            Monthly amortization happens automatically in the step() method.

        See Also:
            :meth:`calculate_net_income`: Uses tracked premiums for tax calculations.
            :class:`InsuranceAccounting`: Handles premium amortization logic.
        """
        if premium_amount > 0:
            if is_annual:
                # Record as prepaid asset using insurance accounting module
                result = self.insurance_accounting.pay_annual_premium(premium_amount)

                # Update balance sheet
                self.prepaid_insurance = result["prepaid_asset"]
                self.cash -= result["cash_outflow"]

                # Store for later amortization tracking
                self._original_prepaid_premium = premium_amount

                logger.info(f"Paid annual insurance premium: ${premium_amount:,.2f}")
                logger.info(f"Monthly expense will be: ${result['monthly_expense']:,.2f}")
            else:
                # Record as direct expense for the period (backward compatibility)
                self.period_insurance_premiums += premium_amount
                # Don't reduce cash here - expense is handled through net income calculation
                # to avoid double-counting (premiums reduce operating income -> net income -> equity)

                logger.info(f"Recorded insurance premium expense: ${premium_amount:,.2f}")
                logger.debug(f"Period premiums total: ${self.period_insurance_premiums:,.2f}")

    def record_insurance_loss(self, loss_amount: float) -> None:
        """Record insurance loss (deductible/retention) for tax deduction tracking.

        This method tracks insurance losses paid by the company during the current
        period for proper tax treatment. Company-paid losses (deductibles, retentions,
        excess over limits) are tax-deductible business expenses.

        Args:
            loss_amount (float): Loss amount paid by company in the current period.
                Must be >= 0.

        Examples:
            Record deductible payment on claim::

                # Company pays $500K deductible on claim
                deductible_paid = 500_000
                manufacturer.record_insurance_loss(deductible_paid)

                # Loss will be tax-deductible in next net income calculation

        Side Effects:
            - Increases period_insurance_losses by loss_amount
            - No immediate impact on assets/equity (handled by payment schedule)

        Note:
            This method is automatically called by process_insurance_claim()
            for the company payment portion. The tax deduction is taken when
            the loss is incurred (accrual basis), not when paid (cash basis).

        See Also:
            :meth:`calculate_net_income`: Uses tracked losses for tax calculations.
            :meth:`process_insurance_claim`: Automatically records company losses.
        """
        if loss_amount > 0:
            self.period_insurance_losses += loss_amount
            logger.debug(f"Recorded insurance loss: ${loss_amount:,.2f}")
            logger.debug(f"Period losses total: ${self.period_insurance_losses:,.2f}")

    def reset_period_insurance_costs(self) -> None:
        """Reset period insurance cost tracking for new period.

        This method clears the accumulated insurance premiums and losses
        from the current period, typically called at the end of each
        simulation step to prepare for the next period.

        Side Effects:
            - Resets period_insurance_premiums to 0.0
            - Resets period_insurance_losses to 0.0

        Note:
            Called automatically at the end of each step() to ensure
            costs are only counted once per period.
        """
        self.period_insurance_premiums = 0.0
        self.period_insurance_losses = 0.0

    @property
    def available_assets(self) -> float:
        """Calculate available (unrestricted) assets for operations.

        This property is an alias for net_assets, providing semantic clarity
        when referring to assets available for business operations. These are
        the assets the company can freely use without violating collateral
        requirements.

        Returns:
            float: Available assets in dollars. Equal to total assets minus
                restricted assets pledged as insurance collateral.

        Examples:
            Check operational capacity::

                if manufacturer.available_assets < minimum_operating_cash:
                    logger.warning("Low available assets for operations")

                # Calculate maximum dividend possible
                max_dividend = manufacturer.available_assets * 0.1

        See Also:
            :attr:`net_assets`: Identical calculation with different semantic meaning.
            :attr:`restricted_assets`: Assets not available for operations.
        """
        return float(self.total_assets - self.restricted_assets)

    @property
    def total_claim_liabilities(self) -> float:
        """Calculate total outstanding claim liabilities.

        Sums the remaining unpaid amounts across all active claim liabilities.
        This represents the company's total financial obligation for insurance
        claims that are being paid over time according to development schedules.

        Returns:
            float: Total outstanding liability in dollars. Returns 0.0 if no
                active claims exist.

        Examples:
            Monitor outstanding liabilities::

                # Process several claims
                manufacturer.process_insurance_claim(2_000_000, 500_000, 5_000_000)
                manufacturer.process_insurance_claim(1_000_000, 200_000, 3_000_000)

                # Check total outstanding
                total = manufacturer.total_claim_liabilities
                print(f"Outstanding claims: ${total:,.2f}")

                # After payments in subsequent years
                manufacturer.pay_claim_liabilities()
                remaining = manufacturer.total_claim_liabilities

        Note:
            This amount should equal the total collateral posted for insurance
            claims, as collateral is required dollar-for-dollar with claim
            liabilities.

        See Also:
            :attr:`collateral`: Should equal this amount for consistency.
            :meth:`pay_claim_liabilities`: Method that reduces these liabilities.
        """
        return sum(claim.remaining_amount for claim in self.claim_liabilities)

    def calculate_revenue(
        self, working_capital_pct: float = 0.0, apply_stochastic: bool = False
    ) -> float:
        """Calculate revenue based on available assets and turnover ratio.

        Revenue is calculated using the asset turnover ratio, which represents
        how efficiently the company converts assets into sales. The calculation
        accounts for working capital requirements and can include stochastic
        shocks for realistic modeling.

        The working capital adjustment recognizes that some portion of revenue
        is tied up in inventory and receivables, reducing the effective assets
        available for revenue generation.

        Args:
            working_capital_pct (float): Percentage of revenue tied up in working
                capital (inventory + receivables - payables). Typical values
                range from 0.15 to 0.25. Defaults to 0.0 for simplified modeling.
            apply_stochastic (bool): Whether to apply stochastic shock to revenue
                calculation. Requires stochastic_process to be initialized.
                Defaults to False for deterministic calculation.

        Returns:
            float: Annual revenue in dollars. Always non-negative.

        Raises:
            ValueError: If working_capital_pct < 0 or >= 1.
            RuntimeError: If apply_stochastic=True but no stochastic process provided.

        Examples:
            Basic revenue calculation::

                # Deterministic revenue
                revenue = manufacturer.calculate_revenue()
                print(f"Base revenue: ${revenue:,.2f}")

            With working capital::

                # Account for 20% working capital
                revenue_wc = manufacturer.calculate_revenue(
                    working_capital_pct=0.2
                )
                # Revenue will be lower due to working capital tie-up

            With stochastic shocks::

                from ergodic_insurance.stochastic_processes import (
                    LognormalVolatility
                )

                shock_process = LognormalVolatility(volatility=0.15)
                manufacturer = WidgetManufacturer(config, shock_process)

                # Revenue with random volatility
                volatile_revenue = manufacturer.calculate_revenue(
                    apply_stochastic=True
                )

        Note:
            The asset turnover ratio can be modified during simulation to model
            business growth or decline. Working capital adjustments follow the
            formula: Effective Assets = Total Assets / (1 + Turnover * WC%).

        See Also:
            :attr:`asset_turnover_ratio`: Core parameter for revenue calculation.
            :class:`~ergodic_insurance.stochastic_processes.StochasticProcess`:
            For stochastic modeling options.
        """
        # Adjust for working capital if specified
        # Ensure assets are non-negative for revenue calculation
        # (negative assets would mean business has ceased operations)
        available_assets = max(0, self.total_assets)
        if working_capital_pct > 0:
            # Working capital reduces assets available for operations
            # Revenue = Available Assets * Turnover, where
            # Available Assets = Total Assets - Working Capital
            # Working Capital = Revenue * working_capital_pct
            # Solving: Revenue = Assets * Turnover / (1 + Turnover * WC%)
            denominator = 1 + self.asset_turnover_ratio * working_capital_pct
            available_assets = max(0, self.total_assets) / denominator

        revenue = available_assets * self.asset_turnover_ratio

        # Apply stochastic shock if requested and process is available
        if apply_stochastic and self.stochastic_process is not None:
            shock = self.stochastic_process.generate_shock(revenue)
            revenue *= shock
            logger.debug(f"Applied stochastic shock: {shock:.4f}")

        logger.debug(f"Revenue calculated: ${revenue:,.2f} from assets ${self.total_assets:,.2f}")
        return float(revenue)

    def calculate_operating_income(
        self, revenue: float, depreciation_expense: float = 0.0
    ) -> float:
        """Calculate operating income including insurance and depreciation as operating expenses.

        Operating income represents earnings before interest and taxes (EBIT),
        calculated by applying the base operating margin to revenue and then
        subtracting insurance costs (premiums and losses) and depreciation.
        This reflects the true operating profitability after all operating expenses.

        Args:
            revenue (float): Annual revenue in dollars. Must be >= 0.
            depreciation_expense (float): Depreciation expense for the period.
                Defaults to 0.0 for backward compatibility.

        Returns:
            float: Operating income in dollars after insurance costs and depreciation.
                Equal to (revenue * base_operating_margin) - insurance costs - depreciation.

        Examples:
            Calculate operating income with insurance and depreciation::

                revenue = manufacturer.calculate_revenue()
                depreciation = manufacturer.gross_ppe / 10  # 10-year useful life
                operating_income = manufacturer.calculate_operating_income(revenue, depreciation)

                # Actual margin will be lower than base margin due to insurance and depreciation
                actual_margin = operating_income / revenue

        Note:
            The base operating margin represents the core margin before insurance
            and depreciation. Actual operating margins will be lower when these
            costs are included.

        See Also:
            :attr:`base_operating_margin`: The core margin percentage before expenses.
            :meth:`calculate_net_income`: Includes financing costs and taxes.
        """
        # Calculate base operating income using base margin
        base_operating_income = revenue * self.base_operating_margin

        # Subtract insurance costs and depreciation to get actual operating income
        actual_operating_income = (
            base_operating_income
            - self.period_insurance_premiums
            - self.period_insurance_losses
            - depreciation_expense
        )

        logger.debug(
            f"Operating income: ${actual_operating_income:,.2f} "
            f"(base: ${base_operating_income:,.2f}, insurance: "
            f"${self.period_insurance_premiums + self.period_insurance_losses:,.2f}, "
            f"depreciation: ${depreciation_expense:,.2f})"
        )
        return float(actual_operating_income)

    def calculate_collateral_costs(
        self, letter_of_credit_rate: float = 0.015, time_period: str = "annual"
    ) -> float:
        """Calculate costs for letter of credit collateral.

        Letter of credit costs represent the financing expense for collateral
        posted to support insurance claim liabilities. These costs are similar
        to interest expense and reduce the company's profitability.

        Args:
            letter_of_credit_rate (float): Annual interest rate for letter of
                credit. Market rates typically range from 0.01 to 0.02 (1-2%).
                Defaults to 0.015 (1.5%).
            time_period (str): Time period for cost calculation. Must be "annual"
                or "monthly". "monthly" scales rate by 1/12. Defaults to "annual".

        Returns:
            float: Collateral costs for the specified period in dollars.
                Returns 0.0 if no collateral is posted.

        Examples:
            Calculate annual collateral costs::

                # After processing claims with $5M insurance coverage
                annual_cost = manufacturer.calculate_collateral_costs(
                    letter_of_credit_rate=0.015
                )
                # If collateral = $5M, cost = $75,000 annually

            Monthly cost tracking::

                monthly_cost = manufacturer.calculate_collateral_costs(
                    letter_of_credit_rate=0.015,
                    time_period="monthly"
                )
                # Monthly cost = annual_cost / 12

        Raises:
            ValueError: If time_period not "annual" or "monthly".
            ValueError: If letter_of_credit_rate < 0.

        Note:
            Collateral costs are calculated on the full collateral amount,
            regardless of when individual claims will be paid. This reflects
            the banking requirement to maintain full collateral availability.

        See Also:
            :attr:`collateral`: Amount of collateral posted.
            :meth:`calculate_net_income`: Includes these costs in profit calculation.
        """
        if time_period == "monthly":
            period_rate = letter_of_credit_rate / 12
        else:
            period_rate = letter_of_credit_rate

        collateral_costs = self.collateral * period_rate
        if collateral_costs > 0:
            logger.debug(
                f"Collateral costs ({time_period}): ${collateral_costs:,.2f} on ${self.collateral:,.2f} collateral"
            )
        return collateral_costs

    def calculate_net_income(
        self,
        operating_income: float,
        collateral_costs: float,
        insurance_premiums: float = 0.0,
        insurance_losses: float = 0.0,
        use_accrual: bool = True,
        time_resolution: str = "annual",
    ) -> float:
        """Calculate net income after collateral costs and taxes.

        Net income represents the final profitability available to shareholders
        after all operating expenses, financing costs, and taxes. Insurance costs
        can be handled in two ways for backward compatibility:
        1. New way: Already included in operating_income via calculate_operating_income()
        2. Legacy way: Passed as separate parameters to this method

        Args:
            operating_income (float): Operating income (EBIT). May or may not include
                insurance costs depending on how it was calculated.
            collateral_costs (float): Financing costs for letter of credit
                collateral. Must be >= 0.
            insurance_premiums (float): Insurance premium costs to deduct. If operating_income
                already includes these, pass 0. Defaults to 0.0.
            insurance_losses (float): Insurance loss/deductible costs to deduct. If operating_income
                already includes these, pass 0. Defaults to 0.0.
            use_accrual (bool): Whether to use accrual accounting for taxes.
                Defaults to True for quarterly tax payment schedule.
            time_resolution (str): Time resolution for tax accrual calculation.
                "annual" accrues full annual taxes, "monthly" accrues monthly portion.
                Defaults to "annual".

        Returns:
            float: Net income after all expenses and taxes. Can be negative
                if the company operates at a loss after financing costs and taxes.

        Examples:
            Calculate full income statement with insurance costs::

                revenue = manufacturer.calculate_revenue(working_capital_pct=0.2)
                operating_income = manufacturer.calculate_operating_income(revenue)
                collateral_costs = manufacturer.calculate_collateral_costs(0.015)
                net_income = manufacturer.calculate_net_income(
                    operating_income, collateral_costs,
                    insurance_premiums=500_000, insurance_losses=200_000
                )

                # Net margin
                net_margin = net_income / revenue if revenue > 0 else 0

        Note:
            Tax treatment follows proper accounting principles:
            - Insurance premiums are tax-deductible business expenses
            - Insurance losses/claims are tax-deductible business expenses
            - Collateral costs are tax-deductible financing expenses
            - Taxes are only applied to positive pre-tax income
            - Loss years generate no tax benefit in this model
            - When use_accrual=True, taxes are accrued and paid quarterly

            The tax calculation is:
            - Income before tax = operating_income - collateral_costs
              (Note: insurance costs are already included in operating_income)
            - Taxes = max(0, income_before_tax * tax_rate)
            - Net income = income_before_tax - taxes

        See Also:
            :attr:`tax_rate`: Tax rate applied to positive income.
            :attr:`retention_ratio`: Portion of net income retained vs. distributed.
        """
        # Deduct all costs from operating income
        # For backward compatibility, also deduct insurance costs if provided as parameters
        total_insurance_costs = insurance_premiums + insurance_losses
        income_before_tax = operating_income - collateral_costs - total_insurance_costs

        # Calculate taxes (only on positive income)
        taxes = max(0, income_before_tax * self.tax_rate)

        # Handle tax accruals if enabled
        if use_accrual and taxes > 0:
            # In monthly mode, only accrue taxes at specific points to avoid duplication
            # Taxes are accrued quarterly in months 2, 5, 8, 11 (for Q1, Q2, Q3, Q4)
            # This aligns with quarterly estimated tax payment requirements
            should_accrue = False
            payment_dates = None
            if time_resolution == "annual":
                # In annual mode, accrue the full year's taxes
                # Payments will be made quarterly in the NEXT year
                should_accrue = True
                description = f"Year {self.current_year} tax liability"
                # Set payment dates for next year's quarterly payments
                next_year_base = (self.current_year + 1) * 12
                payment_dates = [next_year_base + month for month in [3, 5, 8, 11]]
            elif time_resolution == "monthly" and self.current_month in [2, 5, 8, 11]:
                # In monthly mode, accrue quarterly taxes at end of each quarter
                should_accrue = True
                quarter = (self.current_month // 3) + 1
                description = f"Year {self.current_year} Q{quarter} tax liability"
                # For quarterly accruals, use immediate payment
                payment_dates = None  # Will use default QUARTERLY schedule

            if should_accrue:
                # Record tax expense as accrual with quarterly payment schedule
                if payment_dates:
                    # Use custom payment dates for annual accrual
                    self.accrual_manager.record_expense_accrual(
                        item_type=AccrualType.TAXES,
                        amount=taxes,
                        payment_schedule=PaymentSchedule.CUSTOM,
                        payment_dates=payment_dates,
                        description=description,
                    )
                else:
                    # Use default quarterly schedule
                    self.accrual_manager.record_expense_accrual(
                        item_type=AccrualType.TAXES,
                        amount=taxes,
                        payment_schedule=PaymentSchedule.QUARTERLY,
                        description=description,
                    )

        net_income = income_before_tax - taxes

        # Enhanced profit waterfall logging for complete transparency
        logger.info("===== PROFIT WATERFALL =====")
        logger.info(f"Operating Income:        ${operating_income:,.2f}")
        if insurance_premiums > 0:
            logger.info(f"  - Insurance Premiums:  ${insurance_premiums:,.2f}")
        if insurance_losses > 0:
            logger.info(f"  - Insurance Losses:    ${insurance_losses:,.2f}")
        if collateral_costs > 0:
            logger.info(f"  - Collateral Costs:    ${collateral_costs:,.2f}")
        logger.info(f"Income Before Tax:       ${income_before_tax:,.2f}")
        logger.info(f"  - Taxes (@{self.tax_rate:.1%}):      ${taxes:,.2f}")
        if use_accrual and taxes > 0:
            logger.info(f"    (Accrued for quarterly payment)")
        logger.info(f"NET INCOME:              ${net_income:,.2f}")
        logger.info("============================")

        # Validation assertion: ensure net income is less than or equal to operating income
        # when additional costs exist (beyond those already in operating income)
        # Note: Since insurance costs may already be included in operating_income,
        # we only check for meaningful differences
        if (
            total_insurance_costs + collateral_costs > 1e-9
        ):  # Use small epsilon for float comparison
            assert (
                net_income <= operating_income + 1e-9  # Allow for floating point precision
            ), f"Net income ({net_income}) should be less than or equal to operating income ({operating_income}) when costs exist"

        return float(net_income)

    def update_balance_sheet(self, net_income: float, growth_rate: float = 0.0) -> None:
        """Update balance sheet with retained earnings and dividend distribution.

        This method processes the financial results of a period by allocating
        net income between retained earnings (which increase assets and equity)
        and dividend payments (which are distributed to shareholders). The
        allocation is controlled by the retention ratio configuration.

        Args:
            net_income (float): Net income for the period in dollars. Can be
                negative for loss periods, which will reduce equity.
            growth_rate (float): Revenue growth rate parameter. Currently unused
                but maintained for interface compatibility. Defaults to 0.0.

        Examples:
            Process profitable period::

                net_income = 800_000
                initial_equity = manufacturer.equity

                manufacturer.update_balance_sheet(net_income)

                # With 70% retention ratio
                retained = net_income * 0.7  # $560,000
                dividends = net_income * 0.3  # $240,000

                assert manufacturer.equity == initial_equity + retained

            Handle loss period::

                net_income = -200_000  # Operating loss
                initial_equity = manufacturer.equity

                manufacturer.update_balance_sheet(net_income)

                # Loss reduces equity, no dividends paid
                loss_retained = net_income * manufacturer.retention_ratio
                assert manufacturer.equity == initial_equity + loss_retained

        Side Effects:
            - Increases assets by retained earnings amount
            - Increases equity by retained earnings amount
            - Implicitly distributes dividends (reduces available cash)
            - Logs balance sheet changes and dividend payments

        Note:
            The method assumes no debt financing, so equity changes equal
            asset changes. Dividend payments are implicitly handled by not
            adding them to assets, rather than explicitly reducing cash.

            For loss periods, the full loss (scaled by retention ratio) reduces
            equity, reflecting the shareholders' absorption of losses.

        See Also:
            :attr:`retention_ratio`: Fraction of earnings retained vs. distributed.
            :attr:`assets`: Total assets updated by retained earnings.
            :attr:`equity`: Shareholder equity updated by retained earnings.
        """
        # Validation: retention ratio should be applied to net income (not revenue or operating income)
        # This is the profit after ALL costs including taxes
        assert 0 <= self.retention_ratio <= 1, f"Invalid retention ratio: {self.retention_ratio}"

        # Calculate retained earnings
        retained_earnings = net_income * self.retention_ratio
        dividends = net_income * (1 - self.retention_ratio)

        # Log retention calculation details
        logger.info("===== RETENTION CALCULATION =====")
        logger.info(f"Net Income:              ${net_income:,.2f}")
        logger.info(f"Retention Ratio:         {self.retention_ratio:.1%}")
        logger.info(f"Retained Earnings:       ${retained_earnings:,.2f}")
        if net_income > 0:
            logger.info(f"Dividends Distributed:   ${dividends:,.2f}")
        else:
            logger.info(f"Loss Absorption:         ${retained_earnings:,.2f}")
        logger.info("=================================")

        # Add retained earnings to cash (increases assets, which increases equity through accounting equation)
        # Allow cash to go negative to properly detect insolvency
        self.cash = self.cash + retained_earnings

        logger.info(
            f"Balance sheet updated: Assets=${self.total_assets:,.2f}, Equity=${self.equity:,.2f}"
        )
        if dividends > 0:
            logger.info(f"Dividends paid: ${dividends:,.2f}")

    def calculate_working_capital_components(
        self, revenue: float, dso: float = 45, dio: float = 60, dpo: float = 30
    ) -> Dict[str, float]:
        """Calculate individual working capital components based on revenue and ratios.

        Uses standard financial ratios to calculate accounts receivable, inventory,
        and accounts payable from annual revenue. These components provide detailed
        insight into the company's working capital management.

        Args:
            revenue (float): Annual revenue in dollars.
            dso (float): Days Sales Outstanding - average collection period.
                Typical manufacturing: 30-60 days. Defaults to 45.
            dio (float): Days Inventory Outstanding - average inventory holding period.
                Typical manufacturing: 45-90 days. Defaults to 60.
            dpo (float): Days Payable Outstanding - average payment period.
                Typical manufacturing: 30-45 days. Defaults to 30.

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'accounts_receivable': Outstanding customer receivables
                - 'inventory': Value of inventory on hand
                - 'accounts_payable': Outstanding vendor payables
                - 'net_working_capital': AR + Inventory - AP
                - 'cash_conversion_cycle': DSO + DIO - DPO in days

        Examples:
            Calculate working capital with $10M revenue::

                components = manufacturer.calculate_working_capital_components(
                    revenue=10_000_000,
                    dso=45,  # 45 days to collect
                    dio=60,  # 60 days of inventory
                    dpo=30   # Pay vendors in 30 days
                )

                # Expected values:
                # AR = 10M * 45/365 = ~$1.23M
                # Inventory = 10M * 60/365 = ~$1.64M
                # AP = 10M * 30/365 = ~$0.82M
                # Net WC = $1.23M + $1.64M - $0.82M = ~$2.05M
        """
        # Calculate cost of goods sold (approximate as % of revenue)
        cogs = revenue * (1 - self.base_operating_margin)

        # Calculate new working capital components
        new_ar = revenue * (dso / 365)
        new_inventory = cogs * (dio / 365)  # Inventory based on COGS not revenue
        new_ap = cogs * (dpo / 365)  # AP based on COGS not revenue

        # Calculate the change in working capital assets
        ar_change = new_ar - self.accounts_receivable
        inventory_change = new_inventory - self.inventory
        ap_change = new_ap - self.accounts_payable

        # Update components
        self.accounts_receivable = new_ar
        self.inventory = new_inventory
        self.accounts_payable = new_ap

        # Note: Working capital is just a reallocation of assets, not a change in total assets
        # AR and inventory are funded from operations, not from reducing cash
        # The cash impact happens through the revenue/expense cycle, not here

        # Calculate net working capital and cash conversion cycle
        net_working_capital = self.accounts_receivable + self.inventory - self.accounts_payable
        cash_conversion_cycle = dso + dio - dpo

        logger.debug(
            f"Working capital components: AR=${self.accounts_receivable:,.0f}, "
            f"Inv=${self.inventory:,.0f}, AP=${self.accounts_payable:,.0f}, "
            f"Net WC=${net_working_capital:,.0f}"
        )

        return {
            "accounts_receivable": self.accounts_receivable,
            "inventory": self.inventory,
            "accounts_payable": self.accounts_payable,
            "net_working_capital": net_working_capital,
            "cash_conversion_cycle": cash_conversion_cycle,
        }

    def record_prepaid_insurance(self, annual_premium: float) -> None:
        """Record annual insurance premium payment as prepaid expense.

        Records the payment of an annual insurance premium as a prepaid asset
        that will be amortized monthly over the coverage period.

        Args:
            annual_premium (float): Annual insurance premium paid in advance.

        Side Effects:
            - Increases prepaid_insurance by annual_premium
            - Decreases cash by annual_premium

        Examples:
            Record annual premium payment::

                manufacturer.record_prepaid_insurance(1_200_000)
                # Prepaid insurance increases by $1.2M
                # Will amortize at $100K/month over 12 months
        """
        if annual_premium > 0:
            # Use insurance accounting module to properly track prepaid insurance
            result = self.insurance_accounting.pay_annual_premium(annual_premium)

            # Update balance sheet
            self.prepaid_insurance = result["prepaid_asset"]
            self.cash -= result["cash_outflow"]

            logger.info(f"Recorded prepaid insurance: ${annual_premium:,.2f}")

    def amortize_prepaid_insurance(self, months: int = 1) -> float:
        """Amortize prepaid insurance over time using GAAP straight-line method.

        Reduces prepaid insurance balance and records the expense for the period.
        Typically called monthly to amortize annual premiums. Uses the insurance
        accounting module to ensure proper amortization calculations.

        Args:
            months (int): Number of months to amortize. Defaults to 1.

        Returns:
            float: Amount amortized (insurance expense for the period).

        Side Effects:
            - Decreases prepaid_insurance by amortization amount
            - Increases period_insurance_premiums by amortization amount
            - Updates insurance accounting records

        Examples:
            Monthly amortization::

                # After paying $1.2M annual premium
                monthly_expense = manufacturer.amortize_prepaid_insurance(1)
                # Returns $100K, reduces prepaid by $100K
        """
        total_amortized = 0.0

        # Use insurance accounting module for proper amortization
        for _ in range(months):
            if self.prepaid_insurance > 0:
                result = self.insurance_accounting.record_monthly_expense()

                # Update balance sheet and P&L
                self.prepaid_insurance = result["remaining_prepaid"]
                self.period_insurance_premiums += result["insurance_expense"]
                total_amortized += result["insurance_expense"]

                logger.debug(
                    f"Month {self.insurance_accounting.current_month}: "
                    f"Expense ${result['insurance_expense']:,.2f}, "
                    f"Remaining prepaid ${result['remaining_prepaid']:,.2f}"
                )

        return total_amortized

    def receive_insurance_recovery(
        self, amount: float, claim_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Receive payment from insurance for a claim recovery.

        Records cash receipt from insurance company for previously approved
        claim recoveries. Updates cash position and reduces insurance receivables.

        Args:
            amount (float): Amount received from insurance company.
            claim_id (Optional[str]): Specific claim ID for the recovery.
                If None, applies to oldest outstanding recovery.

        Returns:
            Dictionary with payment details:
                - cash_received: Amount of cash received
                - receivable_reduction: Reduction in receivables
                - remaining_receivables: Total outstanding receivables

        Examples:
            Receive insurance payment::

                # Receive $500K insurance recovery payment
                result = manufacturer.receive_insurance_recovery(500_000)
                print(f"Received ${result['cash_received']:,.2f}")
                print(f"Outstanding: ${result['remaining_receivables']:,.2f}")

        Side Effects:
            - Increases cash by payment amount
            - Reduces insurance receivables
            - Updates insurance accounting records
        """
        if amount <= 0:
            return {"cash_received": 0, "receivable_reduction": 0, "remaining_receivables": 0}

        # Record receipt through insurance accounting module
        result = self.insurance_accounting.receive_recovery_payment(amount, claim_id)

        # Update cash position
        self.cash += result["cash_received"]

        logger.info(f"Received insurance recovery: ${result['cash_received']:,.2f}")
        logger.debug(f"Remaining receivables: ${result['remaining_receivables']:,.2f}")

        return result

    def record_depreciation(self, useful_life_years: float = 10) -> float:
        """Record straight-line depreciation on PP&E.

        Calculates and records annual depreciation expense using the straight-line
        method. Depreciation reduces the net book value of fixed assets over time.

        Args:
            useful_life_years (float): Average useful life of PP&E in years.
                Typical manufacturing equipment: 7-15 years. Defaults to 10.

        Returns:
            float: Annual depreciation expense recorded.

        Side Effects:
            - Increases accumulated_depreciation by depreciation amount
            - Does not directly affect cash (non-cash expense)

        Examples:
            Record annual depreciation::

                # With $10M gross PP&E and 10-year life
                depreciation = manufacturer.record_depreciation(10)
                # Returns $1M, increases accumulated depreciation by $1M

        Note:
            Depreciation is a non-cash expense that reduces taxable income
            but does not affect cash flow directly.
        """
        if self.gross_ppe > 0 and useful_life_years > 0:
            annual_depreciation = self.gross_ppe / useful_life_years

            # Don't depreciate below zero net book value
            net_ppe = self.gross_ppe - self.accumulated_depreciation
            if net_ppe > 0:
                depreciation_expense = min(annual_depreciation, net_ppe)
                self.accumulated_depreciation += depreciation_expense

                logger.debug(
                    f"Recorded depreciation: ${depreciation_expense:,.2f}, "
                    f"Accumulated: ${self.accumulated_depreciation:,.2f}"
                )
                return depreciation_expense
        return 0.0

    @property
    def net_ppe(self) -> float:
        """Calculate net property, plant & equipment after depreciation.

        Returns:
            float: Net PP&E (gross PP&E minus accumulated depreciation).
        """
        return self.gross_ppe - self.accumulated_depreciation

    def process_insurance_claim(
        self,
        claim_amount: float,
        deductible_amount: float = 0.0,
        insurance_limit: float = float("inf"),
        insurance_recovery: Optional[float] = None,
    ) -> tuple[float, float]:
        """Process an insurance claim with deductible and limit, setting up collateral.

        This method handles the complete processing of an insurance claim,
        including immediate cash flows and establishment of collateral for
        the insurance portion. The company pays its deductible immediately,
        while the insurance portion creates a liability with associated
        letter of credit collateral.

        The method supports both legacy parameter style (deductible/limit) and
        preferred style (pre-calculated amounts) for integration with insurance
        program calculations.

        Args:
            claim_amount (float): Total amount of the loss/claim in dollars.
                Must be >= 0.
            deductible_amount (float): Amount company must pay before insurance kicks in
                (legacy parameter). Defaults to 0.0.
            insurance_limit (float): Maximum amount insurance will pay per claim
                (legacy parameter). Defaults to unlimited. Use insurance_recovery
                instead for new code.
            insurance_recovery (Optional[float]): Pre-calculated insurance recovery
                amount (preferred). If provided, overrides deductible/limit
                calculation. Should be the exact amount insurance will pay.

        Returns:
            tuple[float, float]: Tuple of (company_payment, insurance_payment)
                where:
                - company_payment: Amount paid immediately by the company
                - insurance_payment: Amount covered by insurance (creates liability)

        Examples:
            Basic claim processing::

                # $5M claim with $1M deductible, $10M limit
                company_paid, insurance_paid = manufacturer.process_insurance_claim(
                    claim_amount=5_000_000,
                    deductible=1_000_000,
                    insurance_limit=10_000_000
                )
                # company_paid = $1,000,000, insurance_paid = $4,000,000

            Using preferred parameter style::

                # Pre-calculated amounts from insurance program
                company_paid, insurance_paid = manufacturer.process_insurance_claim(
                    claim_amount=5_000_000,
                    insurance_recovery=3_800_000,  # After all layers
                    deductible_amount=1_200_000    # Company retention
                )

            Large claim exceeding limits::

                # $50M claim with $25M total insurance limit
                company_paid, insurance_paid = manufacturer.process_insurance_claim(
                    claim_amount=50_000_000,
                    deductible=1_000_000,
                    insurance_limit=25_000_000
                )
                # company_paid = $26,000,000 (deductible + excess over limit)
                # insurance_paid = $25,000,000

        Side Effects:
            - Increases collateral by company_payment amount (not insurance_payment)
            - Increases restricted_assets by company_payment amount
            - Creates ClaimLiability for company_payment with payment schedule
            - Insurance payment has no impact on company balance sheet
            - Assets/equity reduced over time as claim payments are made

        Note:
            The company portion creates a liability that will be paid over
            multiple years according to the ClaimLiability payment schedule.
            Collateral is posted immediately but released as payments are made.
            The insurance portion has no financial impact on the company.

        Warning:
            Large company payments may require significant collateral, restricting
            available assets for operations. Monitor available assets after
            processing large claims with high deductibles.

        See Also:
            :class:`ClaimLiability`: For understanding payment schedules.
            :meth:`pay_claim_liabilities`: For processing scheduled payments.
            :class:`~ergodic_insurance.insurance_program.InsuranceProgram`:
            For complex multi-layer insurance calculations.
        """
        # Handle new style parameters if provided
        if insurance_recovery is not None:
            # Use pre-calculated recovery
            insurance_payment = insurance_recovery
            company_payment = claim_amount - insurance_payment
        else:
            # Calculate insurance coverage
            if claim_amount <= deductible_amount:
                # Below deductible, company pays all
                company_payment = claim_amount
                insurance_payment = 0
            else:
                # Above deductible
                company_payment = deductible_amount
                insurance_payment = int(min(claim_amount - deductible_amount, insurance_limit))
                # Company also pays any amount above the limit
                if claim_amount > deductible_amount + insurance_limit:
                    company_payment += claim_amount - deductible_amount - insurance_limit

        # Company payment is collateralized and paid over time
        if company_payment > 0:
            # Post letter of credit as collateral for company payment
            # Transfer cash to restricted assets (no change in total assets)
            self.collateral += company_payment
            self.restricted_assets += company_payment
            self.cash -= company_payment  # Move cash to restricted

            # Create claim liability with payment schedule for company portion
            claim = ClaimLiability(
                original_amount=company_payment,
                remaining_amount=company_payment,
                year_incurred=self.current_year,
                is_insured=True,  # This is the company portion of an insured claim
            )
            self.claim_liabilities.append(claim)

            # Record the company payment (deductible) immediately for tax purposes
            # This ensures proper tax treatment in the same period as the claim
            self.record_insurance_loss(company_payment)

            logger.info(
                f"Company portion: ${company_payment:,.2f} - collateralized with payment schedule"
            )
            logger.info(
                f"Posted ${company_payment:,.2f} letter of credit as collateral for company portion"
            )

        # Insurance payment creates a receivable
        if insurance_payment > 0:
            # Record insurance recovery as receivable
            claim_id = f"CLAIM_{self.current_year}_{len(self.claim_liabilities)}"
            self.insurance_accounting.record_claim_recovery(
                recovery_amount=insurance_payment, claim_id=claim_id, year=self.current_year
            )
            logger.info(f"Insurance covering ${insurance_payment:,.2f} - recorded as receivable")

        logger.info(
            f"Total claim: ${claim_amount:,.2f} (Company: ${company_payment:,.2f}, Insurance: ${insurance_payment:,.2f})"
        )

        return company_payment, insurance_payment

    def process_uninsured_claim(
        self, claim_amount: float, immediate_payment: bool = False
    ) -> float:
        """Process an uninsured claim paid by company over time without collateral.

        This method handles claims where the company has no insurance coverage
        and must pay the full amount over time. Unlike insured claims, no collateral
        is required since there's no insurance company to secure payment to.

        Args:
            claim_amount (float): Total amount of the claim in dollars. Must be >= 0.
            immediate_payment (bool): If True, pays entire amount immediately.
                If False, creates liability with payment schedule. Defaults to False.

        Returns:
            float: The claim amount processed (for consistency with other methods).

        Examples:
            Process claim with payment schedule::

                # $500K claim paid over default schedule
                amount = manufacturer.process_uninsured_claim(500_000)
                # Creates liability, no immediate asset reduction

            Process claim with immediate payment::

                # $500K claim paid immediately
                amount = manufacturer.process_uninsured_claim(500_000, immediate_payment=True)
                # Immediately reduces assets and equity by $500K

        Side Effects:
            - If immediate_payment=True: Reduces assets and equity immediately
            - If immediate_payment=False: Creates ClaimLiability without collateral
            - Records claim amount as tax-deductible insurance loss

        Note:
            Unlike process_insurance_claim(), this method does not require collateral
            since there's no insurance company requiring security for payments.
        """
        if claim_amount <= 0:
            return 0.0

        if immediate_payment:
            # Pay immediately - reduce cash
            # First, try to pay from cash
            cash_payment = min(claim_amount, self.cash)
            remaining_to_pay = claim_amount - cash_payment

            # If cash isn't enough, liquidate other current assets proportionally
            if remaining_to_pay > 0:
                # Total liquid assets we can use (cash + AR + inventory)
                liquid_assets = self.cash + self.accounts_receivable + self.inventory
                if liquid_assets > 0:
                    # Pay what we can from liquid assets
                    actual_payment = min(claim_amount, liquid_assets)

                    # Reduce liquid assets proportionally
                    if liquid_assets > claim_amount:
                        # We have enough liquid assets
                        reduction_ratio = claim_amount / liquid_assets
                        self.cash *= 1 - reduction_ratio
                        self.accounts_receivable *= 1 - reduction_ratio
                        self.inventory *= 1 - reduction_ratio
                    else:
                        # Use all liquid assets
                        self.cash = 0
                        self.accounts_receivable = 0
                        self.inventory = 0
                else:
                    actual_payment = 0
            else:
                # Cash was sufficient
                actual_payment = cash_payment
                self.cash -= cash_payment

            # Record as tax-deductible loss
            self.period_insurance_losses += actual_payment

            # If we couldn't pay the full amount, create a liability for the shortfall
            shortfall = claim_amount - actual_payment
            if shortfall > 0:
                claim = ClaimLiability(
                    original_amount=shortfall,
                    remaining_amount=shortfall,
                    year_incurred=self.current_year,
                    is_insured=False,  # This is an uninsured claim
                )
                self.claim_liabilities.append(claim)
                logger.info(
                    f"Paid uninsured claim: ${actual_payment:,.2f} immediately, "
                    f"created liability for shortfall: ${shortfall:,.2f}"
                )
            else:
                logger.info(f"Paid uninsured claim immediately: ${actual_payment:,.2f}")
            return float(claim_amount)  # Return the full claim amount processed

        # Create liability without collateral for payment over time
        claim = ClaimLiability(
            original_amount=claim_amount,
            remaining_amount=claim_amount,
            year_incurred=self.current_year,
            is_insured=False,  # This is an uninsured claim
        )
        self.claim_liabilities.append(claim)
        logger.info(
            f"Created uninsured claim liability: ${claim_amount:,.2f} (no collateral required)"
        )
        return claim_amount

    def pay_claim_liabilities(self) -> float:
        """Pay scheduled claim liabilities for the current year.

        This method processes all scheduled claim payments based on each claim's
        development pattern and the years elapsed since incurrence. It maintains
        a minimum cash balance and reduces both collateral and restricted assets
        as payments are made.

        The method automatically removes fully paid claims from the active
        liability list to maintain clean accounting records.

        Returns:
            float: Total amount paid toward claims in dollars. May be less than
                scheduled if insufficient cash is available.

        Examples:
            Process annual claim payments::

                # After several years of operations
                initial_liabilities = manufacturer.total_claim_liabilities

                payments_made = manufacturer.pay_claim_liabilities()

                remaining_liabilities = manufacturer.total_claim_liabilities
                print(f"Paid ${payments_made:,.2f} toward claims")
                print(f"Remaining: ${remaining_liabilities:,.2f}")

            Check payment capacity::

                available_cash = manufacturer.assets - 100_000  # Minimum cash
                scheduled_payments = sum(
                    claim.get_payment(
                        manufacturer.current_year - claim.year_incurred
                    ) for claim in manufacturer.claim_liabilities
                )

                # Payments limited by cash availability
                expected_payment = min(available_cash, scheduled_payments)

        Side Effects:
            - Reduces assets by payment amounts
            - Reduces equity by payment amounts
            - Reduces collateral by payment amounts
            - Reduces restricted_assets by payment amounts
            - Updates remaining_amount for each claim
            - Removes fully paid claims from claim_liabilities list

        Note:
            The method maintains a minimum cash balance of $100,000 to ensure
            operational continuity. In severe cash constraints, this may result
            in deferred claim payments.

        Warning:
            Large scheduled payments may strain cash flow, especially if the
            company has limited assets relative to outstanding liabilities.
            Monitor solvency after calling this method.

        See Also:
            :class:`ClaimLiability`: Understanding payment schedules.
            :attr:`claim_liabilities`: List of active claims.
            :attr:`total_claim_liabilities`: Total outstanding amount.
        """
        total_paid = 0.0

        for claim in self.claim_liabilities:
            years_since = self.current_year - claim.year_incurred
            scheduled_payment = claim.get_payment(years_since)

            if scheduled_payment > 0:
                if claim.is_insured:
                    # For insured claims: Pay from restricted assets (collateral)
                    available_for_payment = min(scheduled_payment, self.restricted_assets)
                    actual_payment = available_for_payment

                    if actual_payment > 0:
                        claim.make_payment(actual_payment)
                        total_paid += actual_payment
                        # The collateral was set aside for this purpose
                        self.restricted_assets -= actual_payment
                        self.collateral -= actual_payment
                        logger.debug(
                            f"Reduced collateral and restricted assets by ${actual_payment:,.2f}"
                        )
                        # Do NOT record as tax-deductible loss here - already recorded when claim incurred
                else:
                    # For uninsured claims: Pay from available cash
                    available_for_payment = max(0, self.cash - 100_000)  # Keep minimum cash
                    actual_payment = min(scheduled_payment, available_for_payment)

                    if actual_payment > 0:
                        claim.make_payment(actual_payment)
                        total_paid += actual_payment
                        self.cash -= actual_payment  # Reduce cash for uninsured claims
                        logger.debug(
                            f"Paid ${actual_payment:,.2f} toward uninsured claim (regular business expense)"
                        )

        # Remove fully paid claims
        self.claim_liabilities = [c for c in self.claim_liabilities if c.remaining_amount > 0]

        if total_paid > 0:
            logger.info(f"Paid ${total_paid:,.2f} toward claim liabilities")

        return total_paid

    def process_insurance_claim_with_development(
        self,
        claim_amount: float,
        deductible: float = 0.0,
        insurance_limit: float = float("inf"),
        development_pattern: Optional["ClaimDevelopment"] = None,
        claim_type: str = "general_liability",
    ) -> tuple[float, float, Optional["Claim"]]:
        """Process an insurance claim with custom development pattern integration.

        This enhanced method extends basic claim processing to support custom
        claim development patterns from the claim_development module. It provides
        more sophisticated actuarial modeling while maintaining compatibility
        with the basic claim processing workflow.

        The method first processes the immediate financial impact using standard
        logic, then optionally creates a detailed Claim object with custom
        development patterns for advanced cash flow modeling.

        Args:
            claim_amount (float): Total amount of the loss/claim in dollars.
                Must be >= 0.
            deductible (float): Amount company must pay before insurance coverage
                begins. Defaults to 0.0 for full coverage.
            insurance_limit (float): Maximum amount insurance will pay per claim.
                Defaults to unlimited coverage.
            development_pattern (Optional[ClaimDevelopment]): Custom actuarial
                development pattern for the claim. If None, uses default
                ClaimLiability schedule. Enables sophisticated reserving and
                payment modeling.
            claim_type (str): Classification of claim type for actuarial analysis.
                Common types: "general_liability", "product_liability",
                "workers_compensation", "property". Defaults to "general_liability".

        Returns:
            tuple[float, float, Optional[Claim]]: Three-element tuple containing:
                - company_payment (float): Amount paid immediately by company
                - insurance_payment (float): Amount covered by insurance
                - claim_object (Optional[Claim]): Detailed claim tracking object
                  with development pattern. None if no development_pattern provided
                  or insurance_payment is 0.

        Examples:
            Basic enhanced claim processing::

                from ergodic_insurance.claim_development import (
                    ClaimDevelopment, WorkersCompensation
                )

                # Custom development pattern for workers' comp
                wc_pattern = WorkersCompensation()

                company_paid, insurance_paid, claim_obj = manufacturer.process_insurance_claim_with_development(
                    claim_amount=2_000_000,
                    deductible=250_000,
                    insurance_limit=5_000_000,
                    development_pattern=wc_pattern,
                    claim_type="workers_compensation"
                )

                if claim_obj:
                    print(f"Claim ID: {claim_obj.claim_id}")
                    print(f"Development pattern: {claim_obj.development_pattern.pattern_name}")

            Fallback to basic processing::

                # Without development pattern, behaves like basic method
                company_paid, insurance_paid, claim_obj = manufacturer.process_insurance_claim_with_development(
                    claim_amount=1_000_000,
                    deductible=100_000
                )

                assert claim_obj is None  # No enhanced tracking

        Side Effects:
            Same as :meth:`process_insurance_claim` plus:
            - Creates Claim object with unique ID if development_pattern provided
            - Enables integration with claim_development module workflows
            - Provides enhanced claim tracking for actuarial analysis

        Note:
            The claim_object provides additional functionality beyond basic
            ClaimLiability tracking, including reserving calculations,
            development factor analysis, and integration with external
            actuarial systems.

        Warning:
            This method requires the claim_development module to be available
            when development_pattern is provided. Import errors will be raised
            if the module is not accessible.

        See Also:
            :meth:`process_insurance_claim`: Basic claim processing without
            development patterns.
            :class:`~ergodic_insurance.claim_development.ClaimDevelopment`:
            Base class for development patterns.
            :class:`~ergodic_insurance.claim_development.Claim`: Enhanced
            claim tracking with reserving capabilities.
        """
        # Process the immediate financial impact
        company_payment, insurance_payment = self.process_insurance_claim(
            claim_amount, deductible, insurance_limit
        )

        # If a development pattern is provided, create a Claim object
        claim_object = None
        if development_pattern is not None and insurance_payment > 0:
            # Import here to avoid circular dependency
            from .claim_development import Claim

            claim_object = Claim(
                claim_id=f"CL_{self.current_year}_{len(self.claim_liabilities):04d}",
                accident_year=self.current_year,
                reported_year=self.current_year,
                initial_estimate=insurance_payment,
                claim_type=claim_type,
                development_pattern=development_pattern,
            )

            # The claim is already added to liabilities in process_insurance_claim
            # but we can enhance the tracking with the Claim object
            logger.info(
                f"Created claim with {development_pattern.pattern_name} development pattern"
            )

        return company_payment, insurance_payment, claim_object

    def check_solvency(self) -> bool:
        """Check if the company is solvent and update ruin status.

        Evaluates the company's financial solvency based on both equity position
        and payment capacity. A company is considered insolvent (ruined) when:
        1. Equity falls to zero or below (traditional balance sheet insolvency)
        2. Scheduled claim payments exceed sustainable cash flow capacity (payment insolvency)

        Payment insolvency occurs when the company cannot realistically service
        its claim payment obligations given its revenue-generating capacity.

        Returns:
            bool: True if company is solvent, False if insolvent.
                Once False, the company remains ruined for the simulation.

        Examples:
            Monitor solvency during simulation::

                # After processing large claim
                manufacturer.process_insurance_claim(
                    claim_amount=15_000_000,
                    deductible=5_000_000
                )

                if not manufacturer.check_solvency():
                    print(f"Company became insolvent")
                    break  # Exit simulation

            Solvency-aware simulation loop::

                for year in range(simulation_years):
                    # Process annual events
                    metrics = manufacturer.step()

                    # Check solvency (already called in step())
                    if not manufacturer.check_solvency():
                        print(f"Bankruptcy in year {year}")
                        break

        Side Effects:
            - Updates :attr:`is_ruined` to True if insolvent
            - Logs warning message when insolvency first detected
            - Once ruined, company remains ruined for simulation duration

        Note:
            This method is automatically called during :meth:`step` execution.
            Payment insolvency is detected when scheduled claim payments exceed
            80% of revenue, indicating unsustainable claim service burden.

        Warning:
            Insolvency is an absorbing state - once ruined, the company cannot
            recover within the same simulation run. Use :meth:`reset` to
            restore solvency for new simulation scenarios.

        See Also:
            :attr:`is_ruined`: Current ruin status flag.
            :attr:`equity`: Financial equity determining solvency.
            :meth:`step`: Automatically includes solvency checking.
        """
        # Traditional balance sheet insolvency
        if self.equity <= 0:
            if not self.is_ruined:  # Only log once
                self.is_ruined = True
                logger.warning(f"Company became insolvent - negative equity: ${self.equity:,.2f}")
            return False

        # Payment insolvency - check if claim payment obligations are unsustainable
        if self.claim_liabilities:
            # Calculate scheduled payments for the current year
            current_year_payments = 0.0
            for claim in self.claim_liabilities:
                years_since = self.current_year - claim.year_incurred
                scheduled_payment = claim.get_payment(years_since)
                current_year_payments += scheduled_payment

            # Check if payments are sustainable relative to revenue capacity
            if current_year_payments > 0:
                current_revenue = self.calculate_revenue()
                payment_burden_ratio = (
                    current_year_payments / current_revenue if current_revenue > 0 else float("inf")
                )

                # Company is insolvent if claim payments exceed 80% of revenue
                # This threshold represents realistic maximum debt service capacity
                if payment_burden_ratio > 0.80:
                    if not self.is_ruined:  # Only log once
                        self.is_ruined = True
                        logger.warning(
                            f"Company became insolvent - unsustainable payment burden: "
                            f"${current_year_payments:,.0f} payments vs ${current_revenue:,.0f} revenue "
                            f"({payment_burden_ratio:.1%} burden ratio)"
                        )
                    return False

        return True

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive financial metrics for analysis.

        This method computes a complete set of financial metrics including
        balance sheet items, income statement components, and financial ratios.
        It provides a standardized view of the company's financial performance
        and position for simulation analysis.

        Returns:
            Dict[str, float]: Comprehensive metrics dictionary with keys:

                Balance Sheet Metrics:
                - 'assets': Total company assets
                - 'equity': Shareholder equity
                - 'collateral': Letter of credit collateral posted
                - 'restricted_assets': Assets pledged as collateral
                - 'available_assets': Assets available for operations
                - 'net_assets': Total assets minus restricted assets
                - 'claim_liabilities': Outstanding claim payment obligations

                Income Statement Metrics:
                - 'revenue': Annual revenue
                - 'operating_income': Earnings before interest and taxes
                - 'net_income': Final profit after all expenses

                Financial Ratios:
                - 'roe': Return on equity (net_income / equity)
                - 'roa': Return on assets (net_income / assets)
                - 'asset_turnover': Revenue efficiency (revenue / assets)
                - 'base_operating_margin': Operating profit margin
                - 'collateral_to_equity': Leverage from collateral requirements
                - 'collateral_to_assets': Asset restriction from collateral

                Status Indicators:
                - 'is_solvent': Boolean indicating company solvency

        Examples:
            Analyze financial performance::

                metrics = manufacturer.calculate_metrics()

                print(f"ROE: {metrics['roe']:.1%}")
                print(f"ROA: {metrics['roa']:.1%}")
                print(f"Asset Turnover: {metrics['asset_turnover']:.2f}x")
                print(f"Solvency: {metrics['is_solvent']}")

            Track key balance sheet ratios::

                collateral_leverage = metrics['collateral_to_equity']
                asset_restriction = metrics['collateral_to_assets']

                if collateral_leverage > 2.0:
                    print("High collateral leverage")

                if asset_restriction > 0.5:
                    print("Significant asset restrictions")

            Monitor operational efficiency::

                if metrics['base_operating_margin'] < 0:
                    print("Operating losses")

                if metrics['asset_turnover'] < 0.5:
                    print("Low asset utilization")

        Note:
            Financial ratios handle division by zero gracefully by returning 0.
            All monetary amounts are in dollars. Ratios are in decimal form
            (0.08 = 8%).

        Warning:
            Metrics are calculated based on current state and may not reflect
            cash flows or timing differences in actual business operations.

        See Also:
            :meth:`step`: Updates metrics automatically during simulation.
            :attr:`metrics_history`: Historical metrics storage.
        """
        metrics = {}

        # Basic balance sheet metrics
        metrics["assets"] = self.total_assets
        metrics["collateral"] = self.collateral
        metrics["restricted_assets"] = self.restricted_assets
        metrics["available_assets"] = self.available_assets
        metrics["equity"] = self.equity
        metrics["net_assets"] = self.net_assets
        metrics["claim_liabilities"] = self.total_claim_liabilities
        metrics["is_solvent"] = not self.is_ruined

        # Enhanced balance sheet components for GAAP compliance
        metrics["cash"] = self.cash
        metrics["accounts_receivable"] = self.accounts_receivable
        metrics["inventory"] = self.inventory
        metrics["prepaid_insurance"] = self.prepaid_insurance
        metrics["accounts_payable"] = self.accounts_payable

        # Get detailed accrual breakdown from AccrualManager
        accrual_items = self.accrual_manager.get_balance_sheet_items()

        # Update total accrued expenses from accrual manager
        total_accrued_expenses = accrual_items.get("accrued_expenses", 0)
        # Also include any legacy accrued_expenses not in the manager
        metrics["accrued_expenses"] = max(self.accrued_expenses, total_accrued_expenses)

        metrics["gross_ppe"] = self.gross_ppe
        metrics["accumulated_depreciation"] = self.accumulated_depreciation
        metrics["net_ppe"] = self.net_ppe

        # Add detailed accrual breakdown
        metrics["accrued_wages"] = accrual_items.get("accrued_wages", 0)
        metrics["accrued_taxes"] = accrual_items.get("accrued_taxes", 0)
        metrics["accrued_interest"] = accrual_items.get("accrued_interest", 0)
        metrics["accrued_revenues"] = accrual_items.get("accrued_revenues", 0)

        # Calculate operating metrics for current state
        revenue = self.calculate_revenue()
        # Calculate depreciation for metrics (annual basis)
        annual_depreciation = self.gross_ppe / 10 if self.gross_ppe > 0 else 0.0
        operating_income = self.calculate_operating_income(revenue, annual_depreciation)
        collateral_costs = self.calculate_collateral_costs()
        # Insurance costs already deducted in calculate_operating_income
        # Don't accrue taxes here since this is just for metrics reporting
        net_income = self.calculate_net_income(
            operating_income,
            collateral_costs,
            0,  # Insurance premiums already deducted in operating_income
            0,  # Insurance losses already deducted in operating_income
            use_accrual=False,  # Don't accrue in metrics calculation
        )

        metrics["revenue"] = revenue
        metrics["operating_income"] = operating_income
        metrics["net_income"] = net_income

        # Track insurance expenses for transparency
        metrics["insurance_premiums"] = self.period_insurance_premiums
        metrics["insurance_losses"] = self.period_insurance_losses
        metrics["total_insurance_costs"] = (
            self.period_insurance_premiums + self.period_insurance_losses
        )

        # Calculate and track dividends paid
        dividends_paid = 0.0
        if net_income > 0:
            dividends_paid = net_income * (1 - self.retention_ratio)
        metrics["dividends_paid"] = dividends_paid

        # Track depreciation expense for cash flow statement
        metrics["depreciation_expense"] = annual_depreciation

        # Financial ratios - ROE now includes all expenses
        metrics["asset_turnover"] = revenue / self.total_assets if self.total_assets > 0 else 0

        # Report both base and actual operating margins for transparency
        metrics["base_operating_margin"] = self.base_operating_margin
        metrics["actual_operating_margin"] = operating_income / revenue if revenue > 0 else 0
        metrics["insurance_impact_on_margin"] = (
            metrics["base_operating_margin"] - metrics["actual_operating_margin"]
        )

        # Define minimum equity threshold for meaningful ROE calculation
        # When equity is <= 100, ROE becomes meaningless (approaches infinity)
        MIN_EQUITY_THRESHOLD = 100

        metrics["roe"] = (
            net_income / self.equity if self.equity > MIN_EQUITY_THRESHOLD else 0
        )  # Return 0 for very small or negative equity to avoid extreme values
        metrics["roa"] = net_income / self.total_assets if self.total_assets > 0 else 0

        # Leverage metrics (collateral-based instead of debt)
        metrics["collateral_to_equity"] = self.collateral / self.equity if self.equity > 0 else 0
        metrics["collateral_to_assets"] = (
            self.collateral / self.total_assets if self.total_assets > 0 else 0
        )

        return metrics

    def _handle_insolvent_step(self, time_resolution: str) -> Dict[str, float]:
        """Handle a simulation step when the company is already insolvent.

        Args:
            time_resolution: "annual" or "monthly" for simulation step.

        Returns:
            Dictionary of metrics for this time step.
        """
        logger.warning("Company is already insolvent, skipping step")
        metrics = self.calculate_metrics()
        metrics["year"] = self.current_year
        metrics["month"] = float(self.current_month) if time_resolution == "monthly" else 0.0
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

    def process_accrued_payments(self, time_resolution: str = "annual") -> float:
        """Process due accrual payments for the current period.

        Checks for accrual payments due in the current period and processes
        them, reducing cash and clearing the accruals. This method supports
        quarterly tax payments and other scheduled accrual payments.

        Args:
            time_resolution: "annual" or "monthly" for determining current period

        Returns:
            Total cash payments made for accruals in this period
        """
        # Determine current period for accrual manager
        # Periods are always tracked in months for consistency
        if time_resolution == "monthly":
            period = self.current_year * 12 + self.current_month
        else:
            # Annual resolution - convert year to months (assuming end of year)
            period = self.current_year * 12

        # Sync accrual manager period
        self.accrual_manager.current_period = period

        # Get all payments due this period
        payments_due = self.accrual_manager.get_payments_due(period)

        total_paid = 0.0
        for accrual_type, amount_due in payments_due.items():
            # Process the payment
            self.accrual_manager.process_payment(accrual_type, amount_due, period)

            # Reduce cash for payment
            self.cash -= amount_due
            total_paid += amount_due

            # Update accrued_expenses balance
            self.accrued_expenses = max(0, self.accrued_expenses - amount_due)

            logger.debug(f"Paid accrued {accrual_type.value}: ${amount_due:,.2f}")

        if total_paid > 0:
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
        # Update balance sheet accrued expenses
        self.accrued_expenses += amount

    def record_claim_accrual(
        self, claim_amount: float, development_pattern: Optional[List[float]] = None
    ) -> None:
        """Record insurance claim with multi-year payment schedule.

        Args:
            claim_amount: Total claim amount to be paid
            development_pattern: Optional custom payment pattern over years
        """
        payment_schedule = self.accrual_manager.get_claim_payment_schedule(
            claim_amount, development_pattern
        )

        # Create separate accrual items for each payment to preserve amounts
        for period, payment_amount in payment_schedule:
            self.accrual_manager.record_expense_accrual(
                item_type=AccrualType.INSURANCE_CLAIMS,
                amount=payment_amount,
                payment_schedule=PaymentSchedule.CUSTOM,
                payment_dates=[period],
                description=f"Claim from year {self.current_year}",
            )

    def _apply_growth(
        self, growth_rate: float, time_resolution: str, apply_stochastic: bool
    ) -> None:
        """Apply revenue growth by adjusting asset turnover ratio.

        Args:
            growth_rate: Revenue growth rate for the period.
            time_resolution: "annual" or "monthly" for simulation step.
            apply_stochastic: Whether to apply stochastic shocks.
        """
        if growth_rate == 0 or not (time_resolution == "annual" or self.current_month == 11):
            return

        base_growth = 1 + growth_rate

        # Add stochastic component to growth if enabled
        if apply_stochastic and self.stochastic_process is not None:
            # Use a separate shock for growth rate
            growth_shock = self.stochastic_process.generate_shock(1.0)
            # Combine deterministic and stochastic growth
            total_growth = base_growth * growth_shock
            self.asset_turnover_ratio *= total_growth
            logger.debug(
                f"Applied growth: {total_growth:.4f} (base={base_growth:.4f}, shock={growth_shock:.4f})"
            )
        else:
            self.asset_turnover_ratio *= base_growth

    def step(
        self,
        working_capital_pct: float = 0.2,
        letter_of_credit_rate: float = 0.015,
        growth_rate: float = 0.0,
        time_resolution: str = "annual",
        apply_stochastic: bool = False,
    ) -> Dict[str, float]:
        """Execute one time step of the financial model simulation.

        This is the main simulation method that advances the manufacturer's
        financial state by one time period. It processes all business operations
        including revenue generation, expense payment, claim liability payments,
        growth application, and solvency checking.

        The method supports both annual and monthly time resolution for flexible
        modeling. Monthly resolution provides more granular cash flow tracking
        but requires careful scaling of annual parameters.

        Args:
            working_capital_pct (float): Working capital as percentage of sales.
                Represents inventory and receivables minus payables. Typical
                manufacturing values: 0.15-0.25. Defaults to 0.2 (20%).
            letter_of_credit_rate (float): Annual interest rate for letter of
                credit collateral. Market rates typically 0.01-0.02 (1-2%).
                Defaults to 0.015 (1.5%).
            growth_rate (float): Revenue growth rate for the period. Applied
                annually or when current_month=11 for monthly resolution.
                Defaults to 0.0 for no growth.
            time_resolution (str): Time step resolution. Must be "annual" or
                "monthly". Annual is standard for most analyses. Monthly provides
                granular cash flow modeling. Defaults to "annual".
            apply_stochastic (bool): Whether to apply stochastic shocks to
                revenue and growth. Requires stochastic_process to be initialized.
                Defaults to False for deterministic modeling.

        Returns:
            Dict[str, float]: Comprehensive financial metrics dictionary containing:
                - Balance sheet items: assets, equity, collateral, etc.
                - Income statement: revenue, operating_income, net_income
                - Financial ratios: roe, roa, base_operating_margin, etc.
                - Solvency indicators: is_solvent
                - Time tracking: year, month (if monthly resolution)

        Examples:
            Basic annual simulation step::

                # Standard annual step with 20% working capital
                metrics = manufacturer.step(
                    working_capital_pct=0.2,
                    letter_of_credit_rate=0.015
                )

                print(f"ROE: {metrics['roe']:.1%}")
                print(f"Assets: ${metrics['assets']:,.2f}")

            Monthly simulation with growth::

                # Monthly steps with 5% annual growth
                for month in range(12):
                    metrics = manufacturer.step(
                        working_capital_pct=0.18,
                        growth_rate=0.05,
                        time_resolution="monthly"
                    )

                    if month == 11:  # Growth applied in December
                        print(f"Growth applied: {metrics['asset_turnover']}")

            Stochastic simulation::

                # With revenue volatility
                metrics = manufacturer.step(
                    working_capital_pct=0.2,
                    apply_stochastic=True
                )

                # Revenue will vary based on stochastic process

        Side Effects:
            - Updates current_year and/or current_month
            - Modifies balance sheet (assets, equity, collateral)
            - Processes scheduled claim payments
            - May trigger insolvency if losses exceed equity
            - Appends metrics to metrics_history
            - Applies growth by modifying asset_turnover_ratio

        Raises:
            ValueError: If time_resolution not "annual" or "monthly".
            ValueError: If working_capital_pct < 0 or >= 1.
            RuntimeError: If apply_stochastic=True but no stochastic process.

        Note:
            If the company is already insolvent, the method returns immediately
            with minimal processing. Monthly resolution scales revenue and
            operating income by 1/12 but keeps collateral costs as calculated
            monthly rates.

        Warning:
            Growth is applied cumulatively to asset_turnover_ratio. In long
            simulations with high growth rates, monitor for unrealistic values.

        See Also:
            :meth:`calculate_revenue`: Core revenue calculation logic.
            :meth:`pay_claim_liabilities`: Claim payment processing.
            :meth:`check_solvency`: Solvency evaluation.
            :meth:`calculate_metrics`: Metrics calculation details.
        """
        # Check if already ruined
        if self.is_ruined:
            return self._handle_insolvent_step(time_resolution)

        # Process accrual payments due this period (e.g., quarterly taxes)
        self.process_accrued_payments(time_resolution)

        # Pay scheduled claim liabilities first (annual payments)
        if time_resolution == "annual" or self.current_month == 0:
            self.pay_claim_liabilities()

        # Store initial revenue for working capital calculation in monthly mode
        # This must happen BEFORE any balance sheet changes
        if time_resolution == "monthly" and self.current_month == 0:
            # Calculate the annual revenue with working capital adjustment
            # This ensures consistency with annual mode
            self._annual_revenue_for_wc = self.calculate_revenue(
                working_capital_pct, apply_stochastic
            )

        # Calculate financial performance
        revenue = self.calculate_revenue(working_capital_pct, apply_stochastic)

        # Calculate depreciation expense for the period
        if time_resolution == "annual":
            depreciation_expense = self.record_depreciation(useful_life_years=10)
        elif time_resolution == "monthly":
            # For monthly, record 1/12 of annual depreciation
            depreciation_expense = self.record_depreciation(useful_life_years=10 * 12)
        else:
            depreciation_expense = 0.0

        # Calculate operating income including depreciation
        operating_income = self.calculate_operating_income(revenue, depreciation_expense)

        # Calculate collateral costs (monthly if specified)
        if time_resolution == "monthly":
            # Monthly collateral costs
            collateral_costs = self.calculate_collateral_costs(letter_of_credit_rate, "monthly")
            # Scale other flows to monthly
            revenue = revenue / 12
            operating_income = operating_income / 12
        else:
            # Annual collateral costs (sum of 12 monthly payments)
            collateral_costs = self.calculate_collateral_costs(letter_of_credit_rate, "annual")

        # Calculate net income (insurance costs already included in operating_income)
        net_income = self.calculate_net_income(
            operating_income,
            collateral_costs,
            0,  # Insurance premiums already deducted in operating_income
            0,  # Insurance losses already deducted in operating_income
            use_accrual=True,
            time_resolution=time_resolution,
        )

        # Calculate working capital components
        # Use consistent revenue measure to avoid compounding effects
        if time_resolution == "annual":
            # Annual mode: use the annual revenue
            self.calculate_working_capital_components(revenue)
        elif time_resolution == "monthly":
            # Monthly mode: use the stored annual revenue from year start
            # This was calculated before any balance sheet changes
            if hasattr(self, "_annual_revenue_for_wc"):
                self.calculate_working_capital_components(self._annual_revenue_for_wc)
            else:
                # Fallback: use current assets (should not happen normally)
                annual_revenue = self.total_assets * self.asset_turnover_ratio
                self.calculate_working_capital_components(annual_revenue)

        # Update balance sheet with retained earnings AFTER working capital calculation
        # This prevents working capital from compounding with asset growth
        self.update_balance_sheet(net_income, growth_rate)

        # Amortize prepaid insurance if applicable
        if time_resolution == "monthly":
            self.amortize_prepaid_insurance(months=1)

        # Cash is already properly updated through:
        # 1. Retained earnings in update_balance_sheet()
        # 2. Claim payments in pay_claim_liabilities()
        # 3. Other specific cash transactions
        # No need to recalculate as residual

        # Apply revenue growth by adjusting asset turnover ratio
        self._apply_growth(growth_rate, time_resolution, apply_stochastic)

        # Check solvency
        self.check_solvency()

        # Calculate and store metrics
        metrics = self.calculate_metrics()
        metrics["year"] = self.current_year
        metrics["month"] = float(self.current_month) if time_resolution == "monthly" else 0.0
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
        runs from the same starting point. It is essential for Monte Carlo
        analysis and parameter sensitivity studies.

        Side Effects:
            - Resets assets to config.initial_assets
            - Clears all collateral and restricted assets
            - Restores equity to initial assets (no debt assumption)
            - Resets asset_turnover_ratio to config value
            - Clears all outstanding claim liabilities
            - Resets time tracking to year 0, month 0
            - Restores solvency status to healthy
            - Clears metrics_history
            - Resets stochastic process (if present) to initial state

        Examples:
            Run multiple simulations::

                results = []
                for seed in range(100):
                    manufacturer.reset()

                    # Set different random seed for each run
                    if manufacturer.stochastic_process:
                        manufacturer.stochastic_process.seed = seed

                    # Run simulation
                    for year in range(20):
                        metrics = manufacturer.step(apply_stochastic=True)

                    results.append(manufacturer.metrics_history)

            Parameter sensitivity analysis::

                base_config = manufacturer.config
                margins = [0.06, 0.08, 0.10]
                results = {}

                for margin in margins:
                    manufacturer.reset()
                    manufacturer.base_operating_margin = margin

                    # Run simulation with different margin
                    final_metrics = manufacturer.step()
                    results[margin] = final_metrics['roe']

        Note:
            The reset method creates a clean slate but preserves the original
            configuration. Any runtime modifications to parameters (like
            base_operating_margin) will need to be reapplied after reset.

        See Also:
            :meth:`copy`: Create independent manufacturer instances.
            :attr:`config`: Original configuration parameters.
        """
        # Reset collateral and restricted assets
        self.collateral = 0.0
        self.restricted_assets = 0.0

        # Reset operating parameters
        self.asset_turnover_ratio = self.config.asset_turnover_ratio
        self.claim_liabilities = []
        self.current_year = 0
        self.current_month = 0
        self.is_ruined = False
        self.metrics_history = []

        # Reset enhanced balance sheet components
        # PP&E allocation depends on operating margin
        if self.config.base_operating_margin < 0.10:
            ppe_ratio = 0.3  # Low margin businesses need more working capital, less PP&E
        elif self.config.base_operating_margin < 0.15:
            ppe_ratio = 0.5  # Medium margin can support moderate PP&E
        else:
            ppe_ratio = 0.7  # High margin businesses can support more PP&E

        self.gross_ppe = self.config.initial_assets * ppe_ratio
        self.accumulated_depreciation = 0.0
        self.cash = self.config.initial_assets * (1 - ppe_ratio)
        self.accounts_receivable = 0.0
        self.inventory = 0.0
        self.prepaid_insurance = 0.0
        self.accounts_payable = 0.0
        self.accrued_expenses = 0.0
        self._original_prepaid_premium = 0.0

        # Reset period insurance cost tracking
        self.period_insurance_premiums = 0.0
        self.period_insurance_losses = 0.0

        # Reset initial values (for exposure bases)
        self._initial_assets = self.config.initial_assets
        self._initial_equity = self.config.initial_assets

        # Reset accrual manager
        self.accrual_manager = AccrualManager()

        # Reset stochastic process if present
        if self.stochastic_process is not None:
            self.stochastic_process.reset()

        logger.info("Manufacturer reset to initial state")

    def copy(self) -> "WidgetManufacturer":
        """Create a deep copy of the manufacturer for parallel simulations.

        This method creates a completely independent manufacturer instance
        with the same configuration but fresh initial state. It is designed
        for parallel simulation scenarios where multiple independent paths
        need to be explored simultaneously.

        Returns:
            WidgetManufacturer: A new manufacturer instance with:
                - Same configuration parameters
                - Deep copy of stochastic process (if present)
                - Reset to initial financial state
                - Independent random number generation

        Examples:
            Parallel Monte Carlo simulation::

                base_manufacturer = WidgetManufacturer(config, stochastic_process)

                # Create multiple independent copies
                manufacturers = [base_manufacturer.copy() for _ in range(1000)]

                # Run parallel simulations
                results = []
                for i, mfg in enumerate(manufacturers):
                    # Each has independent random sequence
                    if mfg.stochastic_process:
                        mfg.stochastic_process.seed = i

                    # Run independent simulation
                    for year in range(10):
                        mfg.step(apply_stochastic=True)

                    results.append(mfg.equity)

            Scenario analysis::

                base_case = manufacturer.copy()
                stress_case = manufacturer.copy()

                # Apply different parameters to each
                stress_case.base_operating_margin = 0.04  # Stress scenario

                # Run independent simulations
                base_result = base_case.step()
                stress_result = stress_case.step()

        Note:
            The copy starts in initial state regardless of the original
            manufacturer's current state. This ensures consistent starting
            points for comparative analysis.

        Warning:
            Stochastic processes are deep copied, which means each copy will
            have independent random number generation. Set different seeds
            if you want truly independent random sequences.

        See Also:
            :meth:`reset`: Reset existing instance instead of copying.
            :class:`~ergodic_insurance.stochastic_processes.StochasticProcess`:
            For stochastic process copying behavior.
        """
        import copy

        # Create a new instance with the same config
        new_manufacturer = WidgetManufacturer(
            config=self.config,
            stochastic_process=copy.deepcopy(self.stochastic_process)
            if self.stochastic_process
            else None,
        )

        # The new manufacturer starts fresh at initial state
        # No need to copy current state since we want independent simulations

        logger.debug("Created copy of manufacturer")
        return new_manufacturer

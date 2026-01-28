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

        from ergodic_insurance.config import ManufacturerConfig
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
    - :mod:`~ergodic_insurance.config`: Parameter configuration and validation
    - :mod:`~ergodic_insurance.stochastic_processes`: Uncertainty modeling
    - :mod:`~ergodic_insurance.loss_distributions`: Automated loss generation
    - :mod:`~ergodic_insurance.insurance_program`: Multi-layer insurance structures
    - :mod:`~ergodic_insurance.simulation`: High-level simulation orchestration

See Also:
    :class:`~ergodic_insurance.config.ManufacturerConfig`: Configuration parameters
    :mod:`~ergodic_insurance.stochastic_processes`: Stochastic modeling options
    :mod:`~ergodic_insurance.claim_development`: Advanced actuarial patterns
    :mod:`~ergodic_insurance.simulation`: Simulation framework integration

Note:
    This module forms the financial foundation for ergodic insurance optimization.
    All business logic assumes a debt-free balance sheet where equity equals
    net assets, simplifying the model while maintaining realistic cash flows.
"""

from dataclasses import dataclass, field
from decimal import Decimal
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    # Try absolute import first (for installed package)
    from ergodic_insurance.accrual_manager import AccrualManager, AccrualType, PaymentSchedule
    from ergodic_insurance.claim_development import ClaimDevelopment
    from ergodic_insurance.config import ManufacturerConfig
    from ergodic_insurance.decimal_utils import ONE, ZERO, quantize_currency, to_decimal
    from ergodic_insurance.insurance_accounting import InsuranceAccounting
    from ergodic_insurance.ledger import AccountName, Ledger, TransactionType
    from ergodic_insurance.stochastic_processes import StochasticProcess
except ImportError:
    try:
        # Try relative import (for package context)
        from .accrual_manager import AccrualManager, AccrualType, PaymentSchedule
        from .claim_development import ClaimDevelopment
        from .config import ManufacturerConfig
        from .decimal_utils import ONE, ZERO, quantize_currency, to_decimal
        from .insurance_accounting import InsuranceAccounting
        from .ledger import AccountName, Ledger, TransactionType
        from .stochastic_processes import StochasticProcess
    except ImportError:
        # Fall back to direct import (for notebooks/scripts)
        from accrual_manager import (  # type: ignore[no-redef]
            AccrualManager,
            AccrualType,
            PaymentSchedule,
        )
        from claim_development import ClaimDevelopment  # type: ignore[no-redef]
        from config import ManufacturerConfig  # type: ignore[no-redef]
        from decimal_utils import ONE, ZERO, quantize_currency, to_decimal  # type: ignore[no-redef]
        from insurance_accounting import InsuranceAccounting  # type: ignore[no-redef]
        from ledger import AccountName, Ledger, TransactionType  # type: ignore[no-redef]
        from stochastic_processes import StochasticProcess  # type: ignore[no-redef]

# Optional import for claim development integration
if TYPE_CHECKING:
    from .claim_development import Claim, ClaimDevelopment

logger = logging.getLogger(__name__)


@dataclass
class ClaimLiability:
    """Represents an outstanding insurance claim liability with payment schedule.

    This class tracks insurance claims that require multi-year payment
    schedules and manages the collateral required to support them. It uses
    the Strategy Pattern with ClaimDevelopment to define payment timing.

    The default development strategy follows a typical long-tail liability pattern
    where claims are paid over 10 years, with higher percentages in early years
    and gradually decreasing payments over time. This pattern is commonly observed
    in general liability and workers' compensation claims.

    Attributes:
        original_amount (Decimal): The original claim amount at inception.
        remaining_amount (Decimal): The unpaid balance of the claim.
        year_incurred (int): The year when the claim was first incurred.
        is_insured (bool): Whether this claim involves insurance coverage.
            True for insured claims (company deductible), False for uninsured claims.
        development_strategy (ClaimDevelopment): Strategy for payment development.
            Default uses ClaimDevelopment.create_long_tail_10yr() which follows:
            - Years 1-3: Front-loaded payments (10%, 20%, 20%)
            - Years 4-6: Moderate payments (15%, 10%, 8%)
            - Years 7-10: Tail payments (7%, 5%, 3%, 2%)

    Examples:
        Create and track a claim liability::

            # Create a $1M claim liability with default 10-year pattern
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

        Custom development pattern::

            # Create claim with immediate payment pattern
            immediate_claim = ClaimLiability(
                original_amount=500_000,
                remaining_amount=500_000,
                year_incurred=2023,
                development_strategy=ClaimDevelopment.create_immediate()
            )

            # Create claim with custom pattern
            custom_pattern = ClaimDevelopment(
                pattern_name="CUSTOM",
                development_factors=[0.5, 0.3, 0.2]  # 50%, 30%, 20%
            )
            custom_claim = ClaimLiability(
                original_amount=500_000,
                remaining_amount=500_000,
                year_incurred=2023,
                development_strategy=custom_pattern
            )

    Note:
        The development factors should sum to 1.0 for full claim payout.
        Payments beyond the schedule length return 0.0 (unless tail_factor is set).

    See Also:
        :class:`~ergodic_insurance.claim_development.ClaimDevelopment`: For more
        sophisticated claim development patterns including factory methods.
        :class:`~ergodic_insurance.claim_development.Claim`: For comprehensive
        claim tracking with reserving.
    """

    original_amount: Decimal
    remaining_amount: Decimal
    year_incurred: int
    is_insured: bool = True  # Default to insured for backward compatibility
    development_strategy: ClaimDevelopment = field(
        default_factory=ClaimDevelopment.create_long_tail_10yr
    )

    @property
    def payment_schedule(self) -> List[float]:
        """Return development factors for backward compatibility.

        This property provides access to the development factors from the
        underlying ClaimDevelopment strategy, maintaining API compatibility
        with code that expects a payment_schedule list.

        Returns:
            List[float]: The development factors from the strategy.
        """
        return self.development_strategy.development_factors

    def get_payment(self, years_since_incurred: int) -> Decimal:
        """Calculate payment due for a given year after claim incurred.

        This method delegates to the ClaimDevelopment strategy to calculate
        the scheduled payment amount. It handles boundary conditions gracefully,
        returning zero for negative years or years beyond the development pattern.

        Args:
            years_since_incurred (int): Number of years since the claim was first
                incurred. Must be >= 0. Year 0 represents the year of incurrence.

        Returns:
            Decimal: Payment amount due for this year in absolute dollars. Returns
                ZERO if years_since_incurred is negative or beyond the schedule.

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
            The method delegates to development_strategy.calculate_payments() and
            does not check against remaining balance.
        """
        if years_since_incurred < 0:
            return ZERO
        # Delegate to ClaimDevelopment strategy
        # calculate_payments uses (claim_amount, accident_year, payment_year)
        payment_year = self.year_incurred + years_since_incurred
        payment = self.development_strategy.calculate_payments(
            claim_amount=float(self.original_amount),
            accident_year=self.year_incurred,
            payment_year=payment_year,
        )
        return to_decimal(payment)

    def make_payment(self, amount: Union[Decimal, float]) -> Decimal:
        """Make a payment against the liability and update remaining balance.

        This method processes a payment against the claim liability, reducing
        the remaining amount. If the requested payment exceeds the remaining
        balance, only the remaining amount is paid.

        Args:
            amount: Requested payment amount in dollars. Should be >= 0.
                Negative amounts are treated as zero payment.

        Returns:
            Decimal: Actual payment made in dollars. May be less than requested
                if insufficient liability remains. Returns ZERO if amount <= 0
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
        amount_decimal = to_decimal(amount)
        payment = min(amount_decimal, self.remaining_amount)
        self.remaining_amount -= payment
        return payment

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ClaimLiability":
        """Create a deep copy of this claim liability.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this ClaimLiability
        """
        import copy

        # Create a new ClaimDevelopment with copied development_factors
        copied_strategy = ClaimDevelopment(
            pattern_name=self.development_strategy.pattern_name,
            development_factors=copy.deepcopy(self.development_strategy.development_factors, memo),
            tail_factor=self.development_strategy.tail_factor,
        )

        return ClaimLiability(
            original_amount=copy.deepcopy(self.original_amount, memo),
            remaining_amount=copy.deepcopy(self.remaining_amount, memo),
            year_incurred=self.year_incurred,
            is_insured=self.is_insured,
            development_strategy=copied_strategy,
        )


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
                # Generate losses for this year
                losses, _ = loss_generator.generate_losses(duration=1, revenue=revenue)

                # Process losses through insurance
                for loss in losses:
                    manufacturer.process_insurance_claim(
                        loss.amount, deductible, limit
                    )

                # Run annual business operations
                metrics = manufacturer.step(
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

        # Initialize the event-sourcing ledger FIRST - it's the single source of truth
        # for all balance sheet accounts (Issue #275: Cash Flow Logic Divergence)
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

        # Claim tracking - initialize early for property calculations
        self.claim_liabilities: List[ClaimLiability] = []
        self.current_year = 0
        self.current_month = 0  # Track months for monthly LoC payments

        # Insurance cost tracking for tax purposes
        self.period_insurance_premiums: Decimal = ZERO  # Premiums paid this period
        self.period_insurance_losses: Decimal = ZERO  # Losses paid this period (deductibles)

        # Track actual dividends paid (for cash flow statement accuracy)
        # This tracks dividends that were actually paid considering cash constraints
        self._last_dividends_paid: Decimal = ZERO

        # Solvency tracking
        self.is_ruined = False

        # Metrics tracking
        self.metrics_history: List[Dict[str, Union[Decimal, float, int, bool]]] = []

        # Store initial values for base comparisons (for exposure bases)
        self._initial_assets: Decimal = to_decimal(config.initial_assets)
        self._initial_equity: Decimal = to_decimal(config.initial_assets)

        # Compute initial balance sheet values (these will be recorded to ledger)
        # Enhanced balance sheet components for GAAP compliance
        # Fixed Assets (allocate first to determine cash)
        # Use PPE ratio from config (defaults based on operating margin if not specified)
        # Type ignore: ppe_ratio is guaranteed non-None after model_validator
        initial_gross_ppe: Decimal = to_decimal(config.initial_assets * config.ppe_ratio)  # type: ignore
        initial_accumulated_depreciation: Decimal = ZERO  # Will accumulate over time

        # Current Assets - initialize working capital to steady state
        # Calculate initial revenue based on asset turnover
        initial_revenue = to_decimal(config.initial_assets * config.asset_turnover_ratio)
        # Calculate COGS for inventory
        initial_cogs = initial_revenue * to_decimal(1 - config.base_operating_margin)

        # Initialize AR and Inventory to steady state levels based on industry-standard ratios
        # DSO (Days Sales Outstanding) = 45, DIO (Days Inventory Outstanding) = 60
        # These match the defaults in calculate_working_capital_components()
        # This prevents Year 1 "warm-up" distortion where working capital builds from zero
        initial_accounts_receivable: Decimal = initial_revenue * to_decimal(
            45 / 365
        )  # Based on DSO
        initial_inventory: Decimal = initial_cogs * to_decimal(60 / 365)  # Based on DIO
        initial_prepaid_insurance: Decimal = ZERO  # Annual premiums paid in advance

        # Adjust cash to fund the working capital assets (AR + Inventory)
        # This maintains total_assets = initial_assets while establishing steady-state working capital
        working_capital_assets = initial_accounts_receivable + initial_inventory
        initial_cash: Decimal = to_decimal(config.initial_assets * (1 - config.ppe_ratio)) - working_capital_assets  # type: ignore

        # Current Liabilities - AP will build up on first step based on actual COGS
        # Starting with zero AP maintains total_assets = initial_assets at initialization
        initial_accounts_payable: Decimal = ZERO  # Will be calculated on first step
        # Note: Accrued expenses are tracked solely via AccrualManager (see issue #238)

        # Initial collateral and restricted assets are zero
        initial_collateral: Decimal = ZERO
        initial_restricted_assets: Decimal = ZERO

        # Record all initial balances to ledger (single source of truth)
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
        Uses equity (retained_earnings) as the balancing entry for all positions.

        The ledger is the single source of truth for all balance sheet accounts.
        After this method is called, all balance sheet properties read from the ledger.

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
        # Record initial cash position
        if cash > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=cash,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial cash position",
            )

        # Record initial accounts receivable
        if accounts_receivable > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=accounts_receivable,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial accounts receivable",
            )

        # Record initial inventory
        if inventory > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.INVENTORY,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=inventory,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial inventory",
            )

        # Record initial prepaid insurance (if any)
        if prepaid_insurance > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.PREPAID_INSURANCE,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=prepaid_insurance,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial prepaid insurance",
            )

        # Record initial gross PP&E
        if gross_ppe > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.GROSS_PPE,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=gross_ppe,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial gross PP&E",
            )

        # Record initial accumulated depreciation (contra-asset)
        if accumulated_depreciation > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.ACCUMULATED_DEPRECIATION,
                amount=accumulated_depreciation,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial accumulated depreciation",
            )

        # Record initial accounts payable (if any)
        if accounts_payable > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.ACCOUNTS_PAYABLE,
                amount=accounts_payable,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial accounts payable",
            )

        # Record initial collateral (if any)
        if collateral > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.COLLATERAL,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=collateral,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial collateral",
            )

        # Record initial restricted assets (if any)
        if restricted_assets > ZERO:
            self.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RESTRICTED_CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=restricted_assets,
                transaction_type=TransactionType.EQUITY_ISSUANCE,
                description="Initial restricted assets",
            )

    # ========================================================================
    # Balance Sheet Properties - Ledger is Single Source of Truth (Issue #275)
    # ========================================================================
    # These properties read from the ledger, ensuring that Direct and Indirect
    # cash flow methods produce consistent results. All modifications must go
    # through ledger transactions, not direct attribute assignment.

    @property
    def cash(self) -> Decimal:
        """Cash balance derived from ledger (single source of truth).

        Returns:
            Current cash balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.CASH)

    @property
    def accounts_receivable(self) -> Decimal:
        """Accounts receivable balance derived from ledger (single source of truth).

        Returns:
            Current accounts receivable balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.ACCOUNTS_RECEIVABLE)

    @property
    def inventory(self) -> Decimal:
        """Inventory balance derived from ledger (single source of truth).

        Returns:
            Current inventory balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.INVENTORY)

    @property
    def prepaid_insurance(self) -> Decimal:
        """Prepaid insurance balance derived from ledger (single source of truth).

        Returns:
            Current prepaid insurance balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.PREPAID_INSURANCE)

    @property
    def gross_ppe(self) -> Decimal:
        """Gross PP&E balance derived from ledger (single source of truth).

        Returns:
            Current gross property, plant & equipment balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.GROSS_PPE)

    @property
    def accumulated_depreciation(self) -> Decimal:
        """Accumulated depreciation balance derived from ledger (single source of truth).

        Accumulated depreciation is a contra-asset account stored with credit-normal
        balance (negative in the ledger). This property returns the absolute value
        for intuitive usage - a positive value representing total depreciation.

        Returns:
            Current accumulated depreciation balance as a positive Decimal.
        """
        return abs(self.ledger.get_balance(AccountName.ACCUMULATED_DEPRECIATION))

    @property
    def restricted_assets(self) -> Decimal:
        """Restricted assets balance derived from ledger (single source of truth).

        Returns:
            Current restricted assets (restricted cash) balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.RESTRICTED_CASH)

    @property
    def accounts_payable(self) -> Decimal:
        """Accounts payable balance derived from ledger (single source of truth).

        Returns:
            Current accounts payable balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.ACCOUNTS_PAYABLE)

    @property
    def collateral(self) -> Decimal:
        """Letter of credit collateral balance derived from ledger (single source of truth).

        Returns:
            Current collateral balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.COLLATERAL)

    # ========================================================================
    # Ledger Helper Methods for Balance Sheet Modifications (Issue #275)
    # ========================================================================
    # These methods ensure all balance sheet changes go through the ledger,
    # maintaining consistency between Direct and Indirect cash flow methods.

    def _record_cash_adjustment(
        self,
        amount: Decimal,
        description: str,
        transaction_type: TransactionType = TransactionType.ADJUSTMENT,
    ) -> None:
        """Record a cash adjustment through the ledger.

        Used for insolvency adjustments, limited liability enforcement, etc.

        Args:
            amount: Positive to increase cash, negative to decrease
            description: Description of the adjustment
            transaction_type: Type of transaction (default: ADJUSTMENT)
        """
        if amount > ZERO:
            # Increase cash - debit cash, credit retained earnings
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=amount,
                transaction_type=transaction_type,
                description=description,
            )
        elif amount < ZERO:
            # Decrease cash - debit retained earnings, credit cash
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=-amount,
                transaction_type=transaction_type,
                description=description,
            )

    def _record_asset_transfer(
        self,
        from_account: AccountName,
        to_account: AccountName,
        amount: Decimal,
        description: str,
    ) -> None:
        """Record a transfer between asset accounts through the ledger.

        Used for moving cash to restricted assets, etc.

        Args:
            from_account: Source account
            to_account: Destination account
            amount: Amount to transfer (must be positive)
            description: Description of the transfer
        """
        if amount <= ZERO:
            return
        self.ledger.record_double_entry(
            date=self.current_year,
            debit_account=to_account,
            credit_account=from_account,
            amount=amount,
            transaction_type=TransactionType.TRANSFER,
            description=description,
        )

    def _record_proportional_revaluation(
        self,
        target_total: Decimal,
        description: str = "Proportional asset revaluation",
    ) -> None:
        """Record proportional revaluation of all assets through the ledger.

        Adjusts all asset accounts proportionally to reach a target total.
        Used by the total_assets setter.

        Args:
            target_total: Target total asset value
            description: Description of the revaluation
        """
        current_total = self.total_assets
        if current_total <= ZERO or target_total <= ZERO:
            # Handle edge cases - write off all assets
            self._write_off_all_assets(description)
            if target_total > ZERO:
                # Record new cash position
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.CASH,
                    credit_account=AccountName.RETAINED_EARNINGS,
                    amount=target_total,
                    transaction_type=TransactionType.REVALUATION,
                    description=description,
                )
            return

        ratio = target_total / current_total
        if ratio == ONE:
            return

        # Calculate adjustments for each account
        accounts = [
            (AccountName.CASH, self.cash),
            (AccountName.ACCOUNTS_RECEIVABLE, self.accounts_receivable),
            (AccountName.INVENTORY, self.inventory),
            (AccountName.PREPAID_INSURANCE, self.prepaid_insurance),
            (AccountName.GROSS_PPE, self.gross_ppe),
            (AccountName.RESTRICTED_CASH, self.restricted_assets),
        ]

        for account, current_balance in accounts:
            if current_balance <= ZERO:
                continue
            new_balance = current_balance * ratio
            adjustment = new_balance - current_balance

            if adjustment > ZERO:
                # Increase asset
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=account,
                    credit_account=AccountName.RETAINED_EARNINGS,
                    amount=adjustment,
                    transaction_type=TransactionType.REVALUATION,
                    description=f"{description} - {account.value}",
                )
            elif adjustment < ZERO:
                # Decrease asset
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=account,
                    amount=-adjustment,
                    transaction_type=TransactionType.REVALUATION,
                    description=f"{description} - {account.value}",
                )

        # Also adjust accumulated depreciation proportionally (contra-asset)
        current_accum_depr = self.accumulated_depreciation
        if current_accum_depr > ZERO:
            new_accum_depr = current_accum_depr * ratio
            adjustment = new_accum_depr - current_accum_depr
            if adjustment > ZERO:
                # Increase accumulated depreciation (credit)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=AccountName.ACCUMULATED_DEPRECIATION,
                    amount=adjustment,
                    transaction_type=TransactionType.REVALUATION,
                    description=f"{description} - accumulated_depreciation",
                )
            elif adjustment < ZERO:
                # Decrease accumulated depreciation (debit)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.ACCUMULATED_DEPRECIATION,
                    credit_account=AccountName.RETAINED_EARNINGS,
                    amount=-adjustment,
                    transaction_type=TransactionType.REVALUATION,
                    description=f"{description} - accumulated_depreciation",
                )

    def _write_off_all_assets(self, description: str = "Asset write-off") -> None:
        """Write off all asset balances to zero through the ledger.

        Args:
            description: Description of the write-off
        """
        # Write off each asset account that has a balance
        asset_accounts = [
            (AccountName.CASH, self.cash),
            (AccountName.ACCOUNTS_RECEIVABLE, self.accounts_receivable),
            (AccountName.INVENTORY, self.inventory),
            (AccountName.PREPAID_INSURANCE, self.prepaid_insurance),
            (AccountName.GROSS_PPE, self.gross_ppe),
            (AccountName.RESTRICTED_CASH, self.restricted_assets),
            (AccountName.COLLATERAL, self.collateral),
        ]

        for account, balance in asset_accounts:
            if balance > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=account,
                    amount=balance,
                    transaction_type=TransactionType.WRITE_OFF,
                    description=f"{description} - {account.value}",
                )

        # Write off accumulated depreciation (contra-asset - has credit balance)
        accum_depr = self.accumulated_depreciation
        if accum_depr > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.ACCUMULATED_DEPRECIATION,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=accum_depr,
                transaction_type=TransactionType.WRITE_OFF,
                description=f"{description} - accumulated_depreciation",
            )

    def _record_liquidation(
        self,
        amount: Decimal,
        description: str = "Liquidation loss",
    ) -> None:
        """Record liquidation loss through the ledger.

        Used during bankruptcy to record asset liquidation costs.

        Args:
            amount: Amount of liquidation loss
            description: Description of the liquidation
        """
        if amount <= ZERO:
            return
        self.ledger.record_double_entry(
            date=self.current_year,
            debit_account=AccountName.RETAINED_EARNINGS,
            credit_account=AccountName.CASH,
            amount=amount,
            transaction_type=TransactionType.LIQUIDATION,
            description=description,
        )

    def _record_liquid_asset_reduction(
        self,
        total_reduction: Decimal,
        description: str = "Liquid asset reduction for claim payment",
    ) -> None:
        """Reduce liquid assets proportionally to make a payment.

        Reduces cash, accounts receivable, and inventory proportionally
        to fund a payment amount.

        Args:
            total_reduction: Total amount to reduce from liquid assets
            description: Description of the reduction
        """
        if total_reduction <= ZERO:
            return

        # Get current liquid asset balances
        current_cash = self.cash
        current_ar = self.accounts_receivable
        current_inventory = self.inventory
        total_liquid = current_cash + current_ar + current_inventory

        if total_liquid <= ZERO:
            return

        # Calculate reduction ratio
        if total_liquid <= total_reduction:
            # Use all liquid assets
            reduction_ratio = ONE
        else:
            reduction_ratio = total_reduction / total_liquid

        # Reduce each account proportionally
        cash_reduction = current_cash * reduction_ratio
        ar_reduction = current_ar * reduction_ratio
        inventory_reduction = current_inventory * reduction_ratio

        if cash_reduction > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=quantize_currency(cash_reduction),
                transaction_type=TransactionType.WRITE_OFF,
                description=f"{description} - cash",
            )

        if ar_reduction > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.ACCOUNTS_RECEIVABLE,
                amount=quantize_currency(ar_reduction),
                transaction_type=TransactionType.WRITE_OFF,
                description=f"{description} - accounts_receivable",
            )

        if inventory_reduction > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.INVENTORY,
                amount=quantize_currency(inventory_reduction),
                transaction_type=TransactionType.WRITE_OFF,
                description=f"{description} - inventory",
            )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "WidgetManufacturer":
        """Create a deep copy preserving all state for Monte Carlo forking.

        This method enables proper state preservation when forking simulations
        from a "warmed-up" state (e.g., Year 5 of a base trajectory). All
        internal state including accruals, ledger entries, claim liabilities,
        and metrics history are preserved.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this WidgetManufacturer with all state preserved

        Example:
            Fork a simulation from Year 5 state::

                import copy

                # Run to Year 5
                for _ in range(5):
                    manufacturer.step()

                # Fork for Monte Carlo
                forked = copy.deepcopy(manufacturer)
                forked.step()  # Year 6 in forked copy

                # Original still at Year 5
                assert manufacturer.current_year == 5
                assert forked.current_year == 6
        """
        import copy

        # Create new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Deep copy all attributes
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))

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

    @property
    def total_assets(self) -> Decimal:
        """Calculate total assets from all asset components.

        Total assets include all current and non-current assets following
        the accounting equation: Assets = Liabilities + Equity.

        Returns:
            Decimal: Total assets in dollars, sum of all asset components.
        """
        # Current assets
        current = self.cash + self.accounts_receivable + self.inventory + self.prepaid_insurance
        # Non-current assets
        net_ppe = self.gross_ppe - self.accumulated_depreciation
        # Total
        return current + net_ppe + self.restricted_assets

    @total_assets.setter
    def total_assets(self, value: Union[Decimal, float]) -> None:
        """Set total assets by proportionally adjusting all asset components.

        This setter maintains the relative proportions of all asset components
        when changing the total asset value. All changes go through the ledger
        to maintain consistency between Direct and Indirect cash flow methods.

        Args:
            value: New total asset value in dollars.
        """
        value_decimal = to_decimal(value)
        self._record_proportional_revaluation(
            target_total=value_decimal,
            description="Proportional asset revaluation via total_assets setter",
        )

    @property
    def total_liabilities(self) -> Decimal:
        """Calculate total liabilities from all liability components.

        Total liabilities include current liabilities and long-term claim liabilities.

        Note:
            ClaimLiability objects are the single source of truth for insurance claim
            liabilities. INSURANCE_CLAIMS in AccrualManager are excluded from this
            calculation to prevent double-counting. See GitHub issue #213.

        Returns:
            Decimal: Total liabilities in dollars, sum of all liability components.
        """
        # Get accrued expenses from accrual manager (excludes INSURANCE_CLAIMS to avoid
        # double-counting with ClaimLiability objects which are the source of truth)
        accrual_items = self.accrual_manager.get_balance_sheet_items()
        total_accrued_expenses = to_decimal(accrual_items.get("accrued_expenses", ZERO))

        # Subtract any insurance claims from accrual manager to prevent double-counting
        # ClaimLiability is the authoritative source for claim liabilities
        insurance_claims_in_accrual = sum(
            (
                to_decimal(a.remaining_balance)
                for a in self.accrual_manager.accrued_expenses.get(AccrualType.INSURANCE_CLAIMS, [])
                if not a.is_fully_paid
            ),
            ZERO,
        )
        adjusted_accrued_expenses = total_accrued_expenses - insurance_claims_in_accrual

        # Current liabilities - AccrualManager is single source of truth (issue #238)
        current_liabilities = self.accounts_payable + adjusted_accrued_expenses

        # Long-term liabilities (claim liabilities) - single source of truth
        claim_liability_total = sum(
            (liability.remaining_amount for liability in self.claim_liabilities), ZERO
        )

        return current_liabilities + claim_liability_total

    @property
    def equity(self) -> Decimal:
        """Calculate equity using the accounting equation.

        Equity is derived as Assets - Liabilities, ensuring the accounting
        equation always balances: Assets = Liabilities + Equity.

        Returns:
            Decimal: Shareholder equity in dollars.
        """
        return self.total_assets - self.total_liabilities

    @property
    def net_assets(self) -> Decimal:
        """Calculate net assets (total assets minus restricted assets).

        Net assets represent the portion of total assets that are available
        for operational use. Restricted assets are those pledged as collateral
        for insurance claims and cannot be used for general business purposes.

        Returns:
            Decimal: Net assets in dollars. Always non-negative as restricted
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
        return self.total_assets - self.restricted_assets

    def record_insurance_premium(
        self, premium_amount: Union[Decimal, float], is_annual: bool = False
    ) -> None:
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
        premium_decimal = to_decimal(premium_amount)
        if premium_decimal > ZERO:
            if is_annual:
                # COMPULSORY INSURANCE CHECK: Company cannot operate without upfront insurance
                # If unable to pay, company becomes insolvent
                if self.cash < premium_decimal:
                    logger.error(
                        f"INSOLVENCY: Cannot afford compulsory annual insurance premium. "
                        f"Required: ${premium_decimal:,.2f}, Available cash: ${self.cash:,.2f}. "
                        f"Company cannot operate without insurance."
                    )
                    # Mark as insolvent - company cannot operate without insurance
                    self.handle_insolvency()
                    return  # Exit - company is now insolvent and cannot proceed

                # Record as prepaid asset using insurance accounting module
                result = self.insurance_accounting.pay_annual_premium(premium_decimal)

                # Update balance sheet via ledger (Issue #275)
                # Debit prepaid insurance (asset increases), credit cash (decreases)
                cash_outflow = result["cash_outflow"]
                if cash_outflow > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.PREPAID_INSURANCE,
                        credit_account=AccountName.CASH,
                        amount=cash_outflow,
                        transaction_type=TransactionType.INSURANCE_PREMIUM,
                        description=f"Annual insurance premium payment",
                        month=self.current_month,
                    )

                # Store for later amortization tracking
                self._original_prepaid_premium = premium_decimal

                logger.info(f"Paid annual insurance premium: ${premium_decimal:,.2f}")
                logger.info(f"Monthly expense will be: ${result['monthly_expense']:,.2f}")
            else:
                # Record as direct expense for the period (backward compatibility)
                self.period_insurance_premiums += premium_decimal
                # Don't reduce cash here - expense is handled through net income calculation
                # to avoid double-counting (premiums reduce operating income -> net income -> equity)

                logger.info(f"Recorded insurance premium expense: ${premium_decimal:,.2f}")
                logger.debug(f"Period premiums total: ${self.period_insurance_premiums:,.2f}")

    def record_insurance_loss(self, loss_amount: Union[Decimal, float]) -> None:
        """Record insurance loss (deductible/retention) for tax deduction tracking.

        This method tracks insurance losses paid by the company during the current
        period for proper tax treatment. Company-paid losses (deductibles, retentions,
        excess over limits) are tax-deductible business expenses.

        Args:
            loss_amount: Loss amount paid by company in the current period.
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
        loss_decimal = to_decimal(loss_amount)
        if loss_decimal > ZERO:
            self.period_insurance_losses += loss_decimal
            logger.debug(f"Recorded insurance loss: ${loss_decimal:,.2f}")
            logger.debug(f"Period losses total: ${self.period_insurance_losses:,.2f}")

    def reset_period_insurance_costs(self) -> None:
        """Reset period insurance cost tracking for new period.

        This method clears the accumulated insurance premiums and losses
        from the current period, typically called at the end of each
        simulation step to prepare for the next period.

        Side Effects:
            - Resets period_insurance_premiums to ZERO
            - Resets period_insurance_losses to ZERO

        Note:
            Called automatically at the end of each step() to ensure
            costs are only counted once per period.
        """
        self.period_insurance_premiums = ZERO
        self.period_insurance_losses = ZERO

    @property
    def available_assets(self) -> Decimal:
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
        return self.total_assets - self.restricted_assets

    @property
    def total_claim_liabilities(self) -> Decimal:
        """Calculate total outstanding claim liabilities.

        Sums the remaining unpaid amounts across all active claim liabilities.
        This represents the company's total financial obligation for insurance
        claims that are being paid over time according to development schedules.

        Returns:
            Decimal: Total outstanding liability in dollars. Returns ZERO if no
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
        return sum((claim.remaining_amount for claim in self.claim_liabilities), ZERO)

    def calculate_revenue(self, apply_stochastic: bool = False) -> Decimal:
        """Calculate revenue based on available assets and turnover ratio.

        Revenue is calculated using the asset turnover ratio, which represents
        how efficiently the company converts assets into sales. The calculation
        can include stochastic shocks for realistic modeling.

        Args:
            apply_stochastic (bool): Whether to apply stochastic shock to revenue
                calculation. Requires stochastic_process to be initialized.
                Defaults to False for deterministic calculation.

        Returns:
            float: Annual revenue in dollars. Always non-negative.

        Raises:
            RuntimeError: If apply_stochastic=True but no stochastic process provided.

        Examples:
            Basic revenue calculation::

                # Deterministic revenue
                revenue = manufacturer.calculate_revenue()
                print(f"Base revenue: ${revenue:,.2f}")

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
            business growth or decline.

            Issue #244: Working capital components (AR, Inventory, AP) are calculated
            separately via :meth:`calculate_working_capital_components` and impact
            the cash flow statement. Revenue is not adjusted for working capital
            to avoid double-counting (GAAP-compliant).

        See Also:
            :attr:`asset_turnover_ratio`: Core parameter for revenue calculation.
            :meth:`calculate_working_capital_components`: Working capital calculation.
            :class:`~ergodic_insurance.stochastic_processes.StochasticProcess`:
            For stochastic modeling options.
        """
        # Ensure assets are non-negative for revenue calculation
        # (negative assets would mean business has ceased operations)
        available_assets = max(ZERO, self.total_assets)
        revenue = available_assets * to_decimal(self.asset_turnover_ratio)

        # Apply stochastic shock if requested and process is available
        if apply_stochastic and self.stochastic_process is not None:
            shock = self.stochastic_process.generate_shock(float(revenue))
            revenue *= to_decimal(shock)
            logger.debug(f"Applied stochastic shock: {shock:.4f}")

        logger.debug(f"Revenue calculated: ${revenue:,.2f} from assets ${self.total_assets:,.2f}")
        return revenue

    def calculate_operating_income(
        self, revenue: Union[Decimal, float], depreciation_expense: Union[Decimal, float] = ZERO
    ) -> Decimal:
        """Calculate operating income including insurance and depreciation as operating expenses.

        Operating income represents earnings before interest and taxes (EBIT),
        calculated by applying the base operating margin to revenue and then
        subtracting insurance costs (premiums and losses) and depreciation.
        This reflects the true operating profitability after all operating expenses.

        Args:
            revenue: Annual revenue in dollars. Must be >= 0.
            depreciation_expense: Depreciation expense for the period.
                Defaults to ZERO for backward compatibility.

        Returns:
            Decimal: Operating income in dollars after insurance costs and depreciation.
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
        revenue_decimal = to_decimal(revenue)
        depreciation_decimal = to_decimal(depreciation_expense)

        # Calculate base operating income using base margin
        base_operating_income = revenue_decimal * to_decimal(self.base_operating_margin)

        # Subtract insurance costs and depreciation to get actual operating income
        actual_operating_income = (
            base_operating_income
            - self.period_insurance_premiums
            - self.period_insurance_losses
            - depreciation_decimal
        )

        logger.debug(
            f"Operating income: ${actual_operating_income:,.2f} "
            f"(base: ${base_operating_income:,.2f}, insurance: "
            f"${self.period_insurance_premiums + self.period_insurance_losses:,.2f}, "
            f"depreciation: ${depreciation_decimal:,.2f})"
        )
        return actual_operating_income

    def calculate_collateral_costs(
        self, letter_of_credit_rate: Union[Decimal, float] = 0.015, time_period: str = "annual"
    ) -> Decimal:
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
            Decimal: Collateral costs for the specified period in dollars.
                Returns ZERO if no collateral is posted.

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
        insurance_premiums: Union[Decimal, float] = ZERO,
        insurance_losses: Union[Decimal, float] = ZERO,
        use_accrual: bool = True,
        time_resolution: str = "annual",
    ) -> Decimal:
        """Calculate net income after collateral costs and taxes.

        Net income represents the final profitability available to shareholders
        after all operating expenses, financing costs, and taxes. Insurance costs
        can be handled in two ways for backward compatibility:
        1. New way: Already included in operating_income via calculate_operating_income()
        2. Legacy way: Passed as separate parameters to this method

        Args:
            operating_income: Operating income (EBIT). May or may not include
                insurance costs depending on how it was calculated.
            collateral_costs: Financing costs for letter of credit
                collateral. Must be >= 0.
            insurance_premiums: Insurance premium costs to deduct. If operating_income
                already includes these, pass 0. Defaults to ZERO.
            insurance_losses: Insurance loss/deductible costs to deduct. If operating_income
                already includes these, pass 0. Defaults to ZERO.
            use_accrual: Whether to use accrual accounting for taxes.
                Defaults to True for quarterly tax payment schedule.
            time_resolution: Time resolution for tax accrual calculation.
                "annual" accrues full annual taxes, "monthly" accrues monthly portion.
                Defaults to "annual".

        Returns:
            Decimal: Net income after all expenses and taxes. Can be negative
                if the company operates at a loss after financing costs and taxes.

        Examples:
            Calculate full income statement with insurance costs::

                revenue = manufacturer.calculate_revenue()
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

            The tax calculation is delegated to :class:`TaxHandler` which:
            - Income before tax = operating_income - collateral_costs
              (Note: insurance costs are already included in operating_income)
            - Taxes = max(0, income_before_tax * tax_rate)
            - LIMITED LIABILITY: Caps tax accrual at current equity
            - Net income = income_before_tax - actual_tax_expense

            Tax Flow (No Circular Dependency):
            The tax logic reads equity BEFORE recording the tax accrual.
            This ensures no circular dependency: the equity used to cap the
            tax does NOT include the tax being calculated. See TaxHandler
            class documentation for detailed explanation.

        See Also:
            :class:`TaxHandler`: Consolidated tax calculation logic.
            :attr:`tax_rate`: Tax rate applied to positive income.
            :attr:`retention_ratio`: Portion of net income retained vs. distributed.
            :meth:`process_accrued_payments`: Pays accrued taxes.
        """
        # Convert inputs to Decimal
        operating_income_decimal = to_decimal(operating_income)
        collateral_costs_decimal = to_decimal(collateral_costs)
        insurance_premiums_decimal = to_decimal(insurance_premiums)
        insurance_losses_decimal = to_decimal(insurance_losses)

        # Deduct all costs from operating income
        # For backward compatibility, also deduct insurance costs if provided as parameters
        total_insurance_costs = insurance_premiums_decimal + insurance_losses_decimal
        income_before_tax = (
            operating_income_decimal - collateral_costs_decimal - total_insurance_costs
        )

        # Use TaxHandler for consolidated tax calculation
        # See TaxHandler docstring for explanation of tax flow and why there's no circular dependency
        tax_handler = TaxHandler(
            tax_rate=self.tax_rate,
            accrual_manager=self.accrual_manager,
        )

        # Calculate theoretical tax for logging purposes
        taxes = tax_handler.calculate_tax_liability(income_before_tax)

        # Read equity BEFORE tax accrual (critical for non-circular flow)
        # This equity value does NOT include the tax we're about to calculate
        current_equity = self.equity

        # Calculate and optionally accrue tax using consolidated handler
        actual_tax_expense, was_capped = tax_handler.calculate_and_accrue_tax(
            income_before_tax=income_before_tax,
            current_equity=current_equity,
            use_accrual=use_accrual,
            time_resolution=time_resolution,
            current_year=self.current_year,
            current_month=self.current_month,
        )

        # Net income uses the actual tax expense (which may be capped)
        net_income = income_before_tax - actual_tax_expense

        # Enhanced profit waterfall logging for complete transparency
        logger.info("===== PROFIT WATERFALL =====")
        logger.info(f"Operating Income:        ${operating_income_decimal:,.2f}")
        if insurance_premiums_decimal > ZERO:
            logger.info(f"  - Insurance Premiums:  ${insurance_premiums_decimal:,.2f}")
        if insurance_losses_decimal > ZERO:
            logger.info(f"  - Insurance Losses:    ${insurance_losses_decimal:,.2f}")
        if collateral_costs_decimal > ZERO:
            logger.info(f"  - Collateral Costs:    ${collateral_costs_decimal:,.2f}")
        logger.info(f"Income Before Tax:       ${income_before_tax:,.2f}")
        logger.info(f"  - Taxes (@{self.tax_rate:.1%}):      ${actual_tax_expense:,.2f}")
        if was_capped:
            logger.info(f"    (Capped from ${taxes:,.2f} due to limited liability)")
        if use_accrual and actual_tax_expense > 0:
            logger.info(f"    (Accrued for quarterly payment)")
        logger.info(f"NET INCOME:              ${net_income:,.2f}")
        logger.info("============================")

        # Validation assertion: ensure net income is less than or equal to operating income
        # when additional costs exist (beyond those already in operating income)
        # Note: Since insurance costs may already be included in operating_income,
        # we only check for meaningful differences
        epsilon = to_decimal("0.000000001")  # Use small epsilon for Decimal comparison
        if total_insurance_costs + collateral_costs_decimal > epsilon:
            assert (
                net_income <= operating_income_decimal + epsilon
            ), f"Net income ({net_income}) should be less than or equal to operating income ({operating_income_decimal}) when costs exist"

        return net_income

    def update_balance_sheet(
        self, net_income: Union[Decimal, float], growth_rate: Union[Decimal, float] = 0.0
    ) -> None:
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
        # Convert input to Decimal
        net_income_decimal = to_decimal(net_income)

        # Validation: retention ratio should be applied to net income (not revenue or operating income)
        # This is the profit after ALL costs including taxes
        assert 0 <= self.retention_ratio <= 1, f"Invalid retention ratio: {self.retention_ratio}"

        # Calculate retained earnings
        retention_decimal = to_decimal(self.retention_ratio)
        retained_earnings = net_income_decimal * retention_decimal
        dividends = net_income_decimal * (to_decimal(1) - retention_decimal)

        # Log retention calculation details
        logger.info("===== RETENTION CALCULATION =====")
        logger.info(f"Net Income:              ${net_income_decimal:,.2f}")
        logger.info(f"Retention Ratio:         {self.retention_ratio:.1%}")
        logger.info(f"Retained Earnings:       ${retained_earnings:,.2f}")
        if net_income_decimal > ZERO:
            logger.info(f"Dividends Distributed:   ${dividends:,.2f}")
        else:
            logger.info(f"Loss Absorption:         ${retained_earnings:,.2f}")
        logger.info("=================================")

        # LIMITED LIABILITY: Check liquidity and solvency before absorbing losses
        if retained_earnings < ZERO:
            current_equity = self.equity
            available_cash = self.cash
            tolerance = to_decimal(self.config.insolvency_tolerance)
            loss_amount = abs(retained_earnings)

            # Check 1: Already insolvent by equity threshold
            if current_equity <= tolerance:
                logger.warning(
                    f"LIMITED LIABILITY: Equity too low (${current_equity:,.2f}) to absorb any losses. "
                    f"Cannot absorb loss of ${loss_amount:,.2f}. "
                    f"Company is already insolvent (threshold: ${tolerance:,.2f})."
                )
                # Don't reduce cash - company is already insolvent
                # Insolvency will be detected in check_solvency()
                self._last_dividends_paid = ZERO  # No dividends when insolvent
                return

            # Check 2: LIQUIDITY CHECK - must have cash to pay loss
            # If loss exceeds available cash, company cannot meet obligations → insolvency
            if loss_amount > available_cash:
                logger.error(
                    f"LIQUIDITY CRISIS → INSOLVENCY: Loss ${loss_amount:,.2f} exceeds "
                    f"available cash ${available_cash:,.2f}. Equity=${current_equity:,.2f}. "
                    f"Company cannot meet obligations despite positive book equity."
                )
                self._last_dividends_paid = ZERO  # No dividends when insolvent
                self.handle_insolvency()
                return  # Exit - company is now insolvent

            # Check 3: Would paying the loss trigger equity insolvency?
            equity_after_loss = current_equity - loss_amount
            if equity_after_loss <= tolerance:
                logger.error(
                    f"EQUITY INSOLVENCY: Loss ${loss_amount:,.2f} would push equity to "
                    f"${equity_after_loss:,.2f}, below threshold ${tolerance:,.2f}. "
                    f"Current equity=${current_equity:,.2f}. Triggering insolvency."
                )
                # Apply the loss to the balance sheet via ledger (Issue #275)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=AccountName.CASH,
                    amount=loss_amount,
                    transaction_type=TransactionType.EXPENSE,
                    description=f"Year {self.current_year} operating loss (pre-insolvency)",
                    month=self.current_month,
                )
                self._last_dividends_paid = ZERO  # No dividends when insolvent
                self.handle_insolvency()
                return  # Exit - company is now insolvent

            # All checks passed - absorb the full loss via ledger (Issue #275)
            # The ledger entry is the single source of truth for cash balance
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=loss_amount,
                transaction_type=TransactionType.EXPENSE,
                description=f"Year {self.current_year} operating loss",
                month=self.current_month,
            )

            self._last_dividends_paid = ZERO  # No dividends on losses
            logger.info(
                f"Absorbed loss: ${loss_amount:,.2f}. "
                f"Remaining cash: ${self.cash:,.2f}, Remaining equity: ${self.equity:,.2f}"
            )
        else:
            # Positive retained earnings - check cash constraints for dividends
            # Issue #239: Only pay dividends if there's actually cash to pay them
            projected_cash = self.cash + retained_earnings

            if projected_cash <= ZERO:
                # Can't pay any dividends - company has no cash after operations
                actual_dividends = ZERO
                additional_retained = dividends  # Keep all earnings
                logger.warning(
                    f"DIVIDEND CONSTRAINT: Projected cash ${projected_cash:,.2f} <= 0. "
                    f"No dividends can be paid. All ${dividends:,.2f} retained."
                )
            elif projected_cash < dividends:
                # Can only pay partial dividends
                actual_dividends = projected_cash  # Pay what we can, keeping cash at $0
                additional_retained = dividends - actual_dividends
                logger.warning(
                    f"DIVIDEND CONSTRAINT: Projected cash ${projected_cash:,.2f} "
                    f"< theoretical dividends ${dividends:,.2f}. "
                    f"Paying only ${actual_dividends:,.2f}."
                )
            else:
                # Can pay full dividends
                actual_dividends = dividends
                additional_retained = ZERO

            # Record net income via ledger (Issue #275)
            # Net income increases assets (cash) and retained earnings by the FULL amount
            # Dividends are then paid from cash, reducing assets
            # Net effect: assets increase by (net_income - dividends) = retained_earnings
            self._last_dividends_paid = actual_dividends

            # Calculate total amount to add to cash:
            # = net_income (the full profit) - actual_dividends (paid out)
            # = retained_earnings + (dividends - actual_dividends) = total actually retained
            total_retained = retained_earnings + additional_retained
            # Net cash increase = total_retained (what stays in company after dividends)
            # This is equivalent to: net_income - actual_dividends

            if total_retained > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.CASH,
                    credit_account=AccountName.RETAINED_EARNINGS,
                    amount=total_retained,
                    transaction_type=TransactionType.REVENUE,
                    description=f"Year {self.current_year} retained earnings",
                    month=self.current_month,
                )

            # Note: Dividends are NOT recorded as a separate cash outflow here because
            # the net_income already excludes them. The total_retained calculation
            # (retained_earnings + additional_retained) gives us the net cash impact.
            # Recording dividends as cash outflow would double-count.
            # However, we do record the dividend declaration for financial statement purposes.
            if actual_dividends > ZERO:
                # Record dividends declared (affects equity, not cash again)
                # This entry: Debit Retained Earnings, Credit Dividends Payable (or direct to equity)
                # But since we're on a cash basis here, we just track it for reporting
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=AccountName.DIVIDENDS,
                    amount=actual_dividends,
                    transaction_type=TransactionType.DIVIDEND,
                    description=f"Year {self.current_year} dividends declared",
                    month=self.current_month,
                )

        logger.info(
            f"Balance sheet updated: Assets=${self.total_assets:,.2f}, Equity=${self.equity:,.2f}"
        )
        if self._last_dividends_paid > ZERO:
            logger.info(f"Dividends paid: ${self._last_dividends_paid:,.2f}")

    def calculate_working_capital_components(
        self, revenue: Union[Decimal, float], dso: float = 45, dio: float = 60, dpo: float = 30
    ) -> Dict[str, Decimal]:
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
        # Convert inputs to Decimal
        revenue_decimal = to_decimal(revenue)
        dso_decimal = to_decimal(dso)
        dio_decimal = to_decimal(dio)
        dpo_decimal = to_decimal(dpo)
        days_per_year = to_decimal(365)

        # Calculate cost of goods sold (approximate as % of revenue)
        cogs = revenue_decimal * to_decimal(1 - self.base_operating_margin)

        # Calculate new working capital components
        new_ar = revenue_decimal * (dso_decimal / days_per_year)
        new_inventory = cogs * (dio_decimal / days_per_year)  # Inventory based on COGS not revenue
        new_ap = cogs * (dpo_decimal / days_per_year)  # AP based on COGS not revenue

        # Calculate the change in working capital components
        ar_change = new_ar - self.accounts_receivable
        inventory_change = new_inventory - self.inventory
        ap_change = new_ap - self.accounts_payable

        # Record working capital changes in ledger
        # AR increase: Debit AR, Credit Cash (collection cycle)
        if ar_change > 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account="accounts_receivable",
                credit_account="cash",
                amount=ar_change,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description=f"AR increase from revenue growth",
                month=self.current_month,
            )
        elif ar_change < 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account="cash",
                credit_account="accounts_receivable",
                amount=abs(ar_change),
                transaction_type=TransactionType.COLLECTION,
                description=f"AR decrease (net collections)",
                month=self.current_month,
            )

        # Inventory increase: Debit Inventory, Credit Cash (purchase cycle)
        if inventory_change > 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account="inventory",
                credit_account="cash",
                amount=inventory_change,
                transaction_type=TransactionType.INVENTORY_PURCHASE,
                description=f"Inventory increase",
                month=self.current_month,
            )
        elif inventory_change < 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account="cash",
                credit_account="inventory",
                amount=abs(inventory_change),
                transaction_type=TransactionType.WORKING_CAPITAL,
                description=f"Inventory decrease",
                month=self.current_month,
            )

        # AP increase: Debit Cash, Credit AP (we owe more to vendors)
        if ap_change > 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account="cash",
                credit_account="accounts_payable",
                amount=ap_change,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description=f"AP increase from purchases",
                month=self.current_month,
            )
        elif ap_change < 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account="accounts_payable",
                credit_account="cash",
                amount=abs(ap_change),
                transaction_type=TransactionType.PAYMENT,
                description=f"AP decrease (payments to vendors)",
                month=self.current_month,
            )

        # Working capital components are now updated via ledger entries above (Issue #275)
        # The ledger is the single source of truth - no direct assignments needed

        # Calculate cash impact for logging (the actual impact is recorded in ledger entries)
        # Increases in AR and inventory reduce cash (cash converted to these assets)
        # Increases in AP increase cash (we have the cash but owe it to vendors)
        cash_impact = -(ar_change + inventory_change) + ap_change

        # LIMITED LIABILITY: Check if ledger-based changes would make cash negative
        # The ledger entries above have already been recorded, so we need to handle
        # the case where cash went negative by recording an adjustment
        if self.cash < ZERO:
            # Cash went negative from working capital changes - record adjustment
            adjustment = -self.cash
            logger.warning(
                f"LIMITED LIABILITY: Working capital changes pushed cash to ${self.cash:,.2f}. "
                f"Adjusting to floor at $0."
            )
            self._record_cash_adjustment(
                amount=adjustment,
                description="Limited liability floor - working capital cash constraint",
                transaction_type=TransactionType.ADJUSTMENT,
            )
            # Recalculate effective cash impact
            cash_impact = cash_impact + adjustment

        # Now total assets remain constant - we've just reallocated between components
        # This prevents artificial asset creation that was causing growth distortions

        # Calculate net working capital and cash conversion cycle
        net_working_capital = self.accounts_receivable + self.inventory - self.accounts_payable
        cash_conversion_cycle_days = dso_decimal + dio_decimal - dpo_decimal

        logger.debug(
            f"Working capital components: AR=${self.accounts_receivable:,.0f}, "
            f"Inv=${self.inventory:,.0f}, AP=${self.accounts_payable:,.0f}, "
            f"Net WC=${net_working_capital:,.0f}, Cash impact=${cash_impact:,.0f}"
        )

        return {
            "accounts_receivable": self.accounts_receivable,
            "inventory": self.inventory,
            "accounts_payable": self.accounts_payable,
            "net_working_capital": net_working_capital,
            "cash_conversion_cycle": cash_conversion_cycle_days,
            "cash_impact": cash_impact,
        }

    def record_prepaid_insurance(self, annual_premium: Union[Decimal, float]) -> None:
        """Record annual insurance premium payment as prepaid expense.

        Records the payment of an annual insurance premium as a prepaid asset
        that will be amortized monthly over the coverage period.

        Args:
            annual_premium (float): Annual insurance premium paid in advance.

        Side Effects:
            - Increases prepaid_insurance by annual_premium
            - Decreases cash by annual_premium
            - May trigger insolvency if company cannot afford payment

        Examples:
            Record annual premium payment::

                manufacturer.record_prepaid_insurance(1_200_000)
                # Prepaid insurance increases by $1.2M
                # Will amortize at $100K/month over 12 months

        Note:
            Annual insurance is considered compulsory for operation. If the
            company cannot afford the premium, it becomes insolvent.
        """
        annual_premium_decimal = to_decimal(annual_premium)
        if annual_premium_decimal > ZERO:
            # COMPULSORY INSURANCE CHECK: Company cannot operate without upfront insurance
            # If unable to pay, company becomes insolvent
            if self.cash < annual_premium_decimal:
                logger.error(
                    f"INSOLVENCY: Cannot afford compulsory annual insurance premium. "
                    f"Required: ${annual_premium_decimal:,.2f}, Available cash: ${self.cash:,.2f}. "
                    f"Company cannot operate without insurance."
                )
                # Mark as insolvent - company cannot operate without insurance
                self.handle_insolvency()
                return  # Exit - company is now insolvent and cannot proceed

            # Use insurance accounting module to properly track prepaid insurance
            result = self.insurance_accounting.pay_annual_premium(annual_premium_decimal)

            # Update balance sheet via ledger only (Issue #275)
            # The ledger is the single source of truth
            if result["cash_outflow"] > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.PREPAID_INSURANCE,
                    credit_account=AccountName.CASH,
                    amount=result["cash_outflow"],
                    transaction_type=TransactionType.INSURANCE_PREMIUM,
                    description=f"Annual insurance premium payment",
                    month=self.current_month,
                )

            logger.info(f"Recorded prepaid insurance: ${annual_premium_decimal:,.2f}")

    def amortize_prepaid_insurance(self, months: int = 1) -> Decimal:
        """Amortize prepaid insurance over time using GAAP straight-line method.

        Reduces prepaid insurance balance and records the expense for the period.
        Typically called monthly to amortize annual premiums. Uses the insurance
        accounting module to ensure proper amortization calculations.

        Args:
            months (int): Number of months to amortize. Defaults to 1.

        Returns:
            Decimal: Amount amortized (insurance expense for the period).

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
        total_amortized = ZERO

        # Use insurance accounting module for proper amortization
        for _ in range(months):
            if self.prepaid_insurance > ZERO:
                result = self.insurance_accounting.record_monthly_expense()

                # Update P&L tracking (not balance sheet - that's via ledger)
                self.period_insurance_premiums += result["insurance_expense"]
                total_amortized += result["insurance_expense"]

                # Record insurance expense amortization via ledger only (Issue #275)
                # The ledger is the single source of truth for prepaid_insurance balance
                if result["insurance_expense"] > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.INSURANCE_EXPENSE,
                        credit_account=AccountName.PREPAID_INSURANCE,
                        amount=result["insurance_expense"],
                        transaction_type=TransactionType.EXPENSE,
                        description=f"Insurance premium amortization",
                        month=self.current_month,
                    )

                logger.debug(
                    f"Month {self.insurance_accounting.current_month}: "
                    f"Expense ${result['insurance_expense']:,.2f}, "
                    f"Remaining prepaid ${result['remaining_prepaid']:,.2f}"
                )

        return total_amortized

    def receive_insurance_recovery(
        self, amount: Union[Decimal, float], claim_id: Optional[str] = None
    ) -> Dict[str, Decimal]:
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
            return {
                "cash_received": ZERO,
                "receivable_reduction": ZERO,
                "remaining_receivables": ZERO,
            }

        # Record receipt through insurance accounting module
        result = self.insurance_accounting.receive_recovery_payment(amount, claim_id)

        # Update cash position via ledger (Issue #275)
        if result["cash_received"] > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.INSURANCE_RECEIVABLES,
                amount=result["cash_received"],
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description=f"Insurance recovery received",
                month=self.current_month,
            )

        logger.info(f"Received insurance recovery: ${result['cash_received']:,.2f}")
        logger.debug(f"Remaining receivables: ${result['remaining_receivables']:,.2f}")

        return result

    def record_depreciation(self, useful_life_years: Union[Decimal, float, int] = 10) -> Decimal:
        """Record straight-line depreciation on PP&E.

        Calculates and records annual depreciation expense using the straight-line
        method. Depreciation reduces the net book value of fixed assets over time.

        Args:
            useful_life_years: Average useful life of PP&E in years.
                Typical manufacturing equipment: 7-15 years. Defaults to 10.

        Returns:
            Decimal: Annual depreciation expense recorded.

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
        useful_life = to_decimal(useful_life_years)
        if self.gross_ppe > ZERO and useful_life > ZERO:
            annual_depreciation = self.gross_ppe / useful_life

            # Don't depreciate below zero net book value
            net_ppe = self.gross_ppe - self.accumulated_depreciation
            if net_ppe > ZERO:
                depreciation_expense = min(annual_depreciation, net_ppe)

                # Record depreciation via ledger only (Issue #275)
                # The ledger is the single source of truth for accumulated_depreciation
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.DEPRECIATION_EXPENSE,
                    credit_account=AccountName.ACCUMULATED_DEPRECIATION,
                    amount=depreciation_expense,
                    transaction_type=TransactionType.DEPRECIATION,
                    description=f"Year {self.current_year} depreciation",
                    month=self.current_month,
                )

                logger.debug(
                    f"Recorded depreciation: ${depreciation_expense:,.2f}, "
                    f"Accumulated: ${self.accumulated_depreciation:,.2f}"
                )
                return depreciation_expense
        return ZERO

    @property
    def net_ppe(self) -> Decimal:
        """Calculate net property, plant & equipment after depreciation.

        Returns:
            Decimal: Net PP&E (gross PP&E minus accumulated depreciation).
        """
        return self.gross_ppe - self.accumulated_depreciation

    def process_insurance_claim(
        self,
        claim_amount: Union[Decimal, float],
        deductible_amount: Union[Decimal, float] = ZERO,
        insurance_limit: Union[Decimal, float, None] = None,
        insurance_recovery: Optional[Union[Decimal, float]] = None,
    ) -> tuple[Decimal, Decimal]:
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
            claim_amount: Total amount of the loss/claim in dollars.
                Must be >= 0.
            deductible_amount: Amount company must pay before insurance kicks in
                (legacy parameter). Defaults to 0.0.
            insurance_limit: Maximum amount insurance will pay per claim
                (legacy parameter). Defaults to unlimited. Use insurance_recovery
                instead for new code.
            insurance_recovery: Pre-calculated insurance recovery
                amount (preferred). If provided, overrides deductible/limit
                calculation. Should be the exact amount insurance will pay.

        Returns:
            tuple[Decimal, Decimal]: Tuple of (company_payment, insurance_payment)
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
        # Convert all inputs to Decimal
        claim = to_decimal(claim_amount)
        deductible = to_decimal(deductible_amount)
        limit = (
            to_decimal(insurance_limit) if insurance_limit is not None else to_decimal(1e18)
        )  # Practical "infinity"

        # Handle new style parameters if provided
        if insurance_recovery is not None:
            # Use pre-calculated recovery
            insurance_payment = to_decimal(insurance_recovery)
            company_payment = claim - insurance_payment
        else:
            # Calculate insurance coverage
            if claim <= deductible:
                # Below deductible, company pays all
                company_payment = claim
                insurance_payment = ZERO
            else:
                # Above deductible
                company_payment = deductible
                insurance_payment = min(claim - deductible, limit)
                # Company also pays any amount above the limit
                if claim > deductible + limit:
                    company_payment += claim - deductible - limit

        # Company payment is collateralized and paid over time
        if company_payment > ZERO:
            # LIMITED LIABILITY: Cap company payment at available equity AND available cash
            current_equity = self.equity
            available_cash = self.cash
            # Can only post collateral up to the lesser of equity and cash
            max_payable: Decimal = (
                min(company_payment, current_equity, available_cash)
                if current_equity > ZERO
                else ZERO
            )
            unpayable_amount = company_payment - max_payable

            if max_payable > ZERO:
                # Post letter of credit as collateral for the payable amount
                # Transfer cash to restricted assets via ledger (Issue #275)
                # This maintains consistency between Direct and Indirect cash flow methods
                self._record_asset_transfer(
                    from_account=AccountName.CASH,
                    to_account=AccountName.RESTRICTED_CASH,
                    amount=max_payable,
                    description=f"Cash to restricted for insurance claim collateral",
                )
                # Also record the collateral posting
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.COLLATERAL,
                    credit_account=AccountName.RETAINED_EARNINGS,
                    amount=max_payable,
                    transaction_type=TransactionType.TRANSFER,
                    description=f"Letter of credit collateral posted for insurance claim",
                )

                # Create claim liability with payment schedule for payable portion
                # Adjust year_incurred only if current_year > 0 (after step() has been called)
                year_incurred = (
                    self.current_year - 1 if self.current_year > 0 else self.current_year
                )
                claim_liability = ClaimLiability(
                    original_amount=max_payable,
                    remaining_amount=max_payable,
                    year_incurred=year_incurred,  # Adjust for timing: claim occurred before step() incremented year
                    is_insured=True,  # This is the company portion of an insured claim
                )
                self.claim_liabilities.append(claim_liability)

                logger.info(
                    f"Company portion: ${max_payable:,.2f} - collateralized with payment schedule"
                )
                logger.info(
                    f"Posted ${max_payable:,.2f} letter of credit as collateral for company portion"
                )

            # Handle unpayable portion (exceeds equity/cash)
            if unpayable_amount > ZERO:
                # LIMITED LIABILITY: Only create liability if we can afford it (won't make equity negative)
                # Check current equity after posting collateral
                current_equity_after_collateral = self.equity
                max_liability: Decimal = (
                    min(unpayable_amount, current_equity_after_collateral)
                    if current_equity_after_collateral > ZERO
                    else ZERO
                )

                if max_liability > ZERO:
                    # Create liability for the amount we can afford
                    # Adjust year_incurred only if current_year > 0 (after step() has been called)
                    year_incurred = (
                        self.current_year - 1 if self.current_year > 0 else self.current_year
                    )
                    unpayable_claim = ClaimLiability(
                        original_amount=max_liability,
                        remaining_amount=max_liability,
                        year_incurred=year_incurred,  # Adjust for timing: claim occurred before step() incremented year
                        is_insured=False,  # No insurance coverage for this portion
                    )
                    self.claim_liabilities.append(unpayable_claim)

                    logger.warning(
                        f"LIMITED LIABILITY: Company payment capped at ${max_payable:,.2f} (cash/equity). "
                        f"Additional liability recorded: ${max_liability:,.2f}"
                    )

                # Log the truly unpayable amount that can't even be recorded as liability
                truly_unpayable = unpayable_amount - max_liability
                if truly_unpayable > ZERO:
                    logger.warning(
                        f"LIMITED LIABILITY: Cannot record ${truly_unpayable:,.2f} as liability "
                        f"(would violate limited liability). Company is insolvent."
                    )

                # Check if company is now insolvent
                if self.equity <= ZERO:
                    self.check_solvency()

            # Note: We don't record an insurance loss expense here because the liability
            # creation already reduces equity via the accounting equation (Assets - Liabilities = Equity).
            # Recording it as both a liability AND an expense would double-count the impact.
            # The tax deduction flows through naturally as the liability impacts equity.

        # Insurance payment creates a receivable
        if insurance_payment > ZERO:
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
        self, claim_amount: Union[Decimal, float], immediate_payment: bool = False
    ) -> Decimal:
        """Process an uninsured claim paid by company over time without collateral.

        This method handles claims where the company has no insurance coverage
        and must pay the full amount over time. Unlike insured claims, no collateral
        is required since there's no insurance company to secure payment to.

        Args:
            claim_amount: Total amount of the claim in dollars. Must be >= 0.
            immediate_payment: If True, pays entire amount immediately.
                If False, creates liability with payment schedule. Defaults to False.

        Returns:
            Decimal: The claim amount processed (for consistency with other methods).

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
        claim = to_decimal(claim_amount)
        if claim <= ZERO:
            return ZERO

        if immediate_payment:
            # LIMITED LIABILITY: Cap payment at available equity to prevent negative equity
            equity_before_payment = self.equity
            max_payable: Decimal = (
                min(claim, equity_before_payment) if equity_before_payment > ZERO else ZERO
            )

            # Pay immediately - reduce cash, capped at equity
            # First, try to pay from cash
            cash_payment: Decimal = min(max_payable, self.cash)
            remaining_to_pay: Decimal = max_payable - cash_payment

            # If cash isn't enough, liquidate other current assets proportionally
            if remaining_to_pay > ZERO:
                # Total liquid assets we can use (cash + AR + inventory)
                liquid_assets = self.cash + self.accounts_receivable + self.inventory
                if liquid_assets > ZERO:
                    # Pay what we can from liquid assets, but not more than max_payable
                    actual_payment: Decimal = min(max_payable, liquid_assets)

                    # Reduce liquid assets proportionally via ledger (Issue #275)
                    self._record_liquid_asset_reduction(
                        total_reduction=actual_payment,
                        description="Liquid asset reduction for uninsured claim payment",
                    )
                else:
                    actual_payment = ZERO
            else:
                # Cash was sufficient - record via ledger (Issue #275)
                actual_payment = cash_payment
                if cash_payment > ZERO:
                    self._record_liquidation(
                        amount=cash_payment,
                        description="Cash payment for uninsured claim",
                    )

            # Record as tax-deductible loss (only what we actually paid)
            self.period_insurance_losses += actual_payment

            # Create a liability for the unpaid portion (shortfall)
            shortfall = claim - actual_payment
            if shortfall > ZERO:
                # LIMITED LIABILITY: After making payment, create liability up to available equity
                # The payment already reduced equity. Creating additional liability reduces equity further.
                # We can ONLY create liability up to remaining equity to prevent negative equity.

                current_equity_after_payment = self.equity

                # Create liability up to available equity (prevents equity from going below zero)
                # This properly accounts for the loss while enforcing limited liability
                max_liability: Decimal = (
                    min(shortfall, current_equity_after_payment)
                    if current_equity_after_payment > ZERO
                    else ZERO
                )

                if max_liability > ZERO:
                    # Create liability for the portion we can afford without going insolvent
                    claim_liability = ClaimLiability(
                        original_amount=max_liability,
                        remaining_amount=max_liability,
                        year_incurred=self.current_year,  # No adjustment: immediate payment occurs in current period
                        is_insured=False,  # This is an uninsured claim
                    )
                    self.claim_liabilities.append(claim_liability)
                    logger.info(
                        f"LIMITED LIABILITY: Immediate payment ${actual_payment:,.2f}, "
                        f"created liability for ${max_liability:,.2f} (total claim: ${claim:,.2f})"
                    )

                # Log the truly unpayable amount (amount exceeding both liquid assets and equity)
                truly_unpayable = shortfall - max_liability
                if truly_unpayable > ZERO:
                    logger.warning(
                        f"LIMITED LIABILITY: Cannot record ${truly_unpayable:,.2f} of ${claim:,.2f} claim as liability "
                        f"(would violate limited liability). "
                        f"Paid ${actual_payment:,.2f}, liability ${max_liability:,.2f}, shortfall ${truly_unpayable:,.2f}."
                    )

                # Check if company is now insolvent
                if self.equity <= ZERO:
                    self.check_solvency()
            else:
                logger.info(f"Paid uninsured claim immediately: ${actual_payment:,.2f}")
            return claim  # Return the full claim amount processed

        # Create liability without collateral for payment over time
        # LIMITED LIABILITY: Only create liability up to available equity
        current_equity = self.equity
        deferred_max_liability: Decimal = (
            min(claim, current_equity) if current_equity > ZERO else ZERO
        )

        if deferred_max_liability > ZERO:
            # Adjust year_incurred only if current_year > 0 (after step() has been called)
            year_incurred = self.current_year - 1 if self.current_year > 0 else self.current_year
            claim_liability = ClaimLiability(
                original_amount=deferred_max_liability,
                remaining_amount=deferred_max_liability,
                year_incurred=year_incurred,  # Adjust for timing: claim occurred before step() incremented year
                is_insured=False,  # This is an uninsured claim
            )
            self.claim_liabilities.append(claim_liability)
            logger.info(
                f"Created uninsured claim liability: ${deferred_max_liability:,.2f} (no collateral required)"
            )

        # Log truly unpayable amount
        unpayable = claim - deferred_max_liability
        if unpayable > ZERO:
            logger.warning(
                f"LIMITED LIABILITY: Cannot record ${unpayable:,.2f} as liability "
                f"(would violate limited liability). Company may become insolvent."
            )
            # Check solvency if we couldn't create the full liability
            if self.equity <= ZERO:
                self.check_solvency()

        return claim

    def pay_claim_liabilities(self, max_payable: Optional[Union[Decimal, float]] = None) -> Decimal:
        """Pay scheduled claim liabilities for the current year.

        This method processes all scheduled claim payments based on each claim's
        development pattern and the years elapsed since incurrence. It maintains
        a minimum cash balance and reduces both collateral and restricted assets
        as payments are made.

        The method automatically removes fully paid claims from the active
        liability list to maintain clean accounting records.

        Args:
            max_payable: Optional maximum amount that can be paid (for coordinated
                limited liability enforcement). If None, caps at current equity.

        Returns:
            Decimal: Total amount paid toward claims in dollars. May be less than
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
        total_paid: Decimal = ZERO
        min_cash_balance = to_decimal(100_000)

        # LIMITED LIABILITY: Calculate total scheduled payments and cap at equity or provided max
        total_scheduled: Decimal = ZERO
        for claim_item in self.claim_liabilities:
            years_since = self.current_year - claim_item.year_incurred
            scheduled_payment = claim_item.get_payment(years_since)
            total_scheduled += scheduled_payment

        # Cap total payments at available liquid resources or provided max
        if max_payable is not None:
            # Use coordinated cap from step() method
            max_total_payable: Decimal = min(total_scheduled, to_decimal(max_payable))
        else:
            # Fallback to liquidity-based cap if called standalone
            available_liquidity = self.cash + self.restricted_assets
            max_total_payable = (
                min(total_scheduled, available_liquidity) if available_liquidity > ZERO else ZERO
            )

        # If we need to cap payments, calculate reduction ratio
        payment_ratio: Decimal = ONE
        if total_scheduled > max_total_payable and total_scheduled > ZERO:
            payment_ratio = max_total_payable / total_scheduled
            logger.warning(
                f"LIQUIDITY CONSTRAINT: Capping claim payments at ${max_total_payable:,.2f} "
                f"(scheduled: ${total_scheduled:,.2f}, available liquidity: ${self.cash + self.restricted_assets:,.2f})"
            )

        for claim_item in self.claim_liabilities:
            years_since = self.current_year - claim_item.year_incurred
            scheduled_payment = claim_item.get_payment(years_since)

            if scheduled_payment > ZERO:
                # Apply payment ratio to cap at available cash
                capped_scheduled = scheduled_payment * payment_ratio

                if claim_item.is_insured:
                    # For insured claims: Pay from restricted assets (collateral)
                    available_for_payment = min(capped_scheduled, self.restricted_assets)
                    actual_payment = available_for_payment

                    if actual_payment > ZERO:
                        claim_item.make_payment(actual_payment)
                        total_paid += actual_payment

                        # Record claim payment from restricted assets via ledger only (Issue #275)
                        # The ledger is the single source of truth for restricted_assets
                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.CLAIM_LIABILITIES,
                            credit_account=AccountName.RESTRICTED_CASH,
                            amount=actual_payment,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description=f"Insured claim payment from collateral",
                            month=self.current_month,
                        )
                        # Also reduce collateral tracking via ledger
                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.RETAINED_EARNINGS,
                            credit_account=AccountName.COLLATERAL,
                            amount=actual_payment,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description=f"Collateral released after claim payment",
                            month=self.current_month,
                        )

                        logger.debug(
                            f"Reduced collateral and restricted assets by ${actual_payment:,.2f}"
                        )
                        # Do NOT record as tax-deductible loss here - already recorded when claim incurred
                else:
                    # For uninsured claims: Pay from available cash
                    available_for_payment = max(
                        ZERO, self.cash - min_cash_balance
                    )  # Keep minimum cash
                    actual_payment = min(capped_scheduled, available_for_payment)

                    if actual_payment > ZERO:
                        claim_item.make_payment(actual_payment)
                        total_paid += actual_payment

                        # Record uninsured claim payment via ledger only (Issue #275)
                        # The ledger is the single source of truth for cash
                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.CLAIM_LIABILITIES,
                            credit_account=AccountName.CASH,
                            amount=actual_payment,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description=f"Uninsured claim payment",
                            month=self.current_month,
                        )

                        logger.debug(
                            f"Paid ${actual_payment:,.2f} toward uninsured claim (regular business expense)"
                        )

        # Remove fully paid claims
        self.claim_liabilities = [c for c in self.claim_liabilities if c.remaining_amount > ZERO]

        if total_paid > ZERO:
            logger.info(f"Paid ${total_paid:,.2f} toward claim liabilities")

        # Check solvency after making payments
        if payment_ratio < ONE or self.equity <= ZERO:
            self.check_solvency()

        return total_paid

    def process_insurance_claim_with_development(
        self,
        claim_amount: Union[Decimal, float],
        deductible: Union[Decimal, float] = ZERO,
        insurance_limit: Union[Decimal, float, None] = None,
        development_pattern: Optional["ClaimDevelopment"] = None,
        claim_type: str = "general_liability",
    ) -> tuple[Decimal, Decimal, Optional["Claim"]]:
        """Process an insurance claim with custom development pattern integration.

        This enhanced method extends basic claim processing to support custom
        claim development patterns from the claim_development module. It provides
        more sophisticated actuarial modeling while maintaining compatibility
        with the basic claim processing workflow.

        The method first processes the immediate financial impact using standard
        logic, then optionally creates a detailed Claim object with custom
        development patterns for advanced cash flow modeling.

        Args:
            claim_amount: Total amount of the loss/claim in dollars.
                Must be >= 0.
            deductible: Amount company must pay before insurance coverage
                begins. Defaults to 0.0 for full coverage.
            insurance_limit: Maximum amount insurance will pay per claim.
                Defaults to unlimited coverage.
            development_pattern: Custom actuarial
                development pattern for the claim. If None, uses default
                ClaimLiability schedule. Enables sophisticated reserving and
                payment modeling.
            claim_type: Classification of claim type for actuarial analysis.
                Common types: "general_liability", "product_liability",
                "workers_compensation", "property". Defaults to "general_liability".

        Returns:
            tuple[Decimal, Decimal, Optional[Claim]]: Three-element tuple containing:
                - company_payment: Amount paid immediately by company
                - insurance_payment: Amount covered by insurance
                - claim_object: Detailed claim tracking object
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
        if development_pattern is not None and insurance_payment > ZERO:
            # Import here to avoid circular dependency
            from .claim_development import Claim

            claim_object = Claim(
                claim_id=f"CL_{self.current_year}_{len(self.claim_liabilities):04d}",
                accident_year=self.current_year,
                reported_year=self.current_year,
                initial_estimate=float(insurance_payment),  # Claim class expects float
                claim_type=claim_type,
                development_pattern=development_pattern,
            )

            # The claim is already added to liabilities in process_insurance_claim
            # but we can enhance the tracking with the Claim object
            logger.info(
                f"Created claim with {development_pattern.pattern_name} development pattern"
            )

        return company_payment, insurance_payment, claim_object

    def handle_insolvency(self) -> None:
        """Handle insolvency by enforcing zero equity floor and freezing operations.

        Implements standard bankruptcy accounting with limited liability:
        - Sets equity floor at $0 (never negative)
        - Marks company as insolvent (is_ruined = True)
        - Keeps unpayable liabilities on the books
        - Freezes all further business operations

        This method enforces the limited liability principle that shareholders
        cannot lose more than their equity investment. When equity reaches zero,
        the company enters insolvency and creditors cannot claim beyond available
        assets.

        Side Effects:
            - Sets :attr:`is_ruined` to True
            - May adjust equity to exactly 0 if slightly negative due to rounding
            - Logs insolvency event with current financial state
            - Company remains insolvent for remainder of simulation

        Note:
            Under limited liability and standard bankruptcy accounting:
            - Equity is floored at $0 (Assets - Liabilities may be negative)
            - Unpayable liabilities remain on books until bankruptcy proceedings
            - No further operations or payments after insolvency
            - This is an absorbing state in the simulation

        See Also:
            :meth:`check_solvency`: Detects insolvency and calls this method.
            :attr:`is_ruined`: Insolvency status flag.
        """
        # Ensure equity never goes below zero (limited liability)
        if self.equity < ZERO:
            # Small negative equity due to rounding - adjust to exactly zero
            # Record via ledger to maintain consistency (Issue #275)
            adjustment = -self.equity
            self._record_cash_adjustment(
                amount=adjustment,
                description="Limited liability floor adjustment - equity to zero",
                transaction_type=TransactionType.ADJUSTMENT,
            )
            logger.info(
                f"Adjusted equity from ${self.equity - adjustment:,.2f} to $0 "
                f"(limited liability floor)"
            )

        # Mark as insolvent
        if not self.is_ruined:
            self.is_ruined = True
            total_liabilities = self.total_claim_liabilities
            pre_liquidation_equity = self.equity
            pre_liquidation_assets = self.total_assets

            logger.warning(
                f"INSOLVENCY: Company is now insolvent. "
                f"Equity: ${self.equity:,.2f}, "
                f"Assets: ${self.total_assets:,.2f}, "
                f"Liabilities: ${total_liabilities:,.2f}, "
                f"Unpayable debt: ${max(0, total_liabilities - self.total_assets):,.2f}"
            )

            # Apply bankruptcy/liquidation: Company's assets are liquidated
            # In bankruptcy, assets are sold at a discount and costs are incurred
            # Set remaining value to insolvency_tolerance (near-zero value after liquidation)
            insolvency_tolerance = to_decimal(self.config.insolvency_tolerance)
            if self.equity > insolvency_tolerance:
                # Reduce cash to represent liquidation costs and asset haircuts
                # Record via ledger to maintain consistency (Issue #275)
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
        # LIMITED LIABILITY ENFORCEMENT: Cash should never be negative
        if self.cash < ZERO:
            logger.warning(
                f"Cash is negative (${self.cash:,.2f}). Adjusting to $0 to enforce limited liability."
            )
            # Record via ledger to maintain consistency (Issue #275)
            # Bring cash back to zero by recording the negative amount as an adjustment
            self._record_cash_adjustment(
                amount=-self.cash,  # Negative cash -> positive adjustment
                description="Limited liability enforcement - cash floor at zero",
                transaction_type=TransactionType.ADJUSTMENT,
            )

        # Traditional balance sheet insolvency
        # Handle any case where equity <= 0, including negative equity from operations
        # The handle_insolvency() method will adjust cash to enforce equity floor at $0
        if self.equity <= ZERO:
            # Call handle_insolvency to enforce limited liability and freeze operations
            self.handle_insolvency()
            return False

        # Payment insolvency - check if claim payment obligations are unsustainable
        if self.claim_liabilities:
            # Calculate scheduled payments for the current year
            current_year_payments: Decimal = ZERO
            for claim_item in self.claim_liabilities:
                years_since = self.current_year - claim_item.year_incurred
                scheduled_payment = claim_item.get_payment(years_since)
                current_year_payments += scheduled_payment

            # Check if payments are sustainable relative to revenue capacity
            if current_year_payments > ZERO:
                current_revenue = self.calculate_revenue()
                payment_burden_ratio = (
                    current_year_payments / current_revenue
                    if current_revenue > ZERO
                    else to_decimal(float("inf"))
                )

                # Company is insolvent if claim payments exceed 80% of revenue
                # This threshold represents realistic maximum debt service capacity
                if payment_burden_ratio > to_decimal(0.80):
                    if not self.is_ruined:  # Only log once
                        self.is_ruined = True
                        logger.warning(
                            f"Company became insolvent - unsustainable payment burden: "
                            f"${current_year_payments:,.0f} payments vs ${current_revenue:,.0f} revenue "
                            f"({payment_burden_ratio:.1%} burden ratio)"
                        )
                    return False

        return True

    def calculate_metrics(
        self,
        period_revenue: Optional[Union[Decimal, float]] = None,
        letter_of_credit_rate: Union[Decimal, float] = 0.015,
    ) -> Dict[str, Union[Decimal, float, int, bool]]:
        """Calculate comprehensive financial metrics for analysis.

        This method computes a complete set of financial metrics including
        balance sheet items, income statement components, and financial ratios.
        It provides a standardized view of the company's financial performance
        and position for simulation analysis.

        Args:
            period_revenue (Optional[float]): Actual revenue earned during the period.
                If None, calculates based on current assets.
            letter_of_credit_rate (float): Annual interest rate for letter of credit
                collateral. Defaults to 0.015 (1.5%).

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
        metrics: Dict[str, Union[Decimal, float, int, bool]] = {}

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

        # Get detailed accrual breakdown from AccrualManager (single source of truth - issue #238)
        accrual_items = self.accrual_manager.get_balance_sheet_items()
        metrics["accrued_expenses"] = accrual_items.get("accrued_expenses", ZERO)

        metrics["gross_ppe"] = self.gross_ppe
        metrics["accumulated_depreciation"] = self.accumulated_depreciation
        metrics["net_ppe"] = self.net_ppe

        # Add detailed accrual breakdown
        metrics["accrued_wages"] = accrual_items.get("accrued_wages", ZERO)
        metrics["accrued_taxes"] = accrual_items.get("accrued_taxes", ZERO)
        metrics["accrued_interest"] = accrual_items.get("accrued_interest", ZERO)
        metrics["accrued_revenues"] = accrual_items.get("accrued_revenues", ZERO)

        # Calculate operating metrics for current state
        # Use period revenue if provided (actual revenue earned during the period)
        # Otherwise calculate based on current assets (for standalone metrics)
        revenue = (
            to_decimal(period_revenue) if period_revenue is not None else self.calculate_revenue()
        )
        # Calculate depreciation for metrics (annual basis)
        annual_depreciation = self.gross_ppe / to_decimal(10) if self.gross_ppe > ZERO else ZERO
        operating_income = self.calculate_operating_income(revenue, annual_depreciation)
        collateral_costs = self.calculate_collateral_costs(letter_of_credit_rate, "annual")
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

        # Track actual dividends paid (from update_balance_sheet)
        # Issue #239: Use actual dividends paid (considering cash constraints)
        # instead of theoretical calculation based on retention ratio
        metrics["dividends_paid"] = self._last_dividends_paid

        # Track depreciation expense for cash flow statement
        metrics["depreciation_expense"] = annual_depreciation

        # Issue #255: Calculate COGS and SG&A breakdown explicitly
        # This removes hardcoded ratios from the reporting layer and puts
        # the business logic where it belongs - in the Manufacturer model.
        expense_ratios = getattr(self.config, "expense_ratios", None)

        # Get expense ratios from config or use defaults
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
            # Default ratios (matching former hardcoded values in financial_statements.py)
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
        cogs_ratio = ONE - gross_margin_ratio
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

        # Store expense ratios for reporting reference
        metrics["gross_margin_ratio"] = gross_margin_ratio
        metrics["sga_expense_ratio"] = sga_expense_ratio

        # Financial ratios - ROE now includes all expenses
        metrics["asset_turnover"] = (
            revenue / self.total_assets if self.total_assets > ZERO else ZERO
        )

        # Report both base and actual operating margins for transparency
        base_margin = to_decimal(self.base_operating_margin)
        actual_margin = operating_income / revenue if revenue > ZERO else ZERO
        metrics["base_operating_margin"] = base_margin
        metrics["actual_operating_margin"] = actual_margin
        metrics["insurance_impact_on_margin"] = base_margin - actual_margin

        # Define minimum equity threshold for meaningful ROE calculation
        # When equity is <= 100, ROE becomes meaningless (approaches infinity)
        MIN_EQUITY_THRESHOLD = to_decimal(100)

        metrics["roe"] = (
            net_income / self.equity if self.equity > MIN_EQUITY_THRESHOLD else ZERO
        )  # Return 0 for very small or negative equity to avoid extreme values
        metrics["roa"] = net_income / self.total_assets if self.total_assets > ZERO else ZERO

        # Leverage metrics (collateral-based instead of debt)
        metrics["collateral_to_equity"] = (
            self.collateral / self.equity if self.equity > ZERO else ZERO
        )
        metrics["collateral_to_assets"] = (
            self.collateral / self.total_assets if self.total_assets > ZERO else ZERO
        )

        return metrics

    def _handle_insolvent_step(
        self, time_resolution: str
    ) -> Dict[str, Union[Decimal, float, int, bool]]:
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

    def process_accrued_payments(
        self, time_resolution: str = "annual", max_payable: Optional[Union[Decimal, float]] = None
    ) -> Decimal:
        """Process due accrual payments for the current period.

        Checks for accrual payments due in the current period and processes
        them, reducing cash and clearing the accruals. This method supports
        quarterly tax payments and other scheduled accrual payments.

        Args:
            time_resolution: "annual" or "monthly" for determining current period
            max_payable: Optional maximum amount that can be paid (for coordinated
                limited liability enforcement). If None, caps at current equity.

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

        # LIMITED LIABILITY: Cap TOTAL payments at available equity or provided max
        total_due = sum((to_decimal(v) for v in payments_due.values()), ZERO)
        if max_payable is not None:
            # Use coordinated cap from step() method
            max_total_payable: Decimal = min(total_due, to_decimal(max_payable))
        else:
            # Fallback to equity-based cap if called standalone
            current_equity = self.equity
            max_total_payable = min(total_due, current_equity) if current_equity > ZERO else ZERO

        if total_due > max_total_payable:
            logger.warning(
                f"LIMITED LIABILITY: Capping total accrued payments. "
                f"Due: ${total_due:,.2f}, Payable: ${max_total_payable:,.2f}"
            )

        # Calculate payment ratio to proportionally reduce each accrual
        payment_ratio: Decimal = max_total_payable / total_due if total_due > ZERO else ZERO

        total_paid: Decimal = ZERO
        for accrual_type, amount_due in payments_due.items():
            # Apply payment ratio to this accrual
            payable_amount: Decimal = to_decimal(amount_due) * payment_ratio
            unpayable_amount = to_decimal(amount_due) - payable_amount

            if payable_amount > ZERO:
                # Process the payment (proportional share of what we can afford)
                self.accrual_manager.process_payment(accrual_type, float(payable_amount), period)

                # Record accrued payment via ledger only (Issue #275)
                # The ledger is the single source of truth for cash
                total_paid += payable_amount

                # Determine transaction type and account based on accrual type
                if accrual_type == AccrualType.TAXES:
                    trans_type = TransactionType.TAX_PAYMENT
                    debit_account = AccountName.ACCRUED_TAXES
                elif accrual_type == AccrualType.WAGES:
                    trans_type = TransactionType.PAYMENT
                    debit_account = AccountName.ACCRUED_WAGES
                elif accrual_type == AccrualType.INTEREST:
                    trans_type = TransactionType.PAYMENT
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

            # LIMITED LIABILITY: Discharge unpayable accrued expenses from liabilities
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

    def record_claim_accrual(
        self,
        claim_amount: Union[Decimal, float],
        development_pattern: Optional[ClaimDevelopment] = None,
    ) -> None:
        """Record insurance claim with multi-year payment schedule.

        This method creates a ClaimLiability object which is the single source of
        truth for claim lifecycle tracking. The claim will be paid via
        pay_claim_liabilities() according to the development pattern.

        Note:
            ClaimLiability is the authoritative source for claim tracking.
            This method no longer creates AccrualManager entries to prevent
            state drift between the two tracking systems. See GitHub issue #213.

            For claims with insurance coverage, use process_insurance_claim()
            or process_uninsured_claim() instead, which handle collateral
            and insurance accounting properly.

        Args:
            claim_amount: Total claim amount to be paid
            development_pattern: Optional ClaimDevelopment strategy for payment timing.
                Defaults to ClaimDevelopment.create_long_tail_10yr() if None.

        Examples:
            Record claim with default pattern::

                manufacturer.record_claim_accrual(1_000_000)

            Record claim with immediate payment::

                manufacturer.record_claim_accrual(
                    500_000,
                    development_pattern=ClaimDevelopment.create_immediate()
                )

            Record claim with custom pattern::

                custom = ClaimDevelopment(
                    pattern_name="CUSTOM",
                    development_factors=[0.4, 0.3, 0.2, 0.1]
                )
                manufacturer.record_claim_accrual(750_000, development_pattern=custom)
        """
        amount = to_decimal(claim_amount)
        # Create a ClaimLiability as the single source of truth
        # Use year_incurred - 1 if current_year > 0 to match timing convention
        year_incurred = self.current_year - 1 if self.current_year > 0 else self.current_year

        # Create claim with custom pattern if provided, otherwise use ClaimLiability default
        if development_pattern is not None:
            new_claim = ClaimLiability(
                original_amount=amount,
                remaining_amount=amount,
                year_incurred=year_incurred,
                is_insured=False,  # Standalone accrual without insurance coverage
                development_strategy=development_pattern,
            )
        else:
            new_claim = ClaimLiability(
                original_amount=amount,
                remaining_amount=amount,
                year_incurred=year_incurred,
                is_insured=False,  # Standalone accrual without insurance coverage
                # Uses ClaimLiability default development_strategy
            )
        self.claim_liabilities.append(new_claim)

        pattern_name = development_pattern.pattern_name if development_pattern else "default"
        logger.info(
            f"Created claim liability via record_claim_accrual: ${amount:,.2f} "
            f"with pattern {pattern_name}"
        )

    def _apply_growth(
        self, growth_rate: Union[Decimal, float], time_resolution: str, apply_stochastic: bool
    ) -> None:
        """Apply revenue growth by adjusting asset turnover ratio.

        Args:
            growth_rate: Revenue growth rate for the period.
            time_resolution: "annual" or "monthly" for simulation step.
            apply_stochastic: Whether to apply stochastic shocks.
        """
        rate = to_decimal(growth_rate) if not isinstance(growth_rate, Decimal) else growth_rate
        if rate == ZERO or not (time_resolution == "annual" or self.current_month == 11):
            return

        base_growth = float(ONE + rate)  # Use float for compatibility with stochastic process

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
        letter_of_credit_rate: Union[Decimal, float] = 0.015,
        growth_rate: Union[Decimal, float] = 0.0,
        time_resolution: str = "annual",
        apply_stochastic: bool = False,
    ) -> Dict[str, Union[Decimal, float, int, bool]]:
        """Execute one time step of the financial model simulation.

        This is the main simulation method that advances the manufacturer's
        financial state by one time period. It processes all business operations
        including revenue generation, expense payment, claim liability payments,
        growth application, and solvency checking.

        The method supports both annual and monthly time resolution for flexible
        modeling. Monthly resolution provides more granular cash flow tracking
        but requires careful scaling of annual parameters.

        Working capital components (AR, Inventory, AP) are automatically calculated
        each step based on revenue using standard DSO/DIO/DPO ratios. These
        components affect the balance sheet and cash flow statement (Issue #244).

        Args:
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

                metrics = manufacturer.step(letter_of_credit_rate=0.015)

                print(f"ROE: {metrics['roe']:.1%}")
                print(f"Assets: ${metrics['assets']:,.2f}")

            Monthly simulation with growth::

                # Monthly steps with 5% annual growth
                for month in range(12):
                    metrics = manufacturer.step(
                        growth_rate=0.05,
                        time_resolution="monthly"
                    )

                    if month == 11:  # Growth applied in December
                        print(f"Growth applied: {metrics['asset_turnover']}")

            Stochastic simulation::

                # With revenue volatility
                metrics = manufacturer.step(apply_stochastic=True)

                # Revenue will vary based on stochastic process

        Side Effects:
            - Updates current_year and/or current_month
            - Modifies balance sheet (assets, equity, collateral)
            - Updates working capital components (AR, Inventory, AP)
            - Processes scheduled claim payments
            - May trigger insolvency if losses exceed equity
            - Appends metrics to metrics_history
            - Applies growth by modifying asset_turnover_ratio

        Raises:
            ValueError: If time_resolution not "annual" or "monthly".
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
            :meth:`calculate_working_capital_components`: Working capital calculation.
            :meth:`pay_claim_liabilities`: Claim payment processing.
            :meth:`check_solvency`: Solvency evaluation.
            :meth:`calculate_metrics`: Metrics calculation details.
        """
        # Check if already ruined
        if self.is_ruined:
            return self._handle_insolvent_step(time_resolution)

        # Store initial revenue for working capital calculation in monthly mode
        # This must happen BEFORE any balance sheet changes
        if time_resolution == "monthly" and self.current_month == 0:
            # Calculate the annual revenue for working capital consistency
            self._annual_revenue_for_wc = self.calculate_revenue(apply_stochastic)

        # Calculate financial performance
        revenue = self.calculate_revenue(apply_stochastic)

        # Calculate working capital components BEFORE payment coordination
        # Working capital changes can affect AP (liabilities) which changes equity
        # So we need to update working capital BEFORE calculating payment caps
        # Issue #244: Always calculate working capital components (AR, Inventory, AP)
        # to maintain accurate balance sheet. Working capital impact flows through
        # the cash flow statement, not revenue.
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
                annual_revenue = self.total_assets * to_decimal(self.asset_turnover_ratio)
                self.calculate_working_capital_components(annual_revenue)

        # COORDINATED LIMITED LIABILITY ENFORCEMENT
        # Calculate total payments needed for both accruals and claims
        # Then cap at current equity (AFTER working capital adjustments) and allocate proportionally

        # Determine current period for accrual manager
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

        # Calculate total claim payments scheduled (if applicable)
        total_claim_due: Decimal = ZERO
        if time_resolution == "annual" or self.current_month == 0:
            for claim_item in self.claim_liabilities:
                years_since = self.current_year - claim_item.year_incurred
                scheduled_payment = claim_item.get_payment(years_since)
                total_claim_due += scheduled_payment

        # Cap TOTAL payments at available liquid resources (cash + restricted assets for claims)
        total_payments_due = total_accrual_due + total_claim_due
        available_liquidity = self.cash + self.restricted_assets
        max_total_payable: Decimal = (
            min(total_payments_due, available_liquidity) if available_liquidity > ZERO else ZERO
        )

        # Allocate the capped amount proportionally between accruals and claims
        if total_payments_due > ZERO:
            allocation_ratio = max_total_payable / total_payments_due
            max_accrual_payable: Decimal = total_accrual_due * allocation_ratio
            max_claim_payable: Decimal = total_claim_due * allocation_ratio
        else:
            max_accrual_payable = ZERO
            max_claim_payable = ZERO

        # Log coordination if payments are capped
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
            # For monthly, record 1/12 of annual depreciation
            depreciation_expense = self.record_depreciation(useful_life_years=10 * 12)
        else:
            depreciation_expense = ZERO

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

        # Update balance sheet with retained earnings
        # Working capital was already calculated earlier (before payment coordination)
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
        # Pass the actual period revenue and LoC rate to get accurate metrics
        metrics = self.calculate_metrics(
            period_revenue=revenue, letter_of_credit_rate=letter_of_credit_rate
        )
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
        # Reset operating parameters
        self.asset_turnover_ratio = self.config.asset_turnover_ratio
        self.claim_liabilities = []
        self.current_year = 0
        self.current_month = 0
        self.is_ruined = False
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

        # Compute initial balance sheet values for reset
        # PP&E allocation depends on operating margin
        if self.config.base_operating_margin < 0.10:
            ppe_ratio = to_decimal(
                0.3
            )  # Low margin businesses need more working capital, less PP&E
        elif self.config.base_operating_margin < 0.15:
            ppe_ratio = to_decimal(0.5)  # Medium margin can support moderate PP&E
        else:
            ppe_ratio = to_decimal(0.7)  # High margin businesses can support more PP&E

        initial_gross_ppe: Decimal = initial_assets * ppe_ratio
        initial_accumulated_depreciation: Decimal = ZERO
        initial_cash: Decimal = initial_assets * (ONE - ppe_ratio)
        initial_accounts_receivable: Decimal = ZERO
        initial_inventory: Decimal = ZERO
        initial_prepaid_insurance: Decimal = ZERO
        initial_accounts_payable: Decimal = ZERO
        initial_collateral: Decimal = ZERO
        initial_restricted_assets: Decimal = ZERO

        # Reset ledger FIRST (single source of truth for all balance sheet accounts)
        # Then record initial balances - this is the only way to set balance sheet values
        # Direct assignment is not possible since properties read from ledger (Issue #275)
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
            stochastic_process=(
                copy.deepcopy(self.stochastic_process) if self.stochastic_process else None
            ),
        )

        # The new manufacturer starts fresh at initial state
        # No need to copy current state since we want independent simulations

        logger.debug("Created copy of manufacturer")
        return new_manufacturer

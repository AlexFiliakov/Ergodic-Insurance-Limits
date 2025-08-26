"""Widget manufacturer financial model implementation.

This module implements the core financial model for a widget manufacturing
company, including balance sheet management, insurance claim processing,
and stochastic modeling capabilities.

The manufacturer model simulates:
    - Revenue generation based on assets and turnover ratios
    - Operating income with configurable margins
    - Insurance claim processing with deductibles and limits
    - Letter of credit collateral management
    - Multi-year claim payment schedules
    - Balance sheet evolution over time
    - Integration with stochastic processes for uncertainty

Example:
    Basic manufacturer setup and simulation::

        from ergodic_insurance.src.config import ManufacturerConfig
        from ergodic_insurance.src.manufacturer import WidgetManufacturer

        # Configure manufacturer
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            operating_margin=0.08,
            tax_rate=0.25
        )

        # Create manufacturer
        manufacturer = WidgetManufacturer(config)

        # Process a claim
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=5_000_000,
            deductible=1_000_000,
            insurance_limit=10_000_000
        )

        # Run annual step
        metrics = manufacturer.step(working_capital_pct=0.2)

Note:
    This module is the core of the financial simulation and integrates with
    claim generators, stochastic processes, and insurance programs.
"""

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from .config import ManufacturerConfig
from .stochastic_processes import StochasticProcess

# Optional import for claim development integration
if TYPE_CHECKING:
    from .claim_development import Claim, ClaimDevelopment

logger = logging.getLogger(__name__)


@dataclass
class ClaimLiability:
    """Represents an outstanding insurance claim liability.

    This class tracks insurance claims that require multi-year payment
    schedules and manages the collateral required to support them.
    """

    original_amount: float
    remaining_amount: float
    year_incurred: int
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

        Args:
            years_since_incurred: Number of years since the claim was first incurred.

        Returns:
            Payment amount due for this year as a percentage of original amount.
        """
        if years_since_incurred < 0 or years_since_incurred >= len(self.payment_schedule):
            return 0.0
        return self.original_amount * self.payment_schedule[years_since_incurred]

    def make_payment(self, amount: float) -> float:
        """Make a payment against the liability.

        Args:
            amount: Requested payment amount.

        Returns:
            Actual payment made (may be less than requested if insufficient liability remains).
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

        Args:
            config: Manufacturing configuration parameters.
            stochastic_process: Optional stochastic process for revenue volatility.
        """
        self.config = config
        self.stochastic_process = stochastic_process

        # Balance sheet items
        self.assets = config.initial_assets
        self.collateral = 0.0  # Letter of credit collateral for claims
        self.restricted_assets = 0.0  # Assets restricted as collateral
        self.equity = self.assets  # No debt, so equity = assets initially

        # Operating parameters
        self.asset_turnover_ratio = config.asset_turnover_ratio
        self.operating_margin = config.operating_margin
        self.tax_rate = config.tax_rate
        self.retention_ratio = config.retention_ratio

        # Claim tracking
        self.claim_liabilities: List[ClaimLiability] = []
        self.current_year = 0
        self.current_month = 0  # Track months for monthly LoC payments

        # Solvency tracking
        self.is_ruined = False

        # Metrics tracking
        self.metrics_history: List[Dict[str, float]] = []

    @property
    def net_assets(self) -> float:
        """Calculate net assets (total assets minus restricted assets).

        Returns:
            Net assets available to the company.
        """
        return self.assets - self.restricted_assets

    @property
    def available_assets(self) -> float:
        """Calculate available (unrestricted) assets for operations.

        Returns:
            Assets available for operational use (not restricted as collateral).
        """
        return self.assets - self.restricted_assets

    @property
    def total_claim_liabilities(self) -> float:
        """Calculate total outstanding claim liabilities.

        Returns:
            Sum of all remaining claim liability amounts.
        """
        return sum(claim.remaining_amount for claim in self.claim_liabilities)

    def calculate_revenue(
        self, working_capital_pct: float = 0.0, apply_stochastic: bool = False
    ) -> float:
        """Calculate revenue based on available assets.

        Args:
            working_capital_pct: Percentage of revenue tied up in working capital.
            apply_stochastic: Whether to apply stochastic shock to revenue.

        Returns:
            Annual revenue.
        """
        # Adjust for working capital if specified
        available_assets = self.assets
        if working_capital_pct > 0:
            # Working capital reduces assets available for operations
            # Revenue = Available Assets * Turnover, where
            # Available Assets = Total Assets - Working Capital
            # Working Capital = Revenue * working_capital_pct
            # Solving: Revenue = Assets * Turnover / (1 + Turnover * WC%)
            denominator = 1 + self.asset_turnover_ratio * working_capital_pct
            available_assets = self.assets / denominator

        revenue = available_assets * self.asset_turnover_ratio

        # Apply stochastic shock if requested and process is available
        if apply_stochastic and self.stochastic_process is not None:
            shock = self.stochastic_process.generate_shock(revenue)
            revenue *= shock
            logger.debug(f"Applied stochastic shock: {shock:.4f}")

        logger.debug(f"Revenue calculated: ${revenue:,.2f} from assets ${self.assets:,.2f}")
        return revenue

    def calculate_operating_income(self, revenue: float) -> float:
        """Calculate operating income from revenue.

        Args:
            revenue: Annual revenue.

        Returns:
            Operating income before interest and taxes.
        """
        operating_income = revenue * self.operating_margin
        logger.debug(
            f"Operating income: ${operating_income:,.2f} ({self.operating_margin:.1%} margin)"
        )
        return operating_income

    def calculate_collateral_costs(
        self, letter_of_credit_rate: float = 0.015, time_period: str = "annual"
    ) -> float:
        """Calculate costs for letter of credit collateral.

        Args:
            letter_of_credit_rate: Annual rate for letter of credit (default 1.5%).
            time_period: "annual" or "monthly" for cost calculation.

        Returns:
            Collateral costs for the period.
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

    def calculate_net_income(self, operating_income: float, collateral_costs: float) -> float:
        """Calculate net income after collateral costs and taxes.

        Args:
            operating_income: Operating income before collateral and taxes.
            collateral_costs: Annual collateral costs.

        Returns:
            Net income after taxes.
        """
        # Deduct collateral costs (like interest expense)
        income_before_tax = operating_income - collateral_costs

        # Calculate taxes (only on positive income)
        taxes = max(0, income_before_tax * self.tax_rate)

        net_income = income_before_tax - taxes
        logger.debug(f"Net income: ${net_income:,.2f} after ${taxes:,.2f} taxes")
        return net_income

    def update_balance_sheet(self, net_income: float, growth_rate: float = 0.0) -> None:
        """Update balance sheet with retained earnings and growth.

        Args:
            net_income: Net income for the period.
            growth_rate: Revenue growth rate to apply.
        """
        # Calculate retained earnings
        retained_earnings = net_income * self.retention_ratio
        dividends = net_income * (1 - self.retention_ratio)

        # Add retained earnings to assets
        self.assets += retained_earnings

        # Update equity (no debt, so equity changes by retained earnings)
        self.equity += retained_earnings

        logger.debug(
            f"Balance sheet updated: Assets=${self.assets:,.2f}, Equity=${self.equity:,.2f}"
        )
        if dividends > 0:
            logger.debug(f"Dividends paid: ${dividends:,.2f}")

    def process_insurance_claim(
        self, claim_amount: float, deductible: float = 0.0, insurance_limit: float = float("inf")
    ) -> tuple[float, float]:
        """Process an insurance claim with deductible and limit, setting up collateral.

        Args:
            claim_amount: Total amount of the loss/claim.
            deductible: Amount company must pay before insurance kicks in.
            insurance_limit: Maximum amount insurance will pay.

        Returns:
            Tuple of (company_payment, insurance_payment).
        """
        # Calculate insurance coverage
        if claim_amount <= deductible:
            # Below deductible, company pays all
            company_payment = claim_amount
            insurance_payment = 0
        else:
            # Above deductible
            company_payment = deductible
            insurance_payment = int(min(claim_amount - deductible, insurance_limit))
            # Company also pays any amount above the limit
            if claim_amount > deductible + insurance_limit:
                company_payment += claim_amount - deductible - insurance_limit

        # Company must immediately pay its portion
        if company_payment > 0:
            actual_payment = min(company_payment, self.assets)  # Pay what we can
            self.assets -= actual_payment
            self.equity -= actual_payment  # Reduce equity by the same amount
            logger.info(f"Company paid ${actual_payment:,.2f} (deductible/excess)")

        # Insurance payment requires collateral and creates liability
        if insurance_payment > 0:
            # Post letter of credit as collateral for insurance payment
            # This restricts assets but doesn't change total assets or equity
            self.collateral += insurance_payment
            self.restricted_assets += insurance_payment

            # Create claim liability with payment schedule for insurance portion
            claim = ClaimLiability(
                original_amount=insurance_payment,
                remaining_amount=insurance_payment,
                year_incurred=self.current_year,
            )
            self.claim_liabilities.append(claim)

            logger.info(f"Insurance covering ${insurance_payment:,.2f}")
            logger.info(f"Posted ${insurance_payment:,.2f} letter of credit as collateral")

        logger.info(
            f"Total claim: ${claim_amount:,.2f} (Company: ${company_payment:,.2f}, Insurance: ${insurance_payment:,.2f})"
        )

        return company_payment, insurance_payment

    def pay_claim_liabilities(self) -> float:
        """Pay scheduled claim liabilities for the current year.

        Returns:
            Total amount paid toward claims.
        """
        total_paid = 0.0

        for claim in self.claim_liabilities:
            years_since = self.current_year - claim.year_incurred
            scheduled_payment = claim.get_payment(years_since)

            if scheduled_payment > 0:
                # Pay from available assets
                available_for_payment = max(0, self.assets - 100_000)  # Keep minimum cash
                actual_payment = min(scheduled_payment, available_for_payment)

                if actual_payment > 0:
                    claim.make_payment(actual_payment)
                    self.assets -= actual_payment
                    self.equity -= actual_payment  # Reduce equity when paying claims
                    total_paid += actual_payment

                    # Reduce collateral and restricted assets by payment amount
                    self.collateral -= actual_payment
                    self.restricted_assets -= actual_payment
                    logger.debug(
                        f"Reduced collateral and restricted assets by ${actual_payment:,.2f}"
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
        """Process an insurance claim with custom development pattern.

        This enhanced method allows specifying a custom claim development
        pattern for more realistic cash flow modeling. If no pattern is
        provided, it falls back to the default behavior.

        Args:
            claim_amount: Total amount of the loss/claim.
            deductible: Amount company must pay before insurance kicks in.
            insurance_limit: Maximum amount insurance will pay.
            development_pattern: Optional custom development pattern for the claim.
            claim_type: Type of claim for categorization.

        Returns:
            Tuple of (company_payment, insurance_payment, claim_object).
            The claim_object is None if no development pattern is used.
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
        """Check if the company is solvent (equity > 0).

        Returns:
            True if solvent, False if ruined.
        """
        if self.equity <= 0:
            self.is_ruined = True
            logger.warning(f"Company became insolvent with equity: ${self.equity:,.2f}")
        return not self.is_ruined

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate key financial metrics.

        Returns:
            Dictionary of financial metrics.
        """
        metrics = {}

        # Basic balance sheet metrics
        metrics["assets"] = self.assets
        metrics["collateral"] = self.collateral
        metrics["restricted_assets"] = self.restricted_assets
        metrics["available_assets"] = self.available_assets
        metrics["equity"] = self.equity
        metrics["net_assets"] = self.net_assets
        metrics["claim_liabilities"] = self.total_claim_liabilities
        metrics["is_solvent"] = not self.is_ruined

        # Calculate operating metrics for current state
        revenue = self.calculate_revenue()
        operating_income = self.calculate_operating_income(revenue)
        collateral_costs = self.calculate_collateral_costs()
        net_income = self.calculate_net_income(operating_income, collateral_costs)

        metrics["revenue"] = revenue
        metrics["operating_income"] = operating_income
        metrics["net_income"] = net_income

        # Financial ratios
        metrics["asset_turnover"] = revenue / self.assets if self.assets > 0 else 0
        metrics["operating_margin"] = self.operating_margin
        metrics["roe"] = net_income / self.equity if self.equity > 0 else 0
        metrics["roa"] = net_income / self.assets if self.assets > 0 else 0

        # Leverage metrics (collateral-based instead of debt)
        metrics["collateral_to_equity"] = self.collateral / self.equity if self.equity > 0 else 0
        metrics["collateral_to_assets"] = self.collateral / self.assets if self.assets > 0 else 0

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
        """Execute one time step of the financial model.

        Args:
            working_capital_pct: Working capital as percentage of sales.
            letter_of_credit_rate: Annual rate for letter of credit.
            growth_rate: Revenue growth rate for the period.
            time_resolution: "annual" or "monthly" for simulation step.
            apply_stochastic: Whether to apply stochastic shocks.

        Returns:
            Dictionary of metrics for this time step.
        """
        # Check if already ruined
        if self.is_ruined:
            return self._handle_insolvent_step(time_resolution)

        # Pay scheduled claim liabilities first (annual payments)
        if time_resolution == "annual" or self.current_month == 0:
            self.pay_claim_liabilities()

        # Calculate financial performance
        revenue = self.calculate_revenue(working_capital_pct, apply_stochastic)
        operating_income = self.calculate_operating_income(revenue)

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

        net_income = self.calculate_net_income(operating_income, collateral_costs)

        # Update balance sheet with retained earnings
        self.update_balance_sheet(net_income, growth_rate)

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

        return metrics

    def reset(self) -> None:
        """Reset the manufacturer to initial state.

        Restores all financial parameters to their initial values and
        clears historical data.
        """
        self.assets = self.config.initial_assets
        self.collateral = 0.0
        self.restricted_assets = 0.0
        self.equity = self.assets
        self.asset_turnover_ratio = self.config.asset_turnover_ratio
        self.claim_liabilities = []
        self.current_year = 0
        self.current_month = 0
        self.is_ruined = False
        self.metrics_history = []

        # Reset stochastic process if present
        if self.stochastic_process is not None:
            self.stochastic_process.reset()

        logger.info("Manufacturer reset to initial state")

    def copy(self) -> "WidgetManufacturer":
        """Create a deep copy of the manufacturer.

        Returns:
            A new WidgetManufacturer instance with the same configuration
            but reset to initial state. This is useful for running parallel
            simulations from the same starting point.
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

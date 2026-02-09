"""Simulation execution and financial parameter configuration.

Contains configuration classes that control how simulations run and how
financial dynamics evolve: time resolution, growth models, debt policies,
and working capital management.

Since:
    Version 0.9.0 (Issue #458)
"""

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class WorkingCapitalConfig(BaseModel):
    """Working capital management parameters.

    This class configures how working capital requirements are calculated
    as a percentage of sales revenue. Working capital represents the funds
    tied up in day-to-day operations (inventory, receivables, etc.).

    Attributes:
        percent_of_sales: Working capital as percentage of sales. Typically
            15-25% for manufacturers depending on payment terms and inventory
            turnover.

    Examples:
        Efficient working capital::

            wc_config = WorkingCapitalConfig(
                percent_of_sales=0.15  # 15% - lean operations
            )

        Conservative working capital::

            wc_config = WorkingCapitalConfig(
                percent_of_sales=0.30  # 30% - higher inventory/receivables
            )

    Note:
        Higher working capital requirements reduce available cash for
        growth investments but provide operational cushion.
    """

    percent_of_sales: float = Field(
        default=0.20, ge=0, le=1, description="Working capital as percentage of sales"
    )

    @field_validator("percent_of_sales")
    @classmethod
    def validate_working_capital(cls, v: float) -> float:
        """Validate working capital percentage.

        Args:
            v: Working capital percentage to validate (as decimal).

        Returns:
            float: The validated working capital percentage.

        Raises:
            ValueError: If working capital percentage exceeds 50% of sales,
                which would indicate severe operational inefficiency.
        """
        if v > 0.5:
            raise ValueError(f"Working capital {v:.1%} of sales is unrealistically high")
        return v


class GrowthConfig(BaseModel):
    """Growth model parameters.

    Configures whether the simulation uses deterministic or stochastic
    growth models, along with the associated parameters. Stochastic models
    add realistic business volatility to growth trajectories.

    Attributes:
        type: Growth model type - 'deterministic' for fixed growth or
            'stochastic' for random variation.
        annual_growth_rate: Base annual growth rate (e.g., 0.05 for 5%).
            Can be negative for declining businesses.
        volatility: Growth rate volatility (standard deviation) for stochastic
            models. Zero for deterministic models.

    Examples:
        Stable growth::

            growth = GrowthConfig(
                type='deterministic',
                annual_growth_rate=0.03  # 3% steady growth
            )

        Volatile growth::

            growth = GrowthConfig(
                type='stochastic',
                annual_growth_rate=0.05,  # 5% expected
                volatility=0.15           # 15% std dev
            )

    Note:
        Stochastic growth uses geometric Brownian motion to model
        realistic business volatility patterns.
    """

    type: Literal["deterministic", "stochastic"] = Field(
        default="deterministic", description="Growth model type"
    )
    annual_growth_rate: float = Field(
        default=0.05, ge=-0.5, le=1.0, description="Annual growth rate"
    )
    volatility: float = Field(
        ge=0, le=1, default=0.0, description="Growth rate volatility (std dev)"
    )

    @model_validator(mode="after")
    def validate_stochastic_params(self):
        """Ensure volatility is set for stochastic models.

        Returns:
            GrowthConfig: The validated config object.

        Raises:
            ValueError: If stochastic model is selected but volatility is zero,
                which would make it effectively deterministic.
        """
        if self.type == "stochastic" and self.volatility == 0:
            raise ValueError("Stochastic model requires non-zero volatility")
        return self


class DebtConfig(BaseModel):
    """Debt financing parameters for insurance claims.

    Configures debt financing options and constraints for handling
    large insurance claims and maintaining liquidity. Companies may need
    to borrow to cover deductibles or claims exceeding insurance limits.

    Attributes:
        interest_rate: Annual interest rate on debt (e.g., 0.05 for 5%).
        max_leverage_ratio: Maximum debt-to-equity ratio allowed. Higher
            ratios increase financial risk.
        minimum_cash_balance: Minimum cash balance to maintain for operations.

    Examples:
        Conservative debt policy::

            debt = DebtConfig(
                interest_rate=0.04,        # 4% borrowing cost
                max_leverage_ratio=1.0,    # Max 1:1 debt/equity
                minimum_cash_balance=1_000_000
            )

        Aggressive leverage::

            debt = DebtConfig(
                interest_rate=0.06,        # Higher rate for risk
                max_leverage_ratio=3.0,    # 3:1 leverage allowed
                minimum_cash_balance=500_000
            )

    Note:
        Higher leverage increases return on equity but also increases
        bankruptcy risk during adverse claim events.
    """

    interest_rate: float = Field(
        default=0.05, ge=0, le=0.5, description="Annual interest rate on debt"
    )
    max_leverage_ratio: float = Field(
        default=2.0, ge=0, le=10, description="Maximum debt-to-equity ratio"
    )
    minimum_cash_balance: float = Field(
        default=500_000, ge=0, description="Minimum cash balance to maintain"
    )


class SimulationConfig(BaseModel):
    """Simulation execution parameters.

    Controls how the simulation runs, including time resolution,
    horizon, and randomization settings. These parameters affect
    computational performance and result granularity.

    Attributes:
        time_resolution: Simulation time step - 'annual' or 'monthly'.
            Monthly provides more granularity but increases computation.
        time_horizon_years: Simulation horizon in years. Longer horizons
            reveal ergodic properties but require more computation.
        max_horizon_years: Maximum supported horizon to prevent excessive
            memory usage.
        random_seed: Random seed for reproducibility. None for random.
        fiscal_year_end: Month of fiscal year end (1-12). Default is 12
            (December) for calendar year alignment. Set to 6 for June,
            3 for March, etc. to match different fiscal calendars.

    Examples:
        Quick test simulation::

            sim = SimulationConfig(
                time_resolution='annual',
                time_horizon_years=10,
                random_seed=42  # Reproducible
            )

        Long-term ergodic analysis::

            sim = SimulationConfig(
                time_resolution='annual',
                time_horizon_years=500,
                max_horizon_years=1000,
                random_seed=None  # Random each run
            )

        Non-calendar fiscal year::

            sim = SimulationConfig(
                time_resolution='annual',
                time_horizon_years=50,
                fiscal_year_end=6  # June fiscal year end
            )

    Note:
        For ergodic analysis, horizons of 100+ years are recommended
        to observe long-term time averages.
    """

    time_resolution: Literal["annual", "monthly"] = Field(
        default="annual", description="Simulation time step"
    )
    time_horizon_years: int = Field(
        default=50, gt=0, le=1000, description="Simulation horizon in years"
    )
    max_horizon_years: int = Field(
        default=1000, ge=100, le=10000, description="Maximum supported horizon"
    )
    random_seed: Optional[int] = Field(
        default=None, ge=0, description="Random seed for reproducibility"
    )
    fiscal_year_end: int = Field(
        default=12,
        ge=1,
        le=12,
        description="Month of fiscal year end (1-12). Default is 12 (December) for calendar year.",
    )

    @model_validator(mode="after")
    def validate_horizons(self):
        """Ensure time horizon doesn't exceed maximum.

        Returns:
            SimulationConfig: The validated config object.

        Raises:
            ValueError: If time horizon exceeds maximum allowed value,
                preventing potential memory issues.
        """
        if self.time_horizon_years > self.max_horizon_years:
            raise ValueError(
                f"Time horizon {self.time_horizon_years} exceeds maximum "
                f"{self.max_horizon_years}"
            )
        return self


class WorkingCapitalRatiosConfig(BaseModel):
    """Enhanced working capital configuration with detailed component ratios.

    This extends the basic WorkingCapitalConfig to provide detailed control over
    individual working capital components using standard financial ratios.
    """

    days_sales_outstanding: float = Field(
        default=45,
        ge=0,
        le=365,
        description="Days Sales Outstanding (DSO) - average collection period for receivables",
    )
    days_inventory_outstanding: float = Field(
        default=60,
        ge=0,
        le=365,
        description="Days Inventory Outstanding (DIO) - average days inventory held",
    )
    days_payable_outstanding: float = Field(
        default=30,
        ge=0,
        le=365,
        description="Days Payable Outstanding (DPO) - average payment period for payables",
    )

    @model_validator(mode="after")
    def validate_cash_conversion_cycle(self):
        """Validate that cash conversion cycle is reasonable."""
        ccc = (
            self.days_sales_outstanding
            + self.days_inventory_outstanding
            - self.days_payable_outstanding
        )
        if ccc < 0:
            logger.warning("Negative cash conversion cycle (%.0f days)", ccc)
        elif ccc > 180:
            logger.warning("Very long cash conversion cycle (%.0f days)", ccc)
        return self

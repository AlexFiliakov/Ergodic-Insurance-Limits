"""Configuration management using Pydantic v2 models.

This module provides comprehensive configuration classes for the Ergodic
Insurance simulation framework. It uses Pydantic models for validation,
type safety, and automatic serialization/deserialization of configuration
parameters.

The configuration system is hierarchical, with specialized configs for
different aspects of the simulation (manufacturer, insurance, simulation
parameters, etc.) that can be composed into a master configuration.

Key Features:
    - Type-safe configuration with automatic validation
    - Hierarchical configuration structure
    - Environment variable support
    - JSON/YAML serialization support
    - Default values with business logic constraints
    - Cross-field validation for consistency

Examples:
    Basic configuration setup::

        from ergodic_insurance.config import Config, ManufacturerConfig

        # Create manufacturer config
        manufacturer = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7
        )

        # Create master config
        config = Config(
            manufacturer=manufacturer,
            simulation_years=50
        )

    Loading from file::

        # Load from JSON
        config = Config.from_json('config.json')

        # Load from environment
        config = Config.from_env()

Note:
    All monetary values are in nominal dollars unless otherwise specified.
    Rates and ratios are expressed as decimals (0.1 = 10%).

Since:
    Version 0.1.0
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

# --- Module-level financial constants ---
# Issue #314: Centralized constants to eliminate hardcoded values across modules

DEFAULT_RISK_FREE_RATE: float = 0.02
"""Default risk-free rate (2%) used for Sharpe ratio and risk-adjusted calculations."""


@dataclass
class BusinessOptimizerConfig:
    """Calibration parameters for BusinessOptimizer financial heuristics.

    Issue #314 (C1): Consolidates all hardcoded financial multipliers from
    BusinessOptimizer into a single, documentable configuration object.

    These are simplified model parameters used by the optimizer's heuristic
    methods (_estimate_roe, _estimate_bankruptcy_risk, _estimate_growth_rate,
    etc.). They are NOT derived from manufacturer data—they are tuning knobs
    for the optimizer's internal scoring functions.
    """

    # _estimate_roe parameters
    base_roe: float = 0.15
    """Base return on equity (15%) before insurance adjustments."""
    protection_benefit_factor: float = 0.05
    """Coverage-to-assets ratio multiplier for protection benefit."""
    roe_noise_std: float = 0.1
    """Standard deviation of multiplicative noise applied to ROE."""

    # _estimate_bankruptcy_risk parameters
    base_bankruptcy_risk: float = 0.02
    """Base annual bankruptcy probability (2%)."""
    max_risk_reduction: float = 0.015
    """Maximum risk reduction from insurance coverage (1.5%)."""
    premium_burden_risk_factor: float = 0.5
    """Multiplier converting premium burden ratio to risk increase."""
    time_risk_constant: float = 20.0
    """Time constant (years) for exponential risk accumulation."""

    # _estimate_growth_rate parameters
    base_growth_rate: float = 0.10
    """Base growth rate (10%) before insurance adjustments."""
    growth_boost_factor: float = 0.03
    """Coverage ratio multiplier for growth boost (up to 3%)."""
    premium_drag_factor: float = 0.5
    """Multiplier for premium-to-revenue drag on growth."""
    asset_growth_factor: float = 0.8
    """Growth adjustment factor for asset metric."""
    equity_growth_factor: float = 1.1
    """Growth adjustment factor for equity metric."""

    # _calculate_capital_efficiency parameters
    risk_transfer_benefit_rate: float = 0.05
    """Fraction of coverage limit freed up by risk transfer (5%)."""

    # _estimate_insurance_return parameters
    risk_reduction_value: float = 0.03
    """Return contribution from risk reduction (3%)."""
    stability_value: float = 0.02
    """Return contribution from stability improvement (2%)."""
    growth_enablement_value: float = 0.03
    """Return contribution from growth enablement (3%)."""

    # _calculate_ergodic_growth parameters
    assumed_volatility: float = 0.20
    """Assumed base volatility for ergodic correction."""
    volatility_reduction_factor: float = 0.05
    """Coverage ratio multiplier for volatility reduction."""
    min_volatility: float = 0.05
    """Floor for adjusted volatility."""


@dataclass
class DecisionEngineConfig:
    """Calibration parameters for InsuranceDecisionEngine heuristics.

    Issue #314 (C2): Consolidates hardcoded values from the decision engine's
    growth estimation and simulation methods.
    """

    # _estimate_growth_rate parameters
    base_growth_rate: float = 0.08
    """Base growth rate (8%) for decision engine growth estimation."""
    volatility_reduction_factor: float = 0.3
    """Coverage ratio multiplier for volatility reduction."""
    max_volatility_reduction: float = 0.15
    """Maximum volatility reduction (15%)."""
    growth_benefit_factor: float = 0.5
    """Simplified growth benefit multiplier."""


class ManufacturerConfig(BaseModel):
    """Financial parameters for the widget manufacturer.

    This class defines the core financial parameters used to initialize
    and configure a widget manufacturing company in the simulation. All
    parameters are validated to ensure realistic business constraints.

    Attributes:
        initial_assets: Starting asset value in dollars. Must be positive.
        asset_turnover_ratio: Revenue per dollar of assets. Typically 0.5-2.0
            for manufacturing companies.
        base_operating_margin: Core operating margin before insurance costs
            (EBIT before insurance / Revenue). Typically 5-15% for healthy
            manufacturers.
        tax_rate: Corporate tax rate. Typically 20-30% depending on jurisdiction.
        retention_ratio: Portion of earnings retained vs distributed as dividends.
            Higher retention supports faster growth.
        ppe_ratio: Property, Plant & Equipment allocation ratio as fraction of
            initial assets. Defaults based on operating margin if not specified.

    Examples:
        Conservative manufacturer::

            config = ManufacturerConfig(
                initial_assets=5_000_000,
                asset_turnover_ratio=0.6,  # Low turnover
                base_operating_margin=0.05,      # 5% base margin
                tax_rate=0.25,
                retention_ratio=0.9         # High retention
            )

        Aggressive growth manufacturer::

            config = ManufacturerConfig(
                initial_assets=20_000_000,
                asset_turnover_ratio=1.2,  # High turnover
                base_operating_margin=0.12,      # 12% base margin
                tax_rate=0.25,
                retention_ratio=1.0         # Full retention
            )

        Custom PP&E allocation::

            config = ManufacturerConfig(
                initial_assets=15_000_000,
                asset_turnover_ratio=0.9,
                base_operating_margin=0.10,
                tax_rate=0.25,
                retention_ratio=0.8,
                ppe_ratio=0.6  # Override default PP&E allocation
            )

    Note:
        The asset turnover ratio and base operating margin together determine
        the core return on assets (ROA) before insurance costs and taxes.
        Actual operating margins will be lower when insurance costs are included.
    """

    initial_assets: float = Field(gt=0, description="Starting asset value in dollars")
    asset_turnover_ratio: float = Field(gt=0, le=5, description="Revenue per dollar of assets")
    base_operating_margin: float = Field(
        gt=-1,
        lt=1,
        description="Core operating margin before insurance costs (EBIT before insurance / Revenue)",
    )
    tax_rate: float = Field(ge=0, le=1, description="Corporate tax rate")
    nol_carryforward_enabled: bool = Field(
        default=True,
        description="Enable NOL carryforward tracking per IRC §172. "
        "When False, losses generate no future tax benefit (legacy behavior).",
    )
    nol_limitation_pct: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="NOL deduction limitation as fraction of taxable income. "
        "Set to 0.80 per IRC §172(a)(2) post-TCJA. "
        "Set to 1.0 for pre-2018 NOLs or non-US jurisdictions.",
    )
    retention_ratio: float = Field(ge=0, le=1, description="Portion of earnings retained")
    ppe_ratio: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Property, Plant & Equipment allocation ratio (fraction of initial assets). "
        "If None, defaults based on operating margin: <10%: 0.3, 10-15%: 0.5, >15%: 0.7",
    )
    insolvency_tolerance: float = Field(
        default=10_000,
        gt=0,
        description="Insolvency threshold in dollars. Company is considered insolvent when "
        "equity falls below this level. Default of $10,000 (0.1%% of typical $10M assets) "
        "represents practical insolvency where company cannot maintain operations.",
    )
    expense_ratios: Optional["ExpenseRatioConfig"] = Field(
        default=None,
        description="Expense ratio configuration for COGS and SG&A breakdown. "
        "If None, default ratios from ExpenseRatioConfig are used. "
        "(Issue #255: Enables explicit COGS/SG&A calculation in Manufacturer)",
    )

    # Mid-year liquidity configuration (Issue #279)
    premium_payment_month: int = Field(
        default=0,
        ge=0,
        le=11,
        description="Month when annual insurance premium is paid (0-11, where 0=January). "
        "Used for intra-period liquidity estimation to detect mid-year insolvency.",
    )
    revenue_pattern: Literal["uniform", "seasonal", "back_loaded"] = Field(
        default="uniform",
        description="Revenue distribution pattern throughout the year. "
        "'uniform': equal monthly revenue, 'seasonal': higher in Q4, "
        "'back_loaded': 60% in H2. Used for mid-year liquidity estimation.",
    )
    check_intra_period_liquidity: bool = Field(
        default=True,
        description="Whether to check for potential mid-year insolvency by estimating "
        "minimum cash point within each period. When True, the simulation estimates "
        "the lowest cash point and triggers insolvency if it goes negative.",
    )

    @model_validator(mode="after")
    def set_default_ppe_ratio(self):
        """Set default PPE ratio based on operating margin if not provided."""
        if self.ppe_ratio is None:
            if self.base_operating_margin < 0.10:
                self.ppe_ratio = 0.3  # Low margin businesses need more working capital
            elif self.base_operating_margin < 0.15:
                self.ppe_ratio = 0.5  # Medium margin can support moderate PP&E
            else:
                self.ppe_ratio = 0.7  # High margin businesses can support more PP&E
        return self

    @field_validator("base_operating_margin")
    @classmethod
    def validate_margin(cls, v: float) -> float:
        """Warn if base operating margin is unusually high or negative.

        Args:
            v: Base operating margin value to validate (as decimal, e.g., 0.1 for 10%).

        Returns:
            float: The validated base operating margin value.

        Note:
            Margins above 30% are flagged as unusual for manufacturing.
            Negative margins indicate unprofitable operations before insurance.
        """
        if v > 0.3:
            print(f"Warning: Base operating margin {v:.1%} is unusually high")
        elif v < 0:
            print(f"Warning: Base operating margin {v:.1%} is negative")
        return v

    @classmethod
    def from_industry_config(cls, industry_config, **kwargs):
        """Create ManufacturerConfig from an IndustryConfig instance.

        Args:
            industry_config: IndustryConfig instance with industry-specific parameters
            **kwargs: Additional parameters to override or supplement

        Returns:
            ManufacturerConfig instance with parameters derived from industry config
        """
        # Map industry config parameters to manufacturer config
        # Use provided kwargs to override any derived values
        config_params = kwargs.copy()

        # Set base operating margin from industry config if not provided
        if "base_operating_margin" not in config_params:
            config_params["base_operating_margin"] = industry_config.operating_margin

        # Set PPE ratio from industry config if not provided
        if "ppe_ratio" not in config_params:
            config_params["ppe_ratio"] = industry_config.ppe_ratio

        # Set other defaults if not provided
        if "initial_assets" not in config_params:
            config_params["initial_assets"] = 10_000_000  # Default $10M
        if "asset_turnover_ratio" not in config_params:
            config_params["asset_turnover_ratio"] = 0.8  # Default 0.8x
        if "tax_rate" not in config_params:
            config_params["tax_rate"] = 0.25  # Default 25%
        if "retention_ratio" not in config_params:
            config_params["retention_ratio"] = 0.7  # Default 70%

        return cls(**config_params)


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
        ge=0, le=1, description="Working capital as percentage of sales"
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
    annual_growth_rate: float = Field(ge=-0.5, le=1.0, description="Annual growth rate")
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

    interest_rate: float = Field(ge=0, le=0.5, description="Annual interest rate on debt")
    max_leverage_ratio: float = Field(ge=0, le=10, description="Maximum debt-to-equity ratio")
    minimum_cash_balance: float = Field(ge=0, description="Minimum cash balance to maintain")


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
    time_horizon_years: int = Field(gt=0, le=1000, description="Simulation horizon in years")
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


class OutputConfig(BaseModel):
    """Output and results configuration.

    Controls where and how simulation results are saved, including
    file formats and checkpoint frequencies.
    """

    output_directory: str = Field(default="outputs", description="Directory for saving results")
    file_format: Literal["csv", "parquet", "json"] = Field(
        default="csv", description="Output file format"
    )
    checkpoint_frequency: int = Field(
        ge=0, default=0, description="Save checkpoints every N years (0=disabled)"
    )
    detailed_metrics: bool = Field(default=True, description="Include detailed metrics in output")

    @property
    def output_path(self) -> Path:
        """Get output directory as Path object.

        Returns:
            Path object for the output directory.
        """
        return Path(self.output_directory)


class LoggingConfig(BaseModel):
    """Logging configuration.

    Controls logging behavior including level, output destinations,
    and message formatting.
    """

    enabled: bool = Field(default=True, description="Enable logging")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None, description="Log file path (None=no file logging)"
    )
    console_output: bool = Field(default=True, description="Log to console")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )


class Config(BaseModel):
    """Complete configuration for the Ergodic Insurance simulation.

    This is the main configuration class that combines all sub-configurations
    and provides methods for loading, saving, and manipulating configurations.
    """

    manufacturer: ManufacturerConfig
    working_capital: WorkingCapitalConfig
    growth: GrowthConfig
    debt: DebtConfig
    simulation: SimulationConfig
    output: OutputConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Config object with validated parameters.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValidationError: If configuration is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Remove private anchors if present
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict, base_config: Optional["Config"] = None) -> "Config":
        """Create config from dictionary, optionally overriding base config.

        Args:
            data: Dictionary with configuration parameters.
            base_config: Optional base configuration to override.

        Returns:
            Config object with validated parameters.
        """
        if base_config is None:
            return cls(**data)

        # Start with base config as dict
        config_dict = base_config.model_dump()

        # Deep merge the override data
        def deep_merge(base: dict, override: dict) -> dict:
            """Recursively merge override into base.

            Args:
                base: Base dictionary to merge into.
                override: Override dictionary to merge from.

            Returns:
                Merged dictionary with overrides applied.
            """
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(config_dict, data)
        return cls(**merged)

    def override(self, **kwargs) -> "Config":
        """Create a new config with overridden parameters.

        Args:
            **kwargs: Parameters to override in dot notation
                     e.g., manufacturer__operating_margin=0.1.

        Returns:
            New Config object with overrides applied.
        """
        # Convert dot notation to nested dict
        override_dict: Dict[str, Any] = {}
        for key, value in kwargs.items():
            parts = key.split("__")
            current = override_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        return Config.from_dict(override_dict, base_config=self)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path where to save the configuration.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def setup_logging(self) -> None:
        """Configure logging based on settings.

        Sets up logging handlers for console and/or file output based
        on the logging configuration.
        """
        if not self.logging.enabled:
            return

        import logging
        import sys

        # Create logger
        logger = logging.getLogger("ergodic_insurance")
        logger.setLevel(getattr(logging, self.logging.level))
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(self.logging.format)

        # Console handler
        if self.logging.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if self.logging.log_file:
            log_path = Path(self.output.output_directory) / self.logging.log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def validate_paths(self) -> None:
        """Create output directories if they don't exist.

        Ensures that the configured output directory exists,
        creating it if necessary.
        """
        Path(self.output.output_directory).mkdir(parents=True, exist_ok=True)


class PricingScenario(BaseModel):
    """Individual market pricing scenario configuration.

    Represents a specific market condition (soft/normal/hard) with
    associated pricing parameters and market characteristics.
    """

    name: str = Field(description="Scenario name (e.g., 'Soft Market')")
    description: str = Field(description="Detailed scenario description")
    market_condition: Literal["soft", "normal", "hard"] = Field(description="Market condition type")

    # Layer-specific rates
    primary_layer_rate: float = Field(gt=0, le=0.05, description="Primary layer rate as % of limit")
    first_excess_rate: float = Field(gt=0, le=0.05, description="First excess rate as % of limit")
    higher_excess_rate: float = Field(gt=0, le=0.05, description="Higher excess rate as % of limit")

    # Market characteristics
    capacity_factor: float = Field(gt=0.5, le=2.0, description="Capacity relative to normal (1.0)")
    competition_level: Literal["low", "moderate", "high"] = Field(
        description="Level of market competition"
    )

    # Pricing factors
    retention_discount: float = Field(ge=0, le=0.5, description="Discount for higher retentions")
    volume_discount: float = Field(ge=0, le=0.5, description="Discount for large programs")
    loss_ratio_target: float = Field(gt=0, lt=1, description="Target loss ratio for insurers")
    expense_ratio: float = Field(gt=0, lt=1, description="Expense ratio for insurers")

    # Risk appetite
    new_business_appetite: Literal["restrictive", "selective", "aggressive"] = Field(
        description="Appetite for new business"
    )
    renewal_retention_focus: Literal["low", "balanced", "high"] = Field(
        description="Focus on retaining renewals"
    )
    coverage_enhancement_willingness: Literal["low", "moderate", "high"] = Field(
        description="Willingness to enhance coverage"
    )

    @model_validator(mode="after")
    def validate_rate_ordering(self) -> "PricingScenario":
        """Ensure premium rates follow expected ordering.

        Primary rates should be higher than excess rates, and first
        excess should be higher than higher excess layers.
        """
        if not self.primary_layer_rate >= self.first_excess_rate >= self.higher_excess_rate:
            raise ValueError(
                f"Rate ordering violation: primary ({self.primary_layer_rate:.3f}) >= "
                f"first_excess ({self.first_excess_rate:.3f}) >= "
                f"higher_excess ({self.higher_excess_rate:.3f}) must be maintained"
            )
        return self


class TransitionProbabilities(BaseModel):
    """Market state transition probabilities."""

    # From soft market
    soft_to_soft: float = Field(ge=0, le=1)
    soft_to_normal: float = Field(ge=0, le=1)
    soft_to_hard: float = Field(ge=0, le=1)

    # From normal market
    normal_to_soft: float = Field(ge=0, le=1)
    normal_to_normal: float = Field(ge=0, le=1)
    normal_to_hard: float = Field(ge=0, le=1)

    # From hard market
    hard_to_soft: float = Field(ge=0, le=1)
    hard_to_normal: float = Field(ge=0, le=1)
    hard_to_hard: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "TransitionProbabilities":
        """Ensure transition probabilities sum to 1.0 for each state."""
        soft_sum = self.soft_to_soft + self.soft_to_normal + self.soft_to_hard
        normal_sum = self.normal_to_soft + self.normal_to_normal + self.normal_to_hard
        hard_sum = self.hard_to_soft + self.hard_to_normal + self.hard_to_hard

        tolerance = 1e-6
        if abs(soft_sum - 1.0) > tolerance:
            raise ValueError(f"Soft market transitions sum to {soft_sum:.4f}, not 1.0")
        if abs(normal_sum - 1.0) > tolerance:
            raise ValueError(f"Normal market transitions sum to {normal_sum:.4f}, not 1.0")
        if abs(hard_sum - 1.0) > tolerance:
            raise ValueError(f"Hard market transitions sum to {hard_sum:.4f}, not 1.0")

        return self


class MarketCycles(BaseModel):
    """Market cycle configuration and dynamics."""

    average_duration_years: float = Field(gt=0, le=20)
    soft_market_duration: float = Field(gt=0, le=10)
    normal_market_duration: float = Field(gt=0, le=10)
    hard_market_duration: float = Field(gt=0, le=10)

    transition_probabilities: TransitionProbabilities = Field(
        description="Annual transition probabilities between market states"
    )

    @model_validator(mode="after")
    def validate_cycle_duration(self) -> "MarketCycles":
        """Validate that cycle durations are reasonable."""
        total_duration = (
            self.soft_market_duration + self.normal_market_duration + self.hard_market_duration
        )

        # Check if average duration is reasonable given components
        expected_avg = total_duration / 3
        if abs(self.average_duration_years - expected_avg) > expected_avg * 0.5:
            print(
                f"Warning: Average duration ({self.average_duration_years:.1f} years) "
                f"differs significantly from component average ({expected_avg:.1f} years)"
            )

        return self


class PricingScenarioConfig(BaseModel):
    """Complete pricing scenario configuration.

    Contains all market scenarios and cycle dynamics for
    insurance pricing sensitivity analysis.
    """

    scenarios: Dict[str, PricingScenario] = Field(
        description="Market scenarios (inexpensive/baseline/expensive)"
    )
    market_cycles: MarketCycles = Field(description="Market cycle dynamics and transitions")

    def get_scenario(self, scenario_name: str) -> PricingScenario:
        """Get a specific pricing scenario by name.

        Args:
            scenario_name: Name of the scenario to retrieve

        Returns:
            PricingScenario configuration

        Raises:
            KeyError: If scenario_name not found
        """
        if scenario_name not in self.scenarios:
            available = ", ".join(self.scenarios.keys())
            raise KeyError(
                f"Scenario '{scenario_name}' not found. " f"Available scenarios: {available}"
            )
        return self.scenarios[scenario_name]

    def get_rate_multiplier(self, from_scenario: str, to_scenario: str) -> float:
        """Calculate rate change multiplier between scenarios.

        Args:
            from_scenario: Starting scenario name
            to_scenario: Target scenario name

        Returns:
            Multiplier for premium rates when transitioning
        """
        from_rates = self.scenarios[from_scenario]
        to_rates = self.scenarios[to_scenario]

        # Average the rate changes across layers
        primary_mult = to_rates.primary_layer_rate / from_rates.primary_layer_rate
        excess_mult = to_rates.first_excess_rate / from_rates.first_excess_rate
        higher_mult = to_rates.higher_excess_rate / from_rates.higher_excess_rate

        return (primary_mult + excess_mult + higher_mult) / 3


class ProfileMetadata(BaseModel):
    """Metadata for configuration profiles."""

    name: str = Field(description="Profile name")
    description: str = Field(description="Profile description")
    version: str = Field(default="2.0.0", description="Profile version")
    extends: Optional[str] = Field(default=None, description="Parent profile to extend")
    includes: List[str] = Field(default_factory=list, description="Modules to include")
    presets: Dict[str, str] = Field(default_factory=dict, description="Presets to apply")
    author: Optional[str] = Field(default=None, description="Profile author")
    created: Optional[datetime] = Field(default_factory=datetime.now, description="Creation date")
    tags: List[str] = Field(default_factory=list, description="Profile tags for discovery")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure profile name is valid.

        Args:
            v: Profile name to validate.

        Returns:
            Validated profile name.

        Raises:
            ValueError: If name contains invalid characters.
        """
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid profile name: {v}")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version string.

        Args:
            v: Version string to validate.

        Returns:
            Validated version string.

        Raises:
            ValueError: If version format is invalid.
        """
        import re

        if not re.match(r"^\d+\.\d+\.\d+(-[\w.]+)?$", v):
            raise ValueError(f"Invalid version format: {v}")
        return v


class InsuranceLayerConfig(BaseModel):
    """Configuration for a single insurance layer."""

    name: str = Field(description="Layer name")
    limit: float = Field(gt=0, description="Layer limit in dollars")
    attachment: float = Field(ge=0, description="Attachment point in dollars")
    base_premium_rate: float = Field(gt=0, le=1, description="Premium as percentage of limit")
    reinstatements: int = Field(default=0, ge=0, description="Number of reinstatements")
    aggregate_limit: Optional[float] = Field(
        default=None, gt=0, description="Aggregate limit if applicable"
    )
    limit_type: str = Field(
        default="per-occurrence",
        description="Type of limit: 'per-occurrence', 'aggregate', or 'hybrid'",
    )
    per_occurrence_limit: Optional[float] = Field(
        default=None, gt=0, description="Per-occurrence limit for hybrid type"
    )

    @model_validator(mode="after")
    def validate_layer_structure(self):
        """Ensure layer structure is valid.

        Returns:
            Validated layer config.

        Raises:
            ValueError: If layer structure is invalid.
        """
        # Validate limit type
        valid_limit_types = ["per-occurrence", "aggregate", "hybrid"]
        if self.limit_type not in valid_limit_types:
            raise ValueError(
                f"Invalid limit_type: {self.limit_type}. Must be one of {valid_limit_types}"
            )

        # Validate based on limit type
        if self.limit_type == "hybrid":
            # For hybrid, need both per-occurrence and aggregate limits
            if self.per_occurrence_limit is None and self.aggregate_limit is None:
                raise ValueError(
                    "Hybrid limit type requires both per_occurrence_limit and aggregate_limit to be set"
                )

        return self


class InsuranceConfig(BaseModel):
    """Enhanced insurance configuration."""

    enabled: bool = Field(default=True, description="Whether insurance is enabled")
    layers: List[InsuranceLayerConfig] = Field(default_factory=list, description="Insurance layers")
    deductible: float = Field(default=0, ge=0, description="Deductible amount")
    coinsurance: float = Field(default=1.0, gt=0, le=1, description="Coinsurance percentage")
    waiting_period_days: int = Field(default=0, ge=0, description="Waiting period for claims")
    claims_handling_cost: float = Field(
        default=0.05, ge=0, le=1, description="Claims handling cost as percentage"
    )

    @model_validator(mode="after")
    def validate_layers(self):
        """Ensure layers don't overlap and are properly ordered.

        Returns:
            Validated insurance config.

        Raises:
            ValueError: If layers overlap or are misordered.
        """
        if not self.layers:
            return self

        # Sort layers by attachment point
        sorted_layers = sorted(self.layers, key=lambda x: x.attachment)

        for i in range(len(sorted_layers) - 1):
            current = sorted_layers[i]
            next_layer = sorted_layers[i + 1]

            # Check for gaps or overlaps
            if current.attachment + current.limit < next_layer.attachment:
                print(f"Warning: Gap between layers {current.name} and {next_layer.name}")
            elif current.attachment + current.limit > next_layer.attachment:
                raise ValueError(f"Layers {current.name} and {next_layer.name} overlap")

        return self


class LossDistributionConfig(BaseModel):
    """Configuration for loss distributions."""

    frequency_distribution: str = Field(
        default="poisson", description="Frequency distribution type"
    )
    frequency_annual: float = Field(gt=0, description="Annual expected frequency")
    severity_distribution: str = Field(
        default="lognormal", description="Severity distribution type"
    )
    severity_mean: float = Field(gt=0, description="Mean severity")
    severity_std: float = Field(gt=0, description="Severity standard deviation")
    correlation_factor: float = Field(
        default=0.0, ge=-1, le=1, description="Correlation between frequency and severity"
    )
    tail_alpha: float = Field(default=2.0, gt=1, description="Tail heaviness parameter")

    @field_validator("frequency_distribution")
    @classmethod
    def validate_frequency_dist(cls, v: str) -> str:
        """Validate frequency distribution type.

        Args:
            v: Distribution type.

        Returns:
            Validated distribution type.

        Raises:
            ValueError: If distribution type is invalid.
        """
        valid_dists = ["poisson", "negative_binomial", "binomial"]
        if v not in valid_dists:
            raise ValueError(f"Invalid frequency distribution: {v}. Must be one of {valid_dists}")
        return v

    @field_validator("severity_distribution")
    @classmethod
    def validate_severity_dist(cls, v: str) -> str:
        """Validate severity distribution type.

        Args:
            v: Distribution type.

        Returns:
            Validated distribution type.

        Raises:
            ValueError: If distribution type is invalid.
        """
        valid_dists = ["lognormal", "gamma", "pareto", "weibull"]
        if v not in valid_dists:
            raise ValueError(f"Invalid severity distribution: {v}. Must be one of {valid_dists}")
        return v


class ModuleConfig(BaseModel):
    """Base class for configuration modules."""

    module_name: str = Field(description="Module identifier")
    module_version: str = Field(default="2.0.0", description="Module version")
    dependencies: List[str] = Field(default_factory=list, description="Required modules")

    model_config = {"extra": "allow"}  # Allow additional fields


class PresetConfig(BaseModel):
    """Configuration for a preset."""

    preset_name: str = Field(description="Preset identifier")
    preset_type: str = Field(description="Type of preset (market, layers, risk, etc.)")
    description: str = Field(description="Preset description")
    parameters: Dict[str, Any] = Field(description="Preset parameters")

    @field_validator("preset_type")
    @classmethod
    def validate_preset_type(cls, v: str) -> str:
        """Validate preset type.

        Args:
            v: Preset type.

        Returns:
            Validated preset type.

        Raises:
            ValueError: If preset type is invalid.
        """
        valid_types = ["market", "layers", "risk", "optimization", "scenario"]
        if v not in valid_types:
            raise ValueError(f"Invalid preset type: {v}. Must be one of {valid_types}")
        return v


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
            print(f"Warning: Negative cash conversion cycle ({ccc:.0f} days)")
        elif ccc > 180:
            print(f"Warning: Very long cash conversion cycle ({ccc:.0f} days)")
        return self


class ExpenseRatioConfig(BaseModel):
    """Configuration for expense categorization and allocation.

    Defines how revenue translates to expenses with proper GAAP categorization
    between COGS and operating expenses (SG&A).

    Issue #255: COGS and SG&A breakdown ratios are now configurable to allow
    the Manufacturer to calculate these values explicitly, rather than having
    the Reporting layer estimate them with hardcoded ratios.
    """

    gross_margin_ratio: float = Field(
        default=0.15,
        gt=0,
        lt=1,
        description="Gross margin ratio (Revenue - COGS) / Revenue",
    )
    sga_expense_ratio: float = Field(
        default=0.07,
        gt=0,
        lt=1,
        description="SG&A expenses as percentage of revenue",
    )
    manufacturing_depreciation_allocation: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Percentage of depreciation allocated to COGS (manufacturing)",
    )
    admin_depreciation_allocation: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Percentage of depreciation allocated to SG&A (administrative)",
    )

    # COGS breakdown ratios (Issue #255)
    direct_materials_ratio: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Direct materials as percentage of COGS (excluding depreciation)",
    )
    direct_labor_ratio: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Direct labor as percentage of COGS (excluding depreciation)",
    )
    manufacturing_overhead_ratio: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Manufacturing overhead as percentage of COGS (excluding depreciation)",
    )

    # SG&A breakdown ratios (Issue #255)
    selling_expense_ratio: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Selling expenses as percentage of SG&A (excluding depreciation)",
    )
    general_admin_ratio: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="General & Admin as percentage of SG&A (excluding depreciation)",
    )

    @model_validator(mode="after")
    def validate_depreciation_allocation(self):
        """Ensure depreciation allocations sum to 100%."""
        total = self.manufacturing_depreciation_allocation + self.admin_depreciation_allocation
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Depreciation allocations must sum to 100%, got {total*100:.1f}%")
        return self

    @model_validator(mode="after")
    def validate_cogs_breakdown(self):
        """Ensure COGS breakdown ratios sum to 100%."""
        total = (
            self.direct_materials_ratio
            + self.direct_labor_ratio
            + self.manufacturing_overhead_ratio
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"COGS breakdown ratios must sum to 100%, got {total*100:.1f}%")
        return self

    @model_validator(mode="after")
    def validate_sga_breakdown(self):
        """Ensure SG&A breakdown ratios sum to 100%."""
        total = self.selling_expense_ratio + self.general_admin_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"SG&A breakdown ratios must sum to 100%, got {total*100:.1f}%")
        return self

    @property
    def cogs_ratio(self) -> float:
        """Calculate COGS as percentage of revenue."""
        return 1.0 - self.gross_margin_ratio

    @property
    def operating_margin_ratio(self) -> float:
        """Calculate operating margin after all operating expenses."""
        return self.gross_margin_ratio - self.sga_expense_ratio


class DepreciationConfig(BaseModel):
    """Configuration for depreciation and amortization tracking.

    Defines how fixed assets depreciate and prepaid expenses amortize over time.
    """

    ppe_useful_life_years: float = Field(
        default=10,
        gt=0,
        le=50,
        description="Average useful life of PP&E in years for straight-line depreciation",
    )
    prepaid_insurance_amortization_months: int = Field(
        default=12,
        gt=0,
        le=24,
        description="Number of months over which prepaid insurance amortizes",
    )
    initial_accumulated_depreciation: float = Field(
        default=0, ge=0, description="Starting accumulated depreciation balance"
    )

    @property
    def annual_depreciation_rate(self) -> float:
        """Calculate annual depreciation rate."""
        return 1.0 / self.ppe_useful_life_years

    @property
    def monthly_insurance_amortization_rate(self) -> float:
        """Calculate monthly insurance amortization rate."""
        return 1.0 / self.prepaid_insurance_amortization_months


class ExcelReportConfig(BaseModel):
    """Configuration for Excel report generation."""

    enabled: bool = Field(default=True, description="Whether Excel reporting is enabled")
    output_path: str = Field(default="./reports", description="Directory for Excel reports")
    include_balance_sheet: bool = Field(default=True, description="Include balance sheet")
    include_income_statement: bool = Field(default=True, description="Include income statement")
    include_cash_flow: bool = Field(default=True, description="Include cash flow statement")
    include_reconciliation: bool = Field(default=True, description="Include reconciliation report")
    include_metrics_dashboard: bool = Field(default=True, description="Include metrics dashboard")
    include_pivot_data: bool = Field(default=True, description="Include pivot-ready data")
    engine: str = Field(default="auto", description="Excel engine: xlsxwriter, openpyxl, or auto")
    currency_format: str = Field(default="$#,##0", description="Currency format string")
    decimal_places: int = Field(default=0, ge=0, le=10, description="Number of decimal places")
    date_format: str = Field(default="yyyy-mm-dd", description="Date format string")

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        """Validate Excel engine selection.

        Args:
            v: Engine name to validate.

        Returns:
            Validated engine name.

        Raises:
            ValueError: If engine is not valid.
        """
        valid_engines = ["xlsxwriter", "openpyxl", "auto", "pandas"]
        if v not in valid_engines:
            raise ValueError(f"Invalid Excel engine: {v}. Must be one of {valid_engines}")
        return v


@dataclass
class IndustryConfig:
    """Base configuration for different industry types.

    This class defines industry-specific financial parameters that determine
    how businesses operate, including working capital needs, margin structures,
    asset composition, and depreciation policies.

    Attributes:
        industry_type: Name of the industry (e.g., 'manufacturing', 'services')

        Working capital ratios:
        days_sales_outstanding: Average collection period for receivables (days)
        days_inventory_outstanding: Average inventory holding period (days)
        days_payables_outstanding: Average payment period to suppliers (days)

        Margin structure:
        gross_margin: Gross profit as percentage of revenue
        operating_expense_ratio: Operating expenses as percentage of revenue

        Asset composition:
        current_asset_ratio: Current assets as fraction of total assets
        ppe_ratio: Property, Plant & Equipment as fraction of total assets
        intangible_ratio: Intangible assets as fraction of total assets

        Depreciation:
        ppe_useful_life: Average useful life of PP&E in years
        depreciation_method: Method for calculating depreciation
    """

    industry_type: str = "manufacturing"

    # Working capital ratios (in days)
    days_sales_outstanding: float = 45
    days_inventory_outstanding: float = 60
    days_payables_outstanding: float = 30

    # Margin structure (as percentages)
    gross_margin: float = 0.35
    operating_expense_ratio: float = 0.25

    # Asset composition (must sum to 1.0)
    current_asset_ratio: float = 0.4
    ppe_ratio: float = 0.5
    intangible_ratio: float = 0.1

    # Depreciation settings
    ppe_useful_life: int = 10  # years
    depreciation_method: str = "straight_line"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate that all parameters are within reasonable bounds."""
        # Validate margins
        assert (
            0 <= self.gross_margin <= 1
        ), f"Gross margin must be between 0 and 1, got {self.gross_margin}"
        assert (
            0 <= self.operating_expense_ratio <= 1
        ), f"Operating expense ratio must be between 0 and 1, got {self.operating_expense_ratio}"

        # Validate asset composition
        asset_sum = self.current_asset_ratio + self.ppe_ratio + self.intangible_ratio
        assert abs(asset_sum - 1.0) < 0.01, f"Asset ratios must sum to 1.0, got {asset_sum}"

        # Validate working capital days
        assert self.days_sales_outstanding >= 0, "Days sales outstanding must be non-negative"
        assert (
            self.days_inventory_outstanding >= 0
        ), "Days inventory outstanding must be non-negative"
        assert self.days_payables_outstanding >= 0, "Days payables outstanding must be non-negative"

        # Validate depreciation
        assert self.ppe_useful_life > 0, "PPE useful life must be positive"
        assert self.depreciation_method in [
            "straight_line",
            "declining_balance",
        ], f"Unknown depreciation method: {self.depreciation_method}"

    @property
    def working_capital_days(self) -> float:
        """Calculate net working capital cycle in days."""
        return (
            self.days_sales_outstanding
            + self.days_inventory_outstanding
            - self.days_payables_outstanding
        )

    @property
    def operating_margin(self) -> float:
        """Calculate operating margin (EBIT margin)."""
        return self.gross_margin - self.operating_expense_ratio


class ManufacturingConfig(IndustryConfig):
    """Configuration for manufacturing companies.

    Manufacturing businesses typically have:
    - Significant inventory holdings
    - Moderate to high PP&E requirements
    - Working capital needs for raw materials and WIP
    - Gross margins of 25-40%
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with manufacturing-specific defaults."""
        defaults: Dict[str, Any] = {
            "industry_type": "manufacturing",
            "days_sales_outstanding": 45,
            "days_inventory_outstanding": 60,
            "days_payables_outstanding": 30,
            "gross_margin": 0.35,
            "operating_expense_ratio": 0.25,
            "current_asset_ratio": 0.4,
            "ppe_ratio": 0.5,
            "intangible_ratio": 0.1,
            "ppe_useful_life": 10,
            "depreciation_method": "straight_line",
        }
        # Override defaults with any provided kwargs
        defaults.update(kwargs)
        super().__init__(**defaults)


class ServiceConfig(IndustryConfig):
    """Configuration for service companies.

    Service businesses typically have:
    - Minimal or no inventory
    - Lower PP&E requirements
    - Faster cash conversion cycles
    - Higher gross margins but also higher operating expenses
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with service-specific defaults."""
        defaults: Dict[str, Any] = {
            "industry_type": "services",
            "days_sales_outstanding": 30,
            "days_inventory_outstanding": 0,  # No inventory for services
            "days_payables_outstanding": 20,
            "gross_margin": 0.60,
            "operating_expense_ratio": 0.45,
            "current_asset_ratio": 0.6,
            "ppe_ratio": 0.2,  # Less capital intensive
            "intangible_ratio": 0.2,  # More intangibles (brand, IP)
            "ppe_useful_life": 5,
            "depreciation_method": "straight_line",
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class RetailConfig(IndustryConfig):
    """Configuration for retail companies.

    Retail businesses typically have:
    - High inventory turnover
    - Moderate PP&E (stores, fixtures)
    - Fast cash collection (often immediate)
    - Lower gross margins but efficient operations
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with retail-specific defaults."""
        defaults: Dict[str, Any] = {
            "industry_type": "retail",
            "days_sales_outstanding": 5,  # Mostly cash/credit card sales
            "days_inventory_outstanding": 45,
            "days_payables_outstanding": 35,
            "gross_margin": 0.30,
            "operating_expense_ratio": 0.22,
            "current_asset_ratio": 0.5,
            "ppe_ratio": 0.4,
            "intangible_ratio": 0.1,
            "ppe_useful_life": 7,
            "depreciation_method": "straight_line",
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class ConfigV2(BaseModel):
    """Enhanced unified configuration model for the 3-tier system."""

    profile: ProfileMetadata
    manufacturer: ManufacturerConfig
    working_capital: WorkingCapitalConfig
    growth: GrowthConfig
    debt: DebtConfig
    simulation: SimulationConfig
    output: OutputConfig
    logging: LoggingConfig
    insurance: Optional[InsuranceConfig] = None
    losses: Optional[LossDistributionConfig] = None
    excel_reporting: Optional[ExcelReportConfig] = None
    working_capital_ratios: Optional[WorkingCapitalRatiosConfig] = None
    expense_ratios: Optional[ExpenseRatioConfig] = None
    depreciation: Optional[DepreciationConfig] = None
    industry_config: Optional[IndustryConfig] = Field(
        default=None, description="Industry-specific configuration for financial parameters"
    )

    # Additional fields for extensibility
    custom_modules: Dict[str, ModuleConfig] = Field(
        default_factory=dict, description="Custom modules"
    )
    applied_presets: List[str] = Field(default_factory=list, description="List of applied presets")
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Runtime overrides")

    @classmethod
    def from_profile(cls, profile_path: Path) -> "ConfigV2":
        """Load configuration from a profile file.

        Args:
            profile_path: Path to the profile YAML file.

        Returns:
            Loaded and validated ConfigV2 instance.

        Raises:
            FileNotFoundError: If profile file doesn't exist.
            ValidationError: If configuration is invalid.
        """
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)

        # Remove YAML anchors
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        return cls(**data)

    @classmethod
    def with_inheritance(cls, profile_path: Path, config_dir: Path) -> "ConfigV2":
        """Load configuration with profile inheritance.

        Args:
            profile_path: Path to the profile YAML file.
            config_dir: Root configuration directory.

        Returns:
            Loaded ConfigV2 with inheritance applied.
        """
        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)

        # Handle inheritance
        if "profile" in data and "extends" in data["profile"] and data["profile"]["extends"]:
            parent_name = data["profile"]["extends"]
            parent_path = config_dir / "profiles" / f"{parent_name}.yaml"

            if parent_path.exists():
                parent_config = cls.with_inheritance(parent_path, config_dir)
                parent_data = parent_config.model_dump()

                # Deep merge parent with child
                merged_data = cls._deep_merge(parent_data, data)
                data = merged_data

        return cls(**data)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary.
            override: Override dictionary.

        Returns:
            Merged dictionary.
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigV2._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def apply_module(self, module_path: Path) -> None:
        """Apply a configuration module.

        Args:
            module_path: Path to the module YAML file.
        """
        with open(module_path, "r") as f:
            module_data = yaml.safe_load(f)

        # Apply module data to current config
        for key, value in module_data.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    current = getattr(self, key)
                    if isinstance(current, BaseModel):
                        # Update Pydantic model
                        updated = current.model_dump()
                        updated.update(value)
                        setattr(self, key, type(current)(**updated))
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)

    def apply_preset(self, preset_name: str, preset_data: Dict[str, Any]) -> None:
        """Apply a preset to the configuration.

        Args:
            preset_name: Name of the preset.
            preset_data: Preset parameters to apply.
        """
        # Track applied preset
        self.applied_presets.append(preset_name)

        # Apply preset data
        for key, value in preset_data.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    current = getattr(self, key)
                    if isinstance(current, BaseModel):
                        updated = current.model_dump()
                        updated.update(value)
                        setattr(self, key, type(current)(**updated))
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)

    def with_overrides(self, **kwargs) -> "ConfigV2":
        """Create a new config with runtime overrides.

        Args:
            **kwargs: Override parameters in format section__field=value.

        Returns:
            New ConfigV2 instance with overrides applied.
        """
        # Create a copy of current config
        data = self.model_dump()

        # Apply overrides
        for key, value in kwargs.items():
            if "__" in key:
                # Handle nested overrides like manufacturer__initial_assets
                parts = key.split("__")
                current = data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # For nested objects, merge instead of replace
                if isinstance(value, dict) and key in data and isinstance(data[key], dict):
                    # Merge dictionaries recursively
                    data[key] = {**data[key], **value}
                else:
                    data[key] = value

        # Track overrides
        data["overrides"] = kwargs

        return ConfigV2(**data)

    def validate_completeness(self) -> List[str]:
        """Validate configuration completeness.

        Returns:
            List of missing or invalid configuration items.
        """
        issues = []

        # Check required sections
        required_sections = ["manufacturer", "simulation", "growth"]
        for section in required_sections:
            if not getattr(self, section, None):
                issues.append(f"Missing required section: {section}")

        # Check for logical consistency
        if self.insurance and self.insurance.enabled and not self.losses:
            issues.append("Insurance enabled but no loss distribution configured")

        return issues


class PresetLibrary(BaseModel):
    """Collection of presets for a specific type."""

    library_type: str = Field(description="Type of preset library")
    description: str = Field(description="Library description")
    presets: Dict[str, PresetConfig] = Field(default_factory=dict, description="Available presets")

    @classmethod
    def from_yaml(cls, path: Path) -> "PresetLibrary":
        """Load preset library from YAML file.

        Args:
            path: Path to preset library YAML file.

        Returns:
            Loaded PresetLibrary instance.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Determine library type from filename
        library_type = path.stem.replace("_", " ").title()

        # Map filename to valid preset type
        preset_type_map = {
            "market_conditions": "market",
            "risk_profiles": "risk",
            "layer_structures": "layers",
            "optimization_settings": "optimization",
            "scenario_definitions": "scenario",
        }
        # Use mapped type or default to "scenario"
        preset_type = preset_type_map.get(path.stem, "scenario")

        presets = {}
        for name, params in data.items():
            presets[name] = PresetConfig(
                preset_name=name,
                preset_type=preset_type,
                description=f"{name} preset for {library_type}",
                parameters=params,
            )

        return cls(
            library_type=library_type,
            description=f"Preset library for {library_type}",
            presets=presets,
        )

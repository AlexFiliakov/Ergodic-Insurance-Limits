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

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    retention_ratio: float = Field(ge=0, le=1, description="Portion of earnings retained")

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
        import yaml

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
        import yaml

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

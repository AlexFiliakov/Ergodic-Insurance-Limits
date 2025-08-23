"""Configuration management using Pydantic v2 models."""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ManufacturerConfig(BaseModel):
    """Financial parameters for the widget manufacturer."""

    initial_assets: float = Field(gt=0, description="Starting asset value in dollars")
    asset_turnover_ratio: float = Field(gt=0, le=5, description="Revenue per dollar of assets")
    operating_margin: float = Field(
        gt=-1, lt=1, description="Operating income as percentage of revenue"
    )
    tax_rate: float = Field(ge=0, le=1, description="Corporate tax rate")
    retention_ratio: float = Field(ge=0, le=1, description="Portion of earnings retained")

    @field_validator("operating_margin")
    @classmethod
    def validate_margin(cls, v: float) -> float:
        """Warn if operating margin is unusually high or negative."""
        if v > 0.3:
            print(f"Warning: Operating margin {v:.1%} is unusually high")
        elif v < 0:
            print(f"Warning: Operating margin {v:.1%} is negative")
        return v


class WorkingCapitalConfig(BaseModel):
    """Working capital management parameters."""

    percent_of_sales: float = Field(
        ge=0, le=1, description="Working capital as percentage of sales"
    )

    @field_validator("percent_of_sales")
    @classmethod
    def validate_working_capital(cls, v: float) -> float:
        """Validate working capital percentage."""
        if v > 0.5:
            raise ValueError(f"Working capital {v:.1%} of sales is unrealistically high")
        return v


class GrowthConfig(BaseModel):
    """Growth model parameters."""

    type: Literal["deterministic", "stochastic"] = Field(
        default="deterministic", description="Growth model type"
    )
    annual_growth_rate: float = Field(ge=-0.5, le=1.0, description="Annual growth rate")
    volatility: float = Field(
        ge=0, le=1, default=0.0, description="Growth rate volatility (std dev)"
    )

    @model_validator(mode="after")
    def validate_stochastic_params(self):
        """Ensure volatility is set for stochastic models."""
        if self.type == "stochastic" and self.volatility == 0:
            raise ValueError("Stochastic model requires non-zero volatility")
        return self


class DebtConfig(BaseModel):
    """Debt financing parameters for insurance claims."""

    interest_rate: float = Field(ge=0, le=0.5, description="Annual interest rate on debt")
    max_leverage_ratio: float = Field(ge=0, le=10, description="Maximum debt-to-equity ratio")
    minimum_cash_balance: float = Field(ge=0, description="Minimum cash balance to maintain")


class SimulationConfig(BaseModel):
    """Simulation execution parameters."""

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
        """Ensure time horizon doesn't exceed maximum."""
        if self.time_horizon_years > self.max_horizon_years:
            raise ValueError(
                f"Time horizon {self.time_horizon_years} exceeds maximum "
                f"{self.max_horizon_years}"
            )
        return self


class OutputConfig(BaseModel):
    """Output and results configuration."""

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
        """Get output directory as Path object."""
        return Path(self.output_directory)


class LoggingConfig(BaseModel):
    """Logging configuration."""

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
    """Complete configuration for the Ergodic Insurance simulation."""

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
            path: Path to YAML configuration file

        Returns:
            Config object with validated parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
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
            data: Dictionary with configuration parameters
            base_config: Optional base configuration to override

        Returns:
            Config object with validated parameters
        """
        if base_config is None:
            return cls(**data)

        # Start with base config as dict
        config_dict = base_config.model_dump()

        # Deep merge the override data
        def deep_merge(base: dict, override: dict) -> dict:
            """Recursively merge override into base."""
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
                     e.g., manufacturer__operating_margin=0.1

        Returns:
            New Config object with overrides applied
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
            path: Path where to save the configuration
        """
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
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
        """Create output directories if they don't exist."""
        Path(self.output.output_directory).mkdir(parents=True, exist_ok=True)

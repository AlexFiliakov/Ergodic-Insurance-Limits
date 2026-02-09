"""Master configuration classes composing all sub-configurations.

Contains the top-level ``Config`` (v1) and ``ConfigV2`` (3-tier system) classes
that aggregate all domain-specific configuration sub-modules into unified
configuration objects with loading, saving, and override capabilities.

Since:
    Version 0.9.0 (Issue #458)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import yaml

from .insurance import InsuranceConfig, LossDistributionConfig
from .manufacturer import (
    DepreciationConfig,
    ExpenseRatioConfig,
    IndustryConfig,
    ManufacturerConfig,
)
from .presets import ModuleConfig, ProfileMetadata
from .reporting import ExcelReportConfig, LoggingConfig, OutputConfig
from .simulation import (
    DebtConfig,
    GrowthConfig,
    SimulationConfig,
    WorkingCapitalConfig,
    WorkingCapitalRatiosConfig,
)
from .utils import deep_merge


class Config(BaseModel):
    """Complete configuration for the Ergodic Insurance simulation.

    This is the main configuration class that combines all sub-configurations
    and provides methods for loading, saving, and manipulating configurations.

    All sub-configs have sensible defaults, so ``Config()`` with no arguments
    creates a valid configuration for a $10M widget manufacturer.

    Examples:
        Minimal usage::

            config = Config()

        Override specific parameters::

            config = Config(
                manufacturer=ManufacturerConfig(initial_assets=20_000_000)
            )

        From basic company info::

            config = Config.from_company(initial_assets=50_000_000, operating_margin=0.12)
    """

    manufacturer: ManufacturerConfig = Field(default_factory=ManufacturerConfig)
    working_capital: WorkingCapitalConfig = Field(default_factory=WorkingCapitalConfig)
    growth: GrowthConfig = Field(default_factory=GrowthConfig)
    debt: DebtConfig = Field(default_factory=DebtConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_company(
        cls,
        initial_assets: float = 10_000_000,
        operating_margin: float = 0.08,
        industry: str = "manufacturing",
        tax_rate: float = 0.25,
        growth_rate: float = 0.05,
        time_horizon_years: int = 50,
        **kwargs,
    ) -> "Config":
        """Create a Config from basic company information.

        This factory derives reasonable sub-config defaults from a small number
        of intuitive business parameters, so actuaries and risk managers can get
        started quickly without understanding every sub-config class.

        Args:
            initial_assets: Starting asset value in dollars.
            operating_margin: Base operating margin (e.g. 0.08 for 8%).
            industry: Industry type for deriving defaults.
                Supported values: "manufacturing", "service", "retail".
            tax_rate: Corporate tax rate.
            growth_rate: Annual growth rate.
            time_horizon_years: Simulation horizon in years.
            **kwargs: Additional overrides passed to sub-configs.

        Returns:
            Config object with parameters derived from company info.

        Examples:
            Minimal::

                config = Config.from_company(initial_assets=50_000_000)

            With industry defaults::

                config = Config.from_company(
                    initial_assets=25_000_000,
                    operating_margin=0.15,
                    industry="service",
                )
        """
        # Industry-specific defaults
        industry_defaults = {
            "manufacturing": {
                "asset_turnover_ratio": 0.8,
                "retention_ratio": 0.7,
                "percent_of_sales": 0.20,
                "minimum_cash_balance": initial_assets * 0.05,
            },
            "service": {
                "asset_turnover_ratio": 1.2,
                "retention_ratio": 0.6,
                "percent_of_sales": 0.15,
                "minimum_cash_balance": initial_assets * 0.03,
            },
            "retail": {
                "asset_turnover_ratio": 1.5,
                "retention_ratio": 0.5,
                "percent_of_sales": 0.25,
                "minimum_cash_balance": initial_assets * 0.04,
            },
        }

        defaults = industry_defaults.get(industry, industry_defaults["manufacturing"])

        return cls(
            manufacturer=ManufacturerConfig(
                initial_assets=initial_assets,
                asset_turnover_ratio=kwargs.get(
                    "asset_turnover_ratio", defaults["asset_turnover_ratio"]
                ),
                base_operating_margin=operating_margin,
                tax_rate=tax_rate,
                retention_ratio=kwargs.get("retention_ratio", defaults["retention_ratio"]),
            ),
            working_capital=WorkingCapitalConfig(
                percent_of_sales=kwargs.get("percent_of_sales", defaults["percent_of_sales"]),
            ),
            growth=GrowthConfig(annual_growth_rate=growth_rate),
            debt=DebtConfig(
                minimum_cash_balance=kwargs.get(
                    "minimum_cash_balance", defaults["minimum_cash_balance"]
                ),
            ),
            simulation=SimulationConfig(time_horizon_years=time_horizon_years),
        )

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
    def with_inheritance(
        cls,
        profile_path: Path,
        config_dir: Path,
        _visited: Optional[frozenset] = None,
    ) -> "ConfigV2":
        """Load configuration with profile inheritance.

        Args:
            profile_path: Path to the profile YAML file.
            config_dir: Root configuration directory.
            _visited: Internal set of already-visited profile paths for
                cycle detection.  Callers should not pass this argument.

        Returns:
            Loaded ConfigV2 with inheritance applied.

        Raises:
            ValueError: If circular inheritance is detected.
        """
        resolved = profile_path.resolve()
        visited = _visited or frozenset()
        if resolved in visited:
            chain = " -> ".join(str(p) for p in visited)
            raise ValueError(f"Circular profile inheritance detected: {chain} -> {resolved}")
        visited = visited | {resolved}

        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)

        # Handle inheritance
        if "profile" in data and "extends" in data["profile"] and data["profile"]["extends"]:
            parent_name = data["profile"]["extends"]
            parent_path = config_dir / "profiles" / f"{parent_name}.yaml"

            if parent_path.exists():
                parent_config = cls.with_inheritance(parent_path, config_dir, _visited=visited)
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
        return deep_merge(base, override)

    def apply_module(self, module_path: Path) -> None:
        """Apply a configuration module.

        Merges module data via dict-dump-merge-reconstruct so that every
        field change goes through Pydantic validation.

        Args:
            module_path: Path to the module YAML file.
        """
        with open(module_path, "r") as f:
            module_data = yaml.safe_load(f)

        # Dump → merge → reconstruct to enforce Pydantic validation
        current_data = self.model_dump()
        merged = deep_merge(current_data, module_data)
        updated = self.model_validate(merged)

        # Copy all validated fields back
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))

    def apply_preset(self, preset_name: str, preset_data: Dict[str, Any]) -> None:
        """Apply a preset to the configuration.

        Merges preset data via dict-dump-merge-reconstruct so that every
        field change goes through Pydantic validation.

        Args:
            preset_name: Name of the preset.
            preset_data: Preset parameters to apply.
        """
        # Dump → merge → reconstruct to enforce Pydantic validation
        current_data = self.model_dump()
        current_data.setdefault("applied_presets", [])
        current_data["applied_presets"].append(preset_name)
        merged = deep_merge(current_data, preset_data)
        updated = self.model_validate(merged)

        # Copy all validated fields back
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))

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
                    data[key] = deep_merge(data[key], value)
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

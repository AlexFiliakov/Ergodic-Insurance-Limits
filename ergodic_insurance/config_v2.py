"""Enhanced configuration models for the new 3-tier configuration system.

This module provides Pydantic v2 models for the new profiles/modules/presets
configuration architecture with support for inheritance, composition, and validation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

# Import existing config models that we'll extend
try:
    # Try absolute import first (for installed package)
    from ergodic_insurance.config import (
        DebtConfig,
        GrowthConfig,
        LoggingConfig,
        ManufacturerConfig,
        OutputConfig,
        SimulationConfig,
        WorkingCapitalConfig,
    )
except ImportError:
    try:
        # Try relative import (for package context)
        from .config import (
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            ManufacturerConfig,
            OutputConfig,
            SimulationConfig,
            WorkingCapitalConfig,
        )
    except ImportError:
        # Fall back to direct import (for notebooks/scripts)
        from config import (  # type: ignore[no-redef]
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            ManufacturerConfig,
            OutputConfig,
            SimulationConfig,
            WorkingCapitalConfig,
        )


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
    premium_rate: float = Field(gt=0, le=1, description="Premium as percentage of limit")
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

    @model_validator(mode="after")
    def validate_depreciation_allocation(self):
        """Ensure depreciation allocations sum to 100%."""
        total = self.manufacturing_depreciation_allocation + self.admin_depreciation_allocation
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Depreciation allocations must sum to 100%, got {total*100:.1f}%")
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

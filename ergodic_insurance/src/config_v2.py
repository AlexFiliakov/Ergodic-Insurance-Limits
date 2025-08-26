"""Enhanced configuration models for the new 3-tier configuration system.

This module provides Pydantic v2 models for the new profiles/modules/presets
configuration architecture with support for inheritance, composition, and validation.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

# Import existing config models that we'll extend
from ergodic_insurance.src.config import (
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

    @model_validator(mode="after")
    def validate_layer_structure(self):
        """Ensure layer structure is valid.

        Returns:
            Validated layer config.

        Raises:
            ValueError: If layer structure is invalid.
        """
        if self.aggregate_limit and self.aggregate_limit < self.limit:
            raise ValueError(
                f"Aggregate limit {self.aggregate_limit} cannot be less than per-occurrence limit {self.limit}"
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

        presets = {}
        for name, params in data.items():
            presets[name] = PresetConfig(
                preset_name=name,
                preset_type=library_type.lower().replace(" ", "_"),
                description=f"{name} preset for {library_type}",
                parameters=params,
            )

        return cls(
            library_type=library_type,
            description=f"Preset library for {library_type}",
            presets=presets,
        )

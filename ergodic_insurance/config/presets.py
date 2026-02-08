"""Configuration presets, modules, and profile metadata.

Contains classes for the configuration extensibility system: profile metadata
for versioning and inheritance, base module definitions, preset templates,
and the preset library for loading collections from YAML.

Since:
    Version 0.9.0 (Issue #458)
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
import yaml


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

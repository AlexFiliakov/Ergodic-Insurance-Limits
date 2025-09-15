"""Configuration manager for the new 3-tier configuration system.

This module provides the main interface for loading and managing configurations
using profiles, modules, and presets. It implements a modern configuration
architecture that supports inheritance, composition, and runtime overrides.

The configuration system is organized into three tiers:
    1. Profiles: Complete configuration sets (default, conservative, aggressive)
    2. Modules: Reusable components (insurance, losses, stochastic, business)
    3. Presets: Quick-apply templates (market conditions, layer structures)

Example:
    Basic usage of ConfigManager::

        from ergodic_insurance.src.config_manager import ConfigManager

        # Initialize manager
        manager = ConfigManager()

        # Load a profile
        config = manager.load_profile("default")

        # Load with overrides
        config = manager.load_profile(
            "conservative",
            manufacturer={"base_operating_margin": 0.12},
            growth={"annual_growth_rate": 0.08}
        )

        # Apply presets
        config = manager.load_profile(
            "default",
            presets=["hard_market", "high_volatility"]
        )

Note:
    This module replaces the legacy ConfigLoader and provides full backward
    compatibility through the config_compat module.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import warnings

import yaml

try:
    # Try absolute import first (for installed package)
    from ergodic_insurance.src.config_v2 import ConfigV2, PresetLibrary
except ImportError:
    try:
        # Try relative import (for package context)
        from .config_v2 import ConfigV2, PresetLibrary
    except ImportError:
        # Fall back to direct import (for notebooks/scripts)
        from config_v2 import ConfigV2, PresetLibrary


class ConfigManager:
    """Manages configuration loading with profiles, modules, and presets.

    This class provides a comprehensive configuration management system that
    supports profile inheritance, module composition, preset application, and
    runtime parameter overrides. It includes caching for performance and
    validation for correctness.

    Attributes:
        config_dir: Root configuration directory path
        profiles_dir: Directory containing profile configurations
        modules_dir: Directory containing module configurations
        presets_dir: Directory containing preset libraries
        _cache: Internal cache for loaded configurations
        _preset_libraries: Cached preset library definitions

    Example:
        Loading configurations with various options::

            manager = ConfigManager()

            # Simple profile load
            config = manager.load_profile("default")

            # With module selection
            config = manager.load_profile(
                "default",
                modules=["insurance", "stochastic"]
            )

            # With inheritance chain
            config = manager.load_profile("custom/client_abc")
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Root configuration directory. If None, defaults to
                ergodic_insurance/data/config relative to this module.

        Raises:
            FileNotFoundError: If the configuration directory doesn't exist.

        Note:
            The manager expects a specific directory structure with profiles/,
            modules/, and presets/ subdirectories.
        """
        if config_dir is None:
            # Try to find config directory relative to this file
            module_dir = Path(__file__).parent.parent
            config_dir = module_dir / "data" / "config"

        self.config_dir = Path(config_dir)
        self.profiles_dir = self.config_dir / "profiles"
        self.modules_dir = self.config_dir / "modules"
        self.presets_dir = self.config_dir / "presets"

        # Cache for loaded configurations
        self._cache: Dict[str, ConfigV2] = {}
        self._preset_libraries: Dict[str, PresetLibrary] = {}

        # Validate directory structure
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that the configuration directory structure exists."""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")

        if not self.profiles_dir.exists():
            warnings.warn(f"Profiles directory not found: {self.profiles_dir}")

        if not self.modules_dir.exists():
            warnings.warn(f"Modules directory not found: {self.modules_dir}")

        if not self.presets_dir.exists():
            warnings.warn(f"Presets directory not found: {self.presets_dir}")

    def load_profile(
        self, profile_name: str = "default", use_cache: bool = True, **overrides
    ) -> ConfigV2:
        """Load a configuration profile with optional overrides.

        This method loads a configuration profile, applies any inheritance chain,
        includes specified modules, applies presets, and finally applies runtime
        overrides. The result is cached for performance.

        Args:
            profile_name: Name of the profile to load. Can be a simple name
                (e.g., "default") or a path to custom profiles (e.g.,
                "custom/client_abc").
            use_cache: Whether to use cached configurations. Set to False when
                configuration files might have changed during runtime.
            **overrides: Runtime overrides organized by section. Supports:
                - modules: List of module names to include
                - presets: List of preset names to apply
                - Any configuration section with nested parameters

        Returns:
            ConfigV2: Fully loaded, validated, and merged configuration instance.

        Raises:
            FileNotFoundError: If the specified profile doesn't exist.
            ValueError: If configuration validation fails.
            yaml.YAMLError: If YAML parsing fails.

        Example:
            Various ways to load profiles::

                # Basic load
                config = manager.load_profile("default")

                # With overrides
                config = manager.load_profile(
                    "conservative",
                    manufacturer={"base_operating_margin": 0.12},
                    simulation={"time_horizon_years": 50}
                )

                # With presets and modules
                config = manager.load_profile(
                    "default",
                    modules=["insurance", "stochastic"],
                    presets=["hard_market"]
                )
        """

        # Check cache first
        def make_hashable(obj):
            """Convert nested dicts/lists to hashable format."""
            if isinstance(obj, dict):
                return frozenset((k, make_hashable(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            return obj

        cache_key = f"{profile_name}_{hash(make_hashable(overrides))}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Load the profile
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        if not profile_path.exists():
            # Try custom profiles
            custom_path = self.profiles_dir / "custom" / f"{profile_name}.yaml"
            if custom_path.exists():
                profile_path = custom_path
            else:
                available = self.list_profiles()
                raise FileNotFoundError(
                    f"Profile '{profile_name}' not found. Available profiles: {', '.join(available)}"
                )

        # Load with inheritance
        config = self._load_with_inheritance(profile_path)

        # Apply includes (modules)
        if config.profile.includes:
            for module_name in config.profile.includes:
                self._apply_module(config, module_name)

        # Apply presets
        if config.profile.presets:
            for preset_type, preset_name in config.profile.presets.items():
                self._apply_preset(config, preset_type, preset_name)

        # Apply runtime overrides
        if overrides:
            config = config.with_overrides(**overrides)

        # Validate completeness
        issues = config.validate_completeness()
        if issues:
            warnings.warn(f"Configuration issues: {', '.join(issues)}")

        # Cache the result
        if use_cache:
            self._cache[cache_key] = config

        return config

    def _load_with_inheritance(self, profile_path: Path) -> ConfigV2:
        """Load a profile with inheritance support.

        Args:
            profile_path: Path to the profile file.

        Returns:
            Loaded ConfigV2 with inheritance applied.
        """
        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)

        # Remove YAML anchors
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        # Handle inheritance
        if "profile" in data and "extends" in data["profile"] and data["profile"]["extends"]:
            parent_name = data["profile"]["extends"]
            parent_path = self.profiles_dir / f"{parent_name}.yaml"

            if not parent_path.exists():
                parent_path = self.profiles_dir / "custom" / f"{parent_name}.yaml"

            if parent_path.exists():
                parent_config = self._load_with_inheritance(parent_path)
                parent_data = parent_config.model_dump()

                # Deep merge parent with child
                merged_data = self._deep_merge(parent_data, data)
                data = merged_data
            else:
                warnings.warn(f"Parent profile '{parent_name}' not found")

        return ConfigV2(**data)

    def _apply_module(self, config: ConfigV2, module_name: str) -> None:
        """Apply a configuration module to a config.

        Args:
            config: Configuration to modify.
            module_name: Name of the module to apply.
        """
        module_path = self.modules_dir / f"{module_name}.yaml"
        if not module_path.exists():
            warnings.warn(f"Module '{module_name}' not found")
            return

        with open(module_path, "r") as f:
            module_data = yaml.safe_load(f)

        # Apply module data to config
        for key, value in module_data.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    current = getattr(config, key)
                    if current is None:
                        # Create new instance if field is None
                        from ergodic_insurance.src.config_v2 import (
                            InsuranceConfig,
                            LossDistributionConfig,
                        )

                        # Map key names to config classes
                        field_mapping = {
                            "insurance": InsuranceConfig,
                            "losses": LossDistributionConfig,
                        }

                        if key in field_mapping:
                            field_class = field_mapping[key]
                            setattr(config, key, field_class(**value))
                    elif hasattr(current, "model_dump"):
                        # Update Pydantic model
                        updated = current.model_dump()
                        updated = self._deep_merge(updated, value)
                        setattr(config, key, type(current)(**updated))
                    else:
                        setattr(config, key, value)
                else:
                    setattr(config, key, value)

    def _apply_preset(self, config: ConfigV2, preset_type: str, preset_name: str) -> None:
        """Apply a preset to a configuration.

        Args:
            config: Configuration to modify.
            preset_type: Type of preset (e.g., 'market', 'layers').
            preset_name: Name of the specific preset.
        """
        # Load preset library if not cached
        library_key = preset_type
        if library_key not in self._preset_libraries:
            preset_file = self.presets_dir / f"{preset_type}.yaml"
            if not preset_file.exists():
                # Try with underscores
                preset_file = self.presets_dir / f"{preset_type.replace('-', '_')}.yaml"

            if not preset_file.exists():
                warnings.warn(f"Preset library '{preset_type}' not found")
                return

            with open(preset_file, "r") as f:
                library_data = yaml.safe_load(f)
                self._preset_libraries[library_key] = library_data
        else:
            library_data = self._preset_libraries[library_key]

        # Get the specific preset
        if preset_name not in library_data:
            available = list(library_data.keys())
            warnings.warn(
                f"Preset '{preset_name}' not found in {preset_type}. "
                f"Available: {', '.join(available)}"
            )
            return

        preset_data = library_data[preset_name]

        # Apply preset data
        config.apply_preset(f"{preset_type}:{preset_name}", preset_data)

    def with_preset(self, config: ConfigV2, preset_type: str, preset_name: str) -> ConfigV2:
        """Create a new configuration with a preset applied.

        Args:
            config: Base configuration.
            preset_type: Type of preset.
            preset_name: Name of the preset.

        Returns:
            New ConfigV2 instance with preset applied.
        """
        # Create a copy
        new_config = ConfigV2(**config.model_dump())

        # Apply the preset
        self._apply_preset(new_config, preset_type, preset_name)

        return new_config

    def with_overrides(self, config: ConfigV2, **overrides) -> ConfigV2:
        """Create a new configuration with runtime overrides.

        Args:
            config: Base configuration.
            **overrides: Override parameters.

        Returns:
            New ConfigV2 instance with overrides applied.
        """
        return config.with_overrides(**overrides)

    def validate(self, config: ConfigV2) -> List[str]:
        """Validate a configuration for completeness and consistency.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation issues, empty if valid.
        """
        issues: List[str] = config.validate_completeness()

        # Additional validation logic
        if config.simulation.time_horizon_years > 1000:
            issues.append(
                f"Time horizon {config.simulation.time_horizon_years} years may be too long"
            )

        if config.manufacturer.base_operating_margin > 0.5:
            issues.append(
                f"Base operating margin {config.manufacturer.base_operating_margin} seems unrealistic"
            )

        return issues

    def list_profiles(self) -> List[str]:
        """List all available configuration profiles.

        Returns:
            List of profile names.
        """
        profiles = []

        # Check standard profiles
        if self.profiles_dir.exists():
            for path in self.profiles_dir.glob("*.yaml"):
                if path.stem != "README":
                    profiles.append(path.stem)

        # Check custom profiles
        custom_dir = self.profiles_dir / "custom"
        if custom_dir.exists():
            for path in custom_dir.glob("*.yaml"):
                profiles.append(f"custom/{path.stem}")

        return sorted(profiles)

    def list_modules(self) -> List[str]:
        """List all available configuration modules.

        Returns:
            List of module names.
        """
        modules = []

        if self.modules_dir.exists():
            for path in self.modules_dir.glob("*.yaml"):
                if path.stem != "README":
                    modules.append(path.stem)

        return sorted(modules)

    def list_presets(self) -> Dict[str, List[str]]:
        """List all available presets by type.

        Returns:
            Dictionary mapping preset types to list of preset names.
        """
        presets = {}

        if self.presets_dir.exists():
            for path in self.presets_dir.glob("*.yaml"):
                if path.stem != "README":
                    preset_type = path.stem
                    with open(path, "r") as f:
                        library_data = yaml.safe_load(f)
                        presets[preset_type] = list(library_data.keys())

        return presets

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        self._preset_libraries.clear()

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @lru_cache(maxsize=32)
    def get_profile_metadata(self, profile_name: str) -> Dict[str, Any]:
        """Get metadata for a profile without loading the full configuration.

        Args:
            profile_name: Name of the profile.

        Returns:
            Profile metadata dictionary.
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        if not profile_path.exists():
            profile_path = self.profiles_dir / "custom" / f"{profile_name}.yaml"

        if not profile_path.exists():
            return {}

        with open(profile_path, "r") as f:
            data = yaml.safe_load(f)

        return data.get("profile", {})  # type: ignore

    def create_profile(
        self,
        name: str,
        description: str,
        base_profile: str = "default",
        custom: bool = True,
        **config_params,
    ) -> Path:
        """Create a new configuration profile.

        Args:
            name: Profile name.
            description: Profile description.
            base_profile: Profile to extend from.
            custom: Whether to save as custom profile.
            **config_params: Configuration parameters.

        Returns:
            Path to the created profile file.
        """
        # Load base profile
        base_config = self.load_profile(base_profile)

        # Create new profile data
        profile_data = {
            "profile": {
                "name": name,
                "description": description,
                "extends": base_profile,
                "version": "2.0.0",
            }
        }

        # Add configuration parameters
        profile_data.update(config_params)

        # Determine save path
        if custom:
            save_dir = self.profiles_dir / "custom"
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.profiles_dir

        save_path = save_dir / f"{name}.yaml"

        # Save the profile
        with open(save_path, "w") as f:
            yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)

        return save_path

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

        from ergodic_insurance.config_manager import ConfigManager

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
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import warnings

import yaml

logger = logging.getLogger(__name__)

# Pattern for safe names: alphanumeric, hyphens, and underscores only
_SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

try:
    # Try absolute import first (for installed package)
    from ergodic_insurance.config import Config, PresetLibrary
    from ergodic_insurance.config.utils import deep_merge as _deep_merge_fn
except ImportError:
    try:
        # Try relative import (for package context)
        from .config import Config, PresetLibrary
        from .config.utils import deep_merge as _deep_merge_fn
    except ImportError:
        # Fall back to direct import (for notebooks/scripts)
        from config import Config, PresetLibrary  # type: ignore[no-redef]
        from config.utils import deep_merge as _deep_merge_fn  # type: ignore[no-redef]


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
            # Use the ergodic_insurance/data/config directory
            module_dir = Path(__file__).parent
            config_dir = module_dir / "data" / "config"

        self.config_dir = Path(config_dir)
        self.profiles_dir = self.config_dir / "profiles"
        self.modules_dir = self.config_dir / "modules"
        self.presets_dir = self.config_dir / "presets"

        # Cache for loaded configurations
        self._cache: Dict[str, Config] = {}
        self._preset_libraries: Dict[str, PresetLibrary] = {}

        # Validate directory structure
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that the configuration directory structure exists."""
        if not self.config_dir.exists():
            # For backward compatibility, create the directory if it doesn't exist
            logger.warning(f"Configuration directory not found, creating: {self.config_dir}")
            self.config_dir.mkdir(parents=True, exist_ok=True)

        if not self.profiles_dir.exists():
            logger.debug(f"Profiles directory not found: {self.profiles_dir}")

        if not self.modules_dir.exists():
            logger.debug(f"Modules directory not found: {self.modules_dir}")

        if not self.presets_dir.exists():
            logger.debug(f"Presets directory not found: {self.presets_dir}")

    @staticmethod
    def _validate_name(name: str, *, allow_slashes: bool = False) -> None:
        """Validate that a name contains only safe characters.

        Args:
            name: The name to validate.
            allow_slashes: If True, allow forward slashes for subdirectory paths
                (e.g., ``custom/client_abc``).

        Raises:
            ValueError: If the name contains unsafe characters.
        """
        if allow_slashes:
            parts = name.replace("\\", "/").split("/")
            for part in parts:
                if not _SAFE_NAME_PATTERN.match(part):
                    raise ValueError(
                        f"Invalid name '{name}': each path component must contain "
                        "only alphanumeric characters, hyphens, and underscores"
                    )
        else:
            if not _SAFE_NAME_PATTERN.match(name):
                raise ValueError(
                    f"Invalid name '{name}': must contain only alphanumeric "
                    "characters, hyphens, and underscores"
                )

    @staticmethod
    def _validate_path_containment(path: Path, allowed_parent: Path) -> None:
        """Verify that *path* resolves to a location inside *allowed_parent*.

        Args:
            path: The path to validate.
            allowed_parent: The directory that must contain the path.

        Raises:
            ValueError: If the path resolves outside the allowed parent.
        """
        try:
            path.resolve().relative_to(allowed_parent.resolve())
        except ValueError:
            raise ValueError(
                f"Path traversal detected: resolved path is outside "
                f"the allowed directory '{allowed_parent}'"
            )

    def load_profile(
        self, profile_name: str = "default", use_cache: bool = True, **overrides
    ) -> Config:
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
            Config: Fully loaded, validated, and merged configuration instance.

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

        # Validate profile name against path traversal
        self._validate_name(profile_name, allow_slashes=True)

        # Check cache first
        def make_hashable(obj):
            """Convert nested dicts/lists to hashable format."""
            if isinstance(obj, dict):
                return frozenset((k, make_hashable(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            return obj

        cache_key = f"{profile_name}_{hashlib.sha256(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:16]}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Load the profile
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        self._validate_path_containment(profile_path, self.profiles_dir)
        if not profile_path.exists():
            # Try custom profiles
            custom_path = self.profiles_dir / "custom" / f"{profile_name}.yaml"
            self._validate_path_containment(custom_path, self.profiles_dir)
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
        if config.profile and config.profile.includes:
            for module_name in config.profile.includes:
                config = self._apply_module(config, module_name)

        # Apply presets
        if config.profile and config.profile.presets:
            for preset_type, preset_name in config.profile.presets.items():
                config = self._apply_preset(config, preset_type, preset_name)

        # Apply runtime overrides
        if overrides:
            config = config.with_overrides(overrides)

        # Validate completeness
        issues = config.validate_completeness()
        if issues:
            warnings.warn(f"Configuration issues: {', '.join(issues)}")

        # Cache the result
        if use_cache:
            self._cache[cache_key] = config

        return config

    def _load_with_inheritance(
        self, profile_path: Path, _visited: frozenset | None = None
    ) -> Config:
        """Load a profile with inheritance support.

        Args:
            profile_path: Path to the profile file.
            _visited: Internal set of already-visited profile paths for
                cycle detection.  Callers should not pass this argument.

        Returns:
            Loaded Config with inheritance applied.

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

        # Remove YAML anchors
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        # Handle inheritance
        if "profile" in data and "extends" in data["profile"] and data["profile"]["extends"]:
            parent_name = data["profile"]["extends"]
            self._validate_name(parent_name, allow_slashes=True)
            parent_path = self.profiles_dir / f"{parent_name}.yaml"
            self._validate_path_containment(parent_path, self.profiles_dir)

            if not parent_path.exists():
                parent_path = self.profiles_dir / "custom" / f"{parent_name}.yaml"
                self._validate_path_containment(parent_path, self.profiles_dir)

            if parent_path.exists():
                parent_config = self._load_with_inheritance(parent_path, _visited=visited)
                parent_data = parent_config.model_dump()

                # Deep merge parent with child
                merged_data = self._deep_merge(parent_data, data)
                data = merged_data
            else:
                warnings.warn(f"Parent profile '{parent_name}' not found")

        return Config(**data)

    def _apply_module(self, config: Config, module_name: str) -> Config:
        """Apply a configuration module, returning a new Config.

        Args:
            config: Base configuration (not mutated).
            module_name: Name of the module to apply.

        Returns:
            New Config with the module applied, or the original config
            unchanged if the module file is not found.
        """
        self._validate_name(module_name)
        module_path = self.modules_dir / f"{module_name}.yaml"
        self._validate_path_containment(module_path, self.modules_dir)
        if not module_path.exists():
            warnings.warn(f"Module '{module_name}' not found")
            return config

        return config.with_module(module_path)

    def _apply_preset(self, config: Config, preset_type: str, preset_name: str) -> Config:
        """Apply a preset, returning a new Config.

        Args:
            config: Base configuration (not mutated).
            preset_type: Type of preset (e.g., 'market', 'layers').
            preset_name: Name of the specific preset.

        Returns:
            New Config with the preset applied, or the original config
            unchanged if the preset library/name is not found.
        """
        self._validate_name(preset_type)

        # Load preset library if not cached
        library_key = preset_type
        if library_key not in self._preset_libraries:
            preset_file = self.presets_dir / f"{preset_type}.yaml"
            self._validate_path_containment(preset_file, self.presets_dir)
            if not preset_file.exists():
                # Try with underscores
                preset_file = self.presets_dir / f"{preset_type.replace('-', '_')}.yaml"
                self._validate_path_containment(preset_file, self.presets_dir)

            if not preset_file.exists():
                warnings.warn(f"Preset library '{preset_type}' not found")
                return config

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
            return config

        preset_data = library_data[preset_name]

        # Apply preset data — return new instance
        return config.with_preset(f"{preset_type}:{preset_name}", preset_data)

    def with_preset(self, config: Config, preset_type: str, preset_name: str) -> Config:
        """Create a new configuration with a preset applied.

        Args:
            config: Base configuration (not mutated).
            preset_type: Type of preset.
            preset_name: Name of the preset.

        Returns:
            New Config instance with preset applied.
        """
        return self._apply_preset(config, preset_type, preset_name)

    def with_overrides(self, config: Config, overrides: Dict[str, Any]) -> Config:
        """Create a new configuration with runtime overrides.

        Args:
            config: Base configuration.
            overrides: Override parameters as a dictionary with dot-notation
                keys or section-level dictionaries.

        Returns:
            New Config instance with overrides applied.
        """
        return config.with_overrides(overrides)

    def validate(self, config: Config) -> List[str]:
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
        return _deep_merge_fn(base, override)

    @lru_cache(maxsize=32)
    def get_profile_metadata(self, profile_name: str) -> Dict[str, Any]:
        """Get metadata for a profile without loading the full configuration.

        Args:
            profile_name: Name of the profile.

        Returns:
            Profile metadata dictionary.
        """
        self._validate_name(profile_name, allow_slashes=True)
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        self._validate_path_containment(profile_path, self.profiles_dir)
        if not profile_path.exists():
            profile_path = self.profiles_dir / "custom" / f"{profile_name}.yaml"
            self._validate_path_containment(profile_path, self.profiles_dir)

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
        # Validate profile name
        self._validate_name(name)

        # Load base profile
        _base_config = self.load_profile(base_profile)

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
        self._validate_path_containment(save_path, self.profiles_dir)

        # Save the profile
        with open(save_path, "w") as f:
            yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)

        return save_path

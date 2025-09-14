"""Backward compatibility layer for the legacy configuration system.

This module provides adapters and shims to ensure existing code continues
to work while transitioning to the new 3-tier configuration system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import yaml

try:
    # Try absolute import first (for installed package)
    from ergodic_insurance.src.config import Config
    from ergodic_insurance.src.config_manager import ConfigManager
    from ergodic_insurance.src.config_v2 import ConfigV2
except ImportError:
    try:
        # Try relative import (for package context)
        from .config import Config
        from .config_manager import ConfigManager
        from .config_v2 import ConfigV2
    except ImportError:
        # Fall back to direct import (for notebooks/scripts)
        from config import Config
        from config_manager import ConfigManager
        from config_v2 import ConfigV2


class LegacyConfigAdapter:
    """Adapter to support old ConfigLoader interface using new ConfigManager."""

    def __init__(self):
        """Initialize the legacy adapter."""
        self.config_manager = ConfigManager()
        self._profile_mapping = {
            "baseline": "default",
            "conservative": "conservative",
            "optimistic": "aggressive",
            "aggressive": "aggressive",
        }
        self._deprecated_warning_shown = False

    def load(
        self,
        config_name: str = "baseline",
        override_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Config:
        """Load configuration using legacy interface.

        Args:
            config_name: Legacy configuration name.
            override_params: Dictionary of override parameters.
            **kwargs: Additional override parameters.

        Returns:
            Config object for backward compatibility.
        """
        # Show deprecation warning once
        if not self._deprecated_warning_shown:
            warnings.warn(
                "ConfigLoader is deprecated and will be removed in version 3.0.0. "
                "Please migrate to ConfigManager.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._deprecated_warning_shown = True

        # Map legacy config names to new profiles
        profile_name = self._profile_mapping.get(config_name, config_name)

        # Combine overrides
        overrides = {}
        if override_params:
            overrides.update(self._flatten_dict(override_params))
        overrides.update(kwargs)

        # Load using new system
        try:
            config_v2 = self.config_manager.load_profile(profile_name, **overrides)

            # Convert to legacy Config format
            return self._convert_to_legacy(config_v2)

        except FileNotFoundError:
            # Fall back to loading from legacy location
            return self._load_legacy_direct(config_name, overrides)

    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_name: str = "baseline",
        **overrides,
    ) -> Config:
        """Alternative legacy loading method.

        Args:
            config_path: Path to configuration file (ignored, for compatibility).
            config_name: Configuration name.
            **overrides: Override parameters.

        Returns:
            Config object.
        """
        return self.load(config_name, override_params=overrides)

    def _convert_to_legacy(self, config_v2: ConfigV2) -> Config:
        """Convert ConfigV2 to legacy Config format.

        Args:
            config_v2: New format configuration.

        Returns:
            Legacy format Config object.
        """
        # Extract the sections needed for legacy Config
        try:
            # Try absolute import first (for installed package)
            from ergodic_insurance.src.config import (
                Config,
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
                    Config,
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
                from config import (
                    Config,
                    DebtConfig,
                    GrowthConfig,
                    LoggingConfig,
                    ManufacturerConfig,
                    OutputConfig,
                    SimulationConfig,
                    WorkingCapitalConfig,
                )

        return Config(
            manufacturer=ManufacturerConfig(**config_v2.manufacturer.model_dump()),
            working_capital=WorkingCapitalConfig(**config_v2.working_capital.model_dump()),
            growth=GrowthConfig(**config_v2.growth.model_dump()),
            debt=DebtConfig(**config_v2.debt.model_dump()),
            simulation=SimulationConfig(**config_v2.simulation.model_dump()),
            output=OutputConfig(**config_v2.output.model_dump()),
            logging=LoggingConfig(**config_v2.logging.model_dump()),
        )

    def _load_legacy_direct(self, config_name: str, overrides: Dict[str, Any]) -> Config:
        """Load configuration directly from legacy location.

        Args:
            config_name: Legacy configuration name.
            overrides: Override parameters.

        Returns:
            Config object.
        """
        # Try to find legacy config file
        # Use absolute path based on current module location
        module_path = Path(__file__).parent.parent
        legacy_dir = module_path / "data" / "parameters"
        config_file = legacy_dir / f"{config_name}.yaml"

        if not config_file.exists():
            # Try without .yaml extension
            config_file = legacy_dir / config_name

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration '{config_name}' not found in legacy or new locations"
            )

        # Load the legacy config
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        # Handle empty or invalid YAML files
        if data is None:
            data = {}

        # Remove YAML anchors
        data = {k: v for k, v in data.items() if not k.startswith("_")}

        # Apply overrides
        for key, value in overrides.items():
            if "__" in key:
                # Handle nested keys
                parts = key.split("__")
                current = data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                data[key] = value

        return Config(**data)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, str]:
        """Flatten nested dictionary to support __ notation.

        Args:
            d: Dictionary to flatten.
            parent_key: Parent key for recursion.

        Returns:
            Flattened dictionary.
        """
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}__{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)


# Global adapter instance for drop-in replacement
_adapter = LegacyConfigAdapter()


def load_config(
    config_name: str = "baseline", override_params: Optional[Dict[str, Any]] = None, **kwargs
) -> Config:
    """Legacy function interface for loading configurations.

    Args:
        config_name: Configuration name.
        override_params: Override parameters.
        **kwargs: Additional overrides.

    Returns:
        Config object.
    """
    return _adapter.load(config_name, override_params, **kwargs)


def migrate_config_usage(file_path: Path) -> None:
    """Helper to migrate old config usage in a Python file.

    Args:
        file_path: Path to Python file to migrate.
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Track if changes were made
    original_content = content

    # Replace imports
    content = content.replace(
        "from ergodic_insurance.src.config_loader import ConfigLoader",
        "from ergodic_insurance.src.config_manager import ConfigManager",
    )
    content = content.replace(
        "from ergodic_insurance.src.config_loader import load_config",
        "from ergodic_insurance.src.config_compat import load_config  # TODO: Migrate to ConfigManager",
    )

    # Replace ConfigLoader usage
    content = content.replace("ConfigLoader()", "ConfigManager()")
    content = content.replace("ConfigLoader.load(", "ConfigManager().load_profile(")

    # Save if changes were made
    if content != original_content:
        # Create backup
        backup_path = file_path.with_suffix(".bak")
        with open(backup_path, "w") as f:
            f.write(original_content)

        # Write updated content
        with open(file_path, "w") as f:
            f.write(content)

        print(f"✓ Migrated {file_path}")
        print(f"  Backup saved to {backup_path}")
    else:
        print(f"  No changes needed for {file_path}")


class ConfigTranslator:
    """Utilities for translating between old and new configuration formats."""

    @staticmethod
    def legacy_to_v2(legacy_config: Config) -> Dict[str, Any]:
        """Convert legacy Config to ConfigV2 format.

        Args:
            legacy_config: Legacy configuration object.

        Returns:
            Dictionary suitable for ConfigV2 initialization.
        """
        v2_data = {
            "profile": {
                "name": "migrated",
                "description": "Migrated from legacy configuration",
                "version": "2.0.0",
            }
        }

        # Convert each section
        v2_data.update(legacy_config.model_dump())

        return v2_data

    @staticmethod
    def v2_to_legacy(config_v2: ConfigV2) -> Dict[str, Any]:
        """Convert ConfigV2 to legacy Config format.

        Args:
            config_v2: New format configuration.

        Returns:
            Dictionary suitable for legacy Config initialization.
        """
        # Extract only the sections that exist in legacy Config
        legacy_sections = [
            "manufacturer",
            "working_capital",
            "growth",
            "debt",
            "simulation",
            "output",
            "logging",
        ]

        legacy_data = {}
        for section in legacy_sections:
            if hasattr(config_v2, section):
                value = getattr(config_v2, section)
                if value is not None:
                    legacy_data[section] = (
                        value.model_dump() if hasattr(value, "model_dump") else value
                    )

        return legacy_data

    @staticmethod
    def validate_translation(
        original: Union[Config, ConfigV2], translated: Union[Config, ConfigV2]
    ) -> bool:
        """Validate that translation preserved essential data.

        Args:
            original: Original configuration.
            translated: Translated configuration.

        Returns:
            True if translation is valid.
        """
        # Check critical fields
        critical_fields = [
            ("manufacturer", "initial_assets"),
            ("simulation", "time_horizon_years"),
            ("growth", "annual_growth_rate"),
        ]

        for section, field in critical_fields:
            if hasattr(original, section) and hasattr(translated, section):
                orig_section = getattr(original, section)
                trans_section = getattr(translated, section)

                if orig_section and trans_section:
                    orig_value = getattr(orig_section, field, None)
                    trans_value = getattr(trans_section, field, None)

                    if orig_value != trans_value:
                        warnings.warn(
                            f"Translation mismatch: {section}.{field} "
                            f"({orig_value} != {trans_value})"
                        )
                        return False

        return True

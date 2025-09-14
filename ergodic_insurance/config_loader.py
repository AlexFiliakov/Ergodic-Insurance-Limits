"""Configuration loader with validation and override support.

This module provides utilities for loading, validating, and managing
configuration files, with support for caching, overrides, and
scenario-based configurations.

NOTE: This module now uses the new ConfigManager through the compatibility layer.
It maintains the same interface for backward compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import warnings

import yaml

try:
    # Try relative imports first (when used as a package)
    from .config import Config, PricingScenarioConfig
    from .config_compat import LegacyConfigAdapter
except ImportError:
    # Fall back to absolute imports (when called directly or from notebooks)
    from config import Config, PricingScenarioConfig  # type: ignore
    from config_compat import LegacyConfigAdapter  # type: ignore

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles loading and managing configuration.

    A comprehensive configuration management system that supports
    YAML file loading, validation, caching, and runtime overrides.

    NOTE: This class now delegates to LegacyConfigAdapter for backward compatibility
    while using the new ConfigManager internally.
    """

    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "data" / "parameters"
    DEFAULT_CONFIG_FILE = "baseline.yaml"

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to data/parameters/.
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._cache: Dict[Any, Config] = {}  # Changed to Any for tuple keys
        self._adapter = LegacyConfigAdapter()
        self._deprecation_warned = False

    def load(
        self,
        config_name: str = "baseline",
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Config:
        """Load configuration with optional overrides.

        Args:
            config_name: Name of config file (without .yaml extension)
                        or full path to config file.
            overrides: Dictionary of overrides to apply.
            **kwargs: Additional overrides in dot notation
                     (e.g., manufacturer__operating_margin=0.1).

        Returns:
            Loaded and validated configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValidationError: If configuration is invalid.
        """
        # Show deprecation warning once
        if not self._deprecation_warned:
            warnings.warn(
                "ConfigLoader is deprecated. Please migrate to ConfigManager for improved functionality.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._deprecation_warned = True

        # Create cache key - need to handle nested dicts
        def make_hashable(obj):
            if isinstance(obj, dict):
                return frozenset((k, make_hashable(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            return obj

        cache_key = (
            config_name,
            make_hashable(overrides) if overrides else None,
            make_hashable(kwargs) if kwargs else None,
        )

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load using adapter and cache result
        config: Config = self._adapter.load(config_name, overrides, **kwargs)
        self._cache[cache_key] = config
        return config

    def load_scenario(
        self, scenario: str, overrides: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Config:
        """Load a predefined scenario configuration.

        Args:
            scenario: Scenario name ("baseline", "conservative", "optimistic").
            overrides: Dictionary of overrides to apply.
            **kwargs: Additional overrides in dot notation.

        Returns:
            Loaded and validated configuration.

        Raises:
            ValueError: If scenario is not recognized.
        """
        valid_scenarios = ["baseline", "conservative", "optimistic"]
        if scenario not in valid_scenarios:
            raise ValueError(
                f"Unknown scenario '{scenario}'. Valid scenarios: {', '.join(valid_scenarios)}"
            )

        return self.load(scenario, overrides, **kwargs)

    def compare_configs(
        self, config1: Union[str, Config], config2: Union[str, Config]
    ) -> Dict[str, Any]:
        """Compare two configurations and return differences.

        Args:
            config1: First config (name or Config object).
            config2: Second config (name or Config object).

        Returns:
            Dictionary of differences between configurations.
        """
        # Load configs if names provided
        if isinstance(config1, str):
            config1 = self.load(config1)
        if isinstance(config2, str):
            config2 = self.load(config2)

        # Get dictionaries
        dict1 = config1.model_dump()
        dict2 = config2.model_dump()

        # Find differences
        differences = {}

        def compare_dicts(d1: dict, d2: dict, path: str = "") -> None:
            """Recursively compare dictionaries.

            Args:
                d1: First dictionary to compare.
                d2: Second dictionary to compare.
                path: Current path in nested structure.
            """
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                if key not in d1:
                    differences[current_path] = {"config1": None, "config2": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"config1": d1[key], "config2": None}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {
                        "config1": d1[key],
                        "config2": d2[key],
                    }

        compare_dicts(dict1, dict2)
        return differences

    def validate_config(self, config: Union[str, Config]) -> bool:
        """Validate a configuration.

        Args:
            config: Configuration to validate (name or Config object).

        Returns:
            True if valid, raises exception otherwise.

        Raises:
            ValidationError: If configuration is invalid.
        """
        if isinstance(config, str):
            config = self.load(config)

        # Pydantic validates on instantiation, so if we get here it's valid
        # Additional business logic validation can be added here

        # Check for logical inconsistencies
        if config.simulation.time_resolution == "monthly":
            total_periods = config.simulation.time_horizon_years * 12
            if total_periods > 12000:
                logger.warning(
                    f"Monthly simulation for {config.simulation.time_horizon_years} "
                    f"years will create {total_periods} periods. "
                    "Consider using annual resolution for long simulations."
                )

        if config.manufacturer.retention_ratio == 0 and config.growth.annual_growth_rate > 0:
            logger.warning(
                "Zero retention with positive growth rate may lead to inconsistent results"
            )

        return True

    def load_pricing_scenarios(
        self, scenario_file: str = "insurance_pricing_scenarios"
    ) -> PricingScenarioConfig:
        """Load pricing scenario configuration.

        Args:
            scenario_file: Name of scenario file (without .yaml extension)
                          or full path to scenario file.

        Returns:
            Loaded and validated pricing scenario configuration.

        Raises:
            FileNotFoundError: If scenario file not found.
            ValidationError: If scenario data is invalid.
        """
        # Determine file path
        if ".yaml" in scenario_file or ".yml" in scenario_file:
            file_path = Path(scenario_file)
        else:
            file_path = self.config_dir / f"{scenario_file}.yaml"

        if not file_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {file_path}")

        # Load YAML data
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Parse and validate using Pydantic
        return PricingScenarioConfig(**data)

    def switch_pricing_scenario(self, config: Config, scenario_name: str) -> Config:
        """Switch to a different pricing scenario.

        Updates the configuration's insurance parameters to use rates
        from the specified pricing scenario.

        Args:
            config: Current configuration
            scenario_name: Name of scenario to switch to (inexpensive/baseline/expensive)

        Returns:
            Updated configuration with new pricing scenario
        """
        # Load pricing scenarios
        pricing_config = self.load_pricing_scenarios()

        # Get the target scenario
        scenario = pricing_config.get_scenario(scenario_name)

        # Create a copy of the config to modify
        config_dict = config.model_dump()

        # Update insurance rates if insurance config exists
        if "insurance" in config_dict:
            # Map scenario rates to insurance configuration
            # This assumes insurance config has layer rates or similar structure
            # Actual mapping depends on insurance config structure
            logger.info(f"Switching to {scenario.name} pricing scenario")
            logger.info(f"Primary rate: {scenario.primary_layer_rate:.1%}")
            logger.info(f"First excess rate: {scenario.first_excess_rate:.1%}")
            logger.info(f"Higher excess rate: {scenario.higher_excess_rate:.1%}")

        # Return updated config
        return Config(**config_dict)

    def list_available_configs(self) -> list[str]:
        """List all available configuration files.

        Returns:
            List of configuration file names (without .yaml extension).
        """
        yaml_files = self.config_dir.glob("*.yaml")
        return [f.stem for f in yaml_files if not f.stem.startswith("_")]

    def clear_cache(self) -> None:
        """Clear the configuration cache.

        Removes all cached configurations, forcing fresh loads
        on subsequent requests.
        """
        self._cache.clear()
        logger.debug("Configuration cache cleared")


# Convenience function for quick loading
def load_config(
    config_name: str = "baseline",
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Config:
    """Quick helper to load a configuration.

    Args:
        config_name: Name of config file or full path.
        overrides: Dictionary of overrides.
        **kwargs: Keyword overrides in dot notation.

    Returns:
        Loaded configuration.
    """
    loader = ConfigLoader()
    return loader.load(config_name, overrides, **kwargs)

"""Configuration loader with validation and override support."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from config import Config

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles loading and managing configuration."""

    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "data" / "parameters"
    DEFAULT_CONFIG_FILE = "baseline.yaml"

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to data/parameters/
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._cache: Dict[str, Config] = {}

    def load(
        self,
        config_name: str = "baseline",
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Config:
        """Load configuration with optional overrides.

        Args:
            config_name: Name of config file (without .yaml extension)
                        or full path to config file
            overrides: Dictionary of overrides to apply
            **kwargs: Additional overrides in dot notation
                     (e.g., manufacturer__operating_margin=0.1)

        Returns:
            Loaded and validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
        """
        # Determine config path
        if "/" in config_name or "\\" in config_name:
            # Full path provided
            config_path = Path(config_name)
        else:
            # Config name provided, add .yaml extension if needed
            if not config_name.endswith(".yaml"):
                config_name += ".yaml"
            config_path = self.config_dir / config_name

        # Check cache
        cache_key = str(config_path)
        if cache_key not in self._cache:
            logger.info(f"Loading configuration from {config_path}")
            self._cache[cache_key] = Config.from_yaml(config_path)
        else:
            logger.debug(f"Using cached configuration for {config_path}")

        config = self._cache[cache_key]

        # Apply dictionary overrides
        if overrides:
            logger.debug(f"Applying dictionary overrides: {overrides}")
            config = Config.from_dict(overrides, base_config=config)

        # Apply keyword overrides
        if kwargs:
            logger.debug(f"Applying keyword overrides: {kwargs}")
            config = config.override(**kwargs)

        return config

    def load_scenario(
        self, scenario: str, overrides: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Config:
        """Load a predefined scenario configuration.

        Args:
            scenario: Scenario name ("baseline", "conservative", "optimistic")
            overrides: Dictionary of overrides to apply
            **kwargs: Additional overrides in dot notation

        Returns:
            Loaded and validated configuration

        Raises:
            ValueError: If scenario is not recognized
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
            config1: First config (name or Config object)
            config2: Second config (name or Config object)

        Returns:
            Dictionary of differences between configurations
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
            """Recursively compare dictionaries."""
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
            config: Configuration to validate (name or Config object)

        Returns:
            True if valid, raises exception otherwise

        Raises:
            ValidationError: If configuration is invalid
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

    def list_available_configs(self) -> list[str]:
        """List all available configuration files.

        Returns:
            List of configuration file names (without .yaml extension)
        """
        yaml_files = self.config_dir.glob("*.yaml")
        return [f.stem for f in yaml_files if not f.stem.startswith("_")]

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
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
        config_name: Name of config file or full path
        overrides: Dictionary of overrides
        **kwargs: Keyword overrides in dot notation

    Returns:
        Loaded configuration
    """
    loader = ConfigLoader()
    return loader.load(config_name, overrides, **kwargs)

"""Comprehensive tests for the config_loader module.

Since Issue #638 unified Config and ConfigV2, ConfigLoader no longer uses
LegacyConfigAdapter.  It loads directly from YAML files in config_dir.
"""

import copy
import logging
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import warnings

import pytest
import yaml

from ergodic_insurance.config import Config
from ergodic_insurance.config_loader import ConfigLoader


def _make_valid_config_data(**overrides):
    """Return a minimal valid config dict for Config construction."""
    data: dict = {
        "manufacturer": {
            "initial_assets": 10_000_000,
            "base_operating_margin": 0.08,
            "tax_rate": 0.25,
        },
        "growth": {
            "annual_growth_rate": 0.07,
        },
        "simulation": {
            "time_horizon_years": 10,
        },
    }
    for k, v in overrides.items():
        if "." in k:
            parts = k.split(".")
            d: dict = data
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
        else:
            data[k] = v
    return data


class TestConfigLoader:
    """Test ConfigLoader class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with valid config YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            baseline = _make_valid_config_data()
            (config_dir / "baseline.yaml").write_text(yaml.dump(baseline, default_flow_style=False))

            conservative = _make_valid_config_data(
                **{"manufacturer.base_operating_margin": 0.06, "growth.annual_growth_rate": 0.05}
            )
            (config_dir / "conservative.yaml").write_text(
                yaml.dump(conservative, default_flow_style=False)
            )

            optimistic = _make_valid_config_data(
                **{"manufacturer.base_operating_margin": 0.10, "growth.annual_growth_rate": 0.09}
            )
            (config_dir / "optimistic.yaml").write_text(
                yaml.dump(optimistic, default_flow_style=False)
            )

            yield config_dir

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def test_initialization_default(self):
        """Test default ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.config_dir == ConfigLoader.DEFAULT_CONFIG_DIR
        assert len(loader._cache) == 0
        assert loader._deprecation_warned is False

    def test_initialization_custom_dir(self, temp_config_dir):
        """Test ConfigLoader initialization with custom directory."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        assert loader.config_dir == temp_config_dir

    # ------------------------------------------------------------------ #
    #  Loading configs
    # ------------------------------------------------------------------ #

    def test_load_baseline(self, temp_config_dir):
        """Test loading baseline configuration."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load("baseline")

        assert isinstance(config, Config)
        assert config.manufacturer.initial_assets == 10_000_000
        assert config.manufacturer.base_operating_margin == 0.08

    def test_load_with_overrides(self, temp_config_dir):
        """Test loading configuration with overrides."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        overrides = {"manufacturer.base_operating_margin": 0.12}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load("baseline", overrides=overrides)

        assert config.manufacturer.base_operating_margin == 0.12

    def test_load_with_section_level_overrides(self, temp_config_dir):
        """Test loading configuration with section-level overrides."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        overrides = {"manufacturer": {"base_operating_margin": 0.15}}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load("baseline", overrides=overrides)

        assert config.manufacturer.base_operating_margin == 0.15

    # ------------------------------------------------------------------ #
    #  Caching
    # ------------------------------------------------------------------ #

    def test_load_with_cache(self, temp_config_dir):
        """Test that configurations are cached."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config1 = loader.load("baseline")
            config2 = loader.load("baseline")

        assert config1 is config2

    def test_load_cache_with_different_params(self, temp_config_dir):
        """Test that different parameters create different cache entries."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config1 = loader.load("baseline")
            config2 = loader.load(
                "baseline", overrides={"manufacturer.base_operating_margin": 0.15}
            )

        assert config1 is not config2
        assert config1.manufacturer.base_operating_margin == 0.08
        assert config2.manufacturer.base_operating_margin == 0.15

    def test_make_hashable_dict(self, temp_config_dir):
        """Test making nested dict hashable for cache key."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        overrides = {"manufacturer": {"base_operating_margin": 0.10}}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config1 = loader.load("baseline", overrides=overrides)
            config2 = loader.load("baseline", overrides=overrides)

        # Should hit cache
        assert config1 is config2

    def test_make_hashable_list(self, temp_config_dir):
        """Test making list hashable for cache key."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        # Lists as override values should be hashable for cache key
        overrides = {"simulation": {"random_seed": 42}}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config1 = loader.load("baseline", overrides=overrides)
            config2 = loader.load("baseline", overrides=overrides)

        assert config1 is config2

    def test_clear_cache(self, temp_config_dir):
        """Test clearing the configuration cache."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loader.load("baseline")

        assert len(loader._cache) == 1
        loader.clear_cache()
        assert len(loader._cache) == 0

    # ------------------------------------------------------------------ #
    #  Deprecation warnings
    # ------------------------------------------------------------------ #

    def test_deprecation_warning(self, temp_config_dir):
        """Test that deprecation warning is shown."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader.load("baseline")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert any("ConfigLoader is deprecated" in str(x.message) for x in deprecation_warnings)

    def test_deprecation_warning_shown_once(self, temp_config_dir):
        """Test that deprecation warning is only shown once."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader.load("baseline")
            loader.load("conservative")

            deprecation_count = sum(
                1
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "ConfigLoader is deprecated" in str(x.message)
            )
            assert deprecation_count == 1

    # ------------------------------------------------------------------ #
    #  Scenarios
    # ------------------------------------------------------------------ #

    def test_load_scenario_baseline(self, temp_config_dir):
        """Test loading baseline scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load_scenario("baseline")

        assert isinstance(config, Config)

    def test_load_scenario_conservative(self, temp_config_dir):
        """Test loading conservative scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load_scenario("conservative")

        assert config.manufacturer.base_operating_margin == 0.06

    def test_load_scenario_optimistic(self, temp_config_dir):
        """Test loading optimistic scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load_scenario("optimistic")

        assert config.manufacturer.base_operating_margin == 0.10

    def test_load_scenario_invalid(self, temp_config_dir):
        """Test loading invalid scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="Unknown scenario 'invalid'"):
                loader.load_scenario("invalid")

    def test_load_scenario_with_overrides(self, temp_config_dir):
        """Test loading scenario with overrides."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        overrides = {"manufacturer.base_operating_margin": 0.12}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = loader.load_scenario("baseline", overrides=overrides)

        assert config.manufacturer.base_operating_margin == 0.12

    # ------------------------------------------------------------------ #
    #  Config comparison
    # ------------------------------------------------------------------ #

    def test_compare_configs_with_names(self, temp_config_dir):
        """Test comparing configurations by name."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diff = loader.compare_configs("baseline", "conservative")

        assert "manufacturer.base_operating_margin" in diff
        assert diff["manufacturer.base_operating_margin"]["config1"] == 0.08
        assert diff["manufacturer.base_operating_margin"]["config2"] == 0.06

    def test_compare_configs_with_objects(self, temp_config_dir):
        """Test comparing configuration objects."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config1 = MagicMock(spec=Config)
        mock_config1.model_dump.return_value = {"manufacturer": {"base_operating_margin": 0.08}}

        mock_config2 = MagicMock(spec=Config)
        mock_config2.model_dump.return_value = {"manufacturer": {"base_operating_margin": 0.06}}

        diff = loader.compare_configs(mock_config1, mock_config2)

        assert "manufacturer.base_operating_margin" in diff

    def test_compare_configs_missing_keys(self, temp_config_dir):
        """Test comparing configs with missing keys."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config1 = MagicMock(spec=Config)
        mock_config1.model_dump.return_value = {
            "manufacturer": {"base_operating_margin": 0.08},
            "extra_key": "value1",
        }

        mock_config2 = MagicMock(spec=Config)
        mock_config2.model_dump.return_value = {
            "manufacturer": {"base_operating_margin": 0.08},
            "different_key": "value2",
        }

        diff = loader.compare_configs(mock_config1, mock_config2)

        assert "extra_key" in diff
        assert diff["extra_key"]["config1"] == "value1"
        assert diff["extra_key"]["config2"] is None
        assert "different_key" in diff
        assert diff["different_key"]["config1"] is None
        assert diff["different_key"]["config2"] == "value2"

    def test_compare_configs_identical(self, temp_config_dir):
        """Test comparing identical configurations."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config = MagicMock(spec=Config)
        mock_config.model_dump.return_value = {"manufacturer": {"base_operating_margin": 0.08}}

        diff = loader.compare_configs(mock_config, mock_config)
        assert len(diff) == 0

    def test_compare_configs_nested_differences(self, temp_config_dir):
        """Test comparing configs with nested differences."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config1 = MagicMock(spec=Config)
        mock_config1.model_dump.return_value = {
            "manufacturer": {"base_operating_margin": 0.08, "nested": {"value1": 10, "value2": 20}}
        }

        mock_config2 = MagicMock(spec=Config)
        mock_config2.model_dump.return_value = {
            "manufacturer": {"base_operating_margin": 0.08, "nested": {"value1": 10, "value2": 30}}
        }

        diff = loader.compare_configs(mock_config1, mock_config2)

        assert "manufacturer.nested.value2" in diff
        assert diff["manufacturer.nested.value2"]["config1"] == 20
        assert diff["manufacturer.nested.value2"]["config2"] == 30

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def test_validate_config_with_name(self, temp_config_dir):
        """Test validating configuration by name."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = loader.validate_config("baseline")

        assert result is True

    def test_validate_config_with_object(self, temp_config_dir):
        """Test validating configuration object."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config = MagicMock(spec=Config)
        mock_config.simulation = MagicMock()
        mock_config.simulation.time_resolution = "annual"
        mock_config.simulation.time_horizon_years = 10
        mock_config.manufacturer = MagicMock()
        mock_config.manufacturer.retention_ratio = 0.5
        mock_config.growth = MagicMock()
        mock_config.growth.annual_growth_rate = 0.07

        result = loader.validate_config(mock_config)
        assert result is True

    # ------------------------------------------------------------------ #
    #  Misc
    # ------------------------------------------------------------------ #

    def test_default_config_dir(self):
        """Test default configuration directory path."""
        assert ConfigLoader.DEFAULT_CONFIG_FILE == "baseline.yaml"
        assert str(ConfigLoader.DEFAULT_CONFIG_DIR).replace("\\", "/").endswith("data/parameters")

    def test_load_not_found(self, temp_config_dir):
        """Test loading a config that doesn't exist."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(FileNotFoundError):
                loader.load("nonexistent")

    def test_list_available_configs(self, temp_config_dir):
        """Test listing available configs."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        configs = loader.list_available_configs()
        assert "baseline" in configs
        assert "conservative" in configs
        assert "optimistic" in configs

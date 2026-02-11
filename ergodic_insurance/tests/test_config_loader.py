"""Comprehensive tests for the config_loader module."""

import copy
import logging
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, PropertyMock, patch
import warnings

import pytest
import yaml

from ergodic_insurance.config import Config
from ergodic_insurance.config_loader import ConfigLoader


class TestConfigLoader:
    """Test ConfigLoader class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Create a baseline.yaml file
            baseline_config = {
                "manufacturer": {
                    "initial_assets": 10000000,
                    "base_operating_margin": 0.08,
                    "tax_rate": 0.25,
                    "fixed_costs": 500000,
                    "working_capital_ratio": 0.2,
                    "min_assets": 1000000,
                },
                "growth": {
                    "annual_growth_rate": 0.07,
                    "growth_volatility": 0.15,
                    "revenue_asset_ratio": 1.0,
                    "recession_probability": 0.1,
                },
                "claims": {
                    "attritional": {"frequency": 5.0, "mean": 50000, "variance": 625000000},
                    "large_loss": {"frequency": 0.3, "mean": 2000000, "variance": 1e12},
                    "catastrophe": {"frequency": 0.05, "mean": 10000000, "variance": 2.5e14},
                },
                "insurance": {
                    "attachment_point": 100000,
                    "coverage_limit": 5000000,
                    "base_premium_rate": 0.015,
                    "minimum_premium": 50000,
                    "profit_loading": 0.3,
                },
                "simulation": {
                    "n_simulations": 10000,
                    "time_horizon_years": 10,
                    "random_seed": 42,
                    "convergence_check_interval": 1000,
                    "progress_bar": True,
                },
                "optimization": {
                    "enabled": True,
                    "method": "powell",
                    "tolerance": 0.001,
                    "max_iterations": 100,
                    "objective": "maximize_growth",
                },
            }
            baseline_path = config_dir / "baseline.yaml"
            with open(baseline_path, "w") as f:
                yaml.dump(baseline_config, f)

            # Create a conservative.yaml file
            conservative_config = copy.deepcopy(baseline_config)
            conservative_config["manufacturer"]["base_operating_margin"] = 0.06  # type: ignore[index]
            conservative_config["growth"]["annual_growth_rate"] = 0.05  # type: ignore[index]
            conservative_path = config_dir / "conservative.yaml"
            with open(conservative_path, "w") as f:
                yaml.dump(conservative_config, f)

            # Create an optimistic.yaml file
            optimistic_config = copy.deepcopy(baseline_config)
            optimistic_config["manufacturer"]["base_operating_margin"] = 0.10  # type: ignore[index]
            optimistic_config["growth"]["annual_growth_rate"] = 0.09  # type: ignore[index]
            optimistic_path = config_dir / "optimistic.yaml"
            with open(optimistic_path, "w") as f:
                yaml.dump(optimistic_config, f)

            yield config_dir

    def test_initialization_default(self):
        """Test default ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.config_dir == ConfigLoader.DEFAULT_CONFIG_DIR
        assert len(loader._cache) == 0
        assert loader._adapter is not None
        assert loader._deprecation_warned is False

    def test_initialization_custom_dir(self, temp_config_dir):
        """Test ConfigLoader initialization with custom directory."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        assert loader.config_dir == temp_config_dir

    def test_load_baseline(self, temp_config_dir):
        """Test loading baseline configuration."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        # Mock the adapter to return a Config object
        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load("baseline")

            assert config == mock_config
            mock_load.assert_called_once_with("baseline", None)

    def test_load_with_overrides(self, temp_config_dir):
        """Test loading configuration with overrides."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            overrides = {"manufacturer": {"base_operating_margin": 0.12}}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load("baseline", overrides=overrides)

            mock_load.assert_called_once_with("baseline", overrides)

    def test_load_with_dot_notation_overrides(self, temp_config_dir):
        """Test loading configuration with dot-notation overrides."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            overrides = {
                "manufacturer.base_operating_margin": 0.12,
                "growth.annual_growth_rate": 0.08,
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load("baseline", overrides=overrides)

            mock_load.assert_called_once_with("baseline", overrides)

    def test_load_with_cache(self, temp_config_dir):
        """Test that configurations are cached."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # Load twice with same parameters
                config1 = loader.load("baseline")
                config2 = loader.load("baseline")

            # Should only call load once due to caching
            assert mock_load.call_count == 1
            assert config1 is config2

    def test_load_cache_with_different_params(self, temp_config_dir):
        """Test that different parameters create different cache entries."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config1 = MagicMock(spec=Config)
            mock_config2 = MagicMock(spec=Config)
            mock_load.side_effect = [mock_config1, mock_config2]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config1 = loader.load("baseline")
                config2 = loader.load("baseline", overrides={"test": "value"})

            assert mock_load.call_count == 2
            assert config1 is not config2

    def test_deprecation_warning(self, temp_config_dir):
        """Test that deprecation warning is shown."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = loader.load("baseline")

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "ConfigLoader is deprecated" in str(w[0].message)

    def test_deprecation_warning_shown_once(self, temp_config_dir):
        """Test that deprecation warning is only shown once."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config1 = loader.load("baseline")
                config2 = loader.load("conservative")

                # Warning should only be shown once
                assert (
                    len(
                        [
                            warning
                            for warning in w
                            if issubclass(warning.category, DeprecationWarning)
                        ]
                    )
                    == 1
                )

    def test_load_scenario_baseline(self, temp_config_dir):
        """Test loading baseline scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load_scenario("baseline")

            mock_load.assert_called_once_with("baseline", None)

    def test_load_scenario_conservative(self, temp_config_dir):
        """Test loading conservative scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load_scenario("conservative")

            mock_load.assert_called_once_with("conservative", None)

    def test_load_scenario_optimistic(self, temp_config_dir):
        """Test loading optimistic scenario."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load_scenario("optimistic")

            mock_load.assert_called_once_with("optimistic", None)

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

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            overrides = {"manufacturer": {"base_operating_margin": 0.12}}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = loader.load_scenario("baseline", overrides=overrides)

            mock_load.assert_called_once_with("baseline", overrides)

    def test_compare_configs_with_names(self, temp_config_dir):
        """Test comparing configurations by name."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader, "load") as mock_load:
            mock_config1 = MagicMock(spec=Config)
            mock_config1.model_dump.return_value = {
                "manufacturer": {"base_operating_margin": 0.08},
                "growth": {"annual_growth_rate": 0.07},
            }

            mock_config2 = MagicMock(spec=Config)
            mock_config2.model_dump.return_value = {
                "manufacturer": {"base_operating_margin": 0.06},
                "growth": {"annual_growth_rate": 0.05},
            }

            mock_load.side_effect = [mock_config1, mock_config2]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                diff = loader.compare_configs("baseline", "conservative")

            assert "manufacturer.base_operating_margin" in diff
            assert diff["manufacturer.base_operating_margin"]["config1"] == 0.08
            assert diff["manufacturer.base_operating_margin"]["config2"] == 0.06
            assert "growth.annual_growth_rate" in diff
            assert diff["growth.annual_growth_rate"]["config1"] == 0.07
            assert diff["growth.annual_growth_rate"]["config2"] == 0.05

    def test_compare_configs_with_objects(self, temp_config_dir):
        """Test comparing configuration objects."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config1 = MagicMock(spec=Config)
        mock_config1.model_dump.return_value = {"manufacturer": {"base_operating_margin": 0.08}}

        mock_config2 = MagicMock(spec=Config)
        mock_config2.model_dump.return_value = {"manufacturer": {"base_operating_margin": 0.06}}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diff = loader.compare_configs(mock_config1, mock_config2)

        assert "manufacturer.base_operating_margin" in diff
        assert diff["manufacturer.base_operating_margin"]["config1"] == 0.08
        assert diff["manufacturer.base_operating_margin"]["config2"] == 0.06

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            diff = loader.compare_configs(mock_config1, mock_config2)

        assert "manufacturer.nested.value2" in diff
        assert diff["manufacturer.nested.value2"]["config1"] == 20
        assert diff["manufacturer.nested.value2"]["config2"] == 30

    def test_validate_config_with_name(self, temp_config_dir):
        """Test validating configuration by name."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            # Add required attributes for validation
            mock_config.simulation = MagicMock()
            mock_config.simulation.time_resolution = "annual"
            mock_config.simulation.time_horizon_years = 10
            mock_config.manufacturer = MagicMock()
            mock_config.manufacturer.retention_ratio = 0.5
            mock_config.growth = MagicMock()
            mock_config.growth.annual_growth_rate = 0.07
            mock_load.return_value = mock_config

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                result = loader.validate_config("baseline")

            assert result is True
            mock_load.assert_called_once_with("baseline")

    def test_validate_config_with_object(self, temp_config_dir):
        """Test validating configuration object."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config = MagicMock(spec=Config)
        # Add required attributes for validation
        mock_config.simulation = MagicMock()
        mock_config.simulation.time_resolution = "annual"
        mock_config.simulation.time_horizon_years = 10
        mock_config.manufacturer = MagicMock()
        mock_config.manufacturer.retention_ratio = 0.5
        mock_config.growth = MagicMock()
        mock_config.growth.annual_growth_rate = 0.07

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = loader.validate_config(mock_config)

        assert result is True

    def test_validate_config_custom_validation(self, temp_config_dir):
        """Test custom validation logic."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        mock_config = MagicMock(spec=Config)
        mock_config.manufacturer = MagicMock()
        mock_config.manufacturer.base_operating_margin = 0.08
        mock_config.manufacturer.retention_ratio = 0.5
        # Add required simulation attributes
        mock_config.simulation = MagicMock()
        mock_config.simulation.time_resolution = "annual"
        mock_config.simulation.time_horizon_years = 10
        mock_config.growth = MagicMock()
        mock_config.growth.annual_growth_rate = 0.07

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = loader.validate_config(mock_config)

        # Check that margin is positive
        assert result is True
        assert mock_config.manufacturer.base_operating_margin > 0

    def test_make_hashable_dict(self, temp_config_dir):
        """Test making nested dict hashable."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        # Access the internal function through load method
        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            overrides = {"level1": {"level2": {"value": 10}}}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # This will use make_hashable internally for cache key
                config1 = loader.load("baseline", overrides=overrides)
                config2 = loader.load("baseline", overrides=overrides)

            # Should hit cache, so only one call
            assert mock_load.call_count == 1

    def test_make_hashable_list(self, temp_config_dir):
        """Test making list hashable."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with patch.object(loader._adapter, "load") as mock_load:
            mock_config = MagicMock(spec=Config)
            mock_load.return_value = mock_config

            overrides = {"values": [1, 2, 3, 4, 5]}

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config1 = loader.load("baseline", overrides=overrides)
                config2 = loader.load("baseline", overrides=overrides)

            # Should hit cache
            assert mock_load.call_count == 1

    def test_default_config_dir(self):
        """Test default configuration directory path."""
        import os

        assert ConfigLoader.DEFAULT_CONFIG_FILE == "baseline.yaml"
        # Use os.sep to handle both Windows and Unix paths
        assert str(ConfigLoader.DEFAULT_CONFIG_DIR).replace("\\", "/").endswith("data/parameters")

    def test_adapter_initialization(self, temp_config_dir):
        """Test that LegacyConfigAdapter is initialized."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        assert loader._adapter is not None
        assert hasattr(loader._adapter, "load")

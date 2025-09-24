"""Comprehensive tests for config_compat module to achieve 90% coverage."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch
import warnings

import pytest
import yaml

from ergodic_insurance.config import (
    Config,
    DebtConfig,
    GrowthConfig,
    LoggingConfig,
    ManufacturerConfig,
    OutputConfig,
    SimulationConfig,
    WorkingCapitalConfig,
)
from ergodic_insurance.config_compat import (
    ConfigTranslator,
    LegacyConfigAdapter,
    load_config,
    migrate_config_usage,
)
from ergodic_insurance.config_manager import ConfigManager
from ergodic_insurance.config_v2 import ConfigV2


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "manufacturer": {
            "initial_assets": 10_000_000.0,
            "asset_turnover_ratio": 1.0,
            "base_operating_margin": 0.08,
            "tax_rate": 0.25,
            "retention_ratio": 0.7,
        },
        "working_capital": {"percent_of_sales": 0.20},
        "growth": {
            "type": "deterministic",
            "annual_growth_rate": 0.05,
            "volatility": 0.0,
        },
        "debt": {
            "max_leverage_ratio": 2.0,
            "interest_rate": 0.06,
            "minimum_cash_balance": 100_000,
        },
        "simulation": {
            "time_horizon_years": 50,
            "time_resolution": "monthly",
            "random_seed": 42,
            "max_horizon_years": 1000,
        },
        "output": {
            "output_directory": "results",
            "file_format": "csv",
            "checkpoint_frequency": 10,
            "detailed_metrics": True,
        },
        "logging": {
            "enabled": True,
            "level": "INFO",
            "log_file": "test.log",
            "console_output": True,
            "format": "%(levelname)s: %(message)s",
        },
    }


@pytest.fixture
def sample_config_v2_dict():
    """Sample ConfigV2 dictionary."""
    return {
        "profile": {
            "name": "test",
            "description": "Test configuration",
            "version": "2.0.0",
        },
        "manufacturer": {
            "initial_assets": 10_000_000.0,
            "asset_turnover_ratio": 1.0,
            "base_operating_margin": 0.08,
            "tax_rate": 0.25,
            "retention_ratio": 0.7,
        },
        "working_capital": {"percent_of_sales": 0.20},
        "growth": {
            "type": "deterministic",
            "annual_growth_rate": 0.05,
            "volatility": 0.0,
        },
        "debt": {
            "max_leverage_ratio": 2.0,
            "interest_rate": 0.06,
            "minimum_cash_balance": 100_000,
        },
        "simulation": {
            "time_horizon_years": 50,
            "time_resolution": "monthly",
            "random_seed": 42,
            "max_horizon_years": 1000,
        },
        "output": {
            "output_directory": "results",
            "file_format": "csv",
            "checkpoint_frequency": 10,
            "detailed_metrics": True,
        },
        "logging": {
            "enabled": True,
            "level": "INFO",
            "log_file": "test.log",
            "console_output": True,
            "format": "%(levelname)s: %(message)s",
        },
        "insurance": {
            "enabled": True,
            "layers": [
                {
                    "name": "Primary",
                    "limit": 5_000_000.0,
                    "attachment": 0.0,
                    "base_premium_rate": 0.015,
                    "reinstatements": 0,
                    "aggregate_limit": None,
                }
            ],
            "deductible": 100_000.0,
            "coinsurance": 1.0,
            "waiting_period_days": 0,
            "claims_handling_cost": 0.05,
        },
        "losses": {
            "frequency_distribution": "poisson",
            "frequency_annual": 5.0,
            "severity_distribution": "lognormal",
            "severity_mean": 50_000.0,
            "severity_std": 75_000.0,
        },
    }


@pytest.fixture
def legacy_adapter():
    """Create a LegacyConfigAdapter instance."""
    return LegacyConfigAdapter()


@pytest.fixture
def sample_config_obj(sample_config_dict):
    """Create a sample Config object."""
    return Config(**sample_config_dict)


@pytest.fixture
def sample_config_v2_obj(sample_config_v2_dict):
    """Create a sample ConfigV2 object."""
    return ConfigV2(**sample_config_v2_dict)


class TestLegacyConfigAdapter:
    """Test LegacyConfigAdapter class."""

    def test_init(self, legacy_adapter):
        """Test adapter initialization."""
        assert isinstance(legacy_adapter.config_manager, ConfigManager)
        assert "baseline" in legacy_adapter._profile_mapping
        assert legacy_adapter._deprecated_warning_shown is False

    def test_load_with_deprecation_warning(self, legacy_adapter):
        """Test that deprecation warning is shown."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
                mock_config_v2 = MagicMock()
                mock_load.return_value = mock_config_v2
                with patch.object(legacy_adapter, "_convert_to_legacy") as mock_convert:
                    mock_convert.return_value = MagicMock(spec=Config)

                    result = legacy_adapter.load("baseline")

                    assert len(w) == 1
                    assert issubclass(w[0].category, DeprecationWarning)
                    assert "ConfigLoader is deprecated" in str(w[0].message)

    def test_load_profile_mapping(self, legacy_adapter):
        """Test profile name mapping."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mappings = [
                ("baseline", "default"),
                ("conservative", "conservative"),
                ("optimistic", "aggressive"),
                ("aggressive", "aggressive"),
            ]

            for old_name, new_name in mappings:
                with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
                    mock_config_v2 = MagicMock()
                    mock_load.return_value = mock_config_v2
                    with patch.object(legacy_adapter, "_convert_to_legacy") as mock_convert:
                        mock_convert.return_value = MagicMock(spec=Config)

                        legacy_adapter.load(old_name)
                        mock_load.assert_called_with(new_name, use_cache=True)

    def test_load_with_overrides(self, legacy_adapter):
        """Test loading with override parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            override_params = {
                "manufacturer": {"initial_assets": 20_000_000},
                "simulation": {"time_horizon_years": 100},
            }

            kwargs = {"random_seed": 123}

            with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
                mock_config_v2 = MagicMock()
                mock_load.return_value = mock_config_v2
                with patch.object(legacy_adapter, "_convert_to_legacy") as mock_convert:
                    mock_convert.return_value = MagicMock(spec=Config)

                    legacy_adapter.load("baseline", override_params, **kwargs)

                    # Check that overrides were flattened and passed
                    call_args = mock_load.call_args
                    assert call_args[0][0] == "default"
                    assert "manufacturer__initial_assets" in call_args[1]
                    assert "simulation__time_horizon_years" in call_args[1]
                    assert "random_seed" in call_args[1]

    def test_load_fallback_to_legacy(self, legacy_adapter, sample_config_dict, tmp_path):
        """Test fallback to legacy loading when profile not found."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create a temporary legacy config file
            legacy_dir = tmp_path / "ergodic_insurance" / "data" / "parameters"
            legacy_dir.mkdir(parents=True)
            config_file = legacy_dir / "test_config.yaml"

            with open(config_file, "w") as f:
                yaml.dump(sample_config_dict, f)

            # Mock the config_manager to raise FileNotFoundError
            with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
                mock_load.side_effect = FileNotFoundError("Profile not found")

                # Mock Path to use our temp directory
                with patch("ergodic_insurance.config_compat.Path") as mock_path:
                    mock_path.return_value = tmp_path / "ergodic_insurance" / "data" / "parameters"

                    with patch.object(legacy_adapter, "_load_legacy_direct") as mock_legacy:
                        mock_legacy.return_value = MagicMock(spec=Config)

                        result = legacy_adapter.load("test_config")
                        mock_legacy.assert_called_once()

    def test_load_config_method(self, legacy_adapter):
        """Test the load_config method."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with patch.object(legacy_adapter, "load") as mock_load:
                mock_load.return_value = MagicMock(spec=Config)

                result = legacy_adapter.load_config(
                    config_path="/some/path",
                    config_name="test",
                    some_override="value",
                )

                mock_load.assert_called_with("test", override_params={"some_override": "value"})

    def test_convert_to_legacy(self, legacy_adapter, sample_config_v2_obj):
        """Test conversion from ConfigV2 to legacy Config."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = legacy_adapter._convert_to_legacy(sample_config_v2_obj)

            assert isinstance(result, Config)
            assert isinstance(result.manufacturer, ManufacturerConfig)
            assert result.manufacturer.initial_assets == 10_000_000.0
            assert isinstance(result.simulation, SimulationConfig)
            assert result.simulation.time_horizon_years == 50

    def test_load_legacy_direct_with_yaml(self, legacy_adapter, sample_config_dict, tmp_path):
        """Test direct legacy loading from YAML file."""
        # Create a temporary legacy config file
        legacy_dir = tmp_path / "ergodic_insurance" / "data" / "parameters"
        legacy_dir.mkdir(parents=True)
        config_file = legacy_dir / "test.yaml"

        # Add YAML anchors to test filtering
        config_with_anchors = {"_base_config": {"some": "anchor"}, **sample_config_dict}

        with open(config_file, "w") as f:
            yaml.dump(config_with_anchors, f)

        # Mock __file__ to point to our temp directory
        with patch(
            "ergodic_insurance.config_compat.__file__",
            str(tmp_path / "ergodic_insurance" / "config_compat.py"),
        ):
            result = legacy_adapter._load_legacy_direct("test", {})

            assert isinstance(result, Config)
            assert result.manufacturer.initial_assets == 10_000_000.0

    def test_load_legacy_direct_with_overrides(self, legacy_adapter, sample_config_dict, tmp_path):
        """Test legacy loading with nested overrides."""
        # Create a temporary legacy config file
        legacy_dir = tmp_path / "ergodic_insurance" / "data" / "parameters"
        legacy_dir.mkdir(parents=True)
        config_file = legacy_dir / "test.yaml"

        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        overrides = {
            "manufacturer__initial_assets": 15_000_000,
            "simulation__time_horizon_years": 75,
            "simple_override": "value",
            "new_section__new_field": "new_value",  # This will create a new section
        }

        # Mock __file__ to point to our temp directory
        with patch(
            "ergodic_insurance.config_compat.__file__",
            str(tmp_path / "ergodic_insurance" / "config_compat.py"),
        ):
            result = legacy_adapter._load_legacy_direct("test", overrides)

            assert result.manufacturer.initial_assets == 15_000_000
            assert result.simulation.time_horizon_years == 75

    def test_load_with_dict_overrides_merging(self, legacy_adapter, sample_config_dict):
        """Test that dictionary overrides merge instead of replace."""
        # Create a mock ConfigV2 with proper structure
        from ergodic_insurance.config_v2 import ProfileMetadata

        # Build a proper ConfigV2 object from sample config
        config_data = sample_config_dict.copy()
        config_data["profile"] = ProfileMetadata(name="test", description="Test configuration")
        mock_config_v2 = ConfigV2(**config_data)

        with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
            # Return our properly structured config
            mock_load.return_value = mock_config_v2

            # Test with dictionary override - suppress expected deprecation warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                result = legacy_adapter.load(
                    "baseline",
                    override_params={"manufacturer": {"base_operating_margin": 0.12}},
                )

            # Verify the flatten_dict was used to handle nested params
            # The override should be flattened to manufacturer__operating_margin
            call_args = mock_load.call_args
            assert call_args is not None
            assert call_args[0][0] == "default"  # mapped profile name
            # Check that the override was properly flattened
            assert "manufacturer__base_operating_margin" in call_args[1]
            assert call_args[1]["manufacturer__base_operating_margin"] == 0.12

            # Verify the result is a proper Config object
            assert isinstance(result, Config)
            assert result.manufacturer.initial_assets == 10_000_000

    def test_load_legacy_direct_file_not_found(self, legacy_adapter):
        """Test legacy loading when file doesn't exist."""
        # Test with a file name that definitely doesn't exist
        with pytest.raises(FileNotFoundError) as exc_info:
            legacy_adapter._load_legacy_direct("nonexistent_config_12345", {})

        assert "not found in legacy or new locations" in str(exc_info.value)

    def test_flatten_dict(self, legacy_adapter):
        """Test dictionary flattening."""
        nested = {
            "level1": {"level2": {"value": 123}},
            "simple": "value",
        }

        result = legacy_adapter._flatten_dict(nested)

        assert result == {
            "level1__level2__value": 123,
            "simple": "value",
        }


class TestModuleFunctions:
    """Test module-level functions."""

    def test_load_config_function(self):
        """Test the global load_config function."""
        with patch("ergodic_insurance.config_compat._adapter") as mock_adapter:
            mock_adapter.load.return_value = MagicMock(spec=Config)

            result = load_config("test", {"override": "value"}, extra="param")

            mock_adapter.load.assert_called_with("test", {"override": "value"}, extra="param")

    def test_migrate_config_usage_with_changes(self, tmp_path):
        """Test migrating Python file with config usage."""
        test_file = tmp_path / "test.py"
        original_content = """
from ergodic_insurance.config_loader import ConfigLoader
from ergodic_insurance.config_loader import load_config

loader = ConfigLoader()
config = ConfigLoader.load("baseline")
"""

        with open(test_file, "w") as f:
            f.write(original_content)

        with patch("builtins.print") as mock_print:
            migrate_config_usage(test_file)

        # Check that file was modified
        with open(test_file, "r") as f:
            new_content = f.read()

        assert "ConfigManager" in new_content
        assert "config_compat" in new_content
        assert "TODO: Migrate to ConfigManager" in new_content

        # Check backup was created
        backup_file = test_file.with_suffix(".bak")
        assert backup_file.exists()

        # Verify print statements
        mock_print.assert_any_call(f"✓ Migrated {test_file}")
        mock_print.assert_any_call(f"  Backup saved to {backup_file}")

    def test_migrate_config_usage_no_changes(self, tmp_path):
        """Test migrating file that doesn't need changes."""
        test_file = tmp_path / "test.py"
        original_content = """
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()
config = manager.load_profile("default")
"""

        with open(test_file, "w") as f:
            f.write(original_content)

        with patch("builtins.print") as mock_print:
            migrate_config_usage(test_file)

        # Check that file wasn't modified
        with open(test_file, "r") as f:
            new_content = f.read()

        assert new_content == original_content

        # Check no backup was created
        backup_file = test_file.with_suffix(".bak")
        assert not backup_file.exists()

        # Verify print statement
        mock_print.assert_any_call(f"  No changes needed for {test_file}")


class TestConfigTranslator:
    """Test ConfigTranslator class."""

    def test_legacy_to_v2(self, sample_config_obj):
        """Test converting legacy Config to V2 format."""
        result = ConfigTranslator.legacy_to_v2(sample_config_obj)

        assert isinstance(result, dict)
        assert "profile" in result
        assert result["profile"]["name"] == "migrated"
        assert result["profile"]["version"] == "2.0.0"
        assert "manufacturer" in result
        assert result["manufacturer"]["initial_assets"] == 10_000_000.0

    def test_v2_to_legacy(self, sample_config_v2_obj):
        """Test converting ConfigV2 to legacy format."""
        result = ConfigTranslator.v2_to_legacy(sample_config_v2_obj)

        assert isinstance(result, dict)
        # Should only include legacy sections
        assert "manufacturer" in result
        assert "simulation" in result
        assert "profile" not in result  # V2-specific section
        assert "insurance" not in result  # V2-specific section
        assert "losses" not in result  # V2-specific section

    def test_v2_to_legacy_with_none_values(self):
        """Test V2 to legacy conversion with None values."""
        # Create minimal ConfigV2 with some None sections
        config_v2 = MagicMock()
        config_v2.manufacturer = MagicMock()
        config_v2.manufacturer.model_dump.return_value = {"initial_assets": 10_000_000}
        config_v2.working_capital = None
        config_v2.growth = MagicMock()
        config_v2.growth.model_dump.return_value = {"annual_growth_rate": 0.05}
        config_v2.debt = None
        config_v2.simulation = MagicMock()
        config_v2.simulation.model_dump.return_value = {"time_horizon_years": 50}
        config_v2.output = None
        config_v2.logging = None

        result = ConfigTranslator.v2_to_legacy(config_v2)

        assert "manufacturer" in result
        assert "growth" in result
        assert "simulation" in result
        assert "working_capital" not in result
        assert "debt" not in result

    def test_validate_translation_success(self, sample_config_obj):
        """Test successful translation validation."""
        # Create a "translated" config with same values
        translated = MagicMock()
        translated.manufacturer = MagicMock()
        translated.manufacturer.initial_assets = 10_000_000.0
        translated.simulation = MagicMock()
        translated.simulation.time_horizon_years = 50
        translated.growth = MagicMock()
        translated.growth.annual_growth_rate = 0.05

        result = ConfigTranslator.validate_translation(sample_config_obj, translated)

        assert result is True

    def test_validate_translation_mismatch(self, sample_config_obj):
        """Test translation validation with mismatch."""
        # Create a translated config with different values
        translated = MagicMock()
        translated.manufacturer = MagicMock()
        translated.manufacturer.initial_assets = 20_000_000.0  # Different value
        translated.simulation = MagicMock()
        translated.simulation.time_horizon_years = 50
        translated.growth = MagicMock()
        translated.growth.annual_growth_rate = 0.05

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = ConfigTranslator.validate_translation(sample_config_obj, translated)

            assert result is False
            assert len(w) == 1
            assert "Translation mismatch" in str(w[0].message)

    def test_validate_translation_missing_sections(self):
        """Test validation with missing sections."""
        original = MagicMock()
        original.manufacturer = None
        original.simulation = MagicMock()
        original.simulation.time_horizon_years = 50
        original.growth = MagicMock()
        original.growth.annual_growth_rate = 0.05

        translated = MagicMock()
        translated.manufacturer = None
        translated.simulation = MagicMock()
        translated.simulation.time_horizon_years = 50
        translated.growth = MagicMock()
        translated.growth.annual_growth_rate = 0.05

        result = ConfigTranslator.validate_translation(original, translated)
        assert result is True

    def test_validate_translation_none_fields(self):
        """Test validation when fields have None values."""
        original = MagicMock()
        original.manufacturer = MagicMock()
        original.manufacturer.initial_assets = None
        original.simulation = MagicMock()
        original.simulation.time_horizon_years = 50
        original.growth = MagicMock()
        original.growth.annual_growth_rate = 0.05

        translated = MagicMock()
        translated.manufacturer = MagicMock()
        translated.manufacturer.initial_assets = None
        translated.simulation = MagicMock()
        translated.simulation.time_horizon_years = 50
        translated.growth = MagicMock()
        translated.growth.annual_growth_rate = 0.05

        result = ConfigTranslator.validate_translation(original, translated)
        assert result is True


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_deprecated_warning_shown_once(self, legacy_adapter):
        """Test that deprecation warning is only shown once."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
                mock_config_v2 = MagicMock()
                mock_load.return_value = mock_config_v2
                with patch.object(legacy_adapter, "_convert_to_legacy") as mock_convert:
                    mock_convert.return_value = MagicMock(spec=Config)

                    # First call - should show warning
                    legacy_adapter.load("baseline")
                    assert len(w) == 1

                    # Second call - should not show warning
                    legacy_adapter.load("conservative")
                    assert len(w) == 1  # Still only one warning

    def test_complex_nested_overrides(self, legacy_adapter):
        """Test complex nested override scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            overrides = {
                "manufacturer": {
                    "initial_assets": 15_000_000,
                    "base_operating_margin": 0.10,
                },
                "simulation": {
                    "time_horizon_years": 100,
                    "num_simulations": 5000,
                },
            }

            with patch.object(legacy_adapter.config_manager, "load_profile") as mock_load:
                mock_config_v2 = MagicMock()
                mock_load.return_value = mock_config_v2
                with patch.object(legacy_adapter, "_convert_to_legacy") as mock_convert:
                    mock_convert.return_value = MagicMock(spec=Config)

                    legacy_adapter.load("baseline", overrides)

                    # Verify all nested overrides were flattened
                    call_kwargs = mock_load.call_args[1]
                    assert call_kwargs["manufacturer__initial_assets"] == 15_000_000
                    assert call_kwargs["manufacturer__base_operating_margin"] == 0.10
                    assert call_kwargs["simulation__time_horizon_years"] == 100
                    assert call_kwargs["simulation__num_simulations"] == 5000

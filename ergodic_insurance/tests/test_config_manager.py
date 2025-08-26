"""Tests for the new ConfigManager system."""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ergodic_insurance.src.config_manager import ConfigManager
from ergodic_insurance.src.config_v2 import ConfigV2, ProfileMetadata


class TestConfigManager:
    """Test suite for ConfigManager class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory structure."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"

        # Create directory structure
        (config_dir / "profiles").mkdir(parents=True)
        (config_dir / "profiles" / "custom").mkdir()
        (config_dir / "modules").mkdir()
        (config_dir / "presets").mkdir()

        # Create test profile
        test_profile = {
            "profile": {"name": "test", "description": "Test profile", "version": "2.0.0"},
            "manufacturer": {
                "initial_assets": 10000000,
                "asset_turnover_ratio": 0.8,
                "operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.7,
            },
            "working_capital": {"percent_of_sales": 0.2},
            "growth": {"type": "deterministic", "annual_growth_rate": 0.05, "volatility": 0.0},
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 2.0,
                "minimum_cash_balance": 100000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 100,
                "max_horizon_years": 1000,
                "random_seed": 42,
            },
            "output": {
                "output_directory": "outputs",
                "file_format": "csv",
                "checkpoint_frequency": 0,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "console_output": True,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

        with open(config_dir / "profiles" / "test.yaml", "w") as f:
            yaml.dump(test_profile, f)

        # Create test module
        test_module = {"insurance": {"enabled": True, "deductible": 100000}}

        with open(config_dir / "modules" / "insurance.yaml", "w") as f:
            yaml.dump(test_module, f)

        # Create test preset
        test_preset = {
            "stable": {"manufacturer": {"revenue_volatility": 0.10}},
            "volatile": {"manufacturer": {"revenue_volatility": 0.25}},
        }

        with open(config_dir / "presets" / "market.yaml", "w") as f:
            yaml.dump(test_preset, f)

        yield config_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_init(self, temp_config_dir):
        """Test ConfigManager initialization."""
        manager = ConfigManager(config_dir=temp_config_dir)

        assert manager.config_dir == temp_config_dir
        assert manager.profiles_dir == temp_config_dir / "profiles"
        assert manager.modules_dir == temp_config_dir / "modules"
        assert manager.presets_dir == temp_config_dir / "presets"

    def test_load_profile_basic(self, temp_config_dir):
        """Test basic profile loading."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test")

        assert config.profile.name == "test"
        assert config.manufacturer.initial_assets == 10000000
        assert config.simulation.time_horizon_years == 100

    def test_load_profile_with_overrides(self, temp_config_dir):
        """Test profile loading with runtime overrides."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile(
            "test", manufacturer__initial_assets=20000000, simulation__time_horizon_years=200
        )

        assert config.manufacturer.initial_assets == 20000000
        assert config.simulation.time_horizon_years == 200

    def test_load_profile_not_found(self, temp_config_dir):
        """Test loading non-existent profile."""
        manager = ConfigManager(config_dir=temp_config_dir)

        with pytest.raises(FileNotFoundError, match="Profile 'nonexistent' not found"):
            manager.load_profile("nonexistent")

    def test_list_profiles(self, temp_config_dir):
        """Test listing available profiles."""
        manager = ConfigManager(config_dir=temp_config_dir)
        profiles = manager.list_profiles()

        assert "test" in profiles

    def test_list_modules(self, temp_config_dir):
        """Test listing available modules."""
        manager = ConfigManager(config_dir=temp_config_dir)
        modules = manager.list_modules()

        assert "insurance" in modules

    def test_list_presets(self, temp_config_dir):
        """Test listing available presets."""
        manager = ConfigManager(config_dir=temp_config_dir)
        presets = manager.list_presets()

        assert "market" in presets
        assert "stable" in presets["market"]
        assert "volatile" in presets["market"]

    def test_cache_functionality(self, temp_config_dir):
        """Test configuration caching."""
        manager = ConfigManager(config_dir=temp_config_dir)

        # Load config twice with caching
        config1 = manager.load_profile("test", use_cache=True)
        config2 = manager.load_profile("test", use_cache=True)

        # Should be the same object due to caching
        assert config1 is config2

        # Load without cache should be different object
        config3 = manager.load_profile("test", use_cache=False)
        assert config3 is not config1

    def test_clear_cache(self, temp_config_dir):
        """Test cache clearing."""
        manager = ConfigManager(config_dir=temp_config_dir)

        # Load and cache
        config1 = manager.load_profile("test")
        assert len(manager._cache) > 0

        # Clear cache
        manager.clear_cache()
        assert len(manager._cache) == 0

        # Load again should be different object
        config2 = manager.load_profile("test")
        assert config2 is not config1

    def test_create_profile(self, temp_config_dir):
        """Test creating a new profile."""
        manager = ConfigManager(config_dir=temp_config_dir)

        # Create new profile
        path = manager.create_profile(
            name="custom_test",
            description="Custom test profile",
            base_profile="test",
            custom=True,
            manufacturer={"initial_assets": 15000000},
        )

        assert path.exists()
        assert path.parent.name == "custom"

        # Load the created profile
        config = manager.load_profile("custom/custom_test")
        assert config.profile.name == "custom_test"
        assert config.profile.extends == "test"

    def test_get_profile_metadata(self, temp_config_dir):
        """Test getting profile metadata without full load."""
        manager = ConfigManager(config_dir=temp_config_dir)

        metadata = manager.get_profile_metadata("test")

        assert metadata["name"] == "test"
        assert metadata["description"] == "Test profile"
        assert metadata["version"] == "2.0.0"

    def test_validate_config(self, temp_config_dir):
        """Test configuration validation."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test")

        # Should pass validation
        issues = manager.validate(config)
        assert len(issues) == 0

        # Test with extreme values
        config.simulation.time_horizon_years = 5000
        config.manufacturer.operating_margin = 0.8

        issues = manager.validate(config)
        assert len(issues) > 0
        assert any("Time horizon" in issue for issue in issues)
        assert any("Operating margin" in issue for issue in issues)

    def test_profile_inheritance(self, temp_config_dir):
        """Test profile inheritance."""
        # Create parent profile
        parent_profile = {
            "profile": {"name": "parent", "description": "Parent profile", "version": "2.0.0"},
            "manufacturer": {
                "initial_assets": 5000000,
                "asset_turnover_ratio": 0.7,
                "operating_margin": 0.07,
                "tax_rate": 0.25,
                "retention_ratio": 0.6,
            },
            "working_capital": {"percent_of_sales": 0.15},
            "growth": {"type": "deterministic", "annual_growth_rate": 0.03, "volatility": 0.0},
            "debt": {
                "interest_rate": 0.04,
                "max_leverage_ratio": 1.5,
                "minimum_cash_balance": 50000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 50,
                "max_horizon_years": 1000,
                "random_seed": 123,
            },
            "output": {
                "output_directory": "outputs",
                "file_format": "csv",
                "checkpoint_frequency": 0,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "console_output": True,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        }

        with open(temp_config_dir / "profiles" / "parent.yaml", "w") as f:
            yaml.dump(parent_profile, f)

        # Create child profile
        child_profile = {
            "profile": {
                "name": "child",
                "description": "Child profile",
                "extends": "parent",
                "version": "2.0.0",
            },
            "manufacturer": {"initial_assets": 8000000},  # Override parent value
        }

        with open(temp_config_dir / "profiles" / "child.yaml", "w") as f:
            yaml.dump(child_profile, f)

        # Load child profile
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("child")

        # Should have parent values except for overridden fields
        assert config.manufacturer.initial_assets == 8000000  # From child
        assert config.manufacturer.operating_margin == 0.07  # From parent
        assert config.simulation.time_horizon_years == 50  # From parent

    @pytest.mark.filterwarnings("ignore:Configuration issues:UserWarning")
    def test_module_inclusion(self, temp_config_dir):
        """Test module inclusion in profiles."""
        # Create profile with module inclusion
        profile_with_module = {
            "profile": {
                "name": "with_module",
                "description": "Profile with module",
                "version": "2.0.0",
                "includes": ["insurance"],
            },
            "manufacturer": {
                "initial_assets": 10000000,
                "asset_turnover_ratio": 0.8,
                "operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.7,
            },
            "working_capital": {"percent_of_sales": 0.2},
            "growth": {"type": "deterministic", "annual_growth_rate": 0.05, "volatility": 0.0},
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 2.0,
                "minimum_cash_balance": 100000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 100,
                "max_horizon_years": 1000,
                "random_seed": 42,
            },
            "output": {
                "output_directory": "outputs",
                "file_format": "csv",
                "checkpoint_frequency": 0,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "console_output": True,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        }

        with open(temp_config_dir / "profiles" / "with_module.yaml", "w") as f:
            yaml.dump(profile_with_module, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("with_module")

        # Should have insurance module applied
        assert config.insurance is not None
        assert config.insurance.enabled is True
        assert config.insurance.deductible == 100000

    def test_preset_application(self, temp_config_dir):
        """Test preset application to profiles."""
        # Create profile with preset
        profile_with_preset = {
            "profile": {
                "name": "with_preset",
                "description": "Profile with preset",
                "version": "2.0.0",
                "presets": {"market": "volatile"},
            },
            "manufacturer": {
                "initial_assets": 10000000,
                "asset_turnover_ratio": 0.8,
                "operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.7,
            },
            "working_capital": {"percent_of_sales": 0.2},
            "growth": {"type": "deterministic", "annual_growth_rate": 0.05, "volatility": 0.0},
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 2.0,
                "minimum_cash_balance": 100000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 100,
                "max_horizon_years": 1000,
                "random_seed": 42,
            },
            "output": {
                "output_directory": "outputs",
                "file_format": "csv",
                "checkpoint_frequency": 0,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "console_output": True,
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        }

        with open(temp_config_dir / "profiles" / "with_preset.yaml", "w") as f:
            yaml.dump(profile_with_preset, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("with_preset")

        # Should have preset applied
        assert "market:volatile" in config.applied_presets

"""Tests for configuration management system."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

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
from config_loader import ConfigLoader, load_config


class TestManufacturerConfig:
    """Test manufacturer configuration validation."""

    def test_valid_manufacturer_config(self):
        """Test creating valid manufacturer config."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )
        assert config.initial_assets == 10_000_000
        assert config.operating_margin == 0.08

    def test_invalid_initial_assets(self):
        """Test that negative initial assets are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ManufacturerConfig(
                initial_assets=-1000,
                asset_turnover_ratio=1.0,
                operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=1.0,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_tax_rate(self):
        """Test that invalid tax rates are rejected."""
        with pytest.raises(ValidationError):
            ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                operating_margin=0.08,
                tax_rate=1.5,  # > 1
                retention_ratio=1.0,
            )

    def test_high_margin_warning(self, capsys):
        """Test warning for unusually high operating margin."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            operating_margin=0.4,  # 40% - unusually high
            tax_rate=0.25,
            retention_ratio=1.0,
        )
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "unusually high" in captured.out


class TestWorkingCapitalConfig:
    """Test working capital configuration."""

    def test_valid_working_capital(self):
        """Test valid working capital configuration."""
        config = WorkingCapitalConfig(percent_of_sales=0.2)
        assert config.percent_of_sales == 0.2

    def test_excessive_working_capital(self):
        """Test that excessive working capital is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            WorkingCapitalConfig(percent_of_sales=0.6)  # 60% is too high
        assert "unrealistically high" in str(exc_info.value)


class TestGrowthConfig:
    """Test growth configuration."""

    def test_deterministic_growth(self):
        """Test deterministic growth configuration."""
        config = GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.0)
        assert config.type == "deterministic"
        assert config.volatility == 0.0

    def test_stochastic_requires_volatility(self):
        """Test that stochastic growth requires volatility."""
        with pytest.raises(ValidationError) as exc_info:
            GrowthConfig(type="stochastic", annual_growth_rate=0.05, volatility=0.0)
        assert "requires non-zero volatility" in str(exc_info.value)

    def test_stochastic_with_volatility(self):
        """Test valid stochastic growth configuration."""
        config = GrowthConfig(type="stochastic", annual_growth_rate=0.05, volatility=0.1)
        assert config.volatility == 0.1


class TestSimulationConfig:
    """Test simulation configuration."""

    def test_valid_simulation_config(self):
        """Test valid simulation configuration."""
        config = SimulationConfig(
            time_resolution="annual",
            time_horizon_years=100,
            max_horizon_years=1000,
            random_seed=42,
        )
        assert config.time_horizon_years == 100
        assert config.random_seed == 42

    def test_horizon_exceeds_maximum(self):
        """Test that horizon cannot exceed maximum."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationConfig(
                time_resolution="annual",
                time_horizon_years=2000,
                max_horizon_years=1000,
            )
        assert "less than or equal to 1000" in str(exc_info.value)

    def test_monthly_resolution(self):
        """Test monthly resolution configuration."""
        config = SimulationConfig(
            time_resolution="monthly", time_horizon_years=10, max_horizon_years=1000
        )
        assert config.time_resolution == "monthly"


class TestCompleteConfig:
    """Test complete configuration loading and validation."""

    @pytest.fixture
    def sample_config_dict(self):
        """Create a sample configuration dictionary."""
        return {
            "manufacturer": {
                "initial_assets": 10_000_000,
                "asset_turnover_ratio": 1.0,
                "operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 1.0,
            },
            "working_capital": {"percent_of_sales": 0.2},
            "growth": {
                "type": "deterministic",
                "annual_growth_rate": 0.05,
                "volatility": 0.0,
            },
            "debt": {
                "interest_rate": 0.015,
                "max_leverage_ratio": 2.0,
                "minimum_cash_balance": 100_000,
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
                "checkpoint_frequency": 10,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "simulation.log",
                "console_output": True,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    def test_complete_config_from_dict(self, sample_config_dict):
        """Test creating complete config from dictionary."""
        config = Config(**sample_config_dict)
        assert config.manufacturer.initial_assets == 10_000_000
        assert config.simulation.time_horizon_years == 100
        assert config.output.file_format == "csv"

    def test_config_from_yaml(self, tmp_path, sample_config_dict):
        """Test loading config from YAML file."""
        # Create temporary YAML file
        yaml_file = tmp_path / "test_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        # Load config
        config = Config.from_yaml(yaml_file)
        assert config.manufacturer.initial_assets == 10_000_000

    def test_config_override(self, sample_config_dict):
        """Test configuration override mechanism."""
        config = Config(**sample_config_dict)

        # Override using dot notation
        new_config = config.override(
            manufacturer__operating_margin=0.1,
            simulation__time_horizon_years=200,
        )

        assert new_config.manufacturer.operating_margin == 0.1
        assert new_config.simulation.time_horizon_years == 200
        # Original should be unchanged
        assert config.manufacturer.operating_margin == 0.08

    def test_config_from_dict_with_base(self, sample_config_dict):
        """Test creating config from dict with base config."""
        base_config = Config(**sample_config_dict)

        # Override some values
        override_dict = {
            "manufacturer": {"operating_margin": 0.12},
            "growth": {"annual_growth_rate": 0.08},
        }

        new_config = Config.from_dict(override_dict, base_config=base_config)

        assert new_config.manufacturer.operating_margin == 0.12
        assert new_config.growth.annual_growth_rate == 0.08
        # Unchanged values should remain
        assert new_config.manufacturer.initial_assets == 10_000_000


class TestConfigLoader:
    """Test configuration loader functionality."""

    @pytest.fixture
    def config_loader(self, project_root):
        """Create config loader with test directory."""
        return ConfigLoader(project_root / "data" / "parameters")

    def test_load_baseline(self, config_loader):
        """Test loading baseline configuration."""
        config = config_loader.load("baseline")
        assert config.manufacturer.initial_assets == 10_000_000
        assert config.manufacturer.operating_margin == 0.08

    def test_load_conservative(self, config_loader):
        """Test loading conservative configuration."""
        config = config_loader.load("conservative")
        assert config.growth.annual_growth_rate == 0.03  # Conservative growth
        assert config.manufacturer.operating_margin == 0.06  # Lower margin

    def test_load_optimistic(self, config_loader):
        """Test loading optimistic configuration."""
        config = config_loader.load("optimistic")
        assert config.growth.annual_growth_rate == 0.08  # Optimistic growth
        assert config.manufacturer.operating_margin == 0.12  # Higher margin

    def test_load_with_overrides(self, config_loader):
        """Test loading config with overrides."""
        config = config_loader.load(
            "baseline",
            overrides={"manufacturer": {"operating_margin": 0.10}},
            simulation__time_horizon_years=200,
        )
        assert config.manufacturer.operating_margin == 0.10
        assert config.simulation.time_horizon_years == 200

    def test_load_scenario(self, config_loader):
        """Test loading predefined scenarios."""
        baseline = config_loader.load_scenario("baseline")
        conservative = config_loader.load_scenario("conservative")
        optimistic = config_loader.load_scenario("optimistic")

        assert baseline.growth.annual_growth_rate == 0.05
        assert conservative.growth.annual_growth_rate == 0.03
        assert optimistic.growth.annual_growth_rate == 0.08

    def test_invalid_scenario(self, config_loader):
        """Test loading invalid scenario raises error."""
        with pytest.raises(ValueError) as exc_info:
            config_loader.load_scenario("invalid")
        assert "Unknown scenario" in str(exc_info.value)

    def test_compare_configs(self, config_loader):
        """Test comparing two configurations."""
        differences = config_loader.compare_configs("baseline", "conservative")

        # Check some expected differences
        assert "growth.annual_growth_rate" in differences
        assert differences["growth.annual_growth_rate"]["config1"] == 0.05
        assert differences["growth.annual_growth_rate"]["config2"] == 0.03

    def test_list_available_configs(self, config_loader):
        """Test listing available configurations."""
        configs = config_loader.list_available_configs()
        assert "baseline" in configs
        assert "conservative" in configs
        assert "optimistic" in configs

    def test_config_caching(self, config_loader):
        """Test that configs are cached after first load."""
        # Load twice
        config1 = config_loader.load("baseline")
        config2 = config_loader.load("baseline")

        # Should be the same cached object
        assert config1 is config2

        # Clear cache and load again
        config_loader.clear_cache()
        config3 = config_loader.load("baseline")

        # Should be different object after cache clear
        assert config1 is not config3


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_output_path_property(self):
        """Test output path property."""
        config = OutputConfig(
            output_directory="test/outputs",
            file_format="csv",
            checkpoint_frequency=0,
            detailed_metrics=True,
        )
        assert isinstance(config.output_path, Path)
        assert config.output_path == Path("test/outputs")

    def test_logging_setup(self, tmp_path, monkeypatch):
        """Test logging configuration setup."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        config = Config(
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=1.0,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.0),
            debt=DebtConfig(
                interest_rate=0.015,
                max_leverage_ratio=2.0,
                minimum_cash_balance=100_000,
            ),
            simulation=SimulationConfig(
                time_resolution="annual",
                time_horizon_years=100,
                max_horizon_years=1000,
                random_seed=42,
            ),
            output=OutputConfig(
                output_directory=str(tmp_path / "outputs"),
                file_format="csv",
                checkpoint_frequency=10,
                detailed_metrics=True,
            ),
            logging=LoggingConfig(
                enabled=True,
                level="INFO",
                log_file="test.log",
                console_output=True,
                format="%(message)s",
            ),
        )

        # Setup logging
        config.setup_logging()

        # Verify log file would be created in correct location
        expected_log = tmp_path / "outputs" / "test.log"
        assert expected_log.parent.exists()


class TestQuickLoad:
    """Test quick load convenience function."""

    def test_load_config_function(self, project_root):
        """Test the quick load_config function."""
        config = load_config("baseline")
        assert config.manufacturer.initial_assets == 10_000_000

    def test_load_config_with_overrides(self):
        """Test quick load with overrides."""
        config = load_config("baseline", manufacturer__operating_margin=0.15)
        assert config.manufacturer.operating_margin == 0.15

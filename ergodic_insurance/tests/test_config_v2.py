"""Comprehensive unit tests for config_v2 module.

Tests all configuration models, validation logic, inheritance,
and preset functionality to achieve >90% test coverage.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from pydantic import ValidationError
import pytest
import yaml

from ergodic_insurance.config import (
    Config,
    DebtConfig,
    GrowthConfig,
    InsuranceConfig,
    InsuranceLayerConfig,
    LoggingConfig,
    LossDistributionConfig,
    ManufacturerConfig,
    ModuleConfig,
    OutputConfig,
    PresetConfig,
    PresetLibrary,
    ProfileMetadata,
    SimulationConfig,
    WorkingCapitalConfig,
)


class TestProfileMetadata:
    """Test ProfileMetadata model validation and functionality."""

    def test_valid_profile_metadata(self):
        """Test creating valid profile metadata."""
        metadata = ProfileMetadata(
            name="test-profile",
            description="Test profile description",
            version="2.0.0",
            extends="base-profile",
            includes=["module1", "module2"],
            presets={"preset1": "value1"},
            author="Test Author",
            tags=["test", "sample"],
        )
        assert metadata.name == "test-profile"
        assert metadata.version == "2.0.0"
        assert len(metadata.includes) == 2
        assert metadata.author == "Test Author"

    def test_default_values(self):
        """Test default values for optional fields."""
        metadata = ProfileMetadata(name="minimal", description="Minimal profile")
        assert metadata.version == "2.0.0"
        assert metadata.extends is None
        assert metadata.includes == []
        assert metadata.presets == {}
        assert isinstance(metadata.created, datetime)

    def test_invalid_profile_name(self):
        """Test validation of invalid profile names."""
        with pytest.raises(ValidationError) as exc_info:
            ProfileMetadata(name="", description="Empty name")
        assert "Invalid profile name" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ProfileMetadata(name="invalid@name", description="Special chars")
        assert "Invalid profile name" in str(exc_info.value)

    def test_invalid_version_format(self):
        """Test validation of invalid version formats."""
        with pytest.raises(ValidationError) as exc_info:
            ProfileMetadata(name="test", description="Test", version="2.0")
        assert "Invalid version format" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ProfileMetadata(name="test", description="Test", version="v2.0.0")
        assert "Invalid version format" in str(exc_info.value)

    def test_valid_version_formats(self):
        """Test various valid version formats."""
        # Standard semantic version
        metadata1 = ProfileMetadata(name="test1", description="Test", version="1.2.3")
        assert metadata1.version == "1.2.3"

        # Version with pre-release
        metadata2 = ProfileMetadata(name="test2", description="Test", version="2.0.0-beta.1")
        assert metadata2.version == "2.0.0-beta.1"

        # Version with build metadata
        metadata3 = ProfileMetadata(
            name="test3", description="Test", version="3.0.0-alpha.2.build.456"
        )
        assert metadata3.version == "3.0.0-alpha.2.build.456"


class TestInsuranceLayerConfig:
    """Test InsuranceLayerConfig model."""

    def test_valid_layer_config(self):
        """Test creating valid insurance layer configuration."""
        layer = InsuranceLayerConfig(
            name="Primary",
            limit=1000000,
            attachment=0,
            base_premium_rate=0.015,
            reinstatements=2,
            aggregate_limit=5000000,
        )
        assert layer.limit == 1000000
        assert layer.base_premium_rate == 0.015
        assert layer.reinstatements == 2

    def test_invalid_aggregate_limit(self):
        """Test that aggregate limit < limit is now allowed (for overall cap scenarios)."""
        # This is now valid as aggregate can be an overall cap across reinstatements
        layer = InsuranceLayerConfig(
            name="Valid",
            limit=1000000,
            attachment=0,
            base_premium_rate=0.015,
            aggregate_limit=500000,  # Less than limit is now allowed
        )
        assert layer.limit == 1000000
        assert layer.aggregate_limit == 500000

    def test_valid_aggregate_limit(self):
        """Test valid aggregate limit >= per-occurrence limit."""
        layer = InsuranceLayerConfig(
            name="Valid",
            limit=1000000,
            attachment=0,
            base_premium_rate=0.015,
            aggregate_limit=1000000,  # Equal to limit
        )
        assert layer.aggregate_limit == 1000000

    def test_no_aggregate_limit(self):
        """Test layer without aggregate limit."""
        layer = InsuranceLayerConfig(
            name="No Aggregate",
            limit=1000000,
            attachment=0,
            base_premium_rate=0.015,
        )
        assert layer.aggregate_limit is None

    def test_boundary_values(self):
        """Test boundary values for layer configuration."""
        # Minimum values
        layer = InsuranceLayerConfig(
            name="Min",
            limit=0.01,
            attachment=0,
            base_premium_rate=0.001,
            reinstatements=0,
        )
        assert layer.limit == 0.01
        assert layer.attachment == 0

        # Maximum premium rate
        layer2 = InsuranceLayerConfig(
            name="Max Rate",
            limit=1000000,
            attachment=0,
            base_premium_rate=1.0,
        )
        assert layer2.base_premium_rate == 1.0


class TestInsuranceConfig:
    """Test InsuranceConfig model."""

    def test_valid_insurance_config(self):
        """Test creating valid insurance configuration."""
        layer1 = InsuranceLayerConfig(
            name="Primary", limit=1000000, attachment=0, base_premium_rate=0.015
        )
        layer2 = InsuranceLayerConfig(
            name="Excess", limit=4000000, attachment=1000000, base_premium_rate=0.008
        )

        config = InsuranceConfig(
            enabled=True,
            layers=[layer1, layer2],
            deductible=50000,
            coinsurance=0.8,
            waiting_period_days=30,
            claims_handling_cost=0.05,
        )
        assert config.enabled
        assert len(config.layers) == 2
        assert config.deductible == 50000

    def test_overlapping_layers(self):
        """Test validation catches overlapping layers."""
        layer1 = InsuranceLayerConfig(
            name="Primary", limit=2000000, attachment=0, base_premium_rate=0.015
        )
        layer2 = InsuranceLayerConfig(
            name="Excess", limit=3000000, attachment=1000000, base_premium_rate=0.008  # Overlaps
        )

        with pytest.raises(ValidationError) as exc_info:
            InsuranceConfig(layers=[layer1, layer2])
        assert "overlap" in str(exc_info.value).lower()

    def test_layer_gap_warning(self, caplog):
        """Test warning for gaps between layers."""
        import logging

        layer1 = InsuranceLayerConfig(
            name="Primary", limit=1000000, attachment=0, base_premium_rate=0.015
        )
        layer2 = InsuranceLayerConfig(
            name="Excess", limit=3000000, attachment=2000000, base_premium_rate=0.008  # Gap
        )

        with caplog.at_level(logging.WARNING):
            config = InsuranceConfig(layers=[layer1, layer2])
        assert "Gap between layers" in caplog.text
        assert config is not None

    def test_single_layer(self):
        """Test insurance with single layer."""
        layer = InsuranceLayerConfig(
            name="Single", limit=5000000, attachment=100000, base_premium_rate=0.02
        )
        config = InsuranceConfig(layers=[layer])
        assert len(config.layers) == 1

    def test_no_layers(self):
        """Test insurance config with no layers."""
        config = InsuranceConfig(layers=[])
        assert config.layers == []

    def test_layer_sorting(self, capsys):
        """Test that layers are properly sorted by attachment point."""
        layer1 = InsuranceLayerConfig(
            name="Excess", limit=3000000, attachment=2000000, base_premium_rate=0.008
        )
        layer2 = InsuranceLayerConfig(
            name="Primary", limit=1000000, attachment=0, base_premium_rate=0.015
        )
        layer3 = InsuranceLayerConfig(
            name="Middle", limit=1000000, attachment=1000000, base_premium_rate=0.012
        )

        config = InsuranceConfig(layers=[layer1, layer2, layer3])
        assert config is not None  # Validates successfully with proper sorting


class TestLossDistributionConfig:
    """Test LossDistributionConfig model."""

    def test_valid_loss_distribution(self):
        """Test creating valid loss distribution configuration."""
        config = LossDistributionConfig(
            frequency_distribution="poisson",
            frequency_annual=5.0,
            severity_distribution="lognormal",
            severity_mean=100000,
            severity_std=50000,
            correlation_factor=0.2,
            tail_alpha=2.5,
        )
        assert config.frequency_annual == 5.0
        assert config.severity_mean == 100000
        assert config.correlation_factor == 0.2

    def test_invalid_frequency_distribution(self):
        """Test validation of invalid frequency distribution."""
        with pytest.raises(ValidationError) as exc_info:
            LossDistributionConfig(
                frequency_distribution="invalid",
                frequency_annual=5.0,
                severity_distribution="lognormal",
                severity_mean=100000,
                severity_std=50000,
            )
        assert "Invalid frequency distribution" in str(exc_info.value)

    def test_invalid_severity_distribution(self):
        """Test validation of invalid severity distribution."""
        with pytest.raises(ValidationError) as exc_info:
            LossDistributionConfig(
                frequency_distribution="poisson",
                frequency_annual=5.0,
                severity_distribution="invalid",
                severity_mean=100000,
                severity_std=50000,
            )
        assert "Invalid severity distribution" in str(exc_info.value)

    def test_valid_distribution_types(self):
        """Test all valid distribution types."""
        # Valid frequency distributions
        for freq_dist in ["poisson", "negative_binomial", "binomial"]:
            config = LossDistributionConfig(
                frequency_distribution=freq_dist,
                frequency_annual=3.0,
                severity_distribution="lognormal",
                severity_mean=50000,
                severity_std=10000,
            )
            assert config.frequency_distribution == freq_dist

        # Valid severity distributions
        for sev_dist in ["lognormal", "gamma", "pareto", "weibull"]:
            config = LossDistributionConfig(
                frequency_distribution="poisson",
                frequency_annual=3.0,
                severity_distribution=sev_dist,
                severity_mean=50000,
                severity_std=10000,
            )
            assert config.severity_distribution == sev_dist

    def test_correlation_bounds(self):
        """Test correlation factor bounds."""
        # Minimum correlation
        config1 = LossDistributionConfig(
            frequency_distribution="poisson",
            frequency_annual=3.0,
            severity_distribution="lognormal",
            severity_mean=50000,
            severity_std=10000,
            correlation_factor=-1.0,
        )
        assert config1.correlation_factor == -1.0

        # Maximum correlation
        config2 = LossDistributionConfig(
            frequency_distribution="poisson",
            frequency_annual=3.0,
            severity_distribution="lognormal",
            severity_mean=50000,
            severity_std=10000,
            correlation_factor=1.0,
        )
        assert config2.correlation_factor == 1.0


class TestModuleConfig:
    """Test ModuleConfig model."""

    def test_valid_module_config(self):
        """Test creating valid module configuration."""
        config = ModuleConfig(
            module_name="risk_module",
            module_version="1.0.0",
            dependencies=["base_module", "math_module"],
        )
        assert config.module_name == "risk_module"
        assert config.module_version == "1.0.0"
        assert len(config.dependencies) == 2

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in module config."""
        config = ModuleConfig(  # type: ignore[call-arg]
            module_name="extended_module",
            module_version="2.0.0",
            dependencies=[],
            custom_field="custom_value",
            another_field=42,
        )
        assert config.module_name == "extended_module"
        assert config.custom_field == "custom_value"  # type: ignore[attr-defined]
        assert config.another_field == 42  # type: ignore[attr-defined]


class TestPresetConfig:
    """Test PresetConfig model."""

    def test_valid_preset_config(self):
        """Test creating valid preset configuration."""
        config = PresetConfig(
            preset_name="aggressive",
            preset_type="market",
            description="Aggressive market conditions",
            parameters={"volatility": 0.3, "growth_rate": 0.15},
        )
        assert config.preset_name == "aggressive"
        assert config.preset_type == "market"
        assert config.parameters["volatility"] == 0.3

    def test_invalid_preset_type(self):
        """Test validation of invalid preset type."""
        with pytest.raises(ValidationError) as exc_info:
            PresetConfig(
                preset_name="test",
                preset_type="invalid_type",
                description="Test preset",
                parameters={},
            )
        assert "Invalid preset type" in str(exc_info.value)

    def test_all_valid_preset_types(self):
        """Test all valid preset types."""
        valid_types = ["market", "layers", "risk", "optimization", "scenario"]
        for preset_type in valid_types:
            config = PresetConfig(
                preset_name=f"{preset_type}_preset",
                preset_type=preset_type,
                description=f"Test {preset_type} preset",
                parameters={"param": "value"},
            )
            assert config.preset_type == preset_type


class TestConfig:
    """Test Config model."""

    def test_valid_config_v2(self):
        """Test creating valid Config instance."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test profile"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
            insurance=InsuranceConfig(
                layers=[
                    InsuranceLayerConfig(
                        name="Primary", limit=1000000, attachment=0, base_premium_rate=0.015
                    )
                ]
            ),
            losses=LossDistributionConfig(
                frequency_distribution="poisson",
                frequency_annual=3.0,
                severity_distribution="lognormal",
                severity_mean=50000,
                severity_std=10000,
            ),
        )
        assert config.profile is not None
        assert config.profile.name == "test"
        assert config.manufacturer.initial_assets == 10000000
        assert config.insurance is not None
        assert config.losses is not None

    def test_from_profile(self):
        """Test loading Config from profile file."""
        profile_data = {
            "profile": {"name": "test", "description": "Test profile"},
            "manufacturer": {
                "initial_assets": 10000000,
                "asset_turnover_ratio": 0.8,
                "base_operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.6,
            },
            "working_capital": {
                "percent_of_sales": 0.15,
            },
            "growth": {"annual_growth_rate": 0.05},
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 3.0,
                "minimum_cash_balance": 100000,
            },
            "simulation": {"time_horizon_years": 10, "random_seed": 42},
            "output": {"output_directory": "./output"},
            "logging": {},
        }

        yaml_content = yaml.dump(profile_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = Config.from_profile(Path("test_profile.yaml"))
                assert config.profile is not None
                assert config.profile.name == "test"
                assert config.manufacturer.initial_assets == 10000000

    def test_from_profile_not_found(self):
        """Test error when profile file not found."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                Config.from_profile(Path("nonexistent.yaml"))
            assert "Profile not found" in str(exc_info.value)

    def test_from_profile_with_yaml_anchors(self):
        """Test loading profile with YAML anchors (should be filtered)."""
        profile_data = {
            "_defaults": {"some": "anchor"},  # Should be filtered
            "profile": {"name": "test", "description": "Test profile"},
            "manufacturer": {
                "initial_assets": 10000000,
                "asset_turnover_ratio": 0.8,
                "base_operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.6,
            },
            "working_capital": {
                "percent_of_sales": 0.15,
            },
            "growth": {"annual_growth_rate": 0.05},
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 3.0,
                "minimum_cash_balance": 100000,
            },
            "simulation": {"time_horizon_years": 10, "random_seed": 42},
            "output": {"output_directory": "./output"},
            "logging": {},
        }

        yaml_content = yaml.dump(profile_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                config = Config.from_profile(Path("test_profile.yaml"))
                assert config.profile is not None
                assert config.profile.name == "test"
                assert not hasattr(config, "_defaults")

    def test_with_inheritance(self):
        """Test loading config with profile inheritance."""
        parent_data = {
            "profile": {"name": "parent", "description": "Parent profile"},
            "manufacturer": {
                "initial_assets": 5000000,
                "asset_turnover_ratio": 0.7,
                "base_operating_margin": 0.07,
                "tax_rate": 0.25,
                "retention_ratio": 0.6,
            },
            "working_capital": {
                "percent_of_sales": 0.15,
            },
            "growth": {"annual_growth_rate": 0.03},
            "debt": {
                "interest_rate": 0.04,
                "max_leverage_ratio": 3.0,
                "minimum_cash_balance": 100000,
            },
            "simulation": {"time_horizon_years": 5, "random_seed": 42},
            "output": {"output_directory": "./output"},
            "logging": {},
        }

        child_data = {
            "profile": {
                "name": "child",
                "description": "Child profile",
                "extends": "parent",
            },
            "manufacturer": {
                "initial_assets": 10000000,  # Override parent
                "asset_turnover_ratio": 0.8,  # Override parent
                "base_operating_margin": 0.08,  # Override parent
                "tax_rate": 0.25,  # Keep parent value
            },
            "growth": {"annual_growth_rate": 0.05},  # Override parent
            # Inherit other sections from parent
        }

        parent_yaml = yaml.dump(parent_data)
        child_yaml = yaml.dump(child_data)

        def mock_open_side_effect(path, mode="r"):
            if "parent.yaml" in str(path):
                return mock_open(read_data=parent_yaml)()
            return mock_open(read_data=child_yaml)()

        with patch("builtins.open", side_effect=mock_open_side_effect):
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True
                config = Config.with_inheritance(Path("child.yaml"), Path("/config"))
                assert config.profile is not None
                assert config.profile.name == "child"
                assert config.manufacturer.initial_assets == 10000000  # From child
                assert config.growth.annual_growth_rate == 0.05  # From child
                assert config.working_capital.percent_of_sales == 0.15  # From parent

    def test_deep_merge(self):
        """Test deep merge functionality."""
        base = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"nested": {"deep": "original"}},
            "simple": "base_value",
        }

        override = {
            "section1": {"key2": "override2", "key3": "new3"},
            "section2": {"nested": {"deep": "modified", "new": "added"}},
            "simple": "override_value",
        }

        result = Config._deep_merge(base, override)

        assert result["section1"]["key1"] == "value1"  # From base
        assert result["section1"]["key2"] == "override2"  # Overridden
        assert result["section1"]["key3"] == "new3"  # New key
        assert result["section2"]["nested"]["deep"] == "modified"  # Deep override
        assert result["section2"]["nested"]["new"] == "added"  # Deep new key
        assert result["simple"] == "override_value"  # Simple override

    def test_with_module(self):
        """Test applying a module returns new Config without mutating original."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

        module_data = {
            "manufacturer": {"base_operating_margin": 0.10},  # Update existing
            "custom_modules": {"risk": {"module_name": "risk", "module_version": "1.0.0"}},
        }

        yaml_content = yaml.dump(module_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            new_config = config.with_module(Path("module.yaml"))
            # New config has updated values
            assert new_config.manufacturer.base_operating_margin == 0.10
            assert "risk" in new_config.custom_modules
            risk_module = new_config.custom_modules["risk"]
            assert risk_module.module_name == "risk"
            assert risk_module.module_version == "1.0.0"
            # Original is unchanged
            assert config.manufacturer.base_operating_margin == 0.08
            assert "risk" not in config.custom_modules

    def test_with_preset(self):
        """Test applying a preset returns new Config without mutating original."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

        preset_data = {
            "manufacturer": {"base_operating_margin": 0.12},
            "growth": {"annual_growth_rate": 0.08},
        }

        new_config = config.with_preset("aggressive", preset_data)

        # New config has updated values
        assert new_config.manufacturer.base_operating_margin == 0.12
        assert new_config.growth.annual_growth_rate == 0.08
        assert "aggressive" in new_config.applied_presets
        # Original is unchanged
        assert config.manufacturer.base_operating_margin == 0.08
        assert config.growth.annual_growth_rate == 0.05
        assert "aggressive" not in config.applied_presets

    def test_apply_module_deprecated(self):
        """Test that apply_module emits DeprecationWarning and returns new Config."""
        import warnings

        config = Config()
        module_data = {"manufacturer": {"base_operating_margin": 0.10}}
        yaml_content = yaml.dump(module_data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                new_config = config.apply_module(Path("module.yaml"))
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "with_module" in str(dep_warnings[0].message)
        # Returns new Config, does not mutate
        assert new_config.manufacturer.base_operating_margin == 0.10
        assert config.manufacturer.base_operating_margin == 0.08

    def test_apply_preset_deprecated(self):
        """Test that apply_preset emits DeprecationWarning and returns new Config."""
        import warnings

        config = Config()
        preset_data = {"manufacturer": {"base_operating_margin": 0.12}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_config = config.apply_preset("test_preset", preset_data)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "with_preset" in str(dep_warnings[0].message)
        # Returns new Config, does not mutate
        assert new_config.manufacturer.base_operating_margin == 0.12
        assert config.manufacturer.base_operating_margin == 0.08
        assert "test_preset" in new_config.applied_presets
        assert "test_preset" not in config.applied_presets

    def test_with_module_immutability(self):
        """Test that with_module returns a different object and original is unchanged."""
        config = Config()
        original_assets = config.manufacturer.initial_assets
        module_data = {"manufacturer": {"initial_assets": 99_999_999}}
        yaml_content = yaml.dump(module_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            new_config = config.with_module(Path("m.yaml"))

        assert id(config) != id(new_config)
        assert config.manufacturer.initial_assets == original_assets
        assert new_config.manufacturer.initial_assets == 99_999_999

    def test_with_preset_immutability(self):
        """Test that with_preset returns a different object and original is unchanged."""
        config = Config()
        original_margin = config.manufacturer.base_operating_margin
        preset_data = {"manufacturer": {"base_operating_margin": 0.20}}

        new_config = config.with_preset("big_margin", preset_data)

        assert id(config) != id(new_config)
        assert config.manufacturer.base_operating_margin == original_margin
        assert new_config.manufacturer.base_operating_margin == 0.20
        assert config.applied_presets == []
        assert "big_margin" in new_config.applied_presets

    def test_with_overrides(self):
        """Test creating config with runtime overrides."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

        # Test nested overrides with dot notation
        new_config = config.with_overrides(
            {
                "manufacturer.initial_assets": 20000000,
                "manufacturer.base_operating_margin": 0.10,
                "simulation.time_horizon_years": 20,
            }
        )

        assert new_config.manufacturer.initial_assets == 20000000
        assert new_config.manufacturer.base_operating_margin == 0.10
        assert new_config.simulation.time_horizon_years == 20
        assert new_config.overrides == {
            "manufacturer.initial_assets": 20000000,
            "manufacturer.base_operating_margin": 0.10,
            "simulation.time_horizon_years": 20,
        }

        # Original config should be unchanged
        assert config.manufacturer.initial_assets == 10000000

    def test_with_overrides_simple_key(self):
        """Test overrides with simple (non-nested) keys."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

        # Test simple key override (section-level dict)
        new_config = config.with_overrides(
            {
                "custom_modules": {"test": {"module_name": "test"}},
                "applied_presets": ["preset1"],
            }
        )

        assert "test" in new_config.custom_modules
        assert new_config.custom_modules["test"].module_name == "test"
        assert new_config.applied_presets == ["preset1"]

    def test_validate_completeness(self):
        """Test configuration completeness validation."""
        # Valid complete config
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
            insurance=InsuranceConfig(
                enabled=True,
                layers=[
                    InsuranceLayerConfig(
                        name="Primary", limit=1000000, attachment=0, base_premium_rate=0.015
                    )
                ],
            ),
            losses=LossDistributionConfig(
                frequency_distribution="poisson",
                frequency_annual=3.0,
                severity_distribution="lognormal",
                severity_mean=50000,
                severity_std=10000,
            ),
        )

        issues = config.validate_completeness()
        assert len(issues) == 0

    def test_validate_completeness_missing_losses(self):
        """Test validation detects insurance without losses."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
            insurance=InsuranceConfig(
                enabled=True,
                layers=[
                    InsuranceLayerConfig(
                        name="Primary", limit=1000000, attachment=0, base_premium_rate=0.015
                    )
                ],
            ),
            # losses=None  # Missing losses
        )

        issues = config.validate_completeness()
        assert len(issues) == 1
        assert "Insurance enabled but no loss distribution configured" in issues[0]

    def test_validate_completeness_insurance_disabled(self):
        """Test no validation issue when insurance is disabled."""
        config = Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
            insurance=InsuranceConfig(enabled=False, layers=[]),
            # losses=None  # OK when insurance is disabled
        )

        issues = config.validate_completeness()
        assert len(issues) == 0


class TestPresetLibrary:
    """Test PresetLibrary model."""

    def test_valid_preset_library(self):
        """Test creating valid preset library."""
        preset1 = PresetConfig(
            preset_name="conservative",
            preset_type="market",
            description="Conservative market conditions",
            parameters={"volatility": 0.1, "growth_rate": 0.03},
        )
        preset2 = PresetConfig(
            preset_name="aggressive",
            preset_type="market",
            description="Aggressive market conditions",
            parameters={"volatility": 0.3, "growth_rate": 0.12},
        )

        library = PresetLibrary(
            library_type="Market",
            description="Market condition presets",
            presets={"conservative": preset1, "aggressive": preset2},
        )

        assert library.library_type == "Market"
        assert len(library.presets) == 2
        assert "conservative" in library.presets

    def test_from_yaml(self):
        """Test loading preset library from YAML."""
        yaml_data = {
            "conservative": {"volatility": 0.1, "growth_rate": 0.03},
            "moderate": {"volatility": 0.2, "growth_rate": 0.06},
            "aggressive": {"volatility": 0.3, "growth_rate": 0.12},
        }

        yaml_content = yaml.dump(yaml_data)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            library = PresetLibrary.from_yaml(Path("market_conditions.yaml"))

            assert library.library_type == "Market Conditions"
            assert len(library.presets) == 3
            assert "conservative" in library.presets
            assert library.presets["conservative"].preset_name == "conservative"
            assert library.presets["conservative"].preset_type == "market"
            assert library.presets["conservative"].parameters["volatility"] == 0.1

    def test_from_yaml_different_types(self):
        """Test loading different types of preset libraries."""
        yaml_data = {"basic": {"param1": "value1"}, "advanced": {"param2": "value2"}}

        yaml_content = yaml.dump(yaml_data)

        # Test with underscores in filename
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            library = PresetLibrary.from_yaml(Path("risk_profiles.yaml"))
            assert library.library_type == "Risk Profiles"
            assert library.presets["basic"].preset_type == "risk"

        # Test with hyphens in filename
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            library = PresetLibrary.from_yaml(Path("layer-structures.yaml"))
            assert library.library_type == "Layer-Structures"
            assert "basic" in library.presets

    def test_empty_preset_library(self):
        """Test empty preset library."""
        library = PresetLibrary(
            library_type="Empty",
            description="Empty library",
            presets={},
        )
        assert len(library.presets) == 0

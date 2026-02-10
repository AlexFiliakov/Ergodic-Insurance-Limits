"""Tests for issue #357 — config system additional fixes.

Covers:
  1. switch_pricing_scenario actually applies rates to insurance layers.
  2. ConfigV2.with_overrides performs recursive (deep) merge.
  3. Cycle detection in profile inheritance for both ConfigV2.with_inheritance
     and ConfigManager._load_with_inheritance.
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from ergodic_insurance.config import (
    ConfigV2,
    DebtConfig,
    GrowthConfig,
    InsuranceConfig,
    InsuranceLayerConfig,
    LoggingConfig,
    ManufacturerConfig,
    OutputConfig,
    ProfileMetadata,
    SimulationConfig,
    WorkingCapitalConfig,
)
from ergodic_insurance.config.market import PricingScenario
from ergodic_insurance.config_loader import ConfigLoader
from ergodic_insurance.config_manager import ConfigManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_configv2(**overrides: object) -> ConfigV2:
    """Create a minimal valid ConfigV2 for testing."""
    defaults: dict = {
        "profile": ProfileMetadata(name="test", description="test"),
        "manufacturer": ManufacturerConfig(),
        "working_capital": WorkingCapitalConfig(),
        "growth": GrowthConfig(),
        "debt": DebtConfig(),
        "simulation": SimulationConfig(),
        "output": OutputConfig(),
        "logging": LoggingConfig(),
    }
    defaults.update(overrides)
    return ConfigV2(**defaults)


def _make_pricing_scenario(
    name="test",
    primary=0.010,
    first_excess=0.005,
    higher_excess=0.002,
):
    return PricingScenario(
        name=name,
        description="Test scenario",
        market_condition="normal",
        primary_layer_rate=primary,
        first_excess_rate=first_excess,
        higher_excess_rate=higher_excess,
        capacity_factor=1.0,
        competition_level="moderate",
        retention_discount=0.1,
        volume_discount=0.05,
        loss_ratio_target=0.6,
        expense_ratio=0.3,
        new_business_appetite="selective",
        renewal_retention_focus="balanced",
        coverage_enhancement_willingness="moderate",
    )


# ===========================================================================
# 1. switch_pricing_scenario must actually apply rates
# ===========================================================================


class TestSwitchPricingScenario:
    """Verify that switch_pricing_scenario writes rates into config."""

    def test_rates_applied_to_insurance_layers(self):
        """Scenario rates must update each layer's base_premium_rate."""
        # Build a ConfigV2 with three insurance layers
        config = _make_configv2(
            insurance=InsuranceConfig(
                enabled=True,
                layers=[
                    InsuranceLayerConfig(
                        name="Primary",
                        limit=1_000_000,
                        attachment=0,
                        base_premium_rate=0.999,  # will be replaced
                    ),
                    InsuranceLayerConfig(
                        name="First Excess",
                        limit=1_000_000,
                        attachment=1_000_000,
                        base_premium_rate=0.999,
                    ),
                    InsuranceLayerConfig(
                        name="Higher Excess",
                        limit=1_000_000,
                        attachment=2_000_000,
                        base_premium_rate=0.999,
                    ),
                ],
            ),
        )

        scenario = _make_pricing_scenario(
            primary=0.010,
            first_excess=0.005,
            higher_excess=0.002,
        )

        loader = ConfigLoader.__new__(ConfigLoader)

        with patch.object(loader, "load_pricing_scenarios") as mock_load:
            mock_pricing = MagicMock()
            mock_pricing.get_scenario.return_value = scenario
            mock_load.return_value = mock_pricing

            result = loader.switch_pricing_scenario(config, "baseline")

        # Verify rates were actually applied (type(config)() preserves ConfigV2)
        assert result.insurance.layers[0].base_premium_rate == 0.010  # type: ignore[union-attr]
        assert result.insurance.layers[1].base_premium_rate == 0.005  # type: ignore[union-attr]
        assert result.insurance.layers[2].base_premium_rate == 0.002  # type: ignore[union-attr]

    def test_returns_same_type_as_input(self):
        """Return type must match input type (ConfigV2 in, ConfigV2 out)."""
        config = _make_configv2(
            insurance=InsuranceConfig(
                enabled=True,
                layers=[
                    InsuranceLayerConfig(
                        name="Primary",
                        limit=1_000_000,
                        attachment=0,
                        base_premium_rate=0.01,
                    ),
                ],
            ),
        )

        scenario = _make_pricing_scenario()
        loader = ConfigLoader.__new__(ConfigLoader)

        with patch.object(loader, "load_pricing_scenarios") as mock_load:
            mock_pricing = MagicMock()
            mock_pricing.get_scenario.return_value = scenario
            mock_load.return_value = mock_pricing

            result = loader.switch_pricing_scenario(config, "baseline")

        assert isinstance(result, ConfigV2)

    def test_no_insurance_still_returns_valid_config(self):
        """When there is no insurance section, config passes through unchanged."""
        config = _make_configv2()
        assert config.insurance is None

        scenario = _make_pricing_scenario()
        loader = ConfigLoader.__new__(ConfigLoader)

        with patch.object(loader, "load_pricing_scenarios") as mock_load:
            mock_pricing = MagicMock()
            mock_pricing.get_scenario.return_value = scenario
            mock_load.return_value = mock_pricing

            result = loader.switch_pricing_scenario(config, "baseline")

        # Should return a valid config identical to input
        assert result.manufacturer == config.manufacturer

    def test_higher_excess_layers_all_get_higher_excess_rate(self):
        """Layers beyond index 2 should all use higher_excess_rate."""
        config = _make_configv2(
            insurance=InsuranceConfig(
                enabled=True,
                layers=[
                    InsuranceLayerConfig(
                        name="Primary",
                        limit=1_000_000,
                        attachment=0,
                        base_premium_rate=0.999,
                    ),
                    InsuranceLayerConfig(
                        name="First Excess",
                        limit=1_000_000,
                        attachment=1_000_000,
                        base_premium_rate=0.999,
                    ),
                    InsuranceLayerConfig(
                        name="Second Excess",
                        limit=1_000_000,
                        attachment=2_000_000,
                        base_premium_rate=0.999,
                    ),
                    InsuranceLayerConfig(
                        name="Third Excess",
                        limit=1_000_000,
                        attachment=3_000_000,
                        base_premium_rate=0.999,
                    ),
                ],
            ),
        )

        scenario = _make_pricing_scenario(
            primary=0.010,
            first_excess=0.005,
            higher_excess=0.002,
        )

        loader = ConfigLoader.__new__(ConfigLoader)
        with patch.object(loader, "load_pricing_scenarios") as mock_load:
            mock_pricing = MagicMock()
            mock_pricing.get_scenario.return_value = scenario
            mock_load.return_value = mock_pricing
            result = loader.switch_pricing_scenario(config, "baseline")

        assert result.insurance.layers[2].base_premium_rate == 0.002  # type: ignore[union-attr]
        assert result.insurance.layers[3].base_premium_rate == 0.002  # type: ignore[union-attr]


# ===========================================================================
# 2. with_overrides must deep-merge nested dicts
# ===========================================================================


class TestWithOverridesDeepMerge:
    """Verify with_overrides uses deep merge for nested dict values."""

    def test_nested_fields_preserved(self):
        """Fields not mentioned in the override must survive the merge."""
        config = _make_configv2()
        original_tax_rate = config.manufacturer.tax_rate
        original_retention = config.manufacturer.retention_ratio

        # Override only operating margin inside 'manufacturer'
        result = config.with_overrides(manufacturer={"base_operating_margin": 0.15})

        # The overridden field is updated
        assert result.manufacturer.base_operating_margin == 0.15
        # Other fields in the same section are preserved (not lost)
        assert result.manufacturer.tax_rate == original_tax_rate
        assert result.manufacturer.retention_ratio == original_retention

    def test_deeply_nested_override(self):
        """Overriding a sub-dict two levels deep must not clobber siblings."""
        config = _make_configv2()
        original_console = config.logging.console_output

        result = config.with_overrides(logging={"level": "DEBUG"})

        assert result.logging.level == "DEBUG"
        assert result.logging.console_output == original_console

    def test_non_dict_value_replaces(self):
        """Scalar overrides still replace the value entirely."""
        config = _make_configv2()

        result = config.with_overrides(manufacturer={"initial_assets": 99_000_000})

        assert result.manufacturer.initial_assets == 99_000_000

    def test_dunder_notation_still_works(self):
        """Double-underscore notation for deep keys should still work."""
        config = _make_configv2()

        result = config.with_overrides(manufacturer__base_operating_margin=0.20)

        assert result.manufacturer.base_operating_margin == 0.20


# ===========================================================================
# 3. Circular inheritance detection
# ===========================================================================


class TestCircularInheritanceConfigV2:
    """ConfigV2.with_inheritance must raise on circular profiles."""

    def test_direct_cycle_raises(self, tmp_path):
        """A extends B, B extends A → ValueError."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Profile A extends B
        profile_a = {
            "profile": {
                "name": "profile-a",
                "description": "A",
                "extends": "profile-b",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }
        # Profile B extends A
        profile_b = {
            "profile": {
                "name": "profile-b",
                "description": "B",
                "extends": "profile-a",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }

        (profiles_dir / "profile-a.yaml").write_text(yaml.dump(profile_a))
        (profiles_dir / "profile-b.yaml").write_text(yaml.dump(profile_b))

        with pytest.raises(ValueError, match="[Cc]ircular"):
            ConfigV2.with_inheritance(profiles_dir / "profile-a.yaml", tmp_path)

    def test_self_cycle_raises(self, tmp_path):
        """A extends A → ValueError."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_a = {
            "profile": {
                "name": "self-ref",
                "description": "Self",
                "extends": "self-ref",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }

        (profiles_dir / "self-ref.yaml").write_text(yaml.dump(profile_a))

        with pytest.raises(ValueError, match="[Cc]ircular"):
            ConfigV2.with_inheritance(profiles_dir / "self-ref.yaml", tmp_path)

    def test_no_cycle_succeeds(self, tmp_path):
        """Linear chain A → B (no cycle) should work fine."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_b = {
            "profile": {
                "name": "base",
                "description": "Base profile",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }

        profile_a = {
            "profile": {
                "name": "child",
                "description": "Child profile",
                "extends": "base",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }

        (profiles_dir / "base.yaml").write_text(yaml.dump(profile_b))
        (profiles_dir / "child.yaml").write_text(yaml.dump(profile_a))

        # Should not raise
        result = ConfigV2.with_inheritance(profiles_dir / "child.yaml", tmp_path)
        assert result.profile.name == "child"


class TestCircularInheritanceConfigManager:
    """ConfigManager._load_with_inheritance must raise on circular profiles."""

    def test_direct_cycle_raises(self, tmp_path):
        """A extends B, B extends A → ValueError in ConfigManager."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_a = {
            "profile": {
                "name": "profile-a",
                "description": "A",
                "extends": "profile-b",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }
        profile_b = {
            "profile": {
                "name": "profile-b",
                "description": "B",
                "extends": "profile-a",
            },
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }

        (profiles_dir / "profile-a.yaml").write_text(yaml.dump(profile_a))
        (profiles_dir / "profile-b.yaml").write_text(yaml.dump(profile_b))

        manager = ConfigManager.__new__(ConfigManager)
        manager.profiles_dir = profiles_dir

        with pytest.raises(ValueError, match="[Cc]ircular"):
            manager._load_with_inheritance(profiles_dir / "profile-a.yaml")

    def test_three_node_cycle_raises(self, tmp_path):
        """A → B → C → A should be detected as a cycle."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        base_data = {
            "manufacturer": ManufacturerConfig().model_dump(),
            "working_capital": WorkingCapitalConfig().model_dump(),
            "growth": GrowthConfig().model_dump(),
            "debt": DebtConfig().model_dump(),
            "simulation": SimulationConfig().model_dump(),
            "output": OutputConfig().model_dump(),
            "logging": LoggingConfig().model_dump(),
        }

        for name, extends in [("a", "b"), ("b", "c"), ("c", "a")]:
            data = {
                "profile": {
                    "name": name,
                    "description": name.upper(),
                    "extends": extends,
                },
                **base_data,
            }
            (profiles_dir / f"{name}.yaml").write_text(yaml.dump(data))

        manager = ConfigManager.__new__(ConfigManager)
        manager.profiles_dir = profiles_dir

        with pytest.raises(ValueError, match="[Cc]ircular"):
            manager._load_with_inheritance(profiles_dir / "a.yaml")

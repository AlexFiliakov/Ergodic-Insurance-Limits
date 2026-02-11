"""Additional tests for ConfigManager to cover missing lines.

Targets missing coverage lines:
57-63 (fallback imports), 200-204 (make_hashable), 216 (custom profile path),
272 (parent in custom dir), 282 (missing parent warning), 295-296 (module not found),
319-327 (module apply logic), 343 (preset hyphen-to-underscore), 346-347 (preset lib not found),
353 (cached preset library), 357-362 (preset name not found), 381-386 (with_preset method),
398 (with_overrides method), 442 (custom profile listing), 516/519 (get_profile_metadata custom/missing),
567 (create_profile custom=False)
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import warnings

import pytest
import yaml

from ergodic_insurance.config import ConfigV2
from ergodic_insurance.config_manager import ConfigManager

# ---------------------------------------------------------------------------
# Shared fixture: creates a temporary config directory structure
# ---------------------------------------------------------------------------


def _make_full_profile(name="test", description="Test profile", extends=None):
    """Build a complete profile dictionary suitable for ConfigV2."""
    profile = {
        "profile": {
            "name": name,
            "description": description,
            "version": "2.0.0",
        },
        "manufacturer": {
            "initial_assets": 10_000_000,
            "asset_turnover_ratio": 0.8,
            "base_operating_margin": 0.08,
            "tax_rate": 0.25,
            "retention_ratio": 0.7,
        },
        "working_capital": {"percent_of_sales": 0.2},
        "growth": {"type": "deterministic", "annual_growth_rate": 0.05, "volatility": 0.0},
        "debt": {
            "interest_rate": 0.05,
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
    if extends:
        profile["profile"]["extends"] = extends
    return profile


@pytest.fixture
def temp_config_dir():
    """Create a rich temporary config directory for testing."""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / "config"

    # Create directory structure
    (config_dir / "profiles").mkdir(parents=True)
    (config_dir / "profiles" / "custom").mkdir()
    (config_dir / "modules").mkdir()
    (config_dir / "presets").mkdir()

    # Default profile
    with open(config_dir / "profiles" / "default.yaml", "w") as f:
        yaml.dump(_make_full_profile("default", "Default profile"), f)

    # A second profile for testing
    with open(config_dir / "profiles" / "test.yaml", "w") as f:
        yaml.dump(_make_full_profile("test", "Test profile"), f)

    # Custom profile in custom/ directory
    custom_profile = _make_full_profile("my_custom", "Custom profile")
    with open(config_dir / "profiles" / "custom" / "my_custom.yaml", "w") as f:
        yaml.dump(custom_profile, f)

    # A module
    module_data = {"manufacturer": {"fixed_costs": 500_000}}
    with open(config_dir / "modules" / "insurance.yaml", "w") as f:
        yaml.dump(module_data, f)

    # Preset library
    preset_data = {
        "stable": {"manufacturer": {"revenue_volatility": 0.10}},
        "volatile": {"manufacturer": {"revenue_volatility": 0.25}},
    }
    with open(config_dir / "presets" / "market.yaml", "w") as f:
        yaml.dump(preset_data, f)

    yield config_dir
    shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# Tests targeting make_hashable (lines 200-204)
# ---------------------------------------------------------------------------


class TestMakeHashable:
    """Test the internal make_hashable function via load_profile caching."""

    def test_load_profile_with_dict_overrides_uses_cache(self, temp_config_dir):
        """Overrides containing dicts/lists exercise make_hashable (lines 200-204)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config1 = manager.load_profile(
            "test",
            manufacturer={"initial_assets": 20_000_000},
        )
        # Load again with identical overrides -- should hit cache
        config2 = manager.load_profile(
            "test",
            manufacturer={"initial_assets": 20_000_000},
        )
        assert config1 is config2

    def test_load_profile_with_list_overrides(self, temp_config_dir):
        """Overrides containing lists exercise the list branch in make_hashable."""
        manager = ConfigManager(config_dir=temp_config_dir)
        # Even if ignored by the model, the hashing function still runs on the overrides
        config = manager.load_profile(
            "test",
            modules=["insurance"],
        )
        assert config is not None


# ---------------------------------------------------------------------------
# Tests targeting custom profile path (line 216)
# ---------------------------------------------------------------------------


class TestCustomProfilePath:
    """Test loading profiles from the custom/ subdirectory."""

    def test_load_custom_profile(self, temp_config_dir):
        """Loading custom/<name> should fall through to custom directory (line 216)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("custom/my_custom")
        assert config.profile.name == "my_custom"


# ---------------------------------------------------------------------------
# Tests targeting inheritance edge cases (lines 272, 282)
# ---------------------------------------------------------------------------


class TestInheritanceEdgeCases:
    """Test inheritance with custom parent paths and missing parents."""

    def test_inherit_from_custom_parent(self, temp_config_dir):
        """Parent profile in custom/ subdirectory (line 272)."""
        # Create a profile that extends a parent only in custom/
        parent = _make_full_profile("custom_parent", "Custom parent")
        with open(temp_config_dir / "profiles" / "custom" / "custom_parent.yaml", "w") as f:
            yaml.dump(parent, f)

        child = _make_full_profile("inheritor", "Inherits custom parent", extends="custom_parent")
        # remove top-level profiles/custom_parent.yaml so the code must look in custom/
        child_path = temp_config_dir / "profiles" / "inheritor.yaml"
        with open(child_path, "w") as f:
            yaml.dump(child, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("inheritor")
        assert config.profile.name == "inheritor"

    def test_inherit_from_missing_parent_warns(self, temp_config_dir):
        """Missing parent profile emits a warning (line 282)."""
        child = _make_full_profile("orphan", "Orphan profile", extends="nonexistent_parent")
        with open(temp_config_dir / "profiles" / "orphan.yaml", "w") as f:
            yaml.dump(child, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = manager.load_profile("orphan", use_cache=False)
            parent_warnings = [x for x in w if "Parent profile" in str(x.message)]
            assert len(parent_warnings) > 0
        assert config.profile.name == "orphan"


# ---------------------------------------------------------------------------
# Tests targeting module application (lines 295-296, 319-327)
# ---------------------------------------------------------------------------


class TestModuleApplication:
    """Test _apply_module edge cases."""

    def test_apply_nonexistent_module_warns(self, temp_config_dir):
        """Applying a non-existent module emits a warning (lines 295-296)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager._apply_module(config, "does_not_exist")
            mod_warnings = [x for x in w if "Module" in str(x.message)]
            assert len(mod_warnings) > 0

    def test_apply_module_with_pydantic_model_update(self, temp_config_dir):
        """Module data that updates an existing Pydantic sub-model (lines 319-323)."""
        # Create a module that updates manufacturer (which is a Pydantic model)
        module_data = {"manufacturer": {"base_operating_margin": 0.12}}
        with open(temp_config_dir / "modules" / "margin_boost.yaml", "w") as f:
            yaml.dump(module_data, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)
        manager._apply_module(config, "margin_boost")
        assert config.manufacturer.base_operating_margin == 0.12

    def test_apply_module_with_non_dict_value(self, temp_config_dir):
        """Module data with a non-dict value for an existing attribute (lines 326-327)."""
        # Create a module that sets a scalar attribute
        module_data = {"applied_presets": ["market:stable"]}
        with open(temp_config_dir / "modules" / "scalar_module.yaml", "w") as f:
            yaml.dump(module_data, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)
        manager._apply_module(config, "scalar_module")
        assert config.applied_presets == ["market:stable"]


# ---------------------------------------------------------------------------
# Tests targeting preset application (lines 343, 346-347, 353, 357-362)
# ---------------------------------------------------------------------------


class TestPresetApplication:
    """Test _apply_preset edge cases."""

    def test_preset_library_not_found_warns(self, temp_config_dir):
        """Non-existent preset library emits a warning (lines 346-347)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager._apply_preset(config, "nonexistent_lib", "any_name")
            lib_warnings = [x for x in w if "Preset library" in str(x.message)]
            assert len(lib_warnings) > 0

    def test_preset_name_not_found_warns(self, temp_config_dir):
        """Non-existent preset name in an existing library emits warning (lines 357-362)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager._apply_preset(config, "market", "nonexistent_preset")
            name_warnings = [
                x for x in w if "Preset 'nonexistent_preset' not found" in str(x.message)
            ]
            assert len(name_warnings) > 0

    def test_preset_library_caching(self, temp_config_dir):
        """Second call to same library type uses cache (line 353)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)

        # First call loads from disk
        manager._apply_preset(config, "market", "stable")
        assert "market" in manager._preset_libraries

        # Second call uses cache
        config2 = manager.load_profile("test", use_cache=False)
        manager._apply_preset(config2, "market", "volatile")
        # Should still be the same cached library object
        assert "market" in manager._preset_libraries

    def test_preset_hyphen_to_underscore(self, temp_config_dir):
        """Preset file lookup with hyphen-to-underscore conversion (line 343)."""
        # Create a preset file with underscores
        preset_data = {"low_freq": {"manufacturer": {"base_operating_margin": 0.06}}}
        with open(temp_config_dir / "presets" / "loss_types.yaml", "w") as f:
            yaml.dump(preset_data, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)
        # Use hyphenated name -- should fall through to underscore version
        manager._apply_preset(config, "loss-types", "low_freq")
        assert "loss-types" in manager._preset_libraries or "loss_types" in str(
            manager._preset_libraries
        )


# ---------------------------------------------------------------------------
# Tests targeting with_preset and with_overrides (lines 381-386, 398)
# ---------------------------------------------------------------------------


class TestWithPresetAndOverrides:
    """Test public with_preset and with_overrides methods."""

    def test_with_preset_creates_new_config(self, temp_config_dir):
        """with_preset returns a new ConfigV2 instance (lines 381-386)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)

        new_config = manager.with_preset(config, "market", "stable")
        assert new_config is not config
        assert "market:stable" in new_config.applied_presets

    def test_with_overrides_delegates(self, temp_config_dir):
        """with_overrides delegates to config.with_overrides (line 398)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)

        new_config = manager.with_overrides(
            config, {"manufacturer": {"initial_assets": 99_999_999}}
        )
        assert new_config.manufacturer.initial_assets == 99_999_999


# ---------------------------------------------------------------------------
# Tests targeting list_profiles with custom dir (line 442)
# ---------------------------------------------------------------------------


class TestListProfilesCustom:
    """Test listing profiles that include custom/ subdirectory."""

    def test_list_profiles_includes_custom(self, temp_config_dir):
        """Custom profiles should appear in listing (line 442)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        profiles = manager.list_profiles()
        assert "custom/my_custom" in profiles


# ---------------------------------------------------------------------------
# Tests targeting get_profile_metadata fallback (lines 516, 519)
# ---------------------------------------------------------------------------


class TestGetProfileMetadata:
    """Test get_profile_metadata for custom and missing profiles."""

    def test_metadata_for_custom_profile(self, temp_config_dir):
        """Metadata lookup falls through to custom/ directory (line 516)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        # Need to clear lru_cache between calls
        manager.get_profile_metadata.cache_clear()
        metadata = manager.get_profile_metadata("my_custom")
        assert metadata.get("name") == "my_custom"

    def test_metadata_for_missing_profile(self, temp_config_dir):
        """Metadata for non-existent profile returns empty dict (line 519)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.get_profile_metadata.cache_clear()
        metadata = manager.get_profile_metadata("totally_missing")
        assert metadata == {}


# ---------------------------------------------------------------------------
# Tests targeting create_profile with custom=False (line 567)
# ---------------------------------------------------------------------------


class TestCreateProfileNotCustom:
    """Test create_profile with custom=False."""

    def test_create_profile_in_standard_dir(self, temp_config_dir):
        """create_profile with custom=False saves in profiles/ root (line 567)."""
        manager = ConfigManager(config_dir=temp_config_dir)
        path = manager.create_profile(
            name="standard_new",
            description="Standard new profile",
            base_profile="test",
            custom=False,
        )
        assert path.exists()
        assert path.parent == temp_config_dir / "profiles"
        assert path.name == "standard_new.yaml"


# ---------------------------------------------------------------------------
# Tests targeting custom profile fallback without "custom/" prefix (line 216)
# ---------------------------------------------------------------------------


class TestCustomProfileWithoutPrefix:
    """Test loading a profile that only exists in custom/ without 'custom/' prefix."""

    def test_load_profile_falls_back_to_custom_dir(self, temp_config_dir):
        """Profile name without 'custom/' prefix that only exists in custom/ (line 216).

        When profiles_dir/{name}.yaml does not exist but
        profiles_dir/custom/{name}.yaml does, the code should find it.
        """
        # Create a profile only in custom/ directory
        only_in_custom = _make_full_profile("only_custom", "Only in custom dir")
        with open(temp_config_dir / "profiles" / "custom" / "only_custom.yaml", "w") as f:
            yaml.dump(only_in_custom, f)

        # Ensure it does NOT exist in profiles/ root
        assert not (temp_config_dir / "profiles" / "only_custom.yaml").exists()

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("only_custom")
        assert config.profile.name == "only_custom"


# ---------------------------------------------------------------------------
# Tests targeting default config_dir (lines 117-118)
# ---------------------------------------------------------------------------


class TestDefaultConfigDir:
    """Test ConfigManager initialization with default config_dir."""

    def test_init_with_default_dir(self):
        """ConfigManager() without config_dir uses module-relative path (lines 117-118)."""
        manager = ConfigManager()
        # Should have set config_dir to ergodic_insurance/data/config
        assert manager.config_dir.name == "config"
        assert "data" in str(manager.config_dir)


# ---------------------------------------------------------------------------
# Tests targeting module apply with non-Pydantic dict value (line 325)
# ---------------------------------------------------------------------------


class TestModuleApplyNonPydanticDict:
    """Test module application with a dict value for a non-Pydantic field."""

    def test_module_with_dict_for_plain_dict_attr(self, temp_config_dir):
        """Module updates a dict attribute that is not a Pydantic model (line 325).

        The 'overrides' field on ConfigV2 is a plain dict, not a Pydantic model.
        """
        module_data = {"overrides": {"custom_key": "custom_value"}}
        with open(temp_config_dir / "modules" / "dict_module.yaml", "w") as f:
            yaml.dump(module_data, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_profile("test", use_cache=False)
        manager._apply_module(config, "dict_module")
        # The overrides dict should have been set
        assert config.overrides == {"custom_key": "custom_value"}

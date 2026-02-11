"""Security tests for ConfigManager path traversal prevention.

Verifies that profile names, module names, and preset types are validated
against path traversal attacks (issue #620).
"""

from pathlib import Path
import shutil
import tempfile

import pytest
import yaml

from ergodic_insurance.config_manager import ConfigManager


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
    """Create a temporary config directory for security tests."""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / "config"

    # Create directory structure
    (config_dir / "profiles").mkdir(parents=True)
    (config_dir / "profiles" / "custom").mkdir()
    (config_dir / "modules").mkdir()
    (config_dir / "presets").mkdir()

    # Create a valid default profile
    with open(config_dir / "profiles" / "default.yaml", "w") as f:
        yaml.dump(_make_full_profile("default", "Default profile"), f)

    # Create a valid test profile
    with open(config_dir / "profiles" / "test.yaml", "w") as f:
        yaml.dump(_make_full_profile("test", "Test profile"), f)

    # Create a valid custom profile
    with open(config_dir / "profiles" / "custom" / "client-abc.yaml", "w") as f:
        yaml.dump(_make_full_profile("client-abc", "Client ABC", extends="default"), f)

    yield config_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def manager(temp_config_dir):
    return ConfigManager(config_dir=temp_config_dir)


# ── _validate_name unit tests ──────────────────────────────────────────


class TestValidateName:
    """Unit tests for the _validate_name static method."""

    @pytest.mark.parametrize(
        "name",
        ["default", "my-profile", "my_profile", "Profile123", "a", "A-B_c-3"],
    )
    def test_safe_names_accepted(self, name):
        ConfigManager._validate_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "../etc/passwd",
            "..\\windows\\system32",
            "foo/../bar",
            "..",
            ".",
            "name with spaces",
            "name;rm -rf",
            "name\x00null",
            "",
        ],
    )
    def test_unsafe_names_rejected(self, name):
        with pytest.raises(ValueError, match="Invalid name"):
            ConfigManager._validate_name(name)

    @pytest.mark.parametrize(
        "name",
        ["custom/client-abc", "sub/dir/profile", "a/b"],
    )
    def test_slashes_allowed_when_enabled(self, name):
        ConfigManager._validate_name(name, allow_slashes=True)

    @pytest.mark.parametrize(
        "name",
        [
            "../escape",
            "custom/../escape",
            "custom/../../etc/passwd",
            "custom/..\\escape",
            "ok/..",
        ],
    )
    def test_dotdot_rejected_even_with_slashes(self, name):
        with pytest.raises(ValueError, match="Invalid name"):
            ConfigManager._validate_name(name, allow_slashes=True)


# ── _validate_path_containment unit tests ──────────────────────────────


class TestValidatePathContainment:

    def test_child_path_accepted(self, tmp_path):
        parent = tmp_path / "config"
        parent.mkdir()
        child = parent / "profiles" / "default.yaml"
        ConfigManager._validate_path_containment(child, parent)

    def test_traversal_path_rejected(self, tmp_path):
        parent = tmp_path / "config"
        parent.mkdir()
        escaped = parent / ".." / "evil.yaml"
        with pytest.raises(ValueError, match="Path traversal detected"):
            ConfigManager._validate_path_containment(escaped, parent)


# ── load_profile path traversal tests ──────────────────────────────────


class TestLoadProfilePathTraversal:

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "../../etc/passwd",
            "../secret",
            "..\\windows\\system32\\config",
            "foo/../../../etc/shadow",
            "..",
        ],
    )
    def test_traversal_in_profile_name_rejected(self, manager, malicious_name):
        with pytest.raises(ValueError):
            manager.load_profile(malicious_name)

    def test_valid_profile_loads(self, manager):
        config = manager.load_profile("test")
        assert config.profile.name == "test"

    def test_valid_custom_profile_loads(self, manager):
        config = manager.load_profile("custom/client-abc")
        assert config.profile.name == "client-abc"


# ── create_profile path traversal tests ────────────────────────────────


class TestCreateProfilePathTraversal:

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "../../etc/cron.d/evil",
            "../escape",
            "..\\..\\windows\\tasks\\evil",
            "legitimate/../../../etc/passwd",
            "..",
        ],
    )
    def test_traversal_in_create_rejected(self, manager, malicious_name):
        with pytest.raises(ValueError):
            manager.create_profile(
                name=malicious_name,
                description="malicious",
                base_profile="default",
            )

    def test_valid_create_succeeds(self, manager, temp_config_dir):
        path = manager.create_profile(
            name="safe-profile",
            description="A safe profile",
            base_profile="default",
        )
        assert path.exists()
        assert str(path.resolve()).startswith(str((temp_config_dir / "profiles").resolve()))


# ── get_profile_metadata path traversal tests ─────────────────────────


class TestGetProfileMetadataPathTraversal:

    @pytest.mark.parametrize(
        "malicious_name",
        ["../../etc/passwd", "../secret", ".."],
    )
    def test_traversal_rejected(self, manager, malicious_name):
        with pytest.raises(ValueError):
            manager.get_profile_metadata(malicious_name)


# ── _load_with_inheritance path traversal tests ────────────────────────


class TestInheritancePathTraversal:

    def test_malicious_extends_rejected(self, manager, temp_config_dir):
        """A profile whose 'extends' field contains traversal is rejected."""
        malicious_profile = _make_full_profile("evil", "Evil profile", extends="../../etc/passwd")
        profile_path = temp_config_dir / "profiles" / "evil.yaml"
        with open(profile_path, "w") as f:
            yaml.dump(malicious_profile, f)

        with pytest.raises(ValueError):
            manager.load_profile("evil")


# ── _apply_module path traversal tests ─────────────────────────────────


class TestApplyModulePathTraversal:

    def test_malicious_module_name_rejected(self, manager, temp_config_dir):
        """A profile that includes a traversal module name is rejected."""
        profile_with_module = _make_full_profile("mod-test", "Module test")
        profile_with_module["profile"]["includes"] = ["../../etc/passwd"]
        with open(temp_config_dir / "profiles" / "mod-test.yaml", "w") as f:
            yaml.dump(profile_with_module, f)

        with pytest.raises(ValueError):
            manager.load_profile("mod-test")


# ── _apply_preset path traversal tests ─────────────────────────────────


class TestApplyPresetPathTraversal:

    def test_malicious_preset_type_rejected(self, manager, temp_config_dir):
        """A profile that references a traversal preset type is rejected."""
        profile_with_preset = _make_full_profile("preset-test", "Preset test")
        profile_with_preset["profile"]["presets"] = {"../../etc/cron": "evil"}
        with open(temp_config_dir / "profiles" / "preset-test.yaml", "w") as f:
            yaml.dump(profile_with_preset, f)

        with pytest.raises(ValueError):
            manager.load_profile("preset-test")

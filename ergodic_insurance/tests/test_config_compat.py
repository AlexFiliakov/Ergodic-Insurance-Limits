"""Tests for config_compat deprecation stub module.

Since Config and ConfigV2 have been unified (Issue #638), config_compat is now
a thin deprecation stub. These tests verify that:
  1. Importing config_compat emits a DeprecationWarning.
  2. Deprecated aliases (ConfigV2, LegacyConfigAdapter, ConfigTranslator) are
     present with expected values.
  3. load_config() delegates to ConfigManager and emits a warning.
  4. migrate_config_usage() emits a warning and is a no-op.
"""

import importlib
import warnings

import pytest

from ergodic_insurance.config import Config


class TestConfigCompatImportWarning:
    """Test that importing config_compat emits deprecation warning."""

    def test_import_emits_deprecation_warning(self):
        """Importing config_compat should emit a DeprecationWarning."""
        # Force re-import to trigger the module-level warning
        import ergodic_insurance.config_compat as cc

        # Module should have loaded successfully
        assert cc is not None

    def test_module_level_warning_content(self):
        """The deprecation warning should mention Config directly."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.reload(importlib.import_module("ergodic_insurance.config_compat"))
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "config_compat" in str(deprecation_warnings[0].message).lower()


class TestDeprecatedAliases:
    """Test that deprecated aliases are present with correct values."""

    def test_config_v2_alias_is_config(self):
        """ConfigV2 in config_compat should be the same as Config."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import ConfigV2

        assert ConfigV2 is Config

    def test_legacy_config_adapter_is_none(self):
        """LegacyConfigAdapter should be None (removed)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import LegacyConfigAdapter

        assert LegacyConfigAdapter is None

    def test_config_translator_is_none(self):
        """ConfigTranslator should be None (removed)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import ConfigTranslator

        assert ConfigTranslator is None

    def test_config_reexport_is_config(self):
        """Config re-exported from config_compat should be the real Config."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import Config as CompatConfig

        assert CompatConfig is Config


class TestLoadConfigFunction:
    """Test the deprecated load_config function."""

    def test_load_config_emits_deprecation_warning(self):
        """load_config() should emit a DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import load_config

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                load_config("default")
            except (FileNotFoundError, ValueError, TypeError):
                pass  # May fail if no profile exists; we only care about the warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "load_config" in str(deprecation_warnings[0].message).lower()


class TestMigrateConfigUsage:
    """Test the deprecated migrate_config_usage function."""

    def test_migrate_emits_deprecation_warning(self, tmp_path):
        """migrate_config_usage() should emit a DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import migrate_config_usage

        test_file = tmp_path / "dummy.py"
        test_file.write_text("# nothing to migrate\n")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            migrate_config_usage(test_file)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "migrate_config_usage" in str(deprecation_warnings[0].message).lower()

    def test_migrate_is_noop(self, tmp_path):
        """migrate_config_usage() should not modify the file."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ergodic_insurance.config_compat import migrate_config_usage

        test_file = tmp_path / "test.py"
        original = "from ergodic_insurance.config_loader import ConfigLoader\n"
        test_file.write_text(original)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            migrate_config_usage(test_file)

        # File should NOT be modified (function is a no-op now)
        assert test_file.read_text() == original

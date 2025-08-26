"""Tests for configuration migration tools."""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ergodic_insurance.src.config_migrator import ConfigMigrator


class TestConfigMigrator:
    """Test suite for ConfigMigrator class."""

    @pytest.fixture
    def migrator(self):
        """Create a ConfigMigrator instance for testing."""
        return ConfigMigrator()

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        legacy_dir = Path(temp_dir) / "parameters"
        new_dir = Path(temp_dir) / "config"
        legacy_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)

        yield legacy_dir, new_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_convert_baseline(self, migrator):
        """Test baseline.yaml conversion to default profile."""
        # Mock the file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = """
manufacturer:
  initial_assets: 10000000
  asset_turnover_ratio: 0.8
simulation:
  time_horizon_years: 100
"""
            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "manufacturer": {"initial_assets": 10000000, "asset_turnover_ratio": 0.8},
                    "simulation": {"time_horizon_years": 100},
                }

                result = migrator.convert_baseline()

                assert "profile" in result
                assert result["profile"]["name"] == "default"
                assert "manufacturer" in result
                assert result["manufacturer"]["initial_assets"] == 10000000

    def test_convert_conservative(self, migrator):
        """Test conservative.yaml conversion."""
        with patch("builtins.open", create=True) as mock_open:
            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "manufacturer": {"initial_assets": 15000000, "operating_margin": 0.06}
                }

                result = migrator.convert_conservative()

                assert result["profile"]["name"] == "conservative"
                assert result["profile"]["extends"] == "default"
                assert "manufacturer" in result

    def test_convert_optimistic(self, migrator):
        """Test optimistic.yaml conversion to aggressive profile."""
        with patch("builtins.open", create=True) as mock_open:
            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"growth": {"annual_growth_rate": 0.15}}

                result = migrator.convert_optimistic()

                assert result["profile"]["name"] == "aggressive"
                assert result["profile"]["extends"] == "default"
                assert "growth" in result

    def test_extract_modules(self, migrator):
        """Test extraction of configuration modules."""
        # Mock the file operations
        with patch("builtins.open", create=True) as mock_open:
            with patch("yaml.safe_load") as mock_yaml_load:
                with patch("yaml.dump") as mock_yaml_dump:
                    with patch.object(Path, "exists", return_value=True):
                        # Setup mock return values
                        mock_yaml_load.side_effect = [
                            {"insurance": {"premium_rate": 0.02}},
                            {"insurance": {"layers": []}},
                            {"insurance": {"structures": {}}},
                            {"pricing": {"scenarios": []}},
                            {"losses": {"frequency": 5}},
                            {"loss_distributions": {"severity": "lognormal"}},
                            {"stochastic": {"volatility": 0.15}},
                            {"business": {"optimization": True}},
                        ]

                        migrator.extract_modules()

                        # Verify files were created
                        assert mock_yaml_dump.call_count == 4  # 4 modules
                        assert len(migrator.migration_report) > 0

    def test_create_presets(self, migrator):
        """Test creation of preset libraries."""
        with patch("builtins.open", create=True) as mock_open:
            with patch("yaml.dump") as mock_yaml_dump:
                migrator.create_presets()

                # Should create 3 preset files
                assert mock_yaml_dump.call_count == 3

                # Check that presets were created with correct structure
                calls = mock_yaml_dump.call_args_list
                for call in calls:
                    preset_data = call[0][0]
                    assert isinstance(preset_data, dict)
                    assert len(preset_data) > 0

    def test_deep_merge(self, migrator):
        """Test deep merge functionality."""
        target = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2]}
        source = {"b": {"c": 5, "f": 6}, "g": 7}

        migrator._deep_merge(target, source)

        assert target["a"] == 1
        assert target["b"]["c"] == 5  # type: ignore  # Overwritten
        assert target["b"]["d"] == 3  # type: ignore  # Preserved
        assert target["b"]["f"] == 6  # type: ignore  # Added
        assert target["g"] == 7  # Added

    def test_validate_migration_success(self, migrator):
        """Test validation when all files exist."""
        with patch.object(Path, "exists", return_value=True):
            result = migrator.validate_migration()
            assert result is True
            assert "successfully migrated" in migrator.migration_report[-1]

    def test_validate_migration_missing_profile(self, migrator):
        """Test validation when profile is missing."""
        with patch.object(Path, "exists") as mock_exists:
            # First call returns False (missing profile)
            mock_exists.side_effect = [False] + [True] * 20

            result = migrator.validate_migration()
            assert result is False
            assert "Missing profile" in migrator.migration_report[-1]

    def test_generate_migration_report(self, migrator):
        """Test migration report generation."""
        migrator.migration_report = ["Item 1", "Item 2", "Item 3"]
        report = migrator.generate_migration_report()

        assert "Configuration Migration Report" in report
        assert "Item 1" in report
        assert "Item 2" in report
        assert "Item 3" in report
        assert "Total items processed: 3" in report

    def test_run_migration_success(self, migrator):
        """Test successful migration run."""
        with patch.object(
            migrator, "convert_baseline", return_value={"profile": {"name": "default"}}
        ):
            with patch.object(
                migrator, "convert_conservative", return_value={"profile": {"name": "conservative"}}
            ):
                with patch.object(
                    migrator, "convert_optimistic", return_value={"profile": {"name": "aggressive"}}
                ):
                    with patch.object(migrator, "extract_modules"):
                        with patch.object(migrator, "create_presets"):
                            with patch.object(migrator, "validate_migration", return_value=True):
                                with patch("builtins.open", create=True):
                                    with patch("yaml.dump"):
                                        result = migrator.run_migration()
                                        assert result is True

    def test_run_migration_failure(self, migrator):
        """Test migration failure handling."""
        with patch.object(
            migrator, "convert_baseline", side_effect=FileNotFoundError("Test error")
        ):
            result = migrator.run_migration()
            assert result is False
            assert "Migration failed" in migrator.migration_report[-1]

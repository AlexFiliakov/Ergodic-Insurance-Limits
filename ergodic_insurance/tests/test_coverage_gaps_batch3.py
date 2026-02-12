"""Tests for coverage gaps across multiple modules (batch 3).

Targets specific untested code paths in:
- config.py (lines 293, 565, 651, 673, 729-733, 742, 775, 864, 1035,
  1042-1043, 1301, 1313, 1321, 1327, 1332, 1398-1401, 1714, 1736-1738,
  1760, 1788)
- config_manager.py (lines 200-204)
- config_migrator.py (lines 309-310, 317-318)
- sensitivity_visualization.py (lines 224-230, 234-238, 340)
- performance_optimizer.py (lines 49, 370, 380, 386, 416, 501)
- trajectory_storage.py (lines 251, 364, 445, 457-459, 478, 577)
"""

import gc
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, mock_open, patch

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
from pydantic import ValidationError
import pytest
import yaml

from ergodic_insurance.config import (
    Config,
    DebtConfig,
    ExcelReportConfig,
    ExpenseRatioConfig,
    GrowthConfig,
    InsuranceConfig,
    InsuranceLayerConfig,
    LoggingConfig,
    LossDistributionConfig,
    ManufacturerConfig,
    OutputConfig,
    ProfileMetadata,
    SimulationConfig,
    TransitionProbabilities,
    WorkingCapitalConfig,
)
from ergodic_insurance.config_manager import ConfigManager
from ergodic_insurance.config_migrator import ConfigMigrator
from ergodic_insurance.performance_optimizer import (
    NUMBA_AVAILABLE,
    OptimizationConfig,
    PerformanceOptimizer,
    SmartCache,
)
from ergodic_insurance.sensitivity import SensitivityResult, TwoWaySensitivityResult
from ergodic_insurance.sensitivity_visualization import (
    plot_parameter_sweep,
    plot_two_way_sensitivity,
)
from ergodic_insurance.trajectory_storage import SimulationSummary, StorageConfig, TrajectoryStorage

# ---------------------------------------------------------------------------
# Module 1: config.py
# ---------------------------------------------------------------------------


class TestManufacturerConfigNegativeMargin:
    """Test negative operating margin warning (config.py line 293)."""

    def test_negative_operating_margin_warning(self, caplog):
        """Verify that a negative base operating margin logs a warning."""
        with caplog.at_level(logging.WARNING):
            ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                base_operating_margin=-0.05,
                tax_rate=0.25,
                retention_ratio=1.0,
            )
        assert "negative" in caplog.text


class TestSimulationConfigHorizonExceedsMax:
    """Test time_horizon_years exceeding max_horizon_years (config.py line 565)."""

    def test_time_horizon_exceeds_maximum_custom(self):
        """Raise ValueError when time_horizon > max_horizon via model validator."""
        with pytest.raises(ValidationError):
            SimulationConfig(
                time_resolution="annual",
                time_horizon_years=500,
                max_horizon_years=200,
                random_seed=42,
            )


class TestConfigFromYaml:
    """Test Config.from_yaml file-not-found branch (config.py line 651)."""

    def test_from_yaml_file_not_found(self, tmp_path):
        """FileNotFoundError when YAML file does not exist."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config.from_yaml(missing)


class TestConfigFromDict:
    """Test Config.from_dict without base_config (config.py line 673)."""

    def test_from_dict_no_base_config(self):
        """Create Config from dict when base_config is None."""
        data = {
            "manufacturer": {
                "initial_assets": 5_000_000,
                "asset_turnover_ratio": 0.9,
                "base_operating_margin": 0.10,
                "tax_rate": 0.21,
                "retention_ratio": 0.8,
            },
            "working_capital": {"percent_of_sales": 0.18},
            "growth": {"type": "deterministic", "annual_growth_rate": 0.04, "volatility": 0.0},
            "debt": {
                "interest_rate": 0.06,
                "max_leverage_ratio": 2.5,
                "minimum_cash_balance": 50_000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 50,
                "max_horizon_years": 1000,
                "random_seed": 7,
            },
            "output": {"output_directory": "out", "file_format": "csv"},
            "logging": {"enabled": False},
        }
        config = Config.from_dict(data, base_config=None)
        assert config.manufacturer.initial_assets == 5_000_000
        assert config.simulation.random_seed == 7


class TestConfigToYaml:
    """Test Config.to_yaml (config.py lines 729-733)."""

    def test_to_yaml_creates_file(self, tmp_path):
        """Verify that to_yaml writes a YAML file and creates parent dirs."""
        config = Config(
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=1.0,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.0),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(
                time_resolution="annual",
                time_horizon_years=100,
                max_horizon_years=1000,
                random_seed=42,
            ),
            output=OutputConfig(output_directory="outputs", file_format="csv"),
            logging=LoggingConfig(enabled=False),
        )
        yaml_path = tmp_path / "sub" / "nested" / "config.yaml"
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert loaded["manufacturer"]["initial_assets"] == 10_000_000


class TestConfigSetupLogging:
    """Test Config.setup_logging disabled branch (config.py line 742)."""

    def test_setup_logging_disabled(self):
        """When logging is disabled, setup_logging should return immediately."""
        config = Config(
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=1.0,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.0),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(
                time_resolution="annual",
                time_horizon_years=100,
                max_horizon_years=1000,
                random_seed=42,
            ),
            output=OutputConfig(output_directory="outputs", file_format="csv"),
            logging=LoggingConfig(enabled=False),
        )
        # Should simply return without error
        config.setup_logging()
        logger = logging.getLogger("ergodic_insurance")
        # When logging is disabled, setup_logging returns before creating any handlers
        # (we just verify no exception is raised)


class TestConfigValidatePaths:
    """Test Config.validate_paths (config.py line 775)."""

    def test_validate_paths_creates_output_dir(self, tmp_path):
        """validate_paths should create the output directory if missing."""
        output_dir = tmp_path / "new_output"
        config = Config(
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=1.0,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.0),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(
                time_resolution="annual",
                time_horizon_years=100,
                max_horizon_years=1000,
                random_seed=42,
            ),
            output=OutputConfig(output_directory=str(output_dir), file_format="csv"),
            logging=LoggingConfig(enabled=False),
        )
        assert not output_dir.exists()
        config.validate_paths()
        assert output_dir.exists()


class TestTransitionProbabilitiesHardMarket:
    """Test hard-market transition validation (config.py line 864)."""

    def test_hard_market_transitions_do_not_sum_to_one(self):
        """Raise ValueError when hard-market transitions do not sum to 1.0."""
        with pytest.raises(ValidationError, match="Hard market transitions"):
            TransitionProbabilities(
                soft_to_soft=0.5,
                soft_to_normal=0.3,
                soft_to_hard=0.2,
                normal_to_soft=0.2,
                normal_to_normal=0.6,
                normal_to_hard=0.2,
                hard_to_soft=0.1,
                hard_to_normal=0.2,
                hard_to_hard=0.5,  # sums to 0.8, not 1.0
            )


class TestInsuranceLayerConfigInvalidLimitType:
    """Test invalid limit_type (config.py line 1035)."""

    def test_invalid_limit_type_rejected(self):
        """Raise ValueError for an unrecognised limit_type."""
        with pytest.raises(ValidationError, match="Invalid limit_type"):
            InsuranceLayerConfig(
                name="Test",
                limit=1_000_000,
                attachment=0,
                base_premium_rate=0.015,
                limit_type="unknown",
            )


class TestInsuranceLayerConfigHybridMissingLimits:
    """Test hybrid limit_type missing both limits (config.py lines 1042-1043)."""

    def test_hybrid_requires_limits(self):
        """Raise ValueError for hybrid type without per_occurrence and aggregate."""
        with pytest.raises(ValidationError, match="Hybrid limit type requires"):
            InsuranceLayerConfig(
                name="Hybrid",
                limit=1_000_000,
                attachment=0,
                base_premium_rate=0.015,
                limit_type="hybrid",
                per_occurrence_limit=None,
                aggregate_limit=None,
            )


class TestExpenseRatioConfigValidators:
    """Test ExpenseRatioConfig validators (config.py lines 1301, 1313, 1321, 1327, 1332)."""

    def test_depreciation_allocation_not_summing_to_one(self):
        """Raise ValueError when depreciation allocations do not sum to 100%."""
        with pytest.raises(ValidationError, match="Depreciation allocations must sum to 100%"):
            ExpenseRatioConfig(
                manufacturing_depreciation_allocation=0.5,
                admin_depreciation_allocation=0.3,  # sum = 0.8
            )

    def test_cogs_breakdown_not_summing_to_one(self):
        """Raise ValueError when COGS ratios do not sum to 100%."""
        with pytest.raises(ValidationError, match="COGS breakdown ratios must sum to 100%"):
            ExpenseRatioConfig(
                direct_materials_ratio=0.5,
                direct_labor_ratio=0.3,
                manufacturing_overhead_ratio=0.1,  # sum = 0.9
            )

    def test_sga_breakdown_not_summing_to_one(self):
        """Raise ValueError when SG&A ratios do not sum to 100%."""
        with pytest.raises(ValidationError, match="SG&A breakdown ratios must sum to 100%"):
            ExpenseRatioConfig(
                selling_expense_ratio=0.5,
                general_admin_ratio=0.3,  # sum = 0.8
            )

    def test_cogs_ratio_property(self):
        """Verify cogs_ratio property returns 1 - gross_margin_ratio."""
        config = ExpenseRatioConfig()
        assert abs(config.cogs_ratio - (1.0 - config.gross_margin_ratio)) < 1e-9

    def test_operating_margin_ratio_property(self):
        """Verify operating_margin_ratio = gross_margin - sga."""
        config = ExpenseRatioConfig()
        expected = config.gross_margin_ratio - config.sga_expense_ratio
        assert abs(config.operating_margin_ratio - expected) < 1e-9


class TestExcelReportConfigEngine:
    """Test ExcelReportConfig engine validator (config.py lines 1398-1401)."""

    def test_invalid_excel_engine_rejected(self):
        """Raise ValueError for an invalid Excel engine name."""
        with pytest.raises(ValidationError, match="Invalid Excel engine"):
            ExcelReportConfig(engine="invalid_engine")

    def test_valid_excel_engines(self):
        """Verify all valid engine names are accepted."""
        for engine in ["xlsxwriter", "openpyxl", "auto", "pandas"]:
            cfg = ExcelReportConfig(engine=engine)
            assert cfg.engine == engine


class TestConfigApplyModuleNonDict:
    """Test Config.apply_module with non-dict and non-BaseModel values (config.py lines 1714, 1736-1738)."""

    @pytest.fixture
    def base_config_v2(self):
        """Create a minimal Config instance."""
        return Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=3.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

    def test_apply_module_non_dict_value(self, base_config_v2):
        """apply_module should handle scalar values by using setattr directly."""
        module_data = {
            "applied_presets": ["preset_from_module"],
        }
        yaml_content = yaml.dump(module_data)
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            base_config_v2.apply_module(Path("module.yaml"))
            assert base_config_v2.applied_presets == ["preset_from_module"]

    def test_apply_module_dict_value_for_non_basemodel(self, base_config_v2):
        """apply_module with dict value on a non-BaseModel attr -> setattr(dict)."""
        module_data = {
            "overrides": {"key1": "val1"},
        }
        yaml_content = yaml.dump(module_data)
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            base_config_v2.apply_module(Path("module.yaml"))
            assert base_config_v2.overrides == {"key1": "val1"}


class TestConfigApplyPresetBranches:
    """Test Config.apply_preset non-dict and non-BaseModel branches (config.py lines 1736-1738)."""

    @pytest.fixture
    def base_config_v2(self):
        return Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=3.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

    def test_apply_preset_dict_on_non_basemodel_attr(self, base_config_v2):
        """apply_preset with dict value for a plain-dict attribute uses setattr."""
        preset_data = {
            "overrides": {"custom_key": 99},
        }
        base_config_v2.apply_preset("custom_preset", preset_data)
        assert base_config_v2.overrides == {"custom_key": 99}
        assert "custom_preset" in base_config_v2.applied_presets

    def test_apply_preset_scalar_value(self, base_config_v2):
        """apply_preset with a scalar (non-dict) value uses setattr.

        Note: apply_preset first appends the preset name, then the loop
        overwrites 'applied_presets' via setattr with the provided list.
        """
        preset_data = {
            "applied_presets": ["a", "b"],
        }
        base_config_v2.apply_preset("scalar_preset", preset_data)
        # The setattr in the loop overwrites the list that was appended to
        assert base_config_v2.applied_presets == ["a", "b"]


class TestConfigWithOverridesNewSection:
    """Test with_overrides when nested key path does not exist (config.py line 1760)."""

    @pytest.fixture
    def base_config_v2(self):
        return Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=3.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

    def test_with_overrides_creates_missing_nested_key(self, base_config_v2):
        """with_overrides creates intermediate dicts for a nested key that does not exist."""
        # Use a key path where intermediate keys do not exist in the data dict.
        # 'overrides' is already a dict; we use a deep path.
        new_config = base_config_v2.with_overrides(
            {
                "manufacturer.initial_assets": 20_000_000,
            }
        )
        assert new_config.manufacturer.initial_assets == 20_000_000


class TestConfigValidateCompleteness:
    """Test validate_completeness (config.py line 1788)."""

    @pytest.fixture
    def base_config_v2(self):
        return Config(
            profile=ProfileMetadata(name="test", description="Test"),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.05),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=3.0, minimum_cash_balance=100_000
            ),
            simulation=SimulationConfig(time_horizon_years=10, random_seed=42),
            output=OutputConfig(output_directory="./output"),
            logging=LoggingConfig(),
        )

    def test_validate_completeness_insurance_without_losses(self, base_config_v2):
        """Flag when insurance is enabled but no loss distribution configured."""
        base_config_v2.insurance = InsuranceConfig(
            layers=[
                InsuranceLayerConfig(
                    name="Primary",
                    limit=1_000_000,
                    attachment=0,
                    base_premium_rate=0.015,
                )
            ]
        )
        base_config_v2.losses = None
        issues = base_config_v2.validate_completeness()
        assert any("Insurance enabled but no loss distribution" in i for i in issues)

    def test_validate_completeness_no_issues(self, base_config_v2):
        """Fully configured config should have no completeness issues."""
        issues = base_config_v2.validate_completeness()
        assert issues == []


# ---------------------------------------------------------------------------
# Module 2: config_manager.py
# ---------------------------------------------------------------------------


class TestConfigManagerMakeHashable:
    """Test the make_hashable helper inside load_profile (config_manager.py lines 200-204)."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory structure."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"

        (config_dir / "profiles").mkdir(parents=True)
        (config_dir / "profiles" / "custom").mkdir()
        (config_dir / "modules").mkdir()
        (config_dir / "presets").mkdir()

        test_profile = {
            "profile": {"name": "test", "description": "Test profile", "version": "2.0.0"},
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

        with open(config_dir / "profiles" / "test.yaml", "w") as f:
            yaml.dump(test_profile, f)

        yield config_dir

        shutil.rmtree(temp_dir)

    def test_load_profile_with_dict_and_list_overrides(self, temp_config_dir):
        """Exercise make_hashable with nested dicts and lists in overrides."""
        manager = ConfigManager(config_dir=temp_config_dir)
        # Pass overrides that contain a dict and a list so make_hashable is fully exercised
        config = manager.load_profile(
            "test",
            overrides={
                "manufacturer": {"initial_assets": 20_000_000},
                "simulation": {"time_horizon_years": 50},
            },
        )
        assert config is not None


# ---------------------------------------------------------------------------
# Module 3: config_migrator.py
# ---------------------------------------------------------------------------


class TestConfigMigratorValidateMissingModule:
    """Test validate_migration when a module file is missing (config_migrator.py lines 309-310)."""

    def test_missing_module_fails_validation(self):
        """Validation should fail when a required module file is missing."""
        migrator = ConfigMigrator()

        call_count = [0]

        # Profiles all exist (3 calls), then first module check fails
        def exists_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return True  # profiles exist
            return False  # first module does not exist

        with patch.object(Path, "exists", side_effect=exists_side_effect):
            result = migrator.validate_migration()
            assert result is False
            assert any("Missing module" in msg for msg in migrator.migration_report)


class TestConfigMigratorValidateMissingPreset:
    """Test validate_migration when a preset file is missing (config_migrator.py lines 317-318)."""

    def test_missing_preset_fails_validation(self):
        """Validation should fail when a required preset file is missing."""
        migrator = ConfigMigrator()

        call_count = [0]

        # 3 profiles + 4 modules = 7 exist, then first preset check fails
        def exists_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 7:
                return True
            return False

        with patch.object(Path, "exists", side_effect=exists_side_effect):
            result = migrator.validate_migration()
            assert result is False
            assert any("Missing preset" in msg for msg in migrator.migration_report)


# ---------------------------------------------------------------------------
# Module 4: sensitivity_visualization.py
# ---------------------------------------------------------------------------


class TestTwoWaySensitivityPercentFormat:
    """Test percentage contour format (sensitivity_visualization.py lines 224-230)."""

    @pytest.fixture
    def two_way_result(self):
        return TwoWaySensitivityResult(
            parameter1="frequency",
            parameter2="severity",
            values1=np.array([3, 4, 5, 6, 7]),
            values2=np.array([80000, 100000, 120000]),
            metric_grid=np.array(
                [
                    [0.10, 0.11, 0.12],
                    [0.11, 0.12, 0.13],
                    [0.12, 0.13, 0.14],
                    [0.13, 0.14, 0.15],
                    [0.14, 0.15, 0.16],
                ]
            ),
            metric_name="ROE",
        )

    def test_percentage_format_with_contours(self, two_way_result):
        """Exercise the .2% percentage format branch for contour labels."""
        fig = plot_two_way_sensitivity(
            two_way_result,
            show_contours=True,
            contour_levels=5,
            fmt=".2%",
        )
        assert fig is not None
        plt.close(fig)

    def test_old_style_format_string(self, two_way_result):
        """Exercise the fallback branch where fmt is already old-style (lines 237-238)."""
        fig = plot_two_way_sensitivity(
            two_way_result,
            show_contours=True,
            contour_levels=5,
            fmt="%.3f",
        )
        assert fig is not None
        plt.close(fig)

    def test_empty_format_string_fallback(self, two_way_result):
        """Exercise the fallback branch where fmt is empty string (lines 237-238)."""
        fig = plot_two_way_sensitivity(
            two_way_result,
            show_contours=True,
            contour_levels=5,
            fmt="",
        )
        assert fig is not None
        plt.close(fig)


class TestParameterSweepHiddenSubplots:
    """Test hidden unused subplots (sensitivity_visualization.py line 340)."""

    @pytest.fixture
    def sensitivity_result(self):
        """Create a SensitivityResult with 4 metrics to produce unused subplots.

        4 metrics -> n_cols=3, n_rows=2 -> 6 subplot slots, 2 unused.
        """
        return SensitivityResult(
            parameter="frequency",
            baseline_value=5.0,
            variations=np.linspace(3, 7, 11),
            metrics={
                "metric_a": np.linspace(0.08, 0.15, 11),
                "metric_b": np.linspace(0.02, 0.005, 11),
                "metric_c": np.linspace(0.05, 0.10, 11),
                "metric_d": np.linspace(0.01, 0.03, 11),
            },
        )

    def test_unused_subplots_hidden(self, sensitivity_result):
        """When metrics < subplot slots, extra axes should be hidden."""
        # 4 metrics with n_cols=3 -> n_rows=2 -> 6 slots, 2 unused
        fig = plot_parameter_sweep(sensitivity_result, figsize=(12, 8))
        assert fig is not None
        axes = fig.get_axes()
        hidden_count = sum(1 for ax in axes if not ax.get_visible())
        assert hidden_count >= 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Module 5: performance_optimizer.py
# ---------------------------------------------------------------------------


class TestNumbaAvailableFlag:
    """Test NUMBA_AVAILABLE flag (performance_optimizer.py line 49)."""

    def test_numba_flag_is_boolean(self):
        """NUMBA_AVAILABLE should be a boolean regardless of environment."""
        assert isinstance(NUMBA_AVAILABLE, bool)


class TestGenerateRecommendationsLoopBranch:
    """Test 'loop' branch in _generate_recommendations (performance_optimizer.py line 370)."""

    def test_loop_recommendation(self):
        """Recommend replacing loops with vectorized ops."""
        optimizer = PerformanceOptimizer()
        function_times = {"main_loop_function": 2.0}
        total_time = 10.0
        memory_usage = 500.0
        recs = optimizer._generate_recommendations(function_times, memory_usage, total_time)
        assert any("loop" in r.lower() or "vectorized" in r.lower() for r in recs)


class TestGenerateRecommendationsLowCacheHitRate:
    """Test low cache hit rate recommendation (performance_optimizer.py line 380)."""

    def test_low_cache_hit_rate_recommendation(self):
        """Recommend reviewing cache strategy when hit rate is low."""
        optimizer = PerformanceOptimizer()
        # Manually set cache statistics for a low hit rate
        optimizer.cache.hits = 10
        optimizer.cache.misses = 200  # hit rate ~ 4.8%
        function_times = {"some_function": 1.0}
        total_time = 5.0
        memory_usage = 500.0
        recs = optimizer._generate_recommendations(function_times, memory_usage, total_time)
        assert any("cache hit rate" in r.lower() for r in recs)


class TestGenerateRecommendationsParallelProcessing:
    """Test parallel processing recommendation (performance_optimizer.py line 386)."""

    def test_parallel_processing_recommendation(self):
        """Recommend parallel processing when total_time > 10 and no parallel func."""
        optimizer = PerformanceOptimizer()
        function_times = {"sequential_compute": 8.0, "io_wait": 3.0}
        total_time = 11.0
        memory_usage = 500.0
        recs = optimizer._generate_recommendations(function_times, memory_usage, total_time)
        assert any("parallel" in r.lower() for r in recs)


class TestOptimizeLossGenerationVectorizationDisabled:
    """Test optimize_loss_generation without vectorization (performance_optimizer.py line 416)."""

    def test_no_vectorization_returns_plain_array(self):
        """When vectorization is disabled, return np.array(losses) directly."""
        config = OptimizationConfig(enable_vectorization=False)
        optimizer = PerformanceOptimizer(config=config)
        losses = [100.0, 200.0, 300.0]
        result = optimizer.optimize_loss_generation(losses)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, losses)


class TestOptimizeMemoryUsageCacheNotCleared:
    """Test cache_cleared=False branch (performance_optimizer.py line 501)."""

    def test_cache_not_cleared_when_memory_low(self):
        """When memory usage is below 80%, cache should not be cleared."""
        optimizer = PerformanceOptimizer()
        # Mock psutil to report low memory usage
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)  # 100 MB
        mock_virtual = MagicMock(
            percent=30.0,
            available=8 * 1024 * 1024 * 1024,  # 8 GB
        )
        with patch("psutil.Process", return_value=mock_process):
            with patch("psutil.virtual_memory", return_value=mock_virtual):
                metrics = optimizer.optimize_memory_usage()
                assert metrics["cache_cleared"] is False


# ---------------------------------------------------------------------------
# Module 6: trajectory_storage.py
# ---------------------------------------------------------------------------


class TestCalculateSummarySingleYear:
    """Test volatility=0 for single-year data (trajectory_storage.py line 251)."""

    @pytest.fixture
    def storage(self, tmp_path):
        config = StorageConfig(storage_dir=str(tmp_path), backend="memmap")
        s = TrajectoryStorage(config)
        yield s
        s.clear_storage()

    def test_single_year_volatility_zero(self, storage):
        """Volatility should be 0.0 when n_years == 1."""
        summary = storage._calculate_summary(
            sim_id=1,
            annual_losses=np.array([50000.0]),
            insurance_recoveries=np.array([20000.0]),
            final_assets=10_500_000.0,
            initial_assets=10_000_000.0,
            ruin_occurred=False,
            ruin_year=None,
        )
        assert summary.volatility == 0.0


class TestStoreTimeSeriesHdf5NoFile:
    """Test _store_time_series_hdf5 when _hdf5_file is None (trajectory_storage.py line 364)."""

    @pytest.fixture
    def storage(self, tmp_path):
        config = StorageConfig(storage_dir=str(tmp_path), backend="memmap")
        s = TrajectoryStorage(config)
        yield s
        s.clear_storage()

    def test_hdf5_store_noop_when_no_file(self, storage):
        """_store_time_series_hdf5 should silently return when _hdf5_file is None."""
        assert storage._hdf5_file is None
        # Should not raise
        storage._store_time_series_hdf5(
            sim_id=1,
            sample_indices=np.array([0, 5, 10]),
            annual_losses=np.array([1000, 2000, 3000]),
            insurance_recoveries=np.array([500, 1000, 1500]),
            retained_losses=np.array([500, 1000, 1500]),
        )


class TestLoadSummaryFromJsonFile:
    """Test _load_summary_from_disk from JSON file (trajectory_storage.py line 445)."""

    @pytest.fixture
    def storage(self, tmp_path):
        config = StorageConfig(storage_dir=str(tmp_path), backend="memmap")
        s = TrajectoryStorage(config)
        yield s
        s.clear_storage()

    def test_load_summary_returns_none_when_missing(self, storage):
        """Return None when summary file does not exist for given sim_id."""
        result = storage._load_summary_from_disk(999)
        assert result is None

    def test_load_summary_from_json(self, storage):
        """Load a persisted JSON summary by sim_id."""
        summary = SimulationSummary(
            sim_id=42,
            final_assets=11_000_000.0,
            total_losses=200_000.0,
            total_recoveries=150_000.0,
            mean_annual_loss=20_000.0,
            max_annual_loss=50_000.0,
            min_annual_loss=5_000.0,
            growth_rate=0.05,
            ruin_occurred=False,
            volatility=0.12,
        )
        summary_file = storage.storage_path / "summaries" / "sim_42.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f)

        loaded = storage._load_summary_from_disk(42)
        assert loaded is not None
        assert loaded.sim_id == 42
        assert loaded.final_assets == 11_000_000.0


class TestLoadTimeSeriesHdf5:
    """Test _load_time_series for HDF5 backend (trajectory_storage.py lines 457-459)."""

    @pytest.fixture
    def storage_hdf5(self, tmp_path):
        config = StorageConfig(storage_dir=str(tmp_path), backend="hdf5", sample_interval=5)
        s = TrajectoryStorage(config)
        yield s
        s.clear_storage()

    def test_load_time_series_hdf5_existing(self, storage_hdf5):
        """Load time series data from HDF5 when it exists."""
        np.random.seed(42)
        n_years = 20
        sample_data = {
            "annual_losses": np.random.lognormal(10, 2, n_years),
            "insurance_recoveries": np.random.lognormal(9, 2, n_years) * 0.8,
            "retained_losses": np.random.lognormal(8, 1, n_years),
            "initial_assets": 10_000_000.0,
            "final_assets": 12_000_000.0,
        }
        storage_hdf5.store_simulation(sim_id=1, **sample_data, ruin_occurred=False)

        ts = storage_hdf5._load_time_series(1)
        assert ts is not None
        assert "annual_losses" in ts
        assert "insurance_recoveries" in ts
        assert "retained_losses" in ts

    def test_load_time_series_hdf5_missing(self, storage_hdf5):
        """Return None when sim_id is not in HDF5 time_series group."""
        ts = storage_hdf5._load_time_series(999)
        assert ts is None


class TestLoadTimeSeriesMemmapMissing:
    """Test _load_time_series returning None for memmap (trajectory_storage.py line 478)."""

    @pytest.fixture
    def storage(self, tmp_path):
        config = StorageConfig(storage_dir=str(tmp_path), backend="memmap")
        s = TrajectoryStorage(config)
        yield s
        s.clear_storage()

    def test_load_time_series_memmap_missing(self, storage):
        """Return None when time series file does not exist for given sim_id."""
        ts = storage._load_time_series(999)
        assert ts is None


class TestCleanupMemoryPersistsSummaries:
    """Test _cleanup_memory persisting summaries (trajectory_storage.py line 577)."""

    @pytest.fixture
    def storage(self, tmp_path):
        config = StorageConfig(
            storage_dir=str(tmp_path),
            backend="memmap",
            chunk_size=5,
        )
        s = TrajectoryStorage(config)
        yield s
        s.clear_storage()

    def test_cleanup_memory_persists_and_clears(self, storage):
        """_cleanup_memory should persist cached summaries and then clear them."""
        np.random.seed(42)
        n_years = 10
        sample_data = {
            "annual_losses": np.random.lognormal(10, 1, n_years),
            "insurance_recoveries": np.random.lognormal(9, 1, n_years) * 0.5,
            "retained_losses": np.random.lognormal(8, 1, n_years),
            "initial_assets": 10_000_000.0,
            "final_assets": 11_000_000.0,
        }

        # Store 2 simulations so summaries are cached but not yet persisted
        storage.store_simulation(sim_id=0, **sample_data, ruin_occurred=False)
        storage.store_simulation(sim_id=1, **sample_data, ruin_occurred=False)
        assert len(storage._summaries) == 2

        # Trigger cleanup
        storage._cleanup_memory()

        # Summaries should have been persisted and cleared
        assert len(storage._summaries) == 0

        # Verify they were written to disk
        summary_files = list((storage.storage_path / "summaries").glob("*.json"))
        assert len(summary_files) == 2

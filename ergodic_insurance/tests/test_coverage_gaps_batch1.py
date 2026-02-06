"""Tests targeting specific untested code paths across multiple modules.

This test file covers coverage gaps identified in:
- insurance_accounting.py (lines 39, 41, 97, 99, 101)
- config_loader.py (lines 204-206, 213, 237, 278-281)
- monte_carlo_worker.py (lines 117, 157-161)
- statistical_tests.py (lines 196, 329, 415, 524, 641)
- result_aggregator.py (lines 36-38, 221, 234, 535, 576)
"""

from decimal import Decimal
import logging
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pytest
import yaml

from ergodic_insurance.config_loader import ConfigLoader
from ergodic_insurance.insurance_accounting import InsuranceAccounting, InsuranceRecovery
from ergodic_insurance.result_aggregator import (
    AggregationConfig,
    HierarchicalAggregator,
    ResultAggregator,
    ResultExporter,
)
from ergodic_insurance.statistical_tests import (
    HypothesisTestResult,
    bootstrap_hypothesis_test,
    multiple_comparison_correction,
    paired_comparison_test,
    ratio_of_metrics_test,
)

# ===========================================================================
# Module 1: insurance_accounting.py - __post_init__ Decimal conversions
# Lines 39, 41 in InsuranceRecovery.__post_init__
# Lines 97, 99, 101 in InsuranceAccounting.__post_init__
# ===========================================================================


class TestInsuranceRecoveryDecimalConversion:
    """Test __post_init__ Decimal conversion in InsuranceRecovery (lines 39, 41)."""

    def test_float_amount_converted_to_decimal(self):
        """Passing a float for amount triggers the isinstance branch and converts to Decimal (line 39)."""
        recovery = InsuranceRecovery(amount=100.0, claim_id="C1", year_approved=2024)  # type: ignore[arg-type]
        assert isinstance(recovery.amount, Decimal)
        # Verify the numeric value is correct after conversion
        assert recovery.amount == Decimal("100.0")

    def test_int_amount_converted_to_decimal(self):
        """Passing an int for amount triggers the isinstance branch and converts to Decimal (line 39)."""
        recovery = InsuranceRecovery(amount=250, claim_id="C2", year_approved=2024)  # type: ignore[arg-type]
        assert isinstance(recovery.amount, Decimal)
        assert recovery.amount == Decimal("250")

    def test_float_amount_received_converted_to_decimal(self):
        """Passing a float for amount_received triggers conversion to Decimal (line 41)."""
        recovery = InsuranceRecovery(
            amount=Decimal("100"),
            claim_id="C1",
            year_approved=2024,
            amount_received=50.0,  # type: ignore[arg-type]
        )
        assert isinstance(recovery.amount_received, Decimal)
        assert recovery.amount_received == Decimal("50.0")

    def test_both_amount_fields_as_floats_converted(self):
        """Both amount and amount_received as floats should both be converted (lines 39, 41)."""
        recovery = InsuranceRecovery(
            amount=200.0,  # type: ignore[arg-type]
            claim_id="C3",
            year_approved=2024,
            amount_received=75.0,  # type: ignore[arg-type]
        )
        assert isinstance(recovery.amount, Decimal)
        assert isinstance(recovery.amount_received, Decimal)
        assert recovery.amount == Decimal("200.0")
        assert recovery.amount_received == Decimal("75.0")

    def test_decimal_amount_remains_unchanged(self):
        """Passing a Decimal for amount should leave it unchanged (no conversion needed)."""
        recovery = InsuranceRecovery(
            amount=Decimal("100.00"),
            claim_id="C4",
            year_approved=2024,
        )
        assert isinstance(recovery.amount, Decimal)
        assert recovery.amount == Decimal("100.00")


class TestInsuranceAccountingDecimalConversion:
    """Test __post_init__ Decimal conversion in InsuranceAccounting (lines 97, 99, 101)."""

    def test_float_prepaid_insurance_converted(self):
        """Passing float for prepaid_insurance triggers Decimal conversion (line 97)."""
        acct = InsuranceAccounting(prepaid_insurance=1000.0)  # type: ignore[arg-type]
        assert isinstance(acct.prepaid_insurance, Decimal)

    def test_float_monthly_expense_converted(self):
        """Passing float for monthly_expense triggers Decimal conversion (line 99)."""
        acct = InsuranceAccounting(monthly_expense=100.0)  # type: ignore[arg-type]
        assert isinstance(acct.monthly_expense, Decimal)

    def test_float_annual_premium_converted(self):
        """Passing float for annual_premium triggers Decimal conversion (line 101)."""
        acct = InsuranceAccounting(annual_premium=1200.0)  # type: ignore[arg-type]
        assert isinstance(acct.annual_premium, Decimal)

    def test_all_float_fields_converted(self):
        """All three float fields should be converted to Decimal (lines 97, 99, 101)."""
        acct = InsuranceAccounting(
            prepaid_insurance=1000.0,  # type: ignore[arg-type]
            monthly_expense=100.0,  # type: ignore[arg-type]
            annual_premium=1200.0,  # type: ignore[arg-type]
        )
        assert isinstance(acct.prepaid_insurance, Decimal)
        assert isinstance(acct.monthly_expense, Decimal)
        assert isinstance(acct.annual_premium, Decimal)

    def test_int_values_converted(self):
        """Integer values should also be converted to Decimal (lines 97, 99, 101)."""
        acct = InsuranceAccounting(
            prepaid_insurance=1000,  # type: ignore[arg-type]
            monthly_expense=100,  # type: ignore[arg-type]
            annual_premium=1200,  # type: ignore[arg-type]
        )
        assert isinstance(acct.prepaid_insurance, Decimal)
        assert isinstance(acct.monthly_expense, Decimal)
        assert isinstance(acct.annual_premium, Decimal)

    def test_decimal_values_unchanged(self):
        """Decimal values should remain as Decimal without re-conversion."""
        acct = InsuranceAccounting(
            prepaid_insurance=Decimal("1000"),
            monthly_expense=Decimal("100"),
            annual_premium=Decimal("1200"),
        )
        assert isinstance(acct.prepaid_insurance, Decimal)
        assert acct.prepaid_insurance == Decimal("1000")
        assert acct.monthly_expense == Decimal("100")
        assert acct.annual_premium == Decimal("1200")


# ===========================================================================
# Module 2: config_loader.py - Validation warnings and scenario loading
# Lines 204-206: monthly time resolution with long horizon warning
# Line 213: zero retention with positive growth warning
# Line 237: load_pricing_scenarios with .yaml extension
# Lines 278-281: switch_pricing_scenario logging
# ===========================================================================


class TestConfigLoaderValidationWarnings:
    """Test validation warnings in config_loader.py (lines 204-206, 213)."""

    def test_validate_config_monthly_long_horizon_triggers_warning(self, caplog):
        """Monthly resolution with >1000 year horizon should warn about excessive periods (lines 204-206).

        With time_horizon_years=1001 and monthly resolution, total_periods = 1001 * 12 = 12012,
        which exceeds the 12000 threshold and triggers the logging warning.
        """
        loader = ConfigLoader()

        mock_config = MagicMock()
        mock_config.simulation.time_resolution = "monthly"
        mock_config.simulation.time_horizon_years = 1001
        mock_config.manufacturer.retention_ratio = 0.5
        mock_config.growth.annual_growth_rate = 0.05

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.config_loader"):
            result = loader.validate_config(mock_config)

        assert result is True
        assert "Consider using annual resolution" in caplog.text
        assert "12012 periods" in caplog.text

    def test_validate_config_zero_retention_positive_growth_triggers_warning(self, caplog):
        """Zero retention_ratio with positive growth_rate should log an inconsistency warning (line 213)."""
        loader = ConfigLoader()

        mock_config = MagicMock()
        mock_config.simulation.time_resolution = "annual"
        mock_config.simulation.time_horizon_years = 10
        mock_config.manufacturer.retention_ratio = 0
        mock_config.growth.annual_growth_rate = 0.05

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.config_loader"):
            result = loader.validate_config(mock_config)

        assert result is True
        assert "Zero retention with positive growth rate" in caplog.text

    def test_validate_config_monthly_short_horizon_no_warning(self, caplog):
        """Monthly resolution with short horizon (< 1000 years) should not trigger the warning."""
        loader = ConfigLoader()

        mock_config = MagicMock()
        mock_config.simulation.time_resolution = "monthly"
        mock_config.simulation.time_horizon_years = 50  # 600 periods, well below 12000
        mock_config.manufacturer.retention_ratio = 0.5
        mock_config.growth.annual_growth_rate = 0.05

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.config_loader"):
            result = loader.validate_config(mock_config)

        assert result is True
        assert "Consider using annual resolution" not in caplog.text

    def test_validate_config_annual_resolution_long_horizon_no_warning(self, caplog):
        """Annual resolution should not trigger the monthly periods warning regardless of horizon."""
        loader = ConfigLoader()

        mock_config = MagicMock()
        mock_config.simulation.time_resolution = "annual"
        mock_config.simulation.time_horizon_years = 5000
        mock_config.manufacturer.retention_ratio = 0.5
        mock_config.growth.annual_growth_rate = 0.05

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.config_loader"):
            result = loader.validate_config(mock_config)

        assert result is True
        assert "Consider using annual resolution" not in caplog.text


class TestConfigLoaderPricingScenarios:
    """Test pricing scenario loading (lines 237, 278-281)."""

    def _create_pricing_scenario_yaml(self, filepath):
        """Helper to create a valid pricing scenario YAML file.

        Args:
            filepath: Path where the YAML file should be written.
        """
        scenario_data = {
            "scenarios": {
                "baseline": {
                    "name": "Normal Market",
                    "description": "Normal market conditions",
                    "market_condition": "normal",
                    "primary_layer_rate": 0.015,
                    "first_excess_rate": 0.01,
                    "higher_excess_rate": 0.005,
                    "capacity_factor": 1.0,
                    "competition_level": "moderate",
                    "retention_discount": 0.1,
                    "volume_discount": 0.05,
                    "loss_ratio_target": 0.6,
                    "expense_ratio": 0.35,
                    "new_business_appetite": "selective",
                    "renewal_retention_focus": "balanced",
                    "coverage_enhancement_willingness": "moderate",
                },
                "inexpensive": {
                    "name": "Soft Market",
                    "description": "Soft market conditions with lower rates",
                    "market_condition": "soft",
                    "primary_layer_rate": 0.01,
                    "first_excess_rate": 0.007,
                    "higher_excess_rate": 0.003,
                    "capacity_factor": 1.5,
                    "competition_level": "high",
                    "retention_discount": 0.15,
                    "volume_discount": 0.1,
                    "loss_ratio_target": 0.65,
                    "expense_ratio": 0.30,
                    "new_business_appetite": "aggressive",
                    "renewal_retention_focus": "high",
                    "coverage_enhancement_willingness": "high",
                },
            },
            "market_cycles": {
                "average_duration_years": 4.0,
                "soft_market_duration": 4.0,
                "normal_market_duration": 5.0,
                "hard_market_duration": 3.0,
                "transition_probabilities": {
                    "soft_to_soft": 0.5,
                    "soft_to_normal": 0.3,
                    "soft_to_hard": 0.2,
                    "normal_to_soft": 0.2,
                    "normal_to_normal": 0.65,
                    "normal_to_hard": 0.15,
                    "hard_to_soft": 0.1,
                    "hard_to_normal": 0.35,
                    "hard_to_hard": 0.55,
                },
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(scenario_data, f)

    def test_load_pricing_scenarios_with_yaml_extension(self, tmp_path):
        """Filename containing '.yaml' should be treated as a direct path (line 237).

        When the scenario_file argument contains '.yaml', the code takes
        the branch at line 237 that constructs a Path directly from the
        filename instead of looking in the config_dir.
        """
        yaml_file = tmp_path / "test_pricing.yaml"
        self._create_pricing_scenario_yaml(yaml_file)

        loader = ConfigLoader()
        pricing_config = loader.load_pricing_scenarios(str(yaml_file))

        assert pricing_config is not None
        assert "baseline" in pricing_config.scenarios
        assert "inexpensive" in pricing_config.scenarios

    def test_load_pricing_scenarios_with_yml_extension(self, tmp_path):
        """Filename containing '.yml' should also be treated as a direct path (line 237)."""
        yml_file = tmp_path / "test_pricing.yml"
        self._create_pricing_scenario_yaml(yml_file)

        loader = ConfigLoader()
        pricing_config = loader.load_pricing_scenarios(str(yml_file))

        assert pricing_config is not None
        assert "baseline" in pricing_config.scenarios

    def test_switch_pricing_scenario_logs_rates(self, caplog):
        """switch_pricing_scenario should log scenario name and rates (lines 278-281).

        When the config has an 'insurance' key, the function logs the
        scenario name, primary rate, first excess rate, and higher excess rate.
        """
        loader = ConfigLoader()

        # Create mock pricing scenario with rate attributes
        mock_scenario = MagicMock()
        mock_scenario.name = "Soft Market"
        mock_scenario.primary_layer_rate = 0.01
        mock_scenario.first_excess_rate = 0.007
        mock_scenario.higher_excess_rate = 0.003

        mock_pricing_config = MagicMock()
        mock_pricing_config.get_scenario.return_value = mock_scenario

        # Create mock config that includes an 'insurance' section in model_dump
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {
            "insurance": {"some_key": "some_value"},
            "manufacturer": {},
        }

        with patch.object(loader, "load_pricing_scenarios", return_value=mock_pricing_config):
            with patch("ergodic_insurance.config_loader.Config") as MockConfig:
                MockConfig.return_value = MagicMock()

                with caplog.at_level(logging.INFO, logger="ergodic_insurance.config_loader"):
                    loader.switch_pricing_scenario(mock_config, "inexpensive")

        assert "Switching to Soft Market pricing scenario" in caplog.text
        assert "Primary rate:" in caplog.text
        assert "First excess rate:" in caplog.text
        assert "Higher excess rate:" in caplog.text


# ===========================================================================
# Module 3: monte_carlo_worker.py - Error handling and ruin marking
# Line 117: AttributeError when loss_generator has no generate_losses
# Lines 157-161: Ruin marking when equity drops below insolvency_tolerance
# ===========================================================================


class TestMonteCarloWorkerErrorHandling:
    """Test error handling in monte_carlo_worker.py (line 117)."""

    def test_loss_generator_without_generate_losses_raises_attribute_error(self):
        """A loss generator without generate_losses method should raise AttributeError (line 117).

        The code checks hasattr(loss_generator, 'generate_losses') and raises
        AttributeError with a descriptive message when the method is missing.
        """
        from ergodic_insurance.monte_carlo_worker import run_chunk_standalone

        # Create a loss generator that lacks generate_losses method
        class IncompleteLossGenerator:
            """A loss generator missing the required generate_losses method."""

        bad_generator = IncompleteLossGenerator()

        # Create a mock manufacturer with the minimum required interface
        mock_manufacturer = MagicMock()
        mock_manufacturer.stochastic_process = None
        mock_manufacturer.config = MagicMock()
        mock_manufacturer.config.initial_assets = 10_000_000
        mock_manufacturer.config.asset_turnover_ratio = 0.8

        # Mock the sim_manufacturer returned by create_fresh
        mock_sim_manufacturer = MagicMock()
        mock_sim_manufacturer.calculate_revenue.return_value = Decimal("1000000")
        mock_sim_manufacturer.equity = Decimal("10000000")
        mock_sim_manufacturer.total_assets = Decimal("10000000")

        # Create mock insurance program
        mock_insurance = MagicMock()
        mock_insurance.calculate_annual_premium.return_value = 50000.0

        config_dict = {
            "n_years": 1,
            "use_float32": False,
        }

        chunk = (0, 1, None)  # seed=None to skip reseed call

        # Patch create_fresh to avoid building a real WidgetManufacturer from the mock config
        with patch(
            "ergodic_insurance.monte_carlo_worker.WidgetManufacturer.create_fresh",
            return_value=mock_sim_manufacturer,
        ):
            with pytest.raises(AttributeError, match="has no generate_losses method"):
                run_chunk_standalone(
                    chunk=chunk,
                    loss_generator=bad_generator,  # type: ignore[arg-type]
                    insurance_program=mock_insurance,
                    manufacturer=mock_manufacturer,
                    config_dict=config_dict,
                )


class TestMonteCarloWorkerRuinMarking:
    """Test ruin marking in monte_carlo_worker.py (lines 157-161)."""

    def test_ruin_marked_for_future_evaluation_years(self):
        """When equity drops below insolvency_tolerance, future eval years are marked ruined (lines 157-161).

        This test creates a tiny manufacturer whose equity cannot absorb
        the large uninsured loss.  After the first year's step the equity
        check triggers ruin marking for all future evaluation years.
        """
        from ergodic_insurance.config import ManufacturerConfig
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.monte_carlo_worker import run_chunk_standalone

        # Tiny company: initial equity ≈ 50k, loss = 5M → guaranteed ruin.
        # Disable mid-year liquidity check so the full loss flows through
        # the accounting step and equity actually drops below tolerance.
        mfg_config = ManufacturerConfig(
            initial_assets=50_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.9,
            check_intra_period_liquidity=False,
        )
        manufacturer = WidgetManufacturer(mfg_config)

        # Create mock loss that generates a large loss amount
        mock_loss = MagicMock()
        mock_loss.amount = 5_000_000

        mock_loss_generator = MagicMock()
        mock_loss_generator.generate_losses.return_value = ([mock_loss], None)

        mock_insurance = MagicMock()
        mock_insurance.calculate_annual_premium.return_value = 50000.0
        mock_insurance.process_claim.return_value = {
            "insurance_recovery": 0.0,
            "deductible_paid": 5_000_000.0,
        }

        # Tolerance set above initial equity (~50k) so the worker's
        # equity check fires even when the manufacturer's internal
        # insolvency flag short-circuits the accounting step.
        config_dict = {
            "n_years": 5,
            "use_float32": False,
            "ruin_evaluation": [3, 5],
            "insolvency_tolerance": 100_000,
            "letter_of_credit_rate": 0.015,
            "growth_rate": 0.05,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }

        chunk = (0, 1, None)

        result = run_chunk_standalone(
            chunk=chunk,
            loss_generator=mock_loss_generator,
            insurance_program=mock_insurance,
            manufacturer=manufacturer,
            config_dict=config_dict,
        )

        # Verify ruin data is present in results
        assert "ruin_at_year" in result
        ruin_data = result["ruin_at_year"]
        assert len(ruin_data) == 1  # One simulation

        # Equity (~50k) < tolerance (100k) at year 0, so all
        # evaluation years (3 and 5) should be marked True.
        ruin_at_year = ruin_data[0]
        assert ruin_at_year[3] is True, "Eval year 3 should be marked as ruined"
        assert ruin_at_year[5] is True, "Eval year 5 should be marked as ruined"

    def test_ruin_not_marked_when_equity_above_tolerance(self):
        """When equity stays above insolvency_tolerance, ruin should not be marked."""
        from ergodic_insurance.config import ManufacturerConfig
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.monte_carlo_worker import run_chunk_standalone

        # Large company: initial equity ≈ 10M, loss = 100k → stays solvent
        mfg_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.9,
        )
        manufacturer = WidgetManufacturer(mfg_config)

        # Create a small loss that won't cause insolvency
        mock_loss = MagicMock()
        mock_loss.amount = 100_000

        mock_loss_generator = MagicMock()
        mock_loss_generator.generate_losses.return_value = ([mock_loss], None)

        mock_insurance = MagicMock()
        mock_insurance.calculate_annual_premium.return_value = 50000.0
        mock_insurance.process_claim.return_value = {
            "insurance_recovery": 50000.0,
            "deductible_paid": 50000.0,
        }

        config_dict = {
            "n_years": 3,
            "use_float32": False,
            "ruin_evaluation": [2, 3],
            "insolvency_tolerance": 10_000,
            "letter_of_credit_rate": 0.015,
            "growth_rate": 0.05,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }

        chunk = (0, 1, None)

        result = run_chunk_standalone(
            chunk=chunk,
            loss_generator=mock_loss_generator,
            insurance_program=mock_insurance,
            manufacturer=manufacturer,
            config_dict=config_dict,
        )

        # Verify ruin data exists but no ruin was marked
        assert "ruin_at_year" in result
        ruin_data = result["ruin_at_year"]
        assert len(ruin_data) == 1
        ruin_at_year = ruin_data[0]
        assert ruin_at_year[2] is False, "Eval year 2 should not be marked as ruined"
        assert ruin_at_year[3] is False, "Eval year 3 should not be marked as ruined"


# ===========================================================================
# Module 4: statistical_tests.py - Edge cases and error branches
# Line 196: HypothesisTestResult.summary() when reject_null is False
# Line 329: ratio_of_metrics_test invalid alternative
# Line 415: paired_comparison_test invalid alternative
# Line 524: bootstrap_hypothesis_test invalid alternative
# Line 641: multiple_comparison_correction invalid method
# ===========================================================================


class TestHypothesisTestResultSummary:
    """Test HypothesisTestResult.summary() non-rejection branch (line 196)."""

    def test_summary_when_null_not_rejected(self):
        """summary() should include 'No significant difference' when reject_null is False (line 196)."""
        result = HypothesisTestResult(
            test_statistic=0.5,
            p_value=0.3,
            reject_null=False,
            confidence_interval=(-1.0, 2.0),
            null_hypothesis="mean1 = mean2",
            alternative="two-sided",
            alpha=0.05,
            method="bootstrap permutation test",
        )

        summary = result.summary()
        assert "No significant difference" in summary
        assert "p >= 0.05" in summary
        # Also verify basic structure
        assert "Reject Null: No" in summary
        assert "P-value: 0.3000" in summary

    def test_summary_when_null_rejected(self):
        """summary() should include 'Significant difference detected' when reject_null is True."""
        result = HypothesisTestResult(
            test_statistic=3.5,
            p_value=0.001,
            reject_null=True,
            confidence_interval=(1.0, 5.0),
            null_hypothesis="mean1 = mean2",
            alternative="two-sided",
            alpha=0.05,
            method="bootstrap permutation test",
        )

        summary = result.summary()
        assert "Significant difference detected" in summary
        assert "p < 0.05" in summary
        assert "Reject Null: Yes" in summary


class TestStatisticalTestsInvalidAlternative:
    """Test ValueError for invalid alternative parameter (lines 329, 415, 524)."""

    def test_ratio_of_metrics_test_invalid_alternative(self):
        """ratio_of_metrics_test should raise ValueError for unrecognized alternative (line 329)."""
        sample1 = np.array([1.0, 2.0, 3.0])
        sample2 = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="Alternative must be"):
            ratio_of_metrics_test(sample1, sample2, alternative="invalid")

    def test_paired_comparison_test_invalid_alternative(self):
        """paired_comparison_test should raise ValueError for unrecognized alternative (line 415)."""
        differences = np.array([0.1, -0.2, 0.3, 0.15])

        with pytest.raises(ValueError, match="Alternative must be"):
            paired_comparison_test(differences, alternative="invalid")

    def test_bootstrap_hypothesis_test_invalid_alternative(self):
        """bootstrap_hypothesis_test should raise ValueError for unrecognized alternative (line 524)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Alternative must be"):
            bootstrap_hypothesis_test(
                data,
                null_hypothesis=lambda x: x - np.mean(x),
                test_statistic=np.mean,
                alternative="invalid",
            )


class TestMultipleComparisonCorrectionInvalidMethod:
    """Test ValueError for invalid correction method (line 641)."""

    def test_invalid_correction_method_raises_value_error(self):
        """multiple_comparison_correction should raise ValueError for unknown method (line 641)."""
        p_values = [0.01, 0.04, 0.03]

        with pytest.raises(ValueError, match="Method must be"):
            multiple_comparison_correction(p_values, method="invalid_method")

    def test_valid_methods_do_not_raise(self):
        """Verify that valid methods (bonferroni, holm, fdr) do not raise."""
        p_values = [0.01, 0.04, 0.03, 0.20]

        for method in ["bonferroni", "holm", "fdr"]:
            adjusted_p, reject = multiple_comparison_correction(p_values, method=method)
            assert len(adjusted_p) == len(p_values)
            assert len(reject) == len(p_values)


# ===========================================================================
# Module 5: result_aggregator.py - Edge cases
# Lines 36-38: h5py import failure (HAS_H5PY = False)
# Line 221: _fit_distributions normal fit except clause
# Line 234: _fit_distributions lognormal fit except clause
# Line 535: _write_to_hdf5 with array data
# Line 576: HierarchicalAggregator non-dict/non-array leaf
# ===========================================================================


class TestResultExporterHdf5Unavailable:
    """Test behavior when h5py is not available (lines 36-38)."""

    def test_to_hdf5_raises_import_error_when_h5py_unavailable(self, tmp_path):
        """to_hdf5 should raise ImportError when h5py is not available.

        Lines 36-38 set HAS_H5PY=False and h5py=None when the import fails.
        The to_hdf5 method checks HAS_H5PY and raises ImportError accordingly.
        """
        with patch("ergodic_insurance.result_aggregator.HAS_H5PY", False):
            with patch("ergodic_insurance.result_aggregator.h5py", None):
                with pytest.raises(ImportError, match="h5py is required"):
                    ResultExporter.to_hdf5(
                        {"test_metric": 1.0},
                        tmp_path / "output.h5",
                    )

    def test_write_to_hdf5_raises_import_error_when_h5py_unavailable(self):
        """_write_to_hdf5 should raise ImportError when HAS_H5PY is False."""
        with patch("ergodic_insurance.result_aggregator.HAS_H5PY", False):
            with pytest.raises(ImportError, match="h5py is required"):
                ResultExporter._write_to_hdf5(MagicMock(), {"test": 1.0})


class TestResultAggregatorDistributionFitExceptions:
    """Test _fit_distributions exception handling (lines 221, 234)."""

    def test_normal_distribution_fit_failure_caught(self):
        """ValueError from stats.norm.fit should be caught gracefully (line 221).

        When stats.norm.fit raises ValueError, the except clause at line 221
        catches it and skips adding the normal distribution to the results.
        """
        aggregator = ResultAggregator()

        with patch("ergodic_insurance.result_aggregator.stats") as mock_stats:
            mock_stats.norm.fit.side_effect = ValueError("fit failed")
            mock_stats.lognorm.fit.side_effect = ValueError("fit failed")
            mock_stats.skew.return_value = 0.0
            mock_stats.kurtosis.return_value = 0.0

            result = aggregator._fit_distributions(np.array([1.0, 2.0, 3.0]))

        # Both fits failed, so distributions dict should be empty
        assert result == {}

    def test_lognormal_distribution_fit_failure_caught(self):
        """ValueError from stats.lognorm.fit should be caught gracefully (line 234).

        When stats.lognorm.fit raises ValueError, the except clause at line 234
        catches it while the normal distribution fit may still succeed.
        """
        aggregator = ResultAggregator()

        with patch("ergodic_insurance.result_aggregator.stats") as mock_stats:
            # Normal fit succeeds
            mock_stats.norm.fit.return_value = (0.0, 1.0)
            mock_stats.kstest.return_value = (0.1, 0.5)
            # Lognormal fit fails
            mock_stats.lognorm.fit.side_effect = ValueError("lognormal fit failed")
            mock_stats.skew.return_value = 0.0
            mock_stats.kurtosis.return_value = 0.0

            result = aggregator._fit_distributions(np.array([1.0, 2.0, 3.0]))

        # Normal should be present, lognormal should not
        assert "normal" in result
        assert "lognormal" not in result

    def test_type_error_in_fit_also_caught(self):
        """TypeError from fit functions should also be caught (lines 221, 234)."""
        aggregator = ResultAggregator()

        with patch("ergodic_insurance.result_aggregator.stats") as mock_stats:
            mock_stats.norm.fit.side_effect = TypeError("unexpected type")
            mock_stats.lognorm.fit.side_effect = TypeError("unexpected type")
            mock_stats.skew.return_value = 0.0
            mock_stats.kurtosis.return_value = 0.0

            result = aggregator._fit_distributions(np.array([1.0, 2.0, 3.0]))

        assert result == {}


class TestResultExporterWriteToHdf5WithArrays:
    """Test _write_to_hdf5 with array and list data (line 535)."""

    def test_write_to_hdf5_creates_datasets_for_arrays(self):
        """_write_to_hdf5 should call create_dataset for np.ndarray values (line 535)."""
        with patch("ergodic_insurance.result_aggregator.HAS_H5PY", True):
            mock_group = MagicMock()
            data = {
                "numpy_array": np.array([1.0, 2.0, 3.0]),
                "python_list": [4.0, 5.0, 6.0],
                "scalar_value": 42.0,
            }

            ResultExporter._write_to_hdf5(mock_group, data)

            # create_dataset called for both array and list
            assert mock_group.create_dataset.call_count == 2

            # Scalar stored as attribute
            mock_group.attrs.__setitem__.assert_called_once_with("scalar_value", 42.0)

    def test_write_to_hdf5_creates_subgroups_for_dicts(self):
        """_write_to_hdf5 should create subgroups for nested dicts."""
        with patch("ergodic_insurance.result_aggregator.HAS_H5PY", True):
            mock_group = MagicMock()
            mock_subgroup = MagicMock()
            mock_group.create_group.return_value = mock_subgroup

            data = {
                "nested": {
                    "inner_value": 10.0,
                },
            }

            ResultExporter._write_to_hdf5(mock_group, data)

            mock_group.create_group.assert_called_once_with("nested")
            mock_subgroup.attrs.__setitem__.assert_called_once_with("inner_value", 10.0)


class TestHierarchicalAggregatorLeafTypes:
    """Test HierarchicalAggregator with non-dict/non-array leaf data (line 576)."""

    def test_non_dict_non_array_leaf_returned_as_is(self):
        """When past all levels and data is not dict or ndarray, return data as-is (line 576).

        The aggregate_hierarchy method has a fallback at line 576 for data types
        that are neither dict nor np.ndarray when the recursion has reached
        beyond all defined levels.
        """
        aggregator = HierarchicalAggregator(levels=["scenario"])
        data = {"scenario_a": 42.0, "scenario_b": 99.0}

        result = aggregator.aggregate_hierarchy(data)

        # The numeric leaf values should pass through line 576
        assert result["items"]["scenario_a"] == 42.0
        assert result["items"]["scenario_b"] == 99.0

    def test_string_leaf_returned_as_is(self):
        """String leaf data should be returned unchanged via line 576."""
        aggregator = HierarchicalAggregator(levels=["category"])
        data = {"cat_a": "text_value", "cat_b": "other_value"}

        result = aggregator.aggregate_hierarchy(data)

        assert result["items"]["cat_a"] == "text_value"
        assert result["items"]["cat_b"] == "other_value"

    def test_none_leaf_returned_as_is(self):
        """None leaf data should be returned unchanged via line 576."""
        aggregator = HierarchicalAggregator(levels=["level"])
        data = {"item": None}

        result = aggregator.aggregate_hierarchy(data)

        assert result["items"]["item"] is None

    def test_ndarray_leaf_is_aggregated(self):
        """np.ndarray leaf data should be aggregated (not returned raw)."""
        aggregator = HierarchicalAggregator(levels=["scenario"])
        data = {"scenario_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        result = aggregator.aggregate_hierarchy(data)

        # ndarray should be aggregated by ResultAggregator, producing a dict
        aggregated = result["items"]["scenario_a"]
        assert isinstance(aggregated, dict)
        assert "mean" in aggregated

    def test_dict_leaf_returned_as_is(self):
        """Dict leaf data at terminal level should be returned unchanged."""
        aggregator = HierarchicalAggregator(levels=["level"])
        data = {"item": {"key": "value", "count": 5}}

        result = aggregator.aggregate_hierarchy(data)

        assert result["items"]["item"] == {"key": "value", "count": 5}

    def test_empty_levels_with_scalar_data(self):
        """With no levels defined, scalar data should be returned directly (line 576)."""
        aggregator = HierarchicalAggregator(levels=[])

        # Scalar data (not dict, not ndarray) triggers line 576 immediately
        result = aggregator.aggregate_hierarchy(42.0, level=0)  # type: ignore[arg-type]
        assert result == 42.0

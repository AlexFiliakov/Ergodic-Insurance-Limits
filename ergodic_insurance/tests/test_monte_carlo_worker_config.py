"""Tests for custom step configuration in monte_carlo_worker.run_chunk_standalone.

Validates that the 4 configurable step parameters (letter_of_credit_rate,
growth_rate, time_resolution, apply_stochastic) are correctly extracted from
config_dict and passed to manufacturer.step().
"""

from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloConfig
from ergodic_insurance.monte_carlo_worker import run_chunk_standalone


@pytest.fixture
def worker_components():
    """Create minimal components for run_chunk_standalone tests."""
    # Loss generator with small losses
    loss_generator = Mock(spec=ManufacturingLossGenerator)
    loss_generator.generate_losses.return_value = (
        [LossEvent(time=0.5, amount=1_000, loss_type="test")],
        {"total_amount": 1_000},
    )

    # Simple insurance program
    layer = EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.02)
    insurance_program = InsuranceProgram(layers=[layer])

    # Manufacturer with enough assets to survive
    manufacturer_config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        base_operating_margin=0.1,
        tax_rate=0.25,
        retention_ratio=0.8,
    )
    manufacturer = WidgetManufacturer(manufacturer_config)

    return loss_generator, insurance_program, manufacturer


class TestDefaultStepConfig:
    """Test that run_chunk_standalone uses correct defaults when config_dict
    does not contain step parameters."""

    def test_runs_without_step_params_in_config(self, worker_components):
        """Worker should run successfully with only required config keys."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 2,
        }
        chunk = (0, 1, 42)

        result = run_chunk_standalone(
            chunk, loss_generator, insurance_program, manufacturer, config_dict
        )

        assert "final_assets" in result
        assert len(result["final_assets"]) == 1
        assert cast(np.ndarray, result["annual_losses"]).shape == (1, 2)

    def test_default_step_params_passed(self, worker_components):
        """Without explicit config, defaults should be used for step()."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            # Need to patch on the class because deepcopy creates a new instance
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            # step should have been called with default values
            mock_step.assert_called_once_with(
                0.015,  # default letter_of_credit_rate
                0,  # default growth_rate (note: config_dict default is 0, not 0.0)
                "annual",  # default time_resolution
                False,  # default apply_stochastic
            )


class TestCustomLetterOfCreditRate:
    """Test custom letter_of_credit_rate configuration."""

    def test_custom_loc_rate_passed_to_step(self, worker_components):
        """Custom letter_of_credit_rate should be forwarded to step()."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "letter_of_credit_rate": 0.05,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            mock_step.assert_called_once()
            args = mock_step.call_args[0]
            assert args[0] == 0.05  # letter_of_credit_rate

    def test_different_loc_rates_produce_different_step_calls(self, worker_components):
        """Different LoC rates should result in step() being called with the respective values."""
        loss_generator, insurance_program, manufacturer = worker_components

        # Run with low LoC rate
        config_low = {"n_years": 1, "letter_of_credit_rate": 0.001}
        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                (0, 1, 42), loss_generator, insurance_program, manufacturer, config_low
            )
            low_rate_arg = mock_step.call_args[0][0]

        # Run with high LoC rate
        config_high = {"n_years": 1, "letter_of_credit_rate": 0.10}
        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                (0, 1, 42), loss_generator, insurance_program, manufacturer, config_high
            )
            high_rate_arg = mock_step.call_args[0][0]

        assert low_rate_arg == 0.001
        assert high_rate_arg == 0.10
        assert low_rate_arg != high_rate_arg

    def test_zero_loc_rate(self, worker_components):
        """Zero LoC rate should be valid and result in no collateral costs."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "letter_of_credit_rate": 0.0,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            result = run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            args = mock_step.call_args[0]
            assert args[0] == 0.0
            assert cast(np.ndarray, result["final_assets"])[0] > 0


class TestCustomGrowthRate:
    """Test custom growth_rate configuration."""

    def test_custom_growth_rate_passed_to_step(self, worker_components):
        """Custom growth_rate should be forwarded to step()."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "growth_rate": 0.05,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            args = mock_step.call_args[0]
            assert args[1] == 0.05  # growth_rate

    def test_positive_growth_increases_assets(self, worker_components):
        """Positive growth rate should result in higher final assets."""
        loss_generator, insurance_program, manufacturer = worker_components

        # Run with no growth
        config_no_growth = {"n_years": 5, "growth_rate": 0.0}
        result_no_growth = run_chunk_standalone(
            (0, 3, 42), loss_generator, insurance_program, manufacturer, config_no_growth
        )

        # Run with positive growth
        config_growth = {"n_years": 5, "growth_rate": 0.10}
        result_growth = run_chunk_standalone(
            (0, 3, 42), loss_generator, insurance_program, manufacturer, config_growth
        )

        mean_no_growth = np.mean(cast(np.ndarray, result_no_growth["final_assets"]))
        mean_growth = np.mean(cast(np.ndarray, result_growth["final_assets"]))
        assert mean_growth > mean_no_growth, (
            f"Positive growth should increase assets: "
            f"no_growth={mean_no_growth:.0f}, growth={mean_growth:.0f}"
        )

    def test_negative_growth_rate(self, worker_components):
        """Negative growth rate should reduce final assets compared to zero growth."""
        loss_generator, insurance_program, manufacturer = worker_components

        config_zero = {"n_years": 3, "growth_rate": 0.0}
        result_zero = run_chunk_standalone(
            (0, 3, 42), loss_generator, insurance_program, manufacturer, config_zero
        )

        config_neg = {"n_years": 3, "growth_rate": -0.05}
        result_neg = run_chunk_standalone(
            (0, 3, 42), loss_generator, insurance_program, manufacturer, config_neg
        )

        mean_zero = np.mean(cast(np.ndarray, result_zero["final_assets"]))
        mean_neg = np.mean(cast(np.ndarray, result_neg["final_assets"]))
        assert (
            mean_neg < mean_zero
        ), f"Negative growth should reduce assets: zero={mean_zero:.0f}, neg={mean_neg:.0f}"


class TestCustomTimeResolution:
    """Test custom time_resolution configuration."""

    def test_annual_resolution_passed_to_step(self, worker_components):
        """Annual time resolution should be forwarded to step()."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "time_resolution": "annual",
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            args = mock_step.call_args[0]
            assert args[2] == "annual"  # time_resolution

    def test_monthly_resolution_passed_to_step(self, worker_components):
        """Monthly time resolution should be forwarded to step()."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "time_resolution": "monthly",
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            args = mock_step.call_args[0]
            assert args[2] == "monthly"

    def test_annual_vs_monthly_differ(self, worker_components):
        """Annual and monthly resolution should produce different results."""
        loss_generator, insurance_program, manufacturer = worker_components

        config_annual = {"n_years": 2, "time_resolution": "annual"}
        result_annual = run_chunk_standalone(
            (0, 3, 42), loss_generator, insurance_program, manufacturer, config_annual
        )

        config_monthly = {"n_years": 2, "time_resolution": "monthly"}
        result_monthly = run_chunk_standalone(
            (0, 3, 42), loss_generator, insurance_program, manufacturer, config_monthly
        )

        # Monthly and annual should produce different final assets
        # because step() calculates finances differently for each resolution
        assert not np.array_equal(
            cast(np.ndarray, result_annual["final_assets"]),
            cast(np.ndarray, result_monthly["final_assets"]),
        ), "Annual and monthly resolution should produce different results"


class TestCustomApplyStochastic:
    """Test custom apply_stochastic configuration."""

    def test_stochastic_false_passed_to_step(self, worker_components):
        """apply_stochastic=False should be forwarded to step()."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "apply_stochastic": False,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            args = mock_step.call_args[0]
            assert args[3] is False  # apply_stochastic

    def test_stochastic_true_passed_to_step(self, worker_components):
        """apply_stochastic=True should be forwarded to step().

        Note: This will raise RuntimeError inside step() if no stochastic
        process is initialized, so we patch step() to avoid the error.
        """
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "apply_stochastic": True,
        }
        chunk = (0, 1, 42)

        # Create a mock step that returns a valid metrics dict
        mock_metrics = {
            "assets": 10_000_000.0,
            "equity": 10_000_000.0,
            "revenue": 5_000_000.0,
            "operating_income": 500_000.0,
            "net_income": 375_000.0,
            "roe": 0.0375,
            "roa": 0.0375,
            "base_operating_margin": 0.1,
            "is_solvent": True,
            "year": 0,
            "month": 0.0,
        }

        with patch.object(WidgetManufacturer, "step", return_value=mock_metrics) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            args = mock_step.call_args[0]
            assert args[3] is True  # apply_stochastic


class TestAllCustomParameters:
    """Test all 4 custom parameters together."""

    def test_all_params_passed_together(self, worker_components):
        """All 4 custom params should be forwarded to step() simultaneously."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 1,
            "letter_of_credit_rate": 0.025,
            "growth_rate": 0.03,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            mock_step.assert_called_once_with(0.025, 0.03, "annual", False)

    def test_multiple_years_consistent_params(self, worker_components):
        """Custom params should be used consistently across all years."""
        loss_generator, insurance_program, manufacturer = worker_components
        n_years = 3
        config_dict = {
            "n_years": n_years,
            "letter_of_credit_rate": 0.02,
            "growth_rate": 0.04,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }
        chunk = (0, 1, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            # step() called once per year
            assert mock_step.call_count == n_years
            # Every call should use the same custom parameters
            for c in mock_step.call_args_list:
                assert c[0] == (0.02, 0.04, "annual", False)

    def test_multiple_sims_same_params(self, worker_components):
        """Custom params should apply to every simulation in the chunk."""
        loss_generator, insurance_program, manufacturer = worker_components
        n_sims = 3
        config_dict = {
            "n_years": 1,
            "letter_of_credit_rate": 0.03,
            "growth_rate": 0.07,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }
        chunk = (0, n_sims, 42)

        with patch.object(WidgetManufacturer, "step", wraps=manufacturer.step) as mock_step:
            run_chunk_standalone(
                chunk, loss_generator, insurance_program, manufacturer, config_dict
            )

            # step() called once per year per sim
            assert mock_step.call_count == n_sims
            for c in mock_step.call_args_list:
                assert c[0] == (0.03, 0.07, "annual", False)


class TestConfigDictPropagation:
    """Test that MonteCarloConfig fields propagate to config_dict
    in the parallel execution path."""

    def test_simulation_config_has_step_fields(self):
        """MonteCarloConfig should have the 4 step parameter fields."""
        config = MonteCarloConfig()
        assert hasattr(config, "letter_of_credit_rate")
        assert hasattr(config, "growth_rate")
        assert hasattr(config, "time_resolution")
        assert hasattr(config, "apply_stochastic")

    def test_simulation_config_defaults(self):
        """MonteCarloConfig defaults should match worker defaults."""
        config = MonteCarloConfig()
        assert config.letter_of_credit_rate == 0.015
        assert config.growth_rate == 0.0
        assert config.time_resolution == "annual"
        assert config.apply_stochastic is False

    def test_simulation_config_custom_values(self):
        """MonteCarloConfig should accept custom step parameter values."""
        config = MonteCarloConfig(
            letter_of_credit_rate=0.05,
            growth_rate=0.10,
            time_resolution="monthly",
            apply_stochastic=True,
        )
        assert config.letter_of_credit_rate == 0.05
        assert config.growth_rate == 0.10
        assert config.time_resolution == "monthly"
        assert config.apply_stochastic is True


class TestWorkerResultStructure:
    """Test that worker results are correct regardless of step configuration."""

    def test_result_shapes_with_custom_config(self, worker_components):
        """Results should have correct shapes with custom step config."""
        loss_generator, insurance_program, manufacturer = worker_components
        n_sims = 5
        n_years = 3
        config_dict = {
            "n_years": n_years,
            "letter_of_credit_rate": 0.02,
            "growth_rate": 0.05,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }
        chunk = (0, n_sims, 42)

        result = run_chunk_standalone(
            chunk, loss_generator, insurance_program, manufacturer, config_dict
        )

        assert cast(np.ndarray, result["final_assets"]).shape == (n_sims,)
        assert cast(np.ndarray, result["annual_losses"]).shape == (n_sims, n_years)
        assert cast(np.ndarray, result["insurance_recoveries"]).shape == (n_sims, n_years)
        assert cast(np.ndarray, result["retained_losses"]).shape == (n_sims, n_years)

    def test_ruin_evaluation_with_custom_config(self, worker_components):
        """Ruin evaluation should work with custom step parameters."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 5,
            "ruin_evaluation": [3, 5],
            "letter_of_credit_rate": 0.02,
            "growth_rate": 0.0,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }
        chunk = (0, 2, 42)

        result = run_chunk_standalone(
            chunk, loss_generator, insurance_program, manufacturer, config_dict
        )

        assert "ruin_at_year" in result
        assert len(result["ruin_at_year"]) == 2  # one per simulation
        for ruin_data in result["ruin_at_year"]:
            assert 3 in ruin_data
            assert 5 in ruin_data

    def test_float32_with_custom_config(self, worker_components):
        """Float32 mode should work with custom step parameters."""
        loss_generator, insurance_program, manufacturer = worker_components
        config_dict = {
            "n_years": 2,
            "use_float32": True,
            "letter_of_credit_rate": 0.03,
            "growth_rate": 0.02,
        }
        chunk = (0, 2, 42)

        result = run_chunk_standalone(
            chunk, loss_generator, insurance_program, manufacturer, config_dict
        )

        assert cast(np.ndarray, result["final_assets"]).dtype == np.float32
        assert cast(np.ndarray, result["annual_losses"]).dtype == np.float32

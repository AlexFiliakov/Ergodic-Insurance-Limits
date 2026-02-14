"""Coverage tests for ruin_probability.py targeting specific uncovered lines.

Missing lines: 92-116, 280-283, 363-366, 494-495, 540
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.ergodic_types import ClaimResult
from ergodic_insurance.ruin_probability import (
    RuinProbabilityAnalyzer,
    RuinProbabilityConfig,
    RuinProbabilityResults,
)


def make_mock_manufacturer():
    """Create a mock manufacturer for testing."""
    manufacturer = MagicMock()
    manufacturer.total_assets = 10_000_000
    manufacturer.is_ruined = False
    manufacturer.ruin_month = None
    manufacturer.equity = 5_000_000
    manufacturer.debt = 0
    manufacturer.stochastic_process = None
    manufacturer.copy.return_value = manufacturer
    manufacturer.calculate_revenue.return_value = 12_000_000
    manufacturer.step.return_value = {
        "equity": 5_000_000,
        "assets": 10_000_000,
        "operating_income": 1_000_000,
    }
    manufacturer.process_uninsured_claim.return_value = None
    return manufacturer


def make_mock_loss_generator():
    """Create a mock loss generator."""
    loss_gen = MagicMock()
    loss_gen.generate_losses.return_value = ([], {})
    return loss_gen


def make_mock_insurance_program():
    """Create a mock insurance program."""
    program = MagicMock()
    program.process_claim.return_value = ClaimResult(
        total_claim=0.0,
        deductible_paid=0.0,
        insurance_recovery=0.0,
        uncovered_loss=0.0,
        reinstatement_premiums=0.0,
    )
    return program


def make_mock_sim_config():
    """Create a mock simulation config."""
    config = MagicMock()
    config.progress_bar = False
    return config


class TestRuinProbabilityResultsSummary:
    """Tests for RuinProbabilityResults.summary() (lines 92-116)."""

    def test_summary_basic(self):
        """Lines 92-116: Basic summary report generation."""
        # survival_curves must be a homogeneous 2D array (n_horizons x max_years)
        survival = np.array(
            [
                [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90],
                [0.95, 0.93, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84],
                [0.99, 0.97, 0.95, 0.93, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86],
            ]
        )
        results = RuinProbabilityResults(
            time_horizons=np.array([1, 5, 10]),
            ruin_probabilities=np.array([0.01, 0.05, 0.12]),
            confidence_intervals=np.array([[0.005, 0.015], [0.03, 0.07], [0.09, 0.15]]),
            bankruptcy_causes={
                "asset_threshold": np.array([0.3, 0.2, 0.1]),
                "equity_threshold": np.array([0.7, 0.6, 0.5]),
                "consecutive_negative": np.array([0.0, 0.1, 0.2]),
                "debt_service": np.array([0.0, 0.1, 0.2]),
            },
            survival_curves=survival,
            execution_time=5.5,
            n_simulations=10000,
            convergence_achieved=True,
        )
        summary = results.summary()
        assert "Ruin Probability Analysis Results" in summary
        assert "10,000" in summary
        assert "5.50 seconds" in summary

    def test_summary_with_mid_year_ruin(self):
        """Lines 91-116: Summary includes mid-year ruin statistics."""
        results = RuinProbabilityResults(
            time_horizons=np.array([10]),
            ruin_probabilities=np.array([0.15]),
            confidence_intervals=np.array([[0.10, 0.20]]),
            bankruptcy_causes={
                "asset_threshold": np.array([0.1]),
                "equity_threshold": np.array([0.5]),
                "consecutive_negative": np.array([0.2]),
                "debt_service": np.array([0.2]),
            },
            survival_curves=np.zeros((1, 10)),
            execution_time=3.0,
            n_simulations=5000,
            convergence_achieved=True,
            mid_year_ruin_count=250,
            ruin_month_distribution={0: 50, 3: 80, 6: 70, 9: 50},
        )
        summary = results.summary()
        assert "Mid-Year Ruin Analysis" in summary
        assert "250" in summary
        assert "Jan" in summary or "Apr" in summary

    def test_summary_without_10_year_horizon(self):
        """Summary handles missing 10-year horizon gracefully."""
        results = RuinProbabilityResults(
            time_horizons=np.array([5, 20]),
            ruin_probabilities=np.array([0.05, 0.20]),
            confidence_intervals=np.array([[0.03, 0.07], [0.15, 0.25]]),
            bankruptcy_causes={
                "asset_threshold": np.array([0.1, 0.1]),
                "equity_threshold": np.array([0.5, 0.5]),
                "consecutive_negative": np.array([0.2, 0.2]),
                "debt_service": np.array([0.2, 0.2]),
            },
            survival_curves=np.zeros((2, 20)),
            execution_time=2.0,
            n_simulations=1000,
            convergence_achieved=False,
        )
        summary = results.summary()
        assert "Ruin Probability Analysis Results" in summary


class TestRunRuinSimulationsSequentialMidYear:
    """Tests for mid-year ruin tracking in sequential sims (lines 280-283)."""

    def test_mid_year_ruin_tracking_in_sequential(self):
        """Lines 280-283: Track mid-year ruin events in sequential simulations."""
        manufacturer = make_mock_manufacturer()
        loss_gen = make_mock_loss_generator()
        insurance = make_mock_insurance_program()
        sim_config = make_mock_sim_config()

        analyzer = RuinProbabilityAnalyzer(manufacturer, loss_gen, insurance, sim_config)

        # Mock _run_single_ruin_simulation to return mid-year ruin
        def mock_single_sim(sim_id, max_horizon, config):
            return {
                "bankruptcy_year": 3,
                "causes": {
                    "asset_threshold": np.zeros(max_horizon, dtype=bool),
                    "equity_threshold": np.zeros(max_horizon, dtype=bool),
                    "consecutive_negative": np.zeros(max_horizon, dtype=bool),
                    "debt_service": np.zeros(max_horizon, dtype=bool),
                },
                "is_mid_year_ruin": True,
                "ruin_month": 6,
            }

        with patch.object(analyzer, "_run_single_ruin_simulation", side_effect=mock_single_sim):
            ruin_config = RuinProbabilityConfig(n_simulations=5, time_horizons=[5], parallel=False)
            result = analyzer._run_ruin_simulations_sequential(ruin_config)

        assert result["mid_year_ruin_count"] == 5
        assert result["ruin_month_distribution"][6] == 5


class TestRunRuinChunkMidYear:
    """Tests for mid-year ruin tracking in chunk sims (lines 363-366)."""

    def test_mid_year_ruin_tracking_in_chunk(self):
        """Lines 363-366: Track mid-year ruin events in parallel chunks."""
        manufacturer = make_mock_manufacturer()
        loss_gen = make_mock_loss_generator()
        insurance = make_mock_insurance_program()
        sim_config = make_mock_sim_config()

        analyzer = RuinProbabilityAnalyzer(manufacturer, loss_gen, insurance, sim_config)

        def mock_single_sim(sim_id, max_horizon, config):
            return {
                "bankruptcy_year": 2,
                "causes": {
                    "asset_threshold": np.zeros(max_horizon, dtype=bool),
                    "equity_threshold": np.zeros(max_horizon, dtype=bool),
                    "consecutive_negative": np.zeros(max_horizon, dtype=bool),
                    "debt_service": np.zeros(max_horizon, dtype=bool),
                },
                "is_mid_year_ruin": sim_id % 2 == 0,
                "ruin_month": 3 if sim_id % 2 == 0 else None,
            }

        with patch.object(analyzer, "_run_single_ruin_simulation", side_effect=mock_single_sim):
            ruin_config = RuinProbabilityConfig(n_simulations=4, time_horizons=[5], seed=42)
            chunk = (0, 4, 5, ruin_config, 42)
            result = analyzer._run_ruin_chunk(chunk)

        assert result["mid_year_ruin_count"] == 2
        assert 3 in result["ruin_month_distribution"]


class TestRunSingleRuinSimulationMidYear:
    """Tests for mid-year ruin detection in single sim (lines 494-495)."""

    def test_mid_year_ruin_detected(self):
        """Lines 494-495: Detect mid-year ruin via manufacturer.ruin_month."""
        manufacturer = make_mock_manufacturer()
        loss_gen = make_mock_loss_generator()
        insurance = make_mock_insurance_program()
        sim_config = make_mock_sim_config()

        # After step, manufacturer becomes ruined with ruin_month
        def side_effect_step(growth_rate=0.0):
            manufacturer.is_ruined = True
            manufacturer.ruin_month = 7
            manufacturer.equity = -100
            return {"equity": -100, "assets": 500_000, "operating_income": 0}

        manufacturer.step.side_effect = side_effect_step

        # Patch create_fresh to return the same manufacturer so the
        # side_effect_step closure can mutate it during the simulation.
        with patch("ergodic_insurance.ruin_probability.WidgetManufacturer") as MockWM:
            MockWM.create_fresh.return_value = manufacturer
            analyzer = RuinProbabilityAnalyzer(manufacturer, loss_gen, insurance, sim_config)
            ruin_config = RuinProbabilityConfig(time_horizons=[5], early_stopping=True)
            result = analyzer._run_single_ruin_simulation(0, 5, ruin_config)

        assert result["is_mid_year_ruin"] is True
        assert result["ruin_month"] == 7


class TestCombineRuinResults:
    """Tests for _combine_ruin_results (line 540)."""

    def test_combine_multiple_chunks(self):
        """Line 540: Combine ruin results from multiple chunks."""
        manufacturer = make_mock_manufacturer()
        loss_gen = make_mock_loss_generator()
        insurance = make_mock_insurance_program()
        sim_config = make_mock_sim_config()

        analyzer = RuinProbabilityAnalyzer(manufacturer, loss_gen, insurance, sim_config)

        chunk1 = {
            "bankruptcy_years": np.array([3, 6, 11], dtype=np.int32),
            "bankruptcy_causes": {
                "asset_threshold": np.zeros((3, 10), dtype=bool),
                "equity_threshold": np.zeros((3, 10), dtype=bool),
                "consecutive_negative": np.zeros((3, 10), dtype=bool),
                "debt_service": np.zeros((3, 10), dtype=bool),
            },
            "mid_year_ruin_count": 1,
            "ruin_month_distribution": {3: 1},
        }
        chunk2 = {
            "bankruptcy_years": np.array([2, 11], dtype=np.int32),
            "bankruptcy_causes": {
                "asset_threshold": np.zeros((2, 10), dtype=bool),
                "equity_threshold": np.zeros((2, 10), dtype=bool),
                "consecutive_negative": np.zeros((2, 10), dtype=bool),
                "debt_service": np.zeros((2, 10), dtype=bool),
            },
            "mid_year_ruin_count": 1,
            "ruin_month_distribution": {3: 1, 6: 1},
        }

        combined = analyzer._combine_ruin_results([chunk1, chunk2])
        assert len(combined["bankruptcy_years"]) == 5
        assert combined["mid_year_ruin_count"] == 2
        assert combined["ruin_month_distribution"][3] == 2
        assert combined["ruin_month_distribution"][6] == 1


class TestCheckRuinConvergence:
    """Tests for _check_ruin_convergence."""

    def test_insufficient_data_not_converged(self):
        """Too few data points for convergence check."""
        manufacturer = make_mock_manufacturer()
        analyzer = RuinProbabilityAnalyzer(
            manufacturer,
            make_mock_loss_generator(),
            make_mock_insurance_program(),
            make_mock_sim_config(),
        )
        short_data = np.array([3, 5, 7, 11], dtype=np.int32)
        result = analyzer._check_ruin_convergence(short_data)
        assert result is False

    def test_converged_uniform_data(self):
        """Uniform data converges."""
        manufacturer = make_mock_manufacturer()
        analyzer = RuinProbabilityAnalyzer(
            manufacturer,
            make_mock_loss_generator(),
            make_mock_insurance_program(),
            make_mock_sim_config(),
        )
        # All simulations result in same outcome
        uniform_data = np.full(1000, 11, dtype=np.int32)
        result = analyzer._check_ruin_convergence(uniform_data)
        assert result is True

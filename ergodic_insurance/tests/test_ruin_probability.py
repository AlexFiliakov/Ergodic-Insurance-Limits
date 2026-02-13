"""Comprehensive test suite for ruin probability analysis.

Tests all aspects of the RuinProbabilityAnalyzer including sequential and parallel
simulation, bankruptcy conditions, confidence intervals, and convergence checking.
"""

from concurrent.futures import Future
import time
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

from ergodic_insurance.ergodic_types import ClaimResult
from ergodic_insurance.ruin_probability import (
    RuinProbabilityAnalyzer,
    RuinProbabilityConfig,
    RuinProbabilityResults,
)


class TestRuinProbabilityConfig:
    """Test RuinProbabilityConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RuinProbabilityConfig()
        assert config.time_horizons == [1, 5, 10]
        assert config.n_simulations == 10000
        assert config.min_assets_threshold == 1_000_000
        assert config.min_equity_threshold == 0.0
        assert config.debt_service_coverage_ratio == 1.25
        assert config.consecutive_negative_periods == 3
        assert config.early_stopping is True
        assert config.parallel is True
        assert config.n_workers is None
        assert config.seed is None
        assert config.n_bootstrap == 1000
        assert config.bootstrap_confidence_level == 0.95

    def test_custom_config(self):
        """Test custom configuration."""
        config = RuinProbabilityConfig(
            time_horizons=[2, 4, 6],
            n_simulations=5000,
            min_assets_threshold=500_000,
            parallel=False,
            seed=42,
        )
        assert config.time_horizons == [2, 4, 6]
        assert config.n_simulations == 5000
        assert config.min_assets_threshold == 500_000
        assert config.parallel is False
        assert config.seed == 42


class TestRuinProbabilityResults:
    """Test RuinProbabilityResults dataclass."""

    def test_results_creation(self):
        """Test creating results object."""
        results = RuinProbabilityResults(
            time_horizons=np.array([1, 5, 10]),
            ruin_probabilities=np.array([0.01, 0.05, 0.10]),
            confidence_intervals=np.array([[0.008, 0.012], [0.045, 0.055], [0.09, 0.11]]),
            bankruptcy_causes={
                "asset_threshold": np.array([0.005, 0.025, 0.05]),
                "equity_threshold": np.array([0.003, 0.015, 0.03]),
                "consecutive_negative": np.array([0.001, 0.005, 0.01]),
                "debt_service": np.array([0.001, 0.005, 0.01]),
            },
            survival_curves=np.array(
                [
                    np.array([0.99, 0.98]),
                    np.array([0.95, 0.94, 0.93, 0.92, 0.91]),
                    np.array([0.9] * 10),
                ],
                dtype=object,
            ),
            execution_time=10.5,
            n_simulations=10000,
            convergence_achieved=True,
        )
        assert results.n_simulations == 10000
        assert results.execution_time == 10.5
        assert results.convergence_achieved is True
        assert len(results.time_horizons) == 3

    def test_summary_report(self):
        """Test summary report generation."""
        results = RuinProbabilityResults(
            time_horizons=np.array([1, 5, 10]),
            ruin_probabilities=np.array([0.01, 0.05, 0.10]),
            confidence_intervals=np.array([[0.008, 0.012], [0.045, 0.055], [0.09, 0.11]]),
            bankruptcy_causes={
                "asset_threshold": np.array([0.005, 0.025, 0.05]),
                "equity_threshold": np.array([0.003, 0.015, 0.03]),
                "consecutive_negative": np.array([0.001, 0.005, 0.01]),
                "debt_service": np.array([0.001, 0.005, 0.01]),
            },
            survival_curves=np.zeros((3, 10)),
            execution_time=10.5,
            n_simulations=10000,
            convergence_achieved=True,
        )

        summary = results.summary()
        assert "Ruin Probability Analysis Results" in summary
        assert "10,000" in summary
        assert "10.50 seconds" in summary
        assert "Convergence achieved: True" in summary
        assert "1 years:  1.00%" in summary
        assert "5 years:  5.00%" in summary
        assert "10 years: 10.00%" in summary
        assert "Bankruptcy Causes" in summary

    def test_summary_no_10_year_horizon(self):
        """Test summary when 10-year horizon is not present."""
        results = RuinProbabilityResults(
            time_horizons=np.array([1, 5]),
            ruin_probabilities=np.array([0.01, 0.05]),
            confidence_intervals=np.array([[0.008, 0.012], [0.045, 0.055]]),
            bankruptcy_causes={
                "asset_threshold": np.array([0.005, 0.025]),
            },
            survival_curves=np.zeros((2, 5)),
            execution_time=5.0,
            n_simulations=5000,
            convergence_achieved=False,
        )

        summary = results.summary()
        assert "5,000" in summary
        # Check that 10 years is not in the time horizons section
        lines = summary.split("\n")
        for line in lines:
            if "years:" in line and "10 years" in line:
                assert False, "10 years should not be in time horizons"


class TestRuinProbabilityAnalyzer:
    """Test RuinProbabilityAnalyzer class."""

    @pytest.fixture
    def mock_manufacturer(self):
        """Create mock manufacturer."""
        manufacturer = MagicMock()
        manufacturer.total_assets = 10_000_000
        manufacturer.debt = 0  # Set default debt value
        manufacturer.is_ruined = False
        manufacturer.calculate_revenue.return_value = 5_000_000
        manufacturer.step.return_value = {"equity": 1_000_000, "operating_income": 500_000}

        # Create a proper copy method that preserves attributes
        def copy_manufacturer():
            copy_mfg = MagicMock()
            copy_mfg.total_assets = manufacturer.total_assets
            copy_mfg.debt = 0  # Important: set debt on copy
            copy_mfg.is_ruined = False
            copy_mfg.calculate_revenue = manufacturer.calculate_revenue
            copy_mfg.step = manufacturer.step
            copy_mfg.process_insurance_claim = MagicMock()
            copy_mfg.copy = copy_manufacturer
            return copy_mfg

        manufacturer.copy = copy_manufacturer
        manufacturer.process_insurance_claim = MagicMock()
        return manufacturer

    @pytest.fixture
    def mock_loss_generator(self):
        """Create mock loss generator."""
        generator = MagicMock()
        event = MagicMock()
        event.amount = 100_000
        generator.generate_losses.return_value = ([event], {"total_loss": 100_000})
        return generator

    @pytest.fixture
    def mock_insurance_program(self):
        """Create mock insurance program."""
        program = MagicMock()
        program.process_claim.return_value = ClaimResult(
            total_claim=100_000,
            deductible_paid=50_000,
            insurance_recovery=50_000,
            uncovered_loss=0.0,
            reinstatement_premiums=0.0,
        )
        return program

    @pytest.fixture
    def mock_config(self):
        """Create mock simulation config."""
        config = MagicMock()
        config.progress_bar = False
        return config

    @pytest.fixture
    def analyzer(self, mock_manufacturer, mock_loss_generator, mock_insurance_program, mock_config):
        """Create analyzer instance."""
        return RuinProbabilityAnalyzer(
            mock_manufacturer, mock_loss_generator, mock_insurance_program, mock_config
        )

    def test_initialization(self, analyzer, mock_manufacturer):
        """Test analyzer initialization."""
        assert analyzer.manufacturer == mock_manufacturer
        assert analyzer.loss_generator is not None
        assert analyzer.insurance_program is not None
        assert analyzer.config is not None

    def test_analyze_ruin_probability_sequential(self, analyzer):
        """Test sequential ruin probability analysis."""
        config = RuinProbabilityConfig(
            time_horizons=[1, 2],
            n_simulations=10,
            parallel=False,
        )

        results = analyzer.analyze_ruin_probability(config)

        assert isinstance(results, RuinProbabilityResults)
        assert len(results.time_horizons) == 2
        assert len(results.ruin_probabilities) == 2
        assert results.n_simulations == 10
        assert results.execution_time > 0

    def test_analyze_ruin_probability_parallel(self, analyzer):
        """Test parallel ruin probability analysis."""
        config = RuinProbabilityConfig(
            time_horizons=[1, 2],
            n_simulations=2000,  # Enough to trigger parallel
            parallel=True,
            n_workers=2,
        )

        with patch.object(analyzer, "_run_ruin_simulations_parallel") as mock_parallel:
            mock_parallel.return_value = {
                "bankruptcy_years": np.array([3, 2, 4, 1] * 500),
                "bankruptcy_causes": {
                    "asset_threshold": np.zeros((2000, 2), dtype=bool),
                    "equity_threshold": np.zeros((2000, 2), dtype=bool),
                    "consecutive_negative": np.zeros((2000, 2), dtype=bool),
                    "debt_service": np.zeros((2000, 2), dtype=bool),
                },
            }

            results = analyzer.analyze_ruin_probability(config)
            mock_parallel.assert_called_once()
            assert results.n_simulations == 2000

    def test_run_single_ruin_simulation(self, analyzer):
        """Test single ruin simulation."""
        config = RuinProbabilityConfig(early_stopping=True)

        result = analyzer._run_single_ruin_simulation(sim_id=0, max_horizon=5, config=config)

        assert "bankruptcy_year" in result
        assert "causes" in result
        assert result["bankruptcy_year"] <= 6  # max_horizon + 1
        assert all(cause in result["causes"] for cause in ["asset_threshold", "equity_threshold"])

    def test_check_bankruptcy_conditions_asset_threshold(self, analyzer):
        """Test bankruptcy due to asset threshold."""
        manufacturer = MagicMock()
        manufacturer.total_assets = 500_000
        manufacturer.debt = 0
        manufacturer.is_ruined = False

        config = RuinProbabilityConfig(min_assets_threshold=1_000_000)
        causes = {
            "asset_threshold": np.zeros(10, dtype=bool),
            "equity_threshold": np.zeros(10, dtype=bool),
            "consecutive_negative": np.zeros(10, dtype=bool),
            "debt_service": np.zeros(10, dtype=bool),
        }

        is_bankrupt, _ = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": 100_000}, 0, config, causes, 0
        )

        assert is_bankrupt is True
        assert bool(causes["asset_threshold"][0])

    def test_check_bankruptcy_conditions_equity_threshold(self, analyzer):
        """Test bankruptcy due to equity threshold."""
        manufacturer = MagicMock()
        manufacturer.total_assets = 2_000_000
        manufacturer.debt = 0
        manufacturer.is_ruined = False

        config = RuinProbabilityConfig(min_equity_threshold=100_000)
        causes = {
            "asset_threshold": np.zeros(10, dtype=bool),
            "equity_threshold": np.zeros(10, dtype=bool),
            "consecutive_negative": np.zeros(10, dtype=bool),
            "debt_service": np.zeros(10, dtype=bool),
        }

        is_bankrupt, _ = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": 50_000}, 0, config, causes, 0
        )

        assert is_bankrupt is True
        assert bool(causes["equity_threshold"][0])

    def test_check_bankruptcy_conditions_consecutive_negative(self, analyzer):
        """Test bankruptcy due to consecutive negative equity."""
        manufacturer = MagicMock()
        manufacturer.total_assets = 2_000_000
        manufacturer.debt = 0
        manufacturer.is_ruined = False

        config = RuinProbabilityConfig(
            consecutive_negative_periods=3,
            min_equity_threshold=-100_000,  # Set threshold below test value to isolate consecutive negative test
        )
        causes = {
            "asset_threshold": np.zeros(10, dtype=bool),
            "equity_threshold": np.zeros(10, dtype=bool),
            "consecutive_negative": np.zeros(10, dtype=bool),
            "debt_service": np.zeros(10, dtype=bool),
        }

        # First two negative periods - not bankrupt
        is_bankrupt, count = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": -50_000}, 0, config, causes, 0
        )
        assert is_bankrupt is False
        assert count == 1

        is_bankrupt, count = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": -50_000}, 1, config, causes, count
        )
        assert is_bankrupt is False
        assert count == 2

        # Third negative period - bankrupt
        is_bankrupt, count = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": -50_000}, 2, config, causes, count
        )
        assert is_bankrupt is True
        assert bool(causes["consecutive_negative"][2])

    def test_check_bankruptcy_conditions_debt_service(self, analyzer):
        """Test bankruptcy due to debt service coverage."""
        manufacturer = MagicMock()
        manufacturer.total_assets = 2_000_000
        manufacturer.debt = 1_000_000
        manufacturer.is_ruined = False

        config = RuinProbabilityConfig(debt_service_coverage_ratio=1.25)
        causes = {
            "asset_threshold": np.zeros(10, dtype=bool),
            "equity_threshold": np.zeros(10, dtype=bool),
            "consecutive_negative": np.zeros(10, dtype=bool),
            "debt_service": np.zeros(10, dtype=bool),
        }

        # Low operating income relative to debt service
        is_bankrupt, _ = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": 500_000, "operating_income": 50_000}, 0, config, causes, 0
        )

        assert is_bankrupt is True
        assert bool(causes["debt_service"][0])  # numpy bool comparison

    def test_check_bankruptcy_conditions_reset_consecutive(self, analyzer):
        """Test resetting consecutive negative count."""
        manufacturer = MagicMock()
        manufacturer.total_assets = 2_000_000
        manufacturer.debt = 0
        manufacturer.is_ruined = False

        config = RuinProbabilityConfig()
        causes = {
            "asset_threshold": np.zeros(10, dtype=bool),
            "equity_threshold": np.zeros(10, dtype=bool),
            "consecutive_negative": np.zeros(10, dtype=bool),
            "debt_service": np.zeros(10, dtype=bool),
        }

        # Negative equity
        _, count = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": -50_000}, 0, config, causes, 0
        )
        assert count == 1

        # Positive equity - resets count
        _, count = analyzer._check_bankruptcy_conditions(
            manufacturer, {"equity": 50_000}, 1, config, causes, count
        )
        assert count == 0

    def test_process_simulation_year(self, analyzer, mock_manufacturer):
        """Test processing a single simulation year."""
        metrics = analyzer._process_simulation_year(mock_manufacturer, 0)

        assert isinstance(metrics, dict)
        assert "equity" in metrics
        mock_manufacturer.calculate_revenue.assert_called_once()
        mock_manufacturer.step.assert_called_once()

    def test_process_simulation_year_per_occurrence(self, mock_manufacturer, mock_config):
        """Test that each loss event is processed individually through process_claim.

        Regression test for issue #1136: previously all annual losses were summed
        into a single claim, breaking per-occurrence deductible and layer semantics.
        """
        # Create 3 separate loss events
        event1 = MagicMock()
        event1.amount = 200_000
        event2 = MagicMock()
        event2.amount = 300_000
        event3 = MagicMock()
        event3.amount = 500_000

        loss_generator = MagicMock()
        loss_generator.generate_losses.return_value = (
            [event1, event2, event3],
            {"total_loss": 1_000_000},
        )

        # Insurance program with per-occurrence deductible of 100K
        # Each event should be processed separately:
        #   event1: 200K claim -> 100K deductible, 100K recovery
        #   event2: 300K claim -> 100K deductible, 200K recovery
        #   event3: 500K claim -> 100K deductible, 400K recovery
        # Total recovery = 700K, retained = 300K
        insurance_program = MagicMock()
        insurance_program.process_claim.side_effect = [
            ClaimResult(
                total_claim=200_000,
                deductible_paid=100_000,
                insurance_recovery=100_000,
                uncovered_loss=0.0,
                reinstatement_premiums=0.0,
            ),
            ClaimResult(
                total_claim=300_000,
                deductible_paid=100_000,
                insurance_recovery=200_000,
                uncovered_loss=0.0,
                reinstatement_premiums=0.0,
            ),
            ClaimResult(
                total_claim=500_000,
                deductible_paid=100_000,
                insurance_recovery=400_000,
                uncovered_loss=0.0,
                reinstatement_premiums=0.0,
            ),
        ]

        analyzer = RuinProbabilityAnalyzer(
            mock_manufacturer, loss_generator, insurance_program, mock_config
        )

        analyzer._process_simulation_year(mock_manufacturer, 0)

        # Verify process_claim was called 3 times (once per event), NOT once with the sum
        assert insurance_program.process_claim.call_count == 3
        insurance_program.process_claim.assert_any_call(200_000)
        insurance_program.process_claim.assert_any_call(300_000)
        insurance_program.process_claim.assert_any_call(500_000)

        # Retained = 1_000_000 total - 700_000 recovery = 300_000
        mock_manufacturer.process_uninsured_claim.assert_called_once_with(
            claim_amount=300_000,
            immediate_payment=False,
        )

    def test_process_simulation_year_no_events(self, mock_config, mock_manufacturer):
        """Test that no claims are processed when there are no loss events."""
        loss_generator = MagicMock()
        loss_generator.generate_losses.return_value = ([], {"total_loss": 0})

        insurance_program = MagicMock()

        analyzer = RuinProbabilityAnalyzer(
            mock_manufacturer, loss_generator, insurance_program, mock_config
        )

        analyzer._process_simulation_year(mock_manufacturer, 0)

        # No events means no claims processed and no uninsured loss
        insurance_program.process_claim.assert_not_called()
        mock_manufacturer.process_uninsured_claim.assert_not_called()

    def test_pad_survival_curves(self, analyzer):
        """Test padding survival curves to uniform length."""
        curves = [
            np.array([1.0, 0.9]),
            np.array([1.0, 0.95, 0.9, 0.85]),
            np.array([1.0]),
        ]

        padded = analyzer._pad_survival_curves(curves)

        assert padded.shape == (3, 4)  # 3 curves, max length 4
        assert np.array_equal(padded[0], [1.0, 0.9, 0.0, 0.0])
        assert np.array_equal(padded[1], [1.0, 0.95, 0.9, 0.85])
        assert np.array_equal(padded[2], [1.0, 0.0, 0.0, 0.0])

    def test_pad_survival_curves_empty(self, analyzer):
        """Test padding empty survival curves."""
        padded = analyzer._pad_survival_curves([])
        assert padded.shape == (0, 0)

    def test_analyze_horizons(self, analyzer):
        """Test analyzing results for different time horizons."""
        simulation_results = {
            "bankruptcy_years": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "bankruptcy_causes": {
                "asset_threshold": np.ones((10, 10), dtype=bool),
                "equity_threshold": np.zeros((10, 10), dtype=bool),
                "consecutive_negative": np.zeros((10, 10), dtype=bool),
                "debt_service": np.zeros((10, 10), dtype=bool),
            },
        }
        config = RuinProbabilityConfig(
            time_horizons=[2, 5, 10],
            n_simulations=10,
        )

        result = analyzer._analyze_horizons(simulation_results, config)

        assert len(result["ruin_probs"]) == 3
        assert result["ruin_probs"][0] == 0.2  # 2 out of 10 bankrupt by year 2
        assert result["ruin_probs"][1] == 0.5  # 5 out of 10 bankrupt by year 5
        assert result["ruin_probs"][2] == 1.0  # All bankrupt by year 10
        assert len(result["survival_curves"]) == 3

    def test_create_simulation_chunks(self, analyzer):
        """Test creating chunks for parallel processing."""
        config = RuinProbabilityConfig(
            n_simulations=1000,
            n_workers=4,
            seed=42,
        )

        chunks = analyzer._create_simulation_chunks(config, max_horizon=10)

        assert len(chunks) >= 4  # At least as many chunks as workers
        # Check first chunk
        assert chunks[0][0] == 0  # start_idx
        assert chunks[0][2] == 10  # max_horizon
        assert chunks[0][4] == 42  # seed

        # Check chunks cover all simulations
        total_sims = sum(chunk[1] - chunk[0] for chunk in chunks)
        assert total_sims == 1000

    def test_create_simulation_chunks_no_seed(self, analyzer):
        """Test creating chunks without seed."""
        config = RuinProbabilityConfig(
            n_simulations=500,
            n_workers=2,
            seed=None,
        )

        chunks = analyzer._create_simulation_chunks(config, max_horizon=5)

        assert all(chunk[4] is None for chunk in chunks)

    def test_run_ruin_chunk(self, analyzer):
        """Test running a single chunk of simulations."""
        config = RuinProbabilityConfig()
        chunk = (0, 5, 3, config, 42)  # 5 simulations, 3 year horizon

        result = analyzer._run_ruin_chunk(chunk)

        assert "bankruptcy_years" in result
        assert len(result["bankruptcy_years"]) == 5
        assert all(year <= 4 for year in result["bankruptcy_years"])  # max_horizon + 1
        assert "bankruptcy_causes" in result

    def test_combine_ruin_results(self, analyzer):
        """Test combining results from multiple chunks."""
        chunk1 = {
            "bankruptcy_years": np.array([1, 2, 3]),
            "bankruptcy_causes": {
                "asset_threshold": np.array([[True, False], [False, True], [True, True]]),
                "equity_threshold": np.array([[False, False], [True, False], [False, True]]),
                "consecutive_negative": np.zeros((3, 2), dtype=bool),
                "debt_service": np.zeros((3, 2), dtype=bool),
            },
        }
        chunk2 = {
            "bankruptcy_years": np.array([4, 5]),
            "bankruptcy_causes": {
                "asset_threshold": np.array([[False, False], [True, False]]),
                "equity_threshold": np.array([[True, True], [False, False]]),
                "consecutive_negative": np.zeros((2, 2), dtype=bool),
                "debt_service": np.zeros((2, 2), dtype=bool),
            },
        }

        combined = analyzer._combine_ruin_results([chunk1, chunk2])

        assert len(combined["bankruptcy_years"]) == 5
        assert list(combined["bankruptcy_years"]) == [1, 2, 3, 4, 5]
        assert combined["bankruptcy_causes"]["asset_threshold"].shape == (5, 2)
        assert combined["bankruptcy_causes"]["equity_threshold"].shape == (5, 2)

    def test_calculate_bootstrap_ci(self, analyzer):
        """Test bootstrap confidence interval calculation."""
        np.random.seed(42)
        bankruptcy_years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100)
        time_horizons = [5, 10]

        ci = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, time_horizons, n_bootstrap=100, confidence_level=0.95
        )

        assert ci.shape == (2, 2)
        # Check that CI contains the true value
        true_prob_5 = np.mean(bankruptcy_years <= 5)
        assert ci[0, 0] <= true_prob_5 <= ci[0, 1]

        true_prob_10 = np.mean(bankruptcy_years <= 10)
        assert ci[1, 0] <= true_prob_10 <= ci[1, 1]

    def test_check_ruin_convergence_converged(self, analyzer):
        """Test convergence checking - converged case."""
        # Create data that should converge - consistent ruin probability
        np.random.seed(42)
        bankruptcy_years = np.random.choice([1, 2, 11, 12], size=400, p=[0.05, 0.05, 0.45, 0.45])

        converged = analyzer._check_ruin_convergence(bankruptcy_years, n_chains=4)

        assert isinstance(converged, bool)
        # With consistent probabilities across chains, should converge

    def test_check_ruin_convergence_not_converged(self, analyzer):
        """Test convergence checking - not converged case."""
        # Create data that should not converge - different patterns
        bankruptcy_years = np.concatenate(
            [
                np.ones(100) * 1,  # Chain 1: all bankrupt early
                np.ones(100) * 11,  # Chain 2: none bankrupt
                np.ones(100) * 5,  # Chain 3: mixed
                np.ones(100) * 15,  # Chain 4: none bankrupt
            ]
        )

        converged = analyzer._check_ruin_convergence(bankruptcy_years, n_chains=4)

        assert converged is False

    def test_check_ruin_convergence_insufficient_data(self, analyzer):
        """Test convergence checking with insufficient data."""
        bankruptcy_years = np.array([1, 2, 3, 4, 5])

        converged = analyzer._check_ruin_convergence(bankruptcy_years, n_chains=4)

        assert converged is False

    def test_check_ruin_convergence_zero_variance(self, analyzer):
        """Test convergence checking with zero within-chain variance."""
        # All chains have identical values
        bankruptcy_years = np.ones(400) * 5

        converged = analyzer._check_ruin_convergence(bankruptcy_years, n_chains=4)

        assert converged is True

    def test_run_ruin_simulations_parallel_with_progress(self, analyzer, mock_config):
        """Test parallel execution with progress bar."""
        mock_config.progress_bar = True
        config = RuinProbabilityConfig(
            n_simulations=100,
            parallel=True,
            n_workers=2,
        )

        with patch("ergodic_insurance.ruin_probability.ProcessPoolExecutor") as mock_executor:
            with patch("ergodic_insurance.ruin_probability.tqdm") as mock_tqdm:
                # Setup mock executor
                mock_future = Mock(spec=Future)
                mock_future.result.return_value = {
                    "bankruptcy_years": np.array([1, 2]),
                    "bankruptcy_causes": {
                        "asset_threshold": np.zeros((2, 5), dtype=bool),
                        "equity_threshold": np.zeros((2, 5), dtype=bool),
                        "consecutive_negative": np.zeros((2, 5), dtype=bool),
                        "debt_service": np.zeros((2, 5), dtype=bool),
                    },
                }

                mock_executor_instance = mock_executor.return_value.__enter__.return_value
                mock_executor_instance.submit.return_value = mock_future

                # Mock as_completed to return our futures
                with patch("ergodic_insurance.ruin_probability.as_completed") as mock_as_completed:
                    mock_as_completed.return_value = [mock_future]

                    result = analyzer._run_ruin_simulations_parallel(config)

                    assert mock_tqdm.called
                    assert "bankruptcy_years" in result

    def test_early_stopping(self, analyzer, mock_manufacturer):
        """Test early stopping when bankruptcy occurs."""
        config = RuinProbabilityConfig(
            early_stopping=True,
            min_assets_threshold=15_000_000,  # Higher than initial assets
        )

        # Manufacturer will be bankrupt immediately
        result = analyzer._run_single_ruin_simulation(0, 10, config)

        assert result["bankruptcy_year"] == 1  # Stopped early

    def test_no_early_stopping(self, analyzer, mock_manufacturer):
        """Test simulation continues without early stopping."""
        config = RuinProbabilityConfig(
            early_stopping=False,
            min_assets_threshold=15_000_000,
        )

        result = analyzer._run_single_ruin_simulation(0, 5, config)

        # Even if bankrupt, continues to max_horizon
        assert result["bankruptcy_year"] <= 5

    def test_bankruptcy_year_capping(self, analyzer, mock_manufacturer):
        """Test that bankruptcy year is capped at max_horizon."""
        config = RuinProbabilityConfig(
            early_stopping=False,
            min_assets_threshold=5_000_000,  # Won't trigger bankruptcy
        )

        # Mock to make it bankrupt late
        mock_manufacturer.total_assets = 6_000_000

        result = analyzer._run_single_ruin_simulation(0, 3, config)

        assert result["bankruptcy_year"] <= 4  # max_horizon + 1

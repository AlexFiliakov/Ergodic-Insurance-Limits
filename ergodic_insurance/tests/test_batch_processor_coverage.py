"""Coverage-targeted tests for batch_processor.py.

Targets specific uncovered lines: 281-283, 314, 320-323, 329-330, 342-344,
348-350, 539, 559, 710, 714, 717-719, 731-761.
"""

from concurrent.futures import Future
from datetime import datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.batch_processor import (
    AggregatedResults,
    BatchProcessor,
    BatchResult,
    CheckpointData,
    ProcessingStatus,
)
from ergodic_insurance.insurance_program import InsuranceProgram
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import SimulationConfig, SimulationResults
from ergodic_insurance.safe_pickle import safe_dump, safe_load
from ergodic_insurance.scenario_manager import ScenarioConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    loss_gen = MagicMock(spec=ManufacturingLossGenerator)
    insurance = MagicMock(spec=InsuranceProgram)
    manufacturer = MagicMock(spec=WidgetManufacturer)
    return loss_gen, insurance, manufacturer


def _make_sim_results(ruin_prob=None, growth_rates=None, final_assets=None, metrics=None):
    """Helper to create mock SimulationResults."""
    sim = MagicMock(spec=SimulationResults)
    sim.ruin_probability = ruin_prob or {"5": 0.01, "10": 0.03}
    sim.growth_rates = growth_rates if growth_rates is not None else np.array([0.08, 0.09])
    sim.final_assets = final_assets if final_assets is not None else np.array([1e7, 1.1e7])
    sim.metrics = metrics or {
        "var_95": 500000,
        "var_99": 1000000,
        "tvar_95": 750000,
        "tvar_99": 1500000,
    }
    return sim


# ---------------------------------------------------------------------------
# Serial processing: checkpoint periodically (lines 281-283)
# ---------------------------------------------------------------------------
class TestSerialCheckpoint:
    """Test periodic checkpointing in _process_serial."""

    def test_serial_checkpoint_at_interval(self, temp_checkpoint_dir, mock_components):
        """Lines 281-283: _process_serial saves checkpoint at checkpoint_interval."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            use_parallel=False,
            progress_bar=False,
        )

        # Create scenarios that exceed the checkpoint interval
        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(5)]

        with (
            patch.object(processor, "_process_scenario") as mock_process,
            patch.object(processor, "_save_checkpoint") as mock_save,
        ):
            mock_process.return_value = BatchResult("idx", "name", ProcessingStatus.COMPLETED)
            # checkpoint_interval=2: checkpoint at indices 1, 3
            results = processor._process_serial(scenarios, checkpoint_interval=2, max_failures=None)

        # Should have saved checkpoint at least twice (after 2nd and 4th scenario)
        assert mock_save.call_count >= 2


# ---------------------------------------------------------------------------
# Parallel processing coverage (lines 314, 320-323, 329-330, 342-344, 348-350)
# ---------------------------------------------------------------------------
class TestParallelProcessingCoverage:
    """Cover parallel processing paths: progress bar, failure cancellation, exception handling."""

    def test_parallel_with_progress_bar(self, temp_checkpoint_dir, mock_components):
        """Line 314: tqdm wraps iterator when progress_bar is True."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=2,
            use_parallel=True,
            progress_bar=True,  # Enable progress bar
        )

        scenarios = [
            ScenarioConfig("id1", "Scenario 1"),
            ScenarioConfig("id2", "Scenario 2"),
        ]

        with (
            patch.object(processor, "_process_scenario") as mock_process,
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.tqdm") as mock_tqdm,
            patch("ergodic_insurance.batch_processor.as_completed") as mock_completed,
        ):
            # Set up the mock pool and futures
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            future1 = MagicMock(spec=Future)
            future1.result.return_value = BatchResult(
                "id1", "Scenario 1", ProcessingStatus.COMPLETED
            )
            future2 = MagicMock(spec=Future)
            future2.result.return_value = BatchResult(
                "id2", "Scenario 2", ProcessingStatus.COMPLETED
            )

            mock_executor.submit.side_effect = [future1, future2]
            mock_completed.return_value = [future1, future2]
            mock_tqdm.return_value = [future1, future2]

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=None
            )

        # tqdm should have been called (line 314)
        mock_tqdm.assert_called_once()

    def test_parallel_max_failures_cancels_futures(self, temp_checkpoint_dir, mock_components):
        """Lines 320-323: Remaining futures are cancelled when max_failures is reached."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=2,
            use_parallel=True,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(5)]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.as_completed") as mock_completed,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            # Create futures where all fail
            futures = []
            for i, s in enumerate(scenarios):
                f = MagicMock(spec=Future)
                f.result.return_value = BatchResult(
                    s.scenario_id, s.name, ProcessingStatus.FAILED, error_message="fail"
                )
                futures.append(f)
            mock_executor.submit.side_effect = futures
            mock_completed.return_value = futures

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=2
            )

        # Should have stopped after 2 failures and cancelled remaining
        assert len([r for r in results if r.status == ProcessingStatus.FAILED]) <= 3

    def test_parallel_future_exception(self, temp_checkpoint_dir, mock_components):
        """Lines 329-330: Future raising exception creates FAILED BatchResult."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=2,
            use_parallel=True,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig("id1", "Scenario 1")]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.as_completed") as mock_completed,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            future = MagicMock(spec=Future)
            future.result.side_effect = RuntimeError("process failed")
            mock_executor.submit.return_value = future
            mock_completed.return_value = [future]

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=None
            )

        assert len(results) == 1
        assert results[0].status == ProcessingStatus.FAILED
        assert results[0].error_message is not None
        assert "process failed" in results[0].error_message

    def test_parallel_failed_scenario_tracking(self, temp_checkpoint_dir, mock_components):
        """Lines 342-344: Failed scenarios are tracked in processor state."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=2,
            use_parallel=True,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig("fail_id", "Fail Scenario")]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.as_completed") as mock_completed,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            future = MagicMock(spec=Future)
            future.result.return_value = BatchResult(
                "fail_id", "Fail Scenario", ProcessingStatus.FAILED
            )
            mock_executor.submit.return_value = future
            mock_completed.return_value = [future]

            processor._process_parallel(scenarios, checkpoint_interval=100, max_failures=None)

        assert "fail_id" in processor.failed_scenarios

    def test_parallel_checkpoint_at_interval(self, temp_checkpoint_dir, mock_components):
        """Lines 348-350: Checkpoint saved at checkpoint_interval during parallel processing."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=2,
            use_parallel=True,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(4)]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.as_completed") as mock_completed,
            patch.object(processor, "_save_checkpoint") as mock_save,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            futures = []
            for i, s in enumerate(scenarios):
                f = MagicMock(spec=Future)
                f.result.return_value = BatchResult(
                    s.scenario_id, s.name, ProcessingStatus.COMPLETED
                )
                futures.append(f)
            mock_executor.submit.side_effect = futures
            mock_completed.return_value = futures

            # checkpoint_interval=2 should trigger checkpoint after 2nd scenario
            processor._process_parallel(scenarios, checkpoint_interval=2, max_failures=None)

        assert mock_save.call_count >= 1


# ---------------------------------------------------------------------------
# Aggregation: ruin_probability_rank (line 539)
# ---------------------------------------------------------------------------
class TestAggregationRuinProbRank:
    """Test aggregation produces ruin_probability_rank column."""

    def test_ruin_probability_rank_in_ranking(self, temp_checkpoint_dir):
        """Line 539: Rankings include ruin_probability_rank."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        sim1 = _make_sim_results(ruin_prob={"5": 0.02, "10": 0.05})
        sim2 = _make_sim_results(ruin_prob={"5": 0.01, "10": 0.03})

        processor.batch_results = [
            BatchResult("id1", "Scenario 1", ProcessingStatus.COMPLETED, simulation_results=sim1),
            BatchResult("id2", "Scenario 2", ProcessingStatus.COMPLETED, simulation_results=sim2),
        ]

        aggregated = processor._aggregate_results()
        assert "rankings" in aggregated.comparison_metrics
        rankings = aggregated.comparison_metrics["rankings"]
        assert "ruin_probability_rank" in rankings.columns


# ---------------------------------------------------------------------------
# Sensitivity analysis: baseline without sim results (line 559)
# ---------------------------------------------------------------------------
class TestSensitivityBaselineNoResults:
    """Test sensitivity analysis when baseline has no simulation results."""

    def test_baseline_no_sim_results_returns_none(self, temp_checkpoint_dir):
        """Line 559: Returns None when baseline has no simulation_results."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        processor.batch_results = [
            BatchResult(
                "baseline",
                "Baseline",
                ProcessingStatus.COMPLETED,
                simulation_results=None,
                metadata={"tags": ["baseline"]},
            ),
        ]

        result = processor._perform_sensitivity_analysis()
        assert result is None


# ---------------------------------------------------------------------------
# Export excel with sensitivity and comparison metrics (lines 710, 714, 717-719)
# ---------------------------------------------------------------------------
class TestExportExcelAdvanced:
    """Test Excel export with sensitivity analysis and comparison metrics."""

    def test_export_excel_with_sensitivity_and_comparisons(self, temp_checkpoint_dir):
        """Lines 710, 714, 717-719: Excel export writes sensitivity and comparison sheets."""
        pytest.importorskip("openpyxl")

        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        sim1 = _make_sim_results(ruin_prob={"5": 0.02, "10": 0.05})
        sim2 = _make_sim_results(ruin_prob={"5": 0.01, "10": 0.03})

        processor.batch_results = [
            BatchResult(
                "baseline",
                "Baseline",
                ProcessingStatus.COMPLETED,
                simulation_results=sim1,
                metadata={"tags": ["baseline"]},
            ),
            BatchResult(
                "sens1",
                "Sensitivity 1",
                ProcessingStatus.COMPLETED,
                simulation_results=sim2,
                metadata={"tags": ["sensitivity"]},
            ),
        ]

        export_path = temp_checkpoint_dir / "results.xlsx"
        processor.export_results(export_path, "excel")

        assert export_path.exists()
        with pd.ExcelFile(export_path) as xl_file:
            assert "Summary" in xl_file.sheet_names
            assert "Details" in xl_file.sheet_names


# ---------------------------------------------------------------------------
# Export excel_financial (lines 731-761)
# ---------------------------------------------------------------------------
class TestExportFinancialStatements:
    """Test export_financial_statements method."""

    def test_export_financial_calls_reporter(self, temp_checkpoint_dir, mock_components):
        """Lines 731-761: export_financial_statements generates reports."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )

        sim_results = _make_sim_results()
        processor.batch_results = [
            BatchResult(
                "id1",
                "Scenario_1",
                ProcessingStatus.COMPLETED,
                simulation_results=sim_results,
            ),
        ]

        export_path = temp_checkpoint_dir / "financial_reports"

        with patch("ergodic_insurance.batch_processor.ExcelReporter") as MockReporter:
            mock_reporter = MockReporter.return_value
            processor.export_financial_statements(export_path)

            # Should have been called once for the completed scenario
            mock_reporter.generate_monte_carlo_report.assert_called_once()

    def test_export_financial_handles_error(self, temp_checkpoint_dir, mock_components):
        """Lines 731-761: export_financial_statements handles report errors."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )

        sim_results = _make_sim_results()
        processor.batch_results = [
            BatchResult(
                "id1",
                "Error_Scenario",
                ProcessingStatus.COMPLETED,
                simulation_results=sim_results,
            ),
        ]

        export_path = temp_checkpoint_dir / "financial_reports"

        with patch("ergodic_insurance.batch_processor.ExcelReporter") as MockReporter:
            mock_reporter = MockReporter.return_value
            mock_reporter.generate_monte_carlo_report.side_effect = OSError("write failed")
            # Should not raise - error is caught and printed
            processor.export_financial_statements(export_path)

    def test_export_financial_skips_non_completed(self, temp_checkpoint_dir, mock_components):
        """export_financial_statements skips non-completed scenarios."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )

        processor.batch_results = [
            BatchResult("id1", "Failed", ProcessingStatus.FAILED),
            BatchResult("id2", "NoResults", ProcessingStatus.COMPLETED, simulation_results=None),
        ]

        export_path = temp_checkpoint_dir / "financial_reports"

        with patch("ergodic_insurance.batch_processor.ExcelReporter") as MockReporter:
            mock_reporter = MockReporter.return_value
            processor.export_financial_statements(export_path)
            mock_reporter.generate_monte_carlo_report.assert_not_called()

    def test_export_excel_financial_format(self, temp_checkpoint_dir, mock_components):
        """Lines 717-719: export_results with 'excel_financial' format delegates to export_financial_statements."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )
        processor.batch_results = []

        with patch.object(processor, "export_financial_statements") as mock_export:
            processor.export_results(temp_checkpoint_dir / "output", "excel_financial")
            mock_export.assert_called_once()


# ---------------------------------------------------------------------------
# get_final_ruin_probability helper
# ---------------------------------------------------------------------------
class TestGetFinalRuinProbability:
    """Test _get_final_ruin_probability helper."""

    def test_empty_dict_returns_zero(self, temp_checkpoint_dir):
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        assert processor._get_final_ruin_probability({}) == 0.0

    def test_returns_max_year_value(self, temp_checkpoint_dir):
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        result = processor._get_final_ruin_probability({"5": 0.01, "10": 0.03, "20": 0.05})
        assert result == 0.05

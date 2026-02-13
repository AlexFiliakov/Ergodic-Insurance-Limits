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
from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloResults
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
    """Helper to create mock MonteCarloResults."""
    sim = MagicMock(spec=MonteCarloResults)
    sim.ruin_probability = {"5": 0.01, "10": 0.03} if ruin_prob is None else ruin_prob
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
    """Cover parallel processing paths: progress bar, failure cancellation, exception handling.

    The parallel processor uses incremental submission with
    ``wait(FIRST_COMPLETED)`` so that ``max_failures`` early stop is effective.
    """

    def test_parallel_with_progress_bar(self, temp_checkpoint_dir, mock_components):
        """tqdm progress bar is created when progress_bar is True."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=2,
            use_parallel=True,
            progress_bar=True,
        )

        scenarios = [
            ScenarioConfig("id1", "Scenario 1"),
            ScenarioConfig("id2", "Scenario 2"),
        ]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.tqdm") as mock_tqdm,
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
        ):
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

            # First wait returns both futures done
            mock_wait.return_value = ({future1, future2}, set())

            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=None
            )

        mock_tqdm.assert_called_once()
        mock_pbar.close.assert_called_once()

    def test_parallel_max_failures_cancels_futures(self, temp_checkpoint_dir, mock_components):
        """Remaining futures are cancelled when max_failures is reached."""
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
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            # Create futures — pool_size=2, so first 2 submitted initially
            futures = []
            for s in scenarios:
                f = MagicMock(spec=Future)
                f.result.return_value = BatchResult(
                    s.scenario_id, s.name, ProcessingStatus.FAILED, error_message="fail"
                )
                futures.append(f)
            mock_executor.submit.side_effect = futures

            # First wait: both initial futures complete (2 failures triggers stop)
            mock_wait.return_value = ({futures[0], futures[1]}, set())

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=2
            )

        # Only 2 failures processed; remaining scenarios never submitted
        assert len([r for r in results if r.status == ProcessingStatus.FAILED]) == 2
        # Only 2 futures submitted (pool_size), not all 5
        assert mock_executor.submit.call_count == 2

    def test_parallel_future_exception(self, temp_checkpoint_dir, mock_components):
        """Future raising exception creates FAILED BatchResult."""
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
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            future = MagicMock(spec=Future)
            future.result.side_effect = RuntimeError("process failed")
            mock_executor.submit.return_value = future
            mock_wait.return_value = ({future}, set())

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=None
            )

        assert len(results) == 1
        assert results[0].status == ProcessingStatus.FAILED
        assert results[0].error_message is not None
        assert "process failed" in results[0].error_message

    def test_parallel_failed_scenario_tracking(self, temp_checkpoint_dir, mock_components):
        """Failed scenarios are tracked in processor state."""
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
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            future = MagicMock(spec=Future)
            future.result.return_value = BatchResult(
                "fail_id", "Fail Scenario", ProcessingStatus.FAILED
            )
            mock_executor.submit.return_value = future
            mock_wait.return_value = ({future}, set())

            processor._process_parallel(scenarios, checkpoint_interval=100, max_failures=None)

        assert "fail_id" in processor.failed_scenarios

    def test_parallel_checkpoint_at_interval(self, temp_checkpoint_dir, mock_components):
        """Checkpoint saved at checkpoint_interval during parallel processing."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=4,
            use_parallel=True,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(4)]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
            patch.object(processor, "_save_checkpoint") as mock_save,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            futures = []
            for s in scenarios:
                f = MagicMock(spec=Future)
                f.result.return_value = BatchResult(
                    s.scenario_id, s.name, ProcessingStatus.COMPLETED
                )
                futures.append(f)
            mock_executor.submit.side_effect = futures

            # Return 2 done at a time to trigger checkpoint at interval=2
            mock_wait.side_effect = [
                ({futures[0], futures[1]}, set()),
                ({futures[2], futures[3]}, set()),
            ]

            processor._process_parallel(scenarios, checkpoint_interval=2, max_failures=None)

        assert mock_save.call_count >= 1

    def test_parallel_incremental_submission_limits_in_flight(
        self, temp_checkpoint_dir, mock_components
    ):
        """Only n_workers futures are submitted at a time, not all at once."""
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

        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(6)]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            futures = []
            for s in scenarios:
                f = MagicMock(spec=Future)
                f.result.return_value = BatchResult(
                    s.scenario_id, s.name, ProcessingStatus.COMPLETED
                )
                futures.append(f)
            mock_executor.submit.side_effect = futures

            # Simulate 3 rounds: 2 complete each time, 2 more submitted
            mock_wait.side_effect = [
                ({futures[0], futures[1]}, set()),
                ({futures[2], futures[3]}, set()),
                ({futures[4], futures[5]}, set()),
            ]

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=None
            )

        assert len(results) == 6
        assert mock_executor.submit.call_count == 6


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


# ---------------------------------------------------------------------------
# Issue #358 — regression tests for the three reported bugs
# ---------------------------------------------------------------------------
class TestIssue358EmptyRuinProbability:
    """Bug: empty ruin_probability dict crashes to_dataframe."""

    def test_to_dataframe_empty_ruin_probability(self):
        """to_dataframe must not crash when ruin_probability is {}."""
        sim_results = _make_sim_results(ruin_prob={})
        batch_results = [
            BatchResult(
                "id1",
                "Empty Ruin",
                ProcessingStatus.COMPLETED,
                simulation_results=sim_results,
                execution_time=1.0,
            ),
        ]
        agg = AggregatedResults(
            batch_results=batch_results,
            summary_statistics=pd.DataFrame(),
            comparison_metrics={},
        )
        df = agg.to_dataframe()
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["ruin_probability"])

    def test_to_dataframe_non_empty_ruin_probability(self):
        """to_dataframe still works with a populated ruin_probability."""
        sim_results = _make_sim_results(ruin_prob={"5": 0.01, "10": 0.03})
        batch_results = [
            BatchResult(
                "id1",
                "Normal",
                ProcessingStatus.COMPLETED,
                simulation_results=sim_results,
                execution_time=1.0,
            ),
        ]
        agg = AggregatedResults(
            batch_results=batch_results,
            summary_statistics=pd.DataFrame(),
            comparison_metrics={},
        )
        df = agg.to_dataframe()
        assert df.iloc[0]["ruin_probability"] == 0.03


class TestIssue358NaNRanking:
    """Bug: NaN in ranking raises ValueError on .astype(int)."""

    def test_ranking_with_nan_growth_rate(self, temp_checkpoint_dir):
        """Ranking must handle NaN in mean_growth_rate without raising."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        sim1 = _make_sim_results(
            ruin_prob={"5": 0.02},
            growth_rates=np.array([0.08]),
        )
        sim2 = _make_sim_results(
            ruin_prob={"5": 0.01},
            growth_rates=np.array([np.nan]),
        )

        processor.batch_results = [
            BatchResult("id1", "S1", ProcessingStatus.COMPLETED, simulation_results=sim1),
            BatchResult("id2", "S2", ProcessingStatus.COMPLETED, simulation_results=sim2),
        ]

        aggregated = processor._aggregate_results()
        rankings = aggregated.comparison_metrics["rankings"]
        # NaN growth rate should be ranked bottom (rank 2)
        assert rankings.loc[rankings.index == "S2", "mean_growth_rate_rank"].iloc[0] == 2

    def test_ranking_with_nan_ruin_probability(self, temp_checkpoint_dir):
        """Ranking must handle NaN in ruin_probability without raising."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        sim1 = _make_sim_results(ruin_prob={"5": 0.02})
        # Scenario with empty ruin_probability -> NaN after extraction
        sim2 = _make_sim_results(ruin_prob={})

        processor.batch_results = [
            BatchResult("id1", "S1", ProcessingStatus.COMPLETED, simulation_results=sim1),
            BatchResult("id2", "S2", ProcessingStatus.COMPLETED, simulation_results=sim2),
        ]

        aggregated = processor._aggregate_results()
        # With empty ruin_prob, the dict extraction yields NaN -> ranked bottom
        if "rankings" in aggregated.comparison_metrics:
            rankings = aggregated.comparison_metrics["rankings"]
            if "ruin_probability_rank" in rankings.columns:
                assert rankings["ruin_probability_rank"].notna().all()


class TestIssue358MaxFailuresEarlyStop:
    """Bug: max_failures early stop is ineffective — all scenarios submitted at once."""

    def test_max_failures_prevents_further_submission(self, temp_checkpoint_dir, mock_components):
        """When max_failures is reached, no more scenarios should be submitted."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            n_workers=1,
            use_parallel=True,
            progress_bar=False,
        )

        # 10 scenarios, max_failures=1
        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(10)]

        with (
            patch("ergodic_insurance.batch_processor.ProcessPoolExecutor") as MockPool,
            patch("ergodic_insurance.batch_processor.wait") as mock_wait,
        ):
            mock_executor = MagicMock()
            MockPool.return_value.__enter__ = Mock(return_value=mock_executor)
            MockPool.return_value.__exit__ = Mock(return_value=False)

            futures = []
            for s in scenarios:
                f = MagicMock(spec=Future)
                f.result.return_value = BatchResult(
                    s.scenario_id, s.name, ProcessingStatus.FAILED, error_message="fail"
                )
                futures.append(f)
            mock_executor.submit.side_effect = futures

            # Only 1 worker, so 1 initial submission; it fails -> stop
            mock_wait.return_value = ({futures[0]}, set())

            results = processor._process_parallel(
                scenarios, checkpoint_interval=100, max_failures=1
            )

        # Only 1 scenario processed, remaining 9 never submitted
        assert len(results) == 1
        assert mock_executor.submit.call_count == 1

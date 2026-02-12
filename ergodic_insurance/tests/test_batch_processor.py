"""Comprehensive tests for the batch_processor module."""

from concurrent.futures import Future
from datetime import datetime
import json
from pathlib import Path
import pickle
import tempfile
import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

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


class TestProcessingStatus:
    """Test ProcessingStatus enum."""

    def test_enum_values(self):
        """Test that all expected status values exist."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.RUNNING.value == "running"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.SKIPPED.value == "skipped"


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_initialization_minimal(self):
        """Test minimal BatchResult initialization."""
        result = BatchResult(
            scenario_id="test_id", scenario_name="Test Scenario", status=ProcessingStatus.COMPLETED
        )
        assert result.scenario_id == "test_id"
        assert result.scenario_name == "Test Scenario"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.simulation_results is None
        assert result.execution_time == 0.0
        assert result.error_message is None
        assert result.metadata == {}

    def test_initialization_full(self):
        """Test full BatchResult initialization."""
        sim_results = MagicMock(spec=MonteCarloResults)
        result = BatchResult(
            scenario_id="test_id",
            scenario_name="Test Scenario",
            status=ProcessingStatus.COMPLETED,
            simulation_results=sim_results,
            execution_time=10.5,
            error_message="Some error",
            metadata={"key": "value"},
        )
        assert result.simulation_results == sim_results
        assert result.execution_time == 10.5
        assert result.error_message == "Some error"
        assert result.metadata == {"key": "value"}


class TestAggregatedResults:
    """Test AggregatedResults dataclass."""

    def test_initialization(self):
        """Test AggregatedResults initialization."""
        batch_results = [
            BatchResult("id1", "Name1", ProcessingStatus.COMPLETED),
            BatchResult("id2", "Name2", ProcessingStatus.FAILED),
        ]
        summary_stats = pd.DataFrame({"col": [1, 2, 3]})
        comparison_metrics = {"metric1": pd.DataFrame({"data": [1, 2]})}
        sensitivity = pd.DataFrame({"sensitivity": [0.1, 0.2]})
        execution_summary = {"total": 2}

        results = AggregatedResults(
            batch_results=batch_results,
            summary_statistics=summary_stats,
            comparison_metrics=comparison_metrics,
            sensitivity_analysis=sensitivity,
            execution_summary=execution_summary,
        )
        assert len(results.batch_results) == 2
        assert results.summary_statistics.equals(summary_stats)
        assert "metric1" in results.comparison_metrics
        assert results.sensitivity_analysis is not None
        assert results.execution_summary["total"] == 2

    def test_get_successful_results(self):
        """Test getting only successful results."""
        batch_results = [
            BatchResult("id1", "Name1", ProcessingStatus.COMPLETED),
            BatchResult("id2", "Name2", ProcessingStatus.FAILED),
            BatchResult("id3", "Name3", ProcessingStatus.COMPLETED),
            BatchResult("id4", "Name4", ProcessingStatus.SKIPPED),
        ]
        results = AggregatedResults(
            batch_results=batch_results, summary_statistics=pd.DataFrame(), comparison_metrics={}
        )
        successful = results.get_successful_results()
        assert len(successful) == 2
        assert all(r.status == ProcessingStatus.COMPLETED for r in successful)

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        sim_results = MagicMock(spec=MonteCarloResults)
        sim_results.ruin_probability = {"5": 0.01}
        sim_results.growth_rates = np.array([0.08, 0.09, 0.07])
        sim_results.final_assets = np.array([1e7, 1.1e7, 0.9e7])
        sim_results.metrics = {"var_99": 1000000, "tvar_99": 1500000}

        batch_results = [
            BatchResult(
                "id1",
                "Name1",
                ProcessingStatus.COMPLETED,
                simulation_results=sim_results,
                execution_time=5.0,
            ),
            BatchResult("id2", "Name2", ProcessingStatus.FAILED, execution_time=2.0),
        ]

        results = AggregatedResults(
            batch_results=batch_results, summary_statistics=pd.DataFrame(), comparison_metrics={}
        )
        df = results.to_dataframe()
        assert len(df) == 2
        assert df.iloc[0]["scenario_id"] == "id1"
        assert df.iloc[0]["status"] == "completed"
        assert df.iloc[0]["ruin_probability"] == 0.01
        assert np.isclose(df.iloc[0]["mean_growth_rate"], 0.08)
        assert df.iloc[0]["var_99"] == 1000000
        assert df.iloc[1]["scenario_id"] == "id2"
        assert df.iloc[1]["status"] == "failed"


class TestCheckpointData:
    """Test CheckpointData dataclass."""

    def test_initialization(self):
        """Test CheckpointData initialization."""
        batch_results = [BatchResult("id1", "Name1", ProcessingStatus.COMPLETED)]
        checkpoint = CheckpointData(
            completed_scenarios={"id1", "id2"},
            failed_scenarios={"id3"},
            batch_results=batch_results,
            timestamp=datetime.now(),
            metadata={"key": "value"},
        )
        assert len(checkpoint.completed_scenarios) == 2
        assert len(checkpoint.failed_scenarios) == 1
        assert len(checkpoint.batch_results) == 1
        assert checkpoint.metadata["key"] == "value"


class TestBatchProcessor:
    """Test BatchProcessor class."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        loss_gen = MagicMock(spec=ManufacturingLossGenerator)
        insurance = MagicMock(spec=InsuranceProgram)
        manufacturer = MagicMock(spec=WidgetManufacturer)
        return loss_gen, insurance, manufacturer

    def test_initialization_minimal(self, temp_checkpoint_dir):
        """Test minimal BatchProcessor initialization."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        assert processor.loss_generator is None
        assert processor.insurance_program is None
        assert processor.manufacturer is None
        assert processor.n_workers is None
        assert processor.checkpoint_dir == temp_checkpoint_dir
        assert processor.use_parallel is True
        assert processor.progress_bar is True
        assert len(processor.batch_results) == 0
        assert len(processor.completed_scenarios) == 0
        assert len(processor.failed_scenarios) == 0

    def test_initialization_full(self, temp_checkpoint_dir, mock_components):
        """Test full BatchProcessor initialization."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            n_workers=4,
            checkpoint_dir=temp_checkpoint_dir,
            use_parallel=False,
            progress_bar=False,
        )
        assert processor.loss_generator == loss_gen
        assert processor.insurance_program == insurance
        assert processor.manufacturer == manufacturer
        assert processor.n_workers == 4
        assert processor.use_parallel is False
        assert processor.progress_bar is False

    def test_process_batch_empty_scenarios(self, temp_checkpoint_dir, mock_components):
        """Test processing empty scenario list."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            use_parallel=False,
        )
        results = processor.process_batch([])
        assert len(results.batch_results) == 0

    def test_process_batch_priority_filter(self, temp_checkpoint_dir, mock_components):
        """Test scenario filtering by priority."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            use_parallel=False,
        )

        scenarios = [
            ScenarioConfig("id1", "Low Priority", priority=1),
            ScenarioConfig("id2", "Medium Priority", priority=5),
            ScenarioConfig("id3", "High Priority", priority=10),
        ]

        with patch.object(processor, "_process_serial") as mock_serial:
            mock_serial.return_value = []
            processor.process_batch(scenarios, priority_threshold=5)
            # Should process only scenarios with priority <= 5
            called_scenarios = mock_serial.call_args[0][0]
            assert len(called_scenarios) == 2
            assert all(s.priority <= 5 for s in called_scenarios)

    def test_process_batch_serial(self, temp_checkpoint_dir, mock_components):
        """Test serial processing of scenarios."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            use_parallel=False,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig("id1", "Scenario 1"), ScenarioConfig("id2", "Scenario 2")]

        # Mock _process_scenario to return successful results
        with patch.object(processor, "_process_scenario") as mock_process:
            mock_process.side_effect = [
                BatchResult("id1", "Scenario 1", ProcessingStatus.COMPLETED, execution_time=1.0),
                BatchResult("id2", "Scenario 2", ProcessingStatus.COMPLETED, execution_time=2.0),
            ]
            results = processor.process_batch(scenarios, resume_from_checkpoint=False)

        assert len(results.batch_results) == 2
        assert results.execution_summary["completed"] == 2
        assert results.execution_summary["failed"] == 0

    def test_process_batch_parallel(self, temp_checkpoint_dir, mock_components):
        """Test parallel processing of scenarios."""
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

        scenarios = [
            ScenarioConfig("id1", "Scenario 1"),
            ScenarioConfig("id2", "Scenario 2"),
            ScenarioConfig("id3", "Scenario 3"),
        ]

        # Mock _process_parallel to avoid pickling issues with MagicMock
        def mock_parallel_func(scenarios, checkpoint_interval, max_failures):
            # Update processor state as the real method would
            processor.completed_scenarios.add("id1")
            processor.completed_scenarios.add("id3")
            processor.failed_scenarios.add("id2")
            return [
                BatchResult("id1", "Scenario 1", ProcessingStatus.COMPLETED),
                BatchResult("id2", "Scenario 2", ProcessingStatus.FAILED),
                BatchResult("id3", "Scenario 3", ProcessingStatus.COMPLETED),
            ]

        with patch.object(processor, "_process_parallel", side_effect=mock_parallel_func):
            results = processor.process_batch(scenarios, resume_from_checkpoint=False)

        assert len(results.batch_results) == 3
        assert results.execution_summary["completed"] == 2
        assert results.execution_summary["failed"] == 1

    def test_process_batch_max_failures(self, temp_checkpoint_dir, mock_components):
        """Test stopping at max failures."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
            use_parallel=False,
            progress_bar=False,
        )

        scenarios = [ScenarioConfig(f"id{i}", f"Scenario {i}") for i in range(10)]

        # Mock _process_scenario to return failures
        with patch.object(processor, "_process_scenario") as mock_process:
            mock_process.return_value = BatchResult("id", "name", ProcessingStatus.FAILED)
            results = processor.process_batch(
                scenarios, resume_from_checkpoint=False, max_failures=3
            )

        # Should stop after 3 failures
        assert len(results.batch_results) == 3
        assert results.execution_summary["failed"] == 3

    def test_process_scenario_success(self, temp_checkpoint_dir, mock_components):
        """Test successful scenario processing."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )

        scenario = ScenarioConfig(
            "test_id", "Test Scenario", simulation_config=MonteCarloConfig(n_simulations=10)
        )

        # Mock MonteCarloEngine
        with patch("ergodic_insurance.batch_processor.MonteCarloEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_results = MagicMock(spec=MonteCarloResults)
            mock_engine.run.return_value = mock_results

            with patch("time.time") as mock_time:
                # Mock time to ensure execution_time > 0
                mock_time.side_effect = [0.0, 1.5]  # start_time=0, end_time=1.5
                result = processor._process_scenario(scenario)

        assert result.scenario_id == "test_id"
        assert result.scenario_name == "Test Scenario"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.simulation_results == mock_results
        assert result.execution_time == 1.5

    def test_process_scenario_with_overrides(self, temp_checkpoint_dir, mock_components):
        """Test scenario processing with parameter overrides."""
        loss_gen, insurance, manufacturer = mock_components
        manufacturer.base_operating_margin = 0.08

        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )

        scenario = ScenarioConfig(
            "test_id",
            "Test Scenario",
            parameter_overrides={"manufacturer.base_operating_margin": 0.12},
        )

        with patch("ergodic_insurance.batch_processor.MonteCarloEngine") as MockEngine:
            mock_engine = MockEngine.return_value
            mock_results = MagicMock(spec=MonteCarloResults)
            mock_engine.run.return_value = mock_results

            result = processor._process_scenario(scenario)

            # Check that override was applied to the copy
            called_manufacturer = MockEngine.call_args[1]["manufacturer"]
            assert called_manufacturer.base_operating_margin == 0.12

    def test_process_scenario_failure(self, temp_checkpoint_dir, mock_components):
        """Test scenario processing failure."""
        loss_gen, insurance, manufacturer = mock_components
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            checkpoint_dir=temp_checkpoint_dir,
        )

        scenario = ScenarioConfig("test_id", "Test Scenario")

        # Mock MonteCarloEngine to raise an error
        with patch("ergodic_insurance.batch_processor.MonteCarloEngine") as MockEngine:
            MockEngine.side_effect = ValueError("Test error")
            result = processor._process_scenario(scenario)

        assert result.status == ProcessingStatus.FAILED
        assert result.error_message == "Test error"

    def test_process_scenario_missing_components(self, temp_checkpoint_dir):
        """Test scenario processing with missing components."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        scenario = ScenarioConfig("test_id", "Test Scenario")

        result = processor._process_scenario(scenario)
        assert result.status == ProcessingStatus.FAILED
        assert result.error_message and "requires" in result.error_message

    def test_aggregate_results_empty(self, temp_checkpoint_dir):
        """Test aggregation with no results."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        aggregated = processor._aggregate_results()
        assert len(aggregated.batch_results) == 0
        assert aggregated.summary_statistics.empty

    def test_aggregate_results_with_data(self, temp_checkpoint_dir):
        """Test aggregation with simulation results."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        # Create mock simulation results
        sim_results = MagicMock(spec=MonteCarloResults)
        sim_results.ruin_probability = {"5": 0.01}
        sim_results.growth_rates = np.array([0.08, 0.09, 0.07])
        sim_results.final_assets = np.array([1e7, 1.1e7, 0.9e7])
        sim_results.metrics = {
            "var_95": 500000,
            "var_99": 1000000,
            "tvar_95": 750000,
            "tvar_99": 1500000,
        }

        processor.batch_results = [
            BatchResult(
                "id1",
                "Scenario 1",
                ProcessingStatus.COMPLETED,
                simulation_results=sim_results,
                execution_time=5.0,
            ),
            BatchResult("id2", "Scenario 2", ProcessingStatus.FAILED, execution_time=2.0),
        ]

        aggregated = processor._aggregate_results()
        assert len(aggregated.batch_results) == 2
        assert len(aggregated.summary_statistics) == 1  # Only completed scenarios
        assert "mean_growth_rate" in aggregated.summary_statistics.columns

    def test_aggregate_results_with_comparison(self, temp_checkpoint_dir):
        """Test aggregation with comparison metrics."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        # Create multiple simulation results
        sim_results1 = MagicMock(spec=MonteCarloResults)
        sim_results1.ruin_probability = {"5": 0.02, "10": 0.05, "20": 0.08}
        sim_results1.growth_rates = np.array([0.08, 0.09])
        sim_results1.final_assets = np.array([1e7, 1.1e7])
        sim_results1.metrics = {"var_95": 500000, "var_99": 1000000}

        sim_results2 = MagicMock(spec=MonteCarloResults)
        sim_results2.ruin_probability = {"5": 0.01, "10": 0.03, "20": 0.06}
        sim_results2.growth_rates = np.array([0.06, 0.07])
        sim_results2.final_assets = np.array([0.9e7, 0.95e7])
        sim_results2.metrics = {"var_95": 600000, "var_99": 1200000}

        processor.batch_results = [
            BatchResult(
                "id1", "Scenario 1", ProcessingStatus.COMPLETED, simulation_results=sim_results1
            ),
            BatchResult(
                "id2", "Scenario 2", ProcessingStatus.COMPLETED, simulation_results=sim_results2
            ),
        ]

        aggregated = processor._aggregate_results()
        assert "relative_performance" in aggregated.comparison_metrics
        assert "rankings" in aggregated.comparison_metrics

    def test_perform_sensitivity_analysis(self, temp_checkpoint_dir):
        """Test sensitivity analysis."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        # Create baseline and sensitivity results
        baseline_results = MagicMock(spec=MonteCarloResults)
        baseline_results.ruin_probability = {"5": 0.01}
        baseline_results.growth_rates = np.array([0.08])
        baseline_results.final_assets = np.array([1e7])

        sensitivity_results = MagicMock(spec=MonteCarloResults)
        sensitivity_results.ruin_probability = {"5": 0.015}
        sensitivity_results.growth_rates = np.array([0.085])
        sensitivity_results.final_assets = np.array([1.05e7])

        processor.batch_results = [
            BatchResult(
                "baseline",
                "Baseline",
                ProcessingStatus.COMPLETED,
                simulation_results=baseline_results,
                metadata={"tags": ["baseline"]},
            ),
            BatchResult(
                "sens1",
                "Sensitivity 1",
                ProcessingStatus.COMPLETED,
                simulation_results=sensitivity_results,
                metadata={"tags": ["sensitivity"]},
            ),
        ]

        sensitivity = processor._perform_sensitivity_analysis()
        assert sensitivity is not None
        assert len(sensitivity) == 1
        assert "growth_rate_change_pct" in sensitivity.columns

    def test_save_checkpoint(self, temp_checkpoint_dir):
        """Test checkpoint saving."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        processor.completed_scenarios.add("id1")
        processor.failed_scenarios.add("id2")
        processor.batch_results.append(BatchResult("id1", "Name1", ProcessingStatus.COMPLETED))

        processor._save_checkpoint()

        # Check that checkpoint file was created
        checkpoints = list(temp_checkpoint_dir.glob("checkpoint_*.pkl"))
        assert len(checkpoints) == 1

        # Load and verify checkpoint
        with open(checkpoints[0], "rb") as f:
            checkpoint = safe_load(f)
        assert "id1" in checkpoint.completed_scenarios
        assert "id2" in checkpoint.failed_scenarios
        assert len(checkpoint.batch_results) == 1

    def test_load_checkpoint(self, temp_checkpoint_dir):
        """Test checkpoint loading."""
        # Create a checkpoint file
        checkpoint = CheckpointData(
            completed_scenarios={"id1", "id2"},
            failed_scenarios={"id3"},
            batch_results=[BatchResult("id1", "Name1", ProcessingStatus.COMPLETED)],
            timestamp=datetime.now(),
        )
        checkpoint_path = temp_checkpoint_dir / "checkpoint_20240101_120000.pkl"
        with open(checkpoint_path, "wb") as f:
            safe_dump(checkpoint, f)

        # Load checkpoint
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        loaded = processor._load_checkpoint()

        assert loaded is True
        assert len(processor.completed_scenarios) == 2
        assert len(processor.failed_scenarios) == 1
        assert len(processor.batch_results) == 1

    def test_clear_checkpoints(self, temp_checkpoint_dir):
        """Test clearing checkpoints."""
        # Create checkpoint files
        for i in range(3):
            path = temp_checkpoint_dir / f"checkpoint_{i}.pkl"
            path.touch()

        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        processor.completed_scenarios.add("id1")
        processor.batch_results.append(BatchResult("id1", "Name1", ProcessingStatus.COMPLETED))

        processor.clear_checkpoints()

        # Check that files are removed and state is cleared
        assert len(list(temp_checkpoint_dir.glob("checkpoint_*.pkl"))) == 0
        assert len(processor.completed_scenarios) == 0
        assert len(processor.batch_results) == 0

    def test_export_results_csv(self, temp_checkpoint_dir):
        """Test exporting results to CSV."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        processor.batch_results = [
            BatchResult("id1", "Name1", ProcessingStatus.COMPLETED, execution_time=5.0)
        ]

        export_path = temp_checkpoint_dir / "results.csv"
        processor.export_results(export_path, "csv")

        assert export_path.exists()
        df = pd.read_csv(export_path)
        assert len(df) == 1
        assert df.iloc[0]["scenario_id"] == "id1"

    def test_export_results_json(self, temp_checkpoint_dir):
        """Test exporting results to JSON."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        processor.batch_results = [
            BatchResult(
                "id1", "Name1", ProcessingStatus.COMPLETED, execution_time=5.0, error_message="test"
            )
        ]

        export_path = temp_checkpoint_dir / "results.json"
        processor.export_results(export_path, "json")

        assert export_path.exists()
        with open(export_path, "r") as f:
            data = json.load(f)
        assert len(data["batch_results"]) == 1
        assert data["batch_results"][0]["scenario_id"] == "id1"

    def test_export_results_excel(self, temp_checkpoint_dir):
        """Test exporting results to Excel."""
        pytest.importorskip("openpyxl")  # Skip test if openpyxl is not installed

        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        sim_results = MagicMock(spec=MonteCarloResults)
        sim_results.ruin_probability = {"5": 0.01}
        sim_results.growth_rates = np.array([0.08])
        sim_results.final_assets = np.array([1e7])
        sim_results.metrics = {}

        processor.batch_results = [
            BatchResult("id1", "Name1", ProcessingStatus.COMPLETED, simulation_results=sim_results)
        ]

        export_path = temp_checkpoint_dir / "results.xlsx"
        processor.export_results(export_path, "excel")

        assert export_path.exists()
        # Verify Excel file has expected sheets
        with pd.ExcelFile(export_path) as xl_file:
            assert "Summary" in xl_file.sheet_names
            assert "Details" in xl_file.sheet_names

    def test_apply_overrides(self, temp_checkpoint_dir):
        """Test applying parameter overrides."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)

        # Create a mock object with attributes
        obj = Mock()
        obj.param1 = "original"
        obj.param2 = 10

        overrides = {"test.param1": "modified", "test.param2": 20, "other.param": "ignored"}

        result = processor._apply_overrides(obj, "test.", overrides)
        assert result.param1 == "modified"
        assert result.param2 == 20

    def test_apply_overrides_none(self, temp_checkpoint_dir):
        """Test applying overrides with None object."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        result = processor._apply_overrides(None, "test.", {"test.param": "value"})
        assert result is None

    def test_apply_overrides_empty(self, temp_checkpoint_dir):
        """Test applying empty overrides."""
        processor = BatchProcessor(checkpoint_dir=temp_checkpoint_dir)
        obj = Mock()
        result = processor._apply_overrides(obj, "test.", {})
        assert result == obj

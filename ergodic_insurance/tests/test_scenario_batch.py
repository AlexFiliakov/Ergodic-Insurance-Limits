"""Comprehensive tests for scenario batch processing framework."""

import copy
import json
from pathlib import Path
import pickle
import tempfile
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
from ergodic_insurance.config import Config
from ergodic_insurance.insurance_program import InsuranceProgram
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import SimulationConfig, SimulationResults
from ergodic_insurance.scenario_manager import (
    ParameterSpec,
    ScenarioConfig,
    ScenarioManager,
    ScenarioType,
)

# Fixtures


@pytest.fixture
def scenario_manager():
    """Create scenario manager instance."""
    return ScenarioManager()


@pytest.fixture
def sample_scenario():
    """Create sample scenario configuration."""
    return ScenarioConfig(
        scenario_id="test_scenario_001",
        name="Test Scenario",
        description="Test scenario for unit tests",
        simulation_config=SimulationConfig(n_simulations=100, n_years=5),
        parameter_overrides={"manufacturer.initial_assets": 15000000},
        tags={"test", "sample"},
        priority=50,
    )


@pytest.fixture
def mock_simulation_results():
    """Create mock simulation results."""
    n_sims = 100
    n_years = 5

    return SimulationResults(
        final_assets=np.random.lognormal(16, 1, n_sims),
        annual_losses=np.random.exponential(100000, (n_sims, n_years)),
        insurance_recoveries=np.random.exponential(50000, (n_sims, n_years)),
        retained_losses=np.random.exponential(50000, (n_sims, n_years)),
        growth_rates=np.random.normal(0.05, 0.02, n_sims),
        ruin_probability=0.02,
        metrics={"var_95": 14000000, "var_99": 12000000, "tvar_95": 13000000, "tvar_99": 11000000},
        convergence={},
        execution_time=5.5,
        config=SimulationConfig(n_simulations=100, n_years=5),
    )


@pytest.fixture
def mock_components():
    """Create mock components for batch processor."""
    loss_generator = Mock(spec=ManufacturingLossGenerator)
    insurance_program = Mock(spec=InsuranceProgram)
    manufacturer = Mock(spec=WidgetManufacturer)
    manufacturer.initial_assets = 10000000

    return loss_generator, insurance_program, manufacturer


# ScenarioConfig Tests


class TestScenarioConfig:
    """Tests for ScenarioConfig class."""

    def test_scenario_creation(self, sample_scenario):
        """Test scenario configuration creation."""
        assert sample_scenario.scenario_id == "test_scenario_001"
        assert sample_scenario.name == "Test Scenario"
        assert sample_scenario.priority == 50
        assert "test" in sample_scenario.tags
        assert sample_scenario.parameter_overrides["manufacturer.initial_assets"] == 15000000

    def test_scenario_id_generation(self):
        """Test automatic scenario ID generation."""
        scenario = ScenarioConfig(
            scenario_id="",
            name="Auto ID Test",
        )
        assert scenario.scenario_id.startswith("scenario_")
        assert len(scenario.scenario_id) > 8

    def test_apply_overrides(self, sample_scenario):
        """Test parameter override application."""
        # Create mock config object
        config = Mock()
        config.manufacturer = Mock()
        config.manufacturer.initial_assets = 10000000

        # Apply overrides
        updated = sample_scenario.apply_overrides(config)

        # Check override was applied
        assert updated.manufacturer.initial_assets == 15000000

    def test_to_dict(self, sample_scenario):
        """Test scenario serialization to dictionary."""
        scenario_dict = sample_scenario.to_dict()

        assert scenario_dict["scenario_id"] == "test_scenario_001"
        assert scenario_dict["name"] == "Test Scenario"
        assert scenario_dict["priority"] == 50
        assert "test" in scenario_dict["tags"]
        assert "created_at" in scenario_dict


# ParameterSpec Tests


class TestParameterSpec:
    """Tests for ParameterSpec class."""

    def test_parameter_spec_creation(self):
        """Test parameter specification creation."""
        spec = ParameterSpec(
            name="manufacturer.initial_assets", values=[10000000, 15000000, 20000000]
        )
        assert spec.name == "manufacturer.initial_assets"
        assert spec.values is not None
        assert len(spec.values) == 3

    def test_generate_values_grid_search(self):
        """Test value generation for grid search."""
        spec = ParameterSpec(name="test_param", values=[1, 2, 3, 4, 5])
        values = spec.generate_values(ScenarioType.GRID_SEARCH)
        assert values == [1, 2, 3, 4, 5]

    def test_generate_values_random_search(self):
        """Test value generation for random search."""
        spec = ParameterSpec(
            name="test_param", min_value=0.0, max_value=1.0, n_samples=10, distribution="uniform"
        )
        np.random.seed(42)
        values = spec.generate_values(ScenarioType.RANDOM_SEARCH)
        assert len(values) == 10
        assert all(0 <= v <= 1 for v in values)

    def test_generate_values_log_distribution(self):
        """Test value generation with log distribution."""
        spec = ParameterSpec(
            name="test_param", min_value=1.0, max_value=100.0, n_samples=10, distribution="log"
        )
        np.random.seed(42)
        values = spec.generate_values(ScenarioType.RANDOM_SEARCH)
        assert len(values) == 10
        assert all(1 <= v <= 100 for v in values)
        # Log distribution should have more values in lower range
        median = np.median(values)
        assert median < 50  # Median should be less than arithmetic middle

    def test_generate_values_sensitivity(self):
        """Test value generation for sensitivity analysis."""
        spec = ParameterSpec(name="test_param", base_value=100, variation_pct=0.2)
        values = spec.generate_values(ScenarioType.SENSITIVITY)
        assert len(values) == 3
        assert values[0] == 80  # -20%
        assert values[1] == 100  # base
        assert values[2] == 120  # +20%


# ScenarioManager Tests


class TestScenarioManager:
    """Tests for ScenarioManager class."""

    def test_create_scenario(self, scenario_manager):
        """Test single scenario creation."""
        scenario = scenario_manager.create_scenario(
            name="Test Scenario", description="A test scenario", priority=10
        )

        assert scenario.name == "Test Scenario"
        assert scenario.priority == 10
        assert len(scenario_manager.scenarios) == 1
        assert scenario.scenario_id in scenario_manager.scenario_index

    def test_add_duplicate_scenario(self, scenario_manager, sample_scenario):
        """Test duplicate scenario handling."""
        scenario_manager.add_scenario(sample_scenario)
        scenario_manager.add_scenario(sample_scenario)  # Add same scenario again

        # Should not add duplicate
        assert len(scenario_manager.scenarios) == 1

    def test_create_grid_search(self, scenario_manager):
        """Test grid search scenario generation."""
        param_specs = [
            ParameterSpec(name="param1", values=[1, 2, 3]),
            ParameterSpec(name="param2", values=[10, 20]),
        ]

        scenarios = scenario_manager.create_grid_search(
            name_template="grid_{index}", parameter_specs=param_specs, tags={"grid", "test"}
        )

        # Should create 3 * 2 = 6 scenarios
        assert len(scenarios) == 6
        assert all("grid_search" in s.tags for s in scenarios)
        assert all("test" in s.tags for s in scenarios)

        # Check all combinations are present
        combinations = [
            (s.parameter_overrides["param1"], s.parameter_overrides["param2"]) for s in scenarios
        ]
        expected = [(1, 10), (1, 20), (2, 10), (2, 20), (3, 10), (3, 20)]
        assert set(combinations) == set(expected)

    def test_create_random_search(self, scenario_manager):
        """Test random search scenario generation."""
        param_specs = [
            ParameterSpec(name="param1", min_value=0, max_value=100, n_samples=50),
            ParameterSpec(name="param2", min_value=0.1, max_value=1.0, n_samples=50),
        ]

        scenarios = scenario_manager.create_random_search(
            name_template="random_{index}", parameter_specs=param_specs, n_scenarios=20, seed=42
        )

        assert len(scenarios) == 20
        assert all("random_search" in s.tags for s in scenarios)

        # Check parameter ranges
        for scenario in scenarios:
            assert 0 <= scenario.parameter_overrides["param1"] <= 100
            assert 0.1 <= scenario.parameter_overrides["param2"] <= 1.0

    def test_create_sensitivity_analysis(self, scenario_manager):
        """Test sensitivity analysis scenario generation."""
        param_specs = [
            ParameterSpec(name="param1", base_value=100, variation_pct=0.1),
            ParameterSpec(name="param2", base_value=50, variation_pct=0.2),
        ]

        scenarios = scenario_manager.create_sensitivity_analysis(
            base_name="sensitivity", parameter_specs=param_specs
        )

        # Should create 1 baseline + 2 params * 2 directions = 5 scenarios
        assert len(scenarios) == 5

        # Check for baseline
        baseline = [s for s in scenarios if "baseline" in s.tags]
        assert len(baseline) == 1
        assert baseline[0].parameter_overrides == {}

        # Check for high/low variations
        high_scenarios = [s for s in scenarios if "high" in s.tags]
        low_scenarios = [s for s in scenarios if "low" in s.tags]
        assert len(high_scenarios) == 2
        assert len(low_scenarios) == 2

    def test_get_scenarios_by_tag(self, scenario_manager):
        """Test filtering scenarios by tag."""
        scenario_manager.create_scenario(name="S1", tags={"tag1", "tag2"})
        scenario_manager.create_scenario(name="S2", tags={"tag1"})
        scenario_manager.create_scenario(name="S3", tags={"tag2"})

        tag1_scenarios = scenario_manager.get_scenarios_by_tag("tag1")
        assert len(tag1_scenarios) == 2

        tag2_scenarios = scenario_manager.get_scenarios_by_tag("tag2")
        assert len(tag2_scenarios) == 2

        tag3_scenarios = scenario_manager.get_scenarios_by_tag("tag3")
        assert len(tag3_scenarios) == 0

    def test_get_scenarios_by_priority(self, scenario_manager):
        """Test filtering scenarios by priority."""
        scenario_manager.create_scenario(name="S1", priority=10)
        scenario_manager.create_scenario(name="S2", priority=50)
        scenario_manager.create_scenario(name="S3", priority=100)

        high_priority = scenario_manager.get_scenarios_by_priority(50)
        assert len(high_priority) == 2
        assert high_priority[0].priority == 10
        assert high_priority[1].priority == 50

    def test_clear_scenarios(self, scenario_manager):
        """Test clearing all scenarios."""
        scenario_manager.create_scenario(name="S1")
        scenario_manager.create_scenario(name="S2")
        assert len(scenario_manager.scenarios) == 2

        scenario_manager.clear_scenarios()
        assert len(scenario_manager.scenarios) == 0
        assert len(scenario_manager.scenario_index) == 0


# BatchProcessor Tests


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_batch_processor_initialization(self, mock_components):
        """Test batch processor initialization."""
        loss_gen, insurance, manufacturer = mock_components

        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            n_workers=4,
        )

        assert processor.loss_generator == loss_gen
        assert processor.insurance_program == insurance
        assert processor.manufacturer == manufacturer
        assert processor.n_workers == 4
        assert processor.checkpoint_dir.exists()

    @patch("ergodic_insurance.src.batch_processor.time")
    @patch("ergodic_insurance.src.batch_processor.MonteCarloEngine")
    def test_process_single_scenario(
        self,
        mock_engine_class,
        mock_time,
        mock_components,
        sample_scenario,
        mock_simulation_results,
    ):
        """Test processing a single scenario."""
        loss_gen, insurance, manufacturer = mock_components

        # Setup mock Monte Carlo engine
        mock_engine = Mock()
        mock_engine.run.return_value = mock_simulation_results
        mock_engine_class.return_value = mock_engine

        # Mock time.time() to return different values
        mock_time.time.side_effect = [100.0, 101.5]  # Start time, end time

        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            use_parallel=False,
        )

        # Process scenario
        result = processor._process_scenario(sample_scenario)

        assert result.scenario_id == sample_scenario.scenario_id
        assert result.status == ProcessingStatus.COMPLETED
        assert result.simulation_results == mock_simulation_results
        assert result.execution_time == 1.5  # 101.5 - 100.0

    @patch("ergodic_insurance.src.batch_processor.MonteCarloEngine")
    def test_process_batch_serial(
        self, mock_engine_class, mock_components, scenario_manager, mock_simulation_results
    ):
        """Test batch processing in serial mode."""
        loss_gen, insurance, manufacturer = mock_components

        # Setup mock Monte Carlo engine
        mock_engine = Mock()
        mock_engine.run.return_value = mock_simulation_results
        mock_engine_class.return_value = mock_engine

        # Create test scenarios
        scenarios = [
            scenario_manager.create_scenario(name=f"Scenario {i}", priority=i) for i in range(3)
        ]

        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            use_parallel=False,
            progress_bar=False,
        )

        # Process batch
        results = processor.process_batch(scenarios, resume_from_checkpoint=False)

        assert isinstance(results, AggregatedResults)
        assert len(results.batch_results) == 3
        assert all(r.status == ProcessingStatus.COMPLETED for r in results.batch_results)
        assert not results.summary_statistics.empty
        assert results.execution_summary["completed"] == 3

    @patch("ergodic_insurance.src.batch_processor.ProcessPoolExecutor")
    @patch("ergodic_insurance.src.batch_processor.MonteCarloEngine")
    def test_process_batch_parallel(
        self,
        mock_engine_class,
        mock_executor_class,
        mock_components,
        scenario_manager,
        mock_simulation_results,
    ):
        """Test batch processing in parallel mode."""
        loss_gen, insurance, manufacturer = mock_components

        # Setup mock Monte Carlo engine
        mock_engine = Mock()
        mock_engine.run.return_value = mock_simulation_results
        mock_engine_class.return_value = mock_engine

        # Setup mock executor
        mock_future = Mock()
        mock_future.result.return_value = BatchResult(
            scenario_id="test",
            scenario_name="Test",
            status=ProcessingStatus.COMPLETED,
            simulation_results=mock_simulation_results,
        )

        mock_executor = Mock()
        mock_executor.submit.return_value = mock_future
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)
        mock_executor_class.return_value = mock_executor

        # Create test scenarios
        scenarios = [scenario_manager.create_scenario(name=f"Scenario {i}") for i in range(3)]

        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            use_parallel=True,
            n_workers=2,
            progress_bar=False,
        )

        # Mock as_completed to return futures
        with patch("ergodic_insurance.src.batch_processor.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future] * 3

            # Process batch
            results = processor.process_batch(scenarios, resume_from_checkpoint=False)

            assert isinstance(results, AggregatedResults)
            assert len(results.batch_results) == 3

    def test_checkpoint_save_and_load(self, mock_components):
        """Test checkpoint saving and loading."""
        loss_gen, insurance, manufacturer = mock_components

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = BatchProcessor(
                loss_generator=loss_gen,
                insurance_program=insurance,
                manufacturer=manufacturer,
                checkpoint_dir=Path(tmpdir),
            )

            # Add some completed scenarios
            processor.completed_scenarios = {"scenario_1", "scenario_2"}
            processor.failed_scenarios = {"scenario_3"}
            processor.batch_results = [
                BatchResult(
                    scenario_id="scenario_1",
                    scenario_name="Scenario 1",
                    status=ProcessingStatus.COMPLETED,
                )
            ]

            # Save checkpoint
            processor._save_checkpoint()

            # Clear state
            processor.completed_scenarios.clear()
            processor.failed_scenarios.clear()
            processor.batch_results.clear()

            # Load checkpoint
            loaded = processor._load_checkpoint()

            assert loaded
            assert processor.completed_scenarios == {"scenario_1", "scenario_2"}
            assert processor.failed_scenarios == {"scenario_3"}
            assert len(processor.batch_results) == 1

    def test_aggregated_results_creation(self, mock_components, mock_simulation_results):
        """Test aggregated results creation."""
        loss_gen, insurance, manufacturer = mock_components

        processor = BatchProcessor(
            loss_generator=loss_gen, insurance_program=insurance, manufacturer=manufacturer
        )

        # Add mock results
        processor.batch_results = [
            BatchResult(
                scenario_id=f"scenario_{i}",
                scenario_name=f"Scenario {i}",
                status=ProcessingStatus.COMPLETED,
                simulation_results=mock_simulation_results,
                execution_time=5.0,
            )
            for i in range(3)
        ]

        # Aggregate results
        aggregated = processor._aggregate_results()

        assert isinstance(aggregated, AggregatedResults)
        assert len(aggregated.batch_results) == 3
        assert len(aggregated.summary_statistics) == 3
        assert "scenario" in aggregated.summary_statistics.columns
        assert "ruin_probability" in aggregated.summary_statistics.columns
        assert "mean_growth_rate" in aggregated.summary_statistics.columns

    def test_sensitivity_analysis(self, mock_components, mock_simulation_results):
        """Test sensitivity analysis generation."""
        loss_gen, insurance, manufacturer = mock_components

        processor = BatchProcessor(
            loss_generator=loss_gen, insurance_program=insurance, manufacturer=manufacturer
        )

        # Create baseline and sensitivity results
        baseline_results = copy.deepcopy(mock_simulation_results)
        baseline_results.growth_rates = np.full_like(baseline_results.growth_rates, 0.05)
        baseline_results.ruin_probability = 0.02

        high_results = copy.deepcopy(mock_simulation_results)
        high_results.growth_rates = np.full_like(high_results.growth_rates, 0.06)
        high_results.ruin_probability = 0.015

        processor.batch_results = [
            BatchResult(
                scenario_id="baseline",
                scenario_name="Baseline",
                status=ProcessingStatus.COMPLETED,
                simulation_results=baseline_results,
                metadata={"tags": ["baseline", "sensitivity"]},
            ),
            BatchResult(
                scenario_id="param1_high",
                scenario_name="Param1 High",
                status=ProcessingStatus.COMPLETED,
                simulation_results=high_results,
                metadata={"tags": ["sensitivity", "high"]},
            ),
        ]

        # Perform sensitivity analysis
        sensitivity = processor._perform_sensitivity_analysis()

        assert sensitivity is not None
        assert len(sensitivity) == 1
        assert "growth_rate_change_pct" in sensitivity.columns
        assert "ruin_prob_change_pct" in sensitivity.columns

        # Check calculations
        # pylint: disable-next=unsubscriptable-object
        growth_change = sensitivity["growth_rate_change_pct"].iloc[0]
        assert np.isclose(growth_change, 20.0, rtol=0.01)  # (0.06 - 0.05) / 0.05 * 100

    def test_export_results(self, mock_components, mock_simulation_results):
        """Test exporting results to different formats."""
        loss_gen, insurance, manufacturer = mock_components

        processor = BatchProcessor(
            loss_generator=loss_gen, insurance_program=insurance, manufacturer=manufacturer
        )

        # Add mock results
        processor.batch_results = [
            BatchResult(
                scenario_id="scenario_1",
                scenario_name="Scenario 1",
                status=ProcessingStatus.COMPLETED,
                simulation_results=mock_simulation_results,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV export
            csv_path = Path(tmpdir) / "results.csv"
            processor.export_results(csv_path, export_format="csv")
            assert csv_path.exists()

            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert "scenario_name" in df.columns

            # Test JSON export
            json_path = Path(tmpdir) / "results.json"
            processor.export_results(json_path, export_format="json")
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)
            assert "summary" in data
            assert "execution_summary" in data


# AggregatedResults Tests


class TestAggregatedResults:
    """Tests for AggregatedResults class."""

    def test_get_successful_results(self, mock_simulation_results):
        """Test filtering successful results."""
        batch_results = [
            BatchResult(
                scenario_id="s1",
                scenario_name="S1",
                status=ProcessingStatus.COMPLETED,
                simulation_results=mock_simulation_results,
            ),
            BatchResult(scenario_id="s2", scenario_name="S2", status=ProcessingStatus.FAILED),
            BatchResult(
                scenario_id="s3",
                scenario_name="S3",
                status=ProcessingStatus.COMPLETED,
                simulation_results=mock_simulation_results,
            ),
        ]

        aggregated = AggregatedResults(
            batch_results=batch_results, summary_statistics=pd.DataFrame(), comparison_metrics={}
        )

        successful = aggregated.get_successful_results()
        assert len(successful) == 2
        assert all(r.status == ProcessingStatus.COMPLETED for r in successful)

    def test_to_dataframe(self, mock_simulation_results):
        """Test conversion to DataFrame."""
        batch_results = [
            BatchResult(
                scenario_id=f"s{i}",
                scenario_name=f"Scenario {i}",
                status=ProcessingStatus.COMPLETED,
                simulation_results=mock_simulation_results,
                execution_time=5.0,
            )
            for i in range(3)
        ]

        aggregated = AggregatedResults(
            batch_results=batch_results, summary_statistics=pd.DataFrame(), comparison_metrics={}
        )

        df = aggregated.to_dataframe()
        assert len(df) == 3
        assert "scenario_id" in df.columns
        assert "ruin_probability" in df.columns
        assert "mean_growth_rate" in df.columns
        assert all(df["status"] == "completed")


# Integration Tests


class TestIntegration:
    """Integration tests for the batch processing framework."""

    @patch("ergodic_insurance.src.batch_processor.MonteCarloEngine")
    def test_end_to_end_workflow(self, mock_engine_class, mock_components, mock_simulation_results):
        """Test complete workflow from scenario creation to results."""
        loss_gen, insurance, manufacturer = mock_components

        # Setup mock engine
        mock_engine = Mock()
        mock_engine.run.return_value = mock_simulation_results
        mock_engine_class.return_value = mock_engine

        # Create scenario manager
        manager = ScenarioManager()

        # Create grid search scenarios
        param_specs = [
            ParameterSpec(name="manufacturer.assets", values=[10e6, 15e6]),
            ParameterSpec(name="insurance.premium", values=[0.01, 0.02]),
        ]

        scenarios = manager.create_grid_search(
            name_template="grid_{index}", parameter_specs=param_specs
        )

        # Create batch processor - start fresh without any previous results
        processor = BatchProcessor(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            use_parallel=False,
            progress_bar=False,
        )

        # Clear any existing results from other tests
        processor.batch_results.clear()
        processor.completed_scenarios.clear()

        # Process batch
        results = processor.process_batch(scenarios, resume_from_checkpoint=False)

        # Verify results
        assert isinstance(results, AggregatedResults)
        assert len(results.batch_results) == 4  # 2x2 grid
        assert results.execution_summary["completed"] == 4
        assert not results.summary_statistics.empty

        # Test export
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "results.csv"
            processor.export_results(export_path)
            assert export_path.exists()


# Performance Tests


class TestPerformance:
    """Performance tests for batch processing."""

    def test_large_batch_memory_usage(self, mock_components):
        """Test memory usage with large batch."""
        loss_gen, insurance, manufacturer = mock_components

        processor = BatchProcessor(
            loss_generator=loss_gen, insurance_program=insurance, manufacturer=manufacturer
        )

        # Create many scenarios
        manager = ScenarioManager()
        scenarios = [manager.create_scenario(name=f"Scenario {i}") for i in range(100)]

        # Check that scenarios can be created without memory issues
        assert len(scenarios) == 100
        assert len(manager.scenarios) == 100

    def test_checkpoint_performance(self, mock_components):
        """Test checkpoint save/load performance."""
        loss_gen, insurance, manufacturer = mock_components

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = BatchProcessor(
                loss_generator=loss_gen,
                insurance_program=insurance,
                manufacturer=manufacturer,
                checkpoint_dir=Path(tmpdir),
            )

            # Add many completed scenarios
            for i in range(1000):
                processor.completed_scenarios.add(f"scenario_{i}")

            # Time checkpoint save
            import time

            start = time.time()
            processor._save_checkpoint()
            save_time = time.time() - start

            # Should be fast even with many scenarios
            assert save_time < 1.0  # Less than 1 second

            # Time checkpoint load
            start = time.time()
            processor._load_checkpoint()
            load_time = time.time() - start

            assert load_time < 1.0

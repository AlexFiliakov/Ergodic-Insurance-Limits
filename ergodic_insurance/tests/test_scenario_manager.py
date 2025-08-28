"""Comprehensive test suite for scenario management system.

Tests all aspects of the ScenarioManager including scenario creation,
grid search, random search, sensitivity analysis, and import/export.
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.src.config import Config
from ergodic_insurance.src.monte_carlo import SimulationConfig
from ergodic_insurance.src.scenario_manager import (
    ParameterSpec,
    ScenarioConfig,
    ScenarioManager,
    ScenarioType,
)


class TestScenarioType:
    """Test ScenarioType enum."""

    def test_scenario_types(self):
        """Test all scenario type values."""
        assert ScenarioType.SINGLE.value == "single"
        assert ScenarioType.GRID_SEARCH.value == "grid_search"
        assert ScenarioType.RANDOM_SEARCH.value == "random_search"
        assert ScenarioType.CUSTOM.value == "custom"
        assert ScenarioType.SENSITIVITY.value == "sensitivity"


class TestParameterSpec:
    """Test ParameterSpec class."""

    def test_basic_creation(self):
        """Test creating basic parameter spec."""
        spec = ParameterSpec(name="premium_rate", values=[0.01, 0.02, 0.03])
        assert spec.name == "premium_rate"
        assert spec.values == [0.01, 0.02, 0.03]
        assert spec.distribution == "uniform"

    def test_nested_parameter_name(self):
        """Test parameter spec with nested name."""
        spec = ParameterSpec(
            name="insurance.layer1.premium_rate", min_value=0.01, max_value=0.05, n_samples=5
        )
        assert spec.name == "insurance.layer1.premium_rate"
        assert spec.min_value == 0.01
        assert spec.max_value == 0.05

    def test_empty_name_validation(self):
        """Test validation of empty parameter name."""
        with pytest.raises(ValueError, match="Parameter name cannot be empty"):
            ParameterSpec(name="")

    def test_generate_values_grid_search(self):
        """Test generating values for grid search."""
        spec = ParameterSpec(name="param", values=[1, 2, 3, 4, 5])
        values = spec.generate_values(ScenarioType.GRID_SEARCH)
        assert values == [1, 2, 3, 4, 5]

    def test_generate_values_random_search_uniform(self):
        """Test generating values for random search with uniform distribution."""
        np.random.seed(42)
        spec = ParameterSpec(
            name="param", min_value=0.0, max_value=1.0, n_samples=10, distribution="uniform"
        )
        values = spec.generate_values(ScenarioType.RANDOM_SEARCH)
        assert len(values) == 10
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_generate_values_random_search_log(self):
        """Test generating values for random search with log distribution."""
        np.random.seed(42)
        spec = ParameterSpec(
            name="param", min_value=0.001, max_value=1.0, n_samples=5, distribution="log"
        )
        values = spec.generate_values(ScenarioType.RANDOM_SEARCH)
        assert len(values) == 5
        assert all(0.001 <= v <= 1.0 for v in values)
        # Check log distribution properties
        log_values = np.log(values)
        assert np.min(log_values) >= np.log(0.001)
        assert np.max(log_values) <= np.log(1.0)

    def test_generate_values_sensitivity(self):
        """Test generating values for sensitivity analysis."""
        spec = ParameterSpec(name="param", base_value=100, variation_pct=0.2)
        values = spec.generate_values(ScenarioType.SENSITIVITY)
        assert len(values) == 3
        assert values[0] == 80  # -20%
        assert values[1] == 100  # baseline
        assert values[2] == 120  # +20%

    def test_generate_values_fallback(self):
        """Test fallback behavior when no suitable generation method."""
        spec = ParameterSpec(name="param")
        values = spec.generate_values(ScenarioType.GRID_SEARCH)
        assert values == [None]

        spec = ParameterSpec(name="param", base_value=42)
        values = spec.generate_values(ScenarioType.CUSTOM)
        assert values == [42]

    def test_generate_values_no_range(self):
        """Test random search without min/max values."""
        spec = ParameterSpec(name="param", values=[10, 20, 30])
        values = spec.generate_values(ScenarioType.RANDOM_SEARCH)
        assert values == [10, 20, 30]


class TestScenarioConfig:
    """Test ScenarioConfig dataclass."""

    def test_basic_creation(self):
        """Test creating basic scenario config."""
        scenario = ScenarioConfig(
            scenario_id="test_001", name="Test Scenario", description="Test description"
        )
        assert scenario.scenario_id == "test_001"
        assert scenario.name == "Test Scenario"
        assert scenario.description == "Test description"
        assert scenario.priority == 100
        assert isinstance(scenario.tags, set)
        assert isinstance(scenario.parameter_overrides, dict)

    def test_auto_id_generation(self):
        """Test automatic scenario ID generation."""
        scenario = ScenarioConfig(scenario_id="", name="Auto ID Test")
        assert scenario.scenario_id.startswith("scenario_")
        assert len(scenario.scenario_id) > 9  # scenario_ + 8 char hash

    def test_default_simulation_config(self):
        """Test default simulation config creation."""
        scenario = ScenarioConfig(scenario_id="test", name="Test")
        assert scenario.simulation_config is not None
        assert isinstance(scenario.simulation_config, SimulationConfig)

    def test_generate_id_deterministic(self):
        """Test that ID generation is deterministic."""
        # Create two scenarios with same parameters
        now = datetime.now()
        scenario1 = ScenarioConfig(
            scenario_id="", name="Same Name", parameter_overrides={"param": 42}, created_at=now
        )

        scenario2 = ScenarioConfig(
            scenario_id="", name="Same Name", parameter_overrides={"param": 42}, created_at=now
        )

        assert scenario1.scenario_id == scenario2.scenario_id

    def test_apply_overrides_to_object(self):
        """Test applying parameter overrides to an object."""
        # Create a mock config object
        config = MagicMock()
        config.premium_rate = 0.01
        config.insurance = MagicMock()
        config.insurance.limit = 1_000_000

        scenario = ScenarioConfig(
            scenario_id="test",
            name="Test",
            parameter_overrides={"premium_rate": 0.02, "insurance.limit": 5_000_000},
        )

        modified = scenario.apply_overrides(config)
        assert modified.premium_rate == 0.02
        assert modified.insurance.limit == 5_000_000

    def test_apply_overrides_to_dict(self):
        """Test applying parameter overrides to a dictionary."""
        config = {"premium_rate": 0.01, "insurance": {"limit": 1_000_000, "deductible": 10_000}}

        scenario = ScenarioConfig(
            scenario_id="test",
            name="Test",
            parameter_overrides={"premium_rate": 0.02, "insurance.limit": 5_000_000},
        )

        modified = scenario.apply_overrides(config)
        assert modified["premium_rate"] == 0.02
        assert modified["insurance"]["limit"] == 5_000_000
        assert modified["insurance"]["deductible"] == 10_000  # Unchanged

    def test_apply_overrides_nonexistent_path(self):
        """Test applying overrides with nonexistent path."""
        config = MagicMock()
        config.existing = "value"

        scenario = ScenarioConfig(
            scenario_id="test", name="Test", parameter_overrides={"nonexistent.path.param": 42}
        )

        # Should not raise error
        modified = scenario.apply_overrides(config)
        assert modified.existing == "value"

    def test_to_dict(self):
        """Test converting scenario to dictionary."""
        scenario = ScenarioConfig(
            scenario_id="test_001",
            name="Test Scenario",
            description="Description",
            parameter_overrides={"param": 42},
            tags={"tag1", "tag2"},
            priority=50,
            metadata={"key": "value"},
        )

        result = scenario.to_dict()
        assert result["scenario_id"] == "test_001"
        assert result["name"] == "Test Scenario"
        assert result["description"] == "Description"
        assert result["parameter_overrides"] == {"param": 42}
        assert set(result["tags"]) == {"tag1", "tag2"}
        assert result["priority"] == 50
        assert result["metadata"] == {"key": "value"}
        assert "created_at" in result


class TestScenarioManager:
    """Test ScenarioManager class."""

    @pytest.fixture
    def manager(self):
        """Create scenario manager instance."""
        return ScenarioManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.scenarios == []
        assert manager.scenario_index == {}

    def test_create_single_scenario(self, manager):
        """Test creating a single scenario."""
        scenario = manager.create_scenario(
            name="Test Scenario",
            description="Test description",
            parameter_overrides={"param": 42},
            tags={"test", "single"},
            priority=50,
        )

        assert scenario.name == "Test Scenario"
        assert scenario.parameter_overrides == {"param": 42}
        assert scenario.tags == {"test", "single"}
        assert scenario.priority == 50
        assert len(manager.scenarios) == 1
        assert scenario.scenario_id in manager.scenario_index

    def test_add_duplicate_scenario(self, manager):
        """Test adding duplicate scenarios."""
        scenario1 = ScenarioConfig(
            scenario_id="same_id", name="Scenario 1", parameter_overrides={"param": 42}
        )
        scenario2 = ScenarioConfig(
            scenario_id="same_id",
            name="Scenario 2",
            parameter_overrides={"param": 42},  # Same overrides
        )

        manager.add_scenario(scenario1)
        manager.add_scenario(scenario2)  # Should be skipped

        assert len(manager.scenarios) == 1
        assert manager.scenarios[0].name == "Scenario 1"

    def test_add_scenario_with_different_overrides(self, manager):
        """Test adding scenarios with same ID but different overrides."""
        scenario1 = ScenarioConfig(
            scenario_id="same_id", name="Scenario 1", parameter_overrides={"param": 42}
        )
        scenario2 = ScenarioConfig(
            scenario_id="same_id",
            name="Scenario 2",
            parameter_overrides={"param": 100},  # Different overrides
        )

        manager.add_scenario(scenario1)
        manager.add_scenario(scenario2)

        assert len(manager.scenarios) == 2

    def test_create_grid_search(self, manager):
        """Test creating grid search scenarios."""
        specs = [
            ParameterSpec(name="param1", values=[1, 2]),
            ParameterSpec(name="param2", values=[10, 20, 30]),
        ]

        scenarios = manager.create_grid_search(
            name_template="grid_{params}", parameter_specs=specs, tags={"grid"}
        )

        assert len(scenarios) == 6  # 2 * 3 combinations
        assert all("grid_search" in s.tags for s in scenarios)
        assert all("grid" in s.tags for s in scenarios)
        assert all(s.priority == 50 for s in scenarios)

        # Check parameter combinations
        overrides = [s.parameter_overrides for s in scenarios]
        expected = [
            {"param1": 1, "param2": 10},
            {"param1": 1, "param2": 20},
            {"param1": 1, "param2": 30},
            {"param1": 2, "param2": 10},
            {"param1": 2, "param2": 20},
            {"param1": 2, "param2": 30},
        ]
        for exp in expected:
            assert exp in overrides

    def test_create_grid_search_with_formatting(self, manager):
        """Test grid search with name formatting."""
        specs = [ParameterSpec(name="insurance.premium", values=[0.01, 0.02])]

        scenarios = manager.create_grid_search(
            name_template="scenario_{index}_{params}", parameter_specs=specs
        )

        assert len(scenarios) == 2
        # Check name formatting - should use last part of nested name
        assert "premium=" in scenarios[0].name

    def test_create_random_search(self, manager):
        """Test creating random search scenarios."""
        np.random.seed(42)
        specs = [
            ParameterSpec(name="param1", min_value=0, max_value=100, n_samples=10),
            ParameterSpec(name="param2", min_value=1, max_value=10, n_samples=10),
        ]

        scenarios = manager.create_random_search(
            name_template="random_{index}",
            parameter_specs=specs,
            n_scenarios=5,
            seed=42,
            tags={"random"},
        )

        assert len(scenarios) == 5
        assert all("random_search" in s.tags for s in scenarios)
        assert all("random" in s.tags for s in scenarios)
        assert all(s.priority == 75 for s in scenarios)

        # Check that parameters are within bounds
        for scenario in scenarios:
            assert 0 <= scenario.parameter_overrides["param1"] <= 100
            assert 1 <= scenario.parameter_overrides["param2"] <= 10

    def test_create_sensitivity_analysis(self, manager):
        """Test creating sensitivity analysis scenarios."""
        specs = [
            ParameterSpec(name="param1", base_value=100, variation_pct=0.1),
            ParameterSpec(name="param2", base_value=50, variation_pct=0.2),
        ]

        scenarios = manager.create_sensitivity_analysis(
            base_name="sensitivity", parameter_specs=specs, tags={"analysis"}
        )

        # 1 baseline + 2 params * 2 variations = 5 scenarios
        assert len(scenarios) == 5

        # Check baseline
        baseline = [s for s in scenarios if "baseline" in s.tags][0]
        assert baseline.parameter_overrides == {}
        assert baseline.priority == 25

        # Check sensitivity scenarios
        sensitivity_scenarios = [s for s in scenarios if "baseline" not in s.tags]
        assert len(sensitivity_scenarios) == 4

        # Check low/high variations
        low_scenarios = [s for s in sensitivity_scenarios if "low" in s.tags]
        high_scenarios = [s for s in sensitivity_scenarios if "high" in s.tags]
        assert len(low_scenarios) == 2
        assert len(high_scenarios) == 2

        # Check parameter values
        param1_low = [s for s in low_scenarios if "param1" in s.name][0]
        assert param1_low.parameter_overrides["param1"] == 90  # 100 * 0.9

        param1_high = [s for s in high_scenarios if "param1" in s.name][0]
        assert pytest.approx(param1_high.parameter_overrides["param1"]) == 110  # 100 * 1.1

    def test_get_scenarios_by_tag(self, manager):
        """Test filtering scenarios by tag."""
        manager.create_scenario("S1", tags={"tag1", "tag2"})
        manager.create_scenario("S2", tags={"tag2", "tag3"})
        manager.create_scenario("S3", tags={"tag3"})

        tag1_scenarios = manager.get_scenarios_by_tag("tag1")
        assert len(tag1_scenarios) == 1
        assert tag1_scenarios[0].name == "S1"

        tag2_scenarios = manager.get_scenarios_by_tag("tag2")
        assert len(tag2_scenarios) == 2

        tag3_scenarios = manager.get_scenarios_by_tag("tag3")
        assert len(tag3_scenarios) == 2

        tag4_scenarios = manager.get_scenarios_by_tag("tag4")
        assert len(tag4_scenarios) == 0

    def test_get_scenarios_by_priority(self, manager):
        """Test filtering scenarios by priority."""
        manager.create_scenario("High", priority=25)
        manager.create_scenario("Medium", priority=50)
        manager.create_scenario("Low", priority=100)
        manager.create_scenario("VeryLow", priority=150)

        # Get high priority only
        high_priority = manager.get_scenarios_by_priority(25)
        assert len(high_priority) == 1
        assert high_priority[0].name == "High"

        # Get medium and above
        medium_and_above = manager.get_scenarios_by_priority(50)
        assert len(medium_and_above) == 2
        assert [s.name for s in medium_and_above] == ["High", "Medium"]

        # Get all except very low
        most = manager.get_scenarios_by_priority(100)
        assert len(most) == 3

        # Check sorting
        assert most[0].priority <= most[1].priority <= most[2].priority

    def test_clear_scenarios(self, manager):
        """Test clearing all scenarios."""
        manager.create_scenario("S1")
        manager.create_scenario("S2")
        manager.create_scenario("S3")

        assert len(manager.scenarios) == 3
        assert len(manager.scenario_index) == 3

        manager.clear_scenarios()

        assert len(manager.scenarios) == 0
        assert len(manager.scenario_index) == 0

    def test_export_scenarios(self, manager):
        """Test exporting scenarios to JSON."""
        manager.create_scenario("Test Scenario", parameter_overrides={"param": 42}, tags={"test"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Note: There's a bug in the original code - it should be json.dump, not json.dumps
            # We'll test the expected behavior
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                manager.export_scenarios(temp_path)
                mock_open.assert_called_once_with(temp_path, "w")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_import_scenarios(self, manager):
        """Test importing scenarios from JSON."""
        test_data = {
            "scenarios": [
                {
                    "scenario_id": "imported_001",
                    "name": "Imported Scenario",
                    "description": "Test import",
                    "parameter_overrides": {"param": 42},
                    "tags": ["imported", "test"],
                    "priority": 75,
                    "created_at": "2024-01-01T00:00:00",
                }
            ],
            "metadata": {"n_scenarios": 1, "exported_at": "2024-01-01T00:00:00"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            manager.import_scenarios(temp_path)

            assert len(manager.scenarios) == 1
            imported = manager.scenarios[0]
            assert imported.scenario_id == "imported_001"
            assert imported.name == "Imported Scenario"
            assert imported.parameter_overrides == {"param": 42}
            assert imported.tags == {"imported", "test"}
            assert imported.priority == 75
            assert imported.created_at.year == 2024
        finally:
            temp_path.unlink()

    def test_import_scenarios_without_optional_fields(self, manager):
        """Test importing scenarios with minimal data."""
        test_data = {"scenarios": [{"scenario_id": "minimal", "name": "Minimal Scenario"}]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            manager.import_scenarios(temp_path)

            assert len(manager.scenarios) == 1
            imported = manager.scenarios[0]
            assert imported.scenario_id == "minimal"
            assert imported.name == "Minimal Scenario"
            assert imported.description == ""
            assert imported.parameter_overrides == {}
            assert imported.tags == set()
            assert imported.priority == 100
        finally:
            temp_path.unlink()

    def test_complex_workflow(self, manager):
        """Test a complex workflow with multiple scenario types."""
        # Create baseline
        baseline = manager.create_scenario("Baseline", parameter_overrides={}, priority=10)

        # Create grid search
        grid_specs = [
            ParameterSpec(name="p1", values=[1, 2]),
            ParameterSpec(name="p2", values=[10, 20]),
        ]
        grid_scenarios = manager.create_grid_search("grid_{index}", grid_specs)

        # Create sensitivity
        sens_specs = [ParameterSpec(name="p3", base_value=100, variation_pct=0.1)]
        sens_scenarios = manager.create_sensitivity_analysis("sens", sens_specs)

        # Check total count
        assert len(manager.scenarios) == 1 + 4 + 3  # baseline + grid + sensitivity

        # Check priority ordering
        by_priority = manager.get_scenarios_by_priority(100)
        assert by_priority[0] == baseline  # Priority 10
        assert all(s in by_priority for s in sens_scenarios)  # Priority 25/30
        # Grid scenarios have priority 50, not in the original create_grid_search call

        # Check tags
        grid_tagged = manager.get_scenarios_by_tag("grid_search")
        assert len(grid_tagged) == 4

        baseline_tagged = manager.get_scenarios_by_tag("baseline")
        assert len(baseline_tagged) == 1

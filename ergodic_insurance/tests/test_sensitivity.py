"""Comprehensive tests for sensitivity analysis module.

This module tests all functionality of the sensitivity analysis tools,
including parameter variations, impact calculations, caching, and
result structures.

Author: Alex Filiakov
Date: 2025-01-29
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    TwoWaySensitivityResult,
)


@dataclass
class MockOptimizationResult:
    """Mock optimization result for testing."""

    @dataclass
    class Strategy:
        expected_roe: float
        bankruptcy_risk: float
        growth_rate: float
        capital_efficiency: float
        deductible: float
        premium_rate: float

    def __init__(self, config: Dict[str, Any]):
        """Create mock result based on config."""
        # Create deterministic but varying results based on config
        base_value = sum(
            int(hashlib.sha256(str(k).encode()).hexdigest(), 16) % (10**9) * v
            for k, v in config.items()
            if isinstance(v, (int, float))
        )
        base_value = abs(base_value) % 100

        self.optimal_strategy = self.Strategy(
            expected_roe=0.1 + (base_value % 10) * 0.01,
            bankruptcy_risk=0.01 + (base_value % 5) * 0.002,
            growth_rate=0.05 + (base_value % 8) * 0.005,
            capital_efficiency=0.8 + (base_value % 4) * 0.05,
            deductible=100000 + base_value * 1000,
            premium_rate=0.02 + (base_value % 3) * 0.005,
        )


class MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self.call_count = 0
        self.configs_seen = []

    def optimize(self, config: Dict[str, Any]) -> MockOptimizationResult:
        """Mock optimization method."""
        self.call_count += 1
        self.configs_seen.append(config.copy())
        return MockOptimizationResult(config)


class TestSensitivityResult:
    """Test SensitivityResult class."""

    def test_initialization(self):
        """Test SensitivityResult initialization."""
        result = SensitivityResult(
            parameter="test_param",
            baseline_value=10.0,
            variations=np.array([8, 9, 10, 11, 12]),
            metrics={
                "metric1": np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
                "metric2": np.array([1.6, 1.8, 2.0, 2.2, 2.4]),
            },
        )

        assert result.parameter == "test_param"
        assert result.baseline_value == 10.0
        assert len(result.variations) == 5
        assert "metric1" in result.metrics
        assert "metric2" in result.metrics

    def test_calculate_impact(self):
        """Test point elasticity calculation."""
        result = SensitivityResult(
            parameter="test_param",
            baseline_value=10.0,
            variations=np.array([7, 8.5, 10, 11.5, 13]),  # ±30% range
            metrics={
                "linear": np.array([0.7, 0.85, 1.0, 1.15, 1.3]),  # Linear relationship
                "quadratic": np.array([0.49, 0.72, 1.0, 1.32, 1.69]),  # Quadratic
                "constant": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # No sensitivity
            },
        )

        # Linear impact should be ~1 (elasticity of 1)
        linear_impact = result.calculate_impact("linear")
        assert abs(linear_impact - 1.0) < 0.01

        # Constant should have zero impact
        constant_impact = result.calculate_impact("constant")
        assert constant_impact == 0.0

        # Quadratic: central diff at baseline → dM/dP = (1.32 - 0.72)/(11.5 - 8.5) = 0.2
        # elasticity = 0.2 * (10.0 / 1.0) = 2.0
        quadratic_impact = result.calculate_impact("quadratic")
        assert quadratic_impact > linear_impact

    def test_calculate_impact_sign(self):
        """Test that point elasticity preserves sign (issue #1334)."""
        # M = 2*P - 5 at P=5 → M=5, dM/dP=2, elasticity = 2*(5/5) = 2.0
        result_pos = SensitivityResult(
            parameter="param",
            baseline_value=5.0,
            variations=np.array([4.0, 5.0, 6.0]),
            metrics={"m": np.array([3.0, 5.0, 7.0])},
        )
        assert result_pos.calculate_impact("m") == pytest.approx(2.0, abs=1e-9)

        # M = -3*P + 25 at P=5 → M=10, dM/dP=-3, elasticity = -3*(5/10) = -1.5
        result_neg = SensitivityResult(
            parameter="param",
            baseline_value=5.0,
            variations=np.array([4.0, 5.0, 6.0]),
            metrics={"m": np.array([13.0, 10.0, 7.0])},
        )
        assert result_neg.calculate_impact("m") == pytest.approx(-1.5, abs=1e-9)

    def test_calculate_impact_edge_cases(self):
        """Test impact calculation edge cases."""
        result = SensitivityResult(
            parameter="test_param",
            baseline_value=10.0,
            variations=np.array([10, 10, 10]),  # No variation
            metrics={"metric": np.array([1.0, 1.0, 1.0])},
        )

        # No variation should give zero impact
        assert result.calculate_impact("metric") == 0.0

        # Test with zero baseline
        result.baseline_value = 0.0
        assert result.calculate_impact("metric") == 0.0

        # Test with missing metric
        with pytest.raises(KeyError):
            result.calculate_impact("nonexistent")

    def test_get_metric_bounds(self):
        """Test getting metric bounds."""
        result = SensitivityResult(
            parameter="test_param",
            baseline_value=10.0,
            variations=np.array([8, 9, 10, 11, 12]),
            metrics={"metric": np.array([0.5, 0.7, 1.0, 1.3, 1.8])},
        )

        min_val, max_val = result.get_metric_bounds("metric")
        assert min_val == 0.5
        assert max_val == 1.8

        # Test with missing metric
        with pytest.raises(KeyError):
            result.get_metric_bounds("nonexistent")

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        result = SensitivityResult(
            parameter="test_param",
            baseline_value=10.0,
            variations=np.array([8, 10, 12]),
            metrics={"metric1": np.array([0.8, 1.0, 1.2]), "metric2": np.array([1.6, 2.0, 2.4])},
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "parameter_value" in df.columns
        assert "metric1" in df.columns
        assert "metric2" in df.columns
        assert df["parameter_value"].tolist() == [8, 10, 12]


class TestTwoWaySensitivityResult:
    """Test TwoWaySensitivityResult class."""

    def test_initialization(self):
        """Test TwoWaySensitivityResult initialization."""
        result = TwoWaySensitivityResult(
            parameter1="param1",
            parameter2="param2",
            values1=np.array([1, 2, 3]),
            values2=np.array([10, 20]),
            metric_grid=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            metric_name="test_metric",
        )

        assert result.parameter1 == "param1"
        assert result.parameter2 == "param2"
        assert result.metric_grid.shape == (3, 2)

    def test_find_optimal_region(self):
        """Test finding optimal parameter regions."""
        result = TwoWaySensitivityResult(
            parameter1="param1",
            parameter2="param2",
            values1=np.array([1, 2, 3]),
            values2=np.array([10, 20]),
            metric_grid=np.array([[0.1, 0.2], [0.15, 0.25], [0.3, 0.4]]),
            metric_name="metric",
        )

        # Find regions close to 0.15
        mask = result.find_optimal_region(0.15, tolerance=0.1)

        assert mask.shape == (3, 2)
        assert mask[1, 0]  # 0.15 should be True
        assert not mask[2, 1]  # 0.4 should be False

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        result = TwoWaySensitivityResult(
            parameter1="param1",
            parameter2="param2",
            values1=np.array([1, 2]),
            values2=np.array([10, 20]),
            metric_grid=np.array([[0.1, 0.2], [0.3, 0.4]]),
            metric_name="metric",
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2x2 grid
        assert "param1" in df.columns
        assert "param2" in df.columns
        assert "metric" in df.columns


class TestSensitivityAnalyzer:
    """Test SensitivityAnalyzer class."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        return {
            "frequency": 5.0,
            "severity_mean": 100000,
            "premium_rate": 0.02,
            "retention": 50000,
            "manufacturer": {"base_operating_margin": 0.08, "tax_rate": 0.25},
        }

    @pytest.fixture
    def analyzer(self, base_config):
        """Create analyzer with mock optimizer."""
        optimizer = MockOptimizer()
        return SensitivityAnalyzer(base_config, optimizer)

    def test_initialization(self, base_config):
        """Test analyzer initialization."""
        optimizer = MockOptimizer()
        analyzer = SensitivityAnalyzer(base_config, optimizer)

        assert analyzer.base_config == base_config
        assert analyzer.optimizer == optimizer
        assert len(analyzer.results_cache) == 0

    def test_cache_key_generation(self, analyzer):
        """Test cache key generation."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Same as config1, different order
        config3 = {"a": 1, "b": 3}  # Different from config1

        key1 = analyzer._get_cache_key(config1)
        key2 = analyzer._get_cache_key(config2)
        key3 = analyzer._get_cache_key(config3)

        # Same configs should have same key
        assert key1 == key2
        # Different configs should have different keys
        assert key1 != key3

    def test_analyze_parameter_basic(self, analyzer):
        """Test basic parameter analysis."""
        result = analyzer.analyze_parameter("frequency", param_range=(3, 7), n_points=5)

        assert result.parameter == "frequency"
        assert result.baseline_value == 5.0
        assert len(result.variations) == 5
        assert result.variations[0] == 3
        assert result.variations[-1] == 7

        # Check that all metrics are present
        for metric in ["optimal_roe", "bankruptcy_risk", "growth_rate"]:
            assert metric in result.metrics
            assert len(result.metrics[metric]) == 5

    def test_analyze_parameter_nested(self, analyzer):
        """Test analysis of nested parameters."""
        result = analyzer.analyze_parameter(
            "base_operating_margin",
            param_path="manufacturer.base_operating_margin",
            param_range=(0.06, 0.10),
            n_points=3,
        )

        assert result.parameter == "base_operating_margin"
        assert result.baseline_value == 0.08
        assert result.parameter_path == "manufacturer.base_operating_margin"

    def test_analyze_parameter_relative_range(self, analyzer):
        """Test using relative range."""
        result = analyzer.analyze_parameter("severity_mean", relative_range=0.2, n_points=3)  # ±20%

        assert result.baseline_value == 100000
        assert result.variations[0] == pytest.approx(80000)
        assert result.variations[-1] == pytest.approx(120000)

    def test_analyze_parameter_missing(self, analyzer):
        """Test analyzing non-existent parameter."""
        with pytest.raises(KeyError):
            analyzer.analyze_parameter("nonexistent_param")

    def test_caching_mechanism(self, analyzer):
        """Test that caching reduces optimizer calls."""
        # First analysis
        result1 = analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

        initial_calls = analyzer.optimizer.call_count

        # Same analysis again - should use cache
        result2 = analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

        # Optimizer shouldn't be called again
        assert analyzer.optimizer.call_count == initial_calls

        # Results should be identical
        np.testing.assert_array_equal(
            result1.metrics["optimal_roe"], result2.metrics["optimal_roe"]
        )

    def test_persistent_caching(self, base_config):
        """Test persistent caching to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create analyzer with cache directory
            optimizer1 = MockOptimizer()
            analyzer1 = SensitivityAnalyzer(base_config, optimizer1, cache_dir)

            # Run analysis
            result1 = analyzer1.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

            # Create new analyzer with same cache directory
            optimizer2 = MockOptimizer()
            analyzer2 = SensitivityAnalyzer(base_config, optimizer2, cache_dir)

            # Run same analysis - should use persistent cache
            result2 = analyzer2.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

            # Second optimizer shouldn't be called
            assert optimizer2.call_count == 0

            # Results should match
            np.testing.assert_array_equal(
                result1.metrics["optimal_roe"], result2.metrics["optimal_roe"]
            )

    def test_create_tornado_diagram(self, analyzer):
        """Test tornado diagram generation."""
        parameters = ["frequency", "severity_mean", "premium_rate"]

        tornado_data = analyzer.create_tornado_diagram(
            parameters, metric="optimal_roe", relative_range=0.2, n_points=3
        )

        assert isinstance(tornado_data, pd.DataFrame)
        assert len(tornado_data) <= len(parameters)

        # Check required columns
        required_cols = [
            "parameter",
            "impact",
            "direction",
            "low_value",
            "high_value",
            "baseline",
            "baseline_param",
        ]
        for col in required_cols:
            assert col in tornado_data.columns

        # Should be sorted by impact
        impacts = tornado_data["impact"].values
        assert all(impacts[i] >= impacts[i + 1] for i in range(len(impacts) - 1))

    def test_create_tornado_diagram_with_nested(self, analyzer):
        """Test tornado diagram with nested parameters."""
        parameters = [
            "frequency",
            ("base_operating_margin", "manufacturer.base_operating_margin"),
            ("tax_rate", "manufacturer.tax_rate"),
        ]

        tornado_data = analyzer.create_tornado_diagram(parameters)

        assert isinstance(tornado_data, pd.DataFrame)
        assert "base_operating_margin" in tornado_data["parameter"].values
        assert "tax_rate" in tornado_data["parameter"].values

    def test_analyze_two_way(self, analyzer):
        """Test two-way sensitivity analysis."""
        result = analyzer.analyze_two_way(
            "frequency",
            "severity_mean",
            param1_range=(4, 6),
            param2_range=(80000, 120000),
            n_points1=3,
            n_points2=3,
            metric="optimal_roe",
        )

        assert isinstance(result, TwoWaySensitivityResult)
        assert result.parameter1 == "frequency"
        assert result.parameter2 == "severity_mean"
        assert result.metric_grid.shape == (3, 3)
        assert len(result.values1) == 3
        assert len(result.values2) == 3

    def test_analyze_two_way_with_nested(self, analyzer):
        """Test two-way analysis with nested parameters."""
        result = analyzer.analyze_two_way(
            ("margin", "manufacturer.base_operating_margin"),
            "frequency",
            n_points1=2,
            n_points2=2,
            relative_range=0.1,
        )

        assert result.parameter1 == "margin"
        assert result.parameter2 == "frequency"
        assert result.metric_grid.shape == (2, 2)

    def test_clear_cache(self, analyzer):
        """Test cache clearing."""
        # Run analysis to populate cache
        analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

        assert len(analyzer.results_cache) > 0

        # Clear cache
        analyzer.clear_cache()

        assert len(analyzer.results_cache) == 0

    def test_analyze_parameter_group(self, analyzer):
        """Test analyzing a group of parameters."""
        parameter_group = {
            "frequency": (3, 7),
            "severity_mean": (80000, 120000),
            "premium_rate": (0.01, 0.03),
        }

        results = analyzer.analyze_parameter_group(
            parameter_group, n_points=5, metric="optimal_roe"
        )

        assert len(results) == 3
        assert "frequency" in results
        assert "severity_mean" in results
        assert "premium_rate" in results

        for param, result in results.items():
            assert isinstance(result, SensitivityResult)
            assert len(result.variations) == 5

    def test_update_nested_config(self, analyzer):
        """Test updating nested configuration values."""
        config: Dict[str, Any] = {"a": {"b": {"c": 1}}, "d": 2}

        # Update nested value
        new_config = analyzer._update_nested_config(config, "a.b.c", 5)
        assert new_config["a"]["b"]["c"] == 5
        assert new_config["d"] == 2
        # Original should not be modified (it returns a copy)
        assert config["a"]["b"]["c"] == 1

        # Update top-level value
        new_config2 = analyzer._update_nested_config(config, "d", 10)
        assert new_config2["d"] == 10
        assert new_config2["a"]["b"]["c"] == 1
        # Original should not be modified
        assert config["d"] == 2

    def test_extract_metric(self, analyzer):
        """Test metric extraction from results."""
        result = MockOptimizationResult({"test": 1})

        # Test different metrics
        assert (
            analyzer._extract_metric(result, "optimal_roe") == result.optimal_strategy.expected_roe
        )
        assert (
            analyzer._extract_metric(result, "bankruptcy_risk")
            == result.optimal_strategy.bankruptcy_risk
        )
        assert (
            analyzer._extract_metric(result, "growth_rate") == result.optimal_strategy.growth_rate
        )

        # Test with simple result structure (no optimal_strategy)
        simple_result = MagicMock(spec=["roe", "ruin_prob"])
        simple_result.roe = 0.15
        simple_result.ruin_prob = 0.01

        assert analyzer._extract_metric(simple_result, "optimal_roe") == 0.15
        assert analyzer._extract_metric(simple_result, "bankruptcy_risk") == 0.01

    def test_update_nested_config_no_deepcopy(self, analyzer):
        """Verify _update_nested_config uses shallow copies, not deepcopy.

        Sibling dicts should share identity with the original config,
        proving that no full deep copy is performed (#484).
        """
        config: Dict[str, Any] = {
            "a": {"b": {"c": 1}, "sibling": {"x": 99}},
            "d": 2,
        }

        new_config = analyzer._update_nested_config(config, "a.b.c", 5)

        # Modified value is updated
        assert new_config["a"]["b"]["c"] == 5
        # Original is unmodified
        assert config["a"]["b"]["c"] == 1

        # Sibling dict shares identity (shallow copy, not deep copy)
        assert new_config["a"]["sibling"] is config["a"]["sibling"]

    def test_analyze_two_way_parallel(self, analyzer):
        """Parallel analyze_two_way produces same results as sequential (#492)."""
        common_kwargs: Dict[str, Any] = {
            "param1": "frequency",
            "param2": "severity_mean",
            "param1_range": (4, 6),
            "param2_range": (80000, 120000),
            "n_points1": 3,
            "n_points2": 3,
            "metric": "optimal_roe",
        }

        sequential = analyzer.analyze_two_way(**common_kwargs)

        # Clear cache so parallel path actually runs optimizations
        analyzer.clear_cache()

        parallel = analyzer.analyze_two_way(**common_kwargs, max_workers=2)

        np.testing.assert_array_equal(sequential.metric_grid, parallel.metric_grid)
        assert sequential.parameter1 == parallel.parameter1
        assert sequential.parameter2 == parallel.parameter2

    def test_analyze_two_way_parallel_with_cache(self, analyzer):
        """Cached results are not re-submitted to workers (#492)."""
        kwargs: Dict[str, Any] = {
            "param1": "frequency",
            "param2": "severity_mean",
            "param1_range": (4, 6),
            "param2_range": (80000, 120000),
            "n_points1": 3,
            "n_points2": 3,
            "metric": "optimal_roe",
        }

        # First run fills cache
        analyzer.analyze_two_way(**kwargs)
        calls_after_first = analyzer.optimizer.call_count

        # Second run with max_workers — all results should come from cache
        analyzer.analyze_two_way(**kwargs, max_workers=2)

        assert analyzer.optimizer.call_count == calls_after_first


class TestIntegration:
    """Integration tests for sensitivity analysis."""

    def test_full_sensitivity_workflow(self):
        """Test complete sensitivity analysis workflow."""
        # Setup
        config = {
            "frequency": 5.0,
            "severity_mean": 100000,
            "premium_rate": 0.02,
            "company": {"size": "medium", "revenue": 10000000},
        }

        optimizer = MockOptimizer()
        analyzer = SensitivityAnalyzer(config, optimizer)

        # 1. Single parameter analysis
        freq_result = analyzer.analyze_parameter("frequency", param_range=(3, 8), n_points=6)

        assert freq_result.parameter == "frequency"
        assert len(freq_result.variations) == 6

        # 2. Tornado diagram
        tornado_data = analyzer.create_tornado_diagram(
            ["frequency", "severity_mean", "premium_rate"], metric="optimal_roe"
        )

        assert len(tornado_data) == 3
        assert tornado_data.iloc[0]["impact"] >= tornado_data.iloc[1]["impact"]

        # 3. Two-way sensitivity
        two_way_result = analyzer.analyze_two_way(
            "frequency", "severity_mean", n_points1=4, n_points2=4
        )

        assert two_way_result.metric_grid.shape == (4, 4)

        # 4. Parameter group analysis
        group_results = analyzer.analyze_parameter_group(
            {"frequency": (3, 7), "severity_mean": (50000, 150000)}
        )

        assert len(group_results) == 2

        # Verify caching worked
        assert len(analyzer.results_cache) > 0

    def test_sensitivity_with_constraints(self):
        """Test sensitivity analysis respecting constraints."""
        config = {
            "frequency": 5.0,
            "severity_mean": 100000,
            "min_retention": 10000,
            "max_retention": 500000,
        }

        # Create optimizer that respects constraints
        class ConstrainedOptimizer:
            def optimize(self, config):
                result = MockOptimizationResult(config)
                # Ensure retention is within bounds
                retention = config.get("min_retention", 0)
                result.optimal_strategy.deductible = max(
                    config.get("min_retention", 0),
                    min(config.get("max_retention", float("inf")), retention),
                )
                return result

        optimizer = ConstrainedOptimizer()
        analyzer = SensitivityAnalyzer(config, optimizer)

        result = analyzer.analyze_parameter("min_retention", param_range=(5000, 50000), n_points=5)

        # Verify all retentions are within bounds
        retentions = result.metrics["optimal_retention"]
        assert all(r >= 5000 for r in retentions)

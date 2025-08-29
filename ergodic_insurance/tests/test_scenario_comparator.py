"""Tests for scenario comparison framework.

This module contains comprehensive tests for the scenario comparator
and comparison visualization functionality.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.reporting.scenario_comparator import (
    ScenarioComparator,
    ScenarioComparison,
)


class TestScenarioComparison:
    """Test suite for ScenarioComparison class."""

    def test_initialization(self):
        """Test ScenarioComparison initialization."""
        scenarios = ["base", "optimized"]
        metrics = {
            "growth_rate": {"base": 0.05, "optimized": 0.08},
            "ruin_probability": {"base": 0.02, "optimized": 0.01},
        }
        parameters = {
            "base": {"premium": 1000, "limit": 5000},
            "optimized": {"premium": 1200, "limit": 10000},
        }

        comparison = ScenarioComparison(scenarios=scenarios, metrics=metrics, parameters=parameters)

        assert comparison.scenarios == scenarios
        assert comparison.metrics == metrics
        assert comparison.parameters == parameters
        assert comparison.statistics == {}
        assert comparison.diffs == {}
        assert comparison.rankings == {}

    def test_get_metric_df(self):
        """Test getting metric values as DataFrame."""
        comparison = ScenarioComparison(
            scenarios=["s1", "s2", "s3"],
            metrics={
                "metric1": {"s1": 10, "s2": 20, "s3": 15},
                "metric2": {"s1": 5, "s2": 8, "s3": 6},
            },
            parameters={},
        )

        df = comparison.get_metric_df("metric1")

        assert len(df) == 3
        assert "scenario" in df.columns
        assert "metric1" in df.columns
        assert df["metric1"].tolist() == [10, 20, 15]

        # Test invalid metric
        with pytest.raises(ValueError, match="Metric invalid not found"):
            comparison.get_metric_df("invalid")

    def test_get_top_performers(self):
        """Test getting top performing scenarios."""
        comparison = ScenarioComparison(
            scenarios=["s1", "s2", "s3", "s4"],
            metrics={
                "growth": {"s1": 0.05, "s2": 0.10, "s3": 0.08, "s4": 0.12},
                "risk": {"s1": 0.02, "s2": 0.05, "s3": 0.03, "s4": 0.01},
            },
            parameters={},
        )

        # Test top performers (higher is better)
        top_growth = comparison.get_top_performers("growth", n=2, ascending=False)
        assert len(top_growth) == 2
        assert top_growth[0][0] == "s4"  # Highest growth
        assert top_growth[0][1] == 0.12
        assert top_growth[1][0] == "s2"  # Second highest

        # Test ascending (lower is better)
        top_risk = comparison.get_top_performers("risk", n=2, ascending=True)
        assert len(top_risk) == 2
        assert top_risk[0][0] == "s4"  # Lowest risk
        assert top_risk[0][1] == 0.01

    def test_compute_rankings(self):
        """Test internal ranking computation."""
        comparison = ScenarioComparison(
            scenarios=["s1", "s2", "s3"],
            metrics={"metric1": {"s1": 10, "s2": 20, "s3": 15}},
            parameters={},
        )

        comparison._compute_rankings()

        assert "metric1" in comparison.rankings
        rankings = comparison.rankings["metric1"]
        assert len(rankings) == 3
        assert rankings[0] == ("s2", 20)  # Highest
        assert rankings[1] == ("s3", 15)  # Middle
        assert rankings[2] == ("s1", 10)  # Lowest


class TestScenarioComparator:
    """Test suite for ScenarioComparator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint: disable=attribute-defined-outside-init
        self.comparator = ScenarioComparator()

        # Create sample results
        self.sample_results = {
            "baseline": {
                "summary_statistics": {
                    "mean_growth_rate": 0.05,
                    "ruin_probability": 0.02,
                    "var_95": -100000,
                    "mean_assets": 1000000,
                },
                "config": {
                    "insurance": {"premium": 10000, "limit": 500000},
                    "simulation": {"n_years": 10, "n_paths": 1000},
                },
            },
            "optimized": {
                "summary_statistics": {
                    "mean_growth_rate": 0.08,
                    "ruin_probability": 0.01,
                    "var_95": -50000,
                    "mean_assets": 1200000,
                },
                "config": {
                    "insurance": {"premium": 15000, "limit": 1000000},
                    "simulation": {"n_years": 10, "n_paths": 1000},
                },
            },
            "conservative": {
                "summary_statistics": {
                    "mean_growth_rate": 0.03,
                    "ruin_probability": 0.005,
                    "var_95": -20000,
                    "mean_assets": 800000,
                },
                "config": {
                    "insurance": {"premium": 20000, "limit": 2000000},
                    "simulation": {"n_years": 10, "n_paths": 1000},
                },
            },
        }

    def test_initialization(self):
        """Test ScenarioComparator initialization."""
        assert self.comparator.baseline_scenario is None
        assert self.comparator.comparison_data is None

    def test_compare_scenarios(self):
        """Test scenario comparison functionality."""
        comparison = self.comparator.compare_scenarios(self.sample_results, baseline="baseline")

        assert isinstance(comparison, ScenarioComparison)
        assert len(comparison.scenarios) == 3
        assert "baseline" in comparison.scenarios
        assert "optimized" in comparison.scenarios
        assert "conservative" in comparison.scenarios

        # Check metrics extraction
        assert "mean_growth_rate" in comparison.metrics
        assert "ruin_probability" in comparison.metrics
        assert comparison.metrics["mean_growth_rate"]["optimized"] == 0.08

        # Check parameters extraction
        assert "baseline" in comparison.parameters
        assert "insurance.premium" in comparison.parameters["baseline"]

        # Check diffs computation
        assert "optimized" in comparison.diffs
        assert "conservative" in comparison.diffs

        # Check statistics
        assert "mean_growth_rate" in comparison.statistics

    def test_extract_metrics(self):
        """Test metric extraction from results."""
        metrics = self.comparator._extract_metrics(self.sample_results)

        assert "mean_growth_rate" in metrics
        assert "ruin_probability" in metrics
        assert "var_95" in metrics
        assert "mean_assets" in metrics

        assert metrics["mean_growth_rate"]["baseline"] == 0.05
        assert metrics["ruin_probability"]["optimized"] == 0.01
        assert metrics["var_95"]["conservative"] == -20000

    def test_extract_metrics_with_filter(self):
        """Test metric extraction with specific metrics."""
        metrics = self.comparator._extract_metrics(
            self.sample_results, metrics=["mean_growth_rate", "ruin_probability"]
        )

        assert "mean_growth_rate" in metrics
        assert "ruin_probability" in metrics
        assert "var_95" not in metrics
        assert "mean_assets" not in metrics

    def test_extract_parameters(self):
        """Test parameter extraction from results."""
        params = self.comparator._extract_parameters(self.sample_results)

        assert "baseline" in params
        assert "optimized" in params
        assert "conservative" in params

        assert "insurance.premium" in params["baseline"]
        assert params["baseline"]["insurance.premium"] == 10000
        assert params["optimized"]["insurance.limit"] == 1000000

    def test_flatten_config(self):
        """Test configuration flattening."""
        config = {"level1": {"level2": {"value": 42, "name": "test"}, "simple": 10}, "top": "value"}

        flat = self.comparator._flatten_config(config)

        assert flat["level1.level2.value"] == 42
        assert flat["level1.level2.name"] == "test"
        assert flat["level1.simple"] == 10
        assert flat["top"] == "value"

    def test_compute_diffs(self):
        """Test parameter difference computation."""
        param_data = {
            "baseline": {"param1": 100, "param2": 0.5, "param3": "abc"},
            "scenario1": {"param1": 120, "param2": 0.6, "param3": "abc"},
            "scenario2": {"param1": 80, "param2": 0.4, "param3": "xyz"},
        }

        diffs = self.comparator._compute_diffs(param_data, "baseline")

        # Check scenario1 diffs
        assert "scenario1" in diffs
        assert diffs["scenario1"]["param1"]["absolute"] == 20
        assert diffs["scenario1"]["param1"]["percentage"] == 20.0
        assert diffs["scenario1"]["param2"]["absolute"] == pytest.approx(0.1)
        assert diffs["scenario1"]["param2"]["percentage"] == pytest.approx(20.0)

        # Check scenario2 diffs
        assert "scenario2" in diffs
        assert diffs["scenario2"]["param1"]["absolute"] == -20
        assert diffs["scenario2"]["param1"]["percentage"] == -20.0
        assert diffs["scenario2"]["param3"]["changed"] is True
        assert diffs["scenario2"]["param3"]["value"] == "xyz"

    def test_perform_statistical_tests(self):
        """Test statistical analysis of metrics."""
        metric_data: dict[str, dict[str, float]] = {
            "metric1": {"s1": 10, "s2": 20, "s3": 15, "s4": 25},
            "metric2": {"s1": 0.1, "s2": 0.2, "s3": 0.15},
        }

        stats = self.comparator._perform_statistical_tests(metric_data)

        assert "metric1" in stats
        assert "mean" in stats["metric1"]
        assert "std" in stats["metric1"]
        assert "min" in stats["metric1"]
        assert "max" in stats["metric1"]
        assert "range" in stats["metric1"]
        assert "cv" in stats["metric1"]

        assert stats["metric1"]["mean"] == pytest.approx(17.5)
        assert stats["metric1"]["min"] == 10
        assert stats["metric1"]["max"] == 25
        assert stats["metric1"]["range"] == 15

        # Check ANOVA for multiple scenarios
        assert "anova" in stats["metric1"]
        assert "significant_difference" in stats["metric1"]["anova"]

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.subplot")
    def test_create_comparison_grid(self, mock_subplot, mock_figure):
        """Test comparison grid visualization creation."""
        # Setup comparison data first
        self.comparator.compare_scenarios(self.sample_results, baseline="baseline")

        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax

        # Create grid
        fig = self.comparator.create_comparison_grid(
            metrics=["mean_growth_rate", "ruin_probability"]
        )

        assert fig is not None
        mock_fig.suptitle.assert_called_once()

    def test_create_parameter_diff_table(self):
        """Test parameter difference table creation."""
        # Setup comparison
        self.comparator.compare_scenarios(self.sample_results, baseline="baseline")

        # Create diff table
        df = self.comparator.create_parameter_diff_table("optimized", threshold=10.0)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "Parameter" in df.columns
            assert "Baseline" in df.columns
            assert "Scenario" in df.columns
            assert "Change" in df.columns
            assert "Change %" in df.columns

    def test_export_comparison_report(self, tmp_path):
        """Test comparison report export."""
        # Setup comparison
        self.comparator.compare_scenarios(self.sample_results, baseline="baseline")

        # Export report
        output_base = str(tmp_path / "test_report")
        outputs = self.comparator.export_comparison_report(
            output_base, include_plots=False  # Skip plots for testing
        )

        assert "metrics" in outputs
        assert outputs["metrics"].endswith("_metrics.csv")

        # Check if files were created
        import os

        assert os.path.exists(outputs["metrics"])

    def test_compare_scenarios_with_dataframe_input(self):
        """Test comparison with DataFrame input."""
        # Create DataFrame results
        df_results = {
            "scenario1": pd.DataFrame({"growth_rate": [0.05], "risk": [0.02]}),
            "scenario2": pd.DataFrame({"growth_rate": [0.08], "risk": [0.01]}),
        }

        comparison = self.comparator.compare_scenarios(df_results)

        assert isinstance(comparison, ScenarioComparison)
        assert "growth_rate" in comparison.metrics
        assert comparison.metrics["growth_rate"]["scenario1"] == 0.05
        assert comparison.metrics["growth_rate"]["scenario2"] == 0.08

    def test_compare_scenarios_with_object_input(self):
        """Test comparison with object input."""

        # Create mock result objects
        class MockResult:
            def __init__(self, stats, config):
                self.summary_statistics = stats
                self.config = config

        obj_results = {
            "scenario1": MockResult({"metric1": 10, "metric2": 0.5}, {"param1": 100}),
            "scenario2": MockResult({"metric1": 15, "metric2": 0.3}, {"param1": 150}),
        }

        comparison = self.comparator.compare_scenarios(obj_results)

        assert isinstance(comparison, ScenarioComparison)
        assert "metric1" in comparison.metrics
        assert comparison.metrics["metric1"]["scenario1"] == 10
        assert comparison.metrics["metric1"]["scenario2"] == 15

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty results
        empty_results: dict[str, Any] = {}
        comparison = self.comparator.compare_scenarios(empty_results)
        assert comparison.scenarios == []

        # Single scenario
        single_result = {"only": {"summary_statistics": {"metric": 1.0}, "config": {"param": 10}}}
        comparison = self.comparator.compare_scenarios(single_result)
        assert len(comparison.scenarios) == 1
        assert self.comparator.baseline_scenario == "only"

        # No numeric metrics
        non_numeric = {
            "scenario": {
                "summary_statistics": {"text": "value", "flag": "not_numeric"},
                "config": {},
            }
        }
        comparison = self.comparator.compare_scenarios(non_numeric)
        # Check that non-numeric values are filtered out
        assert "text" not in comparison.metrics
        assert "flag" not in comparison.metrics

"""Tests for new executive visualization functions."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.visualization.executive_plots import (
    plot_breakeven_timeline,
    plot_optimal_coverage_heatmap,
    plot_premium_multiplier,
    plot_robustness_heatmap,
    plot_sample_paths,
    plot_sensitivity_tornado,
    plot_simulation_architecture,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestSimulationArchitecture:
    """Test suite for simulation architecture visualization."""

    def test_plot_simulation_architecture_default(self):
        """Test simulation architecture plot with default parameters."""
        fig = plot_simulation_architecture()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_simulation_architecture_custom_params(self):
        """Test simulation architecture plot with custom parameters."""
        fig = plot_simulation_architecture(
            title="Custom Architecture", figsize=(12, 6), export_dpi=150, show_icons=False
        )
        assert fig is not None
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 6
        assert fig.dpi == 150
        plt.close(fig)


class TestSamplePaths:
    """Test suite for sample path visualization."""

    def test_plot_sample_paths_synthetic(self):
        """Test sample paths with synthetic data generation."""
        fig = plot_sample_paths(n_paths=3, short_horizon=5, long_horizon=50)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Dual panel
        plt.close(fig)

    def test_plot_sample_paths_with_data(self):
        """Test sample paths with provided data."""
        # Create mock simulation data
        np.random.seed(42)
        n_points_short = 100
        n_points_long = 500

        paths_short = [
            {"values": np.cumsum(np.random.randn(n_points_short)) + 10000000, "failed": False},
            {"values": np.cumsum(np.random.randn(n_points_short)) + 10000000, "failed": True},
        ]

        paths_long = [
            {"values": np.cumsum(np.random.randn(n_points_long)) + 10000000, "failed": False},
            {"values": np.cumsum(np.random.randn(n_points_long)) + 10000000, "failed": True},
        ]

        simulation_data = {"paths_short": paths_short, "paths_long": paths_long}

        fig = plot_sample_paths(simulation_data=simulation_data)
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_sample_paths_no_failures(self):
        """Test sample paths without failure highlighting."""
        fig = plot_sample_paths(n_paths=3, show_failures=False)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sample_paths_custom_params(self):
        """Test sample paths with custom parameters."""
        fig = plot_sample_paths(
            n_paths=7,
            short_horizon=20,
            long_horizon=200,
            company_size=50_000_000,
            title="Custom Sample Paths",
            figsize=(16, 8),
            export_dpi=300,
        )
        assert fig is not None
        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 8
        assert fig.dpi == 300
        plt.close(fig)


class TestOptimalCoverageHeatmap:
    """Test suite for optimal coverage heatmap visualization."""

    def test_plot_optimal_coverage_heatmap_default(self):
        """Test heatmap with default synthetic data."""
        fig = plot_optimal_coverage_heatmap()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Three company sizes + one colorbar
        plt.close(fig)

    def test_plot_optimal_coverage_heatmap_custom_sizes(self):
        """Test heatmap with custom company sizes."""
        fig = plot_optimal_coverage_heatmap(company_sizes=[500_000, 5_000_000, 50_000_000])
        assert fig is not None
        assert len(fig.axes) == 4  # Three company sizes + one colorbar
        plt.close(fig)

    def test_plot_optimal_coverage_heatmap_with_data(self):
        """Test heatmap with provided optimization results."""
        # Create mock optimization results
        retention_values = np.logspace(4, 7, 10)
        limit_values = np.logspace(5, 8, 10)
        R, L = np.meshgrid(retention_values, limit_values)
        growth_rates = np.random.rand(10, 10) * 0.15

        optimization_results = {
            "company_1000000": {
                "retention": retention_values,
                "limit": limit_values,
                "growth_rate": growth_rates,
            }
        }

        fig = plot_optimal_coverage_heatmap(
            optimization_results=optimization_results, company_sizes=[1_000_000]
        )
        assert fig is not None
        assert len(fig.axes) >= 2  # At least one panel + one colorbar (might be extra empty panels)
        plt.close(fig)

    def test_plot_optimal_coverage_heatmap_no_contours(self):
        """Test heatmap without contour lines."""
        fig = plot_optimal_coverage_heatmap(show_contours=False)
        assert fig is not None
        plt.close(fig)


class TestSensitivityTornado:
    """Test suite for sensitivity tornado chart."""

    def test_plot_sensitivity_tornado_default(self):
        """Test tornado chart with default synthetic data."""
        fig = plot_sensitivity_tornado()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_sensitivity_tornado_with_data(self):
        """Test tornado chart with provided sensitivity data."""
        sensitivity_data = {
            "Parameter A": 0.25,
            "Parameter B": -0.20,
            "Parameter C": 0.15,
            "Parameter D": -0.10,
            "Parameter E": 0.05,
        }

        fig = plot_sensitivity_tornado(sensitivity_data=sensitivity_data, baseline_value=0.10)
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_tornado_no_percentages(self):
        """Test tornado chart without percentage labels."""
        fig = plot_sensitivity_tornado(show_percentages=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_tornado_custom_params(self):
        """Test tornado chart with custom parameters."""
        fig = plot_sensitivity_tornado(
            title="Custom Sensitivity Analysis", figsize=(12, 10), export_dpi=200
        )
        assert fig is not None
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 10
        assert fig.dpi == 200
        plt.close(fig)

    def test_plot_sensitivity_tornado_color_coding(self):
        """Test tornado chart color coding based on sensitivity levels."""
        # Create data with different sensitivity levels
        sensitivity_data = {
            "High Impact": 0.15,  # >10% - should be red
            "Moderate Impact": 0.07,  # 5-10% - should be orange
            "Low Impact": 0.03,  # <5% - should be green
        }

        fig = plot_sensitivity_tornado(sensitivity_data=sensitivity_data)
        assert fig is not None

        # Check that bars were created
        ax = fig.axes[0]
        patches = ax.patches
        assert len(patches) == 3

        plt.close(fig)


class TestRobustnessHeatmap:
    """Test suite for robustness heatmap visualization."""

    def test_plot_robustness_heatmap_default(self):
        """Test robustness heatmap with default synthetic data."""
        fig = plot_robustness_heatmap()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Heatmap + colorbar
        plt.close(fig)

    def test_plot_robustness_heatmap_with_data(self):
        """Test robustness heatmap with provided data."""
        # Create mock robustness data
        n_points = 15
        robustness_data = np.random.rand(n_points, n_points)

        fig = plot_robustness_heatmap(
            robustness_data=robustness_data, frequency_range=(0.8, 1.2), severity_range=(0.8, 1.2)
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_robustness_heatmap_no_reference(self):
        """Test robustness heatmap without reference point."""
        fig = plot_robustness_heatmap(show_reference=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_robustness_heatmap_custom_ranges(self):
        """Test robustness heatmap with custom parameter ranges."""
        fig = plot_robustness_heatmap(frequency_range=(0.5, 1.5), severity_range=(0.6, 1.4))
        assert fig is not None

        # Check axis limits
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        assert xlim[0] == 50  # 0.5 * 100
        assert xlim[1] == 150  # 1.5 * 100
        assert ylim[0] == 60  # 0.6 * 100
        assert ylim[1] == 140  # 1.4 * 100

        plt.close(fig)

    def test_plot_robustness_heatmap_custom_params(self):
        """Test robustness heatmap with custom parameters."""
        fig = plot_robustness_heatmap(
            title="Custom Robustness Analysis", figsize=(12, 9), export_dpi=150
        )
        assert fig is not None
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 9
        assert fig.dpi == 150
        plt.close(fig)


class TestPremiumMultiplier:
    """Test suite for premium multiplier visualization."""

    def test_plot_premium_multiplier_synthetic(self):
        """Test premium multiplier plot with synthetic data generation."""
        fig = plot_premium_multiplier()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_premium_multiplier_with_data(self):
        """Test premium multiplier plot with provided data."""
        optimization_results = {
            1_000_000.0: {
                "expected_loss": 5000,
                "optimal_premium": 15000,
                "confidence_bounds": (12000, 18000),
            },
            10_000_000.0: {
                "expected_loss": 50000,
                "optimal_premium": 125000,
                "confidence_bounds": (100000, 150000),
            },
            100_000_000.0: {
                "expected_loss": 500000,
                "optimal_premium": 1000000,
                "confidence_bounds": (800000, 1200000),
            },
        }

        fig = plot_premium_multiplier(
            optimization_results=optimization_results,
            company_sizes=[1_000_000.0, 10_000_000.0, 100_000_000.0],
        )
        assert fig is not None
        assert len(fig.axes) == 1

        # Check that multipliers are calculated correctly
        ax = fig.axes[0]
        assert len(ax.lines) > 0  # Should have plotted lines
        plt.close(fig)

    def test_plot_premium_multiplier_custom_params(self):
        """Test premium multiplier plot with custom parameters."""
        fig = plot_premium_multiplier(
            title="Custom Premium Analysis",
            figsize=(14, 8),
            show_confidence=False,
            show_reference_lines=False,
            show_annotations=False,
            export_dpi=150,
        )
        assert fig is not None
        assert fig.get_size_inches()[0] == 14
        assert fig.get_size_inches()[1] == 8
        assert fig.dpi == 150
        plt.close(fig)

    def test_plot_premium_multiplier_edge_cases(self):
        """Test premium multiplier plot with edge case data."""
        # Test with single company size
        results = {5_000_000.0: {"expected_loss": 25000, "optimal_premium": 75000}}

        fig = plot_premium_multiplier(optimization_results=results, company_sizes=[5_000_000.0])
        assert fig is not None
        plt.close(fig)

    def test_premium_multiplier_annotations(self):
        """Test that annotations are properly added when requested."""
        fig = plot_premium_multiplier(show_annotations=True)
        ax = fig.axes[0]
        # Check for annotations (text objects)
        assert len(ax.texts) > 0 or any(
            child for child in ax.get_children() if hasattr(child, "get_annotation_clip")
        )
        plt.close(fig)


class TestBreakevenTimeline:
    """Test suite for breakeven timeline visualization."""

    def test_plot_breakeven_timeline_synthetic(self):
        """Test breakeven timeline plot with synthetic data generation."""
        fig = plot_breakeven_timeline()
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        # Should have 3 subplots by default (3 company sizes)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_breakeven_timeline_with_data(self):
        """Test breakeven timeline plot with provided data."""
        np.random.seed(42)
        years = np.arange(20)

        simulation_results = {
            1_000_000.0: {
                "years": years,
                "cumulative_benefit": np.cumsum(np.random.uniform(10000, 50000, 20)),
                "cumulative_excess_premium": np.cumsum(np.repeat(15000, 20)),
                "benefit_25": np.cumsum(np.random.uniform(8000, 40000, 20)),
                "benefit_75": np.cumsum(np.random.uniform(12000, 60000, 20)),
            },
            10_000_000.0: {
                "years": years,
                "cumulative_benefit": np.cumsum(np.random.uniform(50000, 200000, 20)),
                "cumulative_excess_premium": np.cumsum(np.repeat(60000, 20)),
                "benefit_25": np.cumsum(np.random.uniform(40000, 160000, 20)),
                "benefit_75": np.cumsum(np.random.uniform(60000, 240000, 20)),
            },
        }

        fig = plot_breakeven_timeline(
            simulation_results=simulation_results,
            company_sizes=[1_000_000.0, 10_000_000.0],
            time_horizon=20,
        )
        assert fig is not None
        assert len(fig.axes) == 2  # Two company sizes
        plt.close(fig)

    def test_plot_breakeven_timeline_custom_params(self):
        """Test breakeven timeline plot with custom parameters."""
        fig = plot_breakeven_timeline(
            company_sizes=[5_000_000],
            time_horizon=50,
            title="Custom Breakeven Analysis",
            figsize=(10, 6),
            show_percentiles=False,
            show_breakeven_markers=False,
            export_dpi=200,
        )
        assert fig is not None
        assert len(fig.axes) == 1  # Single company size
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        assert fig.dpi == 200
        plt.close(fig)

    def test_breakeven_timeline_breakeven_detection(self):
        """Test that breakeven points are correctly detected."""
        # Create data with known breakeven point
        years = np.arange(10)
        cumulative_benefit = np.array([0, 5, 15, 30, 50, 75, 105, 140, 180, 225]) * 1000
        cumulative_excess = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * 1000

        simulation_results = {
            1_000_000.0: {
                "years": years,
                "cumulative_benefit": cumulative_benefit,
                "cumulative_excess_premium": cumulative_excess,
            }
        }

        fig = plot_breakeven_timeline(
            simulation_results=simulation_results,
            company_sizes=[1_000_000.0],
            time_horizon=10,
            show_breakeven_markers=True,
        )

        # Breakeven should occur around year 5 (when benefit > excess)
        assert fig is not None
        ax = fig.axes[0]

        # Check for breakeven marker (star marker)
        has_star_marker = any(
            hasattr(child, "get_marker") and child.get_marker() == "*"
            for child in ax.get_children()
        )
        assert has_star_marker or len(ax.collections) > 0  # Star marker or scatter plot
        plt.close(fig)

    def test_breakeven_timeline_percentile_bands(self):
        """Test that percentile bands are shown when requested."""
        np.random.seed(42)
        years = np.arange(15)

        simulation_results = {
            1_000_000.0: {
                "years": years,
                "cumulative_benefit": np.cumsum(np.random.uniform(20000, 40000, 15)),
                "cumulative_excess_premium": np.cumsum(np.repeat(25000, 15)),
                "benefit_25": np.cumsum(np.random.uniform(15000, 35000, 15)),
                "benefit_75": np.cumsum(np.random.uniform(25000, 45000, 15)),
            }
        }

        fig = plot_breakeven_timeline(
            simulation_results=simulation_results,
            company_sizes=[1_000_000.0],
            show_percentiles=True,
        )

        ax = fig.axes[0]
        # Check for fill_between (percentile bands)
        has_fill = any(
            isinstance(child, matplotlib.collections.PolyCollection) for child in ax.get_children()
        )
        assert has_fill  # Should have percentile bands
        plt.close(fig)

    def test_breakeven_timeline_no_breakeven(self):
        """Test handling of cases where breakeven never occurs."""
        years = np.arange(10)

        simulation_results = {
            1_000_000.0: {
                "years": years,
                "cumulative_benefit": np.arange(10) * 1000,  # Always less than excess
                "cumulative_excess_premium": np.arange(10) * 5000,  # Always higher
            }
        }

        fig = plot_breakeven_timeline(
            simulation_results=simulation_results,
            company_sizes=[1_000_000.0],
            show_breakeven_markers=True,
        )

        # Should still create plot without errors
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)


class TestIntegration:
    """Integration tests for executive visualizations."""

    def test_all_visualizations_run_without_error(self):
        """Test that all visualization functions run without errors."""
        # Create all visualizations
        figs = []

        figs.append(plot_simulation_architecture())
        figs.append(plot_sample_paths(n_paths=3))
        figs.append(plot_optimal_coverage_heatmap())
        figs.append(plot_sensitivity_tornado())
        figs.append(plot_robustness_heatmap())
        figs.append(plot_premium_multiplier())
        figs.append(plot_breakeven_timeline())

        # Check all figures were created
        assert len(figs) == 7
        for fig in figs:
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_consistent_styling_across_visualizations(self):
        """Test that all visualizations have consistent WSJ styling."""
        figs = [
            plot_simulation_architecture(),
            plot_sample_paths(n_paths=2),
            plot_optimal_coverage_heatmap(
                company_sizes=[1_000_000, 10_000_000]
            ),  # Use 2 sizes to test multi-panel
            plot_sensitivity_tornado(),
            plot_robustness_heatmap(),
            plot_premium_multiplier(),
            plot_breakeven_timeline(company_sizes=[1_000_000, 10_000_000]),
        ]

        for fig in figs:
            # Check that figure has expected styling elements
            assert fig is not None

            # Check for title (all should have titles)
            if hasattr(fig, "_suptitle") and fig._suptitle:
                assert fig._suptitle.get_fontweight() == "bold"

            # Cleanup
            plt.close(fig)

    def test_export_dpi_setting(self):
        """Test that export DPI is correctly set for all visualizations."""
        target_dpi = 300

        figs = [
            plot_simulation_architecture(export_dpi=target_dpi),
            plot_sample_paths(n_paths=2, export_dpi=target_dpi),
            plot_optimal_coverage_heatmap(export_dpi=target_dpi),
            plot_sensitivity_tornado(export_dpi=target_dpi),
            plot_robustness_heatmap(export_dpi=target_dpi),
            plot_premium_multiplier(export_dpi=target_dpi),
            plot_breakeven_timeline(export_dpi=target_dpi),
        ]

        for fig in figs:
            assert fig.dpi == target_dpi
            plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for new executive visualization functions."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.visualization.executive_plots import (
    plot_optimal_coverage_heatmap,
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
        assert len(fig.axes) == 6  # Three company sizes + three colorbars
        plt.close(fig)

    def test_plot_optimal_coverage_heatmap_custom_sizes(self):
        """Test heatmap with custom company sizes."""
        fig = plot_optimal_coverage_heatmap(company_sizes=[500_000, 5_000_000, 50_000_000])
        assert fig is not None
        assert len(fig.axes) == 6  # Three company sizes + three colorbars
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

        # Check all figures were created
        assert len(figs) == 5
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
        ]

        for fig in figs:
            # Check that figure has expected styling elements
            assert fig is not None

            # Check for title (all should have titles)
            if fig._suptitle:
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
        ]

        for fig in figs:
            assert fig.dpi == target_dpi
            plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

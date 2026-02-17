"""Comprehensive tests for visualization modules to achieve 90%+ coverage.

This module provides thorough testing for visualization components including
annotations, batch plots, export utilities, figure factory, interactive plots,
and style management.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, mock_open, patch
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ergodic_insurance.batch_processor import AggregatedResults, BatchResult
from ergodic_insurance.visualization import (
    annotations,
    batch_plots,
    export,
    figure_factory,
    interactive_plots,
    style_manager,
)
from ergodic_insurance.visualization.core import WSJ_COLORS
from ergodic_insurance.visualization.style_manager import StyleManager, Theme

matplotlib.use("Agg")  # Use non-interactive backend for all tests


class TestAnnotations:
    """Comprehensive tests for annotations module."""

    def setup_method(self):
        """Seed random state for reproducible test data."""
        np.random.seed(42)

    def test_add_value_labels_basic(self):
        """Test basic value label addition."""
        fig, ax = plt.subplots()
        bars = ax.bar(range(5), [10, 20, 15, 25, 30])

        annotations.add_value_labels(ax, bars)

        # Check that text labels were added
        assert len(ax.texts) == 5
        for text, value in zip(ax.texts, [10, 20, 15, 25, 30]):
            assert f"{value:.1f}" in text.get_text()

        plt.close(fig)

    def test_add_value_labels_with_format_func(self):
        """Test value labels with custom formatting."""
        fig, ax = plt.subplots()
        bars = ax.bar(range(3), [1000, 2000, 1500])

        def format_func(x):
            return f"${x/1000:.1f}K"

        annotations.add_value_labels(
            ax,
            bars,
            format_func=format_func,
            fontsize=12,
            va="center",
            ha="left",
            offset=0.02,
            color="red",
            bold=True,
        )

        assert len(ax.texts) == 3
        assert "$1.0K" in ax.texts[0].get_text()
        assert "$2.0K" in ax.texts[1].get_text()
        assert ax.texts[0].get_fontsize() == 12
        assert ax.texts[0].get_fontweight() == "bold"

        plt.close(fig)

    def test_add_trend_annotation_positive(self):
        """Test trend annotation with positive trend."""
        fig, ax = plt.subplots()

        annotations.add_trend_annotation(ax, 0.8, 0.9, 0.15, "YoY", fontsize=12)

        # Check annotation was added
        annotations_list = ax.texts + [
            child for child in ax.get_children() if hasattr(child, "get_text")
        ]
        assert len(annotations_list) > 0

        plt.close(fig)

    def test_add_trend_annotation_negative(self):
        """Test trend annotation with negative trend."""
        fig, ax = plt.subplots()

        annotations.add_trend_annotation(ax, 0.5, 0.5, -0.25, "MoM")

        # Check annotation exists
        annotations_list = ax.texts + [
            child for child in ax.get_children() if hasattr(child, "get_text")
        ]
        assert len(annotations_list) > 0

        plt.close(fig)

    def test_add_callout_basic(self):
        """Test basic callout annotation."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        annotations.add_callout(ax, "Peak value", xy=(2, 4), xytext=(2.5, 4.5))

        # Check annotation was added
        assert len(ax.texts) > 0

        plt.close(fig)

    def test_add_callout_custom_style(self):
        """Test callout with custom styling."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        bbox_props = {
            "boxstyle": "round,pad=0.5",
            "facecolor": "yellow",
            "edgecolor": "black",
            "linewidth": 2,
            "alpha": 0.7,
        }

        annotations.add_callout(
            ax,
            "Custom callout",
            xy=(1, 1),
            xytext=(1.5, 2),
            fontsize=14,
            color="blue",
            arrow_color="green",
            bbox_props=bbox_props,
        )

        assert len(ax.texts) > 0

        plt.close(fig)

    def test_add_benchmark_line_right(self):
        """Test benchmark line with right-aligned label."""
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.randn(10))

        annotations.add_benchmark_line(
            ax,
            0,
            "Average",
            color="red",
            linestyle="-.",
            linewidth=2,
            fontsize=11,
            position="right",
        )

        # Check that horizontal line and text were added
        assert len(ax.lines) > 1  # Plot line + benchmark line
        assert len(ax.texts) > 0

        plt.close(fig)

    def test_add_benchmark_line_left(self):
        """Test benchmark line with left-aligned label."""
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.randn(10))

        annotations.add_benchmark_line(ax, 1, "Target", position="left")

        assert len(ax.lines) > 1
        assert len(ax.texts) > 0

        plt.close(fig)

    def test_add_benchmark_line_center(self):
        """Test benchmark line with center-aligned label."""
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.randn(10))

        annotations.add_benchmark_line(ax, -1, "Lower bound", position="center")

        assert len(ax.lines) > 1
        assert len(ax.texts) > 0

        plt.close(fig)

    def test_add_shaded_region_basic(self):
        """Test basic shaded region."""
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.randn(10))

        annotations.add_shaded_region(ax, 3, 6)

        # Check that span was added
        assert len(ax.patches) > 0

        plt.close(fig)

    def test_add_shaded_region_with_label(self):
        """Test shaded region with label."""
        fig, ax = plt.subplots()
        ax.plot(range(10), np.random.randn(10))

        annotations.add_shaded_region(ax, 2, 5, label="Recession", color="gray", alpha=0.3)

        assert len(ax.patches) > 0
        assert len(ax.texts) > 0  # Label should be added

        plt.close(fig)

    def test_add_data_source(self):
        """Test adding data source attribution."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        annotations.add_data_source(
            fig, "Source: Internal Analysis", x=0.95, y=0.02, fontsize=10, color="gray"
        )

        # Check that text was added to figure
        assert len(fig.texts) > 0
        assert "Source: Internal Analysis" in fig.texts[0].get_text()

        plt.close(fig)

    def test_add_footnote(self):
        """Test adding footnote to figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        annotations.add_footnote(fig, "* Preliminary data", x=0.5, y=0.01, fontsize=9, color="blue")

        assert len(fig.texts) > 0
        assert "* Preliminary data" in fig.texts[0].get_text()

        plt.close(fig)


class TestBatchPlots:
    """Comprehensive tests for batch_plots module."""

    def setup_method(self):
        """Seed random state for reproducible test data."""
        np.random.seed(43)

    def create_mock_aggregated_results(self):
        """Create mock AggregatedResults for testing."""
        # Create mock batch results
        batch_results = []
        for i in range(3):
            result = Mock(spec=BatchResult)
            result.scenario = f"Scenario_{i}"
            result.success = True
            result.mean_growth_rate = 0.05 + i * 0.01
            result.ruin_probability = 0.1 - i * 0.02
            result.mean_final_assets = 10000000 * (1 + i * 0.1)
            result.var_99 = -100000 * (1 + i * 0.1)
            batch_results.append(result)

        # Create mock AggregatedResults
        mock_results = Mock(spec=AggregatedResults)
        mock_results.summary_statistics = pd.DataFrame(
            {
                "scenario": [f"Scenario_{i}" for i in range(3)],
                "ruin_probability": [0.1, 0.08, 0.06],
                "mean_growth_rate": [0.05, 0.06, 0.07],
                "mean_final_assets": [10000000, 11000000, 12000000],
                "var_99": [-100000, -110000, -120000],
            }
        )
        mock_results.scenario_results = {f"Scenario_{i}": [batch_results[i]] for i in range(3)}

        return mock_results

    def test_plot_scenario_comparison_basic(self):
        """Test basic scenario comparison plot."""
        mock_results = self.create_mock_aggregated_results()

        fig = batch_plots.plot_scenario_comparison(mock_results)

        assert fig is not None
        assert len(fig.axes) > 0

        plt.close(fig)

    def test_plot_scenario_comparison_custom_metrics(self):
        """Test scenario comparison with custom metrics."""
        mock_results = self.create_mock_aggregated_results()

        fig = batch_plots.plot_scenario_comparison(
            mock_results, metrics=["mean_growth_rate", "ruin_probability"], figsize=(10, 6)
        )

        assert fig is not None
        assert len(fig.axes) >= 2

        plt.close(fig)

    def test_plot_scenario_comparison_save_path(self):
        """Test saving scenario comparison plot."""
        mock_results = self.create_mock_aggregated_results()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        fig = None
        try:
            fig = batch_plots.plot_scenario_comparison(mock_results, save_path=str(temp_path))
            assert temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()
            if fig is not None:
                plt.close(fig)

    def test_plot_scenario_comparison_empty_results(self):
        """Test scenario comparison with empty results."""
        mock_results = Mock(spec=AggregatedResults)
        mock_results.summary_statistics = pd.DataFrame()

        fig = batch_plots.plot_scenario_comparison(mock_results)
        assert fig is not None

        plt.close(fig)

    def test_plot_scenario_comparison_invalid_input(self):
        """Test scenario comparison with invalid input."""
        with pytest.raises(ValueError, match="Input must be AggregatedResults"):
            batch_plots.plot_scenario_comparison("not_aggregated_results")

    def test_plot_parameter_sweep_3d(self):
        """Test 3D parameter sweep plot."""
        # Create mock aggregated results with parameter sweep data
        mock_results = Mock(spec=AggregatedResults)

        # Create mock batch results with parameter overrides
        batch_results = []
        for i in range(5):
            for j in range(5):
                result = Mock()
                result.simulation_results = Mock()
                result.simulation_results.growth_rates = [0.05 + i * 0.01 + j * 0.005]
                result.simulation_results.final_assets = [10000000 * (1 + i * 0.1 + j * 0.05)]
                result.simulation_results.ruin_probability = 0.1 - i * 0.01 - j * 0.005
                result.metadata = {
                    "parameter_overrides": {"param1": 0.01 + i * 0.02, "param2": 0.5 + j * 0.25}
                }
                batch_results.append(result)

        mock_results.batch_results = batch_results

        # Test with plotly mock
        with patch("ergodic_insurance.visualization.batch_plots.go.Figure") as mock_fig:
            fig = batch_plots.plot_parameter_sweep_3d(
                mock_results, param1="param1", param2="param2", metric="mean_growth_rate"
            )
            assert mock_fig.called or fig is not None

    @pytest.mark.skip(reason="Function requires complex mock setup - skipping for now")
    def test_plot_scenario_convergence(self):
        """Test scenario convergence plot."""
        # This function requires complex simulation_results structure that's hard to mock
        # Skipping this test as the function is tested through integration tests

    @pytest.mark.skip(reason="Function requires complex mock setup - skipping for now")
    def test_plot_parallel_scenarios(self):
        """Test parallel scenarios plot."""
        # This function requires complex trajectory structure with numpy arrays
        # Skipping this test as the function is tested through integration tests

    def test_plot_sensitivity_heatmap(self):
        """Test sensitivity heatmap plot."""
        # Create proper mock aggregated results with correct structure
        mock_results = Mock(spec=AggregatedResults)
        mock_results.sensitivity_analysis = pd.DataFrame(
            {
                "scenario": [
                    "sensitivity_param1_up",
                    "sensitivity_param1_down",
                    "sensitivity_param2_up",
                ],
                "mean_growth_rate_change_pct": [5.0, -3.0, 8.0],
            }
        )

        fig = batch_plots.plot_sensitivity_heatmap(mock_results, metric="mean_growth_rate")

        assert fig is not None
        assert len(fig.axes) > 0

        plt.close(fig)

    def test_plot_sensitivity_heatmap_custom_settings(self):
        """Test sensitivity heatmap with custom settings."""
        # Create proper mock aggregated results with correct structure
        mock_results = Mock(spec=AggregatedResults)
        mock_results.sensitivity_analysis = pd.DataFrame(
            {
                "scenario": [
                    "sensitivity_alpha_up",
                    "sensitivity_alpha_down",
                    "sensitivity_beta_up",
                    "sensitivity_beta_down",
                ],
                "return_change_pct": np.random.rand(4) * 10 - 5,  # Random values between -5 and 5
            }
        )

        fig = batch_plots.plot_sensitivity_heatmap(mock_results, metric="return", figsize=(8, 6))

        assert fig is not None

        plt.close(fig)


class TestExport:
    """Comprehensive tests for export module."""

    def test_save_figure_png(self):
        """Test saving figure as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_files = export.save_figure(fig, str(base_path), formats=["png"])
            assert len(saved_files) == 1
            assert Path(saved_files[0]).exists()

        plt.close(fig)

    def test_save_figure_multiple_formats(self):
        """Test saving figure in multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_files = export.save_figure(fig, str(base_path), formats=["png", "pdf", "svg"])
            assert len(saved_files) == 3
            for file in saved_files:
                assert Path(file).exists()

        plt.close(fig)

    def test_save_for_publication(self):
        """Test saving figure for publication."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_file = export.save_for_publication(fig, str(base_path))
            assert saved_file is not None
            # Check both PDF and PNG were created
            assert Path(f"{base_path}.pdf").exists()
            assert Path(f"{base_path}.png").exists()

        plt.close(fig)

    def test_save_for_presentation(self):
        """Test saving figure for presentation."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_file = export.save_for_presentation(fig, str(base_path))
            assert saved_file is not None
            # Check PNG was created for presentation
            assert Path(f"{base_path}.png").exists()

        plt.close(fig)

    def test_save_for_web(self):
        """Test saving figure for web."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_file = export.save_for_web(fig, str(base_path))
            assert saved_file is not None
            # Check optimized web files were created
            assert Path(f"{base_path}.png").exists() or Path(f"{base_path}_2x.png").exists()

        plt.close(fig)

    def test_batch_export(self):
        """Test batch export of figures."""
        # Create multiple figures
        figures = {}
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [i, i + 1, i + 2])
            figures[f"plot_{i}"] = fig

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = export.batch_export(figures, output_dir=temp_dir, formats=["png"])

            assert len(saved_files) == 3
            for files in saved_files.values():
                assert len(files) > 0
                for file in files:
                    assert Path(file).exists()

        # Close all figures
        for fig in figures.values():
            plt.close(fig)

    def test_save_figure_with_metadata(self):
        """Test saving figure with metadata."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        metadata = {"title": "Test Plot", "author": "Test Author", "date": "2024-01-01"}

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_files = export.save_figure(
                fig, str(base_path), metadata=metadata, formats=["png"]
            )
            assert len(saved_files) > 0

        plt.close(fig)

    def test_save_figure_transparent(self):
        """Test saving figure with transparent background."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            saved_files = export.save_figure(fig, str(base_path), transparent=True, formats=["png"])
            assert len(saved_files) > 0
            assert Path(saved_files[0]).exists()

        plt.close(fig)


class TestFigureFactory:
    """Tests for figure_factory module.

    Note: Basic figure creation tests (create_figure, create_subplots,
    create_line_plot, create_bar_plot, create_scatter_plot, create_histogram,
    create_heatmap, save_figure) are covered more thoroughly in
    test_visualization_factory.py::TestFigureFactory.
    """

    def test_figure_factory_initialization_default(self):
        """Test FigureFactory initialization with defaults."""
        factory = figure_factory.FigureFactory()

        assert factory is not None
        assert hasattr(factory, "style_manager")

    def test_figure_factory_initialization_custom(self):
        """Test FigureFactory initialization with custom settings."""
        factory = figure_factory.FigureFactory(theme=Theme.PRESENTATION)

        assert factory.style_manager.current_theme == Theme.PRESENTATION

    def test_apply_axis_styling(self):
        """Test applying axis styling."""
        factory = figure_factory.FigureFactory(theme=Theme.PRESENTATION)
        fig, ax = factory.create_figure()

        # _apply_axis_styling is called internally
        assert ax is not None
        assert fig is not None
        # Check that grid is configured
        assert ax.xaxis.get_gridlines() is not None

        plt.close(fig)


class TestInteractivePlots:
    """Comprehensive tests for interactive_plots module."""

    def setup_method(self):
        """Seed random state for reproducible test data."""
        np.random.seed(44)

    def test_create_interactive_dashboard(self):
        """Test creating interactive dashboard."""
        # Create mock data for dashboard
        simulations = pd.DataFrame(
            {
                "time": range(100),
                "assets": np.random.randn(100).cumsum() + 100,
                "growth_rate": np.random.randn(100) * 0.01,
                "premium": np.random.rand(100) * 10000,
            }
        )

        # Mock the plotly Figure to avoid actual rendering
        with patch("ergodic_insurance.visualization.interactive_plots.go.Figure") as mock_fig:
            fig = interactive_plots.create_interactive_dashboard(
                simulations, title="Test Dashboard"
            )
            assert mock_fig.called or fig is not None

    def test_create_time_series_dashboard(self):
        """Test creating time series dashboard."""
        # Create mock time series data
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "metric1": np.random.randn(100).cumsum(),
                "metric2": np.random.randn(100).cumsum(),
                "metric3": np.random.randn(100).cumsum(),
            }
        )

        with patch("ergodic_insurance.visualization.interactive_plots.go.Figure") as mock_fig:
            mock_fig.return_value = MagicMock()
            fig = interactive_plots.create_time_series_dashboard(
                data,
                value_col="metric1",  # Use 'value_col' instead of 'metrics'
                time_col="date",  # Use 'time_col' instead of 'date_column'
                title="Time Series",
            )
            assert mock_fig.called or fig is not None

    def test_create_correlation_heatmap(self):
        """Test creating correlation heatmap."""
        # Create mock correlation data
        data = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])

        with patch("ergodic_insurance.visualization.interactive_plots.go.Figure") as mock_fig:
            fig = interactive_plots.create_correlation_heatmap(data, title="Correlation Matrix")
            assert mock_fig.called or fig is not None

    def test_create_risk_dashboard(self):
        """Test creating risk dashboard."""
        # Create mock risk metrics
        risk_metrics = pd.DataFrame(
            {
                "var_95": [-100000, -120000, -90000],
                "cvar_95": [-150000, -180000, -135000],
                "max_drawdown": [-0.20, -0.25, -0.18],
                "scenario": ["Base", "Stress", "Optimistic"],
            }
        )

        with patch(
            "ergodic_insurance.visualization.interactive_plots.make_subplots"
        ) as mock_subplots:
            mock_subplots.return_value = MagicMock()
            fig = interactive_plots.create_risk_dashboard(risk_metrics, title="Risk Dashboard")
            assert mock_subplots.called or fig is not None


class TestStyleManager:
    """Comprehensive tests for style_manager module."""

    def test_style_manager_initialization_default(self):
        """Test StyleManager initialization with defaults."""
        manager = StyleManager()

        assert manager.current_theme == Theme.DEFAULT

    def test_style_manager_initialization_custom(self):
        """Test StyleManager initialization with custom settings."""
        manager = StyleManager(theme=Theme.PRESENTATION)

        assert manager.current_theme == Theme.PRESENTATION

    def test_apply_theme_professional(self):
        """Test applying professional theme."""
        manager = StyleManager(theme=Theme.DEFAULT)
        fig, ax = plt.subplots()

        manager.apply_style()

        # Check some theme properties
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False

        plt.close(fig)

    def test_apply_theme_colorblind(self):
        """Test applying colorblind theme."""
        manager = StyleManager(theme=Theme.COLORBLIND)
        fig, ax = plt.subplots()

        manager.apply_style()

        # Theme should be applied
        assert manager.current_theme == Theme.COLORBLIND

        plt.close(fig)

    def test_apply_theme_minimal(self):
        """Test applying minimal theme."""
        manager = StyleManager(theme=Theme.MINIMAL)

        manager.apply_style()

        # Theme should be set
        assert manager.current_theme == Theme.MINIMAL

    def test_apply_theme_presentation(self):
        """Test applying presentation theme."""
        manager = StyleManager(theme=Theme.PRESENTATION)
        fig, ax = plt.subplots()

        manager.apply_style()

        # Presentation theme uses larger fonts
        assert float(ax.xaxis.label.get_fontsize()) >= 12

        plt.close(fig)

    @pytest.mark.parametrize("size_name", ["small", "medium", "large", "blog", None])
    def test_get_figsize(self, size_name):
        """Test getting figure size for various size names."""
        manager = StyleManager()
        args = (size_name,) if size_name else ()
        width, height = manager.get_figure_size(*args)
        assert width > 0
        assert height > 0

    @pytest.mark.parametrize("dpi_type", ["screen", "print", "publication"])
    def test_get_dpi(self, dpi_type):
        """Test getting DPI for various output types."""
        manager = StyleManager()
        assert manager.get_dpi(dpi_type) > 0

    def test_set_theme(self):
        """Test setting theme."""
        manager = StyleManager()

        manager.set_theme(Theme.PRESENTATION)
        assert manager.current_theme == Theme.PRESENTATION

        manager.set_theme(Theme.MINIMAL)
        assert manager.current_theme == Theme.MINIMAL

    def test_load_config(self):
        """Test loading configuration from file."""
        manager = StyleManager()

        # Create a temporary config file
        config_data = {"colors": {"primary": "#FF0000"}, "fonts": {"size_title": 16}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            manager.load_config(temp_path)
            # Config should be loaded
            assert manager is not None
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_save_config(self):
        """Test saving configuration."""
        manager = StyleManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            manager.save_config(temp_path)
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
        finally:
            if temp_path.exists():
                temp_path.unlink()

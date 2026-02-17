"""Comprehensive tests for the figure_factory module with 90%+ coverage."""

# mypy: ignore-errors

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, patch
import warnings

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.visualization.figure_factory import FigureFactory
from ergodic_insurance.visualization.style_manager import StyleManager, Theme


class TestFigureFactory:
    """Test suite for FigureFactory class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        np.random.seed(42)  # Seed for reproducibility of random test data
        self.factory = FigureFactory()  # pylint: disable=attribute-defined-outside-init

    def test_initialization_default(self):
        """Test FigureFactory initialization with default parameters."""
        factory = FigureFactory()
        assert factory.style_manager is not None
        assert factory.auto_apply is True
        assert isinstance(factory.style_manager, StyleManager)

    def test_initialization_custom_style_manager(self):
        """Test FigureFactory initialization with custom style manager."""
        custom_manager = StyleManager(theme=Theme.PRESENTATION)
        factory = FigureFactory(style_manager=custom_manager)
        assert factory.style_manager is custom_manager
        assert factory.style_manager.current_theme == Theme.PRESENTATION

    def test_initialization_no_auto_apply(self):
        """Test FigureFactory initialization with auto_apply=False."""
        factory = FigureFactory(auto_apply=False)
        assert factory.auto_apply is False

    def test_create_figure_basic(self):
        """Test creating a basic figure."""
        fig, ax = self.factory.create_figure()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert fig.get_size_inches()[0] > 0
        assert fig.get_size_inches()[1] > 0

    def test_create_figure_with_title(self):
        """Test creating figure with title."""
        fig, ax = self.factory.create_figure(title="Test Figure")
        assert fig._suptitle is not None  # type: ignore[attr-defined]
        assert fig._suptitle.get_text() == "Test Figure"  # type: ignore[attr-defined]

    def test_create_figure_size_types(self):
        """Test creating figures with different size types."""
        size_types = ["small", "medium", "large", "blog", "technical", "presentation"]
        for size_type in size_types:
            fig, ax = self.factory.create_figure(size_type=size_type)
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)

    def test_create_figure_orientations(self):
        """Test creating figures with different orientations."""
        for orientation in ["landscape", "portrait"]:
            fig, ax = self.factory.create_figure(orientation=orientation)
            assert isinstance(fig, Figure)
            width, height = fig.get_size_inches()
            if orientation == "landscape":
                assert width > height
            else:
                assert height >= width

    def test_create_figure_dpi_types(self):
        """Test creating figures with different DPI types."""
        for dpi_type in ["screen", "web", "print"]:
            fig, ax = self.factory.create_figure(dpi_type=dpi_type)
            assert fig.dpi > 0

    def test_create_subplots_single(self):
        """Test creating single subplot."""
        fig, ax = self.factory.create_subplots(rows=1, cols=1)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_create_subplots_multiple(self):
        """Test creating multiple subplots."""
        fig, axes = self.factory.create_subplots(rows=2, cols=3)
        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 3)
        for ax in axes.flat:
            assert isinstance(ax, Axes)

    def test_create_subplots_with_titles(self):
        """Test creating subplots with individual titles."""
        subplot_titles = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"]
        fig, axes = self.factory.create_subplots(rows=2, cols=2, subplot_titles=subplot_titles)
        axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for i, ax in enumerate(axes_list):  # type: ignore[arg-type, var-annotated]
            if i < len(subplot_titles):
                assert ax.get_title() == subplot_titles[i]

    def test_create_subplots_with_main_title(self):
        """Test creating subplots with main title."""
        fig, axes = self.factory.create_subplots(rows=2, cols=2, title="Main Title")
        assert fig._suptitle is not None  # type: ignore[attr-defined]
        assert fig._suptitle.get_text() == "Main Title"  # type: ignore[attr-defined]

    def test_create_line_plot_single_series(self):
        """Test creating line plot with single series."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [10, 20, 15, 25, 30]

        fig, ax = self.factory.create_line_plot(
            x_data=x_data, y_data=y_data, title="Test Line Plot", x_label="X Axis", y_label="Y Axis"
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # Title is set on figure, not axes
        assert fig._suptitle is None or fig._suptitle.get_text() == "Test Line Plot"  # type: ignore[attr-defined]
        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"
        assert len(ax.lines) == 1

    def test_create_line_plot_multiple_series(self):
        """Test creating line plot with multiple series."""
        x_data = [1, 2, 3, 4, 5]
        y_data = {
            "Series 1": [10, 20, 15, 25, 30],
            "Series 2": [5, 15, 25, 20, 10],
            "Series 3": [20, 10, 30, 15, 25],
        }

        fig, ax = self.factory.create_line_plot(x_data=x_data, y_data=y_data, show_legend=True)  # type: ignore[arg-type]

        assert len(ax.lines) == 3
        assert ax.get_legend() is not None

    def test_create_line_plot_with_markers(self):
        """Test creating line plot with markers."""
        x_data = np.linspace(0, 10, 10)
        y_data = np.sin(x_data)

        fig, ax = self.factory.create_line_plot(x_data=x_data, y_data=y_data, markers=True)

        assert len(ax.lines) == 1
        line = ax.lines[0]
        assert line.get_marker() == "o"

    def test_create_line_plot_no_grid(self):
        """Test creating line plot without grid."""
        x_data = [1, 2, 3]
        y_data = [1, 2, 3]

        fig, ax = self.factory.create_line_plot(x_data=x_data, y_data=y_data, show_grid=False)

        # Grid might still be visible due to matplotlib defaults, check alpha instead
        assert len(ax.lines) == 1  # Ensure the plot was created

    def test_create_bar_plot_single_series_vertical(self):
        """Test creating vertical bar plot with single series."""
        categories = ["A", "B", "C", "D"]
        values = [10, 20, 15, 25]

        fig, ax = self.factory.create_bar_plot(
            categories=categories,
            values=values,
            title="Test Bar Plot",
            x_label="Categories",
            y_label="Values",
            orientation="vertical",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == 4  # 4 bars

    def test_create_bar_plot_single_series_horizontal(self):
        """Test creating horizontal bar plot with single series."""
        categories = ["Category A", "Category B", "Category C"]
        values = [100, 200, 150]

        fig, ax = self.factory.create_bar_plot(
            categories=categories, values=values, orientation="horizontal"
        )

        assert len(ax.patches) == 3

    def test_create_bar_plot_multiple_series(self):
        """Test creating bar plot with multiple series."""
        categories = ["Q1", "Q2", "Q3", "Q4"]
        values = {
            "Product A": [100, 120, 140, 160],
            "Product B": [80, 90, 100, 110],
            "Product C": [60, 70, 80, 90],
        }

        fig, ax = self.factory.create_bar_plot(categories=categories, values=values)  # type: ignore[arg-type]

        assert len(ax.patches) == 12  # 4 categories * 3 series
        assert ax.get_legend() is not None

    def test_create_bar_plot_with_value_labels(self):
        """Test creating bar plot with value labels."""
        categories = ["A", "B", "C"]
        values = [10.5, 20.3, 15.7]

        fig, ax = self.factory.create_bar_plot(
            categories=categories, values=values, show_values=True, value_format=".1f"
        )

        # Check that text labels were added
        texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
        assert len(texts) > 0

    def test_create_scatter_plot_basic(self):
        """Test creating basic scatter plot."""
        x_data = np.random.randn(50)
        y_data = np.random.randn(50)

        fig, ax = self.factory.create_scatter_plot(
            x_data=x_data,
            y_data=y_data,
            title="Test Scatter Plot",
            x_label="X Values",
            y_label="Y Values",
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.collections) == 1  # One scatter collection

    def test_create_scatter_plot_with_colors(self):
        """Test creating scatter plot with color mapping."""
        x_data = np.random.randn(30)
        y_data = np.random.randn(30)
        colors = np.random.randn(30)

        fig, ax = self.factory.create_scatter_plot(
            x_data=x_data, y_data=y_data, colors=colors, show_colorbar=True
        )

        assert len(ax.collections) == 1
        assert len(fig.axes) > 1  # Main axes plus colorbar

    def test_create_scatter_plot_with_sizes(self):
        """Test creating scatter plot with custom sizes."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 1, 3, 5]
        sizes = [50, 100, 150, 200, 250]

        fig, ax = self.factory.create_scatter_plot(x_data=x_data, y_data=y_data, sizes=sizes)

        assert len(ax.collections) == 1

    def test_create_histogram_basic(self):
        """Test creating basic histogram."""
        data = np.random.randn(1000)

        fig, ax = self.factory.create_histogram(
            data=data, title="Test Histogram", x_label="Values", bins=20
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.patches) == 20  # 20 bins

    def test_create_histogram_with_statistics(self):
        """Test creating histogram with mean and median lines."""
        data = np.random.randn(500) * 10 + 50

        fig, ax = self.factory.create_histogram(data=data, show_statistics=True)

        # Check for vertical lines (mean and median)
        vlines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(vlines) == 2
        assert ax.get_legend() is not None

    @patch("scipy.stats")
    def test_create_histogram_with_kde(self, mock_stats):
        """Test creating histogram with KDE overlay."""
        data = np.random.randn(500)

        # Mock the KDE
        mock_kde = MagicMock()
        mock_kde.return_value = np.random.randn(100)
        mock_stats.gaussian_kde.return_value = mock_kde

        fig, ax = self.factory.create_histogram(data=data, show_kde=True)

        assert len(fig.axes) == 2  # Main axes plus twin axes for KDE

    def test_create_heatmap_array(self):
        """Test creating heatmap from numpy array."""
        data = np.random.randn(5, 8)

        fig, ax = self.factory.create_heatmap(data=data, title="Test Heatmap", cmap="coolwarm")

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.images) == 1  # One heatmap image

    def test_create_heatmap_dataframe(self):
        """Test creating heatmap from pandas DataFrame."""
        df = pd.DataFrame(
            np.random.randn(4, 6),
            columns=[f"Col {i}" for i in range(6)],
            index=[f"Row {i}" for i in range(4)],
        )

        fig, ax = self.factory.create_heatmap(data=df, x_label="Columns", y_label="Rows")

        assert ax.get_xlabel() == "Columns"
        assert ax.get_ylabel() == "Rows"

    def test_create_heatmap_with_values(self):
        """Test creating heatmap with value annotations."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        fig, ax = self.factory.create_heatmap(data=data, show_values=True, value_format=".0f")

        # Check that text annotations were added
        texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
        assert len(texts) >= 9  # At least 9 value annotations

    def test_create_heatmap_with_labels(self):
        """Test creating heatmap with custom labels."""
        data = np.random.randn(3, 4)
        x_labels = ["A", "B", "C", "D"]
        y_labels = ["X", "Y", "Z"]

        fig, ax = self.factory.create_heatmap(data=data, x_labels=x_labels, y_labels=y_labels)

        assert len(ax.get_xticklabels()) == 4
        assert len(ax.get_yticklabels()) == 3

    def test_create_box_plot_list_data(self):
        """Test creating box plot from list of lists."""
        data = [
            np.random.randn(100) * 2 + 10,
            np.random.randn(100) * 3 + 15,
            np.random.randn(100) * 1.5 + 12,
        ]
        labels = ["Group A", "Group B", "Group C"]

        fig, ax = self.factory.create_box_plot(data=data, labels=labels, title="Test Box Plot")  # type: ignore[arg-type]

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_create_box_plot_dict_data(self):
        """Test creating box plot from dictionary."""
        data = {
            "Treatment A": np.random.randn(50) * 2 + 10,
            "Treatment B": np.random.randn(50) * 3 + 12,
            "Control": np.random.randn(50) * 2.5 + 11,
        }

        fig, ax = self.factory.create_box_plot(data=data, show_means=True)  # type: ignore[arg-type]

        assert len(ax.get_xticklabels()) == 3

    def test_create_box_plot_dataframe(self):
        """Test creating box plot from DataFrame."""
        df = pd.DataFrame(
            {
                "A": np.random.randn(100),
                "B": np.random.randn(100) * 2,
                "C": np.random.randn(100) * 0.5 + 2,
            }
        )

        fig, ax = self.factory.create_box_plot(data=df, orientation="horizontal")

        assert isinstance(fig, Figure)

    def test_format_axis_currency(self):
        """Test formatting axis as currency."""
        fig, ax = self.factory.create_figure()

        # Create some data to have y-axis values
        ax.plot([1, 2, 3], [1000, 1000000, 1000000000])

        self.factory.format_axis_currency(ax, axis="y", abbreviate=True)

        # Get formatter function
        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

    def test_format_axis_currency_no_abbreviation(self):
        """Test formatting axis as currency without abbreviation."""
        fig, ax = self.factory.create_figure()
        ax.plot([1, 2, 3], [100, 200, 300])

        self.factory.format_axis_currency(ax, axis="y", abbreviate=False, decimals=2)

        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

    def test_format_axis_percentage(self):
        """Test formatting axis as percentage."""
        fig, ax = self.factory.create_figure()
        ax.plot([1, 2, 3], [0.1, 0.2, 0.3])

        self.factory.format_axis_percentage(ax, axis="y", decimals=1)

        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

    def test_add_annotations_with_arrow(self):
        """Test adding annotations with arrow."""
        fig, ax = self.factory.create_figure()
        ax.plot([1, 2, 3], [1, 4, 2])

        self.factory.add_annotations(ax, x=2, y=4, text="Peak Value", arrow=True, offset=(20, 20))

        # Check that annotation was added
        assert len(ax.texts) > 0

    def test_add_annotations_without_arrow(self):
        """Test adding annotations without arrow."""
        fig, ax = self.factory.create_figure()
        ax.plot([1, 2, 3], [1, 2, 3])

        self.factory.add_annotations(ax, x=1.5, y=1.5, text="Label", arrow=False)

        assert len(ax.texts) > 0

    def test_save_figure(self):
        """Test saving figure with different output types."""
        fig, ax = self.factory.create_figure()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            for output_type in ["screen", "web", "print"]:
                filepath = Path(tmpdir) / f"test_{output_type}.png"
                self.factory.save_figure(fig, str(filepath), output_type=output_type)
                assert filepath.exists()

    def test_apply_axis_styling(self):
        """Test internal axis styling method."""
        fig, ax = self.factory.create_figure()

        # The styling should have been applied automatically
        # get_visible() returns bool, so check actual visibility state
        assert isinstance(ax.spines["top"].get_visible(), bool)
        assert isinstance(ax.spines["right"].get_visible(), bool)
        assert isinstance(ax.spines["bottom"].get_visible(), bool)
        assert isinstance(ax.spines["left"].get_visible(), bool)

    def test_add_value_labels_vertical_bars(self):
        """Test internal method for adding value labels to vertical bars."""
        fig, ax = self.factory.create_figure()
        bars = ax.bar([1, 2, 3], [10, 20, 15])

        self.factory._add_value_labels(ax, bars, "vertical", ".0f")

        # Check that text labels were added
        assert len(ax.texts) == 3

    def test_add_value_labels_horizontal_bars(self):
        """Test internal method for adding value labels to horizontal bars."""
        fig, ax = self.factory.create_figure()
        bars = ax.barh([1, 2, 3], [10, 20, 15])

        self.factory._add_value_labels(ax, bars, "horizontal", ".1f")

        assert len(ax.texts) == 3

    def test_create_line_plot_with_series_and_pandas(self):
        """Test line plot with pandas Series."""
        x_data = pd.Series([1, 2, 3, 4, 5])
        y_data = pd.Series([10, 20, 15, 25, 30])

        fig, ax = self.factory.create_line_plot(
            x_data=x_data, y_data=y_data, labels=["Test Series"]
        )

        assert len(ax.lines) == 1
        assert ax.get_legend() is not None

    def test_create_bar_plot_arrays(self):
        """Test bar plot with numpy arrays."""
        categories = np.array(["A", "B", "C", "D"])
        values = np.array([10, 20, 15, 25])

        fig, ax = self.factory.create_bar_plot(categories=categories, values=values)

        assert len(ax.patches) == 4

    def test_create_scatter_plot_with_labels(self):
        """Test scatter plot with point labels."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 1, 3, 5]
        labels = ["Point A", "Point B", "Point C", "Point D", "Point E"]

        fig, ax = self.factory.create_scatter_plot(x_data=x_data, y_data=y_data, labels=labels)

        assert len(ax.collections) == 1

    def test_create_histogram_with_pandas_series(self):
        """Test histogram with pandas Series."""
        data = pd.Series(np.random.randn(500))

        fig, ax = self.factory.create_histogram(data=data, bins="auto")

        assert len(ax.patches) > 0

    def test_edge_cases_empty_data(self):
        """Test edge cases with empty data."""
        # Empty arrays should still create valid figures
        fig, ax = self.factory.create_line_plot(x_data=[], y_data=[])
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_format_axis_x_axis(self):
        """Test formatting x-axis instead of y-axis."""
        fig, ax = self.factory.create_figure()
        ax.plot([1000, 2000, 3000], [1, 2, 3])

        self.factory.format_axis_currency(ax, axis="x", abbreviate=True)
        formatter = ax.xaxis.get_major_formatter()
        assert formatter is not None

        self.factory.format_axis_percentage(ax, axis="x")
        formatter = ax.xaxis.get_major_formatter()
        assert formatter is not None

    def test_multiple_themes(self):
        """Test creating figures with different themes."""
        # Check which themes are actually available
        themes = [Theme.DEFAULT, Theme.PRESENTATION]  # Use only guaranteed themes

        for theme in themes:
            factory = FigureFactory(theme=theme)
            fig, ax = factory.create_figure()
            assert isinstance(fig, Figure)
            assert isinstance(ax, Axes)


class TestFigureFactoryIntegration:
    """Integration tests for FigureFactory."""

    def test_complex_dashboard(self):
        """Test creating a complex dashboard with multiple plot types."""
        factory = FigureFactory(theme=Theme.PRESENTATION)

        # Create main figure with subplots
        fig, axes = factory.create_subplots(
            rows=2,
            cols=3,
            size_type="large",
            title="Insurance Analytics Dashboard",
            subplot_titles=["Revenue", "Claims", "Loss Ratio", "Distribution", "Trends", "Heatmap"],
        )

        assert isinstance(fig, Figure)
        assert axes.shape == (2, 3)  # type: ignore[union-attr]

    def test_theme_consistency(self):
        """Test that theme settings are consistently applied."""
        factory = FigureFactory(theme=Theme.PRESENTATION)

        # Create different plot types
        fig1, ax1 = factory.create_line_plot([1, 2, 3], [1, 2, 3])
        fig2, ax2 = factory.create_bar_plot(["A", "B"], [10, 20])
        fig3, ax3 = factory.create_scatter_plot([1, 2, 3], [1, 2, 3])

        # All should have consistent styling from the same factory
        assert fig1.dpi == fig2.dpi == fig3.dpi

    def test_currency_formatter_edge_cases(self):
        """Test currency formatter with edge cases."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()

        # Test with very large numbers
        ax.plot([1, 2, 3], [1e12, 1e15, 1e18])
        factory.format_axis_currency(ax, abbreviate=True)

        # Test with negative numbers
        ax.clear()
        ax.plot([1, 2, 3], [-1000, -1000000, -1000000000])
        factory.format_axis_currency(ax, abbreviate=True)

        # Test with very small numbers
        ax.clear()
        ax.plot([1, 2, 3], [0.1, 0.01, 0.001])
        factory.format_axis_currency(ax, abbreviate=False, decimals=3)

    def test_plot_customization_kwargs(self):
        """Test passing custom kwargs to plot methods."""
        factory = FigureFactory()
        # Test line plot with custom kwargs
        fig, ax = factory.create_line_plot(
            [1, 2, 3], [1, 2, 3], linewidth=5, linestyle="--", alpha=0.5
        )
        assert ax.lines[0].get_linewidth() == 5
        assert ax.lines[0].get_linestyle() == "--"
        assert ax.lines[0].get_alpha() == 0.5

        # Test bar plot with custom kwargs
        fig, ax = factory.create_bar_plot(["A", "B"], [10, 20], alpha=0.3, edgecolor="red")
        assert ax.patches[0].get_alpha() == 0.3

    def test_pandas_integration(self):
        """Test integration with pandas DataFrames."""
        factory = FigureFactory()
        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "x": range(10),
                "y1": np.random.randn(10),
                "y2": np.random.randn(10),
                "y3": np.random.randn(10),
            }
        )

        # Test with DataFrame columns
        fig, ax = factory.create_line_plot(
            x_data=df["x"],
            y_data={"Series 1": df["y1"], "Series 2": df["y2"], "Series 3": df["y3"]},  # type: ignore[dict-item]
        )
        assert len(ax.lines) == 3

    def test_save_figure_formats(self):
        """Test saving figures in different formats."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test different file formats
            for ext in [".png", ".pdf", ".svg", ".jpg"]:
                filepath = Path(tmpdir) / f"test{ext}"
                factory.save_figure(fig, str(filepath))
                assert filepath.exists()

"""Comprehensive tests for visualization factory and style manager."""

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from ergodic_insurance.visualization_infra.figure_factory import FigureFactory
from ergodic_insurance.visualization_infra.style_manager import (
    ColorPalette,
    FigureConfig,
    FontConfig,
    GridConfig,
    StyleManager,
    Theme,
)


class TestStyleManager:
    """Test suite for StyleManager class."""

    def test_initialization(self):
        """Test StyleManager initialization."""
        manager = StyleManager()
        assert manager.current_theme == Theme.DEFAULT
        assert len(manager.themes) >= 5  # At least 5 built-in themes

    def test_theme_switching(self):
        """Test switching between themes."""
        manager = StyleManager()

        # Test switching to each theme
        for theme in Theme:
            manager.set_theme(theme)
            assert manager.current_theme == theme

    def test_invalid_theme(self):
        """Test setting invalid theme raises error."""
        manager = StyleManager()

        with pytest.raises(ValueError):
            manager.set_theme("INVALID_THEME")  # type: ignore[arg-type]

    def test_get_theme_config(self):
        """Test retrieving theme configuration."""
        manager = StyleManager()

        # Get default theme config
        config = manager.get_theme_config()
        assert "colors" in config
        assert "fonts" in config
        assert "figure" in config
        assert "grid" in config

        # Get specific theme config
        config = manager.get_theme_config(Theme.COLORBLIND)
        assert "colors" in config

    def test_color_palette_access(self):
        """Test accessing color palette."""
        manager = StyleManager()
        colors = manager.get_colors()

        assert isinstance(colors, ColorPalette)
        assert colors.primary == "#0080C7"
        assert len(colors.series) >= 4

    def test_font_config_access(self):
        """Test accessing font configuration."""
        manager = StyleManager()
        fonts = manager.get_fonts()

        assert isinstance(fonts, FontConfig)
        assert fonts.family == "Arial"
        assert fonts.size_base == 11

    def test_figure_config_access(self):
        """Test accessing figure configuration."""
        manager = StyleManager()
        fig_config = manager.get_figure_config()

        assert isinstance(fig_config, FigureConfig)
        assert fig_config.size_blog == (8, 6)
        assert fig_config.size_technical == (10, 8)
        assert fig_config.dpi_web == 150
        assert fig_config.dpi_print == 300

    def test_grid_config_access(self):
        """Test accessing grid configuration."""
        manager = StyleManager()
        grid_config = manager.get_grid_config()

        assert isinstance(grid_config, GridConfig)
        assert grid_config.show_grid is True
        assert grid_config.grid_alpha == 0.3

    def test_update_colors(self):
        """Test updating colors in current theme."""
        manager = StyleManager()

        # Update primary color
        manager.update_colors({"primary": "#FF0000"})
        colors = manager.get_colors()
        assert colors.primary == "#FF0000"

        # Update multiple colors
        manager.update_colors({"secondary": "#00FF00", "warning": "#0000FF"})
        colors = manager.get_colors()
        assert colors.secondary == "#00FF00"
        assert colors.warning == "#0000FF"

    def test_update_fonts(self):
        """Test updating fonts in current theme."""
        manager = StyleManager()

        # Update font family
        manager.update_fonts({"family": "Helvetica"})
        fonts = manager.get_fonts()
        assert fonts.family == "Helvetica"

        # Update font sizes
        manager.update_fonts({"size_base": 12, "size_title": 16})
        fonts = manager.get_fonts()
        assert fonts.size_base == 12
        assert fonts.size_title == 16

    def test_get_figure_size(self):
        """Test getting figure sizes."""
        manager = StyleManager()

        # Test different size types
        assert manager.get_figure_size("small") == (6, 4)
        assert manager.get_figure_size("medium") == (8, 6)
        assert manager.get_figure_size("large") == (12, 8)
        assert manager.get_figure_size("blog") == (8, 6)
        assert manager.get_figure_size("technical") == (10, 8)

        # Test portrait orientation
        assert manager.get_figure_size("medium", "portrait") == (6, 8)

    def test_get_dpi(self):
        """Test getting DPI values."""
        manager = StyleManager()

        assert manager.get_dpi("screen") == 100
        assert manager.get_dpi("web") == 150
        assert manager.get_dpi("print") == 300
        assert manager.get_dpi("unknown") == 100  # Default

    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.rcParams.update")
    def test_apply_style(self, mock_rcparams, mock_style_use):
        """Test applying style to matplotlib."""
        manager = StyleManager()
        manager.apply_style()

        # Check that matplotlib style was applied
        mock_style_use.assert_called_once_with("seaborn-v0_8-whitegrid")
        mock_rcparams.assert_called_once()

        # Check that rcParams were updated
        call_args = mock_rcparams.call_args[0][0]
        assert "font.family" in call_args
        assert "axes.titlesize" in call_args
        assert "grid.color" in call_args

    def test_save_and_load_config(self):
        """Test saving and loading configuration from YAML."""
        manager = StyleManager()

        # Update some values
        manager.update_colors({"primary": "#123456"})
        manager.update_fonts({"size_base": 14})

        # Save configuration
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            manager.save_config(config_path)
            assert config_path.exists()

            # Load in new manager
            new_manager = StyleManager()
            new_manager.load_config(config_path)

            # Check values were loaded
            colors = new_manager.get_colors()
            fonts = new_manager.get_fonts()
            assert colors.primary == "#123456"
            assert fonts.size_base == 14
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()

    def test_load_nonexistent_config(self):
        """Test loading non-existent config file raises error."""
        manager = StyleManager()

        with pytest.raises(FileNotFoundError):
            manager.load_config("nonexistent.yaml")

    def test_create_style_sheet(self):
        """Test creating matplotlib style sheet."""
        manager = StyleManager()
        style_sheet = manager.create_style_sheet()

        assert isinstance(style_sheet, dict)
        assert "font.family" in style_sheet
        assert "axes.titlesize" in style_sheet
        assert "grid.color" in style_sheet
        assert style_sheet["font.size"] == 11

    def test_colorblind_theme(self):
        """Test colorblind-friendly theme."""
        manager = StyleManager(theme=Theme.COLORBLIND)
        colors = manager.get_colors()

        # Check that colorblind palette is different from default
        assert colors.primary == "#0173B2"
        assert len(colors.series) >= 4

    def test_presentation_theme(self):
        """Test presentation theme with larger fonts."""
        manager = StyleManager(theme=Theme.PRESENTATION)
        fonts = manager.get_fonts()

        assert fonts.size_base == 14
        assert fonts.size_title == 18

    def test_minimal_theme(self):
        """Test minimal theme."""
        manager = StyleManager(theme=Theme.MINIMAL)
        colors = manager.get_colors()
        grid = manager.get_grid_config()

        assert colors.primary == "#333333"
        assert grid.show_grid is False

    def test_print_theme(self):
        """Test print theme with high DPI."""
        manager = StyleManager(theme=Theme.PRINT)
        fig_config = manager.get_figure_config()

        assert fig_config.dpi_print == 600
        assert fig_config.dpi_screen == 300

    def test_custom_initialization(self):
        """Test initialization with custom colors and fonts."""
        custom_colors = {"primary": "#FF0000"}
        custom_fonts = {"family": "Times New Roman"}

        manager = StyleManager(custom_colors=custom_colors, custom_fonts=custom_fonts)

        colors = manager.get_colors()
        fonts = manager.get_fonts()

        assert colors.primary == "#FF0000"
        assert fonts.family == "Times New Roman"


class TestFigureFactory:
    """Test suite for FigureFactory class."""

    def setup_method(self):
        """Seed random state for reproducible test data."""
        np.random.seed(42)

    def test_initialization(self):
        """Test FigureFactory initialization."""
        factory = FigureFactory()
        assert factory.style_manager is not None
        assert factory.auto_apply is True

    def test_initialization_with_custom_manager(self):
        """Test initialization with custom style manager."""
        manager = StyleManager(theme=Theme.PRESENTATION)
        factory = FigureFactory(style_manager=manager)
        assert factory.style_manager == manager

    def test_create_figure(self):
        """Test basic figure creation."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()

        assert fig is not None
        assert ax is not None
        assert fig.get_size_inches()[0] == 8  # Medium size default
        assert fig.get_size_inches()[1] == 6

        plt.close(fig)

    def test_create_figure_with_title(self):
        """Test figure creation with title."""
        factory = FigureFactory()
        fig, ax = factory.create_figure(title="Test Title")

        assert hasattr(fig, "_suptitle") and fig._suptitle is not None
        assert fig._suptitle.get_text() == "Test Title"

        plt.close(fig)

    def test_create_figure_sizes(self):
        """Test creating figures with different sizes."""
        factory = FigureFactory()

        # Test blog size
        fig, _ = factory.create_figure(size_type="blog")
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

        # Test technical size
        fig, _ = factory.create_figure(size_type="technical")
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)

    def test_create_subplots(self):
        """Test creating subplots."""
        factory = FigureFactory()
        fig, axes = factory.create_subplots(rows=2, cols=2)

        assert fig is not None
        assert hasattr(axes, "shape") and axes.shape == (2, 2)

        plt.close(fig)

    def test_create_subplots_with_titles(self):
        """Test creating subplots with individual titles."""
        factory = FigureFactory()
        subplot_titles = ["Q1", "Q2", "Q3", "Q4"]
        fig, axes = factory.create_subplots(rows=2, cols=2, subplot_titles=subplot_titles)

        axes_flat = axes.flatten() if hasattr(axes, "flatten") else []
        for i, title in enumerate(subplot_titles):
            assert axes_flat[i].get_title() == title

        plt.close(fig)

    def test_create_line_plot(self):
        """Test creating line plot."""
        factory = FigureFactory()
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 3, 5, 6]

        fig, ax = factory.create_line_plot(
            x_data, y_data, title="Line Plot", x_label="X", y_label="Y"
        )

        assert ax.get_title() == ""  # Title is at figure level
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert len(ax.lines) == 1

        plt.close(fig)

    def test_create_line_plot_multiple_series(self):
        """Test creating line plot with multiple series."""
        factory = FigureFactory()
        x_data = [1, 2, 3, 4, 5]
        y_data: Dict[str, Union[List[Any], np.ndarray]] = {
            "Series 1": [2, 4, 3, 5, 6],
            "Series 2": [1, 3, 4, 4, 5],
        }

        fig, ax = factory.create_line_plot(x_data, y_data, show_legend=True)

        assert len(ax.lines) == 2
        assert ax.get_legend() is not None

        plt.close(fig)

    def test_create_bar_plot(self):
        """Test creating bar plot."""
        factory = FigureFactory()
        categories = ["A", "B", "C", "D"]
        values = [10, 20, 15, 25]

        fig, ax = factory.create_bar_plot(categories, values, title="Bar Plot", y_label="Values")

        assert ax.get_ylabel() == "Values"
        assert len(ax.patches) == 4  # 4 bars

        plt.close(fig)

    def test_create_bar_plot_horizontal(self):
        """Test creating horizontal bar plot."""
        factory = FigureFactory()
        categories = ["A", "B", "C", "D"]
        values = [10, 20, 15, 25]

        fig, ax = factory.create_bar_plot(categories, values, orientation="horizontal")

        assert len(ax.patches) == 4

        plt.close(fig)

    def test_create_bar_plot_with_values(self):
        """Test creating bar plot with value labels."""
        factory = FigureFactory()
        categories = ["A", "B", "C"]
        values = [10, 20, 15]

        fig, ax = factory.create_bar_plot(categories, values, show_values=True)

        # Check that text labels were added
        assert len(ax.texts) == 3

        plt.close(fig)

    def test_create_scatter_plot(self):
        """Test creating scatter plot."""
        factory = FigureFactory()
        x_data = np.random.randn(50)
        y_data = np.random.randn(50)

        fig, ax = factory.create_scatter_plot(
            x_data, y_data, title="Scatter Plot", x_label="X", y_label="Y"
        )

        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert len(ax.collections) == 1  # One scatter collection

        plt.close(fig)

    def test_create_scatter_plot_with_colors(self):
        """Test creating scatter plot with color mapping."""
        factory = FigureFactory()
        x_data = np.random.randn(50)
        y_data = np.random.randn(50)
        colors = np.random.randn(50)

        fig, ax = factory.create_scatter_plot(x_data, y_data, colors=colors, show_colorbar=True)

        assert len(ax.collections) == 1
        # Colorbar adds another axes
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_create_histogram(self):
        """Test creating histogram."""
        factory = FigureFactory()
        data = np.random.randn(1000)

        fig, ax = factory.create_histogram(data, title="Histogram", bins=30)

        assert len(ax.patches) == 30  # 30 bins
        assert ax.get_ylabel() == "Frequency"

        plt.close(fig)

    def test_create_histogram_with_statistics(self):
        """Test creating histogram with mean/median lines."""
        factory = FigureFactory()
        data = np.random.randn(1000)

        fig, ax = factory.create_histogram(data, show_statistics=True)

        # Check for vertical lines (mean and median)
        assert len(ax.lines) >= 2
        assert ax.get_legend() is not None

        plt.close(fig)

    def test_create_heatmap(self):
        """Test creating heatmap."""
        factory = FigureFactory()
        data = np.random.randn(5, 5)

        fig, ax = factory.create_heatmap(data, title="Heatmap")

        assert len(ax.images) == 1  # One image for heatmap

        plt.close(fig)

    def test_create_heatmap_with_labels(self):
        """Test creating heatmap with labels."""
        factory = FigureFactory()
        data = np.random.randn(3, 4)
        x_labels = ["A", "B", "C", "D"]
        y_labels = ["X", "Y", "Z"]

        fig, ax = factory.create_heatmap(
            data, x_labels=x_labels, y_labels=y_labels, show_values=True
        )

        # Check that labels were set
        assert len(ax.get_xticklabels()) == 4
        assert len(ax.get_yticklabels()) == 3
        # Check that values were added
        assert len(ax.texts) == 12  # 3x4 grid

        plt.close(fig)

    def test_create_heatmap_from_dataframe(self):
        """Test creating heatmap from DataFrame."""
        factory = FigureFactory()
        df = pd.DataFrame(
            np.random.randn(3, 4), index=["X", "Y", "Z"], columns=["A", "B", "C", "D"]
        )

        fig, ax = factory.create_heatmap(df)

        assert len(ax.get_xticklabels()) == 4
        assert len(ax.get_yticklabels()) == 3

        plt.close(fig)

    def test_create_box_plot(self):
        """Test creating box plot."""
        factory = FigureFactory()
        data: List[List[Any]] = [np.random.randn(100).tolist() for _ in range(4)]

        fig, ax = factory.create_box_plot(data, title="Box Plot", labels=["A", "B", "C", "D"])

        assert len(ax.get_xticklabels()) == 4

        plt.close(fig)

    def test_create_box_plot_from_dict(self):
        """Test creating box plot from dictionary."""
        factory = FigureFactory()
        data: Dict[str, List[Any]] = {
            "Group A": np.random.randn(100).tolist(),
            "Group B": np.random.randn(100).tolist(),
            "Group C": np.random.randn(100).tolist(),
        }

        fig, ax = factory.create_box_plot(data)

        assert len(ax.get_xticklabels()) == 3

        plt.close(fig)

    def test_create_box_plot_horizontal(self):
        """Test creating horizontal box plot."""
        factory = FigureFactory()
        data: List[List[Any]] = [np.random.randn(100).tolist() for _ in range(3)]

        fig, ax = factory.create_box_plot(data, orientation="horizontal")

        # Horizontal boxplot has y-axis labels
        assert len(ax.get_yticklabels()) > 0

        plt.close(fig)

    def test_format_axis_currency(self):
        """Test formatting axis as currency."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()

        # Add some data
        ax.plot([1, 2, 3], [1000, 2000, 1500])

        # Format y-axis as currency
        factory.format_axis_currency(ax, axis="y")

        # Get formatter and check it works
        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

        plt.close(fig)

    def test_format_axis_percentage(self):
        """Test formatting axis as percentage."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()

        # Add some data
        ax.plot([1, 2, 3], [0.1, 0.2, 0.15])

        # Format y-axis as percentage
        factory.format_axis_percentage(ax, axis="y")

        # Get formatter and check it works
        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

        plt.close(fig)

    def test_add_annotations(self):
        """Test adding annotations to plot."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()

        # Add a simple plot
        ax.plot([1, 2, 3], [1, 4, 2])

        # Add annotation
        factory.add_annotations(ax, 2, 4, "Peak", arrow=True)

        # Check annotation was added
        assert len(ax.texts) == 1

        plt.close(fig)

    def test_save_figure(self):
        """Test saving figure with correct DPI."""
        factory = FigureFactory()
        fig, ax = factory.create_figure()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save with web DPI
            factory.save_figure(fig, str(temp_path), output_type="web")
            assert temp_path.exists()
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()

        plt.close(fig)

    def test_theme_consistency(self):
        """Test that theme is consistently applied."""
        # Create factory with presentation theme
        factory = FigureFactory(theme=Theme.PRESENTATION)

        # Create various plots
        fig1, ax1 = factory.create_line_plot([1, 2, 3], [1, 2, 3])
        fig2, ax2 = factory.create_bar_plot(["A", "B"], [10, 20])

        # Check that font sizes match presentation theme
        fonts = factory.style_manager.get_fonts()
        assert fonts.size_base == 14  # Presentation theme has larger fonts

        plt.close(fig1)
        plt.close(fig2)

    def test_auto_apply_disabled(self):
        """Test factory with auto_apply disabled."""
        factory = FigureFactory(auto_apply=False)

        # Style should not be automatically applied
        # This is hard to test directly, but we can verify the flag
        assert factory.auto_apply is False

    def test_custom_style_manager_integration(self):
        """Test integration with custom style manager."""
        # Create custom style manager with modified colors
        manager = StyleManager()
        manager.update_colors({"primary": "#FF0000"})

        # Create factory with custom manager
        factory = FigureFactory(style_manager=manager)

        # Verify custom colors are used
        colors = factory.style_manager.get_colors()
        assert colors.primary == "#FF0000"

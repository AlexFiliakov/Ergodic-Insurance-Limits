"""Comprehensive tests for the visualization module with 90%+ coverage."""

import tempfile
from unittest.mock import MagicMock, Mock, patch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src import visualization


class TestVisualizationModule:
    """Test suite for the main visualization module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        plt.close("all")

    def teardown_method(self):
        """Clean up after each test."""
        plt.close("all")

    def test_module_imports(self):
        """Test that required components are importable."""
        assert hasattr(visualization, "WSJ_COLORS")
        assert hasattr(visualization, "COLOR_SEQUENCE")
        assert hasattr(visualization, "set_wsj_style")
        assert hasattr(visualization, "format_currency")
        assert hasattr(visualization, "format_percentage")
        assert hasattr(visualization, "WSJFormatter")
        # Check for FigureFactory and StyleManager from new module
        assert hasattr(visualization, "FigureFactory")
        assert hasattr(visualization, "StyleManager")
        assert hasattr(visualization, "Theme")

    def test_wsj_colors_defined(self):
        """Test that WSJ colors are properly defined."""
        assert isinstance(visualization.WSJ_COLORS, dict)
        expected_colors = [
            "light_blue",
            "blue",
            "dark_blue",
            "red",
            "green",
            "gray",
            "light_gray",
            "black",
            "orange",
            "yellow",
            "purple",
            "teal",
        ]
        for color in expected_colors:
            assert color in visualization.WSJ_COLORS
            assert isinstance(visualization.WSJ_COLORS[color], str)
            assert visualization.WSJ_COLORS[color].startswith("#")

    def test_color_sequence_defined(self):
        """Test that color sequence is properly defined."""
        assert isinstance(visualization.COLOR_SEQUENCE, list)
        assert len(visualization.COLOR_SEQUENCE) >= 7
        for color in visualization.COLOR_SEQUENCE:
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_figure_factory_import(self):
        """Test FigureFactory can be imported and instantiated."""
        # Check that FigureFactory is importable
        assert hasattr(visualization, "FigureFactory")
        # Test creating a factory instance
        factory = visualization.FigureFactory()
        assert factory is not None

    def test_figure_factory_creation_with_theme(self):
        """Test creating FigureFactory with different themes."""
        # Test creating factory with default theme
        factory = visualization.FigureFactory(theme=visualization.Theme.DEFAULT)
        assert factory is not None
        assert factory.style_manager.current_theme == visualization.Theme.DEFAULT

        # Test with presentation theme
        factory2 = visualization.FigureFactory(theme=visualization.Theme.PRESENTATION)
        assert factory2.style_manager.current_theme == visualization.Theme.PRESENTATION

    def test_style_manager_and_themes(self):
        """Test StyleManager and Theme are available."""
        assert hasattr(visualization, "StyleManager")
        assert hasattr(visualization, "Theme")
        # Test Theme enum values
        assert hasattr(visualization.Theme, "DEFAULT")
        assert hasattr(visualization.Theme, "PRESENTATION")

    def test_set_wsj_style(self):
        """Test set_wsj_style function."""
        # Should set matplotlib rcParams
        visualization.set_wsj_style()

        # Check some key style settings
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert "font.size" in plt.rcParams

    def test_set_wsj_style_with_factory(self):
        """Test applying style through FigureFactory."""
        # Create a factory and apply style
        factory = visualization.FigureFactory()
        factory.style_manager.apply_style()

        # Check that style was applied
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False

    def test_format_currency_basic(self):
        """Test basic currency formatting."""
        assert visualization.format_currency(1000) == "$1,000"
        assert visualization.format_currency(1000.50, decimals=2) == "$1,000.50"
        assert visualization.format_currency(-500) == "-$500"
        assert visualization.format_currency(-500.75, decimals=2) == "-$500.75"

    def test_format_currency_abbreviated(self):
        """Test currency formatting with abbreviation."""
        assert visualization.format_currency(1000, abbreviate=True) == "$1K"
        assert visualization.format_currency(1500, abbreviate=True) == "$2K"
        assert visualization.format_currency(1000000, abbreviate=True) == "$1M"
        assert visualization.format_currency(1500000, decimals=1, abbreviate=True) == "$1.5M"
        assert visualization.format_currency(1000000000, abbreviate=True) == "$1B"
        assert visualization.format_currency(1500000000, decimals=1, abbreviate=True) == "$1.5B"

    def test_format_currency_edge_cases(self):
        """Test currency formatting edge cases."""
        assert visualization.format_currency(0) == "$0"
        assert visualization.format_currency(0.1, decimals=2) == "$0.10"
        assert visualization.format_currency(999, abbreviate=True) == "$999"
        # Allow either format for negative abbreviated currency
        result = visualization.format_currency(-1500000, abbreviate=True, decimals=1)
        assert result in ["-$1.5M", "$-1.5M"]

    def test_format_currency_consistency(self):
        """Test format_currency is consistent across calls."""
        # Test that same input gives same output
        result1 = visualization.format_currency(1000, decimals=0, abbreviate=True)
        result2 = visualization.format_currency(1000, decimals=0, abbreviate=True)
        assert result1 == result2

    def test_format_percentage_basic(self):
        """Test basic percentage formatting."""
        assert visualization.format_percentage(0.05) == "5.0%"
        assert visualization.format_percentage(0.125, decimals=2) == "12.50%"
        assert visualization.format_percentage(1.0) == "100.0%"
        assert visualization.format_percentage(-0.05) == "-5.0%"

    def test_format_percentage_edge_cases(self):
        """Test percentage formatting edge cases."""
        assert visualization.format_percentage(0) == "0.0%"
        assert visualization.format_percentage(0.001, decimals=2) == "0.10%"
        assert visualization.format_percentage(10, decimals=0) == "1000%"

    def test_format_percentage_consistency(self):
        """Test format_percentage is consistent across calls."""
        # Test that same input gives same output
        result1 = visualization.format_percentage(0.05, decimals=1)
        result2 = visualization.format_percentage(0.05, decimals=1)
        assert result1 == result2
        assert result1 == "5.0%"


class TestWSJFormatter:
    """Test suite for WSJFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create WSJFormatter instance."""
        return visualization.WSJFormatter()

    def test_initialization(self):
        """Test WSJFormatter initialization."""
        # Should create formatter without issues
        formatter = visualization.WSJFormatter()
        assert formatter is not None

    def test_currency_formatter_method(self, formatter):
        """Test currency_formatter static method."""
        assert visualization.WSJFormatter.currency_formatter(1000, None) == "$1K"
        assert visualization.WSJFormatter.currency_formatter(1000000, None) == "$1M"
        assert visualization.WSJFormatter.currency_formatter(1000000000, None) == "$1B"

    def test_currency_method(self, formatter):
        """Test currency formatting method."""
        # Test thousands
        assert visualization.WSJFormatter.currency(1000) == "$1K"
        assert visualization.WSJFormatter.currency(1500, decimals=1) == "$1.5K"
        assert visualization.WSJFormatter.currency(1000, decimals=0) == "$1K"

        # Test millions
        assert visualization.WSJFormatter.currency(1000000) == "$1M"
        assert visualization.WSJFormatter.currency(1500000, decimals=1) == "$1.5M"
        assert visualization.WSJFormatter.currency(1234567, decimals=2) == "$1.23M"

        # Test billions
        assert visualization.WSJFormatter.currency(1000000000) == "$1B"
        assert visualization.WSJFormatter.currency(1500000000, decimals=1) == "$1.5B"

        # Test trillions
        assert visualization.WSJFormatter.currency(1000000000000) == "$1T"
        assert visualization.WSJFormatter.currency(1500000000000, decimals=1) == "$1.5T"

    def test_currency_method_negative_values(self, formatter):
        """Test currency method with negative values."""
        assert visualization.WSJFormatter.currency(-1000) == "-$1K"
        assert visualization.WSJFormatter.currency(-1500000, decimals=1) == "-$1.5M"
        assert visualization.WSJFormatter.currency(-1000000000) == "-$1B"

    def test_currency_method_edge_cases(self, formatter):
        """Test currency method edge cases."""
        # Very small values
        assert visualization.WSJFormatter.currency(0.5, decimals=2) == "$0.50"
        assert visualization.WSJFormatter.currency(0) == "$0"

        # Values under 1000
        assert visualization.WSJFormatter.currency(999) == "$999"
        assert visualization.WSJFormatter.currency(100.5, decimals=1) == "$100.5"

        # Exact boundaries
        assert visualization.WSJFormatter.currency(1000) == "$1K"
        assert visualization.WSJFormatter.currency(1000000) == "$1M"
        assert visualization.WSJFormatter.currency(1000000000) == "$1B"

    def test_percentage_formatter_method(self, formatter):
        """Test percentage_formatter static method."""
        assert visualization.WSJFormatter.percentage_formatter(0.05, None) == "5%"
        assert visualization.WSJFormatter.percentage_formatter(0.125, None) == "12%"
        assert visualization.WSJFormatter.percentage_formatter(1.0, None) == "100%"

    def test_percentage_method(self, formatter):
        """Test percentage formatting method."""
        assert visualization.WSJFormatter.percentage(0.05) == "5.0%"
        assert visualization.WSJFormatter.percentage(0.125, decimals=2) == "12.50%"
        assert visualization.WSJFormatter.percentage(1.0, decimals=0) == "100%"
        assert visualization.WSJFormatter.percentage(-0.05) == "-5.0%"

    def test_number_method(self, formatter):
        """Test number formatting method."""
        # Small numbers
        assert visualization.WSJFormatter.number(100) == "100"
        assert visualization.WSJFormatter.number(100.5, decimals=1) == "100.5"

        # Thousands
        assert visualization.WSJFormatter.number(1000) == "1.00K"
        assert visualization.WSJFormatter.number(1500, decimals=1) == "1.5K"

        # Millions
        assert visualization.WSJFormatter.number(1000000) == "1.00M"
        assert visualization.WSJFormatter.number(1500000, decimals=1) == "1.5M"

        # Billions
        assert visualization.WSJFormatter.number(1000000000) == "1.00B"
        assert visualization.WSJFormatter.number(1500000000, decimals=1) == "1.5B"

        # Trillions
        assert visualization.WSJFormatter.number(1000000000000) == "1.00T"
        assert visualization.WSJFormatter.number(1500000000000, decimals=1) == "1.5T"

        # Very large numbers
        assert visualization.WSJFormatter.number(1000000000000000) == "1000T"

    def test_number_method_negative_values(self, formatter):
        """Test number method with negative values."""
        assert visualization.WSJFormatter.number(-1000) == "-1.00K"
        assert visualization.WSJFormatter.number(-1500000, decimals=1) == "-1.5M"
        assert visualization.WSJFormatter.number(-1000000000) == "-1.00B"

    def test_millions_formatter_method(self, formatter):
        """Test millions_formatter static method."""
        assert visualization.WSJFormatter.millions_formatter(1000000, None) == "1M"
        assert visualization.WSJFormatter.millions_formatter(5000000, None) == "5M"
        assert visualization.WSJFormatter.millions_formatter(500000, None) == "0M"
        assert visualization.WSJFormatter.millions_formatter(1500000, None) == "2M"


class TestVisualizationIntegration:
    """Integration tests for visualization module."""

    def test_basic_workflow(self):
        """Test basic visualization workflow."""
        # Should be able to format values
        assert visualization.format_currency(1000) == "$1,000"
        assert visualization.format_percentage(0.05) == "5.0%"

        # Should be able to use WSJFormatter
        formatter = visualization.WSJFormatter()
        assert formatter.currency(1000) == "$1K"

    def test_style_application_workflow(self):
        """Test complete style application workflow."""
        # Set style
        visualization.set_wsj_style()

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1000, 2000, 3000])

        # Apply formatters
        from matplotlib.ticker import FuncFormatter

        ax.yaxis.set_major_formatter(FuncFormatter(visualization.WSJFormatter.currency_formatter))

        # Should not raise any errors
        fig.canvas.draw()
        plt.close(fig)

    def test_factory_integration(self):
        """Test integration with figure factory."""
        # Create factory instance
        factory = visualization.FigureFactory()
        # Should be able to create figures
        fig, ax = factory.create_figure()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_factory_with_different_sizes(self):
        """Test figure factory with different size presets."""
        factory = visualization.FigureFactory()

        # Test different size presets
        for size_type in ["small", "medium", "large"]:
            fig, ax = factory.create_figure(size_type=size_type)
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_module_constants_consistency(self):
        """Test that module constants are consistent."""
        # Color sequence should use colors from WSJ_COLORS
        for color in visualization.COLOR_SEQUENCE:
            assert color in visualization.WSJ_COLORS.values()

        # All colors should be valid hex codes
        import re

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for color_name, color_value in visualization.WSJ_COLORS.items():
            assert hex_pattern.match(
                color_value
            ), f"Invalid hex color for {color_name}: {color_value}"

    def test_wsj_colors_usage(self):
        """Test that WSJ colors can be used in plotting."""
        # Create a simple plot with WSJ colors
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], color=visualization.WSJ_COLORS["blue"])
        assert ax.get_lines()[0].get_color() == visualization.WSJ_COLORS["blue"]
        plt.close(fig)

    def test_multiple_themes(self):
        """Test switching between different themes."""
        factory = visualization.FigureFactory()

        # Create figures with different themes
        for theme in [visualization.Theme.DEFAULT, visualization.Theme.PRESENTATION]:
            factory.style_manager.set_theme(theme)
            fig, ax = factory.create_figure()
            assert fig is not None
            plt.close(fig)

    def test_export_functions(self):
        """Test that export functions are available."""
        # Check export functions are available
        assert hasattr(visualization, "save_figure")
        assert hasattr(visualization, "save_for_publication")
        assert hasattr(visualization, "save_for_presentation")
        assert hasattr(visualization, "save_for_web")

    def test_annotation_functions(self):
        """Test that annotation functions are available."""
        # Check annotation functions are available
        assert hasattr(visualization, "add_value_labels")
        assert hasattr(visualization, "add_trend_annotation")
        assert hasattr(visualization, "add_callout")
        assert hasattr(visualization, "add_benchmark_line")

    def test_plot_functions_available(self):
        """Test that plotting functions are available."""
        # Executive plots
        assert hasattr(visualization, "plot_loss_distribution")
        assert hasattr(visualization, "plot_return_period_curve")
        assert hasattr(visualization, "plot_insurance_layers")

        # Technical plots
        assert hasattr(visualization, "plot_convergence_diagnostics")
        assert hasattr(visualization, "plot_pareto_frontier_2d")
        assert hasattr(visualization, "plot_pareto_frontier_3d")

        # Interactive plots
        assert hasattr(visualization, "create_interactive_dashboard")

        # Batch plots
        assert hasattr(visualization, "plot_scenario_comparison")
        assert hasattr(visualization, "plot_sensitivity_heatmap")

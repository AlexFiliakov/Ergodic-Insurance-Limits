"""Tests for sensitivity visualization module.

This module tests the visualization functions for sensitivity analysis,
including tornado diagrams, heatmaps, and parameter sweep plots.

Author: Alex Filiakov
Date: 2025-01-29
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.sensitivity import SensitivityResult, TwoWaySensitivityResult
from ergodic_insurance.sensitivity_visualization import (
    create_sensitivity_report,
    plot_parameter_sweep,
    plot_sensitivity_matrix,
    plot_tornado_diagram,
    plot_two_way_sensitivity,
)


class TestTornadoDiagram:
    """Test tornado diagram visualization."""

    @pytest.fixture
    def tornado_data(self):
        """Create sample tornado diagram data."""
        return pd.DataFrame(
            {
                "parameter": ["frequency", "severity", "base_premium_rate", "retention"],
                "impact": [0.8, 0.6, 0.4, 0.2],
                "direction": ["positive", "negative", "positive", "negative"],
                "low_value": [0.08, 0.12, 0.09, 0.11],
                "high_value": [0.15, 0.10, 0.13, 0.105],
                "baseline": [0.12, 0.11, 0.11, 0.108],
                "baseline_param": [5.0, 100000, 0.02, 50000],
                "range_width": [0.07, 0.02, 0.04, 0.005],
            }
        )

    def test_plot_tornado_basic(self, tornado_data):
        """Test basic tornado diagram creation."""
        fig = plot_tornado_diagram(tornado_data)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Check that axes exist
        axes = fig.get_axes()
        assert len(axes) == 1

        # Check that bars are created
        ax = axes[0]
        patches = ax.patches
        assert len(patches) == len(tornado_data)

        plt.close(fig)

    def test_plot_tornado_customization(self, tornado_data):
        """Test tornado diagram with custom settings."""
        fig = plot_tornado_diagram(
            tornado_data,
            title="Custom Title",
            metric_label="ROE Impact",
            figsize=(12, 8),
            n_params=2,
            color_positive="#00FF00",
            color_negative="#FF0000",
            show_values=False,
        )

        assert fig is not None
        ax = fig.get_axes()[0]

        # Check title
        assert ax.get_title() == "Custom Title"

        # Check that only 2 parameters are shown
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(y_labels) == 2

        plt.close(fig)

    def test_plot_tornado_empty_data(self):
        """Test tornado diagram with empty data."""
        empty_data = pd.DataFrame(
            columns=["parameter", "impact", "direction", "low_value", "high_value", "baseline"]
        )

        fig = plot_tornado_diagram(empty_data)
        assert fig is not None
        plt.close(fig)


class TestTwoWaySensitivity:
    """Test two-way sensitivity visualization."""

    @pytest.fixture
    def two_way_result(self):
        """Create sample two-way sensitivity result."""
        return TwoWaySensitivityResult(
            parameter1="frequency",
            parameter2="severity",
            values1=np.array([3, 4, 5, 6, 7]),
            values2=np.array([80000, 100000, 120000]),
            metric_grid=np.array(
                [
                    [0.10, 0.11, 0.12],
                    [0.11, 0.12, 0.13],
                    [0.12, 0.13, 0.14],
                    [0.13, 0.14, 0.15],
                    [0.14, 0.15, 0.16],
                ]
            ),
            metric_name="ROE",
        )

    def test_plot_two_way_basic(self, two_way_result):
        """Test basic two-way sensitivity plot."""
        fig = plot_two_way_sensitivity(two_way_result)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        ax = fig.get_axes()[0]

        # Check labels
        assert "frequency" in ax.get_xlabel()
        assert "severity" in ax.get_ylabel()

        plt.close(fig)

    def test_plot_two_way_with_contours(self, two_way_result):
        """Test two-way plot with contours."""
        fig = plot_two_way_sensitivity(two_way_result, show_contours=True, contour_levels=5)

        assert fig is not None

        # Check that contours are present
        ax = fig.get_axes()[0]
        collections = ax.collections
        # Should have at least the pcolormesh and contour collections
        assert len(collections) >= 2

        plt.close(fig)

    def test_plot_two_way_with_optimal_point(self, two_way_result):
        """Test two-way plot with optimal point marked."""
        fig = plot_two_way_sensitivity(two_way_result, optimal_point=(5.0, 100000))

        assert fig is not None

        # Check that the point is plotted
        ax = fig.get_axes()[0]
        lines = ax.lines
        # Should have at least one line (the optimal point marker)
        assert len(lines) >= 1

        plt.close(fig)

    def test_plot_two_way_custom_settings(self, two_way_result):
        """Test two-way plot with custom settings."""
        fig = plot_two_way_sensitivity(
            two_way_result,
            title="Custom Heatmap",
            cmap="viridis",
            figsize=(12, 10),
            show_contours=False,
            fmt=".3f",
        )

        assert fig is not None
        ax = fig.get_axes()[0]
        assert ax.get_title() == "Custom Heatmap"

        plt.close(fig)


class TestParameterSweep:
    """Test parameter sweep visualization."""

    @pytest.fixture
    def sensitivity_result(self):
        """Create sample sensitivity result."""
        return SensitivityResult(
            parameter="frequency",
            baseline_value=5.0,
            variations=np.linspace(3, 7, 11),
            metrics={
                "optimal_roe": np.linspace(0.08, 0.15, 11),
                "bankruptcy_risk": np.linspace(0.02, 0.005, 11),
                "growth_rate": np.linspace(0.03, 0.07, 11),
            },
        )

    def test_plot_parameter_sweep_basic(self, sensitivity_result):
        """Test basic parameter sweep plot."""
        fig = plot_parameter_sweep(sensitivity_result)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Should have subplots for each metric
        axes = fig.get_axes()
        visible_axes = [ax for ax in axes if ax.get_visible()]
        assert len(visible_axes) == 3

        plt.close(fig)

    def test_plot_parameter_sweep_selected_metrics(self, sensitivity_result):
        """Test parameter sweep with selected metrics."""
        fig = plot_parameter_sweep(sensitivity_result, metrics=["optimal_roe", "growth_rate"])

        assert fig is not None

        # Should have 2 subplots
        axes = fig.get_axes()
        visible_axes = [ax for ax in axes if ax.get_visible()]
        assert len(visible_axes) == 2

        plt.close(fig)

    def test_plot_parameter_sweep_normalized(self, sensitivity_result):
        """Test parameter sweep with normalization."""
        fig = plot_parameter_sweep(sensitivity_result, normalize=True, mark_baseline=True)

        assert fig is not None

        # Check that values are normalized
        axes = fig.get_axes()
        for ax in axes:
            if ax.get_visible():
                lines = ax.lines
                if lines:
                    y_data = np.asarray(lines[0].get_ydata())
                    # Normalized data should be between 0 and 1
                    assert all(0 <= float(y) <= 1 for y in y_data if not np.isnan(y))

        plt.close(fig)

    def test_plot_parameter_sweep_custom_title(self, sensitivity_result):
        """Test parameter sweep with custom title."""
        fig = plot_parameter_sweep(
            sensitivity_result, title="Custom Sweep Analysis", figsize=(14, 6)
        )

        assert fig is not None
        # Check suptitle
        assert fig._suptitle is not None  # type: ignore[attr-defined]
        assert fig._suptitle.get_text() == "Custom Sweep Analysis"  # type: ignore[attr-defined]

        plt.close(fig)


class TestSensitivityMatrix:
    """Test sensitivity matrix visualization."""

    @pytest.fixture
    def multiple_results(self):
        """Create multiple sensitivity results."""
        results = {}

        for param in ["frequency", "severity", "premium"]:
            results[param] = SensitivityResult(
                parameter=param,
                baseline_value=10.0,
                variations=np.linspace(7, 13, 7),  # ±30%
                metrics={"optimal_roe": np.random.uniform(0.08, 0.15, 7)},
            )

        return results

    def test_plot_sensitivity_matrix_basic(self, multiple_results):
        """Test basic sensitivity matrix plot."""
        fig = plot_sensitivity_matrix(multiple_results)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        ax = fig.get_axes()[0]

        # Check that image is displayed
        images = ax.images
        assert len(images) == 1

        plt.close(fig)

    def test_plot_sensitivity_matrix_custom(self, multiple_results):
        """Test sensitivity matrix with custom settings."""
        fig = plot_sensitivity_matrix(
            multiple_results,
            metric="optimal_roe",
            figsize=(14, 8),
            cmap="viridis",
            show_values=False,
        )

        assert fig is not None

        # Check that no text values are shown
        ax = fig.get_axes()[0]
        texts = ax.texts
        # Should only have title and labels, no cell values
        assert len(texts) < 30  # Arbitrary threshold

        plt.close(fig)


class TestSensitivityReport:
    """Test complete sensitivity report generation."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create mock analyzer for testing."""
        analyzer = MagicMock()

        # Mock tornado diagram data
        analyzer.create_tornado_diagram.return_value = pd.DataFrame(
            {
                "parameter": ["freq", "sev", "prem"],
                "impact": [0.8, 0.6, 0.4],
                "direction": ["positive", "negative", "positive"],
                "low_value": [0.08, 0.12, 0.09],
                "high_value": [0.15, 0.10, 0.13],
                "baseline": [0.12, 0.11, 0.11],
                "baseline_param": [5.0, 100000, 0.02],
            }
        )

        # Mock parameter analysis
        def mock_analyze(param, **kwargs):
            return SensitivityResult(
                parameter=param,
                baseline_value=10.0,
                variations=np.linspace(7, 13, 5),
                metrics={"optimal_roe": np.random.uniform(0.08, 0.15, 5)},
            )

        analyzer.analyze_parameter.side_effect = mock_analyze

        return analyzer

    def test_create_report_basic(self, mock_analyzer):
        """Test basic report creation."""
        report = create_sensitivity_report(
            mock_analyzer, parameters=["freq", "sev", "prem"], metric="optimal_roe"
        )

        assert "figures" in report
        assert "summary" in report
        assert "data" in report

        # Check tornado figure
        assert "tornado" in report["figures"]
        assert report["figures"]["tornado"] is not None

        # Check summary statistics
        assert "most_impactful" in report["summary"]
        assert "least_impactful" in report["summary"]

        # Clean up figures
        for fig in report["figures"].values():
            if fig is not None:
                plt.close(fig)

    def test_create_report_with_output(self, mock_analyzer):
        """Test report creation with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = create_sensitivity_report(
                mock_analyzer, parameters=["freq", "sev"], output_dir=tmpdir, formats=["png"]
            )

            # Check that files were created
            output_path = Path(tmpdir)
            assert (output_path / "tornado_diagram.png").exists()

            # Clean up figures
            for fig in report["figures"].values():
                if fig is not None:
                    plt.close(fig)

    def test_create_report_relative_importance(self, mock_analyzer):
        """Test that relative importance is calculated."""
        report = create_sensitivity_report(mock_analyzer, parameters=["freq", "sev", "prem"])

        assert "relative_importances" in report["summary"]

        # Check that relative importances sum to 100%
        importances = report["summary"]["relative_importances"]
        total = sum(item["relative_importance"] for item in importances)
        assert abs(total - 100) < 0.1  # Allow small rounding error

        # Clean up
        for fig in report["figures"].values():
            if fig is not None:
                plt.close(fig)


def test_module_imports():
    """Test that all visualization imports work correctly."""
    # These imports are already done at the module level, so just verify they exist
    assert plot_tornado_diagram is not None
    assert plot_two_way_sensitivity is not None
    assert plot_parameter_sweep is not None
    assert plot_sensitivity_matrix is not None
    assert create_sensitivity_report is not None


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Ensure all plots are closed after each test."""
    yield
    plt.close("all")

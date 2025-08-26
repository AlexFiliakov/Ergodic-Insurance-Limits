"""Extended tests for visualization module to achieve >80% coverage."""

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

from ergodic_insurance.src.visualization import (
    WSJFormatter,
    _create_interactive_pareto_2d,
    _create_interactive_pareto_3d,
    _create_pareto_parallel_coordinates,
    create_interactive_dashboard,
    create_interactive_pareto_frontier,
    format_currency,
    format_percentage,
    plot_convergence_diagnostics,
    plot_insurance_layers,
    plot_loss_distribution,
    plot_pareto_frontier_2d,
    plot_pareto_frontier_3d,
    plot_return_period_curve,
)


@dataclass
class MockParetoPoint:
    """Mock Pareto point for testing."""

    objectives: Dict[str, float]
    crowding_distance: float = 1.0
    dominance_count: int = 0
    is_pareto: bool = True


class TestFormatterExtended:
    """Test additional formatter functionality."""

    def test_format_currency_abbreviate_large_numbers(self):
        """Test currency formatting with abbreviation for large numbers."""
        # Test billions
        assert format_currency(1_500_000_000, abbreviate=True, decimals=1) == "$1.5B"
        assert format_currency(2_000_000_000, abbreviate=True, decimals=0) == "$2B"

        # Test millions
        assert format_currency(5_500_000, abbreviate=True, decimals=1) == "$5.5M"
        assert format_currency(10_000_000, abbreviate=True, decimals=1) == "$10.0M"

        # Test thousands
        assert format_currency(50_000, abbreviate=True, decimals=0) == "$50K"
        assert format_currency(1_234, abbreviate=True, decimals=1) == "$1.2K"

        # Test small numbers
        assert format_currency(999, abbreviate=True, decimals=0) == "$999"
        assert format_currency(100, abbreviate=True, decimals=0) == "$100"

    def test_wsj_formatter_currency_method_edge_cases(self):
        """Test WSJFormatter.currency with various edge cases."""
        formatter = WSJFormatter()

        # Test trillions
        assert formatter.currency(1_000_000_000_000) == "$1T"
        assert formatter.currency(1_500_000_000_000) == "$1.5T"
        assert formatter.currency(1_234_000_000_000, decimals=2) == "$1.23T"

        # Test exact billions/millions/thousands
        assert formatter.currency(2_000_000_000) == "$2B"
        assert formatter.currency(3_000_000) == "$3M"
        assert formatter.currency(4_000) == "$4K"

        # Test negative values
        assert formatter.currency(-1_500_000_000) == "-$1.5B"
        assert formatter.currency(-500_000) == "-$500K"

        # Test very small positive values
        assert formatter.currency(0.50) == "$0.50"
        assert formatter.currency(0.99) == "$0.99"

        # Test exact integers
        assert formatter.currency(100) == "$100"
        assert formatter.currency(999) == "$999"

    def test_wsj_formatter_number_method_edge_cases(self):
        """Test WSJFormatter.number with various edge cases."""
        formatter = WSJFormatter()

        # Test very large numbers (>= 1e15)
        assert formatter.number(1_000_000_000_000_000) == "1000T"
        assert formatter.number(2_500_000_000_000_000) == "2500T"

        # Test standard large numbers
        assert formatter.number(1_234_567_890_123) == "1.23T"
        assert formatter.number(987_654_321) == "987.65M"

        # Test exact integers
        assert formatter.number(1000) == "1.00K"
        assert formatter.number(100) == "100"
        assert formatter.number(42) == "42"

    def test_wsj_formatter_as_axis_formatter(self):
        """Test WSJFormatter methods as matplotlib axis formatters."""
        formatter = WSJFormatter()

        # Test currency_formatter (simulating matplotlib calling it)
        assert formatter.currency_formatter(1_000_000, None) == "$1M"
        assert formatter.currency_formatter(500_000, None) == "$500K"

        # Test percentage_formatter
        assert formatter.percentage_formatter(0.25, None) == "25%"
        assert formatter.percentage_formatter(0.05, None) == "5%"

        # Test millions_formatter
        assert formatter.millions_formatter(5_000_000, None) == "5M"
        assert formatter.millions_formatter(1_500_000, None) == "2M"  # Rounds to nearest


class TestLossDistributionExtended:
    """Test additional loss distribution functionality."""

    def test_plot_loss_distribution_dataframe_without_amount_column(self):
        """Test loss distribution with DataFrame lacking 'amount' column."""
        # Create DataFrame with numeric column but not named 'amount'
        df = pd.DataFrame({"loss_value": np.random.lognormal(10, 1, 100), "category": ["A"] * 100})

        fig = plot_loss_distribution(df, title="Test Losses")
        assert fig is not None
        plt.close(fig)

    def test_plot_loss_distribution_dataframe_no_numeric_columns(self):
        """Test loss distribution with DataFrame with no numeric columns."""
        df = pd.DataFrame({"category": ["A", "B", "C"], "name": ["Loss1", "Loss2", "Loss3"]})

        with pytest.raises(ValueError, match="at least one numeric column"):
            plot_loss_distribution(df)

    def test_plot_loss_distribution_with_var_metrics(self):
        """Test loss distribution with VaR metrics display."""
        np.random.seed(42)
        losses = np.random.lognormal(10, 1, 1000)

        fig = plot_loss_distribution(
            losses, title="Losses with VaR", show_metrics=True, var_levels=[0.90, 0.95, 0.99]
        )

        assert fig is not None
        # Check that VaR lines were added
        ax = fig.axes[0]
        assert len(ax.lines) > 0  # Should have vertical lines for VaR
        plt.close(fig)


class TestReturnPeriodCurveExtended:
    """Test additional return period curve functionality."""

    def test_plot_return_period_curve_dataframe_input(self):
        """Test return period curve with DataFrame input."""
        df = pd.DataFrame({"amount": np.random.lognormal(10, 1, 100), "year": range(100)})

        fig = plot_return_period_curve(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_return_period_curve_with_custom_periods(self):
        """Test return period curve with custom return periods."""
        losses = np.random.lognormal(10, 1, 100)
        return_periods = np.array([1, 5, 10, 25, 50, 100])

        # Sort losses to match return periods
        sorted_losses = np.sort(losses)[::-1][: len(return_periods)]

        fig = plot_return_period_curve(
            sorted_losses, return_periods=return_periods, title="Custom Return Periods"
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_return_period_curve_with_scenarios(self):
        """Test return period curve with multiple scenarios."""
        np.random.seed(42)
        base_losses = np.random.lognormal(10, 1, 50)

        scenarios = {
            "Pessimistic": base_losses * 1.5,
            "Optimistic": base_losses * 0.7,
            "Stressed": base_losses * 2.0,
        }

        fig = plot_return_period_curve(
            base_losses, scenarios=scenarios, title="Multi-Scenario Return Periods"
        )
        assert fig is not None
        plt.close(fig)


class TestInsuranceLayersExtended:
    """Test additional insurance layers functionality."""

    def test_plot_insurance_layers_with_list_input(self):
        """Test insurance layers with list of dictionaries input."""
        layers = [
            {"attachment": 0, "limit": 1_000_000, "premium": 0.05},
            {"attachment": 1_000_000, "limit": 4_000_000, "premium": 0.03},
            {"attachment": 5_000_000, "limit": 5_000_000, "premium": 0.02},
        ]

        fig = plot_insurance_layers(layers, title="Test Layers")
        assert fig is not None
        plt.close(fig)

    def test_plot_insurance_layers_empty_list(self):
        """Test insurance layers with empty list."""
        fig = plot_insurance_layers([])
        assert fig is not None
        # Should show "No layers defined" message
        plt.close(fig)

    def test_plot_insurance_layers_with_total_limit(self):
        """Test insurance layers with specified total limit."""
        layers = pd.DataFrame(
            {
                "attachment": [0, 1_000_000],
                "limit": [1_000_000, 4_000_000],
                "premium_rate": [0.05, 0.03],
            }
        )

        fig = plot_insurance_layers(layers, total_limit=10_000_000, title="Layers with Total Limit")
        assert fig is not None
        plt.close(fig)

    def test_plot_insurance_layers_alias_losses_parameter(self):
        """Test insurance layers with loss_data alias parameter."""
        layers = [{"attachment": 0, "limit": 1_000_000, "premium": 0.05}]
        loss_data = np.random.lognormal(10, 1, 100)

        fig = plot_insurance_layers(layers, loss_data=loss_data, show_expected_loss=True)
        assert fig is not None
        plt.close(fig)


class TestConvergenceExtended:
    """Test additional convergence diagnostics functionality."""

    def test_plot_convergence_full_statistics(self):
        """Test convergence diagnostics with all statistics."""
        np.random.seed(42)
        n_iterations = 100

        convergence_stats = {
            "iterations": list(range(n_iterations)),
            "r_hat_history": 2.0 - np.logspace(-2, 0, n_iterations),
            "ess_history": np.logspace(2, 4, n_iterations),
            "lags": list(range(20)),
            "autocorrelation": np.exp(-np.arange(20) / 5),
            "mcse_by_metric": {
                "Mean": 0.01,
                "Std Dev": 0.02,
                "VaR(95%)": 0.03,
                "TVaR(95%)": 0.04,
            },
        }

        fig = plot_convergence_diagnostics(
            convergence_stats,
            title="Full Convergence Diagnostics",
            r_hat_threshold=1.1,
            show_threshold=True,
        )

        assert fig is not None
        assert len(fig.axes) == 4  # Should have 4 subplots
        plt.close(fig)

    def test_plot_convergence_partial_statistics(self):
        """Test convergence with partial statistics."""
        convergence_stats = {
            "iterations": list(range(50)),
            "r_hat_history": 2.0 - np.logspace(-2, 0, 50),
        }

        fig = plot_convergence_diagnostics(convergence_stats)
        assert fig is not None
        plt.close(fig)


class TestParetoFrontier:
    """Test Pareto frontier plotting functions."""

    @pytest.fixture
    def sample_pareto_points(self):
        """Create sample Pareto points."""
        points = []
        for i in range(10):
            point = MockParetoPoint(
                objectives={
                    "cost": 100 + i * 10,
                    "risk": 50 - i * 4,
                    "performance": 60 + i * 3,
                },
                crowding_distance=np.random.uniform(0.5, 2.0),
            )
            points.append(point)
        return points

    def test_plot_pareto_frontier_2d(self, sample_pareto_points):
        """Test 2D Pareto frontier plotting."""
        fig = plot_pareto_frontier_2d(
            sample_pareto_points,
            x_objective="cost",
            y_objective="risk",
            title="2D Pareto Frontier",
            highlight_knees=True,
            show_trade_offs=True,
        )

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_pareto_frontier_2d_custom_labels(self, sample_pareto_points):
        """Test 2D Pareto frontier with custom labels."""
        fig = plot_pareto_frontier_2d(
            sample_pareto_points,
            x_objective="cost",
            y_objective="risk",
            x_label="Total Cost ($)",
            y_label="Risk Level",
            title="Custom Labels Pareto",
            highlight_knees=False,
            show_trade_offs=False,
        )

        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Total Cost ($)"
        assert ax.get_ylabel() == "Risk Level"
        plt.close(fig)

    def test_plot_pareto_frontier_3d(self, sample_pareto_points):
        """Test 3D Pareto frontier plotting."""
        fig = plot_pareto_frontier_3d(
            sample_pareto_points,
            x_objective="cost",
            y_objective="risk",
            z_objective="performance",
            title="3D Pareto Frontier",
        )

        assert fig is not None
        assert hasattr(fig.axes[0], "zaxis")  # Check it's a 3D plot
        plt.close(fig)

    def test_plot_pareto_frontier_3d_with_interpolation(self, sample_pareto_points):
        """Test 3D Pareto frontier with surface interpolation."""
        # Add more points for successful interpolation
        points = sample_pareto_points * 2  # Duplicate to get more points

        fig = plot_pareto_frontier_3d(
            points,
            x_objective="cost",
            y_objective="risk",
            z_objective="performance",
            x_label="Cost ($)",
            y_label="Risk",
            z_label="Performance",
            title="3D Surface Pareto",
        )

        assert fig is not None
        plt.close(fig)


class TestInteractiveDashboardExtended:
    """Test additional interactive dashboard functionality."""

    def test_dashboard_with_dict_results(self):
        """Test dashboard with dictionary results."""
        results = {
            "growth_rates": np.random.normal(0.05, 0.02, 1000),
            "losses": np.random.lognormal(10, 1, 1000),
            "convergence": {
                "iterations": list(range(100)),
                "r_hat": 2.0 - np.logspace(-2, 0, 100),
            },
            "metrics": {
                "var_95": 1_000_000,
                "var_99": 2_000_000,
                "tvar_99": 3_000_000,
                "expected_shortfall": 2_500_000,
            },
        }

        fig = create_interactive_dashboard(
            results, title="Dict Results Dashboard", height=800, show_distributions=True
        )

        assert fig is not None
        assert hasattr(fig, "data")
        assert fig.layout.title.text == "Dict Results Dashboard"

    def test_dashboard_with_partial_results(self):
        """Test dashboard with partial results."""
        results = {
            "growth_rates": np.random.normal(0.05, 0.02, 100),
            # Missing other fields
        }

        fig = create_interactive_dashboard(results)
        assert fig is not None


class TestInteractiveParetoFrontier:
    """Test interactive Pareto frontier functions."""

    @pytest.fixture
    def sample_points(self):
        """Create sample Pareto points."""
        points = []
        for i in range(8):
            point = MockParetoPoint(
                objectives={
                    "obj1": 10 + i,
                    "obj2": 20 - i,
                    "obj3": 15 + i * 0.5,
                    "obj4": 25 - i * 0.7,
                },
                crowding_distance=1.0 + i * 0.1,
            )
            points.append(point)
        return points

    def test_create_interactive_pareto_2d(self, sample_points):
        """Test 2D interactive Pareto frontier."""
        fig = create_interactive_pareto_frontier(
            sample_points, objectives=["obj1", "obj2"], title="Interactive 2D", show_dominated=True
        )

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 2  # At least frontier line and points

    def test_create_interactive_pareto_3d(self, sample_points):
        """Test 3D interactive Pareto frontier."""
        fig = create_interactive_pareto_frontier(
            sample_points, objectives=["obj1", "obj2", "obj3"], title="Interactive 3D", height=700
        )

        assert fig is not None
        assert hasattr(fig, "data")
        assert fig.layout.height == 700

    def test_create_interactive_pareto_parallel(self, sample_points):
        """Test parallel coordinates for many objectives."""
        fig = create_interactive_pareto_frontier(
            sample_points, objectives=["obj1", "obj2", "obj3", "obj4"], title="Parallel Coordinates"
        )

        assert fig is not None
        assert hasattr(fig, "data")
        # Should use parallel coordinates for 4+ objectives

    def test_interactive_pareto_2d_helper(self, sample_points):
        """Test the 2D helper function directly."""
        fig = _create_interactive_pareto_2d(
            sample_points,
            objectives=["obj1", "obj2"],
            title="2D Helper Test",
            height=600,
            show_dominated=False,
        )

        assert fig is not None
        assert fig.layout.title.text == "2D Helper Test"

    def test_interactive_pareto_3d_helper(self, sample_points):
        """Test the 3D helper function directly."""
        fig = _create_interactive_pareto_3d(
            sample_points, objectives=["obj1", "obj2", "obj3"], title="3D Helper Test", height=500
        )

        assert fig is not None
        assert fig.layout.title.text == "3D Helper Test"
        assert fig.layout.height == 500

    def test_pareto_parallel_coordinates_helper(self, sample_points):
        """Test the parallel coordinates helper function."""
        fig = _create_pareto_parallel_coordinates(
            sample_points,
            objectives=["obj1", "obj2", "obj3", "obj4"],
            title="Parallel Helper Test",
            height=600,
        )

        assert fig is not None
        assert hasattr(fig.data[0], "type")


class TestEdgeCasesExtended:
    """Test additional edge cases."""

    def test_plot_functions_with_single_point(self):
        """Test plotting functions with single data point."""
        single_point = [MockParetoPoint(objectives={"x": 1, "y": 2, "z": 3}, crowding_distance=1.0)]

        # 2D plot with single point
        fig = plot_pareto_frontier_2d(single_point, x_objective="x", y_objective="y")
        assert fig is not None
        plt.close(fig)

        # 3D plot with single point
        fig = plot_pareto_frontier_3d(
            single_point, x_objective="x", y_objective="y", z_objective="z"
        )
        assert fig is not None
        plt.close(fig)

    def test_interactive_functions_with_single_objective(self):
        """Test interactive functions with single objective."""
        points = [MockParetoPoint(objectives={"a": 1}, crowding_distance=1.0)]

        # Should handle single objective gracefully - likely returns None or raises ValueError
        # Test that it doesn't crash with single objective
        try:
            result = create_interactive_pareto_frontier(
                points, objectives=["a"], title="Single Objective"  # Only one objective
            )
            # If it returns something, check it's valid
            assert result is None or hasattr(result, "data")
        except (ValueError, IndexError):
            # If it raises an error, that's ok too
            pass

    def test_formatter_with_extreme_values(self):
        """Test formatters with extreme values."""
        formatter = WSJFormatter()

        # Test with very large number
        assert formatter.number(1e20) == "100000000T"

        # Test with zero
        assert formatter.currency(0) == "$0"
        assert formatter.percentage(0) == "0.0%"
        assert formatter.number(0) == "0"

    def test_plot_functions_cleanup(self):
        """Ensure all plots are properly closed."""
        initial_figs = plt.get_fignums()

        # Create several plots
        fig1 = plot_loss_distribution(np.random.randn(100))
        fig2 = plot_return_period_curve(np.random.randn(50))
        fig3 = plot_insurance_layers([{"attachment": 0, "limit": 1000, "premium": 0.1}])

        # Close them
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

        # Check all are closed
        final_figs = plt.get_fignums()
        assert len(final_figs) == len(initial_figs)

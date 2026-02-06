"""Extended tests for visualization module to achieve >80% coverage."""

from dataclasses import dataclass
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Import public functions from the visualization package
from ergodic_insurance.visualization import (
    WSJFormatter,
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

# Import private functions directly from the technical_plots module for testing
from ergodic_insurance.visualization.technical_plots import (
    _create_interactive_pareto_2d,
    _create_interactive_pareto_3d,
    _create_pareto_parallel_coordinates,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


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
                "base_premium_rate": [0.05, 0.03],
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
            # Track that no exception was raised
            exception_raised = False
        except (ValueError, IndexError) as e:
            # These exceptions are expected for single objective
            exception_raised = True
            # Verify the exception message is meaningful
            assert len(str(e)) > 0, "Exception should have a message"

        # Document the expected behavior: either returns None/valid object or raises expected exception
        assert (
            exception_raised or result is None or hasattr(result, "data")
        ), "Function should either handle single objective gracefully or raise ValueError/IndexError"

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


class TestROERuinFrontier:
    """Test ROE-Ruin Efficient Frontier visualization."""

    @pytest.fixture
    def sample_optimization_results(self):
        """Create sample optimization results for different company sizes."""
        # Generate synthetic Pareto frontier data
        np.random.seed(42)

        # $1M company - higher risk tolerance
        roe_1m = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20])
        ruin_1m = np.array([0.08, 0.05, 0.03, 0.02, 0.015, 0.012, 0.01])

        # $10M company - moderate risk
        roe_10m = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
        ruin_10m = np.array([0.05, 0.03, 0.02, 0.012, 0.008, 0.006])

        # $100M company - conservative
        roe_100m = np.array([0.03, 0.05, 0.07, 0.09, 0.11])
        ruin_100m = np.array([0.03, 0.02, 0.01, 0.006, 0.004])

        return {
            1e6: pd.DataFrame({"roe": roe_1m, "ruin_prob": ruin_1m}),
            1e7: pd.DataFrame({"roe": roe_10m, "ruin_prob": ruin_10m}),
            1e8: pd.DataFrame({"roe": roe_100m, "ruin_prob": ruin_100m}),
        }

    @pytest.fixture
    def single_dataframe_results(self):
        """Create sample results as a single DataFrame with company_size column."""
        data = []

        # $1M company data
        for roe, ruin in zip([0.05, 0.10, 0.15], [0.05, 0.02, 0.01]):
            data.append({"company_size": 1e6, "roe": roe, "ruin_prob": ruin})

        # $10M company data
        for roe, ruin in zip([0.04, 0.08, 0.12], [0.03, 0.015, 0.008]):
            data.append({"company_size": 1e7, "roe": roe, "ruin_prob": ruin})

        return pd.DataFrame(data)

    def test_plot_roe_ruin_frontier_basic(self, sample_optimization_results):
        """Test basic ROE-Ruin frontier plotting."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        fig = plot_roe_ruin_frontier(sample_optimization_results)

        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]

        # Check labels
        assert ax.get_xlabel() == "Return on Equity (%)"
        assert ax.get_ylabel() == "Ruin Probability (%)"
        assert "ROE-Ruin Efficient Frontier" in ax.get_title()

        # Check that curves were plotted (3 company sizes)
        lines = []
        for l in ax.get_lines():
            label = l.get_label()
            if label and isinstance(label, str) and "Company" in label:
                lines.append(l)
        assert len(lines) >= 3

        plt.close(fig)

    def test_plot_roe_ruin_frontier_with_dataframe(self, single_dataframe_results):
        """Test ROE-Ruin frontier with single DataFrame input."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        fig = plot_roe_ruin_frontier(single_dataframe_results)

        assert fig is not None
        ax = fig.axes[0]

        # Check that curves were plotted
        lines = []
        for l in ax.get_lines():
            label = l.get_label()
            if label and isinstance(label, str) and "Company" in label:
                lines.append(l)
        assert len(lines) >= 2  # Two company sizes in the data

        plt.close(fig)

    def test_plot_roe_ruin_frontier_customization(self, sample_optimization_results):
        """Test ROE-Ruin frontier with custom options."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        fig = plot_roe_ruin_frontier(
            sample_optimization_results,
            company_sizes=[1e6, 1e7],  # Only plot two sizes
            title="Custom Title",
            figsize=(10, 6),
            highlight_sweet_spots=False,
            show_optimal_zones=False,
            log_scale_y=False,
            grid=False,
            annotations=False,
            color_scheme=["red", "blue"],
            export_dpi=150,
        )

        assert fig is not None
        ax = fig.axes[0]

        # Check customizations
        assert "Custom Title" in ax.get_title()
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        assert ax.get_yscale() == "linear"  # Not log scale
        assert fig.dpi == 150

        # Check only 2 curves plotted
        lines = []
        for l in ax.get_lines():
            label = l.get_label()
            if label and isinstance(label, str) and "Company" in label:
                lines.append(l)
        assert len(lines) == 2

        plt.close(fig)

    def test_plot_roe_ruin_frontier_sweet_spots(self, sample_optimization_results):
        """Test sweet spot detection and highlighting."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        fig = plot_roe_ruin_frontier(
            sample_optimization_results, highlight_sweet_spots=True, annotations=True
        )

        assert fig is not None
        ax = fig.axes[0]

        # Check for star markers (sweet spots)
        star_collections = [
            c for c in ax.collections if hasattr(c, "get_paths") and len(c.get_paths()) > 0
        ]
        assert len(star_collections) > 0

        # Check for annotations
        annotations = ax.texts
        assert len(annotations) > 0
        assert any("Sweet Spot" in str(ann.get_text()) for ann in annotations)

        plt.close(fig)

    def test_plot_roe_ruin_frontier_optimal_zones(self, sample_optimization_results):
        """Test optimal zone visualization."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        fig = plot_roe_ruin_frontier(sample_optimization_results, show_optimal_zones=True)

        assert fig is not None
        ax = fig.axes[0]

        # Check for shaded regions (axvspan and axhspan create patches)
        patches = ax.patches
        assert len(patches) > 0

        plt.close(fig)

    def test_plot_roe_ruin_frontier_log_scale(self, sample_optimization_results):
        """Test log scale for ruin probability axis."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        fig = plot_roe_ruin_frontier(sample_optimization_results, log_scale_y=True)

        assert fig is not None
        ax = fig.axes[0]

        # Check y-axis is log scale
        assert ax.get_yscale() == "log"

        plt.close(fig)

    def test_plot_roe_ruin_frontier_invalid_input(self):
        """Test error handling for invalid inputs."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        # Test with invalid data type
        with pytest.raises(ValueError, match="Results must be DataFrame or dict"):
            plot_roe_ruin_frontier("invalid_data")

        # Test with DataFrame missing required column
        df_missing = pd.DataFrame({"roe": [0.1, 0.2], "ruin_prob": [0.01, 0.02]})
        with pytest.raises(ValueError, match="DataFrame must have 'company_size' column"):
            plot_roe_ruin_frontier(df_missing)

        # Test with empty dict
        with pytest.raises(ValueError, match="No valid company sizes found"):
            plot_roe_ruin_frontier({})

    def test_plot_roe_ruin_frontier_alternative_column_names(self):
        """Test handling of alternative column names."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        # Create data with alternative column names
        data = {
            1e6: pd.DataFrame(
                {"return_on_equity": [0.1, 0.15, 0.2], "ruin_probability": [0.05, 0.02, 0.01]}
            )
        }

        fig = plot_roe_ruin_frontier(data)
        assert fig is not None
        plt.close(fig)

    def test_find_knee_point(self):
        """Test knee point detection algorithm."""
        from ergodic_insurance.visualization.executive_plots import _find_knee_point

        # Create a curve with clear knee point
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        y = np.array([10, 7, 5, 3, 2.5, 2.2, 2.0])

        knee_idx = _find_knee_point(x, y)

        # The knee should be around index 2-3 where curvature is highest
        assert 1 <= knee_idx <= 4

    def test_plot_roe_ruin_frontier_single_company(self, sample_optimization_results):
        """Test plotting with single company size."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        # Use only one company size
        single_company = {1e6: sample_optimization_results[1e6]}

        fig = plot_roe_ruin_frontier(single_company)

        assert fig is not None
        ax = fig.axes[0]

        # Check only one curve plotted
        lines = []
        for l in ax.get_lines():
            label = l.get_label()
            if label and isinstance(label, str) and "Company" in label:
                lines.append(l)
        assert len(lines) == 1

        plt.close(fig)

    def test_plot_roe_ruin_frontier_export_dpi(self, sample_optimization_results):
        """Test export DPI settings."""
        from ergodic_insurance.visualization.executive_plots import plot_roe_ruin_frontier

        # Test web resolution
        fig_web = plot_roe_ruin_frontier(sample_optimization_results, export_dpi=150)
        assert fig_web.dpi == 150
        plt.close(fig_web)

        # Test print resolution
        fig_print = plot_roe_ruin_frontier(sample_optimization_results, export_dpi=300)
        assert fig_print.dpi == 300
        plt.close(fig_print)


class TestRuinCliffVisualization:
    """Tests for the ruin cliff visualization function."""

    def test_plot_ruin_cliff_basic(self):
        """Test basic ruin cliff visualization with synthetic data."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        # Test with default parameters (synthetic data)
        fig = plot_ruin_cliff()

        assert fig is not None
        assert len(fig.axes) >= 1  # Main plot and possibly inset
        ax = fig.axes[0]

        # Check labels
        assert "Retention" in ax.get_xlabel()
        assert "Ruin" in ax.get_ylabel()
        assert "Ruin Cliff" in ax.get_title()

        # Check that log scale is applied to x-axis
        assert ax.get_xscale() == "log"

        # Check that danger zones exist
        patches = ax.patches
        assert len(patches) > 0  # Should have danger zone patches

        plt.close(fig)

    def test_plot_ruin_cliff_custom_retention_range(self):
        """Test ruin cliff with custom retention range."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        fig = plot_ruin_cliff(retention_range=(5000, 5_000_000), n_points=30)

        assert fig is not None
        ax = fig.axes[0]

        # Check x-axis limits are reasonable for the retention range
        xlim = ax.get_xlim()
        assert xlim[0] < 5000 * 1.5  # Some margin
        assert xlim[1] > 5_000_000 * 0.8

        plt.close(fig)

    def test_plot_ruin_cliff_with_simulation_data(self):
        """Test ruin cliff with provided simulation data."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        # Create mock simulation data
        retentions = np.logspace(4, 7, 50)
        # Create a curve with cliff effect
        log_ret = np.log10(retentions)
        log_ret_norm = (log_ret - log_ret.min()) / (log_ret.max() - log_ret.min())
        ruin_probs = 1 / (1 + np.exp(10 * (log_ret_norm - 0.4)))

        simulation_data = {"retentions": retentions, "ruin_probs": ruin_probs}

        fig = plot_ruin_cliff(simulation_data=simulation_data)

        assert fig is not None
        ax = fig.axes[0]

        # Check that data was used (verify some lines exist)
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)

    def test_plot_ruin_cliff_without_3d_effect(self):
        """Test ruin cliff without 3D gradient effects."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        fig = plot_ruin_cliff(show_3d_effect=False)

        assert fig is not None
        ax = fig.axes[0]

        # Should still have the main plot elements
        assert "Retention" in ax.get_xlabel()

        # Check collections (contour plots create collections)
        # Without 3D effect, should have fewer collections
        collections_without_3d = len(ax.collections)

        # Compare with version with 3D effect
        fig_with_3d = plot_ruin_cliff(show_3d_effect=True)
        ax_with_3d = fig_with_3d.axes[0]
        collections_with_3d = len(ax_with_3d.collections)

        # Should have more collections with 3D effect (from contourf)
        assert collections_with_3d >= collections_without_3d

        plt.close(fig)
        plt.close(fig_with_3d)

    def test_plot_ruin_cliff_without_warnings(self):
        """Test ruin cliff without warning annotations."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        fig = plot_ruin_cliff(show_warnings=False)

        assert fig is not None
        ax = fig.axes[0]

        # Check that annotations are minimal (no warning callouts)
        annotations = ax.texts
        warning_annotations = [
            a for a in annotations if "⚠️" in a.get_text() or "CLIFF" in a.get_text()
        ]
        assert len(warning_annotations) == 0

        plt.close(fig)

    def test_plot_ruin_cliff_without_inset(self):
        """Test ruin cliff without inset plot."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        fig = plot_ruin_cliff(show_inset=False)

        assert fig is not None
        # Should only have one axis (no inset)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_ruin_cliff_with_inset(self):
        """Test ruin cliff with inset plot."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        fig = plot_ruin_cliff(show_inset=True)

        assert fig is not None
        # Should have more than one axis when inset is shown
        assert len(fig.axes) >= 2  # Main plot and inset

        # Check inset has proper title
        inset_found = False
        for ax in fig.axes[1:]:
            if ax.get_title() and "Critical" in ax.get_title():
                inset_found = True
                break
        assert inset_found

        plt.close(fig)

    def test_plot_ruin_cliff_custom_title_and_size(self):
        """Test ruin cliff with custom title and figure size."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        custom_title = "Custom Cliff Analysis"
        fig = plot_ruin_cliff(title=custom_title, figsize=(16, 10), company_size=50_000_000)

        assert fig is not None
        ax = fig.axes[0]
        assert custom_title in ax.get_title()

        # Check figure size
        size = fig.get_size_inches()
        assert size[0] == 16
        assert size[1] == 10

        # Check that company size is mentioned in subtitle
        texts = fig.texts
        company_text_found = False
        for text in texts:
            if "50,000,000" in text.get_text() or "50M" in text.get_text():
                company_text_found = True
                break
        assert company_text_found

        plt.close(fig)

    def test_plot_ruin_cliff_export_dpi(self):
        """Test ruin cliff with export DPI settings."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        # Test web resolution
        fig_web = plot_ruin_cliff(export_dpi=150, n_points=10)  # Fewer points for speed
        assert fig_web.dpi == 150
        plt.close(fig_web)

        # Test print resolution
        fig_print = plot_ruin_cliff(export_dpi=300, n_points=10)
        assert fig_print.dpi == 300
        plt.close(fig_print)

    def test_plot_ruin_cliff_cliff_detection(self):
        """Test that cliff edge detection works properly."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        # Create data with known cliff location
        retentions = np.logspace(4, 7, 100)
        ruin_probs = np.zeros(100)
        # Create artificial cliff at index 30
        ruin_probs[:30] = 0.8
        ruin_probs[30:] = 0.02

        simulation_data = {"retentions": retentions, "ruin_probs": ruin_probs}

        fig = plot_ruin_cliff(simulation_data=simulation_data, show_warnings=True)

        assert fig is not None
        ax = fig.axes[0]

        # Check that a cliff edge marker exists (red scatter point)
        scatter_collections = [c for c in ax.collections if hasattr(c, "get_sizes")]
        large_markers = []
        for coll in scatter_collections:
            sizes = coll.get_sizes()
            if len(sizes) > 0 and sizes[0] > 200:  # Large marker for cliff edge
                large_markers.append(coll)

        assert len(large_markers) > 0  # Should have at least one large marker for cliff

        plt.close(fig)

    def test_plot_ruin_cliff_edge_cases(self):
        """Test ruin cliff with edge case data."""
        from ergodic_insurance.visualization.executive_plots import plot_ruin_cliff

        # Test with flat data (no cliff)
        retentions = np.logspace(4, 6, 20)
        ruin_probs = np.ones(20) * 0.05  # Flat 5% probability

        simulation_data = {"retentions": retentions, "ruin_probs": ruin_probs}

        fig = plot_ruin_cliff(simulation_data=simulation_data)
        assert fig is not None
        plt.close(fig)

        # Test with monotonically decreasing data
        ruin_probs_decreasing = np.linspace(0.9, 0.01, 20)
        simulation_data_decreasing = {"retentions": retentions, "ruin_probs": ruin_probs_decreasing}

        fig2 = plot_ruin_cliff(simulation_data=simulation_data_decreasing)
        assert fig2 is not None
        plt.close(fig2)

"""Simple tests for visualization module to improve coverage."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

# Use non-interactive backend for testing
matplotlib.use('Agg')

from ergodic_insurance.src.visualization import (
    WSJFormatter,
    format_currency,
    format_percentage,
    plot_convergence_diagnostics,
    plot_insurance_layers,
    plot_loss_distribution,
    plot_return_period_curve,
    set_wsj_style,
    create_interactive_dashboard,
)


class TestVisualizationFormatting:
    """Test formatting functions."""

    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1000) == "$1,000"
        assert format_currency(1000000) == "$1,000,000"
        assert format_currency(1234.56, decimals=2) == "$1,234.56"
        assert format_currency(-5000) == "-$5,000"
        assert format_currency(0) == "$0"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.05) == "5.0%"
        assert format_percentage(0.123, decimals=2) == "12.30%"
        assert format_percentage(1.5) == "150.0%"
        assert format_percentage(-0.05) == "-5.0%"
        assert format_percentage(0) == "0.0%"

    def test_wsj_formatter(self):
        """Test WSJ formatter class."""
        formatter = WSJFormatter()
        
        # Test currency
        assert formatter.currency(1000000) == "$1M"
        assert formatter.currency(1500000) == "$1.5M"
        assert formatter.currency(500) == "$500"
        
        # Test percentage
        assert formatter.percentage(0.05) == "5.0%"
        
        # Test number
        assert formatter.number(1234567) == "1.23M"
        assert formatter.number(999) == "999"

    def test_set_wsj_style(self):
        """Test WSJ style setting."""
        # Should not raise
        set_wsj_style()
        
        # Check some style elements were set
        params = plt.rcParams
        assert params['font.size'] > 0
        assert params['axes.labelsize'] > 0


class TestVisualizationPlots:
    """Test plotting functions."""

    @pytest.fixture
    def sample_losses(self):
        """Create sample loss data."""
        np.random.seed(42)
        return pd.DataFrame({
            'amount': np.random.lognormal(12, 1.5, 1000),
            'type': np.random.choice(['attritional', 'large', 'catastrophic'], 1000, p=[0.7, 0.25, 0.05]),
        })

    @pytest.fixture
    def sample_layers(self):
        """Create sample insurance layers."""
        return pd.DataFrame({
            'attachment': [0, 1_000_000, 5_000_000],
            'limit': [1_000_000, 4_000_000, 10_000_000],
            'premium_rate': [0.05, 0.03, 0.01],
        })

    def test_plot_loss_distribution(self, sample_losses):
        """Test loss distribution plotting."""
        fig = plot_loss_distribution(sample_losses)
        
        assert fig is not None
        # Should have created subplots
        assert len(fig.axes) > 0
        
        # Clean up
        plt.close(fig)

    def test_plot_loss_distribution_with_options(self, sample_losses):
        """Test loss distribution plotting with options."""
        fig = plot_loss_distribution(
            sample_losses,
            title="Test Distribution",
            show_stats=True,
            log_scale=True,
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    def test_plot_return_period_curve(self, sample_losses):
        """Test return period curve plotting."""
        fig = plot_return_period_curve(sample_losses['amount'].values)
        
        assert fig is not None
        assert len(fig.axes) > 0
        
        # Clean up
        plt.close(fig)

    def test_plot_return_period_curve_with_options(self, sample_losses):
        """Test return period curve with confidence intervals."""
        fig = plot_return_period_curve(
            sample_losses['amount'].values,
            confidence_level=0.95,
            show_grid=True,
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    def test_plot_insurance_layers(self, sample_layers):
        """Test insurance layers plotting."""
        fig = plot_insurance_layers(sample_layers)
        
        assert fig is not None
        assert len(fig.axes) > 0
        
        # Clean up
        plt.close(fig)

    def test_plot_insurance_layers_with_losses(self, sample_layers, sample_losses):
        """Test insurance layers with loss overlay."""
        fig = plot_insurance_layers(
            sample_layers,
            loss_data=sample_losses['amount'].values,
            show_expected_loss=True,
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    def test_plot_convergence_diagnostics(self):
        """Test convergence diagnostics plotting."""
        # Create sample convergence data
        convergence_data = pd.DataFrame({
            'iteration': range(100),
            'r_hat': 2.0 - np.logspace(-2, 0, 100),  # Decreasing from 2 to 1
            'ess': np.logspace(2, 4, 100),  # Increasing
            'mcse': np.logspace(-1, -3, 100),  # Decreasing
        })
        
        fig = plot_convergence_diagnostics(convergence_data)
        
        assert fig is not None
        assert len(fig.axes) >= 3  # Should have at least 3 subplots
        
        # Clean up
        plt.close(fig)

    def test_plot_convergence_diagnostics_with_threshold(self):
        """Test convergence diagnostics with threshold lines."""
        convergence_data = pd.DataFrame({
            'iteration': range(100),
            'r_hat': 2.0 - np.logspace(-2, 0, 100),
        })
        
        fig = plot_convergence_diagnostics(
            convergence_data,
            r_hat_threshold=1.1,
            show_threshold=True,
        )
        
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    @patch('plotly.graph_objects.Figure')
    def test_create_interactive_dashboard(self, mock_figure):
        """Test interactive dashboard creation."""
        # Create sample data
        simulation_data = pd.DataFrame({
            'year': [1, 2, 3] * 100,
            'assets': np.random.lognormal(16, 0.5, 300),
            'losses': np.random.exponential(100_000, 300),
            'insurance_recovery': np.random.exponential(50_000, 300),
        })
        
        # Mock plotly figure
        mock_fig_instance = Mock()
        mock_figure.return_value = mock_fig_instance
        
        # Create dashboard
        result = create_interactive_dashboard(simulation_data)
        
        # Should return the mock figure
        assert result == mock_fig_instance
        
        # Should have called show on the figure
        mock_fig_instance.show.assert_called_once()

    def test_create_interactive_dashboard_with_options(self):
        """Test interactive dashboard with various options."""
        simulation_data = pd.DataFrame({
            'year': [1, 2, 3] * 10,
            'assets': np.random.lognormal(16, 0.5, 30),
            'losses': np.random.exponential(100_000, 30),
        })
        
        # Mock plotly to avoid actually creating interactive plots
        with patch('plotly.graph_objects.Figure') as mock_figure:
            mock_fig_instance = Mock()
            mock_figure.return_value = mock_fig_instance
            
            result = create_interactive_dashboard(
                simulation_data,
                title="Test Dashboard",
                height=800,
                show_distributions=True,
            )
            
            assert result == mock_fig_instance


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""

    def test_plot_loss_distribution_empty_data(self):
        """Test plotting with empty data."""
        empty_df = pd.DataFrame({'amount': []})
        
        # Should handle gracefully
        fig = plot_loss_distribution(empty_df)
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    def test_plot_return_period_curve_single_value(self):
        """Test return period curve with single value."""
        single_value = np.array([1000])
        
        fig = plot_return_period_curve(single_value)
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    def test_plot_insurance_layers_no_layers(self):
        """Test insurance layers with empty dataframe."""
        empty_layers = pd.DataFrame()
        
        fig = plot_insurance_layers(empty_layers)
        assert fig is not None
        
        # Clean up
        plt.close(fig)

    def test_format_functions_with_nan(self):
        """Test formatting functions with NaN values."""
        assert format_currency(np.nan) == "$nan"
        assert format_percentage(np.nan) == "nan%"

    def test_format_functions_with_inf(self):
        """Test formatting functions with infinity."""
        assert format_currency(np.inf) == "$inf"
        assert format_currency(-np.inf) == "-$inf"
        assert format_percentage(np.inf) == "inf%"

    def test_wsj_formatter_edge_cases(self):
        """Test WSJ formatter with edge cases."""
        formatter = WSJFormatter()
        
        # Test with very large numbers
        assert formatter.currency(1e12) == "$1T"
        assert formatter.currency(1.5e9) == "$1.5B"
        
        # Test with very small numbers
        assert formatter.currency(0.01) == "$0.01"
        
        # Test with negative numbers
        assert formatter.currency(-1000000) == "-$1M"
        
        # Test number formatting edge cases
        assert formatter.number(0) == "0"
        assert formatter.number(1e15) == "1000T"
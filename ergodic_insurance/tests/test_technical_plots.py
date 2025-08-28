"""Tests for technical appendix visualization functions.

This module tests the technical visualization functions including convergence
diagnostics, loss distribution validation, and Monte Carlo convergence analysis.
"""

from unittest.mock import MagicMock, patch

from matplotlib.figure import Figure
import numpy as np
import pytest

from ergodic_insurance.src.visualization.technical_plots import (
    plot_enhanced_convergence_diagnostics,
    plot_loss_distribution_validation,
    plot_monte_carlo_convergence,
    plot_trace_plots,
)


class TestTracePlots:
    """Test suite for trace plot visualization."""

    def test_trace_plots_single_chain_single_param(self):
        """Test trace plots with single chain and parameter."""
        # Generate test data
        np.random.seed(42)
        chains = np.random.randn(100)  # 1D array

        # Create plot
        fig = plot_trace_plots(chains, burn_in=20)

        # Assertions
        assert isinstance(fig, Figure)
        assert len(fig.axes) > 0

        # Check that burn-in line is present
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert any(line.get_xdata()[0] == 20 for line in lines if len(line.get_xdata()) == 2)

    def test_trace_plots_multiple_chains_multiple_params(self):
        """Test trace plots with multiple chains and parameters."""
        # Generate test data
        np.random.seed(42)
        chains = np.random.randn(4, 500, 3)  # 4 chains, 500 iterations, 3 parameters
        param_names = ["alpha", "beta", "sigma"]

        # Create plot
        fig = plot_trace_plots(chains, parameter_names=param_names, burn_in=100)

        # Assertions
        assert isinstance(fig, Figure)
        # Should have at least 3 subplots for 3 parameters
        assert len([ax for ax in fig.axes if ax.get_visible()]) >= 3

        # Check titles contain parameter names
        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        for name in param_names:
            assert any(name in title for title in titles)

    def test_trace_plots_2d_input(self):
        """Test trace plots with 2D input (single chain, multiple parameters)."""
        # Generate test data
        np.random.seed(42)
        chains = np.random.randn(200, 2)  # 200 iterations, 2 parameters

        # Create plot
        fig = plot_trace_plots(chains)

        # Assertions
        assert isinstance(fig, Figure)
        assert len([ax for ax in fig.axes if ax.get_visible()]) >= 2


class TestLossDistributionValidation:
    """Test suite for loss distribution validation plots."""

    def test_loss_distribution_validation_basic(self):
        """Test basic loss distribution validation plot."""
        # Generate test data
        np.random.seed(42)
        attritional_losses = np.random.lognormal(10, 1, 1000)
        large_losses = np.random.lognormal(15, 2, 100)

        # Create plot
        fig = plot_loss_distribution_validation(attritional_losses, large_losses)

        # Assertions
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4  # 2x2 grid

        # Check that all subplots have content
        for ax in fig.axes:
            assert len(ax.get_lines()) > 0 or len(ax.collections) > 0

    def test_loss_distribution_validation_with_dist_params(self):
        """Test loss distribution validation with theoretical distribution parameters."""
        # Generate test data
        np.random.seed(42)
        attritional_losses = np.random.lognormal(10, 1, 1000)
        large_losses = np.random.lognormal(15, 2, 100)

        # Define theoretical distributions
        attritional_dist = {"name": "Lognormal", "shape": 1, "loc": 0, "scale": np.exp(10)}
        large_dist = {"name": "Lognormal", "shape": 2, "loc": 0, "scale": np.exp(15)}

        # Create plot
        fig = plot_loss_distribution_validation(
            attritional_losses,
            large_losses,
            attritional_dist=attritional_dist,
            large_dist=large_dist,
        )

        # Assertions
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4

        # Check that K-S test results are displayed
        for ax in fig.axes[:2]:  # Q-Q plots
            # Look for text containing "K-S"
            texts = ax.texts
            assert any("K-S" in text.get_text() for text in texts)

    def test_loss_distribution_validation_cdf_comparison(self):
        """Test that CDF comparison includes goodness-of-fit metrics."""
        # Generate test data
        np.random.seed(42)
        attritional_losses = np.random.lognormal(10, 1, 500)
        large_losses = np.random.lognormal(15, 2, 50)

        # Create plot
        fig = plot_loss_distribution_validation(attritional_losses, large_losses)

        # Check CDF plots (bottom row)
        cdf_axes = fig.axes[2:]
        for ax in cdf_axes:
            # Should have empirical and theoretical lines
            assert len(ax.get_lines()) >= 2

            # Check for MSE and Max Dev metrics
            texts = ax.texts
            assert any("MSE" in text.get_text() for text in texts)
            assert any("Max Dev" in text.get_text() for text in texts)


class TestMonteCarloConvergence:
    """Test suite for Monte Carlo convergence analysis plots."""

    def test_monte_carlo_convergence_basic(self):
        """Test basic Monte Carlo convergence plot."""
        # Generate test data with convergence behavior
        np.random.seed(42)
        n_iterations = 500

        # Simulate converging metrics
        roe_values = 0.08 + 0.02 * np.random.randn(n_iterations) / np.sqrt(
            np.arange(1, n_iterations + 1)
        )
        ruin_prob_values = 0.05 + 0.01 * np.random.randn(n_iterations) / np.sqrt(
            np.arange(1, n_iterations + 1)
        )

        metrics_history = {
            "ROE": roe_values.tolist(),
            "Ruin Probability": ruin_prob_values.tolist(),
        }

        # Create plot
        fig = plot_monte_carlo_convergence(metrics_history)

        # Assertions
        assert isinstance(fig, Figure)
        assert len([ax for ax in fig.axes if ax.get_visible()]) == 2

        # Check that each subplot has running mean and CI bands
        for ax in fig.axes[:2]:
            lines = ax.get_lines()
            assert len(lines) >= 2  # Raw values and running mean

            # Check for confidence bands (fill_between creates patch collections)
            assert len(ax.collections) >= 1

    def test_monte_carlo_convergence_with_thresholds(self):
        """Test Monte Carlo convergence with convergence thresholds."""
        # Generate test data
        np.random.seed(42)
        n_iterations = 200

        metrics_history = {
            "Metric 1": (
                0.5 + 0.1 * np.random.randn(n_iterations) / np.sqrt(np.arange(1, n_iterations + 1))
            ).tolist(),
            "Metric 2": (
                0.3 + 0.05 * np.random.randn(n_iterations) / np.sqrt(np.arange(1, n_iterations + 1))
            ).tolist(),
        }

        thresholds = {"Metric 1": 0.52, "Metric 2": 0.31}

        # Create plot
        fig = plot_monte_carlo_convergence(
            metrics_history, convergence_thresholds=thresholds, log_scale=False
        )

        # Assertions
        assert isinstance(fig, Figure)

        # Check that threshold lines are present
        for idx, (metric_name, threshold) in enumerate(thresholds.items()):
            ax = fig.axes[idx]
            # Look for horizontal line at threshold value
            for line in ax.get_lines():
                y_data = line.get_ydata()
                # Check if it's a horizontal line (all y values the same)
                if len(y_data) >= 2:
                    if np.allclose(y_data[0], threshold, rtol=0.01) and np.allclose(
                        y_data[-1], threshold, rtol=0.01
                    ):
                        break
            else:
                # If no threshold line found in lines, check if label mentions threshold
                labels = [line.get_label() for line in ax.get_lines()]
                assert any("Threshold" in label for label in labels)

    def test_monte_carlo_convergence_status(self):
        """Test convergence status calculation and display."""
        # Generate test data with clear convergence
        np.random.seed(42)
        n_iterations = 150

        # Create data that converges
        converged_data = np.ones(n_iterations) * 0.5
        converged_data[:50] += 0.1 * np.random.randn(50)  # Initial variance
        converged_data[50:] += 0.001 * np.random.randn(100)  # Converged

        metrics_history = {"Converged Metric": converged_data.tolist()}

        # Create plot
        fig = plot_monte_carlo_convergence(metrics_history)

        # Check for convergence status text
        ax = fig.axes[0]
        texts = ax.texts
        # Should have status text if n_iterations > 100
        assert len(texts) > 0
        assert any(
            "Converged" in text.get_text() or "Not converged" in text.get_text() for text in texts
        )


class TestEnhancedConvergenceDiagnostics:
    """Test suite for enhanced convergence diagnostics."""

    def test_enhanced_convergence_diagnostics_basic(self):
        """Test basic enhanced convergence diagnostics plot."""
        # Generate test data
        np.random.seed(42)
        chains = np.random.randn(4, 500, 2)  # 4 chains, 500 iterations, 2 parameters
        param_names = ["mu", "sigma"]

        # Create plot
        fig = plot_enhanced_convergence_diagnostics(
            chains, parameter_names=param_names, burn_in=100
        )

        # Assertions
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 5  # Trace, R-hat, ESS, ACF, MCSE

        # Check that trace plot contains chain lines
        trace_ax = fig.axes[0]
        assert len(trace_ax.get_lines()) >= 4  # At least 4 chains

    @patch("ergodic_insurance.src.convergence.ConvergenceDiagnostics")
    def test_enhanced_convergence_diagnostics_calculations(self, mock_conv_diag):
        """Test that convergence diagnostics are calculated correctly."""
        # Setup mock
        mock_diag_instance = MagicMock()
        mock_diag_instance.calculate_r_hat.return_value = 1.05
        mock_diag_instance.calculate_ess.return_value = 1500.0
        mock_conv_diag.return_value = mock_diag_instance

        # Generate test data
        np.random.seed(42)
        chains = np.random.randn(2, 300, 1)  # 2 chains for R-hat calculation

        # Create plot
        fig = plot_enhanced_convergence_diagnostics(chains)

        # Verify diagnostics were calculated
        assert mock_diag_instance.calculate_r_hat.called
        assert mock_diag_instance.calculate_ess.called

        # Check that values appear on plot
        assert isinstance(fig, Figure)

    def test_enhanced_convergence_diagnostics_single_chain(self):
        """Test enhanced convergence diagnostics with single chain."""
        # Generate test data
        np.random.seed(42)
        chains = np.random.randn(300)  # 1D array (single chain, single parameter)

        # Create plot
        fig = plot_enhanced_convergence_diagnostics(chains)

        # Assertions
        assert isinstance(fig, Figure)

        # Should still create diagnostic plots (except R-hat which needs multiple chains)
        assert len(fig.axes) >= 4  # At least trace, ESS, ACF, MCSE


class TestIntegration:
    """Integration tests for technical plots."""

    def test_all_plots_with_realistic_data(self):
        """Test all technical plots with realistic simulation data."""
        np.random.seed(42)

        # Generate realistic MCMC chains
        n_chains = 4
        n_iterations = 1000
        n_params = 3

        # Simulate chains with burn-in and convergence
        chains = np.zeros((n_chains, n_iterations, n_params))
        for c in range(n_chains):
            for p in range(n_params):
                # Add burn-in period with higher variance
                chains[c, :200, p] = np.random.normal(0, 2, 200)
                # Converged period
                chains[c, 200:, p] = np.random.normal(1, 0.5, 800)

        # Generate loss data
        attritional_losses = np.random.lognormal(10, 1, 1000)
        large_losses = np.random.lognormal(15, 2, 100)

        # Generate convergence history
        metrics_history = {
            "ROE": (0.08 + 0.02 * np.random.randn(500) / np.sqrt(np.arange(1, 501))).tolist(),
            "Ruin Probability": (
                0.05 + 0.01 * np.random.randn(500) / np.sqrt(np.arange(1, 501))
            ).tolist(),
            "Sharpe Ratio": (
                1.2 + 0.3 * np.random.randn(500) / np.sqrt(np.arange(1, 501))
            ).tolist(),
        }

        # Test all plots
        fig1 = plot_trace_plots(chains, burn_in=200)
        assert isinstance(fig1, Figure)

        fig2 = plot_loss_distribution_validation(attritional_losses, large_losses)
        assert isinstance(fig2, Figure)

        fig3 = plot_monte_carlo_convergence(metrics_history)
        assert isinstance(fig3, Figure)

        fig4 = plot_enhanced_convergence_diagnostics(chains, burn_in=200)
        assert isinstance(fig4, Figure)

    def test_plots_handle_edge_cases(self):
        """Test that plots handle edge cases gracefully."""
        # Small data
        small_chains = np.random.randn(10)
        small_losses = np.random.lognormal(10, 1, 10)
        small_history = {"Metric": [0.1, 0.2, 0.15]}

        # All functions should handle small data without errors
        fig1 = plot_trace_plots(small_chains)
        assert isinstance(fig1, Figure)

        fig2 = plot_loss_distribution_validation(small_losses, small_losses)
        assert isinstance(fig2, Figure)

        fig3 = plot_monte_carlo_convergence(small_history)
        assert isinstance(fig3, Figure)

        # Large data
        large_chains = np.random.randn(10, 10000, 5)
        large_history = {f"Metric {i}": np.random.randn(10000).tolist() for i in range(8)}

        fig4 = plot_trace_plots(large_chains[:, ::100, :])  # Subsample for performance
        assert isinstance(fig4, Figure)

        fig5 = plot_monte_carlo_convergence(large_history)
        assert isinstance(fig5, Figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

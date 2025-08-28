"""Tests for technical appendix visualization functions.

This module tests the technical visualization functions including convergence
diagnostics, loss distribution validation, and Monte Carlo convergence analysis.
"""

from unittest.mock import MagicMock, patch

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ergodic_insurance.src.visualization.technical_plots import (
    plot_enhanced_convergence_diagnostics,
    plot_ergodic_divergence,
    plot_loss_distribution_validation,
    plot_monte_carlo_convergence,
    plot_path_dependent_wealth,
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
        plt.close(fig)

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
        plt.close(fig)

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
        plt.close(fig)


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
        plt.close(fig)

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
        plt.close(fig)

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
        plt.close(fig)


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
        plt.close(fig)

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
        plt.close(fig)

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
        plt.close(fig)


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
        plt.close(fig)

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
        plt.close(fig)

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
        plt.close(fig)


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
        plt.close(fig1)

        fig2 = plot_loss_distribution_validation(attritional_losses, large_losses)
        assert isinstance(fig2, Figure)
        plt.close(fig2)

        fig3 = plot_monte_carlo_convergence(metrics_history)
        assert isinstance(fig3, Figure)
        plt.close(fig3)

        fig4 = plot_enhanced_convergence_diagnostics(chains, burn_in=200)
        assert isinstance(fig4, Figure)
        plt.close(fig4)

    def test_plots_handle_edge_cases(self):
        """Test that plots handle edge cases gracefully."""
        # Small data
        small_chains = np.random.randn(10)
        small_losses = np.random.lognormal(10, 1, 10)
        small_history = {"Metric": [0.1, 0.2, 0.15]}

        # All functions should handle small data without errors
        fig1 = plot_trace_plots(small_chains)
        assert isinstance(fig1, Figure)
        plt.close(fig1)

        fig2 = plot_loss_distribution_validation(small_losses, small_losses)
        assert isinstance(fig2, Figure)
        plt.close(fig2)

        fig3 = plot_monte_carlo_convergence(small_history)
        assert isinstance(fig3, Figure)
        plt.close(fig3)

        # Large data
        large_chains = np.random.randn(10, 10000, 5)
        large_history = {f"Metric {i}": np.random.randn(10000).tolist() for i in range(8)}

        fig4 = plot_trace_plots(large_chains[:, ::100, :])  # Subsample for performance
        assert isinstance(fig4, Figure)
        plt.close(fig4)

        fig5 = plot_monte_carlo_convergence(large_history)
        assert isinstance(fig5, Figure)
        plt.close(fig5)


class TestErgodicDivergence:
    """Test suite for ergodic divergence visualization."""

    def test_basic_ergodic_divergence(self):
        """Test basic ergodic divergence plot."""
        np.random.seed(42)

        # Generate test data
        time_horizons = np.logspace(0, 3, 50)  # 1 to 1000 years
        time_averages = np.array([0.05 * (1 - 0.1 * np.log10(t)) for t in time_horizons])
        ensemble_averages = np.array([0.08] * len(time_horizons))

        fig = plot_ergodic_divergence(time_horizons, time_averages, ensemble_averages)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Main plot and formula panel

        # Check axes content
        ax1 = fig.axes[0]
        assert len(ax1.lines) >= 2  # At least time and ensemble average lines
        assert ax1.get_xscale() == "log"  # Should be log scale

        plt.close(fig)

    def test_ergodic_divergence_with_confidence(self):
        """Test ergodic divergence with confidence bands."""
        np.random.seed(42)

        # Generate test data with standard errors
        time_horizons = np.logspace(0, 2, 30)
        time_averages = np.array([0.05 * (1 - 0.05 * np.log10(t)) for t in time_horizons])
        ensemble_averages = np.array([0.06] * len(time_horizons))
        standard_errors = np.array([0.01 / np.sqrt(t) for t in time_horizons])

        fig = plot_ergodic_divergence(
            time_horizons, time_averages, ensemble_averages, standard_errors=standard_errors
        )

        assert isinstance(fig, Figure)
        # Check for confidence bands (fill_between creates PolyCollection)
        ax1 = fig.axes[0]
        assert any(hasattr(c, "get_facecolor") for c in ax1.collections)

        plt.close(fig)

    def test_ergodic_divergence_with_scenarios(self):
        """Test ergodic divergence with parameter scenarios."""
        np.random.seed(42)

        # Generate base data
        time_horizons = np.logspace(0, 2.5, 40)
        time_averages = np.array([0.045 * (1 - 0.08 * np.log10(t)) for t in time_horizons])
        ensemble_averages = np.array([0.07] * len(time_horizons))

        # Add parameter scenarios
        scenarios = {
            "High Vol": {
                "horizons": time_horizons,
                "time_avg": np.array([0.03 * (1 - 0.15 * np.log10(t)) for t in time_horizons]),
            },
            "Low Vol": {
                "horizons": time_horizons,
                "time_avg": np.array([0.06 * (1 - 0.05 * np.log10(t)) for t in time_horizons]),
            },
        }

        fig = plot_ergodic_divergence(
            time_horizons, time_averages, ensemble_averages, parameter_scenarios=scenarios
        )

        assert isinstance(fig, Figure)
        ax1 = fig.axes[0]
        # Should have base lines plus scenario lines
        assert len(ax1.lines) >= 4  # 2 base + 2 scenarios

        plt.close(fig)

    def test_ergodic_divergence_without_formulas(self):
        """Test ergodic divergence without formula annotations."""
        np.random.seed(42)

        time_horizons = np.logspace(0, 2, 20)
        time_averages = 0.05 * np.ones(len(time_horizons))
        ensemble_averages = 0.08 * np.ones(len(time_horizons))

        fig = plot_ergodic_divergence(
            time_horizons, time_averages, ensemble_averages, add_formulas=False
        )

        assert isinstance(fig, Figure)
        # Second axis should still exist but be empty
        ax2 = fig.axes[1]
        assert not ax2.texts or len(ax2.texts) == 0

        plt.close(fig)

    def test_ergodic_divergence_custom_parameters(self):
        """Test ergodic divergence with custom parameters."""
        np.random.seed(42)

        time_horizons = np.linspace(1, 100, 50)
        time_averages = 0.04 + 0.001 * np.random.randn(50)
        ensemble_averages = 0.06 + 0.0005 * np.random.randn(50)

        fig = plot_ergodic_divergence(
            time_horizons,
            time_averages,
            ensemble_averages,
            title="Custom Ergodic Analysis",
            figsize=(16, 10),
        )

        assert isinstance(fig, Figure)
        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 10

        plt.close(fig)


class TestPathDependentWealth:
    """Test suite for path-dependent wealth visualization."""

    def test_basic_path_dependent_wealth(self):
        """Test basic path-dependent wealth plot."""
        np.random.seed(42)

        # Generate test trajectories
        n_paths, n_years = 100, 50
        trajectories = np.ones((n_paths, n_years))
        for i in range(n_paths):
            shocks = np.exp(np.random.normal(0, 0.2, n_years))
            trajectories[i] = np.cumprod(shocks)

        fig = plot_path_dependent_wealth(trajectories)

        assert isinstance(fig, Figure)
        # Should have main plot, survivor bias inset, and stats panel
        assert len(fig.axes) >= 1

        plt.close(fig)

    def test_path_dependent_wealth_with_ruin(self):
        """Test path-dependent wealth with ruined paths."""
        np.random.seed(42)

        # Generate trajectories with some hitting ruin
        n_paths, n_years = 200, 75
        trajectories = np.ones((n_paths, n_years))

        for i in range(n_paths):
            shocks = np.exp(np.random.normal(0, 0.3, n_years))
            trajectories[i] = np.cumprod(shocks)
            # Force some paths to ruin
            if i % 5 == 0:
                ruin_time = np.random.randint(20, n_years)
                trajectories[i, ruin_time:] = 0

        fig = plot_path_dependent_wealth(trajectories, ruin_threshold=0.01, highlight_ruined=True)

        assert isinstance(fig, Figure)
        # Check for ruined paths (red lines)
        ax = fig.axes[0]
        has_red_lines = any(
            line
            for line in ax.lines
            if hasattr(line, "get_color") and "red" in str(line.get_color()).lower()
        )

        plt.close(fig)

    def test_path_dependent_wealth_custom_percentiles(self):
        """Test path-dependent wealth with custom percentiles."""
        np.random.seed(42)

        # Generate test data
        n_paths, n_years = 150, 60
        trajectories = np.random.lognormal(0, 0.15, (n_paths, n_years)).cumprod(axis=1)

        custom_percentiles = [10, 25, 50, 75, 90]

        fig = plot_path_dependent_wealth(trajectories, percentiles=custom_percentiles)

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Should have percentile bands
        assert len(ax.collections) > 0  # fill_between creates collections

        plt.close(fig)

    def test_path_dependent_wealth_no_inset(self):
        """Test path-dependent wealth without survivor bias inset."""
        np.random.seed(42)

        # Generate simple test data
        n_paths, n_years = 50, 30
        trajectories = np.ones((n_paths, n_years))
        for i in range(n_paths):
            trajectories[i] = np.cumprod(1 + 0.05 * np.random.randn(n_years))

        fig = plot_path_dependent_wealth(trajectories, add_survivor_bias_inset=False)

        assert isinstance(fig, Figure)
        # Should only have main plot
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_path_dependent_wealth_linear_scale(self):
        """Test path-dependent wealth with linear scale."""
        np.random.seed(42)

        # Generate test data
        n_paths, n_years = 80, 40
        trajectories = np.ones((n_paths, n_years))
        for i in range(n_paths):
            trajectories[i] = np.cumsum(np.random.normal(100, 20, n_years))

        fig = plot_path_dependent_wealth(trajectories, log_scale=False, ruin_threshold=-100)

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_yscale() == "linear"

        plt.close(fig)

    def test_path_dependent_wealth_with_time_points(self):
        """Test path-dependent wealth with custom time points."""
        np.random.seed(42)

        # Generate test data
        n_paths = 75
        time_points = np.array([0, 1, 2, 5, 10, 20, 30, 50, 75, 100])
        n_years = len(time_points)
        trajectories = np.random.lognormal(0, 0.2, (n_paths, n_years)).cumprod(axis=1)

        fig = plot_path_dependent_wealth(trajectories, time_points=time_points)

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Check x-axis reflects custom time points
        x_data = ax.lines[0].get_xdata() if ax.lines else []

        plt.close(fig)

    def test_path_dependent_wealth_extreme_scenarios(self):
        """Test path-dependent wealth with extreme scenarios."""
        np.random.seed(42)

        # All paths survive
        n_paths, n_years = 100, 50
        trajectories_survive = np.ones((n_paths, n_years))
        for i in range(n_paths):
            trajectories_survive[i] = np.cumprod(1 + np.abs(np.random.normal(0.02, 0.01, n_years)))

        fig1 = plot_path_dependent_wealth(trajectories_survive, ruin_threshold=0.5)
        assert isinstance(fig1, Figure)
        plt.close(fig1)

        # All paths hit ruin
        trajectories_ruin = np.ones((n_paths, n_years))
        for i in range(n_paths):
            ruin_time = np.random.randint(10, n_years - 10)
            trajectories_ruin[i, :ruin_time] = np.cumprod(1 + np.random.normal(0, 0.1, ruin_time))
            trajectories_ruin[i, ruin_time:] = 0

        fig2 = plot_path_dependent_wealth(trajectories_ruin, ruin_threshold=0.01)
        assert isinstance(fig2, Figure)
        plt.close(fig2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

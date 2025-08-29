"""Tests for real-time convergence plotting module."""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ergodic_insurance.src.convergence_plots import RealTimeConvergencePlotter


class TestRealTimeConvergencePlotter:
    """Test suite for RealTimeConvergencePlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create plotter instance."""
        return RealTimeConvergencePlotter(
            n_parameters=2, n_chains=3, buffer_size=500, update_interval=50
        )

    @pytest.fixture
    def sample_chains(self):
        """Create sample chain data."""
        np.random.seed(42)
        n_chains = 3
        n_iterations = 1000
        n_params = 2

        chains = np.zeros((n_chains, n_iterations, n_params))
        for c in range(n_chains):
            for p in range(n_params):
                chains[c, 0, p] = np.random.randn()
                for i in range(1, n_iterations):
                    chains[c, i, p] = 0.5 * chains[c, i - 1, p] + np.random.randn()

        return chains

    def test_initialization(self):
        """Test plotter initialization."""
        plotter = RealTimeConvergencePlotter(
            n_parameters=3, n_chains=4, buffer_size=1000, update_interval=100
        )

        assert plotter.n_parameters == 3
        assert plotter.n_chains == 4
        assert plotter.buffer_size == 1000
        assert plotter.update_interval == 100
        assert len(plotter.trace_buffers) == 3
        assert len(plotter.trace_buffers[0]) == 4

    def test_setup_figure_basic(self, plotter):
        """Test basic figure setup."""
        fig = plotter.setup_figure(show_diagnostics=False)

        assert isinstance(fig, Figure)
        assert plotter.fig is fig
        assert len(plotter.axes) == 2  # 2 parameters, trace plots only
        assert "trace_0" in plotter.axes
        assert "trace_1" in plotter.axes

        plt.close(fig)

    def test_setup_figure_with_diagnostics(self, plotter):
        """Test figure setup with diagnostics."""
        fig = plotter.setup_figure(parameter_names=["Param A", "Param B"], show_diagnostics=True)

        assert isinstance(fig, Figure)
        assert len(plotter.axes) == 6  # 2 params x 3 plot types
        assert "trace_0" in plotter.axes
        assert "rhat_0" in plotter.axes
        assert "ess_0" in plotter.axes
        assert "trace_1" in plotter.axes
        assert "rhat_1" in plotter.axes
        assert "ess_1" in plotter.axes

        plt.close(fig)

    def test_update_data(self, plotter):
        """Test data update functionality."""
        # Setup figure first
        plotter.setup_figure()

        # Create sample data
        chains_data = np.array(
            [[1.0, 2.0], [1.5, 2.5], [1.2, 2.2]]  # Chain 1  # Chain 2  # Chain 3
        )

        diagnostics = {"r_hat": [1.05, 1.02], "ess": [500, 800]}

        # Update data
        plotter.update_data(100, chains_data, diagnostics)

        assert plotter.iteration_count == 100
        assert len(plotter.iteration_buffer) == 1
        assert len(plotter.r_hat_history[0]) == 1
        assert plotter.r_hat_history[0][0] == 1.05
        assert len(plotter.ess_history[1]) == 1
        assert plotter.ess_history[1][0] == 800

        plt.close(plotter.fig)

    def test_plot_static_convergence(self, plotter, sample_chains):
        """Test static convergence plotting."""
        fig = plotter.plot_static_convergence(sample_chains, burn_in=100, thin=10)

        assert isinstance(fig, Figure)
        # Check that axes were created
        axes = fig.get_axes()
        assert len(axes) > 0

        plt.close(fig)

    def test_plot_ess_evolution(self, plotter):
        """Test ESS evolution plotting."""
        ess_values = [100, 200, 400, 600, 800, 900, 950, 1000, 1050, 1100]

        fig = plotter.plot_ess_evolution(ess_values, target_ess=1000)

        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) == 2  # Two subplots

        plt.close(fig)

    def test_plot_autocorrelation_surface(self, plotter, sample_chains):
        """Test 3D autocorrelation surface plotting."""
        fig = plotter.plot_autocorrelation_surface(sample_chains, max_lag=20, param_idx=0)

        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) > 0

        plt.close(fig)

    def test_create_convergence_dashboard(self, plotter, sample_chains):
        """Test comprehensive dashboard creation."""
        diagnostics = {
            "r_hat_0": [1.1, 1.05, 1.02, 1.01],
            "r_hat_1": [1.15, 1.08, 1.03, 1.01],
            "ess_0": [100, 300, 600, 1000],
            "ess_1": [150, 400, 700, 1100],
        }

        fig = plotter.create_convergence_dashboard(
            sample_chains, diagnostics, parameter_names=["Alpha", "Beta"]
        )

        assert isinstance(fig, Figure)
        # Should have created multiple subplots
        axes = fig.get_axes()
        assert len(axes) > 0

        # Check that summary text was added
        texts = fig.texts
        assert len(texts) > 0

        plt.close(fig)

    def test_calculate_running_variance(self, plotter):
        """Test running variance calculation."""
        chain = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        running_var = plotter._calculate_running_variance(chain)

        assert len(running_var) == len(chain)
        assert running_var[0] == 0  # First element has no variance
        assert running_var[-1] > 0  # Should have variance at end
        # Variance should stabilize
        assert abs(running_var[-1] - np.var(chain, ddof=1)) < 0.5

    def test_calculate_acf(self, plotter):
        """Test autocorrelation function calculation."""
        # Create autocorrelated chain
        np.random.seed(42)
        n = 100
        chain = np.zeros(n)
        chain[0] = np.random.randn()
        for i in range(1, n):
            chain[i] = 0.7 * chain[i - 1] + np.random.randn()

        acf = plotter._calculate_acf(chain, max_lag=10)

        assert len(acf) == 11
        assert acf[0] == 1.0
        # Should decay for autocorrelated chain
        assert acf[1] > acf[5]
        assert all(a <= 1.0 for a in acf)
        assert all(a >= -1.0 for a in acf)

    def test_generate_convergence_summary(self, plotter):
        """Test convergence summary generation."""
        chains = np.random.randn(2, 100, 2)
        diagnostics = {"r_hat_0": [1.05], "r_hat_1": [1.12], "ess_0": [1200], "ess_1": [800]}

        summary = plotter._generate_convergence_summary(chains, diagnostics)

        assert isinstance(summary, str)
        assert "CONVERGENCE SUMMARY" in summary
        assert "Chains: 2" in summary
        assert "Iterations: 100" in summary
        assert "Parameters: 2" in summary
        assert "✓" in summary  # Param 0 converged
        assert "✗" in summary  # Param 1 not converged
        assert "NOT CONVERGED" in summary  # Overall status

    def test_buffer_overflow(self):
        """Test that buffers respect max length."""
        # Create plotter with small buffer size
        plotter = RealTimeConvergencePlotter(n_parameters=2, n_chains=3, buffer_size=10)
        plotter.setup_figure()

        # Add more data than buffer size
        for i in range(20):
            chains_data = np.random.randn(3, 2)
            plotter.update_data(i, chains_data)

        # Check buffers don't exceed max size (deque with maxlen automatically limits)
        assert len(plotter.iteration_buffer) <= plotter.buffer_size
        assert all(
            len(plotter.trace_buffers[p][c]) <= plotter.buffer_size
            for p in range(2)
            for c in range(3)
        )

        plt.close(plotter.fig)

    def test_plot_with_single_chain(self):
        """Test plotting with single chain."""
        plotter = RealTimeConvergencePlotter(n_parameters=1, n_chains=1)

        fig = plotter.setup_figure()

        chains_data = np.array([[1.5]])
        plotter.update_data(1, chains_data)

        assert len(plotter.trace_buffers[0][0]) == 1

        plt.close(fig)

    def test_plot_with_no_diagnostics(self, plotter):
        """Test plotting without diagnostics."""
        plotter.setup_figure(show_diagnostics=False)

        chains_data = np.random.randn(3, 2)
        plotter.update_data(1, chains_data, diagnostics=None)

        # Should still update trace buffers
        assert len(plotter.iteration_buffer) == 1
        # But no diagnostic history
        assert len(plotter.r_hat_history[0]) == 0
        assert len(plotter.ess_history[0]) == 0

        plt.close(plotter.fig)

    def test_edge_cases_in_calculations(self, plotter):
        """Test edge cases in calculation methods."""
        # Empty chain
        acf = plotter._calculate_acf(np.array([]), max_lag=5)
        assert len(acf) == 1  # Should return single value for empty chain

        # Single value
        acf = plotter._calculate_acf(np.array([1.0]), max_lag=5)
        assert len(acf) == 1
        assert acf[0] == 1.0

        # Constant chain
        constant_chain = np.ones(50)
        acf = plotter._calculate_acf(constant_chain, max_lag=10)
        assert acf[0] == 1.0
        # Note: constant chain has undefined ACF for lag > 0 due to zero variance

    def test_convergence_dashboard_edge_cases(self, plotter):
        """Test dashboard with edge cases."""
        # Single iteration chain
        chains = np.random.randn(2, 1, 2)
        diagnostics: Dict[str, Any] = {}

        fig = plotter.create_convergence_dashboard(chains, diagnostics)
        assert isinstance(fig, Figure)

        plt.close(fig)

        # Empty diagnostics
        chains = np.random.randn(2, 100, 2)
        fig = plotter.create_convergence_dashboard(chains, {})
        assert isinstance(fig, Figure)

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_figure_not_shown_automatically(self, mock_show, plotter):
        """Test that figures are not automatically shown."""
        fig = plotter.setup_figure()
        mock_show.assert_not_called()

        fig = plotter.plot_static_convergence(np.random.randn(2, 100, 2))
        mock_show.assert_not_called()

        plt.close("all")

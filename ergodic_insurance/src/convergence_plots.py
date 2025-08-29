"""Real-time convergence visualization for Monte Carlo simulations.

This module provides real-time plotting capabilities for monitoring convergence
during long-running simulations with minimal computational overhead.
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import warnings

from matplotlib import animation, gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


class RealTimeConvergencePlotter:
    """Real-time convergence plotting with minimal overhead.

    Provides animated visualization of convergence diagnostics during
    Monte Carlo simulations with efficient updating mechanisms.
    """

    def __init__(
        self,
        n_parameters: int = 1,
        n_chains: int = 1,
        buffer_size: int = 1000,
        update_interval: int = 100,
        figsize: Tuple[float, float] = (12, 8),
    ):
        """Initialize real-time plotter.

        Args:
            n_parameters: Number of parameters to monitor
            n_chains: Number of MCMC chains
            buffer_size: Size of data buffer for efficiency
            update_interval: Update plot every N iterations
            figsize: Figure size for plots
        """
        self.n_parameters = n_parameters
        self.n_chains = n_chains
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.figsize = figsize

        # Data storage with efficient circular buffers
        self.trace_buffers: List[List[Deque[float]]] = [
            [deque(maxlen=buffer_size) for _ in range(n_chains)] for _ in range(n_parameters)
        ]
        self.iteration_buffer: Deque[int] = deque(maxlen=buffer_size)

        # Convergence metrics storage
        self.r_hat_history: List[List[float]] = [[] for _ in range(n_parameters)]
        self.ess_history: List[List[float]] = [[] for _ in range(n_parameters)]
        self.mean_history: List[List[float]] = [[] for _ in range(n_parameters)]
        self.variance_history: List[List[float]] = [[] for _ in range(n_parameters)]

        # Plotting elements
        self.fig: Optional[Figure] = None
        self.axes: Optional[Dict[str, Axes]] = None
        self.lines: Dict[str, List[Line2D]] = {}
        self.texts: Dict[str, Any] = {}
        self.patches: Dict[str, Any] = {}

        # Animation control
        self.animation: Optional[animation.FuncAnimation] = None
        self.is_running = False
        self.iteration_count = 0

        # Parameter names (initialized in setup_figure)
        self.parameter_names: Optional[List[str]] = None

    def setup_figure(
        self, parameter_names: Optional[List[str]] = None, show_diagnostics: bool = True
    ) -> Figure:
        """Setup the figure with subplots for real-time monitoring.

        Args:
            parameter_names: Names of parameters being monitored
            show_diagnostics: Whether to show diagnostic panels

        Returns:
            Matplotlib figure object
        """
        if parameter_names is None:
            parameter_names = [f"Parameter {i+1}" for i in range(self.n_parameters)]

        self.parameter_names = parameter_names

        # Create figure with custom layout
        self.fig = plt.figure(figsize=self.figsize)

        if show_diagnostics:
            # Complex layout with diagnostics
            gs = gridspec.GridSpec(
                3, self.n_parameters, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3
            )
        else:
            # Simple trace plots only
            gs = gridspec.GridSpec(1, self.n_parameters, wspace=0.3)

        self.axes = {}

        # Create trace plot axes
        for i, param_name in enumerate(parameter_names):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_title(param_name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            self.axes[f"trace_{i}"] = ax

            # Initialize trace lines
            self.lines[f"trace_{i}"] = []
            for chain in range(self.n_chains):
                (line,) = ax.plot([], [], alpha=0.7, linewidth=0.8)
                self.lines[f"trace_{i}"].append(line)

        if show_diagnostics:
            # Create R-hat axes
            for i in range(self.n_parameters):
                ax = self.fig.add_subplot(gs[1, i])
                ax.set_xlabel("Iteration")
                ax.set_ylabel("R-hat")
                ax.set_ylim(0.95, 1.2)
                ax.axhline(y=1.1, color="red", linestyle="--", alpha=0.5)
                ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
                ax.grid(True, alpha=0.3)
                self.axes[f"rhat_{i}"] = ax

                # Initialize R-hat line
                (line,) = ax.plot([], [], "b-", linewidth=1.5)
                self.lines[f"rhat_{i}"] = [line]

            # Create ESS axes
            for i in range(self.n_parameters):
                ax = self.fig.add_subplot(gs[2, i])
                ax.set_xlabel("Iteration")
                ax.set_ylabel("ESS")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                self.axes[f"ess_{i}"] = ax

                # Initialize ESS line
                (line,) = ax.plot([], [], "g-", linewidth=1.5)
                self.lines[f"ess_{i}"] = [line]

        self.fig.suptitle("Real-Time Convergence Monitor", fontsize=14, fontweight="bold")

        return self.fig

    def update_data(
        self,
        iteration: int,
        chains_data: np.ndarray,
        diagnostics: Optional[Dict[str, List[float]]] = None,
    ):
        """Update data buffers with new samples.

        Args:
            iteration: Current iteration number
            chains_data: Array of shape (n_chains, n_parameters)
            diagnostics: Optional dictionary with R-hat, ESS values
        """
        self.iteration_count = iteration
        self.iteration_buffer.append(iteration)

        # Update trace buffers
        for param_idx in range(self.n_parameters):
            for chain_idx in range(self.n_chains):
                value = (
                    chains_data[chain_idx, param_idx]
                    if chains_data.ndim > 1
                    else chains_data[chain_idx]
                )
                self.trace_buffers[param_idx][chain_idx].append(value)

        # Update diagnostics if provided
        if diagnostics:
            if "r_hat" in diagnostics:
                for i, r_hat in enumerate(diagnostics["r_hat"]):
                    if i < self.n_parameters:
                        self.r_hat_history[i].append(r_hat)

            if "ess" in diagnostics:
                for i, ess in enumerate(diagnostics["ess"]):
                    if i < self.n_parameters:
                        self.ess_history[i].append(ess)

    def plot_static_convergence(  # pylint: disable=too-many-locals
        self, chains: np.ndarray, burn_in: Optional[int] = None, thin: int = 1
    ) -> Figure:
        """Create static convergence plots for completed chains.

        Args:
            chains: Array of shape (n_chains, n_iterations, n_parameters)
            burn_in: Burn-in period to highlight
            thin: Thinning interval for display

        Returns:
            Figure with convergence plots
        """
        n_chains, n_iterations, n_params = chains.shape
        iterations = np.arange(0, n_iterations, thin)

        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.5))
        gs = gridspec.GridSpec(4, n_params, height_ratios=[2, 1, 1, 1], hspace=0.4, wspace=0.3)

        for param_idx in range(n_params):
            # Trace plots
            ax_trace = fig.add_subplot(gs[0, param_idx])
            for chain_idx in range(n_chains):
                ax_trace.plot(
                    iterations,
                    chains[chain_idx, ::thin, param_idx],
                    alpha=0.7,
                    linewidth=0.8,
                    label=f"Chain {chain_idx + 1}",
                )

            if burn_in:
                ax_trace.axvspan(0, burn_in, alpha=0.2, color="red", label="Burn-in")

            ax_trace.set_title(f"Parameter {param_idx + 1}")
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Value")
            ax_trace.grid(True, alpha=0.3)
            if param_idx == 0:
                ax_trace.legend(loc="upper right", fontsize=8)

            # Running mean plot
            ax_mean = fig.add_subplot(gs[1, param_idx])
            for chain_idx in range(n_chains):
                running_mean = np.cumsum(chains[chain_idx, :, param_idx]) / np.arange(
                    1, n_iterations + 1
                )
                ax_mean.plot(iterations, running_mean[::thin], alpha=0.7, linewidth=1)

            ax_mean.set_xlabel("Iteration")
            ax_mean.set_ylabel("Running Mean")
            ax_mean.grid(True, alpha=0.3)

            # Running variance plot
            ax_var = fig.add_subplot(gs[2, param_idx])
            for chain_idx in range(n_chains):
                running_var = self._calculate_running_variance(chains[chain_idx, :, param_idx])
                ax_var.plot(iterations, running_var[::thin], alpha=0.7, linewidth=1)

            ax_var.set_xlabel("Iteration")
            ax_var.set_ylabel("Running Variance")
            ax_var.set_yscale("log")
            ax_var.grid(True, alpha=0.3)

            # Autocorrelation plot
            ax_acf = fig.add_subplot(gs[3, param_idx])
            max_lag = min(50, n_iterations // 4)

            for chain_idx in range(n_chains):
                if burn_in:
                    chain_data = chains[chain_idx, burn_in:, param_idx]
                else:
                    chain_data = chains[chain_idx, :, param_idx]

                acf = self._calculate_acf(chain_data, max_lag)
                ax_acf.plot(acf, alpha=0.7, linewidth=1)

            ax_acf.set_xlabel("Lag")
            ax_acf.set_ylabel("ACF")
            ax_acf.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax_acf.grid(True, alpha=0.3)

        fig.suptitle("Convergence Diagnostics", fontsize=14, fontweight="bold")

        return fig

    def plot_ess_evolution(
        self,
        ess_values: Union[List[float], np.ndarray],
        iterations: Optional[np.ndarray] = None,
        target_ess: float = 1000,
    ) -> Figure:
        """Plot evolution of effective sample size.

        Args:
            ess_values: ESS values over iterations
            iterations: Iteration numbers (generated if None)
            target_ess: Target ESS threshold

        Returns:
            Figure with ESS evolution plot
        """
        if iterations is None:
            iterations = np.arange(len(ess_values))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # ESS evolution
        ax1.plot(iterations, ess_values, "b-", linewidth=2, label="ESS")
        ax1.axhline(
            y=target_ess, color="red", linestyle="--", alpha=0.5, label=f"Target ({target_ess})"
        )
        ax1.fill_between(iterations, 0, ess_values, alpha=0.3)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Effective Sample Size")
        ax1.set_title("ESS Evolution")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # ESS per iteration (efficiency)
        ess_per_iter = ess_values / (iterations + 1)
        ax2.plot(iterations, ess_per_iter, "g-", linewidth=2)
        ax2.fill_between(iterations, 0, ess_per_iter, alpha=0.3, color="green")

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("ESS / Iteration")
        ax2.set_title("Sampling Efficiency")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Effective Sample Size Analysis", fontsize=14, fontweight="bold")

        return fig

    def plot_autocorrelation_surface(  # pylint: disable=too-many-locals
        self, chains: np.ndarray, max_lag: int = 50, param_idx: int = 0
    ) -> Figure:
        """Create 3D surface plot of autocorrelation over time.

        Args:
            chains: Array of shape (n_chains, n_iterations, n_parameters)
            max_lag: Maximum lag for ACF
            param_idx: Parameter index to plot

        Returns:
            Figure with 3D autocorrelation surface
        """
        from mpl_toolkits.mplot3d import Axes3D

        n_chains, n_iterations, _ = chains.shape

        # Calculate ACF at different time windows
        window_size = n_iterations // 20
        n_windows = n_iterations // window_size

        acf_matrix = np.zeros((n_windows, max_lag + 1))
        window_centers = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_centers[i] = (start_idx + end_idx) / 2

            # Average ACF across chains
            chain_acfs = []
            for chain_idx in range(n_chains):
                chain_segment = chains[chain_idx, start_idx:end_idx, param_idx]
                acf = self._calculate_acf(chain_segment, max_lag)
                chain_acfs.append(acf)

            acf_matrix[i] = np.mean(chain_acfs, axis=0)

        # Create 3D surface plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        lags = np.arange(max_lag + 1)
        X, Y = np.meshgrid(lags, window_centers)

        surf = ax.plot_surface(X, Y, acf_matrix, cmap="viridis", alpha=0.8)

        ax.set_xlabel("Lag")
        ax.set_ylabel("Iteration")
        ax.set_zlabel("Autocorrelation")
        ax.set_title(f"Autocorrelation Evolution - Parameter {param_idx + 1}")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        return fig

    def create_convergence_dashboard(  # pylint: disable=too-many-locals,too-many-statements
        self,
        chains: np.ndarray,
        diagnostics: Dict[str, Any],
        parameter_names: Optional[List[str]] = None,
    ) -> Figure:
        """Create comprehensive convergence dashboard.

        Args:
            chains: Array of shape (n_chains, n_iterations, n_parameters)
            diagnostics: Dictionary with convergence diagnostics
            parameter_names: Names of parameters

        Returns:
            Figure with comprehensive dashboard
        """
        n_chains, n_iterations, n_params = chains.shape

        if parameter_names is None:
            parameter_names = [f"Param {i+1}" for i in range(n_params)]

        # Create figure with many subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, n_params, height_ratios=[2, 1, 1, 1], hspace=0.4, wspace=0.3)

        # Color palette for chains
        from matplotlib import colormaps

        colors = colormaps["tab10"](np.linspace(0, 1, n_chains))

        for param_idx, param_name in enumerate(parameter_names):
            # 1. Trace plots with density
            ax_trace = fig.add_subplot(gs[0, param_idx])

            for chain_idx in range(n_chains):
                ax_trace.plot(
                    chains[chain_idx, :, param_idx],
                    alpha=0.6,
                    linewidth=0.5,
                    color=colors[chain_idx],
                    label=f"Chain {chain_idx + 1}",
                )

            ax_trace.set_title(param_name, fontweight="bold")
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Value")
            ax_trace.grid(True, alpha=0.3)

            # Add marginal density
            ax_density = ax_trace.twinx()
            for chain_idx in range(n_chains):
                hist, bins = np.histogram(chains[chain_idx, :, param_idx], bins=30, density=True)
                ax_density.barh(
                    bins[:-1], hist * 0.1, height=np.diff(bins), alpha=0.3, color=colors[chain_idx]
                )
            ax_density.set_ylim(ax_trace.get_ylim())
            ax_density.axis("off")

            # 2. R-hat evolution
            if f"r_hat_{param_idx}" in diagnostics:
                ax_rhat = fig.add_subplot(gs[1, param_idx])
                r_hat_values = diagnostics[f"r_hat_{param_idx}"]
                iterations_rhat = np.linspace(0, n_iterations, len(r_hat_values))

                ax_rhat.plot(iterations_rhat, r_hat_values, "b-", linewidth=1.5)
                ax_rhat.axhline(y=1.1, color="red", linestyle="--", alpha=0.5)
                ax_rhat.axhline(y=1.05, color="orange", linestyle="--", alpha=0.5)
                ax_rhat.axhline(y=1.01, color="green", linestyle="--", alpha=0.5)

                # Color background based on convergence
                if len(r_hat_values) > 0:
                    latest_rhat = r_hat_values[-1]
                    if latest_rhat < 1.01:
                        ax_rhat.patch.set_facecolor("#90EE90")
                        ax_rhat.patch.set_alpha(0.1)
                    elif latest_rhat < 1.05:
                        ax_rhat.patch.set_facecolor("#FFD700")
                        ax_rhat.patch.set_alpha(0.1)
                    else:
                        ax_rhat.patch.set_facecolor("#FF6B6B")
                        ax_rhat.patch.set_alpha(0.1)

                ax_rhat.set_xlabel("Iteration")
                ax_rhat.set_ylabel("R-hat")
                ax_rhat.set_ylim(0.98, max(1.2, max(r_hat_values) * 1.1) if r_hat_values else 1.2)
                ax_rhat.grid(True, alpha=0.3)

            # 3. ESS evolution
            if f"ess_{param_idx}" in diagnostics:
                ax_ess = fig.add_subplot(gs[2, param_idx])
                ess_values = diagnostics[f"ess_{param_idx}"]
                iterations_ess = np.linspace(0, n_iterations, len(ess_values))

                ax_ess.plot(iterations_ess, ess_values, "g-", linewidth=1.5)
                ax_ess.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="Target")

                ax_ess.set_xlabel("Iteration")
                ax_ess.set_ylabel("ESS")
                ax_ess.set_yscale("log")
                ax_ess.grid(True, alpha=0.3)

            # 4. Running statistics
            ax_stats = fig.add_subplot(gs[3, param_idx])

            # Calculate and plot running mean with confidence bands
            all_chains = chains[:, :, param_idx].flatten()
            running_mean = np.cumsum(all_chains) / np.arange(1, len(all_chains) + 1)

            # Subsample for plotting efficiency
            subsample = slice(None, None, max(1, len(all_chains) // 1000))
            iterations_stats = np.arange(len(all_chains))[subsample]

            ax_stats.plot(
                iterations_stats, running_mean[subsample], "b-", linewidth=1, label="Mean"
            )

            # Add confidence bands
            running_std = np.array([np.std(all_chains[: i + 1]) for i in range(len(all_chains))])[
                subsample
            ]
            running_se = running_std / np.sqrt(iterations_stats + 1)

            ax_stats.fill_between(
                iterations_stats,
                running_mean[subsample] - 1.96 * running_se,
                running_mean[subsample] + 1.96 * running_se,
                alpha=0.3,
                label="95% CI",
            )

            ax_stats.set_xlabel("Combined Iteration")
            ax_stats.set_ylabel("Running Mean")
            ax_stats.grid(True, alpha=0.3)
            ax_stats.legend(loc="upper right", fontsize=8)

        fig.suptitle("Convergence Dashboard", fontsize=16, fontweight="bold")

        # Add summary text
        summary_text = self._generate_convergence_summary(chains, diagnostics)
        fig.text(
            0.02,
            0.02,
            summary_text,
            fontsize=8,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        return fig

    # Private helper methods

    def _calculate_running_variance(self, chain: np.ndarray) -> np.ndarray:
        """Calculate running variance efficiently."""
        n = len(chain)
        running_var = np.zeros(n)

        mean = 0
        M2 = 0

        for i in range(n):
            delta = chain[i] - mean
            mean += delta / (i + 1)
            delta2 = chain[i] - mean
            M2 += delta * delta2

            if i > 0:
                running_var[i] = M2 / i
            else:
                running_var[i] = 0

        return running_var

    def _calculate_acf(self, chain: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(chain)

        # Handle edge cases
        if n == 0:
            return np.array([1.0])  # Return single value for empty chain
        if n == 1:
            return np.array([1.0])  # Single value has perfect autocorrelation at lag 0

        chain_centered = chain - np.mean(chain)
        c0 = np.dot(chain_centered, chain_centered) / n

        # Determine actual max lag based on chain length
        actual_max_lag = min(max_lag, n - 1)

        acf = np.zeros(actual_max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, actual_max_lag + 1):
            c_lag = np.dot(chain_centered[:-lag], chain_centered[lag:]) / n
            acf[lag] = c_lag / c0 if c0 > 0 else 0

        return acf

    def _generate_convergence_summary(self, chains: np.ndarray, diagnostics: Dict[str, Any]) -> str:
        """Generate text summary of convergence status."""
        n_chains, n_iterations, n_params = chains.shape

        summary_lines = [
            "CONVERGENCE SUMMARY",
            "=" * 40,
            f"Chains: {n_chains}",
            f"Iterations: {n_iterations}",
            f"Parameters: {n_params}",
            "",
        ]

        # Check R-hat values
        r_hat_ok = True
        for i in range(n_params):
            if f"r_hat_{i}" in diagnostics and len(diagnostics[f"r_hat_{i}"]) > 0:
                latest_rhat = diagnostics[f"r_hat_{i}"][-1]
                status = "✓" if latest_rhat < 1.1 else "✗"
                summary_lines.append(f"Param {i+1} R-hat: {latest_rhat:.3f} {status}")
                if latest_rhat >= 1.1:
                    r_hat_ok = False

        summary_lines.append("")

        # Check ESS values
        ess_ok = True
        for i in range(n_params):
            if f"ess_{i}" in diagnostics and len(diagnostics[f"ess_{i}"]) > 0:
                latest_ess = diagnostics[f"ess_{i}"][-1]
                status = "✓" if latest_ess >= 1000 else "✗"
                summary_lines.append(f"Param {i+1} ESS: {latest_ess:.0f} {status}")
                if latest_ess < 1000:
                    ess_ok = False

        summary_lines.append("")

        # Overall status
        if r_hat_ok and ess_ok:
            summary_lines.append("OVERALL: CONVERGED ✓")
        else:
            summary_lines.append("OVERALL: NOT CONVERGED ✗")

        return "\n".join(summary_lines)

"""Technical appendix visualization functions.

This module provides detailed technical visualization functions for
convergence diagnostics, Pareto frontier analysis, loss distribution validation,
and Monte Carlo convergence analysis.
"""
# pylint: disable=too-many-lines

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .core import COLOR_SEQUENCE, WSJ_COLORS, set_wsj_style


def plot_convergence_diagnostics(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    convergence_stats: Dict[str, Any],
    title: str = "Convergence Diagnostics",
    figsize: Tuple[int, int] = (12, 8),
    r_hat_threshold: float = 1.1,
    show_threshold: bool = False,
) -> Figure:
    """Create comprehensive convergence diagnostics plot.

    Visualizes convergence metrics including R-hat statistics, effective sample size,
    autocorrelation, and Monte Carlo standard errors.

    Args:
        convergence_stats: Dictionary with convergence statistics
        title: Plot title
        figsize: Figure size (width, height)
        r_hat_threshold: R-hat convergence threshold
        show_threshold: Whether to show threshold lines

    Returns:
        Matplotlib figure with convergence diagnostics

    Examples:
        >>> stats = {
        ...     "r_hat_history": [1.5, 1.3, 1.15, 1.05],
        ...     "iterations": [100, 200, 300, 400],
        ...     "ess_history": [500, 800, 1200, 1500]
        ... }
        >>> fig = plot_convergence_diagnostics(stats)
    """
    set_wsj_style()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # R-hat over iterations
    ax = axes[0, 0]
    if "r_hat_history" in convergence_stats:
        iterations = convergence_stats["iterations"]
        r_hat = convergence_stats["r_hat_history"]

        ax.plot(iterations, r_hat, color=WSJ_COLORS["blue"], linewidth=2)
        ax.axhline(
            y=1.1, color=WSJ_COLORS["red"], linestyle="--", linewidth=1.5, label="Threshold (1.1)"
        )
        ax.axhline(
            y=1.05, color=WSJ_COLORS["orange"], linestyle="--", linewidth=1.5, label="Target (1.05)"
        )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("R-hat Statistic")
        ax.set_title("Gelman-Rubin Convergence")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # ESS over iterations
    ax = axes[0, 1]
    if "ess_history" in convergence_stats:
        ess = convergence_stats["ess_history"]

        ax.plot(iterations, ess, color=WSJ_COLORS["green"], linewidth=2)
        ax.axhline(
            y=1000, color=WSJ_COLORS["red"], linestyle="--", linewidth=1.5, label="Minimum (1000)"
        )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Effective Sample Size")
        ax.set_title("ESS Evolution")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    # Autocorrelation function
    ax = axes[1, 0]
    if "autocorrelation" in convergence_stats:
        lags = convergence_stats["lags"]
        acf = convergence_stats["autocorrelation"]

        ax.bar(lags, acf, color=WSJ_COLORS["purple"], alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation Function")
        ax.grid(True, alpha=0.3)

    # MCSE by metric
    ax = axes[1, 1]
    if "mcse_by_metric" in convergence_stats:
        metrics = list(convergence_stats["mcse_by_metric"].keys())
        mcse_values = list(convergence_stats["mcse_by_metric"].values())

        bars = ax.bar(range(len(metrics)), mcse_values, color=COLOR_SEQUENCE[: len(metrics)])
        ax.set_xlabel("Metric")
        ax.set_ylabel("Monte Carlo Standard Error")
        ax.set_title("MCSE by Metric")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for mcse_bar, val in zip(bars, mcse_values):
            height = mcse_bar.get_height()
            ax.text(
                mcse_bar.get_x() + mcse_bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_pareto_frontier_2d(  # pylint: disable=too-many-locals
    frontier_points: List[Any],
    x_objective: str,
    y_objective: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: str = "Pareto Frontier",
    highlight_knees: bool = True,
    show_trade_offs: bool = False,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot 2D Pareto frontier with WSJ styling.

    Visualizes the trade-off between two objectives with optional knee point
    highlighting and dominated region shading.

    Args:
        frontier_points: List of ParetoPoint objects
        x_objective: Name of objective for x-axis
        y_objective: Name of objective for y-axis
        x_label: Optional custom label for x-axis
        y_label: Optional custom label for y-axis
        title: Plot title
        highlight_knees: Whether to highlight knee points
        show_trade_offs: Whether to show trade-off annotations
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with 2D Pareto frontier

    Examples:
        >>> points = [ParetoPoint(objectives={"cost": 100, "quality": 0.8})]
        >>> fig = plot_pareto_frontier_2d(points, "cost", "quality")
    """
    set_wsj_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    x_values = [p.objectives[x_objective] for p in frontier_points]
    y_values = [p.objectives[y_objective] for p in frontier_points]

    # Sort points for line connection
    sorted_indices = np.argsort(x_values)
    x_sorted = [x_values[i] for i in sorted_indices]
    y_sorted = [y_values[i] for i in sorted_indices]

    # Plot frontier line
    ax.plot(
        x_sorted,
        y_sorted,
        color=WSJ_COLORS["blue"],
        linewidth=2,
        alpha=0.7,
        label="Pareto Frontier",
    )

    # Plot frontier points
    ax.scatter(
        x_values,
        y_values,
        color=WSJ_COLORS["blue"],
        s=50,
        zorder=5,
        alpha=0.8,
    )

    # Highlight knee points if requested
    if highlight_knees:
        # Find knee points (those with highest crowding distance)
        knee_points = sorted(frontier_points, key=lambda p: p.crowding_distance, reverse=True)[:3]
        knee_x = [p.objectives[x_objective] for p in knee_points]
        knee_y = [p.objectives[y_objective] for p in knee_points]

        ax.scatter(
            knee_x,
            knee_y,
            color=WSJ_COLORS["red"],
            s=100,
            marker="D",
            zorder=6,
            label="Knee Points",
            edgecolors="black",
            linewidths=1,
        )

    # Show trade-offs if requested
    if show_trade_offs and len(frontier_points) > 1:
        for i in range(len(x_sorted) - 1):
            mid_x = (x_sorted[i] + x_sorted[i + 1]) / 2
            mid_y = (y_sorted[i] + y_sorted[i + 1]) / 2

            # Calculate trade-off ratio
            dx = x_sorted[i + 1] - x_sorted[i]
            dy = y_sorted[i + 1] - y_sorted[i]

            if abs(dx) > 1e-10:
                trade_off = dy / dx
                ax.annotate(
                    f"Trade-off: {trade_off:.2f}",
                    xy=(mid_x, mid_y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

    # Shade dominated region
    _x_min, x_max = ax.get_xlim()
    _y_min, y_max = ax.get_ylim()

    # Create polygon for dominated region (assumes minimization for both)
    dominated_x = [x_max] + x_sorted + [x_sorted[-1]]
    dominated_y = [y_sorted[0]] + y_sorted + [y_max]

    ax.fill(
        dominated_x,
        dominated_y,
        color=WSJ_COLORS["light_gray"],
        alpha=0.3,
        label="Dominated Region",
    )

    # Labels and styling
    ax.set_xlabel(x_label or x_objective, fontsize=12)
    ax.set_ylabel(y_label or y_objective, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pareto_frontier_3d(  # pylint: disable=too-many-locals
    frontier_points: List[Any],
    x_objective: str,
    y_objective: str,
    z_objective: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    title: str = "3D Pareto Frontier",
    figsize: Tuple[float, float] = (12, 8),
) -> Figure:
    """Plot 3D Pareto frontier surface.

    Creates a 3D visualization of the Pareto frontier with optional surface
    interpolation when sufficient points are available.

    Args:
        frontier_points: List of ParetoPoint objects
        x_objective: Name of objective for x-axis
        y_objective: Name of objective for y-axis
        z_objective: Name of objective for z-axis
        x_label: Optional custom label for x-axis
        y_label: Optional custom label for y-axis
        z_label: Optional custom label for z-axis
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with 3D Pareto frontier
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    set_wsj_style()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Extract data
    x_values = np.array([p.objectives[x_objective] for p in frontier_points])
    y_values = np.array([p.objectives[y_objective] for p in frontier_points])
    z_values = np.array([p.objectives[z_objective] for p in frontier_points])

    # Create scatter plot
    scatter = ax.scatter(
        x_values,
        y_values,
        z_values,
        c=z_values,
        cmap="viridis",
        s=50,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # Try to create surface if we have enough points
    if len(frontier_points) > 10:
        try:
            from scipy.interpolate import griddata
            from scipy.spatial import QhullError  # pylint: disable=no-name-in-module

            # Create grid
            xi = np.linspace(x_values.min(), x_values.max(), 30)
            yi = np.linspace(y_values.min(), y_values.max(), 30)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate z values
            zi = griddata(
                (x_values, y_values),
                z_values,
                (xi, yi),
                method="linear",
            )

            # Plot surface
            ax.plot_surface(
                xi,
                yi,
                zi,
                alpha=0.3,
                cmap="viridis",
                edgecolor="none",
            )
        except (ValueError, TypeError, QhullError):
            # If interpolation fails (e.g., coplanar points), just show points
            pass

    # Add colorbar
    fig.colorbar(scatter, ax=ax, pad=0.1, label=z_label or z_objective)

    # Labels and styling
    ax.set_xlabel(x_label or x_objective, fontsize=11)
    ax.set_ylabel(y_label or y_objective, fontsize=11)
    ax.set_zlabel(z_label or z_objective, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


def create_interactive_pareto_frontier(
    frontier_points: List[Any],
    objectives: List[str],
    title: str = "Interactive Pareto Frontier",
    height: int = 600,
    show_dominated: bool = True,
) -> go.Figure:
    """Create interactive Plotly Pareto frontier visualization.

    Creates an interactive visualization that automatically adapts to the number
    of objectives (2D scatter, 3D scatter, or parallel coordinates).

    Args:
        frontier_points: List of ParetoPoint objects
        objectives: List of objective names to display
        title: Plot title
        height: Plot height in pixels
        show_dominated: Whether to show dominated region (2D only)

    Returns:
        Plotly figure with interactive Pareto frontier
    """
    # Handle 2D or 3D based on number of objectives
    if len(objectives) == 2:
        return _create_interactive_pareto_2d(
            frontier_points, objectives, title, height, show_dominated
        )
    if len(objectives) == 3:
        return _create_interactive_pareto_3d(frontier_points, objectives, title, height)
    # For more than 3 objectives, create parallel coordinates
    return _create_pareto_parallel_coordinates(frontier_points, objectives, title, height)


def _create_interactive_pareto_2d(  # pylint: disable=too-many-locals
    frontier_points: List[Any],
    objectives: List[str],
    title: str,
    height: int,
    show_dominated: bool,
) -> go.Figure:
    """Create 2D interactive Pareto frontier."""
    x_obj, y_obj = objectives[0], objectives[1]

    # Extract data
    x_values = [p.objectives[x_obj] for p in frontier_points]
    y_values = [p.objectives[y_obj] for p in frontier_points]

    # Sort for line connection
    sorted_indices = np.argsort(x_values)
    x_sorted = [x_values[i] for i in sorted_indices]
    y_sorted = [y_values[i] for i in sorted_indices]

    fig = go.Figure()

    # Add frontier line
    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_sorted,
            mode="lines",
            name="Pareto Frontier",
            line={"color": WSJ_COLORS["blue"], "width": 2},
            hovertemplate=f"{x_obj}: %{{x:.3f}}<br>{y_obj}: %{{y:.3f}}<extra></extra>",
        )
    )

    # Add frontier points
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name="Solutions",
            marker={
                "size": 10,
                "color": [p.crowding_distance for p in frontier_points],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Crowding<br>Distance"},
            },
            text=[f"Point {i}" for i in range(len(frontier_points))],
            hovertemplate=(
                f"{x_obj}: %{{x:.3f}}<br>"
                f"{y_obj}: %{{y:.3f}}<br>"
                "Crowding: %{marker.color:.3f}<br>"
                "%{text}<extra></extra>"
            ),
        )
    )

    # Add dominated region if requested
    if show_dominated:
        x_max = max(x_values) * 1.1
        y_max = max(y_values) * 1.1

        dominated_x = x_sorted + [x_max, x_max, x_sorted[0]]
        dominated_y = y_sorted + [y_sorted[-1], y_max, y_max]

        fig.add_trace(
            go.Scatter(
                x=dominated_x,
                y=dominated_y,
                fill="toself",
                fillcolor="rgba(200, 200, 200, 0.2)",
                line={"width": 0},
                showlegend=True,
                name="Dominated Region",
                hoverinfo="skip",
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_obj,
        yaxis_title=y_obj,
        height=height,
        hovermode="closest",
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    return fig


def _create_interactive_pareto_3d(
    frontier_points: List[Any],
    objectives: List[str],
    title: str,
    height: int,
) -> go.Figure:
    """Create 3D interactive Pareto frontier."""
    x_obj, y_obj, z_obj = objectives[0], objectives[1], objectives[2]

    # Extract data
    x_values = [p.objectives[x_obj] for p in frontier_points]
    y_values = [p.objectives[y_obj] for p in frontier_points]
    z_values = [p.objectives[z_obj] for p in frontier_points]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="markers",
                marker={
                    "size": 8,
                    "color": z_values,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": z_obj},
                },
                text=[f"Point {i}" for i in range(len(frontier_points))],
                hovertemplate=(
                    f"{x_obj}: %{{x:.3f}}<br>"
                    f"{y_obj}: %{{y:.3f}}<br>"
                    f"{z_obj}: %{{z:.3f}}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": x_obj,
            "yaxis_title": y_obj,
            "zaxis_title": z_obj,
        },
        height=height,
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    return fig


def _create_pareto_parallel_coordinates(
    frontier_points: List[Any],
    objectives: List[str],
    title: str,
    height: int,
) -> go.Figure:
    """Create parallel coordinates plot for many objectives."""
    # Prepare data for parallel coordinates
    data = []
    for i, point in enumerate(frontier_points):
        row = {"index": i}
        for obj in objectives:
            row[obj] = point.objectives[obj]
        data.append(row)

    df = pd.DataFrame(data)

    # Create dimensions
    dimensions = []
    for obj in objectives:
        dimensions.append(
            {
                "label": obj,
                "values": df[obj],
                "range": [df[obj].min(), df[obj].max()],
            }
        )

    # Add crowding distance as color
    colors = [p.crowding_distance for p in frontier_points]

    fig = go.Figure(
        data=go.Parcoords(
            line={
                "color": colors,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Crowding<br>Distance"},
            },
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    return fig


def plot_trace_plots(
    chains: np.ndarray,
    parameter_names: Optional[List[str]] = None,
    burn_in: Optional[int] = None,
    title: str = "Trace Plots",
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """Create trace plots for MCMC chains with burn-in indicators.

    Args:
        chains: Array of shape (n_chains, n_iterations, n_parameters)
        parameter_names: Names of parameters (optional)
        burn_in: Number of burn-in iterations to mark
        title: Overall plot title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with trace plots

    Examples:
        >>> chains = np.random.randn(4, 1000, 3)  # 4 chains, 1000 iterations, 3 parameters
        >>> fig = plot_trace_plots(chains, ["param1", "param2", "param3"], burn_in=200)
    """
    set_wsj_style()

    # Handle different input shapes
    if chains.ndim == 2:
        # Single chain, multiple parameters
        chains = chains.reshape(1, chains.shape[0], chains.shape[1])
    elif chains.ndim == 1:
        # Single chain, single parameter
        chains = chains.reshape(1, -1, 1)

    n_chains, n_iterations, n_params = chains.shape

    # Default parameter names
    if parameter_names is None:
        parameter_names = [f"Parameter {i+1}" for i in range(n_params)]

    # Create subplots
    n_cols = 2 if n_params > 1 else 1
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    iterations = np.arange(n_iterations)

    for param_idx in range(n_params):
        ax = axes[param_idx]

        # Plot each chain
        for chain_idx in range(n_chains):
            chain_data = chains[chain_idx, :, param_idx]
            ax.plot(
                iterations, chain_data, alpha=0.7, linewidth=0.8, label=f"Chain {chain_idx + 1}"
            )

        # Mark burn-in period
        if burn_in is not None:
            ax.axvline(
                x=burn_in,
                color=WSJ_COLORS["red"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Burn-in",
            )
            # Shade burn-in region
            ax.axvspan(0, burn_in, alpha=0.1, color=WSJ_COLORS["light_gray"])

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_title(parameter_names[param_idx])
        ax.grid(True, alpha=0.3)

        # Only show legend on first plot
        if param_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_loss_distribution_validation(  # pylint: disable=too-many-statements
    attritional_losses: np.ndarray,
    large_losses: np.ndarray,
    attritional_dist: Optional[Dict[str, Any]] = None,
    large_dist: Optional[Dict[str, Any]] = None,
    title: str = "Loss Distribution Validation",
    figsize: Tuple[int, int] = (12, 10),
) -> Figure:
    """Create comprehensive loss distribution validation plots (Figure B1).

    Generates Q-Q plots and CDF comparisons for attritional and large losses,
    with K-S test statistics and goodness-of-fit metrics.

    Args:
        attritional_losses: Empirical attritional loss data
        large_losses: Empirical large loss data
        attritional_dist: Theoretical distribution parameters for attritional losses
        large_dist: Theoretical distribution parameters for large losses
        title: Overall plot title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with validation plots

    Examples:
        >>> attritional = np.random.lognormal(10, 1, 1000)
        >>> large = np.random.lognormal(15, 2, 100)
        >>> fig = plot_loss_distribution_validation(attritional, large)
    """
    set_wsj_style()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Helper function for Q-Q plots
    def create_qq_plot(ax, data, dist_params, loss_type):
        """Create Q-Q plot with theoretical overlay."""
        # Sort the data
        sorted_data = np.sort(data)
        n = len(sorted_data)

        # Calculate theoretical quantiles
        probs = (np.arange(n) + 0.5) / n

        # Default to lognormal if no distribution specified
        if dist_params is None:
            # Fit lognormal
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            theoretical_quantiles = stats.lognorm.ppf(probs, shape, loc, scale)
            dist_name = "Lognormal (fitted)"
        else:
            dist_name = dist_params.get("name", "Theoretical")
            if "lognorm" in dist_name.lower():
                shape = dist_params.get("shape", 1)
                loc = dist_params.get("loc", 0)
                scale = dist_params.get("scale", np.exp(np.mean(np.log(data))))
                theoretical_quantiles = stats.lognorm.ppf(probs, shape, loc, scale)
            else:
                # Default to normal
                mean = dist_params.get("mean", np.mean(data))
                std = dist_params.get("std", np.std(data))
                theoretical_quantiles = stats.norm.ppf(probs, mean, std)

        # Plot Q-Q plot
        ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=20, color=WSJ_COLORS["blue"])

        # Add reference line
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.7, label="y=x"
        )

        # Perform K-S test
        if dist_params is None:
            ks_stat, p_value = stats.kstest(data, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
        else:
            if "lognorm" in dist_name.lower():
                ks_stat, p_value = stats.kstest(
                    data, lambda x: stats.lognorm.cdf(x, shape, loc, scale)
                )
            else:
                ks_stat, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))

        # Add K-S test result
        ax.text(
            0.05,
            0.95,
            f"K-S stat: {ks_stat:.4f}\np-value: {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        ax.set_xlabel(f"Theoretical Quantiles ({dist_name})")
        ax.set_ylabel("Empirical Quantiles")
        ax.set_title(f"Q-Q Plot: {loss_type}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    # Helper function for CDF comparisons
    def create_cdf_comparison(ax, data, dist_params, loss_type):
        """Create empirical vs theoretical CDF comparison."""
        # Sort the data
        sorted_data = np.sort(data)
        n = len(sorted_data)

        # Empirical CDF
        empirical_cdf = np.arange(1, n + 1) / n

        # Plot empirical CDF
        ax.plot(
            sorted_data,
            empirical_cdf,
            label="Empirical",
            color=WSJ_COLORS["blue"],
            linewidth=2,
            alpha=0.7,
        )

        # Theoretical CDF
        if dist_params is None:
            # Fit lognormal
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            theoretical_cdf = stats.lognorm.cdf(sorted_data, shape, loc, scale)
            dist_name = "Lognormal (fitted)"
        else:
            dist_name = dist_params.get("name", "Theoretical")
            if "lognorm" in dist_name.lower():
                shape = dist_params.get("shape", 1)
                loc = dist_params.get("loc", 0)
                scale = dist_params.get("scale", np.exp(np.mean(np.log(data))))
                theoretical_cdf = stats.lognorm.cdf(sorted_data, shape, loc, scale)
            else:
                mean = dist_params.get("mean", np.mean(data))
                std = dist_params.get("std", np.std(data))
                theoretical_cdf = stats.norm.cdf(sorted_data, mean, std)

        ax.plot(
            sorted_data,
            theoretical_cdf,
            label=dist_name,
            color=WSJ_COLORS["red"],
            linewidth=2,
            linestyle="--",
            alpha=0.7,
        )

        # Calculate goodness-of-fit metrics
        mse = np.mean((empirical_cdf - theoretical_cdf) ** 2)
        max_deviation = np.max(np.abs(empirical_cdf - theoretical_cdf))

        # Add metrics
        ax.text(
            0.95,
            0.05,
            f"MSE: {mse:.6f}\nMax Dev: {max_deviation:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        ax.set_xlabel("Loss Amount")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF Comparison: {loss_type}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    # Create plots
    create_qq_plot(axes[0, 0], attritional_losses, attritional_dist, "Attritional Losses")
    create_qq_plot(axes[0, 1], large_losses, large_dist, "Large Losses")
    create_cdf_comparison(axes[1, 0], attritional_losses, attritional_dist, "Attritional Losses")
    create_cdf_comparison(axes[1, 1], large_losses, large_dist, "Large Losses")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_monte_carlo_convergence(  # pylint: disable=too-many-locals
    metrics_history: Dict[str, List[float]],
    iterations: Optional[np.ndarray] = None,
    convergence_thresholds: Optional[Dict[str, float]] = None,
    title: str = "Monte Carlo Convergence Analysis",
    figsize: Tuple[int, int] = (14, 10),
    log_scale: bool = True,
) -> Figure:
    """Create Monte Carlo convergence analysis plots (Figure C3).

    Visualizes convergence of key metrics (ROE, ruin probability, etc.) as a
    function of Monte Carlo iterations, with running statistics and thresholds.

    Args:
        metrics_history: Dictionary of metric names to lists of values over iterations
        iterations: Array of iteration counts (optional, will be inferred)
        convergence_thresholds: Dictionary of metric names to convergence thresholds
        title: Overall plot title
        figsize: Figure size (width, height)
        log_scale: Whether to use log scale for x-axis

    Returns:
        Matplotlib figure with convergence analysis

    Examples:
        >>> history = {
        ...     "ROE": [0.08, 0.082, 0.081, 0.0805],
        ...     "Ruin Probability": [0.05, 0.048, 0.049, 0.0495]
        ... }
        >>> fig = plot_monte_carlo_convergence(history)
    """
    set_wsj_style()

    # Determine number of metrics and create subplots
    n_metrics = len(metrics_history)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Create iterations array if not provided
    if iterations is None:
        max_length = max(len(values) for values in metrics_history.values())
        iterations = np.arange(1, max_length + 1)

    # Default convergence thresholds
    if convergence_thresholds is None:
        convergence_thresholds = {}

    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[idx]

        # Ensure we have the right number of iterations
        n_values = len(values)
        iter_subset = iterations[:n_values]

        # Calculate running mean
        running_mean = np.array([np.mean(values[: i + 1]) for i in range(n_values)])

        # Calculate running standard error
        running_se = np.array(
            [np.std(values[: i + 1]) / np.sqrt(i + 1) if i > 0 else 0 for i in range(n_values)]
        )

        # Plot the metric values
        ax.plot(
            iter_subset,
            values,
            alpha=0.3,
            color=WSJ_COLORS["light_gray"],
            linewidth=0.5,
            label="Raw values",
        )

        # Plot running mean
        ax.plot(
            iter_subset, running_mean, color=WSJ_COLORS["blue"], linewidth=2, label="Running mean"
        )

        # Plot confidence bands (±2 SE)
        ax.fill_between(
            iter_subset,
            running_mean - 2 * running_se,
            running_mean + 2 * running_se,
            alpha=0.2,
            color=WSJ_COLORS["blue"],
            label="95% CI",
        )

        # Add convergence threshold if provided
        if metric_name in convergence_thresholds:
            threshold = convergence_thresholds[metric_name]
            ax.axhline(
                y=threshold,
                color=WSJ_COLORS["red"],
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold ({threshold:.3f})",
            )

        # Calculate and display convergence metrics
        if n_values > 100:
            # Calculate relative change in running mean
            window = min(100, n_values // 10)
            recent_mean = np.mean(values[-window:])
            earlier_mean = (
                np.mean(values[-2 * window : -window])
                if n_values > 2 * window
                else np.mean(values[:window])
            )
            relative_change = float(
                abs(recent_mean - earlier_mean) / abs(earlier_mean) if earlier_mean != 0 else 0
            )

            # Add convergence status
            converged = relative_change < 0.01  # 1% relative change threshold
            status_text = "Converged" if converged else "Not converged"
            status_color = WSJ_COLORS["green"] if converged else WSJ_COLORS["orange"]

            ax.text(
                0.95,
                0.95,
                f"{status_text}\nRel. change: {relative_change:.4f}",
                transform=ax.transAxes,
                fontsize=9,
                horizontalalignment="right",
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": status_color, "alpha": 0.3},
            )

        # Formatting
        if log_scale:
            ax.set_xscale("log")

        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Convergence")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which="both" if log_scale else "major")

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_enhanced_convergence_diagnostics(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    chains: np.ndarray,
    parameter_names: Optional[List[str]] = None,
    burn_in: Optional[int] = None,
    title: str = "Enhanced Convergence Diagnostics",
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """Create comprehensive convergence diagnostics with trace plots and statistics.

    Enhanced version of plot_convergence_diagnostics that includes trace plots,
    R-hat evolution, ESS calculations, and autocorrelation analysis.

    Args:
        chains: Array of shape (n_chains, n_iterations, n_parameters)
        parameter_names: Names of parameters
        burn_in: Number of burn-in iterations
        title: Overall plot title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with enhanced diagnostics

    Examples:
        >>> chains = np.random.randn(4, 1000, 2)
        >>> fig = plot_enhanced_convergence_diagnostics(
        ...     chains,
        ...     parameter_names=["mu", "sigma"],
        ...     burn_in=200
        ... )
    """
    set_wsj_style()

    # Handle different input shapes
    if chains.ndim == 2:
        chains = chains.reshape(1, chains.shape[0], chains.shape[1])
    elif chains.ndim == 1:
        chains = chains.reshape(1, -1, 1)

    n_chains, n_iterations, n_params = chains.shape

    # Default parameter names
    if parameter_names is None:
        parameter_names = [f"Parameter {i+1}" for i in range(n_params)]

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Trace plots (top row, spanning both columns)
    ax_trace = fig.add_subplot(gs[0, :])

    # Plot traces for first parameter (or all if single parameter)
    param_idx = 0
    iterations_array = np.arange(n_iterations)

    for chain_idx in range(n_chains):
        chain_data = chains[chain_idx, :, param_idx]
        ax_trace.plot(
            iterations_array, chain_data, alpha=0.7, linewidth=0.8, label=f"Chain {chain_idx + 1}"
        )

    if burn_in is not None:
        ax_trace.axvline(
            x=burn_in, color=WSJ_COLORS["red"], linestyle="--", linewidth=1.5, label="Burn-in"
        )
        ax_trace.axvspan(0, burn_in, alpha=0.1, color=WSJ_COLORS["light_gray"])

    ax_trace.set_xlabel("Iteration")
    ax_trace.set_ylabel("Value")
    ax_trace.set_title(f"Trace Plot: {parameter_names[param_idx]}")
    ax_trace.legend(loc="upper right", fontsize=8)
    ax_trace.grid(True, alpha=0.3)

    # Calculate diagnostics
    from ..convergence import ConvergenceDiagnostics

    diag = ConvergenceDiagnostics()

    # R-hat evolution (middle left)
    ax_rhat = fig.add_subplot(gs[1, 0])

    if n_chains > 1:
        r_hat_history = []
        check_points_list = []
        check_points = np.linspace(100, n_iterations, min(50, n_iterations // 20), dtype=int)

        for check_point in check_points:
            if check_point > (burn_in if burn_in else 0):
                subset_chains = chains[:, burn_in if burn_in else 0 : check_point, :]
                r_hat = diag.calculate_r_hat(subset_chains)
                r_hat_history.append(r_hat)
                check_points_list.append(check_point)

        if r_hat_history:
            ax_rhat.plot(check_points_list, r_hat_history, color=WSJ_COLORS["blue"], linewidth=2)
        ax_rhat.axhline(y=1.1, color=WSJ_COLORS["red"], linestyle="--", label="Threshold")
        ax_rhat.axhline(y=1.05, color=WSJ_COLORS["orange"], linestyle="--", label="Target")
        ax_rhat.set_xlabel("Iteration")
        ax_rhat.set_ylabel("R-hat")
        ax_rhat.set_title("R-hat Evolution")
        ax_rhat.legend(loc="upper right", fontsize=8)
        ax_rhat.grid(True, alpha=0.3)

    # ESS calculation (middle right)
    ax_ess = fig.add_subplot(gs[1, 1])

    ess_values = []
    for param_idx in range(n_params):
        # Pool chains for ESS calculation
        pooled_chain = chains[:, burn_in if burn_in else 0 :, param_idx].flatten()
        ess = diag.calculate_ess(pooled_chain)
        ess_values.append(ess)

    bars = ax_ess.bar(range(n_params), ess_values, color=COLOR_SEQUENCE[:n_params], alpha=0.7)

    ax_ess.axhline(y=1000, color=WSJ_COLORS["red"], linestyle="--", label="Min ESS")
    ax_ess.set_xlabel("Parameter")
    ax_ess.set_ylabel("Effective Sample Size")
    ax_ess.set_title("ESS by Parameter")
    ax_ess.set_xticks(range(n_params))
    ax_ess.set_xticklabels(parameter_names, rotation=45 if n_params > 3 else 0, ha="right")
    ax_ess.legend(loc="upper right", fontsize=8)
    ax_ess.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for rect, val in zip(bars, ess_values):
        height = rect.get_height()
        ax_ess.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Autocorrelation (bottom left)
    ax_acf = fig.add_subplot(gs[2, 0])

    # Calculate autocorrelation for first chain, first parameter
    chain_data = chains[0, burn_in if burn_in else 0 :, 0]
    max_lag = min(50, len(chain_data) // 4)

    acf_values = []
    for lag in range(max_lag):
        if lag == 0:
            acf_values.append(1.0)
        else:
            acf = np.corrcoef(chain_data[:-lag], chain_data[lag:])[0, 1]
            acf_values.append(acf)

    lags = np.arange(max_lag)
    ax_acf.bar(lags, acf_values, color=WSJ_COLORS["purple"], alpha=0.7)
    ax_acf.axhline(y=0, color="black", linewidth=0.5)
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_title(f"ACF: {parameter_names[0]}")
    ax_acf.grid(True, alpha=0.3)

    # MCSE summary (bottom right)
    ax_mcse = fig.add_subplot(gs[2, 1])

    mcse_values = []
    for param_idx in range(n_params):
        pooled_chain = chains[:, burn_in if burn_in else 0 :, param_idx].flatten()
        mcse = np.std(pooled_chain) / np.sqrt(diag.calculate_ess(pooled_chain))
        mcse_values.append(mcse)

    bars = ax_mcse.bar(range(n_params), mcse_values, color=COLOR_SEQUENCE[:n_params], alpha=0.7)

    ax_mcse.set_xlabel("Parameter")
    ax_mcse.set_ylabel("MCSE")
    ax_mcse.set_title("Monte Carlo Standard Error")
    ax_mcse.set_xticks(range(n_params))
    ax_mcse.set_xticklabels(parameter_names, rotation=45 if n_params > 3 else 0, ha="right")
    ax_mcse.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for rect, val in zip(bars, mcse_values):
        height = rect.get_height()
        ax_mcse.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle(title, fontsize=14, fontweight="bold")

    return fig


def plot_ergodic_divergence(  # pylint: disable=too-many-locals
    time_horizons: np.ndarray,
    time_averages: np.ndarray,
    ensemble_averages: np.ndarray,
    standard_errors: Optional[np.ndarray] = None,
    parameter_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "Ergodic vs Ensemble Average Divergence",
    figsize: Tuple[float, float] = (14, 8),
    add_formulas: bool = True,
) -> Figure:
    """Create ergodic vs ensemble divergence visualization (Figure C1).

    Demonstrates the fundamental difference between time-average (ergodic) and
    ensemble-average growth rates as time horizon increases, showing why insurance
    decisions must consider individual path dynamics rather than expected values.

    Args:
        time_horizons: Array of time horizons (e.g., 1 to 1000 years)
        time_averages: Time-average growth rates for each horizon
        ensemble_averages: Ensemble-average growth rates for each horizon
        standard_errors: Optional standard errors for confidence bands
        parameter_scenarios: Optional dict of scenario name to parameter values
        title: Plot title
        figsize: Figure size (width, height)
        add_formulas: Whether to add LaTeX formula annotations

    Returns:
        Matplotlib figure showing divergence visualization

    Examples:
        >>> horizons = np.logspace(0, 3, 50)  # 1 to 1000 years
        >>> time_avg = np.array([0.05 * (1 - 0.1 * np.log10(t)) for t in horizons])
        >>> ensemble_avg = np.array([0.08] * len(horizons))
        >>> fig = plot_ergodic_divergence(horizons, time_avg, ensemble_avg)
    """
    set_wsj_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Main divergence plot (left)
    ax1.semilogx(
        time_horizons,
        time_averages,
        color=WSJ_COLORS["blue"],
        linewidth=2.5,
        label="Time Average (Ergodic)",
    )
    ax1.semilogx(
        time_horizons,
        ensemble_averages,
        color=WSJ_COLORS["red"],
        linewidth=2.5,
        linestyle="--",
        label="Ensemble Average",
    )

    # Add confidence bands if provided
    if standard_errors is not None:
        ax1.fill_between(
            time_horizons,
            time_averages - 2 * standard_errors,
            time_averages + 2 * standard_errors,
            alpha=0.2,
            color=WSJ_COLORS["blue"],
            label="95% CI (Time Average)",
        )

    # Add parameter sensitivity scenarios if provided
    if parameter_scenarios:
        colors = ["purple", "orange", "green"]
        for idx, (scenario_name, scenario_data) in enumerate(parameter_scenarios.items()):
            color = WSJ_COLORS.get(colors[idx % len(colors)], colors[idx % len(colors)])
            ax1.semilogx(
                scenario_data["horizons"],
                scenario_data["time_avg"],
                color=color,
                linewidth=1.5,
                alpha=0.7,
                linestyle=":",
                label=f"Scenario: {scenario_name}",
            )

    # Add divergence region shading
    divergence_start = None
    for i, _ in enumerate(time_horizons):
        if abs(time_averages[i] - ensemble_averages[i]) > 0.005:  # 0.5% divergence threshold
            divergence_start = time_horizons[i]
            break

    if divergence_start:
        ax1.axvspan(
            divergence_start,
            time_horizons[-1],
            alpha=0.15,  # Increased alpha for better visibility
            color=WSJ_COLORS["red"],  # Changed to red for better visibility
            label="Divergence Region",
        )
        ax1.axvline(x=divergence_start, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
        ax1.text(
            divergence_start,
            ax1.get_ylim()[0] + 0.01,
            f"Divergence at {divergence_start:.0f} years",
            rotation=90,
            va="bottom",
            fontsize=9,
            alpha=0.7,
        )

    ax1.set_xlabel("Time Horizon (years)", fontsize=12)
    ax1.set_ylabel("Growth Rate", fontsize=12)
    ax1.set_title("Time vs Ensemble Average Growth", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Mathematical formulas (right panel)
    ax2.axis("off")
    ax2.set_title(
        "Mathematical Framework", fontsize=13, fontweight="bold", loc="center", y=1.0
    )  # Aligned with left title

    if add_formulas:
        formula_text = r"""
$\mathbf{Time\ Average\ (Ergodic):}$
$g_{time} = \lim_{T \to \infty} \frac{1}{T} \ln\left(\frac{W(T)}{W(0)}\right)$

$\mathbf{Ensemble\ Average:}$
$g_{ensemble} = \mathbb{E}[\ln(W(T)/W(0))]$

$\mathbf{For\ Multiplicative\ Processes:}$
$W(t+1) = W(t) \cdot (1 + r + \sigma \xi_t)$

$\mathbf{Key\ Result:}$
$g_{time} = \mu - \frac{\sigma^2}{2}$ (volatility drag)
$g_{ensemble} = \mu$

$\mathbf{Insurance\ Impact:}$
Premium $p$ reduces both averages equally
But insurance caps losses at $L$:
$g_{time}^{ins} > g_{time}^{no\ ins}$ for high $\sigma$
"""
        # Center-align formulas horizontally and position below title
        ax2.text(
            0.5,  # Horizontal centering
            0.92,  # Move formulas up to follow the title position
            formula_text,
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="center",  # Center alignment
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.3},
        )

        # Add interpretation box with bold header and better spacing
        interpretation = r"""
$\mathbf{Key\ Insights:}$
• Time and ensemble averages diverge for multiplicative processes
• Volatility creates a "drag" on time-average growth
• Insurance reduces effective volatility, improving time-average growth
• This justifies higher premiums than expected loss alone
        """
        # Center-align Key Insights horizontally with more space below formulas
        ax2.text(
            0.5,  # Changed from 0.1 to 0.5 for horizontal centering
            0.15,  # Changed from 0.3 to 0.15 to create more space below formulas
            interpretation.strip(),  # Remove leading/trailing whitespace
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",  # Added horizontal center alignment
            bbox={"boxstyle": "round,pad=0.5", "facecolor": WSJ_COLORS["light_gray"], "alpha": 0.2},
        )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig


def plot_path_dependent_wealth(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    trajectories: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    ruin_threshold: float = 0.0,
    percentiles: Optional[List[int]] = None,
    highlight_ruined: bool = True,
    add_survivor_bias_inset: bool = True,
    title: str = "Path-Dependent Wealth Evolution",
    figsize: Tuple[float, float] = (14, 8),
    log_scale: bool = True,
) -> Figure:
    """Create path-dependent wealth evolution visualization (Figure C2).

    Shows multiple wealth trajectories over time with percentile bands, highlighting
    paths that hit ruin and demonstrating survivor bias effects. This visualization
    makes clear why ensemble averages mislead decision-making.

    Args:
        trajectories: Array of shape (n_paths, n_time_points) with wealth values
        time_points: Optional array of time points (defaults to years)
        ruin_threshold: Wealth level considered as ruin (default 0)
        percentiles: List of percentiles to show (default [5, 25, 50, 75, 95])
        highlight_ruined: Whether to highlight paths that hit ruin
        add_survivor_bias_inset: Whether to add survivor bias analysis inset
        title: Plot title
        figsize: Figure size (width, height)
        log_scale: Whether to use log scale for wealth axis

    Returns:
        Matplotlib figure showing path evolution

    Examples:
        >>> n_paths, n_years = 1000, 100
        >>> trajectories = np.random.lognormal(0, 0.2, (n_paths, n_years)).cumprod(axis=1)
        >>> fig = plot_path_dependent_wealth(trajectories)
    """
    set_wsj_style()

    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    # Generate time points if not provided
    if time_points is None:
        time_points = np.arange(trajectories.shape[1])

    # Calculate statistics
    n_paths, n_time = trajectories.shape

    # Identify ruined paths
    ruined_mask = np.any(trajectories <= ruin_threshold, axis=1)
    survived_mask = ~ruined_mask
    n_ruined = np.sum(ruined_mask)
    n_survived = np.sum(survived_mask)

    # Calculate percentiles for all paths and survivors only
    percentile_values = np.percentile(trajectories, percentiles, axis=0)
    _survivor_percentiles = (
        np.percentile(trajectories[survived_mask], percentiles, axis=0)
        if n_survived > 0
        else percentile_values
    )

    # Create figure with optional inset
    if add_survivor_bias_inset:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3
        )
        ax = fig.add_subplot(gs[0, :])
        ax_inset = fig.add_subplot(gs[1, 0])
        ax_stats = fig.add_subplot(gs[1, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot individual trajectories
    # Sample paths to plot (limit for performance)
    max_paths_to_plot = min(100, n_paths)
    sample_indices = np.random.choice(n_paths, max_paths_to_plot, replace=False)

    for idx in sample_indices:
        trajectory = trajectories[idx]
        is_ruined = ruined_mask[idx]

        if is_ruined and highlight_ruined:
            # Find ruin point
            ruin_point = np.where(trajectory <= ruin_threshold)[0]
            if len(ruin_point) > 0:
                ruin_idx = ruin_point[0]
                # Plot until ruin in red
                ax.plot(
                    time_points[: ruin_idx + 1],
                    trajectory[: ruin_idx + 1],
                    alpha=0.3,
                    linewidth=0.5,
                    color=WSJ_COLORS["red"],
                )
            else:
                ax.plot(
                    time_points,
                    trajectory,
                    alpha=0.1,
                    linewidth=0.5,
                    color=WSJ_COLORS["light_gray"],
                )
        else:
            ax.plot(
                time_points, trajectory, alpha=0.1, linewidth=0.5, color=WSJ_COLORS["light_gray"]
            )

    # Plot percentile bands
    colors = ["green", "blue", "purple", "blue", "green"]
    alphas = [0.3, 0.4, 0.6, 0.4, 0.3]

    for i in range(len(percentiles) // 2):
        lower_idx = i
        upper_idx = -(i + 1)

        ax.fill_between(
            time_points,
            percentile_values[lower_idx],
            percentile_values[upper_idx],
            alpha=alphas[i],
            color=WSJ_COLORS[colors[i]],
            label=f"{percentiles[lower_idx]}-{percentiles[upper_idx]}%",
        )

    # Plot median
    median_idx = len(percentiles) // 2
    ax.plot(
        time_points,
        percentile_values[median_idx],
        color=WSJ_COLORS["black"],
        linewidth=2,
        label="Median",
    )

    # Formatting
    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Wealth (log scale)" if log_scale else "Wealth", fontsize=12)
    ax.set_title("Wealth Trajectories with Percentile Bands", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add ruin region shading
    if highlight_ruined and n_ruined > 0:
        ax.axhspan(
            0,
            ruin_threshold if ruin_threshold > 0 else ax.get_ylim()[0],
            alpha=0.1,
            color=WSJ_COLORS["red"],
            label="Ruin Region",
        )

    # Survivor bias inset
    if add_survivor_bias_inset:
        # Calculate survival rate over time
        survival_rate = np.zeros(n_time)
        for t in range(n_time):
            survival_rate[t] = np.mean(trajectories[:, t] > ruin_threshold) * 100

        ax_inset.plot(time_points, survival_rate, color=WSJ_COLORS["blue"], linewidth=2)
        ax_inset.fill_between(time_points, 0, survival_rate, alpha=0.3, color=WSJ_COLORS["blue"])
        ax_inset.set_xlabel("Time (years)", fontsize=10)
        ax_inset.set_ylabel("Survival Rate (%)", fontsize=10)
        ax_inset.set_title("Survivor Bias Effect", fontsize=11, fontweight="bold")
        ax_inset.grid(True, alpha=0.3)

        # Add annotation for final survival rate
        final_survival = survival_rate[-1]
        ax_inset.text(
            0.95,
            0.05,
            f"Final: {final_survival:.1f}%",
            transform=ax_inset.transAxes,
            horizontalalignment="right",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        # Statistics panel
        ax_stats.axis("off")

        stats_text = f"""
Path Statistics:
• Total paths: {n_paths:,}
• Survived: {n_survived:,} ({n_survived/n_paths*100:.1f}%)
• Ruined: {n_ruined:,} ({n_ruined/n_paths*100:.1f}%)

Final Wealth (survivors):
• Median: {np.median(trajectories[survived_mask, -1]) if n_survived > 0 else 0:.2f}
• Mean: {np.mean(trajectories[survived_mask, -1]) if n_survived > 0 else 0:.2f}

All Paths (inc. ruined):
• Median: {np.median(trajectories[:, -1]):.2f}
• Mean: {np.mean(trajectories[:, -1]):.2f}

Survivor Bias Factor:
{np.mean(trajectories[survived_mask, -1]) / np.mean(trajectories[:, -1]) if n_survived > 0 else 1:.2f}x
"""
        ax_stats.text(
            0.1,
            0.9,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": WSJ_COLORS["light_gray"], "alpha": 0.2},
        )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Use constrained_layout or manual adjustment instead of tight_layout for GridSpec
    if add_survivor_bias_inset:
        # GridSpec already handles spacing, no need for tight_layout
        pass
    else:
        # For simple subplot, tight_layout works fine
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig


def plot_correlation_structure(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    data: Dict[str, np.ndarray],
    correlation_type: str = "pearson",
    risk_types: Optional[List[str]] = None,
    title: str = "Risk Correlation Structure",
    figsize: Tuple[float, float] = (14, 10),
    show_copula: bool = True,
) -> Figure:
    """Create correlation structure visualization with copula analysis (Figure B2).

    Visualizes correlation matrices, copula density plots, and scatter plots with
    fitted copulas to show dependencies between different risk types.

    Args:
        data: Dictionary mapping risk type names to data arrays (n_samples, n_variables)
        correlation_type: Type of correlation ('pearson', 'spearman', 'kendall')
        risk_types: List of risk types to analyze (defaults to all in data)
        title: Plot title
        figsize: Figure size (width, height)
        show_copula: Whether to show copula density plots

    Returns:
        Matplotlib figure with correlation structure visualization

    Examples:
        >>> data = {
        ...     "operational": np.random.randn(1000, 3),
        ...     "financial": np.random.randn(1000, 3)
        ... }
        >>> fig = plot_correlation_structure(data)
    """
    from scipy.stats import gaussian_kde
    import seaborn as sns

    set_wsj_style()

    # Determine risk types to plot
    if risk_types is None:
        risk_types = list(data.keys())

    # Create figure with subplots
    n_risk_types = len(risk_types)
    fig = plt.figure(figsize=figsize)

    if n_risk_types == 1:
        # Single risk type: 2x2 layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    else:
        # Multiple risk types: dynamic layout
        gs = fig.add_gridspec(2, n_risk_types, hspace=0.3, wspace=0.3)

    # Plot correlation matrix heatmap
    for idx, risk_type in enumerate(risk_types):
        if risk_type not in data:
            continue

        risk_data = data[risk_type]

        # Calculate correlation matrix
        if correlation_type == "pearson":
            corr_matrix = np.corrcoef(risk_data.T)
        elif correlation_type == "spearman":
            from scipy.stats import spearmanr

            if risk_data.shape[1] == 1:
                # spearmanr doesn't handle single variable well
                corr_matrix = np.array([[1.0]])
            else:
                corr_result, _ = spearmanr(risk_data)
                if risk_data.shape[1] == 2:
                    # spearmanr returns a scalar for 2 variables
                    if np.isscalar(corr_result):
                        corr_matrix = np.array([[1.0, corr_result], [corr_result, 1.0]])
                    else:
                        corr_matrix = np.array(corr_result)
                else:
                    corr_matrix = np.array(corr_result)
        elif correlation_type == "kendall":
            from scipy.stats import kendalltau

            n_vars = risk_data.shape[1]
            corr_matrix = np.ones((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    tau, _ = kendalltau(risk_data[:, i], risk_data[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")

        # Create correlation heatmap
        if n_risk_types == 1:
            ax_corr = fig.add_subplot(gs[0, 0])
        else:
            ax_corr = fig.add_subplot(gs[0, idx])

        # Handle edge case where corr_matrix might be 0-dimensional
        if corr_matrix.ndim == 0:
            corr_matrix = corr_matrix.reshape(1, 1)
        elif corr_matrix.ndim == 1:
            # If 1D, make it a proper correlation matrix
            n = len(corr_matrix)
            if n == 1:
                corr_matrix = corr_matrix.reshape(1, 1)
            else:
                # Assume it's the upper triangle, create full matrix
                corr_matrix = np.eye(n)

        # Use seaborn heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax_corr,
            cbar_kws={"shrink": 0.8},
        )
        ax_corr.set_title(f"{risk_type.title()} Correlations", fontsize=12, fontweight="bold")

        # Create scatter plot with copula if we have at least 2 variables
        if risk_data.shape[1] >= 2 and show_copula:
            if n_risk_types == 1:
                ax_scatter = fig.add_subplot(gs[0, 1])
            else:
                ax_scatter = fig.add_subplot(gs[1, idx])

            # Use first two variables for scatter plot
            x_data = risk_data[:, 0]
            y_data = risk_data[:, 1]

            # Transform to uniform marginals for copula
            x_uniform = stats.rankdata(x_data) / (len(x_data) + 1)
            y_uniform = stats.rankdata(y_data) / (len(y_data) + 1)

            # Create scatter plot
            ax_scatter.scatter(x_uniform, y_uniform, alpha=0.3, s=10, color=WSJ_COLORS["blue"])

            # Fit and plot copula density contours
            try:
                kde = gaussian_kde(np.vstack([x_uniform, y_uniform]))
                xi, yi = np.mgrid[0:1:50j, 0:1:50j]  # type: ignore[misc]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                ax_scatter.contour(xi, yi, zi, colors=WSJ_COLORS["red"], alpha=0.5, linewidths=1)
            except (ValueError, np.linalg.LinAlgError):
                # If KDE fails, skip contours
                pass

            ax_scatter.set_xlabel("U1 (Uniform)", fontsize=10)
            ax_scatter.set_ylabel("U2 (Uniform)", fontsize=10)
            ax_scatter.set_title(f"{risk_type.title()} Copula", fontsize=12, fontweight="bold")
            ax_scatter.grid(True, alpha=0.3)

    # Add copula density plot if single risk type and enough variables
    if n_risk_types == 1 and show_copula and risk_types:
        risk_data = data[risk_types[0]]
        if risk_data.shape[1] >= 2:
            # Create 2D copula density plot
            ax_density = fig.add_subplot(gs[1, 0])

            x_data = risk_data[:, 0]
            y_data = risk_data[:, 1]

            # Transform to uniform marginals for tail dependence calculation
            x_uniform = stats.rankdata(x_data) / (len(x_data) + 1)
            y_uniform = stats.rankdata(y_data) / (len(y_data) + 1)

            # Transform to normal scores for Gaussian copula
            from scipy.stats import norm

            x_normal = norm.ppf(x_uniform)
            y_normal = norm.ppf(y_uniform)

            # Create density plot
            try:
                kde = gaussian_kde(np.vstack([x_normal, y_normal]))
                xi, yi = np.mgrid[-3:3:100j, -3:3:100j]  # type: ignore[misc]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                im = ax_density.contourf(xi, yi, zi, levels=15, cmap="viridis", alpha=0.7)
                fig.colorbar(im, ax=ax_density, label="Density")
            except (ValueError, np.linalg.LinAlgError):
                # If KDE fails, show scatter instead
                ax_density.scatter(x_normal, y_normal, alpha=0.3, s=10)

            ax_density.set_xlabel("Normal Score 1", fontsize=10)
            ax_density.set_ylabel("Normal Score 2", fontsize=10)
            ax_density.set_title("Gaussian Copula Density", fontsize=12, fontweight="bold")
            ax_density.grid(True, alpha=0.3)

            # Add tail dependence coefficient
            ax_info = fig.add_subplot(gs[1, 1])
            ax_info.axis("off")

            # Calculate empirical tail dependence
            threshold = 0.95
            upper_tail = np.sum((x_uniform > threshold) & (y_uniform > threshold)) / np.sum(
                x_uniform > threshold
            )
            lower_tail = np.sum(
                (x_uniform < (1 - threshold)) & (y_uniform < (1 - threshold))
            ) / np.sum(x_uniform < (1 - threshold))

            info_text = f"""
Dependence Measures:
• {correlation_type.title()}: {corr_matrix[0, 1] if corr_matrix.shape[0] > 1 else 1.0:.3f}
• Upper Tail: {upper_tail:.3f}
• Lower Tail: {lower_tail:.3f}

Sample Size: {len(x_data):,}
Variables: {risk_data.shape[1]}
"""
            ax_info.text(
                0.1,
                0.9,
                info_text,
                transform=ax_info.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.3},
            )
        elif risk_data.shape[1] == 1:
            # Single variable case - show info only
            ax_info = fig.add_subplot(gs[1, 0])
            ax_info.axis("off")

            info_text = f"""
Single Variable Analysis:
• Variable: {risk_types[0]}
• Sample Size: {len(risk_data):,}
• Mean: {np.mean(risk_data):.3f}
• Std Dev: {np.std(risk_data):.3f}

Note: Correlation analysis requires
at least 2 variables.
"""
            ax_info.text(
                0.1,
                0.9,
                info_text,
                transform=ax_info.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.3},
            )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        except (ValueError, TypeError, RuntimeError):
            # Ignore layout errors for complex subplot arrangements
            pass

    return fig


def _add_percentage_labels(ax, group_labels, component_data, component_names):
    """Helper function to add percentage labels to stacked bars."""
    for i in range(len(group_labels)):
        total = sum(component_data[comp][i] for comp in component_names)
        if total <= 0:
            continue

        cumulative = 0
        for component in component_names:
            value = component_data[component][i]
            if value <= 0:
                cumulative += value
                continue

            percentage = (value / total) * 100
            if percentage >= 5:  # Only show label if segment is large enough
                y_pos = cumulative + value / 2
                ax.text(
                    i,
                    y_pos,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9,
                )
            cumulative += value


def plot_premium_decomposition(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    premium_components: Dict[str, Dict[str, Dict[str, float]]],
    company_sizes: Optional[List[str]] = None,
    layers: Optional[List[str]] = None,
    title: str = "Premium Loading Decomposition",
    figsize: Tuple[float, float] = (14, 8),
    show_percentages: bool = True,
    color_scheme: Optional[Dict[str, str]] = None,
) -> Figure:
    """Create premium loading decomposition visualization (Figure C4).

    Shows stacked bar charts breaking down insurance premium into components:
    expected loss (base), volatility load, tail load, expense load, and profit margin.

    Args:
        premium_components: Nested dict with structure:
            {company_size: {layer: {component: value}}}
            Components: 'expected_loss', 'volatility_load', 'tail_load',
            'expense_load', 'profit_margin'
        company_sizes: List of company sizes to show (defaults to all)
        layers: List of insurance layers to show (defaults to all)
        title: Plot title
        figsize: Figure size (width, height)
        show_percentages: Whether to show percentage labels on segments
        color_scheme: Dict mapping component names to colors

    Returns:
        Matplotlib figure with premium decomposition

    Examples:
        >>> components = {
        ...     "Small": {
        ...         "Primary": {"expected_loss": 100, "volatility_load": 20,
        ...                    "tail_load": 15, "expense_load": 10, "profit_margin": 5}
        ...     }
        ... }
        >>> fig = plot_premium_decomposition(components)
    """
    set_wsj_style()

    # Default color scheme
    if color_scheme is None:
        color_scheme = {
            "expected_loss": WSJ_COLORS["blue"],
            "volatility_load": WSJ_COLORS["orange"],
            "tail_load": WSJ_COLORS["red"],
            "expense_load": WSJ_COLORS["green"],
            "profit_margin": WSJ_COLORS["purple"],
        }

    # Get company sizes and layers
    if company_sizes is None:
        company_sizes = list(premium_components.keys())
    if layers is None:
        # Get all unique layers across all company sizes
        all_layers: set[str] = set()
        for comp_data in premium_components.values():
            all_layers.update(comp_data.keys())
        layers = sorted(list(all_layers))

    # Prepare data for plotting
    _n_groups = len(company_sizes) * len(layers)
    group_labels = []
    component_names = [
        "expected_loss",
        "volatility_load",
        "tail_load",
        "expense_load",
        "profit_margin",
    ]
    component_data: Dict[str, List[float]] = {comp: [] for comp in component_names}

    for company_size in company_sizes:
        if company_size not in premium_components:
            continue
        for layer in layers:
            group_labels.append(f"{company_size}\n{layer}")
            if layer in premium_components[company_size]:
                layer_data = premium_components[company_size][layer]
                for comp in component_names:
                    component_data[comp].append(layer_data.get(comp, 0))
            else:
                # No data for this combination
                for comp in component_names:
                    component_data[comp].append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bar chart
    x = np.arange(len(group_labels))
    width = 0.6
    bottom = np.zeros(len(group_labels))

    bars = {}
    for component in component_names:
        values = np.array(component_data[component])
        bars[component] = ax.bar(
            x,
            values,
            width,
            bottom=bottom,
            label=component.replace("_", " ").title(),
            color=color_scheme.get(component, "gray"),
            alpha=0.8,
        )
        bottom += values

    # Add percentage labels if requested
    if show_percentages:
        _add_percentage_labels(ax, group_labels, component_data, component_names)

    # Add total premium values on top of bars
    for i, (_label, total_height) in enumerate(zip(group_labels, bottom)):
        if total_height > 0:
            ax.text(
                i,
                total_height + total_height * 0.01,
                f"${total_height:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Formatting
    ax.set_xlabel("Company Size / Layer", fontsize=12)
    ax.set_ylabel("Premium Amount ($)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=0, ha="center")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add component breakdown table as inset
    if len(company_sizes) <= 3:
        # Create inset axes for summary table
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_inset = inset_axes(ax, width="40%", height="30%", loc="upper right", borderpad=3)
        ax_inset.axis("off")

        # Calculate average percentages
        avg_percentages = {}
        for component in component_names:
            total_value = sum(component_data[component])
            total_premium = sum(bottom)
            avg_percentages[component] = (
                (total_value / total_premium * 100) if total_premium > 0 else 0
            )

        table_text = "Average Composition:\n" + "-" * 20 + "\n"
        for component in component_names:
            comp_name = component.replace("_", " ").title()
            table_text += f"{comp_name}: {avg_percentages[component]:.1f}%\n"

        ax_inset.text(
            0.1,
            0.9,
            table_text,
            transform=ax_inset.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.3},
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            plt.tight_layout()
        except (ValueError, TypeError, RuntimeError):
            # Ignore layout errors for complex subplot arrangements
            pass
    return fig


def plot_capital_efficiency_frontier_3d(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    efficiency_data: Dict[str, Dict[str, np.ndarray]],
    company_sizes: Optional[List[str]] = None,
    optimal_paths: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Capital Efficiency Frontier",
    figsize: Tuple[float, float] = (14, 10),
    view_angles: Optional[Tuple[float, float]] = None,
    export_views: bool = False,
) -> Union[Figure, List[Figure]]:
    """Create 3D capital efficiency frontier visualization (Figure C5).

    Shows 3D surface plot with ROE, Ruin Probability, and Insurance Spend axes,
    with separate surfaces for each company size and highlighted optimal paths.

    Args:
        efficiency_data: Nested dict with structure:
            ``{company_size: {'roe': 2D array (n_ruin x n_spend), 'ruin_prob': 1D array (n_ruin), 'insurance_spend': 1D array (n_spend)}}``
        company_sizes: List of company sizes to show (defaults to all)
        optimal_paths: Dict mapping company size to optimal path coordinates:
            ``{company_size: array of shape (n_points, 3) with [roe, ruin, spend]}``
        title: Plot title
        figsize: Figure size (width, height)
        view_angles: Tuple of (elevation, azimuth) angles for 3D view
        export_views: Whether to return multiple figures with different view angles

    Returns:
        Single figure or list of figures with different viewing angles if export_views=True

    Examples:
        >>> data = {
        ...     "Small": {
        ...         "roe": np.random.rand(20, 30),
        ...         "ruin_prob": np.linspace(0, 0.1, 20),
        ...         "insurance_spend": np.linspace(0, 1e6, 30)
        ...     }
        ... }
        >>> fig = plot_capital_efficiency_frontier_3d(data)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import

    set_wsj_style()

    # Get company sizes
    if company_sizes is None:
        company_sizes = list(efficiency_data.keys())

    # Default view angles
    if view_angles is None:
        view_angles = (20, 45)

    # Color map for different company sizes
    size_colors = {
        "Small": "Blues",
        "Medium": "Greens",
        "Large": "Reds",
    }

    # Create main figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot surfaces for each company size
    for _idx, company_size in enumerate(company_sizes):
        if company_size not in efficiency_data:
            continue

        data = efficiency_data[company_size]
        roe_surface = data.get("roe")
        ruin_probs = data.get("ruin_prob")
        insurance_spends = data.get("insurance_spend")

        if roe_surface is None or ruin_probs is None or insurance_spends is None:
            continue

        # Create meshgrid
        X, Y = np.meshgrid(ruin_probs, insurance_spends)
        Z = roe_surface.T  # Transpose for correct orientation

        # Choose colormap
        cmap_name = size_colors.get(company_size, "viridis")
        cmap = plt.colormaps[cmap_name]

        # Plot surface with transparency
        _surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            alpha=0.6,
            edgecolor="none",
            label=company_size,
        )

        # Add contour lines at the bottom
        ax.contour(X, Y, Z, zdir="z", offset=np.min(Z), colors="gray", alpha=0.3, linewidths=0.5)

    # Plot optimal paths if provided
    if optimal_paths is not None:
        for company_size, path in optimal_paths.items():
            if path is not None and len(path) > 0:
                # Path should have shape (n_points, 3) with columns [ruin_prob, insurance_spend, roe]
                ax.plot(
                    path[:, 0],  # Ruin probability
                    path[:, 1],  # Insurance spend
                    path[:, 2],  # ROE
                    color=WSJ_COLORS.get("red", "red"),
                    linewidth=3,
                    alpha=0.9,
                    label=f"Optimal Path ({company_size})",
                )

                # Mark start and end points
                ax.scatter(
                    [path[0, 0]],
                    [path[0, 1]],
                    [path[0, 2]],
                    color="green",
                    s=100,
                    marker="o",
                    label="Start",
                )
                ax.scatter(
                    [path[-1, 0]],
                    [path[-1, 1]],
                    [path[-1, 2]],
                    color="red",
                    s=100,
                    marker="^",
                    label="End",
                )

    # Labels and formatting
    ax.set_xlabel("Ruin Probability", fontsize=11, labelpad=10)
    ax.set_ylabel("Insurance Spend ($M)", fontsize=11, labelpad=10)
    ax.set_zlabel("ROE (%)", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Set view angle
    ax.view_init(elev=view_angles[0], azim=view_angles[1])

    # Format axes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend (create proxy artists for surfaces)
    from matplotlib.artist import Artist
    from matplotlib.patches import Patch

    legend_elements: List[Artist] = []
    for company_size in company_sizes:
        if company_size in efficiency_data:
            color = size_colors.get(company_size, "viridis")
            cmap = plt.colormaps[color]
            legend_elements.append(
                Patch(facecolor=cmap(0.5), alpha=0.6, label=f"{company_size} Company")
            )

    if optimal_paths:
        from matplotlib.lines import Line2D

        legend_elements.append(
            Line2D([0], [0], color=WSJ_COLORS.get("red", "red"), linewidth=3, label="Optimal Path")
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()

    # Export multiple views if requested
    if export_views:
        figures = [fig]

        # Additional viewing angles
        additional_views = [
            (30, 60),  # Higher elevation, rotated
            (10, 120),  # Low angle from side
            (45, 225),  # Bird's eye from opposite corner
            (5, 0),  # Near ground level, front view
        ]

        for elev, azim in additional_views:
            fig_view = plt.figure(figsize=figsize)
            ax_view = fig_view.add_subplot(111, projection="3d")

            # Recreate the plot with new viewing angle
            for _idx, company_size in enumerate(company_sizes):
                if company_size not in efficiency_data:
                    continue

                data = efficiency_data[company_size]
                roe_surface = data.get("roe")
                ruin_probs = data.get("ruin_prob")
                insurance_spends = data.get("insurance_spend")

                if roe_surface is None or ruin_probs is None or insurance_spends is None:
                    continue

                X, Y = np.meshgrid(ruin_probs, insurance_spends)
                Z = roe_surface.T

                cmap_name = size_colors.get(company_size, "viridis")
                cmap = plt.colormaps[cmap_name]

                ax_view.plot_surface(
                    X,
                    Y,
                    Z,
                    cmap=cmap,
                    alpha=0.6,
                    edgecolor="none",
                )

            # Plot optimal paths
            if optimal_paths is not None:
                for company_size, path in optimal_paths.items():
                    if path is not None and len(path) > 0:
                        ax_view.plot(
                            path[:, 0],
                            path[:, 1],
                            path[:, 2],
                            color=WSJ_COLORS.get("red", "red"),
                            linewidth=3,
                            alpha=0.9,
                        )

            # Set labels and view
            ax_view.set_xlabel("Ruin Probability", fontsize=11, labelpad=10)
            ax_view.set_ylabel("Insurance Spend ($M)", fontsize=11, labelpad=10)
            ax_view.set_zlabel("ROE (%)", fontsize=11, labelpad=10)
            ax_view.set_title(
                f"{title} (View {len(figures)})", fontsize=14, fontweight="bold", pad=20
            )
            ax_view.view_init(elev=elev, azim=azim)

            # Format axes
            ax_view.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2%}"))
            ax_view.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
            ax_view.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

            ax_view.grid(True, alpha=0.3)
            plt.tight_layout()
            figures.append(fig_view)

        return figures

    return fig

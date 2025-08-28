"""Technical appendix visualization functions.

This module provides detailed technical visualization functions for
convergence diagnostics, Pareto frontier analysis, loss distribution validation,
and Monte Carlo convergence analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .core import COLOR_SEQUENCE, WSJ_COLORS, set_wsj_style


def plot_convergence_diagnostics(  # pylint: disable=too-many-locals
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
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

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
    from mpl_toolkits.mplot3d import Axes3D

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


def plot_loss_distribution_validation(
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
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
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
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
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


def plot_monte_carlo_convergence(
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
                bbox=dict(boxstyle="round", facecolor=status_color, alpha=0.3),
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


def plot_enhanced_convergence_diagnostics(
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
    for bar, val in zip(bars, ess_values):
        height = bar.get_height()
        ax_ess.text(
            bar.get_x() + bar.get_width() / 2.0,
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
    for bar, val in zip(bars, mcse_values):
        height = bar.get_height()
        ax_mcse.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle(title, fontsize=14, fontweight="bold")

    return fig

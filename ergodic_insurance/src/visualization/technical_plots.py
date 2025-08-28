"""Technical appendix visualization functions.

This module provides detailed technical visualization functions for
convergence diagnostics, Pareto frontier analysis, and advanced metrics.
"""

from typing import Any, Dict, List, Optional, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

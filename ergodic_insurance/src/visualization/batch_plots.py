"""Batch processing and scenario comparison visualizations.

This module provides visualization functions for batch simulation results,
scenario comparisons, and sensitivity analyses.
"""

from typing import Any, Dict, List, Optional, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .core import COLOR_SEQUENCE, WSJ_COLORS, format_currency, set_wsj_style


def plot_scenario_comparison(  # pylint: disable=too-many-locals
    aggregated_results: Any,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Create comprehensive scenario comparison visualization.

    Compares multiple scenarios across different metrics with bar charts
    highlighting the best performer for each metric.

    Args:
        aggregated_results: AggregatedResults object from batch processing
        metrics: List of metrics to compare (default: key metrics)
        figsize: Figure size (width, height)
        save_path: Path to save figure

    Returns:
        Matplotlib figure with scenario comparisons

    Examples:
        >>> from ergodic_insurance.src.batch_processor import AggregatedResults
        >>> results = AggregatedResults(batch_results)
        >>> fig = plot_scenario_comparison(results, metrics=["mean_growth_rate"])
    """
    from ..batch_processor import AggregatedResults

    if not isinstance(aggregated_results, AggregatedResults):
        raise ValueError("Input must be AggregatedResults from batch processing")

    # Get successful results only
    df = aggregated_results.summary_statistics

    if df.empty:
        print("No successful scenarios to visualize")
        return plt.figure()

    # Default metrics if not specified
    if metrics is None:
        metrics = ["ruin_probability", "mean_growth_rate", "mean_final_assets", "var_99"]
        metrics = [m for m in metrics if m in df.columns]

    # Create subplot grid
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    set_wsj_style()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Create bar plot
        scenarios = df["scenario"]
        values = df[metric]

        bars = ax.bar(range(len(scenarios)), values, color=WSJ_COLORS["blue"], alpha=0.8)

        # Highlight best performer
        if metric == "ruin_probability":  # Lower is better
            best_idx = values.idxmin()
        else:  # Higher is better
            best_idx = values.idxmax()

        bars[best_idx].set_color(WSJ_COLORS["green"])
        bars[best_idx].set_alpha(1.0)

        # Format
        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for j, (value_bar, val) in enumerate(zip(bars, values)):
            height = value_bar.get_height()
            format_str = f"{val:.2%}" if "probability" in metric else f"{val:.2g}"
            ax.text(
                value_bar.get_x() + value_bar.get_width() / 2,
                height,
                format_str,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Remove empty subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Scenario Comparison Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_sensitivity_heatmap(  # pylint: disable=too-many-locals
    aggregated_results: Any,
    metric: str = "mean_growth_rate",
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Create sensitivity analysis heatmap.

    Visualizes sensitivity of outcomes to parameter changes using
    a horizontal bar chart color-coded by impact direction.

    Args:
        aggregated_results: AggregatedResults with sensitivity analysis
        metric: Metric to visualize
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure with sensitivity analysis
    """
    from ..batch_processor import AggregatedResults

    if not isinstance(aggregated_results, AggregatedResults):
        raise ValueError("Input must be AggregatedResults from batch processing")

    sensitivity_df = aggregated_results.sensitivity_analysis

    if sensitivity_df is None or sensitivity_df.empty:
        print("No sensitivity analysis data available")
        return plt.figure()

    # Prepare data for heatmap
    sensitivity_matrix: List[List[float]] = []
    param_names = []

    for _, row in sensitivity_df.iterrows():
        scenario_name = row["scenario"]
        # Extract parameter name from scenario name
        parts = scenario_name.split("_")
        if len(parts) >= 2:
            param = "_".join(parts[1:-1])  # Remove prefix and direction
            if param not in param_names:
                param_names.append(param)

    # Create matrix of sensitivity values
    metric_col = f"{metric}_change_pct"
    if metric_col not in sensitivity_df.columns:
        available = [c for c in sensitivity_df.columns if "_change_pct" in c]
        if available:
            metric_col = available[0]
            print(f"Using {metric_col} instead of requested metric")
        else:
            print("No sensitivity metrics found")
            return plt.figure()

    set_wsj_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Create simple bar plot if matrix creation fails
    scenarios = sensitivity_df["scenario"]
    values = sensitivity_df[metric_col]

    bars = ax.barh(scenarios, values, color=WSJ_COLORS["blue"], alpha=0.8)

    # Color code by positive/negative
    for sens_bar, val in zip(bars, values):
        if val < 0:
            sens_bar.set_color(WSJ_COLORS["red"])
        else:
            sens_bar.set_color(WSJ_COLORS["green"])

    ax.set_xlabel(f"% Change in {metric.replace('_', ' ').title()}")
    ax.set_ylabel("Scenario")
    ax.set_title(f"Sensitivity Analysis: {metric.replace('_', ' ').title()}")
    ax.axvline(x=0, color=WSJ_COLORS["black"], linestyle="-", linewidth=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_parameter_sweep_3d(
    aggregated_results: Any,
    param1: str,
    param2: str,
    metric: str = "mean_growth_rate",
    height: int = 600,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create 3D surface plot for parameter sweep results.

    Visualizes how a metric varies across two parameter dimensions
    using an interactive 3D scatter plot.

    Args:
        aggregated_results: AggregatedResults from grid search
        param1: First parameter name
        param2: Second parameter name
        metric: Metric to plot on z-axis
        height: Figure height in pixels
        save_path: Path to save figure

    Returns:
        Plotly figure with 3D parameter sweep
    """
    from ..batch_processor import AggregatedResults

    if not isinstance(aggregated_results, AggregatedResults):
        raise ValueError("Input must be AggregatedResults from batch processing")

    # Extract parameter values and metric from results
    param1_values = []
    param2_values = []
    metric_values = []

    for result in aggregated_results.batch_results:
        if result.simulation_results:
            overrides = result.metadata.get("parameter_overrides", {})
            if param1 in overrides and param2 in overrides:
                param1_values.append(overrides[param1])
                param2_values.append(overrides[param2])

                if metric == "mean_growth_rate":
                    metric_values.append(np.mean(result.simulation_results.growth_rates))
                elif metric == "ruin_probability":
                    metric_values.append(result.simulation_results.ruin_probability)
                elif metric == "mean_final_assets":
                    metric_values.append(np.mean(result.simulation_results.final_assets))
                else:
                    metric_values.append(result.simulation_results.metrics.get(metric, np.nan))

    if not param1_values:
        print("No parameter sweep data found")
        return go.Figure()

    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=param1_values,
                y=param2_values,
                z=metric_values,
                mode="markers",
                marker={
                    "size": 8,
                    "color": metric_values,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": metric.replace("_", " ").title()},
                },
                text=[
                    f"{param1}: {p1:.3g}<br>{param2}: {p2:.3g}<br>{metric}: {m:.3g}"
                    for p1, p2, m in zip(param1_values, param2_values, metric_values)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"Parameter Sweep: {metric.replace('_', ' ').title()}",
        scene={
            "xaxis_title": param1.replace("_", " ").title(),
            "yaxis_title": param2.replace("_", " ").title(),
            "zaxis_title": metric.replace("_", " ").title(),
        },
        height=height,
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_scenario_convergence(
    batch_results: List[Any],
    metric: str = "mean_growth_rate",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot convergence of metric across scenarios.

    Shows how a metric converges as more scenarios are processed,
    with execution time distribution.

    Args:
        batch_results: List of BatchResult objects
        metric: Metric to track
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure with convergence analysis
    """
    set_wsj_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract metric values in order
    scenarios = []
    values = []
    times = []

    for result in batch_results:
        if result.simulation_results:
            scenarios.append(result.scenario_name)
            times.append(result.execution_time)

            if metric == "mean_growth_rate":
                values.append(np.mean(result.simulation_results.growth_rates))
            elif metric == "ruin_probability":
                values.append(result.simulation_results.ruin_probability)
            elif metric == "mean_final_assets":
                values.append(np.mean(result.simulation_results.final_assets))
            else:
                values.append(result.simulation_results.metrics.get(metric, np.nan))

    if not values:
        print("No data to plot")
        return fig

    # Plot 1: Running average
    running_avg = np.cumsum(values) / np.arange(1, len(values) + 1)
    ax1.plot(running_avg, color=WSJ_COLORS["blue"], linewidth=2)
    ax1.fill_between(range(len(running_avg)), running_avg, alpha=0.3, color=WSJ_COLORS["blue"])
    ax1.set_xlabel("Scenario Number")
    ax1.set_ylabel(f"Running Average {metric.replace('_', ' ').title()}")
    ax1.set_title("Metric Convergence")
    ax1.grid(True, alpha=0.3)

    # Add convergence band
    final_avg = running_avg[-1]
    ax1.axhline(final_avg, color=WSJ_COLORS["red"], linestyle="--", alpha=0.7)
    ax1.fill_between(
        range(len(running_avg)),
        [final_avg * 0.95] * len(running_avg),
        [final_avg * 1.05] * len(running_avg),
        alpha=0.2,
        color=WSJ_COLORS["red"],
    )

    # Plot 2: Execution time distribution
    ax2.hist(times, bins=20, color=WSJ_COLORS["green"], alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Execution Time (seconds)")
    ax2.set_ylabel("Count")
    ax2.set_title("Scenario Execution Times")
    ax2.axvline(
        np.mean(times),
        color=WSJ_COLORS["red"],
        linestyle="--",
        label=f"Mean: {np.mean(times):.1f}s",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Batch Processing Analysis ({len(values)} scenarios)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_parallel_scenarios(
    batch_results: List[Any],
    metrics: List[str],
    figsize: Tuple[float, float] = (12, 8),
    normalize: bool = True,
) -> Figure:
    """Create parallel coordinates plot for scenario comparison.

    Visualizes multiple scenarios across multiple metrics using
    parallel coordinates for comprehensive comparison.

    Args:
        batch_results: List of BatchResult objects
        metrics: List of metrics to include
        figsize: Figure size
        normalize: Whether to normalize metrics to [0, 1]

    Returns:
        Matplotlib figure with parallel coordinates
    """
    import matplotlib.patches as mpatches

    set_wsj_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    data = []
    scenario_names = []

    for result in batch_results:
        if result.simulation_results:
            scenario_names.append(result.scenario_name)
            row = []

            for metric in metrics:
                if metric == "mean_growth_rate":
                    value = np.mean(result.simulation_results.growth_rates)
                elif metric == "ruin_probability":
                    value = result.simulation_results.ruin_probability
                elif metric == "mean_final_assets":
                    value = np.mean(result.simulation_results.final_assets)
                else:
                    value = result.simulation_results.metrics.get(metric, np.nan)
                row.append(value)

            data.append(row)

    if not data:
        print("No data to plot")
        return fig

    data = np.array(data)

    # Normalize if requested
    if normalize:
        for i in range(data.shape[1]):
            col_min = np.nanmin(data[:, i])
            col_max = np.nanmax(data[:, i])
            if col_max > col_min:
                data[:, i] = (data[:, i] - col_min) / (col_max - col_min)

    # Create parallel coordinates
    x = np.arange(len(metrics))

    for i, scenario in enumerate(data):
        color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
        ax.plot(x, scenario, "o-", color=color, alpha=0.7, label=scenario_names[i])

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], rotation=0)
    ax.set_ylabel("Normalized Value" if normalize else "Value")
    ax.set_title("Parallel Scenarios Comparison")
    ax.grid(True, alpha=0.3)

    # Add legend
    if len(scenario_names) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig

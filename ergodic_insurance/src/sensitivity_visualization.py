"""Visualization utilities for sensitivity analysis results.

This module provides publication-ready visualization functions for sensitivity
analysis results, including tornado diagrams, two-way sensitivity heatmaps,
and parameter impact charts.

Example:
    Creating a tornado diagram::

        from ergodic_insurance.src.sensitivity_visualization import plot_tornado_diagram

        # Assuming tornado_data is a DataFrame from SensitivityAnalyzer
        fig = plot_tornado_diagram(
            tornado_data,
            title="Parameter Sensitivity Analysis",
            metric_label="ROE Impact"
        )
        fig.savefig("tornado_diagram.png", dpi=300, bbox_inches='tight')

Author: Alex Filiakov
Date: 2025-01-29
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set default style for publication-ready plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_tornado_diagram(
    tornado_data: pd.DataFrame,
    title: str = "Sensitivity Analysis - Tornado Diagram",
    metric_label: str = "Impact on Objective",
    figsize: Tuple[float, float] = (10, 6),
    n_params: Optional[int] = None,
    color_positive: str = "#2E7D32",
    color_negative: str = "#C62828",
    show_values: bool = True,
) -> Figure:
    """Create a tornado diagram for sensitivity analysis results.

    Args:
        tornado_data: DataFrame with columns: parameter, impact, direction,
                     low_value, high_value, baseline
        title: Plot title
        metric_label: Label for the x-axis
        figsize: Figure size as (width, height)
        n_params: Number of top parameters to show (None for all)
        color_positive: Color for positive impacts
        color_negative: Color for negative impacts
        show_values: Whether to show numeric values on bars

    Returns:
        Matplotlib Figure object
    """
    # Select top n parameters if specified
    if n_params is not None and len(tornado_data) > n_params:
        tornado_data = tornado_data.head(n_params)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    n = len(tornado_data)
    y_pos = np.arange(n)

    # Calculate bar widths (centered on baseline)
    baseline_values = tornado_data["baseline"].values
    low_values = tornado_data["low_value"].values
    high_values = tornado_data["high_value"].values

    # Normalize to percentage change from baseline
    low_change = (low_values - baseline_values) / np.abs(baseline_values) * 100
    high_change = (high_values - baseline_values) / np.abs(baseline_values) * 100

    # Create bars
    for i, (idx, row) in enumerate(tornado_data.iterrows()):
        color = color_positive if row["direction"] == "positive" else color_negative

        # Draw bar from low to high
        left = low_change[i]
        width = high_change[i] - low_change[i]

        bar = ax.barh(
            i, width, left=left, height=0.6, color=color, alpha=0.7, edgecolor="black", linewidth=1
        )

        # Add value labels if requested
        if show_values:
            # Low value
            ax.text(left - 0.5, i, f"{low_values[i]:.2g}", ha="right", va="center", fontsize=9)
            # High value
            ax.text(
                left + width + 0.5, i, f"{high_values[i]:.2g}", ha="left", va="center", fontsize=9
            )

    # Add baseline line
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0, n + 0.1, "Baseline", ha="center", fontsize=10, style="italic")

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tornado_data["parameter"].values)
    ax.set_xlabel(f"{metric_label} (% change from baseline)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_two_way_sensitivity(
    result: "TwoWaySensitivityResult",
    title: Optional[str] = None,
    cmap: str = "RdYlGn",
    figsize: Tuple[float, float] = (10, 8),
    show_contours: bool = True,
    contour_levels: Optional[int] = 10,
    optimal_point: Optional[Tuple[float, float]] = None,
    fmt: str = ".2f",
) -> Figure:
    """Create a heatmap for two-way sensitivity analysis.

    Args:
        result: TwoWaySensitivityResult object
        title: Plot title (auto-generated if None)
        cmap: Colormap name
        figsize: Figure size as (width, height)
        show_contours: Whether to show contour lines
        contour_levels: Number of contour levels
        optimal_point: Optional (param1_value, param2_value) to mark
        fmt: Format string for value annotations

    Returns:
        Matplotlib Figure object
    """
    from ergodic_insurance.src.sensitivity import TwoWaySensitivityResult

    if title is None:
        title = f"{result.metric_name} Sensitivity: {result.parameter1} vs {result.parameter2}"

    fig, ax = plt.subplots(figsize=figsize)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(result.values1, result.values2, indexing="ij")

    # Create heatmap
    im = ax.pcolormesh(X, Y, result.metric_grid, cmap=cmap, shading="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=result.metric_name)

    # Add contours if requested
    if show_contours:
        contours = ax.contour(
            X,
            Y,
            result.metric_grid,
            levels=contour_levels,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt=fmt)

    # Mark optimal point if provided
    if optimal_point is not None:
        ax.plot(optimal_point[0], optimal_point[1], "r*", markersize=15, label="Optimal Point")
        ax.legend()

    # Customize axes
    ax.set_xlabel(result.parameter1, fontsize=12)
    ax.set_ylabel(result.parameter2, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()

    return fig


def plot_parameter_sweep(
    result: "SensitivityResult",
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    normalize: bool = False,
    mark_baseline: bool = True,
) -> Figure:
    """Plot multiple metrics against parameter variations.

    Args:
        result: SensitivityResult object
        metrics: List of metrics to plot (None for all)
        title: Plot title (auto-generated if None)
        figsize: Figure size as (width, height)
        normalize: Whether to normalize metrics to [0, 1]
        mark_baseline: Whether to mark the baseline value

    Returns:
        Matplotlib Figure object
    """
    from ergodic_insurance.src.sensitivity import SensitivityResult

    if metrics is None:
        metrics = list(result.metrics.keys())

    if title is None:
        title = f"Sensitivity Analysis: {result.parameter}"

    # Determine subplot layout
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = result.metrics[metric]

        # Normalize if requested
        if normalize and values.max() != values.min():
            values = (values - values.min()) / (values.max() - values.min())

        # Plot line
        ax.plot(result.variations, values, "o-", linewidth=2, markersize=6)

        # Mark baseline
        if mark_baseline:
            baseline_idx = len(result.variations) // 2
            ax.axvline(
                x=result.baseline_value, color="red", linestyle="--", alpha=0.5, label="Baseline"
            )
            ax.plot(result.baseline_value, values[baseline_idx], "r*", markersize=12)

        # Customize subplot
        ax.set_xlabel(result.parameter, fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add trend annotation
        trend = np.polyfit(result.variations, values, 1)[0]
        trend_text = "↑" if trend > 0 else "↓" if trend < 0 else "→"
        ax.text(
            0.95,
            0.95,
            trend_text,
            transform=ax.transAxes,
            fontsize=16,
            ha="right",
            va="top",
            alpha=0.5,
        )

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    # Add main title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    return fig


def create_sensitivity_report(
    analyzer: "SensitivityAnalyzer",
    parameters: List[str],
    output_dir: Optional[str] = None,
    metric: str = "optimal_roe",
    formats: List[str] = ["png", "pdf"],
) -> Dict[str, Any]:
    """Generate a complete sensitivity analysis report.

    Args:
        analyzer: SensitivityAnalyzer object with results
        parameters: List of parameters to analyze
        output_dir: Directory to save figures (None for no saving)
        metric: Primary metric for analysis
        formats: File formats to save figures in

    Returns:
        Dictionary with generated figures and analysis summary
    """
    from pathlib import Path

    report = {"figures": {}, "summary": {}, "data": {}}

    # Generate tornado diagram
    print("Generating tornado diagram...")
    tornado_data = analyzer.create_tornado_diagram(parameters, metric=metric)
    report["data"]["tornado"] = tornado_data

    fig_tornado = plot_tornado_diagram(
        tornado_data,
        title=f"Sensitivity Analysis - {metric.replace('_', ' ').title()}",
        metric_label=metric.replace("_", " ").title(),
    )
    report["figures"]["tornado"] = fig_tornado

    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            filename = output_path / f"tornado_diagram.{fmt}"
            fig_tornado.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved: {filename}")

    # Generate parameter sweeps for top 3 most impactful parameters
    top_params = tornado_data.head(3)["parameter"].values

    for param in top_params:
        print(f"Analyzing parameter: {param}")

        # Run sensitivity analysis
        result = analyzer.analyze_parameter(param)
        report["data"][f"sweep_{param}"] = result

        # Create sweep plot
        fig_sweep = plot_parameter_sweep(result, title=f"Parameter Sweep: {param}")
        report["figures"][f"sweep_{param}"] = fig_sweep

        # Save if requested
        if output_dir:
            for fmt in formats:
                filename = output_path / f"sweep_{param}.{fmt}"
                fig_sweep.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"Saved: {filename}")

    # Generate summary statistics
    report["summary"]["most_impactful"] = tornado_data.iloc[0]["parameter"]
    report["summary"]["least_impactful"] = tornado_data.iloc[-1]["parameter"]
    report["summary"]["total_parameters"] = len(tornado_data)
    report["summary"]["primary_metric"] = metric

    # Calculate relative importances
    total_impact = tornado_data["impact"].sum()
    if total_impact > 0:
        tornado_data["relative_importance"] = tornado_data["impact"] / total_impact * 100
        report["summary"]["relative_importances"] = tornado_data[
            ["parameter", "relative_importance"]
        ].to_dict("records")

    print("Sensitivity report generation complete!")

    return report


def plot_sensitivity_matrix(
    results: Dict[str, "SensitivityResult"],
    metric: str = "optimal_roe",
    figsize: Tuple[float, float] = (12, 10),
    cmap: str = "coolwarm",
    show_values: bool = True,
) -> Figure:
    """Create a matrix plot showing sensitivity across multiple parameters.

    Args:
        results: Dictionary of parameter names to SensitivityResult objects
        metric: Metric to display
        figsize: Figure size as (width, height)
        cmap: Colormap name
        show_values: Whether to show numeric values in cells

    Returns:
        Matplotlib Figure object
    """
    # Extract data for matrix
    params = list(results.keys())
    n_params = len(params)

    # Find common variation points (assuming normalized to percentages)
    variation_points = [-30, -20, -10, 0, 10, 20, 30]  # Percentage changes

    # Create matrix
    matrix = np.zeros((n_params, len(variation_points)))

    for i, param in enumerate(params):
        result = results[param]
        baseline_idx = len(result.variations) // 2
        baseline_metric = result.metrics[metric][baseline_idx]

        # Interpolate to common points
        param_pct = (result.variations - result.baseline_value) / result.baseline_value * 100
        metric_pct = (result.metrics[metric] - baseline_metric) / abs(baseline_metric) * 100

        for j, pct in enumerate(variation_points):
            # Find closest point or interpolate
            idx = np.argmin(np.abs(param_pct - pct))
            matrix[i, j] = metric_pct[idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=f"{metric} (% change)")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(variation_points)))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels([f"{v:+d}%" for v in variation_points])
    ax.set_yticklabels(params)

    # Add values if requested
    if show_values:
        for i in range(n_params):
            for j in range(len(variation_points)):
                text = ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if abs(matrix[i, j]) > 5 else "black",
                    fontsize=8,
                )

    # Customize
    ax.set_xlabel("Parameter Change (%)", fontsize=12)
    ax.set_title(
        f'Sensitivity Matrix: {metric.replace("_", " ").title()}',
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add grid
    ax.set_xticks(np.arange(len(variation_points) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_params + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    return fig

"""Visualization utilities for professional WSJ-style plots.

This module provides standardized plotting functions with Wall Street Journal
aesthetic for insurance analysis and risk metrics visualization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

# WSJ Color Palette
WSJ_COLORS = {
    "blue": "#0080C7",  # Primary blue
    "dark_blue": "#003F5C",  # Dark blue
    "red": "#D32F2F",  # Red for negative/warning
    "green": "#4CAF50",  # Green for positive
    "gray": "#666666",  # Gray for secondary
    "light_gray": "#E0E0E0",  # Light gray for grid
    "black": "#000000",  # Black for text
    "orange": "#FF9800",  # Orange for highlights
    "purple": "#7B1FA2",  # Purple for special
    "teal": "#00796B",  # Teal for alternative
}

# Professional color sequence for multiple series
COLOR_SEQUENCE = [
    WSJ_COLORS["blue"],
    WSJ_COLORS["red"],
    WSJ_COLORS["green"],
    WSJ_COLORS["orange"],
    WSJ_COLORS["purple"],
    WSJ_COLORS["teal"],
    WSJ_COLORS["dark_blue"],
]


def set_wsj_style():
    """Set matplotlib to use WSJ-style formatting."""
    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Update rcParams for WSJ style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": WSJ_COLORS["gray"],
            "axes.linewidth": 0.8,
            "grid.color": WSJ_COLORS["light_gray"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "lines.linewidth": 2,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
        }
    )


def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as currency with appropriate suffix.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "$1.5M", "$750K")
    """
    if abs(value) >= 1e9:
        return f"${value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage.

    Args:
        value: Numeric value (0.05 = 5%)
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "5.0%")
    """
    return f"{value*100:.{decimals}f}%"


class WSJFormatter:
    """Formatter for WSJ-style axis labels."""

    @staticmethod
    def currency_formatter(x, pos):
        """Format axis values as currency."""
        return format_currency(x, decimals=0)

    @staticmethod
    def percentage_formatter(x, pos):
        """Format axis values as percentage."""
        return format_percentage(x, decimals=0)

    @staticmethod
    def millions_formatter(x, pos):
        """Format axis values in millions."""
        return f"{x/1e6:.0f}M"


def plot_loss_distribution(
    losses: np.ndarray,
    title: str = "Loss Distribution",
    bins: int = 50,
    show_metrics: bool = True,
    var_levels: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """Create WSJ-style loss distribution plot.

    Args:
        losses: Array of loss values
        title: Plot title
        bins: Number of histogram bins
        show_metrics: Whether to show VaR/TVaR lines
        var_levels: VaR confidence levels to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    set_wsj_style()

    if var_levels is None:
        var_levels = [0.95, 0.99]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(
        losses, bins=bins, color=WSJ_COLORS["blue"], alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax1.set_xlabel("Loss Amount")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Losses")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(WSJFormatter.currency_formatter))
    ax1.grid(True, alpha=0.3)

    # Add VaR lines if requested
    if show_metrics:
        from ergodic_insurance.src.risk_metrics import RiskMetrics

        metrics = RiskMetrics(losses)

        colors = [WSJ_COLORS["red"], WSJ_COLORS["orange"]]
        for i, level in enumerate(var_levels):
            var = metrics.var(level)
            ax1.axvline(
                var,
                color=colors[i % len(colors)],
                linestyle="--",
                linewidth=1.5,
                label=f"VaR({level:.0%}): {format_currency(var)}",
            )
        ax1.legend(loc="upper right")

    # Q-Q plot
    from scipy import stats

    stats.probplot(losses, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normal)")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.grid(True, alpha=0.3)

    # Format Q-Q plot y-axis
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(WSJFormatter.currency_formatter))

    # Style the Q-Q plot line
    lines = ax2.get_lines()
    if lines:
        lines[0].set_color(WSJ_COLORS["blue"])
        lines[0].set_marker("o")
        lines[0].set_markersize(3)
        lines[0].set_markerfacecolor(WSJ_COLORS["blue"])
        lines[0].set_alpha(0.5)
        if len(lines) > 1:
            lines[1].set_color(WSJ_COLORS["red"])
            lines[1].set_linewidth(2)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_return_period_curve(
    return_periods: np.ndarray,
    losses: np.ndarray,
    scenarios: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Return Period Curves",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Create WSJ-style return period curve.

    Args:
        return_periods: Array of return periods (years)
        losses: Corresponding loss amounts
        scenarios: Optional dict of scenario names to loss arrays
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    set_wsj_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot main curve
    ax.semilogx(
        return_periods,
        losses / 1e6,
        "o-",
        color=WSJ_COLORS["blue"],
        linewidth=2.5,
        markersize=8,
        label="Base Case",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    # Plot additional scenarios if provided
    if scenarios:
        for i, (name, scenario_losses) in enumerate(scenarios.items()):
            ax.semilogx(
                return_periods,
                scenario_losses / 1e6,
                "o-",
                color=COLOR_SEQUENCE[(i + 1) % len(COLOR_SEQUENCE)],
                linewidth=2,
                markersize=6,
                label=name,
                alpha=0.8,
            )

    ax.set_xlabel("Return Period (years)", fontsize=12)
    ax.set_ylabel("Loss Amount ($M)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor=WSJ_COLORS["gray"])

    # Add annotations for key return periods
    key_periods = [10, 100, 250]
    for period in key_periods:
        if period in return_periods:
            idx = np.where(return_periods == period)[0][0]
            loss_val = losses[idx] / 1e6
            ax.annotate(
                f"{period}yr\n${loss_val:.1f}M",
                xy=(period, loss_val),
                xytext=(period, loss_val * 1.1),
                fontsize=9,
                ha="center",
                color=WSJ_COLORS["gray"],
            )

    plt.tight_layout()
    return fig


def plot_insurance_layers(
    layers: List[Dict[str, float]],
    total_limit: float,
    title: str = "Insurance Program Structure",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Create WSJ-style insurance layer visualization.

    Args:
        layers: List of layer dictionaries with 'attachment', 'limit', 'premium'
        total_limit: Total program limit
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    set_wsj_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Layer structure chart
    bottoms = []
    heights = []
    colors = []
    labels = []

    for i, layer in enumerate(layers):
        bottoms.append(layer["attachment"])
        heights.append(layer["limit"])
        colors.append(COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)])
        labels.append(f"Layer {i+1}")

    bars = ax1.bar(
        range(len(layers)),
        heights,
        bottom=bottoms,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add layer annotations
    for i, (bar, layer) in enumerate(zip(bars, layers)):
        height = bar.get_height()
        bottom = bar.get_y()

        # Layer info
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bottom + height / 2,
            f'{format_currency(layer["limit"])}\n@ {layer["premium"]:.2%}',
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Attachment point
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bottom,
            f"{format_currency(bottom)}",
            ha="center",
            va="top",
            fontsize=9,
            color=WSJ_COLORS["gray"],
        )

    ax1.set_ylabel("Coverage Level", fontsize=12)
    ax1.set_title("Layer Structure", fontsize=12)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(labels)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(WSJFormatter.currency_formatter))
    ax1.grid(True, axis="y", alpha=0.3)

    # Premium breakdown pie chart
    premiums = [layer["premium"] * layer["limit"] for layer in layers]

    wedges, texts, autotexts = ax2.pie(
        premiums, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )

    # Style the pie chart
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax2.set_title("Premium Distribution", fontsize=12)

    # Add total premium annotation
    total_premium = sum(premiums)
    fig.text(
        0.5,
        0.02,
        f"Total Annual Premium: {format_currency(total_premium)}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def create_interactive_dashboard(
    results: Dict[str, Any], title: str = "Monte Carlo Simulation Dashboard"
) -> go.Figure:
    """Create interactive Plotly dashboard with WSJ styling.

    Args:
        results: Dictionary with simulation results
        title: Dashboard title

    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Growth Rate Distribution",
            "Loss Exceedance Curve",
            "Convergence Diagnostics",
            "Risk Metrics",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # WSJ-style layout
    layout_theme = {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 11, "color": WSJ_COLORS["black"]},
        "title": {"font": {"size": 16, "color": WSJ_COLORS["black"]}},
        "xaxis": {"gridcolor": WSJ_COLORS["light_gray"], "gridwidth": 0.5},
        "yaxis": {"gridcolor": WSJ_COLORS["light_gray"], "gridwidth": 0.5},
        "colorway": COLOR_SEQUENCE,
    }

    # Growth rate histogram
    if "growth_rates" in results:
        fig.add_trace(
            go.Histogram(
                x=results["growth_rates"],
                nbinsx=50,
                marker_color=WSJ_COLORS["blue"],
                opacity=0.7,
                name="Growth Rate",
            ),
            row=1,
            col=1,
        )

    # Loss exceedance curve
    if "losses" in results:
        sorted_losses = np.sort(results["losses"])[::-1]
        exceedance_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

        fig.add_trace(
            go.Scatter(
                x=sorted_losses / 1e6,
                y=exceedance_prob,
                mode="lines",
                line=dict(color=WSJ_COLORS["red"], width=2),
                name="Exceedance",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Loss Amount ($M)", row=1, col=2)
        fig.update_yaxes(title_text="Exceedance Probability", type="log", row=1, col=2)

    # Convergence diagnostics
    if "convergence" in results:
        iterations = results["convergence"].get("iterations", [])
        r_hat = results["convergence"].get("r_hat", [])

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=r_hat,
                mode="lines+markers",
                line=dict(color=WSJ_COLORS["green"], width=2),
                marker=dict(size=6),
                name="R-hat",
            ),
            row=2,
            col=1,
        )

        # Add convergence threshold line
        fig.add_hline(
            y=1.1,
            line_dash="dash",
            line_color=WSJ_COLORS["orange"],
            annotation_text="Convergence Threshold",
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Iterations", row=2, col=1)
        fig.update_yaxes(title_text="R-hat Statistic", row=2, col=1)

    # Risk metrics bar chart
    if "metrics" in results:
        metric_names = ["VaR(95%)", "VaR(99%)", "TVaR(99%)", "Expected Shortfall"]
        metric_values = [
            results["metrics"].get("var_95", 0) / 1e6,
            results["metrics"].get("var_99", 0) / 1e6,
            results["metrics"].get("tvar_99", 0) / 1e6,
            results["metrics"].get("expected_shortfall", 0) / 1e6,
        ]

        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=COLOR_SEQUENCE[: len(metric_names)],
                text=[f"${v:.1f}M" for v in metric_values],
                textposition="outside",
                name="Risk Metrics",
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="Amount ($M)", row=2, col=2)

    # Update layout
    fig.update_layout(title_text=title, showlegend=False, height=800, **layout_theme)

    # Update all axes
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=WSJ_COLORS["light_gray"])
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=WSJ_COLORS["light_gray"])

    return fig


def plot_convergence_diagnostics(
    convergence_stats: Dict[str, Any],
    title: str = "Convergence Diagnostics",
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """Create comprehensive convergence diagnostics plot.

    Args:
        convergence_stats: Dictionary with convergence statistics
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
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
        for bar, val in zip(bars, mcse_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig

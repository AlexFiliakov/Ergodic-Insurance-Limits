"""Visualization utilities for professional WSJ-style plots.

This module provides standardized plotting functions with Wall Street Journal
aesthetic for insurance analysis and risk metrics visualization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Import new visualization infrastructure
try:
    from .visualization.figure_factory import FigureFactory
    from .visualization.style_manager import StyleManager, Theme

    _FACTORY_AVAILABLE = True
except ImportError:
    _FACTORY_AVAILABLE = False

# WSJ Color Palette
WSJ_COLORS = {
    "light_blue": "#ADD8E6",  # Light Blue for additional styling
    "blue": "#0080C7",  # Primary blue
    "dark_blue": "#003F5C",  # Dark blue
    "red": "#D32F2F",  # Red for negative/warning
    "green": "#4CAF50",  # Green for positive
    "gray": "#666666",  # Gray for secondary
    "light_gray": "#E0E0E0",  # Light gray for grid
    "black": "#000000",  # Black for text
    "orange": "#FF9800",  # Orange for highlights
    "yellow": "#FFD700",  # Yellow for highlights
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

# Global factory instance (created on first use)
_global_factory = None


def get_figure_factory(theme: Optional["Theme"] = None) -> Optional["FigureFactory"]:
    """Get or create global figure factory instance.

    Args:
        theme: Optional theme to use (defaults to DEFAULT)

    Returns:
        FigureFactory instance if available, None otherwise
    """
    global _global_factory
    if _FACTORY_AVAILABLE:
        if _global_factory is None or theme is not None:
            theme = theme or Theme.DEFAULT
            _global_factory = FigureFactory(theme=theme)
        return _global_factory
    return None


def set_wsj_style(use_factory: bool = False, theme: Optional["Theme"] = None):
    """Set matplotlib to use WSJ-style formatting.

    Args:
        use_factory: Whether to use new factory-based styling if available
        theme: Optional theme to use with factory (defaults to DEFAULT)
    """
    # Use factory-based styling if available and requested
    if use_factory and _FACTORY_AVAILABLE:
        factory = get_figure_factory(theme)
        if factory:
            factory.style_manager.apply_style()
            return

    # Fallback to original WSJ style implementation
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


def format_currency(value: float, decimals: int = 0, abbreviate: bool = False) -> str:
    """Format value as currency.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        abbreviate: If True, use K/M/B notation for large numbers

    Returns:
        Formatted string (e.g., "$1,000" or "$1K" if abbreviate=True)
    """
    if abbreviate:
        if abs(value) >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        if abs(value) >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        if abs(value) >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        return f"${value:.{decimals}f}"
    # Handle negative values
    if value < 0:
        return f"-${abs(value):,.{decimals}f}"
    return f"${value:,.{decimals}f}"


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
        return format_currency(x, decimals=0, abbreviate=True)

    @staticmethod
    def currency(x: float, decimals: int = 1) -> str:  # pylint: disable=too-many-return-statements
        """Format value as currency (shortened method name)."""
        sign = "-" if x < 0 else ""
        x = abs(x)

        if x >= 1e12:
            if x == int(x / 1e12) * 1e12:  # Whole trillions
                return f"{sign}${int(x/1e12)}T"
            return f"{sign}${x/1e12:.{decimals}f}T"
        if x >= 1e9:
            if x == int(x / 1e9) * 1e9:  # Whole billions
                return f"{sign}${int(x/1e9)}B"
            return f"{sign}${x/1e9:.{decimals}f}B"
        if x >= 1e6:
            if x == int(x / 1e6) * 1e6:  # Whole millions
                return f"{sign}${int(x/1e6)}M"
            return f"{sign}${x/1e6:.{decimals}f}M"
        if x >= 1e3:
            if x == int(x / 1e3) * 1e3:  # Whole thousands
                return f"{sign}${int(x/1e3)}K"
            return f"{sign}${x/1e3:.{decimals}f}K"
        if 0 < x < 1:
            return f"{sign}${x:.2f}"
        return f"{sign}${int(x)}" if x == int(x) else f"{sign}${x:.{decimals}f}"

    @staticmethod
    def percentage_formatter(x, pos):
        """Format axis values as percentage."""
        return format_percentage(x, decimals=0)

    @staticmethod
    def percentage(x: float, decimals: int = 1) -> str:
        """Format value as percentage (shortened method name)."""
        return f"{x*100:.{decimals}f}%"

    @staticmethod
    def number(x: float, decimals: int = 2) -> str:
        """Format large numbers with appropriate suffix."""
        if abs(x) >= 1e12:
            if abs(x) >= 1e15:
                # Very large numbers - show in trillions with multiplier
                return f"{int(x/1e12)}T"
            return f"{x/1e12:.{decimals}f}T"
        if abs(x) >= 1e9:
            return f"{x/1e9:.{decimals}f}B"
        if abs(x) >= 1e6:
            return f"{x/1e6:.{decimals}f}M"
        if abs(x) >= 1e3:
            return f"{x/1e3:.{decimals}f}K"
        return f"{int(x)}" if x == int(x) else f"{x:.{decimals}f}"

    @staticmethod
    def millions_formatter(x, pos):
        """Format axis values in millions."""
        return f"{x/1e6:.0f}M"


def create_styled_figure(
    size_type: str = "medium", theme: Optional["Theme"] = None, use_factory: bool = True, **kwargs
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """Create a figure with automatic styling applied.

    Args:
        size_type: Size preset (small, medium, large, blog, technical, presentation)
        theme: Optional theme to use
        use_factory: Whether to use factory if available
        **kwargs: Additional arguments for figure creation

    Returns:
        Tuple of (figure, axes)
    """
    if use_factory and _FACTORY_AVAILABLE:
        factory = get_figure_factory(theme)
        if factory:
            return factory.create_figure(size_type=size_type, **kwargs)

    # Fallback to standard matplotlib
    size_map = {
        "small": (6, 4),
        "medium": (8, 6),
        "large": (12, 8),
        "blog": (8, 6),
        "technical": (10, 8),
        "presentation": (10, 7.5),
    }
    figsize = size_map.get(size_type, (8, 6))
    set_wsj_style()
    return plt.subplots(figsize=figsize, **kwargs)


def plot_loss_distribution(  # pylint: disable=too-many-locals,too-many-statements
    losses: Union[np.ndarray, pd.DataFrame],
    title: str = "Loss Distribution",
    bins: int = 50,
    show_metrics: bool = True,
    var_levels: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_stats: bool = False,
    log_scale: bool = False,
    use_factory: bool = False,
    theme: Optional["Theme"] = None,
) -> Figure:
    """Create WSJ-style loss distribution plot.

    Args:
        losses: Array of loss values or DataFrame with 'amount' column
        title: Plot title
        bins: Number of histogram bins
        show_metrics: Whether to show VaR/TVaR lines
        var_levels: VaR confidence levels to show
        figsize: Figure size
        show_stats: Whether to show statistics
        log_scale: Whether to use log scale
        use_factory: Whether to use new visualization factory if available
        theme: Optional theme to use with factory

    Returns:
        Matplotlib figure
    """
    set_wsj_style(use_factory=use_factory, theme=theme)

    # Handle DataFrame input
    if isinstance(losses, pd.DataFrame):
        if "amount" in losses.columns:
            losses_array: np.ndarray = np.asarray(losses["amount"].values)
        else:
            # Use the first numeric column
            numeric_cols = losses.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                losses_array = np.asarray(losses[numeric_cols[0]].values)
            else:
                raise ValueError("DataFrame must have at least one numeric column")
    else:
        # Convert to numpy array if needed
        losses_array = np.asarray(losses)

    losses = losses_array

    if var_levels is None:
        var_levels = [0.95, 0.99]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Check for empty data and create empty plot
    if len(losses) == 0:
        ax1.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
        )
        ax1.set_title("Distribution of Losses")
        ax2.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Q-Q Plot")
        plt.tight_layout()
        return fig

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
        from .risk_metrics import RiskMetrics

        metrics = RiskMetrics(losses)

        colors = [WSJ_COLORS["red"], WSJ_COLORS["orange"]]
        for i, level in enumerate(var_levels):
            var = metrics.var(level)
            var_value = var if isinstance(var, float) else var.value
            ax1.axvline(
                var_value,
                color=colors[i % len(colors)],
                linestyle="--",
                linewidth=1.5,
                label=f"VaR({level:.0%}): {format_currency(var_value)}",
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


def plot_return_period_curve(  # pylint: disable=too-many-locals
    losses: Union[np.ndarray, pd.DataFrame],
    return_periods: Optional[np.ndarray] = None,
    scenarios: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Return Period Curves",
    figsize: Tuple[int, int] = (10, 6),
    confidence_level: float = 0.95,
    show_grid: bool = True,
) -> Figure:
    """Create WSJ-style return period curve.

    Args:
        losses: Loss amounts (array or DataFrame)
        return_periods: Array of return periods (years), optional
        scenarios: Optional dict of scenario names to loss arrays
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    set_wsj_style()

    # Handle DataFrame input
    if isinstance(losses, pd.DataFrame):
        if "amount" in losses.columns:
            losses_array: np.ndarray = np.asarray(losses["amount"].values)
        else:
            # Use the first numeric column
            numeric_cols = losses.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                losses_array = np.asarray(losses[numeric_cols[0]].values)
            else:
                raise ValueError("DataFrame must have at least one numeric column")
    else:
        # Convert to numpy array if needed
        losses_array = np.asarray(losses)

    losses = losses_array

    # Calculate return periods if not provided
    if return_periods is None:
        # Sort losses in descending order
        sorted_losses = np.sort(losses)[::-1]
        n = len(sorted_losses)
        # Calculate empirical return periods
        return_periods = n / np.arange(1, n + 1)
        losses = sorted_losses
    else:
        # Ensure losses are sorted by return period
        sort_idx = np.argsort(return_periods)
        return_periods = return_periods[sort_idx]
        losses = losses[sort_idx]

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


def plot_insurance_layers(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    layers: Union[List[Dict[str, float]], pd.DataFrame],
    total_limit: Optional[float] = None,
    title: str = "Insurance Program Structure",
    figsize: Tuple[int, int] = (10, 6),
    losses: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    loss_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    show_expected_loss: bool = False,
) -> Figure:
    """Create WSJ-style insurance layer visualization.

    Args:
        layers: List of layer dictionaries or DataFrame with 'attachment', 'limit' columns
        total_limit: Total program limit (calculated from layers if not provided)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    set_wsj_style()

    # Handle loss_data parameter (alias for losses)
    if loss_data is not None and losses is None:
        losses = loss_data

    # Handle DataFrame input
    if isinstance(layers, pd.DataFrame):
        # Check for empty DataFrame
        if layers.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.text(
                0.5,
                0.5,
                "No layers defined",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=12,
            )
            ax1.set_title("Layer Structure")
            ax2.text(
                0.5,
                0.5,
                "No layers defined",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("Premium Distribution")
            plt.tight_layout()
            return fig

        # Convert DataFrame to list of dicts
        layer_list = []
        for _, row in layers.iterrows():
            layer_dict = {
                "attachment": row.get("attachment", 0),
                "limit": row.get("limit", 0),
                "premium": row.get("premium_rate", row.get("premium", 0)),
            }
            layer_list.append(layer_dict)
        layers = layer_list

    # Check for empty list
    if not layers:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.text(
            0.5,
            0.5,
            "No layers defined",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
        )
        ax1.set_title("Layer Structure")
        ax2.text(
            0.5,
            0.5,
            "No layers defined",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Premium Distribution")
        plt.tight_layout()
        return fig

    # Calculate total limit if not provided
    if total_limit is None:
        total_limit = max(layer["attachment"] + layer["limit"] for layer in layers)

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
    for i, (layer_bar, layer) in enumerate(zip(bars, layers)):
        height = layer_bar.get_height()
        bottom = layer_bar.get_y()

        # Layer info
        ax1.text(
            layer_bar.get_x() + layer_bar.get_width() / 2,
            bottom + height / 2,
            f'{format_currency(layer["limit"])}\n@ {layer["premium"]:.2%}',
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Attachment point
        ax1.text(
            layer_bar.get_x() + layer_bar.get_width() / 2,
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

    # Add loss overlay if provided
    if losses is not None:
        # Convert to numpy array if needed
        loss_values: np.ndarray
        if isinstance(losses, np.ndarray):
            # Handle numpy arrays first
            loss_values = losses.flatten()
        elif hasattr(losses, "values"):
            # Handle pandas DataFrame and Series
            values = getattr(losses, "values")
            if hasattr(values, "flatten"):
                loss_values = values.flatten()
            else:
                loss_values = values
        else:
            # Handle any other array-like objects
            loss_values = np.asarray(losses).flatten()

        # Add loss distribution overlay on the layer structure chart
        if len(loss_values) > 0 and show_expected_loss:
            # Calculate expected loss
            expected_loss = np.mean(loss_values)

            # Add horizontal line for expected loss
            ax1.axhline(
                y=expected_loss,
                color=WSJ_COLORS["orange"],
                linestyle="--",
                linewidth=2,
                label=f"Expected Loss: {format_currency(expected_loss)}",
                alpha=0.7,
            )

            # Add text annotation for expected loss
            ax1.text(
                len(layers) - 0.5,
                expected_loss,
                f"Expected: {format_currency(expected_loss)}",
                va="bottom",
                ha="right",
                fontsize=9,
                color=WSJ_COLORS["orange"],
                fontweight="bold",
            )

            # Add legend if we have the expected loss line
            ax1.legend(loc="upper left", framealpha=0.9)

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
    results: Union[Dict[str, Any], pd.DataFrame],
    title: str = "Monte Carlo Simulation Dashboard",
    height: int = 600,
    show_distributions: bool = False,
) -> go.Figure:
    """Create interactive Plotly dashboard with WSJ styling.

    Args:
        results: Dictionary with simulation results or DataFrame
        title: Dashboard title
        height: Dashboard height in pixels
        show_distributions: Whether to show distribution plots

    Returns:
        Plotly figure
    """
    # Handle DataFrame input
    if isinstance(results, pd.DataFrame):
        # Convert DataFrame to dictionary format expected by dashboard
        results_dict = {
            "data": results,
            "summary": {
                "mean_assets": (
                    results.get("assets", pd.Series()).mean() if "assets" in results.columns else 0
                ),
                "mean_losses": (
                    results.get("losses", pd.Series()).mean() if "losses" in results.columns else 0
                ),
                "years": results["year"].nunique() if "year" in results.columns else 1,
            },
        }
        results = results_dict
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
        losses_data = np.asarray(results["losses"])
        sorted_losses = np.sort(losses_data)[::-1]
        exceedance_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

        fig.add_trace(
            go.Scatter(
                x=sorted_losses / 1e6,
                y=exceedance_prob,
                mode="lines",
                line={"color": WSJ_COLORS["red"], "width": 2},
                name="Exceedance",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Loss Amount ($M)", row=1, col=2)
        fig.update_yaxes(title_text="Exceedance Probability", type="log", row=1, col=2)

    # Convergence diagnostics
    if "convergence" in results and isinstance(results["convergence"], dict):
        iterations = results["convergence"].get("iterations", [])
        r_hat = results["convergence"].get("r_hat", [])

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=r_hat,
                mode="lines+markers",
                line={"color": WSJ_COLORS["green"], "width": 2},
                marker={"size": 6},
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
    if "metrics" in results and isinstance(results["metrics"], dict):
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
    fig.update_layout(title_text=title, showlegend=False, height=height, **layout_theme)

    # Update all axes
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=WSJ_COLORS["light_gray"])
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=WSJ_COLORS["light_gray"])

    return fig


def plot_convergence_diagnostics(  # pylint: disable=too-many-locals
    convergence_stats: Dict[str, Any],
    title: str = "Convergence Diagnostics",
    figsize: Tuple[int, int] = (12, 8),
    r_hat_threshold: float = 1.1,
    show_threshold: bool = False,
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

    Args:
        frontier_points: List of ParetoPoint objects
        x_objective: Name of objective for x-axis
        y_objective: Name of objective for y-axis
        x_label: Optional custom label for x-axis
        y_label: Optional custom label for y-axis
        title: Plot title
        highlight_knees: Whether to highlight knee points
        show_trade_offs: Whether to show trade-off annotations
        figsize: Figure size

    Returns:
        Matplotlib figure
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

    Args:
        frontier_points: List of ParetoPoint objects
        x_objective: Name of objective for x-axis
        y_objective: Name of objective for y-axis
        z_objective: Name of objective for z-axis
        x_label: Optional custom label for x-axis
        y_label: Optional custom label for y-axis
        z_label: Optional custom label for z-axis
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
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

    Args:
        frontier_points: List of ParetoPoint objects
        objectives: List of objective names to display
        title: Plot title
        height: Plot height in pixels
        show_dominated: Whether to show dominated region

    Returns:
        Plotly figure
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


# Scenario Batch Processing Visualizations


def plot_scenario_comparison(  # pylint: disable=too-many-locals
    aggregated_results: Any,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Create comprehensive scenario comparison visualization.

    Args:
        aggregated_results: AggregatedResults object from batch processing
        metrics: List of metrics to compare (default: key metrics)
        figsize: Figure size (width, height)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from .batch_processor import AggregatedResults

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

    Args:
        aggregated_results: AggregatedResults with sensitivity analysis
        metric: Metric to visualize
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from .batch_processor import AggregatedResults

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

    Args:
        aggregated_results: AggregatedResults from grid search
        param1: First parameter name
        param2: Second parameter name
        metric: Metric to plot on z-axis
        height: Figure height in pixels
        save_path: Path to save figure

    Returns:
        Plotly figure
    """
    from .batch_processor import AggregatedResults

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

    Args:
        batch_results: List of BatchResult objects
        metric: Metric to track
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
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

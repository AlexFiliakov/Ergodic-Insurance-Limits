"""Executive-level visualization functions.

This module provides high-level visualization functions for executive reporting
including loss distributions, return period curves, and insurance layer diagrams.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .core import COLOR_SEQUENCE, WSJ_COLORS, WSJFormatter, format_currency, set_wsj_style


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
    theme: Optional[Any] = None,
) -> Figure:
    """Create WSJ-style loss distribution plot.

    Creates a two-panel visualization showing loss distribution histogram
    and Q-Q plot for normality assessment.

    Args:
        losses: Array of loss values or DataFrame with 'amount' column
        title: Plot title
        bins: Number of histogram bins
        show_metrics: Whether to show VaR/TVaR lines
        var_levels: VaR confidence levels to show (default: [0.95, 0.99])
        figsize: Figure size (width, height)
        show_stats: Whether to show statistics
        log_scale: Whether to use log scale
        use_factory: Whether to use new visualization factory if available
        theme: Optional theme to use with factory

    Returns:
        Matplotlib figure with distribution plots

    Examples:
        >>> losses = np.random.lognormal(10, 2, 1000)
        >>> fig = plot_loss_distribution(losses, title="Annual Loss Distribution")
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
        from ..risk_metrics import RiskMetrics

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

    Visualizes the relationship between return periods and loss magnitudes,
    commonly used in catastrophe modeling and risk assessment.

    Args:
        losses: Loss amounts (array or DataFrame)
        return_periods: Array of return periods (years), optional
        scenarios: Optional dict of scenario names to loss arrays
        title: Plot title
        figsize: Figure size (width, height)
        confidence_level: Confidence level for bands
        show_grid: Whether to show grid

    Returns:
        Matplotlib figure with return period curve

    Examples:
        >>> losses = np.random.lognormal(10, 2, 1000)
        >>> fig = plot_return_period_curve(losses)
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

    Visualizes insurance program structure with layer attachments, limits,
    and premium distribution in a two-panel display.

    Args:
        layers: List of layer dictionaries or DataFrame with 'attachment', 'limit' columns
        total_limit: Total program limit (calculated from layers if not provided)
        title: Plot title
        figsize: Figure size (width, height)
        losses: Optional loss data for overlay
        loss_data: Alias for losses parameter
        show_expected_loss: Whether to show expected loss line

    Returns:
        Matplotlib figure with layer structure and premium distribution

    Examples:
        >>> layers = [
        ...     {"attachment": 0, "limit": 5e6, "premium": 0.015},
        ...     {"attachment": 5e6, "limit": 10e6, "premium": 0.008},
        ... ]
        >>> fig = plot_insurance_layers(layers)
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

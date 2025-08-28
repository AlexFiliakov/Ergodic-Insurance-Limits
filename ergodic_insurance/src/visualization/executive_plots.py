"""Executive-level visualization functions.

This module provides high-level visualization functions for executive reporting
including loss distributions, return period curves, and insurance layer diagrams.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import argrelextrema

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


def plot_roe_ruin_frontier(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    results: Union[Dict[float, pd.DataFrame], pd.DataFrame],
    company_sizes: Optional[List[float]] = None,
    title: str = "ROE-Ruin Efficient Frontier",
    figsize: Tuple[int, int] = (12, 8),
    highlight_sweet_spots: bool = True,
    show_optimal_zones: bool = True,
    export_dpi: Optional[int] = None,
    log_scale_y: bool = True,
    grid: bool = True,
    annotations: bool = True,
    color_scheme: Optional[List[str]] = None,
) -> Figure:
    """Create ROE-Ruin efficient frontier visualization.

    Visualizes the Pareto frontier showing trade-offs between Return on Equity (ROE)
    and Ruin Probability for different company sizes. This helps executives understand
    optimal insurance purchasing decisions.

    Args:
        results: Either a dict of company_size (float) -> optimization results DataFrame,
                or a single DataFrame with 'company_size' column
        company_sizes: List of company sizes to plot (e.g., [1e6, 1e7, 1e8])
                      If None, will use all available sizes in results
        title: Plot title
        figsize: Figure size (width, height)
        highlight_sweet_spots: Whether to highlight knee points on curves
        show_optimal_zones: Whether to show shaded optimal zones
        export_dpi: DPI for export (150 for web, 300 for print, None for screen)
        log_scale_y: Whether to use log scale for ruin probability axis
        grid: Whether to show grid lines
        annotations: Whether to show annotations for key points
        color_scheme: List of colors for different company sizes

    Returns:
        Matplotlib figure with ROE-Ruin efficient frontier plots

    Raises:
        ValueError: If results format is invalid or no data available

    Examples:
        >>> # With dictionary of results
        >>> results = {
        ...     1e6: pd.DataFrame({'roe': [0.1, 0.15, 0.2],
        ...                       'ruin_prob': [0.05, 0.02, 0.01]}),
        ...     1e7: pd.DataFrame({'roe': [0.08, 0.12, 0.18],
        ...                       'ruin_prob': [0.03, 0.015, 0.008]})
        ... }
        >>> fig = plot_roe_ruin_frontier(results)

        >>> # With single DataFrame
        >>> df = pd.DataFrame({
        ...     'company_size': [1e6, 1e6, 1e7, 1e7],
        ...     'roe': [0.1, 0.15, 0.08, 0.12],
        ...     'ruin_prob': [0.05, 0.02, 0.03, 0.015]
        ... })
        >>> fig = plot_roe_ruin_frontier(df)
    """
    set_wsj_style()

    # Process input data
    data_dict = {}

    if isinstance(results, pd.DataFrame):
        # Single DataFrame with company_size column
        if "company_size" not in results.columns:
            raise ValueError("DataFrame must have 'company_size' column")

        for size in results["company_size"].unique():
            size_data = results[results["company_size"] == size].copy()
            data_dict[size] = size_data
    elif isinstance(results, dict):
        # Dictionary of company_size -> DataFrame
        data_dict = results
    else:
        raise ValueError("Results must be DataFrame or dict of DataFrames")

    # Determine company sizes to plot
    if company_sizes is None:
        company_sizes = sorted(data_dict.keys())
    else:
        # Filter to requested sizes that exist in data
        company_sizes = [s for s in company_sizes if s in data_dict]

    if not company_sizes:
        raise ValueError("No valid company sizes found in data")

    # Set up color scheme
    if color_scheme is None:
        # Default corporate blues with good contrast
        color_scheme = [
            WSJ_COLORS["blue"],  # Primary blue
            "#4A90E2",  # Lighter blue
            "#1E3A8A",  # Darker blue
            WSJ_COLORS["orange"],  # Accent for contrast if > 3 sizes
            WSJ_COLORS["red"],  # Additional contrast
        ]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Storage for sweet spots
    sweet_spots = []

    # Plot each company size
    for idx, size in enumerate(company_sizes):
        df = data_dict[size]

        # Ensure required columns exist
        required_cols = ["roe", "ruin_prob"]
        if not all(col in df.columns for col in required_cols):
            # Try alternative column names
            roe_col = next(
                (c for c in df.columns if "roe" in c.lower() or "return" in c.lower()), None
            )
            ruin_col = next((c for c in df.columns if "ruin" in c.lower()), None)

            if roe_col and ruin_col:
                df = df.rename(columns={roe_col: "roe", ruin_col: "ruin_prob"})
            else:
                raise ValueError(f"Company size {size} missing ROE or ruin probability data")

        # Sort by ROE for proper curve drawing
        df = df.sort_values("roe")

        # Extract data
        roe_values = df["roe"].values * 100  # Convert to percentage
        ruin_values = df["ruin_prob"].values * 100  # Convert to percentage

        # Apply smoothing if enough points
        if len(roe_values) > 3:
            try:
                # Create smooth curve using spline interpolation
                x_smooth = np.linspace(roe_values.min(), roe_values.max(), 100)
                spline = make_interp_spline(roe_values, ruin_values, k=min(3, len(roe_values) - 1))
                y_smooth = spline(x_smooth)
                y_smooth = np.clip(y_smooth, 0, 100)  # Ensure values stay in valid range
            except Exception:  # pylint: disable=broad-except
                # Fallback to linear interpolation on error
                x_smooth = roe_values
                y_smooth = ruin_values
        else:
            x_smooth = roe_values
            y_smooth = ruin_values

        # Plot the frontier curve
        color = color_scheme[idx % len(color_scheme)]
        label = f"${format_currency(size, decimals=0).replace('$', '')} Company"

        ax.plot(x_smooth, y_smooth, "-", color=color, linewidth=2.5, label=label)
        ax.scatter(roe_values, ruin_values, color=color, s=50, alpha=0.6, zorder=5)

        # Find and highlight sweet spot (knee point)
        if highlight_sweet_spots and len(roe_values) > 2:
            sweet_spot_idx = _find_knee_point(roe_values, ruin_values)
            sweet_roe = roe_values[sweet_spot_idx]
            sweet_ruin = ruin_values[sweet_spot_idx]
            sweet_spots.append((sweet_roe, sweet_ruin, size))

            # Mark sweet spot
            ax.scatter(
                sweet_roe,
                sweet_ruin,
                color=color,
                s=200,
                marker="*",
                edgecolor="white",
                linewidth=2,
                zorder=10,
            )

            # Add annotation if enabled
            if annotations:
                ax.annotate(
                    f"Sweet Spot\n{sweet_roe:.1f}% ROE\n{sweet_ruin:.2f}% Risk",
                    xy=(sweet_roe, sweet_ruin),
                    xytext=(sweet_roe + 2, sweet_ruin * 1.5),
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.5, "lw": 1},
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": color,
                        "alpha": 0.8,
                    },
                )

    # Add optimal zones if requested
    if show_optimal_zones and sweet_spots:
        # Calculate optimal zone bounds based on sweet spots
        roe_range = [min(s[0] for s in sweet_spots) - 2, max(s[0] for s in sweet_spots) + 2]
        ruin_range = [min(s[1] for s in sweet_spots) * 0.5, max(s[1] for s in sweet_spots) * 2]

        # Add shaded optimal zone
        ax.axvspan(roe_range[0], roe_range[1], alpha=0.05, color="green", label="Optimal Zone")
        ax.axhspan(ruin_range[0], ruin_range[1], alpha=0.05, color="green")

    # Format axes
    ax.set_xlabel("Return on Equity (%)", fontsize=12)
    ax.set_ylabel("Ruin Probability (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Apply log scale to y-axis if requested
    if log_scale_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.2f}%"))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}%"))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))

    # Add grid if requested
    if grid:
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

    # Add legend
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor=WSJ_COLORS["gray"])

    # Set axis limits for better visualization
    ax.set_xlim(left=0)
    if not log_scale_y:
        ax.set_ylim(bottom=0)

    # Adjust layout
    plt.tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_ruin_cliff(  # pylint: disable=too-many-locals,too-many-statements
    retention_range: Optional[Tuple[float, float]] = None,
    n_points: int = 50,
    company_size: float = 10_000_000,
    simulation_data: Optional[Dict[str, Any]] = None,
    title: str = "The Ruin Cliff: Retention vs Failure Risk",
    figsize: Tuple[int, int] = (14, 8),
    show_inset: bool = True,
    show_warnings: bool = True,
    show_3d_effect: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create dramatic ruin cliff visualization with 3D effects.

    Visualizes the relationship between insurance retention (deductible) levels
    and ruin probability, highlighting the "cliff edge" where risk dramatically
    increases. Features 3D-style gradient effects and warning zones.

    Args:
        retention_range: Tuple of (min, max) retention values in dollars.
                        Default: (10_000, 10_000_000)
        n_points: Number of points to calculate along retention axis
        company_size: Company asset size for scaling retention levels
        simulation_data: Optional pre-computed simulation results with keys:
                        'retentions', 'ruin_probs', 'confidence_intervals'
        title: Plot title
        figsize: Figure size (width, height)
        show_inset: Whether to show zoomed inset of critical region
        show_warnings: Whether to show warning callouts and annotations
        show_3d_effect: Whether to add 3D gradient background effects
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with ruin cliff visualization

    Examples:
        >>> # Basic usage with synthetic data
        >>> fig = plot_ruin_cliff()

        >>> # With custom retention range
        >>> fig = plot_ruin_cliff(retention_range=(5000, 5_000_000))

        >>> # With pre-computed simulation data
        >>> data = {
        ...     'retentions': np.logspace(4, 7, 50),
        ...     'ruin_probs': np.array([...]),
        ... }
        >>> fig = plot_ruin_cliff(simulation_data=data)

    Notes:
        The visualization uses a log scale for retention values to show
        the full range from small to large deductibles. The cliff edge
        is detected using derivative analysis to find the steepest point
        of increase in ruin probability.
    """
    set_wsj_style()

    # Set default retention range if not provided
    if retention_range is None:
        retention_range = (10_000, 10_000_000)

    # Generate or use provided data
    if simulation_data is not None:
        retentions = simulation_data["retentions"]
        ruin_probs = simulation_data["ruin_probs"]
    else:
        # Generate synthetic demonstration data
        retentions = np.logspace(
            np.log10(retention_range[0]), np.log10(retention_range[1]), n_points
        )

        # Create realistic ruin probability curve with cliff effect
        # Low retention = high ruin probability, high retention = low probability
        # but with a steep cliff in the middle
        log_ret = np.log10(retentions)
        log_ret_norm = (log_ret - log_ret.min()) / (log_ret.max() - log_ret.min())

        # Sigmoid-like function with adjustable steepness for cliff effect
        cliff_center = 0.3  # Position of cliff (30% along log scale)
        cliff_steepness = 15  # How steep the cliff is
        base_curve = 1 / (1 + np.exp(cliff_steepness * (log_ret_norm - cliff_center)))

        # Add some noise and ensure realistic bounds
        noise = np.random.RandomState(42).normal(0, 0.01, len(retentions))
        ruin_probs = np.clip(base_curve + noise, 0.001, 0.99)

    # Find cliff edge (steepest point)
    derivatives = np.gradient(ruin_probs, np.log10(retentions))
    cliff_idx = np.argmax(np.abs(derivatives))
    cliff_retention = retentions[cliff_idx]
    cliff_ruin = ruin_probs[cliff_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Add 3D gradient effect background if requested
    if show_3d_effect:
        # Create gradient mesh for contour fill
        x_mesh = np.logspace(np.log10(retentions.min()), np.log10(retentions.max()), 100)
        y_mesh = np.linspace(0, 1, 100)
        X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)

        # Create Z values based on distance from the curve
        Z_mesh = np.zeros_like(X_mesh)
        for i, x_val in enumerate(x_mesh):
            # Interpolate ruin probability at this retention
            interp_ruin = np.interp(x_val, retentions, ruin_probs)
            for j, y_val in enumerate(y_mesh):
                # Distance from the curve determines intensity
                dist = abs(y_val - interp_ruin)
                Z_mesh[j, i] = 1 - dist

        # Create custom colormap for 3D effect
        from matplotlib.colors import LinearSegmentedColormap

        colors_3d = ["#ffffff", "#e8f4fd", "#c5e4fd", "#7ec0ee", "#4a90e2", "#ff6b6b"]
        n_bins = 100
        cmap_3d = LinearSegmentedColormap.from_list("ruin_cliff", colors_3d, N=n_bins)

        # Add gradient background
        contour = ax.contourf(
            X_mesh, Y_mesh, Z_mesh, levels=20, cmap=cmap_3d, alpha=0.3, antialiased=True
        )

    # Define color zones based on ruin probability
    danger_threshold = 0.05  # 5% ruin probability
    warning_threshold = 0.02  # 2% ruin probability

    # Add danger zones
    ax.axhspan(danger_threshold, 1.0, alpha=0.1, color="red", label="Danger Zone (>5% risk)")
    ax.axhspan(
        warning_threshold,
        danger_threshold,
        alpha=0.1,
        color="orange",
        label="Warning Zone (2-5% risk)",
    )
    ax.axhspan(0, warning_threshold, alpha=0.1, color="green", label="Safe Zone (<2% risk)")

    # Plot main curve with color gradient based on risk level
    for i in range(len(retentions) - 1):
        # Determine color based on ruin probability
        if ruin_probs[i] > danger_threshold:
            color = "#ff4444"  # Red
        elif ruin_probs[i] > warning_threshold:
            color = "#ff8800"  # Orange
        else:
            color = "#00aa00"  # Green

        ax.plot(retentions[i : i + 2], ruin_probs[i : i + 2], color=color, linewidth=3, alpha=0.9)

    # Mark the cliff edge
    ax.scatter(
        cliff_retention,
        cliff_ruin,
        s=300,
        color="red",
        marker="o",
        edgecolor="darkred",
        linewidth=3,
        zorder=10,
        label=f"Cliff Edge: ${cliff_retention:,.0f}",
    )

    # Add warning callouts if requested
    if show_warnings:
        # Cliff edge warning
        ax.annotate(
            f"⚠️ CLIFF EDGE\n${cliff_retention:,.0f} retention\n{cliff_ruin:.1%} ruin risk",
            xy=(cliff_retention, cliff_ruin),
            xytext=(cliff_retention * 3, cliff_ruin + 0.15),
            fontsize=11,
            fontweight="bold",
            color="darkred",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="yellow",
                edgecolor="red",
                alpha=0.9,
                linewidth=2,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.3",
                color="red",
                linewidth=2,
                shrinkA=5,
                shrinkB=5,
            ),
        )

        # Safe zone indicator
        safe_idx = np.where(ruin_probs < warning_threshold)[0]
        if len(safe_idx) > 0:
            safe_retention = retentions[safe_idx[0]]
            ax.annotate(
                f"✓ Safe Zone\nRetention > ${safe_retention:,.0f}",
                xy=(safe_retention * 2, 0.01),
                fontsize=10,
                color="darkgreen",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor="green", alpha=0.8
                ),
            )

    # Add inset plot for critical region if requested
    if show_inset:
        # Define inset region around cliff
        inset_range = (cliff_retention * 0.3, cliff_retention * 3)
        inset_mask = (retentions >= inset_range[0]) & (retentions <= inset_range[1])

        if np.any(inset_mask):
            # Create inset axes
            ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=3)

            # Plot zoomed region
            ax_inset.semilogx(retentions[inset_mask], ruin_probs[inset_mask], "b-", linewidth=2)
            ax_inset.scatter(
                cliff_retention,
                cliff_ruin,
                s=100,
                color="red",
                marker="o",
                edgecolor="darkred",
                linewidth=2,
            )

            # Style inset
            ax_inset.set_xlabel("Retention ($)", fontsize=9)
            ax_inset.set_ylabel("Ruin Probability", fontsize=9)
            ax_inset.set_title("Critical Region Detail", fontsize=10, fontweight="bold")
            ax_inset.grid(True, alpha=0.3)
            ax_inset.xaxis.set_major_formatter(
                mticker.FuncFormatter(WSJFormatter.currency_formatter)
            )
            ax_inset.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

            # Add shading to inset
            ax_inset.axhspan(danger_threshold, 1.0, alpha=0.2, color="red")
            ax_inset.axhspan(warning_threshold, danger_threshold, alpha=0.2, color="orange")

    # Format main axes
    ax.set_xscale("log")
    ax.set_xlabel("Retention Level (Deductible)", fontsize=13, fontweight="bold")
    ax.set_ylabel("10-Year Ruin Probability", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Format axes
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(WSJFormatter.currency_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0%}"))

    # Set sensible limits
    ax.set_xlim(retentions.min() * 0.8, retentions.max() * 1.2)
    ax.set_ylim(0, min(1.0, ruin_probs.max() * 1.1))

    # Add grid
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.grid(True, which="minor", alpha=0.1, linestyle=":")

    # Add legend
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.95,
        fontsize=10,
    )

    # Add subtitle with company info
    fig.text(
        0.5,
        0.94,
        f"Company Size: {format_currency(company_size)} | Analysis: 10-Year Time Horizon",
        ha="center",
        fontsize=11,
        style="italic",
        color=WSJ_COLORS["gray"],
    )

    # Adjust layout
    plt.tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def _find_knee_point(x: np.ndarray, y: np.ndarray) -> int:
    """Find the knee point (elbow) in a curve using curvature analysis.

    Args:
        x: X-axis values (ROE)
        y: Y-axis values (Ruin probability)

    Returns:
        Index of the knee point
    """
    # Normalize data
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    # Calculate distances from each point to the line between start and end
    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])

    distances = []
    for x_val, y_val in zip(x_norm, y_norm):
        point = np.array([x_val, y_val])
        # Distance from point to line using 2D cross product formula
        # For 2D vectors, cross product gives scalar: (a × b) = ax*by - ay*bx
        line_vec = end - start
        point_vec = point - start
        cross_prod = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
        dist = np.abs(cross_prod) / np.linalg.norm(line_vec)
        distances.append(dist)

    # The knee point is where distance is maximum
    knee_idx = np.argmax(distances)

    # Refine by looking for the point where curvature changes most
    if len(x) > 5:
        # Calculate curvature using second derivative approximation
        curvature = np.abs(np.gradient(np.gradient(y_norm)))
        # Find local maxima in curvature
        maxima = argrelextrema(curvature, np.greater)[0]
        if len(maxima) > 0:
            # Choose the maximum closest to our initial estimate
            distances_to_knee = np.abs(maxima - knee_idx)
            knee_idx = maxima[np.argmin(distances_to_knee)]

    return int(knee_idx)

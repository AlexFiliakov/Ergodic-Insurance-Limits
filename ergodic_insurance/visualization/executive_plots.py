"""Executive-level visualization functions.

This module provides high-level visualization functions for executive reporting
including loss distributions, return period curves, and insurance layer diagrams.
"""

# pylint: disable=too-many-lines

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import argrelextrema

from .core import COLOR_SEQUENCE, WSJ_COLORS, WSJFormatter, format_currency, set_wsj_style


def safe_tight_layout():
    """Apply tight_layout with warning suppression."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        try:
            plt.tight_layout()
        except (ValueError, RuntimeError):
            # If tight_layout fails, continue without it
            pass


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
        safe_tight_layout()
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
    safe_tight_layout()

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

    safe_tight_layout()
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
            safe_tight_layout()
            return fig

        # Convert DataFrame to list of dicts
        layer_list = []
        for _, row in layers.iterrows():
            layer_dict = {
                "attachment": row.get("attachment", 0),
                "limit": row.get("limit", 0),
                "premium": row.get(
                    "base_premium_rate", row.get("premium_rate", row.get("premium", 0))
                ),
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
        safe_tight_layout()
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
    safe_tight_layout()

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
    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_ruin_cliff(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
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
        noise = np.random.default_rng(42).normal(0, 0.01, len(retentions))
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
            f"[!] CLIFF EDGE\n${cliff_retention:,.0f} retention\n{cliff_ruin:.1%} ruin risk",
            xy=(cliff_retention, cliff_ruin),
            xytext=(cliff_retention * 3, cliff_ruin + 0.15),
            fontsize=11,
            fontweight="bold",
            color="darkred",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "yellow",
                "edgecolor": "red",
                "alpha": 0.9,
                "linewidth": 2,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "connectionstyle": "arc3,rad=0.3",
                "color": "red",
                "linewidth": 2,
                "shrinkA": 5,
                "shrinkB": 5,
            },
        )

        # Safe zone indicator
        safe_idx = np.where(ruin_probs < warning_threshold)[0]
        if len(safe_idx) > 0:
            safe_retention = retentions[safe_idx[0]]
            ax.annotate(
                f"[OK] Safe Zone\nRetention > ${safe_retention:,.0f}",
                xy=(safe_retention * 2, 0.01),
                fontsize=10,
                color="darkgreen",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "lightgreen",
                    "edgecolor": "green",
                    "alpha": 0.8,
                },
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
    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_simulation_architecture(  # pylint: disable=too-many-locals
    title: str = "Simulation Architecture Flow",
    figsize: Tuple[int, int] = (14, 8),
    export_dpi: Optional[int] = None,
    show_icons: bool = True,
) -> Figure:
    """Create simulation architecture flow diagram.

    Visualizes the data flow from parameters through simulation to insights
    using a clean flowchart style with boxes and arrows.

    Args:
        title: Plot title
        figsize: Figure size (width, height)
        export_dpi: DPI for export (150 for web, 300 for print)
        show_icons: Whether to show icons in boxes

    Returns:
        Matplotlib figure with architecture diagram

    Examples:
        >>> fig = plot_simulation_architecture()
        >>> fig.savefig("architecture.png", dpi=150)
    """
    set_wsj_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Define box positions and sizes
    box_width = 0.15
    box_height = 0.12
    arrow_width = 0.02

    # Main flow boxes (left to right)
    boxes = [
        {"x": 0.1, "y": 0.5, "label": "Parameters\n& Config", "color": WSJ_COLORS["blue"]},
        {"x": 0.35, "y": 0.5, "label": "Monte Carlo\nSimulation", "color": WSJ_COLORS["orange"]},
        {"x": 0.6, "y": 0.5, "label": "Analysis\nEngine", "color": WSJ_COLORS["red"]},
        {"x": 0.85, "y": 0.5, "label": "Insights &\nDecisions", "color": WSJ_COLORS["green"]},
    ]

    # Sub-component boxes
    sub_boxes = [
        {
            "x": 0.1,
            "y": 0.75,
            "label": "Company\nProfile",
            "color": WSJ_COLORS["gray"],
            "size": 0.8,
        },
        {
            "x": 0.1,
            "y": 0.25,
            "label": "Insurance\nProgram",
            "color": WSJ_COLORS["gray"],
            "size": 0.8,
        },
        {
            "x": 0.35,
            "y": 0.75,
            "label": "Loss\nGeneration",
            "color": WSJ_COLORS["gray"],
            "size": 0.8,
        },
        {
            "x": 0.35,
            "y": 0.25,
            "label": "Financial\nDynamics",
            "color": WSJ_COLORS["gray"],
            "size": 0.8,
        },
        {
            "x": 0.6,
            "y": 0.75,
            "label": "Ergodic\nCalculations",
            "color": WSJ_COLORS["gray"],
            "size": 0.8,
        },
        {"x": 0.6, "y": 0.25, "label": "Risk\nMetrics", "color": WSJ_COLORS["gray"], "size": 0.8},
    ]

    # Draw main boxes
    for box in boxes:
        rect = plt.Rectangle(
            (float(box["x"]) - box_width / 2, float(box["y"]) - box_height / 2),  # type: ignore[arg-type]
            box_width,
            box_height,
            facecolor=box["color"],
            alpha=0.3,
            edgecolor=box["color"],
            linewidth=2,
        )
        ax.add_patch(rect)

        # Add text
        ax.text(
            float(box["x"]),  # type: ignore[arg-type]
            float(box["y"]),  # type: ignore[arg-type]
            str(box["label"]),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=box["color"],
        )

    # Draw sub-component boxes
    for box in sub_boxes:
        size_factor = box.get("size", 1.0)
        rect = plt.Rectangle(
            (
                float(box["x"]) - box_width * float(size_factor) * 0.4,  # type: ignore[arg-type]
                float(box["y"]) - box_height * float(size_factor) * 0.4,  # type: ignore[arg-type]
            ),
            box_width * float(size_factor) * 0.8,  # type: ignore[arg-type]
            box_height * float(size_factor) * 0.8,  # type: ignore[arg-type]
            facecolor="white",
            alpha=0.8,
            edgecolor=box["color"],
            linewidth=1,
            linestyle="--",
        )
        ax.add_patch(rect)

        ax.text(
            float(box["x"]),  # type: ignore[arg-type]
            float(box["y"]),  # type: ignore[arg-type]
            str(box["label"]),
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color=box["color"],
        )

    # Draw arrows between main boxes
    arrow_props = {
        "arrowstyle": "-|>",
        "connectionstyle": "arc3,rad=0",
        "color": WSJ_COLORS["gray"],
        "linewidth": 2,
        "alpha": 0.7,
    }

    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(float(boxes[i + 1]["x"]) - box_width / 2 - 0.01, float(boxes[i + 1]["y"])),  # type: ignore[arg-type]
            xytext=(float(boxes[i]["x"]) + box_width / 2 + 0.01, float(boxes[i]["y"])),  # type: ignore[arg-type]
            arrowprops=arrow_props,
        )

    # Draw connecting arrows from sub-components
    thin_arrow_props = {
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=0",  # Straight arrows, no curve
        "color": WSJ_COLORS["gray"],
        "linewidth": 1,
        "alpha": 0.5,
    }

    # Connect sub-boxes to their parent main boxes (downward arrows)
    sub_connections = [
        (0.1, 0.75, 0.1, 0.5 + box_height / 2),  # Company Profile -> Parameters
        (0.1, 0.25, 0.1, 0.5 - box_height / 2),  # Insurance Program -> Parameters
        (0.35, 0.75, 0.35, 0.5 + box_height / 2),  # Loss Generation -> Monte Carlo
        (0.35, 0.25, 0.35, 0.5 - box_height / 2),  # Financial Dynamics -> Monte Carlo
        (0.6, 0.75, 0.6, 0.5 + box_height / 2),  # Ergodic Calculations -> Analysis
        (0.6, 0.25, 0.6, 0.5 - box_height / 2),  # Risk Metrics -> Analysis
    ]

    for x1, y1, x2, y2 in sub_connections:
        # Adjust arrow direction based on position
        if y1 > 0.5:  # Box is above main box
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1 - box_height * 0.32),
                arrowprops=thin_arrow_props,
            )
        else:  # Box is below main box
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1 + box_height * 0.32),
                arrowprops=thin_arrow_props,
            )

    # Add title at the top
    ax.text(
        0.5,
        0.95,
        title,
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=WSJ_COLORS["black"],
    )

    # Add annotation at the bottom
    ax.text(
        0.5,
        0.05,
        "Each stage transforms data to extract actionable business insights",
        ha="center",
        fontsize=10,
        color=WSJ_COLORS["gray"],
    )

    # Style the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_sample_paths(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    simulation_data: Optional[Dict[str, Any]] = None,
    n_paths: int = 5,
    short_horizon: int = 10,
    long_horizon: int = 100,
    company_size: float = 10_000_000,
    title: str = "Sample Path Visualization",
    figsize: Tuple[int, int] = (14, 8),
    show_failures: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create sample path visualization showing trajectory evolution.

    Displays representative paths over short and long time horizons,
    highlighting survivors vs failed companies with transparency effects.

    Args:
        simulation_data: Optional pre-computed simulation results with paths
        n_paths: Number of paths to display (default 5)
        short_horizon: Years for short-term view (default 10)
        long_horizon: Years for long-term view (default 100)
        company_size: Starting company size
        title: Plot title
        figsize: Figure size (width, height)
        show_failures: Whether to highlight failed paths
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with dual-panel path visualization

    Examples:
        >>> fig = plot_sample_paths(n_paths=5)
        >>> fig.savefig("sample_paths.png", dpi=150)
    """
    set_wsj_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Generate or use provided data
    if simulation_data is not None:
        paths_short = simulation_data.get("paths_short", None)
        paths_long = simulation_data.get("paths_long", None)
    else:
        # Generate synthetic demonstration paths
        rng = np.random.default_rng(42)

        # Short horizon paths
        time_short = np.linspace(0, short_horizon, 100)
        paths_short = []

        for i in range(n_paths):
            # Generate path with random walk and drift
            drift = rng.normal(0.08, 0.02)  # Annual growth
            volatility = rng.uniform(0.15, 0.25)
            shocks = rng.normal(0, volatility, len(time_short))
            cumulative = np.cumsum(drift / len(time_short) + shocks / np.sqrt(len(time_short)))
            path = company_size * np.exp(cumulative)

            # Randomly determine if path fails
            fails = rng.random() < 0.2  # 20% failure rate
            if fails and show_failures:
                fail_time = rng.uniform(short_horizon * 0.3, short_horizon * 0.9)
                fail_idx = int(fail_time / short_horizon * len(time_short))
                path[fail_idx:] = path[fail_idx] * np.exp(-np.arange(len(path) - fail_idx) * 0.1)

            paths_short.append({"values": path, "failed": fails})

        # Long horizon paths
        time_long = np.linspace(0, long_horizon, 500)
        paths_long = []

        for i in range(n_paths):
            drift = rng.normal(0.08, 0.02)
            volatility = rng.uniform(0.15, 0.30)
            shocks = rng.normal(0, volatility, len(time_long))
            cumulative = np.cumsum(drift / len(time_long) + shocks / np.sqrt(len(time_long)))
            path = company_size * np.exp(cumulative)

            fails = rng.random() < 0.3  # 30% failure rate long-term
            if fails and show_failures:
                fail_time = rng.uniform(long_horizon * 0.2, long_horizon * 0.8)
                fail_idx = int(fail_time / long_horizon * len(time_long))
                path[fail_idx:] = path[fail_idx] * np.exp(-np.arange(len(path) - fail_idx) * 0.05)

            paths_long.append({"values": path, "failed": fails})

    # Plot short horizon
    if paths_short is not None:
        for i, path_data in enumerate(paths_short):
            path = path_data["values"]
            failed = path_data.get("failed", False)

            if failed:
                color = WSJ_COLORS["red"]
                alpha = 0.7
                linewidth = 1.5
                label = "Failed" if i == 0 else None
            else:
                color = WSJ_COLORS["blue"]
                alpha = 0.8
                linewidth = 2
                label = "Survivor" if i == 0 else None

            ax1.plot(
                (
                    time_short
                    if "time_short" in locals()
                    else np.linspace(0, short_horizon, len(path))
                ),
                path / 1e6,  # Convert to millions
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=label,
            )

    # Add starting point marker
    ax1.scatter(
        0,
        company_size / 1e6,
        s=100,
        color=WSJ_COLORS["green"],
        marker="o",
        zorder=10,
        label=f"Start: ${company_size/1e6:.0f}M",
    )

    ax1.set_xlabel("Years", fontsize=11)
    ax1.set_ylabel("Company Value ($M)", fontsize=11)
    ax1.set_title(f"{short_horizon}-Year Horizon", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:.0f}M"))

    # Plot long horizon
    if paths_long is not None:
        for i, path_data in enumerate(paths_long):
            path = path_data["values"]
            failed = path_data.get("failed", False)

            if failed:
                color = WSJ_COLORS["red"]
                alpha = 0.6
                linewidth = 1.5
            else:
                color = WSJ_COLORS["blue"]
                alpha = 0.7
                linewidth = 2

            ax2.plot(
                time_long if "time_long" in locals() else np.linspace(0, long_horizon, len(path)),
                path / 1e6,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    # Add starting point
    ax2.scatter(0, company_size / 1e6, s=100, color=WSJ_COLORS["green"], marker="o", zorder=10)

    ax2.set_xlabel("Years", fontsize=11)
    ax2.set_ylabel("Company Value ($M)", fontsize=11)
    ax2.set_title(f"{long_horizon}-Year Horizon", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:.0f}M"))

    # Use log scale for long horizon
    ax2.set_yscale("log")

    # Main title
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Add annotation
    fig.text(
        0.5,
        0.02,
        "Trajectories show potential outcomes: survivors grow exponentially, failures decline to zero",
        ha="center",
        fontsize=10,
        style="italic",
        color=WSJ_COLORS["gray"],
    )

    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_optimal_coverage_heatmap(  # pylint: disable=too-many-locals
    optimization_results: Optional[Dict[str, Any]] = None,
    company_sizes: Optional[List[float]] = None,
    title: str = "Optimal Coverage Heatmap",
    figsize: Tuple[int, int] = (16, 6),
    show_contours: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create optimal insurance coverage heatmap for different company sizes.

    Visualizes the relationship between retention, limit, and growth rate
    using color intensity to show optimal configurations.

    Args:
        optimization_results: Optional pre-computed optimization data
        company_sizes: List of company sizes (default: [1e6, 1e7, 1e8])
        title: Plot title
        figsize: Figure size (width, height)
        show_contours: Whether to show contour lines
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with 3-panel heatmap

    Examples:
        >>> fig = plot_optimal_coverage_heatmap(
        ...     company_sizes=[1e6, 1e7, 1e8]
        ... )
        >>> fig.savefig("coverage_heatmap.png", dpi=150)
    """
    set_wsj_style()

    if company_sizes is None:
        company_sizes = [1_000_000, 10_000_000, 100_000_000]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # First pass: collect all data and determine global percentage ranges
    all_growth_rates = []
    all_retention_pcts = []
    all_limit_pcts = []
    data_for_plots = []

    for idx, company_size in enumerate(company_sizes):
        # Generate or use provided data
        if optimization_results is not None:
            data = optimization_results.get(f"company_{company_size}", None)
            if data is not None:
                retention_values = data["retention"]
                limit_values = data["limit"]
                growth_rates = data["growth_rate"]
        else:
            # Generate synthetic demonstration data
            retention_values = np.logspace(4, np.log10(company_size * 0.5), 20)
            limit_values = np.logspace(5, np.log10(company_size * 5), 20)

            # Create mesh grid
            R, L = np.meshgrid(retention_values, limit_values)

            # Synthetic growth rate function
            # Higher growth with moderate retention and appropriate limits
            optimal_retention = company_size * 0.01  # 1% of company size
            optimal_limit = company_size * 0.5  # 50% of company size

            growth_rates = 0.12 * np.exp(
                -0.5
                * (
                    (np.log10(R) - np.log10(optimal_retention)) ** 2 / 1.5
                    + (np.log10(L) - np.log10(optimal_limit)) ** 2 / 2.0
                )
            )

            # Add some noise
            growth_rates += np.random.default_rng(42 + idx).normal(0, 0.005, growth_rates.shape)

        # Convert to percentages
        retention_pcts = retention_values / company_size * 100  # As percentage
        limit_pcts = limit_values / company_size * 100  # As percentage

        # Store data for second pass
        data_for_plots.append(
            {
                "retention_pcts": retention_pcts,
                "limit_pcts": limit_pcts,
                "growth_rates": growth_rates,
                "company_size": company_size,
            }
        )

        # Collect all data for unified scales
        all_growth_rates.append(growth_rates * 100)  # Convert to percentage
        all_retention_pcts.extend(retention_pcts)
        all_limit_pcts.extend(limit_pcts)

    # Determine global ranges for consistent axes
    # Focus on the useful range around optimal configurations
    # Find the range that contains the high-growth regions (top 80% of growth rates)

    # Collect optimal regions for each company size
    optimal_retention_pcts = []
    optimal_limit_pcts = []

    for idx, plot_data in enumerate(data_for_plots):
        growth_rates = plot_data["growth_rates"]
        retention_pcts = plot_data["retention_pcts"]
        limit_pcts = plot_data["limit_pcts"]

        # Find the threshold for "good" growth (e.g., top 25% of growth rates for tighter focus)
        growth_threshold = np.percentile(growth_rates, 75)

        # Find retention and limit ranges where growth is above threshold
        R, L = np.meshgrid(retention_pcts, limit_pcts)
        good_growth_mask = growth_rates >= growth_threshold

        if np.any(good_growth_mask):
            optimal_retention_pcts.extend(R[good_growth_mask])
            optimal_limit_pcts.extend(L[good_growth_mask])

    # Set ranges to cover the optimal regions with some padding
    # Use percentiles to avoid outliers
    if optimal_retention_pcts:
        min_retention_pct = max(0, np.percentile(optimal_retention_pcts, 10) * 0.7)
        max_retention_pct = min(100, np.percentile(optimal_retention_pcts, 90) * 1.3)
    else:
        # Fallback to reasonable defaults if no optimal region found
        min_retention_pct = 0.1  # 0.1%
        max_retention_pct = 20  # 20%

    if optimal_limit_pcts:
        min_limit_pct = max(0, np.percentile(optimal_limit_pcts, 10) * 0.7)
        max_limit_pct = min(1000, np.percentile(optimal_limit_pcts, 90) * 1.3)
    else:
        # Fallback to reasonable defaults
        min_limit_pct = 10  # 10%
        max_limit_pct = 200  # 200%

    # Ensure we have reasonable minimum ranges
    if max_retention_pct - min_retention_pct < 5:
        # Ensure at least 5% range
        max_retention_pct = min_retention_pct + 5

    if max_limit_pct - min_limit_pct < 50:
        # Ensure at least 50% range
        max_limit_pct = min_limit_pct + 50

    # Round to nice numbers for cleaner axes
    min_retention_pct = np.floor(min_retention_pct * 2) / 2  # Round down to nearest 0.5%
    max_retention_pct = np.ceil(max_retention_pct * 2) / 2  # Round up to nearest 0.5%
    min_limit_pct = np.floor(min_limit_pct / 10) * 10  # Round down to nearest 10%
    max_limit_pct = np.ceil(max_limit_pct / 10) * 10  # Round up to nearest 10%

    # Determine global min and max for unified color scale
    vmin = min(np.min(rates) for rates in all_growth_rates)
    vmax = max(np.max(rates) for rates in all_growth_rates)

    # Create unified levels for all plots
    levels = np.linspace(vmin, vmax, 20)

    # Define consistent percentage grid for interpolation
    # Use linear scale for percentages (not log) for clearer interpretation
    common_retention_pcts = np.linspace(min_retention_pct, max_retention_pct, 50)
    common_limit_pcts = np.linspace(min_limit_pct, max_limit_pct, 50)

    # Second pass: create plots with unified scale and axes
    for idx, (ax, plot_data) in enumerate(zip(axes, data_for_plots)):
        retention_pcts = plot_data["retention_pcts"]
        limit_pcts = plot_data["limit_pcts"]
        growth_rates = plot_data["growth_rates"]
        company_size = plot_data["company_size"]

        # Create common mesh grid
        R_common, L_common = np.meshgrid(common_retention_pcts, common_limit_pcts)

        # Interpolate growth rates to common grid
        from scipy.interpolate import griddata

        # Create points from original data
        R_orig, L_orig = np.meshgrid(retention_pcts, limit_pcts)
        points = np.column_stack((R_orig.ravel(), L_orig.ravel()))
        values = (growth_rates * 100).ravel()  # Convert to percentage

        # Interpolate to common grid
        growth_interp = griddata(
            points, values, (R_common, L_common), method="cubic", fill_value=np.nan
        )

        # Create heatmap with unified scale and axes
        im = ax.contourf(
            R_common,
            L_common,
            growth_interp,
            levels=levels,
            cmap="RdYlGn",
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )

        # Add contour lines if requested
        if show_contours:
            contours = ax.contour(
                R_common,
                L_common,
                growth_interp,
                levels=5,
                colors="black",
                alpha=0.3,
                linewidths=1,
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f%%")

        # Find and mark optimal point in original data
        max_idx = np.unravel_index(growth_rates.argmax(), growth_rates.shape)
        ax.scatter(
            retention_pcts[max_idx[1]],
            limit_pcts[max_idx[0]],
            s=200,
            color="white",
            marker="*",
            edgecolor="black",
            linewidth=2,
            zorder=10,
            label="Optimal",
        )

        # Set consistent axes limits for all plots
        ax.set_xlim(min_retention_pct, max_retention_pct)
        ax.set_ylim(min_limit_pct, max_limit_pct)

        # Format axes labels
        ax.set_xlabel("Retention (% of Company Size)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Coverage Limit (% of Company Size)", fontsize=10)

        # Format tick labels as percentages
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))

        # Set consistent tick locations
        x_ticks = np.linspace(0, max_retention_pct, 6)
        y_ticks = np.linspace(0, max_limit_pct, 6)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Add company size label
        ax.set_title(
            f"${format_currency(company_size, decimals=0).replace('$', '')} Company",
            fontsize=11,
            fontweight="bold",
        )

        # Add colorbar only to the rightmost plot
        if idx == 2:  # Only for the last (rightmost) plot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Growth Rate (%)", fontsize=9)

        ax.grid(True, alpha=0.2, which="both")
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_sensitivity_tornado(  # pylint: disable=too-many-locals
    sensitivity_data: Optional[Dict[str, float]] = None,
    baseline_value: Optional[float] = None,
    title: str = "Sensitivity Analysis - Tornado Chart",
    figsize: Tuple[int, int] = (10, 8),
    show_percentages: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create tornado chart showing parameter sensitivity analysis.

    Visualizes the impact of parameter variations on key metrics using
    horizontal bars sorted by influence magnitude.

    Args:
        sensitivity_data: Dict of parameter names to impact values
        baseline_value: Baseline metric value for reference
        title: Plot title
        figsize: Figure size (width, height)
        show_percentages: Whether to show percentage labels
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with tornado chart

    Examples:
        >>> sensitivity = {
        ...     "Premium Rate": 0.15,
        ...     "Loss Frequency": -0.12,
        ...     "Loss Severity": -0.08,
        ... }
        >>> fig = plot_sensitivity_tornado(sensitivity)
    """
    set_wsj_style()

    # Generate or use provided data
    if sensitivity_data is None:
        # Generate synthetic demonstration data
        sensitivity_data = {
            "Premium Rate": 0.18,
            "Loss Frequency": -0.15,
            "Loss Severity": -0.12,
            "Retention Level": 0.10,
            "Coverage Limit": 0.08,
            "Investment Return": 0.07,
            "Operating Margin": 0.06,
            "Tax Rate": -0.05,
            "Working Capital": -0.04,
            "Growth Rate": 0.03,
        }

    if baseline_value is None:
        baseline_value = 0.12  # 12% ROE baseline

    # Sort by absolute impact
    sorted_params = sorted(sensitivity_data.items(), key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    params = [item[0] for item in sorted_params]
    impacts = [item[1] for item in sorted_params]

    # Calculate bar positions
    y_pos = np.arange(len(params))

    # Color coding based on impact magnitude
    colors = []
    for impact in impacts:
        abs_impact = abs(impact)
        if abs_impact >= 0.10:  # High sensitivity (>10%)
            colors.append(WSJ_COLORS["red"])
        elif abs_impact >= 0.05:  # Moderate sensitivity (5-10%)
            colors.append(WSJ_COLORS["orange"])
        else:  # Low sensitivity (<5%)
            colors.append(WSJ_COLORS["green"])

    # Create bars
    bars = ax.barh(y_pos, impacts, color=colors, alpha=0.7, edgecolor="black", linewidth=1)

    # Add baseline reference line
    ax.axvline(x=0, color=WSJ_COLORS["gray"], linewidth=2, linestyle="-", alpha=0.5)
    ax.text(
        0,
        len(params),
        "Baseline",
        ha="center",
        va="bottom",
        fontsize=10,
        color=WSJ_COLORS["gray"],
        fontweight="bold",
    )

    # Add percentage labels
    if show_percentages:
        for bar_item, param, impact in zip(bars, params, impacts):
            width = bar_item.get_width()
            label_x = width * 1.02 if width > 0 else width * 1.02
            ax.text(
                label_x,
                bar_item.get_y() + bar_item.get_height() / 2,
                f"{impact:+.1%}",
                ha="left" if width > 0 else "right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # Format axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel("Impact on Key Metric (%)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:+.0%}"))

    # Add grid
    ax.grid(True, axis="x", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=WSJ_COLORS["red"], alpha=0.7, label="High Sensitivity (>10%)"),
        Patch(facecolor=WSJ_COLORS["orange"], alpha=0.7, label="Moderate (5-10%)"),
        Patch(facecolor=WSJ_COLORS["green"], alpha=0.7, label="Low (<5%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add annotation
    fig.text(
        0.5,
        0.02,
        f"Baseline Value: {baseline_value:.1%} | Analysis shows ±20% parameter variation impact",
        ha="center",
        fontsize=10,
        style="italic",
        color=WSJ_COLORS["gray"],
    )

    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_robustness_heatmap(  # pylint: disable=too-many-locals,too-many-statements
    robustness_data: Optional[np.ndarray] = None,
    frequency_range: Optional[Tuple[float, float]] = None,
    severity_range: Optional[Tuple[float, float]] = None,
    title: str = "Insurance Program Robustness Analysis",
    figsize: Tuple[int, int] = (10, 8),
    show_reference: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create robustness heatmap showing stability across parameter variations.

    Visualizes how optimal coverage changes with variations in loss
    frequency and severity parameters.

    Args:
        robustness_data: 2D array of stability metrics
        frequency_range: Tuple of (min, max) frequency variation (default 0.7-1.3)
        severity_range: Tuple of (min, max) severity variation (default 0.7-1.3)
        title: Plot title
        figsize: Figure size (width, height)
        show_reference: Whether to show reference point at 100%/100%
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with robustness heatmap

    Examples:
        >>> fig = plot_robustness_heatmap()
        >>> fig.savefig("robustness.png", dpi=150)
    """
    set_wsj_style()

    if frequency_range is None:
        frequency_range = (0.7, 1.3)
    if severity_range is None:
        severity_range = (0.7, 1.3)

    fig, ax = plt.subplots(figsize=figsize)

    # Generate or use provided data
    if robustness_data is None:
        # Generate synthetic demonstration data
        n_points = 20
        freq_values = np.linspace(frequency_range[0], frequency_range[1], n_points)
        sev_values = np.linspace(severity_range[0], severity_range[1], n_points)

        F, S = np.meshgrid(freq_values, sev_values)

        # Create stability metric (1 = stable, 0 = unstable)
        # More stable near baseline (1.0, 1.0)
        distance_from_baseline = np.sqrt((F - 1.0) ** 2 + (S - 1.0) ** 2)
        robustness_data = np.exp(-2 * distance_from_baseline)

        # Add some structure/noise
        robustness_data = robustness_data + 0.1 * np.sin(5 * F) * np.cos(5 * S)
        robustness_data = cast(np.ndarray, robustness_data)  # Guaranteed not None after assignment
        robustness_data = np.clip(robustness_data, 0, 1)
    else:
        n_points = robustness_data.shape[0]
        freq_values = np.linspace(frequency_range[0], frequency_range[1], n_points)
        sev_values = np.linspace(severity_range[0], severity_range[1], n_points)
        F, S = np.meshgrid(freq_values, sev_values)

    # Create heatmap
    im = ax.contourf(
        F * 100,  # Convert to percentage
        S * 100,
        robustness_data,
        levels=20,
        cmap="RdYlGn",
        alpha=0.8,
    )

    # Add contour lines
    contours = ax.contour(
        F * 100,
        S * 100,
        robustness_data,
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="black",
        alpha=0.3,
        linewidths=1,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Mark reference point if requested
    if show_reference:
        ax.scatter(
            100,
            100,  # Baseline at 100%
            s=200,
            color="white",
            marker="o",
            edgecolor="black",
            linewidth=3,
            zorder=10,
            label="Baseline",
        )

        # Add crosshairs
        ax.axhline(y=100, color=WSJ_COLORS["gray"], linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=100, color=WSJ_COLORS["gray"], linestyle="--", alpha=0.5, linewidth=1)

    # Add regions
    # Stable region (robustness > 0.7)
    stable_mask = robustness_data > 0.7
    if np.any(stable_mask):
        ax.contour(
            F * 100,
            S * 100,
            stable_mask.astype(float),
            levels=[0.5],
            colors=WSJ_COLORS["green"],
            linewidths=2,
            linestyles="-",
            alpha=0.7,
        )

        # Find center of stable region
        stable_indices = np.where(stable_mask)
        if len(stable_indices[0]) > 0:
            center_y = S[stable_indices] * 100
            center_x = F[stable_indices] * 100
            ax.text(
                center_x.mean(),
                center_y.mean(),
                "STABLE\nREGION",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=WSJ_COLORS["green"],
                alpha=0.7,
            )

    # Format axes
    ax.set_xlabel("Loss Frequency Variation (%)", fontsize=11)
    ax.set_ylabel("Loss Severity Variation (%)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Set axis limits
    ax.set_xlim(frequency_range[0] * 100, frequency_range[1] * 100)
    ax.set_ylim(severity_range[0] * 100, severity_range[1] * 100)

    # Format tick labels
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coverage Stability (0=Unstable, 1=Stable)", fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.2)

    # Add legend only if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9)

    # Add annotations
    fig.text(
        0.5,
        0.02,
        "Green regions indicate robust insurance configurations that remain optimal despite parameter uncertainty",
        ha="center",
        fontsize=10,
        style="italic",
        color=WSJ_COLORS["gray"],
    )

    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_premium_multiplier(  # pylint: disable=too-many-locals,too-many-statements
    optimization_results: Optional[Dict[float, Dict[str, Any]]] = None,
    company_sizes: Optional[List[float]] = None,
    title: str = "Premium Multiplier Analysis",
    figsize: Tuple[int, int] = (12, 8),
    show_confidence: bool = True,
    show_reference_lines: bool = True,
    show_annotations: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create premium multiplier analysis visualization.

    Visualizes the optimal premium as a multiple of expected loss for different
    company sizes, demonstrating why premiums 2-5× expected losses are optimal
    from an ergodic perspective.

    Args:
        optimization_results: Dict of company_size -> optimization data containing
                            'expected_loss', 'optimal_premium', and optionally
                            'confidence_bounds' for each company size
        company_sizes: List of company sizes to analyze (default: [1e6, 1e7, 1e8])
        title: Plot title
        figsize: Figure size (width, height)
        show_confidence: Whether to show confidence intervals
        show_reference_lines: Whether to show horizontal reference lines
        show_annotations: Whether to add explanatory annotations
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with premium multiplier analysis

    Examples:
        >>> results = {
        ...     1e6: {'expected_loss': 50000, 'optimal_premium': 150000},
        ...     1e7: {'expected_loss': 200000, 'optimal_premium': 600000}
        ... }
        >>> fig = plot_premium_multiplier(results)
    """
    set_wsj_style()

    if company_sizes is None:
        company_sizes = [1_000_000, 10_000_000, 100_000_000]

    fig, ax = plt.subplots(figsize=figsize)

    # Generate or use provided data
    if optimization_results is None:
        # Generate synthetic demonstration data
        optimization_results = {}
        for size in company_sizes:
            # Expected loss scales with company size
            expected_loss = size * 0.005  # 0.5% of company size

            # Optimal premium multiplier varies by company size
            # Smaller companies need higher multiples for safety
            size_factor = np.log10(size / 1_000_000)
            base_multiplier = 3.5 - 0.3 * size_factor  # Decreases with size

            # Add some structure
            optimal_premium = expected_loss * base_multiplier

            # Generate confidence bounds
            confidence_lower = optimal_premium * 0.85
            confidence_upper = optimal_premium * 1.15

            optimization_results[size] = {
                "expected_loss": expected_loss,
                "optimal_premium": optimal_premium,
                "confidence_bounds": (confidence_lower, confidence_upper),
                "percentiles": {
                    "25": optimal_premium * 0.9,
                    "75": optimal_premium * 1.1,
                },
            }

    # Prepare data for plotting
    sizes_log = np.array([np.log10(s) for s in company_sizes])
    multipliers = []
    confidence_lower = []
    confidence_upper = []

    for size in company_sizes:
        data = optimization_results[size]
        multiplier = data["optimal_premium"] / data["expected_loss"]
        multipliers.append(multiplier)

        if "confidence_bounds" in data and show_confidence:
            confidence_lower.append(data["confidence_bounds"][0] / data["expected_loss"])
            confidence_upper.append(data["confidence_bounds"][1] / data["expected_loss"])

    # Create smooth interpolation for better visualization
    sizes_smooth = np.linspace(sizes_log.min(), sizes_log.max(), 100)

    # Use cubic spline for smooth curve
    spline = make_interp_spline(sizes_log, multipliers, k=min(3, len(multipliers) - 1))
    multipliers_smooth = spline(sizes_smooth)

    # Convert back to actual company sizes for x-axis
    company_sizes_smooth = 10**sizes_smooth

    # Main curve
    ax.semilogx(
        company_sizes_smooth,
        multipliers_smooth,
        color=WSJ_COLORS["blue"],
        linewidth=3,
        label="Optimal Premium/Expected Loss",
        zorder=5,
    )

    # Data points
    ax.scatter(
        company_sizes,
        multipliers,
        color=WSJ_COLORS["blue"],
        s=100,
        edgecolor="white",
        linewidth=2,
        zorder=10,
    )

    # Confidence intervals
    if confidence_lower and show_confidence:
        # Interpolate confidence bounds
        spline_lower = make_interp_spline(
            sizes_log, confidence_lower, k=min(3, len(confidence_lower) - 1)
        )
        spline_upper = make_interp_spline(
            sizes_log, confidence_upper, k=min(3, len(confidence_upper) - 1)
        )
        confidence_lower_smooth = spline_lower(sizes_smooth)
        confidence_upper_smooth = spline_upper(sizes_smooth)

        ax.fill_between(
            company_sizes_smooth,
            confidence_lower_smooth,
            confidence_upper_smooth,
            alpha=0.2,
            color=WSJ_COLORS["blue"],
            label="95% Confidence Interval",
        )

    # Reference lines
    if show_reference_lines:
        reference_levels = [1, 2, 3, 5]
        colors = [WSJ_COLORS["gray"], WSJ_COLORS["orange"], WSJ_COLORS["green"], WSJ_COLORS["red"]]
        styles = [":", "--", "--", "--"]

        for level, color, style in zip(reference_levels, colors, styles):
            ax.axhline(
                y=level,
                color=color,
                linestyle=style,
                alpha=0.5,
                linewidth=1.5,
                label=f"{level}× Expected Loss",
            )

    # Annotations
    if show_annotations:
        # Annotate key insight regions
        ax.annotate(
            "Small Companies\nNeed Higher\nMultiples",
            xy=(company_sizes[0], multipliers[0]),
            xytext=(company_sizes[0] * 0.3, multipliers[0] + 0.5),
            fontsize=10,
            color=WSJ_COLORS["gray"],
            fontweight="bold",
            ha="center",
            arrowprops={"arrowstyle": "->", "color": WSJ_COLORS["gray"], "alpha": 0.5, "lw": 1},
        )

        if len(company_sizes) > 2:
            ax.annotate(
                "Large Companies\nCan Accept\nLower Multiples",
                xy=(company_sizes[-1], multipliers[-1]),
                xytext=(company_sizes[-1] * 1.5, multipliers[-1] - 0.5),
                fontsize=10,
                color=WSJ_COLORS["gray"],
                fontweight="bold",
                ha="center",
                arrowprops={"arrowstyle": "->", "color": WSJ_COLORS["gray"], "alpha": 0.5, "lw": 1},
            )

        # Add shaded optimal zone
        ax.axhspan(2, 5, alpha=0.05, color="green", label="Optimal Zone (2-5×)")

    # Format axes
    ax.set_xlabel("Company Size (Assets)", fontsize=12)
    ax.set_ylabel("Premium Multiple (Premium / Expected Loss)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Format x-axis
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(WSJFormatter.currency_formatter))

    # Format y-axis
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}×"))

    # Grid
    ax.grid(True, which="both", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    # Legend
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor=WSJ_COLORS["gray"])

    # Set reasonable y-limits
    ax.set_ylim(0, max(multipliers) * 1.3)

    safe_tight_layout()

    # Set export DPI if specified
    if export_dpi:
        fig.dpi = export_dpi

    return fig


def plot_breakeven_timeline(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    simulation_results: Optional[Dict[float, Dict[str, Any]]] = None,
    company_sizes: Optional[List[float]] = None,
    time_horizon: int = 30,
    title: str = "Insurance Break-even Timeline Analysis",
    figsize: Tuple[int, int] = (14, 8),
    show_percentiles: bool = True,
    show_breakeven_markers: bool = True,
    export_dpi: Optional[int] = None,
) -> Figure:
    """Create break-even timeline visualization.

    Shows when the cumulative benefits of optimal insurance exceed the cumulative
    excess premiums paid (premiums above expected losses), demonstrating the
    long-term value proposition.

    Args:
        simulation_results: Dict of company_size -> simulation data containing
                          'cumulative_benefit', 'cumulative_excess_premium',
                          and optionally percentile data
        company_sizes: List of company sizes to analyze (default: [1e6, 1e7, 1e8])
        time_horizon: Years to simulate (default: 30)
        title: Plot title
        figsize: Figure size (width, height)
        show_percentiles: Whether to show 25th/75th percentile bands
        show_breakeven_markers: Whether to mark break-even points
        export_dpi: DPI for export (150 for web, 300 for print)

    Returns:
        Matplotlib figure with break-even timeline analysis

    Examples:
        >>> results = {
        ...     1e6: {
        ...         'cumulative_benefit': np.array([...]),
        ...         'cumulative_excess_premium': np.array([...])
        ...     }
        ... }
        >>> fig = plot_breakeven_timeline(results)
    """
    set_wsj_style()

    if company_sizes is None:
        company_sizes = [1_000_000, 10_000_000, 100_000_000]

    # Create subplot grid
    n_companies = len(company_sizes)
    fig, axes = plt.subplots(1, n_companies, figsize=figsize, sharey=True)

    if n_companies == 1:
        axes = [axes]

    # Generate or use provided data
    if simulation_results is None:
        # Generate synthetic demonstration data
        simulation_results = {}
        rng = np.random.default_rng(42)

        for size in company_sizes:
            years = np.arange(time_horizon)

            # Expected annual loss and optimal premium
            expected_loss = size * 0.005  # 0.5% of size
            size_factor = np.log10(size / 1_000_000)
            premium_multiplier = 3.5 - 0.3 * size_factor
            optimal_premium = expected_loss * premium_multiplier
            excess_premium = optimal_premium - expected_loss

            # Cumulative excess premium paid (linear accumulation)
            cumulative_excess = excess_premium * (years + 1)

            # Cumulative benefit (grows exponentially due to avoided ruin)
            # Starts slow, accelerates over time
            base_benefit = size * 0.001  # 0.1% annual benefit initially
            growth_factor = 1.08  # Benefit compounds
            cumulative_benefit = base_benefit * (
                (growth_factor ** (years + 1) - 1) / (growth_factor - 1)
            )

            # Add some realistic noise
            noise = rng.normal(0, size * 0.0005, len(years))
            cumulative_benefit += np.cumsum(noise)

            # Calculate percentiles
            benefit_25 = cumulative_benefit * 0.85
            benefit_75 = cumulative_benefit * 1.15

            simulation_results[size] = {
                "years": years,
                "cumulative_benefit": cumulative_benefit,
                "cumulative_excess_premium": cumulative_excess,
                "benefit_25": benefit_25,
                "benefit_75": benefit_75,
                "net_benefit": cumulative_benefit - cumulative_excess,
            }

    # Plot each company size
    for idx, (ax, company_size) in enumerate(zip(axes, company_sizes)):
        data = simulation_results[company_size]
        years = data.get("years", np.arange(time_horizon))

        # Main lines
        ax.plot(
            years,
            data["cumulative_benefit"] / 1e6,
            color=WSJ_COLORS["blue"],
            linewidth=2.5,
            label="Cumulative Benefit",
        )

        ax.plot(
            years,
            data["cumulative_excess_premium"] / 1e6,
            color=WSJ_COLORS["red"],
            linewidth=2.5,
            label="Excess Premium Paid",
        )

        # Percentile bands
        if show_percentiles and "benefit_25" in data:
            ax.fill_between(
                years,
                data["benefit_25"] / 1e6,
                data["benefit_75"] / 1e6,
                alpha=0.2,
                color=WSJ_COLORS["blue"],
                label="25-75% Percentile",
            )

        # Find break-even point
        net_benefit = data["cumulative_benefit"] - data["cumulative_excess_premium"]
        breakeven_idx = np.where(net_benefit > 0)[0]

        if len(breakeven_idx) > 0 and show_breakeven_markers:
            breakeven_year = years[breakeven_idx[0]]
            breakeven_benefit = data["cumulative_benefit"][breakeven_idx[0]]

            # Mark break-even point
            ax.scatter(
                breakeven_year,
                breakeven_benefit / 1e6,
                s=200,
                color=WSJ_COLORS["green"],
                marker="*",
                edgecolor="white",
                linewidth=2,
                zorder=10,
                label=f"Break-even: Year {breakeven_year}",
            )

            # Add vertical line at break-even
            ax.axvline(
                x=breakeven_year,
                color=WSJ_COLORS["green"],
                linestyle="--",
                alpha=0.5,
                linewidth=1.5,
            )

            # Add annotation
            ax.annotate(
                f"Break-even\nYear {breakeven_year}",
                xy=(breakeven_year, breakeven_benefit / 1e6),
                xytext=(breakeven_year + 2, breakeven_benefit / 1e6 * 0.8),
                fontsize=9,
                fontweight="bold",
                color=WSJ_COLORS["green"],
                arrowprops={
                    "arrowstyle": "->",
                    "color": WSJ_COLORS["green"],
                    "alpha": 0.5,
                    "lw": 1,
                },
            )

        # Shade positive NPV region
        positive_region = net_benefit > 0
        if np.any(positive_region):
            ax.fill_between(
                years,
                0,
                (data["cumulative_benefit"] - data["cumulative_excess_premium"]) / 1e6,
                where=positive_region,
                alpha=0.1,
                color="green",
                label="Positive NPV",
            )

        # Format axes
        ax.set_xlabel("Years", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Cumulative Value ($M)", fontsize=11)

        # Title for each subplot
        ax.set_title(
            f"${format_currency(company_size, decimals=0).replace('$', '')} Company",
            fontsize=11,
            fontweight="bold",
        )

        # Grid
        ax.grid(True, alpha=0.3)

        # Legend (only on first subplot to avoid clutter)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)

        # Set reasonable limits
        ax.set_xlim(0, time_horizon)
        ax.set_ylim(bottom=0)

    # Main title
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Add footer annotation
    fig.text(
        0.5,
        0.02,
        "Analysis shows when insurance transforms from cost to investment through compound benefits",
        ha="center",
        fontsize=10,
        style="italic",
        color=WSJ_COLORS["gray"],
    )

    safe_tight_layout()

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

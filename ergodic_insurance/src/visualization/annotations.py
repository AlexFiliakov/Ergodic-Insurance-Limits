"""Annotation and labeling utilities for visualizations.

This module provides utilities for adding professional annotations,
labels, and callouts to plots.
"""

from typing import Optional, Tuple

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from .core import WSJ_COLORS


def add_value_labels(
    ax: Axes,
    bars,
    format_func=None,
    fontsize: int = 9,
    va: str = "bottom",
    ha: str = "center",
    offset: float = 0.01,
    color: Optional[str] = None,
    bold: bool = False,
) -> None:
    """Add value labels to bar chart.

    Adds formatted value labels on top of or inside bars in a bar chart.

    Args:
        ax: Matplotlib axes
        bars: Bar container from ax.bar()
        format_func: Optional function to format values
        fontsize: Font size for labels
        va: Vertical alignment ('bottom', 'center', 'top')
        ha: Horizontal alignment ('left', 'center', 'right')
        offset: Vertical offset as fraction of y-range
        color: Text color (default: gray)
        bold: Whether to make text bold

    Examples:
        >>> fig, ax = plt.subplots()
        >>> bars = ax.bar(range(5), [10, 20, 15, 25, 30])
        >>> add_value_labels(ax, bars)
    """
    color = color or WSJ_COLORS["gray"]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset_value = offset * y_range

    for bar in bars:
        height = bar.get_height()
        if format_func:
            label = format_func(height)
        else:
            label = f"{height:.1f}"

        y_pos = height + offset_value if va == "bottom" else height / 2

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            label,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight="bold" if bold else "normal",
        )


def add_trend_annotation(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    trend: float,
    period: str = "YoY",
    fontsize: int = 10,
) -> None:
    """Add trend annotation with arrow.

    Adds a trend annotation showing percentage change with an up/down arrow.

    Args:
        ax: Matplotlib axes
        x_pos: X position for annotation
        y_pos: Y position for annotation
        trend: Trend value (e.g., 0.15 for 15% increase)
        period: Period description (e.g., "YoY", "MoM")
        fontsize: Font size for annotation

    Examples:
        >>> fig, ax = plt.subplots()
        >>> add_trend_annotation(ax, 0.8, 0.9, 0.15, "YoY")
    """
    if trend >= 0:
        arrow = "↑"
        color = WSJ_COLORS["green"]
        sign = "+"
    else:
        arrow = "↓"
        color = WSJ_COLORS["red"]
        sign = ""

    text = f"{arrow} {sign}{trend*100:.1f}% {period}"

    ax.annotate(
        text,
        xy=(x_pos, y_pos),
        xycoords="axes fraction",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
    )


def add_callout(
    ax: Axes,
    text: str,
    xy: Tuple[float, float],
    xytext: Tuple[float, float],
    fontsize: int = 9,
    color: Optional[str] = None,
    arrow_color: Optional[str] = None,
    bbox_props: Optional[dict] = None,
) -> None:
    """Add a callout annotation with arrow.

    Creates a professional callout annotation with customizable styling.

    Args:
        ax: Matplotlib axes
        text: Callout text
        xy: Point to annotate (data coordinates)
        xytext: Text position (data coordinates)
        fontsize: Font size
        color: Text color
        arrow_color: Arrow color
        bbox_props: Box properties for text background

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> add_callout(ax, "Peak value", xy=(2, 4), xytext=(2.5, 4.5))
    """
    color = color or WSJ_COLORS["black"]
    arrow_color = arrow_color or WSJ_COLORS["gray"]

    if bbox_props is None:
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=WSJ_COLORS["gray"],
            linewidth=0.5,
            alpha=0.9,
        )

    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        color=color,
        bbox=bbox_props,
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=0.2",
            color=arrow_color,
            linewidth=1,
        ),
    )


def add_benchmark_line(
    ax: Axes,
    value: float,
    label: str,
    color: Optional[str] = None,
    linestyle: str = "--",
    linewidth: float = 1.5,
    fontsize: int = 9,
    position: str = "right",
) -> None:
    """Add a horizontal benchmark line with label.

    Draws a horizontal reference line with an integrated label.

    Args:
        ax: Matplotlib axes
        value: Y-value for benchmark line
        label: Label for the benchmark
        color: Line color
        linestyle: Line style
        linewidth: Line width
        fontsize: Font size for label
        position: Label position ('left', 'right', 'center')

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(range(10), np.random.randn(10))
        >>> add_benchmark_line(ax, 0, "Average", color="red")
    """
    color = color or WSJ_COLORS["gray"]

    # Add the line
    line = ax.axhline(
        y=value,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.7,
    )

    # Add the label
    x_pos = 0.02 if position == "left" else 0.98 if position == "right" else 0.5
    ha = "left" if position == "left" else "right" if position == "right" else "center"

    ax.text(
        x_pos,
        value,
        f" {label} ",
        transform=ax.get_yaxis_transform(),
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color,
            linewidth=0.5,
            alpha=0.9,
        ),
    )


def add_shaded_region(
    ax: Axes,
    x_start: float,
    x_end: float,
    label: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 0.2,
) -> None:
    """Add a shaded vertical region.

    Creates a shaded vertical band to highlight a specific time period or range.

    Args:
        ax: Matplotlib axes
        x_start: Start x-coordinate
        x_end: End x-coordinate
        label: Optional label for the region
        color: Fill color
        alpha: Transparency

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(range(10), np.random.randn(10))
        >>> add_shaded_region(ax, 3, 6, label="Recession", color="gray")
    """
    color = color or WSJ_COLORS["gray"]

    ax.axvspan(x_start, x_end, alpha=alpha, color=color, label=label)

    if label:
        # Add text label in the middle of the region
        x_mid = (x_start + x_end) / 2
        y_pos = ax.get_ylim()[1] * 0.95

        ax.text(
            x_mid,
            y_pos,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color=WSJ_COLORS["gray"],
            fontweight="bold",
        )


def add_data_source(
    fig,
    source: str,
    x: float = 0.99,
    y: float = 0.01,
    fontsize: int = 8,
    color: Optional[str] = None,
) -> None:
    """Add data source attribution.

    Adds a data source note to the figure, typically at the bottom right.

    Args:
        fig: Matplotlib figure
        source: Source text (e.g., "Source: Company Reports")
        x: X position in figure coordinates
        y: Y position in figure coordinates
        fontsize: Font size
        color: Text color

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> add_data_source(fig, "Source: Internal Analysis")
    """
    color = color or WSJ_COLORS["gray"]

    fig.text(
        x,
        y,
        source,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color=color,
        style="italic",
    )


def add_footnote(
    fig,
    text: str,
    x: float = 0.5,
    y: float = 0.02,
    fontsize: int = 8,
    color: Optional[str] = None,
) -> None:
    """Add footnote to figure.

    Adds a footnote or explanatory text to the bottom of the figure.

    Args:
        fig: Matplotlib figure
        text: Footnote text
        x: X position in figure coordinates
        y: Y position in figure coordinates
        fontsize: Font size
        color: Text color

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> add_footnote(fig, "* Preliminary data subject to revision")
    """
    color = color or WSJ_COLORS["gray"]

    fig.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        wrap=True,
    )

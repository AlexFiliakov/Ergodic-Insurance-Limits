"""Annotation and labeling utilities for visualizations.

This module provides utilities for adding professional annotations,
labels, and callouts to plots with smart placement and leader line routing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist

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

    for bar_elem in bars:
        height = bar_elem.get_height()
        if format_func:
            label = format_func(height)
        else:
            label = f"{height:.1f}"

        y_pos = height + offset_value if va == "bottom" else height / 2

        ax.text(
            bar_elem.get_x() + bar_elem.get_width() / 2,
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
        bbox_props = {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": WSJ_COLORS["gray"],
            "linewidth": 0.5,
            "alpha": 0.9,
        }

    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        color=color,
        bbox=bbox_props,
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.2",
            "color": arrow_color,
            "linewidth": 1,
        },
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
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.5,
            "alpha": 0.9,
        },
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


@dataclass
class AnnotationBox:
    """Container for annotation box properties.

    Attributes:
        text: Annotation text
        position: (x, y) position in data coordinates
        width: Box width in axes fraction
        height: Box height in axes fraction
        priority: Priority for placement (higher = more important)
    """

    text: str
    position: Tuple[float, float]
    width: float = 0.15
    height: float = 0.08
    priority: int = 50

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x, y, width, height)."""
        return (self.position[0], self.position[1], self.width, self.height)

    def overlaps(self, other: "AnnotationBox", margin: float = 0.01) -> bool:
        """Check if this box overlaps with another.

        Args:
            other: Another annotation box
            margin: Additional margin to consider

        Returns:
            True if boxes overlap
        """
        x1, y1, w1, h1 = self.get_bounds()
        x2, y2, w2, h2 = other.get_bounds()

        # Add margin
        x1 -= margin
        y1 -= margin
        w1 += 2 * margin
        h1 += 2 * margin

        # Check overlap
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)


class SmartAnnotationPlacer:
    """Smart placement system for annotations without overlaps."""

    def __init__(self, ax: Axes):
        """Initialize placer with axes.

        Args:
            ax: Matplotlib axes
        """
        self.ax = ax
        self.placed_annotations: List[AnnotationBox] = []
        self.candidate_positions = self._generate_candidate_positions()

    def _generate_candidate_positions(self, n_positions: int = 20) -> List[Tuple[float, float]]:
        """Generate candidate positions for annotations.

        Args:
            n_positions: Number of candidate positions

        Returns:
            List of (x, y) positions in axes fraction
        """
        positions = []

        # Grid positions
        for x in np.linspace(0.1, 0.9, 5):
            for y in np.linspace(0.1, 0.9, 4):
                positions.append((x, y))

        # Corner positions (preferred)
        corners = [(0.05, 0.95), (0.95, 0.95), (0.05, 0.05), (0.95, 0.05)]
        positions = corners + positions

        return positions

    def find_best_position(
        self,
        target_point: Tuple[float, float],
        text: str,
        priority: int = 50,
        preferred_quadrant: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Find best position for annotation near target point.

        Args:
            target_point: Point to annotate (data coordinates)
            text: Annotation text
            priority: Annotation priority
            preferred_quadrant: Preferred quadrant ('NE', 'NW', 'SE', 'SW')

        Returns:
            Best position in axes fraction coordinates
        """
        # Convert target point to axes fraction
        trans = self.ax.transData + self.ax.transAxes.inverted()
        target_axes = trans.transform(target_point)

        # Filter positions by quadrant if specified
        candidates = self.candidate_positions.copy()

        if preferred_quadrant:
            quadrant_filters = {
                "NE": lambda p: p[0] > target_axes[0] and p[1] > target_axes[1],
                "NW": lambda p: p[0] < target_axes[0] and p[1] > target_axes[1],
                "SE": lambda p: p[0] > target_axes[0] and p[1] < target_axes[1],
                "SW": lambda p: p[0] < target_axes[0] and p[1] < target_axes[1],
            }

            if preferred_quadrant in quadrant_filters:
                filter_func = quadrant_filters[preferred_quadrant]
                filtered = [p for p in candidates if filter_func(p)]
                if filtered:
                    candidates = filtered

        # Score each candidate position
        best_position = None
        best_score = float("inf")

        for pos in candidates:
            # Create test annotation box
            test_box = AnnotationBox(text, pos, priority=priority)

            # Check for overlaps
            overlap_penalty = sum(
                100 if test_box.overlaps(existing) else 0 for existing in self.placed_annotations
            )

            # Distance penalty (prefer closer positions)
            distance = np.linalg.norm(np.array(pos) - np.array(target_axes))
            distance_penalty = distance * 10

            # Edge penalty (avoid edges)
            edge_penalty = 0
            if pos[0] < 0.1 or pos[0] > 0.9:
                edge_penalty += 20
            if pos[1] < 0.1 or pos[1] > 0.9:
                edge_penalty += 20

            # Total score
            score = overlap_penalty + distance_penalty + edge_penalty

            if score < best_score:
                best_score = score
                best_position = pos

        # If no good position found, use offset from target
        if best_position is None or best_score > 200:
            offset_x = 0.1 if target_axes[0] < 0.5 else -0.1
            offset_y = 0.1 if target_axes[1] < 0.5 else -0.1
            best_position = (
                max(0.05, min(0.95, target_axes[0] + offset_x)),
                max(0.05, min(0.95, target_axes[1] + offset_y)),
            )

        # Record placed annotation
        self.placed_annotations.append(AnnotationBox(text, best_position, priority=priority))

        return best_position

    def add_smart_callout(
        self,
        text: str,
        target_point: Tuple[float, float],
        fontsize: int = 9,
        priority: int = 50,
        preferred_quadrant: Optional[str] = None,
        color: Optional[str] = None,
        arrow_color: Optional[str] = None,
    ) -> None:
        """Add callout with smart placement.

        Args:
            text: Callout text
            target_point: Point to annotate (data coordinates)
            fontsize: Font size
            priority: Annotation priority (higher = more important)
            preferred_quadrant: Preferred quadrant for placement
            color: Text color
            arrow_color: Arrow color
        """
        # Find best position
        best_pos = self.find_best_position(target_point, text, priority, preferred_quadrant)

        # Convert back to data coordinates
        trans = self.ax.transAxes + self.ax.transData.inverted()
        text_pos = trans.transform(best_pos)

        # Add the callout
        add_callout(
            self.ax,
            text,
            xy=target_point,
            xytext=text_pos,
            fontsize=fontsize,
            color=color,
            arrow_color=arrow_color,
        )

    def add_smart_annotations(self, annotations: List[Dict[str, Any]], fontsize: int = 9) -> None:
        """Add multiple annotations with smart placement.

        Args:
            annotations: List of annotation dicts with 'text', 'point', 'priority'
            fontsize: Font size for all annotations

        Examples:
            >>> placer = SmartAnnotationPlacer(ax)
            >>> annotations = [
            ...     {'text': 'Peak', 'point': (2, 4), 'priority': 80},
            ...     {'text': 'Valley', 'point': (5, 1), 'priority': 60}
            ... ]
            >>> placer.add_smart_annotations(annotations)
        """
        # Sort by priority (highest first)
        sorted_annotations = sorted(annotations, key=lambda x: x.get("priority", 50), reverse=True)

        for ann in sorted_annotations:
            self.add_smart_callout(
                text=ann["text"],
                target_point=ann["point"],
                fontsize=fontsize,
                priority=ann.get("priority", 50),
                preferred_quadrant=ann.get("quadrant"),
                color=ann.get("color"),
                arrow_color=ann.get("arrow_color"),
            )


def create_leader_line(
    ax: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    style: str = "curved",
    color: Optional[str] = None,
    linewidth: float = 1.0,
    alpha: float = 0.7,
) -> None:
    """Create a leader line with intelligent routing.

    Args:
        ax: Matplotlib axes
        start: Start point (data coordinates)
        end: End point (data coordinates)
        style: Line style ('straight', 'curved', 'elbow')
        color: Line color
        linewidth: Line width
        alpha: Line transparency
    """
    color = color or WSJ_COLORS["gray"]

    if style == "straight":
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

    elif style == "curved":
        # Create curved path
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Add curve based on relative positions
        if abs(end[0] - start[0]) > abs(end[1] - start[1]):
            # Horizontal emphasis
            control_y = mid_y + (end[1] - start[1]) * 0.3
            verts = [start, (mid_x, control_y), end]
        else:
            # Vertical emphasis
            control_x = mid_x + (end[0] - start[0]) * 0.3
            verts = [start, (control_x, mid_y), end]

        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = PathPatch(
            path, facecolor="none", edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=1
        )
        ax.add_patch(patch)

    elif style == "elbow":
        # Create elbow (L-shaped) path
        if abs(end[0] - start[0]) > abs(end[1] - start[1]):
            # Horizontal first
            corner = (end[0], start[1])
        else:
            # Vertical first
            corner = (start[0], end[1])

        ax.plot(
            [start[0], corner[0]],
            [start[1], corner[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )
        ax.plot(
            [corner[0], end[0]],
            [corner[1], end[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )


def auto_annotate_peaks_valleys(
    ax: Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_peaks: int = 3,
    n_valleys: int = 2,
    peak_color: Optional[str] = None,
    valley_color: Optional[str] = None,
    fontsize: int = 9,
) -> None:
    """Automatically annotate peaks and valleys in data.

    Args:
        ax: Matplotlib axes
        x_data: X coordinates
        y_data: Y coordinates
        n_peaks: Number of peaks to annotate
        n_valleys: Number of valleys to annotate
        peak_color: Color for peak annotations
        valley_color: Color for valley annotations
        fontsize: Font size
    """
    peak_color = peak_color or WSJ_COLORS["green"]
    valley_color = valley_color or WSJ_COLORS["red"]

    # Find peaks (local maxima)
    from scipy.signal import find_peaks

    peaks, peak_props = find_peaks(y_data, prominence=np.std(y_data) * 0.5)
    valleys, valley_props = find_peaks(-y_data, prominence=np.std(y_data) * 0.5)

    # Sort by prominence
    if len(peaks) > 0:
        peak_prominences = peak_props["prominences"]
        top_peaks = peaks[np.argsort(peak_prominences)[-n_peaks:]]

        placer = SmartAnnotationPlacer(ax)

        for idx in top_peaks:
            value = y_data[idx]
            placer.add_smart_callout(
                text=f"Peak: {value:.2f}",
                target_point=(x_data[idx], value),
                fontsize=fontsize,
                priority=80,
                color=peak_color,
                arrow_color=peak_color,
            )

    if len(valleys) > 0:
        valley_prominences = valley_props["prominences"]
        top_valleys = valleys[np.argsort(valley_prominences)[-n_valleys:]]

        if len(peaks) == 0:
            placer = SmartAnnotationPlacer(ax)

        for idx in top_valleys:
            value = y_data[idx]
            placer.add_smart_callout(
                text=f"Valley: {value:.2f}",
                target_point=(x_data[idx], value),
                fontsize=fontsize,
                priority=70,
                color=valley_color,
                arrow_color=valley_color,
            )

"""Annotation and labeling utilities for visualizations.

This module provides utilities for adding professional annotations,
labels, and callouts to plots with smart placement and leader line routing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path
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
    _line = ax.axhline(
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
        self.used_colors: Set[str] = set()  # Track colors to avoid conflicts
        self.annotation_cache: Dict[str, Tuple[float, float]] = (
            {}
        )  # Cache positions for consistency

    def _generate_candidate_positions(self, n_positions: int = 20) -> List[Tuple[float, float]]:
        """Generate candidate positions for annotations.

        Args:
            n_positions: Number of candidate positions

        Returns:
            List of (x, y) positions in axes fraction
        """
        positions = []

        # Create a denser, more uniform grid with safe margins
        # Stay well within bounds to avoid off-screen annotations
        x_margins = [0.08, 0.92]  # Increased margins from edges
        y_margins = [0.08, 0.92]

        # Primary grid - denser for better coverage
        for x in np.linspace(x_margins[0], x_margins[1], 8):
            for y in np.linspace(y_margins[0], y_margins[1], 6):
                positions.append((x, y))

        # Secondary strategic positions for common annotation areas
        # Upper region (for peaks) - spread out horizontally
        for x in np.linspace(0.15, 0.85, 6):
            positions.append((x, 0.82))
            positions.append((x, 0.72))

        # Middle region (general annotations)
        for x in np.linspace(0.12, 0.88, 7):
            positions.append((x, 0.55))
            positions.append((x, 0.45))

        # Lower region (for valleys)
        for x in np.linspace(0.15, 0.85, 6):
            positions.append((x, 0.28))
            positions.append((x, 0.18))

        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for pos in positions:
            pos_tuple = (round(pos[0], 3), round(pos[1], 3))
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_positions.append(pos)

        return unique_positions

    def find_best_position(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
        trans = self.ax.transData.transform
        inv_trans = self.ax.transAxes.inverted().transform

        # Transform to display coordinates then to axes fraction
        display_point = trans(np.array(target_point))
        target_axes = inv_trans(display_point)

        # Ensure target_axes is in valid range
        target_axes = np.clip(target_axes, 0.0, 1.0)

        # Generate dynamic candidate positions around target
        dynamic_positions = []

        # Create positions in a circle around the target point
        n_angles = 8
        for _i, angle in enumerate(np.linspace(0, 2 * np.pi, n_angles, endpoint=False)):
            # Vary radius based on position in plot
            if target_axes[1] > 0.7:  # Upper part - annotations below
                if angle > np.pi:  # Lower half of circle preferred
                    radius = 0.15
                else:
                    radius = 0.25
            elif target_axes[1] < 0.3:  # Lower part - annotations above
                if angle < np.pi:  # Upper half of circle preferred
                    radius = 0.15
                else:
                    radius = 0.25
            else:  # Middle part - flexible
                radius = 0.2

            x = target_axes[0] + radius * np.cos(angle)
            y = target_axes[1] + radius * np.sin(angle)

            # Keep within plot bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)

            dynamic_positions.append((x, y))

        # Add some offset positions
        offsets = [
            (0.12, 0.08),
            (-0.12, 0.08),  # Above
            (0.12, -0.08),
            (-0.12, -0.08),  # Below
            (0.15, 0.0),
            (-0.15, 0.0),  # Sides
        ]

        for dx, dy in offsets:
            x = np.clip(target_axes[0] + dx, 0.05, 0.95)
            y = np.clip(target_axes[1] + dy, 0.05, 0.95)
            dynamic_positions.append((x, y))

        # Combine dynamic and static positions
        candidates = dynamic_positions + self.candidate_positions

        # Filter by preferred quadrant if specified
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
                200 if test_box.overlaps(existing) else 0 for existing in self.placed_annotations
            )

            # Distance penalty (prefer closer positions but not too close)
            distance = np.linalg.norm(np.array(pos) - np.array(target_axes))
            if distance < 0.05:  # Too close
                distance_penalty = 100.0
            else:
                distance_penalty = float(distance * 20)

            # Edge penalty (avoid edges but less harsh)
            edge_penalty = 0
            if pos[0] < 0.05 or pos[0] > 0.95:
                edge_penalty += 30
            if pos[1] < 0.05 or pos[1] > 0.95:
                edge_penalty += 30

            # Prefer positions that maintain readability
            readability_penalty = 0
            # Prefer annotations above for lower targets
            if target_axes[1] < 0.3 and pos[1] < target_axes[1]:
                readability_penalty += 50
            # Prefer annotations below for upper targets
            elif target_axes[1] > 0.7 and pos[1] > target_axes[1]:
                readability_penalty += 50

            # Total score
            score = overlap_penalty + distance_penalty + edge_penalty + readability_penalty

            if score < best_score:
                best_score = score
                best_position = pos

        # If no good position found, use smart offset from target
        if best_position is None or best_score > 300:
            # Determine best offset based on target position
            if target_axes[1] > 0.7:  # High target - put annotation below
                offset_y = -0.12
            elif target_axes[1] < 0.3:  # Low target - put annotation above
                offset_y = 0.12
            else:  # Middle target - offset based on crowding
                offset_y = 0.1 if target_axes[1] < 0.5 else -0.1

            offset_x = 0.1 if target_axes[0] < 0.5 else -0.1

            best_position = (
                np.clip(target_axes[0] + offset_x, 0.05, 0.95),
                np.clip(target_axes[1] + offset_y, 0.05, 0.95),
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
        # Find best position (returns axes fraction coordinates)
        best_pos = self.find_best_position(target_point, text, priority, preferred_quadrant)

        # Convert axes fraction to display coordinates then to data coordinates
        display_pos = self.ax.transAxes.transform(best_pos)
        text_pos = self.ax.transData.inverted().transform(display_pos)

        # Add the callout
        add_callout(
            self.ax,
            text,
            xy=target_point,
            xytext=tuple(text_pos),
            fontsize=fontsize,
            color=color,
            arrow_color=arrow_color,
        )

    def add_smart_annotations(self, annotations: List[Dict[str, Any]], fontsize: int = 9) -> None:
        """Add multiple annotations with smart placement.

        Args:
            annotations: List of annotation dicts with 'text', 'point', 'priority', 'color'
            fontsize: Font size for all annotations

        Examples:
            >>> placer = SmartAnnotationPlacer(ax)
            >>> annotations = [
            ...     {'text': 'Peak', 'point': (2, 4), 'priority': 80, 'color': 'green'},
            ...     {'text': 'Valley', 'point': (5, 1), 'priority': 60, 'color': 'red'}
            ... ]
            >>> placer.add_smart_annotations(annotations)
        """
        # Sort by priority (highest first)
        sorted_annotations = sorted(annotations, key=lambda x: x.get("priority", 50), reverse=True)

        # Get plot limits
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()

        # Process each annotation using the smart placement system
        for ann in sorted_annotations:
            target = ann["point"]
            text = ann["text"]
            priority = ann.get("priority", 50)
            color = ann.get("color", "black")

            # Ensure color doesn't conflict with line colors
            color = self._get_non_conflicting_color(color)

            # Find best position using the overlap detection system
            best_pos_axes = self.find_best_position(
                target, text, priority, preferred_quadrant=self._get_preferred_quadrant(target)
            )

            # Convert axes fraction to data coordinates for text position
            display_pos = self.ax.transAxes.transform(best_pos_axes)
            text_pos = self.ax.transData.inverted().transform(display_pos)

            # Ensure the text position is within reasonable bounds
            text_x = np.clip(
                text_pos[0],
                xlims[0] + (xlims[1] - xlims[0]) * 0.05,
                xlims[1] - (xlims[1] - xlims[0]) * 0.05,
            )
            text_y = np.clip(
                text_pos[1],
                ylims[0] + (ylims[1] - ylims[0]) * 0.05,
                ylims[1] - (ylims[1] - ylims[0]) * 0.05,
            )

            # Choose arrow style based on distance
            distance = np.sqrt((text_x - target[0]) ** 2 + (text_y - target[1]) ** 2)
            x_range = xlims[1] - xlims[0]
            y_range = ylims[1] - ylims[0]
            normalized_dist = distance / np.sqrt(x_range**2 + y_range**2)

            if normalized_dist < 0.1:
                connection_style = "arc3,rad=0.1"
            elif normalized_dist < 0.2:
                connection_style = "arc3,rad=0.2"
            else:
                connection_style = "arc3,rad=0.3"

            # Add the annotation with improved styling
            self.ax.annotate(
                text,
                xy=target,
                xytext=(text_x, text_y),
                fontsize=fontsize,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.95,
                    "linewidth": 1.0,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "connectionstyle": connection_style,
                    "color": ann.get("arrow_color", color),
                    "linewidth": 1.2,
                    "alpha": 0.8,
                    "shrinkA": 5,  # Shrink arrow from point
                    "shrinkB": 5,  # Shrink arrow from text
                },
                zorder=1000 + priority,  # Layer by priority
                ha="center",
                va="center",
            )

            # Track the color as used
            self.used_colors.add(color)

    def _get_preferred_quadrant(self, point: Tuple[float, float]) -> Optional[str]:
        """Determine preferred quadrant for annotation based on point location."""
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()

        x_ratio = (point[0] - xlims[0]) / (xlims[1] - xlims[0])
        y_ratio = (point[1] - ylims[0]) / (ylims[1] - ylims[0])

        # Prefer opposite quadrant for better visibility
        if x_ratio < 0.3 and y_ratio < 0.3:
            return "NE"  # Point in SW, annotate in NE
        if x_ratio > 0.7 and y_ratio < 0.3:
            return "NW"  # Point in SE, annotate in NW
        if x_ratio < 0.3 and y_ratio > 0.7:
            return "SE"  # Point in NW, annotate in SE
        if x_ratio > 0.7 and y_ratio > 0.7:
            return "SW"  # Point in NE, annotate in SW

        return None  # Let algorithm decide for middle points

    def _get_non_conflicting_color(self, requested_color: str) -> str:
        """Get a color that doesn't conflict with line colors or used annotation colors."""
        # Get line colors from the plot
        line_colors = set()
        for line in self.ax.get_lines():
            line_color = line.get_color()
            if line_color:
                line_colors.add(line_color)

        # Define color alternatives
        color_alternatives = {
            "green": ["#2ca02c", "#006400", "#228B22", "#32CD32"],
            "red": ["#d62728", "#8B0000", "#DC143C", "#FF6347"],
            "blue": ["#1f77b4", "#000080", "#4169E1", "#6495ED"],
            "orange": ["#ff7f0e", "#FF8C00", "#FFA500", "#FFB347"],
            "purple": ["#9467bd", "#800080", "#8B008B", "#9370DB"],
            "brown": ["#8c564b", "#8B4513", "#A0522D", "#D2691E"],
            "pink": ["#e377c2", "#FF69B4", "#FFB6C1", "#FFC0CB"],
            "gray": ["#7f7f7f", "#696969", "#808080", "#A9A9A9"],
            "olive": ["#bcbd22", "#808000", "#6B8E23", "#556B2F"],
            "cyan": ["#17becf", "#00CED1", "#40E0D0", "#48D1CC"],
        }

        # Try to find a non-conflicting variant
        base_color = requested_color.lower()
        if base_color in color_alternatives:
            for alt_color in color_alternatives[base_color]:
                if alt_color not in line_colors and alt_color not in self.used_colors:
                    return alt_color

        # If requested color doesn't conflict, use it
        if requested_color not in line_colors:
            return requested_color

        # Find any available color
        all_colors = []
        for colors in color_alternatives.values():
            all_colors.extend(colors)

        for color in all_colors:
            if color not in line_colors and color not in self.used_colors:
                return color

        # Fallback to the requested color with modification
        return requested_color


def create_leader_line(  # pylint: disable=too-many-locals
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


def auto_annotate_peaks_valleys(  # pylint: disable=too-many-locals
    ax: Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_peaks: int = 3,
    n_valleys: int = 2,
    peak_color: Optional[str] = None,
    valley_color: Optional[str] = None,
    fontsize: int = 9,
    placer: Optional[SmartAnnotationPlacer] = None,
) -> SmartAnnotationPlacer:
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
        placer: Existing SmartAnnotationPlacer to use (creates new if None)

    Returns:
        The SmartAnnotationPlacer instance used
    """
    # Use distinct colors that won't conflict with typical line colors
    peak_color = peak_color or "#2ca02c"  # Distinct green
    valley_color = valley_color or "#d62728"  # Distinct red

    # Find peaks (local maxima)
    from scipy.signal import find_peaks

    peaks, peak_props = find_peaks(y_data, prominence=np.std(y_data) * 0.5)
    valleys, valley_props = find_peaks(-y_data, prominence=np.std(y_data) * 0.5)

    # Use existing placer or create new one
    if placer is None:
        placer = SmartAnnotationPlacer(ax)

    # Collect all peak/valley annotations
    peak_valley_annotations = []

    # Process peaks
    if len(peaks) > 0:
        peak_prominences = peak_props["prominences"]
        top_peaks = peaks[np.argsort(peak_prominences)[-n_peaks:]]

        for i, idx in enumerate(top_peaks):
            value = y_data[idx]
            peak_valley_annotations.append(
                {
                    "text": f"Peak: {value:.2f}",
                    "point": (x_data[idx], value),
                    "priority": 90 - i,  # Highest prominence gets highest priority
                    "color": peak_color,
                    "arrow_color": peak_color,
                }
            )

    # Process valleys
    if len(valleys) > 0:
        valley_prominences = valley_props["prominences"]
        top_valleys = valleys[np.argsort(valley_prominences)[-n_valleys:]]

        for i, idx in enumerate(top_valleys):
            value = y_data[idx]
            peak_valley_annotations.append(
                {
                    "text": f"Valley: {value:.2f}",
                    "point": (x_data[idx], value),
                    "priority": 80 - i,  # Slightly lower priority than peaks
                    "color": valley_color,
                    "arrow_color": valley_color,
                }
            )

    # Add all annotations using smart placement
    if peak_valley_annotations:
        placer.add_smart_annotations(peak_valley_annotations, fontsize=fontsize)

    return placer

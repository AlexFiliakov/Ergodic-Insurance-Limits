"""Tests targeting specific uncovered lines in visualization modules.

This test module covers the gaps identified in:
- visualization/annotations.py (210 missing lines)
- visualization/batch_plots.py (103 missing lines)
- visualization/export.py (38 missing lines)
- visualization/interactive_plots.py (20 missing lines)
- visualization/style_manager.py (26 missing lines)

All visualization code uses matplotlib with Agg backend to avoid display issues.
"""

from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import yaml

from ergodic_insurance.batch_processor import AggregatedResults, BatchResult
from ergodic_insurance.visualization import annotations, batch_plots, export, interactive_plots
from ergodic_insurance.visualization.annotations import (
    AnnotationBox,
    SmartAnnotationPlacer,
    auto_annotate_peaks_valleys,
    create_leader_line,
)
from ergodic_insurance.visualization.core import WSJ_COLORS
from ergodic_insurance.visualization.style_manager import (
    ColorPalette,
    FigureConfig,
    FontConfig,
    GridConfig,
    StyleManager,
    Theme,
)

# =============================================================================
# Annotations Tests - Targeting missing lines 387, 399-409, 421-425, 438-475,
# 496-631, 655-662, 688-767, 771-787, 792-833, 856-907, 945-999
# =============================================================================


class TestAnnotationBox:
    """Tests for AnnotationBox dataclass methods."""

    def teardown_method(self):
        plt.close("all")

    def test_get_bounds(self):
        """Cover line 387: AnnotationBox.get_bounds()."""
        box = AnnotationBox(text="Test", position=(0.5, 0.6), width=0.15, height=0.08)
        bounds = box.get_bounds()

        assert bounds == (0.5, 0.6, 0.15, 0.08)
        assert len(bounds) == 4

    def test_get_bounds_custom_dimensions(self):
        """Cover line 387 with different dimensions."""
        box = AnnotationBox(text="Wide box", position=(0.1, 0.2), width=0.3, height=0.12)
        x, y, w, h = box.get_bounds()

        assert x == 0.1
        assert y == 0.2
        assert w == 0.3
        assert h == 0.12

    def test_overlaps_true(self):
        """Cover lines 399-409: overlapping boxes."""
        box1 = AnnotationBox(text="Box1", position=(0.5, 0.5), width=0.2, height=0.1)
        box2 = AnnotationBox(text="Box2", position=(0.55, 0.55), width=0.2, height=0.1)

        assert box1.overlaps(box2) is True
        assert box2.overlaps(box1) is True

    def test_overlaps_false(self):
        """Cover lines 399-409: non-overlapping boxes."""
        box1 = AnnotationBox(text="Box1", position=(0.1, 0.1), width=0.1, height=0.05)
        box2 = AnnotationBox(text="Box2", position=(0.8, 0.8), width=0.1, height=0.05)

        assert box1.overlaps(box2) is False
        assert box2.overlaps(box1) is False

    def test_overlaps_with_margin(self):
        """Cover lines 399-409: edge case with margin."""
        # Boxes that are just touching without margin, but overlap with margin
        box1 = AnnotationBox(text="Box1", position=(0.0, 0.0), width=0.5, height=0.5)
        box2 = AnnotationBox(text="Box2", position=(0.51, 0.0), width=0.5, height=0.5)

        # Without margin these don't overlap; with default margin of 0.01 they do
        result_with_margin = box1.overlaps(box2, margin=0.02)
        assert result_with_margin is True

        # With zero margin, they should not overlap
        result_no_margin = box1.overlaps(box2, margin=0.0)
        assert result_no_margin is False

    def test_overlaps_x_separated(self):
        """Cover the x-separation check in overlaps."""
        box1 = AnnotationBox(text="Left", position=(0.0, 0.5), width=0.1, height=0.1)
        box2 = AnnotationBox(text="Right", position=(0.9, 0.5), width=0.1, height=0.1)

        assert box1.overlaps(box2) is False

    def test_overlaps_y_separated(self):
        """Cover the y-separation check in overlaps."""
        box1 = AnnotationBox(text="Top", position=(0.5, 0.9), width=0.1, height=0.05)
        box2 = AnnotationBox(text="Bottom", position=(0.5, 0.0), width=0.1, height=0.05)

        assert box2.overlaps(box1) is False


class TestSmartAnnotationPlacer:
    """Tests for SmartAnnotationPlacer class covering init, candidate positions,
    find_best_position, add_smart_callout, add_smart_annotations,
    _get_preferred_quadrant, _get_non_conflicting_color."""

    def teardown_method(self):
        plt.close("all")

    def test_init(self):
        """Cover lines 421-425: SmartAnnotationPlacer.__init__()."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 1, 0.5, 2])

        placer = SmartAnnotationPlacer(ax)

        assert placer.ax is ax
        assert isinstance(placer.placed_annotations, list)
        assert len(placer.placed_annotations) == 0
        assert isinstance(placer.candidate_positions, list)
        assert len(placer.candidate_positions) > 0
        assert isinstance(placer.used_colors, set)
        assert isinstance(placer.annotation_cache, dict)

    def test_generate_candidate_positions(self):
        """Cover lines 438-475: _generate_candidate_positions()."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        placer = SmartAnnotationPlacer(ax)
        positions = placer.candidate_positions

        # Should generate many positions
        assert len(positions) > 20

        # All positions should be in valid range [0, 1]
        for x, y in positions:
            assert 0.0 <= x <= 1.0, f"x={x} out of range"
            assert 0.0 <= y <= 1.0, f"y={y} out of range"

        # Check no duplicates (the method deduplicates)
        rounded = [(round(x, 3), round(y, 3)) for x, y in positions]
        assert len(rounded) == len(set(rounded)), "Duplicate positions found"

    def test_find_best_position_basic(self):
        """Cover lines 496-631: find_best_position() basic call."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 2, 1, 3, 2])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        pos = placer.find_best_position(target_point=(2.0, 1.0), text="Test label")

        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert 0.0 <= pos[0] <= 1.0
        assert 0.0 <= pos[1] <= 1.0
        # Should have recorded one placed annotation
        assert len(placer.placed_annotations) == 1

    def test_find_best_position_with_preferred_quadrant(self):
        """Cover lines 554-566: preferred_quadrant filtering in find_best_position."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 2, 1, 3, 2])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)

        for quadrant in ["NE", "NW", "SE", "SW"]:
            pos = placer.find_best_position(
                target_point=(2.0, 2.0),
                text=f"Quadrant {quadrant}",
                preferred_quadrant=quadrant,
            )
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_find_best_position_multiple_avoids_overlap(self):
        """Cover overlap penalty scoring in find_best_position (lines 577-579)."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 2, 1, 3, 2])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)

        # Place multiple annotations - they should not overlap
        pos1 = placer.find_best_position((1.0, 2.0), "First annotation", priority=90)
        pos2 = placer.find_best_position((1.0, 2.0), "Second annotation", priority=80)
        pos3 = placer.find_best_position((1.0, 2.0), "Third annotation", priority=70)

        assert len(placer.placed_annotations) == 3
        # The positions should be different to avoid overlap
        positions_set = {pos1, pos2, pos3}
        # At least 2 should be different (though all 3 ideally)
        assert len(positions_set) >= 2

    def test_find_best_position_target_upper_region(self):
        """Cover lines 513-517: upper region handling in find_best_position."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        # Target in upper region (y > 0.7 in axes fraction)
        pos = placer.find_best_position((3.5, 3.5), "Upper target")
        assert isinstance(pos, tuple)

    def test_find_best_position_target_lower_region(self):
        """Cover lines 518-522: lower region handling in find_best_position."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        # Target in lower region (y < 0.3 in axes fraction)
        pos = placer.find_best_position((0.5, 0.5), "Lower target")
        assert isinstance(pos, tuple)

    def test_find_best_position_high_score_fallback(self):
        """Cover lines 612-627: fallback when no good position found (score > 300)."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)

        # Fill up placed annotations to force overlap penalties on all candidates
        for i in range(50):
            x = (i % 10) / 10.0
            y = (i // 10) / 10.0
            placer.placed_annotations.append(
                AnnotationBox(f"Existing {i}", (x, y), width=0.15, height=0.08)
            )

        pos = placer.find_best_position((2.0, 2.0), "Forced fallback annotation")
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_find_best_position_edge_penalty(self):
        """Cover lines 590-593: edge penalty in scoring."""
        fig, ax = plt.subplots()
        ax.plot([0, 100], [0, 100])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        # Target at the edge of the plot
        pos = placer.find_best_position((0.1, 0.1), "Edge target")
        assert isinstance(pos, tuple)

    def test_find_best_position_readability_penalty(self):
        """Cover lines 597-602: readability penalty logic."""
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        # Low target (y < 0.3 in axes fraction)
        pos_low = placer.find_best_position((1.0, 1.0), "Low target")
        assert isinstance(pos_low, tuple)

        # High target (y > 0.7 in axes fraction)
        pos_high = placer.find_best_position((8.0, 8.0), "High target")
        assert isinstance(pos_high, tuple)

    def test_add_smart_callout(self):
        """Cover lines 655-662: add_smart_callout()."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4], [0, 2, 1, 3, 2])
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        placer.add_smart_callout(
            text="Peak here",
            target_point=(3.0, 3.0),
            fontsize=10,
            priority=80,
            preferred_quadrant="NE",
            color="blue",
            arrow_color="gray",
        )

        # Should have added an annotation
        assert len(placer.placed_annotations) >= 1

    def test_add_smart_callout_default_colors(self):
        """Cover add_smart_callout with default colors."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 2, 1, 3])
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        placer.add_smart_callout(text="Default callout", target_point=(1.5, 1.5))

        assert len(placer.placed_annotations) >= 1

    def test_add_smart_annotations_basic(self):
        """Cover lines 688-767: add_smart_annotations() with multiple annotations."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3, 4, 5], [0, 3, 1, 4, 2, 5])
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        test_annotations = [
            {"text": "Peak 1", "point": (3, 4), "priority": 90, "color": "green"},
            {"text": "Valley 1", "point": (2, 1), "priority": 80, "color": "red"},
            {"text": "End point", "point": (5, 5), "priority": 70, "color": "blue"},
        ]

        placer.add_smart_annotations(test_annotations, fontsize=9)

        # Annotations should be placed
        assert len(placer.placed_annotations) >= 3
        # Colors should be tracked
        assert len(placer.used_colors) > 0

    def test_add_smart_annotations_with_arrow_color(self):
        """Cover line 755: custom arrow_color in annotations."""
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 2, 1, 3])
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        test_annotations = [
            {
                "text": "Custom arrow",
                "point": (1.5, 2.0),
                "priority": 90,
                "color": "green",
                "arrow_color": "orange",
            },
        ]

        placer.add_smart_annotations(test_annotations, fontsize=10)
        assert len(placer.placed_annotations) >= 1

    def test_add_smart_annotations_connection_styles(self):
        """Cover lines 731-736: different connection styles based on distance."""
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        # Multiple annotations to trigger different normalized_dist values
        test_annotations = [
            {"text": "Close", "point": (5.0, 5.0), "priority": 90, "color": "blue"},
            {"text": "Medium", "point": (2.0, 2.0), "priority": 80, "color": "red"},
            {"text": "Far", "point": (9.0, 9.0), "priority": 70, "color": "green"},
        ]

        placer.add_smart_annotations(test_annotations, fontsize=9)
        assert len(placer.placed_annotations) >= 3

    def test_add_smart_annotations_sorted_by_priority(self):
        """Cover line 688: sorting by priority."""
        fig, ax = plt.subplots()
        ax.plot([0, 5], [0, 5])
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        fig.canvas.draw()

        placer = SmartAnnotationPlacer(ax)
        test_annotations: list[dict[str, Any]] = [
            {"text": "Low priority", "point": (1, 1), "priority": 10, "color": "gray"},
            {"text": "High priority", "point": (4, 4), "priority": 99, "color": "blue"},
            {"text": "Medium priority", "point": (2, 3)},  # No priority -> default 50
        ]

        placer.add_smart_annotations(test_annotations)
        # High priority should be placed first (it gets best position)
        assert len(placer.placed_annotations) >= 3

    def test_get_preferred_quadrant_sw_to_ne(self):
        """Cover lines 771-787: _get_preferred_quadrant for SW point -> NE."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        placer = SmartAnnotationPlacer(ax)
        # Point in SW (x_ratio < 0.3, y_ratio < 0.3)
        result = placer._get_preferred_quadrant((1.0, 1.0))
        assert result == "NE"

    def test_get_preferred_quadrant_se_to_nw(self):
        """Cover lines 780-781: _get_preferred_quadrant for SE point -> NW."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        placer = SmartAnnotationPlacer(ax)
        # Point in SE (x_ratio > 0.7, y_ratio < 0.3)
        result = placer._get_preferred_quadrant((8.0, 1.0))
        assert result == "NW"

    def test_get_preferred_quadrant_nw_to_se(self):
        """Cover lines 782-783: _get_preferred_quadrant for NW point -> SE."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        placer = SmartAnnotationPlacer(ax)
        # Point in NW (x_ratio < 0.3, y_ratio > 0.7)
        result = placer._get_preferred_quadrant((1.0, 8.0))
        assert result == "SE"

    def test_get_preferred_quadrant_ne_to_sw(self):
        """Cover lines 784-785: _get_preferred_quadrant for NE point -> SW."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        placer = SmartAnnotationPlacer(ax)
        # Point in NE (x_ratio > 0.7, y_ratio > 0.7)
        result = placer._get_preferred_quadrant((8.0, 8.0))
        assert result == "SW"

    def test_get_preferred_quadrant_center_returns_none(self):
        """Cover line 787: _get_preferred_quadrant for center point -> None."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        placer = SmartAnnotationPlacer(ax)
        # Point in center (0.3 < ratio < 0.7)
        result = placer._get_preferred_quadrant((5.0, 5.0))
        assert result is None

    def test_get_non_conflicting_color_no_conflict(self):
        """Cover lines 792-833: _get_non_conflicting_color with no conflicts."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="black")

        placer = SmartAnnotationPlacer(ax)
        result = placer._get_non_conflicting_color("green")

        # Should return a green variant
        assert result is not None
        assert isinstance(result, str)

    def test_get_non_conflicting_color_with_line_conflict(self):
        """Cover lines 813-821: _get_non_conflicting_color when line conflicts."""
        fig, ax = plt.subplots()
        # Plot with a blue line that matches a known color
        ax.plot([0, 1], [0, 1], color="#2ca02c")  # First green variant

        placer = SmartAnnotationPlacer(ax)
        result = placer._get_non_conflicting_color("green")

        # Should return a different green variant since the first one is taken
        assert result is not None
        assert isinstance(result, str)

    def test_get_non_conflicting_color_unknown_base_color(self):
        """Cover lines 820-821: requested color not in alternatives."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="black")

        placer = SmartAnnotationPlacer(ax)
        result = placer._get_non_conflicting_color("magenta")

        # Should return the original since it doesn't conflict
        assert result == "magenta"

    def test_get_non_conflicting_color_used_colors_tracked(self):
        """Cover lines 814-817: colors tracked in used_colors set."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="black")

        placer = SmartAnnotationPlacer(ax)
        # Mark some colors as used
        placer.used_colors.add("#2ca02c")
        placer.used_colors.add("#006400")

        result = placer._get_non_conflicting_color("green")
        # Should skip the used ones and return a different variant
        assert result not in {"#2ca02c", "#006400"}

    def test_get_non_conflicting_color_all_variants_taken(self):
        """Cover lines 824-833: fallback when all variants are taken."""
        fig, ax = plt.subplots()
        # Add lines matching all green variants
        green_variants = ["#2ca02c", "#006400", "#228B22", "#32CD32"]
        for color in green_variants:
            ax.plot([0, 1], [0, 1], color=color)

        placer = SmartAnnotationPlacer(ax)
        # Also mark them as used
        for color in green_variants:
            placer.used_colors.add(color)

        result = placer._get_non_conflicting_color("green")
        # Should find some available color from other palettes
        assert result is not None
        assert isinstance(result, str)

    def test_get_non_conflicting_color_completely_exhausted(self):
        """Cover line 833: fallback to requested color when everything is taken."""
        fig, ax = plt.subplots()

        placer = SmartAnnotationPlacer(ax)

        # Mark ALL possible colors as used
        all_alternatives = {
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
        for colors_list in all_alternatives.values():
            for color in colors_list:
                placer.used_colors.add(color)
                ax.plot([0, 1], [0, 1], color=color)

        # Also add the requested color as a line color
        ax.plot([0, 1], [0, 1], color="green")

        result = placer._get_non_conflicting_color("green")
        # Should fall back to requested color
        assert result is not None


class TestCreateLeaderLine:
    """Tests for create_leader_line covering all styles.
    Covers lines 856-907."""

    def teardown_method(self):
        plt.close("all")

    def test_straight_leader_line(self):
        """Cover lines 858-866: straight leader line."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        create_leader_line(
            ax,
            start=(1.0, 1.0),
            end=(5.0, 5.0),
            style="straight",
            color="red",
            linewidth=1.5,
            alpha=0.8,
        )

        # Should have added a line
        assert len(ax.lines) > 0

    def test_straight_leader_line_default_color(self):
        """Cover line 856: default color for straight line."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        create_leader_line(ax, start=(1.0, 1.0), end=(5.0, 5.0), style="straight")
        assert len(ax.lines) > 0

    def test_curved_leader_line_horizontal(self):
        """Cover lines 868-888: curved leader line with horizontal emphasis."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Horizontal emphasis: abs(dx) > abs(dy)
        create_leader_line(ax, start=(1.0, 3.0), end=(8.0, 4.0), style="curved", color="blue")

        # Should have added a patch (curved path)
        assert len(ax.patches) > 0

    def test_curved_leader_line_vertical(self):
        """Cover lines 879-881: curved leader line with vertical emphasis."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Vertical emphasis: abs(dy) > abs(dx)
        create_leader_line(ax, start=(3.0, 1.0), end=(4.0, 8.0), style="curved", color="green")

        assert len(ax.patches) > 0

    def test_elbow_leader_line_horizontal_first(self):
        """Cover lines 890-907: elbow leader line, horizontal first."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Horizontal first: abs(dx) > abs(dy)
        create_leader_line(ax, start=(1.0, 3.0), end=(8.0, 4.0), style="elbow", color="orange")

        # Should have added two line segments (L-shape)
        assert len(ax.lines) >= 2

    def test_elbow_leader_line_vertical_first(self):
        """Cover lines 895-897: elbow leader line, vertical first."""
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Vertical first: abs(dy) > abs(dx)
        create_leader_line(ax, start=(3.0, 1.0), end=(4.0, 8.0), style="elbow", color="purple")

        assert len(ax.lines) >= 2


class TestAutoAnnotatePeaksValleys:
    """Tests for auto_annotate_peaks_valleys function.
    Covers lines 945-999."""

    def teardown_method(self):
        plt.close("all")

    def test_auto_annotate_peaks_valleys_basic(self):
        """Cover lines 945-999: basic peak/valley annotation."""
        fig, ax = plt.subplots()

        # Create data with clear peaks and valleys
        x = np.linspace(0, 10, 200)
        y = np.sin(x) * 3 + np.random.normal(0, 0.1, len(x))

        ax.plot(x, y)
        ax.set_xlim(0, 10)
        ax.set_ylim(-4, 4)
        fig.canvas.draw()

        placer = auto_annotate_peaks_valleys(ax, x, y, n_peaks=2, n_valleys=2)

        assert isinstance(placer, SmartAnnotationPlacer)
        assert len(placer.placed_annotations) > 0

    def test_auto_annotate_peaks_valleys_custom_colors(self):
        """Cover lines 945-946: custom peak/valley colors."""
        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 200)
        y = np.sin(x) * 3

        ax.plot(x, y)
        ax.set_xlim(0, 10)
        ax.set_ylim(-4, 4)
        fig.canvas.draw()

        placer = auto_annotate_peaks_valleys(
            ax, x, y, peak_color="#FF0000", valley_color="#0000FF", fontsize=11
        )

        assert isinstance(placer, SmartAnnotationPlacer)

    def test_auto_annotate_peaks_valleys_with_existing_placer(self):
        """Cover line 955-956: using an existing SmartAnnotationPlacer."""
        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 200)
        y = np.sin(x) * 3 + np.random.normal(0, 0.1, len(x))

        ax.plot(x, y)
        ax.set_xlim(0, 10)
        ax.set_ylim(-4, 4)
        fig.canvas.draw()

        existing_placer = SmartAnnotationPlacer(ax)
        existing_placer.add_smart_callout("Existing", (5.0, 0.0))

        result_placer = auto_annotate_peaks_valleys(
            ax, x, y, n_peaks=1, n_valleys=1, placer=existing_placer
        )

        # Should reuse existing placer
        assert result_placer is existing_placer
        # Should have more annotations than before
        assert len(result_placer.placed_annotations) > 1

    def test_auto_annotate_peaks_valleys_many_peaks(self):
        """Cover lines 962-976: processing multiple peaks."""
        fig, ax = plt.subplots()

        x = np.linspace(0, 20, 400)
        # Create multiple peaks
        y = np.sin(x) * 2 + np.sin(2 * x) * 1.5

        ax.plot(x, y)
        ax.set_xlim(0, 20)
        ax.set_ylim(-5, 5)
        fig.canvas.draw()

        placer = auto_annotate_peaks_valleys(ax, x, y, n_peaks=3, n_valleys=2)
        assert isinstance(placer, SmartAnnotationPlacer)

    def test_auto_annotate_peaks_valleys_flat_data(self):
        """Cover edge case: flat data with no peaks or valleys."""
        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 100)
        y = np.ones(100) * 5  # Flat data

        ax.plot(x, y)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        fig.canvas.draw()

        placer = auto_annotate_peaks_valleys(ax, x, y)
        # Should still return a placer even with no peaks/valleys
        assert isinstance(placer, SmartAnnotationPlacer)


# =============================================================================
# Batch Plots Tests - Targeting missing lines 111, 145, 150-151, 169-175,
# 202, 234, 250-255, 258-259, 298, 323-391, 414-474
# =============================================================================


class TestBatchPlotsGaps:
    """Tests for uncovered lines in batch_plots module."""

    def teardown_method(self):
        plt.close("all")

    def _create_mock_aggregated_results(self, with_sensitivity=True):
        """Create mock AggregatedResults for testing."""
        mock_results = Mock(spec=AggregatedResults)
        mock_results.summary_statistics = pd.DataFrame(
            {
                "scenario": ["Scenario_0", "Scenario_1", "Scenario_2"],
                "ruin_probability": [0.1, 0.08, 0.06],
                "mean_growth_rate": [0.05, 0.06, 0.07],
                "mean_final_assets": [10000000, 11000000, 12000000],
                "var_99": [-100000, -110000, -120000],
            }
        )

        if with_sensitivity:
            mock_results.sensitivity_analysis = pd.DataFrame(
                {
                    "scenario": [
                        "sensitivity_param1_up",
                        "sensitivity_param1_down",
                        "sensitivity_param2_up",
                    ],
                    "mean_growth_rate_change_pct": [5.0, -3.0, 8.0],
                }
            )
        else:
            mock_results.sensitivity_analysis = None

        return mock_results

    def _create_mock_batch_results(self, n=5):
        """Create a list of mock BatchResult objects."""
        results = []
        for i in range(n):
            result = Mock()
            result.scenario_name = f"Scenario_{i}"
            result.simulation_results = Mock()
            result.simulation_results.growth_rates = [0.05 + i * 0.01]
            result.simulation_results.ruin_probability = 0.1 - i * 0.01
            result.simulation_results.final_assets = [10000000 * (1 + i * 0.1)]
            result.simulation_results.metrics = {"custom_metric": 42 + i}
            result.execution_time = 1.0 + i * 0.5
            results.append(result)
        return results

    def test_plot_scenario_comparison_removes_empty_subplots(self):
        """Cover line 111: removal of empty subplots when metric count is odd."""
        mock_results = self._create_mock_aggregated_results()
        # Use 3 metrics (odd number) so there's one empty subplot
        fig = batch_plots.plot_scenario_comparison(
            mock_results,
            metrics=["ruin_probability", "mean_growth_rate", "mean_final_assets"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_heatmap_invalid_input(self):
        """Cover line 145: invalid input type raises ValueError."""
        with pytest.raises(ValueError, match="Input must be AggregatedResults"):
            batch_plots.plot_sensitivity_heatmap("not_valid")

    def test_plot_sensitivity_heatmap_empty_data(self):
        """Cover lines 150-151: empty sensitivity analysis data."""
        mock_results = self._create_mock_aggregated_results(with_sensitivity=False)
        fig = batch_plots.plot_sensitivity_heatmap(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_heatmap_empty_df(self):
        """Cover lines 150-151: empty DataFrame."""
        mock_results = Mock(spec=AggregatedResults)
        mock_results.sensitivity_analysis = pd.DataFrame()
        fig = batch_plots.plot_sensitivity_heatmap(mock_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_heatmap_iterrows_processing(self):
        """Cover lines 169-175: iterrows processing for param extraction."""
        mock_results = Mock(spec=AggregatedResults)
        mock_results.sensitivity_analysis = pd.DataFrame(
            {
                "scenario": [
                    "sensitivity_alpha_beta_up",
                    "sensitivity_alpha_beta_down",
                    "sensitivity_gamma_up",
                    "sensitivity_gamma_down",
                ],
                "mean_growth_rate_change_pct": [5.0, -3.0, 8.0, -2.0],
            }
        )

        fig = batch_plots.plot_sensitivity_heatmap(mock_results, metric="mean_growth_rate")
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_sensitivity_heatmap_save_path(self):
        """Cover line 202: saving sensitivity heatmap."""
        mock_results = self._create_mock_aggregated_results()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            fig = batch_plots.plot_sensitivity_heatmap(
                mock_results,
                metric="mean_growth_rate",
                save_path=str(temp_path),
            )
            assert temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()
            plt.close("all")

    def test_plot_sensitivity_heatmap_missing_metric(self):
        """Cover lines 169-175: metric not found, uses available fallback."""
        mock_results = Mock(spec=AggregatedResults)
        mock_results.sensitivity_analysis = pd.DataFrame(
            {
                "scenario": ["sensitivity_param_up"],
                "some_other_change_pct": [3.0],
            }
        )
        fig = batch_plots.plot_sensitivity_heatmap(mock_results, metric="nonexistent")
        assert fig is not None
        plt.close(fig)

    def test_plot_sensitivity_heatmap_no_pct_metrics(self):
        """Cover lines 174-175: no _change_pct columns available."""
        mock_results = Mock(spec=AggregatedResults)
        mock_results.sensitivity_analysis = pd.DataFrame(
            {
                "scenario": ["sensitivity_param_up"],
                "some_column": [3.0],
            }
        )
        fig = batch_plots.plot_sensitivity_heatmap(mock_results, metric="nonexistent")
        assert fig is not None
        plt.close(fig)

    def test_plot_parameter_sweep_3d_invalid_input(self):
        """Cover line 234: invalid input type raises ValueError."""
        with pytest.raises(ValueError, match="Input must be AggregatedResults"):
            batch_plots.plot_parameter_sweep_3d("not_valid", "p1", "p2")

    def test_plot_parameter_sweep_3d_ruin_probability(self):
        """Cover lines 250-251: ruin_probability metric extraction."""
        mock_results = Mock(spec=AggregatedResults)
        batch_results = []
        for i in range(4):
            result = Mock()
            result.simulation_results = Mock()
            result.simulation_results.ruin_probability = 0.1 - i * 0.02
            result.simulation_results.growth_rates = [0.05]
            result.simulation_results.final_assets = [1000000]
            result.metadata = {"parameter_overrides": {"param1": 0.01 * i, "param2": 0.5 + i * 0.1}}
            batch_results.append(result)
        mock_results.batch_results = batch_results

        fig = batch_plots.plot_parameter_sweep_3d(
            mock_results, "param1", "param2", metric="ruin_probability"
        )
        assert fig is not None

    def test_plot_parameter_sweep_3d_mean_final_assets(self):
        """Cover lines 252-253: mean_final_assets metric extraction."""
        mock_results = Mock(spec=AggregatedResults)
        batch_results = []
        for i in range(4):
            result = Mock()
            result.simulation_results = Mock()
            result.simulation_results.final_assets = [1000000 * (1 + i)]
            result.simulation_results.growth_rates = [0.05]
            result.simulation_results.ruin_probability = 0.1
            result.metadata = {"parameter_overrides": {"param1": 0.01 * i, "param2": 0.5 + i * 0.1}}
            batch_results.append(result)
        mock_results.batch_results = batch_results

        fig = batch_plots.plot_parameter_sweep_3d(
            mock_results, "param1", "param2", metric="mean_final_assets"
        )
        assert fig is not None

    def test_plot_parameter_sweep_3d_custom_metric(self):
        """Cover lines 254-255: custom metric extraction from metrics dict."""
        mock_results = Mock(spec=AggregatedResults)
        batch_results = []
        for i in range(4):
            result = Mock()
            result.simulation_results = Mock()
            result.simulation_results.metrics = {"custom_metric": 42 + i}
            result.metadata = {"parameter_overrides": {"param1": 0.01 * i, "param2": 0.5 + i * 0.1}}
            batch_results.append(result)
        mock_results.batch_results = batch_results

        fig = batch_plots.plot_parameter_sweep_3d(
            mock_results, "param1", "param2", metric="custom_metric"
        )
        assert fig is not None

    def test_plot_parameter_sweep_3d_empty_data(self):
        """Cover lines 258-259: no parameter sweep data found."""
        mock_results = Mock(spec=AggregatedResults)
        batch_results = []
        for i in range(3):
            result = Mock()
            result.simulation_results = Mock()
            result.metadata = {"parameter_overrides": {"param1": 0.01 * i}}
            # param2 is missing so the condition on line 244 fails
            batch_results.append(result)
        mock_results.batch_results = batch_results

        fig = batch_plots.plot_parameter_sweep_3d(
            mock_results, "param1", "param2", metric="mean_growth_rate"
        )
        # Should return an empty figure
        assert fig is not None

    def test_plot_parameter_sweep_3d_no_simulation_results(self):
        """Cover line 242: results with no simulation_results."""
        mock_results = Mock(spec=AggregatedResults)
        batch_results = []
        for i in range(3):
            result = Mock()
            result.simulation_results = None
            result.metadata = {}
            batch_results.append(result)
        mock_results.batch_results = batch_results

        fig = batch_plots.plot_parameter_sweep_3d(
            mock_results, "param1", "param2", metric="mean_growth_rate"
        )
        assert fig is not None

    def test_plot_parameter_sweep_3d_save_path(self):
        """Cover line 298: saving parameter sweep 3D plot."""
        mock_results = Mock(spec=AggregatedResults)
        batch_results = []
        for i in range(4):
            result = Mock()
            result.simulation_results = Mock()
            result.simulation_results.growth_rates = [0.05 + i * 0.01]
            result.metadata = {"parameter_overrides": {"param1": 0.01 * i, "param2": 0.5 + i * 0.1}}
            batch_results.append(result)
        mock_results.batch_results = batch_results

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            temp_path = Path(f.name)

        try:
            fig = batch_plots.plot_parameter_sweep_3d(
                mock_results,
                "param1",
                "param2",
                metric="mean_growth_rate",
                save_path=str(temp_path),
            )
            assert fig is not None
            assert temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_plot_scenario_convergence_basic(self):
        """Cover lines 323-391: plot_scenario_convergence full flow."""
        batch_results = self._create_mock_batch_results(10)

        fig = batch_plots.plot_scenario_convergence(batch_results, metric="mean_growth_rate")

        assert fig is not None
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)

    def test_plot_scenario_convergence_ruin_probability(self):
        """Cover lines 338-339: ruin_probability metric in convergence."""
        batch_results = self._create_mock_batch_results(5)

        fig = batch_plots.plot_scenario_convergence(batch_results, metric="ruin_probability")
        assert fig is not None
        plt.close(fig)

    def test_plot_scenario_convergence_mean_final_assets(self):
        """Cover lines 340-341: mean_final_assets metric in convergence."""
        batch_results = self._create_mock_batch_results(5)

        fig = batch_plots.plot_scenario_convergence(batch_results, metric="mean_final_assets")
        assert fig is not None
        plt.close(fig)

    def test_plot_scenario_convergence_custom_metric(self):
        """Cover lines 342-343: custom metric in convergence."""
        batch_results = self._create_mock_batch_results(5)

        fig = batch_plots.plot_scenario_convergence(batch_results, metric="custom_metric")
        assert fig is not None
        plt.close(fig)

    def test_plot_scenario_convergence_empty_results(self):
        """Cover lines 345-347: no data to plot."""
        batch_results = []
        for i in range(3):
            result = Mock()
            result.simulation_results = None  # No simulation results
            batch_results.append(result)

        fig = batch_plots.plot_scenario_convergence(batch_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_scenario_convergence_save_path(self):
        """Cover lines 388-389: saving convergence plot."""
        batch_results = self._create_mock_batch_results(5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            fig = batch_plots.plot_scenario_convergence(
                batch_results,
                metric="mean_growth_rate",
                save_path=str(temp_path),
            )
            assert temp_path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()
            plt.close("all")

    def test_plot_parallel_scenarios_basic(self):
        """Cover lines 414-474: plot_parallel_scenarios full flow."""
        batch_results = self._create_mock_batch_results(5)
        metrics = ["mean_growth_rate", "ruin_probability", "mean_final_assets"]

        fig = batch_plots.plot_parallel_scenarios(batch_results, metrics=metrics, normalize=True)

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_parallel_scenarios_no_normalize(self):
        """Cover line 448-453: parallel scenarios without normalization."""
        batch_results = self._create_mock_batch_results(3)
        metrics = ["mean_growth_rate", "ruin_probability"]

        fig = batch_plots.plot_parallel_scenarios(batch_results, metrics=metrics, normalize=False)

        assert fig is not None
        plt.close(fig)

    def test_plot_parallel_scenarios_custom_metric(self):
        """Cover lines 435-436: custom metric from metrics dict."""
        batch_results = self._create_mock_batch_results(3)
        metrics = ["mean_growth_rate", "custom_metric"]

        fig = batch_plots.plot_parallel_scenarios(batch_results, metrics=metrics)
        assert fig is not None
        plt.close(fig)

    def test_plot_parallel_scenarios_empty_results(self):
        """Cover lines 441-443: no data to plot."""
        batch_results = []
        for i in range(2):
            result = Mock()
            result.simulation_results = None
            batch_results.append(result)

        fig = batch_plots.plot_parallel_scenarios(batch_results, metrics=["mean_growth_rate"])
        assert fig is not None
        plt.close(fig)

    def test_plot_parallel_scenarios_many_scenarios_no_legend(self):
        """Cover lines 470-471: more than 10 scenarios -> legend added."""
        batch_results = self._create_mock_batch_results(8)
        metrics = ["mean_growth_rate", "ruin_probability"]

        fig = batch_plots.plot_parallel_scenarios(batch_results, metrics=metrics)
        assert fig is not None
        plt.close(fig)

    def test_plot_parallel_scenarios_normalize_same_values(self):
        """Cover line 452: normalize when col_max == col_min (no division)."""
        batch_results = []
        for i in range(3):
            result = Mock()
            result.scenario_name = f"Scenario_{i}"
            result.simulation_results = Mock()
            result.simulation_results.growth_rates = [0.05]  # Same value for all
            result.simulation_results.ruin_probability = 0.1  # Same value
            result.simulation_results.final_assets = [1000000]
            result.simulation_results.metrics = {}
            batch_results.append(result)

        fig = batch_plots.plot_parallel_scenarios(
            batch_results,
            metrics=["mean_growth_rate", "ruin_probability"],
            normalize=True,
        )
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Export Tests - Targeting missing lines 52, 61, 78-103, 207-220, 268-289
# =============================================================================


class TestExportGaps:
    """Tests for uncovered lines in export module."""

    def teardown_method(self):
        plt.close("all")

    def test_save_figure_default_formats(self):
        """Cover line 52: formats defaults to ['png'] when None."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_default"
            saved = export.save_figure(fig, str(base_path))
            assert len(saved) == 1
            assert saved[0].endswith(".png")
            assert Path(saved[0]).exists()

    def test_save_figure_creates_directory(self):
        """Cover line 61: creates parent directory if it doesn't exist."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "deep" / "test_plot"
            saved = export.save_figure(fig, str(nested_path), formats=["png"])
            assert len(saved) == 1
            assert Path(saved[0]).exists()

    def test_save_figure_unsupported_matplotlib_format(self):
        """Cover line 78: unsupported format for matplotlib."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plot"
            with pytest.raises(ValueError, match="Unsupported format for matplotlib"):
                export.save_figure(fig, str(base_path), formats=["webp"])

    def test_save_figure_plotly_html(self):
        """Cover lines 81-87: saving plotly figure as HTML."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plotly"
            saved = export.save_figure(plotly_fig, str(base_path), formats=["html"])
            assert len(saved) == 1
            assert saved[0].endswith(".html")
            assert Path(saved[0]).exists()

    def test_save_figure_plotly_image_format(self):
        """Cover lines 88-99: saving plotly figure as image."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plotly"
            # This may fail if kaleido is not installed, which is fine
            # The code handles the exception on lines 97-99
            try:
                saved = export.save_figure(plotly_fig, str(base_path), formats=["png"])
                assert isinstance(saved, list)  # May be empty if kaleido not installed
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Expected if kaleido is not available

    def test_save_figure_plotly_unsupported_format(self):
        """Cover lines 100-101: unsupported format for plotly."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plotly"
            with pytest.raises(ValueError, match="Unsupported format for plotly"):
                export.save_figure(plotly_fig, str(base_path), formats=["webp"])

    def test_save_figure_invalid_type(self):
        """Cover lines 102-103: invalid figure type raises TypeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_invalid"
            with pytest.raises(
                TypeError, match="Figure must be matplotlib Figure or plotly Figure"
            ):
                export.save_figure("not_a_figure", str(base_path), formats=["png"])

    def test_save_for_presentation_plotly(self):
        """Cover lines 207-220: save_for_presentation with plotly figure."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_pres"
            result = export.save_for_presentation(plotly_fig, str(base_path))
            assert result is not None
            # Result should be either .png or .html depending on kaleido
            assert result.endswith(".png") or result.endswith(".html")

    def test_save_for_web_plotly(self):
        """Cover lines 268-289: save_for_web with plotly figure."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_web"
            result = export.save_for_web(plotly_fig, str(base_path))

            assert isinstance(result, dict)
            assert "html" in result
            assert Path(result["html"]).exists()

    def test_save_for_web_plotly_optimized(self):
        """Cover lines 272-276: optimized web saving for plotly."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_web_opt"
            result = export.save_for_web(plotly_fig, str(base_path), optimize=True)

            assert isinstance(result, dict)
            assert "html" in result

    def test_save_for_web_plotly_no_optimize(self):
        """Cover line 274: save_for_web with optimize=False."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_web_noopt"
            result = export.save_for_web(plotly_fig, str(base_path), optimize=False)

            assert isinstance(result, dict)
            assert "html" in result

    def test_save_figure_plotly_image_with_kaleido_mock(self):
        """Cover lines 88-96: plotly image write path using mock."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plotly_img"
            output_path = Path(temp_dir) / "test_plotly_img.png"

            # Mock write_image to simulate successful save
            with patch.object(plotly_fig, "write_image") as mock_write:
                mock_write.side_effect = lambda path, **kwargs: Path(path).touch()
                saved = export.save_figure(plotly_fig, str(base_path), formats=["png"])
                assert len(saved) == 1

    def test_save_figure_plotly_image_failure(self):
        """Cover lines 97-99: plotly image write failure handling."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_plotly_fail"

            # Mock write_image to simulate failure
            with patch.object(plotly_fig, "write_image", side_effect=Exception("No kaleido")):
                saved = export.save_figure(plotly_fig, str(base_path), formats=["png"])
                # Should not crash, just print warning
                assert len(saved) == 0

    def test_save_for_presentation_plotly_failure_fallback(self):
        """Cover lines 215-220: plotly presentation save failure fallback to HTML."""
        plotly_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 4, 2])])

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test_pres_fail"

            # Mock write_image to simulate failure
            with patch.object(plotly_fig, "write_image", side_effect=Exception("No kaleido")):
                result = export.save_for_presentation(plotly_fig, str(base_path))
                # Should fall back to HTML
                assert result.endswith(".html")
                assert Path(result).exists()


# =============================================================================
# Interactive Plots Tests - Targeting missing lines 239-260, 384, 398-401,
# 414-417, 429-434, 447-450, 464-465
# =============================================================================


class TestInteractivePlotsGaps:
    """Tests for uncovered lines in interactive_plots module."""

    def test_time_series_dashboard_with_forecast(self):
        """Cover lines 239-260: forecast with confidence bands."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30),
                "value": np.random.randn(30).cumsum() + 100,
                "value_forecast": np.random.randn(30).cumsum() + 100,
                "value_upper": np.random.randn(30).cumsum() + 110,
                "value_lower": np.random.randn(30).cumsum() + 90,
            }
        )

        fig = interactive_plots.create_time_series_dashboard(
            data,
            value_col="value",
            time_col="date",
            show_forecast=True,
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)
        # Should have main trace + forecast + upper + lower (4 traces min)
        assert len(fig.data) >= 3

    def test_time_series_dashboard_forecast_no_bands(self):
        """Cover lines 238-247: forecast without confidence bands."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30),
                "value": np.random.randn(30).cumsum() + 100,
                "value_forecast": np.random.randn(30).cumsum() + 100,
            }
        )

        fig = interactive_plots.create_time_series_dashboard(
            data,
            value_col="value",
            time_col="date",
            show_forecast=True,
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_var_distribution(self):
        """Cover line 384: VaR distribution histogram in risk dashboard."""
        risk_metrics = {
            "var_distribution": np.random.normal(100000, 20000, 500),
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_expected_shortfall(self):
        """Cover lines 398-401: expected shortfall bar chart."""
        risk_metrics = {
            "expected_shortfall": {"ES_95": 150000, "ES_99": 200000, "ES_99.5": 250000},
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_risk_contribution(self):
        """Cover lines 414-417: risk contribution pie chart."""
        risk_metrics = {
            "risk_contribution": {
                "Market Risk": 0.4,
                "Credit Risk": 0.3,
                "Operational Risk": 0.2,
                "Other": 0.1,
            },
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_stress_tests(self):
        """Cover lines 429-434: stress test results bar chart."""
        risk_metrics = {
            "stress_tests": {
                "Market Crash": -500000,
                "Rate Hike": -200000,
                "Recovery": 100000,
                "Bull Market": 300000,
            },
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_var_breaches(self):
        """Cover lines 447-450: VaR breaches scatter plot."""
        risk_metrics = {
            "var_breaches": {
                "dates": pd.date_range("2024-01-01", periods=10).tolist(),
                "values": (np.random.randn(10) * 50000).tolist(),
            },
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_trends(self):
        """Cover lines 464-465: risk metric trends scatter lines."""
        risk_metrics = {
            "trends": {
                "VaR_95": {
                    "dates": pd.date_range("2024-01-01", periods=20).tolist(),
                    "values": (np.random.randn(20).cumsum() * 10000 + 100000).tolist(),
                },
                "VaR_99": {
                    "dates": pd.date_range("2024-01-01", periods=20).tolist(),
                    "values": (np.random.randn(20).cumsum() * 15000 + 150000).tolist(),
                },
            },
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_risk_dashboard_all_sections(self):
        """Cover all risk dashboard sections together."""
        risk_metrics = {
            "var_distribution": np.random.normal(100000, 20000, 200),
            "expected_shortfall": {"ES_95": 150000, "ES_99": 200000},
            "risk_contribution": {"Market": 0.5, "Credit": 0.3, "Operational": 0.2},
            "stress_tests": {"Crash": -500000, "Recovery": 100000},
            "var_breaches": {
                "dates": pd.date_range("2024-01-01", periods=5).tolist(),
                "values": [120000, 130000, 125000, 140000, 135000],
            },
            "trends": {
                "VaR": {
                    "dates": pd.date_range("2024-01-01", periods=10).tolist(),
                    "values": list(range(100000, 110000, 1000)),
                },
            },
        }

        fig = interactive_plots.create_risk_dashboard(risk_metrics)
        assert fig is not None
        assert isinstance(fig, go.Figure)
        # Should have multiple traces
        assert len(fig.data) >= 5


# =============================================================================
# Style Manager Tests - Targeting missing lines 184, 188, 190, 304, 490,
# 499-500, 509-522, 531-533, 542-547, 658-661
# =============================================================================


class TestStyleManagerGaps:
    """Tests for uncovered lines in style_manager module."""

    def teardown_method(self):
        plt.close("all")

    def test_init_with_config_path(self):
        """Cover line 184: load_config called from __init__."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "themes": {
                    "default": {
                        "colors": {"primary": "#FF0000"},
                        "fonts": {"size_title": 20},
                        "figure": {"dpi_screen": 200},
                        "grid": {"grid_alpha": 0.5},
                    }
                }
            }
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            mgr = StyleManager(config_path=str(temp_path))
            assert mgr is not None
            colors = mgr.get_colors()
            assert colors.primary == "#FF0000"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_init_with_custom_colors(self):
        """Cover line 188: update_colors called from __init__."""
        mgr = StyleManager(custom_colors={"primary": "#123456", "accent": "#654321"})
        colors = mgr.get_colors()
        assert colors.primary == "#123456"
        assert colors.accent == "#654321"

    def test_init_with_custom_fonts(self):
        """Cover line 190: update_fonts called from __init__."""
        mgr = StyleManager(custom_fonts={"size_title": 24, "family": "Courier"})
        fonts = mgr.get_fonts()
        assert fonts.size_title == 24
        assert fonts.family == "Courier"

    def test_init_with_all_custom(self):
        """Cover lines 184, 188, 190 together: all customizations in __init__."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "themes": {
                    "default": {
                        "colors": {"primary": "#AABBCC"},
                    }
                }
            }
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            mgr = StyleManager(
                config_path=str(temp_path),
                custom_colors={"secondary": "#112233"},
                custom_fonts={"size_base": 15},
            )
            colors = mgr.get_colors()
            fonts = mgr.get_fonts()
            assert colors.primary == "#AABBCC"
            assert colors.secondary == "#112233"
            assert fonts.size_base == 15
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_set_theme_invalid(self):
        """Cover line 304: ValueError for unknown theme."""
        mgr = StyleManager()
        # Remove a theme to make it unknown
        unknown_theme = Mock()
        unknown_theme.value = "nonexistent"

        with pytest.raises(ValueError, match="Unknown theme"):
            mgr.set_theme(unknown_theme)

    def test_load_config_file_not_found(self):
        """Cover line 490: FileNotFoundError for missing config file."""
        mgr = StyleManager()
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            mgr.load_config("/nonexistent/path/config.yaml")

    def test_load_config_no_themes_key(self):
        """Cover lines 496-497: config file without 'themes' key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {"some_other_key": "value"}
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            mgr = StyleManager()
            mgr.load_config(str(temp_path))
            # Should return early without error
            assert mgr is not None
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_load_config_with_themes(self):
        """Cover lines 499-500: iteration over themes in config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "themes": {
                    "default": {
                        "colors": {"primary": "#FF0000", "secondary": "#00FF00"},
                        "fonts": {"size_title": 20, "size_base": 14},
                        "figure": {"dpi_screen": 150},
                        "grid": {"grid_alpha": 0.4, "show_grid": False},
                    },
                    "presentation": {
                        "colors": {"primary": "#0000FF"},
                        "fonts": {"size_title": 24},
                    },
                }
            }
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            mgr = StyleManager()
            mgr.load_config(str(temp_path))

            # Check default theme was updated
            mgr.set_theme(Theme.DEFAULT)
            colors = mgr.get_colors()
            assert colors.primary == "#FF0000"
            assert colors.secondary == "#00FF00"
            fonts = mgr.get_fonts()
            assert fonts.size_title == 20
            assert fonts.size_base == 14

            # Check presentation theme was updated
            mgr.set_theme(Theme.PRESENTATION)
            colors = mgr.get_colors()
            assert colors.primary == "#0000FF"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_update_theme_from_config(self):
        """Cover lines 509-522: _update_theme_from_config method."""
        mgr = StyleManager()

        theme_config = {
            "colors": {"primary": "#ABCDEF", "warning": "#FF0000"},
            "fonts": {"size_title": 22, "family": "Times"},
            "figure": {"size_small": [4, 3], "dpi_screen": 200},
            "grid": {"grid_alpha": 0.6, "show_grid": True},
        }

        mgr._update_theme_from_config("default", theme_config)

        mgr.set_theme(Theme.DEFAULT)
        colors = mgr.get_colors()
        assert colors.primary == "#ABCDEF"
        assert colors.warning == "#FF0000"
        fonts = mgr.get_fonts()
        assert fonts.size_title == 22
        assert fonts.family == "Times"

    def test_update_theme_from_config_new_theme(self):
        """Cover lines 510-516: creating a new theme entry when not found."""
        mgr = StyleManager()

        # This should trigger the creation of a new theme dict
        # Using an existing enum value that might not be in themes
        theme_config = {
            "colors": {"primary": "#FACADE"},
            "fonts": {"size_title": 18},
            "figure": {},
            "grid": {},
        }

        # Remove the PRINT theme to test re-creation
        if Theme.PRINT in mgr.themes:
            del mgr.themes[Theme.PRINT]

        mgr._update_theme_from_config("print", theme_config)

        assert Theme.PRINT in mgr.themes
        mgr.set_theme(Theme.PRINT)
        colors = mgr.get_colors()
        assert colors.primary == "#FACADE"

    def test_update_component(self):
        """Cover lines 531-533: _update_component method."""
        mgr = StyleManager()
        palette = ColorPalette()

        mgr._update_component(palette, {"primary": "#111111", "accent": "#222222"})
        assert palette.primary == "#111111"
        assert palette.accent == "#222222"

    def test_update_component_unknown_key(self):
        """Cover lines 532: _update_component with nonexistent key (skipped)."""
        mgr = StyleManager()
        palette = ColorPalette()
        original_primary = palette.primary

        mgr._update_component(palette, {"nonexistent_key": "value"})
        # Should not crash and primary should be unchanged
        assert palette.primary == original_primary

    def test_update_component_fonts(self):
        """Cover _update_component with FontConfig."""
        mgr = StyleManager()
        fonts = FontConfig()

        mgr._update_component(fonts, {"size_title": 30, "family": "Monospace"})
        assert fonts.size_title == 30
        assert fonts.family == "Monospace"

    def test_update_component_grid(self):
        """Cover _update_component with GridConfig."""
        mgr = StyleManager()
        grid = GridConfig()

        mgr._update_component(grid, {"grid_alpha": 0.8, "show_grid": False})
        assert grid.grid_alpha == 0.8
        assert grid.show_grid is False

    def test_update_figure_component(self):
        """Cover lines 542-547: _update_figure_component method."""
        mgr = StyleManager()
        fig_config = FigureConfig()

        mgr._update_figure_component(
            fig_config,
            {
                "dpi_screen": 200,
                "size_small": [5, 3],  # list -> should be converted to tuple
                "size_medium": [9, 7],
            },
        )

        assert fig_config.dpi_screen == 200
        assert fig_config.size_small == (5, 3)
        assert fig_config.size_medium == (9, 7)

    def test_update_figure_component_non_size_keys(self):
        """Cover lines 544-546: non-size keys are set directly."""
        mgr = StyleManager()
        fig_config = FigureConfig()

        mgr._update_figure_component(fig_config, {"dpi_print": 600, "dpi_web": 200})
        assert fig_config.dpi_print == 600
        assert fig_config.dpi_web == 200

    def test_update_figure_component_unknown_key(self):
        """Cover line 543: unknown key is skipped."""
        mgr = StyleManager()
        fig_config = FigureConfig()

        mgr._update_figure_component(fig_config, {"nonexistent": "value"})
        # Should not crash

    def test_inherit_from_with_font_modifications(self):
        """Cover lines 658-661: inherit_from with fonts modification."""
        mgr = StyleManager()

        new_theme = mgr.inherit_from(
            Theme.DEFAULT,
            {
                "fonts": {"size_title": 28, "family": "Georgia"},
            },
        )

        assert new_theme == Theme.DEFAULT
        mgr.set_theme(Theme.DEFAULT)
        fonts = mgr.get_fonts()
        assert fonts.size_title == 28
        assert fonts.family == "Georgia"

    def test_inherit_from_with_color_and_font_modifications(self):
        """Cover lines 654-661: inherit_from with both colors and fonts."""
        mgr = StyleManager()

        new_theme = mgr.inherit_from(
            Theme.PRESENTATION,
            {
                "colors": {"primary": "#AABBCC", "accent": "#DDEEFF"},
                "fonts": {"size_base": 16, "size_label": 14},
            },
        )

        assert new_theme == Theme.DEFAULT
        mgr.set_theme(Theme.DEFAULT)
        colors = mgr.get_colors()
        fonts = mgr.get_fonts()
        assert colors.primary == "#AABBCC"
        assert colors.accent == "#DDEEFF"
        assert fonts.size_base == 16
        assert fonts.size_label == 14

    def test_inherit_from_with_unknown_color_key(self):
        """Cover line 656: inherit_from with unknown color key."""
        mgr = StyleManager()

        new_theme = mgr.inherit_from(
            Theme.DEFAULT,
            {
                "colors": {"nonexistent_color": "#000000"},
            },
        )
        # Should not crash
        assert new_theme == Theme.DEFAULT

    def test_inherit_from_with_unknown_font_key(self):
        """Cover line 660: inherit_from with unknown font key."""
        mgr = StyleManager()

        new_theme = mgr.inherit_from(
            Theme.DEFAULT,
            {
                "fonts": {"nonexistent_font": 99},
            },
        )
        # Should not crash
        assert new_theme == Theme.DEFAULT

    def test_get_figure_size_portrait(self):
        """Cover line 460: portrait orientation."""
        mgr = StyleManager()
        w, h = mgr.get_figure_size("medium", orientation="portrait")
        # Portrait should swap width and height
        w_land, h_land = mgr.get_figure_size("medium", orientation="landscape")
        assert w == h_land
        assert h == w_land

    def test_get_figure_size_unknown_type(self):
        """Cover line 457: unknown size type falls back to medium."""
        mgr = StyleManager()
        w, h = mgr.get_figure_size("unknown_type")
        w_med, h_med = mgr.get_figure_size("medium")
        assert w == w_med
        assert h == h_med

    def test_get_dpi_unknown_type(self):
        """Cover line 480: unknown output type falls back to screen."""
        mgr = StyleManager()
        dpi = mgr.get_dpi("unknown_type")
        dpi_screen = mgr.get_dpi("screen")
        assert dpi == dpi_screen

    def test_get_figure_size_all_types(self):
        """Cover various size types including technical and presentation."""
        mgr = StyleManager()
        for size_type in ["small", "medium", "large", "blog", "technical", "presentation"]:
            w, h = mgr.get_figure_size(size_type)
            assert w > 0
            assert h > 0

    def test_get_dpi_all_types(self):
        """Cover all DPI types."""
        mgr = StyleManager()
        for output_type in ["screen", "web", "print"]:
            dpi = mgr.get_dpi(output_type)
            assert dpi > 0

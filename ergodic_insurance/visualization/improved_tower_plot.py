"""Improved insurance tower visualization with stacked bar chart and smart annotations."""

from typing import Dict, List, Optional, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_insurance_tower(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    layers: Union[List[Dict[str, float]], pd.DataFrame],
    title: str = "Insurance Tower Structure",
    figsize: Tuple[int, int] = (10, 12),
    min_height_for_text: float = 0.02,  # Minimum relative height to display text inside (lower for log scale)
    color_scheme: str = "viridis",
    show_summary: bool = True,
    log_scale: bool = True,  # Use logarithmic scale for better layer visibility
) -> Figure:
    """Create an improved insurance tower visualization with stacked bars and smart annotations.

    Args:
        layers: List of layer dictionaries or DataFrame with columns:
            - attachment: Layer attachment point
            - limit: Layer limit
            - premium: Premium amount or rate
            - expected_loss (optional): Expected loss for the layer
            - rate_on_line (optional): Rate on line for the layer
        title: Plot title
        figsize: Figure size (width, height)
        min_height_for_text: Minimum relative height (as fraction of total) to display text inside bar
        color_scheme: Matplotlib colormap name
        show_summary: Whether to show summary statistics
        log_scale: Use logarithmic scale for y-axis to better show layers of different sizes

    Returns:
        Matplotlib figure with improved tower visualization
    """

    # Convert DataFrame to list of dicts if needed
    if isinstance(layers, pd.DataFrame):
        layer_list = []
        for _, row in layers.iterrows():
            layer_dict = {
                "attachment": row.get("attachment", 0),
                "limit": row.get("limit", 0),
                "premium": row.get("premium", 0),
                "expected_loss": row.get("expected_loss", None),
                "rate_on_line": row.get("rate_on_line", None),
            }
            layer_list.append(layer_dict)
        layers = layer_list

    if not layers:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No layers defined",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig

    # Calculate total height for normalization
    total_limit = max(layer["attachment"] + layer["limit"] for layer in layers)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Generate colors
    cmap = plt.cm.get_cmap(color_scheme)
    colors = [cmap(0.3 + 0.5 * i / len(layers)) for i in range(len(layers))]

    # Track annotations for layers that are too small
    external_annotations = []

    # Plot stacked bars (vertical)
    bar_width = 0.6
    bar_center = 0.5

    # If using log scale and first layer starts at 0, we need to handle it specially
    if log_scale:
        # Find minimum non-zero value for log scale base
        non_zero_attachments = [layer["attachment"] for layer in layers if layer["attachment"] > 0]
        if non_zero_attachments:
            min_nonzero = min(non_zero_attachments)
        else:
            # If all attachments are 0, use smallest limit
            min_nonzero = min(layer["limit"] for layer in layers) / 10

        # Set a reasonable minimum for display
        log_scale_min = min_nonzero / 100
    else:
        log_scale_min = 0

    for i, (layer, color) in enumerate(zip(layers, colors)):
        attachment = layer["attachment"]
        limit = layer["limit"]

        # For log scale, we need to handle zero attachment specially
        if log_scale and attachment == 0:
            # Display the bar from a small positive value up to the limit
            display_bottom = log_scale_min
            display_height = limit - log_scale_min

            rect = ax.bar(
                bar_center,
                display_height,
                bottom=display_bottom,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            # Store actual values for text positioning
            actual_attachment = 0.0
            actual_limit = limit
        else:
            # Create the vertical bar normally
            rect = ax.bar(
                bar_center,
                limit,
                bottom=attachment,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=2,
                alpha=0.8,
            )
            actual_attachment = attachment
            actual_limit = limit

        # Determine if text fits inside
        # For log scale, check visual height rather than absolute height
        if log_scale:
            if actual_attachment == 0:
                visual_bottom = log_scale_min
                visual_top = actual_limit
            else:
                visual_bottom = actual_attachment
                visual_top = actual_attachment + actual_limit
            # Check if visual height is sufficient (log scale expands small layers)
            if visual_top > 0 and visual_bottom > 0:
                visual_height = np.log10(visual_top) - np.log10(visual_bottom)
                total_visual_height = np.log10(total_limit) - np.log10(log_scale_min)
                relative_height = (
                    visual_height / total_visual_height if total_visual_height > 0 else 0
                )
            else:
                relative_height = actual_limit / total_limit
        else:
            relative_height = actual_limit / total_limit

        # Format layer information
        layer_num = i + 1
        attach_str = _format_currency(actual_attachment)
        limit_str = _format_currency(actual_limit)

        # Build layer label (compact for better fit)
        label_parts = [f"Layer {layer_num}"]
        label_parts.append(f"{limit_str} xs {attach_str}")

        # Add additional info only if there's room
        if relative_height > 0.08:  # Only add extra info if plenty of room
            if layer.get("rate_on_line") is not None:
                label_parts.append(f"RoL: {layer['rate_on_line']:.2%}")
            elif layer.get("premium") is not None and layer["premium"] > 0:
                if layer["premium"] < 1:  # Assume it's a rate
                    label_parts.append(f"Premium: {layer['premium']:.2%}")
                else:
                    label_parts.append(f"Premium: {_format_currency(layer['premium'])}")

        if relative_height >= min_height_for_text:
            # Text fits inside - place it in the center of the bar
            label = "\n".join(label_parts)
            # For log scale, use geometric mean for center position
            if log_scale:
                if actual_attachment == 0:
                    # For base layer, position text in geometric center of visible bar
                    text_y = (
                        np.sqrt(log_scale_min * actual_limit)
                        if actual_limit > 0
                        else actual_limit / 2
                    )
                else:
                    # For other layers, geometric mean of top and bottom
                    text_y = np.sqrt(actual_attachment * (actual_attachment + actual_limit))
            else:
                text_y = actual_attachment + actual_limit / 2

            ax.text(
                bar_center,
                text_y,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.3},
                rotation=0,  # Keep text horizontal
            )
        else:
            # Text doesn't fit - store for external annotation
            external_annotations.append(
                {
                    "layer_num": layer_num,
                    "attachment": actual_attachment,
                    "limit": actual_limit,
                    "label_parts": label_parts,
                    "color": color,
                }
            )

    # Add external annotations for small layers (alternate left and right)
    for i, ann in enumerate(external_annotations):
        # Find the midpoint of the layer
        if log_scale:
            if ann["attachment"] == 0:
                # For base layer, use geometric mean with the log scale minimum
                layer_midpoint = np.sqrt(log_scale_min * ann["limit"])
            else:
                # For other layers, geometric mean of top and bottom
                layer_midpoint = np.sqrt(ann["attachment"] * (ann["attachment"] + ann["limit"]))
        else:
            layer_midpoint = ann["attachment"] + ann["limit"] / 2

        # Create annotation text (more compact for log scale)
        if log_scale:
            # In log scale, usually more room, so can be more detailed
            label = f"Layer {ann['layer_num']}: {_format_currency(ann['limit'])} xs {_format_currency(ann['attachment'])}"
            if len(ann["label_parts"]) > 2:
                label += "\n" + " • ".join(ann["label_parts"][2:])
        else:
            label = " • ".join(ann["label_parts"][:2])
            if len(ann["label_parts"]) > 2:
                label += "\n" + " • ".join(ann["label_parts"][2:])

        # Alternate between left and right sides
        if i % 2 == 0:
            # Left side annotation
            x_text = 0.15
            ha = "left"
        else:
            # Right side annotation
            x_text = 0.85
            ha = "right"

        # Add annotation with arrow
        ax.annotate(
            label,
            xy=(bar_center, layer_midpoint),
            xytext=(x_text, layer_midpoint),
            xycoords="data",
            textcoords=("axes fraction", "data"),
            fontsize=8,
            ha=ha,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": ann["color"],
                "alpha": 0.7,
                "edgecolor": "white",
            },
            arrowprops={
                "arrowstyle": "->",
                "connectionstyle": "arc3,rad=0.2",
                "color": "gray",
                "linestyle": "--",
                "alpha": 0.7,
                "linewidth": 1,
            },
        )

    # Customize plot
    if log_scale:
        ax.set_yscale("log")
        # For log scale, set limits with a small buffer
        ax.set_ylim(
            log_scale_min / 2, total_limit * 1.5 if external_annotations else total_limit * 1.1
        )
    else:
        ax.set_ylim(0, total_limit * 1.3 if external_annotations else total_limit * 1.05)

    ax.set_xlim(0, 1)
    ax.set_ylabel(
        "Coverage Amount (Log Scale)" if log_scale else "Coverage Amount",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_currency_axis(x)))

    # Remove x-axis ticks
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Add grid for y-axis
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both" if log_scale else "major")
    if log_scale:
        ax.grid(True, axis="y", alpha=0.15, linestyle=":", which="minor")

    # Add horizontal lines at attachment points
    for layer in layers:
        if layer["attachment"] > 0 or not log_scale:  # Skip zero on log scale
            ax.axhline(y=layer["attachment"], color="gray", alpha=0.3, linestyle=":")
    ax.axhline(y=total_limit, color="gray", alpha=0.3, linestyle=":")

    # Add summary box if requested (positioned to avoid bar overlap)
    if show_summary:
        summary_text = _create_summary_text(layers)
        # Position summary box to the left side, away from the center bar
        ax.text(
            0.02,
            0.85,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "alpha": 0.95,
                "edgecolor": "gray",
            },
        )

    # Remove legend - it's not helpful for tower visualizations

    plt.tight_layout()
    return fig


def _format_currency(value: Union[float, str]) -> str:
    """Format currency values with appropriate units."""
    # Handle string inputs that might already be formatted
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return str(value)  # Return as-is if can't convert

    if value >= 1e9:
        return f"{value/1e9:.1f}B"
    if value >= 1e6:
        return f"{value/1e6:.1f}M"
    if value >= 1e3:
        return f"{value/1e3:.0f}K"
    return f"{value:.0f}"


def _format_currency_axis(value: float) -> str:
    """Format currency values for axis labels."""
    if value == 0:
        return "$0"
    if value >= 1e9:
        return f"${value/1e9:.1f}B"
    if value >= 1e6:
        return f"${value/1e6:.0f}M"
    if value >= 1e3:
        return f"${value/1e3:.0f}K"
    return f"${value:.0f}"


def _create_summary_text(layers: List[Dict[str, float]]) -> str:
    """Create summary statistics text."""
    total_limit = max(layer["attachment"] + layer["limit"] for layer in layers)
    total_premium = sum(
        layer.get("premium", 0) * (1 if layer.get("premium", 0) >= 1 else layer["limit"])
        for layer in layers
    )
    total_expected = sum(layer.get("expected_loss", 0) for layer in layers)

    lines = [
        "Tower Summary:",
        f"Total Limit: {_format_currency(total_limit)}",
        f"Number of Layers: {len(layers)}",
    ]

    if total_premium > 0:
        lines.append(f"Total Premium: {_format_currency(total_premium)}")

    if total_expected > 0:
        lines.append(f"Total E[Loss]: {_format_currency(total_expected)}")
        if total_premium > 0:
            lines.append(f"Loss Ratio: {(total_expected/total_premium)*100:.1f}%")

    return "\n".join(lines)

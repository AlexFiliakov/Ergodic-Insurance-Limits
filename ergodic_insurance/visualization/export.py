"""Export utilities for saving visualizations in various formats.

This module provides functions to export visualizations to different formats
including high-resolution images, PDFs, and web-ready formats.

.. versionchanged:: 0.7.0
    Replaced bare ``print()`` warning calls with ``logging.warning()``.
    See :issue:`382`.
"""

__all__ = [
    "save_figure",
    "save_for_publication",
    "save_for_presentation",
    "save_for_web",
    "batch_export",
]

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from matplotlib.figure import Figure
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def save_figure(
    fig: Union[Figure, go.Figure],
    filename: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    transparent: bool = False,
    formats: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Save figure in multiple formats.

    Saves a matplotlib or plotly figure to disk in one or more formats
    with professional quality settings.

    Args:
        fig: Matplotlib Figure or Plotly Figure to save
        filename: Base filename (without extension)
        dpi: Resolution for raster formats
        bbox_inches: How to handle figure bounds
        transparent: Whether to use transparent background
        formats: List of formats to save (default: ['png'])
        metadata: Optional metadata to embed in files

    Returns:
        List of saved file paths

    Raises:
        ValueError: If unsupported format is requested

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> save_figure(fig, "my_plot", formats=["png", "pdf"])
        ['my_plot.png', 'my_plot.pdf']
    """
    if formats is None:
        formats = ["png"]

    saved_files = []
    base_path = Path(filename)
    base_dir = base_path.parent
    base_name = base_path.stem

    # Create directory if it doesn't exist
    if base_dir and not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    # Handle matplotlib figures
    if isinstance(fig, Figure):
        for fmt in formats:
            output_path = base_path.parent / f"{base_name}.{fmt}"

            if fmt in ["png", "jpg", "jpeg", "svg", "pdf", "eps"]:
                fig.savefig(
                    output_path,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    transparent=transparent,
                    metadata=metadata,
                )
                saved_files.append(str(output_path))
            else:
                raise ValueError(f"Unsupported format for matplotlib: {fmt}")

    # Handle plotly figures
    elif isinstance(fig, go.Figure):
        for fmt in formats:
            output_path = base_path.parent / f"{base_name}.{fmt}"

            if fmt == "html":
                fig.write_html(output_path)
                saved_files.append(str(output_path))
            elif fmt in ["png", "jpg", "jpeg", "svg", "pdf"]:
                try:
                    fig.write_image(
                        output_path,
                        width=1920,
                        height=1080,
                        scale=2 if fmt in ["png", "jpg", "jpeg"] else 1,
                    )
                    saved_files.append(str(output_path))
                except Exception as e:
                    logger.warning(
                        "Could not save plotly figure as %s: %s. "
                        "Install kaleido for static image export: pip install kaleido",
                        fmt,
                        e,
                    )
            else:
                raise ValueError(f"Unsupported format for plotly: {fmt}")
    else:
        raise TypeError("Figure must be matplotlib Figure or plotly Figure")

    return saved_files


def save_for_publication(
    fig: Figure,
    filename: str,
    width: float = 7,
    height: float = 5,
    dpi: int = 600,
) -> str:
    """Save figure with publication-quality settings.

    Exports a figure with settings optimized for academic publication
    or professional reports.

    Args:
        fig: Matplotlib figure
        filename: Output filename (without extension)
        width: Figure width in inches
        height: Figure height in inches
        dpi: Resolution (600 for print quality)

    Returns:
        Path to saved PDF file

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> save_for_publication(fig, "figure_1")
        'figure_1.pdf'
    """
    # Resize figure to exact dimensions
    fig.set_size_inches(width, height)

    # Save as PDF (vector format preferred for publication)
    output_path = f"{filename}.pdf"
    fig.savefig(
        output_path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        transparent=False,
        metadata={
            "Title": filename,
            "Producer": "Ergodic Insurance Analysis",
        },
    )

    # Also save high-res PNG for review
    fig.savefig(
        f"{filename}.png",
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        transparent=False,
    )

    return output_path


def save_for_presentation(
    fig: Union[Figure, go.Figure],
    filename: str,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Save figure optimized for presentations.

    Exports a figure with settings optimized for PowerPoint or
    other presentation software.

    Args:
        fig: Matplotlib or Plotly figure
        filename: Output filename (without extension)
        width: Width in pixels
        height: Height in pixels

    Returns:
        Path to saved file

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> save_for_presentation(fig, "slide_1")
        'slide_1.png'
    """
    output_path = f"{filename}.png"

    if isinstance(fig, Figure):
        # Calculate DPI to achieve desired pixel dimensions
        fig_width, fig_height = fig.get_size_inches()
        dpi_x = width / fig_width
        dpi_y = height / fig_height
        dpi = min(dpi_x, dpi_y)

        fig.savefig(
            output_path,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,  # Transparent for slide backgrounds
        )
    elif isinstance(fig, go.Figure):
        try:
            fig.write_image(
                output_path,
                width=width,
                height=height,
                scale=1,
            )
        except Exception as e:
            logger.warning(
                "Could not save plotly figure: %s. "
                "Install kaleido for static image export: pip install kaleido",
                e,
            )
            # Fallback to HTML
            output_path = f"{filename}.html"
            fig.write_html(output_path)

    return output_path


def save_for_web(
    fig: Union[Figure, go.Figure],
    filename: str,
    optimize: bool = True,
) -> Dict[str, str]:
    """Save figure optimized for web display.

    Creates web-optimized versions of the figure including responsive
    formats and multiple resolutions.

    Args:
        fig: Matplotlib or Plotly figure
        filename: Base filename (without extension)
        optimize: Whether to optimize file sizes

    Returns:
        Dictionary of format to file path mappings

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> files = save_for_web(fig, "chart")
        >>> print(files)
        {'thumbnail': 'chart_thumb.png', 'full': 'chart.png', 'svg': 'chart.svg'}
    """
    saved_files = {}

    if isinstance(fig, Figure):
        # Save thumbnail (low res for quick loading)
        thumb_path = f"{filename}_thumb.png"
        fig.savefig(thumb_path, dpi=72, bbox_inches="tight")
        saved_files["thumbnail"] = thumb_path

        # Save full resolution PNG
        full_path = f"{filename}.png"
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
        saved_files["full"] = full_path

        # Save SVG for scalable graphics
        svg_path = f"{filename}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        saved_files["svg"] = svg_path

    elif isinstance(fig, go.Figure):
        # Save interactive HTML
        html_path = f"{filename}.html"
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",  # Use CDN for smaller file
            config={"displayModeBar": False} if optimize else None,
        )
        saved_files["html"] = html_path

        # Try to save static versions
        try:
            # Thumbnail
            thumb_path = f"{filename}_thumb.png"
            fig.write_image(thumb_path, width=400, height=300)
            saved_files["thumbnail"] = thumb_path

            # Full size
            full_path = f"{filename}.png"
            fig.write_image(full_path, width=1200, height=800)
            saved_files["full"] = full_path
        except Exception:
            pass  # Kaleido not installed

    return saved_files


def batch_export(
    figures: Dict[str, Union[Figure, go.Figure]],
    output_dir: str,
    formats: Optional[List[str]] = None,
    dpi: int = 300,
) -> Dict[str, List[str]]:
    """Export multiple figures in batch.

    Saves multiple figures to a directory with consistent settings.

    Args:
        figures: Dictionary mapping names to figures
        output_dir: Output directory path
        formats: List of formats to save each figure
        dpi: Resolution for raster formats

    Returns:
        Dictionary mapping figure names to lists of saved files

    Examples:
        >>> fig1, ax1 = plt.subplots()
        >>> fig2, ax2 = plt.subplots()
        >>> figures = {"chart1": fig1, "chart2": fig2}
        >>> batch_export(figures, "output/", formats=["png"])
    """
    if formats is None:
        formats = ["png", "pdf"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_saved = {}

    for name, fig in figures.items():
        output_base = output_path / name
        saved = save_figure(
            fig,
            str(output_base),
            dpi=dpi,
            formats=formats,
        )
        all_saved[name] = saved

    return all_saved

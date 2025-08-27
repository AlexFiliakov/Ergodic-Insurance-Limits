"""Visualization infrastructure for professional report generation.

This package provides centralized styling and figure generation capabilities:
- StyleManager: Manages themes, colors, fonts, and figure configurations
- FigureFactory: Creates standardized plots with consistent styling
"""

from .figure_factory import FigureFactory
from .style_manager import StyleManager, Theme

__all__ = ["StyleManager", "Theme", "FigureFactory"]

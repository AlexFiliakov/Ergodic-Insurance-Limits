"""Style management for consistent visualization across all reports.

This module provides centralized style configuration for all visualizations,
including color palettes, fonts, figure sizes, and DPI settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import yaml


class Theme(Enum):
    """Available visualization themes."""

    DEFAULT = "default"
    COLORBLIND = "colorblind"
    PRESENTATION = "presentation"
    MINIMAL = "minimal"
    PRINT = "print"


@dataclass
class ColorPalette:
    """Color palette configuration for a theme.

    Attributes:
        primary: Main color for primary elements
        secondary: Secondary color for supporting elements
        accent: Accent color for highlights
        warning: Color for warnings or negative values
        success: Color for positive values or success states
        neutral: Neutral gray tones
        background: Background color
        text: Text color
        grid: Grid line color
        series: List of colors for multiple data series
    """

    primary: str = "#0080C7"  # Corporate blue
    secondary: str = "#003F5C"  # Dark blue
    accent: str = "#FF9800"  # Orange
    warning: str = "#D32F2F"  # Red
    success: str = "#4CAF50"  # Green
    neutral: str = "#666666"  # Gray
    background: str = "#FFFFFF"  # White
    text: str = "#000000"  # Black
    grid: str = "#E0E0E0"  # Light gray
    series: List[str] = field(
        default_factory=lambda: [
            "#0080C7",
            "#D32F2F",
            "#4CAF50",
            "#FF9800",
            "#7B1FA2",
            "#00796B",
            "#003F5C",
            "#FFD700",
        ]
    )


@dataclass
class FontConfig:
    """Font configuration for a theme.

    Attributes:
        family: Font family name
        size_base: Base font size
        size_title: Title font size
        size_label: Label font size
        size_tick: Tick label font size
        size_legend: Legend font size
        weight_normal: Normal font weight
        weight_bold: Bold font weight
    """

    family: str = "Arial"
    size_base: int = 11
    size_title: int = 14
    size_label: int = 12
    size_tick: int = 10
    size_legend: int = 10
    weight_normal: str = "normal"
    weight_bold: str = "bold"


@dataclass
class FigureConfig:
    """Figure size and DPI configuration.

    Attributes:
        size_small: Small figure size (width, height) in inches
        size_medium: Medium figure size
        size_large: Large figure size
        size_blog: Blog-optimized size (8x6)
        size_technical: Technical appendix size (10x8)
        size_presentation: Presentation slide size
        dpi_screen: DPI for screen display
        dpi_web: DPI for web publishing (150)
        dpi_print: DPI for print quality (300)
    """

    size_small: Tuple[float, float] = (6, 4)
    size_medium: Tuple[float, float] = (8, 6)
    size_large: Tuple[float, float] = (12, 8)
    size_blog: Tuple[float, float] = (8, 6)
    size_technical: Tuple[float, float] = (10, 8)
    size_presentation: Tuple[float, float] = (10, 7.5)
    dpi_screen: int = 100
    dpi_web: int = 150
    dpi_print: int = 300


@dataclass
class GridConfig:
    """Grid and axis configuration.

    Attributes:
        show_grid: Whether to show grid lines
        grid_alpha: Grid transparency
        grid_linewidth: Grid line width
        spine_top: Show top spine
        spine_right: Show right spine
        spine_bottom: Show bottom spine
        spine_left: Show left spine
        spine_linewidth: Spine line width
        tick_major_width: Major tick width
        tick_minor_width: Minor tick width
    """

    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.5
    spine_top: bool = False
    spine_right: bool = False
    spine_bottom: bool = True
    spine_left: bool = True
    spine_linewidth: float = 0.8
    tick_major_width: float = 0.8
    tick_minor_width: float = 0.4


class StyleManager:
    """Manages visualization styles and themes.

    This class provides centralized style management for all visualizations,
    supporting multiple themes, custom configurations, and style inheritance.

    Example:
        >>> style_mgr = StyleManager()
        >>> style_mgr.set_theme(Theme.PRESENTATION)
        >>> style_mgr.apply_style()
        >>> # Create plots with consistent styling

        >>> # Or with custom configuration
        >>> style_mgr = StyleManager(config_path="custom_style.yaml")
        >>> style_mgr.apply_style()
    """

    def __init__(
        self,
        theme: Theme = Theme.DEFAULT,
        config_path: Optional[Union[str, Path]] = None,
        custom_colors: Optional[Dict[str, str]] = None,
        custom_fonts: Optional[Dict[str, Any]] = None,
    ):
        """Initialize style manager.

        Args:
            theme: Initial theme to use
            config_path: Path to YAML configuration file
            custom_colors: Custom color overrides
            custom_fonts: Custom font overrides
        """
        self.current_theme = theme
        self.themes: Dict[Theme, Dict[str, Any]] = self._initialize_themes()

        # Load custom configuration if provided
        if config_path:
            self.load_config(config_path)

        # Apply custom overrides
        if custom_colors:
            self.update_colors(custom_colors)
        if custom_fonts:
            self.update_fonts(custom_fonts)

    def _initialize_themes(self) -> Dict[Theme, Dict[str, Any]]:
        """Initialize built-in themes.

        Returns:
            Dictionary mapping themes to their configurations
        """
        themes = {}

        # Default corporate theme
        themes[Theme.DEFAULT] = {
            "colors": ColorPalette(),
            "fonts": FontConfig(),
            "figure": FigureConfig(),
            "grid": GridConfig(),
        }

        # Colorblind-friendly theme
        themes[Theme.COLORBLIND] = {
            "colors": ColorPalette(
                primary="#0173B2",  # Blue
                secondary="#DE8F05",  # Orange
                accent="#029E73",  # Green
                warning="#CC78BC",  # Light purple
                success="#56B4E9",  # Sky blue
                series=[
                    "#0173B2",
                    "#DE8F05",
                    "#029E73",
                    "#CC78BC",
                    "#ECE133",
                    "#56B4E9",
                    "#949494",
                    "#FBE5D6",
                ],
            ),
            "fonts": FontConfig(),
            "figure": FigureConfig(),
            "grid": GridConfig(),
        }

        # Presentation theme (larger fonts, bolder colors)
        themes[Theme.PRESENTATION] = {
            "colors": ColorPalette(
                primary="#003F7F",  # Darker blue
                warning="#FF0000",  # Bright red
                success="#00AA00",  # Bright green
            ),
            "fonts": FontConfig(
                size_base=14,
                size_title=18,
                size_label=16,
                size_tick=12,
                size_legend=12,
            ),
            "figure": FigureConfig(
                size_medium=(10, 7.5),
                size_large=(14, 10),
            ),
            "grid": GridConfig(grid_alpha=0.2),
        }

        # Minimal theme
        themes[Theme.MINIMAL] = {
            "colors": ColorPalette(
                primary="#333333",
                secondary="#666666",
                accent="#999999",
                warning="#CC0000",
                success="#006600",
                series=["#333333", "#666666", "#999999", "#CCCCCC"],
            ),
            "fonts": FontConfig(family="Helvetica"),
            "figure": FigureConfig(),
            "grid": GridConfig(
                show_grid=False,
                spine_linewidth=0.5,
            ),
        }

        # Print theme (high contrast, thicker lines)
        themes[Theme.PRINT] = {
            "colors": ColorPalette(
                background="#FFFFFF",
                text="#000000",
                grid="#CCCCCC",
            ),
            "fonts": FontConfig(
                size_base=10,
                size_title=12,
                size_label=11,
            ),
            "figure": FigureConfig(
                dpi_screen=300,
                dpi_web=300,
                dpi_print=600,
            ),
            "grid": GridConfig(
                grid_linewidth=0.3,
                spine_linewidth=1.0,
                tick_major_width=1.0,
            ),
        }

        return themes

    def set_theme(self, theme: Theme) -> None:
        """Set the current theme.

        Args:
            theme: Theme to activate
        """
        if theme not in self.themes:
            raise ValueError(f"Unknown theme: {theme}")
        self.current_theme = theme

    def get_theme_config(self, theme: Optional[Theme] = None) -> Dict[str, Any]:
        """Get configuration for a theme.

        Args:
            theme: Theme to get config for (defaults to current)

        Returns:
            Theme configuration dictionary
        """
        theme = theme or self.current_theme
        return self.themes[theme].copy()

    def get_colors(self) -> ColorPalette:
        """Get current color palette.

        Returns:
            Current theme's color palette
        """
        colors = self.themes[self.current_theme]["colors"]
        assert isinstance(colors, ColorPalette)
        return colors

    def get_fonts(self) -> FontConfig:
        """Get current font configuration.

        Returns:
            Current theme's font configuration
        """
        fonts = self.themes[self.current_theme]["fonts"]
        assert isinstance(fonts, FontConfig)
        return fonts

    def get_figure_config(self) -> FigureConfig:
        """Get current figure configuration.

        Returns:
            Current theme's figure configuration
        """
        figure = self.themes[self.current_theme]["figure"]
        assert isinstance(figure, FigureConfig)
        return figure

    def get_grid_config(self) -> GridConfig:
        """Get current grid configuration.

        Returns:
            Current theme's grid configuration
        """
        grid = self.themes[self.current_theme]["grid"]
        assert isinstance(grid, GridConfig)
        return grid

    def update_colors(self, updates: Dict[str, str]) -> None:
        """Update colors in current theme.

        Args:
            updates: Dictionary of color updates
        """
        colors = self.get_colors()
        for key, value in updates.items():
            if hasattr(colors, key):
                setattr(colors, key, value)

    def update_fonts(self, updates: Dict[str, Any]) -> None:
        """Update fonts in current theme.

        Args:
            updates: Dictionary of font updates
        """
        fonts = self.get_fonts()
        for key, value in updates.items():
            if hasattr(fonts, key):
                setattr(fonts, key, value)

    def apply_style(self) -> None:
        """Apply current theme to matplotlib.

        This updates matplotlib's rcParams to match the current theme settings.
        """
        colors = self.get_colors()
        fonts = self.get_fonts()
        grid = self.get_grid_config()

        # Set matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid")

        # Update rcParams
        plt.rcParams.update(
            {
                # Font settings
                "font.family": "sans-serif",
                "font.sans-serif": [fonts.family, "Arial", "Helvetica", "DejaVu Sans"],
                "font.size": fonts.size_base,
                "axes.titlesize": fonts.size_title,
                "axes.labelsize": fonts.size_label,
                "xtick.labelsize": fonts.size_tick,
                "ytick.labelsize": fonts.size_tick,
                "legend.fontsize": fonts.size_legend,
                "figure.titlesize": fonts.size_title + 2,
                # Spine settings
                "axes.spines.top": grid.spine_top,
                "axes.spines.right": grid.spine_right,
                "axes.spines.left": grid.spine_left,
                "axes.spines.bottom": grid.spine_bottom,
                "axes.edgecolor": colors.neutral,
                "axes.linewidth": grid.spine_linewidth,
                # Grid settings
                "axes.grid": grid.show_grid,
                "grid.color": colors.grid,
                "grid.linewidth": grid.grid_linewidth,
                "grid.alpha": grid.grid_alpha,
                # Line settings
                "lines.linewidth": 2,
                "patch.linewidth": 0.5,
                # Tick settings
                "xtick.major.width": grid.tick_major_width,
                "ytick.major.width": grid.tick_major_width,
                "xtick.minor.width": grid.tick_minor_width,
                "ytick.minor.width": grid.tick_minor_width,
                # Figure settings
                "figure.facecolor": colors.background,
                "axes.facecolor": colors.background,
                # Color cycle
                "axes.prop_cycle": plt.cycler("color", colors.series),
            }
        )

    def get_figure_size(
        self, size_type: str = "medium", orientation: str = "landscape"
    ) -> Tuple[float, float]:
        """Get figure size for a given type.

        Args:
            size_type: Size type (small, medium, large, blog, technical, presentation)
            orientation: Figure orientation (landscape or portrait)

        Returns:
            Tuple of (width, height) in inches
        """
        fig_config = self.get_figure_config()

        size_map = {
            "small": fig_config.size_small,
            "medium": fig_config.size_medium,
            "large": fig_config.size_large,
            "blog": fig_config.size_blog,
            "technical": fig_config.size_technical,
            "presentation": fig_config.size_presentation,
        }

        size = size_map.get(size_type, fig_config.size_medium)

        if orientation == "portrait":
            return (size[1], size[0])
        return size

    def get_dpi(self, output_type: str = "screen") -> int:
        """Get DPI for output type.

        Args:
            output_type: Output type (screen, web, print)

        Returns:
            DPI value
        """
        fig_config = self.get_figure_config()

        dpi_map = {
            "screen": fig_config.dpi_screen,
            "web": fig_config.dpi_web,
            "print": fig_config.dpi_print,
        }

        return dpi_map.get(output_type, fig_config.dpi_screen)

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Update theme configurations
        if "themes" not in config:
            return

        for theme_name, theme_config in config["themes"].items():
            self._update_theme_from_config(theme_name, theme_config)

    def _update_theme_from_config(self, theme_name: str, theme_config: Dict[str, Any]) -> None:
        """Update a single theme from configuration.

        Args:
            theme_name: Name of the theme
            theme_config: Configuration for the theme
        """
        theme = Theme[theme_name.upper()]
        if theme not in self.themes:
            self.themes[theme] = {
                "colors": ColorPalette(),
                "fonts": FontConfig(),
                "figure": FigureConfig(),
                "grid": GridConfig(),
            }

        # Update each component
        self._update_component(self.themes[theme]["colors"], theme_config.get("colors", {}))
        self._update_component(self.themes[theme]["fonts"], theme_config.get("fonts", {}))
        self._update_figure_component(self.themes[theme]["figure"], theme_config.get("figure", {}))
        self._update_component(self.themes[theme]["grid"], theme_config.get("grid", {}))

    def _update_component(self, component: Any, config_dict: Dict[str, Any]) -> None:
        """Update a component with configuration values.

        Args:
            component: Component to update
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if hasattr(component, key):
                setattr(component, key, value)

    def _update_figure_component(self, component: Any, config_dict: Dict[str, Any]) -> None:
        """Update figure component with special handling for sizes.

        Args:
            component: Figure component to update
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if hasattr(component, key):
                # Handle tuple conversions for sizes
                if "size" in key and isinstance(value, list):
                    value = tuple(value)
                setattr(component, key, value)

    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file.

        Args:
            config_path: Path to save YAML configuration
        """
        config_path = Path(config_path)

        # Build configuration dictionary
        config: Dict[str, Any] = {"themes": {}}

        for theme, theme_config in self.themes.items():
            config["themes"][theme.value] = {
                "colors": {
                    key: (
                        list(val)
                        if isinstance(val := getattr(theme_config["colors"], key), tuple)
                        else val
                    )
                    for key in theme_config["colors"].__dataclass_fields__
                },
                "fonts": {
                    key: (
                        list(val)
                        if isinstance(val := getattr(theme_config["fonts"], key), tuple)
                        else val
                    )
                    for key in theme_config["fonts"].__dataclass_fields__
                },
                "figure": {
                    key: (
                        list(val)
                        if isinstance(val := getattr(theme_config["figure"], key), tuple)
                        else val
                    )
                    for key in theme_config["figure"].__dataclass_fields__
                },
                "grid": {
                    key: (
                        list(val)
                        if isinstance(val := getattr(theme_config["grid"], key), tuple)
                        else val
                    )
                    for key in theme_config["grid"].__dataclass_fields__
                },
            }

        # Save to file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    def create_style_sheet(self) -> Dict[str, Any]:
        """Create matplotlib style sheet dictionary.

        Returns:
            Style sheet dictionary compatible with matplotlib
        """
        colors = self.get_colors()
        fonts = self.get_fonts()
        grid = self.get_grid_config()

        return {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": [fonts.family],
            "font.size": fonts.size_base,
            # Axes
            "axes.titlesize": fonts.size_title,
            "axes.labelsize": fonts.size_label,
            "axes.edgecolor": colors.neutral,
            "axes.linewidth": grid.spine_linewidth,
            "axes.grid": grid.show_grid,
            "axes.spines.top": grid.spine_top,
            "axes.spines.right": grid.spine_right,
            "axes.spines.bottom": grid.spine_bottom,
            "axes.spines.left": grid.spine_left,
            "axes.facecolor": colors.background,
            "axes.prop_cycle": f"cycler('color', {colors.series})",
            # Grid
            "grid.color": colors.grid,
            "grid.linewidth": grid.grid_linewidth,
            "grid.alpha": grid.grid_alpha,
            # Ticks
            "xtick.labelsize": fonts.size_tick,
            "ytick.labelsize": fonts.size_tick,
            "xtick.major.width": grid.tick_major_width,
            "ytick.major.width": grid.tick_major_width,
            # Legend
            "legend.fontsize": fonts.size_legend,
            # Figure
            "figure.facecolor": colors.background,
            "figure.titlesize": fonts.size_title + 2,
        }

    def inherit_from(self, parent_theme: Theme, modifications: Dict[str, Any]) -> Theme:
        """Create a new theme inheriting from a parent with modifications.

        Args:
            parent_theme: Theme to inherit from
            modifications: Dictionary of modifications

        Returns:
            New theme enum value
        """
        # This would typically create a new custom theme
        # For now, we'll just modify the current theme
        import copy

        new_config = copy.deepcopy(self.themes[parent_theme])

        # Apply modifications
        for key, value in modifications.items():
            if key == "colors" and isinstance(value, dict):
                for color_key, color_value in value.items():
                    if hasattr(new_config["colors"], color_key):
                        setattr(new_config["colors"], color_key, color_value)
            elif key == "fonts" and isinstance(value, dict):
                for font_key, font_value in value.items():
                    if hasattr(new_config["fonts"], font_key):
                        setattr(new_config["fonts"], font_key, font_value)

        # Store as a custom theme (would need enum extension in production)
        self.themes[Theme.DEFAULT] = new_config
        return Theme.DEFAULT

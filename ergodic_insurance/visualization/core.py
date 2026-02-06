"""Core visualization utilities and constants.

This module provides the foundational elements for visualization including
WSJ-style color palettes, formatters, and base configuration settings.
"""

import matplotlib.pyplot as plt

# WSJ Color Palette
WSJ_COLORS = {
    "light_blue": "#ADD8E6",  # Light Blue for additional styling
    "blue": "#0080C7",  # Primary blue
    "dark_blue": "#003F5C",  # Dark blue
    "red": "#D32F2F",  # Red for negative/warning
    "green": "#4CAF50",  # Green for positive
    "gray": "#666666",  # Gray for secondary
    "light_gray": "#E0E0E0",  # Light gray for grid
    "black": "#000000",  # Black for text
    "orange": "#FF9800",  # Orange for highlights
    "yellow": "#FFD700",  # Yellow for highlights
    "purple": "#7B1FA2",  # Purple for special
    "teal": "#00796B",  # Teal for alternative
}

# Professional color sequence for multiple series
COLOR_SEQUENCE = [
    WSJ_COLORS["blue"],
    WSJ_COLORS["red"],
    WSJ_COLORS["green"],
    WSJ_COLORS["orange"],
    WSJ_COLORS["purple"],
    WSJ_COLORS["teal"],
    WSJ_COLORS["dark_blue"],
]


def set_wsj_style():
    """Set matplotlib to use WSJ-style formatting.

    This function applies Wall Street Journal aesthetic styling to matplotlib
    plots including font choices, spine visibility, grid settings, and colors.
    """
    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Update rcParams for WSJ style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": WSJ_COLORS["gray"],
            "axes.linewidth": 0.8,
            "grid.color": WSJ_COLORS["light_gray"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "lines.linewidth": 2,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
        }
    )


def format_currency(value: float, decimals: int = 0, abbreviate: bool = False) -> str:
    """Format value as currency.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        abbreviate: If True, use K/M/B notation for large numbers

    Returns:
        Formatted string (e.g., "$1,000" or "$1K" if abbreviate=True)

    Examples:
        >>> format_currency(1000)
        '$1,000'
        >>> format_currency(1500000, abbreviate=True)
        '$1.5M'
        >>> format_currency(2500.50, decimals=2)
        '$2,500.50'
    """
    if abbreviate:
        if abs(value) >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        if abs(value) >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        if abs(value) >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        return f"${value:.{decimals}f}"
    # Handle negative values
    if value < 0:
        return f"-${abs(value):,.{decimals}f}"
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage.

    Args:
        value: Numeric value (0.05 = 5%)
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., "5.0%")

    Examples:
        >>> format_percentage(0.05)
        '5.0%'
        >>> format_percentage(0.1234, decimals=2)
        '12.34%'
    """
    return f"{value*100:.{decimals}f}%"


class WSJFormatter:
    """Formatter for WSJ-style axis labels.

    This class provides static methods for formatting axis values in various
    styles consistent with Wall Street Journal publication standards.

    Methods:
        currency_formatter: Format axis values as currency
        currency: Format value as currency (shortened method name)
        percentage_formatter: Format axis values as percentage
        percentage: Format value as percentage (shortened method name)
        number: Format large numbers with appropriate suffix
        millions_formatter: Format axis values in millions
    """

    @staticmethod
    def currency_formatter(x, pos):
        """Format axis values as currency.

        Args:
            x: The value to format
            pos: The position (unused but required by matplotlib)

        Returns:
            Formatted currency string
        """
        return format_currency(x, decimals=0, abbreviate=True)

    @staticmethod
    def currency(x: float, decimals: int = 1) -> str:  # pylint: disable=too-many-return-statements
        """Format value as currency (shortened method name).

        Args:
            x: The value to format
            decimals: Number of decimal places

        Returns:
            Formatted currency string with appropriate suffix
        """
        sign = "-" if x < 0 else ""
        x = abs(x)

        if x >= 1e12:
            if x == int(x / 1e12) * 1e12:  # Whole trillions
                return f"{sign}${int(x/1e12)}T"
            return f"{sign}${x/1e12:.{decimals}f}T"
        if x >= 1e9:
            if x == int(x / 1e9) * 1e9:  # Whole billions
                return f"{sign}${int(x/1e9)}B"
            return f"{sign}${x/1e9:.{decimals}f}B"
        if x >= 1e6:
            if x == int(x / 1e6) * 1e6:  # Whole millions
                return f"{sign}${int(x/1e6)}M"
            return f"{sign}${x/1e6:.{decimals}f}M"
        if x >= 1e3:
            if x == int(x / 1e3) * 1e3:  # Whole thousands
                return f"{sign}${int(x/1e3)}K"
            return f"{sign}${x/1e3:.{decimals}f}K"
        if 0 < x < 1:
            return f"{sign}${x:.2f}"
        return f"{sign}${int(x)}" if x == int(x) else f"{sign}${x:.{decimals}f}"

    @staticmethod
    def percentage_formatter(x, pos):
        """Format axis values as percentage.

        Args:
            x: The value to format
            pos: The position (unused but required by matplotlib)

        Returns:
            Formatted percentage string
        """
        return format_percentage(x, decimals=0)

    @staticmethod
    def percentage(x: float, decimals: int = 1) -> str:
        """Format value as percentage (shortened method name).

        Args:
            x: The value to format (0.05 = 5%)
            decimals: Number of decimal places

        Returns:
            Formatted percentage string
        """
        return f"{x*100:.{decimals}f}%"

    @staticmethod
    def number(x: float, decimals: int = 2) -> str:
        """Format large numbers with appropriate suffix.

        Args:
            x: The value to format
            decimals: Number of decimal places

        Returns:
            Formatted number string with K/M/B/T suffix

        Examples:
            >>> WSJFormatter.number(1500000)
            '1.50M'
            >>> WSJFormatter.number(2500)
            '2.50K'
        """
        if abs(x) >= 1e12:
            if abs(x) >= 1e15:
                # Very large numbers - show in trillions with multiplier
                return f"{int(x/1e12)}T"
            return f"{x/1e12:.{decimals}f}T"
        if abs(x) >= 1e9:
            return f"{x/1e9:.{decimals}f}B"
        if abs(x) >= 1e6:
            return f"{x/1e6:.{decimals}f}M"
        if abs(x) >= 1e3:
            return f"{x/1e3:.{decimals}f}K"
        return f"{int(x)}" if x == int(x) else f"{x:.{decimals}f}"

    @staticmethod
    def millions_formatter(x, pos):
        """Format axis values in millions.

        Args:
            x: The value to format
            pos: The position (unused but required by matplotlib)

        Returns:
            Value formatted as millions with M suffix
        """
        return f"{x/1e6:.0f}M"

"""Formatting utilities for table generation and report creation.

This module provides comprehensive formatting functions for numbers, currency,
percentages, and color coding for tables in various output formats.

Google-style docstrings are used throughout for consistency.
"""

from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


class NumberFormatter:
    """Format numbers for display in tables and reports.

    This class provides methods to format various numeric types including
    currency, percentages, and scientific notation with consistent precision
    and locale-aware formatting.

    Attributes:
        currency_symbol: Symbol to use for currency formatting.
        decimal_places: Default number of decimal places.
        thousands_separator: Character for thousands separation.
        decimal_separator: Character for decimal separation.
    """

    def __init__(
        self,
        currency_symbol: str = "$",
        decimal_places: int = 2,
        thousands_separator: str = ",",
        decimal_separator: str = ".",
    ):
        """Initialize NumberFormatter.

        Args:
            currency_symbol: Symbol for currency (default "$").
            decimal_places: Default decimal precision (default 2).
            thousands_separator: Thousands separator (default ",").
            decimal_separator: Decimal separator (default ".").
        """
        self.currency_symbol = currency_symbol
        self.decimal_places = decimal_places
        self.thousands_separator = thousands_separator
        self.decimal_separator = decimal_separator

    def format_currency(
        self,
        value: Union[float, int, Decimal],
        decimals: Optional[int] = None,
        abbreviate: bool = False,
    ) -> str:
        """Format a number as currency.

        Args:
            value: Numeric value to format.
            decimals: Number of decimal places (uses default if None).
            abbreviate: Whether to abbreviate large numbers (e.g., $1.5M).

        Returns:
            Formatted currency string.

        Examples:
            >>> formatter = NumberFormatter()
            >>> formatter.format_currency(1234567.89)
            '$1,234,567.89'
            >>> formatter.format_currency(1234567.89, abbreviate=True)
            '$1.23M'
        """
        if pd.isna(value):
            return "-"

        decimals = decimals if decimals is not None else self.decimal_places

        if abbreviate and abs(value) >= 1_000_000_000:
            return f"{self.currency_symbol}{value/1_000_000_000:.{decimals}f}B"
        elif abbreviate and abs(value) >= 1_000_000:
            return f"{self.currency_symbol}{value/1_000_000:.{decimals}f}M"
        elif abbreviate and abs(value) >= 1_000:
            return f"{self.currency_symbol}{value/1_000:.{decimals}f}K"
        else:
            # Format with thousands separator
            formatted = f"{value:,.{decimals}f}"
            # Replace default separators with configured ones
            if self.thousands_separator != ",":
                formatted = formatted.replace(",", self.thousands_separator)
            if self.decimal_separator != ".":
                formatted = formatted.replace(".", self.decimal_separator)
            return f"{self.currency_symbol}{formatted}"

    def format_percentage(
        self,
        value: Union[float, int],
        decimals: Optional[int] = None,
        multiply_by_100: bool = True,
    ) -> str:
        """Format a number as percentage.

        Args:
            value: Numeric value to format.
            decimals: Number of decimal places (default 1).
            multiply_by_100: Whether to multiply by 100 (default True).

        Returns:
            Formatted percentage string.

        Examples:
            >>> formatter = NumberFormatter()
            >>> formatter.format_percentage(0.1234)
            '12.34%'
            >>> formatter.format_percentage(12.34, multiply_by_100=False)
            '12.34%'
        """
        if pd.isna(value):
            return "-"

        decimals = decimals if decimals is not None else 2

        if multiply_by_100:
            value = value * 100

        return f"{value:.{decimals}f}%"

    def format_number(
        self,
        value: Union[float, int],
        decimals: Optional[int] = None,
        scientific: bool = False,
        abbreviate: bool = False,
    ) -> str:
        """Format a general number.

        Args:
            value: Numeric value to format.
            decimals: Number of decimal places.
            scientific: Use scientific notation for large/small numbers.
            abbreviate: Abbreviate large numbers (K, M, B).

        Returns:
            Formatted number string.

        Examples:
            >>> formatter = NumberFormatter()
            >>> formatter.format_number(1234567.89)
            '1,234,567.89'
            >>> formatter.format_number(0.00001234, scientific=True)
            '1.23e-05'
        """
        if pd.isna(value):
            return "-"

        decimals = decimals if decimals is not None else self.decimal_places

        if scientific and (abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0)):
            return f"{value:.{decimals}e}"
        elif abbreviate and abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.{decimals}f}B"
        elif abbreviate and abs(value) >= 1_000_000:
            return f"{value/1_000_000:.{decimals}f}M"
        elif abbreviate and abs(value) >= 1_000:
            return f"{value/1_000:.{decimals}f}K"
        else:
            formatted = f"{value:,.{decimals}f}"
            if self.thousands_separator != ",":
                formatted = formatted.replace(",", self.thousands_separator)
            if self.decimal_separator != ".":
                formatted = formatted.replace(".", self.decimal_separator)
            return formatted

    def format_ratio(self, value: Union[float, int], decimals: int = 2) -> str:
        """Format a ratio value.

        Args:
            value: Ratio value to format.
            decimals: Number of decimal places.

        Returns:
            Formatted ratio string.

        Examples:
            >>> formatter = NumberFormatter()
            >>> formatter.format_ratio(1.5)
            '1.50x'
        """
        if pd.isna(value):
            return "-"

        return f"{value:.{decimals}f}x"


class ColorCoder:
    """Apply color coding to values for visual indicators.

    This class provides methods for traffic light coloring, heatmaps,
    and threshold-based coloring for different output formats.

    Attributes:
        output_format: Target output format (html, latex, terminal).
        color_scheme: Color scheme to use.
    """

    # Default color schemes
    TRAFFIC_LIGHT = {
        "good": "#28a745",  # Green
        "warning": "#ffc107",  # Yellow/Amber
        "bad": "#dc3545",  # Red
    }

    HEATMAP_COLORS = {
        "low": "#e3f2fd",  # Light blue
        "medium_low": "#90caf9",  # Medium blue
        "medium": "#42a5f5",  # Blue
        "medium_high": "#ffb74d",  # Orange
        "high": "#ef5350",  # Red
    }

    def __init__(
        self,
        output_format: Literal["html", "latex", "terminal", "none"] = "none",
        color_scheme: Optional[Dict[str, str]] = None,
    ):
        """Initialize ColorCoder.

        Args:
            output_format: Target output format.
            color_scheme: Custom color scheme (uses defaults if None).
        """
        self.output_format = output_format
        self.color_scheme = color_scheme or self.TRAFFIC_LIGHT

    def traffic_light(
        self,
        value: Union[float, int],
        thresholds: Dict[str, Tuple[Optional[float], Optional[float]]],
        text: Optional[str] = None,
    ) -> str:
        """Apply traffic light coloring based on thresholds.

        Args:
            value: Numeric value to evaluate.
            thresholds: Dict with keys 'good', 'warning', 'bad' and (min, max) tuples.
            text: Text to display (uses value if None).

        Returns:
            Formatted string with appropriate coloring.

        Examples:
            >>> coder = ColorCoder(output_format="html")
            >>> thresholds = {
            ...     'good': (0.15, None),
            ...     'warning': (0.10, 0.15),
            ...     'bad': (None, 0.10)
            ... }
            >>> coder.traffic_light(0.18, thresholds)
            '<span style="color: #28a745;">0.18</span>'
        """
        if pd.isna(value):
            return "-"

        display_text = text if text is not None else str(value)

        # Determine color based on thresholds
        color_key = "bad"  # Default
        for key, (min_val, max_val) in thresholds.items():
            if min_val is not None and max_val is not None:
                if min_val <= value <= max_val:
                    color_key = key
                    break
            elif min_val is not None:
                if value >= min_val:
                    color_key = key
                    break
            elif max_val is not None:
                if value <= max_val:
                    color_key = key
                    break

        return self._apply_color(display_text, self.color_scheme.get(color_key, "#000000"))

    def heatmap(
        self,
        value: Union[float, int],
        min_val: float,
        max_val: float,
        text: Optional[str] = None,
    ) -> str:
        """Apply heatmap coloring based on value range.

        Args:
            value: Numeric value to color.
            min_val: Minimum value in range.
            max_val: Maximum value in range.
            text: Text to display (uses value if None).

        Returns:
            Formatted string with heatmap coloring.

        Examples:
            >>> coder = ColorCoder(output_format="html")
            >>> coder.heatmap(50, 0, 100)
            '<span style="background-color: #42a5f5;">50</span>'
        """
        if pd.isna(value):
            return "-"

        display_text = text if text is not None else str(value)

        # Normalize value to 0-1 range
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0, min(1, normalized))  # Clamp to [0, 1]

        # Select color based on normalized value
        if normalized < 0.2:
            color = self.HEATMAP_COLORS["low"]
        elif normalized < 0.4:
            color = self.HEATMAP_COLORS["medium_low"]
        elif normalized < 0.6:
            color = self.HEATMAP_COLORS["medium"]
        elif normalized < 0.8:
            color = self.HEATMAP_COLORS["medium_high"]
        else:
            color = self.HEATMAP_COLORS["high"]

        return self._apply_color(display_text, color, is_background=True)

    def threshold_color(
        self,
        value: Union[float, int],
        threshold: float,
        above_color: str = "#28a745",
        below_color: str = "#dc3545",
        text: Optional[str] = None,
    ) -> str:
        """Apply binary coloring based on threshold.

        Args:
            value: Numeric value to evaluate.
            threshold: Threshold value.
            above_color: Color for values above threshold.
            below_color: Color for values below threshold.
            text: Text to display.

        Returns:
            Formatted string with threshold-based coloring.
        """
        if pd.isna(value):
            return "-"

        display_text = text if text is not None else str(value)
        color = above_color if value >= threshold else below_color
        return self._apply_color(display_text, color)

    def _apply_color(
        self,
        text: str,
        color: str,
        is_background: bool = False,
    ) -> str:
        """Apply color formatting based on output format.

        Args:
            text: Text to format.
            color: Color code (hex or name).
            is_background: Whether to apply as background color.

        Returns:
            Formatted text string.
        """
        if self.output_format == "html":
            if is_background:
                return f'<span style="background-color: {color}; padding: 2px 4px;">{text}</span>'
            else:
                return f'<span style="color: {color};">{text}</span>'
        elif self.output_format == "latex":
            # LaTeX requires color package
            if is_background:
                return f"\\colorbox{{{color}}}{{{text}}}"
            else:
                return f"\\textcolor{{{color}}}{{{text}}}"
        elif self.output_format == "terminal":
            # Use Unicode symbols for terminal
            if "good" in str(color) or "#28a745" in str(color):
                return f"✓ {text}"
            elif "warning" in str(color) or "#ffc107" in str(color):
                return f"⚠ {text}"
            elif "bad" in str(color) or "#dc3545" in str(color):
                return f"✗ {text}"
            else:
                return text
        else:
            return text


class TableFormatter:
    """High-level table formatting utilities.

    This class combines number formatting and color coding to provide
    comprehensive table formatting capabilities.

    Attributes:
        number_formatter: NumberFormatter instance.
        color_coder: ColorCoder instance.
    """

    def __init__(
        self,
        output_format: Literal["html", "latex", "terminal", "none"] = "none",
        currency_symbol: str = "$",
        decimal_places: int = 2,
    ):
        """Initialize TableFormatter.

        Args:
            output_format: Target output format.
            currency_symbol: Currency symbol to use.
            decimal_places: Default decimal precision.
        """
        self.number_formatter = NumberFormatter(
            currency_symbol=currency_symbol,
            decimal_places=decimal_places,
        )
        self.color_coder = ColorCoder(output_format=output_format)
        self.output_format = output_format

    def format_dataframe(
        self,
        df: pd.DataFrame,
        column_formats: Optional[Dict[str, Dict[str, Any]]] = None,
        row_colors: Optional[Dict[int, str]] = None,
        alternating_rows: bool = False,
    ) -> pd.DataFrame:
        """Apply formatting to entire DataFrame.

        Args:
            df: Input DataFrame.
            column_formats: Format specifications per column.
            row_colors: Colors for specific rows.
            alternating_rows: Whether to use alternating row colors.

        Returns:
            Formatted DataFrame.

        Examples:
            >>> formatter = TableFormatter()
            >>> formats = {
            ...     'Revenue': {'type': 'currency', 'abbreviate': True},
            ...     'Growth': {'type': 'percentage'},
            ...     'Risk': {'type': 'traffic_light', 'thresholds': {...}}
            ... }
            >>> formatted_df = formatter.format_dataframe(df, formats)
        """
        formatted_df = df.copy()

        if column_formats:
            for col, fmt in column_formats.items():
                if col not in formatted_df.columns:
                    continue

                fmt_type = fmt.get("type", "number")

                if fmt_type == "currency":
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: self.number_formatter.format_currency(
                            x,
                            decimals=fmt.get("decimals"),
                            abbreviate=fmt.get("abbreviate", False),
                        )
                    )
                elif fmt_type == "percentage":
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: self.number_formatter.format_percentage(
                            x,
                            decimals=fmt.get("decimals"),
                            multiply_by_100=fmt.get("multiply_by_100", True),
                        )
                    )
                elif fmt_type == "number":
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: self.number_formatter.format_number(
                            x,
                            decimals=fmt.get("decimals"),
                            scientific=fmt.get("scientific", False),
                            abbreviate=fmt.get("abbreviate", False),
                        )
                    )
                elif fmt_type == "ratio":
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: self.number_formatter.format_ratio(
                            x,
                            decimals=fmt.get("decimals", 2),
                        )
                    )
                elif fmt_type == "traffic_light":
                    thresholds = fmt.get("thresholds", {})
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: self.color_coder.traffic_light(x, thresholds)
                    )
                elif fmt_type == "heatmap":
                    min_val = fmt.get("min", formatted_df[col].min())
                    max_val = fmt.get("max", formatted_df[col].max())
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: self.color_coder.heatmap(x, min_val, max_val)
                    )

        return formatted_df

    def add_totals_row(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        label: str = "Total",
        operation: Literal["sum", "mean", "median"] = "sum",
    ) -> pd.DataFrame:
        """Add a totals row to DataFrame.

        Args:
            df: Input DataFrame.
            columns: Columns to total (None for all numeric).
            label: Label for totals row.
            operation: Aggregation operation.

        Returns:
            DataFrame with totals row added.
        """
        df_with_totals = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        totals_row = {}
        for col in df.columns:
            if col in columns:
                if operation == "sum":
                    totals_row[col] = df[col].sum()
                elif operation == "mean":
                    totals_row[col] = df[col].mean()
                elif operation == "median":
                    totals_row[col] = df[col].median()
            else:
                totals_row[col] = label if df.columns.get_loc(col) == 0 else ""

        df_with_totals = pd.concat(
            [df_with_totals, pd.DataFrame([totals_row])],
            ignore_index=True,
        )

        return df_with_totals

    def add_footnotes(
        self,
        table_str: str,
        footnotes: List[str],
        format: Optional[str] = None,
    ) -> str:
        """Add footnotes to a table string.

        Args:
            table_str: Table string.
            footnotes: List of footnote texts.
            format: Output format (uses instance format if None).

        Returns:
            Table with footnotes added.
        """
        format = format or self.output_format

        if not footnotes:
            return table_str

        footnote_str = ""
        if format == "html":
            footnote_str = "<div class='footnotes'><small>"
            for i, note in enumerate(footnotes, 1):
                footnote_str += f"<sup>{i}</sup> {note}<br/>"
            footnote_str += "</small></div>"
        elif format == "latex":
            for i, note in enumerate(footnotes, 1):
                footnote_str += f"\\footnote{{{note}}}"
        else:
            footnote_str = "\n" + "-" * 40 + "\n"
            for i, note in enumerate(footnotes, 1):
                footnote_str += f"[{i}] {note}\n"

        return table_str + "\n" + footnote_str


def format_for_export(
    df: pd.DataFrame,
    format: Literal["csv", "excel", "latex", "html", "markdown"],
    include_index: bool = False,
    **kwargs,
) -> Union[str, None]:
    """Format DataFrame for export to various formats.

    Args:
        df: DataFrame to export.
        format: Export format.
        include_index: Whether to include row index.
        **kwargs: Additional format-specific arguments.

    Returns:
        Formatted string or None for file-based exports.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> csv_str = format_for_export(df, 'csv')
        >>> latex_str = format_for_export(df, 'latex', caption='My Table')
    """
    if format == "csv":
        return df.to_csv(index=include_index, **kwargs)
    elif format == "excel":
        # Excel export requires file path
        file_path = kwargs.get("file_path")
        if file_path:
            df.to_excel(file_path, index=include_index, **kwargs)
        return None
    elif format == "latex":
        caption = kwargs.pop("caption", None)
        label = kwargs.pop("label", None)
        latex_str = df.to_latex(index=include_index, **kwargs)

        if caption or label:
            # Wrap in table environment
            table_str = "\\begin{table}[htbp]\n\\centering\n"
            if caption:
                table_str += f"\\caption{{{caption}}}\n"
            if label:
                table_str += f"\\label{{{label}}}\n"
            table_str += latex_str
            table_str += "\\end{table}"
            return table_str
        return latex_str
    elif format == "html":
        table_id = kwargs.pop("table_id", None)
        classes = kwargs.pop("classes", None)
        html_str = df.to_html(index=include_index, **kwargs)

        if table_id:
            html_str = html_str.replace("<table", f'<table id="{table_id}"')
        if classes:
            html_str = html_str.replace("<table", f'<table class="{classes}"')

        return html_str
    elif format == "markdown":
        return df.to_markdown(index=include_index, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

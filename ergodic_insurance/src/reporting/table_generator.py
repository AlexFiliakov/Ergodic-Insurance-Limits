"""Table generation utilities for report creation.

This module provides functions to generate and format tables from various data sources,
supporting multiple output formats including Markdown, HTML, and LaTeX.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate  # type: ignore[import-untyped]


class TableGenerator:
    """Generate formatted tables for reports.

    This class provides methods to create professionally formatted tables
    from pandas DataFrames or raw data, supporting various output formats
    and styling options.

    Attributes:
        default_format: Default output format for tables.
        precision: Default precision for numeric values.
        max_width: Maximum width for text columns.
    """

    def __init__(
        self,
        default_format: Literal["markdown", "html", "latex", "grid"] = "markdown",
        precision: int = 2,
        max_width: int = 50,
    ):
        """Initialize TableGenerator.

        Args:
            default_format: Default output format.
            precision: Number of decimal places for floats.
            max_width: Maximum width for text columns.
        """
        self.default_format = default_format
        self.precision = precision
        self.max_width = max_width
        self._format_map = {
            "markdown": "pipe",
            "html": "html",
            "latex": "latex_booktabs",
            "grid": "grid",
        }

    def generate(
        self,
        data: Union[pd.DataFrame, Dict, List],
        caption: str = "",
        columns: Optional[List[str]] = None,
        index: bool = False,
        format: Optional[str] = None,
        precision: Optional[int] = None,
        style: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a formatted table from data.

        Args:
            data: Input data (DataFrame, dict, or list).
            caption: Table caption.
            columns: Columns to include (None for all).
            index: Whether to include row index.
            format: Output format (uses default if None).
            precision: Decimal precision (uses default if None).
            style: Additional styling options.

        Returns:
            Formatted table string.

        Examples:
            >>> gen = TableGenerator()
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> print(gen.generate(df, caption="Sample Table"))
        """
        # Convert data to DataFrame
        df = self._to_dataframe(data)

        # Select columns if specified
        if columns:
            df = df[columns]

        # Apply precision formatting
        df = self._format_precision(df, precision or self.precision)

        # Apply additional styling
        if style:
            df = self._apply_style(df, style)

        # Generate table in specified format
        format_key = format or self.default_format
        table_format = self._format_map.get(format_key, "pipe")

        # Generate base table
        table = tabulate(
            df,
            headers=df.columns,
            tablefmt=table_format,
            showindex=index,
            floatfmt=f".{precision or self.precision}f",
        )

        # Add caption if provided
        if caption:
            table = self._add_caption(table, caption, format_key)

        return table  # type: ignore[no-any-return]

    def generate_summary_statistics(
        self, df: pd.DataFrame, metrics: Optional[List[str]] = None, format: Optional[str] = None
    ) -> str:
        """Generate summary statistics table.

        Args:
            df: Input DataFrame.
            metrics: List of metrics to compute.
            format: Output format.

        Returns:
            Formatted summary statistics table.
        """
        if metrics is None:
            metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

        summary = df.describe()

        # Filter to requested metrics
        available_metrics = [m for m in metrics if m in summary.index]
        summary = summary.loc[available_metrics]

        return self.generate(summary.T, caption="Summary Statistics", index=True, format=format)

    def generate_comparison_table(
        self,
        data: Dict[str, pd.Series],
        caption: str = "Comparison Table",
        format: Optional[str] = None,
    ) -> str:
        """Generate comparison table from multiple series.

        Args:
            data: Dictionary mapping names to series.
            caption: Table caption.
            format: Output format.

        Returns:
            Formatted comparison table.
        """
        df = pd.DataFrame(data)
        return self.generate(df, caption=caption, format=format, index=True)

    def generate_decision_matrix(
        self,
        alternatives: List[str],
        criteria: List[str],
        scores: np.ndarray,
        weights: Optional[List[float]] = None,
        format: Optional[str] = None,
    ) -> str:
        """Generate a decision matrix table.

        Args:
            alternatives: List of alternative names.
            criteria: List of criteria names.
            scores: Score matrix (alternatives x criteria).
            weights: Optional criteria weights.
            format: Output format.

        Returns:
            Formatted decision matrix.
        """
        df = pd.DataFrame(scores, index=alternatives, columns=criteria)

        # Add weighted scores if weights provided
        if weights is not None:
            weighted_scores = scores * np.array(weights)
            df["Weighted Total"] = weighted_scores.sum(axis=1)
            df = df.round(self.precision)

        return self.generate(df, caption="Decision Matrix", index=True, format=format)

    def _to_dataframe(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """Convert various data types to DataFrame.

        Args:
            data: Input data.

        Returns:
            pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _format_precision(self, df: pd.DataFrame, precision: int) -> pd.DataFrame:
        """Format numeric columns to specified precision.

        Args:
            df: Input DataFrame.
            precision: Number of decimal places.

        Returns:
            Formatted DataFrame.
        """
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(precision)
        return df

    def _apply_style(self, df: pd.DataFrame, style: Dict[str, Any]) -> pd.DataFrame:
        """Apply styling options to DataFrame.

        Args:
            df: Input DataFrame.
            style: Style dictionary.

        Returns:
            Styled DataFrame.
        """
        # Apply column width limits
        if "max_col_width" in style:
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].apply(
                    lambda x: str(x)[: style["max_col_width"]] + "..."
                    if len(str(x)) > style["max_col_width"]
                    else x
                )

        # Apply highlighting for specific values
        if "highlight" in style:
            highlight_config = style["highlight"]
            if "max" in highlight_config:
                # Could implement cell highlighting for max values
                pass
            if "min" in highlight_config:
                # Could implement cell highlighting for min values
                pass

        return df

    def _add_caption(self, table: str, caption: str, format: str) -> str:
        """Add caption to table based on format.

        Args:
            table: Table string.
            caption: Caption text.
            format: Table format.

        Returns:
            Table with caption.
        """
        if format == "markdown":
            return f"**Table: {caption}**\n\n{table}"
        elif format == "html":
            return f"<table>\n<caption>{caption}</caption>\n{table}\n</table>"
        elif format == "latex":
            return f"\\begin{{table}}[htbp]\n\\caption{{{caption}}}\n{table}\n\\end{{table}}"
        else:
            return f"{caption}\n{'-' * len(caption)}\n{table}"


def create_performance_table(results: Dict[str, Any]) -> str:
    """Create a performance metrics table from simulation results.

    Args:
        results: Dictionary of performance metrics.

    Returns:
        Formatted performance table.
    """
    gen = TableGenerator()

    # Extract metrics
    metrics_df = pd.DataFrame(
        {
            "Metric": ["ROE", "Ruin Probability", "Growth Rate", "Sharpe Ratio", "Max Drawdown"],
            "Value": [
                results.get("roe", 0.0),
                results.get("ruin_prob", 0.0),
                results.get("growth_rate", 0.0),
                results.get("sharpe", 0.0),
                results.get("max_drawdown", 0.0),
            ],
            "Status": [
                "✓" if results.get("roe", 0) > 0.15 else "⚠",
                "✓" if results.get("ruin_prob", 1) < 0.01 else "✗",
                "✓" if results.get("growth_rate", 0) > 0.05 else "⚠",
                "✓" if results.get("sharpe", 0) > 1.0 else "⚠",
                "✓" if results.get("max_drawdown", 1) < 0.2 else "⚠",
            ],
        }
    )

    return gen.generate(metrics_df, caption="Performance Metrics", format="markdown")


def create_parameter_table(params: Dict[str, Any]) -> str:
    """Create a parameter summary table.

    Args:
        params: Dictionary of parameters.

    Returns:
        Formatted parameter table.
    """
    gen = TableGenerator()

    # Flatten nested parameters
    flat_params = []
    for category, values in params.items():
        if isinstance(values, dict):
            for key, val in values.items():
                flat_params.append({"Category": category, "Parameter": key, "Value": val})
        else:
            flat_params.append({"Category": "General", "Parameter": category, "Value": values})

    df = pd.DataFrame(flat_params)
    return gen.generate(df, caption="Model Parameters", format="markdown")


def create_sensitivity_table(
    base_case: float,
    sensitivities: Dict[str, List[float]],
    parameter_ranges: Dict[str, List[float]],
) -> str:
    """Create a sensitivity analysis table.

    Args:
        base_case: Base case value.
        sensitivities: Sensitivity results by parameter.
        parameter_ranges: Parameter test ranges.

    Returns:
        Formatted sensitivity table.
    """
    gen = TableGenerator()

    rows = []
    for param, values in sensitivities.items():
        ranges = parameter_ranges.get(param, [])
        for i, (range_val, result) in enumerate(zip(ranges, values)):
            rows.append(
                {
                    "Parameter": param if i == 0 else "",
                    "Test Value": range_val,
                    "Result": result,
                    "Change from Base": f"{(result/base_case - 1)*100:.1f}%",
                }
            )

    df = pd.DataFrame(rows)
    return gen.generate(df, caption="Sensitivity Analysis", format="markdown")

"""Table generation utilities for report creation.

This module provides functions to generate and format tables from various data sources,
supporting multiple output formats including Markdown, HTML, and LaTeX.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tabulate import tabulate  # type: ignore[import-untyped]

from .formatters import ColorCoder, NumberFormatter, TableFormatter, format_for_export


class TableGenerator:
    """Generate formatted tables for reports.

    This class provides methods to create professionally formatted tables
    from pandas DataFrames or raw data, supporting various output formats
    and styling options. Enhanced with comprehensive table generation methods
    for executive and technical reports.

    Attributes:
        default_format: Default output format for tables.
        precision: Default precision for numeric values.
        max_width: Maximum width for text columns.
        number_formatter: NumberFormatter instance for formatting.
        color_coder: ColorCoder instance for color coding.
        table_formatter: TableFormatter instance for comprehensive formatting.
    """

    def __init__(
        self,
        default_format: Literal["markdown", "html", "latex", "grid", "csv", "excel"] = "markdown",
        precision: int = 2,
        max_width: int = 50,
        currency_symbol: str = "$",
    ):
        """Initialize TableGenerator.

        Args:
            default_format: Default output format.
            precision: Number of decimal places for floats.
            max_width: Maximum width for text columns.
            currency_symbol: Symbol for currency formatting.
        """
        self.default_format = default_format
        self.precision = precision
        self.max_width = max_width
        self.currency_symbol = currency_symbol
        self._format_map = {
            "markdown": "pipe",
            "html": "html",
            "latex": "latex_booktabs",
            "grid": "grid",
            "csv": "csv",
            "excel": "excel",
        }

        # Map TableGenerator formats to ColorCoder/TableFormatter formats
        formatter_format: Literal["html", "latex", "terminal", "none"]
        if default_format == "html":
            formatter_format = "html"
        elif default_format == "latex":
            formatter_format = "latex"
        else:
            formatter_format = "none"  # Default for markdown, grid, csv, excel

        # Initialize formatters
        self.number_formatter = NumberFormatter(
            currency_symbol=currency_symbol,
            decimal_places=precision,
        )
        self.color_coder = ColorCoder(output_format=formatter_format)
        self.table_formatter = TableFormatter(
            output_format=formatter_format,
            currency_symbol=currency_symbol,
            decimal_places=precision,
        )

    def generate(
        self,
        data: Union[pd.DataFrame, Dict, List],
        caption: str = "",
        columns: Optional[List[str]] = None,
        index: bool = False,
        output_format: Optional[str] = None,
        precision: Optional[int] = None,
        style: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a formatted table from data.

        Args:
            data: Input data (DataFrame, dict, or list).
            caption: Table caption.
            columns: Columns to include (None for all).
            index: Whether to include row index.
            output_format: Output format (uses default if None).
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
        format_key = output_format or self.default_format
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
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """Generate summary statistics table.

        Args:
            df: Input DataFrame.
            metrics: List of metrics to compute.
            output_format: Output format.

        Returns:
            Formatted summary statistics table.
        """
        if metrics is None:
            metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

        summary = df.describe()

        # Filter to requested metrics
        available_metrics = [m for m in metrics if m in summary.index]
        summary = summary.loc[available_metrics]

        return self.generate(
            summary.T, caption="Summary Statistics", index=True, output_format=output_format
        )

    def generate_comparison_table(
        self,
        data: Dict[str, pd.Series],
        caption: str = "Comparison Table",
        output_format: Optional[str] = None,
    ) -> str:
        """Generate comparison table from multiple series.

        Args:
            data: Dictionary mapping names to series.
            caption: Table caption.
            output_format: Output format.

        Returns:
            Formatted comparison table.
        """
        df = pd.DataFrame(data)
        return self.generate(df, caption=caption, output_format=output_format, index=True)

    def generate_decision_matrix(
        self,
        alternatives: List[str],
        criteria: List[str],
        scores: np.ndarray,
        weights: Optional[List[float]] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """Generate a decision matrix table.

        Args:
            alternatives: List of alternative names.
            criteria: List of criteria names.
            scores: Score matrix (alternatives x criteria).
            weights: Optional criteria weights.
            output_format: Output format.

        Returns:
            Formatted decision matrix.
        """
        df = pd.DataFrame(scores, index=alternatives, columns=criteria)

        # Add weighted scores if weights provided
        if weights is not None:
            weighted_scores = scores * np.array(weights)
            df["Weighted Total"] = weighted_scores.sum(axis=1)
            df = df.round(self.precision)

        return self.generate(df, caption="Decision Matrix", index=True, output_format=output_format)

    def _to_dataframe(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """Convert various data types to DataFrame.

        Args:
            data: Input data.

        Returns:
            pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, list):
            return pd.DataFrame(data)
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
                    lambda x: (
                        str(x)[: style["max_col_width"]] + "..."
                        if len(str(x)) > style["max_col_width"]
                        else x
                    )
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

    def _add_caption(self, table: str, caption: str, output_format: str) -> str:
        """Add caption to table based on format.

        Args:
            table: Table string.
            caption: Caption text.
            output_format: Table format.

        Returns:
            Table with caption.
        """
        if output_format == "markdown":
            return f"**Table: {caption}**\n\n{table}"
        if output_format == "html":
            return f"<table>\n<caption>{caption}</caption>\n{table}\n</table>"
        if output_format == "latex":
            return f"\\begin{{table}}[htbp]\n\\caption{{{caption}}}\n{table}\n\\end{{table}}"
        return f"{caption}\n{'-' * len(caption)}\n{table}"

    # Executive Table Methods

    def generate_optimal_limits_by_size(
        self,
        company_sizes: List[float],
        optimal_limits: Dict[float, Dict[str, float]],
        output_format: Optional[str] = None,
        include_percentages: bool = True,
    ) -> str:
        """Generate Table 1: Optimal Insurance Limits by Company Size.

        Creates a comprehensive table showing optimal insurance structures
        for different company sizes, including retention, primary limits,
        excess limits, and total premiums.

        Args:
            company_sizes: List of company asset sizes in dollars.
            optimal_limits: Dict mapping size to limits structure.
            output_format: Output format (uses default if None).
            include_percentages: Whether to include % of assets columns.

        Returns:
            Formatted table string.

        Examples:
            >>> gen = TableGenerator()
            >>> sizes = [1_000_000, 10_000_000, 100_000_000]
            >>> limits = {
            ...     1_000_000: {'retention': 50000, 'primary': 500000,
            ...                 'excess': 1000000, 'premium': 25000},
            ...     # ... more sizes
            ... }
            >>> table = gen.generate_optimal_limits_by_size(sizes, limits)
        """
        rows = []
        for size in company_sizes:
            limits = optimal_limits.get(size, {})
            row = {
                "Company Size": self.number_formatter.format_currency(size, abbreviate=True),
                "Retention": self.number_formatter.format_currency(
                    limits.get("retention", 0), abbreviate=True
                ),
                "Primary Limit": self.number_formatter.format_currency(
                    limits.get("primary", 0), abbreviate=True
                ),
                "Excess Limit": self.number_formatter.format_currency(
                    limits.get("excess", 0), abbreviate=True
                ),
                "Total Premium": self.number_formatter.format_currency(
                    limits.get("premium", 0), abbreviate=True
                ),
            }

            if include_percentages:
                row["Retention %"] = self.number_formatter.format_percentage(
                    limits.get("retention", 0) / size if size > 0 else 0
                )
                row["Primary %"] = self.number_formatter.format_percentage(
                    limits.get("primary", 0) / size if size > 0 else 0
                )
                row["Premium %"] = self.number_formatter.format_percentage(
                    limits.get("premium", 0) / size if size > 0 else 0
                )

            rows.append(row)

        df = pd.DataFrame(rows)
        return self.generate(
            df,
            caption="Optimal Insurance Limits by Company Size",
            output_format=output_format,
        )

    def generate_quick_reference_matrix(
        self,
        characteristics: List[str],
        recommendations: Dict[str, Dict[str, Any]],
        output_format: Optional[str] = None,
        use_traffic_lights: bool = True,
    ) -> str:
        """Generate Table 2: Quick Reference Decision Matrix.

        Creates a decision matrix with company characteristics as rows
        and recommended insurance structures as columns, with optional
        traffic light coloring for visual indicators.

        Args:
            characteristics: List of company characteristic names.
            recommendations: Dict mapping characteristics to recommendations.
            output_format: Output format.
            use_traffic_lights: Whether to apply traffic light coloring.

        Returns:
            Formatted decision matrix table.

        Examples:
            >>> gen = TableGenerator()
            >>> chars = ['High Growth', 'Stable', 'Distressed']
            >>> recs = {
            ...     'High Growth': {'retention': 'Low', 'coverage': 'High',
            ...                     'premium_budget': '2-3%', 'risk_level': 'good'},
            ...     # ... more recommendations
            ... }
            >>> table = gen.generate_quick_reference_matrix(chars, recs)
        """
        rows = []
        for char in characteristics:
            rec = recommendations.get(char, {})
            row = {
                "Company Profile": char,
                "Retention Level": rec.get("retention", "-"),
                "Coverage Level": rec.get("coverage", "-"),
                "Premium Budget": rec.get("premium_budget", "-"),
                "Excess Layers": rec.get("excess_layers", "-"),
            }

            if use_traffic_lights and "risk_level" in rec:
                risk_level = rec["risk_level"]
                # Apply traffic light indicators
                if risk_level == "good":
                    row["Risk Assessment"] = "✓ Low Risk"
                elif risk_level == "warning":
                    row["Risk Assessment"] = "⚠ Medium Risk"
                else:
                    row["Risk Assessment"] = "✗ High Risk"
            else:
                row["Risk Assessment"] = rec.get("risk_assessment", "-")

            rows.append(row)

        df = pd.DataFrame(rows)
        return self.generate(
            df,
            caption="Quick Reference - Insurance Decision Matrix",
            output_format=output_format,
        )

    # Technical Table Methods

    def generate_parameter_grid(
        self,
        parameters: Dict[str, Dict[str, Any]],
        scenarios: Optional[List[str]] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """Generate Table A1: Complete Parameter Grid.

        Creates a comprehensive parameter grid showing all simulation
        parameters with ranges for different scenarios.

        Args:
            parameters: Dict of parameter categories and values.
            scenarios: List of scenario names.
            output_format: Output format.

        Returns:
            Formatted parameter grid table.

        Examples:
            >>> gen = TableGenerator()
            >>> params = {
            ...     'Growth': {'mean': [0.05, 0.08, 0.12], 'volatility': [0.15, 0.20, 0.30]},
            ...     'Losses': {'frequency': [3, 5, 8], 'severity': [50000, 100000, 200000]}
            ... }
            >>> table = gen.generate_parameter_grid(params)
        """
        if scenarios is None:
            scenarios = ["Baseline", "Conservative", "Aggressive"]

        rows = []
        for category, params in parameters.items():
            for param_name, values in params.items():
                row = {
                    "Category": category,
                    "Parameter": param_name,
                }

                if isinstance(values, list) and len(values) == len(scenarios):
                    for i, scenario in enumerate(scenarios):
                        value = values[i]
                        # Format based on value type
                        if isinstance(value, float) and value < 1:
                            formatted = self.number_formatter.format_percentage(value)
                        elif isinstance(value, (int, float)) and value > 10000:
                            formatted = self.number_formatter.format_currency(
                                value, abbreviate=True
                            )
                        else:
                            formatted = str(value)
                        row[scenario] = formatted
                else:
                    # Single value for all scenarios
                    formatted = str(values)
                    for scenario in scenarios:
                        row[scenario] = formatted

                rows.append(row)

        df = pd.DataFrame(rows)
        return self.generate(
            df,
            caption="Complete Parameter Grid - All Scenarios",
            output_format=output_format,
        )

    def generate_loss_distribution_params(
        self,
        loss_types: List[str],
        distribution_params: Dict[str, Dict[str, Any]],
        output_format: Optional[str] = None,
        include_correlations: bool = True,
    ) -> str:
        """Generate Table A2: Loss Distribution Parameters.

        Creates a table showing frequency and severity parameters for
        different loss types, including correlation coefficients and
        development patterns.

        Args:
            loss_types: List of loss type names.
            distribution_params: Parameters for each loss type.
            output_format: Output format.
            include_correlations: Whether to include correlation matrix.

        Returns:
            Formatted loss distribution parameters table.
        """
        rows = []
        for loss_type in loss_types:
            params = distribution_params.get(loss_type, {})
            row = {
                "Loss Type": loss_type,
                "Frequency (λ)": params.get("frequency", 0),
                "Severity Mean": self.number_formatter.format_currency(
                    params.get("severity_mean", 0), abbreviate=True
                ),
                "Severity Std": self.number_formatter.format_currency(
                    params.get("severity_std", 0), abbreviate=True
                ),
                "Distribution": params.get("distribution", "Lognormal"),
                "Development Pattern": params.get("development", "Immediate"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if include_correlations and "correlations" in distribution_params:
            # Add correlation information as a note
            corr_note = "\nCorrelation Matrix:\n"
            # Convert correlations dict to proper format for DataFrame
            corr_data = distribution_params["correlations"]
            if isinstance(corr_data, dict):
                # If it's a flat dict of correlations, just list them
                corr_lines = []
                for pair, value in corr_data.items():
                    corr_lines.append(f"{pair}: {value}")
                corr_note += "\n".join(corr_lines)
            # Note: Removed unreachable else branch since correlations is always a dict

            base_table = self.generate(
                df,
                caption="Loss Distribution Parameters",
                output_format=output_format,
            )
            return base_table + corr_note

        return self.generate(
            df,
            caption="Loss Distribution Parameters",
            output_format=output_format,
        )

    def generate_insurance_pricing_grid(
        self,
        layers: List[Tuple[float, float]],
        pricing_params: Dict[Tuple[float, float], Dict[str, float]],
        output_format: Optional[str] = None,
    ) -> str:
        """Generate Table A3: Insurance Pricing Grid.

        Creates a pricing grid showing premium rates by layer and
        attachment point, including loading factors.

        Args:
            layers: List of (attachment, limit) tuples.
            pricing_params: Pricing parameters for each layer.
            output_format: Output format.

        Returns:
            Formatted insurance pricing grid.

        Examples:
            >>> gen = TableGenerator()
            >>> layers = [(0, 1_000_000), (1_000_000, 5_000_000)]
            >>> pricing = {
            ...     (0, 1_000_000): {'rate': 0.015, 'loading': 1.3},
            ...     (1_000_000, 5_000_000): {'rate': 0.008, 'loading': 1.2}
            ... }
            >>> table = gen.generate_insurance_pricing_grid(layers, pricing)
        """
        rows = []
        for attachment, limit in layers:
            params = pricing_params.get((attachment, limit), {})
            row = {
                "Layer": f"{self.number_formatter.format_currency(attachment, abbreviate=True)} x {self.number_formatter.format_currency(limit, abbreviate=True)}",
                "Attachment Point": self.number_formatter.format_currency(
                    attachment, abbreviate=True
                ),
                "Limit": self.number_formatter.format_currency(limit, abbreviate=True),
                "Base Rate": self.number_formatter.format_percentage(params.get("rate", 0)),
                "Loading Factor": self.number_formatter.format_ratio(params.get("loading", 1.0)),
                "Expense Ratio": self.number_formatter.format_percentage(
                    params.get("expense_ratio", 0)
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return self.generate(
            df,
            caption="Insurance Layer Pricing Grid",
            output_format=output_format,
        )

    def generate_statistical_validation(
        self,
        metrics: Dict[str, Dict[str, float]],
        output_format: Optional[str] = None,
        include_thresholds: bool = True,
    ) -> str:
        """Generate Table B1: Statistical Validation Metrics.

        Creates a table of statistical validation metrics including
        goodness-of-fit tests, convergence statistics, and out-of-sample
        performance.

        Args:
            metrics: Dictionary of metric categories and values.
            output_format: Output format.
            include_thresholds: Whether to include pass/fail thresholds.

        Returns:
            Formatted statistical validation table.
        """
        rows = []

        # Define thresholds for pass/fail
        thresholds = {
            "KS Statistic": 0.05,
            "Anderson-Darling": 2.5,
            "R-squared": 0.8,
            "RMSE": 0.1,
            "Convergence R-hat": 1.1,
            "ESS per Chain": 1000,
        }

        for category, category_metrics in metrics.items():
            for metric_name, value in category_metrics.items():
                row = {
                    "Category": category,
                    "Metric": metric_name,
                    "Value": self.number_formatter.format_number(value, decimals=4),
                }

                if include_thresholds and metric_name in thresholds:
                    threshold = thresholds[metric_name]
                    row["Threshold"] = str(threshold)

                    # Determine pass/fail
                    if metric_name in ["R-squared", "ESS per Chain"]:
                        # Higher is better
                        passed = value >= threshold
                    else:
                        # Lower is better
                        passed = value <= threshold

                    row["Status"] = "✓ Pass" if passed else "✗ Fail"

                rows.append(row)

        df = pd.DataFrame(rows)
        return self.generate(
            df,
            caption="Statistical Validation Metrics",
            output_format=output_format,
        )

    def generate_comprehensive_results(
        self,
        results: List[Dict[str, Any]],
        ranking_metric: str = "roe",
        output_format: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> str:
        """Generate Table C1: Comprehensive Optimization Results.

        Creates a comprehensive results table showing all parameter
        combinations tested with ranking by specified metric.

        Args:
            results: List of result dictionaries from optimization.
            ranking_metric: Metric to use for ranking.
            output_format: Output format.
            top_n: Show only top N results (None for all).

        Returns:
            Formatted comprehensive results table.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Sort by ranking metric
        if ranking_metric in df.columns:
            ascending = ranking_metric in ["ruin_probability", "max_drawdown"]
            df = df.sort_values(ranking_metric, ascending=ascending)

        # Limit to top N if specified
        if top_n:
            df = df.head(top_n)

        # Format columns appropriately
        formatted_rows: List[Dict[str, Union[int, str]]] = []
        for _, row in df.iterrows():
            formatted_row: Dict[str, Union[int, str]] = {}

            # Add rank
            formatted_row["Rank"] = len(formatted_rows) + 1

            # Format parameters
            if "retention" in row:
                formatted_row["Retention"] = self.number_formatter.format_currency(
                    row["retention"], abbreviate=True
                )
            if "primary_limit" in row:
                formatted_row["Primary"] = self.number_formatter.format_currency(
                    row["primary_limit"], abbreviate=True
                )
            if "excess_limit" in row:
                formatted_row["Excess"] = self.number_formatter.format_currency(
                    row["excess_limit"], abbreviate=True
                )
            if "premium" in row:
                formatted_row["Premium"] = self.number_formatter.format_currency(
                    row["premium"], abbreviate=True
                )

            # Format metrics
            if "roe" in row:
                formatted_row["ROE"] = self.number_formatter.format_percentage(row["roe"])
            if "ruin_probability" in row:
                formatted_row["Ruin Prob"] = self.number_formatter.format_percentage(
                    row["ruin_probability"]
                )
            if "growth_rate" in row:
                formatted_row["Growth"] = self.number_formatter.format_percentage(
                    row["growth_rate"]
                )
            if "sharpe_ratio" in row:
                formatted_row["Sharpe"] = self.number_formatter.format_number(
                    row["sharpe_ratio"], decimals=2
                )

            formatted_rows.append(formatted_row)

        result_df = pd.DataFrame(formatted_rows)
        return self.generate(
            result_df,
            caption=f"Comprehensive Optimization Results (Ranked by {ranking_metric})",
            output_format=output_format,
        )

    def generate_walk_forward_validation(
        self,
        validation_results: List[Dict[str, Any]],
        output_format: Optional[str] = None,
    ) -> str:
        """Generate Table C2: Walk-Forward Validation Results.

        Creates a table showing rolling window analysis results and
        strategy stability metrics over time.

        Args:
            validation_results: List of validation period results.
            output_format: Output format.

        Returns:
            Formatted walk-forward validation table.
        """
        rows = []
        for result in validation_results:
            row = {
                "Period": result.get("period", "-"),
                "In-Sample ROE": self.number_formatter.format_percentage(
                    result.get("in_sample_roe", 0)
                ),
                "Out-Sample ROE": self.number_formatter.format_percentage(
                    result.get("out_sample_roe", 0)
                ),
                "Tracking Error": self.number_formatter.format_percentage(
                    result.get("tracking_error", 0)
                ),
                "Strategy Change": result.get("strategy_change", "No"),
                "Stability Score": self.number_formatter.format_percentage(
                    result.get("stability_score", 0)
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add summary statistics row
        summary_row = {
            "Period": "Average",
            "In-Sample ROE": self.number_formatter.format_percentage(
                np.mean([r.get("in_sample_roe", 0) for r in validation_results])
            ),
            "Out-Sample ROE": self.number_formatter.format_percentage(
                np.mean([r.get("out_sample_roe", 0) for r in validation_results])
            ),
            "Tracking Error": self.number_formatter.format_percentage(
                np.mean([r.get("tracking_error", 0) for r in validation_results])
            ),
            "Strategy Change": f"{sum(r.get('strategy_change', 'No') == 'Yes' for r in validation_results)}/{len(validation_results)}",
            "Stability Score": self.number_formatter.format_percentage(
                np.mean([r.get("stability_score", 0) for r in validation_results])
            ),
        }

        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        return self.generate(
            df,
            caption="Walk-Forward Validation Results",
            output_format=output_format,
        )

    def export_to_file(
        self,
        df: pd.DataFrame,
        file_path: str,
        output_format: Literal["csv", "excel", "latex", "html"],
        **kwargs,
    ) -> None:
        """Export DataFrame to file in specified format.

        Args:
            df: DataFrame to export.
            file_path: Path to save file.
            output_format: Export format.
            **kwargs: Additional format-specific arguments.
        """
        if output_format == "excel":
            df.to_excel(file_path, index=False, **kwargs)
        elif output_format == "csv":
            df.to_csv(file_path, index=False, **kwargs)
        elif output_format == "latex":
            latex_str = format_for_export(df, "latex", **kwargs)
            if latex_str is not None:
                Path(file_path).write_text(latex_str)
        elif output_format == "html":
            html_str = format_for_export(df, "html", **kwargs)
            if html_str is not None:
                Path(file_path).write_text(html_str)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")


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

    return gen.generate(metrics_df, caption="Performance Metrics", output_format="markdown")


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
    return gen.generate(df, caption="Model Parameters", output_format="markdown")


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
    return gen.generate(df, caption="Sensitivity Analysis", output_format="markdown")

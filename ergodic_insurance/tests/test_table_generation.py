"""Comprehensive tests for table generation and formatting functionality.

This module tests the TableGenerator class and formatters module to ensure
all table generation methods work correctly with proper formatting and validation.

Google-style docstrings are used throughout for consistency.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.reporting.formatters import (
    ColorCoder,
    NumberFormatter,
    TableFormatter,
    format_for_export,
)
from ergodic_insurance.src.reporting.table_generator import (
    TableGenerator,
    create_parameter_table,
    create_performance_table,
    create_sensitivity_table,
)


class TestNumberFormatter:
    """Test NumberFormatter class for number formatting utilities."""

    def test_format_currency(self):
        """Test currency formatting with various options."""
        formatter = NumberFormatter()

        # Basic formatting
        assert formatter.format_currency(1234567.89) == "$1,234,567.89"
        assert formatter.format_currency(1000) == "$1,000.00"
        assert formatter.format_currency(0) == "$0.00"

        # Abbreviated formatting
        assert formatter.format_currency(1234567.89, abbreviate=True) == "$1.23M"
        assert formatter.format_currency(1234567890, abbreviate=True) == "$1.23B"
        assert formatter.format_currency(1234, abbreviate=True) == "$1.23K"

        # Custom decimals
        assert formatter.format_currency(1234.567, decimals=3) == "$1,234.567"
        assert formatter.format_currency(1234.567, decimals=0) == "$1,235"

        # Handle NaN
        assert formatter.format_currency(np.nan) == "-"

    def test_format_percentage(self):
        """Test percentage formatting."""
        formatter = NumberFormatter()

        # Basic percentage
        assert formatter.format_percentage(0.1234) == "12.34%"
        assert formatter.format_percentage(0.01) == "1.00%"
        assert formatter.format_percentage(1.5) == "150.00%"

        # Without multiplication
        assert formatter.format_percentage(12.34, multiply_by_100=False) == "12.34%"

        # Custom decimals
        assert formatter.format_percentage(0.12345, decimals=3) == "12.345%"
        assert formatter.format_percentage(0.12345, decimals=0) == "12%"

        # Handle NaN
        assert formatter.format_percentage(np.nan) == "-"

    def test_format_number(self):
        """Test general number formatting."""
        formatter = NumberFormatter()

        # Basic formatting
        assert formatter.format_number(1234567.89) == "1,234,567.89"
        assert formatter.format_number(0.123) == "0.12"

        # Scientific notation
        assert formatter.format_number(1234567890, scientific=True) == "1.23e+09"
        assert formatter.format_number(0.00001234, scientific=True) == "1.23e-05"

        # Abbreviation
        assert formatter.format_number(1234567, abbreviate=True) == "1.23M"

        # Handle NaN
        assert formatter.format_number(np.nan) == "-"

    def test_format_ratio(self):
        """Test ratio formatting."""
        formatter = NumberFormatter()

        assert formatter.format_ratio(1.5) == "1.50x"
        assert formatter.format_ratio(0.75, decimals=3) == "0.750x"
        assert formatter.format_ratio(np.nan) == "-"

    def test_custom_separators(self):
        """Test custom thousand and decimal separators."""
        formatter = NumberFormatter(thousands_separator=" ", decimal_separator=",")

        assert formatter.format_currency(1234567.89) == "$1 234 567,89"
        assert formatter.format_number(1234.56) == "1 234,56"


class TestColorCoder:
    """Test ColorCoder class for color coding functionality."""

    def test_traffic_light_html(self):
        """Test traffic light coloring for HTML output."""
        coder = ColorCoder(output_format="html")

        thresholds = {"good": (0.15, None), "warning": (0.10, 0.15), "bad": (None, 0.10)}

        # Test good value
        result = coder.traffic_light(0.18, thresholds)
        assert "#28a745" in result  # Green color
        assert "0.18" in result

        # Test warning value
        result = coder.traffic_light(0.12, thresholds)
        assert "#ffc107" in result  # Yellow color

        # Test bad value
        result = coder.traffic_light(0.05, thresholds)
        assert "#dc3545" in result  # Red color

        # Test with custom text
        result = coder.traffic_light(0.18, thresholds, text="Good Performance")
        assert "Good Performance" in result

    def test_traffic_light_terminal(self):
        """Test traffic light coloring for terminal output."""
        coder = ColorCoder(output_format="terminal")

        thresholds = {"good": (0.15, None), "warning": (0.10, 0.15), "bad": (None, 0.10)}

        # Test indicators
        assert "✓" in coder.traffic_light(0.18, thresholds)
        assert "⚠" in coder.traffic_light(0.12, thresholds)
        assert "✗" in coder.traffic_light(0.05, thresholds)

    def test_heatmap(self):
        """Test heatmap coloring."""
        coder = ColorCoder(output_format="html")

        result = coder.heatmap(50, 0, 100)
        assert "background-color" in result
        assert "50" in result

        # Test edge cases
        result_low = coder.heatmap(10, 0, 100)
        result_high = coder.heatmap(90, 0, 100)

        # Different colors should be applied
        assert result_low != result_high

    def test_threshold_color(self):
        """Test threshold-based coloring."""
        coder = ColorCoder(output_format="html")

        # Above threshold (green)
        result = coder.threshold_color(0.8, 0.5)
        assert "#28a745" in result

        # Below threshold (red)
        result = coder.threshold_color(0.3, 0.5)
        assert "#dc3545" in result


class TestTableFormatter:
    """Test TableFormatter class for comprehensive table formatting."""

    def test_format_dataframe(self):
        """Test DataFrame formatting with various column types."""
        formatter = TableFormatter(output_format="none")

        df = pd.DataFrame(
            {
                "Revenue": [1000000, 2000000, 3000000],
                "Growth": [0.10, 0.15, 0.20],
                "Risk": [0.01, 0.05, 0.10],
                "Status": ["Active", "Active", "Inactive"],
            }
        )

        column_formats: Dict[str, Dict[str, Any]] = {
            "Revenue": {"type": "currency", "abbreviate": True},
            "Growth": {"type": "percentage"},
            "Risk": {"type": "percentage", "decimals": 1},
        }

        formatted_df = formatter.format_dataframe(df, column_formats)

        # Check formatting applied
        assert "$1.00M" in str(formatted_df["Revenue"].iloc[0])
        assert "10.00%" in str(formatted_df["Growth"].iloc[0])
        assert "1.0%" in str(formatted_df["Risk"].iloc[0])
        assert formatted_df["Status"].iloc[0] == "Active"

    def test_add_totals_row(self):
        """Test adding totals row to DataFrame."""
        formatter = TableFormatter()

        df = pd.DataFrame(
            {"Category": ["A", "B", "C"], "Value": [100, 200, 300], "Count": [10, 20, 30]}
        )

        # Add sum totals
        df_with_totals = formatter.add_totals_row(df, ["Value", "Count"])
        assert len(df_with_totals) == 4
        assert df_with_totals["Value"].iloc[-1] == 600
        assert df_with_totals["Count"].iloc[-1] == 60

        # Add mean totals
        df_with_mean = formatter.add_totals_row(df, ["Value"], operation="mean")
        assert df_with_mean["Value"].iloc[-1] == 200

    def test_add_footnotes(self):
        """Test adding footnotes to tables."""
        formatter = TableFormatter()

        table_str = "Sample table content"
        footnotes = ["Note 1", "Note 2"]

        # Test markdown format
        result = formatter.add_footnotes(table_str, footnotes, output_format="none")
        assert "[1] Note 1" in result
        assert "[2] Note 2" in result

        # Test HTML format
        formatter.output_format = "html"
        result = formatter.add_footnotes(table_str, footnotes, output_format="html")
        assert "<sup>1</sup>" in result
        assert "<sup>2</sup>" in result


class TestTableGenerator:
    """Test enhanced TableGenerator class with all new table methods."""

    @pytest.fixture
    def generator(self):
        """Create a TableGenerator instance for testing."""
        return TableGenerator(default_format="markdown")

    def test_basic_generation(self, generator):
        """Test basic table generation functionality."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        result = generator.generate(df, caption="Test Table")
        assert "Test Table" in result
        assert "|" in result  # Markdown table format

    def test_generate_optimal_limits_by_size(self, generator):
        """Test Table 1: Optimal Insurance Limits by Company Size."""
        company_sizes = [1_000_000, 10_000_000, 100_000_000]
        optimal_limits = {
            1_000_000: {
                "retention": 50_000,
                "primary": 500_000,
                "excess": 1_000_000,
                "premium": 25_000,
            },
            10_000_000: {
                "retention": 250_000,
                "primary": 2_500_000,
                "excess": 5_000_000,
                "premium": 150_000,
            },
            100_000_000: {
                "retention": 1_000_000,
                "primary": 10_000_000,
                "excess": 25_000_000,
                "premium": 1_000_000,
            },
        }

        result = generator.generate_optimal_limits_by_size(
            company_sizes, optimal_limits, include_percentages=True
        )

        assert "Optimal Insurance Limits by Company Size" in result
        assert "$1.00M" in result  # Company size formatting
        assert "$50.00K" in result  # Retention formatting
        assert "5.00%" in result  # Percentage formatting

    def test_generate_quick_reference_matrix(self, generator):
        """Test Table 2: Quick Reference Decision Matrix."""
        characteristics = ["High Growth", "Stable", "Distressed"]
        recommendations = {
            "High Growth": {
                "retention": "Low",
                "coverage": "High",
                "premium_budget": "2-3%",
                "excess_layers": "Multiple",
                "risk_level": "good",
            },
            "Stable": {
                "retention": "Medium",
                "coverage": "Medium",
                "premium_budget": "1-2%",
                "excess_layers": "1-2",
                "risk_level": "warning",
            },
            "Distressed": {
                "retention": "High",
                "coverage": "Low",
                "premium_budget": "<1%",
                "excess_layers": "None",
                "risk_level": "bad",
            },
        }

        result = generator.generate_quick_reference_matrix(
            characteristics, recommendations, use_traffic_lights=True
        )

        assert "Quick Reference - Insurance Decision Matrix" in result
        assert "High Growth" in result
        assert "✓ Low Risk" in result
        assert "⚠ Medium Risk" in result
        assert "✗ High Risk" in result

    def test_generate_parameter_grid(self, generator):
        """Test Table A1: Complete Parameter Grid."""
        parameters = {
            "Growth": {"Mean Return": [0.05, 0.08, 0.12], "Volatility": [0.15, 0.20, 0.30]},
            "Losses": {"Frequency": [3, 5, 8], "Severity": [50_000, 100_000, 200_000]},
        }

        result = generator.generate_parameter_grid(
            parameters, scenarios=["Baseline", "Conservative", "Aggressive"]
        )

        assert "Complete Parameter Grid" in result
        assert "Growth" in result
        assert "Mean Return" in result
        assert "5.00%" in result  # Percentage formatting
        assert "$50.00K" in result  # Currency formatting

    def test_generate_loss_distribution_params(self, generator):
        """Test Table A2: Loss Distribution Parameters."""
        loss_types = ["Attritional", "Large", "Catastrophic"]
        distribution_params = {
            "Attritional": {
                "frequency": 5.0,
                "severity_mean": 50_000,
                "severity_std": 20_000,
                "distribution": "Lognormal",
                "development": "Immediate",
            },
            "Large": {
                "frequency": 0.5,
                "severity_mean": 1_000_000,
                "severity_std": 500_000,
                "distribution": "Lognormal",
                "development": "6 months",
            },
            "Catastrophic": {
                "frequency": 0.1,
                "severity_mean": 10_000_000,
                "severity_std": 5_000_000,
                "distribution": "Pareto",
                "development": "12 months",
            },
            "correlations": {
                "Attritional-Large": 0.3,
                "Attritional-Catastrophic": 0.1,
                "Large-Catastrophic": 0.5,
            },
        }

        result = generator.generate_loss_distribution_params(
            loss_types, distribution_params, include_correlations=True
        )

        assert "Loss Distribution Parameters" in result
        assert "Attritional" in result
        assert "Frequency (λ)" in result
        assert "$50.00K" in result
        assert "Lognormal" in result
        assert "Correlation Matrix" in result

    def test_generate_insurance_pricing_grid(self, generator):
        """Test Table A3: Insurance Pricing Grid."""
        layers = [(0, 1_000_000), (1_000_000, 5_000_000), (5_000_000, 25_000_000)]
        pricing_params = {
            (0, 1_000_000): {"rate": 0.015, "loading": 1.3, "expense_ratio": 0.15},
            (1_000_000, 5_000_000): {"rate": 0.008, "loading": 1.2, "expense_ratio": 0.10},
            (5_000_000, 25_000_000): {"rate": 0.004, "loading": 1.1, "expense_ratio": 0.08},
        }

        result = generator.generate_insurance_pricing_grid(layers, pricing_params)

        assert "Insurance Layer Pricing Grid" in result
        assert "$0.00 x $1.00M" in result  # Fixed to match actual formatting
        assert "1.50%" in result  # Base rate
        assert "1.30x" in result  # Loading factor
        assert "15.00%" in result  # Expense ratio

    def test_generate_statistical_validation(self, generator):
        """Test Table B1: Statistical Validation Metrics."""
        metrics = {
            "Goodness of Fit": {
                "KS Statistic": 0.045,
                "Anderson-Darling": 2.1,
                "Chi-Square p-value": 0.15,
            },
            "Convergence": {"Convergence R-hat": 1.05, "ESS per Chain": 1500, "MCSE": 0.002},
            "Out-of-Sample": {"R-squared": 0.85, "RMSE": 0.08, "MAE": 0.06},
        }

        result = generator.generate_statistical_validation(metrics, include_thresholds=True)

        assert "Statistical Validation Metrics" in result
        assert "Goodness of Fit" in result
        assert "✓ Pass" in result
        assert "Threshold" in result

    def test_generate_comprehensive_results(self, generator):
        """Test Table C1: Comprehensive Optimization Results."""
        results = [
            {
                "retention": 100_000,
                "primary_limit": 1_000_000,
                "excess_limit": 5_000_000,
                "premium": 150_000,
                "roe": 0.18,
                "ruin_probability": 0.008,
                "growth_rate": 0.12,
                "sharpe_ratio": 1.5,
            },
            {
                "retention": 200_000,
                "primary_limit": 2_000_000,
                "excess_limit": 10_000_000,
                "premium": 250_000,
                "roe": 0.16,
                "ruin_probability": 0.005,
                "growth_rate": 0.10,
                "sharpe_ratio": 1.3,
            },
            {
                "retention": 50_000,
                "primary_limit": 500_000,
                "excess_limit": 2_000_000,
                "premium": 80_000,
                "roe": 0.20,
                "ruin_probability": 0.015,
                "growth_rate": 0.14,
                "sharpe_ratio": 1.7,
            },
        ]

        result = generator.generate_comprehensive_results(results, ranking_metric="roe", top_n=2)

        assert "Comprehensive Optimization Results" in result
        assert "Ranked by roe" in result
        assert "Rank" in result
        assert "20.00%" in result  # Top ROE
        assert len(result.split("\n")) > 3  # Has content

    def test_generate_walk_forward_validation(self, generator):
        """Test Table C2: Walk-Forward Validation Results."""
        validation_results = [
            {
                "period": "Q1 2023",
                "in_sample_roe": 0.18,
                "out_sample_roe": 0.17,
                "tracking_error": 0.02,
                "strategy_change": "No",
                "stability_score": 0.95,
            },
            {
                "period": "Q2 2023",
                "in_sample_roe": 0.17,
                "out_sample_roe": 0.16,
                "tracking_error": 0.03,
                "strategy_change": "Yes",
                "stability_score": 0.88,
            },
            {
                "period": "Q3 2023",
                "in_sample_roe": 0.19,
                "out_sample_roe": 0.18,
                "tracking_error": 0.02,
                "strategy_change": "No",
                "stability_score": 0.92,
            },
        ]

        result = generator.generate_walk_forward_validation(validation_results)

        assert "Walk-Forward Validation Results" in result
        assert "Q1 2023" in result
        assert "Average" in result  # Summary row
        assert "18.00%" in result  # ROE percentage
        assert "1/3" in result  # Strategy changes count

    def test_export_functionality(self, generator, tmp_path):
        """Test export to various file formats."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Test CSV export
        csv_path = tmp_path / "test.csv"
        generator.export_to_file(df, str(csv_path), "csv")
        assert csv_path.exists()

        # Read back and verify
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == 3
        assert list(df_read.columns) == ["A", "B"]

        # Skip Excel export test if openpyxl not installed
        try:
            import openpyxl

            # Test Excel export
            excel_path = tmp_path / "test.xlsx"
            generator.export_to_file(df, str(excel_path), "excel")
            assert excel_path.exists()
        except ImportError:
            pass  # Skip Excel test if openpyxl not available

        # Test HTML export
        html_path = tmp_path / "test.html"
        generator.export_to_file(df, str(html_path), "html")
        assert html_path.exists()
        assert "<table" in html_path.read_text()


class TestHelperFunctions:
    """Test standalone helper functions."""

    def test_create_performance_table(self):
        """Test performance table creation."""
        results = {
            "roe": 0.18,
            "ruin_prob": 0.008,
            "growth_rate": 0.12,
            "sharpe": 1.5,
            "max_drawdown": 0.15,
        }

        table = create_performance_table(results)

        assert "Performance Metrics" in table
        assert "ROE" in table
        assert "0.18" in str(table)
        assert "✓" in table  # Good performance indicators

    def test_create_parameter_table(self):
        """Test parameter table creation."""
        params = {
            "Growth": {"mean": 0.08, "volatility": 0.20},
            "Insurance": {"retention": 100_000, "premium_rate": 0.015},
        }

        table = create_parameter_table(params)

        assert "Model Parameters" in table
        assert "Growth" in table
        assert "Insurance" in table
        assert "mean" in table

    def test_create_sensitivity_table(self):
        """Test sensitivity analysis table creation."""
        base_case = 0.15
        sensitivities = {"Premium Rate": [0.14, 0.15, 0.16], "Retention": [0.13, 0.15, 0.17]}
        parameter_ranges: Dict[str, List[float]] = {
            "Premium Rate": [0.01, 0.015, 0.02],
            "Retention": [50_000, 100_000, 150_000],
        }

        table = create_sensitivity_table(base_case, sensitivities, parameter_ranges)

        assert "Sensitivity Analysis" in table
        assert "Premium Rate" in table
        assert "Change from Base" in table


class TestFormatForExport:
    """Test the format_for_export function."""

    def test_csv_export(self):
        """Test CSV format export."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = format_for_export(df, "csv")

        assert result is not None
        assert "A,B" in result
        assert "1,3" in result
        assert "2,4" in result

    def test_latex_export(self):
        """Test LaTeX format export."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = format_for_export(df, "latex", caption="Test Table", label="tab:test")

        assert result is not None
        assert "\\begin{table}" in result
        assert "\\caption{Test Table}" in result
        assert "\\label{tab:test}" in result
        assert "\\end{table}" in result

    def test_html_export(self):
        """Test HTML format export."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = format_for_export(df, "html", table_id="test-table", classes="table table-striped")

        assert result is not None
        assert 'id="test-table"' in result  # Fixed to be more flexible about attribute order
        assert 'class="table table-striped"' in result
        assert "<td>1</td>" in result

    def test_markdown_export(self):
        """Test Markdown format export."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = format_for_export(df, "markdown")

        assert result is not None
        assert "|" in result
        assert "A" in result
        assert "B" in result


class TestIntegration:
    """Integration tests for the complete table generation system."""

    def test_full_executive_report_tables(self):
        """Test generating all executive report tables."""
        generator = TableGenerator()

        # Generate Table 1
        sizes: List[float] = [1_000_000, 10_000_000]
        limits: Dict[float, Dict[str, float]] = {
            1_000_000: {
                "retention": 50_000,
                "primary": 500_000,
                "excess": 1_000_000,
                "premium": 25_000,
            },
            10_000_000: {
                "retention": 250_000,
                "primary": 2_500_000,
                "excess": 5_000_000,
                "premium": 150_000,
            },
        }
        table1 = generator.generate_optimal_limits_by_size(sizes, limits)

        # Generate Table 2
        chars = ["Growth", "Stable"]
        recs = {
            "Growth": {
                "retention": "Low",
                "coverage": "High",
                "premium_budget": "2-3%",
                "risk_level": "good",
            },
            "Stable": {
                "retention": "Medium",
                "coverage": "Medium",
                "premium_budget": "1-2%",
                "risk_level": "warning",
            },
        }
        table2 = generator.generate_quick_reference_matrix(chars, recs)

        # Verify both tables generated
        assert len(table1) > 0
        assert len(table2) > 0
        assert "Optimal Insurance Limits" in table1
        assert "Quick Reference" in table2

    def test_full_technical_report_tables(self):
        """Test generating all technical report tables."""
        generator = TableGenerator()

        # Generate parameter grid
        params: Dict[str, Dict[str, Any]] = {
            "Model": {"timesteps": [100, 1000, 10000]},
            "Risk": {"volatility": [0.15, 0.20, 0.30]},
        }
        table_a1 = generator.generate_parameter_grid(params)

        # Generate loss distribution parameters
        loss_types = ["Small", "Large"]
        dist_params = {
            "Small": {
                "frequency": 5,
                "severity_mean": 10_000,
                "severity_std": 5_000,
                "distribution": "Lognormal",
            },
            "Large": {
                "frequency": 0.5,
                "severity_mean": 100_000,
                "severity_std": 50_000,
                "distribution": "Lognormal",
            },
        }
        table_a2 = generator.generate_loss_distribution_params(loss_types, dist_params)

        # Verify tables generated
        assert "Parameter Grid" in table_a1
        assert "Loss Distribution" in table_a2

    def test_table_with_formatting_and_export(self, tmp_path):
        """Test complete workflow: generate, format, and export table."""
        # Create generator with HTML output
        generator = TableGenerator(default_format="html")

        # Generate a table with formatting
        results = [
            {"retention": 100_000, "roe": 0.18, "ruin_probability": 0.008},
            {"retention": 200_000, "roe": 0.16, "ruin_probability": 0.005},
        ]

        table = generator.generate_comprehensive_results(results, ranking_metric="roe")

        # Verify HTML formatting
        assert "<table" in table

        # Export to file
        df = pd.DataFrame(results)
        export_path = tmp_path / "results.html"
        generator.export_to_file(df, str(export_path), "html")

        assert export_path.exists()
        content = export_path.read_text()
        assert "<table" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

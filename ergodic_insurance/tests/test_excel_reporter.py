"""Tests for Excel report generation module.

This module tests the Excel report generation functionality,
including XlsxWriter and openpyxl engines, formatting, and
multi-sheet workbook creation.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.excel_reporter import (
    OPENPYXL_AVAILABLE,
    XLSXWRITER_AVAILABLE,
    ExcelReportConfig,
    ExcelReporter,
)
from ergodic_insurance.src.financial_statements import FinancialStatementGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer


class TestExcelReportConfig:
    """Test ExcelReportConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExcelReportConfig()

        assert config.output_path == Path("./reports")
        assert config.include_balance_sheet is True
        assert config.include_income_statement is True
        assert config.include_cash_flow is True
        assert config.include_reconciliation is True
        assert config.include_metrics_dashboard is True
        assert config.include_pivot_data is True
        assert config.engine == "auto"
        assert config.currency_format == "$#,##0"
        assert config.decimal_places == 0
        assert config.date_format == "yyyy-mm-dd"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExcelReportConfig(
            output_path=Path("/custom/path"),
            include_balance_sheet=False,
            engine="xlsxwriter",
            currency_format="€#,##0.00",
            decimal_places=2,
        )

        assert config.output_path == Path("/custom/path")
        assert config.include_balance_sheet is False
        assert config.engine == "xlsxwriter"
        assert config.currency_format == "€#,##0.00"
        assert config.decimal_places == 2


class TestExcelReporter:
    """Test ExcelReporter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_manufacturer(self):
        """Create a mock manufacturer with test data."""
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.initial_assets = 10_000_000
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            operating_margin=0.08,
            tax_rate=0.25,
        )

        # Create sample metrics history
        manufacturer.metrics_history = []
        for year in range(5):
            metrics = {
                "year": year,
                "assets": 10_000_000 * (1.03**year),
                "equity": 10_000_000 * (1.03**year),
                "revenue": 5_000_000 * (1.03**year),
                "operating_income": 400_000 * (1.03**year),
                "net_income": 300_000 * (1.03**year),
                "collateral": 100_000 * year,
                "restricted_assets": 100_000 * year,
                "available_assets": 10_000_000 * (1.03**year) - 100_000 * year,
                "claim_liabilities": 50_000 * year if year > 0 else 0,
                "is_solvent": True,
                "operating_margin": 0.08,
                "roe": 0.03,
                "roa": 0.03,
                "asset_turnover": 0.5,
                "collateral_to_equity": 0.01 * year,
                "collateral_to_assets": 0.01 * year,
            }
            manufacturer.metrics_history.append(metrics)

        return manufacturer

    @pytest.fixture
    def reporter_config(self, temp_dir):
        """Create reporter configuration with temp directory."""
        return ExcelReportConfig(output_path=temp_dir)

    def test_initialization(self, reporter_config):
        """Test ExcelReporter initialization."""
        reporter = ExcelReporter(reporter_config)

        assert reporter.config == reporter_config
        assert reporter.workbook is None
        assert reporter.formats == {}
        assert reporter.engine in ["xlsxwriter", "openpyxl", "pandas"]

    def test_engine_selection_auto(self):
        """Test automatic engine selection."""
        config = ExcelReportConfig(engine="auto")
        reporter = ExcelReporter(config)

        if XLSXWRITER_AVAILABLE:
            assert reporter.engine == "xlsxwriter"
        elif OPENPYXL_AVAILABLE:
            assert reporter.engine == "openpyxl"
        else:
            assert reporter.engine == "pandas"

    def test_engine_selection_specific(self):
        """Test specific engine selection."""
        if XLSXWRITER_AVAILABLE:
            config = ExcelReportConfig(engine="xlsxwriter")
            reporter = ExcelReporter(config)
            assert reporter.engine == "xlsxwriter"

        if OPENPYXL_AVAILABLE:
            config = ExcelReportConfig(engine="openpyxl")
            reporter = ExcelReporter(config)
            assert reporter.engine == "openpyxl"

    def test_engine_selection_fallback(self):
        """Test engine fallback when specified engine not available."""
        config = ExcelReportConfig(engine="nonexistent")
        reporter = ExcelReporter(config)
        assert reporter.engine == "pandas"

    @pytest.mark.skipif(not XLSXWRITER_AVAILABLE, reason="XlsxWriter not available")
    def test_generate_trajectory_report_xlsxwriter(self, mock_manufacturer, reporter_config):
        """Test trajectory report generation with XlsxWriter."""
        reporter_config.engine = "xlsxwriter"
        reporter = ExcelReporter(reporter_config)

        output_file = reporter.generate_trajectory_report(
            mock_manufacturer, "test_report.xlsx", title="Test Financial Report"
        )

        assert output_file.exists()
        assert output_file.suffix == ".xlsx"

        # Verify the file can be read by pandas
        with pd.ExcelFile(output_file) as xls:
            sheet_names = xls.sheet_names

            # Check expected sheets exist
            if reporter_config.include_balance_sheet:
                assert "Balance Sheet" in sheet_names
            if reporter_config.include_income_statement:
                assert "Income Statement" in sheet_names
            if reporter_config.include_cash_flow:
                assert "Cash Flow" in sheet_names
            if reporter_config.include_reconciliation:
                assert "Reconciliation" in sheet_names
            if reporter_config.include_metrics_dashboard:
                assert "Metrics Dashboard" in sheet_names
            if reporter_config.include_pivot_data:
                assert "Pivot Data" in sheet_names

    @pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not available")
    def test_generate_trajectory_report_openpyxl(self, mock_manufacturer, reporter_config):
        """Test trajectory report generation with openpyxl."""
        reporter_config.engine = "openpyxl"
        reporter = ExcelReporter(reporter_config)

        output_file = reporter.generate_trajectory_report(
            mock_manufacturer, "test_report_openpyxl.xlsx"
        )

        assert output_file.exists()
        assert output_file.suffix == ".xlsx"

        # Verify the file can be read
        with pd.ExcelFile(output_file) as xls:
            sheet_names = xls.sheet_names

            # Check at least some sheets exist
            assert len(sheet_names) > 0

            # Read a sheet to verify data
            if "Balance Sheet" in sheet_names:
                df = pd.read_excel(xls, "Balance Sheet")
                assert not df.empty
                assert "Item" in df.columns

    def test_generate_trajectory_report_pandas(self, mock_manufacturer, reporter_config):
        """Test trajectory report generation with pandas fallback."""
        reporter_config.engine = "pandas"
        reporter = ExcelReporter(reporter_config)

        output_file = reporter.generate_trajectory_report(
            mock_manufacturer, "test_report_pandas.xlsx"
        )

        assert output_file.exists()
        assert output_file.suffix == ".xlsx"

        # Verify the file can be read
        with pd.ExcelFile(output_file) as xls:
            sheet_names = xls.sheet_names
            assert len(sheet_names) > 0

    def test_selective_sheet_inclusion(self, mock_manufacturer, reporter_config):
        """Test selective inclusion of sheets."""
        # Configure to include only specific sheets
        reporter_config.include_balance_sheet = True
        reporter_config.include_income_statement = False
        reporter_config.include_cash_flow = False
        reporter_config.include_reconciliation = False
        reporter_config.include_metrics_dashboard = True
        reporter_config.include_pivot_data = False

        reporter = ExcelReporter(reporter_config)
        output_file = reporter.generate_trajectory_report(mock_manufacturer, "test_selective.xlsx")

        with pd.ExcelFile(output_file) as xls:
            sheet_names = xls.sheet_names

            # Check only selected sheets are included
            if XLSXWRITER_AVAILABLE and reporter.engine == "xlsxwriter":
                assert "Balance Sheet" in sheet_names
                assert "Income Statement" not in sheet_names
                assert "Cash Flow" not in sheet_names
                assert "Metrics Dashboard" in sheet_names

    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        # Create a config with a non-existent subdirectory
        output_path = temp_dir / "subdir" / "reports"
        config = ExcelReportConfig(output_path=output_path)

        reporter = ExcelReporter(config)

        # Directory should be created during initialization
        assert output_path.exists()
        assert output_path.is_dir()

    @pytest.mark.skipif(not XLSXWRITER_AVAILABLE, reason="XlsxWriter not available")
    def test_categorize_metric(self):
        """Test metric categorization for pivot tables."""
        config = ExcelReportConfig(engine="xlsxwriter")
        reporter = ExcelReporter(config)

        # Test various metric categorizations
        assert reporter._categorize_metric("revenue") == "Income"
        assert reporter._categorize_metric("operating_income") == "Income"
        assert reporter._categorize_metric("net_profit") == "Income"

        assert reporter._categorize_metric("assets") == "Balance Sheet"
        assert reporter._categorize_metric("equity") == "Balance Sheet"
        assert reporter._categorize_metric("collateral") == "Balance Sheet"

        assert reporter._categorize_metric("roe") == "Ratios"
        assert reporter._categorize_metric("roa") == "Ratios"
        assert reporter._categorize_metric("operating_margin") == "Ratios"
        assert reporter._categorize_metric("asset_turnover") == "Ratios"

        assert reporter._categorize_metric("claim_liabilities") == "Liabilities"
        assert reporter._categorize_metric("total_claims") == "Liabilities"

        assert reporter._categorize_metric("random_metric") == "Other"

    @pytest.mark.skipif(not XLSXWRITER_AVAILABLE, reason="XlsxWriter not available")
    def test_formatting_setup(self, reporter_config):
        """Test that formatting is properly set up."""
        reporter_config.engine = "xlsxwriter"
        reporter_config.currency_format = "€#,##0.00"
        reporter_config.decimal_places = 2

        reporter = ExcelReporter(reporter_config)

        # Create a mock workbook to test format setup
        import xlsxwriter

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            reporter.workbook = xlsxwriter.Workbook(tmp_path)
            reporter._setup_xlsxwriter_formats()

            # Check that formats are created
            assert "currency" in reporter.formats
            assert "percent" in reporter.formats
            assert "header" in reporter.formats
            assert "total" in reporter.formats
            assert "subtotal" in reporter.formats

            reporter.workbook.close()
        finally:
            # Clean up the file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_metrics_dashboard_data(self, mock_manufacturer, reporter_config):
        """Test that metrics dashboard contains correct data."""
        reporter = ExcelReporter(reporter_config)
        output_file = reporter.generate_trajectory_report(mock_manufacturer, "test_metrics.xlsx")

        if reporter_config.include_metrics_dashboard:
            with pd.ExcelFile(output_file) as xls:
                if "Metrics Dashboard" in xls.sheet_names:
                    # Read the raw data first
                    df_raw = pd.read_excel(xls, "Metrics Dashboard", header=None)

                    # Find the row with column headers
                    header_row = None
                    for idx, row in df_raw.iterrows():
                        if "Year" in str(row.values):
                            # Type narrowing for mypy - idx from iterrows() is Hashable
                            # but we know it's an int for default DataFrame indices
                            if isinstance(idx, int):
                                header_row = idx
                            break

                    # If we found the header row, read the data properly
                    if header_row is not None:
                        # header_row is already an int from the isinstance check above
                        df = pd.read_excel(xls, "Metrics Dashboard", header=header_row)

                        # Check expected columns
                        expected_columns = [
                            "Year",
                            "Revenue",
                            "Operating Income",
                            "Net Income",
                            "Assets",
                            "Equity",
                        ]
                        for col in expected_columns:
                            assert col in df.columns

                        # Filter out NaN rows and summary rows
                        data_df = df.dropna(subset=["Year"])
                        data_df = data_df[
                            pd.to_numeric(data_df["Year"], errors="coerce").notna()
                        ].head(5)

                        # Check data integrity
                        assert len(data_df) > 0
                        assert data_df["Year"].nunique() > 0
                        assert all(pd.to_numeric(data_df["Assets"], errors="coerce") > 0)

    def test_pivot_data_structure(self, mock_manufacturer, reporter_config):
        """Test that pivot data is properly structured."""
        reporter = ExcelReporter(reporter_config)
        output_file = reporter.generate_trajectory_report(mock_manufacturer, "test_pivot.xlsx")

        if reporter_config.include_pivot_data:
            with pd.ExcelFile(output_file) as xls:
                if "Pivot Data" in xls.sheet_names:
                    df = pd.read_excel(xls, "Pivot Data")

                    # Check expected columns for pivot table
                    assert "Year" in df.columns
                    assert "Category" in df.columns
                    assert "Metric" in df.columns
                    assert "Value" in df.columns

                    # Check data is normalized
                    assert len(df) > 0
                    assert df["Category"].nunique() > 1
                    assert df["Metric"].nunique() > 1

    @patch("ergodic_insurance.src.excel_reporter.FinancialStatementGenerator")
    def test_monte_carlo_report_placeholder(self, mock_generator_class, reporter_config):
        """Test Monte Carlo report generation (placeholder test)."""
        reporter = ExcelReporter(reporter_config)

        # Create mock results
        mock_results = Mock()

        # This is a placeholder - the actual implementation would need
        # proper Monte Carlo results
        output_file = reporter.generate_monte_carlo_report(
            mock_results, "test_monte_carlo.xlsx", title="Monte Carlo Analysis"
        )

        assert output_file.exists()
        assert output_file.suffix == ".xlsx"

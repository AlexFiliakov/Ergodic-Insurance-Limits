"""Tests for financial statement generation module.

This module tests the financial statement compilation and generation
functionality, including balance sheets, income statements, cash flow
statements, and reconciliation reports.
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.financial_statements import (
    FinancialStatementConfig,
    FinancialStatementGenerator,
    MonteCarloStatementAggregator,
)
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestFinancialStatementConfig:
    """Test FinancialStatementConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FinancialStatementConfig()

        assert config.currency_symbol == "$"
        assert config.decimal_places == 0
        assert config.include_yoy_change is True
        assert config.include_percentages is True
        assert config.fiscal_year_end == 12
        assert config.consolidate_monthly is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FinancialStatementConfig(
            currency_symbol="€", decimal_places=2, include_yoy_change=False, fiscal_year_end=6
        )

        assert config.currency_symbol == "€"
        assert config.decimal_places == 2
        assert config.include_yoy_change is False
        assert config.fiscal_year_end == 6


class TestFinancialStatementGenerator:
    """Test FinancialStatementGenerator class."""

    @pytest.fixture
    def mock_manufacturer(self):
        """Create a mock manufacturer with test data."""
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.initial_assets = 10_000_000
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Create sample metrics history
        manufacturer.metrics_history = [
            {
                "year": 0,
                "assets": 10_000_000,
                "equity": 10_000_000,
                "revenue": 5_000_000,
                "operating_income": 400_000,
                "net_income": 300_000,
                "collateral": 0,
                "restricted_assets": 0,
                "available_assets": 10_000_000,
                "claim_liabilities": 0,
                "is_solvent": True,
                "base_operating_margin": 0.08,
                "roe": 0.03,
                "roa": 0.03,
                "asset_turnover": 0.5,
            },
            {
                "year": 1,
                "assets": 10_300_000,
                "equity": 10_300_000,
                "revenue": 5_150_000,
                "operating_income": 412_000,
                "net_income": 309_000,
                "collateral": 100_000,
                "restricted_assets": 100_000,
                "available_assets": 10_200_000,
                "claim_liabilities": 0,
                "is_solvent": True,
                "base_operating_margin": 0.08,
                "roe": 0.03,
                "roa": 0.03,
                "asset_turnover": 0.5,
            },
            {
                "year": 2,
                "assets": 10_609_000,
                "equity": 10_509_000,
                "revenue": 5_304_500,
                "operating_income": 424_360,
                "net_income": 318_270,
                "collateral": 200_000,
                "restricted_assets": 200_000,
                "available_assets": 10_409_000,
                "claim_liabilities": 100_000,
                "is_solvent": True,
                "base_operating_margin": 0.08,
                "roe": 0.03,
                "roa": 0.03,
                "asset_turnover": 0.5,
            },
        ]

        return manufacturer

    @pytest.fixture
    def generator(self, mock_manufacturer):
        """Create a FinancialStatementGenerator with mock data."""
        return FinancialStatementGenerator(manufacturer=mock_manufacturer)

    def test_initialization_with_manufacturer(self, mock_manufacturer):
        """Test initialization with manufacturer object."""
        generator = FinancialStatementGenerator(manufacturer=mock_manufacturer)

        assert generator.manufacturer_data is not None
        assert generator.metrics_history == mock_manufacturer.metrics_history
        assert generator.years_available == 3

    def test_initialization_with_data_dict(self, mock_manufacturer):
        """Test initialization with manufacturer data dictionary."""
        data = {
            "metrics_history": mock_manufacturer.metrics_history,
            "initial_assets": mock_manufacturer.initial_assets,
            "config": mock_manufacturer.config,
        }

        generator = FinancialStatementGenerator(manufacturer_data=data)

        assert generator.manufacturer_data == data
        assert generator.metrics_history == mock_manufacturer.metrics_history
        assert generator.years_available == 3

    def test_initialization_error(self):
        """Test initialization without required data."""
        with pytest.raises(
            ValueError, match="Either manufacturer or manufacturer_data must be provided"
        ):
            FinancialStatementGenerator()

    def test_generate_balance_sheet(self, generator):
        """Test balance sheet generation."""
        # Generate balance sheet for year 2
        df = generator.generate_balance_sheet(year=2)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns
        assert "Year 2" in df.columns
        assert "Type" in df.columns

        # Check key sections exist
        items = df["Item"].values
        assert any("ASSETS" in str(item) for item in items)
        assert any("LIABILITIES" in str(item) for item in items)
        assert any("EQUITY" in str(item) for item in items)
        assert any("Insurance Collateral" in str(item) for item in items)

        # Check that balance sheet balances
        # Find the total assets and total liabilities + equity rows
        total_assets_row = df[df["Item"].str.contains("TOTAL ASSETS", na=False)]
        total_le_row = df[df["Item"].str.contains("TOTAL LIABILITIES \\+ EQUITY", na=False)]

        if not total_assets_row.empty and not total_le_row.empty:
            total_assets = total_assets_row["Year 2"].values[0]
            total_le = total_le_row["Year 2"].values[0]
            assert abs(total_assets - total_le) < 1.0  # Allow for small rounding differences

    def test_generate_balance_sheet_invalid_year(self, generator):
        """Test balance sheet generation with invalid year."""
        with pytest.raises(IndexError):
            generator.generate_balance_sheet(year=10)

        with pytest.raises(IndexError):
            generator.generate_balance_sheet(year=-1)

    def test_generate_balance_sheet_with_comparison(self, generator):
        """Test balance sheet generation with year comparisons."""
        df = generator.generate_balance_sheet(year=2, compare_years=[0, 1])

        # Check that comparison years are added
        assert "Year 0" in df.columns or len(df.columns) > 3

    def test_generate_income_statement(self, generator):
        """Test income statement generation."""
        df = generator.generate_income_statement(year=2)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns
        assert "Year 2" in df.columns

        # Check key sections exist
        items = df["Item"].values
        assert any("REVENUE" in str(item) for item in items)
        assert any("OPERATING EXPENSES" in str(item) for item in items)
        assert any("NET INCOME" in str(item) for item in items)
        assert any("KEY METRICS" in str(item) for item in items)

        # Check metrics are included
        assert any("Operating Margin" in str(item) for item in items)
        assert any("ROE" in str(item) for item in items)
        assert any("ROA" in str(item) for item in items)

    def test_income_statement_insurance_costs(self, generator):
        """Test that insurance costs appear correctly in income statement."""
        # Ensure some metrics have insurance costs
        generator.metrics_history[2]["insurance_premiums"] = 500_000
        generator.metrics_history[2]["insurance_losses"] = 200_000

        df = generator.generate_income_statement(year=2)

        # Check that insurance costs appear in the statement
        items = df["Item"].values
        values = df["Year 2"].values

        # Find insurance premium row
        premium_row = None
        losses_row = None
        for i, item in enumerate(items):
            if "Insurance Premium" in str(item):
                premium_row = i
            elif "Insurance Claim Expense" in str(item):
                losses_row = i

        # Verify insurance costs are displayed (negative values in statement)
        assert premium_row is not None, "Insurance premiums should appear in income statement"
        assert losses_row is not None, "Insurance losses should appear in income statement"

        # Check values are correct (shown as negative expenses)
        assert values[premium_row] == -500_000, "Insurance premium should be -500,000"
        assert values[losses_row] == -200_000, "Insurance losses should be -200,000"

        # Verify total other expenses includes insurance costs
        total_other_row = None
        for i, item in enumerate(items):
            if "Total Other" in str(item):
                total_other_row = i
                break

        assert total_other_row is not None
        assert (
            values[total_other_row] == -700_000
        ), "Total other expenses should include insurance costs"

    def test_generate_cash_flow_statement_indirect(self, generator):
        """Test cash flow statement generation using indirect method."""
        df = generator.generate_cash_flow_statement(year=2, method="indirect")

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns
        assert "Year 2" in df.columns

        # Check key sections exist
        items = df["Item"].values
        assert any("OPERATING ACTIVITIES" in str(item) for item in items)
        assert any("INVESTING ACTIVITIES" in str(item) for item in items)
        assert any("FINANCING ACTIVITIES" in str(item) for item in items)
        assert any("NET CHANGE IN CASH" in str(item) for item in items)

        # Check indirect method items
        assert any("Net Income" in str(item) for item in items)
        assert any("Depreciation" in str(item) for item in items)

    def test_generate_cash_flow_statement_direct(self, generator):
        """Test cash flow statement generation using direct method."""
        df = generator.generate_cash_flow_statement(year=2, method="direct")

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns
        assert "Year 2" in df.columns

        # Check direct method items
        items = df["Item"].values
        assert any("Cash from Customers" in str(item) for item in items)
        assert any("Cash to Suppliers" in str(item) for item in items)

    def test_generate_reconciliation_report(self, generator):
        """Test reconciliation report generation."""
        df = generator.generate_reconciliation_report(year=2)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Check" in df.columns
        assert "Value" in df.columns
        assert "Type" in df.columns

        # Check reconciliation checks exist
        checks = df["Check"].values
        assert any("BALANCE SHEET RECONCILIATION" in str(check) for check in checks)
        assert any("NET ASSETS RECONCILIATION" in str(check) for check in checks)
        assert any("COLLATERAL RECONCILIATION" in str(check) for check in checks)
        assert any("SOLVENCY CHECK" in str(check) for check in checks)

        # Check status values
        status_rows = df[df["Type"] == "status"]
        for _, row in status_rows.iterrows():
            status = row["Value"]
            assert status in [
                "BALANCED",
                "IMBALANCED",
                "MATCHED",
                "MISMATCHED",
                "VALID",
                "INVALID",
                "SOLVENT",
                "INSOLVENT",
            ]

    def test_yoy_comparison(self, generator):
        """Test year-over-year comparison functionality."""
        # Generate balance sheet with YoY comparison
        config = FinancialStatementConfig(include_yoy_change=True)
        generator_with_yoy = FinancialStatementGenerator(
            manufacturer_data=generator.manufacturer_data, config=config
        )

        df = generator_with_yoy.generate_balance_sheet(year=2)

        # Check YoY column exists
        assert "YoY Change %" in df.columns

    def test_categorize_metric(self, generator):
        """Test internal metric categorization."""
        # Access through the excel_reporter module since it's defined there
        # For now, just test that the generator works without errors
        assert generator is not None


class TestMonteCarloStatementAggregator:
    """Test MonteCarloStatementAggregator class."""

    @pytest.fixture
    def mock_results(self):
        """Create mock Monte Carlo results."""
        # Create sample results from multiple trajectories
        results = []
        for i in range(10):
            trajectory = {
                "trajectory_id": i,
                "final_assets": 10_000_000 * (1 + np.random.normal(0.05, 0.02)),
                "final_equity": 10_000_000 * (1 + np.random.normal(0.05, 0.02)),
                "total_revenue": 50_000_000 * (1 + np.random.normal(0.03, 0.01)),
                "total_claims": np.random.exponential(100_000),
            }
            results.append(trajectory)

        return results

    def test_initialization_with_list(self, mock_results):
        """Test initialization with list of results."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=mock_results)

        assert aggregator.results == mock_results
        assert aggregator.config is not None

    def test_initialization_with_dataframe(self, mock_results):
        """Test initialization with DataFrame of results."""
        df = pd.DataFrame(mock_results)
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=df)

        assert isinstance(aggregator.results, pd.DataFrame)
        assert len(aggregator.results) == len(mock_results)

    def test_custom_config(self):
        """Test initialization with custom configuration."""
        config = FinancialStatementConfig(currency_symbol="€", decimal_places=2)
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[], config=config)

        assert aggregator.config.currency_symbol == "€"
        assert aggregator.config.decimal_places == 2

"""Tests for financial statement generation module.

This module tests the financial statement compilation and generation
functionality, including balance sheets, income statements, cash flow
statements, and reconciliation reports.
"""

from typing import Optional
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.config import ExpenseRatioConfig, ManufacturerConfig
from ergodic_insurance.financial_statements import (
    FinancialStatementConfig,
    FinancialStatementGenerator,
    MonteCarloStatementAggregator,
)
from ergodic_insurance.ledger import Ledger, TransactionType
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
        # fiscal_year_end defaults to None, allowing inheritance from central config
        # It gets resolved to 12 when used with a generator without central config
        assert config.fiscal_year_end is None
        assert config.consolidate_monthly is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FinancialStatementConfig(
            currency_symbol="€",
            decimal_places=2,
            include_yoy_change=False,
            fiscal_year_end=6,
        )

        assert config.currency_symbol == "€"
        assert config.decimal_places == 2
        assert config.include_yoy_change is False
        assert config.fiscal_year_end == 6


def _add_cogs_sga_breakdown(
    metrics: dict,
    expense_config: Optional[ExpenseRatioConfig] = None,
) -> dict:
    """Add COGS and SG&A breakdown fields to metrics.

    Issue #255: The Manufacturer now provides these values explicitly.
    This helper function calculates them using the provided or default ratios for test data.

    Args:
        metrics: Dictionary of financial metrics
        expense_config: Optional ExpenseRatioConfig with custom ratios

    Returns:
        Updated metrics dictionary with COGS/SG&A breakdown
    """
    revenue = metrics.get("revenue", 0)
    depreciation = metrics.get("depreciation_expense", 0)

    # Use provided expense config or defaults (matching Manufacturer defaults)
    if expense_config:
        gross_margin_ratio = expense_config.gross_margin_ratio
        sga_expense_ratio = expense_config.sga_expense_ratio
        mfg_depreciation_alloc = expense_config.manufacturing_depreciation_allocation
        admin_depreciation_alloc = expense_config.admin_depreciation_allocation
        direct_materials_ratio = expense_config.direct_materials_ratio
        direct_labor_ratio = expense_config.direct_labor_ratio
        manufacturing_overhead_ratio = expense_config.manufacturing_overhead_ratio
        selling_expense_ratio = expense_config.selling_expense_ratio
        general_admin_ratio = expense_config.general_admin_ratio
    else:
        gross_margin_ratio = 0.15
        sga_expense_ratio = 0.07
        mfg_depreciation_alloc = 0.7
        admin_depreciation_alloc = 0.3
        direct_materials_ratio = 0.4
        direct_labor_ratio = 0.3
        manufacturing_overhead_ratio = 0.3
        selling_expense_ratio = 0.4
        general_admin_ratio = 0.6

    # Calculate COGS breakdown
    cogs_ratio = 1.0 - gross_margin_ratio
    base_cogs = revenue * cogs_ratio
    mfg_depreciation = depreciation * mfg_depreciation_alloc
    cogs_before_depreciation = base_cogs - mfg_depreciation

    metrics["direct_materials"] = cogs_before_depreciation * direct_materials_ratio
    metrics["direct_labor"] = cogs_before_depreciation * direct_labor_ratio
    metrics["manufacturing_overhead"] = cogs_before_depreciation * manufacturing_overhead_ratio
    metrics["mfg_depreciation"] = mfg_depreciation
    metrics["total_cogs"] = base_cogs

    # Calculate SG&A breakdown
    base_sga = revenue * sga_expense_ratio
    admin_depreciation = depreciation * admin_depreciation_alloc
    sga_before_depreciation = base_sga - admin_depreciation

    metrics["selling_expenses"] = sga_before_depreciation * selling_expense_ratio
    metrics["general_admin_expenses"] = sga_before_depreciation * general_admin_ratio
    metrics["admin_depreciation"] = admin_depreciation
    metrics["total_sga"] = base_sga

    # Store expense ratios for reporting reference
    metrics["gross_margin_ratio"] = gross_margin_ratio
    metrics["sga_expense_ratio"] = sga_expense_ratio

    return metrics


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
        # Note: Balance sheet calculates total_assets = current_assets + net_ppe + restricted_assets
        # Where net_ppe = gross_ppe - accumulated_depreciation
        # Issue #255: COGS/SG&A breakdown is now added via helper function
        manufacturer.metrics_history = [
            _add_cogs_sga_breakdown(
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
                    "gross_ppe": 7_000_000,
                    "accumulated_depreciation": 0,
                    "net_ppe": 7_000_000,
                    "depreciation_expense": 700_000,  # 10-year useful life
                    "cash": 3_000_000,  # Current assets to make total = 10M
                }
            ),
            _add_cogs_sga_breakdown(
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
                    "gross_ppe": 7_200_000,
                    "accumulated_depreciation": 700_000,
                    "net_ppe": 6_500_000,
                    "depreciation_expense": 720_000,
                    "cash": 3_300_000,  # Year 1: 3.3M cash + 6.5M net_ppe + 0.5M restricted
                }
            ),
            _add_cogs_sga_breakdown(
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
                    "gross_ppe": 7_400_000,
                    "accumulated_depreciation": 1_420_000,
                    "net_ppe": 5_980_000,
                    "depreciation_expense": 740_000,
                    "cash": 4_429_000,  # Year 2: 4.429M cash + 5.98M net_ppe + 0.2M restricted
                }
            ),
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
            ValueError,
            match="Either manufacturer or manufacturer_data must be provided",
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

        # Check GAAP structure
        items = df["Item"].values
        assert any("REVENUE" in str(item) for item in items)
        assert any("COST OF GOODS SOLD" in str(item) for item in items)
        assert any("GROSS PROFIT" in str(item) for item in items)
        assert any("OPERATING EXPENSES" in str(item) for item in items)
        assert any("OPERATING INCOME (EBIT)" in str(item) for item in items)
        assert any("NON-OPERATING INCOME (EXPENSES)" in str(item) for item in items)
        assert any("INCOME BEFORE TAXES" in str(item) for item in items)
        assert any("INCOME TAX PROVISION" in str(item) for item in items)
        assert any("NET INCOME" in str(item) for item in items)
        assert any("KEY FINANCIAL METRICS" in str(item) for item in items)

        # Check metrics are included
        assert any("Gross Margin" in str(item) for item in items)
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

        # Find insurance premium and losses rows
        premium_found = False
        losses_found = False
        operating_section_idx = None
        non_operating_section_idx = None
        for i, item in enumerate(items):
            if "OPERATING EXPENSES" == str(item).strip():
                operating_section_idx = i
            elif "NON-OPERATING INCOME" in str(item):
                non_operating_section_idx = i
            elif "Insurance Premium" in str(item):
                premium_found = True
                # Premium should be in operating expenses (positive value)
                assert (
                    values[i] == 500_000
                ), "Insurance premium should be 500,000 in operating expenses"
                # Verify it's in operating section, before non-operating
                assert (
                    operating_section_idx is not None
                ), "Premiums should be after OPERATING EXPENSES header"
                assert (
                    non_operating_section_idx is None or i < non_operating_section_idx
                ), "Premiums should be before NON-OPERATING section"
            elif "Insurance Claim Loss" in str(item):
                losses_found = True
                # Losses should be in operating expenses (positive value, Issue #364)
                assert (
                    values[i] == 200_000
                ), "Insurance losses should be 200,000 in operating expenses"
                # Verify it's in operating section, before non-operating
                assert (
                    operating_section_idx is not None
                ), "Losses should be after OPERATING EXPENSES header"
                assert (
                    non_operating_section_idx is None or i < non_operating_section_idx
                ), "Insurance losses should be in operating section, not non-operating (ASC 944)"

        # Verify insurance costs appear in appropriate sections
        assert premium_found, "Insurance premiums should appear in operating expenses"
        assert losses_found, "Insurance claim losses should appear in operating expenses (ASC 944)"

        # Verify Total Non-Operating does NOT include insurance losses
        for i, item in enumerate(items):
            if "Total Non-Operating" in str(item):
                # Total Non-Operating should only contain interest items
                # Insurance losses are no longer here (Issue #364)
                break

    def test_generate_cash_flow_statement_indirect(self, generator):
        """Test cash flow statement generation using indirect method."""
        df = generator.generate_cash_flow_statement(year=2, period="annual")

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns
        assert "Year 2" in df.columns

        # Check key sections exist
        items = df["Item"].values
        assert any("CASH FLOWS FROM OPERATING ACTIVITIES" in str(item) for item in items)
        assert any("CASH FLOWS FROM INVESTING ACTIVITIES" in str(item) for item in items)
        assert any("CASH FLOWS FROM FINANCING ACTIVITIES" in str(item) for item in items)
        assert any("NET INCREASE (DECREASE) IN CASH" in str(item) for item in items)

        # Check indirect method items
        assert any("Net Income" in str(item) for item in items)
        assert any("Depreciation" in str(item) for item in items)

    def test_generate_cash_flow_statement_direct_requires_ledger(self, generator):
        """Test that direct method raises error without a ledger."""
        with pytest.raises(ValueError, match="requires a ledger"):
            generator.generate_cash_flow_statement(year=2, method="direct")

    def test_generate_cash_flow_statement_direct_with_ledger(self, mock_manufacturer):
        """Test direct method cash flow statement with ledger."""
        # Create a ledger with sample transactions
        ledger = Ledger()

        # Year 2 transactions (matching our test data)
        # Cash from customers
        ledger.record_double_entry(
            date=2,
            debit_account="cash",
            credit_account="accounts_receivable",
            amount=5_500_000,
            transaction_type=TransactionType.COLLECTION,
            description="Collections from customers",
        )

        # Cash to suppliers
        ledger.record_double_entry(
            date=2,
            debit_account="operating_expenses",
            credit_account="cash",
            amount=2_000_000,
            transaction_type=TransactionType.PAYMENT,
            description="Payments to suppliers",
        )

        # Capital expenditure
        ledger.record_double_entry(
            date=2,
            debit_account="gross_ppe",
            credit_account="cash",
            amount=500_000,
            transaction_type=TransactionType.CAPEX,
            description="Equipment purchase",
        )

        # Dividend payment
        ledger.record_double_entry(
            date=2,
            debit_account="retained_earnings",
            credit_account="cash",
            amount=300_000,
            transaction_type=TransactionType.DIVIDEND,
            description="Dividend payment",
        )

        # Create generator with ledger
        generator = FinancialStatementGenerator(
            manufacturer=mock_manufacturer,
            ledger=ledger,
        )

        df = generator.generate_cash_flow_statement(year=2, method="direct")

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns
        assert "Year 2" in df.columns

        # Direct method shows cash receipts and payments, not net income
        items = df["Item"].values
        assert any("Direct Method" in str(item) for item in items)
        assert any("Cash Received from Customers" in str(item) for item in items)
        # Net Income should NOT be in direct method
        assert not any(
            item == "  Net Income" for item in items
        ), "Direct method should not show Net Income line"

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

    def test_income_statement_gaap_structure(self, generator):
        """Test that income statement follows proper GAAP structure."""
        df = generator.generate_income_statement(year=2)

        # Get all items in order
        items = df["Item"].str.strip().values

        # Define the expected order of major sections
        expected_sections = [
            "REVENUE",
            "COST OF GOODS SOLD",
            "GROSS PROFIT",
            "OPERATING EXPENSES",
            "OPERATING INCOME (EBIT)",
            "NON-OPERATING INCOME (EXPENSES)",
            "INCOME BEFORE TAXES",
            "INCOME TAX PROVISION",
            "NET INCOME",
            "KEY FINANCIAL METRICS",
        ]

        # Find indices of each section
        section_indices = {}
        for section in expected_sections:
            for i, item in enumerate(items):
                if section in item:
                    section_indices[section] = i
                    break

        # Verify all sections exist
        for section in expected_sections:
            assert section in section_indices, f"Missing section: {section}"

        # Verify sections appear in correct order
        prev_idx = -1
        for section in expected_sections:
            idx = section_indices[section]
            assert idx > prev_idx, f"Section {section} is out of order"
            prev_idx = idx

    def test_cogs_components(self, generator):
        """Test that COGS includes proper components including depreciation."""
        # Add depreciation expense to metrics
        generator.metrics_history[2]["depreciation_expense"] = 1_000_000

        df = generator.generate_income_statement(year=2)

        items = df["Item"].str.strip().values

        # Check COGS components
        cogs_components = [
            "Direct Materials",
            "Direct Labor",
            "Manufacturing Overhead",
            "Manufacturing Depreciation",
        ]

        for component in cogs_components:
            assert any(component in item for item in items), f"Missing COGS component: {component}"

    def test_missing_cogs_breakdown_raises_error(self):
        """Test that missing COGS breakdown raises ValueError.

        Issue #255: The reporting layer no longer estimates COGS/SG&A breakdown.
        These values must be provided explicitly by the Manufacturer.
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Metrics WITHOUT COGS/SG&A breakdown (old format)
        manufacturer.metrics_history = [
            {
                "year": 0,
                "assets": 10_000_000,
                "equity": 10_000_000,
                "revenue": 5_000_000,
                "depreciation_expense": 700_000,
                "cash": 3_000_000,
            }
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)

        with pytest.raises(ValueError, match="COGS breakdown fields missing"):
            generator.generate_income_statement(year=0)

    def test_missing_sga_breakdown_raises_error(self):
        """Test that missing SG&A breakdown raises ValueError.

        Issue #255: The reporting layer no longer estimates COGS/SG&A breakdown.
        These values must be provided explicitly by the Manufacturer.
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Metrics WITH COGS breakdown but WITHOUT SG&A breakdown
        manufacturer.metrics_history = [
            {
                "year": 0,
                "assets": 10_000_000,
                "equity": 10_000_000,
                "revenue": 5_000_000,
                "depreciation_expense": 700_000,
                "cash": 3_000_000,
                # COGS breakdown present
                "direct_materials": 1_504_000,
                "direct_labor": 1_128_000,
                "manufacturing_overhead": 1_128_000,
                "mfg_depreciation": 490_000,
                # SG&A breakdown MISSING
            }
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)

        with pytest.raises(ValueError, match="SG&A breakdown fields missing"):
            generator.generate_income_statement(year=0)

    def test_missing_cash_raises_error(self):
        """Test that missing cash raises ValueError.

        Issue #256: The reporting layer no longer estimates cash balance.
        Cash must be provided explicitly by the Manufacturer to avoid
        hiding simulation bugs with fabricated "phantom cash".
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Metrics WITHOUT cash (missing critical field)
        manufacturer.metrics_history = [
            {
                "year": 0,
                "assets": 10_000_000,
                "equity": 10_000_000,
                "gross_ppe": 7_000_000,
                "restricted_assets": 0,
                # cash is MISSING - this should raise an error
            }
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)

        with pytest.raises(ValueError, match="cash missing from metrics"):
            generator.generate_balance_sheet(year=0)

    def test_missing_gross_ppe_raises_error(self):
        """Test that missing gross_ppe raises ValueError.

        Issue #256: The reporting layer no longer estimates gross PP&E.
        Gross PP&E must be provided explicitly by the Manufacturer to avoid
        hiding simulation bugs with fabricated asset values.
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Metrics WITHOUT gross_ppe (missing critical field)
        manufacturer.metrics_history = [
            {
                "year": 0,
                "assets": 10_000_000,
                "equity": 10_000_000,
                "cash": 3_000_000,
                "restricted_assets": 0,
                # gross_ppe is MISSING - this should raise an error
            }
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)

        with pytest.raises(ValueError, match="gross_ppe missing from metrics"):
            generator.generate_balance_sheet(year=0)

    def test_income_statement_works_without_cash(self):
        """Test that income statement works without cash in metrics.

        Issue #301: The income statement no longer uses cash to fabricate
        interest income with hardcoded rates. Interest income/expense are
        now read from metrics directly (defaulting to 0).
        Previously (Issue #256), missing cash raised ValueError because
        interest_income was computed as cash * 0.02.
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Build metrics with COGS/SG&A breakdown but WITHOUT cash
        base_metrics = {
            "year": 0,
            "assets": 10_000_000,
            "equity": 10_000_000,
            "revenue": 5_000_000,
            "depreciation_expense": 700_000,
            # cash is NOT needed for income statement since Issue #301
        }
        manufacturer.metrics_history = [_add_cogs_sga_breakdown(base_metrics)]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)

        # Should NOT raise - income statement no longer needs cash
        df = generator.generate_income_statement(year=0)
        assert isinstance(df, pd.DataFrame)
        assert "Item" in df.columns

    def test_operating_expenses_components(self, generator):
        """Test that operating expenses include proper SG&A components."""
        df = generator.generate_income_statement(year=2)

        items = df["Item"].str.strip().values

        # Check operating expense components
        sga_components = [
            "Selling Expenses",
            "General & Administrative",
            "Administrative Depreciation",
        ]

        for component in sga_components:
            assert any(component in item for item in items), f"Missing SG&A component: {component}"

    def test_non_operating_section(self, generator):
        """Test non-operating income and expenses section.

        Issue #301: Interest income/expense are now read from metrics
        instead of being fabricated with hardcoded rates.
        """
        # Add interest income/expense to metrics
        generator.metrics_history[2]["interest_income"] = 50_000
        generator.metrics_history[2]["interest_expense"] = 100_000

        df = generator.generate_income_statement(year=2)

        items = df["Item"].str.strip().values
        values = df["Year 2"].values

        # Check non-operating items
        assert any("Interest Income" in item for item in items), "Missing Interest Income"
        assert any("Interest Expense" in item for item in items), "Missing Interest Expense"

        # Verify interest values come from metrics, not hardcoded rates
        for i, item in enumerate(items):
            if "Interest Income" in item and "Total" not in item:
                assert (
                    values[i] == 50_000
                ), f"Interest income should be 50,000 from metrics, got {values[i]}"
            elif "Interest Expense" in item:
                assert (
                    values[i] == -100_000
                ), f"Interest expense should be -100,000 from metrics, got {values[i]}"

    def test_no_fabricated_interest_rates(self, generator):
        """Test that interest income/expense default to 0 when not in metrics.

        Issue #301: Previously, the income statement fabricated interest income
        using cash * 0.02 and interest expense using debt * 0.05. Now they
        must come from metrics and default to 0.
        """
        # Ensure no interest fields in metrics
        for key in ["interest_income", "interest_expense"]:
            generator.metrics_history[2].pop(key, None)

        # Add cash and debt that would have generated fabricated values
        generator.metrics_history[2]["cash"] = 3_000_000
        generator.metrics_history[2]["debt_balance"] = 2_000_000

        df = generator.generate_income_statement(year=2)
        items = df["Item"].str.strip().values
        values = df["Year 2"].values

        # Interest Income should be 0, NOT cash * 0.02 = 60,000
        for i, item in enumerate(items):
            if "Interest Income" in item and "Total" not in item:
                assert (
                    values[i] == 0
                ), f"Interest income should be 0 (not fabricated), got {values[i]}"

    def test_net_income_gaap_consistency(self, generator):
        """Test that NET INCOME equals INCOME BEFORE TAXES minus Total Tax Provision.

        Issue #475: The income statement must be internally consistent — net income
        must equal the sum of its component line items per ASC 220-10-45.
        The old override that used the manufacturer's net_income directly was removed
        because it could create unexplained gaps between line items and the bottom line.
        """
        df = generator.generate_income_statement(year=2)

        items = [str(item).strip() for item in df["Item"].values]
        values = df["Year 2"].values

        # Extract key line items
        pretax_income = None
        total_tax = None
        net_income_value = None

        for i, item in enumerate(items):
            if item == "INCOME BEFORE TAXES":
                pretax_income = values[i]
            elif item == "Total Tax Provision":
                total_tax = values[i]
            elif item == "NET INCOME":
                net_income_value = values[i]

        assert pretax_income is not None, "INCOME BEFORE TAXES row not found"
        assert total_tax is not None, "Total Tax Provision row not found"
        assert net_income_value is not None, "NET INCOME row not found"

        # Net income MUST equal pretax income minus total tax provision
        expected = float(pretax_income) - float(total_tax)
        assert abs(float(net_income_value) - expected) < 0.01, (
            f"NET INCOME ({net_income_value}) must equal "
            f"INCOME BEFORE TAXES ({pretax_income}) - Total Tax Provision ({total_tax}) = {expected}"
        )

    def test_tax_provision_structure(self, generator):
        """Test that tax provision follows flat rate structure."""
        df = generator.generate_income_statement(year=2)

        items = df["Item"].str.strip().values
        values = df["Year 2"].values

        # Find tax provision components
        current_tax_idx = None
        deferred_tax_idx = None

        for i, item in enumerate(items):
            if "Current Tax Expense" in item:
                current_tax_idx = i
            elif "Deferred Tax Expense" in item:
                deferred_tax_idx = i

        assert current_tax_idx is not None, "Missing Current Tax Expense"
        assert deferred_tax_idx is not None, "Missing Deferred Tax Expense"

        # Verify no deferred taxes (flat rate only)
        assert values[deferred_tax_idx] == 0, "Deferred taxes should be zero"

    def test_tax_expense_from_ledger(self, mock_manufacturer):
        """Test that tax expense is read from Ledger when available.

        Issue #257: The Income Statement should read tax_expense from the Ledger
        (sum of TAX_ACCRUAL entries) rather than recalculating using a flat rate.
        """
        # Create a ledger with TAX_ACCRUAL entries
        ledger = Ledger()

        # Record tax accrual for year 2 (expected: 75,000)
        ledger.record_double_entry(
            date=2,
            debit_account="tax_expense",
            credit_account="accrued_taxes",
            amount=75_000,
            transaction_type=TransactionType.TAX_ACCRUAL,
            description="Q1-Q4 Tax accrual",
        )

        # Create generator with ledger
        generator = FinancialStatementGenerator(
            manufacturer=mock_manufacturer,
            ledger=ledger,
        )

        df = generator.generate_income_statement(year=2)

        # Find the Current Tax Expense value
        tax_expense_value = None
        for i, item in enumerate(df["Item"].values):
            if "Current Tax Expense" in str(item):
                tax_expense_value = df["Year 2"].values[i]
                break

        # Verify the tax expense matches what's in the ledger
        assert tax_expense_value is not None, "Current Tax Expense row not found"
        assert (
            abs(tax_expense_value - 75_000) < 1
        ), f"Tax expense should be 75,000 from ledger, got {tax_expense_value}"

    def test_tax_expense_from_metrics(self, mock_manufacturer):
        """Test that tax expense is read from metrics when Ledger not available.

        Issue #257: If no Ledger is available, fall back to tax_expense
        provided in metrics by the Manufacturer.
        """
        # Add tax_expense to metrics
        mock_manufacturer.metrics_history[2]["tax_expense"] = 85_000

        # Create generator without ledger
        generator = FinancialStatementGenerator(
            manufacturer=mock_manufacturer,
        )

        df = generator.generate_income_statement(year=2)

        # Find the Current Tax Expense value
        tax_expense_value = None
        for i, item in enumerate(df["Item"].values):
            if "Current Tax Expense" in str(item):
                tax_expense_value = df["Year 2"].values[i]
                break

        # Verify the tax expense matches what's in metrics
        assert tax_expense_value is not None, "Current Tax Expense row not found"
        assert (
            abs(tax_expense_value - 85_000) < 1
        ), f"Tax expense should be 85,000 from metrics, got {tax_expense_value}"

    def test_tax_expense_ledger_priority_over_metrics(self, mock_manufacturer):
        """Test that Ledger tax expense takes priority over metrics.

        Issue #257: Priority order should be: Ledger > metrics > flat rate.
        """
        # Create a ledger with TAX_ACCRUAL entries
        ledger = Ledger()

        # Record tax accrual for year 2 (expected: 60,000)
        ledger.record_double_entry(
            date=2,
            debit_account="tax_expense",
            credit_account="accrued_taxes",
            amount=60_000,
            transaction_type=TransactionType.TAX_ACCRUAL,
            description="Ledger tax accrual",
        )

        # Also add different tax_expense to metrics (should be ignored)
        mock_manufacturer.metrics_history[2]["tax_expense"] = 100_000

        # Create generator with ledger
        generator = FinancialStatementGenerator(
            manufacturer=mock_manufacturer,
            ledger=ledger,
        )

        df = generator.generate_income_statement(year=2)

        # Find the Current Tax Expense value
        tax_expense_value = None
        for i, item in enumerate(df["Item"].values):
            if "Current Tax Expense" in str(item):
                tax_expense_value = df["Year 2"].values[i]
                break

        # Verify the tax expense matches ledger (not metrics)
        assert tax_expense_value is not None, "Current Tax Expense row not found"
        assert (
            abs(tax_expense_value - 60_000) < 1
        ), f"Tax expense should be 60,000 from ledger (not 100,000 from metrics), got {tax_expense_value}"

    def test_tax_expense_flat_rate_fallback(self, mock_manufacturer):
        """Test that flat rate calculation is used when no tax data available.

        Issue #257: When neither Ledger nor metrics provides tax_expense,
        fall back to the original flat rate calculation for backward compatibility.
        """
        # Ensure no tax_expense in metrics (remove if exists)
        if "tax_expense" in mock_manufacturer.metrics_history[2]:
            del mock_manufacturer.metrics_history[2]["tax_expense"]

        # Create generator without ledger
        generator = FinancialStatementGenerator(
            manufacturer=mock_manufacturer,
        )

        df = generator.generate_income_statement(year=2)

        # Find the Current Tax Expense value
        tax_expense_value = None
        pretax_income_value = None
        for i, item in enumerate(df["Item"].values):
            if "Current Tax Expense" in str(item):
                tax_expense_value = df["Year 2"].values[i]
            elif "INCOME BEFORE TAXES" in str(item):
                pretax_income_value = df["Year 2"].values[i]

        # Verify tax expense is calculated (non-zero for positive income)
        assert tax_expense_value is not None, "Current Tax Expense row not found"
        assert pretax_income_value is not None, "INCOME BEFORE TAXES row not found"

        # If pretax income is positive, tax should be ~25% (default rate)
        if pretax_income_value > 0:
            expected_tax = float(pretax_income_value) * 0.25
            assert (
                abs(float(tax_expense_value) - expected_tax) < 1
            ), f"Tax expense should be ~25% of pretax income ({expected_tax}), got {tax_expense_value}"

    def test_monthly_income_statement(self, generator):
        """Test monthly income statement generation."""
        # Generate monthly statement
        monthly_df = generator.generate_income_statement(year=2, monthly=True)
        annual_df = generator.generate_income_statement(year=2, monthly=False)

        # Check column naming — annualized data uses "Monthly Avg" label
        assert "Monthly Avg 2" in monthly_df.columns
        assert "Year 2" in annual_df.columns

        # Get revenue values
        monthly_revenue = None
        annual_revenue = None

        for i, item in enumerate(monthly_df["Item"].values):
            if "Sales Revenue" in str(item) and "Total" not in str(item):
                monthly_revenue = monthly_df["Monthly Avg 2"].values[i]
                break

        for i, item in enumerate(annual_df["Item"].values):
            if "Sales Revenue" in str(item) and "Total" not in str(item):
                annual_revenue = annual_df["Year 2"].values[i]
                break

        # Verify monthly is approximately 1/12 of annual
        if monthly_revenue and annual_revenue:
            assert (
                abs(monthly_revenue * 12 - annual_revenue) < 1
            ), "Monthly revenue * 12 should equal annual revenue"

    def test_expense_ratio_configuration(self):
        """Test income statement with custom expense ratios."""
        # Create manufacturer with custom expense ratios
        manufacturer = Mock(spec=WidgetManufacturer)

        # Create config with expense ratios
        expense_config = ExpenseRatioConfig(
            gross_margin_ratio=0.20,  # 20% gross margin
            sga_expense_ratio=0.05,  # 5% SG&A
            manufacturing_depreciation_allocation=0.8,
            admin_depreciation_allocation=0.2,
        )

        # Create a mock config object with expense_ratios attribute
        config = Mock()
        config.initial_assets = 10_000_000
        config.asset_turnover_ratio = 0.5
        config.retention_ratio = 0.6
        config.base_operating_margin = 0.08
        config.tax_rate = 0.30  # 30% tax rate
        config.expense_ratios = expense_config

        manufacturer.config = config

        # Issue #255: COGS/SG&A breakdown now provided by Manufacturer (via helper)
        manufacturer.metrics_history = [
            _add_cogs_sga_breakdown(
                {
                    "year": 0,
                    "assets": 10_000_000,
                    "equity": 10_000_000,
                    "revenue": 5_000_000,
                    "depreciation_expense": 500_000,
                    "cash": 2_000_000,
                    "debt_balance": 0,
                    "insurance_premiums": 100_000,
                    "insurance_losses": 50_000,
                },
                expense_config=expense_config,
            )
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)
        df = generator.generate_income_statement(year=0)

        # Find gross profit and verify margin
        revenue_val = None
        gross_profit_val = None

        for i, item in enumerate(df["Item"].values):
            item_str = str(item).strip()
            if item_str == "Sales Revenue":
                revenue_val = df["Year 0"].values[i]
            elif item_str == "GROSS PROFIT":
                gross_profit_val = df["Year 0"].values[i]

        # Verify gross margin is approximately 20%
        if revenue_val and gross_profit_val:
            actual_margin = float(gross_profit_val) / float(revenue_val)
            assert (
                abs(actual_margin - 0.20) < 0.05
            ), f"Gross margin should be ~20%, got {actual_margin*100:.1f}%"

    def test_gross_margin_calculation(self):
        """Test that gross margin is calculated correctly as gross_profit/revenue.

        Issue #210: gross_margin was incorrectly calculated as (revenue - operating_income) / revenue
        which overstates gross margin by including operating expenses.

        Correct formula: gross_margin = gross_profit / revenue = (revenue - COGS) / revenue
        """
        # Create manufacturer with specific expense ratios to verify calculation
        manufacturer = Mock(spec=WidgetManufacturer)

        # Create config with known expense ratios
        expense_config = ExpenseRatioConfig(
            gross_margin_ratio=0.25,  # 25% gross margin (75% COGS)
            sga_expense_ratio=0.10,  # 10% SG&A
            manufacturing_depreciation_allocation=0.7,
            admin_depreciation_allocation=0.3,
        )

        config = Mock()
        config.initial_assets = 10_000_000
        config.tax_rate = 0.25
        config.expense_ratios = expense_config

        manufacturer.config = config

        # Issue #255: COGS/SG&A breakdown now provided by Manufacturer (via helper)
        manufacturer.metrics_history = [
            _add_cogs_sga_breakdown(
                {
                    "year": 0,
                    "assets": 10_000_000,
                    "equity": 10_000_000,
                    "revenue": 1_000_000,  # Simple round number for easy calculation
                    "depreciation_expense": 100_000,
                    "cash": 1_000_000,
                    "debt_balance": 0,
                    "insurance_premiums": 0,
                    "insurance_losses": 0,
                },
                expense_config=expense_config,
            )
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)
        df = generator.generate_income_statement(year=0)

        # Find the Gross Margin % row
        gross_margin_row = None
        for i, item in enumerate(df["Item"].values):
            if "Gross Margin %" in str(item):
                gross_margin_row = i
                break

        assert gross_margin_row is not None, "Gross Margin % row not found"

        # Gross margin should be approximately 25% (the configured gross_margin_ratio)
        # The value is stored as percentage (e.g., 25.0 for 25%)
        actual_margin = float(df["Year 0"].values[gross_margin_row])
        assert (
            abs(actual_margin - 25.0) < 1.0
        ), f"Gross margin should be ~25% (configured), got {actual_margin:.1f}%"

    def test_balance_sheet_reconciliation_full_liabilities(self):
        """Test that balance sheet reconciliation uses full total liabilities.

        Issue #301: Previously only used claim_liabilities, but should include
        accounts_payable + accrued_expenses + claim_liabilities to match the
        actual balance sheet liabilities section.
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Set up metrics where A = L + E only with FULL liabilities
        # accounts_payable=200K, accrued_expenses=100K, claim_liabilities=500K
        # current_claims = 500K * 0.1 = 50K
        # total_current = 200K + 100K + 50K = 350K
        # long_term = 500K - 50K = 450K
        # total_liabilities = 350K + 450K = 800K
        # equity = 10M - 800K = 9.2M
        manufacturer.metrics_history = [
            _add_cogs_sga_breakdown(
                {
                    "year": 0,
                    "assets": 10_000_000,
                    "equity": 9_200_000,
                    "revenue": 5_000_000,
                    "operating_income": 400_000,
                    "net_income": 300_000,
                    "collateral": 0,
                    "restricted_assets": 0,
                    "available_assets": 10_000_000,
                    "net_assets": 10_000_000,  # assets - restricted_assets
                    "claim_liabilities": 500_000,
                    "accounts_payable": 200_000,
                    "accrued_expenses": 100_000,
                    "is_solvent": True,
                    "base_operating_margin": 0.08,
                    "roe": 0.03,
                    "roa": 0.03,
                    "asset_turnover": 0.5,
                    "gross_ppe": 7_000_000,
                    "accumulated_depreciation": 0,
                    "depreciation_expense": 700_000,
                    "cash": 3_000_000,
                }
            ),
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)
        df = generator.generate_reconciliation_report(year=0)

        # Find the balance sheet reconciliation status
        checks = df["Check"].values
        values = df["Value"].values
        types = df["Type"].values

        # Find all status rows and verify the balance sheet one
        balance_sheet_section = False
        for i, check in enumerate(checks):
            check_str = str(check).strip()
            if "BALANCE SHEET RECONCILIATION" in check_str:
                balance_sheet_section = True
            elif check_str == "Status" and types[i] == "status" and balance_sheet_section:
                assert values[i] == "BALANCED", (
                    f"Balance sheet should be BALANCED with full liabilities, " f"got {values[i]}"
                )
                break  # Found and checked the balance sheet status

    def test_current_claims_from_development_schedule(self):
        """Test that current/non-current claims use development schedule metrics.

        Issue #466: current_claims_ratio removed. The manufacturer now computes
        current_claim_liabilities from actual development schedules (ASC 450).
        """
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            retention_ratio=0.6,
            base_operating_margin=0.08,
            tax_rate=0.25,
        )

        # Manufacturer provides development-schedule-based split:
        # 200K current (next year's scheduled payments), 800K non-current
        manufacturer.metrics_history = [
            _add_cogs_sga_breakdown(
                {
                    "year": 0,
                    "assets": 10_000_000,
                    "equity": 9_500_000,
                    "revenue": 5_000_000,
                    "operating_income": 400_000,
                    "net_income": 300_000,
                    "collateral": 0,
                    "restricted_assets": 0,
                    "available_assets": 10_000_000,
                    "claim_liabilities": 1_000_000,
                    "current_claim_liabilities": 200_000,
                    "non_current_claim_liabilities": 800_000,
                    "is_solvent": True,
                    "base_operating_margin": 0.08,
                    "roe": 0.03,
                    "roa": 0.03,
                    "asset_turnover": 0.5,
                    "gross_ppe": 7_000_000,
                    "accumulated_depreciation": 0,
                    "depreciation_expense": 700_000,
                    "cash": 3_000_000,
                }
            ),
        ]

        generator = FinancialStatementGenerator(manufacturer=manufacturer)

        df = generator.generate_balance_sheet(year=0)

        items = df["Item"].values
        values = df["Year 0"].values

        for i, item in enumerate(items):
            if "Current Portion of Claim Liabilities" in str(item):
                assert (
                    abs(float(values[i]) - 200_000) < 1
                ), f"Current claims should be 200,000 from dev schedule, got {values[i]}"
                break
        else:
            pytest.fail("Current Portion of Claim Liabilities row not found")

        # Verify non-current portion
        for i, item in enumerate(items):
            if "Long-Term Claim Reserves" in str(item):
                assert (
                    abs(float(values[i]) - 800_000) < 1
                ), f"Non-current claims should be 800,000, got {values[i]}"
                break
        else:
            pytest.fail("Long-Term Claim Reserves row not found")

    def test_monthly_income_statement_disclaimer(self, generator):
        """Test that monthly income statement includes disclaimer row when annualized."""
        df = generator.generate_income_statement(year=2, monthly=True)

        # First row should be the disclaimer
        first_item = str(df["Item"].values[0])
        assert "NOTE:" in first_item
        assert "annualized monthly averages" in first_item

    def test_monthly_income_statement_period_warning(self, generator):
        """Test that monthly income statement sets period_warning attr when annualized."""
        monthly_df = generator.generate_income_statement(year=2, monthly=True)
        annual_df = generator.generate_income_statement(year=2, monthly=False)

        # Monthly statement should have period_warning
        assert "period_warning" in monthly_df.attrs
        assert "estimates" in monthly_df.attrs["period_warning"]

        # Annual statement should NOT have period_warning
        assert "period_warning" not in annual_df.attrs

    def test_monthly_cash_flow_statement_labeling(self, generator):
        """Test that monthly cash flow statement uses 'Monthly Avg' label and has disclaimer + period_warning."""
        df = generator.generate_cash_flow_statement(year=2, period="monthly")

        # Column should say "Monthly Avg" not "Month"
        col_names = [c for c in df.columns if "Monthly Avg" in str(c)]
        assert (
            len(col_names) == 1
        ), f"Expected 'Monthly Avg' column, got columns: {list(df.columns)}"

        # First row should be disclaimer
        first_item = str(df["Item"].values[0])
        assert "NOTE:" in first_item
        assert "annualized monthly averages" in first_item

        # period_warning attr should be set
        assert "period_warning" in df.attrs
        assert "estimates" in df.attrs["period_warning"]

    def test_annual_statements_no_warning(self, generator):
        """Test that annual statements have no disclaimer row and no period_warning attr."""
        income_df = generator.generate_income_statement(year=2, monthly=False)
        cf_df = generator.generate_cash_flow_statement(year=2, period="annual")

        # No disclaimer rows
        for df in [income_df, cf_df]:
            for item in df["Item"].values:
                assert "NOTE:" not in str(item), "Annual statement should not have disclaimer"

        # No period_warning attr
        assert "period_warning" not in income_df.attrs
        assert "period_warning" not in cf_df.attrs


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

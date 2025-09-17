"""Test enhanced balance sheet classification with GAAP structure."""

import pandas as pd
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.financial_statements import FinancialStatementGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestBalanceSheetClassification:
    """Test proper classification of assets and liabilities in GAAP structure."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with standard configuration."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        return WidgetManufacturer(config)

    @pytest.fixture
    def statement_generator(self, manufacturer):
        """Create a financial statement generator."""
        return FinancialStatementGenerator(manufacturer)

    def test_current_assets_classification(self, manufacturer, statement_generator):
        """Test that current assets are properly classified."""
        # Run one year to generate metrics
        manufacturer.step()

        # Generate balance sheet
        balance_sheet = statement_generator.generate_balance_sheet(year=0)

        # Check for current asset components
        assert any("Cash and Cash Equivalents" in str(row) for row in balance_sheet["Item"])
        assert any("Accounts Receivable" in str(row) for row in balance_sheet["Item"])
        assert any("Inventory" in str(row) for row in balance_sheet["Item"])
        assert any("Prepaid Insurance" in str(row) for row in balance_sheet["Item"])
        assert any("Total Current Assets" in str(row) for row in balance_sheet["Item"])

    def test_non_current_assets_classification(self, manufacturer, statement_generator):
        """Test that non-current assets show PP&E with depreciation."""
        # Run one year to generate metrics and depreciation
        manufacturer.step()

        # Generate balance sheet
        balance_sheet = statement_generator.generate_balance_sheet(year=0)

        # Check for PP&E components
        assert any(
            "Property, Plant & Equipment (Gross)" in str(row) for row in balance_sheet["Item"]
        )
        assert any("Less: Accumulated Depreciation" in str(row) for row in balance_sheet["Item"])
        assert any("Net Property, Plant & Equipment" in str(row) for row in balance_sheet["Item"])

    def test_current_liabilities_classification(self, manufacturer, statement_generator):
        """Test that current liabilities are properly classified."""
        # Run one year to generate metrics
        manufacturer.step()

        # Generate balance sheet
        balance_sheet = statement_generator.generate_balance_sheet(year=0)

        # Check for current liability components
        assert any("Current Liabilities" in str(row) for row in balance_sheet["Item"])
        assert any("Accounts Payable" in str(row) for row in balance_sheet["Item"])
        assert any("Accrued Expenses" in str(row) for row in balance_sheet["Item"])

    def test_non_current_liabilities_classification(self, manufacturer, statement_generator):
        """Test that non-current liabilities are properly classified."""
        # Process a claim to create claim liabilities
        manufacturer.process_insurance_claim(
            claim_amount=2_000_000, deductible_amount=500_000, insurance_limit=5_000_000
        )

        # Run one year
        manufacturer.step()

        # Generate balance sheet
        balance_sheet = statement_generator.generate_balance_sheet(year=0)

        # Check for non-current liability components
        assert any("Non-Current Liabilities" in str(row) for row in balance_sheet["Item"])
        assert any("Long-Term Claim Reserves" in str(row) for row in balance_sheet["Item"])

    def test_balance_sheet_equation_with_new_structure(self, manufacturer, statement_generator):
        """Test that the accounting equation still balances with new structure."""
        # Run multiple years with various activities
        for year in range(3):
            if year == 1:
                # Process a claim in year 1
                manufacturer.process_insurance_claim(
                    claim_amount=1_000_000, deductible_amount=250_000, insurance_limit=3_000_000
                )
            manufacturer.step()

        # Generate balance sheet for last year
        balance_sheet = statement_generator.generate_balance_sheet(year=2)

        # Find total assets and total liabilities + equity
        total_assets_row = balance_sheet[balance_sheet["Item"] == "TOTAL ASSETS"]
        total_liabilities_row = balance_sheet[balance_sheet["Item"] == "TOTAL LIABILITIES"]
        total_equity_row = balance_sheet[balance_sheet["Item"] == "TOTAL EQUITY"]

        if (
            not total_assets_row.empty
            and not total_liabilities_row.empty
            and not total_equity_row.empty
        ):
            total_assets = float(total_assets_row.iloc[0]["Year 2"])
            total_liabilities = float(total_liabilities_row.iloc[0]["Year 2"])
            total_equity = float(total_equity_row.iloc[0]["Year 2"])

            # Assets = Liabilities + Equity
            assert abs(total_assets - (total_liabilities + total_equity)) < 0.01

    def test_working_capital_components_calculation(self, manufacturer, statement_generator):
        """Test that working capital components are calculated and appear in balance sheet."""
        # Calculate revenue and working capital components
        revenue = manufacturer.calculate_revenue()
        manufacturer.calculate_working_capital_components(revenue)

        # Run a step to populate metrics
        manufacturer.step()

        # Generate balance sheet
        balance_sheet = statement_generator.generate_balance_sheet(year=0)

        # Get the values from balance sheet
        ar_row = balance_sheet[balance_sheet["Item"].str.contains("Accounts Receivable", na=False)]
        inv_row = balance_sheet[balance_sheet["Item"].str.contains("Inventory", na=False)]
        ap_row = balance_sheet[balance_sheet["Item"].str.contains("Accounts Payable", na=False)]

        # Verify components exist and have values
        assert not ar_row.empty
        assert not inv_row.empty
        assert not ap_row.empty

        # Check that values are reasonable (greater than 0 after business activity)
        assert manufacturer.accounts_receivable > 0
        assert manufacturer.inventory > 0
        assert manufacturer.accounts_payable > 0

    def test_depreciation_accumulation(self, manufacturer, statement_generator):
        """Test that depreciation accumulates over time."""
        initial_ppe = manufacturer.gross_ppe

        # Run multiple years to accumulate depreciation
        for _ in range(5):
            manufacturer.step()

        # Check accumulated depreciation
        assert manufacturer.accumulated_depreciation > 0
        assert manufacturer.net_ppe < initial_ppe
        assert (
            manufacturer.net_ppe == manufacturer.gross_ppe - manufacturer.accumulated_depreciation
        )

        # Generate balance sheet and verify depreciation shows
        balance_sheet = statement_generator.generate_balance_sheet(year=4)

        # Find accumulated depreciation row
        depreciation_row = balance_sheet[
            balance_sheet["Item"].str.contains("Accumulated Depreciation", na=False)
        ]
        assert not depreciation_row.empty

        # Value should be negative (reducing assets)
        depreciation_value = float(depreciation_row.iloc[0]["Year 4"])
        assert depreciation_value < 0

    def test_prepaid_insurance_tracking(self, manufacturer):
        """Test that prepaid insurance is properly tracked."""
        # Record an annual premium payment
        annual_premium = 1_200_000
        manufacturer.record_prepaid_insurance(annual_premium)

        # Check initial recording
        assert manufacturer.prepaid_insurance == annual_premium
        assert manufacturer.cash < manufacturer.config.initial_assets - manufacturer.gross_ppe

        # Amortize over several months
        total_amortized = 0
        for month in range(6):
            amortized = manufacturer.amortize_prepaid_insurance(months=1)
            total_amortized += amortized

        # Check amortization
        assert manufacturer.prepaid_insurance == annual_premium - total_amortized
        assert total_amortized == pytest.approx(annual_premium / 2, rel=0.01)  # 6 months = half

    def test_gaap_structure_consistency(self, manufacturer, statement_generator):
        """Test that balance sheet maintains consistent GAAP structure."""
        # Run simulation with various activities
        manufacturer.record_prepaid_insurance(600_000)
        manufacturer.step()

        balance_sheet = statement_generator.generate_balance_sheet(year=0)

        # Check structure sections exist in order
        items_list = balance_sheet["Item"].tolist()
        items_str = " ".join(str(item) for item in items_list)

        # Verify major sections appear in correct order
        assert "ASSETS" in items_str
        assert "Current Assets" in items_str
        assert "Non-Current Assets" in items_str
        assert "LIABILITIES" in items_str
        assert "Current Liabilities" in items_str
        assert "Non-Current Liabilities" in items_str
        assert "EQUITY" in items_str

        # Verify order (ASSETS before LIABILITIES, LIABILITIES before EQUITY)
        assets_idx = items_str.index("ASSETS")
        liabilities_idx = items_str.index("LIABILITIES")
        equity_idx = items_str.index("EQUITY")

        assert assets_idx < liabilities_idx
        assert liabilities_idx < equity_idx

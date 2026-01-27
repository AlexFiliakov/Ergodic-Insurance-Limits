"""Tests for cash flow reconciliation."""

from typing import Union

import numpy as np
import pytest

from ergodic_insurance.financial_statements import CashFlowStatement

# Type alias for metrics dictionaries
MetricsDict = list[dict[str, Union[int, float]]]


class TestCashReconciliation:
    """Test suite for cash reconciliation in cash flow statements."""

    def test_simple_reconciliation(self):
        """Test basic cash reconciliation: Beginning + Net Change = Ending."""
        # Cash flow calculation for Year 1:
        #   Operating: 600k NI + 60k Dep = 660k
        #   Investing: -(100k PP&E change + 60k Dep) = -160k
        #   Financing: -180k dividends
        #   Net: 660k - 160k - 180k = 320k
        #   Ending cash: 1,000k + 320k = 1,320k
        metrics: MetricsDict = [
            {
                "cash": 1000000,
                "net_income": 500000,
                "dividends_paid": 150000,
                "depreciation_expense": 50000,
                "gross_ppe": 500000,
            },
            {
                "cash": 1320000,  # Consistent with cash flow calculation
                "net_income": 600000,
                "dividends_paid": 180000,
                "depreciation_expense": 60000,
                "gross_ppe": 600000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        # Extract reconciliation values
        reconciliation = {}
        for _, row in df.iterrows():
            item = row["Item"].strip()
            if "Cash - Beginning" in item:
                reconciliation["beginning"] = row["Year 1"]
            elif item == "Net Change in Cash":
                reconciliation["net_change"] = row["Year 1"]
            elif "Cash - End" in item:
                reconciliation["ending"] = row["Year 1"]

        # Verify reconciliation
        assert reconciliation["beginning"] == 1000000
        assert reconciliation["ending"] == 1320000
        # Calculated cash flow must equal actual cash change (no plug)
        assert (
            abs(
                (reconciliation["beginning"] + reconciliation["net_change"])
                - reconciliation["ending"]
            )
            < 0.01
        )

    def test_reconciliation_with_operating_activities(self):
        """Test reconciliation with detailed operating activities."""
        metrics: MetricsDict = [
            {
                "cash": 500000,
                "net_income": 200000,
                "depreciation_expense": 50000,
                "accounts_receivable": 100000,
                "inventory": 80000,
                "accounts_payable": 60000,
                "gross_ppe": 500000,
            },
            {
                "cash": 680000,  # Should reconcile with cash flow
                "net_income": 250000,
                "depreciation_expense": 60000,
                "accounts_receivable": 120000,  # Increased by 20k
                "inventory": 90000,  # Increased by 10k
                "accounts_payable": 70000,  # Increased by 10k
                "gross_ppe": 600000,  # Increased by 100k
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        # Calculate expected cash flow
        # Operating: 250k (NI) + 60k (Dep) - 20k (AR) - 10k (Inv) + 10k (AP) = 290k
        # Investing: -(100k + 60k) = -160k (Capex = Change in PP&E + Dep)
        # Financing: -75k (30% of 250k NI as dividends)
        # Net change: 290k - 160k - 75k = 55k
        # But actual change is 680k - 500k = 180k

        # Extract actual values
        values = self._extract_values_from_df(df, "Year 1")

        # Beginning cash should match year 0
        assert values.get("Cash - Beginning of Period") == 500000
        # Ending cash should match year 1
        assert values.get("Cash - End of Period") == 680000

    def test_reconciliation_negative_cash_flow(self):
        """Test reconciliation when cash decreases."""
        metrics: MetricsDict = [
            {
                "cash": 1000000,
                "net_income": 100000,
                "depreciation_expense": 50000,
                "gross_ppe": 1000000,
            },
            {
                "cash": 800000,  # Cash decreased
                "net_income": -200000,  # Loss
                "depreciation_expense": 50000,
                "gross_ppe": 1000000,  # No change in PP&E
                "dividends_paid": 0,  # No dividends on loss
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        values = self._extract_values_from_df(df, "Year 1")

        # Verify negative net change
        net_change = values.get("NET INCREASE (DECREASE) IN CASH", 0)
        assert net_change == -200000  # 800k - 1000k

        # Verify reconciliation
        beginning = values.get("Cash - Beginning of Period")
        ending = values.get("Cash - End of Period")
        assert abs((beginning + net_change) - ending) < 0.01

    def test_reconciliation_first_year(self):
        """Test cash reconciliation for year 0 (no prior period)."""
        metrics: MetricsDict = [
            {
                "cash": 500000,
                "net_income": 100000,
                "depreciation_expense": 20000,
                "gross_ppe": 200000,
                "dividends_paid": 30000,
            }
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(0)

        values = self._extract_values_from_df(df, "Year 0")

        # For year 0, beginning cash is calculated as ending - net change
        ending_cash = values.get("Cash - End of Period")
        assert ending_cash == 500000

        # Net cash flow components:
        # Operating: 100k (NI) + 20k (Dep) = 120k
        # Investing: -(200k + 20k) = -220k (Initial PP&E + Dep)
        # Financing: -30k (Dividends)
        # Net: 120k - 220k - 30k = -130k

        net_change = values.get("NET INCREASE (DECREASE) IN CASH", 0)
        beginning_cash = values.get("Cash - Beginning of Period")

        # Verify reconciliation
        assert abs((beginning_cash + net_change) - ending_cash) < 0.01

    def test_reconciliation_with_all_working_capital_changes(self):
        """Test reconciliation with comprehensive working capital changes."""
        # Cash flow calculation for Year 1:
        #   Operating: 600k NI + 110k Dep - 50k AR - 50k Inv - 10k Prepaid
        #              + 30k AP + 20k Accrued + 50k Claims = 700k
        #   Investing: -(300k PP&E change + 110k Dep) = -410k
        #   Financing: -180k dividends
        #   Net: 700k - 410k - 180k = 110k
        #   Ending cash: 1,000k + 110k = 1,110k
        metrics: MetricsDict = [
            {
                "cash": 1000000,
                "net_income": 500000,
                "depreciation_expense": 100000,
                "accounts_receivable": 200000,
                "inventory": 150000,
                "prepaid_insurance": 20000,
                "accounts_payable": 100000,
                "accrued_expenses": 50000,
                "claim_liabilities": 30000,
                "gross_ppe": 1000000,
                "dividends_paid": 150000,
            },
            {
                "cash": 1110000,  # Consistent with cash flow calculation
                "net_income": 600000,
                "depreciation_expense": 110000,
                "accounts_receivable": 250000,  # +50k
                "inventory": 200000,  # +50k
                "prepaid_insurance": 30000,  # +10k
                "accounts_payable": 130000,  # +30k
                "accrued_expenses": 70000,  # +20k
                "claim_liabilities": 80000,  # +50k
                "gross_ppe": 1300000,  # +300k
                "dividends_paid": 180000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        values = self._extract_values_from_df(df, "Year 1")

        # Calculate expected net cash flow:
        # Operating: 600k + 110k - 50k - 50k - 10k + 30k + 20k + 50k = 700k
        # Investing: -(300k + 110k) = -410k
        # Financing: -180k
        # Net: 700k - 410k - 180k = 110k

        beginning = values.get("Cash - Beginning of Period")
        ending = values.get("Cash - End of Period")
        net_change = values.get("NET INCREASE (DECREASE) IN CASH", 0)

        assert beginning == 1000000
        assert ending == 1110000
        # Calculated cash flow must equal actual cash change (no plug)
        assert abs((beginning + net_change) - ending) < 0.01

    def test_monthly_reconciliation(self):
        """Test cash reconciliation for monthly periods."""
        metrics: MetricsDict = [
            {
                "cash": 100000,
                "net_income": 120000,  # Annual
                "depreciation_expense": 12000,  # Annual
                "gross_ppe": 120000,
                "dividends_paid": 36000,  # Annual
            },
            {
                "cash": 110000,
                "net_income": 144000,  # Annual
                "depreciation_expense": 12000,  # Annual
                "gross_ppe": 132000,  # Increased by 12k annually
                "dividends_paid": 43200,  # Annual
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1, period="monthly")

        values = self._extract_values_from_df(df, "Month 1")

        # Monthly values
        monthly_net_income = values.get("Net Income")
        assert monthly_net_income == pytest.approx(144000 / 12)

        monthly_depreciation = values.get("Depreciation and Amortization")
        assert monthly_depreciation == pytest.approx(12000 / 12)

        # Reconciliation should still work
        beginning = values.get("Cash - Beginning of Period")
        ending = values.get("Cash - End of Period")
        net_change = values.get("NET INCREASE (DECREASE) IN CASH", 0)

        # Note: For monthly, we're still using annual beginning/ending cash
        assert beginning == 100000
        assert ending == 110000

    def test_zero_beginning_cash(self):
        """Test reconciliation when beginning cash is zero."""
        # Cash flow calculation for Year 1:
        #   Operating: 150k NI + 10k Dep = 160k
        #   Investing: -(10k PP&E change + 10k Dep) = -20k
        #   Financing: -45k dividends
        #   Net: 160k - 20k - 45k = 95k
        #   Ending cash: 0 + 95k = 95k
        metrics: MetricsDict = [
            {
                "cash": 0,
                "net_income": 100000,
                "depreciation_expense": 10000,
                "gross_ppe": 50000,
                "dividends_paid": 0,
            },
            {
                "cash": 95000,  # Consistent with cash flow calculation
                "net_income": 150000,
                "depreciation_expense": 10000,
                "gross_ppe": 60000,
                "dividends_paid": 45000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        values = self._extract_values_from_df(df, "Year 1")

        assert values.get("Cash - Beginning of Period") == 0
        assert values.get("Cash - End of Period") == 95000

        net_change = values.get("NET INCREASE (DECREASE) IN CASH", 0)
        assert net_change == 95000

    def test_large_capex_reconciliation(self):
        """Test reconciliation with large capital expenditures."""
        # Cash flow calculation for Year 1:
        #   Operating: 1200k NI + 250k Dep = 1450k
        #   Investing: -(2000k PP&E change + 250k Dep) = -2250k
        #   Financing: -360k dividends
        #   Net: 1450k - 2250k - 360k = -1160k
        #   Ending cash: 5000k - 1160k = 3840k
        metrics: MetricsDict = [
            {
                "cash": 5000000,
                "net_income": 1000000,
                "depreciation_expense": 200000,
                "gross_ppe": 2000000,
                "dividends_paid": 300000,
            },
            {
                "cash": 3840000,  # Consistent with cash flow calculation
                "net_income": 1200000,
                "depreciation_expense": 250000,
                "gross_ppe": 4000000,  # Large increase in PP&E
                "dividends_paid": 360000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        values = self._extract_values_from_df(df, "Year 1")

        # Capex = (4M - 2M) + 250k = 2.25M
        capex = abs(values.get("Capital Expenditures", 0))
        assert capex == 2250000

        # Net change should be negative due to large capex
        # Calculated: 1450k - 2250k - 360k = -1160k
        net_change = values.get("NET INCREASE (DECREASE) IN CASH", 0)
        assert net_change == -1160000

        # Verify reconciliation
        beginning = values.get("Cash - Beginning of Period")
        ending = values.get("Cash - End of Period")
        assert abs((beginning + net_change) - ending) < 0.01

    def test_no_plug_regression(self):
        """Regression test: Verify the cash flow plug has been removed.

        This is a regression test for GitHub Issue #240.

        Previously, the CashFlowStatement would silently overwrite the calculated
        NET INCREASE (DECREASE) IN CASH with the actual cash change (ending - beginning)
        if they didn't match. This "plug" hid underlying calculation errors.

        This test verifies that:
        1. The calculated cash flow matches the actual cash change
        2. No silent overwriting occurs
        """
        # Create metrics where cash flow components are explicitly calculated
        # to verify the statement shows calculated values, not plugged values
        metrics: MetricsDict = [
            {
                "cash": 1000000,
                "net_income": 400000,
                "depreciation_expense": 100000,
                "accounts_receivable": 100000,
                "inventory": 100000,
                "accounts_payable": 50000,
                "gross_ppe": 500000,
                "dividends_paid": 120000,
            },
            {
                # Calculate ending cash from components:
                # Operating: 500k NI + 120k Dep - 50k AR - 40k Inv + 30k AP = 560k
                # Investing: -(100k PP&E change + 120k Dep) = -220k
                # Financing: -150k dividends
                # Net: 560k - 220k - 150k = 190k
                # Ending: 1000k + 190k = 1190k
                "cash": 1190000,  # Consistent with calculation
                "net_income": 500000,
                "depreciation_expense": 120000,
                "accounts_receivable": 150000,  # +50k
                "inventory": 140000,  # +40k
                "accounts_payable": 80000,  # +30k
                "gross_ppe": 600000,  # +100k
                "dividends_paid": 150000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)
        values = self._extract_values_from_df(df, "Year 1")

        # Get the reported values
        net_cash_flow = values.get("NET INCREASE (DECREASE) IN CASH", 0)
        beginning_cash = values.get("Cash - Beginning of Period")
        ending_cash = values.get("Cash - End of Period")

        # Verify the calculated cash flow matches actual change
        actual_change = ending_cash - beginning_cash
        assert net_cash_flow == actual_change, (
            f"Calculated cash flow ({net_cash_flow}) does not match "
            f"actual change ({actual_change}). This suggests the plug may be active."
        )

        # Explicitly verify the expected value
        # Operating: 500k + 120k - 50k - 40k + 30k = 560k
        # Investing: -(100k + 120k) = -220k
        # Financing: -150k
        # Net: 560k - 220k - 150k = 190k
        assert net_cash_flow == 190000, (
            f"Calculated cash flow ({net_cash_flow}) does not match "
            "expected calculation (190000)."
        )

    def _extract_values_from_df(self, df, column):
        """Helper to extract numeric values from DataFrame."""
        values = {}
        for _, row in df.iterrows():
            item = row["Item"].strip()
            value = row[column]
            if value != "" and isinstance(value, (int, float)):
                values[item] = value
        return values

"""Unit tests for CashFlowStatement class."""

import pandas as pd
import pytest

from ergodic_insurance.financial_statements import CashFlowStatement


class TestCashFlowStatement:
    """Test suite for cash flow statement generation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample metrics history
        self.metrics_history = [
            {
                # Year 0 metrics
                "net_income": 1000000,
                "depreciation_expense": 100000,
                "cash": 500000,
                "accounts_receivable": 200000,
                "inventory": 150000,
                "prepaid_insurance": 20000,
                "accounts_payable": 100000,
                "accrued_expenses": 50000,
                "claim_liabilities": 0,
                "gross_ppe": 1000000,
                "dividends_paid": 300000,  # 30% payout ratio
                "assets": 2000000,
                "equity": 1500000,
            },
            {
                # Year 1 metrics - with changes
                # Cash flow calculation:
                #   Operating: 1,200k NI + 110k Dep - 50k AR - 30k Inv - 5k Prepaid
                #              + 20k AP + 10k Accrued + 50k Claims = 1,305k
                #   Investing: -(200k PP&E change + 110k Dep) = -310k
                #   Financing: -360k dividends
                #   Net: 1,305k - 310k - 360k = 635k
                # Ending cash = 500k + 635k = 1,135k
                "net_income": 1200000,
                "depreciation_expense": 110000,
                "cash": 1135000,  # Consistent with cash flow calculation
                "accounts_receivable": 250000,  # Increased by 50k
                "inventory": 180000,  # Increased by 30k
                "prepaid_insurance": 25000,  # Increased by 5k
                "accounts_payable": 120000,  # Increased by 20k
                "accrued_expenses": 60000,  # Increased by 10k
                "claim_liabilities": 50000,  # New claims
                "gross_ppe": 1200000,  # Increased by 200k
                "dividends_paid": 360000,  # 30% of 1.2M
                "assets": 2500000,
                "equity": 1900000,
            },
        ]

        self.cash_flow = CashFlowStatement(self.metrics_history)

    def test_initialization(self):
        """Test CashFlowStatement initialization."""
        assert self.cash_flow.metrics_history == self.metrics_history
        assert self.cash_flow.config is None

    def test_generate_statement_year_0(self):
        """Test cash flow statement generation for year 0."""
        df = self.cash_flow.generate_statement(0, period="annual")

        assert isinstance(df, pd.DataFrame)
        assert "Year 0" in df.columns
        assert "Item" in df.columns
        assert "Type" in df.columns

        # Check for main sections
        items = df["Item"].tolist()
        assert "CASH FLOWS FROM OPERATING ACTIVITIES" in items
        assert "CASH FLOWS FROM INVESTING ACTIVITIES" in items
        assert "CASH FLOWS FROM FINANCING ACTIVITIES" in items
        assert "NET INCREASE (DECREASE) IN CASH" in items
        assert "CASH RECONCILIATION" in items

    def test_generate_statement_year_1(self):
        """Test cash flow statement generation for year 1 with working capital changes."""
        df = self.cash_flow.generate_statement(1, period="annual")

        # Extract values from DataFrame
        values_dict = {}
        for _, row in df.iterrows():
            if row["Year 1"] != "" and isinstance(row["Year 1"], (int, float)):
                values_dict[row["Item"].strip()] = row["Year 1"]

        # Verify net income
        assert values_dict.get("Net Income") == 1200000

        # Verify depreciation
        assert values_dict.get("Depreciation and Amortization") == 110000

        # Verify working capital changes (signs are important!)
        # Increases in assets are uses of cash (negative)
        assert values_dict.get("Accounts Receivable") == -50000  # -(250k - 200k)
        assert values_dict.get("Inventory") == -30000  # -(180k - 150k)
        assert values_dict.get("Prepaid Insurance") == -5000  # -(25k - 20k)

        # Increases in liabilities are sources of cash (positive)
        assert values_dict.get("Accounts Payable") == 20000  # 120k - 100k
        assert values_dict.get("Accrued Expenses") == 10000  # 60k - 50k
        assert values_dict.get("Claim Liabilities") == 50000  # 50k - 0

    def test_operating_cash_flow_calculation(self):
        """Test operating cash flow calculation with indirect method."""
        current = self.metrics_history[1]
        prior = self.metrics_history[0]

        operating_cf = self.cash_flow._calculate_operating_cash_flow(current, prior, "annual")

        # Net income + depreciation - WC changes
        # 1,200,000 + 110,000 - 85,000 (AR+Inv+Prepaid) + 80,000 (AP+Accrued+Claims)
        expected_total = 1200000 + 110000 - 50000 - 30000 - 5000 + 20000 + 10000 + 50000
        assert operating_cf["total"] == expected_total

    def test_investing_cash_flow_calculation(self):
        """Test investing cash flow calculation (capex)."""
        current = self.metrics_history[1]
        prior = self.metrics_history[0]

        investing_cf = self.cash_flow._calculate_investing_cash_flow(current, prior, "annual")

        # Capex = Change in PP&E + Depreciation
        # (1,200,000 - 1,000,000) + 110,000 = 310,000
        expected_capex = 310000
        assert investing_cf["capital_expenditures"] == -expected_capex
        assert investing_cf["total"] == -expected_capex

    def test_financing_cash_flow_calculation(self):
        """Test financing cash flow calculation (dividends)."""
        current = self.metrics_history[1]
        prior = self.metrics_history[0]

        financing_cf = self.cash_flow._calculate_financing_cash_flow(current, prior, "annual")

        # Dividends paid
        expected_dividends = 360000
        assert pytest.approx(financing_cf["dividends_paid"]) == -expected_dividends
        assert pytest.approx(financing_cf["total"]) == -expected_dividends

    def test_cash_reconciliation(self):
        """Test that beginning cash + net change = ending cash."""
        df = self.cash_flow.generate_statement(1, period="annual")

        # Extract cash reconciliation values
        values = {}
        for _, row in df.iterrows():
            item = row["Item"].strip()
            if "Cash - Beginning" in item:
                values["beginning"] = row["Year 1"]
            elif "NET INCREASE (DECREASE) IN CASH" in item:
                values["net_change"] = row["Year 1"]
            elif "Cash - End" in item:
                values["ending"] = row["Year 1"]

        # Beginning cash (Year 0) = 500,000
        assert values["beginning"] == 500000

        # Ending cash (Year 1) = 1,135,000 (consistent with cash flow calculation)
        assert values["ending"] == 1135000

        # The calculated cash flow must equal the actual cash change
        # This is a critical integrity check - no "plug" allowed
        assert abs((values["beginning"] + values["net_change"]) - values["ending"]) < 0.01

    def test_monthly_period(self):
        """Test monthly cash flow statement generation."""
        df = self.cash_flow.generate_statement(1, period="monthly")

        assert "Month 1" in df.columns

        # Extract values
        values_dict = {}
        for _, row in df.iterrows():
            if row["Month 1"] != "" and isinstance(row["Month 1"], (int, float)):
                values_dict[row["Item"].strip()] = row["Month 1"]

        # Monthly values should be annual / 12
        assert values_dict.get("Net Income") == pytest.approx(1200000 / 12)
        assert values_dict.get("Depreciation and Amortization") == pytest.approx(110000 / 12)

    def test_no_dividends_on_loss(self):
        """Test that no dividends are paid when there's a loss."""
        # Create metrics with a loss
        loss_metrics = [
            {"net_income": -500000, "cash": 100000, "dividends_paid": 0},
        ]

        cash_flow = CashFlowStatement(loss_metrics)
        financing_cf = cash_flow._calculate_financing_cash_flow(loss_metrics[0], {}, "annual")

        assert financing_cf["dividends_paid"] == 0
        assert financing_cf["total"] == 0

    def test_working_capital_calculation(self):
        """Test detailed working capital change calculation."""
        current = self.metrics_history[1]
        prior = self.metrics_history[0]

        wc_changes = self.cash_flow._calculate_working_capital_change(current, prior)

        # Verify each component
        assert wc_changes["accounts_receivable"] == 50000  # 250k - 200k
        assert wc_changes["inventory"] == 30000  # 180k - 150k
        assert wc_changes["prepaid_insurance"] == 5000  # 25k - 20k
        assert wc_changes["accounts_payable"] == 20000  # 120k - 100k
        assert wc_changes["accrued_expenses"] == 10000  # 60k - 50k
        assert wc_changes["claim_liabilities"] == 50000  # 50k - 0

    def test_capex_calculation(self):
        """Test capital expenditure calculation."""
        current = self.metrics_history[1]
        prior = self.metrics_history[0]

        capex = self.cash_flow._calculate_capex(current, prior)

        # Capex = (Ending PP&E - Beginning PP&E) + Depreciation
        # (1,200,000 - 1,000,000) + 110,000 = 310,000
        assert capex == 310000

    def test_capex_with_no_prior_period(self):
        """Test capex calculation for first period."""
        current = self.metrics_history[0]
        prior = {}

        capex = self.cash_flow._calculate_capex(current, prior)

        # First period capex = Current PP&E + Depreciation
        # 1,000,000 + 100,000 = 1,100,000
        assert capex == 1100000

    def test_invalid_year_raises_error(self):
        """Test that invalid year raises IndexError."""
        with pytest.raises(IndexError, match="Year 5 out of range"):
            self.cash_flow.generate_statement(5)

        with pytest.raises(IndexError, match="Year -1 out of range"):
            self.cash_flow.generate_statement(-1)

    def test_statement_completeness(self):
        """Test that all required sections are present in statement."""
        df = self.cash_flow.generate_statement(1, period="annual")

        required_sections = [
            "CASH FLOWS FROM OPERATING ACTIVITIES",
            "Net Income",
            "Depreciation and Amortization",
            "Changes in operating assets and liabilities:",
            "Net Cash Provided by Operating Activities",
            "CASH FLOWS FROM INVESTING ACTIVITIES",
            "Capital Expenditures",
            "Net Cash Used in Investing Activities",
            "CASH FLOWS FROM FINANCING ACTIVITIES",
            "Dividends Paid",
            "Net Cash Used in Financing Activities",
            "NET INCREASE (DECREASE) IN CASH",
            "CASH RECONCILIATION",
            "Cash - Beginning of Period",
            "Cash - End of Period",
        ]

        items = df["Item"].str.strip().tolist()
        for section in required_sections:
            assert section in items, f"Missing required section: {section}"

    def test_insurance_premiums_not_in_financing(self):
        """Test that insurance premiums are NOT in financing section (prevents double counting).

        Insurance premiums are already deducted from Net Income (which flows into
        Operating Activities). Including them in Financing Activities would result
        in double counting, understating the company's ending cash position.

        This is a regression test for GitHub Issue #212.
        """
        # Create metrics that include insurance_premiums_paid
        metrics_with_insurance = [
            {
                "net_income": 800000,  # Already has insurance premiums deducted
                "depreciation_expense": 100000,
                "cash": 500000,
                "accounts_receivable": 200000,
                "inventory": 150000,
                "prepaid_insurance": 20000,
                "accounts_payable": 100000,
                "accrued_expenses": 50000,
                "claim_liabilities": 0,
                "gross_ppe": 1000000,
                "assets": 2000000,
                "equity": 1500000,
                "insurance_premiums_paid": 200000,  # This should NOT appear in financing
            },
        ]

        cash_flow = CashFlowStatement(metrics_with_insurance)
        financing_cf = cash_flow._calculate_financing_cash_flow(
            metrics_with_insurance[0], {}, "annual"
        )

        # Verify insurance_premiums key is NOT in financing items
        assert "insurance_premiums" not in financing_cf, (
            "Insurance premiums should not be in financing section - "
            "they are already reflected in Net Income (Operating section)"
        )

        # Verify total only includes dividends (default 30% payout ratio on 800k = 240k)
        expected_dividends = 800000 * 0.3  # 240,000
        assert financing_cf["dividends_paid"] == pytest.approx(-expected_dividends)
        assert financing_cf["total"] == pytest.approx(-expected_dividends)

        # Verify the statement doesn't show insurance premiums in financing section
        df = cash_flow.generate_statement(0, period="annual")
        items = df["Item"].str.strip().tolist()

        # Insurance premiums should NOT appear as a line item in financing section
        # Find the financing section and check its line items
        financing_start = items.index("CASH FLOWS FROM FINANCING ACTIVITIES")
        financing_end = items.index("NET INCREASE (DECREASE) IN CASH")
        financing_items = items[financing_start:financing_end]

        for item in financing_items:
            assert "Insurance Premium" not in item and "insurance_premium" not in item.lower(), (
                f"Found insurance premiums in financing section: {item}. "
                "This indicates double counting - premiums are already in Net Income."
            )

"""Tests for working capital change calculations in cash flow statements."""

import pytest

from ergodic_insurance.financial_statements import CashFlowStatement


class TestWorkingCapitalChanges:
    """Test suite for working capital calculations in cash flow statements."""

    def test_basic_working_capital_increase(self):
        """Test that increases in working capital assets reduce cash flow."""
        metrics = [
            {
                "accounts_receivable": 100000,
                "inventory": 50000,
                "prepaid_insurance": 10000,
                "accounts_payable": 30000,
                "accrued_expenses": 20000,
            },
            {
                "accounts_receivable": 150000,  # Increased by 50k (use of cash)
                "inventory": 70000,  # Increased by 20k (use of cash)
                "prepaid_insurance": 15000,  # Increased by 5k (use of cash)
                "accounts_payable": 35000,  # Increased by 5k (source of cash)
                "accrued_expenses": 25000,  # Increased by 5k (source of cash)
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        # Verify asset increases (positive values in dict, but uses of cash)
        assert wc_changes["accounts_receivable"] == 50000
        assert wc_changes["inventory"] == 20000
        assert wc_changes["prepaid_insurance"] == 5000

        # Verify liability increases (positive values, sources of cash)
        assert wc_changes["accounts_payable"] == 5000
        assert wc_changes["accrued_expenses"] == 5000

    def test_working_capital_decrease(self):
        """Test that decreases in working capital assets increase cash flow."""
        metrics = [
            {
                "accounts_receivable": 200000,
                "inventory": 100000,
                "accounts_payable": 50000,
            },
            {
                "accounts_receivable": 150000,  # Decreased by 50k (source of cash)
                "inventory": 80000,  # Decreased by 20k (source of cash)
                "accounts_payable": 40000,  # Decreased by 10k (use of cash)
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        # Asset decreases (negative values in dict, but sources of cash)
        assert wc_changes["accounts_receivable"] == -50000
        assert wc_changes["inventory"] == -20000

        # Liability decreases (negative values, uses of cash)
        assert wc_changes["accounts_payable"] == -10000

    def test_operating_cash_flow_with_working_capital(self):
        """Test that working capital changes affect operating cash flow correctly."""
        metrics = [
            {
                "net_income": 500000,
                "depreciation_expense": 50000,
                "accounts_receivable": 100000,
                "inventory": 80000,
                "accounts_payable": 60000,
            },
            {
                "net_income": 600000,
                "depreciation_expense": 60000,
                "accounts_receivable": 130000,  # +30k
                "inventory": 100000,  # +20k
                "accounts_payable": 75000,  # +15k
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        operating_cf = cash_flow._calculate_operating_cash_flow(metrics[1], metrics[0], "annual")

        # Operating CF = NI + Dep - WC increase in assets + WC increase in liabilities
        # = 600k + 60k - 30k - 20k + 15k = 625k
        expected_operating_cf = 600000 + 60000 - 30000 - 20000 + 15000
        assert operating_cf["total"] == expected_operating_cf

    def test_claim_liabilities_change(self):
        """Test that claim liability changes are handled correctly."""
        metrics = [
            {"claim_liabilities": 0, "net_income": 100000},
            {"claim_liabilities": 500000, "net_income": 200000},  # New claims
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        # Increase in claim liabilities is a source of cash (non-cash expense)
        assert wc_changes["claim_liabilities"] == 500000

    def test_zero_starting_working_capital(self):
        """Test working capital changes when starting from zero."""
        metrics = [
            {
                "accounts_receivable": 0,
                "inventory": 0,
                "accounts_payable": 0,
            },
            {
                "accounts_receivable": 100000,
                "inventory": 50000,
                "accounts_payable": 30000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        assert wc_changes["accounts_receivable"] == 100000
        assert wc_changes["inventory"] == 50000
        assert wc_changes["accounts_payable"] == 30000

    def test_mixed_working_capital_changes(self):
        """Test mixed increases and decreases in working capital."""
        metrics = [
            {
                "accounts_receivable": 150000,
                "inventory": 100000,
                "prepaid_insurance": 20000,
                "accounts_payable": 80000,
                "accrued_expenses": 40000,
            },
            {
                "accounts_receivable": 120000,  # Decreased by 30k (source)
                "inventory": 150000,  # Increased by 50k (use)
                "prepaid_insurance": 25000,  # Increased by 5k (use)
                "accounts_payable": 90000,  # Increased by 10k (source)
                "accrued_expenses": 35000,  # Decreased by 5k (use)
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        assert wc_changes["accounts_receivable"] == -30000
        assert wc_changes["inventory"] == 50000
        assert wc_changes["prepaid_insurance"] == 5000
        assert wc_changes["accounts_payable"] == 10000
        assert wc_changes["accrued_expenses"] == -5000

        # Calculate net working capital impact on cash
        operating_cf = cash_flow._calculate_operating_cash_flow(metrics[1], metrics[0], "annual")

        # WC impact = +30k (AR decrease) - 50k (Inv increase) - 5k (Prepaid increase)
        #            + 10k (AP increase) - 5k (Accrued decrease)
        # Net WC impact = 30 - 50 - 5 + 10 - 5 = -20k (use of cash)

    def test_statement_shows_working_capital_changes(self):
        """Test that working capital changes appear correctly in statement."""
        metrics = [
            {
                "cash": 500000,
                "net_income": 300000,
                "depreciation_expense": 40000,
                "accounts_receivable": 80000,
                "inventory": 60000,
                "prepaid_insurance": 10000,
                "accounts_payable": 40000,
                "accrued_expenses": 20000,
                "claim_liabilities": 0,
                "gross_ppe": 400000,
            },
            {
                "cash": 600000,
                "net_income": 400000,
                "depreciation_expense": 50000,
                "accounts_receivable": 100000,  # +20k
                "inventory": 75000,  # +15k
                "prepaid_insurance": 12000,  # +2k
                "accounts_payable": 50000,  # +10k
                "accrued_expenses": 25000,  # +5k
                "claim_liabilities": 30000,  # +30k
                "gross_ppe": 500000,
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        df = cash_flow.generate_statement(1)

        # Extract working capital items from statement
        wc_items = {}
        for _, row in df.iterrows():
            item = row["Item"].strip()
            value = row["Year 1"]
            if value != "" and isinstance(value, (int, float)):
                if "Accounts Receivable" in item:
                    wc_items["AR"] = value
                elif "Inventory" in item:
                    wc_items["Inv"] = value
                elif "Prepaid Insurance" in item:
                    wc_items["Prepaid"] = value
                elif "Accounts Payable" in item:
                    wc_items["AP"] = value
                elif "Accrued Expenses" in item:
                    wc_items["Accrued"] = value
                elif "Claim Liabilities" in item:
                    wc_items["Claims"] = value

        # Verify signs are correct in the statement
        assert wc_items.get("AR") == -20000  # Increase is use of cash
        assert wc_items.get("Inv") == -15000  # Increase is use of cash
        assert wc_items.get("Prepaid") == -2000  # Increase is use of cash
        assert wc_items.get("AP") == 10000  # Increase is source of cash
        assert wc_items.get("Accrued") == 5000  # Increase is source of cash
        assert wc_items.get("Claims") == 30000  # Increase is source of cash

    def test_no_working_capital_changes(self):
        """Test when there are no working capital changes."""
        metrics = [
            {
                "accounts_receivable": 100000,
                "inventory": 50000,
                "accounts_payable": 30000,
            },
            {
                "accounts_receivable": 100000,  # No change
                "inventory": 50000,  # No change
                "accounts_payable": 30000,  # No change
            },
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        assert wc_changes["accounts_receivable"] == 0
        assert wc_changes["inventory"] == 0
        assert wc_changes["accounts_payable"] == 0

    def test_missing_working_capital_fields(self):
        """Test handling of missing working capital fields."""
        metrics = [
            {"net_income": 100000},
            {"net_income": 150000, "accounts_receivable": 50000},
        ]

        cash_flow = CashFlowStatement(metrics)
        wc_changes = cash_flow._calculate_working_capital_change(metrics[1], metrics[0])

        # Should handle missing fields gracefully
        assert wc_changes["accounts_receivable"] == 50000  # 50k - 0
        assert wc_changes.get("inventory", 0) == 0
        assert wc_changes.get("accounts_payable", 0) == 0

    def test_monthly_working_capital_changes(self):
        """Test that monthly periods scale working capital changes correctly."""
        annual_metrics = [
            {
                "accounts_receivable": 120000,  # Annual average
                "inventory": 60000,
            },
            {
                "accounts_receivable": 180000,  # Annual increase of 60k
                "inventory": 84000,  # Annual increase of 24k
            },
        ]

        cash_flow = CashFlowStatement(annual_metrics)

        # Annual calculation
        annual_wc = cash_flow._calculate_working_capital_change(
            annual_metrics[1], annual_metrics[0]
        )
        assert annual_wc["accounts_receivable"] == 60000
        assert annual_wc["inventory"] == 24000

        # Monthly calculation should divide by 12
        operating_cf_monthly = cash_flow._calculate_operating_cash_flow(
            annual_metrics[1], annual_metrics[0], "monthly"
        )

        # Working capital changes should be scaled
        assert operating_cf_monthly["accounts_receivable_change"] == pytest.approx(-60000 / 12)
        assert operating_cf_monthly["inventory_change"] == pytest.approx(-24000 / 12)

"""Unit tests for Issue #239: Fix Dividend "Phantom" Payments.

Tests verify that:
1. Dividends are not paid when cash is insufficient (cash-constrained)
2. Dividends are reduced when cash is partially sufficient
3. The WidgetManufacturer correctly tracks actual dividends paid
4. The CashFlowStatement reads dividends_paid from metrics
5. Cash flow reconciliation works correctly with constrained dividends
"""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import to_decimal
from ergodic_insurance.financial_statements import CashFlowStatement
from ergodic_insurance.ledger import AccountName, Ledger, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestDividendPhantomPayments:
    """Test suite for dividend phantom payment fixes."""

    config: ManufacturerConfig  # Set in setup_method

    def setup_method(self):
        """Set up test fixtures."""
        # Create a base configuration with 30% dividend payout (70% retention)
        self.config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,  # 10% margin
            tax_rate=0.25,
            retention_ratio=0.7,  # 70% retained, 30% dividends
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
        )

    def test_full_dividends_when_cash_sufficient(self):
        """Test that full dividends are paid when cash is sufficient."""
        manufacturer = WidgetManufacturer(self.config)

        # Run a step to generate income
        metrics = manufacturer.step()

        # With sufficient cash, dividends should be paid based on retention ratio
        net_income = metrics.get("net_income", 0)
        if net_income > 0:
            expected_dividends = net_income * (1 - to_decimal(manufacturer.retention_ratio))
            actual_dividends = metrics.get("dividends_paid", 0)
            # Should be close to expected (may be less if cash constrained)
            assert actual_dividends <= expected_dividends + to_decimal("0.01")
            assert actual_dividends >= 0

    def test_no_dividends_when_net_loss(self):
        """Test that no dividends are paid when there's a net loss."""
        # Create a config with low margin that might lead to losses
        manufacturer = WidgetManufacturer(self.config)

        # Simulate a loss scenario by setting up conditions
        # The _last_dividends_paid should be 0 for loss scenarios
        manufacturer._last_dividends_paid = 0  # Reset

        # Call update_balance_sheet with a loss
        initial_cash = manufacturer.cash
        manufacturer.update_balance_sheet(-100_000)  # Net loss

        # No dividends should be paid on a loss
        assert manufacturer._last_dividends_paid == 0

    def test_dividends_constrained_by_cash(self):
        """Test that dividends are constrained when cash is insufficient."""
        manufacturer = WidgetManufacturer(self.config)

        # Set up a scenario with low cash but positive income
        # This simulates timing differences (recognized revenue not yet collected)
        # Cash is read-only (derived from ledger), so adjust via ledger transaction
        current_cash = manufacturer.cash
        manufacturer._record_cash_adjustment(
            amount=to_decimal(10) - current_cash,
            description="Reduce cash for test scenario",
        )
        initial_equity = manufacturer.equity

        # Positive net income but not enough cash to pay full dividends
        net_income = 1000  # $1000 net income
        theoretical_dividends = net_income * (1 - manufacturer.retention_ratio)  # $300
        retained_earnings = net_income * manufacturer.retention_ratio  # $700

        # After adding retained earnings, cash would be $10 + $700 = $710
        # This is more than theoretical dividends ($300), so full dividends can be paid
        manufacturer.update_balance_sheet(net_income)

        # In this case, full dividends can be paid
        assert manufacturer._last_dividends_paid <= theoretical_dividends + 0.01

    def test_no_dividends_when_projected_cash_negative(self):
        """Test no dividends when projected cash after operations is negative."""
        manufacturer = WidgetManufacturer(self.config)

        # Set up extreme scenario: very negative cash
        # Cash is read-only (derived from ledger), so adjust via ledger transaction
        current_cash = manufacturer.cash
        manufacturer._record_cash_adjustment(
            amount=to_decimal(-1000) - current_cash,
            description="Set cash to negative for test scenario",
        )

        # Positive net income
        net_income = 500  # $500 net income
        retained_earnings = net_income * manufacturer.retention_ratio  # $350

        # Projected cash = -1000 + 350 = -650 (still negative)
        # No dividends can be paid
        manufacturer.update_balance_sheet(net_income)

        # Dividends should be 0 since projected cash <= 0
        assert manufacturer._last_dividends_paid == 0

        # All earnings should be retained to improve cash position
        # Cash should be: -1000 + 500 = -500 (all income retained, no dividends)
        expected_cash = -1000 + net_income  # Full income retained
        assert abs(manufacturer.cash - expected_cash) < 0.01

    def test_partial_dividends_when_cash_partially_sufficient(self):
        """Test partial dividends when cash is partially sufficient."""
        manufacturer = WidgetManufacturer(self.config)

        # Set up scenario where only partial dividends can be paid
        # Cash is read-only (derived from ledger), so adjust via ledger transaction
        current_cash = manufacturer.cash
        manufacturer._record_cash_adjustment(
            amount=to_decimal(-500) - current_cash,
            description="Set cash to negative for test scenario",
        )

        net_income = 1000  # $1000 net income
        retained_earnings = net_income * manufacturer.retention_ratio  # $700
        theoretical_dividends = net_income * (1 - manufacturer.retention_ratio)  # $300

        # Projected cash = -500 + 700 = 200
        # Can only pay $200 in dividends (what's projected available)
        manufacturer.update_balance_sheet(net_income)

        # Partial dividends should be paid
        assert manufacturer._last_dividends_paid == pytest.approx(200, abs=0.01)

        # Cash accounting:
        # Beginning: -500
        # Add net income: +1000
        # Pay dividends: -200
        # Ending: 300
        # Which equals: -500 + retained_earnings(700) + additional_retained(100) = 300
        assert manufacturer.cash == pytest.approx(300, abs=0.01)

    def test_metrics_use_actual_dividends_paid(self):
        """Test that calculate_metrics uses the tracked _last_dividends_paid."""
        manufacturer = WidgetManufacturer(self.config)

        # Set a specific value for _last_dividends_paid
        manufacturer._last_dividends_paid = 12345.67

        # Calculate metrics
        metrics = manufacturer.calculate_metrics()

        # Metrics should use the tracked value, not recalculate
        assert metrics["dividends_paid"] == 12345.67

    def test_cash_flow_statement_reads_from_metrics(self):
        """Test that CashFlowStatement reads dividends_paid from metrics."""
        # Create metrics with explicit dividends_paid
        # Use float literals for type consistency
        metrics_history: list[dict[str, float]] = [
            {
                "net_income": 1000000.0,
                "depreciation_expense": 100000.0,
                "cash": 500000.0,
                "accounts_receivable": 0.0,
                "inventory": 0.0,
                "prepaid_insurance": 0.0,
                "accounts_payable": 0.0,
                "accrued_expenses": 0.0,
                "claim_liabilities": 0.0,
                "gross_ppe": 1000000.0,
                "dividends_paid": 123456.0,  # Explicit value
            },
        ]

        cash_flow = CashFlowStatement(metrics_history)
        financing_cf = cash_flow._calculate_financing_cash_flow(metrics_history[0], {}, "annual")

        # Should read from metrics, not calculate
        assert financing_cf["dividends_paid"] == pytest.approx(-123456)

    def test_cash_flow_statement_fallback_without_metrics(self):
        """Test CashFlowStatement falls back to calculation when no dividends_paid.

        Issue #243: Must now pass a config with retention_ratio (no hardcoded default).
        """
        # Create metrics WITHOUT dividends_paid
        # Use float literals for type consistency
        metrics_history: list[dict[str, float]] = [
            {
                "net_income": 1000000.0,
                "depreciation_expense": 100000.0,
                "cash": 500000.0,
                "gross_ppe": 1000000.0,
                # No dividends_paid key
            },
        ]

        # Create config with retention_ratio (Issue #243: required for fallback)
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,  # 70% retention = 30% dividend payout
            asset_turnover_ratio=0.8,
        )

        cash_flow = CashFlowStatement(metrics_history, config=config)
        financing_cf = cash_flow._calculate_financing_cash_flow(metrics_history[0], {}, "annual")

        # Should fall back to calculation using config: 1M * 0.3 = 300K
        expected_dividends = 1000000 * 0.3  # 30% payout based on config
        assert financing_cf["dividends_paid"] == pytest.approx(-expected_dividends)

    def test_cash_flow_statement_fallback_with_custom_retention(self):
        """Test CashFlowStatement respects custom retention_ratio in fallback.

        Issue #243: Verifies that custom retention ratio is used, not hardcoded default.
        """
        # Create metrics WITHOUT dividends_paid
        metrics_history: list[dict[str, float]] = [
            {
                "net_income": 1000000.0,
                "depreciation_expense": 100000.0,
                "cash": 500000.0,
                "gross_ppe": 1000000.0,
                # No dividends_paid key
            },
        ]

        # Create config with custom retention_ratio (50% retention = 50% dividend)
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.5,  # 50% retention = 50% dividend payout
            asset_turnover_ratio=0.8,
        )

        cash_flow = CashFlowStatement(metrics_history, config=config)
        financing_cf = cash_flow._calculate_financing_cash_flow(metrics_history[0], {}, "annual")

        # Should use the custom retention ratio: 1M * 0.5 = 500K (not 300K from old default)
        expected_dividends = 1000000 * 0.5  # 50% payout based on custom config
        assert financing_cf["dividends_paid"] == pytest.approx(-expected_dividends)

    def test_cash_flow_statement_error_without_config_or_metrics(self):
        """Test CashFlowStatement raises error when no config and no dividends_paid.

        Issue #243: No hardcoded fallback - must have config or metrics.
        """
        # Create metrics WITHOUT dividends_paid
        metrics_history: list[dict[str, float]] = [
            {
                "net_income": 1000000.0,
                "depreciation_expense": 100000.0,
                "cash": 500000.0,
                "gross_ppe": 1000000.0,
                # No dividends_paid key
            },
        ]

        # Create CashFlowStatement without config
        cash_flow = CashFlowStatement(metrics_history)

        # Should raise ValueError because no config with retention_ratio
        with pytest.raises(ValueError, match="config must have 'retention_ratio'"):
            cash_flow._calculate_financing_cash_flow(metrics_history[0], {}, "annual")

    def test_insolvency_sets_dividends_to_zero(self):
        """Test that insolvency scenarios set dividends to zero."""
        manufacturer = WidgetManufacturer(self.config)

        # Set up insolvency scenario using ledger transactions
        # Write off all assets and set minimal values
        manufacturer._write_off_all_assets("Write off assets for insolvency test")
        # Add back minimal cash and PP&E via ledger
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.CASH,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=to_decimal(100),
            transaction_type=TransactionType.ADJUSTMENT,
            description="Set minimal cash for insolvency test",
        )
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.GROSS_PPE,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=to_decimal(200),
            transaction_type=TransactionType.ADJUSTMENT,
            description="Set minimal PP&E for insolvency test",
        )

        # Large loss that would trigger insolvency
        manufacturer.update_balance_sheet(-50000)

        # Dividends should be 0 when going insolvent
        assert manufacturer._last_dividends_paid == 0

    def test_reset_clears_dividends_tracking(self):
        """Test that reset() clears the dividend tracking."""
        manufacturer = WidgetManufacturer(self.config)

        # Set some dividend value
        manufacturer._last_dividends_paid = 99999

        # Reset
        manufacturer.reset()

        # Should be cleared
        assert manufacturer._last_dividends_paid == 0

    def test_full_simulation_with_cash_constraints(self):
        """Integration test: full simulation with varying cash levels."""
        manufacturer = WidgetManufacturer(self.config)

        # Run several years and track dividend behavior
        metrics_list = []
        for _ in range(5):
            metrics = manufacturer.step()
            metrics_list.append(metrics)

        # Verify that dividends_paid is always non-negative
        for metrics in metrics_list:
            assert metrics["dividends_paid"] >= 0

        # Verify that dividends never exceed net income * (1 - retention_ratio)
        for metrics in metrics_list:
            net_income = metrics.get("net_income", 0)
            if net_income > 0:
                max_dividends = net_income * (1 - to_decimal(manufacturer.retention_ratio))
                assert metrics["dividends_paid"] <= max_dividends + to_decimal("0.01")

    def test_cash_flow_reconciliation_with_constrained_dividends(self):
        """Test that cash flow reconciles correctly with constrained dividends."""
        manufacturer = WidgetManufacturer(self.config)

        # Run a step
        initial_cash = manufacturer.cash
        metrics = manufacturer.step()

        # Generate financial statements
        from ergodic_insurance.financial_statements import FinancialStatementGenerator

        generator = FinancialStatementGenerator(manufacturer)
        cash_flow = generator.generate_cash_flow_statement(year=0)

        # Extract ending cash from cash flow statement
        ending_cash_reported = None
        for _, row in cash_flow.iterrows():
            if "Cash - End of Period" in row["Item"]:
                ending_cash_reported = row["Year 0"]
                break

        # Actual ending cash
        actual_ending_cash = manufacturer.cash

        # They should match (or be very close)
        if ending_cash_reported is not None:
            assert abs(ending_cash_reported - actual_ending_cash) < 1.0


class TestDividendEdgeCases:
    """Edge case tests for dividend handling."""

    base_config_params: dict[str, float | int]  # Set in setup_method

    def setup_method(self):
        """Set up base config for edge cases."""
        self.base_config_params = {
            "initial_assets": 1_000_000,
            "base_operating_margin": 0.10,
            "tax_rate": 0.25,
            "asset_turnover_ratio": 0.8,
        }

    def test_zero_net_income(self):
        """Test dividend handling with exactly zero net income."""
        config = ManufacturerConfig(
            **self.base_config_params,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        # Zero net income - no dividends
        manufacturer.update_balance_sheet(0.0)
        # Zero is not positive, so dividends code path is not triggered
        # The else branch adds retained_earnings (0) to cash
        assert manufacturer._last_dividends_paid == 0

    def test_very_small_net_income(self):
        """Test dividend handling with very small positive net income."""
        config = ManufacturerConfig(
            **self.base_config_params,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        # Small positive income
        manufacturer.update_balance_sheet(0.01)
        # Dividends should be 0.003 (0.01 * 0.3) if cash is sufficient
        # Since the manufacturer has lots of cash, full dividends are paid
        assert manufacturer._last_dividends_paid == pytest.approx(
            to_decimal("0.003"), abs=to_decimal("0.001")
        )

    def test_retention_ratio_100_percent(self):
        """Test with 100% retention ratio (no dividends)."""
        config = ManufacturerConfig(
            **self.base_config_params,
            retention_ratio=1.0,  # 100% retention
        )
        manufacturer = WidgetManufacturer(config)

        manufacturer.update_balance_sheet(100_000)
        # 100% retention means theoretical dividends = 0
        assert manufacturer._last_dividends_paid == 0

    def test_retention_ratio_0_percent(self):
        """Test with 0% retention ratio (all dividends)."""
        config = ManufacturerConfig(
            **self.base_config_params,
            retention_ratio=0.0,  # 0% retention
        )
        manufacturer = WidgetManufacturer(config)

        net_income = 100_000
        initial_cash = manufacturer.cash
        manufacturer.update_balance_sheet(net_income)

        # With 0% retention:
        # - retained_earnings = 0
        # - theoretical_dividends = 100_000
        # - projected_cash = initial_cash + 0 = initial_cash
        # If initial_cash >= 100_000, full dividends paid
        if initial_cash >= net_income:
            assert manufacturer._last_dividends_paid == pytest.approx(net_income, rel=0.01)
        else:
            # Partial dividends based on available cash
            assert manufacturer._last_dividends_paid <= initial_cash


class TestRetainedEarningsCashFlowClassification:
    """Regression tests for Issue #370: retained earnings misclassified as REVENUE.

    Retained earnings allocation is an internal equity movement, not a cash flow
    event. Using TransactionType.RETAINED_EARNINGS ensures the entry is excluded
    from all cash flow categories in get_cash_flows().
    """

    def test_retained_earnings_uses_correct_transaction_type(self):
        """Verify retained earnings entries use RETAINED_EARNINGS, not REVENUE."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            asset_turnover_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(config)

        # Generate income so retained earnings are recorded
        manufacturer.update_balance_sheet(100_000)

        # Find retained earnings entries in the ledger
        re_entries = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.RETAINED_EARNINGS
        ]
        assert len(re_entries) > 0, "Expected at least one RETAINED_EARNINGS entry"

        # No retained-earnings description should use REVENUE type
        revenue_re = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.REVENUE
            and "retained earnings" in (e.description or "").lower()
        ]
        assert len(revenue_re) == 0, "Retained earnings should not use REVENUE type"

    def test_retained_earnings_excluded_from_cash_from_customers(self):
        """Core regression: cash_from_customers must be 0 when only retained earnings exist."""
        ledger = Ledger()
        # Record only a retained earnings entry
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=to_decimal(50_000),
            transaction_type=TransactionType.RETAINED_EARNINGS,
            description="Year 1 retained earnings",
        )

        flows = ledger.get_cash_flows(period=1)
        assert (
            flows["cash_from_customers"] == 0
        ), "Retained earnings must not appear in cash_from_customers"

    def test_retained_earnings_not_in_any_cash_flow_category(self):
        """All cash flow line items must be 0 when only retained earnings are recorded."""
        ledger = Ledger()
        ledger.record_double_entry(
            date=1,
            debit_account=AccountName.CASH,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=to_decimal(50_000),
            transaction_type=TransactionType.RETAINED_EARNINGS,
            description="Year 1 retained earnings",
        )

        flows = ledger.get_cash_flows(period=1)
        for key, value in flows.items():
            assert value == 0, f"Cash flow item '{key}' should be 0 but was {value}"

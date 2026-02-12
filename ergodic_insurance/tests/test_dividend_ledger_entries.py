"""Unit tests for Issue #683: Dividend payments recorded as separate ledger entries.

Tests verify that:
1. Dividend payments are recorded as separate TransactionType.DIVIDEND ledger entries
2. Direct-method cash flow statement shows non-zero dividends_paid when dividends are paid
3. Indirect and direct method cash flow statements produce consistent financing totals
4. _last_dividends_paid matches the sum of DIVIDEND transactions for the period
5. Run profitable period -> verify DIVIDEND transaction exists in ledger
6. Verify direct-method cash flow financing section shows dividends
"""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import to_decimal
from ergodic_insurance.financial_statements import FinancialStatementGenerator
from ergodic_insurance.ledger import AccountName, EntryType, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer

ZERO = to_decimal(0)


class TestDividendLedgerEntries:
    """Test that dividends create separate DIVIDEND ledger entries (Issue #683)."""

    config: ManufacturerConfig  # Set in setup_method

    def setup_method(self):
        """Set up test fixtures with a profitable manufacturer."""
        self.config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,  # 30% paid as dividends
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
            capex_to_depreciation_ratio=0.0,
        )

    def test_dividend_transaction_exists_in_ledger(self):
        """After a profitable period, DIVIDEND transactions must exist in ledger."""
        manufacturer = WidgetManufacturer(self.config)
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income)

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        assert (
            len(dividend_entries) > 0
        ), "Expected at least one DIVIDEND entry in ledger after profitable period"

    def test_dividend_entry_amount_matches_actual_dividends(self):
        """The DIVIDEND entry amount should match actual dividends paid."""
        manufacturer = WidgetManufacturer(self.config)
        net_income = 500_000
        expected_dividends = to_decimal(net_income) * (
            to_decimal(1) - to_decimal(self.config.retention_ratio)
        )

        manufacturer.update_balance_sheet(net_income)

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        # Sum the debit entries (Dr RETAINED_EARNINGS) for the dividend amount
        dividend_debit_sum = sum(
            e.amount for e in dividend_entries if e.entry_type == EntryType.DEBIT
        )
        assert dividend_debit_sum == expected_dividends

    def test_dividend_entry_debits_retained_earnings_credits_cash(self):
        """DIVIDEND entry should be Dr RETAINED_EARNINGS / Cr CASH."""
        manufacturer = WidgetManufacturer(self.config)
        manufacturer.update_balance_sheet(500_000)

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        debit_entries = [e for e in dividend_entries if e.entry_type == EntryType.DEBIT]
        credit_entries = [e for e in dividend_entries if e.entry_type == EntryType.CREDIT]

        assert len(debit_entries) == 1
        assert len(credit_entries) == 1
        assert debit_entries[0].account == AccountName.RETAINED_EARNINGS.value
        assert credit_entries[0].account == AccountName.CASH.value

    def test_no_dividend_entry_when_full_retention(self):
        """With 100% retention, no DIVIDEND entry should be created."""
        self.config.retention_ratio = 1.0
        manufacturer = WidgetManufacturer(self.config)
        manufacturer.update_balance_sheet(500_000)

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        assert len(dividend_entries) == 0

    def test_no_dividend_entry_on_loss(self):
        """With a net loss, no DIVIDEND entry should be created."""
        manufacturer = WidgetManufacturer(self.config)
        manufacturer.update_balance_sheet(-200_000)

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        assert len(dividend_entries) == 0

    def test_last_dividends_paid_matches_dividend_ledger_entries(self):
        """_last_dividends_paid must match the sum of DIVIDEND transactions."""
        manufacturer = WidgetManufacturer(self.config)
        manufacturer.update_balance_sheet(500_000)

        # Sum DIVIDEND credit entries on cash (outflow)
        dividend_cash_credits = sum(
            e.amount
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.DIVIDEND
            and e.account == AccountName.CASH.value
            and e.entry_type == EntryType.CREDIT
        )
        assert manufacturer._last_dividends_paid == dividend_cash_credits


class TestDirectMethodCashFlowDividends:
    """Test that direct-method cash flow shows dividends from ledger (Issue #683)."""

    config: ManufacturerConfig  # Set in setup_method

    def setup_method(self):
        self.config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
            capex_to_depreciation_ratio=0.0,
        )

    def test_direct_method_shows_nonzero_dividends(self):
        """Direct-method cash flow must show non-zero dividends when dividends are paid."""
        manufacturer = WidgetManufacturer(self.config)
        metrics = manufacturer.step()

        net_income = to_decimal(metrics.get("net_income", 0))
        if net_income <= ZERO:
            pytest.skip("Need positive net income for this test")

        # step() increments current_year after recording entries, so
        # entries are at current_year - 1
        step_year = manufacturer.current_year - 1
        flows = manufacturer.ledger.get_cash_flows(period=step_year)
        assert flows["dividends_paid"] > ZERO, (
            "Direct-method cash flow should show non-zero dividends_paid "
            "when dividends are actually paid"
        )

    def test_direct_method_dividends_match_actual(self):
        """Direct-method dividends_paid should match _last_dividends_paid."""
        manufacturer = WidgetManufacturer(self.config)
        metrics = manufacturer.step()

        net_income = to_decimal(metrics.get("net_income", 0))
        if net_income <= ZERO:
            pytest.skip("Need positive net income for this test")

        step_year = manufacturer.current_year - 1
        flows = manufacturer.ledger.get_cash_flows(period=step_year)
        assert flows["dividends_paid"] == manufacturer._last_dividends_paid

    def test_direct_method_financing_section_in_statement(self):
        """The cash flow statement's financing section should include dividends."""
        manufacturer = WidgetManufacturer(self.config)
        metrics = manufacturer.step()

        net_income = to_decimal(metrics.get("net_income", 0))
        if net_income <= ZERO:
            pytest.skip("Need positive net income for this test")

        step_year = manufacturer.current_year - 1
        generator = FinancialStatementGenerator(
            manufacturer=manufacturer,
            ledger=manufacturer.ledger,
        )

        df = generator.generate_cash_flow_statement(year=step_year, method="direct")

        # Find the Dividends Paid row
        items = df["Item"].values
        dividends_rows = [i for i, item in enumerate(items) if "Dividends Paid" in str(item)]
        assert (
            len(dividends_rows) > 0
        ), "Direct-method cash flow statement should have a 'Dividends Paid' line"

        # The value should be negative (cash outflow)
        year_col = f"Year {step_year}"
        dividends_value = df.iloc[dividends_rows[0]][year_col]
        assert (
            to_decimal(dividends_value) < ZERO
        ), "Dividends Paid should be negative (cash outflow)"


class TestIndirectDirectConsistency:
    """Test indirect and direct methods produce consistent financing totals."""

    config: ManufacturerConfig  # Set in setup_method

    def setup_method(self):
        self.config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
            capex_to_depreciation_ratio=0.0,
        )

    def test_dividend_amounts_consistent_between_methods(self):
        """Both methods should report the same dividend amount."""
        manufacturer = WidgetManufacturer(self.config)
        metrics = manufacturer.step()

        net_income = to_decimal(metrics.get("net_income", 0))
        if net_income <= ZERO:
            pytest.skip("Need positive net income for this test")

        step_year = manufacturer.current_year - 1
        generator = FinancialStatementGenerator(
            manufacturer=manufacturer,
            ledger=manufacturer.ledger,
        )

        direct_df = generator.generate_cash_flow_statement(year=step_year, method="direct")
        indirect_df = generator.generate_cash_flow_statement(year=step_year, method="indirect")

        # Extract Dividends Paid from both statements
        def get_dividends_paid(df):
            year_col = f"Year {step_year}"
            for i, item in enumerate(df["Item"].values):
                if "Dividends Paid" in str(item):
                    return to_decimal(df.iloc[i][year_col])
            return None

        direct_dividends = get_dividends_paid(direct_df)
        indirect_dividends = get_dividends_paid(indirect_df)

        assert direct_dividends is not None, "Direct method should show Dividends Paid"
        assert indirect_dividends is not None, "Indirect method should show Dividends Paid"
        assert direct_dividends == indirect_dividends, (
            f"Dividend amounts should match: direct={direct_dividends}, "
            f"indirect={indirect_dividends}"
        )


class TestDividendLedgerEdgeCases:
    """Edge cases for dividend ledger entries."""

    def test_cash_constrained_partial_dividends(self):
        """When cash is constrained, partial dividend should still be recorded."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
            capex_to_depreciation_ratio=0.0,
        )
        manufacturer = WidgetManufacturer(config)

        # Drain cash to create a constraint
        drain = manufacturer.cash - to_decimal(100)
        if drain > ZERO:
            manufacturer.ledger.record_double_entry(
                date=manufacturer.current_year,
                debit_account=AccountName.ACCOUNTS_PAYABLE,
                credit_account=AccountName.CASH,
                amount=drain,
                transaction_type=TransactionType.PAYMENT,
                description="Drain cash for test",
            )

        # Net income whose dividends would exceed available cash
        manufacturer.update_balance_sheet(1000)  # 30% = 300 dividend, but cash ~100

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        if manufacturer._last_dividends_paid > ZERO:
            assert (
                len(dividend_entries) > 0
            ), "Partial dividend should still create DIVIDEND ledger entry"
            # Dividend amount should match the constrained amount
            dividend_credit = sum(
                e.amount
                for e in dividend_entries
                if e.account == AccountName.CASH.value and e.entry_type == EntryType.CREDIT
            )
            assert dividend_credit == manufacturer._last_dividends_paid

    def test_zero_net_income_no_dividend_entry(self):
        """With zero net income, no dividend or retained earnings entry."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
            capex_to_depreciation_ratio=0.0,
        )
        manufacturer = WidgetManufacturer(config)
        initial_entry_count = len(manufacturer.ledger)

        manufacturer.update_balance_sheet(0)

        dividend_entries = [
            e for e in manufacturer.ledger.entries if e.transaction_type == TransactionType.DIVIDEND
        ]
        assert len(dividend_entries) == 0

    def test_balance_sheet_unchanged_by_split_entries(self):
        """Splitting into two entries should not change final balance sheet values."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.6,
            asset_turnover_ratio=0.8,
            insolvency_tolerance=100,
            capex_to_depreciation_ratio=0.0,
        )
        manufacturer = WidgetManufacturer(config)
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income)

        # 60% retention → retained = 300K
        retained = to_decimal(net_income) * to_decimal(0.6)
        assert manufacturer.total_assets == initial_assets + retained
        assert manufacturer.equity == initial_equity + retained

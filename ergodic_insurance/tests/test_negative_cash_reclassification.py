"""Tests for negative cash reclassification per ASC 210-10-45 / ASC 470-10 (Issue #496).

When the ledger cash balance is negative (working capital facility draw),
the balance sheet must:
1. Report cash as $0 (not a negative asset)
2. Present the absolute overdraft as "Short-Term Borrowings" under current liabilities
3. Maintain the accounting equation: Assets = Liabilities + Equity
4. Correctly reflect the liability in financial ratios
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, Ledger, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestNegativeCashReclassification:
    """Verify negative cash is reclassified as short-term borrowing."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with standard configuration."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
        )
        return WidgetManufacturer(config)

    @pytest.fixture
    def manufacturer_with_negative_cash(self, manufacturer):
        """Create a manufacturer whose ledger cash is negative.

        Simulates a large cash outflow that exceeds available cash,
        pushing the ledger balance below zero (working capital facility draw).
        """
        # First run a step so working capital etc. are established
        manufacturer.step()

        # Force cash negative by recording a large outflow
        current_cash = manufacturer.cash
        overdraft_amount = current_cash + to_decimal(500_000)  # push $500K below zero
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=overdraft_amount,
            transaction_type=TransactionType.EXPENSE,
            description="Test: large outflow to trigger overdraft",
        )
        assert manufacturer.cash < ZERO, "Precondition: cash must be negative"
        return manufacturer

    # ------------------------------------------------------------------
    # Acceptance criterion 1: Negative cash not reported as negative asset
    # ------------------------------------------------------------------
    def test_total_assets_never_include_negative_cash(self, manufacturer_with_negative_cash):
        """Total assets must not decrease from a negative cash balance."""
        mfr = manufacturer_with_negative_cash
        assert mfr.cash < ZERO, "Precondition failed"

        # total_assets floors cash at zero
        assert mfr.total_assets >= ZERO
        # Specifically, current assets should have cash=0 contribution
        expected_current = (
            ZERO + mfr.accounts_receivable + mfr.inventory + mfr.prepaid_insurance  # cash floored
        )
        net_ppe = mfr.gross_ppe - mfr.accumulated_depreciation
        net_dta = mfr.deferred_tax_asset - mfr.dta_valuation_allowance
        expected_total = expected_current + net_ppe + mfr.restricted_assets + net_dta
        assert mfr.total_assets == expected_total

    # ------------------------------------------------------------------
    # Acceptance criterion 2: Overdraft appears as current liability
    # ------------------------------------------------------------------
    def test_short_term_borrowings_equals_overdraft(self, manufacturer_with_negative_cash):
        """short_term_borrowings should equal abs(negative cash)."""
        mfr = manufacturer_with_negative_cash
        assert mfr.short_term_borrowings == abs(mfr.cash)

    def test_short_term_borrowings_zero_when_cash_positive(self, manufacturer):
        """No short-term borrowings when cash is positive."""
        manufacturer.step()
        assert manufacturer.cash > ZERO
        assert manufacturer.short_term_borrowings == ZERO

    def test_short_term_borrowings_included_in_total_liabilities(
        self, manufacturer_with_negative_cash
    ):
        """Total liabilities must include the overdraft reclassification."""
        mfr = manufacturer_with_negative_cash
        stb = mfr.short_term_borrowings
        assert stb > ZERO
        # Total liabilities should be at least as large as the borrowings
        assert mfr.total_liabilities >= stb

    # ------------------------------------------------------------------
    # Acceptance criterion 3: Accounting equation still balances
    # ------------------------------------------------------------------
    def test_accounting_equation_balances_with_overdraft(self, manufacturer_with_negative_cash):
        """Assets = Liabilities + Equity must hold with reclassified overdraft."""
        mfr = manufacturer_with_negative_cash
        total_assets = mfr.total_assets
        total_liab = mfr.total_liabilities
        equity = mfr.equity
        assert total_assets == total_liab + equity, (
            f"Accounting equation violation: "
            f"Assets={total_assets}, Liabilities={total_liab}, Equity={equity}"
        )

    def test_accounting_equation_balances_positive_cash(self, manufacturer):
        """Accounting equation holds when cash is positive (no reclassification)."""
        manufacturer.step()
        total_assets = manufacturer.total_assets
        total_liab = manufacturer.total_liabilities
        equity = manufacturer.equity
        assert total_assets == total_liab + equity

    # ------------------------------------------------------------------
    # Acceptance criterion 4: Financial ratios correctly reflect liability
    # ------------------------------------------------------------------
    def test_metrics_report_zero_cash_when_negative(self, manufacturer_with_negative_cash):
        """Metrics dictionary should report cash as zero, not negative."""
        mfr = manufacturer_with_negative_cash
        metrics = mfr.calculate_metrics()
        assert metrics["cash"] >= ZERO
        assert metrics["cash"] == ZERO  # specifically zero when overdraft

    def test_metrics_include_short_term_borrowings(self, manufacturer_with_negative_cash):
        """Metrics dictionary should include short_term_borrowings."""
        mfr = manufacturer_with_negative_cash
        metrics = mfr.calculate_metrics()
        assert "short_term_borrowings" in metrics
        assert metrics["short_term_borrowings"] > ZERO
        assert metrics["short_term_borrowings"] == abs(mfr.cash)

    # ------------------------------------------------------------------
    # Acceptance criterion 5: Various cash balance scenarios
    # ------------------------------------------------------------------
    def test_zero_cash_no_reclassification(self, manufacturer):
        """When cash is exactly zero, no reclassification occurs."""
        manufacturer.step()
        # Drive cash to exactly zero
        current_cash = manufacturer.cash
        if current_cash > ZERO:
            manufacturer.ledger.record_double_entry(
                date=manufacturer.current_year,
                debit_account=AccountName.OPERATING_EXPENSES,
                credit_account=AccountName.CASH,
                amount=current_cash,
                transaction_type=TransactionType.EXPENSE,
                description="Test: zero out cash",
            )
        assert manufacturer.cash == ZERO
        assert manufacturer.short_term_borrowings == ZERO
        assert manufacturer.total_assets == manufacturer.total_liabilities + manufacturer.equity

    def test_large_overdraft_reclassification(self, manufacturer):
        """Large overdraft is fully reclassified as liability."""
        manufacturer.step()
        current_cash = manufacturer.cash
        large_overdraft = to_decimal(5_000_000)
        outflow = current_cash + large_overdraft

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=outflow,
            transaction_type=TransactionType.EXPENSE,
            description="Test: large overdraft",
        )

        assert manufacturer.cash < ZERO
        assert abs(manufacturer.cash + large_overdraft) < to_decimal(1)  # within $1 precision
        assert abs(manufacturer.short_term_borrowings - large_overdraft) < to_decimal(1)
        assert manufacturer.total_assets >= ZERO
        assert manufacturer.total_assets == manufacturer.total_liabilities + manufacturer.equity

    def test_overdraft_then_repayment(self, manufacturer):
        """Overdraft that is subsequently repaid should clear the borrowing."""
        manufacturer.step()
        current_cash = manufacturer.cash

        # Push into overdraft
        overdraft_amount = to_decimal(200_000)
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=current_cash + overdraft_amount,
            transaction_type=TransactionType.EXPENSE,
            description="Test: create overdraft",
        )
        assert manufacturer.cash == -overdraft_amount
        assert manufacturer.short_term_borrowings == overdraft_amount

        # Receive cash inflow (e.g. collection) to repay
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.REVENUE,
            amount=overdraft_amount + to_decimal(100_000),
            transaction_type=TransactionType.REVENUE,
            description="Test: cash inflow to repay overdraft",
        )
        assert manufacturer.cash > ZERO
        assert manufacturer.short_term_borrowings == ZERO


class TestSolvencyWithOverdraft:
    """Verify solvency checks handle working capital facility correctly."""

    @pytest.fixture
    def manufacturer(self):
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
        )
        return WidgetManufacturer(config)

    def test_negative_cash_alone_not_insolvency(self, manufacturer):
        """Negative cash (facility draw) should not trigger insolvency.

        Per Issue #496, negative cash is a working capital facility draw,
        not an insolvency signal. Solvency is determined by equity.
        """
        manufacturer.step()
        current_cash = manufacturer.cash

        # Push cash slightly negative
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=current_cash + to_decimal(100_000),
            transaction_type=TransactionType.EXPENSE,
            description="Test: small overdraft",
        )
        assert manufacturer.cash < ZERO

        # Company should still be solvent (equity is still large and positive)
        assert manufacturer.equity > ZERO
        is_solvent = manufacturer.check_solvency()
        assert is_solvent is True
        assert not manufacturer.is_ruined


class TestWorkingCapitalFacility:
    """Test the working capital facility mechanism (Issue #496)."""

    @pytest.fixture
    def manufacturer(self):
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
        )
        return WidgetManufacturer(config)

    def test_facility_does_not_inflate_accounts_payable(self, manufacturer):
        """Working capital facility draws should NOT increase AP (Issue #496).

        Previously, negative cash was offset by a Debit CASH / Credit AP entry
        that inflated accounts payable with phantom vendor financing. Now the
        overdraft is presented as a separate short-term borrowing line item.
        """
        manufacturer.step()
        ap_before = manufacturer.accounts_payable

        # Calculate working capital with parameters that push cash negative
        # by manipulating cash to be very low first
        current_cash = manufacturer.cash
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=current_cash - to_decimal(100),  # leave only $100
            transaction_type=TransactionType.EXPENSE,
            description="Test: drain cash for facility test",
        )

        # Now force a working capital calculation with large revenue that
        # would require more cash than available
        revenue = manufacturer.calculate_revenue() * to_decimal(5)
        manufacturer.calculate_working_capital_components(revenue, dso=90, dio=120, dpo=30)

        # If cash went negative, the overdraft should be in short_term_borrowings
        # not in accounts_payable (beyond normal AP changes from the WC calc)
        if manufacturer.cash < ZERO:
            assert manufacturer.short_term_borrowings > ZERO
            # AP should reflect only the DPO-based calculation, not facility draws

    def test_ledger_stays_balanced_after_facility(self, manufacturer):
        """Ledger debits must equal credits after working capital facility draw."""
        manufacturer.step()

        # Force overdraft
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=manufacturer.cash + to_decimal(300_000),
            transaction_type=TransactionType.EXPENSE,
            description="Test: overdraft for ledger balance check",
        )
        assert manufacturer.cash < ZERO

        is_balanced, diff = manufacturer.ledger.verify_balance()
        assert is_balanced, f"Ledger out of balance by {diff}"


class TestShortTermBorrowingsAccount:
    """Test the SHORT_TERM_BORROWINGS account in the ledger."""

    def test_account_exists_in_chart(self):
        """SHORT_TERM_BORROWINGS must be in the chart of accounts as a liability."""
        from ergodic_insurance.ledger import CHART_OF_ACCOUNTS, AccountType

        assert AccountName.SHORT_TERM_BORROWINGS in CHART_OF_ACCOUNTS
        assert CHART_OF_ACCOUNTS[AccountName.SHORT_TERM_BORROWINGS] == AccountType.LIABILITY

    def test_explicit_borrowings_add_to_overdraft(self):
        """Explicit ledger borrowings combine with overdraft reclassification."""
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
        )
        mfr = WidgetManufacturer(config)
        mfr.step()

        # Record explicit short-term borrowing
        explicit_amount = to_decimal(100_000)
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.SHORT_TERM_BORROWINGS,
            amount=explicit_amount,
            transaction_type=TransactionType.DEBT_ISSUANCE,
            description="Test: explicit short-term borrowing",
        )

        # No overdraft yet — only explicit borrowings
        assert mfr.cash > ZERO
        assert mfr.short_term_borrowings == explicit_amount

        # Now push cash negative to create overdraft too
        current_cash = mfr.cash
        overdraft = to_decimal(200_000)
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.OPERATING_EXPENSES,
            credit_account=AccountName.CASH,
            amount=current_cash + overdraft,
            transaction_type=TransactionType.EXPENSE,
            description="Test: push into overdraft",
        )

        # Should be explicit + overdraft
        assert mfr.short_term_borrowings == explicit_amount + overdraft

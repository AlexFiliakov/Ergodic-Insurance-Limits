"""Tests for accrual-basis revenue recognition (Issue #1302).

Verifies that revenue is recorded as Dr AR / Cr SALES_REVENUE (ASC 606)
and cash collections are handled separately by the working capital module,
eliminating the double-counting where revenue credited CASH fully while
working capital also debited CASH for the AR buildup.
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, EntryType, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def manufacturer():
    """Create a manufacturer with standard config for revenue/AR tests."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.10,
        tax_rate=0.25,
        retention_ratio=1.0,
        capex_to_depreciation_ratio=0.0,
    )
    return WidgetManufacturer(config)


class TestRevenueARAccrual:
    """Verify accrual-basis revenue recognition eliminates double-counting."""

    def test_ar_balance_matches_dso_ratio_of_revenue(self, manufacturer):
        """AR balance should equal revenue * DSO/365 after step (Issue #1302)."""
        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        revenue = to_decimal(metrics["revenue"])
        dso = to_decimal(45)
        expected_ar = revenue * (dso / to_decimal(365))

        actual_ar = float(manufacturer.accounts_receivable)
        assert actual_ar == pytest.approx(
            float(expected_ar), rel=0.01
        ), f"AR ({actual_ar}) should match revenue * DSO/365 = {expected_ar}"

    def test_revenue_does_not_debit_cash(self, manufacturer):
        """Revenue entry should debit AR, not CASH (Issue #1302)."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # Find debit-side revenue entries in ledger
        revenue_debit_entries = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.REVENUE and e.entry_type == EntryType.DEBIT
        ]

        assert len(revenue_debit_entries) > 0, "Should have revenue debit entries"

        # All revenue debit entries should target AR, not CASH
        for entry in revenue_debit_entries:
            assert (
                entry.account == AccountName.ACCOUNTS_RECEIVABLE.value
            ), f"Revenue debit should target AR, not {entry.account}"
            assert (
                entry.account != AccountName.CASH.value
            ), "Revenue entry must NOT debit CASH (Issue #1302)"

    def test_collections_recorded_as_dr_cash_cr_ar(self, manufacturer):
        """Cash collections should be recorded as Dr CASH / Cr AR."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        # Find collection debit entries (should be CASH)
        collection_debits = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.COLLECTION and e.entry_type == EntryType.DEBIT
        ]
        # Find collection credit entries (should be AR)
        collection_credits = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.COLLECTION and e.entry_type == EntryType.CREDIT
        ]

        assert len(collection_debits) > 0, "Should have at least one collection debit"
        assert len(collection_credits) > 0, "Should have at least one collection credit"

        for entry in collection_debits:
            assert (
                entry.account == AccountName.CASH.value
            ), f"Collection debit should be CASH, got {entry.account}"
        for entry in collection_credits:
            assert (
                entry.account == AccountName.ACCOUNTS_RECEIVABLE.value
            ), f"Collection credit should be AR, got {entry.account}"

    def test_no_double_counting_of_cash(self, manufacturer):
        """Cash should not be double-counted via revenue + WC AR adjustment.

        The key invariant: after step(), cash change should equal
        NI + depreciation - ΔAR - ΔInv + ΔAP (indirect-method OCF),
        not NI + depreciation (which would ignore WC impacts).
        """
        initial_cash = manufacturer.cash
        initial_ar = manufacturer.accounts_receivable
        initial_inv = manufacturer.inventory
        initial_ap = manufacturer.accounts_payable

        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        ni = to_decimal(metrics["net_income"])
        dep = to_decimal(metrics["depreciation_expense"])
        delta_ar = manufacturer.accounts_receivable - initial_ar
        delta_inv = manufacturer.inventory - initial_inv
        delta_ap = manufacturer.accounts_payable - initial_ap

        # OCF (indirect method) = NI + dep - ΔAR - ΔInv + ΔAP
        expected_ocf = ni + dep - delta_ar - delta_inv + delta_ap
        actual_cash_change = manufacturer.cash - initial_cash

        # Cash change should match indirect-method OCF (before investing/financing)
        # Allow for capex, dividends, insurance, and other non-OCF items
        # The key assertion: cash change should NOT equal NI + dep
        # (which would indicate WC changes are missing from the OCF)
        #
        # In practice, with capex_to_depreciation_ratio=0 and retention_ratio=1.0,
        # the only non-OCF cash flow is capex (which is 0 here).
        assert float(actual_cash_change) == pytest.approx(float(expected_ocf), rel=0.01), (
            f"Cash change ({actual_cash_change}) should match indirect-method OCF "
            f"({expected_ocf}) = NI ({ni}) + dep ({dep}) - ΔAR ({delta_ar}) "
            f"- ΔInv ({delta_inv}) + ΔAP ({delta_ap})"
        )

    def test_collections_equal_revenue_minus_ar_change(self, manufacturer):
        """Total collections should equal period revenue minus net AR change."""
        initial_ar = manufacturer.accounts_receivable

        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        revenue = to_decimal(metrics["revenue"])
        final_ar = manufacturer.accounts_receivable
        delta_ar = final_ar - initial_ar

        # Collections = revenue - ΔAR (net cash received from customers)
        expected_collections = revenue - delta_ar

        # Sum up actual collection debit entries (Dr CASH)
        actual_collections = sum(
            e.amount
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.COLLECTION
            and e.entry_type == EntryType.DEBIT
            and e.account == AccountName.CASH.value
        )

        assert float(actual_collections) == pytest.approx(float(expected_collections), rel=0.01), (
            f"Collections ({actual_collections}) should equal "
            f"revenue ({revenue}) - ΔAR ({delta_ar}) = {expected_collections}"
        )

    def test_step_re_change_equals_net_income(self, manufacturer):
        """RE change should still equal net_income after accrual revenue fix."""
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)

        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        final_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        re_change = final_re - initial_re

        expected = to_decimal(metrics["net_income"])
        assert (
            re_change == expected
        ), f"RE change ({re_change}) should equal net_income ({expected})"

    def test_temporary_accounts_zeroed_after_step(self, manufacturer):
        """All temporary accounts should be zero after step (unchanged by fix)."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        assert manufacturer.ledger.get_balance(AccountName.SALES_REVENUE) == ZERO
        assert manufacturer.ledger.get_balance(AccountName.DEPRECIATION_EXPENSE) == ZERO
        assert manufacturer.ledger.get_balance(AccountName.COST_OF_GOODS_SOLD) == ZERO
        assert manufacturer.ledger.get_balance(AccountName.OPERATING_EXPENSES) == ZERO

    def test_monthly_ar_matches_annual_target(self, manufacturer):
        """After 12 monthly steps, AR should match the annual DSO-based target."""
        # Run 12 monthly steps
        for _ in range(12):
            metrics = manufacturer.step(time_resolution="monthly")

        # AR should approximate annual_revenue * DSO/365
        annual_revenue = to_decimal(manufacturer.config.initial_assets) * to_decimal(
            manufacturer.config.asset_turnover_ratio
        )
        dso = to_decimal(45)
        expected_ar = annual_revenue * (dso / to_decimal(365))

        assert float(manufacturer.accounts_receivable) == pytest.approx(
            float(expected_ar), rel=0.05
        ), (
            f"After 12 months, AR ({manufacturer.accounts_receivable}) should "
            f"approximate annual target ({expected_ar})"
        )

    def test_accounting_equation_holds_after_step(self, manufacturer):
        """Assets = Liabilities + Equity must hold after accrual revenue."""
        manufacturer.step(growth_rate=0.0, time_resolution="annual")

        assets = float(manufacturer.total_assets)
        liabilities = float(manufacturer.total_liabilities)
        equity = float(manufacturer.equity)

        assert assets == pytest.approx(liabilities + equity, rel=0.01), (
            f"Accounting equation violated: Assets ({assets}) != "
            f"Liabilities ({liabilities}) + Equity ({equity})"
        )

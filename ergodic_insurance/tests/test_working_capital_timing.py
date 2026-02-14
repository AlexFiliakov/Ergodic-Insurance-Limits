"""Tests for working capital timing fix (Issue #1308).

Verifies that revenue is recorded BEFORE working capital adjustments in step(),
preventing artificial negative cash balances from ordering artifacts.  The fix
moves the Dr AR / Cr SALES_REVENUE entry ahead of the working capital module
so that AR reflects the period's revenue before collections are computed.
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, EntryType, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def manufacturer():
    """Create a manufacturer with standard config for WC timing tests."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.10,
        tax_rate=0.25,
        retention_ratio=1.0,
        capex_to_depreciation_ratio=0.0,
    )
    return WidgetManufacturer(config)


@pytest.fixture
def high_growth_manufacturer():
    """Manufacturer configured for rapid growth — the scenario most likely
    to trigger artificial negative cash from WC ordering."""
    config = ManufacturerConfig(
        initial_assets=1_000_000,
        asset_turnover_ratio=2.0,
        base_operating_margin=0.10,
        tax_rate=0.25,
        retention_ratio=1.0,
        capex_to_depreciation_ratio=0.0,
    )
    return WidgetManufacturer(config)


class TestWorkingCapitalTiming:
    """Verify revenue is recorded before working capital adjustments (Issue #1308)."""

    def test_revenue_entry_precedes_wc_entries_in_ledger(self, manufacturer):
        """Revenue (Dr AR / Cr REVENUE) should appear in the ledger before
        any working capital collection or adjustment entries."""
        manufacturer.step(growth_rate=0.05, time_resolution="annual")

        entries = manufacturer.ledger.entries

        # Find first revenue entry index
        revenue_indices = [
            i for i, e in enumerate(entries) if e.transaction_type == TransactionType.REVENUE
        ]
        # Find first collection/working-capital entry index
        wc_indices = [
            i
            for i, e in enumerate(entries)
            if e.transaction_type in (TransactionType.COLLECTION, TransactionType.WORKING_CAPITAL)
        ]

        assert len(revenue_indices) > 0, "Should have revenue entries"
        assert len(wc_indices) > 0, "Should have working capital entries"

        first_revenue = min(revenue_indices)
        first_wc = min(wc_indices)

        assert first_revenue < first_wc, (
            f"Revenue entry (index {first_revenue}) must precede "
            f"working capital entry (index {first_wc}) — Issue #1308"
        )

    def test_no_negative_cash_from_wc_ordering(self, high_growth_manufacturer):
        """Cash should never go negative solely due to WC ordering artifacts.

        With the fix, revenue is posted to AR before WC adjustments, so
        the WC module sees the correct AR balance and computes collections
        without creating artificial negative-cash states.
        """
        mfg = high_growth_manufacturer

        # Run multiple steps with high growth to stress-test
        for year in range(5):
            initial_cash = mfg.cash
            metrics = mfg.step(growth_rate=0.15, time_resolution="annual")

            # After each step, cash should reflect the full period economics.
            # A negative cash balance is acceptable when the company truly
            # cannot cover its obligations, but should not arise from
            # ordering artifacts (the phantom overdraft from Issue #1308).
            #
            # With 15% growth on a profitable company, cash should stay positive.
            assert mfg.cash >= ZERO, (
                f"Year {year}: Cash went negative (${float(mfg.cash):,.2f}) — "
                f"likely an ordering artifact. Initial cash was ${float(initial_cash):,.2f}."
            )

    def test_no_negative_cash_monthly_mode(self, high_growth_manufacturer):
        """Monthly mode should also be free of WC ordering artifacts."""
        mfg = high_growth_manufacturer

        for month in range(24):
            metrics = mfg.step(growth_rate=0.10, time_resolution="monthly")

            assert mfg.cash >= ZERO, (
                f"Month {month}: Cash went negative (${float(mfg.cash):,.2f}) — "
                f"possible ordering artifact in monthly mode."
            )

    def test_ar_never_negative_during_step(self, manufacturer):
        """AR should never go negative in the ledger at any point.

        Before the fix, the WC module could collect against revenue that
        hadn't been posted yet, creating negative AR in the intermediate state.
        With revenue posted first, AR starts from old_AR + period_revenue
        and can only decrease by collections (which cap at the AR balance).
        """
        manufacturer.step(growth_rate=0.05, time_resolution="annual")

        # Walk through ledger entries and track AR balance
        ar_balance = ZERO
        for entry in manufacturer.ledger.entries:
            if entry.account == AccountName.ACCOUNTS_RECEIVABLE.value:
                if entry.entry_type == EntryType.DEBIT:
                    ar_balance += entry.amount
                else:
                    ar_balance -= entry.amount

            # AR should never be negative
            if entry.account == AccountName.ACCOUNTS_RECEIVABLE.value:
                assert ar_balance >= ZERO, (
                    f"AR went negative ({float(ar_balance):,.2f}) after entry: "
                    f"{entry.entry_type.value} {entry.account} ${float(entry.amount):,.2f} "
                    f"({entry.transaction_type.value}) — Issue #1308"
                )

    def test_collections_formula_still_correct(self, manufacturer):
        """Collections = period_revenue - ΔAR should still hold after reorder.

        The fix changes WHEN revenue is posted but not the economic result.
        Total collections should still equal revenue minus the change in AR.
        """
        initial_ar = manufacturer.accounts_receivable

        metrics = manufacturer.step(growth_rate=0.0, time_resolution="annual")

        revenue = to_decimal(metrics["revenue"])
        delta_ar = manufacturer.accounts_receivable - initial_ar

        expected_collections = revenue - delta_ar

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

    def test_accounting_equation_holds_after_reorder(self, manufacturer):
        """Assets = Liabilities + Equity must still hold after the reorder."""
        manufacturer.step(growth_rate=0.05, time_resolution="annual")

        assets = float(manufacturer.total_assets)
        liabilities = float(manufacturer.total_liabilities)
        equity = float(manufacturer.equity)

        assert assets == pytest.approx(liabilities + equity, rel=0.01), (
            f"Accounting equation violated: Assets ({assets}) != "
            f"Liabilities ({liabilities}) + Equity ({equity})"
        )

    def test_cash_flow_reconciliation_after_reorder(self, manufacturer):
        """Cash change = NI + dep - ΔAR - ΔInv + ΔAP (indirect-method OCF)."""
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

        expected_ocf = ni + dep - delta_ar - delta_inv + delta_ap
        actual_cash_change = manufacturer.cash - initial_cash

        assert float(actual_cash_change) == pytest.approx(float(expected_ocf), rel=0.01), (
            f"Cash change ({actual_cash_change}) should match indirect-method OCF "
            f"({expected_ocf}) = NI ({ni}) + dep ({dep}) - ΔAR ({delta_ar}) "
            f"- ΔInv ({delta_inv}) + ΔAP ({delta_ap})"
        )

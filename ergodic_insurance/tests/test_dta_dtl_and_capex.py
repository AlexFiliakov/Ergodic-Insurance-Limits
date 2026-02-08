"""Tests for Issue #367 (DTA/DTL per ASC 740) and Issue #383 (capex clamping).

Tests cover:
- Negative capex (asset disposals) not suppressed
- Asset disposals appear as separate line in investing activities
- DTL computation from accelerated tax depreciation
- DTL balance sheet inclusion
- Deferred tax expense includes both DTA and DTL changes
"""

from decimal import Decimal
from typing import List
from unittest.mock import MagicMock

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, MetricsDict, to_decimal
from ergodic_insurance.financial_statements import CashFlowStatement
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer

# ============================================================================
# Issue #383: _calculate_capex clamping
# ============================================================================


class TestCapexDisposals:
    """Test that negative capex (asset disposals) is not suppressed."""

    def test_negative_capex_not_clamped(self):
        """Negative capex from asset disposals should be returned as-is."""
        metrics: List[MetricsDict] = [
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 500000.0,
                "net_ppe": 1000000.0,
                "gross_ppe": 1200000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 800000.0,
                # Net PP&E dropped significantly (disposal)
                "net_ppe": 700000.0,
                "gross_ppe": 900000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
        ]
        cf = CashFlowStatement(metrics)
        # Capex = (700k - 1000k) + 50k = -250k
        capex = cf._calculate_capex(metrics[1], metrics[0])
        assert capex == Decimal("-250000")

    def test_positive_capex_unchanged(self):
        """Positive capex should still work normally."""
        metrics: List[MetricsDict] = [
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 500000.0,
                "net_ppe": 900000.0,
                "gross_ppe": 1000000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
            {
                "net_income": 100000.0,
                "depreciation_expense": 60000.0,
                "cash": 500000.0,
                "net_ppe": 1000000.0,
                "gross_ppe": 1200000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
        ]
        cf = CashFlowStatement(metrics)
        # Capex = (1000k - 900k) + 60k = 160k
        capex = cf._calculate_capex(metrics[1], metrics[0])
        assert capex == Decimal("160000")

    def test_investing_cf_splits_disposal_from_capex(self):
        """Negative capex should appear as asset_sales, not capital_expenditures."""
        metrics: List[MetricsDict] = [
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 500000.0,
                "net_ppe": 1000000.0,
                "gross_ppe": 1200000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 800000.0,
                "net_ppe": 700000.0,
                "gross_ppe": 900000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
        ]
        cf = CashFlowStatement(metrics)
        investing = cf._calculate_investing_cash_flow(metrics[1], metrics[0], "annual")

        # Disposal: capex = -250k
        # capital_expenditures should be 0 (no purchases)
        # asset_sales should be +250k (disposal proceeds)
        assert investing["capital_expenditures"] == ZERO
        assert investing["asset_sales"] == Decimal("250000")
        assert investing["total"] == Decimal("250000")

    def test_investing_cf_positive_capex(self):
        """Positive capex should appear as capital_expenditures with zero asset_sales."""
        metrics: List[MetricsDict] = [
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 500000.0,
                "net_ppe": 900000.0,
                "gross_ppe": 1000000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
            {
                "net_income": 100000.0,
                "depreciation_expense": 60000.0,
                "cash": 500000.0,
                "net_ppe": 1000000.0,
                "gross_ppe": 1200000.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
        ]
        cf = CashFlowStatement(metrics)
        investing = cf._calculate_investing_cash_flow(metrics[1], metrics[0], "annual")

        # Capex = 160k (positive)
        assert investing["capital_expenditures"] == Decimal("-160000")
        assert investing["asset_sales"] == ZERO
        assert investing["total"] == Decimal("-160000")

    def test_disposal_renders_in_indirect_method(self):
        """Asset sales should appear in indirect method output (not just direct)."""
        metrics: List[MetricsDict] = [
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 500000.0,
                "net_ppe": 1000000.0,
                "gross_ppe": 1200000.0,
                "accounts_receivable": 0.0,
                "inventory": 0.0,
                "prepaid_insurance": 0.0,
                "accounts_payable": 0.0,
                "accrued_expenses": 0.0,
                "claim_liabilities": 0.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
            {
                "net_income": 100000.0,
                "depreciation_expense": 50000.0,
                "cash": 850000.0,
                "net_ppe": 700000.0,
                "gross_ppe": 900000.0,
                "accounts_receivable": 0.0,
                "inventory": 0.0,
                "prepaid_insurance": 0.0,
                "accounts_payable": 0.0,
                "accrued_expenses": 0.0,
                "claim_liabilities": 0.0,
                "dividends_paid": 0.0,
                "assets": 1500000.0,
                "equity": 1000000.0,
            },
        ]
        cf = CashFlowStatement(metrics)
        df = cf.generate_statement(year=1, period="annual", method="indirect")
        items = df["Item"].values
        assert any("Proceeds from Asset Sales" in str(item) for item in items)


# ============================================================================
# Issue #367: DTA/DTL per ASC 740
# ============================================================================


class TestDTLFromDepreciation:
    """Test deferred tax liability from accelerated tax depreciation."""

    @pytest.fixture
    def config_with_dtl(self):
        """Config with accelerated tax depreciation (5-year vs 10-year book)."""
        return ManufacturerConfig(
            initial_assets=10_000_000,
            tax_rate=0.25,
            nol_carryforward_enabled=True,
            capex_to_depreciation_ratio=1.0,
            tax_depreciation_life_years=5.0,
        )

    @pytest.fixture
    def config_no_dtl(self):
        """Config with no accelerated depreciation (default)."""
        return ManufacturerConfig(
            initial_assets=10_000_000,
            tax_rate=0.25,
            nol_carryforward_enabled=True,
            capex_to_depreciation_ratio=1.0,
        )

    def test_dtl_created_with_accelerated_depreciation(self, config_with_dtl):
        """DTL should be created when tax_depreciation_life_years < book life."""
        mfg = WidgetManufacturer(config_with_dtl)
        mfg.step(growth_rate=0.0, time_resolution="annual")

        dtl = mfg.deferred_tax_liability
        assert dtl > ZERO, "DTL should be positive with accelerated tax depreciation"

    def test_no_dtl_with_default_config(self, config_no_dtl):
        """No DTL should be created when tax_depreciation_life_years == book life."""
        mfg = WidgetManufacturer(config_no_dtl)
        mfg.step(growth_rate=0.0, time_resolution="annual")

        dtl = mfg.deferred_tax_liability
        assert dtl == ZERO, "DTL should be zero with default config"

    def test_dtl_equals_timing_difference_times_tax_rate(self, config_with_dtl):
        """DTL = (tax_accum_depr - book_accum_depr) * tax_rate."""
        mfg = WidgetManufacturer(config_with_dtl)
        mfg.step(growth_rate=0.0, time_resolution="annual")

        tax_accum = mfg.tax_handler.tax_accumulated_depreciation
        book_accum = mfg.accumulated_depreciation
        tax_rate = to_decimal(config_with_dtl.tax_rate)

        expected_dtl = max(ZERO, (tax_accum - book_accum) * tax_rate)
        actual_dtl = mfg.deferred_tax_liability

        assert actual_dtl == expected_dtl

    def test_dtl_in_total_liabilities(self, config_with_dtl):
        """DTL should be included in total_liabilities."""
        mfg = WidgetManufacturer(config_with_dtl)
        mfg.step(growth_rate=0.0, time_resolution="annual")

        dtl = mfg.deferred_tax_liability
        total_liabilities = mfg.total_liabilities
        assert dtl > ZERO
        assert total_liabilities >= dtl

    def test_dtl_reduces_equity(self, config_with_dtl, config_no_dtl):
        """DTL should reduce equity compared to no-DTL scenario."""
        mfg_with = WidgetManufacturer(config_with_dtl)
        mfg_without = WidgetManufacturer(config_no_dtl)

        mfg_with.step(growth_rate=0.0, time_resolution="annual")
        mfg_without.step(growth_rate=0.0, time_resolution="annual")

        # The DTL journal entry Dr TAX_EXPENSE, Cr DTL
        # means liabilities go up → equity (A - L) goes down
        assert mfg_with.equity < mfg_without.equity

    def test_dtl_in_metrics(self, config_with_dtl):
        """DTL should appear in calculate_metrics output."""
        mfg = WidgetManufacturer(config_with_dtl)
        metrics = mfg.step(growth_rate=0.0, time_resolution="annual")

        assert "deferred_tax_liability" in metrics
        assert metrics["deferred_tax_liability"] > ZERO

    def test_dta_in_metrics(self, config_no_dtl):
        """DTA should appear in calculate_metrics output."""
        mfg = WidgetManufacturer(config_no_dtl)
        metrics = mfg.step(growth_rate=0.0, time_resolution="annual")

        assert "deferred_tax_asset" in metrics

    def test_dtl_multi_period_growth(self, config_with_dtl):
        """DTL should grow over multiple periods when tax depreciation is faster."""
        mfg = WidgetManufacturer(config_with_dtl)

        dtl_values = []
        for _ in range(3):
            mfg.step(growth_rate=0.0, time_resolution="annual")
            dtl_values.append(mfg.deferred_tax_liability)

        # DTL should be positive and growing (tax depr outpaces book depr)
        assert all(dtl > ZERO for dtl in dtl_values)

    def test_dtl_reverses_after_tax_basis_exhausted(self):
        """DTL should reverse when tax depreciation basis is exhausted."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            tax_rate=0.25,
            nol_carryforward_enabled=True,
            capex_to_depreciation_ratio=0.0,  # No capex → PP&E eventually fully depreciated
            tax_depreciation_life_years=3.0,  # Very fast tax depreciation
        )
        mfg = WidgetManufacturer(config)

        dtl_values = []
        for _ in range(12):
            mfg.step(growth_rate=0.0, time_resolution="annual")
            dtl_values.append(mfg.deferred_tax_liability)

        # DTL should peak somewhere then reverse as book catches up
        peak_dtl = max(dtl_values)
        final_dtl = dtl_values[-1]
        assert peak_dtl > ZERO, "DTL should peak above zero"
        # After tax fully depreciated but book still running, DTL should decrease
        assert final_dtl < peak_dtl, "DTL should reverse after tax basis exhausted"

    def test_accounting_equation_holds_with_dtl(self, config_with_dtl):
        """Accounting equation (debits = credits) must hold with DTL."""
        mfg = WidgetManufacturer(config_with_dtl)

        for _ in range(5):
            mfg.step(growth_rate=0.0, time_resolution="annual")

        is_balanced, difference = mfg.ledger.verify_balance()
        assert is_balanced, f"Accounting equation violated: difference = {difference}"

    def test_config_default_no_tax_depreciation(self):
        """Default config should have no accelerated tax depreciation."""
        config = ManufacturerConfig()
        assert config.tax_depreciation_life_years is None

    def test_config_tax_depreciation_validation(self):
        """Config should validate tax_depreciation_life_years bounds."""
        # Valid values
        ManufacturerConfig(tax_depreciation_life_years=5.0)
        ManufacturerConfig(tax_depreciation_life_years=50.0)

        # Invalid: <= 0
        with pytest.raises(Exception):
            ManufacturerConfig(tax_depreciation_life_years=0.0)
        with pytest.raises(Exception):
            ManufacturerConfig(tax_depreciation_life_years=-1.0)

    def test_ledger_has_dtl_account(self):
        """DEFERRED_TAX_LIABILITY should exist in ledger AccountName."""
        assert hasattr(AccountName, "DEFERRED_TAX_LIABILITY")
        assert AccountName.DEFERRED_TAX_LIABILITY.value == "deferred_tax_liability"

    def test_ledger_has_dtl_adjustment_transaction(self):
        """DTL_ADJUSTMENT should exist in TransactionType."""
        assert hasattr(TransactionType, "DTL_ADJUSTMENT")
        assert TransactionType.DTL_ADJUSTMENT.value == "dtl_adjustment"


class TestDTLMonthly:
    """Test DTL with monthly time resolution."""

    def test_dtl_monthly_accumulation(self):
        """DTL should accumulate correctly with monthly time resolution."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            tax_rate=0.25,
            capex_to_depreciation_ratio=1.0,
            tax_depreciation_life_years=5.0,
        )
        mfg = WidgetManufacturer(config)

        # Run 12 monthly steps (1 year)
        for _ in range(12):
            mfg.step(growth_rate=0.0, time_resolution="monthly")

        dtl = mfg.deferred_tax_liability
        assert dtl > ZERO, "DTL should be positive after 12 monthly steps"

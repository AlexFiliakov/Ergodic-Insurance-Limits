"""Additional coverage tests for financial_statements.py.

Targets specific uncovered lines to improve coverage from 88.4% toward 100%.
Focuses on: direct method cash flow, ledger-based metrics, dividend calculation
edge cases, monthly period handling, comparison year methods, Monte Carlo
aggregator stubs, and error paths.
"""

from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, Mock, patch
import warnings

import pandas as pd
import pytest

from ergodic_insurance.config import ExpenseRatioConfig, ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.financial_statements import (
    CashFlowStatement,
    FinancialStatementConfig,
    FinancialStatementGenerator,
    MonteCarloStatementAggregator,
)

# ---------------------------------------------------------------------------
# Helper: build a complete metrics dict with COGS/SG&A breakdown
# ---------------------------------------------------------------------------


def _full_metrics(overrides: Optional[dict] = None) -> dict:
    """Return a complete metrics dictionary with all required fields."""
    base = {
        "year": 0,
        "assets": 10_000_000,
        "equity": 10_000_000,
        "revenue": 5_000_000,
        "operating_income": 400_000,
        "net_income": 300_000,
        "collateral": 0,
        "restricted_assets": 0,
        "available_assets": 10_000_000,
        "net_assets": 10_000_000,
        "claim_liabilities": 0,
        "accounts_payable": 0,
        "accrued_expenses": 0,
        "is_solvent": True,
        "base_operating_margin": 0.08,
        "roe": 0.03,
        "roa": 0.03,
        "asset_turnover": 0.5,
        "gross_ppe": 7_000_000,
        "net_ppe": 7_000_000,
        "accumulated_depreciation": 0,
        "depreciation_expense": 700_000,
        "cash": 3_000_000,
        "insurance_premiums": 0,
        "insurance_losses": 0,
        "total_insurance_costs": 0,
        "dividends_paid": 0,
        # COGS breakdown (Issue #255)
        "direct_materials": 1_504_000,
        "direct_labor": 1_128_000,
        "manufacturing_overhead": 1_128_000,
        "mfg_depreciation": 490_000,
        "total_cogs": 4_250_000,
        # SG&A breakdown
        "selling_expenses": 56_000,
        "general_admin_expenses": 84_000,
        "admin_depreciation": 210_000,
        "total_sga": 350_000,
        "gross_margin_ratio": 0.15,
        "sga_expense_ratio": 0.07,
    }
    if overrides:
        base.update(overrides)
    return base


def _make_manufacturer_mock(*metrics_list):
    """Create a mock manufacturer with provided metrics history."""
    manufacturer = Mock()
    manufacturer.config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        retention_ratio=0.6,
        base_operating_margin=0.08,
        tax_rate=0.25,
    )
    manufacturer.metrics_history = list(metrics_list)
    manufacturer.ledger = None
    return manufacturer


# ===========================================================================
# CashFlowStatement: missing depreciation_expense error  (line 182, 296)
# ===========================================================================


class TestCashFlowStatementMissingDepreciation:
    """Test error when depreciation_expense is missing from metrics."""

    def test_operating_cf_missing_depreciation_raises(self):
        """Line 182: depreciation_expense missing in operating CF."""
        metrics = [{"net_income": 100_000, "cash": 50_000}]  # no depreciation_expense
        cf = CashFlowStatement(metrics)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="depreciation_expense missing"):
            cf.generate_statement(year=0)

    def test_capex_missing_depreciation_raises(self):
        """Line 296: depreciation_expense missing in capex calculation."""
        cf = CashFlowStatement([{"net_income": 100_000}])
        with pytest.raises(ValueError, match="depreciation_expense missing"):
            cf._calculate_capex({"net_ppe": 100}, {"net_ppe": 50})


# ===========================================================================
# CashFlowStatement: dividends edge cases (line 370, 401)
# ===========================================================================


class TestCashFlowStatementDividends:
    """Test dividend calculation edge cases."""

    def test_dividends_negative_income_returns_zero(self):
        """Line 370: net_income <= 0 should return ZERO dividends."""
        config_obj = Mock()
        config_obj.retention_ratio = 0.7
        cf = CashFlowStatement([{"net_income": -100_000}], config=config_obj)
        result = cf._calculate_dividends({"net_income": -100_000})
        assert result == ZERO

    def test_dividends_zero_income_returns_zero(self):
        """Line 370: net_income == 0 should return ZERO dividends."""
        config_obj = Mock()
        config_obj.retention_ratio = 0.7
        cf = CashFlowStatement([{"net_income": 0}], config=config_obj)
        result = cf._calculate_dividends({"net_income": 0})
        assert result == ZERO

    def test_dividends_no_dividends_paid_no_config_raises(self):
        """Line 378-382: missing dividends_paid and no config.retention_ratio."""
        cf = CashFlowStatement([{"net_income": 100_000}], config=None)
        with pytest.raises(ValueError, match="Cannot calculate dividends"):
            cf._calculate_dividends({"net_income": 100_000})

    def test_dividends_with_config_retention_ratio(self):
        """Line 374-384: config with retention_ratio calculates dividends."""
        config_obj = Mock()
        config_obj.retention_ratio = 0.7
        cf = CashFlowStatement([{"net_income": 100_000}], config=config_obj)
        result = cf._calculate_dividends({"net_income": 100_000})
        # dividends = 100_000 * (1 - 0.7) = 30_000
        expected = to_decimal(100_000) * (to_decimal(1) - to_decimal(0.7))
        assert abs(result - expected) < Decimal("0.01")


# ===========================================================================
# CashFlowStatement: direct method ledger None checks (401, 434, 461)
# ===========================================================================


class TestCashFlowDirectMethodLedgerNone:
    """Test direct method raises when ledger is None."""

    def test_operating_cf_direct_no_ledger(self):
        """Line 401: _calculate_operating_cash_flow_direct with no ledger."""
        cf = CashFlowStatement([_full_metrics()], ledger=None)
        with pytest.raises(ValueError, match="Direct method requires a ledger"):
            cf._calculate_operating_cash_flow_direct(0, "annual")

    def test_investing_cf_direct_no_ledger(self):
        """Line 434: _calculate_investing_cash_flow_direct with no ledger."""
        cf = CashFlowStatement([_full_metrics()], ledger=None)
        with pytest.raises(ValueError, match="Direct method requires a ledger"):
            cf._calculate_investing_cash_flow_direct(0, "annual")

    def test_financing_cf_direct_no_ledger(self):
        """Line 461: _calculate_financing_cash_flow_direct with no ledger."""
        cf = CashFlowStatement([_full_metrics()], ledger=None)
        with pytest.raises(ValueError, match="Direct method requires a ledger"):
            cf._calculate_financing_cash_flow_direct(0, "annual")


# ===========================================================================
# CashFlowStatement: direct method monthly period (419, 446, 473)
# ===========================================================================


class TestCashFlowDirectMethodMonthly:
    """Test direct method with monthly period divisions."""

    @pytest.fixture
    def mock_ledger(self):
        """Create a mock ledger that returns representative cash flows."""
        ledger = Mock()
        ledger.get_cash_flows.return_value = {
            "cash_from_customers": Decimal("1_200_000"),
            "cash_from_insurance": Decimal("100_000"),
            "cash_to_suppliers": Decimal("600_000"),
            "cash_for_insurance": Decimal("50_000"),
            "cash_for_taxes": Decimal("80_000"),
            "cash_for_wages": Decimal("200_000"),
            "cash_for_interest": Decimal("30_000"),
            "capital_expenditures": Decimal("500_000"),
            "asset_sales": Decimal("50_000"),
            "dividends_paid": Decimal("120_000"),
            "equity_issuance": Decimal("200_000"),
        }
        return ledger

    def test_operating_cf_direct_monthly(self, mock_ledger):
        """Line 419: monthly division in operating CF direct."""
        cf = CashFlowStatement([_full_metrics()], ledger=mock_ledger)
        result = cf._calculate_operating_cash_flow_direct(0, "monthly")
        # All values should be 1/12 of annual
        assert result["cash_from_customers"] == to_decimal(1_200_000) / 12

    def test_investing_cf_direct_monthly(self, mock_ledger):
        """Line 446: monthly division in investing CF direct."""
        cf = CashFlowStatement([_full_metrics()], ledger=mock_ledger)
        result = cf._calculate_investing_cash_flow_direct(0, "monthly")
        # capital_expenditures should be negative and divided by 12
        assert result["capital_expenditures"] == -to_decimal(500_000) / 12

    def test_financing_cf_direct_monthly(self, mock_ledger):
        """Line 473: monthly division in financing CF direct."""
        cf = CashFlowStatement([_full_metrics()], ledger=mock_ledger)
        result = cf._calculate_financing_cash_flow_direct(0, "monthly")
        assert result["dividends_paid"] == -to_decimal(120_000) / 12


# ===========================================================================
# CashFlowStatement: direct method format_statement lines (515-531, 576, 587)
# ===========================================================================


class TestCashFlowDirectMethodFormatting:
    """Test direct method formatting covers all cash flow line items."""

    @pytest.fixture
    def mock_ledger_with_all_flows(self):
        """Ledger returning all possible cash flow categories (non-zero)."""
        ledger = Mock()
        ledger.get_cash_flows.return_value = {
            "cash_from_customers": Decimal("1_200_000"),
            "cash_from_insurance": Decimal("50_000"),
            "cash_to_suppliers": Decimal("400_000"),
            "cash_for_insurance": Decimal("30_000"),
            "cash_for_taxes": Decimal("60_000"),
            "cash_for_wages": Decimal("200_000"),
            "cash_for_interest": Decimal("10_000"),
            "capital_expenditures": Decimal("500_000"),
            "asset_sales": Decimal("75_000"),
            "dividends_paid": Decimal("100_000"),
            "equity_issuance": Decimal("250_000"),
        }
        return ledger

    def test_direct_method_shows_all_operating_items(self, mock_ledger_with_all_flows):
        """Lines 515, 523, 527, 529, 531: all operating items present in direct output."""
        metrics = [_full_metrics()]
        cf = CashFlowStatement(metrics, ledger=mock_ledger_with_all_flows)
        df = cf.generate_statement(year=0, method="direct")
        items = df["Item"].values

        expected_items = [
            "Cash Received from Insurance",
            "Cash Paid for Insurance",
            "Cash Paid for Taxes",
            "Cash Paid for Wages",
            "Cash Paid for Interest",
        ]
        for expected in expected_items:
            assert any(
                expected in str(item) for item in items
            ), f"Missing '{expected}' in direct method output"

    def test_direct_method_shows_asset_sales(self, mock_ledger_with_all_flows):
        """Line 576: Proceeds from Asset Sales in direct method."""
        metrics = [_full_metrics()]
        cf = CashFlowStatement(metrics, ledger=mock_ledger_with_all_flows)
        df = cf.generate_statement(year=0, method="direct")
        items = df["Item"].values
        assert any("Proceeds from Asset Sales" in str(item) for item in items)

    def test_direct_method_shows_equity_issuance(self, mock_ledger_with_all_flows):
        """Line 587: Proceeds from Equity Issuance in direct method."""
        metrics = [_full_metrics()]
        cf = CashFlowStatement(metrics, ledger=mock_ledger_with_all_flows)
        df = cf.generate_statement(year=0, method="direct")
        items = df["Item"].values
        assert any("Proceeds from Equity Issuance" in str(item) for item in items)


# ===========================================================================
# FinancialStatementGenerator: _get_metrics_from_ledger (lines 727, 745,
#   781-788, 819, 830-835, 878-910)
# ===========================================================================


class TestGetMetricsFromLedger:
    """Test _get_metrics_from_ledger fallback paths."""

    @pytest.fixture
    def mock_ledger_for_metrics(self):
        """Create a mock ledger returning realistic balances."""
        ledger = Mock()

        def mock_get_balance(account, year):
            """Return balance for each account."""
            balances = {
                "cash": Decimal("3_000_000"),
                "accounts_receivable": Decimal("200_000"),
                "inventory": Decimal("500_000"),
                "prepaid_insurance": Decimal("50_000"),
                "insurance_receivables": Decimal("10_000"),
                "gross_ppe": Decimal("7_000_000"),
                "accumulated_depreciation": Decimal("-700_000"),
                "restricted_cash": Decimal("100_000"),
                "collateral": Decimal("50_000"),
                "accounts_payable": Decimal("150_000"),
                "accrued_expenses": Decimal("80_000"),
                "accrued_wages": Decimal("30_000"),
                "accrued_taxes": Decimal("20_000"),
                "accrued_interest": Decimal("10_000"),
                "unearned_revenue": Decimal("5_000"),
                "claim_liabilities": Decimal("200_000"),
                "retained_earnings": Decimal("5_000_000"),
                "revenue": Decimal("10_000_000"),
                "depreciation_expense": Decimal("700_000"),
                "dividends": Decimal("-100_000"),
            }
            return balances.get(account, Decimal("0"))

        ledger.get_balance = mock_get_balance
        ledger.get_cash_flows = Mock(return_value={})
        ledger.get_period_change = Mock(return_value=Decimal("0"))

        return ledger

    def test_get_metrics_from_ledger_no_ledger(self):
        """Line 727: Raises when called without ledger."""
        mfr = _make_manufacturer_mock(_full_metrics())
        gen = FinancialStatementGenerator(manufacturer=mfr)
        with pytest.raises(ValueError, match="Ledger is required"):
            gen._get_metrics_from_ledger(0)

    def test_get_metrics_from_ledger_cash_fallback_to_ledger(self, mock_ledger_for_metrics):
        """Line 745: cash falls back to ledger when mfr_metrics has no cash."""
        metrics_no_cash = _full_metrics()
        del metrics_no_cash["cash"]
        mfr = _make_manufacturer_mock(metrics_no_cash)
        gen = FinancialStatementGenerator(manufacturer=mfr, ledger=mock_ledger_for_metrics)
        result = gen._get_metrics_from_ledger(0)
        assert result["cash"] == Decimal("3_000_000")

    def test_get_metrics_from_ledger_assets_computed_from_ledger(self, mock_ledger_for_metrics):
        """Lines 781-788: assets computed from ledger when mfr_metrics has no assets."""
        metrics_no_assets = _full_metrics()
        del metrics_no_assets["assets"]
        mfr = _make_manufacturer_mock(metrics_no_assets)
        gen = FinancialStatementGenerator(manufacturer=mfr, ledger=mock_ledger_for_metrics)
        result = gen._get_metrics_from_ledger(0)
        # assets should be computed from current_assets + net_ppe + restricted_assets
        assert result["assets"] > 0

    def test_get_metrics_from_ledger_negative_claim_liabilities(self, mock_ledger_for_metrics):
        """Line 819: negative claim_liabilities falls back to mfr_metrics."""
        # Override the ledger to return negative claim_liabilities
        original_get_balance = mock_ledger_for_metrics.get_balance

        def negative_claims_balance(account, year):
            if account == "claim_liabilities":
                return Decimal("-50_000")
            return original_get_balance(account, year)

        mock_ledger_for_metrics.get_balance = negative_claims_balance

        metrics = _full_metrics({"claim_liabilities": 200_000})
        mfr = _make_manufacturer_mock(metrics)
        gen = FinancialStatementGenerator(manufacturer=mfr, ledger=mock_ledger_for_metrics)
        result = gen._get_metrics_from_ledger(0)
        assert result["claim_liabilities"] == 200_000

    def test_get_metrics_from_ledger_equity_fallback(self, mock_ledger_for_metrics):
        """Lines 830-835: equity computed from ledger when mfr_metrics has no equity."""
        metrics_no_equity = _full_metrics()
        del metrics_no_equity["equity"]
        mfr = _make_manufacturer_mock(metrics_no_equity)
        gen = FinancialStatementGenerator(manufacturer=mfr, ledger=mock_ledger_for_metrics)
        result = gen._get_metrics_from_ledger(0)
        # equity should be computed from retained_earnings + revenue - depreciation - dividends
        assert "equity" in result

    def test_get_metrics_from_ledger_no_mfr_metrics_fallback(self, mock_ledger_for_metrics):
        """Lines 878-910: all income statement items from ledger when no mfr_metrics."""
        # Empty metrics_history so mfr_metrics is None for year=0
        mfr = _make_manufacturer_mock()
        mfr.metrics_history = []

        # Setup period_change to return values for income statement items
        def mock_period_change(account, year):
            changes = {
                "revenue": Decimal("5_000_000"),
                "depreciation_expense": Decimal("700_000"),
                "cost_of_goods_sold": Decimal("3_000_000"),
                "operating_expenses": Decimal("500_000"),
                "wage_expense": Decimal("300_000"),
                "insurance_expense": Decimal("100_000"),
                "insurance_loss": Decimal("50_000"),
                "tax_expense": Decimal("75_000"),
                "interest_expense": Decimal("20_000"),
                "collateral_expense": Decimal("10_000"),
                "interest_income": Decimal("5_000"),
                "insurance_recovery": Decimal("30_000"),
                "dividends": Decimal("80_000"),
            }
            return changes.get(account, Decimal("0"))

        mock_ledger_for_metrics.get_period_change = mock_period_change

        gen = FinancialStatementGenerator(
            manufacturer_data={"metrics_history": [], "config": None},
            ledger=mock_ledger_for_metrics,
        )
        result = gen._get_metrics_from_ledger(0)

        assert result["revenue"] == Decimal("5_000_000")
        assert result["depreciation_expense"] == Decimal("700_000")
        assert result["insurance_premiums"] == Decimal("100_000")
        assert result["insurance_losses"] == Decimal("50_000")
        assert result["dividends_paid"] == Decimal("80_000")
        # net_income should be total_revenue - total_expenses
        assert "net_income" in result
        assert result["is_solvent"] is not None


# ===========================================================================
# FinancialStatementGenerator: _get_year_metrics out of range (line 934)
# ===========================================================================


class TestGetYearMetrics:
    """Test _get_year_metrics boundary conditions."""

    def test_get_year_metrics_out_of_range(self):
        """Line 934: raises IndexError when year is out of range and no ledger."""
        mfr = _make_manufacturer_mock(_full_metrics())
        gen = FinancialStatementGenerator(manufacturer=mfr)
        with pytest.raises(IndexError, match="Year 5 out of range"):
            gen._get_year_metrics(5)


# ===========================================================================
# FinancialStatementGenerator: negative year with ledger (966, 1224, 1588)
# ===========================================================================


class TestNegativeYearWithLedger:
    """Test negative year checks when ledger is present."""

    @pytest.fixture
    def gen_with_ledger(self):
        """Create generator with a mock ledger."""
        mfr = _make_manufacturer_mock(_full_metrics())
        ledger = Mock()
        ledger.get_balance = Mock(return_value=Decimal("0"))
        ledger.get_period_change = Mock(return_value=Decimal("0"))
        ledger.get_cash_flows = Mock(return_value={})
        return FinancialStatementGenerator(manufacturer=mfr, ledger=ledger)

    def test_balance_sheet_negative_year_with_ledger(self, gen_with_ledger):
        """Line 966: balance sheet raises IndexError for negative year with ledger."""
        with pytest.raises(IndexError, match="must be non-negative"):
            gen_with_ledger.generate_balance_sheet(year=-1)

    def test_income_statement_negative_year_with_ledger(self, gen_with_ledger):
        """Line 1224: income statement raises IndexError for negative year with ledger."""
        with pytest.raises(IndexError, match="must be non-negative"):
            gen_with_ledger.generate_income_statement(year=-1)

    def test_cash_flow_negative_year_with_ledger(self, gen_with_ledger):
        """Line 1588: cash flow raises IndexError for negative year with ledger."""
        with pytest.raises(IndexError, match="must be non-negative"):
            gen_with_ledger.generate_cash_flow_statement(year=-1)


# ===========================================================================
# FinancialStatementGenerator: out of range without ledger (1226, 1590, 1618)
# ===========================================================================


class TestOutOfRangeWithoutLedger:
    """Test IndexError when year exceeds available data and no ledger present."""

    @pytest.fixture
    def gen_no_ledger(self):
        """Create generator without ledger."""
        mfr = _make_manufacturer_mock(_full_metrics())
        return FinancialStatementGenerator(manufacturer=mfr)

    def test_income_statement_out_of_range(self, gen_no_ledger):
        """Line 1226: income statement raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            gen_no_ledger.generate_income_statement(year=99)

    def test_cash_flow_out_of_range(self, gen_no_ledger):
        """Line 1590: cash flow statement raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            gen_no_ledger.generate_cash_flow_statement(year=99)

    def test_reconciliation_out_of_range(self, gen_no_ledger):
        """Line 1618: reconciliation report raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            gen_no_ledger.generate_reconciliation_report(year=99)


# ===========================================================================
# FinancialStatementGenerator: restricted_other > 0 (line 1069)
# ===========================================================================


class TestRestrictedAssets:
    """Test that restricted assets section shows other restricted."""

    def test_other_restricted_assets_displayed(self):
        """Line 1069: Other Restricted Assets appears when restricted > collateral."""
        metrics = _full_metrics(
            {
                "collateral": 100_000,
                "restricted_assets": 300_000,
            }
        )
        mfr = _make_manufacturer_mock(metrics)
        gen = FinancialStatementGenerator(manufacturer=mfr)
        df = gen.generate_balance_sheet(year=0)
        items = df["Item"].values
        assert any("Other Restricted Assets" in str(item) for item in items)


# ===========================================================================
# FinancialStatementGenerator: accrued details (1119, 1123, 1126)
# ===========================================================================


class TestAccruedDetails:
    """Test accrued expense breakdowns appear on balance sheet."""

    def test_accrued_wages_taxes_interest_displayed(self):
        """Lines 1119, 1123, 1126: accrued wages, taxes, interest, and other."""
        metrics = _full_metrics(
            {
                "accrued_wages": 50_000,
                "accrued_taxes": 30_000,
                "accrued_interest": 20_000,
                "accrued_expenses": 150_000,  # total is more than sum of specifics
            }
        )
        mfr = _make_manufacturer_mock(metrics)
        gen = FinancialStatementGenerator(manufacturer=mfr)
        df = gen.generate_balance_sheet(year=0)
        items = df["Item"].values
        assert any("Accrued Wages" in str(item) for item in items)
        assert any("Accrued Taxes" in str(item) for item in items)
        assert any("Accrued Interest" in str(item) for item in items)
        assert any("Other Accrued" in str(item) for item in items)


# ===========================================================================
# FinancialStatementGenerator: income statement comparison years (1253-1255)
# ===========================================================================


class TestIncomeStatementComparisonYears:
    """Test _add_comparison_year_income method."""

    def test_comparison_years_added_to_income_statement(self):
        """Lines 1253-1255, 1816-1828: compare_years adds columns to income statement."""
        m0 = _full_metrics({"year": 0, "revenue": 4_000_000, "net_income": 250_000})
        m1 = _full_metrics({"year": 1, "revenue": 5_000_000, "net_income": 300_000})
        m2 = _full_metrics({"year": 2, "revenue": 5_500_000, "net_income": 350_000})
        mfr = _make_manufacturer_mock(m0, m1, m2)
        gen = FinancialStatementGenerator(manufacturer=mfr)
        df = gen.generate_income_statement(year=2, compare_years=[0, 1])

        # Check comparison year columns were added
        assert "Year 0" in df.columns
        assert "Year 1" in df.columns

        # Check that Sales Revenue row has comparison values
        for i, item in enumerate(df["Item"].values):
            if "Sales Revenue" in str(item):
                assert df["Year 0"].values[i] == 4_000_000
                break


# ===========================================================================
# FinancialStatementGenerator: tax provision monthly (1491, 1497)
# ===========================================================================


class TestTaxProvisionMonthly:
    """Test tax provision with monthly period."""

    def test_tax_provision_from_ledger_monthly(self):
        """Line 1491: tax provision from ledger divided by 12 for monthly."""
        metrics = _full_metrics()
        mfr = _make_manufacturer_mock(metrics)
        ledger = Mock()
        ledger.get_balance = Mock(return_value=Decimal("0"))
        ledger.get_period_change = Mock(return_value=Decimal("120_000"))
        ledger.get_cash_flows = Mock(return_value={})
        gen = FinancialStatementGenerator(manufacturer=mfr, ledger=ledger)
        df = gen.generate_income_statement(year=0, monthly=True)

        # Find Current Tax Expense
        for i, item in enumerate(df["Item"].values):
            if "Current Tax Expense" in str(item):
                # 120_000 / 12 = 10_000
                assert abs(float(df["Month 0"].values[i]) - 10_000) < 1
                break

    def test_tax_provision_from_metrics_monthly(self):
        """Line 1497: tax_expense from metrics divided by 12 for monthly."""
        metrics = _full_metrics({"tax_expense": 240_000})
        mfr = _make_manufacturer_mock(metrics)
        gen = FinancialStatementGenerator(manufacturer=mfr)
        df = gen.generate_income_statement(year=0, monthly=True)

        for i, item in enumerate(df["Item"].values):
            if "Current Tax Expense" in str(item):
                assert abs(float(df["Month 0"].values[i]) - 20_000) < 1
                break


# ===========================================================================
# FinancialStatementGenerator: tax rate fallback warning (1505-1512)
# ===========================================================================


class TestTaxRateFallbackWarning:
    """Test warning when no tax_rate available in config."""

    def test_tax_rate_fallback_warning(self):
        """Lines 1505-1512: warning emitted and default 25% rate used."""
        metrics = _full_metrics()
        # Remove tax_expense from metrics
        metrics.pop("tax_expense", None)

        # Config object without tax_rate attribute
        config_obj = Mock()
        config_obj.initial_assets = 10_000_000
        del config_obj.tax_rate  # ensure hasattr returns False

        mfr = Mock()
        mfr.metrics_history = [metrics]
        mfr.config = config_obj
        mfr.ledger = None

        gen = FinancialStatementGenerator(
            manufacturer_data={
                "metrics_history": [metrics],
                "config": config_obj,
                "initial_assets": 10_000_000,
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = gen.generate_income_statement(year=0)
            # Check that a warning was emitted about tax rate
            tax_warnings = [x for x in w if "Tax rate" in str(x.message)]
            assert len(tax_warnings) >= 1


# ===========================================================================
# FinancialStatementGenerator: missing depreciation_expense income stmt (1314)
# ===========================================================================


class TestMissingDepreciationInIncomeStatement:
    """Test ValueError when depreciation_expense is missing from income statement."""

    def test_missing_depreciation_raises_in_income_statement(self):
        """Line 1314: missing depreciation_expense raises ValueError."""
        metrics = _full_metrics()
        del metrics["depreciation_expense"]
        mfr = _make_manufacturer_mock(metrics)
        gen = FinancialStatementGenerator(manufacturer=mfr)
        with pytest.raises(ValueError, match="depreciation_expense missing"):
            gen.generate_income_statement(year=0)


# ===========================================================================
# MonteCarloStatementAggregator: aggregate methods (1873-1876, 1893-1896)
# ===========================================================================


class TestMonteCarloAggregator:
    """Test MonteCarloStatementAggregator stub methods."""

    def test_aggregate_balance_sheets_not_implemented(self):
        """Lines 1873-1876: aggregate_balance_sheets raises NotImplementedError."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[])
        with pytest.raises(NotImplementedError):
            aggregator.aggregate_balance_sheets(year=0)

    def test_aggregate_balance_sheets_with_custom_percentiles(self):
        """Lines 1873-1876: with custom percentiles."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[])
        with pytest.raises(NotImplementedError):
            aggregator.aggregate_balance_sheets(year=0, percentiles=[10, 50, 90])

    def test_aggregate_income_statements_not_implemented(self):
        """Lines 1893-1896: aggregate_income_statements raises NotImplementedError."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[])
        with pytest.raises(NotImplementedError):
            aggregator.aggregate_income_statements(year=0)

    def test_aggregate_income_statements_with_custom_percentiles(self):
        """Lines 1893-1896: with custom percentiles."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[])
        with pytest.raises(NotImplementedError):
            aggregator.aggregate_income_statements(year=0, percentiles=[25, 75])

    def test_aggregate_balance_sheets_default_percentiles(self):
        """Lines 1873-1874: default percentiles=[5,25,50,75,95] assigned."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[])
        # Will hit the line that sets default percentiles before raising
        with pytest.raises(NotImplementedError):
            aggregator.aggregate_balance_sheets(year=0, percentiles=None)

    def test_aggregate_income_statements_default_percentiles(self):
        """Lines 1893-1894: default percentiles assigned."""
        aggregator = MonteCarloStatementAggregator(monte_carlo_results=[])
        with pytest.raises(NotImplementedError):
            aggregator.aggregate_income_statements(year=0, percentiles=None)

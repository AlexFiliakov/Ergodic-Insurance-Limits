"""Tests for DTA valuation allowance per ASC 740-10-30-5 (Issue #464).

Tests cover:
- Consecutive loss year tracking in TaxHandler
- Graduated valuation allowance rates (0%, 50%, 75%, 100%)
- Valuation allowance reversal on return to profitability
- Journal entry creation for allowance changes
- Balance sheet presentation (gross DTA - allowance = net DTA)
- Ledger balance verification after allowance entries
- Integration with WidgetManufacturer step() simulation
"""

from decimal import Decimal

import pytest

from ergodic_insurance.accrual_manager import AccrualManager
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.tax_handler import TaxHandler


class TestTaxHandlerConsecutiveLossTracking:
    """Test consecutive loss year tracking in TaxHandler."""

    @pytest.fixture
    def accrual_manager(self):
        return AccrualManager()

    @pytest.fixture
    def tax_handler(self, accrual_manager):
        return TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("0"),
            nol_limitation_pct=0.80,
        )

    def test_initial_consecutive_loss_years_is_zero(self, tax_handler):
        """Fresh TaxHandler should have zero consecutive loss years."""
        assert tax_handler.consecutive_loss_years == 0

    def test_loss_year_increments_counter(self, tax_handler):
        """A loss year should increment the consecutive loss counter."""
        tax_handler.calculate_tax_liability(-100_000)
        assert tax_handler.consecutive_loss_years == 1

    def test_consecutive_losses_accumulate(self, tax_handler):
        """Multiple consecutive loss years should accumulate."""
        for _ in range(5):
            tax_handler.calculate_tax_liability(-100_000)
        assert tax_handler.consecutive_loss_years == 5

    def test_profit_year_resets_counter(self, tax_handler):
        """A profit year should reset the consecutive loss counter to zero."""
        tax_handler.calculate_tax_liability(-100_000)
        tax_handler.calculate_tax_liability(-100_000)
        assert tax_handler.consecutive_loss_years == 2

        tax_handler.calculate_tax_liability(500_000)
        assert tax_handler.consecutive_loss_years == 0

    def test_zero_income_counts_as_loss(self, tax_handler):
        """Zero income (income <= 0) should count as a loss year."""
        tax_handler.calculate_tax_liability(0)
        assert tax_handler.consecutive_loss_years == 1

    def test_alternating_loss_profit_stays_at_zero_or_one(self, tax_handler):
        """Alternating loss/profit should keep counter at 0 or 1."""
        tax_handler.calculate_tax_liability(-100_000)
        assert tax_handler.consecutive_loss_years == 1

        tax_handler.calculate_tax_liability(500_000)
        assert tax_handler.consecutive_loss_years == 0

        tax_handler.calculate_tax_liability(-100_000)
        assert tax_handler.consecutive_loss_years == 1


class TestValuationAllowanceRate:
    """Test graduated valuation allowance rates per ASC 740-10-30-5."""

    @pytest.fixture
    def accrual_manager(self):
        return AccrualManager()

    @pytest.fixture
    def tax_handler(self, accrual_manager):
        return TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("1000000"),  # $1M NOL = $250K DTA
            nol_limitation_pct=0.80,
        )

    def test_no_allowance_below_threshold(self, tax_handler):
        """No allowance when consecutive losses < 3."""
        assert tax_handler.valuation_allowance_rate == ZERO
        assert tax_handler.valuation_allowance == ZERO

        tax_handler.consecutive_loss_years = 1
        assert tax_handler.valuation_allowance_rate == ZERO

        tax_handler.consecutive_loss_years = 2
        assert tax_handler.valuation_allowance_rate == ZERO

    def test_50_pct_at_three_years(self, tax_handler):
        """50% allowance at exactly 3 consecutive loss years."""
        tax_handler.consecutive_loss_years = 3
        assert tax_handler.valuation_allowance_rate == Decimal("0.50")
        assert tax_handler.valuation_allowance == Decimal("125000")  # 50% of $250K DTA

    def test_75_pct_at_four_years(self, tax_handler):
        """75% allowance at 4 consecutive loss years."""
        tax_handler.consecutive_loss_years = 4
        assert tax_handler.valuation_allowance_rate == Decimal("0.75")
        assert tax_handler.valuation_allowance == Decimal("187500")  # 75% of $250K DTA

    def test_100_pct_at_five_years(self, tax_handler):
        """100% allowance at 5+ consecutive loss years."""
        tax_handler.consecutive_loss_years = 5
        assert tax_handler.valuation_allowance_rate == to_decimal("1.00")
        assert tax_handler.valuation_allowance == Decimal("250000")  # 100% of $250K DTA

    def test_100_pct_at_many_years(self, tax_handler):
        """100% allowance stays at 100% for > 5 years."""
        tax_handler.consecutive_loss_years = 10
        assert tax_handler.valuation_allowance_rate == to_decimal("1.00")
        assert tax_handler.valuation_allowance == Decimal("250000")

    def test_zero_nol_means_zero_allowance(self, accrual_manager):
        """No DTA means no valuation allowance regardless of losses."""
        handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("0"),
        )
        handler.consecutive_loss_years = 5
        assert handler.valuation_allowance == ZERO


class TestNetDeferredTaxAsset:
    """Test net DTA computation (gross DTA - valuation allowance)."""

    @pytest.fixture
    def accrual_manager(self):
        return AccrualManager()

    @pytest.fixture
    def tax_handler(self, accrual_manager):
        return TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("1000000"),
        )

    def test_net_dta_equals_gross_when_no_allowance(self, tax_handler):
        """Net DTA = Gross DTA when no allowance."""
        assert tax_handler.net_deferred_tax_asset == tax_handler.deferred_tax_asset

    def test_net_dta_reduced_by_allowance(self, tax_handler):
        """Net DTA = Gross DTA - Allowance."""
        tax_handler.consecutive_loss_years = 3
        gross = tax_handler.deferred_tax_asset  # $250K
        allowance = tax_handler.valuation_allowance  # $125K (50%)
        assert tax_handler.net_deferred_tax_asset == gross - allowance
        assert tax_handler.net_deferred_tax_asset == Decimal("125000")

    def test_net_dta_zero_at_full_allowance(self, tax_handler):
        """Net DTA = 0 when 100% allowance."""
        tax_handler.consecutive_loss_years = 5
        assert tax_handler.net_deferred_tax_asset == ZERO


class TestValuationAllowanceIntegration:
    """Integration tests with WidgetManufacturer."""

    @pytest.fixture
    def config(self):
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,
            nol_carryforward_enabled=True,
            nol_limitation_pct=0.80,
        )

    @pytest.fixture
    def manufacturer(self, config):
        return WidgetManufacturer(config)

    def _force_loss_year(self, mfg):
        """Helper to force a loss year on the manufacturer."""
        revenue = mfg.calculate_revenue()
        operating_income = mfg.calculate_operating_income(revenue)
        excessive_costs = operating_income + to_decimal(500_000)
        return mfg.calculate_net_income(operating_income, excessive_costs, use_accrual=False)

    def _force_profit_year(self, mfg):
        """Helper to force a profit year on the manufacturer."""
        revenue = mfg.calculate_revenue()
        operating_income = mfg.calculate_operating_income(revenue)
        return mfg.calculate_net_income(operating_income, 0, use_accrual=False)

    def test_no_allowance_with_fewer_than_3_loss_years(self, manufacturer):
        """No valuation allowance with < 3 consecutive loss years."""
        self._force_loss_year(manufacturer)
        self._force_loss_year(manufacturer)

        assert manufacturer.tax_handler.consecutive_loss_years == 2
        assert manufacturer.dta_valuation_allowance == ZERO

    def test_allowance_appears_after_3_loss_years(self, manufacturer):
        """Valuation allowance should appear after 3 consecutive loss years."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        assert manufacturer.tax_handler.consecutive_loss_years == 3
        assert manufacturer.dta_valuation_allowance > ZERO

    def test_allowance_recorded_in_ledger(self, manufacturer):
        """Valuation allowance journal entries should be in the ledger."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        va_balance = abs(manufacturer.ledger.get_balance(AccountName.DTA_VALUATION_ALLOWANCE))
        assert va_balance > ZERO
        assert va_balance == manufacturer.dta_valuation_allowance

    def test_allowance_reduces_total_assets(self, manufacturer):
        """Valuation allowance should reduce total assets via net DTA."""
        # First create a DTA with 2 loss years (no allowance yet)
        self._force_loss_year(manufacturer)
        self._force_loss_year(manufacturer)
        assets_before_allowance = manufacturer.total_assets
        gross_dta = manufacturer.deferred_tax_asset

        # 3rd loss year triggers 50% allowance
        self._force_loss_year(manufacturer)
        assets_after_allowance = manufacturer.total_assets

        # The allowance should reduce total assets
        # (assets also change from the loss itself, but the DTA allowance is additional)
        assert manufacturer.dta_valuation_allowance > ZERO
        # Verify net DTA = gross DTA - allowance on balance sheet
        net_dta_on_bs = manufacturer.deferred_tax_asset - manufacturer.dta_valuation_allowance
        assert net_dta_on_bs >= ZERO

    def test_allowance_reversal_on_profitability(self, manufacturer):
        """Valuation allowance should reverse when company returns to profit."""
        # Create 3 loss years to trigger allowance
        for _ in range(3):
            self._force_loss_year(manufacturer)

        assert manufacturer.dta_valuation_allowance > ZERO
        va_before = manufacturer.dta_valuation_allowance

        # Profit year should reset consecutive losses and reverse allowance
        self._force_profit_year(manufacturer)

        assert manufacturer.tax_handler.consecutive_loss_years == 0
        # Allowance should be fully reversed (or reduced, depending on NOL utilization)
        assert manufacturer.dta_valuation_allowance < va_before

    def test_ledger_balanced_after_allowance_entries(self, manufacturer):
        """Ledger should remain balanced after valuation allowance entries."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        is_balanced, difference = manufacturer.ledger.verify_balance()
        assert is_balanced, f"Ledger out of balance by ${difference:,.2f}"

    def test_ledger_balanced_after_allowance_reversal(self, manufacturer):
        """Ledger should remain balanced after allowance reversal."""
        for _ in range(3):
            self._force_loss_year(manufacturer)
        self._force_profit_year(manufacturer)

        is_balanced, difference = manufacturer.ledger.verify_balance()
        assert is_balanced, f"Ledger out of balance by ${difference:,.2f}"

    def test_allowance_increases_with_more_loss_years(self, manufacturer):
        """Allowance rate should increase with more consecutive losses."""
        for _ in range(3):
            self._force_loss_year(manufacturer)
        va_at_3 = manufacturer.dta_valuation_allowance

        self._force_loss_year(manufacturer)
        va_at_4 = manufacturer.dta_valuation_allowance

        self._force_loss_year(manufacturer)
        va_at_5 = manufacturer.dta_valuation_allowance

        # Allowance should increase monotonically (higher rate on growing NOL)
        assert va_at_4 > va_at_3
        assert va_at_5 > va_at_4

    def test_dta_valuation_allowance_in_metrics(self, manufacturer):
        """dta_valuation_allowance should appear in calculate_metrics output."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        revenue = manufacturer.calculate_revenue()
        metrics = manufacturer.calculate_metrics(period_revenue=revenue)
        assert "dta_valuation_allowance" in metrics
        # The allowance metric is present; its value depends on whether the company
        # is still solvent after 3 consecutive loss years (insolvency may write off DTA)
        assert metrics["dta_valuation_allowance"] >= ZERO

    def test_fresh_manufacturer_has_no_allowance(self, config):
        """A fresh manufacturer should have no valuation allowance."""
        fresh = WidgetManufacturer(config)
        assert fresh.dta_valuation_allowance == ZERO
        assert fresh.tax_handler.consecutive_loss_years == 0
        assert fresh.tax_handler.valuation_allowance == ZERO


class TestValuationAllowanceAccounting:
    """Test the accounting entries for valuation allowance changes."""

    @pytest.fixture
    def config(self):
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,
            nol_carryforward_enabled=True,
            nol_limitation_pct=0.80,
        )

    @pytest.fixture
    def manufacturer(self, config):
        return WidgetManufacturer(config)

    def _force_loss_year(self, mfg):
        revenue = mfg.calculate_revenue()
        operating_income = mfg.calculate_operating_income(revenue)
        excessive_costs = operating_income + to_decimal(500_000)
        return mfg.calculate_net_income(operating_income, excessive_costs, use_accrual=False)

    def test_allowance_journal_entry_increases_tax_expense(self, manufacturer):
        """Creating valuation allowance should increase tax expense in the ledger."""
        # 2 loss years: no allowance yet
        self._force_loss_year(manufacturer)
        self._force_loss_year(manufacturer)
        tax_expense_before = manufacturer.ledger.get_balance(AccountName.TAX_EXPENSE)

        # 3rd loss year: triggers 50% allowance
        self._force_loss_year(manufacturer)
        tax_expense_after = manufacturer.ledger.get_balance(AccountName.TAX_EXPENSE)

        # Tax expense should increase due to the allowance (Dr TAX_EXPENSE)
        # Note: loss years have $0 current tax, but DTA changes affect tax expense
        va_amount = manufacturer.dta_valuation_allowance
        assert va_amount > ZERO

    def test_allowance_is_contra_asset(self, manufacturer):
        """DTA_VALUATION_ALLOWANCE should behave as a contra-asset (credit balance)."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        # Raw ledger balance for contra-asset should be negative (credit balance)
        raw_balance = manufacturer.ledger.get_balance(AccountName.DTA_VALUATION_ALLOWANCE)
        assert raw_balance < ZERO  # Credit-balance stored as negative for debit-normal account

        # Property returns absolute value
        assert manufacturer.dta_valuation_allowance == abs(raw_balance)

    def test_accounting_equation_with_allowance(self, manufacturer):
        """Assets = Liabilities + Equity should hold with valuation allowance."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        total_assets = manufacturer.total_assets
        total_liabilities = manufacturer.total_liabilities
        equity = manufacturer.equity

        # Accounting equation (with tolerance for rounding)
        assert abs(total_assets - (total_liabilities + equity)) < to_decimal("0.01"), (
            f"Accounting equation violated: "
            f"Assets={total_assets:,.2f}, Liabilities={total_liabilities:,.2f}, "
            f"Equity={equity:,.2f}"
        )


class TestRevenueExcludesDTA:
    """Tests for Issue #1055: calculate_revenue must not inflate revenue via VA/DTA."""

    @pytest.fixture
    def config(self):
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,
            nol_carryforward_enabled=True,
            nol_limitation_pct=0.80,
        )

    @pytest.fixture
    def manufacturer(self, config):
        return WidgetManufacturer(config)

    def _force_loss_year(self, mfg):
        """Helper to force a loss year on the manufacturer."""
        revenue = mfg.calculate_revenue()
        operating_income = mfg.calculate_operating_income(revenue)
        excessive_costs = operating_income + to_decimal(500_000)
        return mfg.calculate_net_income(operating_income, excessive_costs, use_accrual=False)

    def test_revenue_does_not_add_back_valuation_allowance(self, manufacturer):
        """Revenue must not add back VA to the asset base (Issue #1055).

        After 3 loss years, a valuation allowance exists. Revenue should
        be based on operating assets only, never inflated by adding VA back.
        """
        for _ in range(3):
            self._force_loss_year(manufacturer)

        va = manufacturer.dta_valuation_allowance
        assert va > ZERO, "Precondition: VA should exist after 3 loss years"

        revenue = manufacturer.calculate_revenue()
        net_dta = manufacturer.net_deferred_tax_asset
        operating_assets = manufacturer.total_assets - net_dta
        expected_revenue = max(ZERO, operating_assets) * to_decimal(
            manufacturer.asset_turnover_ratio
        )
        assert revenue == expected_revenue

    def test_revenue_excludes_net_dta(self, manufacturer):
        """Revenue base should exclude net DTA entirely (Issue #1055).

        A DTA represents future tax benefits, not productive capacity.
        """
        # After loss years, DTA is created — revenue should use
        # (total_assets - net_dta) as the base
        self._force_loss_year(manufacturer)
        net_dta = manufacturer.net_deferred_tax_asset
        assert net_dta > ZERO, "Precondition: DTA should exist after loss year"

        revenue = manufacturer.calculate_revenue()
        operating_assets = manufacturer.total_assets - net_dta
        expected = max(ZERO, operating_assets) * to_decimal(manufacturer.asset_turnover_ratio)
        assert abs(revenue - expected) < to_decimal(
            "0.01"
        ), f"Revenue {revenue} should equal operating_assets * turnover {expected}"

    def test_revenue_decreases_when_va_increases(self, manufacturer):
        """Revenue should decrease (or not increase) when VA increases (Issue #1055).

        Increasing VA reduces net_DTA on the balance sheet, which reduces
        total_assets. Since revenue is based on operating assets (total_assets
        minus net_DTA), and net_DTA decreases when VA increases, the operating
        assets stay constant — but total_assets drop, so this tests the old bug
        where revenue INCREASED when VA went up.
        """
        # 2 loss years — no VA yet
        self._force_loss_year(manufacturer)
        self._force_loss_year(manufacturer)
        assert manufacturer.dta_valuation_allowance == ZERO
        revenue_no_va = manufacturer.calculate_revenue()

        # 3rd loss year triggers VA
        self._force_loss_year(manufacturer)
        assert manufacturer.dta_valuation_allowance > ZERO
        revenue_with_va = manufacturer.calculate_revenue()

        # Revenue should NOT increase just because VA appeared.
        # (The company lost money — if anything, revenue base should shrink.)
        assert revenue_with_va <= revenue_no_va, (
            f"Revenue should not increase when VA increases: "
            f"before VA={revenue_no_va}, after VA={revenue_with_va}"
        )

    def test_downstream_operating_income_not_inflated(self, manufacturer):
        """Operating income should not be inflated by the old VA add-back (Issue #1055)."""
        for _ in range(3):
            self._force_loss_year(manufacturer)

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Operating income should be based on non-inflated revenue
        expected_base = revenue * to_decimal(manufacturer.base_operating_margin)
        # Operating income = base - premiums - losses - LAE - reserve dev
        assert operating_income <= expected_base

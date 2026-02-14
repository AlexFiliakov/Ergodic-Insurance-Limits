"""Unit tests for NOL carryforward per ASC 740 / IRC §172 (Issue #365).

Tests cover:
- NOL accumulation in loss years
- 80% taxable income limitation (IRC §172(a)(2))
- Multi-year NOL utilization and exhaustion
- Deferred tax asset on balance sheet
- DTA journal entry creation and reversal
- Backward compatibility with nol_carryforward_enabled=False
- Monte Carlo independence (create_fresh resets NOL)
- Integration with step() simulation
"""

from decimal import Decimal

import pytest

from ergodic_insurance.accrual_manager import AccrualManager
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.tax_handler import TaxHandler


class TestTaxHandlerNOL:
    """Test the TaxHandler NOL carryforward logic directly."""

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

    def test_loss_year_accumulates_nol(self, tax_handler):
        """Loss year: NOL should accumulate, tax should be zero."""
        tax, nol_used = tax_handler.calculate_tax_liability(-1_000_000)
        assert tax == ZERO
        assert nol_used == ZERO
        assert tax_handler.nol_carryforward == Decimal("1000000")

    def test_consecutive_loss_years_accumulate(self, tax_handler):
        """Consecutive loss years should accumulate NOL."""
        tax_handler.calculate_tax_liability(-500_000)
        tax_handler.calculate_tax_liability(-300_000)
        assert tax_handler.nol_carryforward == Decimal("800000")

    def test_profit_year_no_nol_standard_tax(self, tax_handler):
        """Profit year without NOL: standard tax applies."""
        tax, nol_used = tax_handler.calculate_tax_liability(1_000_000)
        assert tax == Decimal("250000")  # 25% of $1M
        assert nol_used == ZERO
        assert tax_handler.nol_carryforward == ZERO

    def test_nol_offset_with_80pct_limit(self, tax_handler):
        """NOL offset limited to 80% of taxable income per IRC §172(a)(2)."""
        # Create $2M NOL pool
        tax_handler.calculate_tax_liability(-2_000_000)
        assert tax_handler.nol_carryforward == Decimal("2000000")

        # Profit year: $1M income, 80% limit = $800K NOL used
        tax, nol_used = tax_handler.calculate_tax_liability(1_000_000)
        assert nol_used == Decimal("800000")  # 80% of $1M
        assert tax_handler.nol_carryforward == Decimal("1200000")  # $2M - $800K
        # Taxable income = $1M - $800K = $200K, tax = $200K * 25% = $50K
        assert tax == Decimal("50000")

    def test_nol_exhaustion(self, tax_handler):
        """NOL pool exhausts when fully utilized."""
        # Create $500K NOL pool
        tax_handler.calculate_tax_liability(-500_000)
        assert tax_handler.nol_carryforward == Decimal("500000")

        # Profit year: $1M income, 80% limit = $800K, but only $500K NOL available
        tax, nol_used = tax_handler.calculate_tax_liability(1_000_000)
        assert nol_used == Decimal("500000")  # Full pool used
        assert tax_handler.nol_carryforward == ZERO
        # Taxable income = $1M - $500K = $500K, tax = $500K * 25% = $125K
        assert tax == Decimal("125000")

    def test_nol_fully_exhausted_then_normal_tax(self, tax_handler):
        """After NOL exhaustion, normal taxation resumes."""
        tax_handler.calculate_tax_liability(-100_000)
        tax_handler.calculate_tax_liability(200_000)  # Uses up to 80% = $160K, uses $100K
        assert tax_handler.nol_carryforward == ZERO

        # Next profit year: no NOL, standard tax
        tax, nol_used = tax_handler.calculate_tax_liability(500_000)
        assert nol_used == ZERO
        assert tax == Decimal("125000")

    def test_issue_spec_example(self, tax_handler):
        """Verify the exact example from the issue specification."""
        # Year 1: -$1,000,000
        tax1, nol1 = tax_handler.calculate_tax_liability(-1_000_000)
        assert tax1 == ZERO
        assert nol1 == ZERO
        assert tax_handler.nol_carryforward == Decimal("1000000")

        # Year 2: +$500,000, 80% limit = $400K
        tax2, nol2 = tax_handler.calculate_tax_liability(500_000)
        assert nol2 == Decimal("400000")
        assert tax_handler.nol_carryforward == Decimal("600000")
        assert tax2 == Decimal("25000")  # (500K - 400K) * 25%

        # Year 3: +$2,000,000, 80% limit = $1.6M, only $600K NOL left
        tax3, nol3 = tax_handler.calculate_tax_liability(2_000_000)
        assert nol3 == Decimal("600000")
        assert tax_handler.nol_carryforward == ZERO
        assert tax3 == Decimal("350000")  # (2M - 600K) * 25%

    def test_deferred_tax_asset_property(self, tax_handler):
        """DTA = NOL carryforward * tax rate."""
        tax_handler.calculate_tax_liability(-1_000_000)
        assert tax_handler.deferred_tax_asset == Decimal("250000")  # $1M * 25%

        # After partial utilization
        tax_handler.calculate_tax_liability(500_000)
        # NOL remaining = $600K, DTA = $600K * 25% = $150K
        assert tax_handler.deferred_tax_asset == Decimal("150000")

    def test_zero_income_no_nol_change(self, tax_handler):
        """Zero income should not accumulate NOL (income <= 0 check)."""
        tax, nol_used = tax_handler.calculate_tax_liability(0)
        assert tax == ZERO
        assert nol_used == ZERO
        # Zero is <= ZERO, so it goes into loss branch, but abs(0) = 0
        assert tax_handler.nol_carryforward == ZERO

    def test_calculate_and_accrue_returns_nol_utilized(self, tax_handler):
        """calculate_and_accrue_tax returns nol_utilized as third element."""
        tax_handler.nol_carryforward = Decimal("500000")
        actual_tax, was_capped, nol_utilized = tax_handler.calculate_and_accrue_tax(
            income_before_tax=1_000_000,
            current_equity=5_000_000,
            use_accrual=True,
            time_resolution="annual",
            current_year=2024,
            current_month=0,
        )
        assert nol_utilized == Decimal("500000")
        assert actual_tax == Decimal("125000")  # (1M - 500K) * 25%
        assert was_capped is False


class TestNOLIntegration:
    """Integration tests with WidgetManufacturer."""

    @pytest.fixture
    def config(self):
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,  # Full retention for simpler math
            nol_carryforward_enabled=True,
            nol_limitation_pct=0.80,
        )

    @pytest.fixture
    def manufacturer(self, config):
        return WidgetManufacturer(config)

    def test_manufacturer_has_tax_handler(self, manufacturer):
        """Manufacturer should have a persistent TaxHandler instance."""
        assert hasattr(manufacturer, "tax_handler")
        assert isinstance(manufacturer.tax_handler, TaxHandler)
        assert manufacturer.tax_handler.nol_carryforward == ZERO

    def test_loss_year_creates_nol(self, manufacturer):
        """A loss year should accumulate NOL on the persistent tax handler."""
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Force a loss by adding excessive collateral costs
        excessive_costs = operating_income + to_decimal(500_000)
        net_income = manufacturer.calculate_net_income(
            operating_income, excessive_costs, use_accrual=False
        )

        assert net_income < ZERO
        assert manufacturer.tax_handler.nol_carryforward > ZERO

    def test_nol_reduces_future_tax(self, manufacturer):
        """NOL from a loss year should reduce tax in a subsequent profit year."""
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Year 1: Loss year — force large collateral cost
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(operating_income, excessive_costs, use_accrual=False)
        nol_after_loss = manufacturer.tax_handler.nol_carryforward
        assert nol_after_loss > ZERO

        # Year 2: Profit year — should use NOL
        net_income_with_nol = manufacturer.calculate_net_income(
            operating_income, 0, use_accrual=False
        )

        # Compare with a fresh manufacturer (no NOL)
        fresh_config = manufacturer.config.model_copy()
        fresh_mfr = WidgetManufacturer(fresh_config)
        revenue2 = fresh_mfr.calculate_revenue()
        operating_income2 = fresh_mfr.calculate_operating_income(revenue2)
        net_income_no_nol = fresh_mfr.calculate_net_income(operating_income2, 0, use_accrual=False)

        # Net income with NOL should be higher (less tax)
        assert net_income_with_nol > net_income_no_nol

    def test_dta_on_balance_sheet(self, manufacturer):
        """DTA should appear on balance sheet after loss year."""
        assert manufacturer.deferred_tax_asset == ZERO

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Force a loss with excessive collateral costs
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(operating_income, excessive_costs, use_accrual=False)

        # DTA should now be positive on the balance sheet
        assert manufacturer.deferred_tax_asset > ZERO
        # DTA = NOL * tax_rate
        expected_dta = manufacturer.tax_handler.nol_carryforward * to_decimal(manufacturer.tax_rate)
        assert manufacturer.deferred_tax_asset == expected_dta

    def test_dta_included_in_total_assets(self, manufacturer):
        """Total assets should include DTA."""
        initial_total_assets = manufacturer.total_assets

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Force a loss that creates DTA via excessive collateral costs
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(operating_income, excessive_costs, use_accrual=False)

        dta = manufacturer.deferred_tax_asset
        assert dta > ZERO

        # DTA is included in total_assets
        # (assets changed due to DTA journal entry, so we check the DTA is in the balance)
        assert manufacturer.deferred_tax_asset == manufacturer.ledger.get_balance(
            AccountName.DEFERRED_TAX_ASSET
        )

    def test_dta_reversal_on_nol_utilization(self, manufacturer):
        """DTA should decrease as NOL is utilized."""
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Create NOL via excessive collateral costs
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(operating_income, excessive_costs, use_accrual=False)
        dta_after_loss = manufacturer.deferred_tax_asset
        assert dta_after_loss > ZERO

        # Profit year — DTA should decrease
        manufacturer.calculate_net_income(operating_income, 0, use_accrual=False)
        dta_after_profit = manufacturer.deferred_tax_asset
        assert dta_after_profit < dta_after_loss

    def test_create_fresh_resets_nol(self, config):
        """create_fresh should start with zero NOL (Monte Carlo independence)."""
        mfr = WidgetManufacturer(config)

        # Create some NOL
        revenue = mfr.calculate_revenue()
        op_income = mfr.calculate_operating_income(revenue)
        mfr.calculate_net_income(op_income, op_income + 100_000, use_accrual=False)
        assert mfr.tax_handler.nol_carryforward > ZERO

        # create_fresh should have zero NOL
        fresh = WidgetManufacturer.create_fresh(config)
        assert fresh.tax_handler.nol_carryforward == ZERO
        assert fresh.deferred_tax_asset == ZERO

    def test_reset_clears_nol(self, manufacturer):
        """reset() should clear NOL state."""
        revenue = manufacturer.calculate_revenue()
        op_income = manufacturer.calculate_operating_income(revenue)
        manufacturer.calculate_net_income(op_income, op_income + 100_000, use_accrual=False)
        assert manufacturer.tax_handler.nol_carryforward > ZERO

        manufacturer.reset()
        assert manufacturer.tax_handler.nol_carryforward == ZERO

    def test_ledger_balanced_after_dta_entries(self, manufacturer):
        """Ledger should remain balanced after DTA journal entries."""
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Loss year — force via excessive collateral costs
        manufacturer.calculate_net_income(
            operating_income, operating_income + 500_000, use_accrual=False
        )
        balanced, diff = manufacturer.ledger.verify_balance()
        assert balanced, f"Ledger unbalanced by ${diff:,.2f} after loss year"

        # Profit year (partial NOL utilization)
        manufacturer.calculate_net_income(operating_income, 0, use_accrual=False)
        balanced, diff = manufacturer.ledger.verify_balance()
        assert balanced, f"Ledger unbalanced by ${diff:,.2f} after recovery year"


class TestNOLBackwardCompatibility:
    """Test backward compatibility when NOL is disabled."""

    @pytest.fixture
    def config_nol_disabled(self):
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,
            nol_carryforward_enabled=False,
        )

    @pytest.fixture
    def manufacturer_no_nol(self, config_nol_disabled):
        return WidgetManufacturer(config_nol_disabled)

    def test_disabled_nol_no_carryforward(self, manufacturer_no_nol):
        """With NOL disabled, losses should not create carryforward."""
        revenue = manufacturer_no_nol.calculate_revenue()
        operating_income = manufacturer_no_nol.calculate_operating_income(revenue)

        # Force a loss via excessive collateral costs
        excessive_costs = operating_income + to_decimal(500_000)
        net_income = manufacturer_no_nol.calculate_net_income(
            operating_income, excessive_costs, use_accrual=False
        )

        assert net_income < ZERO
        # NOL still accumulates on tax_handler since it's called internally,
        # but DTA journal entries are NOT created (no balance sheet impact)
        assert manufacturer_no_nol.deferred_tax_asset == ZERO

    def test_disabled_nol_no_dta_on_balance_sheet(self, manufacturer_no_nol):
        """With NOL disabled, no DTA should appear on balance sheet."""
        revenue = manufacturer_no_nol.calculate_revenue()
        operating_income = manufacturer_no_nol.calculate_operating_income(revenue)

        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer_no_nol.calculate_net_income(
            operating_income, excessive_costs, use_accrual=False
        )

        assert manufacturer_no_nol.deferred_tax_asset == ZERO

    def test_default_config_enables_nol(self):
        """Default ManufacturerConfig should enable NOL."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        assert config.nol_carryforward_enabled is True
        assert config.nol_limitation_pct == 0.80


class TestNOLMultiYear:
    """Test multi-year NOL scenarios using step()."""

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

    def test_step_preserves_nol_across_periods(self, config):
        """NOL should persist across step() calls.

        We inject NOL directly to test persistence through the step()
        lifecycle (not reset, not recreated) without relying on a loss
        scenario that may trigger mid-year insolvency checks.
        """
        mfr = WidgetManufacturer(config)

        # Inject NOL to simulate a prior-year loss
        mfr.tax_handler.nol_carryforward = Decimal("500000")
        nol_before = mfr.tax_handler.nol_carryforward

        # Run a profitable step — should utilize some NOL
        mfr.step()

        # NOL should have been partially utilized (not reset to 0 or lost)
        assert (
            mfr.tax_handler.nol_carryforward < nol_before
        ), "Profit year should reduce NOL via utilization"
        assert mfr.tax_handler.nol_carryforward >= ZERO, "NOL should not go negative"

    def test_nol_custom_limitation_pct(self):
        """Custom NOL limitation percentage should work."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,
            nol_carryforward_enabled=True,
            nol_limitation_pct=1.0,  # No limitation (pre-2018 rules)
        )
        mfr = WidgetManufacturer(config)
        assert mfr.tax_handler.nol_limitation_pct == 1.0

        # With 100% limitation, all NOL can be used
        mfr.tax_handler.nol_carryforward = Decimal("1000000")
        tax, nol_used = mfr.tax_handler.calculate_tax_liability(500_000)
        assert nol_used == Decimal("500000")  # Full 100% (limited by NOL pool, not %)
        assert tax == ZERO  # $500K - $500K = $0 taxable


class TestTCJALimitation:
    """Test TCJA applicability flag per IRC §172(a)(2) (Issue #808).

    Post-TCJA (apply_tcja_limitation=True): NOL deduction limited to 80%.
    Pre-TCJA (apply_tcja_limitation=False): NOL offsets 100% of taxable income.
    """

    @pytest.fixture
    def accrual_manager(self):
        return AccrualManager()

    @pytest.fixture
    def tcja_handler(self, accrual_manager):
        """TaxHandler with TCJA limitation enabled (default, post-2017)."""
        return TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("0"),
            nol_limitation_pct=0.80,
            apply_tcja_limitation=True,
        )

    @pytest.fixture
    def pre_tcja_handler(self, accrual_manager):
        """TaxHandler with TCJA limitation disabled (pre-2018 rules)."""
        return TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("0"),
            nol_limitation_pct=0.80,
            apply_tcja_limitation=False,
        )

    def test_default_applies_tcja(self, accrual_manager):
        """Default TaxHandler applies TCJA limitation."""
        handler = TaxHandler(tax_rate=0.25, accrual_manager=accrual_manager)
        assert handler.apply_tcja_limitation is True

    def test_tcja_limits_nol_to_80pct(self, tcja_handler):
        """Post-TCJA: NOL deduction capped at 80% of taxable income."""
        tcja_handler.nol_carryforward = Decimal("2000000")
        tax, nol_used = tcja_handler.calculate_tax_liability(1_000_000)

        assert nol_used == Decimal("800000")  # 80% of $1M
        assert tcja_handler.nol_carryforward == Decimal("1200000")
        # Taxable = $1M - $800K = $200K, tax = $200K * 25% = $50K
        assert tax == Decimal("50000")

    def test_pre_tcja_allows_full_offset(self, pre_tcja_handler):
        """Pre-TCJA: NOL can offset 100% of taxable income."""
        pre_tcja_handler.nol_carryforward = Decimal("2000000")
        tax, nol_used = pre_tcja_handler.calculate_tax_liability(1_000_000)

        assert nol_used == Decimal("1000000")  # Full 100% of $1M
        assert pre_tcja_handler.nol_carryforward == Decimal("1000000")
        # Taxable = $1M - $1M = $0, tax = $0
        assert tax == ZERO

    def test_pre_tcja_nol_limited_by_pool_not_income(self, pre_tcja_handler):
        """Pre-TCJA: NOL still limited by available pool, not income %."""
        pre_tcja_handler.nol_carryforward = Decimal("300000")
        tax, nol_used = pre_tcja_handler.calculate_tax_liability(1_000_000)

        assert nol_used == Decimal("300000")  # Pool exhausted
        assert pre_tcja_handler.nol_carryforward == ZERO
        # Taxable = $1M - $300K = $700K, tax = $700K * 25% = $175K
        assert tax == Decimal("175000")

    def test_tcja_flag_overrides_limitation_pct(self, accrual_manager):
        """apply_tcja_limitation=False bypasses nol_limitation_pct entirely."""
        handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_limitation_pct=0.50,  # Even a 50% limit is bypassed
            apply_tcja_limitation=False,
        )
        handler.nol_carryforward = Decimal("1000000")
        tax, nol_used = handler.calculate_tax_liability(800_000)

        # Without TCJA, full offset regardless of nol_limitation_pct
        assert nol_used == Decimal("800000")
        assert tax == ZERO

    def test_config_default_applies_tcja(self):
        """ManufacturerConfig defaults to apply_tcja_limitation=True."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        assert config.apply_tcja_limitation is True

    def test_config_pre_tcja_propagates(self):
        """apply_tcja_limitation=False propagates from config to TaxHandler."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=1.0,
            nol_carryforward_enabled=True,
            apply_tcja_limitation=False,
        )
        mfr = WidgetManufacturer(config)
        assert mfr.tax_handler.apply_tcja_limitation is False

    def test_pre_tcja_multi_year_scenario(self, pre_tcja_handler):
        """Pre-TCJA multi-year: losses fully offset future income."""
        # Year 1: -$1M loss
        pre_tcja_handler.calculate_tax_liability(-1_000_000)
        assert pre_tcja_handler.nol_carryforward == Decimal("1000000")

        # Year 2: +$500K profit — fully offset, no tax
        tax2, nol2 = pre_tcja_handler.calculate_tax_liability(500_000)
        assert nol2 == Decimal("500000")
        assert tax2 == ZERO
        assert pre_tcja_handler.nol_carryforward == Decimal("500000")

        # Year 3: +$800K profit — $500K offset, $300K taxable
        tax3, nol3 = pre_tcja_handler.calculate_tax_liability(800_000)
        assert nol3 == Decimal("500000")
        assert pre_tcja_handler.nol_carryforward == ZERO
        assert tax3 == Decimal("75000")  # $300K * 25%

    def test_tcja_vs_pre_tcja_same_scenario(self, accrual_manager):
        """Demonstrate difference between TCJA and pre-TCJA on same scenario."""
        # Post-TCJA handler
        tcja = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            apply_tcja_limitation=True,
        )
        tcja.nol_carryforward = Decimal("1000000")
        tcja_tax, tcja_nol = tcja.calculate_tax_liability(1_000_000)

        # Pre-TCJA handler (fresh accrual manager to avoid shared state)
        pre_tcja = TaxHandler(
            tax_rate=0.25,
            accrual_manager=AccrualManager(),
            apply_tcja_limitation=False,
        )
        pre_tcja.nol_carryforward = Decimal("1000000")
        pre_tcja_tax, pre_tcja_nol = pre_tcja.calculate_tax_liability(1_000_000)

        # Post-TCJA: $800K offset, $50K tax
        assert tcja_nol == Decimal("800000")
        assert tcja_tax == Decimal("50000")

        # Pre-TCJA: $1M offset, $0 tax
        assert pre_tcja_nol == Decimal("1000000")
        assert pre_tcja_tax == ZERO

        # Pre-TCJA always produces less or equal tax
        assert pre_tcja_tax <= tcja_tax


class TestNetIncomeAssertionRemoval:
    """Verify that net_income > operating_income no longer crashes (Issue #1314).

    The old assertion ``net_income <= operating_income`` was conceptually
    flawed because deferred-tax adjustments (DTA recognition, DTL reversal,
    valuation allowance reversal) can legitimately cause net income to exceed
    operating income per ASC 740.  The assertion has been replaced with a
    ``logger.warning()`` call.
    """

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

    def test_no_assertion_error_with_dta_and_collateral_costs(self, manufacturer):
        """DTA recognition with collateral costs must not raise AssertionError.

        Regression test for Issue #1314: a loss year that creates DTA followed
        by a profit year with small collateral costs should complete without
        crashing, even though DTA journal entries modify the TAX_EXPENSE
        ledger balance.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Year 1 — force a loss via excessive collateral costs (creates NOL/DTA)
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(operating_income, excessive_costs, use_accrual=False)
        assert manufacturer.tax_handler.nol_carryforward > ZERO
        assert manufacturer.deferred_tax_asset > ZERO

        # Year 2 — profit year WITH small collateral costs
        # This previously risked an AssertionError when DTA adjustments
        # reduced effective tax expense.
        small_collateral = to_decimal(1_000)
        net_income = manufacturer.calculate_net_income(
            operating_income, small_collateral, use_accrual=False
        )

        # Should complete without error; net_income is valid
        assert isinstance(net_income, Decimal)

    def test_no_assertion_error_with_dtl_reversal_and_collateral(self, manufacturer):
        """DTL reversal with collateral costs must not raise AssertionError.

        A loss year followed by recovery year creates DTA changes. Combined
        with collateral costs, this exercises the path described in #1314.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Force multiple loss years to create substantial DTA
        for _ in range(2):
            excessive_costs = operating_income + to_decimal(200_000)
            manufacturer.calculate_net_income(operating_income, excessive_costs, use_accrual=False)

        # Recovery year with moderate collateral costs
        collateral = to_decimal(50_000)
        net_income = manufacturer.calculate_net_income(
            operating_income, collateral, use_accrual=False
        )
        assert isinstance(net_income, Decimal)

    def test_warning_logged_when_net_income_exceeds_operating_income(self, config):
        """A warning should be logged when net_income > operating_income.

        We patch calculate_and_accrue_tax to simulate the scenario where
        a deferred-tax benefit causes actual_tax_expense to be negative,
        resulting in net_income > operating_income despite collateral costs.
        This is the exact failure mode described in Issue #1314.
        """
        import logging
        from unittest.mock import patch

        manufacturer = WidgetManufacturer(config)
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # To get net_income > operating_income we need actual_tax < 0.
        # Since the current TaxHandler cannot return negative tax, we patch
        # to return -$1000 to simulate a future ASC 740 tax-benefit path.
        # operating_income ≈ $1.5M, collateral = $100
        # income_before_tax ≈ $1,499,900
        # patched tax = -$1000 → net_income = $1,499,900 - (-$1000) = $1,500,900
        # net_income ($1,500,900) > operating_income ($1,500,000) → warning
        logger = logging.getLogger("ergodic_insurance.manufacturer_income")
        with (
            patch.object(
                manufacturer.tax_handler,
                "calculate_and_accrue_tax",
                return_value=(to_decimal("-1000"), False, ZERO),
            ),
            patch.object(logger, "warning") as mock_warning,
        ):
            # This should NOT raise AssertionError (old behavior would crash)
            net_income = manufacturer.calculate_net_income(
                operating_income, to_decimal("100"), use_accrual=False
            )

        # Net income should exceed operating income due to tax benefit
        assert net_income > operating_income

        # Warning should have been emitted
        mock_warning.assert_called_once()
        warning_msg = mock_warning.call_args[0][0]
        assert "exceeds operating income" in warning_msg

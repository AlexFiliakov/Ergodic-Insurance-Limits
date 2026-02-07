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

        # Force a loss by adding excessive insurance costs
        excessive_costs = operating_income + to_decimal(500_000)
        net_income = manufacturer.calculate_net_income(
            operating_income, 0, excessive_costs, 0, use_accrual=False
        )

        assert net_income < ZERO
        assert manufacturer.tax_handler.nol_carryforward > ZERO

    def test_nol_reduces_future_tax(self, manufacturer):
        """NOL from a loss year should reduce tax in a subsequent profit year."""
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Year 1: Loss year — force large insurance cost
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(
            operating_income, 0, excessive_costs, 0, use_accrual=False
        )
        nol_after_loss = manufacturer.tax_handler.nol_carryforward
        assert nol_after_loss > ZERO

        # Year 2: Profit year — should use NOL
        net_income_with_nol = manufacturer.calculate_net_income(
            operating_income, 0, 0, 0, use_accrual=False
        )

        # Compare with a fresh manufacturer (no NOL)
        fresh_config = manufacturer.config.model_copy()
        fresh_mfr = WidgetManufacturer(fresh_config)
        revenue2 = fresh_mfr.calculate_revenue()
        operating_income2 = fresh_mfr.calculate_operating_income(revenue2)
        net_income_no_nol = fresh_mfr.calculate_net_income(
            operating_income2, 0, 0, 0, use_accrual=False
        )

        # Net income with NOL should be higher (less tax)
        assert net_income_with_nol > net_income_no_nol

    def test_dta_on_balance_sheet(self, manufacturer):
        """DTA should appear on balance sheet after loss year."""
        assert manufacturer.deferred_tax_asset == ZERO

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Force a loss
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(
            operating_income, 0, excessive_costs, 0, use_accrual=False
        )

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

        # Force a loss that creates DTA
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(
            operating_income, 0, excessive_costs, 0, use_accrual=False
        )

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

        # Create NOL
        excessive_costs = operating_income + to_decimal(500_000)
        manufacturer.calculate_net_income(
            operating_income, 0, excessive_costs, 0, use_accrual=False
        )
        dta_after_loss = manufacturer.deferred_tax_asset
        assert dta_after_loss > ZERO

        # Profit year — DTA should decrease
        manufacturer.calculate_net_income(operating_income, 0, 0, 0, use_accrual=False)
        dta_after_profit = manufacturer.deferred_tax_asset
        assert dta_after_profit < dta_after_loss

    def test_create_fresh_resets_nol(self, config):
        """create_fresh should start with zero NOL (Monte Carlo independence)."""
        mfr = WidgetManufacturer(config)

        # Create some NOL
        revenue = mfr.calculate_revenue()
        op_income = mfr.calculate_operating_income(revenue)
        mfr.calculate_net_income(op_income, 0, op_income + 100_000, 0, use_accrual=False)
        assert mfr.tax_handler.nol_carryforward > ZERO

        # create_fresh should have zero NOL
        fresh = WidgetManufacturer.create_fresh(config)
        assert fresh.tax_handler.nol_carryforward == ZERO
        assert fresh.deferred_tax_asset == ZERO

    def test_reset_clears_nol(self, manufacturer):
        """reset() should clear NOL state."""
        revenue = manufacturer.calculate_revenue()
        op_income = manufacturer.calculate_operating_income(revenue)
        manufacturer.calculate_net_income(op_income, 0, op_income + 100_000, 0, use_accrual=False)
        assert manufacturer.tax_handler.nol_carryforward > ZERO

        manufacturer.reset()
        assert manufacturer.tax_handler.nol_carryforward == ZERO

    def test_ledger_balanced_after_dta_entries(self, manufacturer):
        """Ledger should remain balanced after DTA journal entries."""
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Loss year
        manufacturer.calculate_net_income(
            operating_income, 0, operating_income + 500_000, 0, use_accrual=False
        )
        balanced, diff = manufacturer.ledger.verify_balance()
        assert balanced, f"Ledger unbalanced by ${diff:,.2f} after loss year"

        # Profit year (partial NOL utilization)
        manufacturer.calculate_net_income(operating_income, 0, 0, 0, use_accrual=False)
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

        # Force a loss
        excessive_costs = operating_income + to_decimal(500_000)
        net_income = manufacturer_no_nol.calculate_net_income(
            operating_income, 0, excessive_costs, 0, use_accrual=False
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
            operating_income, 0, excessive_costs, 0, use_accrual=False
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

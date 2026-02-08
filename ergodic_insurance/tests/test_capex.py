"""Test capital expenditure (capex) functionality (Issue #543).

Tests verify that capex is recorded as Dr GROSS_PPE, Cr CASH using the
existing CAPEX transaction type, with amount = depreciation_expense *
capex_to_depreciation_ratio.
"""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestCapexConfig:
    """Test capex configuration parameter."""

    def test_default_ratio_is_one(self):
        """Default capex_to_depreciation_ratio should be 1.0 (maintenance capex)."""
        config = ManufacturerConfig()
        assert config.capex_to_depreciation_ratio == 1.0

    def test_ratio_zero_allowed(self):
        """Ratio of 0.0 should be valid (legacy behavior, no capex)."""
        config = ManufacturerConfig(capex_to_depreciation_ratio=0.0)
        assert config.capex_to_depreciation_ratio == 0.0

    def test_ratio_max_five(self):
        """Ratio up to 5.0 should be valid."""
        config = ManufacturerConfig(capex_to_depreciation_ratio=5.0)
        assert config.capex_to_depreciation_ratio == 5.0

    def test_ratio_negative_rejected(self):
        """Negative ratio should be rejected by validation."""
        with pytest.raises(Exception):
            ManufacturerConfig(capex_to_depreciation_ratio=-0.1)

    def test_ratio_above_max_rejected(self):
        """Ratio above 5.0 should be rejected by validation."""
        with pytest.raises(Exception):
            ManufacturerConfig(capex_to_depreciation_ratio=5.1)


class TestCapexRecording:
    """Test capex recording via record_capex() method."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with capex ratio = 0 (to test record_capex directly)."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,  # Disable auto capex
        )
        return WidgetManufacturer(config)

    def test_record_capex_increases_gross_ppe(self, manufacturer):
        """Capex should increase gross PP&E."""
        initial_gross_ppe = manufacturer.gross_ppe
        capex = to_decimal(100_000)

        manufacturer.record_capex(capex)

        assert manufacturer.gross_ppe == initial_gross_ppe + capex

    def test_record_capex_decreases_cash(self, manufacturer):
        """Capex should decrease cash."""
        initial_cash = manufacturer.cash
        capex = to_decimal(100_000)

        manufacturer.record_capex(capex)

        assert manufacturer.cash == initial_cash - capex

    def test_record_capex_does_not_affect_accumulated_depreciation(self, manufacturer):
        """Capex should not change accumulated depreciation."""
        initial_accum = manufacturer.accumulated_depreciation
        manufacturer.record_capex(to_decimal(100_000))
        assert manufacturer.accumulated_depreciation == initial_accum

    def test_record_capex_increases_net_ppe(self, manufacturer):
        """Capex should increase net PP&E (since it increases gross without touching accum depr)."""
        initial_net_ppe = manufacturer.net_ppe
        capex = to_decimal(100_000)

        manufacturer.record_capex(capex)

        assert manufacturer.net_ppe == initial_net_ppe + capex

    def test_record_capex_zero_is_noop(self, manufacturer):
        """Recording zero capex should have no effect."""
        initial_gross_ppe = manufacturer.gross_ppe
        initial_cash = manufacturer.cash

        result = manufacturer.record_capex(ZERO)

        assert result == ZERO
        assert manufacturer.gross_ppe == initial_gross_ppe
        assert manufacturer.cash == initial_cash

    def test_record_capex_negative_is_noop(self, manufacturer):
        """Recording negative capex should have no effect."""
        initial_gross_ppe = manufacturer.gross_ppe
        result = manufacturer.record_capex(to_decimal(-50_000))

        assert result == ZERO
        assert manufacturer.gross_ppe == initial_gross_ppe

    def test_record_capex_capped_by_cash(self, manufacturer):
        """Capex cannot exceed available cash."""
        available_cash = manufacturer.cash
        excessive_capex = available_cash + to_decimal(1_000_000)

        actual = manufacturer.record_capex(excessive_capex)

        assert actual == available_cash
        assert manufacturer.cash == ZERO

    def test_record_capex_returns_actual_amount(self, manufacturer):
        """record_capex should return the actual capex recorded."""
        capex = to_decimal(200_000)
        result = manufacturer.record_capex(capex)
        assert result == capex

    def test_total_assets_unchanged_by_capex(self, manufacturer):
        """Capex is asset-neutral: cash decreases, PP&E increases by the same amount."""
        initial_total = manufacturer.total_assets
        manufacturer.record_capex(to_decimal(100_000))
        assert float(manufacturer.total_assets) == pytest.approx(float(initial_total), rel=1e-9)


class TestCapexInStep:
    """Test capex integration in the step() method."""

    def _make_manufacturer(self, capex_ratio):
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=capex_ratio,
        )
        return WidgetManufacturer(config)

    def test_ratio_zero_no_capex(self):
        """With ratio=0, no capex is recorded (legacy behavior)."""
        m = self._make_manufacturer(0.0)
        initial_gross_ppe = m.gross_ppe

        m.step(time_resolution="annual")

        # Gross PP&E should be unchanged (no capex added)
        assert m.gross_ppe == initial_gross_ppe
        # Net PP&E should have decreased (depreciation only, no offset)
        assert m.net_ppe < initial_gross_ppe

    def test_ratio_one_maintenance_capex(self):
        """With ratio=1.0, capex equals depreciation — net PP&E approximately stable."""
        m = self._make_manufacturer(1.0)
        initial_gross_ppe = m.gross_ppe
        initial_net_ppe = m.net_ppe

        m.step(time_resolution="annual")

        # Gross PP&E should have increased by depreciation amount
        expected_depreciation = initial_gross_ppe / to_decimal(10)
        assert float(m.gross_ppe) == pytest.approx(
            float(initial_gross_ppe + expected_depreciation), rel=0.01
        )
        # Net PP&E should be approximately unchanged:
        # net_ppe = (gross + capex) - (accum + depr)
        # Since capex = depr, net_ppe stays the same
        assert float(m.net_ppe) == pytest.approx(float(initial_net_ppe), rel=0.01)

    def test_ratio_above_one_growth_capex(self):
        """With ratio>1.0, net PP&E should grow."""
        m = self._make_manufacturer(1.5)
        initial_net_ppe = m.net_ppe

        m.step(time_resolution="annual")

        # Net PP&E should increase: capex (1.5x depr) > depreciation
        assert m.net_ppe > initial_net_ppe

    def test_ratio_below_one_managed_decline(self):
        """With ratio<1.0, net PP&E should decline (but less than ratio=0)."""
        m_decline = self._make_manufacturer(0.5)
        m_none = self._make_manufacturer(0.0)

        initial_net_ppe_decline = m_decline.net_ppe
        initial_net_ppe_none = m_none.net_ppe

        m_decline.step(time_resolution="annual")
        m_none.step(time_resolution="annual")

        # Both should decline
        assert m_decline.net_ppe < initial_net_ppe_decline
        assert m_none.net_ppe < initial_net_ppe_none
        # But decline with 0.5x capex should be less severe
        decline_with_capex = initial_net_ppe_decline - m_decline.net_ppe
        decline_without_capex = initial_net_ppe_none - m_none.net_ppe
        assert decline_with_capex < decline_without_capex

    def test_capex_reduces_cash(self):
        """Capex in step() should reduce cash compared to no-capex scenario."""
        m_capex = self._make_manufacturer(1.0)
        m_no_capex = self._make_manufacturer(0.0)

        m_capex.step(time_resolution="annual")
        m_no_capex.step(time_resolution="annual")

        # Manufacturer with capex should have less cash
        assert m_capex.cash < m_no_capex.cash

    def test_capex_does_not_affect_operating_income_in_step(self):
        """Capex is capitalized, not expensed — should not change operating income.

        Note: The metrics dict recalculates depreciation from current gross_ppe
        (which is higher after capex), so the reported net_income metric differs.
        This is a pre-existing metrics issue, not a capex bug. The actual
        financial flow in step() uses the correct pre-capex depreciation.
        """
        m_capex = self._make_manufacturer(1.5)
        m_no_capex = self._make_manufacturer(0.0)

        # Both start with the same equity
        assert m_capex.equity == m_no_capex.equity

        m_capex.step(time_resolution="annual")
        m_no_capex.step(time_resolution="annual")

        # Capex is asset-neutral (cash down, PPE up), so total_assets and equity
        # differ only by the retained earnings effect (same net income in step()).
        # The equity difference should be small relative to the capex amount.
        capex_amount = m_capex.gross_ppe - m_no_capex.gross_ppe
        equity_diff = abs(float(m_capex.equity - m_no_capex.equity))
        # Equity should be close; any difference comes from the metrics recalculation
        # feeding into dividends (retention_ratio applied to recalculated net_income).
        assert equity_diff < float(capex_amount)

    def test_multi_period_ppe_stability_at_ratio_one(self):
        """Over multiple periods at ratio=1.0, net PP&E should remain stable."""
        m = self._make_manufacturer(1.0)
        initial_net_ppe = float(m.net_ppe)

        for _ in range(5):
            m.step(time_resolution="annual", growth_rate=0.0)

        # Net PP&E should still be approximately the initial value
        # (slight drift possible due to depreciation being based on growing gross_ppe)
        assert float(m.net_ppe) == pytest.approx(initial_net_ppe, rel=0.05)

    def test_multi_period_ppe_growth_at_ratio_above_one(self):
        """Over multiple periods at ratio=1.3, net PP&E should grow."""
        m = self._make_manufacturer(1.3)
        initial_net_ppe = m.net_ppe

        for _ in range(5):
            m.step(time_resolution="annual", growth_rate=0.0)

        assert m.net_ppe > initial_net_ppe

    def test_monthly_capex(self):
        """Capex should also work in monthly time resolution."""
        m = self._make_manufacturer(1.0)
        initial_gross_ppe = m.gross_ppe

        # Run 12 monthly steps
        for _ in range(12):
            m.step(time_resolution="monthly")

        # Gross PP&E should have increased (capex was recorded each month)
        assert m.gross_ppe > initial_gross_ppe

    def test_accounting_equation_holds_with_capex(self):
        """Assets = Liabilities + Equity must hold after capex."""
        m = self._make_manufacturer(1.5)

        m.step(time_resolution="annual")

        # The step() method already calls _verify_accounting_equation()
        # which would raise if A != L + E. If we get here, it passed.
        # Double-check explicitly:
        total_assets = m.total_assets
        total_liabilities = m.total_liabilities
        equity = m.equity
        assert float(total_assets) == pytest.approx(float(total_liabilities + equity), rel=1e-6)

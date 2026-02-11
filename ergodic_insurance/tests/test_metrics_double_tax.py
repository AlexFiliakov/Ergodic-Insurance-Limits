"""Unit tests for Issue #617: calculate_metrics() double-fires tax calculation.

Tests verify that:
- calculate_metrics() does not call calculate_net_income() or mutate TaxHandler state
- NOL carryforward balance is identical whether calculate_metrics() is called 0, 1, or N times
- Ledger contains no duplicate DTA journal entries from metrics calculation
- step() caches net_income so calculate_metrics() can read it without side effects
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def config():
    """Manufacturer config with NOL carryforward enabled."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.2,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,
        nol_carryforward_enabled=True,
    )


@pytest.fixture
def manufacturer(config):
    return WidgetManufacturer(config)


class TestMetricsDoubleTax:
    """Tests for Issue #617: calculate_metrics() must not mutate tax state."""

    def test_step_caches_net_income(self, manufacturer):
        """step() should cache net_income in _period_net_income."""
        assert manufacturer._period_net_income is None
        manufacturer.step()
        assert manufacturer._period_net_income is not None

    def test_calculate_metrics_uses_cached_net_income(self, manufacturer):
        """calculate_metrics() should use _period_net_income from step()."""
        metrics = manufacturer.step()
        cached = manufacturer._period_net_income

        # The net_income in metrics should match the cached value
        assert metrics["net_income"] == cached

    def test_nol_unchanged_after_multiple_calculate_metrics_calls(self, manufacturer):
        """NOL carryforward must not change when calculate_metrics() is called repeatedly."""
        manufacturer.step()
        nol_after_step = manufacturer.tax_handler.nol_carryforward

        # Call calculate_metrics() multiple times
        manufacturer.calculate_metrics()
        nol_after_first = manufacturer.tax_handler.nol_carryforward

        manufacturer.calculate_metrics()
        nol_after_second = manufacturer.tax_handler.nol_carryforward

        manufacturer.calculate_metrics()
        nol_after_third = manufacturer.tax_handler.nol_carryforward

        assert nol_after_first == nol_after_step
        assert nol_after_second == nol_after_step
        assert nol_after_third == nol_after_step

    def test_nol_unchanged_with_loss_year(self, manufacturer):
        """Even in a loss year, calculate_metrics() must not double-accumulate NOL."""
        # Force a loss by setting a very low margin
        manufacturer.base_operating_margin = -0.10
        manufacturer.step()
        nol_after_step = manufacturer.tax_handler.nol_carryforward

        # Multiple metric calls should not change NOL
        manufacturer.calculate_metrics()
        assert manufacturer.tax_handler.nol_carryforward == nol_after_step

        manufacturer.calculate_metrics()
        assert manufacturer.tax_handler.nol_carryforward == nol_after_step

    def test_no_duplicate_dta_journal_entries(self, manufacturer):
        """Ledger must not have duplicate DTA entries from calculate_metrics()."""
        # Force a loss year to create DTA entries
        manufacturer.base_operating_margin = -0.10
        manufacturer.step()

        dta_entries_after_step = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.DTA_ADJUSTMENT
        ]
        count_after_step = len(dta_entries_after_step)

        # Call calculate_metrics() — should NOT add more DTA entries
        manufacturer.calculate_metrics()
        dta_entries_after_metrics = [
            e
            for e in manufacturer.ledger.entries
            if e.transaction_type == TransactionType.DTA_ADJUSTMENT
        ]
        assert len(dta_entries_after_metrics) == count_after_step

    def test_dta_balance_unchanged_by_calculate_metrics(self, manufacturer):
        """DTA balance on ledger must not change from calculate_metrics()."""
        manufacturer.step()
        dta_after_step = manufacturer.ledger.get_balance(AccountName.DEFERRED_TAX_ASSET)

        manufacturer.calculate_metrics()
        dta_after_metrics = manufacturer.ledger.get_balance(AccountName.DEFERRED_TAX_ASSET)

        assert dta_after_metrics == dta_after_step

    def test_consecutive_loss_years_unchanged(self, manufacturer):
        """consecutive_loss_years must not change from calculate_metrics()."""
        manufacturer.base_operating_margin = -0.10
        manufacturer.step()
        loss_years_after_step = manufacturer.tax_handler.consecutive_loss_years

        manufacturer.calculate_metrics()
        assert manufacturer.tax_handler.consecutive_loss_years == loss_years_after_step

    def test_multi_period_nol_stability(self, manufacturer):
        """NOL should track correctly across multiple periods with calculate_metrics() calls."""
        nol_history = []

        for year in range(5):
            # Alternate between loss and profit years
            if year % 2 == 0:
                manufacturer.base_operating_margin = -0.05
            else:
                manufacturer.base_operating_margin = 0.12

            manufacturer.step()
            nol_after_step = manufacturer.tax_handler.nol_carryforward
            nol_history.append(nol_after_step)

            # Call calculate_metrics() extra times — must not affect NOL
            for _ in range(3):
                manufacturer.calculate_metrics()
                assert manufacturer.tax_handler.nol_carryforward == nol_after_step

    def test_metrics_net_income_equals_step_net_income(self, manufacturer):
        """The net_income in metrics should equal what step() computed."""
        metrics_from_step = manufacturer.step()
        metrics_from_calc = manufacturer.calculate_metrics()

        assert metrics_from_calc["net_income"] == metrics_from_step["net_income"]

    def test_calculate_metrics_without_step_warns(self, manufacturer, caplog):
        """calculate_metrics() without prior step() should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            manufacturer.calculate_metrics()

        assert "calculate_metrics() called without prior step()" in caplog.text

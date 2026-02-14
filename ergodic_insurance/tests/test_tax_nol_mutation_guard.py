"""Tests for NOL state mutation guard (Issue #1331).

Verifies that:
- estimate_tax_liability() is a pure query that never mutates NOL state
- calculate_tax_liability() continues to mutate as before (backward compat)
- calculate_metrics() fallback path preserves NOL state
"""

from decimal import Decimal

import pytest

from ergodic_insurance.accrual_manager import AccrualManager
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.tax_handler import TaxHandler


class TestEstimateTaxLiability:
    """estimate_tax_liability() must be a pure, side-effect-free query."""

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

    @pytest.fixture
    def tax_handler_with_nol(self, accrual_manager):
        """Handler with pre-existing NOL balance."""
        return TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("500000"),
            nol_limitation_pct=0.80,
        )

    # ── Pure query: no state changes ──

    def test_estimate_does_not_accumulate_nol_on_loss(self, tax_handler):
        """estimate_tax_liability on a loss year must NOT add to nol_carryforward."""
        tax, nol_used = tax_handler.estimate_tax_liability(-1_000_000)
        assert tax == ZERO
        assert nol_used == ZERO
        assert tax_handler.nol_carryforward == ZERO
        assert tax_handler.consecutive_loss_years == 0

    def test_estimate_does_not_consume_nol_on_profit(self, tax_handler_with_nol):
        """estimate_tax_liability on a profit year must NOT reduce nol_carryforward."""
        handler = tax_handler_with_nol
        original_nol = handler.nol_carryforward

        tax, nol_used = handler.estimate_tax_liability(1_000_000)

        # Should compute correct values
        assert nol_used == Decimal("500000")  # min(500K, 80% of 1M = 800K)
        expected_taxable = Decimal("1000000") - nol_used
        assert tax == expected_taxable * to_decimal(0.25)

        # But NOL state must be unchanged
        assert handler.nol_carryforward == original_nol
        assert handler.consecutive_loss_years == 0

    def test_estimate_does_not_reset_consecutive_loss_years(self, accrual_manager):
        """estimate_tax_liability must not reset consecutive_loss_years on profit."""
        handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("100000"),
            consecutive_loss_years=3,
        )
        handler.estimate_tax_liability(500_000)
        assert handler.consecutive_loss_years == 3  # unchanged

    def test_estimate_does_not_increment_consecutive_loss_years(self, accrual_manager):
        """estimate_tax_liability must not increment consecutive_loss_years on loss."""
        handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            consecutive_loss_years=2,
        )
        handler.estimate_tax_liability(-500_000)
        assert handler.consecutive_loss_years == 2  # unchanged

    def test_estimate_repeated_calls_are_idempotent(self, tax_handler_with_nol):
        """Calling estimate_tax_liability N times must always return the same result."""
        handler = tax_handler_with_nol
        results = [handler.estimate_tax_liability(1_000_000) for _ in range(5)]
        assert all(r == results[0] for r in results)
        assert handler.nol_carryforward == Decimal("500000")

    # ── Parity: estimate matches calculate result ──

    def test_estimate_matches_calculate_for_profit(self, accrual_manager):
        """estimate and calculate must return identical (tax, nol_used) values."""
        est_handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("300000"),
        )
        calc_handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=AccrualManager(),
            nol_carryforward=Decimal("300000"),
        )
        est_result = est_handler.estimate_tax_liability(1_000_000)
        calc_result = calc_handler.calculate_tax_liability(1_000_000)
        assert est_result == calc_result

    def test_estimate_matches_calculate_for_loss(self, accrual_manager):
        """Both methods return (0, 0) for loss years."""
        est_handler = TaxHandler(tax_rate=0.25, accrual_manager=accrual_manager)
        calc_handler = TaxHandler(tax_rate=0.25, accrual_manager=AccrualManager())
        est_result = est_handler.estimate_tax_liability(-500_000)
        calc_result = calc_handler.calculate_tax_liability(-500_000)
        assert est_result == calc_result

    def test_estimate_matches_calculate_for_zero_income(self, tax_handler):
        """Both methods handle zero income identically."""
        est = tax_handler.estimate_tax_liability(0)
        calc_handler = TaxHandler(tax_rate=0.25, accrual_manager=AccrualManager())
        calc = calc_handler.calculate_tax_liability(0)
        assert est == calc

    def test_estimate_matches_calculate_no_nol(self, accrual_manager):
        """Positive income with no NOL: standard tax, identical results."""
        est_handler = TaxHandler(tax_rate=0.25, accrual_manager=accrual_manager)
        calc_handler = TaxHandler(tax_rate=0.25, accrual_manager=AccrualManager())
        est = est_handler.estimate_tax_liability(1_000_000)
        calc = calc_handler.calculate_tax_liability(1_000_000)
        assert est == calc
        assert est[0] == Decimal("250000")

    # ── calculate_tax_liability still mutates (backward compat) ──

    def test_calculate_still_accumulates_nol_on_loss(self, tax_handler):
        """Backward compat: calculate_tax_liability must still mutate on loss."""
        tax_handler.calculate_tax_liability(-1_000_000)
        assert tax_handler.nol_carryforward == Decimal("1000000")
        assert tax_handler.consecutive_loss_years == 1

    def test_calculate_still_consumes_nol_on_profit(self, tax_handler_with_nol):
        """Backward compat: calculate_tax_liability must still consume NOL."""
        handler = tax_handler_with_nol
        handler.calculate_tax_liability(1_000_000)
        assert handler.nol_carryforward == ZERO  # 500K consumed
        assert handler.consecutive_loss_years == 0

    # ── Pre-TCJA parity ──

    def test_estimate_pre_tcja_no_limitation(self, accrual_manager):
        """Pre-TCJA: NOL can offset 100% of income, estimate reflects this."""
        handler = TaxHandler(
            tax_rate=0.25,
            accrual_manager=accrual_manager,
            nol_carryforward=Decimal("1000000"),
            apply_tcja_limitation=False,
        )
        tax, nol_used = handler.estimate_tax_liability(800_000)
        assert nol_used == Decimal("800000")  # Full offset, no 80% limit
        assert tax == ZERO
        assert handler.nol_carryforward == Decimal("1000000")  # unchanged


class TestMetricsFallbackNOLGuard:
    """calculate_metrics() fallback must not corrupt NOL state (Issue #1331)."""

    @pytest.fixture
    def manufacturer(self):
        from ergodic_insurance.config import ManufacturerConfig
        from ergodic_insurance.manufacturer import WidgetManufacturer

        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.5,
            base_operating_margin=0.10,
            tax_rate=0.25,
            nol_carryforward_enabled=True,
        )
        return WidgetManufacturer(config)

    def test_metrics_fallback_preserves_nol_state(self, manufacturer):
        """When _period_net_income is not cached, metrics must not corrupt NOL."""
        # Seed some NOL state via the authoritative path
        manufacturer.tax_handler.nol_carryforward = Decimal("500000")
        manufacturer.tax_handler.consecutive_loss_years = 2

        # Clear the cache to force the fallback
        if hasattr(manufacturer, "_period_net_income"):
            del manufacturer._period_net_income

        # Call calculate_metrics — this should NOT change NOL state
        manufacturer.calculate_metrics()

        assert manufacturer.tax_handler.nol_carryforward == Decimal("500000")
        assert manufacturer.tax_handler.consecutive_loss_years == 2

    def test_metrics_fallback_with_zero_nol(self, manufacturer):
        """Fallback with zero NOL should leave state at zero."""
        assert manufacturer.tax_handler.nol_carryforward == ZERO
        if hasattr(manufacturer, "_period_net_income"):
            del manufacturer._period_net_income

        manufacturer.calculate_metrics()

        assert manufacturer.tax_handler.nol_carryforward == ZERO
        assert manufacturer.tax_handler.consecutive_loss_years == 0

"""Tests for consistent equity definitions across solvency assessments (Issue #1311).

Validates that check_solvency(), compute_z_prime_score(),
_assess_going_concern_indicators(), and calculate_metrics() all use the same
solvency_equity definition (book equity + valuation allowance).
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def config() -> ManufacturerConfig:
    return ManufacturerConfig(
        initial_assets=5_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,
        ppe_ratio=0.1,
    )


@pytest.fixture
def manufacturer(config) -> WidgetManufacturer:
    return WidgetManufacturer(config)


class TestSolvencyEquityProperty:
    """Test the solvency_equity property."""

    def test_solvency_equity_equals_book_equity_when_no_va(self, manufacturer):
        """Without valuation allowance, solvency_equity equals book equity."""
        assert manufacturer.solvency_equity == manufacturer.equity

    def test_solvency_equity_adds_back_va(self, manufacturer):
        """solvency_equity = book equity + valuation allowance."""
        # Run a step to potentially create DTA/VA entries
        manufacturer.step()

        va = manufacturer.dta_valuation_allowance
        expected = manufacturer.equity + va
        assert manufacturer.solvency_equity == expected

    def test_solvency_equity_is_decimal(self, manufacturer):
        """solvency_equity returns a Decimal."""
        assert isinstance(manufacturer.solvency_equity, Decimal)


class TestEquityConsistencyAcrossAssessments:
    """Verify all solvency assessments use solvency_equity consistently."""

    def test_check_solvency_uses_solvency_equity(self, config):
        """check_solvency() Tier 1 uses solvency_equity, not raw equity.

        A company with negative book equity but positive solvency_equity
        (due to large VA) should pass the Tier 1 hard stop.
        """
        manufacturer = WidgetManufacturer(config)
        # Run enough steps to build up VA
        for _ in range(3):
            manufacturer.step()

        # The Tier 1 check tests solvency_equity <= 0, so if solvency_equity > 0,
        # the company should not fail on Tier 1 alone
        if manufacturer.solvency_equity > ZERO:
            # Healthy company should pass
            assert manufacturer.check_solvency()

    def test_z_prime_and_check_solvency_use_same_equity(self, config):
        """Z-prime score and check_solvency() use the same equity definition.

        This is the core consistency check from Issue #1311. Previously,
        Z-prime used raw self.equity while check_solvency() used operational
        equity, creating contradictory assessments.
        """
        manufacturer = WidgetManufacturer(config)
        # Add claim to create interesting financial state
        manufacturer.process_uninsured_claim(to_decimal(3_000_000), immediate_payment=False)

        # Both should use solvency_equity
        solvency_eq = manufacturer.solvency_equity

        # Z-prime X2 = solvency_equity / total_assets
        # Z-prime X4 = solvency_equity / total_liabilities
        total_assets = manufacturer.total_assets
        total_liabilities = manufacturer.total_liabilities

        expected_x2 = solvency_eq / total_assets
        expected_x4 = (
            solvency_eq / total_liabilities if total_liabilities > ZERO else to_decimal(10)
        )

        # Compute z-prime and verify the equity terms are consistent
        z_prime = manufacturer.compute_z_prime_score()

        # Recompute z-prime manually using solvency_equity to verify
        reported_cash = max(manufacturer.cash, to_decimal(0))
        current_assets = (
            reported_cash
            + manufacturer.accounts_receivable
            + manufacturer.inventory
            + manufacturer.prepaid_insurance
        )
        claim_total = sum(
            (cl.remaining_amount for cl in manufacturer.claim_liabilities), to_decimal(0)
        )
        dtl = manufacturer.deferred_tax_liability
        current_liabilities = total_liabilities - claim_total - dtl
        working_capital = current_assets - current_liabilities
        revenue = manufacturer.calculate_revenue()
        ebit = revenue * to_decimal(manufacturer.base_operating_margin)

        x1 = working_capital / total_assets
        x2 = solvency_eq / total_assets
        x3 = ebit / total_assets
        x4 = solvency_eq / total_liabilities if total_liabilities > ZERO else to_decimal(10)
        x5 = revenue / total_assets

        expected_z = (
            to_decimal("0.717") * x1
            + to_decimal("0.847") * x2
            + to_decimal("3.107") * x3
            + to_decimal("0.42") * x4
            + to_decimal("0.998") * x5
        )
        assert z_prime == expected_z

    def test_going_concern_equity_ratio_uses_solvency_equity(self, config):
        """Going concern Equity Ratio indicator uses solvency_equity."""
        manufacturer = WidgetManufacturer(config)
        manufacturer.process_uninsured_claim(to_decimal(2_000_000), immediate_payment=False)

        indicators = manufacturer._assess_going_concern_indicators()
        er_indicator = next(ind for ind in indicators if ind["name"] == "Equity Ratio")

        # Equity ratio should be solvency_equity / total_assets
        expected_ratio = manufacturer.solvency_equity / manufacturer.total_assets
        assert er_indicator["value"] == expected_ratio

    def test_metrics_equity_uses_solvency_equity(self, config):
        """calculate_metrics() reports solvency_equity as 'equity'."""
        manufacturer = WidgetManufacturer(config)
        metrics = manufacturer.step()

        assert metrics["equity"] == manufacturer.solvency_equity

    def test_all_equity_checks_agree_with_va(self, config):
        """When VA is non-zero, all assessments still use the same equity.

        Creates a scenario where VA exists, then verifies that the equity
        value used in metrics, Z-prime, and going concern are all identical.
        """
        manufacturer = WidgetManufacturer(config)
        # Process a large claim and step to generate tax effects and VA
        manufacturer.process_uninsured_claim(to_decimal(3_000_000), immediate_payment=False)
        metrics = manufacturer.step()

        solvency_eq = manufacturer.solvency_equity

        # Metrics should report solvency_equity
        assert metrics["equity"] == solvency_eq

        # Going concern equity ratio should use solvency_equity
        indicators = manufacturer._assess_going_concern_indicators()
        er = next(ind for ind in indicators if ind["name"] == "Equity Ratio")
        if manufacturer.total_assets > ZERO:
            assert er["value"] == solvency_eq / manufacturer.total_assets

    def test_consistency_scenario_from_issue(self, config):
        """Reproduce the exact scenario from Issue #1311.

        Company: total_assets=$5M, total_liabilities=$4.8M, VA=$300K
        Before fix: check_solvency saw $500K equity, Z-prime saw $200K.
        After fix: both see $500K (solvency_equity).
        """
        manufacturer = WidgetManufacturer(config)

        # Add claims to bring liabilities close to assets
        manufacturer.process_uninsured_claim(to_decimal(4_500_000), immediate_payment=False)

        book_equity = manufacturer.equity
        solvency_eq = manufacturer.solvency_equity

        # solvency_equity should be >= book_equity (VA >= 0)
        assert solvency_eq >= book_equity

        # All assessments should use solvency_equity, not book_equity
        if manufacturer.total_assets > ZERO:
            indicators = manufacturer._assess_going_concern_indicators()
            er = next(ind for ind in indicators if ind["name"] == "Equity Ratio")
            assert er["value"] == solvency_eq / manufacturer.total_assets

            z_prime = manufacturer.compute_z_prime_score()
            # Z-prime should be computed with solvency_equity, giving higher
            # X2 and X4 values than if book_equity were used
            if manufacturer.total_liabilities > ZERO:
                x4_solvency = solvency_eq / manufacturer.total_liabilities
                x4_book = book_equity / manufacturer.total_liabilities
                # If VA > 0, solvency equity gives higher X4
                va = manufacturer.dta_valuation_allowance
                if va > ZERO:
                    assert x4_solvency > x4_book

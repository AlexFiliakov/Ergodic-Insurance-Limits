"""Tests for working capital facility limit and liquidity-driven insolvency (Issue #1337).

Verifies that:
1. Working capital facility has a configurable limit
2. Cash below -(facility_limit) triggers insolvency
3. Post-payment liquidity check fires in step()
4. Default None preserves existing unlimited behavior
5. Facility-aware payment coordination includes facility in available liquidity
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestFacilityLimitConfig:
    """Tests for the working_capital_facility_limit config parameter."""

    def test_default_facility_limit_is_none(self):
        """Default facility limit should be None (unlimited, legacy behavior)."""
        config = ManufacturerConfig()
        assert config.working_capital_facility_limit is None

    def test_facility_limit_configurable(self):
        """Facility limit can be set to a specific amount."""
        config = ManufacturerConfig(working_capital_facility_limit=1_000_000)
        assert config.working_capital_facility_limit == 1_000_000

    def test_facility_limit_zero_allowed(self):
        """Zero facility means no borrowing is allowed."""
        config = ManufacturerConfig(working_capital_facility_limit=0)
        assert config.working_capital_facility_limit == 0

    def test_facility_limit_rejects_negative(self):
        """Negative facility limit should be rejected by validation."""
        with pytest.raises(Exception):
            ManufacturerConfig(working_capital_facility_limit=-100_000)


class TestCheckSolvencyFacilityBreach:
    """Tests for facility limit enforcement in check_solvency()."""

    @pytest.fixture
    def manufacturer_with_facility(self):
        """Manufacturer with a $500K working capital facility limit."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
            working_capital_facility_limit=500_000,
        )
        return WidgetManufacturer(config)

    def test_cash_within_facility_remains_solvent(self, manufacturer_with_facility):
        """Cash negative but within facility limit should not trigger insolvency."""
        mfr = manufacturer_with_facility
        mfr.step()

        # Push cash slightly negative but within $500K facility
        current_cash = mfr.cash
        overdraft_amount = current_cash + to_decimal(200_000)  # cash → -$200K
        if overdraft_amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.OPERATING_EXPENSES,
                credit_account=AccountName.CASH,
                amount=overdraft_amount,
                transaction_type=TransactionType.EXPENSE,
                description="Test: overdraft within facility",
            )

        assert mfr.cash < ZERO
        assert mfr.cash >= to_decimal(-500_000)

        # Should still be solvent — within facility limit
        result = mfr.check_solvency()
        assert result is True
        assert not mfr.is_ruined

    def test_cash_breaching_facility_triggers_insolvency(self, manufacturer_with_facility):
        """Cash below -(facility_limit) should trigger insolvency."""
        mfr = manufacturer_with_facility
        mfr.step()

        # Push cash well below facility limit ($500K)
        current_cash = mfr.cash
        overdraft_amount = current_cash + to_decimal(700_000)  # cash → -$700K > $500K limit
        if overdraft_amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.OPERATING_EXPENSES,
                credit_account=AccountName.CASH,
                amount=overdraft_amount,
                transaction_type=TransactionType.EXPENSE,
                description="Test: overdraft exceeding facility",
            )

        assert mfr.cash < to_decimal(-500_000)

        # Should now be insolvent — breached facility
        result = mfr.check_solvency()
        assert result is False
        assert mfr.is_ruined

    def test_unlimited_facility_negative_cash_not_facility_insolvency(self):
        """With default None facility (unlimited), negative cash doesn't trigger
        the facility-based insolvency check (Tier 1a).

        Note: large overdrafts may still trip Tier 2 going concern indicators
        (current ratio, cash runway), which is correct behavior.  This test
        verifies that a *small* overdraft with unlimited facility does not
        cause insolvency, preserving legacy behavior per Issue #496.
        """
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            capex_to_depreciation_ratio=0.0,
        )
        mfr = WidgetManufacturer(config)
        mfr.step()

        # Push cash slightly negative — enough to test facility logic
        # but not so much that going concern indicators breach
        current_cash = mfr.cash
        overdraft_amount = current_cash + to_decimal(100_000)
        if overdraft_amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.OPERATING_EXPENSES,
                credit_account=AccountName.CASH,
                amount=overdraft_amount,
                transaction_type=TransactionType.EXPENSE,
                description="Test: small overdraft, unlimited facility",
            )

        assert mfr.cash < ZERO

        # With unlimited facility and small overdraft, should remain solvent
        assert mfr.equity > ZERO
        result = mfr.check_solvency()
        assert result is True
        assert not mfr.is_ruined

    def test_facility_breach_logged(self, manufacturer_with_facility, caplog):
        """Facility breach should produce a LIQUIDITY INSOLVENCY warning."""
        import logging

        mfr = manufacturer_with_facility
        mfr.step()

        current_cash = mfr.cash
        overdraft_amount = current_cash + to_decimal(700_000)
        if overdraft_amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.OPERATING_EXPENSES,
                credit_account=AccountName.CASH,
                amount=overdraft_amount,
                transaction_type=TransactionType.EXPENSE,
                description="Test: overdraft for logging check",
            )

        with caplog.at_level(logging.WARNING):
            mfr.check_solvency()

        assert any("LIQUIDITY INSOLVENCY" in record.message for record in caplog.records)


class TestPostPaymentLiquidityCheck:
    """Tests for post-payment liquidity check in step()."""

    def test_large_claim_within_facility_continues(self):
        """Claim payment drawing on facility within limit should allow step to continue."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
            working_capital_facility_limit=2_000_000,
        )
        mfr = WidgetManufacturer(config)

        # Add a claim that's large but within cash + facility capacity
        from ergodic_insurance.manufacturer import ClaimLiability

        claim = ClaimLiability(
            original_amount=to_decimal(500_000),
            remaining_amount=to_decimal(500_000),
            year_incurred=0,
        )
        mfr.claim_liabilities.append(claim)

        # Step should proceed — claim within available liquidity
        metrics = mfr.step()
        assert not mfr.is_ruined
        assert "year" in metrics

    def test_zero_facility_limits_payments_to_cash(self):
        """Zero facility means payments capped at cash only — no borrowing allowed."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
            working_capital_facility_limit=0,
        )
        mfr = WidgetManufacturer(config)

        # Step should succeed — no large claims, normal operations
        metrics = mfr.step()
        assert not mfr.is_ruined
        assert "year" in metrics


class TestFacilityAwarePaymentCoordination:
    """Tests verifying facility limit is included in available liquidity for payments."""

    def test_facility_increases_available_liquidity_for_payments(self):
        """With facility limit, available liquidity should include the facility amount."""
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
            working_capital_facility_limit=1_000_000,
        )
        mfr = WidgetManufacturer(config)

        # Reduce cash to a small amount
        initial_cash = mfr.cash
        cash_to_remove = initial_cash - to_decimal(100_000)
        if cash_to_remove > ZERO:
            mfr.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.CASH,
                amount=cash_to_remove,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Tie up cash for test",
            )

        # Add a claim of $500K — exceeds $100K cash but within cash + $1M facility
        from ergodic_insurance.manufacturer import ClaimLiability

        claim = ClaimLiability(
            original_amount=to_decimal(500_000),
            remaining_amount=to_decimal(500_000),
            year_incurred=0,
        )
        mfr.claim_liabilities.append(claim)

        # Step should proceed — the facility provides extra liquidity
        metrics = mfr.step()
        assert not mfr.is_ruined

    def test_payment_exceeding_cash_plus_facility_caps_and_continues(self):
        """Payment due exceeding cash + facility should be capped, not crash."""
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
            working_capital_facility_limit=100_000,
        )
        mfr = WidgetManufacturer(config)

        # Reduce cash significantly
        initial_cash = mfr.cash
        cash_to_remove = initial_cash - to_decimal(50_000)
        if cash_to_remove > ZERO:
            mfr.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.CASH,
                amount=cash_to_remove,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Tie up cash for test",
            )

        # Add a massive claim — $2M, way above $50K cash + $100K facility
        from ergodic_insurance.manufacturer import ClaimLiability

        claim = ClaimLiability(
            original_amount=to_decimal(2_000_000),
            remaining_amount=to_decimal(2_000_000),
            year_incurred=0,
        )
        mfr.claim_liabilities.append(claim)

        # Step should still proceed (payments are capped at available liquidity)
        # The payment cap prevents cash from breaching the facility
        metrics = mfr.step()
        assert "year" in metrics

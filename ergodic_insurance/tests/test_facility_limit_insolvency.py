"""Tests for working capital facility limit and liquidity-driven insolvency (Issue #1337).

Verifies that:
1. Working capital facility has a configurable limit
2. Cash below -(facility_limit) triggers insolvency
3. Post-payment liquidity check fires in step()
4. Default None preserves existing unlimited behavior
5. Facility-aware payment coordination includes facility in available liquidity
"""

from decimal import Decimal
from typing import Any

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


class TestFacilityRatioScaling:
    """Tests for the revenue-scaled working_capital_facility_ratio (Issue #1625).

    When the ratio is set, the *effective* facility limit is
    ``ratio * current_revenue``, so the revolver grows with the firm instead of
    staying a fixed dollar amount.  When the ratio is None the fixed
    ``working_capital_facility_limit`` is used (bit-identical legacy behavior).
    """

    @staticmethod
    def _mfr(**overrides) -> WidgetManufacturer:
        """Build a healthy $10M-asset / $8M-revenue manufacturer for these tests."""
        params: dict[str, Any] = {
            "initial_assets": 10_000_000,
            "asset_turnover_ratio": 0.8,  # revenue = 10M * 0.8 = $8M at t=0
            "base_operating_margin": 0.08,
            "tax_rate": 0.25,
            "retention_ratio": 0.7,
            "capex_to_depreciation_ratio": 0.0,
        }
        params.update(overrides)
        return WidgetManufacturer(ManufacturerConfig(**params))

    @staticmethod
    def _overdraw_to(mfr: WidgetManufacturer, target: Decimal) -> None:
        """Drive cash down to ``target`` (negative) by tying cash up in receivables.

        Uses a working-capital move (Dr AR, Cr CASH) rather than an expense so
        that total_assets — and therefore revenue and the revenue-scaled
        facility — stay constant.  An expense would shrink assets, shrink
        revenue, and shrink the ratio-scaled facility mid-test, confounding the
        Tier-1a breach check we want to isolate.
        """
        amount = mfr.cash - target
        if amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.CASH,
                amount=amount,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Test: tie up cash in receivables to reach target overdraft",
            )

    # --- Config validation / defaults ---

    def test_ratio_default_is_none(self):
        """Default facility ratio is None (fall back to fixed limit / unlimited)."""
        assert ManufacturerConfig().working_capital_facility_ratio is None

    def test_ratio_configurable(self):
        """Facility ratio can be set to a fraction of revenue."""
        config = ManufacturerConfig(working_capital_facility_ratio=0.20)
        assert config.working_capital_facility_ratio == 0.20

    def test_ratio_zero_allowed(self):
        """Zero ratio means no overdraft headroom (mirrors a fixed limit of 0)."""
        config = ManufacturerConfig(working_capital_facility_ratio=0)
        assert config.working_capital_facility_ratio == 0

    def test_ratio_rejects_negative(self):
        """Negative ratio is rejected by validation (Field ge=0)."""
        with pytest.raises(Exception):
            ManufacturerConfig(working_capital_facility_ratio=-0.1)

    # --- effective_facility_limit() resolution ---

    def test_effective_limit_equals_ratio_times_revenue(self):
        """With a ratio set, the effective limit is ratio * current_revenue."""
        mfr = self._mfr(working_capital_facility_ratio=0.20)
        expected = to_decimal(0.20) * mfr.calculate_revenue()  # 0.20 * $8M
        eff = mfr.effective_facility_limit()
        assert eff == expected
        assert eff == to_decimal(1_600_000)

    def test_effective_limit_falls_back_to_fixed_when_ratio_none(self):
        """Ratio None + fixed set -> effective limit is the fixed Decimal value."""
        mfr = self._mfr(working_capital_facility_limit=2_000_000)
        assert mfr.config.working_capital_facility_ratio is None
        assert mfr.effective_facility_limit() == to_decimal(2_000_000)

    def test_effective_limit_none_when_unconfigured(self):
        """Neither ratio nor fixed set -> None (unlimited / legacy behavior)."""
        mfr = self._mfr()
        assert mfr.effective_facility_limit() is None

    def test_ratio_takes_precedence_over_fixed(self):
        """When BOTH are set, the ratio wins (revenue-scaled, not the fixed $)."""
        mfr = self._mfr(
            working_capital_facility_limit=500_000,
            working_capital_facility_ratio=0.20,
        )
        eff = mfr.effective_facility_limit()
        assert eff == to_decimal(0.20) * mfr.calculate_revenue()  # $1.6M
        assert eff == to_decimal(1_600_000)
        assert eff != to_decimal(500_000)

    def test_effective_limit_grows_with_revenue(self):
        """A larger (higher-revenue) firm gets a proportionally larger facility.

        This is the core realism fix: the revolver tracks revenue, so a firm
        that has grown tolerates a proportionally larger overdraft before breach.
        """
        small = self._mfr(initial_assets=10_000_000, working_capital_facility_ratio=0.20)
        large = self._mfr(initial_assets=20_000_000, working_capital_facility_ratio=0.20)
        eff_small = small.effective_facility_limit()
        eff_large = large.effective_facility_limit()
        assert eff_small is not None and eff_large is not None
        assert eff_small == to_decimal(0.20) * small.calculate_revenue()  # $1.6M
        assert eff_large == to_decimal(0.20) * large.calculate_revenue()  # $3.2M
        assert eff_large > eff_small
        # 2x assets -> 2x revenue -> exactly 2x facility
        assert eff_large == eff_small * to_decimal(2)

    # --- Integration: Tier-1a breach uses the effective (ratio-scaled) limit ---

    def test_cash_within_ratio_facility_remains_solvent(self):
        """Cash negative but within the ratio-scaled limit stays solvent."""
        mfr = self._mfr(working_capital_facility_ratio=0.05)  # ~$400K facility
        mfr.step()
        eff = mfr.effective_facility_limit()
        assert eff is not None
        # Overdraw to 40% of the facility — well within, and small enough not to
        # trip the Tier-2 going-concern indicators on this $10M firm.
        self._overdraw_to(mfr, -(eff * to_decimal("0.4")))
        assert mfr.cash < ZERO
        assert mfr.cash >= -eff
        assert mfr.check_solvency() is True
        assert not mfr.is_ruined

    def test_cash_breaching_ratio_facility_triggers_insolvency(self, caplog):
        """Cash below -(ratio-scaled limit) triggers Tier-1a insolvency + log."""
        import logging

        mfr = self._mfr(working_capital_facility_ratio=0.05)
        mfr.step()
        eff = mfr.effective_facility_limit()
        assert eff is not None
        self._overdraw_to(mfr, -(eff * to_decimal("1.5")))  # 150% of facility
        assert mfr.cash < -eff

        with caplog.at_level(logging.WARNING):
            result = mfr.check_solvency()

        assert result is False
        assert mfr.is_ruined
        assert any("LIQUIDITY INSOLVENCY" in record.message for record in caplog.records)

    def test_same_overdraft_breaches_small_facility_but_not_large(self):
        """Same -$2M overdraft, same 0.20 ratio: it breaches the small firm's
        revenue-scaled facility ($1.6M) but sits within the larger firm's
        ($3.2M) — the Tier-1a trigger scales with revenue (Issue #1625)."""
        overdraft = to_decimal(-2_000_000)  # drive cash to -$2M

        small = self._mfr(initial_assets=10_000_000, working_capital_facility_ratio=0.20)
        eff_small = small.effective_facility_limit()
        assert eff_small is not None
        assert eff_small == to_decimal(1_600_000)
        self._overdraw_to(small, overdraft)
        assert small.cash < -eff_small  # Tier-1a condition met
        assert small.check_solvency() is False
        assert small.is_ruined

        large = self._mfr(initial_assets=20_000_000, working_capital_facility_ratio=0.20)
        eff_large = large.effective_facility_limit()
        assert eff_large is not None
        assert eff_large == to_decimal(3_200_000)
        self._overdraw_to(large, overdraft)
        # The same overdraft is within the larger firm's facility, so the Tier-1a
        # facility-breach trigger does not fire (full solvency still depends on
        # the Tier-2 going-concern checks, which are out of scope for this test).
        assert large.cash >= -eff_large

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


class TestIntraYearFacilityFloor:
    """Intra-year liquidity check credits the working-capital facility (Issue #1631).

    ``check_liquidity_constraints()`` must floor the mid-year insolvency decision
    at ``-effective_facility_limit()`` (the shared :meth:`_liquidity_floor`), not
    at ``0``.  A temporary premium-driven cash trough the revolver would cover is
    not insolvency, and an unlimited facility (``None``) must never trigger a
    cash-floor mid-year insolvency.

    Before this fix the check ruined any firm whose projected min-cash dipped
    below ``0`` regardless of the facility.  Because the premium is the one
    material outflow only *insured* firms incur, that spuriously insolved insured
    firms and biased every insured-vs-uninsured comparison against coverage (the
    structural driver of the simple-vs-full verdict gap, #1604).
    """

    @staticmethod
    def _mfr(**overrides: Any) -> WidgetManufacturer:
        """Build a small $1M-asset firm with the intra-year check enabled."""
        params: dict[str, Any] = {
            "initial_assets": 1_000_000,
            "asset_turnover_ratio": 0.8,
            "base_operating_margin": 0.10,
            "tax_rate": 0.25,
            "retention_ratio": 0.7,
            "premium_payment_month": 0,
            "check_intra_period_liquidity": True,
            "capex_to_depreciation_ratio": 0.0,
        }
        params.update(overrides)
        return WidgetManufacturer(ManufacturerConfig(**params))

    @staticmethod
    def _set_cash(mfr: WidgetManufacturer, target: Decimal) -> None:
        """Drive cash to an exact target by moving the delta to/from receivables.

        Uses a working-capital move (Dr/Cr between CASH and ACCOUNTS_RECEIVABLE),
        not an expense, so equity is unchanged. Total assets — and any
        revenue-scaled facility — are preserved only while cash stays >= 0; once
        cash goes negative the overdraft is reclassified to short-term borrowings
        (ASC 470-10) so total assets shift, but equity (and the fixed facility
        used by the callers here) are unaffected.
        """
        delta = mfr.cash - target
        if delta > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.CASH,
                amount=delta,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Test: set cash to exact target",
            )
        elif delta < ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.ACCOUNTS_RECEIVABLE,
                amount=-delta,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Test: set cash to exact target",
            )

    # --- _liquidity_floor() resolution (single source of truth) ---

    def test_liquidity_floor_is_negative_effective_limit(self):
        """_liquidity_floor() == -effective_facility_limit() for fixed/ratio/None."""
        fixed = self._mfr(working_capital_facility_limit=750_000)
        assert fixed._liquidity_floor() == to_decimal(-750_000)

        ratio = self._mfr(working_capital_facility_ratio=0.20)
        assert ratio._liquidity_floor() == -(to_decimal(0.20) * ratio.calculate_revenue())

        unlimited = self._mfr()
        assert unlimited.effective_facility_limit() is None
        assert unlimited._liquidity_floor() is None

    # --- (a) premium trough negative but within facility -> survives ---

    def test_premium_trough_within_facility_survives(self):
        """A premium dip the revolver covers (>= -facility) is NOT insolvency."""
        mfr = self._mfr(working_capital_facility_limit=1_000_000)
        # Size the premium relative to current cash so the projected month-0
        # trough lands within (-facility, 0) regardless of balance-sheet
        # initialization: the trough is cash - (cash + 400k) + one-month revenue
        # = -400k + revenue/12, i.e. genuinely negative but well inside -$1M.
        mfr.period_insurance_premiums = mfr.cash + to_decimal(400_000)
        min_cash, _ = mfr.estimate_minimum_cash_point("annual")
        floor = mfr._liquidity_floor()
        assert floor == to_decimal(-1_000_000)
        assert min_cash < ZERO  # genuinely dips negative intra-year...
        assert min_cash >= floor  # ...but the revolver covers it
        assert mfr.check_liquidity_constraints("annual") is True
        assert not mfr.is_ruined
        assert mfr.ruin_month is None

    # --- (b) min-cash below -facility -> ruined mid-year with correct ruin_month ---

    def test_premium_trough_below_facility_ruins(self, caplog):
        """A trough beyond the facility (< -facility) IS mid-year insolvency."""
        import logging

        mfr = self._mfr(working_capital_facility_limit=100_000)
        mfr.period_insurance_premiums = to_decimal(500_000)
        min_cash, min_month = mfr.estimate_minimum_cash_point("annual")
        floor = mfr._liquidity_floor()
        assert floor is not None and min_cash < floor
        with caplog.at_level(logging.WARNING):
            result = mfr.check_liquidity_constraints("annual")
        assert result is False
        assert mfr.is_ruined
        assert mfr.ruin_month == min_month
        assert any("MID-YEAR INSOLVENCY" in r.message for r in caplog.records)

    def test_ratio_facility_floors_intra_year_check(self):
        """The intra-year floor also honors the revenue-scaled facility ratio."""
        mfr = self._mfr(working_capital_facility_ratio=0.05)  # 0.05 * $800k = $40k
        floor = mfr._liquidity_floor()
        assert floor == -(to_decimal(0.05) * mfr.calculate_revenue())
        mfr.period_insurance_premiums = to_decimal(500_000)  # trough far below -$40k
        assert mfr.check_liquidity_constraints("annual") is False
        assert mfr.is_ruined

    # --- (c) unlimited facility -> no cash-floor mid-year insolvency ---

    def test_unlimited_facility_no_cash_floor_insolvency(self):
        """With ratio and limit both None, no projected dip causes mid-year ruin."""
        mfr = self._mfr()  # both None (default / legacy unlimited)
        assert mfr.config.working_capital_facility_limit is None
        assert mfr.config.working_capital_facility_ratio is None
        assert mfr._liquidity_floor() is None
        mfr.period_insurance_premiums = to_decimal(5_000_000)  # absurd vs $1M firm
        min_cash, _ = mfr.estimate_minimum_cash_point("annual")
        assert min_cash < ZERO
        assert mfr.check_liquidity_constraints("annual") is True
        assert not mfr.is_ruined
        assert mfr.ruin_month is None

    def test_unlimited_facility_step_survives_premium_trough(self):
        """End-to-end: step() does not mid-year-ruin an unlimited-facility firm
        whose premium drives the projected intra-year trough below 0 (#1631).

        The premium is sized relative to current cash to guarantee a negative
        projected trough (which the OLD facility-blind ``< 0`` floor would have
        ruined). It is kept modest so equity stays positive, and the
        going-concern multi-factor trigger is set to require all four indicators
        (that path is exercised separately in ``test_going_concern.py``) — together
        isolating the intra-year liquidity path (the #1631 change) from the
        orthogonal equity- and going-concern-insolvency paths a small firm under a
        heavy premium would otherwise trip.
        """
        mfr = self._mfr(going_concern_min_indicators_breached=4)  # unlimited facility
        assert mfr._liquidity_floor() is None
        mfr.period_insurance_premiums = mfr.cash + to_decimal(400_000)
        min_cash, _ = mfr.estimate_minimum_cash_point("annual")
        assert min_cash < ZERO  # the OLD facility-blind < 0 floor would ruin here
        metrics = mfr.step()
        assert not mfr.is_ruined
        assert metrics.get("is_solvent") is True

    # --- (d) boundary: intra-year and post-payment checks agree at cash == -facility ---

    def test_boundary_intra_year_check_at_and_below_floor(self):
        """Intra-year: min-cash == -facility survives; one dollar below ruins."""
        at = self._mfr(working_capital_facility_limit=250_000)
        assert at._liquidity_floor() == to_decimal(-250_000)
        # min-cash exactly at the -$250k facility floor: strict < => survives.
        at.estimate_minimum_cash_point = lambda time_resolution="annual": (  # type: ignore[method-assign]
            to_decimal(-250_000),
            6,
        )
        assert at.check_liquidity_constraints("annual") is True
        assert not at.is_ruined

        below = self._mfr(working_capital_facility_limit=250_000)
        assert below._liquidity_floor() == to_decimal(-250_000)
        # one dollar below the -$250k facility floor => mid-year insolvency.
        below.estimate_minimum_cash_point = lambda time_resolution="annual": (  # type: ignore[method-assign]
            to_decimal(-250_001),
            6,
        )
        assert below.check_liquidity_constraints("annual") is False
        assert below.is_ruined
        assert below.ruin_month == 6

    def test_boundary_post_payment_check_at_and_below_floor(self):
        """Post-payment (check_solvency Tier 1a, same -facility floor as step()'s
        crisis check): cash == -facility is solvent; one dollar below is not.

        Uses a healthy $10M firm so the Tier-2 going-concern indicators stay
        quiet and the Tier-1a facility floor is isolated — the same boundary
        step()'s post-payment check applies, so the two checks agree at the
        floor (the single source of truth, :meth:`_liquidity_floor`).
        """

        def _big(**kw: Any) -> WidgetManufacturer:
            params: dict[str, Any] = {
                "initial_assets": 10_000_000,
                "asset_turnover_ratio": 0.8,
                "base_operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.7,
                "capex_to_depreciation_ratio": 0.0,
                "working_capital_facility_limit": 500_000,
            }
            params.update(kw)
            return WidgetManufacturer(ManufacturerConfig(**params))

        fac = to_decimal(500_000)

        at = _big()
        at.step()
        assert at._liquidity_floor() == -fac
        self._set_cash(at, -fac)
        assert at.cash == -fac
        assert at.check_solvency() is True
        assert not at.is_ruined

        below = _big()
        below.step()
        self._set_cash(below, -fac - to_decimal(1))
        assert below.cash == -fac - to_decimal(1)
        assert below.check_solvency() is False
        assert below.is_ruined


# ---------------------------------------------------------------------------
# Issue #1631: the same -effective_facility_limit() floor was applied to the
# other (off-the-production-path) cash-floor insolvency DECISIONS that the
# codebase sweep surfaced. The classes below verify the NEW facility-aware
# behavior of each fix (not merely the facility=0 boundary the legacy tests
# pin): record_insurance_premium(is_annual=True), record_prepaid_insurance, and
# update_balance_sheet()'s legacy loss-absorption liquidity check (Check 2).
# ---------------------------------------------------------------------------


def _make_firm(**overrides: Any) -> WidgetManufacturer:
    """Build a healthy $10M firm (ample equity headroom) for facility tests.

    The large equity base keeps the equity / going-concern insolvency paths
    quiet so each test isolates the cash-floor (facility) decision under test.
    """
    params: dict[str, Any] = {
        "initial_assets": 10_000_000,
        "asset_turnover_ratio": 0.8,
        "base_operating_margin": 0.08,
        "tax_rate": 0.25,
        "retention_ratio": 0.7,
        "capex_to_depreciation_ratio": 0.0,
    }
    params.update(overrides)
    return WidgetManufacturer(ManufacturerConfig(**params))


def _drive_cash(mfr: WidgetManufacturer, target: Decimal) -> None:
    """Drive cash to an exact non-negative target via a CASH<->AR move.

    A working-capital reclassification (not an expense) leaves equity unchanged,
    so only the cash position — the quantity the facility floor acts on — moves.
    """
    delta = mfr.cash - target
    if delta > ZERO:
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.CASH,
            amount=delta,
            transaction_type=TransactionType.WORKING_CAPITAL,
            description="Test: drive cash to target",
        )
    elif delta < ZERO:
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.CASH,
            credit_account=AccountName.ACCOUNTS_RECEIVABLE,
            amount=-delta,
            transaction_type=TransactionType.WORKING_CAPITAL,
            description="Test: drive cash to target",
        )


class TestPremiumAffordabilityFacilityFloor:
    """Compulsory-premium affordability checks credit the facility (Issue #1631).

    ``record_insurance_premium(is_annual=True)`` and ``record_prepaid_insurance``
    pay an annual premium up front. Post-#1631 they declare insolvency only when
    the premium exceeds cash PLUS the working-capital facility
    (``cash - premium < -facility``); an unlimited facility (``None``) never
    triggers the check. Before the fix they ruined any firm with
    ``cash < premium`` regardless of the revolver — the dead-path equivalents of
    the live intra-year bug. These paths are off the production simulation path
    (no caller passes ``is_annual=True``; ``record_prepaid_insurance`` has no
    caller), but the floor is now consistent should they ever be rewired.
    """

    def test_is_annual_premium_within_facility_survives(self):
        """is_annual=True: a premium the revolver covers is paid, not insolvency."""
        mfr = _make_firm(working_capital_facility_limit=200_000)
        _drive_cash(mfr, to_decimal(50_000))  # cash 50k + facility 200k >= 100k premium
        mfr.record_insurance_premium(100_000, is_annual=True)
        assert not mfr.is_ruined
        assert mfr._original_prepaid_premium == to_decimal(100_000)

    def test_is_annual_premium_unlimited_facility_survives(self):
        """is_annual=True with an unlimited facility (None) never cash-floor-ruins."""
        mfr = _make_firm()  # facility None => unlimited revolver
        assert mfr._liquidity_floor() is None
        _drive_cash(mfr, to_decimal(50_000))
        mfr.record_insurance_premium(5_000_000, is_annual=True)
        assert not mfr.is_ruined

    def test_is_annual_premium_beyond_facility_ruins(self):
        """is_annual=True: a premium exceeding cash + facility IS insolvency."""
        mfr = _make_firm(working_capital_facility_limit=100_000)
        _drive_cash(mfr, to_decimal(50_000))  # 50k + 100k < 1M premium
        mfr.record_insurance_premium(1_000_000, is_annual=True)
        assert mfr.is_ruined

    def test_prepaid_premium_within_facility_survives(self):
        """record_prepaid_insurance: a premium the revolver covers is not insolvency."""
        mfr = _make_firm(working_capital_facility_limit=200_000)
        _drive_cash(mfr, to_decimal(50_000))
        mfr.record_prepaid_insurance(100_000)
        assert not mfr.is_ruined

    def test_prepaid_premium_unlimited_facility_survives(self):
        """record_prepaid_insurance with an unlimited facility (None) never cash-floor-ruins."""
        mfr = _make_firm()
        assert mfr._liquidity_floor() is None
        _drive_cash(mfr, to_decimal(50_000))
        mfr.record_prepaid_insurance(5_000_000)
        assert not mfr.is_ruined

    def test_prepaid_premium_beyond_facility_ruins(self):
        """record_prepaid_insurance: a premium exceeding cash + facility IS insolvency."""
        mfr = _make_firm(working_capital_facility_limit=100_000)
        _drive_cash(mfr, to_decimal(50_000))
        mfr.record_prepaid_insurance(1_000_000)
        assert mfr.is_ruined


class TestLossAbsorptionFacilityFloor:
    """Loss-absorption liquidity Check 2 credits the facility (Issue #1631).

    ``update_balance_sheet()``'s legacy (non-closing-entry) loss-absorption path
    declares a liquidity crisis only when the cash drain exceeds cash PLUS the
    working-capital facility (``cash - cash_consumed < -facility``); an unlimited
    facility (``None``) never triggers it. Before the fix it ruined any firm
    whose loss exceeded cash, ignoring the revolver. The production ``step()``
    path uses closing entries and never reaches this branch, so these tests
    exercise the legacy path directly by calling ``update_balance_sheet`` with no
    ``period_revenue``. A $10M equity base keeps the Check 1 / Check 3 equity
    insolvency tests quiet so the liquidity floor is isolated.
    """

    def test_loss_within_facility_absorbed(self):
        """A loss exceeding cash but within cash + facility is absorbed, not ruin."""
        mfr = _make_firm(working_capital_facility_limit=2_000_000)
        _drive_cash(mfr, to_decimal(500_000))
        assert mfr.equity > to_decimal(5_000_000)  # ample equity headroom
        mfr.update_balance_sheet(to_decimal(-1_000_000))  # 500k cash + 2M facility >= 1M loss
        assert not mfr.is_ruined

    def test_loss_unlimited_facility_absorbed(self):
        """With an unlimited facility (None) a cash-exceeding loss never cash-floor-ruins."""
        mfr = _make_firm()  # facility None
        assert mfr._liquidity_floor() is None
        _drive_cash(mfr, to_decimal(500_000))
        mfr.update_balance_sheet(to_decimal(-1_000_000))  # beyond cash, within equity
        assert not mfr.is_ruined

    def test_loss_beyond_facility_ruins(self):
        """A loss exceeding cash + facility IS a liquidity-crisis insolvency."""
        mfr = _make_firm(working_capital_facility_limit=100_000)
        _drive_cash(mfr, to_decimal(500_000))  # 500k + 100k < 2M loss
        mfr.update_balance_sheet(to_decimal(-2_000_000))
        assert mfr.is_ruined

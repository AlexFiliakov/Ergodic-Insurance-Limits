"""Tests for the proximate-cause ``ruin_reason`` instrumentation on WidgetManufacturer.

Each distinct insolvency trigger site records a short machine code on
``mfr.ruin_reason`` (and optional ``mfr.ruin_detail``) at the moment it fires,
so the ruin-cause decomposition diagnostic (notebook 07, Part 10b) can attribute
*why* each simulated firm failed. The instrumentation is additive and diagnostic
only -- it must not change any solvency verdict.

These tests reuse the insolvency-forcing recipes from
``test_facility_limit_insolvency.py`` and assert the recorded reason for each
channel:

    facility_breach       check_solvency Tier 1a  (cash below -facility at year end)
    equity_insolvency     check_solvency Tier 1b  (operational equity <= 0)
    going_concern         check_solvency Tier 2   (>= N going-concern indicators)
    midyear_liquidity     check_liquidity_constraints (premium-timing cash trough)
    operating_liquidity   update_balance_sheet Check 2 (loss cash-drain > cash+facility)
    operating_equity      update_balance_sheet Check 3 (loss pushes equity <= 0)
    premium_unaffordable  record_insurance_premium / record_prepaid_insurance
"""

from decimal import Decimal
from typing import Any

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


def _firm(**overrides: Any) -> WidgetManufacturer:
    """Build a healthy $10M-asset / $8M-revenue manufacturer for these tests."""
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


def _move_cash_to_ar(mfr: WidgetManufacturer, amount: Decimal) -> None:
    """Tie up ``amount`` of cash in receivables (equity unchanged)."""
    if amount > ZERO:
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.ACCOUNTS_RECEIVABLE,
            credit_account=AccountName.CASH,
            amount=amount,
            transaction_type=TransactionType.WORKING_CAPITAL,
            description="Test: tie up cash in receivables",
        )


class TestRuinReasonDefaults:
    """A solvent firm carries no ruin reason; reset() clears it."""

    def test_fresh_firm_has_no_ruin_reason(self):
        mfr = _firm()
        assert mfr.is_ruined is False
        assert mfr.ruin_reason is None
        assert mfr.ruin_detail is None

    def test_reset_clears_ruin_reason(self):
        mfr = _firm(working_capital_facility_limit=100_000)
        mfr.step()
        _move_cash_to_ar(mfr, mfr.cash + to_decimal(700_000))  # cash -> -$700K, beyond $100K
        mfr.check_solvency()
        # Capture pre-reset state in locals: asserting on mfr.* here would narrow
        # those attributes (e.g. is_ruined -> Literal[True]) and the narrowing
        # persists across the reset() call, making the post-reset asserts below
        # unreachable to mypy (warn_unreachable). Locals keep mfr.* un-narrowed.
        ruined_before = bool(mfr.is_ruined)
        reason_before = mfr.ruin_reason
        assert ruined_before and reason_before is not None
        mfr.reset()
        assert mfr.is_ruined is False
        assert mfr.ruin_reason is None
        assert mfr.ruin_detail is None


class TestStructuralRuinReasons:
    """Each insolvency trigger records its own proximate-cause code."""

    def test_facility_breach(self):
        """Tier 1a: cash below -(facility) at the solvency check."""
        mfr = _firm(working_capital_facility_limit=500_000)
        mfr.step()
        _move_cash_to_ar(mfr, mfr.cash + to_decimal(700_000))  # cash -> -$700K < -$500K
        assert mfr.cash < to_decimal(-500_000)
        assert mfr.check_solvency() is False
        assert mfr.is_ruined
        assert mfr.ruin_reason == "facility_breach"

    def test_equity_insolvency(self):
        """Tier 1b: operational equity <= 0 (unlimited facility, so Tier 1a is skipped)."""
        mfr = _firm()  # facility None -> Tier 1a skipped
        # Write equity down below zero (Dr Retained Earnings / Cr Cash). With an
        # unlimited facility the resulting negative cash does NOT trip Tier 1a, so
        # the equity hard-stop (Tier 1b) is isolated.
        writedown = mfr.equity + to_decimal(1_000_000)
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.RETAINED_EARNINGS,
            credit_account=AccountName.CASH,
            amount=writedown,
            transaction_type=TransactionType.EXPENSE,
            description="Test: drive equity below zero",
        )
        assert mfr.solvency_equity <= ZERO
        assert mfr.check_solvency() is False
        assert mfr.is_ruined
        assert mfr.ruin_reason == "equity_insolvency"

    def test_going_concern(self):
        """Tier 2: a near-zero cash runway breaches a going-concern indicator.

        Equity stays healthy (>0, ratio ~100%) so Tier 1b does not fire; cash is
        tied up in receivables so the cash-runway indicator breaches. With the
        trigger threshold set to a single indicator, this isolates Tier 2.
        """
        mfr = _firm(going_concern_min_indicators_breached=1)
        _move_cash_to_ar(mfr, mfr.cash - to_decimal(1_000))  # leave ~$1K cash
        assert mfr.solvency_equity > ZERO  # Tier 1b stays quiet
        assert mfr.check_solvency() is False
        assert mfr.is_ruined
        assert mfr.ruin_reason == "going_concern"
        assert mfr.ruin_detail and "Cash Runway" in mfr.ruin_detail

    def test_midyear_liquidity_premium(self):
        """check_liquidity_constraints: a premium-timing trough beyond the facility."""
        mfr = _firm(
            initial_assets=1_000_000,
            base_operating_margin=0.10,
            premium_payment_month=0,
            check_intra_period_liquidity=True,
            working_capital_facility_limit=100_000,
        )
        mfr.period_insurance_premiums = to_decimal(500_000)  # trough far below -$100K
        assert mfr.check_liquidity_constraints("annual") is False
        assert mfr.is_ruined
        assert mfr.ruin_reason == "midyear_liquidity"
        assert mfr.ruin_month is not None

    def test_operating_liquidity_loss_cash_drain(self):
        """update_balance_sheet Check 2: loss cash-drain exceeds cash + facility."""
        mfr = _firm(working_capital_facility_limit=100_000)
        _move_cash_to_ar(mfr, mfr.cash - to_decimal(500_000))  # leave $500K cash
        assert mfr.equity > to_decimal(5_000_000)  # ample equity -> Check 3 stays quiet
        mfr.update_balance_sheet(to_decimal(-2_000_000))  # $2M drain > $500K + $100K
        assert mfr.is_ruined
        assert mfr.ruin_reason == "operating_liquidity"

    def test_operating_equity_loss(self):
        """update_balance_sheet Check 3: loss within liquidity but pushes equity <= 0."""
        # Small-equity firm with a large facility: the loss is fundable (Check 2
        # passes) but exceeds equity (Check 3 fires).
        mfr = _firm(initial_assets=1_000_000, working_capital_facility_limit=5_000_000)
        assert mfr.equity < to_decimal(2_000_000)
        mfr.update_balance_sheet(to_decimal(-2_000_000))  # > equity, < cash + $5M facility
        assert mfr.is_ruined
        assert mfr.ruin_reason == "operating_equity"

    def test_premium_unaffordable_annual(self):
        """record_insurance_premium(is_annual=True): premium exceeds cash + facility."""
        mfr = _firm(working_capital_facility_limit=100_000)
        _move_cash_to_ar(mfr, mfr.cash - to_decimal(50_000))  # leave $50K cash
        mfr.record_insurance_premium(1_000_000, is_annual=True)  # $1M > $50K + $100K
        assert mfr.is_ruined
        assert mfr.ruin_reason == "premium_unaffordable"

    def test_premium_unaffordable_prepaid(self):
        """record_prepaid_insurance: premium exceeds cash + facility."""
        mfr = _firm(working_capital_facility_limit=100_000)
        _move_cash_to_ar(mfr, mfr.cash - to_decimal(50_000))
        mfr.record_prepaid_insurance(1_000_000)
        assert mfr.is_ruined
        assert mfr.ruin_reason == "premium_unaffordable"


class TestRuinReasonFirstWins:
    """The first trigger to fire in a lifetime is the one recorded."""

    def test_reason_not_overwritten_by_later_checks(self):
        """A pre-recorded reason survives a subsequent handle_insolvency() default."""
        mfr = _firm(working_capital_facility_limit=100_000)
        _move_cash_to_ar(mfr, mfr.cash - to_decimal(50_000))
        mfr.record_insurance_premium(1_000_000, is_annual=True)
        assert mfr.ruin_reason == "premium_unaffordable"
        # A later solvency check must not relabel the cause.
        mfr.check_solvency()
        assert mfr.ruin_reason == "premium_unaffordable"

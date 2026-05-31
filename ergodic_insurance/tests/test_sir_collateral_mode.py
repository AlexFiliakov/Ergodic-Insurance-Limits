"""Tests for the SIR collateral model (Issues #1637, #1644).

The default ``sir_collateral_mode`` is now ``"letter_of_credit"``: a retained
deductible/SIR portion of an insured claim is backed by a letter of credit, NOT a cash
lock. It is booked as a deferred claim liability paid from operating cash over the
development schedule (symmetric with an uninsured loss), and a single LOC carry fee accrues
on the OUTSTANDING reserve. The legacy ``"cash"`` model (cash-collateralized to
RESTRICTED_CASH, paid from restricted cash, LOC fee on the restricted balance) remains
available as an opt-in.

These tests pin the two core guarantees:
  * #1637: an insured retention no longer locks more near-term cash than the same loss
    carried uninsured (the liquidity-asymmetry fix).
  * #1644: a retained loss carries exactly ONE collateral cost (the LOC fee on the reserve
    in the default model, OR the cash lock in the legacy model) -- never both.
"""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import to_decimal
from ergodic_insurance.manufacturer import WidgetManufacturer

LAE_RATIO = 0.12  # ManufacturerConfig.lae_ratio default
LOC_RATE = 0.015


def _make(mode="letter_of_credit"):
    """A cash-rich manufacturer so retained claims are not equity/cash constrained."""
    return WidgetManufacturer(
        ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.50,
            ppe_ratio=0.10,  # leaves ample cash on day 1
            sir_collateral_mode=mode,
        )
    )


class TestDefaultIsLetterOfCredit:
    """The default collateral model is the letter-of-credit model."""

    def test_default_mode(self):
        cfg = ManufacturerConfig(initial_assets=10_000_000, asset_turnover_ratio=1.0)
        assert cfg.sir_collateral_mode == "letter_of_credit"

    def test_loc_claim_does_not_lock_cash(self):
        """A retained SIR under the LOC model locks NO cash (Issue #1637)."""
        m = _make("letter_of_credit")
        cash_before = m.cash
        company, insurance = m.process_insurance_claim(
            claim_amount=1_000_000, deductible_amount=300_000, insurance_limit=5_000_000
        )
        assert company == to_decimal(300_000)
        assert insurance == to_decimal(700_000)
        # No cash moved to restricted, and cash is unchanged at claim time.
        assert m.restricted_assets == to_decimal(0)
        assert m.collateral == to_decimal(0)
        assert m.cash == cash_before
        # The retention is booked as a deferred, LOC-collateralized claim liability.
        loc_claims = [c for c in m.claim_liabilities if c.is_loc_collateralized]
        assert len(loc_claims) == 1
        assert loc_claims[0].is_insured is True
        # Reserve = indemnity + LAE.
        expected_reserve = to_decimal(300_000) * to_decimal(1 + LAE_RATIO)
        assert m.loc_collateralized_reserve == pytest.approx(float(expected_reserve))

    def test_loc_fee_charged_on_reserve(self):
        """The single LOC carry fee accrues on the outstanding reserve, not on cash."""
        m = _make("letter_of_credit")
        m.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        reserve = m.loc_collateralized_reserve
        fee = m.calculate_collateral_costs(letter_of_credit_rate=LOC_RATE, time_period="annual")
        assert fee == pytest.approx(float(reserve) * LOC_RATE)
        assert fee > 0


class TestLegacyCashMode:
    """The opt-in cash-trust model preserves the historical behavior."""

    def test_cash_claim_locks_cash(self):
        m = _make("cash")
        cash_before = m.cash
        company, _ = m.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        assert company == to_decimal(300_000)
        # Cash IS locked to restricted; no LOC reserve.
        assert m.restricted_assets == to_decimal(300_000)
        assert m.collateral == to_decimal(300_000)
        assert m.cash == cash_before - to_decimal(300_000)
        assert m.loc_collateralized_reserve == to_decimal(0)

    def test_cash_fee_charged_on_restricted(self):
        m = _make("cash")
        m.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        fee = m.calculate_collateral_costs(letter_of_credit_rate=LOC_RATE)
        assert fee == pytest.approx(300_000 * LOC_RATE)


class TestLiquidityAsymmetryFix:
    """#1637: an insured retention no longer drains more near-term cash than uninsured."""

    def test_loc_no_worse_than_uninsured(self):
        """Insured-retained (LOC) near-term cash >= the same loss carried uninsured."""
        retained = 400_000
        # Insured: deductible == retained, insurer pays the rest (so company keeps `retained`).
        insured = _make("letter_of_credit")
        insured.process_insurance_claim(1_000_000, retained, 5_000_000)
        # Uninsured: company retains the whole loss, deferred (the notebook's default call).
        uninsured = _make("letter_of_credit")
        uninsured.process_uninsured_claim(retained, immediate_payment=False)
        # Neither locks cash at claim time; the insured firm is NOT worse off on liquidity.
        assert insured.cash >= uninsured.cash
        assert float(insured.cash) == pytest.approx(float(uninsured.cash))
        assert insured.restricted_assets == to_decimal(0)

    def test_cash_mode_is_strictly_worse_on_liquidity(self):
        """The legacy cash model locks cash, leaving strictly less than uninsured -- the bug."""
        retained = 400_000
        cash_insured = _make("cash")
        cash_insured.process_insurance_claim(1_000_000, retained, 5_000_000)
        uninsured = _make("cash")
        uninsured.process_uninsured_claim(retained, immediate_payment=False)
        assert cash_insured.cash < uninsured.cash

    def test_equity_hit_is_mode_independent(self):
        """Both models recognize the same retained-loss liability; only cash location differs."""
        loc = _make("letter_of_credit")
        cash = _make("cash")
        loc.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        cash.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        # Equity (assets - liabilities) is the same in both modes for an ample-cash firm.
        assert loc.equity == pytest.approx(float(cash.equity))
        assert loc.total_claim_liabilities == pytest.approx(float(cash.total_claim_liabilities))


class TestSingleCollateralCost:
    """#1644: a retained loss carries exactly one collateral cost, never two."""

    def test_loc_mode_single_cost_no_cash_lock(self):
        """LOC model: the LOC fee is the only cost; cash is NOT also locked."""
        m = _make("letter_of_credit")
        m.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        # Exactly one fee basis is non-zero: the LOC reserve, not restricted cash.
        assert m.collateral == to_decimal(0)  # no restricted-cash lock
        assert m.loc_collateralized_reserve > to_decimal(0)  # LOC reserve carries the fee

    def test_cash_mode_single_basis(self):
        """Cash model: restricted cash is the only fee basis; no separate LOC reserve."""
        m = _make("cash")
        m.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        assert m.loc_collateralized_reserve == to_decimal(0)
        assert m.collateral > to_decimal(0)


class TestSymmetricPaydownTiming:
    """The LOC retention pays down from CASH on the development schedule (like uninsured)."""

    def test_loc_reserve_pays_from_cash_over_schedule(self):
        m = _make("letter_of_credit")
        m.process_insurance_claim(1_000_000, 300_000, 5_000_000)
        reserve0 = m.loc_collateralized_reserve
        assert m.restricted_assets == to_decimal(0)
        # Step several years; the reserve should run off and restricted cash stays zero
        # (payments come from operating cash, not a restricted lock).
        for _ in range(3):
            m.step(letter_of_credit_rate=LOC_RATE)
        assert m.restricted_assets == to_decimal(0)
        assert m.loc_collateralized_reserve < reserve0  # reserve is being paid down


@pytest.mark.parametrize("mode", ["letter_of_credit", "cash"])
def test_claim_split_is_mode_independent(mode):
    """The company/insurer split of a claim does not depend on the collateral model."""
    m = _make(mode)
    company, insurance = m.process_insurance_claim(2_000_000, 100_000, 1_000_000)
    # Company pays deductible + excess over the limit; insurer pays the limit.
    assert company == to_decimal(100_000 + (2_000_000 - 100_000 - 1_000_000))
    assert insurance == to_decimal(1_000_000)

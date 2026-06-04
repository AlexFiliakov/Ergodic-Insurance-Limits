"""Regression tests locking in the notebook-07 simple-vs-full no-loss basis reconciliation (#1648).

Notebook ``07_hjb_insurance_optimization.ipynb`` runs the same insurance experiment through a
transparent simple-cash model (Parts 5/8/10/11, the basis the HJB is solved on) and the full GAAP
:class:`~ergodic_insurance.manufacturer.WidgetManufacturer` (Parts 10b/10c/10d). Issue #1648 closed a
+2.87 pp/yr no-loss equity-growth gap between them, which a decomposition showed to be ~96% pure
**leverage amplification**: operating income is earned on the whole asset base but retained earnings
accrue entirely to equity, so no-loss equity compounds at the leverage-amplified rate
``PHI * ATR * margin / (1 - BASE_LIAB_RATIO)`` rather than the un-amplified ``PHI * ATR * margin``.

These tests lock that relationship in so a future change to the manufacturer (or to the simple-model
drift the notebook calibrates to it) cannot silently re-open the gap.
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.stochastic_processes import StochasticConfig, StochasticProcess

# Notebook-07 calibration (cells 5 / 10 / 42).
ATR = 1.5
OPERATING_MARGIN = 0.125
TAX_RATE = 0.25
RETENTION_RATIO = 0.70
PHI_RETENTION = RETENTION_RATIO * (1.0 - TAX_RATE)  # 0.525 (after tax + dividends)
BASE_LIAB_RATIO = 0.222
INITIAL_ASSETS = 5_000_000.0
N_YEARS = 25

# Equity drift, un-amplified vs leverage-amplified.
DRIFT_ASSETS = PHI_RETENTION * ATR * OPERATING_MARGIN  # ~0.0984 (the OLD simple-model no-loss CAGR)
DRIFT_EQUITY = DRIFT_ASSETS / (1.0 - BASE_LIAB_RATIO)  # ~0.1265 (#1648 leverage-amplified)


class _UnitShock(StochasticProcess):
    """Deterministic stochastic process whose every shock multiplier is exactly 1.0.

    Lets a no-loss WidgetManufacturer run advance on its mean operating path (no volatility drag),
    so the measured equity CAGR is the model's drift, not a noisy realization.
    """

    def __init__(self) -> None:
        super().__init__(StochasticConfig(volatility=0.30, drift=0.0))

    def generate_shock(self, current_value: float) -> float:
        """Return a unit (no-op) multiplier regardless of the current value."""
        return 1.0

    def reset(self, seed=None) -> None:
        """No-op reset (the process is deterministic)."""


def _notebook_manufacturer() -> WidgetManufacturer:
    """Build the WidgetManufacturer exactly as notebook 07 does (cells 16 / 42)."""
    cfg = ManufacturerConfig(
        initial_assets=INITIAL_ASSETS,
        asset_turnover_ratio=ATR,
        base_operating_margin=OPERATING_MARGIN,
        tax_rate=TAX_RATE,
        retention_ratio=RETENTION_RATIO,
        ppe_ratio=0.25,
        working_capital_facility_ratio=0.20,
        sir_collateral_mode="letter_of_credit",
        initial_base_liability_ratio=BASE_LIAB_RATIO,  # #1645 warm-up -> start at e = 1 - k = 0.778
    )
    mfr = WidgetManufacturer(cfg)
    mfr.stochastic_process = _UnitShock()
    return mfr


def test_simple_model_no_loss_drift_is_leverage_amplified():
    """The notebook simple-model no-loss equity step compounds at PHI*ATR*margin/(1-k) (#1648).

    Reproduces the one-line ``simple_step_2state`` update (cell 10) after the #1648 fix: retained
    earnings ``PHI*(operating_income - premium)`` are divided by ``(1 - BASE_LIAB_RATIO)`` before
    accruing, so equity (= ``(1 - k) * A``) compounds at the leverage-amplified rate.
    """
    assets = INITIAL_ASSETS
    for _ in range(N_YEARS):
        operating_income = assets * ATR * OPERATING_MARGIN  # E[shock] = 1
        # cell-10 simple_step_2state, no-loss / no-premium, after the #1648 /(1-k) fix:
        assets = assets + PHI_RETENTION * (operating_income - 0.0) / (1.0 - BASE_LIAB_RATIO)
    start_equity = INITIAL_ASSETS * (1.0 - BASE_LIAB_RATIO)
    end_equity = assets * (1.0 - BASE_LIAB_RATIO)
    cagr = (end_equity / start_equity) ** (1.0 / N_YEARS) - 1.0
    assert cagr == pytest.approx(DRIFT_EQUITY, abs=1e-9)
    # ...and it is genuinely amplified above the old un-amplified PHI*ATR*margin.
    assert cagr > DRIFT_ASSETS + 0.02


def test_full_manufacturer_no_loss_equity_cagr_reconciles():
    """A no-loss WidgetManufacturer compounds equity at ~PHI*ATR*margin/(1-k), the #1648 target.

    This is the full GAAP basis the simple model is reconciled to: equity growth is leverage-amplified
    (operating income earned on assets, retained to equity), so the 25-year no-loss equity CAGR must
    land near the leverage-amplified drift and well above the un-amplified ``PHI*ATR*margin``.
    """
    mfr = _notebook_manufacturer()
    start_equity = INITIAL_ASSETS * (1.0 - BASE_LIAB_RATIO)  # warmed-up t=0 equity (e = 0.778)
    # Sanity: the #1645 warm-up really does start the firm near e = 1 - BASE_LIAB_RATIO.
    e0 = float(mfr.equity) / float(mfr.total_assets)
    assert e0 == pytest.approx(1.0 - BASE_LIAB_RATIO, abs=0.02)

    for _ in range(N_YEARS):
        mfr.step(letter_of_credit_rate=0.015, growth_rate=0.0, apply_stochastic=True)

    end_equity = float(mfr.equity)
    cagr = (end_equity / start_equity) ** (1.0 / N_YEARS) - 1.0

    # Leverage-amplified, well clear of the un-amplified rate (a regression to ~9.8% would fail).
    assert cagr > DRIFT_ASSETS + 0.015
    # ...and within ~1 pp of the leverage-amplified prediction (de-lever + DTL/depreciation are
    # second-order, ~0.2 pp; the deterministic reference run measures ~12.86%/yr).
    assert cagr == pytest.approx(DRIFT_EQUITY, abs=0.01)


def test_full_manufacturer_equity_returns_decimal_and_grows():
    """Guardrail: the no-loss firm stays solvent and its equity strictly grows each measured step."""
    mfr = _notebook_manufacturer()
    assert isinstance(mfr.equity, Decimal)
    prev = float(mfr.equity)
    for _ in range(N_YEARS):
        mfr.step(letter_of_credit_rate=0.015, growth_rate=0.0, apply_stochastic=True)
        cur = float(mfr.equity)
        assert not mfr.is_ruined
        assert cur > prev  # monotone growth in the no-loss case
        prev = cur

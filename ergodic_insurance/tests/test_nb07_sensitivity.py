"""Contract + invariant tests for notebook-07's Part 9 sensitivity section and the DPI=150 charts.

Notebook ``07_hjb_insurance_optimization.ipynb`` gained (issues #1650, #1641):

* **Part 9** -- a fixed-deployed-policy sensitivity sweep: a simple-model tornado of the regret-optimal
  retention and the insurance :math:`g_{CE}` advantage across :math:`\\pm 20\\%` perturbations of seven
  inputs (9a), a full-GAAP-modeler spot-check of the highest-leverage knobs at the shipped HJB policy
  with a paired SE (9b), and a dedicated premium-loading sweep (9c).
* **DPI = 150** on every chart.

The heavy cells are Colab-only (the 2-D HJB solve + 1000-path walk-forwards), so these tests do NOT
execute the notebook.  They instead (a) lock the notebook *contract* -- the section's structure, the
baseline-restore guard that keeps the downstream cells running on the shipped configuration, and the
DPI requirement -- and (b) reproduce the one piece of genuinely unit-testable numerical logic, the
Wang-distortion premium loading that the premium-loading sweep (#1641) is built on, against the real
library severity classes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

from ergodic_insurance.insurance_pricing import LayerPricer
from ergodic_insurance.loss_distributions import LognormalLoss, ParetoLoss

NB_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "optimization"
    / "07_hjb_insurance_optimization.ipynb"
)


# --------------------------------------------------------------------------------------------------
# Notebook loading helpers
# --------------------------------------------------------------------------------------------------
def _load_cells():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return nb["cells"]


def _code_sources(cells):
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def _markdown_sources(cells):
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "markdown"]


def _is_ipython_magic(src: str) -> bool:
    """A cell using line magics / shell escapes (e.g. the Colab `!pip install`) is not Python AST."""
    return any(line.lstrip().startswith(("!", "%")) for line in src.splitlines())


# --------------------------------------------------------------------------------------------------
# DPI = 150 on every chart (the explicit request accompanying #1650/#1641)
# --------------------------------------------------------------------------------------------------
def test_no_chart_uses_the_old_dpi_72():
    """No figure may keep the old dpi=72 anywhere in the notebook."""
    for src in _code_sources(_load_cells()):
        assert "dpi=72" not in src


def test_every_figure_creation_sets_dpi_150():
    """Every ``plt.subplots(`` / ``plt.figure(`` call sets ``dpi=150`` (add_subplot inherits it)."""
    offenders = []
    for src in _code_sources(_load_cells()):
        for line in src.splitlines():
            if ("plt.subplots(" in line) or ("plt.figure(" in line):
                if "dpi=150" not in line:
                    offenders.append(line.strip())
    assert not offenders, f"figure-creation calls missing dpi=150: {offenders}"


# --------------------------------------------------------------------------------------------------
# Part 9 structure / contract (#1650, #1641)
# --------------------------------------------------------------------------------------------------
def test_part9_anchors_and_titles_present():
    """The three Part-9 sub-sections exist with their anchors and headings."""
    md = "\n\n".join(_markdown_sources(_load_cells()))
    assert '<a id="part9"></a>' in md
    assert '<a id="part9c"></a>' in md
    assert '<a id="part9-synthesis"></a>' in md
    assert "Sensitivity of the Insurance-Value Verdict" in md
    assert "Premium-Loading Sensitivity" in md


def test_part9_no_longer_omitted():
    """The old placeholder ('omitted under the 2-D HJB') must be gone."""
    code = "\n".join(_code_sources(_load_cells()))
    md = "\n".join(_markdown_sources(_load_cells()))
    assert "loss-assumption sensitivity sweep) is omitted" not in code
    assert "Loss-Assumption Sensitivity (omitted under the 2-D HJB)" not in md


def test_part9_sweeps_all_required_knobs():
    """9a sweeps at least the seven knobs the acceptance criteria name."""
    code = "\n".join(_code_sources(_load_cells()))
    for kwarg in ("gamma", "lr", "cat_alpha", "cat_freq", "op_margin", "tax", "retention"):
        assert f'"{kwarg}"' in code, f"knob {kwarg!r} missing from the Part-9 sweep"
    assert "PERTURB = 0.20" in code  # +/-20% perturbation (issue #1650)


def test_part9_holds_the_deployed_policy_fixed_no_resolve():
    """The section documents that it holds the policy fixed and does NOT re-solve the HJB."""
    text = "\n".join(_code_sources(_load_cells()) + _markdown_sources(_load_cells())).lower()
    assert "no hjb re-solve" in text or "no re-solve" in text or "without re-solving" in text
    assert "fixed deployed policy" in text or "deployed policy" in text


def test_part9_restores_baseline_globals():
    """Each sweep cell must restore the shipped baseline in a finally block, so the downstream
    Parts 10/10b/10d/11 run on the shipped configuration (the critical correctness guard)."""
    sweep_cells = [
        s
        for s in _code_sources(_load_cells())
        if "_sens_apply_params" in s and "ProcessPoolExecutor" in s
    ]
    assert sweep_cells, "no Part-9 sweep cell found"
    for s in sweep_cells:
        assert "finally:" in s, "sweep cell lacks a try/finally"
        assert "_sens_restore_baseline()" in s, "sweep cell does not restore the baseline globals"
    # the helper itself must rebind every perturbed global back to the snapshot
    helper_cell = next(s for s in sweep_cells if "def _sens_restore_baseline" in s)
    for g in (
        "GAMMA_SEV",
        "TARGET_LOSS_RATIO",
        "WANG_LAMBDA",
        "OPERATING_MARGIN",
        "TAX_RATE",
        "RETENTION_RATIO",
        "PHI_RETENTION",
        "CAT_SEV_ALPHA",
        "CAT_BASE_FREQ",
        "LOSS_PARAMS",
    ):
        assert f'_SENS_BASELINE["{g}"]' in helper_cell, f"restore does not reset {g}"


def test_part9c_premium_loading_sweep_present():
    """9c sweeps the implied loss ratio in both models and marks the shipped loading (#1641)."""
    code = "\n".join(_code_sources(_load_cells()))
    md = "\n".join(_markdown_sources(_load_cells()))
    assert "LR_GRID = [" in code
    assert "SHIPPED_LR" in code
    assert "growth-drag" in md.lower() or "growth drag" in md.lower()  # #1641 tradeoff prose


def test_part9_code_cells_parse():
    """Every Part-9 (and indeed every non-magic) code cell is valid Python."""
    for src in _code_sources(_load_cells()):
        if _is_ipython_magic(src):
            continue
        ast.parse(src)


def test_key_takeaways_point6_updated():
    """Key Takeaways point 6 now reflects that sensitivity IS assessed (at a fixed policy)."""
    md = "\n".join(_markdown_sources(_load_cells()))
    assert "Loss-model and parameter sensitivity were not swept" not in md  # old wording gone
    assert "now assessed at a fixed deployed policy" in md


# --------------------------------------------------------------------------------------------------
# Pure-logic invariant: the Wang-distortion premium loading the #1641 sweep is built on
# --------------------------------------------------------------------------------------------------
# Notebook-07 baseline loss model (cell 10) -- the reference-size tower the loading anchors to.
_ATR = 1.5
_INITIAL_ASSETS = 5_000_000.0
_REFERENCE_REVENUE = _ATR * _INITIAL_ASSETS
_REF_PRIMARY_ATTACH = 250_000
_REF_PRIMARY_LIMIT = 5_000_000 - 250_000


def _reference_pricers():
    """The 3-component (attritional, large, catastrophic) tower at REFERENCE_REVENUE (cell 10)."""
    return (
        LayerPricer(LognormalLoss(mean=10_000, cv=5), frequency=1.25),
        LayerPricer(LognormalLoss(mean=450_000, cv=2.5), frequency=0.4),
        LayerPricer(ParetoLoss(alpha=2.1, xm=800_000), frequency=0.09),
    )


def _severity_survival(dist, x):
    """S(x) = P(X > x) for the lognormal / Pareto severities (notebook cell 10)."""
    if isinstance(dist, LognormalLoss):
        return 1.0 if x <= 0 else float(stats.norm.sf((np.log(x) - dist.mu) / dist.sigma))
    if isinstance(dist, ParetoLoss):
        return 1.0 if x < dist.xm else float((dist.xm / x) ** dist.alpha)
    raise TypeError(type(dist).__name__)


def _wang_g(u, lam):
    """Wang distortion g(u) = Phi(Phi^{-1}(u) + lambda)."""
    u = min(max(u, 1e-300), 1.0)
    return float(stats.norm.cdf(stats.norm.ppf(u) + lam))


def _wang_layer_loss(dist, attachment, limit, lam):
    """Risk-adjusted per-occurrence layer expectation int_a^{a+w} g(S(x)) dx (notebook cell 10)."""
    if limit <= 0:
        return 0.0
    if isinstance(dist, LognormalLoss):
        shifted = LognormalLoss(mu=dist.mu + lam * dist.sigma, sigma=dist.sigma)
        return float(
            shifted.limited_expected_value(attachment + limit)
            - shifted.limited_expected_value(attachment)
        )
    val, _ = quad(
        lambda x: _wang_g(_severity_survival(dist, x), lam),
        attachment,
        attachment + limit,
        limit=200,
    )
    return float(val)


def _wang_loading_multiple(attachment, limit, lam, pricers):
    """premium / E[layer loss] under the Wang transform (notebook cell 10)."""
    e_loss = sum(p.expected_layer_loss(attachment, limit) for p in pricers)
    if e_loss <= 0:
        return 1.0
    prem = sum(p.frequency * _wang_layer_loss(p.severity, attachment, limit, lam) for p in pricers)
    return prem / e_loss


def _calibrate_lambda(target_loss_ratio, pricers):
    """Back-derive the Wang lambda so the primary layer's loading == 1/target_loss_ratio (cell 10)."""
    return float(
        brentq(
            lambda lam: _wang_loading_multiple(
                _REF_PRIMARY_ATTACH, _REF_PRIMARY_LIMIT, lam, pricers
            )
            - 1.0 / target_loss_ratio,
            0.0,
            20.0,
            xtol=1e-8,
        )
    )


def test_wang_loading_is_unity_at_zero_lambda():
    """At lambda=0 the Wang transform is the identity, so premium == E[loss] (loading 1.0x)."""
    pricers = _reference_pricers()
    load0 = _wang_loading_multiple(_REF_PRIMARY_ATTACH, _REF_PRIMARY_LIMIT, 0.0, pricers)
    assert load0 == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("lr", [0.50, 0.65, 0.80])
def test_wang_loading_anchor_reproduces_target_loss_ratio(lr):
    """Calibrating lambda to a target LR makes the PRIMARY loading == 1/LR -- the premium-loading
    sweep's premise (#1641): the implied loss ratio knob maps one-to-one to the primary loading."""
    pricers = _reference_pricers()
    lam = _calibrate_lambda(lr, pricers)
    load = _wang_loading_multiple(_REF_PRIMARY_ATTACH, _REF_PRIMARY_LIMIT, lam, pricers)
    assert load == pytest.approx(1.0 / lr, rel=1e-4)
    assert lam > 0.0  # a positive market price of risk for any LR < 100%


def test_shipped_loss_ratio_065_gives_primary_loading_about_1_54x():
    """The shipped LR=0.65 anchors the primary loading at ~1.54x E[loss] (notebook's printed value)."""
    pricers = _reference_pricers()
    lam = _calibrate_lambda(0.65, pricers)
    load = _wang_loading_multiple(_REF_PRIMARY_ATTACH, _REF_PRIMARY_LIMIT, lam, pricers)
    assert load == pytest.approx(1.0 / 0.65, rel=1e-3)  # ~1.538x
    # lambda matches the notebook's printed WANG_LAMBDA ~ 0.367 (cell 10) within tolerance.
    assert lam == pytest.approx(0.367, abs=0.03)


def test_wang_loading_hardens_up_the_tower():
    """At the calibrated lambda the loading is monotone increasing with attachment: the excess and
    catastrophe layers cost more per dollar of expected loss than the primary (#1601/#1641)."""
    pricers = _reference_pricers()
    lam = _calibrate_lambda(0.65, pricers)
    primary = _wang_loading_multiple(250_000, 4_750_000, lam, pricers)
    first_excess = _wang_loading_multiple(5_000_000, 20_000_000, lam, pricers)
    cat = _wang_loading_multiple(50_000_000, 150_000_000, lam, pricers)
    assert primary < first_excess < cat
    assert primary == pytest.approx(1.0 / 0.65, rel=1e-3)


def test_wang_loading_monotone_increasing_in_lambda():
    """A higher market price of risk (lambda) gives a higher premium loading."""
    pricers = _reference_pricers()
    loads = [
        _wang_loading_multiple(_REF_PRIMARY_ATTACH, _REF_PRIMARY_LIMIT, lam, pricers)
        for lam in (0.0, 0.2, 0.4, 0.6)
    ]
    assert all(b > a for a, b in zip(loads, loads[1:]))

"""Tests for HJB jump-term quadrature primitives (issue #1572 R4b).

The hybrid quadrature scheme uses Gauss-Hermite for lognormal components
and stratified body+tail atoms for Pareto components. These tests verify
each scheme against Monte Carlo references on the same integrand that the
notebook's jump operator computes:
``E_X[V(w - (1 + LAE) * L_retained(X, SIR))]`` where
``L_retained(X, SIR) = min(X, SIR) + max(X - TOWER_TOP, 0)``.

Acceptance per issue #1572 R4b: agreement with 1e7-sample MC within
0.5% relative error at three (w, SIR) test points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from ergodic_insurance.hjb_quadrature import (
    annual_aggregate_retained_quantile,
    build_equity_jump_operator_2d,
    build_equity_jump_operator_2d_sizescaled,
    build_loss_atoms,
    component_atoms,
    equity_capped_retention,
    lognormal_gauss_hermite_nodes,
    make_jump_term_2d,
    multi_loss_insolvency_retention_cap,
    pareto_stratified_atoms,
    severity_cdf,
    single_loss_insolvency_retention_cap,
)
from ergodic_insurance.insurance_pricing import LayerPricer
from ergodic_insurance.loss_distributions import LognormalLoss, ParetoLoss

# Common test parameters
LAE_RATIO = 0.05
TOWER_TOP = 50_000_000.0
W_TEST = 5_000_000.0
TEST_SIRS = [250_000.0, 1_000_000.0, 3_000_000.0]


def _retained(x, sir):
    """Per-event retained loss including LAE: (1+LAE) * (min(X, SIR) + max(X - TOWER_TOP, 0))."""
    return (1.0 + LAE_RATIO) * (np.minimum(x, sir) + np.maximum(x - TOWER_TOP, 0.0))


def _v_log(w, w_min: float = 100_000.0):
    """Smooth log-utility V with a wealth floor (matches the notebook setup)."""
    return np.log(np.maximum(w, w_min))


def _quadrature_expectation(x_atoms, p_atoms, w, sir):
    """Compute sum_k p_k * V(w - L_retained(x_k, sir)) via the quadrature points."""
    post_w = np.maximum(w - _retained(x_atoms, sir), 100_000.0)
    return float(np.sum(p_atoms * _v_log(post_w)))


def _sample_severity(severity, n_samples, rng):
    """Draw n_samples from a LognormalLoss or ParetoLoss using the supplied rng."""
    if hasattr(severity, "mu") and hasattr(severity, "sigma"):
        return rng.lognormal(severity.mu, severity.sigma, size=n_samples)
    if hasattr(severity, "alpha") and hasattr(severity, "xm"):
        # Inverse-CDF method: x = xm / (1-u)^(1/alpha)
        u = rng.uniform(0.0, 1.0, size=n_samples)
        return severity.xm / (1.0 - u) ** (1.0 / severity.alpha)
    raise TypeError(f"Unsupported severity for MC sampling: {type(severity)}")


def _mc_expectation(severity, w, sir, n_samples, rng):
    """Monte Carlo reference for E_X[V(w - L_retained(X, SIR))]."""
    samples = _sample_severity(severity, n_samples, rng)
    post_w = np.maximum(w - _retained(samples, sir), 100_000.0)
    return float(np.mean(_v_log(post_w)))


class TestSeverityCdf:
    """severity_cdf: analytical CDF for lognormal and Pareto."""

    def test_lognormal_cdf_matches_scipy(self):
        sev = LognormalLoss(mean=500_000, cv=2.5)
        for x in [10_000, 100_000, 1_000_000, 10_000_000]:
            expected = float(stats.norm.cdf((np.log(x) - sev.mu) / sev.sigma))
            assert severity_cdf(sev, x) == pytest.approx(expected, rel=1e-10)

    def test_pareto_cdf_matches_closed_form(self):
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        for x in [200_000, 500_000, 5_000_000, 100_000_000]:
            expected = 1.0 - (sev.xm / x) ** sev.alpha
            assert severity_cdf(sev, x) == pytest.approx(expected, rel=1e-10)
        assert severity_cdf(sev, 50_000) == 0.0  # below xm

    def test_cdf_extremes(self):
        sev = LognormalLoss(mean=1_000_000, cv=2.0)
        assert severity_cdf(sev, 0.0) == 0.0
        assert severity_cdf(sev, float("inf")) == 1.0


class TestLognormalGaussHermite:
    """Gauss-Hermite quadrature for lognormal severity components."""

    def test_weights_sum_to_one(self):
        sev = LognormalLoss(mean=500_000, cv=2.5)
        for n in [8, 16, 32, 64]:
            _, w = lognormal_gauss_hermite_nodes(sev, n)
            assert w.sum() == pytest.approx(1.0, abs=1e-12)

    def test_mean_matches_analytical(self):
        """E[X] via Gauss-Hermite should match the lognormal closed-form mean."""
        sev = LognormalLoss(mean=500_000, cv=2.5)
        x, w = lognormal_gauss_hermite_nodes(sev, 32)
        gh_mean = float(np.sum(x * w))
        assert gh_mean == pytest.approx(sev.expected_value(), rel=1e-4)

    @pytest.mark.parametrize("cv", [0.5, 1.5, 3.0])
    @pytest.mark.parametrize("mean", [50_000, 500_000, 5_000_000])
    def test_mc_agreement_on_retained_log(self, mean, cv):
        """E_X[log(w - L_retained)] via Gauss-Hermite matches MC within 0.5% at all SIRs."""
        sev = LognormalLoss(mean=mean, cv=cv)
        x, w = lognormal_gauss_hermite_nodes(sev, 32)
        rng = np.random.default_rng(seed=42 + int(mean) + int(cv * 1000))

        n_mc = 1_000_000  # 1e6 is plenty here; CV(mc) ~ stdev / sqrt(N)
        for sir in TEST_SIRS:
            gh_val = _quadrature_expectation(x, w, W_TEST, sir)
            mc_val = _mc_expectation(sev, W_TEST, sir, n_mc, rng)
            rel = abs(gh_val - mc_val) / abs(mc_val)
            assert rel < 0.005, (
                f"GH/MC disagree at mean={mean}, cv={cv}, sir={sir}: "
                f"GH={gh_val:.6f}, MC={mc_val:.6f}, rel_err={rel:.4e}"
            )


class TestParetoStratifiedAtoms:
    """Stratified body+tail atoms for Pareto severity."""

    def test_probabilities_sum_to_one(self):
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        for n in [50, 200, 1000]:
            _, p = pareto_stratified_atoms(sev, n)
            assert p.sum() == pytest.approx(1.0, abs=1e-10)

    def test_atoms_are_positive(self):
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        x, p = pareto_stratified_atoms(sev, 200)
        assert np.all(x > 0)
        assert np.all(p > 0)

    def test_mean_matches_analytical(self):
        """E[X] via stratified atoms matches the Pareto closed-form mean."""
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        x, p = pareto_stratified_atoms(sev, 1000)
        atom_mean = float(np.sum(x * p))
        assert atom_mean == pytest.approx(sev.expected_value(), rel=5e-3)

    def test_too_few_atoms_raises(self):
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        with pytest.raises(ValueError, match="at least 4"):
            pareto_stratified_atoms(sev, 2)

    def test_mc_agreement_on_retained_log(self):
        """E_X[log(w - L_retained)] via stratified atoms matches MC within 0.5%."""
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        x, p = pareto_stratified_atoms(sev, 1000)
        rng = np.random.default_rng(seed=1234)

        n_mc = 2_000_000
        for sir in TEST_SIRS:
            atom_val = _quadrature_expectation(x, p, W_TEST, sir)
            mc_val = _mc_expectation(sev, W_TEST, sir, n_mc, rng)
            rel = abs(atom_val - mc_val) / abs(mc_val)
            assert rel < 0.005, (
                f"Stratified/MC disagree at sir={sir}: "
                f"atom={atom_val:.6f}, MC={mc_val:.6f}, rel_err={rel:.4e}"
            )


class TestComponentAtomsDispatcher:
    """component_atoms: type-based dispatch to GH vs stratified."""

    def test_lognormal_dispatches_to_gauss_hermite(self):
        sev = LognormalLoss(mean=500_000, cv=2.5)
        x1, w1 = component_atoms(sev, n_atoms=1000, gh_nodes=16)
        x2, w2 = lognormal_gauss_hermite_nodes(sev, n_nodes=16)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(w1, w2)
        # n_atoms is ignored for lognormal
        assert len(x1) == 16

    def test_pareto_dispatches_to_stratified(self):
        sev = ParetoLoss(alpha=2.5, xm=100_000)
        x1, p1 = component_atoms(sev, n_atoms=200, gh_nodes=999)
        x2, p2 = pareto_stratified_atoms(sev, 200)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(p1, p2)

    def test_unsupported_severity_raises(self):
        class _UnknownSev:
            pass

        with pytest.raises(TypeError, match="Unsupported severity"):
            component_atoms(_UnknownSev(), n_atoms=100, gh_nodes=16)


class TestBuildLossAtomsMixture:
    """Mixture build_loss_atoms across multiple components."""

    @pytest.fixture
    def mixture_pricers(self):
        """A 3-component mixture matching the notebook's DEFAULT_PRICERS structure:
        attritional lognormal + large lognormal + catastrophic Pareto."""
        return (
            LayerPricer(LognormalLoss(mean=15_000, cv=3.0), frequency=10.0),
            LayerPricer(LognormalLoss(mean=500_000, cv=2.5), frequency=1.0),
            LayerPricer(ParetoLoss(alpha=2.5, xm=100_000), frequency=0.1),
        )

    def test_mixture_probabilities_sum_to_one(self, mixture_pricers):
        _, p = build_loss_atoms(mixture_pricers, n_per_component=200, gh_nodes=16)
        assert p.sum() == pytest.approx(1.0, abs=1e-10)

    def test_mixture_size_is_sum_of_components(self, mixture_pricers):
        x, p = build_loss_atoms(mixture_pricers, n_per_component=200, gh_nodes=16)
        # 16 + 16 + 200 (lognormal GH + lognormal GH + Pareto stratified)
        assert len(x) == 16 + 16 + 200
        assert len(p) == len(x)

    def test_mixture_mean_matches_analytical(self, mixture_pricers):
        """sum(x*p) should match the frequency-weighted analytical mean."""
        x, p = build_loss_atoms(mixture_pricers, n_per_component=1000, gh_nodes=32)
        total_freq = sum(pr.frequency for pr in mixture_pricers)
        analytical = sum(
            (pr.frequency / total_freq) * pr.severity.expected_value() for pr in mixture_pricers
        )
        atom_mean = float(np.sum(x * p))
        assert atom_mean == pytest.approx(analytical, rel=5e-3)

    @pytest.mark.parametrize("sir", TEST_SIRS)
    def test_mixture_mc_agreement_on_retained_log(self, mixture_pricers, sir):
        """E_X[log(w - L_retained)] over the mixture matches MC within 0.5%."""
        x, p = build_loss_atoms(mixture_pricers, n_per_component=1000, gh_nodes=32)
        atom_val = _quadrature_expectation(x, p, W_TEST, sir)

        rng = np.random.default_rng(seed=2026)
        n_mc = 5_000_000
        # MC over the full mixture: sample component by frequency proportion
        freqs = np.array([pr.frequency for pr in mixture_pricers])
        mix_probs = freqs / freqs.sum()
        comp_idx = rng.choice(len(mixture_pricers), size=n_mc, p=mix_probs)
        samples = np.empty(n_mc)
        for i, pr in enumerate(mixture_pricers):
            mask = comp_idx == i
            if mask.any():
                samples[mask] = _sample_severity(pr.severity, int(mask.sum()), rng)
        post_w = np.maximum(W_TEST - _retained(samples, sir), 100_000.0)
        mc_val = float(np.mean(_v_log(post_w)))

        rel = abs(atom_val - mc_val) / abs(mc_val)
        assert rel < 0.005, (
            f"Mixture quadrature/MC disagree at sir={sir}: "
            f"atom={atom_val:.6f}, MC={mc_val:.6f}, rel_err={rel:.4e}"
        )

    def test_empty_pricers_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_loss_atoms((), n_per_component=100, gh_nodes=16)


# ---------------------------------------------------------------------------
# 2-D equity-jump operator + interp modes (issues #1589, #1612)
# ---------------------------------------------------------------------------

# Economic constants for the synthetic 2-D jump problem (scaled-down notebook 07).
# Typed dict[str, Any] so ``**_JUMP_KW`` unpacks cleanly against make_jump_term_2d's
# heterogeneous signature (floats + the Optional[tuple] ``prebuilt``) under mypy.
_JUMP_KW: dict[str, Any] = {
    "tower_top": 1e8,
    "lae_ratio": 0.12,
    "e_floor": 1e5,
    "atr_local": 1.0,
    "reference_revenue_local": 5e6,
    "freq_scaling_exponent_local": 0.5,
    "lambda_base_local": 2.0,
    "v_ruin": float(np.log(1e5)),
    "phi_retention": 0.525,
}


def _make_jump_problem(n_a=6, n_e=5, n_sir=7, seed=0):
    """Build a small (assets, equity-ratio, SIR) jump problem with a random V.

    Includes an atom above ``tower_top`` so ``p_ruin`` is nonzero in the thin-equity
    / high-SIR region (exercises the ruin-atom term in the linear interpolation).
    """
    a_grid = np.geomspace(1e6, 1e8, n_a)  # log-spaced assets
    e_grid = np.linspace(0.05, 1.0, n_e)  # linear equity ratio
    sir_grid = np.geomspace(1e4, 5e7, n_sir)  # log-spaced SIR control
    x_atoms = np.array([5e4, 5e5, 5e6, 5e7, 2e8])  # last atom > tower_top
    p_atoms = np.array([0.6, 0.2, 0.12, 0.05, 0.03])
    jump_term, j_big, p_ruin = make_jump_term_2d(
        a_grid, e_grid, sir_grid, x_atoms, p_atoms, **_JUMP_KW
    )
    rng = np.random.default_rng(seed)
    value_function = rng.standard_normal((n_a, n_e))
    return {
        "a_grid": a_grid,
        "e_grid": e_grid,
        "sir_grid": sir_grid,
        "x_atoms": x_atoms,
        "p_atoms": p_atoms,
        "jump_term": jump_term,
        "j_big": j_big,
        "p_ruin": p_ruin,
        "V": value_function,
    }


def _eval_jump(jump_term, A, e, sir, value_function, interp):
    """Evaluate the jump_term callback over 1-D arrays of (A, e, SIR) points."""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    e = np.atleast_1d(np.asarray(e, dtype=float))
    sir = np.atleast_1d(np.asarray(sir, dtype=float))
    state = np.column_stack([A, e])
    control = sir[:, None]
    return np.asarray(jump_term(state, control, 0.0, value_function, None, interp=interp))


class TestEquityJumpTerm2D:
    """Linear vs nearest interpolation in the 2-D PIDE jump_term (issue #1612)."""

    def test_linear_equals_nearest_at_all_grid_nodes(self):
        """The default-nearest solve path is reproduced by linear at every node."""
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        ia, je, ks = np.meshgrid(
            np.arange(len(a)), np.arange(len(e)), np.arange(len(s)), indexing="ij"
        )
        A, E, S = a[ia].ravel(), e[je].ravel(), s[ks].ravel()
        near = _eval_jump(p["jump_term"], A, E, S, p["V"], "nearest")
        lin = _eval_jump(p["jump_term"], A, E, S, p["V"], "linear")
        # Exact equality at nodes (weights collapse to 0/1); allow only float noise.
        assert np.allclose(near, lin, rtol=0.0, atol=1e-12)

    def test_linear_equals_nearest_at_axis_edges(self):
        """Index-0 and index-(n-1) edge nodes on each axis match (boundary clips)."""
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        for ia in (0, len(a) - 1):
            for je in (0, len(e) - 1):
                for ks in (0, len(s) - 1):
                    near = _eval_jump(p["jump_term"], a[ia], e[je], s[ks], p["V"], "nearest")
                    lin = _eval_jump(p["jump_term"], a[ia], e[je], s[ks], p["V"], "linear")
                    assert np.allclose(near, lin, rtol=0.0, atol=1e-12)

    def test_linear_differs_from_nearest_off_grid(self):
        """Between nodes the linear value genuinely interpolates (is not snapping)."""
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        A = np.sqrt(a[2] * a[3])  # geometric (log) midpoint
        E = 0.5 * (e[1] + e[2])  # arithmetic (linear) midpoint
        S = np.sqrt(s[3] * s[4])
        near = float(_eval_jump(p["jump_term"], A, E, S, p["V"], "nearest")[0])
        lin = float(_eval_jump(p["jump_term"], A, E, S, p["V"], "linear")[0])
        assert abs(near - lin) > 1e-6

    def test_linear_is_continuous_while_nearest_stairsteps(self):
        """Sweeping SIR off-grid: linear is piecewise-linear; nearest jumps at snaps."""
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        A = float(np.sqrt(a[2] * a[3]))
        E = 0.5 * (e[2] + e[3])
        sir = np.geomspace(s[0], s[3], 240)  # dense, crosses several control nodes
        AA = np.full_like(sir, A)
        EE = np.full_like(sir, E)
        near = _eval_jump(p["jump_term"], AA, EE, sir, p["V"], "nearest")
        lin = _eval_jump(p["jump_term"], AA, EE, sir, p["V"], "linear")
        rng = float(np.ptp(lin))
        assert rng > 1e-6, "test is vacuous if the curve is flat"
        lin_step = float(np.max(np.abs(np.diff(lin))))
        near_step = float(np.max(np.abs(np.diff(near))))
        assert lin_step < 0.1 * rng  # continuous: small uniform steps
        assert near_step > 5 * lin_step  # nearest stair-steps at snap boundaries

    def test_linear_matches_hand_computed_trilinear(self):
        """Independent (bi/tri)linear reconstruction with explicit C-order indices.

        Uses a non-symmetric V so an (a, e) transpose / Fortran-order slip would be
        caught (it would still pass equal-at-nodes but fail this midpoint check).
        """
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        n_e, n_sir = len(e), len(s)
        # Deterministic, position-dependent value function (orientation probe).
        V = (10.0 * np.arange(len(a))[:, None] + np.arange(n_e)[None, :]).astype(float)
        jt = p["jump_term"]
        ia, je, ks = 2, 1, 3  # interior cell
        A = np.sqrt(a[ia] * a[ia + 1])  # w_hi = 0.5 on each log axis
        E = 0.5 * (e[je] + e[je + 1])  # w_hi = 0.5 on the linear axis
        S = np.sqrt(s[ks] * s[ks + 1])

        # Per-node jump value bundle, gathered at the 8 corners (C-order).
        jflat = np.asarray(p["j_big"] @ V.ravel()) + p["p_ruin"].ravel() * _JUMP_KW["v_ruin"]
        v_flat = V.ravel()
        v_cur = 0.25 * sum(v_flat[(ia + di) * n_e + (je + dj)] for di in (0, 1) for dj in (0, 1))
        ev_post = 0.125 * sum(
            jflat[((ia + di) * n_e + (je + dj)) * n_sir + (ks + dk)]
            for di in (0, 1)
            for dj in (0, 1)
            for dk in (0, 1)
        )
        lam = (
            _JUMP_KW["lambda_base_local"]
            * (A * _JUMP_KW["atr_local"] / _JUMP_KW["reference_revenue_local"])
            ** _JUMP_KW["freq_scaling_exponent_local"]
        )
        expected = lam * (ev_post - v_cur)
        got = float(_eval_jump(jt, A, E, S, V, "linear")[0])
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_ruin_atom_term_is_interpolated(self):
        """The p_ruin*v_ruin ruin-atom term participates in the linear blend.

        Varying only v_ruin (reusing the prebuilt operator) must shift the linear
        jump value by lam * trilinear(p_ruin) * delta_v_ruin -- and the test region
        is chosen so trilinear(p_ruin) > 0.
        """
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        n_e, n_sir = len(e), len(s)
        kw = {k: v for k, v in _JUMP_KW.items() if k != "v_ruin"}
        jt_a, _, _ = make_jump_term_2d(
            a,
            e,
            s,
            p["x_atoms"],
            p["p_atoms"],
            v_ruin=-5.0,
            prebuilt=(p["j_big"], p["p_ruin"]),
            **kw,
        )
        jt_b, _, _ = make_jump_term_2d(
            a,
            e,
            s,
            p["x_atoms"],
            p["p_atoms"],
            v_ruin=-9.0,
            prebuilt=(p["j_big"], p["p_ruin"]),
            **kw,
        )
        # Thin-equity, high-SIR region so a single loss can bankrupt -> p_ruin > 0.
        ia, je, ks = 1, 0, len(s) - 2
        A = np.sqrt(a[ia] * a[ia + 1])
        E = 0.5 * (e[je] + e[je + 1])
        S = np.sqrt(s[ks] * s[ks + 1])
        pr_flat = p["p_ruin"].ravel()
        tri_p = 0.125 * sum(
            pr_flat[((ia + di) * n_e + (je + dj)) * n_sir + (ks + dk)]
            for di in (0, 1)
            for dj in (0, 1)
            for dk in (0, 1)
        )
        assert tri_p > 0.0, "test region must have nonzero ruin probability"
        lam = (
            _JUMP_KW["lambda_base_local"]
            * (A * _JUMP_KW["atr_local"] / _JUMP_KW["reference_revenue_local"])
            ** _JUMP_KW["freq_scaling_exponent_local"]
        )
        got_a = float(_eval_jump(jt_a, A, E, S, p["V"], "linear")[0])
        got_b = float(_eval_jump(jt_b, A, E, S, p["V"], "linear")[0])
        assert (got_b - got_a) == pytest.approx(lam * tri_p * (-9.0 - -5.0), rel=1e-9, abs=1e-12)

    def test_out_of_range_flat_extrapolation(self):
        """Queries outside a grid clamp to the edge node (== nearest).

        Holds the other two axes on-grid so the only off-grid behaviour under test is
        the flat extrapolation on the swept axis (w clamps to 0/1 -> edge node).
        """
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        # (axis-under-test out-of-range values, on-grid value for the other two)
        cases = [
            (a[0] * 0.3, e[2], s[3]),  # assets below grid
            (a[-1] * 3.0, e[2], s[3]),  # assets above grid
            (a[2], e[0] - 0.5, s[3]),  # equity ratio below grid
            (a[2], e[-1] + 0.5, s[3]),  # equity ratio above grid
            (a[2], e[2], s[0] * 0.3),  # SIR below grid
            (a[2], e[2], s[-1] * 3.0),  # SIR above grid
        ]
        for A, E, S in cases:
            near = float(_eval_jump(p["jump_term"], A, E, S, p["V"], "nearest")[0])
            lin = float(_eval_jump(p["jump_term"], A, E, S, p["V"], "linear")[0])
            assert lin == pytest.approx(near, rel=0.0, abs=1e-12)

    def test_default_interp_is_nearest(self):
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        A, E, S = np.sqrt(a[2] * a[3]), 0.5 * (e[1] + e[2]), np.sqrt(s[3] * s[4])
        state = np.array([[A, E]])
        control = np.array([[S]])
        default = np.asarray(p["jump_term"](state, control, 0.0, p["V"], None))
        nearest = np.asarray(p["jump_term"](state, control, 0.0, p["V"], None, interp="nearest"))
        assert np.array_equal(default, nearest)

    def test_invalid_interp_raises(self):
        p = _make_jump_problem()
        state = np.array([[5e6, 0.5]])
        control = np.array([[1e6]])
        with pytest.raises(ValueError, match="nearest.*linear|interp"):
            p["jump_term"](state, control, 0.0, p["V"], None, interp="cubic")

    def test_make_jump_term_requires_phi_when_building(self):
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        kw = {k: v for k, v in _JUMP_KW.items() if k != "phi_retention"}
        with pytest.raises(ValueError, match="phi_retention"):
            make_jump_term_2d(a, e, s, p["x_atoms"], p["p_atoms"], **kw)

    def test_prebuilt_path_ignores_phi(self):
        """Reusing a prebuilt operator needs no phi and matches the built operator."""
        p = _make_jump_problem()
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        kw = {k: v for k, v in _JUMP_KW.items() if k != "phi_retention"}
        jt2, _, _ = make_jump_term_2d(
            a, e, s, p["x_atoms"], p["p_atoms"], prebuilt=(p["j_big"], p["p_ruin"]), **kw
        )
        A, E, S = np.sqrt(a[2] * a[3]), 0.5 * (e[1] + e[2]), np.sqrt(s[3] * s[4])
        for interp in ("nearest", "linear"):
            ref = _eval_jump(p["jump_term"], A, E, S, p["V"], interp)
            got = _eval_jump(jt2, A, E, S, p["V"], interp)
            assert np.allclose(ref, got, rtol=0.0, atol=1e-12)

    def test_value_cache_refreshes_on_identity_change(self):
        """The J_big@V cache keys on object identity; a new V refreshes it."""
        p = _make_jump_problem()
        jt = p["jump_term"]
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        A, E, S = np.sqrt(a[2] * a[3]), 0.5 * (e[1] + e[2]), np.sqrt(s[3] * s[4])
        v1 = p["V"]
        first = _eval_jump(jt, A, E, S, v1, "linear")
        again = _eval_jump(jt, A, E, S, v1, "linear")  # same object -> cached
        assert np.array_equal(first, again)
        v2 = v1 + 1.0  # new object with different values -> cache must refresh
        got2 = _eval_jump(jt, A, E, S, v2, "linear")
        # A fresh callback over v2 is the ground truth for the refreshed result.
        jt_fresh, _, _ = make_jump_term_2d(a, e, s, p["x_atoms"], p["p_atoms"], **_JUMP_KW)
        expect2 = _eval_jump(jt_fresh, A, E, S, v2, "linear")
        assert np.allclose(got2, expect2, rtol=0.0, atol=1e-12)
        assert not np.allclose(got2, first)  # actually changed

    def test_linear_equals_nearest_at_nodes_second_grid_size(self):
        """Equal-at-nodes holds on a different grid (cells 16 and 31 differ)."""
        p = _make_jump_problem(n_a=8, n_e=9, n_sir=11, seed=3)
        a, e, s = p["a_grid"], p["e_grid"], p["sir_grid"]
        ia, je, ks = np.meshgrid(
            np.arange(len(a)), np.arange(len(e)), np.arange(len(s)), indexing="ij"
        )
        A, E, S = a[ia].ravel(), e[je].ravel(), s[ks].ravel()
        near = _eval_jump(p["jump_term"], A, E, S, p["V"], "nearest")
        lin = _eval_jump(p["jump_term"], A, E, S, p["V"], "linear")
        assert np.allclose(near, lin, rtol=0.0, atol=1e-12)


class TestEquityJumpOperator2D:
    """Direct properties of the block-diagonal jump operator (issue #1589)."""

    def test_operator_shapes_and_ruin_floor(self):
        p = _make_jump_problem()
        n_a, n_e, n_sir = len(p["a_grid"]), len(p["e_grid"]), len(p["sir_grid"])
        assert p["j_big"].shape == (n_a * n_e * n_sir, n_a * n_e)
        assert p["p_ruin"].shape == (n_a, n_e, n_sir)
        # An atom above tower_top causes ruin at any SIR -> a positive floor at node 0.
        assert float(p["p_ruin"][:, :, 0].max()) > 0.0

    def test_phi_scales_surviving_reduction_only(self):
        """phi_retention scales the surviving (J_big) post-jump, not the ruin mask."""
        a, e, s = (np.geomspace(1e6, 1e8, 6), np.linspace(0.05, 1.0, 5), np.geomspace(1e4, 5e7, 7))
        x = np.array([5e4, 5e5, 5e6, 5e7, 2e8])
        pa = np.array([0.6, 0.2, 0.12, 0.05, 0.03])
        j_lo, pr_lo = build_equity_jump_operator_2d(a, e, s, x, pa, 1e8, 0.12, 1e5, 0.3)
        j_hi, pr_hi = build_equity_jump_operator_2d(a, e, s, x, pa, 1e8, 0.12, 1e5, 0.7)
        # Ruin uses GROSS loss (phi-independent) -> identical masks.
        assert np.array_equal(pr_lo, pr_hi)
        # Surviving operator depends on phi -> the maps differ.
        assert (j_lo - j_hi).nnz > 0 or not np.allclose(j_lo.toarray(), j_hi.toarray())


# ---------------------------------------------------------------------------
# Size-scaled per-assets-slice jump operator (issue #1607)
# ---------------------------------------------------------------------------

# Scaled-down notebook-07 economics for the size-scaled operator tests.
_SS_GRIDS: dict[str, Any] = {
    "a_grid": np.geomspace(5e5, 2e8, 10),  # log-spaced assets
    "e_grid_l": np.linspace(0.01, 1.0, 12),  # linear equity ratio
    "sir_grid_l": np.geomspace(1e4, 5e7, 16),  # log-spaced SIR control
}
_SS_KW: dict[str, Any] = {
    "tower_top": 1e8,
    "lae_ratio": 0.12,
    "e_floor": 1e5,
    "phi_retention": 0.525,
}
_SS_ECON: dict[str, Any] = {
    "atr": 1.5,
    "reference_revenue": 7.5e6,
    "sev_reference_base": 5e6,
}


def _components_for_sizescale(n_atoms=40, gh_nodes=16):
    """Two-component (lognormal "large" + Pareto "cat") atoms + matching frequencies."""
    pricers = (
        LayerPricer(LognormalLoss(mean=450_000, cv=2.5), frequency=0.4),
        LayerPricer(ParetoLoss(alpha=2.05, xm=800_000), frequency=0.075),
    )
    comp_x, comp_p = [], []
    for pr in pricers:
        xa, pa = component_atoms(pr.severity, n_atoms=n_atoms, gh_nodes=gh_nodes)
        comp_x.append(xa)
        comp_p.append(pa)
    freq_ref = [pr.frequency for pr in pricers]
    return pricers, comp_x, comp_p, freq_ref


class TestSizeScaledJumpOperator1607:
    """Size-scaled per-assets-slice equity-jump operator -- issue #1607."""

    def test_gamma0_reduces_to_flat_operator(self):
        """sev_exp=0 + shared freq_exp reproduces the flat operator and scalar-lambda path."""
        pricers, comp_x, comp_p, freq_ref = _components_for_sizescale()
        x_flat, p_flat = build_loss_atoms(pricers, n_per_component=40, gh_nodes=16)
        j_flat, pr_flat = build_equity_jump_operator_2d(
            _SS_GRIDS["a_grid"],
            _SS_GRIDS["e_grid_l"],
            _SS_GRIDS["sir_grid_l"],
            x_flat,
            p_flat,
            **_SS_KW,
        )
        j_ss, pr_ss, lam_ss = build_equity_jump_operator_2d_sizescaled(
            _SS_GRIDS["a_grid"],
            _SS_GRIDS["e_grid_l"],
            _SS_GRIDS["sir_grid_l"],
            comp_x,
            comp_p,
            freq_ref,
            [1.0, 1.0],  # shared frequency exponent
            [0.0, 0.0],  # gamma = 0: no severity scaling
            **_SS_ECON,
            **_SS_KW,
        )
        assert np.allclose(j_flat.toarray(), j_ss.toarray(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(pr_flat, pr_ss, rtol=1e-9, atol=1e-12)
        rev = _SS_GRIDS["a_grid"] * _SS_ECON["atr"]
        expected = sum(freq_ref) * (rev / _SS_ECON["reference_revenue"]) ** 1.0
        np.testing.assert_allclose(lam_ss, expected, rtol=1e-12)

    def test_lambda_2d_is_sum_of_per_component_power_laws(self):
        """Non-uniform freq exponents -> lambda(A) is a sum of power laws, not one power."""
        _, comp_x, comp_p, freq_ref = _components_for_sizescale()
        freq_exp = [1.0, 0.5]
        _, _, lam = build_equity_jump_operator_2d_sizescaled(
            _SS_GRIDS["a_grid"],
            _SS_GRIDS["e_grid_l"],
            _SS_GRIDS["sir_grid_l"],
            comp_x,
            comp_p,
            freq_ref,
            freq_exp,
            [0.0, 0.0],
            **_SS_ECON,
            **_SS_KW,
        )
        ratio = _SS_GRIDS["a_grid"] * _SS_ECON["atr"] / _SS_ECON["reference_revenue"]
        expected = freq_ref[0] * ratio ** freq_exp[0] + freq_ref[1] * ratio ** freq_exp[1]
        np.testing.assert_allclose(lam, expected, rtol=1e-12)

    def test_probability_mass_is_conserved(self):
        """Each atom interpolates to interior nodes or ruins -> row_sum(J) + p_ruin == 1."""
        _, comp_x, comp_p, freq_ref = _components_for_sizescale()
        j_ss, pr_ss, _ = build_equity_jump_operator_2d_sizescaled(
            _SS_GRIDS["a_grid"],
            _SS_GRIDS["e_grid_l"],
            _SS_GRIDS["sir_grid_l"],
            comp_x,
            comp_p,
            freq_ref,
            [1.0, 1.0],
            [0.0, 0.5],  # cat scales -- mass conservation must still hold
            **_SS_ECON,
            **_SS_KW,
        )
        row_sum = np.asarray(j_ss.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sum + pr_ss.ravel(), 1.0, rtol=0, atol=1e-12)

    def test_cat_severity_scaling_raises_ruin_at_grown_firm(self):
        """gamma>0 on cat changes the operator and raises single-loss ruin at the top slice."""
        _, comp_x, comp_p, freq_ref = _components_for_sizescale()
        common = {"component_freq_ref": freq_ref, "component_freq_exponent": [1.0, 1.0]}
        j0, pr0, _ = build_equity_jump_operator_2d_sizescaled(
            _SS_GRIDS["a_grid"],
            _SS_GRIDS["e_grid_l"],
            _SS_GRIDS["sir_grid_l"],
            comp_x,
            comp_p,
            component_sev_exponent=[0.0, 0.0],
            **common,
            **_SS_ECON,
            **_SS_KW,
        )
        jg, prg, _ = build_equity_jump_operator_2d_sizescaled(
            _SS_GRIDS["a_grid"],
            _SS_GRIDS["e_grid_l"],
            _SS_GRIDS["sir_grid_l"],
            comp_x,
            comp_p,
            component_sev_exponent=[0.0, 0.75],
            **common,
            **_SS_ECON,
            **_SS_KW,
        )
        assert not np.allclose(j0.toarray(), jg.toarray())
        # Scaled cat losses are a strict superset of the ruin region at the top assets slice.
        assert np.all(prg[-1] >= pr0[-1] - 1e-15)
        assert prg[-1].sum() > pr0[-1].sum()

    def test_make_jump_term_uses_lambda_2d(self):
        """make_jump_term_2d(lambda_2d=...) scales intensity by lambda_2d, not the power law."""
        _, comp_x, comp_p, freq_ref = _components_for_sizescale()
        a, e, s = _SS_GRIDS["a_grid"], _SS_GRIDS["e_grid_l"], _SS_GRIDS["sir_grid_l"]
        j_ss, pr_ss, lam = build_equity_jump_operator_2d_sizescaled(
            a, e, s, comp_x, comp_p, freq_ref, [1.0, 0.5], [0.0, 0.5], **_SS_ECON, **_SS_KW
        )
        n_atoms = sum(len(x) for x in comp_x)
        dummy_x = np.concatenate(comp_x)
        dummy_p = np.full(n_atoms, 1.0 / n_atoms)
        econ = {
            "atr_local": _SS_ECON["atr"],
            "reference_revenue_local": _SS_ECON["reference_revenue"],
            "freq_scaling_exponent_local": 1.0,
            "lambda_base_local": sum(freq_ref),
            "v_ruin": float(np.log(_SS_KW["e_floor"])),
        }
        shared = {
            "tower_top": _SS_KW["tower_top"],
            "lae_ratio": _SS_KW["lae_ratio"],
            "e_floor": _SS_KW["e_floor"],
            "phi_retention": _SS_KW["phi_retention"],
            "prebuilt": (j_ss, pr_ss),
        }
        jt_pl, _, _ = make_jump_term_2d(a, e, s, dummy_x, dummy_p, **shared, **econ)
        jt_l2, _, _ = make_jump_term_2d(a, e, s, dummy_x, dummy_p, lambda_2d=lam, **shared, **econ)
        rng = np.random.default_rng(0)
        value_function = rng.standard_normal((len(a), len(e)))
        ia = 6
        state = np.array([[a[ia], e[5]]])
        control = np.array([[s[8]]])
        out_pl = float(jt_pl(state, control, 0.0, value_function, None)[0])
        out_l2 = float(jt_l2(state, control, 0.0, value_function, None)[0])
        lam_pl = sum(freq_ref) * (a[ia] * _SS_ECON["atr"] / _SS_ECON["reference_revenue"]) ** 1.0
        assert abs(out_pl) > 1e-9 and abs(lam[ia] - lam_pl) > 1e-6  # the two intensities differ
        np.testing.assert_allclose(out_l2 / out_pl, lam[ia] / lam_pl, rtol=1e-9)


class TestEquityCappedRetention:
    """Deployment-time equity-aware retention cap (issue #1633)."""

    # Matches the notebook constants the cap reasons about.
    LAE = 0.12
    E_FLOOR = 100_000.0
    KAPPA = 0.40

    def test_binds_at_scale(self):
        # Grown firm: raw policy pins near the $50M grid cap, equity is large.
        out = float(equity_capped_retention(50e6, 78e6, self.KAPPA))
        assert out == pytest.approx(self.KAPPA * 78e6)  # 0.40 * $78M = $31.2M

    def test_passthrough_when_below_cap(self):
        # Raw retention already below kappa*equity -> returned unchanged.
        out = float(equity_capped_retention(1e6, 78e6, self.KAPPA))
        assert out == pytest.approx(1e6)

    def test_never_exceeds_kappa_equity(self):
        rng = np.random.default_rng(1633)
        raw = rng.uniform(1e4, 50e6, size=200)
        equity = rng.uniform(1e5, 200e6, size=200)
        out = equity_capped_retention(raw, equity, self.KAPPA)
        assert np.all(out <= self.KAPPA * equity + 1e-6)
        assert np.all(out <= raw + 1e-6)

    def test_monotone_nondecreasing_in_equity(self):
        # At a fixed (large) raw retention the cap binds, so it rises with equity.
        equity = np.linspace(1e5, 100e6, 50)
        out = equity_capped_retention(1e9, equity, self.KAPPA)
        assert np.all(np.diff(out) >= -1e-9)

    def test_single_loss_safety_in_operating_regime(self):
        # The defining property: a single retained per-event loss (gross of LAE) at the
        # capped retention cannot push equity below the ruin floor, for equity in the
        # operating regime (>= $1M >> the ~$181K break-even at kappa=0.40).
        equity = np.linspace(1e6, 200e6, 100)
        capped = equity_capped_retention(1e9, equity, self.KAPPA)  # raw large -> cap binds
        post_loss_equity = equity - (1.0 + self.LAE) * capped
        assert np.all(post_loss_equity >= self.E_FLOOR)

    def test_zero_and_negative_equity_force_max_coverage(self):
        assert float(equity_capped_retention(5e6, 0.0, self.KAPPA)) == 0.0
        assert float(equity_capped_retention(5e6, -1e6, self.KAPPA)) == 0.0

    def test_nonfinite_kappa_returns_raw_uncapped(self):
        # The calibration sweep's "no cap" baseline row passes kappa = inf.
        assert float(equity_capped_retention(50e6, 78e6, float("inf"))) == pytest.approx(50e6)

    def test_vectorized_matches_scalar(self):
        raw = np.array([50e6, 1e6, 5e6, 25e6])
        equity = np.array([78e6, 78e6, 2e6, 30e6])
        vec = equity_capped_retention(raw, equity, self.KAPPA)
        scalar = np.array(
            [float(equity_capped_retention(r, e, self.KAPPA)) for r, e in zip(raw, equity)]
        )
        np.testing.assert_allclose(vec, scalar)


class TestSingleLossInsolvencyRetentionCap:
    """Endogenous single-loss-insolvency retention bound T (issue #1649).

    T = (equity - e_floor) / (1 + LAE) is the largest retention that cannot bankrupt
    the firm on one event; it has no hand-set knob (only the ruin floor and LAE), so
    it can bound the HJB control selection endogenously and replace the #1633 kappa cap.
    """

    LAE = 0.12
    E_FLOOR = 100_000.0

    def test_formula(self):
        eq = 5_000_000.0
        T = float(single_loss_insolvency_retention_cap(eq, self.LAE, self.E_FLOOR))
        assert T == pytest.approx((eq - self.E_FLOOR) / (1.0 + self.LAE))

    def test_worst_single_loss_hits_floor_exactly(self):
        # At SIR = T the worst retained working-layer loss (gross of LAE) drives
        # post-loss equity to precisely the floor -- the boundary of p_ruin's step.
        eq = 8_000_000.0
        T = float(single_loss_insolvency_retention_cap(eq, self.LAE, self.E_FLOOR))
        post_equity = eq - (1.0 + self.LAE) * T
        assert post_equity == pytest.approx(self.E_FLOOR)

    def test_monotone_increasing_in_equity(self):
        eqs = np.array([0.2e6, 1e6, 5e6, 50e6])
        caps = single_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR)
        assert np.all(np.diff(caps) > 0)

    def test_below_equity_in_operating_regime(self):
        # SIR <= T implies SIR < equity for any firm above the floor (the #1589 motive).
        eqs = np.array([0.5e6, 1e6, 10e6, 100e6])
        caps = single_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR)
        assert np.all(caps < eqs)

    def test_impaired_firm_floored_at_zero(self):
        # equity below the floor -> negative raw bound -> floored to 0 (max coverage).
        assert float(single_loss_insolvency_retention_cap(50_000.0, self.LAE, self.E_FLOOR)) == 0.0

    def test_vectorized_matches_scalar(self):
        eqs = np.array([1e6, 5e6, 30e6, 80e6])
        vec = single_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR)
        scalar = np.array(
            [float(single_loss_insolvency_retention_cap(e, self.LAE, self.E_FLOOR)) for e in eqs]
        )
        np.testing.assert_allclose(vec, scalar)


class TestMultiLossInsolvencyRetentionCap:
    """Endogenous N-event-survivability retention bound T / N (issue #1659).

    Generalizes the single-loss bound T to require surviving N worst-case retained events
    in a year. With the knob-free N = E[#losses/yr] (lambda_ref ~ 1.74 at the notebook's
    reference firm), T / N lands at ~0.51 * equity -- reproducing the empirically
    value-creating kappa = 0.50 deployment cap (issue #1633) it retires, with no hand-set
    retention fraction.
    """

    LAE = 0.12
    E_FLOOR = 100_000.0
    LAMBDA_REF = 1.74  # sum of the notebook's base layer frequencies (1.25 + 0.40 + 0.09)

    def test_formula(self):
        eq = 5_000_000.0
        cap = float(multi_loss_insolvency_retention_cap(eq, self.LAE, self.E_FLOOR, 3.0))
        assert cap == pytest.approx((eq - self.E_FLOOR) / (1.0 + self.LAE) / 3.0)

    def test_reduces_to_single_loss_at_n1(self):
        # N = 1 recovers the single-loss bound exactly.
        eqs = np.array([0.5e6, 1e6, 5e6, 50e6])
        multi = multi_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR, 1.0)
        single = single_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR)
        np.testing.assert_allclose(multi, single)

    def test_n_clamped_to_at_least_one(self):
        # N < 1 must not loosen the bound beyond the single-loss cap (preserves SIR <= equity).
        eqs = np.array([1e6, 10e6])
        single = single_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR)
        np.testing.assert_allclose(
            multi_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR, 0.3), single
        )

    def test_never_looser_than_single_loss(self):
        eqs = np.array([0.5e6, 1e6, 10e6, 100e6])
        multi = multi_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR, self.LAMBDA_REF)
        single = single_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR)
        assert np.all(multi <= single)
        assert np.all(multi < eqs)  # strictly inside the SIR <= equity region

    def test_n_worst_events_hit_floor_exactly(self):
        # At SIR = T / N, N worst single-event retentions (gross of LAE) drive equity to the floor.
        eq = 8_000_000.0
        n = 4.0
        sir = float(multi_loss_insolvency_retention_cap(eq, self.LAE, self.E_FLOOR, n))
        post_equity = eq - n * (1.0 + self.LAE) * sir
        assert post_equity == pytest.approx(self.E_FLOOR)

    def test_monotone_increasing_in_equity(self):
        eqs = np.array([0.2e6, 1e6, 5e6, 50e6])
        caps = multi_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR, self.LAMBDA_REF)
        assert np.all(np.diff(caps) > 0)

    def test_reference_fraction_matches_kappa(self):
        # The headline relationship (issue #1659 AC3): T / lambda_ref ~ 0.51 * equity ~ kappa=0.50,
        # near-constant across firm sizes (within the empirically value-creating 0.40-0.55 band).
        for eq in [3.25e6, 6.5e6, 16.25e6, 32.5e6]:
            frac = (
                float(
                    multi_loss_insolvency_retention_cap(eq, self.LAE, self.E_FLOOR, self.LAMBDA_REF)
                )
                / eq
            )
            assert 0.40 <= frac <= 0.55

    def test_vectorized_matches_scalar(self):
        eqs = np.array([1e6, 5e6, 30e6, 80e6])
        vec = multi_loss_insolvency_retention_cap(eqs, self.LAE, self.E_FLOOR, self.LAMBDA_REF)
        scalar = np.array(
            [
                float(
                    multi_loss_insolvency_retention_cap(e, self.LAE, self.E_FLOOR, self.LAMBDA_REF)
                )
                for e in eqs
            ]
        )
        np.testing.assert_allclose(vec, scalar)


class TestAnnualAggregateRetainedQuantile:
    """FFT compound-Poisson annual-aggregate retained-loss quantile (issue #1659).

    The Option-1 cross-validation / fallback for the N-event bound: the q-quantile of the
    annual aggregate retained loss S = sum_i (1+LAE)*(min(X_i, SIR) + (X_i - tower)+), with
    the event count Poisson(lam). Computed by FFT of the compound-Poisson characteristic
    function and validated against the analytic scaled-Poisson (degenerate severity) and a
    seeded Monte Carlo.
    """

    LAE = 0.12

    def test_degenerate_severity_matches_scaled_poisson(self):
        # A single severity atom c (with min(c, SIR) = c) makes S = r * M, M ~ Poisson(lam),
        # r = (1+LAE)*c -- so the q-quantile is r * poisson.ppf(q, lam), exactly.
        for lam, c, sir, q in [
            (2.0, 1e6, 5e6, 0.95),
            (1.74, 8e5, 2e6, 0.99),
            (5.0, 3e5, 4e5, 0.90),
        ]:
            r = (1.0 + self.LAE) * min(c, sir)
            analytic = r * float(stats.poisson.ppf(q, lam))
            fft = annual_aggregate_retained_quantile([c], [1.0], lam, sir, self.LAE, q)
            assert fft == pytest.approx(analytic, rel=0.01)

    def test_zero_intensity_returns_zero(self):
        x = np.geomspace(1e4, 5e6, 40)
        p = np.ones(40) / 40
        assert annual_aggregate_retained_quantile(x, p, 0.0, 2e6, self.LAE, 0.95) == 0.0

    def test_monotone_in_quantile(self):
        x = np.geomspace(1e4, 5e6, 60)
        p = np.ones(60) / 60
        vals = [
            annual_aggregate_retained_quantile(x, p, 1.74, 2e6, self.LAE, q)
            for q in [0.5, 0.8, 0.95, 0.99]
        ]
        assert all(np.diff(vals) >= -1e-6)

    def test_monotone_nondecreasing_in_sir(self):
        # More retention (higher SIR) cannot lower the aggregate retained quantile.
        x = np.geomspace(1e4, 5e6, 60)
        p = np.ones(60) / 60
        vals = [
            annual_aggregate_retained_quantile(x, p, 1.74, s, self.LAE, 0.95)
            for s in [5e5, 1e6, 2e6, 5e6]
        ]
        assert all(np.diff(vals) >= -1e-6)

    def test_matches_monte_carlo(self):
        # Mixed lognormal-ish atom table vs a seeded compound-Poisson MC (loose tolerance).
        rng = np.random.default_rng(20240601)
        x = np.geomspace(1e4, 8e6, 200)
        # a roughly log-uniform severity with a light tail
        p = 1.0 / x
        p = p / p.sum()
        lam, sir, q = 2.5, 1.5e6, 0.95
        fft = annual_aggregate_retained_quantile(x, p, lam, sir, self.LAE, q)
        n_years = 600_000
        counts = rng.poisson(lam, n_years)
        draws = rng.choice(len(x), size=int(counts.sum()), p=p)
        retained = (1.0 + self.LAE) * np.minimum(x[draws], sir)
        totals = np.zeros(n_years)
        np.add.at(totals, np.repeat(np.arange(n_years), counts), retained)
        mc = float(np.quantile(totals, q))
        assert fft == pytest.approx(mc, rel=0.05)

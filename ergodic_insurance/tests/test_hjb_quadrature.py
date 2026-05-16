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

import numpy as np
import pytest

from ergodic_insurance.hjb_quadrature import (
    build_loss_atoms,
    component_atoms,
    lognormal_gauss_hermite_nodes,
    pareto_stratified_atoms,
    severity_cdf,
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
        from scipy import stats

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

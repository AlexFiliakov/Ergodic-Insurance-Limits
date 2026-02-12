"""Tests for analytical layer pricing (Issue #746).

This module tests:
- Limited expected value (LEV) methods on severity distributions
- LayerPricer class for analytical layer pricing
- Increased Limits Factors (ILFs)
- Loss Elimination Ratios (LERs)
- Exposure curves (Lee diagrams)

References:
    - Klugman, Panjer, Willmot — *Loss Models*, Chapter 5
    - Lee (1988) — Loss Distributions
    - Miccolis (1977) — Increased Limits and Excess of Loss Pricing
"""

from __future__ import annotations

import numpy as np
import pytest

from ergodic_insurance.insurance_pricing import LayerPricer
from ergodic_insurance.loss_distributions import (
    GeneralizedParetoLoss,
    LognormalLoss,
    ParetoLoss,
)


# ---------------------------------------------------------------------------
# Limited Expected Value: Lognormal
# ---------------------------------------------------------------------------
class TestLognormalLEV:
    """Test LognormalLoss.limited_expected_value()."""

    @pytest.fixture
    def dist(self):
        return LognormalLoss(mean=100_000, cv=1.0, seed=42)

    def test_lev_zero_limit(self, dist):
        """LEV(0) should be 0."""
        assert dist.limited_expected_value(0) == 0.0

    def test_lev_negative_limit(self, dist):
        """LEV(negative) should be 0."""
        assert dist.limited_expected_value(-100) == 0.0

    def test_lev_very_large_limit_approaches_mean(self, dist):
        """LEV(inf) should approach E[X]."""
        lev = dist.limited_expected_value(1e15)
        assert lev == pytest.approx(dist.expected_value(), rel=1e-4)

    def test_lev_less_than_mean(self, dist):
        """LEV(d) < E[X] for any finite d."""
        lev = dist.limited_expected_value(dist.expected_value())
        assert lev < dist.expected_value()

    def test_lev_monotonically_increasing(self, dist):
        """LEV should be monotonically non-decreasing."""
        limits = [10_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000]
        levs = [dist.limited_expected_value(d) for d in limits]
        for i in range(1, len(levs)):
            assert levs[i] >= levs[i - 1], f"LEV({limits[i]}) < LEV({limits[i-1]})"

    def test_lev_matches_monte_carlo(self):
        """LEV should match Monte Carlo estimate within tolerance."""
        dist = LognormalLoss(mean=50_000, cv=1.5, seed=42)
        limit = 100_000
        samples = dist.generate_severity(500_000)
        mc_lev = float(np.mean(np.minimum(samples, limit)))
        analytical_lev = dist.limited_expected_value(limit)
        assert analytical_lev == pytest.approx(mc_lev, rel=0.02)

    def test_lev_different_parameterizations(self):
        """LEV should give consistent results regardless of parameterization."""
        d1 = LognormalLoss(mean=100_000, cv=1.0)
        d2 = LognormalLoss(mu=d1.mu, sigma=d1.sigma)
        limit = 200_000
        assert d1.limited_expected_value(limit) == pytest.approx(
            d2.limited_expected_value(limit), rel=1e-10
        )


# ---------------------------------------------------------------------------
# Limited Expected Value: Pareto
# ---------------------------------------------------------------------------
class TestParetoLEV:
    """Test ParetoLoss.limited_expected_value()."""

    @pytest.fixture
    def dist(self):
        return ParetoLoss(alpha=2.5, xm=100_000, seed=42)

    def test_lev_zero_limit(self, dist):
        assert dist.limited_expected_value(0) == 0.0

    def test_lev_negative_limit(self, dist):
        assert dist.limited_expected_value(-100) == 0.0

    def test_lev_below_minimum(self, dist):
        """LEV(d) = d when d <= xm (all losses exceed d)."""
        assert dist.limited_expected_value(50_000) == 50_000
        assert dist.limited_expected_value(100_000) == 100_000

    def test_lev_above_minimum(self, dist):
        """LEV should be between xm and E[X] for d > xm."""
        lev = dist.limited_expected_value(500_000)
        assert lev > dist.xm
        assert lev < dist.expected_value()

    def test_lev_approaches_expected_value(self, dist):
        """LEV(d) -> E[X] as d -> inf for alpha > 1."""
        lev = dist.limited_expected_value(1e15)
        assert lev == pytest.approx(dist.expected_value(), rel=1e-3)

    def test_lev_monotonically_increasing(self, dist):
        limits = [50_000, 100_000, 200_000, 500_000, 1_000_000, 10_000_000]
        levs = [dist.limited_expected_value(d) for d in limits]
        for i in range(1, len(levs)):
            assert levs[i] >= levs[i - 1]

    def test_lev_matches_monte_carlo(self):
        """LEV should match Monte Carlo estimate."""
        dist = ParetoLoss(alpha=3.0, xm=50_000, seed=42)
        limit = 200_000
        samples = dist.generate_severity(500_000)
        mc_lev = float(np.mean(np.minimum(samples, limit)))
        analytical_lev = dist.limited_expected_value(limit)
        assert analytical_lev == pytest.approx(mc_lev, rel=0.02)

    def test_lev_known_formula(self):
        """Verify against hand-calculated values.

        Pareto(alpha=2, xm=100000):
        E[X] = 2 * 100000 / (2 - 1) = 200000
        LEV(200000) = 2*100000/(2-1) - 100000^2 * 200000^(-1) / (2-1)
                    = 200000 - 10000000000 / 200000 / 1
                    = 200000 - 50000 = 150000
        """
        dist = ParetoLoss(alpha=2.0, xm=100_000)
        lev = dist.limited_expected_value(200_000)
        assert lev == pytest.approx(150_000, rel=1e-10)

    def test_lev_alpha_equals_one(self):
        """Test special case alpha=1 (log formula)."""
        dist = ParetoLoss(alpha=1.0, xm=100_000)
        lev = dist.limited_expected_value(200_000)
        expected = 100_000 * (1 + np.log(200_000 / 100_000))
        assert lev == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Limited Expected Value: GPD
# ---------------------------------------------------------------------------
class TestGPDLEV:
    """Test GeneralizedParetoLoss.limited_expected_value()."""

    def test_lev_zero_limit(self):
        dist = GeneralizedParetoLoss(severity_shape=0.5, severity_scale=100_000)
        assert dist.limited_expected_value(0) == 0.0

    def test_lev_negative_limit(self):
        dist = GeneralizedParetoLoss(severity_shape=0.5, severity_scale=100_000)
        assert dist.limited_expected_value(-100) == 0.0

    def test_lev_exponential_case(self):
        """When xi=0, GPD is exponential: LEV(d) = beta*(1 - exp(-d/beta))."""
        beta = 100_000
        dist = GeneralizedParetoLoss(severity_shape=0.0, severity_scale=beta)
        limit = 200_000
        expected = beta * (1 - np.exp(-limit / beta))
        assert dist.limited_expected_value(limit) == pytest.approx(expected, rel=1e-10)

    def test_lev_heavy_tailed(self):
        """Test LEV with positive shape (heavy tail)."""
        dist = GeneralizedParetoLoss(severity_shape=0.5, severity_scale=100_000)
        lev = dist.limited_expected_value(500_000)
        # LEV should be less than E[X]
        assert lev < dist.expected_value()
        assert lev > 0

    def test_lev_bounded_support(self):
        """When xi < 0, distribution has bounded support. LEV(d) = E[X] for d >= upper bound."""
        xi = -0.5
        beta = 100_000
        dist = GeneralizedParetoLoss(severity_shape=xi, severity_scale=beta)
        upper_bound = -beta / xi  # = 200_000
        lev_at_bound = dist.limited_expected_value(upper_bound + 1)
        assert lev_at_bound == pytest.approx(dist.expected_value(), rel=1e-10)

    def test_lev_monotonically_increasing(self):
        dist = GeneralizedParetoLoss(severity_shape=0.3, severity_scale=100_000)
        limits = [10_000, 50_000, 100_000, 500_000, 1_000_000]
        levs = [dist.limited_expected_value(d) for d in limits]
        for i in range(1, len(levs)):
            assert levs[i] >= levs[i - 1]

    def test_lev_approaches_expected_value(self):
        dist = GeneralizedParetoLoss(severity_shape=0.3, severity_scale=100_000)
        lev = dist.limited_expected_value(1e15)
        assert lev == pytest.approx(dist.expected_value(), rel=1e-3)

    def test_lev_matches_monte_carlo(self):
        """LEV should match Monte Carlo estimate."""
        dist = GeneralizedParetoLoss(severity_shape=0.3, severity_scale=100_000, seed=42)
        limit = 500_000
        samples = dist.generate_severity(500_000)
        mc_lev = float(np.mean(np.minimum(samples, limit)))
        analytical_lev = dist.limited_expected_value(limit)
        assert analytical_lev == pytest.approx(mc_lev, rel=0.02)


# ---------------------------------------------------------------------------
# LayerPricer
# ---------------------------------------------------------------------------
class TestLayerPricer:
    """Test the LayerPricer analytical pricing class."""

    @pytest.fixture
    def pareto_pricer(self):
        severity = ParetoLoss(alpha=2.5, xm=100_000)
        return LayerPricer(severity, frequency=5.0)

    @pytest.fixture
    def lognormal_pricer(self):
        severity = LognormalLoss(mean=100_000, cv=1.5)
        return LayerPricer(severity, frequency=10.0)

    # -- expected_layer_loss --

    def test_expected_layer_loss_positive(self, pareto_pricer):
        """Layer loss should be positive for a valid layer."""
        loss = pareto_pricer.expected_layer_loss(200_000, 500_000)
        assert loss > 0

    def test_expected_layer_loss_zero_limit(self, pareto_pricer):
        assert pareto_pricer.expected_layer_loss(200_000, 0) == 0.0

    def test_expected_layer_loss_negative_limit(self, pareto_pricer):
        assert pareto_pricer.expected_layer_loss(200_000, -100) == 0.0

    def test_ground_up_plus_excess_equals_unlimited(self, pareto_pricer):
        """LEV(a+l) = LEV(a) + layer_loss/frequency for consistent layers."""
        attachment = 200_000
        limit = 800_000
        sev = pareto_pricer.severity
        lev_top = sev.limited_expected_value(attachment + limit)
        lev_bottom = sev.limited_expected_value(attachment)
        layer_per_occ = (
            pareto_pricer.expected_layer_loss(attachment, limit) / pareto_pricer.frequency
        )
        assert layer_per_occ == pytest.approx(lev_top - lev_bottom, rel=1e-10)

    def test_stacking_layers_equals_total(self):
        """Stacked layers should sum to the total limited loss."""
        severity = ParetoLoss(alpha=2.5, xm=50_000)
        pricer = LayerPricer(severity, frequency=1.0)

        # Ground-up to 500K split into two layers
        layer1 = pricer.expected_layer_loss(0, 250_000)  # 0 xs 0, limit 250K
        layer2 = pricer.expected_layer_loss(250_000, 250_000)  # 250K xs 250K
        total = pricer.expected_layer_loss(0, 500_000)  # 0 xs 0, limit 500K

        assert layer1 + layer2 == pytest.approx(total, rel=1e-10)

    def test_higher_layer_costs_less(self, lognormal_pricer):
        """Higher excess layers should generally have lower expected losses."""
        low_layer = lognormal_pricer.expected_layer_loss(100_000, 500_000)
        high_layer = lognormal_pricer.expected_layer_loss(600_000, 500_000)
        assert high_layer < low_layer

    def test_frequency_scales_linearly(self):
        """Layer loss should scale linearly with frequency."""
        severity = LognormalLoss(mean=100_000, cv=1.0)
        pricer_1 = LayerPricer(severity, frequency=1.0)
        pricer_5 = LayerPricer(severity, frequency=5.0)
        loss_1 = pricer_1.expected_layer_loss(50_000, 200_000)
        loss_5 = pricer_5.expected_layer_loss(50_000, 200_000)
        assert loss_5 == pytest.approx(5.0 * loss_1, rel=1e-10)

    # -- increased_limits_factor --

    def test_ilf_at_basic_limit_is_one(self, pareto_pricer):
        """ILF at the basic limit should be 1.0."""
        ilf = pareto_pricer.increased_limits_factor(500_000, basic_limit=500_000)
        assert ilf == pytest.approx(1.0, rel=1e-10)

    def test_ilf_above_basic_gt_one(self, pareto_pricer):
        """ILF above basic limit should exceed 1.0."""
        ilf = pareto_pricer.increased_limits_factor(1_000_000, basic_limit=500_000)
        assert ilf > 1.0

    def test_ilf_below_basic_lt_one(self, pareto_pricer):
        """ILF below basic limit should be less than 1.0."""
        ilf = pareto_pricer.increased_limits_factor(200_000, basic_limit=500_000)
        assert ilf < 1.0

    def test_ilf_monotonically_increasing(self, pareto_pricer):
        """ILFs should be monotonically increasing with limit."""
        basic = 100_000
        limits = [100_000, 200_000, 500_000, 1_000_000, 5_000_000]
        ilfs = [pareto_pricer.increased_limits_factor(l, basic) for l in limits]
        for i in range(1, len(ilfs)):
            assert ilfs[i] >= ilfs[i - 1]

    # -- loss_elimination_ratio --

    def test_ler_zero_deductible(self, pareto_pricer):
        """LER(0) should be 0."""
        ler = pareto_pricer.loss_elimination_ratio(0)
        assert ler == pytest.approx(0.0, abs=1e-10)

    def test_ler_large_deductible_approaches_one(self, lognormal_pricer):
        """LER should approach 1.0 for very large deductibles."""
        ler = lognormal_pricer.loss_elimination_ratio(1e15)
        assert ler == pytest.approx(1.0, rel=1e-3)

    def test_ler_between_zero_and_one(self, lognormal_pricer):
        """LER should be in [0, 1]."""
        ler = lognormal_pricer.loss_elimination_ratio(100_000)
        assert 0 <= ler <= 1

    def test_ler_monotonically_increasing(self, lognormal_pricer):
        """LER should increase with deductible."""
        deductibles = [10_000, 50_000, 100_000, 500_000, 1_000_000]
        lers = [lognormal_pricer.loss_elimination_ratio(d) for d in deductibles]
        for i in range(1, len(lers)):
            assert lers[i] >= lers[i - 1]

    def test_ler_infinite_mean_returns_zero(self):
        """LER should return 0 when E[X] is infinite."""
        severity = ParetoLoss(alpha=0.5, xm=100_000)  # alpha < 1 → infinite mean
        pricer = LayerPricer(severity, frequency=1.0)
        assert pricer.loss_elimination_ratio(500_000) == 0.0

    # -- exposure_curve --

    def test_exposure_curve_endpoints(self, lognormal_pricer):
        """Exposure curve should start at (0, 0) and end at (1, ~1)."""
        curve = lognormal_pricer.exposure_curve(n_points=50)
        assert curve["retention_pct"][0] == 0.0
        assert curve["loss_eliminated_pct"][0] == pytest.approx(0.0, abs=1e-6)
        assert curve["retention_pct"][-1] == 1.0
        assert curve["loss_eliminated_pct"][-1] == pytest.approx(1.0, rel=1e-3)

    def test_exposure_curve_length(self, lognormal_pricer):
        curve = lognormal_pricer.exposure_curve(n_points=100)
        assert len(curve["retention_pct"]) == 101
        assert len(curve["loss_eliminated_pct"]) == 101

    def test_exposure_curve_monotonic(self, lognormal_pricer):
        """Exposure curve should be monotonically non-decreasing."""
        curve = lognormal_pricer.exposure_curve(n_points=50)
        for i in range(1, len(curve["loss_eliminated_pct"])):
            assert curve["loss_eliminated_pct"][i] >= curve["loss_eliminated_pct"][i - 1]

    def test_exposure_curve_concave(self, lognormal_pricer):
        """Exposure curve is concave (above the 45-degree line).

        For right-skewed distributions, LEV grows faster at low retentions
        than at high ones, so the curve should be concave (above diagonal).
        """
        curve = lognormal_pricer.exposure_curve(n_points=100)
        # Check a mid-point: loss_eliminated > retention (concavity)
        mid_idx = 25  # 25% retention
        assert curve["loss_eliminated_pct"][mid_idx] > curve["retention_pct"][mid_idx]


# ---------------------------------------------------------------------------
# LayerPricer: cross-distribution consistency
# ---------------------------------------------------------------------------
class TestLayerPricerCrossDistribution:
    """Cross-distribution smoke tests."""

    @pytest.mark.parametrize(
        "dist",
        [
            LognormalLoss(mean=200_000, cv=1.5),
            ParetoLoss(alpha=2.5, xm=100_000),
            GeneralizedParetoLoss(severity_shape=0.3, severity_scale=100_000),
        ],
        ids=["lognormal", "pareto", "gpd"],
    )
    def test_layer_loss_positive(self, dist):
        pricer = LayerPricer(dist, frequency=5.0)
        loss = pricer.expected_layer_loss(50_000, 500_000)
        assert loss > 0

    @pytest.mark.parametrize(
        "dist",
        [
            LognormalLoss(mean=200_000, cv=1.5),
            ParetoLoss(alpha=2.5, xm=100_000),
            GeneralizedParetoLoss(severity_shape=0.3, severity_scale=100_000),
        ],
        ids=["lognormal", "pareto", "gpd"],
    )
    def test_ilf_at_basic_equals_one(self, dist):
        pricer = LayerPricer(dist, frequency=1.0)
        assert pricer.increased_limits_factor(500_000, 500_000) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "dist",
        [
            LognormalLoss(mean=200_000, cv=1.5),
            ParetoLoss(alpha=2.5, xm=100_000),
            GeneralizedParetoLoss(severity_shape=0.3, severity_scale=100_000),
        ],
        ids=["lognormal", "pareto", "gpd"],
    )
    def test_ler_bounded(self, dist):
        pricer = LayerPricer(dist, frequency=1.0)
        ler = pricer.loss_elimination_ratio(200_000)
        assert 0 <= ler <= 1

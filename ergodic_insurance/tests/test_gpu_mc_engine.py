"""Tests for the GPU-accelerated Monte Carlo simulation engine.

Unit tests always run with NumPy (no GPU required).  Statistical
equivalence tests compare GPU-path output against expected distributions.
GPU-specific tests are marked with ``@pytest.mark.gpu``.

Since:
    Version 0.11.0 (Issue #961)
"""

from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pytest

from ergodic_insurance.gpu_mc_engine import (
    _MAX_EVENTS_PER_YEAR,
    GPUSimulationParams,
    _scatter_events,
    apply_insurance_vectorized,
    extract_params,
    generate_losses_for_year,
    run_gpu_simulation,
    update_financial_state,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def default_params():
    """A GPUSimulationParams with sensible defaults for testing."""
    return GPUSimulationParams(
        initial_assets=10_000_000.0,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.10,
        tax_rate=0.21,
        retention_ratio=0.80,
        deductible=100_000.0,
        layer_attachments=[100_000.0, 1_000_000.0],
        layer_limits=[900_000.0, 4_000_000.0],
        layer_limit_types=[0, 0],  # both per-occurrence
        layer_aggregate_limits=[float("inf"), float("inf")],
        base_annual_premium=250_000.0,
        attritional_base_freq=3.0,
        attritional_scaling_exp=0.7,
        attritional_ref_revenue=10_000_000.0,
        attritional_mu=10.0,
        attritional_sigma=1.5,
        large_base_freq=0.3,
        large_scaling_exp=0.5,
        large_ref_revenue=10_000_000.0,
        large_mu=13.0,
        large_sigma=1.0,
        cat_base_freq=0.05,
        cat_scaling_exp=0.0,
        cat_ref_revenue=10_000_000.0,
        cat_alpha=1.5,
        cat_xm=1_000_000.0,
        n_simulations=500,
        n_years=5,
        insolvency_tolerance=10_000.0,
        seed=42,
    )


@pytest.fixture
def simple_params():
    """Minimal params — no insurance layers, low event counts."""
    return GPUSimulationParams(
        initial_assets=1_000_000.0,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.10,
        tax_rate=0.21,
        retention_ratio=1.0,
        deductible=0.0,
        layer_attachments=[],
        layer_limits=[],
        layer_limit_types=[],
        layer_aggregate_limits=[],
        base_annual_premium=0.0,
        attritional_base_freq=1.0,
        attritional_scaling_exp=0.0,
        attritional_ref_revenue=1_000_000.0,
        attritional_mu=8.0,
        attritional_sigma=1.0,
        large_base_freq=0.0,
        large_scaling_exp=0.0,
        large_ref_revenue=1_000_000.0,
        large_mu=12.0,
        large_sigma=1.0,
        cat_base_freq=0.0,
        cat_scaling_exp=0.0,
        cat_ref_revenue=1_000_000.0,
        cat_alpha=1.5,
        cat_xm=500_000.0,
        n_simulations=200,
        n_years=3,
        seed=123,
    )


# ── TestExtractParams ─────────────────────────────────────────────────────


class TestExtractParams:
    """Verify parameter extraction from real model objects."""

    def test_extracts_from_real_objects(self):
        """extract_params works with actual model instances."""
        from ergodic_insurance.config import ManufacturerConfig
        from ergodic_insurance.insurance_program import InsuranceProgram
        from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.monte_carlo import MonteCarloConfig

        mfg_config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.75,
        )
        manufacturer = WidgetManufacturer(mfg_config)
        insurance = InsuranceProgram.create_standard_manufacturing_program()
        loss_gen = ManufacturingLossGenerator()
        mc_config = MonteCarloConfig(n_simulations=1000, n_years=10, seed=42)

        params = extract_params(manufacturer, insurance, loss_gen, mc_config)

        assert params.initial_assets == 5_000_000
        assert params.asset_turnover_ratio == 1.2
        assert params.base_operating_margin == 0.08
        assert params.tax_rate == 0.25
        assert params.n_simulations == 1000
        assert params.n_years == 10
        assert params.seed == 42
        assert len(params.layer_attachments) == len(insurance.layers)
        assert params.base_annual_premium > 0
        assert params.attritional_base_freq > 0

    def test_deductible_extracted(self):
        """Deductible from InsuranceProgram is captured."""
        from ergodic_insurance.config import ManufacturerConfig
        from ergodic_insurance.insurance_program import InsuranceProgram
        from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.monte_carlo import MonteCarloConfig

        manufacturer = WidgetManufacturer(ManufacturerConfig())
        insurance = InsuranceProgram.create_standard_manufacturing_program(deductible=500_000)
        loss_gen = ManufacturingLossGenerator()
        mc_config = MonteCarloConfig()

        params = extract_params(manufacturer, insurance, loss_gen, mc_config)
        assert params.deductible == 500_000


# ── TestGenerateLosses ────────────────────────────────────────────────────


class TestGenerateLosses:
    """Loss generation: shapes, positivity, determinism, frequency."""

    def test_output_shapes(self, default_params):
        xp = np
        rng = np.random.default_rng(42)
        dtype = np.dtype(np.float64)
        revenues = xp.full(default_params.n_simulations, 10_000_000.0, dtype=dtype)

        losses, n_events = generate_losses_for_year(default_params, revenues, xp, rng, dtype)

        assert losses.shape[0] == default_params.n_simulations
        assert losses.ndim == 2
        assert n_events.shape == (default_params.n_simulations,)

    def test_losses_non_negative(self, default_params):
        xp = np
        rng = np.random.default_rng(42)
        dtype = np.dtype(np.float64)
        revenues = xp.full(default_params.n_simulations, 10_000_000.0, dtype=dtype)

        losses, _ = generate_losses_for_year(default_params, revenues, xp, rng, dtype)

        assert np.all(losses >= 0)

    def test_deterministic_with_seed(self, default_params):
        xp = np
        dtype = np.dtype(np.float64)
        revenues = xp.full(default_params.n_simulations, 10_000_000.0, dtype=dtype)

        rng1 = np.random.default_rng(99)
        losses1, n1 = generate_losses_for_year(default_params, revenues, xp, rng1, dtype)

        rng2 = np.random.default_rng(99)
        losses2, n2 = generate_losses_for_year(default_params, revenues, xp, rng2, dtype)

        np.testing.assert_array_equal(losses1, losses2)
        np.testing.assert_array_equal(n1, n2)

    def test_frequency_reasonable(self, default_params):
        """Mean event count should be close to expected frequency."""
        xp = np
        rng = np.random.default_rng(42)
        dtype = np.dtype(np.float64)
        n_sims = 5000
        revenues = xp.full(n_sims, default_params.attritional_ref_revenue, dtype=dtype)

        params = GPUSimulationParams(
            **{
                **default_params.__dict__,
                "n_simulations": n_sims,
                "large_base_freq": 0.0,
                "cat_base_freq": 0.0,
            }
        )

        _, n_events = generate_losses_for_year(params, revenues, xp, rng, dtype)

        # At reference revenue with scaling exp, expected freq = base_freq
        mean_events = np.mean(n_events)
        assert abs(mean_events - params.attritional_base_freq) < 0.5

    def test_zero_frequency_gives_no_events(self):
        """When all frequencies are 0, no events are generated."""
        xp = np
        rng = np.random.default_rng(42)
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            attritional_base_freq=0.0,
            large_base_freq=0.0,
            cat_base_freq=0.0,
            n_simulations=100,
        )
        revenues = xp.full(100, 10_000_000.0, dtype=dtype)

        losses, n_events = generate_losses_for_year(params, revenues, xp, rng, dtype)

        assert np.all(n_events == 0)


# ── TestApplyInsurance ────────────────────────────────────────────────────


class TestApplyInsurance:
    """Insurance layer logic: deductible, limits, multi-layer, aggregate."""

    def test_below_deductible_zero_recovery(self):
        """Losses below deductible yield zero insurance recovery."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=100_000.0,
            layer_attachments=[100_000.0],
            layer_limits=[900_000.0],
            layer_limit_types=[0],
            layer_aggregate_limits=[float("inf")],
        )

        # Single event of 50,000 (below deductible)
        loss_amounts = xp.array([[50_000.0]], dtype=dtype)
        n_events = xp.array([1], dtype=xp.int32)

        _, recoveries, retained = apply_insurance_vectorized(
            loss_amounts, n_events, params, xp, dtype
        )

        assert float(recoveries[0]) == 0.0
        assert float(retained[0]) == 50_000.0

    def test_within_limits_correct_recovery(self):
        """Loss within layer limits produces correct recovery."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=100_000.0,
            layer_attachments=[100_000.0],
            layer_limits=[900_000.0],
            layer_limit_types=[0],
            layer_aggregate_limits=[float("inf")],
        )

        # Loss of 500,000: excess above 100k attachment = 400k, within 900k limit
        loss_amounts = xp.array([[500_000.0]], dtype=dtype)
        n_events = xp.array([1], dtype=xp.int32)

        _, recoveries, retained = apply_insurance_vectorized(
            loss_amounts, n_events, params, xp, dtype
        )

        assert abs(float(recoveries[0]) - 400_000.0) < 1.0
        assert abs(float(retained[0]) - 100_000.0) < 1.0

    def test_capped_at_limit(self):
        """Recovery is capped at layer limit."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=100_000.0,
            layer_attachments=[100_000.0],
            layer_limits=[500_000.0],
            layer_limit_types=[0],
            layer_aggregate_limits=[float("inf")],
        )

        # Loss of 2,000,000: excess = 1.9M but limit is 500k
        loss_amounts = xp.array([[2_000_000.0]], dtype=dtype)
        n_events = xp.array([1], dtype=xp.int32)

        _, recoveries, retained = apply_insurance_vectorized(
            loss_amounts, n_events, params, xp, dtype
        )

        assert abs(float(recoveries[0]) - 500_000.0) < 1.0
        assert abs(float(retained[0]) - 1_500_000.0) < 1.0

    def test_multi_layer(self):
        """Multiple layers stack correctly."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=50_000.0,
            layer_attachments=[50_000.0, 500_000.0],
            layer_limits=[450_000.0, 2_000_000.0],
            layer_limit_types=[0, 0],
            layer_aggregate_limits=[float("inf"), float("inf")],
        )

        # Loss of 1,000,000
        # Layer 1: excess above 50k = 950k, capped at 450k → recovery 450k
        # Layer 2: excess above 500k = 500k, capped at 2M → recovery 500k
        # Total recovery = 950k, retained = 50k
        loss_amounts = xp.array([[1_000_000.0]], dtype=dtype)
        n_events = xp.array([1], dtype=xp.int32)

        _, recoveries, retained = apply_insurance_vectorized(
            loss_amounts, n_events, params, xp, dtype
        )

        assert abs(float(recoveries[0]) - 950_000.0) < 1.0
        assert abs(float(retained[0]) - 50_000.0) < 1.0

    def test_padding_ignored(self):
        """Padding zeros beyond n_events are ignored."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=0.0,
            layer_attachments=[0.0],
            layer_limits=[1_000_000.0],
            layer_limit_types=[0],
            layer_aggregate_limits=[float("inf")],
        )

        # 2 actual events, 1 padding zero
        loss_amounts = xp.array([[100_000.0, 200_000.0, 999_999.0]], dtype=dtype)
        n_events = xp.array([2], dtype=xp.int32)

        total, recoveries, retained = apply_insurance_vectorized(
            loss_amounts, n_events, params, xp, dtype
        )

        # Total should be 100k + 200k = 300k (padding ignored)
        assert abs(float(total[0]) - 300_000.0) < 1.0

    def test_aggregate_tracking(self):
        """Aggregate limit caps total annual recovery across events."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=0.0,
            layer_attachments=[0.0],
            layer_limits=[500_000.0],
            layer_limit_types=[1],  # aggregate
            layer_aggregate_limits=[800_000.0],
        )

        # 3 events of 500k each → layer would pay 500k each but aggregate cap = 800k
        loss_amounts = xp.array([[500_000.0, 500_000.0, 500_000.0]], dtype=dtype)
        n_events = xp.array([3], dtype=xp.int32)

        _, recoveries, _ = apply_insurance_vectorized(loss_amounts, n_events, params, xp, dtype)

        assert abs(float(recoveries[0]) - 800_000.0) < 1.0

    def test_no_layers_no_recovery(self):
        """With no insurance layers, recovery is always zero."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            deductible=0.0,
            layer_attachments=[],
            layer_limits=[],
            layer_limit_types=[],
            layer_aggregate_limits=[],
        )

        loss_amounts = xp.array([[500_000.0]], dtype=dtype)
        n_events = xp.array([1], dtype=xp.int32)

        _, recoveries, retained = apply_insurance_vectorized(
            loss_amounts, n_events, params, xp, dtype
        )

        assert float(recoveries[0]) == 0.0
        assert abs(float(retained[0]) - 500_000.0) < 1.0


# ── TestFinancialUpdate ──────────────────────────────────────────────────


class TestFinancialUpdate:
    """Financial state transitions."""

    def test_growth_with_no_losses(self):
        """Assets grow when there are no losses."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            base_operating_margin=0.10,
            tax_rate=0.21,
            retention_ratio=1.0,
        )

        assets = xp.array([1_000_000.0], dtype=dtype)
        retained = xp.array([0.0], dtype=dtype)
        revenues = xp.array([1_000_000.0], dtype=dtype)
        premium = xp.array([0.0], dtype=dtype)

        new_assets, _ = update_financial_state(
            assets, retained, revenues, premium, params, xp, dtype
        )

        # Operating income = 100k, tax = 21k, net = 79k
        expected = 1_000_000.0 + 79_000.0
        assert abs(float(new_assets[0]) - expected) < 1.0

    def test_shrinkage_with_losses(self):
        """Assets shrink when retained losses exceed operating income."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            base_operating_margin=0.10,
            tax_rate=0.21,
            retention_ratio=1.0,
        )

        assets = xp.array([1_000_000.0], dtype=dtype)
        retained = xp.array([200_000.0], dtype=dtype)  # > 100k operating income
        revenues = xp.array([1_000_000.0], dtype=dtype)
        premium = xp.array([0.0], dtype=dtype)

        new_assets, _ = update_financial_state(
            assets, retained, revenues, premium, params, xp, dtype
        )

        # pre_tax = 100k - 200k = -100k → tax = 0 → net = -100k
        expected = 1_000_000.0 - 100_000.0
        assert abs(float(new_assets[0]) - expected) < 1.0

    def test_no_tax_on_negative_income(self):
        """Tax is zero when pre-tax income is negative."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            base_operating_margin=0.05,
            tax_rate=0.30,
            retention_ratio=1.0,
        )

        assets = xp.array([1_000_000.0], dtype=dtype)
        retained = xp.array([100_000.0], dtype=dtype)
        revenues = xp.array([1_000_000.0], dtype=dtype)
        premium = xp.array([50_000.0], dtype=dtype)

        new_assets, _ = update_financial_state(
            assets, retained, revenues, premium, params, xp, dtype
        )

        # operating = 50k, pre_tax = 50k - 100k - 50k = -100k
        # tax = max(0, -100k) * 0.30 = 0
        # net = -100k, assets += -100k
        expected = 1_000_000.0 - 100_000.0
        assert abs(float(new_assets[0]) - expected) < 1.0

    def test_retention_ratio(self):
        """Only retention_ratio fraction of net income flows to assets."""
        xp = np
        dtype = np.dtype(np.float64)
        params = GPUSimulationParams(
            base_operating_margin=0.10,
            tax_rate=0.0,
            retention_ratio=0.5,
        )

        assets = xp.array([1_000_000.0], dtype=dtype)
        retained = xp.array([0.0], dtype=dtype)
        revenues = xp.array([1_000_000.0], dtype=dtype)
        premium = xp.array([0.0], dtype=dtype)

        new_assets, _ = update_financial_state(
            assets, retained, revenues, premium, params, xp, dtype
        )

        # net = 100k, retained_earnings = 100k * 0.5 = 50k
        expected = 1_000_000.0 + 50_000.0
        assert abs(float(new_assets[0]) - expected) < 1.0


# ── TestRunGPUSimulation ─────────────────────────────────────────────────


class TestRunGPUSimulation:
    """End-to-end smoke tests for run_gpu_simulation."""

    def test_smoke_test(self, default_params):
        """Basic run completes and returns expected keys."""
        result = run_gpu_simulation(default_params)

        assert "final_assets" in result
        assert "final_equity" in result
        assert "annual_losses" in result
        assert "insurance_recoveries" in result
        assert "retained_losses" in result

    def test_output_shapes(self, default_params):
        """Output arrays have correct shapes."""
        result = run_gpu_simulation(default_params)

        n_sims = default_params.n_simulations
        n_years = default_params.n_years

        assert result["final_assets"].shape == (n_sims,)
        assert result["final_equity"].shape == (n_sims,)
        assert result["annual_losses"].shape == (n_sims, n_years)
        assert result["insurance_recoveries"].shape == (n_sims, n_years)
        assert result["retained_losses"].shape == (n_sims, n_years)

    def test_numpy_arrays_returned(self, default_params):
        """All returned arrays are numpy (not cupy)."""
        result = run_gpu_simulation(default_params)

        for key, arr in result.items():
            assert isinstance(arr, np.ndarray), f"{key} is not numpy"

    def test_ruin_detection(self):
        """Simulations with very high losses should show some ruin."""
        params = GPUSimulationParams(
            initial_assets=100_000.0,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.05,
            tax_rate=0.21,
            retention_ratio=1.0,
            deductible=0.0,
            layer_attachments=[],
            layer_limits=[],
            layer_limit_types=[],
            layer_aggregate_limits=[],
            base_annual_premium=0.0,
            attritional_base_freq=5.0,
            attritional_scaling_exp=0.0,
            attritional_ref_revenue=100_000.0,
            attritional_mu=11.0,  # mean ~60k per event
            attritional_sigma=0.5,
            large_base_freq=1.0,
            large_scaling_exp=0.0,
            large_ref_revenue=100_000.0,
            large_mu=12.0,
            large_sigma=0.5,
            cat_base_freq=0.0,
            cat_scaling_exp=0.0,
            cat_ref_revenue=100_000.0,
            cat_alpha=1.5,
            cat_xm=100_000.0,
            n_simulations=500,
            n_years=10,
            insolvency_tolerance=10_000.0,
            seed=42,
        )

        result = run_gpu_simulation(params)

        # With these extreme loss parameters, many paths should be ruined
        ruined = np.sum(result["final_assets"] <= params.insolvency_tolerance)
        assert ruined > 0, "Expected at least some paths to be ruined"

    def test_deterministic_with_seed(self, default_params):
        """Same seed produces same results."""
        result1 = run_gpu_simulation(default_params)
        result2 = run_gpu_simulation(default_params)

        np.testing.assert_array_equal(result1["final_assets"], result2["final_assets"])
        np.testing.assert_array_equal(result1["annual_losses"], result2["annual_losses"])

    def test_progress_callback(self, simple_params):
        """Progress callback is invoked."""
        calls = []

        def callback(completed, total, elapsed):
            calls.append((completed, total))

        run_gpu_simulation(simple_params, progress_callback=callback)

        assert len(calls) == simple_params.n_years
        assert calls[-1][0] == simple_params.n_years

    def test_cancel_event(self, default_params):
        """Cancel event stops simulation early."""
        import threading

        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        result = run_gpu_simulation(default_params, cancel_event=cancel)

        # Should still return valid arrays (possibly with zeros for uncompleted years)
        assert "final_assets" in result

    def test_float32_mode(self, default_params):
        """float32 mode produces float32 arrays."""
        params = GPUSimulationParams(**{**default_params.__dict__, "use_float32": True})
        result = run_gpu_simulation(params)

        assert result["annual_losses"].dtype == np.float32
        assert result["final_assets"].dtype == np.float32


# ── TestStatisticalEquivalence ───────────────────────────────────────────


class TestStatisticalEquivalence:
    """Compare GPU-path output against statistical expectations.

    These tests use moderately large N to check that the reduced-form
    model produces results in the right ballpark.
    """

    def test_mean_losses_reasonable(self):
        """Mean total losses should be positive and bounded."""
        params = GPUSimulationParams(
            initial_assets=10_000_000.0,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.10,
            tax_rate=0.21,
            retention_ratio=0.80,
            deductible=100_000.0,
            layer_attachments=[100_000.0],
            layer_limits=[5_000_000.0],
            layer_limit_types=[0],
            layer_aggregate_limits=[float("inf")],
            base_annual_premium=200_000.0,
            attritional_base_freq=3.0,
            attritional_scaling_exp=0.7,
            attritional_ref_revenue=10_000_000.0,
            attritional_mu=10.0,
            attritional_sigma=1.5,
            large_base_freq=0.3,
            large_scaling_exp=0.5,
            large_ref_revenue=10_000_000.0,
            large_mu=13.0,
            large_sigma=1.0,
            cat_base_freq=0.05,
            cat_scaling_exp=0.0,
            cat_ref_revenue=10_000_000.0,
            cat_alpha=1.5,
            cat_xm=1_000_000.0,
            n_simulations=2000,
            n_years=10,
            seed=42,
        )

        result = run_gpu_simulation(params)

        mean_annual_loss = np.mean(result["annual_losses"])
        assert mean_annual_loss > 0, "Mean loss should be positive"
        # Losses shouldn't exceed assets by an absurd factor
        assert mean_annual_loss < params.initial_assets * 10

    def test_insurance_reduces_retained(self):
        """Insurance should reduce retained losses compared to no-insurance."""
        # No insurance
        params_none = GPUSimulationParams(
            initial_assets=10_000_000.0,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.10,
            tax_rate=0.21,
            retention_ratio=0.80,
            attritional_base_freq=3.0,
            attritional_scaling_exp=0.7,
            attritional_ref_revenue=10_000_000.0,
            attritional_mu=10.0,
            attritional_sigma=1.5,
            large_base_freq=0.3,
            large_scaling_exp=0.5,
            large_ref_revenue=10_000_000.0,
            large_mu=13.0,
            large_sigma=1.0,
            cat_base_freq=0.05,
            cat_scaling_exp=0.0,
            cat_ref_revenue=10_000_000.0,
            cat_alpha=1.5,
            cat_xm=1_000_000.0,
            n_simulations=2000,
            n_years=5,
            seed=42,
            deductible=0.0,
            layer_attachments=[],
            layer_limits=[],
            layer_limit_types=[],
            layer_aggregate_limits=[],
            base_annual_premium=0.0,
        )

        # With insurance (same loss parameters, different insurance)
        params_ins = GPUSimulationParams(
            initial_assets=10_000_000.0,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.10,
            tax_rate=0.21,
            retention_ratio=0.80,
            attritional_base_freq=3.0,
            attritional_scaling_exp=0.7,
            attritional_ref_revenue=10_000_000.0,
            attritional_mu=10.0,
            attritional_sigma=1.5,
            large_base_freq=0.3,
            large_scaling_exp=0.5,
            large_ref_revenue=10_000_000.0,
            large_mu=13.0,
            large_sigma=1.0,
            cat_base_freq=0.05,
            cat_scaling_exp=0.0,
            cat_ref_revenue=10_000_000.0,
            cat_alpha=1.5,
            cat_xm=1_000_000.0,
            n_simulations=2000,
            n_years=5,
            seed=42,
            deductible=100_000.0,
            layer_attachments=[100_000.0],
            layer_limits=[5_000_000.0],
            layer_limit_types=[0],
            layer_aggregate_limits=[float("inf")],
            base_annual_premium=200_000.0,
        )

        result_none = run_gpu_simulation(params_none)
        result_ins = run_gpu_simulation(params_ins)

        mean_retained_none = np.mean(np.sum(result_none["retained_losses"], axis=1))
        mean_retained_ins = np.mean(np.sum(result_ins["retained_losses"], axis=1))

        assert (
            mean_retained_ins < mean_retained_none
        ), "Insurance should reduce mean retained losses"

    def test_ruin_probability_bounded(self):
        """Ruin probability should be between 0 and 1."""
        params = GPUSimulationParams(
            n_simulations=3000,
            n_years=10,
            seed=42,
        )

        result = run_gpu_simulation(params)
        ruin_rate = np.mean(result["final_assets"] <= params.insolvency_tolerance)

        assert 0.0 <= ruin_rate <= 1.0


# ── TestGPUFallback ──────────────────────────────────────────────────────


@pytest.mark.gpu
class TestGPUFallback:
    """GPU-specific: verify fallback behavior."""

    def test_fallback_to_numpy_when_unavailable(self, default_params):
        """When GPU is unavailable, simulation still works via NumPy."""
        import ergodic_insurance.gpu_backend as gpu_mod

        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            result = run_gpu_simulation(default_params)

        assert isinstance(result["final_assets"], np.ndarray)
        assert result["final_assets"].shape == (default_params.n_simulations,)


@pytest.mark.gpu
class TestGPUMemoryCleanup:
    """GPU-specific: verify memory pool usage."""

    def test_memory_pool_context_used(self, default_params):
        """run_gpu_simulation uses gpu_memory_pool context manager."""
        import ergodic_insurance.gpu_mc_engine as engine_mod

        original_pool = engine_mod.gpu_memory_pool
        pool_entered = []

        class MockPool:
            def __enter__(self):
                pool_entered.append(True)
                return self

            def __exit__(self, *args):
                pass

        with patch.object(engine_mod, "gpu_memory_pool", MockPool):
            run_gpu_simulation(default_params)

        assert len(pool_entered) > 0, "gpu_memory_pool context manager was not used"


# ── TestMonteCarloEngineGPUIntegration ───────────────────────────────────


class TestMonteCarloEngineGPUIntegration:
    """Integration: MonteCarloEngine with use_gpu=True."""

    def test_gpu_flag_in_config(self):
        """MonteCarloConfig accepts use_gpu field."""
        from ergodic_insurance.monte_carlo import MonteCarloConfig

        config = MonteCarloConfig(use_gpu=True)
        assert config.use_gpu is True

    def test_gpu_path_fallback_warns(self, caplog):
        """With use_gpu=True and no GPU, a warning is issued and CPU is used."""
        import logging

        from ergodic_insurance.config import ManufacturerConfig
        import ergodic_insurance.gpu_backend as gpu_mod
        from ergodic_insurance.insurance_program import InsuranceProgram
        from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloEngine

        config = MonteCarloConfig(
            n_simulations=100,
            n_years=3,
            use_gpu=True,
            parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        manufacturer = WidgetManufacturer(ManufacturerConfig())
        insurance = InsuranceProgram.create_standard_manufacturing_program()
        loss_gen = ManufacturingLossGenerator()

        engine = MonteCarloEngine(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            config=config,
        )

        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            with caplog.at_level(logging.WARNING, logger="ergodic_insurance.monte_carlo"):
                results = engine.run()

                gpu_warnings = [
                    r
                    for r in caplog.records
                    if "use_gpu" in r.message.lower() or "GPU" in r.message
                ]
                assert len(gpu_warnings) > 0, "Expected a GPU fallback warning"

        # Should still produce valid results
        assert results.final_assets.shape == (100,)
        assert results.annual_losses.shape == (100, 3)


# ── TestScatterEvents ────────────────────────────────────────────────────


def _scatter_events_reference(
    loss_amounts_cpu,
    att_counts,
    lg_counts,
    cat_counts,
    att_sevs,
    lg_sevs,
    cat_sevs,
    max_events,
):
    """Scalar reference implementation (the original Python for-loop)."""
    n_sims = loss_amounts_cpu.shape[0]
    att_ptr = lg_ptr = cat_ptr = 0
    for i in range(n_sims):
        col = 0
        for counts_i, sevs_arr, ptr_name in (
            (att_counts[i], att_sevs, "att"),
            (lg_counts[i], lg_sevs, "lg"),
            (cat_counts[i], cat_sevs, "cat"),
        ):
            if counts_i > 0:
                end = min(col + counts_i, max_events)
                count = end - col
                ptr = {"att": att_ptr, "lg": lg_ptr, "cat": cat_ptr}[ptr_name]
                loss_amounts_cpu[i, col:end] = sevs_arr[ptr : ptr + count]
                if ptr_name == "att":
                    att_ptr += counts_i
                elif ptr_name == "lg":
                    lg_ptr += counts_i
                else:
                    cat_ptr += counts_i
                col = end


class TestScatterEvents:
    """Vectorized _scatter_events matches the scalar reference."""

    def test_matches_reference_random(self):
        """Random counts and severities produce identical output."""
        rng = np.random.default_rng(777)
        n_sims = 500
        max_events = 20

        att_counts = rng.poisson(3.0, size=n_sims).astype(np.int32)
        lg_counts = rng.poisson(0.5, size=n_sims).astype(np.int32)
        cat_counts = rng.poisson(0.1, size=n_sims).astype(np.int32)

        att_sevs = rng.lognormal(10, 1.5, size=int(np.sum(att_counts)))
        lg_sevs = rng.lognormal(13, 1.0, size=int(np.sum(lg_counts)))
        cat_sevs = rng.lognormal(14, 0.5, size=int(np.sum(cat_counts)))

        # Vectorized
        result_vec = np.zeros((n_sims, max_events), dtype=np.float64)
        _scatter_events(
            result_vec,
            att_counts,
            lg_counts,
            cat_counts,
            att_sevs,
            lg_sevs,
            cat_sevs,
            max_events,
        )

        # Reference
        result_ref = np.zeros((n_sims, max_events), dtype=np.float64)
        _scatter_events_reference(
            result_ref,
            att_counts,
            lg_counts,
            cat_counts,
            att_sevs,
            lg_sevs,
            cat_sevs,
            max_events,
        )

        np.testing.assert_array_equal(result_vec, result_ref)

    def test_clipping_when_total_exceeds_max_events(self):
        """Events beyond max_events are correctly dropped."""
        n_sims = 3
        max_events = 4
        # Sim 0: 3+2+1=6 events but max=4 → only first 4
        # Sim 1: 1+0+0=1 event
        # Sim 2: 0+0+3=3 events
        att_counts = np.array([3, 1, 0], dtype=np.int32)
        lg_counts = np.array([2, 0, 0], dtype=np.int32)
        cat_counts = np.array([1, 0, 3], dtype=np.int32)

        att_sevs = np.array([10.0, 20.0, 30.0, 40.0])  # 3+1=4
        lg_sevs = np.array([100.0, 200.0])  # 2+0+0=2
        cat_sevs = np.array([1000.0, 2000.0, 3000.0, 4000.0])  # 1+0+3=4

        result_vec = np.zeros((n_sims, max_events), dtype=np.float64)
        _scatter_events(
            result_vec,
            att_counts,
            lg_counts,
            cat_counts,
            att_sevs,
            lg_sevs,
            cat_sevs,
            max_events,
        )

        result_ref = np.zeros((n_sims, max_events), dtype=np.float64)
        _scatter_events_reference(
            result_ref,
            att_counts,
            lg_counts,
            cat_counts,
            att_sevs,
            lg_sevs,
            cat_sevs,
            max_events,
        )

        np.testing.assert_array_equal(result_vec, result_ref)

    def test_all_zeros(self):
        """All-zero counts produce an unchanged output array."""
        n_sims = 10
        max_events = 5
        zeros = np.zeros(n_sims, dtype=np.int32)
        empty = np.empty(0, dtype=np.float64)

        result = np.zeros((n_sims, max_events), dtype=np.float64)
        _scatter_events(result, zeros, zeros, zeros, empty, empty, empty, max_events)

        np.testing.assert_array_equal(result, 0.0)

    def test_benchmark_no_python_loop(self):
        """Vectorized scatter is >= 10x faster than reference at 100K sims."""
        import time

        rng = np.random.default_rng(42)
        n_sims = 100_000
        max_events = 20

        att_counts = rng.poisson(3.0, size=n_sims).astype(np.int32)
        lg_counts = rng.poisson(0.3, size=n_sims).astype(np.int32)
        cat_counts = rng.poisson(0.05, size=n_sims).astype(np.int32)

        att_sevs = rng.lognormal(10, 1.5, size=int(np.sum(att_counts)))
        lg_sevs = rng.lognormal(13, 1.0, size=int(np.sum(lg_counts)))
        cat_sevs = rng.lognormal(14, 0.5, size=int(np.sum(cat_counts)))

        # Warm up
        buf = np.zeros((n_sims, max_events), dtype=np.float64)
        _scatter_events(
            buf, att_counts, lg_counts, cat_counts, att_sevs, lg_sevs, cat_sevs, max_events
        )

        # Time vectorized
        n_runs = 3
        t0 = time.perf_counter()
        for _ in range(n_runs):
            buf[:] = 0.0
            _scatter_events(
                buf, att_counts, lg_counts, cat_counts, att_sevs, lg_sevs, cat_sevs, max_events
            )
        t_vec = (time.perf_counter() - t0) / n_runs

        # Time reference
        t0 = time.perf_counter()
        for _ in range(n_runs):
            buf[:] = 0.0
            _scatter_events_reference(
                buf, att_counts, lg_counts, cat_counts, att_sevs, lg_sevs, cat_sevs, max_events
            )
        t_ref = (time.perf_counter() - t0) / n_runs

        speedup = t_ref / t_vec
        assert speedup >= 10, (
            f"Expected >= 10x speedup, got {speedup:.1f}x "
            f"(vec={t_vec*1000:.1f}ms, ref={t_ref*1000:.1f}ms)"
        )


# ── TestLazyImports ──────────────────────────────────────────────────────


class TestLazyImports:
    """Top-level lazy imports work."""

    def test_import_gpu_simulation_params(self):
        import importlib

        mod = importlib.import_module("ergodic_insurance")
        cls = getattr(mod, "GPUSimulationParams")
        assert cls is not None

    def test_import_run_gpu_simulation(self):
        import importlib

        mod = importlib.import_module("ergodic_insurance")
        func = getattr(mod, "run_gpu_simulation")
        assert callable(func)

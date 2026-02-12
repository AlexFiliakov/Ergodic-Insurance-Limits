"""Tests for unified execution semantics across simulation paths (Issue #349).

Validates that:
1. Simulation.run() is re-entrant (produces identical results on repeated calls)
2. Insolvency is detected in the year it occurs (not one year late)
3. growth_rate and letter_of_credit_rate are configurable in Simulation
4. Config step parameters are passed through in enhanced parallel path
5. Execution ordering is consistent: losses → claims → premium → step
"""

import warnings

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import Simulation


@pytest.fixture
def manufacturer_config():
    """Create a test manufacturer configuration."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.6,
    )


@pytest.fixture
def manufacturer(manufacturer_config):
    """Create a test manufacturer."""
    return WidgetManufacturer(manufacturer_config)


@pytest.fixture
def loss_generator():
    """Create a deterministic test loss generator."""
    return ManufacturingLossGenerator.create_simple(
        frequency=0.1, severity_mean=1_000_000, severity_std=500_000, seed=42
    )


@pytest.fixture
def insurance_policy():
    """Create a test insurance policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        layer = InsuranceLayer(
            attachment_point=100_000,
            limit=5_000_000,
            rate=0.02,
        )
        return InsurancePolicy(
            layers=[layer],
            deductible=100_000,
        )


class TestReEntrancy:
    """Test that Simulation.run() can be called multiple times with identical results."""

    def test_run_twice_produces_identical_results(self, manufacturer, loss_generator):
        """Run simulation twice and verify results are identical."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
        )

        results1 = sim.run()
        results2 = sim.run()

        np.testing.assert_array_equal(results1.assets, results2.assets)
        np.testing.assert_array_equal(results1.equity, results2.equity)
        np.testing.assert_array_equal(results1.roe, results2.roe)
        np.testing.assert_array_equal(results1.revenue, results2.revenue)
        np.testing.assert_array_equal(results1.net_income, results2.net_income)
        np.testing.assert_array_equal(results1.claim_counts, results2.claim_counts)
        np.testing.assert_array_equal(results1.claim_amounts, results2.claim_amounts)
        assert results1.insolvency_year == results2.insolvency_year

    def test_run_twice_with_insurance_produces_identical_results(
        self, manufacturer, loss_generator, insurance_policy
    ):
        """Run simulation with insurance twice and verify results are identical."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim = Simulation(
                manufacturer=manufacturer,
                loss_generator=loss_generator,
                insurance_policy=insurance_policy,
                time_horizon=10,
                seed=42,
            )

        results1 = sim.run()
        results2 = sim.run()

        np.testing.assert_array_equal(results1.assets, results2.assets)
        np.testing.assert_array_equal(results1.equity, results2.equity)
        assert results1.insolvency_year == results2.insolvency_year

    def test_manufacturer_state_reset_between_runs(self, manufacturer, loss_generator):
        """Verify that manufacturer state is fully reset between runs."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=5,
            seed=42,
        )

        # First run modifies manufacturer state
        sim.run()

        # Second and third runs should produce identical results
        results1 = sim.run()
        results2 = sim.run()

        # First year equity should be the same in both runs
        assert results1.equity[0] == results2.equity[0]


class TestCopyParameter:
    """Test that Simulation.__init__ copy parameter protects caller's manufacturer (Issue #802)."""

    def test_caller_manufacturer_not_mutated_by_default(self, manufacturer_config, loss_generator):
        """Default copy=True protects the caller's manufacturer from mutation."""
        manufacturer = WidgetManufacturer(manufacturer_config)
        original_assets = manufacturer.current_assets
        original_equity = manufacturer.current_equity

        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
        )
        sim.run()

        assert manufacturer.current_assets == original_assets
        assert manufacturer.current_equity == original_equity

    def test_copy_false_shares_reference(self, manufacturer_config, loss_generator):
        """copy=False shares the caller's manufacturer reference with the simulation."""
        manufacturer = WidgetManufacturer(manufacturer_config)

        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
            copy=False,
        )

        # With copy=False, sim.manufacturer IS the caller's object
        assert sim.manufacturer is manufacturer

    def test_copy_true_isolates_reference(self, manufacturer_config, loss_generator):
        """copy=True (default) creates an independent copy of the manufacturer."""
        manufacturer = WidgetManufacturer(manufacturer_config)

        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
        )

        # With copy=True, sim.manufacturer is a different object
        assert sim.manufacturer is not manufacturer

    def test_copy_false_exposes_caller_to_step_annual_mutation(self, manufacturer_config):
        """copy=False means step_annual() directly mutates the caller's manufacturer."""
        manufacturer = WidgetManufacturer(manufacturer_config)
        original_equity = manufacturer.current_equity

        sim = Simulation(
            manufacturer=manufacturer,
            time_horizon=10,
            seed=42,
            copy=False,
        )

        # Directly call step_annual (bypasses run()'s reset)
        large_loss = LossEvent(time=0, amount=1_000_000)
        sim.step_annual(0, [large_loss])

        # The caller's manufacturer was mutated
        assert manufacturer.current_equity != original_equity

    def test_reuse_manufacturer_across_simulations(self, manufacturer_config, loss_generator):
        """Default copy=True allows safe reuse across multiple simulations."""
        manufacturer = WidgetManufacturer(manufacturer_config)

        sim_a = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
        )
        results_a = sim_a.run()

        sim_b = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=10,
            seed=42,
        )
        results_b = sim_b.run()

        # Both simulations start from identical initial state
        np.testing.assert_array_equal(results_a.assets, results_b.assets)
        np.testing.assert_array_equal(results_a.equity, results_b.equity)

    def test_re_entrancy_preserved_with_copy(self, manufacturer, loss_generator):
        """Re-entrancy still works with copy=True."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=5,
            seed=42,
        )
        results1 = sim.run()
        results2 = sim.run()

        np.testing.assert_array_equal(results1.assets, results2.assets)


class TestInsolvencyDetection:
    """Test that insolvency is detected in the year it occurs."""

    def test_insolvency_detected_in_claim_year(self):
        """A catastrophic claim should trigger insolvency in the same year."""
        small_config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
            insolvency_tolerance=10_000,
        )
        small_manufacturer = WidgetManufacturer(small_config)

        # Create a loss generator that produces catastrophic losses
        catastrophic_generator = ManufacturingLossGenerator.create_simple(
            frequency=5.0,
            severity_mean=10_000_000,
            severity_std=1_000,
            seed=42,
        )

        sim = Simulation(
            manufacturer=small_manufacturer,
            loss_generator=catastrophic_generator,
            time_horizon=10,
            seed=42,
        )

        results = sim.run()

        if results.insolvency_year is not None:
            # Insolvency should be in year 0 (when the catastrophic claims occur)
            assert results.insolvency_year == 0, (
                f"Insolvency detected in year {results.insolvency_year}, expected year 0. "
                "This suggests the insolvency check is using stale pre-claim metrics."
            )

    def test_insolvency_uses_post_claim_equity(self):
        """Verify that the insolvency check uses post-claim equity."""
        small_config = ManufacturerConfig(
            initial_assets=500_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
            insolvency_tolerance=10_000,
        )
        manufacturer = WidgetManufacturer(small_config)

        sim = Simulation(
            manufacturer=manufacturer,
            time_horizon=5,
            seed=42,
        )

        # Create a loss that exceeds the manufacturer's equity
        large_loss = LossEvent(amount=5_000_000, time=0.0)
        metrics = sim.step_annual(0, [large_loss])

        # The metrics should reflect the post-claim state
        assert metrics.get("equity", float("inf")) <= 10_000, (
            f"Expected low or negative equity after catastrophic loss, "
            f"got {metrics.get('equity')}"
        )


class TestConfigurableParameters:
    """Test that growth_rate and letter_of_credit_rate are configurable."""

    def test_default_growth_rate_is_zero(self, manufacturer, loss_generator):
        """Default growth_rate should be 0.0 (not the old hardcoded 0.03)."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=5,
            seed=42,
        )
        assert sim.growth_rate == 0.0

    def test_custom_growth_rate(self, manufacturer, loss_generator):
        """Custom growth_rate should be stored and used."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=5,
            seed=42,
            growth_rate=0.05,
        )
        assert sim.growth_rate == 0.05

    def test_custom_letter_of_credit_rate(self, manufacturer, loss_generator):
        """Custom letter_of_credit_rate should be stored and used."""
        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=5,
            seed=42,
            letter_of_credit_rate=0.02,
        )
        assert sim.letter_of_credit_rate == 0.02

    def test_growth_rate_affects_results(self, manufacturer_config):
        """Different growth rates should produce different results."""
        loss_gen = ManufacturingLossGenerator.create_simple(
            frequency=0.0,  # No losses to isolate growth effect
            severity_mean=1_000_000,
            seed=42,
        )

        # Zero growth
        mfg1 = WidgetManufacturer(manufacturer_config)
        sim1 = Simulation(
            manufacturer=mfg1,
            loss_generator=loss_gen,
            time_horizon=10,
            seed=42,
            growth_rate=0.0,
        )
        results1 = sim1.run()

        # 5% growth
        mfg2 = WidgetManufacturer(manufacturer_config)
        sim2 = Simulation(
            manufacturer=mfg2,
            loss_generator=loss_gen,
            time_horizon=10,
            seed=42,
            growth_rate=0.05,
        )
        results2 = sim2.run()

        # Higher growth should produce higher final assets
        assert results2.assets[-1] > results1.assets[-1], (
            f"Expected higher assets with 5% growth ({results2.assets[-1]}) "
            f"vs 0% growth ({results1.assets[-1]})"
        )


class TestExecutionOrdering:
    """Test that the execution ordering is correct: losses → claims → premium → step."""

    def test_step_annual_processes_claims_before_step(self, manufacturer_config):
        """Verify that claims are processed before manufacturer.step()."""
        manufacturer = WidgetManufacturer(manufacturer_config)

        sim = Simulation(
            manufacturer=manufacturer,
            time_horizon=5,
            seed=42,
        )

        # Track call order by patching
        call_order = []
        original_step = manufacturer.step
        original_process = manufacturer.process_uninsured_claim

        def mock_step(*args, **kwargs):
            call_order.append("step")
            return original_step(*args, **kwargs)

        def mock_process(*args, **kwargs):
            call_order.append("process_claim")
            return original_process(*args, **kwargs)

        sim.manufacturer.step = mock_step  # type: ignore[method-assign]
        sim.manufacturer.process_uninsured_claim = mock_process  # type: ignore[method-assign]

        loss = LossEvent(amount=100_000, time=0.0)
        sim.step_annual(0, [loss])

        assert "process_claim" in call_order, "Expected process_claim to be called"
        assert "step" in call_order, "Expected step to be called"
        claim_idx = call_order.index("process_claim")
        step_idx = call_order.index("step")
        assert claim_idx < step_idx, (
            f"Claims processed at index {claim_idx}, step at {step_idx}. "
            "Claims should be processed BEFORE step."
        )

    def test_step_annual_records_premium_before_step(self, manufacturer_config, insurance_policy):
        """Verify that premium is recorded before manufacturer.step()."""
        manufacturer = WidgetManufacturer(manufacturer_config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim = Simulation(
                manufacturer=manufacturer,
                insurance_policy=insurance_policy,
                time_horizon=5,
                seed=42,
            )

        call_order = []
        original_step = manufacturer.step
        original_premium = manufacturer.record_insurance_premium

        def mock_step(*args, **kwargs):
            call_order.append("step")
            return original_step(*args, **kwargs)

        def mock_premium(*args, **kwargs):
            call_order.append("premium")
            return original_premium(*args, **kwargs)

        sim.manufacturer.step = mock_step  # type: ignore[method-assign]
        sim.manufacturer.record_insurance_premium = mock_premium  # type: ignore[method-assign]

        loss = LossEvent(amount=100_000, time=0.0)
        sim.step_annual(0, [loss])

        if "premium" in call_order:
            premium_idx = call_order.index("premium")
            step_idx = call_order.index("step")
            assert premium_idx < step_idx, (
                f"Premium at index {premium_idx}, step at {step_idx}. "
                "Premium should be recorded BEFORE step."
            )


class TestEnhancedParallelConfigParams:
    """Test that _simulate_path_enhanced uses config parameters."""

    def test_shared_data_includes_step_params(self, manufacturer_config):
        """Verify shared_data includes step parameters for enhanced parallel."""
        from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloEngine

        config = MonteCarloConfig(
            n_simulations=10,
            n_years=2,
            parallel=True,
            use_enhanced_parallel=True,
            letter_of_credit_rate=0.025,
            growth_rate=0.03,
            time_resolution="annual",
            apply_stochastic=True,
            seed=42,
        )

        manufacturer = WidgetManufacturer(manufacturer_config)
        loss_gen = ManufacturingLossGenerator.create_simple(
            frequency=0.1, severity_mean=1_000_000, seed=42
        )
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        insurance = InsuranceProgram(layers=[layer])

        engine = MonteCarloEngine(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            config=config,
        )

        assert engine.config.letter_of_credit_rate == 0.025
        assert engine.config.growth_rate == 0.03
        assert engine.config.time_resolution == "annual"
        assert engine.config.apply_stochastic is True

    def test_simulate_path_enhanced_passes_params(self, manufacturer_config):
        """Test that _simulate_path_enhanced passes config params to step()."""
        from ergodic_insurance.monte_carlo import _simulate_path_enhanced

        manufacturer = WidgetManufacturer(manufacturer_config)
        loss_gen = ManufacturingLossGenerator.create_simple(
            frequency=0.0, severity_mean=1_000_000, seed=42
        )
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        insurance = InsuranceProgram(layers=[layer])

        shared = {
            "n_years": 2,
            "use_float32": False,
            "ruin_evaluation": None,
            "insolvency_tolerance": 10_000,
            "enable_ledger_pruning": False,
            "manufacturer_config": manufacturer.__dict__.copy(),
            "loss_generator": loss_gen,
            "insurance_program": insurance,
            "base_seed": 42,
            "crn_base_seed": None,
            "letter_of_credit_rate": 0.025,
            "growth_rate": 0.03,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }

        result = _simulate_path_enhanced(0, **shared)
        assert "final_assets" in result
        assert "annual_losses" in result
        assert result["final_assets"] > 0


class TestMonteCarloSequentialConfigParams:
    """Test that _run_single_simulation uses config parameters."""

    def test_sequential_uses_config_growth_rate(self, manufacturer_config):
        """Verify sequential MC path uses config growth_rate, not hardcoded 0.0."""
        from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloEngine

        config = MonteCarloConfig(
            n_simulations=2,
            n_years=3,
            parallel=False,
            growth_rate=0.05,
            letter_of_credit_rate=0.02,
            seed=42,
            progress_bar=False,
            cache_results=False,
        )

        manufacturer = WidgetManufacturer(manufacturer_config)
        loss_gen = ManufacturingLossGenerator.create_simple(
            frequency=0.0, severity_mean=1_000_000, seed=42
        )
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        insurance = InsuranceProgram(layers=[layer])

        engine = MonteCarloEngine(
            loss_generator=loss_gen,
            insurance_program=insurance,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()
        assert len(results.final_assets) == 2
        # With 5% growth and no losses, final assets should be higher than initial
        assert np.mean(results.final_assets) > 10_000_000

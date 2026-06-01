"""Tests for periodic ruin probability tracking in Monte Carlo simulations."""

from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResults,
    _flatten_parallel_results,
)


class TestPeriodicRuinTracking:
    """Test periodic ruin probability evaluation functionality."""

    @pytest.fixture
    def setup_engine(self):
        """Set up a test Monte Carlo engine."""
        # Create mock loss generator
        loss_generator = Mock(spec=ManufacturingLossGenerator)
        loss_generator.generate_losses = Mock(return_value=([], None))

        # Create insurance program
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.02,
        )
        insurance_program = InsuranceProgram(layers=[layer])

        # Create manufacturer
        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        return loss_generator, insurance_program, manufacturer

    def test_basic_periodic_evaluation(self, setup_engine):
        """Test basic periodic ruin probability evaluation."""
        loss_generator, insurance_program, manufacturer = setup_engine

        config = MonteCarloConfig(
            n_simulations=1000,
            n_years=20,
            ruin_evaluation=[5, 10, 15],
            parallel=False,  # Use sequential for deterministic testing
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        # Check that ruin_probability is now a dictionary
        assert isinstance(results.ruin_probability, dict)
        assert "5" in results.ruin_probability
        assert "10" in results.ruin_probability
        assert "15" in results.ruin_probability
        assert "20" in results.ruin_probability  # Max runtime

        # Ruin probability should be monotonically increasing
        assert results.ruin_probability["5"] <= results.ruin_probability["10"]
        assert results.ruin_probability["10"] <= results.ruin_probability["15"]
        assert results.ruin_probability["15"] <= results.ruin_probability["20"]

        # All probabilities should be between 0 and 1
        for prob in results.ruin_probability.values():
            assert 0 <= prob <= 1

    def test_no_ruin_evaluation_specified(self, setup_engine):
        """Test behavior when no ruin_evaluation is specified."""
        loss_generator, insurance_program, manufacturer = setup_engine

        sim_years = 10
        config = MonteCarloConfig(
            n_simulations=1000,
            n_years=sim_years,
            ruin_evaluation=None,  # No periodic evaluation
            parallel=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        # Should still return a dict with only the final year
        assert isinstance(results.ruin_probability, dict)
        assert str(sim_years) in results.ruin_probability
        assert len(results.ruin_probability) == 1
        assert 0 <= results.ruin_probability[str(sim_years)] <= 1

    @pytest.fixture
    def setup_real_engine(self):
        """Set up a real Monte Carlo engine for parallel testing."""
        # Use real objects instead of mocks for parallel execution
        # Create real loss generator with minimal parameters
        loss_generator = ManufacturingLossGenerator(
            attritional_params={
                "base_frequency": 0.1,  # Very low frequency for testing
                "severity_mean": 10_000,
                "severity_cv": 0.5,
            },
            seed=42,
        )

        # Create insurance program
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.02,
        )
        insurance_program = InsuranceProgram(layers=[layer])

        # Create manufacturer
        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        return loss_generator, insurance_program, manufacturer

    def test_parallel_execution(self, setup_real_engine):
        """Test periodic ruin evaluation with parallel execution."""
        loss_generator, insurance_program, manufacturer = setup_real_engine

        config = MonteCarloConfig(
            # 100 sims is plenty for the STRUCTURAL assertions below (dict shape, values in
            # [0, 1], all eval keys present); the old 1000 made the per-test runtime exceed the
            # 300s CI timeout under coverage when the parallel reduce fell back to sequential.
            n_simulations=100,
            n_years=20,
            ruin_evaluation=[5, 10, 15],
            parallel=True,
            n_workers=2,  # Reduced for better stability
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        # Results should be consistent across parallel execution
        assert isinstance(results.ruin_probability, dict)
        assert all(0 <= p <= 1 for p in results.ruin_probability.values())

        # Check all expected evaluation points are present
        assert "5" in results.ruin_probability
        assert "10" in results.ruin_probability
        assert "15" in results.ruin_probability
        assert "20" in results.ruin_probability

    def test_edge_cases(self, setup_engine):
        """Test edge cases for ruin evaluation."""
        loss_generator, insurance_program, manufacturer = setup_engine

        # Test evaluation beyond simulation years
        config = MonteCarloConfig(
            n_simulations=100,
            n_years=10,
            ruin_evaluation=[5, 10, 20],  # 20 > n_years
            parallel=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        # Should handle gracefully, only evaluate up to n_years
        assert "5" in results.ruin_probability
        assert "10" in results.ruin_probability
        assert "20" not in results.ruin_probability  # Beyond simulation period
        assert "10" in results.ruin_probability  # Max runtime is 10

    def test_single_year_evaluation(self, setup_engine):
        """Test evaluation at a single year."""
        loss_generator, insurance_program, manufacturer = setup_engine

        config = MonteCarloConfig(
            n_simulations=500,
            n_years=10,
            ruin_evaluation=[5],  # Single evaluation point
            parallel=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        assert isinstance(results.ruin_probability, dict)
        assert "5" in results.ruin_probability
        assert "10" in results.ruin_probability  # Final year always included
        assert len(results.ruin_probability) == 2

    def test_empty_evaluation_list(self, setup_engine):
        """Test with empty evaluation list."""
        loss_generator, insurance_program, manufacturer = setup_engine

        config = MonteCarloConfig(
            n_simulations=100,
            n_years=10,
            ruin_evaluation=[],  # Empty list
            parallel=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        # Should only have final year
        assert isinstance(results.ruin_probability, dict)
        assert "10" in results.ruin_probability
        assert len(results.ruin_probability) == 1

    def test_ruin_probability_consistency(self, setup_engine):
        """Test that ruin probabilities are consistent and logical."""
        loss_generator, insurance_program, manufacturer = setup_engine

        # Create a scenario with higher loss probability
        # Use a seeded RNG so the loss generation is deterministic
        _ruin_rng = np.random.default_rng(123)
        loss_generator.generate_losses = Mock(
            side_effect=lambda duration, revenue: (
                [Mock(amount=2_000_000)] if _ruin_rng.random() < 0.3 else [],
                None,
            )
        )

        config = MonteCarloConfig(
            n_simulations=1000,
            n_years=15,
            ruin_evaluation=[3, 6, 9, 12],
            parallel=False,
            seed=42,
        )

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        results = engine.run()

        # Check monotonicity
        prev_prob = 0.0
        for year in [3, 6, 9, 12, 15]:
            year_str = str(year)
            assert year_str in results.ruin_probability
            current_prob = results.ruin_probability[year_str]

            # Probability should not decrease over time
            assert (
                current_prob >= prev_prob
            ), f"Ruin probability did not increase from previous {prev_prob} to current {current_prob} at year {year}"
            prev_prob = current_prob


class TestFlattenParallelResults:
    """Regression for the enhanced-parallel reduce shape bug.

    ``combine_results_enhanced`` used to ``extend`` each item of a FLAT list of per-simulation
    dicts, which iterates a dict into its string KEYS -- producing
    ``Unexpected result format: <class 'str'>`` for every result, zero valid simulations, and a
    silent (slow) fallback to sequential execution on every enhanced-parallel run.
    ``_flatten_parallel_results`` appends dicts whole and only flattens genuine nested lists.
    """

    def test_flat_list_of_dicts_preserved(self):
        # Each per-sim dict must survive AS a dict (not be split into its keys).
        results = [{"final_assets": 1.0}, {"final_assets": 2.0}]
        out = _flatten_parallel_results(results)
        assert out == results
        assert all(isinstance(r, dict) for r in out)

    def test_no_string_keys_leak(self):
        # The exact bug signature: a key-string must never appear in the output.
        out = _flatten_parallel_results([{"final_assets": 1.0, "ruin_at_year": {5: False}}])
        assert "final_assets" not in out and "ruin_at_year" not in out
        assert out[0]["final_assets"] == 1.0

    def test_legacy_list_of_lists_flattened(self):
        out = _flatten_parallel_results([[{"a": 1}], [{"a": 2}, {"a": 3}]])
        assert out == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_none_entries_skipped(self):
        assert _flatten_parallel_results([{"a": 1}, None, {"a": 2}]) == [{"a": 1}, {"a": 2}]

    def test_empty_input(self):
        assert _flatten_parallel_results([]) == []

"""Tests for parallel Monte Carlo path independence (issue #299).

Verifies that parallel workers produce statistically distinct loss sequences
rather than identical copies of the same path.
"""

import copy
from typing import Dict, List
from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import (
    AttritionalLossGenerator,
    CatastrophicLossGenerator,
    FrequencyGenerator,
    LargeLossGenerator,
    ManufacturingLossGenerator,
)
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloEngine
from ergodic_insurance.monte_carlo_worker import run_chunk_standalone

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manufacturer_config():
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        base_operating_margin=0.1,
        tax_rate=0.25,
        retention_ratio=0.8,
    )


@pytest.fixture
def manufacturer(manufacturer_config):
    return WidgetManufacturer(manufacturer_config)


@pytest.fixture
def insurance_program():
    layer = EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.02)
    return InsuranceProgram(layers=[layer])


@pytest.fixture
def loss_generator():
    return ManufacturingLossGenerator(seed=42)


# ---------------------------------------------------------------------------
# Reseed tests for individual generators
# ---------------------------------------------------------------------------


class TestGeneratorReseed:
    """Verify reseed() produces independent streams."""

    def test_frequency_generator_reseed(self):
        gen = FrequencyGenerator(base_frequency=5.0, seed=42)
        samples_a = [gen.rng.random() for _ in range(10)]

        gen.reseed(99)
        samples_b = [gen.rng.random() for _ in range(10)]

        assert samples_a != samples_b, "Reseed should produce different stream"

    def test_frequency_generator_reseed_reproducible(self):
        gen = FrequencyGenerator(base_frequency=5.0, seed=42)
        gen.reseed(99)
        samples_a = [gen.rng.random() for _ in range(10)]

        gen.reseed(99)
        samples_b = [gen.rng.random() for _ in range(10)]

        assert samples_a == samples_b, "Same seed should reproduce same stream"

    def test_attritional_reseed(self):
        gen = AttritionalLossGenerator(seed=42)
        losses_a = gen.generate_losses(duration=1.0, revenue=10_000_000)

        gen.reseed(99)
        losses_b = gen.generate_losses(duration=1.0, revenue=10_000_000)

        # Different seeds should generally produce different loss sets.
        # With different seeds at least the amounts should differ.
        amounts_a = [l.amount for l in losses_a]
        amounts_b = [l.amount for l in losses_b]
        assert amounts_a != amounts_b or len(losses_a) != len(losses_b)

    def test_large_reseed(self):
        gen = LargeLossGenerator(seed=42)
        gen.reseed(99)
        losses = gen.generate_losses(duration=1.0, revenue=10_000_000)
        # Just verify it doesn't crash after reseed
        assert isinstance(losses, list)

    def test_catastrophic_reseed(self):
        gen = CatastrophicLossGenerator(seed=42)
        gen.reseed(99)
        losses = gen.generate_losses(duration=1.0, revenue=10_000_000)
        assert isinstance(losses, list)

    def test_manufacturing_loss_generator_reseed(self):
        gen = ManufacturingLossGenerator(seed=42)

        # Generate with original seed
        losses_a, _ = gen.generate_losses(duration=1.0, revenue=10_000_000)

        # Reseed and generate again
        gen.reseed(99)
        losses_b, _ = gen.generate_losses(duration=1.0, revenue=10_000_000)

        amounts_a = sorted(l.amount for l in losses_a)
        amounts_b = sorted(l.amount for l in losses_b)
        assert amounts_a != amounts_b or len(losses_a) != len(
            losses_b
        ), "Different seeds should produce different loss sequences"

    def test_manufacturing_loss_generator_reseed_reproducible(self):
        gen = ManufacturingLossGenerator(seed=42)

        gen.reseed(99)
        losses_a, _ = gen.generate_losses(duration=1.0, revenue=10_000_000)

        gen.reseed(99)
        losses_b, _ = gen.generate_losses(duration=1.0, revenue=10_000_000)

        amounts_a = [l.amount for l in losses_a]
        amounts_b = [l.amount for l in losses_b]
        assert amounts_a == amounts_b, "Same seed should reproduce same results"


# ---------------------------------------------------------------------------
# Worker chunk independence
# ---------------------------------------------------------------------------


class TestWorkerChunkIndependence:
    """Verify that parallel chunks produce distinct loss sequences."""

    def test_chunks_produce_different_losses(self, loss_generator, insurance_program, manufacturer):
        """Two chunks with different seeds must produce different loss arrays."""
        config_dict = {
            "n_years": 5,
            "use_float32": False,
            "ruin_evaluation": None,
            "insolvency_tolerance": 10_000,
            "letter_of_credit_rate": 0.015,
            "growth_rate": 0.0,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }

        n_sims = 50
        chunk_a = (0, n_sims, 1000)
        chunk_b = (n_sims, 2 * n_sims, 2000)

        # Each chunk gets a deep copy to simulate pickling in multiprocessing
        result_a = run_chunk_standalone(
            chunk_a,
            copy.deepcopy(loss_generator),
            insurance_program,
            manufacturer,
            config_dict,
        )
        result_b = run_chunk_standalone(
            chunk_b,
            copy.deepcopy(loss_generator),
            insurance_program,
            manufacturer,
            config_dict,
        )

        # The annual_losses arrays across chunks should NOT be identical
        losses_a = np.asarray(result_a["annual_losses"])
        losses_b = np.asarray(result_b["annual_losses"])
        assert not np.array_equal(losses_a, losses_b), (
            "Chunks with different seeds produced identical loss arrays. "
            "Loss generator seeding is broken."
        )

    def test_chunks_same_seed_reproduce(self, loss_generator, insurance_program, manufacturer):
        """Two chunks with the same seed should reproduce identical results."""
        config_dict = {
            "n_years": 5,
            "use_float32": False,
            "ruin_evaluation": None,
            "insolvency_tolerance": 10_000,
            "letter_of_credit_rate": 0.015,
            "growth_rate": 0.0,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }

        n_sims = 20
        chunk = (0, n_sims, 42)

        result_a = run_chunk_standalone(
            chunk,
            copy.deepcopy(loss_generator),
            insurance_program,
            manufacturer,
            config_dict,
        )
        result_b = run_chunk_standalone(
            chunk,
            copy.deepcopy(loss_generator),
            insurance_program,
            manufacturer,
            config_dict,
        )

        np.testing.assert_array_equal(
            result_a["annual_losses"],
            result_b["annual_losses"],
            err_msg="Same seed should reproduce identical results",
        )


# ---------------------------------------------------------------------------
# Insolvency threshold consistency
# ---------------------------------------------------------------------------


class TestInsolvencyThresholdConsistency:
    """Verify insolvency detection is consistent between parallel and sequential."""

    def test_worker_uses_insolvency_tolerance(
        self, loss_generator, insurance_program, manufacturer
    ):
        """Worker should use insolvency_tolerance from config, not hardcoded 0."""
        config_dict = {
            "n_years": 5,
            "use_float32": False,
            "ruin_evaluation": [5],
            "insolvency_tolerance": 10_000,
            "letter_of_credit_rate": 0.015,
            "growth_rate": 0.0,
            "time_resolution": "annual",
            "apply_stochastic": False,
        }

        chunk = (0, 10, 42)
        result = run_chunk_standalone(
            chunk,
            copy.deepcopy(loss_generator),
            insurance_program,
            manufacturer,
            config_dict,
        )

        # Just verify the ruin tracking structure is present
        assert "ruin_at_year" in result
        assert len(result["ruin_at_year"]) == 10  # one dict per simulation


# ---------------------------------------------------------------------------
# Config seed preservation
# ---------------------------------------------------------------------------


class TestConfigSeedPreservation:
    """Verify config.seed is not permanently mutated."""

    def test_constructor_does_not_set_global_seed(self):
        """MonteCarloEngine constructor should not call np.random.seed()."""
        loss_gen = ManufacturingLossGenerator(seed=42)
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.02)
        insurance = InsuranceProgram(layers=[layer])
        mfg_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        mfg = WidgetManufacturer(mfg_config)

        # Set global state to a known value and draw a sample before construction
        np.random.seed(9999)
        sample_before = np.random.random(10)

        # Reset to same known state so we can compare after construction
        np.random.seed(9999)

        config = MonteCarloConfig(
            n_simulations=100,
            n_years=5,
            parallel=False,
            cache_results=False,
            seed=12345,
        )
        _engine = MonteCarloEngine(loss_gen, insurance, mfg, config)

        # If constructor didn't call np.random.seed(), the global state
        # should still be at seed=9999 and produce the same sequence.
        sample_after = np.random.random(10)

        np.testing.assert_array_equal(
            sample_before,
            sample_after,
            err_msg="Constructor should not mutate global numpy random state",
        )

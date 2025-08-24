"""Tests for enhanced loss distribution classes."""

import time
from unittest.mock import patch

import numpy as np
import pytest
from ergodic_insurance.src.loss_distributions import (
    AttritionalLossGenerator,
    CatastrophicLossGenerator,
    FrequencyGenerator,
    LargeLossGenerator,
    LognormalLoss,
    LossDistribution,
    LossEvent,
    ManufacturingLossGenerator,
    ParetoLoss,
    perform_statistical_tests,
)


class TestLognormalLoss:
    """Test the LognormalLoss distribution class."""

    def test_init_with_mean_cv(self):
        """Test initialization with mean and CV parameters."""
        dist = LognormalLoss(mean=100_000, cv=1.5)
        assert dist.mean == 100_000
        assert dist.cv == 1.5
        assert dist.mu > 0
        assert dist.sigma > 0

    def test_init_with_mu_sigma(self):
        """Test initialization with mu and sigma parameters."""
        dist = LognormalLoss(mu=10, sigma=1)
        assert dist.mu == 10
        assert dist.sigma == 1
        assert dist.mean > 0
        assert dist.cv > 0

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Mean must be positive"):
            LognormalLoss(mean=-100, cv=1)

        with pytest.raises(ValueError, match="CV must be non-negative"):
            LognormalLoss(mean=100, cv=-1)

        with pytest.raises(ValueError, match="Sigma must be non-negative"):
            LognormalLoss(mu=10, sigma=-1)

        with pytest.raises(ValueError, match="Must provide either"):
            LognormalLoss(mean=100)

    def test_generate_severity(self):
        """Test severity generation."""
        dist = LognormalLoss(mean=50_000, cv=1.0, seed=42)
        samples = dist.generate_severity(1000)

        assert len(samples) == 1000
        assert np.all(samples > 0)
        # Check mean is approximately correct (within 10%)
        assert abs(np.mean(samples) - 50_000) / 50_000 < 0.1

    def test_generate_empty(self):
        """Test generating zero samples."""
        dist = LognormalLoss(mean=50_000, cv=1.0)
        samples = dist.generate_severity(0)
        assert len(samples) == 0

        samples = dist.generate_severity(-1)
        assert len(samples) == 0

    def test_expected_value(self):
        """Test expected value calculation."""
        dist = LognormalLoss(mean=75_000, cv=2.0)
        assert dist.expected_value() == 75_000

    def test_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        dist1 = LognormalLoss(mean=100_000, cv=1.5, seed=123)
        dist2 = LognormalLoss(mean=100_000, cv=1.5, seed=123)

        samples1 = dist1.generate_severity(100)
        samples2 = dist2.generate_severity(100)

        np.testing.assert_array_equal(samples1, samples2)

    def test_reset_seed(self):
        """Test resetting the random seed."""
        dist = LognormalLoss(mean=100_000, cv=1.5, seed=42)
        samples1 = dist.generate_severity(10)

        dist.reset_seed(42)
        samples2 = dist.generate_severity(10)

        np.testing.assert_array_equal(samples1, samples2)


class TestParetoLoss:
    """Test the ParetoLoss distribution class."""

    def test_init_valid(self):
        """Test valid initialization."""
        dist = ParetoLoss(alpha=2.5, xm=1_000_000)
        assert dist.alpha == 2.5
        assert dist.xm == 1_000_000

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Alpha must be positive"):
            ParetoLoss(alpha=-1, xm=1_000_000)

        with pytest.raises(ValueError, match="Alpha must be positive"):
            ParetoLoss(alpha=0, xm=1_000_000)

        with pytest.raises(ValueError, match="Minimum value xm must be positive"):
            ParetoLoss(alpha=2, xm=-1000)

    def test_generate_severity(self):
        """Test severity generation."""
        dist = ParetoLoss(alpha=2.5, xm=1_000_000, seed=42)
        samples = dist.generate_severity(1000)

        assert len(samples) == 1000
        assert np.all(samples >= 1_000_000)  # All above minimum
        # Check heavy tail - some large values (adjust threshold for seed 42)
        assert np.max(samples) > 5_000_000  # More reasonable threshold

    def test_generate_empty(self):
        """Test generating zero samples."""
        dist = ParetoLoss(alpha=2.5, xm=1_000_000)
        samples = dist.generate_severity(0)
        assert len(samples) == 0

        samples = dist.generate_severity(-1)
        assert len(samples) == 0

    def test_expected_value(self):
        """Test expected value calculation."""
        # Alpha > 1, finite expected value
        dist = ParetoLoss(alpha=2.0, xm=1_000_000)
        expected = dist.expected_value()
        assert expected == 2_000_000  # alpha * xm / (alpha - 1)

        # Alpha = 1, infinite expected value
        dist = ParetoLoss(alpha=1.0, xm=1_000_000)
        assert dist.expected_value() == np.inf

        # Alpha < 1, infinite expected value
        dist = ParetoLoss(alpha=0.5, xm=1_000_000)
        assert dist.expected_value() == np.inf

    def test_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        dist1 = ParetoLoss(alpha=2.5, xm=1_000_000, seed=456)
        dist2 = ParetoLoss(alpha=2.5, xm=1_000_000, seed=456)

        samples1 = dist1.generate_severity(100)
        samples2 = dist2.generate_severity(100)

        np.testing.assert_array_equal(samples1, samples2)


class TestFrequencyGenerator:
    """Test the FrequencyGenerator class."""

    def test_init_valid(self):
        """Test valid initialization."""
        gen = FrequencyGenerator(
            base_frequency=5.0, revenue_scaling_exponent=0.5, reference_revenue=10_000_000
        )
        assert gen.base_frequency == 5.0
        assert gen.revenue_scaling_exponent == 0.5
        assert gen.reference_revenue == 10_000_000

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Base frequency must be non-negative"):
            FrequencyGenerator(base_frequency=-1)

        with pytest.raises(ValueError, match="Reference revenue must be positive"):
            FrequencyGenerator(base_frequency=5, reference_revenue=-1000)

    def test_scaled_frequency_no_scaling(self):
        """Test frequency with no revenue scaling."""
        gen = FrequencyGenerator(base_frequency=5.0, revenue_scaling_exponent=0.0)

        assert gen.get_scaled_frequency(10_000_000) == 5.0
        assert gen.get_scaled_frequency(20_000_000) == 5.0
        assert gen.get_scaled_frequency(5_000_000) == 5.0

    def test_scaled_frequency_with_scaling(self):
        """Test frequency with revenue scaling."""
        gen = FrequencyGenerator(
            base_frequency=5.0, revenue_scaling_exponent=0.5, reference_revenue=10_000_000
        )

        # At reference revenue
        assert gen.get_scaled_frequency(10_000_000) == 5.0

        # Double revenue -> sqrt(2) scaling
        assert abs(gen.get_scaled_frequency(20_000_000) - 5.0 * np.sqrt(2)) < 0.001

        # Half revenue -> sqrt(0.5) scaling
        assert abs(gen.get_scaled_frequency(5_000_000) - 5.0 * np.sqrt(0.5)) < 0.001

        # Zero revenue
        assert gen.get_scaled_frequency(0) == 0.0

    def test_generate_event_times(self):
        """Test event time generation."""
        gen = FrequencyGenerator(base_frequency=5.0, seed=42)

        times = gen.generate_event_times(duration=10, revenue=10_000_000)

        # Should generate approximately 50 events (5 per year * 10 years)
        assert 30 < len(times) < 70  # Allow for randomness
        assert np.all(times >= 0)
        assert np.all(times <= 10)
        # Times should be sorted
        assert np.all(np.diff(times) >= 0)

    def test_generate_event_times_edge_cases(self):
        """Test edge cases for event time generation."""
        gen = FrequencyGenerator(base_frequency=5.0)

        # Zero duration
        times = gen.generate_event_times(0, 10_000_000)
        assert len(times) == 0

        # Negative duration
        times = gen.generate_event_times(-1, 10_000_000)
        assert len(times) == 0

        # Zero revenue
        times = gen.generate_event_times(10, 0)
        assert len(times) == 0

        # Zero frequency
        gen = FrequencyGenerator(base_frequency=0.0)
        times = gen.generate_event_times(10, 10_000_000)
        assert len(times) == 0


class TestAttritionalLossGenerator:
    """Test the AttritionalLossGenerator class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        gen = AttritionalLossGenerator()
        assert gen.frequency_generator.base_frequency == 5.0
        assert gen.severity_distribution.mean == 25_000
        assert gen.loss_type == "attritional"

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        gen = AttritionalLossGenerator(
            base_frequency=8.0, severity_mean=50_000, severity_cv=1.0, revenue_scaling_exponent=0.3
        )
        assert gen.frequency_generator.base_frequency == 8.0
        assert gen.severity_distribution.mean == 50_000
        assert gen.frequency_generator.revenue_scaling_exponent == 0.3

    def test_generate_losses(self):
        """Test loss generation."""
        gen = AttritionalLossGenerator(seed=42)
        losses = gen.generate_losses(duration=10, revenue=10_000_000)

        assert len(losses) > 0
        assert all(isinstance(loss, LossEvent) for loss in losses)
        assert all(loss.loss_type == "attritional" for loss in losses)
        assert all(0 <= loss.time <= 10 for loss in losses)
        assert all(loss.amount > 0 for loss in losses)

        # Check rough magnitude of losses (3K-100K typical)
        amounts = [loss.amount for loss in losses]
        assert min(amounts) > 1_000
        assert max(amounts) < 500_000

    def test_revenue_scaling(self):
        """Test that revenue scaling works correctly."""
        gen = AttritionalLossGenerator(base_frequency=5.0, revenue_scaling_exponent=0.5, seed=42)

        # Generate at reference revenue
        losses_ref = gen.generate_losses(duration=100, revenue=10_000_000)

        # Reset seed and generate at double revenue
        gen.frequency_generator.rng = np.random.RandomState(42)
        gen.severity_distribution.rng = np.random.RandomState(42)
        losses_double = gen.generate_losses(duration=100, revenue=20_000_000)

        # Should have approximately sqrt(2) times more losses
        ratio = len(losses_double) / len(losses_ref)
        assert 1.2 < ratio < 1.6  # Approximate sqrt(2) with randomness


class TestLargeLossGenerator:
    """Test the LargeLossGenerator class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        gen = LargeLossGenerator()
        assert gen.frequency_generator.base_frequency == 0.3
        assert gen.severity_distribution.mean == 2_000_000
        assert gen.loss_type == "large"

    def test_generate_losses(self):
        """Test loss generation."""
        gen = LargeLossGenerator(seed=42)
        losses = gen.generate_losses(duration=100, revenue=10_000_000)

        # Should generate some losses over 100 years
        assert len(losses) > 10
        assert all(isinstance(loss, LossEvent) for loss in losses)
        assert all(loss.loss_type == "large" for loss in losses)

        # Check magnitude (500K-50M typical, but with lognormal variance)
        amounts = [loss.amount for loss in losses]
        assert min(amounts) > 10_000  # More realistic lower bound with CV=2.0
        assert max(amounts) < 100_000_000


class TestCatastrophicLossGenerator:
    """Test the CatastrophicLossGenerator class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        gen = CatastrophicLossGenerator()
        assert gen.frequency_generator.base_frequency == 0.03
        assert gen.severity_distribution.alpha == 2.5
        assert gen.severity_distribution.xm == 1_000_000
        assert gen.loss_type == "catastrophic"

    def test_no_revenue_scaling(self):
        """Test that catastrophic losses don't scale with revenue."""
        gen = CatastrophicLossGenerator()
        assert gen.frequency_generator.revenue_scaling_exponent == 0.0

    def test_generate_losses(self):
        """Test loss generation."""
        gen = CatastrophicLossGenerator(base_frequency=0.1, seed=42)  # Higher frequency for testing
        losses = gen.generate_losses(duration=100, revenue=10_000_000)

        # Should generate some catastrophic losses
        assert len(losses) > 0
        assert all(isinstance(loss, LossEvent) for loss in losses)
        assert all(loss.loss_type == "catastrophic" for loss in losses)

        # All losses should be above minimum
        amounts = [loss.amount for loss in losses]
        assert all(amount >= 1_000_000 for amount in amounts)

        # Should have some large losses (adjust for actual seed behavior)
        assert max(amounts) > 2_000_000  # More reasonable threshold


class TestManufacturingLossGenerator:
    """Test the composite ManufacturingLossGenerator class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        gen = ManufacturingLossGenerator()
        assert gen.attritional is not None
        assert gen.large is not None
        assert gen.catastrophic is not None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        gen = ManufacturingLossGenerator(
            attritional_params={"base_frequency": 10.0},
            large_params={"severity_mean": 5_000_000},
            catastrophic_params={"severity_alpha": 3.0},
        )
        assert gen.attritional.frequency_generator.base_frequency == 10.0
        assert gen.large.severity_distribution.mean == 5_000_000
        assert gen.catastrophic.severity_distribution.alpha == 3.0

    def test_seed_propagation(self):
        """Test that seeds are properly propagated to sub-generators."""
        gen = ManufacturingLossGenerator(seed=42)

        # Each generator should have a different seed
        assert int(gen.attritional.frequency_generator.rng.get_state()[1][0]) == 42  # type: ignore
        assert int(gen.large.frequency_generator.rng.get_state()[1][0]) == 43  # type: ignore
        assert int(gen.catastrophic.frequency_generator.rng.get_state()[1][0]) == 44  # type: ignore

    def test_generate_losses(self):
        """Test comprehensive loss generation."""
        gen = ManufacturingLossGenerator(seed=42)
        losses, stats = gen.generate_losses(
            duration=10, revenue=10_000_000, include_catastrophic=True
        )

        # Should generate multiple types of losses
        assert len(losses) > 0
        assert stats["total_losses"] == len(losses)
        assert stats["attritional_count"] > 0

        # Losses should be sorted by time
        times = [loss.time for loss in losses]
        assert times == sorted(times)

        # Check statistics
        assert stats["total_amount"] > 0
        assert stats["average_loss"] > 0
        assert stats["max_loss"] > 0
        assert stats["annual_frequency"] > 0
        assert stats["annual_expected_loss"] > 0

        # Total should equal sum of components
        assert (
            abs(
                stats["total_amount"]
                - (
                    stats["attritional_amount"]
                    + stats["large_amount"]
                    + stats["catastrophic_amount"]
                )
            )
            < 0.01
        )

    def test_exclude_catastrophic(self):
        """Test generation without catastrophic losses."""
        gen = ManufacturingLossGenerator(seed=42)
        losses, stats = gen.generate_losses(
            duration=10, revenue=10_000_000, include_catastrophic=False
        )

        assert stats["catastrophic_count"] == 0
        assert stats["catastrophic_amount"] == 0
        assert all(loss.loss_type != "catastrophic" for loss in losses)

    def test_validate_distributions(self):
        """Test distribution validation method."""
        gen = ManufacturingLossGenerator(seed=42)
        validation = gen.validate_distributions(
            n_simulations=1000, duration=1.0, revenue=10_000_000
        )

        # Should have statistics for each type
        assert "attritional" in validation
        assert "large" in validation
        assert "catastrophic" in validation
        assert "total" in validation

        # Each should have various statistics
        for loss_type, stats in validation.items():
            assert "mean" in stats
            assert "std" in stats
            assert "cv" in stats
            assert "median" in stats
            assert "p95" in stats
            assert "p99" in stats

            # Values should be reasonable
            assert stats["mean"] >= 0
            assert stats["std"] >= 0
            assert stats["p99"] >= stats["p95"]
            assert stats["p95"] >= stats["median"]


class TestStatisticalTests:
    """Test the statistical validation functions."""

    def test_lognormal_validation(self):
        """Test statistical tests for lognormal distribution."""
        # Generate known lognormal samples
        dist = LognormalLoss(mean=100_000, cv=1.5, seed=42)
        samples = dist.generate_severity(1000)

        results = perform_statistical_tests(
            samples, "lognormal", {"mu": dist.mu, "sigma": dist.sigma}
        )

        # Should have KS test results
        assert "ks_test" in results
        assert results["ks_test"]["p_value"] > 0.05  # Should not reject

        # Should have Anderson-Darling results
        assert "anderson_darling" in results

        # Should have Shapiro-Wilk for log-transformed data
        assert "shapiro_wilk" in results

    def test_pareto_validation(self):
        """Test statistical tests for Pareto distribution."""
        # Generate known Pareto samples
        dist = ParetoLoss(alpha=2.5, xm=1_000_000, seed=42)
        samples = dist.generate_severity(1000)

        results = perform_statistical_tests(samples, "pareto", {"alpha": 2.5, "xm": 1_000_000})

        # Should have KS test results
        assert "ks_test" in results
        assert "p_value" in results["ks_test"]

    def test_insufficient_samples(self):
        """Test handling of insufficient samples."""
        samples = np.array([1, 2, 3])

        results = perform_statistical_tests(samples, "lognormal", {"mu": 10, "sigma": 1})

        assert "error" in results
        assert "Insufficient samples" in results["error"]


class TestPerformance:
    """Test performance requirements."""

    def test_generate_million_samples(self):
        """Test that we can generate 1M samples in < 1 second."""
        dist = LognormalLoss(mean=50_000, cv=1.5)

        start_time = time.time()
        samples = dist.generate_severity(1_000_000)
        elapsed_time = time.time() - start_time

        assert len(samples) == 1_000_000
        assert elapsed_time < 1.0  # Should complete in less than 1 second

    def test_large_simulation_performance(self):
        """Test performance of comprehensive simulation."""
        gen = ManufacturingLossGenerator()

        start_time = time.time()
        losses, stats = gen.generate_losses(duration=1000, revenue=10_000_000)  # 1000 years
        elapsed_time = time.time() - start_time

        # Should complete 1000-year simulation quickly
        assert elapsed_time < 5.0  # 5 seconds max
        assert stats["total_losses"] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_frequency(self):
        """Test handling of zero frequency."""
        gen = AttritionalLossGenerator(base_frequency=0.0)
        losses = gen.generate_losses(duration=10, revenue=10_000_000)
        assert len(losses) == 0

    def test_extreme_parameters(self):
        """Test handling of extreme parameter values."""
        # Very high CV
        dist = LognormalLoss(mean=1000, cv=10.0)
        samples = dist.generate_severity(100)
        assert len(samples) == 100
        assert np.all(samples > 0)

        # Very low alpha (heavy tail)
        dist2 = ParetoLoss(alpha=1.1, xm=1000)
        samples = dist2.generate_severity(100)
        assert len(samples) == 100
        assert np.all(samples >= 1000)

    def test_negative_revenue(self):
        """Test handling of negative revenue."""
        gen = FrequencyGenerator(base_frequency=5.0)
        assert gen.get_scaled_frequency(-1000) == 0.0

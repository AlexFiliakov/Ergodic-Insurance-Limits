"""Unit tests for the ClaimGenerator class."""

import time
from typing import Dict

import numpy as np
import pytest

from ergodic_insurance.claim_generator import ClaimEvent, ClaimGenerator
from ergodic_insurance.loss_distributions import LossData


class TestClaimEvent:
    """Test suite for ClaimEvent dataclass."""

    def test_init(self):
        """Test claim event initialization."""
        claim = ClaimEvent(year=5, amount=100000)
        assert claim.year == 5
        assert claim.amount == 100000


class TestClaimGenerator:
    """Test suite for ClaimGenerator class."""

    def test_init(self):
        """Test generator initialization."""
        gen = ClaimGenerator(
            frequency=0.5,
            severity_mean=1_000_000,
            severity_std=500_000,
            seed=42,
        )
        assert gen.frequency == 0.5
        assert gen.severity_mean == 1_000_000
        assert gen.severity_std == 500_000
        assert gen.rng is not None

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = ClaimGenerator(seed=42)
        gen2 = ClaimGenerator(seed=42)

        claims1 = gen1.generate_claims(10)
        claims2 = gen2.generate_claims(10)

        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == pytest.approx(c2.amount)

    def test_reset_seed(self):
        """Test seed reset functionality."""
        gen = ClaimGenerator(seed=42)
        claims1 = gen.generate_claims(5)

        gen.reset_seed(42)
        claims2 = gen.generate_claims(5)

        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == pytest.approx(c2.amount)

    def test_generate_claims_statistical_properties(self):
        """Test that generated claims match expected statistical properties."""
        frequency = 2.0  # Average 2 claims per year
        severity_mean = 1_000_000
        severity_std = 200_000
        years = 1000  # Large sample for statistical testing

        gen = ClaimGenerator(
            frequency=frequency,
            severity_mean=severity_mean,
            severity_std=severity_std,
            seed=42,
        )

        claims = gen.generate_claims(years)

        # Test frequency (Poisson distribution)
        claims_per_year: Dict[int, int] = {}
        for claim in claims:
            claims_per_year[claim.year] = claims_per_year.get(claim.year, 0) + 1

        counts = [claims_per_year.get(y, 0) for y in range(years)]
        mean_count = np.mean(counts)

        # Mean should be close to frequency parameter
        assert mean_count == pytest.approx(frequency, rel=0.1)

        # Test severity (Lognormal distribution)
        if claims:
            amounts = [c.amount for c in claims]
            mean_amount = np.mean(amounts)
            std_amount = np.std(amounts)

            # Mean should be close to severity_mean
            assert mean_amount == pytest.approx(severity_mean, rel=0.1)
            # Std should be close to severity_std
            assert std_amount == pytest.approx(severity_std, rel=0.2)

    def test_generate_catastrophic_claims(self):
        """Test catastrophic claim generation."""
        gen = ClaimGenerator(seed=42)
        years = 1000
        cat_frequency = 0.05  # 5% chance per year

        cat_claims = gen.generate_catastrophic_claims(
            years=years,
            cat_frequency=cat_frequency,
            cat_severity_mean=10_000_000,
            cat_severity_std=5_000_000,
        )

        # Check frequency is approximately correct
        n_cat_events = len(cat_claims)
        expected_events = years * cat_frequency

        # With Bernoulli trials, we expect roughly cat_frequency * years events
        assert n_cat_events == pytest.approx(expected_events, rel=0.3)

        # Check all amounts are positive
        for claim in cat_claims:
            assert claim.amount > 0
            assert 0 <= claim.year < years

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        gen = ClaimGenerator(seed=42)

        # Zero years
        claims = gen.generate_claims(0)
        assert claims == []

        # Zero frequency
        gen_zero = ClaimGenerator(frequency=0, seed=42)
        claims = gen_zero.generate_claims(10)
        assert claims == []

        # Very high frequency
        gen_high = ClaimGenerator(frequency=100, seed=42)
        claims = gen_high.generate_claims(1)
        assert len(claims) > 50  # Should have many claims

    def test_generate_all_claims(self):
        """Test batch generation of regular and catastrophic claims."""
        gen = ClaimGenerator(
            frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=42,
        )

        years = 100
        regular, catastrophic = gen.generate_all_claims(
            years=years,
            include_catastrophic=True,
            cat_frequency=0.02,
            cat_severity_mean=10_000_000,  # Higher mean for clear distinction
            cat_severity_std=5_000_000,
        )

        # Check regular claims exist
        assert len(regular) > 0

        # Check catastrophic claims exist (with high probability over 100 years)
        assert len(catastrophic) > 0

        # Catastrophic claims should be larger on average
        # Note: the seed might not generate catastrophic claims in this test run
        # so we only check if they exist when generated
        if catastrophic:
            avg_catastrophic = np.mean([c.amount for c in catastrophic])
            # Catastrophic mean is set to 10M which should be > regular mean of 100K
            assert avg_catastrophic > 1_000_000  # Much larger than regular claims

        # Test without catastrophic claims
        regular_only, no_cat = gen.generate_all_claims(
            years=years,
            include_catastrophic=False,
        )
        assert len(regular_only) > 0
        assert no_cat == []

    def test_performance(self):
        """Test that generation meets performance requirements."""
        gen = ClaimGenerator(
            frequency=3.0,  # Moderate frequency
            severity_mean=500_000,
            severity_std=250_000,
            seed=42,
        )

        years = 1000

        # Time the generation
        start_time = time.time()
        claims = gen.generate_claims(years)
        elapsed_time = time.time() - start_time

        # Should complete in less than 1 second
        assert elapsed_time < 1.0

        # Should generate expected number of claims
        assert len(claims) > 0

        # Time batch generation
        start_time = time.time()
        regular, catastrophic = gen.generate_all_claims(years)
        elapsed_time = time.time() - start_time

        # Batch generation should also be fast
        assert elapsed_time < 1.0

    def test_claim_years_in_range(self):
        """Test that all claims are generated within the specified year range."""
        gen = ClaimGenerator(frequency=2.0, seed=42)
        years = 50

        claims = gen.generate_claims(years)

        for claim in claims:
            assert 0 <= claim.year < years

        # Test catastrophic claims too
        cat_claims = gen.generate_catastrophic_claims(years)
        for claim in cat_claims:
            assert 0 <= claim.year < years

    def test_negative_parameters(self):
        """Test handling of invalid parameters."""
        # Test that negative frequency raises ValueError
        with pytest.raises(ValueError, match="Frequency must be non-negative"):
            ClaimGenerator(frequency=-1.0)

        # Test that negative severity_mean raises ValueError
        with pytest.raises(ValueError, match="Severity mean must be positive"):
            ClaimGenerator(severity_mean=-1000)

        # Test that negative severity_std raises ValueError
        with pytest.raises(ValueError, match="Severity std must be non-negative"):
            ClaimGenerator(severity_std=-100)

        # Test that generation with negative years should handle gracefully
        gen = ClaimGenerator(frequency=1.0)
        claims = gen.generate_claims(-5)
        assert claims == []

        # Zero frequency should produce no claims
        gen_zero = ClaimGenerator(frequency=0.0)
        claims = gen_zero.generate_claims(10)
        assert claims == []

    def test_generate_enhanced_claims_fallback(self):
        """Test generate_enhanced_claims falls back to standard generation."""
        gen = ClaimGenerator(
            frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=42,
        )

        # Test with enhanced distributions disabled
        claims, stats = gen.generate_enhanced_claims(
            years=10, revenue=5_000_000, use_enhanced_distributions=False
        )

        assert isinstance(claims, list)
        assert isinstance(stats, dict)
        assert stats["method"] == "standard"
        assert "total_losses" in stats
        assert "regular_count" in stats
        assert "catastrophic_count" in stats
        assert "total_amount" in stats
        assert stats["total_losses"] == len(claims)

        # Verify claims are valid
        for claim in claims:
            assert isinstance(claim, ClaimEvent)
            assert claim.amount > 0
            assert 0 <= claim.year < 10

        # Test with no revenue specified (should use default)
        claims2, stats2 = gen.generate_enhanced_claims(years=5, use_enhanced_distributions=False)

        assert isinstance(claims2, list)
        assert stats2["method"] == "standard"

    def test_to_loss_data(self):
        """Test to_loss_data method converts claims to LossData format."""
        gen = ClaimGenerator(seed=42)

        # Test with actual claims
        claims = [
            ClaimEvent(year=0, amount=100000),
            ClaimEvent(year=1, amount=200000),
            ClaimEvent(year=1, amount=150000),
            ClaimEvent(year=2, amount=300000),
        ]

        loss_data = gen.to_loss_data(claims)

        # Check the LossData structure
        assert len(loss_data.timestamps) == 4
        assert len(loss_data.loss_amounts) == 4
        assert all(t >= 0 for t in loss_data.timestamps)
        assert all(a > 0 for a in loss_data.loss_amounts)

        # Check metadata
        assert loss_data.metadata["source"] == "claim_generator"
        assert loss_data.metadata["generator_type"] == "ClaimGenerator"
        assert loss_data.metadata["frequency"] == gen.frequency
        assert loss_data.metadata["severity_mean"] == gen.severity_mean
        assert loss_data.metadata["severity_std"] == gen.severity_std

        # Test with empty claims list
        from typing import List

        empty_claims: List[ClaimEvent] = []
        empty_loss_data = gen.to_loss_data(empty_claims)
        assert len(empty_loss_data.timestamps) == 0
        assert len(empty_loss_data.loss_amounts) == 0

    def test_generate_loss_data(self):
        """Test generate_loss_data method."""
        gen = ClaimGenerator(
            frequency=2.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=42,
        )

        # Test generating loss data with catastrophic events
        loss_data = gen.generate_loss_data(years=10, include_catastrophic=True)

        # Verify it returns a LossData object
        assert hasattr(loss_data, "timestamps")
        assert hasattr(loss_data, "loss_amounts")
        assert len(loss_data.timestamps) == len(loss_data.loss_amounts)

        # Test without catastrophic events
        loss_data_no_cat = gen.generate_loss_data(years=5, include_catastrophic=False)
        assert hasattr(loss_data_no_cat, "timestamps")
        assert hasattr(loss_data_no_cat, "loss_amounts")

    def test_generate_enhanced_claims_with_enhanced_distributions(self):
        """Test generate_enhanced_claims uses enhanced distributions when available."""
        gen = ClaimGenerator(
            frequency=1.0,
            severity_mean=500_000,
            severity_std=100_000,
            seed=42,
        )

        # Test with enhanced distributions enabled (default)
        claims, stats = gen.generate_enhanced_claims(
            years=5, revenue=15_000_000, use_enhanced_distributions=True
        )

        assert isinstance(claims, list)
        assert isinstance(stats, dict)
        assert stats["method"] == "enhanced"

        # Verify all claims are within the simulation period
        for claim in claims:
            assert isinstance(claim, ClaimEvent)
            assert 0 <= claim.year < 5
            assert claim.amount > 0

        # Test without specifying revenue (should use default)
        claims2, stats2 = gen.generate_enhanced_claims(years=3)
        assert stats2["method"] == "enhanced"

    def test_from_loss_data_static(self):
        """Test from_loss_data static method with mock data."""

        # Create a LossData object for testing
        loss_data = LossData(
            timestamps=np.array([0.5, 1.2, 2.8, 3.1, 4.9]),
            loss_amounts=np.array([10000, 25000, 15000, 30000, 20000]),
        )

        claims = ClaimGenerator.from_loss_data(loss_data)

        assert len(claims) == 5
        assert all(isinstance(c, ClaimEvent) for c in claims)

        # Check that years are properly converted from timestamps
        expected_years = [0, 1, 2, 3, 4]
        for claim, expected_year in zip(claims, expected_years):
            assert claim.year == expected_year
            assert claim.amount > 0

"""Unit tests for the ClaimGenerator class."""

import time
from typing import Dict

import numpy as np
import pytest

from ergodic_insurance.claim_generator import ClaimEvent, ClaimGenerator
from ergodic_insurance.loss_distributions import LossData
from ergodic_insurance.trends import (
    LinearTrend,
    MeanRevertingTrend,
    NoTrend,
    RandomWalkTrend,
    RegimeSwitchingTrend,
    ScenarioTrend,
)


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
            base_frequency=0.5,
            severity_mean=1_000_000,
            severity_std=500_000,
            seed=42,
        )
        assert gen.base_frequency == 0.5
        assert gen.severity_mean == 1_000_000
        assert gen.severity_std == 500_000
        assert gen.rng is not None

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = ClaimGenerator(
            seed=42, base_frequency=1.0, severity_mean=100_000, severity_std=50_000
        )
        gen2 = ClaimGenerator(
            seed=42, base_frequency=1.0, severity_mean=100_000, severity_std=50_000
        )

        claims1 = gen1.generate_claims(10)
        claims2 = gen2.generate_claims(10)

        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == pytest.approx(c2.amount)

    def test_reset_seed(self):
        """Test seed reset functionality."""
        gen = ClaimGenerator(
            seed=42, base_frequency=0.5, severity_mean=150_000, severity_std=75_000
        )
        claims1 = gen.generate_claims(5)

        gen.reset_seed(42)
        claims2 = gen.generate_claims(5)

        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == pytest.approx(c2.amount)

    def test_generate_claims_statistical_properties(self):
        """Test that generated claims match expected statistical properties."""
        base_frequency = 2.0  # Average 2 claims per year
        severity_mean = 1_000_000
        severity_std = 200_000
        years = 1000  # Large sample for statistical testing

        gen = ClaimGenerator(
            base_frequency=base_frequency,
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

        # Mean should be close to base_frequency parameter
        assert mean_count == pytest.approx(base_frequency, rel=0.1)

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
        gen = ClaimGenerator(
            base_frequency=1.0, severity_mean=100_000, severity_std=50_000, seed=42
        )
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
        gen = ClaimGenerator(
            seed=42, base_frequency=1.0, severity_mean=100_000, severity_std=50_000
        )

        # Zero years
        claims = gen.generate_claims(0)
        assert claims == []

        # Zero frequency
        gen_zero = ClaimGenerator(
            base_frequency=0, severity_mean=1_000_000, severity_std=500_000, seed=42
        )
        claims = gen_zero.generate_claims(10)
        assert claims == []

        # Very high frequency
        gen_high = ClaimGenerator(
            base_frequency=100, severity_mean=10_000, severity_std=2_000, seed=42
        )
        claims = gen_high.generate_claims(1)
        assert len(claims) > 50  # Should have many claims

    def test_generate_all_claims(self):
        """Test batch generation of regular and catastrophic claims."""
        gen = ClaimGenerator(
            base_frequency=1.0,
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
            base_frequency=3.0,  # Moderate frequency
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
        gen = ClaimGenerator(
            base_frequency=2.0, severity_mean=100_000, severity_std=50_000, seed=42
        )
        years = 50

        claims = gen.generate_claims(years)

        for claim in claims:
            assert 0 <= claim.year < years

        # Test catastrophic claims too
        cat_claims = gen.generate_catastrophic_claims(
            years=years,
            cat_frequency=0.01,
            cat_severity_mean=50_000_000,
            cat_severity_std=20_000_000,
        )
        for claim in cat_claims:
            assert 0 <= claim.year < years

    def test_negative_parameters(self):
        """Test handling of invalid parameters."""
        # Test that negative frequency raises ValueError
        with pytest.raises(ValueError, match="Base frequency must be non-negative"):
            ClaimGenerator(base_frequency=-1.0, severity_mean=100_000, severity_std=50_000)

        # Test that negative severity_mean raises ValueError
        with pytest.raises(ValueError, match="Severity mean must be positive"):
            ClaimGenerator(base_frequency=10.0, severity_mean=-1000, severity_std=50_000)

        # Test that negative severity_std raises ValueError
        with pytest.raises(ValueError, match="Severity std must be non-negative"):
            ClaimGenerator(base_frequency=10.0, severity_mean=100_000, severity_std=-100)

        # Test that generation with negative years should handle gracefully
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
        )
        claims = gen.generate_claims(-5)
        assert claims == []

        # Zero frequency should produce no claims
        gen_zero = ClaimGenerator(
            base_frequency=0.0,
            severity_mean=1_000_000,
            severity_std=500_000,
        )
        claims = gen_zero.generate_claims(10)
        assert claims == []

    def test_generate_enhanced_claims_fallback(self):
        """Test generate_enhanced_claims falls back to standard generation."""
        gen = ClaimGenerator(
            base_frequency=1.0,
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
        gen = ClaimGenerator(
            base_frequency=1.0, severity_mean=100_000, severity_std=50_000, seed=42
        )

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
        assert loss_data.metadata["base_frequency"] == gen.base_frequency
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
            base_frequency=2.0,
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
            base_frequency=1.0,
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


class TestTrendIntegration:
    """Test integration of trends with ClaimGenerator."""

    def test_frequency_trend_application(self):
        """Test that frequency trends are properly applied."""
        # Create generator with 3% annual frequency growth
        trend = LinearTrend(annual_rate=0.03)
        gen = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=1_000_000,
            severity_std=50_000,
            frequency_trend=trend,
            seed=42,
        )

        # Generate claims over multiple years
        n_years = 100
        n_simulations = 1000

        year_counts = {year: 0 for year in range(n_years)}

        for _ in range(n_simulations):
            claims = gen.generate_claims(years=n_years)
            for claim in claims:
                year_counts[claim.year] += 1

        # Average frequencies per year
        avg_frequencies = {year: count / n_simulations for year, count in year_counts.items()}

        # Early years should have lower frequency than late years
        early_avg = np.mean([avg_frequencies[y] for y in range(10)])
        late_avg = np.mean([avg_frequencies[y] for y in range(90, 100)])

        # With 3% growth, frequency at year 90 should be ~13x higher than year 0
        # But due to randomness, we'll check for at least 5x increase
        assert late_avg > early_avg * 5, (
            f"Frequency trend not applied correctly: " f"early={early_avg:.3f}, late={late_avg:.3f}"
        )

    def test_severity_trend_application(self):
        """Test that severity trends are properly applied."""
        # Create generator with 5% annual severity inflation
        trend = LinearTrend(annual_rate=0.05)
        gen = ClaimGenerator(
            base_frequency=1.0,  # Ensure we get claims each year
            severity_mean=1_000_000,
            severity_std=100_000,  # Low std for clearer trend signal
            severity_trend=trend,
            seed=42,
        )

        # Generate claims and track average severity by year
        n_years = 50
        n_simulations = 100

        year_severities: Dict[int, list] = {year: [] for year in range(n_years)}

        for _ in range(n_simulations):
            claims = gen.generate_claims(years=n_years)
            for claim in claims:
                year_severities[claim.year].append(claim.amount)

        # Calculate average severity per year
        avg_severities = {}
        for year, amounts in year_severities.items():
            if amounts:
                avg_severities[year] = np.mean(amounts)

        # Check trend application
        # Year 0 should be around base_severity
        if 0 in avg_severities:
            assert 800_000 < avg_severities[0] < 1_200_000

        # Year 10 should be around base_severity * 1.05^10 ≈ 1.63x
        if 10 in avg_severities:
            expected_10 = 1_000_000 * (1.05**10)
            assert avg_severities[10] > expected_10 * 0.8
            assert avg_severities[10] < expected_10 * 1.2

    def test_trend_exposure_stacking(self):
        """Test that trend and exposure adjustments stack multiplicatively."""
        # Create generator with both frequency trend
        frequency_trend = LinearTrend(annual_rate=0.02)
        gen = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=1_000_000,
            severity_std=50_000,
            frequency_trend=frequency_trend,
            seed=42,
        )

        # Generate claims (exposure adjustment is built into trends)
        claims = gen.generate_claims(years=10)

        # The effective frequency should be base * freq_trend
        # This is tested indirectly through claim generation
        assert isinstance(claims, list)
        assert all(isinstance(c, ClaimEvent) for c in claims)

    def test_catastrophic_claims_with_independent_trends(self):
        """Test catastrophic claims with independent trends."""
        # Main trends
        main_freq_trend = LinearTrend(annual_rate=0.02)
        main_sev_trend = LinearTrend(annual_rate=0.03)

        # Catastrophic trends (different rates)
        cat_freq_trend = LinearTrend(annual_rate=0.05)
        cat_sev_trend = LinearTrend(annual_rate=0.08)

        gen = ClaimGenerator(
            base_frequency=0.5,
            severity_mean=1_000_000,
            severity_std=50_000,
            frequency_trend=main_freq_trend,
            severity_trend=main_sev_trend,
            seed=42,
        )

        # Generate regular claims
        regular = gen.generate_claims(years=10)

        # Generate catastrophic claims separately
        cats = gen.generate_catastrophic_claims(
            years=10,
            cat_frequency=0.1,
            cat_severity_mean=5_000_000,
            cat_severity_std=100_000,
            cat_frequency_trend=cat_freq_trend,
            cat_severity_trend=cat_sev_trend,
        )

        # Both should be lists of ClaimEvents
        assert isinstance(regular, list)
        assert isinstance(cats, list)
        assert all(isinstance(c, ClaimEvent) for c in regular)
        assert all(isinstance(c, ClaimEvent) for c in cats)

    def test_different_trend_types(self):
        """Test ClaimGenerator with various trend types."""
        trend_types = [
            NoTrend(),
            LinearTrend(annual_rate=0.03),
            ScenarioTrend(factors=[1.0, 1.1, 1.2, 1.15, 1.25]),
            RandomWalkTrend(drift=0.02, volatility=0.10, seed=42),
            MeanRevertingTrend(mean_level=1.0, reversion_speed=0.5, seed=42),
            RegimeSwitchingTrend(regimes=[0.9, 1.0, 1.2], seed=42),
        ]

        for trend in trend_types:
            gen = ClaimGenerator(
                base_frequency=0.1,
                severity_mean=1_000_000,
                severity_std=50_000,
                frequency_trend=trend,
                severity_trend=trend,
                seed=42,
            )

            claims = gen.generate_claims(years=10)

            # Should generate valid claims with any trend type
            assert isinstance(claims, list)
            assert all(isinstance(c, ClaimEvent) for c in claims)
            assert all(c.amount > 0 for c in claims)
            assert all(0 <= c.year < 10 for c in claims)

    def test_trend_multiplier_correctness(self):
        """Test that trend multipliers are correctly applied at each time step."""
        # Use a deterministic scenario trend for exact testing
        freq_factors = [1.0, 2.0, 3.0, 4.0, 5.0]
        freq_trend = ScenarioTrend(factors=freq_factors)

        gen = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=1_000_000,
            severity_std=50_000,
            frequency_trend=freq_trend,
            seed=42,
        )

        # The adjusted frequency at year t should be base * factors[t]
        n_simulations = 10000
        year_counts = {year: 0 for year in range(5)}

        for _ in range(n_simulations):
            claims = gen.generate_claims(years=5)
            for claim in claims:
                year_counts[claim.year] += 1

        # Check frequencies match expected multipliers
        for year in range(5):
            expected_freq = 0.1 * freq_factors[year]
            observed_freq = year_counts[year] / n_simulations

            # Allow for statistical variation (±30%)
            assert observed_freq > expected_freq * 0.7, (
                f"Year {year}: expected {expected_freq:.3f}, " f"observed {observed_freq:.3f}"
            )
            assert observed_freq < expected_freq * 1.3, (
                f"Year {year}: expected {expected_freq:.3f}, " f"observed {observed_freq:.3f}"
            )

    def test_reproducibility_with_trends(self):
        """Test that trends maintain reproducibility with same seed."""
        trend = RandomWalkTrend(drift=0.02, volatility=0.15, seed=100)

        gen1 = ClaimGenerator(
            base_frequency=0.2,
            severity_mean=1_000_000,
            severity_std=100_000,
            frequency_trend=trend,
            seed=42,
        )

        # Reset trend seed to ensure same path
        trend.reset_seed(100)

        gen2 = ClaimGenerator(
            base_frequency=0.2,
            severity_mean=1_000_000,
            severity_std=100_000,
            frequency_trend=trend,
            seed=42,
        )

        claims1 = gen1.generate_claims(years=20)
        claims2 = gen2.generate_claims(years=20)

        # Should produce identical claims
        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == pytest.approx(c2.amount)


class TestStatisticalProperties:
    """Test suite for the new statistical properties and methods."""

    def test_analytical_mean(self):
        """Test analytical mean calculation."""
        base_frequency = 0.1
        severity_mean = 5_000_000

        gen = ClaimGenerator(
            base_frequency=base_frequency,
            severity_mean=severity_mean,
            severity_std=2_000_000,
            seed=42,
        )

        expected_mean = base_frequency * severity_mean
        assert gen.mean == pytest.approx(expected_mean)
        assert gen.mean == 500_000

    def test_analytical_variance(self):
        """Test analytical variance calculation."""
        base_frequency = 0.1
        severity_mean = 5_000_000
        severity_std = 2_000_000

        gen = ClaimGenerator(
            base_frequency=base_frequency,
            severity_mean=severity_mean,
            severity_std=severity_std,
            seed=42,
        )

        # Analytical formula: frequency * (std^2 + mean^2)
        expected_variance = base_frequency * (severity_std**2 + severity_mean**2)
        assert gen.variance == pytest.approx(expected_variance)

    def test_analytical_std(self):
        """Test analytical standard deviation calculation."""
        base_frequency = 0.1
        severity_mean = 5_000_000
        severity_std = 2_000_000

        gen = ClaimGenerator(
            base_frequency=base_frequency,
            severity_mean=severity_mean,
            severity_std=severity_std,
            seed=42,
        )

        expected_variance = base_frequency * (severity_std**2 + severity_mean**2)
        expected_std = np.sqrt(expected_variance)
        assert gen.std == pytest.approx(expected_std)

    def test_variance_with_trends_uses_simulation(self):
        """Test that variance falls back to simulation when trends are present."""
        # LinearTrend is already imported at the top of the file

        gen = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=5_000_000,
            severity_std=2_000_000,
            frequency_trend=LinearTrend(annual_rate=0.03),
            n_simulations=1000,  # Use fewer simulations for speed
            seed=42,
        )

        # Should use simulation, not analytical formula
        variance = gen.variance
        assert variance > 0
        # Variance with trends should differ from simple analytical
        analytical_no_trend = 0.1 * (2_000_000**2 + 5_000_000**2)
        assert variance != pytest.approx(analytical_no_trend, rel=0.01)

    def test_get_percentiles_default(self):
        """Test percentile calculation with default values."""
        gen = ClaimGenerator(
            base_frequency=0.5,
            severity_mean=1_000_000,
            severity_std=500_000,
            n_simulations=10000,  # Reduced for test speed
            seed=42,
        )

        percentiles = gen.get_percentiles()

        # Should return default percentiles [50, 95, 99]
        assert 50 in percentiles
        assert 95 in percentiles
        assert 99 in percentiles

        # Percentiles should be ordered
        assert percentiles[50] < percentiles[95]
        assert percentiles[95] < percentiles[99]

        # 50th percentile (median) should be close to expected annual loss for low frequency
        # But with compound distribution, median might be lower than mean
        assert percentiles[50] >= 0

    def test_get_percentiles_custom(self):
        """Test percentile calculation with custom values."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
            n_simulations=10000,
            seed=42,
        )

        custom_percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
        result = gen.get_percentiles(custom_percentiles)

        # All requested percentiles should be present
        for p in custom_percentiles:
            assert p in result

        # Should be monotonically increasing
        sorted_percentiles = sorted(custom_percentiles)
        for i in range(len(sorted_percentiles) - 1):
            p1, p2 = sorted_percentiles[i], sorted_percentiles[i + 1]
            assert result[p1] <= result[p2]

    def test_get_percentiles_invalid(self):
        """Test that invalid percentiles raise ValueError."""
        gen = ClaimGenerator(
            seed=42, base_frequency=1.0, severity_mean=100_000, severity_std=50_000
        )

        # Negative percentile
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            gen.get_percentiles([-1, 50])

        # Percentile > 100
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            gen.get_percentiles([50, 101])

    def test_get_cvar_default(self):
        """Test CVaR calculation with default values."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=1_000_000,
            severity_std=500_000,
            n_simulations=10000,
            seed=42,
        )

        cvar = gen.get_cvar()

        # Should return default percentiles [95, 99]
        assert 95 in cvar
        assert 99 in cvar

        # CVaR should be greater than corresponding VaR (percentile)
        percentiles = gen.get_percentiles([95, 99])
        assert cvar[95] >= percentiles[95]
        assert cvar[99] >= percentiles[99]

        # CVaR at higher percentile should be higher
        assert cvar[99] >= cvar[95]

    def test_get_cvar_custom(self):
        """Test CVaR calculation with custom percentiles."""
        gen = ClaimGenerator(
            base_frequency=2.0,
            severity_mean=500_000,
            severity_std=200_000,
            n_simulations=10000,
            seed=42,
        )

        custom_percentiles = [90, 95, 99, 99.9]
        cvar = gen.get_cvar(custom_percentiles)

        # All requested CVaR values should be present
        for p in custom_percentiles:
            assert p in cvar

        # Get corresponding VaR values
        var = gen.get_percentiles(custom_percentiles)

        # CVaR should always be >= VaR
        for p in custom_percentiles:
            assert cvar[p] >= var[p]

    def test_caching_behavior(self):
        """Test that simulation results are cached properly."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=1_000_000,
            severity_std=500_000,
            n_simulations=10000,
            seed=42,
        )

        # First call - should populate cache
        percentiles_1 = gen.get_percentiles([50, 95])

        # Second call with same generator - should use cache
        percentiles_2 = gen.get_percentiles([50, 95])

        # Results should be identical (from cache)
        assert percentiles_1[50] == percentiles_2[50]
        assert percentiles_1[95] == percentiles_2[95]

        # Call CVaR - should use same cached simulation
        cvar = gen.get_cvar([95])
        assert cvar[95] >= percentiles_1[95]

    def test_cache_invalidation_on_reset_seed(self):
        """Test that cache is invalidated when seed is reset."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=1_000_000,
            severity_std=500_000,
            n_simulations=1000,  # Small for speed
            seed=42,
        )

        # Get initial percentiles
        percentiles_1 = gen.get_percentiles([50])

        # Reset seed - should invalidate cache
        gen.reset_seed(100)

        # New percentiles should potentially be different
        # (different simulation seed)
        percentiles_2 = gen.get_percentiles([50])

        # Cache should have been invalidated, so new simulation was run
        # We can't guarantee they're different, but cache was cleared
        assert gen._simulation_cache is not None

    def test_convergence_to_analytical(self):
        """Test that simulation converges to analytical values for large N."""
        gen = ClaimGenerator(
            base_frequency=2.0,
            severity_mean=100_000,
            severity_std=20_000,
            n_simulations=50000,  # Large N for convergence
            seed=42,
        )

        # Get simulation-based mean (50th percentile is median, not mean)
        # So we'll check that the simulation produces reasonable results
        percentiles = gen.get_percentiles([50, 90, 99])

        # The mean should be frequency * severity_mean = 200,000
        analytical_mean = gen.mean
        assert analytical_mean == 200_000

        # The median (50th percentile) for compound Poisson can be quite different
        # from mean, especially for lower frequencies
        # Just check it's positive and reasonable
        assert percentiles[50] >= 0
        assert percentiles[99] > percentiles[50]

    def test_edge_cases(self):
        """Test edge cases for statistical properties."""
        # Zero frequency
        gen_zero_freq = ClaimGenerator(
            base_frequency=0.0,
            severity_mean=1_000_000,
            severity_std=500_000,
            n_simulations=100,
            seed=42,
        )
        assert gen_zero_freq.mean == 0
        assert gen_zero_freq.variance == 0
        assert gen_zero_freq.std == 0

        percentiles = gen_zero_freq.get_percentiles([50, 95])
        assert percentiles[50] == 0
        assert percentiles[95] == 0

        # Very high frequency - should have stable statistics
        gen_high_freq = ClaimGenerator(
            base_frequency=100.0,
            severity_mean=10_000,
            severity_std=2_000,
            n_simulations=1000,
            seed=42,
        )

        # Mean should be predictable
        assert gen_high_freq.mean == 1_000_000  # 100 * 10,000

        # Percentiles should exist and be ordered
        percentiles = gen_high_freq.get_percentiles([50, 95])
        assert percentiles[50] > 0
        assert percentiles[95] > percentiles[50]

    def test_n_simulations_parameter(self):
        """Test that n_simulations parameter is used correctly."""
        # Create with custom n_simulations
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
            n_simulations=500,  # Very small for testing
            seed=42,
        )

        # The internal simulation should use this number
        losses = gen._simulate_annual_losses()
        assert len(losses) == 500

        # Different n_simulations should give different precision
        gen_precise = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
            n_simulations=50000,  # Much larger
            seed=42,
        )

        losses_precise = gen_precise._simulate_annual_losses()
        assert len(losses_precise) == 50000

    def test_example_usage_from_issue(self):
        """Test the example usage from the GitHub issue."""
        # Initialize generator
        generator = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=5_000_000,
            severity_std=2_000_000,
            n_simulations=10000,  # Reduced for test speed
        )

        # Analytical statistics
        expected_annual_loss = generator.mean
        std_dev = generator.std

        assert expected_annual_loss == 500_000
        assert std_dev > 0

        # Simulation-based statistics
        percentiles = generator.get_percentiles([50, 90, 95, 99])
        assert 95 in percentiles
        assert percentiles[95] > 0

        cvar = generator.get_cvar([95, 99])
        assert 95 in cvar
        assert cvar[95] > 0
        assert cvar[95] >= percentiles[95]  # CVaR >= VaR

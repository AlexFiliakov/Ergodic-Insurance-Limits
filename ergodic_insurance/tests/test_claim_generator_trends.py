"""Tests for trend support in ClaimGenerator.

Tests the integration of frequency and severity trends with the ClaimGenerator
class, ensuring proper stacking with exposure bases and backward compatibility.
"""

import numpy as np
import pytest

from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.trends import LinearTrend, NoTrend, ScenarioTrend


class TestClaimGeneratorTrends:
    """Test trend support in ClaimGenerator."""

    def test_default_trends_backward_compatibility(self):
        """Test that default NoTrend maintains backward compatibility."""
        # Generator without explicit trends should behave as before
        gen1 = ClaimGenerator(
            base_frequency=0.5, severity_mean=1_000_000, severity_std=500_000, seed=42
        )

        # Generator with explicit NoTrend should produce same results
        gen2 = ClaimGenerator(
            base_frequency=0.5,
            severity_mean=1_000_000,
            severity_std=500_000,
            frequency_trend=NoTrend(),
            severity_trend=NoTrend(),
            seed=42,
        )

        claims1 = gen1.generate_claims(years=10)
        claims2 = gen2.generate_claims(years=10)

        # Should generate identical claims
        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == c2.amount

    def test_frequency_trend_application(self):
        """Test that frequency trends properly adjust claim counts."""
        # Create generator with increasing frequency trend
        gen = ClaimGenerator(
            base_frequency=2.0,  # Higher base for more reliable statistics
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.20),  # 20% annual growth for clearer effect
            seed=42,  # Different seed for better distribution
        )

        # Check that adjusted frequency increases correctly
        freq_0 = gen.get_adjusted_frequency(0)
        freq_5 = gen.get_adjusted_frequency(5)
        freq_10 = gen.get_adjusted_frequency(10)

        # Verify trend application
        assert abs(freq_0 - 2.0) < 0.001
        assert abs(freq_5 - 2.0 * (1.20**5)) < 0.001
        assert abs(freq_10 - 2.0 * (1.20**10)) < 0.001

        # Generate many years to test statistical behavior
        claims = gen.generate_claims(years=30)
        early = [c for c in claims if c.year < 10]
        late = [c for c in claims if c.year >= 20]

        # With strong trend, late period should have significantly more claims
        # Even with randomness, the difference should be substantial
        if len(early) > 0:  # Protect against edge case
            ratio = len(late) / max(1, len(early))
            assert ratio > 2.0  # Late period should have at least 2x more claims

    def test_severity_trend_application(self):
        """Test that severity trends properly adjust claim amounts."""
        gen = ClaimGenerator(
            base_frequency=2.0,  # Ensure we get claims
            severity_mean=100_000,
            severity_std=10_000,
            severity_trend=LinearTrend(annual_rate=0.05),  # 5% annual inflation
            seed=42,
        )

        # Generate claims and group by year
        claims = gen.generate_claims(years=20)
        from typing import Dict, List

        by_year: Dict[int, List[float]] = {}
        for claim in claims:
            by_year.setdefault(claim.year, []).append(claim.amount)

        # Calculate average severity by period
        early_avg = np.mean(
            [amt for year, amounts in by_year.items() if year < 5 for amt in amounts]
        )
        late_avg = np.mean(
            [amt for year, amounts in by_year.items() if year >= 15 for amt in amounts]
        )

        # Late period should have higher average severity
        # With 5% trend over 15+ years: factor ~2.08
        assert late_avg > early_avg * 1.5

    def test_trends_stack_with_exposure(self):
        """Test that trends and exposure bases stack multiplicatively."""

        # Create a simple mock exposure base with constant 2x multiplier
        class MockExposure:
            def get_frequency_multiplier(self, time: float) -> float:
                return 2.0

        exposure = MockExposure()

        # Create generator with both exposure and trend
        gen = ClaimGenerator(
            base_frequency=1.0,
            exposure_base=exposure,  # type: ignore[arg-type]
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.10),  # 10% trend
            seed=999,
        )

        # Check frequency stacking at year 5
        # Base: 1.0, Exposure: 2.0x, Trend at year 5: ~1.61x
        # Total: 1.0 * 2.0 * 1.61 = ~3.22
        adjusted_freq = gen.get_adjusted_frequency(5)
        expected = 1.0 * 2.0 * (1.10**5)
        assert abs(adjusted_freq - expected) < 0.01

    def test_get_adjusted_severity_method(self):
        """Test the get_adjusted_severity method."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=1_000_000,
            severity_trend=LinearTrend(annual_rate=0.03),  # 3% inflation
        )

        # Check severity at different years
        sev_0 = gen.get_adjusted_severity(0)
        sev_5 = gen.get_adjusted_severity(5)
        sev_10 = gen.get_adjusted_severity(10)

        assert sev_0 == 1_000_000  # No adjustment at year 0
        assert abs(sev_5 - 1_000_000 * (1.03**5)) < 1
        assert abs(sev_10 - 1_000_000 * (1.03**10)) < 1

    def test_catastrophic_claims_with_trends(self):
        """Test that catastrophic claims support independent trends."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.02),  # 2% regular trend
            severity_trend=LinearTrend(annual_rate=0.03),  # 3% regular trend
            seed=111,
        )

        # Generate catastrophic claims with different trends
        cat_claims = gen.generate_catastrophic_claims(
            years=100,
            cat_frequency=0.10,  # Higher frequency for testing
            cat_severity_mean=1_000_000,
            cat_frequency_trend=LinearTrend(annual_rate=0.05),  # 5% cat frequency trend
            cat_severity_trend=LinearTrend(annual_rate=0.07),  # 7% cat severity trend
        )

        # Group by early vs late periods
        early_cat = [c for c in cat_claims if c.year < 30]
        late_cat = [c for c in cat_claims if c.year >= 70]

        # Later period should have more frequent and severe catastrophes
        assert len(late_cat) > len(early_cat)

        if early_cat and late_cat:
            early_avg = np.mean([c.amount for c in early_cat])
            late_avg = np.mean([c.amount for c in late_cat])
            # With 7% trend over 70+ years, severity should be much higher
            assert late_avg > early_avg * 3

    def test_catastrophic_claims_inherit_main_trends(self):
        """Test that catastrophic claims use main trends when not specified."""
        gen = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.04),
            severity_trend=LinearTrend(annual_rate=0.06),
            seed=222,
        )

        # Generate without specifying cat trends - should use main trends
        cat_claims1 = gen.generate_catastrophic_claims(
            years=50, cat_frequency=0.20, cat_severity_mean=500_000  # High frequency for testing
        )

        # Generate with explicit same trends
        cat_claims2 = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.04),
            severity_trend=LinearTrend(annual_rate=0.06),
            seed=222,
        ).generate_catastrophic_claims(
            years=50,
            cat_frequency=0.20,
            cat_severity_mean=500_000,
            cat_frequency_trend=LinearTrend(annual_rate=0.04),
            cat_severity_trend=LinearTrend(annual_rate=0.06),
        )

        # Should produce identical results
        assert len(cat_claims1) == len(cat_claims2)
        for c1, c2 in zip(cat_claims1, cat_claims2):
            assert c1.year == c2.year
            assert abs(c1.amount - c2.amount) < 0.01

    def test_generate_all_claims_with_trends(self):
        """Test generate_all_claims with trend parameters."""
        gen = ClaimGenerator(
            base_frequency=0.5,
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.02),
            severity_trend=LinearTrend(annual_rate=0.03),
            seed=333,
        )

        regular, catastrophic = gen.generate_all_claims(
            years=50,
            include_catastrophic=True,
            cat_frequency=0.05,
            cat_severity_mean=1_000_000,
            cat_frequency_trend=LinearTrend(annual_rate=0.01),
            cat_severity_trend=LinearTrend(annual_rate=0.05),
        )

        # Both types should have claims
        assert len(regular) > 0
        # Catastrophic might be empty due to randomness, but should work without error

        # Check that later regular claims are larger on average
        if len(regular) > 10:
            early_reg = [c.amount for c in regular if c.year < 10]
            late_reg = [c.amount for c in regular if c.year >= 40]
            if early_reg and late_reg:
                assert np.mean(late_reg) > np.mean(early_reg)

    def test_scenario_trend_integration(self):
        """Test integration with ScenarioTrend."""
        # Create a custom scenario with varying rates
        factors = [1.0, 1.02, 1.05, 1.08, 1.15, 1.25]  # Increasing factors
        scenario = ScenarioTrend(factors=factors, time_unit="annual")

        gen = ClaimGenerator(
            base_frequency=2.0,
            severity_mean=100_000,
            severity_std=10_000,  # Lower variance for more predictable test
            frequency_trend=scenario,
            severity_trend=scenario,
            seed=444,
        )

        # Test that trends are applied correctly
        # At year 4, multiplier should be 1.15
        freq_4 = gen.get_adjusted_frequency(4)
        sev_4 = gen.get_adjusted_severity(4)

        assert abs(freq_4 - 2.0 * 1.15) < 0.01
        assert abs(sev_4 - 100_000 * 1.15) < 1

        # Generate claims to verify overall behavior
        claims = gen.generate_claims(years=6)

        # Check that claims exist and follow expected pattern
        assert len(claims) > 0

        # Later claims should generally be larger
        early_claims = [c for c in claims if c.year < 2]
        late_claims = [c for c in claims if c.year >= 4]

        if early_claims and late_claims:
            early_avg = np.mean([c.amount for c in early_claims])
            late_avg = np.mean([c.amount for c in late_claims])
            # With trend factors, late should be higher on average
            assert late_avg > early_avg

    def test_trend_reproducibility(self):
        """Test that trends work with seed reproducibility."""
        trend = LinearTrend(annual_rate=0.05)

        gen1 = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            frequency_trend=trend,
            severity_trend=trend,
            seed=555,
        )

        gen2 = ClaimGenerator(
            base_frequency=1.0,
            severity_mean=100_000,
            frequency_trend=trend,
            severity_trend=trend,
            seed=555,
        )

        claims1 = gen1.generate_claims(years=20)
        claims2 = gen2.generate_claims(years=20)

        # Should generate identical claims
        assert len(claims1) == len(claims2)
        for c1, c2 in zip(claims1, claims2):
            assert c1.year == c2.year
            assert c1.amount == c2.amount

    def test_zero_base_with_trends(self):
        """Test edge case of zero base frequency with trends."""
        gen = ClaimGenerator(
            base_frequency=0.0,  # No base frequency
            severity_mean=100_000,
            frequency_trend=LinearTrend(annual_rate=0.10),
            seed=666,
        )

        claims = gen.generate_claims(years=10)

        # Should generate no claims even with trend (0 * anything = 0)
        assert len(claims) == 0

    def test_high_trend_rates(self):
        """Test system behavior with high trend rates."""
        gen = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=1000,
            severity_std=100,
            frequency_trend=LinearTrend(annual_rate=0.50),  # 50% annual growth
            severity_trend=LinearTrend(annual_rate=0.50),  # 50% annual growth
            seed=777,
        )

        # Generate claims - should work without error
        claims = gen.generate_claims(years=10)

        # Year 9 should have very high frequency and severity
        year_9_claims = [c for c in claims if c.year == 9]
        if year_9_claims:
            # Frequency multiplier at year 9: 1.5^9 ≈ 38.4
            # Severity multiplier at year 9: 1.5^9 ≈ 38.4
            avg_amount = np.mean([c.amount for c in year_9_claims])
            assert avg_amount > 10_000  # Much higher than base 1000

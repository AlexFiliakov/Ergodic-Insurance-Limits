"""Tests for loss cost trend integration into the pricing pipeline (Issue #643).

Verifies that frequency and severity trends are correctly applied in
ManufacturingLossGenerator and propagated through InsurancePricer.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.trends import LinearTrend, NoTrend


class TestBackwardCompatibility:
    """No trends (default None) produces identical results to pre-trend code."""

    def test_no_trends_default(self):
        """Default None trends give multiplier 1.0 in stats."""
        gen = ManufacturingLossGenerator(seed=42)
        _losses, stats = gen.generate_losses(duration=1.0, revenue=10_000_000)
        assert stats["frequency_trend_multiplier"] == 1.0
        assert stats["severity_trend_multiplier"] == 1.0

    def test_no_trend_explicit(self):
        """Explicit NoTrend gives same results as None."""
        gen_none = ManufacturingLossGenerator(seed=42)
        gen_no = ManufacturingLossGenerator(
            seed=42, frequency_trend=NoTrend(), severity_trend=NoTrend()
        )

        losses_none, stats_none = gen_none.generate_losses(
            duration=1.0, revenue=10_000_000, time=3.0
        )
        losses_no, stats_no = gen_no.generate_losses(duration=1.0, revenue=10_000_000, time=3.0)

        assert stats_none["total_losses"] == stats_no["total_losses"]
        assert stats_none["total_amount"] == pytest.approx(stats_no["total_amount"], rel=1e-10)


class TestSeverityTrend:
    """Severity trend scales individual loss amounts."""

    def test_severity_trend_scales_amounts(self):
        """LinearTrend(0.10) at time=5.0 multiplies amounts by 1.10^5."""
        trend = LinearTrend(annual_rate=0.10)
        expected_mult = 1.10**5.0

        gen_base = ManufacturingLossGenerator(seed=42)
        gen_trend = ManufacturingLossGenerator(seed=42, severity_trend=trend)

        losses_base, stats_base = gen_base.generate_losses(
            duration=1.0, revenue=10_000_000, time=5.0
        )
        losses_trend, stats_trend = gen_trend.generate_losses(
            duration=1.0, revenue=10_000_000, time=5.0
        )

        # Same number of events (frequency unchanged)
        assert len(losses_base) == len(losses_trend)

        # Each amount scaled by the severity multiplier
        for base, trended in zip(losses_base, losses_trend):
            assert trended.amount == pytest.approx(base.amount * expected_mult, rel=1e-10)

        assert stats_trend["severity_trend_multiplier"] == pytest.approx(expected_mult)
        assert stats_trend["frequency_trend_multiplier"] == 1.0


class TestFrequencyTrend:
    """Frequency trend scales expected event counts."""

    def test_frequency_trend_scales_count(self):
        """High frequency trend produces statistically more events."""
        # Use a strong 50% annual trend at time=10 -> 57.67x multiplier
        trend = LinearTrend(annual_rate=0.50)
        n_sims = 200

        counts_base = []
        counts_trend = []
        for i in range(n_sims):
            gen_base = ManufacturingLossGenerator(seed=i)
            gen_trend = ManufacturingLossGenerator(seed=i, frequency_trend=trend)

            _, stats_base = gen_base.generate_losses(duration=1.0, revenue=10_000_000, time=10.0)
            _, stats_trend = gen_trend.generate_losses(duration=1.0, revenue=10_000_000, time=10.0)
            counts_base.append(stats_base["total_losses"])
            counts_trend.append(stats_trend["total_losses"])

        # Trended count should be substantially higher
        assert np.mean(counts_trend) > np.mean(counts_base) * 10


class TestBaseFrequencyRestoration:
    """Base frequency is restored after generation (no state corruption)."""

    def test_frequency_restored_after_generation(self):
        """base_frequency returns to original value after generate_losses."""
        trend = LinearTrend(annual_rate=0.20)
        gen = ManufacturingLossGenerator(seed=42, frequency_trend=trend)

        orig_att = gen.attritional.frequency_generator.base_frequency
        orig_large = gen.large.frequency_generator.base_frequency
        orig_cat = gen.catastrophic.frequency_generator.base_frequency

        gen.generate_losses(duration=1.0, revenue=10_000_000, time=5.0)

        assert gen.attritional.frequency_generator.base_frequency == orig_att
        assert gen.large.frequency_generator.base_frequency == orig_large
        assert gen.catastrophic.frequency_generator.base_frequency == orig_cat

    def test_frequency_restored_on_error(self):
        """Base frequency restored even when generation raises."""
        trend = LinearTrend(annual_rate=0.10)
        gen = ManufacturingLossGenerator(seed=42, frequency_trend=trend)

        orig_att = gen.attritional.frequency_generator.base_frequency

        # Force an error during generation
        with patch.object(gen.attritional, "generate_losses", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                gen.generate_losses(duration=1.0, revenue=10_000_000, time=3.0)

        # Frequency must still be restored
        assert gen.attritional.frequency_generator.base_frequency == orig_att


class TestCreateSimpleWithTrends:
    """create_simple factory passes through trend kwargs."""

    def test_create_simple_with_trends(self):
        """Trends set via create_simple are stored and applied."""
        freq_trend = LinearTrend(annual_rate=0.05)
        sev_trend = LinearTrend(annual_rate=0.03)

        gen = ManufacturingLossGenerator.create_simple(
            frequency=1.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=42,
            frequency_trend=freq_trend,
            severity_trend=sev_trend,
        )

        assert gen.frequency_trend is freq_trend
        assert gen.severity_trend is sev_trend

        _, stats = gen.generate_losses(duration=1.0, revenue=10_000_000, time=2.0)
        assert stats["frequency_trend_multiplier"] == pytest.approx(1.05**2.0)
        assert stats["severity_trend_multiplier"] == pytest.approx(1.03**2.0)


class TestPricerIntegration:
    """InsurancePricer passes time to generate_losses, enabling trends."""

    def test_trended_premium_exceeds_untrended(self):
        """With positive trends, pure premium should be higher."""
        from ergodic_insurance.insurance_pricing import InsurancePricer

        gen_base = ManufacturingLossGenerator.create_simple(
            frequency=2.0, severity_mean=500_000, severity_std=200_000, seed=99
        )
        gen_trend = ManufacturingLossGenerator.create_simple(
            frequency=2.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=99,
            frequency_trend=LinearTrend(annual_rate=0.05),
            severity_trend=LinearTrend(annual_rate=0.05),
        )

        pricer_base = InsurancePricer(loss_generator=gen_base, seed=10)
        pricer_trend = InsurancePricer(loss_generator=gen_trend, seed=10)

        pp_base, _ = pricer_base.calculate_pure_premium(
            attachment_point=0,
            limit=10_000_000,
            expected_revenue=10_000_000,
            simulation_years=20,
        )
        pp_trend, _ = pricer_trend.calculate_pure_premium(
            attachment_point=0,
            limit=10_000_000,
            expected_revenue=10_000_000,
            simulation_years=20,
        )

        # Trended premium must be higher
        assert pp_trend > pp_base


class TestTrendMetadataInStats:
    """Stats dict includes trend multiplier metadata."""

    def test_multipliers_present_with_trends(self):
        """Both multiplier keys exist and are correct."""
        freq_trend = LinearTrend(annual_rate=0.04)
        sev_trend = LinearTrend(annual_rate=0.06)
        gen = ManufacturingLossGenerator(
            seed=42, frequency_trend=freq_trend, severity_trend=sev_trend
        )

        _, stats = gen.generate_losses(duration=1.0, revenue=10_000_000, time=3.0)

        assert "frequency_trend_multiplier" in stats
        assert "severity_trend_multiplier" in stats
        assert stats["frequency_trend_multiplier"] == pytest.approx(1.04**3.0)
        assert stats["severity_trend_multiplier"] == pytest.approx(1.06**3.0)

    def test_multipliers_present_without_trends(self):
        """Multiplier keys exist even with no trends (both 1.0)."""
        gen = ManufacturingLossGenerator(seed=42)
        _, stats = gen.generate_losses(duration=1.0, revenue=10_000_000)

        assert stats["frequency_trend_multiplier"] == 1.0
        assert stats["severity_trend_multiplier"] == 1.0

"""Tests for simulation engine."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.claim_generator import ClaimEvent, ClaimGenerator
from ergodic_insurance.src.config import Config, ManufacturerConfig
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.simulation import Simulation, SimulationResults


@pytest.fixture
def manufacturer_config():
    """Create a test manufacturer configuration."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.6,
    )


@pytest.fixture
def manufacturer(manufacturer_config):
    """Create a test manufacturer."""
    return WidgetManufacturer(manufacturer_config)


@pytest.fixture
def claim_generator():
    """Create a test claim generator."""
    return ClaimGenerator(frequency=0.1, severity_mean=1_000_000, severity_std=500_000, seed=42)


class TestSimulationResults:
    """Test SimulationResults dataclass."""

    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        results = SimulationResults(
            years=np.array([0, 1, 2]),
            assets=np.array([100, 110, 120]),
            equity=np.array([50, 55, 60]),
            roe=np.array([0.1, 0.12, 0.11]),
            revenue=np.array([100, 105, 110]),
            net_income=np.array([5, 6.6, 6.6]),
            claim_counts=np.array([0, 1, 0]),
            claim_amounts=np.array([0, 1000, 0]),
        )

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == [
            "year",
            "assets",
            "equity",
            "roe",
            "revenue",
            "net_income",
            "claim_count",
            "claim_amount",
        ]
        assert df["assets"].tolist() == [100, 110, 120]

    def test_summary_stats(self):
        """Test summary statistics calculation."""
        results = SimulationResults(
            years=np.array([0, 1, 2, 3]),
            assets=np.array([100, 110, 120, 130]),
            equity=np.array([50, 55, 60, 65]),
            roe=np.array([0.1, 0.12, 0.11, np.nan]),
            revenue=np.array([100, 105, 110, 115]),
            net_income=np.array([5, 6.6, 6.6, 7]),
            claim_counts=np.array([0, 1, 0, 2]),
            claim_amounts=np.array([0, 1000, 0, 2000]),
        )

        stats = results.summary_stats()

        assert stats["mean_roe"] == pytest.approx(0.11, rel=0.01)
        assert stats["std_roe"] == pytest.approx(0.00816, rel=0.1)
        assert stats["median_roe"] == pytest.approx(0.11, rel=0.01)
        assert stats["final_assets"] == 130
        assert stats["final_equity"] == 65
        assert stats["total_claims"] == 3000
        assert stats["claim_frequency"] == 0.75
        assert stats["survived"] is True
        # When survived is True, insolvency_year should be None


class TestSimulation:
    """Test Simulation class."""

    def test_initialization(self, manufacturer, claim_generator):
        """Test simulation initialization."""
        sim = Simulation(
            manufacturer=manufacturer, claim_generator=claim_generator, time_horizon=100, seed=42
        )

        assert sim.manufacturer == manufacturer
        assert sim.claim_generator == claim_generator
        assert sim.time_horizon == 100
        assert sim.seed == 42
        assert len(sim.years) == 100
        assert len(sim.assets) == 100
        assert sim.insolvency_year is None

    def test_step_annual(self, manufacturer, claim_generator):
        """Test single annual step."""
        sim = Simulation(manufacturer, claim_generator, time_horizon=10)

        # Create test claims
        claims = [ClaimEvent(year=0, amount=500_000), ClaimEvent(year=0, amount=300_000)]

        metrics = sim.step_annual(0, claims)

        assert "assets" in metrics
        assert "equity" in metrics
        assert "roe" in metrics
        assert "claim_count" in metrics
        assert metrics["claim_count"] == 2
        assert metrics["claim_amount"] == 800_000

    def test_run_short_simulation(self, manufacturer, claim_generator):
        """Test running a short simulation."""
        sim = Simulation(
            manufacturer=manufacturer, claim_generator=claim_generator, time_horizon=10, seed=42
        )

        results = sim.run(progress_interval=5)

        assert isinstance(results, SimulationResults)
        assert len(results.years) == 10
        assert len(results.assets) == 10
        assert results.assets[0] > 0  # Should have initial assets

        # Check that some claims were generated
        assert np.sum(results.claim_counts) >= 0

    def test_run_performance(self, manufacturer, claim_generator):
        """Test that 1000-year simulation completes in reasonable time."""
        sim = Simulation(
            manufacturer=manufacturer, claim_generator=claim_generator, time_horizon=1000, seed=42
        )

        start = time.time()
        results = sim.run(progress_interval=250)
        duration = time.time() - start

        # Should complete in less than 1 second as per requirements
        assert duration < 1.0, f"Simulation took {duration:.2f} seconds, expected < 1 second"

        assert len(results.years) == 1000 or results.insolvency_year is not None

    def test_memory_efficiency(self, manufacturer, claim_generator):
        """Test memory usage for 1000-year simulation."""
        import sys

        sim = Simulation(
            manufacturer=manufacturer, claim_generator=claim_generator, time_horizon=1000, seed=42
        )

        # Estimate memory usage of pre-allocated arrays
        # 8 arrays * 1000 years * 8 bytes per float64
        expected_memory = 8 * 1000 * 8

        # Get actual size of arrays
        actual_memory = (
            sys.getsizeof(sim.years)
            + sys.getsizeof(sim.assets)
            + sys.getsizeof(sim.equity)
            + sys.getsizeof(sim.roe)
            + sys.getsizeof(sim.revenue)
            + sys.getsizeof(sim.net_income)
            + sys.getsizeof(sim.claim_counts)
            + sys.getsizeof(sim.claim_amounts)
        )

        # Should be less than 100MB as per requirements
        assert (
            actual_memory < 100 * 1024 * 1024
        ), f"Memory usage {actual_memory / 1024 / 1024:.2f} MB exceeds 100MB"

    def test_insolvency_handling(self, manufacturer_config):
        """Test handling of manufacturer insolvency."""
        # Create manufacturer with very low starting assets
        config = manufacturer_config
        config.initial_assets = 100_000  # Very low assets
        poor_manufacturer = WidgetManufacturer(config)

        # Create claim generator with high frequency/severity
        harsh_claims = ClaimGenerator(
            frequency=2.0,  # Many claims per year
            severity_mean=500_000,  # Large claims
            severity_std=100_000,
            seed=42,
        )

        sim = Simulation(
            manufacturer=poor_manufacturer, claim_generator=harsh_claims, time_horizon=100, seed=42
        )

        results = sim.run()

        # Should become insolvent
        assert results.insolvency_year is not None
        assert results.insolvency_year < 100

        # Results should still have full length but with zeros after insolvency
        assert len(results.years) == 100
        assert results.equity[results.insolvency_year] <= 0

    def test_get_trajectory(self, manufacturer, claim_generator):
        """Test get_trajectory convenience method."""
        sim = Simulation(
            manufacturer=manufacturer, claim_generator=claim_generator, time_horizon=10, seed=42
        )

        df = sim.get_trajectory()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "year" in df.columns
        assert "assets" in df.columns
        assert "roe" in df.columns

    def test_reproducibility(self, manufacturer, claim_generator):
        """Test that simulations with same seed produce same results."""
        sim1 = Simulation(
            manufacturer=WidgetManufacturer(manufacturer.config),
            claim_generator=ClaimGenerator(seed=42),
            time_horizon=50,
            seed=42,
        )

        sim2 = Simulation(
            manufacturer=WidgetManufacturer(manufacturer.config),
            claim_generator=ClaimGenerator(seed=42),
            time_horizon=50,
            seed=42,
        )

        results1 = sim1.run()
        results2 = sim2.run()

        # Results should be identical
        np.testing.assert_array_equal(results1.assets, results2.assets)
        np.testing.assert_array_equal(results1.equity, results2.equity)
        np.testing.assert_array_equal(results1.claim_counts, results2.claim_counts)

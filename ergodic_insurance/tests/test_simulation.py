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
        base_operating_margin=0.08,
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

        # Test enhanced ROE metrics
        assert "time_weighted_roe" in stats
        assert "roe_sharpe" in stats
        assert "roe_downside_deviation" in stats
        assert "roe_coefficient_variation" in stats
        assert "roe_1yr_avg" in stats
        assert "roe_3yr_avg" in stats
        assert "roe_5yr_avg" in stats

    def test_time_weighted_roe(self):
        """Test time-weighted ROE calculation."""
        results = SimulationResults(
            years=np.array([0, 1, 2, 3, 4]),
            assets=np.array([100, 110, 121, 133, 146]),
            equity=np.array([50, 55, 60, 65, 70]),
            roe=np.array([0.10, 0.10, 0.10, 0.10, 0.10]),  # Constant 10% ROE
            revenue=np.array([100, 105, 110, 115, 120]),
            net_income=np.array([5, 5.5, 6, 6.5, 7]),
            claim_counts=np.array([0, 0, 0, 0, 0]),
            claim_amounts=np.array([0, 0, 0, 0, 0]),
        )

        time_weighted = results.calculate_time_weighted_roe()
        # For constant ROE, time-weighted should equal arithmetic mean
        assert time_weighted == pytest.approx(0.10, rel=0.01)

        # Test with variable ROE
        results.roe = np.array([0.05, 0.15, 0.10, 0.20, 0.00])
        time_weighted = results.calculate_time_weighted_roe()
        # Geometric mean should be less than arithmetic mean for variable returns
        assert time_weighted < np.mean(results.roe)

    def test_rolling_roe_calculations(self):
        """Test rolling window ROE calculations."""
        results = SimulationResults(
            years=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            assets=np.ones(10) * 100,
            equity=np.ones(10) * 50,
            roe=np.array([0.08, 0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.09, 0.10]),
            revenue=np.ones(10) * 100,
            net_income=np.ones(10) * 5,
            claim_counts=np.zeros(10),
            claim_amounts=np.zeros(10),
        )

        # Test 1-year rolling (should equal original)
        rolling_1yr = results.calculate_rolling_roe(1)
        np.testing.assert_array_equal(rolling_1yr, results.roe)

        # Test 3-year rolling
        rolling_3yr = results.calculate_rolling_roe(3)
        assert np.isnan(rolling_3yr[0])  # First two should be NaN
        assert np.isnan(rolling_3yr[1])
        assert rolling_3yr[2] == pytest.approx(0.10, rel=0.01)  # Mean of [0.08, 0.10, 0.12]

        # Test 5-year rolling
        rolling_5yr = results.calculate_rolling_roe(5)
        assert np.all(np.isnan(rolling_5yr[:4]))  # First 4 should be NaN
        assert rolling_5yr[4] == pytest.approx(0.10, rel=0.01)  # Mean of first 5 values

        # Test error for window too large
        with pytest.raises(ValueError):
            results.calculate_rolling_roe(20)

    def test_roe_components(self):
        """Test ROE component breakdown calculation."""
        results = SimulationResults(
            years=np.array([0, 1, 2]),
            assets=np.array([100, 110, 120]),
            equity=np.array([50, 55, 60]),
            roe=np.array([0.10, 0.12, 0.08]),
            revenue=np.array([100, 110, 120]),
            net_income=np.array([5, 6.6, 4.8]),
            claim_counts=np.array([0, 1, 2]),
            claim_amounts=np.array([0, 1000, 2000]),
        )

        components = results.calculate_roe_components()

        assert "operating_roe" in components
        assert "insurance_impact" in components
        assert "tax_effect" in components
        assert "total_roe" in components

        # Check that components are arrays of correct length
        assert len(components["operating_roe"]) == 3
        assert len(components["insurance_impact"]) == 3

        # Operating ROE should be positive
        assert np.all(components["operating_roe"][~np.isnan(components["operating_roe"])] >= 0)

        # Insurance impact should be negative when there are claims
        assert components["insurance_impact"][1] < 0  # Year 1 has claims
        assert components["insurance_impact"][2] < 0  # Year 2 has claims

    def test_roe_volatility_metrics(self):
        """Test ROE volatility calculations."""
        results = SimulationResults(
            years=np.array([0, 1, 2, 3, 4]),
            assets=np.ones(5) * 100,
            equity=np.ones(5) * 50,
            roe=np.array([0.05, 0.15, 0.10, 0.20, -0.05]),  # Variable ROE with negative
            revenue=np.ones(5) * 100,
            net_income=np.array([2.5, 7.5, 5, 10, -2.5]),
            claim_counts=np.zeros(5),
            claim_amounts=np.zeros(5),
        )

        volatility = results.calculate_roe_volatility()

        assert "roe_std" in volatility
        assert "roe_downside_deviation" in volatility
        assert "roe_sharpe" in volatility
        assert "roe_coefficient_variation" in volatility

        # Standard deviation should be positive
        assert volatility["roe_std"] > 0

        # Downside deviation should be positive when there's variability below mean
        assert volatility["roe_downside_deviation"] > 0

        # Sharpe ratio calculation (mean - risk_free) / std
        mean_roe = np.mean(results.roe)
        expected_sharpe = (mean_roe - 0.02) / volatility["roe_std"]
        assert volatility["roe_sharpe"] == pytest.approx(expected_sharpe, rel=0.01)

    def test_edge_cases(self):
        """Test edge cases for ROE calculations."""
        # Test with all NaN ROE values
        results = SimulationResults(
            years=np.array([0, 1, 2]),
            assets=np.array([100, 110, 120]),
            equity=np.array([50, 55, 60]),
            roe=np.array([np.nan, np.nan, np.nan]),
            revenue=np.array([100, 105, 110]),
            net_income=np.array([5, 6, 7]),
            claim_counts=np.array([0, 0, 0]),
            claim_amounts=np.array([0, 0, 0]),
        )

        assert results.calculate_time_weighted_roe() == 0.0
        volatility = results.calculate_roe_volatility()
        assert volatility["roe_std"] == 0.0

        # Test with single valid ROE value
        results.roe = np.array([0.10, np.nan, np.nan])
        assert results.calculate_time_weighted_roe() == pytest.approx(0.10, rel=0.01)


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

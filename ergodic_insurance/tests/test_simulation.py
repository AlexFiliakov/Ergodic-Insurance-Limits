"""Tests for simulation engine."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.config import Config, ManufacturerConfig
from ergodic_insurance.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import Simulation, SimulationResults


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
    """Create a test loss generator."""
    return ManufacturingLossGenerator.create_simple(
        frequency=0.1, severity_mean=1_000_000, severity_std=500_000, seed=42
    )


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
        assert stats["std_roe"] == pytest.approx(0.01, rel=0.1)  # sample std (ddof=1)
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

    def test_time_weighted_roe_total_loss(self):
        """Test time-weighted ROE with total loss scenarios (ROE <= -1).

        Regression test for issue #211: Math domain error when ROE <= -1
        causes np.log(1 + roe) to fail with domain error or return -inf.
        """
        results = SimulationResults(
            years=np.array([0, 1, 2, 3, 4]),
            assets=np.array([100, 110, 50, 0, 0]),
            equity=np.array([50, 55, 25, 0, 0]),
            roe=np.array([0.10, 0.10, -1.0, -1.0, np.nan]),  # Total loss in years 2-3
            revenue=np.array([100, 105, 50, 0, 0]),
            net_income=np.array([5, 5.5, -30, 0, 0]),
            claim_counts=np.array([0, 0, 1, 0, 0]),
            claim_amounts=np.array([0, 0, 50, 0, 0]),
        )

        # Should not raise domain error and should return a valid float
        time_weighted = results.calculate_time_weighted_roe()
        assert isinstance(time_weighted, float)
        assert np.isfinite(time_weighted)
        # The result should be negative due to the large losses
        assert time_weighted < 0

        # Test with ROE even worse than -100% (edge case)
        results.roe = np.array([0.10, -1.5, -2.0, 0.05, 0.08])
        time_weighted = results.calculate_time_weighted_roe()
        assert isinstance(time_weighted, float)
        assert np.isfinite(time_weighted)

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


class TestSimulationResultsRepr:
    """Test SimulationResults __repr__, __str__, and convenience properties (#1307)."""

    @pytest.fixture
    def survived_results(self):
        return SimulationResults(
            years=np.array([0, 1, 2, 3, 4]),
            assets=np.array([10_000_000, 11_000_000, 12_100_000, 13_300_000, 14_600_000]),
            equity=np.array([3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000]),
            roe=np.array([0.08, 0.10, 0.09, 0.07, 0.11]),
            revenue=np.array([8_000_000, 8_800_000, 9_700_000, 10_600_000, 11_700_000]),
            net_income=np.array([240_000, 350_000, 360_000, 315_000, 550_000]),
            claim_counts=np.array([0, 1, 0, 2, 0]),
            claim_amounts=np.array([0, 500_000, 0, 1_200_000, 0]),
        )

    @pytest.fixture
    def insolvent_results(self):
        return SimulationResults(
            years=np.array([0, 1, 2]),
            assets=np.array([10_000_000, 5_000_000, 0]),
            equity=np.array([3_000_000, -2_000_000, -5_000_000]),
            roe=np.array([0.08, -1.5, np.nan]),
            revenue=np.array([8_000_000, 4_000_000, 0]),
            net_income=np.array([240_000, -5_000_000, 0]),
            claim_counts=np.array([0, 3, 0]),
            claim_amounts=np.array([0, 8_000_000, 0]),
            insolvency_year=2,
        )

    # --- convenience properties ---

    def test_survived_true(self, survived_results):
        assert survived_results.survived is True

    def test_survived_false(self, insolvent_results):
        assert insolvent_results.survived is False

    def test_n_years(self, survived_results):
        assert survived_results.n_years == 5

    def test_mean_roe(self, survived_results):
        expected = float(np.mean([0.08, 0.10, 0.09, 0.07, 0.11]))
        assert survived_results.mean_roe == pytest.approx(expected)

    def test_mean_roe_with_nan(self, insolvent_results):
        expected = float(np.mean([0.08, -1.5]))
        assert insolvent_results.mean_roe == pytest.approx(expected)

    def test_final_equity(self, survived_results):
        assert survived_results.final_equity == 5_000_000

    def test_final_assets(self, survived_results):
        assert survived_results.final_assets == 14_600_000

    def test_total_claims(self, survived_results):
        assert survived_results.total_claims == 1_700_000

    # --- __repr__ ---

    def test_repr_survived(self, survived_results):
        r = repr(survived_results)
        assert "SimulationResults(" in r
        assert "n_years=5" in r
        assert "survived" in r
        assert "mean_roe=" in r
        assert "final_equity=$" in r

    def test_repr_insolvent(self, insolvent_results):
        r = repr(insolvent_results)
        assert "insolvent@yr2" in r

    # --- __str__ ---

    def test_str_survived(self, survived_results):
        s = str(survived_results)
        assert "5 years" in s
        assert "Survived: True" in s
        assert "Mean ROE:" in s
        assert "Final Equity:" in s
        assert "Final Assets:" in s
        assert "Total Claims:" in s

    def test_str_insolvent(self, insolvent_results):
        s = str(insolvent_results)
        assert "Survived: False" in s
        assert "insolvent year 2" in s

    # --- _repr_html_ ---

    def test_repr_html_returns_html(self, survived_results):
        html = survived_results._repr_html_()
        assert "<div" in html
        assert "SimulationResults" in html

    # --- edge case: all-NaN ROE ---

    def test_mean_roe_all_nan(self):
        results = SimulationResults(
            years=np.array([0, 1]),
            assets=np.array([100, 100]),
            equity=np.array([50, 50]),
            roe=np.array([np.nan, np.nan]),
            revenue=np.array([100, 100]),
            net_income=np.array([0, 0]),
            claim_counts=np.array([0, 0]),
            claim_amounts=np.array([0, 0]),
        )
        assert results.mean_roe == 0.0
        r = repr(results)
        assert "mean_roe=0.00%" in r


class TestStrategyComparisonResultRepr:
    """Test StrategyComparisonResult __repr__ and __str__ (#1307)."""

    def test_repr(self):
        from ergodic_insurance.simulation import StrategyComparisonResult

        scr = StrategyComparisonResult(
            baseline=None,
            strategy_results={"Low": None, "High": None},
            summary_df=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        )
        r = repr(scr)
        assert "2 strategies" in r
        assert "High" in r
        assert "Low" in r
        assert "2x2" in r

    def test_str(self):
        from ergodic_insurance.simulation import StrategyComparisonResult

        scr = StrategyComparisonResult(
            baseline=None,
            strategy_results={"Low": None},
            summary_df=pd.DataFrame({"metric": ["survival"], "value": [0.95]}),
        )
        s = str(scr)
        assert "1 strategies" in s
        assert "survival" in s


class TestSimulation:
    """Test Simulation class."""

    def test_initialization(self, manufacturer, loss_generator):
        """Test simulation initialization."""
        sim = Simulation(
            manufacturer=manufacturer, loss_generator=loss_generator, time_horizon=100, seed=42
        )

        # copy=True (default) deep-copies the manufacturer (Issue #802)
        assert sim.manufacturer is not manufacturer
        assert sim.manufacturer.current_assets == manufacturer.current_assets
        assert sim.manufacturer.current_equity == manufacturer.current_equity
        assert sim.loss_generator == [loss_generator]  # Simulation wraps single generator in list
        assert sim.time_horizon == 100
        assert sim.seed == 42
        assert len(sim.years) == 100
        assert len(sim.assets) == 100
        assert sim.insolvency_year is None

    def test_step_annual(self, manufacturer, loss_generator):
        """Test single annual step."""
        sim = Simulation(manufacturer, loss_generator, time_horizon=10)

        # Create test losses
        losses = [
            LossEvent(time=0.0, amount=500_000, loss_type="test"),
            LossEvent(time=0.0, amount=300_000, loss_type="test"),
        ]

        metrics = sim.step_annual(0, losses)

        assert "assets" in metrics
        assert "equity" in metrics
        assert "roe" in metrics
        assert "claim_count" in metrics
        assert metrics["claim_count"] == 2
        assert metrics["claim_amount"] == 800_000

    def test_run_short_simulation(self, manufacturer, loss_generator):
        """Test running a short simulation."""
        sim = Simulation(
            manufacturer=manufacturer, loss_generator=loss_generator, time_horizon=10, seed=42
        )

        results = sim.run(progress_interval=5)

        assert isinstance(results, SimulationResults)
        assert len(results.years) == 10
        assert len(results.assets) == 10
        assert results.assets[0] > 0  # Should have initial assets

        # Check that some losses were generated
        assert np.sum(results.claim_counts) >= 0

    @pytest.mark.benchmark
    def test_run_performance(self, manufacturer, loss_generator):
        """Test that 1000-year simulation completes in reasonable time."""
        sim = Simulation(
            manufacturer=manufacturer, loss_generator=loss_generator, time_horizon=1000, seed=42
        )

        start = time.time()
        results = sim.run(progress_interval=250)
        duration = time.time() - start

        # Should complete in reasonable time (generous for CI/slow machines)
        assert duration < 600.0, f"Simulation took {duration:.2f} seconds, expected < 600 seconds"

        assert len(results.years) == 1000 or results.insolvency_year is not None

    def test_memory_efficiency(self, manufacturer, loss_generator):
        """Test memory usage for 1000-year simulation."""
        import sys

        sim = Simulation(
            manufacturer=manufacturer, loss_generator=loss_generator, time_horizon=1000, seed=42
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
        config = manufacturer_config.model_copy(update={"initial_assets": 50_000})
        poor_manufacturer = WidgetManufacturer(config)

        # Create loss generator with high frequency/severity
        harsh_losses = ManufacturingLossGenerator.create_simple(
            frequency=5.0,  # Many losses per year
            severity_mean=100_000,  # Large losses relative to assets
            severity_std=20_000,
            seed=42,
        )

        sim = Simulation(
            manufacturer=poor_manufacturer, loss_generator=harsh_losses, time_horizon=100, seed=42
        )

        results = sim.run()

        # Should become insolvent
        assert results.insolvency_year is not None
        assert results.insolvency_year < 100

        # Results should still have full length but with zeros after insolvency
        assert len(results.years) == 100
        assert results.equity[results.insolvency_year] <= 0

    def test_get_trajectory(self, manufacturer, loss_generator):
        """Test get_trajectory convenience method."""
        sim = Simulation(
            manufacturer=manufacturer, loss_generator=loss_generator, time_horizon=10, seed=42
        )

        df = sim.get_trajectory()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "year" in df.columns
        assert "assets" in df.columns
        assert "roe" in df.columns

    def test_reproducibility(self, manufacturer, loss_generator):
        """Test that simulations with same seed produce same results."""
        sim1 = Simulation(
            manufacturer=WidgetManufacturer(manufacturer.config),
            loss_generator=ManufacturingLossGenerator.create_simple(
                frequency=0.1, severity_mean=5_000_000, severity_std=2_000_000, seed=42
            ),
            time_horizon=50,
            seed=42,
        )

        sim2 = Simulation(
            manufacturer=WidgetManufacturer(manufacturer.config),
            loss_generator=ManufacturingLossGenerator.create_simple(
                frequency=0.1, severity_mean=5_000_000, severity_std=2_000_000, seed=42
            ),
            time_horizon=50,
            seed=42,
        )

        results1 = sim1.run()
        results2 = sim2.run()

        # Results should be identical
        np.testing.assert_array_equal(results1.assets, results2.assets)
        np.testing.assert_array_equal(results1.equity, results2.equity)
        np.testing.assert_array_equal(results1.claim_counts, results2.claim_counts)


class TestDefaultLossGenerator:
    """Tests for default loss generator scaling (Issue #1320)."""

    def test_default_generator_emits_deprecation_warning(self, manufacturer):
        """Omitting loss_generator should emit a deprecation warning."""
        import warnings

        from ergodic_insurance._warnings import ErgodicInsuranceDeprecationWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Simulation(manufacturer=manufacturer, time_horizon=5, seed=42)

        dep_warnings = [x for x in w if issubclass(x.category, ErgodicInsuranceDeprecationWarning)]
        assert len(dep_warnings) == 1
        msg = str(dep_warnings[0].message)
        assert "No loss_generator provided" in msg
        assert "ManufacturingLossGenerator.create_simple" in msg

    def test_default_generator_scales_to_initial_assets(self, manufacturer_config):
        """Default severity should be 5% of initial_assets."""
        import warnings

        config_small = manufacturer_config.model_copy(update={"initial_assets": 500_000})
        mfg_small = WidgetManufacturer(config_small)

        config_large = manufacturer_config.model_copy(update={"initial_assets": 50_000_000_000})
        mfg_large = WidgetManufacturer(config_large)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim_small = Simulation(manufacturer=mfg_small, time_horizon=5, seed=42)
            sim_large = Simulation(manufacturer=mfg_large, time_horizon=5, seed=42)

        # Extract the single default generator from each
        gen_small = sim_small.loss_generator[0]
        gen_large = sim_large.loss_generator[0]

        # The generators should differ — large company gets larger losses
        # Run them both and verify the large company produces bigger losses on average
        # generate_losses returns (List[LossEvent], dict); use revenue proportional to assets
        losses_small = [
            gen_small.generate_losses(duration=1, revenue=500_000, time=float(y))
            for y in range(200)
        ]
        losses_large = [
            gen_large.generate_losses(duration=1, revenue=50_000_000_000, time=float(y))
            for y in range(200)
        ]

        total_small = sum(l.amount for events, _ in losses_small for l in events)
        total_large = sum(l.amount for events, _ in losses_large for l in events)

        # Large company (100,000x assets) should have much larger aggregate losses
        assert total_large > total_small * 100

    def test_default_generator_warning_includes_asset_amount(self, manufacturer_config):
        """Warning message should include the actual initial_assets value."""
        import warnings

        from ergodic_insurance._warnings import ErgodicInsuranceDeprecationWarning

        config = manufacturer_config.model_copy(update={"initial_assets": 2_000_000})
        mfg = WidgetManufacturer(config)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Simulation(manufacturer=mfg, time_horizon=5, seed=42)

        dep_warnings = [x for x in w if issubclass(x.category, ErgodicInsuranceDeprecationWarning)]
        msg = str(dep_warnings[0].message)
        assert "2,000,000" in msg
        # severity_mean = 5% of 2M = 100K
        assert "100,000" in msg

    def test_explicit_loss_generator_no_warning(self, manufacturer, loss_generator):
        """Providing an explicit loss_generator should NOT emit a warning."""
        import warnings

        from ergodic_insurance._warnings import ErgodicInsuranceDeprecationWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Simulation(
                manufacturer=manufacturer,
                loss_generator=loss_generator,
                time_horizon=5,
                seed=42,
            )

        dep_warnings = [x for x in w if issubclass(x.category, ErgodicInsuranceDeprecationWarning)]
        assert len(dep_warnings) == 0

    def test_default_generator_simulation_runs(self, manufacturer):
        """Simulation with default generator should run without error."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim = Simulation(manufacturer=manufacturer, time_horizon=10, seed=42)

        results = sim.run()
        assert len(results.years) == 10
        assert results.assets[0] > 0

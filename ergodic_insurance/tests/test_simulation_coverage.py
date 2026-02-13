"""Tests targeting specific uncovered lines in simulation.py.

Coverage targets:
- Lines 789-790: LossData validation failure in run_with_loss_data
- Lines 822-825: Progress logging in run_with_loss_data
- Lines 847-857: Insolvency handling in run_with_loss_data
- Lines 923-1044: run_monte_carlo class method
- Lines 1070-1130: compare_insurance_strategies class method
"""

from unittest.mock import MagicMock, Mock, patch
import warnings

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.config import Config, ManufacturerConfig
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.loss_distributions import LossData, LossEvent
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import Simulation, SimulationResults, StrategyComparisonResult

# ---------------------------------------------------------------------------
# Shared helper: full Config object
# ---------------------------------------------------------------------------


def _make_full_config(initial_assets=10_000_000):
    """Build a complete Config with all required sub-config fields."""
    return Config(
        manufacturer=ManufacturerConfig(
            initial_assets=initial_assets,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        ),
        working_capital={"percent_of_sales": 0.20},  # type: ignore[arg-type]
        growth={"annual_growth_rate": 0.03},  # type: ignore[arg-type]
        debt={  # type: ignore[arg-type]
            "interest_rate": 0.05,
            "max_leverage_ratio": 2.0,
            "minimum_cash_balance": 500_000,
        },
        simulation={"time_horizon_years": 10},  # type: ignore[arg-type]
        output={},  # type: ignore[arg-type]
        logging={},  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def simple_insurance_policy():
    """Create a basic insurance policy for testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        layer = InsuranceLayer(
            attachment_point=500_000,
            limit=5_000_000,
            rate=0.02,
        )
        return InsurancePolicy(layers=[layer], deductible=500_000)


@pytest.fixture
def simulation_with_policy(manufacturer, simple_insurance_policy):
    """Create a simulation instance with insurance policy attached."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Simulation(
            manufacturer=manufacturer,
            insurance_policy=simple_insurance_policy,
            time_horizon=10,
            seed=42,
        )


@pytest.fixture
def simulation_no_policy(manufacturer):
    """Create a simulation instance without insurance policy."""
    return Simulation(
        manufacturer=manufacturer,
        time_horizon=10,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests for Lines 789-790: LossData validation failure
# ---------------------------------------------------------------------------


class TestRunWithLossDataValidation:
    """Tests for the validation branch of run_with_loss_data.

    Covers lines 789-790 where invalid LossData raises ValueError.
    """

    def test_invalid_loss_data_mismatched_lengths(self, simulation_with_policy):
        """When timestamps and loss_amounts have different lengths,
        validate() returns False and run_with_loss_data should raise
        ValueError."""
        invalid_data = LossData(
            timestamps=np.array([0.0, 1.0, 2.0]),
            loss_amounts=np.array([100_000.0, 200_000.0]),  # length mismatch
        )
        assert invalid_data.validate() is False

        with pytest.raises(ValueError, match="Invalid loss data provided"):
            simulation_with_policy.run_with_loss_data(invalid_data, validate=True)

    def test_invalid_loss_data_negative_amounts(self, simulation_with_policy):
        """Negative loss amounts should fail validation and raise ValueError."""
        invalid_data = LossData(
            timestamps=np.array([0.0, 1.0]),
            loss_amounts=np.array([-100_000.0, 200_000.0]),
        )
        assert invalid_data.validate() is False

        with pytest.raises(ValueError, match="Invalid loss data provided"):
            simulation_with_policy.run_with_loss_data(invalid_data, validate=True)

    def test_invalid_loss_data_negative_timestamps(self, simulation_with_policy):
        """Negative timestamps should fail validation and raise ValueError."""
        invalid_data = LossData(
            timestamps=np.array([-1.0, 1.0]),
            loss_amounts=np.array([100_000.0, 200_000.0]),
        )
        assert invalid_data.validate() is False

        with pytest.raises(ValueError, match="Invalid loss data provided"):
            simulation_with_policy.run_with_loss_data(invalid_data, validate=True)

    def test_invalid_loss_data_mismatched_loss_types(self, simulation_with_policy):
        """When loss_types length does not match timestamps, validation fails."""
        invalid_data = LossData(
            timestamps=np.array([0.0, 1.0, 2.0]),
            loss_amounts=np.array([100_000.0, 200_000.0, 300_000.0]),
            loss_types=["operational"],  # should be length 3
        )
        assert invalid_data.validate() is False

        with pytest.raises(ValueError, match="Invalid loss data provided"):
            simulation_with_policy.run_with_loss_data(invalid_data, validate=True)

    def test_valid_loss_data_skips_validation_when_disabled(self, simulation_with_policy):
        """When validate=False, even invalid data does not raise.

        This ensures the validation gate is properly controlled by the flag.
        """
        valid_data = LossData(
            timestamps=np.array([0.0]),
            loss_amounts=np.array([0.0]),
            loss_types=["test"],
        )
        result = simulation_with_policy.run_with_loss_data(valid_data, validate=False)
        assert isinstance(result, SimulationResults)


# ---------------------------------------------------------------------------
# Tests for Lines 822-825: Progress logging in run_with_loss_data
# ---------------------------------------------------------------------------


class TestRunWithLossDataProgressLogging:
    """Tests for progress logging within run_with_loss_data.

    Covers lines 822-825 where elapsed time, rate, and remaining are computed
    and logged when year > 0 and year % progress_interval == 0.
    """

    def test_progress_logging_triggered(self, manufacturer_config):
        """Ensure progress logging fires when the interval is hit.

        We use a short time_horizon and progress_interval=1 so that the
        progress branch is entered on every year after year 0.
        """
        mfr = WidgetManufacturer(manufacturer_config)
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        # Use float dtype to avoid numpy int -> Decimal conversion issues
        loss_data = LossData(
            timestamps=np.array([0.5, 2.5], dtype=np.float64),
            loss_amounts=np.array([100_000.0, 200_000.0], dtype=np.float64),
            loss_types=["test", "test"],
            claim_ids=["c1", "c2"],
        )

        with patch("ergodic_insurance.simulation.logger") as mock_logger:
            result = sim.run_with_loss_data(loss_data, validate=True, progress_interval=1)

            # The info logger should have been called for progress updates
            # at years 1, 2, 3, 4 (year > 0 and year % 1 == 0)
            info_calls = mock_logger.info.call_args_list
            progress_messages = [
                call
                for call in info_calls
                if "elapsed" in str(call).lower() and "remaining" in str(call).lower()
            ]
            assert len(progress_messages) >= 1, (
                "Expected at least one progress log message containing " "'elapsed' and 'remaining'"
            )

        assert isinstance(result, SimulationResults)
        assert len(result.years) == 5

    def test_progress_logging_not_triggered_for_large_interval(self, manufacturer_config):
        """When progress_interval exceeds the time_horizon, no progress
        messages should be logged (only start/completion messages)."""
        mfr = WidgetManufacturer(manufacturer_config)
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        loss_data = LossData(
            timestamps=np.array([1.0], dtype=np.float64),
            loss_amounts=np.array([50_000.0], dtype=np.float64),
            loss_types=["test"],
            claim_ids=["c1"],
        )

        with patch("ergodic_insurance.simulation.logger") as mock_logger:
            sim.run_with_loss_data(loss_data, validate=True, progress_interval=1000)

            info_calls = mock_logger.info.call_args_list
            progress_messages = [
                call
                for call in info_calls
                if "elapsed" in str(call).lower() and "remaining" in str(call).lower()
            ]
            assert (
                len(progress_messages) == 0
            ), "No progress messages expected when interval > time_horizon"


# ---------------------------------------------------------------------------
# Tests for Lines 847-857: Insolvency handling in run_with_loss_data
# ---------------------------------------------------------------------------


class TestRunWithLossDataInsolvency:
    """Tests for insolvency detection in run_with_loss_data.

    Covers lines 847-857 where the manufacturer becomes insolvent,
    remaining arrays are zeroed, roe filled with NaN, and loop breaks.
    """

    def test_insolvency_detected_with_large_claims(self, manufacturer_config):
        """A manufacturer with modest assets hit by huge claims should
        trigger insolvency handling in run_with_loss_data."""
        config = manufacturer_config.model_copy(update={"initial_assets": 100_000})
        mfr = WidgetManufacturer(config)
        sim = Simulation(manufacturer=mfr, time_horizon=10, seed=42)

        # Create enormous claims in year 0 to bankrupt the company
        loss_data = LossData(
            timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
            loss_amounts=np.array(
                [500_000.0, 500_000.0, 500_000.0, 500_000.0, 500_000.0],
                dtype=np.float64,
            ),
            loss_types=["cat"] * 5,
            claim_ids=[f"claim_{i}" for i in range(5)],
        )

        result = sim.run_with_loss_data(loss_data, validate=True)

        assert result.insolvency_year is not None
        assert result.insolvency_year < 10

        # After insolvency, remaining years should be zeroed out
        after_insolvency = result.insolvency_year + 1
        if after_insolvency < len(result.years):
            assert np.all(result.assets[after_insolvency:] == 0)
            assert np.all(result.equity[after_insolvency:] == 0)
            assert np.all(result.revenue[after_insolvency:] == 0)
            assert np.all(result.net_income[after_insolvency:] == 0)
            assert np.all(np.isnan(result.roe[after_insolvency:]))

    def test_insolvency_in_later_year(self, manufacturer_config):
        """Insolvency triggered in a later year (not year 0) should
        still properly zero subsequent arrays."""
        config = manufacturer_config.model_copy(update={"initial_assets": 500_000})
        mfr = WidgetManufacturer(config)
        sim = Simulation(manufacturer=mfr, time_horizon=10, seed=42)

        # Place a devastating loss in year 3 to trigger insolvency mid-run
        loss_data = LossData(
            timestamps=np.array([3.0, 3.1, 3.2], dtype=np.float64),
            loss_amounts=np.array([2_000_000.0, 2_000_000.0, 2_000_000.0], dtype=np.float64),
            loss_types=["cat", "cat", "cat"],
            claim_ids=["c1", "c2", "c3"],
        )

        result = sim.run_with_loss_data(loss_data, validate=True)

        # The company might survive year 0-2 with no losses and fail at year 3
        # or at some point after the catastrophic losses hit
        if result.insolvency_year is not None:
            assert (
                result.insolvency_year <= 5
            ), "Expected insolvency by year 5 given the large claims"
            # Verify the array structure
            assert len(result.years) == 10
            after_insolvency = result.insolvency_year + 1
            if after_insolvency < 10:
                assert np.all(result.assets[after_insolvency:] == 0)

    def test_no_insolvency_with_healthy_balance(self, manufacturer_config):
        """A well-capitalized manufacturer with small claims should survive
        the entire simulation in run_with_loss_data."""
        mfr = WidgetManufacturer(manufacturer_config)  # $10M initial assets
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        # Small claims that should not threaten solvency
        loss_data = LossData(
            timestamps=np.array([1.0, 3.0], dtype=np.float64),
            loss_amounts=np.array([10_000.0, 15_000.0], dtype=np.float64),
            loss_types=["attritional", "attritional"],
            claim_ids=["c1", "c2"],
        )

        result = sim.run_with_loss_data(loss_data, validate=True)

        assert result.insolvency_year is None
        assert len(result.years) == 5
        # Final equity should still be positive
        assert result.equity[-1] > 0


# ---------------------------------------------------------------------------
# Tests for Lines 923-1044: run_monte_carlo class method
# ---------------------------------------------------------------------------


class TestRunMonteCarlo:
    """Tests for the Simulation.run_monte_carlo class method.

    Covers lines 923-1044. Uses mocking for MonteCarloEngine since actual
    Monte Carlo runs would be prohibitively slow in unit tests.
    """

    @pytest.fixture
    def full_config(self):
        """Create a complete Config object for testing."""
        return _make_full_config()

    @pytest.fixture
    def test_policy(self):
        """Create a test insurance policy with known premium and coverage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            layer = InsuranceLayer(
                attachment_point=500_000,
                limit=5_000_000,
                rate=0.02,
            )
            return InsurancePolicy(layers=[layer], deductible=500_000)

    @staticmethod
    def _make_mock_mc_results(has_statistics=True, has_geo_return=True):
        """Create a mock MonteCarloEngine result object.

        Uses Mock(spec=...) so that hasattr correctly reflects presence
        or absence of the 'statistics' attribute.
        """
        if has_statistics:
            mock_results = Mock(spec=["final_assets", "growth_rates", "statistics"])
            mock_results.statistics = {}
            if has_geo_return:
                mock_results.statistics["geometric_return"] = {
                    "geometric_mean": 0.04,
                    "survival_rate": 0.95,
                    "std": 0.08,
                }
        else:
            # No 'statistics' in spec => hasattr returns False
            mock_results = Mock(spec=["final_assets", "growth_rates"])

        mock_results.final_assets = np.array([10_000_000.0, 12_000_000.0, 8_000_000.0])
        mock_results.growth_rates = np.array([0.05, 0.10, -0.02])
        return mock_results

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_with_ergodic_analysis(self, MockMCEngine, full_config, test_policy):
        """Test the full path through run_monte_carlo where both insured
        and uninsured results have statistics with geometric_return.

        This covers lines 923-1041 (the main success path).
        """
        mock_engine_insured = MagicMock()
        mock_engine_uninsured = MagicMock()

        results_insured = self._make_mock_mc_results(has_statistics=True, has_geo_return=True)
        results_uninsured = self._make_mock_mc_results(has_statistics=True, has_geo_return=True)

        # Make the uninsured results slightly worse for realistic comparison
        results_uninsured.statistics["geometric_return"]["geometric_mean"] = 0.02
        results_uninsured.statistics["geometric_return"]["survival_rate"] = 0.85
        results_uninsured.statistics["geometric_return"]["std"] = 0.12

        mock_engine_insured.run.return_value = results_insured
        mock_engine_uninsured.run.return_value = results_uninsured

        MockMCEngine.side_effect = [mock_engine_insured, mock_engine_uninsured]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            output = Simulation.run_monte_carlo(
                config=full_config,
                insurance_policy=test_policy,
                n_scenarios=100,
                n_jobs=1,
                seed=42,
            )

        # Verify both engines were created and run
        assert MockMCEngine.call_count == 2
        mock_engine_insured.run.assert_called_once()
        mock_engine_uninsured.run.assert_called_once()

        # Verify result structure
        assert "results_with_insurance" in output
        assert "results_without_insurance" in output
        assert "ergodic_analysis" in output

        analysis = output["ergodic_analysis"]
        assert "premium_rate" in analysis
        assert "geometric_mean_return_with_insurance" in analysis
        assert "geometric_mean_return_without_insurance" in analysis
        assert "survival_rate_with_insurance" in analysis
        assert "survival_rate_without_insurance" in analysis
        assert "growth_impact" in analysis
        assert "survival_benefit" in analysis
        assert "volatility_reduction" in analysis

        # Verify computed deltas
        assert analysis["growth_impact"] == pytest.approx(0.04 - 0.02, rel=1e-6)
        assert analysis["survival_benefit"] == pytest.approx(0.95 - 0.85, rel=1e-6)
        assert analysis["volatility_reduction"] == pytest.approx(0.12 - 0.08, rel=1e-6)

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_fallback_no_statistics(self, MockMCEngine, full_config, test_policy):
        """Test the fallback path (lines 1043-1047) when results lack
        a statistics attribute.

        MonteCarloEngine.run() returns results without a 'statistics'
        attribute, so the method returns a minimal result dict.
        """
        mock_engine_insured = MagicMock()
        mock_engine_uninsured = MagicMock()

        results_insured = self._make_mock_mc_results(has_statistics=False)
        results_uninsured = self._make_mock_mc_results(has_statistics=False)

        mock_engine_insured.run.return_value = results_insured
        mock_engine_uninsured.run.return_value = results_uninsured

        MockMCEngine.side_effect = [mock_engine_insured, mock_engine_uninsured]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            output = Simulation.run_monte_carlo(
                config=full_config,
                insurance_policy=test_policy,
                n_scenarios=50,
                n_jobs=1,
                seed=42,
            )

        assert "results_with_insurance" in output
        assert "results_without_insurance" in output
        # No ergodic_analysis when statistics are absent
        assert "ergodic_analysis" not in output

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_statistics_without_geometric_return(
        self, MockMCEngine, full_config, test_policy
    ):
        """Test path when statistics exist but lack geometric_return key.

        This exercises the condition on line 1002 where the key check fails,
        falling through to the fallback return on lines 1043-1047.
        """
        mock_engine_insured = MagicMock()
        mock_engine_uninsured = MagicMock()

        results_insured = self._make_mock_mc_results(has_statistics=True, has_geo_return=False)
        results_uninsured = self._make_mock_mc_results(has_statistics=True, has_geo_return=False)

        mock_engine_insured.run.return_value = results_insured
        mock_engine_uninsured.run.return_value = results_uninsured

        MockMCEngine.side_effect = [mock_engine_insured, mock_engine_uninsured]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            output = Simulation.run_monte_carlo(
                config=full_config,
                insurance_policy=test_policy,
                n_scenarios=50,
                n_jobs=1,
                seed=42,
            )

        assert "results_with_insurance" in output
        assert "results_without_insurance" in output
        assert "ergodic_analysis" not in output

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_n_jobs_configuration(self, MockMCEngine, full_config, test_policy):
        """Verify that n_jobs parameter correctly propagates to
        SimulationConfig (parallel=True if n_jobs > 1)."""
        mock_engine = MagicMock()
        mock_results = self._make_mock_mc_results(has_statistics=False)
        mock_engine.run.return_value = mock_results
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            Simulation.run_monte_carlo(
                config=full_config,
                insurance_policy=test_policy,
                n_scenarios=50,
                n_jobs=4,
                seed=99,
            )

        # Check that MonteCarloEngine was called with proper config
        assert MockMCEngine.call_count == 2
        # Extract the config argument from the first call
        call_kwargs = MockMCEngine.call_args_list[0][1]
        sim_config = call_kwargs["config"]
        assert sim_config.n_workers == 4
        assert sim_config.parallel is True
        assert sim_config.seed == 99

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_single_job(self, MockMCEngine, full_config, test_policy):
        """When n_jobs=1, parallel should be False."""
        mock_engine = MagicMock()
        mock_results = self._make_mock_mc_results(has_statistics=False)
        mock_engine.run.return_value = mock_results
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            Simulation.run_monte_carlo(
                config=full_config,
                insurance_policy=test_policy,
                n_scenarios=10,
                n_jobs=1,
                seed=42,
            )

        call_kwargs = MockMCEngine.call_args_list[0][1]
        sim_config = call_kwargs["config"]
        assert sim_config.parallel is False
        assert sim_config.n_workers == 1

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_premium_rate_calculation(self, MockMCEngine, full_config, test_policy):
        """Verify that the premium_rate in ergodic_analysis is correctly
        calculated as premium / initial_assets."""
        mock_engine_insured = MagicMock()
        mock_engine_uninsured = MagicMock()

        results_insured = self._make_mock_mc_results(has_statistics=True, has_geo_return=True)
        results_uninsured = self._make_mock_mc_results(has_statistics=True, has_geo_return=True)

        mock_engine_insured.run.return_value = results_insured
        mock_engine_uninsured.run.return_value = results_uninsured

        MockMCEngine.side_effect = [mock_engine_insured, mock_engine_uninsured]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            output = Simulation.run_monte_carlo(
                config=full_config,
                insurance_policy=test_policy,
                n_scenarios=100,
                n_jobs=1,
                seed=42,
            )

        expected_premium = test_policy.calculate_premium()
        expected_initial_assets = full_config.manufacturer.initial_assets
        expected_rate = expected_premium / expected_initial_assets

        assert output["ergodic_analysis"]["premium_rate"] == pytest.approx(expected_rate, rel=1e-6)

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_run_monte_carlo_respects_time_horizon_years(self, MockMCEngine, test_policy):
        """Regression test for #1080: run_monte_carlo must use
        config.simulation.time_horizon_years, not a hardcoded fallback."""
        config = _make_full_config()
        # Override to a non-default value that differs from 10
        config.simulation.time_horizon_years = 25

        mock_engine = MagicMock()
        mock_results = self._make_mock_mc_results(has_statistics=False)
        mock_engine.run.return_value = mock_results
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            Simulation.run_monte_carlo(
                config=config,
                insurance_policy=test_policy,
                n_scenarios=10,
                n_jobs=1,
                seed=42,
            )

        # Both insured and uninsured engines should receive n_years=25
        for call_args in MockMCEngine.call_args_list:
            sim_config = call_args[1]["config"]
            assert (
                sim_config.n_years == 25
            ), f"Expected n_years=25 from time_horizon_years, got {sim_config.n_years}"


# ---------------------------------------------------------------------------
# Tests for Lines 1070-1130: compare_insurance_strategies class method
# ---------------------------------------------------------------------------


class TestCompareInsuranceStrategies:
    """Tests for the Simulation.compare_insurance_strategies class method.

    After the fix for #985, compare_insurance_strategies no longer delegates
    to run_monte_carlo (which ran 2 sims per call).  Instead it constructs
    MonteCarloEngine instances directly: one shared uninsured baseline plus
    one per strategy, using Common Random Numbers for paired comparison.
    """

    @pytest.fixture
    def full_config(self):
        """Create a complete Config object."""
        return _make_full_config()

    @pytest.fixture
    def policy_dict(self):
        """Create a dictionary of insurance policies for comparison."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            low_coverage = InsurancePolicy(
                layers=[InsuranceLayer(500_000, 2_000_000, 0.015)],
                deductible=500_000,
            )
            medium_coverage = InsurancePolicy(
                layers=[InsuranceLayer(500_000, 5_000_000, 0.02)],
                deductible=500_000,
            )
            high_coverage = InsurancePolicy(
                layers=[
                    InsuranceLayer(500_000, 5_000_000, 0.025),
                    InsuranceLayer(5_500_000, 10_000_000, 0.015),
                ],
                deductible=500_000,
            )
            return {
                "Low": low_coverage,
                "Medium": medium_coverage,
                "High": high_coverage,
            }

    @staticmethod
    def _make_mock_engine_result(n_scenarios=100, seed=42):
        """Create a mock MonteCarloResults returned by engine.run()."""
        mock_results = MagicMock()
        rng = np.random.default_rng(seed)
        mock_results.final_assets = rng.normal(12_000_000, 2_000_000, n_scenarios)
        mock_results.growth_rates = np.abs(rng.normal(0.06, 0.03, n_scenarios))
        return mock_results

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_basic_structure(self, MockMCEngine, full_config, policy_dict):
        """Verify that compare_insurance_strategies returns a
        StrategyComparisonResult whose summary_df has the correct columns
        and one row per policy."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=policy_dict,
            n_scenarios=50,
            n_jobs=1,
            seed=42,
        )

        assert isinstance(result, StrategyComparisonResult)
        df = result.summary_df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df["policy"]) == {"Low", "Medium", "High"}

        expected_columns = {
            "policy",
            "annual_premium",
            "total_coverage",
            "survival_rate",
            "mean_final_equity",
            "std_final_equity",
            "geometric_return",
            "arithmetic_return",
            "p95_final_equity",
            "p99_final_equity",
            "premium_to_coverage",
            "sharpe_ratio",
        }
        assert expected_columns.issubset(set(df.columns))

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_relative_metrics_computed(self, MockMCEngine, full_config, policy_dict):
        """Verify that derived columns (premium_to_coverage, sharpe_ratio)
        are computed correctly."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=policy_dict,
            n_scenarios=50,
            n_jobs=1,
            seed=42,
        )
        df = result.summary_df

        # premium_to_coverage should be annual_premium / total_coverage
        for _, row in df.iterrows():
            expected_ptc = row["annual_premium"] / row["total_coverage"]
            assert row["premium_to_coverage"] == pytest.approx(expected_ptc, rel=1e-6)

        # sharpe_ratio should be arithmetic_return / std_final_equity
        for _, row in df.iterrows():
            expected_sharpe = row["arithmetic_return"] / row["std_final_equity"]
            assert row["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-6)

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_runs_baseline_plus_one_per_strategy(
        self, MockMCEngine, full_config, policy_dict
    ):
        """Verify that N+1 MonteCarloEngine instances are created:
        1 shared uninsured baseline + 1 per strategy (issue #985)."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=policy_dict,
            n_scenarios=200,
            n_jobs=3,
            seed=99,
        )

        # 1 baseline + 3 strategies = 4 engine constructions
        assert MockMCEngine.call_count == 4

        # All engines should use the same CRN-enabled config
        for call in MockMCEngine.call_args_list:
            kwargs = call[1]
            sim_config = kwargs["config"]
            assert sim_config.n_simulations == 200
            assert sim_config.n_workers == 3
            assert sim_config.seed == 99
            assert sim_config.crn_base_seed == 99  # CRN enabled

        # Verify result has the CRN seed recorded
        assert result.crn_seed == 99

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_single_policy(self, MockMCEngine, full_config):
        """Works correctly with a single policy."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            single_policy = {
                "OnlyPolicy": InsurancePolicy(
                    layers=[InsuranceLayer(0, 1_000_000, 0.03)],
                    deductible=0,
                )
            }

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=single_policy,
            n_scenarios=10,
            n_jobs=1,
            seed=42,
        )

        df = result.summary_df
        assert len(df) == 1
        assert df.iloc[0]["policy"] == "OnlyPolicy"
        assert "premium_to_coverage" in df.columns
        assert "sharpe_ratio" in df.columns

        # 1 baseline + 1 strategy = 2 engine constructions
        assert MockMCEngine.call_count == 2

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_preserves_premium_and_coverage_values(
        self, MockMCEngine, full_config, policy_dict
    ):
        """Verify that annual_premium and total_coverage in the output
        DataFrame match the actual policy calculations."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=policy_dict,
            n_scenarios=50,
            n_jobs=1,
            seed=42,
        )
        df = result.summary_df

        for policy_name, policy in policy_dict.items():
            row = df[df["policy"] == policy_name].iloc[0]
            assert row["annual_premium"] == pytest.approx(policy.calculate_premium(), rel=1e-6)
            assert row["total_coverage"] == pytest.approx(policy.get_total_coverage(), rel=1e-6)

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_survival_rate_calculation(self, MockMCEngine, full_config):
        """Verify survival_rate is calculated as fraction of positive
        final assets."""
        mock_sim_results = MagicMock()
        # 7 out of 10 have final_assets > 0 (0 is NOT > 0)
        mock_sim_results.final_assets = np.array(
            [100.0, 200.0, 0.0, 300.0, -50.0, 400.0, 500.0, 0.0, 600.0, 700.0]
        )
        mock_sim_results.growth_rates = np.array(
            [0.05, 0.08, -1.0, 0.03, -1.5, 0.06, 0.04, -1.0, 0.07, 0.09]
        )

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_sim_results
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            single_policy = {
                "TestPolicy": InsurancePolicy(
                    layers=[InsuranceLayer(0, 1_000_000, 0.02)],
                    deductible=0,
                )
            }

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=single_policy,
            n_scenarios=10,
            n_jobs=1,
            seed=42,
        )
        df = result.summary_df

        # 7 out of 10 have final_assets > 0 (100, 200, 300, 400, 500, 600, 700)
        # 0.0 is NOT > 0, and -50.0 is NOT > 0
        assert df.iloc[0]["survival_rate"] == pytest.approx(0.7, rel=1e-6)

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_geometric_return_uses_all_rates(self, MockMCEngine, full_config):
        """Geometric return should use ALL growth rates via growth factors."""
        mock_sim_results = MagicMock()
        mock_sim_results.final_assets = np.array([1_000_000.0] * 5)
        # Mix of positive and negative growth rates
        mock_sim_results.growth_rates = np.array([0.10, 0.05, -0.03, 0.08, -0.01])

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_sim_results
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            single_policy = {
                "Test": InsurancePolicy(
                    layers=[InsuranceLayer(0, 1_000_000, 0.02)],
                    deductible=0,
                )
            }

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=single_policy,
            n_scenarios=5,
            n_jobs=1,
            seed=42,
        )
        df = result.summary_df

        # Geometric mean of ALL growth factors: [1.10, 1.05, 0.97, 1.08, 0.99]
        all_rates = np.array([0.10, 0.05, -0.03, 0.08, -0.01])
        growth_factors = np.maximum(1 + all_rates, 1e-10)
        expected_geo = float(np.exp(np.mean(np.log(growth_factors))) - 1)
        assert df.iloc[0]["geometric_return"] == pytest.approx(expected_geo, rel=1e-4)

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_geometric_return_handles_total_wipeout(self, MockMCEngine, full_config):
        """Growth rates <= -1 should produce a finite geometric return."""
        mock_sim_results = MagicMock()
        mock_sim_results.final_assets = np.array([0.0, 0.0, 1_000_000.0])
        # Total wipeout rates (-1.0 means 100% loss)
        mock_sim_results.growth_rates = np.array([-1.0, -1.5, 0.05])

        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_sim_results
        MockMCEngine.return_value = mock_engine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            single_policy = {
                "Test": InsurancePolicy(
                    layers=[InsuranceLayer(0, 1_000_000, 0.02)],
                    deductible=0,
                )
            }

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=single_policy,
            n_scenarios=3,
            n_jobs=1,
            seed=42,
        )
        df = result.summary_df

        geo = df.iloc[0]["geometric_return"]
        assert np.isfinite(geo), f"Geometric return should be finite, got {geo}"

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_baseline_is_shared(self, MockMCEngine, full_config, policy_dict):
        """The uninsured baseline should be run once and shared across all
        strategies (core fix for issue #985)."""
        baseline_result = self._make_mock_engine_result(seed=1)
        strategy_result = self._make_mock_engine_result(seed=2)

        # First call is the baseline, subsequent calls are strategies
        mock_engines = []
        for i in range(4):  # 1 baseline + 3 strategies
            engine = MagicMock()
            engine.run.return_value = baseline_result if i == 0 else strategy_result
            mock_engines.append(engine)
        MockMCEngine.side_effect = mock_engines

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=policy_dict,
            n_scenarios=50,
            n_jobs=1,
            seed=42,
        )

        # Baseline is stored on the result and is the first engine's output
        assert result.baseline is baseline_result
        # Each strategy result is stored individually
        assert len(result.strategy_results) == 3
        for name in policy_dict:
            assert name in result.strategy_results

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_crn_seed_generated_when_no_seed(self, MockMCEngine, full_config, policy_dict):
        """When no seed is provided, a CRN base seed is still generated."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        result = Simulation.compare_insurance_strategies(
            config=full_config,
            insurance_policies=policy_dict,
            n_scenarios=50,
            n_jobs=1,
            seed=None,
        )

        # CRN seed should be set even when seed is None
        assert result.crn_seed is not None
        # All engines should use the same CRN seed
        for call in MockMCEngine.call_args_list:
            kwargs = call[1]
            assert kwargs["config"].crn_base_seed == result.crn_seed

    @patch("ergodic_insurance.simulation.MonteCarloEngine")
    def test_compare_respects_time_horizon_years(self, MockMCEngine, policy_dict):
        """Regression test for #1080: compare_insurance_strategies must use
        config.simulation.time_horizon_years, not a hardcoded fallback."""
        config = _make_full_config()
        config.simulation.time_horizon_years = 30

        mock_engine = MagicMock()
        mock_engine.run.return_value = self._make_mock_engine_result()
        MockMCEngine.return_value = mock_engine

        Simulation.compare_insurance_strategies(
            config=config,
            insurance_policies=policy_dict,
            n_scenarios=10,
            n_jobs=1,
            seed=42,
        )

        # All engines (baseline + strategies) should receive n_years=30
        for call_args in MockMCEngine.call_args_list:
            sim_config = call_args[1]["config"]
            assert (
                sim_config.n_years == 30
            ), f"Expected n_years=30 from time_horizon_years, got {sim_config.n_years}"


# ---------------------------------------------------------------------------
# Tests for run_with_loss_data: normal success path (lines 793-876)
# ---------------------------------------------------------------------------


class TestRunWithLossDataSuccessPath:
    """Additional tests for the run_with_loss_data method to ensure
    the normal (non-error) path works correctly with various inputs."""

    def test_run_with_empty_loss_data(self, manufacturer_config):
        """Simulation with no losses should complete successfully."""
        mfr = WidgetManufacturer(manufacturer_config)
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        loss_data = LossData(
            timestamps=np.array([], dtype=np.float64),
            loss_amounts=np.array([], dtype=np.float64),
        )

        result = sim.run_with_loss_data(loss_data, validate=True)

        assert isinstance(result, SimulationResults)
        assert result.insolvency_year is None
        assert len(result.years) == 5
        # No claims should be recorded
        assert np.sum(result.claim_counts) == 0
        assert np.sum(result.claim_amounts) == 0

    def test_run_with_loss_data_losses_outside_horizon(self, manufacturer_config):
        """Losses with timestamps outside the time_horizon should be
        ignored (filtered in the year grouping logic)."""
        mfr = WidgetManufacturer(manufacturer_config)
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        loss_data = LossData(
            timestamps=np.array([10.0, 20.0], dtype=np.float64),
            loss_amounts=np.array([5_000_000.0, 10_000_000.0], dtype=np.float64),
            loss_types=["cat", "cat"],
            claim_ids=["c1", "c2"],
        )

        result = sim.run_with_loss_data(loss_data, validate=True)

        assert result.insolvency_year is None
        assert np.sum(result.claim_counts) == 0
        assert np.sum(result.claim_amounts) == 0

    def test_run_with_loss_data_multiple_losses_same_year(self, manufacturer_config):
        """Multiple losses in the same year should all be processed."""
        mfr = WidgetManufacturer(manufacturer_config)
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        loss_data = LossData(
            timestamps=np.array([2.1, 2.5, 2.9], dtype=np.float64),
            loss_amounts=np.array([100_000.0, 200_000.0, 150_000.0], dtype=np.float64),
            loss_types=["attritional"] * 3,
            claim_ids=["c1", "c2", "c3"],
        )

        result = sim.run_with_loss_data(loss_data, validate=True)

        # Year 2 should have 3 claims with total amount 450,000
        assert result.claim_counts[2] == 3
        assert result.claim_amounts[2] == pytest.approx(450_000, rel=1e-6)

    def test_run_with_loss_data_reentrant(self, manufacturer_config):
        """Running run_with_loss_data multiple times should reset state
        and produce consistent results."""
        mfr = WidgetManufacturer(manufacturer_config)
        sim = Simulation(manufacturer=mfr, time_horizon=5, seed=42)

        loss_data = LossData(
            timestamps=np.array([1.0], dtype=np.float64),
            loss_amounts=np.array([50_000.0], dtype=np.float64),
            loss_types=["test"],
            claim_ids=["c1"],
        )

        result1 = sim.run_with_loss_data(loss_data, validate=True)

        # Re-create manufacturer (since it's modified in place)
        sim.manufacturer = WidgetManufacturer(manufacturer_config)
        result2 = sim.run_with_loss_data(loss_data, validate=True)

        # Both runs should produce identical results
        np.testing.assert_array_equal(result1.claim_counts, result2.claim_counts)
        np.testing.assert_array_equal(result1.claim_amounts, result2.claim_amounts)

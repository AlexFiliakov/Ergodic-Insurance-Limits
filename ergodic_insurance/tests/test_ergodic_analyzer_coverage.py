"""Targeted tests for ergodic_analyzer.py to cover specific missing lines.

Coverage targets:
    Lines 283-285: ErgodicData.validate() edge cases
    Lines 767, 785: calculate_time_average_growth edge branches
    Lines 791-796: _extract_trajectory_values
    Lines 802-803: _calculate_growth_rates
    Lines 809-848: _process_variable_length_trajectories
    Line 1028: calculate_ensemble_average with variable-length list input
    Lines 1533-1535: compare_scenarios when no valid time-average data
    Lines 1923, 1959: analyze_simulation_batch with all-bankrupt scenarios
    Lines 2214-2215, 2230-2231: integrate_loss_ergodic_analysis validation/no-insurance
    Lines 2284-2291: integrate_loss_ergodic_analysis insolvency mid-simulation
    Lines 2581-2585: validate_insurance_ergodic_impact premium check
    Lines 2622-2626: validate_insurance_ergodic_impact collateral check
"""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.ergodic_analyzer import (
    ErgodicAnalysisResults,
    ErgodicAnalyzer,
    ErgodicData,
    ValidationResults,
)
from ergodic_insurance.simulation import SimulationResults

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    """Provide a standard ErgodicAnalyzer instance."""
    return ErgodicAnalyzer(convergence_threshold=0.01)


def _make_sim_result(
    n_years=20,
    equity_start=1_000_000,
    growth=0.05,
    insolvency_year=None,
    claim_total=50_000,
):
    """Helper: build a SimulationResults with deterministic exponential growth."""
    years = np.arange(n_years)
    equity = equity_start * np.exp(growth * years)
    assets = equity * 1.2
    roe = np.full(n_years, growth)
    revenue = np.full(n_years, 500_000.0)
    net_income = np.full(n_years, 50_000.0)
    claim_counts = np.ones(n_years, dtype=int)
    claim_amounts = np.full(n_years, claim_total / n_years)

    if insolvency_year is not None and insolvency_year < n_years:
        equity[insolvency_year:] = 0.0
        assets[insolvency_year:] = 0.0
        roe[insolvency_year:] = np.nan
        revenue[insolvency_year:] = 0.0
        net_income[insolvency_year:] = 0.0

    return SimulationResults(
        years=years,
        assets=assets,
        equity=equity,
        roe=roe,
        revenue=revenue,
        net_income=net_income,
        claim_counts=claim_counts,
        claim_amounts=claim_amounts,
        insolvency_year=insolvency_year,
    )


# ===================================================================
# Lines 283-285: ErgodicData.validate()
# ===================================================================


class TestErgodicDataValidate:
    """Tests for ErgodicData.validate() covering empty and mismatched arrays."""

    def test_validate_empty_time_series_returns_false(self):
        """An empty time_series with non-empty values should fail validation."""
        data = ErgodicData(
            time_series=np.array([]),
            values=np.array([1.0, 2.0]),
            metadata={},
        )
        assert data.validate() is False

    def test_validate_empty_values_returns_false(self):
        """Non-empty time_series with empty values should fail validation."""
        data = ErgodicData(
            time_series=np.array([0.0, 1.0]),
            values=np.array([]),
            metadata={},
        )
        assert data.validate() is False

    def test_validate_both_empty_returns_false(self):
        """Both arrays empty should fail validation (line 283 condition)."""
        data = ErgodicData(
            time_series=np.array([]),
            values=np.array([]),
            metadata={},
        )
        assert data.validate() is False

    def test_validate_mismatched_lengths_returns_false(self):
        """Arrays of different lengths should fail validation (line 285)."""
        data = ErgodicData(
            time_series=np.arange(5),
            values=np.arange(3, dtype=float),
            metadata={},
        )
        assert data.validate() is False

    def test_validate_matching_lengths_returns_true(self):
        """Properly matched arrays should pass validation."""
        data = ErgodicData(
            time_series=np.arange(5),
            values=np.random.default_rng(42).random(5),
            metadata={"source": "test"},
        )
        assert data.validate() is True

    def test_validate_single_element_returns_true(self):
        """Single-element arrays should pass validation."""
        data = ErgodicData(
            time_series=np.array([0.0]),
            values=np.array([100.0]),
            metadata={},
        )
        assert data.validate() is True


# ===================================================================
# Line 767: calculate_time_average_growth all-non-positive but
#           final value positive (valid_mask all False after filtering)
# Line 785: time_horizon <= 0 branch
# ===================================================================


class TestTimeAverageGrowthEdgeCases:
    """Edge cases in calculate_time_average_growth targeting lines 767 and 785."""

    def test_all_negative_except_final_returns_neg_inf(self, analyzer):
        """When all values are non-positive, valid_mask is all False -> -inf (line 767).

        However, since values[-1] <= 0 is checked first (line 758), we need a
        trajectory where values[-1] > 0 but np.any(valid_mask) is False.
        This is impossible because if values[-1] > 0 then valid_mask[-1] is True.
        So line 767 is actually reached when values > 0 has no True entries,
        but values[-1] > 0 already passed. Re-reading: values > 0 uses the full
        array. If values[-1] > 0 then valid_mask[-1] is True, so np.any(valid_mask)
        is True. Therefore line 767 is only reachable if values[-1] > 0 passes
        but somehow valid_mask is all False. This cannot happen with standard arrays.

        Actually, looking more carefully, values[-1] <= 0 returns -inf at line 758-759.
        So if we get past that, values[-1] > 0. Then valid_mask = values > 0 will have
        at least one True entry. Line 767 is therefore dead code in practice, but we
        should still test the nearby logic. Let's test the case where most values are
        zero/negative but the trajectory starts and ends positive.
        """
        # Trajectory where many values are zero but first and last are positive
        values = np.array([100.0, 0.0, 0.0, -50.0, 0.0, 200.0])
        growth = analyzer.calculate_time_average_growth(values)
        # Should compute growth from first valid (100.0) to last (200.0)
        assert np.isfinite(growth)
        assert growth > 0  # doubled from 100 to 200

    def test_time_horizon_zero_positive_values(self, analyzer):
        """When time_horizon effectively becomes 0 -> returns 0.0 (line 785).

        This happens when first_idx equals len(values)-1, meaning the only
        positive value is the last one, and time_horizon = len-1-first_idx = 0.
        """
        # Only the last value is positive; first_idx = 4, time_horizon = 5-1-4 = 0
        values = np.array([-1.0, -2.0, -3.0, -4.0, 5.0])
        growth = analyzer.calculate_time_average_growth(values)
        # time_horizon = len(5)-1-4 = 0; final_value=5>0, initial=5>0, time_horizon=0
        # -> goes to line 785: "return 0.0 if time_horizon <= 0 else -np.inf"
        assert growth == pytest.approx(0.0, abs=1e-10)

    def test_initial_positive_final_positive_normal_growth(self, analyzer):
        """Normal trajectory should yield finite positive growth."""
        values = np.array([100.0, 110.0, 121.0, 133.1])
        growth = analyzer.calculate_time_average_growth(values)
        expected = (1.0 / 3) * np.log(133.1 / 100.0)
        assert growth == pytest.approx(expected, rel=1e-6)

    def test_time_horizon_zero_from_explicit_param(self, analyzer):
        """Explicitly passing time_horizon=0 should return 0.0 (line 785)."""
        values = np.array([100.0, 200.0])
        growth = analyzer.calculate_time_average_growth(values, time_horizon=0)
        assert growth == pytest.approx(0.0, abs=1e-10)


# ===================================================================
# Lines 791-796: _extract_trajectory_values
# Lines 802-803: _calculate_growth_rates
# ===================================================================


class TestPrivateHelperMethods:
    """Tests for _extract_trajectory_values and _calculate_growth_rates."""

    def test_extract_trajectory_values_normal(self, analyzer):
        """Extract final, initial, and length from normal trajectories (lines 791-796)."""
        trajectories = [
            np.array([100.0, 150.0, 200.0]),
            np.array([50.0, 80.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]
        finals, initials, lengths = analyzer._extract_trajectory_values(trajectories)

        np.testing.assert_array_equal(finals, [200.0, 80.0, 40.0])
        np.testing.assert_array_equal(initials, [100.0, 50.0, 10.0])
        np.testing.assert_array_equal(lengths, [3, 2, 4])

    def test_extract_trajectory_values_with_empty_trajectories(self, analyzer):
        """Empty trajectories should be skipped; all-empty returns empty arrays (line 792-793)."""
        trajectories = [np.array([]), np.array([])]
        finals, initials, lengths = analyzer._extract_trajectory_values(trajectories)

        assert len(finals) == 0
        assert len(initials) == 0
        assert len(lengths) == 0

    def test_extract_trajectory_values_mixed_empty_and_normal(self, analyzer):
        """Mix of empty and non-empty trajectories skips empties."""
        trajectories = [
            np.array([]),
            np.array([100.0, 200.0]),
            np.array([]),
            np.array([50.0]),
        ]
        finals, initials, lengths = analyzer._extract_trajectory_values(trajectories)

        np.testing.assert_array_equal(finals, [200.0, 50.0])
        np.testing.assert_array_equal(initials, [100.0, 50.0])
        np.testing.assert_array_equal(lengths, [2, 1])

    def test_calculate_growth_rates_normal(self, analyzer):
        """Growth rates from valid trajectory data (lines 802-803)."""
        finals = np.array([200.0, 80.0])
        initials = np.array([100.0, 40.0])
        lengths = np.array([3, 2])  # t-1 = 2 and 1

        rates = analyzer._calculate_growth_rates(finals, initials, lengths)

        expected_rate_0 = np.log(200.0 / 100.0) / 2  # ln(2)/2
        expected_rate_1 = np.log(80.0 / 40.0) / 1  # ln(2)/1
        np.testing.assert_allclose(rates, [expected_rate_0, expected_rate_1], rtol=1e-10)

    def test_calculate_growth_rates_single_point_trajectories(self, analyzer):
        """Trajectories of length 1 have t-1=0 and are skipped -> empty array."""
        finals = np.array([100.0])
        initials = np.array([100.0])
        lengths = np.array([1])

        rates = analyzer._calculate_growth_rates(finals, initials, lengths)
        assert len(rates) == 0

    def test_calculate_growth_rates_empty_inputs(self, analyzer):
        """Empty inputs should return empty array."""
        rates = analyzer._calculate_growth_rates(np.array([]), np.array([]), np.array([]))
        assert len(rates) == 0


# ===================================================================
# Lines 809-848: _process_variable_length_trajectories
# ===================================================================


class TestProcessVariableLengthTrajectories:
    """Full coverage of _process_variable_length_trajectories."""

    def test_final_value_metric(self, analyzer):
        """Process variable-length trajectories with metric='final_value' (lines 827-830)."""
        trajectories = [
            np.array([100.0, 200.0, 300.0]),
            np.array([50.0, 100.0]),
            np.array([10.0, 0.0]),  # final value is 0 -> filtered by valid_mask
        ]
        results = analyzer._process_variable_length_trajectories(trajectories, metric="final_value")

        assert "mean" in results
        assert "std" in results
        assert "median" in results
        assert "survival_rate" in results
        assert "n_survived" in results
        assert "n_total" in results

        # Only first two trajectories have positive initial AND final values
        # traj[0]: initial=100, final=300 -> valid
        # traj[1]: initial=50, final=100 -> valid
        # traj[2]: initial=10, final=0 -> invalid (final not > 0)
        assert results["mean"] == pytest.approx(200.0)  # (300+100)/2
        assert results["n_total"] == 3

    def test_growth_rate_metric(self, analyzer):
        """Process variable-length trajectories with metric='growth_rate' (lines 831-837)."""
        trajectories = [
            np.array([100.0, 200.0, 400.0]),  # length 3, growth over 2 steps
            np.array([50.0, 100.0]),  # length 2, growth over 1 step
        ]
        results = analyzer._process_variable_length_trajectories(trajectories, metric="growth_rate")

        assert "mean" in results
        assert "std" in results
        assert "median" in results

        # rate_0 = ln(400/100)/2 = ln(4)/2
        # rate_1 = ln(100/50)/1 = ln(2)
        expected_0 = np.log(4) / 2
        expected_1 = np.log(2)
        assert results["mean"] == pytest.approx((expected_0 + expected_1) / 2, rel=1e-6)

    def test_full_metric_returns_none(self, analyzer):
        """Metric 'full' returns None for mean/std trajectory (lines 838-840)."""
        trajectories = [
            np.array([100.0, 200.0]),
            np.array([50.0, 100.0, 150.0]),
        ]
        results = analyzer._process_variable_length_trajectories(trajectories, metric="full")

        assert results["mean_trajectory"] is None
        assert results["std_trajectory"] is None
        assert "survival_rate" in results

    def test_survival_statistics(self, analyzer):
        """Survival rate computed correctly (lines 843-846)."""
        trajectories = [
            np.array([100.0, 200.0]),  # survived (final > 0)
            np.array([100.0, 0.0]),  # bankrupt (final == 0)
            np.array([100.0, -50.0]),  # bankrupt (final < 0)
            np.array([100.0, 50.0]),  # survived
        ]
        results = analyzer._process_variable_length_trajectories(trajectories, metric="final_value")

        assert results["survival_rate"] == pytest.approx(0.5)
        assert results["n_survived"] == 2
        assert results["n_total"] == 4

    def test_empty_trajectories_after_filtering(self, analyzer):
        """When all trajectories are empty, metrics default to 0.0 (line 824-825)."""
        trajectories = [np.array([]), np.array([])]
        results = analyzer._process_variable_length_trajectories(trajectories, metric="final_value")

        assert results["mean"] == 0.0
        assert results["std"] == 0.0
        assert results["median"] == 0.0
        assert results["survival_rate"] == 0.0

    def test_all_invalid_growth_rates(self, analyzer):
        """When no valid growth rates exist, defaults to 0.0 (line 835)."""
        # All trajectories length 1 -> no growth rate can be computed
        trajectories = [np.array([100.0]), np.array([200.0])]
        results = analyzer._process_variable_length_trajectories(trajectories, metric="growth_rate")

        assert results["mean"] == 0.0


# ===================================================================
# Line 1028: calculate_ensemble_average with variable-length lists
# ===================================================================


class TestCalculateEnsembleAverageVariableLength:
    """Test the variable-length branch in calculate_ensemble_average."""

    def test_variable_length_list_routes_to_variable_handler(self, analyzer):
        """List with different-length arrays -> _process_variable_length_trajectories (line 1028)."""
        trajectories = [
            np.array([100.0, 200.0, 300.0]),
            np.array([50.0, 100.0]),
        ]
        results = analyzer.calculate_ensemble_average(trajectories, metric="final_value")

        assert "mean" in results
        assert "survival_rate" in results
        # Both trajectories end positive
        assert results["survival_rate"] == 1.0

    def test_variable_length_growth_rate(self, analyzer):
        """Variable-length list with growth_rate metric."""
        trajectories = [
            np.array([100.0, 200.0, 400.0]),
            np.array([50.0]),  # length 1, growth rate undefined
        ]
        results = analyzer.calculate_ensemble_average(trajectories, metric="growth_rate")

        assert "mean" in results
        # Only the first trajectory contributes a growth rate
        expected_rate = np.log(400.0 / 100.0) / 2
        assert results["mean"] == pytest.approx(expected_rate, rel=1e-6)


# ===================================================================
# Lines 1533-1535: compare_scenarios with no valid time-average data
# ===================================================================


class TestCompareScenariosNoValidData:
    """Test compare_scenarios when all trajectories yield -inf growth rates."""

    def test_all_bankrupt_yields_nan_significance(self, analyzer):
        """When all paths go bankrupt, significance fields are NaN (lines 1533-1535)."""
        # Create trajectories that all end at zero -> growth = -inf
        insured = [np.array([100.0, 0.0]), np.array([200.0, 0.0])]
        uninsured = [np.array([150.0, 0.0]), np.array([300.0, 0.0])]

        results = analyzer.compare_scenarios(insured, uninsured, metric="equity")

        assert np.isnan(results["ergodic_advantage"]["t_statistic"])
        assert np.isnan(results["ergodic_advantage"]["p_value"])
        assert results["ergodic_advantage"]["significant"] is False

    def test_one_side_bankrupt_yields_nan_significance(self, analyzer):
        """When only insured side is all bankrupt, still hits nan branch."""
        insured = [np.array([100.0, 0.0])]
        uninsured = [np.array([100.0, 200.0])]

        results = analyzer.compare_scenarios(insured, uninsured, metric="equity")

        # insured_time_avg_valid is empty -> hits else branch
        assert np.isnan(results["ergodic_advantage"]["t_statistic"])
        assert np.isnan(results["ergodic_advantage"]["p_value"])
        assert results["ergodic_advantage"]["significant"] is False


# ===================================================================
# Lines 1923, 1959: analyze_simulation_batch with all-bankrupt results
# ===================================================================


class TestAnalyzeSimulationBatchAllBankrupt:
    """Test analyze_simulation_batch when all simulations result in insolvency."""

    def test_all_bankrupt_convergence_and_divergence(self, analyzer):
        """All bankrupt -> converged=False, se=inf, ergodic_divergence=nan (lines 1923, 1959)."""
        results = []
        for i in range(5):
            result = _make_sim_result(
                n_years=10,
                equity_start=1_000_000,
                growth=-0.5,
                insolvency_year=3,
            )
            results.append(result)

        analysis = analyzer.analyze_simulation_batch(results, label="All Bankrupt")

        # Line 1923: converged, se = False, np.inf
        assert analysis["convergence"]["converged"] is False
        assert analysis["convergence"]["standard_error"] == np.inf

        # Line 1959: analysis["ergodic_divergence"] = np.nan
        assert np.isnan(analysis["ergodic_divergence"])

        # Time average stats should indicate no valid data
        assert analysis["time_average"]["mean"] == -np.inf


# ===================================================================
# Lines 2214-2215: integrate_loss_ergodic_analysis with invalid loss_data
# Lines 2230-2231: integrate_loss_ergodic_analysis with no insurance_program
# Lines 2284-2291: integrate_loss_ergodic_analysis insolvency mid-simulation
# ===================================================================


class TestIntegrateLossErgodicAnalysis:
    """Tests for integrate_loss_ergodic_analysis covering validation and insolvency."""

    def test_invalid_loss_data_returns_failed_results(self, analyzer):
        """When loss_data.validate() returns False, return invalid results (lines 2214-2215)."""
        from ergodic_insurance.loss_distributions import LossData

        # Create invalid LossData: mismatched array lengths
        invalid_loss_data = LossData(
            timestamps=np.array([0.0, 1.0]),
            loss_amounts=np.array([100.0]),  # length mismatch -> validate() returns False
        )
        assert invalid_loss_data.validate() is False

        mock_manufacturer = MagicMock()
        result = analyzer.integrate_loss_ergodic_analysis(
            loss_data=invalid_loss_data,
            insurance_program=None,
            manufacturer=mock_manufacturer,
            time_horizon=10,
            n_simulations=5,
        )

        assert isinstance(result, ErgodicAnalysisResults)
        assert result.validation_passed is False
        assert result.time_average_growth == -np.inf
        assert result.survival_rate == 0.0
        assert "error" in result.metadata

    def test_no_insurance_program_uses_raw_loss_data(self, analyzer):
        """When insurance_program is None, raw loss_data is used (lines 2230-2231).

        This test verifies the code path where insured_loss_data = loss_data
        and insurance_metadata = {}.
        """
        from ergodic_insurance.loss_distributions import LossData

        valid_loss_data = LossData(
            timestamps=np.array([0.0, 1.0, 2.0, 5.0]),
            loss_amounts=np.array([10_000.0, 20_000.0, 15_000.0, 5_000.0]),
        )
        assert valid_loss_data.validate() is True

        # We need a manufacturer that deepcopy works on and has a step method.
        # We also need Simulation to work. Let's mock step_annual to return
        # metrics that keep the company solvent.
        mock_metrics = {
            "assets": 1_000_000,
            "equity": 500_000,
            "roe": 0.10,
            "revenue": 200_000,
            "net_income": 20_000,
            "claim_count": 1,
            "claim_amount": 10_000,
        }

        with patch("ergodic_insurance.simulation.Simulation") as MockSim:
            sim_instance = MagicMock()
            sim_instance.step_annual.return_value = mock_metrics
            sim_instance.insolvency_year = None
            sim_instance.years = np.arange(5)
            sim_instance.assets = np.full(5, 1_000_000.0)
            sim_instance.equity = np.full(5, 500_000.0)
            sim_instance.roe = np.full(5, 0.10)
            sim_instance.revenue = np.full(5, 200_000.0)
            sim_instance.net_income = np.full(5, 20_000.0)
            sim_instance.claim_counts = np.ones(5, dtype=int)
            sim_instance.claim_amounts = np.full(5, 10_000.0)
            MockSim.return_value = sim_instance

            mock_manufacturer = MagicMock()

            result = analyzer.integrate_loss_ergodic_analysis(
                loss_data=valid_loss_data,
                insurance_program=None,  # This is the key: no insurance
                manufacturer=mock_manufacturer,
                time_horizon=5,
                n_simulations=3,
            )

        assert isinstance(result, ErgodicAnalysisResults)

    def test_insolvency_during_simulation(self, analyzer):
        """When equity goes to zero mid-simulation, insolvency is recorded (lines 2284-2291).

        The code sets insolvency_year, fills remaining arrays with zeros, and breaks.
        """
        from ergodic_insurance.loss_distributions import LossData

        valid_loss_data = LossData(
            timestamps=np.array([0.0, 1.0, 2.0]),
            loss_amounts=np.array([100.0, 200.0, 300.0]),
        )

        # Return metrics that go insolvent at year 2
        call_count = [0]

        def step_side_effect(year, losses):
            call_count[0] += 1
            if year < 2:
                return {
                    "assets": 1_000_000,
                    "equity": 500_000,
                    "roe": 0.10,
                    "revenue": 200_000,
                    "net_income": 20_000,
                    "claim_count": 1,
                    "claim_amount": 100,
                }
            # Insolvent: equity <= 0
            return {
                "assets": 0,
                "equity": 0,  # triggers insolvency
                "roe": 0,
                "revenue": 0,
                "net_income": 0,
                "claim_count": 1,
                "claim_amount": 300,
            }

        with patch("ergodic_insurance.simulation.Simulation") as MockSim:
            sim_instance = MagicMock()
            sim_instance.step_annual.side_effect = step_side_effect
            sim_instance.insolvency_year = None
            sim_instance.years = np.arange(5)
            sim_instance.assets = np.zeros(5)
            sim_instance.equity = np.zeros(5)
            sim_instance.roe = np.zeros(5)
            sim_instance.revenue = np.zeros(5)
            sim_instance.net_income = np.zeros(5)
            sim_instance.claim_counts = np.zeros(5, dtype=int)
            sim_instance.claim_amounts = np.zeros(5)
            MockSim.return_value = sim_instance

            mock_manufacturer = MagicMock()

            result = analyzer.integrate_loss_ergodic_analysis(
                loss_data=valid_loss_data,
                insurance_program=None,
                manufacturer=mock_manufacturer,
                time_horizon=5,
                n_simulations=2,
            )

        assert isinstance(result, ErgodicAnalysisResults)

    def test_with_insurance_program_applies_insurance(self, analyzer):
        """When insurance_program is provided, apply_insurance is called (lines 2227-2228, 2328).

        This covers the truthy branch of 'if insurance_program:' and the
        insurance_metadata population path.
        """
        from ergodic_insurance.loss_distributions import LossData

        valid_loss_data = LossData(
            timestamps=np.array([0.0, 1.0, 2.0]),
            loss_amounts=np.array([10_000.0, 20_000.0, 15_000.0]),
        )

        # Create a mock insurance program that apply_insurance can use
        mock_insurance = MagicMock()
        # apply_insurance returns a new LossData with metadata
        insured_loss_data = LossData(
            timestamps=np.array([0.0, 1.0, 2.0]),
            loss_amounts=np.array([5_000.0, 10_000.0, 7_500.0]),
            metadata={
                "insurance_applied": True,
                "total_recoveries": 22_500.0,
                "total_premiums": 5_000.0,
                "net_benefit": 17_500.0,
            },
        )

        mock_metrics = {
            "assets": 1_000_000,
            "equity": 500_000,
            "roe": 0.10,
            "revenue": 200_000,
            "net_income": 20_000,
            "claim_count": 1,
            "claim_amount": 5_000,
        }

        with patch.object(valid_loss_data, "apply_insurance", return_value=insured_loss_data):
            with patch("ergodic_insurance.simulation.Simulation") as MockSim:
                sim_instance = MagicMock()
                sim_instance.step_annual.return_value = mock_metrics
                sim_instance.insolvency_year = None
                sim_instance.years = np.arange(5)
                sim_instance.assets = np.full(5, 1_000_000.0)
                sim_instance.equity = np.full(5, 500_000.0)
                sim_instance.roe = np.full(5, 0.10)
                sim_instance.revenue = np.full(5, 200_000.0)
                sim_instance.net_income = np.full(5, 20_000.0)
                sim_instance.claim_counts = np.ones(5, dtype=int)
                sim_instance.claim_amounts = np.full(5, 5_000.0)
                MockSim.return_value = sim_instance

                mock_manufacturer = MagicMock()

                result = analyzer.integrate_loss_ergodic_analysis(
                    loss_data=valid_loss_data,
                    insurance_program=mock_insurance,
                    manufacturer=mock_manufacturer,
                    time_horizon=5,
                    n_simulations=3,
                )

        assert isinstance(result, ErgodicAnalysisResults)
        # insurance_impact should be a dict (may or may not be populated depending on metadata)
        assert isinstance(result.insurance_impact, dict)


# ===================================================================
# Lines 2581-2585: validate_insurance_ergodic_impact premium check
# Lines 2622-2626: validate_insurance_ergodic_impact collateral check
# ===================================================================


class TestValidateInsuranceErgodicImpact:
    """Tests for validate_insurance_ergodic_impact premium and collateral checks."""

    def test_premium_deduction_check(self, analyzer):
        """When insurance_program has calculate_premium, the premium check runs (lines 2581-2585)."""
        base = _make_sim_result(n_years=10, equity_start=1_000_000, growth=0.05, claim_total=50_000)
        insured = _make_sim_result(
            n_years=10, equity_start=1_000_000, growth=0.04, claim_total=30_000
        )

        # Create a mock insurance program with calculate_premium
        mock_program = MagicMock()
        mock_program.calculate_premium.return_value = 10_000.0
        # Ensure it does NOT have collateral_requirement so we isolate premium check
        del mock_program.collateral_requirement

        validation = analyzer.validate_insurance_ergodic_impact(
            base_scenario=base,
            insurance_scenario=insured,
            insurance_program=mock_program,
        )

        assert isinstance(validation, ValidationResults)
        assert "premium_check" in validation.details
        assert "expected" in validation.details["premium_check"]
        assert "actual_diff" in validation.details["premium_check"]
        assert "valid" in validation.details["premium_check"]

    def test_collateral_impacts_check(self, analyzer):
        """When insurance_program has collateral_requirement, collateral check runs (lines 2622-2626)."""
        base = _make_sim_result(n_years=10, equity_start=1_000_000, growth=0.05, claim_total=50_000)
        # Insured scenario with slightly different assets
        insured = _make_sim_result(
            n_years=10, equity_start=1_000_000, growth=0.048, claim_total=30_000
        )

        mock_program = MagicMock()
        mock_program.collateral_requirement = 100_000.0
        # Remove calculate_premium to isolate collateral check
        del mock_program.calculate_premium

        validation = analyzer.validate_insurance_ergodic_impact(
            base_scenario=base,
            insurance_scenario=insured,
            insurance_program=mock_program,
        )

        assert isinstance(validation, ValidationResults)
        assert "collateral_check" in validation.details
        assert "asset_difference" in validation.details["collateral_check"]
        assert "valid" in validation.details["collateral_check"]

    def test_both_premium_and_collateral_checks(self, analyzer):
        """Both checks fire when both attributes exist."""
        base = _make_sim_result(n_years=10, equity_start=1_000_000, growth=0.05, claim_total=50_000)
        insured = _make_sim_result(
            n_years=10, equity_start=1_000_000, growth=0.045, claim_total=30_000
        )

        mock_program = MagicMock()
        mock_program.calculate_premium.return_value = 5_000.0
        mock_program.collateral_requirement = 50_000.0

        validation = analyzer.validate_insurance_ergodic_impact(
            base_scenario=base,
            insurance_scenario=insured,
            insurance_program=mock_program,
        )

        assert isinstance(validation, ValidationResults)
        assert "premium_check" in validation.details
        assert "collateral_check" in validation.details
        assert "recovery_check" in validation.details
        assert "growth_check" in validation.details

    def test_no_insurance_program_skips_premium_and_collateral(self, analyzer):
        """When no insurance_program is provided, premium/collateral checks are skipped."""
        base = _make_sim_result(n_years=10, equity_start=1_000_000, growth=0.05, claim_total=50_000)
        insured = _make_sim_result(
            n_years=10, equity_start=1_000_000, growth=0.04, claim_total=30_000
        )

        validation = analyzer.validate_insurance_ergodic_impact(
            base_scenario=base,
            insurance_scenario=insured,
            insurance_program=None,
        )

        assert isinstance(validation, ValidationResults)
        assert "premium_check" not in validation.details
        assert "collateral_check" not in validation.details
        # Growth and recovery checks should still be present
        assert "recovery_check" in validation.details
        assert "growth_check" in validation.details

    def test_premium_check_valid_when_costs_match(self, analyzer):
        """Premium validation passes when net income difference matches expected premium."""
        n = 10
        base_income = np.full(n, 100_000.0)
        insured_income = np.full(n, 90_000.0)  # 10k less per year

        base = SimulationResults(
            years=np.arange(n),
            assets=np.full(n, 1_200_000.0),
            equity=np.full(n, 1_000_000.0),
            roe=np.full(n, 0.10),
            revenue=np.full(n, 500_000.0),
            net_income=base_income,
            claim_counts=np.ones(n, dtype=int),
            claim_amounts=np.full(n, 5_000.0),
            insolvency_year=None,
        )
        insured = SimulationResults(
            years=np.arange(n),
            assets=np.full(n, 1_200_000.0),
            equity=np.full(n, 1_000_000.0),
            roe=np.full(n, 0.09),
            revenue=np.full(n, 500_000.0),
            net_income=insured_income,
            claim_counts=np.ones(n, dtype=int),
            claim_amounts=np.full(n, 3_000.0),
            insolvency_year=None,
        )

        mock_program = MagicMock()
        # total income diff = sum(100k - 90k) * 10 = 100k
        # expected_premium * len(years) = 10k * 10 = 100k -> match
        mock_program.calculate_premium.return_value = 10_000.0
        del mock_program.collateral_requirement

        validation = analyzer.validate_insurance_ergodic_impact(
            base_scenario=base,
            insurance_scenario=insured,
            insurance_program=mock_program,
        )

        assert bool(validation.details["premium_check"]["valid"]) is True


# ===================================================================
# Additional integration tests to ensure code paths work end-to-end
# ===================================================================


class TestErgodicAnalysisResultsDataclass:
    """Verify ErgodicAnalysisResults can be instantiated with edge values."""

    def test_create_with_negative_infinity(self):
        """Results with -inf growth should instantiate cleanly."""
        result = ErgodicAnalysisResults(
            time_average_growth=-np.inf,
            ensemble_average_growth=0.0,
            survival_rate=0.0,
            ergodic_divergence=-np.inf,
            insurance_impact={},
            validation_passed=False,
            metadata={"error": "all bankrupt"},
        )
        assert result.validation_passed is False
        assert result.time_average_growth == -np.inf

    def test_create_with_normal_values(self):
        """Results with normal values should work correctly."""
        result = ErgodicAnalysisResults(
            time_average_growth=0.05,
            ensemble_average_growth=0.06,
            survival_rate=0.95,
            ergodic_divergence=-0.01,
            insurance_impact={"net_benefit": 100_000},
            validation_passed=True,
            metadata={"n_simulations": 1000},
        )
        assert result.validation_passed is True
        assert result.survival_rate == 0.95


class TestValidationResultsDataclass:
    """Verify ValidationResults fields."""

    def test_overall_valid_reflects_individual_checks(self):
        """Overall validity should match provided value."""
        valid = ValidationResults(
            premium_deductions_correct=True,
            recoveries_credited=True,
            collateral_impacts_included=True,
            time_average_reflects_benefit=True,
            overall_valid=True,
            details={"info": "all checks passed"},
        )
        assert valid.overall_valid is True

    def test_overall_invalid_when_one_fails(self):
        """When one check fails, overall should be false."""
        invalid = ValidationResults(
            premium_deductions_correct=False,
            recoveries_credited=True,
            collateral_impacts_included=True,
            time_average_reflects_benefit=True,
            overall_valid=False,
            details={"premium_check": {"valid": False}},
        )
        assert invalid.overall_valid is False


class TestCompareScenariosMixedPaths:
    """Additional compare_scenarios tests for mixed valid/invalid paths."""

    def test_compare_with_simulation_results_all_bankrupt(self, analyzer):
        """SimulationResults where all paths go bankrupt -> nan significance."""
        insured_results = [
            _make_sim_result(n_years=10, insolvency_year=3),
            _make_sim_result(n_years=10, insolvency_year=5),
        ]
        uninsured_results = [
            _make_sim_result(n_years=10, insolvency_year=2),
            _make_sim_result(n_years=10, insolvency_year=4),
        ]

        results = analyzer.compare_scenarios(insured_results, uninsured_results, metric="equity")

        assert "ergodic_advantage" in results
        # All paths bankrupt -> all growth rates are -inf -> no valid data
        assert np.isnan(results["ergodic_advantage"]["t_statistic"])
        assert np.isnan(results["ergodic_advantage"]["p_value"])
        assert results["ergodic_advantage"]["significant"] is False


# ===================================================================
# Issue #474: Verify removal of growth rate clamp at -1.0
# ===================================================================


class TestGrowthRateNoClamp:
    """Verify that time-average growth rates below -1.0 are preserved (issue #474).

    The previous implementation clamped growth rates to max(g, -1.0), which
    biased ergodic comparisons upward by softening near-bankruptcy trajectories.
    """

    def test_severe_loss_below_negative_one(self, analyzer):
        """99.9% equity loss over 1 year should yield g ≈ -6.9, not -1.0."""
        # $10M -> $100 in 1 year
        values = np.array([10_000_000.0, 100.0])
        growth = analyzer.calculate_time_average_growth(values)
        expected = np.log(100.0 / 10_000_000.0)  # ≈ -11.51
        assert abs(growth - expected) < 1e-6
        assert growth < -1.0, "Growth rate must not be clamped to -1.0"

    def test_99_percent_loss_over_10_years(self, analyzer):
        """$10M -> $10K over 10 years should yield g ≈ -0.69 (above -1.0, no clamp issue)."""
        values = np.array([10_000_000.0] + [0.0] * 8 + [10_000.0])
        # first_idx=0 (10M>0), final=10K, time_horizon=9
        # But values[1..8] are 0, so first valid positive is index 0
        growth = analyzer.calculate_time_average_growth(values)
        expected = (1.0 / 9) * np.log(10_000.0 / 10_000_000.0)
        assert abs(growth - expected) < 1e-6

    def test_99_point_999_percent_loss_single_year(self, analyzer):
        """Near-total loss in 1 year: g = ln(1/10M) ≈ -16.1."""
        values = np.array([10_000_000.0, 1.0])
        growth = analyzer.calculate_time_average_growth(values)
        expected = np.log(1.0 / 10_000_000.0)  # ≈ -16.12
        assert abs(growth - expected) < 1e-6
        assert growth < -10.0

    def test_moderate_loss_preserves_exact_value(self, analyzer):
        """50% loss over 5 years: g = ln(0.5)/5 ≈ -0.139 (was never clamped, but verify)."""
        initial = 1_000_000.0
        final = 500_000.0
        values = np.array([initial, 0.0, 0.0, 0.0, 0.0, final])
        growth = analyzer.calculate_time_average_growth(values)
        expected = (1.0 / 5) * np.log(final / initial)
        assert abs(growth - expected) < 1e-6

    def test_compare_scenarios_handles_very_negative_growth(self, analyzer):
        """compare_scenarios should include very negative finite growth rates in means."""
        # Insured: mild losses
        insured = [np.array([1_000_000.0] * 5 + [900_000.0]) for _ in range(5)]
        # Uninsured: catastrophic losses (below old clamp)
        uninsured = [np.array([1_000_000.0] * 5 + [10.0]) for _ in range(5)]

        results = analyzer.compare_scenarios(insured, uninsured, metric="equity")

        # Uninsured mean should reflect the true severity (g ≈ ln(10/1e6)/5 ≈ -2.3)
        uninsured_mean = results["uninsured"]["time_average_mean"]
        assert (
            uninsured_mean < -1.0
        ), f"Uninsured time_average_mean={uninsured_mean} should be < -1.0"
        # Ergodic advantage should be large and positive
        assert results["ergodic_advantage"]["time_average_gain"] > 1.0

    def test_mean_of_mixed_growth_rates_not_clamped(self, analyzer):
        """Mean of growth rates should reflect unclamped values."""
        # Mix of mild and catastrophic trajectories
        trajectories = [
            np.array([1_000_000.0, 1_050_000.0]),  # +5%
            np.array([1_000_000.0, 100.0]),  # -99.99% -> g ≈ -9.21
        ]
        growths = [analyzer.calculate_time_average_growth(t) for t in trajectories]
        mean_growth = np.mean(growths)
        # With clamp: mean ≈ (0.0488 + (-1.0))/2 = -0.476
        # Without clamp: mean ≈ (0.0488 + (-9.21))/2 = -4.58
        assert mean_growth < -1.0, f"Mean growth {mean_growth} should be < -1.0 without clamping"

"""Tests for the ErgodicAnalyzer class."""

import numpy as np
import pytest

from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
from ergodic_insurance.simulation import SimulationResults


class TestErgodicAnalyzer:
    """Test suite for ErgodicAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create an ErgodicAnalyzer instance."""
        return ErgodicAnalyzer(convergence_threshold=0.01)

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample growth trajectory."""
        # Exponential growth with noise
        time = np.arange(100)
        growth_rate = 0.05
        noise = np.random.normal(0, 0.1, 100)
        values = 1000000 * np.exp(growth_rate * time + np.cumsum(noise))
        return values

    @pytest.fixture
    def multiple_trajectories(self):
        """Create multiple simulation trajectories."""
        n_paths = 100
        n_time = 100
        trajectories = []

        for _ in range(n_paths):
            time = np.arange(n_time)
            growth_rate = np.random.normal(0.05, 0.02)
            noise = np.random.normal(0, 0.1, n_time)
            values = 1000000 * np.exp(growth_rate * time + np.cumsum(noise))
            trajectories.append(values)

        return np.array(trajectories)

    def test_calculate_time_average_growth(self, analyzer, sample_trajectory):
        """Test time-average growth calculation."""
        growth_rate = analyzer.calculate_time_average_growth(sample_trajectory)

        # Should be close to the true growth rate (0.05)
        assert isinstance(growth_rate, float)
        assert 0.03 < growth_rate < 0.08  # Allow for noise from cumulative random walk

    def test_time_average_growth_with_zero_values(self, analyzer):
        """Test handling of zero values in trajectory."""
        # Trajectory that goes to zero
        values = np.array([1000000, 500000, 100000, 0, 0, 0])
        growth_rate = analyzer.calculate_time_average_growth(values)

        assert growth_rate == -np.inf

    def test_time_average_growth_with_negative_values(self, analyzer):
        """Test handling of negative values in trajectory."""
        # Trajectory with negative values
        values = np.array([1000000, 500000, -100000, -200000])
        growth_rate = analyzer.calculate_time_average_growth(values)

        # Should handle negative values appropriately
        assert growth_rate == -np.inf

    def test_calculate_ensemble_average_final_value(self, analyzer, multiple_trajectories):
        """Test ensemble average calculation for final values."""
        results = analyzer.calculate_ensemble_average(multiple_trajectories, metric="final_value")

        assert "mean" in results
        assert "std" in results
        assert "median" in results
        assert "survival_rate" in results

        # Check reasonable values
        assert results["mean"] > 0
        assert results["std"] > 0
        assert 0 <= results["survival_rate"] <= 1

    def test_calculate_ensemble_average_growth_rate(self, analyzer, multiple_trajectories):
        """Test ensemble average calculation for growth rates."""
        results = analyzer.calculate_ensemble_average(multiple_trajectories, metric="growth_rate")

        assert "mean" in results
        assert "std" in results
        assert "median" in results

        # Growth rate should be around 0.05
        assert 0.03 < results["mean"] < 0.07

    def test_calculate_ensemble_average_full_trajectory(self, analyzer, multiple_trajectories):
        """Test ensemble average calculation for full trajectories."""
        results = analyzer.calculate_ensemble_average(multiple_trajectories, metric="full")

        assert "mean_trajectory" in results
        assert "std_trajectory" in results

        # Check shapes
        assert results["mean_trajectory"].shape == (100,)
        assert results["std_trajectory"].shape == (100,)

    def test_check_convergence(self, analyzer):
        """Test convergence checking."""
        # Converged series (low variance)
        converged_values = np.random.normal(0.05, 0.001, 1000)
        converged, se = analyzer.check_convergence(converged_values)

        assert converged is True
        assert se < analyzer.convergence_threshold

        # Not converged series (high variance)
        divergent_values = np.random.normal(0.05, 0.5, 1000)
        converged, se = analyzer.check_convergence(divergent_values)

        assert converged is False
        assert se > analyzer.convergence_threshold

    def test_check_convergence_small_sample(self, analyzer):
        """Test convergence with insufficient data."""
        small_sample = np.array([1, 2, 3, 4, 5])
        converged, se = analyzer.check_convergence(small_sample, window_size=10)

        assert converged is False
        assert se == np.inf

    def test_compare_scenarios_with_arrays(self, analyzer):
        """Test scenario comparison with numpy arrays."""
        # Create insured trajectories (lower volatility)
        n_paths = 50
        n_time = 100

        insured = []
        for _ in range(n_paths):
            time = np.arange(n_time)
            growth = 0.04 + np.cumsum(np.random.normal(0, 0.05, n_time))
            values = 1000000 * np.exp(growth)
            insured.append(values)

        # Create uninsured trajectories (higher volatility, some failures)
        uninsured = []
        for i in range(n_paths):
            time = np.arange(n_time)
            growth = 0.05 + np.cumsum(np.random.normal(0, 0.15, n_time))
            values = 1000000 * np.exp(growth)
            # Add some failures
            if i % 10 == 0:
                values[50:] = 0
            uninsured.append(values)

        results = analyzer.compare_scenarios(
            np.array(insured), np.array(uninsured), metric="equity"
        )

        assert "insured" in results
        assert "uninsured" in results
        assert "ergodic_advantage" in results

        # Check structure
        assert "time_average_mean" in results["insured"]
        assert "ensemble_average" in results["insured"]
        assert "survival_rate" in results["insured"]

        # Insured should have better survival
        assert results["insured"]["survival_rate"] >= results["uninsured"]["survival_rate"]

    def test_compare_scenarios_with_simulation_results(self, analyzer):
        """Test scenario comparison with SimulationResults objects."""
        # Create mock SimulationResults
        insured_results = []
        uninsured_results = []

        for _ in range(10):
            # Mock insured result
            insured = SimulationResults(
                years=np.arange(100),
                assets=np.random.uniform(900000, 1100000, 100),
                equity=np.random.uniform(900000, 1100000, 100),
                roe=np.random.uniform(0.08, 0.12, 100),
                revenue=np.random.uniform(400000, 600000, 100),
                net_income=np.random.uniform(40000, 60000, 100),
                claim_counts=np.random.poisson(3, 100),
                claim_amounts=np.random.lognormal(10, 2, 100),
                insolvency_year=None,
            )
            insured_results.append(insured)

            # Mock uninsured result (some with insolvency)
            uninsured = SimulationResults(
                years=np.arange(100),
                assets=np.random.uniform(800000, 1200000, 100),
                equity=np.random.uniform(800000, 1200000, 100),
                roe=np.random.uniform(0.06, 0.14, 100),
                revenue=np.random.uniform(350000, 650000, 100),
                net_income=np.random.uniform(30000, 70000, 100),
                claim_counts=np.random.poisson(3, 100),
                claim_amounts=np.random.lognormal(10, 2, 100),
                insolvency_year=np.random.choice([None, 50, 75]),
            )
            uninsured_results.append(uninsured)

        results = analyzer.compare_scenarios(insured_results, uninsured_results, metric="equity")

        assert "ergodic_advantage" in results
        assert "t_statistic" in results["ergodic_advantage"]
        assert "p_value" in results["ergodic_advantage"]

    def test_significance_test(self, analyzer):
        """Test statistical significance testing."""
        # Create two samples with different means
        sample1 = np.random.normal(0.05, 0.01, 100)
        sample2 = np.random.normal(0.03, 0.01, 100)

        t_stat, p_value = analyzer.significance_test(sample1, sample2)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert p_value < 0.05  # Should be significant

        # Test with identical samples
        t_stat, p_value = analyzer.significance_test(sample1, sample1)
        assert p_value > 0.95  # Should not be significant

    def test_significance_test_with_nan(self, analyzer):
        """Test significance test with NaN values."""
        sample1 = np.array([0.05, 0.04, np.nan, 0.06, 0.05])
        sample2 = np.array([0.03, np.nan, 0.02, 0.04, 0.03])

        t_stat, p_value = analyzer.significance_test(sample1, sample2)

        # Should handle NaN values gracefully
        assert not np.isnan(t_stat)
        assert not np.isnan(p_value)

    def test_analyze_simulation_batch(self, analyzer):
        """Test batch analysis of simulation results."""
        # Create batch of results
        results = []
        for i in range(20):
            result = SimulationResults(
                years=np.arange(100),
                assets=1000000 * np.exp(0.05 * np.arange(100)),
                equity=1000000 * np.exp(0.045 * np.arange(100)),
                roe=np.random.uniform(0.08, 0.12, 100),
                revenue=np.random.uniform(400000, 600000, 100),
                net_income=np.random.uniform(40000, 60000, 100),
                claim_counts=np.random.poisson(3, 100),
                claim_amounts=np.random.lognormal(10, 2, 100),
                insolvency_year=None if i < 18 else 50,
            )
            results.append(result)

        analysis = analyzer.analyze_simulation_batch(results, label="Test Scenario")

        assert analysis["label"] == "Test Scenario"
        assert analysis["n_simulations"] == 20
        assert "time_average" in analysis
        assert "ensemble_average" in analysis
        assert "convergence" in analysis
        assert "survival_analysis" in analysis
        assert "ergodic_divergence" in analysis

        # Check survival rate (all should survive with positive exponential growth)
        assert (
            analysis["survival_analysis"]["survival_rate"] == 1.0
        )  # All survive with positive growth

    def test_edge_cases(self, analyzer):
        """Test various edge cases."""
        # Empty trajectory
        empty = np.array([])
        growth = analyzer.calculate_time_average_growth(empty)
        assert growth == -np.inf

        # Single value
        single = np.array([1000000])
        growth = analyzer.calculate_time_average_growth(single)
        assert growth == 0.0

        # All zeros
        zeros = np.zeros(100)
        growth = analyzer.calculate_time_average_growth(zeros)
        assert growth == -np.inf

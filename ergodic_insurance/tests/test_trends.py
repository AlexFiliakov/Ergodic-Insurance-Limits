"""Tests for trend module including stochastic trends.

Tests all trend implementations with focus on:
- Reproducibility with seeds
- Correct statistical properties
- Edge cases and validation
- Performance with caching
"""

import numpy as np
import pytest

from ergodic_insurance.trends import (
    LinearTrend,
    MeanRevertingTrend,
    NoTrend,
    RandomWalkTrend,
    RegimeSwitchingTrend,
    ScenarioTrend,
    Trend,
)


class TestNoTrend:
    """Test NoTrend implementation."""

    def test_always_returns_one(self):
        """NoTrend should always return 1.0."""
        trend = NoTrend()
        assert trend.get_multiplier(0) == 1.0
        assert trend.get_multiplier(1) == 1.0
        assert trend.get_multiplier(10) == 1.0
        assert trend.get_multiplier(-5) == 1.0
        assert trend.get_multiplier(0.5) == 1.0


class TestLinearTrend:
    """Test LinearTrend implementation."""

    def test_compound_growth(self):
        """Test compound growth calculation."""
        trend = LinearTrend(annual_rate=0.05)

        # Year 1: 1.05
        assert abs(trend.get_multiplier(1.0) - 1.05) < 1e-10

        # Year 2: 1.05^2 = 1.1025
        assert abs(trend.get_multiplier(2.0) - 1.1025) < 1e-10

        # Year 0.5: 1.05^0.5 ≈ 1.0247
        assert abs(trend.get_multiplier(0.5) - 1.05**0.5) < 1e-10

    def test_negative_rate(self):
        """Test decay with negative rate."""
        trend = LinearTrend(annual_rate=-0.02)

        # Year 1: 0.98
        assert abs(trend.get_multiplier(1.0) - 0.98) < 1e-10

        # Year 10: 0.98^10 ≈ 0.817
        assert abs(trend.get_multiplier(10.0) - 0.98**10) < 1e-10

    def test_negative_time(self):
        """Negative time should return 1.0."""
        trend = LinearTrend(annual_rate=0.05)
        assert trend.get_multiplier(-1.0) == 1.0
        assert trend.get_multiplier(-0.5) == 1.0


class TestScenarioTrend:
    """Test ScenarioTrend implementation."""

    def test_list_factors(self):
        """Test with list of factors."""
        trend = ScenarioTrend(factors=[1.0, 1.1, 1.2, 1.15], time_unit="annual")

        # Exact points
        assert trend.get_multiplier(0.0) == 1.0
        assert trend.get_multiplier(1.0) == 1.1
        assert trend.get_multiplier(2.0) == 1.2
        assert trend.get_multiplier(3.0) == 1.15

        # Linear interpolation
        assert abs(trend.get_multiplier(1.5) - 1.15) < 1e-10
        assert abs(trend.get_multiplier(2.5) - 1.175) < 1e-10

    def test_dict_factors(self):
        """Test with dictionary of factors."""
        trend = ScenarioTrend(factors={0: 1.0, 2: 1.1, 5: 1.3, 10: 1.5})

        # Exact points
        assert trend.get_multiplier(0.0) == 1.0
        assert trend.get_multiplier(2.0) == 1.1
        assert trend.get_multiplier(5.0) == 1.3

        # Linear interpolation between 2 and 5
        # At time 3.5: 1.1 + (1.3-1.1) * (3.5-2)/(5-2) = 1.2
        assert abs(trend.get_multiplier(3.5) - 1.2) < 1e-10

    def test_step_interpolation(self):
        """Test step function interpolation."""
        trend = ScenarioTrend(factors=[1.0, 1.2, 1.5], interpolation="step")

        # Before transition points
        assert trend.get_multiplier(0.5) == 1.0
        assert trend.get_multiplier(0.99) == 1.0

        # At and after transition
        assert trend.get_multiplier(1.0) == 1.2
        assert trend.get_multiplier(1.5) == 1.2
        assert trend.get_multiplier(1.99) == 1.2

    def test_monthly_factors(self):
        """Test monthly time unit."""
        trend = ScenarioTrend(factors=[1.0, 1.01, 1.02, 1.03], time_unit="monthly")

        # Month 0 (time 0)
        assert trend.get_multiplier(0.0) == 1.0

        # Month 1 (1/12 year)
        assert abs(trend.get_multiplier(1 / 12) - 1.01) < 1e-10

        # Month 2 (2/12 year)
        assert abs(trend.get_multiplier(2 / 12) - 1.02) < 1e-10

    def test_edge_cases(self):
        """Test edge cases."""
        trend = ScenarioTrend(factors=[1.0, 1.5])

        # Negative time
        assert trend.get_multiplier(-1.0) == 1.0

        # Beyond last point
        assert trend.get_multiplier(10.0) == 1.5

    def test_validation(self):
        """Test input validation."""
        # Invalid time unit
        with pytest.raises(ValueError, match="time_unit must be"):
            ScenarioTrend(factors=[1.0], time_unit="weekly")

        # Invalid interpolation
        with pytest.raises(ValueError, match="interpolation must be"):
            ScenarioTrend(factors=[1.0], interpolation="cubic")


class TestRandomWalkTrend:
    """Test RandomWalkTrend implementation."""

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        trend1 = RandomWalkTrend(drift=0.02, volatility=0.10, seed=42)
        trend2 = RandomWalkTrend(drift=0.02, volatility=0.10, seed=42)

        # Should produce identical results
        times = [0.5, 1.0, 2.0, 5.0, 10.0]
        for t in times:
            assert trend1.get_multiplier(t) == trend2.get_multiplier(t)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        trend1 = RandomWalkTrend(drift=0.0, volatility=0.20, seed=42)
        trend2 = RandomWalkTrend(drift=0.0, volatility=0.20, seed=43)

        # Should produce different results (with high probability)
        values1 = [trend1.get_multiplier(t) for t in [1, 2, 3, 4, 5]]
        values2 = [trend2.get_multiplier(t) for t in [1, 2, 3, 4, 5]]
        assert values1 != values2

    def test_drift_only(self):
        """Test with drift but no volatility."""
        trend = RandomWalkTrend(drift=0.05, volatility=0.0)

        # Should be deterministic: exp(drift * t)
        assert abs(trend.get_multiplier(1.0) - np.exp(0.05)) < 1e-10
        assert abs(trend.get_multiplier(2.0) - np.exp(0.10)) < 1e-10

    def test_volatility_only(self):
        """Test with volatility but no drift."""
        np.random.seed(42)
        trend = RandomWalkTrend(drift=0.0, volatility=0.15, seed=42)

        # Should have expected value around 1.0 but with variance
        values = [trend.get_multiplier(1.0) for _ in range(100)]

        # Reset and regenerate
        for _ in range(100):
            trend.reset_seed(np.random.randint(10000))
            values.append(trend.get_multiplier(1.0))

        # Mean should be close to exp(-0.5 * volatility^2 * t) ≈ 0.989
        expected_mean = np.exp(-0.5 * 0.15**2 * 1.0)
        assert abs(np.mean(values) - expected_mean) < 0.1

    def test_positive_multipliers(self):
        """Test that multipliers are always positive."""
        trend = RandomWalkTrend(drift=-0.1, volatility=0.3, seed=42)

        times = np.linspace(0, 20, 100)
        for t in times:
            mult = trend.get_multiplier(t)
            assert mult > 0, f"Multiplier at time {t} is not positive: {mult}"

    def test_path_caching(self):
        """Test that path is cached for efficiency."""
        trend = RandomWalkTrend(drift=0.02, volatility=0.10, seed=42)

        # First call generates path
        val1 = trend.get_multiplier(5.0)

        # Second call should use cached path
        val2 = trend.get_multiplier(3.0)

        # Going back to 5.0 should give same value
        val3 = trend.get_multiplier(5.0)
        assert val1 == val3

    def test_reset_seed(self):
        """Test seed reset functionality."""
        trend = RandomWalkTrend(drift=0.0, volatility=0.15, seed=42)

        # Get initial value
        val1 = trend.get_multiplier(5.0)

        # Reset with same seed
        trend.reset_seed(42)
        val2 = trend.get_multiplier(5.0)
        assert val1 == val2

        # Reset with different seed
        trend.reset_seed(100)
        val3 = trend.get_multiplier(5.0)
        assert val1 != val3

    def test_validation(self):
        """Test input validation."""
        # Negative volatility
        with pytest.raises(ValueError, match="volatility must be"):
            RandomWalkTrend(volatility=-0.1)

    def test_edge_cases(self):
        """Test edge cases."""
        trend = RandomWalkTrend(drift=0.05, volatility=0.10, seed=42)

        # Negative time
        assert trend.get_multiplier(-1.0) == 1.0

        # Zero time
        assert trend.get_multiplier(0.0) == 1.0


class TestMeanRevertingTrend:
    """Test MeanRevertingTrend implementation."""

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        trend1 = MeanRevertingTrend(mean_level=1.0, reversion_speed=0.5, volatility=0.1, seed=42)
        trend2 = MeanRevertingTrend(mean_level=1.0, reversion_speed=0.5, volatility=0.1, seed=42)

        times = [0.5, 1.0, 2.0, 5.0, 10.0]
        for t in times:
            assert trend1.get_multiplier(t) == trend2.get_multiplier(t)

    def test_mean_reversion(self):
        """Test that process reverts to mean over time."""
        # Start far from mean
        trend = MeanRevertingTrend(
            mean_level=1.0,
            reversion_speed=2.0,  # Very fast reversion
            volatility=0.001,  # Very low volatility
            initial_level=2.0,  # Start at 2x mean
            seed=42,
        )

        # Should converge toward mean_level
        val_0 = trend.get_multiplier(0.0)
        val_1 = trend.get_multiplier(1.0)
        val_5 = trend.get_multiplier(5.0)
        val_10 = trend.get_multiplier(10.0)
        val_20 = trend.get_multiplier(20.0)

        # Initial value should be 2.0
        assert val_0 == 2.0

        # Should move toward mean
        assert abs(val_1 - 1.0) < abs(val_0 - 1.0)

        # Long-term values should be closer to mean
        # (not necessarily monotonic due to randomness)
        assert abs(val_20 - 1.0) < 0.2

        # Average distance should decrease over time windows
        early_distance = abs(val_1 - 1.0)
        late_distance = abs(val_20 - 1.0)
        assert late_distance < early_distance

    def test_initial_level(self):
        """Test that initial level is respected."""
        trend = MeanRevertingTrend(mean_level=1.0, initial_level=1.5, seed=42)

        assert trend.get_multiplier(0.0) == 1.5

    def test_no_reversion(self):
        """Test with zero reversion speed."""
        trend = MeanRevertingTrend(
            mean_level=1.0, reversion_speed=0.0, volatility=0.0, initial_level=1.5
        )

        # Should stay at initial level
        assert trend.get_multiplier(1.0) == 1.5
        assert trend.get_multiplier(10.0) == 1.5

    def test_positive_multipliers(self):
        """Test that multipliers are always positive."""
        trend = MeanRevertingTrend(
            mean_level=0.5, reversion_speed=0.5, volatility=0.3, initial_level=2.0, seed=42
        )

        times = np.linspace(0, 20, 100)
        for t in times:
            mult = trend.get_multiplier(t)
            assert mult > 0, f"Multiplier at time {t} is not positive: {mult}"

    def test_reset_seed(self):
        """Test seed reset functionality."""
        trend = MeanRevertingTrend(mean_level=1.0, volatility=0.15, seed=42)

        val1 = trend.get_multiplier(5.0)

        trend.reset_seed(42)
        val2 = trend.get_multiplier(5.0)
        assert val1 == val2

        trend.reset_seed(100)
        val3 = trend.get_multiplier(5.0)
        assert val1 != val3

    def test_validation(self):
        """Test input validation."""
        # Negative reversion speed
        with pytest.raises(ValueError, match="reversion_speed must be"):
            MeanRevertingTrend(reversion_speed=-0.1)

        # Negative volatility
        with pytest.raises(ValueError, match="volatility must be"):
            MeanRevertingTrend(volatility=-0.1)

        # Non-positive mean level
        with pytest.raises(ValueError, match="mean_level must be"):
            MeanRevertingTrend(mean_level=0.0)

        # Non-positive initial level
        with pytest.raises(ValueError, match="initial_level must be"):
            MeanRevertingTrend(initial_level=-1.0)

    def test_edge_cases(self):
        """Test edge cases."""
        trend = MeanRevertingTrend(seed=42)

        # Negative time
        assert trend.get_multiplier(-1.0) == 1.0

        # Zero time
        assert trend.get_multiplier(0.0) == 1.0  # initial_level default


class TestRegimeSwitchingTrend:
    """Test RegimeSwitchingTrend implementation."""

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        trend1 = RegimeSwitchingTrend(
            regimes=[0.9, 1.2], transition_probs=[[0.7, 0.3], [0.4, 0.6]], seed=42
        )
        trend2 = RegimeSwitchingTrend(
            regimes=[0.9, 1.2], transition_probs=[[0.7, 0.3], [0.4, 0.6]], seed=42
        )

        times = [0.5, 1.0, 2.0, 5.0, 10.0]
        for t in times:
            assert trend1.get_multiplier(t) == trend2.get_multiplier(t)

    def test_regime_values(self):
        """Test that only regime values are returned."""
        regimes = [0.8, 1.0, 1.3]
        trend = RegimeSwitchingTrend(regimes=regimes, seed=42)

        # Sample many times
        times = np.linspace(0, 20, 200)
        values = [trend.get_multiplier(t) for t in times]

        # All values should be in regimes list
        unique_values = set(values)
        assert unique_values.issubset(set(regimes))

    def test_initial_regime(self):
        """Test that initial regime is respected."""
        trend = RegimeSwitchingTrend(regimes=[0.9, 1.0, 1.2], initial_regime=2, seed=42)

        assert trend.get_multiplier(0.0) == 1.2

    def test_transition_probabilities(self):
        """Test that transitions follow probabilities."""
        # Deterministic transition: always go to next state
        trend = RegimeSwitchingTrend(
            regimes=[1.0, 2.0, 3.0],
            transition_probs=[
                [0.0, 1.0, 0.0],  # 0 -> 1
                [0.0, 0.0, 1.0],  # 1 -> 2
                [1.0, 0.0, 0.0],  # 2 -> 0
            ],
            regime_persistence=0.1,  # Fast transitions
            initial_regime=0,
            seed=42,
        )

        # Should cycle through regimes
        vals = []
        for t in np.linspace(0, 3, 100):
            vals.append(trend.get_multiplier(t))

        # Should see all three regimes
        assert 1.0 in vals
        assert 2.0 in vals
        assert 3.0 in vals

    def test_regime_persistence(self):
        """Test that persistence affects regime duration."""
        # Low persistence (fast switching)
        trend_fast = RegimeSwitchingTrend(
            regimes=[0.9, 1.1],
            transition_probs=[[0.5, 0.5], [0.5, 0.5]],
            regime_persistence=0.1,
            seed=42,
        )

        # High persistence (slow switching)
        trend_slow = RegimeSwitchingTrend(
            regimes=[0.9, 1.1],
            transition_probs=[[0.5, 0.5], [0.5, 0.5]],
            regime_persistence=10.0,
            seed=42,
        )

        # Count transitions in fixed time window
        times = np.linspace(0, 10, 100)

        vals_fast = [trend_fast.get_multiplier(t) for t in times]
        vals_slow = [trend_slow.get_multiplier(t) for t in times]

        # Fast should have more transitions
        transitions_fast = sum(
            1 for i in range(1, len(vals_fast)) if vals_fast[i] != vals_fast[i - 1]
        )
        transitions_slow = sum(
            1 for i in range(1, len(vals_slow)) if vals_slow[i] != vals_slow[i - 1]
        )

        assert transitions_fast > transitions_slow

    def test_default_regimes(self):
        """Test default regime configuration."""
        trend = RegimeSwitchingTrend(seed=42)

        # Should have default regimes
        val = trend.get_multiplier(0.0)
        assert val in [0.9, 1.0, 1.2]

    def test_validation(self):
        """Test input validation."""
        # Transition probs don't sum to 1
        with pytest.raises(ValueError, match="must sum to 1"):
            RegimeSwitchingTrend(regimes=[1.0, 2.0], transition_probs=[[0.5, 0.4], [0.3, 0.7]])

        # Wrong matrix dimensions
        with pytest.raises(ValueError, match="rows.*must match regime count"):
            RegimeSwitchingTrend(regimes=[1.0, 2.0], transition_probs=[[0.5, 0.5]])

        # Invalid initial regime
        with pytest.raises(ValueError, match="initial_regime.*out of bounds"):
            RegimeSwitchingTrend(regimes=[1.0, 2.0], initial_regime=5)

        # Non-positive regime value
        with pytest.raises(ValueError, match="multiplier must be"):
            RegimeSwitchingTrend(regimes=[1.0, 0.0, 2.0])

    def test_edge_cases(self):
        """Test edge cases."""
        trend = RegimeSwitchingTrend(seed=42)

        # Negative time
        assert trend.get_multiplier(-1.0) == 1.0

    def test_reset_seed(self):
        """Test seed reset functionality."""
        trend = RegimeSwitchingTrend(regimes=[0.9, 1.1], seed=42)

        vals1 = [trend.get_multiplier(t) for t in range(10)]

        trend.reset_seed(42)
        vals2 = [trend.get_multiplier(t) for t in range(10)]
        assert vals1 == vals2

        trend.reset_seed(100)
        vals3 = [trend.get_multiplier(t) for t in range(10)]
        assert vals1 != vals3


class TestTrendIntegration:
    """Integration tests for trend combinations."""

    def test_all_trends_implement_interface(self):
        """Test that all trends implement the base interface."""
        trends = [
            NoTrend(),
            LinearTrend(0.03),
            ScenarioTrend([1.0, 1.1]),
            RandomWalkTrend(seed=42),
            MeanRevertingTrend(seed=42),
            RegimeSwitchingTrend(seed=42),
        ]

        for trend in trends:
            # Should be instance of base class
            assert isinstance(trend, Trend)

            # Should have get_multiplier method
            assert hasattr(trend, "get_multiplier")

            # Should return float
            result = trend.get_multiplier(1.0)
            assert isinstance(result, float)

            # Should handle negative time
            assert trend.get_multiplier(-1.0) == 1.0

    def test_stochastic_trends_seedable(self):
        """Test that stochastic trends are all seedable."""
        stochastic_trends = [
            RandomWalkTrend(seed=42),
            MeanRevertingTrend(seed=42),
            RegimeSwitchingTrend(seed=42),
        ]

        for trend in stochastic_trends:
            # Should have seed attribute
            assert hasattr(trend, "seed")
            assert trend.seed == 42

            # Should have reset_seed method
            assert hasattr(trend, "reset_seed")

            # Reset should work
            trend.reset_seed(100)
            assert trend.seed == 100


class TestPerformance:
    """Performance tests for trend implementations."""

    def test_caching_efficiency(self):
        """Test that caching improves performance."""
        trend = RandomWalkTrend(drift=0.02, volatility=0.10, seed=42)

        # First call generates path
        trend.get_multiplier(10.0)

        # These should all use cached path (fast)
        for t in np.linspace(0, 10, 100):
            trend.get_multiplier(t)

        # Should complete quickly due to caching
        # (This is more of a smoke test than a strict performance test)

    def test_long_simulations(self):
        """Test trends work for long time horizons."""
        trends = [
            LinearTrend(0.03),
            RandomWalkTrend(drift=0.02, seed=42),
            MeanRevertingTrend(mean_level=1.0, seed=42),
            RegimeSwitchingTrend(seed=42),
        ]

        for trend in trends:
            # Should handle long time horizons
            result = trend.get_multiplier(100.0)
            assert isinstance(result, float)
            assert result > 0  # Multipliers should stay positive

            # Should handle many queries efficiently
            times = np.linspace(0, 100, 1000)
            results = [trend.get_multiplier(t) for t in times]
            assert all(r > 0 for r in results)

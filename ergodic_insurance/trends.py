"""Trend module for insurance claim frequency and severity adjustments.

This module provides a hierarchy of trend classes that apply multiplicative
adjustments to claim frequencies and severities over time. Trends model
how insurance risks evolve due to inflation, exposure growth, regulatory
changes, or other systematic factors.

Key Concepts:
    - All trends are multiplicative (1.0 = no change, 1.03 = 3% increase)
    - Support both annual and sub-annual (monthly) time steps
    - Seedable for reproducibility in stochastic trends
    - Time-based multipliers for dynamic risk evolution

Example:
    Basic usage with linear trend::

        from ergodic_insurance.trends import LinearTrend, ScenarioTrend

        # 3% annual inflation trend
        inflation = LinearTrend(annual_rate=0.03)
        multiplier_year5 = inflation.get_multiplier(5.0)  # ~1.159

        # Custom scenario with varying rates
        scenario = ScenarioTrend(
            factors=[1.0, 1.05, 1.08, 1.06, 1.10],
            time_unit="annual"
        )
        multiplier_year3 = scenario.get_multiplier(3.5)  # Interpolated

Since:
    Version 0.4.0 - Core trend infrastructure for ClaimGenerator
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np


class Trend(ABC):
    """Abstract base class for all trend implementations.

    Defines the interface that all trend classes must implement. Trends
    provide multiplicative adjustments over time for frequencies and severities
    in insurance claim modeling.

    All trend implementations must provide:
        - get_multiplier(time): Returns multiplicative factor at given time
        - Proper handling of edge cases (negative time, etc.)
        - Reproducibility through seed support (if stochastic)

    Examples:
        Implementing a custom trend::

            class StepTrend(Trend):
                def __init__(self, step_time: float, step_factor: float):
                    self.step_time = step_time
                    self.step_factor = step_factor

                def get_multiplier(self, time: float) -> float:
                    if time < 0:
                        return 1.0
                    return 1.0 if time < self.step_time else self.step_factor
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize trend with optional random seed.

        Args:
            seed: Random seed for reproducibility in stochastic trends.
                  Ignored for deterministic trends but included for interface
                  consistency.
        """
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

    @abstractmethod
    def get_multiplier(self, time: float) -> float:
        """Get the multiplicative adjustment factor at a given time.

        Args:
            time: Time point (in years from start) to get multiplier for.
                  Can be fractional for sub-annual precision.

        Returns:
            float: Multiplicative factor (1.0 = no change, >1.0 = increase,
                   <1.0 = decrease).

        Note:
            Implementations should handle negative time gracefully, typically
            returning 1.0 or the initial value.
        """
        pass

    def reset_seed(self, seed: int) -> None:
        """Reset random seed for stochastic trends.

        Args:
            seed: New random seed to use.

        Note:
            This method allows re-running scenarios with different random
            outcomes while maintaining reproducibility.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)


class NoTrend(Trend):
    """Default trend implementation with no adjustment over time.

    This trend always returns a multiplier of 1.0, representing no change
    in frequency or severity over time. Useful as a default or baseline.

    Examples:
        Using NoTrend as baseline::

            from ergodic_insurance.trends import NoTrend

            baseline = NoTrend()

            # Always returns 1.0
            assert baseline.get_multiplier(0) == 1.0
            assert baseline.get_multiplier(10) == 1.0
            assert baseline.get_multiplier(-5) == 1.0
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize no-trend (constant multiplier of 1.0).

        Args:
            seed: Included for interface consistency but unused.
        """
        super().__init__(seed)

    def get_multiplier(self, time: float) -> float:
        """Return constant multiplier of 1.0.

        Args:
            time: Time point (ignored).

        Returns:
            float: Always returns 1.0.
        """
        return 1.0


class LinearTrend(Trend):
    """Linear compound growth trend with constant annual rate.

    Models exponential growth/decay with a fixed annual rate, similar to
    compound interest. Commonly used for inflation, exposure growth, or
    systematic risk changes.

    The multiplier at time t is calculated as: (1 + annual_rate)^t

    Attributes:
        annual_rate: Annual growth rate (0.03 = 3% growth, -0.02 = 2% decay).

    Examples:
        Modeling inflation::

            from ergodic_insurance.trends import LinearTrend

            # 3% annual inflation
            inflation = LinearTrend(annual_rate=0.03)

            # After 5 years: 1.03^5 ≈ 1.159
            mult_5y = inflation.get_multiplier(5.0)
            print(f"5-year inflation factor: {mult_5y:.3f}")

            # After 6 months: 1.03^0.5 ≈ 1.015
            mult_6m = inflation.get_multiplier(0.5)
            print(f"6-month inflation factor: {mult_6m:.3f}")

        Modeling exposure decay::

            # 2% annual exposure reduction
            reduction = LinearTrend(annual_rate=-0.02)
            mult_10y = reduction.get_multiplier(10.0)  # 0.98^10 ≈ 0.817
    """

    def __init__(self, annual_rate: float = 0.03, seed: Optional[int] = None):
        """Initialize linear trend with specified annual rate.

        Args:
            annual_rate: Annual growth rate. Positive for growth, negative
                        for decay. Default is 0.03 (3% growth).
            seed: Included for interface consistency but unused for this
                  deterministic trend.

        Examples:
            Various growth scenarios::

                # 5% annual growth
                growth = LinearTrend(0.05)

                # 1% annual decay
                decay = LinearTrend(-0.01)

                # No change (equivalent to NoTrend)
                flat = LinearTrend(0.0)
        """
        super().__init__(seed)
        self.annual_rate = annual_rate

    def get_multiplier(self, time: float) -> float:
        """Calculate compound growth multiplier at given time.

        Args:
            time: Time in years from start. Can be fractional for sub-annual
                  calculations. Negative times return 1.0.

        Returns:
            float: Multiplicative factor calculated as (1 + annual_rate)^time.
                   Returns 1.0 for negative times.

        Examples:
            Calculating multipliers::

                trend = LinearTrend(0.04)  # 4% annual

                # Year 1: 1.04
                mult_1 = trend.get_multiplier(1.0)

                # Year 2.5: 1.04^2.5 ≈ 1.104
                mult_2_5 = trend.get_multiplier(2.5)

                # Negative time: 1.0
                mult_neg = trend.get_multiplier(-1.0)
        """
        if time < 0:
            return 1.0

        return float((1 + self.annual_rate) ** time)


class ScenarioTrend(Trend):
    """Trend based on explicit scenario factors with interpolation.

    Allows specifying exact multiplicative factors at specific time points,
    with linear interpolation between points. Useful for modeling known
    future changes, regulatory impacts, or custom scenarios.

    Attributes:
        factors: List or dict of multiplicative factors.
        time_unit: Time unit for the factors ("annual" or "monthly").
        interpolation: Interpolation method ("linear" or "step").

    Examples:
        Annual scenario with known rates::

            from ergodic_insurance.trends import ScenarioTrend

            # Year 0: 1.0, Year 1: 1.05, Year 2: 1.08, etc.
            scenario = ScenarioTrend(
                factors=[1.0, 1.05, 1.08, 1.06, 1.10],
                time_unit="annual"
            )

            # Exact points
            mult_1 = scenario.get_multiplier(1.0)  # 1.05
            mult_2 = scenario.get_multiplier(2.0)  # 1.08

            # Interpolated
            mult_1_5 = scenario.get_multiplier(1.5)  # ≈1.065

        Monthly scenario::

            # Monthly adjustment factors
            monthly = ScenarioTrend(
                factors=[1.0, 1.01, 1.02, 1.015, 1.025, 1.03],
                time_unit="monthly"
            )

            # Month 3 (0.25 years)
            mult_3m = monthly.get_multiplier(0.25)

        Using dictionary for specific times::

            # Specific time points
            custom = ScenarioTrend(
                factors={0: 1.0, 2: 1.1, 5: 1.2, 10: 1.5},
                interpolation="linear"
            )
    """

    def __init__(
        self,
        factors: Union[List[float], Dict[float, float]],
        time_unit: str = "annual",
        interpolation: str = "linear",
        seed: Optional[int] = None,
    ):
        """Initialize scenario trend with explicit factors.

        Args:
            factors: Either:
                    - List[float]: Factors at regular intervals (0, 1, 2, ...)
                    - Dict[float, float]: {time: factor} pairs for irregular times
            time_unit: Time unit for the factors. Options:
                      - "annual": Factors apply at yearly intervals (default)
                      - "monthly": Factors apply at monthly intervals
            interpolation: Method for interpolating between points:
                         - "linear": Linear interpolation (default)
                         - "step": Step function (use previous value)
            seed: Included for interface consistency but unused.

        Raises:
            ValueError: If time_unit is not "annual" or "monthly", or if
                       interpolation is not "linear" or "step".

        Examples:
            Various scenario configurations::

                # Regular annual factors
                annual = ScenarioTrend([1.0, 1.1, 1.15, 1.18])

                # Irregular time points
                irregular = ScenarioTrend({
                    0: 1.0,
                    1.5: 1.08,
                    3: 1.15,
                    7: 1.25
                })

                # Monthly with step interpolation
                monthly_step = ScenarioTrend(
                    factors=[1.0, 1.02, 1.05],
                    time_unit="monthly",
                    interpolation="step"
                )
        """
        super().__init__(seed)

        # Validate inputs
        if time_unit not in ["annual", "monthly"]:
            raise ValueError(f"time_unit must be 'annual' or 'monthly', got {time_unit}")

        if interpolation not in ["linear", "step"]:
            raise ValueError(f"interpolation must be 'linear' or 'step', got {interpolation}")

        self.time_unit = time_unit
        self.interpolation = interpolation

        # Convert to internal representation: sorted (time, factor) pairs
        if isinstance(factors, list):
            # List assumed to be at regular intervals starting from 0
            time_scale = 1.0 if time_unit == "annual" else 1.0 / 12.0
            self.time_factors = [(i * time_scale, f) for i, f in enumerate(factors)]
        elif isinstance(factors, dict):
            # Dictionary with explicit time points
            self.time_factors = sorted(factors.items())
        else:
            raise TypeError("factors must be a list or dict")

        # Ensure we have at least one point
        if not self.time_factors:
            self.time_factors = [(0.0, 1.0)]

        # Extract times and factors for efficient interpolation
        self.times = np.array([t for t, _ in self.time_factors])
        self.factors_array = np.array([f for _, f in self.time_factors])

    def get_multiplier(self, time: float) -> float:
        """Get interpolated multiplier at given time.

        Args:
            time: Time in years from start. Can be fractional.
                  Negative times return 1.0.

        Returns:
            float: Multiplicative factor, interpolated from scenario points.
                   - Before first point: returns 1.0
                   - After last point: returns last factor
                   - Between points: interpolated based on method

        Examples:
            Interpolation behavior::

                scenario = ScenarioTrend([1.0, 1.1, 1.2, 1.15])

                # Exact points
                mult_0 = scenario.get_multiplier(0.0)  # 1.0
                mult_2 = scenario.get_multiplier(2.0)  # 1.2

                # Linear interpolation
                mult_1_5 = scenario.get_multiplier(1.5)  # 1.15

                # Beyond range
                mult_neg = scenario.get_multiplier(-1.0)  # 1.0
                mult_10 = scenario.get_multiplier(10.0)  # 1.15 (last)
        """
        if time < 0:
            return 1.0

        # Handle edge cases
        if time <= self.times[0]:
            return float(self.factors_array[0]) if time == self.times[0] else 1.0

        if time >= self.times[-1]:
            return float(self.factors_array[-1])

        # Interpolation
        if self.interpolation == "linear":
            # Linear interpolation using numpy
            return float(np.interp(time, self.times, self.factors_array))
        else:  # step
            # Find the largest time <= given time
            idx = np.searchsorted(self.times, time, side="right") - 1
            return float(self.factors_array[idx])

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


class RandomWalkTrend(Trend):
    """Random walk trend with drift and volatility.

    Models a geometric random walk (geometric Brownian motion) where the
    multiplier evolves as a cumulative product of random changes. Commonly
    used for modeling market indices, asset prices, or unpredictable long-term
    trends in insurance markets.

    The multiplier at time t follows: M(t) = exp(drift * t + volatility * W(t))
    where W(t) is a Brownian motion.

    Attributes:
        drift: Annual drift rate (expected growth rate).
        volatility: Annual volatility (standard deviation of log returns).
        cached_path: Cached random path for efficiency.
        cached_times: Time points for the cached path.

    Examples:
        Basic random walk with drift::

            from ergodic_insurance.trends import RandomWalkTrend

            # 2% drift with 10% volatility
            trend = RandomWalkTrend(drift=0.02, volatility=0.10, seed=42)

            # Generate multipliers
            mult_1 = trend.get_multiplier(1.0)  # Random around e^0.02
            mult_5 = trend.get_multiplier(5.0)  # More variation

        Market-like volatility::

            # High volatility market
            volatile = RandomWalkTrend(drift=0.0, volatility=0.30)

            # Low volatility with positive drift
            stable = RandomWalkTrend(drift=0.03, volatility=0.05)
    """

    def __init__(self, drift: float = 0.0, volatility: float = 0.10, seed: Optional[int] = None):
        """Initialize random walk trend.

        Args:
            drift: Annual drift rate (mu in geometric Brownian motion).
                  Can be positive (upward trend) or negative (downward).
                  Default is 0.0 (no drift).
            volatility: Annual volatility (sigma in geometric Brownian motion).
                       Must be >= 0. Default is 0.10 (10% annual volatility).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If volatility is negative.

        Examples:
            Various configurations::

                # Pure random walk (no drift)
                pure_rw = RandomWalkTrend(drift=0.0, volatility=0.15)

                # Positive drift with moderate volatility
                growth = RandomWalkTrend(drift=0.05, volatility=0.20)

                # Declining trend with low volatility
                decline = RandomWalkTrend(drift=-0.02, volatility=0.08)
        """
        super().__init__(seed)

        if volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {volatility}")

        self.drift = drift
        self.volatility = volatility
        self.cached_path: Optional[np.ndarray] = None
        self.cached_times: Optional[np.ndarray] = None

    def _generate_path(self, max_time: float, dt: float = 0.01) -> None:
        """Generate and cache a random path.

        Args:
            max_time: Maximum time to generate path for.
            dt: Time step for path generation.
        """
        num_steps = int(max_time / dt) + 1
        times = np.linspace(0, max_time, num_steps)

        # Generate Brownian motion increments
        if self.volatility > 0:
            dW = self.rng.randn(num_steps - 1) * np.sqrt(dt)
            W = np.concatenate([[0], np.cumsum(dW)])
        else:
            W = np.zeros(num_steps)

        # Geometric Brownian motion formula
        # M(t) = exp((drift - 0.5*volatility^2)*t + volatility*W(t))
        log_multipliers = (self.drift - 0.5 * self.volatility**2) * times + self.volatility * W

        self.cached_times = times
        self.cached_path = np.exp(log_multipliers)

    def get_multiplier(self, time: float) -> float:
        """Get random walk multiplier at given time.

        Args:
            time: Time in years from start. Negative times return 1.0.

        Returns:
            float: Multiplicative factor following geometric Brownian motion.
                  Always positive due to exponential transformation.

        Note:
            The path is cached on first call for efficiency. All subsequent
            calls will use the same random path, ensuring consistency within
            a simulation run.
        """
        if time < 0:
            return 1.0

        if time == 0:
            return 1.0

        # Generate or extend cached path if needed
        if self.cached_path is None or self.cached_times is None or time > self.cached_times[-1]:
            max_time = max(time * 1.5, 100.0)  # Generate extra for efficiency
            self._generate_path(max_time)

        # Interpolate from cached path
        assert self.cached_times is not None
        assert self.cached_path is not None
        return float(np.interp(time, self.cached_times, self.cached_path))

    def reset_seed(self, seed: int) -> None:
        """Reset random seed and clear cached path.

        Args:
            seed: New random seed to use.
        """
        super().reset_seed(seed)
        self.cached_path = None
        self.cached_times = None


class MeanRevertingTrend(Trend):
    """Mean-reverting trend using Ornstein-Uhlenbeck process.

    Models a trend that tends to revert to a long-term mean level, commonly
    used for interest rates, insurance market cycles, or any process with
    cyclical behavior around a stable level.

    The process follows: dX(t) = theta*(mu - X(t))*dt + sigma*dW(t)
    where the multiplier M(t) = exp(X(t))

    Attributes:
        mean_level: Long-term mean multiplier level.
        reversion_speed: Speed of mean reversion (theta).
        volatility: Volatility of the process (sigma).
        initial_level: Starting multiplier level.
        cached_path: Cached process path for efficiency.
        cached_times: Time points for the cached path.

    Examples:
        Insurance market cycle::

            from ergodic_insurance.trends import MeanRevertingTrend

            # Market cycles around 1.0 with 5-year half-life
            market = MeanRevertingTrend(
                mean_level=1.0,
                reversion_speed=0.14,  # ln(2)/5 years
                volatility=0.10,
                initial_level=1.1,  # Start in hard market
                seed=42
            )

            # Will gradually revert to 1.0
            mult_1 = market.get_multiplier(1.0)
            mult_10 = market.get_multiplier(10.0)  # Closer to 1.0

        Interest rate model::

            # Interest rates reverting to 3% with high volatility
            rates = MeanRevertingTrend(
                mean_level=1.03,
                reversion_speed=0.5,
                volatility=0.15
            )
    """

    def __init__(
        self,
        mean_level: float = 1.0,
        reversion_speed: float = 0.2,
        volatility: float = 0.10,
        initial_level: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize mean-reverting trend.

        Args:
            mean_level: Long-term mean multiplier level (mu).
                       Default is 1.0 (no long-term trend).
            reversion_speed: Speed of mean reversion (theta).
                            Higher values = faster reversion.
                            Default is 0.2 (moderate speed).
            volatility: Volatility of the process (sigma).
                       Default is 0.10 (10% volatility).
            initial_level: Starting multiplier level.
                          Default is 1.0.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If reversion_speed or volatility is negative,
                       or if mean_level or initial_level is not positive.

        Examples:
            Different mean reversion scenarios::

                # Fast reversion to high mean
                fast = MeanRevertingTrend(
                    mean_level=1.2,
                    reversion_speed=1.0,
                    volatility=0.05
                )

                # Slow reversion with high volatility
                slow = MeanRevertingTrend(
                    mean_level=0.9,
                    reversion_speed=0.05,
                    volatility=0.25
                )

                # Starting far from mean
                displaced = MeanRevertingTrend(
                    mean_level=1.0,
                    reversion_speed=0.3,
                    initial_level=1.5
                )
        """
        super().__init__(seed)

        if reversion_speed < 0:
            raise ValueError(f"reversion_speed must be >= 0, got {reversion_speed}")
        if volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {volatility}")
        if mean_level <= 0:
            raise ValueError(f"mean_level must be > 0, got {mean_level}")
        if initial_level <= 0:
            raise ValueError(f"initial_level must be > 0, got {initial_level}")

        self.mean_level = mean_level
        self.reversion_speed = reversion_speed
        self.volatility = volatility
        self.initial_level = initial_level
        self.cached_path: Optional[np.ndarray] = None
        self.cached_times: Optional[np.ndarray] = None

    def _generate_path(self, max_time: float, dt: float = 0.01) -> None:
        """Generate and cache an Ornstein-Uhlenbeck path.

        Args:
            max_time: Maximum time to generate path for.
            dt: Time step for path generation.
        """
        num_steps = int(max_time / dt) + 1
        times = np.linspace(0, max_time, num_steps)

        # Work in log space for positivity
        log_mean = np.log(self.mean_level)
        log_initial = np.log(self.initial_level)

        # Generate OU process in log space
        X = np.zeros(num_steps)
        X[0] = log_initial

        if self.volatility > 0 or self.reversion_speed > 0:
            for i in range(1, num_steps):
                dW = self.rng.randn() * np.sqrt(dt)
                X[i] = (
                    X[i - 1]
                    + self.reversion_speed * (log_mean - X[i - 1]) * dt
                    + self.volatility * dW
                )
        else:
            X[:] = log_initial

        # Convert to multipliers
        self.cached_times = times
        self.cached_path = np.exp(X)

    def get_multiplier(self, time: float) -> float:
        """Get mean-reverting multiplier at given time.

        Args:
            time: Time in years from start. Negative times return 1.0.

        Returns:
            float: Multiplicative factor following OU process.
                  Always positive. Tends toward mean_level over time.

        Note:
            The path is cached on first call for efficiency. The process
            exhibits mean reversion: starting values far from the mean will
            tend to move toward it over time.
        """
        if time < 0:
            return 1.0

        if time == 0:
            return float(self.initial_level)

        # Generate or extend cached path if needed
        if self.cached_path is None or self.cached_times is None or time > self.cached_times[-1]:
            max_time = max(time * 1.5, 100.0)
            self._generate_path(max_time)

        # Interpolate from cached path
        assert self.cached_times is not None
        assert self.cached_path is not None
        return float(np.interp(time, self.cached_times, self.cached_path))

    def reset_seed(self, seed: int) -> None:
        """Reset random seed and clear cached path.

        Args:
            seed: New random seed to use.
        """
        super().reset_seed(seed)
        self.cached_path = None
        self.cached_times = None


class RegimeSwitchingTrend(Trend):
    """Trend that switches between different market regimes.

    Models discrete regime changes such as hard/soft insurance markets,
    economic cycles, or regulatory environments. Each regime has its own
    multiplier, and transitions occur stochastically based on probabilities.

    Attributes:
        regimes: List of regime multipliers.
        transition_matrix: Matrix of transition probabilities between regimes.
        initial_regime: Starting regime index.
        regime_persistence: How long regimes tend to last.
        cached_regimes: Cached regime path for efficiency.
        cached_times: Time points for the cached path.

    Examples:
        Hard/soft insurance market::

            from ergodic_insurance.trends import RegimeSwitchingTrend

            # Two regimes: soft (0.9x) and hard (1.2x) markets
            market = RegimeSwitchingTrend(
                regimes=[0.9, 1.2],
                transition_probs=[[0.8, 0.2],   # Soft -> [80% stay, 20% to hard]
                                  [0.3, 0.7]],  # Hard -> [30% to soft, 70% stay]
                initial_regime=0,  # Start in soft market
                seed=42
            )

            # Multiplier switches between 0.9 and 1.2
            mult_5 = market.get_multiplier(5.0)

        Three-regime economic cycle::

            # Recession, normal, boom
            economy = RegimeSwitchingTrend(
                regimes=[0.8, 1.0, 1.3],
                transition_probs=[
                    [0.6, 0.4, 0.0],  # Recession
                    [0.1, 0.7, 0.2],  # Normal
                    [0.0, 0.5, 0.5],  # Boom
                ],
                initial_regime=1  # Start in normal
            )
    """

    def __init__(
        self,
        regimes: Optional[List[float]] = None,
        transition_probs: Optional[List[List[float]]] = None,
        initial_regime: int = 0,
        regime_persistence: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize regime-switching trend.

        Args:
            regimes: List of multipliers for each regime.
                    Default is [0.9, 1.0, 1.2] (soft/normal/hard).
            transition_probs: Transition probability matrix where
                            element [i][j] is P(regime j | regime i).
                            Rows must sum to 1.0.
                            Default creates persistent regimes.
            initial_regime: Starting regime index (0-based).
                          Default is 0 (first regime).
            regime_persistence: Time scaling factor. Higher values make
                              regimes last longer. Default is 1.0.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If transition probabilities don't sum to 1,
                       if matrix dimensions don't match regime count,
                       or if initial_regime is out of bounds.

        Examples:
            Custom regime configurations::

                # Simple two-state with equal transitions
                simple = RegimeSwitchingTrend(
                    regimes=[0.95, 1.05],
                    transition_probs=[[0.5, 0.5], [0.5, 0.5]]
                )

                # Highly persistent regimes
                persistent = RegimeSwitchingTrend(
                    regimes=[0.8, 1.2],
                    transition_probs=[[0.95, 0.05], [0.05, 0.95]],
                    regime_persistence=2.0  # Double persistence
                )

                # Four regulatory environments
                regulatory = RegimeSwitchingTrend(
                    regimes=[0.7, 0.9, 1.1, 1.3],
                    transition_probs=[
                        [0.7, 0.2, 0.1, 0.0],
                        [0.2, 0.5, 0.2, 0.1],
                        [0.1, 0.2, 0.5, 0.2],
                        [0.0, 0.1, 0.2, 0.7]
                    ]
                )
        """
        super().__init__(seed)

        # Default regimes: soft, normal, hard markets
        if regimes is None:
            regimes = [0.9, 1.0, 1.2]

        # Default transition matrix: persistent regimes
        if transition_probs is None:
            n = len(regimes)
            transition_probs = []
            for i in range(n):
                row = [0.1 / (n - 1)] * n  # Small prob to other states
                row[i] = 0.9  # High prob to stay
                transition_probs.append(row)

        # Validate inputs
        n_regimes = len(regimes)
        if len(transition_probs) != n_regimes:
            raise ValueError(
                f"transition_probs rows ({len(transition_probs)}) must match regime count ({n_regimes})"
            )

        for i, row in enumerate(transition_probs):
            if len(row) != n_regimes:
                raise ValueError(
                    f"transition_probs row {i} has {len(row)} elements, expected {n_regimes}"
                )
            row_sum = sum(row)
            if abs(row_sum - 1.0) > 1e-6:
                raise ValueError(f"transition_probs row {i} sums to {row_sum}, must sum to 1.0")

        if initial_regime < 0 or initial_regime >= n_regimes:
            raise ValueError(f"initial_regime {initial_regime} out of bounds [0, {n_regimes-1}]")

        for i, regime in enumerate(regimes):
            if regime <= 0:
                raise ValueError(f"regime {i} multiplier must be > 0, got {regime}")

        self.regimes = regimes
        self.transition_matrix = np.array(transition_probs)
        self.initial_regime = initial_regime
        self.regime_persistence = regime_persistence
        self.cached_regimes: Optional[np.ndarray] = None
        self.cached_times: Optional[np.ndarray] = None

    def _generate_regime_path(self, max_time: float, dt: float = 0.1) -> None:
        """Generate and cache a regime path.

        Args:
            max_time: Maximum time to generate path for.
            dt: Time step for regime transitions.
        """
        # Adjust dt by persistence factor
        effective_dt = dt * self.regime_persistence
        num_steps = int(max_time / effective_dt) + 1
        times = np.linspace(0, max_time, num_steps)

        # Generate regime sequence
        regimes = np.zeros(num_steps, dtype=int)
        regimes[0] = self.initial_regime

        for i in range(1, num_steps):
            current_regime = regimes[i - 1]
            probs = self.transition_matrix[current_regime]
            regimes[i] = self.rng.choice(len(self.regimes), p=probs)

        self.cached_times = times
        self.cached_regimes = regimes

    def get_multiplier(self, time: float) -> float:
        """Get regime-based multiplier at given time.

        Args:
            time: Time in years from start. Negative times return 1.0.

        Returns:
            float: Multiplicative factor for the active regime at time t.
                  Changes discretely as regimes switch.

        Note:
            The regime path is cached on first call. Regime changes are
            stochastic but reproducible with the same seed. The actual
            regime durations depend on both transition probabilities and
            the regime_persistence parameter.
        """
        if time < 0:
            return 1.0

        if time == 0:
            return float(self.regimes[self.initial_regime])

        # Generate or extend cached path if needed
        if self.cached_regimes is None or self.cached_times is None or time > self.cached_times[-1]:
            max_time = max(time * 1.5, 100.0)
            self._generate_regime_path(max_time)

        # Find regime at given time (step function, no interpolation)
        assert self.cached_times is not None
        assert self.cached_regimes is not None
        idx = int(np.searchsorted(self.cached_times, time, side="right")) - 1
        idx = max(0, min(idx, len(self.cached_regimes) - 1))
        regime_idx = self.cached_regimes[idx]

        return float(self.regimes[regime_idx])

    def reset_seed(self, seed: int) -> None:
        """Reset random seed and clear cached regime path.

        Args:
            seed: New random seed to use.
        """
        super().reset_seed(seed)
        self.cached_regimes = None
        self.cached_times = None


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

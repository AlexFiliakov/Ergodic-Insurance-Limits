"""Stochastic processes for financial modeling.

This module provides various stochastic process implementations for modeling
financial volatility, including Geometric Brownian Motion, lognormal volatility,
and mean-reverting processes. These are used to add realistic randomness to
revenue and growth modeling in the manufacturing simulation.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StochasticConfig(BaseModel):
    """Configuration for stochastic processes.

    Defines parameters common to all stochastic process implementations,
    including volatility, drift, random seed, and time step parameters.
    """

    volatility: float = Field(ge=0, le=2, description="Annual volatility (standard deviation)")
    drift: float = Field(ge=-1, le=1, description="Annual drift rate")
    random_seed: Optional[int] = Field(
        default=None, ge=0, description="Random seed for reproducibility"
    )
    time_step: float = Field(default=1.0, gt=0, le=1, description="Time step in years")


class StochasticProcess(ABC):
    """Abstract base class for stochastic processes.

    Provides common interface and functionality for all stochastic process
    implementations used in financial modeling. All concrete implementations
    must provide a generate_shock method.
    """

    def __init__(self, config: StochasticConfig):
        """Initialize the stochastic process.

        Args:
            config: Configuration for the stochastic process
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        logger.debug(f"Initialized {self.__class__.__name__} with seed={config.random_seed}")

    @abstractmethod
    def generate_shock(self, current_value: float) -> float:
        """Generate a stochastic shock for the current time step.

        Args:
            current_value: Current value of the process

        Returns:
            Multiplicative shock to apply to the value
        """
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator.

        Args:
            seed: Optional new seed to use
        """
        if seed is not None:
            self.config.random_seed = seed
        self.rng = np.random.RandomState(self.config.random_seed)
        logger.debug(f"Reset RNG with seed={self.config.random_seed}")


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion process using Euler-Maruyama discretization.

    Implements GBM with exact lognormal solution for high numerical accuracy.
    Commonly used for modeling asset prices and growth rates with constant
    relative volatility.
    """

    def generate_shock(self, current_value: float) -> float:
        """Generate a multiplicative shock using GBM.

        Uses the Euler-Maruyama discretization:
        dS = μ*S*dt + σ*S*dW

        Which gives multiplicative shock:
        S(t+dt)/S(t) = exp((μ - σ²/2)*dt + σ*√dt*Z)

        where Z ~ N(0,1)

        Args:
            current_value: Current value (not used in GBM, included for interface)

        Returns:
            Multiplicative shock factor
        """
        dt = self.config.time_step
        sigma = self.config.volatility
        mu = self.config.drift

        # Generate standard normal random variable
        z = self.rng.randn()

        # Calculate multiplicative shock
        # Using exact solution for lognormal
        shock = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

        logger.debug(f"GBM shock: {shock:.4f} (drift={mu:.3f}, vol={sigma:.3f}, z={z:.3f})")
        return float(shock)


class LognormalVolatility(StochasticProcess):
    """Simple lognormal volatility generator for revenue/sales.

    Provides simpler alternative to full GBM by applying lognormal shocks
    centered around 1.0. Suitable for modeling revenue variations without
    drift components.
    """

    def generate_shock(self, current_value: float) -> float:
        """Generate a lognormal multiplicative shock.

        Simpler than full GBM - just applies lognormal volatility around 1.0.
        Shock = exp(σ*Z) where Z ~ N(0,1)

        This gives E[shock] ≈ 1 for small σ (actually exp(σ²/2))

        Args:
            current_value: Current value (not used)

        Returns:
            Multiplicative shock factor centered around 1.0
        """
        sigma = self.config.volatility

        # Generate standard normal random variable
        z = self.rng.randn()

        # Simple lognormal shock
        # Mean-reverting around 1.0 with specified volatility
        shock = np.exp(sigma * z - 0.5 * sigma**2)

        logger.debug(f"Lognormal shock: {shock:.4f} (vol={sigma:.3f}, z={z:.3f})")
        return float(shock)


class MeanRevertingProcess(StochasticProcess):
    """Ornstein-Uhlenbeck mean-reverting process for bounded variables.

    Implements mean-reverting dynamics suitable for modeling variables that
    tend to revert to long-term average levels, such as operating margins
    or capacity utilization rates.
    """

    def __init__(
        self, config: StochasticConfig, mean_level: float = 1.0, reversion_speed: float = 0.5
    ):
        """Initialize mean-reverting process.

        Args:
            config: Base stochastic configuration
            mean_level: Long-term mean level to revert to
            reversion_speed: Speed of mean reversion (0=no reversion, 1=instant)
        """
        super().__init__(config)
        self.mean_level = mean_level
        self.reversion_speed = reversion_speed

    def generate_shock(self, current_value: float) -> float:
        """Generate mean-reverting shock.

        Uses Ornstein-Uhlenbeck process discretization:
        dx = θ*(μ - x)*dt + σ*dW

        Args:
            current_value: Current value of the process

        Returns:
            Additive shock (not multiplicative)
        """
        dt = self.config.time_step
        sigma = self.config.volatility
        theta = self.reversion_speed
        mu = self.mean_level

        # Generate standard normal random variable
        z = self.rng.randn()

        # Calculate new value using OU process
        mean_component = current_value + theta * (mu - current_value) * dt
        random_component = sigma * np.sqrt(dt) * z
        new_value = mean_component + random_component

        # Return as multiplicative shock
        shock = new_value / current_value if current_value != 0 else 1.0

        logger.debug(
            f"Mean-reverting shock: {shock:.4f} (current={current_value:.3f}, "
            f"target={mu:.3f}, speed={theta:.3f})"
        )
        return float(shock)


def create_stochastic_process(
    process_type: str,
    volatility: float,
    drift: float = 0.0,
    random_seed: Optional[int] = None,
    time_step: float = 1.0,
) -> StochasticProcess:
    """Factory function to create stochastic processes.

    Args:
        process_type: Type of process ("gbm", "lognormal", "mean_reverting")
        volatility: Annual volatility
        drift: Annual drift rate (for GBM)
        random_seed: Random seed for reproducibility
        time_step: Time step in years

    Returns:
        StochasticProcess instance

    Raises:
        ValueError: If process_type is not recognized
    """
    config = StochasticConfig(
        volatility=volatility, drift=drift, random_seed=random_seed, time_step=time_step
    )

    process_map = {
        "gbm": GeometricBrownianMotion,
        "geometric_brownian": GeometricBrownianMotion,
        "lognormal": LognormalVolatility,
        "mean_reverting": MeanRevertingProcess,
        "ornstein_uhlenbeck": MeanRevertingProcess,
    }

    process_type_lower = process_type.lower()
    if process_type_lower not in process_map:
        raise ValueError(
            f"Unknown process type: {process_type}. " f"Choose from: {list(process_map.keys())}"
        )

    process_class = process_map[process_type_lower]
    logger.info(f"Created {process_class.__name__} with volatility={volatility:.3f}")

    return process_class(config)  # type: ignore[no-any-return]

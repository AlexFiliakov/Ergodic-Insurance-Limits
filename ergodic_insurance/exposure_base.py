"""Exposure base module for dynamic frequency scaling in insurance claims.

This module provides a hierarchy of exposure classes that dynamically adjust
claim frequencies based on actual business metrics from the simulation.
The exposure bases now work with real financial state from the manufacturer,
not artificial growth projections.

Key Concepts:
    - Exposure bases query actual financial metrics from a state provider
    - Frequency multipliers are calculated from actual vs. base metrics
    - No artificial growth rates or projections
    - Direct integration with WidgetManufacturer financial state

Example:
    Basic usage with state-driven revenue exposure::

        from ergodic_insurance.exposure_base import RevenueExposure
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.claim_generator import ClaimGenerator

        # Create manufacturer
        manufacturer = WidgetManufacturer(config)

        # Create exposure linked to manufacturer's actual state
        exposure = RevenueExposure(state_provider=manufacturer)

        # Create generator with exposure
        generator = ClaimGenerator(
            base_frequency=0.5,
            exposure_base=exposure,
            severity_mean=1_000_000
        )

        # Claims will be generated based on actual revenue during simulation

Since:
    Version 0.3.0 - Complete refactor to state-driven approach
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FinancialStateProvider(Protocol):
    """Protocol for providing current financial state to exposure bases.

    This protocol defines the interface that any class must implement
    to provide financial metrics to exposure bases. The WidgetManufacturer
    class implements this protocol to supply real-time financial data.
    """

    @property
    def current_revenue(self) -> float:
        """Get current revenue."""
        ...

    @property
    def current_assets(self) -> float:
        """Get current total assets."""
        ...

    @property
    def current_equity(self) -> float:
        """Get current equity value."""
        ...

    @property
    def base_revenue(self) -> float:
        """Get base (initial) revenue for comparison."""
        ...

    @property
    def base_assets(self) -> float:
        """Get base (initial) assets for comparison."""
        ...

    @property
    def base_equity(self) -> float:
        """Get base (initial) equity for comparison."""
        ...


class ExposureBase(ABC):
    """Abstract base class for exposure calculations.

    Exposure represents the underlying business metric that drives claim frequency.
    Common examples include revenue, assets, employee count, or production volume.

    Subclasses must implement methods to calculate absolute exposure levels and
    frequency multipliers at different time points.
    """

    @abstractmethod
    def get_exposure(self, time: float) -> float:
        """Get absolute exposure level at given time.

        Args:
            time: Time in years from simulation start (can be fractional).

        Returns:
            float: Exposure level (e.g., revenue in dollars, asset value, etc.).
                Must be non-negative.
        """
        pass

    @abstractmethod
    def get_frequency_multiplier(self, time: float) -> float:
        """Get frequency adjustment factor relative to base.

        The multiplier is applied to the base frequency to determine the
        actual claim frequency at a given time.

        Args:
            time: Time in years from simulation start (can be fractional).

        Returns:
            float: Multiplier to apply to base frequency. A value of 1.0
                means no change from base frequency, 2.0 means double the
                base frequency, etc. Must be non-negative.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset exposure to initial state.

        This method should reset any internal state, cached values, or
        random number generators to their initial conditions. Useful for
        running multiple independent simulations with the same exposure
        configuration.
        """
        pass


@dataclass
class RevenueExposure(ExposureBase):
    """Revenue-based exposure using actual financial state.

    Models claim frequency that scales with actual business revenue from
    the simulation, not artificial growth projections. The exposure directly
    queries the current revenue from the manufacturer's financial state.

    Attributes:
        state_provider: Object providing current and base financial metrics.
            Typically a WidgetManufacturer instance.

    Example:
        Revenue exposure with actual manufacturer state::

            from ergodic_insurance.manufacturer import WidgetManufacturer
            from ergodic_insurance.config import ManufacturerConfig

            manufacturer = WidgetManufacturer(
                ManufacturerConfig(initial_assets=10_000_000)
            )
            exposure = RevenueExposure(state_provider=manufacturer)

            # Exposure reflects actual manufacturer revenue
            current_rev = exposure.get_exposure(1.0)
            multiplier = exposure.get_frequency_multiplier(1.0)
    """

    state_provider: FinancialStateProvider

    def get_exposure(self, time: float) -> float:
        """Return current actual revenue from manufacturer."""
        return self.state_provider.current_revenue

    def get_frequency_multiplier(self, time: float) -> float:
        """Calculate multiplier from actual revenue ratio."""
        if self.state_provider.base_revenue == 0:
            return 0.0
        return self.state_provider.current_revenue / self.state_provider.base_revenue

    def reset(self) -> None:
        """No internal state to reset for state-driven exposure."""
        pass


@dataclass
class AssetExposure(ExposureBase):
    """Asset-based exposure using actual financial state.

    Models claim frequency based on actual asset values from the simulation,
    tracking real asset changes from operations, claims, and business growth.
    Suitable for businesses where physical assets drive risk exposure.

    Frequency scales linearly with assets as more assets generally mean
    more insurable items that can generate claims.

    Attributes:
        state_provider: Object providing current and base financial metrics.
            Typically a WidgetManufacturer instance.

    Example:
        Asset exposure with actual manufacturer state::

            manufacturer = WidgetManufacturer(
                ManufacturerConfig(initial_assets=50_000_000)
            )
            exposure = AssetExposure(state_provider=manufacturer)

            # Exposure reflects actual asset changes
            current_assets = exposure.get_exposure(1.0)
            multiplier = exposure.get_frequency_multiplier(1.0)
    """

    state_provider: FinancialStateProvider

    def get_exposure(self, time: float) -> float:
        """Return current actual assets from manufacturer."""
        return self.state_provider.current_assets

    def get_frequency_multiplier(self, time: float) -> float:
        """Calculate multiplier from actual asset ratio."""
        if self.state_provider.base_assets == 0:
            return 0.0
        return self.state_provider.current_assets / self.state_provider.base_assets

    def reset(self) -> None:
        """No internal state to reset for state-driven exposure."""
        pass


@dataclass
class EquityExposure(ExposureBase):
    """Equity-based exposure using actual financial state.

    Models claim frequency based on actual equity values from the simulation,
    tracking real equity changes from profits, losses, and retained earnings.
    Suitable for financial analysis where equity represents business scale.

    Uses cube root scaling for conservative frequency adjustment, as equity
    growth doesn't directly translate to proportional risk increase.

    Attributes:
        state_provider: Object providing current and base financial metrics.
            Typically a WidgetManufacturer instance.

    Example:
        Equity exposure with actual manufacturer state::

            manufacturer = WidgetManufacturer(
                ManufacturerConfig(initial_assets=20_000_000)
            )
            exposure = EquityExposure(state_provider=manufacturer)

            # Exposure reflects actual equity changes
            current_equity = exposure.get_exposure(1.0)
            multiplier = exposure.get_frequency_multiplier(1.0)
    """

    state_provider: FinancialStateProvider

    def get_exposure(self, time: float) -> float:
        """Return current actual equity from manufacturer."""
        return self.state_provider.current_equity

    def get_frequency_multiplier(self, time: float) -> float:
        """Higher equity implies larger operations (cube root scaling for conservatism)."""
        if self.state_provider.base_equity == 0:
            return 0.0
        # Handle negative equity (bankruptcy) by returning 0
        if self.state_provider.current_equity <= 0:
            return 0.0
        ratio = self.state_provider.current_equity / self.state_provider.base_equity
        return float(ratio ** (1 / 3))

    def reset(self) -> None:
        """No internal state to reset for state-driven exposure."""
        pass


@dataclass
class EmployeeExposure(ExposureBase):
    """Exposure based on employee count.

    Models claim frequency based on workforce size, accounting for hiring
    and automation effects. Suitable for businesses where employee-related
    risks dominate (workers comp, employment practices, etc.).

    Attributes:
        base_employees: Initial number of employees.
        hiring_rate: Annual net hiring rate (can be negative for downsizing).
        automation_factor: Annual reduction in exposure per employee due to automation.

    Example:
        Employee exposure with automation::

            exposure = EmployeeExposure(
                base_employees=500,
                hiring_rate=0.05,  # 5% annual growth
                automation_factor=0.02  # 2% automation improvement
            )
    """

    base_employees: int
    hiring_rate: float = 0.0
    automation_factor: float = 0.0

    def __post_init__(self):
        """Validate inputs."""
        if self.base_employees < 0:
            raise ValueError(f"Base employees must be non-negative, got {self.base_employees}")
        if self.automation_factor < 0 or self.automation_factor > 1:
            raise ValueError(
                f"Automation factor must be between 0 and 1, got {self.automation_factor}"
            )

    def get_exposure(self, time: float) -> float:
        """Calculate employee count with hiring and automation effects."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")
        return float(self.base_employees * (1 + self.hiring_rate) ** time)

    def get_frequency_multiplier(self, time: float) -> float:
        """More employees = more workplace incidents, but automation helps."""
        if self.base_employees == 0:
            return 0.0
        current_employees = self.get_exposure(time)
        automation_reduction = (1 - self.automation_factor) ** time
        return float((current_employees / self.base_employees) * automation_reduction)

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class ProductionExposure(ExposureBase):
    """Exposure based on production volume/units.

    Models claim frequency based on production output, with support for
    seasonal patterns and quality improvements that reduce defect rates.

    Attributes:
        base_units: Initial production volume (units per year).
        growth_rate: Annual production growth rate.
        seasonality: Optional function returning seasonal multiplier.
        quality_improvement_rate: Annual reduction in defect-related claims.

    Example:
        Production exposure with seasonality::

            def seasonal_pattern(time):
                # Higher production in Q4
                return 1.0 + 0.3 * np.sin(2 * np.pi * time)

            exposure = ProductionExposure(
                base_units=100_000,
                growth_rate=0.08,
                seasonality=seasonal_pattern,
                quality_improvement_rate=0.03
            )
    """

    base_units: float
    growth_rate: float = 0.0
    seasonality: Optional[Callable[[float], float]] = None
    quality_improvement_rate: float = 0.0

    def __post_init__(self):
        """Validate inputs."""
        if self.base_units < 0:
            raise ValueError(f"Base units must be non-negative, got {self.base_units}")
        if self.quality_improvement_rate < 0 or self.quality_improvement_rate > 1:
            raise ValueError(
                f"Quality improvement rate must be between .0 and 1, got {self.quality_improvement_rate}"
            )

    def get_exposure(self, time: float) -> float:
        """Calculate production volume with growth and seasonality."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")

        base_production = self.base_units * (1 + self.growth_rate) ** time

        if self.seasonality:
            seasonal_factor = self.seasonality(time)
            base_production *= seasonal_factor

        return float(base_production)

    def get_frequency_multiplier(self, time: float) -> float:
        """More production = more potential defects, but quality improvements help."""
        if self.base_units == 0:
            return 0.0
        current_production = self.get_exposure(time)
        quality_factor = (1 - self.quality_improvement_rate) ** time
        return float((current_production / self.base_units) * quality_factor)

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class CompositeExposure(ExposureBase):
    """Weighted combination of multiple exposure bases.

    Allows modeling complex businesses with multiple risk drivers by
    combining different exposure types with specified weights.

    Attributes:
        exposures: Dictionary of named exposure bases.
        weights: Dictionary of weights for each exposure (will be normalized).

    Example:
        Composite exposure for diversified business::

            composite = CompositeExposure(
                exposures={
                    'revenue': RevenueExposure(base_revenue=50_000_000, growth_rate=0.05),
                    'assets': AssetExposure(base_assets=100_000_000),
                    'employees': EmployeeExposure(base_employees=500)
                },
                weights={'revenue': 0.5, 'assets': 0.3, 'employees': 0.2}
            )
    """

    exposures: Dict[str, ExposureBase]
    weights: Dict[str, float]

    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        if not self.exposures:
            raise ValueError("Must provide at least one exposure")
        if not self.weights:
            raise ValueError("Must provide weights")

        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Sum of weights must be positive")

        self.weights = {k: v / total for k, v in self.weights.items()}

    def get_exposure(self, time: float) -> float:
        """Weighted average of constituent exposures."""
        total = 0.0
        for name, exposure in self.exposures.items():
            weight = self.weights.get(name, 0.0)
            total += weight * exposure.get_exposure(time)
        return total

    def get_frequency_multiplier(self, time: float) -> float:
        """Weighted average of frequency multipliers."""
        total = 0.0
        for name, exposure in self.exposures.items():
            weight = self.weights.get(name, 0.0)
            total += weight * exposure.get_frequency_multiplier(time)
        return total

    def reset(self) -> None:
        """Reset all constituent exposures."""
        for exposure in self.exposures.values():
            exposure.reset()


@dataclass
class ScenarioExposure(ExposureBase):
    """Predefined exposure scenarios for planning and stress testing.

    Allows specification of exact exposure paths for scenario analysis,
    with interpolation between specified time points.

    Attributes:
        scenarios: Dictionary mapping scenario names to exposure paths.
        selected_scenario: Currently active scenario name.
        interpolation: Interpolation method ('linear', 'cubic', 'nearest').

    Example:
        Scenario-based exposure planning::

            scenarios = {
                'baseline': [100, 105, 110, 116, 122],
                'recession': [100, 95, 90, 92, 96],
                'expansion': [100, 112, 125, 140, 155]
            }

            exposure = ScenarioExposure(
                scenarios=scenarios,
                selected_scenario='recession',
                interpolation='linear'
            )
    """

    scenarios: Dict[str, List[float]]
    selected_scenario: str
    interpolation: str = "linear"
    _base_exposure: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate scenarios."""
        if self.selected_scenario not in self.scenarios:
            raise ValueError(
                f"Selected scenario '{self.selected_scenario}' not in available scenarios"
            )
        if self.interpolation not in ["linear", "cubic", "nearest"]:
            raise ValueError(
                f"Interpolation must be 'linear', 'cubic', or 'nearest', got '{self.interpolation}'"
            )
        self.reset()

    def get_exposure(self, time: float) -> float:
        """Interpolate exposure from scenario path."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")

        path = self.scenarios[self.selected_scenario]

        if time <= 0:
            return path[0]
        if time >= len(path) - 1:
            return path[-1]

        if self.interpolation == "nearest":
            return path[round(time)]
        elif self.interpolation == "linear":
            lower = int(time)
            upper = lower + 1
            weight = time - lower
            return path[lower] * (1 - weight) + path[upper] * weight
        else:  # cubic
            # Simple cubic interpolation (could use scipy for better implementation)
            return self._cubic_interpolate(path, time)

    def _cubic_interpolate(self, path: List[float], time: float) -> float:
        """Simple cubic interpolation implementation."""
        # For now, fall back to linear
        # A full implementation would use scipy.interpolate.interp1d
        lower = int(time)
        upper = lower + 1
        weight = time - lower
        return path[lower] * (1 - weight) + path[upper] * weight

    def get_frequency_multiplier(self, time: float) -> float:
        """Derive multiplier from exposure level."""
        if self._base_exposure is None or self._base_exposure == 0:
            return 1.0

        current = self.get_exposure(time)
        return current / self._base_exposure

    def reset(self) -> None:
        """Cache base exposure."""
        self._base_exposure = self.scenarios[self.selected_scenario][0]


@dataclass
class StochasticExposure(ExposureBase):
    """Stochastic exposure evolution using various processes.

    Supports multiple stochastic processes for advanced exposure modeling:
    - Geometric Brownian Motion (GBM)
    - Mean-reverting (Ornstein-Uhlenbeck)
    - Jump diffusion

    Attributes:
        base_value: Initial exposure value.
        process_type: Type of stochastic process ('gbm', 'mean_reverting', 'jump_diffusion').
        parameters: Process-specific parameters.
        seed: Random seed for reproducibility.

    Example:
        GBM exposure process::

            exposure = StochasticExposure(
                base_value=100_000_000,
                process_type='gbm',
                parameters={
                    'drift': 0.05,      # 5% drift
                    'volatility': 0.20  # 20% volatility
                },
                seed=42
            )
    """

    base_value: float
    process_type: str
    parameters: Dict[str, float]
    seed: Optional[int] = None
    _rng: Optional[np.random.RandomState] = field(default=None, init=False, repr=False)
    _path_cache: Dict[float, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize and validate."""
        if self.base_value < 0:
            raise ValueError(f"Base value must be non-negative, got {self.base_value}")
        if self.process_type not in ["gbm", "mean_reverting", "jump_diffusion"]:
            raise ValueError(f"Unknown process type: {self.process_type}")
        self._rng = np.random.RandomState(self.seed)
        self.reset()

    def reset(self):
        """Reset stochastic paths."""
        self._path_cache = {}
        self._rng = np.random.RandomState(self.seed)

    def get_exposure(self, time: float) -> float:
        """Generate or retrieve stochastic path."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")

        if time not in self._path_cache:
            if self.process_type == "gbm":
                self._generate_gbm_path(time)
            elif self.process_type == "mean_reverting":
                self._generate_ou_path(time)
            elif self.process_type == "jump_diffusion":
                self._generate_jump_diffusion_path(time)

        return self._path_cache.get(time, self.base_value)

    def _generate_gbm_path(self, time: float):
        """Generate Geometric Brownian Motion path."""
        if time == 0:
            self._path_cache[time] = self.base_value
            return

        mu = self.parameters.get("drift", 0.05)
        sigma = self.parameters.get("volatility", 0.2)

        # Use exact solution for GBM
        assert self._rng is not None  # Always initialized in __post_init__
        z = self._rng.standard_normal()
        value = self.base_value * np.exp((mu - 0.5 * sigma**2) * time + sigma * np.sqrt(time) * z)
        self._path_cache[time] = value

    def _generate_ou_path(self, time: float):
        """Generate Ornstein-Uhlenbeck (mean-reverting) path."""
        if time == 0:
            self._path_cache[time] = self.base_value
            return

        theta = self.parameters.get("mean_reversion_speed", 0.5)
        mu = self.parameters.get("long_term_mean", self.base_value)
        sigma = self.parameters.get("volatility", 0.2)

        # Exact solution for OU process
        exp_theta_t = np.exp(-theta * time)
        mean = mu + (self.base_value - mu) * exp_theta_t
        variance = (sigma**2 / (2 * theta)) * (1 - exp_theta_t**2)
        assert self._rng is not None  # Always initialized in __post_init__
        value = mean + np.sqrt(variance) * self._rng.standard_normal()
        self._path_cache[time] = max(0, value)  # Ensure non-negative

    def _generate_jump_diffusion_path(self, time: float):
        """Generate jump diffusion path."""
        if time == 0:
            self._path_cache[time] = self.base_value
            return

        mu = self.parameters.get("drift", 0.05)
        sigma = self.parameters.get("volatility", 0.2)
        jump_intensity = self.parameters.get("jump_intensity", 0.1)
        jump_mean = self.parameters.get("jump_mean", 0.0)
        jump_std = self.parameters.get("jump_std", 0.1)

        # GBM component
        assert self._rng is not None  # Always initialized in __post_init__
        gbm_value = self.base_value * np.exp(
            (mu - 0.5 * sigma**2) * time + sigma * np.sqrt(time) * self._rng.standard_normal()
        )

        # Jump component
        n_jumps = self._rng.poisson(jump_intensity * time)
        if n_jumps > 0:
            jumps = self._rng.normal(jump_mean, jump_std, n_jumps)
            jump_factor = np.exp(np.sum(jumps))
            gbm_value *= jump_factor

        self._path_cache[time] = gbm_value

    def get_frequency_multiplier(self, time: float) -> float:
        """Derive multiplier from exposure level."""
        if self.base_value == 0:
            return 0.0
        current = self.get_exposure(time)
        # Use square root scaling for stochastic exposure
        return float(np.sqrt(current / self.base_value))

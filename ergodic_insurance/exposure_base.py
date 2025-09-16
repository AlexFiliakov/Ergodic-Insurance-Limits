"""Exposure base module for dynamic frequency scaling in insurance claims.

This module provides a hierarchy of exposure classes that dynamically adjust
claim frequencies based on various business metrics like revenue, assets,
equity, employees, or production volume. The framework supports both
deterministic and stochastic exposure evolution.

Example:
    Basic usage with revenue exposure::

        from ergodic_insurance.exposure_base import RevenueExposure
        from ergodic_insurance.claim_generator import ClaimGenerator

        # Create exposure that grows with inflation
        exposure = RevenueExposure(
            base_revenue=50_000_000,
            growth_rate=0.03,
            inflation_rate=0.02,
            volatility=0.15
        )

        # Create generator with exposure
        generator = ClaimGenerator(
            base_frequency=0.5,
            exposure_base=exposure,
            severity_mean=1_000_000
        )

        # Generate claims - frequency scales with revenue growth
        claims = generator.generate_claims(years=20)

Since:
    Version 0.2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


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
    """Exposure based on revenue growth.

    Models claim frequency that scales with business revenue, supporting
    both deterministic growth and stochastic volatility through Geometric
    Brownian Motion.

    The frequency scaling uses square root of revenue ratio, which is an
    empirically observed relationship in the insurance industry where
    frequency doesn't scale linearly with size due to economies of scale
    and improved risk management.

    Attributes:
        base_revenue: Initial revenue level in dollars.
        growth_rate: Annual revenue growth rate (e.g., 0.05 for 5%).
        volatility: Revenue volatility for stochastic modeling (0 for deterministic).
        inflation_rate: Annual inflation rate to compound with growth.
        seed: Random seed for reproducible stochastic paths.

    Example:
        Revenue exposure with stochastic growth::

            exposure = RevenueExposure(
                base_revenue=10_000_000,
                growth_rate=0.10,  # 10% growth
                volatility=0.20,   # 20% volatility
                inflation_rate=0.02,  # 2% inflation
                seed=42
            )

            # Get exposure after 5 years
            revenue_5y = exposure.get_exposure(5.0)
            freq_mult_5y = exposure.get_frequency_multiplier(5.0)
    """

    base_revenue: float
    growth_rate: float = 0.0
    volatility: float = 0.0
    inflation_rate: float = 0.0
    seed: Optional[int] = None
    _rng: Optional[np.random.RandomState] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize random number generator."""
        if self.base_revenue < 0:
            raise ValueError(f"Base revenue must be non-negative, got {self.base_revenue}")
        self._rng = np.random.RandomState(self.seed)
        self.reset()

    def get_exposure(self, time: float) -> float:
        """Calculate revenue at time t with growth and inflation."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")

        # Combined growth and inflation
        deterministic_growth = (1 + self.growth_rate + self.inflation_rate) ** time

        if self.volatility > 0 and time > 0:
            # Geometric Brownian Motion
            assert self._rng is not None  # Always initialized in __post_init__
            drift = self.growth_rate - 0.5 * self.volatility**2
            diffusion = self.volatility * np.sqrt(time) * self._rng.standard_normal()
            stochastic_factor = np.exp(drift * time + diffusion)
            # Apply inflation separately from stochastic component
            return float(self.base_revenue * stochastic_factor * (1 + self.inflation_rate) ** time)

        return float(self.base_revenue * deterministic_growth)

    def get_frequency_multiplier(self, time: float) -> float:
        """Frequency scales with revenue."""
        if self.base_revenue == 0:
            return 0.0
        current_revenue = self.get_exposure(time)
        return current_revenue / self.base_revenue

    def reset(self) -> None:
        """Reset random number generator to initial state."""
        self._rng = np.random.RandomState(self.seed)


@dataclass
class AssetExposure(ExposureBase):
    """Exposure based on total assets.

    Models claim frequency based on asset base, accounting for growth,
    depreciation, and capital expenditures. Suitable for businesses where
    physical assets drive risk exposure (manufacturing, real estate, etc.).

    Frequency scales linearly with assets as more assets generally mean
    more insurable items that can generate claims.

    Attributes:
        base_assets: Initial asset value in dollars.
        growth_rate: Annual asset growth rate excluding depreciation.
        depreciation_rate: Annual depreciation rate (e.g., 0.05 for 5%).
        capex_schedule: Planned capital expenditures {time: amount}.
        inflation_rate: Annual inflation rate for asset values.

    Example:
        Asset exposure with depreciation and capex::

            exposure = AssetExposure(
                base_assets=50_000_000,
                growth_rate=0.03,
                depreciation_rate=0.10,
                capex_schedule={
                    2.0: 10_000_000,  # $10M investment at year 2
                    5.0: 15_000_000   # $15M investment at year 5
                }
            )
    """

    base_assets: float
    growth_rate: float = 0.0
    depreciation_rate: float = 0.02
    capex_schedule: Optional[Dict[float, float]] = None
    inflation_rate: float = 0.0

    def __post_init__(self):
        """Validate inputs."""
        if self.base_assets < 0:
            raise ValueError(f"Base assets must be non-negative, got {self.base_assets}")
        if self.depreciation_rate < 0 or self.depreciation_rate > 1:
            raise ValueError(
                f"Depreciation rate must be between 0 and 1, got {self.depreciation_rate}"
            )

    def get_exposure(self, time: float) -> float:
        """Calculate asset value considering growth, depreciation, and capex."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")

        # Base growth with depreciation
        net_growth_rate = self.growth_rate - self.depreciation_rate
        base_value = self.base_assets * (1 + net_growth_rate) ** time

        # Add scheduled capital expenditures
        if self.capex_schedule:
            for capex_time, amount in self.capex_schedule.items():
                if capex_time <= time:
                    years_since = time - capex_time
                    # Capex also depreciates
                    remaining_value = amount * (1 - self.depreciation_rate) ** years_since
                    base_value += remaining_value

        # Apply inflation
        return float(base_value * (1 + self.inflation_rate) ** time)

    def get_frequency_multiplier(self, time: float) -> float:
        """More assets = more things that can break (linear relationship)."""
        if self.base_assets == 0:
            return 0.0
        current_assets = self.get_exposure(time)
        return current_assets / self.base_assets

    def reset(self) -> None:
        """No state to reset for deterministic asset exposure."""
        pass


@dataclass
class EquityExposure(ExposureBase):
    """Exposure based on equity/market cap.

    Models claim frequency based on equity growth through retained earnings.
    Suitable for financial analysis where equity represents business scale.

    Uses cube root scaling for conservative frequency adjustment, as equity
    growth doesn't directly translate to proportional risk increase.

    Attributes:
        base_equity: Initial equity value in dollars.
        roe: Return on equity (e.g., 0.12 for 12% ROE).
        dividend_payout_ratio: Fraction of earnings paid as dividends.
        volatility: Market volatility for equity value.
        inflation_rate: Annual inflation rate.
        seed: Random seed for stochastic modeling.

    Example:
        Equity exposure with retained earnings growth::

            exposure = EquityExposure(
                base_equity=20_000_000,
                roe=0.15,  # 15% return on equity
                dividend_payout_ratio=0.40,  # 40% payout
                volatility=0.25  # 25% market volatility
            )
    """

    base_equity: float
    roe: float = 0.10
    dividend_payout_ratio: float = 0.3
    volatility: float = 0.0
    inflation_rate: float = 0.0
    seed: Optional[int] = None
    _rng: Optional[np.random.RandomState] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize and validate."""
        if self.base_equity < 0:
            raise ValueError(f"Base equity must be non-negative, got {self.base_equity}")
        if self.dividend_payout_ratio < 0 or self.dividend_payout_ratio > 1:
            raise ValueError(
                f"Payout ratio must be between 0 and 1, got {self.dividend_payout_ratio}"
            )
        self._rng = np.random.RandomState(self.seed)
        self.reset()

    def get_exposure(self, time: float) -> float:
        """Calculate equity value with retained earnings growth."""
        if time < 0:
            raise ValueError(f"Time must be non-negative, got {time}")

        retention_ratio = 1 - self.dividend_payout_ratio
        growth_rate = self.roe * retention_ratio

        base_growth = self.base_equity * (1 + growth_rate) ** time

        if self.volatility > 0 and time > 0:
            # Add market volatility
            assert self._rng is not None  # Always initialized in __post_init__
            shock = np.exp(self.volatility * np.sqrt(time) * self._rng.standard_normal())
            base_growth *= shock

        return float(base_growth * (1 + self.inflation_rate) ** time)

    def get_frequency_multiplier(self, time: float) -> float:
        """Higher equity implies larger operations (cube root scaling for conservatism)."""
        if self.base_equity == 0:
            return 0.0
        current_equity = self.get_exposure(time)
        return float((current_equity / self.base_equity) ** (1 / 3))

    def reset(self) -> None:
        """Reset random number generator."""
        self._rng = np.random.RandomState(self.seed)


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

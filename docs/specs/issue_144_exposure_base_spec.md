# Issue #144: ExposureBase Implementation Specification

## Overview
Enhance ClaimGenerator to support dynamic frequency scaling based on exposure metrics. This allows claim frequency to vary with business growth, economic conditions, and other exposure drivers.

## Problem Statement
Current ClaimGenerator uses a static frequency parameter that doesn't reflect real-world dynamics where claim frequency typically scales with business exposure (revenue, assets, employees, etc.). This limitation prevents accurate modeling of:
- Growing businesses with increasing claim exposure
- Economic cycles affecting claim patterns
- Scenario planning with varying exposure levels
- Stochastic exposure evolution

## Solution Architecture

### 1. New Module: `exposure_base.py`

#### Core Classes

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable
import numpy as np

class ExposureBase(ABC):
    """Abstract base class for exposure calculations.

    Exposure represents the underlying business metric that drives claim frequency.
    Common examples include revenue, assets, employee count, or production volume.
    """

    @abstractmethod
    def get_exposure(self, time: float) -> float:
        """Get absolute exposure level at given time.

        Args:
            time: Time in years from simulation start

        Returns:
            Exposure level (e.g., revenue in dollars, asset value, etc.)
        """
        pass

    @abstractmethod
    def get_frequency_multiplier(self, time: float) -> float:
        """Get frequency adjustment factor relative to base.

        Args:
            time: Time in years from simulation start

        Returns:
            Multiplier to apply to base frequency (1.0 = no change)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset exposure to initial state (for multiple simulations)."""
        pass
```

#### Concrete Implementations

```python
@dataclass
class RevenueExposure(ExposureBase):
    """Exposure based on revenue growth."""
    base_revenue: float
    growth_rate: float = 0.0  # Annual growth rate
    volatility: float = 0.0    # For stochastic growth
    inflation_rate: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        self.reset()

    def get_exposure(self, time: float) -> float:
        """Calculate revenue at time t with growth and inflation."""
        deterministic_growth = (1 + self.growth_rate + self.inflation_rate) ** time

        if self.volatility > 0:
            # Geometric Brownian Motion
            drift = self.growth_rate - 0.5 * self.volatility ** 2
            diffusion = self.volatility * np.sqrt(time) * self.rng.standard_normal()
            stochastic_factor = np.exp(drift * time + diffusion)
            return self.base_revenue * stochastic_factor

        return self.base_revenue * deterministic_growth

    def get_frequency_multiplier(self, time: float) -> float:
        """Frequency scales with square root of revenue (empirical relationship)."""
        current_revenue = self.get_exposure(time)
        return np.sqrt(current_revenue / self.base_revenue)

@dataclass
class AssetExposure(ExposureBase):
    """Exposure based on total assets."""
    base_assets: float
    growth_rate: float = 0.0
    depreciation_rate: float = 0.02  # Annual depreciation
    capex_schedule: Optional[Dict[float, float]] = None  # Time -> Investment
    inflation_rate: float = 0.0

    def get_exposure(self, time: float) -> float:
        """Calculate asset value considering growth, depreciation, and capex."""
        # Base growth with depreciation
        base_value = self.base_assets * (1 + self.growth_rate - self.depreciation_rate) ** time

        # Add scheduled capital expenditures
        if self.capex_schedule:
            for capex_time, amount in self.capex_schedule.items():
                if capex_time <= time:
                    years_since = time - capex_time
                    remaining_value = amount * (1 - self.depreciation_rate) ** years_since
                    base_value += remaining_value

        # Apply inflation
        return base_value * (1 + self.inflation_rate) ** time

    def get_frequency_multiplier(self, time: float) -> float:
        """More assets = more things that can break."""
        current_assets = self.get_exposure(time)
        return current_assets / self.base_assets

@dataclass
class EquityExposure(ExposureBase):
    """Exposure based on equity/market cap."""
    base_equity: float
    roe: float = 0.10  # Return on equity
    dividend_payout_ratio: float = 0.3
    volatility: float = 0.0
    inflation_rate: float = 0.0
    seed: Optional[int] = None

    def get_exposure(self, time: float) -> float:
        """Calculate equity value with retained earnings growth."""
        retention_ratio = 1 - self.dividend_payout_ratio
        growth_rate = self.roe * retention_ratio

        base_growth = self.base_equity * (1 + growth_rate) ** time

        if self.volatility > 0:
            # Add market volatility
            shock = np.exp(self.volatility * np.sqrt(time) * self.rng.standard_normal())
            base_growth *= shock

        return base_growth * (1 + self.inflation_rate) ** time

    def get_frequency_multiplier(self, time: float) -> float:
        """Higher equity implies larger operations."""
        current_equity = self.get_exposure(time)
        # Use cube root for more conservative scaling
        return (current_equity / self.base_equity) ** (1/3)

@dataclass
class EmployeeExposure(ExposureBase):
    """Exposure based on employee count."""
    base_employees: int
    hiring_rate: float = 0.0  # Annual net hiring rate
    automation_factor: float = 0.0  # Reduces exposure per employee over time

    def get_exposure(self, time: float) -> float:
        """Calculate employee count with hiring and automation effects."""
        employee_growth = self.base_employees * (1 + self.hiring_rate) ** time
        automation_reduction = (1 - self.automation_factor) ** time
        return employee_growth

    def get_frequency_multiplier(self, time: float) -> float:
        """More employees = more workplace incidents, but automation helps."""
        current_employees = self.get_exposure(time)
        automation_reduction = (1 - self.automation_factor) ** time
        return (current_employees / self.base_employees) * automation_reduction

@dataclass
class ProductionExposure(ExposureBase):
    """Exposure based on production volume/units."""
    base_units: float
    growth_rate: float = 0.0
    seasonality: Optional[Callable[[float], float]] = None
    quality_improvement_rate: float = 0.0  # Reduces defect-related claims

    def get_exposure(self, time: float) -> float:
        """Calculate production volume with growth and seasonality."""
        base_production = self.base_units * (1 + self.growth_rate) ** time

        if self.seasonality:
            seasonal_factor = self.seasonality(time)
            base_production *= seasonal_factor

        return base_production

    def get_frequency_multiplier(self, time: float) -> float:
        """More production = more potential defects, but quality improvements help."""
        current_production = self.get_exposure(time)
        quality_factor = (1 - self.quality_improvement_rate) ** time
        return (current_production / self.base_units) * quality_factor

@dataclass
class CompositeExposure(ExposureBase):
    """Weighted combination of multiple exposure bases."""
    exposures: Dict[str, ExposureBase]
    weights: Dict[str, float]

    def __post_init__(self):
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

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

@dataclass
class ScenarioExposure(ExposureBase):
    """Predefined exposure scenarios."""
    scenarios: Dict[str, List[float]]  # scenario_name -> [year0, year1, ...]
    selected_scenario: str
    interpolation: str = 'linear'  # or 'cubic', 'nearest'

    def get_exposure(self, time: float) -> float:
        """Interpolate exposure from scenario path."""
        path = self.scenarios[self.selected_scenario]

        if time <= 0:
            return path[0]
        if time >= len(path) - 1:
            return path[-1]

        if self.interpolation == 'nearest':
            return path[round(time)]
        elif self.interpolation == 'linear':
            lower = int(time)
            upper = lower + 1
            weight = time - lower
            return path[lower] * (1 - weight) + path[upper] * weight

    def get_frequency_multiplier(self, time: float) -> float:
        """Derive multiplier from exposure level."""
        if not hasattr(self, '_base_exposure'):
            self._base_exposure = self.scenarios[self.selected_scenario][0]

        current = self.get_exposure(time)
        return current / self._base_exposure if self._base_exposure > 0 else 1.0

@dataclass
class StochasticExposure(ExposureBase):
    """Stochastic exposure evolution using various processes."""
    base_value: float
    process_type: str  # 'gbm', 'mean_reverting', 'jump_diffusion'
    parameters: Dict[str, float]
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        self.reset()

    def reset(self):
        """Reset stochastic paths."""
        self._path_cache = {}

    def get_exposure(self, time: float) -> float:
        """Generate or retrieve stochastic path."""
        if time not in self._path_cache:
            if self.process_type == 'gbm':
                self._generate_gbm_path(time)
            elif self.process_type == 'mean_reverting':
                self._generate_ou_path(time)
            elif self.process_type == 'jump_diffusion':
                self._generate_jump_diffusion_path(time)

        return self._path_cache.get(time, self.base_value)

    def _generate_gbm_path(self, time: float):
        """Geometric Brownian Motion."""
        mu = self.parameters.get('drift', 0.05)
        sigma = self.parameters.get('volatility', 0.2)

        dt = 0.01  # Time step
        n_steps = int(time / dt)

        value = self.base_value
        for _ in range(n_steps):
            dW = self.rng.standard_normal() * np.sqrt(dt)
            value *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

        self._path_cache[time] = value
```

### 2. Enhanced ClaimGenerator Integration

```python
# In claim_generator.py

class ClaimGenerator:
    """Enhanced claim generator with exposure base support."""

    def __init__(
        self,
        frequency: Optional[float] = None,  # Deprecated
        base_frequency: Optional[float] = None,  # New parameter
        exposure_base: Optional[ExposureBase] = None,
        severity_mean: float = 5_000_000,
        severity_std: float = 2_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize with optional exposure base.

        Args:
            frequency: Legacy static frequency (deprecated)
            base_frequency: Base frequency at reference exposure level
            exposure_base: Dynamic exposure calculator
            severity_mean: Mean claim severity
            severity_std: Std dev of claim severity
            seed: Random seed
        """
        # Handle backward compatibility
        if frequency is not None and base_frequency is None:
            warnings.warn(
                "Parameter 'frequency' is deprecated. Use 'base_frequency' with exposure_base.",
                DeprecationWarning
            )
            base_frequency = frequency

        self.base_frequency = base_frequency or 0.1
        self.exposure_base = exposure_base
        self.severity_mean = severity_mean
        self.severity_std = severity_std
        self.rng = np.random.RandomState(seed)

    def get_adjusted_frequency(self, year: int) -> float:
        """Get frequency adjusted for exposure at given year."""
        if self.exposure_base is None:
            return self.base_frequency

        multiplier = self.exposure_base.get_frequency_multiplier(float(year))
        return self.base_frequency * multiplier

    def generate_year(self, year: int = 0) -> List[ClaimEvent]:
        """Generate claims for a single year with exposure adjustment."""
        adjusted_frequency = self.get_adjusted_frequency(year)

        # Rest of implementation remains similar
        n_claims = self.rng.poisson(adjusted_frequency)
        # ... generate claims
```

## Unit Tests Specification

### Test Structure

```python
# tests/test_exposure_base.py

class TestRevenueExposure:
    """Tests for revenue-based exposure."""

    def test_constant_revenue_no_growth(self):
        """Verify zero growth maintains base revenue."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.0)

        for t in [0, 1, 5, 10]:
            assert exposure.get_exposure(t) == 10_000_000
            assert exposure.get_frequency_multiplier(t) == 1.0

    def test_deterministic_growth(self):
        """Verify compound growth calculation."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)

        # After 1 year: 10M * 1.1 = 11M
        assert np.isclose(exposure.get_exposure(1), 11_000_000)

        # After 2 years: 10M * 1.1^2 = 12.1M
        assert np.isclose(exposure.get_exposure(2), 12_100_000)

        # Frequency multiplier should be sqrt(revenue_ratio)
        assert np.isclose(exposure.get_frequency_multiplier(1), np.sqrt(1.1))

    def test_inflation_adjustment(self):
        """Verify inflation compounds with growth."""
        exposure = RevenueExposure(
            base_revenue=10_000_000,
            growth_rate=0.05,
            inflation_rate=0.02
        )

        # Combined rate: 5% + 2% = 7%
        assert np.isclose(exposure.get_exposure(1), 10_700_000)

    def test_stochastic_growth_distribution(self):
        """Verify stochastic growth has expected properties."""
        exposure = RevenueExposure(
            base_revenue=10_000_000,
            growth_rate=0.10,
            volatility=0.20,
            seed=42
        )

        # Generate many paths
        values = [exposure.get_exposure(1) for _ in range(1000)]

        # Should have lognormal distribution
        # Expected value ≈ base * exp(growth_rate * time)
        expected = 10_000_000 * np.exp(0.10)
        assert np.abs(np.mean(values) - expected) / expected < 0.1

class TestAssetExposure:
    """Tests for asset-based exposure."""

    def test_depreciation(self):
        """Verify assets depreciate correctly."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            growth_rate=0.0,
            depreciation_rate=0.10
        )

        # After 1 year: 50M * 0.9 = 45M
        assert np.isclose(exposure.get_exposure(1), 45_000_000)

        # After 2 years: 50M * 0.9^2 = 40.5M
        assert np.isclose(exposure.get_exposure(2), 40_500_000)

    def test_capex_schedule(self):
        """Verify capital expenditures are added correctly."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            depreciation_rate=0.10,
            capex_schedule={
                1.0: 10_000_000,  # 10M investment at year 1
                3.0: 5_000_000    # 5M investment at year 3
            }
        )

        # At year 2: Original assets depreciated + Year 1 capex depreciated
        # 50M * 0.9^2 + 10M * 0.9 = 40.5M + 9M = 49.5M
        assert np.isclose(exposure.get_exposure(2), 49_500_000)

class TestEquityExposure:
    """Tests for equity-based exposure."""

    def test_retained_earnings_growth(self):
        """Verify equity grows through retained earnings."""
        exposure = EquityExposure(
            base_equity=20_000_000,
            roe=0.15,
            dividend_payout_ratio=0.40
        )

        # Growth rate = ROE * retention = 0.15 * 0.6 = 0.09
        # After 1 year: 20M * 1.09 = 21.8M
        assert np.isclose(exposure.get_exposure(1), 21_800_000)

    def test_conservative_scaling(self):
        """Verify frequency scales conservatively with equity."""
        exposure = EquityExposure(base_equity=20_000_000, roe=0.12)

        # Frequency uses cube root of equity ratio
        # If equity doubles, frequency multiplier = 2^(1/3) ≈ 1.26
        exposure.base_equity = 10_000_000  # Hack to test doubling
        assert np.isclose(exposure.get_frequency_multiplier(0), 2**(1/3))

class TestEmployeeExposure:
    """Tests for employee-based exposure."""

    def test_hiring_growth(self):
        """Verify employee count grows with hiring."""
        exposure = EmployeeExposure(
            base_employees=100,
            hiring_rate=0.10
        )

        # After 1 year: 100 * 1.1 = 110
        assert np.isclose(exposure.get_exposure(1), 110)

    def test_automation_reduction(self):
        """Verify automation reduces frequency multiplier."""
        exposure = EmployeeExposure(
            base_employees=100,
            hiring_rate=0.10,
            automation_factor=0.05
        )

        # Employees grow but automation reduces incident rate
        # Multiplier = (employee_ratio) * (1 - automation)^time
        assert exposure.get_frequency_multiplier(1) < 1.10  # Less than pure growth

class TestProductionExposure:
    """Tests for production volume exposure."""

    def test_seasonality(self):
        """Verify seasonal patterns apply correctly."""
        def seasonal_pattern(time):
            # Simple sinusoidal pattern
            return 1.0 + 0.3 * np.sin(2 * np.pi * time)

        exposure = ProductionExposure(
            base_units=1000,
            seasonality=seasonal_pattern
        )

        # At time 0.25 (quarter year), sin(π/2) = 1
        # Production = 1000 * (1 + 0.3 * 1) = 1300
        assert np.isclose(exposure.get_exposure(0.25), 1300)

    def test_quality_improvement(self):
        """Verify quality improvements reduce frequency."""
        exposure = ProductionExposure(
            base_units=1000,
            growth_rate=0.10,
            quality_improvement_rate=0.05
        )

        # Production grows but quality improvements offset frequency
        multiplier = exposure.get_frequency_multiplier(1)
        assert multiplier < 1.10  # Less than pure production growth

class TestCompositeExposure:
    """Tests for composite exposure combinations."""

    def test_weighted_combination(self):
        """Verify weighted averaging works correctly."""
        revenue_exp = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)
        asset_exp = AssetExposure(base_assets=50_000_000, growth_rate=0.05)

        composite = CompositeExposure(
            exposures={'revenue': revenue_exp, 'assets': asset_exp},
            weights={'revenue': 0.7, 'assets': 0.3}
        )

        # Weighted multiplier at t=1
        rev_mult = revenue_exp.get_frequency_multiplier(1)
        asset_mult = asset_exp.get_frequency_multiplier(1)
        expected = 0.7 * rev_mult + 0.3 * asset_mult

        assert np.isclose(composite.get_frequency_multiplier(1), expected)

class TestScenarioExposure:
    """Tests for scenario-based exposure."""

    def test_recession_scenario(self):
        """Verify recession scenario path."""
        scenarios = {
            'baseline': [100, 105, 110, 115, 120],
            'recession': [100, 95, 90, 92, 95],
            'boom': [100, 110, 125, 140, 160]
        }

        exposure = ScenarioExposure(
            scenarios=scenarios,
            selected_scenario='recession'
        )

        # At year 2, exposure should be 90
        assert exposure.get_exposure(2) == 90

        # Frequency multiplier = 90/100 = 0.9
        assert np.isclose(exposure.get_frequency_multiplier(2), 0.9)

    def test_interpolation(self):
        """Verify interpolation between years."""
        scenarios = {'test': [100, 110, 120]}

        exposure = ScenarioExposure(
            scenarios=scenarios,
            selected_scenario='test',
            interpolation='linear'
        )

        # At time 0.5, should be halfway between 100 and 110
        assert np.isclose(exposure.get_exposure(0.5), 105)

class TestStochasticExposure:
    """Tests for stochastic exposure processes."""

    def test_gbm_process(self):
        """Verify GBM properties."""
        exposure = StochasticExposure(
            base_value=100,
            process_type='gbm',
            parameters={'drift': 0.05, 'volatility': 0.20},
            seed=42
        )

        # Generate value at t=1
        value = exposure.get_exposure(1.0)

        # Should be positive
        assert value > 0

        # Should be reproducible with same seed
        exposure2 = StochasticExposure(
            base_value=100,
            process_type='gbm',
            parameters={'drift': 0.05, 'volatility': 0.20},
            seed=42
        )
        assert exposure2.get_exposure(1.0) == value

    def test_path_caching(self):
        """Verify paths are cached for consistency."""
        exposure = StochasticExposure(
            base_value=100,
            process_type='gbm',
            parameters={'drift': 0.05, 'volatility': 0.20},
            seed=42
        )

        # Multiple calls should return same value
        val1 = exposure.get_exposure(1.0)
        val2 = exposure.get_exposure(1.0)
        assert val1 == val2

class TestClaimGeneratorIntegration:
    """Integration tests for ClaimGenerator with ExposureBase."""

    def test_backward_compatibility(self):
        """Verify old interface still works."""
        gen = ClaimGenerator(frequency=0.5, seed=42)
        claims = gen.generate_claims(years=10)

        # Should generate claims as before
        assert len(claims) > 0

    def test_exposure_scaling(self):
        """Verify claims scale with exposure."""
        exposure = RevenueExposure(
            base_revenue=10_000_000,
            growth_rate=0.20  # 20% growth
        )

        gen = ClaimGenerator(
            base_frequency=1.0,
            exposure_base=exposure,
            seed=42
        )

        # Generate claims for different years
        claims_y0 = []
        claims_y5 = []

        for _ in range(100):  # Multiple simulations
            gen.reset_seed(42)
            claims_y0.extend(gen.generate_year(year=0))

            gen.reset_seed(42)
            claims_y5.extend(gen.generate_year(year=5))

        # Year 5 should have more claims due to growth
        assert len(claims_y5) > len(claims_y0)

    def test_zero_exposure_no_claims(self):
        """Verify zero exposure generates no claims."""
        exposure = RevenueExposure(base_revenue=0)

        gen = ClaimGenerator(
            base_frequency=1.0,
            exposure_base=exposure
        )

        claims = gen.generate_claims(years=10)
        assert len(claims) == 0

    def test_multiple_year_consistency(self):
        """Verify multi-year generation is consistent."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            growth_rate=0.05
        )

        gen = ClaimGenerator(
            base_frequency=0.5,
            exposure_base=exposure,
            seed=42
        )

        # Generate 10 years at once
        all_claims = gen.generate_claims(years=10)

        # Generate year by year
        gen.reset_seed(42)
        yearly_claims = []
        for year in range(10):
            yearly_claims.extend(gen.generate_year(year))

        # Should produce same total claims
        assert len(all_claims) == len(yearly_claims)

    def test_catastrophic_with_exposure(self):
        """Verify catastrophic claims work with exposure."""
        exposure = EquityExposure(
            base_equity=20_000_000,
            roe=0.15
        )

        gen = ClaimGenerator(
            base_frequency=0.1,
            exposure_base=exposure,
            seed=42
        )

        regular, cat = gen.generate_all_claims(
            years=50,
            include_catastrophic=True,
            cat_frequency=0.02
        )

        # Both types should be generated
        assert len(regular) > 0
        # Catastrophic might be empty due to randomness, but interface should work
        assert isinstance(cat, list)

class TestPerformance:
    """Performance and stress tests."""

    def test_large_simulation_performance(self):
        """Verify performance with large simulations."""
        import time

        exposure = CompositeExposure(
            exposures={
                'revenue': RevenueExposure(base_revenue=10_000_000, growth_rate=0.05),
                'assets': AssetExposure(base_assets=50_000_000),
                'employees': EmployeeExposure(base_employees=100, hiring_rate=0.03)
            },
            weights={'revenue': 0.5, 'assets': 0.3, 'employees': 0.2}
        )

        gen = ClaimGenerator(
            base_frequency=2.0,
            exposure_base=exposure,
            seed=42
        )

        start = time.time()
        claims = gen.generate_claims(years=1000)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max

        # Should generate expected number of claims
        assert len(claims) > 1500  # At least 1.5 per year average

    def test_memory_usage(self):
        """Verify memory usage is reasonable."""
        import sys

        exposure = StochasticExposure(
            base_value=100,
            process_type='gbm',
            parameters={'drift': 0.05, 'volatility': 0.20}
        )

        # Generate many paths
        for t in range(100):
            _ = exposure.get_exposure(float(t))

        # Check cache size is bounded
        cache_size = sys.getsizeof(exposure._path_cache)
        assert cache_size < 100_000  # Less than 100KB

class TestStatisticalValidation:
    """Statistical validation of exposure-adjusted frequencies."""

    def test_long_run_convergence(self):
        """Verify long-run averages converge to expected values."""
        exposure = RevenueExposure(
            base_revenue=10_000_000,
            growth_rate=0.10
        )

        gen = ClaimGenerator(
            base_frequency=1.0,
            exposure_base=exposure,
            seed=42
        )

        # Run many simulations
        total_claims = []
        for seed in range(100):
            gen.reset_seed(seed)
            claims = gen.generate_claims(years=10)
            total_claims.append(len(claims))

        # Average should be close to expected
        # Expected = sum(base_freq * sqrt(1.1^t) for t in range(10))
        expected_per_sim = sum(1.0 * np.sqrt(1.1**t) for t in range(10))
        actual_average = np.mean(total_claims)

        assert np.abs(actual_average - expected_per_sim) / expected_per_sim < 0.1

    def test_frequency_distribution(self):
        """Verify frequency follows Poisson distribution."""
        exposure = RevenueExposure(base_revenue=10_000_000)  # No growth

        gen = ClaimGenerator(
            base_frequency=2.0,
            exposure_base=exposure
        )

        # Generate many single-year samples
        counts = []
        for seed in range(1000):
            gen.reset_seed(seed)
            claims = gen.generate_year(0)
            counts.append(len(claims))

        # Should follow Poisson(2.0)
        mean_count = np.mean(counts)
        var_count = np.var(counts)

        # For Poisson, mean = variance
        assert np.abs(mean_count - 2.0) < 0.1
        assert np.abs(var_count - 2.0) < 0.3
```

## Migration Guide

1. Add ExposureBase implementation
2. Remove legacy frequency parameter
3. Update documentation and examples to use simple ExposureBases
4. Add Docstrings in the Google docstring style, providing implementation tips and simple examples
5. Update Sphinx documentation to autogenerate from new code files

## Example Usage

```python
# Simple revenue-based exposure
from ergodic_insurance.exposure_base import RevenueExposure
from ergodic_insurance.claim_generator import ClaimGenerator

# Create exposure that grows with inflation
exposure = RevenueExposure(
    base_revenue=50_000_000,
    growth_rate=0.03,
    inflation_rate=0.02,
    volatility=0.15  # Add randomness
)

# Create generator with exposure
generator = ClaimGenerator(
    base_frequency=0.5,  # Base: 0.5 claims/year at $50M revenue
    exposure_base=exposure,
    severity_mean=1_000_000,
    seed=42
)

# Generate claims - frequency will scale with revenue growth
claims = generator.generate_claims(years=20)

# Complex composite exposure
from ergodic_insurance.exposure_base import CompositeExposure, AssetExposure, EmployeeExposure

composite = CompositeExposure(
    exposures={
        'revenue': RevenueExposure(base_revenue=50_000_000, growth_rate=0.05),
        'assets': AssetExposure(base_assets=100_000_000, depreciation_rate=0.05),
        'employees': EmployeeExposure(base_employees=500, hiring_rate=0.03)
    },
    weights={'revenue': 0.5, 'assets': 0.3, 'employees': 0.2}
)

# Scenario-based planning
from ergodic_insurance.exposure_base import ScenarioExposure

scenarios = {
    'baseline': [100, 105, 110, 116, 122, 128],
    'recession': [100, 95, 90, 92, 96, 100],
    'expansion': [100, 112, 125, 140, 155, 170]
}

scenario_exposure = ScenarioExposure(
    scenarios=scenarios,
    selected_scenario='recession'
)
```

## Benefits

1. **Realistic Modeling**: Claims scale with business growth
2. **Flexibility**: Multiple exposure types for different industries
3. **Scenario Planning**: Test different growth/recession scenarios
4. **Stochastic Capability**: Model uncertainty in exposure evolution
5. **Backward Compatible**: Existing code continues to work
6. **Extensible**: Easy to add new exposure types

## Implementation Priority

1. Core ExposureBase abstract class
2. RevenueExposure (most common use case)
3. AssetExposure and EquityExposure
4. ClaimGenerator integration
5. ScenarioExposure for planning
6. StochasticExposure for advanced modeling
7. CompositeExposure for complex cases
8. EmployeeExposure and ProductionExposure

## Dependencies

- numpy (existing)
- scipy (for advanced stochastic processes)
- warnings (standard library)
- abc (standard library)
- dataclasses (standard library)

## Review Checklist

- [ ] ExposureBase interface is clear and extensible
- [ ] Legacy approach is eliminated
- [ ] All exposure types have meaningful frequency scaling
- [ ] Unit tests cover edge cases and statistical properties
- [ ] Performance is acceptable for large simulations
- [ ] Documentation includes migration guide
- [ ] Docstrings are in the style of Google Docstrings
- [ ] Examples demonstrate common use cases

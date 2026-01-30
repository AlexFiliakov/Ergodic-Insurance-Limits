# Exposure Base System Architecture

This document describes the state-driven exposure base system that dynamically calculates insurance exposures based on actual financial state rather than artificial projections.

## System Overview

```{mermaid}
graph TB
    %% State Provider
    subgraph Provider["State Provider"]
        STATE_PROVIDER["FinancialStateProvider<br/>(Protocol)"]
        MANUFACTURER["WidgetManufacturer<br/>(Implements Protocol)"]
    end

    %% Abstract Base
    subgraph Abstract["Abstract Layer"]
        EXPOSURE_BASE["ExposureBase<br/>(Abstract Base Class)"]
    end

    %% State-Driven Implementations
    subgraph StateDriven["State-Driven Exposures"]
        REVENUE["RevenueExposure<br/>Tracks actual revenue"]
        ASSET["AssetExposure<br/>Tracks actual assets"]
        EQUITY["EquityExposure<br/>Tracks actual equity"]
    end

    %% Parametric Implementations
    subgraph Parametric["Parametric Exposures"]
        EMPLOYEE["EmployeeExposure<br/>Headcount-based"]
        PRODUCTION["ProductionExposure<br/>Production volume"]
    end

    %% Composite and Advanced Implementations
    subgraph Advanced["Advanced Exposures"]
        COMPOSITE["CompositeExposure<br/>Weighted combination"]
        SCENARIO["ScenarioExposure<br/>Predefined paths"]
        STOCHASTIC["StochasticExposure<br/>Random evolution"]
    end

    %% Business Integration
    subgraph Business["Business Integration"]
        SIMULATION["Simulation"]
        INSURANCE["InsuranceProgram"]
        LOSS_GEN["LossGenerator"]
    end

    %% Relationships
    STATE_PROVIDER -.->|structural typing| MANUFACTURER
    MANUFACTURER --> REVENUE
    MANUFACTURER --> ASSET
    MANUFACTURER --> EQUITY

    EXPOSURE_BASE --> REVENUE
    EXPOSURE_BASE --> ASSET
    EXPOSURE_BASE --> EQUITY
    EXPOSURE_BASE --> EMPLOYEE
    EXPOSURE_BASE --> PRODUCTION
    EXPOSURE_BASE --> COMPOSITE
    EXPOSURE_BASE --> SCENARIO
    EXPOSURE_BASE --> STOCHASTIC

    COMPOSITE --> EXPOSURE_BASE

    SIMULATION --> LOSS_GEN
    LOSS_GEN --> EXPOSURE_BASE
    INSURANCE --> EXPOSURE_BASE

    %% Styling
    classDef protocol fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    classDef abstract fill:#f0f4f8,stroke:#5e72e4,stroke-width:2px
    classDef stateImpl fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef paramImpl fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef advImpl fill:#fce4ec,stroke:#c62828,stroke-width:2px
    classDef business fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    class STATE_PROVIDER protocol
    class EXPOSURE_BASE abstract
    class REVENUE,ASSET,EQUITY stateImpl
    class EMPLOYEE,PRODUCTION paramImpl
    class COMPOSITE,SCENARIO,STOCHASTIC advImpl
    class MANUFACTURER,SIMULATION,INSURANCE,LOSS_GEN business
```

## Inheritance Hierarchy

```{mermaid}
classDiagram
    class FinancialStateProvider {
        <<Protocol>>
        +current_revenue: Decimal
        +current_assets: Decimal
        +current_equity: Decimal
        +base_revenue: Decimal
        +base_assets: Decimal
        +base_equity: Decimal
    }

    class WidgetManufacturer {
        +current_revenue: Decimal
        +current_assets: Decimal
        +current_equity: Decimal
        +base_revenue: Decimal
        +base_assets: Decimal
        +base_equity: Decimal
        +step()
        +process_insurance_claim()
        +process_uninsured_claim()
    }

    class ExposureBase {
        <<ABC>>
        +get_exposure(time: float): float*
        +get_frequency_multiplier(time: float): float*
        +reset(): None*
    }

    class RevenueExposure {
        +state_provider: FinancialStateProvider
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class AssetExposure {
        +state_provider: FinancialStateProvider
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class EquityExposure {
        +state_provider: FinancialStateProvider
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class EmployeeExposure {
        +base_employees: int
        +hiring_rate: float
        +automation_factor: float
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class ProductionExposure {
        +base_units: float
        +growth_rate: float
        +seasonality: Optional~Callable~
        +quality_improvement_rate: float
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class CompositeExposure {
        +exposures: Dict~str, ExposureBase~
        +weights: Dict~str, float~
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class ScenarioExposure {
        +scenarios: Dict~str, List~
        +selected_scenario: str
        +interpolation: str
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    class StochasticExposure {
        +base_value: float
        +process_type: str
        +parameters: Dict~str, float~
        +seed: Optional~int~
        +get_exposure(time): float
        +get_frequency_multiplier(time): float
        +reset(): None
    }

    FinancialStateProvider <|.. WidgetManufacturer : implements
    ExposureBase <|-- RevenueExposure
    ExposureBase <|-- AssetExposure
    ExposureBase <|-- EquityExposure
    ExposureBase <|-- EmployeeExposure
    ExposureBase <|-- ProductionExposure
    ExposureBase <|-- CompositeExposure
    ExposureBase <|-- ScenarioExposure
    ExposureBase <|-- StochasticExposure
    RevenueExposure --> FinancialStateProvider : queries
    AssetExposure --> FinancialStateProvider : queries
    EquityExposure --> FinancialStateProvider : queries
    CompositeExposure o-- ExposureBase : contains
```

## State-Driven Architecture

The exposure base system queries real-time financial state from providers instead of using artificial growth projections:

```{mermaid}
sequenceDiagram
    participant Sim as Simulation
    participant Gen as LossGenerator
    participant Exp as ExposureBase
    participant Mfg as WidgetManufacturer

    loop Each Year
        Sim->>Gen: generate_year(year)
        Gen->>Exp: get_frequency_multiplier(time)
        Exp->>Mfg: Query current state
        Note over Mfg: Returns actual<br/>revenue/assets/equity<br/>as Decimal values
        Mfg-->>Exp: Current financial state
        Note over Exp: Converts Decimal to float<br/>at boundary
        Exp-->>Gen: Frequency multiplier
        Gen-->>Sim: Year's claims
        Sim->>Mfg: Process claims
        Sim->>Mfg: step() - Update financials
    end
```

## Financial State Provider Protocol

The `FinancialStateProvider` is a `@runtime_checkable` Protocol. Any class that defines the required properties satisfies the protocol through structural typing (duck typing) -- no explicit inheritance is needed.

`WidgetManufacturer` implements this protocol structurally with all six required properties.

```python
from typing import Protocol, runtime_checkable
from decimal import Decimal

@runtime_checkable
class FinancialStateProvider(Protocol):
    """Protocol for providing current financial state to exposure bases."""

    @property
    def current_revenue(self) -> Decimal:
        """Get current revenue."""
        ...

    @property
    def current_assets(self) -> Decimal:
        """Get current total assets."""
        ...

    @property
    def current_equity(self) -> Decimal:
        """Get current equity value."""
        ...

    @property
    def base_revenue(self) -> Decimal:
        """Get base (initial) revenue for comparison."""
        ...

    @property
    def base_assets(self) -> Decimal:
        """Get base (initial) assets for comparison."""
        ...

    @property
    def base_equity(self) -> Decimal:
        """Get base (initial) equity for comparison."""
        ...
```

## ExposureBase Abstract Class

All exposure classes extend `ExposureBase`, which defines three abstract methods:

```python
class ExposureBase(ABC):
    """Abstract base class for exposure calculations."""

    @abstractmethod
    def get_exposure(self, time: float) -> float:
        """Get absolute exposure level at given time."""
        pass

    @abstractmethod
    def get_frequency_multiplier(self, time: float) -> float:
        """Get frequency adjustment factor relative to base."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset exposure to initial state."""
        pass
```

## State-Driven Exposure Classes

These three classes query a `FinancialStateProvider` (typically `WidgetManufacturer`) for real-time financial metrics. All three are `@dataclass` types with a single field: `state_provider`.

### RevenueExposure

Tracks actual revenue performance. Uses **linear scaling** for the frequency multiplier:

```python
@dataclass
class RevenueExposure(ExposureBase):
    """Revenue-based exposure using actual financial state."""

    state_provider: FinancialStateProvider

    def get_exposure(self, time: float) -> float:
        """Return current actual revenue from manufacturer."""
        return float(self.state_provider.current_revenue)

    def get_frequency_multiplier(self, time: float) -> float:
        """Linear scaling: multiplier = current_revenue / base_revenue."""
        if self.state_provider.base_revenue == 0:
            return 0.0
        if self.state_provider.current_revenue <= 0:
            return 0.0
        return float(
            self.state_provider.current_revenue / self.state_provider.base_revenue
        )

    def reset(self) -> None:
        """No internal state to reset for state-driven exposure."""
        pass
```

### AssetExposure

Tracks actual asset base. Uses **linear scaling** for the frequency multiplier:

```python
@dataclass
class AssetExposure(ExposureBase):
    """Asset-based exposure using actual financial state."""

    state_provider: FinancialStateProvider

    def get_exposure(self, time: float) -> float:
        """Return current actual assets from manufacturer."""
        return float(self.state_provider.current_assets)

    def get_frequency_multiplier(self, time: float) -> float:
        """Linear scaling: multiplier = current_assets / base_assets."""
        if self.state_provider.base_assets == 0:
            return 0.0
        if self.state_provider.current_assets <= 0:
            return 0.0
        return float(
            self.state_provider.current_assets / self.state_provider.base_assets
        )

    def reset(self) -> None:
        """No internal state to reset for state-driven exposure."""
        pass
```

### EquityExposure

Tracks actual equity position. Uses **linear scaling** for the frequency multiplier:

```python
@dataclass
class EquityExposure(ExposureBase):
    """Equity-based exposure using actual financial state."""

    state_provider: FinancialStateProvider

    def get_exposure(self, time: float) -> float:
        """Return current actual equity from manufacturer."""
        return float(self.state_provider.current_equity)

    def get_frequency_multiplier(self, time: float) -> float:
        """Linear scaling: multiplier = current_equity / base_equity."""
        if self.state_provider.base_equity == 0:
            return 0.0
        if self.state_provider.current_equity <= 0:
            return 0.0  # No exposure when insolvent
        ratio = self.state_provider.current_equity / self.state_provider.base_equity
        return float(ratio)

    def reset(self) -> None:
        """No internal state to reset for state-driven exposure."""
        pass
```

## Parametric Exposure Classes

These classes use fixed parameters and growth formulas rather than querying a state provider.

### EmployeeExposure

Models claim frequency based on workforce size with hiring growth and automation effects:

```python
@dataclass
class EmployeeExposure(ExposureBase):
    """Exposure based on employee count."""

    base_employees: int
    hiring_rate: float = 0.0
    automation_factor: float = 0.0  # Must be between 0 and 1

    def get_exposure(self, time: float) -> float:
        """Employee count with compound hiring growth."""
        return float(self.base_employees * (1 + self.hiring_rate) ** time)

    def get_frequency_multiplier(self, time: float) -> float:
        """Growth adjusted by automation: (employees/base) * (1-automation)^time."""
        if self.base_employees == 0:
            return 0.0
        current_employees = self.get_exposure(time)
        automation_reduction = (1 - self.automation_factor) ** time
        return float((current_employees / self.base_employees) * automation_reduction)

    def reset(self) -> None:
        pass
```

### ProductionExposure

Models claim frequency based on production volume with optional seasonality and quality improvement:

```python
@dataclass
class ProductionExposure(ExposureBase):
    """Exposure based on production volume/units."""

    base_units: float
    growth_rate: float = 0.0
    seasonality: Optional[Callable[[float], float]] = None
    quality_improvement_rate: float = 0.0  # Must be between 0 and 1

    def get_exposure(self, time: float) -> float:
        """Production volume with growth and optional seasonality."""
        base_production = self.base_units * (1 + self.growth_rate) ** time
        if self.seasonality:
            base_production *= self.seasonality(time)
        return float(base_production)

    def get_frequency_multiplier(self, time: float) -> float:
        """Growth adjusted by quality: (production/base) * (1-quality_rate)^time."""
        if self.base_units == 0:
            return 0.0
        current_production = self.get_exposure(time)
        quality_factor = (1 - self.quality_improvement_rate) ** time
        return float((current_production / self.base_units) * quality_factor)

    def reset(self) -> None:
        pass
```

## Advanced Exposure Classes

### CompositeExposure

Weighted combination of multiple exposure bases. Weights are automatically normalized to sum to 1.0 during initialization:

```python
@dataclass
class CompositeExposure(ExposureBase):
    """Weighted combination of multiple exposure bases."""

    exposures: Dict[str, ExposureBase]
    weights: Dict[str, float]  # Normalized to sum to 1.0 in __post_init__

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
            total += weight * float(exposure.get_frequency_multiplier(time))
        return total

    def reset(self) -> None:
        """Reset all constituent exposures."""
        for exposure in self.exposures.values():
            exposure.reset()
```

### ScenarioExposure

Predefined exposure paths with interpolation for planning and stress testing:

```python
@dataclass
class ScenarioExposure(ExposureBase):
    """Predefined exposure scenarios for planning and stress testing."""

    scenarios: Dict[str, List[float]]
    selected_scenario: str
    interpolation: str = "linear"  # 'linear', 'cubic', or 'nearest'

    def get_exposure(self, time: float) -> float:
        """Interpolate exposure from scenario path."""
        # Returns first value if time <= 0, last value if time >= len-1
        # Interpolates between points otherwise
        ...

    def get_frequency_multiplier(self, time: float) -> float:
        """Multiplier = current_exposure / base_exposure (first scenario value)."""
        ...

    def reset(self) -> None:
        """Cache base exposure from first scenario value."""
        self._base_exposure = self.scenarios[self.selected_scenario][0]
```

### StochasticExposure

Stochastic exposure evolution supporting multiple random processes. Uses **square root scaling** for the frequency multiplier:

```python
@dataclass
class StochasticExposure(ExposureBase):
    """Stochastic exposure evolution using various processes."""

    base_value: float
    process_type: str  # 'gbm', 'mean_reverting', or 'jump_diffusion'
    parameters: Dict[str, float]
    seed: Optional[int] = None

    def get_exposure(self, time: float) -> float:
        """Generate or retrieve stochastic path value (cached)."""
        ...

    def get_frequency_multiplier(self, time: float) -> float:
        """Square root scaling: sqrt(current / base_value)."""
        if self.base_value == 0:
            return 0.0
        current = self.get_exposure(time)
        return float(np.sqrt(current / self.base_value))

    def reset(self) -> None:
        """Clear path cache and reinitialize RNG from seed."""
        self._path_cache = {}
        self._rng = np.random.default_rng(self.seed)
```

**Supported stochastic processes:**

| Process | Key Parameters | Description |
|---------|---------------|-------------|
| `gbm` | `drift`, `volatility` | Geometric Brownian Motion -- exact solution |
| `mean_reverting` | `mean_reversion_speed`, `long_term_mean`, `volatility` | Ornstein-Uhlenbeck process |
| `jump_diffusion` | `drift`, `volatility`, `jump_intensity`, `jump_mean`, `jump_std` | GBM with Poisson jumps |

## Frequency Scaling Summary

| Exposure Class | Scaling Formula | Description |
|----------------|----------------|-------------|
| **RevenueExposure** | `current / base` | Linear with revenue ratio |
| **AssetExposure** | `current / base` | Linear with asset ratio |
| **EquityExposure** | `current / base` | Linear with equity ratio |
| **EmployeeExposure** | `(employees / base) * (1 - automation)^t` | Growth with automation offset |
| **ProductionExposure** | `(production / base) * (1 - quality_rate)^t` | Growth with quality offset |
| **CompositeExposure** | `sum(weight_i * multiplier_i)` | Weighted average of components |
| **ScenarioExposure** | `current_path_value / first_path_value` | Ratio to scenario start |
| **StochasticExposure** | `sqrt(current / base)` | Square root of value ratio |

## Usage Examples

### Basic Setup

```python
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.exposure_base import RevenueExposure
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

# Create manufacturer with initial state
config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.12,
    tax_rate=0.25,
    retention_ratio=0.7
)
manufacturer = WidgetManufacturer(config)

# Create state-driven exposure
exposure = RevenueExposure(state_provider=manufacturer)

# Create loss generator
generator = ManufacturingLossGenerator.create_simple(
    frequency=2.0,
    severity_mean=100_000,
    severity_std=50_000,
    seed=42
)
```

### Dynamic Loss Generation

```python
from ergodic_insurance.simulation import Simulation

# Simulation generates losses year-by-year
simulation = Simulation(
    manufacturer=manufacturer,
    loss_generator=generator,
    time_horizon=10
)

# Run simulation - claims adapt to actual business state
for year in range(10):
    # Generate claims based on current state
    claims = simulation.generate_year_claims(year)

    # Process claims
    for claim in claims:
        manufacturer.process_uninsured_claim(claim.amount)

    # Update business state
    manufacturer.step()

    # Frequency automatically adjusts based on new state
    print(f"Year {year}: Revenue={manufacturer.current_revenue:,.0f}, "
          f"Frequency Multiplier={exposure.get_frequency_multiplier(year):.2f}")
```

### Composite Exposure

```python
from ergodic_insurance.exposure_base import (
    RevenueExposure, AssetExposure, EmployeeExposure, CompositeExposure
)

# Combine state-driven and parametric exposures
composite = CompositeExposure(
    exposures={
        'revenue': RevenueExposure(state_provider=manufacturer),
        'assets': AssetExposure(state_provider=manufacturer),
        'employees': EmployeeExposure(base_employees=500, hiring_rate=0.03)
    },
    weights={'revenue': 0.5, 'assets': 0.3, 'employees': 0.2}
)

# Weights are auto-normalized to sum to 1.0
print(composite.weights)  # {'revenue': 0.5, 'assets': 0.3, 'employees': 0.2}
```

### Scenario Analysis

```python
from ergodic_insurance.exposure_base import ScenarioExposure

scenarios = {
    'baseline': [100.0, 105.0, 110.0, 116.0, 122.0],
    'recession': [100.0, 95.0, 90.0, 92.0, 96.0],
    'expansion': [100.0, 112.0, 125.0, 140.0, 155.0]
}

exposure = ScenarioExposure(
    scenarios=scenarios,
    selected_scenario='recession',
    interpolation='linear'
)

# Supports fractional time with interpolation
print(exposure.get_exposure(1.5))  # Interpolates between year 1 and 2
```

### Stochastic Exposure

```python
from ergodic_insurance.exposure_base import StochasticExposure

exposure = StochasticExposure(
    base_value=100_000_000,
    process_type='gbm',
    parameters={
        'drift': 0.05,      # 5% drift
        'volatility': 0.20  # 20% volatility
    },
    seed=42  # Reproducible paths
)

# Values are cached: same time always returns same value
val1 = exposure.get_exposure(1.0)
val2 = exposure.get_exposure(1.0)
assert val1 == val2

# Reset clears cache and re-seeds RNG
exposure.reset()
```

## Key Differences from Previous System

| Aspect | Old System | New State-Driven System |
|--------|------------|------------------------|
| **Growth** | Artificial growth rates | Actual business performance |
| **Claim Generation** | Pre-generated all claims | Generate year-by-year |
| **Frequency Adjustment** | Based on projections | Based on actual state |
| **Coupling** | Tight coupling to parameters | Protocol-based decoupling |
| **Realism** | Theoretical projections | Tracks real financials |
| **Flexibility** | Fixed growth assumptions | Responds to actual events |
| **Type Safety** | float throughout | Decimal in state provider, float at boundary |

## Migration Guide

### Old API (Deprecated)
```python
# DON'T USE - Old artificial growth approach
exposure = RevenueExposure(
    base_revenue=10_000_000,
    growth_rate=0.05  # Artificial 5% growth
)
```

### New API (Current)
```python
# DO USE - State-driven approach
manufacturer = WidgetManufacturer(config)
exposure = RevenueExposure(state_provider=manufacturer)
```

## Benefits of State-Driven System

1. **Realistic Modeling**: Claims frequency responds to actual business performance
2. **Ergodic Alignment**: True time-average behavior without ensemble assumptions
3. **Dynamic Adaptation**: Automatic adjustment to business cycles and shocks
4. **Clean Architecture**: Protocol-based design enables testing and extensibility
5. **No Pre-generation**: Memory efficient, handles long simulations
6. **Event Response**: Claims adapt to actual losses and recovery
7. **Decimal Precision**: Financial state uses `Decimal` for accurate accounting; conversion to `float` happens at the exposure boundary for NumPy compatibility

## Testing the System

```python
def test_state_driven_exposure():
    """Test that exposure tracks actual state changes."""

    # Setup
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.12,
        tax_rate=0.25,
        retention_ratio=0.7
    )
    manufacturer = WidgetManufacturer(config)
    exposure = AssetExposure(state_provider=manufacturer)

    # Initial state
    assert exposure.get_frequency_multiplier(0) == 1.0

    # Simulate large loss
    manufacturer.process_insurance_claim(
        claim_amount=5_000_000,
        deductible_amount=1_000_000,
        insurance_limit=10_000_000
    )

    # Frequency should decrease with asset reduction
    assert exposure.get_frequency_multiplier(1) < 1.0

    # Simulate recovery by setting assets above base
    manufacturer.total_assets = 15_000_000

    # Frequency should increase above baseline
    assert exposure.get_frequency_multiplier(2) > 1.0
```

## Future Enhancements

### Planned Improvements

1. **Industry Benchmarking**: Compare to peer company states
2. **Regulatory Metrics**: Incorporate regulatory capital requirements
3. **Time-Varying Weights**: CompositeExposure weights that change over the simulation
4. **Correlated Stochastic Processes**: Multi-dimensional stochastic exposures with correlation structure
5. **State-Driven Employee/Production**: Extend FinancialStateProvider to include headcount and production metrics

## Conclusion

The state-driven exposure system represents a fundamental improvement in modeling insurance claim frequency. By tracking actual financial state rather than assuming artificial growth, the system provides more realistic simulations that align with ergodic theory principles. This approach better captures the dynamic nature of business risk and the feedback loops between claims, financial health, and future exposure.

All ten exposure classes are fully implemented: three state-driven classes (`RevenueExposure`, `AssetExposure`, `EquityExposure`) that query a `FinancialStateProvider`, two parametric classes (`EmployeeExposure`, `ProductionExposure`) with fixed growth formulas, and three advanced classes (`CompositeExposure`, `ScenarioExposure`, `StochasticExposure`) for composition, scenario analysis, and stochastic modeling.

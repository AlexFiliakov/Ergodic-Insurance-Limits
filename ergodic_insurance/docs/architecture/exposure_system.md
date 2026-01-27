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
    subgraph StateDrivern["State-Driven Exposures"]
        REVENUE["RevenueExposure<br/>Tracks actual revenue"]
        ASSET["AssetExposure<br/>Tracks actual assets"]
        EQUITY["EquityExposure<br/>Tracks actual equity"]
    end

    %% Other Implementations (Future Work)
    subgraph Other["Other Exposures (Future)"]
        EMPLOYEE["EmployeeExposure<br/>Headcount-based"]
        PRODUCTION["ProductionExposure<br/>Production volume"]
        COMPOSITE["CompositeExposure<br/>Multiple factors"]
        SCENARIO["ScenarioExposure<br/>Scenario-dependent"]
        STOCHASTIC["StochasticExposure<br/>Random evolution"]
    end

    %% Business Integration
    subgraph Business["Business Integration"]
        SIMULATION["Simulation"]
        INSURANCE["InsuranceProgram"]
        LOSS_GEN["LossGenerator"]
    end

    %% Relationships
    STATE_PROVIDER --> MANUFACTURER
    MANUFACTURER --> REVENUE
    MANUFACTURER --> ASSET
    MANUFACTURER --> EQUITY

    EXPOSURE_BASE --> REVENUE
    EXPOSURE_BASE --> ASSET
    EXPOSURE_BASE --> EQUITY
    EXPOSURE_BASE -.-> EMPLOYEE
    EXPOSURE_BASE -.-> PRODUCTION
    EXPOSURE_BASE -.-> COMPOSITE
    EXPOSURE_BASE -.-> SCENARIO
    EXPOSURE_BASE -.-> STOCHASTIC

    SIMULATION --> LOSS_GEN
    LOSS_GEN --> EXPOSURE_BASE
    INSURANCE --> EXPOSURE_BASE

    %% Styling
    classDef protocol fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    classDef abstract fill:#f0f4f8,stroke:#5e72e4,stroke-width:2px
    classDef impl fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef future fill:#fafafa,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    classDef business fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    class STATE_PROVIDER protocol
    class EXPOSURE_BASE abstract
    class REVENUE,ASSET,EQUITY impl
    class EMPLOYEE,PRODUCTION,COMPOSITE,SCENARIO,STOCHASTIC future
    class MANUFACTURER,SIMULATION,INSURANCE,LOSS_GEN business
```

## State-Driven Architecture

The new exposure base system queries real-time financial state from providers instead of using artificial growth projections:

```{mermaid}
sequenceDiagram
    participant Sim as Simulation
    participant Gen as LossGenerator
    participant Exp as ExposureBase
    participant Mfg as Manufacturer

    loop Each Year
        Sim->>Gen: generate_year(year)
        Gen->>Exp: get_frequency_multiplier(time)
        Exp->>Mfg: Query current state
        Note over Mfg: Returns actual<br/>revenue/assets/equity
        Mfg-->>Exp: Current financial state
        Exp-->>Gen: Frequency multiplier
        Gen-->>Sim: Year's claims
        Sim->>Mfg: Process claims
        Sim->>Mfg: step() - Update financials
    end
```

## Financial State Provider Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class FinancialStateProvider(Protocol):
    """Protocol for providing current financial state to exposure bases."""

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
        """Get current shareholder equity."""
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
```

## State-Driven Exposure Classes

### RevenueExposure

Tracks actual revenue performance:

```python
class RevenueExposure(ExposureBase):
    """Revenue-based exposure tracking actual business performance."""

    def __init__(self, state_provider: FinancialStateProvider):
        self.state_provider = state_provider

    def get_exposure(self, time: float) -> float:
        """Get current revenue exposure."""
        return self.state_provider.current_revenue

    def get_frequency_multiplier(self, time: float) -> float:
        """Calculate frequency multiplier based on revenue ratio."""
        if self.state_provider.base_revenue == 0:
            return 0

        # Square root scaling for frequency
        revenue_ratio = self.state_provider.current_revenue / self.state_provider.base_revenue
        return np.sqrt(revenue_ratio)
```

### AssetExposure

Tracks actual asset base:

```python
class AssetExposure(ExposureBase):
    """Asset-based exposure tracking actual balance sheet."""

    def __init__(self, state_provider: FinancialStateProvider):
        self.state_provider = state_provider

    def get_exposure(self, time: float) -> float:
        """Get current asset exposure."""
        return self.state_provider.current_assets

    def get_frequency_multiplier(self, time: float) -> float:
        """Linear scaling with asset base."""
        if self.state_provider.base_assets == 0:
            return 0

        return self.state_provider.current_assets / self.state_provider.base_assets
```

### EquityExposure

Tracks actual equity position:

```python
class EquityExposure(ExposureBase):
    """Equity-based exposure with conservative scaling."""

    def __init__(self, state_provider: FinancialStateProvider):
        self.state_provider = state_provider

    def get_exposure(self, time: float) -> float:
        """Get current equity exposure."""
        return self.state_provider.current_equity

    def get_frequency_multiplier(self, time: float) -> float:
        """Conservative cube root scaling for equity."""
        if self.state_provider.current_equity <= 0:
            return 0  # No exposure when insolvent
        if self.state_provider.base_equity <= 0:
            return 0

        equity_ratio = self.state_provider.current_equity / self.state_provider.base_equity
        return equity_ratio ** (1/3)  # Cube root scaling
```

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

## Key Differences from Previous System

| Aspect | Old System | New State-Driven System |
|--------|------------|------------------------|
| **Growth** | Artificial growth rates | Actual business performance |
| **Claim Generation** | Pre-generated all claims | Generate year-by-year |
| **Frequency Adjustment** | Based on projections | Based on actual state |
| **Coupling** | Tight coupling to parameters | Protocol-based decoupling |
| **Realism** | Theoretical projections | Tracks real financials |
| **Flexibility** | Fixed growth assumptions | Responds to actual events |

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

## Testing the New System

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

    # Simulate recovery
    manufacturer.assets = 15_000_000

    # Frequency should increase above baseline
    assert exposure.get_frequency_multiplier(2) > 1.0
```

## Future Enhancements

### Planned Improvements

1. **Composite State Exposures**: Combine multiple state metrics
2. **Scenario-Based States**: Switch between different state trajectories
3. **Stochastic State Evolution**: Add randomness to state transitions
4. **Industry Benchmarking**: Compare to peer company states
5. **Regulatory Metrics**: Incorporate regulatory capital requirements

### Example Future Implementation

```python
# Future: Composite state-driven exposure
class CompositeStateExposure(ExposureBase):
    """Weighted combination of state metrics."""

    def __init__(self, state_provider: FinancialStateProvider, weights: dict):
        self.state_provider = state_provider
        self.weights = weights

    def get_frequency_multiplier(self, time: float) -> float:
        revenue_mult = RevenueExposure(self.state_provider).get_frequency_multiplier(time)
        asset_mult = AssetExposure(self.state_provider).get_frequency_multiplier(time)
        equity_mult = EquityExposure(self.state_provider).get_frequency_multiplier(time)

        return (self.weights.get('revenue', 0) * revenue_mult +
                self.weights.get('assets', 0) * asset_mult +
                self.weights.get('equity', 0) * equity_mult)
```

## Conclusion

The state-driven exposure system represents a fundamental improvement in modeling insurance claim frequency. By tracking actual financial state rather than assuming artificial growth, the system provides more realistic simulations that align with ergodic theory principles. This approach better captures the dynamic nature of business risk and the feedback loops between claims, financial health, and future exposure.

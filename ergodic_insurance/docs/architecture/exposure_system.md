# Exposure Base System Architecture

This document describes the flexible exposure base system that allows for different approaches to calculating insurance exposures and limits.

## System Overview

```{mermaid}
graph TB
    %% Abstract Base
    subgraph Abstract["Abstract Layer"]
        EXPOSURE_BASE["ExposureBase<br/>(Abstract Base Class)"]
    end

    %% Concrete Implementations
    subgraph Implementations["Exposure Implementations"]
        REVENUE["RevenueExposure<br/>Revenue-based limits"]
        ASSET["AssetExposure<br/>Asset-based limits"]
        EQUITY["EquityExposure<br/>Equity-based limits"]
        EMPLOYEE["EmployeeExposure<br/>Headcount-based"]
        PRODUCTION["ProductionExposure<br/>Production volume"]
        COMPOSITE["CompositeExposure<br/>Multiple factors"]
        SCENARIO["ScenarioExposure<br/>Scenario-dependent"]
        STOCHASTIC["StochasticExposure<br/>Random evolution"]
    end

    %% Business Integration
    subgraph Business["Business Integration"]
        MANUFACTURER["WidgetManufacturer"]
        INSURANCE["InsuranceProgram"]
        CLAIM_GEN["ClaimGenerator"]
    end

    %% Configuration
    subgraph Config["Configuration"]
        EXPOSURE_CFG["ExposureConfig"]
        PROFILE["Profile Settings"]
        PARAMS["Parameters"]
    end

    %% Relationships
    EXPOSURE_BASE --> REVENUE
    EXPOSURE_BASE --> ASSET
    EXPOSURE_BASE --> EQUITY
    EXPOSURE_BASE --> EMPLOYEE
    EXPOSURE_BASE --> PRODUCTION
    EXPOSURE_BASE --> COMPOSITE
    EXPOSURE_BASE --> SCENARIO
    EXPOSURE_BASE --> STOCHASTIC

    MANUFACTURER --> EXPOSURE_BASE
    INSURANCE --> EXPOSURE_BASE
    CLAIM_GEN --> EXPOSURE_BASE

    Config --> EXPOSURE_BASE
    EXPOSURE_CFG --> PROFILE
    PROFILE --> PARAMS

    %% Styling
    classDef abstract fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    classDef impl fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef business fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef config fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class EXPOSURE_BASE abstract
    class REVENUE,ASSET,EQUITY,EMPLOYEE,PRODUCTION,COMPOSITE,SCENARIO,STOCHASTIC impl
    class MANUFACTURER,INSURANCE,CLAIM_GEN business
    class EXPOSURE_CFG,PROFILE,PARAMS config
```

## Class Hierarchy

```{mermaid}
classDiagram
    %% Abstract Base Class
    class ExposureBase {
        <<abstract>>
        #name: str
        #description: str
        #parameters: Dict
        +calculate_exposure(manufacturer) float
        +get_growth_factor(year) float
        +apply_inflation(value, rate) float
        +validate_parameters() bool
        #_calculate_base_exposure(manufacturer)*
    }

    %% Simple Exposures
    class RevenueExposure {
        -revenue_multiple: float
        -lookback_years: int
        +_calculate_base_exposure(manufacturer) float
        +calculate_average_revenue(manufacturer) float
    }

    class AssetExposure {
        -asset_percentage: float
        -asset_types: List[str]
        +_calculate_base_exposure(manufacturer) float
        +get_eligible_assets(manufacturer) float
    }

    class EquityExposure {
        -equity_multiple: float
        -include_retained_earnings: bool
        +_calculate_base_exposure(manufacturer) float
        +calculate_total_equity(manufacturer) float
    }

    %% Complex Exposures
    class EmployeeExposure {
        -per_employee_limit: float
        -employee_count: int
        -growth_rate: float
        +_calculate_base_exposure(manufacturer) float
        +project_headcount(year) int
    }

    class ProductionExposure {
        -units_produced: int
        -per_unit_exposure: float
        -production_growth: float
        +_calculate_base_exposure(manufacturer) float
        +calculate_production_value(manufacturer) float
    }

    %% Advanced Exposures
    class CompositeExposure {
        -components: List[ExposureBase]
        -weights: List[float]
        +_calculate_base_exposure(manufacturer) float
        +add_component(exposure, weight) None
        +normalize_weights() None
    }

    class ScenarioExposure {
        -scenarios: Dict[str, ExposureBase]
        -probabilities: Dict[str, float]
        -current_scenario: str
        +_calculate_base_exposure(manufacturer) float
        +switch_scenario(scenario_name) None
        +calculate_expected_exposure(manufacturer) float
    }

    class StochasticExposure {
        -base_exposure: ExposureBase
        -volatility: float
        -mean_reversion: float
        -random_state: RandomState
        +_calculate_base_exposure(manufacturer) float
        +simulate_path(n_periods) ndarray
    }

    %% Relationships
    ExposureBase <|-- RevenueExposure
    ExposureBase <|-- AssetExposure
    ExposureBase <|-- EquityExposure
    ExposureBase <|-- EmployeeExposure
    ExposureBase <|-- ProductionExposure
    ExposureBase <|-- CompositeExposure
    ExposureBase <|-- ScenarioExposure
    ExposureBase <|-- StochasticExposure

    CompositeExposure o-- ExposureBase : contains
    ScenarioExposure o-- ExposureBase : uses
    StochasticExposure o-- ExposureBase : wraps
```

## Usage Patterns

### Strategy Pattern Implementation

```{mermaid}
sequenceDiagram
    participant Config
    participant Factory as ExposureFactory
    participant Base as ExposureBase
    participant Impl as ConcreteExposure
    participant Mfg as Manufacturer

    Config->>Factory: Request exposure type
    Factory->>Impl: Create instance
    Impl-->>Factory: Exposure instance

    Factory-->>Config: Return exposure

    loop Each Year
        Mfg->>Base: calculate_exposure()
        Base->>Impl: _calculate_base_exposure()
        Impl->>Mfg: Get financial metrics
        Mfg-->>Impl: Return metrics
        Impl-->>Base: Base exposure value
        Base->>Base: Apply growth/inflation
        Base-->>Mfg: Final exposure
    end
```

## Exposure Types Comparison

| Exposure Type | Base Calculation | Use Case | Volatility | Complexity |
|--------------|------------------|----------|------------|------------|
| **Revenue** | Annual revenue × Multiple | Service companies | Medium | Low |
| **Asset** | Total assets × Percentage | Capital-intensive | Low | Low |
| **Equity** | Shareholders' equity × Multiple | Financial firms | Medium | Low |
| **Employee** | Headcount × Per-employee limit | Labor-intensive | Low | Medium |
| **Production** | Units × Per-unit exposure | Manufacturing | High | Medium |
| **Composite** | Weighted combination | Diversified businesses | Variable | High |
| **Scenario** | Scenario-dependent | Uncertain environments | Variable | High |
| **Stochastic** | Random evolution | Research/modeling | High | Very High |

## Configuration Examples

### Revenue-Based Exposure
```yaml
exposure:
  type: revenue
  parameters:
    revenue_multiple: 2.5
    lookback_years: 3
    include_projected: true
```

### Composite Exposure
```yaml
exposure:
  type: composite
  components:
    - type: revenue
      weight: 0.5
      parameters:
        revenue_multiple: 2.0
    - type: asset
      weight: 0.3
      parameters:
        asset_percentage: 0.75
    - type: employee
      weight: 0.2
      parameters:
        per_employee_limit: 100000
```

### Scenario-Based Exposure
```yaml
exposure:
  type: scenario
  scenarios:
    normal:
      type: revenue
      probability: 0.6
      parameters:
        revenue_multiple: 2.0
    growth:
      type: revenue
      probability: 0.3
      parameters:
        revenue_multiple: 3.0
    recession:
      type: asset
      probability: 0.1
      parameters:
        asset_percentage: 0.5
```

## Integration with Insurance Program

```python
# Example integration
from exposure_base import ExposureFactory
from insurance_program import InsuranceProgram

# Create exposure calculator
exposure = ExposureFactory.create("revenue", {
    "revenue_multiple": 2.5,
    "lookback_years": 3
})

# Use in insurance program
insurance_program = InsuranceProgram()
manufacturer = WidgetManufacturer(config)

# Calculate dynamic limits
for year in range(simulation_years):
    current_exposure = exposure.calculate_exposure(manufacturer)
    insurance_program.update_limits(current_exposure)

    # Process claims with exposure-based limits
    claims = claim_generator.generate(year)
    for claim in claims:
        payout = insurance_program.process_claim(claim, current_exposure)
        manufacturer.record_claim(payout)
```

## Advanced Features

### Dynamic Exposure Adjustment
```python
class DynamicExposure(ExposureBase):
    """Adjusts exposure based on risk metrics"""

    def calculate_exposure(self, manufacturer):
        base = self._calculate_base_exposure(manufacturer)

        # Adjust for financial health
        if manufacturer.debt_to_equity > 2.0:
            base *= 1.2  # Increase exposure for higher leverage

        # Adjust for profitability
        if manufacturer.operating_margin < 0.05:
            base *= 0.8  # Reduce exposure for low margins

        return base
```

### Machine Learning Integration
```python
class MLExposure(ExposureBase):
    """Uses ML model to predict optimal exposure"""

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def _calculate_base_exposure(self, manufacturer):
        features = self._extract_features(manufacturer)
        prediction = self.model.predict(features)
        return prediction[0]
```

## Benefits of the Exposure System

1. **Flexibility**: Easy to switch between exposure calculation methods
2. **Extensibility**: Simple to add new exposure types
3. **Composability**: Combine multiple exposure types
4. **Testability**: Each exposure type can be tested independently
5. **Configuration-Driven**: Change behavior without code changes
6. **Scenario Analysis**: Compare different exposure strategies

## Future Enhancements

1. **Market-Linked Exposures**
   - Link to market indices
   - Correlation with economic indicators

2. **Risk-Adjusted Exposures**
   - Incorporate VaR/CVaR metrics
   - Dynamic risk scoring

3. **Regulatory Compliance**
   - Solvency II calculations
   - Basel III requirements

4. **Industry Benchmarking**
   - Peer comparison
   - Industry-specific models

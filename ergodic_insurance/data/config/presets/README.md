# Configuration Presets

This directory contains preset libraries for quick configuration of common scenarios.

## Available Preset Libraries

### market_conditions.yaml
Market condition presets affecting revenue and cost parameters:
- `stable` - Low volatility, predictable conditions
- `volatile` - High volatility, uncertain conditions
- `growth` - Expanding market conditions
- `recession` - Contracting market conditions

### layer_structures.yaml
Insurance layer structure presets:
- `basic` - Single primary layer
- `comprehensive` - Multi-layer with excess coverage
- `catastrophic` - Focus on extreme tail risks
- `balanced` - Optimized cost vs coverage

### risk_scenarios.yaml
Risk profile presets:
- `low_risk` - Manufacturing with minimal hazards
- `moderate_risk` - Standard manufacturing risks
- `high_risk` - Complex operations with multiple exposures
- `catastrophic` - Significant tail risk exposure

## Usage

Presets are applied via the profile's `presets` field:

```yaml
profile:
  name: volatile-market-profile
  presets:
    market: volatile
    layers: comprehensive
    risk: moderate_risk
```

## Preset Structure

Each preset library contains named configurations:

```yaml
# market_conditions.yaml
volatile:
  manufacturer:
    revenue_volatility: 0.25
    cost_volatility: 0.20
  simulation:
    stochastic_revenue: true
```

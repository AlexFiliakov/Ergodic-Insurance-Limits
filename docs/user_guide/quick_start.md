---
layout: default
title: Quick Start Guide - Ergodic Insurance Framework
description: Get started with the ergodic insurance optimization framework in minutes
mathjax: true
---

# Quick Start Guide

## Installation

### Prerequisites
- Python 3.12 or higher
- 4GB RAM minimum (8GB recommended for large simulations)

### Install via pip
```bash
pip install ergodic-insurance
```

### Install from source
```bash
git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
cd Ergodic-Insurance-Limits
pip install -e .
```

## Your First Simulation

### Step 1: Import the Framework
```python
from ergodic_insurance.src import (
    Manufacturer,
    InsuranceProgram,
    EnsembleSimulator,
    ErgodicAnalyzer
)
```

### Step 2: Define Your Business
```python
# Create a widget manufacturer with \$10M in assets
business = Manufacturer(
    starting_assets=10_000_000,
    base_revenue=15_000_000,  # \$15M annual revenue
    operating_margin=0.08,      # 8% margin
    tax_rate=0.25              # 25% tax rate
)
```

### Step 3: Configure Insurance Options
```python
# Define insurance layers
insurance = InsuranceProgram([
    {"limit": 5_000_000, "attachment": 0, "premium_rate": 0.015},
    {"limit": 20_000_000, "attachment": 5_000_000, "premium_rate": 0.008},
])
```

### Step 4: Run Simulations
```python
# Simulate 100 trajectories for 20 years
simulator = EnsembleSimulator(
    manufacturer=business,
    insurance=insurance,
    n_trajectories=100,
    n_years=20
)

results = simulator.run()
```

### Step 5: Analyze Results
```python
# Calculate ergodic (time-average) growth rate
analyzer = ErgodicAnalyzer(results)
growth_rate = analyzer.calculate_growth_rate()

print(f"Time-average growth rate: {growth_rate:.2%}")
print(f"Probability of ruin: {analyzer.ruin_probability():.2%}")
print(f"Expected terminal wealth: ${analyzer.expected_wealth():,.0f}")
```

## Understanding the Output

### Key Metrics Explained:
- **Time-Average Growth Rate**: The growth rate a single business experiences over time
- **Ensemble Average Growth**: The average growth across many parallel businesses
- **Probability of Ruin**: Chance of bankruptcy within the simulation period
- **Volatility Reduction**: How much insurance reduces wealth variance

### Typical Results:
```
Without Insurance:
- Growth Rate: 8.2%
- Ruin Probability: 15%
- 20-Year Wealth: \$47M ± \$35M

With Optimal Insurance:
- Growth Rate: 11.3%
- Ruin Probability: <1%
- 20-Year Wealth: \$89M ± \$12M
```

## Visualization of Results

The following graph shows a typical 20-year simulation comparing performance with and without insurance:

![Insurance Impact Visualization](/Ergodic-Insurance-Limits/assets/results/getting_started/output.png)

Key observations from the visualization:
- **Orange lines** mark years with catastrophic losses (>$1M)
- **Without insurance** (blue) shows sharp drops during catastrophic events
- **With insurance** (orange) maintains smoother growth trajectory
- The $100K annual premium provides protection against losses up to $5M

## Next Steps

1. **Customize Parameters**: Adjust business parameters to match your organization
2. **Test Scenarios**: Run sensitivity analysis on premium rates and limits
3. **Optimize Coverage**: Use the optimization module to find ideal insurance structure
4. **Generate Reports**: Export results for presentation to stakeholders

## Common Use Cases

### Manufacturing Company
```python
# High operational risk, moderate margins
Manufacturer(
    starting_assets=10_000_000,
    volatility=0.25,  # Higher volatility
    operating_margin=0.08
)
```

### Tech Startup
```python
# High growth, high volatility
Manufacturer(
    starting_assets=1_000_000,
    growth_rate=0.30,  # 30% growth
    volatility=0.40    # Very high volatility
)
```

### Established Retailer
```python
# Stable, predictable business
Manufacturer(
    starting_assets=50_000_000,
    growth_rate=0.05,  # Modest growth
    volatility=0.10    # Low volatility
)
```

## Getting Help

- [Full Documentation](/Ergodic-Insurance-Limits/docs/overview)
- [Tutorial: Basic Simulation](/Ergodic-Insurance-Limits/tutorials/basic_simulation)
- [GitHub Issues](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues)

---

[← Back to Executive Summary](/Ergodic-Insurance-Limits/docs/user_guide/executive_summary) | [Continue to Decision Framework →](/Ergodic-Insurance-Limits/docs/user_guide/decision_framework)

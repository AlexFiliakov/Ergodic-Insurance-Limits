---
layout: default
title: Getting Started Tutorial - Ergodic Insurance Framework
description: Step-by-step tutorial for your first ergodic insurance analysis
mathjax: true
---

# Getting Started Tutorial

## Overview

This tutorial will walk you through your first complete ergodic insurance analysis. By the end, you'll understand:
- How to model a business with multiplicative dynamics
- How to configure insurance programs
- How to run Monte Carlo simulations
- How to interpret ergodic vs ensemble results

## Prerequisites

Make sure you have the framework installed:
```bash
pip install ergodic-insurance
```

Or if working from source:
```bash
cd Ergodic-Insurance-Limits
pip install -e .
```

## Part 1: Understanding the Problem

### The Widget Manufacturer Scenario

Imagine you run a widget manufacturing company:
- **Assets**: \$10 million in equipment and inventory
- **Revenue**: \$15 million per year
- **Profit Margin**: 8% after all costs
- **Risk**: Subject to operational losses (equipment failure, accidents, lawsuits)

### The Insurance Decision

You need to decide:
1. How much insurance coverage to buy?
2. What deductibles and limits make sense?
3. Is insurance worth the premium cost?

Traditional analysis would compare premium costs to expected losses. The ergodic approach reveals why this is insufficient.

## Part 2: Setting Up the Simulation

### Step 1: Import Required Modules

```python
import numpy as np
import matplotlib.pyplot as plt
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.insurance_program import InsuranceProgram
from ergodic_insurance.monte_carlo import MonteCarloEngine
from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
from ergodic_insurance.config_v2 import ManufacturerConfig
from ergodic_insurance.src import visualization

# Set random seed for reproducibility
np.random.seed(42)
```

### Step 2: Create the Business Model

```python
# Configure the manufacturer
config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.5,  # Revenue = 1.5x assets
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=0.7  # Retain 70% of earnings
)

# Initialize the manufacturer
manufacturer = WidgetManufacturer(config)

print(f"Starting assets: ${manufacturer.assets:,.0f}")
print(f"Expected annual revenue: ${manufacturer.assets * config.asset_turnover_ratio:,.0f}")
print(f"Expected annual profit: ${manufacturer.assets * config.asset_turnover_ratio * config.base_operating_margin * (1 - config.tax_rate):,.0f}")
```

### Step 3: Define Loss Characteristics

```python
# Configure claim generator for operational losses
# Loss frequency scales with revenue (more activity = more risk exposure)
base_frequency = 3.0  # Base frequency for $10M revenue company
revenue = manufacturer.assets * config.asset_turnover_ratio  # Current revenue

claim_generator = ClaimGenerator(
    frequency=base_frequency * (revenue / 10_000_000),  # Scale with revenue
    severity_mean=100_000,      # Mean claim size
    severity_std=200_000,       # Standard deviation
    seed=42                     # For reproducibility
)

# Generate sample claims for one year using revenue-dependent frequency
sample_claim_events, stats = claim_generator.generate_enhanced_claims(
    years=1,
    revenue=revenue,
    use_enhanced_distributions=False
)
sample_claims = [claim.amount for claim in sample_claim_events]
print(f"Sample year losses: ${sum(sample_claims):,.0f}")
```

## Part 3: Comparing Insurance Strategies

### Strategy 1: No Insurance

```python
# Run simulation without insurance
no_insurance_sim = MonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=None,
    n_simulations=1000,
    time_horizon=20
)

results_no_insurance = no_insurance_sim.run()
```

### Strategy 2: Basic Insurance

```python
# Create basic insurance program
basic_insurance = InsuranceProgram([
    {
        "limit": 1_000_000,
        "attachment": 100_000,  # \$100k deductible
        "premium_rate": 0.02
    }
])

basic_sim = MonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=basic_insurance,
    n_simulations=1000,
    time_horizon=20
)

results_basic = basic_sim.run()
```

### Strategy 3: Optimal Insurance

```python
# Create optimized insurance program
optimal_insurance = InsuranceProgram([
    {
        "limit": 5_000_000,
        "attachment": 0,
        "premium_rate": 0.015
    },
    {
        "limit": 20_000_000,
        "attachment": 5_000_000,
        "premium_rate": 0.008
    }
])

optimal_sim = MonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=optimal_insurance,
    n_simulations=1000,
    time_horizon=20
)

results_optimal = optimal_sim.run()
```

## Part 4: Analyzing Results

### Calculate Ergodic Growth Rates

```python
# Analyze each strategy
analyzer_none = ErgodicAnalyzer(results_no_insurance)
analyzer_basic = ErgodicAnalyzer(results_basic)
analyzer_optimal = ErgodicAnalyzer(results_optimal)

print("Time-Average Growth Rates:")
print(f"No Insurance:      {analyzer_none.time_average_growth():.2%}")
print(f"Basic Insurance:   {analyzer_basic.time_average_growth():.2%}")
print(f"Optimal Insurance: {analyzer_optimal.time_average_growth():.2%}")

print("\nProbability of Ruin:")
print(f"No Insurance:      {analyzer_none.ruin_probability():.1%}")
print(f"Basic Insurance:   {analyzer_basic.ruin_probability():.1%}")
print(f"Optimal Insurance: {analyzer_optimal.ruin_probability():.1%}")
```

### Visualize Wealth Trajectories

```python
# Plot wealth paths
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

visualization.plot_wealth_trajectories(
    results_no_insurance,
    ax=axes[0],
    title="No Insurance"
)

visualization.plot_wealth_trajectories(
    results_basic,
    ax=axes[1],
    title="Basic Insurance"
)

visualization.plot_wealth_trajectories(
    results_optimal,
    ax=axes[2],
    title="Optimal Insurance"
)

plt.tight_layout()
plt.show()
```

## Part 5: Key Insights

### The Ergodic Advantage

Notice how:
1. **Without insurance**: High variance, some trajectories fail completely
2. **With basic insurance**: Reduced downside, but growth is limited
3. **With optimal insurance**: Best growth rate AND lowest risk

### Why This Matters

The optimal insurance strategy:
- Costs more in premiums than expected losses
- But delivers superior time-average growth
- Reduces the probability of ruin to near zero
- Creates more predictable business outcomes

## Part 6: Customizing for Your Business

### Adjust Business Parameters

```python
# For a high-growth tech company
tech_company = Manufacturer(
    starting_assets=2_000_000,
    base_revenue=5_000_000,
    base_operating_margin=0.15,  # Higher margins
    growth_rate=0.25,        # 25% growth
    volatility=0.35          # Higher uncertainty
)
```

### Modify Risk Profile

```python
# For lower frequency, higher severity risks
# Catastrophic events are less common but scale differently with revenue
cat_base_frequency = 0.05  # Base catastrophe frequency
revenue = manufacturer.assets * 1.5  # Assuming 1.5x turnover

catastrophic_risks = ClaimGenerator(
    frequency=cat_base_frequency * (revenue / 10_000_000)**0.5,  # Square root scaling
    severity_mean=10_000_000,   # Massive potential losses
    severity_std=5_000_000,
    seed=42
)
```

## Summary and Next Steps

You've learned how to:
✅ Set up a business model with multiplicative dynamics
✅ Configure realistic loss scenarios
✅ Compare insurance strategies using Monte Carlo simulation
✅ Analyze results using ergodic theory
✅ Visualize wealth trajectories

### Next Tutorials:
1. [Basic Simulation Deep Dive](/Ergodic-Insurance-Limits/tutorials/basic_simulation)
2. [Optimization Workflow](/Ergodic-Insurance-Limits/tutorials/optimization_workflow)
3. [Advanced Scenarios](/Ergodic-Insurance-Limits/tutorials/advanced_scenarios)

### Resources:
- [API Documentation](/Ergodic-Insurance-Limits/api/)
- [Theory Background](/Ergodic-Insurance-Limits/theory/01_ergodic_economics)
- [GitHub Repository](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits)

---

[← Back to Home](/Ergodic-Insurance-Limits/) | [Continue to Basic Simulation →](/Ergodic-Insurance-Limits/tutorials/basic_simulation)

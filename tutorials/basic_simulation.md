---
layout: default
title: Basic Simulation Tutorial - Ergodic Insurance Framework
description: Learn the fundamentals of running insurance simulations
mathjax: true
---

# Basic Simulation Tutorial

## Introduction

This tutorial covers the fundamentals of running Monte Carlo simulations for insurance analysis. You'll learn how to set up simulations, understand the parameters, and interpret results.

## Core Concepts

### What is a Monte Carlo Simulation?

Monte Carlo simulation generates thousands of possible future scenarios to understand the range of potential outcomes. For insurance analysis, we simulate:

- Business revenue and costs over time
- Random loss events
- Insurance claim payments
- Resulting wealth trajectories

### Why Multiple Trajectories Matter

A single trajectory might show success, but we need to understand:
- **Typical outcomes** (median case)
- **Best case scenarios** (95th percentile)
- **Worst case scenarios** (5th percentile)
- **Ruin probability** (complete failure rate)

## Setting Up Your First Simulation

### Step 1: Import and Initialize

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ergodic_insurance.src import (
    Manufacturer,
    ClaimGenerator,
    InsuranceProgram,
    MonteCarloEngine
)

# Set seed for reproducibility
np.random.seed(42)
```

### Step 2: Configure the Business Model

```python
# Create a manufacturer with detailed parameters
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    base_revenue=15_000_000,
    operating_margin=0.08,
    tax_rate=0.25,
    growth_rate=0.05,  # 5% annual growth
    volatility=0.15,   # 15% revenue volatility
    working_capital_ratio=0.20
)

# Verify the configuration
print("Business Configuration:")
print(f"  Starting Assets: ${manufacturer.starting_assets:,.0f}")
print(f"  Annual Revenue: ${manufacturer.base_revenue:,.0f}")
print(f"  Expected Annual Profit: ${manufacturer.base_revenue * manufacturer.operating_margin:,.0f}")
```

### Step 3: Define the Risk Profile

```python
# Configure claim generation
claim_generator = ClaimGenerator(
    # Small claims (high frequency, low severity)
    frequency_small=5.0,  # 5 per year on average
    severity_small_mean=20_000,
    severity_small_std=10_000,

    # Medium claims
    frequency_medium=0.5,  # Once every 2 years
    severity_medium_mean=500_000,
    severity_medium_std=200_000,

    # Large claims (low frequency, high severity)
    frequency_large=0.1,  # Once every 10 years
    severity_large_mean=5_000_000,
    severity_large_std=2_000_000
)

# Generate example claims for visualization
example_year_claims = claim_generator.generate_annual_claims()
print(f"\nExample annual loss: ${sum(example_year_claims):,.0f}")
print(f"Number of claims: {len(example_year_claims)}")
```

## Running Different Simulation Scenarios

### Scenario 1: No Insurance

```python
# Run simulation without insurance
engine_no_insurance = MonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=None,  # No insurance
    n_simulations=1000,
    time_horizon=20,
    random_seed=42
)

results_no_insurance = engine_no_insurance.run()
print(f"Simulation complete: {len(results_no_insurance.trajectories)} trajectories")
```

### Scenario 2: Basic Insurance

```python
# Define basic insurance program
basic_insurance = InsuranceProgram([
    {
        "name": "Primary Layer",
        "limit": 1_000_000,
        "attachment": 100_000,  # $100k deductible
        "premium_rate": 0.025,  # 2.5% of limit
    }
])

# Run simulation with basic insurance
engine_basic = MonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=basic_insurance,
    n_simulations=1000,
    time_horizon=20,
    random_seed=42
)

results_basic = engine_basic.run()
```

### Scenario 3: Comprehensive Insurance

```python
# Define comprehensive multi-layer program
comprehensive_insurance = InsuranceProgram([
    {
        "name": "Primary",
        "limit": 2_000_000,
        "attachment": 0,  # No deductible
        "premium_rate": 0.02,
    },
    {
        "name": "Excess 1",
        "limit": 3_000_000,
        "attachment": 2_000_000,
        "premium_rate": 0.012,
    },
    {
        "name": "Excess 2",
        "limit": 10_000_000,
        "attachment": 5_000_000,
        "premium_rate": 0.008,
    }
])

# Run simulation
engine_comprehensive = MonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=comprehensive_insurance,
    n_simulations=1000,
    time_horizon=20,
    random_seed=42
)

results_comprehensive = engine_comprehensive.run()
```

## Understanding Simulation Parameters

### Key Parameters Explained

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| `n_simulations` | Number of trajectories | 1,000 - 100,000 | Higher = more accurate, slower |
| `time_horizon` | Years to simulate | 10 - 50 | Longer reveals ergodic properties |
| `random_seed` | Reproducibility | Any integer | Same seed = same results |
| `volatility` | Business uncertainty | 0.10 - 0.40 | Higher = more variation |
| `frequency` | Claims per year | 0.1 - 10 | Higher = more events |
| `severity` | Claim size | Varies | Higher = larger losses |

### Choosing the Right Parameters

```python
# For stable businesses
stable_params = {
    "volatility": 0.10,
    "time_horizon": 30,
    "n_simulations": 5000
}

# For high-risk businesses
risky_params = {
    "volatility": 0.30,
    "time_horizon": 20,
    "n_simulations": 10000  # Need more simulations for accuracy
}

# For quick testing
test_params = {
    "volatility": 0.15,
    "time_horizon": 10,
    "n_simulations": 100
}
```

## Analyzing Simulation Results

### Extract Key Metrics

```python
def analyze_results(results, label):
    """Analyze simulation results and return key metrics"""

    # Calculate metrics
    final_wealth = [t.wealth[-1] for t in results.trajectories]
    growth_rates = [t.calculate_growth_rate() for t in results.trajectories]
    ruined = sum(1 for t in results.trajectories if min(t.wealth) <= 0)

    metrics = {
        "label": label,
        "median_wealth": np.median(final_wealth),
        "mean_wealth": np.mean(final_wealth),
        "std_wealth": np.std(final_wealth),
        "min_wealth": np.min(final_wealth),
        "max_wealth": np.max(final_wealth),
        "median_growth": np.median(growth_rates),
        "mean_growth": np.mean(growth_rates),
        "ruin_probability": ruined / len(results.trajectories),
        "percentile_5": np.percentile(final_wealth, 5),
        "percentile_95": np.percentile(final_wealth, 95)
    }

    return metrics

# Analyze all scenarios
metrics_none = analyze_results(results_no_insurance, "No Insurance")
metrics_basic = analyze_results(results_basic, "Basic Insurance")
metrics_comprehensive = analyze_results(results_comprehensive, "Comprehensive")

# Display comparison
comparison_df = pd.DataFrame([metrics_none, metrics_basic, metrics_comprehensive])
print(comparison_df.to_string())
```

### Visualize Results

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Wealth trajectories
for i, (results, title) in enumerate([
    (results_no_insurance, "No Insurance"),
    (results_basic, "Basic Insurance"),
    (results_comprehensive, "Comprehensive Insurance")
]):
    ax = axes[0, i]

    # Plot sample trajectories
    for j in range(min(50, len(results.trajectories))):
        traj = results.trajectories[j]
        ax.plot(traj.wealth, alpha=0.3, color='blue')

    # Add median
    median_wealth = np.median([t.wealth for t in results.trajectories], axis=0)
    ax.plot(median_wealth, color='red', linewidth=2, label='Median')

    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 2: Growth rate distributions
for i, (results, title) in enumerate([
    (results_no_insurance, "No Insurance"),
    (results_basic, "Basic Insurance"),
    (results_comprehensive, "Comprehensive Insurance")
]):
    ax = axes[1, i]

    growth_rates = [t.calculate_growth_rate() for t in results.trajectories]
    ax.hist(growth_rates, bins=30, alpha=0.7, color='green')
    ax.axvline(np.median(growth_rates), color='red', linestyle='--', label=f'Median: {np.median(growth_rates):.2%}')

    ax.set_title(f'{title} - Growth Rates')
    ax.set_xlabel('Annual Growth Rate')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Advanced Simulation Techniques

### Parallel Processing for Speed

```python
from ergodic_insurance.src import ParallelMonteCarloEngine

# Use parallel processing for large simulations
parallel_engine = ParallelMonteCarloEngine(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    insurance_program=comprehensive_insurance,
    n_simulations=10000,  # 10x more simulations
    time_horizon=20,
    n_workers=4  # Use 4 CPU cores
)

# Run with progress bar
results_large = parallel_engine.run(show_progress=True)
```

### Batch Processing for Memory Efficiency

```python
# For very large simulations, process in batches
batch_size = 1000
total_simulations = 100000

all_metrics = []

for batch in range(0, total_simulations, batch_size):
    engine = MonteCarloEngine(
        manufacturer=manufacturer,
        claim_generator=claim_generator,
        insurance_program=comprehensive_insurance,
        n_simulations=batch_size,
        time_horizon=20
    )

    results = engine.run()
    metrics = analyze_results(results, f"Batch {batch//batch_size}")
    all_metrics.append(metrics)

    # Clear memory
    del results

# Aggregate results
final_metrics = aggregate_batch_metrics(all_metrics)
```

### Sensitivity Testing

```python
# Test sensitivity to volatility
volatilities = [0.10, 0.15, 0.20, 0.25, 0.30]
sensitivity_results = []

for vol in volatilities:
    # Update manufacturer
    test_manufacturer = Manufacturer(
        starting_assets=10_000_000,
        volatility=vol
    )

    # Run simulation
    engine = MonteCarloEngine(
        manufacturer=test_manufacturer,
        claim_generator=claim_generator,
        insurance_program=comprehensive_insurance,
        n_simulations=1000,
        time_horizon=20
    )

    results = engine.run()
    metrics = analyze_results(results, f"Vol={vol}")
    sensitivity_results.append(metrics)

# Plot sensitivity
plot_sensitivity_analysis(sensitivity_results, parameter="volatility")
```

## Common Pitfalls and Solutions

### Pitfall 1: Too Few Simulations

**Problem:** Results vary significantly between runs
**Solution:** Increase `n_simulations` to at least 1,000, preferably 10,000

### Pitfall 2: Short Time Horizons

**Problem:** Ergodic properties don't emerge
**Solution:** Use at least 20 years, preferably 30-50 for ergodic analysis

### Pitfall 3: Ignoring Correlation

**Problem:** Underestimating aggregate risk
**Solution:** Model correlation between different risk types

### Pitfall 4: Fixed Parameters

**Problem:** Real businesses change over time
**Solution:** Use time-varying parameters or scenario analysis

## Practice Exercises

### Exercise 1: Find the Optimal Deductible

```python
# Test different deductible levels
deductibles = [0, 50_000, 100_000, 250_000, 500_000]

for deductible in deductibles:
    # Create insurance with varying deductible
    insurance = InsuranceProgram([
        {
            "limit": 5_000_000,
            "attachment": deductible,
            "premium_rate": 0.02 * (1 - deductible/1_000_000)  # Lower premium for higher deductible
        }
    ])

    # Run simulation and analyze
    # Your code here
```

### Exercise 2: Compare Industry Risk Profiles

```python
# Define different industry profiles
industries = {
    "tech": {"volatility": 0.35, "frequency": 0.5},
    "manufacturing": {"volatility": 0.15, "frequency": 3.0},
    "retail": {"volatility": 0.20, "frequency": 5.0}
}

# Compare optimal insurance for each
# Your code here
```

## Next Steps

Now that you understand basic simulations:

1. [Learn Optimization Techniques](/Ergodic-Insurance-Limits/tutorials/optimization_workflow)
2. [Analyze Results in Detail](/Ergodic-Insurance-Limits/tutorials/analyzing_results)
3. [Explore Advanced Scenarios](/Ergodic-Insurance-Limits/tutorials/advanced_scenarios)

---

[← Back to Getting Started](/Ergodic-Insurance-Limits/tutorials/getting_started) | [Continue to Optimization →](/Ergodic-Insurance-Limits/tutorials/optimization_workflow)

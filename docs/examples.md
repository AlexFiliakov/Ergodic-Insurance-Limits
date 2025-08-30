---
layout: default
title: Code Examples - Ergodic Insurance Framework
description: Practical code examples for common use cases
mathjax: true
---

# Code Examples

## Basic Examples

### Example 1: Simple Manufacturer Simulation

```python
from ergodic_insurance.src import Manufacturer, simulate_trajectory

# Create a manufacturer
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    base_revenue=15_000_000,
    operating_margin=0.08
)

# Run a single 20-year trajectory
trajectory = simulate_trajectory(manufacturer, years=20)

# Print final wealth
print(f"Final wealth: ${trajectory.final_wealth:,.0f}")
```

### Example 2: Insurance Comparison

```python
from ergodic_insurance.src import Manufacturer, InsuranceProgram, compare_strategies

# Define manufacturer
manufacturer = Manufacturer(starting_assets=10_000_000)

# Define insurance strategies
no_insurance = None
basic_insurance = InsuranceProgram([
    {"limit": 1_000_000, "attachment": 100_000, "premium_rate": 0.02}
])
comprehensive_insurance = InsuranceProgram([
    {"limit": 5_000_000, "attachment": 0, "premium_rate": 0.015},
    {"limit": 20_000_000, "attachment": 5_000_000, "premium_rate": 0.008}
])

# Compare strategies
results = compare_strategies(
    manufacturer,
    strategies={
        "None": no_insurance,
        "Basic": basic_insurance,
        "Comprehensive": comprehensive_insurance
    },
    n_simulations=1000
)

# Display comparison
for name, metrics in results.items():
    print(f"{name}: Growth={metrics['growth']:.2%}, Ruin={metrics['ruin']:.1%}")
```

## Advanced Examples

### Example 3: Optimization with Constraints

```python
from ergodic_insurance.src import (
    Manufacturer,
    InsuranceOptimizer,
    OptimizationConstraints
)

# Define business
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    volatility=0.20
)

# Set constraints
constraints = OptimizationConstraints(
    max_total_premium=500_000,  # Max $500k annual premium
    min_coverage=5_000_000,      # Minimum $5M coverage
    max_ruin_probability=0.01    # Max 1% ruin probability
)

# Run optimization
optimizer = InsuranceOptimizer(manufacturer, constraints)
optimal_program = optimizer.optimize()

print(f"Optimal structure found:")
for layer in optimal_program.layers:
    print(f"  ${layer['limit']:,.0f} xs ${layer['attachment']:,.0f} @ {layer['premium_rate']:.2%}")
```

### Example 4: Sensitivity Analysis

```python
import numpy as np
from ergodic_insurance.src import sensitivity_analysis

# Base case parameters
base_params = {
    "starting_assets": 10_000_000,
    "volatility": 0.15,
    "operating_margin": 0.08,
    "insurance_limit": 5_000_000,
    "premium_rate": 0.015
}

# Parameters to vary
sensitivity_params = {
    "volatility": np.linspace(0.10, 0.30, 10),
    "premium_rate": np.linspace(0.010, 0.025, 10),
    "insurance_limit": np.linspace(2_000_000, 10_000_000, 10)
}

# Run sensitivity analysis
results = sensitivity_analysis(
    base_params,
    sensitivity_params,
    metric="time_average_growth"
)

# Plot results
results.plot_tornado_chart()
results.plot_heatmap("volatility", "premium_rate")
```

### Example 5: Multi-Year Budget Planning

```python
from ergodic_insurance.src import BudgetPlanner

# Initialize planner
planner = BudgetPlanner(
    starting_assets=10_000_000,
    planning_horizon=5  # 5-year plan
)

# Add insurance options
planner.add_option("Low", limit=2_000_000, premium=30_000)
planner.add_option("Medium", limit=5_000_000, premium=75_000)
planner.add_option("High", limit=10_000_000, premium=150_000)

# Run analysis
plan = planner.optimize_multi_year()

# Display recommendations
for year, recommendation in plan.items():
    print(f"Year {year}: {recommendation['option']} (Premium: ${recommendation['premium']:,.0f})")
```

## Industry-Specific Examples

### Example 6: Manufacturing Company

```python
from ergodic_insurance.src.templates import ManufacturingTemplate

# Use manufacturing template
template = ManufacturingTemplate(
    company_size="medium",  # small, medium, large
    industry_risk="moderate",  # low, moderate, high
    geography="US"
)

# Generate configured manufacturer
manufacturer = template.create_manufacturer()
insurance = template.recommended_insurance()

print(f"Recommended insurance for {template}:")
print(insurance.summary())
```

### Example 7: Technology Startup

```python
from ergodic_insurance.src.templates import TechStartupTemplate

# High-growth tech company
startup = TechStartupTemplate(
    funding_stage="series_b",
    burn_rate=500_000,  # Monthly burn
    runway_months=18
)

# Get optimized insurance
insurance = startup.optimize_insurance(
    focus="growth",  # growth, survival, balanced
    risk_tolerance="moderate"
)

# Simulate outcomes
outcomes = startup.simulate_outcomes(insurance, n_scenarios=1000)
print(f"Probability of reaching Series C: {outcomes.success_rate:.1%}")
```

### Example 8: Retail Chain

```python
from ergodic_insurance.src.templates import RetailTemplate

# Multi-location retail
retail = RetailTemplate(
    n_locations=25,
    average_location_value=2_000_000,
    seasonal_variation=0.30  # 30% seasonal swing
)

# Analyze per-location vs blanket coverage
per_location = retail.per_location_insurance()
blanket = retail.blanket_insurance()

comparison = retail.compare_coverage_types(per_location, blanket)
print(f"Recommended: {comparison.recommendation}")
```

## Visualization Examples

### Example 9: Wealth Path Visualization

```python
from ergodic_insurance.src import visualization as viz

# Run simulation
results = simulate_ensemble(manufacturer, insurance, n=100)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Wealth trajectories
viz.plot_wealth_paths(results, ax=axes[0,0], show_median=True)

# Growth rate distribution
viz.plot_growth_distribution(results, ax=axes[0,1])

# Ruin probability over time
viz.plot_ruin_probability(results, ax=axes[1,0])

# Insurance efficiency
viz.plot_insurance_efficiency(results, ax=axes[1,1])

plt.tight_layout()
plt.savefig("analysis_results.png")
```

### Example 10: Interactive Dashboard

```python
from ergodic_insurance.src.dashboard import InteractiveDashboard

# Create dashboard
dashboard = InteractiveDashboard()

# Add data sources
dashboard.add_manufacturer(manufacturer)
dashboard.add_insurance_options([no_insurance, basic, comprehensive])

# Configure views
dashboard.add_view("trajectories", type="wealth_paths")
dashboard.add_view("metrics", type="key_metrics")
dashboard.add_view("optimization", type="interactive_optimizer")

# Launch
dashboard.run(port=8080)
# Open browser to http://localhost:8080
```

## Integration Examples

### Example 11: Excel Integration

```python
from ergodic_insurance.src.excel import ExcelInterface

# Load parameters from Excel
interface = ExcelInterface("parameters.xlsx")
manufacturer = interface.load_manufacturer("Sheet1")
insurance_options = interface.load_insurance_options("Sheet2")

# Run analysis
results = run_analysis(manufacturer, insurance_options)

# Export results to Excel
interface.export_results(results, "results.xlsx")
interface.create_charts("results.xlsx")
```

### Example 12: API Usage

```python
import requests

# API endpoint
api_url = "https://api.ergodic-insurance.com/v1"

# Submit analysis request
payload = {
    "manufacturer": {
        "starting_assets": 10_000_000,
        "volatility": 0.15
    },
    "insurance": {
        "limit": 5_000_000,
        "premium_rate": 0.015
    },
    "simulations": 1000
}

response = requests.post(f"{api_url}/analyze", json=payload)
results = response.json()

print(f"Growth rate: {results['time_average_growth']:.2%}")
print(f"API computation time: {results['compute_time']:.2f}s")
```

## Next Steps

- [Full API Documentation](/Ergodic-Insurance-Limits/api/)
- [Tutorials](/Ergodic-Insurance-Limits/tutorials/getting_started)
- [Theory Background](/Ergodic-Insurance-Limits/theory/01_ergodic_economics)
- [GitHub Repository](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits)

---

[← Back to Overview](/Ergodic-Insurance-Limits/docs/overview) | [View API Docs →](/Ergodic-Insurance-Limits/api/)

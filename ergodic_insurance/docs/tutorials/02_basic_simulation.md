# Basic Simulation

This tutorial covers the core simulation mechanics in detail, including how the business model evolves over time and how to interpret simulation results.

> **Tip:** For a quick insured-vs-uninsured comparison without managing individual objects, use `run_analysis()` — see [Tutorial 1: Getting Started](01_getting_started.md).

## The Widget Manufacturer Model

The framework models a manufacturing business with the following financial dynamics:

1. **Revenue Generation**: Revenue = Assets × Asset Turnover Ratio
2. **Operating Income**: Operating Income = Revenue × Operating Margin
3. **Net Income**: Net Income = Operating Income × (1 - Tax Rate)
4. **Growth**: Assets grow through retained earnings

### Creating a Manufacturer

```python
from ergodic_insurance import Config, ManufacturerConfig, WidgetManufacturer

# Quick start — use defaults ($10M assets, 8% margin, 50-year horizon)
config = Config()
manufacturer = WidgetManufacturer(config.manufacturer)

# Or customize specific parameters
config = Config(
    manufacturer=ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.08,
        retention_ratio=1.0,
        ppe_ratio=0.5,
    )
)
manufacturer = WidgetManufacturer(config.manufacturer)

# Check initial state
print(f"Assets: ${manufacturer.assets:,.0f}")
print(f"Equity: ${manufacturer.equity:,.0f}")
print(f"Expected Revenue: ${manufacturer.assets * config.manufacturer.asset_turnover_ratio:,.0f}")
```

## Running a Year Step-by-Step

The `step()` method advances the simulation by one year:

```python
# Run one year of operations
metrics = manufacturer.step(
    growth_rate=0.05           # 5% asset growth target
)

# Examine year-end metrics
print(f"Revenue: ${metrics['revenue']:,.0f}")
print(f"Operating Income: ${metrics['operating_income']:,.0f}")
print(f"Net Income: ${metrics['net_income']:,.0f}")
print(f"New Assets: ${metrics['assets']:,.0f}")
print(f"ROE: {metrics['roe']:.2%}")
```

## Processing Insurance Claims

When a loss occurs, the manufacturer processes it through insurance:

```python
# Simulate a large loss event
claim_amount = 2_000_000  # $2M claim

# Process through insurance with deductible and limit
manufacturer.process_insurance_claim(
    claim_amount=claim_amount,
    deductible=100_000,        # $100K retained by company
    insurance_limit=10_000_000  # $10M policy limit
)

# Run the year with the claim impact
metrics = manufacturer.step(
    letter_of_credit_rate=0.015  # 1.5% collateral cost
)

print(f"Net Income after claim: ${metrics['net_income']:,.0f}")
print(f"Outstanding Claims: ${metrics['claim_liabilities']:,.0f}")
```

## Understanding Claim Development

Claims don't pay out all at once. The framework models realistic claim development:

```python
# Claim payment schedule (cumulative percentages)
# Year 1:  10%
# Year 2:  30% (+20%)
# Year 3:  50% (+20%)
# Year 4:  65% (+15%)
# Year 5:  75% (+10%)
# Year 6:  83% (+8%)
# Year 7:  90% (+7%)
# Year 8:  95% (+5%)
# Year 9:  98% (+3%)
# Year 10: 100% (+2%)

# The manufacturer tracks collateral requirements during development
print(f"Current Collateral: ${manufacturer.collateral:,.0f}")
```

## The Simulation Class

For multi-year simulations, use the `Simulation` class:

```python
from ergodic_insurance import Simulation, ManufacturingLossGenerator

# Reset manufacturer
manufacturer = WidgetManufacturer(config)

# Create claim generator
claims = ManufacturingLossGenerator.create_simple(
    frequency=0.15,      # 15% annual claim probability
    severity_mean=800_000,    # $800K average claim
    severity_std=960_000,     # High variability (1.2x mean)
    seed=42                   # Reproducible results
)

# Create simulation
sim = Simulation(
    manufacturer=manufacturer,
    loss_generator=claims,
    time_horizon=30           # 30-year horizon
)

# Run simulation
results = sim.run()
```

## Analyzing Simulation Results

### Basic Statistics

```python
import numpy as np

print("=== Simulation Summary ===")
print(f"Survived: {'Yes' if results.insolvency_year is None else 'No'}")
print(f"Final Assets: ${results.assets[-1]:,.0f}")
print(f"Final Equity: ${results.equity[-1]:,.0f}")
print(f"Total Claims: {results.claim_counts.sum():.0f}")
print(f"Total Claim Amounts: ${results.claim_amounts.sum():,.0f}")

# ROE analysis
valid_roe = results.roe[~np.isnan(results.roe)]
print(f"\nROE Statistics:")
print(f"  Mean: {np.mean(valid_roe):.2%}")
print(f"  Median: {np.median(valid_roe):.2%}")
print(f"  Std Dev: {np.std(valid_roe):.2%}")
print(f"  Min: {np.min(valid_roe):.2%}")
print(f"  Max: {np.max(valid_roe):.2%}")
```

### Time-Weighted vs Simple Average ROE

The time-weighted ROE captures the true compound growth experience:

```python
# Simple arithmetic mean
simple_avg = np.mean(valid_roe)

# Time-weighted (geometric) mean
time_weighted = results.calculate_time_weighted_roe()

# Rolling ROE for trend analysis
rolling_5yr = results.calculate_rolling_roe(window=5)

print(f"Simple Average ROE: {simple_avg:.2%}")
print(f"Time-Weighted ROE: {time_weighted:.2%}")
print(f"Difference: {(simple_avg - time_weighted):.2%}")

# The difference indicates non-ergodic behavior
if simple_avg > time_weighted:
    print("-> Volatility drag is reducing actual growth")
```

### Export to DataFrame

```python
import pandas as pd

# Full time series
df = results.to_dataframe()

# Add derived metrics
df['cumulative_return'] = (df['equity'] / df['equity'].iloc[0]) - 1
df['year_over_year_growth'] = df['assets'].pct_change()

# Display
print(df.head(10))

# Save to file
df.to_csv('simulation_results.csv', index=False)
```

## Visualizing Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Asset growth
axes[0, 0].plot(results.years, results.assets / 1e6)
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Assets ($M)')
axes[0, 0].set_title('Asset Growth Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Equity evolution
axes[0, 1].plot(results.years, results.equity / 1e6)
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Equity ($M)')
axes[0, 1].set_title('Equity Evolution')
axes[0, 1].grid(True, alpha=0.3)

# ROE over time
axes[1, 0].bar(results.years, results.roe * 100, alpha=0.7)
axes[1, 0].axhline(y=simple_avg * 100, color='r', linestyle='--', label='Mean')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('ROE (%)')
axes[1, 0].set_title('Annual Return on Equity')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Claim events
axes[1, 1].bar(results.years, results.claim_amounts / 1e6, color='orange', alpha=0.7)
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Claim Amount ($M)')
axes[1, 1].set_title('Claim Events')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_visualization.png', dpi=150)
plt.show()
```

## Handling Insolvency

When equity falls below the insolvency tolerance, the simulation terminates:

```python
# Check for insolvency
if results.insolvency_year is not None:
    print(f"Company went insolvent in year {results.insolvency_year}")

    # Analyze pre-insolvency trajectory
    pre_insolvency = results.equity[:results.insolvency_year + 1]
    print(f"Peak equity: ${max(pre_insolvency):,.0f}")
    print(f"Final equity: ${pre_insolvency[-1]:,.0f}")
else:
    print("Company survived the full simulation period")

    # Calculate compound annual growth rate (CAGR)
    initial = results.equity[0]
    final = results.equity[-1]
    years = len(results.years)
    cagr = (final / initial) ** (1 / years) - 1
    print(f"Equity CAGR: {cagr:.2%}")
```

## Multiple Simulations with Different Seeds

To understand the range of outcomes:

```python
from ergodic_insurance import ManufacturerConfig, WidgetManufacturer, ManufacturingLossGenerator, Simulation

# Run multiple simulations
n_simulations = 10
outcomes = []

for seed in range(n_simulations):
    # Fresh manufacturer for each run
    mfg = WidgetManufacturer(config)

    # Claim generator with different seed
    claims = ManufacturingLossGenerator.create_simple(
        frequency=0.15,
        severity_mean=800_000,
        severity_std=960_000,
        seed=seed
    )

    # Run simulation
    sim = Simulation(manufacturer=mfg, loss_generator=claims, time_horizon=30)
    results = sim.run()

    outcomes.append({
        'seed': seed,
        'survived': results.insolvency_year is None,
        'final_equity': results.equity[-1] if results.insolvency_year is None else 0,
        'time_weighted_roe': results.calculate_time_weighted_roe()
    })

# Summarize outcomes
survived = sum(1 for o in outcomes if o['survived'])
print(f"\nSurvival Rate: {survived}/{n_simulations} ({survived/n_simulations:.0%})")
print(f"Mean Final Equity (survivors): ${np.mean([o['final_equity'] for o in outcomes if o['survived']]):,.0f}")
print(f"Mean Time-Weighted ROE: {np.mean([o['time_weighted_roe'] for o in outcomes]):.2%}")
```

## Next Steps

- [Tutorial 3: Configuring Insurance](03_configuring_insurance.md) - Add insurance to your simulation
- [Tutorial 4: Optimization Workflow](04_optimization_workflow.md) -- Use the optimizer to automatically find the best deductible and limit for your business
- [Tutorial 5: Analyzing Results](05_analyzing_results.md) -- Deep dive into ergodic analysis, volatility drag, and DuPont decomposition
- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) -- Monte Carlo simulations, market cycles, and multi-line programs

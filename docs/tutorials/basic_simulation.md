---
layout: default
title: Basic Simulation
---

# Basic Simulation Tutorial

Learn the fundamentals of running insurance simulations with the Ergodic Insurance Framework.

## Overview

This tutorial covers:
- Simulation architecture
- Key parameters and their effects
- Loss modeling approaches
- Understanding simulation outputs
- Performance optimization

## Simulation Architecture

The framework uses Monte Carlo simulation to model company evolution:

```python
from ergodic_insurance.src.simulation import Simulation
from ergodic_insurance.src.manufacturer import Manufacturer
from ergodic_insurance.src.config_v2 import SimulationConfig

# Initialize components
manufacturer = Manufacturer(starting_assets=10_000_000)
config = SimulationConfig(
    simulation_years=50,
    num_simulations=10_000
)

# Create and run simulation
sim = Simulation(manufacturer, config)
results = sim.run()
```

## Core Components

### 1. The Manufacturer Model

Represents a business entity with:
- Assets and revenue generation
- Operating costs and margins
- Growth dynamics
- Loss exposure

```python
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    asset_turnover=1.0,  # Revenue = 1x Assets
    operating_margin=0.08,  # 8% profit margin
    growth_volatility=0.15,  # 15% annual volatility
    tax_rate=0.25  # 25% corporate tax
)
```

### 2. Loss Generation

Losses follow a compound Poisson process:

```python
from ergodic_insurance.src.claim_generator import ClaimGenerator

claim_gen = ClaimGenerator(
    frequency_lambda=0.5,  # 0.5 losses per year on average
    severity_mean=2_000_000,  # $2M average loss
    severity_cv=2.0,  # High variability
    distribution_type="lognormal"
)

# Generate losses for one year
annual_losses = claim_gen.generate_annual_losses()
```

### 3. Stochastic Processes

Multiple models for uncertainty:

```python
from ergodic_insurance.src.stochastic_processes import (
    GeometricBrownianMotion,
    MeanRevertingProcess
)

# Revenue growth with GBM
gbm = GeometricBrownianMotion(
    drift=0.05,  # 5% expected growth
    volatility=0.15,  # 15% volatility
    dt=1/252  # Daily steps
)

# Mean-reverting operating margin
mr_process = MeanRevertingProcess(
    long_term_mean=0.08,
    mean_reversion_speed=0.5,
    volatility=0.02
)
```

## Running Simulations

### Basic Simulation Loop

```python
import numpy as np

def run_single_simulation(manufacturer, years, claim_generator):
    """Run one simulation path."""
    assets = [manufacturer.starting_assets]

    for year in range(years):
        # Generate revenue
        revenue = assets[-1] * manufacturer.asset_turnover

        # Calculate operating income
        operating_income = revenue * manufacturer.operating_margin

        # Generate and apply losses
        losses = claim_generator.generate_annual_losses()

        # Update assets
        new_assets = assets[-1] + operating_income - losses
        assets.append(max(0, new_assets))  # Can't go negative

        # Check for ruin
        if new_assets <= 0:
            break

    return np.array(assets)
```

### Parallel Processing

For better performance:

```python
from ergodic_insurance.src.parallel_executor import ParallelExecutor

executor = ParallelExecutor(n_workers=8)
results = executor.run_parallel_simulations(
    manufacturer=manufacturer,
    config=config,
    n_simulations=10_000
)
```

## Key Parameters

### Company Parameters

| Parameter | Typical Range | Impact |
|-----------|--------------|--------|
| Asset Turnover | 0.5 - 2.0 | Higher = more revenue per dollar of assets |
| Operating Margin | 5% - 15% | Higher = more profitable |
| Growth Volatility | 10% - 30% | Higher = more uncertainty |
| Working Capital | 15% - 25% | Higher = more liquidity needs |

### Loss Parameters

| Parameter | Typical Range | Impact |
|-----------|--------------|--------|
| Frequency | 0.1 - 2.0/year | More events = higher risk |
| Severity Mean | $100K - $10M | Larger losses = more insurance value |
| Severity CV | 1.0 - 3.0 | Higher = fat-tailed distribution |

### Simulation Parameters

| Parameter | Typical Range | Impact |
|-----------|--------------|--------|
| Years | 10 - 100 | Longer = better ergodic estimates |
| Paths | 1K - 100K | More = higher accuracy |
| Time Steps | 1 - 252/year | Finer = more accurate but slower |

## Understanding Outputs

### Growth Metrics

```python
# Calculate time-average growth
def calculate_time_average_growth(asset_path):
    """Calculate geometric growth rate."""
    if len(asset_path) < 2:
        return 0

    years = len(asset_path) - 1
    final = asset_path[-1]
    initial = asset_path[0]

    if final <= 0 or initial <= 0:
        return -1  # Ruin

    return (final / initial) ** (1/years) - 1

# Calculate ensemble average
ensemble_growth = np.mean([
    calculate_time_average_growth(path)
    for path in all_paths
])
```

### Risk Metrics

```python
from ergodic_insurance.src.risk_metrics import RiskMetrics

metrics = RiskMetrics(simulation_results)

print(f"Ruin Probability: {metrics.ruin_probability:.2%}")
print(f"95% VaR: ${metrics.var_95:,.0f}")
print(f"99% CVaR: ${metrics.cvar_99:,.0f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

## Visualization

### Growth Paths

```python
import matplotlib.pyplot as plt

def plot_simulation_paths(results, n_paths=100):
    """Plot sample of simulation paths."""
    plt.figure(figsize=(12, 6))

    for i in range(min(n_paths, len(results.paths))):
        plt.plot(results.paths[i], alpha=0.3, color='blue')

    # Plot mean path
    mean_path = np.mean(results.paths, axis=0)
    plt.plot(mean_path, color='red', linewidth=2, label='Mean')

    # Plot median path
    median_path = np.median(results.paths, axis=0)
    plt.plot(median_path, color='green', linewidth=2, label='Median')

    plt.xlabel('Years')
    plt.ylabel('Assets ($)')
    plt.title('Simulation Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### Distribution Analysis

```python
def plot_final_distribution(results):
    """Plot distribution of final outcomes."""
    final_values = [path[-1] for path in results.paths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(final_values, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(final_values), color='red', label='Mean')
    ax1.axvline(np.median(final_values), color='green', label='Median')
    ax1.set_xlabel('Final Assets ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Final Assets')
    ax1.legend()

    # Log scale
    ax2.hist(np.log10(final_values + 1), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Log10(Final Assets)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Log Distribution')

    plt.tight_layout()
    plt.show()
```

## Performance Optimization

### 1. Use Caching

```python
from ergodic_insurance.src.monte_carlo import MonteCarloCache

cache = MonteCarloCache("./cache")
results = cache.get_or_compute(
    key="baseline_simulation",
    compute_fn=lambda: sim.run(),
    force_recompute=False
)
```

### 2. Batch Processing

```python
from ergodic_insurance.src.batch_processor import BatchProcessor

processor = BatchProcessor(batch_size=1000)
results = processor.process_simulations(
    total_simulations=100_000,
    simulation_fn=run_single_simulation
)
```

### 3. Vectorization

```python
# Vectorized loss generation
def generate_losses_vectorized(n_years, n_sims):
    """Generate all losses at once."""
    # Frequency: Poisson
    frequencies = np.random.poisson(0.5, (n_sims, n_years))

    # Severity: Lognormal
    max_losses = frequencies.max()
    severities = np.random.lognormal(14, 2, (n_sims, n_years, max_losses))

    # Total losses per year
    losses = np.zeros((n_sims, n_years))
    for i in range(n_sims):
        for j in range(n_years):
            losses[i, j] = severities[i, j, :frequencies[i, j]].sum()

    return losses
```

## Advanced Features

### Scenario Analysis

```python
scenarios = {
    "baseline": {"frequency": 0.5, "severity_mean": 2_000_000},
    "stressed": {"frequency": 1.0, "severity_mean": 5_000_000},
    "benign": {"frequency": 0.2, "severity_mean": 500_000}
}

scenario_results = {}
for name, params in scenarios.items():
    claim_gen = ClaimGenerator(**params)
    results = run_simulation(manufacturer, claim_gen)
    scenario_results[name] = results
```

### Sensitivity Analysis

```python
from ergodic_insurance.src.sensitivity import sensitivity_analysis

# Define parameter ranges
param_ranges = {
    "operating_margin": np.linspace(0.05, 0.15, 11),
    "growth_volatility": np.linspace(0.10, 0.30, 11),
    "loss_frequency": np.linspace(0.1, 1.0, 10)
}

# Run sensitivity analysis
sensitivity_results = sensitivity_analysis(
    base_case=manufacturer,
    param_ranges=param_ranges,
    metric="growth_rate"
)

# Plot tornado diagram
plot_tornado_diagram(sensitivity_results)
```

## Best Practices

1. **Start Simple**: Begin with basic parameters, add complexity gradually
2. **Validate Results**: Check against analytical solutions when possible
3. **Use Fixed Seeds**: For reproducibility during development
4. **Monitor Convergence**: Ensure sufficient simulation paths
5. **Profile Performance**: Identify bottlenecks before scaling up

## Common Pitfalls

1. **Too Few Simulations**: Use at least 1,000 paths for initial estimates
2. **Short Time Horizons**: Ergodic effects need time to manifest
3. **Ignoring Correlations**: Real-world risks are often correlated
4. **Unrealistic Parameters**: Validate against industry benchmarks

## Summary

You now understand:
- How the simulation engine works
- Key parameters and their impacts
- How to interpret results
- Performance optimization techniques

## Next Steps

Continue to:
- [Configuring Insurance](configuring_insurance.md) - Design insurance programs
- [Optimization Workflow](optimization_workflow.md) - Automate parameter selection

For code examples, see the notebooks in `ergodic_insurance/notebooks/`.

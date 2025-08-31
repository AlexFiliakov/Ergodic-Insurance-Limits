---
layout: default
title: Getting Started
---

# Getting Started with Ergodic Insurance Optimization

This tutorial will help you install the framework and run your first insurance optimization analysis.

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager
- Basic command line knowledge

### Step 1: Clone the Repository

```bash
git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
cd Ergodic-Insurance-Limits
```

### Step 2: Set Up Python Environment

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Step 3: Verify Installation

```python
import ergodic_insurance
print(ergodic_insurance.__version__)
```

## Your First Simulation

Let's run a simple insurance optimization for a widget manufacturer:

```python
from ergodic_insurance.src.manufacturer import Manufacturer
from ergodic_insurance.src.simulation import run_simulation
from ergodic_insurance.src.config_v2 import SimulationConfig

# Create a manufacturer with $10M in assets
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    asset_turnover=1.0,  # Revenue = Assets
    operating_margin=0.08,  # 8% profit margin
)

# Configure simulation
config = SimulationConfig(
    simulation_years=20,
    num_simulations=1000,
    random_seed=42
)

# Run simulation
results = run_simulation(manufacturer, config)

# Display results
print(f"Average annual growth: {results.growth_rate:.2%}")
print(f"Ruin probability: {results.ruin_probability:.2%}")
```

## Understanding the Results

The simulation provides key metrics:

1. **Growth Rate**: Annual compound growth rate of assets
2. **Ruin Probability**: Chance of bankruptcy
3. **Volatility**: Standard deviation of returns
4. **Insurance Benefit**: Improvement from insurance

## Adding Insurance

Now let's add insurance to see the improvement:

```python
from ergodic_insurance.src.insurance import Insurance

# Define insurance program
insurance = Insurance(
    retention=250_000,  # $250K deductible
    limit=5_000_000,    # $5M coverage limit
    annual_premium=150_000  # $150K annual cost
)

# Run simulation with insurance
results_with_insurance = run_simulation(
    manufacturer,
    config,
    insurance=insurance
)

# Compare results
print("\nWithout Insurance:")
print(f"  Growth: {results.growth_rate:.2%}")
print(f"  Ruin Prob: {results.ruin_probability:.2%}")

print("\nWith Insurance:")
print(f"  Growth: {results_with_insurance.growth_rate:.2%}")
print(f"  Ruin Prob: {results_with_insurance.ruin_probability:.2%}")
```

## Key Concepts

### Ergodic vs. Ensemble

- **Ensemble Average**: Expected value across many companies
- **Time Average**: Your company's actual growth over time
- **Key Insight**: Insurance improves time-average growth even when "expensive"

### Why Insurance Helps Growth

1. **Eliminates ruin paths**: Can't grow if bankrupt
2. **Reduces volatility drag**: Lower volatility = higher compound growth
3. **Enables aggressive strategies**: Take more operational risk with downside protection

## Next Steps

Congratulations! You've run your first ergodic insurance analysis.

Next tutorials:
- [Basic Simulation](basic_simulation.md) - Deeper dive into simulation mechanics
- [Configuring Insurance](configuring_insurance.md) - Design optimal insurance programs
- [Optimization Workflow](optimization_workflow.md) - Find the best parameters automatically

## Quick Reference

### Common Parameters

```python
# Company parameters
starting_assets = 10_000_000  # Initial capital
asset_turnover = 1.0          # Revenue/Assets ratio
operating_margin = 0.08       # Profit margin

# Loss parameters
loss_frequency = 0.5          # Events per year
loss_severity_mean = 2_000_000  # Average loss size
loss_severity_cv = 2.0        # Coefficient of variation

# Insurance parameters
retention = 250_000           # Deductible
limit = 5_000_000            # Coverage limit
premium_rate = 0.03          # Premium as % of limit
```

### Useful Functions

```python
# Run optimization
from ergodic_insurance.src.optimization import optimize_insurance
optimal = optimize_insurance(manufacturer, config)

# Plot results
from ergodic_insurance.src.visualization import plot_growth_paths
plot_growth_paths(results)

# Sensitivity analysis
from ergodic_insurance.src.sensitivity import analyze_sensitivity
sensitivity = analyze_sensitivity(manufacturer, parameter_ranges)
```

## Troubleshooting

### Common Issues

**ImportError**: Make sure you've activated the virtual environment
**Slow performance**: Reduce `num_simulations` for faster testing
**Memory issues**: Use smaller `simulation_years` or batch processing

For more help, see the [Troubleshooting Guide](troubleshooting.md).

Ready to learn more? Continue to [Basic Simulation](basic_simulation.md)!

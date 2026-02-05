# Getting Started with Ergodic Insurance Limits

This tutorial introduces the Ergodic Insurance Limits framework and guides you through installation and your first simulation.

## What is Ergodic Insurance Optimization?

Traditional insurance analysis uses **ensemble averages** - the expected value across many parallel scenarios. This works well for large insurers diversifying across thousands of policies, but it doesn't capture what happens to a **single business over time**.

Ergodic economics uses **time averages** - what actually happens to one entity as it evolves through time. For businesses facing large, volatile losses, the time-average growth rate can differ dramatically from ensemble expectations.

**Key takeaway**: Insurance that appears "expensive" from an expected-value perspective may actually *increase* long-term wealth growth when analyzed through time averages.

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager
- Git (optional, for cloning the repository)

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
cd "Ergodic Insurance Limits/ergodic_insurance"
```

2. Create and activate a virtual environment:

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Unix/macOS
python -m venv .venv
source .venv/bin/activate
```

3. Install the package:

```bash
# Using pip
pip install -e ".[dev,notebooks]"

# Or using uv (recommended)
uv sync
```

4. Verify the installation:

```bash
python -c "import ergodic_insurance; print(ergodic_insurance.__version__)"
```

## Your First Simulation

Let's run a simple simulation to see the framework in action.

### Step 1: Import the Core Components

```python
from ergodic_insurance import Config, ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.simulation import Simulation
```

### Step 2: Configure a Manufacturer

Create a configuration for a widget manufacturing company:

```python
# Define manufacturer financial parameters
manufacturer_config = ManufacturerConfig(
    initial_assets=10_000_000,      # $10M starting assets
    asset_turnover_ratio=1.0,       # Revenue = 1x assets
    base_operating_margin=0.08,     # 8% operating margin
    tax_rate=0.25,                  # 25% corporate tax rate
    retention_ratio=1.0             # Retain all earnings (no dividends)
)

# Create the manufacturer instance
manufacturer = WidgetManufacturer(manufacturer_config)

print(f"Initial Equity: ${manufacturer.equity:,.0f}")
print(f"Initial Cash: ${manufacturer.cash:,.0f}")
```

**Expected Output:**
```
Initial Equity: $10,000,000
Initial Cash: $7,000,000
```

### Step 3: Configure Loss Generation

Set up a loss generator to simulate random loss events:

```python
# Create a loss generator with:
# - 10% annual loss frequency (average 0.1 losses per year)
# - Lognormal severity distribution (mean $500K, std $500K)
# - Reproducible results with seed
loss_generator = ManufacturingLossGenerator.create_simple(
    frequency=0.1,            # Expected losses per year
    severity_mean=500_000,    # Average loss size
    severity_std=500_000,     # Standard deviation
    seed=42
)
```

### Step 4: Run a Basic Simulation

```python
# Create and run the simulation
# Note: Simulation will use the loss generator internally
simulation = Simulation(
    manufacturer=manufacturer,
    claim_generator=loss_generator,  # Accepts both ClaimGenerator and ManufacturingLossGenerator
    time_horizon=20  # Simulate 20 years
)

results = simulation.run()

# Display summary statistics
print("\n=== Simulation Results ===")
print(f"Final Assets: ${results.assets[-1]:,.0f}")
print(f"Final Equity: ${results.equity[-1]:,.0f}")
print(f"Mean ROE: {results.roe.mean():.2%}")
print(f"Survived: {'Yes' if results.insolvency_year is None else f'No (Year {results.insolvency_year})'}")
```

### Step 5: Examine the Results

The `SimulationResults` object contains detailed time series data:

```python
import pandas as pd

# Convert to DataFrame for easy analysis
df = results.to_dataframe()
print("\nYear-by-year results:")
print(df.head(10).to_string())

# Calculate time-weighted ROE (ergodic measure)
time_weighted_roe = results.calculate_time_weighted_roe()
print(f"\nTime-weighted ROE: {time_weighted_roe:.2%}")
print(f"Simple average ROE: {results.roe.mean():.2%}")
```

## Understanding the Output

| Metric | Description |
|--------|-------------|
| `assets` | Total assets at each year-end |
| `equity` | Shareholder equity (assets minus liabilities) |
| `roe` | Return on equity for each year |
| `revenue` | Annual revenue generated |
| `net_income` | Annual profit after taxes |
| `claim_counts` | Number of claims in each year |
| `claim_amounts` | Total claim dollars in each year |
| `insolvency_year` | Year of bankruptcy (None if survived) |

## Next Steps

Now that you've run your first simulation, continue with:

- [Tutorial 2: Basic Simulation](02_basic_simulation.md) - Deeper dive into simulation mechanics
- [Tutorial 3: Configuring Insurance](03_configuring_insurance.md) - Add insurance to your simulation
- [Tutorial 4: Optimization Workflow](04_optimization_workflow.md) -- Use the optimizer to automatically find the best deductible and limit for your business
- [Tutorial 5: Analyzing Results](05_analyzing_results.md) -- Deep dive into ergodic analysis, volatility drag, and DuPont decomposition
- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) -- Monte Carlo simulations, market cycles, and multi-line programs

## Quick Reference

### Key Classes

| Class | Purpose |
|-------|---------|
| `ManufacturerConfig` | Configure business financial parameters |
| `WidgetManufacturer` | Business model with financial operations |
| `ManufacturingLossGenerator` | Generate random loss events (recommended) |
| `ClaimGenerator` | ⚠️ **Deprecated** - Use ManufacturingLossGenerator instead |
| `Simulation` | Run time evolution of the business |
| `SimulationResults` | Container for simulation output |

### Common Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `initial_assets` | $1M - $100M | Starting capital |
| `asset_turnover_ratio` | 0.5 - 2.0 | Revenue efficiency |
| `base_operating_margin` | 5% - 15% | Profitability before insurance |
| `base_frequency` | 0.05 - 0.5 | Claims per year |
| `time_horizon` | 10 - 100 years | Simulation length |

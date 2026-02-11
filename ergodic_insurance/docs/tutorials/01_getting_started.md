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

## Your First Analysis

The fastest way to get results is `run_analysis()` — one import, one call:

```python
from ergodic_insurance import run_analysis

results = run_analysis(
    initial_assets=10_000_000,       # $10M starting assets
    operating_margin=0.08,           # 8% operating margin
    loss_frequency=2.5,              # ~2.5 losses per year
    loss_severity_mean=1_000_000,    # $1M average loss
    deductible=500_000,              # $500K self-insured retention
    coverage_limit=10_000_000,       # $10M policy limit
    premium_rate=0.025,              # 2.5% rate on limit
    n_simulations=1000,              # Monte Carlo paths
    time_horizon=20,                 # 20-year horizon
)

print(results.summary())
```

That's it — `run_analysis` builds the manufacturer, loss model, insurance
policy, and simulation engine internally, runs both *insured* and
*uninsured* Monte Carlo batches, and returns a rich result object.

### Inspect the Results

```python
# Export per-simulation metrics to a DataFrame
df = results.to_dataframe()
print(df.head())

# Quick 2x2 comparison chart
results.plot()
```

### Under the Hood

`run_analysis` is a convenience wrapper around the framework's building
blocks. Power users can import them directly:

```python
from ergodic_insurance import Config, Simulation
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
```

See [Tutorial 2: Basic Simulation](02_basic_simulation.md) for a deeper
dive into the individual components.

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

| Class / Function | Purpose |
|------------------|---------|
| `run_analysis()` | **Quick-start** — one call for a full insured-vs-uninsured comparison |
| `AnalysisResults` | Container returned by `run_analysis` with `.summary()`, `.to_dataframe()`, `.plot()` |
| `ManufacturerConfig` | Configure business financial parameters |
| `WidgetManufacturer` | Business model with financial operations |
| `ManufacturingLossGenerator` | Generate random loss events (recommended) |
| `Simulation` | Run time evolution of the business |
| `SimulationResults` | Container for single-path simulation output |
| `MonteCarloResults` | Container for Monte Carlo simulation output |

### Common Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `initial_assets` | $1M - $100M | Starting capital |
| `asset_turnover_ratio` | 0.5 - 2.0 | Revenue efficiency |
| `base_operating_margin` | 5% - 15% | Profitability before insurance |
| `base_frequency` | 0.05 - 0.5 | Claims per year |
| `time_horizon` | 10 - 100 years | Simulation length |

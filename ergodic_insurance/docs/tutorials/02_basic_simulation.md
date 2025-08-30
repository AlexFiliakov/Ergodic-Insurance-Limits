# Basic Simulation Tutorial

This tutorial provides a comprehensive guide to running simulations with the Ergodic Insurance Framework. You'll learn how to configure manufacturers, generate realistic losses, and run various simulation scenarios.

## Learning Objectives

After completing this tutorial, you will be able to:
- Configure manufacturer models with realistic parameters
- Generate different types of loss distributions
- Run Monte Carlo simulations
- Analyze simulation results statistically
- Visualize outcomes effectively

## Setting Up a Manufacturer

The `Manufacturer` class models a company's financial dynamics. Let's explore its key parameters:

### Basic Configuration

```python
from ergodic_insurance.src.manufacturer import Manufacturer
import numpy as np

# Create a medium-sized manufacturer
manufacturer = Manufacturer(
    initial_assets=10_000_000,     # \$10M starting assets
    asset_turnover=1.0,             # Revenue = 1x assets
    operating_margin=0.08,          # 8% operating margin
    tax_rate=0.25,                  # 25% corporate tax
    working_capital_ratio=0.20      # 20% of revenue tied up in WC
)

# Calculate key metrics
annual_revenue = manufacturer.initial_assets * manufacturer.asset_turnover
operating_income = annual_revenue * manufacturer.operating_margin
net_income = operating_income * (1 - manufacturer.tax_rate)

print(f"Company Financial Profile:")
print(f"  Assets: ${manufacturer.initial_assets:,.0f}")
print(f"  Annual Revenue: ${annual_revenue:,.0f}")
print(f"  Operating Income: ${operating_income:,.0f}")
print(f"  Net Income: ${net_income:,.0f}")
print(f"  ROA: {net_income / manufacturer.initial_assets:.1%}")
```

### Industry-Specific Configurations

Different industries have different financial characteristics:

```python
# High-margin software company
software_company = Manufacturer(
    initial_assets=5_000_000,
    asset_turnover=2.0,      # High efficiency
    operating_margin=0.25,   # 25% margins
    tax_rate=0.21
)

# Capital-intensive manufacturing
heavy_industry = Manufacturer(
    initial_assets=50_000_000,
    asset_turnover=0.5,      # Low turnover
    operating_margin=0.05,   # 5% margins
    tax_rate=0.25
)

# Retail business
retail_store = Manufacturer(
    initial_assets=2_000_000,
    asset_turnover=4.0,      # Very high turnover
    operating_margin=0.03,   # 3% margins
    tax_rate=0.25
)

# Compare profitability
for company, name in [(software_company, "Software"),
                      (heavy_industry, "Heavy Industry"),
                      (retail_store, "Retail")]:
    revenue = company.initial_assets * company.asset_turnover
    profit = revenue * company.operating_margin * (1 - company.tax_rate)
    roe = profit / company.initial_assets
    print(f"{name}: ROE = {roe:.1%}")
```

## Generating Losses

The `ClaimGenerator` creates realistic loss scenarios. Let's explore different loss patterns:

### Basic Loss Generation

```python
from ergodic_insurance.src.claim_generator import ClaimGenerator

# Standard loss generator
standard_losses = ClaimGenerator(
    frequency=5,           # 5 losses per year on average
    severity_mu=10.0,      # Log-mean (median ~\$22K)
    severity_sigma=1.5     # Log-std (high variability)
)

# Generate 5 years of losses
np.random.seed(42)
losses_by_year = []
for year in range(5):
    annual_losses = standard_losses.generate_claims(years=1)
    losses_by_year.append(annual_losses)
    total = sum(annual_losses) if annual_losses else 0
    print(f"Year {year+1}: {len(annual_losses)} losses, Total: ${total:,.0f}")
```

### Different Risk Profiles

```python
# Low frequency, high severity (catastrophic risk)
catastrophic_risk = ClaimGenerator(
    frequency=0.5,         # One loss every 2 years
    severity_mu=14.0,      # Much larger losses
    severity_sigma=2.0     # More variability
)

# High frequency, low severity (operational risk)
operational_risk = ClaimGenerator(
    frequency=20,          # Many small losses
    severity_mu=8.0,       # Smaller losses
    severity_sigma=0.5     # Less variability
)

# Simulate and compare
np.random.seed(42)
years = 10
risk_profiles = {
    "Standard": standard_losses,
    "Catastrophic": catastrophic_risk,
    "Operational": operational_risk
}

for name, generator in risk_profiles.items():
    all_losses = []
    for _ in range(years):
        annual = generator.generate_claims(years=1)
        all_losses.extend(annual)

    if all_losses:
        print(f"\n{name} Risk Profile ({years} years):")
        print(f"  Total losses: {len(all_losses)}")
        print(f"  Average loss: ${np.mean(all_losses):,.0f}")
        print(f"  Largest loss: ${max(all_losses):,.0f}")
        print(f"  Total amount: ${sum(all_losses):,.0f}")
```

## Running Simulations

Now let's run comprehensive simulations with different scenarios:

### Single Path Simulation

```python
from ergodic_insurance.src.simulation import Simulation

# Create simulation
sim = Simulation(
    manufacturer=manufacturer,
    claim_generator=standard_losses
)

# Run single 20-year simulation
np.random.seed(42)
result = sim.run(
    n_years=20,
    retention=1_000_000,    # \$1M deductible
    limit=10_000_000,       # \$10M coverage
    premium_rate=0.02       # 2% of limit
)

print(f"20-Year Simulation Results:")
print(f"  Starting Wealth: ${manufacturer.initial_assets:,.0f}")
print(f"  Final Wealth: ${result.final_wealth:,.0f}")
print(f"  Annualized Growth: {result.growth_rate:.2%}")
print(f"  Survived: {result.survived}")
print(f"  Max Drawdown: ${min(result.wealth_trajectory) - manufacturer.initial_assets:,.0f}")
```

### Monte Carlo Simulation

For robust analysis, we need multiple simulation paths:

```python
from ergodic_insurance.src.monte_carlo import MonteCarloAnalyzer

# Create Monte Carlo analyzer
mc_analyzer = MonteCarloAnalyzer(
    manufacturer=manufacturer,
    claim_generator=standard_losses
)

# Run 1000 simulations
mc_results = mc_analyzer.run_simulations(
    n_simulations=1000,
    n_years=20,
    retention=1_000_000,
    limit=10_000_000,
    premium_rate=0.02,
    seed=42
)

# Analyze results
growth_rates = mc_results['growth_rates']
survival_rates = mc_results['survival_rate']
final_wealths = mc_results['final_wealths']

print(f"\nMonte Carlo Results (1000 simulations):")
print(f"  Survival Rate: {survival_rates:.1%}")
print(f"  Mean Growth Rate: {np.mean(growth_rates):.2%}")
print(f"  Median Growth Rate: {np.median(growth_rates):.2%}")
print(f"  95% VaR Growth: {np.percentile(growth_rates, 5):.2%}")
print(f"  Mean Final Wealth: ${np.mean(final_wealths):,.0f}")
```

### Comparing Insurance Strategies

```python
# Define different insurance strategies
strategies = [
    {"name": "No Insurance", "retention": float('inf'), "limit": 0, "premium_rate": 0},
    {"name": "High Deductible", "retention": 2_000_000, "limit": 10_000_000, "premium_rate": 0.015},
    {"name": "Medium Coverage", "retention": 500_000, "limit": 5_000_000, "premium_rate": 0.02},
    {"name": "Full Coverage", "retention": 100_000, "limit": 20_000_000, "premium_rate": 0.025}
]

# Run simulations for each strategy
results_comparison = {}
for strategy in strategies:
    mc_results = mc_analyzer.run_simulations(
        n_simulations=500,
        n_years=20,
        retention=strategy['retention'],
        limit=strategy['limit'],
        premium_rate=strategy['premium_rate'],
        seed=42
    )
    results_comparison[strategy['name']] = mc_results

# Compare strategies
print("\nStrategy Comparison:")
print(f"{'Strategy':<20} {'Survival':<12} {'Mean Growth':<12} {'Volatility':<12}")
print("-" * 56)
for name, results in results_comparison.items():
    survival = results['survival_rate']
    mean_growth = np.mean(results['growth_rates'])
    vol = np.std(results['growth_rates'])
    print(f"{name:<20} {survival:>10.1%}   {mean_growth:>10.2%}   {vol:>10.2%}")
```

## Visualizing Results

Effective visualization helps understand simulation outcomes:

### Wealth Trajectories

```python
import matplotlib.pyplot as plt

# Run simulation with multiple seeds
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
strategies_to_plot = strategies[:4]

for ax, strategy in zip(axes.flat, strategies_to_plot):
    # Run 10 simulations
    trajectories = []
    for seed in range(10):
        np.random.seed(seed)
        result = sim.run(
            n_years=20,
            retention=strategy['retention'],
            limit=strategy['limit'],
            premium_rate=strategy['premium_rate']
        )
        trajectories.append(result.wealth_trajectory)

    # Plot trajectories
    for traj in trajectories:
        ax.plot(traj, alpha=0.5, linewidth=0.8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=manufacturer.initial_assets, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(strategy['name'])
    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth ($)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Distribution of Outcomes

```python
# Plot distribution of final wealth
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Growth rate distribution
ax1 = axes[0]
for name, results in results_comparison.items():
    ax1.hist(results['growth_rates'], bins=30, alpha=0.5, label=name)
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Annualized Growth Rate')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Growth Rates')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Survival analysis
ax2 = axes[1]
names = list(results_comparison.keys())
survival_rates = [results_comparison[name]['survival_rate'] for name in names]
colors = ['red' if sr < 0.95 else 'green' for sr in survival_rates]
ax2.bar(names, survival_rates, color=colors, alpha=0.7)
ax2.axhline(y=0.95, color='blue', linestyle='--', alpha=0.5, label='95% Target')
ax2.set_ylabel('Survival Rate')
ax2.set_title('Survival Probability by Strategy')
ax2.legend()
ax2.set_ylim([0, 1.05])
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Risk-Return Scatter

```python
# Create risk-return scatter plot
plt.figure(figsize=(10, 8))

for name, results in results_comparison.items():
    mean_return = np.mean(results['growth_rates'])
    risk = 1 - results['survival_rate']  # Ruin probability as risk

    plt.scatter(risk * 100, mean_return * 100, s=200, alpha=0.7)
    plt.annotate(name, (risk * 100, mean_return * 100),
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Ruin Probability (%)')
plt.ylabel('Mean Growth Rate (%)')
plt.title('Risk-Return Trade-off')
plt.grid(True, alpha=0.3)

# Add efficient frontier concept
plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
plt.axvline(x=1, color='red', linestyle='--', alpha=0.3, label='1% Risk Target')
plt.legend()
plt.show()
```

## Advanced Simulation Features

### Using Different Random Processes

```python
from ergodic_insurance.src.stochastic_processes import GeometricBrownianMotion

# Add market volatility to assets
gbm = GeometricBrownianMotion(
    drift=0.05,      # 5% drift
    volatility=0.15  # 15% volatility
)

# Run simulation with stochastic assets
# (This would require extending the base Simulation class)
```

### Parallel Processing

```python
from ergodic_insurance.src.parallel_executor import ParallelExecutor

# Use parallel processing for large simulations
executor = ParallelExecutor(n_workers=4)

# Run parallel Monte Carlo
results = executor.run_parallel_monte_carlo(
    manufacturer=manufacturer,
    claim_generator=standard_losses,
    n_simulations=10000,
    n_years=50,
    retention=1_000_000,
    limit=10_000_000,
    premium_rate=0.02
)

print(f"Parallel execution completed: {len(results)} simulations")
```

## Best Practices

### 1. Choose Appropriate Time Horizons
- Short-term (1-5 years): Focus on survival
- Medium-term (5-20 years): Balance growth and safety
- Long-term (20+ years): Ergodic effects dominate

### 2. Set Realistic Parameters
```python
# Industry benchmarks
benchmarks = {
    "Manufacturing": {"turnover": 0.8, "margin": 0.06},
    "Technology": {"turnover": 1.5, "margin": 0.15},
    "Retail": {"turnover": 3.0, "margin": 0.03},
    "Financial": {"turnover": 0.1, "margin": 0.20}
}
```

### 3. Validate Results
```python
# Sanity checks
def validate_simulation(result, manufacturer):
    """Validate simulation results for reasonableness."""
    checks = {
        "Positive initial wealth": result.wealth_trajectory[0] > 0,
        "Reasonable growth": -0.5 < result.growth_rate < 0.5,
        "Trajectory length matches": len(result.wealth_trajectory) == len(result.periods),
        "Final wealth matches": abs(result.final_wealth - result.wealth_trajectory[-1]) < 1
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    return all(checks.values())
```

## Troubleshooting Common Issues

### Issue: Simulation runs too slowly
**Solution**: Reduce number of simulations or use parallel processing

```python
# Quick test with fewer simulations
quick_test = mc_analyzer.run_simulations(
    n_simulations=100,  # Reduced from 1000
    n_years=10,         # Reduced from 20
    # ... other parameters
)
```

### Issue: All paths show bankruptcy
**Solution**: Check if losses are too severe relative to assets

```python
# Diagnostic: Check loss severity
annual_losses = [standard_losses.generate_claims(1) for _ in range(100)]
avg_annual_loss = np.mean([sum(losses) for losses in annual_losses])
loss_ratio = avg_annual_loss / manufacturer.initial_assets

if loss_ratio > 0.1:
    print(f"⚠️ High loss ratio: {loss_ratio:.1%} of assets")
    print("Consider: Higher retention, more coverage, or lower loss severity")
```

### Issue: Results vary wildly between runs
**Solution**: Use seeds and increase simulation count

```python
# Ensure reproducibility
np.random.seed(42)
# Increase simulations for stability
stable_results = mc_analyzer.run_simulations(
    n_simulations=5000,  # More simulations
    seed=42             # Fixed seed
)
```

## Next Steps

Now that you understand basic simulations:

1. **[Configuring Insurance](03_configuring_insurance.md)**: Learn about multi-layer programs
2. **[Optimization Workflow](04_optimization_workflow.md)**: Find optimal insurance parameters
3. **[Analyzing Results](05_analyzing_results.md)**: Deep dive into metrics and decisions

## Summary

You've learned how to:
- ✅ Configure manufacturer models for different industries
- ✅ Generate various loss distributions
- ✅ Run single-path and Monte Carlo simulations
- ✅ Compare insurance strategies
- ✅ Visualize and interpret results
- ✅ Apply best practices and troubleshoot issues

You're now ready to explore more complex insurance structures and optimization!

# Analyzing Results

This tutorial covers how to interpret simulation results through the lens of ergodic economics, comparing time-average and ensemble-average metrics.

## Ergodic vs Non-Ergodic Systems

In an **ergodic system**, time averages equal ensemble averages. The average experience of one entity over time equals the average across many entities at a point in time.

Business growth with large losses is **non-ergodic**:
- Ensemble average: "Expected" growth across parallel universes
- Time average: What actually happens to one business over time

This distinction is crucial for insurance decisions.

## Using the Ergodic Analyzer

```python
from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
import numpy as np

# Create analyzer
analyzer = ErgodicAnalyzer(convergence_threshold=0.01)

# Example equity trajectories (multiple simulation paths)
# Each row is a different simulation, columns are years
insured_trajectories = [
    np.array([10e6, 10.5e6, 11.0e6, 11.6e6, 12.2e6]),  # Steady growth
    np.array([10e6, 10.3e6, 10.8e6, 11.2e6, 11.8e6]),  # Steady growth
    np.array([10e6, 10.2e6, 10.6e6, 11.0e6, 11.5e6]),  # Steady growth
    np.array([10e6, 10.4e6, 10.9e6, 11.4e6, 12.0e6]),  # Steady growth
]

uninsured_trajectories = [
    np.array([10e6, 11.0e6, 8.0e6, 10.0e6, 12.0e6]),   # Volatile recovery
    np.array([10e6, 10.5e6, 5.0e6, 0, 0]),             # Bankruptcy
    np.array([10e6, 10.8e6, 11.5e6, 12.5e6, 14.0e6]), # Lucky growth
    np.array([10e6, 9.5e6, 7.0e6, 4.0e6, 0]),         # Decline to ruin
]

# Compare scenarios
comparison = analyzer.compare_scenarios(
    scenario_a=insured_trajectories,
    scenario_b=uninsured_trajectories,
    labels=("Insured", "Uninsured"),
    metric="equity"
)
```

## Understanding the Comparison Output

```python
# Time-average growth rates
print("=== Time-Average Growth ===")
print(f"Insured:   {comparison['insured']['time_average_mean']:.2%}")
print(f"Uninsured: {comparison['uninsured']['time_average_mean']:.2%}")

# Ensemble-average growth rates
print("\n=== Ensemble-Average Growth ===")
print(f"Insured:   {comparison['insured']['ensemble_average']:.2%}")
print(f"Uninsured: {comparison['uninsured']['ensemble_average']:.2%}")

# Survival rates
print("\n=== Survival Rates ===")
print(f"Insured:   {comparison['insured']['survival_rate']:.1%}")
print(f"Uninsured: {comparison['uninsured']['survival_rate']:.1%}")

# Ergodic advantage
print("\n=== Ergodic Advantage ===")
print(f"Time-average gain: {comparison['ergodic_advantage']['time_average_gain']:.2%}")
print(f"Survival gain: {comparison['ergodic_advantage']['survival_gain']:.1%}")
```

## Time-Average vs Ensemble-Average

The key insight: these can diverge significantly!

```python
# Calculate for a single scenario
def calculate_averages(trajectories):
    """Calculate both time and ensemble averages."""

    # Time average: geometric mean of growth rates for each path
    time_averages = []
    for traj in trajectories:
        if traj[-1] > 0:  # Survived
            growth_rate = (traj[-1] / traj[0]) ** (1 / (len(traj) - 1)) - 1
            time_averages.append(growth_rate)
        else:
            time_averages.append(-1.0)  # Ruin

    # Ensemble average: mean wealth at each time, then growth rate
    ensemble = np.mean(trajectories, axis=0)
    ensemble_growth = (ensemble[-1] / ensemble[0]) ** (1 / (len(ensemble) - 1)) - 1

    return {
        'time_average': np.mean(time_averages),
        'time_average_median': np.median(time_averages),
        'ensemble_average': ensemble_growth,
        'divergence': np.mean(time_averages) - ensemble_growth
    }

# Demonstrate divergence
uninsured_metrics = calculate_averages(uninsured_trajectories)
print(f"Uninsured Time Average: {uninsured_metrics['time_average']:.2%}")
print(f"Uninsured Ensemble Average: {uninsured_metrics['ensemble_average']:.2%}")
print(f"Divergence: {uninsured_metrics['divergence']:.2%}")

# Insurance reduces divergence
insured_metrics = calculate_averages(insured_trajectories)
print(f"\nInsured Time Average: {insured_metrics['time_average']:.2%}")
print(f"Insured Ensemble Average: {insured_metrics['ensemble_average']:.2%}")
print(f"Divergence: {insured_metrics['divergence']:.2%}")
```

## Ergodic Divergence Analysis

```python
# Detailed divergence analysis
divergence = analyzer.calculate_ergodic_divergence(
    trajectories=uninsured_trajectories,
    window_sizes=[5, 10, 20]  # Different time horizons
)

print("=== Ergodic Divergence by Time Horizon ===")
for window, div in divergence.items():
    print(f"{window}-year window: {div:.2%}")

# Divergence typically grows with time horizon for non-ergodic systems
```

## Survival Analysis

```python
# Detailed survival analysis
survival = analyzer.survival_analysis(
    scenario_a=insured_trajectories,
    scenario_b=uninsured_trajectories
)

print("=== Survival Analysis ===")
print(f"Insured survival rate: {survival['insured']['rate']:.1%}")
print(f"Insured mean survival time: {survival['insured']['mean_time']:.1f} years")

print(f"\nUninsured survival rate: {survival['uninsured']['rate']:.1%}")
print(f"Uninsured mean survival time: {survival['uninsured']['mean_time']:.1f} years")

# Kaplan-Meier survival curve data
print("\n=== Survival Curve Data ===")
for year, prob in survival['insured']['curve'].items():
    print(f"Year {year}: {prob:.1%} surviving")
```

## Growth Rate Analysis

```python
# Comprehensive growth analysis
growth_analysis = analyzer.growth_rate_analysis(insured_trajectories)

print("=== Growth Rate Analysis ===")
print(f"Arithmetic mean: {growth_analysis['arithmetic_mean']:.2%}")
print(f"Geometric mean (time-average): {growth_analysis['geometric_mean']:.2%}")
print(f"Volatility (std dev): {growth_analysis['volatility']:.2%}")
print(f"Sharpe ratio: {growth_analysis['sharpe_ratio']:.2f}")
print(f"Sortino ratio: {growth_analysis['sortino_ratio']:.2f}")

# Percentile distribution
print("\nGrowth Rate Distribution:")
for pct, value in growth_analysis['percentiles'].items():
    print(f"  {pct}th percentile: {value:.2%}")
```

## Visualizing Ergodic Effects

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Trajectory comparison
ax1 = axes[0, 0]
for traj in insured_trajectories:
    ax1.plot(range(len(traj)), traj / 1e6, color='blue', alpha=0.5, linewidth=1)
for traj in uninsured_trajectories:
    ax1.plot(range(len(traj)), traj / 1e6, color='red', alpha=0.5, linewidth=1)
ax1.plot([], [], color='blue', label='Insured')
ax1.plot([], [], color='red', label='Uninsured')
ax1.set_xlabel('Year')
ax1.set_ylabel('Equity ($M)')
ax1.set_title('Individual Trajectories')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Ensemble average vs time average
ax2 = axes[0, 1]
years = range(len(insured_trajectories[0]))
insured_ensemble = np.mean(insured_trajectories, axis=0)
uninsured_ensemble = np.mean(uninsured_trajectories, axis=0)
ax2.plot(years, insured_ensemble / 1e6, color='blue', linewidth=2, label='Insured (ensemble)')
ax2.plot(years, uninsured_ensemble / 1e6, color='red', linewidth=2, label='Uninsured (ensemble)')
ax2.set_xlabel('Year')
ax2.set_ylabel('Mean Equity ($M)')
ax2.set_title('Ensemble Average (Misleading!)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Time-average growth comparison
ax3 = axes[1, 0]
insured_ta = [insured_metrics['time_average'], insured_metrics['time_average_median']]
uninsured_ta = [uninsured_metrics['time_average'], uninsured_metrics['time_average']]
x = ['Mean', 'Median']
width = 0.35
ax3.bar([i - width/2 for i in range(2)], [v * 100 for v in insured_ta], width, label='Insured', color='blue')
ax3.bar([i + width/2 for i in range(2)], [v * 100 for v in uninsured_ta], width, label='Uninsured', color='red')
ax3.set_ylabel('Time-Average Growth (%)')
ax3.set_title('Time-Average Growth Rates')
ax3.set_xticks(range(2))
ax3.set_xticklabels(x)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Divergence illustration
ax4 = axes[1, 1]
scenarios = ['Insured', 'Uninsured']
time_avg = [insured_metrics['time_average'] * 100, uninsured_metrics['time_average'] * 100]
ensemble_avg = [insured_metrics['ensemble_average'] * 100, uninsured_metrics['ensemble_average'] * 100]
x = np.arange(len(scenarios))
ax4.bar(x - width/2, time_avg, width, label='Time Average', color='green')
ax4.bar(x + width/2, ensemble_avg, width, label='Ensemble Average', color='orange')
ax4.set_ylabel('Growth Rate (%)')
ax4.set_title('Time vs Ensemble Average')
ax4.set_xticks(x)
ax4.set_xticklabels(scenarios)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ergodic_analysis.png', dpi=150)
plt.show()
```

## Convergence Analysis

For Monte Carlo results, check if you have enough simulations:

```python
# Check convergence of time-average estimate
convergence = analyzer.check_convergence(
    trajectories=insured_trajectories,
    metric='time_average_growth'
)

print("=== Convergence Analysis ===")
print(f"Current estimate: {convergence['estimate']:.2%}")
print(f"Standard error: {convergence['standard_error']:.4%}")
print(f"95% CI: [{convergence['ci_lower']:.2%}, {convergence['ci_upper']:.2%}]")
print(f"Converged: {'Yes' if convergence['converged'] else 'No (need more simulations)'}")
```

## Full Ergodic Analysis Pipeline

```python
# Complete analysis from simulation results
from ergodic_insurance import (
    ManufacturerConfig, Simulation, ErgodicAnalyzer
)
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.insurance_program import InsuranceProgram, EnhancedInsuranceLayer

# Configuration
config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0
)

# Run many simulations for both scenarios
n_sims = 100
time_horizon = 30

insured_results = []
uninsured_results = []

for seed in range(n_sims):
    # Insured simulation
    mfg = WidgetManufacturer(config)
    claims = ClaimGenerator(base_frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=seed)
    insurance = InsuranceProgram(deductible=100_000)
    insurance.add_layer(EnhancedInsuranceLayer(
        attachment_point=100_000, limit=5_000_000, base_premium_rate=0.02
    ))
    sim = Simulation(manufacturer=mfg, claim_generator=claims,
                    insurance_program=insurance, time_horizon=time_horizon)
    insured_results.append(sim.run())

    # Uninsured simulation
    mfg = WidgetManufacturer(config)
    claims = ClaimGenerator(base_frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=seed)
    sim = Simulation(manufacturer=mfg, claim_generator=claims, time_horizon=time_horizon)
    uninsured_results.append(sim.run())

# Extract trajectories
insured_trajectories = [r.equity for r in insured_results]
uninsured_trajectories = [r.equity for r in uninsured_results]

# Full ergodic analysis
analyzer = ErgodicAnalyzer()
full_comparison = analyzer.compare_scenarios(
    insured_trajectories, uninsured_trajectories,
    labels=("Insured", "Uninsured")
)

# Report
print("=" * 60)
print("FULL ERGODIC ANALYSIS REPORT")
print("=" * 60)
print(f"\nSimulations: {n_sims}")
print(f"Time horizon: {time_horizon} years")
print(f"\n{'Metric':<30} {'Insured':>12} {'Uninsured':>12}")
print("-" * 56)
print(f"{'Survival Rate':<30} {full_comparison['insured']['survival_rate']:>11.1%} {full_comparison['uninsured']['survival_rate']:>11.1%}")
print(f"{'Time-Average Growth':<30} {full_comparison['insured']['time_average_mean']:>11.2%} {full_comparison['uninsured']['time_average_mean']:>11.2%}")
print(f"{'Ensemble-Average Growth':<30} {full_comparison['insured']['ensemble_average']:>11.2%} {full_comparison['uninsured']['ensemble_average']:>11.2%}")
print(f"{'Ergodic Divergence':<30} {full_comparison['insured']['divergence']:>11.2%} {full_comparison['uninsured']['divergence']:>11.2%}")
print("-" * 56)
print(f"\nErgodic Advantage (Insured vs Uninsured):")
print(f"  Time-average gain: {full_comparison['ergodic_advantage']['time_average_gain']:.2%}")
print(f"  Survival improvement: {full_comparison['ergodic_advantage']['survival_gain']:.1%}")
```

## Interpreting Results

| Metric | What It Means |
|--------|---------------|
| **Time-average growth** | True long-term compound growth rate |
| **Ensemble-average growth** | Expected value (often misleading) |
| **Ergodic divergence** | How much ensemble overestimates time average |
| **Survival rate** | Probability of avoiding bankruptcy |
| **Sharpe ratio** | Risk-adjusted return (time basis) |

**Key takeaway**: If ergodic divergence is large and negative, the ensemble average is significantly overestimating what a single business will actually experience over time.

## Next Steps

- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) - Monte Carlo, market cycles, and complex configurations

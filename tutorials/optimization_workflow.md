---
layout: default
title: Optimization Workflow Tutorial - Ergodic Insurance Framework
description: Learn how to find optimal insurance structures using ergodic theory
mathjax: true
---

# Optimization Workflow Tutorial

## Overview

This tutorial teaches you how to find the optimal insurance structure for your business using ergodic optimization techniques. You'll learn to balance growth, risk, and cost to maximize long-term wealth accumulation.

## The Optimization Problem

### What We're Optimizing

The goal is to find the insurance structure that maximizes the time-average growth rate:

$$g^* = \max_{I} \lim_{T \to \infty} \frac{1}{T} \ln\left(\frac{W_T^I}{W_0}\right)$$

Where:
- $g^*$ is the optimal growth rate
- $I$ represents the insurance structure
- $W_T^I$ is wealth at time $T$ with insurance $I$

### Constraints to Consider

1. **Budget Constraint**: Total premium ≤ maximum affordable
2. **Risk Constraint**: Probability of ruin ≤ acceptable threshold
3. **Regulatory Constraint**: Minimum coverage requirements
4. **Practical Constraint**: Available insurance products

## Step-by-Step Optimization Process

### Step 1: Define the Search Space

```python
import numpy as np
from ergodic_insurance.src import (
    Manufacturer,
    InsuranceOptimizer,
    OptimizationConstraints,
    GridSearchOptimizer,
    GradientOptimizer
)

# Define the business
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    base_revenue=15_000_000,
    operating_margin=0.08,
    volatility=0.15
)

# Define the search space for insurance parameters
search_space = {
    'primary_limit': np.linspace(1_000_000, 10_000_000, 10),
    'primary_attachment': [0, 100_000, 250_000, 500_000],
    'primary_rate': np.linspace(0.01, 0.03, 10),
    'excess_limit': np.linspace(5_000_000, 50_000_000, 10),
    'excess_rate': np.linspace(0.005, 0.015, 10)
}

print(f"Total combinations to test: {np.prod([len(v) for v in search_space.values()]):,}")
```

### Step 2: Set Optimization Constraints

```python
# Define constraints
constraints = OptimizationConstraints(
    max_total_premium=750_000,  # Maximum 5% of revenue
    min_total_coverage=5_000_000,  # Minimum $5M coverage
    max_ruin_probability=0.01,  # Maximum 1% ruin probability over 20 years
    min_growth_rate=0.05,  # Minimum 5% annual growth
    time_horizon=20
)

# Validate constraints are feasible
if constraints.validate():
    print("Constraints are feasible")
else:
    print("Warning: Constraints may be too restrictive")
```

### Step 3: Grid Search Optimization

```python
# Initialize grid search optimizer
grid_optimizer = GridSearchOptimizer(
    manufacturer=manufacturer,
    constraints=constraints,
    search_space=search_space,
    n_simulations_per_point=1000  # Simulations for each configuration
)

# Run grid search (this may take time)
grid_results = grid_optimizer.optimize(
    show_progress=True,
    parallel=True,
    n_workers=4
)

# Extract best configuration
best_config = grid_results.best_configuration
print(f"\nBest Configuration Found:")
print(f"  Primary: ${best_config['primary_limit']:,.0f} xs ${best_config['primary_attachment']:,.0f}")
print(f"  Excess: ${best_config['excess_limit']:,.0f} xs ${best_config['primary_limit']:,.0f}")
print(f"  Total Premium: ${best_config['total_premium']:,.0f}")
print(f"  Expected Growth: {best_config['growth_rate']:.2%}")
print(f"  Ruin Probability: {best_config['ruin_prob']:.2%}")
```

### Step 4: Gradient-Based Refinement

```python
# Use gradient optimization for refinement
gradient_optimizer = GradientOptimizer(
    manufacturer=manufacturer,
    constraints=constraints,
    initial_point=best_config,  # Start from grid search result
    learning_rate=0.01,
    tolerance=0.001
)

# Run gradient optimization
refined_results = gradient_optimizer.optimize(
    max_iterations=100,
    n_simulations_per_gradient=5000
)

# Compare results
print(f"\nImprovement from gradient optimization:")
print(f"  Growth rate: {best_config['growth_rate']:.3%} → {refined_results['growth_rate']:.3%}")
print(f"  Premium: ${best_config['total_premium']:,.0f} → ${refined_results['total_premium']:,.0f}")
```

## Advanced Optimization Techniques

### Multi-Objective Optimization

```python
from ergodic_insurance.src import ParetoOptimizer

# Define multiple objectives
objectives = {
    'growth_rate': 'maximize',
    'premium_cost': 'minimize',
    'ruin_probability': 'minimize',
    'volatility': 'minimize'
}

# Initialize Pareto optimizer
pareto_optimizer = ParetoOptimizer(
    manufacturer=manufacturer,
    objectives=objectives,
    constraints=constraints
)

# Find Pareto frontier
pareto_frontier = pareto_optimizer.find_frontier(
    n_points=50,
    n_simulations=1000
)

# Visualize Pareto frontier
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Growth vs Premium
axes[0].scatter(
    pareto_frontier['premium_cost'],
    pareto_frontier['growth_rate'],
    c=pareto_frontier['ruin_probability'],
    cmap='RdYlGn_r'
)
axes[0].set_xlabel('Annual Premium ($)')
axes[0].set_ylabel('Growth Rate (%)')
axes[0].set_title('Growth vs Cost Trade-off')

# Growth vs Risk
axes[1].scatter(
    pareto_frontier['ruin_probability'],
    pareto_frontier['growth_rate'],
    c=pareto_frontier['premium_cost'],
    cmap='viridis'
)
axes[1].set_xlabel('Ruin Probability (%)')
axes[1].set_ylabel('Growth Rate (%)')
axes[1].set_title('Growth vs Risk Trade-off')

# 3D view
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(133, projection='3d')
ax.scatter(
    pareto_frontier['premium_cost'],
    pareto_frontier['ruin_probability'],
    pareto_frontier['growth_rate']
)
ax.set_xlabel('Premium ($)')
ax.set_ylabel('Ruin Prob (%)')
ax.set_zlabel('Growth (%)')
ax.set_title('3D Pareto Frontier')

plt.tight_layout()
plt.show()
```

### Bayesian Optimization

```python
from ergodic_insurance.src import BayesianOptimizer

# Initialize Bayesian optimizer
bayes_optimizer = BayesianOptimizer(
    manufacturer=manufacturer,
    constraints=constraints,
    acquisition_function='expected_improvement',
    kernel='matern'
)

# Run Bayesian optimization
bayes_results = bayes_optimizer.optimize(
    n_initial_points=20,  # Random exploration
    n_iterations=50,  # Bayesian iterations
    n_simulations_per_point=2000
)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(bayes_results.convergence_history)
plt.xlabel('Iteration')
plt.ylabel('Best Growth Rate Found')
plt.title('Bayesian Optimization Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

### Genetic Algorithm Optimization

```python
from ergodic_insurance.src import GeneticOptimizer

# Configure genetic algorithm
genetic_optimizer = GeneticOptimizer(
    manufacturer=manufacturer,
    constraints=constraints,
    population_size=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elite_size=10
)

# Run genetic optimization
genetic_results = genetic_optimizer.evolve(
    n_generations=50,
    n_simulations_per_individual=500,
    show_progress=True
)

# Visualize evolution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Fitness over generations
axes[0].plot(genetic_results.best_fitness_history, label='Best')
axes[0].plot(genetic_results.mean_fitness_history, label='Population Mean')
axes[0].set_xlabel('Generation')
axes[0].set_ylabel('Fitness (Growth Rate)')
axes[0].set_title('Genetic Algorithm Evolution')
axes[0].legend()

# Final population distribution
axes[1].hist(genetic_results.final_population_fitness, bins=20)
axes[1].set_xlabel('Growth Rate')
axes[1].set_ylabel('Count')
axes[1].set_title('Final Population Distribution')

plt.tight_layout()
plt.show()
```

## Optimization for Different Business Types

### High-Growth Startup

```python
# High growth, high volatility
startup = Manufacturer(
    starting_assets=2_000_000,
    base_revenue=3_000_000,
    operating_margin=0.15,
    growth_rate=0.50,  # 50% growth
    volatility=0.40  # High uncertainty
)

# Optimize with growth focus
startup_optimizer = OptimizationWorkflow(
    manufacturer=startup,
    optimization_focus='aggressive_growth',
    risk_tolerance='high'
)

startup_insurance = startup_optimizer.run()
print(f"Startup optimal: {startup_insurance.summary()}")
```

### Stable Manufacturer

```python
# Stable, predictable business
stable_co = Manufacturer(
    starting_assets=50_000_000,
    base_revenue=75_000_000,
    operating_margin=0.06,
    growth_rate=0.03,
    volatility=0.10
)

# Optimize with stability focus
stable_optimizer = OptimizationWorkflow(
    manufacturer=stable_co,
    optimization_focus='capital_preservation',
    risk_tolerance='low'
)

stable_insurance = stable_optimizer.run()
print(f"Stable company optimal: {stable_insurance.summary()}")
```

### Cyclical Business

```python
# Cyclical with time-varying parameters
cyclical = Manufacturer(
    starting_assets=20_000_000,
    base_revenue=30_000_000,
    operating_margin=0.10,
    volatility=0.25,
    cyclical=True,
    cycle_period=7  # 7-year cycles
)

# Optimize considering cycles
cyclical_optimizer = OptimizationWorkflow(
    manufacturer=cyclical,
    optimization_focus='cycle_smoothing',
    time_horizon=21  # 3 full cycles
)

cyclical_insurance = cyclical_optimizer.run()
```

## Sensitivity Analysis of Optimal Solutions

### Parameter Sensitivity

```python
def sensitivity_analysis(base_manufacturer, optimal_insurance):
    """Test sensitivity of optimal solution to parameter changes"""

    parameters = ['volatility', 'growth_rate', 'operating_margin']
    variations = [-20, -10, 0, 10, 20]  # Percentage changes

    results = {}

    for param in parameters:
        param_results = []

        for variation in variations:
            # Create modified manufacturer
            test_manufacturer = base_manufacturer.copy()
            current_value = getattr(test_manufacturer, param)
            setattr(test_manufacturer, param, current_value * (1 + variation/100))

            # Test with optimal insurance
            engine = MonteCarloEngine(
                manufacturer=test_manufacturer,
                insurance_program=optimal_insurance,
                n_simulations=1000,
                time_horizon=20
            )

            results = engine.run()
            growth_rate = results.calculate_ergodic_growth()

            param_results.append({
                'variation': variation,
                'growth_rate': growth_rate
            })

        results[param] = param_results

    return results

# Run sensitivity analysis
sensitivity = sensitivity_analysis(manufacturer, refined_results['insurance'])

# Plot tornado diagram
plot_tornado_diagram(sensitivity)
```

### Robustness Testing

```python
def test_robustness(optimal_insurance, n_scenarios=100):
    """Test optimal solution across different scenarios"""

    scenarios = []

    for i in range(n_scenarios):
        # Generate random business parameters
        test_manufacturer = Manufacturer(
            starting_assets=np.random.uniform(5e6, 20e6),
            volatility=np.random.uniform(0.10, 0.30),
            operating_margin=np.random.uniform(0.04, 0.12),
            growth_rate=np.random.uniform(0.0, 0.15)
        )

        # Test optimal insurance
        engine = MonteCarloEngine(
            manufacturer=test_manufacturer,
            insurance_program=optimal_insurance,
            n_simulations=500,
            time_horizon=20
        )

        results = engine.run()

        scenarios.append({
            'volatility': test_manufacturer.volatility,
            'margin': test_manufacturer.operating_margin,
            'growth': results.calculate_ergodic_growth(),
            'ruin_prob': results.calculate_ruin_probability()
        })

    return pd.DataFrame(scenarios)

# Test robustness
robustness_df = test_robustness(refined_results['insurance'])

print(f"Robustness Statistics:")
print(f"  Mean growth: {robustness_df['growth'].mean():.2%}")
print(f"  Std growth: {robustness_df['growth'].std():.2%}")
print(f"  Failed scenarios: {(robustness_df['ruin_prob'] > 0.05).sum()}")
```

## Implementing Optimal Solutions

### Generate Implementation Report

```python
def generate_implementation_report(optimal_solution):
    """Create detailed implementation report"""

    report = f"""
    OPTIMAL INSURANCE PROGRAM IMPLEMENTATION REPORT
    ================================================

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    RECOMMENDED STRUCTURE
    ---------------------
    """

    total_premium = 0
    for i, layer in enumerate(optimal_solution['layers']):
        report += f"""
    Layer {i+1}: {layer['name']}
      Limit: ${layer['limit']:,.0f}
      Attachment: ${layer['attachment']:,.0f}
      Premium Rate: {layer['rate']:.3%}
      Annual Premium: ${layer['premium']:,.0f}
        """
        total_premium += layer['premium']

    report += f"""

    TOTAL ANNUAL PREMIUM: ${total_premium:,.0f}

    EXPECTED OUTCOMES
    -----------------
    Time-Average Growth Rate: {optimal_solution['growth_rate']:.2%}
    Ensemble-Average Growth: {optimal_solution['ensemble_growth']:.2%}
    20-Year Ruin Probability: {optimal_solution['ruin_prob']:.2%}
    Expected 20-Year Wealth: ${optimal_solution['expected_wealth']:,.0f}
    95% Confidence Interval: [${optimal_solution['ci_lower']:,.0f}, ${optimal_solution['ci_upper']:,.0f}]

    RISK METRICS
    ------------
    Annual VaR (95%): ${optimal_solution['var_95']:,.0f}
    Annual CVaR (95%): ${optimal_solution['cvar_95']:,.0f}
    Maximum Drawdown: {optimal_solution['max_drawdown']:.1%}
    Recovery Time: {optimal_solution['recovery_time']:.1f} years

    IMPLEMENTATION STEPS
    --------------------
    1. Contact insurance broker with specifications
    2. Request quotes from minimum 3 carriers
    3. Negotiate rates based on model assumptions
    4. Review policy terms for exclusions
    5. Implement claims management procedures
    6. Schedule quarterly reviews

    MONITORING METRICS
    ------------------
    - Monthly loss tracking vs. model predictions
    - Quarterly premium/limit review
    - Annual full reoptimization
    - Trigger: Reoptimize if volatility changes >20%
    """

    return report

# Generate report
implementation_report = generate_implementation_report(refined_results)
print(implementation_report)

# Save to file
with open('optimal_insurance_implementation.txt', 'w') as f:
    f.write(implementation_report)
```

## Common Optimization Patterns

### Pattern 1: The Convexity Effect
Optimal premium often exceeds expected losses by 2-3x due to volatility reduction benefits.

### Pattern 2: Layer Efficiency
Multiple layers typically outperform single large policies:
- Primary layer: High frequency claims
- Excess layers: Catastrophic protection

### Pattern 3: Attachment Point Sweet Spot
Optimal attachment often equals 1-2 months of operating cash flow.

### Pattern 4: Growth-Risk Frontier
Higher growth businesses benefit more from comprehensive coverage.

## Exercises

### Exercise 1: Find Your Optimal Structure
Use your own business parameters to find the optimal insurance configuration.

### Exercise 2: Compare Optimization Methods
Compare grid search, Bayesian, and genetic algorithms for the same problem.

### Exercise 3: Multi-Year Optimization
Optimize a changing insurance program over a 5-year planning horizon.

## Next Steps

- [Analyze Your Results](/Ergodic-Insurance-Limits/tutorials/analyzing_results)
- [Advanced Scenarios](/Ergodic-Insurance-Limits/tutorials/advanced_scenarios)
- [Case Studies](/Ergodic-Insurance-Limits/docs/user_guide/case_studies)

---

[← Back to Basic Simulation](/Ergodic-Insurance-Limits/tutorials/basic_simulation) | [Continue to Analyzing Results →](/Ergodic-Insurance-Limits/tutorials/analyzing_results)

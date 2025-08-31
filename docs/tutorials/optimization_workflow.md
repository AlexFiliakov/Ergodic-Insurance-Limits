---
layout: default
title: Optimization Workflow
---

# Optimization Workflow

Learn how to automatically find optimal insurance parameters using the framework's optimization tools.

## Overview

This tutorial covers:
- Setting up optimization problems
- Choosing objective functions
- Running optimizations
- Interpreting results
- Advanced optimization techniques

## Setting Up an Optimization

### Basic Optimization

```python
from ergodic_insurance.src.optimization import InsuranceOptimizer
from ergodic_insurance.src.manufacturer import Manufacturer

# Define the company
manufacturer = Manufacturer(
    starting_assets=10_000_000,
    asset_turnover=1.0,
    operating_margin=0.08
)

# Create optimizer
optimizer = InsuranceOptimizer(manufacturer)

# Run optimization
optimal_insurance = optimizer.optimize(
    objective="maximize_growth",
    constraints={"max_premium": 500_000}
)

print(f"Optimal retention: ${optimal_insurance.retention:,.0f}")
print(f"Optimal limit: ${optimal_insurance.limit:,.0f}")
```

### Defining Objectives

```python
# Single objective optimization
objectives = {
    "maximize_growth": lambda r: r.growth_rate,
    "minimize_ruin": lambda r: -r.ruin_probability,
    "maximize_sharpe": lambda r: r.sharpe_ratio,
    "minimize_cost": lambda r: -r.total_premium
}

# Choose your objective
optimal = optimizer.optimize(objective=objectives["maximize_growth"])
```

## Multi-Objective Optimization

### Pareto Frontier

```python
from ergodic_insurance.src.pareto_frontier import ParetoFrontier

# Define multiple objectives
frontier = ParetoFrontier()

# Add objectives
frontier.add_objective("growth", lambda r: r.growth_rate, maximize=True)
frontier.add_objective("risk", lambda r: r.ruin_probability, maximize=False)
frontier.add_objective("cost", lambda r: r.premium_ratio, maximize=False)

# Find Pareto optimal solutions
solutions = frontier.compute(
    parameter_ranges={
        "retention": (100_000, 5_000_000),
        "limit": (1_000_000, 50_000_000)
    },
    n_points=100
)

# Visualize frontier
frontier.plot()
```

### Weighted Objectives

```python
def weighted_objective(results, weights):
    """Combine multiple objectives with weights."""
    score = 0
    score += weights["growth"] * results.growth_rate
    score -= weights["risk"] * results.ruin_probability
    score -= weights["cost"] * results.premium_ratio
    return score

# Optimize with weights
weights = {"growth": 0.5, "risk": 0.3, "cost": 0.2}
optimal = optimizer.optimize(
    objective=lambda r: weighted_objective(r, weights)
)
```

## Constraint Handling

### Hard Constraints

```python
# Define constraints
constraints = {
    "max_premium": 500_000,  # Budget constraint
    "max_ruin_prob": 0.01,   # Risk constraint
    "min_roe": 0.10          # Return constraint
}

# Optimize with constraints
optimal = optimizer.optimize(
    objective="maximize_growth",
    constraints=constraints
)
```

### Soft Constraints (Penalties)

```python
def penalized_objective(results, penalties):
    """Apply penalties for constraint violations."""
    score = results.growth_rate

    # Penalty for exceeding premium budget
    if results.total_premium > 500_000:
        excess = results.total_premium - 500_000
        score -= penalties["premium"] * excess

    # Penalty for high ruin probability
    if results.ruin_probability > 0.01:
        excess = results.ruin_probability - 0.01
        score -= penalties["risk"] * excess

    return score

# Optimize with soft constraints
optimal = optimizer.optimize(
    objective=lambda r: penalized_objective(r, {"premium": 0.001, "risk": 10})
)
```

## Optimization Algorithms

### Grid Search

```python
from ergodic_insurance.src.optimization import GridSearchOptimizer

# Define parameter grid
param_grid = {
    "retention": [250_000, 500_000, 1_000_000, 2_000_000],
    "limit": [5_000_000, 10_000_000, 25_000_000, 50_000_000]
}

# Grid search
grid_optimizer = GridSearchOptimizer(manufacturer)
results = grid_optimizer.search(param_grid)

# Best parameters
best_params = results.best_params
print(f"Best retention: ${best_params['retention']:,.0f}")
print(f"Best limit: ${best_params['limit']:,.0f}")
```

### Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real

def objective(params):
    """Objective function for Bayesian optimization."""
    retention, limit = params

    insurance = Insurance(
        retention=retention,
        limit=limit,
        premium_rate=0.03
    )

    results = run_simulation(manufacturer, insurance)
    return -results.growth_rate  # Minimize negative growth

# Define search space
space = [
    Real(100_000, 5_000_000, name='retention'),
    Real(1_000_000, 50_000_000, name='limit')
]

# Run Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,
    random_state=42
)

print(f"Optimal retention: ${result.x[0]:,.0f}")
print(f"Optimal limit: ${result.x[1]:,.0f}")
```

### Genetic Algorithms

```python
from ergodic_insurance.src.optimization import GeneticOptimizer

# Configure genetic algorithm
ga_optimizer = GeneticOptimizer(
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8
)

# Define gene representation
genes = {
    "retention": {"min": 100_000, "max": 5_000_000, "type": "float"},
    "limit": {"min": 1_000_000, "max": 50_000_000, "type": "float"},
    "layers": {"min": 1, "max": 5, "type": "int"}
}

# Run evolution
best_solution = ga_optimizer.evolve(
    fitness_function=lambda x: simulate_and_score(x),
    genes=genes
)
```

## Sensitivity Analysis

### Parameter Sensitivity

```python
from ergodic_insurance.src.sensitivity import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(manufacturer)

# Analyze sensitivity to key parameters
sensitivity = analyzer.analyze(
    base_params={"retention": 500_000, "limit": 10_000_000},
    vary_params={
        "retention": np.linspace(100_000, 2_000_000, 20),
        "limit": np.linspace(5_000_000, 25_000_000, 20)
    },
    metric="growth_rate"
)

# Plot sensitivity
analyzer.plot_tornado_diagram(sensitivity)
analyzer.plot_heatmap(sensitivity)
```

### Monte Carlo Sensitivity

```python
def monte_carlo_sensitivity(n_samples=1000):
    """Sensitivity analysis via Monte Carlo."""
    results = []

    for _ in range(n_samples):
        # Random parameters from distributions
        params = {
            "retention": np.random.lognormal(13, 0.5),  # ~500K median
            "limit": np.random.lognormal(16, 0.5),      # ~10M median
            "premium_rate": np.random.uniform(0.02, 0.05)
        }

        insurance = Insurance(**params)
        result = run_simulation(manufacturer, insurance)

        results.append({
            **params,
            "growth_rate": result.growth_rate,
            "ruin_prob": result.ruin_probability
        })

    return pd.DataFrame(results)

# Analyze correlations
sensitivity_df = monte_carlo_sensitivity()
correlations = sensitivity_df.corr()["growth_rate"].sort_values()
print("Parameter importance for growth:")
print(correlations)
```

## Optimization Workflow Example

### Complete Workflow

```python
class OptimizationWorkflow:
    """Complete optimization workflow."""

    def __init__(self, manufacturer):
        self.manufacturer = manufacturer
        self.results_history = []

    def run(self):
        """Execute full optimization workflow."""

        # Step 1: Baseline analysis
        print("Step 1: Baseline Analysis")
        baseline = self.analyze_baseline()

        # Step 2: Single-layer optimization
        print("Step 2: Single-Layer Optimization")
        single_layer = self.optimize_single_layer()

        # Step 3: Multi-layer optimization
        print("Step 3: Multi-Layer Optimization")
        multi_layer = self.optimize_multi_layer()

        # Step 4: Sensitivity analysis
        print("Step 4: Sensitivity Analysis")
        sensitivity = self.run_sensitivity()

        # Step 5: Stress testing
        print("Step 5: Stress Testing")
        stress_results = self.stress_test(multi_layer)

        # Step 6: Final recommendation
        print("Step 6: Final Recommendation")
        recommendation = self.make_recommendation()

        return recommendation

    def analyze_baseline(self):
        """Analyze without insurance."""
        return run_simulation(self.manufacturer, insurance=None)

    def optimize_single_layer(self):
        """Find optimal single layer."""
        optimizer = InsuranceOptimizer(self.manufacturer)
        return optimizer.optimize(objective="maximize_growth")

    def optimize_multi_layer(self):
        """Find optimal multi-layer program."""
        optimizer = MultiLayerOptimizer(self.manufacturer)
        return optimizer.optimize(
            n_layers_range=(2, 4),
            total_limit=50_000_000
        )

    def run_sensitivity(self):
        """Perform sensitivity analysis."""
        analyzer = SensitivityAnalyzer(self.manufacturer)
        return analyzer.analyze_all_parameters()

    def stress_test(self, insurance_program):
        """Stress test the optimal program."""
        scenarios = ["baseline", "moderate", "severe", "extreme"]
        results = {}

        for scenario in scenarios:
            results[scenario] = run_stress_scenario(
                self.manufacturer,
                insurance_program,
                scenario
            )

        return results

    def make_recommendation(self):
        """Generate final recommendation."""
        # Analyze all results
        recommendation = {
            "program": self.select_best_program(),
            "expected_benefit": self.calculate_expected_benefit(),
            "risk_metrics": self.compile_risk_metrics(),
            "implementation_notes": self.generate_notes()
        }

        return recommendation
```

## Performance Tips

### Caching Results

```python
from functools import lru_cache
import hashlib
import pickle

class OptimizationCache:
    """Cache optimization results."""

    def __init__(self, cache_dir="./optimization_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, params):
        """Generate cache key from parameters."""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def get_or_compute(self, params, compute_fn):
        """Get from cache or compute."""
        key = self.get_cache_key(params)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        result = compute_fn(params)

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result
```

### Parallel Optimization

```python
from multiprocessing import Pool

def parallel_optimization(param_sets, n_workers=8):
    """Run optimizations in parallel."""

    def optimize_single(params):
        """Optimize for one parameter set."""
        optimizer = InsuranceOptimizer(params["manufacturer"])
        return optimizer.optimize(**params["options"])

    with Pool(n_workers) as pool:
        results = pool.map(optimize_single, param_sets)

    return results
```

## Summary

You've learned to:
- Set up optimization problems
- Use different optimization algorithms
- Handle constraints and multiple objectives
- Perform sensitivity analysis
- Build complete optimization workflows

## Next Steps

- [Analyzing Results](analyzing_results.md) - Interpret optimization outputs
- [Advanced Scenarios](advanced_scenarios.md) - Complex real-world applications

For more examples, see `ergodic_insurance/examples/demo_optimization.py`.

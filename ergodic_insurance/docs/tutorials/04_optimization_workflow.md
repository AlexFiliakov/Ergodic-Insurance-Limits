# Optimization Workflow Tutorial

This tutorial teaches you how to find optimal insurance strategies using the framework's optimization tools. You'll learn to define objectives, set constraints, run optimizations, and interpret results for decision-making.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Define optimization objectives (ROE, growth rate, risk-adjusted returns)
- Set appropriate constraints (ruin probability, budget)
- Use different optimization algorithms
- Interpret Pareto frontiers
- Make data-driven insurance decisions

## Understanding Optimization Goals

### The Fundamental Trade-off

```python
import numpy as np
import matplotlib.pyplot as plt
from ergodic_insurance.src.manufacturer import Manufacturer
from ergodic_insurance.src.claim_generator import ClaimGenerator
from ergodic_insurance.src.optimization import InsuranceOptimizer

# Setup
manufacturer = Manufacturer(
    initial_assets=10_000_000,
    asset_turnover=1.0,
    base_operating_margin=0.08
)

claim_generator = ClaimGenerator(
    frequency=5,
    severity_mu=10.0,
    severity_sigma=1.5
)

# Initialize optimizer
optimizer = InsuranceOptimizer(
    manufacturer=manufacturer,
    claim_generator=claim_generator
)

print("Optimization Goals:")
print("1. Maximize Growth Rate (long-term wealth accumulation)")
print("2. Minimize Ruin Probability (survival)")
print("3. Maximize Risk-Adjusted Returns (Sharpe ratio)")
print("4. Minimize Cost (premium expenses)")
print("\nThe challenge: These goals conflict!")
```

### Defining Objective Functions

```python
# Different objective functions
def growth_rate_objective(retention, limit, premium_rate, simulator):
    """Maximize expected growth rate."""
    results = simulator.run_monte_carlo(
        n_simulations=100,
        n_years=20,
        retention=retention,
        limit=limit,
        premium_rate=premium_rate
    )
    return np.mean(results['growth_rates'])

def survival_objective(retention, limit, premium_rate, simulator):
    """Maximize survival probability."""
    results = simulator.run_monte_carlo(
        n_simulations=100,
        n_years=20,
        retention=retention,
        limit=limit,
        premium_rate=premium_rate
    )
    return results['survival_rate']

def sharpe_ratio_objective(retention, limit, premium_rate, simulator):
    """Maximize risk-adjusted returns (Sharpe ratio)."""
    results = simulator.run_monte_carlo(
        n_simulations=100,
        n_years=20,
        retention=retention,
        limit=limit,
        premium_rate=premium_rate
    )
    growth_rates = results['growth_rates']
    if len(growth_rates) > 1:
        return np.mean(growth_rates) / np.std(growth_rates)
    return 0

# Example: Evaluate different objectives for a strategy
test_retention = 1_000_000
test_limit = 10_000_000
test_premium_rate = 0.02

print("\nObjective Function Values for Test Strategy:")
print(f"  Retention: ${test_retention:,.0f}")
print(f"  Limit: ${test_limit:,.0f}")
print(f"  Premium Rate: {test_premium_rate:.2%}")
print("\n  [Evaluating objectives - this would run simulations]")
```

## Single-Objective Optimization

### Optimizing for Maximum Growth

```python
from scipy.optimize import minimize_scalar, minimize

# Optimize retention for maximum growth
def optimize_retention_for_growth(limit=10_000_000, premium_rate=0.02):
    """Find optimal retention for maximum growth rate."""

    def objective(retention):
        # Negative because we minimize (scipy minimizes by default)
        result = optimizer.evaluate_strategy(
            retention=retention,
            limit=limit,
            premium_rate=premium_rate,
            n_simulations=100,
            n_years=20
        )
        return -result['mean_growth_rate']

    # Optimization bounds
    bounds = (100_000, 5_000_000)

    # Run optimization
    result = minimize_scalar(
        objective,
        bounds=bounds,
        method='bounded'
    )

    optimal_retention = result.x
    optimal_growth = -result.fun

    return optimal_retention, optimal_growth

# Find optimal retention
print("Single-Objective Optimization: Maximum Growth")
print("-" * 50)
optimal_ret, optimal_growth = optimize_retention_for_growth()
print(f"Optimal Retention: ${optimal_ret:,.0f}")
print(f"Expected Growth Rate: {optimal_growth:.2%}")
```

### Grid Search Optimization

```python
# Grid search for optimal parameters
def grid_search_optimization():
    """Perform grid search over retention and limit combinations."""

    # Define search grid
    retentions = np.linspace(250_000, 2_000_000, 8)
    limits = np.linspace(5_000_000, 20_000_000, 4)
    premium_rates = [0.015, 0.02, 0.025]

    results = []

    print("Running Grid Search...")
    for retention in retentions:
        for limit in limits:
            for premium_rate in premium_rates:
                # Evaluate strategy
                result = optimizer.evaluate_strategy(
                    retention=retention,
                    limit=limit,
                    premium_rate=premium_rate,
                    n_simulations=50,  # Fewer for speed
                    n_years=10
                )

                results.append({
                    'retention': retention,
                    'limit': limit,
                    'premium_rate': premium_rate,
                    'growth_rate': result['mean_growth_rate'],
                    'survival_rate': result['survival_rate'],
                    'sharpe_ratio': result['sharpe_ratio']
                })

    # Find best strategy
    best_growth = max(results, key=lambda x: x['growth_rate'])
    best_survival = max(results, key=lambda x: x['survival_rate'])
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])

    return results, best_growth, best_survival, best_sharpe

# Run grid search
grid_results, best_g, best_s, best_sr = grid_search_optimization()

print("\nGrid Search Results:")
print("\nBest for Growth Rate:")
print(f"  Retention: ${best_g['retention']:,.0f}")
print(f"  Limit: ${best_g['limit']:,.0f}")
print(f"  Premium: {best_g['premium_rate']:.2%}")
print(f"  Growth: {best_g['growth_rate']:.2%}")

print("\nBest for Survival:")
print(f"  Retention: ${best_s['retention']:,.0f}")
print(f"  Limit: ${best_s['limit']:,.0f}")
print(f"  Premium: {best_s['premium_rate']:.2%}")
print(f"  Survival: {best_s['survival_rate']:.1%}")

print("\nBest for Sharpe Ratio:")
print(f"  Retention: ${best_sr['retention']:,.0f}")
print(f"  Limit: ${best_sr['limit']:,.0f}")
print(f"  Premium: {best_sr['premium_rate']:.2%}")
print(f"  Sharpe: {best_sr['sharpe_ratio']:.2f}")
```

## Multi-Objective Optimization

### Pareto Frontier Analysis

```python
from ergodic_insurance.src.pareto_frontier import ParetoFrontier

# Create Pareto frontier analyzer
pareto = ParetoFrontier(
    manufacturer=manufacturer,
    claim_generator=claim_generator
)

# Generate Pareto frontier
def generate_pareto_frontier():
    """Generate Pareto-optimal strategies."""

    strategies = []

    # Test various strategies
    for retention in np.linspace(100_000, 3_000_000, 15):
        for limit in np.linspace(5_000_000, 20_000_000, 10):
            for premium_rate in [0.01, 0.015, 0.02, 0.025, 0.03]:

                # Evaluate strategy
                result = optimizer.evaluate_strategy(
                    retention=retention,
                    limit=limit,
                    premium_rate=premium_rate,
                    n_simulations=100,
                    n_years=20
                )

                strategies.append({
                    'retention': retention,
                    'limit': limit,
                    'premium_rate': premium_rate,
                    'growth_rate': result['mean_growth_rate'],
                    'ruin_prob': 1 - result['survival_rate'],
                    'cost': limit * premium_rate
                })

    # Find Pareto-optimal strategies
    pareto_strategies = pareto.find_pareto_optimal(
        strategies,
        objectives=['growth_rate', 'ruin_prob'],  # Maximize growth, minimize ruin
        minimize=[False, True]  # Max growth, min ruin
    )

    return pareto_strategies

# Generate and visualize Pareto frontier
pareto_strategies = generate_pareto_frontier()

# Plot Pareto frontier
plt.figure(figsize=(10, 8))
growth_rates = [s['growth_rate'] for s in pareto_strategies]
ruin_probs = [s['ruin_prob'] for s in pareto_strategies]

plt.scatter(ruin_probs, growth_rates, s=100, alpha=0.6, c='blue', edgecolors='black')
plt.plot(ruin_probs, growth_rates, 'b--', alpha=0.3)

# Annotate some key points
for i, strategy in enumerate(pareto_strategies[::3]):  # Every 3rd point
    plt.annotate(
        f"R:${strategy['retention']/1e6:.1f}M\nL:${strategy['limit']/1e6:.0f}M",
        (strategy['ruin_prob'], strategy['growth_rate']),
        xytext=(5, 5), textcoords='offset points', fontsize=8
    )

plt.xlabel('Ruin Probability')
plt.ylabel('Expected Growth Rate')
plt.title('Pareto Frontier: Growth vs Risk')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.5, label='1% Risk Target')
plt.legend()
plt.show()
```

### Three-Dimensional Optimization

```python
# Optimize across three objectives
def three_objective_optimization():
    """Optimize for growth, survival, and cost simultaneously."""

    from mpl_toolkits.mplot3d import Axes3D

    strategies = []

    # Generate strategies
    for retention in np.linspace(250_000, 2_000_000, 10):
        for limit in np.linspace(5_000_000, 15_000_000, 8):
            for premium_rate in [0.01, 0.02, 0.03]:

                result = optimizer.evaluate_strategy(
                    retention=retention,
                    limit=limit,
                    premium_rate=premium_rate,
                    n_simulations=50,
                    n_years=20
                )

                strategies.append({
                    'retention': retention,
                    'limit': limit,
                    'premium_rate': premium_rate,
                    'growth': result['mean_growth_rate'],
                    'survival': result['survival_rate'],
                    'cost': limit * premium_rate / 1e6  # In millions
                })

    # Find Pareto-optimal set
    pareto_3d = pareto.find_pareto_optimal(
        strategies,
        objectives=['growth', 'survival', 'cost'],
        minimize=[False, False, True]  # Max growth, max survival, min cost
    )

    # Visualize 3D Pareto frontier
    fig = plt.figure(figsize=(12, 5))

    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    growth = [s['growth'] for s in pareto_3d]
    survival = [s['survival'] for s in pareto_3d]
    cost = [s['cost'] for s in pareto_3d]

    ax1.scatter(cost, survival, growth, c=growth, cmap='viridis', s=100)
    ax1.set_xlabel('Annual Cost ($M)')
    ax1.set_ylabel('Survival Rate')
    ax1.set_zlabel('Growth Rate')
    ax1.set_title('3D Pareto Frontier')

    # 2D projections
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(survival, growth, c=cost, cmap='coolwarm', s=100)
    ax2.set_xlabel('Survival Rate')
    ax2.set_ylabel('Growth Rate')
    ax2.set_title('Growth vs Survival (colored by cost)')
    plt.colorbar(scatter, label='Cost ($M)')

    plt.tight_layout()
    plt.show()

    return pareto_3d

# Run 3D optimization
pareto_3d_results = three_objective_optimization()

print(f"\nFound {len(pareto_3d_results)} Pareto-optimal strategies")
```

## Constrained Optimization

### With Risk Constraints

```python
from scipy.optimize import minimize

def optimize_with_constraints():
    """Optimize growth subject to risk constraints."""

    # Define objective (negative growth for minimization)
    def objective(params):
        retention, limit = params
        premium_rate = 0.02  # Fixed premium rate

        result = optimizer.evaluate_strategy(
            retention=retention,
            limit=limit,
            premium_rate=premium_rate,
            n_simulations=100,
            n_years=20
        )
        return -result['mean_growth_rate']

    # Define constraint (ruin probability <= 1%)
    def risk_constraint(params):
        retention, limit = params
        premium_rate = 0.02

        result = optimizer.evaluate_strategy(
            retention=retention,
            limit=limit,
            premium_rate=premium_rate,
            n_simulations=100,
            n_years=20
        )
        ruin_prob = 1 - result['survival_rate']
        return 0.01 - ruin_prob  # Must be >= 0

    # Initial guess
    x0 = [1_000_000, 10_000_000]

    # Bounds
    bounds = [(100_000, 3_000_000), (5_000_000, 20_000_000)]

    # Constraint
    constraints = {'type': 'ineq', 'fun': risk_constraint}

    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_retention = result.x[0]
    optimal_limit = result.x[1]
    optimal_growth = -result.fun

    return optimal_retention, optimal_limit, optimal_growth

# Run constrained optimization
print("Constrained Optimization (Risk <= 1%):")
print("-" * 50)
opt_ret, opt_lim, opt_growth = optimize_with_constraints()
print(f"Optimal Retention: ${opt_ret:,.0f}")
print(f"Optimal Limit: ${opt_lim:,.0f}")
print(f"Expected Growth: {opt_growth:.2%}")
```

### With Budget Constraints

```python
def optimize_with_budget(max_premium_budget):
    """Optimize insurance within premium budget."""

    def objective(params):
        retention = params[0]
        limit = params[1]
        # Premium rate determined by market
        premium_rate = 0.015 + 0.01 * np.exp(-retention/1e6)  # Lower retention = higher rate

        result = optimizer.evaluate_strategy(
            retention=retention,
            limit=limit,
            premium_rate=premium_rate,
            n_simulations=50,
            n_years=20
        )
        return -result['mean_growth_rate']

    def budget_constraint(params):
        retention = params[0]
        limit = params[1]
        premium_rate = 0.015 + 0.01 * np.exp(-retention/1e6)
        annual_premium = limit * premium_rate
        return max_premium_budget - annual_premium

    # Optimize
    x0 = [500_000, 5_000_000]
    bounds = [(100_000, 2_000_000), (2_000_000, 15_000_000)]
    constraints = {'type': 'ineq', 'fun': budget_constraint}

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x[0], result.x[1], -result.fun

# Test different budget levels
budgets = [100_000, 200_000, 300_000, 400_000]
budget_results = []

print("\nBudget-Constrained Optimization:")
print("-" * 60)
print(f"{'Budget':<15} {'Retention':<15} {'Limit':<15} {'Growth':<10}")
print("-" * 60)

for budget in budgets:
    ret, lim, growth = optimize_with_budget(budget)
    budget_results.append({'budget': budget, 'retention': ret, 'limit': lim, 'growth': growth})
    print(f"${budget:<14,.0f} ${ret:<14,.0f} ${lim:<14,.0f} {growth:<9.2%}")
```

## Dynamic Optimization

### Time-Varying Strategies

```python
def optimize_dynamic_strategy(time_horizon=20):
    """Optimize insurance strategy that changes over time."""

    # Different strategies for different growth phases
    phases = [
        {"years": [0, 5], "name": "Startup", "risk_tolerance": "low"},
        {"years": [5, 15], "name": "Growth", "risk_tolerance": "medium"},
        {"years": [15, 20], "name": "Mature", "risk_tolerance": "high"}
    ]

    optimal_strategies = []

    for phase in phases:
        # Set parameters based on risk tolerance
        if phase["risk_tolerance"] == "low":
            # Conservative: low retention, high coverage
            retention = 250_000
            limit = 15_000_000
            premium_rate = 0.025
        elif phase["risk_tolerance"] == "medium":
            # Balanced
            retention = 750_000
            limit = 10_000_000
            premium_rate = 0.02
        else:
            # Aggressive: high retention, lower coverage
            retention = 1_500_000
            limit = 5_000_000
            premium_rate = 0.015

        optimal_strategies.append({
            "phase": phase["name"],
            "years": phase["years"],
            "retention": retention,
            "limit": limit,
            "premium_rate": premium_rate
        })

    return optimal_strategies

# Generate dynamic strategy
dynamic_strategy = optimize_dynamic_strategy()

print("\nDynamic Optimization Strategy:")
print("-" * 70)
print(f"{'Phase':<10} {'Years':<10} {'Retention':<15} {'Limit':<15} {'Premium':<10}")
print("-" * 70)

for strategy in dynamic_strategy:
    years = f"{strategy['years'][0]}-{strategy['years'][1]}"
    premium = strategy['limit'] * strategy['premium_rate']
    print(f"{strategy['phase']:<10} {years:<10} ${strategy['retention']:<14,.0f} "
          f"${strategy['limit']:<14,.0f} ${premium:<9,.0f}")

# Visualize dynamic strategy
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Retention over time
ax1 = axes[0]
for strategy in dynamic_strategy:
    ax1.barh(strategy['phase'], strategy['retention']/1e6,
             color='steelblue', alpha=0.7)
ax1.set_xlabel('Retention ($M)')
ax1.set_title('Retention by Phase')

# Limit over time
ax2 = axes[1]
for strategy in dynamic_strategy:
    ax2.barh(strategy['phase'], strategy['limit']/1e6,
             color='green', alpha=0.7)
ax2.set_xlabel('Limit ($M)')
ax2.set_title('Coverage Limit by Phase')

# Premium over time
ax3 = axes[2]
for strategy in dynamic_strategy:
    premium = strategy['limit'] * strategy['premium_rate']
    ax3.barh(strategy['phase'], premium/1e3,
             color='orange', alpha=0.7)
ax3.set_xlabel('Annual Premium ($K)')
ax3.set_title('Premium by Phase')

plt.tight_layout()
plt.show()
```

## Interpreting Optimization Results

### Decision Matrix

```python
def create_decision_matrix(strategies):
    """Create decision matrix for strategy selection."""

    import pandas as pd

    # Score each strategy on multiple criteria
    scores = []

    for strategy in strategies:
        score = {
            'Strategy': f"R:{strategy['retention']/1e6:.1f}M L:{strategy['limit']/1e6:.0f}M",
            'Growth Rate': strategy['growth_rate'],
            'Survival Rate': strategy['survival_rate'],
            'Annual Cost': strategy['limit'] * strategy['premium_rate'],
            'Risk Score': 1 - strategy['survival_rate'],
            'Efficiency': strategy['growth_rate'] / (strategy['limit'] * strategy['premium_rate'] / 1e6)
        }
        scores.append(score)

    df = pd.DataFrame(scores)

    # Normalize scores (0-100 scale)
    for col in ['Growth Rate', 'Survival Rate', 'Efficiency']:
        df[f'{col} Score'] = 100 * (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    for col in ['Annual Cost', 'Risk Score']:
        df[f'{col} Score'] = 100 * (df[col].max() - df[col]) / (df[col].max() - df[col].min())

    # Calculate weighted overall score
    weights = {
        'Growth Rate Score': 0.3,
        'Survival Rate Score': 0.3,
        'Annual Cost Score': 0.2,
        'Efficiency Score': 0.2
    }

    df['Overall Score'] = sum(df[col] * weight for col, weight in weights.items())

    return df.sort_values('Overall Score', ascending=False)

# Create decision matrix for sample strategies
sample_strategies = [
    {'retention': 250_000, 'limit': 15_000_000, 'premium_rate': 0.025,
     'growth_rate': 0.045, 'survival_rate': 0.99},
    {'retention': 500_000, 'limit': 10_000_000, 'premium_rate': 0.02,
     'growth_rate': 0.055, 'survival_rate': 0.97},
    {'retention': 1_000_000, 'limit': 8_000_000, 'premium_rate': 0.018,
     'growth_rate': 0.065, 'survival_rate': 0.95},
    {'retention': 1_500_000, 'limit': 5_000_000, 'premium_rate': 0.015,
     'growth_rate': 0.070, 'survival_rate': 0.92}
]

decision_matrix = create_decision_matrix(sample_strategies)
print("\nDecision Matrix:")
print(decision_matrix[['Strategy', 'Overall Score', 'Growth Rate', 'Survival Rate', 'Annual Cost']])
```

### Sensitivity Analysis

```python
def sensitivity_analysis(base_retention=1_000_000, base_limit=10_000_000):
    """Analyze sensitivity to parameter changes."""

    # Test sensitivity to different parameters
    sensitivity_results = {
        'retention': [],
        'limit': [],
        'premium_rate': [],
        'frequency': [],
        'severity': []
    }

    # Base case
    base_result = optimizer.evaluate_strategy(
        retention=base_retention,
        limit=base_limit,
        premium_rate=0.02,
        n_simulations=100,
        n_years=20
    )
    base_growth = base_result['mean_growth_rate']

    # Vary retention
    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        result = optimizer.evaluate_strategy(
            retention=base_retention * factor,
            limit=base_limit,
            premium_rate=0.02,
            n_simulations=50,
            n_years=20
        )
        sensitivity_results['retention'].append({
            'factor': factor,
            'value': base_retention * factor,
            'growth': result['mean_growth_rate'],
            'change': (result['mean_growth_rate'] - base_growth) / base_growth
        })

    # Vary limit
    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        result = optimizer.evaluate_strategy(
            retention=base_retention,
            limit=base_limit * factor,
            premium_rate=0.02,
            n_simulations=50,
            n_years=20
        )
        sensitivity_results['limit'].append({
            'factor': factor,
            'value': base_limit * factor,
            'growth': result['mean_growth_rate'],
            'change': (result['mean_growth_rate'] - base_growth) / base_growth
        })

    # Visualize sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Retention sensitivity
    ax1 = axes[0]
    factors = [r['factor'] for r in sensitivity_results['retention']]
    changes = [r['change'] * 100 for r in sensitivity_results['retention']]
    ax1.plot(factors, changes, 'o-', markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Retention Factor')
    ax1.set_ylabel('Growth Rate Change (%)')
    ax1.set_title('Sensitivity to Retention')
    ax1.grid(True, alpha=0.3)

    # Limit sensitivity
    ax2 = axes[1]
    factors = [r['factor'] for r in sensitivity_results['limit']]
    changes = [r['change'] * 100 for r in sensitivity_results['limit']]
    ax2.plot(factors, changes, 'o-', markersize=8, color='green')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Limit Factor')
    ax2.set_ylabel('Growth Rate Change (%)')
    ax2.set_title('Sensitivity to Limit')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return sensitivity_results

# Run sensitivity analysis
sensitivity = sensitivity_analysis()

print("\nSensitivity Analysis Summary:")
print("Parameter changes from -50% to +50% of base case")
print("\nMost sensitive to: [would show which parameter has highest impact]")
```

## Best Practices

### 1. Start Simple

```python
# Progressive optimization approach
optimization_steps = [
    {
        "step": 1,
        "action": "Single parameter optimization",
        "method": "Grid search on retention only",
        "complexity": "Low"
    },
    {
        "step": 2,
        "action": "Two-parameter optimization",
        "method": "Grid search on retention and limit",
        "complexity": "Medium"
    },
    {
        "step": 3,
        "action": "Multi-objective optimization",
        "method": "Pareto frontier analysis",
        "complexity": "High"
    },
    {
        "step": 4,
        "action": "Constrained optimization",
        "method": "Add risk and budget constraints",
        "complexity": "High"
    },
    {
        "step": 5,
        "action": "Dynamic optimization",
        "method": "Time-varying strategies",
        "complexity": "Very High"
    }
]

print("Progressive Optimization Approach:")
for step in optimization_steps:
    print(f"\nStep {step['step']}: {step['action']}")
    print(f"  Method: {step['method']}")
    print(f"  Complexity: {step['complexity']}")
```

### 2. Validate Results

```python
def validate_optimization_results(optimal_strategy):
    """Validate optimization results for reasonableness."""

    checks = []

    # Check 1: Premium reasonable % of limit
    premium_pct = optimal_strategy['premium_rate'] * 100
    checks.append({
        'check': 'Premium 1-5% of limit',
        'passed': 1 <= premium_pct <= 5,
        'value': f"{premium_pct:.1f}%"
    })

    # Check 2: Retention reasonable % of assets
    retention_pct = (optimal_strategy['retention'] / manufacturer.initial_assets) * 100
    checks.append({
        'check': 'Retention 1-20% of assets',
        'passed': 1 <= retention_pct <= 20,
        'value': f"{retention_pct:.1f}%"
    })

    # Check 3: Positive expected growth
    checks.append({
        'check': 'Positive expected growth',
        'passed': optimal_strategy['growth_rate'] > 0,
        'value': f"{optimal_strategy['growth_rate']:.2%}"
    })

    # Check 4: Acceptable survival rate
    checks.append({
        'check': 'Survival rate > 95%',
        'passed': optimal_strategy['survival_rate'] > 0.95,
        'value': f"{optimal_strategy['survival_rate']:.1%}"
    })

    print("\nOptimization Validation:")
    print("-" * 40)
    for check in checks:
        status = "✅" if check['passed'] else "❌"
        print(f"{status} {check['check']}: {check['value']}")

    return all(check['passed'] for check in checks)

# Validate sample optimal strategy
sample_optimal = {
    'retention': 1_000_000,
    'limit': 10_000_000,
    'premium_rate': 0.02,
    'growth_rate': 0.065,
    'survival_rate': 0.97
}

is_valid = validate_optimization_results(sample_optimal)
print(f"\nOptimization Valid: {is_valid}")
```

## Next Steps

Now that you can optimize insurance strategies:

1. **[Analyzing Results](05_analyzing_results.md)**: Deep dive into metrics and interpretation
2. **[Advanced Scenarios](06_advanced_scenarios.md)**: Complex real-world applications

## Summary

You've learned to:
- ✅ Define optimization objectives
- ✅ Perform single and multi-objective optimization
- ✅ Apply constraints (risk, budget)
- ✅ Analyze Pareto frontiers
- ✅ Implement dynamic strategies
- ✅ Validate and interpret results

You're ready to find optimal insurance strategies for any scenario!

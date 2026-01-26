# Optimization Workflow

This tutorial covers how to use the optimization tools to find the best insurance strategy for your business, considering both traditional metrics and ergodic growth rates.

## Optimization Overview

The framework provides tools to optimize:
- **Deductible/Retention levels**: How much risk to retain
- **Coverage limits**: How much insurance to purchase
- **Layer structure**: How to stack coverage efficiently
- **Premium budget allocation**: Where to spend insurance dollars

## Using the Business Optimizer

The `BusinessOptimizer` finds optimal insurance strategies based on your objectives:

```python
from ergodic_insurance import (
    BusinessOptimizer,
    BusinessObjective,
    BusinessConstraints,
    ManufacturerConfig
)

# Create optimizer
optimizer = BusinessOptimizer()

# Define your business configuration
manufacturer_config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0
)

# Define constraints
constraints = BusinessConstraints(
    min_survival_prob=0.95,      # Require 95% survival probability
    max_premium_ratio=0.02,      # Max 2% of assets for premium
    min_roe=0.05                 # Minimum 5% ROE target
)

# Run optimization
result = optimizer.optimize(
    manufacturer_config=manufacturer_config,
    objective=BusinessObjective.MAXIMIZE_GROWTH,
    constraints=constraints,
    time_horizon=50,
    n_simulations=1000
)

# Display results
print("=== Optimization Results ===")
print(f"Optimal Deductible: ${result.optimal_strategy.deductible:,.0f}")
print(f"Optimal Limit: ${result.optimal_strategy.limit:,.0f}")
print(f"Expected Growth Rate: {result.expected_growth_rate:.2%}")
print(f"Survival Probability: {result.survival_probability:.1%}")
```

## Business Objectives

Choose the optimization objective that matches your goals:

```python
from ergodic_insurance import BusinessObjective

# Available objectives
objectives = {
    BusinessObjective.MAXIMIZE_GROWTH: "Maximize time-average growth rate",
    BusinessObjective.MINIMIZE_RUIN: "Minimize probability of ruin",
    BusinessObjective.MAXIMIZE_TERMINAL_WEALTH: "Maximize expected final equity",
    BusinessObjective.MAXIMIZE_RISK_ADJUSTED_RETURN: "Maximize Sharpe-like ratio"
}

for obj, description in objectives.items():
    print(f"{obj.name}: {description}")
```

### Example: Maximize Growth

```python
result = optimizer.optimize(
    manufacturer_config=manufacturer_config,
    objective=BusinessObjective.MAXIMIZE_GROWTH,
    constraints=constraints,
    time_horizon=50,
    n_simulations=500
)

print(f"Growth-optimal deductible: ${result.optimal_strategy.deductible:,.0f}")
print(f"Time-average growth: {result.expected_growth_rate:.2%}")
```

### Example: Minimize Ruin Probability

```python
result = optimizer.optimize(
    manufacturer_config=manufacturer_config,
    objective=BusinessObjective.MINIMIZE_RUIN,
    constraints=constraints,
    time_horizon=50,
    n_simulations=500
)

print(f"Survival-optimal deductible: ${result.optimal_strategy.deductible:,.0f}")
print(f"Survival probability: {result.survival_probability:.1%}")
```

## Constraint Configuration

Fine-tune your constraints based on risk tolerance:

```python
# Conservative constraints
conservative = BusinessConstraints(
    min_survival_prob=0.99,      # Very high survival requirement
    max_premium_ratio=0.03,      # Willing to spend more on insurance
    min_roe=0.03,                # Lower return acceptable
    max_drawdown=0.20            # Limited downside tolerance
)

# Aggressive constraints
aggressive = BusinessConstraints(
    min_survival_prob=0.90,      # Accept more risk
    max_premium_ratio=0.01,      # Minimize insurance spend
    min_roe=0.08,                # Higher return required
    max_drawdown=0.40            # Accept larger drawdowns
)
```

## Sensitivity Analysis

Understand how results change with parameters:

```python
from ergodic_insurance import SensitivityAnalyzer, SensitivityResult

# Create analyzer
analyzer = SensitivityAnalyzer()

# Run one-way sensitivity on deductible
deductible_range = [50_000, 100_000, 250_000, 500_000, 1_000_000]

results = []
for deductible in deductible_range:
    result = optimizer.evaluate_strategy(
        manufacturer_config=manufacturer_config,
        deductible=deductible,
        limit=5_000_000,
        time_horizon=50,
        n_simulations=200
    )
    results.append({
        'deductible': deductible,
        'growth_rate': result.expected_growth_rate,
        'survival_prob': result.survival_probability,
        'premium': result.annual_premium
    })

# Display sensitivity table
import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
```

## Two-Way Sensitivity

Analyze interactions between parameters:

```python
# Two-way analysis: Deductible vs Limit
deductibles = [100_000, 250_000, 500_000]
limits = [2_000_000, 5_000_000, 10_000_000]

print("\nTime-Average Growth Rate by Deductible and Limit:")
print("-" * 50)
print(f"{'Deductible':>12}", end="")
for limit in limits:
    print(f"  ${limit/1e6:.0f}M Limit", end="")
print()

for deductible in deductibles:
    print(f"${deductible/1e3:>10.0f}K", end="")
    for limit in limits:
        result = optimizer.evaluate_strategy(
            manufacturer_config=manufacturer_config,
            deductible=deductible,
            limit=limit,
            time_horizon=30,
            n_simulations=100
        )
        print(f"     {result.expected_growth_rate:>6.2%}", end="")
    print()
```

## Parameter Sweep

Systematically explore the parameter space:

```python
from ergodic_insurance.parameter_sweep import ParameterSweep

# Define parameter ranges
sweep = ParameterSweep(
    deductible_range=(50_000, 1_000_000, 10),    # 10 points
    limit_range=(1_000_000, 20_000_000, 10),     # 10 points
    premium_loading_range=(0.2, 0.5, 5)           # 5 points
)

# Run sweep (this may run many simulations)
sweep_results = sweep.run(
    manufacturer_config=manufacturer_config,
    n_simulations=100,
    time_horizon=30
)

# Find optimal combination
best = sweep_results.best_combination(objective='growth_rate')
print(f"\nBest combination found:")
print(f"  Deductible: ${best['deductible']:,.0f}")
print(f"  Limit: ${best['limit']:,.0f}")
print(f"  Growth Rate: {best['growth_rate']:.2%}")
```

## Strategy Backtesting

Validate strategies against historical scenarios:

```python
from ergodic_insurance import StrategyBacktester

# Create backtester
backtester = StrategyBacktester()

# Define strategies to test
strategies = [
    {'name': 'No Insurance', 'deductible': float('inf'), 'limit': 0},
    {'name': 'Low Retention', 'deductible': 100_000, 'limit': 5_000_000},
    {'name': 'Medium Retention', 'deductible': 500_000, 'limit': 5_000_000},
    {'name': 'High Retention', 'deductible': 1_000_000, 'limit': 5_000_000},
]

# Run backtest
backtest_results = backtester.run(
    strategies=strategies,
    manufacturer_config=manufacturer_config,
    n_scenarios=500,
    time_horizon=30
)

# Compare strategies
print("\n=== Strategy Comparison ===")
print(f"{'Strategy':<20} {'Survival':>10} {'Growth':>10} {'Sharpe':>10}")
print("-" * 52)
for strategy, result in backtest_results.items():
    print(f"{strategy:<20} {result['survival']:.1%} {result['growth']:.2%} {result['sharpe']:.2f}")
```

## Walk-Forward Validation

Test strategy robustness over time:

```python
from ergodic_insurance import WalkForwardValidator

validator = WalkForwardValidator(
    training_window=20,   # 20 years to train
    testing_window=5,     # 5 years out-of-sample
    step_size=5           # Move forward 5 years each iteration
)

# Run walk-forward analysis
wf_results = validator.validate(
    optimizer=optimizer,
    manufacturer_config=manufacturer_config,
    total_years=50,
    n_simulations=200
)

# Analyze stability
print("\nWalk-Forward Results:")
print(f"In-sample mean growth: {wf_results['in_sample_mean']:.2%}")
print(f"Out-of-sample mean growth: {wf_results['out_of_sample_mean']:.2%}")
print(f"Strategy stability: {wf_results['stability_score']:.2f}")
```

## Optimization with Market Cycles

Account for insurance market conditions:

```python
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()

# Test under different market conditions
market_conditions = ['soft_market', 'hard_market', 'high_volatility']

print("\n=== Optimal Strategy by Market Condition ===")
for market in market_conditions:
    config = manager.load_profile("default", presets=[market])

    result = optimizer.optimize(
        manufacturer_config=manufacturer_config,
        objective=BusinessObjective.MAXIMIZE_GROWTH,
        constraints=constraints,
        market_config=config,
        time_horizon=30,
        n_simulations=200
    )

    print(f"\n{market.replace('_', ' ').title()}:")
    print(f"  Optimal Deductible: ${result.optimal_strategy.deductible:,.0f}")
    print(f"  Optimal Limit: ${result.optimal_strategy.limit:,.0f}")
    print(f"  Expected Growth: {result.expected_growth_rate:.2%}")
```

## Saving and Loading Optimization Results

```python
import json

# Save results
with open('optimization_results.json', 'w') as f:
    json.dump({
        'optimal_deductible': result.optimal_strategy.deductible,
        'optimal_limit': result.optimal_strategy.limit,
        'expected_growth': result.expected_growth_rate,
        'survival_prob': result.survival_probability,
        'parameters': {
            'time_horizon': 50,
            'n_simulations': 1000,
            'objective': 'MAXIMIZE_GROWTH'
        }
    }, f, indent=2)

print("Results saved to optimization_results.json")
```

## Best Practices

1. **Start with more simulations**: Use 500+ for reliable results
2. **Validate out-of-sample**: Always test on scenarios not used in optimization
3. **Consider multiple objectives**: Run optimization for different goals
4. **Account for model uncertainty**: Results depend on assumed distributions
5. **Update regularly**: Re-optimize as business conditions change

## Next Steps

- [Tutorial 5: Analyzing Results](05_analyzing_results.md) - Deep dive into ergodic analysis
- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) - Monte Carlo and complex setups

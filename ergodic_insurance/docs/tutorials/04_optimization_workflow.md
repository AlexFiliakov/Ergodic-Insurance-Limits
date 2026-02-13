# Tutorial 4: Optimization Workflow

**Prerequisites:** Tutorials 01-03 (basic concepts, loss modeling, insurance structures)

---

## The Story So Far

NovaTech Plastics has been running a basic insurance program (set up in Tutorial 03) with a $5M primary layer, $250K deductible, and a small excess layer. It works, but the CFO has a problem: NovaTech is planning to grow from $10M to $25M in assets over the next five years. The current insurance program was chosen by gut feel. As assets scale, mistakes in insurance design compound, and the CFO wants data-driven decisions.

This tutorial covers the optimization workflow: how to let the framework find the best insurance strategy for NovaTech's growth plan, test its sensitivity to assumptions, backtest alternatives, and validate robustness over time.

> **What "optimization" means here:** Traditional insurance buying optimizes for lowest premium. Ergodic optimization maximizes long-term *time-average growth rate*, the metric that actually determines whether your company survives and thrives across decades. These two objectives often produce very different answers.

---

## 1. Setting Up the BusinessOptimizer

The `BusinessOptimizer` wraps a manufacturer and searches over insurance parameters (coverage limit, deductible, premium rate) to find combinations that maximize business outcomes subject to constraints. It requires a `WidgetManufacturer` instance (the same one you have been using throughout the earlier tutorials).

```python
from ergodic_insurance import ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.business_optimizer import (
    BusinessOptimizer,
    BusinessObjective,
    BusinessConstraints,
    OptimizationDirection,
)

# NovaTech Plastics -- current state before expansion
mfg_config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0,
)
manufacturer = WidgetManufacturer(mfg_config)

# Create the optimizer around the manufacturer
optimizer = BusinessOptimizer(manufacturer=manufacturer)
```

That single call gives you access to several optimization methods:

| Method | What it does |
|--------|-------------|
| `maximize_roe_with_insurance()` | Maximize ROE subject to business constraints |
| `minimize_bankruptcy_risk()` | Minimize ruin probability while hitting growth targets |
| `optimize_business_outcomes()` | Multi-objective optimization with weighted objectives |
| `optimize_capital_efficiency()` | Allocate capital across insurance, working capital, growth |
| `analyze_time_horizon_impact()` | Compare strategies across short, medium, and long horizons |

---

## 2. Defining Objectives and Constraints

### Business Objectives

`BusinessObjective` is a dataclass, not an enum. You construct individual objectives and pass them as a list. Each objective has a name, a weight, and a direction (maximize or minimize).

```python
# Objective 1: Maximize time-average growth rate (most important for NovaTech's expansion)
growth_objective = BusinessObjective(
    name="growth_rate",
    weight=0.6,
    optimization_direction=OptimizationDirection.MAXIMIZE,
)

# Objective 2: Keep bankruptcy risk low
safety_objective = BusinessObjective(
    name="bankruptcy_risk",
    weight=0.3,
    optimization_direction=OptimizationDirection.MINIMIZE,
)

# Objective 3: Maintain strong ROE for investors
roe_objective = BusinessObjective(
    name="ROE",
    weight=0.1,
    optimization_direction=OptimizationDirection.MAXIMIZE,
)
```

The `name` field maps to an internal evaluation function. Supported objective names include `"growth_rate"`, `"bankruptcy_risk"`, `"ROE"`, and `"capital_efficiency"`.

You can also attach hard constraints directly to an objective:

```python
# Growth rate must be at least 5%
growth_floor = BusinessObjective(
    name="growth_rate",
    weight=0.6,
    optimization_direction=OptimizationDirection.MAXIMIZE,
    constraint_type=">=",
    constraint_value=0.05,
)
```

### Business Constraints

`BusinessConstraints` defines the guardrails that every feasible solution must satisfy. The defaults reflect a moderate risk posture:

```python
# NovaTech's CFO is conservative -- low risk tolerance, modest premium budget
constraints = BusinessConstraints(
    max_risk_tolerance=0.01,      # 1% max bankruptcy probability
    min_roe_threshold=0.10,       # At least 10% ROE
    max_leverage_ratio=2.0,       # No more than 2:1 debt-to-equity
    min_liquidity_ratio=1.2,      # 1.2x current ratio minimum
    max_premium_budget=0.02,      # Premium capped at 2% of revenue
    min_coverage_ratio=0.5,       # Coverage must be at least 50% of assets
)
```

Each constraint maps to an inequality check during optimization. If the optimizer cannot find a feasible solution, it will log a warning and return the best infeasible point it found. You can check feasibility on the result:

```python
result = ...  # (we'll run this next)
if not result.is_feasible():
    print("Warning: not all constraints satisfied")
    print(result.constraint_satisfaction)
```

---

## 3. Running Optimization

### Multi-Objective Optimization

The `optimize_business_outcomes()` method is the most powerful entry point. It accepts a list of objectives, a constraints object, a time horizon, and an optimization method.

```python
result = optimizer.optimize_business_outcomes(
    objectives=[growth_objective, safety_objective, roe_objective],
    constraints=constraints,
    time_horizon=10,            # 10-year planning horizon for NovaTech
    method="weighted_sum",      # also supports "epsilon_constraint", "pareto"
)

# The result is a BusinessOptimizationResult
strategy = result.optimal_strategy

print("=== NovaTech Optimal Insurance Strategy ===")
print(f"Coverage Limit:    ${strategy.coverage_limit:,.0f}")
print(f"Deductible:        ${strategy.deductible:,.0f}")
print(f"Premium Rate:      {strategy.premium_rate:.2%}")
print(f"Expected ROE:      {strategy.expected_roe:.2%}")
print(f"Bankruptcy Risk:   {strategy.bankruptcy_risk:.4%}")
print(f"Growth Rate:       {strategy.growth_rate:.2%}")
print(f"Capital Efficiency:{strategy.capital_efficiency:.3f}")

print("\nRecommendations:")
for rec in strategy.recommendations:
    print(f"  - {rec}")
```

The result object also contains detailed diagnostics:

```python
# Did the optimizer converge?
print(f"Converged: {result.convergence_info['converged']}")
print(f"Iterations: {result.convergence_info['iterations']}")

# How well did each objective score?
for obj_name, value in result.objective_values.items():
    print(f"  {obj_name}: {value:.4f}")

# Which constraints are binding?
for constraint_name, satisfied in result.constraint_satisfaction.items():
    status = "OK" if satisfied else "VIOLATED"
    print(f"  {constraint_name}: {status}")
```

### Single-Objective Shortcuts

For simpler cases, you can use the dedicated single-objective methods:

```python
# Maximize ROE directly
roe_strategy = optimizer.maximize_roe_with_insurance(
    constraints=constraints,
    time_horizon=10,
    n_simulations=1000,
)
print(f"ROE-optimal deductible: ${roe_strategy.deductible:,.0f}")
print(f"Expected ROE: {roe_strategy.expected_roe:.2%}")

# Minimize bankruptcy risk while hitting growth targets
safe_strategy = optimizer.minimize_bankruptcy_risk(
    growth_targets={"revenue": 0.08, "assets": 0.05},
    budget_constraint=200_000,   # max annual premium
    time_horizon=10,
)
print(f"Safety-optimal coverage: ${safe_strategy.coverage_limit:,.0f}")
print(f"Bankruptcy risk: {safe_strategy.bankruptcy_risk:.4%}")
```

### Time Horizon Analysis

NovaTech's CFO wants to understand how the optimal strategy changes as the planning horizon extends. This is where ergodic effects become most visible: short horizons favor ensemble-average thinking, but long horizons reward time-average optimization.

```python
strategies = [
    {"name": "Low Retention",  "coverage_limit": 8_000_000, "deductible": 100_000, "premium_rate": 0.02},
    {"name": "Medium Retention","coverage_limit": 8_000_000, "deductible": 500_000, "premium_rate": 0.015},
    {"name": "High Retention", "coverage_limit": 5_000_000, "deductible": 1_000_000,"premium_rate": 0.01},
]

horizon_df = optimizer.analyze_time_horizon_impact(
    strategies=strategies,
    time_horizons=[1, 3, 10, 30],
)

# Show how strategies rank differently at different horizons
print(horizon_df[["strategy", "horizon_years", "expected_roe", "bankruptcy_risk", "growth_rate"]])
```

Over short horizons (1-3 years), the low-retention strategy often looks best because premium savings boost near-term ROE. Over long horizons (10-30 years), moderate retention tends to win because it balances cost against the catastrophic downside that erodes time-average growth.

---

## 4. Sensitivity Analysis

The optimizer gives you one answer, but how fragile is it? Sensitivity analysis reveals which assumptions matter most and where the strategy is robust.

### One-Way Sensitivity

The `SensitivityAnalyzer` takes a base configuration dictionary and an optimizer, then sweeps individual parameters to see how outcomes change.

```python
from ergodic_insurance.sensitivity import SensitivityAnalyzer

# Define the base configuration -- the starting point for perturbation
base_config = {
    "deductible": 250_000,
    "limit": 5_000_000,
    "premium_rate": 0.02,
    "frequency": 5,
    "severity_mean": 200_000,
}

analyzer = SensitivityAnalyzer(
    base_config=base_config,
    optimizer=optimizer,
)

# How sensitive is the result to the deductible level?
deductible_sensitivity = analyzer.analyze_parameter(
    param_name="deductible",
    param_range=(50_000, 1_000_000),
    n_points=10,
)

# Convert to DataFrame for easy inspection
df = deductible_sensitivity.to_dataframe()
print(df[["parameter_value", "optimal_roe", "bankruptcy_risk", "growth_rate"]])
```

The `SensitivityResult` object also lets you compute standardized impact scores:

```python
# How elastic is ROE with respect to the deductible?
impact = deductible_sensitivity.calculate_impact("optimal_roe")
print(f"Deductible elasticity on ROE: {impact:.3f}")

# What are the bounds on growth rate across the deductible range?
low, high = deductible_sensitivity.get_metric_bounds("growth_rate")
print(f"Growth rate range: {low:.2%} to {high:.2%}")
```

### Tornado Diagram

A tornado diagram ranks parameters by their impact on a chosen metric, showing which assumptions NovaTech's CFO should worry about most.

```python
# Which parameters have the largest impact on growth rate?
tornado_df = analyzer.create_tornado_diagram(
    parameters=["deductible", "limit", "premium_rate", "frequency", "severity_mean"],
    metric="growth_rate",
    relative_range=0.5,   # +/- 50% from baseline
    n_points=5,
)

print("=== Tornado Diagram: Impact on Growth Rate ===")
print(tornado_df[["parameter", "impact", "low_value", "high_value", "baseline"]].to_string(index=False))
```

The result is sorted by impact magnitude. Typically, loss frequency and severity dominate, which reinforces why insurance (which caps their effect) has outsized value for growth.

### Two-Way Sensitivity

When two parameters interact, one-way analysis misses the picture. A two-way analysis creates a grid of outcomes.

```python
two_way_result = analyzer.analyze_two_way(
    param1="deductible",
    param2="limit",
    param1_range=(50_000, 1_000_000),
    param2_range=(2_000_000, 10_000_000),
    n_points1=8,
    n_points2=8,
    metric="growth_rate",
)

# Convert to DataFrame
grid_df = two_way_result.to_dataframe()
print(grid_df.head(10))

# Find parameter combinations that achieve at least 8% growth
target_region = two_way_result.find_optimal_region(
    target_value=0.08,
    tolerance=0.10,   # within 10% of target
)
print(f"Number of feasible combinations: {target_region.sum()}")
```

This is especially useful for NovaTech because deductible and limit interact: a high deductible with a low limit might save premium but expose the company to a gap in coverage that destroys long-term growth.

---

## 5. Strategy Backtesting

Optimization tells you what *should* work. Backtesting tells you what *would have* worked across many simulated scenarios. The framework provides several predefined strategy classes that you can test head-to-head.

### Predefined Strategies

```python
from ergodic_insurance.strategy_backtester import (
    NoInsuranceStrategy,
    ConservativeFixedStrategy,
    AggressiveFixedStrategy,
    AdaptiveStrategy,
    StrategyBacktester,
)
from ergodic_insurance.monte_carlo import MonteCarloConfig

# 1. No insurance (baseline)
no_insurance = NoInsuranceStrategy()

# 2. Conservative: high limits, low deductible, multiple layers
conservative = ConservativeFixedStrategy(
    primary_limit=5_000_000,
    excess_limit=20_000_000,
    deductible=50_000,
)

# 3. Aggressive: lower limits, higher deductible
aggressive = AggressiveFixedStrategy(
    primary_limit=2_000_000,
    excess_limit=5_000_000,
    deductible=250_000,
)

# 4. Adaptive: adjusts limits based on recent loss experience
adaptive = AdaptiveStrategy(
    base_deductible=100_000,
    base_primary=3_000_000,
    base_excess=10_000_000,
    adaptation_window=3,       # look at last 3 years
    adjustment_factor=0.2,     # adjust limits by up to 20%
)
```

### Running the Backtest

```python
# Configure the simulation
sim_config = MonteCarloConfig(
    n_simulations=500,
    n_years=10,
    seed=42,
)

# Create the backtester
backtester = StrategyBacktester(simulation_engine=None)

# Test a single strategy
result = backtester.test_strategy(
    strategy=conservative,
    manufacturer=manufacturer,
    config=sim_config,
    use_cache=True,
)

print(f"Strategy: {result.strategy_name}")
print(f"ROE: {result.metrics.roe:.2%}")
print(f"Ruin Probability: {result.metrics.ruin_probability:.4%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")
print(f"Execution Time: {result.execution_time:.1f}s")
```

### Comparing Multiple Strategies

The real power comes from side-by-side comparison:

```python
strategies = [no_insurance, conservative, aggressive, adaptive]

comparison_df = backtester.test_multiple_strategies(
    strategies=strategies,
    manufacturer=manufacturer,
    config=sim_config,
)

# Sort by growth rate to see the ergodic winner
comparison_df = comparison_df.sort_values("growth_rate", ascending=False)
print("\n=== Strategy Comparison (sorted by growth rate) ===")
print(comparison_df.to_string(index=False))
```

The results typically reveal a pattern: "no insurance" has the highest ensemble-average ROE but the worst ruin probability and time-average growth. The conservative strategy survives reliably but pays too much in premium. The adaptive strategy often strikes the best balance, adjusting coverage as loss experience evolves.

---

## 6. Walk-Forward Validation

A backtest on the full dataset can overfit. Walk-forward validation splits the timeline into rolling train/test windows to check whether a strategy that looks good in-sample also performs well out-of-sample.

```python
from ergodic_insurance.walk_forward_validator import WalkForwardValidator

validator = WalkForwardValidator(
    window_size=3,     # each window is 3 years
    step_size=1,       # roll forward 1 year at a time
)

# Validate NovaTech's candidate strategies
validation_result = validator.validate_strategies(
    strategies=[conservative, aggressive, adaptive],
    n_years=10,
    n_simulations=1000,
    manufacturer=manufacturer,
)

# Which strategy wins overall?
print(f"\nBest Strategy: {validation_result.best_strategy}")

# Strategy rankings across all windows
print("\n=== Strategy Rankings ===")
print(validation_result.strategy_rankings.to_string(index=False))
```

### Overfitting and Consistency Scores

The validator computes two key diagnostics:

```python
# Overfitting score: how much does in-sample performance exceed out-of-sample?
# Lower is better (< 0.2 = good, > 0.4 = concerning)
print("\nOverfitting Analysis:")
for strategy_name, score in validation_result.overfitting_analysis.items():
    label = "Low" if score < 0.2 else "Moderate" if score < 0.4 else "High"
    print(f"  {strategy_name}: {score:.3f} ({label})")

# Consistency score: how stable is performance across windows?
# Higher is better (> 0.8 = stable)
print("\nConsistency Scores:")
for strategy_name, score in validation_result.consistency_scores.items():
    label = "High" if score > 0.8 else "Moderate" if score > 0.6 else "Low"
    print(f"  {strategy_name}: {score:.3f} ({label})")
```

### Generating Reports

The validator can produce a full HTML report with visualizations:

```python
report_files = validator.generate_report(
    validation_result,
    output_dir="./reports",
    include_visualizations=True,
)
print(f"Report saved to: {report_files['html']}")
```

---

## 7. Market Cycle Considerations

Insurance markets swing between soft markets (cheap, broad coverage) and hard markets (expensive, restrictive coverage). An optimal strategy in one regime may be suboptimal in another. The `ConfigManager` provides preset market conditions that you can layer onto your analysis.

```python
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()

# Load the default configuration
default_config = manager.load_profile("default")

# Apply a hard market preset
hard_market_config = manager.load_profile("default", presets=["hard_market"])

# Apply a high volatility preset
volatile_config = manager.load_profile("default", presets=["high_volatility"])
```

You can run optimization under each regime and compare:

```python
market_conditions = {
    "Normal Market": manager.load_profile("default"),
    "Hard Market": manager.load_profile("default", presets=["hard_market"]),
}

for market_name, config in market_conditions.items():
    result = optimizer.optimize_business_outcomes(
        objectives=[growth_objective, safety_objective],
        constraints=constraints,
        time_horizon=10,
    )
    strategy = result.optimal_strategy

    print(f"\n{market_name}:")
    print(f"  Optimal Deductible:  ${strategy.deductible:,.0f}")
    print(f"  Optimal Limit:       ${strategy.coverage_limit:,.0f}")
    print(f"  Expected Growth:     {strategy.growth_rate:.2%}")
```

In hard markets, the optimizer typically pushes toward higher retentions (accepting more self-insured risk) because premium costs eat into growth. In soft markets, it may recommend buying more coverage because the cost-benefit ratio is favorable.

---

## 8. Best Practices

**1. Use enough simulations.** For reliable optimization, use at least 100,000 simulations, particularly when operational volatility is incorporated as described in the Advanced Scenarios Tutorial. For final decisions, use 250,000 or more. The optimizer's internal methods use fewer simulations for speed during search, but you should validate the final answer with more.

**2. Always validate out-of-sample.** An optimized strategy can overfit to the specific loss distribution and time horizon. Use `WalkForwardValidator` before committing to a strategy.

**3. Run sensitivity analysis before optimization.** Understanding which parameters matter most (via tornado diagrams) helps you focus the optimizer on the right levers and interpret its results.

**4. Consider multiple objectives.** Maximizing growth alone can lead to fragile strategies. Include a risk or survival objective to prevent the optimizer from recommending dangerously thin coverage.

**5. Account for model uncertainty.** The optimizer treats the loss distribution as known, but in practice it is estimated. Test your optimal strategy against perturbed distributions (higher frequency, fatter tails) to ensure it is not brittle.

**6. Re-optimize as conditions change.** The optimal strategy depends on asset levels, operating margins, loss experience, and market conditions. Re-run the optimization annually or after significant changes to NovaTech's business.

**7. Compare against simple baselines.** Always include "no insurance" and a simple fixed strategy in your backtest. If the optimized strategy does not clearly beat these baselines out-of-sample, the added complexity is not justified.

---

## 9. Exercises

### Exercise 1: Risk-Averse vs. Growth-Focused Optimization

Define two sets of objectives for NovaTech:

- **Risk-averse:** 70% weight on minimizing bankruptcy risk, 30% on ROE. Use tighter constraints (`max_risk_tolerance=0.005`, `min_roe_threshold=0.08`).
- **Growth-focused:** 70% weight on maximizing growth rate, 30% on ROE. Use looser constraints (`max_risk_tolerance=0.02`, `min_roe_threshold=0.12`).

Run `optimize_business_outcomes()` for each set and compare the resulting optimal strategies. How do the deductible, coverage limit, and premium rate differ? Which strategy has a higher capital efficiency ratio?

### Exercise 2: Tornado Diagram Analysis

Set up a `SensitivityAnalyzer` with the following base configuration:

```python
base_config = {
    "deductible": 250_000,
    "limit": 5_000_000,
    "premium_rate": 0.02,
    "frequency": 5,
    "severity_mean": 200_000,
}
```

Run a tornado diagram for all five parameters against the `"optimal_roe"` metric with a `relative_range` of 0.3. Which parameter has the highest impact? Run it again with `"bankruptcy_risk"` as the metric. Does the ranking change? Write a brief interpretation of what this means for NovaTech's risk management priorities.

### Exercise 3: Strategy Backtest and Ranking

Create four strategies using the predefined classes:

1. `NoInsuranceStrategy()`
2. `ConservativeFixedStrategy(primary_limit=5_000_000, deductible=50_000)`
3. `AggressiveFixedStrategy(primary_limit=2_000_000, deductible=250_000)`
4. `AdaptiveStrategy(base_deductible=100_000, base_primary=3_000_000, adaptation_window=3)`

Backtest all four using `test_multiple_strategies()` with 500 simulations over 10 years. Sort the results by `growth_rate`, then by `sharpe_ratio`. Do the rankings differ? Which strategy would you recommend to NovaTech's CFO, and why?

---

## 10. Next Steps

- [Tutorial 5: Analyzing Results](05_analyzing_results.md) -- Deep dive into ergodic vs. ensemble analysis and how to interpret simulation output
- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) -- Monte Carlo techniques, parameter sweeps, and complex multi-layer programs

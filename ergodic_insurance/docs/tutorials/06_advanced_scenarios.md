# Advanced Scenarios

This tutorial covers advanced features including Monte Carlo simulations, market cycle modeling, complex insurance structures, and performance optimization.

## Monte Carlo Simulation

For robust analysis, run thousands of simulations:

```python
from ergodic_insurance.monte_carlo import MonteCarloEngine, SimulationConfig
from ergodic_insurance import ManufacturerConfig
from ergodic_insurance.insurance_program import InsuranceProgram, EnhancedInsuranceLayer

# Configuration
manufacturer_config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0
)

# Simulation configuration
sim_config = SimulationConfig(
    n_simulations=1000,
    time_horizon=50,
    seed=42,
    parallel=True,         # Use multiple CPU cores
    n_workers=4            # Number of parallel workers
)

# Insurance program
insurance = InsuranceProgram(deductible=250_000)
insurance.add_layer(EnhancedInsuranceLayer(
    attachment_point=250_000, limit=5_000_000, base_premium_rate=0.025
))
insurance.add_layer(EnhancedInsuranceLayer(
    attachment_point=5_250_000, limit=10_000_000, base_premium_rate=0.012
))

# Create and run Monte Carlo engine
engine = MonteCarloEngine(
    manufacturer_config=manufacturer_config,
    insurance_program=insurance,
    claim_frequency=0.15,
    claim_severity_mean=1_000_000,
    claim_severity_cv=1.5
)

results = engine.run(sim_config)

# Summary statistics
print("=== Monte Carlo Results ===")
print(f"Simulations: {results.n_simulations}")
print(f"Time horizon: {results.time_horizon} years")
print(f"\nSurvival rate: {results.survival_rate:.1%}")
print(f"Mean final equity: ${results.mean_final_equity:,.0f}")
print(f"Median final equity: ${results.median_final_equity:,.0f}")
print(f"Time-average growth: {results.time_average_growth:.2%}")
print(f"Ensemble-average growth: {results.ensemble_average_growth:.2%}")
```

## Parallel Processing

For large simulations, use parallel execution:

```python
from ergodic_insurance.parallel_executor import ParallelExecutor

# Create parallel executor
executor = ParallelExecutor(
    n_workers=8,           # Use 8 CPU cores
    chunk_size=100,        # Process 100 simulations per chunk
    progress_bar=True      # Show progress
)

# Run parallel Monte Carlo
parallel_results = executor.run_monte_carlo(
    manufacturer_config=manufacturer_config,
    insurance_program=insurance,
    n_simulations=10000,
    time_horizon=50
)

print(f"Completed {parallel_results.n_simulations} simulations")
print(f"Total runtime: {parallel_results.runtime:.1f} seconds")
print(f"Simulations per second: {parallel_results.simulations_per_second:.0f}")
```

## Configuration Manager Profiles

Use predefined profiles for common scenarios:

```python
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()

# Available profiles
profiles = ['default', 'conservative', 'aggressive']

print("=== Profile Comparison ===")
for profile_name in profiles:
    config = manager.load_profile(profile_name)
    print(f"\n{profile_name.upper()} Profile:")
    print(f"  Growth rate: {config.growth.annual_growth_rate:.1%}")
    print(f"  Volatility: {config.growth.volatility:.1%}")
    print(f"  Operating margin: {config.manufacturer.base_operating_margin:.1%}")
```

### Profile Inheritance

Extend existing profiles with overrides:

```python
# Load profile with overrides
custom_config = manager.load_profile(
    "conservative",
    manufacturer={
        "base_operating_margin": 0.06,
        "initial_assets": 15_000_000
    },
    growth={
        "volatility": 0.25
    }
)

print(f"Custom margin: {custom_config.manufacturer.base_operating_margin:.1%}")
print(f"Custom volatility: {custom_config.growth.volatility:.1%}")
```

### Using Presets

Apply market condition presets:

```python
# Hard market conditions
hard_market = manager.load_profile("default", presets=["hard_market"])

# Soft market conditions
soft_market = manager.load_profile("default", presets=["soft_market"])

# High volatility scenario
volatile = manager.load_profile("default", presets=["high_volatility"])

# Multiple presets
recession = manager.load_profile(
    "conservative",
    presets=["hard_market", "high_volatility"]
)
```

## Market Cycle Modeling

Model insurance market cycles:

```python
from ergodic_insurance.insurance_pricing import MarketCycle

# Define market cycle
cycle = MarketCycle(
    cycle_length=10,           # 10-year cycle
    soft_market_years=4,       # 4 years soft market
    hard_market_years=3,       # 3 years hard market
    transition_years=3,        # 3 years transition
    soft_market_factor=0.8,    # 20% premium discount in soft market
    hard_market_factor=1.4     # 40% premium increase in hard market
)

# Apply to simulation
for year in range(20):
    premium_factor = cycle.get_factor(year)
    phase = cycle.get_phase(year)
    print(f"Year {year:2d}: {phase:12s} - Premium factor: {premium_factor:.2f}")
```

## Complex Insurance Structures

### Layered Program with Reinstatements

```python
from ergodic_insurance.insurance_program import (
    InsuranceProgram, EnhancedInsuranceLayer, ReinstatementType
)

# Complex program
program = InsuranceProgram(deductible=500_000)

# Primary layer with 2 reinstatements
primary = EnhancedInsuranceLayer(
    attachment_point=500_000,
    limit=2_500_000,
    base_premium_rate=0.04,
    reinstatements=2,
    reinstatement_premium=1.0,
    reinstatement_type=ReinstatementType.PRO_RATA
)

# First excess with quota share
excess_1 = EnhancedInsuranceLayer(
    attachment_point=3_000_000,
    limit=5_000_000,
    base_premium_rate=0.02,
    participation_rate=0.75  # 75% placement
)

# High excess layer
excess_2 = EnhancedInsuranceLayer(
    attachment_point=8_000_000,
    limit=10_000_000,
    base_premium_rate=0.008,
    limit_type="aggregate",
    aggregate_limit=15_000_000
)

program.add_layer(primary)
program.add_layer(excess_1)
program.add_layer(excess_2)

# Display structure
print("=== Complex Insurance Tower ===")
print(f"Deductible: ${program.deductible:,.0f}")
for i, layer in enumerate(program.layers, 1):
    print(f"\nLayer {i}:")
    print(f"  Attachment: ${layer.attachment_point:,.0f}")
    print(f"  Limit: ${layer.limit:,.0f}")
    print(f"  Premium Rate: {layer.base_premium_rate:.2%}")
    if layer.reinstatements > 0:
        print(f"  Reinstatements: {layer.reinstatements}")
    if layer.participation_rate < 1.0:
        print(f"  Participation: {layer.participation_rate:.0%}")
    if layer.limit_type != "per-occurrence":
        print(f"  Limit Type: {layer.limit_type}")

print(f"\nTotal Premium: ${program.total_premium():,.0f}")
print(f"Total Coverage: ${program.total_coverage():,.0f}")
```

## Stochastic Processes

Model revenue and loss volatility:

```python
from ergodic_insurance.stochastic_processes import (
    GeometricBrownianMotion, MeanRevertingProcess, JumpDiffusion
)

# Geometric Brownian Motion for base growth
gbm = GeometricBrownianMotion(
    drift=0.08,         # 8% expected growth
    volatility=0.15,    # 15% volatility
    seed=42
)

# Generate paths
years = 50
n_paths = 1000
paths = gbm.generate_paths(
    initial_value=10_000_000,
    time_horizon=years,
    n_paths=n_paths
)

print(f"Generated {n_paths} paths over {years} years")
print(f"Mean final value: ${paths[:, -1].mean():,.0f}")
print(f"Std final value: ${paths[:, -1].std():,.0f}")
```

## Loss Distribution Fitting

Fit distributions to historical loss data:

```python
from ergodic_insurance.loss_distributions import LossDistributionFitter

# Example historical claims data
historical_claims = [
    150_000, 75_000, 250_000, 1_200_000, 80_000,
    320_000, 45_000, 890_000, 2_500_000, 175_000
]

# Fit distribution
fitter = LossDistributionFitter()
best_fit = fitter.fit(
    data=historical_claims,
    distributions=['lognormal', 'pareto', 'weibull', 'gamma']
)

print("=== Distribution Fit Results ===")
print(f"Best fit: {best_fit.distribution_name}")
print(f"Parameters: {best_fit.parameters}")
print(f"AIC: {best_fit.aic:.2f}")
print(f"KS statistic: {best_fit.ks_statistic:.4f}")

# Generate synthetic claims
synthetic_claims = best_fit.sample(100)
print(f"\nSynthetic claims mean: ${synthetic_claims.mean():,.0f}")
print(f"Synthetic claims std: ${synthetic_claims.std():,.0f}")
```

## Performance Benchmarking

Measure and optimize performance:

```python
from ergodic_insurance import BenchmarkSuite, BenchmarkConfig

# Create benchmark configuration
benchmark_config = BenchmarkConfig(
    n_iterations=5,
    warmup_iterations=2,
    simulation_sizes=[100, 500, 1000],
    time_horizons=[10, 30, 50]
)

# Run benchmarks
suite = BenchmarkSuite()
benchmark_results = suite.run(benchmark_config)

# Display results
print("=== Performance Benchmarks ===")
for result in benchmark_results:
    print(f"\n{result.test_name}:")
    print(f"  Mean time: {result.mean_time:.3f}s")
    print(f"  Std time: {result.std_time:.3f}s")
    print(f"  Simulations/sec: {result.simulations_per_second:.1f}")
```

## Caching for Speed

Enable caching for repeated calculations:

```python
from ergodic_insurance import SmartCache

# Enable caching
cache = SmartCache(max_size=1000, ttl_seconds=3600)

# Expensive calculation with caching
@cache.memoize
def expensive_calculation(deductible, limit, n_simulations):
    # This would normally run many simulations
    # Results are cached based on parameters
    engine = MonteCarloEngine(...)
    return engine.run(...)

# First call: computes result
result1 = expensive_calculation(100_000, 5_000_000, 1000)

# Second call: returns cached result instantly
result2 = expensive_calculation(100_000, 5_000_000, 1000)  # Cached!

# Different parameters: new computation
result3 = expensive_calculation(200_000, 5_000_000, 1000)  # New
```

## Scenario Analysis

Run comprehensive scenario analysis:

```python
from ergodic_insurance.scenario_manager import ScenarioManager

# Define scenarios
manager = ScenarioManager()

scenarios = {
    'base_case': {
        'claim_frequency': 0.15,
        'claim_severity_mean': 1_000_000,
        'growth_rate': 0.08,
        'volatility': 0.15
    },
    'recession': {
        'claim_frequency': 0.20,
        'claim_severity_mean': 1_200_000,
        'growth_rate': -0.02,
        'volatility': 0.25
    },
    'expansion': {
        'claim_frequency': 0.10,
        'claim_severity_mean': 800_000,
        'growth_rate': 0.12,
        'volatility': 0.12
    },
    'catastrophe': {
        'claim_frequency': 0.25,
        'claim_severity_mean': 2_000_000,
        'growth_rate': 0.05,
        'volatility': 0.30
    }
}

# Run all scenarios
results = {}
for name, params in scenarios.items():
    result = manager.run_scenario(
        name=name,
        manufacturer_config=manufacturer_config,
        insurance_program=insurance,
        n_simulations=500,
        time_horizon=30,
        **params
    )
    results[name] = result

# Compare scenarios
print("=== Scenario Comparison ===")
print(f"{'Scenario':<15} {'Survival':>10} {'Growth':>10} {'Final Equity':>15}")
print("-" * 52)
for name, result in results.items():
    print(f"{name:<15} {result.survival_rate:>9.1%} {result.time_average_growth:>9.2%} ${result.mean_final_equity:>13,.0f}")
```

## Excel Reports

Generate comprehensive Excel reports:

```python
from ergodic_insurance.excel_reporter import ExcelReporter

# Create reporter
reporter = ExcelReporter()

# Generate report
reporter.generate(
    results=results,
    output_path='insurance_analysis_report.xlsx',
    include_charts=True,
    include_raw_data=True,
    include_summary=True
)

print("Report saved to insurance_analysis_report.xlsx")
```

## Summary

This tutorial covered:
- Monte Carlo simulation with parallel processing
- Configuration profiles and presets
- Market cycle modeling
- Complex insurance structures
- Stochastic process modeling
- Loss distribution fitting
- Performance benchmarking and caching
- Comprehensive scenario analysis
- Excel reporting

## Further Resources

- **API Documentation**: See the module docstrings for detailed API reference
- **Examples**: Check `ergodic_insurance/examples/` for more code samples
- **Notebooks**: Interactive examples in `ergodic_insurance/notebooks/`
- **Theory**: Background on ergodic economics and insurance optimization

# Tutorial 6: Advanced Scenarios

**Prerequisites:** Tutorials 01-05 (business modeling, simulation, insurance structures, optimization, result analysis)

---

## The Story So Far

NovaTech Plastics has grown. After applying the optimization workflow from Tutorial 4 and analyzing results in Tutorial 5, the company has expanded from $10M to **$25M in total assets**. The CFO now faces a harder question: NovaTech is building a **five-year strategic plan** and needs to stress-test the insurance program against recessions, market hardening, catastrophic loss years, and expansion scenarios -- simultaneously.

Single-path simulations are no longer sufficient. NovaTech needs Monte Carlo analysis at scale, parallel processing to keep runtimes manageable, configuration profiles for repeatable experiments, and structured scenario comparison to present findings to the board.

This tutorial covers the advanced features that make that kind of analysis possible.

---

## 1. Monte Carlo Simulation with MonteCarloEngine

The `MonteCarloEngine` runs thousands of independent simulation paths in parallel, each with its own loss history, and aggregates the results into robust statistical summaries. Unlike the single-trajectory `Simulation` engine used in earlier tutorials, Monte Carlo analysis captures the **full distribution** of outcomes -- including the tail events that determine survival.

### Setting Up the Engine

The engine requires three components: a **loss generator** (how losses arrive), an **insurance program** (how losses are transferred), and a **manufacturer** (the business being modeled). Configuration is handled through `MonteCarloConfig`.

```python
from ergodic_insurance.monte_carlo import MonteCarloEngine, MonteCarloConfig
from ergodic_insurance import (
    InsuranceProgram, EnhancedInsuranceLayer, ManufacturerConfig,
)
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

# --- NovaTech at $25M ---
mfg_config = ManufacturerConfig(
    initial_assets=25_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0,
    ppe_ratio=0.5,
    insolvency_tolerance=10_000
)
manufacturer = WidgetManufacturer(mfg_config)

# Loss generator calibrated for a $25M manufacturer
loss_gen = ManufacturingLossGenerator(seed=42)

# Current insurance program: $500K deductible, two layers
program = InsuranceProgram(
    layers=[
        EnhancedInsuranceLayer(
            attachment_point=500_000,
            limit=5_000_000,
            base_premium_rate=0.025
        ),
        EnhancedInsuranceLayer(
            attachment_point=5_500_000,
            limit=10_000_000,
            base_premium_rate=0.012
        ),
    ],
    deductible=500_000,
)

# Simulation configuration
sim_config = MonteCarloConfig(
    n_simulations=1_000,      # Start moderate; scale up later
    n_years=50,               # Long horizon for ergodic effects
    parallel=True,            # Use multiple CPU cores
    n_workers=4,              # Number of parallel workers
    chunk_size=100,           # Simulations processed per chunk
    seed=42,                  # Reproducibility
    progress_bar=True         # Show progress during execution
)

# Build and run the engine
engine = MonteCarloEngine(
    loss_generator=loss_gen,
    insurance_program=program,
    manufacturer=manufacturer,
    config=sim_config
)

results = engine.run()
```

> **Note:** Both the `Simulation` engine and `MonteCarloEngine` accept `InsuranceProgram` (with `EnhancedInsuranceLayer`). The legacy `InsurancePolicy`/`InsuranceLayer` classes are deprecated.

### Reading the Results

`SimulationResults` contains numpy arrays for every simulation path, plus pre-computed risk metrics.

```python
import numpy as np

print("=== NovaTech Monte Carlo Results ===")
print(f"Simulations: {results.config.n_simulations:,}")
print(f"Years per path: {results.config.n_years}")
print(f"Execution time: {results.execution_time:.1f}s")
print()

# Ruin probability at different horizons
print("Ruin Probability:")
for year_str in sorted(results.ruin_probability.keys(), key=int):
    print(f"  Year {year_str}: {results.ruin_probability[year_str]:.2%}")

# Asset distribution at final year
print(f"\nMean final assets: ${np.mean(results.final_assets):,.0f}")
print(f"Median final assets: ${np.median(results.final_assets):,.0f}")
print(f"10th percentile: ${np.percentile(results.final_assets, 10):,.0f}")
print(f"90th percentile: ${np.percentile(results.final_assets, 90):,.0f}")

# Growth rates
print(f"\nMean growth rate: {np.mean(results.growth_rates):.4f}")

# Risk metrics (VaR, TVaR)
print(f"VaR(99%): ${results.metrics.get('var_99', 0):,.0f}")
print(f"TVaR(99%): ${results.metrics.get('tvar_99', 0):,.0f}")

# Full formatted summary
print(results.summary())
```

### Comparing Insured vs. Uninsured

Run two engines with identical seeds but different insurance programs to isolate the impact of insurance.

```python
# Uninsured scenario: no coverage
no_insurance = InsuranceProgram(deductible=999_999_999)  # effectively uninsured

engine_uninsured = MonteCarloEngine(
    loss_generator=ManufacturingLossGenerator(seed=42),
    insurance_program=no_insurance,
    manufacturer=WidgetManufacturer(mfg_config),
    config=sim_config
)
results_uninsured = engine_uninsured.run()

# Compare survival
print(f"Insured ruin probability (50yr):   "
      f"{results.ruin_probability.get('50', 0):.2%}")
print(f"Uninsured ruin probability (50yr): "
      f"{results_uninsured.ruin_probability.get('50', 0):.2%}")
print(f"Mean growth (insured):   {np.mean(results.growth_rates):.4f}")
print(f"Mean growth (uninsured): {np.mean(results_uninsured.growth_rates):.4f}")
```

---

## 2. Parallel Processing for Speed

The `MonteCarloEngine` handles parallelism internally through the `ParallelExecutor`, but you can also configure the executor directly for fine-grained control over resource usage.

### How Parallelism Works

When `parallel=True` in `MonteCarloConfig`, the engine distributes simulation paths across worker processes. Each worker receives a chunk of simulation IDs, runs them independently, and returns results for aggregation. Shared read-only data (manufacturer config, insurance program) is passed to workers efficiently.

```python
from ergodic_insurance.parallel_executor import ParallelExecutor

# Create a standalone executor for inspection
executor = ParallelExecutor(
    n_workers=8,                  # Auto-detect if None
    monitor_performance=True      # Track timing breakdowns
)

print(f"Workers: {executor.n_workers}")
print(f"CPU cores detected: {executor.cpu_profile.n_cores}")
print(f"Available memory: {executor.cpu_profile.available_memory / 1e9:.1f} GB")
```

### Tuning for Your Hardware

The key parameters live in `MonteCarloConfig`:

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `n_workers` | `None` (auto) | Set to physical cores minus 1 for best throughput |
| `chunk_size` | `10_000` | Smaller chunks = more overhead but better load balancing |
| `use_enhanced_parallel` | `True` | Uses the optimized `ParallelExecutor` path |
| `adaptive_chunking` | `True` | Automatically adjusts chunk sizes based on workload |
| `shared_memory` | `True` | Shares read-only data across workers to reduce copying |
| `use_float32` | `False` | Set `True` to halve memory usage at slight precision cost |

```python
# Configuration for a 4-core laptop running 10K simulations
laptop_config = MonteCarloConfig(
    n_simulations=10_000,
    n_years=30,
    parallel=True,
    n_workers=3,              # Leave one core for the OS
    chunk_size=500,           # Moderate chunks for 4 cores
    use_float32=True,         # Save memory on limited hardware
    adaptive_chunking=True,   # Let the engine optimize chunk sizes
    shared_memory=True,       # Share config data across workers
    progress_bar=True
)

# Configuration for a 16-core workstation running 100K simulations
workstation_config = MonteCarloConfig(
    n_simulations=100_000,
    n_years=50,
    parallel=True,
    n_workers=14,             # Leave 2 cores free
    chunk_size=2_000,         # Larger chunks reduce IPC overhead
    use_float32=False,        # Full precision
    adaptive_chunking=True,
    shared_memory=True,
    progress_bar=True,
    seed=42
)
```

### Monitoring Performance

When `monitor_performance=True` (the default), the engine tracks detailed timing breakdowns. Access them through the results object.

```python
engine = MonteCarloEngine(
    loss_generator=loss_gen,
    insurance_program=program,
    manufacturer=WidgetManufacturer(mfg_config),
    config=MonteCarloConfig(
        n_simulations=5_000,
        n_years=20,
        parallel=True,
        monitor_performance=True
    )
)

results = engine.run()

if results.performance_metrics:
    print(results.performance_metrics.summary())
    # Shows: total time, setup time, computation time,
    #        serialization overhead, throughput, speedup factor
```

---

## 3. Configuration Profiles and Presets

When running many scenarios, manually specifying every parameter becomes tedious and error-prone. The `ConfigManager` provides a **three-tier configuration system**: profiles (complete configs), modules (reusable components), and presets (quick-apply templates for market conditions).

### Loading Profiles

```python
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()

# Load the default profile
config = manager.load_profile("default")

# Load a conservative profile (lower growth, higher margins)
conservative = manager.load_profile("conservative")

# Compare what the profiles contain
for name in ["default", "conservative"]:
    cfg = manager.load_profile(name)
    print(f"\n{name.upper()} Profile:")
    print(f"  Growth rate: {cfg.growth.annual_growth_rate:.1%}")
    print(f"  Volatility: {cfg.growth.volatility:.1%}")
    print(f"  Operating margin: {cfg.manufacturer.base_operating_margin:.1%}")
```

### Applying Market Condition Presets

Presets modify a base profile to reflect specific conditions. This is the fastest way to model market changes.

```python
# Hard market: premiums increase, capacity tightens
hard_market_config = manager.load_profile("default", presets=["hard_market"])

# Soft market: premiums decrease, capacity is abundant
soft_market_config = manager.load_profile("default", presets=["soft_market"])

# Recession: combine hard market with high volatility
recession_config = manager.load_profile(
    "conservative",
    presets=["hard_market", "high_volatility"]
)
```

Presets stack: applying `["hard_market", "high_volatility"]` first applies the hard market adjustments, then overlays high volatility settings. The order can matter when presets modify the same parameters.

---

## 4. Insurance Market Conditions

Insurance markets cycle between soft markets (cheap, abundant coverage) and hard markets (expensive, restricted coverage). The `InsurancePricer` lets you price NovaTech's program under different market conditions using the `MarketCycle` enum.

### MarketCycle Enum

`MarketCycle` defines three states, each corresponding to a target loss ratio that insurers use for pricing:

| Market State | Loss Ratio | Effect on Premiums |
|-------------|-----------|-------------------|
| `MarketCycle.SOFT` | 80% | Lower premiums (buyer's market) |
| `MarketCycle.NORMAL` | 70% | Standard premiums |
| `MarketCycle.HARD` | 60% | Higher premiums (seller's market) |

### Pricing Under Different Markets

```python
from ergodic_insurance.insurance_pricing import InsurancePricer, MarketCycle

# Create pricers for each market condition
soft_pricer = InsurancePricer(
    loss_generator=loss_gen,
    market_cycle=MarketCycle.SOFT
)

normal_pricer = InsurancePricer(
    loss_generator=loss_gen,
    market_cycle=MarketCycle.NORMAL
)

hard_pricer = InsurancePricer(
    loss_generator=loss_gen,
    market_cycle=MarketCycle.HARD
)

# Price NovaTech's program under each condition
for label, pricer in [("Soft", soft_pricer), ("Normal", normal_pricer), ("Hard", hard_pricer)]:
    priced_program = pricer.price_insurance_program(program, expected_revenue=25_000_000)
    total_premium = priced_program.calculate_annual_premium()
    print(f"{label:>6} market premium: ${total_premium:>12,.0f}")
```

### Running Monte Carlo Across Market Conditions

To see how market conditions affect long-term outcomes, run separate Monte Carlo engines with programs priced under each market state.

```python
market_results = {}

for cycle_name, cycle in [("soft", MarketCycle.SOFT),
                           ("normal", MarketCycle.NORMAL),
                           ("hard", MarketCycle.HARD)]:
    # Price the program for this market
    pricer = InsurancePricer(loss_generator=loss_gen, market_cycle=cycle)
    priced_program = pricer.price_insurance_program(
        program, expected_revenue=25_000_000
    )

    # Run Monte Carlo
    engine = MonteCarloEngine(
        loss_generator=ManufacturingLossGenerator(seed=42),
        insurance_program=priced_program,
        manufacturer=WidgetManufacturer(mfg_config),
        config=MonteCarloConfig(n_simulations=1_000, n_years=30, seed=42)
    )
    market_results[cycle_name] = engine.run()

# Compare survival across markets
print(f"{'Market':<10} {'Ruin Prob (30yr)':>18} {'Mean Growth':>14}")
print("-" * 44)
for name, res in market_results.items():
    ruin = res.ruin_probability.get('30', res.ruin_probability.get(
        str(max(int(k) for k in res.ruin_probability.keys())), 0))
    growth = np.mean(res.growth_rates)
    print(f"{name:<10} {ruin:>17.2%} {growth:>13.4f}")
```

---

## 5. Complex Insurance Structures

Real-world insurance programs are more sophisticated than the basic two-layer tower in the earlier tutorials. `EnhancedInsuranceLayer` supports reinstatements, participation rates, aggregate limits, and hybrid limit types.

### Reinstatements

When a layer's limit is exhausted by a loss, **reinstatements** restore the coverage for the remainder of the policy year. The insured typically pays an additional premium (the reinstatement premium) for this restoration.

```python
from ergodic_insurance.insurance_program import (
    InsuranceProgram, EnhancedInsuranceLayer, ReinstatementType
)

# NovaTech's enhanced program for the 5-year plan
# Primary layer: $2.5M limit with 2 reinstatements
# Pro-rata reinstatement means the premium is proportional
# to the fraction of the limit consumed.
primary = EnhancedInsuranceLayer(
    attachment_point=500_000,
    limit=2_500_000,
    base_premium_rate=0.04,
    reinstatements=2,
    reinstatement_premium=1.0,        # 100% of original premium per reinstatement
    reinstatement_type=ReinstatementType.PRO_RATA
)

# First excess: 75% participation (co-insurance with another carrier)
first_excess = EnhancedInsuranceLayer(
    attachment_point=3_000_000,
    limit=5_000_000,
    base_premium_rate=0.02,
    participation_rate=0.75           # Insurer covers 75%, NovaTech retains 25%
)

# High excess layer with aggregate limit
# This layer only pays up to $15M total per year across ALL claims
high_excess = EnhancedInsuranceLayer(
    attachment_point=8_000_000,
    limit=10_000_000,
    base_premium_rate=0.008,
    limit_type="aggregate",
    aggregate_limit=15_000_000
)

enhanced_program = InsuranceProgram(
    layers=[primary, first_excess, high_excess],
    deductible=500_000,
)

# Display the tower
print("=== NovaTech Enhanced Insurance Tower ===")
print(f"Deductible (SIR): ${enhanced_program.deductible:,.0f}\n")

for i, layer in enumerate(enhanced_program.layers, 1):
    print(f"Layer {i}:")
    print(f"  Attachment:      ${layer.attachment_point:,.0f}")
    print(f"  Limit:           ${layer.limit:,.0f}")
    print(f"  Premium Rate:    {layer.base_premium_rate:.2%}")
    if layer.reinstatements > 0:
        print(f"  Reinstatements:  {layer.reinstatements} "
              f"({layer.reinstatement_type.value})")
    if layer.participation_rate < 1.0:
        print(f"  Participation:   {layer.participation_rate:.0%}")
    if layer.limit_type != "per-occurrence":
        print(f"  Limit Type:      {layer.limit_type}")
        if layer.aggregate_limit:
            print(f"  Aggregate Limit: ${layer.aggregate_limit:,.0f}")
    print()

print(f"Total Premium: ${enhanced_program.calculate_annual_premium():,.0f}")
print(f"Total Coverage: ${enhanced_program.get_total_coverage():,.0f}")
```

### How Reinstatements Affect Tail Risk

Reinstatements matter most in years with multiple large losses. Without reinstatements, NovaTech's primary layer pays out once and is exhausted. With two reinstatements, it can respond to up to three large events in a single year.

```python
# Compare programs with and without reinstatements
no_reinstatement_program = InsuranceProgram(
    layers=[
        EnhancedInsuranceLayer(
            attachment_point=500_000,
            limit=2_500_000,
            base_premium_rate=0.04,
            reinstatements=0  # No reinstatements
        ),
    ],
    deductible=500_000,
)

for label, prog in [("With reinstatements", enhanced_program),
                     ("Without reinstatements", no_reinstatement_program)]:
    engine = MonteCarloEngine(
        loss_generator=ManufacturingLossGenerator(seed=42),
        insurance_program=prog,
        manufacturer=WidgetManufacturer(mfg_config),
        config=MonteCarloConfig(n_simulations=1_000, n_years=30, seed=42)
    )
    res = engine.run()
    print(f"{label}: Mean retained loss = "
          f"${np.mean(res.retained_losses):,.0f}, "
          f"Mean recovery = ${np.mean(res.insurance_recoveries):,.0f}")
```

---

## 6. Stochastic Processes for Revenue Modeling

In earlier tutorials, revenue grew deterministically. In reality, NovaTech's revenue fluctuates year to year. The `stochastic_processes` module provides models for adding realistic randomness to growth.

### Geometric Brownian Motion (GBM)

GBM is the standard model for asset prices and revenue with constant relative volatility. The `GeometricBrownianMotion` class generates multiplicative shocks: each year, revenue is multiplied by a random factor drawn from a lognormal distribution.

```python
from ergodic_insurance.stochastic_processes import (
    GeometricBrownianMotion, MeanRevertingProcess, StochasticConfig
)

# GBM for NovaTech's revenue: 8% drift, 15% annual volatility
gbm_config = StochasticConfig(
    volatility=0.15,      # 15% annual volatility
    drift=0.08,           # 8% expected annual growth
    random_seed=42,
    time_step=1.0         # Annual time steps
)
gbm = GeometricBrownianMotion(config=gbm_config)

# Simulate 10 years of revenue shocks
revenue = 25_000_000.0
print(f"Year  0: Revenue = ${revenue:,.0f}")
for year in range(1, 11):
    shock = gbm.generate_shock(current_value=revenue)
    revenue *= shock
    print(f"Year {year:2d}: Revenue = ${revenue:,.0f}  (shock = {shock:.3f})")
```

### Mean-Reverting Process (Ornstein-Uhlenbeck)

For variables that tend to return to a long-run level -- like operating margins or capacity utilization -- use `MeanRevertingProcess`. This prevents unrealistic drift away from fundamentals.

```python
# Mean-reverting operating margin: reverts to 1.0 (100% of base margin)
mr_config = StochasticConfig(
    volatility=0.10,
    drift=0.0,
    random_seed=42,
    time_step=1.0
)
mr_process = MeanRevertingProcess(
    config=mr_config,
    mean_level=1.0,           # Long-run mean multiplier
    reversion_speed=0.5       # How fast it reverts (0=never, 1=instant)
)

# Simulate margin shocks over 10 years
margin_multiplier = 1.0
for year in range(1, 11):
    shock = mr_process.generate_shock(current_value=margin_multiplier)
    margin_multiplier *= shock
    effective_margin = 0.08 * margin_multiplier
    print(f"Year {year:2d}: Margin multiplier = {margin_multiplier:.3f} "
          f"-> Effective margin = {effective_margin:.2%}")
```

### Attaching Stochastic Processes to the Manufacturer

To use stochastic processes inside Monte Carlo simulations, attach them to the `WidgetManufacturer` and enable `apply_stochastic` in the simulation config.

```python
# Create manufacturer with stochastic revenue
stochastic_manufacturer = WidgetManufacturer(mfg_config)
stochastic_manufacturer.stochastic_process = gbm

# Enable stochastic shocks in simulation
stochastic_sim_config = MonteCarloConfig(
    n_simulations=1_000,
    n_years=30,
    apply_stochastic=True,   # Activate stochastic shocks each step
    seed=42,
    parallel=True
)

engine = MonteCarloEngine(
    loss_generator=ManufacturingLossGenerator(seed=42),
    insurance_program=program,
    manufacturer=stochastic_manufacturer,
    config=stochastic_sim_config
)
results_stochastic = engine.run()

print(f"With stochastic revenue: mean growth = "
      f"{np.mean(results_stochastic.growth_rates):.4f}")
```

---

## 7. Scenario Analysis

NovaTech needs to present four scenarios to the board: base case, recession, expansion, and catastrophe. The `ScenarioManager` provides a structured framework for creating, organizing, and comparing scenarios.

### Defining Scenarios

```python
from ergodic_insurance.scenario_manager import ScenarioManager, ScenarioConfig
from ergodic_insurance.monte_carlo import MonteCarloConfig

# Create the scenario manager
scenario_mgr = ScenarioManager()

# Define each scenario with parameter overrides
scenarios = {
    "base_case": {
        "description": "Current economic conditions persist",
        "overrides": {},  # No changes from base config
        "tags": {"baseline", "board-presentation"}
    },
    "recession": {
        "description": "Economic downturn with increased loss frequency",
        "overrides": {
            "growth.annual_growth_rate": -0.02,
            "growth.volatility": 0.25,
        },
        "tags": {"stress-test", "board-presentation"}
    },
    "expansion": {
        "description": "Rapid growth with favorable conditions",
        "overrides": {
            "growth.annual_growth_rate": 0.12,
            "growth.volatility": 0.12,
        },
        "tags": {"optimistic", "board-presentation"}
    },
    "catastrophe": {
        "description": "Major loss event with market disruption",
        "overrides": {
            "growth.volatility": 0.30,
        },
        "tags": {"stress-test", "board-presentation"}
    }
}

for name, spec in scenarios.items():
    scenario_mgr.create_scenario(
        name=name,
        simulation_config=MonteCarloConfig(
            n_simulations=1_000,
            n_years=30,
            seed=42
        ),
        parameter_overrides=spec["overrides"],
        description=spec["description"],
        tags=spec["tags"]
    )

print(f"Created {len(scenario_mgr.scenarios)} scenarios")
for sc in scenario_mgr.scenarios:
    print(f"  - {sc.name}: {sc.description}")
```

### Running Scenarios with Monte Carlo

Each scenario modifies the base simulation. Here we run them and collect results.

```python
# Run each scenario through the Monte Carlo engine
scenario_results = {}

for sc in scenario_mgr.scenarios:
    print(f"\nRunning scenario: {sc.name}...")

    # Apply scenario-specific overrides to create varied conditions
    # In practice, you would apply sc.parameter_overrides to the
    # config objects. Here we show a direct approach:
    engine = MonteCarloEngine(
        loss_generator=ManufacturingLossGenerator(seed=42),
        insurance_program=enhanced_program,
        manufacturer=WidgetManufacturer(mfg_config),
        config=sc.simulation_config
    )
    scenario_results[sc.name] = engine.run()

# Compare all scenarios
print("\n=== NovaTech 5-Year Strategic Plan: Scenario Comparison ===")
print(f"{'Scenario':<15} {'Ruin Prob':>10} {'Mean Growth':>12} "
      f"{'Mean Final Assets':>20}")
print("-" * 60)

for name, res in scenario_results.items():
    max_year = str(max(int(k) for k in res.ruin_probability.keys()))
    ruin = res.ruin_probability.get(max_year, 0)
    growth = np.mean(res.growth_rates)
    mean_assets = np.mean(res.final_assets)
    print(f"{name:<15} {ruin:>9.2%} {growth:>11.4f} ${mean_assets:>18,.0f}")
```

### Grid Search for Optimal Deductible

The `ScenarioManager` also supports systematic parameter sweeps. Use `create_grid_search` to find the optimal deductible across a range of values.

```python
from ergodic_insurance.scenario_manager import ParameterSpec

# Search over deductibles from $100K to $1M
deductible_search = scenario_mgr.create_grid_search(
    name_template="deductible_{params}",
    parameter_specs=[
        ParameterSpec(
            name="insurance.deductible",
            values=[100_000, 250_000, 500_000, 750_000, 1_000_000]
        )
    ],
    simulation_config=MonteCarloConfig(
        n_simulations=500,
        n_years=20,
        seed=42
    ),
    tags={"deductible-optimization"}
)

print(f"\nGenerated {len(deductible_search)} deductible scenarios")
for sc in deductible_search:
    print(f"  {sc.name}: overrides = {sc.parameter_overrides}")
```

### Common Random Numbers for Fair Comparison

When comparing scenarios, you want differences in results to come from the scenario parameters, not from random variation. The `crn_base_seed` option in `MonteCarloConfig` ensures that each `(simulation_id, year)` combination uses the same underlying random draws across scenarios.

```python
# Enable Common Random Numbers for precise comparison
crn_config = MonteCarloConfig(
    n_simulations=1_000,
    n_years=30,
    seed=42,
    crn_base_seed=12345  # Same random draws across scenarios
)

# Now two runs with different deductibles will experience
# the same underlying loss events, isolating the effect
# of the deductible change.
```

---

## 8. Generating Reports

After running scenarios, NovaTech needs professional reports for the board. The `ExcelReporter` generates formatted Excel workbooks with financial statements, metrics dashboards, and Monte Carlo summaries.

### Single-Trajectory Reports

For a detailed look at one simulation path:

```python
from ergodic_insurance.excel_reporter import ExcelReporter, ExcelReportConfig

# Configure the reporter
report_config = ExcelReportConfig(
    include_balance_sheet=True,
    include_income_statement=True,
    include_cash_flow=True,
    include_metrics_dashboard=True
)
reporter = ExcelReporter(config=report_config)

# Generate a trajectory report for NovaTech
output_path = reporter.generate_trajectory_report(
    manufacturer=manufacturer,
    output_file='novatech_trajectory.xlsx',
    title='NovaTech Plastics - Single Path Analysis'
)
print(f"Trajectory report saved to: {output_path}")
```

### Monte Carlo Summary Reports

For aggregate results across thousands of simulations:

```python
# Generate Monte Carlo report from scenario results
mc_output = reporter.generate_monte_carlo_report(
    results=results,
    output_file='novatech_monte_carlo.xlsx',
    title='NovaTech Plastics - Monte Carlo Analysis (1,000 paths)'
)
print(f"Monte Carlo report saved to: {mc_output}")
```

### Formatted Summary Report in Console

If you need a quick text summary without generating an Excel file, enable the summary report in the simulation config:

```python
summary_config = MonteCarloConfig(
    n_simulations=1_000,
    n_years=30,
    generate_summary_report=True,
    summary_report_format="markdown",
    seed=42
)

engine = MonteCarloEngine(
    loss_generator=loss_gen,
    insurance_program=program,
    manufacturer=WidgetManufacturer(mfg_config),
    config=summary_config
)
summary_results = engine.run()

# Print the auto-generated summary
if summary_results.summary_report:
    print(summary_results.summary_report)
```

---

## 9. Performance Tips

As you scale to 10,000+ simulations or 50+ year horizons, runtime and memory become real constraints. Here are practical tips.

### Memory Management

| Technique | Savings | When to Use |
|-----------|---------|-------------|
| `use_float32=True` | ~50% RAM | When precision beyond 7 digits is unnecessary |
| `enable_ledger_pruning=True` | Significant | Long horizons (50+ years) |
| `enable_trajectory_storage=True` | Disk swap | When you need full paths but have limited RAM |
| Reduce `n_simulations` | Linear | For quick iteration during development |

```python
# Memory-optimized config for 100K simulations on 16GB RAM
memory_config = MonteCarloConfig(
    n_simulations=100_000,
    n_years=50,
    use_float32=True,
    enable_ledger_pruning=True,
    parallel=True,
    n_workers=6,
    chunk_size=5_000,
    progress_bar=True
)
```

### Caching Results

Set `cache_results=True` (the default) to avoid re-running identical simulations. The engine hashes the configuration and components to generate a cache key.

```python
cached_config = MonteCarloConfig(
    n_simulations=10_000,
    n_years=30,
    cache_results=True,    # Default: True
    seed=42
)

# First run: full computation
engine = MonteCarloEngine(
    loss_generator=loss_gen,
    insurance_program=program,
    manufacturer=WidgetManufacturer(mfg_config),
    config=cached_config
)
results1 = engine.run()  # Takes N seconds

# Second run with same config: loads from cache
results2 = engine.run()  # Near-instant
```

### Benchmarking Your Setup

Use `BenchmarkSuite` to measure your system's throughput and identify bottlenecks.

```python
from ergodic_insurance.internals import BenchmarkSuite, BenchmarkConfig

benchmark_config = BenchmarkConfig(
    scales=[1_000, 5_000, 10_000],  # Test at these simulation counts
    n_years=10,
    n_workers=4,
    repetitions=3,                   # Run each scale 3 times
    warmup_runs=2                    # Discard first 2 runs
)

suite = BenchmarkSuite()

# Benchmark at each scale
for scale in benchmark_config.scales:
    result = suite.benchmark_scale(
        engine=engine,
        scale=scale,
        config=benchmark_config
    )
    print(f"Scale {scale:>6,}: {result.mean_time:.2f}s "
          f"({result.simulations_per_second:.0f} sims/s)")
```

### Bootstrap Confidence Intervals

For reporting, you often need confidence intervals around your key metrics. Enable bootstrap CI computation in the simulation config.

```python
ci_config = MonteCarloConfig(
    n_simulations=5_000,
    n_years=30,
    compute_bootstrap_ci=True,
    bootstrap_confidence_level=0.95,
    bootstrap_n_iterations=10_000,
    seed=42
)

engine = MonteCarloEngine(
    loss_generator=loss_gen,
    insurance_program=program,
    manufacturer=WidgetManufacturer(mfg_config),
    config=ci_config
)
ci_results = engine.run()

if ci_results.bootstrap_confidence_intervals:
    print("\n95% Confidence Intervals:")
    for metric, (lower, upper) in ci_results.bootstrap_confidence_intervals.items():
        print(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
```

---

## 10. Exercises

These exercises build on the NovaTech scenario from this tutorial. Each one is self-contained, and you should be able to complete it using only the APIs covered above.

### Exercise 1: Parallel Monte Carlo Comparison

Run **1,000 Monte Carlo simulations** comparing NovaTech's current two-layer program (from Section 1) against the enhanced three-layer program with reinstatements (from Section 5). Use parallel execution with at least 4 workers.

**Tasks:**
1. Create both insurance programs.
2. Configure `MonteCarloConfig` with `parallel=True`, `n_workers=4`, and `n_simulations=1_000`.
3. Run both through `MonteCarloEngine` with identical seeds.
4. Compare: ruin probability, mean final assets, and mean growth rate.
5. Print a formatted comparison table.

**Expected insight:** The enhanced program with reinstatements should show lower ruin probability despite higher total premium.

---

### Exercise 2: Market Condition Stress Test

Create and compare simulations for **three market conditions** (soft, normal, hard) and report survival rates and growth.

**Tasks:**
1. Use `InsurancePricer` with `MarketCycle.SOFT`, `MarketCycle.NORMAL`, and `MarketCycle.HARD` to price NovaTech's program.
2. Run 1,000 Monte Carlo simulations for each priced program over 30 years.
3. For each condition, report: total annual premium, ruin probability at year 30, mean growth rate.
4. Answer: In which market condition does insurance add the most value relative to going uninsured?

**Hint:** Use the same `seed` across all three runs to ensure the underlying loss experience is comparable. Better yet, use `crn_base_seed` for Common Random Numbers.

---

### Exercise 3: Complex Tower with Catastrophe Stress Test

Build a four-layer insurance tower with reinstatements and aggregate limits, then stress-test it under a catastrophe scenario.

**Tasks:**
1. Create an `InsuranceProgram` with:
   - $500K deductible
   - Layer 1: $500K xs $500K, rate 5%, 2 reinstatements (pro-rata)
   - Layer 2: $2.5M xs $1M, rate 3%, 1 reinstatement (full)
   - Layer 3: $5M xs $3.5M, rate 1.5%, 80% participation
   - Layer 4: $10M xs $8.5M, rate 0.8%, aggregate limit of $12M
2. Run 1,000 Monte Carlo simulations over 50 years under normal conditions.
3. Run 1,000 simulations under catastrophe conditions (use `ConfigManager` with `presets=["high_volatility"]`).
4. Compare: ruin probability at years 10, 20, 30, and 50 for both conditions.
5. Identify which layer is most critical for survival in the catastrophe scenario.

**Hint:** Use `ruin_evaluation=[10, 20, 30, 50]` in `MonteCarloConfig` to get ruin probabilities at specific horizons.

---

## 11. Further Resources

- **API Reference**: See the reference link on bottom left and also module docstrings in the source code for detailed parameter documentation.
- **Notebooks**: Interactive examples in `ergodic_insurance/notebooks/` demonstrate end-to-end workflows.
- **Architecture Docs**: `ergodic_insurance/docs/architecture/monte_carlo_architecture.md` explains the parallel execution pipeline in detail.
- **Configuration Reference**: `ergodic_insurance/configs/` contains YAML profile and preset definitions.
- **Tutorials 01-05**: Review earlier tutorials for foundational concepts (business model, loss generation, basic insurance, optimization, result interpretation).

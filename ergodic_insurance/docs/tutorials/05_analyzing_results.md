# Analyzing Results: The Ergodic Advantage

This tutorial covers how to analyze simulation results through the lens of ergodic economics, comparing time-average and ensemble-average metrics to reveal the true value of insurance. By the end, you will be able to run paired simulations, use `ErgodicAnalyzer.compare_scenarios()`, interpret divergence metrics, and build visualizations that communicate the ergodic advantage to stakeholders.

> **Prerequisites**: You should be comfortable running simulations (Tutorial 2) and configuring insurance (Tutorial 3). This tutorial assumes you already have insured and uninsured simulation pipelines ready. If not, review those tutorials first.

---

## The Board Presentation Problem

NovaTech Plastics has a familiar challenge. Their risk manager, Dana, needs to present to the board next Thursday. The board has a simple question:

> *"We only have one major loss every five years on average. Why are we paying $200K per year in insurance premiums? That is a million dollars every five years for a loss that averages $800K. We are literally overpaying."*

The board is thinking in **ensemble averages** -- expected values across hypothetical parallel universes. Dana needs to show them what happens to *one company over time*. That is the **time-average** perspective, and it tells a completely different story.

This tutorial walks through the analysis Dana will use to build her case.

---

## Ergodic vs Non-Ergodic Systems

Before diving into code, let us establish the core concept.

In an **ergodic system**, the time average equals the ensemble average. The average experience of one entity over many time periods is the same as the average across many entities at a single point in time. Rolling a fair die is ergodic: your long-run average converges to the same value as the group average.

Business growth with catastrophic losses is **non-ergodic**. Here is why:

- **Ensemble average (expected value)**: Take 1,000 companies, average their outcomes at year 30. Some went bankrupt, some grew enormously. The average looks fine -- maybe 6% annual growth. The board uses this number.
- **Time average**: Follow one company for 30 years. A single catastrophic loss can permanently impair its growth trajectory. Compound effects of losses accumulate multiplicatively. The typical outcome is substantially worse than the expected value.

The mathematical reason is **Jensen's inequality** applied to the logarithmic growth function. For a concave function like log(wealth), the expected value of the function is less than the function of the expected value:

```
E[log(X)] <= log(E[X])
```

This gap is the **ergodic divergence**, and insurance closes it by reducing the variance of the multiplicative process. That is Dana's core argument.

---

## Running Paired Simulations

To make a fair comparison, we run insured and uninsured simulations with **the same loss seeds**. This ensures both scenarios face identical loss events; the only difference is whether insurance responds.

We assume you have your manufacturer and loss generator configured (see Tutorials 1-3). Here we set up the paired runs:

```python
from ergodic_insurance import (
    ManufacturerConfig, WidgetManufacturer, ManufacturingLossGenerator,
    InsuranceProgram, Simulation,
)

# NovaTech's financial profile
config = ManufacturerConfig(
    initial_assets=10_000_000,      # $10M starting equity
    asset_turnover_ratio=1.0,       # Revenue = 1x assets
    base_operating_margin=0.08,     # 8% operating margin
    tax_rate=0.25,                  # 25% tax rate
    retention_ratio=1.0             # Retain all earnings
)

# NovaTech's proposed insurance program
policy = InsuranceProgram.simple(
    deductible=100_000,
    limit=5_000_000,
    rate=0.02
)

# Run paired simulations
n_sims = 100
time_horizon = 30

insured_results = []
uninsured_results = []

for seed in range(n_sims):
    # --- Insured scenario ---
    mfg_insured = WidgetManufacturer(config)
    loss_gen_insured = ManufacturingLossGenerator.create_simple(
        frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=seed
    )
    sim_insured = Simulation(
        manufacturer=mfg_insured,
        loss_generator=loss_gen_insured,
        insurance_program=policy,
        time_horizon=time_horizon,
        seed=seed
    )
    insured_results.append(sim_insured.run())

    # --- Uninsured scenario (same seed = same losses) ---
    mfg_uninsured = WidgetManufacturer(config)
    loss_gen_uninsured = ManufacturingLossGenerator.create_simple(
        frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=seed
    )
    sim_uninsured = Simulation(
        manufacturer=mfg_uninsured,
        loss_generator=loss_gen_uninsured,
        # No insurance_program here
        time_horizon=time_horizon,
        seed=seed
    )
    uninsured_results.append(sim_uninsured.run())
```

The key design decision: **same seed for both runs**. This creates paired observations where the only variable is insurance. Without pairing, random variation in loss timing would obscure the true insurance effect.

---

## Using ErgodicAnalyzer.compare_scenarios()

The `ErgodicAnalyzer` is the core analysis tool. Its `compare_scenarios()` method accepts either lists of `SimulationResults` objects or raw numpy arrays and returns a comprehensive comparison dictionary.

```python
from ergodic_insurance import ErgodicAnalyzer

analyzer = ErgodicAnalyzer(convergence_threshold=0.01)

# Run the comparison
comparison = analyzer.compare_scenarios(
    insured_results=insured_results,
    uninsured_results=uninsured_results,
    metric="equity"
)
```

The `metric` parameter specifies which financial time series to analyze. The default `"equity"` tracks shareholder equity over time, which is the most meaningful metric for growth analysis. You can also use `"assets"` or `"cash"`.

### Reading the Comparison Output

The returned dictionary has three top-level keys: `'insured'`, `'uninsured'`, and `'ergodic_advantage'`.

```python
# ---- Insured scenario metrics ----
insured = comparison['insured']
print("=== Insured Scenario ===")
print(f"  Time-average growth (mean):   {insured['time_average_mean']:.2%}")
print(f"  Time-average growth (median): {insured['time_average_median']:.2%}")
print(f"  Time-average growth (std):    {insured['time_average_std']:.2%}")
print(f"  Ensemble-average growth:      {insured['ensemble_average']:.2%}")
print(f"  Survival rate:                {insured['survival_rate']:.1%}")
print(f"  Survivors:                    {insured['n_survived']} / {n_sims}")

# ---- Uninsured scenario metrics ----
uninsured = comparison['uninsured']
print("\n=== Uninsured Scenario ===")
print(f"  Time-average growth (mean):   {uninsured['time_average_mean']:.2%}")
print(f"  Time-average growth (median): {uninsured['time_average_median']:.2%}")
print(f"  Time-average growth (std):    {uninsured['time_average_std']:.2%}")
print(f"  Ensemble-average growth:      {uninsured['ensemble_average']:.2%}")
print(f"  Survival rate:                {uninsured['survival_rate']:.1%}")
print(f"  Survivors:                    {uninsured['n_survived']} / {n_sims}")

# ---- Ergodic advantage ----
advantage = comparison['ergodic_advantage']
print("\n=== Ergodic Advantage (Insured vs Uninsured) ===")
print(f"  Time-average gain:     {advantage['time_average_gain']:.2%}")
print(f"  Ensemble-average gain: {advantage['ensemble_average_gain']:.2%}")
print(f"  Survival improvement:  {advantage['survival_gain']:.1%}")
print(f"  t-statistic:           {advantage['t_statistic']:.3f}")
print(f"  p-value:               {advantage['p_value']:.4f}")
print(f"  Statistically significant: {advantage['significant']}")
```

Notice the structure: each scenario gets `time_average_mean`, `time_average_median`, `time_average_std`, `ensemble_average`, `survival_rate`, and `n_survived`. The `ergodic_advantage` section computes the differences and includes a t-test for statistical significance.

---

## Time-Average vs Ensemble-Average Growth

This is the heart of Dana's argument to the board. Let us walk through a concrete numerical example.

Suppose 100 simulations over 30 years produce these results:

| Metric | Insured | Uninsured |
|--------|---------|-----------|
| **Ensemble-average growth** | 5.8% | 6.5% |
| **Time-average growth** | 5.5% | 2.1% |
| **Survival rate** | 97% | 68% |

The board's traditional analysis focuses on the first row: the uninsured ensemble average is *higher*. "See? We grow faster without insurance."

But the time-average row tells the truth. **The typical company** (the one NovaTech actually is) grows at only 2.1% without insurance, compared to 5.5% with it. The ensemble average is inflated by a few "lucky" paths that never experienced a catastrophic loss.

### Why Do They Diverge?

The divergence arises because losses act **multiplicatively** on equity:

1. A $5M loss on a $10M company wipes out 50% of equity.
2. To recover, the company must grow 100% (not 50%) to return to $10M.
3. This asymmetry means losses hurt more than gains help, in compound terms.

Insurance caps the downside of the multiplicative process, reducing variance and thereby increasing the **geometric** (time-average) growth rate, even though it reduces the **arithmetic** (ensemble-average) growth rate through premium costs.

```python
# Calculate the ergodic divergence for each scenario
# Divergence = time-average growth - ensemble-average growth
# For non-ergodic systems, this is typically negative

insured_divergence = insured['time_average_mean'] - insured['ensemble_average']
uninsured_divergence = uninsured['time_average_mean'] - uninsured['ensemble_average']

print("=== Ergodic Divergence ===")
print(f"  Insured divergence:   {insured_divergence:.2%}")
print(f"  Uninsured divergence: {uninsured_divergence:.2%}")
print(f"  Divergence reduction: {abs(uninsured_divergence) - abs(insured_divergence):.2%}")

if abs(uninsured_divergence) > abs(insured_divergence):
    print("\n  -> Insurance reduces the ergodic divergence.")
    print("     The insured company's expected value is a better predictor")
    print("     of what it will actually experience over time.")
```

A large negative divergence means the ensemble average is substantially overestimating what a real business will experience. Insurance narrows this gap by making the process "more ergodic."

---

## Understanding Ergodic Divergence Over Time

The divergence between time and ensemble averages typically **grows** with the time horizon. This is critical: the board's short-term thinking ("it is only 5 years of data") underestimates the long-term compounding effect of uninsured losses.

```python
import numpy as np

# Analyze divergence at different time horizons using sub-windows
# of our 30-year simulations
horizons = [5, 10, 15, 20, 25, 30]

print("=== Ergodic Divergence by Time Horizon ===")
print(f"{'Horizon':>8}  {'Insured Div':>12}  {'Uninsured Div':>14}  {'Gap':>8}")
print("-" * 48)

for h in horizons:
    # Truncate equity trajectories to horizon h
    insured_truncated = [r.equity[:h+1] for r in insured_results if len(r.equity) > h]
    uninsured_truncated = [r.equity[:h+1] for r in uninsured_results if len(r.equity) > h]

    # Time-average: geometric mean of individual path growth rates
    def time_avg(trajs, horizon):
        rates = []
        for traj in trajs:
            if traj[horizon] > 0:
                rates.append((traj[horizon] / traj[0]) ** (1.0 / horizon) - 1)
            else:
                rates.append(-1.0)
        return np.mean(rates)

    # Ensemble average: growth of the mean trajectory
    def ensemble_avg(trajs, horizon):
        mean_traj = np.mean([t[:horizon+1] for t in trajs], axis=0)
        return (mean_traj[horizon] / mean_traj[0]) ** (1.0 / horizon) - 1

    ins_ta = time_avg(insured_truncated, h)
    ins_ea = ensemble_avg(insured_truncated, h)
    unins_ta = time_avg(uninsured_truncated, h)
    unins_ea = ensemble_avg(uninsured_truncated, h)

    ins_div = ins_ta - ins_ea
    unins_div = unins_ta - unins_ea

    print(f"{h:>5} yr  {ins_div:>11.2%}  {unins_div:>13.2%}  {abs(unins_div) - abs(ins_div):>7.2%}")
```

You should observe the uninsured divergence growing more negative over longer horizons, while the insured divergence stays relatively contained. This is the compound cost of uninsured volatility.

---

## Survival Analysis

Survival is binary: either the company remains solvent or it does not. The board may not care about growth rate nuances, but they care about staying in business.

```python
# Survival analysis from paired simulations
insured_survived = sum(1 for r in insured_results if r.insolvency_year is None)
uninsured_survived = sum(1 for r in uninsured_results if r.insolvency_year is None)

print("=== Survival Analysis ===")
print(f"  Insured survival rate:   {insured_survived}/{n_sims} ({insured_survived/n_sims:.1%})")
print(f"  Uninsured survival rate: {uninsured_survived}/{n_sims} ({uninsured_survived/n_sims:.1%})")

# Time to ruin for those that went bankrupt
insured_ruin_years = [r.insolvency_year for r in insured_results if r.insolvency_year is not None]
uninsured_ruin_years = [r.insolvency_year for r in uninsured_results if r.insolvency_year is not None]

if insured_ruin_years:
    print(f"\n  Insured ruin (when it happens):")
    print(f"    Mean time to ruin:   {np.mean(insured_ruin_years):.1f} years")
    print(f"    Earliest ruin:       Year {min(insured_ruin_years)}")

if uninsured_ruin_years:
    print(f"\n  Uninsured ruin (when it happens):")
    print(f"    Mean time to ruin:   {np.mean(uninsured_ruin_years):.1f} years")
    print(f"    Earliest ruin:       Year {min(uninsured_ruin_years)}")

# Build survival curves
insured_survival_curve = []
uninsured_survival_curve = []
for year in range(time_horizon + 1):
    ins_alive = sum(1 for r in insured_results
                    if r.insolvency_year is None or r.insolvency_year > year)
    unins_alive = sum(1 for r in uninsured_results
                      if r.insolvency_year is None or r.insolvency_year > year)
    insured_survival_curve.append(ins_alive / n_sims)
    uninsured_survival_curve.append(unins_alive / n_sims)
```

When Dana presents these numbers, the difference is stark. Telling the board "there is a 32% chance we will not exist in 30 years" gets attention in a way that growth rate differentials do not.

---

## Growth Rate Analysis

Beyond the aggregate comparison, examining individual simulation paths reveals the distribution of outcomes.

```python
# Use SimulationResults built-in methods for individual path analysis
# Pick a representative insured and uninsured result
sample_insured = insured_results[0]
sample_uninsured = uninsured_results[0]

# Time-weighted ROE (geometric mean - the "true" compound return)
ins_tw_roe = sample_insured.calculate_time_weighted_roe()
unins_tw_roe = sample_uninsured.calculate_time_weighted_roe()
print(f"Sample path time-weighted ROE:")
print(f"  Insured:   {ins_tw_roe:.2%}")
print(f"  Uninsured: {unins_tw_roe:.2%}")

# Rolling ROE shows how growth rate evolves over time
rolling_insured = sample_insured.calculate_rolling_roe(window=5)
rolling_uninsured = sample_uninsured.calculate_rolling_roe(window=5)

# ROE volatility metrics
vol_insured = sample_insured.calculate_roe_volatility()
vol_uninsured = sample_uninsured.calculate_roe_volatility()
print(f"\nROE Volatility:")
print(f"  Insured std:    {vol_insured['roe_std']:.2%}")
print(f"  Uninsured std:  {vol_uninsured['roe_std']:.2%}")
print(f"  Insured Sharpe: {vol_insured['roe_sharpe']:.2f}")
print(f"  Uninsured Sharpe: {vol_uninsured['roe_sharpe']:.2f}")

# Summary stats aggregate everything into one dictionary
stats_insured = sample_insured.summary_stats()
stats_uninsured = sample_uninsured.summary_stats()
print(f"\nSummary stats (single path):")
print(f"  Insured mean ROE:   {stats_insured['mean_roe']:.2%}")
print(f"  Uninsured mean ROE: {stats_uninsured['mean_roe']:.2%}")
```

### Distribution Across All Paths

For the board presentation, aggregate metrics across all simulation paths to show the full picture:

```python
# Collect time-weighted ROE across all paths
all_insured_roe = [r.calculate_time_weighted_roe() for r in insured_results]
all_uninsured_roe = [r.calculate_time_weighted_roe() for r in uninsured_results]

print("=== Growth Rate Distribution (All Paths) ===")
print(f"{'Percentile':>12}  {'Insured':>10}  {'Uninsured':>10}")
print("-" * 36)
for pct in [10, 25, 50, 75, 90]:
    ins_val = np.percentile(all_insured_roe, pct)
    unins_val = np.percentile(all_uninsured_roe, pct)
    print(f"  {pct:>8}th   {ins_val:>9.2%}   {unins_val:>9.2%}")

print(f"\n  Mean:        {np.mean(all_insured_roe):>9.2%}   {np.mean(all_uninsured_roe):>9.2%}")
print(f"  Std Dev:     {np.std(all_insured_roe):>9.2%}   {np.std(all_uninsured_roe):>9.2%}")
```

Notice how the uninsured distribution has a much fatter left tail. The median and lower percentiles are dramatically worse, even if the top percentiles look better. That asymmetry is what makes insurance valuable from a time-average perspective.

---

## Visualizing Ergodic Effects

Visualizations are essential for Dana's board presentation. Four charts tell the story.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("NovaTech Plastics: Ergodic Insurance Analysis", fontsize=14, fontweight='bold')

# ---- Panel 1: Individual equity trajectories ----
ax1 = axes[0, 0]
for r in insured_results[:20]:  # Plot first 20 paths for clarity
    ax1.plot(r.years, r.equity / 1e6, color='steelblue', alpha=0.3, linewidth=0.8)
for r in uninsured_results[:20]:
    ax1.plot(r.years, r.equity / 1e6, color='indianred', alpha=0.3, linewidth=0.8)
# Legend entries
ax1.plot([], [], color='steelblue', linewidth=2, label='Insured')
ax1.plot([], [], color='indianred', linewidth=2, label='Uninsured')
ax1.set_xlabel('Year')
ax1.set_ylabel('Equity ($M)')
ax1.set_title('Individual Trajectories (20 paths each)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# ---- Panel 2: Ensemble average (the "misleading" view) ----
ax2 = axes[0, 1]
insured_ensemble = np.mean(
    [r.equity[:time_horizon+1] for r in insured_results if len(r.equity) >= time_horizon+1],
    axis=0
)
uninsured_ensemble = np.mean(
    [r.equity[:time_horizon+1] for r in uninsured_results if len(r.equity) >= time_horizon+1],
    axis=0
)
years = range(len(insured_ensemble))
ax2.plot(years, insured_ensemble / 1e6, color='steelblue', linewidth=2.5, label='Insured')
ax2.plot(years, uninsured_ensemble / 1e6, color='indianred', linewidth=2.5, label='Uninsured')
ax2.set_xlabel('Year')
ax2.set_ylabel('Mean Equity ($M)')
ax2.set_title('Ensemble Average (Board\'s View)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.annotate('This chart is misleading!\nIt hides bankruptcies.',
             xy=(0.5, 0.05), xycoords='axes fraction',
             fontsize=9, fontstyle='italic', color='gray',
             ha='center')

# ---- Panel 3: Time-average vs ensemble-average (the key chart) ----
ax3 = axes[1, 0]
insured_data = comparison['insured']
uninsured_data = comparison['uninsured']
categories = ['Insured', 'Uninsured']
time_avgs = [insured_data['time_average_mean'] * 100, uninsured_data['time_average_mean'] * 100]
ensemble_avgs = [insured_data['ensemble_average'] * 100, uninsured_data['ensemble_average'] * 100]

x = np.arange(len(categories))
width = 0.3
bars1 = ax3.bar(x - width/2, time_avgs, width, label='Time Average (reality)',
                color=['steelblue', 'indianred'], edgecolor='black', linewidth=0.5)
bars2 = ax3.bar(x + width/2, ensemble_avgs, width, label='Ensemble Average (expectation)',
                color=['lightsteelblue', 'lightsalmon'], edgecolor='black', linewidth=0.5)
ax3.set_ylabel('Annual Growth Rate (%)')
ax3.set_title('Time Average vs Ensemble Average')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# ---- Panel 4: Survival curves ----
ax4 = axes[1, 1]
years_range = range(time_horizon + 1)
ax4.plot(years_range, [s * 100 for s in insured_survival_curve],
         color='steelblue', linewidth=2.5, label='Insured')
ax4.plot(years_range, [s * 100 for s in uninsured_survival_curve],
         color='indianred', linewidth=2.5, label='Uninsured')
ax4.set_xlabel('Year')
ax4.set_ylabel('Survival Rate (%)')
ax4.set_title('Survival Curves')
ax4.set_ylim(0, 105)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('novatech_ergodic_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Presentation tips for Dana:**
- Panel 1 makes bankruptcy real: you can see the red lines dropping to zero.
- Panel 2 is what the board currently believes. Show it, then explain why it is misleading.
- Panel 3 is the key chart: the gap between the dark bars (reality) and the light bars (expectation) is the ergodic divergence. Insurance shrinks that gap.
- Panel 4 is the closer. "Without insurance, there is a 1-in-3 chance we do not make it to 2055."

---

## Convergence Analysis

Monte Carlo results are only reliable when you have run enough simulations. The `ErgodicAnalyzer` checks convergence based on the `convergence_threshold` you set at initialization.

```python
# Check: do we have enough simulations for reliable results?
# A simple bootstrap approach to estimate standard error

n_bootstrap = 50
bootstrap_advantages = []

for _ in range(n_bootstrap):
    # Resample with replacement
    idx = np.random.choice(n_sims, size=n_sims, replace=True)
    boot_insured = [insured_results[i] for i in idx]
    boot_uninsured = [uninsured_results[i] for i in idx]

    boot_comparison = analyzer.compare_scenarios(
        insured_results=boot_insured,
        uninsured_results=boot_uninsured,
        metric="equity"
    )
    bootstrap_advantages.append(
        boot_comparison['ergodic_advantage']['time_average_gain']
    )

se = np.std(bootstrap_advantages)
mean_advantage = np.mean(bootstrap_advantages)
ci_lower = mean_advantage - 1.96 * se
ci_upper = mean_advantage + 1.96 * se

print("=== Convergence Analysis ===")
print(f"  Time-average gain estimate: {mean_advantage:.2%}")
print(f"  Standard error:             {se:.4%}")
print(f"  95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")

if se < 0.005:
    print("  Status: Converged (SE < 0.5%)")
elif se < 0.01:
    print("  Status: Approximately converged. Consider more simulations for precision.")
else:
    print(f"  Status: NOT converged. Consider increasing n_sims beyond {n_sims}.")
```

**Rule of thumb**: For most NovaTech-scale scenarios, 100 simulations give directionally correct results. For precise estimates (standard error below 0.5%), you typically need 500 or more. For board presentations, 200 paired simulations is usually a good balance of precision and runtime.

---

## Full Analysis Pipeline

Here is the complete end-to-end analysis pipeline that Dana would run before her presentation. This brings together simulation, analysis, and reporting in one script.

```python
import numpy as np
from ergodic_insurance import (
    ManufacturerConfig, WidgetManufacturer, ManufacturingLossGenerator,
    InsuranceProgram, EnhancedInsuranceLayer, Simulation, ErgodicAnalyzer,
)

# ============================================================
# Configuration
# ============================================================
config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0
)

policy = InsuranceProgram(
    layers=[EnhancedInsuranceLayer(attachment_point=100_000, limit=5_000_000, base_premium_rate=0.02)],
    deductible=100_000
)

n_sims = 200
time_horizon = 30

# ============================================================
# Run paired simulations
# ============================================================
insured_results = []
uninsured_results = []

for seed in range(n_sims):
    # Insured
    mfg = WidgetManufacturer(config)
    loss_gen = ManufacturingLossGenerator.create_simple(
        frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=seed
    )
    sim = Simulation(
        manufacturer=mfg, loss_generator=loss_gen,
        insurance_program=policy, time_horizon=time_horizon, seed=seed
    )
    insured_results.append(sim.run())

    # Uninsured
    mfg = WidgetManufacturer(config)
    loss_gen = ManufacturingLossGenerator.create_simple(
        frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=seed
    )
    sim = Simulation(
        manufacturer=mfg, loss_generator=loss_gen,
        time_horizon=time_horizon, seed=seed
    )
    uninsured_results.append(sim.run())

# ============================================================
# Ergodic analysis
# ============================================================
analyzer = ErgodicAnalyzer(convergence_threshold=0.01)
comparison = analyzer.compare_scenarios(
    insured_results=insured_results,
    uninsured_results=uninsured_results,
    metric="equity"
)

# ============================================================
# Generate board report
# ============================================================
ins = comparison['insured']
unins = comparison['uninsured']
adv = comparison['ergodic_advantage']

# Calculate cost-benefit ratio
annual_premium = policy.calculate_annual_premium()
expected_annual_loss = 0.2 * 1_000_000  # frequency x severity mean
premium_loading = (annual_premium / expected_annual_loss - 1) * 100

print("=" * 64)
print("   NOVATECH PLASTICS -- ERGODIC INSURANCE ANALYSIS REPORT")
print("=" * 64)
print(f"  Simulations: {n_sims} paired runs over {time_horizon} years")
print(f"  Annual premium: ${annual_premium:,.0f}")
print(f"  Expected annual loss: ${expected_annual_loss:,.0f}")
print(f"  Premium loading: {premium_loading:.0f}%")
print()

print(f"  {'Metric':<32} {'Insured':>12} {'Uninsured':>12}")
print("  " + "-" * 58)
print(f"  {'Survival Rate':<32} {ins['survival_rate']:>11.1%} {unins['survival_rate']:>11.1%}")
print(f"  {'Time-Average Growth':<32} {ins['time_average_mean']:>11.2%} {unins['time_average_mean']:>11.2%}")
print(f"  {'Ensemble-Average Growth':<32} {ins['ensemble_average']:>11.2%} {unins['ensemble_average']:>11.2%}")

ins_div = ins['time_average_mean'] - ins['ensemble_average']
unins_div = unins['time_average_mean'] - unins['ensemble_average']
print(f"  {'Ergodic Divergence':<32} {ins_div:>11.2%} {unins_div:>11.2%}")
print("  " + "-" * 58)

print(f"\n  Ergodic Advantage (Insured vs Uninsured):")
print(f"    Time-average growth improvement: {adv['time_average_gain']:.2%}")
print(f"    Survival rate improvement:       {adv['survival_gain']:.1%}")
print(f"    Statistically significant:       {'Yes' if adv['significant'] else 'No'}")
print(f"    p-value:                         {adv['p_value']:.4f}")
print("=" * 64)
```

---

## Interpreting Results

Use the following table as a reference when reading `compare_scenarios()` output:

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| **Time-average growth** | The compound annual growth rate experienced by a single business over time. Calculated as the geometric mean of individual path growth rates. | This is what NovaTech will actually experience. It captures the drag from volatility. |
| **Ensemble-average growth** | The expected growth rate across many parallel universes. Calculated from the growth of the mean trajectory. | This is what the board thinks will happen. It is the traditional "expected value" view. |
| **Ergodic divergence** | Time-average minus ensemble-average. Typically negative for volatile processes. | A large negative value means the expected value is dangerously optimistic. Insurance reduces this gap. |
| **Survival rate** | Fraction of simulation paths that avoid bankruptcy over the time horizon. | The ultimate binary outcome. A 30% bankruptcy probability over 30 years is not a tail risk; it is a core strategic threat. |
| **Time-average gain** | Insured time-average growth minus uninsured time-average growth. | The bottom line: how much faster does the insured company actually grow? |
| **t-statistic / p-value** | Statistical significance test for the time-average gain. | If p < 0.05, the difference is unlikely due to chance. Needed for a rigorous board presentation. |

**Key takeaway for Dana's presentation**: If the ergodic divergence is large and negative for the uninsured scenario, the ensemble average is significantly overestimating what NovaTech will actually experience over time. Insurance closes this gap, converting "expected" growth into "experienced" growth. The premium is not a cost, it is the price of making the expected value a reliable prediction.

---

## Exercises

### Exercise 1: Current vs Proposed Insurance Programs

Run 200 paired simulations comparing NovaTech's current insurance program (low limits, high retention) against a proposed enhanced program (higher limits, lower retention). Calculate the ergodic divergence for each. Specifically:

- **Current program**: `InsuranceProgram(layers=[EnhancedInsuranceLayer(500_000, 2_000_000, base_premium_rate=0.015)], deductible=500_000)`
- **Proposed program**: `InsuranceProgram(layers=[EnhancedInsuranceLayer(100_000, 5_000_000, base_premium_rate=0.02)], deductible=100_000)`

Compare the `time_average_gain` and `survival_gain` from `compare_scenarios()` for each program against the uninsured baseline. Which program produces a larger ergodic advantage?

### Exercise 2: Divergence Over Time

Create a visualization showing how the ensemble average diverges from the time average as the simulation horizon increases from 5 to 50 years. Plot two lines on the same chart:
- Ensemble-average growth rate vs. time horizon
- Time-average growth rate vs. time horizon

Do this for both the insured and uninsured scenarios (four lines total). At what horizon does the uninsured divergence become larger than the annual insurance premium in growth-rate terms?

### Exercise 3: Board Summary

Using actual simulation output from 200 runs, draft a 5-point summary for NovaTech's board. Each point should reference a specific metric from `compare_scenarios()`. Structure it as:

1. **Survival**: "Without insurance, our 30-year bankruptcy probability is X%."
2. **Growth reality**: "Our actual growth rate without insurance is X%, not the X% you see in projections."
3. **Insurance ROI**: "For every $1 in premium, we gain $X in expected terminal equity."
4. **Statistical confidence**: "These results are statistically significant (p = X)."
5. **Recommendation**: "Increasing our insurance budget by $X/year improves our compound growth rate by X%."

---

## Next Steps

- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) -- Monte Carlo engines, market cycles, and multi-layer optimization strategies

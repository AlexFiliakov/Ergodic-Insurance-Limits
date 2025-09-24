# Analyzing Results Tutorial

This tutorial teaches you how to analyze simulation results, interpret key metrics, and make data-driven insurance decisions. You'll learn about ergodic vs ensemble averages, risk metrics, and how to communicate findings effectively.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand and calculate key performance metrics
- Compare ergodic vs ensemble averages
- Perform statistical analysis on results
- Create decision-ready visualizations
- Make and justify insurance recommendations

## Understanding Key Metrics

THIS TUTORIAL IS OTUDATED!!!

### Growth Rate Metrics

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from ergodic_insurance.manufacturer import Manufacturer
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.monte_carlo import MonteCarloAnalyzer
from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer

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

# Run simulations
mc_analyzer = MonteCarloAnalyzer(manufacturer, claim_generator)
results = mc_analyzer.run_simulations(
    n_simulations=1000,
    n_years=20,
    retention=1_000_000,
    limit=10_000_000,
    premium_rate=0.02,
    seed=42
)

# Calculate different growth metrics
def analyze_growth_metrics(results):
    """Calculate various growth rate metrics."""

    final_wealths = results['final_wealths']
    initial_wealth = manufacturer.initial_assets
    n_years = 20

    # Arithmetic mean growth
    arithmetic_mean = np.mean((final_wealths / initial_wealth) ** (1/n_years) - 1)

    # Geometric mean growth (time average)
    geometric_mean = (np.prod(final_wealths / initial_wealth) ** (1/len(final_wealths))) ** (1/n_years) - 1

    # Median growth
    median_growth = np.median((final_wealths / initial_wealth) ** (1/n_years) - 1)

    # Growth rate distribution
    growth_rates = (final_wealths / initial_wealth) ** (1/n_years) - 1

    metrics = {
        'arithmetic_mean': arithmetic_mean,
        'geometric_mean': geometric_mean,
        'median': median_growth,
        'std_dev': np.std(growth_rates),
        'skewness': stats.skew(growth_rates),
        'kurtosis': stats.kurtosis(growth_rates)
    }

    return metrics, growth_rates

metrics, growth_rates = analyze_growth_metrics(results)

print("Growth Rate Analysis:")
print("-" * 40)
print(f"Arithmetic Mean: {metrics['arithmetic_mean']:.2%}")
print(f"Geometric Mean:  {metrics['geometric_mean']:.2%}")
print(f"Median:          {metrics['median']:.2%}")
print(f"Std Deviation:   {metrics['std_dev']:.2%}")
print(f"Skewness:        {metrics['skewness']:.2f}")
print(f"Kurtosis:        {metrics['kurtosis']:.2f}")

# Visualize growth distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
ax1 = axes[0]
ax1.hist(growth_rates * 100, bins=50, alpha=0.7, edgecolor='black')
ax1.axvline(metrics['arithmetic_mean'] * 100, color='red', linestyle='--', label='Arithmetic Mean')
ax1.axvline(metrics['geometric_mean'] * 100, color='green', linestyle='--', label='Geometric Mean')
ax1.axvline(metrics['median'] * 100, color='blue', linestyle='--', label='Median')
ax1.set_xlabel('Annual Growth Rate (%)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Growth Rates')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Q-Q plot
ax2 = axes[1]
stats.probplot(growth_rates, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Normality Check)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Risk Metrics

```python
from ergodic_insurance.risk_metrics import RiskMetrics

def calculate_risk_metrics(results):
    """Calculate comprehensive risk metrics."""

    risk_calc = RiskMetrics()

    final_wealths = results['final_wealths']
    trajectories = results['trajectories']

    # Value at Risk (VaR)
    var_95 = risk_calc.calculate_var(final_wealths, confidence=0.95)
    var_99 = risk_calc.calculate_var(final_wealths, confidence=0.99)

    # Conditional Value at Risk (CVaR)
    cvar_95 = risk_calc.calculate_cvar(final_wealths, confidence=0.95)
    cvar_99 = risk_calc.calculate_cvar(final_wealths, confidence=0.99)

    # Maximum Drawdown
    max_drawdowns = []
    for trajectory in trajectories[:100]:  # Sample for speed
        peaks = np.maximum.accumulate(trajectory)
        drawdowns = (trajectory - peaks) / peaks
        max_drawdowns.append(np.min(drawdowns))

    avg_max_drawdown = np.mean(max_drawdowns)
    worst_drawdown = np.min(max_drawdowns)

    # Ruin probability
    ruin_prob = np.mean(final_wealths <= 0)

    # Near-ruin (wealth < 10% of initial)
    near_ruin_prob = np.mean(final_wealths < 0.1 * manufacturer.initial_assets)

    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'avg_max_drawdown': avg_max_drawdown,
        'worst_drawdown': worst_drawdown,
        'ruin_prob': ruin_prob,
        'near_ruin_prob': near_ruin_prob
    }

risk_metrics = calculate_risk_metrics(results)

print("\nRisk Metrics:")
print("-" * 40)
print(f"VaR (95%):        ${risk_metrics['var_95']:,.0f}")
print(f"VaR (99%):        ${risk_metrics['var_99']:,.0f}")
print(f"CVaR (95%):       ${risk_metrics['cvar_95']:,.0f}")
print(f"CVaR (99%):       ${risk_metrics['cvar_99']:,.0f}")
print(f"Avg Max Drawdown: {risk_metrics['avg_max_drawdown']:.1%}")
print(f"Worst Drawdown:   {risk_metrics['worst_drawdown']:.1%}")
print(f"Ruin Probability: {risk_metrics['ruin_prob']:.2%}")
print(f"Near-Ruin Prob:   {risk_metrics['near_ruin_prob']:.2%}")

# Risk visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# VaR/CVaR visualization
ax1 = axes[0, 0]
sorted_wealth = np.sort(results['final_wealths'])
ax1.plot(sorted_wealth)
ax1.axhline(y=risk_metrics['var_95'], color='orange', linestyle='--', label='VaR 95%')
ax1.axhline(y=risk_metrics['cvar_95'], color='red', linestyle='--', label='CVaR 95%')
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_xlabel('Scenario (sorted)')
ax1.set_ylabel('Final Wealth ($)')
ax1.set_title('Value at Risk Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Drawdown distribution
ax2 = axes[0, 1]
ax2.hist(np.array(max_drawdowns) * 100, bins=30, alpha=0.7, edgecolor='black')
ax2.axvline(x=risk_metrics['avg_max_drawdown'] * 100, color='red', linestyle='--', label='Average')
ax2.axvline(x=risk_metrics['worst_drawdown'] * 100, color='darkred', linestyle='--', label='Worst')
ax2.set_xlabel('Maximum Drawdown (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Maximum Drawdown Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Survival curve
ax3 = axes[1, 0]
survival_by_year = []
for year in range(21):
    year_wealth = [traj[year] if year < len(traj) else traj[-1]
                  for traj in results['trajectories'][:100]]
    survival_rate = np.mean(np.array(year_wealth) > 0)
    survival_by_year.append(survival_rate)

ax3.plot(survival_by_year, 'o-', linewidth=2)
ax3.set_xlabel('Year')
ax3.set_ylabel('Survival Probability')
ax3.set_title('Survival Curve Over Time')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.05])

# Wealth percentiles over time
ax4 = axes[1, 1]
percentiles = [5, 25, 50, 75, 95]
colors = ['red', 'orange', 'green', 'orange', 'red']
for p, color in zip(percentiles, colors):
    percentile_trajectory = []
    for year in range(21):
        year_wealth = [traj[year] if year < len(traj) else traj[-1]
                      for traj in results['trajectories'][:100]]
        percentile_trajectory.append(np.percentile(year_wealth, p))
    ax4.plot(percentile_trajectory, label=f'{p}th percentile', color=color, alpha=0.7)

ax4.set_xlabel('Year')
ax4.set_ylabel('Wealth ($)')
ax4.set_title('Wealth Percentiles Over Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Ergodic vs Ensemble Analysis

### The Fundamental Difference

```python
# Create ergodic analyzer
ergodic_analyzer = ErgodicAnalyzer()

def compare_ergodic_ensemble(n_paths=100, n_years=50):
    """Compare ergodic (time) vs ensemble (space) averages."""

    # Run simulations
    np.random.seed(42)
    trajectories = []

    for i in range(n_paths):
        sim_result = mc_analyzer.run_single_simulation(
            n_years=n_years,
            retention=1_000_000,
            limit=10_000_000,
            premium_rate=0.02,
            seed=i
        )
        trajectories.append(sim_result['wealth_trajectory'])

    trajectories = np.array(trajectories)

    # Calculate ensemble average (across paths at each time)
    ensemble_avg = np.mean(trajectories, axis=0)

    # Calculate time average (for typical path)
    typical_path_idx = n_paths // 2  # Middle path as "typical"
    typical_path = trajectories[typical_path_idx]

    # Time average growth rate
    time_avg_growth = np.zeros(n_years)
    for t in range(1, n_years + 1):
        time_avg_growth[t-1] = (typical_path[t] / typical_path[0]) ** (1/t) - 1

    # Ensemble average growth rate
    ensemble_growth = np.zeros(n_years)
    for t in range(1, n_years + 1):
        ensemble_growth[t-1] = (ensemble_avg[t] / ensemble_avg[0]) ** (1/t) - 1

    return {
        'trajectories': trajectories,
        'ensemble_avg': ensemble_avg,
        'typical_path': typical_path,
        'time_avg_growth': time_avg_growth,
        'ensemble_growth': ensemble_growth
    }

# Run comparison
ergodic_results = compare_ergodic_ensemble(n_paths=100, n_years=30)

# Visualize the difference
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trajectories with averages
ax1 = axes[0, 0]
for i in range(20):  # Plot subset
    ax1.plot(ergodic_results['trajectories'][i], alpha=0.1, color='gray')
ax1.plot(ergodic_results['ensemble_avg'], 'b-', linewidth=3, label='Ensemble Average')
ax1.plot(ergodic_results['typical_path'], 'r-', linewidth=2, label='Typical Path')
ax1.set_xlabel('Year')
ax1.set_ylabel('Wealth ($)')
ax1.set_title('Ensemble vs Typical Path')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Growth rates comparison
ax2 = axes[0, 1]
ax2.plot(ergodic_results['ensemble_growth'] * 100, 'b-', linewidth=2, label='Ensemble Growth')
ax2.plot(ergodic_results['time_avg_growth'] * 100, 'r-', linewidth=2, label='Time Average Growth')
ax2.set_xlabel('Time Horizon (years)')
ax2.set_ylabel('Annualized Growth Rate (%)')
ax2.set_title('Growth Rate: Ensemble vs Time Average')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Distribution evolution
ax3 = axes[1, 0]
years_to_plot = [0, 5, 10, 20, 29]
for year in years_to_plot:
    wealth_dist = ergodic_results['trajectories'][:, year]
    ax3.hist(wealth_dist / 1e6, bins=30, alpha=0.5, label=f'Year {year}')
ax3.set_xlabel('Wealth ($M)')
ax3.set_ylabel('Frequency')
ax3.set_title('Wealth Distribution Evolution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Ergodicity breaking visualization
ax4 = axes[1, 1]
# Calculate coefficient of variation over time
cv_over_time = []
for t in range(30):
    wealth_at_t = ergodic_results['trajectories'][:, t]
    cv = np.std(wealth_at_t) / np.mean(wealth_at_t) if np.mean(wealth_at_t) > 0 else 0
    cv_over_time.append(cv)

ax4.plot(cv_over_time, 'g-', linewidth=2)
ax4.set_xlabel('Year')
ax4.set_ylabel('Coefficient of Variation')
ax4.set_title('Wealth Inequality Over Time (Ergodicity Breaking)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nErgodic vs Ensemble Analysis:")
print("-" * 50)
print(f"Final Ensemble Average: ${ergodic_results['ensemble_avg'][-1]:,.0f}")
print(f"Final Typical Path:     ${ergodic_results['typical_path'][-1]:,.0f}")
print(f"Ratio:                  {ergodic_results['ensemble_avg'][-1] / ergodic_results['typical_path'][-1]:.2f}x")
print(f"\nFinal Growth Rates:")
print(f"Ensemble Growth:        {ergodic_results['ensemble_growth'][-1]:.2%}")
print(f"Time Average Growth:    {ergodic_results['time_avg_growth'][-1]:.2%}")
print(f"Difference:             {(ergodic_results['ensemble_growth'][-1] - ergodic_results['time_avg_growth'][-1]) * 100:.1f} bps")
```

## Statistical Analysis

### Hypothesis Testing

```python
def statistical_comparison(strategy_a_results, strategy_b_results):
    """Perform statistical tests to compare strategies."""

    # Extract growth rates
    growth_a = strategy_a_results['growth_rates']
    growth_b = strategy_b_results['growth_rates']

    # T-test for mean difference
    t_stat, p_value_t = stats.ttest_ind(growth_a, growth_b)

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_u = stats.mannwhitneyu(growth_a, growth_b)

    # Kolmogorov-Smirnov test (distribution difference)
    ks_stat, p_value_ks = stats.ks_2samp(growth_a, growth_b)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(growth_a)**2 + np.std(growth_b)**2) / 2)
    cohens_d = (np.mean(growth_a) - np.mean(growth_b)) / pooled_std

    # Survival rate comparison (chi-square)
    survival_a = np.sum(strategy_a_results['final_wealths'] > 0)
    survival_b = np.sum(strategy_b_results['final_wealths'] > 0)
    total_a = len(strategy_a_results['final_wealths'])
    total_b = len(strategy_b_results['final_wealths'])

    contingency_table = np.array([
        [survival_a, total_a - survival_a],
        [survival_b, total_b - survival_b]
    ])
    chi2, p_value_chi2, _, _ = stats.chi2_contingency(contingency_table)

    return {
        't_test': {'statistic': t_stat, 'p_value': p_value_t},
        'mann_whitney': {'statistic': u_stat, 'p_value': p_value_u},
        'ks_test': {'statistic': ks_stat, 'p_value': p_value_ks},
        'cohens_d': cohens_d,
        'chi2_survival': {'statistic': chi2, 'p_value': p_value_chi2}
    }

# Compare two strategies
strategy_a = mc_analyzer.run_simulations(
    n_simulations=500,
    n_years=20,
    retention=500_000,
    limit=10_000_000,
    premium_rate=0.025,
    seed=42
)

strategy_b = mc_analyzer.run_simulations(
    n_simulations=500,
    n_years=20,
    retention=1_500_000,
    limit=5_000_000,
    premium_rate=0.015,
    seed=42
)

comparison = statistical_comparison(strategy_a, strategy_b)

print("\nStatistical Comparison:")
print("-" * 50)
print("Strategy A: Low retention, high coverage")
print("Strategy B: High retention, low coverage")
print()
print(f"Mean Growth A: {np.mean(strategy_a['growth_rates']):.2%}")
print(f"Mean Growth B: {np.mean(strategy_b['growth_rates']):.2%}")
print()
print("Statistical Tests:")
print(f"T-test p-value:        {comparison['t_test']['p_value']:.4f}")
print(f"Mann-Whitney p-value:  {comparison['mann_whitney']['p_value']:.4f}")
print(f"KS test p-value:       {comparison['ks_test']['p_value']:.4f}")
print(f"Cohen's d:             {comparison['cohens_d']:.3f}")
print(f"Chi-square (survival): {comparison['chi2_survival']['p_value']:.4f}")
print()
if comparison['t_test']['p_value'] < 0.05:
    print("✅ Strategies are statistically different (p < 0.05)")
else:
    print("❌ No significant difference detected (p >= 0.05)")
```

### Confidence Intervals

```python
def calculate_confidence_intervals(results, confidence=0.95):
    """Calculate confidence intervals for key metrics."""

    growth_rates = results['growth_rates']
    final_wealths = results['final_wealths']

    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_survival = []

    n_samples = len(growth_rates)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_growth = growth_rates[indices]
        sample_wealth = final_wealths[indices]

        bootstrap_means.append(np.mean(sample_growth))
        bootstrap_medians.append(np.median(sample_growth))
        bootstrap_survival.append(np.mean(sample_wealth > 0))

    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100

    ci_mean = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    ci_median = np.percentile(bootstrap_medians, [lower_percentile, upper_percentile])
    ci_survival = np.percentile(bootstrap_survival, [lower_percentile, upper_percentile])

    # Standard error
    se_mean = np.std(bootstrap_means)
    se_median = np.std(bootstrap_medians)
    se_survival = np.std(bootstrap_survival)

    return {
        'mean': {'estimate': np.mean(growth_rates), 'ci': ci_mean, 'se': se_mean},
        'median': {'estimate': np.median(growth_rates), 'ci': ci_median, 'se': se_median},
        'survival': {'estimate': np.mean(final_wealths > 0), 'ci': ci_survival, 'se': se_survival}
    }

# Calculate confidence intervals
ci_results = calculate_confidence_intervals(results, confidence=0.95)

print("\n95% Confidence Intervals:")
print("-" * 50)
for metric, values in ci_results.items():
    print(f"\n{metric.capitalize()}:")
    print(f"  Estimate: {values['estimate']:.3f}")
    print(f"  95% CI:   [{values['ci'][0]:.3f}, {values['ci'][1]:.3f}]")
    print(f"  Std Error: {values['se']:.4f}")

# Visualize confidence intervals
fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = list(ci_results.keys())
estimates = [ci_results[m]['estimate'] for m in metrics_names]
lower_bounds = [ci_results[m]['ci'][0] for m in metrics_names]
upper_bounds = [ci_results[m]['ci'][1] for m in metrics_names]
errors = [[estimates[i] - lower_bounds[i] for i in range(len(estimates))],
         [upper_bounds[i] - estimates[i] for i in range(len(estimates))]]

x_pos = np.arange(len(metrics_names))
ax.errorbar(x_pos, estimates, yerr=errors, fmt='o', markersize=10,
           capsize=5, capthick=2, elinewidth=2)

ax.set_xticks(x_pos)
ax.set_xticklabels([m.capitalize() for m in metrics_names])
ax.set_ylabel('Value')
ax.set_title('Key Metrics with 95% Confidence Intervals')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Decision Making Framework

### Creating Decision Tables

```python
def create_decision_table(strategies_results):
    """Create comprehensive decision table."""

    decision_data = []

    for name, results in strategies_results.items():
        growth_rates = results['growth_rates']
        final_wealths = results['final_wealths']

        row = {
            'Strategy': name,
            'Mean Growth': np.mean(growth_rates),
            'Median Growth': np.median(growth_rates),
            'Std Dev': np.std(growth_rates),
            'Survival Rate': np.mean(final_wealths > 0),
            'VaR 95%': np.percentile(final_wealths, 5),
            'CVaR 95%': np.mean(final_wealths[final_wealths <= np.percentile(final_wealths, 5)]),
            'Sharpe Ratio': np.mean(growth_rates) / np.std(growth_rates) if np.std(growth_rates) > 0 else 0
        }
        decision_data.append(row)

    df = pd.DataFrame(decision_data)

    # Add rankings
    df['Growth Rank'] = df['Mean Growth'].rank(ascending=False)
    df['Survival Rank'] = df['Survival Rate'].rank(ascending=False)
    df['Sharpe Rank'] = df['Sharpe Ratio'].rank(ascending=False)
    df['Overall Rank'] = (df['Growth Rank'] + df['Survival Rank'] + df['Sharpe Rank']) / 3

    return df.sort_values('Overall Rank')

# Test multiple strategies
strategies_to_test = {
    'Conservative': {'retention': 250_000, 'limit': 15_000_000, 'premium_rate': 0.025},
    'Balanced': {'retention': 1_000_000, 'limit': 10_000_000, 'premium_rate': 0.02},
    'Aggressive': {'retention': 2_000_000, 'limit': 5_000_000, 'premium_rate': 0.015},
    'Minimal': {'retention': 3_000_000, 'limit': 2_000_000, 'premium_rate': 0.01}
}

all_results = {}
for name, params in strategies_to_test.items():
    all_results[name] = mc_analyzer.run_simulations(
        n_simulations=500,
        n_years=20,
        **params,
        seed=42
    )

decision_table = create_decision_table(all_results)

print("\nDecision Table:")
print("=" * 120)
print(decision_table.to_string(index=False, float_format='%.3f'))

# Visualize decision space
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Risk-Return scatter
ax1 = axes[0]
for idx, row in decision_table.iterrows():
    ax1.scatter(1 - row['Survival Rate'], row['Mean Growth'],
               s=200, alpha=0.7, label=row['Strategy'])
    ax1.annotate(row['Strategy'],
                (1 - row['Survival Rate'], row['Mean Growth']),
                xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('Ruin Probability')
ax1.set_ylabel('Mean Growth Rate')
ax1.set_title('Risk-Return Trade-off')
ax1.grid(True, alpha=0.3)

# Multi-criteria radar chart
ax2 = axes[1]
categories = ['Growth', 'Survival', 'Sharpe', 'Low Volatility']
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for idx, row in decision_table.iterrows():
    values = [
        row['Mean Growth'] * 10,  # Scale for visibility
        row['Survival Rate'],
        row['Sharpe Ratio'] / 2,  # Scale
        1 - row['Std Dev'] * 5    # Invert and scale
    ]
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=row['Strategy'])
    ax2.fill(angles, values, alpha=0.25)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_ylim([0, 1])
ax2.set_title('Multi-Criteria Performance')
ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Sensitivity Analysis

```python
def perform_sensitivity_analysis(base_params, param_ranges):
    """Analyze sensitivity to parameter changes."""

    sensitivity_results = {}

    for param_name, values in param_ranges.items():
        param_results = []

        for value in values:
            # Create modified parameters
            test_params = base_params.copy()
            test_params[param_name] = value

            # Run simulation
            results = mc_analyzer.run_simulations(
                n_simulations=200,
                n_years=20,
                **test_params,
                seed=42
            )

            param_results.append({
                'value': value,
                'mean_growth': np.mean(results['growth_rates']),
                'survival_rate': results['survival_rate']
            })

        sensitivity_results[param_name] = param_results

    return sensitivity_results

# Define base case and ranges
base_params = {
    'retention': 1_000_000,
    'limit': 10_000_000,
    'premium_rate': 0.02
}

param_ranges = {
    'retention': np.linspace(250_000, 2_000_000, 8),
    'limit': np.linspace(5_000_000, 15_000_000, 6),
    'premium_rate': np.linspace(0.01, 0.03, 5)
}

sensitivity = perform_sensitivity_analysis(base_params, param_ranges)

# Visualize sensitivity
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (param_name, results) in enumerate(sensitivity.items()):
    ax = axes[idx]

    values = [r['value'] for r in results]
    growth = [r['mean_growth'] * 100 for r in results]
    survival = [r['survival_rate'] * 100 for r in results]

    ax2 = ax.twinx()

    line1 = ax.plot(values, growth, 'b-o', label='Growth Rate')
    line2 = ax2.plot(values, survival, 'r-s', label='Survival Rate')

    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel('Growth Rate (%)', color='b')
    ax2.set_ylabel('Survival Rate (%)', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title(f'Sensitivity to {param_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)

    # Mark base case
    base_value = base_params[param_name]
    ax.axvline(x=base_value, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\nSensitivity Analysis Summary:")
print("-" * 50)
for param_name, results in sensitivity.items():
    growth_values = [r['mean_growth'] for r in results]
    growth_range = max(growth_values) - min(growth_values)
    print(f"{param_name.replace('_', ' ').title():20} Range: {growth_range:.2%}")
```

## Communicating Results

### Executive Summary Generation

```python
def generate_executive_summary(strategy_name, results, params):
    """Generate executive-friendly summary."""

    growth_rates = results['growth_rates']
    final_wealths = results['final_wealths']

    summary = f"""
EXECUTIVE SUMMARY: {strategy_name} Insurance Strategy
{'=' * 60}

RECOMMENDATION: {"IMPLEMENT" if np.mean(growth_rates) > 0.05 and results['survival_rate'] > 0.95 else "REVIEW"}

KEY METRICS:
• Expected Annual Growth: {np.mean(growth_rates):.1%}
• Survival Probability: {results['survival_rate']:.1%}
• Risk-Adjusted Return (Sharpe): {np.mean(growth_rates)/np.std(growth_rates):.2f}

INSURANCE STRUCTURE:
• Retention (Deductible): ${params['retention']:,.0f}
• Coverage Limit: ${params['limit']:,.0f}
• Annual Premium: ${params['limit'] * params['premium_rate']:,.0f}
• Premium as % of Revenue: {(params['limit'] * params['premium_rate']) / (manufacturer.initial_assets * manufacturer.asset_turnover) * 100:.2f}%

RISK PROFILE:
• Probability of Ruin: {(1 - results['survival_rate']) * 100:.1f}%
• Value at Risk (95%): ${np.percentile(final_wealths, 5):,.0f}
• Expected Final Wealth: ${np.mean(final_wealths):,.0f}

COMPARISON TO NO INSURANCE:
• Growth Improvement: +{(np.mean(growth_rates) - 0.04) * 100:.0f} bps
• Survival Improvement: +{(results['survival_rate'] - 0.80) * 100:.0f}%
• ROI on Premium: {(np.mean(growth_rates) - 0.04) / (params['premium_rate'] * params['limit'] / manufacturer.initial_assets):.1f}x

RECOMMENDATION RATIONALE:
{"✅ Growth rate exceeds 5% target" if np.mean(growth_rates) > 0.05 else "⚠️ Growth rate below 5% target"}
{"✅ Survival rate exceeds 95% threshold" if results['survival_rate'] > 0.95 else "⚠️ Survival rate below 95% threshold"}
{"✅ Positive risk-adjusted returns" if np.mean(growth_rates)/np.std(growth_rates) > 0 else "⚠️ Negative risk-adjusted returns"}

NEXT STEPS:
1. Review with risk committee
2. Obtain insurance quotes from 3+ carriers
3. Implement monitoring dashboard
4. Schedule quarterly reviews
"""

    return summary

# Generate summary for best strategy
best_strategy_name = 'Balanced'
best_results = all_results[best_strategy_name]
best_params = strategies_to_test[best_strategy_name]

summary = generate_executive_summary(best_strategy_name, best_results, best_params)
print(summary)
```

### Creating Final Report Visualizations

```python
def create_report_visualizations(results, strategy_name):
    """Create publication-ready visualizations."""

    fig = plt.figure(figsize=(16, 10))

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Wealth trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    sample_trajectories = results['trajectories'][:20]
    for traj in sample_trajectories:
        ax1.plot(traj/1e6, alpha=0.3, color='gray')
    mean_trajectory = np.mean(results['trajectories'][:100], axis=0)
    ax1.plot(mean_trajectory/1e6, 'b-', linewidth=2, label='Mean Path')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Wealth ($M)')
    ax1.set_title(f'{strategy_name}: Wealth Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Growth distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results['growth_rates']*100, bins=30, alpha=0.7,
            edgecolor='black', color='green')
    ax2.axvline(x=np.mean(results['growth_rates'])*100,
               color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Annual Growth Rate (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Growth Rate Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Risk metrics
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['Survival\nRate', 'Sharpe\nRatio', 'Growth\nRate']
    values = [
        results['survival_rate'],
        np.mean(results['growth_rates'])/np.std(results['growth_rates']),
        np.mean(results['growth_rates']) * 10  # Scale for visibility
    ]
    colors = ['green' if v > 0.5 else 'red' for v in values]
    ax3.bar(metrics, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Value')
    ax3.set_title('Key Performance Indicators')
    ax3.set_ylim([0, max(values) * 1.2])

    # 4. Percentile fan chart
    ax4 = fig.add_subplot(gs[1, :2])
    percentiles = [5, 25, 50, 75, 95]
    colors_map = {5: 'red', 25: 'orange', 50: 'green', 75: 'orange', 95: 'red'}
    alphas = {5: 0.3, 25: 0.5, 50: 1.0, 75: 0.5, 95: 0.3}

    for p in percentiles:
        percentile_path = []
        for t in range(21):
            year_wealth = [traj[t] if t < len(traj) else traj[-1]
                          for traj in results['trajectories'][:100]]
            percentile_path.append(np.percentile(year_wealth, p))
        ax4.plot(percentile_path, label=f'{p}th %ile',
                color=colors_map[p], alpha=alphas[p], linewidth=2)

    ax4.set_xlabel('Year')
    ax4.set_ylabel('Wealth ($)')
    ax4.set_title('Wealth Percentiles (Fan Chart)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('tight')
    ax5.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['Mean Growth', f"{np.mean(results['growth_rates']):.2%}"],
        ['Median Growth', f"{np.median(results['growth_rates']):.2%}"],
        ['Std Deviation', f"{np.std(results['growth_rates']):.2%}"],
        ['Survival Rate', f"{results['survival_rate']:.1%}"],
        ['VaR (95%)', f"${np.percentile(results['final_wealths'], 5):,.0f}"],
        ['Mean Final Wealth', f"${np.mean(results['final_wealths']):,.0f}"]
    ]

    table = ax5.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle(f'{strategy_name} Strategy Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

# Create report visualization
report_fig = create_report_visualizations(all_results['Balanced'], 'Balanced')
plt.show()

# Save for presentation
# report_fig.savefig('insurance_analysis_report.png', dpi=300, bbox_inches='tight')
```

## Next Steps

Now that you can analyze results comprehensively:

1. **[Advanced Scenarios](06_advanced_scenarios.md)**: Apply analysis to complex real-world cases

## Summary

You've mastered:
- ✅ Calculating and interpreting growth and risk metrics
- ✅ Understanding ergodic vs ensemble differences
- ✅ Performing statistical analysis and hypothesis testing
- ✅ Creating decision tables and frameworks
- ✅ Conducting sensitivity analysis
- ✅ Generating executive summaries and reports

You're ready to make data-driven insurance decisions with confidence!

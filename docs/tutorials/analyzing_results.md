---
layout: default
title: Analyzing Results
---

# Analyzing Results

Learn how to interpret simulation outputs and make data-driven insurance decisions.

## Overview

This tutorial covers:
- Understanding key metrics
- Visualization techniques
- Statistical analysis
- Decision criteria
- Reporting results

## Key Metrics

### Growth Metrics

```python
from ergodic_insurance.src.result_aggregator import ResultAggregator

# Aggregate simulation results
aggregator = ResultAggregator(simulation_results)

# Calculate growth metrics
growth_metrics = {
    "ensemble_average": aggregator.ensemble_average_growth(),
    "time_average": aggregator.time_average_growth(),
    "median_growth": aggregator.median_growth(),
    "growth_volatility": aggregator.growth_volatility()
}

print("Growth Analysis:")
print(f"Ensemble Average: {growth_metrics['ensemble_average']:.2%}")
print(f"Time Average: {growth_metrics['time_average']:.2%}")
print(f"Ergodic Gap: {growth_metrics['ensemble_average'] - growth_metrics['time_average']:.2%}")
```

### Risk Metrics

```python
from ergodic_insurance.src.risk_metrics import calculate_risk_metrics

risk_metrics = calculate_risk_metrics(simulation_results)

print("\nRisk Analysis:")
print(f"Ruin Probability: {risk_metrics['ruin_probability']:.2%}")
print(f"95% VaR: ${risk_metrics['var_95']:,.0f}")
print(f"99% CVaR: ${risk_metrics['cvar_99']:,.0f}")
print(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}")
print(f"Recovery Time: {risk_metrics['avg_recovery_time']:.1f} years")
```

### Financial Metrics

```python
def calculate_financial_metrics(results, insurance_program):
    """Calculate key financial metrics."""

    metrics = {}

    # Return metrics
    metrics['roe'] = results.net_income / results.avg_equity
    metrics['roa'] = results.net_income / results.avg_assets

    # Insurance efficiency
    metrics['loss_ratio'] = results.paid_losses / insurance_program.total_premium
    metrics['combined_ratio'] = (results.paid_losses + insurance_program.expenses) / insurance_program.total_premium

    # Cost metrics
    metrics['premium_to_revenue'] = insurance_program.total_premium / results.avg_revenue
    metrics['tcor'] = (results.retained_losses + insurance_program.total_premium) / results.avg_revenue

    return metrics

financial = calculate_financial_metrics(results, insurance)
print(f"\nROE: {financial['roe']:.2%}")
print(f"Premium/Revenue: {financial['premium_to_revenue']:.2%}")
print(f"Total Cost of Risk: {financial['tcor']:.2%}")
```

## Visualization Techniques

### Distribution Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_outcome_distribution(results):
    """Plot distribution of final outcomes."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Final asset distribution
    final_assets = [path[-1] for path in results.asset_paths]
    axes[0, 0].hist(final_assets, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(final_assets), color='red', linestyle='--', label='Mean')
    axes[0, 0].axvline(np.median(final_assets), color='green', linestyle='--', label='Median')
    axes[0, 0].set_xlabel('Final Assets ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Final Assets')
    axes[0, 0].legend()

    # Growth rate distribution
    growth_rates = [calculate_growth_rate(path) for path in results.asset_paths]
    axes[0, 1].hist(growth_rates, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(growth_rates), color='red', linestyle='--', label='Ensemble')
    axes[0, 1].axvline(calculate_time_average(growth_rates), color='blue', linestyle='--', label='Time')
    axes[0, 1].set_xlabel('Annual Growth Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Growth Rate Distribution')
    axes[0, 1].legend()

    # Ruin time distribution (for ruined paths)
    ruin_times = [find_ruin_time(path) for path in results.asset_paths if path[-1] <= 0]
    if ruin_times:
        axes[1, 0].hist(ruin_times, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Years to Ruin')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Ruin Time Distribution ({len(ruin_times)} ruined)')

    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(growth_rates, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Growth Rate Normality')

    plt.tight_layout()
    plt.show()
```

### Time Series Plots

```python
def plot_percentile_paths(results, percentiles=[5, 25, 50, 75, 95]):
    """Plot percentile paths over time."""

    plt.figure(figsize=(12, 6))

    # Calculate percentiles at each time point
    time_points = range(len(results.asset_paths[0]))
    percentile_paths = {}

    for p in percentiles:
        percentile_paths[p] = [
            np.percentile([path[t] for path in results.asset_paths], p)
            for t in time_points
        ]

    # Plot percentile bands
    plt.fill_between(time_points,
                     percentile_paths[5], percentile_paths[95],
                     alpha=0.2, label='5-95%')
    plt.fill_between(time_points,
                     percentile_paths[25], percentile_paths[75],
                     alpha=0.3, label='25-75%')

    # Plot median
    plt.plot(time_points, percentile_paths[50],
             color='blue', linewidth=2, label='Median')

    # Plot mean
    mean_path = [np.mean([path[t] for path in results.asset_paths])
                 for t in time_points]
    plt.plot(time_points, mean_path,
             color='red', linewidth=2, linestyle='--', label='Mean')

    plt.xlabel('Years')
    plt.ylabel('Assets ($)')
    plt.title('Asset Evolution: Percentile Bands')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### Comparison Plots

```python
def plot_insurance_comparison(results_without, results_with):
    """Compare results with and without insurance."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    scenarios = [
        ("Without Insurance", results_without),
        ("With Insurance", results_with)
    ]

    # Growth rates
    ax = axes[0, 0]
    for name, results in scenarios:
        growth_rates = [calculate_growth_rate(p) for p in results.asset_paths]
        ax.hist(growth_rates, bins=30, alpha=0.5, label=name)
    ax.set_xlabel('Growth Rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Growth Rate Comparison')
    ax.legend()

    # Ruin probability over time
    ax = axes[0, 1]
    for name, results in scenarios:
        ruin_by_year = calculate_ruin_by_year(results)
        ax.plot(ruin_by_year, label=name, linewidth=2)
    ax.set_xlabel('Years')
    ax.set_ylabel('Cumulative Ruin Probability')
    ax.set_title('Ruin Probability Evolution')
    ax.legend()

    # Final wealth distribution
    ax = axes[0, 2]
    for name, results in scenarios:
        final_wealth = [p[-1] for p in results.asset_paths]
        ax.boxplot([final_wealth], labels=[name])
    ax.set_ylabel('Final Assets ($)')
    ax.set_title('Final Wealth Distribution')

    # Volatility comparison
    ax = axes[1, 0]
    for name, results in scenarios:
        volatilities = calculate_rolling_volatility(results)
        ax.plot(volatilities, label=name, linewidth=2)
    ax.set_xlabel('Years')
    ax.set_ylabel('Rolling Volatility')
    ax.set_title('Volatility Over Time')
    ax.legend()

    # Cost-benefit analysis
    ax = axes[1, 1]
    metrics = {
        "Without": calculate_metrics(results_without),
        "With": calculate_metrics(results_with)
    }

    categories = ['Growth', 'Ruin Prob', 'Volatility', 'Sharpe']
    x = np.arange(len(categories))
    width = 0.35

    without_vals = [metrics["Without"][cat] for cat in categories.lower()]
    with_vals = [metrics["With"][cat] for cat in categories.lower()]

    ax.bar(x - width/2, without_vals, width, label='Without')
    ax.bar(x + width/2, with_vals, width, label='With')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title('Metric Comparison')
    ax.legend()

    # Efficient frontier
    ax = axes[1, 2]
    plot_efficient_frontier(ax, results_without, results_with)

    plt.tight_layout()
    plt.show()
```

## Statistical Analysis

### Hypothesis Testing

```python
from scipy import stats

def test_insurance_benefit(results_without, results_with):
    """Test if insurance significantly improves outcomes."""

    # Extract growth rates
    growth_without = [calculate_growth_rate(p) for p in results_without.asset_paths]
    growth_with = [calculate_growth_rate(p) for p in results_with.asset_paths]

    # T-test for means
    t_stat, p_value = stats.ttest_ind(growth_with, growth_without)

    print("Growth Rate Comparison:")
    print(f"Without Insurance: {np.mean(growth_without):.4f} ± {np.std(growth_without):.4f}")
    print(f"With Insurance: {np.mean(growth_with):.4f} ± {np.std(growth_with):.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    if p_value < 0.05:
        print("✓ Insurance significantly improves growth (p < 0.05)")
    else:
        print("✗ No significant difference detected")

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(growth_with, growth_without, alternative='greater')
    print(f"\nMann-Whitney U test p-value: {p_value_mw:.4e}")

    # Effect size (Cohen's d)
    cohens_d = (np.mean(growth_with) - np.mean(growth_without)) / np.sqrt(
        (np.std(growth_with)**2 + np.std(growth_without)**2) / 2
    )
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")

    return {
        "significant": p_value < 0.05,
        "effect_size": cohens_d,
        "p_value": p_value
    }
```

### Confidence Intervals

```python
def calculate_confidence_intervals(results, confidence=0.95):
    """Calculate confidence intervals for key metrics."""

    from scipy import stats

    # Growth rate CI
    growth_rates = [calculate_growth_rate(p) for p in results.asset_paths]
    growth_ci = stats.t.interval(
        confidence,
        len(growth_rates)-1,
        loc=np.mean(growth_rates),
        scale=stats.sem(growth_rates)
    )

    # Ruin probability CI (using Wilson score interval)
    n_ruined = sum(1 for p in results.asset_paths if p[-1] <= 0)
    n_total = len(results.asset_paths)
    ruin_ci = proportion_confint(n_ruined, n_total, alpha=1-confidence, method='wilson')

    # Bootstrap CI for complex metrics
    sharpe_ci = bootstrap_confidence_interval(
        results,
        metric_fn=calculate_sharpe_ratio,
        confidence=confidence
    )

    print(f"\nConfidence Intervals ({confidence*100:.0f}%):")
    print(f"Growth Rate: [{growth_ci[0]:.4f}, {growth_ci[1]:.4f}]")
    print(f"Ruin Probability: [{ruin_ci[0]:.4f}, {ruin_ci[1]:.4f}]")
    print(f"Sharpe Ratio: [{sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f}]")

    return {
        "growth": growth_ci,
        "ruin": ruin_ci,
        "sharpe": sharpe_ci
    }
```

## Decision Criteria

### Decision Matrix

```python
def create_decision_matrix(insurance_options):
    """Create decision matrix for insurance selection."""

    import pandas as pd

    # Evaluate each option
    evaluations = []

    for option in insurance_options:
        results = run_simulation(manufacturer, option)

        evaluation = {
            "Option": option.name,
            "Premium": option.total_premium,
            "Growth Rate": results.growth_rate,
            "Ruin Prob": results.ruin_probability,
            "Sharpe Ratio": results.sharpe_ratio,
            "ROE": results.roe,
            "Score": calculate_weighted_score(results)
        }

        evaluations.append(evaluation)

    # Create DataFrame
    df = pd.DataFrame(evaluations)

    # Rank by score
    df['Rank'] = df['Score'].rank(ascending=False)

    # Format for display
    df = df.sort_values('Rank')

    print("\nDecision Matrix:")
    print(df.to_string(index=False))

    # Highlight best option
    best = df.iloc[0]
    print(f"\n✓ Recommended: {best['Option']}")
    print(f"  - Growth: {best['Growth Rate']:.2%}")
    print(f"  - Risk: {best['Ruin Prob']:.2%}")
    print(f"  - Cost: ${best['Premium']:,.0f}")

    return df
```

### Scenario Analysis

```python
def scenario_analysis(insurance_program):
    """Test insurance under different scenarios."""

    scenarios = {
        "Base Case": {"frequency": 0.5, "severity": 2_000_000},
        "Mild Stress": {"frequency": 0.8, "severity": 3_000_000},
        "Severe Stress": {"frequency": 1.5, "severity": 5_000_000},
        "Black Swan": {"frequency": 0.1, "severity": 50_000_000}
    }

    results = {}

    for name, params in scenarios.items():
        # Configure scenario
        scenario_config = create_scenario_config(params)

        # Run simulation
        scenario_results = run_simulation(
            manufacturer,
            insurance_program,
            scenario_config
        )

        results[name] = {
            "growth": scenario_results.growth_rate,
            "ruin": scenario_results.ruin_probability,
            "recovery": scenario_results.avg_recovery_time
        }

    # Display results
    print("\nScenario Analysis:")
    print("-" * 60)
    print(f"{'Scenario':<15} {'Growth':<10} {'Ruin Prob':<12} {'Recovery':<10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<15} {metrics['growth']:>9.2%} {metrics['ruin']:>11.2%} {metrics['recovery']:>9.1f}y")

    return results
```

## Reporting Results

### Executive Summary

```python
def generate_executive_summary(results, insurance_program):
    """Generate executive summary of results."""

    summary = f"""
    EXECUTIVE SUMMARY: Insurance Optimization Analysis
    ================================================

    Company Profile:
    - Starting Assets: ${manufacturer.starting_assets:,.0f}
    - Annual Revenue: ${manufacturer.annual_revenue:,.0f}
    - Operating Margin: {manufacturer.operating_margin:.1%}

    Recommended Insurance Program:
    - Structure: {insurance_program.structure_description}
    - Total Limit: ${insurance_program.total_limit:,.0f}
    - Annual Premium: ${insurance_program.total_premium:,.0f}
    - Premium/Revenue: {insurance_program.premium_ratio:.2%}

    Expected Benefits:
    - Growth Rate Improvement: {results.growth_improvement:.1%}
    - Ruin Probability Reduction: {results.ruin_reduction:.1%}
    - ROE Enhancement: {results.roe_improvement:.1%}

    Key Findings:
    1. Insurance increases long-term growth by {results.growth_improvement:.1%} annually
    2. Ruin probability drops from {results.baseline_ruin:.1%} to {results.insured_ruin:.1%}
    3. Optimal premium is {results.premium_multiplier:.1f}x expected losses
    4. Break-even occurs in year {results.breakeven_year}

    Recommendation:
    ✓ Implement the recommended insurance program
    ✓ Review and adjust annually based on loss experience
    ✓ Consider additional coverage for emerging risks

    Risk Warnings:
    - Results assume {results.n_simulations:,} Monte Carlo paths
    - Historical losses may not predict future events
    - Correlation assumptions may understate systemic risk
    """

    return summary
```

### Detailed Report

```python
def generate_detailed_report(results, insurance_program, output_path="report.pdf"):
    """Generate comprehensive PDF report."""

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        # Page 1: Summary
        fig = create_summary_page(results, insurance_program)
        pdf.savefig(fig)

        # Page 2: Growth Analysis
        fig = create_growth_analysis_page(results)
        pdf.savefig(fig)

        # Page 3: Risk Analysis
        fig = create_risk_analysis_page(results)
        pdf.savefig(fig)

        # Page 4: Insurance Structure
        fig = create_insurance_structure_page(insurance_program)
        pdf.savefig(fig)

        # Page 5: Sensitivity Analysis
        fig = create_sensitivity_page(results)
        pdf.savefig(fig)

        # Page 6: Recommendations
        fig = create_recommendations_page(results)
        pdf.savefig(fig)

    print(f"Report saved to {output_path}")
```

## Summary

You've learned to:
- Calculate and interpret key metrics
- Create effective visualizations
- Perform statistical analysis
- Apply decision criteria
- Generate professional reports

## Next Steps

- [Advanced Scenarios](advanced_scenarios.md) - Complex real-world applications
- Review example notebooks for complete analyses

For questions, see the [Troubleshooting Guide](troubleshooting.md).

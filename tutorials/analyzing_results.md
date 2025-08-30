---
layout: default
title: Analyzing Results Tutorial - Ergodic Insurance Framework
description: Learn how to interpret and present simulation results effectively
mathjax: true
---

# Analyzing Results Tutorial

## Overview

This tutorial teaches you how to analyze, interpret, and present the results from your insurance simulations. You'll learn to extract meaningful insights and make data-driven decisions.

## Understanding Key Metrics

### Time-Average vs Ensemble-Average Growth

The fundamental distinction in ergodic economics:

```python
import numpy as np
from ergodic_insurance.src import ErgodicAnalyzer

def calculate_growth_metrics(results):
    """Calculate both time and ensemble average growth rates"""

    analyzer = ErgodicAnalyzer(results)

    # Time-average: Growth rate of individual trajectories
    time_avg = analyzer.time_average_growth()

    # Ensemble-average: Growth of average wealth
    ensemble_avg = analyzer.ensemble_average_growth()

    # Ergodicity gap
    ergodicity_gap = ensemble_avg - time_avg

    print(f"Time-Average Growth: {time_avg:.2%}")
    print(f"Ensemble-Average Growth: {ensemble_avg:.2%}")
    print(f"Ergodicity Gap: {ergodicity_gap:.2%}")

    if ergodicity_gap > 0:
        print("⚠️ Non-ergodic: Individual experience differs from average")

    return time_avg, ensemble_avg, ergodicity_gap
```

### Risk Metrics

```python
def calculate_risk_metrics(results):
    """Calculate comprehensive risk metrics"""

    metrics = {}

    # Ruin probability
    trajectories = results.trajectories
    ruined = sum(1 for t in trajectories if min(t.wealth) <= 0)
    metrics['ruin_probability'] = ruined / len(trajectories)

    # Value at Risk (VaR)
    final_wealth = [t.wealth[-1] for t in trajectories]
    metrics['var_95'] = np.percentile(final_wealth, 5)
    metrics['var_99'] = np.percentile(final_wealth, 1)

    # Conditional Value at Risk (CVaR)
    threshold_95 = metrics['var_95']
    losses_beyond_var = [w for w in final_wealth if w < threshold_95]
    metrics['cvar_95'] = np.mean(losses_beyond_var) if losses_beyond_var else threshold_95

    # Maximum Drawdown
    for t in trajectories:
        peaks = np.maximum.accumulate(t.wealth)
        drawdowns = (peaks - t.wealth) / peaks
        t.max_drawdown = np.max(drawdowns)
    metrics['max_drawdown'] = np.mean([t.max_drawdown for t in trajectories])

    # Volatility
    returns = []
    for t in trajectories:
        period_returns = np.diff(t.wealth) / t.wealth[:-1]
        returns.extend(period_returns)
    metrics['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized

    return metrics

# Example usage
risk_metrics = calculate_risk_metrics(simulation_results)
print("\nRisk Metrics:")
for metric, value in risk_metrics.items():
    print(f"  {metric}: {value:.2%}" if 'probability' in metric or 'volatility' in metric
          else f"  {metric}: ${value:,.0f}")
```

## Statistical Analysis

### Distribution Analysis

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

def analyze_distribution(results):
    """Analyze the distribution of outcomes"""

    final_wealth = [t.wealth[-1] for t in results.trajectories]
    growth_rates = [t.calculate_growth_rate() for t in results.trajectories]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Wealth distribution
    ax = axes[0, 0]
    ax.hist(final_wealth, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(final_wealth), color='red', linestyle='--', label=f'Mean: ${np.mean(final_wealth):,.0f}')
    ax.axvline(np.median(final_wealth), color='green', linestyle='--', label=f'Median: ${np.median(final_wealth):,.0f}')
    ax.set_xlabel('Final Wealth ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Wealth Distribution')
    ax.legend()

    # Q-Q plot for normality
    ax = axes[0, 1]
    stats.probplot(growth_rates, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Growth Rates')

    # Log-wealth distribution (should be more normal)
    ax = axes[1, 0]
    log_wealth = np.log(final_wealth)
    ax.hist(log_wealth, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Log(Final Wealth)')
    ax.set_ylabel('Frequency')
    ax.set_title('Log-Wealth Distribution')

    # Skewness and kurtosis
    ax = axes[1, 1]
    ax.axis('off')

    skewness = stats.skew(final_wealth)
    kurtosis = stats.kurtosis(final_wealth)
    jarque_bera = stats.jarque_bera(growth_rates)

    text = f"""Distribution Statistics:

    Skewness: {skewness:.3f}
    {'(Positive skew - long right tail)' if skewness > 0 else '(Negative skew - long left tail)'}

    Kurtosis: {kurtosis:.3f}
    {'(Heavy tails - more extreme outcomes)' if kurtosis > 0 else '(Light tails - fewer extremes)'}

    Jarque-Bera Test:
    Statistic: {jarque_bera.statistic:.3f}
    P-value: {jarque_bera.pvalue:.3f}
    {'Reject normality' if jarque_bera.pvalue < 0.05 else 'Cannot reject normality'}
    """

    ax.text(0.1, 0.5, text, fontsize=11, verticalalignment='center')

    plt.tight_layout()
    plt.show()

    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera': jarque_bera
    }
```

### Confidence Intervals

```python
def calculate_confidence_intervals(results, confidence_levels=[0.90, 0.95, 0.99]):
    """Calculate confidence intervals for key metrics"""

    final_wealth = [t.wealth[-1] for t in results.trajectories]
    growth_rates = [t.calculate_growth_rate() for t in results.trajectories]

    intervals = {}

    for confidence in confidence_levels:
        alpha = 1 - confidence

        # Wealth confidence interval
        wealth_lower = np.percentile(final_wealth, alpha/2 * 100)
        wealth_upper = np.percentile(final_wealth, (1 - alpha/2) * 100)

        # Growth rate confidence interval (using bootstrap for better accuracy)
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(growth_rates, size=len(growth_rates), replace=True)
            bootstrap_means.append(np.mean(sample))

        growth_lower = np.percentile(bootstrap_means, alpha/2 * 100)
        growth_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

        intervals[confidence] = {
            'wealth': (wealth_lower, wealth_upper),
            'growth': (growth_lower, growth_upper)
        }

    # Display results
    print("\nConfidence Intervals:")
    for confidence, values in intervals.items():
        print(f"\n{confidence*100:.0f}% Confidence:")
        print(f"  Wealth: [${values['wealth'][0]:,.0f}, ${values['wealth'][1]:,.0f}]")
        print(f"  Growth: [{values['growth'][0]:.2%}, {values['growth'][1]:.2%}]")

    return intervals
```

## Comparative Analysis

### Comparing Insurance Strategies

```python
def compare_strategies(results_dict):
    """Compare multiple insurance strategies"""

    comparison = []

    for strategy_name, results in results_dict.items():
        analyzer = ErgodicAnalyzer(results)

        metrics = {
            'Strategy': strategy_name,
            'Time-Avg Growth': analyzer.time_average_growth(),
            'Ensemble-Avg Growth': analyzer.ensemble_average_growth(),
            'Median Wealth': analyzer.median_terminal_wealth(),
            'Ruin Prob': analyzer.ruin_probability(),
            'Volatility': analyzer.volatility(),
            'Sharpe Ratio': analyzer.sharpe_ratio(),
            'Premium Cost': analyzer.total_premium_paid()
        }

        comparison.append(metrics)

    # Create comparison dataframe
    import pandas as pd
    df = pd.DataFrame(comparison)

    # Add efficiency metrics
    df['Growth per Premium $'] = df['Time-Avg Growth'] / df['Premium Cost'] * 1000000
    df['Risk-Adjusted Growth'] = df['Time-Avg Growth'] / df['Volatility']

    # Rank strategies
    df['Growth Rank'] = df['Time-Avg Growth'].rank(ascending=False)
    df['Risk Rank'] = df['Ruin Prob'].rank(ascending=True)
    df['Overall Score'] = (df['Growth Rank'] + df['Risk Rank']) / 2

    # Display formatted table
    print("\nStrategy Comparison:")
    print(df.to_string(index=False))

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Growth comparison
    ax = axes[0, 0]
    strategies = df['Strategy'].tolist()
    x_pos = np.arange(len(strategies))
    ax.bar(x_pos, df['Time-Avg Growth'] * 100, alpha=0.7, label='Time-Avg')
    ax.bar(x_pos, df['Ensemble-Avg Growth'] * 100, alpha=0.7, label='Ensemble-Avg')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45)
    ax.set_ylabel('Growth Rate (%)')
    ax.set_title('Growth Rate Comparison')
    ax.legend()

    # Risk comparison
    ax = axes[0, 1]
    ax.bar(x_pos, df['Ruin Prob'] * 100, color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45)
    ax.set_ylabel('Ruin Probability (%)')
    ax.set_title('Risk Comparison')

    # Efficiency scatter
    ax = axes[1, 0]
    scatter = ax.scatter(df['Premium Cost'], df['Time-Avg Growth'] * 100,
                        s=1000 * (1 - df['Ruin Prob']),
                        c=df['Overall Score'], cmap='RdYlGn_r')
    ax.set_xlabel('Annual Premium ($)')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_title('Cost-Benefit Analysis (size = safety)')
    for i, txt in enumerate(strategies):
        ax.annotate(txt, (df['Premium Cost'].iloc[i], df['Time-Avg Growth'].iloc[i] * 100))

    # Pareto frontier
    ax = axes[1, 1]
    ax.scatter(df['Volatility'] * 100, df['Time-Avg Growth'] * 100, s=100)
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_title('Risk-Return Frontier')
    for i, txt in enumerate(strategies):
        ax.annotate(txt, (df['Volatility'].iloc[i] * 100, df['Time-Avg Growth'].iloc[i] * 100))

    plt.tight_layout()
    plt.show()

    return df
```

## Time Series Analysis

### Analyzing Wealth Trajectories

```python
def analyze_trajectories(results, n_samples=5):
    """Detailed analysis of wealth trajectories over time"""

    trajectories = results.trajectories
    time_points = len(trajectories[0].wealth)

    # Calculate statistics at each time point
    wealth_matrix = np.array([t.wealth for t in trajectories])

    statistics = {
        'mean': np.mean(wealth_matrix, axis=0),
        'median': np.median(wealth_matrix, axis=0),
        'std': np.std(wealth_matrix, axis=0),
        'percentile_5': np.percentile(wealth_matrix, 5, axis=0),
        'percentile_95': np.percentile(wealth_matrix, 95, axis=0)
    }

    # Convergence analysis
    growth_rates_over_time = []
    for t in range(1, time_points):
        rates = []
        for trajectory in trajectories:
            rate = (np.log(trajectory.wealth[t]) - np.log(trajectory.wealth[0])) / t
            rates.append(rate)
        growth_rates_over_time.append(np.mean(rates))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Wealth fan chart
    ax = axes[0, 0]
    time_axis = np.arange(time_points)

    # Plot percentile bands
    ax.fill_between(time_axis, statistics['percentile_5'], statistics['percentile_95'],
                     alpha=0.3, color='blue', label='90% CI')
    ax.plot(time_axis, statistics['median'], color='red', linewidth=2, label='Median')
    ax.plot(time_axis, statistics['mean'], color='green', linewidth=2,
            linestyle='--', label='Mean')

    # Add sample trajectories
    for i in range(n_samples):
        ax.plot(time_axis, trajectories[i].wealth, alpha=0.3, color='gray')

    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth ($)')
    ax.set_title('Wealth Evolution with Confidence Bands')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Volatility over time
    ax = axes[0, 1]
    rolling_vol = []
    window = min(5, time_points // 4)
    for t in range(window, time_points):
        window_returns = []
        for trajectory in trajectories:
            returns = np.diff(np.log(trajectory.wealth[t-window:t+1]))
            window_returns.extend(returns)
        rolling_vol.append(np.std(window_returns) * np.sqrt(252))

    ax.plot(range(window, time_points), rolling_vol)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annualized Volatility')
    ax.set_title('Rolling Volatility (5-year window)')
    ax.grid(True, alpha=0.3)

    # Growth rate convergence
    ax = axes[1, 0]
    ax.plot(range(1, time_points), growth_rates_over_time)
    ax.axhline(y=growth_rates_over_time[-1], color='red', linestyle='--',
               label=f'Converged: {growth_rates_over_time[-1]:.2%}')
    ax.set_xlabel('Time Horizon (years)')
    ax.set_ylabel('Average Growth Rate')
    ax.set_title('Growth Rate Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ruin probability over time
    ax = axes[1, 1]
    ruin_by_time = []
    for t in range(time_points):
        ruined = sum(1 for traj in trajectories if traj.wealth[t] <= 0)
        ruin_by_time.append(ruined / len(trajectories))

    ax.plot(ruin_by_time)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Ruin Probability')
    ax.set_title('Ruin Probability Evolution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return statistics, growth_rates_over_time
```

## Creating Professional Reports

### Executive Summary Generator

```python
def generate_executive_summary(results, insurance_program):
    """Generate an executive summary of results"""

    analyzer = ErgodicAnalyzer(results)

    summary = f"""
    EXECUTIVE SUMMARY - INSURANCE ANALYSIS RESULTS
    =============================================

    Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
    Simulations Run: {len(results.trajectories):,}
    Time Horizon: {len(results.trajectories[0].wealth)} years

    INSURANCE PROGRAM ANALYZED
    --------------------------
    {insurance_program.summary()}

    KEY FINDINGS
    ------------

    Growth Performance:
    • Time-Average Growth Rate: {analyzer.time_average_growth():.2%}
    • Ensemble-Average Growth: {analyzer.ensemble_average_growth():.2%}
    • Ergodicity Gap: {analyzer.ensemble_average_growth() - analyzer.time_average_growth():.2%}

    Risk Metrics:
    • Probability of Ruin: {analyzer.ruin_probability():.2%}
    • 95% Value at Risk: ${analyzer.value_at_risk(0.95):,.0f}
    • Maximum Drawdown: {analyzer.max_drawdown():.1%}

    Wealth Projections:
    • Median Terminal Wealth: ${analyzer.median_terminal_wealth():,.0f}
    • Mean Terminal Wealth: ${analyzer.mean_terminal_wealth():,.0f}
    • 90% Confidence Interval: [${analyzer.percentile(5):,.0f}, ${analyzer.percentile(95):,.0f}]

    RECOMMENDATION
    --------------
    """

    if analyzer.time_average_growth() > 0.08 and analyzer.ruin_probability() < 0.01:
        summary += "✅ STRONGLY RECOMMENDED - Excellent growth with minimal risk"
    elif analyzer.time_average_growth() > 0.05 and analyzer.ruin_probability() < 0.05:
        summary += "✅ RECOMMENDED - Good balance of growth and risk"
    elif analyzer.time_average_growth() > 0.03:
        summary += "⚠️ ACCEPTABLE - Consider optimizing for better growth"
    else:
        summary += "❌ NOT RECOMMENDED - Insufficient growth protection"

    summary += f"""

    INSIGHTS
    --------
    1. Insurance {'reduces' if analyzer.volatility_reduction() > 0 else 'increases'} volatility by {abs(analyzer.volatility_reduction()):.1%}
    2. Break-even probability: {analyzer.break_even_probability():.1%}
    3. Expected time to double wealth: {analyzer.doubling_time():.1f} years
    4. Insurance ROI: {analyzer.insurance_roi():.1%}

    NEXT STEPS
    ----------
    1. Review detailed metrics in appendix
    2. Compare with alternative structures
    3. Conduct sensitivity analysis
    4. Implement recommended program
    """

    return summary

# Generate and save summary
summary = generate_executive_summary(results, insurance_program)
print(summary)

with open('executive_summary.txt', 'w') as f:
    f.write(summary)
```

### Interactive Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dashboard(results):
    """Create an interactive Plotly dashboard"""

    # Prepare data
    trajectories = results.trajectories
    final_wealth = [t.wealth[-1] for t in trajectories]
    growth_rates = [t.calculate_growth_rate() for t in trajectories]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Wealth Trajectories', 'Growth Distribution',
                       'Risk Metrics', 'Time Evolution'),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'indicator'}, {'type': 'scatter'}]]
    )

    # Wealth trajectories
    for i in range(min(100, len(trajectories))):
        fig.add_trace(
            go.Scatter(y=trajectories[i].wealth, mode='lines',
                      opacity=0.1, showlegend=False),
            row=1, col=1
        )

    # Growth distribution
    fig.add_trace(
        go.Histogram(x=growth_rates, nbinsx=30, name='Growth Rates'),
        row=1, col=2
    )

    # Risk indicators
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=analyzer.ruin_probability() * 100,
            title={'text': "Ruin Risk (%)"},
            gauge={'axis': {'range': [0, 20]},
                   'bar': {'color': "darkred"},
                   'steps': [
                       {'range': [0, 1], 'color': "lightgreen"},
                       {'range': [1, 5], 'color': "yellow"},
                       {'range': [5, 20], 'color': "lightred"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 5}}
        ),
        row=2, col=1
    )

    # Time evolution
    median_wealth = np.median([t.wealth for t in trajectories], axis=0)
    fig.add_trace(
        go.Scatter(y=median_wealth, mode='lines', name='Median Wealth'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title="Insurance Analysis Dashboard",
        showlegend=False,
        height=800
    )

    # Save and show
    fig.write_html("dashboard.html")
    fig.show()

    return fig
```

## Best Practices for Results Presentation

### 1. Lead with Key Insights
- Start with the most important finding
- Use clear, non-technical language
- Provide context for numbers

### 2. Visual Hierarchy
- Use charts for trends
- Tables for precise values
- Color coding for good/bad outcomes

### 3. Comparative Context
- Always compare to baseline (no insurance)
- Show industry benchmarks if available
- Include confidence intervals

### 4. Action-Oriented Conclusions
- Clear recommendations
- Specific next steps
- Implementation timeline

## Next Steps

- [Advanced Scenarios](/Ergodic-Insurance-Limits/tutorials/advanced_scenarios)
- [Case Studies](/Ergodic-Insurance-Limits/docs/user_guide/case_studies)
- [API Documentation](/Ergodic-Insurance-Limits/api/)

---

[← Back to Optimization](/Ergodic-Insurance-Limits/tutorials/optimization_workflow) | [Continue to Advanced →](/Ergodic-Insurance-Limits/tutorials/advanced_scenarios)

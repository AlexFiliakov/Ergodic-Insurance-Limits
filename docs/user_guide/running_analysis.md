---
layout: default
title: Running Analysis
---

# Running Analysis

This guide walks you through performing a complete insurance optimization analysis for your company using the Ergodic Insurance Framework.

## Overview

Running an analysis involves:
1. Setting up your company parameters
2. Defining loss scenarios
3. Configuring insurance options
4. Running simulations
5. Interpreting results

## Step 1: Company Setup

First, gather your company's financial information:

### Required Data
- **Starting Assets**: Total company assets
- **Revenue**: Annual revenue
- **Operating Margin**: Profit margin before insurance
- **Growth Rate**: Expected annual growth rate
- **Working Capital Needs**: Cash requirements

### Example Configuration
```python
company_params = {
    "starting_assets": 10_000_000,  # $10M
    "asset_turnover": 1.0,          # Revenue = 1x assets
    "operating_margin": 0.08,        # 8% margin
    "growth_volatility": 0.15,       # 15% volatility
    "tax_rate": 0.25                 # 25% tax rate
}
```

## Step 2: Loss Modeling

Define your expected loss patterns:

### Loss Categories
1. **Attritional Losses** (high frequency, low severity)
   - Frequency: 3-8 events per year
   - Severity: $3,000 - $100,000

2. **Large Losses** (low frequency, high severity)
   - Frequency: 0.1-0.5 events per year
   - Severity: $500,000 - $50M

### Configuration Example
```python
loss_params = {
    "attritional": {
        "frequency": 5,
        "severity_mean": 25_000,
        "severity_cv": 1.5
    },
    "large": {
        "frequency": 0.3,
        "severity_mean": 2_000_000,
        "severity_cv": 2.0
    }
}
```

## Step 3: Insurance Structure

Configure insurance layers:

### Typical Program Structure
- **Primary Layer**: $0 - $5M (retention: $250K)
- **Excess Layer**: $5M - $25M
- **Catastrophic Layer**: $25M+

### Premium Calculation
```python
insurance_layers = [
    {"limit": 5_000_000, "retention": 250_000, "rate": 0.015},
    {"limit": 20_000_000, "retention": 5_000_000, "rate": 0.008},
    {"limit": 25_000_000, "retention": 25_000_000, "rate": 0.004}
]
```

## Step 4: Running Simulations

Execute the optimization:

### Basic Simulation
```python
from ergodic_insurance import run_optimization

results = run_optimization(
    company_params=company_params,
    loss_params=loss_params,
    insurance_layers=insurance_layers,
    simulation_years=100,
    num_simulations=10_000
)
```

### Advanced Options
- **Time Horizon**: 10-1000 years
- **Monte Carlo Paths**: 1,000-100,000
- **Optimization Objective**: Growth rate, ROE, or ruin probability

## Step 5: Interpreting Results

### Key Metrics to Review

1. **Growth Rate Comparison**
   - Ensemble average (traditional)
   - Time average (ergodic)
   - Difference (the "ergodic gap")

2. **Risk Metrics**
   - Ruin probability
   - Value at Risk (VaR)
   - Conditional Value at Risk (CVaR)

3. **Financial Performance**
   - Return on Equity (ROE)
   - Return on Assets (ROA)
   - Premium as % of revenue

### Decision Criteria

✅ **Good Insurance Program**:
- Time-average growth > ensemble average
- Ruin probability < 1%
- ROE improvement > premium cost

❌ **Poor Insurance Program**:
- Minimal growth improvement
- High ruin probability despite premiums
- ROE degradation

## Practical Example

### Widget Manufacturer Analysis

**Company Profile**:
- Assets: $10M
- Revenue: $10M/year
- Operating margin: 8%
- Loss exposure: Up to $5M

**Analysis Results**:
- Without insurance: 4.2% growth, 15% ruin probability
- With optimal insurance: 6.8% growth, <1% ruin probability
- Premium: $450K/year (4.5% of revenue)
- **Decision**: Implement insurance program

## Common Scenarios

### Scenario 1: High-Growth Company
- Higher volatility requires more insurance
- Focus on catastrophic coverage
- Accept higher retentions for efficiency

### Scenario 2: Stable Company
- Lower insurance needs
- Focus on attritional losses
- Consider self-insurance for predictable losses

### Scenario 3: Distressed Company
- Insurance critical for survival
- Lower retentions essential
- May need to accept higher premium rates

## Troubleshooting

### Issue: Simulation Takes Too Long
**Solution**: Reduce simulation paths or time horizon initially

### Issue: Results Show No Benefit
**Solution**: Check if losses are too small relative to company size

### Issue: Premiums Seem Too High
**Solution**: Review loss assumptions and consider higher retentions

## Next Steps

1. Start with conservative assumptions
2. Run sensitivity analysis on key parameters
3. Compare multiple insurance structures
4. Document your decision rationale

For more advanced techniques, see [Advanced Topics](advanced_topics.md).

## Summary

Running an effective analysis requires:
- Accurate company financial data
- Realistic loss assumptions
- Appropriate insurance structure
- Sufficient simulation paths
- Clear decision criteria

The framework handles the complex mathematics while you focus on business decisions.

# Selecting Excess Insurance Limits: An Ergodic Approach

*A quantitative framework for optimal insurance limit selection using time-average growth theory*

## Executive Summary

- **The Paradigm Shift**: Moving from expected value optimization to time-average growth optimization
- **Key Finding**: Optimal insurance limits can justify premium costs 200-500% above expected losses
- **Target Audience**: Experienced actuaries and risk managers seeking data-driven limit selection
- **Practical Impact**: Framework provides concrete methodology for excess limit optimization
- **Bottom Line**: Insurance transforms from cost center to growth enabler under ergodic analysis

## Basic Conclusions of Ergodic Economics

### What is Ergodic Economics?
- **Definition**: Analysis of economic systems where time averages differ from ensemble averages
- **Core Insight**: Individual entities experience time sequences, not parallel universes
- **Mathematical Foundation**: Multiplicative processes create divergence between growth measures
- **Historical Context**: Kelly criterion, geometric vs arithmetic means, survival bias

### Key Principles for Insurance Applications
- **Time-Average vs Ensemble-Average**: Why traditional expected value analysis fails
- **Multiplicative Wealth Dynamics**: How losses compound differently than gains
- **Survival Probability**: The critical importance of avoiding ruin scenarios
- **Growth Rate Optimization**: Maximizing geometric mean rather than arithmetic mean
- **Path Dependency**: Why the sequence of events matters as much as probabilities

### Implications for Risk Management
- **Insurance as Growth Enabler**: Premium costs justified by volatility reduction
- **Optimal vs Maximum Coverage**: Finding the sweet spot for limit selection
- **Time Horizon Effects**: How optimization changes with planning periods
- **Capital Allocation**: Ergodic approach to risk capital deployment

## The Question We Seek to Resolve

### Problem Statement
**What excess insurance limits should a company purchase to optimize their long-run financial performance?**

### Traditional Approach Limitations
- **Expected Loss Focus**: Overemphasis on actuarial fairness and loss ratios
- **Static Analysis**: Snapshot risk assessment ignoring dynamic growth impacts
- **Ensemble Thinking**: Averaging across many companies rather than single-company time paths
- **Cost-Benefit Myopia**: Treating insurance as pure cost rather than growth investment

### Ergodic Reframing
- **Time-Average ROE**: Optimizing long-term compound growth rates
- **Survival-Contingent Growth**: Accounting for ruin probability in limit selection
- **Dynamic Capital Effects**: How insurance affects reinvestment capacity
- **Volatility as Enemy**: Quantifying the growth drag from earnings variability

### Success Metrics
- **Primary**: Maximized time-average ROE subject to ruin constraints
- **Secondary**: Minimized coefficient of variation in earnings
- **Risk Constraint**: <1% probability of ruin over planning horizon
- **Practical**: Clear decision rules for limit selection across company sizes

## Setup of Our Model Company

### Company Profile: Widget Manufacturing Inc.
- **Industry**: Manufacturing widgets with stable demand
- **Starting Assets**: $10M baseline (scalable for analysis)
- **Business Model**: Asset-intensive manufacturing with predictable margins
- **Growth Strategy**: Reinvestment of profits for organic expansion
- **Risk Profile**: Operational risks scale with revenue, financial risks with assets

### Key Business Characteristics
- **Asset Turnover**: 0.8x (revenue = 0.8 × assets)
- **Operating Margin**: 8% of revenue before losses and taxes
- **Tax Rate**: 25% corporate tax rate
- **Working Capital**: 20% of revenue tied up in operations
- **Reinvestment Rate**: 100% of after-tax profits reinvested for growth

### Why This Model Company?
- **Representativeness**: Typical mid-market manufacturing characteristics
- **Scalability**: Results applicable across company sizes
- **Simplicity**: Focus on core dynamics without sector-specific complications
- **Generalizability**: Framework transferable to other industries with modifications

### Sensitivity Parameters
- **Asset Turnover**: 0.5x to 1.5x for different business models
- **Operating Margin**: 5% to 12% for margin sensitivity analysis
- **Starting Size**: $1M to $100M for scale effects
- **Risk Tolerance**: 0.5% to 2% ruin probability thresholds

## Overview of Our Model Financial Dynamics

### Modeling Balance Sheet and Revenue

#### Balance Sheet Structure
- **Assets**: Starting assets grow through retained earnings reinvestment
- **Liabilities**: Working capital financing (20% of revenue)
- **Equity**: Balancing item, grows with profitable operations
- **Growth Mechanism**: Assets_{t+1} = Assets_t + Retained_Earnings_t

#### Revenue Generation
- **Formula**: Revenue_t = Asset_Turnover × Assets_t
- **Stability Assumption**: Consistent turnover ratio over time
- **Market Dynamics**: No secular growth trends or cyclical effects
- **Capacity Constraints**: Linear relationship between assets and revenue capacity

#### Operating Performance
- **Gross Margin**: 8% of revenue (before losses and insurance)
- **Operating Leverage**: Fixed percentage margin assumption
- **Tax Treatment**: 25% rate applied to net income after losses
- **Reinvestment Policy**: 100% retention for growth funding

### Modeling Losses and Claims

#### Loss Categories
- **Attritional Losses**: High frequency, low severity operational losses
  - Frequency: 3-8 events per year
  - Severity: $3K-$100K per event
  - Distribution: Poisson frequency, Lognormal severity
- **Large Losses**: Low frequency, high severity catastrophic events
  - Frequency: 0.1-0.5 events per year  
  - Severity: $500K-$50M per event
  - Distribution: Poisson frequency, Pareto severity

#### Frequency Scaling with Revenue
- **Core Principle**: Loss frequency increases with business activity
- **Attritional Scaling**: Frequency ∝ Revenue^0.8 (economies of scale in safety)
- **Large Loss Scaling**: Frequency ∝ Revenue^0.6 (exposure grows slower than revenue)
- **Baseline Calibration**: Frequencies calibrated to $10M revenue company
- **Economic Justification**: More operations → more exposure → more claims

#### Loss Severity Distributions
- **Attritional Losses**: Lognormal(μ=10.5, σ=1.2) → mean ~$50K
- **Large Losses**: Pareto(α=1.8, x_min=$500K) → heavy tail characteristics
- **Independence**: Severity independent of company size (external factors dominate)
- **Correlation Structure**: 25% correlation between attritional and large loss years

#### Claims Payment Timing
- **Immediate Payment**: Losses paid in year incurred (no reserves)
- **Insurance Recovery**: Excess payments recovered instantly
- **Cash Flow Impact**: Direct reduction in available reinvestment capital

### Simplifying Assumptions

#### No Inflation Effects
- **Rationale**: Focus on real growth dynamics, not monetary effects
- **Loss Trends**: No systematic inflation in claim costs over time
- **Premium Stability**: Insurance costs remain constant in real terms
- **Impact**: Results represent real economic optimization

#### No Market Dynamics
- **Interest Rates**: No discounting or investment income on reserves
- **Economic Cycles**: No recession/expansion effects on margins or losses
- **Industry Competition**: Stable market share and pricing power
- **Regulatory Environment**: No changing compliance costs or requirements

#### No Dynamic Strategy Adjustments
- **Fixed Parameters**: Operating margins, asset turnover remain constant
- **Static Insurance**: No mid-simulation changes to coverage levels
- **Passive Management**: No tactical adjustments based on performance
- **Consistent Policy**: Same decision rules applied throughout simulation

#### Other Key Simplifications
- **No Debt**: Pure equity financing model
- **No Dividends**: 100% earnings retention for growth
- **No Acquisitions**: Organic growth only through reinvestment
- **No Technology Change**: Stable production functions over time
- **Perfect Information**: Complete knowledge of loss distributions

## Performance Metrics

### Return on Equity (ROE)

#### Calculation Methodology
- **Formula**: ROE_t = Net_Income_t / Equity_t
- **Components**: (Revenue × Margin - Losses - Insurance_Premiums - Taxes) / Equity
- **Time-Average ROE**: Geometric mean of annual ROE over simulation horizon
- **Ergodic Focus**: Long-run compound growth rate optimization

#### Why Time-Average ROE Matters
- **Growth Reality**: Companies experience sequences, not averages
- **Compounding Effects**: Volatility reduces geometric mean growth
- **Survival Conditional**: Growth only matters if company survives
- **Investment Decision**: Shareholders care about compound returns

#### ROE Decomposition Analysis
- **Operating Component**: Base profitability from business operations  
- **Loss Impact**: How claims reduce available equity returns
- **Insurance Effect**: Premium costs vs volatility reduction benefits
- **Tax Shield**: After-tax analysis for realistic decision-making

### Risk of Ruin

#### Definition and Calculation
- **Ruin Event**: Equity falls below zero at any point in simulation
- **Measurement**: Percentage of simulation paths reaching ruin
- **Time Horizon**: Probability over full simulation period (typically 100-1000 years)
- **Constraint**: Target <1% ruin probability for acceptable risk levels

#### Economic Interpretation
- **Business Continuity**: Ability to survive adverse loss sequences
- **Capital Adequacy**: Minimum equity buffer for operations
- **Stakeholder Protection**: Avoiding bankruptcy costs and disruption
- **Growth Prerequisite**: Cannot compound returns if business fails

#### Sensitivity to Insurance Levels
- **Coverage Impact**: How different limits affect ruin probability
- **Optimal Tradeoff**: Premium costs vs survival probability improvement
- **Diminishing Returns**: Marginal benefit curves for additional coverage
- **Threshold Effects**: Critical coverage levels for meaningful protection

### Additional Performance Measures

#### Coefficient of Variation
- **Formula**: CV = StdDev(ROE) / Mean(ROE)
- **Volatility Measure**: Earnings stability across simulation years
- **Insurance Benefit**: Premium costs justified by CV reduction
- **Growth Impact**: Lower CV correlates with higher geometric mean returns

#### Value at Risk (VaR)
- **Definition**: Worst-case ROE at 95% and 99% confidence levels
- **Tail Risk**: Extreme loss scenario impacts
- **Insurance Effect**: How coverage truncates loss distributions
- **Management Tool**: Risk budget allocation decisions

#### Time to Ruin Analysis
- **Survival Curves**: Probability of surviving to various time horizons
- **Early vs Late Risk**: How ruin probability changes over time
- **Capital Building**: How equity growth affects survival probability
- **Insurance Timing**: When coverage provides maximum survival benefit

## Simulation Approach

### Monte Carlo Framework

#### Simulation Design
- **Path Generation**: 10,000+ independent company life cycles
- **Time Horizon**: 100-1000 year simulations for long-term analysis
- **Random Seeds**: Reproducible results with documented seed values
- **Performance Target**: Full simulation suite in <30 minutes

#### Ensemble Size: 10,000+ Paths
- **Statistical Power**: Sufficient observations for robust percentile estimates
- **Ruin Probability**: Accurate measurement of rare events (1% threshold)
- **Confidence Intervals**: Tight bounds on performance metrics
- **Computational Efficiency**: Balance between accuracy and runtime

#### Path Independence
- **No Cross-Path Effects**: Each simulation represents independent company
- **Market Assumptions**: No systemic risks or correlated economic shocks
- **Pure Time-Series**: Focus on individual company dynamics over time
- **Ensemble vs Time**: Clear distinction between cross-sectional and temporal analysis

### Limit Selection Testing

#### Coverage Level Grid
- **Primary Layer**: $0 deductible, limits from $1M to $10M
- **Excess Layers**: $5M xs $5M, $15M xs $10M, $25M xs $25M
- **Premium Rates**: Market-based pricing (1.5% to 0.4% by layer)
- **Comprehensive Testing**: All combinations for optimization

#### Optimization Algorithm
- **Objective Function**: Maximize time-average ROE
- **Constraint**: Ruin probability ≤ 1%
- **Search Method**: Grid search with interpolation for fine-tuning
- **Validation**: Out-of-sample testing on separate simulation runs

### Results Presentation

#### Primary Outputs
- **Optimal Limits Table**: Coverage recommendations by company size
- **Performance Metrics**: ROE, ruin probability, CV for each configuration
- **Sensitivity Analysis**: How results change with key parameter variations
- **Cost-Benefit Charts**: Premium costs vs performance improvement

#### Visualization Strategy
- **Efficiency Frontier**: ROE vs ruin probability for different coverage levels
- **Heat Maps**: Performance across coverage combinations
- **Time Series**: Sample paths showing insurance impact over time
- **Distribution Plots**: ROE distributions with and without optimal coverage

#### Actionable Recommendations
- **Decision Rules**: Clear guidelines for limit selection by company characteristics
- **Implementation Steps**: Practical advice for actuaries and risk managers
- **Monitoring Framework**: KPIs for ongoing coverage optimization
- **Future Enhancements**: Extensions for sector-specific applications

---

*This blog post will provide a comprehensive, quantitative framework that transforms how actuaries think about excess limit selection, moving beyond traditional expected loss analysis to embrace ergodic optimization principles that maximize long-term company growth and survival.*
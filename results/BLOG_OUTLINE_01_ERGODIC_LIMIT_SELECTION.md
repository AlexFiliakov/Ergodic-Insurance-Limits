# Selecting Excess Insurance Limits: An Ergodic Approach
*A three-part quantitative framework for optimal insurance limit selection using time-average growth theory*

## Series Overview

**Part 1: The Ergodic Foundation** - Theory, principles, and problem setup
**Part 2: Methodology and Results** - Monte Carlo framework and optimal limit findings
**Part 3: Implementation Guide** - Practical decision rules and industry applications

---

# Part 1: The Ergodic Foundation

*Transforming how actuaries think about excess insurance from cost center to growth enabler*

## Executive Summary (Part 1)

- **The Paradigm Shift**: Moving from expected value optimization to time-average growth optimization
- **Key Insight**: Individual companies experience time sequences, not ensemble averages
- **Model Subject**: Widget Manufacturing Inc. - our guinea pig for thousand-year stress testing
- **The Big Question**: What excess limits should they buy to maximize long-term growth?
- **Part 1 Preview**: Foundation principles that will shock traditional actuarial thinking

## What is Ergodic Economics?

*[Content as written - condensed version of current section]*

- **Core Concept**: Time averages ≠ ensemble averages for business growth
- **Casino Analogy**: You play one sequence of spins, not parallel universe averages
- **Mathematical Reality**: Multiplicative wealth processes make volatility the enemy
- **Insurance Transformation**: From necessary evil to growth accelerator

## Key Principles for Insurance Applications

*[Condensed version focusing on core concepts]*

### Time-Average vs Ensemble-Average: The Critical Distinction
- **The Fatal Flaw**: Traditional analysis optimizes for parallel companies, not your company
- **Mathematical Framework**: Growth follows multiplicative dynamics where volatility kills compounding
- **Numerical Reality**: 15% average returns with high volatility < 12% with low volatility

### Path Dependency: Sequence Matters
- **Early Loss Impact**: $15M loss in year 1 ≠ $15M loss in year 10
- **Compound Growth Effect**: Early losses destroy the base for all future compounding
- **Claims Timing**: Long-tail development creates different impacts than immediate payments

### Growth Rate Optimization: Beyond Loss Minimization
- **New Objective**: Maximize geometric mean ROE, not minimize expected losses
- **Survival Constraint**: <1% ruin probability over planning horizon
- **Premium Justification**: Volatility reduction can justify 300%+ loss cost ratios

## Implications for Risk Management

Meet **Widget Manufacturing Inc.**, our brave volunteer for actuarial experimentation. They make widgets (obviously), have $10M in assets, and are about to embark on time travel across parallel universes.

### Our Model Company Profile
- **Business**: Manufacturing widgets with predictable demand (widgets are always needed!)
- **Size**: $10M assets generating $8M revenue (0.8x turnover)
- **Margins**: 8% operating margin (before the inevitable losses)
- **Growth Strategy**: Reinvest everything and hope for the best
- **Risk Tolerance**: Conservative management (1% ruin probability maximum)

### The Experimental Setup
We're going to subject Widget Manufacturing to:
- **10,000+ parallel life simulations** (because one life isn't enough data)
- **1,000-year time horizons** (longer than most civilizations last)
- **Realistic loss distributions** (Poisson frequency, heavy-tailed severity)
- **Multiple insurance scenarios** (from "wing it" to "insure everything")

### Our Delightfully Unrealistic Assumptions
To keep this analysis tractable, we assume:
- **No inflation** (egg prices stay the same forever)
- **No business cycles** (eternal stable growth)
- **No strategic pivots** (Widget Manufacturing will make the same widgets forever)
- **Perfect information** (omniscient insurers know loss distributions exactly, but don't know the outcomes)
- **No debt** (equity financing only)

*Don't worry - these simplifications actually make our results MORE conservative. Reality would likely favor insurance even more strongly.*

## The Question We Seek to Resolve

### The Central Challenge
**What excess insurance limits should Widget Manufacturing Inc. purchase to optimize their long-run financial performance?**

Traditional actuarial thinking would focus on:
- Expected annual losses vs. premium costs
- Loss ratios and actuarial fairness
- Industry benchmarks and peer comparisons

### The Ergodic Reframing
Our analysis will instead optimize:
- **Time-average ROE** over 1,000-year simulations
- **Survival probability** across adverse loss sequences
- **Geometric growth rates** that account for volatility drag
- **Path-dependent effects** of loss timing on compound returns

### What We'll Discover in Parts 2 & 3

*Spoiler alert for those who can't wait:*

**Part 2** will reveal our Monte Carlo simulation results, including:
- Optimal excess attachment points for different company sizes
- Why paying $250K premiums for $50K expected losses makes perfect sense
- Heat maps showing the ROE/survival tradeoff across coverage levels
- Specific recommendations for Widget Manufacturing's optimal program

**Part 3** will provide practical implementation guidance:
- Decision trees for limit selection by company characteristics
- Sensitivity analysis for key business parameters
- Integration with existing risk management frameworks
- Extensions to other industries beyond widget manufacturing

### The Cliffhanger Question

After 1,000 years and 10,000 simulations, what will we recommend for Widget Manufacturing Inc.?

- **Option A**: Minimal coverage ($25M excess limit, $240K premium)
  - Saves premium dollars but retains severe loss exposure
  - Expected ROE: 14.6% but ruin probability: 8.2%

- **Option B**: Conservative coverage ($200M excess attachment, $380K premium)
  - Expensive premium but comprehensive protection
  - Expected ROE: 12.4% but ruin probability: 0.3%

- **Option C**: The ergodic optimum (spoiler: somewhere in between)
  - Balances growth and survival for maximum long-term wealth
  - The sweet spot that traditional analysis would miss

**Which option maximizes Widget Manufacturing's time-average growth over the next millennium?**

*Find out in Part 2, where we unleash 10,000 Monte Carlo simulations to stress-test these scenarios across every conceivable loss sequence...*

---

# Part 2: Methodology and Results

*Coming next: Monte Carlo framework, simulation results, and optimal limit recommendations*

## Overview (Part 2 Preview)

### Monte Carlo Framework
- **Simulation Architecture**: 10,000+ paths × 1,000 years of company evolution
- **Loss Modeling**: Realistic frequency/severity distributions calibrated to manufacturing risks
- **Coverage Testing**: Comprehensive grid search across attachment points and limits
- **Performance Metrics**: Time-average ROE, ruin probability, coefficient of variation

### Key Results to be Revealed
- **Optimal Limits Table**: Specific recommendations by company size and risk tolerance
- **Efficiency Frontier**: ROE vs. ruin probability tradeoffs across coverage levels
- **Sensitivity Analysis**: How results change with margin, turnover, and loss assumptions
- **Cost-Benefit Quantification**: Precise premium justification for each coverage layer

### Surprising Findings Preview
- Why $500K premiums for $125K expected losses maximizes long-term growth
- How attachment points should scale with company assets (hint: it's not linear)
- The critical coverage threshold where ruin probability drops dramatically
- Why traditional loss cost ratios are completely irrelevant for optimization

---

# Part 3: Implementation Guide

*Coming last: Practical decision rules, industry applications, and actionable frameworks*

## Overview (Part 3 Preview)

### Decision Framework
- **Limit Selection Algorithm**: Step-by-step process for determining optimal coverage
- **Company Sizing Rules**: How recommendations scale across different business sizes
- **Risk Tolerance Calibration**: Adjusting recommendations for different ruin probability targets
- **Implementation Checklist**: Practical steps for actuaries and risk managers

### Industry Applications
- **Sector Adaptations**: Modifying the framework for different industries
- **Parameter Estimation**: How to calibrate loss distributions from limited data
- **Integration Methods**: Incorporating ergodic analysis into existing risk management
- **Monitoring Framework**: KPIs for ongoing coverage optimization

### Future Enhancements
- **Dynamic Strategies**: Adjusting coverage as companies grow
- **Portfolio Effects**: Multi-company optimization for large organizations
- **Advanced Modeling**: Incorporating economic cycles, inflation, and market dynamics
- **Technology Integration**: Automated optimization tools and dashboards

---

*This three-part series will fundamentally transform how actuaries approach excess limit selection, providing both theoretical foundation and practical implementation guidance for ergodic insurance optimization.*

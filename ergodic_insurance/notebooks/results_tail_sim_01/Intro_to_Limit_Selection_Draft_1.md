# Introduction to Limit Selection

A single employee error triggers a product recall affecting 2 million units. Total losses: $380 million. Your company purchased a $100 million liability policy because it "covered losses at the 99.9th percentile." The actuary who recommended that limit followed best practices, used sophisticated models, and got the probability calculations right.

And it doesn't matter. The remaining $280 million in uncovered losses wipes out 70% of your equity. Your credit rating collapses. Your expansion plans are cancelled. Competitors acquire market share while you're managing the crisis. Ten years later, you're still recovering from an event that happened in a single afternoon.

This is the tail wagging the dog. Not figuratively. Literally. A single extreme event, occurring once in a company's lifetime, dominates every strategic decision that came before it. Traditional insurance analytics, focused on expected values and ensemble averages, systematically underestimate how much these rare catastrophes matter to the one company experiencing them.

This is why insurance limit selection matters more than most actuaries realize. It's not about expected values or ensemble averages. It's about protecting the actual trajectory of one company operating in one timeline. And that requires a fundamentally different analytical framework.

*Want to see the results first? [Skip to Results & Analysis](#results--analysis).*

## The Problem

Traditional actuarial limit selection optimizes ensemble averages. But companies don't operate across ensembles, they operate in time. This post explores how that distinction changes everything about optimal insurance limits.

## The Ergodic Perspective

Ergodicity economics asks a deceptively simple question: does the average outcome across many parallel universes match the average outcome one entity experiences over time? For most business decisions with tail risk, the answer is no.

> **For Students**: Ensemble averages calculate what happens "on average" if we ran 1,000 parallel companies. Time averages calculate what happens to one company over 1,000 periods. With tail risk, these diverge dramatically.

When a company faces potential catastrophic losses, the ensemble average calculated across thousands of simulated companies may show strong expected returns.

But the time average (what one actual company experiences) can be devastated by a single extreme event that occurs early in its history.

Insurance transforms these dynamics by redistributing tail risk across the ensemble, allowing individual companies to track closer to time-average growth rather than suffering the full consequences of statistical outliers.

For this analysis, I simulate corporate trajectories over 50 years to allow ergodic effects to fully materialize.

While a 25-year horizon might be more realistic for strategic planning, the longer timeframe lets us observe how different insurance strategies affect growth convergence (or divergence) over time.

Each corporate configuration is tested across 250,000 scenarios, providing a robust distribution of potential 50-year outcomes.

## Simulation Framework

The framework builds on the methodology detailed in my research paper at [https://mostlyoptimal.com/research](https://mostlyoptimal.com/research), which introduced an ergodic approach to deductible selection. Here, I apply that framework to analyze insurance limits.

### Loss Structure

The simulation employs a four-tier loss structure that reflects the reality of corporate risk exposure:

| Loss Type | Frequency Model | Severity Model | Purpose/Example |
|-----------|----------------|----------------|-----------------|
| **Attritional** | Poisson (High) | Lognormal | Routine operational losses (property damage, small claims) |
| **Large** | Poisson (Medium) | Lognormal | Significant but manageable incidents (product liability, injuries) |
| **Catastrophic** | Poisson (Low) | Pareto | Low-frequency events introducing meaningful tail risk (major recalls, lawsuits) |
| **Extreme** | Threshold-triggered | Generalized Pareto (GPD) | Replaces the tail with thicker distribution at specified threshold (true black swans) |

Revenue serves as the exposure base for all loss frequencies, scaling risk with company size.

This four-tier approach captures the full spectrum from routine operational losses to true black swans, excepting loss correlations.

### Financial Model

#### Corporate Accounting

The balance sheet simulation includes realistic (though intentionally simplified) corporate accounting.

Revenue is deterministic and proportional to assets, making net income stochastic purely through loss realizations. This isolates the impact of insurance decisions from other business volatility.

#### Insurance Structure

The insurance structure consists of:
- A per-occurrence deductible (held constant at $500K for this analysis, but other deductibles don't materially sway the analysis of limits)
- A single per-occurrence limit (the variable under study)

#### Insurance Pricing

Insurance pricing assumes carriers can accurately estimate the underlying loss distribution; a strong but necessary assumption for this analysis.

Premiums are loaded using a loss ratio to account for risk margin, profit load, and administrative expenses. For this particular study, I use a 60% loss ratio.

To maintain analytical clarity, I've removed time value of money effects, treating all calculations on an approximate present-value basis.

### Key Assumptions & Limitations

This framework makes several simplifying assumptions:
- Uniform loss ratios across all insurance layers (reality would show varying margins by layer)
- Perfect insurer knowledge of loss distributions
- Uncorrelated loss events
- Deterministic revenue growth
- No inflation or discounting effects

These simplifications allow us to isolate the ergodic effects of limit selection, but they also mean **these specific results do not generalize directly to real corporate structures**. However, the directional insights and underlying mechanisms should transfer to more complex scenarios, and the framework can be adapted to study individual cases.

## Understanding Tail Risk

Before diving into results, we need to establish what "tail risk" means in this context.

The tail of a probability distribution represents extreme outcomes: events far from the mean that occur rarely but carry enormous consequences.

Tail "thickness" determines how much probability mass sits in these extremes and how quickly probabilities decay as events become more severe.

A thick-tailed distribution (like the Pareto we use for catastrophic losses) assigns non-trivial probability to events many standard deviations from the mean.

This has profound implications for insurance: the thicker the tail, the more value high limits provide.

For this study, I restrict analysis to distributions with finite mean and finite variance.

While this excludes some "pathological" heavy-tailed distributions (like power laws with infinite variance), it still captures substantial tail risk and keeps the mathematics tractable.

Future posts will explore those more extreme scenarios.

The Shape parameter in our results controls tail thickness: lower Shape values produce thicker tails with higher probability of extreme events.

> **For Practitioners**: When estimating tail risk for your business, the Shape parameter is critical. A Shape of 0.5 vs 0.9 can mean the difference between recommending a $250M limit and a $500M+ limit for the same company. Historical loss data and industry benchmarks help calibrate this, but uncertainty remains substantial.

## Results & Analysis

The visualization below shows annualized growth rates over 50 years plotted against per-occurrence insurance limits:

![Time-Average Growth vs Insurance Limit](cache/selected_growth_v_limit_ded_500K_lr_0p6_thresh_0.0005_scale_1.png)

### Reading the Visualization

The grid structure spans two key dimensions:
- **Columns** (left to right): Increasing initial capitalization from $25M to $100M
- **Rows** (top to bottom): Increasing tail thickness (Shape parameters 0.5, 0.7, and 0.9)

Each subplot displays:
- Distribution bands (shaded blue/purple regions) showing various percentiles (10th, 25th, 50th/median, 75th, 90th, 99th)
- Navy color represents the region of the median
- Red line with markers indicates the mean

### Pattern Analysis

Several striking patterns emerge:

**1. Mean-Median Divergence**

As limits increase, we observe a consistent pattern: median growth rates remain relatively flat or decline slightly, while the separation between mean and median narrows. This is the ergodic effect in action.

With low limits, catastrophic losses that exceed coverage devastate individual company trajectories, creating massive dispersion in outcomes.

The mean gets pulled down by these tail events while the median (representing the "typical" outcome) stays relatively high.

High limits eliminate this downside, compressing the distribution and bringing mean and median closer together.

This represents a fundamental tradeoff: **higher limits sacrifice some median performance to protect against worst-case scenarios**.

From an ensemble perspective (comparing many companies), lower limits look attractive. From a time-average perspective (one company's actual path), higher limits provide crucial protection.

**2. Capitalization Effects**

The relationship between company size and optimal limits is non-obvious but actuarially sound:

- **$25M capitalization**: Across all tail assumptions, limits in the $350M-$500M range appear optimal. These companies are vulnerable to mid-sized catastrophic events and need substantial coverage relative to their capital base.

- **$50M capitalization**: Tail assumptions become crucial. With thinner tails (Shape 0.5), limits around $250M suffice. With thicker tails (Shape 0.9), optimal limits push toward $500M or higher. The risk profile materially changes the calculus.

- **$75M-$100M capitalization**: These larger companies show significant limit appetite, especially under thick-tail scenarios. Optimal limits can rationally exceed $500M.

This pattern aligns with the capital preservation logic: companies with higher capitalization have more to lose from extreme events.

While large companies often self-insure smaller losses (choosing high deductibles), they simultaneously purchase high limits to protect their substantial capital base.

The limited liability structure of modern corporations reinforces this dynamic. A $25M company can "only" lose $25M, but a $100M company faces quadruple that downside.

High limits become proportionally more valuable as capitalization increases.

**3. Tail Thickness Amplification**

Moving down the rows (increasing tail thickness), we see divergence patterns amplify.

With thin tails (Shape 0.5, top row), all capitalizations show relatively modest mean-median gaps and limited sensitivity to insurance limits beyond certain thresholds.

With thick tails (Shape 0.9, bottom row), the distributions spread dramatically.

The mean drops significantly relative to the median, and the value of high limits becomes unmistakable.

The 10th percentile outcomes diverge sharply from median outcomes, illustrating how tail events can permanently impair growth trajectories.

This underscores a crucial point: **tail risk assessment is not an academic exercise**. Getting the tail wrong by even modest amounts can lead to catastrophically wrong limit selections.

### Interpreting the Mean Limits (Red Line)

The red line traces the mean annualized growth rate after 50 years and 250,000 scenarios. This trajectory reveals intuitive patterns:

- Optimal limits increase with capitalization
- Optimal limits increase with tail thickness
- The rate of increase accelerates in the thick-tail scenarios
- Even "small" companies with heavy tail exposures should consider surprisingly high limits

## Key Insights & Implications

### What Generalizes

Several insights from this analysis should transfer to more complex scenarios:

1. **The ergodic tradeoff is real**: Higher limits protect time-average growth at the expense of ensemble-average efficiency. This tradeoff exists regardless of specific assumptions.

2. **Capitalization drives limit appetite**: Larger companies rationally purchase higher limits to protect capital bases, even as they may self-insure smaller losses. This isn't about risk aversion, it's about preserving optionality for future growth.

3. **Tail uncertainty matters enormously**: Small changes in tail assumptions drive large changes in optimal limits. Companies uncertain about their tail exposure should err toward higher coverage.

4. **The value of limits compounds over time**: The 50-year horizon reveals cumulative effects that single-period analysis misses. A "slightly suboptimal" limit selection can compound into substantially different growth trajectories.

### What Doesn't Generalize

These specific results depend heavily on the configuration choices:

- The tail assumptions
- The $500K deductible assumption
- The 60% uniform loss ratio
- The deterministic revenue model
- The absence of loss correlations
- The present-value simplification

Real companies facing real insurance decisions need to model their specific exposure profiles, capital structures, and risk appetites. These results provide directional guidance, not prescriptive answers.

### Practical Implications for Actuaries and Risk Managers

This analysis suggests several shifts in how we might approach limit selection:

1. **Time horizons matter**: Evaluating limits over multi-decade horizons rather than single policy periods reveals different optimal choices. Strategic insurance buying should reflect strategic time horizons.

2. **Distribution moments aren't enough**: Mean and variance don't fully capture tail dynamics. Actuaries need to engage seriously with distribution shape, particularly in the tails.

3. **Capital structure integration**: Limit selection shouldn't happen in isolation from capital planning. The optimal insurance program depends fundamentally on balance sheet structure and growth objectives.

4. **Ensemble thinking can mislead**: Traditional actuarial methods that focus on expected values across portfolios can produce poor guidance for individual entity decisions under tail risk.

## Limitations & Future Research

This analysis represents an early iteration of the framework. Several enhancements would make it production-ready:

- **Layered loss ratios**: Different insurance layers typically carry different margins. High excess layers often have lower loss ratios than primary layers due to basis risk and data uncertainty.

- **Time value of money**: Incorporating proper discounting and inflation would affect the relative value of premium expenditure versus loss protection.

- **Loss correlations**: Many catastrophic events trigger multiple loss types simultaneously. Correlation structures would likely increase the value of high limits.

- **Stochastic revenue**: Allowing revenue to vary with economic cycles, competitive dynamics, and recovery from losses would create additional interactions with insurance strategy.

- **GPU parallelization**: The 250,000 scenarios per configuration are computationally intensive. GPU acceleration would enable faster iteration and more complex models.

Beyond technical enhancements, several research questions emerge:

- What does the demand side of the limits look like under layered insurance structures? What is the surface of optimal limits when varying loss ratios?
- Can we identify heuristics or decision rules that approximate optimal limit decisions without full tail modeling?
- How do these dynamics change with infinite-variance distributions (Generalized Pareto with Shape of 1.0+)?

My immediate next research focus will explore tail thickness more deeply and investigate layered loss ratio structures that better reflect real insurance markets.

## Conclusion

Insurance limit selection, viewed through an ergodic lens, reveals patterns that traditional actuarial analysis can miss. The tradeoff between median and mean performance, the interaction between capitalization and limit appetite, and the amplifying effects of tail thickness all emerge clearly from long-term simulation.

Remember that $380M product recall from the opening? A company using this framework would have seen why a $100M limit left them catastrophically exposed. The mean growth trajectories for companies with Shape 0.9 and $50M capitalization clearly show that optimal limits approach or exceed $400M—precisely what's needed to avoid the tail-wagging-dog scenario that devastated our hypothetical company.

For actuaries and risk managers, the message is clear: **insurance limits aren't just about expected loss coverage. They're about protecting time-average growth trajectories from tail events that can permanently impair corporate prospects**.

The specific limits that emerge from this analysis apply only to the configuration studied. But the framework and approach generalize. Companies serious about strategic risk management should model their specific exposures, run long-term simulations, and evaluate insurance decisions as a dynamic strategy over multi-decade horizons rather than single policy periods.

**Want to explore your own limit selection scenarios?** The complete methodology is at [mostlyoptimal.com/research](https://mostlyoptimal.com/research), and the framework tutorial is at [mostlyoptimal.com/tutorial](https://mostlyoptimal.com/tutorial).

I welcome your feedback in the comments: Where do these insights align with your experience? Where do the simplifications break down most severely? What other scenarios would be valuable to explore?

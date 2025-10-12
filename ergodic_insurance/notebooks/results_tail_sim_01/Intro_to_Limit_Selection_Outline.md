# Introduction to Limit Selection

## Overview of Insurance Limits
- To put simply, limits are the maximum an insurance company will cover on a per-occurrence or in aggregate.
- In this post, I study only the effects of per-occurrence limits, but the aggregate structure should generally have similar dynamics.

## Tail Risk
- Describe briefly the tail of a distribution, implications of thick tails for insurance.
- We will restrict our analysis in this post to finite mean and finite variance, and in later posts we'll explore "pathological" cases where mean and variance cease to exist.

## Ergodicity Economics
- Briefly describe ergodicity economics.
- Explain that we study corporate strategy over 50 years to let ergodic effects settle, even though 25-year horizon would be more realistic.
- We study the distribution of 50-year outcomes across 250,000 scenarios in each company configuration.

## Insurance Company Setup
- Four-tier structure for losses with revenue as the exposure base:
  - Attritional losses (Poisson Frequency / Lognormal Severity)
  - Large losses (Poisson Frequency  / Lognormal Severity)
  - Catastrophic losses (Poisson Frequency / Pareto Severity)
  - Extreme losses (Generalized Pareto Distribution that attaches to the tail of the underlying distribution at a specified threshold and replaces severity with a thicker tail)
- Realistic yet simplified financial accounting with a full balance sheet simulation. Full description of the framework is available at https://mostlyoptimal.com/research
- Deterministic annual revenue proportional to assets, with net income stochastic only by virtue of stochastic losses.
- Insurance structure is a per-occurrence deductible with a single per-occurrence limit
- Insurance is priced by assuming the carrier can accurately guess the underlying loss distribution, then loaded using a Loss Ratio to account for risk margin, profit load, and administrative expenses.
- To keep the analysis simple, we make assumptions that remove the impact of time value of money, so all future calculations are roughly on present-value basis.

## Results
- Analyze the plot in "ergodic_insurance\notebooks\results_tail_sim_01\cache\selected_growth_v_limit_ded_500K_lr_0p6_thresh_0.0005_scale_1.png" and come up with conclusions from the study.
  - This plots Annualized Growth Percentage over 50 years versus Per-Occurrence Insurance Limit
  - The grid of plots is constructed as follows: rows represent increasing tail thickness going down, while columns represent increasing initial company capitalization.
  - Blue region contains the median
  - Red line represents the mean
    - As you can see, thicker tails drag the mean lower through uncovered insurance losses.
  - Increased limits trade median performance for mean performance, which is driven largely by worst 10% of losses.
  - Since modern corporations have limited liability, companies with higher capitalization stand to lose more to large losses, hence limit appetite increases with capitalization, unlike deductibles (Large companies may choose to self-insure against small losses, but select high limits for capital preservation in case of extreme losses)
  - Under this specific configuration, we see that for a company with $25M capitalization, a limit of around $350M to $500M is rational under all of the selected tail assumptions.
    - Meanwhile, for a $50M company, tail assumption is a big driver of limit selection (ligher tail can get away with $250M limit, while heavier tail assumptions can rationally purchase significantly more)
    - Finally, we observet hat $100M companies and above have significantly higher limit appetite with heavier tails, and can rationally purchase insurance limits upwards of $500M
  - Tail risk is important and needs to be understood thoroughly when dealing with high limits. Thus, I plan to explore heavy tails in future posts.

## Future Research
- There are many ways the underlying framework can be enhanced. Some immediate needs to make it ready for production include:
  - Loss ratios variable by insurance layers
  - Inflation and time value of money
  - Loss correlations and copulas
  - GPU parallelization
  - Stochastic revenue

Nevertheless, the current iteration of the framework allows us to gain a deeper understanding into the nature of deductibles and limits. My first research paper, mentioned earlier, provides an approach to deductible selection, while this post explores limit selection. My future research will focus on tail thickness and demand-side layered loss ratios.

# Ergodic Insurance Under Volatility: Preliminary Results Analysis

## Analysis of Results

Results presented here focus on the \$5M capitalization with Asset Turnover Ratio (ATR) of 1.0 for detailed analysis, though the qualitative conclusions hold directionally across the full parameter grid (\$5M--\$25M capitalization and ATR 0.8--1.2).

---

## Finding 1: Insurance Value Compounds Over Time

![Year-by-year growth lift facets](output/publication/interesting/year_by_year_growth_lift_facets.png)

Across all capitalizations tested (\$5M, \$10M, \$25M) and all asset turnover ratios (0.8, 1.0, 1.2), every insured configuration outperforms the uninsured baseline, and the advantage grows exponentially with time (linear on log-wealth scale). This growth-rate advantage compounds multiplicatively over the firm's lifetime. It demonstrates the core ergodic thesis: when analyzed through time-averages rather than ensemble expectations, insurance transforms from a cost center (negative expected value in the single-period ensemble view) into a growth engine (positive compounding advantage in the multi-period time-average view).

Guaranteed Cost (GC) option dominates all configurations, and the effects scale with capitalization in the modeled region. Lower deductibles produce larger growth advantages at every time horizon and every capitalization level. The ranking \$0 > \$100K > \$250K > \$500K never reverses in the modeled region. This is a direct consequence of the Volatility Tax mechanism detailed in Finding 2. We expect the effect to reverse at higher capitalizations, however, and for some level of retention to become optimal.

---

## Finding 2: The Volatility Tax Overwhelms Premium Savings

![Volatility tax vs premium savings](output/publication/interesting/volatility_tax_vs_premium_savings.png)

This chart decomposes the growth rate into two components for each insurance configuration: the **Expected-Value Growth** (the growth rate one would predict from a deterministic, expected-loss analysis) and the **Volatility Tax** (the additional penalty or bonus arising from stochastic fluctuations). The Actual Growth (blue bars) is the net of these two effects.

### What is the Volatility Tax?

In any multiplicative growth process (a firm's year-over-year asset accumulation is inherently multiplicative), variance directly erodes the compound growth rate. This is a fundamental result from stochastic calculus: for a process with expected return $\mu$ and variance $\sigma^2$, the time-average (geometric) growth rate is approximately $\mu - \sigma^2/2$. The $\sigma^2/2$ term is the **Volatility Tax**.

Consider a simple example: a firm earning +50% one year and -50% the next does *not* break even. Starting at \$100, it grows to \$150, then falls to \$75, a cumulative loss of 25% despite a 0% average arithmetic return. The difference between the arithmetic average (0%) and the geometric outcome (-13.4% annualized) is the volatility tax.

For a business carrying insurance risk, the relevant variance includes both operational revenue volatility (here modeled at $\sigma = 0.15$ via GBM) and the variance of retained losses. Insurance reduces retained-loss variance, thereby reducing the volatility tax.

### The Reversal

The expected-value analysis (shown in the green bars) tells one story: No Insurance has the highest expected growth (255 bps) because it avoids paying any premium, and higher deductibles save premium relative to GC. A traditional ensemble-average analysis would therefore recommend *against* insurance, or augmented with some notion of risk aversion, at minimum recommend high deductibles to minimize premium expenditure.

The time-average analysis tells the opposite story: GC achieves the highest *actual* compound growth (235 bps) despite having the lowest expected growth (208 bps). The volatility tax of -132 bps for the uninsured path wipes out its 47-bps premium-savings advantage over GC, leaving it 111 bps behind in actual growth.

Uniquely, GC shows a positive volatility effect (+27 bps). This means the actual stochastic growth exceeds the deterministic estimate. This occurs for two technical reasons:

1. **Complete loss transfer.** With a $0 deductible, all losses above zero are covered. The firm pays a fixed, deterministic premium each year. This eliminates retained-loss variance entirely, leaving only the uninsurable revenue volatility from the GBM process.

3. **Tax shield efficiency.** Under accrual-basis tax accounting, a constant premium expense is fully deductible every year with no timing mismatch. In contrast, variable retained losses can create years where deductions exceed income (generating net operating losses with limited carryforward utility) followed by years of full taxation on profits. This asymmetric tax treatment penalizes variance, an effect that GC insurance eliminates.

This chart makes the case that traditional insurance purchasing decisions based on expected-cost minimization are not just slightly wrong but directionally wrong. The rational time-average optimizer buys more insurance, not less, because the volatility tax on retained risk overwhelms any premium savings. We anticipate this conclusion may weaken for larger firms (where the loss-to-asset ratio is smaller, reducing relative variance) and for firms with lower operational volatility. These extensions are left for future work.

---

## Finding 3: Survival and Median Outperformance

![Wealth fan chart](output/publication/interesting/wealth_fan_chart.png)

This side-by-side fan chart compares the distribution of 50-year wealth trajectories for GC-insured ($0 deductible) versus uninsured firms, both on a log scale. Two identically-parameterized sets of firms, facing the same loss events and revenue shocks across 250K scenarios, end up in dramatically different places depending on whether they purchased insurance.

### Survival

- **Without insurance: 37.8% insolvency by year 50, with a median time-to-ruin of 20 years.** More than one in three uninsured firms fails during the simulation horizon. The insolvency dots appear evenly across time, indicating that ruin is not primarily a late-life phenomenon, but can occur in any year due to an unforeseen catastrophe.

- **With GC insurance: 0.01% insolvency.** Virtually every insured path survives the full 50 years. This is mechanically expected because coverage is unlimited, so there is no scenario in which a single loss (or sequence of losses) can overwhelm the insurance protection. The rare insolvencies that do occur are attributable to extreme revenue declines under the GBM process rather than uninsured losses.

The high uninsured ruin rate (38%) is consistent with the heavy-tailed catastrophic loss model (Pareto $\alpha = 2.5$, $x_m = \$5M$). For a firm with only \$5M in assets, even a single catastrophic event can be terminal. Insurance absorbs these tail events completely, converting a survival gamble into a near-certainty.

### Median Outperformance

The more surprising result is visible in the median lines:

- The **GC-insured median path** (dark blue line, left panel) reaches approximately \$25M by year 50.
- The **uninsured median path** (dark orange line, right panel, survivors only) reaches approximately \$14M--\$15M by year 50.

The insured median is roughly **1.7x the uninsured median**, and this comparison is biased in favor of the uninsured, because it's conditioned on survival. The uninsured median excludes the 37.8% of paths that went bankrupt. If we included those failed paths (at zero terminal wealth), the uninsured median would be near zero.

The fan width also tells a story: the insured fan (5th--95th percentile) is remarkably tight, while the uninsured fan is wide and includes many paths that drop below \$1M before recovering. This reduced dispersion is the manifestation of volatility tax reduction. Insurance compresses the distribution of outcomes, and under multiplicative dynamics, compression benefits the geometric mean.

---

## Finding 4: Outcome Distribution

![Insurance outcome distribution](output/publication/interesting/insurance_outcome_distribution_basic.png)

This histogram shows the distribution of log-wealth advantage (insured minus uninsured) across the subset of paths where **both** the insured and uninsured firms survived all 50 years. The x-axis is the difference in log-assets at year 50; positive values mean the insured firm ended wealthier in terms of assets.

The median log-wealth advantage is 0.30 (corresponding to 1.4x wealth). This means the typical insured firm has 40% more assets than its uninsured counterpart.

The histogram results are summarized in the following table:

| Metric | Value |
|---|---|
| Saved from ruin | 37.8% (94,607 of 250,000 paths) |
| Wealthier at year 50 (all paths) | 86.2% (215,538 of 250,000 paths) |
| Avg gain when insurance helps | +$7M (across 215,538 paths) |
| Avg loss when insurance hurts | -$2M (across 34,462 paths) |
| Gain/Loss asymmetry | 3.4x |

The 3.4x gain/loss ratio means that the expected dollar benefit when insurance helps is more than three times the expected dollar cost when it hurts.

---

### Limitations and Extensions

Real insurance programs have policy limits and aggregates. Introducing finite limits would reduce but not eliminate the GC advantage, as the most catastrophic tail events would revert to uninsured status.

In practice, premium loading varies across the insurance tower and premiums adjust year-to-year based on loss experience, market conditions, and the insurer's own capital constraints. Dynamic premium modeling could attenuate the growth advantage by introducing premium volatility.

The analysis anticipates that the GC advantage will diminish for larger firms (\$50M+ capitalization) where the loss-to-asset ratio decreases and operational volatility has a proportionally smaller impact on growth rates. This is supported by the observation that even within the tested range, the per-year growth lift for GC decreases as capitalization increases (on a relative, bps basis), though it remains positive and economically significant.

---

## Conclusions

Under ergodic (time-average) analysis with realistic stochastic volatility ($\sigma = 0.15$), Guaranteed Cost insurance delivers a persistent compound growth advantage of approximately 111 basis points per year over the uninsured baseline for small manufacturers (\$5M capitalization). This advantage compounds over time in the form of growth-rate premium that widens multiplicatively over the firm's horizon. The main driver of this is volatility reduction via the elimination of Volatility Tax, not expected claim recoveries.

While traditional expected-value analysis recommends high deductibles or self-insurance, time-average analysis recommends Guaranteed Cost (lowest possible deductible). This finding scales with vulnerability and is strongest for firms where loss-to-asset variance is highest (small capitalization, high asset turnover, high operational volatility).

### Reframing Insurance as a Growth Catalyst, Not a Cost

For small-to-medium manufacturers operating in volatile markets, these results suggest that the conventional wisdom of "retain more risk to save premium" is not just suboptimal but wealth-destroying in the long run. The Volatility Tax on retained risk compounds silently, year after year, eroding the very premium savings that motivated the higher retention. CFOs evaluating insurance purchasing decisions should:

1. Reframe insurance as a growth investment, not a cost. Insurance is not "money lost to premiums," it is money invested in volatility reduction that pays compound dividends.

2. Evaluate insurance decisions over the firm's full strategic planning horizon, not a single policy period.

3. Recognize that survival probability is a critical input to long-run growth. The 37.8% ruin rate for the uninsured baseline in our example is not an abstract tail risk; it represents a realistic probability of business failure over a multi-decade horizon. Insurance doesn't just improve expected outcomes; it keeps the firm alive to experience those outcomes.

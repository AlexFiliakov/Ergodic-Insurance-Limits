# Analysis Brainstorming: Volatility Simulation Results

## Simulation Parameter Space (36 Configurations)

| Dimension | Values | Count |
|-----------|--------|-------|
| Initial Assets (Cap) | $5M, $10M, $25M | 3 |
| Asset Turnover (ATR) | 0.8, 1.0, 1.2 | 3 |
| Per-Occurrence Deductible | $0, $100K, $250K, $500K | 4 |
| EBITA Margin | 12.5% (fixed) | 1 |
| Loss Ratio | 70% (fixed) | 1 |
| Revenue Volatility | 15% (fixed) | 1 |

Each configuration has 250,000 simulations over 50 years, with full annual trajectory data.
Paired insured/uninsured runs via Common Random Numbers (CRN).

---

## A. Single-Configuration Illustrative Analyses

These are deep dives into one or two configurations, designed to build reader intuition before presenting systematic results. They answer: *"What does it actually look like when a company operates over 50 years with and without insurance?"*

### A1. Wealth Trajectory Fan Chart

**What it shows:** A fan chart of asset trajectories over 50 years for a single (Cap, ATR, Deductible) configuration. The median path is a bold line; shaded bands show the 10th-90th, 25th-75th percentile envelopes. Overlay the paired no-insurance fan chart in a contrasting color.

**Why it engages the audience:** Actuaries understand distributions, but most insurance analyses present single-year snapshots. Showing 50 years of wealth evolution makes the *compounding* effect of insurance tangible. The visual gap between the insured and uninsured fans is the ergodic argument made visible. Senior risk managers immediately see that insurance isn't a cost---it's a trajectory shifter.

**Recommended configuration:** Start with the most dramatic case: $5M Cap, ATR=1.2, Ded=$100K (high revenue intensity, small company, moderate retention). This maximizes the visual contrast. Then show $25M Cap as a counterpoint where insurance matters less.

**Implementation notes:**
- Subsample ~1,000 paths for the transparent spaghetti lines behind the fan
- Use log scale on the y-axis so both surviving and ruined paths are visible
- Mark the ruin threshold clearly with a horizontal dashed line
- Consider a dual-panel (insured left, uninsured right) with shared y-axis

---

### A2. Ruin Probability Over Time (Survival Curves)

**What it shows:** Kaplan-Meier-style survival curves plotting the fraction of simulations still solvent at each year, for all four deductible levels within a single (Cap, ATR) pair.

**Why it engages the audience:** Ruin probability is the single metric that most directly challenges expected-value thinking. Showing that ruin *accumulates* over time---even for profitable companies---hits a nerve. Most actuarial analyses quote annual ruin probability; showing the 50-year cumulative version is a wake-up call. The separation between deductible curves directly quantifies how much retention appetite affects survival.

**Recommended configuration:** Use $5M Cap, ATR=1.0 as the base case (mid-revenue, small company). The deductible curves should separate clearly here.

**Implementation notes:**
- Plot all 4 deductible lines plus the no-insurance baseline (5 curves)
- Add confidence bands (bootstrap from the 250K paths) to show statistical precision
- Include a zoomed inset for the first 10 years where most separation occurs
- Mark any crossover points where a lower deductible overtakes a higher one

---

### A3. Annual Profit/Loss Decomposition Waterfall

**What it shows:** A stacked area chart or waterfall diagram showing the annual breakdown for a single path: revenue, operating profit, loss events (retained + insured), insurance premium, tax, dividends, and change in equity.

**Why it engages the audience:** This is the "day in the life" view that CFOs and risk managers relate to. It grounds the abstract simulation in recognizable P&L mechanics. When a catastrophic loss hits in year 17 and the insurance recovery kicks in, the audience sees exactly how the balance sheet absorbs the shock. It also makes premium cost visually proportional to its role, countering the "insurance is too expensive" reflex.

**Recommended approach:** Pick 2-3 specific paths from the $5M Cap, ATR=1.0, Ded=$100K configuration:
1. A "typical" path near the median outcome
2. A path where a catastrophic loss nearly causes ruin but insurance saves the company
3. A path where self-insurance fails (compare the same path index in the no-insurance run)

**Implementation notes:**
- The CRN pairing means the same sim_id experiences the same losses under both policies
- Use the full trajectory data (annual assets, equity, revenue, claim amounts, insurance recoveries)
- Annotate major loss events with callout labels

---

### A4. Distribution of Terminal Wealth (Year 50)

**What it shows:** Overlaid histograms or kernel density estimates of log(final assets) for insured vs. uninsured at a single configuration. Annotate the mean of the log (the time-average growth rate) and the log of the mean (the ensemble average).

**Why it engages the audience:** This is the core ergodicity visualization. The gap between the mean-of-logs and the log-of-means is Jensen's inequality made concrete. Actuaries will immediately recognize that the "average outcome" is misleading because no single company experiences the average---every company experiences one path. The insured distribution shifts right (higher median) despite the premium cost, which is counterintuitive under expected-value logic.

**Recommended configuration:** $5M Cap, ATR=1.0 to establish the baseline case, then optionally overlay $25M Cap to show how the distributions converge as capitalization grows.

**Implementation notes:**
- Use log scale on x-axis (or plot log-wealth directly)
- Mark the 5th percentile and ruin fraction explicitly
- A KDE overlay is cleaner than raw histograms at 250K samples
- Vertical lines for: geometric mean, arithmetic mean, median

---

### A5. Insurance Value Distribution (CRN-Paired)

**What it shows:** For each of the 250K paired simulations, compute Delta_i = log(final_assets_insured_i) - log(final_assets_uninsured_i). Plot the distribution of Delta_i.

**Why it engages the audience:** Most analyses report the *average* benefit of insurance. This shows the *distribution* of benefit across scenarios. Some paths benefit enormously (catastrophic loss absorbed), some paths are net losers (paid premium but never had a large claim), and the shape of this distribution tells a more honest story. The fact that the mean of Delta is positive despite many negative realizations is a compelling narrative for experienced practitioners.

**Recommended configuration:** Do this for each deductible level at $5M Cap, ATR=1.0. The distribution shape should shift meaningfully across deductible levels.

**Implementation notes:**
- Handle ruined paths carefully: if uninsured ruins but insured survives, Delta is effectively infinite. Consider reporting separately: "fraction of paths where insurance prevented ruin" and "conditional benefit given both survived"
- This analysis is only valid within a single (Cap, ATR) pair where insured and uninsured share the same CRN
- A box plot across deductible levels is a useful companion chart

---

## B. Cross-Configuration Comparative Analyses

These use the full 36-configuration grid to reveal systematic patterns. They answer: *"How does the optimal insurance strategy change with company characteristics?"*

### B1. Optimal Deductible Heatmap (Cap x ATR)

**What it shows:** A 3x3 heatmap where rows are Cap ($5M, $10M, $25M), columns are ATR (0.8, 1.0, 1.2), and the cell color/label indicates which deductible maximizes the time-average growth rate.

**Why it engages the audience:** This is the paper's core actionable result. An actuary at a major broker advising a client can look at this and say: "Given your capitalization and revenue intensity, the ergodic framework recommends this retention level." It directly challenges the one-size-fits-all deductible recommendations that dominate current practice. The pattern should show that larger companies (higher Cap) can tolerate higher deductibles, while revenue-intensive businesses (higher ATR) need more protection.

**Implementation notes:**
- Color by optimal deductible level (categorical colormap)
- Annotate each cell with the growth rate at the optimum AND the growth rate at the worst deductible, so readers see the cost of getting it wrong
- Consider a second heatmap showing the "growth rate penalty" for choosing guaranteed cost ($0 ded) vs. optimal

---

### B2. Growth Lift Surface (3D or Contour)

**What it shows:** For each deductible level, plot the growth lift (insured - uninsured time-average growth rate) as a function of Cap and ATR. This creates four surfaces or contour plots (one per deductible).

**Why it engages the audience:** It reveals the interaction between company characteristics and insurance value. The contour lines show "iso-value" curves where insurance provides equal benefit---a novel way to segment client portfolios. Brokers and consultants constantly segment clients; giving them an analytically grounded segmentation is directly useful.

**Implementation notes:**
- 3x3 grid is coarse for smooth contours; consider interpolation with a caveat
- Alternatively, present as a grouped bar chart (3 Cap groups, ATR as x-axis within each, bars colored by deductible)
- Include error bars based on the standard error of the mean growth rate (SE is tiny at 250K sims, but the gesture of rigor matters for this audience)

---

### B3. Risk-Return Frontier by Configuration

**What it shows:** Scatter plot with x-axis = ruin probability (at year 25 or 50) and y-axis = mean time-average growth rate. Each dot is one of the 36 configurations. Color by deductible, shape by Cap, size by ATR.

**Why it engages the audience:** The Pareto frontier here is the efficient frontier of insurance strategy. Configurations on the frontier represent optimal trade-offs; configurations below it are dominated. This framing maps directly to portfolio theory, which many actuaries and risk managers already understand. It also makes the case that some self-insurance decisions are *inefficient*---accepting more ruin risk without proportionally higher growth.

**Implementation notes:**
- Consider using year-25 ruin probability for a more practically relevant horizon
- Label frontier points with their (Cap, ATR, Ded) combination
- Draw the frontier line connecting non-dominated solutions
- Highlight the "guaranteed cost" ($0 ded) and "self-insured" (no insurance) extremes

---

### B4. Deductible Efficiency Curve (Growth Lift per Dollar Retained)

**What it shows:** For each (Cap, ATR) pair, plot the growth lift as a function of deductible, normalizing by the expected retained loss at that deductible. This gives a "growth lift per dollar of risk retained" metric.

**Why it engages the audience:** Actuaries think in terms of efficiency and cost-benefit. This chart answers: "At what point does the next dollar of retention stop earning its keep?" The curve should show diminishing returns as deductible increases, with the inflection point depending on company size. This is directly actionable for structuring client programs.

**Implementation notes:**
- The expected retained loss at each deductible is computed during the pricing phase (already stored)
- Plot 9 curves (3 Cap x 3 ATR) on the same chart, or use a 3x1 panel (one per Cap)
- The x-axis is deductible amount; y-axis is growth lift / E[retained loss]

---

### B5. Insurance Value Decay with Scale

**What it shows:** Line chart showing how the growth lift at each deductible level decays as capitalization increases, with separate lines for each ATR level.

**Why it engages the audience:** This directly quantifies the intuition that "larger companies need less insurance." But it also reveals *how fast* the decay happens and whether ATR modulates it. If high-ATR companies retain more insurance value at larger sizes, that's a finding brokers can use to differentiate advice between asset-light and asset-heavy businesses.

**Implementation notes:**
- Three panels (one per ATR) or overlaid with different line styles
- Express growth lift in basis points for intuitive scale
- Add a horizontal reference line at zero to show where insurance becomes value-neutral

---

## C. CRN-Enabled Unique Analyses

These exploit the Common Random Numbers design to perform analyses that are impossible with independent simulation runs. **The cross-configuration CRN pairing needs verification** (see Technical Note below), but within each (Cap, ATR) pair, the insured/uninsured comparison is fully CRN-paired.

### C1. "Life or Death" Attribution: Paths Saved by Insurance

**What it shows:** For each configuration, count the paths where the uninsured company went bankrupt but the insured company survived. Decompose these "saved" paths by the year of divergence (when did the uninsured version fail?).

**Why it engages the audience:** This is the most emotionally compelling analysis. It answers: "How many times, out of 250,000 alternate histories, did insurance literally save the company?" Even if the average growth lift is modest, knowing that insurance prevented ruin in 12% of scenarios (hypothetical) has enormous persuasive power. Risk managers and boards of directors respond strongly to this framing.

**Implementation notes:**
- For each sim_id, check insolvency_year for insured vs. uninsured
- Classify into: both survived, both failed, insured survived/uninsured failed, insured failed/uninsured survived
- The last category (insured failed but uninsured survived) should be rare/zero and would indicate a premium burden problem
- Present as a stacked bar chart across deductible levels

---

### C2. Conditional Path Analysis: What Happens During a Catastrophic Loss?

**What it shows:** Filter for paths where a catastrophic loss (>$2M, say) occurs in a specific year window. Compare the subsequent wealth trajectories of insured vs. uninsured versions of those same paths.

**Why it engages the audience:** This is a "stress test" analysis that directly mirrors how risk managers think. They ask: "What happens when the big one hits?" Instead of hypothetical stress scenarios, this uses actual simulated catastrophes and shows the recovery arc with and without insurance. The CRN pairing makes this a controlled experiment, not a statistical average.

**Implementation notes:**
- Use the annual claim_amounts data to identify paths with large loss events
- Select paths where a single loss > some threshold occurs
- Plot the insured and uninsured wealth trajectories for these specific paths (small multiples or spaghetti plot with transparency)
- Show the distribution of "recovery time" for insured paths vs. uninsured failures

---

### C3. Cross-Configuration CRN: Same Storm, Different Ships (If Verified)

**What it shows:** If CRN pairing holds across configurations (same sim_id = same loss draws regardless of Cap/ATR/Ded), then for a single sim_id, trace the wealth trajectory across all 36 configurations. This shows the same sequence of business events unfolding for companies of different sizes with different insurance choices.

**Why it engages the audience:** This is a powerful storytelling device. "Company A and Company B face identical loss events over 50 years. Company A has $5M in assets and a $100K deductible. Company B has $25M and self-insures. Here's what happens." This makes the abstract optimization tangible and personal. Consulting actuaries can use this exact framing when presenting to boards.

**Technical note on CRN cross-configuration validity:** The CRN reseeds at every `(sim_id, year)` boundary using `SeedSequence([crn_base_seed, sim_id, year])`. This seed is *independent of the configuration parameters* (Cap, ATR, Ded). Therefore, the underlying random draws (loss frequency, severity, revenue shocks) should be identical across all configurations for the same sim_id. **However,** the loss frequency is scaled by current revenue (`2.85 * current_revenue / 10M`), and since current_revenue diverges across configurations as paths evolve, the *realized loss events* will differ after year 1. The random draws are the same, but the revenue-dependent frequency transform means the actual losses depend on the path.

**What this means practically:** Year-1 losses are identical across configurations (same starting revenue for same ATR), but subsequent years diverge because revenue evolves differently. This is actually *more realistic*---it captures that a larger company generates more exposure. The CRN pairing is still valuable for controlling the randomness, but "same storm, different ships" is approximate after year 1. **Within the insured/uninsured pair for the same (Cap, ATR), the pairing is exact because both start with the same revenue and the revenue GBM uses the same seed.**

**Recommended approach:**
1. First verify by loading two configs with same (Cap, ATR) but different deductibles, same sim_id, and checking year-1 losses
2. If year-1 losses match, the cross-config pairing is confirmed for at least the first year
3. For the paper, focus on the exact insured/uninsured pair within a config, and use cross-config comparison qualitatively

---

### C4. Variance Reduction Verification

**What it shows:** Compare the standard error of the growth lift estimate using CRN-paired estimation vs. independent estimation. Report the variance reduction factor.

**Why it engages the audience:** This is a methodological contribution that strengthens the paper's credibility with actuarial researchers. It demonstrates that the simulation is designed with statistical rigor, not just brute-force scale. A variance reduction factor of 10x (hypothetical) means 250K CRN-paired sims are equivalent to 2.5M independent sims.

**Implementation notes:**
- Paired estimate: mean of (growth_rate_insured_i - growth_rate_uninsured_i)
- Independent estimate: mean(growth_rate_insured) - mean(growth_rate_uninsured)
- Compare SE of each
- Report the ratio of variances

---

## D. Temporal Dynamics Analyses

These exploit the full 50-year trajectory data. They answer: *"How does insurance value evolve over time? Is there a horizon effect?"*

### D1. Year-by-Year Growth Lift (Insurance Value Over Time)

**What it shows:** For each year t, compute the mean log-wealth of insured paths minus the mean log-wealth of uninsured paths. Plot this "cumulative insurance value" over 50 years.

**Why it engages the audience:** Most insurance analyses are single-period. This shows that insurance value *compounds*---the gap widens over time as the preserved capital earns returns. If the curve accelerates, that's the ergodic argument in its purest form: insurance isn't a per-year cost-benefit; it's a compounding advantage. CFOs who think in terms of NPV will immediately grasp this.

**Recommended configuration:** Do this for all 4 deductible levels at $5M Cap, ATR=1.0. Also compare across the three Cap levels to show how the curve flattens for larger companies.

**Implementation notes:**
- Compute at each year: mean(log(assets_insured_i[t])) - mean(log(assets_uninsured_i[t]))
- This is the geometric mean ratio of wealth levels
- Include a confidence band (bootstrap or analytic)
- Mark the year where the cumulative premium paid equals the cumulative insurance benefit (breakeven)

---

### D2. Wealth Distribution Evolution (Ridge Plot or Animation)

**What it shows:** A ridge plot (or animated GIF) showing the distribution of log(assets) at years 1, 5, 10, 20, 30, 40, 50 for a single configuration. Overlay insured and uninsured.

**Why it engages the audience:** This is visually striking and tells a complete story in one chart. The distributions start identical (year 0), then gradually separate as insurance shifts the insured distribution rightward and the uninsured distribution develops a heavier left tail (ruin). The evolving shape shows how the distribution becomes increasingly non-Gaussian over time, undermining normal-distribution assumptions.

**Implementation notes:**
- Use ridge plot (joy plot) with 7 rows for the 7 time points
- Two colors: insured and uninsured
- Normalize each row's distribution for visual comparison
- Alternative: animation showing distribution morphing year by year

---

### D3. Ruin Hazard Rate Over Time

**What it shows:** The *incremental* ruin probability per year (hazard rate), not the cumulative. This shows whether the risk of ruin is front-loaded or whether it persists throughout the 50-year horizon.

**Why it engages the audience:** Actuaries are trained to think in hazard rates. If the ruin hazard is roughly constant, it implies a memoryless process. If it increases over time, it suggests that the compounding effect of losses erodes the balance sheet. If it decreases (companies that survive get stronger), that's evidence of a "survival of the fittest" effect. Each pattern has different implications for insurance structuring.

**Implementation notes:**
- Hazard rate at year t = (number of paths that first ruin at year t) / (number of paths still solvent at year t-1)
- Plot as a step function for each deductible level
- Overlay insured and uninsured

---

### D4. Breakeven Time: When Does Insurance Pay for Itself?

**What it shows:** For each configuration, compute the year at which the median insured wealth first exceeds the median uninsured wealth. Before this year, the premium burden dominates; after it, the ruin-prevention benefit dominates.

**Why it engages the audience:** This directly addresses the CFO objection: "How long until I see a return on this insurance spend?" The answer (which should vary by company size and retention) gives practitioners a concrete talking point. If the breakeven is 5-10 years for a $5M company, that's within a typical strategic planning horizon.

**Implementation notes:**
- Compute median(assets_insured[t]) and median(assets_uninsured[t]) at each year
- Find the crossover point
- Report as a table: (Cap, ATR, Ded) -> breakeven year
- Also compute using geometric mean instead of median for robustness

---

## E. Ergodicity-Specific "Money Charts"

These are the analyses that make the ergodicity argument vivid and memorable. They should form the visual core of the paper.

### E1. Ensemble Average vs. Time Average Divergence

**What it shows:** For a single configuration, plot two lines over 50 years:
1. The arithmetic mean of final wealth across all 250K paths at each year (ensemble average)
2. The exponential of the mean of log-wealth across all paths at each year (geometric mean, i.e., the typical path)

The lines start together and diverge, with the ensemble average pulled up by a shrinking number of extreme winners.

**Why it engages the audience:** This is the *flagship visualization* for the entire ergodicity argument. Every reader should come away remembering this chart. It shows that the "expected value" (ensemble average) is a fantasy experienced by no individual company. The geometric mean---what a typical company actually experiences---is lower, and the gap grows. This is the intellectual foundation for why insurance is rational even when priced above expected losses.

**Recommended configuration:** $5M Cap, ATR=1.0, no insurance (to show the raw effect), then overlay the insured version to show that insurance closes the gap.

**Implementation notes:**
- Include a third line: the median path (should track close to the geometric mean)
- Annotate the divergence with an explanatory callout
- Use linear scale to emphasize the divergence (log scale would compress it)
- In the paper, this should be Figure 1 or 2---it sets up the entire argument

---

### E2. Growth Rate Distribution: Arithmetic vs. Geometric

**What it shows:** Two overlaid histograms of per-path annualized growth rates: (a) the arithmetic annual return, and (b) the geometric (compounded) annual return. The arithmetic mean of the geometric returns is the time-average growth rate.

**Why it engages the audience:** This makes the Kelly criterion argument visual. The geometric return is always less than or equal to the arithmetic return for any individual path (with equality only at zero variance). Insurance, by reducing variance, closes this gap---meaning the *typical* company grows faster despite paying premium. This is counterintuitive enough to be memorable.

**Implementation notes:**
- The growth_rates array in SimulationResults should contain the geometric (time-average) growth rates
- For arithmetic return, compute as mean of annual log-returns (equivalent for continuous compounding)
- Show both the insured and uninsured distributions on the same axes
- Annotate the means with vertical lines

---

### E3. The "Probability of Outperformance" Chart

**What it shows:** For each (Cap, ATR), plot the fraction of CRN-paired paths where the insured version outperforms the uninsured version (in terms of final wealth) as a function of deductible. Also show the average outperformance conditional on outperformance, and the average underperformance conditional on underperformance.

**Why it engages the audience:** This reframes insurance from "does it help on average" to "how likely is it to help, and by how much when it does vs. doesn't?" Even if insurance helps in only 60% of paths, if the help in those paths is 3x larger than the harm in the other 40%, the decision is clear. This asymmetry is the essence of the ergodic argument, and presenting it this way respects the sophistication of the actuarial audience.

**Implementation notes:**
- For each paired sim_id: outperformance_i = final_assets_insured_i > final_assets_uninsured_i
- Probability = mean(outperformance_i)
- Conditional values: mean(delta | delta > 0) vs. mean(delta | delta < 0)
- Plot as a grouped bar chart or table

---

### E4. Premium Multiple Rationality Frontier

**What it shows:** For each configuration, compute the ratio of premium to expected insured loss (the premium multiple). Plot growth lift vs. premium multiple. Draw the frontier showing the maximum premium multiple at which insurance still enhances the time-average growth rate.

**Why it engages the audience:** This directly addresses the paper's claim that "premiums can exceed expected losses by 200-500% while enhancing growth." By plotting the actual frontier, the paper provides evidence for *exactly how much* premium loading is rational, and how this varies by company characteristics. This is a finding that challenges insurance pricing orthodoxy and will generate discussion.

**Implementation notes:**
- Premium multiple = premium / E[insured loss] = 1 / loss_ratio = 1/0.7 = 1.43 for all configs at LR=0.7
- Since LR is fixed at 0.7, the premium multiple is constant across all configs. This limits the chart's power.
- **Workaround:** Instead, show the implied maximum tolerable loss ratio (i.e., at what LR would the growth lift drop to zero?) by extrapolation or qualitative argument. Or reference this as motivation for a follow-up run varying LR.
- Alternative framing: compute the "insurance efficiency" = growth lift / premium cost, and show how this varies across the grid

---

## F. Narrative / Storytelling Analyses

These are designed to support specific rhetorical goals in the paper.

### F1. "Your Peer Benchmark Is Wrong" Comparison

**What it shows:** Take two configurations with the same ATR (same industry, same revenue intensity) but different capitalizations. Show that the optimal deductible differs substantially. This directly undermines peer benchmarking---the practice of setting retentions based on what similar-revenue companies do.

**Why it engages the audience:** Challenging peer benchmarking is provocative because it's the default approach at most brokers. If the paper can demonstrate that a $5M company and a $25M company *in the same industry* should have different retentions even with identical revenue, that's a publishable finding that will be discussed in practice.

**Implementation notes:**
- Select ATR=1.0 as the common "industry"
- Compare Cap=$5M vs. Cap=$25M
- Show optimal deductible, growth lift, and ruin probability side by side
- Frame as: "Two companies with identical revenue ($Xm) but different balance sheets should make fundamentally different insurance decisions"

---

### F2. "The Cost of Getting It Wrong" Sensitivity Table

**What it shows:** For each (Cap, ATR), compute the growth rate at the optimal deductible and at each suboptimal deductible. Report the growth penalty (in basis points or percentage) for choosing wrong.

**Why it engages the audience:** Actuaries and risk managers need to justify the effort of this analysis. If the penalty for choosing the wrong deductible is 5 basis points of annual growth, it's academic curiosity. If it's 50-100 basis points, it's material to shareholder value. The magnitude of the penalty *is* the business case for the framework.

**Implementation notes:**
- Table format: rows = (Cap, ATR), columns = deductible levels, cells = growth rate
- Bold the optimal in each row
- Add a column for "max penalty" (best - worst deductible)
- Translate growth rate differences to wealth multiples over 50 years: a 50bp difference compounds to a 28% wealth gap over 50 years

---

### F3. "Revenue Intensity as a Hidden Risk Driver" Analysis

**What it shows:** Compare companies with the same capitalization but different ATRs. At ATR=0.8, the company has $4M revenue on $5M assets; at ATR=1.2, it has $6M revenue. The latter generates more losses (frequency scales with revenue) and should need more insurance.

**Why it engages the audience:** Asset turnover is rarely considered in insurance purchasing decisions. Demonstrating that it materially affects optimal retention gives actuaries a new dimension to incorporate into their analyses. It's particularly relevant for asset-light businesses (tech, services) vs. asset-heavy businesses (manufacturing, real estate).

**Implementation notes:**
- Fix Cap=$10M and compare ATR=0.8 vs. 1.0 vs. 1.2
- Show growth lift, ruin probability, and optimal deductible for each
- Frame the revenue difference in dollar terms: $8M vs. $10M vs. $12M annual revenue
- Connect to the DuPont decomposition (ROE = margin x turnover x leverage)

---

## G. Interactive Exploration Tools (Pre-Publication)

These are for your own exploration to identify the most interesting findings before committing to publication figures.

### G1. Parameter Space Explorer Dashboard

**What it builds:** A Plotly or Panel dashboard with dropdown selectors for Cap, ATR, and Deductible. Selecting a configuration shows:
- Wealth fan chart
- Growth rate distribution
- Ruin probability curve
- Key metrics (mean growth rate, median terminal wealth, ruin probability at years 10/25/50)

**Why it's useful:** With 36 configurations, you need a fast way to scan for the most interesting cases. The dashboard lets you flip through configurations and identify which ones to feature in the paper.

### G2. Paired Path Browser

**What it builds:** A tool that lets you select a sim_id and see the insured vs. uninsured wealth trajectories side by side for a specific configuration. Filter by: paths with catastrophic events, paths where ruin occurred, paths near the median, etc.

**Why it's useful:** For selecting the specific paths to feature in analyses A3 and C2. The CRN pairing makes individual paths meaningful, but you need a way to find the compelling ones.

---

## H. Recommended Priority Order

### Tier 1: Must-Have for the Paper
1. **E1** - Ensemble vs. Time Average Divergence (the flagship chart)
2. **B1** - Optimal Deductible Heatmap (the core actionable result)
3. **A1** - Wealth Trajectory Fan Chart (builds reader intuition)
4. **A2** - Survival Curves (ruin accumulation over time)
5. **C1** - "Life or Death" Attribution (emotional impact)

### Tier 2: Strong Candidates
6. **D1** - Year-by-Year Growth Lift (compounding argument)
7. **B3** - Risk-Return Frontier (efficient frontier framing)
8. **F1** - "Peer Benchmark Is Wrong" (provocative, discussion-worthy)
9. **A5** - Insurance Value Distribution (honest, nuanced)
10. **F2** - "Cost of Getting It Wrong" (business case for the framework)

### Tier 3: Supporting / Appendix Material
11. **B5** - Insurance Value Decay with Scale
12. **D4** - Breakeven Time
13. **E3** - Probability of Outperformance
14. **A4** - Terminal Wealth Distribution
15. **C4** - Variance Reduction Verification (methodological appendix)

### Tier 4: Exploratory / Future Work
16. **C3** - Cross-Config CRN Path Comparison (pending verification)
17. **D2** - Wealth Distribution Evolution (visually impressive but complex)
18. **D3** - Ruin Hazard Rate (technically interesting but niche)
19. **E2** - Arithmetic vs. Geometric Growth Rates
20. **B2** - Growth Lift Surface (limited by 3x3 grid resolution)

---

## I. Technical Notes

### CRN Cross-Configuration Verification Protocol
To confirm whether sim_id pairing works across configurations:
```python
# Load two configs with same (Cap, ATR) but different Ded
results_ded0 = pickle.load(open("Cap (5M) - ATR (1.0) - ... - Ded (0K) - ....pkl", "rb"))
results_ded100 = pickle.load(open("Cap (5M) - ATR (1.0) - ... - Ded (100K) - ....pkl", "rb"))

# Check if annual losses for sim_id=0 are identical in year 1
# (They should be, since year-1 revenue is deterministic and CRN seeds match)
```

### Handling Ruined Paths
For paths that go insolvent:
- **Growth rate**: Should be -infinity (or a very large negative number)
- **Terminal wealth**: Zero (or the insolvency-year equity, which is negative)
- **Recommendation**: Report ruin probability separately, then condition growth rate statistics on survival. Always report both.

### Statistical Precision
At 250K simulations, standard errors of mean growth rates should be very small. But tail statistics (ruin probability at 50 years if it's ~0.1%) will have meaningful sampling error. Report confidence intervals for any tail metric.

### Figure Sizing for LaTeX
- Single-column: 3.5 inches wide
- Double-column: 7.0 inches wide
- Target resolution: 300+ DPI
- Font size: match the LaTeX body font (typically 10-12pt)
- Use colorblind-safe palettes (avoid red-green contrasts)

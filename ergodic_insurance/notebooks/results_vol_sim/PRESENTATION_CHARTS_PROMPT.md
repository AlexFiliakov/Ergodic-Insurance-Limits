Make engaging professional publication visuals with 300dpi that support colorblind accessibility and look engaging to a varied audience, yet subdued so they are appropriate for a research paper that will be distributed in full color. The visuals are specified in subsections below.

The visuals should be generated using a Jupyter Notebook created in `ergodic_insurance\notebooks\results_vol_sim\3. generate_publication_viz.ipynb` and should feed from the data parsed in `ergodic_insurance\notebooks\results_vol_sim\1. process_vol_sim_results.ipynb`

Please ask me clarifying questions and implement these visuals.

## Visual Aesthetic Extraction

To support future visual development, please organize the visuals you used into a specification document in `ergodic_insurance\notebooks\results_vol_sim\VISUAL_SPECIFICATION.md` to include things like:
- Color palette used and how to apply it
- List when to use of line thickness
- Use of solid lines, dashed, dotted, and other line types
- Whether to use background chart grids and how to use them
- Appropriate fonts that will look good in a LaTeX paper with its default font
- Guidance on the use of outlines

Ensure the aesthetics are consistent across graphics while remaining varied enough to remain engaging and to signal that they represent different concepts.

## Individual Charts to Develop

### Ensemble vs. Time Average Divergence

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

### Optimal Deductible Heatmap

**What it shows:** A 3x3 heatmap where rows are Cap ($5M, $10M, $25M), columns are ATR (0.8, 1.0, 1.2), and the cell color/label indicates which deductible maximizes the time-average growth rate. (Expect that all the data will eventually be available, even though some simulations are still running.)

**Why it engages the audience:** This is the paper's core actionable result. An actuary at a major broker advising a client can look at this and say: "Given your capitalization and revenue intensity, the ergodic framework recommends this retention level." It directly challenges the one-size-fits-all deductible recommendations that dominate current practice. The pattern should show that larger companies (higher Cap) can tolerate higher deductibles, while revenue-intensive businesses (higher ATR) need more protection.

**Implementation notes:**
- Color by optimal deductible level (categorical colormap)
- Annotate each cell with the growth rate at the optimum AND the growth rate at the worst deductible, so readers see the cost of getting it wrong
- Consider a second heatmap showing the "growth rate penalty" for choosing guaranteed cost ($0 ded) vs. optimal

### Wealth Trajectory Fan Chart

**What it shows:** A fan chart of asset trajectories over 50 years for a single (Cap, ATR, Deductible) configuration. The median path is a bold line; shaded bands show the 10th-90th, 25th-75th percentile envelopes. Overlay the paired no-insurance fan chart in a contrasting color.

**Why it engages the audience:** Actuaries understand distributions, but most insurance analyses present single-year snapshots. Showing 50 years of wealth evolution makes the *compounding* effect of insurance tangible. The visual gap between the insured and uninsured fans is the ergodic argument made visible. Senior risk managers immediately see that insurance isn't a cost---it's a trajectory shifter.

**Recommended configuration:** Start with the most dramatic case: $5M Cap, ATR=1.2, Ded=$100K (high revenue intensity, small company, moderate retention). This maximizes the visual contrast. Then show $25M Cap as a counterpoint where insurance matters less.

**Implementation notes:**
- Subsample ~1,000 paths for the transparent spaghetti lines behind the fan
- Use log scale on the y-axis so both surviving and ruined paths are visible
- Mark the ruin threshold clearly with a horizontal dashed line
- Consider a dual-panel (insured left, uninsured right) with shared y-axis

### Survival Curves

**What it shows:** Kaplan-Meier-style survival curves plotting the fraction of simulations still solvent at each year, for all four deductible levels within a single (Cap, ATR) pair.

**Why it engages the audience:** Ruin probability is the single metric that most directly challenges expected-value thinking. Showing that ruin *accumulates* over time---even for profitable companies---hits a nerve. Most actuarial analyses quote annual ruin probability; showing the 50-year cumulative version is a wake-up call. The separation between deductible curves directly quantifies how much retention appetite affects survival.

**Recommended configuration:** Use $5M Cap, ATR=1.0 as the base case (mid-revenue, small company). The deductible curves should separate clearly here.

**Implementation notes:**
- Plot all 4 deductible lines plus the no-insurance baseline (5 curves)
- Add confidence bands (bootstrap from the 250K paths) to show statistical precision
- Include a zoomed inset for the first 10 years where most separation occurs
- Mark any crossover points where a lower deductible overtakes a higher one

### "Life or Death" Attribution

**What it shows:** For each configuration, count the paths where the uninsured company went bankrupt but the insured company survived. Decompose these "saved" paths by the year of divergence (when did the uninsured version fail?).

**Why it engages the audience:** This is the most emotionally compelling analysis. It answers: "How many times, out of 250,000 alternate histories, did insurance literally save the company?" Even if the average growth lift is modest, knowing that insurance prevented ruin in 12% of scenarios (hypothetical) has enormous persuasive power. Risk managers and boards of directors respond strongly to this framing.

**Implementation notes:**
- For each sim_id, check insolvency_year for insured vs. uninsured
- Classify into: both survived, both failed, insured survived/uninsured failed, insured failed/uninsured survived
- The last category (insured failed but uninsured survived) should be rare/zero and would indicate a premium burden problem
- Present as a stacked bar chart across deductible levels

### Year-by-Year Growth Lift

**What it shows:** For each year t, compute the mean log-wealth of insured paths minus the mean log-wealth of uninsured paths. Plot this "cumulative insurance value" over 50 years.

**Why it engages the audience:** Most insurance analyses are single-period. This shows that insurance value *compounds*---the gap widens over time as the preserved capital earns returns. If the curve accelerates, that's the ergodic argument in its purest form: insurance isn't a per-year cost-benefit; it's a compounding advantage. CFOs who think in terms of NPV will immediately grasp this.

**Recommended configuration:** Do this for all 4 deductible levels at $5M Cap, ATR=1.0. Also compare across the three Cap levels to show how the curve flattens for larger companies.

**Implementation notes:**
- Compute at each year: mean(log(assets_insured_i[t])) - mean(log(assets_uninsured_i[t]))
- This is the geometric mean ratio of wealth levels
- Include a confidence band (bootstrap or analytic)
- Mark the year where the cumulative premium paid equals the cumulative insurance benefit (breakeven)

### "Peer Benchmark Is Wrong"

**What it shows:** Take two configurations with the same ATR (same industry, same revenue intensity) but different capitalizations. Show that the optimal deductible differs substantially. This directly undermines peer benchmarking---the practice of setting retentions based on what similar-revenue companies do.

**Why it engages the audience:** Challenging peer benchmarking is provocative because it's the default approach at most brokers. If the paper can demonstrate that a $5M company and a $25M company *in the same industry* should have different retentions even with identical revenue, that's a publishable finding that will be discussed in practice.

**Implementation notes:**
- Select ATR=1.0 as the common "industry"
- Compare Cap=$5M vs. Cap=$25M
- Show optimal deductible, growth lift, and ruin probability side by side
- Frame as: "Two companies with identical revenue ($Xm) but different balance sheets should make fundamentally different insurance decisions"

### Insurance Value Distribution

**What it shows:** For each of the 250K paired simulations, compute Delta_i = log(final_assets_insured_i) - log(final_assets_uninsured_i). Plot the distribution of Delta_i.

**Why it engages the audience:** Most analyses report the *average* benefit of insurance. This shows the *distribution* of benefit across scenarios. Some paths benefit enormously (catastrophic loss absorbed), some paths are net losers (paid premium but never had a large claim), and the shape of this distribution tells a more honest story. The fact that the mean of Delta is positive despite many negative realizations is a compelling narrative for experienced practitioners.

**Recommended configuration:** Do this for each deductible level at $5M Cap, ATR=1.0. The distribution shape should shift meaningfully across deductible levels.

**Implementation notes:**
- Handle ruined paths carefully: if uninsured ruins but insured survives, Delta is effectively infinite. Consider reporting separately: "fraction of paths where insurance prevented ruin" and "conditional benefit given both survived"
- This analysis is only valid within a single (Cap, ATR) pair where insured and uninsured share the same CRN
- A box plot across deductible levels is a useful companion chart

### "Cost of Getting It Wrong"

**What it shows:** For each (Cap, ATR), compute the growth rate at the optimal deductible and at each suboptimal deductible. Report the growth penalty (in basis points or percentage) for choosing wrong.

**Why it engages the audience:** Actuaries and risk managers need to justify the effort of this analysis. If the penalty for choosing the wrong deductible is 5 basis points of annual growth, it's academic curiosity. If it's 50-100 basis points, it's material to shareholder value. The magnitude of the penalty *is* the business case for the framework.

**Implementation notes:**
- Table format: rows = (Cap, ATR), columns = deductible levels, cells = growth rate
- Bold the optimal in each row
- Add a column for "max penalty" (best - worst deductible)
- Translate growth rate differences to wealth multiples over 50 years: a 50bp difference compounds to a 28% wealth gap over 50 years

### Insurance Value Decay with Scale

**What it shows:** Line chart showing how the growth lift at each deductible level decays as capitalization increases, with separate lines for each ATR level.

**Why it engages the audience:** This directly quantifies the intuition that "larger companies need less insurance." But it also reveals *how fast* the decay happens and whether ATR modulates it. If high-ATR companies retain more insurance value at larger sizes, that's a finding brokers can use to differentiate advice between asset-light and asset-heavy businesses.

**Implementation notes:**
- Three panels (one per ATR) or overlaid with different line styles
- Express growth lift in basis points for intuitive scale
- Add a horizontal reference line at zero to show where insurance becomes value-neutral

### Breakeven Time

**What it shows:** For each configuration, compute the year at which the median insured wealth first exceeds the median uninsured wealth. Before this year, the premium burden dominates; after it, the ruin-prevention benefit dominates.

**Why it engages the audience:** This directly addresses the CFO objection: "How long until I see a return on this insurance spend?" The answer (which should vary by company size and retention) gives practitioners a concrete talking point. If the breakeven is 5-10 years for a $5M company, that's within a typical strategic planning horizon.

**Implementation notes:**
- Compute median(assets_insured[t]) and median(assets_uninsured[t]) at each year
- Find the crossover point
- Report as a table: (Cap, ATR, Ded) -> breakeven year
- Also compute using geometric mean instead of median for robustness

### Probability of Outperformance

**What it shows:** For each (Cap, ATR), plot the fraction of CRN-paired paths where the insured version outperforms the uninsured version (in terms of final wealth) as a function of deductible. Also show the average outperformance conditional on outperformance, and the average underperformance conditional on underperformance.

**Why it engages the audience:** This reframes insurance from "does it help on average" to "how likely is it to help, and by how much when it does vs. doesn't?" Even if insurance helps in only 60% of paths, if the help in those paths is 3x larger than the harm in the other 40%, the decision is clear. This asymmetry is the essence of the ergodic argument, and presenting it this way respects the sophistication of the actuarial audience.

**Implementation notes:**
- For each paired sim_id: outperformance_i = final_assets_insured_i > final_assets_uninsured_i
- Probability = mean(outperformance_i)
- Conditional values: mean(delta | delta > 0) vs. mean(delta | delta < 0)
- Plot as a grouped bar chart or table

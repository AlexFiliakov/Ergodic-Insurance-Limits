# Time Averages, Not Expectations: An Ergodic Framework for Insurance Limit Optimization

**An Open-Source Simulation Platform for Corporate Insurance Purchasing Decisions**

---

## Executive Summary

Most corporate insurance purchasing decisions rest on a single, seemingly unassailable principle: buy coverage when the premium is less than the expected loss. This expected-value logic, inherited from classical actuarial science and embedded in boardroom heuristics, treats the firm as though it simultaneously occupies every possible future. It does not. A firm traverses one path through time, and on that path, a sufficiently large uninsured loss is not merely expensive — it is irreversible.

This paper introduces the Ergodic Insurance Limits Framework, an open-source Python simulation platform that applies ergodicity economics — the study of how time-averaged outcomes diverge from ensemble averages in multiplicative processes — to the problem of setting insurance retentions and limits. The framework models a representative commercial insured over multi-decade horizons, simulating thousands of stochastic trajectories through realistic loss environments. It measures what matters to the individual firm: the compound growth rate actually realized over time, not the average across hypothetical parallel firms.

Key findings from the framework's simulation architecture include:

- **Time-average growth rates diverge materially from ensemble averages.** Insurance that appears costly under expected-value analysis can enhance the time-average growth rate by 2–4 percentage points annually.
- **Survival probability is a first-order growth variable.** Uninsured firms face 20–30 percentage point lower survival rates over 20-year horizons, and bankruptcy permanently removes compounding capacity.
- **Optimal premiums can rationally exceed expected losses by 200–500%.** When the metric shifts from expected cost to time-average growth, the willingness to pay for tail protection increases substantially.

The framework provides actuaries, CFOs, and risk managers with tools to construct efficient frontiers across retention, limit, and premium space — enabling insurance decisions grounded in the mathematics of what a single firm actually experiences.

---

## 1. The Problem: Why Expected-Value Frameworks Mislead Limit-Setting Decisions

Insurance purchasing decisions at commercial insureds follow a well-worn pattern. The risk manager assembles loss data, the broker markets the program, and the CFO evaluates competing structures by comparing premiums against expected losses. If the premium exceeds the expected loss by a margin the organization considers unreasonable, the instinct is to retain more risk. The reasoning is straightforward: over the long run, paying more than you expect to collect cannot be rational.

This reasoning is correct — but only under assumptions that do not hold for real firms.

The expected-value framework implicitly treats the firm as an ensemble: a large collection of identical companies experiencing every possible loss outcome simultaneously. In this framing, the law of large numbers guarantees that aggregate outcomes converge to their mathematical expectation. Insurance that costs more than expected losses becomes, by definition, a negative expected-value proposition — a drag on profitability that sophisticated firms should minimize.

But a firm is not an ensemble. It is a single entity moving through time, accumulating and compounding the consequences of sequential decisions. This distinction, which might seem philosophical, has concrete mathematical implications.

### The Compounding Problem

Consider a manufacturer with $10 million in assets, an 8% operating margin, and an annual loss exposure that averages $430,000 but has a small probability of producing a $5 million or $10 million event. Under expected-value analysis, the expected annual loss is modest relative to assets, and a high retention — say $1 million or $2 million — appears to produce the best long-run economics. The firm retains the predictable losses, saves premium, and expects to come out ahead.

Now follow this firm through time rather than across a hypothetical ensemble. In most years, the strategy works. The firm retains its predictable attritional losses, saves on premium, and grows. But wealth accumulates multiplicatively: this year's equity is the base for next year's growth. A single catastrophic loss that impairs equity by 30–50% does not merely cost the firm that year's income. It shrinks the base from which all future growth compounds. If the firm had $10 million in equity and suffers a $4 million uninsured loss, it now compounds from $6 million. Five years of 8% growth from $6 million yields $8.8 million — still below the $10 million it started with, and far below the $14.7 million it would have reached without the loss.

At the extreme, a loss that renders the firm insolvent truncates all future compounding permanently. Bankruptcy is an absorbing state: the firm's growth rate becomes negative infinity, and no amount of favorable outcomes in other scenarios offsets this outcome for the firm that experienced it.

### Ensemble Averages Conceal Individual Trajectories

The expected-value framework obscures this dynamic because it averages across scenarios rather than across time. When 10,000 simulated firms are averaged, the survivors — including a few that experienced exceptional growth — pull the mean upward. The ensemble average suggests robust performance even as a meaningful fraction of individual firms have failed. The expected value of terminal equity may be $15 million, but the median may be $9 million, and 15% of paths may have ended in insolvency.

This is not a minor statistical artifact. For multiplicative processes — and corporate wealth accumulation is fundamentally multiplicative — the arithmetic mean of outcomes across scenarios is not equal to, and systematically overstates, the outcome that a typical firm will experience over time. The time average, which measures the growth rate along a single path, is the quantity that governs what actually happens to the business.

The divergence between these two averages is not a matter of pessimism or conservatism. It is a mathematical property of multiplicative dynamics, first formalized in the context of economic decision-making by Ole Peters and Alexander Adamou at the London Mathematical Laboratory. Their work on ergodicity economics demonstrates that expected-value maximization — the foundation of classical decision theory — produces systematically different prescriptions than time-average optimization whenever the underlying process is non-ergodic. Corporate wealth accumulation, subject to multiplicative growth and the possibility of ruin, is precisely such a process.

---

## 2. Conceptual Foundation: Ergodicity Economics and the Time-Average Distinction

### What Ergodicity Means

A process is **ergodic** if its time average equals its ensemble average — if the outcome experienced by one participant over a long period is the same as the average outcome across many participants at a single point in time. Flipping a fair coin for points is ergodic: your long-run average gain per flip converges to the average across all possible flips. Temperature in a well-mixed room is ergodic: the time-averaged temperature at one point equals the spatial average at one instant.

A process is **non-ergodic** if these two averages diverge. Crucially, multiplicative processes with the possibility of ruin are non-ergodic. The canonical example is a gamble that multiplies your wealth by 1.5 with 50% probability and by 0.6 with 50% probability. The expected (ensemble-average) wealth after one round is 1.05 times the starting wealth — a 5% expected gain. Yet the time-average growth rate per round is the geometric mean: √(1.5 × 0.6) ≈ 0.949, a 5.1% decline. An individual playing this gamble repeatedly will, with near certainty, lose wealth over time, even though the "expected" outcome at any single round is positive.

The same mathematics applies to a firm facing stochastic losses. The expected value of the firm's equity after 20 years — the ensemble average — includes scenarios where exceptional luck produced extraordinary growth. But no individual firm experiences the ensemble average. Each firm experiences one path, and on that path, the geometric compounding of gains and losses determines the outcome.

### Ensemble Average vs. Time Average in Insurance Context

To make this concrete, consider two insurance strategies for the same firm:

**Strategy A (Minimal Insurance):** The firm purchases only catastrophic coverage, retaining the first $2 million of any loss. Annual premium: $150,000. Expected annual retained loss: $400,000.

**Strategy B (Comprehensive Insurance):** The firm purchases coverage with a $250,000 retention. Annual premium: $600,000. Expected annual retained loss: $100,000.

Under expected-value analysis, Strategy A is superior. The total expected annual cost (premium plus expected retained loss) is $550,000, compared to $700,000 for Strategy B. A rational expected-value optimizer chooses Strategy A every time.

Now simulate 10,000 firms over 20 years under each strategy. The ensemble-average terminal equity under Strategy A may indeed be higher — perhaps 6% compound growth versus 5% under Strategy B. But examine the time-average growth rate: the geometric mean of growth rates across all paths. Under Strategy A, the time average may be only 3–4%, dragged down by the large number of paths where major losses impaired the compounding base. Under Strategy B, the time average may be 5–6%, because the insured firm's equity trajectory is more stable, and fewer paths experience the compounding interruptions that depress geometric growth.

The divergence arises because the ensemble average weights lucky outcomes heavily — a few paths that avoided all large losses and compounded at maximum rates pull the mean upward. The time average reflects what a typical firm actually experiences, accounting for the mathematical certainty that, over a long enough horizon, tail losses will eventually occur.

### The Volatility Tax

Ergodicity economics provides a precise mechanism for this divergence: the **volatility tax**. For a multiplicative process with arithmetic mean return μ and variance σ², the time-average growth rate is approximately:

> g ≈ μ − σ²/2

The term σ²/2 is the volatility tax — the penalty that converts the ensemble-average return into the time-average return. Higher volatility means a larger gap between what the "average" firm achieves and what any particular firm is likely to experience.

Insurance reduces σ² by truncating the loss distribution at the policy limit. This reduction in variance may come at a cost (the premium), which reduces μ. But if the reduction in σ²/2 exceeds the reduction in μ, insurance increases the time-average growth rate even when it decreases the ensemble average. This is the central insight: insurance can create genuine economic value — not merely redistribute risk — when evaluated through the lens of time-average growth.

---

## 3. Framework Overview: Simulation Architecture and Design

The Ergodic Insurance Limits Framework translates these theoretical insights into a computational platform capable of evaluating insurance structures for realistic commercial risks. The framework is implemented in Python and is available as an open-source package.

### 3.1 Architecture

The framework is organized in layers, progressing from configuration and input specification through simulation and analysis to reporting:

**Configuration Layer.** A unified configuration system specifies the firm's financial characteristics (initial assets, operating margin, growth rate, capital structure), loss environment (frequency and severity distributions across loss tiers), and insurance program structure (retentions, limits, layers, premium rates). Configurations support inheritance and composition: a base profile can be extended with modules for specific industries or market conditions, and presets allow rapid exploration of hard-market, normal-market, and soft-market environments.

**Business Logic Layer.** At its core, the framework models a representative commercial insured as a going concern with a double-entry financial ledger, accrual-based accounting, and multi-period balance sheet dynamics. The firm generates revenue as a function of its asset base and a stochastic growth process, incurs operating expenses, pays insurance premiums, absorbs retained losses, receives insurance recoveries, and updates its balance sheet accordingly. Equity compounds multiplicatively from period to period, and insolvency occurs when equity is exhausted.

**Simulation Layer.** A Monte Carlo engine runs thousands of independent firm trajectories in parallel, each drawing from the same loss distributions but following different stochastic paths. Convergence diagnostics — including R-hat statistics, effective sample size, and Monte Carlo standard error — provide confidence that simulation results have stabilized. The engine supports configurable time horizons, typically 20 to 50 years, reflecting the long-term perspective required for ergodic analysis.

**Analysis Layer.** The ergodic analyzer computes both ensemble-average and time-average metrics for each scenario, identifies the divergence between them, and performs statistical significance testing (Welch's t-test) to confirm that observed differences are not attributable to sampling variation. Additional analysis modules compute risk metrics (Value at Risk, Tail Value at Risk, maximum drawdown), ruin probability surfaces, sensitivity analyses, and Pareto efficiency frontiers.

**Optimization Layer.** A business optimizer searches the retention–limit space to identify insurance structures that maximize a configurable objective — typically time-average growth rate, subject to constraints on premium budget, minimum survival probability, or maximum acceptable ruin probability. The optimizer employs both gradient-based (SLSQP) and global (differential evolution) search algorithms, with optional Hamilton-Jacobi-Bellman stochastic control formulations for continuous-time problems.

### 3.2 Key Inputs

**Loss Models.** The framework categorizes loss exposure into three tiers, consistent with standard actuarial practice:

- *Attritional losses* (high frequency, low severity): Poisson-distributed occurrence counts with lognormal severities. These represent routine operational hazards — minor equipment failures, small liability claims, inventory damage.
- *Large losses* (moderate frequency, moderate severity): Poisson occurrence with lognormal severity at higher parameters. These represent material disruptions — significant equipment failure, product liability events, supply chain interruptions.
- *Catastrophic losses* (low frequency, high severity): Poisson occurrence with Pareto-distributed severity. The heavy-tailed Pareto distribution captures the extreme-value behavior of catastrophic events — facility destruction, major product recalls, natural disasters.

Each tier's frequency can scale dynamically with the firm's financial state through configurable exposure bases (revenue-based, asset-based, or equity-based), reflecting the actuarial reality that a growing firm's loss exposure grows with it.

**Severity trend models** — including linear, mean-reverting, random walk, regime-switching, and scenario-based trends — allow the framework to incorporate loss cost inflation, regulatory changes, or economic cycle effects.

**Insurance Program Structure.** The framework supports multi-layer coverage towers, where each layer is defined by an attachment point, per-occurrence limit, aggregate limit, participation rate, and reinstatement provisions. This mirrors the structure of a real commercial insurance program, where primary coverage, excess layers, and umbrella policies each respond to different portions of the loss distribution. Premium calculation follows an actuarially grounded pipeline: pure premium (from simulation or limited expected value functions), plus allocated and unallocated loss adjustment expense loadings, plus operating expense and profit provisions, adjusted for market cycle conditions.

**Claim Development.** Losses are developed to ultimate values using standard age-to-ultimate factors, consistent with standard actuarial loss development methodology. The framework provides built-in development patterns for short-tail (property), standard (commercial multi-peril), and long-tail (liability) lines.

### 3.3 Key Outputs

**Ergodic Metrics.** For each insurance scenario, the framework reports:

- *Time-average growth rate*: the compound annual growth rate along individual trajectories, averaged across surviving paths
- *Ensemble-average growth rate*: the growth rate implied by the mean terminal equity across all paths
- *Ergodic divergence*: the difference between time-average and ensemble-average growth rates — a direct measure of how misleading expected-value analysis would be for this scenario
- *Survival rate*: the fraction of simulated paths that avoid insolvency over the full time horizon

**Efficiency Frontiers.** By sweeping across the retention–limit space, the framework generates Pareto frontiers showing the trade-off between competing objectives: time-average growth versus premium cost, survival probability versus capital efficiency, or return on equity versus ruin probability. These frontiers identify the set of insurance structures where no objective can be improved without worsening another — the efficient set from which the firm should choose.

**Optimal Retention–Limit Pairs.** The optimizer identifies specific insurance program configurations that maximize time-average growth subject to constraints. Sensitivity analysis reveals how these optima shift under different assumptions about loss frequency, severity trends, market conditions, and risk tolerance.

---

## 4. Research Applications

The framework is designed as research infrastructure, not a finished product. Its modular architecture supports extension along several dimensions that are, at present, open problems in the intersection of ergodicity economics and insurance science.

### 4.1 Multi-Line Optimization

The current implementation models a single aggregate loss exposure. Commercial insureds, however, maintain programs across multiple lines — property, general liability, professional liability, workers' compensation, cyber, and others — each with distinct frequency-severity characteristics, correlation structures, and market dynamics. Extending the framework to jointly optimize retentions and limits across correlated lines of business represents a natural and high-value research direction. The mathematical challenge lies in modeling dependence: how does a catastrophic property loss affect the firm's liability exposure, or vice versa? Copula-based dependence structures, already standard in enterprise risk management, could be integrated into the loss generation layer with modest architectural effort.

### 4.2 Dynamic and Adaptive Strategies

The framework currently evaluates static insurance programs — fixed retentions and limits held constant over the simulation horizon. In practice, firms adjust their programs annually in response to loss experience, market conditions, and balance sheet changes. Modeling dynamic strategies, where the firm's retention adjusts as a function of its current financial state, is a compelling research direction. The framework's Hamilton-Jacobi-Bellman solver provides a starting point for deriving continuous-time optimal policies, and the strategy backtesting module enables out-of-sample validation of adaptive approaches against static benchmarks.

### 4.3 Sector-Specific Calibration

The default parameterization models a stylized manufacturer. Calibrating the framework to sector-specific loss data — using publicly available industry loss ratios, frequency-severity benchmarks from statistical agents, and financial ratios from SEC filings — would produce sector-specific efficiency frontiers. Such calibration could demonstrate, for example, that the ergodic advantage of insurance is larger for capital-light technology firms (where a single liability event can represent a large fraction of equity) than for capital-heavy utilities (where the loss-to-equity ratio is typically lower).

### 4.4 Reinsurance and Risk Transfer Chain Analysis

The framework models the ceding insured's perspective. Extending it to model the full risk transfer chain — from insured to primary insurer to reinsurer — would allow investigation of how ergodic effects propagate through the chain. Does the ergodic advantage of ceding risk diminish at each step? How do reinsurance program structures (quota share, excess of loss, aggregate stop-loss) create or destroy value when measured in time-average terms? These questions are relevant to both reinsurance buyers and regulators concerned with systemic risk.

### 4.5 Behavioral and Organizational Factors

The current framework assumes a rational decision-maker optimizing for time-average growth. In practice, insurance purchasing decisions are influenced by organizational incentives (the risk manager's career risk from an uninsured loss), behavioral biases (probability weighting, status quo bias), and institutional constraints (board-mandated retention levels, rating agency expectations). Incorporating these factors into the objective function — perhaps through prospect-theoretic utility functions or constraint sets that reflect organizational realities — would increase the framework's practical relevance.

---

## 5. Practical Implications for Insurance Buyers and Brokers

### For Insurance Buyers

The framework's findings challenge several common heuristics in corporate insurance purchasing:

**"We should retain more risk because our loss ratio is favorable."** A favorable loss ratio over a short observation period may reflect good fortune rather than low underlying exposure. The expected-value calculation based on this experience treats the firm as though it will always be lucky. The time-average perspective recognizes that tail events, while rare in any given year, are near-certain over multi-decade horizons. A firm that has not experienced a catastrophic loss in 10 years is not immune — it is overdue, in the sense that the conditional probability of a tail event in the next 10 years, given none in the last 10, is virtually unchanged from the unconditional probability.

**"Insurance is a cost center that we should minimize."** Under expected-value analysis, insurance is indeed a cost center: the premium exceeds the expected recovery. Under time-average analysis, insurance can be a growth investment: by truncating downside variance, it increases the rate at which the firm's equity compounds. The framework quantifies this effect, showing that optimal insurance programs can enhance the time-average growth rate by 2–4 percentage points annually — a return that compares favorably to many capital investments.

**"Our retention should be based on our ability to fund a single loss."** Conventional retention-setting asks: can we absorb a loss of this size without material financial distress? This is a necessary condition but not sufficient. The ergodic framework asks a different question: what retention level maximizes our long-run compound growth rate, accounting for the full distribution of possible losses and their sequential impact on our balance sheet? The answer often differs significantly from the single-event affordability threshold.

### For Brokers

Brokers operate at the intersection of client advisory and market execution. The framework provides brokers with a quantitative platform for demonstrating the value proposition of insurance programs that might otherwise appear expensive relative to expected losses.

**Differentiated client advisory.** Rather than competing on market access alone, brokers can offer clients a simulation-based analysis showing how alternative program structures affect long-term financial outcomes. The framework's efficiency frontiers provide a visual and analytical basis for program design conversations that move beyond premium benchmarking.

**Quantified value of limits.** The framework produces explicit measures of how each incremental dollar of coverage contributes to the client's time-average growth rate. This allows brokers to quantify the value of excess layers that may seem expensive on a rate-on-line basis but provide meaningful protection against compounding disruption.

**Market cycle strategy.** The framework's market cycle presets (hard, normal, soft) enable brokers to model how clients should adjust their programs as pricing conditions change. In a hard market, the temptation is to increase retentions to control premium costs. The ergodic framework can identify the point at which retention increases begin to erode time-average growth — the point at which saving premium becomes counterproductive.

---

## 6. Methodological Considerations and Limitations

Intellectual honesty requires acknowledging the boundaries of any analytical framework.

**Parameter sensitivity.** The framework's outputs are sensitive to input assumptions, particularly the tail behavior of the loss distribution. Pareto-distributed catastrophic losses with different shape parameters can produce meaningfully different optimal retention levels. Users should conduct sensitivity analysis across plausible parameter ranges — a capability the framework provides explicitly — rather than treating any single parameterization as definitive.

**Model risk.** The framework models a stylized firm with simplified financial dynamics. Real firms face complexities not captured here: multi-year claim development with uncertain reserves, regulatory capital requirements, tax shield effects of loss deductions, covariance between loss experience and operating performance, and the strategic behavior of counterparties in the insurance market. These omissions do not invalidate the ergodic insight, but they do mean that the framework's quantitative outputs should be treated as directional rather than prescriptive.

**Stationarity assumptions.** The loss distributions and growth parameters are assumed to be stationary (or follow specified trends) over the simulation horizon. In practice, loss environments shift in ways that are difficult to model: new liability theories emerge, supply chains restructure, climate patterns evolve. The framework's regime-switching trend model provides a partial accommodation, but fundamental non-stationarity remains a limitation of any simulation-based approach.

**Computational cost.** Optimization over the retention–limit space requires evaluating many candidate programs, each requiring a full Monte Carlo simulation. While the framework employs parallel execution and convergence-based early stopping, comprehensive optimization can be computationally intensive. The framework addresses this through adaptive chunking strategies and configurable convergence thresholds.

---

## 7. Conclusion and Call to Further Research

The expected-value framework has served the actuarial profession well for over a century. It is the right tool for the problems it was designed to solve: pricing risk across large, diversified portfolios where the law of large numbers applies. But when the question shifts from "How should an insurer price this risk?" to "How should a firm manage this risk over time?", the ensemble-average perspective becomes insufficient. The firm does not experience the ensemble. It experiences time.

The Ergodic Insurance Limits Framework provides a computational bridge between the theoretical insights of ergodicity economics and the practical demands of corporate insurance decision-making. By simulating individual firm trajectories through realistic loss environments and measuring time-average growth rates, the framework reveals a consistent pattern: insurance structures that appear expensive under expected-value analysis can be demonstrably value-creating when evaluated on the metric that actually governs long-term firm outcomes.

This finding is not a repudiation of actuarial science. It is an extension — an application of rigorous mathematics to a question that traditional tools were not designed to answer. The premium that exceeds expected losses is not irrational. It is the price of preserving the firm's compounding capacity against the irreversible damage of tail losses. It is the cost of converting a non-ergodic process into something closer to an ergodic one.

The framework is open-source and designed for extension. We invite actuaries, risk managers, financial economists, and applied mathematicians to calibrate it to specific industries, extend it to multi-line and dynamic strategies, and subject its findings to empirical validation against actual corporate outcomes. The intersection of ergodicity economics and insurance science is newly charted territory, and the questions it raises — about optimal risk transfer, the true cost of self-insurance, and the conditions under which insurance creates economic value — are both practically important and intellectually rich.

The code, documentation, and example analyses are available at [github.com/AlexFiliakov/Ergodic-Insurance-Limits](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits).

---

## References

Peters, O. (2019). The ergodicity problem in economics. *Nature Physics*, 15(12), 1216–1221.

Peters, O., & Adamou, A. (2018). The time interpretation of expected utility theory. *arXiv preprint arXiv:1801.03680*.

Peters, O., & Adamou, A. (2021). *Ergodicity Economics*. London Mathematical Laboratory.

---

*The Ergodic Insurance Limits Framework is an open-source research tool. It is not a substitute for professional actuarial judgment or licensed insurance advice. All numerical illustrations in this paper are derived from stylized simulation parameters and should not be interpreted as predictions of actual insurance program performance.*

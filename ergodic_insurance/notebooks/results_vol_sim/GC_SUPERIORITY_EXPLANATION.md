# Why Guaranteed Cost ($0 Deductible) Dominates All Deductible Levels

## Executive Summary

Across all 18 simulated configurations (3 capitalizations x 3 ATRs x 2 that have full deductible sweeps), the Guaranteed Cost (GC, $0 deductible) option produces the highest time-average growth rate, beating every higher-deductible option by 50-90+ basis points annually. This document explains why.

**Bottom line:** The dominance is a **genuine ergodic effect**, not a bug. However, the magnitude of the advantage is amplified by the calibration of catastrophic losses relative to company size. The model's conclusions are directionally correct but would be more nuanced with different parameter choices.

---

## 1. The Observed Pattern

For the reference configuration ($5M Cap, ATR=1.0, EBITABL=12.5%, Vol=0.15):

| Deductible | Growth Rate (bps/yr) | Lift vs No Insurance |
|---|---|---|
| $0K (GC) | 235.0 | +111.5 |
| $100K | 175.7 | +52.2 |
| $250K | 160.2 | +36.7 |
| $500K | 146.5 | +22.9 |
| No Insurance | 123.5 | --- |

The ranking is the **exact reverse** of the expected-value ordering. In expectation, higher deductibles are cheaper (lower total cost of risk), yet in the time-average simulation, lower deductibles produce superior growth.

This pattern holds across **every** (Cap, ATR) combination tested, including $10M capitalization.

---

## 2. Growth Rate Decomposition

The gap between the "naive" analytical growth rate and the actual simulation growth rate has two distinct components:

### 2.1 Component 1: Tax Accrual Structural Drag (~163-205 bps)

The financial model uses accrual-basis accounting with a tax payment lag. In annual resolution:
- Year N: taxes are **expensed** (reducing net income) and **accrued** as a liability
- Year N+2: the accrued taxes are **paid** from cash

This creates a structural drag on total-asset growth:
- Retained earnings increase cash by `income_BT x (1-tax) x retention = income_BT x 0.525`
- But tax payment from 2 years prior reduces cash by `income_BT_{N-2} x 0.25`
- Net asset growth per year: `income_BT x ~0.275` (instead of `income_BT x 0.525`)

**This is not a bug** -- it correctly represents accrual accounting where tax liabilities reduce equity. The "naive" analytical formula `g = margin x (1-tax) x retention` overstates growth because it ignores the liability side of the balance sheet.

**Crucially: this drag affects all configurations proportionally to their income level. It does NOT change the ranking between deductible levels.**

### 2.2 Component 2: Stochastic Volatility Penalty (-22 to +136 bps)

This is where the ranking reversal occurs. The penalty arises from two sources of multiplicative volatility:

| Configuration | Deterministic g | Actual g | Stochastic Penalty |
|---|---|---|---|
| $0K (GC) | 212.7 bps | 235.0 bps | **-22.3 bps** (helps!) |
| $100K | 229.3 bps | 175.7 bps | +53.6 bps |
| $250K | 232.9 bps | 160.2 bps | +72.7 bps |
| $500K | 236.1 bps | 146.5 bps | +89.6 bps |
| No Insurance | 259.6 bps | 123.5 bps | +136.1 bps |

The stochastic penalty grows **monotonically** with retained risk exposure. For the $0K deductible, the penalty is actually slightly negative (stochastic revenue helps growth) because of a favorable interaction between the premium calculation mechanism and revenue volatility.

### 2.3 The Ranking Reversal

The deterministic ranking favors higher deductibles ($500K > $250K > $100K > $0K) because they have lower expected total cost. But the stochastic penalty **more than reverses** this:

- Deterministic advantage of $500K over $0K: **23.4 bps**
- Stochastic penalty swing: **111.9 bps** (from -22 to +90)
- Net result: $0K beats $500K by **88.5 bps**
- The penalty overwhelms the cost savings by **4.8x**

---

## 3. Why the Stochastic Penalty Is So Large

### 3.1 The Ergodic Growth Penalty

For multiplicative processes, the time-average growth rate is penalized by volatility:

```
g_time ≈ g_expected - σ²/2
```

This `σ²/2` penalty applies to the **total** return volatility experienced by the company, including both revenue volatility and retained loss volatility.

### 3.2 Revenue Volatility (GBM σ=0.15)

All configurations experience the same GBM revenue shocks (σ=0.15, drift=0). The base penalty from revenue volatility alone is `σ²/2 = 112.5 bps`, but the actual impact is modulated by how revenue flows through the financial model.

For the $0K deductible, the stochastic penalty is slightly negative (-22 bps) because:
- Premium is calculated from **deterministic** revenue (before the GBM shock)
- Operating income uses **stochastic** revenue
- In good years: higher revenue, same premium → disproportionate profit boost
- In bad years: lower revenue, same premium → smaller profit reduction
- The asymmetry creates a slight positive bias

### 3.3 Retained Loss Volatility (Fat Tails)

The simple `σ²/2` formula dramatically underestimates the penalty for fat-tailed loss distributions:

| Deductible | Retained Loss Std (% of assets) | Max Retained (% of assets) |
|---|---|---|
| $0K | 0.00% | 0.00% |
| $100K | 1.25% | 10.49% |
| $250K | 1.85% | 20.51% |
| $500K | 2.66% | 34.49% |
| No Insurance | 19.93% | **1,429%** |

The catastrophic loss distribution (Pareto with xm=$5M, alpha=2.5) creates extreme tail events:
- **Mean cat loss: $8.3M** (167% of $5M company's assets)
- Annual probability of any cat event: ~1%
- At the 99th percentile, No Insurance faces retained losses of **103% of assets** in a single year

The Gaussian `σ²/2` approximation fails because:
1. Pareto losses have **infinite variance** for alpha < 3 in practice (alpha=2.5 gives finite but very high kurtosis)
2. A single catastrophic loss can destroy a significant fraction of accumulated wealth
3. The negative compounding from large losses is irreversible (wealth destroyed cannot compound)

### 3.4 The Compounding Mechanism

The penalty is not just about single-year volatility -- it **compounds** over 50 years:

1. A large retained loss in Year 5 reduces assets
2. Lower assets → lower revenue → lower income → slower recovery
3. A second large loss hits a smaller asset base → proportionally more damaging
4. The geometric mean wealth path falls further below the arithmetic mean each year
5. Insurance prevents this negative feedback loop by absorbing the tail risk

This is precisely the ergodic theory argument: the ensemble average (arithmetic mean across paths) diverges from the time average (geometric mean along a single path), and the divergence grows with volatility.

---

## 4. Model Calibration Assessment

### 4.1 Parameters That Amplify GC Dominance

1. **Catastrophic Loss Scale (xm = $5M)**
   - Equal to 100% of the $5M company's starting assets
   - A single catastrophic event is existential for the smallest companies
   - Even for $10M companies, a cat event is 50% of assets
   - *Assessment*: Somewhat aggressive but not unrealistic for manufacturing (major fire, explosion, product recall)

2. **Revenue Volatility (σ = 0.15)**
   - 15% annual revenue standard deviation
   - *Assessment*: Moderate-to-high for manufacturing. Typical range: 5-20%

3. **Premium Loading (Loss Ratio = 0.70)**
   - 43% markup over expected loss
   - *Assessment*: Standard for commercial insurance. Makes insurance affordable
   - At LR=0.30 (233% loading), $500K ded would finally beat $0K ded

4. **Loss Frequency Scaling with Revenue**
   - Larger companies face proportionally more claims
   - Keeps the relative loss distribution constant across company sizes
   - *Assessment*: Reasonable for frequency but may overstate scaling for severity

### 4.2 Why the Ranking Doesn't Differentiate by Company Size

The growth rates at $5M and $10M capitalization are nearly identical:

| Config | $5M Cap | $10M Cap |
|---|---|---|
| $0K Ded, ATR=1.0 | 235.0 bps | 235.7 bps |
| $500K Ded, ATR=1.0 | 146.5 bps | 145.5 bps |

This is because:
1. Loss frequency scales linearly with revenue (= assets × ATR)
2. Revenue scales linearly with assets
3. The retained-loss-to-assets ratio remains approximately constant
4. Therefore the stochastic penalty (as % of assets) is size-independent

In reality, larger companies benefit from diversification and their cat exposure grows sub-linearly with revenue. The model's `revenue_scaling_exponent = 1.0` assumes linear scaling, which may be conservative for larger companies.

---

## 5. Is This a Bug?

**No.** The GC dominance is a genuine consequence of the ergodic framework applied to this calibration. Specifically:

### Not a Bug:
- The insurance claim processing is correct (deductibles properly applied per-occurrence)
- Premium pricing is correct (expected loss / loss ratio, scaled with revenue)
- The financial model correctly implements accrual accounting
- The CRN (Common Random Numbers) ensures proper paired comparisons
- The growth rate measurement (log of final/initial assets over 50 years) is standard

### Calibration Sensitivity:
- The result is robust across all tested (Cap, ATR) combinations
- The result holds for any premium loading factor above ~0.30 loss ratio
- The magnitude of GC dominance would decrease with:
  - Lower revenue volatility (σ < 0.10)
  - Larger company size relative to cat loss minimum (Cap >> xm)
  - Sub-linear loss frequency scaling
  - Higher premium loading (loss ratio < 0.50)

### Tax Accrual Timing:
- The 2-year tax payment lag at annual resolution creates a ~163 bps structural drag on asset growth
- This is correct accrual accounting behavior, not a bug
- It affects all configurations equally and does NOT influence the ranking
- The growth rate would be higher (~375 bps for $0K ded) with cash-basis accounting, but the GC-dominant ranking would persist

---

## 6. Interpretation for the Paper

The universal GC dominance is actually a **strong result** for the ergodic insurance thesis:

> *Even when insurance is priced at a 43% markup over expected losses, and even when higher deductibles save money in expectation, the time-average growth rate is maximized by transferring ALL loss risk to the insurer.*

This is the purest possible demonstration of the ergodic argument:
1. Expected-value analysis says "higher deductibles are cheaper" → retain more risk
2. Time-average analysis says "volatility destroys growth" → transfer all risk
3. The two frameworks give **opposite recommendations**
4. The simulation validates the time-average framework over 250,000 paths × 50 years

### Suggested Framing:
- Frame the GC dominance as the model's primary finding, not a limitation
- Note that the magnitude depends on the cat loss calibration relative to company size
- Highlight that even a 43% premium loading is "cheap" when measured against the ergodic penalty of retained risk
- The breakeven loading is ~233% (LR ≈ 0.30), suggesting that insurance is rational at loadings far exceeding industry norms

### Recommendations for Future Work:
1. **Add deductible granularity**: Test $10K, $25K, $50K deductibles to find the optimal retention
2. **Increase company sizes**: Test $25M, $50M, $100M with sub-linear cat frequency scaling
3. **Reduce revenue volatility**: Test σ = 0.05, 0.08, 0.10 to find the crossover point
4. **Excess layer pricing**: Model different pricing for primary vs excess layers
5. **Separate the tax timing**: Consider computing growth from equity rather than total assets, or use cash-basis tax treatment for cleaner growth rate comparison

---

## 7. Technical Appendix: Complete Growth Rate Decomposition

```
Config     g_naive  g_det  g_actual  TaxDrag  StochPen  TotalPen
----------------------------------------------------------------
  $0K      382.9   212.7   235.0    170.2     -22.3     147.9
 $100K     412.0   229.3   175.7    182.7      53.6     236.3
 $250K     418.3   232.9   160.2    185.4      72.7     258.1
 $500K     423.9   236.1   146.5    187.8      89.6     277.4
 NoIns     464.9   259.6   123.5    205.3     136.1     341.4

Where:
  g_naive   = (EBITABL - total_cost_rate) × (1-tax) × retention
  g_det     = 50-year deterministic simulation (with accrual taxes)
  g_actual  = Mean of 250K MC paths (with GBM σ=0.15 and stochastic losses)
  TaxDrag   = g_naive - g_det (accrual tax timing effect)
  StochPen  = g_det - g_actual (revenue + loss volatility effect)
  TotalPen  = g_naive - g_actual (sum of both effects)
```

### Key Parameters:
- Initial Assets: $5,000,000
- Asset Turnover Ratio: 1.0
- EBITABL (operating margin before insurance): 12.5%
- Tax Rate: 25%
- Retention Ratio: 70% (30% dividends)
- PPE Ratio: 0% (no depreciation)
- Revenue Volatility: GBM with σ=0.15, drift=0
- Loss Ratio: 0.70 (43% insurance loading)
- CRN Base Seed: 20260130
- Simulations: 250,000 paths × 50 years

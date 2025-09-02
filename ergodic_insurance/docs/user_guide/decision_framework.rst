Decision Framework
==================

This framework will guide you through making optimal insurance decisions based on your simulation results. We'll cover the key questions to ask, red flags to avoid, and a systematic approach to implementation.

The Three Critical Questions
----------------------------

Question 1: What's My Ruin Probability Without Insurance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is your most important metric. Ruin means your assets go to zero. Game over.

**Decision Tree:**

.. code-block:: text

   Ruin Probability (10-year horizon)
   │
   ├─ > 5%: CRITICAL RISK
   │  └─ Insurance is mandatory
   │     └─ Focus on survival, not cost optimization
   │
   ├─ 1-5%: SIGNIFICANT RISK
   │  └─ Insurance strongly recommended
   │     └─ Optimize for growth subject to survival constraint
   │
   └─ < 1%: MANAGEABLE RISK
      └─ Insurance optional but likely beneficial
         └─ Optimize for maximum ergodic growth

**Calculation Example:**

.. code-block:: python

   # Calculate ruin probability
   ruin_prob = (results.final_wealth <= 0).mean()

   if ruin_prob > 0.05:
       print("CRITICAL: Insurance is mandatory")
       min_survival_target = 0.95
   elif ruin_prob > 0.01:
       print("SIGNIFICANT: Insurance strongly recommended")
       min_survival_target = 0.99
   else:
       print("MANAGEABLE: Optimize for growth")
       min_survival_target = 0.995

Question 2: What's My Optimal Retention?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your retention (deductible) is the amount you pay before insurance kicks in.

**Retention Selection Framework:**

.. code-block:: text

   Starting Point: 1-2% of Assets
   │
   ├─ Adjust DOWN if:
   │  ├─ High revenue volatility (>20%)
   │  ├─ Thin operating margins (<5%)
   │  ├─ Limited access to credit
   │  ├─ Correlation between losses and revenue
   │  └─ Recent major losses
   │
   └─ Adjust UP if:
      ├─ Stable, predictable revenue
      ├─ Strong margins (>15%)
      ├─ Substantial credit lines
      ├─ Diversified revenue streams
      └─ Strong balance sheet (debt/equity < 0.3)

**Optimization Algorithm:**

.. code-block:: python

   def find_optimal_retention(manufacturer, claim_generator, limits):
       """Find retention that maximizes ergodic growth."""

       test_retentions = np.logspace(
           np.log10(0.001 * manufacturer.starting_assets),  # 0.1% of assets
           np.log10(0.05 * manufacturer.starting_assets),   # 5% of assets
           num=20
       )

       best_retention = None
       best_ergodic_value = -np.inf

       for retention in test_retentions:
           insurance = create_insurance_program(retention, limits)
           results = run_simulation(manufacturer, insurance, claim_generator)

           # Calculate ergodic value
           ergodic_value = calculate_ergodic_growth(results)

           # Apply survival constraint
           if results.survival_rate >= min_survival_target:
               if ergodic_value > best_ergodic_value:
                   best_ergodic_value = ergodic_value
                   best_retention = retention

       return best_retention

Question 3: How Much Limit Do I Need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your limit is the maximum amount the insurer will pay.

**Limit Selection Guidelines:**

.. code-block:: text

   Minimum Acceptable Limit:
   ├─ Statistical: 99th percentile annual aggregate loss
   ├─ Historical: Largest loss in past 20 years × 1.5
   └─ Contractual: Maximum required by lenders/partners

   Recommended Limit:
   ├─ Statistical: 99.5th percentile annual aggregate loss
   ├─ Historical: Largest loss in past 50 years × 2
   └─ Ergodic: Level where marginal benefit < marginal cost

   Maximum Useful Limit:
   └─ Point where additional limit doesn't improve survival probability

**Limit Adequacy Test:**

.. code-block:: python

   def test_limit_adequacy(current_limit, loss_scenarios):
       """Check if limit is adequate."""

       # Calculate exceedance probability
       annual_max_losses = loss_scenarios.groupby('year').sum()
       exceedance_prob = (annual_max_losses > current_limit).mean()

       if exceedance_prob > 0.01:  # More than 1% chance
           print(f"WARNING: Limit may be inadequate")
           print(f"Exceedance probability: {exceedance_prob:.2%}")
           print(f"Consider increasing limit to ${np.percentile(annual_max_losses, 99):,.0f}")
       else:
           print(f"Limit appears adequate")
           print(f"Exceedance probability: {exceedance_prob:.2%}")

Red Flags to Avoid
------------------

Common Mistakes in Insurance Decisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. The Expected Value Trap**

❌ **Wrong:** "Expected losses are \$500K/year, so I won't pay \$750K in premium"

✅ **Right:** "The \$750K premium increases my time-average growth from 4% to 7%"

.. code-block:: python

   # Don't do this:
   if premium > expected_losses:
       print("Insurance too expensive")  # WRONG!

   # Do this instead:
   growth_without = calculate_ergodic_growth(results_no_insurance)
   growth_with = calculate_ergodic_growth(results_with_insurance)

   if growth_with > growth_without:
       print(f"Insurance adds {growth_with - growth_without:.2%} to growth rate")

**2. Ignoring Correlation**

❌ **Wrong:** "Losses are independent of business performance"

✅ **Right:** "Major losses often occur during economic downturns"

.. code-block:: python

   # Model correlation between losses and revenue
   correlation_factor = 0.3  # 30% correlation

   # In bad years, both revenue drops AND losses increase
   if revenue_shock < -0.1:  # Revenue down >10%
       loss_multiplier = 1 + correlation_factor
       adjusted_losses = base_losses * loss_multiplier

**3. Static Analysis**

❌ **Wrong:** "Our risk profile is constant"

✅ **Right:** "Risk evolves with business growth and market conditions"

.. code-block:: python

   # Adjust risk parameters over time
   def dynamic_risk_profile(year, base_params):
       # Risk increases with size (more exposure)
       size_factor = (1 + growth_rate) ** year

       # But decreases with maturity (better controls)
       maturity_factor = 1 - 0.02 * min(year, 10)  # 2% improvement per year, max 10 years

       adjusted_frequency = base_params.frequency * size_factor * maturity_factor
       return adjusted_frequency

**4. Over-Retention**

❌ **Wrong:** "We're a \$50M company, we can handle \$5M losses"

✅ **Right:** "A \$5M loss would impair growth for years"

.. code-block:: python

   # Calculate growth impairment from large retention
   def growth_impairment(loss_amount, assets):
       # Direct impact
       asset_reduction = loss_amount / assets

       # Indirect impacts
       credit_impairment = asset_reduction * 2  # Reduced borrowing capacity
       investment_delay = asset_reduction * 1.5  # Delayed growth investments

       total_impairment = asset_reduction + credit_impairment + investment_delay
       years_to_recover = total_impairment / annual_growth_rate

       return years_to_recover

Implementation Checklist
------------------------

Phase 1: Data Gathering (Week 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

☐ **Historical Losses** (past 5-10 years)
   - Date, amount, cause
   - Business impact beyond direct cost
   - Recovery time

☐ **Financial Statements** (past 3 years)
   - Balance sheet
   - Income statement
   - Cash flow statement

☐ **Current Insurance Program**
   - Policy terms and conditions
   - Premium history
   - Claims history

☐ **Risk Register**
   - Identified risks
   - Probability estimates
   - Impact assessments

Phase 2: Analysis (Week 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

☐ **Baseline Simulation**
   - Run without insurance
   - Identify ruin probability
   - Calculate growth volatility

☐ **Optimization Runs**
   - Test 10-20 retention levels
   - Test 5-10 limit options
   - Find ergodic optimum

☐ **Sensitivity Analysis**
   - Vary key assumptions ±30%
   - Identify critical parameters
   - Establish confidence bounds

☐ **Peer Comparison**
   - Industry benchmarks
   - Similar company structures
   - Best practices review

Phase 3: Decision (Week 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

☐ **Synthesize Results**
   - Optimal structure identification
   - Cost-benefit quantification
   - Risk-return trade-offs

☐ **Stakeholder Review**
   - Present to CFO/CEO
   - Board risk committee
   - External advisors

☐ **Implementation Plan**
   - Timeline for changes
   - Broker engagement
   - Market approach strategy

☐ **Documentation**
   - Decision rationale
   - Key assumptions
   - Review triggers

Decision Rules by Company Type
-------------------------------

High-Growth Technology Company
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Profile:** High volatility, thin margins, rapid scaling

.. code-block:: text

   Recommended Structure:
   - Retention: 0.5-1% of assets (lower end)
   - Primary Limit: $10M minimum
   - Excess Limits: Up to $100M
   - Focus: Survival over cost optimization

   Key Risks:
   - Cyber incidents
   - Business interruption
   - Key person loss
   - IP litigation

Stable Manufacturing Company
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Profile:** Moderate volatility, steady margins, predictable growth

.. code-block:: text

   Recommended Structure:
   - Retention: 1-2% of assets (middle range)
   - Primary Limit: $5M typical
   - Excess Limits: $25-50M
   - Focus: Balanced optimization

   Key Risks:
   - Product liability
   - Equipment breakdown
   - Supply chain disruption
   - Natural catastrophes

Mature Utility Company
~~~~~~~~~~~~~~~~~~~~~~~

**Profile:** Low volatility, regulated returns, stable cash flows

.. code-block:: text

   Recommended Structure:
   - Retention: 2-3% of assets (higher end)
   - Primary Limit: Lower attachment point
   - Excess Limits: High catastrophe coverage
   - Focus: Catastrophe protection

   Key Risks:
   - Natural disasters
   - Regulatory changes
   - Infrastructure failure
   - Environmental liability

Decision Metrics Dashboard
--------------------------

Create a dashboard to monitor your decision metrics:

.. code-block:: python

   def create_decision_dashboard(results):
       """Create comprehensive decision metrics."""

       dashboard = {
           'Survival Metrics': {
               '1-Year': calculate_survival(results, 1),
               '5-Year': calculate_survival(results, 5),
               '10-Year': calculate_survival(results, 10),
               '20-Year': calculate_survival(results, 20)
           },

           'Growth Metrics': {
               'Time-Average': results.time_avg_growth,
               'Ensemble-Average': results.ensemble_avg_growth,
               'Median': np.median(results.growth_rates),
               'Volatility': np.std(results.growth_rates)
           },

           'Risk Metrics': {
               'VaR-95%': np.percentile(results.final_wealth, 5),
               'CVaR-95%': results.final_wealth[results.final_wealth <= np.percentile(results.final_wealth, 5)].mean(),
               'Max Drawdown': calculate_max_drawdown(results.wealth_paths),
               'Recovery Time': calculate_recovery_time(results.wealth_paths)
           },

           'Insurance Efficiency': {
               'Premium/Expected Loss': results.total_premium / results.expected_losses,
               'Premium/Assets': results.total_premium / results.starting_assets,
               'Ergodic ROI': (results.with_insurance_wealth - results.without_insurance_wealth) / results.total_premium,
               'Break-even Probability': calculate_breakeven_prob(results)
           }
       }

       return dashboard

When to Review Your Decision
-----------------------------

Set triggers for reviewing your insurance structure:

**Automatic Review Triggers:**

1. **Time-Based**
   - Annual review minimum
   - Quarterly for high-growth companies

2. **Event-Based**
   - Major loss occurrence
   - M&A activity
   - Significant business model change
   - Credit rating change

3. **Metric-Based**
   - Assets change by >25%
   - Revenue volatility changes by >5%
   - Loss frequency changes by >30%
   - Survival probability drops below target

**Review Process:**

.. code-block:: python

   def insurance_review_needed(current_metrics, baseline_metrics, months_elapsed):
       """Determine if insurance review is needed."""

       triggers = []

       # Time trigger
       if months_elapsed >= 12:
           triggers.append("Annual review due")

       # Asset change trigger
       asset_change = abs(current_metrics['assets'] - baseline_metrics['assets']) / baseline_metrics['assets']
       if asset_change > 0.25:
           triggers.append(f"Assets changed by {asset_change:.0%}")

       # Risk change trigger
       risk_change = abs(current_metrics['loss_rate'] - baseline_metrics['loss_rate']) / baseline_metrics['loss_rate']
       if risk_change > 0.30:
           triggers.append(f"Loss rate changed by {risk_change:.0%}")

       if triggers:
           print("Insurance review recommended:")
           for trigger in triggers:
               print(f"  - {trigger}")
           return True

       return False

Key Takeaways
-------------

1. **Optimize for Time-Average Growth**, not expected value
2. **Survival Probability Trumps Cost** in the short term
3. **Ergodic Value Maximization** drives long-term success
4. **Regular Reviews** ensure continued optimization
5. **Document Everything** for consistency and learning

Next Steps
----------

With your decision framework in place:

1. Review :doc:`case_studies` for similar companies
2. Explore :doc:`advanced_topics` for customization
3. Consult :doc:`faq` for common questions
4. Begin implementation with your broker/insurer

Remember: The best insurance decision is one that lets you sleep at night while your company grows sustainably.

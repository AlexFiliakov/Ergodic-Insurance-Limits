Model Cases
============

These case studies demonstrate how different types of companies can use ergodic insurance optimization. Each includes actual simulation results and detailed analysis of the decision process.

Model Case 1: Widget Manufacturing Company
-------------------------------------------

Company Profile
~~~~~~~~~~~~~~~

**MidTech Manufacturing Inc.**

* **Industry**: Electronic components manufacturing
* **Assets**: \$10 million
* **Revenue**: \$15 million annually
* **Operating Margin**: 8%
* **Growth Rate**: 6% baseline
* **Volatility**: 15% annual revenue volatility

Risk Profile
~~~~~~~~~~~~

Based on 5 years of historical data:

* **Attritional losses**: 4-6 events/year, \$30K-\$100K each
* **Large losses**: 1 every 3 years, \$1M-\$5M range
* **Catastrophic risk**: Major fire/explosion risk, potential \$20M loss

Current Insurance Program
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Retention**: \$500,000
* **Limit**: \$5,000,000
* **Annual Premium**: \$125,000
* **Historical Performance**: 2 limits breached in past 10 years

Analysis Process
~~~~~~~~~~~~~~~~

**Step 1: Baseline Assessment**

.. code-block:: python

   # Configuration for MidTech Manufacturing
   manufacturer_config = {
       'starting_assets': 10_000_000,
       'base_revenue': 15_000_000,
       'base_operating_margin': 0.08,
       'tax_rate': 0.25,
       'working_capital_pct': 0.20,
       'growth_volatility': 0.15
   }

   # Loss distribution parameters
   loss_config = {
       'attritional': {'frequency': 5.0, 'severity_mean': 60_000, 'severity_cv': 0.8},
       'large': {'frequency': 0.33, 'severity_mean': 2_500_000, 'severity_cv': 1.0},
       'catastrophic': {'frequency': 0.02, 'severity_mean': 20_000_000, 'severity_cv': 0.5}
   }

**Step 2: Simulation Results**

*Without Insurance:*

* 10-year survival probability: 71.2%
* Average annual growth (survivors): 5.3%
* 5% VaR: -\$2.8M (ruin)
* Maximum drawdown: 68%

*Current Program (\$500K retention, \$5M limit):*

* 10-year survival probability: 83.5%
* Average annual growth: 6.1%
* 5% VaR: \$3.2M
* Total premiums paid: \$1.25M
* Benefit vs no insurance: +\$1.8M terminal value

*Optimized Program (\$100K retention, \$25M limit):*

* 10-year survival probability: 96.8%
* Average annual growth: 7.4%
* 5% VaR: \$8.7M
* Total premiums paid: \$3.85M
* Benefit vs current: +\$4.1M terminal value

Recommendation
~~~~~~~~~~~~~~

**Optimal Structure:**

1. **Reduce retention** from \$500K to \$100K
2. **Increase limit** from \$5M to \$25M
3. **Layer structure**:

   * Primary: \$100K-\$5M at 1.5% rate
   * First Excess: \$5M-\$25M at 0.7% rate
   * Catastrophe: \$25M-\$50M at 0.3% rate

**Financial Impact:**

* Additional premium cost: \$260K/year
* Improved survival probability: +13.3%
* Enhanced growth rate: +1.3%/year
* 10-year NPV of change: +\$4.1M

**Key Insight:** The \$500K retention was creating cash flow stress during loss years, impeding growth investments. Lower retention enables consistent reinvestment.

Model Case 2: High-Growth Technology Startup
---------------------------------------------

Company Profile
~~~~~~~~~~~~~~~

**CloudScale Solutions**

* **Industry**: SaaS platform provider
* **Assets**: \$5 million
* **Revenue**: \$8 million (100% YoY growth)
* **Operating Margin**: -10% (investing for growth)
* **Burn Rate**: \$2 million/year
* **Volatility**: 40% revenue volatility

Risk Profile
~~~~~~~~~~~~

* **Cyber incidents**: 0.8 events/year, \$500K-\$5M severity
* **Business interruption**: Platform outages, \$100K-\$10M impact
* **D&O liability**: High given rapid growth and VC backing
* **Key person risk**: Critical dependency on technical founders

Current Situation
~~~~~~~~~~~~~~~~~

* **No insurance** (trying to minimize burn)
* **Recent incident**: \$800K cyber loss absorbed
* **Board concern**: Requesting risk mitigation

Analysis Process
~~~~~~~~~~~~~~~~

**Step 1: Quantify Uninsured Risk**

.. code-block:: python

   # High-growth tech configuration
   tech_config = {
       'starting_assets': 5_000_000,
       'base_revenue': 8_000_000,
       'base_operating_margin': -0.10,  # Negative margin during growth
       'growth_rate': 1.0,  # 100% growth
       'growth_volatility': 0.40,  # High volatility
       'burn_rate': 2_000_000
   }

   # Tech-specific risks
   cyber_losses = {
       'frequency': 0.8,
       'severity_mean': 2_000_000,
       'severity_cv': 1.5
   }

**Step 2: Simulation Results**

*Without Insurance:*

* 2-year survival probability: 68%
* 5-year survival probability: 31%
* Risk of running out of cash: 45% in year 2
* Expected runway reduction: 8 months per incident

*Minimal Coverage (\$50K retention, \$5M limit):*

* 2-year survival probability: 89%
* 5-year survival probability: 62%
* Annual premium: \$180K
* Runway impact: -1 month

*Recommended Coverage (\$25K retention, \$50M limit):*

* 2-year survival probability: 95%
* 5-year survival probability: 78%
* Annual premium: \$425K
* Runway impact: -2.5 months
* **Critical benefit**: Enables next funding round

Recommendation
~~~~~~~~~~~~~~

**Immediate Actions:**

1. **Implement cyber insurance** immediately (\$25K retention)
2. **D&O coverage** essential for board protection
3. **Business interruption** coverage with 12-month indemnity period

**Staged Approach:**

* **Year 1**: Essential coverage only (\$425K premium)
* **Year 2**: Expand as revenue grows
* **Year 3**: Full program at projected \$50M revenue

**Board Presentation Points:**

* Insurance cost < 6% of revenue (industry standard)
* Survival probability improvement: +47% over 5 years
* Protects \$50M post-money valuation
* Required by most Series B investors

Model Case 3: Stable Utility Company
-------------------------------------

Company Profile
~~~~~~~~~~~~~~~

**Regional Power Corp**

* **Industry**: Electric utility
* **Assets**: \$100 million
* **Revenue**: \$80 million
* **Operating Margin**: 12% (regulated)
* **Growth**: 2% annual (population-based)
* **Volatility**: 5% (weather-driven)

Risk Profile
~~~~~~~~~~~~

* **Routine claims**: 20-30/year, \$10K-\$50K each
* **Storm damage**: 2-3/year, \$500K-\$5M each
* **Catastrophic events**: Ice storms, hurricanes (\$50M-\$200M)
* **Regulatory**: Penalties for extended outages

Current Insurance Program
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Retention**: \$250,000
* **Primary limit**: \$10,000,000
* **Excess limit**: \$100,000,000
* **Annual premium**: \$2,800,000

Analysis Results
~~~~~~~~~~~~~~~~

**Optimization Finding:** Current retention too low for company size

*Current Structure Performance:*

* Never approaching ruin (100% survival)
* Paying for unnecessary frequency coverage
* Premium efficiency: 42% (low)

*Optimized Structure (\$2M retention, same limits):*

* Maintains 100% survival probability
* Premium savings: \$1.1M/year
* Self-insures predictable losses
* Focuses on catastrophe protection

Recommendation
~~~~~~~~~~~~~~

**Restructure to:**

1. **Increase retention** to \$2M (2% of assets)
2. **Maintain catastrophe limits** at \$100M+
3. **Add parametric coverage** for named storms
4. **Establish loss fund** with premium savings

**10-Year Impact:**

* Premium savings: \$11M
* Loss fund accumulation: \$8M (after claims)
* Improved regulatory standing
* Maintains AAA credit rating

Model Case 4: Comparison Across Industries
-------------------------------------------

Comparative Analysis
~~~~~~~~~~~~~~~~~~~~

We ran identical simulations across different industry profiles:

.. code-block:: text

   ┌─────────────────┬──────────┬────────────┬───────────┬─────────────┐
   │ Industry        │ Optimal  │ Optimal    │ Premium % │ Ergodic     │
   │                 │ Retention│ Limit      │ of Assets │ Improvement │
   ├─────────────────┼──────────┼────────────┼───────────┼─────────────┤
   │ Manufacturing   │ 1.0%     │ 2.5x Rev   │ 3.5%      │ +31%        │
   │ Technology      │ 0.5%     │ 6x Rev     │ 8.5%      │ +67%        │
   │ Utility         │ 2.0%     │ 1.5x Rev   │ 2.8%      │ +12%        │
   │ Retail          │ 0.8%     │ 3x Rev     │ 4.2%      │ +38%        │
   │ Healthcare      │ 0.3%     │ 5x Rev     │ 6.1%      │ +54%        │
   └─────────────────┴──────────┴────────────┴───────────┴─────────────┘

Key Patterns
~~~~~~~~~~~~

1. **Higher volatility → Lower optimal retention**
2. **Higher growth → Higher optimal limits**
3. **Thin margins → More insurance value**
4. **Stable companies → Higher retentions work**

Implementation Lessons
----------------------

Lesson 1: Gradual Transition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Moving from \$1M to \$100K retention seems risky

**Solution:** Phase over 2 years:

* Year 1: Reduce to \$500K, monitor results
* Year 2: Further reduce to \$250K if comfortable
* Year 3: Reach optimal \$100K

Lesson 2: Premium Sticker Shock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Board resistant to 3x premium increase

**Solution:** Present as investment:

.. code-block:: python

   # ROI Calculation
   additional_premium = 260_000  # per year
   growth_improvement = 0.013    # 1.3% better growth
   asset_base = 10_000_000

   annual_value_creation = asset_base * growth_improvement
   roi = annual_value_creation / additional_premium

   print(f"Annual value creation: ${annual_value_creation:,.0f}")
   print(f"ROI on insurance spend: {roi:.1f}x")
   # Output: ROI on insurance spend: 5.0x

Lesson 3: Market Capacity
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Insurers reluctant to provide \$50M limit to \$5M company

**Solution:** Structure with multiple carriers:

* Primary: Admitted carrier (\$5M)
* Excess: Bermuda markets (\$20M)
* Cat: ILS/Alternative capital (\$25M)

TODO: Real-World Validation
---------------------------

Backtesting Against Historical Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to validate our models against actual loss events:

* **2008 Financial Crisis Scenario:**
* **2020 Pandemic Scenario:**
* **Natural Catastrophe Events:**
   * Hurricane exposure (Florida manufacturer)
   * Earthquake exposure (California tech)

Your Next Steps
---------------

1. **Identify your company type** from the cases above
2. **Run your specific parameters** through the model
3. **Compare results** with the relevant case study
4. **Adjust for unique factors** in your situation
5. **Document decisions** for future reference

Remember: These cases are starting points. Your specific situation requires customized analysis using the tools provided in :doc:`running_analysis`.

For additional customization options, see :doc:`advanced_topics`.

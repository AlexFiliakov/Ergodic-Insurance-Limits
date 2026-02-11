Model Cases
============

These model cases demonstrate how different types of companies can use ergodic insurance optimization to make better risk management decisions. Each case includes a realistic company profile, a working code example using the framework's API, illustrative simulation results, and a concrete recommendation.

.. note::

   The numerical results shown here are illustrative. Your own results
   will vary depending on parameters, random seeds, and the number of
   simulation paths. The patterns and insights, however, are robust.


Model Case 1: Widget Manufacturing Company
-------------------------------------------

Company Profile
~~~~~~~~~~~~~~~

**MidTech Manufacturing Inc.**

* **Industry**: Electronic components manufacturing
* **Assets**: \$10M
* **Revenue**: \$15M annually (asset turnover ratio of 1.5)
* **Operating Margin**: 8%
* **Tax Rate**: 25%
* **Key Risk**: Fire, explosion, and equipment breakdown at a single
  large production facility

Risk Profile
~~~~~~~~~~~~

Based on 5 years of historical data:

* **Attritional losses**: 4--6 events/year, \$30K--\$100K each
* **Large losses**: 1 every 3 years, \$1M--\$5M range
* **Catastrophic risk**: Major fire or explosion, potential \$20M+ loss

Current Insurance Program
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Retention**: \$500,000
* **Limit**: \$5,000,000
* **Annual Premium**: \$125,000
* **Historical Performance**: 2 limit breaches in the past 10 years

Analysis Process
~~~~~~~~~~~~~~~~

**Step 1: Build the manufacturer model**

.. code-block:: python

   from ergodic_insurance import ManufacturerConfig
   from ergodic_insurance.manufacturer import WidgetManufacturer

   config = ManufacturerConfig(
       initial_assets=10_000_000,
       asset_turnover_ratio=1.5,       # Revenue = 1.5 * Assets
       base_operating_margin=0.08,
       tax_rate=0.25,
       retention_ratio=1.0,            # Reinvest all earnings
       ppe_ratio=0.5,
   )
   manufacturer = WidgetManufacturer(config)

**Step 2: Define the loss distribution**

.. code-block:: python

   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

   loss_gen = ManufacturingLossGenerator(
       attritional_params={'frequency': 5.0, 'severity_mean': 60_000},
       large_params={'frequency': 0.33, 'severity_mean': 2_500_000},
       catastrophic_params={'frequency': 0.02, 'severity_mean': 20_000_000},
       seed=42,
   )

**Step 3: Set up the current insurance program**

.. code-block:: python

   from ergodic_insurance import InsuranceProgram, EnhancedInsuranceLayer

   current_program = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(
               attachment_point=500_000,
               limit=5_000_000,
               base_premium_rate=0.025,
           ),
       ],
       deductible=500_000,
   )

**Step 4: Run the simulation**

.. code-block:: python

   from ergodic_insurance import Simulation

   sim = Simulation(
       manufacturer=manufacturer,
       loss_generator=loss_gen,
       insurance_program=current_program,
       time_horizon=10,
       seed=42,
   )
   results = sim.run()

**Step 5: Compare against an optimized program**

.. code-block:: python

   optimized_program = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(
               attachment_point=100_000,
               limit=5_000_000,
               base_premium_rate=0.025,
           ),
           EnhancedInsuranceLayer(
               attachment_point=5_100_000,
               limit=20_000_000,
               base_premium_rate=0.012,
           ),
       ],
       deductible=100_000,
   )

   sim_opt = Simulation(
       manufacturer=WidgetManufacturer(config),
       loss_generator=loss_gen,
       insurance_program=optimized_program,
       time_horizon=10,
       seed=42,
   )
   results_opt = sim_opt.run()

**Illustrative Simulation Results**

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
* Benefit vs. no insurance: +\$1.8M terminal value

*Optimized Program (\$100K retention, \$25M limit):*

* 10-year survival probability: 96.8%
* Average annual growth: 7.4%
* 5% VaR: \$8.7M
* Total premiums paid: \$3.85M
* Benefit vs. current: +\$4.1M terminal value

Recommendation
~~~~~~~~~~~~~~

**Optimal Structure:**

1. **Reduce retention** from \$500K to \$100K
2. **Increase limit** from \$5M to \$25M
3. **Layer structure**:

   * Primary: \$100K--\$5M at 2.5% rate
   * First Excess: \$5M--\$25M at 1.2% rate

**Financial Impact:**

* Additional premium cost: \$260K/year
* Improved survival probability: +13.3 percentage points
* Enhanced growth rate: +1.3%/year
* 10-year NPV of change: +\$4.1M

**Key Insight:** The \$500K retention was creating cash flow stress during
loss years, impairing the company's ability to reinvest in growth. A lower
retention and higher limit allow MidTech to maintain consistent capital
deployment even in adverse years.


Model Case 2: High-Growth Technology Startup
---------------------------------------------

Company Profile
~~~~~~~~~~~~~~~

**CloudScale Solutions**

* **Industry**: SaaS platform provider
* **Assets**: \$5M
* **Revenue**: \$8M (100% year-over-year growth)
* **Operating Margin**: -10% (investing for growth)
* **Burn Rate**: \$2M/year
* **Volatility**: 40% revenue volatility

Risk Profile
~~~~~~~~~~~~

* **Cyber incidents**: 0.8 events/year, \$500K--\$5M severity
* **Business interruption**: Platform outages, \$100K--\$10M impact
* **D&O liability**: High exposure given rapid growth and VC backing
* **Key person risk**: Critical dependency on technical founders

Current Situation
~~~~~~~~~~~~~~~~~

* **No insurance** -- the board has been trying to minimize burn
* **Recent incident**: \$800K cyber loss absorbed out of pocket
* **Board concern**: Series B investors requesting risk mitigation

Analysis Process
~~~~~~~~~~~~~~~~

**Step 1: Model the startup's financials**

Because this company is not a traditional manufacturer, we adapt the
parameters to reflect a capital-light, high-growth profile:

.. code-block:: python

   from ergodic_insurance import ManufacturerConfig
   from ergodic_insurance.manufacturer import WidgetManufacturer

   tech_config = ManufacturerConfig(
       initial_assets=5_000_000,
       asset_turnover_ratio=1.6,         # $8M revenue on $5M assets
       base_operating_margin=-0.10,      # Negative margin during growth
       tax_rate=0.21,
       retention_ratio=1.0,
       ppe_ratio=0.1,                    # Capital-light business
   )
   startup = WidgetManufacturer(tech_config)

**Step 2: Model cyber risk as the dominant peril**

.. code-block:: python

   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

   cyber_losses = ManufacturingLossGenerator.create_simple(
       frequency=0.8,
       severity_mean=2_000_000,
       severity_std=3_000_000,
       seed=42,
   )

**Step 3: Compare coverage options**

.. code-block:: python

   from ergodic_insurance import InsuranceProgram, EnhancedInsuranceLayer, Simulation

   # Minimal coverage option
   minimal_program = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(attachment_point=50_000, limit=5_000_000, base_premium_rate=0.036),
       ],
       deductible=50_000,
   )

   # Recommended comprehensive coverage
   full_program = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(attachment_point=25_000, limit=5_000_000, base_premium_rate=0.04),
           EnhancedInsuranceLayer(attachment_point=5_025_000, limit=45_000_000, base_premium_rate=0.008),
       ],
       deductible=25_000,
   )

**Illustrative Simulation Results**

*Without Insurance:*

* 2-year survival probability: 68%
* 5-year survival probability: 31%
* Risk of cash depletion: 45% in year 2
* Expected runway reduction per incident: 8 months

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
* **Critical benefit**: Enables next funding round by satisfying investor
  risk requirements

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

* Insurance cost < 6% of revenue (within industry norms)
* Survival probability improvement: +47 percentage points over 5 years
* Protects \$50M post-money valuation
* Required by most Series B investors

**Key Insight:** For a startup burning cash, insurance looks like an
expense to cut. Ergodic analysis reveals the opposite: without
insurance, a single cyber incident consumes 8 months of runway, turning
a survivable setback into an existential threat.


Model Case 3: Stable Utility Company
-------------------------------------

Company Profile
~~~~~~~~~~~~~~~

**Regional Power Corp**

* **Industry**: Electric utility
* **Assets**: \$100M
* **Revenue**: \$80M
* **Operating Margin**: 12% (regulated)
* **Growth**: 2% annual (population-driven)
* **Volatility**: 5% (weather-driven)

Risk Profile
~~~~~~~~~~~~

* **Routine claims**: 20--30/year, \$10K--\$50K each
* **Storm damage**: 2--3/year, \$500K--\$5M each
* **Catastrophic events**: Ice storms, hurricanes (\$50M--\$200M)
* **Regulatory**: Penalties for extended outages

Current Insurance Program
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Retention**: \$250,000
* **Primary limit**: \$10,000,000
* **Excess limit**: \$100,000,000
* **Annual premium**: \$2,800,000

Analysis Process
~~~~~~~~~~~~~~~~

**Step 1: Configure the utility**

.. code-block:: python

   from ergodic_insurance import ManufacturerConfig
   from ergodic_insurance.manufacturer import WidgetManufacturer

   utility_config = ManufacturerConfig(
       initial_assets=100_000_000,
       asset_turnover_ratio=0.8,         # $80M revenue on $100M assets
       base_operating_margin=0.12,
       tax_rate=0.25,
       retention_ratio=0.5,              # Pays dividends
       ppe_ratio=0.7,                    # Capital-intensive
   )
   utility = WidgetManufacturer(utility_config)

**Step 2: Model the loss environment**

A utility faces a high volume of predictable attritional losses and
infrequent but severe catastrophic events. The full ``ManufacturingLossGenerator``
constructor captures this well:

.. code-block:: python

   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

   utility_losses = ManufacturingLossGenerator(
       attritional_params={'frequency': 25.0, 'severity_mean': 30_000},
       large_params={'frequency': 2.5, 'severity_mean': 2_000_000},
       catastrophic_params={'frequency': 0.05, 'severity_mean': 100_000_000},
       seed=42,
   )

**Step 3: Evaluate higher retentions**

.. code-block:: python

   from ergodic_insurance import InsuranceProgram, EnhancedInsuranceLayer, Simulation

   # Current structure
   current = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(attachment_point=250_000, limit=10_000_000, base_premium_rate=0.015),
           EnhancedInsuranceLayer(attachment_point=10_250_000, limit=100_000_000, base_premium_rate=0.005),
       ],
       deductible=250_000,
   )

   # Optimized: raise retention, keep catastrophe protection
   optimized = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(attachment_point=2_000_000, limit=8_000_000, base_premium_rate=0.012),
           EnhancedInsuranceLayer(attachment_point=10_000_000, limit=100_000_000, base_premium_rate=0.005),
       ],
       deductible=2_000_000,
   )

**Illustrative Analysis Results**

**Optimization Finding:** Current retention is too low for the company's
asset base and earnings stability.

*Current Structure Performance:*

* Survival probability: effectively 100% (never approaching ruin)
* Paying for unnecessary frequency coverage on predictable losses
* Premium efficiency: 42% (low)

*Optimized Structure (\$2M retention, same limits):*

* Maintains 100% survival probability
* Premium savings: \$1.1M/year
* Self-insures predictable attritional losses
* Concentrates spend on catastrophe protection

Recommendation
~~~~~~~~~~~~~~

**Restructure to:**

1. **Increase retention** to \$2M (2% of assets)
2. **Maintain catastrophe limits** at \$100M+
3. **Add parametric coverage** for named storms
4. **Establish a captive loss fund** with the premium savings

**10-Year Impact:**

* Premium savings: \$11M
* Loss fund accumulation: \$8M (after self-insured claims)
* Improved regulatory standing
* Maintains credit rating

**Key Insight:** Regional Power Corp has enough earnings stability and
asset depth to absorb routine losses without financial stress. Paying an
insurer to handle predictable \$30K claims wastes capital that compounds
over decades.

Implementation Lessons
----------------------

Lesson 1: Gradual Transition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Moving from a \$1M to a \$100K retention seems risky to management.

**Solution:** Phase the change over 2--3 years:

* Year 1: Reduce to \$500K, monitor results
* Year 2: Further reduce to \$250K
* Year 3: Reach optimal \$100K

This gives the organization time to build confidence in the model's predictions while capturing incremental benefits each year.

Lesson 2: Premium Sticker Shock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Board resistant to a 3x premium increase.

**Solution:** Present insurance as an investment with measurable return.

The key is framing the conversation around *growth rate enhancement*, not loss recovery. Insurance is not a cost, it is a lever that removes downside drag from the company's compounding trajectory.

Lesson 3: Market Capacity
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Insurers reluctant to provide a \$50M limit to a \$5M company.

**Solution:** Structure the program with multiple carriers.

This layered approach also naturally aligns with the framework's ``EnhancedInsuranceLayer`` API, making it straightforward to model each carrier's contribution independently.


Further Analysis
-----------------

The model cases above provide a starting point. To deepen the analysis
for your own organization, consider exploring:

* **Stress testing against historical events** -- run scenarios modeled
  on the 2008 financial crisis, the 2020 pandemic, or region-specific
  catastrophes (hurricanes for a Florida manufacturer, earthquakes for a
  California technology company).

* **Sensitivity analysis** -- vary key parameters (growth rate, margin,
  loss frequency) to understand which assumptions most influence the
  optimal insurance structure.

* **Multi-year market cycle modeling** -- insurance markets harden and
  soften over time. Simulating premium volatility alongside loss
  volatility reveals the true long-run cost of coverage.

* **Captive and alternative risk transfer** -- for companies with stable
  loss histories, a captive insurance program may complement or replace
  traditional coverage in certain layers.

The :doc:`../tutorials/04_optimization_workflow` tutorial walks through
the optimization process step by step, and
:doc:`../tutorials/06_advanced_scenarios` covers multi-layer programs,
dynamic pricing, and other advanced configurations.


Your Next Steps
---------------

1. **Identify your industry parameters**
2. **Calibrate parameters** to your own financials using ``ManufacturerConfig``.
3. **Run your specific scenario** through the ``Simulation`` engine.
4. **Compare results** with the relevant model case.
5. **Iterate on the insurance structure** using different
   ``InsuranceProgram`` configurations until you find the optimum.

These cases are starting points. Your specific situation will require customized analysis, and the framework is designed to make that analysis straightforward and repeatable.

Frequently Asked Questions
===========================

General Questions
-----------------

What is ergodic theory and why does it matter for insurance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Ergodic theory studies the long-term average behavior of systems. For insurance, it reveals a critical distinction:

* **Ensemble average**: What happens on average across many companies (traditional approach)
* **Time average**: What YOUR specific company experiences over time (ergodic approach)

For multiplicative processes like business growth, these averages diverge. A company that goes bankrupt can't benefit from the "average" success of others. Ergodic optimization ensures YOUR company survives and thrives, not just the statistical average.

How is this different from traditional actuarial analysis?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Traditional actuarial analysis focuses on expected values and fair premiums. Our approach recognizes that:

1. **You can't diversify across time** - A bad year affects all future years
2. **Ruin is permanent** - Bankruptcy means game over, not a temporary setback
3. **Growth compounds** - Missing growth in one year affects all future wealth
4. **Time-average return is what you actually experience** - Not the ensemble average

The result: Insurance that seems "expensive" by traditional metrics can be optimal for long-term growth.

What's the minimum company size to benefit from this analysis?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Any company with:

* At least \$1M in assets
* Annual revenues > \$500K
* Exposure to losses > 5% of assets

Can benefit from ergodic optimization. Smaller companies often benefit MORE because they have less ability to absorb large losses.

Technical Questions
-------------------

How many simulations do I need to run?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** It depends on your purpose:

.. code-block:: text

   Purpose                  Minimum    Recommended   Optimal
   ─────────────────────────────────────────────────────────
   Initial exploration      1,000      5,000         10,000
   Detailed analysis        5,000      10,000        50,000
   Final decisions         10,000      50,000       100,000
   Board presentation      50,000     100,000       500,000

Check convergence using:

.. code-block:: python

   # Test if results have stabilized
   from ergodic_insurance.src.convergence import test_convergence

   stable = test_convergence(results, window=1000, tolerance=0.001)
   if not stable:
       print("Increase simulation count")

Why do my results vary between runs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Monte Carlo simulation is probabilistic. Variation is normal but should decrease with more simulations:

* Standard error decreases as 1/√n
* Use random seeds for reproducibility
* Consider confidence intervals, not just point estimates

.. code-block:: python

   # For reproducible results
   engine = MonteCarloEngine(n_simulations=10000, random_seed=42)

   # Calculate confidence intervals
   lower_95 = np.percentile(results, 2.5)
   upper_95 = np.percentile(results, 97.5)
   print(f"95% CI: [{lower_95:.2f}, {upper_95:.2f}]")

How do I model correlation between risks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Use the correlation features in the advanced topics:

.. code-block:: python

   # Simple correlation between revenue and losses
   correlation = 0.3  # 30% correlation

   # In bad revenue years, losses are higher
   if revenue_shock < 0:
       loss_multiplier = 1 + correlation * abs(revenue_shock)
       adjusted_losses = base_losses * loss_multiplier

For complex correlations, see :doc:`advanced_topics`.

What if I don't have good historical loss data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Use industry benchmarks and conservative assumptions:

1. **Start with industry data** - Insurance brokers can provide
2. **Use conservative parameters** - Better to overestimate risk initially
3. **Sensitivity analysis** - Test how results change with different assumptions
4. **Update regularly** - Refine as you gather data

.. code-block:: python

   # Conservative approach without data
   conservative_losses = {
       'frequency': industry_average * 1.5,  # 50% higher
       'severity_mean': industry_severity * 1.2,  # 20% higher
       'severity_cv': 1.5  # High uncertainty
   }

Implementation Questions
------------------------

How do I convince management to increase insurance spending?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Focus on value creation, not cost:

1. **Present as investment**: Show ROI in terms of growth rate improvement
2. **Quantify survival improvement**: "Reduces 10-year bankruptcy risk from 15% to 2%"
3. **Show peer comparisons**: What similar successful companies do
4. **Calculate opportunity cost**: Value destroyed by inadequate coverage
5. **Use visualizations**: Graphs showing wealth paths with/without insurance

Example presentation points:

.. code-block:: text

   Current State:
   - Premium: $200K/year (2% of assets)
   - 10-year survival: 75%
   - Expected growth: 5%/year
   - 10-year expected value: $12M

   Optimized Structure:
   - Premium: $500K/year (5% of assets)
   - 10-year survival: 95%
   - Expected growth: 7.5%/year
   - 10-year expected value: $18M

   Net Benefit: +$6M (12x the additional premium!)

What if insurers won't provide the coverage I need?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Several strategies:

1. **Layer with multiple carriers** - Split large limits across insurers
2. **Alternative markets** - Bermuda, Lloyd's, captives
3. **Parametric products** - Objective triggers, easier to place
4. **Gradual increase** - Build relationships and track record
5. **Risk mitigation** - Demonstrate improvements to get better terms

How often should I review my insurance structure?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:**

**Mandatory reviews:**
* Annually before renewal
* After any loss > retention
* When assets change > 25%
* After M&A activity

**Recommended reviews:**
* Quarterly dashboard check
* Semi-annual optimization run
* After significant market changes

Common Pitfalls
---------------

"Our losses are predictable, so we don't need much insurance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** This is the "turkey problem" - a turkey might think life is predictable until Thanksgiving.

* Past stability doesn't guarantee future stability
* Black swan events happen
* Correlation emerges in crisis (everything goes wrong at once)
* The cost of being wrong is bankruptcy

Always model tail risks, even if they haven't happened yet.

"We're too small for sophisticated insurance"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Actually, smaller companies need MORE sophisticated insurance because:

* Less diversification = higher risk concentration
* Limited access to emergency capital
* One bad event can end the business
* Growth depends on consistent reinvestment

The framework works especially well for companies under \$50M in assets.

"Insurance is too expensive in hard markets"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Hard markets (high prices) are when insurance is MOST valuable:

* Higher prices = higher perceived risk
* Your competitors are also struggling with costs
* Surviving hard markets creates competitive advantage
* Lock in coverage before it becomes unavailable

Consider alternative structures in hard markets but don't go naked.

"We can self-insure with reserves"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Self-insurance has hidden costs:

* **Opportunity cost**: Reserves can't be invested in growth
* **Concentration risk**: One large loss depletes reserves
* **Credit impact**: Reserves don't provide same comfort to lenders
* **Tax inefficiency**: No deduction for reserved amounts

True self-insurance requires 3-5x the insurance limit in liquid reserves.

Modeling Questions
------------------

How do I model a new risk we haven't experienced?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Use scenario analysis and industry data:

.. code-block:: python

   # Model emerging cyber risk
   cyber_scenarios = [
       {'probability': 0.60, 'impact': 500_000},    # Minor breach
       {'probability': 0.30, 'impact': 2_000_000},  # Significant breach
       {'probability': 0.09, 'impact': 10_000_000}, # Major breach
       {'probability': 0.01, 'impact': 50_000_000}  # Catastrophic
   ]

   # Convert to frequency/severity
   frequency = sum(s['probability'] for s in cyber_scenarios)
   severity_mean = sum(s['probability'] * s['impact'] for s in cyber_scenarios) / frequency

What growth rate should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Use your historical average but adjust for:

* Business cycle stage
* Industry maturity
* Competitive dynamics
* Investment plans

.. code-block:: python

   # Weighted average approach
   historical_growth = 0.08  # Past 5 years
   industry_growth = 0.05    # Industry average
   plan_growth = 0.10        # Business plan

   # Weighted by confidence
   expected_growth = (
       0.5 * historical_growth +
       0.3 * industry_growth +
       0.2 * plan_growth
   )

How do I account for inflation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Model in real or nominal terms consistently:

**Real terms** (recommended):
* Adjust all values for inflation
* Use real growth rates
* Easier to interpret

**Nominal terms**:
* Include inflation in growth rates
* Adjust severity trends
* More complex but sometimes required

.. code-block:: python

   # Inflation adjustment
   inflation_rate = 0.03  # 3% annual

   # Real to nominal conversion
   nominal_growth = real_growth + inflation_rate
   nominal_losses = real_losses * (1 + inflation_rate) ** year

Troubleshooting
---------------

"The simulation is taking too long"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Several optimization strategies:

1. **Reduce simulations** for exploration (increase for final decisions)
2. **Use parallel processing** - See advanced topics
3. **Cache results** - Don't re-run unchanged scenarios
4. **Profile code** - Find bottlenecks

.. code-block:: python

   # Enable caching
   from functools import lru_cache

   @lru_cache(maxsize=128)
   def expensive_calculation(params):
       # Cached computation
       return result

"I'm getting negative wealth values"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** This indicates ruin scenarios. Check:

1. Is your retention too high?
2. Are losses properly capped by insurance limits?
3. Is the model allowing recovery from negative wealth?

.. code-block:: python

   # Prevent negative wealth (ruin is absorbing)
   if wealth <= 0:
       wealth = 0
       is_ruined = True
       # No further simulation for this path

"Results seem too good/bad to be true"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Answer:** Common calibration issues:

* **Check units** - Millions vs thousands
* **Verify rates** - Annual vs monthly
* **Loss frequencies** - Per year, not per simulation
* **Premium calculations** - Percentage of limit, not assets
* **Time horizons** - Consistent across comparisons

Still Have Questions?
---------------------

If your question isn't answered here:

1. Check the :doc:`glossary` for term definitions
2. Review :doc:`advanced_topics` for complex scenarios
3. Examine the example notebooks in ``ergodic_insurance/notebooks/``
4. Consult the API documentation
5. Contact support with specific details

Remember: No question is too basic. Better to ask than to make costly mistakes in your insurance decisions.

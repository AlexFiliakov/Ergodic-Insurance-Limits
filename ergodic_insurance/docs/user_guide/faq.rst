Frequently Asked Questions
===========================

Conceptual
----------

What is the "ergodic" approach and why should I care?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traditional actuarial pricing uses **ensemble averages** -- the mean outcome across many independent companies. Ergodic analysis uses **time averages** -- the compounded outcome a single company actually experiences over successive years.

For multiplicative processes like business growth, these two averages diverge. A company that suffers ruin in year 5 does not participate in the ensemble's "average" recovery. The time average captures this asymmetry; the ensemble average hides it.

The practical consequence: insurance that looks expensive under expected-value pricing can improve a company's long-run growth rate by preventing the irreversible losses that drag down compounded returns.

How is this different from what my actuary already does?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard actuarial analysis prices insurance by comparing expected losses to premiums and applies risk loads for volatility. That framework answers "Is this premium fair?" but not "Does buying this policy make my company grow faster over 10 years?"

This framework answers the second question by simulating year-over-year wealth paths and measuring the **time-average growth rate**, the geometric mean of annual returns, rather than the arithmetic mean. When tail losses can compound into ruin, the two metrics give different answers about the value of insurance.

Does this mean expected-value analysis is wrong?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. Expected values are correct for one-shot, diversifiable bets. They become misleading when:

1. Outcomes compound multiplicatively (each year's equity depends on the prior year's)
2. Ruin is absorbing (you cannot recover from zero)
3. You cannot diversify across time (you live one path, not the ensemble)

Most businesses satisfy all three conditions. The framework quantifies how much that matters for a given risk profile and insurance structure.


Using the Framework
-------------------

What data do I need to get started?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At minimum:

- **Assets** and **annual revenue** (to configure ``ManufacturerConfig``)
- **Operating margin** and **tax rate**
- **Loss frequency and severity estimates** (even rough ones work for ``ManufacturingLossGenerator.create_simple()``)

If you lack historical loss data, use industry benchmarks from your broker and run a sensitivity analysis to see how results change with different assumptions (see :doc:`../tutorials/04_optimization_workflow`).

What is the difference between InsurancePolicy and InsuranceProgram?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

They serve different simulation engines:

- **``InsurancePolicy``** with **``InsuranceLayer``** -- used by the single-path ``Simulation`` engine. Each layer has an ``attachment_point``, ``limit``, and ``rate``.

- **``InsuranceProgram``** with **``EnhancedInsuranceLayer``** -- used by ``MonteCarloEngine``. Supports reinstatements, aggregate limits, participation rates, and per-occurrence vs. aggregate limit types.

Start with ``InsurancePolicy`` for exploration. Move to ``InsuranceProgram`` when you need Monte Carlo analysis or advanced layer features. See :doc:`../tutorials/03_configuring_insurance` for a walkthrough.

How many simulations should I run?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For single-path ``Simulation``, running 20--50 seeds gives a reasonable picture of path variability.

For ``MonteCarloEngine``, start with 1,000 simulations for exploration and increase to 10,000+ for decisions you plan to present. If tail statistics (e.g., ruin probability below 5%) are important, use 50,000 or more. You can check convergence by running the same analysis at two different simulation counts and comparing the metrics that matter to your decision.

Why do results change between runs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monte Carlo simulation is stochastic. Set a starting seed for reproducibility:

.. code-block:: python

   from ergodic_insurance.monte_carlo import SimulationConfig

   config = SimulationConfig(n_simulations=10_000, seed=42)

Standard error decreases as 1/sqrt(n). If a metric swings materially between runs at your chosen simulation count, increase it.


Interpretation
--------------

What does "time-average growth rate" actually measure?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is the geometric mean of annual growth factors along a single wealth path:

.. math::

   g_{\text{time}} = \left(\prod_{t=1}^{T} \frac{W_t}{W_{t-1}}\right)^{1/T} - 1

If your company starts at $10M and ends at $18M after 10 years, the time-average growth rate is about 6.1% regardless of the path taken. The framework averages this quantity across simulated paths.

This is the growth rate you *actually experience*, as opposed to the arithmetic mean of annual returns (the ensemble average), which overstates realized growth when returns are volatile.

Can insurance really improve growth even after paying premiums?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, under specific conditions. Insurance truncates the left tail of annual returns. Removing the worst outcomes raises the geometric mean even after subtracting premium costs from every year's return. The effect is largest when:

- Loss severity is high relative to equity
- The company has thin margins (limited ability to absorb shocks)
- The time horizon is long (compounding amplifies the effect)

The framework lets you quantify the break-even premium loading at which this benefit disappears. Tutorial 3 walks through a concrete example (:doc:`../tutorials/03_configuring_insurance`).

How do I present these results to a board or CFO?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Focus on three metrics:

1. **Survival probability** over the planning horizon (e.g., "95% vs. 75% over 10 years")
2. **Median terminal equity** with and without insurance
3. **Time-average growth rate differential** (the ergodic advantage)

Pair these with a visualization showing insured vs. uninsured wealth paths. Tutorial 5 includes a four-panel chart designed for board presentations (:doc:`../tutorials/05_analyzing_results`).


Troubleshooting
---------------

My survival rate is very low.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check your retention. If retention exceeds 5% of assets, a single large loss can push the company toward ruin. Try lowering the deductible or adding a primary layer with a lower attachment point.

Results seem unreasonable (too good or too bad).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common calibration mistakes:

- **Units**: mixing thousands and millions in asset or loss parameters
- **Rates**: using monthly figures where annual ones are expected (or vice versa)
- **Loss frequency**: specifying per-simulation frequency instead of per-year
- **Premium rate**: applying the rate to assets instead of to the layer limit

Double-check ``ManufacturerConfig`` fields against your source data and verify ``InsuranceLayer`` parameters match your broker's terms.

The simulation is slow.
~~~~~~~~~~~~~~~~~~~~~~~~

For ``MonteCarloEngine``, enable parallelism:

.. code-block:: python

   config = SimulationConfig(
       n_simulations=50_000,
       n_years=10,
       parallel=True,
       n_workers=4,
   )

For exploration, reduce ``n_simulations`` first. See :doc:`../tutorials/06_advanced_scenarios` for performance tuning details.


Further Reading
---------------

- :doc:`glossary` -- definitions of technical terms used in this documentation
- :doc:`case_studies` -- worked examples across manufacturing, technology, and utility sectors
- :doc:`../tutorials/index` -- step-by-step tutorials from installation to advanced scenarios

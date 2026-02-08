Quick Start Guide
=================

This guide will help you run your first insurance optimization analysis in
under 10 minutes. By the end you'll have a complete insured-vs-uninsured
comparison with survival rates, growth metrics, and visualizations.

Prerequisites
-------------

Before starting, ensure you have:

1. Python 3.12+ with the ``ergodic_insurance`` package installed
2. A Python IDE or Jupyter notebook

You'll also want these numbers handy for your company:

* Current assets/capital
* Operating margin
* Expected loss frequency and severity
* Historical loss data (if available)

Step 1: Run Your First Simulation
----------------------------------

The ``run_analysis()`` function is the fastest path to results — one import,
one call. It builds the company model, generates losses, applies insurance,
runs Monte Carlo simulations, and compares insured vs uninsured outcomes:

.. code-block:: python
   :caption: first_simulation.py

   from ergodic_insurance import run_analysis

   results = run_analysis(
       # Company
       initial_assets=10_000_000,   # $10M asset base
       operating_margin=0.08,       # 8% profit margin

       # Losses
       loss_frequency=5,            # ~5 losses per year
       loss_severity_mean=100_000,  # Average $100K each

       # Insurance
       deductible=100_000,          # You pay first $100K
       coverage_limit=5_000_000,    # Insurer pays up to $5M above that
       premium_rate=0.015,          # 1.5% of limit = $75K/year

       # Simulation
       n_simulations=1_000,
       time_horizon=10,             # 10-year horizon
       seed=42,                     # Reproducible results
   )
   print(results.summary())

That's it. ``run_analysis()`` returns an :class:`AnalysisResults` object with
everything you need:

.. code-block:: python

   # Export per-simulation metrics to a DataFrame
   df = results.to_dataframe()
   print(df.head())

   # Quick 2x2 visualization (survival, equity, ROE, fan chart)
   results.plot()

Key Parameters Explained
~~~~~~~~~~~~~~~~~~~~~~~~~

**initial_assets**: Your company's current capital base. This is what
insurance protects.

**operating_margin**: Profit margin before extraordinary items. Higher margins
provide more buffer against losses.

**loss_frequency / loss_severity_mean**: How often losses occur and how big
they are on average. The framework uses a Poisson process for frequency and a
lognormal distribution for severity.

**deductible**: Your self-insured retention — the amount you pay out of pocket
before insurance kicks in.

**coverage_limit**: Maximum the insurer pays per occurrence above the
deductible.

**premium_rate**: Annual premium as a fraction of the coverage limit.

Step 2: Understanding Your Risk Profile
----------------------------------------

Insurance decisions depend on your loss exposure. We model three categories:

**Attritional Losses** (High Frequency, Low Severity)
   * Equipment breakdowns, minor accidents, small liability claims
   * Typically retained by the company
   * With the Step 1 defaults: ~4.5 losses/year averaging $50K each

**Large Losses** (Medium Frequency, Medium Severity)
   * Major equipment failure, significant liability event, supply chain disruption
   * Primary insurance layer target
   * With the Step 1 defaults: ~1 loss every 2 years averaging $200K

**Catastrophic Losses** (Low Frequency, High Severity)
   * Natural disasters, major product recall, cyber attack
   * Excess insurance layer target
   * With the Step 1 defaults: ~1 loss every 1,000 years, Pareto-distributed above $500K

The ``run_analysis()`` call above automatically splits your ``loss_frequency``
and ``loss_severity_mean`` into all three tiers: 90% of frequency as
attritional at half the mean severity, 10% as large at 2x the mean, plus rare
Pareto-distributed catastrophic events at 5x the mean. For full control over
each tier's parameters, see ``docs/tutorials/03_configuring_insurance.md``.

Step 3: Exploring Insurance Structures
---------------------------------------

A real insurance program is a tower of layers, each covering a slice of loss:

::

    Loss Amount    Coverage           Annual Premium
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    $50M ┌─────────────────────┐
         │   Second Excess      │     $100K
    $25M ├─────────────────────┤
         │   First Excess       │     $160K
     $5M ├─────────────────────┤
         │   Primary Layer      │     $75K
    $100K├─────────────────────┤
         │   Retention          │     You Pay
      $0 └─────────────────────┘
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                              Total Premium: $335K/year

For quick single-layer policies, ``run_analysis()`` handles everything
internally. You can also build a policy object directly:

.. code-block:: python

   from ergodic_insurance import InsurancePolicy

   policy = InsurancePolicy.from_simple(
       deductible=100_000,
       limit=5_000_000,
       premium_rate=0.015,
   )

For multi-layer towers and advanced structures, see
``docs/tutorials/03_configuring_insurance.md``.

Step 4: Interpreting Results
-----------------------------

Your simulation will produce metrics like:

.. code-block:: text

   ============================================================
   Ergodic Insurance Analysis Summary
   ============================================================
   Simulations: 1000
   Time Horizon: 10 years

   --- Insured Scenario ---
   Survival Rate: 947/1000 (94.7%)
   Mean Final Equity (survivors): $17,800,000
   Median Final Equity (survivors): $16,200,000
   Mean Time-Weighted ROE: 7.20%
   Annual Premium: $75,000

   --- Uninsured Scenario ---
   Survival Rate: 723/1000 (72.3%)
   Mean Final Equity (survivors): $14,200,000
   Mean Time-Weighted ROE: 5.80%

   --- Ergodic Advantage (Insured - Uninsured) ---
   Time-Average Growth Gain: +1.40%
   Survival Rate Gain: +22.4%
   Statistically Significant: Yes
   ============================================================

Key Metrics to Focus On
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Survival Rate**: Percentage of scenarios avoiding ruin
2. **Time-Average Growth**: Your actual experienced growth rate (the ergodic measure)
3. **Terminal Wealth Distribution**: Range of possible outcomes
4. **Ergodic Advantage**: The growth rate gain from insurance (this is the core insight)

Step 5: Explore Pre-Built Notebooks
-------------------------------------

For interactive analysis, use the pre-configured Jupyter notebooks:

**Basic Analysis** (``notebooks/01_basic_manufacturer.ipynb``)
   Start here to understand the manufacturer model and basic simulations.

**Long-Term Simulations** (``notebooks/02_long_term_simulation.ipynb``)
   Explore 10, 20, and 50-year horizons to see compounding effects.

**Growth Dynamics** (``notebooks/03_growth_dynamics.ipynb``)
   Understand how insurance affects growth trajectories.

**Ergodic Demo** (``notebooks/04_ergodic_demo.ipynb``)
   See the difference between time and ensemble averages.

**Risk Metrics** (``notebooks/05_risk_metrics.ipynb``)
   Calculate VaR, CVaR, and other risk measures.

To run a notebook:

.. code-block:: bash

   cd ergodic_insurance/notebooks
   jupyter notebook 01_basic_manufacturer.ipynb

Quick Decision Rules
---------------------

Based on thousands of simulations, here are rules of thumb:

**When to Buy More Insurance:**
   * Survival rate < 90% over 10 years
   * VaR shows negative terminal wealth
   * Growth volatility > 20%
   * Correlation between revenue and losses > 0.3

**Optimal Retention Level:**
   * Start with 1-2% of assets
   * Lower if: High volatility, thin margins
   * Higher if: Stable revenue, strong balance sheet

**Limit Selection:**
   * Minimum: 99th percentile annual loss
   * Recommended: 99.5th percentile
   * Consider: Largest historical loss x 2

Common Issues
-------------

**"My survival rate is very low"**
   Your retention might be too high. Try reducing it by 50%.

**"Insurance seems too expensive"**
   Check if you're modeling correlation between losses and revenue correctly.

**"Results vary significantly between runs"**
   Increase ``n_simulations`` to 10,000 for more stable results.

**"How do I model my specific industry?"**
   See :doc:`advanced_topics` for customizing loss distributions.

Next Steps
----------

Now that you've run your first simulation:

1. Proceed to :doc:`running_analysis` for detailed analysis procedures
2. Use :doc:`decision_framework` to interpret results
3. Review :doc:`case_studies` for similar companies
4. Explore :doc:`advanced_topics` for customization

Quick Start Guide
=================

This guide will help you get started with insurance optimization analysis in under 30 minutes. We'll walk through setting up your company profile, understanding your risks, and running your first simulation.

Prerequisites
-------------

Before starting, ensure you have:

1. Python environment with the ergodic_insurance package installed
2. Access to Jupyter notebooks or Python IDE
3. Basic company financial information:

   * Current assets/capital
   * Annual revenue
   * Operating margin
   * Historical loss data (if available)

Step 1: Understanding Your Company Profile
-------------------------------------------

First, let's define your company using our configuration system. Create a YAML file describing your business:

.. code-block:: yaml
   :caption: my_company.yaml

   company:
     name: "My Manufacturing Co"
     starting_assets: 10_000_000  # Current asset base in dollars

   financial:
     base_revenue: 15_000_000     # Annual revenue
     base_operating_margin: 0.08        # 8% profit margin
     tax_rate: 0.25                # 25% corporate tax

   growth:
     base_growth_rate: 0.06        # 6% organic growth
     growth_volatility: 0.15       # 15% standard deviation

Key Parameters Explained
~~~~~~~~~~~~~~~~~~~~~~~~~

**Starting Assets**: Your company's current capital base. This is what you're protecting with insurance.

**Operating Margin**: Your profit margin before extraordinary items. Higher margins provide more buffer against losses.

**Growth Volatility**: How much your revenue varies year-to-year. Tech companies might be 30-40%, utilities 5-10%.

Step 2: Defining Your Risk Profile
-----------------------------------

Next, characterize your loss exposure. We categorize losses into three buckets:

.. code-block:: yaml
   :caption: my_risk_profile.yaml

   losses:
     attritional:
       frequency: 5.0              # Average 5 small losses per year
       severity_mean: 50_000       # Average $50K per loss
       severity_cv: 0.8            # Coefficient of variation

     large:
       frequency: 0.3              # One large loss every ~3 years
       severity_mean: 2_000_000    # Average $2M when occurs
       severity_cv: 1.2

     catastrophic:
       frequency: 0.02             # One catastrophic loss every ~50 years
       severity_mean: 15_000_000   # Average $15M when occurs
       severity_cv: 0.5

Understanding Loss Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Attritional Losses** (High Frequency, Low Severity)
   * Equipment breakdowns
   * Minor accidents
   * Small liability claims
   * Typically retained by company

**Large Losses** (Medium Frequency, Medium Severity)
   * Major equipment failure
   * Significant liability event
   * Supply chain disruption
   * Primary insurance layer target

**Catastrophic Losses** (Low Frequency, High Severity)
   * Natural disasters
   * Major product recall
   * Cyber attack
   * Excess insurance layer target

Step 3: Exploring Insurance Structures
---------------------------------------

Our framework models multi-layer insurance programs. Here's a typical structure:

.. code-block:: yaml
   :caption: insurance_structure.yaml

   insurance_program:
     retention: 100_000            # You pay first $100K of any loss

     layers:
       - name: "Primary"
         limit: 5_000_000          # Covers $100K to $5.1M
         base_premium_rate: 0.015       # 1.5% of limit = $75K/year

       - name: "First Excess"
         limit: 20_000_000         # Covers $5.1M to $25.1M
         base_premium_rate: 0.008       # 0.8% of limit = $160K/year

       - name: "Second Excess"
         limit: 25_000_000         # Covers $25.1M to $50.1M
         base_premium_rate: 0.004       # 0.4% of limit = $100K/year

Visualizing Your Insurance Tower
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Step 4: Running Your First Simulation
--------------------------------------

Now let's run a basic simulation using Python:

.. code-block:: python
   :caption: first_simulation.py

   from ergodic_insurance.config import Config
   from ergodic_insurance.manufacturer import WidgetManufacturer
   from ergodic_insurance.insurance_program import InsuranceProgram, EnhancedInsuranceLayer
   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
   from ergodic_insurance.monte_carlo import MonteCarloEngine, SimulationConfig

   # Create a config with defaults — or use Config.from_company() for customization
   config = Config.from_company(
       initial_assets=10_000_000,
       operating_margin=0.08,
   )
   manufacturer = WidgetManufacturer(config.manufacturer)

   # Define insurance program
   insurance = InsuranceProgram(
       layers=[
           EnhancedInsuranceLayer(
               attachment_point=100_000, limit=5_000_000, base_premium_rate=0.015
           ),
           EnhancedInsuranceLayer(
               attachment_point=5_100_000, limit=20_000_000, base_premium_rate=0.008
           ),
       ],
       deductible=100_000,
   )

   # Configure and run Monte Carlo simulation
   loss_gen = ManufacturingLossGenerator.create_simple(
       frequency=5, severity_mean=100_000, severity_std=50_000, seed=42
   )
   sim_config = SimulationConfig(n_simulations=1_000, n_years=10, seed=42)

   engine = MonteCarloEngine(
       loss_generator=loss_gen,
       insurance_program=insurance,
       manufacturer=manufacturer,
       config=sim_config
   )
   results = engine.run()

   # Display results
   import numpy as np
   print(f"Mean Final Assets: ${np.mean(results.final_assets):,.0f}")
   print(f"Mean Growth Rate: {np.mean(results.growth_rates):.4f}")
   print(results.summary())

Step 5: Using Pre-Built Notebooks
----------------------------------

For easier analysis, use our pre-configured Jupyter notebooks:

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

Step 6: Interpreting Initial Results
-------------------------------------

Your first simulation will produce metrics like:

.. code-block:: text

   ===== Simulation Results =====
   Scenarios Run: 1,000
   Time Horizon: 10 years

   Without Insurance:
   - Survival Rate: 72.3%
   - Mean Growth (survivors): 5.8%/year
   - Median Terminal Wealth: $14.2M
   - 5% VaR: -$3.1M (ruin)

   With Insurance ($100K retention, $25M limit):
   - Survival Rate: 94.7%
   - Mean Growth: 7.2%/year
   - Median Terminal Wealth: $17.8M
   - 5% VaR: $8.9M
   - Total Premiums Paid: $3.35M
   - Net Benefit: +$3.6M

Key Metrics to Focus On
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Survival Rate**: Percentage of scenarios avoiding ruin
2. **Time-Average Growth**: Your actual experienced growth rate
3. **Terminal Wealth Distribution**: Range of possible outcomes
4. **Value at Risk (VaR)**: Worst-case scenarios (5th percentile)

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
   * Consider: Largest historical loss × 2

Next Steps
----------

Now that you've run your first simulation:

1. Proceed to :doc:`running_analysis` for detailed analysis procedures
2. Use :doc:`decision_framework` to interpret results
3. Review :doc:`case_studies` for similar companies
4. Explore :doc:`advanced_topics` for customization

Common Issues
-------------

**"My survival rate is very low"**
   Your retention might be too high. Try reducing it by 50%.

**"Insurance seems too expensive"**
   Check if you're modeling correlation between losses and revenue correctly.

**"Results vary significantly between runs"**
   Increase simulations to 10,000 for more stable results.

**"How do I model my specific industry?"**
   See :doc:`advanced_topics` for customizing loss distributions.

Ready for More?
---------------

You've successfully:
✓ Set up your company profile
✓ Defined your risk parameters
✓ Configured insurance structures
✓ Run your first simulation
✓ Interpreted basic results

Continue to :doc:`running_analysis` to dive deeper into optimization techniques.

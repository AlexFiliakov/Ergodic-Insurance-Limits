Quick Start Guide
=================

Get up and running with Ergodic Insurance Limits in minutes.

Basic Usage
-----------

1. Load Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.config_manager import ConfigManager

   # Initialize configuration manager
   manager = ConfigManager()

   # Load default configuration
   config = manager.load_profile("default")

2. Create Manufacturer
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.manufacturer import WidgetManufacturer

   # Create manufacturer with configuration
   manufacturer = WidgetManufacturer(config.manufacturer)

   print(f"Initial assets: ${manufacturer.total_assets:,.0f}")
   print(f"Base operating margin: {manufacturer.config.base_operating_margin:.1%}")

3. Generate Losses
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

   # Set up loss generator
   generator = ManufacturingLossGenerator.create_simple(
       frequency=5,
       severity_mean=100_000,
       severity_std=50_000,
       seed=42
   )

   # Generate annual losses
   losses, stats = generator.generate_losses(
       duration=1,
       revenue=10_000_000
   )

4. Run Simulation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.simulation import Simulation

   # Create and run simulation
   sim = Simulation(
       manufacturer=manufacturer,
       loss_generator=generator,
       time_horizon=10
   )

   results = sim.run()
   stats = results.summary_stats()

   print(f"Final equity: ${stats['final_equity']:,.0f}")
   print(f"Average ROE: {stats['mean_roe']:.1%}")

Complete Example
----------------

Here's a complete example that demonstrates the key features:

.. code-block:: python

   from ergodic_insurance.config_manager import ConfigManager
   from ergodic_insurance.manufacturer import WidgetManufacturer
   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
   from ergodic_insurance import InsurancePolicy, InsuranceLayer
   from ergodic_insurance.simulation import Simulation

   # Configuration
   manager = ConfigManager()
   config = manager.load_profile(
       "default",
       manufacturer={"base_operating_margin": 0.12},
       simulation={"time_horizon_years": 50}
   )

   # Setup
   manufacturer = WidgetManufacturer(config.manufacturer)
   generator = ManufacturingLossGenerator.create_simple(
       frequency=5,
       severity_mean=100_000,
       severity_std=50_000,
       seed=42
   )

   # Create insurance policy
   layer = InsuranceLayer(attachment_point=100_000, limit=5_000_000, rate=0.02)
   policy = InsurancePolicy(layers=[layer], deductible=100_000)

   # Run simulation with insurance
   sim = Simulation(
       manufacturer=manufacturer,
       loss_generator=generator,
       insurance_policy=policy,
       time_horizon=50
   )
   results = sim.run()

   print(f"Final equity: ${results.equity[-1]:,.0f}")
   print(f"Time-weighted ROE: {results.calculate_time_weighted_roe():.2%}")

Using Different Profiles
------------------------

Conservative Scenario
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load conservative profile
   config = manager.load_profile("conservative")

   # Lower growth, higher safety margins
   print(f"Growth rate: {config.growth.annual_growth_rate:.1%}")
   print(f"Operating margin: {config.manufacturer.base_operating_margin:.1%}")

Aggressive Growth
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load aggressive profile with overrides
   config = manager.load_profile(
       "aggressive",
       growth={"annual_growth_rate": 0.20},
       manufacturer={"retention_ratio": 0.95}
   )

Custom Scenarios
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create custom configuration
   config = manager.load_profile(
       "default",
       presets=["hard_market", "high_volatility"],
       modules=["insurance", "stochastic"],
       manufacturer={
           "initial_assets": 50_000_000,
           "base_operating_margin": 0.15
       }
   )

Ergodic Analysis
----------------

Compare time-average vs ensemble-average growth:

.. code-block:: python

   from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer

   analyzer = ErgodicAnalyzer()

   # Compare insured vs uninsured results
   comparison = analyzer.compare_scenarios(
       insured_results=results_insured,
       uninsured_results=results_uninsured,
       metric="equity"
   )

   print(f"Ergodic advantage: {comparison['ergodic_advantage']:.4f}")
   print(f"Insured time-avg growth: {comparison['insured_time_avg']:.4f}")
   print(f"Uninsured time-avg growth: {comparison['uninsured_time_avg']:.4f}")

Visualization
-------------

Quick visualizations of results:

.. code-block:: python

   from ergodic_insurance.visualization import plot_simulation_results

   # Run simulation (time_horizon set in constructor)
   results = sim.run()

   # Convert to DataFrame for analysis
   df = results.to_dataframe()
   print(df[['assets', 'equity', 'roe']].head(10))

Next Steps
----------

Now that you've run your first simulation:

1. **Explore Configurations**: See :doc:`config_best_practices`
2. **Understand the Theory**: Read :doc:`theory`
3. **Run Notebooks**: Try the Jupyter notebooks in ``ergodic_insurance/notebooks/``
4. **Customize**: Create your own profiles in ``data/config/profiles/custom/``
5. **Optimize**: Use :doc:`api/optimization` for advanced analysis

Tips
----

* Use caching for faster repeated simulations
* Start with shorter time horizons for testing
* Monitor convergence with :doc:`api/convergence`
* Save results to checkpoints for long simulations

Getting Help
------------

* Check the :doc:`user_guide/faq`
* Review :doc:`examples`
* See API documentation: :doc:`api/modules`
* Report issues on `GitHub <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues>`__

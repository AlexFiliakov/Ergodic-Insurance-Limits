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
   from ergodic_insurance.insurance import optimize_insurance_limit

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

   # Optimize insurance
   optimal_limit = optimize_insurance_limit(
       manufacturer=manufacturer,
       loss_generator=generator,
       limits_to_test=[5e6, 10e6, 15e6, 20e6],
       n_simulations=100
   )

   print(f"Optimal insurance limit: ${optimal_limit:,.0f}")

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

   # Analyze with and without insurance
   results = analyzer.compare_strategies(
       manufacturer=manufacturer,
       strategies={
           "no_insurance": {"limit": 0, "premium": 0},
           "basic": {"limit": 5_000_000, "base_premium_rate": 0.015},
           "comprehensive": {"limit": 20_000_000, "base_premium_rate": 0.012}
       },
       n_paths=1000,
       time_horizon=100
   )

   # Display results
   analyzer.plot_growth_comparison(results)

   for strategy, metrics in results.items():
       print(f"{strategy}:")
       print(f"  Time-average growth: {metrics['time_avg_growth']:.2%}")
       print(f"  Ensemble-average: {metrics['ensemble_avg']:.2%}")
       print(f"  Ergodic advantage: {metrics['ergodic_advantage']:.2%}")

Visualization
-------------

Quick visualizations of results:

.. code-block:: python

   from ergodic_insurance.visualization import plot_simulation_results

   # Run simulation
   results = sim.run(years=20)

   # Plot results
   plot_simulation_results(
       results,
       metrics=["assets", "equity", "roe"],
       title="20-Year Simulation"
   )

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

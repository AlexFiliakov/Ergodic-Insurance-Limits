Ergodic Insurance Limits Documentation
=======================================

.. image:: ../../assets/repo_banner_small.png
   :alt: Ergodic Insurance Limits
   :align: center

Welcome to the Ergodic Insurance Limits documentation! This project implements a framework for optimizing
insurance limits using ergodic (time-average) theory rather than traditional ensemble approaches.

.. note::
   **Version 2.0 Update**: The project now features a modern 3-tier configuration system with profiles,
   modules, and presets. See the :doc:`migration_guide` for upgrading from the legacy system.

Key Innovation
--------------

This framework demonstrates how insurance transforms from a cost center to a growth enabler when analyzed
through time averages, with potential for **30-50% better long-term performance** in widget manufacturing scenarios.

The key insight: **optimal insurance premiums can exceed expected losses by 200-500%** while still enhancing growth
when optimizing time-average growth rates rather than expected values.

Quick Start with Configuration v2
----------------------------------

.. code-block:: python

   from ergodic_insurance.src.config_manager import ConfigManager
   from ergodic_insurance.src.manufacturer import WidgetManufacturer
   from ergodic_insurance.src.simulation import run_simulation

   # Load configuration using the new system
   manager = ConfigManager()
   config = manager.load_profile(
       "default",
       presets=["steady_market"],
       manufacturer={"operating_margin": 0.10}
   )

   # Create manufacturer
   manufacturer = WidgetManufacturer(config.manufacturer)

   # Run simulation
   results = run_simulation(
       manufacturer=manufacturer,
       config=config.simulation,
       claims_config=config.losses if hasattr(config, 'losses') else None
   )

   print(f"Final equity: ${results.final_equity:,.0f}")
   print(f"Time-average growth: {results.time_average_growth:.2%}")

Configuration System Overview
-----------------------------

The new 3-tier configuration architecture:

1. **Profiles**: Complete configuration sets (default, conservative, aggressive)
2. **Modules**: Reusable components (insurance, losses, stochastic, business)
3. **Presets**: Quick-apply templates (market conditions, layer structures)

.. code-block:: yaml

   # Example: profiles/custom/high_growth.yaml
   extends: aggressive
   description: "High growth scenario for expansion phase"

   modules:
     - insurance
     - stochastic

   presets:
     - soft_market
     - growth_phase

   overrides:
     manufacturer:
       operating_margin: 0.15
     growth:
       annual_growth_rate: 0.25

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview
   installation
   quick_start
   migration_guide
   config_best_practices

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/executive_summary
   user_guide/decision_framework
   user_guide/running_analysis
   user_guide/case_studies

.. toctree::
   :maxdepth: 2
   :caption: Theory & Concepts

   theory
   ergodic_theory
   insurance_optimization
   risk_metrics

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
   api/config_manager
   api/manufacturer
   api/simulation
   api/insurance_program
   api/ergodic_analyzer

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/README
   architecture/context_diagram
   architecture/module_overview
   architecture/configuration_v2

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   user_guide/faq
   glossary
   changelog
   contributing

Key Features
------------

Configuration Management v2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **3-tier architecture** with profiles, modules, and presets
* **Profile inheritance** for easy customization
* **Runtime overrides** without file editing
* **Preset libraries** for common scenarios
* **Full backward compatibility** with legacy system

Financial Modeling
~~~~~~~~~~~~~~~~~~

* **Widget manufacturer model** with comprehensive balance sheet
* **Stochastic processes** including GBM and mean-reversion
* **Multi-year claim development** patterns
* **Letter of credit** collateral management

Insurance & Risk
~~~~~~~~~~~~~~~~

* **Multi-layer insurance programs** with attachment points
* **Comprehensive loss distributions** (attritional, large, catastrophic)
* **Risk metrics** including VaR, CVaR, and tail risk
* **Ergodic optimization** for time-average growth

Simulation & Analysis
~~~~~~~~~~~~~~~~~~~~~

* **Monte Carlo framework** with parallel processing
* **Convergence analysis** tools
* **Performance benchmarking** utilities
* **Comprehensive visualization** suite

Example: Loading Custom Profiles
---------------------------------

The new configuration system makes it easy to create and use custom scenarios:

.. code-block:: python

   # Load a stress test scenario
   config = manager.load_profile("custom/stress_test")

   # Or create one dynamically
   config = manager.load_profile(
       "conservative",
       modules=["insurance", "losses"],
       presets=["hard_market", "high_volatility"],
       manufacturer={
           "operating_margin": 0.04,  # Compressed margins
           "tax_rate": 0.30           # Higher taxes
       },
       simulation={
           "time_horizon_years": 10,
           "random_seed": 13
       }
   )

Example: Ergodic Analysis
-------------------------

.. code-block:: python

   from ergodic_insurance.src.ergodic_analyzer import ErgodicAnalyzer

   # Run ergodic analysis
   analyzer = ErgodicAnalyzer()

   # Compare time-average vs ensemble-average
   results = analyzer.analyze_insurance_impact(
       manufacturer=manufacturer,
       insurance_configs=[
           {"limit": 5_000_000, "premium_rate": 0.015},
           {"limit": 10_000_000, "premium_rate": 0.012},
           {"limit": 25_000_000, "premium_rate": 0.010}
       ],
       n_simulations=1000,
       time_horizon=100
   )

   # Plot ergodic advantage
   analyzer.plot_ergodic_advantage(results)

Performance & Testing
---------------------

The project maintains high quality standards:

* **100% test coverage** across 30+ test files
* **Type safety** with MyPy validation
* **Google-style docstrings** throughout
* **Pre-commit hooks** for code quality
* **Performance benchmarks** for all algorithms

Project Status
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Status
   * - Core Financial Model
     - ✅ Complete with stochastic extensions
   * - Configuration System v2
     - ✅ Fully implemented with migration tools
   * - Insurance Optimization
     - ✅ Multi-layer programs operational
   * - Ergodic Analysis
     - ✅ Framework complete
   * - Documentation
     - ✅ Comprehensive with examples
   * - Test Coverage
     - ✅ 100% achieved

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

Citation
--------

If you use this project in your research, please cite:

.. code-block:: bibtex

   @software{ergodic_insurance_2024,
     title={Ergodic Insurance Limits: A Time-Average Optimization Framework},
     author={Filiakov, Alex},
     year={2024},
     url={https://github.com/AlexFiliakov/Ergodic-Insurance-Limits}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

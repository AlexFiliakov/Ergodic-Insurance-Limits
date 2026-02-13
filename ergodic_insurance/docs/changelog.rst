Changelog
=========

Version 2.0.0 (2024-01-15)
--------------------------

Major Features
~~~~~~~~~~~~~~

* **Configuration System v2**: Complete redesign with 3-tier architecture

  - Profile-based configuration with inheritance
  - Module composition for feature sets
  - Preset library for common scenarios
  - 50% reduction in configuration files
  - Full backward compatibility through ConfigCompat

* **Enhanced Documentation**: Comprehensive documentation overhaul

  - Google-style docstrings across all modules
  - Sphinx documentation system integration
  - API reference auto-generation
  - User guides and tutorials

* **Stochastic Modeling**: Full stochastic process implementation

  - Geometric Brownian Motion (GBM)
  - Lognormal volatility shocks
  - Mean-reverting processes
  - Configurable random seeds for reproducibility

Improvements
~~~~~~~~~~~~

* **Testing**: Achieved 100% test coverage across all modules
* **Type Safety**: Complete mypy type checking compliance
* **Performance**: Optimized simulations with caching and vectorization
* **Validation**: Pydantic v2 models for all configuration

API Changes
~~~~~~~~~~~

* ``ConfigLoader`` deprecated in favor of ``ConfigManager``
* Unified ``Config`` Pydantic model replaces dictionary configs (``ConfigV2`` is a deprecated alias)
* ``load_config()`` -> ``load_profile()`` with enhanced options

Bug Fixes
~~~~~~~~~

* Fixed Unicode encoding issues in Windows environments
* Resolved caching problems with unhashable dict types
* Corrected test isolation issues with configuration state

Migration Guide
~~~~~~~~~~~~~~~

From v1.x to v2.0:

.. code-block:: python

   # Old way (still supported but deprecated)
   from ergodic_insurance.config_loader import ConfigLoader
   config = ConfigLoader.load_config("baseline")

   # New way
   from ergodic_insurance.config_manager import ConfigManager
   manager = ConfigManager()
   config = manager.load_profile("default")

Version 1.5.0 (2023-12-01)
--------------------------

Features
~~~~~~~~

* Multi-layer insurance program optimization
* Pareto frontier analysis for multi-objective optimization
* Enhanced visualization with interactive plots
* Claim development triangles for cash flow modeling

Improvements
~~~~~~~~~~~~

* 30% performance improvement in Monte Carlo simulations
* Better convergence diagnostics
* Extended loss distribution models

Bug Fixes
~~~~~~~~~

* Fixed numerical instability in extreme tail scenarios
* Corrected premium calculation for excess layers

Version 1.0.0 (2023-10-15)
--------------------------

Initial Release
~~~~~~~~~~~~~~~

* Core manufacturer financial model
* Basic insurance optimization
* Ergodic analysis framework
* Monte Carlo simulation engine
* Comprehensive test suite
* Documentation and examples

Core Features
~~~~~~~~~~~~~

* Widget manufacturer simulation
* Single-layer insurance optimization
* Time-average vs ensemble-average comparison
* Risk metrics calculation
* Basic visualization tools

Known Issues
~~~~~~~~~~~~

* Performance limitations for very long simulations (>1000 years)
* Limited to single-layer insurance programs
* No stochastic volatility modeling

Version 0.9.0 (2023-09-01) - Beta
----------------------------------

Pre-release Features
~~~~~~~~~~~~~~~~~~~~

* Proof of concept implementation
* Basic ergodic calculations
* Simple insurance model
* Initial test coverage

Limitations
~~~~~~~~~~~

* No configuration management
* Limited documentation
* Single scenario analysis only

Roadmap
-------

Version 2.1.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~~

* Real-time dashboard for simulation monitoring
* Web API for cloud deployment
* Enhanced optimization algorithms
* Machine learning integration for parameter estimation

Version 2.2.0 (Planned)
~~~~~~~~~~~~~~~~~~~~~~~~

* Multi-agent simulation framework
* Network effects and systemic risk
* Regulatory compliance modules
* Advanced catastrophe modeling

Version 3.0.0 (Future)
~~~~~~~~~~~~~~~~~~~~~~

* Complete UI/UX redesign
* Cloud-native architecture
* Real-world data integration
* Production-ready deployment tools

Contributing
------------

See :doc:`contributing` for guidelines on contributing to this project.

For detailed release notes and commit history, see the `GitHub repository <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits>`_.

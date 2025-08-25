Configuration Loader Module
===========================

The config_loader module provides utilities for loading, validating, and managing
configuration files with support for caching, overrides, and scenario-based configurations.

.. automodule:: ergodic_insurance.src.config_loader
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Classes
-------

.. autoclass:: ergodic_insurance.src.config_loader.ConfigLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. Key Methods section removed to avoid duplication with automodule
.. Methods are already included via automodule directive above

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from ergodic_insurance.src.config_loader import load_config

    # Load baseline configuration
    config = load_config("baseline")

    # Load with overrides
    config = load_config(
        "baseline",
        manufacturer__operating_margin=0.12,
        simulation__time_horizon_years=200
    )

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

    from ergodic_insurance.src.config_loader import ConfigLoader

    loader = ConfigLoader()

    # Compare scenarios
    differences = loader.compare_configs("baseline", "conservative")

    # List available configurations
    available = loader.list_available_configs()
    print(f"Available configs: {available}")

    # Load specific scenario
    config = loader.load_scenario("optimistic")

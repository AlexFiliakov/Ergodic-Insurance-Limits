Configuration Loader Module
===========================

The config_loader module provides utilities for loading, validating, and managing
configuration files with support for caching, overrides, and scenario-based configurations.

.. automodule:: config_loader
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: ConfigLoader
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Loading and Caching
~~~~~~~~~~~~~~~~~~~

.. automethod:: ConfigLoader.load
.. automethod:: ConfigLoader.load_scenario
.. automethod:: ConfigLoader.clear_cache

Scenario Management
~~~~~~~~~~~~~~~~~~~

.. automethod:: ConfigLoader.compare_configs
.. automethod:: ConfigLoader.list_available_configs
.. automethod:: ConfigLoader.validate_config

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_config

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from ergodic_insurance.config_loader import load_config

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

    from ergodic_insurance.config_loader import ConfigLoader

    loader = ConfigLoader()

    # Compare scenarios
    differences = loader.compare_configs("baseline", "conservative")

    # List available configurations
    available = loader.list_available_configs()
    print(f"Available configs: {available}")

    # Load specific scenario
    config = loader.load_scenario("optimistic")

config_manager module
=====================

.. automodule:: ergodic_insurance.src.config_manager
   :members:
   :undoc-members:
   :show-inheritance:

Related Modules
---------------

config_v2 module
~~~~~~~~~~~~~~~~

.. automodule:: ergodic_insurance.src.config_v2
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

config_migrator module
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ergodic_insurance.src.config_migrator
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

config_compat module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: ergodic_insurance.src.config_compat
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.src.config_manager import ConfigManager

   # Initialize manager
   manager = ConfigManager()

   # Load default profile
   config = manager.load_profile("default")

   # Load with overrides
   config = manager.load_profile(
       "default",
       manufacturer={"operating_margin": 0.12},
       simulation={"time_horizon_years": 50}
   )

Profile Composition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Combine modules and presets
   config = manager.load_profile(
       "default",
       modules=["insurance", "stochastic"],
       presets=["hard_market", "high_volatility"]
   )

   # Custom profile with inheritance
   config = manager.load_profile(
       "custom_profile",
       base_profile="conservative",
       presets=["catastrophic_risk"]
   )

Migration from Legacy
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.src.config_migrator import ConfigMigrator

   # Migrate legacy configs
   migrator = ConfigMigrator()
   results = migrator.run_migration()

   # Validate migration
   is_valid = migrator.validate_migration()

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.src.config_compat import LegacyConfigAdapter

   # Use legacy interface (deprecated)
   adapter = LegacyConfigAdapter()
   config = adapter.load_config("baseline")

   # Modern equivalent
   manager = ConfigManager()
   config_v2 = manager.load_profile("default")

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validate before loading
   manager = ConfigManager()
   config = manager.load_profile("my_profile")

   # Validate configuration
   is_valid = manager.validate(config)
   if not is_valid:
       print("Configuration validation failed")

See Also
--------

* :doc:`config_best_practices` - Best practices
* :doc:`../migration_guide` - Migration from v1
* :doc:`../examples` - More examples

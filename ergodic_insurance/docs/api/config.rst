Configuration Module
====================

The config module provides Pydantic-based configuration management with comprehensive
validation for all simulation parameters.

.. automodule:: config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
---------------------

Core Configuration
~~~~~~~~~~~~~~~~~~

.. autoclass:: Config
   :members:
   :undoc-members:
   :show-inheritance:

Financial Parameters
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ManufacturerConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WorkingCapitalConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DebtConfig
   :members:
   :undoc-members:
   :show-inheritance:

Growth and Simulation
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GrowthConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SimulationConfig
   :members:
   :undoc-members:
   :show-inheritance:

Output and Logging
~~~~~~~~~~~~~~~~~~

.. autoclass:: OutputConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LoggingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

.. automethod:: Config.from_yaml
.. automethod:: Config.from_dict  
.. automethod:: Config.override
.. automethod:: Config.to_yaml
.. automethod:: Config.setup_logging
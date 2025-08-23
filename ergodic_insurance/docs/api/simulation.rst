Simulation Module
=================

The simulation module provides the main simulation engine that orchestrates
the time evolution of the widget manufacturer financial model.

.. automodule:: simulation
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: Simulation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SimulationResults
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Simulation Execution
~~~~~~~~~~~~~~~~~~~~

.. automethod:: Simulation.run
.. automethod:: Simulation.run_to_dataframe
.. automethod:: Simulation.step

Results Analysis
~~~~~~~~~~~~~~~~

.. automethod:: SimulationResults.to_dataframe
.. automethod:: SimulationResults.summary_statistics

Performance Features
~~~~~~~~~~~~~~~~~~~~

The simulation engine is optimized for:

* **Long-term horizons**: 1000+ year simulations in under 1 minute
* **Large-scale analysis**: 100K Monte Carlo iterations in under 10 minutes
* **Memory efficiency**: Handles massive datasets with optimized storage
* **Progress tracking**: Real-time progress reporting for long simulations

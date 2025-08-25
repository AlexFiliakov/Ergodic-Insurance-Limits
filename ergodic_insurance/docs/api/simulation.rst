Simulation Module
=================

The simulation module provides the main simulation engine that orchestrates
the time evolution of the widget manufacturer financial model.

.. automodule:: ergodic_insurance.src.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: ergodic_insurance.src.simulation.Simulation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ergodic_insurance.src.simulation.SimulationResults
   :members:
   :undoc-members:
   :show-inheritance:

.. Key Methods section removed to avoid duplication with automodule
.. Methods are already included via automodule directive above

Performance Features
~~~~~~~~~~~~~~~~~~~~

The simulation engine is optimized for:

* **Long-term horizons**: 1000+ year simulations in under 1 minute
* **Large-scale analysis**: 100K Monte Carlo iterations in under 10 minutes
* **Memory efficiency**: Handles massive datasets with optimized storage
* **Progress tracking**: Real-time progress reporting for long simulations

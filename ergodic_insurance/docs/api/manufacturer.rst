Manufacturer Module
==================

The manufacturer module contains the core financial model for widget manufacturing companies.
This includes revenue generation, cost management, working capital, and debt financing.

.. automodule:: manufacturer
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: WidgetManufacturer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ClaimLiability
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Financial Operations
~~~~~~~~~~~~~~~~~~~~

.. automethod:: WidgetManufacturer.process_period
.. automethod:: WidgetManufacturer.calculate_revenue
.. automethod:: WidgetManufacturer.calculate_operating_income
.. automethod:: WidgetManufacturer.calculate_net_income

Asset Management
~~~~~~~~~~~~~~~~

.. automethod:: WidgetManufacturer.get_total_assets
.. automethod:: WidgetManufacturer.calculate_required_working_capital

Risk Management
~~~~~~~~~~~~~~~

.. automethod:: WidgetManufacturer.process_claim
.. automethod:: WidgetManufacturer.check_solvency

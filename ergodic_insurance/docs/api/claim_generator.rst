Claim Generator Module  
======================

The claim_generator module handles insurance claim generation with realistic
frequency and severity modeling for both attritional and large losses.

.. automodule:: claim_generator
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: ClaimGenerator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ClaimEvent
   :members:
   :undoc-members:  
   :show-inheritance:

Key Methods
-----------

Claim Generation
~~~~~~~~~~~~~~~~

.. automethod:: ClaimGenerator.generate_claims
.. automethod:: ClaimGenerator.generate_catastrophic_claims
.. automethod:: ClaimGenerator.generate_all_claims

Configuration
~~~~~~~~~~~~~

.. automethod:: ClaimGenerator.reset_seed
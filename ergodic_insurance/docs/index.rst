.. Ergodic Insurance Limits documentation master file, created by
   sphinx-quickstart on Sat Aug 23 14:48:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ergodic Insurance Limits Documentation
=======================================

Welcome to the documentation for Ergodic Insurance Limits, a revolutionary framework for optimizing
insurance limits using ergodic (time-average) theory rather than traditional ensemble approaches.

The framework demonstrates how insurance transforms from a cost center to a growth enabler when
analyzed through time averages, with potential for 30-50% better long-term performance.

.. note::
   **New to insurance optimization?** Start with our comprehensive :doc:`Business User Guide <user_guide/index>`
   designed for CFOs, risk managers, and actuaries. No advanced mathematics required!

Quick Links
-----------

.. raw:: html

   <div style="display: flex; gap: 20px; margin: 20px 0;">
     <div style="flex: 1; padding: 15px; border: 2px solid #4CAF50; border-radius: 8px;">
       <h3 style="margin-top: 0;">📊 Business User Guide</h3>
       <p>Complete guide for business decision makers</p>
       <a href="user_guide/index.html" style="color: #4CAF50; font-weight: bold;">Get Started in 10 Minutes →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #2196F3; border-radius: 8px;">
       <h3 style="margin-top: 0;">🚀 Quick Start</h3>
       <p>Run your first simulation</p>
       <a href="user_guide/quick_start.html" style="color: #2196F3; font-weight: bold;">Start Analysis →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #FF9800; border-radius: 8px;">
       <h3 style="margin-top: 0;">💡 Case Studies</h3>
       <p>Real-world examples</p>
       <a href="user_guide/case_studies.html" style="color: #FF9800; font-weight: bold;">View Cases →</a>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Business User Guide:
   :hidden:

   user_guide/index
   user_guide/executive_summary
   user_guide/quick_start
   user_guide/running_analysis
   user_guide/decision_framework
   user_guide/case_studies
   user_guide/advanced_topics
   user_guide/faq
   user_guide/glossary

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation:

   overview
   getting_started
   examples
   theory

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api/modules
   api/manufacturer
   api/config
   api/claim_generator
   api/simulation
   api/config_loader

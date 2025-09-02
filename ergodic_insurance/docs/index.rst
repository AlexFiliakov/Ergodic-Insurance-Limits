.. Ergodic Insurance Limits documentation master file, created by
   sphinx-quickstart on Sat Aug 23 14:48:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ergodic Insurance Limits Documentation
=======================================

Welcome to the documentation for Ergodic Insurance Limits, a framework for optimizing insurance
limits using ergodic (time-average) theory rather than traditional ensemble approaches.

The framework demonstrates how insurance transforms from a cost center to a growth enabler when
analyzed through time averages, with potential for 30-50% better long-term performance.

.. note::
   **New to insurance optimization?** Start with our comprehensive :doc:`Business User Guide <user_guide/index>`
   designed for CFOs, risk managers, and actuaries. No advanced mathematics required!

Quick Links
-----------

.. raw:: html

   <div style="display: flex; gap: 20px; margin: 20px 0;">
     <div style="flex: 1; padding: 15px; border: 2px solid #E91E63; border-radius: 8px; background-color: #FCE4EC;">
       <h3 style="margin-top: 0; color: #b8154cff !important;">🚀 Quick Start</h3>
       <p>Run your first simulation</p>
       <a href="tutorials/01_getting_started.html" style="color: #E91E63 !important; font-weight: bold; text-decoration: none;">Get Started →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #4CAF50; border-radius: 8px; background-color: #E8F5E9;">
       <h3 style="margin-top: 0; color: #3b933eff !important;">📚 Tutorials</h3>
       <p>Step-by-step learning guides</p>
       <a href="tutorials/index.html" style="color: #4CAF50 !important; font-weight: bold; text-decoration: none;">Start Tutorial →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #FF9800; border-radius: 8px; background-color: #FFF3E0;">
       <h3 style="margin-top: 0; color: #d37f00ff !important;">💡 Model Cases</h3>
       <p>Realistic examples</p>
       <a href="user_guide/case_studies.html" style="color: #FF9800 !important; font-weight: bold; text-decoration: none;">View Cases →</a>
     </div>
   </div>

   <div style="display: flex; gap: 20px; margin: 20px 0;">
     <div style="flex: 1; padding: 15px; border: 2px solid #2196F3; border-radius: 8px; background-color: #E3F2FD;">
       <h3 style="margin-top: 0; color: #1e82d3ff !important;">📖 Theory</h3>
       <p>Mathematical foundations</p>
       <a href="theory/index.html" style="color: #2196F3 !important; font-weight: bold; text-decoration: none;">Learn Theory →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #9C27B0; border-radius: 8px; background-color: #F3E5F5;">
       <h3 style="margin-top: 0; color: #7e208fff !important;">🏗️ Architecture</h3>
       <p>System design diagrams</p>
       <a href="architecture/index.html" style="color: #9C27B0 !important; font-weight: bold; text-decoration: none;">View Architecture →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #795548; border-radius: 8px; background-color: #EFEBE9;">
       <h3 style="margin-top: 0; color: #5b4036ff !important;">📋 API Reference</h3>
       <p>Complete API documentation</p>
       <a href="api/modules.html" style="color: #795548 !important; font-weight: bold; text-decoration: none;">Browse API →</a>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Business User Guide:

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & How-To Guides:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation:

   overview
   getting_started
   examples
   theory/index
   architecture/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules

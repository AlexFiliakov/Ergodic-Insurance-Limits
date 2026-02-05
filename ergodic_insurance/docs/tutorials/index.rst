Tutorials
=========

This tutorial series walks you through the Ergodic Insurance Limits framework, from installation to advanced optimization. The goal is to help you understand how time-average (ergodic) analysis changes insurance purchasing decisions compared to traditional expected-value approaches.

.. note::
   **New here?** Start with :doc:`01_getting_started` for installation and a first simulation.

Tutorial Overview
-----------------

The tutorials are split into two groups.

**Foundations (Tutorials 1--2):**

1. **Getting Started** -- Installation, environment setup, and running your first simulation
2. **Basic Simulation** -- The Widget Manufacturer model, step-by-step year simulation, and loss processing

**Applied Workflow (Tutorials 3--6):**

Tutorials 3 and 4 follow **NovaTech Plastics**, a fictional $10M plastics manufacturer with an 8% operating margin, as the running example. The same manufacturer parameters carry through Tutorials 5 and 6.

3. **Configuring Insurance** -- Building single-layer and multi-layer insurance towers for NovaTech
4. **Optimization Workflow** -- Using ``BusinessOptimizer`` to find data-driven insurance strategies as NovaTech plans its expansion
5. **Analyzing Results** -- Comparing time-average vs. ensemble-average growth rates with ``ErgodicAnalyzer``
6. **Advanced Scenarios** -- Monte Carlo simulations, market cycle modeling, and scenario analysis

Quick Start Paths
-----------------

You do not need to complete every tutorial. Pick a path based on what you want to learn:

**Actuaries and Risk Managers:**
   Start with the foundations, then focus on insurance structure and results interpretation.

   1. :doc:`01_getting_started`
   2. :doc:`03_configuring_insurance`
   3. :doc:`05_analyzing_results`

**Financial Analysts and CFOs:**
   Focus on the business case for insurance and the optimization workflow.

   1. :doc:`01_getting_started`
   2. :doc:`04_optimization_workflow`
   3. :doc:`05_analyzing_results`

**Developers and Researchers:**
   Dive into the simulation engine and advanced techniques.

   1. :doc:`02_basic_simulation`
   2. :doc:`04_optimization_workflow`
   3. :doc:`06_advanced_scenarios`

Tutorials
---------

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: Step-by-Step Tutorials:

   01_getting_started
   02_basic_simulation
   03_configuring_insurance
   04_optimization_workflow
   05_analyzing_results
   06_advanced_scenarios

Support Resources
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Help & Support:

   troubleshooting

Learning Objectives
-------------------

After working through these tutorials you should be able to:

* Install the framework and run a simulation end-to-end
* Configure multi-layer insurance programs with deductibles, attachment points, and limits
* Use the optimizer to search for insurance strategies that maximize time-average growth
* Interpret the difference between ensemble-average and time-average growth rates
* Run Monte Carlo simulations and analyze survival probabilities, ROE, and final equity distributions

Prerequisites
-------------

**Required:**
   * Python 3.12 or higher
   * Comfort with Python basics: importing packages, running scripts, reading tracebacks
   * Familiarity with probability concepts (distributions, expected value, variance) at the level of Actuarial Exam P

**Helpful:**
   * Working knowledge of insurance terms -- deductible, retention, limit, attachment point, premium
   * Exposure to financial metrics such as ROE, operating margin, and asset turnover
   * Experience with NumPy arrays and Matplotlib (used throughout the code examples)

Getting Help
------------

If you get stuck:

1. Check the :doc:`troubleshooting` guide for common errors and fixes
2. Consult the :doc:`../api/modules` for detailed function and class documentation
3. Open an issue on `GitHub <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues>`__

Insurance Optimization
======================

This guide covers the insurance optimization algorithms and strategies implemented in the project.

Overview
--------

The insurance optimization framework determines:

1. **Optimal coverage limits** - How much insurance to buy
2. **Optimal retention levels** - Deductibles and self-insurance
3. **Layer structuring** - Primary, excess, and umbrella coverage
4. **Premium efficiency** - Cost-benefit analysis

Optimization Approaches
-----------------------

Single-Layer Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

For basic insurance programs with a single layer:

.. code-block:: python

   from ergodic_insurance.insurance import optimize_insurance_limit

   optimal_limit = optimize_insurance_limit(
       manufacturer=manufacturer,
       claim_generator=generator,
       limits_to_test=np.linspace(1e6, 50e6, 20),
       base_premium_rate=np.linspace(0.01, 0.03, 20),
       optimization_metric="time_average_growth",
       n_simulations=500
   )

   print(f"Optimal limit: ${optimal_limit['limit']:,.0f}")
   print(f"Optimal premium rate: {optimal_limit['rate']:.2%}")

Multi-Layer Programs
~~~~~~~~~~~~~~~~~~~~

For sophisticated insurance programs with multiple layers:

.. code-block:: python

   from ergodic_insurance import InsuranceProgram
   from ergodic_insurance.optimization import optimize_program_structure

   # Define layer structure
   layers = [
       {"name": "Primary", "attachment": 0, "limit": 5e6},
       {"name": "First Excess", "attachment": 5e6, "limit": 15e6},
       {"name": "Second Excess", "attachment": 20e6, "limit": 30e6}
   ]

   # Optimize the program
   optimal_program = optimize_program_structure(
       layers=layers,
       manufacturer=manufacturer,
       claim_generator=generator,
       constraints={
           "max_total_premium": 0.02 * manufacturer.assets,
           "min_retention": 1e6,
           "max_layers": 5
       }
   )

Optimization Metrics
--------------------

Time-Average Growth
~~~~~~~~~~~~~~~~~~~

The primary metric for ergodic optimization:

.. math::

   g = \lim_{T \to \infty} \frac{1}{T} \ln\left(\frac{W(T)}{W(0)}\right)

Maximizes long-term wealth accumulation.

Risk-Adjusted Return
~~~~~~~~~~~~~~~~~~~~

Sharpe ratio variant for insurance decisions:

.. math::

   S = \frac{g - r_f}{\sigma_g}

Where :math:`r_f` is risk-free rate and :math:`\sigma_g` is growth volatility.

Bankruptcy Probability
~~~~~~~~~~~~~~~~~~~~~~

Constraint-based optimization:

.. math::

   \text{maximize } g \text{ subject to } P(\text{bankruptcy}) < \alpha

Typically :math:`\alpha = 0.01` (1% ruin probability).

Advanced Techniques
-------------------

Pareto Frontier Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Multi-objective optimization balancing growth and risk:

.. code-block:: python

   from ergodic_insurance.pareto_frontier import ParetoFrontier

   frontier = ParetoFrontier()

   # Add objectives
   frontier.add_objective("growth", maximize=True)
   frontier.add_objective("bankruptcy_prob", maximize=False)
   frontier.add_objective("premium_cost", maximize=False)

   # Find Pareto-optimal solutions
   optimal_set = frontier.optimize(
       decision_variables=["limit", "deductible", "base_premium_rate"],
       n_iterations=1000
   )

   # Visualize trade-offs
   frontier.plot_3d()

Dynamic Programming
~~~~~~~~~~~~~~~~~~~

For time-varying insurance decisions:

.. code-block:: python

   from ergodic_insurance.optimal_control import DynamicInsuranceOptimizer

   optimizer = DynamicInsuranceOptimizer()

   # Define state-dependent policy
   policy = optimizer.solve_hjb(
       states=["assets", "claims_history", "market_condition"],
       controls=["insurance_limit", "retention"],
       time_horizon=50,
       discount_rate=0.05
   )

   # Get optimal action for current state
   current_state = {"assets": 10e6, "claims_history": [100e3, 500e3]}
   optimal_action = policy.get_action(current_state)

Stochastic Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Incorporating uncertainty in optimization:

.. code-block:: python

   from ergodic_insurance.optimization import StochasticOptimizer

   optimizer = StochasticOptimizer()

   # Define uncertain parameters
   uncertain_params = {
       "claim_frequency": ("poisson", 3),
       "claim_severity": ("lognormal", 10, 2),
       "base_premium_rates": ("uniform", 0.01, 0.03)
   }

   # Robust optimization
   robust_solution = optimizer.optimize_robust(
       objective="expected_utility",
       uncertain_params=uncertain_params,
       confidence_level=0.95
   )

Real-World Constraints
----------------------

Regulatory Requirements
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   constraints = {
       "min_coverage": 5e6,  # Regulatory minimum
       "max_deductible": 0.1 * manufacturer.assets,  # 10% of assets
       "solvency_ratio": 1.5  # Required capital ratio
   }

Market Conditions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Hard market conditions
   hard_market = {
       "premium_multiplier": 1.5,
       "capacity_reduction": 0.7,
       "higher_retentions": True
   }

   # Soft market conditions
   soft_market = {
       "premium_multiplier": 0.8,
       "capacity_increase": 1.3,
       "lower_retentions": True
   }

Business Constraints
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   business_constraints = {
       "max_premium_as_pct_revenue": 0.02,
       "min_liquidity_after_deductible": 1e6,
       "max_collateral_requirements": 5e6
   }

Model Case: Widget Manufacturer
--------------------------------

Optimization Process
~~~~~~~~~~~~~~~~~~~~

1. **Baseline Analysis**

   .. code-block:: python

      # No insurance baseline
      baseline = simulate_without_insurance(manufacturer, n_years=100)
      print(f"Bankruptcy rate: {baseline['bankruptcy_rate']:.1%}")
      print(f"Time-avg growth: {baseline['time_avg_growth']:.2%}")

2. **Single Layer Optimization**

   .. code-block:: python

      # Find optimal single layer
      single_layer = optimize_single_layer(
          manufacturer,
          limits=np.logspace(6, 8, 50)  # $1M to $100M
      )

3. **Multi-Layer Refinement**

   .. code-block:: python

      # Build optimal program
      program = build_optimal_program(
          manufacturer,
          n_layers=3,
          total_limit=single_layer['limit']
      )

4. **Sensitivity Analysis**

   .. code-block:: python

      # Test robustness
      sensitivity = analyze_sensitivity(
          program,
          vary_params=["claim_frequency", "severity", "correlation"],
          n_scenarios=1000
      )

Results
~~~~~~~

* **Optimal limit**: \$15M (1.5x annual revenue)
* **Optimal retention**: \$1M (10% of assets)
* **Premium rate**: 1.8% of limit
* **Time-average growth improvement**: +3.2% annually
* **Bankruptcy reduction**: 15% → 0.8%

Implementation Guide
--------------------

Step 1: Define Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   objectives = {
       "primary": "maximize_time_avg_growth",
       "constraints": [
           "bankruptcy_prob < 0.01",
           "premium_cost < 0.02 * revenue"
       ]
   }

Step 2: Set Up Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.decision_engine import InsuranceDecisionEngine

   engine = InsuranceDecisionEngine(
       manufacturer=manufacturer,
       objectives=objectives
   )

Step 3: Run Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   optimal_decision = engine.optimize(
       method="differential_evolution",
       n_iterations=1000,
       parallel=True
   )

Step 4: Validate Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   validation = engine.validate_decision(
       optimal_decision,
       n_simulations=10000,
       confidence_level=0.95
   )

   print(f"Expected improvement: {validation['expected_improvement']:.2%}")
   print(f"Confidence interval: {validation['ci_lower']:.2%} - {validation['ci_upper']:.2%}")

Best Practices
--------------

1. **Start simple**: Begin with single-layer optimization
2. **Use appropriate metrics**: Time-average for long-term, VaR for short-term
3. **Consider correlation**: Model dependency between operational and financial risks
4. **Validate robustness**: Test across different economic scenarios
5. **Monitor and adjust**: Re-optimize as conditions change

See Also
--------

* :doc:`api/insurance` - Insurance module API
* :doc:`api/optimization` - Optimization algorithms
* :doc:`user_guide/decision_framework` - Decision-making guide
* :doc:`examples` - Practical examples

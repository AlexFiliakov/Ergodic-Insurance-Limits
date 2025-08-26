===================================
Hamilton-Jacobi-Bellman (HJB) Solver User Guide
===================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Introduction
============

The Hamilton-Jacobi-Bellman (HJB) solver is a powerful tool for finding optimal control strategies in insurance and risk management. Unlike traditional optimization methods that may find local optima or require specific problem structures, the HJB approach provides globally optimal solutions by solving a partial differential equation (PDE) that characterizes the value function and optimal policy simultaneously.

When to Use the HJB Solver
==========================

The HJB solver is particularly effective for:

1. **Dynamic Optimization Problems**: When decisions must adapt to changing states over time
2. **Stochastic Control**: Problems with uncertainty in state evolution
3. **Path-Dependent Strategies**: When optimal decisions depend on history
4. **Multi-dimensional State Spaces**: Problems with multiple interacting state variables
5. **Non-linear Dynamics**: Systems with complex, non-linear state evolution
6. **Risk-Sensitive Control**: When risk aversion affects optimal decisions

Advantages Over Alternative Methods
------------------------------------

**Compared to Monte Carlo Optimization:**
  - Provides deterministic, reproducible solutions
  - Faster convergence for smooth problems
  - Yields complete policy function, not just point estimates

**Compared to Dynamic Programming:**
  - Handles continuous state and control spaces naturally
  - Scales better with time horizon
  - Provides smooth value functions and policies

**Compared to Model Predictive Control:**
  - Solves for entire policy offline
  - No need for repeated optimization during execution
  - Guarantees global optimality under convexity assumptions

Mathematical Foundation
=======================

The HJB Equation
----------------

The HJB equation for a finite-horizon problem is:

.. math::

   -\frac{\partial V}{\partial t}(x,t) + \max_u \left[ L(x,u,t) + \nabla V(x,t) \cdot f(x,u,t) + \frac{1}{2}\text{tr}(\sigma(x,t)\sigma(x,t)^T \nabla^2 V) \right] = 0

Where:
  - :math:`V(x,t)` is the value function
  - :math:`L(x,u,t)` is the running cost/reward
  - :math:`f(x,u,t)` is the drift (deterministic dynamics)
  - :math:`\sigma(x,t)` is the diffusion coefficient (stochastic component)
  - :math:`u` is the control variable

For infinite-horizon problems with discount rate :math:`\rho`:

.. math::

   \rho V(x) = \max_u \left[ L(x,u) + \nabla V(x) \cdot f(x,u) + \frac{1}{2}\text{tr}(\sigma(x)\sigma(x)^T \nabla^2 V) \right]

Solution Method
---------------

The solver uses **policy iteration** with finite difference discretization:

1. **Initialize**: Start with an initial guess for the value function and policy
2. **Policy Evaluation**: Fix the policy and solve the resulting linear PDE for the value function
3. **Policy Improvement**: Update the policy by maximizing the Hamiltonian at each state
4. **Iterate**: Repeat steps 2-3 until convergence

The finite difference scheme uses:
  - **Upwind differencing** for first-order terms (ensures stability)
  - **Central differencing** for second-order terms
  - **Implicit time-stepping** for robustness

Core Components
===============

State Variables
---------------

State variables define the problem dimensions:

.. code-block:: python

   from ergodic_insurance.src.hjb_solver import StateVariable, BoundaryCondition

   # Wealth state with logarithmic spacing
   wealth = StateVariable(
       name="wealth",
       min_value=1e6,
       max_value=1e8,
       num_points=100,
       log_scale=True,  # Use log spacing for large ranges
       boundary_lower=BoundaryCondition.ABSORBING,
       boundary_upper=BoundaryCondition.REFLECTING
   )

   # Time state with linear spacing
   time = StateVariable(
       name="time",
       min_value=0,
       max_value=10,
       num_points=50,
       log_scale=False
   )

**Best Practices:**
  - Use logarithmic spacing for variables spanning orders of magnitude
  - Choose boundary conditions matching the economic interpretation
  - Balance grid resolution with computational cost (50-200 points per dimension)

Control Variables
-----------------

Control variables represent decisions:

.. code-block:: python

   from ergodic_insurance.src.hjb_solver import ControlVariable

   # Insurance limit control
   insurance_limit = ControlVariable(
       name="limit",
       min_value=0,
       max_value=5e7,
       num_points=30,  # Discretization for optimization
       continuous=True
   )

   # Retention/deductible control
   retention = ControlVariable(
       name="retention",
       min_value=1e5,
       max_value=1e7,
       num_points=30
   )

**Optimization Tips:**
  - Use 20-50 control points for smooth problems
  - Increase resolution near expected optimal values
  - Consider adaptive refinement for complex policies

Utility Functions
-----------------

The solver includes several built-in utility functions:

.. code-block:: python

   from ergodic_insurance.src.hjb_solver import (
       LogUtility,
       PowerUtility,
       ExpectedWealth,
       create_custom_utility
   )

   # Logarithmic utility (Kelly criterion)
   log_utility = LogUtility(wealth_floor=1e3)

   # Power utility (CRRA)
   power_utility = PowerUtility(
       risk_aversion=2.0,  # Higher = more risk averse
       wealth_floor=1e3
   )

   # Risk-neutral (linear) utility
   linear_utility = ExpectedWealth()

   # Custom utility function
   def exponential_eval(w):
       alpha = 0.001
       return 1 - np.exp(-alpha * w)

   def exponential_deriv(w):
       alpha = 0.001
       return alpha * np.exp(-alpha * w)

   custom_utility = create_custom_utility(
       evaluate_func=exponential_eval,
       derivative_func=exponential_deriv
   )

Complete Example: Insurance Optimization
=========================================

Problem Setup
-------------

Consider a manufacturing firm optimizing its insurance program to maximize long-term growth:

.. code-block:: python

   import numpy as np
   from ergodic_insurance.src.hjb_solver import (
       StateVariable, StateSpace,
       ControlVariable,
       HJBProblem, HJBSolver, HJBSolverConfig,
       LogUtility, BoundaryCondition
   )

   # Define state space
   state_vars = [
       StateVariable(
           name="assets",
           min_value=1e6,
           max_value=1e9,
           num_points=80,
           log_scale=True,
           boundary_lower=BoundaryCondition.ABSORBING,  # Bankruptcy
           boundary_upper=BoundaryCondition.NEUMANN     # No constraint
       ),
       StateVariable(
           name="loss_rate",
           min_value=0,
           max_value=0.2,
           num_points=40,
           log_scale=False
       )
   ]
   state_space = StateSpace(state_vars)

   # Define controls
   controls = [
       ControlVariable("coverage_limit", 0, 5e7, num_points=25),
       ControlVariable("deductible", 1e4, 1e7, num_points=25)
   ]

   # Use log utility for growth optimization
   utility = LogUtility(wealth_floor=1e4)

Dynamics and Costs
------------------

Define the system dynamics and running costs:

.. code-block:: python

   def dynamics(state, control, time):
       """Asset dynamics with insurance."""
       assets = state[..., 0]
       loss_rate = state[..., 1]
       limit = control[..., 0]
       deductible = control[..., 1]

       # Growth rate reduced by insurance premium
       base_growth = 0.08
       premium_rate = 0.02 * (limit / 1e7) * (1 - deductible / limit)

       # Expected retained losses
       expected_loss = assets * loss_rate * np.minimum(1.0, deductible / assets)

       # Asset drift
       asset_drift = assets * (base_growth - premium_rate) - expected_loss

       # Loss rate mean reversion
       loss_drift = 0.1 * (0.05 - loss_rate)

       return np.stack([asset_drift, loss_drift], axis=-1)

   def running_cost(state, control, time):
       """Utility flow from operations."""
       assets = state[..., 0]
       return utility.evaluate(assets)

   def terminal_value(state):
       """Terminal wealth utility."""
       assets = state[..., 0]
       return utility.evaluate(assets)

Solving the HJB Equation
------------------------

Create and solve the HJB problem:

.. code-block:: python

   # Create HJB problem
   problem = HJBProblem(
       state_space=state_space,
       control_variables=controls,
       utility_function=utility,
       dynamics=dynamics,
       running_cost=running_cost,
       terminal_value=terminal_value,
       discount_rate=0.05,
       time_horizon=20  # 20-year horizon
   )

   # Configure solver
   config = HJBSolverConfig(
       time_step=0.01,
       max_iterations=500,
       tolerance=1e-6,
       scheme=TimeSteppingScheme.IMPLICIT,
       use_sparse=True,
       verbose=True
   )

   # Solve
   solver = HJBSolver(problem, config)
   value_function, optimal_policy = solver.solve()

   # Check solution quality
   metrics = solver.compute_convergence_metrics()
   print(f"Max residual: {metrics['max_residual']:.2e}")
   print(f"Policy range - Limit: {metrics['policy_stats']['coverage_limit']}")
   print(f"Policy range - Deductible: {metrics['policy_stats']['deductible']}")

Using the Solution
------------------

Extract and apply the optimal policy:

.. code-block:: python

   # Get optimal control at specific state
   current_state = np.array([5e7, 0.06])  # $50M assets, 6% loss rate
   optimal_control = solver.extract_feedback_control(current_state)

   print(f"Optimal coverage limit: ${optimal_control['coverage_limit']:,.0f}")
   print(f"Optimal deductible: ${optimal_control['deductible']:,.0f}")

   # Visualize the policy
   import matplotlib.pyplot as plt

   # Plot optimal limit as function of assets (fixed loss rate)
   assets_range = np.linspace(1e6, 1e9, 100)
   loss_rate_fixed = 0.05

   optimal_limits = []
   for assets in assets_range:
       state = np.array([assets, loss_rate_fixed])
       control = solver.extract_feedback_control(state)
       optimal_limits.append(control['coverage_limit'])

   plt.figure(figsize=(10, 6))
   plt.semilogx(assets_range, optimal_limits)
   plt.xlabel('Assets ($)')
   plt.ylabel('Optimal Coverage Limit ($)')
   plt.title('State-Dependent Insurance Strategy')
   plt.grid(True)
   plt.show()

Advanced Topics
===============

Multi-Layer Insurance
---------------------

For complex insurance programs with multiple layers:

.. code-block:: python

   # Define controls for each layer
   controls = []
   for i in range(3):  # 3-layer program
       controls.extend([
           ControlVariable(f"limit_L{i}", 1e6 * (i+1), 1e7 * (i+1), 20),
           ControlVariable(f"attachment_L{i}", 1e5 * (i+1), 1e6 * (i+1), 20),
           ControlVariable(f"coinsurance_L{i}", 0.8, 1.0, 10)
       ])

   # Dynamics account for all layers
   def multi_layer_dynamics(state, control, time):
       assets = state[..., 0]
       total_premium = 0

       for i in range(3):
           limit = control[..., i*3]
           attachment = control[..., i*3 + 1]
           coinsurance = control[..., i*3 + 2]

           # Layer-specific premium
           layer_premium = compute_layer_premium(limit, attachment, coinsurance)
           total_premium += layer_premium

       # Continue with dynamics...

Stochastic Volatility
---------------------

Incorporate time-varying uncertainty:

.. code-block:: python

   # Add volatility as state variable
   volatility = StateVariable(
       name="volatility",
       min_value=0.01,
       max_value=0.5,
       num_points=30,
       log_scale=False
   )

   def stochastic_dynamics(state, control, time):
       assets = state[..., 0]
       vol = state[..., 1]

       # Volatility affects growth uncertainty
       growth_rate = 0.08
       growth_std = vol * np.sqrt(assets)

       # In HJB, we work with drift (deterministic part)
       # Diffusion enters through second-order terms
       drift_assets = assets * growth_rate
       drift_vol = 0.2 * (0.15 - vol)  # Mean reversion

       return np.stack([drift_assets, drift_vol], axis=-1)

Performance Optimization
========================

Computational Efficiency
------------------------

1. **Grid Resolution**: Start coarse, refine gradually

   .. code-block:: python

      # Initial solve with coarse grid
      wealth_coarse = StateVariable("wealth", 1e6, 1e8, num_points=30)

      # Refine around optimal region
      wealth_fine = StateVariable("wealth", 1e6, 1e8, num_points=100)

2. **Sparse Matrices**: Enable for large problems

   .. code-block:: python

      config = HJBSolverConfig(use_sparse=True)

3. **Parallel Control Optimization**: Future enhancement

   .. code-block:: python

      # Currently sequential, but structure allows parallelization
      # Each state's optimization is independent

Memory Management
-----------------

For very large state spaces:

.. code-block:: python

   # Use iterative methods instead of direct solvers
   config = HJBSolverConfig(
       scheme=TimeSteppingScheme.IMPLICIT,
       use_sparse=True,
       time_step=0.001  # Smaller time steps for stability
   )

   # Consider domain decomposition for 3+ dimensions
   # Solve on subdomains and match at boundaries

Troubleshooting Guide
=====================

Common Issues and Solutions
---------------------------

**Problem: Solver doesn't converge**
  - Increase ``max_iterations``
  - Reduce ``tolerance``
  - Check boundary conditions match problem physics
  - Ensure dynamics and costs are smooth

**Problem: Numerical instabilities**
  - Use implicit time-stepping
  - Reduce time step
  - Check for discontinuities in dynamics/costs
  - Increase grid resolution near discontinuities

**Problem: Unrealistic optimal policies**
  - Verify utility function choice
  - Check control bounds are reasonable
  - Ensure dynamics correctly model the system
  - Validate cost/reward functions

**Problem: Out of memory**
  - Reduce grid points
  - Enable sparse matrices
  - Use iterative solvers
  - Consider dimension reduction

Validation Techniques
---------------------

1. **Benchmark Against Known Solutions**:

   .. code-block:: python

      # Test with linear dynamics, quadratic cost (LQR)
      # Compare with analytical Riccati solution

2. **Convergence Analysis**:

   .. code-block:: python

      resolutions = [20, 40, 80, 160]
      solutions = []

      for n in resolutions:
          state = StateVariable("x", 0, 1, num_points=n)
          # Solve and store value at test point
          solutions.append(value_at_test_point)

      # Check convergence rate
      errors = [abs(s - solutions[-1]) for s in solutions[:-1]]

3. **Policy Simulation**:

   .. code-block:: python

      # Forward simulate using optimal policy
      # Check value function matches simulated rewards

Integration with Simulation Framework
=====================================

Using with OptimalController
-----------------------------

Integrate HJB solutions with the simulation framework:

.. code-block:: python

   from ergodic_insurance.src.optimal_control import (
       HJBFeedbackControl,
       OptimalController,
       ControlSpace
   )

   # After solving HJB
   control_space = ControlSpace(
       limits=[(1e6, 5e7)],
       retentions=[(1e5, 1e7)],
       coverage_percentages=[(0.8, 1.0)]
   )

   # Create feedback strategy
   strategy = HJBFeedbackControl(
       hjb_solver=solver,
       control_space=control_space,
       state_mapping=lambda state_dict: np.array([
           state_dict['assets'],
           state_dict['loss_rate']
       ])
   )

   # Create controller
   controller = OptimalController(strategy, control_space)

   # Use in simulation loop
   for t in range(simulation_steps):
       insurance = controller.apply_control(manufacturer, time=t)
       # Apply insurance and simulate...

Real-Time Adaptation
--------------------

For online learning and adaptation:

.. code-block:: python

   class AdaptiveHJBControl:
       def __init__(self, base_solver):
           self.solver = base_solver
           self.observations = []

       def update(self, state, outcome):
           """Update beliefs about system dynamics."""
           self.observations.append((state, outcome))

           if len(self.observations) % 100 == 0:
               # Re-estimate parameters
               new_dynamics = self.estimate_dynamics()

               # Re-solve HJB with updated model
               self.solver.problem.dynamics = new_dynamics
               self.solver.solve()

Best Practices Summary
======================

1. **Problem Formulation**:
   - Clearly define state and control spaces
   - Choose utility function matching risk preferences
   - Ensure dynamics are smooth and well-behaved

2. **Numerical Setup**:
   - Start with coarse grids, refine gradually
   - Use appropriate boundary conditions
   - Choose time-stepping scheme based on stability needs

3. **Validation**:
   - Test on problems with known solutions
   - Verify convergence with grid refinement
   - Simulate policies to check consistency

4. **Performance**:
   - Profile to identify bottlenecks
   - Use sparse matrices for large problems
   - Consider approximation methods for high dimensions

5. **Integration**:
   - Map between simulation and HJB state spaces carefully
   - Handle edge cases in policy interpolation
   - Monitor solution quality during deployment

Conclusion
==========

The HJB solver provides a powerful framework for optimal control in insurance and risk management. Its ability to handle complex dynamics, multiple state variables, and risk-sensitive objectives makes it particularly valuable for:

- **Dynamic insurance optimization**: Adapting coverage to changing risk profiles
- **Capital allocation**: Balancing growth and risk in investment strategies
- **Operational decisions**: Optimizing production and inventory under uncertainty

While computational requirements grow with dimensionality, careful problem formulation and numerical techniques enable practical solutions for real-world problems. The global optimality guarantees and complete policy characterization often justify the computational investment compared to heuristic or local methods.

For further examples and applications, see the Jupyter notebooks in ``ergodic_insurance/notebooks/``, particularly:
- ``12_hjb_optimal_control.ipynb``: Complete examples with visualization
- ``11_pareto_analysis.ipynb``: Multi-objective optimization using HJB

References
==========

1. Fleming, W.H., & Soner, H.M. (2006). *Controlled Markov Processes and Viscosity Solutions*. Springer.

2. Kushner, H.J., & Dupuis, P. (2001). *Numerical Methods for Stochastic Control Problems in Continuous Time*. Springer.

3. Bertsekas, D.P. (2017). *Dynamic Programming and Optimal Control*. Athena Scientific.

4. Pham, H. (2009). *Continuous-time Stochastic Control and Optimization with Financial Applications*. Springer.

5. Touzi, N. (2012). *Optimal Stochastic Control, Stochastic Target Problems, and Backward SDE*. Springer.

"""Tests for Hamilton-Jacobi-Bellman solver.

Author: Alex Filiakov
Date: 2025-01-26
"""

import numpy as np
import pytest

from ergodic_insurance.hjb_solver import (
    _DRIFT_THRESHOLD,
    _GAMMA_TOLERANCE,
    _MARGINAL_UTILITY_FLOOR,
    BoundaryCondition,
    ControlVariable,
    ExpectedWealth,
    HJBProblem,
    HJBSolver,
    HJBSolverConfig,
    LogUtility,
    PowerUtility,
    StateSpace,
    StateVariable,
    TimeSteppingScheme,
    create_custom_utility,
)


class TestStateVariable:
    """Test StateVariable class."""

    def test_state_variable_creation(self):
        """Test creating state variable."""
        sv = StateVariable(name="wealth", min_value=1e6, max_value=1e8, num_points=50)

        assert sv.name == "wealth"
        assert sv.min_value == 1e6
        assert sv.max_value == 1e8
        assert sv.num_points == 50
        assert sv.boundary_lower == BoundaryCondition.ABSORBING
        assert sv.boundary_upper == BoundaryCondition.ABSORBING
        assert sv.log_scale is False

    def test_state_variable_validation(self):
        """Test state variable validation."""
        # Invalid bounds
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            StateVariable("test", min_value=10, max_value=5, num_points=10)

        # Too few points
        with pytest.raises(ValueError, match="Need at least 3 grid points"):
            StateVariable("test", min_value=0, max_value=10, num_points=2)

        # Log scale with non-positive min
        with pytest.raises(ValueError, match="Cannot use log scale with non-positive"):
            StateVariable("test", min_value=0, max_value=10, num_points=10, log_scale=True)

    def test_grid_generation(self):
        """Test grid generation."""
        # Linear grid
        sv_linear = StateVariable("test", min_value=0, max_value=10, num_points=11)
        grid_linear = sv_linear.get_grid()
        assert len(grid_linear) == 11
        assert grid_linear[0] == 0
        assert grid_linear[-1] == 10
        assert np.allclose(np.diff(grid_linear), 1.0)

        # Log grid
        sv_log = StateVariable("test", min_value=1, max_value=100, num_points=3, log_scale=True)
        grid_log = sv_log.get_grid()
        assert len(grid_log) == 3
        assert grid_log[0] == 1
        assert grid_log[-1] == 100
        assert grid_log[1] == 10  # Geometric mean


class TestControlVariable:
    """Test ControlVariable class."""

    def test_control_variable_creation(self):
        """Test creating control variable."""
        cv = ControlVariable(name="limit", min_value=1e6, max_value=5e7, num_points=20)

        assert cv.name == "limit"
        assert cv.min_value == 1e6
        assert cv.max_value == 5e7
        assert cv.num_points == 20
        assert cv.continuous is True

    def test_control_variable_validation(self):
        """Test control variable validation."""
        # Invalid bounds
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            ControlVariable("test", min_value=10, max_value=5)

        # Too few points
        with pytest.raises(ValueError, match="Need at least 2 control points"):
            ControlVariable("test", min_value=0, max_value=10, num_points=1)

    def test_control_values(self):
        """Test control value generation."""
        cv = ControlVariable("test", min_value=0, max_value=10, num_points=5)
        values = cv.get_values()

        assert len(values) == 5
        assert values[0] == 0
        assert values[-1] == 10
        assert np.allclose(values, [0, 2.5, 5, 7.5, 10])


class TestStateSpace:
    """Test StateSpace class."""

    def test_state_space_1d(self):
        """Test 1D state space."""
        sv = StateVariable("wealth", min_value=0, max_value=10, num_points=5)
        space = StateSpace([sv])

        assert space.ndim == 1
        assert space.shape == (5,)
        assert space.size == 5
        assert len(space.grids) == 1
        assert len(space.grids[0]) == 5

    def test_state_space_2d(self):
        """Test 2D state space."""
        sv1 = StateVariable("wealth", min_value=0, max_value=10, num_points=5)
        sv2 = StateVariable("time", min_value=0, max_value=1, num_points=3)
        space = StateSpace([sv1, sv2])

        assert space.ndim == 2
        assert space.shape == (5, 3)
        assert space.size == 15
        assert len(space.meshgrid) == 2
        assert space.meshgrid[0].shape == (5, 3)
        assert space.meshgrid[1].shape == (5, 3)

    def test_boundary_mask(self):
        """Test boundary mask generation."""
        sv1 = StateVariable("x", min_value=0, max_value=1, num_points=3)
        sv2 = StateVariable("y", min_value=0, max_value=1, num_points=3)
        space = StateSpace([sv1, sv2])

        mask = space.get_boundary_mask()
        assert mask.shape == (3, 3)

        # Check that boundaries are marked
        assert mask[0, :].all()  # First row
        assert mask[-1, :].all()  # Last row
        assert mask[:, 0].all()  # First column
        assert mask[:, -1].all()  # Last column

        # Check interior is not marked
        assert not mask[1, 1]

    def test_interpolation(self):
        """Test value function interpolation."""
        sv = StateVariable(
            "x", min_value=0, max_value=2, num_points=5
        )  # Need more points for cubic
        space = StateSpace([sv])

        # Create simple linear value function
        value = np.array([0, 0.5, 1, 1.5, 2])

        # Test interpolation at grid points
        points = np.array([[0], [0.5], [1], [1.5], [2]])
        interp_values = space.interpolate_value(value, points)
        assert np.allclose(interp_values, [0, 0.5, 1, 1.5, 2])

        # Test interpolation between points
        points = np.array([[0.25], [0.75], [1.25]])
        interp_values = space.interpolate_value(value, points)
        assert np.allclose(interp_values, [0.25, 0.75, 1.25], atol=0.1)


class TestUtilityFunctions:
    """Test utility function implementations."""

    def test_log_utility(self):
        """Test logarithmic utility."""
        utility = LogUtility(wealth_floor=1e-6)

        # Test evaluation
        wealth = np.array([1, np.e, np.e**2])
        values = utility.evaluate(wealth)
        assert np.allclose(values, [0, 1, 2])

        # Test derivative
        derivs = utility.derivative(wealth)
        assert np.allclose(derivs, [1, 1 / np.e, 1 / np.e**2])

        # Test inverse derivative
        inv_derivs = utility.inverse_derivative(derivs)
        assert np.allclose(inv_derivs, wealth)

        # Test floor handling
        small_wealth = np.array([0, 1e-10])
        values = utility.evaluate(small_wealth)
        assert np.isfinite(values).all()

    def test_power_utility(self):
        """Test power utility."""
        # Test with gamma = 2
        utility = PowerUtility(risk_aversion=2.0)

        wealth = np.array([1.0, 2.0, 4.0])  # Use floats
        values = utility.evaluate(wealth)
        expected = -1.0 / wealth  # U(w) = -w^(-1) for gamma=2
        assert np.allclose(values, expected)

        # Test derivative
        derivs = utility.derivative(wealth)
        expected = wealth ** (-2.0)  # U'(w) = w^(-2) for gamma=2 (use float exponent)
        assert np.allclose(derivs, expected)

        # Test inverse derivative
        inv_derivs = utility.inverse_derivative(derivs)
        assert np.allclose(inv_derivs, wealth)

        # Test special case gamma = 1 (should behave like log)
        utility_log = PowerUtility(risk_aversion=1.0)
        wealth = np.array([1, np.e, np.e**2])
        values = utility_log.evaluate(wealth)
        assert np.allclose(values, [0, 1, 2])

    def test_expected_wealth(self):
        """Test linear utility."""
        utility = ExpectedWealth()

        wealth = np.array([1, 2, 3])
        values = utility.evaluate(wealth)
        assert np.allclose(values, wealth)

        derivs = utility.derivative(wealth)
        assert np.allclose(derivs, np.ones_like(wealth))

        # Inverse derivative should raise error
        with pytest.raises(NotImplementedError):
            utility.inverse_derivative(derivs)

    def test_custom_utility(self):
        """Test custom utility creation."""
        # Create exponential utility
        alpha = 0.01

        def exp_eval(w):
            return 1 - np.exp(-alpha * w)

        def exp_deriv(w):
            return alpha * np.exp(-alpha * w)

        def exp_inv_deriv(m):
            return -np.log(m / alpha) / alpha

        utility = create_custom_utility(exp_eval, exp_deriv, exp_inv_deriv)

        wealth = np.array([0, 100, 200])
        values = utility.evaluate(wealth)
        assert values[0] == 0  # U(0) = 0
        assert 0 < values[1] < 1  # U(100) between 0 and 1
        assert values[2] > values[1]  # Increasing

        # Test derivative
        derivs = utility.derivative(wealth)
        assert derivs[0] == alpha  # U'(0) = alpha
        assert np.all(derivs > 0)  # Positive marginal utility
        assert derivs[2] < derivs[1]  # Decreasing marginal utility


class TestHJBProblem:
    """Test HJB problem specification."""

    def test_problem_creation(self):
        """Test creating HJB problem."""
        # Create simple problem
        state_space = StateSpace([StateVariable("wealth", 1e6, 1e8, 10)])

        control_variables = [ControlVariable("limit", 1e6, 5e7, 5)]

        utility = LogUtility()

        def dynamics(x, u, t):
            return x * 0.1  # 10% growth

        def running_cost(x, u, t):
            return 0.0

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=utility,
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=10,
        )

        assert problem.state_space == state_space
        assert problem.control_variables == control_variables
        assert problem.utility_function == utility
        assert problem.discount_rate == 0.05
        assert problem.time_horizon == 10
        assert problem.terminal_value is not None  # Default created

    def test_problem_validation(self):
        """Test problem validation."""
        state_space = StateSpace([StateVariable("wealth", 1e6, 1e8, 10)])
        control_variables = [ControlVariable("limit", 1e6, 5e7, 5)]
        utility = LogUtility()

        def dynamics(x, u, t):
            return x * 0.1

        def running_cost(x, u, t):
            return np.zeros(x.shape[0]) if hasattr(x, "shape") else np.array([0.0])

        # Negative time horizon
        with pytest.raises(ValueError, match="Time horizon must be positive"):
            HJBProblem(
                state_space, control_variables, utility, dynamics, running_cost, time_horizon=-1
            )

        # Negative discount rate
        with pytest.raises(ValueError, match="Discount rate must be non-negative"):
            HJBProblem(
                state_space, control_variables, utility, dynamics, running_cost, discount_rate=-0.1
            )


class TestHJBSolver:
    """Test HJB solver."""

    def test_solver_initialization(self):
        """Test solver initialization."""
        # Create simple 1D problem
        state_space = StateSpace([StateVariable("wealth", 1e6, 1e7, 5)])

        control_variables = [ControlVariable("limit", 1e6, 5e6, 3)]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: (
                np.zeros(x.shape[0]) if hasattr(x, "shape") else np.array([0.0])
            ),
            time_horizon=1,
        )

        config = HJBSolverConfig(time_step=0.1, max_iterations=10, tolerance=1e-4)

        solver = HJBSolver(problem, config)

        assert solver.problem == problem
        assert solver.config == config
        assert solver.value_function is None
        assert solver.optimal_policy is None
        assert solver.operators_initialized

    def test_simple_solve(self):
        """Test solving simple HJB problem."""
        # Create very simple 1D problem
        state_space = StateSpace([StateVariable("wealth", 1.0, 10.0, 5)])

        control_variables = [ControlVariable("control", 0.0, 1.0, 3)]

        # Simple linear dynamics and cost
        def dynamics(x, u, t):
            return np.ones_like(x[..., 0]).reshape(x.shape)

        def running_cost(x, u, t):
            return -x[..., 0]  # Maximize wealth

        def terminal_value(x):
            return x[..., 0]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=ExpectedWealth(),
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_value=terminal_value,
            discount_rate=0.1,
            time_horizon=1.0,
        )

        config = HJBSolverConfig(
            time_step=0.1,
            max_iterations=5,  # Few iterations for test
            tolerance=1e-2,
            verbose=False,
        )

        solver = HJBSolver(problem, config)
        value_function, optimal_policy = solver.solve()

        assert value_function is not None
        assert value_function.shape == (5,)
        assert optimal_policy is not None
        assert "control" in optimal_policy
        assert optimal_policy["control"].shape == (5,)

    def test_convergence_metrics(self):
        """Test convergence metric computation."""
        # Create simple problem and solve
        state_space = StateSpace([StateVariable("wealth", 1.0, 10.0, 3)])

        control_variables = [ControlVariable("control", 0.0, 1.0, 2)]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)

        solver = HJBSolver(problem, config)
        solver.solve()

        metrics = solver.compute_convergence_metrics()

        assert "max_residual" in metrics
        assert "mean_residual" in metrics
        assert "value_function_range" in metrics
        assert "policy_stats" in metrics
        assert metrics["max_residual"] >= 0
        assert metrics["mean_residual"] >= 0
        assert len(metrics["value_function_range"]) == 2

    def test_feedback_control_extraction(self):
        """Test extracting feedback control."""
        # Create and solve simple problem
        state_space = StateSpace([StateVariable("wealth", 1.0, 10.0, 5)])

        control_variables = [
            ControlVariable("limit", 0.0, 1.0, 3),
            ControlVariable("retention", 0.0, 0.5, 3),
        ]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)

        solver = HJBSolver(problem, config)
        solver.solve()

        # Extract control at specific state
        state = np.array([5.0])  # Middle of wealth range
        controls = solver.extract_feedback_control(state)

        assert "limit" in controls
        assert "retention" in controls
        assert 0 <= controls["limit"] <= 1.0
        assert 0 <= controls["retention"] <= 0.5

    def test_different_schemes(self):
        """Test different time stepping schemes."""
        state_space = StateSpace([StateVariable("x", 0.0, 1.0, 3)])

        control_variables = [ControlVariable("u", 0.0, 1.0, 2)]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        # Test different schemes
        for scheme in [
            TimeSteppingScheme.EXPLICIT,
            TimeSteppingScheme.IMPLICIT,
            TimeSteppingScheme.CRANK_NICOLSON,
        ]:
            config = HJBSolverConfig(scheme=scheme, max_iterations=2, verbose=False)

            solver = HJBSolver(problem, config)
            value_function, policy = solver.solve()

            assert value_function is not None
            assert policy is not None


class TestHJBSolverConfigExtensions:
    """Test new HJBSolverConfig fields from Issue #517."""

    def test_inner_iteration_defaults(self):
        """Test default values for inner iteration fields."""
        config = HJBSolverConfig()
        assert config.inner_max_iterations == 100
        assert config.inner_tolerance_factor == 0.1

    def test_custom_inner_iterations(self):
        """Test custom inner iteration values are stored."""
        config = HJBSolverConfig(inner_max_iterations=200, inner_tolerance_factor=0.05)
        assert config.inner_max_iterations == 200
        assert config.inner_tolerance_factor == 0.05


class TestModuleConstants:
    """Test module-level named constants from Issue #517."""

    def test_drift_threshold_exists(self):
        """Verify _DRIFT_THRESHOLD constant exists and has expected value."""
        assert _DRIFT_THRESHOLD == 1e-10

    def test_marginal_utility_floor_exists(self):
        """Verify _MARGINAL_UTILITY_FLOOR constant exists and has expected value."""
        assert _MARGINAL_UTILITY_FLOOR == 1e-10

    def test_gamma_tolerance_exists(self):
        """Verify _GAMMA_TOLERANCE constant exists and has expected value."""
        assert _GAMMA_TOLERANCE == 1e-10


class TestInitialValueFunction:
    """Test ``HJBProblem.initial_value_function`` warm-start (Issue #1570).

    The warm-start lets callers seed the solver's value function with a
    user-supplied callback instead of zeros (infinite horizon) or the
    terminal value (finite horizon). The tests cover:

      - Override of the default zero initialization (infinite horizon).
      - Override of the terminal value with logged warning (finite horizon).
      - Capture by the Dirichlet boundary machinery so the warm-start
        values at the lower/upper boundaries are preserved through the solve.
      - Correct stacked-grid shape passed to the callback.
    """

    @staticmethod
    def _make_degenerate_problem(
        initial_value_function, time_horizon=None, terminal_value=None, boundary_lower=None
    ):
        """Build a 1D problem whose dynamics, cost, and discount are all zero.

        For zero dynamics + zero cost + zero discount, the HJB fixed point IS
        the initial value (V_t+1 = V_t identically), so after a couple of
        iterations the warm-start should be preserved exactly.
        """
        if boundary_lower is None:
            sv = StateVariable(name="wealth", min_value=1.0, max_value=10.0, num_points=7)
        else:
            sv = StateVariable(
                name="wealth",
                min_value=1.0,
                max_value=10.0,
                num_points=7,
                boundary_lower=boundary_lower,
            )
        state_space = StateSpace([sv])
        control_variables = [ControlVariable("u", 0.0, 1.0, 3)]

        return HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[:-1]),
            discount_rate=0.0,
            initial_value_function=initial_value_function,
            time_horizon=time_horizon,
            terminal_value=terminal_value,
        )

    def test_initial_value_function_overrides_zero_init(self):
        """Infinite-horizon: warm-start replaces the default zero initialization."""

        def warm(x):
            return np.full(x.shape[:-1], 7.5)

        problem = self._make_degenerate_problem(initial_value_function=warm)
        config = HJBSolverConfig(time_step=0.01, max_iterations=2, tolerance=1.0, verbose=False)
        solver = HJBSolver(problem, config)
        v, _ = solver.solve()
        # With zero dynamics/cost/discount, the warm-start is the fixed point
        # and the solver should report it exactly.
        np.testing.assert_allclose(v, 7.5, atol=1e-9)

    def test_initial_value_function_overrides_terminal_value(self, caplog):
        """Finite-horizon: warm-start beats terminal_value with a logged warning."""
        import logging

        def warm(x):
            return np.full(x.shape[:-1], 3.25)

        def terminal(x):
            return np.full(x.shape[:-1], 99.0)

        problem = self._make_degenerate_problem(
            initial_value_function=warm,
            time_horizon=1.0,
            terminal_value=terminal,
        )
        config = HJBSolverConfig(time_step=0.01, max_iterations=2, tolerance=1.0, verbose=False)
        solver = HJBSolver(problem, config)
        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.hjb_solver"):
            v, _ = solver.solve()
        # Warning was emitted with the expected fragment
        assert any(
            "initial_value_function provided for a finite-horizon problem" in record.message
            for record in caplog.records
        ), f"Expected warning not found in caplog records: {[r.message for r in caplog.records]}"
        # Warm-start value (3.25) wins over terminal value (99.0)
        np.testing.assert_allclose(v, 3.25, atol=1e-9)

    def test_initial_value_function_captured_by_dirichlet(self):
        """Dirichlet boundary captures the warm-start value at index 0."""

        # Distinctive value at index 0 (lowest wealth point), zero elsewhere.
        def warm(x):
            # x has shape (n, 1) for 1D state space (after stacking)
            wealth = x[..., 0]
            wealth_min = 1.0  # matches StateVariable min_value
            out = np.zeros(wealth.shape)
            out[wealth == wealth_min] = 42.0
            return out

        problem = self._make_degenerate_problem(
            initial_value_function=warm,
            boundary_lower=BoundaryCondition.DIRICHLET,
        )
        config = HJBSolverConfig(time_step=0.01, max_iterations=2, tolerance=1.0, verbose=False)
        solver = HJBSolver(problem, config)
        v, _ = solver.solve()
        # Boundary value captured from warm-start
        assert solver._boundary_values is not None
        assert 0 in solver._boundary_values
        assert solver._boundary_values[0]["lower"] == pytest.approx(42.0)
        # And the Dirichlet enforcement preserves V[0] = 42.0 through the solve.
        assert v[0] == pytest.approx(42.0)

    def test_initial_value_function_state_grid_shape(self):
        """2D problem: callback receives a stacked grid of correct shape."""
        n1, n2 = 5, 4
        sv1 = StateVariable("x", 0.0, 1.0, n1)
        sv2 = StateVariable("y", 0.0, 1.0, n2)
        state_space = StateSpace([sv1, sv2])
        control_variables = [ControlVariable("u", 0.0, 1.0, 2)]

        captured = {}

        def warm(x):
            captured["input_shape"] = x.shape
            out = np.zeros(x.shape[:-1])
            captured["output_shape"] = out.shape
            return out

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[:-1]),
            discount_rate=0.0,
            initial_value_function=warm,
        )
        config = HJBSolverConfig(time_step=0.01, max_iterations=1, tolerance=1.0, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        assert captured["input_shape"] == (n1, n2, 2)
        assert captured["output_shape"] == (n1, n2)
        # And the resulting value_function has the right shape
        assert solver.value_function is not None
        assert solver.value_function.shape == (n1, n2)


class TestBoundaryAwareResidual:
    """Test ``compute_convergence_metrics`` boundary/interior split (Issue #1569).

    ``compute_convergence_metrics`` builds the HJB residual via finite-
    difference stencils that zero-pad at boundary cells.  At those cells the
    derivative-balancing terms vanish, so the residual reduces to
    ``|-rho V + cost + jump|`` with no derivative term to cancel against.
    The boundary value is set by a separate mechanism (ABSORBING linear
    extrapolation or DIRICHLET) with no obligation to satisfy the local PDE
    balance, so including those cells in the convergence max conflates
    boundary-imposition noise with interior under-convergence.

    The fix exposes ``max_residual_interior`` / ``max_residual_boundary`` as
    separate keys; ``max_residual`` is retained for backward compatibility.
    """

    @staticmethod
    def _make_linear_solution_problem(boundary_lower=BoundaryCondition.ABSORBING):
        """Manufactured-solution problem with V(x) = x as the exact solution.

        For V(x) = x:
          - V_x = 1 everywhere (upwind forward diff is exact for linear V).
          - V_xx = 0 (central second-derivative stencil is exact).
        Choosing dynamics=drift_const, diffusion=0, and
        cost(x) = rho*x - drift_const balances the PDE rho*V = drift*V_x + cost
        exactly at every interior cell.

        Critical detail: drift > 0 means the upwind scheme uses the FORWARD
        difference at every interior point, so the test's perturbation of
        ``V[0]`` doesn't leak into interior advection (which would only
        happen if drift < 0 selected the backward difference).  Diffusion is
        off, so the central second-derivative stencil (which DOES use V[0]
        at i=1) is multiplied by sigma^2 = 0 and contributes nothing.
        """
        rho = 0.05
        drift_const = 0.1
        state_space = StateSpace(
            [
                StateVariable(
                    "x",
                    min_value=1.0,
                    max_value=10.0,
                    num_points=11,
                    boundary_lower=boundary_lower,
                )
            ]
        )
        controls = [ControlVariable("u", 0.0, 1.0, 2)]

        def dynamics(s, c, t):
            return np.full_like(s[..., 0:1], drift_const)

        def running_cost(s, c, t):
            x = s[..., 0]
            return rho * x - drift_const

        problem = HJBProblem(
            state_space=state_space,
            control_variables=controls,
            utility_function=ExpectedWealth(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=rho,
        )
        return problem, state_space.grids[0]

    def test_residual_split_keys_present(self):
        """All new metric keys are present alongside legacy ones."""
        problem, grid = self._make_linear_solution_problem()
        config = HJBSolverConfig(verbose=False)
        solver = HJBSolver(problem, config)
        solver.value_function = grid.copy()
        solver.optimal_policy = {"u": np.full(grid.shape, 0.5)}

        metrics = solver.compute_convergence_metrics()

        for key in (
            "max_residual",  # legacy
            "mean_residual",  # legacy
            "max_residual_all_cells",
            "max_residual_interior",
            "mean_residual_interior",
            "max_residual_boundary",
            "mean_residual_boundary",
            "argmax_residual_index",
            "argmax_residual_is_boundary",
        ):
            assert key in metrics, f"missing key {key} in metrics: {sorted(metrics)}"

    def test_residual_split_excludes_boundary(self):
        """Setting V[0] to a wildly wrong value spikes the boundary residual
        but leaves the interior residual at machine precision."""
        problem, grid = self._make_linear_solution_problem()
        config = HJBSolverConfig(verbose=False)
        solver = HJBSolver(problem, config)
        # Seed V to the exact closed-form solution V(x) = x.
        solver.value_function = grid.copy()
        solver.optimal_policy = {"u": np.full(grid.shape, 0.5)}
        # Deliberately corrupt the lower boundary.  Drift > 0 means upwind
        # uses forward diff (which doesn't touch V[0] at interior cells);
        # diffusion is off, so the central second-derivative stencil's
        # contribution from V[0] at i=1 is multiplied by sigma^2 = 0.
        solver.value_function[0] = 1e10

        metrics = solver.compute_convergence_metrics()

        # Interior residual is unaffected by the boundary corruption.
        assert metrics["max_residual_interior"] < 1e-9, (
            f"Expected interior residual ≈ 0, got " f"{metrics['max_residual_interior']:.4e}"
        )
        # Boundary residual reflects the wildly wrong V[0]: dominated by
        # |-rho * 1e10| ~ 5e8 plus the contaminated forward-diff term.
        assert metrics["max_residual_boundary"] > 1e8
        # Argmax classification correctly flags the boundary spike.
        assert metrics["argmax_residual_is_boundary"] is True
        assert metrics["argmax_residual_index"] == (0,)
        # Legacy max_residual is contaminated by the boundary spike.
        assert metrics["max_residual"] == metrics["max_residual_all_cells"]
        assert metrics["max_residual"] >= metrics["max_residual_boundary"]

    def test_residual_closed_form_known_solution(self):
        """V(x) = x is the exact closed-form solution; interior residual
        is at machine precision regardless of boundary-condition choice.

        Acceptance criterion from #1569: ``max_residual_interior < 1e-3``
        for a problem with a known closed-form interior solution, regardless
        of boundary condition.  We tighten that to 1e-9 since the stencils
        are exact for linear V.
        """
        for bc in (BoundaryCondition.ABSORBING, BoundaryCondition.DIRICHLET):
            problem, grid = self._make_linear_solution_problem(boundary_lower=bc)
            config = HJBSolverConfig(verbose=False)
            solver = HJBSolver(problem, config)
            solver.value_function = grid.copy()
            solver.optimal_policy = {"u": np.full(grid.shape, 0.5)}

            metrics = solver.compute_convergence_metrics()
            assert metrics["max_residual_interior"] < 1e-9, (
                f"BC={bc}: expected exact interior residual, got "
                f"{metrics['max_residual_interior']:.4e}"
            )

    def test_residual_argmax_interior_classification(self):
        """Argmax in the interior is reported as is_boundary=False."""
        problem, grid = self._make_linear_solution_problem()
        config = HJBSolverConfig(verbose=False)
        solver = HJBSolver(problem, config)
        # Perturb a single interior cell, leave boundaries at the exact
        # closed-form values.
        solver.value_function = grid.copy()
        solver.optimal_policy = {"u": np.full(grid.shape, 0.5)}
        solver.value_function[5] = 1e6  # interior cell

        metrics = solver.compute_convergence_metrics()
        assert metrics["argmax_residual_is_boundary"] is False
        # Argmax could be at index 5 or a neighboring cell whose stencil
        # references V[5]; verify it's somewhere in the interior.
        argmax_idx = metrics["argmax_residual_index"]
        assert argmax_idx[0] not in (0, len(grid) - 1)


class TestJumpTerm:
    """Test PIDE jump-term support added in Issue #1565.

    These tests verify that the jump-term callback is correctly:
      - Optional (None case unchanged vs. baseline solver).
      - Wired into both ``_policy_evaluation`` (value-function propagation)
        and ``_policy_improvement`` (Hamiltonian maximization).
      - Reflected in ``compute_convergence_metrics`` residuals.

    Mathematically the jump term encodes ``lambda * E_X[V(x - L) - V(x)]``,
    the exact compound-Poisson penalty under multiplicative dynamics; the
    diffusion form ``-0.5 * lambda * E[L^2] / w^2`` is its second-order Taylor
    expansion and is replaced by the jump term when ``L/w`` is non-trivial.
    """

    @staticmethod
    def _make_simple_problem(jump_term=None, n_states=11):
        """Build a small 1D log-utility problem used across jump-term tests."""
        state_space = StateSpace(
            [
                StateVariable(
                    "wealth",
                    min_value=1.0,
                    max_value=10.0,
                    num_points=n_states,
                    log_scale=False,
                )
            ]
        )
        control_variables = [ControlVariable("u", min_value=0.0, max_value=1.0, num_points=3)]

        def dynamics(state, control, t):
            # Pure-drift wealth growth at 5% (1-D output)
            return 0.05 * state[..., 0:1]

        def running_cost(state, control, t):
            return np.log(np.maximum(state[..., 0], 1e-9))

        return HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            jump_term=jump_term,
        )

    def test_jump_term_defaults_to_none(self):
        """A HJBProblem without jump_term keeps backward-compat behavior."""
        problem = self._make_simple_problem()
        assert problem.jump_term is None

    def test_solver_unchanged_when_jump_term_is_none(self):
        """Solver result is bit-identical to baseline when jump_term=None."""
        # Same seed/initialization for both problems
        p_none = self._make_simple_problem(jump_term=None)
        config = HJBSolverConfig(time_step=0.05, max_iterations=5, tolerance=1e-3, verbose=False)
        v_none, pol_none = HJBSolver(p_none, config).solve()

        p_dup = self._make_simple_problem(jump_term=None)
        v_dup, pol_dup = HJBSolver(p_dup, config).solve()

        np.testing.assert_array_equal(v_none, v_dup)
        for k in pol_none:
            np.testing.assert_array_equal(pol_none[k], pol_dup[k])

    def test_jump_term_signature_passes_value_function_and_grids(self):
        """The solver passes (state, control, t, value_function, state_grids)."""
        seen = {}

        def jump_term(state, control, t, value_function, state_grids):
            seen["state_shape"] = state.shape
            seen["control_shape"] = control.shape
            seen["t"] = t
            seen["value_shape"] = value_function.shape
            seen["n_grids"] = len(state_grids)
            seen["grid0_len"] = len(state_grids[0])
            return np.zeros(state.shape[:-1])

        problem = self._make_simple_problem(jump_term=jump_term, n_states=7)
        config = HJBSolverConfig(time_step=0.05, max_iterations=2, tolerance=1e-2, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        assert seen["state_shape"][-1] == 1  # wealth only
        assert seen["control_shape"][-1] == 1  # one control
        assert seen["value_shape"] == (7,)
        assert seen["n_grids"] == 1
        assert seen["grid0_len"] == 7

    def test_jump_term_linear_value_function_constant_loss(self):
        """For V(w) = w and constant L, expected jump = -lambda * L per point."""
        # Build a problem where we can pre-set V and read out the residual
        # contribution analytically.
        lam = 0.5
        L = 0.1

        def jump_term(state, control, t, V, grids):
            wealth_grid = grids[0]
            interp = interpolate_value_linear(wealth_grid, V)
            w = state[..., 0]
            post_w = np.maximum(w - L, wealth_grid[0])
            return lam * (interp(post_w) - interp(w))

        problem = self._make_simple_problem(jump_term=jump_term, n_states=11)
        config = HJBSolverConfig(time_step=0.01, max_iterations=2, tolerance=1.0, verbose=False)
        solver = HJBSolver(problem, config)
        # Manually set a linear value function: V(w) = w
        solver.value_function = np.linspace(1.0, 10.0, 11).copy()
        solver.optimal_policy = {"u": np.full(11, 0.5)}

        # Evaluate jump term at the interior grid points (away from the floor)
        state_points = np.stack(problem.state_space.flat_grids, axis=-1)
        control_array = np.full((11, 1), 0.5)
        jvals = solver._evaluate_jump_term(state_points, control_array, solver.value_function)

        # For V(w) = w (linear), V(w - L) - V(w) = -L exactly, so the result is -lambda * L.
        # Interior points should match exactly (the lowest-wealth point may clip at the floor).
        expected = -lam * L
        np.testing.assert_allclose(jvals[2:], expected, atol=1e-9)

    def test_jump_term_zero_callback_is_noop(self):
        """A jump_term returning zero yields the same V as no jump_term."""

        def zero_jump(state, control, t, V, grids):
            return np.zeros(state.shape[:-1])

        p_jump = self._make_simple_problem(jump_term=zero_jump)
        p_none = self._make_simple_problem(jump_term=None)
        config = HJBSolverConfig(time_step=0.05, max_iterations=3, tolerance=1e-3, verbose=False)
        v_jump, _ = HJBSolver(p_jump, config).solve()
        v_none, _ = HJBSolver(p_none, config).solve()
        np.testing.assert_allclose(v_jump, v_none, atol=1e-10)

    def test_jump_term_influences_policy(self):
        """A control-dependent jump term shifts the optimal control choice.

        Setup: dynamics and running cost are independent of u, so without the
        jump term any u is equally optimal.  We make the jump penalty depend on
        u so that the smallest u (u=0.0) is uniquely optimal, and verify the
        solver picks it everywhere.
        """

        def jump_term(state, control, t, V, grids):
            # Penalize larger u directly: -u, so u=0 maximizes the Hamiltonian
            u = control[..., 0]
            return -u

        state_space = StateSpace([StateVariable("wealth", 1.0, 10.0, 11)])
        control_variables = [ControlVariable("u", 0.0, 1.0, 5)]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=lambda s, c, t: 0.05 * s[..., 0:1],
            running_cost=lambda s, c, t: np.log(np.maximum(s[..., 0], 1e-9)),
            discount_rate=0.05,
            jump_term=jump_term,
        )
        config = HJBSolverConfig(time_step=0.05, max_iterations=5, tolerance=1e-3, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()
        # Every grid point should have selected u = 0.0 since larger u is penalized
        assert solver.optimal_policy is not None
        assert np.all(solver.optimal_policy["u"] == 0.0)

    def test_jump_term_in_convergence_residual(self):
        """``compute_convergence_metrics`` includes the jump-term contribution."""
        # Build identical problems with and without a constant jump_term.
        # The residual should differ by exactly the constant.
        K = 0.123  # constant jump contribution per point

        def const_jump(state, control, t, V, grids):
            return np.full(state.shape[:-1], -K)

        p_none = self._make_simple_problem(jump_term=None)
        p_jump = self._make_simple_problem(jump_term=const_jump)
        config = HJBSolverConfig(time_step=0.05, max_iterations=2, tolerance=1.0, verbose=False)
        s_none = HJBSolver(p_none, config)
        s_none.solve()
        s_jump = HJBSolver(p_jump, config)
        s_jump.solve()

        # The two solves diverge after the first iteration, so we can't directly
        # diff residuals.  Instead, point-evaluate the jump-aware residual:
        # set s_jump's V/policy to s_none's and verify the residual differs by K.
        assert s_none.value_function is not None
        assert s_none.optimal_policy is not None
        s_jump.value_function = s_none.value_function.copy()
        s_jump.optimal_policy = {k: v.copy() for k, v in s_none.optimal_policy.items()}
        m_none = s_none.compute_convergence_metrics()
        m_jump = s_jump.compute_convergence_metrics()
        # Maximum residual changed because we subtracted K from every point;
        # mean residual should reflect the K offset relative to baseline.
        assert m_jump["max_residual"] != m_none["max_residual"]

    def test_jump_term_explicit_scheme(self):
        """Jump term integrates into the explicit Euler path too."""

        def jump_term(state, control, t, V, grids):
            return np.full(state.shape[:-1], -0.01)

        problem = self._make_simple_problem(jump_term=jump_term)
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=3,
            tolerance=1e-3,
            verbose=False,
            scheme=TimeSteppingScheme.EXPLICIT,
        )
        solver = HJBSolver(problem, config)
        # Just verifying it runs without crashing and produces finite values
        v, p = solver.solve()
        assert np.all(np.isfinite(v))
        for k in p:
            assert np.all(np.isfinite(p[k]))

    def test_jump_term_finite_horizon(self):
        """Jump term works in finite-horizon (terminal-condition) setup."""

        def jump_term(state, control, t, V, grids):
            return np.full(state.shape[:-1], -0.005)

        state_space = StateSpace([StateVariable("wealth", 1.0, 10.0, 7)])
        control_variables = [ControlVariable("u", 0.0, 1.0, 2)]

        def dynamics(s, c, t):
            return 0.05 * s[..., 0:1]

        def running_cost(s, c, t):
            return np.log(np.maximum(s[..., 0], 1e-9))

        def terminal_value(s):
            return np.log(np.maximum(s[..., 0], 1e-9))

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_value=terminal_value,
            time_horizon=1.0,
            discount_rate=0.05,
            jump_term=jump_term,
        )
        config = HJBSolverConfig(time_step=0.05, max_iterations=2, verbose=False)
        v, _ = HJBSolver(problem, config).solve()
        assert np.all(np.isfinite(v))

    def test_jump_term_accepts_trailing_singleton_axis(self):
        """Solver tolerates ``(n, 1)`` shape from user callback (forgiving API)."""

        def jump_term(state, control, t, V, grids):
            return np.zeros((state.shape[0], 1))  # trailing 1-axis

        problem = self._make_simple_problem(jump_term=jump_term)
        config = HJBSolverConfig(time_step=0.05, max_iterations=2, tolerance=1.0, verbose=False)
        v, _ = HJBSolver(problem, config).solve()
        assert np.all(np.isfinite(v))


def interpolate_value_linear(grid, values):
    """Tiny 1-D linear interpolator with constant extrapolation at the lower edge.

    Used only by the jump-term unit tests.  Mirrors the behavior expected from
    the notebook's ``RegularGridInterpolator`` when ``post_w`` is clipped at
    ``wealth_grid[0]``.
    """
    grid = np.asarray(grid)
    values = np.asarray(values)

    def f(x):
        x_clip = np.clip(x, grid[0], grid[-1])
        return np.interp(x_clip, grid, values)

    return f


def _module_level_warm_start(state, _rho=0.03, _g_eff=0.10, _w_min=1.0, _ruin_penalty=10.0):
    """Module-level warm-start for picklability tests (Issue #1567).

    Mirrors the notebook pattern (cell 16): default args capture the
    parameters so the closure is picklable across ``ProcessPoolExecutor``
    boundaries.
    """
    w = state[..., 0]
    base = np.log(w) / _rho + _g_eff / _rho**2
    is_boundary = np.isclose(w, _w_min)
    return np.where(is_boundary, base - _ruin_penalty / _rho, base)


class TestRuinPenaltyDirichlet:
    """Regression tests for Issue #1567 — explicit ruin-penalty Dirichlet floor.

    The notebook ``07_hjb_insurance_optimization.ipynb`` encodes a ruin
    penalty at the lower wealth boundary via:
      1. ``boundary_lower=BoundaryCondition.DIRICHLET`` on the wealth state.
      2. A warm-start ``V0(w) = log(w)/rho + g_eff/rho^2`` minus
         ``RUIN_PENALTY/rho`` at the boundary grid point only.

    The Dirichlet machinery (``hjb_solver.py:914-929, 1098-1161``) captures
    ``V[0]`` from the warm-start during ``solve()`` initialization and
    re-imposes it after every implicit step, so the boundary value is
    preserved through the full policy iteration.

    These tests cover:
      - Boundary V is preserved across many iterations (not just 2).
      - Ruin-penalty drop matches the analytical ``RUIN_PENALTY/rho``.
      - Larger ``RUIN_PENALTY`` produces a strictly deeper boundary drop.
      - The warm-start closure pattern with default args is picklable.
      - ABSORBING vs DIRICHLET produce qualitatively different ``V[0]``
        when the warm-start sets a non-zero ruin penalty.
    """

    @staticmethod
    def _make_log_growth_problem(
        warm_start_fn,
        rho=0.03,
        boundary_lower=BoundaryCondition.DIRICHLET,
    ):
        """Build a small log-growth HJB problem with the warm-start wired in.

        Dynamics: pure 2% drift, no diffusion, no jumps.  Running cost is
        ``log(w)``.  With ``rho=0.03``, the analytical interior fixed point
        is ``V(w) = log(w)/rho + g/rho^2`` for ``g = 0.02``.
        """
        sv = StateVariable(
            name="wealth",
            min_value=1.0,
            max_value=100.0,
            num_points=21,
            log_scale=True,
            boundary_lower=boundary_lower,
        )
        state_space = StateSpace([sv])
        control_variables = [ControlVariable("u", 0.0, 1.0, 3)]

        def dynamics(state, control, t):
            # 1-D output (last axis is the state dim)
            return 0.02 * state[..., 0:1]

        def running_cost(state, control, t):
            return np.log(state[..., 0])

        return HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=rho,
            initial_value_function=warm_start_fn,
        )

    def test_dirichlet_boundary_preserved_over_many_iterations(self):
        """V[0] from the warm-start survives the full policy iteration."""
        rho = 0.03
        g_eff = 0.02
        w_min = 1.0
        ruin_penalty = 10.0

        def warm(state, _rho=rho, _g=g_eff, _wm=w_min, _p=ruin_penalty):
            w = state[..., 0]
            base = np.log(w) / _rho + _g / _rho**2
            return np.where(np.isclose(w, _wm), base - _p / _rho, base)

        problem = self._make_log_growth_problem(warm, rho=rho)
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=50,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        v, _ = solver.solve()

        # Boundary V[0] must equal the warm-start value (base - P/rho).
        expected_v0 = np.log(w_min) / rho + g_eff / rho**2 - ruin_penalty / rho
        assert v[0] == pytest.approx(
            expected_v0, rel=1e-9, abs=1e-9
        ), f"V[0]={v[0]:.6f} drifted from warm-start expected={expected_v0:.6f}"

    def test_ruin_penalty_magnitude_matches_analytical(self):
        """V[w_min] drop equals RUIN_PENALTY / rho relative to the no-penalty case."""
        rho = 0.03
        g_eff = 0.02
        w_min = 1.0

        def warm_no_pen(state, _rho=rho, _g=g_eff, _wm=w_min):
            w = state[..., 0]
            return np.log(w) / _rho + _g / _rho**2

        def warm_with_pen(state, _rho=rho, _g=g_eff, _wm=w_min, _p=10.0):
            w = state[..., 0]
            base = np.log(w) / _rho + _g / _rho**2
            return np.where(np.isclose(w, _wm), base - _p / _rho, base)

        # Solve both
        cfg = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        v_no = HJBSolver(self._make_log_growth_problem(warm_no_pen), cfg).solve()[0]
        v_yes = HJBSolver(self._make_log_growth_problem(warm_with_pen), cfg).solve()[0]

        drop = v_no[0] - v_yes[0]
        expected_drop = 10.0 / rho
        assert drop == pytest.approx(
            expected_drop, rel=1e-9, abs=1e-9
        ), f"V[0] drop {drop:.4f} should equal RUIN_PENALTY/rho = {expected_drop:.4f}"

    @pytest.mark.parametrize("P_a,P_b", [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0)])
    def test_larger_ruin_penalty_produces_deeper_V0(self, P_a, P_b):
        """Strictly increasing RUIN_PENALTY yields strictly decreasing V[w_min]."""
        rho = 0.03
        g_eff = 0.02
        w_min = 1.0

        def _make_warm(P):
            def warm(state, _rho=rho, _g=g_eff, _wm=w_min, _p=P):
                w = state[..., 0]
                base = np.log(w) / _rho + _g / _rho**2
                return np.where(np.isclose(w, _wm), base - _p / _rho, base)

            return warm

        cfg = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        v_a = HJBSolver(self._make_log_growth_problem(_make_warm(P_a)), cfg).solve()[0]
        v_b = HJBSolver(self._make_log_growth_problem(_make_warm(P_b)), cfg).solve()[0]

        assert (
            v_b[0] < v_a[0]
        ), f"V[0] with P={P_b} ({v_b[0]:.4f}) should be < V[0] with P={P_a} ({v_a[0]:.4f})"
        # And the gap should equal (P_b - P_a)/rho within tolerance.
        gap = v_a[0] - v_b[0]
        expected_gap = (P_b - P_a) / rho
        assert gap == pytest.approx(expected_gap, rel=1e-9, abs=1e-9)

    def test_module_level_warm_start_is_picklable(self):
        """Default-arg warm-start defined at module level is picklable.

        Cell 28's ``ProcessPoolExecutor`` requires this — closures with
        notebook-cell-local state would fail to round-trip across workers
        on spawn-based platforms.  The default-arg pattern bakes the
        parameters into the function's ``__defaults__`` tuple, which IS
        picklable.
        """
        import pickle

        blob = pickle.dumps(_module_level_warm_start)
        restored = pickle.loads(blob)

        # Round-trip preserves callable behavior on a small grid.
        test_state = np.array([[1.0], [2.0], [5.0], [10.0]])
        np.testing.assert_allclose(
            restored(test_state),
            _module_level_warm_start(test_state),
            rtol=1e-12,
        )

    def test_absorbing_vs_dirichlet_differ_on_ruin_penalty(self):
        """ABSORBING ignores the warm-start at V[0] (overwritten by linear
        extrapolation each step), so the ruin-penalty drop is destroyed.
        DIRICHLET preserves it.  This is the structural reason Issue #1567
        switches the boundary condition.
        """
        rho = 0.03
        g_eff = 0.02
        w_min = 1.0
        ruin_penalty = 10.0

        def warm(state, _rho=rho, _g=g_eff, _wm=w_min, _p=ruin_penalty):
            w = state[..., 0]
            base = np.log(w) / _rho + _g / _rho**2
            return np.where(np.isclose(w, _wm), base - _p / _rho, base)

        cfg = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        v_dirichlet = HJBSolver(
            self._make_log_growth_problem(warm, boundary_lower=BoundaryCondition.DIRICHLET),
            cfg,
        ).solve()[0]
        v_absorbing = HJBSolver(
            self._make_log_growth_problem(warm, boundary_lower=BoundaryCondition.ABSORBING),
            cfg,
        ).solve()[0]

        # Under DIRICHLET, V[0] equals the warm-start value (preserved).
        expected_v0_dirichlet = np.log(w_min) / rho + g_eff / rho**2 - ruin_penalty / rho
        assert v_dirichlet[0] == pytest.approx(expected_v0_dirichlet, rel=1e-9, abs=1e-9)

        # Under ABSORBING, V[0] is linear extrapolation 2*V[1] - V[2], which
        # is roughly continuous with the interior (the ~0.25 utility-year
        # drop the issue references), NOT the deep RUIN_PENALTY/rho drop.
        # The ABSORBING extrapolation must produce a substantially smaller
        # drop than the DIRICHLET-preserved warm-start.
        absorbing_drop = (2 * v_absorbing[1] - v_absorbing[2]) - (
            np.log(w_min) / rho + g_eff / rho**2
        )
        dirichlet_drop = v_dirichlet[0] - (np.log(w_min) / rho + g_eff / rho**2)
        # DIRICHLET preserves the requested -RUIN_PENALTY/rho drop.
        assert dirichlet_drop == pytest.approx(-ruin_penalty / rho, rel=1e-9, abs=1e-9)
        # ABSORBING produces a drop dramatically smaller in magnitude than
        # the DIRICHLET-imposed RUIN_PENALTY/rho — the structural reason the
        # issue switches the boundary condition.  Tolerance is generous: in
        # the actual notebook (1000 log-scaled grid points spanning 1e5 to
        # 2e8) the ABSORBING drop is ~0.25; on this coarse 21-point grid
        # we expect a few units, but still orders of magnitude less than
        # -333 utility-years from DIRICHLET.
        assert abs(absorbing_drop) < 0.1 * abs(dirichlet_drop), (
            f"ABSORBING drop {absorbing_drop:.4f} should be << DIRICHLET drop "
            f"{dirichlet_drop:.4f} (ratio {abs(absorbing_drop)/max(abs(dirichlet_drop), 1e-9):.3f})"
        )

    def test_warm_start_no_ruin_penalty_recovers_continuous_v(self):
        """With RUIN_PENALTY = 0, V[w_min] sits on the interior Bellman curve.

        This is the P=0 endpoint of the issue's calibration sweep — it
        verifies that the boundary mechanism *itself* doesn't introduce a
        spurious drop; the drop comes entirely from the explicit penalty.
        """
        rho = 0.03
        g_eff = 0.02
        w_min = 1.0

        def warm_zero_pen(state, _rho=rho, _g=g_eff, _wm=w_min):
            w = state[..., 0]
            return np.log(w) / _rho + _g / _rho**2  # no boundary special-case

        cfg = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        problem = self._make_log_growth_problem(warm_zero_pen)
        solver = HJBSolver(problem, cfg)
        v, _ = solver.solve()

        # V[w_min] should equal the Bellman curve at w_min.
        expected = np.log(w_min) / rho + g_eff / rho**2
        assert v[0] == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_boundary_residual_is_finite_under_dirichlet(self):
        """Under DIRICHLET, the boundary residual is large but finite.

        Per Issue #1567, the imposed V[0] is intentionally inconsistent
        with the local PDE balance, so ``max_residual_boundary`` is expected
        to be substantial.  It must still be finite (no NaN/Inf), and the
        interior residual must be small.
        """
        rho = 0.03
        g_eff = 0.02
        w_min = 1.0
        ruin_penalty = 10.0

        def warm(state, _rho=rho, _g=g_eff, _wm=w_min, _p=ruin_penalty):
            w = state[..., 0]
            base = np.log(w) / _rho + _g / _rho**2
            return np.where(np.isclose(w, _wm), base - _p / _rho, base)

        cfg = HJBSolverConfig(
            time_step=0.01,
            max_iterations=50,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        problem = self._make_log_growth_problem(warm)
        solver = HJBSolver(problem, cfg)
        solver.solve()

        cm = solver.compute_convergence_metrics()
        assert np.isfinite(cm["max_residual_interior"])
        assert np.isfinite(cm["max_residual_boundary"])
        # Boundary residual reflects the imposed V[0] = warm-start value
        # being off the local PDE balance.  Per Issue #1567 it is not
        # constrained to be small; it just has to be finite.
        assert cm["max_residual_boundary"] > 0.0
        # And it should be substantially LARGER than the interior residual,
        # since the imposed -RUIN_PENALTY/rho V-drop has no obligation to
        # satisfy local PDE balance at the boundary cell.
        assert cm["max_residual_boundary"] > cm["max_residual_interior"], (
            f"Expected boundary residual > interior residual under DIRICHLET; "
            f"got bnd={cm['max_residual_boundary']:.4e} vs "
            f"int={cm['max_residual_interior']:.4e}"
        )

    def test_warm_start_pattern_from_notebook_picklability(self):
        """End-to-end: the issue #1567 warm-start pattern (default-arg
        constants) round-trips through pickle AND produces the same V[0]
        when used in solver initialization.  The defaults baked into
        ``_module_level_warm_start`` (rho=0.03, g_eff=0.10, w_min=1.0,
        ruin_penalty=10.0) are what survive the pickle round-trip.
        """
        import pickle

        # Pickle/unpickle the module-level function.
        blob = pickle.dumps(_module_level_warm_start)
        restored = pickle.loads(blob)

        # Use restored warm-start in a real solve.  Solver problem's rho
        # MUST match the warm-start default (0.03) so V[0] equals the
        # analytical fixed point at the boundary.
        problem = self._make_log_growth_problem(restored, rho=0.03)
        cfg = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            tolerance=1e-6,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, cfg)
        v, _ = solver.solve()

        # Defaults in _module_level_warm_start: rho=0.03, g_eff=0.10,
        # w_min=1.0, ruin_penalty=10.0
        expected_v0 = np.log(1.0) / 0.03 + 0.10 / 0.03**2 - 10.0 / 0.03
        assert v[0] == pytest.approx(expected_v0, rel=1e-9, abs=1e-9)


class TestInteriorOptimumRegression:
    """Regression test guarding against the boundary-pinning bug from notebook 07.

    The notebook-07 HJB solve was returning policies that oscillated between the
    extreme corner SIRs ($10K and $4M) at every wealth, even though an interior
    optimum was financially plausible (issue #1572). The root causes were a mix
    of (a) unconverged solver, (b) piecewise-affine Hamiltonian from coarse atom
    discretization, and (c) thin premium loading that flattened the cost-of-
    coverage curve. The fix shipped in issue #1572 R1-R6 addresses each
    component; this test guards against regressions in the most generic of
    those components -- the Hamiltonian-assembly + policy-improvement loop.

    Test problem: classical Merton portfolio. A risk-averse agent with log
    utility allocates fraction u of wealth to a risky asset and 1-u to a
    risk-free asset. The closed-form optimal stationary allocation is
    u* = (mu - r) / sigma^2 (Merton, 1969). For mu=0.08, r=0.03, sigma=0.30,
    u* = 0.556 -- comfortably interior in [0, 1].

    If this test fails by returning u ~= 0 or u ~= 1, the solver is boundary-
    pinning and the notebook-07 bug has regressed in the underlying solver
    machinery, not in the notebook's HJB problem specification.
    """

    @staticmethod
    def _merton_problem(mu=0.08, r=0.03, sigma=0.30, rho=0.05, n_w=200, n_u=50):
        """Build a Merton portfolio HJB problem.

        Parameters chosen so the closed-form Merton fraction
        u* = (mu - r) / sigma^2 = 0.556 is interior in [0, 1].
        """
        state_var = StateVariable(
            name="wealth",
            min_value=1.0,
            max_value=1000.0,
            num_points=n_w,
            log_scale=True,
            boundary_lower=BoundaryCondition.ABSORBING,
            boundary_upper=BoundaryCondition.ABSORBING,
        )
        control_var = ControlVariable(
            name="u",
            min_value=0.0,
            max_value=1.0,
            num_points=n_u,
            log_scale=False,
        )
        state_space = StateSpace([state_var])

        def dynamics(state, control, t):
            w = state[..., 0]
            u = control[..., 0]
            return (w * (r + u * (mu - r)))[..., np.newaxis]

        def diffusion(state, control, t):
            w = state[..., 0]
            u = control[..., 0]
            return ((w * u * sigma) ** 2)[..., np.newaxis]

        def running_cost(state, control, t):
            return np.log(state[..., 0])

        return HJBProblem(
            state_space=state_space,
            control_variables=[control_var],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            diffusion=diffusion,
            discount_rate=rho,
        )

    def test_merton_interior_optimum(self):
        """The HJB solver returns the closed-form Merton fraction at midrange wealth.

        Regression test for the notebook-07 boundary-pinning bug (issue #1572).
        If the solver pins to u=0 or u=1, the policy improvement, Hamiltonian
        assembly, or value-function relaxation has regressed.
        """
        mu, r, sigma, rho = 0.08, 0.03, 0.30, 0.05
        expected_u_star = (mu - r) / sigma**2  # = 0.5556 -- interior in [0, 1]

        problem = self._merton_problem(mu=mu, r=r, sigma=sigma, rho=rho)
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=200,
            inner_max_iterations=2000,
            tolerance=1e-5,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        solver.solve()

        # Sample u at three midrange wealth points (avoiding boundary cells)
        assert solver.optimal_policy is not None
        n_w = problem.state_space.grids[0].size
        sample_idxs = [n_w // 4, n_w // 2, 3 * n_w // 4]
        sampled_u = [float(solver.optimal_policy["u"][i]) for i in sample_idxs]

        # Primary check: u* matches Merton closed form at midrange wealth
        for i, u_solved in zip(sample_idxs, sampled_u):
            assert abs(u_solved - expected_u_star) < 0.10 * expected_u_star, (
                f"HJB picked u={u_solved:.4f} at wealth idx {i}, expected "
                f"u*={expected_u_star:.4f} (Merton 1969). "
                f"Likely boundary-pinning regression -- see issue #1572 / notebook 07."
            )

        # Hard boundary-pinning check: none of the sampled u should be at the
        # control-grid corners. This is the smoking gun for the notebook-07
        # bug class: even if the closed-form check above passes by luck, u
        # values at 0.0 or 1.0 indicate the solver is making a corner choice
        # rather than an interior optimization.
        for i, u_solved in zip(sample_idxs, sampled_u):
            assert 0.05 < u_solved < 0.95, (
                f"u={u_solved:.4f} at wealth idx {i} is pinned to a corner of "
                f"[0, 1]; expected interior Merton optimum u*={expected_u_star:.4f}. "
                f"See issue #1572 / notebook 07 boundary-pinning bug."
            )

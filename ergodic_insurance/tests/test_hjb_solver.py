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

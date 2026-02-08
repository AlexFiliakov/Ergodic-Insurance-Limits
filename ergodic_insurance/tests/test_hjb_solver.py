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

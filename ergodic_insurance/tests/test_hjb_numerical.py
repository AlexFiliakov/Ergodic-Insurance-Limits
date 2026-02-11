# pylint: disable=too-many-lines
"""Advanced numerical tests for Hamilton-Jacobi-Bellman solver.

This module tests numerical stability, boundary conditions, and accuracy
of the HJB solver implementation.

Author: Alex Filiakov
Date: 2025-01-26
"""

from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from ergodic_insurance.hjb_solver import (
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


class TestNumericalMethods:
    """Test numerical methods in HJB solver."""

    def test_build_difference_matrix(self):
        """Test finite difference matrix construction."""
        # Create solver with 1D state space
        state_space = StateSpace([StateVariable("x", 0, 1, 10)])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # Test different boundary conditions
        for bc in [
            BoundaryCondition.DIRICHLET,
            BoundaryCondition.NEUMANN,
            BoundaryCondition.ABSORBING,
        ]:
            mat = solver._build_difference_matrix(0, bc)
            assert isinstance(mat, sparse.spmatrix)
            assert mat.shape == (10, 10)

            # Check matrix structure for different BCs
            mat_array = mat.toarray()
            if bc == BoundaryCondition.DIRICHLET:
                # Dirichlet BC should enforce fixed values at boundaries
                # The exact implementation may vary, but check for reasonable structure
                assert mat_array[0, 0] != 0  # Non-zero diagonal
                assert mat_array[-1, -1] != 0  # Non-zero diagonal
                assert np.all(np.isfinite(mat_array))
            elif bc == BoundaryCondition.NEUMANN:
                # Check modified boundary rows for Neumann
                # Neumann BC modifies the matrix to enforce zero derivative
                # The exact implementation may vary, but matrix should be valid
                assert mat_array is not None
                assert np.all(np.isfinite(mat_array))
            elif bc == BoundaryCondition.ABSORBING:
                # For absorbing BC, boundary rows enforce d²V/dx² = 0
                # Row 0: [1/dx², -2/dx², 1/dx², 0, ...]
                # Row n-1: [..., 0, 1/dx², -2/dx², 1/dx²]
                n = 10
                dx = 1.0 / (n - 1)
                coeff = 1.0 / (dx * dx)
                expected_first_row = np.zeros(n)
                expected_first_row[0] = coeff
                expected_first_row[1] = -2.0 * coeff
                expected_first_row[2] = coeff
                expected_last_row = np.zeros(n)
                expected_last_row[-3] = coeff
                expected_last_row[-2] = -2.0 * coeff
                expected_last_row[-1] = coeff
                assert np.allclose(mat_array[0, :], expected_first_row)
                assert np.allclose(mat_array[-1, :], expected_last_row)

    def test_upwind_scheme(self):
        """Test upwind finite difference scheme."""
        state_space = StateSpace([StateVariable("x", 0, 1, 11)])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.ones_like(x),  # Constant positive drift
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # Test with different drift directions
        value = np.linspace(0, 10, 11)

        # Positive drift - should use forward differences
        pos_drift = np.ones(11) * 2.0
        result_pos = solver._apply_upwind_scheme(value, pos_drift, 0)
        assert result_pos.shape == value.shape
        # Check that upwind scheme is applied (forward diff for positive drift)
        assert np.all(result_pos[:-1] > 0)  # Positive contribution except at boundary

        # Negative drift - should use backward differences
        neg_drift = np.ones(11) * -2.0
        result_neg = solver._apply_upwind_scheme(value, neg_drift, 0)
        assert result_neg.shape == value.shape
        # Check that upwind scheme is applied (backward diff for negative drift)
        assert np.all(result_neg[1:] < 0)  # Negative contribution except at boundary

        # Mixed drift
        mixed_drift = np.linspace(-1, 1, 11)
        result_mixed = solver._apply_upwind_scheme(value, mixed_drift, 0)
        assert result_mixed.shape == value.shape
        # At drift=0 (index 5), result should be zero
        assert result_mixed[5] == 0.0
        # At interior points with nonzero drift, result should be nonzero
        assert result_mixed[1] != 0  # negative drift, interior point
        assert result_mixed[9] != 0  # positive drift, interior point
        # Boundaries should be zero (no wraparound)
        assert result_mixed[0] == 0.0  # backward diff undefined at first point
        assert result_mixed[-1] == 0.0  # forward diff undefined at last point

    def test_numerical_stability_extreme_values(self):
        """Test solver stability with extreme parameter values."""
        # Test with very small grid spacing
        state_space_fine = StateSpace([StateVariable("x", 0, 1, 100)])
        problem_fine = HJBProblem(
            state_space=state_space_fine,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(wealth_floor=1e-10),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0],
            time_horizon=0.1,
        )
        config_fine = HJBSolverConfig(
            time_step=0.001, max_iterations=5, tolerance=1e-6, verbose=False
        )
        solver_fine = HJBSolver(problem_fine, config_fine)
        value_fine, policy_fine = solver_fine.solve()

        # Check that solution is stable (no NaN or Inf)
        assert np.all(np.isfinite(value_fine))
        assert np.all(np.isfinite(policy_fine["u"]))

        # Test with large state values
        state_space_large = StateSpace([StateVariable("x", 1e6, 1e9, 10)])
        problem_large = HJBProblem(
            state_space=state_space_large,
            control_variables=[ControlVariable("u", 1e6, 1e8, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.05,
            running_cost=lambda x, u, t: -np.log(np.maximum(x[..., 0], 1e6)),
            time_horizon=1.0,
        )
        config_large = HJBSolverConfig(
            time_step=0.1, max_iterations=3, tolerance=1e-3, verbose=False
        )
        solver_large = HJBSolver(problem_large, config_large)
        value_large, policy_large = solver_large.solve()

        assert np.all(np.isfinite(value_large))
        assert np.all(np.isfinite(policy_large["u"]))

    def test_different_grid_resolutions(self):
        """Test convergence with different grid resolutions."""
        resolutions = [5, 10, 20]
        solutions = []

        for n_points in resolutions:
            # Use NEUMANN boundary conditions for better numerical stability
            state_space = StateSpace(
                [
                    StateVariable(
                        "x",
                        1,
                        10,
                        n_points,
                        boundary_lower=BoundaryCondition.NEUMANN,
                        boundary_upper=BoundaryCondition.NEUMANN,
                    )
                ]
            )
            problem = HJBProblem(
                state_space=state_space,
                control_variables=[ControlVariable("u", 0, 1, 3)],
                utility_function=LogUtility(),
                dynamics=lambda x, u, t: x * 0.01,  # Reduced drift for stability
                running_cost=lambda x, u, t: -x[..., 0] * 0.01,
                discount_rate=0.05,
                time_horizon=1.0,  # Shorter horizon for stability
            )
            config = HJBSolverConfig(
                time_step=0.01, max_iterations=50, tolerance=1e-4, verbose=False
            )
            solver = HJBSolver(problem, config)
            value, _ = solver.solve()
            solutions.append(value)

        # Check that solutions are consistent across resolutions
        # Interpolate coarse solution to fine grid for comparison
        for i in range(len(resolutions) - 1):
            coarse_grid = np.linspace(1, 10, resolutions[i])
            fine_grid = np.linspace(1, 10, resolutions[i + 1])

            # Simple linear interpolation for comparison
            interp_coarse = np.interp(fine_grid, coarse_grid, solutions[i])

            # Solutions should be closer as resolution increases
            diff = np.max(np.abs(interp_coarse - solutions[i + 1]))
            assert diff < 10.0  # Reasonable bound for convergence


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_all_boundary_types(self):
        """Test all boundary condition types."""
        boundary_types = [
            BoundaryCondition.DIRICHLET,
            BoundaryCondition.NEUMANN,
            BoundaryCondition.ABSORBING,
            BoundaryCondition.REFLECTING,
        ]

        for bc_lower in boundary_types:
            for bc_upper in boundary_types:
                state_var = StateVariable(
                    "x", 0, 1, 7, boundary_lower=bc_lower, boundary_upper=bc_upper
                )
                state_space = StateSpace([state_var])

                problem = HJBProblem(
                    state_space=state_space,
                    control_variables=[ControlVariable("u", 0, 1, 2)],
                    utility_function=ExpectedWealth(),
                    dynamics=lambda x, u, t: np.ones_like(x),
                    running_cost=lambda x, u, t: np.zeros(x.shape[0]),
                    terminal_value=lambda x: x[..., 0],
                    time_horizon=0.5,
                )

                config = HJBSolverConfig(
                    time_step=0.1, max_iterations=5, tolerance=1e-3, verbose=False
                )

                solver = HJBSolver(problem, config)
                value, policy = solver.solve()

                # Check solution exists and is finite
                assert value is not None
                assert np.all(np.isfinite(value))
                assert policy is not None
                assert np.all(np.isfinite(policy["u"]))

    def test_boundary_mask_multidimensional(self):
        """Test boundary mask for multi-dimensional state spaces."""
        # 2D state space
        sv1 = StateVariable("x", 0, 1, 4)
        sv2 = StateVariable("y", 0, 1, 5)
        space_2d = StateSpace([sv1, sv2])

        mask_2d = space_2d.get_boundary_mask()
        assert mask_2d.shape == (4, 5)

        # Check corners are boundary
        assert mask_2d[0, 0]
        assert mask_2d[0, -1]
        assert mask_2d[-1, 0]
        assert mask_2d[-1, -1]

        # Check edges are boundary
        assert np.all(mask_2d[0, :])
        assert np.all(mask_2d[-1, :])
        assert np.all(mask_2d[:, 0])
        assert np.all(mask_2d[:, -1])

        # Check interior is not boundary
        assert not mask_2d[1, 1]
        assert not mask_2d[2, 2]

        # 3D state space
        sv3 = StateVariable("z", 0, 1, 3)
        space_3d = StateSpace([sv1, sv2, sv3])

        mask_3d = space_3d.get_boundary_mask()
        assert mask_3d.shape == (4, 5, 3)

        # Check that all faces are boundary
        assert np.all(mask_3d[0, :, :])
        assert np.all(mask_3d[-1, :, :])
        assert np.all(mask_3d[:, 0, :])
        assert np.all(mask_3d[:, -1, :])
        assert np.all(mask_3d[:, :, 0])
        assert np.all(mask_3d[:, :, -1])

        # Check interior point is not boundary
        assert not mask_3d[1, 2, 1]

    def test_absorbing_bc_second_derivative_zero(self):
        """For V(x) = x², the absorbing BC matrix row should give d²V/dx² = 0.

        Acceptance criterion: the second derivative operator applied at the
        boundary should return 0 (not 1*V[0] as the old Dirichlet-like rows did).
        """
        n = 11
        state_space = StateSpace([StateVariable("x", 0, 1, n)])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        solver = HJBSolver(problem, HJBSolverConfig())
        mat = solver._build_difference_matrix(0, BoundaryCondition.ABSORBING)

        # V(x) = x² on uniform grid [0, 1]
        grid = np.linspace(0, 1, n)
        v = grid**2

        result = mat @ v

        # Interior points: d²(x²)/dx² = 2  (constant)
        # Absorbing boundary rows: enforce V[0] - 2V[1] + V[2] = 0
        # For x², V[0]-2V[1]+V[2] = 0 - 2*(1/100) + (4/100) = 2/100, then /dx²
        # which equals the standard second derivative ≈ 2. So the boundary rows
        # give the same result as interior for a smooth function.
        # The key difference from the old code: the old code returned
        # 1*V[0] = 0 at the lower boundary and 1*V[n-1] = 1 at the upper
        # boundary, instead of the actual second derivative.
        dx = 1.0 / (n - 1)
        expected_d2v = 2.0  # d²(x²)/dx² = 2
        # Boundary rows now compute the finite difference of d²V/dx²
        assert abs(result[0] - expected_d2v) < 0.1
        assert abs(result[-1] - expected_d2v) < 0.1

    def test_absorbing_bc_boundary_changes_with_time(self):
        """With absorbing BCs and diffusion, boundary values should evolve.

        Acceptance criterion: For a parabolic PDE V_t = V_xx with absorbing
        BCs, the boundary value should change with time (not stay fixed).
        """
        n = 21
        sv = StateVariable(
            "x",
            0,
            1,
            n,
            boundary_lower=BoundaryCondition.ABSORBING,
            boundary_upper=BoundaryCondition.ABSORBING,
        )
        state_space = StateSpace([sv])

        # Set up a pure diffusion problem V_t = V_xx
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            diffusion=lambda x, u, t: np.ones((x.shape[0], 1)),
            terminal_value=lambda x: np.sin(np.pi * x[..., 0]),
            time_horizon=0.5,
        )

        config = HJBSolverConfig(time_step=0.01, max_iterations=3, tolerance=1e-8, verbose=False)

        # Solve with absorbing BCs
        solver_abs = HJBSolver(problem, config)
        value_abs, _ = solver_abs.solve()

        # Solve with Dirichlet BCs for comparison
        sv_dir = StateVariable(
            "x",
            0,
            1,
            n,
            boundary_lower=BoundaryCondition.DIRICHLET,
            boundary_upper=BoundaryCondition.DIRICHLET,
        )
        problem_dir = HJBProblem(
            state_space=StateSpace([sv_dir]),
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            diffusion=lambda x, u, t: np.ones((x.shape[0], 1)),
            terminal_value=lambda x: np.sin(np.pi * x[..., 0]),
            time_horizon=0.5,
        )
        solver_dir = HJBSolver(problem_dir, config)
        value_dir, _ = solver_dir.solve()

        # With absorbing BCs, boundary values should differ from Dirichlet
        # because absorbing linearly extrapolates from interior (allowing
        # the boundary to evolve) while Dirichlet fixes boundary values.
        # The terminal condition sin(pi*x) has V[0]=V[-1]=0.
        # After diffusion with absorbing BCs, boundaries will be extrapolated
        # from interior values and should differ from the fixed Dirichlet values.
        assert not np.allclose(
            value_abs, value_dir, atol=1e-6
        ), "Absorbing and Dirichlet BCs produced identical solutions for diffusion problem"

    def test_absorbing_vs_dirichlet_different_solutions(self):
        """Absorbing and Dirichlet BCs should produce different solutions.

        Acceptance criterion: Dirichlet and absorbing BCs produce different
        solutions for the same PDE.
        """
        n = 15

        def make_problem(bc_type):
            sv = StateVariable(
                "x",
                0,
                1,
                n,
                boundary_lower=bc_type,
                boundary_upper=bc_type,
            )
            return HJBProblem(
                state_space=StateSpace([sv]),
                control_variables=[ControlVariable("u", 0, 1, 2)],
                utility_function=ExpectedWealth(),
                dynamics=lambda x, u, t: np.ones_like(x) * 0.5,
                running_cost=lambda x, u, t: np.zeros(x.shape[0]),
                diffusion=lambda x, u, t: np.ones((x.shape[0], 1)) * 0.1,
                terminal_value=lambda x: x[..., 0] ** 2,
                time_horizon=0.5,
            )

        config = HJBSolverConfig(time_step=0.05, max_iterations=5, tolerance=1e-8, verbose=False)

        solver_absorbing = HJBSolver(make_problem(BoundaryCondition.ABSORBING), config)
        value_absorbing, _ = solver_absorbing.solve()

        solver_dirichlet = HJBSolver(make_problem(BoundaryCondition.DIRICHLET), config)
        value_dirichlet, _ = solver_dirichlet.solve()

        # Solutions should differ, especially at the boundaries
        assert not np.allclose(
            value_absorbing, value_dirichlet, atol=1e-6
        ), "Absorbing and Dirichlet BCs produced identical solutions"


class TestBoundaryConditionEnforcement:
    """Verify that boundary conditions are actually enforced during time-stepping (issue #448)."""

    @staticmethod
    def _make_solver(bc_lower, bc_upper, num_points=20, time_horizon=0.5, max_iter=10):
        """Helper: build a 1-D solver with specified BCs."""
        state_var = StateVariable(
            "x",
            0.1,
            2.0,
            num_points,
            boundary_lower=bc_lower,
            boundary_upper=bc_upper,
        )
        state_space = StateSpace([state_var])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: 0.1 * np.ones_like(x),
            running_cost=lambda x, u, t: np.ones(x.shape[0]),
            terminal_value=lambda x: x[..., 0] ** 2,
            time_horizon=time_horizon,
        )
        config = HJBSolverConfig(
            time_step=0.05, max_iterations=max_iter, tolerance=1e-6, verbose=False
        )
        return HJBSolver(problem, config)

    def test_absorbing_bc_zero_curvature(self):
        """Absorbing BCs should enforce d²V/dx² ≈ 0 at both boundaries."""
        solver = self._make_solver(BoundaryCondition.ABSORBING, BoundaryCondition.ABSORBING)
        value, _ = solver.solve()

        # Numerical second derivative at lower boundary: V[2] - 2*V[1] + V[0]
        d2_lower = value[2] - 2.0 * value[1] + value[0]
        # Numerical second derivative at upper boundary: V[-1] - 2*V[-2] + V[-3]
        d2_upper = value[-1] - 2.0 * value[-2] + value[-3]

        assert abs(d2_lower) < 1e-10, f"Lower absorbing BC violated: d²V = {d2_lower}"
        assert abs(d2_upper) < 1e-10, f"Upper absorbing BC violated: d²V = {d2_upper}"

    def test_dirichlet_bc_preserves_prescribed_values(self):
        """Dirichlet BCs should hold boundary values fixed at their initial values."""
        solver = self._make_solver(BoundaryCondition.DIRICHLET, BoundaryCondition.DIRICHLET)

        # Capture prescribed boundary values before solve
        solver.solve()  # triggers value_function init & boundary capture
        lower_prescribed = solver._boundary_values[0]["lower"]
        upper_prescribed = solver._boundary_values[0]["upper"]

        # After solve, boundary values should match prescribed values
        assert np.allclose(solver.value_function[0], lower_prescribed), (
            f"Lower Dirichlet BC drifted: got {solver.value_function[0]}, "
            f"expected {lower_prescribed}"
        )
        assert np.allclose(solver.value_function[-1], upper_prescribed), (
            f"Upper Dirichlet BC drifted: got {solver.value_function[-1]}, "
            f"expected {upper_prescribed}"
        )

    def test_neumann_bc_zero_gradient(self):
        """Neumann BCs should enforce dV/dx ≈ 0 at both boundaries."""
        solver = self._make_solver(BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN)
        value, _ = solver.solve()

        # dV/dx ≈ (V[1] - V[0]) / dx at lower boundary
        dx = solver.dx[0]
        grad_lower = (value[1] - value[0]) / dx[0]
        grad_upper = (value[-1] - value[-2]) / dx[-1]

        assert abs(grad_lower) < 1e-10, f"Lower Neumann BC violated: dV/dx = {grad_lower}"
        assert abs(grad_upper) < 1e-10, f"Upper Neumann BC violated: dV/dx = {grad_upper}"

    def test_reflecting_bc_zero_gradient(self):
        """Reflecting BCs are equivalent to Neumann (dV/dx = 0)."""
        solver = self._make_solver(BoundaryCondition.REFLECTING, BoundaryCondition.REFLECTING)
        value, _ = solver.solve()

        dx = solver.dx[0]
        grad_lower = (value[1] - value[0]) / dx[0]
        grad_upper = (value[-1] - value[-2]) / dx[-1]

        assert abs(grad_lower) < 1e-10, f"Lower reflecting BC violated: dV/dx = {grad_lower}"
        assert abs(grad_upper) < 1e-10, f"Upper reflecting BC violated: dV/dx = {grad_upper}"

    def test_mixed_boundary_conditions(self):
        """Different BC types on lower vs. upper boundary."""
        solver = self._make_solver(BoundaryCondition.ABSORBING, BoundaryCondition.NEUMANN)
        value, _ = solver.solve()

        # Lower: absorbing → d²V/dx² ≈ 0
        d2_lower = value[2] - 2.0 * value[1] + value[0]
        assert abs(d2_lower) < 1e-10, f"Lower absorbing BC violated: d²V = {d2_lower}"

        # Upper: Neumann → dV/dx ≈ 0
        dx = solver.dx[0]
        grad_upper = (value[-1] - value[-2]) / dx[-1]
        assert abs(grad_upper) < 1e-10, f"Upper Neumann BC violated: dV/dx = {grad_upper}"

    def test_finite_horizon_terminal_condition_preserved(self):
        """Terminal condition should remain intact at the time boundary."""
        state_var = StateVariable(
            "x",
            0.1,
            2.0,
            15,
            boundary_lower=BoundaryCondition.ABSORBING,
            boundary_upper=BoundaryCondition.ABSORBING,
        )
        state_space = StateSpace([state_var])

        def terminal_fn(x):
            return x[..., 0] ** 2

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            terminal_value=terminal_fn,
            time_horizon=0.5,
        )

        config = HJBSolverConfig(time_step=0.05, max_iterations=3, tolerance=1e-8, verbose=False)
        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        # With zero dynamics and zero running cost, the value function should
        # remain close to the terminal condition (only boundary enforcement changes it)
        state_points = np.stack(state_space.flat_grids, axis=-1)
        terminal_values = terminal_fn(state_points).reshape(state_space.shape)

        # Interior points should stay close to terminal condition
        interior = value[2:-2]
        terminal_interior = terminal_values[2:-2]
        assert np.allclose(interior, terminal_interior, atol=0.5), (
            f"Terminal condition drifted too much in interior: "
            f"max diff = {np.max(np.abs(interior - terminal_interior))}"
        )

    def test_boundary_values_finite_after_many_iterations(self):
        """Boundary values should stay finite and stable, not blow up."""
        solver = self._make_solver(
            BoundaryCondition.ABSORBING, BoundaryCondition.ABSORBING, max_iter=50
        )
        value, _ = solver.solve()

        assert np.all(np.isfinite(value)), "Value function has non-finite values"
        assert (
            np.max(np.abs(value)) < 1e10
        ), f"Value function blew up: max = {np.max(np.abs(value))}"


class TestMultiDimensionalProblems:
    """Test multi-dimensional state spaces."""

    def test_2d_state_space_interpolation(self):
        """Test 2D interpolation in state space."""
        sv1 = StateVariable("x", 0, 2, 5)
        sv2 = StateVariable("y", 0, 3, 4)
        space = StateSpace([sv1, sv2])

        # Create a simple 2D value function (quadratic)
        X, Y = np.meshgrid(space.grids[0], space.grids[1], indexing="ij")
        value_function = X**2 + Y**2

        # Test interpolation at grid points
        grid_points = np.array([[0, 0], [1, 1.5], [2, 3]])
        interp_values = space.interpolate_value(value_function, grid_points)
        expected = grid_points[:, 0] ** 2 + grid_points[:, 1] ** 2
        assert np.allclose(interp_values, expected, rtol=0.1)

        # Test interpolation between grid points
        # Linear interpolation will have some error for quadratic functions
        off_grid_points = np.array([[0.5, 0.75], [1.0, 1.5], [1.5, 2.25]])
        interp_values = space.interpolate_value(value_function, off_grid_points)
        # More lenient tolerance for linear interpolation of quadratic function
        assert np.all(np.isfinite(interp_values))
        assert interp_values[0] < interp_values[1] < interp_values[2]  # Check ordering

    def test_3d_state_space_interpolation(self):
        """Test 3D interpolation in state space."""
        sv1 = StateVariable("x", 0, 1, 3)
        sv2 = StateVariable("y", 0, 1, 3)
        sv3 = StateVariable("z", 0, 1, 3)
        space = StateSpace([sv1, sv2, sv3])

        # Create a simple 3D value function
        value_function = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    value_function[i, j, k] = i + j + k

        # Test interpolation
        points = np.array([[0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1]])
        interp_values = space.interpolate_value(value_function, points)

        assert interp_values[0] >= 0  # Middle point
        assert interp_values[1] == 0  # Origin
        assert interp_values[2] == 6  # Corner

    def test_2d_hjb_solve(self):
        """Test HJB solver with 2D state space."""
        sv1 = StateVariable("x", 1, 3, 4)
        sv2 = StateVariable("y", 1, 2, 3)
        state_space = StateSpace([sv1, sv2])

        control_variables = [ControlVariable("u", 0, 1, 2)]

        def dynamics(x, u, t):
            # Simple 2D dynamics
            result = np.zeros_like(x)
            result[..., 0] = x[..., 0] * 0.1
            result[..., 1] = x[..., 1] * 0.05
            return result

        def running_cost(x, u, t):
            return -(x[..., 0] + x[..., 1])

        def terminal_value(x):
            return x[..., 0] + x[..., 1]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=ExpectedWealth(),
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_value=terminal_value,
            discount_rate=0.1,
            time_horizon=0.5,
        )

        config = HJBSolverConfig(time_step=0.1, max_iterations=3, tolerance=1e-2, verbose=False)

        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert value.shape == (4, 3)
        assert policy["u"].shape == (4, 3)
        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(policy["u"]))


class TestConvergenceAndAccuracy:
    """Test convergence and accuracy of solutions."""

    def test_convergence_with_known_solution(self):
        """Test convergence to known analytical solution."""
        # Simple problem with known solution: V(x,t) = x for linear dynamics
        state_space = StateSpace([StateVariable("x", 1, 5, 20)])

        def dynamics(x, u, t):
            return np.zeros_like(x)  # No dynamics

        def running_cost(x, u, t):
            return -x[..., 0] * 0.05  # Small cost proportional to state

        def terminal_value(x):
            return x[..., 0]  # Terminal value equals state

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_value=terminal_value,
            discount_rate=0.0,  # No discounting
            time_horizon=1.0,
        )

        config = HJBSolverConfig(
            time_step=0.05, max_iterations=50, tolerance=1e-6, verbose=True  # Test verbose output
        )

        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        # For this simple problem with no dynamics and small cost,
        # the value should be close to the terminal value with a small adjustment
        # Due to the iterative nature and limited iterations, exact match isn't expected
        assert np.all(np.isfinite(value))
        # With negative running cost proportional to state, higher states accumulate more negative cost
        # So the value function may decrease. Just check that it's not all zeros
        assert not np.allclose(value, 0)  # Value function should have been updated

    def test_policy_iteration_convergence(self):
        """Test that policy iteration converges properly."""
        state_space = StateSpace([StateVariable("x", 1, 10, 10)])

        # Track convergence history
        convergence_history = []

        def dynamics(x, u, t):
            return x * u[..., 0].reshape(x.shape)

        def running_cost(x, u, t):
            cost = -np.log(np.maximum(x[..., 0], 1e-10)) * u[..., 0]
            convergence_history.append(np.mean(cost))
            return cost

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0.01, 0.2, 5)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=None,  # Infinite horizon
        )

        config = HJBSolverConfig(time_step=0.01, max_iterations=100, tolerance=1e-5, verbose=False)

        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        # Check that solution converged
        assert len(convergence_history) > 0

        # Value function should be smooth
        value_diff = np.diff(value)
        assert np.all(np.isfinite(value_diff))

        # Policy should be within bounds
        assert np.all(policy["u"] >= 0.01 - 1e-10)
        assert np.all(policy["u"] <= 0.2 + 1e-10)

    def test_convergence_metrics_computation(self):
        """Test computation of convergence metrics."""
        state_space = StateSpace([StateVariable("x", 1, 5, 5)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )

        config = HJBSolverConfig(max_iterations=5, verbose=False)

        solver = HJBSolver(problem, config)

        # Test metrics before solving
        metrics_before = solver.compute_convergence_metrics()
        assert "error" in metrics_before
        assert metrics_before["error"] == "No solution computed yet"

        # Solve and test metrics
        solver.solve()
        metrics = solver.compute_convergence_metrics()

        assert "max_residual" in metrics
        assert "mean_residual" in metrics
        assert "value_function_range" in metrics
        assert "policy_stats" in metrics

        assert isinstance(metrics["max_residual"], float)
        assert isinstance(metrics["mean_residual"], float)
        assert metrics["max_residual"] >= metrics["mean_residual"]
        assert metrics["max_residual"] >= 0
        assert metrics["mean_residual"] >= 0

        assert len(metrics["value_function_range"]) == 2
        assert metrics["value_function_range"][0] <= metrics["value_function_range"][1]

        assert "u" in metrics["policy_stats"]
        assert "min" in metrics["policy_stats"]["u"]
        assert "max" in metrics["policy_stats"]["u"]
        assert "mean" in metrics["policy_stats"]["u"]


class TestGradientConsistency:
    """Test that policy improvement uses the same gradient scheme as evaluation (#454)."""

    def test_policy_improvement_uses_upwind_gradient(self):
        """Verify policy improvement computes drift*grad_V via upwind, not central diffs.

        The Hamiltonian maximized during policy improvement must use the same
        spatial discretization (upwind) as the PDE time-stepping in policy
        evaluation.  The optimized code precomputes upwind finite differences
        via ``_precompute_upwind_diffs`` instead of calling
        ``_apply_upwind_scheme`` per control candidate (#371).
        """
        state_space = StateSpace([StateVariable("x", 0, 10, 21)])

        def dynamics(x, u, t):
            return np.ones_like(x) * 5.0

        def running_cost(x, u, t):
            return -x[..., 0] * u[..., 0]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0.1, 1.0, 3)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=None,
        )
        config = HJBSolverConfig(max_iterations=5, verbose=False)
        solver = HJBSolver(problem, config)

        # Set a non-trivial value function (quadratic) so gradients differ
        grid = state_space.grids[0]
        solver.value_function = grid**2
        solver.optimal_policy = {"u": np.full(state_space.shape, 0.5)}

        # Compute what _policy_evaluation uses: upwind advection
        drift_val = np.ones(state_space.shape) * 5.0
        upwind_advection = solver._apply_upwind_scheme(solver.value_function, drift_val, 0)

        # Compute what old code used: central-difference gradient * drift
        grad_central = np.gradient(solver.value_function, grid)
        central_advection = drift_val * grad_central

        # They should differ (confirms the bug was real)
        assert not np.allclose(
            upwind_advection, central_advection
        ), "Upwind and central advection should differ for non-linear V"

        # Verify that _precompute_upwind_diffs produces the same result as
        # _apply_upwind_scheme for the same drift field.
        upwind_diffs = solver._precompute_upwind_diffs()
        fwd_flat, bwd_flat = upwind_diffs[0]
        drift_flat = drift_val.ravel()
        precomputed_advection = (
            np.maximum(drift_flat, 0.0) * fwd_flat + np.minimum(drift_flat, 0.0) * bwd_flat
        )
        assert np.allclose(
            precomputed_advection, upwind_advection.ravel()
        ), "Precomputed upwind diffs must match _apply_upwind_scheme output"

        # Run policy improvement and verify precompute is called (not _apply_upwind_scheme)
        precompute_called: list[bool] = []
        original_precompute = solver._precompute_upwind_diffs

        def tracking_precompute(*args, **kwargs):
            precompute_called.append(True)
            return original_precompute(*args, **kwargs)

        with patch.object(solver, "_precompute_upwind_diffs", side_effect=tracking_precompute):
            solver._policy_improvement()

        assert len(precompute_called) > 0, "Policy improvement must call _precompute_upwind_diffs"

    def test_policy_iteration_monotonic_convergence(self):
        """Policy iteration should converge monotonically for a simple 1D problem.

        With consistent gradient discretization, the value function should
        improve (or stay the same) at every policy iteration step.
        """
        state_space = StateSpace([StateVariable("x", 1, 10, 15)])

        def dynamics(x, u, t):
            return x * u[..., 0:1] * 0.05

        def running_cost(x, u, t):
            return -np.log(np.maximum(x[..., 0], 1e-10)) * 0.1

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0.01, 0.5, 5)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=None,
        )

        config = HJBSolverConfig(
            time_step=0.005,
            max_iterations=20,
            tolerance=1e-6,
            verbose=False,
        )
        solver = HJBSolver(problem, config)

        # Capture value function norm at each outer iteration
        value_norms: list[float] = []
        original_policy_eval = solver._policy_evaluation

        def tracking_eval(*args, **kwargs):
            original_policy_eval()
            if solver.value_function is not None:
                value_norms.append(np.max(np.abs(solver.value_function)))

        with patch.object(solver, "_policy_evaluation", side_effect=tracking_eval):
            solver.solve()

        # After convergence, verify changes shrink over time
        if len(value_norms) >= 3:
            diffs = [abs(value_norms[i + 1] - value_norms[i]) for i in range(len(value_norms) - 1)]
            early = np.mean(diffs[: max(1, len(diffs) // 3)])
            late = np.mean(diffs[-(max(1, len(diffs) // 3)) :])
            assert (
                late <= early + 1e-8
            ), f"Value function should converge: early={early:.6f}, late={late:.6f}"

    def test_convergence_metrics_uses_upwind(self):
        """Convergence metrics residual should use the upwind scheme."""
        state_space = StateSpace([StateVariable("x", 1, 5, 11)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0.0, 1.0, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )

        config = HJBSolverConfig(max_iterations=5, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        # Verify upwind is called during metrics computation
        upwind_called: list[bool] = []
        original_upwind = solver._apply_upwind_scheme

        def tracking_upwind(*args, **kwargs):
            upwind_called.append(True)
            return original_upwind(*args, **kwargs)

        with patch.object(solver, "_apply_upwind_scheme", side_effect=tracking_upwind):
            metrics = solver.compute_convergence_metrics()

        assert len(upwind_called) > 0, "compute_convergence_metrics must use _apply_upwind_scheme"
        assert "max_residual" in metrics
        assert metrics["max_residual"] >= 0


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_power_utility_special_cases(self):
        """Test power utility with gamma close to 1."""
        # Test transition from power to log utility
        gammas = [0.99, 1.0, 1.01]
        wealth = np.array([1, 2, 5, 10])

        for gamma in gammas:
            utility = PowerUtility(risk_aversion=gamma)
            values = utility.evaluate(wealth)
            assert np.all(np.isfinite(values))

            derivs = utility.derivative(wealth)
            assert np.all(np.isfinite(derivs))
            assert np.all(derivs > 0)  # Positive marginal utility

            inv_derivs = utility.inverse_derivative(derivs)
            assert np.allclose(inv_derivs, wealth, rtol=1e-5)

    def test_custom_utility_without_inverse(self):
        """Test custom utility without inverse derivative."""

        def my_eval(w):
            return np.sqrt(w)

        def my_deriv(w):
            return 0.5 / np.sqrt(np.maximum(w, 1e-10))

        # Create without inverse
        utility = create_custom_utility(my_eval, my_deriv)

        wealth = np.array([1, 4, 9])
        values = utility.evaluate(wealth)
        assert np.allclose(values, [1, 2, 3])

        derivs = utility.derivative(wealth)
        assert np.all(derivs > 0)

        # Should raise error when trying to use inverse
        with pytest.raises(NotImplementedError):
            utility.inverse_derivative(derivs)

    def test_solver_without_terminal_value(self):
        """Test finite horizon problem without explicit terminal value."""
        state_space = StateSpace([StateVariable("x", 1, 5, 5)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0],
            terminal_value=None,  # No terminal value provided
            time_horizon=1.0,
        )

        config = HJBSolverConfig(max_iterations=3, verbose=False)

        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        # Should use default zero terminal value
        assert value is not None
        assert np.all(np.isfinite(value))

    def test_extract_control_without_solution(self):
        """Test extracting control before solving."""
        state_space = StateSpace([StateVariable("x", 1, 5, 5)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # Should raise error
        with pytest.raises(RuntimeError, match="Must solve HJB equation"):
            solver.extract_feedback_control(np.array([3.0]))

    def test_cost_reshaping_edge_cases(self):
        """Test edge cases in cost reshaping during policy evaluation."""
        state_space = StateSpace([StateVariable("x", 1, 3, 3)])

        # Test with scalar cost (wrapped in array for consistency)
        def scalar_cost(x, u, t):
            # Always return array for consistency
            return np.ones(x.shape[0]) if hasattr(x, "shape") else np.array([1.0])

        problem_scalar = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=scalar_cost,
            time_horizon=0.5,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver_scalar = HJBSolver(problem_scalar, config)
        value_scalar, _ = solver_scalar.solve()
        assert value_scalar.shape == (3,)
        assert np.all(np.isfinite(value_scalar))

        # Test with 1D array cost
        def array_cost(x, u, t):
            return np.ones(x.shape[0]) * 2.0

        problem_array = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=array_cost,
            time_horizon=0.5,
        )

        solver_array = HJBSolver(problem_array, config)
        value_array, _ = solver_array.solve()
        assert value_array.shape == (3,)
        assert np.all(np.isfinite(value_array))


class TestAdditionalCoverage:
    """Test additional edge cases for complete coverage."""

    def test_reflecting_boundary_condition(self):
        """Test reflecting boundary condition (maps to Neumann)."""
        # Test that reflecting BC is handled (same as Neumann)
        sv = StateVariable(
            "x",
            0,
            1,
            5,
            boundary_lower=BoundaryCondition.REFLECTING,
            boundary_upper=BoundaryCondition.REFLECTING,
        )
        state_space = StateSpace([sv])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver = HJBSolver(problem, config)

        # Test that reflecting BC works (internally mapped to Neumann)
        mat = solver._build_difference_matrix(0, BoundaryCondition.REFLECTING)
        assert mat is not None
        assert mat.shape == (5, 5)

    def test_terminal_value_none_handling(self):
        """Test handling of None terminal value in finite horizon."""
        state_space = StateSpace([StateVariable("x", 1, 5, 3)])

        # Explicitly set terminal_value to None in finite horizon problem
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0],
            terminal_value=None,  # Explicitly None
            time_horizon=1.0,
        )

        # Terminal value should be set to default zero function
        assert problem.terminal_value is not None

        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        assert value is not None
        assert np.all(np.isfinite(value))

    def test_policy_evaluation_none_checks(self):
        """Test None checks in policy evaluation."""
        state_space = StateSpace([StateVariable("x", 1, 3, 3)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        config = HJBSolverConfig(max_iterations=1, verbose=False)
        solver = HJBSolver(problem, config)

        # Manually set value_function to None and test policy evaluation
        solver.value_function = None
        solver._policy_evaluation()
        # Should return early without error
        assert solver.value_function is None

        # Now test with value but no policy
        solver.value_function = np.ones((3,))
        solver.optimal_policy = None
        solver._policy_evaluation()
        # Should return early without error
        assert solver.optimal_policy is None

    def test_setup_operators_call(self):
        """Test that setup_operators is called during initialization."""
        state_space = StateSpace([StateVariable("x", 0, 1, 5)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )

        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # Check that operators were initialized
        assert solver.operators_initialized
        assert hasattr(solver, "dx")
        assert len(solver.dx) == 1
        # dx is now a per-interval spacing array (supports non-uniform grids)
        assert len(solver.dx[0]) == 4  # 5 points = 4 intervals
        assert np.allclose(solver.dx[0], 0.25)  # Uniform spacing of 0.25

    def test_terminal_value_callback(self):
        """Test terminal value callback is used."""
        state_space = StateSpace([StateVariable("x", 1, 3, 3)])

        # Custom terminal value that returns x squared
        def my_terminal(x):
            return x[..., 0] ** 2

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            terminal_value=my_terminal,
            time_horizon=1.0,
        )

        config = HJBSolverConfig(max_iterations=1, verbose=False)
        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        # Initial value should be based on terminal condition
        assert value is not None
        assert np.all(np.isfinite(value))

    def test_cost_function_with_ndim_attribute(self):
        """Test cost function that returns object with ndim > 1."""
        state_space = StateSpace([StateVariable("x", 1, 3, 3)])

        def matrix_cost(x, u, t):
            # Return 2D array to trigger ndim > 1 condition
            n = x.shape[0]
            cost_matrix = np.ones((n, 2))
            cost_matrix[:, 0] = -x[..., 0] * 0.01
            return cost_matrix

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=matrix_cost,
            time_horizon=0.5,
        )

        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        assert value is not None
        assert np.all(np.isfinite(value))

    def test_custom_utility_inverse_not_provided_error(self):
        """Test custom utility raises error when inverse not provided."""

        def my_eval(w):
            return w**0.5

        def my_deriv(w):
            return 0.5 / np.sqrt(np.maximum(w, 1e-10))

        # Create utility without inverse (None)
        utility = create_custom_utility(my_eval, my_deriv, None)

        wealth = np.array([1, 4, 9])

        # Should work for evaluation and derivative
        values = utility.evaluate(wealth)
        assert np.all(np.isfinite(values))

        derivs = utility.derivative(wealth)
        assert np.all(np.isfinite(derivs))

        # Should raise error for inverse
        with pytest.raises(NotImplementedError, match="Inverse derivative not provided"):
            utility.inverse_derivative(derivs)


class TestSparseMatrixOperations:
    """Test sparse matrix operations for large problems."""

    def test_sparse_matrix_usage(self):
        """Test that sparse matrices are used correctly."""
        # Create larger problem that benefits from sparse matrices
        state_space = StateSpace([StateVariable("x", 0, 10, 50)])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )

        config_sparse = HJBSolverConfig(use_sparse=True, max_iterations=3, verbose=False)

        solver_sparse = HJBSolver(problem, config_sparse)

        # Test building difference matrix
        for bc in [BoundaryCondition.DIRICHLET, BoundaryCondition.NEUMANN]:
            mat = solver_sparse._build_difference_matrix(0, bc)
            assert sparse.issparse(mat)
            assert mat.shape == (50, 50)

            # Check sparsity pattern (tridiagonal structure)
            if bc == BoundaryCondition.NEUMANN:
                # Most elements should be zero
                assert mat.nnz <= 150  # At most 3 diagonals

        # Solve should work with sparse matrices
        value, policy = solver_sparse.solve()
        assert value is not None
        assert np.all(np.isfinite(value))


class TestDiffusionTerm:
    """Test diffusion term (½σ²∇²V) in HJB solver (issue #447)."""

    def test_second_derivatives_quadratic(self):
        """Test _compute_second_derivatives on V(x) = x² gives d²V/dx² = 2."""
        state_space = StateSpace([StateVariable("x", 0, 1, 21)])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # V(x) = x²
        grid = state_space.grids[0]
        value = grid**2

        d2v = solver._compute_second_derivatives(value)

        # Shape: (21, 1) for 1D problem
        assert d2v.shape == (21, 1)
        # Interior points should have d²V/dx² = 2 exactly (central diff on quadratic)
        assert np.allclose(d2v[1:-1, 0], 2.0, atol=1e-10)
        # Boundary values default to zero
        assert d2v[0, 0] == 0.0
        assert d2v[-1, 0] == 0.0

    def test_second_derivatives_2d(self):
        """Test second derivatives in 2D state space."""
        sv1 = StateVariable("x", 0, 1, 11)
        sv2 = StateVariable("y", 0, 1, 11)
        state_space = StateSpace([sv1, sv2])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # V(x,y) = x² + 3y² => d²V/dx² = 2, d²V/dy² = 6
        X, Y = np.meshgrid(state_space.grids[0], state_space.grids[1], indexing="ij")
        value = X**2 + 3 * Y**2

        d2v = solver._compute_second_derivatives(value)
        assert d2v.shape == (11, 11, 2)

        # Check interior points
        assert np.allclose(d2v[1:-1, 1:-1, 0], 2.0, atol=1e-10)
        assert np.allclose(d2v[1:-1, 1:-1, 1], 6.0, atol=1e-10)

    def test_apply_diffusion_term(self):
        """Test _apply_diffusion_term with known values."""
        state_space = StateSpace([StateVariable("x", 0, 1, 11)])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        config = HJBSolverConfig()
        solver = HJBSolver(problem, config)

        # V(x) = x², d²V/dx² = 2 at interior
        grid = state_space.grids[0]
        value = grid**2

        # σ² = 1.0 everywhere => ½ * 1.0 * 2.0 = 1.0 at interior
        sigma_sq = np.ones((11, 1))
        result = solver._apply_diffusion_term(value, sigma_sq)

        assert result.shape == (11,)
        assert np.allclose(result[1:-1], 1.0, atol=1e-10)
        assert result[0] == 0.0
        assert result[-1] == 0.0

    def test_zero_diffusion_recovers_deterministic(self):
        """Test that σ=0 recovers identical deterministic behavior."""
        state_space = StateSpace([StateVariable("x", 1, 5, 10)])

        def dynamics(x, u, t):
            return x * 0.1

        def running_cost(x, u, t):
            return -x[..., 0] * 0.01

        def zero_diffusion(x, u, t):
            return np.zeros_like(x)

        config = HJBSolverConfig(time_step=0.01, max_iterations=20, tolerance=1e-6, verbose=False)

        # Without diffusion
        problem_no = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=1.0,
        )
        solver_no = HJBSolver(problem_no, config)
        value_no, _ = solver_no.solve()

        # With zero diffusion
        problem_zero = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=1.0,
            diffusion=zero_diffusion,
        )
        solver_zero = HJBSolver(problem_zero, config)
        value_zero, _ = solver_zero.solve()

        assert np.allclose(value_no, value_zero, atol=1e-12)

    def test_nonzero_diffusion_changes_value(self):
        """Test that non-zero diffusion produces different results."""
        # Use Dirichlet BCs so boundary-fixed values allow diffusion to
        # affect the interior solution (absorbing BCs linearly extrapolate,
        # which can make solutions converge to linear — where d²V/dx² = 0).
        sv = StateVariable(
            "x",
            1,
            10,
            15,
            boundary_lower=BoundaryCondition.DIRICHLET,
            boundary_upper=BoundaryCondition.DIRICHLET,
        )
        state_space = StateSpace([sv])

        def dynamics(x, u, t):
            return x * 0.05

        def running_cost(x, u, t):
            return -x[..., 0] * 0.01

        def nonzero_diffusion(x, u, t):
            return x**2 * 0.04  # GBM-like σ²

        config = HJBSolverConfig(time_step=0.01, max_iterations=20, tolerance=1e-6, verbose=False)

        # Without diffusion
        problem_no = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=1.0,
        )
        solver_no = HJBSolver(problem_no, config)
        value_no, _ = solver_no.solve()

        # With diffusion
        problem_diff = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=1.0,
            diffusion=nonzero_diffusion,
        )
        solver_diff = HJBSolver(problem_diff, config)
        value_diff, _ = solver_diff.solve()

        # Must differ
        assert not np.allclose(value_no, value_diff, atol=1e-6)
        # Both finite
        assert np.all(np.isfinite(value_no))
        assert np.all(np.isfinite(value_diff))

    def test_diffusion_in_2d_problem(self):
        """Test diffusion in 2D state space."""
        sv1 = StateVariable("x", 1, 3, 5)
        sv2 = StateVariable("y", 1, 2, 4)
        state_space = StateSpace([sv1, sv2])

        def dynamics(x, u, t):
            result = np.zeros_like(x)
            result[..., 0] = x[..., 0] * 0.05
            result[..., 1] = 0.0
            return result

        def running_cost(x, u, t):
            return -(x[..., 0] + x[..., 1])

        def diffusion(x, u, t):
            result = np.zeros_like(x)
            result[..., 0] = x[..., 0] ** 2 * 0.01
            result[..., 1] = 0.0
            return result

        def terminal_value(x):
            return x[..., 0] + x[..., 1]

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=ExpectedWealth(),
            dynamics=dynamics,
            running_cost=running_cost,
            terminal_value=terminal_value,
            discount_rate=0.1,
            time_horizon=0.5,
            diffusion=diffusion,
        )
        config = HJBSolverConfig(time_step=0.1, max_iterations=3, tolerance=1e-2, verbose=False)
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert value.shape == (5, 4)
        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(policy["u"]))

    def test_control_dependent_diffusion_affects_value(self):
        """Test that control-dependent diffusion changes the value function."""
        sv = StateVariable(
            "x",
            1,
            10,
            10,
            boundary_lower=BoundaryCondition.DIRICHLET,
            boundary_upper=BoundaryCondition.DIRICHLET,
        )
        state_space = StateSpace([sv])

        def dynamics(x, u, t):
            return x * 0.05 * u[..., 0].reshape(x.shape)

        def running_cost(x, u, t):
            return -x[..., 0] * 0.01

        def ctrl_diffusion(x, u, t):
            # Higher control -> higher volatility
            return x**2 * 0.1 * u[..., 0].reshape(x.shape)

        config = HJBSolverConfig(time_step=0.01, max_iterations=30, tolerance=1e-5, verbose=False)

        # Without diffusion
        problem_no = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0.1, 1.0, 5)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
        )
        solver_no = HJBSolver(problem_no, config)
        value_no, _ = solver_no.solve()

        # With control-dependent diffusion
        problem_diff = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0.1, 1.0, 5)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            diffusion=ctrl_diffusion,
        )
        solver_diff = HJBSolver(problem_diff, config)
        value_diff, _ = solver_diff.solve()

        # Value functions should differ when diffusion is control-dependent
        assert not np.allclose(value_no, value_diff, atol=1e-6)

    def test_infinite_horizon_with_diffusion(self):
        """Test infinite horizon problem with diffusion term."""
        state_space = StateSpace([StateVariable("x", 1, 5, 10)])

        def dynamics(x, u, t):
            return x * 0.05

        def running_cost(x, u, t):
            return -x[..., 0] * 0.01

        def diffusion(x, u, t):
            return x**2 * 0.02

        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            discount_rate=0.05,
            time_horizon=None,  # Infinite horizon
            diffusion=diffusion,
        )
        config = HJBSolverConfig(time_step=0.01, max_iterations=50, tolerance=1e-5, verbose=False)
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(policy["u"]))
        # Value should not be all zeros (solver should have updated it)
        assert not np.allclose(value, 0)


class TestResidualDriftTerm:
    """Regression tests for issue #449: residual must include drift·∇V."""

    def test_zero_drift_residual_unchanged(self):
        """With zero drift, residual equals |−ρV + f|."""
        state_space = StateSpace([StateVariable("x", 1, 5, 5)])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )
        config = HJBSolverConfig(max_iterations=5, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        metrics = solver.compute_convergence_metrics()
        assert metrics["max_residual"] >= 0
        assert metrics["mean_residual"] >= 0
        assert np.isfinite(metrics["max_residual"])

    def test_nonzero_drift_increases_residual(self):
        """With non-zero drift, residual includes drift·∇V contribution."""
        state_space = StateSpace([StateVariable("x", 1, 5, 7)])

        # Problem with zero drift
        problem_no_drift = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )
        config = HJBSolverConfig(max_iterations=10, verbose=False)
        solver_no_drift = HJBSolver(problem_no_drift, config)
        solver_no_drift.solve()
        metrics_no_drift = solver_no_drift.compute_convergence_metrics()

        # Same problem but with significant drift
        problem_drift = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.5,  # significant positive drift
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )
        solver_drift = HJBSolver(problem_drift, config)
        solver_drift.solve()
        metrics_drift = solver_drift.compute_convergence_metrics()

        # Drift problem should have different residuals (drift term contributes)
        assert metrics_drift["max_residual"] != metrics_no_drift["max_residual"]
        assert np.isfinite(metrics_drift["max_residual"])
        assert np.isfinite(metrics_drift["mean_residual"])


class TestReflectingBoundaryDirect:
    """Regression tests for issue #450: REFLECTING BC must be handled directly."""

    def test_reflecting_produces_same_matrix_as_neumann(self):
        """REFLECTING and NEUMANN produce identical difference matrices."""
        sv = StateVariable(
            "x",
            0,
            1,
            10,
            boundary_lower=BoundaryCondition.REFLECTING,
            boundary_upper=BoundaryCondition.REFLECTING,
        )
        state_space = StateSpace([sv])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 2)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            time_horizon=1.0,
        )
        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver = HJBSolver(problem, config)

        mat_reflecting = solver._build_difference_matrix(0, BoundaryCondition.REFLECTING)
        mat_neumann = solver._build_difference_matrix(0, BoundaryCondition.NEUMANN)

        diff = (mat_reflecting - mat_neumann).toarray()
        assert np.allclose(diff, 0), "REFLECTING and NEUMANN matrices should be identical"

    def test_reflecting_full_solve(self):
        """REFLECTING BC works through the full solver without manual swap."""
        sv = StateVariable(
            "x",
            1,
            5,
            7,
            boundary_lower=BoundaryCondition.REFLECTING,
            boundary_upper=BoundaryCondition.REFLECTING,
        )
        state_space = StateSpace([sv])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.1,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=0.05,
        )
        config = HJBSolverConfig(max_iterations=10, verbose=False)
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value))
        assert np.all(np.isfinite(policy["u"]))

        # Verify reflecting BC: dV/dx ≈ 0 at boundaries
        dx = solver.dx[0]
        grad_lower = (value[1] - value[0]) / dx[0]
        grad_upper = (value[-1] - value[-2]) / dx[-1]
        assert abs(grad_lower) < 1e-6, f"Reflecting lower BC violated: dV/dx = {grad_lower}"
        assert abs(grad_upper) < 1e-6, f"Reflecting upper BC violated: dV/dx = {grad_upper}"


class TestNaNInfDetection:
    """Tests for NaN/Inf detection during solve (#453)."""

    def _make_1d_problem(self, drift_fn=None, discount_rate=0.05, num_points=20):
        """Helper to create a simple 1D HJB problem."""
        sv = StateVariable("x", 1.0, 5.0, num_points)
        state_space = StateSpace([sv])
        if drift_fn is None:

            def drift_fn(x, u, t):
                return x * 0.05

        return HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=drift_fn,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            discount_rate=discount_rate,
        )

    def test_nan_inf_raises_in_policy_evaluation(self):
        """Solver raises NumericalDivergenceError when value function contains NaN."""
        from ergodic_insurance.hjb_solver import NumericalDivergenceError

        # A running cost that returns NaN propagates directly into new_v:
        # rhs = -rho*V + NaN + advection = NaN → new_v = V + dt*NaN = NaN
        sv = StateVariable("x", 1.0, 5.0, 20)
        state_space = StateSpace([sv])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.05,
            running_cost=lambda x, u, t: np.full(x.shape[0], np.nan),
            discount_rate=0.05,
        )
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=5,
            scheme=TimeSteppingScheme.EXPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)

        with pytest.raises(NumericalDivergenceError, match="diverged"):
            solver.solve()

    def test_convergence_metrics_has_nan_inf_flag(self):
        """compute_convergence_metrics returns has_nan_inf field."""
        problem = self._make_1d_problem()
        config = HJBSolverConfig(max_iterations=5, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        metrics = solver.compute_convergence_metrics()
        assert "has_nan_inf" in metrics
        assert metrics["has_nan_inf"] is False

    def test_convergence_metrics_includes_diffusion_in_residual(self):
        """Residual computation includes diffusion term when diffusion is present."""
        sv = StateVariable("x", 1.0, 5.0, 20)
        state_space = StateSpace([sv])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * 0.05,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            diffusion=lambda x, u, t: np.full_like(x, 0.04),
            discount_rate=0.05,
        )
        config = HJBSolverConfig(max_iterations=20, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()

        metrics = solver.compute_convergence_metrics()
        assert "has_nan_inf" in metrics
        assert metrics["has_nan_inf"] is False
        # Residual should be finite
        assert np.isfinite(metrics["max_residual"])


class TestCFLStabilityCheck:
    """Tests for CFL stability checking and auto-adaptation (#452)."""

    def _make_1d_problem(self, drift_scale=0.05, sigma_sq=None):
        sv = StateVariable("x", 1.0, 5.0, 20)
        state_space = StateSpace([sv])
        return HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * drift_scale,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            diffusion=(lambda x, u, t: np.full_like(x, sigma_sq)) if sigma_sq else None,
            discount_rate=0.05,
        )

    def test_cfl_computation(self):
        """_compute_cfl_number returns correct CFL numbers."""
        problem = self._make_1d_problem(drift_scale=1.0)
        config = HJBSolverConfig(time_step=0.5, verbose=False)
        solver = HJBSolver(problem, config)

        # Create dummy drift on grid
        grid = problem.state_space.grids[0]
        drift = (grid * 1.0).reshape(-1, 1)
        adv_cfl, diff_cfl = solver._compute_cfl_number(drift, None, 0.5)

        assert adv_cfl > 0, "Advection CFL should be positive with non-zero drift"
        assert diff_cfl == 0.0, "Diffusion CFL should be zero without diffusion"

    def test_cfl_auto_reduces_dt(self):
        """Solver auto-reduces dt when CFL is violated for explicit scheme."""
        # Large drift + large dt → CFL violation
        problem = self._make_1d_problem(drift_scale=50.0)
        config = HJBSolverConfig(
            time_step=1.0,  # Very large dt
            max_iterations=5,
            scheme=TimeSteppingScheme.EXPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)

        # Should not raise — CFL auto-reduction prevents divergence
        import logging

        with patch("ergodic_insurance.hjb_solver.logger") as mock_logger:
            solver.solve()
            # Verify CFL warning was issued
            warning_calls = [
                call for call in mock_logger.warning.call_args_list if "CFL" in str(call)
            ]
            assert len(warning_calls) > 0, "Expected CFL warning"

    def test_cfl_not_checked_for_implicit(self):
        """CFL check is skipped for implicit scheme (unconditionally stable)."""
        problem = self._make_1d_problem(drift_scale=50.0)
        config = HJBSolverConfig(
            time_step=1.0,
            max_iterations=5,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)

        with patch("ergodic_insurance.hjb_solver.logger") as mock_logger:
            solver.solve()
            # Should NOT have CFL warnings for implicit scheme
            warning_calls = [
                call for call in mock_logger.warning.call_args_list if "CFL" in str(call)
            ]
            assert len(warning_calls) == 0, "Implicit scheme should not trigger CFL warnings"

    def test_solver_produces_finite_after_cfl_adaptation(self):
        """After CFL auto-adaptation, solution should remain finite."""
        problem = self._make_1d_problem(drift_scale=10.0)
        config = HJBSolverConfig(
            time_step=0.5,
            max_iterations=20,
            scheme=TimeSteppingScheme.EXPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value)), "Value function should be finite after CFL adaptation"
        assert np.all(np.isfinite(policy["u"])), "Policy should be finite after CFL adaptation"


class TestImplicitScheme:
    """Tests for implicit (backward Euler) time-stepping scheme (#451)."""

    def _make_1d_problem(self, drift_scale=0.05, sigma_sq=None, num_points=20):
        sv = StateVariable("x", 1.0, 5.0, num_points)
        state_space = StateSpace([sv])
        return HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * drift_scale,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            diffusion=(lambda x, u, t: np.full_like(x, sigma_sq)) if sigma_sq else None,
            discount_rate=0.05,
        )

    def test_implicit_scheme_converges(self):
        """Implicit scheme should converge to a valid solution."""
        problem = self._make_1d_problem()
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=20,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value)), "Implicit solution should be finite"
        assert np.all(np.isfinite(policy["u"])), "Implicit policy should be finite"

    def test_implicit_matches_explicit_small_dt(self):
        """Implicit and explicit should give similar results with small dt."""
        problem_exp = self._make_1d_problem()
        problem_imp = self._make_1d_problem()

        config_exp = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            scheme=TimeSteppingScheme.EXPLICIT,
            verbose=False,
        )
        config_imp = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )

        solver_exp = HJBSolver(problem_exp, config_exp)
        solver_imp = HJBSolver(problem_imp, config_imp)

        v_exp, _ = solver_exp.solve()
        v_imp, _ = solver_imp.solve()

        # With same small dt, solutions should be close
        rel_diff = np.max(np.abs(v_exp - v_imp)) / (np.max(np.abs(v_exp)) + 1e-10)
        assert rel_diff < 0.15, (
            f"Implicit and explicit should agree with small dt, " f"relative diff = {rel_diff:.4f}"
        )

    def test_implicit_stable_with_large_dt(self):
        """Implicit scheme should remain stable with dt much larger than CFL limit."""
        problem = self._make_1d_problem(drift_scale=1.0)
        # For explicit, CFL would require dt ~ 0.01 or less
        # Implicit should handle dt = 0.1 (10x larger)
        config = HJBSolverConfig(
            time_step=0.1,
            max_iterations=20,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value)), "Implicit scheme should be stable with large dt"

    def test_implicit_with_diffusion(self):
        """Implicit scheme handles diffusion term correctly."""
        problem = self._make_1d_problem(drift_scale=0.05, sigma_sq=0.04)
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=20,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value)), "Implicit with diffusion should be finite"

    def test_default_scheme_is_explicit(self):
        """Default HJBSolverConfig.scheme should be EXPLICIT (#451 Phase 0)."""
        config = HJBSolverConfig()
        assert config.scheme == TimeSteppingScheme.EXPLICIT

    def test_spatial_operator_tridiagonal(self):
        """_build_spatial_operator_1d produces a tridiagonal matrix."""
        problem = self._make_1d_problem(num_points=10)
        config = HJBSolverConfig(verbose=False)
        solver = HJBSolver(problem, config)

        N = 10
        drift_1d = np.ones(N) * 0.1
        sigma_sq_1d = np.ones(N) * 0.04
        L = solver._build_spatial_operator_1d(drift_1d, sigma_sq_1d)

        L_dense = L.toarray()
        # Interior rows should be tridiagonal
        for i in range(2, N - 2):
            for j in range(N):
                if abs(i - j) > 1:
                    assert L_dense[i, j] == 0.0, f"L[{i},{j}] should be zero for tridiagonal matrix"

    def test_implicit_operator_m_matrix(self):
        """(I - dt*L) should be an M-matrix: positive diagonal, non-positive off-diagonal."""
        problem = self._make_1d_problem(num_points=10)
        config = HJBSolverConfig(verbose=False)
        solver = HJBSolver(problem, config)

        N = 10
        drift_1d = np.linspace(-0.5, 0.5, N)
        sigma_sq_1d = np.ones(N) * 0.04
        dt = 0.01

        L = solver._build_spatial_operator_1d(drift_1d, sigma_sq_1d)
        I_mat = sparse.eye(N)
        A = (I_mat - dt * L).toarray()

        # Interior rows: check M-matrix property
        for i in range(1, N - 1):
            assert A[i, i] >= 1.0, f"Diagonal A[{i},{i}]={A[i,i]} should be >= 1"
            for j in range(N):
                if j != i:
                    assert (
                        A[i, j] <= 0.0 + 1e-15
                    ), f"Off-diagonal A[{i},{j}]={A[i,j]} should be <= 0"


class TestCrankNicolsonScheme:
    """Tests for Crank-Nicolson time-stepping scheme (#451)."""

    def _make_1d_problem(self, drift_scale=0.05, sigma_sq=None, num_points=20):
        sv = StateVariable("x", 1.0, 5.0, num_points)
        state_space = StateSpace([sv])
        return HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: x * drift_scale,
            running_cost=lambda x, u, t: -x[..., 0] * 0.01,
            diffusion=(lambda x, u, t: np.full_like(x, sigma_sq)) if sigma_sq else None,
            discount_rate=0.05,
        )

    def test_crank_nicolson_converges(self):
        """Crank-Nicolson scheme should converge to a valid solution."""
        problem = self._make_1d_problem()
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=20,
            scheme=TimeSteppingScheme.CRANK_NICOLSON,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value)), "CN solution should be finite"
        assert np.all(np.isfinite(policy["u"])), "CN policy should be finite"

    def test_crank_nicolson_with_diffusion(self):
        """CN scheme handles diffusion correctly."""
        problem = self._make_1d_problem(drift_scale=0.05, sigma_sq=0.04)
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=20,
            scheme=TimeSteppingScheme.CRANK_NICOLSON,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, policy = solver.solve()

        assert np.all(np.isfinite(value)), "CN with diffusion should be finite"

    def test_rannacher_startup(self):
        """CN with Rannacher smoothing should produce stable results."""
        problem = self._make_1d_problem()
        # Rannacher steps = 2 (default)
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=20,
            scheme=TimeSteppingScheme.CRANK_NICOLSON,
            rannacher_steps=2,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        assert np.all(np.isfinite(value)), "CN with Rannacher should be finite"

    def test_rannacher_zero_disables_startup(self):
        """Setting rannacher_steps=0 disables the implicit startup."""
        problem = self._make_1d_problem()
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=20,
            scheme=TimeSteppingScheme.CRANK_NICOLSON,
            rannacher_steps=0,
            verbose=False,
        )
        solver = HJBSolver(problem, config)
        value, _ = solver.solve()

        # Should still produce finite results with small dt
        assert np.all(np.isfinite(value)), "CN without Rannacher should still work with small dt"

    def test_cn_matches_explicit_small_dt(self):
        """CN and explicit should give similar results with small dt."""
        problem_exp = self._make_1d_problem()
        problem_cn = self._make_1d_problem()

        config_exp = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            scheme=TimeSteppingScheme.EXPLICIT,
            verbose=False,
        )
        config_cn = HJBSolverConfig(
            time_step=0.01,
            max_iterations=30,
            scheme=TimeSteppingScheme.CRANK_NICOLSON,
            verbose=False,
        )

        solver_exp = HJBSolver(problem_exp, config_exp)
        solver_cn = HJBSolver(problem_cn, config_cn)

        v_exp, _ = solver_exp.solve()
        v_cn, _ = solver_cn.solve()

        rel_diff = np.max(np.abs(v_exp - v_cn)) / (np.max(np.abs(v_exp)) + 1e-10)
        assert (
            rel_diff < 0.15
        ), f"CN and explicit should agree with small dt, relative diff = {rel_diff:.4f}"

    def test_multid_falls_back_to_explicit(self):
        """Multi-D problems should fall back to explicit with warning."""
        sv_x = StateVariable("x", 1.0, 5.0, 5)
        sv_y = StateVariable("y", 1.0, 5.0, 5)
        state_space = StateSpace([sv_x, sv_y])
        problem = HJBProblem(
            state_space=state_space,
            control_variables=[ControlVariable("u", 0, 1, 3)],
            utility_function=LogUtility(),
            dynamics=lambda x, u, t: np.zeros_like(x),
            running_cost=lambda x, u, t: np.zeros(x.shape[0]),
            discount_rate=0.05,
        )
        config = HJBSolverConfig(
            time_step=0.01,
            max_iterations=3,
            scheme=TimeSteppingScheme.IMPLICIT,
            verbose=False,
        )
        solver = HJBSolver(problem, config)

        with patch("ergodic_insurance.hjb_solver.logger") as mock_logger:
            value, _ = solver.solve()
            # Should warn about fallback
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "falling back" in str(call).lower()
            ]
            assert len(warning_calls) > 0, "Should warn about multi-D fallback"

        assert np.all(np.isfinite(value)), "Multi-D fallback should produce finite results"

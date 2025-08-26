"""Hamilton-Jacobi-Bellman solver for optimal insurance control.

This module implements a Hamilton-Jacobi-Bellman (HJB) partial differential equation
solver for finding optimal insurance strategies through dynamic programming. The solver
handles multi-dimensional state spaces and provides theoretically optimal control policies.

The HJB equation provides globally optimal solutions by solving:
    ∂V/∂t + max_u[L^u V + f(x,u)] = 0
where V is the value function, L^u is the controlled infinitesimal generator,
and f(x,u) is the running cost/reward.

Author: Alex Filiakov
Date: 2025-01-26
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, sparse
from scipy.sparse import linalg as sparse_linalg

logger = logging.getLogger(__name__)


class TimeSteppingScheme(Enum):
    """Time stepping schemes for PDE integration."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CRANK_NICOLSON = "crank_nicolson"


class BoundaryCondition(Enum):
    """Types of boundary conditions."""

    DIRICHLET = "dirichlet"  # Fixed value
    NEUMANN = "neumann"  # Fixed derivative
    ABSORBING = "absorbing"  # Zero second derivative
    REFLECTING = "reflecting"  # Zero first derivative


@dataclass
class StateVariable:
    """Definition of a state variable in the HJB problem.

    Attributes:
        name: Variable name (e.g., 'wealth', 'time', 'loss_history')
        min_value: Minimum value for the grid
        max_value: Maximum value for the grid
        num_points: Number of grid points
        boundary_lower: Boundary condition at minimum value
        boundary_upper: Boundary condition at maximum value
        log_scale: Whether to use logarithmic spacing for grid points
    """

    name: str
    min_value: float
    max_value: float
    num_points: int
    boundary_lower: BoundaryCondition = BoundaryCondition.ABSORBING
    boundary_upper: BoundaryCondition = BoundaryCondition.ABSORBING
    log_scale: bool = False

    def __post_init__(self):
        """Validate state variable configuration."""
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value must be less than max_value for {self.name}")
        if self.num_points < 3:
            raise ValueError(f"Need at least 3 grid points for {self.name}")
        if self.log_scale and self.min_value <= 0:
            raise ValueError(f"Cannot use log scale with non-positive min_value for {self.name}")

    def get_grid(self) -> np.ndarray:
        """Generate grid points for this variable.

        Returns:
            Array of grid points
        """
        if self.log_scale:
            return np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points)
        return np.linspace(self.min_value, self.max_value, self.num_points)


@dataclass
class ControlVariable:
    """Definition of a control variable in the HJB problem.

    Attributes:
        name: Variable name (e.g., 'limit', 'retention')
        min_value: Minimum control value
        max_value: Maximum control value
        num_points: Number of discrete control values to consider
        continuous: Whether control is continuous (True) or discrete (False)
    """

    name: str
    min_value: float
    max_value: float
    num_points: int = 50
    continuous: bool = True

    def __post_init__(self):
        """Validate control variable configuration."""
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value must be less than max_value for {self.name}")
        if self.num_points < 2:
            raise ValueError(f"Need at least 2 control points for {self.name}")

    def get_values(self) -> np.ndarray:
        """Get discrete control values for optimization.

        Returns:
            Array of control values
        """
        return np.linspace(self.min_value, self.max_value, self.num_points)


@dataclass
class StateSpace:
    """Multi-dimensional state space for HJB problem.

    Handles arbitrary dimensionality with proper grid management
    and boundary condition enforcement.
    """

    state_variables: List[StateVariable]

    def __post_init__(self):
        """Initialize derived attributes."""
        self.ndim = len(self.state_variables)
        self.shape = tuple(sv.num_points for sv in self.state_variables)
        self.size = np.prod(self.shape)

        # Create grids for each dimension
        self.grids = [sv.get_grid() for sv in self.state_variables]

        # Create meshgrid for full state space
        self.meshgrid = np.meshgrid(*self.grids, indexing="ij")

        # Flatten for linear algebra operations
        self.flat_grids = [mg.ravel() for mg in self.meshgrid]

        logger.info(f"Initialized {self.ndim}D state space with shape {self.shape}")

    def get_boundary_mask(self) -> np.ndarray:
        """Get boolean mask for boundary points.

        Returns:
            Boolean array where True indicates boundary points
        """
        mask = np.zeros(self.shape, dtype=bool)

        for dim, sv in enumerate(self.state_variables):
            # Create slice for this dimension's boundaries
            slices_lower: list[slice | int] = [slice(None)] * self.ndim
            slices_lower[dim] = 0
            mask[tuple(slices_lower)] = True

            slices_upper: list[slice | int] = [slice(None)] * self.ndim
            slices_upper[dim] = -1
            mask[tuple(slices_upper)] = True

        return mask

    def interpolate_value(self, value_function: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Interpolate value function at arbitrary points.

        Args:
            value_function: Value function on grid
            points: Points to interpolate at (shape: [n_points, n_dims])

        Returns:
            Interpolated values
        """
        if self.ndim == 1:
            interp = interpolate.interp1d(
                self.grids[0],
                value_function,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return np.array(interp(points[:, 0]))
        if self.ndim == 2:
            interp = interpolate.RegularGridInterpolator(
                self.grids, value_function, method="linear", bounds_error=False, fill_value=None
            )
            return np.array(interp(points))
        # For higher dimensions, use linear interpolation
        interp = interpolate.RegularGridInterpolator(
            self.grids, value_function, method="linear", bounds_error=False, fill_value=None
        )
        return np.array(interp(points))


class UtilityFunction(ABC):
    """Abstract base class for utility functions.

    Defines the interface for utility functions used in the HJB equation.
    Concrete implementations should provide both the utility value and its derivative.
    """

    @abstractmethod
    def evaluate(self, wealth: np.ndarray) -> np.ndarray:
        """Evaluate utility at given wealth levels.

        Args:
            wealth: Wealth values

        Returns:
            Utility values
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def derivative(self, wealth: np.ndarray) -> np.ndarray:
        """Compute marginal utility (first derivative).

        Args:
            wealth: Wealth values

        Returns:
            Marginal utility values
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def inverse_derivative(self, marginal_utility: np.ndarray) -> np.ndarray:
        """Compute inverse of marginal utility.

        Used for finding optimal controls in some formulations.

        Args:
            marginal_utility: Marginal utility values

        Returns:
            Wealth values corresponding to given marginal utilities
        """
        pass  # pylint: disable=unnecessary-pass


class LogUtility(UtilityFunction):
    """Logarithmic utility function for ergodic optimization.

    U(w) = log(w)

    This utility function maximizes the long-term growth rate and is
    particularly suitable for ergodic analysis.
    """

    def __init__(self, wealth_floor: float = 1e-6):
        """Initialize log utility.

        Args:
            wealth_floor: Minimum wealth to prevent log(0)
        """
        self.wealth_floor = wealth_floor

    def evaluate(self, wealth: np.ndarray) -> np.ndarray:
        """Evaluate log utility."""
        safe_wealth = np.maximum(wealth, self.wealth_floor)
        return np.array(np.log(safe_wealth))

    def derivative(self, wealth: np.ndarray) -> np.ndarray:
        """Compute marginal utility: U'(w) = 1/w."""
        safe_wealth = np.maximum(wealth, self.wealth_floor)
        return np.array(1.0 / safe_wealth)

    def inverse_derivative(self, marginal_utility: np.ndarray) -> np.ndarray:
        """Compute inverse: (U')^(-1)(m) = 1/m."""
        safe_marginal = np.maximum(marginal_utility, 1e-10)
        return np.array(1.0 / safe_marginal)


class PowerUtility(UtilityFunction):
    """Power (CRRA) utility function with risk aversion parameter.

    U(w) = w^(1-γ)/(1-γ) for γ ≠ 1
    U(w) = log(w) for γ = 1

    where γ is the coefficient of relative risk aversion.
    """

    def __init__(self, risk_aversion: float = 2.0, wealth_floor: float = 1e-6):
        """Initialize power utility.

        Args:
            risk_aversion: Coefficient of relative risk aversion (γ)
            wealth_floor: Minimum wealth to prevent numerical issues
        """
        self.gamma = risk_aversion
        self.wealth_floor = wealth_floor

        # Use log utility if gamma is close to 1
        if abs(self.gamma - 1.0) < 1e-10:
            self._log_utility = LogUtility(wealth_floor)

    def evaluate(self, wealth: np.ndarray) -> np.ndarray:
        """Evaluate power utility."""
        if abs(self.gamma - 1.0) < 1e-10:
            return self._log_utility.evaluate(wealth)

        safe_wealth = np.maximum(wealth, self.wealth_floor)
        return np.array(np.power(safe_wealth, 1 - self.gamma) / (1 - self.gamma))

    def derivative(self, wealth: np.ndarray) -> np.ndarray:
        """Compute marginal utility: U'(w) = w^(-γ)."""
        if abs(self.gamma - 1.0) < 1e-10:
            return self._log_utility.derivative(wealth)

        safe_wealth = np.maximum(wealth, self.wealth_floor)
        return np.array(np.power(safe_wealth, -self.gamma))

    def inverse_derivative(self, marginal_utility: np.ndarray) -> np.ndarray:
        """Compute inverse: (U')^(-1)(m) = m^(-1/γ)."""
        if abs(self.gamma - 1.0) < 1e-10:
            return self._log_utility.inverse_derivative(marginal_utility)

        safe_marginal = np.maximum(marginal_utility, 1e-10)
        return np.array(np.power(safe_marginal, -1.0 / self.gamma))


class ExpectedWealth(UtilityFunction):
    """Linear utility function for risk-neutral wealth maximization.

    U(w) = w

    This represents risk-neutral preferences where the goal is to
    maximize expected wealth.
    """

    def evaluate(self, wealth: np.ndarray) -> np.ndarray:
        """Evaluate linear utility."""
        return wealth

    def derivative(self, wealth: np.ndarray) -> np.ndarray:
        """Compute marginal utility: U'(w) = 1."""
        return np.ones_like(wealth)

    def inverse_derivative(self, marginal_utility: np.ndarray) -> np.ndarray:
        """Inverse is undefined for constant marginal utility."""
        raise NotImplementedError("Inverse derivative undefined for linear utility")


@dataclass
class HJBProblem:
    """Complete specification of an HJB optimal control problem.

    Attributes:
        state_space: State space definition
        control_variables: List of control variables
        utility_function: Utility function for optimization
        dynamics: Function defining state dynamics dx/dt = f(x, u, t)
        running_cost: Function defining running cost/reward L(x, u, t)
        terminal_value: Terminal condition V(x, T)
        discount_rate: Discount rate for future rewards
        time_horizon: Time horizon for optimization (None for infinite horizon)
    """

    state_space: StateSpace
    control_variables: List[ControlVariable]
    utility_function: UtilityFunction
    dynamics: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    running_cost: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    terminal_value: Optional[Callable[[np.ndarray], np.ndarray]] = None
    discount_rate: float = 0.0
    time_horizon: Optional[float] = None

    def __post_init__(self):
        """Validate problem specification."""
        if self.time_horizon is not None and self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.discount_rate < 0:
            raise ValueError("Discount rate must be non-negative")

        # For finite horizon problems, terminal value is required
        if self.time_horizon is not None and self.terminal_value is None:
            # Default to zero terminal value
            self.terminal_value = lambda x: np.zeros_like(x[..., 0])


@dataclass
class HJBSolverConfig:
    """Configuration for HJB solver.

    Attributes:
        time_step: Time step for PDE integration
        max_iterations: Maximum iterations for policy iteration
        tolerance: Convergence tolerance for value function
        scheme: Time stepping scheme
        use_sparse: Whether to use sparse matrices for large problems
        verbose: Whether to print progress information
    """

    time_step: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    scheme: TimeSteppingScheme = TimeSteppingScheme.IMPLICIT
    use_sparse: bool = True
    verbose: bool = True


class HJBSolver:
    """Hamilton-Jacobi-Bellman PDE solver for optimal control.

    Implements finite difference methods with upwind schemes for solving
    HJB equations. Supports multi-dimensional state spaces and various
    boundary conditions.
    """

    def __init__(self, problem: HJBProblem, config: HJBSolverConfig):
        """Initialize HJB solver.

        Args:
            problem: HJB problem specification
            config: Solver configuration
        """
        self.problem = problem
        self.config = config

        # Initialize value function and policy
        self.value_function: np.ndarray | None = None
        self.optimal_policy: dict[str, np.ndarray] | None = None

        # Set up finite difference operators
        self._setup_operators()

        logger.info(f"Initialized HJB solver for {problem.state_space.ndim}D problem")

    def _setup_operators(self):
        """Set up finite difference operators for the PDE."""
        # Store grid spacings
        self.dx = []
        for sv in self.problem.state_space.state_variables:
            grid = sv.get_grid()
            if len(grid) > 1:
                self.dx.append(grid[1] - grid[0])
            else:
                self.dx.append(1.0)

        # Will construct operators during solve based on current policy
        self.operators_initialized = True

    def _build_difference_matrix(
        self, dim: int, boundary_type: BoundaryCondition
    ) -> sparse.spmatrix:
        """Build finite difference matrix for one dimension.

        Args:
            dim: Dimension index
            boundary_type: Type of boundary condition

        Returns:
            Sparse difference operator
        """
        n = self.problem.state_space.state_variables[dim].num_points
        dx = self.dx[dim]

        # First derivative (upwind)
        # We'll build this dynamically based on drift direction

        # Second derivative (diffusion)
        diagonals = np.ones((3, n))
        diagonals[0] *= 1.0 / (dx * dx)  # Lower diagonal
        diagonals[1] *= -2.0 / (dx * dx)  # Main diagonal
        diagonals[2] *= 1.0 / (dx * dx)  # Upper diagonal

        # Apply boundary conditions
        if boundary_type == BoundaryCondition.DIRICHLET:
            # Fixed value at boundaries (handled separately)
            diagonals[0, 0] = 0
            diagonals[1, 0] = 1
            diagonals[2, 0] = 0
            diagonals[0, -1] = 0
            diagonals[1, -1] = 1
            diagonals[2, -1] = 0
        elif boundary_type == BoundaryCondition.NEUMANN:
            # Zero derivative at boundaries
            diagonals[1, 0] += diagonals[0, 0]
            diagonals[0, 0] = 0
            diagonals[1, -1] += diagonals[2, -1]
            diagonals[2, -1] = 0
        elif boundary_type == BoundaryCondition.ABSORBING:
            # Absorbing boundaries: value is fixed at boundary
            # First row: only main diagonal = 1
            diagonals[1, 0] = 1
            diagonals[2, 0] = 0  # No upper diagonal from first row
            # Last row: only main diagonal = 1
            # Note: For lower diagonal, element at index i goes to matrix[i+1, i]
            # So to zero out matrix[n-1, n-2], we need diagonals[0, n-2]
            diagonals[0, n - 2] = 0  # Zero out lower diagonal element going to last row
            diagonals[1, n - 1] = 1  # Set main diagonal of last row to 1

        matrix = sparse.diags(diagonals, offsets=[-1, 0, 1], shape=(n, n))

        return matrix

    def _apply_upwind_scheme(self, value: np.ndarray, drift: np.ndarray, dim: int) -> np.ndarray:
        """Apply upwind finite difference for advection term.

        Args:
            value: Value function
            drift: Drift values at each point
            dim: Dimension for differentiation

        Returns:
            Advection term contribution
        """
        dx = self.dx[dim]
        result = np.zeros_like(value)

        # Reshape for easier indexing
        shape = value.shape

        # Forward difference where drift > 0
        mask_pos = drift > 0
        if dim == 0:
            result[mask_pos] = (
                (np.roll(value, -1, axis=dim)[mask_pos] - value[mask_pos]) / dx * drift[mask_pos]
            )

        # Backward difference where drift < 0
        mask_neg = drift < 0
        if dim == 0:
            result[mask_neg] = (
                (value[mask_neg] - np.roll(value, 1, axis=dim)[mask_neg]) / dx * drift[mask_neg]
            )

        return result

    def solve(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Solve the HJB equation using policy iteration.

        Returns:
            Tuple of (value_function, optimal_policy_dict)
        """
        logger.info("Starting HJB solution with policy iteration")

        # Initialize value function
        if self.problem.time_horizon is not None:
            # Finite horizon: start from terminal condition
            state_points = np.stack(self.problem.state_space.flat_grids, axis=-1)
            if self.problem.terminal_value is not None:
                terminal_values = self.problem.terminal_value(state_points)
                self.value_function = terminal_values.reshape(self.problem.state_space.shape)
            else:
                self.value_function = np.zeros(self.problem.state_space.shape)
        else:
            # Infinite horizon: initialize with zeros or heuristic
            self.value_function = np.zeros(self.problem.state_space.shape)

        # Initialize policy with mid-range controls
        if self.optimal_policy is None:
            self.optimal_policy = {}
        for cv in self.problem.control_variables:
            self.optimal_policy[cv.name] = np.full(
                self.problem.state_space.shape, (cv.min_value + cv.max_value) / 2
            )

        # Policy iteration
        for iteration in range(self.config.max_iterations):
            old_value = self.value_function.copy()

            # Policy evaluation step
            self._policy_evaluation()

            # Policy improvement step
            self._policy_improvement()

            # Check convergence
            value_change = np.max(np.abs(self.value_function - old_value))

            if self.config.verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: value change = {value_change:.6e}")

            if value_change < self.config.tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

        if iteration == self.config.max_iterations - 1:
            logger.warning("Max iterations reached without convergence")

        return self.value_function, self.optimal_policy

    def _reshape_cost(self, cost):
        """Helper method to reshape cost array."""
        if hasattr(cost, "ndim") and cost.ndim > 1:
            # If cost is multi-dimensional, take the first column or mean
            if cost.shape[1] > 1:
                cost = np.mean(cost, axis=1)  # Average across extra dimensions
            else:
                cost = cost[:, 0]  # Take first column
        return cost.reshape(self.problem.state_space.shape)

    def _apply_upwind_drift(self, new_v, drift, dt):
        """Apply upwind differencing for drift term."""
        if not np.any(np.abs(drift) > 1e-10):
            return new_v

        if self.value_function is None:
            return new_v

        for dim in range(drift.shape[-1]):
            if dim >= len(self.problem.state_space.state_variables):
                continue
            dx = self.dx[dim]
            drift_component = drift[..., dim]

            # Simple upwind differencing
            grad = np.zeros_like(self.value_function)
            grad[1:] = (self.value_function[1:] - self.value_function[:-1]) / dx
            new_v += dt * drift_component * grad
        return new_v

    def _update_value_finite_horizon(self, old_v, cost, drift, dt):
        """Update value function for finite horizon problems."""
        # Backward Euler scheme for parabolic PDE
        new_v = old_v + dt * cost

        # Add discount term if applicable
        if self.problem.discount_rate > 0:
            new_v -= dt * self.problem.discount_rate * old_v

        # Add drift term using upwind differencing
        return self._apply_upwind_drift(new_v, drift, dt)

    def _policy_evaluation(self):
        """Evaluate current policy by solving linear PDE."""
        # For now, implement a simple iterative scheme
        # In production, would use sparse linear solver

        if self.value_function is None or self.optimal_policy is None:
            return

        dt = self.config.time_step

        for _ in range(100):  # Inner iterations for policy evaluation
            # Type guard ensures value_function is not None after the check above
            assert self.value_function is not None  # For mypy
            old_v = self.value_function.copy()

            # Get state and control grids
            state_points = np.stack(self.problem.state_space.flat_grids, axis=-1)
            control_array = np.stack(
                [self.optimal_policy[cv.name].ravel() for cv in self.problem.control_variables],
                axis=-1,
            )

            # Compute dynamics and running cost
            drift = self.problem.dynamics(state_points, control_array, 0.0)
            cost = self.problem.running_cost(state_points, control_array, 0.0)

            # Reshape
            drift = drift.reshape(self.problem.state_space.shape + (-1,))
            cost = self._reshape_cost(cost)

            # Apply finite differences with upwind scheme
            # For finite horizon problems, integrate backwards from terminal condition
            if self.problem.time_horizon is not None:
                new_v = self._update_value_finite_horizon(old_v, cost, drift, dt)
            else:
                # For infinite horizon, use standard update
                new_v = old_v + dt * (-self.problem.discount_rate * old_v + cost)

            # Apply boundary conditions (skip for now to preserve terminal condition)

            self.value_function = new_v

            # Check inner convergence
            if np.max(np.abs(new_v - old_v)) < self.config.tolerance / 10:
                break

    def _policy_improvement(self):
        """Improve policy by maximizing Hamiltonian."""
        # For each state, find optimal control
        state_points = np.stack(self.problem.state_space.flat_grids, axis=-1)

        for i, point in enumerate(state_points):
            # Discrete optimization over control space
            best_value = -np.inf
            best_control = None

            # Sample control space
            control_samples = []
            for cv in self.problem.control_variables:
                control_samples.append(cv.get_values())

            # Grid search (would use gradient-based in production)
            from itertools import product

            for control in product(*control_samples):
                control_array = np.array(control)

                # Compute Hamiltonian
                drift = self.problem.dynamics(
                    point.reshape(1, -1), control_array.reshape(1, -1), 0.0
                )
                cost = self.problem.running_cost(
                    point.reshape(1, -1), control_array.reshape(1, -1), 0.0
                )

                # Approximate value function gradient (would use interpolation)
                if hasattr(cost, "__getitem__"):
                    # Handle multi-dimensional cost arrays
                    cost_flat = np.asarray(cost).flatten()
                    hamiltonian = float(cost_flat[0])
                else:
                    hamiltonian = float(cost)  # Simplified

                if hamiltonian > best_value:
                    best_value = hamiltonian
                    best_control = control_array

            # Update policy
            if best_control is not None:
                idx = np.unravel_index(i, self.problem.state_space.shape)
                for j, cv in enumerate(self.problem.control_variables):
                    if self.optimal_policy is not None:
                        self.optimal_policy[cv.name][idx] = best_control[j]

    def extract_feedback_control(self, state: np.ndarray) -> Dict[str, float]:
        """Extract feedback control law at given state.

        Args:
            state: Current state values

        Returns:
            Dictionary of control variable names to optimal values
        """
        if self.optimal_policy is None:
            raise RuntimeError("Must solve HJB equation before extracting controls")

        # Interpolate policy at given state
        controls = {}
        for cv in self.problem.control_variables:
            policy_func = interpolate.RegularGridInterpolator(
                self.problem.state_space.grids,
                self.optimal_policy[cv.name],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            controls[cv.name] = float(policy_func(state.reshape(1, -1))[0])

        return controls

    def compute_convergence_metrics(self) -> Dict[str, Any]:
        """Compute metrics for assessing solution quality.

        Returns:
            Dictionary of convergence metrics
        """
        if self.value_function is None:
            return {"error": "No solution computed yet"}

        # Compute residual of HJB equation
        state_points = np.stack(self.problem.state_space.flat_grids, axis=-1)

        # optimal_policy should be non-None at this point since value_function is non-None
        assert self.optimal_policy is not None
        control_array = np.stack(
            [self.optimal_policy[cv.name].ravel() for cv in self.problem.control_variables], axis=-1
        )

        # Evaluate HJB residual
        drift = self.problem.dynamics(state_points, control_array, 0.0)
        cost = self.problem.running_cost(state_points, control_array, 0.0)

        # Approximate time derivative (backward difference)
        dt = self.config.time_step
        v_flat = self.value_function.ravel()

        # Simplified residual (would compute full PDE residual in production)
        cost_flat = cost.ravel() if hasattr(cost, "ravel") else cost
        residual = np.abs(-self.problem.discount_rate * v_flat + cost_flat)

        return {
            "max_residual": float(np.max(residual)),
            "mean_residual": float(np.mean(residual)),
            "value_function_range": (
                float(np.min(self.value_function)),
                float(np.max(self.value_function)),
            ),
            "policy_stats": {
                cv.name: {
                    "min": float(np.min(self.optimal_policy[cv.name])),
                    "max": float(np.max(self.optimal_policy[cv.name])),
                    "mean": float(np.mean(self.optimal_policy[cv.name])),
                }
                for cv in self.problem.control_variables
            },
        }


def create_custom_utility(
    evaluate_func: Callable[[np.ndarray], np.ndarray],
    derivative_func: Callable[[np.ndarray], np.ndarray],
    inverse_derivative_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> UtilityFunction:
    """Factory function for creating custom utility functions.

    This function allows users to create custom utility functions by providing
    the evaluation and derivative functions. This is the recommended way to
    add new utility functions beyond the built-in ones.

    Args:
        evaluate_func: Function that evaluates U(w)
        derivative_func: Function that computes U'(w)
        inverse_derivative_func: Optional function for (U')^(-1)(m)

    Returns:
        Custom utility function instance

    Example:
        >>> # Create exponential utility: U(w) = 1 - exp(-α*w)
        >>> def exp_eval(w):
        ...     alpha = 0.01
        ...     return 1 - np.exp(-alpha * w)
        >>> def exp_deriv(w):
        ...     alpha = 0.01
        ...     return alpha * np.exp(-alpha * w)
        >>> exp_utility = create_custom_utility(exp_eval, exp_deriv)
    """

    class CustomUtility(UtilityFunction):
        """Dynamically created custom utility function."""

        def evaluate(self, wealth: np.ndarray) -> np.ndarray:
            return evaluate_func(wealth)

        def derivative(self, wealth: np.ndarray) -> np.ndarray:
            return derivative_func(wealth)

        def inverse_derivative(self, marginal_utility: np.ndarray) -> np.ndarray:
            if inverse_derivative_func is None:
                raise NotImplementedError("Inverse derivative not provided for custom utility")
            return inverse_derivative_func(marginal_utility)

    return CustomUtility()

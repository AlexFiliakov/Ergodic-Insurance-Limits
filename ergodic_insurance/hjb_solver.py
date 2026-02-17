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
from dataclasses import dataclass
from enum import Enum
from itertools import product as itertools_product
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate, sparse

logger = logging.getLogger(__name__)

# Module-level named constants for numerical tolerances
_DRIFT_THRESHOLD = 1e-10
_MARGINAL_UTILITY_FLOOR = 1e-10
_GAMMA_TOLERANCE = 1e-10

# Policy improvement strategy thresholds
_VECTORIZE_COMBO_THRESHOLD = 5000  # Below: full vectorized batch
_COARSE_STRIDE = 3  # Adaptive: every 3rd point
_REFINE_RADIUS = 2  # Adaptive: +/-2 points around optimum
_DEFAULT_MEMORY_BUDGET_MB = 256  # Max memory for batched evaluation


class NumericalDivergenceError(RuntimeError):
    """Raised when the HJB solver detects NaN or Inf in the value function."""

    pass


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
    """Definition of a state variable in the HJB problem."""

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
    """Definition of a control variable in the HJB problem."""

    name: str
    min_value: float
    max_value: float
    num_points: int = 50
    continuous: bool = True
    log_scale: bool = False

    def __post_init__(self):
        """Validate control variable configuration."""
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value must be less than max_value for {self.name}")
        if self.num_points < 2:
            raise ValueError(f"Need at least 2 control points for {self.name}")
        if self.log_scale and self.min_value <= 0:
            raise ValueError(f"Cannot use log scale with non-positive min_value for {self.name}")

    def get_values(self) -> np.ndarray:
        """Get discrete control values for optimization.

        Returns:
            Array of control values
        """
        if self.log_scale:
            return np.geomspace(self.min_value, self.max_value, self.num_points)
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

        for dim, _sv in enumerate(self.state_variables):
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
        safe_marginal = np.maximum(marginal_utility, _MARGINAL_UTILITY_FLOOR)
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
        if abs(self.gamma - 1.0) < _GAMMA_TOLERANCE:
            self._log_utility = LogUtility(wealth_floor)

    def evaluate(self, wealth: np.ndarray) -> np.ndarray:
        """Evaluate power utility."""
        if abs(self.gamma - 1.0) < _GAMMA_TOLERANCE:
            return self._log_utility.evaluate(wealth)

        safe_wealth = np.maximum(wealth, self.wealth_floor)
        return np.array(np.power(safe_wealth, 1 - self.gamma) / (1 - self.gamma))

    def derivative(self, wealth: np.ndarray) -> np.ndarray:
        """Compute marginal utility: U'(w) = w^(-γ)."""
        if abs(self.gamma - 1.0) < _GAMMA_TOLERANCE:
            return self._log_utility.derivative(wealth)

        safe_wealth = np.maximum(wealth, self.wealth_floor)
        return np.array(np.power(safe_wealth, -self.gamma))

    def inverse_derivative(self, marginal_utility: np.ndarray) -> np.ndarray:
        """Compute inverse: (U')^(-1)(m) = m^(-1/γ)."""
        if abs(self.gamma - 1.0) < _GAMMA_TOLERANCE:
            return self._log_utility.inverse_derivative(marginal_utility)

        safe_marginal = np.maximum(marginal_utility, _MARGINAL_UTILITY_FLOOR)
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
    """Complete specification of an HJB optimal control problem."""

    state_space: StateSpace
    control_variables: List[ControlVariable]
    utility_function: UtilityFunction
    dynamics: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    running_cost: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    terminal_value: Optional[Callable[[np.ndarray], np.ndarray]] = None
    discount_rate: float = 0.0
    time_horizon: Optional[float] = None
    diffusion: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
    """Optional callback returning σ²(x,u,t) with same shape as dynamics output.
    When provided, the solver includes the ½σ²·∇²V diffusion term."""

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
    """Configuration for HJB solver."""

    time_step: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    scheme: TimeSteppingScheme = TimeSteppingScheme.EXPLICIT
    use_sparse: bool = True
    verbose: bool = True
    inner_max_iterations: int = 100
    inner_tolerance_factor: float = 0.1  # inner_tol = tolerance * this
    rannacher_steps: int = 2  # Number of implicit half-step pairs for CN startup
    control_search_strategy: str = "auto"  # "auto", "vectorized", "adaptive", "loop", "gradient"
    control_memory_budget_mb: int = 256  # Max memory for batched control evaluation


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

        # Boundary values for Dirichlet BCs (captured from initial/terminal condition)
        self._boundary_values: dict[int, dict[str, np.ndarray]] | None = None

        # Cache for factorized sparse operators: (theta, dt, dim) -> (solve_func, B_or_None)
        self._operator_cache: Dict[
            Tuple[float, float, int], Tuple[Any, Optional[sparse.spmatrix]]
        ] = {}

        # Set up finite difference operators
        self._setup_operators()

        logger.info(f"Initialized HJB solver for {problem.state_space.ndim}D problem")

    def _setup_operators(self):
        """Set up finite difference operators for the PDE."""
        # Store per-interval grid spacings as arrays (supports non-uniform grids)
        self.dx = []
        for sv in self.problem.state_space.state_variables:
            grid = sv.get_grid()
            if len(grid) > 1:
                self.dx.append(np.diff(grid))
            else:
                self.dx.append(np.array([1.0]))

        # Will construct operators during solve based on current policy
        self.operators_initialized = True

    def _compute_cfl_number(
        self,
        drift: np.ndarray,
        sigma_sq: np.ndarray | None,
        dt: float,
    ) -> tuple[float, float]:
        """Compute CFL numbers for advection and diffusion stability.

        Args:
            drift: Drift values on the state grid, shape state_shape + (ndim,)
            sigma_sq: Diffusion coefficients on state grid, shape state_shape + (ndim,),
                or None if no diffusion.
            dt: Time step size.

        Returns:
            Tuple of (advection_cfl, diffusion_cfl).
        """
        advection_cfl = 0.0
        diffusion_cfl = 0.0

        n_dims = min(drift.shape[-1], len(self.problem.state_space.state_variables))
        for dim in range(n_dims):
            dx_arr = self.dx[dim]
            dx_min = float(np.min(dx_arr))
            dx_mean = float(np.mean(dx_arr))

            drift_max = float(np.max(np.abs(drift[..., dim])))
            if dx_min > 0:
                advection_cfl = max(advection_cfl, drift_max * dt / dx_min)

            if sigma_sq is not None and dx_mean > 0:
                # Effective diffusion coefficient is D = 0.5 * sigma_sq
                sigma_sq_max = float(np.max(np.abs(sigma_sq[..., dim])))
                diffusion_cfl = max(diffusion_cfl, 0.5 * sigma_sq_max * dt / (dx_mean**2))

        return advection_cfl, diffusion_cfl

    def _build_spatial_operator_1d(
        self,
        drift_1d: np.ndarray,
        sigma_sq_1d: np.ndarray | None,
        dim: int = 0,
    ) -> sparse.spmatrix:
        """Build the 1D spatial operator L as a sparse tridiagonal matrix.

        The operator represents: L(V)[i] = -rho*V[i] + drift*dV/dx + 0.5*sigma^2*d2V/dx2
        using upwind first derivatives and central second derivatives, consistent
        with the explicit scheme.

        Args:
            drift_1d: Drift values at each grid point, shape (N,).
            sigma_sq_1d: Diffusion coefficient sigma^2 at each grid point, shape (N,),
                or None for pure advection.
            dim: Dimension index (for grid spacing lookup).

        Returns:
            Sparse CSR matrix of shape (N, N).
        """
        N = len(drift_1d)
        dx_arr = self.dx[dim]
        dx_mean = float(np.mean(dx_arr))
        rho = self.problem.discount_rate

        # Sub-diagonal (V[i-1] coefficients), super-diagonal (V[i+1] coefficients),
        # main diagonal (V[i] coefficients) — all for interior points
        sub = np.zeros(N)
        main = np.zeros(N)
        sup = np.zeros(N)

        # Interior points: i = 1, ..., N-2
        interior = slice(1, N - 1)
        drift_int = drift_1d[interior]
        drift_pos = np.maximum(drift_int, 0.0)
        drift_neg = np.minimum(drift_int, 0.0)

        # dx for backward diff at point i: dx_arr[i-1]
        dx_back = dx_arr[0 : N - 2]
        # dx for forward diff at point i: dx_arr[i]
        dx_fwd = dx_arr[1 : N - 1]

        # Diffusion coefficient at interior points
        if sigma_sq_1d is not None:
            D = 0.5 * sigma_sq_1d[interior]
        else:
            D = np.zeros(N - 2)

        dx_sq = dx_mean**2

        # Operator coefficients (consistent with _apply_upwind_scheme).
        #
        # The HJB PDE is V_t = drift*V_x + ..., which is V_t + (-drift)*V_x = 0.
        # The effective advection coefficient is a = -drift, so the upwind
        # direction is OPPOSITE to standard advection references that assume
        # V_t + a*V_x = 0 with a = drift.  Concretely:
        #   drift > 0  =>  a < 0  =>  upwind = forward diff
        #   drift < 0  =>  a > 0  =>  upwind = backward diff
        sub[interior] = -drift_neg / dx_back + D / dx_sq
        main[interior] = -drift_pos / dx_fwd + drift_neg / dx_back - 2.0 * D / dx_sq - rho
        sup[interior] = drift_pos / dx_fwd + D / dx_sq

        # Boundary rows are zero (will be handled after solve)

        L = sparse.diags(
            [sub[1:], main, sup[:-1]],
            offsets=[-1, 0, 1],
            shape=(N, N),
            format="csr",
        )
        return L

    def _invalidate_operator_cache(self):
        """Clear the cached sparse matrix factorizations.

        Must be called whenever drift or diffusion coefficients may change
        (i.e., at the start of each policy evaluation cycle).
        """
        self._operator_cache.clear()

    def _theta_step_1d(
        self,
        old_v: np.ndarray,
        cost: np.ndarray,
        drift_1d: np.ndarray,
        sigma_sq_1d: np.ndarray | None,
        dt: float,
        theta: float,
        dim: int = 0,
    ) -> np.ndarray:
        """Perform one theta-scheme time step for a 1D problem.

        Solves: (I - theta*dt*L)*V_new = (I + (1-theta)*dt*L)*V_old + dt*cost

        For theta=1: fully implicit (backward Euler).
        For theta=0.5: Crank-Nicolson.
        For theta=0: explicit (forward Euler).

        Args:
            old_v: Current value function, shape (N,).
            cost: Running cost at each grid point, shape (N,).
            drift_1d: Drift at each grid point, shape (N,).
            sigma_sq_1d: Diffusion coefficient sigma^2, shape (N,) or None.
            dt: Time step.
            theta: Implicitness parameter in [0, 1].
            dim: Dimension index.

        Returns:
            Updated value function, shape (N,).
        """
        N = len(old_v)
        cache_key = (theta, dt, dim)

        if cache_key in self._operator_cache:
            # Cache hit: reuse factorized solver and B matrix
            solve_func, B = self._operator_cache[cache_key]
        else:
            # Cache miss: build, factorize, and store
            L = self._build_spatial_operator_1d(drift_1d, sigma_sq_1d, dim)
            I_mat = sparse.eye(N, format="csr")

            # LHS: A = I - theta*dt*L
            A = I_mat - theta * dt * L

            # Set boundary rows to identity (boundary values handled by
            # _apply_boundary_conditions after return)
            A = A.tolil()
            A[0, :] = 0
            A[0, 0] = 1.0
            A[N - 1, :] = 0
            A[N - 1, N - 1] = 1.0
            A = A.tocsc()

            # Factorize once via SuperLU
            solve_func = sparse.linalg.splu(A).solve

            # Build B matrix for Crank-Nicolson (theta < 1)
            if theta < 1.0:
                B = I_mat + (1.0 - theta) * dt * L
            else:
                B = None

            self._operator_cache[cache_key] = (solve_func, B)

        # RHS: B @ old_v + dt * cost
        if B is not None:
            rhs = B @ old_v + dt * cost
        else:
            rhs = old_v + dt * cost

        # Preserve old boundary values in RHS (will be overwritten by BCs)
        rhs[0] = old_v[0]
        rhs[N - 1] = old_v[N - 1]

        new_v: np.ndarray = solve_func(rhs)
        return new_v

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
        dx = float(np.mean(self.dx[dim]))

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
        elif boundary_type in (BoundaryCondition.NEUMANN, BoundaryCondition.REFLECTING):
            # Zero first derivative at boundaries (ghost-node reflection)
            diagonals[1, 0] += diagonals[0, 0]
            diagonals[0, 0] = 0
            diagonals[1, -1] += diagonals[2, -1]
            diagonals[2, -1] = 0
        elif boundary_type == BoundaryCondition.ABSORBING:
            # Absorbing boundary: enforce d²V/dx² = 0 (linear extrapolation)
            # Lower boundary: V[0] - 2*V[1] + V[2] = 0
            # Upper boundary: V[n-3] - 2*V[n-2] + V[n-1] = 0
            # Row 0 needs entry at column 2 (offset +2), and row n-1 needs
            # entry at column n-3 (offset -2), so we build as tridiagonal
            # then add the extra entries.
            coeff = 1.0 / (dx * dx)
            # Row 0: [coeff, -2*coeff, 0, ..., 0] (tridiagonal part)
            diagonals[1, 0] = coeff
            diagonals[2, 0] = -2.0 * coeff
            # Row n-1: [0, ..., 0, -2*coeff, coeff] (tridiagonal part)
            diagonals[0, n - 2] = -2.0 * coeff
            diagonals[1, n - 1] = coeff

        matrix = sparse.diags(diagonals, offsets=[-1, 0, 1], shape=(n, n))

        if boundary_type == BoundaryCondition.ABSORBING:
            # Add the off-tridiagonal entries for absorbing BCs
            coeff = 1.0 / (dx * dx)
            matrix = matrix.tolil()
            matrix[0, 2] = coeff  # V[2] coefficient in lower boundary row
            matrix[n - 1, n - 3] = coeff  # V[n-3] coefficient in upper boundary row
            matrix = matrix.tocsr()

        return matrix

    def _apply_upwind_scheme(self, value: np.ndarray, drift: np.ndarray, dim: int) -> np.ndarray:
        """Apply upwind finite difference for advection term.

        Uses proper boundary-aware slicing (no wraparound) and supports
        non-uniform grid spacing for all dimensions.

        Args:
            value: Value function on state grid
            drift: Drift values at each grid point
            dim: Dimension for differentiation

        Returns:
            Advection term contribution (drift * dV/dx)
        """
        dx_array = self.dx[dim]  # Array of per-interval spacings
        ndim = value.ndim
        n = value.shape[dim]
        result = np.zeros_like(value)

        # Build index slices for interior points
        hi = [slice(None)] * ndim
        lo = [slice(None)] * ndim
        hi[dim] = slice(1, None)  # indices 1..N-1
        lo[dim] = slice(None, -1)  # indices 0..N-2

        # Shape dx_array for broadcasting: (1, ..., N-1, ..., 1)
        dx_shape = [1] * ndim
        dx_shape[dim] = n - 1
        dx_bc = dx_array.reshape(dx_shape)

        # Forward difference: (V[i+1] - V[i]) / dx[i] for i=0..N-2
        # At i=N-1, the forward difference is 0 (boundary)
        fwd_diff = np.zeros_like(value)
        fwd_diff[tuple(lo)] = (value[tuple(hi)] - value[tuple(lo)]) / dx_bc

        # Backward difference: (V[i] - V[i-1]) / dx[i-1] for i=1..N-1
        # At i=0, the backward difference is 0 (boundary)
        bwd_diff = np.zeros_like(value)
        bwd_diff[tuple(hi)] = (value[tuple(hi)] - value[tuple(lo)]) / dx_bc

        # Upwind selection based on the HJB sign convention V_t = drift*V_x + ...
        # The effective advection coefficient is -drift, so:
        #   drift > 0 => forward diff (upwind for negative effective coefficient)
        #   drift < 0 => backward diff (upwind for positive effective coefficient)
        mask_pos = drift > 0
        mask_neg = drift < 0

        result[mask_pos] = fwd_diff[mask_pos] * drift[mask_pos]
        result[mask_neg] = bwd_diff[mask_neg] * drift[mask_neg]

        return result

    def _compute_gradient(self) -> np.ndarray:
        """Compute numerical gradient of the value function.

        Uses np.gradient which handles non-uniform grids with second-order
        accurate central differences in the interior and first-order accurate
        one-sided differences at the boundaries.

        Returns:
            Gradient array with shape state_shape + (ndim,)
        """
        if self.value_function is None:
            shape = self.problem.state_space.shape + (self.problem.state_space.ndim,)
            return np.zeros(shape)

        grids = self.problem.state_space.grids

        if self.problem.state_space.ndim == 1:
            grad_components = [np.gradient(self.value_function, grids[0])]
        else:
            grad_components = np.gradient(self.value_function, *grids)

        return np.stack(grad_components, axis=-1)

    def _compute_second_derivatives(self, value: np.ndarray) -> np.ndarray:
        """Compute second derivatives d²V/dx_i² for each dimension.

        Uses central finite differences for interior points. Boundary
        values default to zero (no diffusion contribution at boundaries).

        Args:
            value: Value function on state grid

        Returns:
            Array with shape state_shape + (ndim,) containing d²V/dx_i²
        """
        ndim = self.problem.state_space.ndim
        components = []

        for dim in range(ndim):
            dx_array = self.dx[dim]
            dx = float(np.mean(dx_array))

            d2v = np.zeros_like(value)

            # Interior: (V[i+1] - 2*V[i] + V[i-1]) / dx²
            hi = [slice(None)] * ndim
            mid = [slice(None)] * ndim
            lo = [slice(None)] * ndim
            hi[dim] = slice(2, None)
            mid[dim] = slice(1, -1)
            lo[dim] = slice(None, -2)

            d2v[tuple(mid)] = (value[tuple(hi)] - 2 * value[tuple(mid)] + value[tuple(lo)]) / (
                dx * dx
            )

            components.append(d2v)

        return np.stack(components, axis=-1)

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

        # Capture boundary values for Dirichlet enforcement
        self._boundary_values = {}
        ndim = self.problem.state_space.ndim
        for dim in range(ndim):
            sv = self.problem.state_space.state_variables[dim]
            if (
                sv.boundary_lower == BoundaryCondition.DIRICHLET
                or sv.boundary_upper == BoundaryCondition.DIRICHLET
            ):
                lo_idx: List[Any] = [slice(None)] * ndim
                hi_idx: List[Any] = [slice(None)] * ndim
                lo_idx[dim] = 0
                hi_idx[dim] = -1
                self._boundary_values[dim] = {
                    "lower": self.value_function[tuple(lo_idx)].copy(),
                    "upper": self.value_function[tuple(hi_idx)].copy(),
                }

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

            # Check for NaN/Inf after policy evaluation (#453)
            if not np.all(np.isfinite(self.value_function)):
                n_nan = int(np.sum(np.isnan(self.value_function)))
                n_inf = int(np.sum(np.isinf(self.value_function)))
                raise NumericalDivergenceError(
                    f"HJB solver diverged at outer iteration {iteration}: "
                    f"value function contains {n_nan} NaN and {n_inf} Inf values. "
                    f"Consider reducing time_step (current: {self.config.time_step}) "
                    f"or using an implicit scheme."
                )

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
        """Apply upwind differencing for drift term across all dimensions."""
        if not np.any(np.abs(drift) > _DRIFT_THRESHOLD):
            return new_v

        if self.value_function is None:
            return new_v

        for dim in range(drift.shape[-1]):
            if dim >= len(self.problem.state_space.state_variables):
                continue
            drift_component = drift[..., dim]
            advection = self._apply_upwind_scheme(self.value_function, drift_component, dim)
            new_v = new_v + dt * advection
        return new_v

    def _apply_diffusion_term(
        self,
        value: np.ndarray,
        sigma_sq: np.ndarray,
    ) -> np.ndarray:
        """Compute diffusion contribution ½σ²∇²V.

        Args:
            value: Value function on state grid
            sigma_sq: Diffusion coefficients σ²(x,u) with shape state_shape + (ndim,)

        Returns:
            Diffusion term (scalar field, same shape as value)
        """
        d2v = self._compute_second_derivatives(value)

        diffusion = np.zeros_like(value)
        n_dims = min(sigma_sq.shape[-1], d2v.shape[-1])
        for dim in range(n_dims):
            if dim >= len(self.problem.state_space.state_variables):
                continue
            diffusion += 0.5 * sigma_sq[..., dim] * d2v[..., dim]

        return diffusion

    def _apply_boundary_conditions(self, value: np.ndarray) -> np.ndarray:
        """Enforce boundary conditions on the value function.

        Applies the prescribed boundary condition for each dimension:
        - ABSORBING: linear extrapolation so d²V/dx² = 0 at boundary
        - DIRICHLET: reset boundary to prescribed (initial/terminal) values
        - NEUMANN / REFLECTING: copy adjacent interior value so dV/dx = 0

        Args:
            value: Value function on state grid.

        Returns:
            Value function with boundary conditions enforced.
        """
        ndim = self.problem.state_space.ndim
        result = value.copy()

        for dim in range(ndim):
            sv = self.problem.state_space.state_variables[dim]

            # --- lower boundary ---
            lo: List[Any] = [slice(None)] * ndim
            p1: List[Any] = [slice(None)] * ndim
            p2: List[Any] = [slice(None)] * ndim
            lo[dim] = 0
            p1[dim] = 1
            p2[dim] = 2

            if sv.boundary_lower == BoundaryCondition.ABSORBING:
                # V[0] = 2*V[1] - V[2]  →  d²V/dx² = 0
                result[tuple(lo)] = 2.0 * result[tuple(p1)] - result[tuple(p2)]
            elif sv.boundary_lower == BoundaryCondition.DIRICHLET:
                # Reset to initial value (stored during initialization)
                if self._boundary_values is not None and dim in self._boundary_values:
                    result[tuple(lo)] = self._boundary_values[dim]["lower"]
            elif sv.boundary_lower in (
                BoundaryCondition.NEUMANN,
                BoundaryCondition.REFLECTING,
            ):
                # dV/dx = 0  →  V[0] = V[1]
                result[tuple(lo)] = result[tuple(p1)]

            # --- upper boundary ---
            hi: List[Any] = [slice(None)] * ndim
            m1: List[Any] = [slice(None)] * ndim
            m2: List[Any] = [slice(None)] * ndim
            hi[dim] = -1
            m1[dim] = -2
            m2[dim] = -3

            if sv.boundary_upper == BoundaryCondition.ABSORBING:
                # V[-1] = 2*V[-2] - V[-3]  →  d²V/dx² = 0
                result[tuple(hi)] = 2.0 * result[tuple(m1)] - result[tuple(m2)]
            elif sv.boundary_upper == BoundaryCondition.DIRICHLET:
                if self._boundary_values is not None and dim in self._boundary_values:
                    result[tuple(hi)] = self._boundary_values[dim]["upper"]
            elif sv.boundary_upper in (
                BoundaryCondition.NEUMANN,
                BoundaryCondition.REFLECTING,
            ):
                # dV/dx = 0  →  V[-1] = V[-2]
                result[tuple(hi)] = result[tuple(m1)]

        return result

    def _update_value_finite_horizon(self, old_v, cost, drift, dt, sigma_sq=None):
        """Update value function for finite horizon problems."""
        # Backward Euler scheme for parabolic PDE
        new_v = old_v + dt * cost

        # Add discount term if applicable
        if self.problem.discount_rate > 0:
            new_v -= dt * self.problem.discount_rate * old_v

        # Add drift term using upwind differencing
        new_v = self._apply_upwind_drift(new_v, drift, dt)

        # Add diffusion term: ½σ²∇²V
        if sigma_sq is not None:
            new_v += dt * self._apply_diffusion_term(old_v, sigma_sq)

        return new_v

    def _policy_evaluation(self):
        """Evaluate current policy by solving linear PDE.

        Supports explicit, implicit, and Crank-Nicolson time-stepping schemes.
        For explicit scheme, performs CFL stability check and auto-adapts dt.
        """
        if self.value_function is None or self.optimal_policy is None:
            return

        # Policy may have changed since last evaluation; invalidate cached operators
        self._invalidate_operator_cache()

        dt = self.config.time_step
        scheme = self.config.scheme

        for _ in range(self.config.inner_max_iterations):  # Inner iterations for policy evaluation
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

            # Compute diffusion coefficient if specified
            sigma_sq = None
            if self.problem.diffusion is not None:
                sigma_sq_raw = self.problem.diffusion(state_points, control_array, 0.0)
                sigma_sq = sigma_sq_raw.reshape(self.problem.state_space.shape + (-1,))

            # CFL stability check for explicit scheme (#452)
            if scheme == TimeSteppingScheme.EXPLICIT and _ == 0:
                adv_cfl, diff_cfl = self._compute_cfl_number(drift, sigma_sq, dt)
                if adv_cfl > 1.0 or diff_cfl > 1.0:
                    # Compute safe dt
                    max_rate = 0.0
                    n_dims = min(drift.shape[-1], len(self.problem.state_space.state_variables))
                    for dim in range(n_dims):
                        dx_arr = self.dx[dim]
                        dx_min = float(np.min(dx_arr))
                        dx_mean = float(np.mean(dx_arr))
                        drift_max = float(np.max(np.abs(drift[..., dim])))
                        if dx_min > 0:
                            max_rate += drift_max / dx_min
                        if sigma_sq is not None and dx_mean > 0:
                            max_rate += (
                                0.5 * float(np.max(np.abs(sigma_sq[..., dim]))) / (dx_mean**2)
                            )
                    max_rate += self.problem.discount_rate
                    if max_rate > 0:
                        dt_safe = 0.9 / max_rate
                        logger.warning(
                            f"CFL condition violated (advection CFL={adv_cfl:.2f}, "
                            f"diffusion CFL={diff_cfl:.2f}). "
                            f"Auto-reducing dt from {dt:.4e} to {dt_safe:.4e}."
                        )
                        dt = dt_safe

            # Time-stepping: branch on scheme (#451)
            use_implicit = scheme in (
                TimeSteppingScheme.IMPLICIT,
                TimeSteppingScheme.CRANK_NICOLSON,
            )
            can_use_implicit = use_implicit and self.problem.state_space.ndim == 1

            if use_implicit and not can_use_implicit:
                if _ == 0:
                    logger.warning(
                        f"Implicit/CN schemes not yet supported for "
                        f"{self.problem.state_space.ndim}D problems; "
                        f"falling back to explicit."
                    )

            if can_use_implicit:
                # Implicit or Crank-Nicolson 1D step
                drift_1d = drift[:, 0] if drift.ndim > 1 else drift
                sigma_sq_1d = (
                    sigma_sq[:, 0] if sigma_sq is not None and sigma_sq.ndim > 1 else sigma_sq
                )
                cost_1d = cost.ravel()

                if scheme == TimeSteppingScheme.CRANK_NICOLSON:
                    if _ < self.config.rannacher_steps:
                        # Rannacher startup: two implicit half-steps
                        half_v = self._theta_step_1d(
                            old_v.ravel(),
                            cost_1d,
                            drift_1d,
                            sigma_sq_1d,
                            dt / 2.0,
                            theta=1.0,
                        )
                        half_v = self._apply_boundary_conditions(
                            half_v.reshape(old_v.shape)
                        ).ravel()
                        new_v = self._theta_step_1d(
                            half_v,
                            cost_1d,
                            drift_1d,
                            sigma_sq_1d,
                            dt / 2.0,
                            theta=1.0,
                        )
                    else:
                        # Crank-Nicolson step (theta=0.5)
                        new_v = self._theta_step_1d(
                            old_v.ravel(),
                            cost_1d,
                            drift_1d,
                            sigma_sq_1d,
                            dt,
                            theta=0.5,
                        )
                else:
                    # Fully implicit step (theta=1)
                    new_v = self._theta_step_1d(
                        old_v.ravel(),
                        cost_1d,
                        drift_1d,
                        sigma_sq_1d,
                        dt,
                        theta=1.0,
                    )
                new_v = new_v.reshape(old_v.shape)
            else:
                # Explicit scheme (or fallback for multi-D)
                if self.problem.time_horizon is not None:
                    new_v = self._update_value_finite_horizon(old_v, cost, drift, dt, sigma_sq)
                else:
                    advection = np.zeros_like(old_v)
                    for dim in range(drift.shape[-1]):
                        if dim >= len(self.problem.state_space.state_variables):
                            continue
                        drift_component = drift[..., dim]
                        advection += self._apply_upwind_scheme(old_v, drift_component, dim)

                    rhs = -self.problem.discount_rate * old_v + cost + advection
                    if sigma_sq is not None:
                        rhs += self._apply_diffusion_term(old_v, sigma_sq)
                    new_v = old_v + dt * rhs

            # Enforce boundary conditions after each time step
            new_v = self._apply_boundary_conditions(new_v)

            # Check for NaN/Inf after each inner step (#453)
            if not np.all(np.isfinite(new_v)):
                n_nan = int(np.sum(np.isnan(new_v)))
                n_inf = int(np.sum(np.isinf(new_v)))
                raise NumericalDivergenceError(
                    f"HJB solver diverged during policy evaluation "
                    f"(inner iteration {_}): "
                    f"{n_nan} NaN and {n_inf} Inf values detected. "
                    f"Value function range before step: "
                    f"[{float(np.nanmin(old_v)):.4e}, {float(np.nanmax(old_v)):.4e}]."
                )

            self.value_function = new_v

            # Check inner convergence
            if (
                np.max(np.abs(new_v - old_v))
                < self.config.tolerance * self.config.inner_tolerance_factor
            ):
                break

    def _precompute_upwind_diffs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Precompute forward and backward finite differences of the value function.

        Returns a list (one entry per dimension) of (fwd_diff_flat, bwd_diff_flat)
        arrays, each of shape (n_states,).  These encode the same upwind scheme
        used by ``_apply_upwind_scheme`` so that the advection term for an
        arbitrary drift field can be assembled as::

            advection = sum_dim( max(drift, 0)*fwd + min(drift, 0)*bwd )

        without recomputing the differences for every control candidate.
        """
        assert self.value_function is not None
        ndim = self.problem.state_space.ndim
        diffs: List[Tuple[np.ndarray, np.ndarray]] = []

        for dim in range(ndim):
            dx_array = self.dx[dim]
            n = self.value_function.shape[dim]

            # Broadcast dx along the correct axis
            dx_shape = [1] * ndim
            dx_shape[dim] = n - 1
            dx_bc = dx_array.reshape(dx_shape)

            hi = [slice(None)] * ndim
            lo = [slice(None)] * ndim
            hi[dim] = slice(1, None)  # indices 1..N-1
            lo[dim] = slice(None, -1)  # indices 0..N-2

            diff_vals = (self.value_function[tuple(hi)] - self.value_function[tuple(lo)]) / dx_bc

            # Forward difference: defined at indices 0..N-2, zero at N-1
            fwd = np.zeros_like(self.value_function)
            fwd[tuple(lo)] = diff_vals

            # Backward difference: defined at indices 1..N-1, zero at 0
            bwd = np.zeros_like(self.value_function)
            bwd[tuple(hi)] = diff_vals

            diffs.append((fwd.ravel(), bwd.ravel()))

        return diffs

    @staticmethod
    def _build_control_combos(control_samples: List[np.ndarray]) -> np.ndarray:
        """Build all control combinations as an (n_combos, n_controls) array.

        Uses np.meshgrid instead of itertools.product for efficient
        Cartesian product construction.
        """
        grids = np.meshgrid(*control_samples, indexing="ij")
        return np.column_stack([g.ravel() for g in grids])

    def _compute_chunk_size(self, n_combos: int, n_states: int, n_controls: int) -> int:
        """Determine batch size from memory budget.

        Each combo-state pair requires storage for controls, drift,
        cost, and advection arrays (~4 * n_dims * 8 bytes per pair).
        """
        ndim = self.problem.state_space.ndim
        budget_bytes = self.config.control_memory_budget_mb * 1024 * 1024
        # Estimate bytes per combo: states * (controls + drift + cost + advection) * 8
        bytes_per_combo = n_states * (n_controls + ndim + 1 + 1) * 8
        if bytes_per_combo == 0:
            return n_combos
        chunk = max(1, budget_bytes // bytes_per_combo)
        return min(chunk, n_combos)

    def _evaluate_and_update_best(
        self,
        combos: np.ndarray,
        state_points: np.ndarray,
        n_states: int,
        n_controls: int,
        ndim: int,
        fwd_arr: np.ndarray,
        bwd_arr: np.ndarray,
        d2V_flat: Optional[np.ndarray],
        best_values: np.ndarray,
        best_controls: np.ndarray,
    ) -> None:
        """Evaluate Hamiltonian for a batch of control combos and update best.

        Processes combos in chunks determined by the memory budget,
        vectorizing over both states and combos within each chunk.
        """
        chunk_size = self._compute_chunk_size(len(combos), n_states, n_controls)

        for start in range(0, len(combos), chunk_size):
            chunk = combos[start : start + chunk_size]
            n_chunk = len(chunk)

            # Tile state_points for all combos in chunk: (n_chunk * n_states, state_dim)
            states_tiled = np.tile(state_points, (n_chunk, 1))
            # Repeat each combo for all states: (n_chunk * n_states, n_controls)
            controls_tiled = np.repeat(chunk, n_states, axis=0)

            # Evaluate dynamics and running cost in one vectorized call
            drift = self.problem.dynamics(states_tiled, controls_tiled, 0.0)
            cost = self.problem.running_cost(states_tiled, controls_tiled, 0.0)

            # Reduce cost to 1D
            cost = np.asarray(cost)
            if cost.ndim > 1:
                if cost.shape[-1] > 1:
                    cost = np.mean(cost, axis=-1)
                else:
                    cost = cost[..., 0]
            cost = cost.flatten()

            # Reshape to (n_chunk, n_states)
            cost_2d = cost.reshape(n_chunk, n_states)

            # Compute advection: upwind scheme
            drift_flat = np.asarray(drift).reshape(n_chunk * n_states, -1)
            n_dims = min(drift_flat.shape[1], ndim)
            drift_pos = np.maximum(drift_flat[:, :n_dims], 0.0)
            drift_neg = np.minimum(drift_flat[:, :n_dims], 0.0)

            # Tile fwd/bwd arrays for all combos in chunk
            fwd_tiled = np.tile(fwd_arr[:, :n_dims], (n_chunk, 1))
            bwd_tiled = np.tile(bwd_arr[:, :n_dims], (n_chunk, 1))

            advection = np.sum(drift_pos * fwd_tiled + drift_neg * bwd_tiled, axis=1)
            advection_2d = advection.reshape(n_chunk, n_states)

            hamiltonian_2d = cost_2d + advection_2d

            # Add diffusion term
            if self.problem.diffusion is not None and d2V_flat is not None:
                sigma_sq = self.problem.diffusion(states_tiled, controls_tiled, 0.0)
                sigma_sq = np.asarray(sigma_sq).reshape(n_chunk * n_states, -1)
                n_diff = min(sigma_sq.shape[1], d2V_flat.shape[1], n_dims)
                diff_term = 0.5 * np.sum(
                    sigma_sq[:, :n_diff] * np.tile(d2V_flat[:, :n_diff], (n_chunk, 1)),
                    axis=1,
                )
                hamiltonian_2d += diff_term.reshape(n_chunk, n_states)

            # Find best combo per state within this chunk
            chunk_best_idx = np.argmax(hamiltonian_2d, axis=0)  # (n_states,)
            chunk_best_vals = hamiltonian_2d[chunk_best_idx, np.arange(n_states)]

            # Update global best
            improved = chunk_best_vals > best_values
            best_values[improved] = chunk_best_vals[improved]
            best_controls[improved] = chunk[chunk_best_idx[improved]]

    def _policy_improvement_loop(
        self,
        state_points: np.ndarray,
        n_states: int,
        n_controls: int,
        ndim: int,
        fwd_arr: np.ndarray,
        bwd_arr: np.ndarray,
        d2V_flat: Optional[np.ndarray],
        best_values: np.ndarray,
        best_controls: np.ndarray,
        control_samples: List[np.ndarray],
    ) -> None:
        """Legacy loop-based policy improvement (original implementation)."""
        for control_combo in itertools_product(*control_samples):
            control_array = np.array(control_combo)
            control_broadcast = np.tile(control_array, (n_states, 1))

            drift = self.problem.dynamics(state_points, control_broadcast, 0.0)
            cost = self.problem.running_cost(state_points, control_broadcast, 0.0)

            cost = np.asarray(cost)
            if cost.ndim > 1:
                if cost.shape[-1] > 1:
                    cost = np.mean(cost, axis=-1)
                else:
                    cost = cost[..., 0]
            cost = cost.flatten()

            drift_flat = np.asarray(drift).reshape(n_states, -1)
            n_dims = min(drift_flat.shape[1], ndim)
            drift_pos = np.maximum(drift_flat[:, :n_dims], 0.0)
            drift_neg = np.minimum(drift_flat[:, :n_dims], 0.0)
            advection_flat = np.sum(
                drift_pos * fwd_arr[:, :n_dims] + drift_neg * bwd_arr[:, :n_dims],
                axis=1,
            )

            hamiltonian = cost + advection_flat

            if self.problem.diffusion is not None and d2V_flat is not None:
                sigma_sq = self.problem.diffusion(state_points, control_broadcast, 0.0)
                sigma_sq = np.asarray(sigma_sq).reshape(n_states, -1)
                n_diff = min(sigma_sq.shape[1], d2V_flat.shape[1], n_dims)
                hamiltonian += 0.5 * np.sum(sigma_sq[:, :n_diff] * d2V_flat[:, :n_diff], axis=1)

            improved = hamiltonian > best_values
            best_values[improved] = hamiltonian[improved]
            best_controls[improved] = control_array

    def _policy_improvement_vectorized(
        self,
        state_points: np.ndarray,
        n_states: int,
        n_controls: int,
        ndim: int,
        fwd_arr: np.ndarray,
        bwd_arr: np.ndarray,
        d2V_flat: Optional[np.ndarray],
        best_values: np.ndarray,
        best_controls: np.ndarray,
        control_samples: List[np.ndarray],
    ) -> None:
        """Fully vectorized policy improvement over all control combos."""
        combos = self._build_control_combos(control_samples)
        self._evaluate_and_update_best(
            combos,
            state_points,
            n_states,
            n_controls,
            ndim,
            fwd_arr,
            bwd_arr,
            d2V_flat,
            best_values,
            best_controls,
        )

    def _policy_improvement_adaptive(
        self,
        state_points: np.ndarray,
        n_states: int,
        n_controls: int,
        ndim: int,
        fwd_arr: np.ndarray,
        bwd_arr: np.ndarray,
        d2V_flat: Optional[np.ndarray],
        best_values: np.ndarray,
        best_controls: np.ndarray,
        control_samples: List[np.ndarray],
    ) -> None:
        """Two-pass adaptive policy improvement: coarse search then local refinement."""
        # Pass 1: Coarse grid (every _COARSE_STRIDE-th point per control)
        coarse_samples = [s[::_COARSE_STRIDE] for s in control_samples]
        coarse_combos = self._build_control_combos(coarse_samples)

        self._evaluate_and_update_best(
            coarse_combos,
            state_points,
            n_states,
            n_controls,
            ndim,
            fwd_arr,
            bwd_arr,
            d2V_flat,
            best_values,
            best_controls,
        )

        # Pass 2: Refine around coarse optima
        # Find unique coarse optima (each row of best_controls is a combo)
        unique_optima = np.unique(best_controls, axis=0)

        # For each unique optimum, build a refined grid of nearby combos
        refined_combos_list = []
        for optimum in unique_optima:
            per_control_refined = []
            for j, full_grid in enumerate(control_samples):
                # Find the closest index in the full grid
                closest_idx = int(np.argmin(np.abs(full_grid - optimum[j])))
                lo = max(0, closest_idx - _REFINE_RADIUS)
                hi = min(len(full_grid), closest_idx + _REFINE_RADIUS + 1)
                per_control_refined.append(full_grid[lo:hi])
            local_combos = self._build_control_combos(per_control_refined)
            refined_combos_list.append(local_combos)

        if refined_combos_list:
            all_refined = np.vstack(refined_combos_list)
            # Deduplicate
            all_refined = np.unique(all_refined, axis=0)
            self._evaluate_and_update_best(
                all_refined,
                state_points,
                n_states,
                n_controls,
                ndim,
                fwd_arr,
                bwd_arr,
                d2V_flat,
                best_values,
                best_controls,
            )

    def _policy_improvement(self):
        """Improve policy by maximizing the Hamiltonian.

        H(x,u) = f(x,u) + drift(x,u)·∇V(x) + ½σ²(x,u)·∇²V(x)

        Dispatches to vectorized, adaptive, or loop-based strategy
        based on ``config.control_search_strategy``.  The precomputation
        of upwind finite differences is shared across all strategies.
        """
        if self.value_function is None or self.optimal_policy is None:
            return

        state_points = np.stack(self.problem.state_space.flat_grids, axis=-1)
        n_states = state_points.shape[0]
        n_controls = len(self.problem.control_variables)
        ndim = self.problem.state_space.ndim

        # Initialize tracking arrays
        best_values = np.full(n_states, -np.inf)
        best_controls = np.zeros((n_states, n_controls))

        # Compute second derivatives for diffusion term (independent of control)
        d2V_flat = None
        if self.problem.diffusion is not None:
            d2V = self._compute_second_derivatives(self.value_function)
            # Check for NaN in second derivatives (#453)
            if not np.all(np.isfinite(d2V)):
                logger.warning(
                    "NaN/Inf detected in second derivatives during policy improvement; "
                    "skipping policy update."
                )
                return
            d2V_flat = d2V.reshape(n_states, -1)

        # Precompute upwind finite differences of V (control-independent).
        # Each entry is (fwd_flat, bwd_flat) with shape (n_states,).
        upwind_diffs = self._precompute_upwind_diffs()

        # Stack into (n_states, ndim) for vectorized advection computation
        fwd_arr = np.column_stack([f for f, _ in upwind_diffs])  # (n_states, ndim)
        bwd_arr = np.column_stack([b for _, b in upwind_diffs])  # (n_states, ndim)

        # Get discrete control samples for each control variable
        control_samples = [cv.get_values() for cv in self.problem.control_variables]

        # Shared arguments for all strategies
        args = (
            state_points,
            n_states,
            n_controls,
            ndim,
            fwd_arr,
            bwd_arr,
            d2V_flat,
            best_values,
            best_controls,
            control_samples,
        )

        # Determine strategy
        strategy = self.config.control_search_strategy
        if strategy == "gradient":
            raise NotImplementedError(
                "Gradient-based control search is reserved for future implementation."
            )

        if strategy == "auto":
            n_combos = 1
            for s in control_samples:
                n_combos *= len(s)
            strategy = "vectorized" if n_combos <= _VECTORIZE_COMBO_THRESHOLD else "adaptive"

        if strategy == "vectorized":
            self._policy_improvement_vectorized(*args)
        elif strategy == "adaptive":
            self._policy_improvement_adaptive(*args)
        else:
            # "loop" or any unrecognized value falls back to legacy
            self._policy_improvement_loop(*args)

        # Write optimal controls back to policy arrays
        for j, cv in enumerate(self.problem.control_variables):
            self.optimal_policy[cv.name] = best_controls[:, j].reshape(
                self.problem.state_space.shape
            )

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

        # Evaluate HJB residual: |−ρV + f(x,u) + drift·∇V|
        drift = self.problem.dynamics(state_points, control_array, 0.0)
        cost = self.problem.running_cost(state_points, control_array, 0.0)

        v_flat = self.value_function.ravel()
        cost_flat = cost.ravel() if hasattr(cost, "ravel") else cost

        # Compute drift * grad_V using upwind scheme (consistent with PDE)
        drift_reshaped = drift.reshape(self.problem.state_space.shape + (-1,))
        advection = np.zeros(self.problem.state_space.shape)
        n_dims = min(drift_reshaped.shape[-1], len(self.problem.state_space.state_variables))
        for dim in range(n_dims):
            drift_component = drift_reshaped[..., dim]
            advection += self._apply_upwind_scheme(self.value_function, drift_component, dim)
        advection_flat = advection.ravel()

        # Include diffusion in residual if present
        diffusion_flat = np.zeros_like(v_flat)
        if self.problem.diffusion is not None:
            sigma_sq = self.problem.diffusion(state_points, control_array, 0.0)
            sigma_sq_reshaped = sigma_sq.reshape(self.problem.state_space.shape + (-1,))
            diffusion_term = self._apply_diffusion_term(self.value_function, sigma_sq_reshaped)
            diffusion_flat = diffusion_term.ravel()

        residual = np.abs(
            -self.problem.discount_rate * v_flat + cost_flat + advection_flat + diffusion_flat
        )

        # Check for NaN/Inf (#453)
        has_nan_inf = not np.all(np.isfinite(self.value_function))

        return {
            "has_nan_inf": has_nan_inf,
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

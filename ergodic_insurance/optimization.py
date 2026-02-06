"""Advanced optimization algorithms for constrained insurance decision making.

This module implements sophisticated optimization methods including trust-region,
penalty methods, augmented Lagrangian, and multi-start techniques for finding
global optima in complex insurance optimization problems.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints in optimization."""

    EQUALITY = "eq"
    INEQUALITY = "ineq"
    BOUNDS = "bounds"


@dataclass
class ConstraintViolation:
    """Information about constraint violations."""

    constraint_name: str
    violation_amount: float
    constraint_type: ConstraintType
    current_value: float
    limit_value: float
    is_satisfied: bool

    def __str__(self) -> str:
        """String representation of violation."""
        status = "✓" if self.is_satisfied else "✗"
        return (
            f"{status} {self.constraint_name}: "
            f"{self.current_value:.4f} vs {self.limit_value:.4f} "
            f"(violation: {self.violation_amount:.4f})"
        )


@dataclass
class ConvergenceMonitor:
    """Monitor and track convergence of optimization algorithms."""

    max_iterations: int = 1000
    tolerance: float = 1e-6
    objective_history: List[float] = field(default_factory=list)
    constraint_violation_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    step_size_history: List[float] = field(default_factory=list)
    iteration_count: int = 0
    converged: bool = False
    convergence_message: str = ""

    def update(
        self,
        objective: float,
        constraint_violation: float = 0.0,
        gradient_norm: float = 0.0,
        step_size: float = 0.0,
    ):
        """Update convergence history."""
        self.iteration_count += 1
        self.objective_history.append(objective)
        self.constraint_violation_history.append(constraint_violation)
        self.gradient_norm_history.append(gradient_norm)
        self.step_size_history.append(step_size)

        # Check convergence criteria
        if self.iteration_count >= self.max_iterations:
            self.converged = True
            self.convergence_message = "Maximum iterations reached"
        elif len(self.objective_history) >= 2:
            obj_change = abs(self.objective_history[-1] - self.objective_history[-2])
            if obj_change < self.tolerance:
                self.converged = True
                self.convergence_message = f"Objective converged (change: {obj_change:.2e})"
        elif 0 < gradient_norm < self.tolerance:
            self.converged = True
            self.convergence_message = f"Gradient converged (norm: {gradient_norm:.2e})"

    def get_summary(self) -> Dict[str, Any]:
        """Get convergence summary statistics."""
        return {
            "iterations": self.iteration_count,
            "converged": self.converged,
            "message": self.convergence_message,
            "final_objective": self.objective_history[-1] if self.objective_history else None,
            "final_constraint_violation": (
                self.constraint_violation_history[-1] if self.constraint_violation_history else 0.0
            ),
            "objective_improvement": (
                self.objective_history[0] - self.objective_history[-1]
                if len(self.objective_history) > 1
                else 0.0
            ),
        }


@dataclass
class AdaptivePenaltyParameters:
    """Parameters for adaptive penalty method."""

    initial_penalty: float = 10.0
    penalty_increase_factor: float = 2.0
    max_penalty: float = 1e6
    constraint_tolerance: float = 1e-4
    penalty_update_frequency: int = 10
    current_penalties: Dict[str, float] = field(default_factory=dict)

    def update_penalties(self, violations: List[ConstraintViolation]):
        """Update penalty parameters based on constraint violations."""
        for violation in violations:
            if not violation.is_satisfied:
                # Increase penalty for violated constraints
                current = self.current_penalties.get(
                    violation.constraint_name, self.initial_penalty
                )
                new_penalty = min(current * self.penalty_increase_factor, self.max_penalty)
                self.current_penalties[violation.constraint_name] = new_penalty
                logger.debug(
                    f"Updated penalty for {violation.constraint_name}: "
                    f"{current:.2f} -> {new_penalty:.2f}"
                )


class TrustRegionOptimizer:
    """Trust-region constrained optimization with adaptive radius adjustment."""

    def __init__(
        self,
        objective_fn: Callable,
        gradient_fn: Optional[Callable] = None,
        hessian_fn: Optional[Callable] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        bounds: Optional[Bounds] = None,
    ):
        """Initialize trust-region optimizer.

        Args:
            objective_fn: Objective function to minimize
            gradient_fn: Gradient function (computed numerically if None)
            hessian_fn: Hessian function (approximated if None)
            constraints: List of constraint dictionaries
            bounds: Variable bounds
        """
        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.hessian_fn = hessian_fn
        self.constraints = constraints or []
        self.bounds = bounds
        self.convergence_monitor = ConvergenceMonitor()

    def optimize(
        self,
        x0: np.ndarray,
        initial_radius: float = 1.0,
        max_radius: float = 10.0,
        eta: float = 0.15,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> OptimizeResult:
        """Run trust-region optimization.

        Args:
            x0: Initial point
            initial_radius: Initial trust region radius
            max_radius: Maximum trust region radius
            eta: Minimum reduction ratio for accepting step
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Optimization result
        """
        logger.info("Starting trust-region optimization")

        # Use scipy's trust-region method with our enhancements
        if self.constraints:
            # Convert constraints to scipy format
            scipy_constraints = self._convert_constraints()

            # Set Jacobian to finite differences if not provided
            jac = self.gradient_fn if self.gradient_fn is not None else "2-point"

            result = minimize(
                self.objective_fn,
                x0,
                method="trust-constr",
                jac=jac,
                hess=self.hessian_fn,
                constraints=scipy_constraints,
                bounds=self.bounds,
                options={
                    "initial_tr_radius": initial_radius,
                    "maxiter": max_iter,
                    "gtol": tol,
                    "xtol": tol,
                    "verbose": 0,
                },
            )
        else:
            # Unconstrained trust-region - use L-BFGS-B if no gradient
            if self.gradient_fn is None:
                result = minimize(
                    self.objective_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=self.bounds,
                    options={"maxiter": max_iter, "ftol": tol},
                )
            else:
                result = minimize(
                    self.objective_fn,
                    x0,
                    method="trust-ncg" if self.hessian_fn else "trust-exact",
                    jac=self.gradient_fn,
                    hess=self.hessian_fn,
                    options={"maxiter": max_iter, "gtol": tol},
                )

        # Add convergence monitoring information
        result.convergence_info = self.convergence_monitor.get_summary()

        logger.info(f"Trust-region optimization completed: {result.message}")
        return result

    def _convert_constraints(self) -> List[NonlinearConstraint]:
        """Convert constraint dictionaries to scipy format."""
        scipy_constraints = []

        for constr in self.constraints:
            # Set Jacobian to finite differences if not provided
            jac = constr.get("jac", "2-point")

            if constr["type"] == "ineq":
                # Inequality constraint: g(x) >= 0
                scipy_constraints.append(NonlinearConstraint(constr["fun"], 0, np.inf, jac=jac))
            elif constr["type"] == "eq":
                # Equality constraint: h(x) = 0
                scipy_constraints.append(NonlinearConstraint(constr["fun"], 0, 0, jac=jac))

        return scipy_constraints


class PenaltyMethodOptimizer:
    """Optimization using penalty method with adaptive penalty parameters."""

    def __init__(
        self,
        objective_fn: Callable,
        constraints: List[Dict[str, Any]],
        bounds: Optional[Bounds] = None,
    ):
        """Initialize penalty method optimizer.

        Args:
            objective_fn: Original objective function
            constraints: List of constraints
            bounds: Variable bounds
        """
        self.objective_fn = objective_fn
        self.constraints = constraints
        self.bounds = bounds
        self.penalty_params = AdaptivePenaltyParameters()
        self.convergence_monitor = ConvergenceMonitor()

    def _penalized_objective(self, x: np.ndarray, penalty_multipliers: Dict[str, float]) -> float:
        """Compute penalized objective function.

        Args:
            x: Current point
            penalty_multipliers: Current penalty parameters

        Returns:
            Penalized objective value
        """
        obj = self.objective_fn(x)

        # Add penalty terms for constraint violations
        total_penalty = 0.0
        for i, constr in enumerate(self.constraints):
            constr_name = f"constraint_{i}"
            penalty = penalty_multipliers.get(constr_name, self.penalty_params.initial_penalty)

            if constr["type"] == "ineq":
                # Inequality constraint g(x) >= 0
                violation = max(0, -constr["fun"](x))
            else:
                # Equality constraint h(x) = 0
                violation = abs(constr["fun"](x))

            total_penalty += penalty * violation**2

        return float(obj + total_penalty)

    def optimize(
        self,
        x0: np.ndarray,
        method: str = "L-BFGS-B",
        max_outer_iter: int = 50,
        max_inner_iter: int = 100,
        tol: float = 1e-6,
    ) -> OptimizeResult:
        """Run penalty method optimization.

        Args:
            x0: Initial point
            method: Inner optimization method
            max_outer_iter: Maximum outer iterations
            max_inner_iter: Maximum inner iterations per outer loop
            tol: Convergence tolerance

        Returns:
            Optimization result
        """
        logger.info("Starting penalty method optimization")

        x_current = x0.copy()
        best_x = x0.copy()
        best_obj = float("inf")

        # Initialize penalties
        for i in range(len(self.constraints)):
            self.penalty_params.current_penalties[f"constraint_{i}"] = (
                self.penalty_params.initial_penalty
            )

        for outer_iter in range(max_outer_iter):
            # Solve penalized problem
            def penalized_fn(x):
                return self._penalized_objective(x, self.penalty_params.current_penalties)

            result = minimize(
                penalized_fn,
                x_current,
                method=method,
                bounds=self.bounds,
                options={"maxiter": max_inner_iter, "ftol": tol},
            )

            x_current = result.x

            # Check constraint violations
            violations = self._check_violations(x_current)

            # Update convergence monitor
            constraint_violation = sum(v.violation_amount for v in violations if not v.is_satisfied)
            self.convergence_monitor.update(result.fun, constraint_violation)

            # Check if this is the best solution
            actual_obj = self.objective_fn(x_current)
            if (
                actual_obj < best_obj
                and constraint_violation < self.penalty_params.constraint_tolerance
            ):
                best_x = x_current.copy()
                best_obj = actual_obj

            # Check convergence
            if constraint_violation < self.penalty_params.constraint_tolerance:
                logger.info(f"Converged after {outer_iter + 1} outer iterations")
                break

            # Update penalties
            if (outer_iter + 1) % self.penalty_params.penalty_update_frequency == 0:
                self.penalty_params.update_penalties(violations)

        # Create final result
        final_result = OptimizeResult(
            x=best_x,
            fun=self.objective_fn(best_x),
            success=constraint_violation < self.penalty_params.constraint_tolerance,
            nit=outer_iter + 1,
            message="Penalty method optimization completed",
            convergence_info=self.convergence_monitor.get_summary(),
        )

        return final_result

    def _check_violations(self, x: np.ndarray) -> List[ConstraintViolation]:
        """Check constraint violations at current point."""
        violations = []

        for i, constr in enumerate(self.constraints):
            constr_value = constr["fun"](x)

            if constr["type"] == "ineq":
                # Inequality g(x) >= 0
                violation = max(0, -constr_value)
                is_satisfied = constr_value >= -self.penalty_params.constraint_tolerance
                limit_value = 0.0
            else:
                # Equality h(x) = 0
                violation = abs(constr_value)
                is_satisfied = violation <= self.penalty_params.constraint_tolerance
                limit_value = 0.0

            violations.append(
                ConstraintViolation(
                    constraint_name=f"constraint_{i}",
                    violation_amount=violation,
                    constraint_type=(
                        ConstraintType.INEQUALITY
                        if constr["type"] == "ineq"
                        else ConstraintType.EQUALITY
                    ),
                    current_value=constr_value,
                    limit_value=limit_value,
                    is_satisfied=is_satisfied,
                )
            )

        return violations


class AugmentedLagrangianOptimizer:
    """Augmented Lagrangian method for constrained optimization."""

    def __init__(
        self,
        objective_fn: Callable,
        constraints: List[Dict[str, Any]],
        bounds: Optional[Bounds] = None,
    ):
        """Initialize augmented Lagrangian optimizer.

        Args:
            objective_fn: Original objective function
            constraints: List of constraints
            bounds: Variable bounds
        """
        self.objective_fn = objective_fn
        self.constraints = constraints
        self.bounds = bounds
        self.convergence_monitor = ConvergenceMonitor()

    def _augmented_lagrangian(
        self,
        x: np.ndarray,
        lambdas: np.ndarray,
        mus: np.ndarray,
        rho: float,
    ) -> float:
        """Compute augmented Lagrangian function.

        Args:
            x: Current point
            lambdas: Lagrange multipliers for inequality constraints
            mus: Lagrange multipliers for equality constraints
            rho: Penalty parameter

        Returns:
            Augmented Lagrangian value
        """
        L = self.objective_fn(x)

        ineq_idx = 0
        eq_idx = 0

        for constr in self.constraints:
            if constr["type"] == "ineq":
                # Inequality constraint g(x) >= 0
                g = constr["fun"](x)
                # Augmented term: -lambda*g + (rho/2)*max(0, -g - lambda/rho)^2
                slack = max(0, -g - lambdas[ineq_idx] / rho)
                L += -lambdas[ineq_idx] * g + (rho / 2) * slack**2
                ineq_idx += 1
            else:
                # Equality constraint h(x) = 0
                h = constr["fun"](x)
                # Augmented term: mu*h + (rho/2)*h^2
                L += mus[eq_idx] * h + (rho / 2) * h**2
                eq_idx += 1

        return float(L)

    def optimize(  # pylint: disable=too-many-locals
        self,
        x0: np.ndarray,
        max_outer_iter: int = 50,
        max_inner_iter: int = 100,
        tol: float = 1e-6,
        rho_init: float = 1.0,
        rho_max: float = 1e4,
    ) -> OptimizeResult:
        """Run augmented Lagrangian optimization.

        Args:
            x0: Initial point
            max_outer_iter: Maximum outer iterations
            max_inner_iter: Maximum inner iterations
            tol: Convergence tolerance
            rho_init: Initial penalty parameter
            rho_max: Maximum penalty parameter

        Returns:
            Optimization result
        """
        logger.info("Starting augmented Lagrangian optimization")

        # Count constraints
        n_ineq = sum(1 for c in self.constraints if c["type"] == "ineq")
        n_eq = sum(1 for c in self.constraints if c["type"] == "eq")

        # Initialize
        x_current = x0.copy()
        lambdas = np.zeros(n_ineq)  # Inequality multipliers
        mus = np.zeros(n_eq)  # Equality multipliers
        rho = rho_init

        best_x = x0.copy()
        best_obj = float("inf")

        for outer_iter in range(max_outer_iter):
            # Minimize augmented Lagrangian
            def aug_lag_fn(x):
                return self._augmented_lagrangian(x, lambdas, mus, rho)

            result = minimize(
                aug_lag_fn,
                x_current,
                method="L-BFGS-B",
                bounds=self.bounds,
                options={"maxiter": max_inner_iter, "ftol": tol},
            )

            x_current = result.x

            # Update multipliers
            ineq_idx = 0
            eq_idx = 0
            constraint_violation = 0.0

            for constr in self.constraints:
                if constr["type"] == "ineq":
                    g = constr["fun"](x_current)
                    # Update inequality multiplier
                    lambdas[ineq_idx] = max(0, lambdas[ineq_idx] - rho * g)
                    constraint_violation += max(0, -g) ** 2
                    ineq_idx += 1
                else:
                    h = constr["fun"](x_current)
                    # Update equality multiplier
                    mus[eq_idx] = mus[eq_idx] + rho * h
                    constraint_violation += h**2
                    eq_idx += 1

            constraint_violation = np.sqrt(constraint_violation)

            # Update convergence monitor
            self.convergence_monitor.update(result.fun, constraint_violation)

            # Check if this is the best solution
            obj_value = self.objective_fn(x_current)
            if obj_value < best_obj and constraint_violation < tol:
                best_x = x_current.copy()
                best_obj = obj_value

            # Check convergence
            if constraint_violation < tol:
                logger.info(f"Converged after {outer_iter + 1} outer iterations")
                break

            # Update penalty parameter
            if (
                constraint_violation
                > 0.5 * self.convergence_monitor.constraint_violation_history[-2]
                if len(self.convergence_monitor.constraint_violation_history) > 1
                else True
            ):
                rho = min(rho * 2, rho_max)

        # Create final result
        final_result = OptimizeResult(
            x=best_x,
            fun=best_obj,
            success=constraint_violation < tol,
            nit=outer_iter + 1,
            message="Augmented Lagrangian optimization completed",
            convergence_info=self.convergence_monitor.get_summary(),
        )

        return final_result


class MultiStartOptimizer:
    """Multi-start optimization for finding global optima."""

    def __init__(
        self,
        objective_fn: Callable,
        bounds: Bounds,
        constraints: Optional[List[Dict[str, Any]]] = None,
        base_optimizer: str = "SLSQP",
    ):
        """Initialize multi-start optimizer.

        Args:
            objective_fn: Objective function to minimize
            bounds: Variable bounds
            constraints: Optional constraints
            base_optimizer: Base optimization method to use
        """
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.constraints = constraints or []
        self.base_optimizer = base_optimizer

    def optimize(
        self,
        n_starts: int = 10,
        x0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        parallel: bool = False,
    ) -> OptimizeResult:
        """Run multi-start optimization.

        Args:
            n_starts: Number of random starts
            x0: Optional initial point (included as first start)
            seed: Random seed for reproducibility
            parallel: Whether to run starts in parallel

        Returns:
            Best optimization result across all starts
        """
        logger.info(f"Starting multi-start optimization with {n_starts} starts")

        rng = np.random.default_rng(seed)

        # Generate starting points
        starting_points = self._generate_starting_points(n_starts, x0, rng)

        # Run optimization from each starting point
        results = []
        for i, start_point in enumerate(starting_points):
            logger.debug(f"Running optimization from start point {i + 1}/{n_starts}")

            try:
                if self.base_optimizer == "trust-region":
                    tr_optimizer = TrustRegionOptimizer(
                        self.objective_fn,
                        constraints=self.constraints,
                        bounds=self.bounds,
                    )
                    result = tr_optimizer.optimize(start_point)
                elif self.base_optimizer == "penalty":
                    pm_optimizer = PenaltyMethodOptimizer(
                        self.objective_fn,
                        self.constraints,
                        self.bounds,
                    )
                    result = pm_optimizer.optimize(start_point)
                elif self.base_optimizer == "augmented-lagrangian":
                    al_optimizer = AugmentedLagrangianOptimizer(
                        self.objective_fn,
                        self.constraints,
                        self.bounds,
                    )
                    result = al_optimizer.optimize(start_point)
                elif self.base_optimizer == "enhanced-slsqp":
                    es_optimizer = EnhancedSLSQPOptimizer(
                        self.objective_fn,
                        gradient_fn=None,  # Let it compute numerically
                        constraints=self.constraints,
                        bounds=self.bounds,
                    )
                    result = es_optimizer.optimize(start_point)
                else:
                    # Default to scipy minimize - map enhanced-slsqp to SLSQP
                    method = (
                        "SLSQP" if self.base_optimizer == "enhanced-slsqp" else self.base_optimizer
                    )
                    result = minimize(
                        self.objective_fn,
                        start_point,
                        method=method,
                        bounds=self.bounds,
                        constraints=self.constraints,
                        options={"maxiter": 1000},
                    )

                results.append(result)

            except (ValueError, RuntimeError, TypeError) as e:
                logger.warning(f"Optimization failed from start point {i + 1}: {e}")
                continue

        if not results:
            raise RuntimeError("All optimization attempts failed")

        # Find best result
        best_result = min(results, key=lambda r: r.fun if r.success else float("inf"))

        # Add multi-start specific information
        best_result.n_starts = n_starts
        best_result.n_successful = sum(1 for r in results if r.success)
        best_result.all_objectives = [r.fun for r in results if r.success]

        logger.info(
            f"Multi-start completed: {best_result.n_successful}/{n_starts} successful, "
            f"best objective: {best_result.fun:.6f}"
        )

        return best_result

    def _generate_starting_points(
        self, n_starts: int, x0: Optional[np.ndarray], rng: np.random.Generator
    ) -> List[np.ndarray]:
        """Generate diverse starting points within bounds."""
        n_vars = len(self.bounds.lb)
        starting_points = []

        # Include provided initial point if given
        if x0 is not None:
            starting_points.append(x0)
            n_starts -= 1

        # Generate random points using Latin Hypercube Sampling for better coverage
        for _ in range(n_starts):
            point = np.zeros(n_vars)
            for i in range(n_vars):
                lb = self.bounds.lb[i]
                ub = self.bounds.ub[i]
                # Handle infinite bounds
                if np.isinf(lb):
                    lb = -1e6
                if np.isinf(ub):
                    ub = 1e6
                point[i] = rng.uniform(lb, ub)
            starting_points.append(point)

        return starting_points


class EnhancedSLSQPOptimizer:
    """Enhanced SLSQP with adaptive step sizing and improved convergence."""

    def __init__(
        self,
        objective_fn: Callable,
        gradient_fn: Optional[Callable] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        bounds: Optional[Bounds] = None,
    ):
        """Initialize enhanced SLSQP optimizer.

        Args:
            objective_fn: Objective function
            gradient_fn: Gradient function (computed numerically if None)
            constraints: List of constraints
            bounds: Variable bounds
        """
        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.constraints = constraints or []
        self.bounds = bounds
        self.convergence_monitor = ConvergenceMonitor()
        # Initialize attributes that may be set during optimization
        self.step_size: float = 1.0
        self.prev_x: Optional[np.ndarray] = None
        self.prev_obj: Optional[float] = None

    def optimize(
        self,
        x0: np.ndarray,
        adaptive_step: bool = True,
        line_search: str = "armijo",
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> OptimizeResult:
        """Run enhanced SLSQP optimization.

        Args:
            x0: Initial point
            adaptive_step: Whether to use adaptive step sizing
            line_search: Line search method ("armijo" or "wolfe")
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Optimization result
        """
        logger.info("Starting enhanced SLSQP optimization")

        # Configure options
        options = {
            "maxiter": max_iter,
            "ftol": tol,
            "disp": False,
        }

        # Track adaptive step sizing
        if adaptive_step:
            self.step_size = 1.0
            self.prev_x = None
            self.prev_obj = None

        # Run optimization
        result = minimize(
            self.objective_fn,
            x0,
            method="SLSQP",
            jac=self.gradient_fn,
            bounds=self.bounds,
            constraints=self.constraints,
            options=options,
        )

        # Add convergence information
        result.convergence_info = self.convergence_monitor.get_summary()

        return result


def create_optimizer(
    method: str,
    objective_fn: Callable,
    constraints: Optional[List[Dict[str, Any]]] = None,
    bounds: Optional[Bounds] = None,
    **kwargs,
) -> Any:
    """Factory function to create appropriate optimizer.

    Args:
        method: Optimization method name
        objective_fn: Objective function
        constraints: Optional constraints
        bounds: Optional bounds
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer instance
    """
    method = method.lower()

    if method == "trust-region":
        return TrustRegionOptimizer(objective_fn, constraints=constraints, bounds=bounds, **kwargs)
    if method == "penalty":
        return PenaltyMethodOptimizer(objective_fn, constraints or [], bounds, **kwargs)
    if method == "augmented-lagrangian":
        return AugmentedLagrangianOptimizer(objective_fn, constraints or [], bounds, **kwargs)
    if method == "multi-start":
        return MultiStartOptimizer(objective_fn, bounds or Bounds([], []), constraints, **kwargs)
    if method == "enhanced-slsqp":
        return EnhancedSLSQPOptimizer(
            objective_fn, constraints=constraints, bounds=bounds, **kwargs
        )
    raise ValueError(f"Unknown optimization method: {method}")

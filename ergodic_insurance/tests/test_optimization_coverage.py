"""Coverage tests for optimization.py targeting specific uncovered lines.

Missing lines: 94-95, 228, 254-256, 657-662, 664-669, 671-676, 701-703, 706, 743, 745
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.optimize import Bounds

from ergodic_insurance.optimization import (
    AugmentedLagrangianOptimizer,
    ConvergenceMonitor,
    MultiStartOptimizer,
    PenaltyMethodOptimizer,
    TrustRegionOptimizer,
)


def rosenbrock(x):
    """Standard Rosenbrock function for optimization testing."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    """Gradient of Rosenbrock function."""
    dfdx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dfdx1 = 200 * (x[1] - x[0] ** 2)
    return np.array([dfdx0, dfdx1])


def simple_quadratic(x):
    """Simple quadratic for quick convergence tests."""
    return x[0] ** 2 + x[1] ** 2


class TestConvergenceMonitorGradient:
    """Tests for ConvergenceMonitor.update gradient convergence (lines 94-95)."""

    def test_gradient_convergence(self):
        """Lines 94-95: Convergence via small gradient norm."""
        monitor = ConvergenceMonitor(tolerance=1e-4, max_iterations=100)

        # First update sets objective but no convergence yet (need at least
        # one previous value for obj_change comparison, but gradient check
        # does not require it -- however, the elif only fires if the previous
        # condition about len(objective_history) >= 2 is False)
        monitor.update(objective=1.0, constraint_violation=0.0, gradient_norm=1e-5, step_size=0.1)
        assert monitor.converged is True
        assert "Gradient converged" in monitor.convergence_message

    def test_gradient_convergence_not_triggered_when_zero(self):
        """Gradient norm == 0 does not trigger gradient convergence (0 < 0 is False)."""
        monitor = ConvergenceMonitor(tolerance=1e-4, max_iterations=100)
        monitor.update(objective=1.0, constraint_violation=0.0, gradient_norm=0.0, step_size=0.1)
        # Should NOT converge from gradient (0 < gradient_norm requires > 0)
        assert monitor.converged is False

    def test_objective_convergence_takes_priority(self):
        """Objective change < tolerance triggers before gradient check."""
        monitor = ConvergenceMonitor(tolerance=1e-4, max_iterations=100)
        monitor.update(objective=1.0, constraint_violation=0.0, gradient_norm=1.0, step_size=0.1)
        monitor.update(objective=1.0, constraint_violation=0.0, gradient_norm=1e-5, step_size=0.1)
        assert monitor.converged is True
        assert "Objective converged" in monitor.convergence_message


class TestTrustRegionUnconstrainedWithGradient:
    """Tests for TrustRegionOptimizer unconstrained trust-exact (line 228)."""

    def test_unconstrained_with_gradient_and_hessian_trust_ncg(self):
        """Line 228: Trust-ncg method when both gradient and hessian provided."""
        optimizer = TrustRegionOptimizer(
            objective_fn=simple_quadratic,
            gradient_fn=lambda x: 2 * x,
            hessian_fn=lambda x: 2 * np.eye(2),
            constraints=[],
            bounds=None,
        )
        result = optimizer.optimize(
            x0=np.array([5.0, 3.0]),
            max_iter=100,
            tol=1e-6,
        )
        assert result.fun < 1.0

    def test_unconstrained_with_gradient_only_trust_exact(self):
        """Line 228: trust-exact path (gradient only, no hessian).

        This exercises the else branch at line 228 where hessian_fn is None,
        causing scipy to select trust-exact which requires a hessian.
        scipy raises an error since trust-exact needs a Hessian.
        """
        optimizer = TrustRegionOptimizer(
            objective_fn=simple_quadratic,
            gradient_fn=lambda x: 2 * x,
            hessian_fn=None,
            constraints=[],
            bounds=None,
        )
        # trust-exact requires a Hessian; this exercises the code path (line 228)
        # scipy will raise an error (ValueError or TypeError depending on version)
        with pytest.raises((ValueError, TypeError)):
            optimizer.optimize(
                x0=np.array([5.0, 3.0]),
                max_iter=100,
                tol=1e-6,
            )


class TestConvertConstraintsEquality:
    """Tests for _convert_constraints equality constraint (lines 254-256)."""

    def test_equality_constraint_converted(self):
        """Lines 254-256: Equality constraints are converted to NonlinearConstraint(0,0)."""
        constraints = [
            {"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0},
        ]
        optimizer = TrustRegionOptimizer(
            objective_fn=simple_quadratic,
            constraints=constraints,
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
        )
        scipy_constraints = optimizer._convert_constraints()
        assert len(scipy_constraints) == 1
        # Check it is an equality constraint (lb=0, ub=0)
        nc = scipy_constraints[0]
        assert nc.lb == 0
        assert nc.ub == 0

    def test_mixed_inequality_and_equality_constraints(self):
        """Both inequality and equality constraints are converted."""
        constraints = [
            {"type": "ineq", "fun": lambda x: x[0] - 1.0},
            {"type": "eq", "fun": lambda x: x[0] + x[1] - 2.0},
        ]
        optimizer = TrustRegionOptimizer(
            objective_fn=simple_quadratic,
            constraints=constraints,
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
        )
        scipy_constraints = optimizer._convert_constraints()
        assert len(scipy_constraints) == 2
        # First is inequality (lb=0, ub=inf)
        assert scipy_constraints[0].lb == 0
        assert scipy_constraints[0].ub == np.inf
        # Second is equality (lb=0, ub=0)
        assert scipy_constraints[1].lb == 0
        assert scipy_constraints[1].ub == 0


class TestMultiStartTrustRegionBase:
    """Tests for MultiStartOptimizer with trust-region base (lines 657-662)."""

    def test_trust_region_base_optimizer(self):
        """Lines 657-662: Multi-start with trust-region base optimizer."""
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=[],
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
            base_optimizer="trust-region",
        )
        result = optimizer.optimize(
            x0=np.array([5.0, 3.0]),
            n_starts=3,
            seed=42,
        )
        assert result.fun < 5.0
        assert hasattr(result, "n_starts")


class TestMultiStartPenaltyBase:
    """Tests for MultiStartOptimizer with penalty base (lines 664-669)."""

    def test_penalty_base_optimizer(self):
        """Lines 664-669: Multi-start with penalty method base optimizer."""
        constraints = [
            {"type": "ineq", "fun": lambda x: 10.0 - x[0] - x[1]},
        ]
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=constraints,
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
            base_optimizer="penalty",
        )
        result = optimizer.optimize(
            x0=np.array([3.0, 3.0]),
            n_starts=2,
            seed=42,
        )
        assert hasattr(result, "n_starts")


class TestMultiStartAugmentedLagrangianBase:
    """Tests for MultiStartOptimizer with augmented-lagrangian base (lines 671-676)."""

    def test_augmented_lagrangian_base_optimizer(self):
        """Lines 671-676: Multi-start with augmented Lagrangian base optimizer."""
        constraints = [
            {"type": "ineq", "fun": lambda x: 10.0 - x[0] - x[1]},
        ]
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=constraints,
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
            base_optimizer="augmented-lagrangian",
        )
        result = optimizer.optimize(
            x0=np.array([3.0, 3.0]),
            n_starts=2,
            seed=42,
        )
        assert hasattr(result, "n_starts")


class TestMultiStartAllFail:
    """Tests for MultiStartOptimizer all attempts failing (lines 701-703, 706)."""

    def test_all_starts_fail_raises_runtime_error(self):
        """Lines 701-703: All optimization attempts failing raises RuntimeError."""

        def always_fail(x):
            raise ValueError("intentional failure")

        optimizer = MultiStartOptimizer(
            objective_fn=always_fail,
            constraints=[],
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
            base_optimizer="trust-region",
        )
        with pytest.raises(RuntimeError, match="All optimization attempts failed"):
            optimizer.optimize(
                x0=np.array([1.0, 1.0]),
                n_starts=3,
                seed=42,
            )

    def test_some_starts_fail_picks_best(self):
        """Line 706: When some starts fail, best result among successes is returned.

        MultiStartOptimizer uses scipy.minimize directly, so we patch minimize
        to make some starts fail.
        """
        from scipy.optimize import OptimizeResult

        call_count = 0
        original_minimize = __import__("scipy.optimize", fromlist=["minimize"]).minimize

        def patched_minimize(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ValueError("intentional failure on even calls")
            return original_minimize(*args, **kwargs)

        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=[],
            bounds=Bounds([0.0, 0.0], [5.0, 5.0]),
        )
        call_count = 0
        with patch("ergodic_insurance.optimization.minimize", patched_minimize):
            result = optimizer.optimize(
                x0=np.array([1.0, 1.0]),
                n_starts=4,
                seed=42,
            )
        # Should succeed with at least some starts
        assert hasattr(result, "n_starts")
        assert result.n_successful >= 1


class TestGenerateStartingPointsInfiniteBounds:
    """Tests for _generate_starting_points with infinite bounds (lines 743, 745)."""

    def test_infinite_lower_bound_clipped(self):
        """Line 743: Infinite lower bound replaced with -1e6."""
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=[],
            bounds=Bounds([-np.inf, 0.0], [10.0, 10.0]),
        )
        rng = np.random.default_rng(42)
        points = optimizer._generate_starting_points(3, None, rng)
        assert len(points) == 3
        for point in points:
            assert point[0] >= -1e6
            assert point[0] <= 10.0

    def test_infinite_upper_bound_clipped(self):
        """Line 745: Infinite upper bound replaced with 1e6."""
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=[],
            bounds=Bounds([0.0, 0.0], [np.inf, 10.0]),
        )
        rng = np.random.default_rng(42)
        points = optimizer._generate_starting_points(3, None, rng)
        assert len(points) == 3
        for point in points:
            assert point[0] >= 0.0
            assert point[0] <= 1e6

    def test_both_infinite_bounds_clipped(self):
        """Both infinite bounds replaced with +/-1e6."""
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=[],
            bounds=Bounds([-np.inf, -np.inf], [np.inf, np.inf]),
        )
        rng = np.random.default_rng(42)
        points = optimizer._generate_starting_points(5, None, rng)
        assert len(points) == 5
        for point in points:
            assert -1e6 <= point[0] <= 1e6
            assert -1e6 <= point[1] <= 1e6

    def test_initial_point_included(self):
        """x0 is included as first starting point."""
        optimizer = MultiStartOptimizer(
            objective_fn=simple_quadratic,
            constraints=[],
            bounds=Bounds([0.0, 0.0], [10.0, 10.0]),
        )
        rng = np.random.default_rng(42)
        x0 = np.array([2.0, 3.0])
        points = optimizer._generate_starting_points(3, x0, rng)
        assert len(points) == 3
        np.testing.assert_array_equal(points[0], x0)

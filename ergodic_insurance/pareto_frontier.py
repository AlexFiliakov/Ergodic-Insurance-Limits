"""Pareto frontier analysis for multi-objective optimization.

This module provides comprehensive tools for generating, analyzing, and visualizing
Pareto frontiers in multi-objective optimization problems, particularly focused on
insurance optimization trade-offs between ROE, risk, and costs.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Types of objectives in multi-objective optimization."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Objective:
    """Definition of an optimization objective.

    Attributes:
        name: Name of the objective (e.g., 'ROE', 'risk', 'cost')
        type: Whether to maximize or minimize this objective
        weight: Weight for weighted sum method (0-1)
        normalize: Whether to normalize this objective
        bounds: Optional bounds for this objective as (min, max)
    """

    name: str
    type: ObjectiveType
    weight: float = 1.0
    normalize: bool = True
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier.

    Attributes:
        objectives: Dictionary of objective values
        decision_variables: The decision variables that produce these objectives
        is_dominated: Whether this point is dominated by another
        crowding_distance: Crowding distance metric for this point
        trade_offs: Trade-off ratios with neighboring points
    """

    objectives: Dict[str, float]
    decision_variables: np.ndarray
    is_dominated: bool = False
    crowding_distance: float = 0.0
    trade_offs: Dict[str, float] = field(default_factory=dict)

    def dominates(self, other: "ParetoPoint", objectives: List[Objective]) -> bool:
        """Check if this point dominates another point.

        Args:
            other: Another Pareto point to compare
            objectives: List of objectives to consider

        Returns:
            True if this point dominates the other
        """
        at_least_one_better = False
        for obj in objectives:
            self_val = self.objectives[obj.name]
            other_val = other.objectives[obj.name]

            if obj.type == ObjectiveType.MAXIMIZE:
                if self_val < other_val:
                    return False
                if self_val > other_val:
                    at_least_one_better = True
            else:  # MINIMIZE
                if self_val > other_val:
                    return False
                if self_val < other_val:
                    at_least_one_better = True

        return at_least_one_better


class ParetoFrontier:
    """Generator and analyzer for Pareto frontiers.

    This class provides methods for generating Pareto frontiers using various
    algorithms and analyzing the resulting trade-offs.
    """

    def __init__(
        self,
        objectives: List[Objective],
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Pareto frontier generator.

        Args:
            objectives: List of objectives to optimize
            objective_function: Function that returns objective values given decision variables
            bounds: Bounds for decision variables
            constraints: Optional constraints for optimization
            seed: Optional random seed for reproducibility
        """
        self.objectives = objectives
        self.objective_function = objective_function
        self.bounds = bounds
        self.constraints = constraints or []
        self.frontier_points: List[ParetoPoint] = []
        self._rng = np.random.default_rng(seed)
        self._validate_objectives()

    def _validate_objectives(self) -> None:
        """Validate objective definitions."""
        if not self.objectives:
            raise ValueError("At least one objective must be defined")

        names = [obj.name for obj in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError("Objective names must be unique")

        for obj in self.objectives:
            if not 0 <= obj.weight <= 1:
                raise ValueError(f"Objective weight must be in [0, 1], got {obj.weight}")

    def generate_weighted_sum(self, n_points: int = 50, method: str = "SLSQP") -> List[ParetoPoint]:
        """Generate Pareto frontier using weighted sum method.

        Args:
            n_points: Number of points to generate on the frontier
            method: Optimization method to use

        Returns:
            List of Pareto points forming the frontier
        """
        points = []

        if len(self.objectives) == 2:
            # For bi-objective, vary weights systematically
            weights = np.linspace(0, 1, n_points)
            for w1 in weights:
                w2 = 1 - w1
                point = self._optimize_weighted_sum([w1, w2], method)
                if point is not None:
                    points.append(point)
        else:
            # For many objectives, use random weights
            for _ in range(n_points):
                weights = self._rng.dirichlet(np.ones(len(self.objectives)))
                point = self._optimize_weighted_sum(weights.tolist(), method)
                if point is not None:
                    points.append(point)

        # Filter dominated points
        self.frontier_points = self._filter_dominated_points(points)
        self._calculate_crowding_distances()
        self._calculate_trade_offs()

        return self.frontier_points

    def generate_epsilon_constraint(
        self, n_points: int = 50, method: str = "SLSQP"
    ) -> List[ParetoPoint]:
        """Generate Pareto frontier using epsilon-constraint method.

        Args:
            n_points: Number of points to generate
            method: Optimization method to use

        Returns:
            List of Pareto points forming the frontier
        """
        if len(self.objectives) < 2:
            raise ValueError("Epsilon-constraint requires at least 2 objectives")

        points = []
        primary_obj = self.objectives[0]
        constraint_objs = self.objectives[1:]

        # Find bounds for constraint objectives
        epsilon_ranges = []
        for obj in constraint_objs:
            # Optimize for this objective alone to find its range
            temp_weights = [0.0] * len(self.objectives)
            idx = self.objectives.index(obj)
            temp_weights[idx] = 1.0
            point = self._optimize_weighted_sum(temp_weights, method)
            if point:
                epsilon_ranges.append(
                    (point.objectives[obj.name], point.objectives[obj.name] * 1.5)
                )
            else:
                epsilon_ranges.append((0, 1))

        # Generate grid of epsilon values
        if len(constraint_objs) == 1:
            epsilons = np.linspace(epsilon_ranges[0][0], epsilon_ranges[0][1], n_points)
            for eps in epsilons:
                point = self._optimize_epsilon_constraint(
                    primary_obj, {constraint_objs[0].name: eps}, method
                )
                if point is not None:
                    points.append(point)
        else:
            # For multiple constraints, use random sampling
            for _ in range(n_points):
                epsilon_dict = {}
                for i, obj in enumerate(constraint_objs):
                    epsilon_dict[obj.name] = self._rng.uniform(
                        epsilon_ranges[i][0], epsilon_ranges[i][1]
                    )
                point = self._optimize_epsilon_constraint(primary_obj, epsilon_dict, method)
                if point is not None:
                    points.append(point)

        # Filter dominated points
        self.frontier_points = self._filter_dominated_points(points)
        self._calculate_crowding_distances()
        self._calculate_trade_offs()

        return self.frontier_points

    def generate_evolutionary(
        self, n_generations: int = 100, population_size: int = 50
    ) -> List[ParetoPoint]:
        """Generate Pareto frontier using evolutionary algorithm.

        Args:
            n_generations: Number of generations for evolution
            population_size: Size of population in each generation

        Returns:
            List of Pareto points forming the frontier
        """

        def multi_objective_wrapper(x):
            obj_vals = self.objective_function(x)
            # Convert to minimization problem
            result = []
            for obj in self.objectives:
                val = obj_vals[obj.name]
                if obj.type == ObjectiveType.MAXIMIZE:
                    result.append(-val)  # Negate for maximization
                else:
                    result.append(val)
            return result

        # Use differential evolution with multiple runs
        points = []
        for _ in range(population_size):
            # Random weights for this run
            weights = self._rng.dirichlet(np.ones(len(self.objectives)))

            def weighted_objective(x, w=weights):
                obj_vals = multi_objective_wrapper(x)
                return np.dot(w, obj_vals)

            result = differential_evolution(
                weighted_objective,
                self.bounds,
                maxiter=n_generations,
                popsize=15,
                seed=int(self._rng.integers(0, 10000)),
            )

            if result.success:
                obj_vals = self.objective_function(result.x)
                point = ParetoPoint(
                    objectives=obj_vals,
                    decision_variables=result.x,
                )
                points.append(point)

        # Filter dominated points
        self.frontier_points = self._filter_dominated_points(points)
        self._calculate_crowding_distances()
        self._calculate_trade_offs()

        return self.frontier_points

    def _optimize_weighted_sum(self, weights: List[float], method: str) -> Optional[ParetoPoint]:
        """Optimize using weighted sum of objectives.

        Args:
            weights: Weights for each objective
            method: Optimization method

        Returns:
            Pareto point if optimization successful, None otherwise
        """

        def weighted_objective(x):
            obj_vals = self.objective_function(x)
            weighted_sum = 0
            for obj, weight in zip(self.objectives, weights):
                val = obj_vals[obj.name]
                # Normalize if requested
                if obj.normalize and obj.bounds:
                    val = (val - obj.bounds[0]) / (obj.bounds[1] - obj.bounds[0])
                # Convert to minimization
                if obj.type == ObjectiveType.MAXIMIZE:
                    val = -val
                weighted_sum += weight * val
            return weighted_sum

        # Initial guess
        x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])

        result = minimize(
            weighted_objective,
            x0,
            method=method,
            bounds=self.bounds,
            constraints=self.constraints,
        )

        if result.success:
            obj_vals = self.objective_function(result.x)
            return ParetoPoint(
                objectives=obj_vals,
                decision_variables=result.x,
            )
        return None

    def _optimize_epsilon_constraint(
        self, primary_obj: Objective, epsilon_constraints: Dict[str, float], method: str
    ) -> Optional[ParetoPoint]:
        """Optimize using epsilon-constraint method.

        Args:
            primary_obj: Primary objective to optimize
            epsilon_constraints: Constraints on other objectives
            method: Optimization method

        Returns:
            Pareto point if optimization successful, None otherwise
        """

        def primary_objective(x):
            obj_vals = self.objective_function(x)
            val = obj_vals[primary_obj.name]
            if primary_obj.type == ObjectiveType.MAXIMIZE:
                return -val
            return val

        # Add epsilon constraints
        additional_constraints = []
        for obj_name, epsilon_val in epsilon_constraints.items():
            obj = next(o for o in self.objectives if o.name == obj_name)

            def make_constraint(name, eps, obj_type):
                if obj_type == ObjectiveType.MAXIMIZE:
                    return {
                        "type": "ineq",
                        "fun": lambda x, n=name, e=eps: self.objective_function(x)[n] - e,
                    }
                return {
                    "type": "ineq",
                    "fun": lambda x, n=name, e=eps: e - self.objective_function(x)[n],
                }

            additional_constraints.append(make_constraint(obj_name, epsilon_val, obj.type))

        all_constraints = self.constraints + additional_constraints

        # Initial guess
        x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])

        result = minimize(
            primary_objective,
            x0,
            method=method,
            bounds=self.bounds,
            constraints=all_constraints,
        )

        if result.success:
            obj_vals = self.objective_function(result.x)
            return ParetoPoint(
                objectives=obj_vals,
                decision_variables=result.x,
            )
        return None

    def _filter_dominated_points(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Filter out dominated points to get true Pareto frontier.

        Args:
            points: List of candidate points

        Returns:
            List of non-dominated points
        """
        if not points:
            return []

        # Mark dominated points
        for i, point1 in enumerate(points):
            for j, point2 in enumerate(points):
                if i != j and not point1.is_dominated:
                    if point2.dominates(point1, self.objectives):
                        point1.is_dominated = True
                        break

        # Return non-dominated points
        return [p for p in points if not p.is_dominated]

    def _calculate_crowding_distances(self) -> None:
        """Calculate crowding distances for frontier points."""
        if len(self.frontier_points) <= 2:
            for point in self.frontier_points:
                point.crowding_distance = float("inf")
            return

        n_points = len(self.frontier_points)
        distances = np.zeros(n_points)

        for obj in self.objectives:
            # Sort points by this objective
            sorted_points = sorted(
                enumerate(self.frontier_points),
                key=lambda x, o=obj: x[1].objectives[o.name],  # type: ignore[misc]
            )

            # Boundary points get infinite distance
            distances[sorted_points[0][0]] = float("inf")
            distances[sorted_points[-1][0]] = float("inf")

            # Calculate distances for interior points
            obj_range = (
                sorted_points[-1][1].objectives[obj.name] - sorted_points[0][1].objectives[obj.name]
            )

            if obj_range > 0:
                for i in range(1, n_points - 1):
                    prev_val = sorted_points[i - 1][1].objectives[obj.name]
                    next_val = sorted_points[i + 1][1].objectives[obj.name]
                    distances[sorted_points[i][0]] += (next_val - prev_val) / obj_range

        # Assign crowding distances
        for i, point in enumerate(self.frontier_points):
            point.crowding_distance = distances[i]

    def _calculate_trade_offs(self) -> None:
        """Calculate trade-off ratios between neighboring points."""
        if len(self.frontier_points) < 2:
            return

        # For 2D frontiers, calculate direct trade-offs
        if len(self.objectives) == 2:
            sorted_points = sorted(
                self.frontier_points,
                key=lambda p: p.objectives[self.objectives[0].name],
            )

            for i in range(len(sorted_points) - 1):
                p1, p2 = sorted_points[i], sorted_points[i + 1]
                obj1_diff = (
                    p2.objectives[self.objectives[0].name] - p1.objectives[self.objectives[0].name]
                )
                obj2_diff = (
                    p2.objectives[self.objectives[1].name] - p1.objectives[self.objectives[1].name]
                )

                if abs(obj1_diff) > 1e-10:
                    trade_off = obj2_diff / obj1_diff
                    p1.trade_offs[
                        f"{self.objectives[1].name}_per_{self.objectives[0].name}"
                    ] = trade_off

    def calculate_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """Calculate hypervolume indicator for the Pareto frontier.

        Args:
            reference_point: Reference point for hypervolume calculation

        Returns:
            Hypervolume value
        """
        if not self.frontier_points:
            return 0.0

        # Set reference point if not provided
        if reference_point is None:
            reference_point = {}
            for obj in self.objectives:
                values = [p.objectives[obj.name] for p in self.frontier_points]
                if obj.type == ObjectiveType.MAXIMIZE:
                    reference_point[obj.name] = min(values) * 0.9
                else:
                    reference_point[obj.name] = max(values) * 1.1

        # For 2D, use simple rectangle sum
        if len(self.objectives) == 2:
            return self._calculate_2d_hypervolume(reference_point)

        # For higher dimensions, use Monte Carlo approximation
        return self._calculate_nd_hypervolume_monte_carlo(reference_point)

    def _calculate_2d_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Calculate hypervolume for 2D frontier.

        Args:
            reference_point: Reference point

        Returns:
            Hypervolume value
        """
        obj_names = [obj.name for obj in self.objectives]

        # Sort points by first objective
        sorted_points = sorted(
            self.frontier_points,
            key=lambda p: p.objectives[obj_names[0]],
        )

        hypervolume = 0.0
        prev_obj2 = reference_point[obj_names[1]]

        for point in sorted_points:
            obj1 = point.objectives[obj_names[0]]
            obj2 = point.objectives[obj_names[1]]

            # Calculate contribution
            if self.objectives[0].type == ObjectiveType.MAXIMIZE:
                width = obj1 - reference_point[obj_names[0]]
            else:
                width = reference_point[obj_names[0]] - obj1

            if self.objectives[1].type == ObjectiveType.MAXIMIZE:
                height = obj2 - prev_obj2
            else:
                height = prev_obj2 - obj2

            if width > 0 and height > 0:
                hypervolume += width * height

            prev_obj2 = obj2

        return hypervolume

    def _calculate_nd_hypervolume_monte_carlo(
        self, reference_point: Dict[str, float], n_samples: int = 10000
    ) -> float:
        """Calculate hypervolume using Monte Carlo approximation for n-D.

        Args:
            reference_point: Reference point
            n_samples: Number of Monte Carlo samples

        Returns:
            Approximate hypervolume value
        """
        # Define bounding box
        bounds = {}
        for obj in self.objectives:
            values = [p.objectives[obj.name] for p in self.frontier_points]
            if obj.type == ObjectiveType.MAXIMIZE:
                bounds[obj.name] = (reference_point[obj.name], max(values))
            else:
                bounds[obj.name] = (min(values), reference_point[obj.name])

        # Generate random samples
        dominated_count = 0
        for _ in range(n_samples):
            sample = {}
            for obj in self.objectives:
                sample[obj.name] = self._rng.uniform(bounds[obj.name][0], bounds[obj.name][1])

            # Check if sample is dominated by any frontier point
            for point in self.frontier_points:
                is_dominated = True
                for obj in self.objectives:
                    if obj.type == ObjectiveType.MAXIMIZE:
                        if sample[obj.name] > point.objectives[obj.name]:
                            is_dominated = False
                            break
                    else:
                        if sample[obj.name] < point.objectives[obj.name]:
                            is_dominated = False
                            break

                if is_dominated:
                    dominated_count += 1
                    break

        # Calculate volume
        total_volume = 1.0
        for obj in self.objectives:
            total_volume *= bounds[obj.name][1] - bounds[obj.name][0]

        return (dominated_count / n_samples) * total_volume

    def get_knee_points(self, n_knees: int = 1) -> List[ParetoPoint]:
        """Find knee points on the Pareto frontier.

        Knee points represent good trade-offs where small improvements in one
        objective require large sacrifices in others.

        Args:
            n_knees: Number of knee points to identify

        Returns:
            List of knee points
        """
        if not self.frontier_points:
            return []

        if len(self.frontier_points) <= n_knees:
            return self.frontier_points.copy()

        # Convert frontier to matrix
        obj_matrix = np.array(
            [[p.objectives[obj.name] for obj in self.objectives] for p in self.frontier_points]
        )

        # Normalize objectives
        normalized = np.zeros_like(obj_matrix)
        for i, obj in enumerate(self.objectives):
            col = obj_matrix[:, i]
            if obj.type == ObjectiveType.MAXIMIZE:
                normalized[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-10)
            else:
                normalized[:, i] = (col.max() - col) / (col.max() - col.min() + 1e-10)

        # Find extreme points (ideal and nadir)
        ideal = normalized.max(axis=0)

        # Calculate distances to ideal point
        distances = np.linalg.norm(ideal - normalized, axis=1)

        # Find knee points as those with smallest distances
        knee_indices = np.argsort(distances)[:n_knees]

        return [self.frontier_points[i] for i in knee_indices]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert frontier points to pandas DataFrame.

        Returns:
            DataFrame with objectives and decision variables
        """
        if not self.frontier_points:
            return pd.DataFrame()

        data = []
        for point in self.frontier_points:
            row = point.objectives.copy()
            row["crowding_distance"] = point.crowding_distance
            row["is_dominated"] = point.is_dominated
            # Add decision variables
            for i, val in enumerate(point.decision_variables):
                row[f"decision_var_{i}"] = val
            data.append(row)

        return pd.DataFrame(data)

"""Coverage tests for pareto_frontier.py targeting specific uncovered lines.

Missing lines: 187, 206, 219-227, 256, 336, 356, 366, 396, 408, 460,
494, 539, 544, 620, 623, 635, 657
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.pareto_frontier import Objective, ObjectiveType, ParetoFrontier, ParetoPoint


def simple_objective_function(x):
    """Simple bi-objective function: maximize return, minimize risk."""
    return {
        "return": float(x[0] * 0.1 - x[0] ** 2 * 0.001),
        "risk": float(x[0] * 0.05 + 0.1),
    }


def tri_objective_function(x):
    """Three-objective function for higher-dimensional tests."""
    return {
        "return": float(x[0] * 0.1),
        "risk": float(x[0] * 0.05 + 0.1),
        "cost": float(x[0] * 0.02 + 0.5),
    }


@pytest.fixture
def bi_objectives():
    """Create bi-objective list."""
    return [
        Objective(name="return", type=ObjectiveType.MAXIMIZE, weight=0.5),
        Objective(name="risk", type=ObjectiveType.MINIMIZE, weight=0.5),
    ]


@pytest.fixture
def tri_objectives():
    """Create tri-objective list."""
    return [
        Objective(name="return", type=ObjectiveType.MAXIMIZE, weight=0.33),
        Objective(name="risk", type=ObjectiveType.MINIMIZE, weight=0.33),
        Objective(name="cost", type=ObjectiveType.MINIMIZE, weight=0.34),
    ]


@pytest.fixture
def frontier(bi_objectives):
    """Create a ParetoFrontier for bi-objective optimization."""
    return ParetoFrontier(
        objectives=bi_objectives,
        objective_function=simple_objective_function,
        bounds=[(0.1, 50.0)],
        seed=42,
    )


class TestParetoFrontierValidation:
    """Tests for _validate_objectives."""

    def test_empty_objectives_raises(self):
        """No objectives raises ValueError."""
        with pytest.raises(ValueError, match="At least one objective"):
            ParetoFrontier(
                objectives=[],
                objective_function=simple_objective_function,
                bounds=[(0, 10)],
            )

    def test_duplicate_objective_names_raises(self):
        """Duplicate names raise ValueError."""
        objs = [
            Objective(name="return", type=ObjectiveType.MAXIMIZE),
            Objective(name="return", type=ObjectiveType.MINIMIZE),
        ]
        with pytest.raises(ValueError, match="unique"):
            ParetoFrontier(
                objectives=objs,
                objective_function=simple_objective_function,
                bounds=[(0, 10)],
            )


class TestEpsilonConstraint:
    """Tests for generate_epsilon_constraint (lines 187, 206, 219-227)."""

    def test_epsilon_constraint_single_objective_raises(self):
        """Line 187: Less than 2 objectives raises ValueError."""
        objs = [Objective(name="return", type=ObjectiveType.MAXIMIZE)]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"return": x[0]},
            bounds=[(0, 10)],
            seed=42,
        )
        with pytest.raises(ValueError, match="at least 2 objectives"):
            frontier.generate_epsilon_constraint(n_points=5)

    def test_epsilon_constraint_fallback_range(self, bi_objectives):
        """Line 206: When optimization fails, use default range (0, 1)."""

        def failing_obj(x):
            return {"return": float(x[0]), "risk": float(x[0])}

        frontier = ParetoFrontier(
            objectives=bi_objectives,
            objective_function=failing_obj,
            bounds=[(0, 10)],
            seed=42,
        )
        points = frontier.generate_epsilon_constraint(n_points=5)
        # Should handle gracefully even if some optimizations fail
        assert isinstance(points, list)

    def test_epsilon_constraint_multi_objective(self, tri_objectives):
        """Lines 219-227: Multi-constraint epsilon method uses random sampling."""
        frontier = ParetoFrontier(
            objectives=tri_objectives,
            objective_function=tri_objective_function,
            bounds=[(0.1, 50.0)],
            seed=42,
        )
        points = frontier.generate_epsilon_constraint(n_points=10)
        assert isinstance(points, list)


class TestGenerateEvolutionary:
    """Tests for generate_evolutionary (line 256)."""

    def test_evolutionary_generation(self, bi_objectives):
        """Line 256: Evolutionary algorithm generates points."""
        frontier = ParetoFrontier(
            objectives=bi_objectives,
            objective_function=simple_objective_function,
            bounds=[(0.1, 50.0)],
            seed=42,
        )
        points = frontier.generate_evolutionary(n_generations=5, population_size=3)
        assert isinstance(points, list)


class TestOptimizeWeightedSum:
    """Tests for _optimize_weighted_sum failure (line 336)."""

    def test_optimization_failure_returns_none(self):
        """Line 336: Failed optimization returns None when minimize reports success=False."""

        def failing_objective(x):
            return {"return": float(x[0]), "risk": float(x[0] ** 2)}

        objs = [
            Objective(name="return", type=ObjectiveType.MAXIMIZE),
            Objective(name="risk", type=ObjectiveType.MINIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=failing_objective,
            bounds=[(0, 10)],
            seed=42,
        )

        # Patch minimize to always return success=False
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.x = np.array([5.0])

        with patch("ergodic_insurance.pareto_frontier.minimize", return_value=mock_result):
            points = frontier.generate_weighted_sum(n_points=3)
        # All optimizations failed -> should return empty list
        assert isinstance(points, list)
        assert len(points) == 0


class TestEpsilonConstraintMinimize:
    """Tests for _optimize_epsilon_constraint minimize branch (lines 356, 366)."""

    def test_epsilon_constraint_minimize_primary(self):
        """Lines 356, 366: Minimize primary objective in epsilon constraint."""
        objs = [
            Objective(name="risk", type=ObjectiveType.MINIMIZE, weight=0.5),
            Objective(name="return", type=ObjectiveType.MAXIMIZE, weight=0.5),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=simple_objective_function,
            bounds=[(0.1, 50.0)],
            seed=42,
        )
        points = frontier.generate_epsilon_constraint(n_points=5)
        assert isinstance(points, list)


class TestEpsilonConstraintOptimizationFailure:
    """Tests for _optimize_epsilon_constraint returning None (line 396)."""

    def test_failed_epsilon_constraint_returns_none(self, bi_objectives):
        """Line 396: Failed epsilon constraint returns None."""
        # Use constraints that are impossible to satisfy
        frontier = ParetoFrontier(
            objectives=bi_objectives,
            objective_function=simple_objective_function,
            bounds=[(0.1, 50.0)],
            constraints=[{"type": "eq", "fun": lambda x: x[0] - 999}],
            seed=42,
        )
        points = frontier.generate_epsilon_constraint(n_points=5)
        # Should handle failures gracefully
        assert isinstance(points, list)


class TestFilterDominatedPoints:
    """Tests for _filter_dominated_points (line 408)."""

    def test_empty_points_returns_empty(self, frontier):
        """Line 408: Empty input returns empty list."""
        result = frontier._filter_dominated_points([])
        assert result == []


class TestCalculateTradeOffs:
    """Tests for _calculate_trade_offs (line 460)."""

    def test_single_point_no_trade_offs(self, frontier):
        """Line 460: Fewer than 2 points means no trade-offs."""
        frontier.frontier_points = [
            ParetoPoint(
                objectives={"return": 5.0, "risk": 1.0},
                decision_variables=np.array([10.0]),
            )
        ]
        frontier._calculate_trade_offs()
        # Should run without error; single point has no trade-offs


class TestCalculateHypervolume:
    """Tests for calculate_hypervolume (line 494)."""

    def test_empty_frontier_returns_zero(self, frontier):
        """Line 494: Empty frontier returns 0.0 hypervolume."""
        frontier.frontier_points = []
        hv = frontier.calculate_hypervolume()
        assert hv == 0.0

    def test_2d_hypervolume_with_reference(self, frontier):
        """Lines 539, 544: 2D hypervolume calculation."""
        frontier.frontier_points = [
            ParetoPoint(
                objectives={"return": 5.0, "risk": 1.0},
                decision_variables=np.array([10.0]),
            ),
            ParetoPoint(
                objectives={"return": 3.0, "risk": 0.5},
                decision_variables=np.array([5.0]),
            ),
        ]
        hv = frontier.calculate_hypervolume(reference_point={"return": 0.0, "risk": 5.0})
        assert hv >= 0.0


class TestNdHypervolumeMonteCarlo:
    """Tests for _calculate_nd_hypervolume_monte_carlo."""

    def test_3d_hypervolume(self, tri_objectives):
        """Higher-dimensional hypervolume uses Monte Carlo."""
        frontier = ParetoFrontier(
            objectives=tri_objectives,
            objective_function=tri_objective_function,
            bounds=[(0.1, 50.0)],
            seed=42,
        )
        frontier.frontier_points = [
            ParetoPoint(
                objectives={"return": 5.0, "risk": 1.0, "cost": 0.5},
                decision_variables=np.array([10.0]),
            ),
            ParetoPoint(
                objectives={"return": 3.0, "risk": 0.5, "cost": 1.0},
                decision_variables=np.array([5.0]),
            ),
        ]
        hv = frontier.calculate_hypervolume()
        assert hv >= 0.0


class TestGetKneePoints:
    """Tests for get_knee_points knee-detection methods."""

    def test_empty_frontier_returns_empty(self, frontier):
        """Empty frontier returns empty list."""
        frontier.frontier_points = []
        knees = frontier.get_knee_points()
        assert knees == []

    def test_fewer_points_than_knees(self, frontier):
        """Fewer points than requested knees returns all points."""
        frontier.frontier_points = [
            ParetoPoint(
                objectives={"return": 5.0, "risk": 1.0},
                decision_variables=np.array([10.0]),
            ),
        ]
        knees = frontier.get_knee_points(n_knees=5)
        assert len(knees) == 1

    def test_knee_points_with_minimize_objective(self):
        """Knee calculation handles MINIMIZE objectives."""
        objs = [
            Objective(name="risk", type=ObjectiveType.MINIMIZE),
            Objective(name="cost", type=ObjectiveType.MINIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"risk": x[0], "cost": 10 - x[0]},
            bounds=[(1, 9)],
            seed=42,
        )
        frontier.frontier_points = [
            ParetoPoint(objectives={"risk": 2.0, "cost": 8.0}, decision_variables=np.array([2.0])),
            ParetoPoint(objectives={"risk": 5.0, "cost": 5.0}, decision_variables=np.array([5.0])),
            ParetoPoint(objectives={"risk": 8.0, "cost": 2.0}, decision_variables=np.array([8.0])),
        ]
        knees = frontier.get_knee_points(n_knees=1)
        assert len(knees) == 1

    def test_invalid_method_raises(self, frontier):
        """Invalid method name raises ValueError."""
        frontier.frontier_points = [
            ParetoPoint(
                objectives={"return": 1.0, "risk": 1.0}, decision_variables=np.array([1.0])
            ),
            ParetoPoint(
                objectives={"return": 2.0, "risk": 2.0}, decision_variables=np.array([2.0])
            ),
            ParetoPoint(
                objectives={"return": 3.0, "risk": 3.0}, decision_variables=np.array([3.0])
            ),
        ]
        with pytest.raises(ValueError, match="method must be one of"):
            frontier.get_knee_points(method="invalid")

    def test_topsis_returns_closest_to_ideal(self):
        """TOPSIS method returns the point closest to the ideal point."""
        objs = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"f1": x[0], "f2": 1 - x[0]},
            bounds=[(0, 1)],
            seed=42,
        )
        # Asymmetric frontier: first four points are collinear near
        # the f1-axis, then a jump to f2-dominant territory.
        frontier.frontier_points = [
            ParetoPoint(objectives={"f1": 10.0, "f2": 1.0}, decision_variables=np.array([1.0])),
            ParetoPoint(objectives={"f1": 9.0, "f2": 2.0}, decision_variables=np.array([2.0])),
            ParetoPoint(objectives={"f1": 8.0, "f2": 3.0}, decision_variables=np.array([3.0])),
            ParetoPoint(objectives={"f1": 7.0, "f2": 4.0}, decision_variables=np.array([4.0])),
            ParetoPoint(objectives={"f1": 3.0, "f2": 9.0}, decision_variables=np.array([5.0])),
            ParetoPoint(objectives={"f1": 1.0, "f2": 10.0}, decision_variables=np.array([6.0])),
        ]
        knees = frontier.get_knee_points(n_knees=1, method="topsis")
        # (7,4) is closest to ideal in normalized space
        assert knees[0].objectives["f1"] == 7.0
        assert knees[0].objectives["f2"] == 4.0

    def test_perpendicular_distance_known_knee(self):
        """Perpendicular distance method finds the geometric knee.

        On a convex quarter-circle frontier the midpoint at 45 degrees
        has the maximum perpendicular distance from the line connecting
        the two extreme points.
        """
        objs = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"f1": x[0], "f2": 1 - x[0]},
            bounds=[(0, 1)],
            seed=42,
        )
        # Quarter-circle: x² + y² = 1
        import math

        frontier.frontier_points = [
            ParetoPoint(
                objectives={"f1": math.cos(a), "f2": math.sin(a)},
                decision_variables=np.array([a]),
            )
            for a in [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]
        ]
        knees = frontier.get_knee_points(n_knees=1, method="perpendicular_distance")
        # The 45-degree point (cos π/4, sin π/4) has the largest
        # perpendicular distance from the line (1,0)–(0,1).
        assert abs(knees[0].objectives["f1"] - math.cos(math.pi / 4)) < 1e-9
        assert abs(knees[0].objectives["f2"] - math.sin(math.pi / 4)) < 1e-9

    def test_perpendicular_distance_differs_from_topsis(self):
        """Perpendicular distance and TOPSIS identify different knees
        on an asymmetric frontier.
        """
        objs = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"f1": x[0], "f2": 1 - x[0]},
            bounds=[(0, 1)],
            seed=42,
        )
        # Collinear segment near f1-axis, then a single outlier near f2.
        frontier.frontier_points = [
            ParetoPoint(objectives={"f1": 10.0, "f2": 1.0}, decision_variables=np.array([1.0])),
            ParetoPoint(objectives={"f1": 9.0, "f2": 2.0}, decision_variables=np.array([2.0])),
            ParetoPoint(objectives={"f1": 8.0, "f2": 3.0}, decision_variables=np.array([3.0])),
            ParetoPoint(objectives={"f1": 7.0, "f2": 4.0}, decision_variables=np.array([4.0])),
            ParetoPoint(objectives={"f1": 3.0, "f2": 9.0}, decision_variables=np.array([5.0])),
            ParetoPoint(objectives={"f1": 1.0, "f2": 10.0}, decision_variables=np.array([6.0])),
        ]

        knee_pd = frontier.get_knee_points(n_knees=1, method="perpendicular_distance")
        knee_tp = frontier.get_knee_points(n_knees=1, method="topsis")

        # Perpendicular distance finds (3,9) — farthest from the
        # extreme-to-extreme line.  TOPSIS finds (7,4) — closest to
        # ideal.  They must differ.
        assert knee_pd[0].objectives["f1"] == 3.0
        assert knee_pd[0].objectives["f2"] == 9.0
        assert knee_tp[0].objectives["f1"] == 7.0
        assert knee_tp[0].objectives["f2"] == 4.0

    def test_angle_method_known_knee(self):
        """Angle method finds the sharpest bend on an L-shaped frontier."""
        objs = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"f1": x[0], "f2": 1 - x[0]},
            bounds=[(0, 1)],
            seed=42,
        )
        # L-shaped frontier: nearly flat along f1, sharp bend at (7,6),
        # then nearly flat along f2.
        frontier.frontier_points = [
            ParetoPoint(objectives={"f1": 10.0, "f2": 1.0}, decision_variables=np.array([1.0])),
            ParetoPoint(objectives={"f1": 9.0, "f2": 1.5}, decision_variables=np.array([2.0])),
            ParetoPoint(objectives={"f1": 8.0, "f2": 2.0}, decision_variables=np.array([3.0])),
            ParetoPoint(objectives={"f1": 7.0, "f2": 6.0}, decision_variables=np.array([4.0])),
            ParetoPoint(objectives={"f1": 2.0, "f2": 8.0}, decision_variables=np.array([5.0])),
            ParetoPoint(objectives={"f1": 1.5, "f2": 9.0}, decision_variables=np.array([6.0])),
            ParetoPoint(objectives={"f1": 1.0, "f2": 10.0}, decision_variables=np.array([7.0])),
        ]
        knees = frontier.get_knee_points(n_knees=1, method="angle")
        # The sharpest bend is at (7,6)
        assert knees[0].objectives["f1"] == 7.0
        assert knees[0].objectives["f2"] == 6.0

    def test_multiple_knees(self):
        """Requesting n_knees > 1 returns multiple distinct knee points."""
        objs = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"f1": x[0], "f2": 1 - x[0]},
            bounds=[(0, 1)],
            seed=42,
        )
        frontier.frontier_points = [
            ParetoPoint(objectives={"f1": 10.0, "f2": 1.0}, decision_variables=np.array([1.0])),
            ParetoPoint(objectives={"f1": 9.0, "f2": 1.5}, decision_variables=np.array([2.0])),
            ParetoPoint(objectives={"f1": 8.0, "f2": 2.0}, decision_variables=np.array([3.0])),
            ParetoPoint(objectives={"f1": 7.0, "f2": 6.0}, decision_variables=np.array([4.0])),
            ParetoPoint(objectives={"f1": 2.0, "f2": 8.0}, decision_variables=np.array([5.0])),
            ParetoPoint(objectives={"f1": 1.5, "f2": 9.0}, decision_variables=np.array([6.0])),
            ParetoPoint(objectives={"f1": 1.0, "f2": 10.0}, decision_variables=np.array([7.0])),
        ]
        knees = frontier.get_knee_points(n_knees=3, method="perpendicular_distance")
        assert len(knees) == 3
        assert len(set(id(k) for k in knees)) == 3  # all distinct

    def test_angle_method_two_points(self):
        """Angle method with only 2 points (no interior) returns one point."""
        objs = [
            Objective(name="f1", type=ObjectiveType.MAXIMIZE),
            Objective(name="f2", type=ObjectiveType.MAXIMIZE),
        ]
        frontier = ParetoFrontier(
            objectives=objs,
            objective_function=lambda x: {"f1": x[0], "f2": 1 - x[0]},
            bounds=[(0, 1)],
            seed=42,
        )
        frontier.frontier_points = [
            ParetoPoint(objectives={"f1": 10.0, "f2": 1.0}, decision_variables=np.array([1.0])),
            ParetoPoint(objectives={"f1": 1.0, "f2": 10.0}, decision_variables=np.array([2.0])),
            ParetoPoint(objectives={"f1": 5.0, "f2": 5.0}, decision_variables=np.array([3.0])),
        ]
        # n_knees=1 but only 2 interior points after sorting; angle
        # method should still return exactly 1 knee.
        knees = frontier.get_knee_points(n_knees=1, method="angle")
        assert len(knees) == 1


class TestToDataFrame:
    """Tests for to_dataframe (line 657)."""

    def test_empty_frontier_returns_empty_df(self, frontier):
        """Line 657: Empty frontier returns empty DataFrame."""
        frontier.frontier_points = []
        df = frontier.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_frontier_to_dataframe(self, frontier):
        """Non-empty frontier converts to DataFrame."""
        frontier.frontier_points = [
            ParetoPoint(
                objectives={"return": 5.0, "risk": 1.0},
                decision_variables=np.array([10.0]),
                crowding_distance=1.5,
                is_dominated=False,
            ),
        ]
        df = frontier.to_dataframe()
        assert "return" in df.columns
        assert "risk" in df.columns
        assert "crowding_distance" in df.columns
        assert "decision_var_0" in df.columns

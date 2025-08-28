"""Tests for the Pareto frontier analysis module."""

import numpy as np
import pytest

from ergodic_insurance.src.pareto_frontier import (
    Objective,
    ObjectiveType,
    ParetoFrontier,
    ParetoPoint,
)


class TestParetoPoint:
    """Test ParetoPoint class."""

    def test_pareto_point_creation(self):
        """Test creating a Pareto point."""
        objectives = {"ROE": 0.15, "risk": 0.02, "cost": 100000}
        decision_vars = np.array([0.5, 1.0, 0.3])

        point = ParetoPoint(objectives=objectives, decision_variables=decision_vars)

        assert point.objectives == objectives
        assert np.array_equal(point.decision_variables, decision_vars)
        assert not point.is_dominated
        assert point.crowding_distance == 0.0

    def test_dominance_maximize(self):
        """Test dominance checking for maximization objectives."""
        objectives = [
            Objective("ROE", ObjectiveType.MAXIMIZE),
            Objective("growth", ObjectiveType.MAXIMIZE),
        ]

        # Point 1 dominates Point 2 (both objectives better)
        point1 = ParetoPoint(
            objectives={"ROE": 0.20, "growth": 0.10},
            decision_variables=np.array([1.0]),
        )
        point2 = ParetoPoint(
            objectives={"ROE": 0.15, "growth": 0.08},
            decision_variables=np.array([0.5]),
        )

        assert point1.dominates(point2, objectives)
        assert not point2.dominates(point1, objectives)

    def test_dominance_minimize(self):
        """Test dominance checking for minimization objectives."""
        objectives = [
            Objective("risk", ObjectiveType.MINIMIZE),
            Objective("cost", ObjectiveType.MINIMIZE),
        ]

        # Point 1 dominates Point 2 (both objectives better - lower)
        point1 = ParetoPoint(
            objectives={"risk": 0.01, "cost": 50000},
            decision_variables=np.array([1.0]),
        )
        point2 = ParetoPoint(
            objectives={"risk": 0.02, "cost": 60000},
            decision_variables=np.array([0.5]),
        )

        assert point1.dominates(point2, objectives)
        assert not point2.dominates(point1, objectives)

    def test_dominance_mixed(self):
        """Test dominance with mixed objective types."""
        objectives = [
            Objective("ROE", ObjectiveType.MAXIMIZE),
            Objective("risk", ObjectiveType.MINIMIZE),
        ]

        # Point 1 dominates Point 2
        point1 = ParetoPoint(
            objectives={"ROE": 0.20, "risk": 0.01},
            decision_variables=np.array([1.0]),
        )
        point2 = ParetoPoint(
            objectives={"ROE": 0.15, "risk": 0.02},
            decision_variables=np.array([0.5]),
        )

        assert point1.dominates(point2, objectives)
        assert not point2.dominates(point1, objectives)

    def test_non_dominance(self):
        """Test non-dominated points."""
        objectives = [
            Objective("ROE", ObjectiveType.MAXIMIZE),
            Objective("risk", ObjectiveType.MINIMIZE),
        ]

        # Neither point dominates the other (trade-off)
        point1 = ParetoPoint(
            objectives={"ROE": 0.20, "risk": 0.02},
            decision_variables=np.array([1.0]),
        )
        point2 = ParetoPoint(
            objectives={"ROE": 0.15, "risk": 0.01},
            decision_variables=np.array([0.5]),
        )

        assert not point1.dominates(point2, objectives)
        assert not point2.dominates(point1, objectives)


class TestObjective:
    """Test Objective class."""

    def test_objective_creation(self):
        """Test creating an objective."""
        obj = Objective(
            name="ROE",
            type=ObjectiveType.MAXIMIZE,
            weight=0.7,
            normalize=True,
            bounds=(0.0, 0.3),
        )

        assert obj.name == "ROE"
        assert obj.type == ObjectiveType.MAXIMIZE
        assert obj.weight == 0.7
        assert obj.normalize
        assert obj.bounds == (0.0, 0.3)

    def test_objective_defaults(self):
        """Test objective default values."""
        obj = Objective(name="risk", type=ObjectiveType.MINIMIZE)

        assert obj.weight == 1.0
        assert obj.normalize
        assert obj.bounds is None


class TestParetoFrontier:
    """Test ParetoFrontier class."""

    @pytest.fixture
    def simple_bi_objective(self):
        """Create a simple bi-objective optimization problem."""
        objectives = [
            Objective("f1", ObjectiveType.MINIMIZE),
            Objective("f2", ObjectiveType.MINIMIZE),
        ]

        def objective_function(x):
            # Simple convex Pareto front: f1 = x, f2 = 1/x
            return {"f1": x[0], "f2": 1.0 / (x[0] + 0.1)}

        bounds = [(0.1, 2.0)]

        return ParetoFrontier(objectives, objective_function, bounds)

    @pytest.fixture
    def multi_objective(self):
        """Create a multi-objective optimization problem."""
        objectives = [
            Objective("ROE", ObjectiveType.MAXIMIZE, weight=0.4),
            Objective("risk", ObjectiveType.MINIMIZE, weight=0.3),
            Objective("cost", ObjectiveType.MINIMIZE, weight=0.3),
        ]

        def objective_function(x):
            # ROE increases with investment but so does risk
            # Cost is proportional to safety measures
            return {
                "ROE": 0.1 + 0.2 * x[0] - 0.05 * x[1],
                "risk": 0.05 * x[0] + 0.01,
                "cost": 50000 + 100000 * x[1],
            }

        bounds = [(0.0, 1.0), (0.0, 1.0)]

        return ParetoFrontier(objectives, objective_function, bounds)

    def test_frontier_creation(self, simple_bi_objective):
        """Test creating a Pareto frontier."""
        frontier = simple_bi_objective

        assert len(frontier.objectives) == 2
        assert len(frontier.bounds) == 1
        assert frontier.frontier_points == []

    def test_invalid_objectives(self):
        """Test validation of objectives."""
        # No objectives
        with pytest.raises(ValueError, match="At least one objective"):
            ParetoFrontier([], lambda x: {}, [(0, 1)])

        # Duplicate names
        objectives = [
            Objective("f1", ObjectiveType.MINIMIZE),
            Objective("f1", ObjectiveType.MAXIMIZE),
        ]
        with pytest.raises(ValueError, match="unique"):
            ParetoFrontier(objectives, lambda x: {}, [(0, 1)])

        # Invalid weight
        objectives = [Objective("f1", ObjectiveType.MINIMIZE, weight=1.5)]
        with pytest.raises(ValueError, match="weight must be in"):
            ParetoFrontier(objectives, lambda x: {}, [(0, 1)])

    def test_weighted_sum_generation(self, simple_bi_objective):
        """Test generating frontier using weighted sum method."""
        frontier = simple_bi_objective
        points = frontier.generate_weighted_sum(n_points=10)

        assert len(points) > 0
        assert all(isinstance(p, ParetoPoint) for p in points)
        assert all(not p.is_dominated for p in points)

    def test_epsilon_constraint_generation(self, simple_bi_objective):
        """Test generating frontier using epsilon-constraint method."""
        frontier = simple_bi_objective
        points = frontier.generate_epsilon_constraint(n_points=10)

        assert len(points) > 0
        assert all(isinstance(p, ParetoPoint) for p in points)
        assert all(not p.is_dominated for p in points)

    def test_evolutionary_generation(self, simple_bi_objective):
        """Test generating frontier using evolutionary algorithm."""
        frontier = simple_bi_objective
        points = frontier.generate_evolutionary(n_generations=10, population_size=10)

        assert len(points) > 0
        assert all(isinstance(p, ParetoPoint) for p in points)

    def test_dominated_filtering(self):
        """Test filtering of dominated points."""
        objectives = [
            Objective("f1", ObjectiveType.MINIMIZE),
            Objective("f2", ObjectiveType.MINIMIZE),
        ]

        def objective_function(x):
            return {"f1": x[0], "f2": x[1]}

        frontier = ParetoFrontier(objectives, objective_function, [(0, 1), (0, 1)])

        # Create test points - some dominated, some not
        points = [
            ParetoPoint({"f1": 0.1, "f2": 0.9}, np.array([0.1, 0.9])),
            ParetoPoint({"f1": 0.5, "f2": 0.5}, np.array([0.5, 0.5])),
            ParetoPoint({"f1": 0.9, "f2": 0.1}, np.array([0.9, 0.1])),
            ParetoPoint({"f1": 0.6, "f2": 0.6}, np.array([0.6, 0.6])),  # Dominated
        ]

        filtered = frontier._filter_dominated_points(points)

        assert len(filtered) == 3  # One point should be dominated
        assert all(not p.is_dominated for p in filtered)

    def test_crowding_distance_calculation(self, simple_bi_objective):
        """Test crowding distance calculation."""
        frontier = simple_bi_objective
        points = frontier.generate_weighted_sum(n_points=10)

        # Check that boundary points have infinite distance
        boundary_points = [p for p in points if p.crowding_distance == float("inf")]
        assert len(boundary_points) >= 2 if len(points) > 2 else len(boundary_points) == len(points)

        # Check that interior points have finite distance
        if len(points) > 2:
            interior_points = [p for p in points if p.crowding_distance != float("inf")]
            assert all(p.crowding_distance >= 0 for p in interior_points)

    def test_trade_off_calculation(self, simple_bi_objective):
        """Test trade-off calculation between objectives."""
        frontier = simple_bi_objective
        points = frontier.generate_weighted_sum(n_points=10)

        # For bi-objective problems, trade-offs should be calculated
        if len(points) > 1:
            points_with_trade_offs = [p for p in points if p.trade_offs]
            assert len(points_with_trade_offs) > 0

    def test_hypervolume_2d(self, simple_bi_objective):
        """Test hypervolume calculation for 2D frontier."""
        frontier = simple_bi_objective
        points = frontier.generate_weighted_sum(n_points=10)

        # Calculate hypervolume with default reference point
        hv = frontier.calculate_hypervolume()
        assert hv > 0

        # Calculate with custom reference point
        ref_point = {"f1": 3.0, "f2": 3.0}
        hv_custom = frontier.calculate_hypervolume(ref_point)
        assert hv_custom > 0

    def test_hypervolume_nd(self, multi_objective):
        """Test hypervolume calculation for n-D frontier."""
        frontier = multi_objective
        points = frontier.generate_weighted_sum(n_points=20)

        # Calculate hypervolume (uses Monte Carlo for 3D)
        hv = frontier.calculate_hypervolume()
        assert hv >= 0  # Can be 0 for poor frontiers

    def test_knee_points(self, simple_bi_objective):
        """Test knee point identification."""
        frontier = simple_bi_objective
        points = frontier.generate_weighted_sum(n_points=20)

        knees = frontier.get_knee_points(n_knees=3)

        assert len(knees) <= 3
        assert len(knees) <= len(points)
        assert all(k in points for k in knees)

    def test_to_dataframe(self, simple_bi_objective):
        """Test conversion to pandas DataFrame."""
        frontier = simple_bi_objective
        points = frontier.generate_weighted_sum(n_points=10)

        df = frontier.to_dataframe()

        assert not df.empty
        assert len(df) == len(points)
        assert "f1" in df.columns
        assert "f2" in df.columns
        assert "crowding_distance" in df.columns
        assert "is_dominated" in df.columns
        assert "decision_var_0" in df.columns

    def test_constraint_handling(self):
        """Test optimization with constraints."""
        objectives = [
            Objective("f1", ObjectiveType.MINIMIZE),
            Objective("f2", ObjectiveType.MINIMIZE),
        ]

        def objective_function(x):
            return {"f1": x[0] ** 2, "f2": (x[0] - 2) ** 2}

        # Add constraint: x >= 0.5
        constraints = [{"type": "ineq", "fun": lambda x: x[0] - 0.5}]

        frontier = ParetoFrontier(objectives, objective_function, [(0.0, 3.0)], constraints)

        points = frontier.generate_weighted_sum(n_points=10)

        # All solutions should satisfy the constraint
        assert all(p.decision_variables[0] >= 0.5 - 1e-6 for p in points)

    def test_real_insurance_scenario(self):
        """Test with a realistic insurance optimization scenario."""
        objectives = [
            Objective("ROE", ObjectiveType.MAXIMIZE, bounds=(0, 0.3)),
            Objective("bankruptcy_risk", ObjectiveType.MINIMIZE, bounds=(0, 0.1)),
        ]

        def insurance_objective(x):
            # x[0] = retention ratio (0-1)
            # x[1] = premium multiplier (1-3)

            retention = x[0]
            premium_mult = x[1]

            # Simple model: higher retention increases ROE but also risk
            # Higher premiums reduce risk but also ROE
            roe = 0.15 * retention - 0.02 * premium_mult
            risk = 0.02 * retention / premium_mult

            return {"ROE": roe, "bankruptcy_risk": risk}

        bounds = [(0.3, 1.0), (1.0, 3.0)]

        frontier = ParetoFrontier(objectives, insurance_objective, bounds)
        points = frontier.generate_weighted_sum(n_points=20)

        assert len(points) > 0

        # Check that we have a proper trade-off
        roe_values = [p.objectives["ROE"] for p in points]
        risk_values = [p.objectives["bankruptcy_risk"] for p in points]

        # Should have variation in both objectives
        assert max(roe_values) > min(roe_values)
        assert max(risk_values) > min(risk_values)

    def test_parallel_objectives(self):
        """Test with more than 3 objectives."""
        objectives = [Objective(f"f{i}", ObjectiveType.MINIMIZE) for i in range(4)]

        def objective_function(x):
            return {f"f{i}": x[0] ** i + x[1] ** (i + 1) for i in range(4)}

        bounds = [(0.1, 2.0), (0.1, 2.0)]

        frontier = ParetoFrontier(objectives, objective_function, bounds)
        points = frontier.generate_weighted_sum(n_points=30)

        assert len(points) > 0
        df = frontier.to_dataframe()
        assert all(f"f{i}" in df.columns for i in range(4))

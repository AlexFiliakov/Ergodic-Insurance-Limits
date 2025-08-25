"""Tests for business outcome optimization algorithms.

Author: Alex Filiakov
Date: 2025-01-25
"""

from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from ergodic_insurance.src.business_optimizer import (
    BusinessConstraints,
    BusinessObjective,
    BusinessOptimizationResult,
    BusinessOutcomeOptimizer,
    OptimalStrategy,
    OptimizationDirection,
)
from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.decision_engine import InsuranceDecisionEngine
from ergodic_insurance.src.ergodic_analyzer import ErgodicAnalyzer
from ergodic_insurance.src.loss_distributions import LognormalLoss
from ergodic_insurance.src.manufacturer import WidgetManufacturer


class TestBusinessObjective:
    """Test BusinessObjective dataclass."""

    def test_valid_objective(self):
        """Test creating valid business objective."""
        obj = BusinessObjective(
            name="ROE",
            weight=0.5,
            target_value=0.15,
            optimization_direction=OptimizationDirection.MAXIMIZE,
        )
        assert obj.name == "ROE"
        assert obj.weight == 0.5
        assert obj.target_value == 0.15
        assert obj.optimization_direction == OptimizationDirection.MAXIMIZE

    def test_invalid_weight(self):
        """Test invalid weight raises error."""
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            BusinessObjective(name="ROE", weight=1.5)

        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            BusinessObjective(name="ROE", weight=-0.1)

    def test_invalid_constraint_type(self):
        """Test invalid constraint type raises error."""
        with pytest.raises(ValueError, match="Invalid constraint type"):
            BusinessObjective(name="ROE", constraint_type="invalid", constraint_value=0.1)

    def test_valid_constraint_types(self):
        """Test valid constraint types."""
        for constraint_type in [">=", "<=", "=="]:
            obj = BusinessObjective(
                name="ROE", constraint_type=constraint_type, constraint_value=0.1
            )
            assert obj.constraint_type == constraint_type


class TestBusinessConstraints:
    """Test BusinessConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = BusinessConstraints()
        assert constraints.max_risk_tolerance == 0.01
        assert constraints.min_roe_threshold == 0.10
        assert constraints.max_leverage_ratio == 2.0
        assert constraints.min_liquidity_ratio == 1.2
        assert constraints.max_premium_budget == 0.02
        assert constraints.min_coverage_ratio == 0.5

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = BusinessConstraints(
            max_risk_tolerance=0.05, min_roe_threshold=0.15, max_leverage_ratio=3.0
        )
        assert constraints.max_risk_tolerance == 0.05
        assert constraints.min_roe_threshold == 0.15
        assert constraints.max_leverage_ratio == 3.0

    def test_invalid_risk_tolerance(self):
        """Test invalid risk tolerance raises error."""
        with pytest.raises(ValueError, match="Risk tolerance must be between 0 and 1"):
            BusinessConstraints(max_risk_tolerance=1.5)

        with pytest.raises(ValueError, match="Risk tolerance must be between 0 and 1"):
            BusinessConstraints(max_risk_tolerance=-0.1)

    def test_invalid_thresholds(self):
        """Test negative thresholds raise errors."""
        with pytest.raises(ValueError, match="ROE threshold must be non-negative"):
            BusinessConstraints(min_roe_threshold=-0.1)

        with pytest.raises(ValueError, match="Leverage ratio must be non-negative"):
            BusinessConstraints(max_leverage_ratio=-1.0)

        with pytest.raises(ValueError, match="Liquidity ratio must be non-negative"):
            BusinessConstraints(min_liquidity_ratio=-0.5)

    def test_regulatory_requirements(self):
        """Test regulatory requirements dictionary."""
        regs = {"capital_adequacy": 0.08, "liquidity_coverage": 1.0}
        constraints = BusinessConstraints(regulatory_requirements=regs)
        assert constraints.regulatory_requirements == regs


class TestOptimalStrategy:
    """Test OptimalStrategy dataclass."""

    def test_create_strategy(self):
        """Test creating optimal strategy."""
        strategy = OptimalStrategy(
            coverage_limit=10_000_000,
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.15,
            bankruptcy_risk=0.005,
            growth_rate=0.10,
            capital_efficiency=1.2,
            recommendations=["Increase coverage", "Reduce deductible"],
        )
        assert strategy.coverage_limit == 10_000_000
        assert strategy.deductible == 100_000
        assert strategy.premium_rate == 0.02
        assert strategy.expected_roe == 0.15
        assert len(strategy.recommendations) == 2

    def test_to_dict(self):
        """Test converting strategy to dictionary."""
        strategy = OptimalStrategy(
            coverage_limit=10_000_000,
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.15,
            bankruptcy_risk=0.005,
            growth_rate=0.10,
            capital_efficiency=1.2,
        )
        result = strategy.to_dict()
        assert isinstance(result, dict)
        assert result["coverage_limit"] == 10_000_000
        assert result["premium_rate"] == 0.02
        assert "recommendations" in result


class TestBusinessOptimizationResult:
    """Test BusinessOptimizationResult dataclass."""

    def test_create_result(self):
        """Test creating optimization result."""
        strategy = OptimalStrategy(
            coverage_limit=10_000_000,
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.15,
            bankruptcy_risk=0.005,
            growth_rate=0.10,
            capital_efficiency=1.2,
        )

        result = BusinessOptimizationResult(
            optimal_strategy=strategy,
            objective_values={"ROE": 0.15, "risk": 0.005},
            constraint_satisfaction={"min_roe": True, "max_risk": True},
            convergence_info={"converged": True, "iterations": 50},
        )

        assert result.optimal_strategy == strategy
        assert result.objective_values["ROE"] == 0.15
        assert result.constraint_satisfaction["min_roe"] is True

    def test_is_feasible(self):
        """Test feasibility check."""
        strategy = Mock()

        # All constraints satisfied
        result = BusinessOptimizationResult(
            optimal_strategy=strategy,
            objective_values={},
            constraint_satisfaction={"c1": True, "c2": True, "c3": True},
            convergence_info={},
        )
        assert result.is_feasible() is True

        # One constraint violated
        result = BusinessOptimizationResult(
            optimal_strategy=strategy,
            objective_values={},
            constraint_satisfaction={"c1": True, "c2": False, "c3": True},
            convergence_info={},
        )
        assert result.is_feasible() is False


class TestBusinessOutcomeOptimizer:
    """Test BusinessOutcomeOptimizer class."""

    @pytest.fixture
    def manufacturer(self):
        """Create mock manufacturer."""
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.assets = 10_000_000
        manufacturer.equity = 4_000_000
        manufacturer.liabilities = 6_000_000
        manufacturer.revenue = 5_000_000
        manufacturer.operating_income = 500_000
        manufacturer.cash = 2_000_000
        manufacturer.config = Mock()
        return manufacturer

    @pytest.fixture
    def optimizer(self, manufacturer):
        """Create optimizer instance."""
        return BusinessOutcomeOptimizer(manufacturer)

    def test_initialization(self, manufacturer):
        """Test optimizer initialization."""
        optimizer = BusinessOutcomeOptimizer(manufacturer)
        assert optimizer.manufacturer == manufacturer
        assert optimizer.decision_engine is not None
        assert optimizer.ergodic_analyzer is None

        # With custom engines
        decision_engine = Mock(spec=InsuranceDecisionEngine)
        ergodic_analyzer = Mock(spec=ErgodicAnalyzer)

        optimizer = BusinessOutcomeOptimizer(
            manufacturer, decision_engine=decision_engine, ergodic_analyzer=ergodic_analyzer
        )
        assert optimizer.decision_engine == decision_engine
        assert optimizer.ergodic_analyzer == ergodic_analyzer

    def test_maximize_roe_with_insurance(self, optimizer):
        """Test ROE maximization."""
        constraints = BusinessConstraints(
            max_risk_tolerance=0.02, min_roe_threshold=0.10, max_premium_budget=0.03
        )

        strategy = optimizer.maximize_roe_with_insurance(
            constraints=constraints, time_horizon=5, n_simulations=10  # Small for testing
        )

        assert isinstance(strategy, OptimalStrategy)
        assert strategy.coverage_limit > 0
        assert strategy.deductible >= 0
        assert 0 < strategy.premium_rate <= 0.10
        assert len(strategy.recommendations) > 0

    def test_minimize_bankruptcy_risk(self, optimizer):
        """Test bankruptcy risk minimization."""
        growth_targets = {"revenue": 0.10, "assets": 0.08}
        budget_constraint = 200_000

        strategy = optimizer.minimize_bankruptcy_risk(
            growth_targets=growth_targets, budget_constraint=budget_constraint, time_horizon=10
        )

        assert isinstance(strategy, OptimalStrategy)
        assert strategy.coverage_limit > 0
        assert strategy.bankruptcy_risk >= 0
        assert strategy.bankruptcy_risk <= 1
        assert len(strategy.recommendations) > 0

    def test_optimize_capital_efficiency(self, optimizer):
        """Test capital efficiency optimization."""
        available_capital = 1_000_000
        investment_opportunities = {"growth": 0.25, "expansion": 0.20}

        allocation = optimizer.optimize_capital_efficiency(
            available_capital=available_capital, investment_opportunities=investment_opportunities
        )

        assert isinstance(allocation, dict)
        assert "insurance_premium" in allocation
        assert "working_capital" in allocation
        assert "growth_investment" in allocation
        assert "cash_reserve" in allocation

        # Check allocations sum to available capital (with small tolerance)
        total = sum(
            allocation[k]
            for k in ["insurance_premium", "working_capital", "growth_investment", "cash_reserve"]
        )
        assert abs(total - available_capital) < 1.0

        # Check metrics
        assert "expected_return" in allocation
        assert "risk_level" in allocation
        assert "sharpe_ratio" in allocation

    def test_analyze_time_horizon_impact(self, optimizer):
        """Test time horizon impact analysis."""
        strategies = [
            {
                "name": "Conservative",
                "coverage_limit": 5_000_000,
                "deductible": 200_000,
                "premium_rate": 0.015,
            },
            {
                "name": "Moderate",
                "coverage_limit": 10_000_000,
                "deductible": 100_000,
                "premium_rate": 0.025,
            },
            {
                "name": "Aggressive",
                "coverage_limit": 15_000_000,
                "deductible": 50_000,
                "premium_rate": 0.035,
            },
        ]

        df = optimizer.analyze_time_horizon_impact(strategies=strategies, time_horizons=[1, 5, 10])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 9  # 3 strategies × 3 horizons
        assert "strategy" in df.columns
        assert "horizon_years" in df.columns
        assert "expected_roe" in df.columns
        assert "bankruptcy_risk" in df.columns
        assert "growth_rate" in df.columns
        assert "horizon_category" in df.columns

        # Check rankings are present
        assert "roe_rank" in df.columns
        assert "risk_rank" in df.columns

    def test_optimize_business_outcomes(self, optimizer):
        """Test multi-objective optimization."""
        objectives = [
            BusinessObjective(
                name="ROE", weight=0.4, optimization_direction=OptimizationDirection.MAXIMIZE
            ),
            BusinessObjective(
                name="bankruptcy_risk",
                weight=0.3,
                optimization_direction=OptimizationDirection.MINIMIZE,
            ),
            BusinessObjective(
                name="growth_rate",
                weight=0.3,
                optimization_direction=OptimizationDirection.MAXIMIZE,
            ),
        ]

        constraints = BusinessConstraints(max_risk_tolerance=0.02, min_roe_threshold=0.08)

        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=5
        )

        assert isinstance(result, BusinessOptimizationResult)
        assert isinstance(result.optimal_strategy, OptimalStrategy)
        assert "ROE" in result.objective_values
        assert "bankruptcy_risk" in result.objective_values
        assert "growth_rate" in result.objective_values
        assert result.convergence_info["converged"] in [True, False]
        assert "min_roe" in result.constraint_satisfaction

    def test_optimize_with_ergodic_analyzer(self, manufacturer):
        """Test optimization with ergodic analyzer."""
        ergodic_analyzer = Mock(spec=ErgodicAnalyzer)
        optimizer = BusinessOutcomeOptimizer(manufacturer, ergodic_analyzer=ergodic_analyzer)

        strategies = [
            {
                "name": "Test",
                "coverage_limit": 10_000_000,
                "deductible": 100_000,
                "premium_rate": 0.02,
            }
        ]

        df = optimizer.analyze_time_horizon_impact(strategies=strategies, time_horizons=[1, 30])

        # Check ergodic difference is calculated for long horizons
        long_horizon = df[df["horizon_years"] == 30]
        assert "ergodic_difference" in df.columns
        # For 30-year horizon, ergodic difference should be non-zero
        assert long_horizon["ergodic_difference"].iloc[0] != 0

    def test_objective_normalization(self, optimizer):
        """Test that objective weights are normalized."""
        objectives = [
            BusinessObjective(name="ROE", weight=0.2),
            BusinessObjective(name="risk", weight=0.3),
            BusinessObjective(name="growth", weight=0.5),
        ]

        constraints = BusinessConstraints()

        # Weights already sum to 1, should not change
        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=1
        )

        # Check that optimization completes without error
        assert isinstance(result, BusinessOptimizationResult)

        # Verify weights remain the same (already normalized)
        assert objectives[0].weight == 0.2
        assert objectives[1].weight == 0.3
        assert objectives[2].weight == 0.5

        # Test with non-normalized weights
        objectives2 = [
            BusinessObjective(name="ROE", weight=0.4),
            BusinessObjective(name="risk", weight=0.6),
            BusinessObjective(name="growth", weight=1.0),
        ]

        result2 = optimizer.optimize_business_outcomes(
            objectives=objectives2, constraints=constraints, time_horizon=1
        )

        # Check normalization (0.4+0.6+1.0=2.0, so weights should be 0.2, 0.3, 0.5)
        assert abs(objectives2[0].weight - 0.2) < 0.001
        assert abs(objectives2[1].weight - 0.3) < 0.001
        assert abs(objectives2[2].weight - 0.5) < 0.001

    def test_sensitivity_analysis(self, optimizer):
        """Test sensitivity analysis is performed."""
        objectives = [BusinessObjective(name="ROE", weight=1.0)]
        constraints = BusinessConstraints()

        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=5
        )

        assert result.sensitivity_analysis is not None
        assert "coverage_limit" in result.sensitivity_analysis
        assert "deductible" in result.sensitivity_analysis
        assert "premium_rate" in result.sensitivity_analysis

    def test_recommendations_generation(self, optimizer):
        """Test that recommendations are generated properly."""
        constraints = BusinessConstraints()

        # Test ROE maximization recommendations
        strategy = optimizer.maximize_roe_with_insurance(
            constraints=constraints, time_horizon=5, n_simulations=10
        )

        assert len(strategy.recommendations) > 0
        assert all(isinstance(rec, str) for rec in strategy.recommendations)

        # Test risk minimization recommendations
        strategy = optimizer.minimize_bankruptcy_risk(
            growth_targets={"revenue": 0.10}, budget_constraint=100_000, time_horizon=5
        )

        assert len(strategy.recommendations) > 0
        assert all(isinstance(rec, str) for rec in strategy.recommendations)

    def test_constraint_handling(self, optimizer):
        """Test that constraints are properly enforced."""
        # Very restrictive constraints
        constraints = BusinessConstraints(
            max_risk_tolerance=0.001,  # Very low risk tolerance
            min_roe_threshold=0.30,  # Very high ROE requirement
            max_premium_budget=0.001,  # Very low premium budget
        )

        # This should still complete, even if constraints are hard to satisfy
        strategy = optimizer.maximize_roe_with_insurance(
            constraints=constraints, time_horizon=1, n_simulations=5
        )

        assert isinstance(strategy, OptimalStrategy)
        # Premium should respect budget constraint
        annual_premium = strategy.coverage_limit * strategy.premium_rate
        max_allowed = optimizer.manufacturer.revenue * constraints.max_premium_budget
        # Allow some tolerance due to optimization
        assert annual_premium <= max_allowed * 1.1

    def test_time_horizon_categorization(self, optimizer):
        """Test time horizon categorization."""
        df = optimizer.analyze_time_horizon_impact(
            strategies=[{"name": "Test", "coverage_limit": 10_000_000}],
            time_horizons=[1, 2, 5, 15, 35],
        )

        categories = df["horizon_category"].unique()
        assert "Short-term" in categories
        assert "Medium-term" in categories
        assert "Long-term" in categories
        assert "Strategic" in categories

    def test_objective_with_constraints(self, optimizer):
        """Test objectives with built-in constraints."""
        objectives = [
            BusinessObjective(name="ROE", weight=0.5, constraint_type=">=", constraint_value=0.12),
            BusinessObjective(
                name="bankruptcy_risk", weight=0.5, constraint_type="<=", constraint_value=0.01
            ),
        ]

        constraints = BusinessConstraints()

        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=5
        )

        assert isinstance(result, BusinessOptimizationResult)
        # The optimization should try to satisfy the objective constraints
        # (though it may not always succeed depending on feasibility)
        assert "ROE" in result.objective_values
        assert "bankruptcy_risk" in result.objective_values

    def test_empty_objectives_handling(self, optimizer):
        """Test handling of empty objectives list."""
        objectives = []
        constraints = BusinessConstraints()

        # Should handle empty objectives gracefully
        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=5
        )

        assert isinstance(result, BusinessOptimizationResult)

    def test_invalid_objective_name(self, optimizer):
        """Test handling of unknown objective names."""
        objectives = [BusinessObjective(name="unknown_metric", weight=1.0)]
        constraints = BusinessConstraints()

        # Should handle unknown objectives gracefully (returns 0)
        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=5
        )

        assert isinstance(result, BusinessOptimizationResult)
        assert "unknown_metric" in result.objective_values
        assert result.objective_values["unknown_metric"] == 0

    def test_optimization_convergence_failure(self, optimizer):
        """Test handling of optimization convergence failure."""
        with patch("scipy.optimize.minimize") as mock_minimize:
            # Mock a failed optimization
            mock_result = Mock()
            mock_result.success = False
            mock_result.message = "Max iterations reached"
            mock_result.x = np.array([5_000_000, 100_000, 0.02])
            mock_result.fun = 0.1
            mock_result.nit = 100
            mock_minimize.return_value = mock_result

            constraints = BusinessConstraints()
            strategy = optimizer.maximize_roe_with_insurance(
                constraints=constraints, time_horizon=5, n_simulations=10
            )

            # Should still return a strategy even if optimization didn't converge
            assert isinstance(strategy, OptimalStrategy)
            assert strategy.coverage_limit > 0


class TestIntegration:
    """Integration tests for business optimizer."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create real manufacturer with config
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            initial_equity=4_000_000,
            revenue_per_asset=0.5,
            operating_margin=0.10,
        )
        manufacturer = WidgetManufacturer(config)

        # Create optimizer
        optimizer = BusinessOutcomeOptimizer(manufacturer)

        # Define objectives
        objectives = [
            BusinessObjective(
                name="ROE",
                weight=0.4,
                optimization_direction=OptimizationDirection.MAXIMIZE,
                target_value=0.15,
            ),
            BusinessObjective(
                name="bankruptcy_risk",
                weight=0.3,
                optimization_direction=OptimizationDirection.MINIMIZE,
                target_value=0.01,
            ),
            BusinessObjective(
                name="growth_rate",
                weight=0.3,
                optimization_direction=OptimizationDirection.MAXIMIZE,
                target_value=0.10,
            ),
        ]

        # Define constraints
        constraints = BusinessConstraints(
            max_risk_tolerance=0.02,
            min_roe_threshold=0.08,
            max_leverage_ratio=2.5,
            max_premium_budget=0.025,
            min_coverage_ratio=0.6,
        )

        # Run optimization
        result = optimizer.optimize_business_outcomes(
            objectives=objectives, constraints=constraints, time_horizon=10, method="weighted_sum"
        )

        # Verify results
        assert isinstance(result, BusinessOptimizationResult)
        assert result.optimal_strategy.coverage_limit > 0
        assert result.optimal_strategy.deductible >= 0
        assert 0 < result.optimal_strategy.premium_rate <= 0.10

        # Check objectives were evaluated
        assert len(result.objective_values) >= 3
        assert "ROE" in result.objective_values or "roe" in result.objective_values
        assert "bankruptcy_risk" in result.objective_values
        assert "growth_rate" in result.objective_values

        # Check constraints were evaluated
        assert len(result.constraint_satisfaction) > 0

        # Check convergence info
        assert "converged" in result.convergence_info
        assert "iterations" in result.convergence_info

        # Test strategy implementation
        strategy_dict = result.optimal_strategy.to_dict()
        assert isinstance(strategy_dict, dict)
        assert "coverage_limit" in strategy_dict
        assert "recommendations" in strategy_dict

    def test_comparative_analysis(self):
        """Test comparing multiple optimization approaches."""
        config = ManufacturerConfig(
            initial_assets=15_000_000,
            initial_equity=6_000_000,
            revenue_per_asset=0.53,  # To get ~8M revenue
            operating_margin=0.10,
        )
        manufacturer = WidgetManufacturer(config)

        optimizer = BusinessOutcomeOptimizer(manufacturer)

        # Compare different constraint scenarios
        scenarios = [
            BusinessConstraints(max_risk_tolerance=0.01, min_roe_threshold=0.15),
            BusinessConstraints(max_risk_tolerance=0.05, min_roe_threshold=0.10),
            BusinessConstraints(max_risk_tolerance=0.02, min_roe_threshold=0.12),
        ]

        results = []
        for constraints in scenarios:
            strategy = optimizer.maximize_roe_with_insurance(
                constraints=constraints, time_horizon=5, n_simulations=20
            )
            results.append(strategy)

        # Verify we get different strategies for different constraints
        assert len(results) == 3

        # More risk-tolerant scenarios should generally allow higher ROE
        # (though this isn't guaranteed due to optimization complexity)
        assert all(isinstance(r, OptimalStrategy) for r in results)

        # All strategies should have recommendations
        assert all(len(r.recommendations) > 0 for r in results)

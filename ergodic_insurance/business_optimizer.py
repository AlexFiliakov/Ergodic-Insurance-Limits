"""Business outcome optimization algorithms for insurance decisions.

This module implements sophisticated optimization algorithms focused on real business
outcomes (ROE, growth rate, survival probability) rather than technical metrics.
These algorithms maximize long-term company value through optimal insurance decisions.

Author: Alex Filiakov
Date: 2025-01-25
"""
# pylint: disable=too-many-lines

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import optimize

from .config import BusinessOptimizerConfig
from .decision_engine import InsuranceDecisionEngine
from .ergodic_analyzer import ErgodicAnalyzer
from .loss_distributions import LossDistribution
from .manufacturer import WidgetManufacturer

logger = logging.getLogger(__name__)


class OptimizationDirection(Enum):
    """Direction of optimization for objectives."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class BusinessObjective:
    """Business optimization objective definition.

    Attributes:
        name: Name of the objective (e.g., 'ROE', 'bankruptcy_risk')
        weight: Weight in multi-objective optimization (0-1)
        target_value: Optional target value for the objective
        optimization_direction: Whether to maximize or minimize
        constraint_type: Optional constraint type ('>=', '<=', '==')
        constraint_value: Optional constraint value
    """

    name: str
    weight: float = 1.0
    target_value: Optional[float] = None
    optimization_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    constraint_type: Optional[str] = None
    constraint_value: Optional[float] = None

    def __post_init__(self):
        """Validate objective configuration."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        if self.constraint_type and self.constraint_type not in [">=", "<=", "=="]:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")


@dataclass
class BusinessConstraints:
    """Business optimization constraints.

    Attributes:
        max_risk_tolerance: Maximum acceptable probability of bankruptcy
        min_roe_threshold: Minimum required return on equity
        max_leverage_ratio: Maximum debt-to-equity ratio
        min_liquidity_ratio: Minimum liquidity requirements
        max_premium_budget: Maximum insurance premium as % of revenue
        min_coverage_ratio: Minimum coverage as % of assets
        regulatory_requirements: Additional regulatory constraints
    """

    max_risk_tolerance: float = 0.01  # 1% bankruptcy risk
    min_roe_threshold: float = 0.10  # 10% minimum ROE
    max_leverage_ratio: float = 2.0  # 2:1 debt-to-equity
    min_liquidity_ratio: float = 1.2  # 1.2x current ratio
    max_premium_budget: float = 0.02  # 2% of revenue
    min_coverage_ratio: float = 0.5  # 50% of assets
    regulatory_requirements: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate constraint values."""
        if self.max_risk_tolerance < 0 or self.max_risk_tolerance > 1:
            raise ValueError("Risk tolerance must be between 0 and 1")
        if self.min_roe_threshold < 0:
            raise ValueError("ROE threshold must be non-negative")
        if self.max_leverage_ratio < 0:
            raise ValueError("Leverage ratio must be non-negative")
        if self.min_liquidity_ratio < 0:
            raise ValueError("Liquidity ratio must be non-negative")


@dataclass
class OptimalStrategy:
    """Optimal insurance strategy result.

    Attributes:
        coverage_limit: Optimal coverage limit amount
        deductible: Optimal deductible amount
        premium_rate: Optimal premium rate
        expected_roe: Expected ROE with this strategy
        bankruptcy_risk: Probability of bankruptcy
        growth_rate: Expected growth rate
        capital_efficiency: Capital efficiency ratio
        recommendations: List of actionable recommendations
    """

    coverage_limit: float
    deductible: float
    premium_rate: float
    expected_roe: float
    bankruptcy_risk: float
    growth_rate: float
    capital_efficiency: float
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Union[float, List[str]]]:
        """Convert to dictionary for serialization."""
        return {
            "coverage_limit": self.coverage_limit,
            "deductible": self.deductible,
            "premium_rate": self.premium_rate,
            "expected_roe": self.expected_roe,
            "bankruptcy_risk": self.bankruptcy_risk,
            "growth_rate": self.growth_rate,
            "capital_efficiency": self.capital_efficiency,
            "recommendations": self.recommendations,
        }


@dataclass
class BusinessOptimizationResult:
    """Result of business outcome optimization.

    Attributes:
        optimal_strategy: The optimal insurance strategy
        objective_values: Values achieved for each objective
        constraint_satisfaction: Status of constraint satisfaction
        convergence_info: Optimization convergence information
        sensitivity_analysis: Sensitivity to parameter changes
    """

    optimal_strategy: OptimalStrategy
    objective_values: Dict[str, float]
    constraint_satisfaction: Dict[str, bool]
    convergence_info: Dict[str, Union[bool, int, float]]
    sensitivity_analysis: Optional[Dict[str, float]] = None

    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied."""
        return all(self.constraint_satisfaction.values())


class BusinessOptimizer:
    """Optimize business outcomes through insurance decisions.

    This class implements sophisticated optimization algorithms focused on
    real business metrics like ROE, growth rate, and survival probability.
    """

    def __init__(
        self,
        manufacturer: WidgetManufacturer,
        decision_engine: Optional[InsuranceDecisionEngine] = None,
        ergodic_analyzer: Optional[ErgodicAnalyzer] = None,
        loss_distribution: Optional[LossDistribution] = None,
        optimizer_config: Optional[BusinessOptimizerConfig] = None,
    ):
        """Initialize business optimizer.

        Args:
            manufacturer: Widget manufacturer model
            decision_engine: Insurance decision engine (optional)
            ergodic_analyzer: Ergodic analysis tools (optional)
            loss_distribution: Loss distribution model (optional)
            optimizer_config: Configuration for optimizer heuristic parameters (optional).
                If None, uses default BusinessOptimizerConfig values.
        """
        self.manufacturer = manufacturer
        self.optimizer_config = optimizer_config or BusinessOptimizerConfig()

        # Create default loss distribution if not provided
        if loss_distribution is None:
            from .loss_distributions import LognormalLoss

            loss_distribution = LognormalLoss(mean=100000, cv=1.5)

        self.loss_distribution = loss_distribution
        self.decision_engine = decision_engine or InsuranceDecisionEngine(
            manufacturer, loss_distribution
        )
        self.ergodic_analyzer = ergodic_analyzer
        self.logger = logging.getLogger(self.__class__.__name__)

    def maximize_roe_with_insurance(
        self, constraints: BusinessConstraints, time_horizon: int = 10, n_simulations: int = 1000
    ) -> OptimalStrategy:
        """Maximize ROE subject to business constraints.

        Objective: max(ROE_with_insurance - ROE_baseline)

        Args:
            constraints: Business constraints to satisfy
            time_horizon: Planning horizon in years
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Optimal insurance strategy maximizing ROE
        """
        self.logger.info(
            f"Maximizing ROE over {time_horizon} years with {n_simulations} simulations"
        )

        # Boundary: float for scipy.optimize
        total_assets = float(self.manufacturer.total_assets)
        revenue = float(self.manufacturer.calculate_revenue())

        # Define optimization bounds
        bounds = [
            (1e6, min(total_assets * 2, 100e6)),  # Coverage limit
            (0, 1e6),  # Deductible
            (0.001, 0.10),  # Premium rate (0.1% to 10%)
        ]

        # Define objective function
        def objective(x):
            coverage_limit, deductible, premium_rate = x

            # Simulate with insurance
            roe_with_insurance = self._simulate_roe(
                coverage_limit=coverage_limit,
                deductible=deductible,
                premium_rate=premium_rate,
                time_horizon=time_horizon,
                n_simulations=n_simulations,
            )

            # Return negative ROE for minimization
            return -roe_with_insurance

        # Define constraints
        constraint_list = []

        # Premium budget constraint
        def premium_constraint(x):
            _, _, premium_rate = x
            coverage_limit = x[0]
            annual_premium = coverage_limit * premium_rate
            max_premium = revenue * constraints.max_premium_budget
            return max_premium - annual_premium

        constraint_list.append({"type": "ineq", "fun": premium_constraint})

        # Bankruptcy risk constraint
        def risk_constraint(x):
            bankruptcy_risk = self._estimate_bankruptcy_risk(
                coverage_limit=x[0], deductible=x[1], premium_rate=x[2], time_horizon=time_horizon
            )
            return constraints.max_risk_tolerance - bankruptcy_risk

        constraint_list.append({"type": "ineq", "fun": risk_constraint})

        # Coverage ratio constraint
        def coverage_constraint(x):
            coverage_limit = x[0]
            min_coverage = total_assets * constraints.min_coverage_ratio
            return coverage_limit - min_coverage

        constraint_list.append({"type": "ineq", "fun": coverage_constraint})

        # Initial guess
        x0 = [
            total_assets * 0.8,  # 80% of assets
            100000,  # $100k deductible
            0.02,  # 2% premium rate
        ]

        # Run optimization
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")

        # Extract optimal values
        optimal_coverage, optimal_deductible, optimal_premium = result.x
        optimal_roe = -result.fun

        # Calculate additional metrics
        bankruptcy_risk = self._estimate_bankruptcy_risk(
            optimal_coverage, optimal_deductible, optimal_premium, time_horizon
        )
        growth_rate = self._estimate_growth_rate(
            optimal_coverage, optimal_deductible, optimal_premium, time_horizon
        )
        capital_efficiency = self._calculate_capital_efficiency(
            optimal_coverage, optimal_deductible, optimal_premium
        )

        # Generate recommendations
        recommendations = self._generate_roe_recommendations(
            optimal_coverage, optimal_deductible, optimal_premium, optimal_roe
        )

        return OptimalStrategy(
            coverage_limit=optimal_coverage,
            deductible=optimal_deductible,
            premium_rate=optimal_premium,
            expected_roe=optimal_roe,
            bankruptcy_risk=bankruptcy_risk,
            growth_rate=growth_rate,
            capital_efficiency=capital_efficiency,
            recommendations=recommendations,
        )

    def minimize_bankruptcy_risk(
        self, growth_targets: Dict[str, float], budget_constraint: float, time_horizon: int = 10
    ) -> OptimalStrategy:
        # pylint: disable=too-many-locals
        """Minimize bankruptcy risk while achieving growth targets.

        Objective: min(P(bankruptcy))

        Args:
            growth_targets: Target growth rates (e.g., {'revenue': 0.15, 'assets': 0.10})
            budget_constraint: Maximum premium budget
            time_horizon: Planning horizon in years

        Returns:
            Risk-minimizing insurance strategy
        """
        self.logger.info(f"Minimizing bankruptcy risk over {time_horizon} years")

        # Define optimization bounds
        total_assets = float(self.manufacturer.total_assets)
        bounds = [
            (1e6, min(total_assets * 3, 150e6)),  # Coverage limit
            (0, 500000),  # Deductible
            (0.001, 0.15),  # Premium rate (0.1% to 15%)
        ]

        # Define objective function (minimize bankruptcy risk)
        def objective(x):
            coverage_limit, deductible, premium_rate = x
            bankruptcy_risk = self._estimate_bankruptcy_risk(
                coverage_limit, deductible, premium_rate, time_horizon
            )
            return bankruptcy_risk

        # Define constraints
        constraint_list = []

        # Budget constraint
        def budget_constraint_fn(x):
            coverage_limit, _, premium_rate = x
            annual_premium = coverage_limit * premium_rate
            return budget_constraint - annual_premium

        constraint_list.append({"type": "ineq", "fun": budget_constraint_fn})

        # Growth target constraints
        for metric, target in growth_targets.items():

            def growth_constraint(x, metric=metric, target=target):
                growth_rate = self._estimate_growth_rate(
                    x[0], x[1], x[2], time_horizon, metric=metric
                )
                return growth_rate - target

            constraint_list.append({"type": "ineq", "fun": growth_constraint})

        # Initial guess
        x0 = [
            total_assets * 1.5,  # 150% of assets
            50000,  # $50k deductible
            0.03,  # 3% premium rate
        ]

        # Run optimization
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 150, "ftol": 1e-7},
        )

        if not result.success:
            self.logger.warning(f"Risk minimization did not converge: {result.message}")

        # Extract optimal values
        optimal_coverage, optimal_deductible, optimal_premium = result.x
        optimal_risk = result.fun

        # Calculate additional metrics
        expected_roe = self._simulate_roe(
            optimal_coverage, optimal_deductible, optimal_premium, time_horizon
        )
        growth_rate = self._estimate_growth_rate(
            optimal_coverage, optimal_deductible, optimal_premium, time_horizon
        )
        capital_efficiency = self._calculate_capital_efficiency(
            optimal_coverage, optimal_deductible, optimal_premium
        )

        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            optimal_coverage, optimal_deductible, optimal_premium, optimal_risk
        )

        return OptimalStrategy(
            coverage_limit=optimal_coverage,
            deductible=optimal_deductible,
            premium_rate=optimal_premium,
            expected_roe=expected_roe,
            bankruptcy_risk=optimal_risk,
            growth_rate=growth_rate,
            capital_efficiency=capital_efficiency,
            recommendations=recommendations,
        )

    def optimize_capital_efficiency(
        self, available_capital: float, investment_opportunities: Dict[str, float]
    ) -> Dict[str, float]:
        """Optimize capital allocation across insurance and investments.

        Args:
            available_capital: Total capital available for allocation
            investment_opportunities: Opportunities with expected returns

        Returns:
            Optimal capital allocation dictionary
        """
        self.logger.info(f"Optimizing capital efficiency with ${available_capital:,.0f}")

        # Categories for capital allocation
        categories = ["insurance_premium", "working_capital", "growth_investment", "cash_reserve"]

        # Expected returns for each category
        expected_returns = {
            "insurance_premium": self._estimate_insurance_return(),
            "working_capital": 0.12,  # Working capital efficiency
            "growth_investment": investment_opportunities.get("growth", 0.20),
            "cash_reserve": 0.02,  # Minimal return on reserves
        }

        # Risk factors for each category
        risk_factors = {
            "insurance_premium": 0.05,  # Low risk due to protection
            "working_capital": 0.15,
            "growth_investment": 0.30,
            "cash_reserve": 0.01,
        }

        # Define optimization problem
        n_categories = len(categories)

        # Objective: maximize risk-adjusted return
        def objective(x):
            total_return = sum(x[i] * expected_returns[cat] for i, cat in enumerate(categories))
            total_risk = np.sqrt(
                sum((x[i] * risk_factors[cat]) ** 2 for i, cat in enumerate(categories))
            )
            sharpe_ratio = total_return / (total_risk + 1e-6)  # Risk-adjusted return
            return -sharpe_ratio  # Negative for minimization

        # Constraints
        constraints = [
            # Sum equals available capital
            {"type": "eq", "fun": lambda x: sum(x) - available_capital},
            # Minimum insurance allocation (1% of capital)
            {"type": "ineq", "fun": lambda x: x[0] - 0.01 * available_capital},
            # Minimum working capital (15% of capital)
            {"type": "ineq", "fun": lambda x: x[1] - 0.15 * available_capital},
            # Minimum cash reserve (5% of capital)
            {"type": "ineq", "fun": lambda x: x[3] - 0.05 * available_capital},
        ]

        # Bounds (all non-negative, up to full capital)
        bounds = [(0, available_capital) for _ in range(n_categories)]

        # Initial guess (equal allocation)
        x0 = [available_capital / n_categories] * n_categories

        # Run optimization
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        if not result.success:
            self.logger.warning("Capital allocation optimization did not fully converge")

        # Create allocation dictionary
        allocation = {cat: result.x[i] for i, cat in enumerate(categories)}

        # Add efficiency metrics
        allocation["expected_return"] = sum(
            allocation[cat] * expected_returns[cat] for cat in categories
        )
        allocation["risk_level"] = np.sqrt(
            sum((allocation[cat] * risk_factors[cat]) ** 2 for cat in categories)
        )
        allocation["sharpe_ratio"] = -result.fun

        return allocation

    def analyze_time_horizon_impact(
        self, strategies: List[Dict[str, Any]], time_horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Analyze strategy performance across different time horizons.

        Args:
            strategies: List of strategy parameters
            time_horizons: List of time horizons to analyze

        Returns:
            DataFrame with performance metrics by time horizon
        """
        if time_horizons is None:
            time_horizons = [1, 3, 10, 30]  # Default horizons

        self.logger.info(
            f"Analyzing {len(strategies)} strategies across {len(time_horizons)} time horizons"
        )

        results = []

        for strategy in strategies:
            coverage_limit = strategy.get(
                "coverage_limit", float(self.manufacturer.total_assets)
            )  # Boundary: float for scipy.optimize
            deductible = strategy.get("deductible", 100000)
            premium_rate = strategy.get("premium_rate", 0.02)
            strategy_name = strategy.get("name", "Strategy")

            for horizon in time_horizons:
                # Calculate metrics for this combination
                roe = self._simulate_roe(coverage_limit, deductible, premium_rate, horizon)
                bankruptcy_risk = self._estimate_bankruptcy_risk(
                    coverage_limit, deductible, premium_rate, horizon
                )
                growth_rate = self._estimate_growth_rate(
                    coverage_limit, deductible, premium_rate, horizon
                )

                # Calculate ergodic vs ensemble difference if analyzer available
                ergodic_diff: float = 0.0
                if self.ergodic_analyzer and horizon >= 10:
                    ergodic_growth = self._calculate_ergodic_growth(
                        coverage_limit, deductible, premium_rate, horizon
                    )
                    ensemble_growth = growth_rate
                    ergodic_diff = ergodic_growth - ensemble_growth

                results.append(
                    {
                        "strategy": strategy_name,
                        "horizon_years": horizon,
                        "coverage_limit": coverage_limit,
                        "deductible": deductible,
                        "premium_rate": premium_rate,
                        "expected_roe": roe,
                        "bankruptcy_risk": bankruptcy_risk,
                        "growth_rate": growth_rate,
                        "ergodic_difference": ergodic_diff,
                        "horizon_category": self._categorize_horizon(horizon),
                    }
                )

        df = pd.DataFrame(results)

        # Add relative performance metrics
        for horizon in time_horizons:
            mask = df["horizon_years"] == horizon
            df.loc[mask, "roe_rank"] = df.loc[mask, "expected_roe"].rank(ascending=False)
            df.loc[mask, "risk_rank"] = df.loc[mask, "bankruptcy_risk"].rank(ascending=True)

        return df

    def optimize_business_outcomes(
        self,
        objectives: List[BusinessObjective],
        constraints: BusinessConstraints,
        time_horizon: int = 10,
        method: str = "weighted_sum",
    ) -> BusinessOptimizationResult:
        # pylint: disable=too-many-locals
        """Multi-objective optimization of business outcomes.

        Args:
            objectives: List of business objectives to optimize
            constraints: Business constraints to satisfy
            time_horizon: Planning horizon in years
            method: Optimization method ('weighted_sum', 'epsilon_constraint', 'pareto')

        Returns:
            Comprehensive optimization result
        """
        self.logger.info(f"Optimizing {len(objectives)} objectives using {method} method")

        # Normalize weights
        total_weight = sum(obj.weight for obj in objectives)
        if total_weight > 0:
            for obj in objectives:
                obj.weight /= total_weight

        # Define optimization bounds
        # Boundary: float for scipy.optimize
        total_assets = float(self.manufacturer.total_assets)
        bounds = [
            (1e6, min(total_assets * 2.5, 100e6)),  # Coverage limit
            (0, 500000),  # Deductible
            (0.001, 0.10),  # Premium rate
        ]

        # Build composite objective function
        def composite_objective(x):
            coverage_limit, deductible, premium_rate = x
            total_score = 0.0
            total_score = 0.0

            for obj in objectives:
                value = self._evaluate_objective(
                    obj.name, coverage_limit, deductible, premium_rate, time_horizon
                )

                # Normalize and apply direction
                if obj.optimization_direction == OptimizationDirection.MAXIMIZE:
                    score = value  # Higher is better
                else:
                    score = -value  # Lower is better (negate for minimization)

                total_score = total_score + obj.weight * score

            return -total_score  # Negative for scipy minimization

        # Build constraints list
        constraint_list = self._build_constraint_list(objectives, constraints, time_horizon)

        # Initial guess
        x0 = [total_assets * 1.0, 100000, 0.025]

        # Run optimization
        result = optimize.minimize(
            composite_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        # Extract results
        optimal_coverage, optimal_deductible, optimal_premium = result.x

        # Calculate all objective values
        objective_values = {}
        for obj in objectives:
            value = self._evaluate_objective(
                obj.name, optimal_coverage, optimal_deductible, optimal_premium, time_horizon
            )
            objective_values[obj.name] = value

        # Check constraint satisfaction
        constraint_satisfaction = self._check_constraints(
            optimal_coverage, optimal_deductible, optimal_premium, constraints, time_horizon
        )

        # Perform sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(
            optimal_coverage, optimal_deductible, optimal_premium, objectives, time_horizon
        )

        # Create optimal strategy
        optimal_strategy = OptimalStrategy(
            coverage_limit=optimal_coverage,
            deductible=optimal_deductible,
            premium_rate=optimal_premium,
            expected_roe=objective_values.get("ROE", 0),
            bankruptcy_risk=objective_values.get("bankruptcy_risk", 0),
            growth_rate=objective_values.get("growth_rate", 0),
            capital_efficiency=self._calculate_capital_efficiency(
                optimal_coverage, optimal_deductible, optimal_premium
            ),
            recommendations=self._generate_comprehensive_recommendations(
                optimal_coverage, optimal_deductible, optimal_premium, objective_values
            ),
        )

        # Build convergence info
        convergence_info = {
            "converged": result.success,
            "iterations": result.nit if hasattr(result, "nit") else 0,
            "function_value": result.fun,
            "message": result.message if hasattr(result, "message") else "Optimization complete",
        }

        return BusinessOptimizationResult(
            optimal_strategy=optimal_strategy,
            objective_values=objective_values,
            constraint_satisfaction=constraint_satisfaction,
            convergence_info=convergence_info,
            sensitivity_analysis=sensitivity,
        )

    # Private helper methods

    def _simulate_roe(
        self,
        coverage_limit: float,
        deductible: float,
        premium_rate: float,
        time_horizon: int,
        n_simulations: int = 100,
    ) -> float:
        """Simulate ROE with given insurance parameters."""
        rng = np.random.default_rng()
        roe_values = []
        # Boundary: float for scipy.optimize
        equity = float(self.manufacturer.equity)
        total_assets = float(self.manufacturer.total_assets)
        annual_premium = coverage_limit * premium_rate

        for _ in range(min(n_simulations, 100)):  # Limit for performance
            # Simple ROE simulation
            base_roe = self.optimizer_config.base_roe

            # Insurance impact
            premium_cost = annual_premium / equity
            protection_benefit = self.optimizer_config.protection_benefit_factor * (
                coverage_limit / total_assets
            )

            # Adjust ROE
            adjusted_roe = base_roe - premium_cost + protection_benefit

            # Add randomness
            adjusted_roe *= rng.normal(1.0, self.optimizer_config.roe_noise_std)
            roe_values.append(adjusted_roe)

        return float(np.mean(roe_values))

    def _estimate_bankruptcy_risk(
        self, coverage_limit: float, deductible: float, premium_rate: float, time_horizon: int
    ) -> float:
        """Estimate probability of bankruptcy."""
        # Convert Decimal properties to float for calculations
        total_assets = float(self.manufacturer.total_assets)
        revenue = float(self.manufacturer.calculate_revenue())
        annual_premium = coverage_limit * premium_rate

        # Simple bankruptcy risk model
        base_risk = self.optimizer_config.base_bankruptcy_risk

        # Insurance reduces risk
        coverage_ratio = coverage_limit / total_assets
        risk_reduction = min(
            coverage_ratio * self.optimizer_config.max_risk_reduction,
            self.optimizer_config.max_risk_reduction,
        )

        # Premium cost increases risk slightly
        premium_burden = annual_premium / revenue
        risk_increase = premium_burden * self.optimizer_config.premium_burden_risk_factor

        # Time horizon effect
        time_factor = 1 - np.exp(
            -time_horizon / self.optimizer_config.time_risk_constant
        )  # Risk increases with time

        bankruptcy_risk = (base_risk - risk_reduction + risk_increase) * time_factor
        return float(max(0, min(1, bankruptcy_risk)))

    def _estimate_growth_rate(
        self,
        coverage_limit: float,
        deductible: float,
        premium_rate: float,
        time_horizon: int,
        metric: str = "revenue",
    ) -> float:
        """Estimate growth rate for given metric."""
        # Convert Decimal properties to float for calculations
        total_assets = float(self.manufacturer.total_assets)
        revenue = float(self.manufacturer.calculate_revenue())
        annual_premium = coverage_limit * premium_rate

        # Base growth rate
        base_growth = self.optimizer_config.base_growth_rate

        # Insurance enables more aggressive growth
        coverage_ratio = coverage_limit / total_assets
        growth_boost = coverage_ratio * self.optimizer_config.growth_boost_factor

        # Premium cost reduces growth
        premium_drag = annual_premium / revenue * self.optimizer_config.premium_drag_factor

        # Calculate adjusted growth
        adjusted_growth = base_growth + growth_boost - premium_drag

        # Adjust for different metrics
        if metric == "assets":
            adjusted_growth *= self.optimizer_config.asset_growth_factor
        elif metric == "equity":
            adjusted_growth *= self.optimizer_config.equity_growth_factor

        return float(max(0, adjusted_growth))

    def _calculate_capital_efficiency(
        self, coverage_limit: float, deductible: float, premium_rate: float
    ) -> float:
        """Calculate capital efficiency ratio."""
        # Convert Decimal properties to float for calculations
        total_assets = float(self.manufacturer.total_assets)
        annual_premium = coverage_limit * premium_rate

        # Capital freed by risk transfer
        risk_transfer_benefit = coverage_limit * self.optimizer_config.risk_transfer_benefit_rate

        # Net capital efficiency
        net_benefit = risk_transfer_benefit - annual_premium
        efficiency_ratio = 1 + (net_benefit / total_assets)

        return float(max(0, efficiency_ratio))

    def _estimate_insurance_return(self) -> float:
        """Estimate return on insurance investment."""
        # Insurance provides value through:
        # 1. Risk reduction (allows higher leverage)
        # 2. Stability (better credit terms)
        # 3. Growth enablement (take more risks)

        risk_reduction_value = self.optimizer_config.risk_reduction_value
        stability_value = self.optimizer_config.stability_value
        growth_enablement = self.optimizer_config.growth_enablement_value

        return risk_reduction_value + stability_value + growth_enablement

    def _calculate_ergodic_growth(
        self, coverage_limit: float, deductible: float, premium_rate: float, time_horizon: int
    ) -> float:
        """Calculate ergodic (time-average) growth rate."""
        if not self.ergodic_analyzer:
            return self._estimate_growth_rate(
                coverage_limit, deductible, premium_rate, time_horizon
            )

        # Use ergodic analyzer for proper calculation
        # This is a simplified version
        ensemble_growth = self._estimate_growth_rate(
            coverage_limit, deductible, premium_rate, time_horizon
        )
        volatility = self.optimizer_config.assumed_volatility

        # Insurance reduces volatility
        # Convert Decimal to float for calculations
        total_assets = float(self.manufacturer.total_assets)
        coverage_ratio = coverage_limit / total_assets
        volatility_reduction = coverage_ratio * self.optimizer_config.volatility_reduction_factor
        adjusted_volatility = max(
            self.optimizer_config.min_volatility, volatility - volatility_reduction
        )

        # Ergodic correction for multiplicative process with reduced volatility
        ergodic_growth = ensemble_growth - 0.5 * adjusted_volatility**2

        return float(ergodic_growth)

    def _categorize_horizon(self, years: int) -> str:
        """Categorize time horizon."""
        if years <= 1:
            return "Short-term"
        if years <= 3:
            return "Medium-term"
        if years <= 10:
            return "Long-term"
        return "Strategic"

    def _evaluate_objective(
        self,
        objective_name: str,
        coverage_limit: float,
        deductible: float,
        premium_rate: float,
        time_horizon: int,
    ) -> float:
        """Evaluate a specific objective."""
        if objective_name.lower() == "roe":
            return self._simulate_roe(coverage_limit, deductible, premium_rate, time_horizon)
        if objective_name.lower() == "bankruptcy_risk":
            return self._estimate_bankruptcy_risk(
                coverage_limit, deductible, premium_rate, time_horizon
            )
        if objective_name.lower() == "growth_rate":
            return self._estimate_growth_rate(
                coverage_limit, deductible, premium_rate, time_horizon
            )
        if objective_name.lower() == "capital_efficiency":
            return self._calculate_capital_efficiency(coverage_limit, deductible, premium_rate)

        self.logger.warning(f"Unknown objective: {objective_name}")
        return 0

    def _build_constraint_list(
        self,
        objectives: List[BusinessObjective],
        constraints: BusinessConstraints,
        time_horizon: int,
    ) -> List[Dict]:
        """Build constraint list for optimization."""
        constraint_list = []

        # Business constraints
        def roe_constraint(x):
            roe = self._simulate_roe(x[0], x[1], x[2], time_horizon)
            return roe - constraints.min_roe_threshold

        constraint_list.append({"type": "ineq", "fun": roe_constraint})

        def risk_constraint(x):
            risk = self._estimate_bankruptcy_risk(x[0], x[1], x[2], time_horizon)
            return constraints.max_risk_tolerance - risk

        constraint_list.append({"type": "ineq", "fun": risk_constraint})

        def premium_constraint(x):
            annual_premium = x[0] * x[2]
            max_premium = (
                float(self.manufacturer.calculate_revenue()) * constraints.max_premium_budget
            )
            return max_premium - annual_premium

        constraint_list.append({"type": "ineq", "fun": premium_constraint})

        # Objective-specific constraints
        for obj in objectives:
            if obj.constraint_type and obj.constraint_value is not None:

                def obj_constraint(x, obj=obj):
                    value = self._evaluate_objective(obj.name, x[0], x[1], x[2], time_horizon)
                    if obj.constraint_type == ">=":
                        return value - obj.constraint_value
                    if obj.constraint_type == "<=":
                        return obj.constraint_value - value
                    # '=='
                    return abs(value - obj.constraint_value) - 0.001

                constraint_list.append(
                    {"type": "ineq" if obj.constraint_type != "==" else "eq", "fun": obj_constraint}
                )

        return constraint_list

    def _check_constraints(
        self,
        coverage_limit: float,
        deductible: float,
        premium_rate: float,
        constraints: BusinessConstraints,
        time_horizon: int,
    ) -> Dict[str, bool]:
        """Check if constraints are satisfied."""
        # Convert Decimal properties to float for calculations
        total_assets = float(self.manufacturer.total_assets)
        equity = float(self.manufacturer.equity)
        revenue = float(self.manufacturer.calculate_revenue())

        satisfaction = {}

        # ROE constraint
        roe = self._simulate_roe(coverage_limit, deductible, premium_rate, time_horizon)
        satisfaction["min_roe"] = roe >= constraints.min_roe_threshold

        # Risk constraint
        risk = self._estimate_bankruptcy_risk(
            coverage_limit, deductible, premium_rate, time_horizon
        )
        satisfaction["max_risk"] = risk <= constraints.max_risk_tolerance

        # Premium budget constraint
        annual_premium = coverage_limit * premium_rate
        max_premium = revenue * constraints.max_premium_budget
        satisfaction["premium_budget"] = annual_premium <= max_premium

        # Coverage ratio constraint
        coverage_ratio = coverage_limit / total_assets
        satisfaction["min_coverage"] = coverage_ratio >= constraints.min_coverage_ratio

        # Leverage constraint (simplified)
        liabilities = total_assets - equity
        leverage = liabilities / (equity + 1e-6)
        satisfaction["max_leverage"] = leverage <= constraints.max_leverage_ratio

        return satisfaction

    def _perform_sensitivity_analysis(
        self,
        coverage_limit: float,
        deductible: float,
        premium_rate: float,
        objectives: List[BusinessObjective],
        time_horizon: int,
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity = {}
        delta = 0.01  # 1% change

        # Base objective value
        base_value = sum(
            obj.weight
            * self._evaluate_objective(
                obj.name, coverage_limit, deductible, premium_rate, time_horizon
            )
            for obj in objectives
        )

        # Coverage limit sensitivity
        coverage_delta = coverage_limit * delta
        value_up = sum(
            obj.weight
            * self._evaluate_objective(
                obj.name, coverage_limit + coverage_delta, deductible, premium_rate, time_horizon
            )
            for obj in objectives
        )
        sensitivity["coverage_limit"] = (value_up - base_value) / (coverage_delta + 1e-6)

        # Deductible sensitivity
        deductible_delta = max(1000, deductible * delta)
        value_up = sum(
            obj.weight
            * self._evaluate_objective(
                obj.name, coverage_limit, deductible + deductible_delta, premium_rate, time_horizon
            )
            for obj in objectives
        )
        sensitivity["deductible"] = (value_up - base_value) / (deductible_delta + 1e-6)

        # Premium rate sensitivity
        premium_delta = premium_rate * delta
        value_up = sum(
            obj.weight
            * self._evaluate_objective(
                obj.name, coverage_limit, deductible, premium_rate + premium_delta, time_horizon
            )
            for obj in objectives
        )
        sensitivity["premium_rate"] = (value_up - base_value) / (premium_delta + 1e-6)

        return sensitivity

    def _generate_roe_recommendations(
        self, coverage_limit: float, deductible: float, premium_rate: float, expected_roe: float
    ) -> List[str]:
        """Generate ROE-focused recommendations."""
        # Convert Decimal properties to float for calculations
        total_assets = float(self.manufacturer.total_assets)

        recommendations = []

        if expected_roe > 0.20:
            recommendations.append(
                "Excellent ROE achieved - consider increasing growth investments"
            )
        elif expected_roe > 0.15:
            recommendations.append("Strong ROE performance - maintain current strategy")
        else:
            recommendations.append(
                "ROE below target - review premium costs and coverage efficiency"
            )

        if premium_rate > 0.05:
            recommendations.append(
                "High premium rate - negotiate better terms or consider alternatives"
            )

        if deductible < 50000:
            recommendations.append(
                "Low deductible may be increasing costs - consider higher retention"
            )
        elif deductible > 500000:
            recommendations.append(
                "High deductible exposes significant risk - evaluate coverage gap"
            )

        coverage_ratio = coverage_limit / total_assets
        if coverage_ratio < 0.5:
            recommendations.append("Coverage may be insufficient for major losses")
        elif coverage_ratio > 1.5:
            recommendations.append("Consider if coverage exceeds actual exposure")

        return recommendations

    def _generate_risk_recommendations(
        self, coverage_limit: float, deductible: float, premium_rate: float, bankruptcy_risk: float
    ) -> List[str]:
        """Generate risk-focused recommendations."""
        # Convert Decimal properties to float for calculations
        total_assets = float(self.manufacturer.total_assets)
        revenue = float(self.manufacturer.calculate_revenue())

        recommendations = []

        if bankruptcy_risk < 0.001:
            recommendations.append("Excellent risk profile - can support aggressive growth")
        elif bankruptcy_risk < 0.01:
            recommendations.append("Risk well-controlled - current insurance adequate")
        else:
            recommendations.append(
                "Elevated bankruptcy risk - increase coverage or reduce leverage"
            )

        if coverage_limit < total_assets * 0.5:
            recommendations.append("Coverage may be insufficient for tail risks")

        if premium_rate * coverage_limit > revenue * 0.03:
            recommendations.append("Insurance costs exceeding 3% of revenue - review cost-benefit")

        return recommendations

    def _generate_comprehensive_recommendations(
        self,
        coverage_limit: float,
        deductible: float,
        premium_rate: float,
        objective_values: Dict[str, float],
    ) -> List[str]:
        """Generate comprehensive recommendations based on all metrics."""
        recommendations = []

        # ROE recommendations
        roe = objective_values.get("ROE", objective_values.get("roe", 0))
        if roe > 0:
            if roe > 0.20:
                recommendations.append("Exceptional ROE - leverage success for expansion")
            elif roe < 0.10:
                recommendations.append("ROE below industry standards - optimize capital structure")

        # Risk recommendations
        risk = objective_values.get("bankruptcy_risk", 0)
        if risk > 0.02:
            recommendations.append("High bankruptcy risk - prioritize risk mitigation")
        elif risk < 0.005:
            recommendations.append("Conservative risk profile - opportunity for higher returns")

        # Growth recommendations
        growth = objective_values.get("growth_rate", 0)
        if growth > 0.15:
            recommendations.append("Strong growth trajectory - ensure adequate risk controls")
        elif growth < 0.05:
            recommendations.append("Low growth - consider strategic initiatives")

        # Insurance structure recommendations
        # Convert Decimal properties to float for calculations
        revenue = float(self.manufacturer.calculate_revenue())
        total_assets = float(self.manufacturer.total_assets)

        annual_premium = coverage_limit * premium_rate
        premium_to_revenue = annual_premium / revenue if revenue > 0 else 0

        if premium_to_revenue > 0.04:
            recommendations.append("Premium costs high - explore alternative risk financing")
        elif premium_to_revenue < 0.01:
            recommendations.append("Low insurance spend - verify adequate protection")

        # Deductible recommendations
        deductible_to_assets = deductible / total_assets if total_assets > 0 else 0
        if deductible_to_assets > 0.05:
            recommendations.append(
                "High deductible relative to assets - monitor retention capacity"
            )

        return recommendations[:5]  # Limit to top 5 recommendations

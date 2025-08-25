"""Algorithmic insurance decision engine for optimal coverage selection.

This module implements a comprehensive decision framework that optimizes
insurance purchasing decisions using multi-objective optimization to balance
growth targets with bankruptcy risk constraints.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult, differential_evolution, minimize

from .config_loader import ConfigLoader
from .ergodic_analyzer import ErgodicAnalyzer
from .insurance_program import EnhancedInsuranceLayer as Layer
from .insurance_program import InsuranceProgram
from .loss_distributions import LossDistribution
from .manufacturer import WidgetManufacturer

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Available optimization methods."""

    SLSQP = "SLSQP"  # Sequential Least Squares Programming
    DIFFERENTIAL_EVOLUTION = "differential_evolution"  # Global optimization
    WEIGHTED_SUM = "weighted_sum"  # Multi-objective via weighted sum


@dataclass
class OptimizationConstraints:
    """Constraints for insurance optimization."""

    max_premium_budget: float = field(default=1_000_000)
    min_coverage_limit: float = field(default=5_000_000)
    max_coverage_limit: float = field(default=100_000_000)
    max_bankruptcy_probability: float = field(default=0.01)
    min_retained_limit: float = field(default=100_000)
    max_retained_limit: float = field(default=10_000_000)
    max_layers: int = field(default=5)
    min_layers: int = field(default=1)
    required_roi_improvement: float = field(default=0.0)  # Minimum ROI improvement


@dataclass
class InsuranceDecision:
    """Represents an insurance purchasing decision."""

    retained_limit: float  # Self-insured retention
    layers: List[Layer]  # Insurance layers to purchase
    total_premium: float  # Total annual premium
    total_coverage: float  # Total coverage limit
    pricing_scenario: str  # Market pricing scenario used
    optimization_method: str  # Method used to find this decision
    convergence_iterations: int = 0  # Iterations to converge
    objective_value: float = 0.0  # Final objective function value

    def __post_init__(self):
        """Calculate derived fields."""
        if self.total_coverage == 0 and self.layers:
            self.total_coverage = self.retained_limit + sum(layer.limit for layer in self.layers)
        if self.total_premium == 0 and self.layers:
            self.total_premium = sum(layer.limit * layer.premium_rate for layer in self.layers)


@dataclass
class DecisionMetrics:
    """Comprehensive metrics for evaluating an insurance decision."""

    ergodic_growth_rate: float  # Time-average growth with insurance
    bankruptcy_probability: float  # Probability of ruin
    expected_roe: float  # Expected return on equity
    roe_improvement: float  # Change in ROE vs no insurance
    premium_to_limit_ratio: float  # Premium efficiency
    coverage_adequacy: float  # Coverage vs expected losses
    capital_efficiency: float  # Benefit per dollar of premium
    value_at_risk_95: float  # 95th percentile loss
    conditional_value_at_risk: float  # Expected loss beyond VaR
    decision_score: float = 0.0  # Overall decision quality score

    # Enhanced ROE metrics
    time_weighted_roe: float = 0.0  # Time-weighted average ROE
    roe_volatility: float = 0.0  # ROE standard deviation
    roe_sharpe_ratio: float = 0.0  # ROE risk-adjusted performance
    roe_downside_deviation: float = 0.0  # Downside risk measure
    roe_1yr_rolling: float = 0.0  # 1-year rolling average ROE
    roe_3yr_rolling: float = 0.0  # 3-year rolling average ROE
    roe_5yr_rolling: float = 0.0  # 5-year rolling average ROE

    # ROE component breakdown
    operating_roe: float = 0.0  # ROE from operations
    insurance_impact_roe: float = 0.0  # ROE impact from insurance
    tax_effect_roe: float = 0.0  # Tax impact on ROE

    def calculate_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted decision score.

        Args:
            weights: Weights for each metric (default: equal weights)

        Returns:
            Weighted score between 0 and 1
        """
        if weights is None:
            weights = {
                "growth": 0.3,
                "risk": 0.3,
                "efficiency": 0.2,
                "adequacy": 0.2,
            }

        # Normalize metrics to [0, 1] scale
        growth_score = min(max(self.ergodic_growth_rate / 0.2, 0), 1)  # Target 20% growth
        risk_score = 1 - min(self.bankruptcy_probability / 0.05, 1)  # Lower is better
        efficiency_score = min(self.capital_efficiency, 1)
        adequacy_score = min(self.coverage_adequacy, 1)

        self.decision_score = (
            weights["growth"] * growth_score
            + weights["risk"] * risk_score
            + weights["efficiency"] * efficiency_score
            + weights["adequacy"] * adequacy_score
        )
        return self.decision_score


@dataclass
class SensitivityReport:
    """Results of sensitivity analysis."""

    base_decision: InsuranceDecision
    base_metrics: DecisionMetrics
    parameter_sensitivities: Dict[str, Dict[str, float]]  # param -> metric -> change
    key_drivers: List[str]  # Most influential parameters
    robust_range: Dict[str, Tuple[float, float]]  # Parameter ranges for stability
    stress_test_results: Dict[str, DecisionMetrics]  # Scenario -> metrics


@dataclass
class Recommendations:
    """Executive-ready recommendations from the decision engine."""

    primary_recommendation: InsuranceDecision
    primary_rationale: str
    alternative_options: List[Tuple[InsuranceDecision, str]]  # (decision, rationale)
    implementation_timeline: List[str]
    risk_considerations: List[str]
    expected_benefits: Dict[str, float]
    confidence_level: float  # 0-1 confidence in recommendation


class InsuranceDecisionEngine:
    """Algorithmic engine for optimizing insurance decisions."""

    def __init__(
        self,
        manufacturer: WidgetManufacturer,
        loss_distribution: LossDistribution,
        pricing_scenario: str = "baseline",
        config_loader: Optional[ConfigLoader] = None,
    ):
        """Initialize decision engine with company context.

        Args:
            manufacturer: Company profile and financials
            loss_distribution: Loss model for the company
            pricing_scenario: Market pricing scenario to use
            config_loader: Configuration loader (creates default if None)
        """
        self.manufacturer = manufacturer
        self.loss_distribution = loss_distribution
        self.pricing_scenario = pricing_scenario
        self.config_loader = config_loader or ConfigLoader()

        # Load pricing scenarios
        self.pricing_config = self.config_loader.load_pricing_scenarios()
        self.current_scenario = self.pricing_config.get_scenario(pricing_scenario)

        # Initialize components
        # InsuranceProgram will be created with layers when needed
        self.insurance_program = None
        self.ergodic_analyzer = ErgodicAnalyzer()

        # Cache for performance
        self._decision_cache: Dict[str, InsuranceDecision] = {}
        self._metrics_cache: Dict[str, DecisionMetrics] = {}

    def optimize_insurance_decision(
        self,
        constraints: OptimizationConstraints,
        method: OptimizationMethod = OptimizationMethod.SLSQP,
        weights: Optional[Dict[str, float]] = None,
    ) -> InsuranceDecision:
        """Find optimal insurance structure given constraints.

        Uses multi-objective optimization to balance growth, risk, and cost.

        Args:
            constraints: Optimization constraints
            method: Optimization method to use
            weights: Objective function weights (default: balanced)

        Returns:
            Optimal insurance decision
        """
        if weights is None:
            weights = {"growth": 0.4, "risk": 0.4, "cost": 0.2}

        logger.info(f"Starting optimization with method: {method.value}")

        # Check cache
        cache_key = f"{constraints}_{method}_{weights}"
        if cache_key in self._decision_cache:
            logger.info("Returning cached decision")
            return self._decision_cache[cache_key]

        # Run optimization based on method
        if method == OptimizationMethod.SLSQP:
            result = self._optimize_slsqp(constraints, weights)
        elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._optimize_differential_evolution(constraints, weights)
        else:  # WEIGHTED_SUM
            result = self._optimize_weighted_sum(constraints, weights)

        # Create decision from optimization result
        decision = self._create_decision_from_result(result, method)

        # Validate decision meets constraints
        if not self._validate_decision(decision, constraints):
            logger.warning("Decision violates constraints, attempting fallback")
            # Try alternative method
            fallback_method = (
                OptimizationMethod.DIFFERENTIAL_EVOLUTION
                if method != OptimizationMethod.DIFFERENTIAL_EVOLUTION
                else OptimizationMethod.WEIGHTED_SUM
            )
            # Recursive call returns InsuranceDecision, not OptimizeResult
            decision = self.optimize_insurance_decision(constraints, fallback_method, weights)

        # Cache result
        self._decision_cache[cache_key] = decision

        logger.info(
            f"Optimization complete: {len(decision.layers)} layers, "
            f"${decision.total_premium:,.0f} premium"
        )

        return decision

    def _optimize_slsqp(
        self, constraints: OptimizationConstraints, weights: Dict[str, float]
    ) -> OptimizeResult:
        """Optimize using Sequential Least Squares Programming.

        Fast convergence for smooth, constrained problems.
        """
        # Define decision variables: [retained_limit, layer1_limit, layer2_limit, ...]
        n_vars = 1 + constraints.max_layers  # retention + layer limits

        # Initial guess: equal layer spacing
        x0 = np.zeros(n_vars)
        x0[0] = constraints.min_retained_limit  # Start with minimum retention
        if constraints.max_layers > 0:
            layer_size = (
                constraints.max_coverage_limit - constraints.min_retained_limit
            ) / constraints.max_layers
            for i in range(1, min(3, n_vars)):  # Start with 2-3 layers
                x0[i] = layer_size

        # Bounds for decision variables
        bounds = [(constraints.min_retained_limit, constraints.max_retained_limit)]
        for _ in range(constraints.max_layers):
            bounds.append((0, constraints.max_coverage_limit / constraints.max_layers))

        # Define objective function
        def objective(x):
            """Objective function for optimization.

            Args:
                x: Decision variables [retained_limit, layer1_limit, ...]

            Returns:
                Objective value (negative for maximization)
            """
            return self._calculate_objective(x, weights)

        # Define constraints
        constraint_list = []

        # Premium budget constraint
        def premium_constraint(x):
            """Constraint function for premium budget.

            Args:
                x: Decision variables [retained_limit, layer1_limit, ...]

            Returns:
                Constraint value (positive when satisfied)
            """
            premium = self._calculate_premium(x)
            return constraints.max_premium_budget - premium

        constraint_list.append({"type": "ineq", "fun": premium_constraint})

        # Coverage limit constraints
        def coverage_min_constraint(x):
            """Constraint function for minimum coverage.

            Args:
                x: Decision variables [retained_limit, layer1_limit, ...]

            Returns:
                Constraint value (positive when satisfied)
            """
            total_coverage = sum(x)
            return total_coverage - constraints.min_coverage_limit

        def coverage_max_constraint(x):
            """Constraint function for maximum coverage.

            Args:
                x: Decision variables [retained_limit, layer1_limit, ...]

            Returns:
                Constraint value (positive when satisfied)
            """
            total_coverage = sum(x)
            return constraints.max_coverage_limit - total_coverage

        constraint_list.extend(
            [
                {"type": "ineq", "fun": coverage_min_constraint},
                {"type": "ineq", "fun": coverage_max_constraint},
            ]
        )

        # Bankruptcy probability constraint
        def bankruptcy_constraint(x):
            """Constraint function for bankruptcy probability.

            Args:
                x: Decision variables [retained_limit, layer1_limit, ...]

            Returns:
                Constraint value (positive when satisfied)
            """
            prob = self._estimate_bankruptcy_probability(x)
            return constraints.max_bankruptcy_probability - prob

        constraint_list.append({"type": "ineq", "fun": bankruptcy_constraint})

        # Run optimization
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        return result

    def _optimize_differential_evolution(
        self, constraints: OptimizationConstraints, weights: Dict[str, float]
    ) -> OptimizeResult:
        """Optimize using Differential Evolution.

        Global optimization method, robust to local optima.
        """
        n_vars = 1 + constraints.max_layers

        # Bounds for decision variables
        bounds = [(constraints.min_retained_limit, constraints.max_retained_limit)]
        for _ in range(constraints.max_layers):
            bounds.append((0, constraints.max_coverage_limit / constraints.max_layers))

        # Define objective with penalty for constraint violations
        def objective_with_penalty(x):
            """Objective function with penalty for constraint violations.

            Args:
                x: Decision variables [retained_limit, layer1_limit, ...]

            Returns:
                Penalized objective value
            """
            obj = self._calculate_objective(x, weights)

            # Add penalties for constraint violations
            penalty = 0.0

            # Premium constraint
            premium = self._calculate_premium(x)
            if premium > constraints.max_premium_budget:
                penalty += 1000 * (premium - constraints.max_premium_budget)

            # Coverage constraints
            total_coverage = sum(x)
            if total_coverage < constraints.min_coverage_limit:
                penalty += 1000 * (constraints.min_coverage_limit - total_coverage)
            if total_coverage > constraints.max_coverage_limit:
                penalty += 1000 * (total_coverage - constraints.max_coverage_limit)

            # Bankruptcy constraint
            bankruptcy_prob = self._estimate_bankruptcy_probability(x)
            if bankruptcy_prob > constraints.max_bankruptcy_probability:
                penalty += 10000 * (bankruptcy_prob - constraints.max_bankruptcy_probability)

            return obj + penalty

        # Run differential evolution
        result = differential_evolution(
            objective_with_penalty,
            bounds,
            strategy="best1bin",
            maxiter=500,
            popsize=15,
            tol=1e-6,
            seed=42,
        )

        return result

    def _optimize_weighted_sum(
        self, constraints: OptimizationConstraints, weights: Dict[str, float]
    ) -> OptimizeResult:
        """Optimize using weighted sum approach for multi-objective."""
        # Similar to SLSQP but with explicit multi-objective handling
        return self._optimize_slsqp(constraints, weights)

    def _calculate_objective(self, x: np.ndarray, weights: Dict[str, float]) -> float:
        """Calculate weighted objective function value.

        Lower values are better.
        """
        # Parse decision variables
        retained_limit = x[0]
        layer_limits = x[1:]

        # Skip if no actual layers
        active_layers = [l for l in layer_limits if l > 1000]  # Min $1k layer
        if not active_layers:
            return 1e10  # Penalize no insurance

        # Calculate components
        try:
            # Growth component (negative because we maximize)
            growth_rate = self._estimate_growth_rate(retained_limit, active_layers)
            growth_obj = -weights["growth"] * growth_rate

            # Risk component (minimize bankruptcy probability)
            bankruptcy_prob = self._estimate_bankruptcy_probability(x)
            risk_obj = weights["risk"] * bankruptcy_prob * 100  # Scale for balance

            # Cost component (minimize premium as % of assets)
            premium = self._calculate_premium(x)
            assets = self.manufacturer.assets
            cost_obj = weights["cost"] * (premium / assets) * 10  # Scale

            total = growth_obj + risk_obj + cost_obj
            return total

        except Exception as e:
            logger.error(f"Error calculating objective: {e}")
            return 1e10

    def _calculate_premium(self, x: np.ndarray) -> float:
        """Calculate total premium for given structure."""
        retained_limit = x[0]
        layer_limits = x[1:]

        total_premium = 0
        current_attachment = retained_limit

        for limit in layer_limits:
            if limit > 1000:  # Minimum meaningful layer
                # Use pricing scenario rates
                if current_attachment < 5_000_000:
                    rate = self.current_scenario.primary_layer_rate
                elif current_attachment < 25_000_000:
                    rate = self.current_scenario.first_excess_rate
                else:
                    rate = self.current_scenario.higher_excess_rate

                premium = limit * rate
                total_premium += premium
                current_attachment += limit

        return total_premium

    def _estimate_growth_rate(self, retained_limit: float, layer_limits: List[float]) -> float:
        """Estimate ergodic growth rate for given insurance structure."""
        # Simplified estimation - in practice would run full simulation
        base_growth = 0.08  # 8% base growth

        # Insurance benefit increases growth by reducing volatility drag
        coverage = sum(layer_limits)
        coverage_ratio = coverage / self.manufacturer.assets

        # Estimate volatility reduction
        volatility_reduction = min(coverage_ratio * 0.3, 0.15)  # Max 15% reduction

        # Ergodic growth benefit
        growth_benefit = volatility_reduction * 0.5  # Simplified

        return base_growth + growth_benefit

    def _calculate_cvar(self, losses: np.ndarray, percentile: float) -> float:
        """Calculate Conditional Value at Risk (CVaR).

        Args:
            losses: Array of loss values
            percentile: Percentile threshold (e.g., 95)

        Returns:
            CVaR value or 0 if no losses exceed threshold
        """
        if len(losses) == 0:
            return 0.0

        threshold = np.percentile(losses, percentile)
        tail_losses = losses[losses > threshold]

        if len(tail_losses) == 0:
            return float(threshold)  # If no losses exceed threshold, return the threshold itself

        return float(np.mean(tail_losses))

    def _estimate_bankruptcy_probability(self, x: np.ndarray) -> float:
        """Estimate bankruptcy probability for given structure."""
        retained_limit = x[0]
        layer_limits = x[1:]

        # Total coverage
        total_coverage = retained_limit + sum(l for l in layer_limits if l > 1000)

        # Simple estimation based on coverage adequacy
        # Estimate max loss using expected value if available
        if hasattr(self.loss_distribution, "expected_value"):
            expected_max_loss = self.loss_distribution.expected_value() * 10
        else:
            expected_max_loss = total_coverage  # Conservative fallback

        coverage_ratio = total_coverage / expected_max_loss if expected_max_loss > 0 else 1.0

        # Map coverage ratio to bankruptcy probability
        if coverage_ratio >= 1.0:
            return 0.001  # Very low if fully covered
        if coverage_ratio >= 0.8:
            return 0.005
        if coverage_ratio >= 0.6:
            return 0.01
        if coverage_ratio >= 0.4:
            return 0.02
        return 0.05

    def _create_decision_from_result(
        self, result: OptimizeResult, method: OptimizationMethod
    ) -> InsuranceDecision:
        """Create InsuranceDecision from optimization result."""
        x = result.x
        retained_limit = x[0]
        layer_limits = x[1:]

        # Create Layer objects
        layers = []
        current_attachment = retained_limit

        for limit in layer_limits:
            if limit > 1000:  # Minimum meaningful layer
                # Determine rate based on attachment
                if current_attachment < 5_000_000:
                    rate = self.current_scenario.primary_layer_rate
                elif current_attachment < 25_000_000:
                    rate = self.current_scenario.first_excess_rate
                else:
                    rate = self.current_scenario.higher_excess_rate

                layer = Layer(
                    attachment_point=current_attachment,
                    limit=limit,
                    premium_rate=rate,
                )
                layers.append(layer)
                current_attachment += limit

        return InsuranceDecision(
            retained_limit=retained_limit,
            layers=layers,
            total_premium=sum(l.limit * l.premium_rate for l in layers),
            total_coverage=retained_limit + sum(l.limit for l in layers),
            pricing_scenario=self.pricing_scenario,
            optimization_method=method.value,
            convergence_iterations=result.nit if hasattr(result, "nit") else 0,
            objective_value=result.fun,
        )

    def _validate_decision(
        self, decision: InsuranceDecision, constraints: OptimizationConstraints
    ) -> bool:
        """Validate that decision meets all constraints."""
        # Check premium budget
        if decision.total_premium > constraints.max_premium_budget:
            logger.warning(f"Premium ${decision.total_premium:,.0f} exceeds budget")
            return False

        # Check coverage limits
        if decision.total_coverage < constraints.min_coverage_limit:
            logger.warning(f"Coverage ${decision.total_coverage:,.0f} below minimum")
            return False
        if decision.total_coverage > constraints.max_coverage_limit:
            logger.warning(f"Coverage ${decision.total_coverage:,.0f} above maximum")
            return False

        # Check retention limits
        if decision.retained_limit < constraints.min_retained_limit:
            return False
        if decision.retained_limit > constraints.max_retained_limit:
            return False

        # Check layer count
        if len(decision.layers) > constraints.max_layers:
            return False

        return True

    def calculate_decision_metrics(self, decision: InsuranceDecision) -> DecisionMetrics:
        """Calculate comprehensive metrics for a decision.

        Args:
            decision: Insurance decision to evaluate

        Returns:
            Comprehensive metrics
        """
        # Check cache
        cache_key = f"{decision.retained_limit}_{len(decision.layers)}_{decision.total_premium}"
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        # Run simulations to calculate metrics
        n_simulations = 1000
        time_horizon = 10  # years

        # Simulate with insurance
        with_insurance_results = self._run_simulation(decision, n_simulations, time_horizon)

        # Simulate without insurance
        no_insurance_decision = InsuranceDecision(
            retained_limit=float("inf"),
            layers=[],
            total_premium=0,
            total_coverage=0,
            pricing_scenario=self.pricing_scenario,
            optimization_method="none",
        )
        without_insurance_results = self._run_simulation(
            no_insurance_decision, n_simulations, time_horizon
        )

        # Import ROEAnalyzer for enhanced metrics
        from .risk_metrics import ROEAnalyzer

        # Calculate enhanced ROE metrics if we have detailed ROE data
        roe_data = with_insurance_results.get("roe_series", None)
        equity_data = with_insurance_results.get("equity_series", None)

        if roe_data is not None and len(roe_data) > 0:
            roe_analyzer = ROEAnalyzer(roe_data, equity_data)

            # Get all enhanced metrics
            volatility_metrics = roe_analyzer.volatility_metrics()
            performance_ratios = roe_analyzer.performance_ratios()
            rolling_stats_1yr = roe_analyzer.rolling_statistics(1) if len(roe_data) >= 1 else {}
            rolling_stats_3yr = roe_analyzer.rolling_statistics(3) if len(roe_data) >= 3 else {}
            rolling_stats_5yr = roe_analyzer.rolling_statistics(5) if len(roe_data) >= 5 else {}

            # Calculate component breakdown (simplified version)
            base_operating_roe = 0.08 / 0.3  # Assuming 8% margin and 30% equity ratio
            insurance_cost_impact = -decision.total_premium / (self.manufacturer.assets * 0.3)

            time_weighted_roe = roe_analyzer.time_weighted_average()
            roe_volatility = volatility_metrics.get("standard_deviation", 0.0)
            roe_sharpe = performance_ratios.get("sharpe_ratio", 0.0)
            roe_downside_dev = volatility_metrics.get("downside_deviation", 0.0)
            roe_1yr = np.nanmean(rolling_stats_1yr.get("mean", [0.0]))
            roe_3yr = np.nanmean(rolling_stats_3yr.get("mean", [0.0]))
            roe_5yr = np.nanmean(rolling_stats_5yr.get("mean", [0.0]))
        else:
            # Fallback to simple calculations
            time_weighted_roe = np.mean(with_insurance_results["roe"])
            roe_volatility = np.std(with_insurance_results["roe"])
            roe_sharpe = (np.mean(with_insurance_results["roe"]) - 0.02) / max(
                roe_volatility, 0.001
            )
            roe_downside_dev = roe_volatility * 0.7  # Rough approximation
            roe_1yr = np.mean(with_insurance_results["roe"])
            roe_3yr = np.mean(with_insurance_results["roe"])
            roe_5yr = np.mean(with_insurance_results["roe"])
            base_operating_roe = 0.08 / 0.3
            insurance_cost_impact = -decision.total_premium / (self.manufacturer.assets * 0.3)

        # Calculate metrics
        metrics = DecisionMetrics(
            ergodic_growth_rate=np.mean(with_insurance_results["growth_rates"]),
            bankruptcy_probability=np.mean(with_insurance_results["bankruptcies"]),
            expected_roe=np.mean(with_insurance_results["roe"]),
            roe_improvement=(
                np.mean(with_insurance_results["roe"]) - np.mean(without_insurance_results["roe"])
            ),
            premium_to_limit_ratio=(
                decision.total_premium / decision.total_coverage
                if decision.total_coverage > 0
                else 0
            ),
            coverage_adequacy=(
                min(decision.total_coverage / (self.loss_distribution.expected_value() * 10), 1.0)
                if decision.total_coverage > 0 and hasattr(self.loss_distribution, "expected_value")
                else 0.0
            ),
            capital_efficiency=(
                np.mean(with_insurance_results["value"])
                - np.mean(without_insurance_results["value"])
            )
            / max(decision.total_premium, 1),
            value_at_risk_95=(
                np.percentile(with_insurance_results["losses"], 95)
                if len(with_insurance_results["losses"]) > 0
                else 0.0
            ),
            conditional_value_at_risk=(
                self._calculate_cvar(with_insurance_results["losses"], 95)
                if len(with_insurance_results["losses"]) > 0
                else 0.0
            ),
            # Enhanced ROE metrics
            time_weighted_roe=time_weighted_roe,
            roe_volatility=roe_volatility,
            roe_sharpe_ratio=roe_sharpe,
            roe_downside_deviation=roe_downside_dev,
            roe_1yr_rolling=roe_1yr,
            roe_3yr_rolling=roe_3yr,
            roe_5yr_rolling=roe_5yr,
            # ROE component breakdown
            operating_roe=base_operating_roe,
            insurance_impact_roe=insurance_cost_impact,
            tax_effect_roe=-0.25 * np.mean(with_insurance_results["roe"]),  # 25% tax rate impact
        )

        # Calculate overall score
        metrics.calculate_score()

        # Cache result
        self._metrics_cache[cache_key] = metrics

        return metrics

    def _run_simulation(
        self, decision: InsuranceDecision, n_simulations: int, time_horizon: int
    ) -> Dict[str, np.ndarray]:
        """Run Monte Carlo simulation for given decision."""
        results = {
            "growth_rates": np.zeros(n_simulations),
            "bankruptcies": np.zeros(n_simulations),
            "roe": np.zeros(n_simulations),
            "value": np.zeros(n_simulations),
            "losses": np.zeros(n_simulations),
            "roe_series": [],  # Store full ROE time series
            "equity_series": [],  # Store equity time series
        }

        # Collect all ROE and equity series for enhanced analysis
        all_roe_series = []
        all_equity_series = []

        for i in range(n_simulations):
            # Initialize company state
            assets = self.manufacturer.assets
            equity = assets * 0.3  # Assume 30% equity ratio

            bankrupt = False
            annual_returns = []
            sim_roe_series = []
            sim_equity_series = []

            for _ in range(time_horizon):
                # Generate revenue
                revenue = assets * self.manufacturer.asset_turnover_ratio

                # Generate losses
                if hasattr(self.loss_distribution, "expected_value"):
                    # Simple lognormal approximation for losses
                    annual_losses = np.random.lognormal(
                        np.log(max(self.loss_distribution.expected_value(), 1)), 0.5
                    )
                else:
                    annual_losses = 0.0

                # Apply insurance
                retained_losses = min(annual_losses, decision.retained_limit)
                insured_losses = 0

                if decision.layers:
                    remaining_loss = max(annual_losses - decision.retained_limit, 0)
                    for layer in decision.layers:
                        layer_loss = min(remaining_loss, layer.limit)
                        insured_losses += layer_loss
                        remaining_loss -= layer_loss
                        if remaining_loss <= 0:
                            break

                # Calculate net income
                operating_income = revenue * self.manufacturer.operating_margin
                net_losses = retained_losses + max(annual_losses - decision.total_coverage, 0)
                net_income = operating_income - net_losses - decision.total_premium

                # Calculate ROE before updating equity
                if equity > 0:
                    roe = net_income / equity
                    annual_returns.append(roe)
                    sim_roe_series.append(roe)
                    sim_equity_series.append(equity)
                else:
                    annual_returns.append(net_income / max(equity, 1))

                # Update equity
                equity += net_income

                # Check bankruptcy
                if equity <= 0:
                    bankrupt = True
                    break

                # Update assets for next period
                assets = equity / 0.3

            # Store results
            results["growth_rates"][i] = np.mean(annual_returns) if annual_returns else 0
            results["bankruptcies"][i] = 1 if bankrupt else 0
            results["roe"][i] = np.mean(annual_returns) if annual_returns else 0
            results["value"][i] = equity
            results["losses"][i] = annual_losses if "annual_losses" in locals() else 0

            # Store series for enhanced analysis
            all_roe_series.extend(sim_roe_series)
            all_equity_series.extend(sim_equity_series)

        # Convert to numpy arrays for analysis
        results["roe_series"] = np.array(all_roe_series) if all_roe_series else np.array([])
        results["equity_series"] = (
            np.array(all_equity_series) if all_equity_series else np.array([])
        )

        return results

    def run_sensitivity_analysis(
        self,
        base_decision: InsuranceDecision,
        parameters: Optional[List[str]] = None,
        variation_range: float = 0.2,
    ) -> SensitivityReport:
        """Analyze decision sensitivity to parameter changes.

        Args:
            base_decision: Base decision to analyze
            parameters: Parameters to test (default: key parameters)
            variation_range: ±% to vary parameters (default: 20%)

        Returns:
            Comprehensive sensitivity report
        """
        if parameters is None:
            parameters = [
                "premium_rates",
                "loss_frequency",
                "loss_severity",
                "growth_rate",
                "capital_base",
            ]

        logger.info(f"Running sensitivity analysis for {len(parameters)} parameters")

        # Calculate base metrics
        base_metrics = self.calculate_decision_metrics(base_decision)

        # Initialize results
        parameter_sensitivities = {}
        stress_test_results = {}

        for param in parameters:
            param_results = {}

            # Test parameter variations
            for variation in [-variation_range, variation_range]:
                # Modify parameter
                modified_scenario = self._modify_parameter(param, variation)

                # Re-optimize with modified parameter
                constraints = OptimizationConstraints(
                    max_premium_budget=base_decision.total_premium * 1.1
                )
                modified_decision = self.optimize_insurance_decision(constraints)

                # Calculate metrics
                modified_metrics = self.calculate_decision_metrics(modified_decision)

                # Calculate sensitivity
                label = "decrease" if variation < 0 else "increase"
                param_results[label] = {
                    "growth_change": (
                        modified_metrics.ergodic_growth_rate - base_metrics.ergodic_growth_rate
                    ),
                    "risk_change": (
                        modified_metrics.bankruptcy_probability
                        - base_metrics.bankruptcy_probability
                    ),
                    "roe_change": modified_metrics.expected_roe - base_metrics.expected_roe,
                }

                # Store stress test results
                stress_test_results[f"{param}_{label}"] = modified_metrics

            parameter_sensitivities[param] = param_results

        # Flatten parameter sensitivities for the report
        flattened_sensitivities = {}
        for param, results in parameter_sensitivities.items():
            # Calculate average impact across variations
            avg_growth_change = np.mean(
                [abs(results[v]["growth_change"]) for v in ["decrease", "increase"]]
            )
            avg_risk_change = np.mean(
                [abs(results[v]["risk_change"]) for v in ["decrease", "increase"]]
            )
            avg_roe_change = np.mean(
                [abs(results[v]["roe_change"]) for v in ["decrease", "increase"]]
            )

            flattened_sensitivities[param] = {
                "growth_sensitivity": float(avg_growth_change),
                "risk_sensitivity": float(avg_risk_change),
                "roe_sensitivity": float(avg_roe_change),
            }

        # Identify key drivers (parameters with highest impact)
        impacts = []
        for param, metrics in flattened_sensitivities.items():
            total_impact = float(
                metrics["growth_sensitivity"]
                + metrics["risk_sensitivity"] * 10
                + metrics["roe_sensitivity"]
            )
            impacts.append((param, total_impact))

        key_drivers = [p for p, _ in sorted(impacts, key=lambda x: float(x[1]), reverse=True)][:3]

        # Determine robust ranges
        robust_range = {}
        for param in parameters:
            # Find range where decision remains stable
            # Simplified: use fixed range for now
            robust_range[param] = (-0.1, 0.1)  # ±10% is typically stable

        return SensitivityReport(
            base_decision=base_decision,
            base_metrics=base_metrics,
            parameter_sensitivities=flattened_sensitivities,
            key_drivers=key_drivers,
            robust_range=robust_range,
            stress_test_results=stress_test_results,
        )

    def _modify_parameter(self, parameter: str, variation: float) -> Any:
        """Modify a parameter for sensitivity analysis."""
        # Store original state
        original_state = {}

        if parameter == "premium_rates":
            # Modify pricing scenario rates
            for attr in ["primary_layer_rate", "first_excess_rate", "higher_excess_rate"]:
                original = getattr(self.current_scenario, attr)
                original_state[attr] = original
                setattr(self.current_scenario, attr, original * (1 + variation))

        elif parameter == "loss_frequency":
            # Modify loss distribution frequency
            if hasattr(self.loss_distribution, "frequency"):
                original_state["frequency"] = self.loss_distribution.frequency
                self.loss_distribution.frequency *= 1 + variation

        elif parameter == "capital_base":
            # Modify manufacturer capital
            original_state["assets"] = self.manufacturer.assets
            self.manufacturer.assets *= 1 + variation

        # Return original state for restoration
        return original_state

    def generate_recommendations(
        self, analysis_results: List[Tuple[InsuranceDecision, DecisionMetrics]]
    ) -> Recommendations:
        """Generate executive-ready recommendations.

        Args:
            analysis_results: List of (decision, metrics) tuples to analyze

        Returns:
            Comprehensive recommendations
        """
        if not analysis_results:
            raise ValueError("No analysis results provided")

        # Sort by decision score
        sorted_results = sorted(analysis_results, key=lambda x: x[1].decision_score, reverse=True)

        # Select primary recommendation
        primary_decision, primary_metrics = sorted_results[0]

        # Generate rationale
        primary_rationale = self._generate_rationale(primary_decision, primary_metrics)

        # Select alternatives (top 3 after primary)
        alternatives = []
        for decision, metrics in sorted_results[1:4]:
            rationale = self._generate_brief_rationale(decision, metrics)
            alternatives.append((decision, rationale))

        # Generate implementation timeline
        timeline = self._generate_timeline(primary_decision)

        # Identify risk considerations
        risk_considerations = self._identify_risks(primary_decision, primary_metrics)

        # Calculate expected benefits
        expected_benefits = {
            "ROE Improvement": primary_metrics.roe_improvement,
            "Risk Reduction": 1 - primary_metrics.bankruptcy_probability,
            "Growth Enhancement": primary_metrics.ergodic_growth_rate,
            "Capital Efficiency": primary_metrics.capital_efficiency,
        }

        # Calculate confidence level
        confidence = self._calculate_confidence(primary_metrics, sorted_results)

        return Recommendations(
            primary_recommendation=primary_decision,
            primary_rationale=primary_rationale,
            alternative_options=alternatives,
            implementation_timeline=timeline,
            risk_considerations=risk_considerations,
            expected_benefits=expected_benefits,
            confidence_level=confidence,
        )

    def _generate_rationale(self, decision: InsuranceDecision, metrics: DecisionMetrics) -> str:
        """Generate detailed rationale for a decision."""
        rationale = f"""
        This insurance structure optimizes long-term value creation by:

        1. **Growth Enhancement**: Achieves {metrics.ergodic_growth_rate:.1%} ergodic growth rate,
           representing a {metrics.roe_improvement:.1%} improvement over self-insurance.

        2. **Risk Management**: Reduces bankruptcy probability to {metrics.bankruptcy_probability:.2%},
           providing robust downside protection while maintaining upside potential.

        3. **Capital Efficiency**: Delivers ${metrics.capital_efficiency:.2f} of value per dollar
           of premium spent, with coverage adequacy of {metrics.coverage_adequacy:.0%}.

        4. **Structure**: {len(decision.layers)} layers totaling ${decision.total_coverage/1e6:.1f}M
           in coverage for ${decision.total_premium/1e3:.0f}K annual premium.

        The structure is optimized for {decision.pricing_scenario} market conditions using
        {decision.optimization_method} optimization.
        """
        return rationale.strip()

    def _generate_brief_rationale(
        self, decision: InsuranceDecision, metrics: DecisionMetrics
    ) -> str:
        """Generate brief rationale for alternative option."""
        return (
            f"{len(decision.layers)} layers, ${decision.total_premium/1e3:.0f}K premium, "
            f"{metrics.ergodic_growth_rate:.1%} growth, {metrics.bankruptcy_probability:.2%} risk"
        )

    def _generate_timeline(self, decision: InsuranceDecision) -> List[str]:
        """Generate implementation timeline."""
        return [
            "Week 1-2: Finalize insurance specifications and requirements",
            "Week 3-4: Solicit quotes from insurance markets",
            "Week 5-6: Negotiate terms and pricing",
            "Week 7: Execute insurance contracts",
            "Week 8: Implement coverage and update risk management procedures",
            "Ongoing: Monitor performance and adjust as needed",
        ]

    def _identify_risks(self, decision: InsuranceDecision, metrics: DecisionMetrics) -> List[str]:
        """Identify key risks in the recommendation."""
        risks = []

        if metrics.bankruptcy_probability > 0.01:
            risks.append(
                f"Residual bankruptcy risk of {metrics.bankruptcy_probability:.2%} remains"
            )

        if decision.total_premium > 1_000_000:
            risks.append(
                f"Significant premium commitment of ${decision.total_premium/1e6:.1f}M annually"
            )

        if len(decision.layers) > 3:
            risks.append(f"Complex structure with {len(decision.layers)} layers to manage")

        if metrics.coverage_adequacy < 0.8:
            risks.append(
                f"Coverage may be insufficient for extreme events (adequacy: {metrics.coverage_adequacy:.0%})"
            )

        risks.append("Market conditions may change, affecting renewal pricing")
        risks.append("Actual losses may differ from modeled distributions")

        return risks

    def _calculate_confidence(self, primary_metrics: DecisionMetrics, all_results: List) -> float:
        """Calculate confidence level in recommendation."""
        confidence = 0.5  # Base confidence

        # Higher score increases confidence
        if primary_metrics.decision_score > 0.8:
            confidence += 0.2
        elif primary_metrics.decision_score > 0.6:
            confidence += 0.1

        # Low bankruptcy probability increases confidence
        if primary_metrics.bankruptcy_probability < 0.005:
            confidence += 0.15
        elif primary_metrics.bankruptcy_probability < 0.01:
            confidence += 0.1

        # Clear separation from alternatives increases confidence
        if len(all_results) > 1:
            score_gap = primary_metrics.decision_score - all_results[1][1].decision_score
            if score_gap > 0.1:
                confidence += 0.1

        # Positive ROE improvement increases confidence
        if primary_metrics.roe_improvement > 0.02:
            confidence += 0.05

        return min(confidence, 0.95)  # Cap at 95%

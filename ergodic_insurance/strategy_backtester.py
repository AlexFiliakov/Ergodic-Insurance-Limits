"""Strategy backtesting framework for insurance decision strategies.

This module provides base classes and implementations for various insurance
strategies that can be tested and compared in walk-forward validation.

Example:
    >>> from strategy_backtester import ConservativeFixedStrategy, StrategyBacktester
    >>> from simulation import SimulationEngine

    >>> # Create and configure a strategy
    >>> strategy = ConservativeFixedStrategy(
    ...     primary_limit=5000000,
    ...     excess_limit=20000000,
    ...     deductible=100000
    ... )
    >>>
    >>> # Run backtest
    >>> backtester = StrategyBacktester(simulation_engine)
    >>> results = backtester.test_strategy(
    ...     strategy=strategy,
    ...     n_simulations=1000,
    ...     n_years=10
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .insurance import InsuranceLayer, InsurancePolicy
from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .monte_carlo import MonteCarloEngine, MonteCarloResults, SimulationConfig
from .optimization import PenaltyMethodOptimizer
from .simulation import Simulation, SimulationResults
from .validation_metrics import MetricCalculator, ValidationMetrics

logger = logging.getLogger(__name__)


class InsuranceStrategy(ABC):
    """Abstract base class for insurance strategies.

    Defines the interface that all insurance strategies must implement
    for use in backtesting and walk-forward validation.
    """

    def __init__(self, name: str):
        """Initialize strategy.

        Args:
            name: Strategy name for identification.
        """
        self.name = name
        self.metadata: Dict[str, Any] = {}
        self.adaptation_history: List[Dict[str, Any]] = []

    @abstractmethod
    def get_insurance_program(
        self,
        manufacturer: WidgetManufacturer,
        historical_losses: Optional[np.ndarray] = None,
        current_year: int = 0,
    ) -> Optional[InsuranceProgram]:
        """Get insurance program for the current state.

        Args:
            manufacturer: Current manufacturer state
            historical_losses: Past loss data for adaptive strategies
            current_year: Current year in simulation

        Returns:
            InsuranceProgram or None for no insurance.
        """

    def update(self, losses: np.ndarray, recoveries: np.ndarray, year: int):
        """Update strategy based on recent experience.

        Args:
            losses: Recent loss amounts
            recoveries: Recent recovery amounts
            year: Current year
        """

    def reset(self):
        """Reset strategy to initial state."""
        self.adaptation_history.clear()

    def get_description(self) -> str:
        """Get strategy description.

        Returns:
            Human-readable strategy description.
        """
        return f"{self.name} strategy"


class NoInsuranceStrategy(InsuranceStrategy):
    """Baseline strategy with no insurance."""

    def __init__(self):
        """Initialize no insurance strategy."""
        super().__init__("No Insurance")

    def get_insurance_program(
        self,
        manufacturer: WidgetManufacturer,
        historical_losses: Optional[np.ndarray] = None,
        current_year: int = 0,
    ) -> Optional[InsuranceProgram]:
        """Return no insurance program.

        Returns:
            None to indicate no insurance.
        """
        return None


class ConservativeFixedStrategy(InsuranceStrategy):
    """Conservative strategy with high limits and low deductible."""

    def __init__(
        self,
        primary_limit: float = 5000000,
        excess_limit: float = 20000000,
        higher_limit: float = 25000000,
        deductible: float = 50000,
    ):
        """Initialize conservative strategy.

        Args:
            primary_limit: Primary layer limit
            excess_limit: Excess layer limit
            higher_limit: Higher excess layer limit
            deductible: Deductible amount
        """
        super().__init__("Conservative Fixed")
        self.primary_limit = primary_limit
        self.excess_limit = excess_limit
        self.higher_limit = higher_limit
        self.deductible = deductible

    def get_insurance_program(
        self,
        manufacturer: WidgetManufacturer,
        historical_losses: Optional[np.ndarray] = None,
        current_year: int = 0,
    ) -> Optional[InsuranceProgram]:
        """Get conservative insurance program.

        Returns:
            InsuranceProgram with high coverage.
        """
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=self.deductible,
                limit=self.primary_limit - self.deductible,
                base_premium_rate=0.015,
                reinstatements=1,
            ),
            EnhancedInsuranceLayer(
                attachment_point=self.primary_limit,
                limit=self.excess_limit,
                base_premium_rate=0.008,
                reinstatements=1,
            ),
            EnhancedInsuranceLayer(
                attachment_point=self.primary_limit + self.excess_limit,
                limit=self.higher_limit,
                base_premium_rate=0.004,
                reinstatements=0,
            ),
        ]

        return InsuranceProgram(layers=layers, deductible=self.deductible)


class AggressiveFixedStrategy(InsuranceStrategy):
    """Aggressive strategy with low limits and high deductible."""

    def __init__(
        self,
        primary_limit: float = 2000000,
        excess_limit: float = 5000000,
        deductible: float = 250000,
    ):
        """Initialize aggressive strategy.

        Args:
            primary_limit: Primary layer limit
            excess_limit: Excess layer limit
            deductible: Deductible amount
        """
        super().__init__("Aggressive Fixed")
        self.primary_limit = primary_limit
        self.excess_limit = excess_limit
        self.deductible = deductible

    def get_insurance_program(
        self,
        manufacturer: WidgetManufacturer,
        historical_losses: Optional[np.ndarray] = None,
        current_year: int = 0,
    ) -> Optional[InsuranceProgram]:
        """Get aggressive insurance program.

        Returns:
            InsuranceProgram with limited coverage.
        """
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=self.deductible,
                limit=self.primary_limit - self.deductible,
                base_premium_rate=0.012,
                reinstatements=0,
            ),
            EnhancedInsuranceLayer(
                attachment_point=self.primary_limit,
                limit=self.excess_limit,
                base_premium_rate=0.006,
                reinstatements=0,
            ),
        ]

        return InsuranceProgram(layers=layers, deductible=self.deductible)


class OptimizedStaticStrategy(InsuranceStrategy):
    """Strategy using optimization to find best static limits."""

    def __init__(
        self,
        optimizer: Optional[PenaltyMethodOptimizer] = None,
        target_roe: float = 0.15,
        max_ruin_prob: float = 0.01,
    ):
        """Initialize optimized strategy.

        Args:
            optimizer: Optimizer instance to use
            target_roe: Target ROE for optimization
            max_ruin_prob: Maximum acceptable ruin probability
        """
        super().__init__("Optimized Static")
        self.optimizer = optimizer
        self.target_roe = target_roe
        self.max_ruin_prob = max_ruin_prob
        self.optimized_params: Optional[Dict[str, float]] = None

    def optimize_limits(self, manufacturer: WidgetManufacturer, simulation_engine: Simulation):
        """Run optimization to find best limits.

        Args:
            manufacturer: Manufacturer instance
            simulation_engine: Simulation engine for evaluation
        """

        # Define optimization problem
        def objective(x):
            # x = [deductible, primary_limit, excess_limit]
            deductible, primary, excess = x

            # Create trial insurance program
            layers = [
                InsuranceLayer(deductible, primary - deductible, 0.015),
                InsuranceLayer(primary, excess, 0.008),
            ]
            policy = InsurancePolicy(layers, deductible)

            # Run short simulation to evaluate
            # Run short simulation using MonteCarloEngine
            from .monte_carlo import MonteCarloEngine
            from .monte_carlo import SimulationConfig as MCConfig

            mc_config = MCConfig(n_simulations=100, n_years=5)

            # Create loss generator
            loss_generator = ManufacturingLossGenerator(seed=42)

            # Create insurance program from policy
            from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram

            program_layers = [
                EnhancedInsuranceLayer(
                    attachment_point=layer.attachment_point,
                    limit=layer.limit,
                    base_premium_rate=layer.rate,
                )
                for layer in policy.layers
            ]
            program = InsuranceProgram(layers=program_layers)

            # Initialize Monte Carlo engine with required parameters
            mc_engine = MonteCarloEngine(
                loss_generator=loss_generator,
                insurance_program=program,
                manufacturer=manufacturer,
                config=mc_config,
            )

            results = mc_engine.run()

            # Maximize ROE (minimize negative ROE)
            return -results.metrics.get("mean_roe", 0)

        # Define constraints
        def ruin_constraint(x):
            deductible, primary, excess = x

            layers = [
                InsuranceLayer(deductible, primary - deductible, 0.015),
                InsuranceLayer(primary, excess, 0.008),
            ]
            policy = InsurancePolicy(layers, deductible)

            # Run short simulation using MonteCarloEngine
            from .monte_carlo import MonteCarloEngine
            from .monte_carlo import SimulationConfig as MCConfig

            mc_config = MCConfig(n_simulations=100, n_years=5)

            # Create loss generator
            loss_generator = ManufacturingLossGenerator(seed=42)

            # Create insurance program from policy
            from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram

            program_layers = [
                EnhancedInsuranceLayer(
                    attachment_point=layer.attachment_point,
                    limit=layer.limit,
                    base_premium_rate=layer.rate,
                )
                for layer in policy.layers
            ]
            program = InsuranceProgram(layers=program_layers)

            # Initialize Monte Carlo engine with required parameters
            mc_engine = MonteCarloEngine(
                loss_generator=loss_generator,
                insurance_program=program,
                manufacturer=manufacturer,
                config=mc_config,
            )

            results = mc_engine.run()

            # Constraint: ruin_prob <= max_ruin_prob
            # Extract ruin probability for the final year
            ruin_prob_value = results.ruin_probability.get(
                str(mc_config.n_years),
                list(results.ruin_probability.values())[-1] if results.ruin_probability else 0.0,
            )
            return self.max_ruin_prob - ruin_prob_value

        # Run optimization using scipy minimize
        from scipy.optimize import minimize

        bounds = [(50000, 500000), (1000000, 10000000), (5000000, 50000000)]
        constraints = [{"type": "ineq", "fun": ruin_constraint}]
        x0 = [100000, 5000000, 20000000]  # Initial guess

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        if result.success:
            self.optimized_params = {
                "deductible": result.x[0],
                "primary_limit": result.x[1],
                "excess_limit": result.x[2],
            }
            logger.info(f"Optimization successful: {self.optimized_params}")
        else:
            # Fall back to conservative defaults
            self.optimized_params = {
                "deductible": 100000,
                "primary_limit": 5000000,
                "excess_limit": 20000000,
            }
            logger.warning(f"Optimization failed, using defaults: {self.optimized_params}")

    def get_insurance_program(
        self,
        manufacturer: WidgetManufacturer,
        historical_losses: Optional[np.ndarray] = None,
        current_year: int = 0,
    ) -> Optional[InsuranceProgram]:
        """Get optimized insurance program.

        Returns:
            InsuranceProgram with optimized parameters.
        """
        if not self.optimized_params:
            # Use conservative defaults if not optimized
            self.optimized_params = {
                "deductible": 100000,
                "primary_limit": 5000000,
                "excess_limit": 20000000,
            }

        layers = [
            EnhancedInsuranceLayer(
                attachment_point=self.optimized_params["deductible"],
                limit=self.optimized_params["primary_limit"] - self.optimized_params["deductible"],
                base_premium_rate=0.015,
                reinstatements=1,
            ),
            EnhancedInsuranceLayer(
                attachment_point=self.optimized_params["primary_limit"],
                limit=self.optimized_params["excess_limit"],
                base_premium_rate=0.008,
                reinstatements=0,
            ),
        ]

        return InsuranceProgram(layers=layers, deductible=self.optimized_params["deductible"])


class AdaptiveStrategy(InsuranceStrategy):
    """Strategy that adjusts based on recent loss experience."""

    def __init__(
        self,
        base_deductible: float = 100000,
        base_primary: float = 3000000,
        base_excess: float = 10000000,
        adaptation_window: int = 3,
        adjustment_factor: float = 0.2,
    ):
        """Initialize adaptive strategy.

        Args:
            base_deductible: Base deductible amount
            base_primary: Base primary limit
            base_excess: Base excess limit
            adaptation_window: Years of history to consider
            adjustment_factor: How much to adjust limits (0-1)
        """
        super().__init__("Adaptive")
        self.base_deductible = base_deductible
        self.base_primary = base_primary
        self.base_excess = base_excess
        self.adaptation_window = adaptation_window
        self.adjustment_factor = adjustment_factor

        # Current adjusted parameters
        self.current_deductible = base_deductible
        self.current_primary = base_primary
        self.current_excess = base_excess

        # Loss history for adaptation
        self.loss_history: List[float] = []

    def update(self, losses: np.ndarray, recoveries: np.ndarray, year: int):
        """Update strategy based on recent losses.

        Args:
            losses: Recent loss amounts
            recoveries: Recent recovery amounts
            year: Current year
        """
        # Add to history
        total_losses = float(np.sum(losses))
        self.loss_history.append(total_losses)

        # Keep only recent history
        if len(self.loss_history) > self.adaptation_window:
            self.loss_history = self.loss_history[-self.adaptation_window :]

        # Adapt if we have enough history
        if len(self.loss_history) >= 2:
            avg_losses = np.mean(self.loss_history)
            recent_losses = self.loss_history[-1]

            # Calculate adjustment ratio
            if avg_losses > 0:
                ratio = float(recent_losses / avg_losses)
            else:
                ratio = 1.0

            # Adjust limits based on recent experience
            if ratio > 1.5:  # Recent losses much higher than average
                # Increase coverage
                adjustment = float(1 + self.adjustment_factor * (ratio - 1))
                self.current_primary = float(
                    min(self.base_primary * adjustment, self.base_primary * 2)
                )
                self.current_excess = float(
                    min(self.base_excess * adjustment, self.base_excess * 2)
                )
                self.current_deductible = float(
                    max(self.base_deductible / adjustment, self.base_deductible * 0.5)
                )
            elif ratio < 0.5:  # Recent losses much lower than average
                # Decrease coverage
                adjustment = float(1 - self.adjustment_factor * (1 - ratio))
                self.current_primary = float(
                    max(self.base_primary * adjustment, self.base_primary * 0.5)
                )
                self.current_excess = float(
                    max(self.base_excess * adjustment, self.base_excess * 0.5)
                )
                self.current_deductible = float(
                    min(self.base_deductible / adjustment, self.base_deductible * 2)
                )
            else:
                # Gradually return to base levels
                self.current_primary = 0.9 * self.current_primary + 0.1 * self.base_primary
                self.current_excess = 0.9 * self.current_excess + 0.1 * self.base_excess
                self.current_deductible = 0.9 * self.current_deductible + 0.1 * self.base_deductible

            # Record adaptation
            self.adaptation_history.append(
                {
                    "year": year,
                    "avg_losses": avg_losses,
                    "recent_losses": recent_losses,
                    "ratio": ratio,
                    "deductible": self.current_deductible,
                    "primary": self.current_primary,
                    "excess": self.current_excess,
                }
            )

    def get_insurance_program(
        self,
        manufacturer: WidgetManufacturer,
        historical_losses: Optional[np.ndarray] = None,
        current_year: int = 0,
    ) -> Optional[InsuranceProgram]:
        """Get adaptive insurance program.

        Returns:
            InsuranceProgram with adapted parameters.
        """
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=self.current_deductible,
                limit=self.current_primary - self.current_deductible,
                base_premium_rate=0.014,  # Slightly lower due to adaptability
                reinstatements=1,
            ),
            EnhancedInsuranceLayer(
                attachment_point=self.current_primary,
                limit=self.current_excess,
                base_premium_rate=0.007,
                reinstatements=0,
            ),
        ]

        return InsuranceProgram(layers=layers, deductible=self.current_deductible)

    def reset(self):
        """Reset strategy to initial state."""
        super().reset()
        self.current_deductible = self.base_deductible
        self.current_primary = self.base_primary
        self.current_excess = self.base_excess
        self.loss_history.clear()


@dataclass
class BacktestResult:
    """Results from strategy backtesting.

    Attributes:
        strategy_name: Name of tested strategy
        simulation_results: Raw simulation results (either Simulation or MC results)
        metrics: Calculated performance metrics
        execution_time: Time taken to run backtest
        config: Configuration used for backtest
    """

    strategy_name: str
    simulation_results: Union[SimulationResults, MonteCarloResults]
    metrics: ValidationMetrics
    execution_time: float
    config: SimulationConfig


class StrategyBacktester:
    """Engine for backtesting insurance strategies."""

    def __init__(
        self,
        simulation_engine: Optional[Simulation] = None,
        metric_calculator: Optional[MetricCalculator] = None,
    ):
        """Initialize backtester.

        Args:
            simulation_engine: Engine for running simulations
            metric_calculator: Calculator for performance metrics
        """
        self.simulation_engine = simulation_engine
        self.metric_calculator = metric_calculator or MetricCalculator()
        self.results_cache: Dict[str, BacktestResult] = {}

    def test_strategy(
        self,
        strategy: InsuranceStrategy,
        manufacturer: WidgetManufacturer,
        config: SimulationConfig,
        use_cache: bool = True,
    ) -> BacktestResult:
        """Test a single strategy.

        Args:
            strategy: Strategy to test
            manufacturer: Manufacturer instance
            config: Simulation configuration
            use_cache: Whether to use cached results

        Returns:
            BacktestResult with performance metrics.
        """
        # Check cache
        cache_key = f"{strategy.name}_{hashlib.sha256(str(config).encode()).hexdigest()[:16]}"
        if use_cache and cache_key in self.results_cache:
            logger.info(f"Using cached results for {strategy.name}")
            return self.results_cache[cache_key]

        logger.info(f"Testing strategy: {strategy.name}")

        # Handle OptimizedStaticStrategy
        if isinstance(strategy, OptimizedStaticStrategy) and not strategy.optimized_params:
            logger.info(f"Running optimization for {strategy.name}")
            # Ensure we pass a valid simulation engine
            if self.simulation_engine is not None:
                strategy.optimize_limits(manufacturer, self.simulation_engine)
            else:
                logger.warning("No simulation engine available for optimization")

        # Get insurance program
        insurance_program = strategy.get_insurance_program(manufacturer)

        # Run simulation
        import time

        start_time = time.time()

        # Create loss generator
        loss_generator = ManufacturingLossGenerator(seed=config.seed)

        # Create a default insurance program if None is returned
        if insurance_program is None:
            from .insurance_program import InsuranceProgram

            insurance_program = InsuranceProgram(layers=[])  # No insurance

        # Initialize Monte Carlo engine with required parameters
        monte_carlo = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )
        simulation_results = monte_carlo.run()

        execution_time = time.time() - start_time

        # Calculate metrics - handle Monte Carlo results
        metrics = self._calculate_metrics_mc(simulation_results, config.n_years)

        # Create result
        result = BacktestResult(
            strategy_name=strategy.name,
            simulation_results=simulation_results,
            metrics=metrics,
            execution_time=execution_time,
            config=config,
        )

        # Cache result
        if use_cache:
            self.results_cache[cache_key] = result

        return result

    def test_multiple_strategies(
        self,
        strategies: List[InsuranceStrategy],
        manufacturer: WidgetManufacturer,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """Test multiple strategies and compare.

        Args:
            strategies: List of strategies to test
            manufacturer: Manufacturer instance
            config: Simulation configuration

        Returns:
            DataFrame comparing strategy performance.
        """
        results = []

        for strategy in strategies:
            result = self.test_strategy(strategy, manufacturer, config)

            # Create summary row
            row = {
                "strategy": strategy.name,
                "roe": result.metrics.roe,
                "ruin_probability": result.metrics.ruin_probability,
                "growth_rate": result.metrics.growth_rate,
                "volatility": result.metrics.volatility,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "execution_time": result.execution_time,
            }
            results.append(row)

        return pd.DataFrame(results)

    def _calculate_metrics_mc(
        self, simulation_results: MonteCarloResults, n_years: int
    ) -> ValidationMetrics:
        """Calculate metrics from Monte Carlo simulation results.

        Args:
            simulation_results: Monte Carlo simulation results
            n_years: Number of years simulated

        Returns:
            ValidationMetrics object.
        """
        # Extract returns and final assets
        returns = simulation_results.growth_rates
        final_assets = simulation_results.final_assets

        # Calculate metrics
        metrics = self.metric_calculator.calculate_metrics(
            returns=returns,
            final_assets=final_assets,
            initial_assets=10000000,  # Default initial assets
            n_years=n_years,
        )

        # Add ruin probability from MC results
        # Extract ruin probability for the final year from the dict
        metrics.ruin_probability = simulation_results.ruin_probability.get(
            str(n_years),
            (
                list(simulation_results.ruin_probability.values())[-1]
                if simulation_results.ruin_probability
                else 0.0
            ),
        )

        return metrics

    def _calculate_metrics(
        self, simulation_results: SimulationResults, n_years: int
    ) -> ValidationMetrics:
        """Calculate metrics from simulation results.

        Args:
            simulation_results: Raw simulation results
            n_years: Number of years simulated

        Returns:
            ValidationMetrics object.
        """
        # Calculate growth rates from ROE
        returns = simulation_results.roe

        # Get final assets from the last asset value
        final_assets = np.array([simulation_results.assets[-1]])

        # Calculate metrics
        metrics = self.metric_calculator.calculate_metrics(
            returns=returns,
            final_assets=final_assets,
            initial_assets=10000000,  # Default initial assets
            n_years=n_years,
        )

        # Calculate ruin probability from insolvency
        metrics.ruin_probability = 1.0 if simulation_results.insolvency_year is not None else 0.0

        return metrics

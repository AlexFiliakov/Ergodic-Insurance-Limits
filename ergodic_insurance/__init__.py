"""Ergodic Insurance Limits - Core Package.

This module provides the main entry point for the Ergodic Insurance Limits
package, exposing the key classes and functions for insurance simulation
and analysis using ergodic theory. The framework helps optimize insurance
retentions and limits for businesses by analyzing time-average outcomes
rather than traditional ensemble approaches.

Key Features:
    - Ergodic analysis of insurance decisions
    - Business optimization with insurance constraints
    - Monte Carlo simulation with trajectory storage
    - Insurance strategy backtesting and validation
    - Performance optimization and benchmarking
    - Comprehensive visualization and reporting

Examples:
    Basic simulation::

        from ergodic_insurance import Simulation, Config

        config = Config()
        sim = Simulation(config)
        results = sim.run(years=50)

    Business optimization::

        from ergodic_insurance import BusinessOptimizer, BusinessObjective

        optimizer = BusinessOptimizer()
        result = optimizer.optimize(
            objective=BusinessObjective.MAXIMIZE_GROWTH,
            constraints=BusinessConstraints(min_survival_prob=0.95)
        )

Note:
    This module uses lazy imports to avoid circular dependencies during
    test discovery. All public API classes are accessible through the
    module's __all__ list.

Since:
    Version 0.4.0
"""
# pylint: disable=undefined-all-variable

__version__ = "0.4.0"
__all__ = [
    "__version__",
    "BusinessObjective",
    "BusinessConstraints",
    "OptimalStrategy",
    "BusinessOptimizationResult",
    "BusinessOptimizer",
    "LossEvent",
    "LossData",
    "ManufacturingLossGenerator",
    "Config",
    "ManufacturerConfig",
    "ErgodicAnalyzer",
    "WidgetManufacturer",
    "Simulation",
    "SimulationResults",
    "ValidationMetrics",
    "MetricCalculator",
    "PerformanceTargets",
    "InsuranceStrategy",
    "NoInsuranceStrategy",
    "ConservativeFixedStrategy",
    "AggressiveFixedStrategy",
    "OptimizedStaticStrategy",
    "AdaptiveStrategy",
    "StrategyBacktester",
    "WalkForwardValidator",
    "PerformanceOptimizer",
    "OptimizationConfig",
    "ProfileResult",
    "SmartCache",
    "VectorizedOperations",
    "AccuracyValidator",
    "ValidationResult",
    "ReferenceImplementations",
    "EdgeCaseTester",
    "BenchmarkSuite",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "SystemProfiler",
    "StyleManager",
    "Theme",
    "FigureFactory",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "TwoWaySensitivityResult",
    # Ledger for event sourcing
    "Ledger",
    "LedgerEntry",
    "TransactionType",
    "EntryType",
    "AccountType",
    "AccountName",
]


def __getattr__(name):
    """Lazy import modules to avoid circular dependencies during test discovery.

    This function implements PEP 562 for lazy loading of submodules, which helps
    reduce import time and avoid circular dependencies during test discovery.

    Args:
        name: The name of the attribute to retrieve.

    Returns:
        The requested module, class, or function.

    Raises:
        AttributeError: If the requested attribute does not exist in the module.

    Note:
        This function is called automatically by Python when accessing module
        attributes that are not yet loaded. It should not be called directly.
    """
    if name in (
        "BusinessObjective",
        "BusinessConstraints",
        "OptimalStrategy",
        "BusinessOptimizationResult",
        "BusinessOptimizer",
    ):
        from .business_optimizer import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            BusinessConstraints,
            BusinessObjective,
            BusinessOptimizationResult,
            BusinessOptimizer,
            OptimalStrategy,
        )

        return locals()[name]
    if name in ("LossEvent", "LossData", "ManufacturingLossGenerator"):
        from .loss_distributions import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            LossData,
            LossEvent,
            ManufacturingLossGenerator,
        )

        return locals()[name]
    if name in ("Config", "ManufacturerConfig"):
        from .config import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            Config,
            ManufacturerConfig,
        )

        return locals()[name]
    if name == "ErgodicAnalyzer":
        from .ergodic_analyzer import ErgodicAnalyzer  # pylint: disable=import-outside-toplevel

        return ErgodicAnalyzer
    if name == "WidgetManufacturer":
        from .manufacturer import WidgetManufacturer  # pylint: disable=import-outside-toplevel

        return WidgetManufacturer
    if name in ("Simulation", "SimulationResults"):
        from .simulation import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            Simulation,
            SimulationResults,
        )

        return locals()[name]
    if name in ("ValidationMetrics", "MetricCalculator", "PerformanceTargets"):
        from .validation_metrics import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            MetricCalculator,
            PerformanceTargets,
            ValidationMetrics,
        )

        return locals()[name]
    if name in (
        "InsuranceStrategy",
        "NoInsuranceStrategy",
        "ConservativeFixedStrategy",
        "AggressiveFixedStrategy",
        "OptimizedStaticStrategy",
        "AdaptiveStrategy",
        "StrategyBacktester",
    ):
        from .strategy_backtester import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            AdaptiveStrategy,
            AggressiveFixedStrategy,
            ConservativeFixedStrategy,
            InsuranceStrategy,
            NoInsuranceStrategy,
            OptimizedStaticStrategy,
            StrategyBacktester,
        )

        return locals()[name]
    if name == "WalkForwardValidator":
        from .walk_forward_validator import (  # pylint: disable=import-outside-toplevel
            WalkForwardValidator,
        )

        return WalkForwardValidator
    if name in (
        "PerformanceOptimizer",
        "OptimizationConfig",
        "ProfileResult",
        "SmartCache",
        "VectorizedOperations",
    ):
        from .performance_optimizer import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            OptimizationConfig,
            PerformanceOptimizer,
            ProfileResult,
            SmartCache,
            VectorizedOperations,
        )

        return locals()[name]
    if name in (
        "AccuracyValidator",
        "ValidationResult",
        "ReferenceImplementations",
        "EdgeCaseTester",
    ):
        from .accuracy_validator import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            AccuracyValidator,
            EdgeCaseTester,
            ReferenceImplementations,
            ValidationResult,
        )

        return locals()[name]
    if name in (
        "BenchmarkSuite",
        "BenchmarkConfig",
        "BenchmarkResult",
        "BenchmarkMetrics",
        "SystemProfiler",
    ):
        from .benchmarking import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            BenchmarkConfig,
            BenchmarkMetrics,
            BenchmarkResult,
            BenchmarkSuite,
            SystemProfiler,
        )

        return locals()[name]
    if name in ("StyleManager", "Theme", "FigureFactory"):
        from .visualization import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            FigureFactory,
            StyleManager,
            Theme,
        )

        return locals()[name]
    if name in ("SensitivityAnalyzer", "SensitivityResult", "TwoWaySensitivityResult"):
        from .sensitivity import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            SensitivityAnalyzer,
            SensitivityResult,
            TwoWaySensitivityResult,
        )

        return locals()[name]
    if name in (
        "Ledger",
        "LedgerEntry",
        "TransactionType",
        "EntryType",
        "AccountType",
        "AccountName",
    ):
        from .ledger import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            AccountName,
            AccountType,
            EntryType,
            Ledger,
            LedgerEntry,
            TransactionType,
        )

        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

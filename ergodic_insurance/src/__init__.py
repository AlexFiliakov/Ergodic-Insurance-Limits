"""Ergodic Insurance Limits - Core Package.

This module provides the main entry point for the Ergodic Insurance Limits
package, exposing the key classes and functions for insurance simulation
and analysis.
"""
# pylint: disable=undefined-all-variable

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "BusinessObjective",
    "BusinessConstraints",
    "OptimalStrategy",
    "BusinessOptimizationResult",
    "BusinessOptimizer",
    "ClaimEvent",
    "ClaimGenerator",
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
]


def __getattr__(name):
    """Lazy import modules to avoid circular dependencies during test discovery."""
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
    if name in ("ClaimEvent", "ClaimGenerator"):
        from .claim_generator import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            ClaimEvent,
            ClaimGenerator,
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

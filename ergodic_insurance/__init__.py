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

Top-level Exports:
    The top-level ``__all__`` exposes the essential classes for most workflows:

    - ``run_analysis`` / ``AnalysisResults`` — one-call analysis entry point
    - ``Config`` / ``ManufacturerConfig`` — configuration
    - ``InsuranceProgram`` / ``EnhancedInsuranceLayer`` — insurance modeling
    - ``Simulation`` / ``SimulationResults`` — running simulations

    All other classes remain importable from their respective submodules
    (see *Import Recipes* below) and via ``from ergodic_insurance import <name>``
    for backward compatibility.

Examples:
    One-call analysis (recommended starting point)::

        from ergodic_insurance import run_analysis

        results = run_analysis(
            initial_assets=10_000_000,
            loss_frequency=2.5,
            loss_severity_mean=1_000_000,
            deductible=500_000,
            coverage_limit=10_000_000,
            premium_rate=0.025,
        )
        print(results.summary())
        results.plot()

    Quick start with defaults (creates a $10M manufacturer, 50-year horizon)::

        from ergodic_insurance import Config

        config = Config()  # All defaults — just works

    From basic company info::

        from ergodic_insurance import Config

        config = Config.from_company(
            initial_assets=50_000_000,
            operating_margin=0.12,
        )

Import Recipes:
    Loss modeling::

        from ergodic_insurance.loss_distributions import (
            LossEvent, LossData, ManufacturingLossGenerator,
        )

    Business simulation::

        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.monte_carlo import MonteCarloEngine, MonteCarloResults

    Ergodic & risk analysis::

        from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
        from ergodic_insurance.risk_metrics import RiskMetrics
        from ergodic_insurance.ruin_probability import RuinProbabilityAnalyzer

    Insurance pricing::

        from ergodic_insurance.insurance_pricing import InsurancePricer, MarketCycle

    Business optimization::

        from ergodic_insurance.business_optimizer import (
            BusinessOptimizer, BusinessObjective, BusinessConstraints,
            OptimalStrategy, BusinessOptimizationResult,
        )

    Strategies & backtesting::

        from ergodic_insurance.strategy_backtester import (
            InsuranceStrategy, NoInsuranceStrategy, ConservativeFixedStrategy,
            AggressiveFixedStrategy, OptimizedStaticStrategy, AdaptiveStrategy,
            StrategyBacktester,
        )
        from ergodic_insurance.walk_forward_validator import WalkForwardValidator

    Sensitivity analysis::

        from ergodic_insurance.sensitivity import (
            SensitivityAnalyzer, SensitivityResult, TwoWaySensitivityResult,
        )

    Visualization::

        from ergodic_insurance.visualization import StyleManager, Theme, FigureFactory

    Validation & performance::

        from ergodic_insurance.validation_metrics import (
            ValidationMetrics, MetricCalculator, PerformanceTargets,
        )
        from ergodic_insurance.accuracy_validator import AccuracyValidator, ValidationResult
        from ergodic_insurance.performance_optimizer import (
            PerformanceOptimizer, OptimizationConfig,
        )

    Ledger (event sourcing)::

        from ergodic_insurance.ledger import (
            Ledger, LedgerEntry, TransactionType, EntryType, AccountType, AccountName,
        )

Note:
    This module uses lazy imports to avoid circular dependencies during
    test discovery. All classes listed in the *Import Recipes* above are
    also accessible as ``from ergodic_insurance import <name>`` for
    backward compatibility, but they are not included in ``__all__``
    and will not appear in IDE auto-complete at the top level.

Since:
    Version 0.4.0
"""

# pylint: disable=undefined-all-variable

try:
    from ergodic_insurance._version import __version__
except ImportError:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("ergodic-insurance")

__all__ = [
    "__version__",
    # Quick-start factory
    "run_analysis",
    "AnalysisResults",
    # Configuration
    "Config",
    "ManufacturerConfig",
    # Insurance modeling
    "InsuranceProgram",
    "EnhancedInsuranceLayer",
    # Simulation
    "Simulation",
    "SimulationResults",
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
    if name in ("run_analysis", "AnalysisResults"):
        from ._run_analysis import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            AnalysisResults,
            run_analysis,
        )

        return locals()[name]
    if name in ("InsurancePolicy", "InsuranceLayer"):
        from .insurance import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            InsuranceLayer,
            InsurancePolicy,
        )

        return locals()[name]
    if name in ("InsuranceProgram", "EnhancedInsuranceLayer"):
        from .insurance_program import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            EnhancedInsuranceLayer,
            InsuranceProgram,
        )

        return locals()[name]
    if name in ("InsurancePricer", "MarketCycle"):
        from .insurance_pricing import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            InsurancePricer,
            MarketCycle,
        )

        return locals()[name]
    if name == "RiskMetrics":
        from .risk_metrics import RiskMetrics  # pylint: disable=import-outside-toplevel

        return RiskMetrics
    if name in ("MonteCarloEngine", "MonteCarloResults"):
        from .monte_carlo import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            MonteCarloEngine,
            MonteCarloResults,
        )

        return locals()[name]
    if name == "RuinProbabilityAnalyzer":
        from .ruin_probability import (  # pylint: disable=import-outside-toplevel
            RuinProbabilityAnalyzer,
        )

        return RuinProbabilityAnalyzer
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
    if name in ("PerformanceOptimizer", "OptimizationConfig"):
        from .performance_optimizer import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            OptimizationConfig,
            PerformanceOptimizer,
        )

        return locals()[name]
    if name in ("AccuracyValidator", "ValidationResult"):
        from .accuracy_validator import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            AccuracyValidator,
            ValidationResult,
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
    if name in (
        "ErgodicInsuranceWarning",
        "ConfigurationWarning",
        "DataQualityWarning",
        "ExportWarning",
    ):
        from ._warnings import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            ConfigurationWarning,
            DataQualityWarning,
            ErgodicInsuranceWarning,
            ExportWarning,
        )

        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

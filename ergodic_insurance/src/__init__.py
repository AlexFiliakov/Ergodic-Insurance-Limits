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
    "BusinessOutcomeOptimizer",
    "ClaimEvent",
    "ClaimGenerator",
    "Config",
    "ManufacturerConfig",
    "ErgodicAnalyzer",
    "WidgetManufacturer",
    "Simulation",
    "SimulationResults",
]


def __getattr__(name):
    """Lazy import modules to avoid circular dependencies during test discovery."""
    if name in (
        "BusinessObjective",
        "BusinessConstraints",
        "OptimalStrategy",
        "BusinessOptimizationResult",
        "BusinessOutcomeOptimizer",
    ):
        from .business_optimizer import (  # pylint: disable=import-outside-toplevel,possibly-unused-variable
            BusinessConstraints,
            BusinessObjective,
            BusinessOptimizationResult,
            BusinessOutcomeOptimizer,
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
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

"""Ergodic Insurance Limits"""

from ._version import __version__

# Use lazy imports to avoid import issues during test discovery
# Direct imports are defined but modules are imported only when accessed

__all__ = [
    "__version__",
    "ClaimEvent",
    "ClaimGenerator",
    "Config",
    "ConfigLoader",
    "DecisionMetrics",
    "EnhancedInsuranceLayer",
    "ErgodicAnalyzer",
    "InsuranceDecision",
    "InsuranceDecisionEngine",
    "InsuranceLayer",
    "InsurancePolicy",
    "InsuranceProgram",
    "LossDistribution",
    "ManufacturerConfig",
    "ManufacturingLossGenerator",
    "MonteCarloEngine",
    "OptimizationConstraints",
    "RiskMetrics",
    "Simulation",
    "SimulationConfig",
    "SimulationResults",
    "WSJ_COLORS",
    "WidgetManufacturer",
    "format_currency",
]


def __getattr__(name):
    """Lazy import modules to avoid circular dependencies during test discovery."""
    if name == "ClaimEvent" or name == "ClaimGenerator":
        from .src.claim_generator import ClaimEvent, ClaimGenerator

        return locals()[name]
    elif name == "Config" or name == "ManufacturerConfig":
        from .src.config import Config, ManufacturerConfig

        return locals()[name]
    elif name == "ConfigLoader":
        from .src.config_loader import ConfigLoader

        return ConfigLoader
    elif name in [
        "DecisionMetrics",
        "InsuranceDecision",
        "InsuranceDecisionEngine",
        "OptimizationConstraints",
    ]:
        from .src.decision_engine import (
            DecisionMetrics,
            InsuranceDecision,
            InsuranceDecisionEngine,
            OptimizationConstraints,
        )

        return locals()[name]
    elif name == "ErgodicAnalyzer":
        from .src.ergodic_analyzer import ErgodicAnalyzer

        return ErgodicAnalyzer
    elif name == "InsuranceLayer" or name == "InsurancePolicy":
        from .src.insurance import InsuranceLayer, InsurancePolicy

        return locals()[name]
    elif name == "EnhancedInsuranceLayer" or name == "InsuranceProgram":
        from .src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram

        return locals()[name]
    elif name == "LossDistribution" or name == "ManufacturingLossGenerator":
        from .src.loss_distributions import LossDistribution, ManufacturingLossGenerator

        return locals()[name]
    elif name == "WidgetManufacturer":
        from .src.manufacturer import WidgetManufacturer

        return WidgetManufacturer
    elif name == "MonteCarloEngine" or name == "SimulationConfig":
        from .src.monte_carlo import MonteCarloEngine, SimulationConfig

        return locals()[name]
    elif name == "RiskMetrics":
        from .src.risk_metrics import RiskMetrics

        return RiskMetrics
    elif name == "Simulation" or name == "SimulationResults":
        from .src.simulation import Simulation, SimulationResults

        return locals()[name]
    elif name == "WSJ_COLORS" or name == "format_currency":
        from .src.visualization import WSJ_COLORS, format_currency

        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

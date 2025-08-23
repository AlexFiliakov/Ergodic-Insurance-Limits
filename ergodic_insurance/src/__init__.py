"""Ergodic Insurance Limits - Core Package.

This module provides the main entry point for the Ergodic Insurance Limits
package, exposing the key classes and functions for insurance simulation
and analysis.
"""

from .claim_generator import ClaimEvent, ClaimGenerator
from .config import Config, ManufacturerConfig
from .manufacturer import WidgetManufacturer
from .simulation import Simulation, SimulationResults

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "ClaimEvent",
    "ClaimGenerator",
    "Config",
    "ManufacturerConfig",
    "WidgetManufacturer",
    "Simulation",
    "SimulationResults",
]

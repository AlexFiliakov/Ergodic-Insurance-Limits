"""Ergodic Insurance package - main module imports."""

# Import only the essential classes to avoid circular dependencies
from claim_generator import ClaimEvent, ClaimGenerator
from config_v2 import ManufacturerConfig
from manufacturer import WidgetManufacturer

__all__ = ["WidgetManufacturer", "ClaimGenerator", "ClaimEvent", "ManufacturerConfig"]

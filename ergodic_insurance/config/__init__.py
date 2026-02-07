"""Configuration management using Pydantic v2 models.

This package provides comprehensive configuration classes for the Ergodic
Insurance simulation framework. It uses Pydantic models for validation,
type safety, and automatic serialization/deserialization of configuration
parameters.

The configuration system is hierarchical, with specialized configs for
different aspects of the simulation (manufacturer, insurance, simulation
parameters, etc.) that can be composed into a master configuration.

Sub-modules:
    constants: Module-level financial constants (e.g., DEFAULT_RISK_FREE_RATE).
    core: Master Config and ConfigV2 classes that compose all sub-configs.
    insurance: Insurance layer, program, and loss distribution configs.
    manufacturer: Business entity, expense ratio, and industry profile configs.
    market: Pricing scenarios, transition probabilities, and market cycles.
    optimizer: BusinessOptimizer and DecisionEngine calibration parameters.
    presets: Profile metadata, modules, presets, and preset libraries.
    reporting: Output, logging, and Excel report generation configs.
    simulation: Simulation execution, growth, debt, and working capital configs.

Key Features:
    - Type-safe configuration with automatic validation
    - Hierarchical configuration structure
    - Environment variable support
    - JSON/YAML serialization support
    - Default values with business logic constraints
    - Cross-field validation for consistency

Examples:
    Quick start with defaults::

        from ergodic_insurance import Config

        # All defaults — $10M manufacturer, 50-year horizon
        config = Config()

    From basic company info::

        config = Config.from_company(
            initial_assets=50_000_000,
            operating_margin=0.12,
            industry="manufacturing",
        )

    Full control::

        from ergodic_insurance import Config, ManufacturerConfig

        config = Config(
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.7,
            )
        )

    Loading from file::

        config = Config.from_yaml(Path('config.yaml'))

Note:
    All monetary values are in nominal dollars unless otherwise specified.
    Rates and ratios are expressed as decimals (0.1 = 10%).

Since:
    Version 0.1.0 (monolithic), refactored in 0.9.0 (Issue #458)
"""

from .constants import DEFAULT_RISK_FREE_RATE
from .core import Config, ConfigV2
from .insurance import InsuranceConfig, InsuranceLayerConfig, LossDistributionConfig
from .manufacturer import (
    DepreciationConfig,
    ExpenseRatioConfig,
    IndustryConfig,
    ManufacturerConfig,
    ManufacturingConfig,
    RetailConfig,
    ServiceConfig,
)
from .market import (
    MarketCycles,
    PricingScenario,
    PricingScenarioConfig,
    TransitionProbabilities,
)
from .optimizer import BusinessOptimizerConfig, DecisionEngineConfig
from .presets import ModuleConfig, PresetConfig, PresetLibrary, ProfileMetadata
from .reporting import ExcelReportConfig, LoggingConfig, OutputConfig
from .simulation import (
    DebtConfig,
    GrowthConfig,
    SimulationConfig,
    WorkingCapitalConfig,
    WorkingCapitalRatiosConfig,
)

__all__ = [
    # Constants
    "DEFAULT_RISK_FREE_RATE",
    # Core
    "Config",
    "ConfigV2",
    # Insurance
    "InsuranceConfig",
    "InsuranceLayerConfig",
    "LossDistributionConfig",
    # Manufacturer & Industry
    "DepreciationConfig",
    "ExpenseRatioConfig",
    "IndustryConfig",
    "ManufacturerConfig",
    "ManufacturingConfig",
    "RetailConfig",
    "ServiceConfig",
    # Market
    "MarketCycles",
    "PricingScenario",
    "PricingScenarioConfig",
    "TransitionProbabilities",
    # Optimizer
    "BusinessOptimizerConfig",
    "DecisionEngineConfig",
    # Presets
    "ModuleConfig",
    "PresetConfig",
    "PresetLibrary",
    "ProfileMetadata",
    # Reporting
    "ExcelReportConfig",
    "LoggingConfig",
    "OutputConfig",
    # Simulation
    "DebtConfig",
    "GrowthConfig",
    "SimulationConfig",
    "WorkingCapitalConfig",
    "WorkingCapitalRatiosConfig",
]

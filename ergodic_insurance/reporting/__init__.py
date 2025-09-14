"""Reporting and caching infrastructure with automated report generation.

This module provides comprehensive report generation capabilities along with
high-performance caching for Monte Carlo simulations and analysis results.

Key Features:
    - Automated report generation (Executive & Technical)
    - Configurable report structures
    - Table generation utilities
    - Multi-format output (Markdown, HTML, PDF)
    - Report validation and quality checks
    - HDF5 storage for large simulation data
    - Parquet support for structured results
    - Hash-based cache invalidation
    - Memory-mapped reading for efficiency
    - Configurable storage backends

Example:
    >>> from ergodic_insurance.src.reporting import ExecutiveReport, CacheManager
    >>> # Generate executive report
    >>> report = ExecutiveReport(results={'roe': 0.18, 'ruin_probability': 0.01})
    >>> report_path = report.generate()
    >>>
    >>> # Use caching for simulations
    >>> cache = CacheManager(cache_dir="./cache")
    >>> cache.cache_simulation_paths(params, paths, metadata={'n_sims': 10000})
"""

from .cache_manager import CacheConfig, CacheKey, CacheManager, CacheStats, StorageBackend
from .config import (
    FigureConfig,
    ReportConfig,
    ReportMetadata,
    ReportStyle,
    SectionConfig,
    TableConfig,
    create_executive_config,
    create_technical_config,
)
from .executive_report import ExecutiveReport
from .formatters import ColorCoder, NumberFormatter, TableFormatter, format_for_export
from .report_builder import ReportBuilder
from .table_generator import (
    TableGenerator,
    create_parameter_table,
    create_performance_table,
    create_sensitivity_table,
)
from .technical_report import TechnicalReport
from .validator import ReportValidator, validate_parameters, validate_results_data

__all__ = [
    # Cache management
    "CacheManager",
    "CacheConfig",
    "CacheStats",
    "StorageBackend",
    "CacheKey",
    # Configuration
    "ReportConfig",
    "ReportMetadata",
    "ReportStyle",
    "SectionConfig",
    "FigureConfig",
    "TableConfig",
    "create_executive_config",
    "create_technical_config",
    # Formatters
    "NumberFormatter",
    "ColorCoder",
    "TableFormatter",
    "format_for_export",
    # Table generation
    "TableGenerator",
    "create_performance_table",
    "create_parameter_table",
    "create_sensitivity_table",
    # Report builders
    "ReportBuilder",
    "ExecutiveReport",
    "TechnicalReport",
    # Validation
    "ReportValidator",
    "validate_results_data",
    "validate_parameters",
]

__version__ = "1.0.0"

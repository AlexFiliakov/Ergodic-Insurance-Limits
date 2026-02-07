"""Re-exports for internal/infrastructure classes.

These classes were previously exported from the package root but are intended
for advanced use cases such as performance tuning, benchmarking, and low-level
validation.  They are still fully supported — just not part of the default
public API surface.

Usage::

    from ergodic_insurance.internals import SmartCache, BenchmarkSuite

Since:
    Version 0.5.0
"""

from .accuracy_validator import EdgeCaseTester, ReferenceImplementations
from .benchmarking import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSuite,
    SystemProfiler,
)
from .performance_optimizer import ProfileResult, SmartCache, VectorizedOperations

__all__ = [
    # From performance_optimizer
    "ProfileResult",
    "SmartCache",
    "VectorizedOperations",
    # From accuracy_validator
    "ReferenceImplementations",
    "EdgeCaseTester",
    # From benchmarking
    "BenchmarkSuite",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "SystemProfiler",
]

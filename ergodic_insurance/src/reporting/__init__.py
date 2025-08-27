"""Reporting and caching infrastructure for expensive computations.

This module provides high-performance caching for Monte Carlo simulations
and analysis results, enabling fast figure regeneration and report compilation.

Key Features:
    - HDF5 storage for large simulation data
    - Parquet support for structured results
    - Hash-based cache invalidation
    - Memory-mapped reading for efficiency
    - Configurable storage backends

Example:
    >>> from ergodic_insurance.src.reporting import CacheManager
    >>> cache = CacheManager(cache_dir="./cache")
    >>> # Cache simulation results
    >>> cache.cache_simulation_paths(
    ...     params=simulation_params,
    ...     paths=simulation_paths,
    ...     metadata={'n_sims': 10000}
    ... )
    >>> # Load cached results
    >>> paths = cache.load_simulation_paths(params)
"""

from .cache_manager import CacheConfig, CacheKey, CacheManager, CacheStats, StorageBackend

__all__ = [
    "CacheManager",
    "CacheConfig",
    "CacheStats",
    "StorageBackend",
    "CacheKey",
]

__version__ = "1.0.0"

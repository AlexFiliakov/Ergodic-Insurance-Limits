"""Performance optimization module for Monte Carlo simulations.

This module provides tools and strategies to optimize the performance of
Monte Carlo simulations, targeting 100K simulations in under 60 seconds
on budget hardware (4-core CPU, 8GB RAM).

Key features:
    - Execution profiling and bottleneck identification
    - Vectorized operations for loss generation and insurance calculations
    - Smart caching for repeated calculations
    - Memory optimization for large-scale simulations
    - Integration with parallel execution framework

Example:
    >>> from performance_optimizer import PerformanceOptimizer
    >>> from monte_carlo import MonteCarloEngine
    >>>
    >>> optimizer = PerformanceOptimizer()
    >>> engine = MonteCarloEngine(config=config)
    >>>
    >>> # Profile execution
    >>> profile_results = optimizer.profile_execution(engine, n_simulations=1000)
    >>> print(profile_results.bottlenecks)
    >>>
    >>> # Apply optimizations
    >>> optimized_engine = optimizer.optimize_engine(engine)
    >>> results = optimized_engine.run()

Google-style docstrings are used throughout for Sphinx documentation.
"""

import cProfile
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import gc
import io
import pstats
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import psutil

# Try to import numba for JIT compilation, but make it optional
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Define dummy decorators when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


@dataclass
class ProfileResult:
    """Results from performance profiling.

    Attributes:
        total_time: Total execution time in seconds
        bottlenecks: List of performance bottlenecks identified
        function_times: Dictionary mapping function names to execution times
        memory_usage: Peak memory usage in MB
        recommendations: List of optimization recommendations
    """

    total_time: float
    bottlenecks: List[str] = field(default_factory=list)
    function_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary of profiling results.

        Returns:
            Formatted summary string.
        """
        summary = f"Performance Profile Summary\n{'='*50}\n"
        summary += f"Total Time: {self.total_time:.2f}s\n"
        summary += f"Peak Memory: {self.memory_usage:.1f} MB\n"

        if self.bottlenecks:
            summary += "\nTop Bottlenecks:\n"
            for bottleneck in self.bottlenecks[:5]:
                summary += f"  - {bottleneck}\n"

        if self.recommendations:
            summary += "\nRecommendations:\n"
            for rec in self.recommendations:
                summary += f"  - {rec}\n"

        return summary


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization.

    Attributes:
        enable_vectorization: Use vectorized operations
        enable_caching: Use smart caching
        cache_size: Maximum cache entries
        enable_numba: Use Numba JIT compilation
        memory_limit_mb: Memory usage limit in MB
        chunk_size: Chunk size for batch processing
    """

    enable_vectorization: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    enable_numba: bool = True
    memory_limit_mb: float = 4000.0
    chunk_size: int = 10000


class SmartCache:
    """Smart caching system for repeated calculations.

    Provides intelligent caching with memory management and
    hit rate tracking.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize smart cache.

        Args:
            max_size: Maximum number of cache entries.
        """
        self.cache: Dict[Tuple, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.access_counts: Dict[Tuple, int] = defaultdict(int)

    def get(self, key: Tuple) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key (must be hashable).

        Returns:
            Cached value or None if not found.
        """
        if key in self.cache:
            self.hits += 1
            self.access_counts[key] += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: Tuple, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key (must be hashable).
            value: Value to cache.
        """
        if len(self.cache) >= self.max_size:
            # Evict least recently accessed
            least_accessed = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            del self.cache[least_accessed]
            del self.access_counts[least_accessed]

        self.cache[key] = value
        self.access_counts[key] = 1

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as percentage.
        """
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_counts.clear()
        self.hits = 0
        self.misses = 0


class VectorizedOperations:
    """Vectorized operations for performance optimization."""

    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_growth_rates(
        final_assets: np.ndarray, initial_assets: float, n_years: float
    ) -> np.ndarray:
        """Calculate growth rates using vectorized operations.

        Args:
            final_assets: Array of final asset values.
            initial_assets: Initial asset value.
            n_years: Number of years.

        Returns:
            Array of growth rates.
        """
        n = len(final_assets)
        growth_rates = np.zeros(n)

        for i in prange(n):
            if final_assets[i] > 0 and initial_assets > 0:
                growth_rates[i] = np.log(final_assets[i] / initial_assets) / n_years

        return growth_rates

    @staticmethod
    def apply_insurance_vectorized(
        losses: np.ndarray, attachment: float, limit: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply insurance coverage using vectorized operations.

        Args:
            losses: Array of loss amounts.
            attachment: Insurance attachment point.
            limit: Insurance limit.

        Returns:
            Tuple of (retained_losses, recovered_amounts).
        """
        # Vectorized insurance calculation
        excess_losses = np.maximum(losses - attachment, 0)
        recovered = np.minimum(excess_losses, limit)
        retained = losses - recovered

        return retained, recovered

    @staticmethod
    def calculate_premiums_vectorized(limits: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Calculate premiums using vectorized operations.

        Args:
            limits: Array of insurance limits.
            rates: Array of premium rates.

        Returns:
            Array of premium amounts.
        """
        return limits * rates  # type: ignore[no-any-return]


class PerformanceOptimizer:
    """Main performance optimization engine.

    Provides profiling, optimization, and monitoring capabilities
    for Monte Carlo simulations.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize performance optimizer.

        Args:
            config: Optimization configuration.
        """
        self.config = config or OptimizationConfig()
        self.cache = SmartCache(max_size=self.config.cache_size)
        self.vectorized = VectorizedOperations()
        self._profile_data = None

    def profile_execution(  # pylint: disable=too-many-locals
        self, func: Callable, *args, **kwargs
    ) -> ProfileResult:
        """Profile function execution to identify bottlenecks.

        Args:
            func: Function to profile.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            ProfileResult with profiling data.
        """
        # Memory tracking
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # CPU profiling
        profiler = cProfile.Profile()

        start_time = time.time()
        profiler.enable()

        try:
            _result = func(*args, **kwargs)
        finally:
            profiler.disable()

        total_time = time.time() - start_time

        # Memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(20)

        # Parse top functions
        function_times = {}
        bottlenecks = []

        lines = s.getvalue().split("\n")
        for line in lines:
            if "function calls" in line or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    cumtime = float(parts[3])
                    func_name = parts[-1]
                    function_times[func_name] = cumtime

                    # Identify bottlenecks (>10% of total time)
                    if total_time > 0 and cumtime > total_time * 0.1:
                        bottlenecks.append(
                            f"{func_name}: {cumtime:.2f}s ({cumtime/total_time*100:.1f}%)"
                        )
                except (ValueError, IndexError):
                    pass

        # Generate recommendations
        recommendations = self._generate_recommendations(function_times, memory_usage, total_time)

        return ProfileResult(
            total_time=total_time,
            bottlenecks=bottlenecks[:5],
            function_times=function_times,
            memory_usage=memory_usage,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, function_times: Dict[str, float], memory_usage: float, total_time: float
    ) -> List[str]:
        """Generate optimization recommendations based on profiling.

        Args:
            function_times: Dictionary of function execution times.
            memory_usage: Memory usage in MB.
            total_time: Total execution time.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Check for slow functions
        for func_name, time_spent in function_times.items():
            if "generate_losses" in func_name and time_spent > total_time * 0.2:
                recommendations.append("Consider vectorizing loss generation")
            elif "insurance" in func_name and time_spent > total_time * 0.15:
                recommendations.append("Optimize insurance calculations with vectorization")
            elif "loop" in func_name.lower() and time_spent > total_time * 0.1:
                recommendations.append("Replace loops with vectorized operations")

        # Memory recommendations
        if memory_usage > 2000:
            recommendations.append("High memory usage detected - consider chunked processing")
        if memory_usage > 3000:
            recommendations.append("Use float32 instead of float64 for memory efficiency")

        # Caching recommendations
        if self.cache.hit_rate < 50 and self.cache.hits + self.cache.misses > 100:
            recommendations.append(
                f"Low cache hit rate ({self.cache.hit_rate:.1f}%) - review cache strategy"
            )

        # Parallel processing
        if total_time > 10 and not any("parallel" in str(f).lower() for f in function_times):
            recommendations.append("Enable parallel processing for better performance")

        return recommendations

    def optimize_loss_generation(self, losses: List[float], batch_size: int = 10000) -> np.ndarray:
        """Optimize loss generation using vectorization.

        Args:
            losses: List of loss values.
            batch_size: Size of processing batches.

        Returns:
            Optimized loss array.
        """
        if self.config.enable_vectorization:
            # Convert to numpy array for vectorized operations
            loss_array = np.array(
                losses, dtype=np.float32 if self.config.memory_limit_mb < 4000 else np.float64
            )

            # Process in chunks if needed
            if len(loss_array) > batch_size:
                chunks = []
                for i in range(0, len(loss_array), batch_size):
                    chunk = loss_array[i : i + batch_size]
                    chunks.append(chunk)
                return np.concatenate(chunks)

            return loss_array

        return np.array(losses)

    def optimize_insurance_calculation(  # pylint: disable=too-many-locals
        self, losses: np.ndarray, layers: List[Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """Optimize insurance calculations using vectorization and caching.

        Args:
            losses: Array of loss amounts.
            layers: List of (attachment, limit, rate) tuples.

        Returns:
            Dictionary with optimized results.
        """
        # Try cache first
        cache_key = (losses.tobytes(), tuple(layers))
        cached_result = self.cache.get(cache_key) if self.config.enable_caching else None

        if cached_result is not None:
            return cast(Dict[str, Any], cached_result)

        total_premiums = 0.0
        total_recovered = np.zeros_like(losses)
        retained_losses = losses.copy()

        if self.config.enable_vectorization:
            for attachment, limit, rate in layers:
                # Vectorized insurance application
                retained, recovered = self.vectorized.apply_insurance_vectorized(
                    retained_losses, attachment, limit
                )
                total_recovered += recovered
                retained_losses = retained
                total_premiums += limit * rate
        else:
            # Non-vectorized fallback
            for attachment, limit, rate in layers:
                for i, loss in enumerate(retained_losses):
                    if loss > attachment:
                        recovery = min(loss - attachment, limit)
                        total_recovered[i] += recovery
                        retained_losses[i] -= recovery
                total_premiums += limit * rate

        result = {
            "retained_losses": retained_losses,
            "total_recovered": total_recovered,
            "total_premiums": total_premiums,
            "net_losses": retained_losses + total_premiums,
        }

        # Cache result
        if self.config.enable_caching:
            self.cache.set(cache_key, result)

        return result

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage for large simulations.

        Returns:
            Dictionary with memory optimization metrics.
        """
        # Force garbage collection
        gc.collect()

        process = psutil.Process()
        memory_info = process.memory_info()

        # Get system memory
        virtual_memory = psutil.virtual_memory()

        metrics = {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "available_memory_mb": virtual_memory.available / 1024 / 1024,
            "memory_percent": virtual_memory.percent,
            "suggested_chunk_size": self._calculate_optimal_chunk_size(virtual_memory.available),
        }

        # Clear caches if memory is tight
        if metrics["memory_percent"] > 80:
            self.cache.clear()
            gc.collect()
            metrics["cache_cleared"] = True
        else:
            metrics["cache_cleared"] = False

        return metrics

    def _calculate_optimal_chunk_size(self, available_memory: int) -> int:
        """Calculate optimal chunk size based on available memory.

        Args:
            available_memory: Available memory in bytes.

        Returns:
            Optimal chunk size.
        """
        # Assume each simulation needs approximately 1KB
        memory_per_sim = 1024  # bytes

        # Use 50% of available memory for safety
        safe_memory = available_memory * 0.5

        # Calculate chunk size
        chunk_size = int(safe_memory / memory_per_sim)

        # Bounds
        chunk_size = max(1000, min(chunk_size, 100000))

        # Round to nearest 1000
        return (chunk_size // 1000) * 1000

    def get_optimization_summary(self) -> str:
        """Get summary of optimization status.

        Returns:
            Formatted optimization summary.
        """
        memory_metrics = self.optimize_memory_usage()

        summary = f"Performance Optimization Summary\n{'='*50}\n"
        summary += "Configuration:\n"
        summary += (
            f"  Vectorization: {'Enabled' if self.config.enable_vectorization else 'Disabled'}\n"
        )
        summary += f"  Caching: {'Enabled' if self.config.enable_caching else 'Disabled'}\n"
        summary += f"  Numba JIT: {'Enabled' if self.config.enable_numba else 'Disabled'}\n"
        summary += f"  Memory Limit: {self.config.memory_limit_mb:.0f} MB\n"
        summary += "\nCache Performance:\n"
        summary += f"  Hit Rate: {self.cache.hit_rate:.1f}%\n"
        summary += f"  Hits: {self.cache.hits:,}\n"
        summary += f"  Misses: {self.cache.misses:,}\n"
        summary += "\nMemory Usage:\n"
        summary += f"  Process: {memory_metrics['process_memory_mb']:.1f} MB\n"
        summary += f"  Available: {memory_metrics['available_memory_mb']:.1f} MB\n"
        summary += f"  System Usage: {memory_metrics['memory_percent']:.1f}%\n"
        summary += f"  Optimal Chunk Size: {memory_metrics['suggested_chunk_size']:,}\n"

        return summary


def cached_calculation(cache_size: int = 128):
    """Decorator for caching expensive calculations.

    Args:
        cache_size: Maximum cache size.

    Returns:
        Decorated function with caching.
    """

    def decorator(func):
        @wraps(func)
        @lru_cache(maxsize=cache_size)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution.

    Args:
        func: Function to profile.

    Returns:
        Decorated function with profiling.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = PerformanceOptimizer()
        result = optimizer.profile_execution(func, *args, **kwargs)
        print(result.summary())
        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    # Example usage

    # Create optimizer
    optimizer = PerformanceOptimizer()

    # Example: Optimize loss calculations
    losses = np.random.exponential(100000, 100000)
    layers = [(0.0, 1000000.0, 0.015), (1000000.0, 4000000.0, 0.008)]

    start = time.time()
    result = optimizer.optimize_insurance_calculation(losses, layers)
    elapsed = time.time() - start

    print(f"Optimized calculation completed in {elapsed:.3f}s")
    print(f"Cache hit rate: {optimizer.cache.hit_rate:.1f}%")
    print("\n" + optimizer.get_optimization_summary())

"""Comprehensive tests for the performance_optimizer module."""

# mypy: ignore-errors

import cProfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import gc
import io
import pstats
import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

from ergodic_insurance.performance_optimizer import (
    NUMBA_AVAILABLE,
    OptimizationConfig,
    PerformanceOptimizer,
    ProfileResult,
    SmartCache,
    VectorizedOperations,
    cached_calculation,
    profile_function,
)


class TestProfileResult:
    """Test ProfileResult dataclass."""

    def test_initialization_minimal(self):
        """Test minimal ProfileResult initialization."""
        result = ProfileResult(total_time=10.5)
        assert result.total_time == 10.5
        assert result.bottlenecks == []
        assert result.function_times == {}
        assert result.memory_usage == 0.0
        assert result.recommendations == []

    def test_initialization_full(self):
        """Test full ProfileResult initialization."""
        result = ProfileResult(
            total_time=10.5,
            bottlenecks=["function1", "function2"],
            function_times={"func1": 5.0, "func2": 3.0},
            memory_usage=256.0,
            recommendations=["Use vectorization", "Enable caching"],
        )
        assert result.total_time == 10.5
        assert len(result.bottlenecks) == 2
        assert result.function_times["func1"] == 5.0
        assert result.memory_usage == 256.0
        assert len(result.recommendations) == 2

    def test_summary(self):
        """Test summary generation."""
        result = ProfileResult(
            total_time=10.5,
            bottlenecks=["slow_function", "memory_intensive"],
            function_times={"func1": 5.0},
            memory_usage=256.0,
            recommendations=["Enable caching", "Use parallel processing"],
        )
        summary = result.summary()
        assert "Performance Profile Summary" in summary
        assert "Total Time: 10.50s" in summary
        assert "Peak Memory: 256.0 MB" in summary
        assert "slow_function" in summary
        assert "Enable caching" in summary

    def test_summary_many_bottlenecks(self):
        """Test summary with more than 5 bottlenecks."""
        bottlenecks = [f"function{i}" for i in range(10)]
        result = ProfileResult(total_time=10.5, bottlenecks=bottlenecks, memory_usage=256.0)
        summary = result.summary()
        # Should only show first 5 bottlenecks
        assert "function0" in summary
        assert "function4" in summary
        assert "function5" not in summary

    def test_summary_empty(self):
        """Test summary with no bottlenecks or recommendations."""
        result = ProfileResult(total_time=5.0)
        summary = result.summary()
        assert "Performance Profile Summary" in summary
        assert "Total Time: 5.00s" in summary
        assert "Bottlenecks" not in summary
        assert "Recommendations" not in summary


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_initialization(self):
        """Test default OptimizationConfig initialization."""
        config = OptimizationConfig()
        assert config.enable_vectorization is True
        assert config.enable_caching is True
        assert config.cache_size == 1000
        assert config.enable_numba is True
        assert config.memory_limit_mb == 4000.0
        assert config.chunk_size == 10000

    def test_custom_initialization(self):
        """Test custom OptimizationConfig initialization."""
        config = OptimizationConfig(
            enable_vectorization=False,
            enable_caching=False,
            cache_size=500,
            enable_numba=False,
            memory_limit_mb=8000.0,
            chunk_size=5000,
        )
        assert config.enable_vectorization is False
        assert config.enable_caching is False
        assert config.cache_size == 500
        assert config.enable_numba is False
        assert config.memory_limit_mb == 8000.0
        assert config.chunk_size == 5000


class TestSmartCache:
    """Test SmartCache class."""

    def test_initialization(self):
        """Test SmartCache initialization."""
        cache = SmartCache(max_size=100)
        assert cache.max_size == 100
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.cache) == 0
        assert len(cache.access_counts) == 0

    def test_set_and_get(self):
        """Test setting and getting values from cache."""
        cache = SmartCache()
        cache.set(("key1",), "value1")

        value = cache.get(("key1",))
        assert value == "value1"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_get_miss(self):
        """Test cache miss."""
        cache = SmartCache()
        value = cache.get(("nonexistent",))
        assert value is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_eviction(self):
        """Test cache eviction when max size is reached."""
        cache = SmartCache(max_size=3)

        # Fill cache
        cache.set(("key1",), "value1")
        cache.set(("key2",), "value2")
        cache.set(("key3",), "value3")

        # Access key2 to increase its access count
        cache.get(("key2",))
        cache.get(("key2",))

        # Add new key, should evict least accessed (key1 or key3)
        cache.set(("key4",), "value4")

        assert len(cache.cache) == 3
        assert cache.get(("key4",)) == "value4"
        assert cache.get(("key2",)) == "value2"  # Should still be in cache

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = SmartCache()

        cache.set(("key1",), "value1")
        cache.get(("key1",))  # Hit
        cache.get(("key1",))  # Hit
        cache.get(("key2",))  # Miss
        cache.get(("key3",))  # Miss

        assert cache.hits == 2
        assert cache.misses == 2
        assert cache.hit_rate == 50.0

    def test_hit_rate_empty(self):
        """Test hit rate with no accesses."""
        cache = SmartCache()
        assert cache.hit_rate == 0.0

    def test_clear(self):
        """Test clearing the cache."""
        cache = SmartCache()

        cache.set(("key1",), "value1")
        cache.set(("key2",), "value2")
        cache.get(("key1",))

        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_counts) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_access_count_tracking(self):
        """Test that access counts are tracked correctly."""
        cache = SmartCache()

        cache.set(("key1",), "value1")
        assert cache.access_counts[("key1",)] == 1

        cache.get(("key1",))
        assert cache.access_counts[("key1",)] == 2

        cache.get(("key1",))
        assert cache.access_counts[("key1",)] == 3

    def test_heap_evicts_least_accessed(self):
        """Test that heap-based eviction removes the least-accessed entry."""
        cache = SmartCache(max_size=3)

        cache.set(("a",), 1)
        cache.set(("b",), 2)
        cache.set(("c",), 3)

        # Access "b" twice and "c" once — "a" has lowest count (1)
        cache.get(("b",))
        cache.get(("b",))
        cache.get(("c",))

        # Inserting a 4th key should evict "a" (access_count=1)
        cache.set(("d",), 4)

        assert ("a",) not in cache.cache
        assert cache.get(("b",)) == 2
        assert cache.get(("c",)) == 3
        assert cache.get(("d",)) == 4

    def test_heap_multiple_evictions(self):
        """Test multiple sequential evictions via the heap."""
        cache = SmartCache(max_size=2)

        cache.set(("a",), 1)
        cache.set(("b",), 2)

        # Evict to make room — "a" and "b" both have count=1,
        # "a" was inserted first so its heap entry is popped first.
        cache.set(("c",), 3)
        assert len(cache.cache) == 2
        assert ("a",) not in cache.cache

        cache.set(("d",), 4)
        assert len(cache.cache) == 2
        assert ("b",) not in cache.cache

    def test_set_existing_key_does_not_evict(self):
        """Test that updating an existing key does not trigger eviction."""
        cache = SmartCache(max_size=2)

        cache.set(("a",), 1)
        cache.set(("b",), 2)

        # Overwrite "a" — should NOT evict anything
        cache.set(("a",), 10)
        assert len(cache.cache) == 2
        assert cache.get(("a",)) == 10
        assert cache.get(("b",)) == 2

    def test_clear_resets_heap(self):
        """Test that clear() resets internal heap state."""
        cache = SmartCache(max_size=3)

        cache.set(("a",), 1)
        cache.set(("b",), 2)
        cache.get(("a",))

        cache.clear()

        assert len(cache._heap) == 0
        assert cache._counter == 0

    def test_eviction_with_many_stale_entries(self):
        """Test eviction when heap contains many stale entries."""
        cache = SmartCache(max_size=2)

        cache.set(("a",), 1)
        cache.set(("b",), 2)

        # Generate many stale heap entries by repeated gets
        for _ in range(50):
            cache.get(("a",))
            cache.get(("b",))

        # Both keys now have high access counts; their old heap entries are stale.
        # Inserting a new key should still evict correctly.
        cache.set(("c",), 3)
        assert len(cache.cache) == 2
        assert ("c",) in cache.cache


class TestVectorizedOperations:
    """Test VectorizedOperations class."""

    def test_calculate_growth_rates(self):
        """Test vectorized growth rate calculation."""
        final_assets = np.array([20000000, 15000000, 25000000])
        initial_assets = 10000000
        n_years = 10

        rates = VectorizedOperations.calculate_growth_rates(final_assets, initial_assets, n_years)

        assert len(rates) == 3
        assert np.isclose(rates[0], np.log(2.0) / 10)

    def test_calculate_growth_rates_with_zeros(self):
        """Test growth rate calculation with zero values."""
        final_assets = np.array([20000000, 0, 25000000])
        initial_assets = 10000000
        n_years = 10

        rates = VectorizedOperations.calculate_growth_rates(final_assets, initial_assets, n_years)

        assert len(rates) == 3
        assert rates[1] == 0.0  # Zero values result in 0 growth rate

    def test_apply_insurance_vectorized(self):
        """Test vectorized insurance application."""
        losses = np.array([50000, 150000, 300000, 1000000])
        attachment = 100000
        limit = 500000

        retained, recovered = VectorizedOperations.apply_insurance_vectorized(
            losses, attachment, limit
        )

        assert len(retained) == 4
        assert len(recovered) == 4

        # First loss below attachment
        assert retained[0] == 50000
        assert recovered[0] == 0

        # Second loss within limit
        assert retained[1] == 100000
        assert recovered[1] == 50000

        # Fourth loss exceeds limit (max recovery is limited to 500k)
        assert retained[3] == 500000  # 1000000 - min(1000000-100000, 500000)
        assert recovered[3] == 500000  # Limited to 500000

    def test_calculate_premiums_vectorized(self):
        """Test vectorized premium calculation."""
        limits = np.array([100000, 500000, 1000000])
        rates = np.array([0.01, 0.02, 0.015])

        premiums = VectorizedOperations.calculate_premiums_vectorized(limits, rates)

        assert len(premiums) == 3
        assert premiums[0] == 1000
        assert premiums[1] == 10000
        assert premiums[2] == 15000

    def test_apply_insurance_with_limit_boundary(self):
        """Test insurance application at limit boundary."""
        losses = np.array([600000])
        attachment = 100000
        limit = 500000

        retained, recovered = VectorizedOperations.apply_insurance_vectorized(
            losses, attachment, limit
        )

        assert retained[0] == 100000  # attachment + (loss - attachment - limit)
        assert recovered[0] == 500000  # exactly at limit

    def test_calculate_growth_rates_array(self):
        """Test growth rates with various scenarios."""
        final_assets = np.array([5000000, 10000000, 20000000])
        initial_assets = 10000000
        n_years = 10

        rates = VectorizedOperations.calculate_growth_rates(final_assets, initial_assets, n_years)

        assert len(rates) == 3
        assert rates[0] < 0  # Negative growth
        assert rates[1] == 0  # No growth
        assert rates[2] > 0  # Positive growth

    def test_apply_insurance_zero_attachment(self):
        """Test insurance with zero attachment point."""
        losses = np.array([50000, 100000, 200000])
        attachment = 0
        limit = 150000

        retained, recovered = VectorizedOperations.apply_insurance_vectorized(
            losses, attachment, limit
        )

        assert retained[0] == 0
        assert recovered[0] == 50000
        assert retained[1] == 0
        assert recovered[1] == 100000
        assert retained[2] == 50000  # Excess over limit
        assert recovered[2] == 150000  # Limited to 150000


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class."""

    def test_initialization_default(self):
        """Test default PerformanceOptimizer initialization."""
        optimizer = PerformanceOptimizer()
        assert optimizer.config.enable_vectorization is True
        assert optimizer.config.enable_caching is True
        assert optimizer.cache is not None
        assert isinstance(optimizer.cache, SmartCache)

    def test_initialization_custom(self):
        """Test custom PerformanceOptimizer initialization."""
        config = OptimizationConfig(enable_caching=False, cache_size=500)
        optimizer = PerformanceOptimizer(config=config)
        assert optimizer.config.enable_caching is False
        assert optimizer.config.cache_size == 500
        assert optimizer.cache is not None  # Cache is always created

    def test_profile_execution(self):
        """Test execution profiling."""
        optimizer = PerformanceOptimizer()

        def test_function(n):
            time.sleep(0.01)
            return sum(range(n))

        result = optimizer.profile_execution(test_function, 1000)

        assert isinstance(result, ProfileResult)
        assert result.total_time > 0.01
        assert isinstance(result.function_times, dict)

    def test_profile_execution_with_error(self):
        """Test profiling when function raises error."""
        optimizer = PerformanceOptimizer()

        def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            optimizer.profile_execution(error_function)

    def test_generate_recommendations_basic(self):
        """Test recommendation generation."""
        optimizer = PerformanceOptimizer()

        function_times = {"generate_losses": 5.0, "apply_insurance": 3.0, "calculate_premiums": 0.5}
        memory_usage = 2500.0
        total_time = 10.0

        recommendations = optimizer._generate_recommendations(
            function_times, memory_usage, total_time
        )

        assert len(recommendations) > 0
        assert any("vectorizing" in r.lower() or "memory" in r.lower() for r in recommendations)

    def test_generate_recommendations_high_memory(self):
        """Test recommendations with high memory usage."""
        optimizer = PerformanceOptimizer()

        function_times = {"some_function": 1.0}
        memory_usage = 3500.0  # High memory
        total_time = 5.0

        recommendations = optimizer._generate_recommendations(
            function_times, memory_usage, total_time
        )

        assert any("memory" in r.lower() or "float32" in r.lower() for r in recommendations)

    def test_optimize_loss_generation(self):
        """Test loss generation optimization."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_vectorization=True))

        losses = [100.0, 200.0, 300.0, 400.0, 500.0]

        optimized = optimizer.optimize_loss_generation(losses)

        assert isinstance(optimized, np.ndarray)
        assert len(optimized) == 5
        assert np.allclose(optimized, losses)

    def test_optimize_insurance_calculation(self):
        """Test insurance calculation optimization."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_vectorization=True))

        losses = np.array([50000, 150000, 300000, 1000000])
        layers = [(100000, 500000, 0.01)]

        result = optimizer.optimize_insurance_calculation(losses, layers)

        assert "retained_losses" in result
        assert "total_recovered" in result
        assert "total_premiums" in result
        assert "net_losses" in result
        assert result["total_premiums"] == 5000  # 500000 * 0.01

    def test_optimize_memory_usage(self):
        """Test memory usage optimization."""
        optimizer = PerformanceOptimizer()

        metrics = optimizer.optimize_memory_usage()

        assert "process_memory_mb" in metrics
        assert "available_memory_mb" in metrics
        assert "memory_percent" in metrics
        assert "suggested_chunk_size" in metrics
        assert "cache_cleared" in metrics
        assert metrics["process_memory_mb"] > 0
        assert metrics["available_memory_mb"] > 0

    def test_calculate_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        optimizer = PerformanceOptimizer()

        # Test with 1GB available memory
        available_memory = 1024 * 1024 * 1024  # 1GB in bytes
        chunk_size = optimizer._calculate_optimal_chunk_size(available_memory)

        assert chunk_size >= 1000
        assert chunk_size <= 100000
        assert chunk_size % 1000 == 0  # Should be rounded to nearest 1000

    def test_get_optimization_summary(self):
        """Test optimization summary generation."""
        optimizer = PerformanceOptimizer()

        summary = optimizer.get_optimization_summary()

        assert "Performance Optimization Summary" in summary
        assert "Configuration" in summary
        assert "Cache Performance" in summary
        assert "Memory Usage" in summary
        assert "Vectorization" in summary
        assert "Hit Rate" in summary

    def test_cache_operations_with_caching(self):
        """Test cache operations with caching enabled."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_caching=True))

        # Test insurance calculation caching
        losses = np.array([100000, 200000])
        layers = [(50000, 100000, 0.01)]

        # First call - cache miss
        result1 = optimizer.optimize_insurance_calculation(losses, layers)

        # Second call - should hit cache
        result2 = optimizer.optimize_insurance_calculation(losses, layers)

        assert np.array_equal(result1["retained_losses"], result2["retained_losses"])
        assert optimizer.cache.hits > 0

    def test_hash_cache_key_no_tobytes_copy(self):
        """Test that cache key uses a hash, not raw tobytes() (#499)."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_caching=True))

        losses = np.random.exponential(100000, 10000)
        layers = [(100000, 500000, 0.015)]

        optimizer.optimize_insurance_calculation(losses, layers)

        # The cache key should be a (hexdigest_string, tuple) — not
        # a (bytes, tuple) which would be the case with tobytes().
        for key in optimizer.cache.cache:
            assert isinstance(key[0], str), "Cache key should be a hash string, not bytes"
            assert len(key[0]) == 64, "SHA-256 hex digest should be 64 chars"

    def test_hash_cache_key_same_array_hits(self):
        """Test that identical arrays produce the same hash key."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_caching=True))

        losses = np.array([1.0, 2.0, 3.0])
        layers = [(0.0, 100.0, 0.01)]

        optimizer.optimize_insurance_calculation(losses, layers)
        assert optimizer.cache.misses == 1

        # Same data — should hit cache
        optimizer.optimize_insurance_calculation(losses.copy(), layers)
        assert optimizer.cache.hits == 1

    def test_hash_cache_key_different_array_misses(self):
        """Test that different arrays produce different hash keys."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_caching=True))

        losses1 = np.array([1.0, 2.0, 3.0])
        losses2 = np.array([1.0, 2.0, 4.0])
        layers = [(0.0, 100.0, 0.01)]

        optimizer.optimize_insurance_calculation(losses1, layers)
        optimizer.optimize_insurance_calculation(losses2, layers)
        assert optimizer.cache.misses == 2  # Two distinct keys

    def test_hash_cache_key_non_contiguous_array(self):
        """Test that non-contiguous arrays still produce correct cache keys."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_caching=True))

        base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        non_contiguous = base[:, 0]  # Column slice — not C-contiguous
        assert not non_contiguous.flags["C_CONTIGUOUS"]

        layers = [(0.0, 100.0, 0.01)]

        # Should work without error
        result = optimizer.optimize_insurance_calculation(non_contiguous, layers)
        assert "retained_losses" in result

    def test_profile_function_decorator(self):
        """Test profile function decorator."""

        @profile_function
        def test_func(n):
            return sum(range(n))

        # Capture stdout to check if summary is printed
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = test_func(1000)
            output = sys.stdout.getvalue()
            assert "Performance Profile Summary" in output
            assert result == sum(range(1000))
        finally:
            sys.stdout = old_stdout

    def test_cached_calculation_decorator(self):
        """Test cached calculation decorator."""

        @cached_calculation(cache_size=10)
        def expensive_calc(x):
            return x**2

        # First call
        result1 = expensive_calc(5)
        assert result1 == 25

        # Should be cached (but we can't directly test lru_cache internals)
        result2 = expensive_calc(5)
        assert result2 == 25

    def test_cache_with_large_batch(self):
        """Test cache with large batch processing."""
        optimizer = PerformanceOptimizer(
            config=OptimizationConfig(enable_vectorization=True, chunk_size=5000)
        )

        # Create large loss array
        large_losses = list(range(20000))

        optimized = optimizer.optimize_loss_generation(large_losses, batch_size=5000)

        assert isinstance(optimized, np.ndarray)
        assert len(optimized) == 20000

    def test_optimize_insurance_without_vectorization(self):
        """Test insurance optimization without vectorization."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_vectorization=False))

        losses = np.array([100000, 200000, 300000])
        layers = [(50000, 100000, 0.01)]

        result = optimizer.optimize_insurance_calculation(losses, layers)

        assert "retained_losses" in result
        assert "total_recovered" in result
        assert result["total_premiums"] == 1000  # 100000 * 0.01

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_optimization(self):
        """Test Numba JIT compilation when available."""
        optimizer = PerformanceOptimizer(config=OptimizationConfig(enable_numba=True))

        # Test that Numba decorators are applied
        assert hasattr(VectorizedOperations.calculate_growth_rates, "__wrapped__")

    def test_memory_optimization_with_high_usage(self):
        """Test memory optimization with simulated high usage."""
        optimizer = PerformanceOptimizer()

        # Add some cache entries
        optimizer.cache.set(("key1",), "value1")
        optimizer.cache.set(("key2",), "value2")

        with patch("psutil.virtual_memory") as mock_memory:
            # Simulate high memory usage
            mock_memory.return_value.percent = 85
            mock_memory.return_value.available = 500 * 1024 * 1024

            metrics = optimizer.optimize_memory_usage()

            assert metrics["cache_cleared"] is True
            assert len(optimizer.cache.cache) == 0  # Cache should be cleared

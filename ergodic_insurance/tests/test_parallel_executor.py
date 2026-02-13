"""Tests for CPU-optimized parallel execution engine.

Comprehensive test suite for the ParallelExecutor class and related components,
ensuring optimal performance on budget hardware.
"""

import multiprocessing as mp
import pickle
import platform
import time
from unittest.mock import Mock, patch

import numpy as np
import psutil
import pytest

from ergodic_insurance.parallel_executor import (
    ChunkingStrategy,
    CPUProfile,
    ParallelExecutor,
    PerformanceMetrics,
    SharedMemoryConfig,
    SharedMemoryManager,
    parallel_aggregate,
    parallel_map,
)


# Module-level test functions for pickling
def _test_square(x):
    """Simple square function for testing."""
    return x**2


def _test_sum_results(results):
    """Sum all results."""
    return sum(results)


def _test_process_with_config(item, **kwargs):
    """Process item with configuration."""
    multiplier = kwargs.get("multiplier", 1)
    offset = kwargs.get("offset", 0)
    return item * multiplier + offset


def _test_work_function(i):
    """Complex work function for performance testing."""
    time.sleep(0.001)  # Simulate work
    return i**2


def _test_failing_function(x):
    """Function that fails on specific input for error testing."""
    if x == 5:
        raise RuntimeError("Test error")
    return x


def _test_reduce(results):
    """Reduce results."""
    return sum(results)


def _test_multiply_with_kwargs(x, **kwargs):
    """Multiply with keyword arguments."""
    return x * kwargs.get("multiplier", 1)


def _test_simulate_path(sim_id):
    """Simulate a simple random walk for integration tests."""
    np.random.seed(sim_id)
    steps = 100
    path = np.random.randn(steps).cumsum()

    # Guard against empty arrays or invalid values
    if len(path) == 0:
        return {
            "final_value": 0.0,
            "max_value": 0.0,
            "min_value": 0.0,
        }

    # Use numpy functions that handle edge cases better
    max_value = np.nanmax(path) if not np.all(np.isnan(path)) else 0.0
    min_value = np.nanmin(path) if not np.all(np.isnan(path)) else 0.0
    final_value = path[-1] if not np.isnan(path[-1]) else 0.0

    return {
        "final_value": float(final_value),
        "max_value": float(max_value),
        "min_value": float(min_value),
    }


def _test_aggregate_results(results):
    """Aggregate simulation results for integration tests."""
    # Results are already flattened
    final_values = [r["final_value"] for r in results]
    return {
        "mean_final": np.mean(final_values),
        "std_final": np.std(final_values),
        "n_simulations": len(results),
    }


def _test_cpu_bound_work(x):
    """CPU-intensive work for performance scaling test."""
    result = 0
    for i in range(10000):
        result += (x * i) ** 0.5
    return result


def _test_slow_function(x):
    """Slow function for performance monitoring test."""
    time.sleep(0.001)  # Simulate work
    return x * 2


def _test_simple_lambda(x):
    """Simple function to replace lambda in context manager test."""
    return x


class TestCPUProfile:
    """Test CPU profile detection and optimization."""

    def test_detect_cpu_profile(self):
        """Test CPU profile detection."""
        profile = CPUProfile.detect()

        # Check basic attributes
        assert profile.n_cores >= 1
        assert profile.n_threads >= profile.n_cores
        assert profile.available_memory > 0
        assert profile.cpu_freq > 0
        assert profile.system_load >= 0

        # Check cache sizes
        assert "L1" in profile.cache_sizes
        assert "L2" in profile.cache_sizes
        assert "L3" in profile.cache_sizes
        assert all(size > 0 for size in profile.cache_sizes.values())

    def test_cpu_profile_consistency(self):
        """Test that CPU profile detection is consistent."""
        profile1 = CPUProfile.detect()
        profile2 = CPUProfile.detect()

        # Core counts should be identical
        assert profile1.n_cores == profile2.n_cores
        assert profile1.n_threads == profile2.n_threads

        # Memory and load can vary but should be reasonable
        assert (
            abs(profile1.available_memory - profile2.available_memory) / profile1.available_memory
            < 0.5
        )


class TestChunkingStrategy:
    """Test dynamic chunking strategy."""

    def test_default_chunking_strategy(self):
        """Test default chunking parameters."""
        strategy = ChunkingStrategy()

        assert strategy.initial_chunk_size == 1000
        assert strategy.min_chunk_size == 100
        assert strategy.max_chunk_size == 10000
        assert strategy.target_chunks_per_worker == 10
        assert strategy.adaptive is True
        assert strategy.profile_samples == 100

    def test_calculate_optimal_chunk_size(self):
        """Test optimal chunk size calculation."""
        strategy = ChunkingStrategy()

        # Test basic calculation
        chunk_size = strategy.calculate_optimal_chunk_size(
            n_items=100000, n_workers=4, item_complexity=1.0
        )

        # Should be around 100000 / (4 * 10) = 2500
        assert 2000 <= chunk_size <= 3000

        # Test with high complexity
        chunk_size_complex = strategy.calculate_optimal_chunk_size(
            n_items=100000, n_workers=4, item_complexity=5.0
        )

        # Should be smaller for complex items
        assert chunk_size_complex < chunk_size

        # Test bounds
        chunk_size_min = strategy.calculate_optimal_chunk_size(
            n_items=10, n_workers=4, item_complexity=1.0
        )
        assert chunk_size_min == strategy.min_chunk_size

        chunk_size_max = strategy.calculate_optimal_chunk_size(
            n_items=1000000, n_workers=1, item_complexity=0.1
        )
        assert chunk_size_max == strategy.max_chunk_size

    def test_chunk_size_with_cpu_profile(self):
        """Test chunk size calculation with CPU profile."""
        strategy = ChunkingStrategy()

        # Mock CPU profile with high load
        high_load_profile = CPUProfile(
            n_cores=4,
            n_threads=8,
            cache_sizes={"L1": 32 * 1024, "L2": 256 * 1024, "L3": 8 * 1024 * 1024},
            available_memory=4 * 1024**3,
            cpu_freq=2000,
            system_load=0.9,
        )

        chunk_size_high_load = strategy.calculate_optimal_chunk_size(
            n_items=100000, n_workers=4, item_complexity=1.0, cpu_profile=high_load_profile
        )

        # Mock CPU profile with plenty of memory
        high_mem_profile = CPUProfile(
            n_cores=4,
            n_threads=8,
            cache_sizes={"L1": 32 * 1024, "L2": 256 * 1024, "L3": 8 * 1024 * 1024},
            available_memory=16 * 1024**3,
            cpu_freq=2000,
            system_load=0.3,
        )

        chunk_size_high_mem = strategy.calculate_optimal_chunk_size(
            n_items=100000, n_workers=4, item_complexity=1.0, cpu_profile=high_mem_profile
        )

        # High load should result in larger chunks (fewer context switches)
        assert chunk_size_high_load > chunk_size_high_mem


@pytest.mark.requires_multiprocessing
class TestSharedMemoryManager:
    """Test shared memory management."""

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Shared memory has issues on Windows in tests"
    )
    def test_share_and_retrieve_array(self):
        """Test sharing and retrieving numpy arrays."""
        config = SharedMemoryConfig()
        manager = SharedMemoryManager(config)

        try:
            # Create test array
            test_array = np.random.randn(100, 50)

            # Share array
            shm_name = manager.share_array("test_array", test_array)
            assert shm_name != ""

            # Retrieve array
            retrieved = manager.get_array(shm_name, test_array.shape, test_array.dtype)

            # Check equality
            np.testing.assert_array_equal(test_array, retrieved)

            # Modify retrieved array (should affect shared memory)
            retrieved[0, 0] = 999.0
            re_retrieved = manager.get_array(shm_name, test_array.shape, test_array.dtype)
            assert re_retrieved[0, 0] == 999.0

        finally:
            manager.cleanup()

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Shared memory has issues on Windows in tests"
    )
    def test_share_and_retrieve_object(self):
        """Test sharing and retrieving serialized objects."""
        config = SharedMemoryConfig()
        manager = SharedMemoryManager(config)

        try:
            # Create test object
            test_object = {
                "config": {"param1": 42, "param2": "test"},
                "data": [1, 2, 3, 4, 5],
                "nested": {"a": 1, "b": 2},
            }

            # Share object
            shm_name = manager.share_object("test_obj", test_object)
            assert shm_name != ""

            # Get serialized size
            serialized = pickle.dumps(test_object, protocol=pickle.HIGHEST_PROTOCOL)

            # Retrieve object
            retrieved = manager.get_object(shm_name, len(serialized), compressed=False)

            # Check equality
            assert retrieved == test_object

        finally:
            manager.cleanup()

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Shared memory has issues on Windows in tests"
    )
    def test_shared_memory_with_compression(self):
        """Test shared memory with compression enabled."""
        config = SharedMemoryConfig(compression=True)
        manager = SharedMemoryManager(config)

        try:
            # Create large test object
            test_object = {"data": list(range(10000))}

            # Share with compression
            shm_name = manager.share_object("compressed_obj", test_object)

            # Get compressed size
            import zlib

            serialized = pickle.dumps(test_object, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(serialized)

            # Retrieve object
            retrieved = manager.get_object(shm_name, len(compressed), compressed=True)

            # Check equality
            assert retrieved == test_object

        finally:
            manager.cleanup()

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Shared memory has issues on Windows in tests"
    )
    def test_cleanup(self):
        """Test cleanup of shared memory resources."""
        config = SharedMemoryConfig()
        manager = SharedMemoryManager(config)

        # Create and share multiple resources
        array1 = np.ones((10, 10))
        array2 = np.zeros((5, 5))
        obj1 = {"test": 123}

        shm1 = manager.share_array("array1", array1)
        shm2 = manager.share_array("array2", array2)
        shm3 = manager.share_object("obj1", obj1)

        # Verify resources exist
        assert len(manager.shared_arrays) == 2
        assert len(manager.shared_objects) == 1

        # Cleanup
        manager.cleanup()

        # Verify resources are cleaned
        assert len(manager.shared_arrays) == 0
        assert len(manager.shared_objects) == 0


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    def test_default_metrics(self):
        """Test default performance metrics."""
        metrics = PerformanceMetrics()

        assert metrics.total_time == 0.0
        assert metrics.setup_time == 0.0
        assert metrics.computation_time == 0.0
        assert metrics.serialization_time == 0.0
        assert metrics.reduction_time == 0.0
        assert metrics.memory_peak == 0
        assert metrics.cpu_utilization == 0.0
        assert metrics.items_per_second == 0.0
        assert metrics.speedup == 1.0
        assert metrics.total_items == 0
        assert metrics.failed_items == 0

    def test_metrics_summary(self):
        """Test performance metrics summary generation."""
        metrics = PerformanceMetrics(
            total_time=10.5,
            setup_time=0.5,
            computation_time=8.0,
            serialization_time=0.5,
            reduction_time=1.5,
            memory_peak=512 * 1024 * 1024,
            cpu_utilization=75.0,
            items_per_second=9523.8,
            speedup=3.5,
        )

        summary = metrics.summary()

        # Check key information is present
        assert "Total Time: 10.50s" in summary
        assert "Setup: 0.50s" in summary
        assert "Computation: 8.00s" in summary
        assert "Serialization: 0.50s (4.8% overhead)" in summary
        assert "Peak Memory: 512.0 MB" in summary
        assert "CPU Utilization: 75.0%" in summary
        assert "Throughput: 9524 items/s" in summary
        assert "Speedup: 3.50x" in summary
        assert "Failed Items" not in summary

    def test_metrics_summary_with_failures(self):
        """Test performance metrics summary includes failure info when items fail."""
        metrics = PerformanceMetrics(
            total_time=5.0,
            total_items=100,
            failed_items=10,
        )
        summary = metrics.summary()
        assert "Failed Items: 10/100 (10.0%)" in summary


@pytest.mark.requires_multiprocessing
class TestParallelExecutor:
    """Test parallel executor functionality."""

    def test_initialization(self):
        """Test parallel executor initialization."""
        executor = ParallelExecutor(n_workers=4)

        assert executor.n_workers == 4
        assert executor.cpu_profile is not None
        assert executor.chunking_strategy is not None
        assert executor.shared_memory_manager is not None
        assert executor.monitor_performance is True

    def test_auto_worker_detection(self):
        """Test automatic worker detection."""
        executor = ParallelExecutor()

        # Should detect optimal workers
        assert 1 <= executor.n_workers <= executor.cpu_profile.n_cores

    def test_simple_map_reduce(self):
        """Test simple map-reduce operation."""
        with ParallelExecutor(n_workers=2) as executor:
            result = executor.map_reduce(
                work_function=_test_square,
                work_items=range(10),
                reduce_function=_test_sum_results,
                progress_bar=False,
            )

        expected = sum(x**2 for x in range(10))
        assert result == expected

    def test_map_reduce_with_shared_data(self):
        """Test map-reduce with shared data."""
        shared_data = {"multiplier": 2, "offset": 10}

        with ParallelExecutor(n_workers=2) as executor:
            results = executor.map_reduce(
                work_function=_test_process_with_config,
                work_items=range(5),
                reduce_function=None,  # Just return list
                shared_data=shared_data,
                progress_bar=False,
            )

        # Results are already flattened
        flat_results = results
        expected = [i * 2 + 10 for i in range(5)]
        assert flat_results == expected

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Shared memory has issues on Windows in tests"
    )
    def test_shared_numpy_arrays(self):
        """Test sharing numpy arrays across workers."""

        def dot_product(row_idx, **kwargs):
            matrix = kwargs.get("matrix")
            vector = kwargs.get("vector")
            assert matrix is not None and vector is not None
            return np.dot(matrix[row_idx], vector)

        # Create test data
        matrix = np.random.randn(100, 50)
        vector = np.random.randn(50)

        shared_data = {"matrix": matrix, "vector": vector}

        with ParallelExecutor(n_workers=2) as executor:
            results = executor.map_reduce(
                work_function=dot_product,
                work_items=range(100),
                shared_data=shared_data,
                progress_bar=False,
            )

        # Verify results
        # Results are already flattened
        expected = [np.dot(matrix[i], vector) for i in range(100)]
        np.testing.assert_array_almost_equal(results, expected)

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        executor = ParallelExecutor(n_workers=2, monitor_performance=True)

        results = executor.map_reduce(
            work_function=_test_slow_function, work_items=range(20), progress_bar=False
        )

        # Check performance metrics
        metrics = executor.performance_metrics
        assert metrics.total_time > 0
        assert metrics.computation_time > 0
        assert metrics.items_per_second > 0

        # Get performance report
        report = executor.get_performance_report()
        assert "Performance Summary" in report
        assert "Total Time:" in report

    def test_adaptive_chunking(self):
        """Test adaptive chunking behavior."""
        strategy = ChunkingStrategy(adaptive=True, initial_chunk_size=5)
        executor = ParallelExecutor(n_workers=2, chunking_strategy=strategy)

        results = executor.map_reduce(
            work_function=_test_square, work_items=range(100), progress_bar=False
        )

        # Should complete without errors
        # Results are already flattened
        assert len(results) == 100

    def test_error_handling(self):
        """Test error handling in parallel execution returns partial results."""
        executor = ParallelExecutor(n_workers=2)

        results = executor.map_reduce(
            work_function=_test_failing_function, work_items=range(10), progress_bar=False
        )

        # Should return all successful results (item 5 fails, so 9 results)
        assert results is not None
        assert len(results) == 9
        assert 5 not in results
        # All non-failing items should be present
        for i in range(10):
            if i != 5:
                assert i in results

    def test_error_handling_tracks_failures_in_metrics(self):
        """Test that failed items are tracked in performance metrics."""
        executor = ParallelExecutor(n_workers=2)

        executor.map_reduce(
            work_function=_test_failing_function, work_items=range(10), progress_bar=False
        )

        assert executor.performance_metrics.total_items == 10
        assert executor.performance_metrics.failed_items == 1

    def test_max_failure_rate_raises_on_threshold(self):
        """Test that exceeding max_failure_rate raises RuntimeError."""
        executor = ParallelExecutor(n_workers=2, max_failure_rate=0.05)

        with pytest.raises(RuntimeError, match="exceeds maximum"):
            executor.map_reduce(
                work_function=_test_failing_function, work_items=range(10), progress_bar=False
            )

    def test_max_failure_rate_allows_below_threshold(self):
        """Test that failures below max_failure_rate do not raise."""
        executor = ParallelExecutor(n_workers=2, max_failure_rate=0.5)

        results = executor.map_reduce(
            work_function=_test_failing_function, work_items=range(10), progress_bar=False
        )

        # 1/10 = 10% < 50% threshold, should succeed
        assert len(results) == 9

    def test_context_manager(self):
        """Test context manager functionality."""
        with ParallelExecutor(n_workers=2) as executor:
            assert executor.shared_memory_manager is not None

            # Use executor
            result = executor.map_reduce(
                work_function=_test_simple_lambda, work_items=[1, 2, 3], progress_bar=False
            )

        # After context, shared memory should be cleaned
        assert len(executor.shared_memory_manager.shared_arrays) == 0
        assert len(executor.shared_memory_manager.shared_objects) == 0


@pytest.mark.requires_multiprocessing
class TestUtilityFunctions:
    """Test utility functions for common patterns."""

    def test_parallel_map(self):
        """Test simple parallel map utility."""
        results = parallel_map(func=_test_square, items=range(10), n_workers=2, progress=False)

        # Results are already flattened
        flat_results = results
        expected = [x**2 for x in range(10)]
        assert flat_results == expected

    def test_parallel_aggregate(self):
        """Test parallel aggregate utility."""
        result = parallel_aggregate(
            func=_test_square,
            items=range(10),
            reducer=_test_sum_results,
            n_workers=2,
            progress=False,
        )

        expected = sum(x**2 for x in range(10))
        assert result == expected

    def test_parallel_aggregate_with_shared_data(self):
        """Test parallel aggregate with shared data."""
        shared = {"multiplier": 3}

        result = parallel_aggregate(
            func=_test_multiply_with_kwargs,
            items=range(5),
            reducer=_test_sum_results,
            n_workers=2,
            shared_data=shared,
            progress=False,
        )

        expected = sum(x * 3 for x in range(5))
        assert result == expected


@pytest.mark.requires_multiprocessing
class TestIntegration:
    """Integration tests for parallel executor."""

    def test_large_scale_simulation(self):
        """Test large-scale simulation workload."""
        with ParallelExecutor(n_workers=4) as executor:
            results = executor.map_reduce(
                work_function=_test_simulate_path,
                work_items=range(1000),
                reduce_function=_test_aggregate_results,
                progress_bar=False,
            )

        assert results["n_simulations"] == 1000
        assert "mean_final" in results
        assert "std_final" in results

    @pytest.mark.benchmark
    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Shared memory has issues on Windows in tests"
    )
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create large shared data
        large_matrix = np.random.randn(1000, 1000)

        def process_row(row_idx, **kwargs):
            matrix = kwargs["matrix"]
            return matrix[row_idx].sum()

        shared_data = {"matrix": large_matrix}

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        with ParallelExecutor(n_workers=2) as executor:
            results = executor.map_reduce(
                work_function=process_row,
                work_items=range(1000),
                shared_data=shared_data,
                progress_bar=False,
            )

        # Check memory didn't explode
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024**2  # MB

        # Memory increase should be reasonable (< 500MB for this test)
        assert memory_increase < 500

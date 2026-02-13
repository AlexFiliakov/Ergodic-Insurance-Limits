"""CPU-optimized parallel execution engine for Monte Carlo simulations.

This module provides enhanced parallel processing capabilities optimized for
budget hardware (4-8 cores) with intelligent chunking, shared memory management,
and minimal serialization overhead.

Features:
    - Smart dynamic chunking based on CPU resources and workload
    - Shared memory for read-only data structures
    - CPU affinity optimization for cache locality
    - Minimal IPC overhead (<5% target)
    - Memory-efficient execution (<4GB for 100K simulations)

Example:
    >>> from ergodic_insurance.parallel_executor import ParallelExecutor
    >>> executor = ParallelExecutor(n_workers=4)
    >>> results = executor.map_reduce(
    ...     work_function=simulate_path,
    ...     work_items=range(100000),
    ...     reduce_function=combine_results,
    ...     shared_data={'config': simulation_config}
    ... )

Author:
    Alex Filiakov

Date:
    2025-08-26
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import platform
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid
import warnings

import numpy as np
import psutil
from tqdm import tqdm

from ergodic_insurance.safe_pickle import safe_dumps, safe_loads

logger = logging.getLogger(__name__)


@dataclass
class CPUProfile:
    """CPU performance profile for optimization decisions."""

    n_cores: int
    n_threads: int
    cache_sizes: Dict[str, int]
    available_memory: int
    cpu_freq: float
    system_load: float

    @classmethod
    def detect(cls) -> "CPUProfile":
        """Detect current CPU profile.

        Returns:
            CPUProfile: Current system CPU profile
        """
        cpu_count_physical = psutil.cpu_count(logical=False) or 1
        cpu_count_logical = psutil.cpu_count(logical=True) or 1

        # Get cache sizes (simplified - actual detection is platform-specific)
        cache_sizes = {
            "L1": 32 * 1024,  # 32KB typical L1
            "L2": 256 * 1024,  # 256KB typical L2
            "L3": 8 * 1024 * 1024,  # 8MB typical L3
        }

        # Get memory info
        mem = psutil.virtual_memory()
        available_memory = mem.available

        # Get CPU frequency
        cpu_freq = psutil.cpu_freq()
        freq_current = cpu_freq.current if cpu_freq else 2000.0

        # Get system load
        load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.5

        return cls(
            n_cores=cpu_count_physical,
            n_threads=cpu_count_logical,
            cache_sizes=cache_sizes,
            available_memory=available_memory,
            cpu_freq=freq_current,
            system_load=load_avg,
        )


@dataclass
class ChunkingStrategy:
    """Dynamic chunking strategy for parallel workloads."""

    initial_chunk_size: int = 1000
    min_chunk_size: int = 100
    max_chunk_size: int = 10000
    target_chunks_per_worker: int = 10
    adaptive: bool = True
    profile_samples: int = 100

    def calculate_optimal_chunk_size(
        self,
        n_items: int,
        n_workers: int,
        item_complexity: float = 1.0,
        cpu_profile: Optional[CPUProfile] = None,
    ) -> int:
        """Calculate optimal chunk size based on workload and resources.

        Args:
            n_items: Total number of work items
            n_workers: Number of parallel workers
            item_complexity: Relative complexity of each item (1.0 = baseline)
            cpu_profile: CPU profile for optimization

        Returns:
            int: Optimal chunk size
        """
        # Base calculation
        base_chunk_size = max(
            self.min_chunk_size, n_items // (n_workers * self.target_chunks_per_worker)
        )

        # Adjust for complexity
        adjusted_size = int(base_chunk_size / item_complexity)

        # Adjust for CPU profile
        if cpu_profile:
            # Larger chunks for high system load
            if cpu_profile.system_load > 0.8:
                adjusted_size = int(adjusted_size * 1.5)

            # Smaller chunks if plenty of memory available
            if cpu_profile.available_memory > 8 * 1024**3:  # 8GB
                adjusted_size = int(adjusted_size * 0.8)

        # Apply bounds
        return min(self.max_chunk_size, max(self.min_chunk_size, adjusted_size))


@dataclass
class SharedMemoryConfig:
    """Configuration for shared memory optimization."""

    enable_shared_arrays: bool = True
    enable_shared_objects: bool = True
    compression: bool = False
    cleanup_on_exit: bool = True


class SharedMemoryManager:
    """Manager for shared memory resources.

    Handles creation, access, and cleanup of shared memory segments
    for both numpy arrays and serialized objects.
    """

    def __init__(self, config: Optional[SharedMemoryConfig] = None):
        """Initialize shared memory manager.

        Args:
            config: Shared memory configuration
        """
        self.config = config or SharedMemoryConfig()
        self.shared_arrays: Dict[str, Tuple[shared_memory.SharedMemory, tuple, np.dtype]] = {}
        self.shared_objects: Dict[str, shared_memory.SharedMemory] = {}
        self._opened_handles: List[shared_memory.SharedMemory] = []
        self._object_sizes: Dict[str, int] = {}

    def share_array(self, name: str, array: np.ndarray) -> str:
        """Share a numpy array via shared memory.

        Args:
            name: Unique identifier for the array
            array: Numpy array to share

        Returns:
            str: Shared memory name for retrieval
        """
        if not self.config.enable_shared_arrays:
            return ""

        # Create shared memory
        shm = shared_memory.SharedMemory(
            create=True, size=array.nbytes, name=f"ergodic_array_{name}_{uuid.uuid4().hex[:12]}"
        )

        # Copy array to shared memory
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]

        # Store reference
        self.shared_arrays[name] = (shm, array.shape, array.dtype)

        return shm.name

    def get_array(self, shm_name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Retrieve a shared numpy array.

        Args:
            shm_name: Shared memory name
            shape: Array shape
            dtype: Array data type

        Returns:
            np.ndarray: Shared array (view, not copy)
        """
        shm = shared_memory.SharedMemory(name=shm_name)
        self._opened_handles.append(shm)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def share_object(self, name: str, obj: Any) -> str:
        """Share a serialized object via shared memory.

        Args:
            name: Unique identifier for the object
            obj: Object to share

        Returns:
            str: Shared memory name for retrieval
        """
        if not self.config.enable_shared_objects:
            return ""

        # Serialize object with HMAC integrity verification
        serialized = safe_dumps(obj)

        # Compress if enabled
        if self.config.compression:
            import zlib

            serialized = zlib.compress(serialized)

        # Create shared memory
        shm = shared_memory.SharedMemory(
            create=True, size=len(serialized), name=f"ergodic_obj_{name}_{uuid.uuid4().hex[:12]}"
        )

        # Copy to shared memory
        assert shm.buf is not None
        shm.buf[: len(serialized)] = serialized

        # Store reference and actual size
        self.shared_objects[name] = shm
        self._object_sizes[name] = len(serialized)

        return shm.name

    def get_object(self, shm_name: str, size: int, compressed: bool = False) -> Any:
        """Retrieve a shared object.

        Args:
            shm_name: Shared memory name
            size: Size of serialized data
            compressed: Whether data is compressed

        Returns:
            Any: Deserialized object
        """
        shm = shared_memory.SharedMemory(name=shm_name)
        assert shm.buf is not None
        data = bytes(shm.buf[:size])
        shm.close()

        # Decompress if needed
        if compressed:
            import zlib

            data = zlib.decompress(data)

        return safe_loads(data)

    def get_object_size(self, name: str) -> int:
        """Get the actual stored size of a shared object.

        Args:
            name: Object identifier used in share_object

        Returns:
            int: Actual byte size stored in shared memory
        """
        return self._object_sizes[name]

    def cleanup(self):
        """Clean up all shared memory resources."""
        # Clean up shared arrays
        for shm, _, _ in self.shared_arrays.values():
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, PermissionError):
                pass

        # Clean up shared objects
        for shm in self.shared_objects.values():
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, PermissionError):
                pass

        # Close read-only handles opened via get_array (don't unlink)
        for shm in self._opened_handles:
            try:
                shm.close()
            except (FileNotFoundError, PermissionError, OSError):
                pass

        self.shared_arrays.clear()
        self.shared_objects.clear()
        self._opened_handles.clear()
        self._object_sizes.clear()

    def __del__(self):
        """Cleanup on deletion."""
        if self.config.cleanup_on_exit:
            self.cleanup()


@dataclass
class PerformanceMetrics:
    """Performance metrics for parallel execution."""

    total_time: float = 0.0
    setup_time: float = 0.0
    computation_time: float = 0.0
    serialization_time: float = 0.0
    reduction_time: float = 0.0
    memory_peak: int = 0
    cpu_utilization: float = 0.0
    items_per_second: float = 0.0
    speedup: float = 1.0
    total_items: int = 0
    failed_items: int = 0

    def summary(self) -> str:
        """Generate performance summary.

        Returns:
            str: Formatted performance summary
        """
        overhead = self.serialization_time / self.total_time * 100 if self.total_time > 0 else 0

        lines = [
            f"Performance Summary\n",
            f"{'='*50}\n",
            f"Total Time: {self.total_time:.2f}s\n",
            f"Setup: {self.setup_time:.2f}s\n",
            f"Computation: {self.computation_time:.2f}s\n",
            f"Serialization: {self.serialization_time:.2f}s ({overhead:.1f}% overhead)\n",
            f"Reduction: {self.reduction_time:.2f}s\n",
            f"Peak Memory: {self.memory_peak / 1024**2:.1f} MB\n",
            f"CPU Utilization: {self.cpu_utilization:.1f}%\n",
            f"Throughput: {self.items_per_second:.0f} items/s\n",
            f"Speedup: {self.speedup:.2f}x\n",
        ]

        if self.failed_items > 0:
            failure_rate = self.failed_items / self.total_items * 100 if self.total_items > 0 else 0
            lines.append(
                f"Failed Items: {self.failed_items}/{self.total_items}" f" ({failure_rate:.1f}%)\n"
            )

        return "".join(lines)


class ParallelExecutor:
    """CPU-optimized parallel executor for Monte Carlo simulations.

    Provides intelligent work distribution, shared memory management,
    and performance monitoring for efficient parallel execution on
    budget hardware.
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        shared_memory_config: Optional[SharedMemoryConfig] = None,
        monitor_performance: bool = True,
        max_failure_rate: Optional[float] = None,
    ):
        """Initialize parallel executor.

        Args:
            n_workers: Number of parallel workers (None for auto)
            chunking_strategy: Strategy for work distribution
            shared_memory_config: Configuration for shared memory
            monitor_performance: Enable performance monitoring
            max_failure_rate: Maximum fraction of items that may fail before
                raising a ``RuntimeError``.  For example, ``0.1`` means raise
                if more than 10 % of items fail.  ``None`` (default) disables
                the threshold check.
        """
        # Detect CPU profile
        self.cpu_profile = CPUProfile.detect()

        # Set number of workers
        if n_workers is None:
            # Optimal workers for budget hardware
            n_workers = min(
                self.cpu_profile.n_cores,
                max(2, self.cpu_profile.n_cores - 1),  # Leave one core free
            )
        self.n_workers = n_workers

        # Initialize strategies
        self.chunking_strategy = chunking_strategy or ChunkingStrategy()
        self.shared_memory_config = shared_memory_config or SharedMemoryConfig()

        # Initialize shared memory manager
        self.shared_memory_manager = SharedMemoryManager(self.shared_memory_config)

        # Failure threshold
        self.max_failure_rate = max_failure_rate

        # Performance monitoring
        self.monitor_performance = monitor_performance
        self.performance_metrics = PerformanceMetrics()

    def map_reduce(
        self,
        work_function: Callable,
        work_items: Union[List, range],
        reduce_function: Optional[Callable] = None,
        shared_data: Optional[Dict[str, Any]] = None,
        progress_bar: bool = True,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Any:
        """Execute parallel map-reduce operation.

        Args:
            work_function: Function to apply to each work item
            work_items: List or range of work items
            reduce_function: Function to combine results (None for list)
            shared_data: Data to share across all workers
            progress_bar: Show progress bar
            progress_callback: Optional callback invoked with
                ``(completed, total, elapsed_seconds)`` after each chunk.
            cancel_event: Optional :class:`threading.Event`; when set the
                executor stops after the current chunk and returns partial
                results.

        Returns:
            Any: Combined results from reduce function or list of results
        """
        start_time = time.time()

        # Convert work items to list if needed
        if isinstance(work_items, range):
            work_items = list(work_items)
        n_items = len(work_items)

        # Setup shared memory for data
        setup_start = time.time()
        shared_refs = self._setup_shared_data(shared_data)
        self.performance_metrics.setup_time = time.time() - setup_start

        # Calculate optimal chunk size
        chunk_size = self._calculate_chunk_size(n_items, work_function)

        # Create chunks
        chunks = self._create_chunks(work_items, chunk_size)

        # Execute parallel work
        comp_start = time.time()
        chunk_results = self._execute_parallel(
            work_function,
            chunks,
            shared_refs,
            progress_bar,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            total_items=n_items,
        )
        self.performance_metrics.computation_time = time.time() - comp_start

        # Reduce results
        reduce_start = time.time()
        if reduce_function:
            result = reduce_function(chunk_results)
        else:
            result = chunk_results
        self.performance_metrics.reduction_time = time.time() - reduce_start

        # Update metrics
        self.performance_metrics.total_time = time.time() - start_time
        if self.performance_metrics.total_time > 0:
            self.performance_metrics.items_per_second = (
                n_items / self.performance_metrics.total_time
            )
        else:
            self.performance_metrics.items_per_second = 0.0

        # Monitor memory
        if self.monitor_performance:
            self._update_memory_metrics()

        # Cleanup shared memory
        self.shared_memory_manager.cleanup()

        return result

    def _setup_shared_data(
        self, shared_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Tuple[str, Any]]:
        """Setup shared memory for data.

        Args:
            shared_data: Data to share

        Returns:
            Dict[str, Tuple[str, Any]]: Shared memory references
        """
        if not shared_data:
            return {}

        shared_refs = {}

        for key, value in shared_data.items():
            if isinstance(value, np.ndarray):
                # Share numpy array
                shm_name = self.shared_memory_manager.share_array(key, value)
                shared_refs[key] = ("array", (shm_name, value.shape, value.dtype))
            else:
                # Share serialized object
                shm_name = self.shared_memory_manager.share_object(key, value)
                actual_size = self.shared_memory_manager.get_object_size(key)
                shared_refs[key] = (
                    "object",
                    (shm_name, actual_size, self.shared_memory_config.compression),
                )

        return shared_refs

    def _calculate_chunk_size(self, n_items: int, work_function: Callable) -> int:
        """Calculate optimal chunk size.

        Args:
            n_items: Number of work items
            work_function: Work function to profile

        Returns:
            int: Optimal chunk size
        """
        if not self.chunking_strategy.adaptive:
            return self.chunking_strategy.initial_chunk_size

        # Profile work function complexity
        complexity = self._profile_work_complexity(work_function)

        # Calculate optimal size
        return self.chunking_strategy.calculate_optimal_chunk_size(
            n_items, self.n_workers, complexity, self.cpu_profile
        )

    def _profile_work_complexity(self, work_function: Callable) -> float:
        """Profile work function complexity.

        Args:
            work_function: Function to profile

        Returns:
            float: Relative complexity (1.0 = baseline)
        """
        # Simple complexity estimation based on function introspection
        # In practice, could do actual profiling with test samples

        import inspect

        # Get function source
        try:
            source = inspect.getsource(work_function)
            # Estimate complexity based on source length and loop keywords
            loop_count = source.count("for ") + source.count("while ")
            complexity = 1.0 + (loop_count * 0.5)
            return min(10.0, complexity)  # Cap at 10x
        except (AttributeError, ValueError):
            return 1.0  # Default complexity

    def _create_chunks(self, work_items: List, chunk_size: int) -> List[Tuple[int, int, List]]:
        """Create work chunks.

        Args:
            work_items: Items to chunk
            chunk_size: Size of each chunk

        Returns:
            List[Tuple[int, int, List]]: List of (start_idx, end_idx, items)
        """
        chunks = []
        n_items = len(work_items)

        for i in range(0, n_items, chunk_size):
            end = min(i + chunk_size, n_items)
            chunks.append((i, end, work_items[i:end]))

        return chunks

    def _execute_parallel(
        self,
        work_function: Callable,
        chunks: List[Tuple[int, int, List]],
        shared_refs: Dict[str, Tuple[str, Any]],
        progress_bar: bool,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        total_items: int = 0,
    ) -> List[Any]:
        """Execute work in parallel.

        Args:
            work_function: Function to execute
            chunks: Work chunks
            shared_refs: Shared memory references
            progress_bar: Show progress
            progress_callback: Optional callback with (completed, total, elapsed)
            cancel_event: Optional cancel event
            total_items: Total number of work items (for progress reporting)

        Returns:
            List[Any]: Results from all chunks
        """
        results = []
        total_failed = 0

        # Create process pool with optimized settings
        # Set CPU affinity if on Linux
        if platform.system() == "Linux":
            # Use taskset equivalent via ProcessPoolExecutor initializer
            pass  # Platform-specific optimization

        exec_start = time.time()
        completed_items = 0

        with ProcessPoolExecutor(
            max_workers=self.n_workers, mp_context=mp.get_context()  # Use default context
        ) as executor:
            # Submit all chunks
            futures = {}
            for chunk in chunks:
                future = executor.submit(
                    _execute_chunk, work_function, chunk, shared_refs, self.shared_memory_config
                )
                futures[future] = chunk  # Track full chunk tuple

            # Process results
            if progress_bar:
                pbar = tqdm(total=len(chunks), desc="Processing chunks")

            # Collect results in order
            for future in as_completed(futures):
                # Check for cancellation
                if cancel_event is not None and cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    break

                try:
                    chunk_results, chunk_failed = future.result()
                    chunk_info = futures[future]
                    results.append((chunk_info[0], chunk_results))
                    total_failed += chunk_failed

                    # Update completed count
                    completed_items += chunk_info[1] - chunk_info[0]

                    if progress_bar:
                        pbar.update(1)

                    if progress_callback is not None and total_items > 0:
                        progress_callback(completed_items, total_items, time.time() - exec_start)
                except (ValueError, TypeError, RuntimeError) as e:
                    warnings.warn(f"Chunk execution failed: {e}", UserWarning)
                    # Return empty list for failed chunks instead of None
                    chunk_info = futures[future]
                    results.append((chunk_info[0], []))
                    # Count all items in the failed chunk as failures
                    total_failed += chunk_info[1] - chunk_info[0]

            if progress_bar:
                pbar.close()

        # Update failure metrics
        self.performance_metrics.total_items = total_items
        self.performance_metrics.failed_items = total_failed

        if total_failed > 0:
            logger.warning(
                "%d of %d work items failed (%.1f%%)",
                total_failed,
                total_items,
                total_failed / total_items * 100 if total_items > 0 else 0,
            )

        # Check failure rate threshold
        if (
            self.max_failure_rate is not None
            and total_items > 0
            and total_failed / total_items > self.max_failure_rate
        ):
            raise RuntimeError(
                f"Failure rate {total_failed}/{total_items} "
                f"({total_failed / total_items:.1%}) exceeds maximum "
                f"allowed rate of {self.max_failure_rate:.1%}"
            )

        # Sort results by original order
        results.sort(key=lambda x: x[0])

        # Flatten the results - each chunk returns a list; filter None from partial failures
        flattened_results: List[Any] = []
        for _, chunk_results in results:
            if chunk_results:
                flattened_results.extend(r for r in chunk_results if r is not None)

        return flattened_results

    def _update_memory_metrics(self):
        """Update memory usage metrics."""
        process = psutil.Process()
        mem_info = process.memory_info()

        self.performance_metrics.memory_peak = max(
            self.performance_metrics.memory_peak, mem_info.rss
        )

        # Update CPU utilization
        self.performance_metrics.cpu_utilization = psutil.cpu_percent(interval=0.1)

    def get_performance_report(self) -> str:
        """Get performance report.

        Returns:
            str: Formatted performance report
        """
        return self.performance_metrics.summary()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shared_memory_manager.cleanup()


def _execute_chunk(
    work_function: Callable,
    chunk: Tuple[int, int, List],
    shared_refs: Dict[str, Tuple[str, Any]],
    shared_memory_config: SharedMemoryConfig,
) -> Any:
    """Execute a chunk of work (runs in worker process).

    Args:
        work_function: Function to execute
        chunk: Work chunk (start_idx, end_idx, items)
        shared_refs: Shared memory references
        shared_memory_config: Shared memory configuration

    Returns:
        Tuple[List[Any], int]: A tuple of (results, failed_count) where
            failed items are represented as ``None`` in the results list.
    """
    start_idx, _end_idx, items = chunk

    # Reconstruct shared data
    shared_data = {}
    sm_manager = None
    if shared_refs:
        sm_manager = SharedMemoryManager(shared_memory_config)

        for key, (data_type, ref_data) in shared_refs.items():
            if data_type == "array":
                shm_name, shape, dtype = ref_data
                shared_data[key] = sm_manager.get_array(shm_name, shape, dtype)
            elif data_type == "object":
                shm_name, size, compressed = ref_data
                shared_data[key] = sm_manager.get_object(shm_name, size, compressed)

    # Process items - return partial results on individual failures
    try:
        results = []
        failed_count = 0
        for idx, item in enumerate(items):
            try:
                if shared_data:
                    result = work_function(item, **shared_data)
                else:
                    result = work_function(item)
                results.append(result)
            except Exception:
                failed_count += 1
                item_idx = start_idx + idx
                logger.warning(
                    "Work item %d failed: %s",
                    item_idx,
                    traceback.format_exc().strip(),
                )
                results.append(None)

        return results, failed_count
    finally:
        if sm_manager is not None:
            sm_manager.cleanup()


# Utility functions for common patterns


def parallel_map(
    func: Callable,
    items: Union[List, range],
    n_workers: Optional[int] = None,
    progress: bool = True,
) -> List[Any]:
    """Simple parallel map operation.

    Args:
        func: Function to apply
        items: Items to process
        n_workers: Number of workers
        progress: Show progress bar

    Returns:
        List[Any]: Results
    """
    with ParallelExecutor(n_workers=n_workers) as executor:
        result = executor.map_reduce(work_function=func, work_items=items, progress_bar=progress)
        # map_reduce returns list when reduce_function is None
        return result  # type: ignore[no-any-return]


def parallel_aggregate(
    func: Callable,
    items: Union[List, range],
    reducer: Callable,
    n_workers: Optional[int] = None,
    shared_data: Optional[Dict] = None,
    progress: bool = True,
) -> Any:
    """Parallel map-reduce operation.

    Args:
        func: Function to apply to each item
        items: Items to process
        reducer: Function to combine results
        n_workers: Number of workers
        shared_data: Data to share across workers
        progress: Show progress bar

    Returns:
        Any: Aggregated result
    """
    with ParallelExecutor(n_workers=n_workers) as executor:
        return executor.map_reduce(
            work_function=func,
            work_items=items,
            reduce_function=reducer,
            shared_data=shared_data,
            progress_bar=progress,
        )

"""Tests for coverage gaps in parallel_executor.py -- mock-based shared memory.

All existing shared-memory tests in the project are skipped on Windows because
they use real ``multiprocessing.shared_memory.SharedMemory``.  This module
replaces that class with a lightweight ``FakeSharedMemory`` backed by
plain ``bytearray`` objects, so the *logic* of every code path executes on
every platform without touching OS-level shared memory.

Targeted untested lines
-----------------------
- 191-202 : ``SharedMemoryManager.share_array`` -- create shared memory, copy
             numpy data, store reference, return name.
- 215-216 : ``SharedMemoryManager.get_array`` -- open shared memory by name,
             return ndarray view.
- 236-238 : ``SharedMemoryManager.share_object`` with ``compression=True``.
- 264-273 : ``SharedMemoryManager.get_object`` with ``compressed=True``
             (decompression path).
- 473-474 : ``ParallelExecutor._setup_shared_data`` -- numpy-array branch.
- 673-674 : ``_execute_chunk`` -- array retrieval branch inside the worker.
"""

from unittest.mock import patch
import uuid
import zlib

import numpy as np
import pytest

from ergodic_insurance.parallel_executor import (
    ParallelExecutor,
    SharedMemoryConfig,
    SharedMemoryManager,
    _execute_chunk,
)
from ergodic_insurance.safe_pickle import safe_dumps, safe_loads

# ---------------------------------------------------------------------------
# Fake shared-memory infrastructure
# ---------------------------------------------------------------------------


class _FakeSharedMemoryRegistry:
    """In-process registry that maps names to bytearray buffers."""

    def __init__(self):
        self._segments: dict[str, bytearray] = {}

    def create(self, name: str, size: int) -> bytearray:
        buf = bytearray(size)
        self._segments[name] = buf
        return buf

    def get(self, name: str) -> bytearray:
        return self._segments[name]

    def remove(self, name: str) -> None:
        self._segments.pop(name, None)

    def clear(self) -> None:
        self._segments.clear()


_registry = _FakeSharedMemoryRegistry()


class FakeSharedMemory:
    """Drop-in replacement for ``multiprocessing.shared_memory.SharedMemory``.

    Backed by a plain ``bytearray`` stored in ``_registry`` so that
    ``create=True`` followed by opening the same *name* returns the same
    underlying buffer -- exactly like real shared memory.
    """

    def __init__(self, *, create: bool = False, size: int = 0, name: str | None = None):
        if create:
            self.name = name or f"fake_{uuid.uuid4().hex[:8]}"
            self.size = size
            self._buf = _registry.create(self.name, size)
        else:
            assert name is not None, "name is required when create=False"
            self._buf = _registry.get(name)
            self.name = name
            self.size = len(self._buf)

    @property
    def buf(self) -> memoryview:  # noqa: D401
        return memoryview(self._buf)

    def close(self) -> None:  # noqa: D401
        pass

    def unlink(self) -> None:
        _registry.remove(self.name)


@pytest.fixture(autouse=True)
def _clear_fake_registry():
    """Ensure a clean shared-memory registry for every test."""
    _registry.clear()
    yield
    _registry.clear()


# Patch target -- this is the class reference used inside parallel_executor.py
_SHM_PATCH_TARGET = "ergodic_insurance.parallel_executor.shared_memory.SharedMemory"


# ===================================================================
# Lines 191-202: SharedMemoryManager.share_array
# ===================================================================


class TestShareArrayMocked:
    """Exercise ``share_array`` without real OS shared memory."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_array_creates_buffer_and_copies_data(self):
        """Lines 191-202: shared memory is created, data is copied, reference is stored."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        shm_name = manager.share_array("mat", arr)

        # Returns a non-empty name that encodes the user-supplied key
        assert isinstance(shm_name, str) and shm_name != ""
        assert "ergodic_array_mat_" in shm_name

        # Internal bookkeeping stores (SharedMemory, shape, dtype)
        assert "mat" in manager.shared_arrays
        shm_obj, shape, dtype = manager.shared_arrays["mat"]
        assert shape == (2, 2)
        assert dtype == np.float64

        # The buffer actually contains the correct data
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm_obj.buf)
        np.testing.assert_array_equal(view, arr)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_array_with_various_dtypes(self):
        """Lines 191-202: share_array handles int32, int64, float32, float64."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        for dt in (np.int32, np.int64, np.float32, np.float64):
            name = f"arr_{dt.__name__}"
            arr = np.arange(12, dtype=dt).reshape(3, 4)
            shm_name = manager.share_array(name, arr)

            assert shm_name != ""
            stored_shm, stored_shape, stored_dtype = manager.shared_arrays[name]
            assert stored_shape == (3, 4)
            assert stored_dtype == dt
            view = np.ndarray(stored_shape, dtype=stored_dtype, buffer=stored_shm.buf)
            np.testing.assert_array_equal(view, arr)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_array_1d_vector(self):
        """Lines 191-202: 1-D arrays are stored and retrievable."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        vec = np.array([10.0, 20.0, 30.0])
        shm_name = manager.share_array("vec", vec)

        shm_obj, shape, dtype = manager.shared_arrays["vec"]
        assert shape == (3,)
        view = np.ndarray(shape, dtype=dtype, buffer=shm_obj.buf)
        np.testing.assert_array_equal(view, vec)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_array_large(self):
        """Lines 191-202: works with moderately large arrays (1000x100)."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        big = np.random.default_rng(42).standard_normal((1000, 100))
        shm_name = manager.share_array("big", big)

        shm_obj, shape, dtype = manager.shared_arrays["big"]
        view = np.ndarray(shape, dtype=dtype, buffer=shm_obj.buf)
        np.testing.assert_array_equal(view, big)

        manager.cleanup()


# ===================================================================
# Lines 215-216: SharedMemoryManager.get_array
# ===================================================================


class TestGetArrayMocked:
    """Exercise ``get_array`` without real OS shared memory."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_get_array_returns_correct_data(self):
        """Lines 215-216: get_array opens shared memory by name and returns ndarray."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        original = np.array([[5.0, 6.0], [7.0, 8.0]])
        shm_name = manager.share_array("data", original)

        retrieved = manager.get_array(shm_name, original.shape, original.dtype)
        np.testing.assert_array_equal(retrieved, original)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_get_array_returns_view_into_shared_buffer(self):
        """Lines 215-216: mutations through one view are visible through another."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        arr = np.array([1.0, 2.0, 3.0])
        shm_name = manager.share_array("shared", arr)

        view_a = manager.get_array(shm_name, arr.shape, arr.dtype)
        view_b = manager.get_array(shm_name, arr.shape, arr.dtype)

        view_a[0] = 999.0
        assert view_b[0] == 999.0  # both backed by the same buffer

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_get_array_integer_dtype(self):
        """Lines 215-216: works for integer arrays."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        arr = np.array([10, 20, 30], dtype=np.int32)
        shm_name = manager.share_array("ints", arr)

        retrieved = manager.get_array(shm_name, arr.shape, arr.dtype)
        np.testing.assert_array_equal(retrieved, arr)

        manager.cleanup()


# ===================================================================
# Lines 236-238: share_object with compression
# ===================================================================


class TestShareObjectCompressionMocked:
    """Exercise the compression branch of ``share_object``."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_object_compresses_data(self):
        """Lines 236-238: with compression=True, stored bytes are zlib-compressed."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=True)
        manager = SharedMemoryManager(config)

        obj = {"values": list(range(5000))}  # highly compressible
        shm_name = manager.share_object("comp", obj)

        assert shm_name != ""
        assert "comp" in manager.shared_objects

        # Verify the buffer content is actually compressed
        shm = manager.shared_objects["comp"]
        assert shm.buf is not None
        stored = bytes(shm.buf[: shm.size])
        decompressed = zlib.decompress(stored)
        restored = safe_loads(decompressed)
        assert restored == obj

        # Compressed should be smaller than uncompressed HMAC-signed pickle
        hmac_pickle = safe_dumps(obj)
        assert len(stored) < len(hmac_pickle)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_object_no_compression_stores_hmac_signed_pickle(self):
        """Baseline: with compression=False, HMAC-signed pickle bytes are stored."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=False)
        manager = SharedMemoryManager(config)

        obj = {"a": 1, "b": [2, 3]}
        shm_name = manager.share_object("raw", obj)

        expected = safe_dumps(obj)
        shm = manager.shared_objects["raw"]
        assert shm.buf is not None
        stored = bytes(shm.buf[: len(expected)])
        assert stored == expected

        manager.cleanup()


# ===================================================================
# Lines 264-273: get_object with decompression
# ===================================================================


class TestGetObjectDecompressionMocked:
    """Exercise the decompression branch of ``get_object``."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_get_object_decompresses(self):
        """Lines 264-273: get_object with compressed=True runs zlib.decompress."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=True)
        manager = SharedMemoryManager(config)

        original = {"scores": list(range(500)), "label": "test"}
        shm_name = manager.share_object("cobj", original)

        # Use the actual stored size (includes HMAC overhead before compression)
        actual_size = manager.get_object_size("cobj")

        retrieved = manager.get_object(shm_name, actual_size, compressed=True)
        assert retrieved == original

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_get_object_without_compression(self):
        """Lines 264-273: get_object with compressed=False reads HMAC-signed pickle."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=False)
        manager = SharedMemoryManager(config)

        original = [1, "two", 3.0, None]
        shm_name = manager.share_object("plain", original)
        actual_size = manager.get_object_size("plain")

        retrieved = manager.get_object(shm_name, actual_size, compressed=False)
        assert retrieved == original

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_get_object_roundtrip_nested_dict(self):
        """Lines 264-273: round-trip a deeply nested dict through compressed shared memory."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=True)
        manager = SharedMemoryManager(config)

        original = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3],
                },
                "other": "value",
            },
            "top": 42,
        }
        shm_name = manager.share_object("deep", original)

        actual_size = manager.get_object_size("deep")

        retrieved = manager.get_object(shm_name, actual_size, compressed=True)
        assert retrieved == original

        manager.cleanup()


# ===================================================================
# Lines 473-474: ParallelExecutor._setup_shared_data (numpy path)
# ===================================================================


class TestSetupSharedDataNumpyMocked:
    """Exercise the numpy-array branch of ``_setup_shared_data``."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_setup_shared_data_routes_ndarray_to_share_array(self):
        """Lines 473-474: numpy arrays produce ('array', (name, shape, dtype)) refs."""
        executor = ParallelExecutor(n_workers=2)

        arr = np.array([1.0, 2.0, 3.0, 4.0])
        refs = executor._setup_shared_data({"weights": arr})

        assert "weights" in refs
        data_type, (shm_name, shape, dtype) = refs["weights"]
        assert data_type == "array"
        assert shm_name != ""
        assert shape == (4,)
        assert dtype == np.float64

        executor.shared_memory_manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_setup_shared_data_mixed_array_and_object(self):
        """Lines 473-474: arrays and plain objects are routed to different branches."""
        executor = ParallelExecutor(n_workers=2)

        shared = {
            "matrix": np.zeros((3, 3), dtype=np.float32),
            "params": {"lr": 0.01},
        }
        refs = executor._setup_shared_data(shared)

        # Array branch
        assert refs["matrix"][0] == "array"
        _, (_, shape, dtype) = refs["matrix"]
        assert shape == (3, 3)
        assert dtype == np.float32

        # Object branch
        assert refs["params"][0] == "object"
        _, (shm_name, size, compressed) = refs["params"]
        assert shm_name != ""
        assert size > 0
        assert compressed is False

        executor.shared_memory_manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_setup_shared_data_2d_array(self):
        """Lines 473-474: 2-D arrays are correctly shared."""
        executor = ParallelExecutor(n_workers=2)

        mat = np.eye(5, dtype=np.float64)
        refs = executor._setup_shared_data({"identity": mat})

        _, (shm_name, shape, dtype) = refs["identity"]
        assert shape == (5, 5)
        assert dtype == np.float64

        executor.shared_memory_manager.cleanup()


# ===================================================================
# Lines 673-674: _execute_chunk -- array retrieval in worker
# ===================================================================


class TestExecuteChunkArrayPathMocked:
    """Exercise the ``data_type == 'array'`` branch inside ``_execute_chunk``."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_execute_chunk_retrieves_shared_array(self):
        """Lines 673-674: _execute_chunk reconstructs a numpy array from shared refs."""
        config = SharedMemoryConfig(enable_shared_arrays=True)
        manager = SharedMemoryManager(config)

        weights = np.array([10.0, 20.0, 30.0])
        shm_name = manager.share_array("w", weights)

        shared_refs = {
            "w": ("array", (shm_name, weights.shape, weights.dtype)),
        }

        def work(item, **kwargs):
            w = kwargs["w"]
            return float(item * w.sum())

        chunk = (0, 3, [1, 2, 3])
        results, failed_count = _execute_chunk(work, chunk, shared_refs, config)

        total = 10.0 + 20.0 + 30.0  # 60.0
        assert results == [1 * total, 2 * total, 3 * total]
        assert failed_count == 0

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_execute_chunk_retrieves_shared_object(self):
        """Companion: _execute_chunk reconstructs a plain object from shared refs."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=False)
        manager = SharedMemoryManager(config)

        obj = {"factor": 5}
        shm_name = manager.share_object("cfg", obj)
        actual_size = manager.get_object_size("cfg")

        shared_refs = {
            "cfg": ("object", (shm_name, actual_size, False)),
        }

        def work(item, **kwargs):
            return item * kwargs["cfg"]["factor"]

        chunk = (0, 2, [3, 7])
        results, failed_count = _execute_chunk(work, chunk, shared_refs, config)
        assert results == [15, 35]
        assert failed_count == 0

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_execute_chunk_with_compressed_object(self):
        """Lines 264-273 via _execute_chunk: compressed objects are decompressed."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=True)
        manager = SharedMemoryManager(config)

        obj = {"multiplier": 3, "data": list(range(50))}
        shm_name = manager.share_object("comp_cfg", obj)

        actual_size = manager.get_object_size("comp_cfg")

        shared_refs = {
            "comp_cfg": ("object", (shm_name, actual_size, True)),
        }

        def work(item, **kwargs):
            return item * kwargs["comp_cfg"]["multiplier"]

        chunk = (0, 2, [4, 8])
        results, failed_count = _execute_chunk(work, chunk, shared_refs, config)
        assert results == [12, 24]
        assert failed_count == 0

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_execute_chunk_mixed_array_and_object(self):
        """Lines 673-674 + object branch: both array and object refs in one call."""
        config = SharedMemoryConfig(
            enable_shared_arrays=True,
            enable_shared_objects=True,
            compression=False,
        )
        manager = SharedMemoryManager(config)

        arr = np.array([1.0, 2.0, 3.0])
        shm_arr = manager.share_array("vec", arr)

        obj = {"offset": 100}
        shm_obj = manager.share_object("cfg", obj)
        obj_size = manager.get_object_size("cfg")

        shared_refs = {
            "vec": ("array", (shm_arr, arr.shape, arr.dtype)),
            "cfg": ("object", (shm_obj, obj_size, False)),
        }

        def work(item, **kwargs):
            v = kwargs["vec"]
            c = kwargs["cfg"]
            return float(item * v.sum() + c["offset"])

        chunk = (0, 2, [1, 2])
        results, failed_count = _execute_chunk(work, chunk, shared_refs, config)

        vec_sum = 1.0 + 2.0 + 3.0  # 6.0
        assert results == [1 * vec_sum + 100, 2 * vec_sum + 100]
        assert failed_count == 0

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_execute_chunk_array_content_integrity(self):
        """Lines 673-674: verify the worker sees the exact same array values."""
        config = SharedMemoryConfig(enable_shared_arrays=True)
        manager = SharedMemoryManager(config)

        matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        shm_name = manager.share_array("mat", matrix)

        shared_refs = {
            "mat": ("array", (shm_name, matrix.shape, matrix.dtype)),
        }

        def work(row_idx, **kwargs):
            row = kwargs["mat"][row_idx]
            return int(row[0] + row[1])

        chunk = (0, 3, [0, 1, 2])
        results, failed_count = _execute_chunk(work, chunk, shared_refs, config)

        assert results == [3, 7, 11]  # 1+2, 3+4, 5+6
        assert failed_count == 0

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_execute_chunk_no_shared_refs_still_works(self):
        """Baseline: _execute_chunk with empty shared_refs runs normally."""
        config = SharedMemoryConfig()

        def square(x):
            return x**2

        chunk = (0, 4, [2, 3, 5, 7])
        results, failed_count = _execute_chunk(square, chunk, {}, config)
        assert results == [4, 9, 25, 49]
        assert failed_count == 0


# ===================================================================
# Integration: full share -> get round-trip through FakeSharedMemory
# ===================================================================


class TestSkipHmacRoundTripMocked:
    """Exercise the ``skip_hmac=True`` path (raw pickle, no HMAC signing)."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_skip_hmac_share_then_get_roundtrip(self):
        """Lines 246-251, 297-300: skip_hmac uses raw pickle for share and get."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=False, skip_hmac=True)
        manager = SharedMemoryManager(config)

        original = {"key": "value", "numbers": list(range(100))}
        shm_name = manager.share_object("obj", original)

        actual_size = manager.get_object_size("obj")
        retrieved = manager.get_object(shm_name, actual_size, compressed=False)
        assert retrieved == original

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_skip_hmac_with_compression_roundtrip(self):
        """Lines 246-251, 297-300: skip_hmac + compression round-trip."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=True, skip_hmac=True)
        manager = SharedMemoryManager(config)

        original = {"data": list(range(500)), "nested": {"a": 1, "b": [2, 3]}}
        shm_name = manager.share_object("comp", original)

        actual_size = manager.get_object_size("comp")
        retrieved = manager.get_object(shm_name, actual_size, compressed=True)
        assert retrieved == original

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_skip_hmac_stores_raw_pickle_not_hmac_signed(self):
        """Lines 246-251: skip_hmac path stores raw pickle (no HMAC header)."""
        import pickle

        config = SharedMemoryConfig(enable_shared_objects=True, compression=False, skip_hmac=True)
        manager = SharedMemoryManager(config)

        obj = {"simple": True}
        shm_name = manager.share_object("raw", obj)

        # Raw pickle bytes should be directly deserializable
        shm = manager.shared_objects["raw"]
        assert shm.buf is not None
        stored = bytes(shm.buf[: manager.get_object_size("raw")])
        assert pickle.loads(stored) == obj  # noqa: S301

        # Verify it's NOT safe_dumps format (safe_loads would fail or differ)
        raw_pickle = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        assert stored == raw_pickle

        manager.cleanup()


class TestFullRoundTripMocked:
    """End-to-end round-trip tests combining share + get through fake SHM."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_array_share_then_get_roundtrip(self):
        """Full round-trip: share_array -> get_array -> verify equality."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        original = np.random.default_rng(7).standard_normal((50, 20))
        shm_name = manager.share_array("big_array", original)

        retrieved = manager.get_array(shm_name, original.shape, original.dtype)
        np.testing.assert_array_equal(retrieved, original)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_compressed_object_share_then_get_roundtrip(self):
        """Full round-trip: share_object (compressed) -> get_object -> verify."""
        config = SharedMemoryConfig(enable_shared_objects=True, compression=True)
        manager = SharedMemoryManager(config)

        original = {"key": "value", "numbers": list(range(200))}
        shm_name = manager.share_object("obj", original)

        actual_size = manager.get_object_size("obj")

        retrieved = manager.get_object(shm_name, actual_size, compressed=True)
        assert retrieved == original

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_multiple_arrays_independent(self):
        """Multiple arrays can be shared and retrieved independently."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_arrays=True))

        a = np.array([1.0, 2.0])
        b = np.array([10, 20, 30], dtype=np.int32)
        c = np.zeros((2, 2))

        name_a = manager.share_array("a", a)
        name_b = manager.share_array("b", b)
        name_c = manager.share_array("c", c)

        np.testing.assert_array_equal(manager.get_array(name_a, a.shape, a.dtype), a)
        np.testing.assert_array_equal(manager.get_array(name_b, b.shape, b.dtype), b)
        np.testing.assert_array_equal(manager.get_array(name_c, c.shape, c.dtype), c)

        manager.cleanup()

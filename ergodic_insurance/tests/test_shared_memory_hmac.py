"""Security tests for HMAC-verified shared memory serialization (#612).

Verifies that SharedMemoryManager uses safe_pickle (HMAC-signed serialization)
instead of raw pickle, and that tampered shared memory payloads are rejected.

Uses a FakeSharedMemory backend so tests run on all platforms without
requiring OS-level shared memory.
"""

import pickle
from unittest.mock import patch
import uuid

import numpy as np
import pytest

from ergodic_insurance.parallel_executor import (
    SharedMemoryConfig,
    SharedMemoryManager,
)
from ergodic_insurance.safe_pickle import _SIGNATURE_LENGTH, safe_dumps

# ---------------------------------------------------------------------------
# Fake shared-memory infrastructure (same pattern as test_coverage_gaps)
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
    """Drop-in replacement for multiprocessing.shared_memory.SharedMemory."""

    def __init__(self, *, create: bool = False, size: int = 0, name: str | None = None):
        if create:
            self.name = name or f"fake_{uuid.uuid4().hex[:8]}"
            self.size = size
            self._buf = _registry.create(self.name, size)
        else:
            assert name is not None
            self._buf = _registry.get(name)
            self.name = name
            self.size = len(self._buf)

    @property
    def buf(self) -> memoryview:
        return memoryview(self._buf)

    def close(self) -> None:
        pass

    def unlink(self) -> None:
        _registry.remove(self.name)


@pytest.fixture(autouse=True)
def _clear_fake_registry():
    _registry.clear()
    yield
    _registry.clear()


_SHM_PATCH_TARGET = "ergodic_insurance.parallel_executor.shared_memory.SharedMemory"


# ===================================================================
# Security tests: HMAC verification on shared memory objects
# ===================================================================


class TestSharedMemoryHmacSerialization:
    """Verify SharedMemoryManager uses HMAC-signed serialization."""

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_share_object_produces_hmac_signed_bytes(self):
        """share_object output must start with a 32-byte HMAC signature."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_objects=True))

        test_obj = {"key": "value", "nums": [1, 2, 3]}
        shm_name = manager.share_object("test", test_obj)

        # Retrieve the raw bytes from shared memory
        size = manager.get_object_size("test")
        shm = FakeSharedMemory(name=shm_name)
        raw_data = bytes(shm.buf[:size])

        # Must be longer than just the pickle payload (HMAC adds 32 bytes)
        raw_pickle = pickle.dumps(test_obj, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(raw_data) == len(raw_pickle) + _SIGNATURE_LENGTH

        # First 32 bytes should be the HMAC signature, not pickle data
        # Pickle protocol 2+ starts with \x80; the HMAC signature is random
        assert raw_data[:_SIGNATURE_LENGTH] != raw_pickle[:_SIGNATURE_LENGTH]

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_roundtrip_object_via_safe_pickle(self):
        """Objects can be shared and retrieved correctly with HMAC verification."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_objects=True))

        test_obj = {"config": {"lr": 0.01}, "data": list(range(100))}
        shm_name = manager.share_object("cfg", test_obj)
        size = manager.get_object_size("cfg")

        retrieved = manager.get_object(shm_name, size)
        assert retrieved == test_obj

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_tampered_payload_rejected(self):
        """Corrupting shared memory bytes must raise ValueError (HMAC mismatch)."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_objects=True))

        test_obj = {"secret": "data"}
        shm_name = manager.share_object("target", test_obj)
        size = manager.get_object_size("target")

        # Tamper with the payload (byte after the HMAC signature)
        shm = FakeSharedMemory(name=shm_name)
        tamper_idx = _SIGNATURE_LENGTH + 1
        if tamper_idx < size:
            original = shm.buf[tamper_idx]
            shm.buf[tamper_idx] = (original + 1) % 256

        with pytest.raises(ValueError, match="HMAC mismatch"):
            manager.get_object(shm_name, size)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_tampered_signature_rejected(self):
        """Corrupting the HMAC signature must raise ValueError."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_objects=True))

        test_obj = [1, 2, 3]
        shm_name = manager.share_object("sig_test", test_obj)
        size = manager.get_object_size("sig_test")

        # Flip a bit in the HMAC signature
        shm = FakeSharedMemory(name=shm_name)
        shm.buf[0] = (shm.buf[0] + 1) % 256

        with pytest.raises(ValueError, match="HMAC mismatch"):
            manager.get_object(shm_name, size)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_raw_pickle_payload_rejected(self):
        """A raw pickle payload (no HMAC) must be rejected by get_object."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_objects=True))

        # Manually create shared memory with raw pickle (no HMAC)
        test_obj = {"attack": True}
        raw_data = pickle.dumps(test_obj, protocol=pickle.HIGHEST_PROTOCOL)

        fake_name = f"ergodic_obj_raw_{uuid.uuid4().hex[:12]}"
        shm = FakeSharedMemory(create=True, size=len(raw_data), name=fake_name)
        shm.buf[: len(raw_data)] = raw_data

        with pytest.raises(ValueError, match="HMAC"):
            manager.get_object(fake_name, len(raw_data))

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_compressed_roundtrip_with_hmac(self):
        """Compressed objects must also use HMAC-signed serialization."""
        manager = SharedMemoryManager(
            SharedMemoryConfig(enable_shared_objects=True, compression=True)
        )

        test_obj = {"data": list(range(1000))}
        shm_name = manager.share_object("compressed", test_obj)
        size = manager.get_object_size("compressed")

        retrieved = manager.get_object(shm_name, size, compressed=True)
        assert retrieved == test_obj

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_compressed_tampered_payload_rejected(self):
        """Tampering with compressed HMAC-signed data must fail."""
        manager = SharedMemoryManager(
            SharedMemoryConfig(enable_shared_objects=True, compression=True)
        )

        test_obj = {"data": list(range(500))}
        shm_name = manager.share_object("comp_tamper", test_obj)
        size = manager.get_object_size("comp_tamper")

        # Tamper with compressed bytes
        shm = FakeSharedMemory(name=shm_name)
        mid = size // 2
        if mid < size:
            shm.buf[mid] = (shm.buf[mid] + 1) % 256

        # Should fail on either decompression or HMAC verification
        with pytest.raises((ValueError, Exception)):
            manager.get_object(shm_name, size, compressed=True)

        manager.cleanup()

    @patch(_SHM_PATCH_TARGET, FakeSharedMemory)
    def test_various_object_types_roundtrip(self):
        """HMAC-signed serialization works for diverse Python types."""
        manager = SharedMemoryManager(SharedMemoryConfig(enable_shared_objects=True))

        test_cases = [
            ("int", 42),
            ("float", 3.14159),
            ("string", "hello world"),
            ("list", [1, "two", 3.0]),
            ("dict", {"nested": {"a": 1}}),
            ("tuple", (1, 2, 3)),
            ("none", None),
            ("bool", True),
            ("set_as_list", sorted([1, 2, 3])),
        ]

        for name, obj in test_cases:
            shm_name = manager.share_object(name, obj)
            size = manager.get_object_size(name)
            retrieved = manager.get_object(shm_name, size)
            assert retrieved == obj, f"Roundtrip failed for {name}: {obj!r}"

        manager.cleanup()

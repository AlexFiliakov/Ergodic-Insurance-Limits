"""Additional tests for safe_pickle to cover missing lines.

Targets missing coverage lines:
50-53 (_get_or_create_hmac_key creates new key),
101 (safe_load HMAC mismatch),
124-127 (safe_dumps function),
143-158 (safe_loads function),
175-176 (deterministic_hash function)
"""

import io
import os
from pathlib import Path
import pickle
import stat
import tempfile
from unittest import mock

import pytest

from ergodic_insurance.safe_pickle import (
    _get_or_create_hmac_key,
    _secure_mkdir,
    _secure_write_key,
    deterministic_hash,
    safe_dump,
    safe_dumps,
    safe_load,
    safe_loads,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_key_dir():
    """Create a temporary directory for HMAC key storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fresh_key_dir():
    """Create a temporary directory without an existing key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_dir = Path(tmpdir) / "fresh_keys"
        # Ensure directory does NOT exist yet
        assert not key_dir.exists()
        yield key_dir


# ---------------------------------------------------------------------------
# _get_or_create_hmac_key (lines 50-53)
# ---------------------------------------------------------------------------


class TestGetOrCreateHmacKey:
    """Test HMAC key creation and retrieval."""

    def test_create_new_key(self, fresh_key_dir):
        """When key file does not exist, create directory and key (lines 50-53)."""
        assert not fresh_key_dir.exists()

        key = _get_or_create_hmac_key(fresh_key_dir)

        # Directory should now exist
        assert fresh_key_dir.exists()
        # Key should be 32 bytes
        assert len(key) == 32
        assert isinstance(key, bytes)

        # Key file should exist
        key_file = fresh_key_dir / ".pickle_hmac_key"
        assert key_file.exists()
        assert key_file.read_bytes() == key

    def test_retrieve_existing_key(self, temp_key_dir):
        """When key file exists, return the same key."""
        key1 = _get_or_create_hmac_key(temp_key_dir)
        key2 = _get_or_create_hmac_key(temp_key_dir)
        assert key1 == key2

    def test_key_is_random(self, fresh_key_dir):
        """Each fresh key directory should produce a unique key."""
        key1 = _get_or_create_hmac_key(fresh_key_dir)

        # Create a second fresh directory
        with tempfile.TemporaryDirectory() as tmpdir2:
            key2 = _get_or_create_hmac_key(Path(tmpdir2) / "other_keys")

        # Keys from different directories should be different (with overwhelming probability)
        assert key1 != key2


# ---------------------------------------------------------------------------
# safe_load HMAC mismatch (line 101)
# ---------------------------------------------------------------------------


class TestSafeLoadHmacMismatch:
    """Test safe_load rejects tampered data."""

    def test_tampered_data_raises(self, temp_key_dir):
        """Tampered pickle data should raise ValueError (line 101)."""
        # Write valid data
        obj = {"hello": "world", "number": 42}
        buf = io.BytesIO()
        safe_dump(obj, buf, key_dir=temp_key_dir)

        # Get the raw bytes and tamper with the payload
        raw = buf.getvalue()
        signature = raw[:32]
        payload = raw[32:]

        # Flip a byte in the payload
        tampered_payload = bytearray(payload)
        tampered_payload[0] ^= 0xFF
        tampered_data = signature + bytes(tampered_payload)

        buf_tampered = io.BytesIO(tampered_data)
        with pytest.raises(ValueError, match="HMAC mismatch"):
            safe_load(buf_tampered, key_dir=temp_key_dir)

    def test_wrong_key_raises(self, temp_key_dir):
        """Data signed with a different key should raise ValueError."""
        obj = [1, 2, 3]
        buf = io.BytesIO()
        safe_dump(obj, buf, key_dir=temp_key_dir)

        # Load with a different key directory
        with tempfile.TemporaryDirectory() as other_dir:
            other_key_dir = Path(other_dir) / "other"
            _get_or_create_hmac_key(other_key_dir)

            buf.seek(0)
            with pytest.raises(ValueError, match="HMAC mismatch"):
                safe_load(buf, key_dir=other_key_dir)

    def test_too_short_data_raises(self, temp_key_dir):
        """Data shorter than HMAC length should raise ValueError."""
        buf = io.BytesIO(b"short")
        with pytest.raises(ValueError, match="too short"):
            safe_load(buf, key_dir=temp_key_dir)


# ---------------------------------------------------------------------------
# safe_dumps / safe_loads (lines 124-127, 143-158)
# ---------------------------------------------------------------------------


class TestSafeDumpsLoads:
    """Test the bytes-based safe_dumps and safe_loads functions."""

    def test_safe_dumps_roundtrip(self, temp_key_dir):
        """safe_dumps/safe_loads roundtrip works correctly (lines 124-127, 143-158)."""
        obj = {"key": "value", "list": [1, 2, 3], "nested": {"a": True}}
        data = safe_dumps(obj, key_dir=temp_key_dir)

        assert isinstance(data, bytes)
        # First 32 bytes are HMAC signature
        assert len(data) > 32

        loaded = safe_loads(data, key_dir=temp_key_dir)
        assert loaded == obj

    def test_safe_dumps_various_types(self, temp_key_dir):
        """safe_dumps handles various Python types."""
        test_objects = [
            42,
            3.14,
            "hello",
            [1, 2, 3],
            {"key": "value"},
            (1, "a", None),
            None,
            True,
            set([1, 2, 3]),
        ]
        for obj in test_objects:
            data = safe_dumps(obj, key_dir=temp_key_dir)
            loaded = safe_loads(data, key_dir=temp_key_dir)
            assert loaded == obj

    def test_safe_loads_tampered_raises(self, temp_key_dir):
        """safe_loads rejects tampered bytes data (lines 152-156)."""
        obj = "test data"
        data = safe_dumps(obj, key_dir=temp_key_dir)

        # Tamper with payload portion
        tampered = bytearray(data)
        tampered[-1] ^= 0xFF
        with pytest.raises(ValueError, match="HMAC mismatch"):
            safe_loads(bytes(tampered), key_dir=temp_key_dir)

    def test_safe_loads_too_short_raises(self, temp_key_dir):
        """safe_loads rejects data shorter than HMAC length (line 143-144)."""
        with pytest.raises(ValueError, match="too short"):
            safe_loads(b"short", key_dir=temp_key_dir)

    def test_safe_loads_wrong_key_raises(self, temp_key_dir):
        """safe_loads with wrong key directory raises ValueError."""
        obj = {"data": 123}
        data = safe_dumps(obj, key_dir=temp_key_dir)

        with tempfile.TemporaryDirectory() as other_dir:
            other_key_dir = Path(other_dir) / "wrong_key"
            _get_or_create_hmac_key(other_key_dir)
            with pytest.raises(ValueError, match="HMAC mismatch"):
                safe_loads(data, key_dir=other_key_dir)

    def test_safe_dumps_protocol(self, temp_key_dir):
        """safe_dumps respects the protocol argument."""
        obj = {"test": True}
        data = safe_dumps(obj, protocol=2, key_dir=temp_key_dir)
        loaded = safe_loads(data, key_dir=temp_key_dir)
        assert loaded == obj


# ---------------------------------------------------------------------------
# deterministic_hash (lines 175-176)
# ---------------------------------------------------------------------------


class TestDeterministicHash:
    """Test the deterministic_hash function."""

    def test_basic_hash(self):
        """deterministic_hash returns a hex string (lines 175-176)."""
        result = deterministic_hash("hello", "world")
        assert isinstance(result, str)
        assert len(result) == 16  # default length
        # Should be hex characters
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        """Same inputs always produce the same hash."""
        hash1 = deterministic_hash("a", "b", "c")
        hash2 = deterministic_hash("a", "b", "c")
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        """Different inputs produce different hashes."""
        hash1 = deterministic_hash("hello")
        hash2 = deterministic_hash("world")
        assert hash1 != hash2

    def test_custom_length(self):
        """deterministic_hash respects length argument."""
        result = deterministic_hash("test", length=8)
        assert len(result) == 8

        result_long = deterministic_hash("test", length=64)
        assert len(result_long) == 64

    def test_single_arg(self):
        """deterministic_hash works with a single argument."""
        result = deterministic_hash("single")
        assert isinstance(result, str)
        assert len(result) == 16

    def test_numeric_args(self):
        """deterministic_hash converts non-string args via str()."""
        result = deterministic_hash("key", "123", "456")
        assert isinstance(result, str)
        assert len(result) == 16

    def test_empty_args(self):
        """deterministic_hash with no args still works."""
        result = deterministic_hash()
        assert isinstance(result, str)
        assert len(result) == 16


# ---------------------------------------------------------------------------
# Secure permissions (issue #613)
# ---------------------------------------------------------------------------


class TestSecurePermissions:
    """Test that HMAC key files and directories have restrictive permissions."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix permissions not available on Windows")
    def test_key_file_permissions_unix(self, fresh_key_dir):
        """Key file should have 0600 permissions on Unix."""
        _get_or_create_hmac_key(fresh_key_dir)
        key_file = fresh_key_dir / ".pickle_hmac_key"
        mode = stat.S_IMODE(key_file.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    @pytest.mark.skipif(os.name == "nt", reason="Unix permissions not available on Windows")
    def test_key_dir_permissions_unix(self, fresh_key_dir):
        """Key directory should have 0700 permissions on Unix."""
        _get_or_create_hmac_key(fresh_key_dir)
        mode = stat.S_IMODE(fresh_key_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_existing_key_not_broken(self, temp_key_dir):
        """Pre-existing keys with any permissions still load correctly."""
        # Create key the normal way first
        key1 = _get_or_create_hmac_key(temp_key_dir)
        # Reading it again should return the same key regardless of permissions
        key2 = _get_or_create_hmac_key(temp_key_dir)
        assert key1 == key2

    def test_race_condition_handling(self, fresh_key_dir):
        """FileExistsError from O_EXCL is handled gracefully (TOCTOU race)."""
        # Pre-create directory so _secure_mkdir succeeds
        fresh_key_dir.mkdir(parents=True, exist_ok=True)
        key_path = fresh_key_dir / ".pickle_hmac_key"
        # Write a key manually to simulate another process winning the race
        existing_key = b"\xab" * 32
        key_path.write_bytes(existing_key)

        # Patch _secure_write_key to raise FileExistsError (simulates O_EXCL race)
        with mock.patch(
            "ergodic_insurance.safe_pickle._secure_write_key",
            side_effect=FileExistsError("File exists"),
        ):
            result = _get_or_create_hmac_key(fresh_key_dir)

        # Should fall back to reading the existing key
        assert result == existing_key

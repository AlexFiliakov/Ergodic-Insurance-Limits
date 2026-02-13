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
import secrets
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
        fresh_key_dir.mkdir(parents=True, exist_ok=True)
        key_path = fresh_key_dir / ".pickle_hmac_key"
        existing_key = b"\xab" * 32

        # Simulate: first read_bytes raises FileNotFoundError (key doesn't exist yet),
        # then _secure_write_key raises FileExistsError (another process won the race),
        # then read_bytes succeeds on the retry loop.
        call_count = {"n": 0}

        def mock_read_bytes(self):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FileNotFoundError("not yet")
            # On second call (retry), the file exists
            return existing_key

        with mock.patch.object(Path, "read_bytes", mock_read_bytes):
            with mock.patch(
                "ergodic_insurance.safe_pickle._secure_write_key",
                side_effect=FileExistsError("File exists"),
            ):
                result = _get_or_create_hmac_key(fresh_key_dir)

        assert result == existing_key


# ---------------------------------------------------------------------------
# TOCTOU regression tests (issue #798)
# ---------------------------------------------------------------------------


class TestTOCTOURegressions:
    """Regression tests for TOCTOU race conditions in key management."""

    def test_no_exists_check_in_read_path(self, fresh_key_dir):
        """Key read must use try/except, not exists() + read_bytes() (issue #798)."""
        import inspect

        source = inspect.getsource(_get_or_create_hmac_key)
        # The fixed code should NOT call .exists() on the key path
        assert "key_path.exists()" not in source

    def test_secure_write_key_uses_o_excl_on_all_platforms(self, fresh_key_dir):
        """_secure_write_key must use O_EXCL on all platforms (issue #798)."""
        fresh_key_dir.mkdir(parents=True, exist_ok=True)
        key_path = fresh_key_dir / ".pickle_hmac_key"
        key = secrets.token_bytes(32)

        # First write succeeds
        _secure_write_key(key_path, key)
        assert key_path.read_bytes() == key

        # Second write to same path MUST raise FileExistsError (O_EXCL)
        with pytest.raises(FileExistsError):
            _secure_write_key(key_path, secrets.token_bytes(32))

    def test_concurrent_deletion_retries(self, fresh_key_dir):
        """If key is deleted between read and write, retry loop recovers."""
        fresh_key_dir.mkdir(parents=True, exist_ok=True)
        key_path = fresh_key_dir / ".pickle_hmac_key"
        expected_key = b"\xcd" * 32

        # Simulate: attempts 1-2 both fail to read AND fail to write,
        # attempt 3 succeeds to read.
        read_count = {"n": 0}

        def mock_read_bytes(self):
            read_count["n"] += 1
            if read_count["n"] <= 2:
                raise FileNotFoundError("gone")
            return expected_key

        write_effects = [
            FileExistsError("race 1"),
            FileExistsError("race 2"),
        ]

        with mock.patch.object(Path, "read_bytes", mock_read_bytes):
            with mock.patch(
                "ergodic_insurance.safe_pickle._secure_write_key",
                side_effect=write_effects,
            ):
                result = _get_or_create_hmac_key(fresh_key_dir)

        assert result == expected_key

    def test_write_key_no_silent_overwrite(self, fresh_key_dir):
        """On Windows, write_bytes() must NOT be used (would silently overwrite)."""
        import inspect

        source = inspect.getsource(_secure_write_key)
        assert (
            "write_bytes" not in source
        ), "_secure_write_key must not use write_bytes — it silently overwrites"

    def test_write_key_includes_o_binary_for_windows(self):
        """O_BINARY flag must be included for correct binary writes on Windows."""
        import inspect

        source = inspect.getsource(_secure_write_key)
        assert "O_BINARY" in source


# ---------------------------------------------------------------------------
# RestrictedUnpickler defense-in-depth (issue #984)
# ---------------------------------------------------------------------------


class TestIsModuleAllowed:
    """Test the _is_module_allowed helper function."""

    def test_exact_match(self):
        """Exact module name in allowlist passes."""
        from ergodic_insurance.safe_pickle import _is_module_allowed

        assert _is_module_allowed("numpy") is True

    def test_submodule_match(self):
        """Submodule of an allowed parent passes."""
        from ergodic_insurance.safe_pickle import _is_module_allowed

        assert _is_module_allowed("numpy.core.multiarray") is True

    def test_no_false_prefix_match(self):
        """Module whose name merely starts with an allowed name is rejected."""
        from ergodic_insurance.safe_pickle import _is_module_allowed

        assert _is_module_allowed("numpytools") is False
        assert _is_module_allowed("numpytools.evil") is False

    def test_blocked_dangerous_modules(self):
        """Known-dangerous modules are not in the allowlist."""
        from ergodic_insurance.safe_pickle import _is_module_allowed

        for mod in ("os", "subprocess", "nt", "posix", "shutil", "ctypes", "socket"):
            assert _is_module_allowed(mod) is False, f"{mod} should be blocked"

    def test_project_module(self):
        """Project package and submodules pass."""
        from ergodic_insurance.safe_pickle import _is_module_allowed

        assert _is_module_allowed("ergodic_insurance") is True
        assert _is_module_allowed("ergodic_insurance.monte_carlo") is True


class TestRestrictedUnpicklerFindClass:
    """Unit tests for RestrictedUnpickler.find_class decisions."""

    def _make_unpickler(self):
        from ergodic_insurance.safe_pickle import RestrictedUnpickler

        return RestrictedUnpickler(io.BytesIO(b""))

    # --- Blocked classes ---

    def test_blocks_os_system(self):
        """os.system is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            self._make_unpickler().find_class("os", "system")

    def test_blocks_subprocess_popen(self):
        """subprocess.Popen is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            self._make_unpickler().find_class("subprocess", "Popen")

    def test_blocks_nt_system(self):
        """nt.system (Windows os.system) is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            self._make_unpickler().find_class("nt", "system")

    def test_blocks_posix_system(self):
        """posix.system (Unix os.system) is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            self._make_unpickler().find_class("posix", "system")

    def test_blocks_builtins_exec(self):
        """builtins.exec is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked.*exec"):
            self._make_unpickler().find_class("builtins", "exec")

    def test_blocks_builtins_eval(self):
        """builtins.eval is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked.*eval"):
            self._make_unpickler().find_class("builtins", "eval")

    def test_blocks_builtins_getattr(self):
        """builtins.getattr (attribute-chain attack vector) is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked.*getattr"):
            self._make_unpickler().find_class("builtins", "getattr")

    def test_blocks_builtins_import(self):
        """builtins.__import__ is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked.*__import__"):
            self._make_unpickler().find_class("builtins", "__import__")

    def test_blocks_builtins_open(self):
        """builtins.open is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked.*open"):
            self._make_unpickler().find_class("builtins", "open")

    def test_blocks_builtins_compile(self):
        """builtins.compile is rejected."""
        with pytest.raises(pickle.UnpicklingError, match="blocked.*compile"):
            self._make_unpickler().find_class("builtins", "compile")

    # --- Allowed classes ---

    def test_allows_builtins_dict(self):
        """builtins.dict is permitted."""
        assert self._make_unpickler().find_class("builtins", "dict") is dict

    def test_allows_builtins_set(self):
        """builtins.set is permitted."""
        assert self._make_unpickler().find_class("builtins", "set") is set

    def test_allows_builtins_int(self):
        """builtins.int is permitted."""
        assert self._make_unpickler().find_class("builtins", "int") is int

    def test_allows_numpy_ndarray(self):
        """numpy.ndarray is permitted."""
        np = pytest.importorskip("numpy")
        result = self._make_unpickler().find_class("numpy", "ndarray")
        assert result is np.ndarray

    def test_allows_numpy_submodule(self):
        """numpy submodule classes are permitted."""
        pytest.importorskip("numpy")
        # Should not raise — exact class depends on numpy version
        self._make_unpickler().find_class("numpy.core.multiarray", "_reconstruct")

    def test_allows_collections_ordereddict(self):
        """collections.OrderedDict is permitted."""
        from collections import OrderedDict

        result = self._make_unpickler().find_class("collections", "OrderedDict")
        assert result is OrderedDict

    def test_allows_datetime(self):
        """datetime.datetime is permitted."""
        import datetime

        result = self._make_unpickler().find_class("datetime", "datetime")
        assert result is datetime.datetime

    def test_allows_copyreg(self):
        """copyreg._reconstructor is permitted (pickle infrastructure)."""
        # copyreg._reconstructor exists at runtime but mypy doesn't see it,
        # so we just verify find_class does not raise.
        result = self._make_unpickler().find_class("copyreg", "_reconstructor")
        assert callable(result)

    def test_allows_project_module(self):
        """ergodic_insurance submodule classes are permitted."""
        result = self._make_unpickler().find_class(
            "ergodic_insurance.safe_pickle", "deterministic_hash"
        )
        assert result is deterministic_hash


class TestRestrictedUnpicklerEndToEnd:
    """Integration tests: malicious payloads blocked through safe_load/safe_loads."""

    def test_subprocess_blocked_with_valid_hmac(self, temp_key_dir):
        """Payload calling subprocess.Popen is blocked even with valid HMAC."""
        import subprocess

        class _Exploit:
            def __reduce__(self):
                return (subprocess.Popen, (["echo", "pwned"],))

        signed = safe_dumps(_Exploit(), key_dir=temp_key_dir)

        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            safe_loads(signed, key_dir=temp_key_dir)

    def test_eval_blocked_with_valid_hmac(self, temp_key_dir):
        """Payload calling builtins.eval is blocked even with valid HMAC."""

        class _Exploit:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned')",))

        signed = safe_dumps(_Exploit(), key_dir=temp_key_dir)

        with pytest.raises(pickle.UnpicklingError, match="blocked.*eval"):
            safe_loads(signed, key_dir=temp_key_dir)

    def test_safe_load_file_blocks_malicious(self, temp_key_dir):
        """File-based safe_load also uses the restricted unpickler."""
        import subprocess

        class _Exploit:
            def __reduce__(self):
                return (subprocess.Popen, (["echo", "pwned"],))

        buf = io.BytesIO()
        safe_dump(_Exploit(), buf, key_dir=temp_key_dir)
        buf.seek(0)

        with pytest.raises(pickle.UnpicklingError, match="blocked"):
            safe_load(buf, key_dir=temp_key_dir)

    def test_roundtrip_basic_types_still_work(self, temp_key_dir):
        """RestrictedUnpickler does not break safe roundtrips of basic types."""
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
            frozenset([4, 5]),
            b"bytes data",
        ]
        for obj in test_objects:
            data = safe_dumps(obj, key_dir=temp_key_dir)
            loaded = safe_loads(data, key_dir=temp_key_dir)
            assert loaded == obj, f"Roundtrip failed for {type(obj).__name__}"

    def test_roundtrip_numpy_array(self, temp_key_dir):
        """numpy arrays survive restricted roundtrip."""
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, 2.0, 3.0])
        data = safe_dumps(arr, key_dir=temp_key_dir)
        loaded = safe_loads(data, key_dir=temp_key_dir)
        np.testing.assert_array_equal(loaded, arr)

    def test_roundtrip_datetime(self, temp_key_dir):
        """datetime objects survive restricted roundtrip."""
        import datetime

        dt = datetime.datetime(2024, 1, 15, 12, 30, 0)
        data = safe_dumps(dt, key_dir=temp_key_dir)
        loaded = safe_loads(data, key_dir=temp_key_dir)
        assert loaded == dt

    def test_roundtrip_collections(self, temp_key_dir):
        """collections types survive restricted roundtrip."""
        from collections import OrderedDict, defaultdict

        od = OrderedDict([("a", 1), ("b", 2)])
        data = safe_dumps(od, key_dir=temp_key_dir)
        loaded = safe_loads(data, key_dir=temp_key_dir)
        assert loaded == od

        dd = defaultdict(int, {"x": 10})
        data = safe_dumps(dd, key_dir=temp_key_dir)
        loaded = safe_loads(data, key_dir=temp_key_dir)
        assert loaded == dd

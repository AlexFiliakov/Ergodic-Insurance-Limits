"""Safe pickle serialization with HMAC integrity validation.

This module provides HMAC-signed pickle operations to prevent arbitrary code
execution from tampered cache files. All file-based pickle operations in the
codebase should use these functions instead of raw pickle.load/pickle.dump.

The HMAC key is stored in a `.pickle_hmac_key` file within the cache directory
(or a default location). Files written with safe_dump can only be loaded by
safe_load if the HMAC signature matches, preventing deserialization of
untrusted data.

Also provides deterministic_hash() as a replacement for Python's
non-deterministic built-in hash() function.
"""

import hashlib
import hmac
import io
import logging
import os
from pathlib import Path
import pickle
import secrets
import stat
from typing import Any, Optional

# Default location for the HMAC key
_DEFAULT_KEY_DIR = Path.home() / ".ergodic_insurance"
_KEY_FILENAME = ".pickle_hmac_key"
_SIGNATURE_LENGTH = 32  # SHA-256 produces 32 bytes
_MAX_KEY_RETRIES = 3  # retry limit for concurrent key creation races


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restricted unpickler allowlist — defense-in-depth (issue #984)
# ---------------------------------------------------------------------------
# Modules (and their submodules) whose classes may be instantiated during
# unpickling.  An entry "foo" permits both the exact module "foo" and any
# submodule "foo.bar", "foo.bar.baz", etc.  The builtins module is gated
# separately via _ALLOWED_BUILTINS to block dangerous functions.

_ALLOWED_MODULES: frozenset = frozenset(
    {
        # --- Project package ---
        "ergodic_insurance",
        # --- Scientific / data stack ---
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        # --- Pickle reconstruction infrastructure ---
        "copyreg",
        "_codecs",
        "copy",
        # --- Standard-library data types and their C accelerators ---
        "collections",
        "_collections",
        "_collections_abc",
        "datetime",
        "_datetime",
        "decimal",
        "_decimal",
        "fractions",
        "enum",
        "dataclasses",
        "pathlib",
        "uuid",
        "array",
        "io",
        "_io",
        "re",
        "_sre",
        "operator",
        "_operator",
        "functools",
        "_functools",
        "itertools",
        "math",
        "numbers",
        "abc",
        "types",
        "struct",
        "_struct",
        "zlib",
    }
)

# Built-in names that are safe for unpickling (data types only).
# Dangerous callables — exec, eval, compile, getattr, __import__, open —
# are intentionally excluded.
_ALLOWED_BUILTINS: frozenset = frozenset(
    {
        # Primitive / scalar types
        "bool",
        "int",
        "float",
        "complex",
        "str",
        "bytes",
        "bytearray",
        # Container types
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        # Misc safe types
        "slice",
        "range",
        "type",
        "object",
        "NoneType",
    }
)


def _is_module_allowed(module: str) -> bool:
    """Return True if *module* is in the allowlist or is a submodule of one."""
    if module in _ALLOWED_MODULES:
        return True
    for allowed in _ALLOWED_MODULES:
        if module.startswith(allowed + "."):
            return True
    return False


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler with a class allowlist to prevent arbitrary code execution.

    Defense-in-depth: even if HMAC verification is bypassed (e.g. key
    compromise), only classes from explicitly allowed modules can be
    instantiated during deserialization.  This blocks common RCE vectors
    such as os.system, subprocess.Popen, and builtins.exec.

    See: https://docs.python.org/3/library/pickle.html#restricting-globals
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins":
            if name in _ALLOWED_BUILTINS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Restricted unpickler blocked builtins.{name}")

        if _is_module_allowed(module):
            return super().find_class(module, name)

        raise pickle.UnpicklingError(f"Restricted unpickler blocked {module}.{name}")


def _restricted_loads(data: bytes) -> Any:
    """Deserialize *data* using the :class:`RestrictedUnpickler`."""
    return RestrictedUnpickler(io.BytesIO(data)).load()


def _get_key_path(key_dir: Optional[Path] = None) -> Path:
    """Get the path to the HMAC key file."""
    directory = key_dir or _DEFAULT_KEY_DIR
    return directory / _KEY_FILENAME


def _secure_mkdir(dir_path: Path) -> None:
    """Create directory with restricted permissions (0700 on Unix)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(str(dir_path), stat.S_IRWXU)


def _secure_write_key(key_path: Path, key: bytes) -> None:
    """Write key file atomically with restricted permissions.

    Uses O_CREAT | O_EXCL on all platforms for atomic creation,
    preventing TOCTOU races where two processes could clobber each
    other's keys.  Raises FileExistsError if the file already exists.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)  # required on Windows for raw bytes
    mode = 0o600 if os.name != "nt" else 0
    fd = os.open(str(key_path), flags, mode)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)


def _get_or_create_hmac_key(key_dir: Optional[Path] = None) -> bytes:
    """Get or create a persistent HMAC key for pickle validation.

    Eliminates TOCTOU races by using try/except instead of exists() + read(),
    and retries if a concurrent process creates or deletes the key file
    between our read and write attempts.

    Args:
        key_dir: Directory to store the key file. Defaults to ~/.ergodic_insurance/

    Returns:
        32-byte HMAC key
    """
    key_path = _get_key_path(key_dir)

    for _attempt in range(_MAX_KEY_RETRIES):
        # Try reading existing key — avoids TOCTOU from exists() + read_bytes()
        try:
            return key_path.read_bytes()
        except FileNotFoundError:
            pass

        # Key doesn't exist yet; create directory and attempt atomic write
        _secure_mkdir(key_path.parent)
        key = secrets.token_bytes(32)
        try:
            _secure_write_key(key_path, key)
            return key
        except FileExistsError:
            # Another process won the race — loop back and read their key
            _logger.debug("HMAC key file created by another process, retrying read")
            continue

    # Exhausted retries — final read attempt (let any exception propagate)
    return key_path.read_bytes()


def safe_dump(
    obj: Any,
    f,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    key_dir: Optional[Path] = None,
) -> None:
    """Pickle dump with HMAC signature prepended.

    Args:
        obj: Object to serialize
        f: Writable binary file object
        protocol: Pickle protocol version
        key_dir: Directory containing the HMAC key
    """
    data = pickle.dumps(obj, protocol=protocol)
    key = _get_or_create_hmac_key(key_dir)
    signature = hmac.new(key, data, hashlib.sha256).digest()
    f.write(signature)
    f.write(data)


def safe_load(f, key_dir: Optional[Path] = None) -> Any:
    """Pickle load with HMAC verification.

    Args:
        f: Readable binary file object
        key_dir: Directory containing the HMAC key

    Returns:
        Deserialized object

    Raises:
        ValueError: If HMAC verification fails or file is too short
    """
    content = f.read()
    if len(content) < _SIGNATURE_LENGTH:
        raise ValueError("Invalid pickle file: too short for HMAC verification")

    signature = content[:_SIGNATURE_LENGTH]
    data = content[_SIGNATURE_LENGTH:]

    key = _get_or_create_hmac_key(key_dir)
    expected_sig = hmac.new(key, data, hashlib.sha256).digest()

    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError(
            "Pickle file integrity check failed: HMAC mismatch. "
            "File may have been tampered with or was created by a different key."
        )

    return _restricted_loads(data)


def safe_dumps(
    obj: Any,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    key_dir: Optional[Path] = None,
) -> bytes:
    """Pickle dumps with HMAC signature prepended.

    Args:
        obj: Object to serialize
        protocol: Pickle protocol version
        key_dir: Directory containing the HMAC key

    Returns:
        HMAC signature + pickled bytes
    """
    data = pickle.dumps(obj, protocol=protocol)
    key = _get_or_create_hmac_key(key_dir)
    signature = hmac.new(key, data, hashlib.sha256).digest()
    return signature + data


def safe_loads(data: bytes, key_dir: Optional[Path] = None) -> Any:
    """Pickle loads with HMAC verification.

    Args:
        data: HMAC signature + pickled bytes
        key_dir: Directory containing the HMAC key

    Returns:
        Deserialized object

    Raises:
        ValueError: If HMAC verification fails or data is too short
    """
    if len(data) < _SIGNATURE_LENGTH:
        raise ValueError("Invalid pickle data: too short for HMAC verification")

    signature = data[:_SIGNATURE_LENGTH]
    payload = data[_SIGNATURE_LENGTH:]

    key = _get_or_create_hmac_key(key_dir)
    expected_sig = hmac.new(key, payload, hashlib.sha256).digest()

    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError(
            "Pickle data integrity check failed: HMAC mismatch. "
            "Data may have been tampered with or was created by a different key."
        )

    return _restricted_loads(payload)


def deterministic_hash(*args: str, length: int = 16) -> str:
    """Generate a deterministic hash from string arguments.

    Uses SHA-256 instead of Python's non-deterministic hash().
    This produces the same result across process restarts regardless
    of PYTHONHASHSEED.

    Args:
        *args: String values to hash
        length: Number of hex characters to return (max 64)

    Returns:
        Hex digest string of specified length
    """
    combined = "|".join(str(a) for a in args)
    return hashlib.sha256(combined.encode()).hexdigest()[:length]

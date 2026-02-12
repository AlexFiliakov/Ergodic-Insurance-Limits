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

    return pickle.loads(data)  # noqa: S301 - HMAC-verified before loading


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

    return pickle.loads(payload)  # noqa: S301 - HMAC-verified before loading


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

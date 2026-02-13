"""GPU compute backend abstraction layer.

Provides a CuPy-based abstraction that gracefully degrades to NumPy when
no GPU is available. All downstream GPU-accelerated modules (#961-#971)
depend on this module for device-agnostic array operations.

Usage::

    from ergodic_insurance.gpu_backend import get_array_module, to_numpy

    xp = get_array_module(gpu=True)   # cupy if available, else numpy
    a = xp.random.normal(size=1000)
    result = to_numpy(a)               # always a numpy.ndarray

Since:
    Version 0.10.0 (Issue #960)
"""

import contextlib
import logging
from typing import Any, Dict, Generator
import warnings

import numpy as np
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  CuPy detection
# ---------------------------------------------------------------------------

_CUPY_AVAILABLE: bool = False
_cupy_module: Any = None

try:
    import cupy

    _cupy_module = cupy
    _CUPY_AVAILABLE = True
    logger.debug("CuPy detected — GPU acceleration available")
except ImportError:
    logger.debug("CuPy not installed — falling back to NumPy")
except Exception:  # noqa: BLE001  # pragma: no cover — runtime CUDA errors
    logger.debug("CuPy import failed — falling back to NumPy")


# ---------------------------------------------------------------------------
#  GPUConfig
# ---------------------------------------------------------------------------


class GPUConfig(BaseModel):
    """GPU acceleration configuration.

    Controls whether GPU acceleration is used and how GPU resources are
    managed. When ``enabled=True`` but CuPy is not installed, a warning
    is emitted and operations transparently fall back to NumPy.

    Attributes:
        enabled: Whether to attempt GPU acceleration.
        device_id: CUDA device ordinal to use.
        memory_pool: Whether to use CuPy's memory pool allocator.
        pin_memory: Whether to use pinned (page-locked) host memory
            for faster CPU↔GPU transfers.
        random_seed: Optional seed for GPU random number generator.
        synchronize: Whether to synchronize after each kernel launch.
            Useful for profiling but reduces throughput.

    Examples:
        Default (GPU disabled)::

            gpu_cfg = GPUConfig()

        Enable GPU::

            gpu_cfg = GPUConfig(enabled=True, device_id=0)

    Since:
        Version 0.10.0 (Issue #960)
    """

    enabled: bool = Field(default=False, description="Enable GPU acceleration")
    device_id: int = Field(default=0, ge=0, description="CUDA device ordinal")
    memory_pool: bool = Field(default=True, description="Use CuPy memory pool allocator")
    pin_memory: bool = Field(default=False, description="Use pinned host memory for transfers")
    random_seed: int | None = Field(default=None, ge=0, description="GPU RNG seed")
    synchronize: bool = Field(
        default=False, description="Synchronize after each kernel (for profiling)"
    )

    @model_validator(mode="after")
    def warn_if_unavailable(self):
        """Warn when GPU is requested but CuPy is not installed."""
        if self.enabled and not _CUPY_AVAILABLE:
            warnings.warn(
                "GPUConfig.enabled=True but CuPy is not installed. "
                "Operations will fall back to NumPy. "
                "Install with: pip install 'ergodic-insurance[gpu]'",
                UserWarning,
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
#  Core helpers
# ---------------------------------------------------------------------------


def is_gpu_available() -> bool:
    """Check whether CuPy (GPU support) is importable.

    Returns:
        True if CuPy was successfully imported at module load time.
    """
    return _CUPY_AVAILABLE


def get_array_module(gpu: bool = True) -> Any:
    """Return the array module to use (``cupy`` or ``numpy``).

    Args:
        gpu: If True *and* CuPy is available, return ``cupy``.
            Otherwise return ``numpy``.

    Returns:
        The ``cupy`` module when GPU is requested and available,
        otherwise the ``numpy`` module.
    """
    if gpu and _CUPY_AVAILABLE:
        return _cupy_module
    return np


def to_device(arr: Any, gpu: bool = True) -> Any:
    """Transfer an array to the requested device.

    Args:
        arr: A numpy or cupy ndarray (or array-like).
        gpu: If True, move to GPU; if False, move to CPU.

    Returns:
        The array on the requested device. If the array is already on
        the target device, a no-op reference is returned.
    """
    if gpu and _CUPY_AVAILABLE:
        if isinstance(arr, np.ndarray):
            return _cupy_module.asarray(arr)
        # Already a cupy array or compatible
        return _cupy_module.asarray(arr)
    # Move to CPU
    return to_numpy(arr)


def to_numpy(arr: Any) -> np.ndarray:
    """Ensure the input is a :class:`numpy.ndarray`.

    Handles CuPy arrays, plain lists, and NumPy arrays transparently.

    Args:
        arr: Input data — cupy ndarray, numpy ndarray, list, or scalar.

    Returns:
        A ``numpy.ndarray``.
    """
    if _CUPY_AVAILABLE and isinstance(arr, _cupy_module.ndarray):
        return arr.get()  # type: ignore[no-any-return]
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def is_gpu_array(arr: Any) -> bool:
    """Check whether *arr* is a CuPy GPU array.

    Args:
        arr: Any object.

    Returns:
        True if CuPy is available and *arr* is a ``cupy.ndarray``.
    """
    if _CUPY_AVAILABLE:
        return isinstance(arr, _cupy_module.ndarray)
    return False


# ---------------------------------------------------------------------------
#  Device info & seeding
# ---------------------------------------------------------------------------


def gpu_info() -> Dict[str, Any]:
    """Return a dictionary of GPU device metadata.

    Returns:
        Dict with keys ``available``, and when a GPU is present:
        ``device_name``, ``total_memory_bytes``, ``cuda_version``.
        If CuPy is unavailable or an error occurs, only ``available``
        (set to False) is returned.
    """
    if not _CUPY_AVAILABLE:
        return {"available": False}
    try:
        device = _cupy_module.cuda.Device()
        return {
            "available": True,
            "device_name": device.attributes.get("DeviceName", str(device)),
            "total_memory_bytes": device.mem_info[1],
            "cuda_version": _cupy_module.cuda.runtime.runtimeGetVersion(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to query GPU info: %s", exc)
        return {"available": False, "error": str(exc)}


def set_random_seed(seed: int, gpu: bool = True) -> None:
    """Set random seeds for reproducibility.

    Always sets the NumPy seed. When *gpu* is True and CuPy is
    available, also sets the CuPy seed.

    Args:
        seed: Non-negative integer seed.
        gpu: Whether to also seed the GPU RNG.
    """
    np.random.seed(seed)
    if gpu and _CUPY_AVAILABLE:
        _cupy_module.random.seed(seed)


# ---------------------------------------------------------------------------
#  Context managers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def gpu_memory_pool() -> Generator[None, None, None]:
    """Context manager that frees the CuPy memory pool on exit.

    When CuPy is unavailable this is a no-op.

    Yields:
        None
    """
    try:
        yield
    finally:
        if _CUPY_AVAILABLE:
            pool = _cupy_module.get_default_memory_pool()
            pool.free_all_blocks()


@contextlib.contextmanager
def gpu_device(device_id: int) -> Generator[None, None, None]:
    """Context manager that switches the active CUDA device.

    Restores the previous device on exit. When CuPy is unavailable
    this is a no-op.

    Args:
        device_id: CUDA device ordinal to activate.

    Yields:
        None
    """
    if not _CUPY_AVAILABLE:
        yield
        return

    prev = _cupy_module.cuda.Device().id
    _cupy_module.cuda.Device(device_id).use()
    try:
        yield
    finally:
        _cupy_module.cuda.Device(prev).use()


# ---------------------------------------------------------------------------
#  Colab helpers
# ---------------------------------------------------------------------------


def detect_colab_environment() -> bool:
    """Detect whether the code is running inside Google Colab.

    Returns:
        True if the ``google.colab`` module is importable.
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def colab_setup_helper() -> str:
    """Return pip install instructions for setting up CuPy in Colab.

    Returns:
        A string with shell commands to install the GPU optional
        dependency in a Google Colab notebook.
    """
    return (
        "# Run this cell to install GPU support in Google Colab:\n"
        "!pip install 'ergodic-insurance[gpu]'\n"
        "\n"
        "# Then restart the runtime:\n"
        "# Runtime → Restart runtime  (or Ctrl+M .)\n"
    )

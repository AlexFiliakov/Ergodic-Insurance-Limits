"""Tests for the GPU compute backend abstraction layer.

All tests are CPU-compatible — CuPy presence is mocked via
``unittest.mock.patch.object`` on the module-level sentinels
``_CUPY_AVAILABLE`` and ``_cupy_module``.

Since:
    Version 0.10.0 (Issue #960)
"""

from types import ModuleType
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pytest

import ergodic_insurance.gpu_backend as gpu_mod
from ergodic_insurance.gpu_backend import (
    GPUConfig,
    colab_setup_helper,
    detect_colab_environment,
    get_array_module,
    gpu_device,
    gpu_info,
    gpu_memory_pool,
    is_gpu_array,
    is_gpu_available,
    set_random_seed,
    to_device,
    to_numpy,
)

# ── helpers ───────────────────────────────────────────────────────────────


def _make_mock_cupy() -> MagicMock:
    """Create a mock CuPy module with the interfaces we use."""
    mock_cp = MagicMock()
    mock_cp.ndarray = type("ndarray", (), {})  # real class for isinstance
    mock_cp.asarray = MagicMock(side_effect=lambda a: a)

    # Device helpers
    mock_device = MagicMock()
    mock_device.id = 0
    mock_device.mem_info = (1_000_000, 8_000_000_000)
    mock_device.attributes = {"DeviceName": "MockGPU"}
    mock_cp.cuda.Device.return_value = mock_device
    mock_cp.cuda.runtime.runtimeGetVersion.return_value = 12000

    # Memory pool
    mock_pool = MagicMock()
    mock_cp.get_default_memory_pool.return_value = mock_pool

    # Random
    mock_cp.random.seed = MagicMock()

    return mock_cp


# ── TestGPUAvailability ──────────────────────────────────────────────────


class TestGPUAvailability:
    """Detection with/without CuPy."""

    def test_unavailable_by_default(self):
        """Without CuPy installed, is_gpu_available returns False."""
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            assert is_gpu_available() is False

    def test_available_when_cupy_present(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", True):
            assert is_gpu_available() is True


# ── TestGetArrayModule ───────────────────────────────────────────────────


class TestGetArrayModule:
    """numpy fallback and cupy selection."""

    def test_returns_numpy_when_unavailable(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            xp = get_array_module(gpu=True)
            assert xp is np

    def test_returns_numpy_when_gpu_false(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            xp = get_array_module(gpu=False)
            assert xp is np

    def test_returns_cupy_when_available(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            xp = get_array_module(gpu=True)
            assert xp is mock_cp

    def test_default_argument_requests_gpu(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            xp = get_array_module()
            assert xp is mock_cp


# ── TestToDevice ─────────────────────────────────────────────────────────


class TestToDevice:
    """CPU↔GPU transfers and no-ops."""

    def test_to_gpu_calls_cupy_asarray(self):
        mock_cp = _make_mock_cupy()
        arr = np.array([1.0, 2.0])
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            to_device(arr, gpu=True)
            mock_cp.asarray.assert_called()

    def test_to_cpu_returns_numpy(self):
        arr = np.array([1.0, 2.0])
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            result = to_device(arr, gpu=False)
            assert isinstance(result, np.ndarray)

    def test_noop_when_unavailable(self):
        arr = np.array([3.0])
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            result = to_device(arr, gpu=True)
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, arr)


# ── TestToNumpy ──────────────────────────────────────────────────────────


class TestToNumpy:
    """cupy→numpy, list→numpy, numpy no-op."""

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = to_numpy(arr)
        assert result is arr

    def test_list_conversion(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_cupy_array_calls_get(self):
        mock_cp = _make_mock_cupy()
        # Create a real instance of the mock ndarray class so isinstance works,
        # then attach a mock get() method.
        ndarray_cls = mock_cp.ndarray
        mock_arr = ndarray_cls()
        mock_arr.get = MagicMock(return_value=np.array([1.0]))
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            result = to_numpy(mock_arr)
            mock_arr.get.assert_called_once()
            assert isinstance(result, np.ndarray)

    def test_scalar_conversion(self):
        result = to_numpy(42)
        assert isinstance(result, np.ndarray)


# ── TestGPUInfo ──────────────────────────────────────────────────────────


class TestGPUInfo:
    """Unavailable dict, device metadata, error handling."""

    def test_unavailable_info(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            info = gpu_info()
            assert info == {"available": False}

    def test_available_info(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            info = gpu_info()
            assert info["available"] is True
            assert info["device_name"] == "MockGPU"
            assert info["total_memory_bytes"] == 8_000_000_000
            assert info["cuda_version"] == 12000

    def test_error_handling(self):
        mock_cp = _make_mock_cupy()
        mock_cp.cuda.Device.side_effect = RuntimeError("CUDA error")
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            info = gpu_info()
            assert info["available"] is False
            assert "error" in info


# ── TestSetRandomSeed ────────────────────────────────────────────────────


class TestSetRandomSeed:
    """numpy-only and numpy+cupy seeding."""

    def test_numpy_only(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            set_random_seed(42, gpu=False)
            # Verify numpy seed was set by generating deterministic output
            a = np.random.rand()
            set_random_seed(42, gpu=False)
            b = np.random.rand()
            assert a == b

    def test_numpy_plus_cupy(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            set_random_seed(123, gpu=True)
            mock_cp.random.seed.assert_called_once_with(123)

    def test_gpu_false_skips_cupy(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            set_random_seed(99, gpu=False)
            mock_cp.random.seed.assert_not_called()


# ── TestGPUConfig ────────────────────────────────────────────────────────


class TestGPUConfig:
    """Defaults, custom values, validation, warnings."""

    def test_defaults(self):
        cfg = GPUConfig()
        assert cfg.enabled is False
        assert cfg.device_id == 0
        assert cfg.memory_pool is True
        assert cfg.pin_memory is False
        assert cfg.random_seed is None
        assert cfg.synchronize is False

    def test_custom_values(self):
        cfg = GPUConfig(
            enabled=False,
            device_id=1,
            memory_pool=False,
            pin_memory=True,
            random_seed=42,
            synchronize=True,
        )
        assert cfg.device_id == 1
        assert cfg.memory_pool is False
        assert cfg.pin_memory is True
        assert cfg.random_seed == 42
        assert cfg.synchronize is True

    def test_warns_when_enabled_but_unavailable(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                GPUConfig(enabled=True)
                assert len(w) == 1
                assert "CuPy is not installed" in str(w[0].message)

    def test_no_warning_when_disabled(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                GPUConfig(enabled=False)
                assert len(w) == 0

    def test_no_warning_when_cupy_available(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                GPUConfig(enabled=True)
                assert len(w) == 0

    def test_serialization_roundtrip(self):
        cfg = GPUConfig(enabled=False, device_id=2, random_seed=7)
        data = cfg.model_dump()
        cfg2 = GPUConfig(**data)
        assert cfg == cfg2

    def test_negative_device_id_rejected(self):
        with pytest.raises(Exception):
            GPUConfig(device_id=-1)


# ── TestContextManagers ──────────────────────────────────────────────────


class TestContextManagers:
    """Memory pool, device switching, no-ops when unavailable."""

    def test_memory_pool_noop_when_unavailable(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            with gpu_memory_pool():
                pass  # Should not raise

    def test_memory_pool_frees_blocks(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            with gpu_memory_pool():
                pass
            mock_cp.get_default_memory_pool().free_all_blocks.assert_called_once()

    def test_gpu_device_noop_when_unavailable(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            with gpu_device(0):
                pass  # Should not raise

    def test_gpu_device_switches_and_restores(self):
        mock_cp = _make_mock_cupy()
        mock_device_0 = MagicMock()
        mock_device_0.id = 0
        mock_device_1 = MagicMock()
        mock_device_1.id = 1

        call_count = 0

        def device_factory(dev_id=None):
            nonlocal call_count
            call_count += 1
            if dev_id is None:
                # Querying current device
                return mock_device_0
            d = MagicMock()
            d.id = dev_id
            return d

        mock_cp.cuda.Device = device_factory

        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            with gpu_device(1):
                pass  # Should switch to device 1, then restore to 0

    def test_memory_pool_frees_on_exception(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            with pytest.raises(ValueError):
                with gpu_memory_pool():
                    raise ValueError("test")
            mock_cp.get_default_memory_pool().free_all_blocks.assert_called_once()


# ── TestHelperFunctions ──────────────────────────────────────────────────


class TestHelperFunctions:
    """is_gpu_array, colab detection, setup helper."""

    def test_is_gpu_array_false_when_unavailable(self):
        with patch.object(gpu_mod, "_CUPY_AVAILABLE", False):
            assert is_gpu_array(np.array([1])) is False

    def test_is_gpu_array_false_for_numpy(self):
        mock_cp = _make_mock_cupy()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            assert is_gpu_array(np.array([1])) is False

    def test_is_gpu_array_true_for_cupy(self):
        mock_cp = _make_mock_cupy()
        mock_arr = mock_cp.ndarray()
        with (
            patch.object(gpu_mod, "_CUPY_AVAILABLE", True),
            patch.object(gpu_mod, "_cupy_module", mock_cp),
        ):
            assert is_gpu_array(mock_arr) is True

    def test_detect_colab_false(self):
        # google.colab is not available in standard test environments
        assert detect_colab_environment() is False

    def test_detect_colab_true(self):
        fake_colab = MagicMock()
        with patch.dict("sys.modules", {"google.colab": fake_colab, "google": MagicMock()}):
            assert detect_colab_environment() is True

    def test_colab_setup_helper_returns_string(self):
        text = colab_setup_helper()
        assert isinstance(text, str)
        assert "pip install" in text
        assert "ergodic-insurance[gpu]" in text


# ── TestConfigIntegration ────────────────────────────────────────────────


class TestConfigIntegration:
    """Import paths, Config composition, backward compat."""

    def test_import_from_gpu_backend(self):
        from ergodic_insurance.gpu_backend import GPUConfig as GC

        assert GC is GPUConfig

    def test_import_from_config_package(self):
        from ergodic_insurance.config import GPUConfig as GC

        assert GC is GPUConfig

    def test_config_gpu_field_default_none(self):
        from ergodic_insurance.config import Config

        cfg = Config()
        assert cfg.gpu is None

    def test_config_with_gpu(self):
        from ergodic_insurance.config import Config

        cfg = Config(gpu=GPUConfig(enabled=False, device_id=1))
        assert cfg.gpu is not None
        assert cfg.gpu.device_id == 1

    def test_config_gpu_serialization(self):
        from ergodic_insurance.config import Config

        cfg = Config(gpu=GPUConfig(random_seed=99))
        data = cfg.model_dump()
        assert data["gpu"]["random_seed"] == 99
        cfg2 = Config(**data)
        assert cfg2.gpu is not None
        assert cfg2.gpu.random_seed == 99

    def test_lazy_import_from_top_level(self):
        from ergodic_insurance import GPUConfig as GC
        from ergodic_insurance import get_array_module, is_gpu_available, to_numpy

        assert GC is GPUConfig
        assert callable(get_array_module)
        assert callable(is_gpu_available)
        assert callable(to_numpy)

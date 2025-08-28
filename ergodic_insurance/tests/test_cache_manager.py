"""Comprehensive test suite for the cache manager.

This module tests all aspects of the CacheManager including storage, retrieval,
invalidation, performance benchmarks, and edge cases.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.reporting import (
    CacheConfig,
    CacheKey,
    CacheManager,
    CacheStats,
    StorageBackend,
)
from ergodic_insurance.src.reporting.cache_manager import BaseStorageBackend, LocalStorageBackend


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.cache_dir == Path("./cache")
        assert config.max_cache_size_gb == 10.0
        assert config.ttl_hours is None
        assert config.compression == "gzip"
        assert config.compression_level == 4
        assert config.enable_memory_mapping is True
        assert config.backend == StorageBackend.LOCAL

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            cache_dir=Path("/tmp/test_cache"),
            max_cache_size_gb=5.0,
            ttl_hours=24,
            compression="lzf",
            compression_level=6,
        )
        assert config.cache_dir == Path("/tmp/test_cache")
        assert config.max_cache_size_gb == 5.0
        assert config.ttl_hours == 24
        assert config.compression == "lzf"
        assert config.compression_level == 6

    def test_invalid_compression(self):
        """Test invalid compression algorithm."""
        with pytest.raises(ValueError, match="Invalid compression"):
            CacheConfig(compression="invalid")

    def test_invalid_compression_level(self):
        """Test invalid compression level."""
        with pytest.raises(ValueError, match="Compression level must be 1-9"):
            CacheConfig(compression_level=10)


class TestCacheKey:
    """Test cache key functionality."""

    def test_cache_key_creation(self):
        """Test cache key creation."""
        key = CacheKey(hash_key="abc123", params={"n_sims": 1000}, size_bytes=1024)
        assert key.hash_key == "abc123"
        assert key.params == {"n_sims": 1000}
        assert key.size_bytes == 1024
        assert key.access_count == 0

    def test_cache_key_serialization(self):
        """Test cache key serialization and deserialization."""
        key = CacheKey(hash_key="abc123", params={"n_sims": 1000}, size_bytes=1024, access_count=5)

        # Serialize
        data = key.to_dict()
        assert data["hash_key"] == "abc123"
        assert data["params"] == {"n_sims": 1000}
        assert data["size_bytes"] == 1024
        assert data["access_count"] == 5

        # Deserialize
        key2 = CacheKey.from_dict(data)
        assert key2.hash_key == key.hash_key
        assert key2.params == key.params
        assert key2.size_bytes == key.size_bytes
        assert key2.access_count == key.access_count


class TestCacheManager:
    """Test cache manager functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create cache manager with temporary directory."""
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            max_cache_size_gb=0.001,  # 1 MB for testing
            compression="gzip",
            compression_level=1,  # Fast compression for tests
        )
        return CacheManager(config)

    def test_initialization(self, cache_manager, temp_cache_dir):
        """Test cache manager initialization."""
        assert cache_manager.config.cache_dir == Path(temp_cache_dir)
        assert isinstance(cache_manager.stats, CacheStats)

        # Check directory structure created
        assert (Path(temp_cache_dir) / "raw_simulations").exists()
        assert (Path(temp_cache_dir) / "processed_results").exists()
        assert (Path(temp_cache_dir) / "figures" / "executive").exists()
        assert (Path(temp_cache_dir) / "figures" / "technical").exists()
        assert (Path(temp_cache_dir) / "metadata").exists()

    def test_cache_key_generation(self, cache_manager):
        """Test consistent cache key generation."""
        params1 = {"n_sims": 1000, "seed": 42}
        params2 = {"seed": 42, "n_sims": 1000}  # Different order
        params3 = {"n_sims": 1001, "seed": 42}  # Different value

        key1 = cache_manager._generate_cache_key(params1)
        key2 = cache_manager._generate_cache_key(params2)
        key3 = cache_manager._generate_cache_key(params3)

        # Same params in different order should give same key
        assert key1 == key2
        # Different params should give different key
        assert key1 != key3
        # Should be valid hex string (SHA256)
        assert len(key1) == 64
        assert all(c in "0123456789abcdef" for c in key1)

    def test_cache_simulation_paths(self, cache_manager):
        """Test caching simulation paths."""
        params = {"n_sims": 100, "n_years": 10, "seed": 42}
        paths = np.random.randn(100, 10)
        metadata = {"generator": "numpy", "version": "1.0"}

        # Cache the paths
        cache_key = cache_manager.cache_simulation_paths(
            params=params, paths=paths, metadata=metadata
        )

        assert cache_key is not None
        assert len(cache_key) == 64  # SHA256 hash

        # Check file created
        file_path = cache_manager.config.cache_dir / "raw_simulations" / f"{cache_key}.h5"
        assert file_path.exists()

        # Verify HDF5 contents
        with h5py.File(file_path, "r") as f:
            assert "paths" in f
            assert f["paths"].shape == (100, 10)
            assert "metadata" in f
            assert f.attrs["params"] == json.dumps(params, sort_keys=True)

        # Check cache stats updated
        assert cache_manager.stats.n_entries == 1
        assert cache_manager.stats.total_size_bytes > 0

    def test_load_simulation_paths(self, cache_manager):
        """Test loading simulation paths from cache."""
        params = {"n_sims": 100, "n_years": 10, "seed": 42}
        original_paths = np.random.randn(100, 10)

        # Cache the paths
        cache_manager.cache_simulation_paths(params=params, paths=original_paths)

        # Load from cache
        loaded_paths = cache_manager.load_simulation_paths(params=params)

        assert loaded_paths is not None
        assert loaded_paths.shape == original_paths.shape
        np.testing.assert_array_almost_equal(loaded_paths, original_paths)

        # Check cache hit statistics
        assert cache_manager.stats.n_hits == 1
        assert cache_manager.stats.n_misses == 0
        assert cache_manager.stats.hit_rate == 1.0

    def test_load_missing_simulation(self, cache_manager):
        """Test loading non-existent simulation."""
        params = {"n_sims": 999, "never": "cached"}

        loaded = cache_manager.load_simulation_paths(params=params)

        assert loaded is None
        assert cache_manager.stats.n_hits == 0
        assert cache_manager.stats.n_misses == 1
        assert cache_manager.stats.hit_rate == 0.0

    def test_cache_processed_results(self, cache_manager):
        """Test caching processed results as Parquet."""
        params = {"analysis": "efficient_frontier", "n_points": 50}
        results = pd.DataFrame(
            {
                "limit": np.linspace(1e6, 50e6, 50),
                "premium": np.linspace(1e4, 5e5, 50),
                "roe": np.random.randn(50) * 0.1 + 0.15,
            }
        )

        # Cache the results
        cache_key = cache_manager.cache_processed_results(
            params=params, results=results, result_type="efficient_frontier"
        )

        assert cache_key is not None

        # Check file created
        file_path = (
            cache_manager.config.cache_dir
            / "processed_results"
            / f"efficient_frontier_{cache_key}.parquet"
        )
        assert file_path.exists()

        # Verify Parquet contents
        loaded_df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(loaded_df, results)

    def test_load_processed_results(self, cache_manager):
        """Test loading processed results from cache."""
        params = {"analysis": "sensitivity", "variable": "margin"}
        original_results = pd.DataFrame(
            {"margin": [0.05, 0.08, 0.10, 0.12], "optimal_limit": [5e6, 10e6, 15e6, 20e6]}
        )

        # Cache the results
        cache_manager.cache_processed_results(
            params=params, results=original_results, result_type="sensitivity_analysis"
        )

        # Load from cache
        loaded_results = cache_manager.load_processed_results(
            params=params, result_type="sensitivity_analysis"
        )

        assert loaded_results is not None
        pd.testing.assert_frame_equal(loaded_results, original_results)

    def test_cache_figure(self, cache_manager):
        """Test caching figure objects."""
        params = {"chart": "growth", "period": "10y"}

        # Create mock figure
        figure = {"data": [1, 2, 3], "layout": {"title": "Test"}}

        # Cache the figure
        cache_key = cache_manager.cache_figure(
            params=params, figure=figure, figure_name="growth_chart", figure_type="technical"
        )

        assert cache_key is not None

        # Check file created
        file_path = (
            cache_manager.config.cache_dir
            / "figures"
            / "technical"
            / f"growth_chart_{cache_key}.pkl"
        )
        assert file_path.exists()

    def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation."""
        params1 = {"sim": 1, "seed": 42}
        params2 = {"sim": 2, "seed": 43}

        # Cache multiple items
        paths1 = np.random.randn(100, 10)
        paths2 = np.random.randn(100, 10)

        cache_manager.cache_simulation_paths(params=params1, paths=paths1)
        cache_manager.cache_simulation_paths(params=params2, paths=paths2)

        assert cache_manager.stats.n_entries == 2

        # Invalidate one entry
        cache_manager.invalidate_cache(params1)

        # Check first is gone, second remains
        assert cache_manager.load_simulation_paths(params1) is None
        assert cache_manager.load_simulation_paths(params2) is not None
        assert cache_manager.stats.n_entries == 1

    def test_clear_cache(self, cache_manager):
        """Test clearing entire cache."""
        # Add some cached items
        for i in range(5):
            params = {"sim": i}
            paths = np.random.randn(10, 10)
            cache_manager.cache_simulation_paths(params=params, paths=paths)

        assert cache_manager.stats.n_entries == 5

        # Clear cache (without confirmation prompt)
        cache_manager.clear_cache(confirm=False)

        assert cache_manager.stats.n_entries == 0
        assert cache_manager.stats.total_size_bytes == 0

    def test_ttl_expiration(self, cache_manager):
        """Test time-to-live expiration."""
        # Configure with short TTL
        cache_manager.config.ttl_hours = 0.0001  # Very short for testing

        params = {"ttl": "test"}
        paths = np.random.randn(10, 10)

        # Cache data
        cache_manager.cache_simulation_paths(params=params, paths=paths)

        # Sleep to let it expire
        time.sleep(0.5)

        # Try to load - should be expired
        loaded = cache_manager.load_simulation_paths(params=params)
        assert loaded is None

    def test_size_limit_enforcement(self, temp_cache_dir):
        """Test cache size limit enforcement with LRU eviction."""
        # Create cache manager with tiny limit
        config = CacheConfig(
            cache_dir=temp_cache_dir, max_cache_size_gb=0.0001, compression="gzip"  # 100 KB
        )
        cache_manager = CacheManager(config)

        # Add multiple small items
        old_params = {"old": True}
        new_params = {"new": True}

        # Cache old item (small enough to fit)
        cache_manager.cache_simulation_paths(
            params=old_params, paths=np.random.randn(10, 10)  # Much smaller array
        )

        time.sleep(0.1)  # Ensure different timestamps

        # Cache new item (should evict old due to size limit)
        cache_manager.cache_simulation_paths(
            params=new_params, paths=np.random.randn(10, 10)  # Much smaller array
        )

        # Both should exist since they're small
        # But if we add more, oldest should be evicted
        for i in range(10):
            cache_manager.cache_simulation_paths(params={"item": i}, paths=np.random.randn(10, 10))

        # Old should be evicted by now
        assert cache_manager.load_simulation_paths(old_params) is None

    def test_warm_cache(self, cache_manager):
        """Test cache warming functionality."""
        scenarios = [{"n_sims": 100, "seed": i} for i in range(3)]

        def compute_func(params):
            """Mock computation function."""
            return np.random.RandomState(params["seed"]).randn(params["n_sims"], 10)

        # Warm the cache
        n_cached = cache_manager.warm_cache(
            scenarios=scenarios, compute_func=compute_func, result_type="simulation"
        )

        assert n_cached == 3
        assert cache_manager.stats.n_entries == 3

        # All scenarios should be cached
        for params in scenarios:
            loaded = cache_manager.load_simulation_paths(params)
            assert loaded is not None
            assert loaded.shape == (params["n_sims"], 10)

    def test_cache_validation(self, cache_manager):
        """Test cache validation functionality."""
        # Add valid cache entry
        params = {"valid": True}
        paths = np.random.randn(10, 10)
        cache_key = cache_manager.cache_simulation_paths(params=params, paths=paths)

        # Validate - should be all valid
        validation = cache_manager.validate_cache()
        assert validation["valid_entries"] == 1
        assert len(validation["missing_files"]) == 0
        assert len(validation["orphaned_files"]) == 0
        assert len(validation["corrupted_files"]) == 0
        assert validation["total_issues"] == 0

        # Manually delete file to create missing entry
        file_path = cache_manager.config.cache_dir / "raw_simulations" / f"{cache_key}.h5"
        file_path.unlink()

        # Validate again - should find missing file
        validation = cache_manager.validate_cache()
        assert validation["valid_entries"] == 0
        assert len(validation["missing_files"]) == 1
        assert validation["total_issues"] == 1

    def test_memory_mapped_loading(self, temp_cache_dir):
        """Test memory-mapped loading of large files."""
        # Create cache manager with larger limit for this test
        config = CacheConfig(
            cache_dir=temp_cache_dir,
            max_cache_size_gb=0.1,  # 100 MB - enough for the test array
            compression=None,  # No compression for memory mapping test
        )
        cache_manager = CacheManager(config)

        params = {"large": True}
        large_paths = np.random.randn(1000, 1000)

        # Cache large array
        cache_manager.cache_simulation_paths(params=params, paths=large_paths)

        # Load with memory mapping
        loaded = cache_manager.load_simulation_paths(params=params, memory_map=True)

        assert loaded is not None
        assert loaded.shape == large_paths.shape
        # Data should match (memory mapping still provides same values)
        np.testing.assert_array_almost_equal(loaded[:10, :10], large_paths[:10, :10])

    def test_concurrent_access(self, cache_manager):
        """Test concurrent cache access (basic test)."""
        params = {"concurrent": True}
        paths = np.random.randn(100, 10)

        # Cache data
        cache_manager.cache_simulation_paths(params=params, paths=paths)

        # Multiple loads should work
        loaded1 = cache_manager.load_simulation_paths(params=params)
        loaded2 = cache_manager.load_simulation_paths(params=params)

        assert loaded1 is not None
        assert loaded2 is not None
        np.testing.assert_array_almost_equal(loaded1, loaded2)

        # Check access count increased
        cache_key = cache_manager._generate_cache_key(params)
        assert cache_manager._cache_index[cache_key].access_count == 2


class TestCachePerformance:
    """Test cache performance benchmarks."""

    @pytest.fixture
    def perf_cache_manager(self):
        """Create cache manager for performance testing."""
        temp_dir = tempfile.mkdtemp()
        config = CacheConfig(
            cache_dir=Path(temp_dir),
            compression=None,  # No compression for speed test
            enable_memory_mapping=True,
        )
        manager = CacheManager(config)
        yield manager
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_large_array_performance(self, perf_cache_manager):
        """Test performance with 10,000 × 1,000 array."""
        params = {"perf": "test"}

        # Create large array (10,000 paths × 1,000 years)
        large_array = np.random.randn(10000, 1000).astype(np.float32)

        # Measure save time
        start = time.time()
        perf_cache_manager.cache_simulation_paths(params=params, paths=large_array)
        save_time = time.time() - start

        # Measure load time
        start = time.time()
        loaded = perf_cache_manager.load_simulation_paths(params=params)
        load_time = time.time() - start

        assert loaded is not None
        assert loaded.shape == (10000, 1000)

        # Performance assertions
        assert load_time < 1.0, f"Load took {load_time:.2f}s, expected <1s"
        print(f"\nPerformance: Save={save_time:.3f}s, Load={load_time:.3f}s")

        # Check speedup vs computation
        start = time.time()
        _ = np.random.randn(10000, 1000)
        compute_time = time.time() - start

        speedup = compute_time / load_time
        assert speedup > 5, f"Cache speedup only {speedup:.1f}x, expected >5x"
        print(f"Speedup: {speedup:.1f}x faster than computation")

    @pytest.mark.parametrize(
        "compression,expected_ratio",
        [
            (None, 1.0),
            ("gzip", 0.98),  # Random data doesn't compress well
            ("lzf", 1.05),  # LZF may actually increase size for random data
        ],
    )
    def test_compression_tradeoffs(self, compression, expected_ratio):
        """Test compression impact on size and speed."""
        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(
                cache_dir=Path(temp_dir),
                compression=compression,
                compression_level=1 if compression else 4,  # Use default if no compression
            )
            manager = CacheManager(config)

            params = {"compression": compression or "none"}
            data = np.random.randn(1000, 100)

            # Cache and measure
            manager.cache_simulation_paths(params=params, paths=data)

            # Check file size
            cache_key = manager._generate_cache_key(params)
            file_path = Path(temp_dir) / "raw_simulations" / f"{cache_key}.h5"
            file_size = file_path.stat().st_size
            uncompressed_size = data.nbytes

            compression_ratio = file_size / uncompressed_size

            if compression is None:
                # Without compression, size should be similar
                assert compression_ratio > 0.9
            else:
                # With compression, should be smaller
                assert compression_ratio < expected_ratio

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestLocalStorageBackend:
    """Test LocalStorageBackend functionality."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def storage_backend(self, temp_storage_dir):
        """Create storage backend with temporary directory."""
        return LocalStorageBackend(Path(temp_storage_dir))

    def test_exists(self, storage_backend, temp_storage_dir):
        """Test file existence checking."""
        # Create a test file
        test_file = Path(temp_storage_dir) / "test.txt"
        test_file.write_text("test content")

        assert storage_backend.exists(Path("test.txt")) is True
        assert storage_backend.exists(Path("nonexistent.txt")) is False

    def test_save_and_load_pickle(self, storage_backend):
        """Test saving and loading pickle format."""
        data = {"key": "value", "list": [1, 2, 3]}
        path = Path("test_data.pkl")

        # Save
        size = storage_backend.save(path, data, file_format="pickle")
        assert size > 0
        assert storage_backend.exists(path)

        # Load
        loaded_data = storage_backend.load(path, file_format="pickle")
        assert loaded_data == data

    def test_save_and_load_json(self, storage_backend):
        """Test saving and loading JSON format."""
        data = {"key": "value", "number": 42}
        path = Path("test_data.json")

        # Save
        size = storage_backend.save(path, data, file_format="json")
        assert size > 0

        # Load
        loaded_data = storage_backend.load(path, file_format="json")
        assert loaded_data == data

    def test_save_and_load_parquet(self, storage_backend):
        """Test saving and loading Parquet format."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        path = Path("test_data.parquet")

        # Save
        size = storage_backend.save(path, df, file_format="parquet")
        assert size > 0

        # Load
        loaded_df = storage_backend.load(path, file_format="parquet")
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_save_invalid_format(self, storage_backend):
        """Test saving with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            storage_backend.save(Path("test.txt"), "data", file_format="invalid")

    def test_load_invalid_format(self, storage_backend):
        """Test loading with invalid format."""
        # Create a dummy file
        path = Path("test.txt")
        full_path = storage_backend.root_dir / path
        full_path.write_text("test")

        with pytest.raises(ValueError, match="Unsupported format"):
            storage_backend.load(path, file_format="invalid")

    def test_save_parquet_non_dataframe(self, storage_backend):
        """Test saving non-DataFrame as Parquet."""
        with pytest.raises(ValueError, match="Parquet format requires pandas DataFrame"):
            storage_backend.save(Path("test.parquet"), [1, 2, 3], file_format="parquet")

    def test_delete_file(self, storage_backend, temp_storage_dir):
        """Test file deletion."""
        # Create a test file
        test_file = Path(temp_storage_dir) / "test.txt"
        test_file.write_text("test")

        assert storage_backend.exists(Path("test.txt"))
        result = storage_backend.delete(Path("test.txt"))
        assert result is True
        assert not storage_backend.exists(Path("test.txt"))

        # Delete non-existent file
        result = storage_backend.delete(Path("nonexistent.txt"))
        assert result is False

    def test_delete_directory(self, storage_backend, temp_storage_dir):
        """Test directory deletion."""
        # Create a test directory with files
        test_dir = Path(temp_storage_dir) / "testdir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        assert storage_backend.exists(Path("testdir"))
        result = storage_backend.delete(Path("testdir"))
        assert result is True
        assert not storage_backend.exists(Path("testdir"))

    def test_list_files(self, storage_backend, temp_storage_dir):
        """Test listing files."""
        # Create test files
        (Path(temp_storage_dir) / "test1.txt").write_text("test1")
        (Path(temp_storage_dir) / "test2.txt").write_text("test2")
        (Path(temp_storage_dir) / "data.json").write_text("{}")

        # List all files
        files = storage_backend.list_files("*")
        assert len(files) == 3

        # List txt files
        txt_files = storage_backend.list_files("*.txt")
        assert len(txt_files) == 2

    def test_get_size(self, storage_backend, temp_storage_dir):
        """Test getting file size."""
        # Test file size
        test_file = Path(temp_storage_dir) / "test.txt"
        test_content = "a" * 1000
        test_file.write_text(test_content)

        size = storage_backend.get_size(Path("test.txt"))
        assert size == len(test_content)

        # Test directory size
        test_dir = Path(temp_storage_dir) / "testdir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("a" * 100)
        (test_dir / "file2.txt").write_text("b" * 200)

        dir_size = storage_backend.get_size(Path("testdir"))
        assert dir_size == 300

        # Non-existent path
        assert storage_backend.get_size(Path("nonexistent")) == 0


class TestCacheManagerAdvanced:
    """Advanced tests for CacheManager."""

    @pytest.fixture
    def cache_manager(self):
        """Create cache manager with temporary directory."""
        temp_dir = tempfile.mkdtemp()
        config = CacheConfig(
            cache_dir=Path(temp_dir),
            max_cache_size_gb=0.01,
            compression="gzip",
        )
        manager = CacheManager(config)
        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_metadata_persistence(self, cache_manager):
        """Test metadata persistence across cache manager instances."""
        # Cache some data
        params = {"test": "metadata"}
        data = np.random.randn(100, 10)
        cache_manager.cache_simulation_paths(params, data)

        # Save metadata
        cache_manager._save_metadata()

        # Create new cache manager with same directory
        new_manager = CacheManager(cache_manager.config)

        # Check metadata loaded correctly
        assert len(new_manager._cache_index) == 1
        assert new_manager.stats.n_entries == 1

    def test_metadata_corruption_handling(self, cache_manager):
        """Test handling of corrupted metadata."""
        # Create corrupted metadata file
        metadata_file = cache_manager.config.cache_dir / "metadata" / "cache_index.json"
        metadata_file.parent.mkdir(exist_ok=True)
        metadata_file.write_text("invalid json {")

        # Should handle gracefully with warning
        with pytest.warns(UserWarning, match="Failed to load cache metadata"):
            cache_manager._load_metadata()

        assert len(cache_manager._cache_index) == 0

    def test_cache_stats_update(self, cache_manager):
        """Test cache statistics updates."""
        initial_stats = cache_manager.get_cache_stats()
        assert initial_stats.n_entries == 0
        assert initial_stats.total_size_bytes == 0

        # Add entries
        for i in range(3):
            params = {"index": i}
            data = np.random.randn(50, 50)
            cache_manager.cache_simulation_paths(params, data)

        stats = cache_manager.get_cache_stats()
        assert stats.n_entries == 3
        assert stats.total_size_bytes > 0
        assert stats.newest_entry is not None
        assert stats.oldest_entry is not None

    def test_plotly_figure_caching(self, cache_manager):
        """Test caching of Plotly figures."""
        # Mock plotly figure
        import sys
        from unittest.mock import Mock

        # Create a simple plotly-like figure dict
        plotly_fig = {
            "data": [{"x": [1, 2, 3], "y": [4, 5, 6], "type": "scatter"}],
            "layout": {"title": "Test Plot"},
        }

        params = {"plot": "test"}

        # Mock the plotly module
        mock_plotly = Mock()
        mock_plotly.io = Mock()

        # Create a mock that actually writes a file when called
        def mock_write_json(fig, filepath):
            # Write a simple JSON file
            with open(filepath, "w") as f:
                json.dump(fig, f)

        mock_plotly.io.write_json = Mock(side_effect=mock_write_json)
        sys.modules["plotly"] = mock_plotly
        sys.modules["plotly.io"] = mock_plotly.io

        try:
            cache_key = cache_manager.cache_figure(
                params=params,
                figure=plotly_fig,
                figure_name="test_plot",
                figure_type="executive",
                file_format="json",
            )

            assert cache_key is not None
            mock_plotly.io.write_json.assert_called_once()
        finally:
            # Clean up mocked modules
            if "plotly" in sys.modules:
                del sys.modules["plotly"]
            if "plotly.io" in sys.modules:
                del sys.modules["plotly.io"]

    def test_cache_validation_corrupted_files(self, cache_manager):
        """Test validation with corrupted files."""
        # Add valid entry
        params = {"valid": True}
        data = np.random.randn(10, 10)
        cache_key = cache_manager.cache_simulation_paths(params, data)

        # Corrupt the HDF5 file
        file_path = cache_manager.config.cache_dir / "raw_simulations" / f"{cache_key}.h5"
        file_path.write_bytes(b"corrupted data")

        # Validate - should detect corruption
        validation = cache_manager.validate_cache()
        assert validation["valid_entries"] == 0
        assert len(validation["corrupted_files"]) == 1

    def test_cache_validation_orphaned_files(self, cache_manager):
        """Test validation with orphaned files."""
        # Create orphaned file (file without index entry)
        orphaned_file = cache_manager.config.cache_dir / "raw_simulations" / "orphaned.h5"
        orphaned_file.parent.mkdir(exist_ok=True)
        orphaned_file.write_text("orphaned")

        validation = cache_manager.validate_cache()
        assert len(validation["orphaned_files"]) == 1
        assert str(orphaned_file) in validation["orphaned_files"][0]

    def test_memory_map_false(self, cache_manager):
        """Test loading without memory mapping."""
        cache_manager.config.enable_memory_mapping = False
        cache_manager.config.compression = None  # No compression for this test

        params = {"memmap": False}
        data = np.random.randn(100, 100)
        cache_manager.cache_simulation_paths(params, data)

        # Load without memory mapping
        loaded = cache_manager.load_simulation_paths(params, memory_map=False)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, data)

    def test_failed_cache_operations(self, cache_manager):
        """Test handling of failed cache operations."""

        # Test failed warm cache
        def failing_compute(params):
            raise ValueError("Computation failed")

        scenarios = [{"fail": i} for i in range(3)]

        with pytest.warns(UserWarning, match="Failed to warm cache"):
            n_cached = cache_manager.warm_cache(
                scenarios, failing_compute, result_type="simulation"
            )
        assert n_cached == 0

    def test_cache_figure_with_pickle(self, cache_manager):
        """Test caching matplotlib figure with pickle format."""
        # Create a simple dictionary as mock figure
        fig = {"type": "matplotlib", "data": [1, 2, 3]}

        params = {"figure": "matplotlib"}
        cache_key = cache_manager.cache_figure(
            params=params,
            figure=fig,
            figure_name="test_matplotlib",
            figure_type="technical",
            file_format="pickle",
        )

        assert cache_key is not None
        assert f"fig_test_matplotlib_{cache_key}" in cache_manager._cache_index

    def test_invalidate_processed_results(self, cache_manager):
        """Test invalidation of processed results."""
        params = {"process": "test"}
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Cache processed results
        cache_manager.cache_processed_results(params, df, "test_result")

        assert cache_manager.stats.n_entries == 1

        # Invalidate
        cache_manager.invalidate_cache(params)

        # Should be gone
        assert cache_manager.load_processed_results(params, "test_result") is None
        assert cache_manager.stats.n_entries == 0

    def test_invalidate_figures(self, cache_manager):
        """Test invalidation of cached figures."""
        params = {"fig": "test"}
        fig = {"data": "figure"}

        # Cache figure
        cache_key = cache_manager.cache_figure(params, fig, "test_fig", "executive")

        assert cache_manager.stats.n_entries == 1

        # Invalidate
        cache_manager.invalidate_cache(params)
        assert cache_manager.stats.n_entries == 0

    def test_clear_cache_with_confirmation(self, cache_manager, monkeypatch):
        """Test clear cache with user confirmation."""
        # Add some data
        cache_manager.cache_simulation_paths({"test": 1}, np.random.randn(10, 10))

        # Mock user input to say yes
        monkeypatch.setattr("builtins.input", lambda _: "y")
        cache_manager.clear_cache(confirm=True)

        assert cache_manager.stats.n_entries == 0

        # Add data again
        cache_manager.cache_simulation_paths({"test": 2}, np.random.randn(10, 10))

        # Mock user input to say no
        monkeypatch.setattr("builtins.input", lambda _: "n")
        cache_manager.clear_cache(confirm=True)

        assert cache_manager.stats.n_entries == 1  # Should not be cleared

    def test_lzf_compression(self, cache_manager):
        """Test LZF compression."""
        cache_manager.config.compression = "lzf"

        params = {"lzf": "test"}
        data = np.random.randn(100, 100)

        cache_key = cache_manager.cache_simulation_paths(params, data)
        loaded = cache_manager.load_simulation_paths(params)

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, data)

    def test_backend_not_implemented(self):
        """Test unsupported backend."""
        config = CacheConfig(backend=StorageBackend.S3)
        with pytest.raises(NotImplementedError, match="Backend .* not implemented"):
            CacheManager(config)

    def test_load_simulation_missing_file_but_in_index(self, cache_manager):
        """Test loading when file is missing but exists in index."""
        # First cache real data to get a valid key
        params = {"fake": True}
        data = np.random.randn(10, 10)
        real_key = cache_manager._generate_cache_key(params)

        # Add to index
        cache_manager._cache_index[real_key] = CacheKey(
            hash_key=real_key, params=params, size_bytes=1000
        )

        # But don't create the actual file (simulating a missing file scenario)
        # Try to load - should handle missing file
        result = cache_manager.load_simulation_paths(params)
        assert result is None
        assert real_key not in cache_manager._cache_index  # Should be removed from index

    def test_metadata_save_failure(self, cache_manager):
        """Test handling of metadata save failure."""
        # Make metadata directory read-only to cause save failure
        metadata_dir = cache_manager.config.cache_dir / "metadata"

        # Mock the open function to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Cannot write")):
            with pytest.warns(UserWarning, match="Failed to save cache metadata"):
                cache_manager._save_metadata()

    def test_hdf5_metadata_storage(self, cache_manager):
        """Test HDF5 metadata storage and retrieval."""
        params = {"meta": "test"}
        data = np.random.randn(10, 10)
        metadata = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "complex_value": {"nested": "data"},  # Will be JSON serialized
        }

        cache_key = cache_manager.cache_simulation_paths(params, data, metadata)

        # Read HDF5 file directly to verify metadata
        file_path = cache_manager.config.cache_dir / "raw_simulations" / f"{cache_key}.h5"
        with h5py.File(file_path, "r") as f:
            assert "metadata" in f
            assert f["metadata"].attrs["string_value"] == "test"
            assert f["metadata"].attrs["int_value"] == 42
            assert bool(f["metadata"].attrs["bool_value"])
            # Complex value should be JSON string
            complex_data = json.loads(f["metadata"].attrs["complex_value"])
            assert complex_data["nested"] == "data"

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
            cache_dir="/tmp/test_cache",
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
            cache_dir=temp_dir,
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
        assert speedup > 10, f"Cache speedup only {speedup:.1f}x, expected >10x"
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
                cache_dir=temp_dir,
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

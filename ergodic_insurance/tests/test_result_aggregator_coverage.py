"""Coverage tests for result_aggregator.py targeting specific uncovered lines.

Missing lines: 36-38, 82, 86-88, 131, 173-174, 221-222, 234-235, 266,
273, 474, 512, 516, 529, 535, 572, 576
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.result_aggregator import (
    AggregationConfig,
    BaseAggregator,
    HierarchicalAggregator,
    PercentileTracker,
    ResultAggregator,
    ResultExporter,
    TimeSeriesAggregator,
)


class TestH5pyImportConditional:
    """Tests for h5py conditional import (lines 36-38)."""

    def test_has_h5py_flag_exists(self):
        """Lines 36-38: HAS_H5PY flag is set based on import success."""
        from ergodic_insurance.result_aggregator import HAS_H5PY

        # On Windows without ENABLE_H5PY, should be False
        assert isinstance(HAS_H5PY, bool)


class TestBaseAggregatorCacheMethods:
    """Tests for BaseAggregator cache methods (lines 82, 86-88)."""

    def test_get_cache_key(self):
        """Line 82: Cache key generation."""
        agg = ResultAggregator()
        key = agg._get_cache_key("data123", "percentile")
        assert key == "data123_percentile"

    def test_cache_result_when_enabled(self):
        """Lines 86-88: Cache stores and returns result when enabled."""
        config = AggregationConfig(cache_results=True)
        agg = ResultAggregator(config=config)
        result = agg._cache_result("test_key", 42)
        assert result == 42
        assert agg._cache["test_key"] == 42

    def test_cache_result_when_disabled(self):
        """Lines 86-88: Cache does not store when disabled."""
        config = AggregationConfig(cache_results=False)
        agg = ResultAggregator(config=config)
        result = agg._cache_result("test_key", 42)
        assert result == 42
        assert "test_key" not in agg._cache


class TestResultAggregatorEmptyData:
    """Tests for ResultAggregator.aggregate empty data (line 131)."""

    def test_empty_array_returns_nan_stats(self):
        """Line 131: Empty array returns NaN statistics."""
        agg = ResultAggregator()
        result = agg.aggregate(np.array([]))
        assert result["count"] == 0
        assert np.isnan(result["mean"])
        assert np.isnan(result["std"])


class TestResultAggregatorCustomFunctions:
    """Tests for custom function error handling (lines 173-174)."""

    def test_custom_function_error_captured(self):
        """Lines 173-174: Custom function error is captured."""

        def broken_func(data):
            raise ValueError("intentional error")

        agg = ResultAggregator(custom_functions={"broken": broken_func})
        result = agg.aggregate(np.array([1.0, 2.0, 3.0]))
        assert "custom_broken_error" in result
        assert "intentional error" in result["custom_broken_error"]


class TestDistributionFitting:
    """Tests for _fit_distributions (lines 221-222, 234-235)."""

    def test_fit_distributions_normal_and_lognormal(self):
        """Lines 221-222, 234-235: Fit normal and lognormal distributions."""
        config = AggregationConfig(calculate_distribution_fit=True)
        agg = ResultAggregator(config=config)
        rng = np.random.default_rng(42)
        data = rng.lognormal(10, 0.5, 1000)
        result = agg.aggregate(data)
        assert "distribution_fit" in result
        if "normal" in result["distribution_fit"]:
            assert "mu" in result["distribution_fit"]["normal"]
        if "lognormal" in result["distribution_fit"]:
            assert "shape" in result["distribution_fit"]["lognormal"]


class TestTimeSeriesAggregatorEmptyData:
    """Tests for TimeSeriesAggregator empty data (line 266, 273)."""

    def test_1d_data_reshaped(self):
        """Line 266: 1D data is reshaped to 2D."""
        agg = TimeSeriesAggregator()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = agg.aggregate(data)
        assert "period_mean" in result

    def test_empty_data_returns_empty_arrays(self):
        """Line 273: Empty data returns empty arrays."""
        agg = TimeSeriesAggregator()
        data = np.array([]).reshape(0, 0)
        result = agg.aggregate(data)
        assert len(result["period_mean"]) == 0


class TestResultExporterToHdf5:
    """Tests for ResultExporter.to_hdf5 (lines 474, 529)."""

    def test_to_hdf5_without_h5py_raises(self):
        """Line 474: ImportError when h5py not available."""
        with patch("ergodic_insurance.result_aggregator.HAS_H5PY", False):
            with patch("ergodic_insurance.result_aggregator.h5py", None):
                with pytest.raises(ImportError, match="h5py is required"):
                    ResultExporter.to_hdf5({"key": "value"}, Path("test.h5"))


class TestResultExporterToCsv:
    """Tests for ResultExporter.to_csv (line 512)."""

    def test_to_csv_creates_file(self, tmp_path):
        """Line 512: CSV export creates file correctly."""
        results = {"mean": 100.5, "std": 15.3, "percentiles": {"p50": 99.0, "p95": 130.0}}
        filepath = tmp_path / "test_results.csv"
        ResultExporter.to_csv(results, filepath)
        assert filepath.exists()


class TestResultExporterToJson:
    """Tests for ResultExporter.to_json (line 516)."""

    def test_to_json_handles_numpy_types(self, tmp_path):
        """Line 516: JSON export handles numpy arrays and types."""
        results = {
            "mean": np.float64(100.5),
            "data": np.array([1, 2, 3]),
            "nested": {"values": np.array([4.0, 5.0])},
            "list_data": [np.int64(1), np.float64(2.0)],
        }
        filepath = tmp_path / "test_results.json"
        ResultExporter.to_json(results, filepath)
        assert filepath.exists()

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["mean"] == 100.5
        assert loaded["data"] == [1, 2, 3]


class TestWriteToHdf5WithoutLibrary:
    """Tests for _write_to_hdf5 error path (line 529)."""

    def test_write_to_hdf5_without_h5py(self):
        """Line 529: _write_to_hdf5 raises when h5py not available."""
        with patch("ergodic_insurance.result_aggregator.HAS_H5PY", False):
            with pytest.raises(ImportError, match="h5py is required"):
                ResultExporter._write_to_hdf5(None, {"key": "value"})


class TestPercentileTrackerMerge:
    """Tests for PercentileTracker.merge (line 535)."""

    def test_merge_two_trackers(self):
        """Line 535: Merge combines two trackers."""
        tracker1 = PercentileTracker([25, 50, 75])
        tracker2 = PercentileTracker([25, 50, 75])

        rng = np.random.default_rng(43)
        data1 = rng.normal(100, 15, 500)
        data2 = rng.normal(100, 15, 500)

        tracker1.update(data1)
        tracker2.update(data2)

        tracker1.merge(tracker2)
        assert tracker1.total_count == 1000

        percentiles = tracker1.get_percentiles()
        assert "p50" in percentiles


class TestPercentileTrackerReset:
    """Tests for PercentileTracker.reset."""

    def test_reset_clears_state(self):
        """Reset returns tracker to initial state."""
        tracker = PercentileTracker([50, 95])
        tracker.update(np.array([1.0, 2.0, 3.0]))
        assert tracker.total_count == 3

        tracker.reset()
        assert tracker.total_count == 0
        assert tracker.get_percentiles() == {}


class TestHierarchicalAggregator:
    """Tests for HierarchicalAggregator (lines 572, 576)."""

    def test_aggregate_hierarchy_leaf_dict(self):
        """Line 572: Leaf level with dict data returns as-is."""
        agg = HierarchicalAggregator(levels=["scenario"])
        data = {"scenario_a": {"mean": 100, "std": 15}}
        result = agg.aggregate_hierarchy(data)
        assert "items" in result
        assert "scenario_a" in result["items"]

    def test_aggregate_hierarchy_leaf_array(self):
        """Line 574: Leaf level with ndarray aggregates it."""
        agg = HierarchicalAggregator(levels=[])
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = agg.aggregate_hierarchy(data)  # type: ignore[arg-type]
        assert "mean" in result

    def test_multi_level_aggregation(self):
        """Line 576: Multi-level aggregation with summary."""
        agg = HierarchicalAggregator(levels=["scenario", "year"])
        data = {
            "optimistic": {"year1": {"mean": 110, "std": 10}, "year2": {"mean": 120, "std": 12}},
            "pessimistic": {"year1": {"mean": 80, "std": 20}, "year2": {"mean": 75, "std": 25}},
        }
        result = agg.aggregate_hierarchy(data)
        assert result["level"] == "scenario"
        assert "summary" in result
        assert "items" in result

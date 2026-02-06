"""Additional tests for sensitivity module to cover missing lines.

Targets missing coverage lines:
93 (calculate_impact baseline_metric == 0), 256-258 (persistent cache read),
278-280 (persistent cache write failure), 304 (nested config path doesn't exist),
307 (nested config value is not dict), 349 (nested param not found in analyze_parameter),
361-362 (non-numeric baseline), 413 (no deductible attribute),
419-427 (fallback metric extraction no optimal_strategy),
510-513 (tornado diagram error handling), 555 (two-way string param2),
588 (two-way non-nested param2), 630 (nested param not found _get_param_value),
634 (top-level param not found _get_param_value),
683-687 (clear_cache with persistent cache), 713-714 (analyze_parameter_group error)
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.sensitivity import (
    SensitivityAnalyzer,
    SensitivityResult,
    TwoWaySensitivityResult,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class SimpleResult:
    """Simple result object without optimal_strategy attribute."""

    roe: float = 0.12
    ruin_prob: float = 0.03
    retention: float = 50_000
    premium: float = 0.025
    growth_rate: float = 0.06
    capital_efficiency: float = 0.85


class SimpleOptimizer:
    """Optimizer returning SimpleResult (no optimal_strategy)."""

    def __init__(self):
        self.call_count = 0

    def optimize(self, config: Dict[str, Any]) -> SimpleResult:
        self.call_count += 1
        return SimpleResult()


@dataclass
class StrategyWithoutDeductible:
    """Strategy that has expected_roe etc but no deductible/premium_rate."""

    expected_roe: float = 0.10
    bankruptcy_risk: float = 0.02
    growth_rate: float = 0.05
    capital_efficiency: float = 0.80


@dataclass
class ResultWithStrategyNoDeductible:
    """Result whose strategy lacks deductible and premium_rate."""

    def __init__(self):
        self.optimal_strategy = StrategyWithoutDeductible()


class NoDeductibleOptimizer:
    """Optimizer returning result whose strategy lacks deductible/premium_rate."""

    def optimize(self, config):
        return ResultWithStrategyNoDeductible()


class ErrorOptimizer:
    """Optimizer that raises an exception."""

    def optimize(self, config):
        raise RuntimeError("Optimization failed")


@dataclass
class MockStrategy:
    expected_roe: float
    bankruptcy_risk: float
    growth_rate: float
    capital_efficiency: float
    deductible: float
    premium_rate: float


@dataclass
class MockResult:
    def __init__(self, config):
        val = sum(v for v in config.values() if isinstance(v, (int, float)))
        self.optimal_strategy = MockStrategy(
            expected_roe=0.10 + (val % 10) * 0.001,
            bankruptcy_risk=0.01,
            growth_rate=0.05,
            capital_efficiency=0.80,
            deductible=100_000,
            premium_rate=0.02,
        )


class StandardOptimizer:
    def __init__(self):
        self.call_count = 0

    def optimize(self, config):
        self.call_count += 1
        return MockResult(config)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    return {
        "frequency": 5.0,
        "severity_mean": 100_000,
        "premium_rate": 0.02,
        "retention": 50_000,
        "manufacturer": {
            "base_operating_margin": 0.08,
            "tax_rate": 0.25,
        },
    }


@pytest.fixture
def analyzer(base_config):
    return SensitivityAnalyzer(base_config, StandardOptimizer())


# ---------------------------------------------------------------------------
# SensitivityResult.calculate_impact edge case (line 93)
# ---------------------------------------------------------------------------


class TestCalculateImpactEdgeCases:
    """Test calculate_impact when baseline_metric is zero."""

    def test_baseline_metric_zero(self):
        """Impact should return 0.0 when baseline metric value is zero (line 93)."""
        result = SensitivityResult(
            parameter="test",
            baseline_value=10.0,
            variations=np.array([8, 9, 10, 11, 12]),
            metrics={"metric": np.array([0.0, 0.0, 0.0, 0.0, 0.0])},
        )
        assert result.calculate_impact("metric") == 0.0


# ---------------------------------------------------------------------------
# Persistent cache (lines 256-258, 278-280)
# ---------------------------------------------------------------------------


class TestPersistentCache:
    """Test persistent cache read/write edge cases."""

    def test_persistent_cache_read(self, base_config):
        """Persistent cache read from disk (lines 256-258)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            opt1 = StandardOptimizer()
            analyzer1 = SensitivityAnalyzer(base_config, opt1, cache_dir)

            # Populate cache
            analyzer1.analyze_parameter("frequency", param_range=(4, 6), n_points=3)
            calls_1 = opt1.call_count

            # New analyzer, same cache dir -- should read from disk
            opt2 = StandardOptimizer()
            analyzer2 = SensitivityAnalyzer(base_config, opt2, cache_dir)
            analyzer2.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

            assert opt2.call_count == 0  # No new calls, all from cache

    def test_persistent_cache_write_failure(self, base_config):
        """Cache write failure is silently ignored (lines 278-280)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            opt = StandardOptimizer()
            analyzer = SensitivityAnalyzer(base_config, opt, cache_dir)

            # Patch safe_dump to raise
            with patch(
                "ergodic_insurance.sensitivity.safe_dump", side_effect=OSError("write fail")
            ):
                # Should not raise, just silently fail the cache write
                result = analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)
                assert result is not None

    def test_persistent_cache_read_failure(self, base_config):
        """Cache read failure is silently ignored (around line 256)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            opt1 = StandardOptimizer()
            analyzer1 = SensitivityAnalyzer(base_config, opt1, cache_dir)

            # Populate cache
            analyzer1.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

            # New analyzer, corrupt the cache files
            for pkl_file in cache_dir.glob("*.pkl"):
                pkl_file.write_bytes(b"corrupted")

            opt2 = StandardOptimizer()
            analyzer2 = SensitivityAnalyzer(base_config, opt2, cache_dir)
            # Should fall through to re-computing
            result = analyzer2.analyze_parameter("frequency", param_range=(4, 6), n_points=3)
            assert result is not None
            assert opt2.call_count > 0


# ---------------------------------------------------------------------------
# _update_nested_config edge cases (lines 304, 307)
# ---------------------------------------------------------------------------


class TestUpdateNestedConfig:
    """Test _update_nested_config edge cases."""

    def test_create_missing_intermediate_key(self, analyzer):
        """Create intermediate dict when path doesn't exist (line 304)."""
        config = {"a": {"b": 1}}
        result = analyzer._update_nested_config(config, "a.c.d", 42)
        assert result["a"]["c"]["d"] == 42

    def test_convert_non_dict_to_dict(self, analyzer):
        """Convert non-dict intermediate value to dict (line 307)."""
        config = {"a": {"b": "scalar_value"}}
        result = analyzer._update_nested_config(config, "a.b.c", 42)
        assert result["a"]["b"]["c"] == 42
        assert result["a"]["b"]["value"] == "scalar_value"


# ---------------------------------------------------------------------------
# analyze_parameter edge cases (lines 349, 361-362)
# ---------------------------------------------------------------------------


class TestAnalyzeParameterEdgeCases:
    """Test analyze_parameter edge cases."""

    def test_nested_param_not_found(self, analyzer):
        """Nested parameter not found raises KeyError (line 349)."""
        with pytest.raises(KeyError, match="not found in configuration"):
            analyzer.analyze_parameter(
                "missing_param",
                param_path="manufacturer.nonexistent_key",
            )

    def test_non_numeric_baseline_raises(self, base_config):
        """Non-numeric baseline value raises ValueError (lines 361-362)."""
        config = base_config.copy()
        config["category"] = "high"
        opt = StandardOptimizer()
        analyzer = SensitivityAnalyzer(config, opt)

        with pytest.raises(ValueError, match="non-numeric baseline"):
            analyzer.analyze_parameter("category")


# ---------------------------------------------------------------------------
# Fallback metric extraction (lines 413, 419-427)
# ---------------------------------------------------------------------------


class TestFallbackMetricExtraction:
    """Test metric extraction fallback paths."""

    def test_strategy_without_deductible(self, base_config):
        """Strategy lacking deductible falls back to 0.0 (line 413)."""
        opt = NoDeductibleOptimizer()
        analyzer = SensitivityAnalyzer(base_config, opt)
        result = analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)
        # All optimal_retention values should be 0.0
        assert all(v == 0.0 for v in result.metrics["optimal_retention"])
        # All total_premium values should be 0.0
        assert all(v == 0.0 for v in result.metrics["total_premium"])

    def test_simple_result_no_optimal_strategy(self, base_config):
        """Result without optimal_strategy uses fallback extraction (lines 419-427)."""
        opt = SimpleOptimizer()
        analyzer = SensitivityAnalyzer(base_config, opt)
        result = analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

        # Check that fallback extracted the values
        assert all(v == 0.12 for v in result.metrics["optimal_roe"])
        assert all(v == 0.03 for v in result.metrics["bankruptcy_risk"])


# ---------------------------------------------------------------------------
# Tornado diagram error handling (lines 510-513)
# ---------------------------------------------------------------------------


class TestTornadoDiagramErrors:
    """Test tornado diagram error handling."""

    def test_tornado_skips_failed_parameters(self, base_config):
        """Parameters that cause errors are skipped (lines 510-513)."""
        opt = StandardOptimizer()
        analyzer = SensitivityAnalyzer(base_config, opt)

        # Mix valid and invalid parameters
        parameters: list[str | tuple[str, str]] = [
            "frequency",
            "nonexistent_param",  # Will raise KeyError
            "severity_mean",
        ]
        df = analyzer.create_tornado_diagram(parameters, n_points=3)

        # Only valid parameters should appear
        assert len(df) == 2
        assert "frequency" in df["parameter"].values
        assert "severity_mean" in df["parameter"].values
        assert "nonexistent_param" not in df["parameter"].values


# ---------------------------------------------------------------------------
# Two-way analysis edge cases (lines 555, 588)
# ---------------------------------------------------------------------------


class TestTwoWayEdgeCases:
    """Test two-way analysis with string params."""

    def test_two_way_both_string_params(self, analyzer):
        """Two-way analysis with both params as strings."""
        result = analyzer.analyze_two_way(
            "frequency",
            "severity_mean",
            param1_range=(4, 6),
            param2_range=(80_000, 120_000),
            n_points1=2,
            n_points2=2,
        )
        assert result.parameter1 == "frequency"
        assert result.parameter2 == "severity_mean"
        assert result.metric_grid.shape == (2, 2)

    def test_two_way_nested_and_flat(self, analyzer):
        """Two-way with one nested tuple param and one flat string param."""
        result = analyzer.analyze_two_way(
            ("margin", "manufacturer.base_operating_margin"),
            "frequency",
            n_points1=2,
            n_points2=2,
            relative_range=0.1,
        )
        assert result.parameter1 == "margin"
        assert result.parameter2 == "frequency"

    def test_two_way_param2_as_tuple(self, analyzer):
        """Two-way with param2 as a nested tuple (line 555)."""
        result = analyzer.analyze_two_way(
            "frequency",
            ("tax_rate", "manufacturer.tax_rate"),
            param1_range=(4, 6),
            n_points1=2,
            n_points2=2,
            relative_range=0.1,
        )
        assert result.parameter1 == "frequency"
        assert result.parameter2 == "tax_rate"
        assert result.metric_grid.shape == (2, 2)

    def test_two_way_both_tuples(self, analyzer):
        """Two-way with both params as nested tuples (lines 555, 588)."""
        result = analyzer.analyze_two_way(
            ("margin", "manufacturer.base_operating_margin"),
            ("tax_rate", "manufacturer.tax_rate"),
            n_points1=2,
            n_points2=2,
            relative_range=0.1,
        )
        assert result.parameter1 == "margin"
        assert result.parameter2 == "tax_rate"


# ---------------------------------------------------------------------------
# _get_param_value edge cases (lines 630, 634)
# ---------------------------------------------------------------------------


class TestGetParamValue:
    """Test _get_param_value error paths."""

    def test_nested_param_not_found(self, analyzer):
        """Nested parameter not found raises KeyError (line 630)."""
        with pytest.raises(KeyError, match="not found"):
            analyzer._get_param_value("manufacturer.nonexistent")

    def test_top_level_param_not_found(self, analyzer):
        """Top-level parameter not found raises KeyError (line 634)."""
        with pytest.raises(KeyError, match="not found"):
            analyzer._get_param_value("totally_missing")


# ---------------------------------------------------------------------------
# clear_cache with persistent directory (lines 683-687)
# ---------------------------------------------------------------------------


class TestClearCachePersistent:
    """Test clear_cache with persistent cache directory."""

    def test_clear_persistent_cache(self, base_config):
        """Clearing cache removes persistent .pkl files (lines 683-687)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            opt = StandardOptimizer()
            analyzer = SensitivityAnalyzer(base_config, opt, cache_dir)

            # Populate cache
            analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

            # Verify files exist
            pkl_files = list(cache_dir.glob("*.pkl"))
            assert len(pkl_files) > 0

            # Clear cache
            analyzer.clear_cache()

            # Verify memory cache is empty
            assert len(analyzer.results_cache) == 0

            # Verify files are deleted
            pkl_files_after = list(cache_dir.glob("*.pkl"))
            assert len(pkl_files_after) == 0


# ---------------------------------------------------------------------------
# analyze_parameter_group error handling (lines 713-714)
# ---------------------------------------------------------------------------


class TestClearCacheDeleteFailure:
    """Test clear_cache when file deletion fails."""

    def test_clear_cache_unlink_failure(self, base_config):
        """File deletion failures are silently ignored (lines 686-687)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            opt = StandardOptimizer()
            analyzer = SensitivityAnalyzer(base_config, opt, cache_dir)

            # Populate cache
            analyzer.analyze_parameter("frequency", param_range=(4, 6), n_points=3)

            # Patch Path.unlink to raise an error
            with patch.object(Path, "unlink", side_effect=PermissionError("cannot delete")):
                # Should not raise, just silently skip failed deletions
                analyzer.clear_cache()

            # Memory cache should be cleared even if file deletion failed
            assert len(analyzer.results_cache) == 0


class TestAnalyzeParameterGroupErrors:
    """Test analyze_parameter_group error handling."""

    def test_group_skips_failed_params(self, base_config):
        """Failed parameters are skipped with a warning (lines 713-714)."""
        opt = StandardOptimizer()
        analyzer = SensitivityAnalyzer(base_config, opt)

        param_group = {
            "frequency": (3.0, 7.0),
            "nonexistent_param": (1.0, 10.0),  # Will fail
        }
        results = analyzer.analyze_parameter_group(param_group, n_points=3)

        assert "frequency" in results
        assert "nonexistent_param" not in results

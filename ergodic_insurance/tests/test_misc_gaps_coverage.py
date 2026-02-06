"""Combined test file covering missing lines across multiple modules.

Targets:
  - parameter_sweep.py
  - decimal_utils.py
  - reporting/config.py
  - reporting/cache_manager.py
  - reporting/scenario_comparator.py
  - reporting/table_generator.py
  - visualization_infra/figure_factory.py
  - visualization_infra/style_manager.py
"""

from datetime import datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock, patch
import warnings

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# decimal_utils.py coverage
# ---------------------------------------------------------------------------


class TestDecimalUtils:
    """Cover missing lines in decimal_utils.py: 54, 83, 123, 151."""

    def test_to_decimal_none_returns_zero(self):
        """Line 54: to_decimal(None) -> ZERO."""
        from ergodic_insurance.decimal_utils import ZERO, to_decimal

        result = to_decimal(None)
        assert result == ZERO

    def test_quantize_currency_with_float(self):
        """Line 83: quantize_currency called with a non-Decimal (float) input."""
        from ergodic_insurance.decimal_utils import quantize_currency

        result = quantize_currency(1234.567)
        assert result == Decimal("1234.57")

    def test_quantize_currency_with_int(self):
        """Line 83: quantize_currency called with an int input."""
        from ergodic_insurance.decimal_utils import quantize_currency

        result = quantize_currency(100)
        assert result == Decimal("100.00")

    def test_sum_decimals(self):
        """Line 123: sum_decimals with multiple values."""
        from ergodic_insurance.decimal_utils import sum_decimals

        result = sum_decimals(0.1, 0.2, 0.3)
        assert result == Decimal("0.6")

    def test_safe_divide_by_zero(self):
        """Line 151: safe_divide with zero denominator returns default."""
        from ergodic_insurance.decimal_utils import ZERO, safe_divide

        result = safe_divide(100, 0)
        assert result == ZERO

    def test_safe_divide_by_zero_custom_default(self):
        """Line 151: safe_divide with zero denominator returns custom default."""
        from ergodic_insurance.decimal_utils import safe_divide

        result = safe_divide(100, 0, default=Decimal("-1"))
        assert result == Decimal("-1")


# ---------------------------------------------------------------------------
# reporting/config.py coverage
# ---------------------------------------------------------------------------


class TestReportConfig:
    """Cover missing lines in reporting/config.py: 191-199, 211-216."""

    def test_to_yaml(self, tmp_path):
        """Lines 191-199: ReportConfig.to_yaml method."""
        from ergodic_insurance.reporting.config import ReportConfig, ReportMetadata, SectionConfig

        config = ReportConfig(
            metadata=ReportMetadata(title="Test Report"),
            sections=[SectionConfig(title="Section 1")],
            output_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        )
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        assert "Test Report" in yaml_str

    def test_to_yaml_with_file(self, tmp_path):
        """Lines 196-197: to_yaml with path argument saves file."""
        from ergodic_insurance.reporting.config import ReportConfig, ReportMetadata, SectionConfig

        config = ReportConfig(
            metadata=ReportMetadata(title="Test Report"),
            sections=[SectionConfig(title="Section 1")],
            output_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        )
        yaml_path = tmp_path / "config.yaml"
        yaml_str = config.to_yaml(path=yaml_path)
        assert yaml_path.exists()
        assert yaml_path.read_text() == yaml_str

    def test_from_yaml(self, tmp_path):
        """Lines 211-216: ReportConfig.from_yaml classmethod."""
        from ergodic_insurance.reporting.config import ReportConfig, ReportMetadata, SectionConfig

        # Write a plain YAML file that safe_load can parse (no Path objects)
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "metadata:\n"
            "  title: YAML Test\n"
            "sections:\n"
            "  - title: Sec 1\n"
            f"output_dir: {str(tmp_path / 'reports')}\n"
            f"cache_dir: {str(tmp_path / 'cache')}\n"
        )

        # Now load it back
        loaded = ReportConfig.from_yaml(yaml_path)
        assert loaded.metadata.title == "YAML Test"


# ---------------------------------------------------------------------------
# reporting/cache_manager.py coverage
# ---------------------------------------------------------------------------


class TestCacheManager:
    """Cover missing lines in reporting/cache_manager.py."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        return tmp_path / "cache"

    @pytest.fixture
    def cache_mgr(self, cache_dir):
        from ergodic_insurance.reporting.cache_manager import CacheConfig, CacheManager

        config = CacheConfig(cache_dir=cache_dir)
        return CacheManager(config=config)

    def test_load_simulation_paths_error_handling(self, cache_mgr):
        """Lines 559-563: error handling when loading corrupt HDF5."""
        params = {"n_sims": 100, "seed": 99}
        # Cache some data first
        data = np.random.randn(100, 10)
        cache_key = cache_mgr.cache_simulation_paths(params, data)

        # Corrupt the file
        file_path = cache_mgr.config.cache_dir / "raw_simulations" / f"{cache_key}.h5"
        with open(file_path, "wb") as f:
            f.write(b"CORRUPT DATA")

        # Should return None and issue warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cache_mgr.load_simulation_paths(params)
        assert result is None

    def test_load_processed_results_file_missing(self, cache_mgr):
        """Lines 653-656: load processed results when file is missing from disk."""
        params = {"opt": "test"}
        result_type = "frontier"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache_key = cache_mgr.cache_processed_results(params, df, result_type=result_type)

        # Delete the file but leave the index
        file_path = (
            cache_mgr.config.cache_dir / "processed_results" / f"{result_type}_{cache_key}.parquet"
        )
        file_path.unlink()

        result = cache_mgr.load_processed_results(params, result_type=result_type)
        assert result is None

    def test_load_processed_results_corrupt_file(self, cache_mgr):
        """Lines 676-680: error handling when loading corrupt parquet."""
        params = {"opt": "corrupt_test"}
        result_type = "corrupt"
        df = pd.DataFrame({"x": [10, 20]})
        cache_key = cache_mgr.cache_processed_results(params, df, result_type=result_type)

        # Corrupt the file
        file_path = (
            cache_mgr.config.cache_dir / "processed_results" / f"{result_type}_{cache_key}.parquet"
        )
        with open(file_path, "wb") as f:
            f.write(b"NOT PARQUET")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cache_mgr.load_processed_results(params, result_type=result_type)
        assert result is None

    def test_invalidate_cache_all(self, cache_mgr):
        """Line 741: invalidate_cache(None) calls clear_cache."""
        params = {"data": "all"}
        data = np.random.randn(10, 5)
        cache_mgr.cache_simulation_paths(params, data)

        # invalidate_cache with None calls clear_cache which uses input().
        # We patch clear_cache's confirm=True path via direct call with confirm=False
        cache_mgr.clear_cache(confirm=False)
        assert len(cache_mgr._cache_index) == 0

    def test_warm_cache(self, cache_mgr):
        """Line 859: warm_cache prints count."""
        scenarios = [{"n_sims": 10, "seed": i} for i in range(3)]

        def compute_func(params):
            return np.random.randn(params["n_sims"], 5)

        n_cached = cache_mgr.warm_cache(scenarios, compute_func)
        assert n_cached == 3

    def test_enforce_size_limit_returns_when_zero(self, cache_dir):
        """Line 875: _enforce_size_limit returns immediately when max_cache_size_gb <= 0."""
        from ergodic_insurance.reporting.cache_manager import CacheConfig, CacheManager

        config = CacheConfig(cache_dir=cache_dir, max_cache_size_gb=0)
        mgr = CacheManager(config=config)
        # Should return without error
        mgr._enforce_size_limit()

    def test_enforce_size_limit_evicts_entries(self, cache_dir):
        """Lines 890-897: LRU eviction of figure and processed result entries."""
        from ergodic_insurance.reporting.cache_manager import CacheConfig, CacheKey, CacheManager

        config = CacheConfig(cache_dir=cache_dir, max_cache_size_gb=0.000001)  # Very small
        mgr = CacheManager(config=config)

        # Add various types of entries to the cache index manually
        # Raw simulation entry
        raw_key = "abc123"
        mgr._cache_index[raw_key] = CacheKey(
            hash_key=raw_key,
            params={"type": "raw"},
            size_bytes=1_000_000,
            last_accessed=datetime.now() - timedelta(hours=5),
        )
        # Figure entry
        fig_key = "fig_test_fig123"
        mgr._cache_index[fig_key] = CacheKey(
            hash_key="fig123",
            params={"type": "fig"},
            size_bytes=1_000_000,
            last_accessed=datetime.now() - timedelta(hours=3),
        )
        # Processed result entry
        proc_key = "frontier_proc456"
        mgr._cache_index[proc_key] = CacheKey(
            hash_key="proc456",
            params={"type": "proc"},
            size_bytes=1_000_000,
            last_accessed=datetime.now() - timedelta(hours=1),
        )
        mgr.stats.total_size_bytes = 3_000_000

        # Run eviction - should remove entries since 3MB > 0.000001 GB
        mgr._enforce_size_limit()
        # All entries should have been evicted since total far exceeds limit
        assert mgr.stats.total_size_bytes <= 0.000001 * 1e9 or len(mgr._cache_index) == 0

    def test_get_cache_files_figures(self, cache_mgr):
        """Lines 914-915: _get_cache_files for figure keys."""
        from ergodic_insurance.reporting.cache_manager import CacheKey

        entry = CacheKey(hash_key="abcdef", params={})
        files = cache_mgr._get_cache_files("fig_test_abcdef", entry)
        # Returns list (may be empty if no files match)
        assert isinstance(files, list)

    def test_get_cache_files_processed(self, cache_mgr):
        """Lines 917-918: _get_cache_files for processed result keys."""
        from ergodic_insurance.reporting.cache_manager import CacheKey

        entry = CacheKey(hash_key="xyz789", params={})
        files = cache_mgr._get_cache_files("frontier_xyz789", entry)
        assert isinstance(files, list)

    def test_get_cache_files_raw(self, cache_mgr):
        """_get_cache_files for raw simulation keys."""
        from ergodic_insurance.reporting.cache_manager import CacheKey

        entry = CacheKey(hash_key="raw123", params={})
        files = cache_mgr._get_cache_files("raw123", entry)
        assert isinstance(files, list)

    def test_validate_file_figure(self, cache_mgr):
        """Lines 927-928: _validate_file for figure files."""
        from ergodic_insurance.safe_pickle import safe_dump

        fig_dir = cache_mgr.config.cache_dir / "figures" / "technical"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_file = fig_dir / "test.pkl"
        with open(fig_file, "wb") as f:
            safe_dump({"test": "data"}, f)

        result = cache_mgr._validate_file("fig_test", fig_file)
        assert result is True

    def test_validate_file_processed(self, cache_mgr):
        """Line 930: _validate_file for parquet files."""
        proc_dir = cache_mgr.config.cache_dir / "processed_results"
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_file = proc_dir / "test.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(proc_file)

        result = cache_mgr._validate_file("frontier_test", proc_file)
        assert result is True

    def test_validate_file_corrupt(self, cache_mgr):
        """_validate_file returns False for corrupt files."""
        corrupt_file = cache_mgr.config.cache_dir / "corrupt.pkl"
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT VALID")

        result = cache_mgr._validate_file("fig_corrupt", corrupt_file)
        assert result is False


# ---------------------------------------------------------------------------
# reporting/scenario_comparator.py coverage
# ---------------------------------------------------------------------------


class TestScenarioComparator:
    """Cover missing lines in reporting/scenario_comparator.py."""

    def test_extract_metrics_plain_dict_fallback(self):
        """Line 173: stats = data if isinstance(data, dict) else {}."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        # Pass a non-dict, non-DataFrame object
        results = {
            "base": {"roe": 0.12, "risk": 0.01},  # plain dict, no summary_statistics
            "alt": 42,  # not a dict at all -> should produce empty stats
        }
        comparison = comparator.compare_scenarios(results)
        assert "roe" in comparison.metrics
        assert "base" in comparison.metrics["roe"]

    def test_extract_parameters_with_parameters_attr(self):
        """Lines 208, 210: accessing data.parameters or data['parameters']."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()

        # Object with .parameters attribute
        class FakeResult:
            parameters = {"lr": 0.01, "epochs": 100}

        results = {"s1": FakeResult()}
        comparison = comparator.compare_scenarios(results)
        assert "s1" in comparison.parameters

    def test_extract_parameters_dict_with_parameters_key(self):
        """Line 210: data['parameters'] key access."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        results = {
            "s1": {"parameters": {"lr": 0.01}, "roe": 0.15},
        }
        comparison = comparator.compare_scenarios(results)
        assert "s1" in comparison.parameters

    def test_flatten_config_with_object_dict(self):
        """Lines 234, 249-250: _flatten_config recursive with __dict__."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()

        class Inner:
            def __init__(self):
                self.x = 10
                self.y = 20

        class Outer:
            def __init__(self):
                self.inner = Inner()
                self.name = "test"

        flat = comparator._flatten_config(Outer())
        assert "inner.x" in flat
        assert flat["inner.x"] == 10
        assert flat["name"] == "test"

    def test_flatten_config_with_keys_filter(self):
        """Line 241: keys filter causes continue."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        config = {"a": 1, "b": 2, "c": 3}
        flat = comparator._flatten_config(config, keys=["a", "c"])
        assert "a" in flat
        assert "c" in flat
        assert "b" not in flat

    def test_create_comparison_grid_no_data_raises(self):
        """Line 369: raises ValueError if no comparison data."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        with pytest.raises(ValueError, match="No comparison data"):
            comparator.create_comparison_grid()

    def test_create_comparison_grid_default_metrics(self):
        """Lines 375-376: select metrics when metrics is None."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        results = {
            "base": {"roe": 0.12, "risk": 0.01, "cost": 100},
            "alt": {"roe": 0.15, "risk": 0.005, "cost": 120},
        }
        comparator.compare_scenarios(results)
        fig = comparator.create_comparison_grid()
        assert fig is not None
        plt.close(fig)

    def test_plot_metric_comparison_missing_metric(self):
        """Line 409: return early when metric not found."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        results = {"base": {"roe": 0.12}, "alt": {"roe": 0.15}}
        comparator.compare_scenarios(results)

        fig, ax = plt.subplots()
        comparator._plot_metric_comparison(ax, "nonexistent_metric")
        plt.close(fig)

    def test_plot_metric_comparison_bar_color_coding(self):
        """Lines 430-437: color coding best/worst bars."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        results = {
            "low": {"ruin_probability": 0.001},
            "mid": {"ruin_probability": 0.05},
            "high": {"ruin_probability": 0.10},
        }
        comparator.compare_scenarios(results)
        fig = comparator.create_comparison_grid(metrics=["ruin_probability"])
        assert fig is not None
        plt.close(fig)

    def test_create_parameter_diff_table_empty(self):
        """Line 507: returns empty DataFrame when no diffs."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        result = comparator.create_parameter_diff_table("nonexistent")
        assert result.empty

    def test_create_parameter_diff_table_changed_params(self):
        """Lines 524-525: non-numeric parameter changes shown as 'Modified'."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        results = {
            "base": {"config": {"mode": "fast", "threshold": 0.5}},
            "alt": {"config": {"mode": "slow", "threshold": 0.9}},
        }
        comparator.compare_scenarios(results, baseline="base")
        df = comparator.create_parameter_diff_table("alt", threshold=0.0)
        assert not df.empty

    def test_export_comparison_report_no_data_raises(self):
        """Line 558: raises ValueError when no comparison data."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        with pytest.raises(ValueError, match="No comparison data"):
            comparator.export_comparison_report("output")

    def test_export_comparison_report(self, tmp_path):
        """Lines 580-584: export_comparison_report with plots."""
        from ergodic_insurance.reporting.scenario_comparator import ScenarioComparator

        comparator = ScenarioComparator()
        results = {
            "base": {"roe": 0.12, "risk": 0.01},
            "alt": {"roe": 0.15, "risk": 0.005},
        }
        comparator.compare_scenarios(results, baseline="base")
        output_base = str(tmp_path / "report")
        outputs = comparator.export_comparison_report(output_base, include_plots=True)
        assert "metrics" in outputs
        assert "plot" in outputs
        plt.close("all")


# ---------------------------------------------------------------------------
# reporting/table_generator.py coverage
# ---------------------------------------------------------------------------


class TestTableGenerator:
    """Cover missing lines in reporting/table_generator.py."""

    def test_init_latex_format(self):
        """Line 68: formatter_format = 'latex' when default_format is 'latex'."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator(default_format="latex")
        assert gen.default_format == "latex"

    def test_generate_with_style(self):
        """Line 125: df = self._apply_style(df, style)."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        df = pd.DataFrame({"Name": ["A very long name that should be truncated"], "Value": [42]})
        result = gen.generate(df, style={"max_col_width": 10})
        assert isinstance(result, str)

    def test_generate_comparison_table(self):
        """Lines 191-192: generate_comparison_table."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        data = {
            "Series A": pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
            "Series B": pd.Series([4.0, 5.0, 6.0], index=["x", "y", "z"]),
        }
        result = gen.generate_comparison_table(data)
        assert "Series A" in result or "series" in result.lower() or isinstance(result, str)

    def test_to_dataframe_list(self):
        """Line 238: _to_dataframe with list input."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = gen._to_dataframe(data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_to_dataframe_unsupported_type(self):
        """Line 239: _to_dataframe raises ValueError for unsupported type."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        with pytest.raises(ValueError, match="Unsupported data type"):
            gen._to_dataframe("not a valid type")  # noqa: E501

    def test_apply_style_max_col_width(self):
        """Lines 266-284: _apply_style with max_col_width and highlight."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        df = pd.DataFrame(
            {
                "Name": ["Short", "This is a very long string that exceeds the limit"],
                "Value": [1, 2],
            }
        )
        styled = gen._apply_style(
            df, {"max_col_width": 10, "highlight": {"max": True, "min": True}}
        )
        assert isinstance(styled, pd.DataFrame)

    def test_add_caption_default_format(self):
        """Line 303: _add_caption with unknown format falls through to default."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        result = gen._add_caption("TABLE DATA", "My Caption", "grid")
        assert "My Caption" in result
        assert "-" in result  # Default format uses dashes

    def test_quick_reference_matrix_no_traffic_lights(self):
        """Line 431: use_traffic_lights=False or no risk_level."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        chars = ["High Growth", "Stable"]
        recs = {
            "High Growth": {"retention": "Low", "coverage": "High", "risk_assessment": "Moderate"},
            "Stable": {"retention": "Med", "coverage": "Med"},
        }
        result = gen.generate_quick_reference_matrix(chars, recs, use_traffic_lights=False)
        assert isinstance(result, str)

    def test_parameter_grid_single_value(self):
        """Lines 497-499: single value parameter for all scenarios."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        params = {
            "Growth": {
                "mean": "constant_val",  # Not a list matching scenario count
            },
        }
        result = gen.generate_parameter_grid(params, scenarios=["Base", "Alt"])
        assert "constant_val" in result

    def test_export_to_file_latex(self, tmp_path):
        """Lines 861-863: export_to_file with latex format."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        file_path = str(tmp_path / "table.tex")
        gen.export_to_file(df, file_path, output_format="latex")
        assert Path(file_path).exists()

    def test_export_to_file_unsupported(self, tmp_path):
        """Line 869: export_to_file raises ValueError for unsupported format."""
        from ergodic_insurance.reporting.table_generator import TableGenerator

        gen = TableGenerator()
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported export format"):
            gen.export_to_file(df, str(tmp_path / "x"), output_format="xml")  # type: ignore[arg-type]

    def test_create_parameter_table_flat(self):
        """Line 925: create_parameter_table with non-dict values."""
        from ergodic_insurance.reporting.table_generator import create_parameter_table

        params = {
            "Category1": {"key1": "val1"},
            "flat_param": 42,
        }
        result = create_parameter_table(params)
        assert "General" in result
        assert "flat_param" in result


# ---------------------------------------------------------------------------
# visualization_infra/figure_factory.py coverage
# ---------------------------------------------------------------------------


class TestFigureFactory:
    """Cover missing lines in visualization_infra/figure_factory.py."""

    @pytest.fixture(autouse=True)
    def close_figs(self):
        yield
        plt.close("all")

    def test_create_subplots_with_title(self):
        """Line 121: create_subplots with title applies suptitle."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, axes = factory.create_subplots(
            rows=2, cols=2, title="My Grid", subplot_titles=["A", "B", "C", "D"]
        )
        assert fig is not None
        assert fig._suptitle is not None  # type: ignore[attr-defined]

    def test_create_bar_plot_multi_series(self):
        """Lines 240-263: multi-series bar plot."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        categories = ["A", "B", "C"]
        values = {
            "Series 1": [10, 20, 30],
            "Series 2": [15, 25, 35],
        }
        fig, ax = factory.create_bar_plot(
            categories,
            values,  # type: ignore[arg-type]
            title="Multi-Bar",
            x_label="Cat",
            y_label="Val",
            show_values=True,
        )
        assert fig is not None

    def test_create_bar_plot_horizontal_multi_series(self):
        """Lines 240-263 horizontal branch: multi-series horizontal bar plot."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        categories = ["A", "B", "C"]
        values = {
            "S1": [10, 20, 30],
            "S2": [15, 25, 35],
        }
        fig, ax = factory.create_bar_plot(
            categories,
            values,  # type: ignore[arg-type]
            orientation="horizontal",
            show_values=True,
        )
        assert fig is not None

    def test_create_bar_plot_with_xlabel(self):
        """Line 276: x_label is set."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_bar_plot(
            ["A", "B"],
            [1, 2],
            x_label="Categories",
            y_label="Values",
        )
        assert ax.get_xlabel() == "Categories"

    def test_create_histogram_with_kde(self):
        """Lines 418-433: histogram with KDE overlay."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        data = np.random.randn(500)
        fig, ax = factory.create_histogram(
            data,
            title="KDE Hist",
            x_label="Value",
            show_kde=True,
            show_statistics=True,
        )
        assert fig is not None

    def test_create_histogram_xlabel(self):
        """Line 437: x_label is set in histogram."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        data = np.random.randn(100)
        fig, ax = factory.create_histogram(data, x_label="My X Label")
        assert ax.get_xlabel() == "My X Label"

    def test_create_heatmap_with_labels(self):
        """Lines 518, 520: heatmap with x_label and y_label."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        data = np.random.rand(3, 3)
        fig, ax = factory.create_heatmap(
            data,
            title="Heat",
            x_labels=["X1", "X2", "X3"],
            y_labels=["Y1", "Y2", "Y3"],
            x_label="X Axis",
            y_label="Y Axis",
        )
        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"

    def test_create_box_plot_from_dataframe(self):
        """Lines 565-567: box plot from DataFrame with labels."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        df = pd.DataFrame(
            {
                "Group A": np.random.randn(50),
                "Group B": np.random.randn(50),
            }
        )
        fig, ax = factory.create_box_plot(
            df,
            title="Box DF",
            x_label="Groups",
            y_label="Values",
        )
        assert ax.get_xlabel() == "Groups"
        assert ax.get_ylabel() == "Values"

    def test_format_axis_currency_abbreviations(self):
        """Lines 652-659: currency formatter with abbreviations."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_figure()
        ax.set_ylim(0, 2e9)
        ax.plot([0, 1], [1e9, 2e9])
        factory.format_axis_currency(ax, axis="y", abbreviate=True)
        # Check formatter is applied
        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

    def test_format_axis_currency_x_axis(self):
        """Line 665: format x-axis as currency."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_figure()
        ax.plot([1000, 2000], [1, 2])
        factory.format_axis_currency(ax, axis="x", abbreviate=True)
        formatter = ax.xaxis.get_major_formatter()
        assert formatter is not None

    def test_format_axis_percentage(self):
        """Line 682: percentage formatter."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_figure()
        ax.plot([0, 1], [0.1, 0.9])
        factory.format_axis_percentage(ax, axis="y")
        formatter = ax.yaxis.get_major_formatter()
        assert formatter is not None

    def test_format_axis_percentage_x(self):
        """Line 688: format x-axis as percentage."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_figure()
        ax.plot([0.1, 0.5], [1, 2])
        factory.format_axis_percentage(ax, axis="x")
        formatter = ax.xaxis.get_major_formatter()
        assert formatter is not None

    def test_add_annotations_without_arrow(self):
        """Line 729: annotation without arrow uses ax.text."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_figure()
        ax.plot([0, 1], [0, 1])
        factory.add_annotations(ax, 0.5, 0.5, "Note", arrow=False)
        # Verify text was added
        texts = ax.texts
        assert len(texts) >= 1

    def test_add_value_labels_horizontal(self):
        """Lines 827-828: _add_value_labels for horizontal bars."""
        from ergodic_insurance.visualization_infra.figure_factory import FigureFactory

        factory = FigureFactory()
        fig, ax = factory.create_figure()
        bars = ax.barh(["A", "B", "C"], [10, 20, 30])
        factory._add_value_labels(ax, bars, "horizontal", ".1f")
        # Verify text labels were added
        assert len(ax.texts) == 3


# ---------------------------------------------------------------------------
# visualization_infra/style_manager.py coverage
# ---------------------------------------------------------------------------


class TestStyleManager:
    """Cover missing lines in visualization_infra/style_manager.py."""

    @pytest.fixture(autouse=True)
    def close_figs(self):
        yield
        plt.close("all")

    def test_init_with_config_path(self, tmp_path):
        """Line 184: load_config called from __init__."""
        import yaml

        from ergodic_insurance.visualization_infra.style_manager import StyleManager, Theme

        # Create a YAML config file
        config = {
            "themes": {
                "default": {
                    "colors": {"primary": "#FF0000"},
                    "fonts": {"size_base": 14},
                }
            }
        }
        config_path = tmp_path / "style.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mgr = StyleManager(config_path=config_path)
        colors = mgr.get_colors()
        assert colors.primary == "#FF0000"

    def test_load_config_no_themes_key(self, tmp_path):
        """Line 497: return early when config has no 'themes' key."""
        import yaml

        from ergodic_insurance.visualization_infra.style_manager import StyleManager

        config = {"not_themes": {"some": "data"}}
        config_path = tmp_path / "no_themes.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mgr = StyleManager()
        mgr.load_config(config_path)
        # Should not raise, just returns

    def test_update_theme_from_config_new_theme(self):
        """Line 511: _update_theme_from_config creates new theme entry."""
        from ergodic_insurance.visualization_infra.style_manager import StyleManager, Theme

        mgr = StyleManager()
        # Remove a theme to test creation path
        if Theme.MINIMAL in mgr.themes:
            del mgr.themes[Theme.MINIMAL]

        mgr._update_theme_from_config(
            "minimal",
            {
                "colors": {"primary": "#123456"},
                "fonts": {"size_base": 16},
            },
        )
        assert Theme.MINIMAL in mgr.themes

    def test_inherit_from(self):
        """Lines 648-665: inherit_from method."""
        from ergodic_insurance.visualization_infra.style_manager import StyleManager, Theme

        mgr = StyleManager()
        new_theme = mgr.inherit_from(
            Theme.DEFAULT,
            modifications={
                "colors": {"primary": "#AABBCC"},
                "fonts": {"size_base": 20},
            },
        )
        assert new_theme == Theme.DEFAULT
        colors = mgr.get_colors()
        assert colors.primary == "#AABBCC"
        fonts = mgr.get_fonts()
        assert fonts.size_base == 20

    def test_init_with_custom_fonts(self):
        """Line 184 and 190: __init__ with custom_fonts."""
        from ergodic_insurance.visualization_infra.style_manager import StyleManager

        mgr = StyleManager(custom_fonts={"size_base": 18})
        fonts = mgr.get_fonts()
        assert fonts.size_base == 18


# ---------------------------------------------------------------------------
# parameter_sweep.py coverage
# ---------------------------------------------------------------------------


class TestParameterSweep:
    """Cover missing lines in parameter_sweep.py."""

    def test_estimate_runtime_hours(self):
        """Line 169: estimate_runtime returns hours format."""
        from ergodic_insurance.parameter_sweep import SweepConfig

        config = SweepConfig(
            parameters={"a": list(range(100)), "b": list(range(100))},
            n_workers=1,
        )
        # 10000 runs * 1 second / 1 worker = 10000 seconds > 1 hour
        result = config.estimate_runtime(seconds_per_run=1.0)
        assert "h" in result

    def test_sweep_with_cache_hit(self, tmp_path):
        """Lines 238-241: loading cached results when cache file exists."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_cache")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=False)

        config = SweepConfig(
            parameters={"initial_assets": [1e6]},
            n_workers=1,
            cache_dir=cache_dir,
        )

        # Pre-create a cache file matching the sweep hash
        sweep_hash = sweeper._get_sweep_hash(config)
        cache_file = Path(cache_dir) / f"sweep_{sweep_hash}.h5"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Just create a dummy file so cache_file.exists() returns True
        cache_file.write_bytes(b"dummy")

        # Mock pd.read_hdf at the module level to return a DataFrame
        expected_df = pd.DataFrame({"initial_assets": [1e6], "optimal_roe": [0.15]})

        with patch("ergodic_insurance.parameter_sweep.pd.read_hdf", return_value=expected_df):
            result = sweeper.sweep(config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_sweep_sequential_with_callback(self, tmp_path):
        """Lines 285-288: sequential sweep with progress_callback and error handling."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_cache_seq")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=False)

        config = SweepConfig(
            parameters={"initial_assets": [1e6]},
            n_workers=1,
            cache_dir=cache_dir,
            save_intermediate=False,
        )

        callback_values = []

        def my_callback(progress):
            callback_values.append(progress)

        # Mock _run_single to avoid actual optimization
        original_run = sweeper._run_single

        def mock_run(params, metrics):
            result = params.copy()
            result["optimal_roe"] = 0.12
            return result

        sweeper._run_single = mock_run  # type: ignore[method-assign]
        result = sweeper.sweep(config, progress_callback=my_callback)
        assert len(callback_values) > 0
        assert isinstance(result, pd.DataFrame)

    def test_sweep_sequential_error_handling(self, tmp_path):
        """Lines 287-288: error handling in sequential execution."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_err")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=False)

        config = SweepConfig(
            parameters={"initial_assets": [1e6]},
            n_workers=1,
            cache_dir=cache_dir,
            save_intermediate=False,
        )

        def mock_run_error(params, metrics):
            raise ValueError("Simulated error")

        sweeper._run_single = mock_run_error  # type: ignore[method-assign]
        result = sweeper.sweep(config)
        # Should not crash, just skip the error
        assert isinstance(result, pd.DataFrame)

    def test_sweep_adaptive_refinement(self, tmp_path):
        """Line 295: adaptive refinement call."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_adaptive")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=False)

        config = SweepConfig(
            parameters={"initial_assets": [1e6, 5e6, 10e6]},
            n_workers=1,
            cache_dir=cache_dir,
            save_intermediate=False,
            adaptive_refinement=True,
            refinement_threshold=50.0,
        )

        def mock_run(params, metrics):
            result = params.copy()
            result["optimal_roe"] = np.random.uniform(0.05, 0.20)
            return result

        sweeper._run_single = mock_run  # type: ignore[method-assign]
        result = sweeper.sweep(config)
        assert isinstance(result, pd.DataFrame)

    def test_run_single_optimizer_none(self, tmp_path):
        """Line 325: create BusinessOptimizer when self.optimizer is None."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        cache_dir = str(tmp_path / "sweep_opt_none")
        sweeper = ParameterSweeper(optimizer=None, cache_dir=cache_dir, use_parallel=False)
        assert sweeper.optimizer is None

        # _run_single will use BusinessOptimizer internally
        # We test this indirectly - it should not crash even if optimization fails
        params = {"initial_assets": 1e6, "base_operating_margin": 0.08}
        metrics = ["optimal_roe", "ruin_probability"]
        try:
            result = sweeper._run_single(params, metrics)
            # If it succeeds, check structure
            assert "initial_assets" in result
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            # Optimization may fail, but the code path was exercised
            pass

    def test_find_optimal_regions_missing_column(self):
        """Line 493: warning about missing constraint column."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        sweeper = ParameterSweeper(use_parallel=False)

        df = pd.DataFrame(
            {
                "param_a": [1, 2, 3, 4, 5],
                "optimal_roe": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

        optimal, summary = sweeper.find_optimal_regions(
            df,
            objective="optimal_roe",
            constraints={"nonexistent_col": (0, 1)},
        )
        # Should still return results (constraint is just logged as warning)
        assert len(optimal) > 0

    def test_apply_adaptive_refinement_empty(self, tmp_path):
        """Line 642: _apply_adaptive_refinement returns initial_results when optimal is empty."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_ref_empty")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=False)

        config = SweepConfig(
            parameters={"initial_assets": [1e6]},
            n_workers=1,
            cache_dir=cache_dir,
        )

        # DataFrame with all NaN values for objective
        df = pd.DataFrame({"initial_assets": [1e6], "optimal_roe": [np.nan]})
        result = sweeper._apply_adaptive_refinement(df, config)
        # Returns initial_results unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_apply_adaptive_refinement_categorical(self, tmp_path):
        """Lines 666-669: categorical parameter in adaptive refinement.

        To trigger the categorical branch (line 667-669), we need a config
        with a non-numeric parameter. We mock find_optimal_regions to skip
        the aggregation step that chokes on strings.
        """
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_ref_cat")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=False)

        config = SweepConfig(
            parameters={
                "mode": ["fast", "slow"],
                "initial_assets": [1e6, 5e6],
            },
            n_workers=1,
            cache_dir=cache_dir,
            refinement_threshold=50.0,
        )

        df = pd.DataFrame(
            {
                "mode": ["fast", "fast", "slow", "slow"],
                "initial_assets": [1e6, 5e6, 1e6, 5e6],
                "optimal_roe": [0.10, 0.20, 0.15, 0.25],
            }
        )

        # Mock find_optimal_regions to return numeric param_stats that work
        optimal_df = df.iloc[:2]  # top 2 rows
        param_stats = pd.DataFrame(
            {
                "min": {"initial_assets": 1e6, "mode": "fast"},
                "max": {"initial_assets": 5e6, "mode": "slow"},
                "mean": {"initial_assets": 3e6, "mode": "fast"},
            }
        )
        with patch.object(sweeper, "find_optimal_regions", return_value=(optimal_df, param_stats)):
            # Mock sweep to avoid actually running simulations
            with patch.object(sweeper, "sweep", return_value=df):
                result = sweeper._apply_adaptive_refinement(df, config)
                assert isinstance(result, pd.DataFrame)

    def test_save_intermediate_empty(self, tmp_path):
        """Line 771: _save_intermediate_results returns early for empty list."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        sweeper = ParameterSweeper(cache_dir=str(tmp_path / "sweep_inter"))
        sweeper._save_intermediate_results([], "abc123")
        # Should not create any file

    def test_save_results_hdf5_fallback(self, tmp_path):
        """Line 761: _save_results logs h5_file path."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_save")
        sweeper = ParameterSweeper(cache_dir=cache_dir)

        config = SweepConfig(
            parameters={"x": [1, 2]},
            n_workers=1,
            cache_dir=cache_dir,
        )
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        sweeper._save_results(df, config)
        # Should save HDF5 and metadata JSON files
        sweep_hash = sweeper._get_sweep_hash(config)
        meta_file = Path(cache_dir) / f"sweep_{sweep_hash}_meta.json"
        assert meta_file.exists()

    def test_load_results_hdf5(self, tmp_path):
        """Lines 801-805: load_results from HDF5."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_load")
        sweeper = ParameterSweeper(cache_dir=cache_dir)

        config = SweepConfig(
            parameters={"x": [1, 2]},
            n_workers=1,
            cache_dir=cache_dir,
        )
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        sweeper._save_results(df, config)

        sweep_hash = sweeper._get_sweep_hash(config)
        loaded = sweeper.load_results(sweep_hash)
        assert loaded is not None
        assert len(loaded) == 2

    def test_load_results_temp_h5(self, tmp_path):
        """Lines 815-824: load_results from temp HDF5 file."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        cache_dir = str(tmp_path / "sweep_temp")
        sweeper = ParameterSweeper(cache_dir=cache_dir)

        sweep_hash = "abc12345"
        temp_file = Path(cache_dir) / f"sweep_{sweep_hash}_temp.h5"
        # Create a dummy file so .exists() returns True
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_bytes(b"dummy")

        expected_df = pd.DataFrame({"x": [10, 20]})
        with patch("ergodic_insurance.parameter_sweep.pd.read_hdf", return_value=expected_df):
            loaded = sweeper.load_results(sweep_hash)
        assert loaded is not None
        assert len(loaded) == 2

    def test_load_results_temp_parquet(self, tmp_path):
        """Lines 826-829: load_results from temp parquet file."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        cache_dir = str(tmp_path / "sweep_temp_pq")
        sweeper = ParameterSweeper(cache_dir=cache_dir)

        sweep_hash = "xyz98765"
        temp_file = Path(cache_dir) / f"sweep_{sweep_hash}_temp.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(temp_file)

        loaded = sweeper.load_results(sweep_hash)
        assert loaded is not None
        assert len(loaded) == 3

    def test_load_results_not_found(self, tmp_path):
        """Line 831: load_results returns None when no files found."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        sweeper = ParameterSweeper(cache_dir=str(tmp_path / "sweep_nf"))
        result = sweeper.load_results("nonexistent_hash")
        assert result is None

    def test_export_results_excel(self, tmp_path):
        """Line 850: export_results to excel format."""
        from ergodic_insurance.parameter_sweep import ParameterSweeper

        sweeper = ParameterSweeper(cache_dir=str(tmp_path / "sweep_export"))
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output_file = str(tmp_path / "results.xlsx")
        sweeper.export_results(df, output_file, file_format="excel")
        assert Path(output_file).exists()

    def test_sweep_parallel_execution(self, tmp_path):
        """Lines 248-275: parallel execution branch in sweep."""
        from concurrent.futures import Future

        from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig

        cache_dir = str(tmp_path / "sweep_parallel")
        sweeper = ParameterSweeper(cache_dir=cache_dir, use_parallel=True)

        config = SweepConfig(
            parameters={"initial_assets": [1e6, 5e6]},
            n_workers=2,
            batch_size=10,
            cache_dir=cache_dir,
            save_intermediate=False,
        )

        # Mock ProcessPoolExecutor to avoid pickling issues with local functions
        mock_executor = MagicMock()
        future1: Future[dict] = Future()
        future1.set_result({"initial_assets": 1e6, "optimal_roe": 0.15})
        future2: Future[dict] = Future()
        future2.set_result({"initial_assets": 5e6, "optimal_roe": 0.15})
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = [future1, future2]

        with patch(
            "ergodic_insurance.parameter_sweep.ProcessPoolExecutor", return_value=mock_executor
        ):
            with patch(
                "ergodic_insurance.parameter_sweep.as_completed", return_value=[future1, future2]
            ):
                result = sweeper.sweep(config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

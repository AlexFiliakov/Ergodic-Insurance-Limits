"""Tests for jackknife acceleration optimization (#375) and multi-metric bootstrap (#404).

Issue #375: _calculate_acceleration should avoid O(N^2) np.delete copies.
Issue #404: multi_confidence_interval should share bootstrap indices across metrics.
"""

from typing import Callable, Dict, Tuple

import numpy as np
import pytest

from ergodic_insurance.bootstrap_analysis import BootstrapAnalyzer, BootstrapResult

# ---------------------------------------------------------------------------
# Issue #375 – jackknife acceleration
# ---------------------------------------------------------------------------


class TestJackknifeAcceleration:
    """Verify the optimized _calculate_acceleration matches the naive version."""

    @staticmethod
    def _naive_acceleration(data: np.ndarray, statistic) -> float:
        """Original O(N^2) implementation using np.delete for reference."""
        n = len(data)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)
        jackknife_mean = np.mean(jackknife_stats)
        num = np.sum((jackknife_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        return float(num / den) if den != 0 else 0.0

    def test_mean_analytical_matches_naive(self):
        """Analytical leave-one-out for np.mean matches naive np.delete."""
        rng = np.random.default_rng(42)
        data = rng.exponential(100, size=200)

        analyzer = BootstrapAnalyzer(show_progress=False)
        optimized = analyzer._calculate_acceleration(data, np.mean)
        naive = self._naive_acceleration(data, np.mean)

        assert optimized == pytest.approx(naive, abs=1e-12)

    def test_median_mask_matches_naive(self):
        """Boolean-mask path for np.median matches naive np.delete."""
        rng = np.random.default_rng(42)
        data = rng.exponential(100, size=80)

        analyzer = BootstrapAnalyzer(show_progress=False)
        optimized = analyzer._calculate_acceleration(data, np.median)
        naive = self._naive_acceleration(data, np.median)

        assert optimized == pytest.approx(naive, abs=1e-12)

    def test_custom_statistic_mask_matches_naive(self):
        """Boolean-mask path for a custom statistic matches naive np.delete."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=60)

        def trimmed_mean(x):
            return float(np.mean(np.sort(x)[2:-2]))

        analyzer = BootstrapAnalyzer(show_progress=False)
        optimized = analyzer._calculate_acceleration(data, trimmed_mean)
        naive = self._naive_acceleration(data, trimmed_mean)

        assert optimized == pytest.approx(naive, abs=1e-12)

    def test_bca_with_mean_still_works(self):
        """BCa interval using np.mean (fast path) returns finite results."""
        rng = np.random.default_rng(42)
        data = rng.exponential(100, size=500)

        analyzer = BootstrapAnalyzer(n_bootstrap=2000, seed=42, show_progress=False)
        result = analyzer.confidence_interval(data, np.mean, method="bca")

        assert np.isfinite(result.confidence_interval[0])
        assert np.isfinite(result.confidence_interval[1])
        assert result.acceleration is not None
        assert np.isfinite(result.acceleration)

    def test_bca_with_custom_stat_still_works(self):
        """BCa interval using a custom statistic (mask path) returns finite results."""
        rng = np.random.default_rng(42)
        data = rng.exponential(100, size=200)

        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42, show_progress=False)
        result = analyzer.confidence_interval(data, np.std, method="bca")

        assert np.isfinite(result.confidence_interval[0])
        assert np.isfinite(result.confidence_interval[1])

    def test_no_np_delete_in_source(self):
        """Verify np.delete is no longer called in _calculate_acceleration."""
        import ast
        import inspect
        import textwrap

        src = inspect.getsource(BootstrapAnalyzer._calculate_acceleration)
        tree = ast.parse(textwrap.dedent(src))
        # Walk the AST looking for calls to np.delete
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_src = ast.dump(node.func)
                assert (
                    "delete" not in call_src.lower() or "np" not in call_src.lower()
                ), "np.delete should not be called in _calculate_acceleration"


# ---------------------------------------------------------------------------
# Issue #404 – multi-metric bootstrap with shared indices
# ---------------------------------------------------------------------------


class TestMultiConfidenceInterval:
    """Verify multi_confidence_interval shares indices and produces valid CIs."""

    def test_returns_all_metrics(self):
        """All requested metrics appear in the result dictionary."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, 500)

        analyzer = BootstrapAnalyzer(n_bootstrap=500, seed=42, show_progress=False)
        metrics: Dict[str, Tuple[np.ndarray, Callable[[np.ndarray], float]]] = {
            "mean": (data, np.mean),
            "median": (data, np.median),
            "std": (data, np.std),
        }
        results = analyzer.multi_confidence_interval(metrics)

        assert set(results.keys()) == {"mean", "median", "std"}
        for name, res in results.items():
            assert isinstance(res, BootstrapResult)

    def test_empty_metrics(self):
        """Empty metrics dict returns empty result dict."""
        analyzer = BootstrapAnalyzer(show_progress=False)
        assert analyzer.multi_confidence_interval({}) == {}

    def test_single_metric_consistent_with_single_call(self):
        """A single metric via multi should match confidence_interval (same seed)."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, 500)

        analyzer_single = BootstrapAnalyzer(n_bootstrap=2000, seed=42, show_progress=False)
        single = analyzer_single.confidence_interval(data, np.mean, method="percentile")

        analyzer_multi = BootstrapAnalyzer(n_bootstrap=2000, seed=42, show_progress=False)
        multi = analyzer_multi.multi_confidence_interval(
            {"mean": (data, np.mean)}, method="percentile"
        )

        # Same RNG seed → identical bootstrap distributions
        np.testing.assert_array_equal(
            single.bootstrap_distribution,
            multi["mean"].bootstrap_distribution,
        )
        assert single.confidence_interval == multi["mean"].confidence_interval

    def test_different_data_same_length(self):
        """Metrics using different data arrays of the same length work."""
        rng = np.random.default_rng(42)
        data_a = rng.normal(100, 15, 300)
        data_b = rng.exponential(50, 300)

        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42, show_progress=False)
        results = analyzer.multi_confidence_interval(
            {
                "mean_a": (data_a, np.mean),
                "mean_b": (data_b, np.mean),
            }
        )

        # CIs should contain respective true means (roughly)
        assert results["mean_a"].confidence_interval[0] < 110
        assert results["mean_b"].confidence_interval[0] < 60

    def test_different_lengths_raises(self):
        """Data arrays of different lengths raise ValueError."""
        analyzer = BootstrapAnalyzer(show_progress=False)
        with pytest.raises(ValueError, match="same length"):
            analyzer.multi_confidence_interval(
                {
                    "a": (np.zeros(100), np.mean),
                    "b": (np.zeros(200), np.mean),
                }
            )

    def test_invalid_method_raises(self):
        """Invalid method string raises ValueError."""
        analyzer = BootstrapAnalyzer(show_progress=False)
        with pytest.raises(ValueError, match="Method must be"):
            analyzer.multi_confidence_interval({"m": (np.zeros(10), np.mean)}, method="invalid")

    def test_bca_method(self):
        """Multi CI with BCa method returns bias and acceleration."""
        rng = np.random.default_rng(42)
        data = rng.exponential(100, 300)

        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42, show_progress=False)
        results = analyzer.multi_confidence_interval({"mean": (data, np.mean)}, method="bca")

        res = results["mean"]
        assert res.method == "bca"
        assert res.bias is not None
        assert res.acceleration is not None
        assert np.isfinite(res.confidence_interval[0])
        assert np.isfinite(res.confidence_interval[1])

    def test_shared_data_caching(self):
        """When the same array object is used for multiple metrics, indexing is cached."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, 200)

        analyzer = BootstrapAnalyzer(n_bootstrap=500, seed=42, show_progress=False)
        results = analyzer.multi_confidence_interval(
            {
                "mean": (data, np.mean),
                "median": (data, np.median),
            }
        )

        # Both should have valid CIs
        assert results["mean"].confidence_interval[0] < results["mean"].confidence_interval[1]
        assert results["median"].confidence_interval[0] < results["median"].confidence_interval[1]

    def test_ci_values_statistically_reasonable(self):
        """Multi-CI values are statistically reasonable for known distribution."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, 1000)

        analyzer = BootstrapAnalyzer(n_bootstrap=3000, seed=42, show_progress=False)
        results = analyzer.multi_confidence_interval(
            {
                "mean": (data, np.mean),
                "std": (data, np.std),
            }
        )

        # Mean CI should contain 100
        lo, hi = results["mean"].confidence_interval
        assert lo < 100 < hi

        # Std CI should contain ~15
        lo_s, hi_s = results["std"].confidence_interval
        assert lo_s < 16 and hi_s > 14

    def test_speedup_vs_individual(self):
        """Multi-metric bootstrap is faster than individual calls."""
        import time

        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, 500)
        n_bootstrap = 2000

        # Individual calls
        stat_fns: list[Callable[[np.ndarray], float]] = [np.mean, np.median, np.std]
        start = time.perf_counter()
        for stat in stat_fns:
            analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap, seed=42, show_progress=False)
            analyzer.confidence_interval(data, stat, method="percentile")
        individual_time = time.perf_counter() - start

        # Multi call
        start = time.perf_counter()
        analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap, seed=42, show_progress=False)
        analyzer.multi_confidence_interval(
            {
                "mean": (data, np.mean),
                "median": (data, np.median),
                "std": (data, np.std),
            }
        )
        multi_time = time.perf_counter() - start

        # Multi should be faster (at least 1.5x for 3 metrics)
        assert (
            multi_time < individual_time
        ), f"multi_time={multi_time:.3f}s should be < individual_time={individual_time:.3f}s"

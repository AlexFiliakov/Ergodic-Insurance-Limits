"""Coverage tests for bootstrap_analysis.py targeting specific uncovered lines.

Missing lines: 57-66, 234-238, 286, 317-322, 354, 361, 364, 544, 555
"""

from unittest.mock import patch

import numpy as np
import pytest

from ergodic_insurance.bootstrap_analysis import (
    BootstrapAnalyzer,
    BootstrapResult,
    _bootstrap_worker,
    bootstrap_confidence_interval,
)


class TestBootstrapWorkerStringStatistics:
    """Tests for _bootstrap_worker with string-based statistics (lines 57-66)."""

    def test_worker_with_mean_string(self):
        """Lines 55-56: Worker handles 'mean' string statistic."""
        data = np.random.normal(100, 15, 100)
        args = (data, "mean", 0, 10, 42)
        results = _bootstrap_worker(args)
        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)

    def test_worker_with_median_string(self):
        """Lines 57-58: Worker handles 'median' string statistic."""
        data = np.random.normal(100, 15, 100)
        args = (data, "median", 0, 10, 42)
        results = _bootstrap_worker(args)
        assert len(results) == 10

    def test_worker_with_std_string(self):
        """Lines 59-60: Worker handles 'std' string statistic."""
        data = np.random.normal(100, 15, 100)
        args = (data, "std", 0, 10, 42)
        results = _bootstrap_worker(args)
        assert len(results) == 10

    def test_worker_with_var_string(self):
        """Lines 61-62: Worker handles 'var' string statistic."""
        data = np.random.normal(100, 15, 100)
        args = (data, "var", 0, 10, 42)
        results = _bootstrap_worker(args)
        assert len(results) == 10

    def test_worker_with_unknown_string_raises(self):
        """Lines 63-64: Worker raises ValueError for unknown string."""
        data = np.random.normal(100, 15, 100)
        args = (data, "unknown_stat", 0, 5, 42)
        with pytest.raises(ValueError, match="Unknown statistic string"):
            _bootstrap_worker(args)

    def test_worker_with_callable(self):
        """Lines 65-66: Worker handles callable statistic."""
        data = np.random.normal(100, 15, 100)
        args = (data, np.mean, 0, 5, 42)
        results = _bootstrap_worker(args)
        assert len(results) == 5


class TestParallelBootstrapFallback:
    """Tests for parallel bootstrap fallback to sequential (lines 234-238)."""

    def test_parallel_fallback_on_error(self):
        """Lines 234-238: Parallel failure falls back to sequential."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=False)
        data = np.random.normal(100, 15, 50)

        # Force parallel to fail by using a non-picklable lambda
        with patch.object(analyzer, "_parallel_bootstrap", side_effect=TypeError("pickle error")):
            result = analyzer.confidence_interval(data, np.mean, method="percentile", parallel=True)
        assert result.confidence_interval[0] < result.confidence_interval[1]

    def test_parallel_fallback_with_progress(self):
        """Lines 236-237: Fallback prints message when show_progress is True."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=True)
        data = np.random.normal(100, 15, 50)

        with patch.object(analyzer, "_parallel_bootstrap", side_effect=ValueError("test error")):
            with patch("builtins.print") as mock_print:
                result = analyzer.confidence_interval(
                    data, np.mean, method="percentile", parallel=True
                )
                mock_print.assert_called()


class TestSequentialBootstrapProgress:
    """Tests for _sequential_bootstrap with progress bar (line 286)."""

    def test_sequential_with_progress_disabled(self):
        """Line 286 branch: show_progress=False skips tqdm."""
        analyzer = BootstrapAnalyzer(n_bootstrap=100, seed=42, show_progress=False)
        data = np.random.normal(100, 15, 50)
        dist = analyzer._sequential_bootstrap(data, np.mean)
        assert len(dist) == 100


class TestSubmitBootstrapJobs:
    """Tests for _submit_bootstrap_jobs string mapping (lines 317-322)."""

    def test_median_statistic_mapped_to_string(self):
        """Lines 317-318: np.median maps to 'median' string."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=False)
        data = np.random.normal(100, 15, 50)
        # This exercises the full parallel path including string mapping
        result = analyzer.confidence_interval(data, np.median, method="percentile", parallel=True)
        assert result.n_bootstrap == 200

    def test_std_statistic_mapped_to_string(self):
        """Lines 319-320: np.std maps to 'std' string."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=False)
        data = np.random.normal(100, 15, 50)
        result = analyzer.confidence_interval(data, np.std, method="percentile", parallel=True)
        assert result.statistic > 0

    def test_var_statistic_mapped_to_string(self):
        """Lines 321-322: np.var maps to 'var' string."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=False)
        data = np.random.normal(100, 15, 50)
        result = analyzer.confidence_interval(data, np.var, method="percentile", parallel=True)
        assert result.statistic > 0


class TestParallelBootstrapProgressBar:
    """Tests for _parallel_bootstrap progress bar (lines 354, 361, 364)."""

    def test_parallel_with_progress(self):
        """Lines 354, 361, 364: Progress bar in parallel bootstrap."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=True)
        data = np.random.normal(100, 15, 50)
        dist = analyzer._parallel_bootstrap(data, np.mean)
        assert len(dist) == 200

    def test_parallel_without_progress(self):
        """Lines 354, 361, 364: No progress bar when show_progress=False."""
        analyzer = BootstrapAnalyzer(n_bootstrap=200, seed=42, n_workers=2, show_progress=False)
        data = np.random.normal(100, 15, 50)
        dist = analyzer._parallel_bootstrap(data, np.mean)
        assert len(dist) == 200


class TestCompareStatisticsRatio:
    """Tests for compare_statistics ratio with NaN (lines 544, 555)."""

    def test_compare_statistics_ratio_with_zero_denominator(self):
        """Line 544: Ratio comparison returns NaN when stat2==0."""
        analyzer = BootstrapAnalyzer(n_bootstrap=100, seed=42, show_progress=False)
        data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        # Use data where mean is near-zero to trigger NaN in some bootstraps
        data2 = np.array([-1.0, 1.0, -1.0, 1.0, 0.0] * 20)
        result = analyzer.compare_statistics(data1, data2, np.mean, comparison="ratio")
        # Result should handle NaN values gracefully (some bootstraps may produce NaN)
        assert result.statistic is not None

    def test_compare_statistics_with_progress(self):
        """Line 555: Progress bar in compare_statistics."""
        analyzer = BootstrapAnalyzer(n_bootstrap=100, seed=42, show_progress=True)
        data1 = np.random.normal(100, 15, 50)
        data2 = np.random.normal(90, 15, 50)
        result = analyzer.compare_statistics(data1, data2, np.mean, comparison="difference")
        assert result.metadata is not None
        assert result.metadata["comparison_type"] == "difference"

    def test_compare_statistics_invalid_comparison_raises(self):
        """Raise ValueError for invalid comparison type."""
        analyzer = BootstrapAnalyzer(n_bootstrap=100, seed=42, show_progress=False)
        with pytest.raises(ValueError, match="Comparison must be"):
            analyzer.compare_statistics(
                np.array([1, 2, 3]), np.array([4, 5, 6]), np.mean, comparison="invalid"
            )


class TestBootstrapResultSummary:
    """Tests for BootstrapResult.summary()."""

    def test_summary_with_bias_and_acceleration(self):
        """Summary includes bias and acceleration when present."""
        result = BootstrapResult(
            statistic=100.0,
            confidence_level=0.95,
            confidence_interval=(95.0, 105.0),
            bootstrap_distribution=np.random.normal(100, 5, 1000),
            method="bca",
            n_bootstrap=1000,
            bias=0.01,
            acceleration=0.005,
            converged=True,
        )
        summary = result.summary()
        assert "Bias:" in summary
        assert "Acceleration:" in summary
        assert "Yes" in summary  # Converged

    def test_summary_without_bias_and_acceleration(self):
        """Summary omits bias and acceleration when None."""
        result = BootstrapResult(
            statistic=100.0,
            confidence_level=0.95,
            confidence_interval=(95.0, 105.0),
            bootstrap_distribution=np.random.normal(100, 5, 1000),
            method="percentile",
            n_bootstrap=1000,
            converged=False,
        )
        summary = result.summary()
        assert "Bias:" not in summary
        assert "No" in summary  # Not converged

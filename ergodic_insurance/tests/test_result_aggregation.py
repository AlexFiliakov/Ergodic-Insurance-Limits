"""Comprehensive tests for result aggregation framework."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ergodic_insurance.result_aggregator import (
    AggregationConfig,
    BaseAggregator,
    HierarchicalAggregator,
    PercentileTracker,
    ResultAggregator,
    ResultExporter,
    TimeSeriesAggregator,
)
from ergodic_insurance.summary_statistics import (
    DistributionFitter,
    QuantileCalculator,
    StatisticalSummary,
    SummaryReportGenerator,
    SummaryStatistics,
    TDigest,
)


class TestAggregationConfig:
    """Test AggregationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AggregationConfig()

        assert config.percentiles == [1, 5, 10, 25, 50, 75, 90, 95, 99]
        assert config.calculate_moments is True
        assert config.calculate_distribution_fit is False
        assert config.chunk_size == 10_000
        assert config.cache_results is True
        assert config.precision == 6

    def test_custom_config(self):
        """Test custom configuration."""
        config = AggregationConfig(
            percentiles=[25, 50, 75], calculate_moments=False, chunk_size=5000, precision=3
        )

        assert config.percentiles == [25, 50, 75]
        assert config.calculate_moments is False
        assert config.chunk_size == 5000
        assert config.precision == 3


class TestResultAggregator:
    """Test ResultAggregator class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.lognormal(10, 2, 10000)

    def test_basic_aggregation(self, sample_data):
        """Test basic aggregation functionality."""
        aggregator = ResultAggregator()
        results = aggregator.aggregate(sample_data)

        # Check basic statistics
        assert "count" in results
        assert "mean" in results
        assert "std" in results
        assert "min" in results
        assert "max" in results

        assert results["count"] == len(sample_data)
        assert abs(results["mean"] - np.mean(sample_data)) < 0.01

    def test_percentile_calculation(self, sample_data):
        """Test percentile calculation."""
        config = AggregationConfig(percentiles=[10, 50, 90])
        aggregator = ResultAggregator(config)
        results = aggregator.aggregate(sample_data)

        assert "percentiles" in results
        assert "p10" in results["percentiles"]
        assert "p50" in results["percentiles"]
        assert "p90" in results["percentiles"]

        # Verify percentile values
        assert abs(results["percentiles"]["p50"] - np.median(sample_data)) < 0.01

    def test_moments_calculation(self, sample_data):
        """Test statistical moments calculation."""
        config = AggregationConfig(calculate_moments=True)
        aggregator = ResultAggregator(config)
        results = aggregator.aggregate(sample_data)

        assert "moments" in results
        assert "variance" in results["moments"]
        assert "skewness" in results["moments"]
        assert "kurtosis" in results["moments"]
        assert "coefficient_variation" in results["moments"]

    def test_distribution_fitting(self, sample_data):
        """Test distribution fitting."""
        config = AggregationConfig(calculate_distribution_fit=True)
        aggregator = ResultAggregator(config)
        results = aggregator.aggregate(sample_data)

        assert "distribution_fit" in results
        assert "normal" in results["distribution_fit"]
        assert "lognormal" in results["distribution_fit"]

        # Check fitted parameters
        assert "mu" in results["distribution_fit"]["normal"]
        assert "sigma" in results["distribution_fit"]["normal"]
        assert "ks_statistic" in results["distribution_fit"]["normal"]

    def test_custom_functions(self, sample_data):
        """Test custom aggregation functions."""

        def custom_median(data):
            return np.median(data)

        def custom_iqr(data):
            return np.percentile(data, 75) - np.percentile(data, 25)

        custom_funcs = {"median": custom_median, "iqr": custom_iqr}

        aggregator = ResultAggregator(custom_functions=custom_funcs)
        results = aggregator.aggregate(sample_data)

        assert "custom_median" in results
        assert "custom_iqr" in results
        assert abs(results["custom_median"] - np.median(sample_data)) < 0.01

    def test_precision_rounding(self):
        """Test precision rounding."""
        data = np.array([1.123456789])
        config = AggregationConfig(precision=3)
        aggregator = ResultAggregator(config)
        results = aggregator.aggregate(data)

        assert results["mean"] == 1.123


class TestTimeSeriesAggregator:
    """Test TimeSeriesAggregator class."""

    @pytest.fixture
    def time_series_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        # 12 periods, 100 simulations
        return np.random.randn(12, 100) * 10 + 100

    def test_period_statistics(self, time_series_data):
        """Test period-wise statistics."""
        aggregator = TimeSeriesAggregator()
        results = aggregator.aggregate(time_series_data)

        assert "period_mean" in results
        assert "period_std" in results
        assert "period_min" in results
        assert "period_max" in results

        assert len(results["period_mean"]) == 12
        assert abs(results["period_mean"][0] - np.mean(time_series_data[0])) < 0.01

    def test_cumulative_statistics(self, time_series_data):
        """Test cumulative statistics."""
        aggregator = TimeSeriesAggregator()
        results = aggregator.aggregate(time_series_data)

        assert "cumulative_mean" in results
        assert "cumulative_std" in results
        assert len(results["cumulative_mean"]) == 12

    def test_rolling_window(self, time_series_data):
        """Test rolling window statistics."""
        aggregator = TimeSeriesAggregator(window_size=3)
        results = aggregator.aggregate(time_series_data)

        assert "rolling_stats" in results
        assert "mean" in results["rolling_stats"]
        assert "std" in results["rolling_stats"]
        assert "volatility" in results["rolling_stats"]

    def test_growth_rates(self, time_series_data):
        """Test growth rate calculations."""
        aggregator = TimeSeriesAggregator()
        results = aggregator.aggregate(time_series_data)

        assert "growth_rate_mean" in results
        assert "growth_rate_std" in results
        assert len(results["growth_rate_mean"]) == 11  # One less than periods

    def test_autocorrelation(self, time_series_data):
        """Test autocorrelation calculation."""
        aggregator = TimeSeriesAggregator()
        results = aggregator.aggregate(time_series_data)

        assert "autocorrelation" in results
        assert "lag_1" in results["autocorrelation"]
        assert isinstance(results["autocorrelation"]["lag_1"], float)


class TestPercentileTracker:
    """Test PercentileTracker class."""

    def test_basic_tracking(self):
        """Test basic percentile tracking."""
        tracker = PercentileTracker([25, 50, 75])

        # Add data in batches
        rng = np.random.default_rng(42)
        for _ in range(10):
            data = rng.standard_normal(100)
            tracker.update(data)

        percentiles = tracker.get_percentiles()

        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles

    def test_streaming_accuracy(self):
        """Test t-digest streaming accuracy for large datasets."""
        tracker = PercentileTracker([50], max_samples=1000)

        rng = np.random.default_rng(42)
        all_data = []
        for _ in range(20):
            data = rng.standard_normal(100)
            tracker.update(data)
            all_data.append(data)

        assert tracker.total_count == 2000

        percentiles = tracker.get_percentiles()
        assert "p50" in percentiles

        # Verify accuracy against exact calculation
        exact_median = float(np.median(np.concatenate(all_data)))
        assert abs(percentiles["p50"] - exact_median) < 0.1

    def test_reset_functionality(self):
        """Test reset functionality."""
        tracker = PercentileTracker([50])
        tracker.update(np.random.randn(100))

        assert tracker.total_count == 100

        tracker.reset()

        assert tracker.total_count == 0
        assert tracker.get_percentiles() == {}

    def test_merge(self):
        """Test merging two PercentileTrackers produces accurate results."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10000)

        # Single tracker on all data
        single = PercentileTracker([1, 5, 25, 50, 75, 95, 99])
        single.update(data)

        # Two trackers on halves, then merged
        left = PercentileTracker([1, 5, 25, 50, 75, 95, 99])
        right = PercentileTracker([1, 5, 25, 50, 75, 95, 99])
        left.update(data[:5000])
        right.update(data[5000:])
        left.merge(right)

        single_p = single.get_percentiles()
        merged_p = left.get_percentiles()

        for key, single_val in single_p.items():
            # Merged result should be very close to single-pass result
            merged_val = merged_p[key]
            if abs(single_val) < 0.1:
                assert abs(single_val - merged_val) < 0.15
            else:
                assert abs(single_val - merged_val) / abs(single_val) < 0.05


class TestResultExporter:
    """Test ResultExporter class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for export testing."""
        return {
            "mean": 100.5,
            "std": 15.2,
            "percentiles": {"p25": 90.1, "p50": 100.5, "p75": 110.9},
            "metrics": {"var_95": 125.5, "tvar_95": 130.2},
        }

    def test_csv_export(self, sample_results):
        """Test CSV export functionality."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            filepath = Path(tmp.name)

        try:
            ResultExporter.to_csv(sample_results, filepath)

            # Read back and verify
            df = pd.read_csv(filepath, index_col=0)
            assert "mean" in df.index
            assert "std" in df.index
            assert "percentiles.p50" in df.index
            assert float(df.loc["mean", "value"]) == 100.5  # type: ignore[arg-type]
        finally:
            filepath.unlink()

    def test_json_export(self, sample_results):
        """Test JSON export functionality."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            filepath = Path(tmp.name)

        try:
            ResultExporter.to_json(sample_results, filepath)

            # Read back and verify
            with open(filepath, "r") as f:
                loaded = json.load(f)

            assert loaded["mean"] == 100.5
            assert loaded["percentiles"]["p50"] == 100.5
        finally:
            filepath.unlink()

    def test_hdf5_export(self, sample_results):
        """Test HDF5 export functionality."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            filepath = Path(tmp.name)

        try:
            ResultExporter.to_hdf5(sample_results, filepath)

            # Read back and verify
            with h5py.File(filepath, "r") as hf:
                assert hf.attrs["mean"] == 100.5
                assert hf.attrs["std"] == 15.2
                assert "percentiles" in hf
                assert "metrics" in hf
        finally:
            filepath.unlink()

    def test_numpy_array_export(self):
        """Test export with numpy arrays."""
        results = {"data": np.array([1, 2, 3, 4, 5]), "matrix": np.array([[1, 2], [3, 4]])}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            filepath = Path(tmp.name)

        try:
            ResultExporter.to_json(results, filepath)

            with open(filepath, "r") as f:
                loaded = json.load(f)

            assert loaded["data"] == [1, 2, 3, 4, 5]
            assert loaded["matrix"] == [[1, 2], [3, 4]]
        finally:
            filepath.unlink()


class TestHierarchicalAggregator:
    """Test HierarchicalAggregator class."""

    def test_two_level_hierarchy(self):
        """Test two-level hierarchical aggregation."""
        # Create sample hierarchical data
        data = {
            "scenario1": {"year1": np.random.randn(100), "year2": np.random.randn(100)},
            "scenario2": {"year1": np.random.randn(100), "year2": np.random.randn(100)},
        }

        aggregator = HierarchicalAggregator(["scenario", "year"])
        results = aggregator.aggregate_hierarchy(data)

        assert results["level"] == "scenario"
        assert "scenario1" in results["items"]
        assert "scenario2" in results["items"]
        assert "summary" in results

    def test_leaf_level_aggregation(self):
        """Test aggregation at leaf level."""
        data = {"item1": np.random.randn(100), "item2": np.random.randn(100)}

        aggregator = HierarchicalAggregator(["level1"])
        results = aggregator.aggregate_hierarchy(data)

        assert "items" in results
        assert "item1" in results["items"]
        assert "mean" in results["items"]["item1"]


class TestSummaryStatistics:
    """Test SummaryStatistics class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        return np.random.lognormal(3, 0.5, 1000)

    def test_basic_statistics(self, sample_data):
        """Test basic statistics calculation."""
        calculator = SummaryStatistics()
        summary = calculator.calculate_summary(sample_data)

        assert isinstance(summary, StatisticalSummary)
        assert "mean" in summary.basic_stats
        assert "median" in summary.basic_stats
        assert "std" in summary.basic_stats
        assert "skewness" in summary.basic_stats
        assert "kurtosis" in summary.basic_stats

    def test_distribution_fitting(self, sample_data):
        """Test distribution fitting in summary."""
        calculator = SummaryStatistics()
        summary = calculator.calculate_summary(sample_data)

        assert summary.distribution_params is not None
        assert "normal" in summary.distribution_params
        assert "lognormal" in summary.distribution_params
        assert "gamma" in summary.distribution_params

    def test_confidence_intervals(self, sample_data):
        """Test confidence interval calculation."""
        calculator = SummaryStatistics(
            confidence_level=0.95, bootstrap_iterations=100  # Reduced for testing speed
        )
        summary = calculator.calculate_summary(sample_data)

        assert summary.confidence_intervals is not None
        assert "mean" in summary.confidence_intervals
        assert "median" in summary.confidence_intervals
        assert "std" in summary.confidence_intervals

        # Check interval structure
        mean_ci = summary.confidence_intervals["mean"]
        assert len(mean_ci) == 2
        assert mean_ci[0] < mean_ci[1]

    def test_hypothesis_tests(self, sample_data):
        """Test hypothesis testing."""
        calculator = SummaryStatistics()
        summary = calculator.calculate_summary(sample_data)

        assert summary.hypothesis_tests is not None
        assert "normality" in summary.hypothesis_tests
        assert "t_test" in summary.hypothesis_tests

        # Check test results
        normality = summary.hypothesis_tests["normality"]
        assert "shapiro_statistic" in normality
        assert "shapiro_pvalue" in normality

    def test_extreme_values(self, sample_data):
        """Test extreme value statistics."""
        calculator = SummaryStatistics()
        summary = calculator.calculate_summary(sample_data)

        assert summary.extreme_values is not None
        assert "percentile_1" in summary.extreme_values
        assert "percentile_99" in summary.extreme_values
        assert "expected_shortfall_5%" in summary.extreme_values

    def test_weighted_statistics(self):
        """Test weighted statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])
        weights = np.array([1, 1, 2, 2, 1])

        calculator = SummaryStatistics()
        summary = calculator.calculate_summary(data, weights)

        # Weighted mean should be (1+2+6+8+5)/7 = 22/7 ≈ 3.14
        assert abs(summary.basic_stats["mean"] - 22 / 7) < 0.01

    def test_summary_to_dataframe(self, sample_data):
        """Test conversion to DataFrame."""
        calculator = SummaryStatistics()
        summary = calculator.calculate_summary(sample_data)

        df = summary.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "category" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns
        assert len(df) > 0


class TestFormatQuantileKey:
    """Test format_quantile_key helper (issue #334)."""

    def test_integer_percentiles(self):
        """Standard integer-percentile values produce 4-digit per-mille keys."""
        from ergodic_insurance.summary_statistics import format_quantile_key

        assert format_quantile_key(0.01) == "q0010"
        assert format_quantile_key(0.25) == "q0250"
        assert format_quantile_key(0.5) == "q0500"
        assert format_quantile_key(0.75) == "q0750"
        assert format_quantile_key(0.99) == "q0990"

    def test_subpercentile_no_collision(self):
        """Sub-percentile quantiles that previously collided are now distinct."""
        from ergodic_insurance.summary_statistics import format_quantile_key

        # These pairs collided under int(q*100) formatting
        assert format_quantile_key(0.001) != format_quantile_key(0.005)
        assert format_quantile_key(0.001) == "q0001"
        assert format_quantile_key(0.005) == "q0005"

        assert format_quantile_key(0.99) != format_quantile_key(0.995)
        assert format_quantile_key(0.99) == "q0990"
        assert format_quantile_key(0.995) == "q0995"

    def test_boundary_values(self):
        """Boundary quantile values format correctly."""
        from ergodic_insurance.summary_statistics import format_quantile_key

        assert format_quantile_key(0.0) == "q0000"
        assert format_quantile_key(1.0) == "q1000"


class TestQuantileCalculator:
    """Test QuantileCalculator class."""

    def test_basic_quantiles(self):
        """Test basic quantile calculation."""
        data = np.arange(100)
        calculator = QuantileCalculator([0.25, 0.5, 0.75])

        quantiles = calculator.calculate(data)

        assert "q0250" in quantiles
        assert "q0500" in quantiles
        assert "q0750" in quantiles

        assert abs(quantiles["q0500"] - 49.5) < 1

    def test_interpolation_methods(self):
        """Test different interpolation methods."""
        data = np.array([1, 2, 3, 4, 5])
        calculator = QuantileCalculator([0.5])

        linear = calculator.calculate(data, method="linear")
        nearest = calculator.calculate(data, method="nearest")
        lower = calculator.calculate(data, method="lower")
        higher = calculator.calculate(data, method="higher")

        assert linear["q0500"] == 3.0
        assert nearest["q0500"] == 3.0
        # Note: behavior varies between numpy versions
        assert lower["q0500"] in [2.0, 3.0]
        assert higher["q0500"] == 3.0

    def test_streaming_quantiles(self):
        """Test streaming quantile approximation using t-digest."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(20000)

        calculator = QuantileCalculator([0.25, 0.5, 0.75], seed=42)

        # Calculate exact quantiles
        exact = calculator.calculate(data)

        # Calculate streaming approximation (t-digest)
        approx = calculator.streaming_quantiles(data, compression=200)

        # t-digest should achieve much tighter accuracy than reservoir sampling
        for key in exact:  # pylint: disable=consider-using-dict-items
            # Use absolute tolerance for values close to zero
            if abs(exact[key]) < 0.1:
                assert abs(exact[key] - approx[key]) < 0.1
            else:
                # t-digest achieves <5% relative error easily
                assert abs(exact[key] - approx[key]) / abs(exact[key]) < 0.05


class TestTDigest:
    """Test TDigest streaming quantile estimation."""

    def test_basic_quantiles_uniform(self):
        """Test quantile accuracy on uniform distribution."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 50000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            exact = float(np.percentile(data, q * 100))
            approx = digest.quantile(q)
            if abs(exact) > 0.01:
                assert (
                    abs(exact - approx) / abs(exact) < 0.02
                ), f"q={q}: exact={exact:.4f}, approx={approx:.4f}"
            else:
                assert abs(exact - approx) < 0.02

    def test_basic_quantiles_normal(self):
        """Test quantile accuracy on normal distribution."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(50000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            exact = float(np.percentile(data, q * 100))
            approx = digest.quantile(q)
            if abs(exact) > 0.1:
                assert (
                    abs(exact - approx) / abs(exact) < 0.02
                ), f"q={q}: exact={exact:.4f}, approx={approx:.4f}"
            else:
                assert abs(exact - approx) < 0.05

    def test_basic_quantiles_exponential(self):
        """Test quantile accuracy on exponential distribution."""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=5.0, size=50000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            exact = float(np.percentile(data, q * 100))
            approx = digest.quantile(q)
            assert (
                abs(exact - approx) / abs(exact) < 0.02
            ), f"q={q}: exact={exact:.4f}, approx={approx:.4f}"

    def test_tail_quantile_accuracy(self):
        """Test that tail quantiles (q01, q05, q95, q99) are within 1% error."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        for q in [0.01, 0.05, 0.95, 0.99]:
            exact = float(np.percentile(data, q * 100))
            approx = digest.quantile(q)
            rel_error = abs(exact - approx) / abs(exact)
            assert (
                rel_error < 0.01
            ), f"Tail q={q}: exact={exact:.4f}, approx={approx:.4f}, error={rel_error:.4f}"

    def test_merge_correctness(self):
        """Test that merging two digests equals one digest on all data."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(20000)

        # Single digest on all data
        single = TDigest(compression=200)
        single.update_batch(data)

        # Two digests on halves, merged
        left = TDigest(compression=200)
        right = TDigest(compression=200)
        left.update_batch(data[:10000])
        right.update_batch(data[10000:])
        left.merge(right)

        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            single_val = single.quantile(q)
            merged_val = left.quantile(q)
            exact = float(np.percentile(data, q * 100))
            # Both should be close to exact
            if abs(exact) > 0.1:
                assert abs(single_val - exact) / abs(exact) < 0.02
                assert abs(merged_val - exact) / abs(exact) < 0.02
            else:
                assert abs(single_val - exact) < 0.05
                assert abs(merged_val - exact) < 0.05

    def test_empty_digest(self):
        """Test behavior of empty digest."""
        digest = TDigest()

        with pytest.raises(ValueError, match="empty"):
            digest.quantile(0.5)

        with pytest.raises(ValueError, match="empty"):
            digest.cdf(0.0)

        assert len(digest) == 0
        assert digest.centroid_count == 0

    def test_single_value(self):
        """Test digest with a single value."""
        digest = TDigest()
        digest.update(42.0)

        assert digest.quantile(0.0) == 42.0
        assert digest.quantile(0.5) == 42.0
        assert digest.quantile(1.0) == 42.0
        assert len(digest) == 1

    def test_duplicate_values(self):
        """Test digest with many duplicate values."""
        digest = TDigest()
        digest.update_batch(np.full(10000, 7.0))

        assert digest.quantile(0.5) == 7.0
        assert digest.quantile(0.01) == 7.0
        assert digest.quantile(0.99) == 7.0

    def test_memory_efficiency(self):
        """Test that centroid count stays within bounds."""
        digest = TDigest(compression=100)
        rng = np.random.default_rng(42)

        # Add a large amount of data
        for _ in range(100):
            digest.update_batch(rng.standard_normal(10000))

        assert len(digest) == 1_000_000
        # Centroid count should be O(compression), much less than data size
        assert digest.centroid_count < 1000

    def test_cdf(self):
        """Test CDF estimation."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(50000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        # CDF at median should be close to 0.5
        median = float(np.median(data))
        assert abs(digest.cdf(median) - 0.5) < 0.02

        # CDF at min should be close to 0
        assert digest.cdf(float(np.min(data)) - 1) == 0.0

        # CDF at max should be close to 1
        assert digest.cdf(float(np.max(data)) + 1) == 1.0

    def test_quantiles_dict(self):
        """Test quantiles() method returns properly formatted dict."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        result = digest.quantiles([0.25, 0.5, 0.75])
        assert "q0250" in result
        assert "q0500" in result
        assert "q0750" in result
        assert result["q0250"] < result["q0500"] < result["q0750"]

    def test_quantiles_no_subpercentile_collision(self):
        """Test that sub-percentile quantiles produce distinct keys (issue #334)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10000)

        digest = TDigest(compression=200)
        digest.update_batch(data)

        # These pairs previously collided under int(q*100) formatting
        result = digest.quantiles([0.001, 0.005, 0.99, 0.995])
        assert (
            len(result) == 4
        ), f"Expected 4 distinct keys, got {len(result)}: {list(result.keys())}"
        assert "q0001" in result  # 0.1th percentile
        assert "q0005" in result  # 0.5th percentile
        assert "q0990" in result  # 99th percentile
        assert "q0995" in result  # 99.5th percentile

    def test_large_dataset_streaming(self):
        """Integration test: 1M points streaming matches np.percentile within 1%."""
        rng = np.random.default_rng(42)

        digest = TDigest(compression=200)
        all_data = []
        for _ in range(100):
            chunk = rng.standard_normal(10000)
            digest.update_batch(chunk)
            all_data.append(chunk)

        full_data = np.concatenate(all_data)

        for q in [0.01, 0.05, 0.5, 0.95, 0.99]:
            exact = float(np.percentile(full_data, q * 100))
            approx = digest.quantile(q)
            if abs(exact) > 0.1:
                assert (
                    abs(exact - approx) / abs(exact) < 0.01
                ), f"q={q}: exact={exact:.4f}, approx={approx:.4f}"
            else:
                assert abs(exact - approx) < 0.02


class TestDistributionFitter:
    """Test DistributionFitter class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data from known distribution."""
        np.random.seed(42)
        return stats.gamma.rvs(2, scale=2, size=1000)

    def test_fit_all_distributions(self, sample_data):
        """Test fitting multiple distributions."""
        fitter = DistributionFitter()
        results = fitter.fit_all(sample_data, distributions=["normal", "gamma", "exponential"])

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "distribution" in results.columns
        assert "aic" in results.columns
        assert "bic" in results.columns
        assert "ks_statistic" in results.columns

    def test_best_distribution_selection(self, sample_data):
        """Test selection of best-fitting distribution."""
        fitter = DistributionFitter()
        fitter.fit_all(sample_data, distributions=["normal", "gamma", "exponential"])

        # Gamma should be the best fit
        best_dist, params = fitter.get_best_distribution(criterion="aic")
        assert best_dist == "gamma"
        assert params is not None

    def test_qq_plot_data(self, sample_data):
        """Test Q-Q plot data generation."""
        fitter = DistributionFitter()
        fitter.fit_all(sample_data, distributions=["normal"])

        theoretical, sample = fitter.generate_qq_plot_data(sample_data, "normal")

        assert len(theoretical) == len(sample_data)
        assert len(sample) == len(sample_data)
        assert np.allclose(sample, np.sort(sample_data))


class TestSummaryReportGenerator:
    """Test SummaryReportGenerator class."""

    @pytest.fixture
    def sample_summary(self):
        """Create sample summary for report generation."""
        return StatisticalSummary(
            basic_stats={"mean": 100.0, "std": 10.0, "median": 99.5},
            distribution_params={"normal": {"mu": 100.0, "sigma": 10.0}},
            confidence_intervals={"mean": (98.0, 102.0)},
            hypothesis_tests={"normality": {"shapiro_pvalue": 0.05}},
            extreme_values={"percentile_1": 75.0, "percentile_99": 125.0},
        )

    def test_markdown_report(self, sample_summary):
        """Test Markdown report generation."""
        generator = SummaryReportGenerator(style="markdown")
        report = generator.generate_report(
            sample_summary, title="Test Report", metadata={"simulations": 1000}
        )

        assert isinstance(report, str)
        assert "# Test Report" in report
        assert "## Basic Statistics" in report
        assert "mean" in report
        assert "100.0" in report

    def test_html_report(self, sample_summary):
        """Test HTML report generation."""
        generator = SummaryReportGenerator(style="html")
        report = generator.generate_report(sample_summary, title="Test Report")

        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "<h1>Test Report</h1>" in report
        assert "<table" in report

    def test_latex_report(self, sample_summary):
        """Test LaTeX report generation."""
        generator = SummaryReportGenerator(style="latex")
        report = generator.generate_report(sample_summary, title="Test Report")

        assert isinstance(report, str)
        assert "\\documentclass{article}" in report
        assert "\\title{Test Report}" in report
        assert "\\begin{document}" in report


class TestIntegration:
    """Integration tests for the aggregation framework."""

    def test_end_to_end_aggregation(self):
        """Test complete aggregation workflow."""
        # Generate synthetic simulation results
        np.random.seed(42)
        n_simulations = 1000
        n_years = 10

        final_assets = np.random.lognormal(15, 1, n_simulations)
        annual_losses = np.random.lognormal(10, 2, (n_simulations, n_years))

        # Perform aggregation
        config = AggregationConfig(
            percentiles=[5, 25, 50, 75, 95], calculate_moments=True, calculate_distribution_fit=True
        )

        aggregator = ResultAggregator(config)
        results = aggregator.aggregate(final_assets)

        # Verify results structure
        assert "mean" in results
        assert "percentiles" in results
        assert "moments" in results
        assert "distribution_fit" in results

        # Time series aggregation
        ts_aggregator = TimeSeriesAggregator(config)
        ts_results = ts_aggregator.aggregate(annual_losses.T)

        assert "period_mean" in ts_results
        assert "cumulative_mean" in ts_results
        assert "growth_rate_mean" in ts_results

        # Statistical summary
        stats_calc = SummaryStatistics()
        summary = stats_calc.calculate_summary(final_assets)

        assert summary.basic_stats is not None
        assert summary.distribution_params is not None
        assert summary.confidence_intervals is not None

        # Report generation
        report_gen = SummaryReportGenerator()
        report = report_gen.generate_report(summary)

        assert isinstance(report, str)
        assert len(report) > 0

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Generate large dataset
        np.random.seed(42)
        n_simulations = 1_000_000
        data = np.random.randn(n_simulations)

        # Test percentile tracking with streaming (t-digest)
        tracker = PercentileTracker([25, 50, 75], max_samples=10_000)

        # Process in chunks
        chunk_size = 10_000
        for i in range(0, n_simulations, chunk_size):
            chunk = data[i : i + chunk_size]
            tracker.update(chunk)

        percentiles = tracker.get_percentiles()

        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles

        # t-digest should produce very accurate approximations
        exact_median = np.median(data)
        approx_median = percentiles["p50"]
        assert abs(exact_median - approx_median) < 0.05

    def test_memory_efficiency(self):
        """Test memory-efficient aggregation with t-digest."""

        # Create generator for large dataset
        def data_generator(n_chunks=100, chunk_size=10_000):
            np.random.seed(42)
            for _ in range(n_chunks):
                yield np.random.randn(chunk_size)

        # Incremental aggregation
        tracker = PercentileTracker([50])

        for chunk in data_generator():
            tracker.update(chunk)

        assert tracker.total_count == 1_000_000
        # t-digest uses bounded memory (centroid count << data size)
        assert tracker._digest.centroid_count < 2000

        percentiles = tracker.get_percentiles()
        assert "p50" in percentiles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

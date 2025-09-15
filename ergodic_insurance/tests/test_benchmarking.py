"""Comprehensive tests for the benchmarking module."""

from dataclasses import dataclass
from datetime import datetime
import gc
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import psutil
import pytest

from ergodic_insurance.benchmarking import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    ComprehensiveBenchmarkResult,
    ConfigurationComparison,
    SystemProfiler,
    run_quick_benchmark,
)


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics dataclass."""

    def test_initialization_minimal(self):
        """Test minimal BenchmarkMetrics initialization."""
        metrics = BenchmarkMetrics(
            execution_time=10.5,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        assert metrics.execution_time == 10.5
        assert metrics.simulations_per_second == 1000.0
        assert metrics.memory_peak_mb == 256.0
        assert metrics.memory_average_mb == 200.0
        assert metrics.cpu_utilization == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.accuracy_score == 1.0
        assert metrics.convergence_iterations == 0

    def test_initialization_full(self):
        """Test full BenchmarkMetrics initialization."""
        metrics = BenchmarkMetrics(
            execution_time=10.5,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            cpu_utilization=75.5,
            cache_hit_rate=85.0,
            accuracy_score=0.99,
            convergence_iterations=50000,
        )
        assert metrics.cpu_utilization == 75.5
        assert metrics.cache_hit_rate == 85.0
        assert metrics.accuracy_score == 0.99
        assert metrics.convergence_iterations == 50000

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = BenchmarkMetrics(
            execution_time=10.5,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            cpu_utilization=75.5,
            cache_hit_rate=85.0,
            accuracy_score=0.99,
            convergence_iterations=50000,
        )
        result = metrics.to_dict()
        assert result["execution_time"] == 10.5
        assert result["simulations_per_second"] == 1000.0
        assert result["memory_peak_mb"] == 256.0
        assert result["memory_average_mb"] == 200.0
        assert result["cpu_utilization"] == 75.5
        assert result["cache_hit_rate"] == 85.0
        assert result["accuracy_score"] == 0.99
        assert result["convergence_iterations"] == 50000


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_initialization_minimal(self):
        """Test minimal BenchmarkResult initialization."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={"n_workers": 4}, timestamp=datetime.now()
        )
        assert result.scale == 10000
        assert result.metrics == metrics
        assert result.configuration["n_workers"] == 4
        assert result.system_info == {}
        assert result.optimizations == []

    def test_initialization_full(self):
        """Test full BenchmarkResult initialization."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        system_info = {"cpu_cores": 4, "ram_gb": 8}
        optimizations = ["vectorization", "caching"]
        result = BenchmarkResult(
            scale=10000,
            metrics=metrics,
            configuration={"n_workers": 4},
            timestamp=datetime.now(),
            system_info=system_info,
            optimizations=optimizations,
        )
        assert result.system_info == system_info
        assert result.optimizations == optimizations

    def test_meets_target_success(self):
        """Test meets_target when targets are met."""
        metrics = BenchmarkMetrics(
            execution_time=5.0,
            simulations_per_second=2000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        assert result.meets_target(target_time=10.0, target_memory=512.0) is True

    def test_meets_target_failure_time(self):
        """Test meets_target when time target is not met."""
        metrics = BenchmarkMetrics(
            execution_time=15.0,
            simulations_per_second=667.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        assert result.meets_target(target_time=10.0, target_memory=512.0) is False

    def test_meets_target_failure_memory(self):
        """Test meets_target when memory target is not met."""
        metrics = BenchmarkMetrics(
            execution_time=5.0,
            simulations_per_second=2000.0,
            memory_peak_mb=600.0,
            memory_average_mb=500.0,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        assert result.meets_target(target_time=10.0, target_memory=512.0) is False

    def test_summary_basic(self):
        """Test summary generation with basic metrics."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        summary = result.summary()
        assert "10,000 simulations" in summary
        assert "Time: 10.00s" in summary
        assert "(1000 sims/s)" in summary
        assert "Memory: 256.0 MB peak" in summary
        assert "200.0 MB avg" in summary
        assert "CPU: 0.0%" in summary

    def test_summary_with_cache(self):
        """Test summary generation with cache metrics."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            cache_hit_rate=85.5,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        summary = result.summary()
        assert "Cache: 85.5% hit rate" in summary

    def test_summary_with_accuracy(self):
        """Test summary generation with accuracy metrics."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            accuracy_score=0.95,
        )
        result = BenchmarkResult(
            scale=10000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        summary = result.summary()
        assert "Accuracy: 0.9500" in summary


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_default_initialization(self):
        """Test default BenchmarkConfig initialization."""
        config = BenchmarkConfig()
        assert config.scales == [1000, 10000, 100000]
        assert config.n_years == 10
        assert config.n_workers == 4
        assert config.memory_limit_mb == 4000.0
        assert config.target_times[1000] == 1.0
        assert config.target_times[10000] == 10.0
        assert config.target_times[100000] == 60.0
        assert config.repetitions == 3
        assert config.warmup_runs == 2
        assert config.enable_profiling is True

    def test_custom_initialization(self):
        """Test custom BenchmarkConfig initialization."""
        config = BenchmarkConfig(
            scales=[500, 5000],
            n_years=20,
            n_workers=8,
            memory_limit_mb=8000.0,
            target_times={500: 0.5, 5000: 5.0},
            repetitions=5,
            warmup_runs=3,
            enable_profiling=False,
        )
        assert config.scales == [500, 5000]
        assert config.n_years == 20
        assert config.n_workers == 8
        assert config.memory_limit_mb == 8000.0
        assert config.target_times[500] == 0.5
        assert config.target_times[5000] == 5.0
        assert config.repetitions == 5
        assert config.warmup_runs == 3
        assert config.enable_profiling is False


class TestSystemProfiler:
    """Test SystemProfiler class."""

    def test_initialization(self):
        """Test SystemProfiler initialization."""
        with patch("ergodic_insurance.benchmarking.psutil.Process") as MockProcess:
            mock_process = MockProcess.return_value
            profiler = SystemProfiler()
            assert profiler.process == mock_process
            assert profiler.cpu_samples == []
            assert profiler.memory_samples == []
            assert profiler.initial_memory == 0.0

    def test_start(self):
        """Test profiling start."""
        with patch("ergodic_insurance.benchmarking.psutil.Process") as MockProcess:
            mock_process = MockProcess.return_value
            mock_process.memory_info.return_value.rss = 256 * 1024 * 1024  # 256 MB

            profiler = SystemProfiler()
            profiler.cpu_samples = [1, 2, 3]  # Pre-existing samples
            profiler.memory_samples = [100, 200]

            profiler.start()

            assert profiler.cpu_samples == []
            assert profiler.memory_samples == []
            assert profiler.initial_memory == 256.0

    def test_sample(self):
        """Test taking resource samples."""
        with patch("ergodic_insurance.benchmarking.psutil.Process") as MockProcess:
            mock_process = MockProcess.return_value
            mock_process.cpu_percent.return_value = 75.5
            mock_process.memory_info.return_value.rss = 512 * 1024 * 1024  # 512 MB

            profiler = SystemProfiler()
            profiler.sample()

            assert profiler.cpu_samples == [75.5]
            assert profiler.memory_samples == [512.0]

            # Take another sample
            mock_process.cpu_percent.return_value = 80.0
            mock_process.memory_info.return_value.rss = 520 * 1024 * 1024
            profiler.sample()

            assert profiler.cpu_samples == [75.5, 80.0]
            assert profiler.memory_samples == [512.0, 520.0]

    def test_get_metrics_empty(self):
        """Test getting metrics with no samples."""
        with patch("ergodic_insurance.benchmarking.psutil.Process"):
            profiler = SystemProfiler()
            profiler.initial_memory = 100.0
            avg_cpu, peak_memory, avg_memory = profiler.get_metrics()
            assert avg_cpu == 0.0
            assert peak_memory == 100.0
            assert avg_memory == 100.0

    def test_get_metrics_with_samples(self):
        """Test getting metrics with samples."""
        with patch("ergodic_insurance.benchmarking.psutil.Process"):
            profiler = SystemProfiler()
            profiler.initial_memory = 100.0
            profiler.cpu_samples = [50.0, 60.0, 70.0, 80.0]
            profiler.memory_samples = [200.0, 250.0, 300.0, 280.0]

            avg_cpu, peak_memory, avg_memory = profiler.get_metrics()
            assert avg_cpu == 70.0  # Average of [60, 70, 80] (first sample excluded)
            assert peak_memory == 300.0  # Max of memory samples
            assert avg_memory == 257.5  # Average of [200, 250, 300, 280]


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""

    @pytest.fixture
    def temp_report_dir(self):
        """Create a temporary directory for reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization_default(self):
        """Test default BenchmarkSuite initialization."""
        suite = BenchmarkSuite()
        assert suite.runner is not None
        assert suite.results == []
        assert suite.system_info is not None

    def test_initialization_custom(self):
        """Test custom BenchmarkSuite initialization."""
        suite = BenchmarkSuite()
        assert hasattr(suite, "runner")
        assert hasattr(suite, "results")
        assert hasattr(suite, "system_info")

    def test_get_system_info(self):
        """Test system information collection."""
        with patch("ergodic_insurance.benchmarking.psutil") as mock_psutil:
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.virtual_memory.return_value.total = 16 * 1024**3
            mock_psutil.cpu_freq.return_value.current = 3500

            info = SystemProfiler.get_system_info()

            assert "cpu" in info
            assert "memory" in info
            assert "platform" in info

    def test_benchmark_scale_basic(self):
        """Test basic benchmark scale method."""
        suite = BenchmarkSuite()
        mock_engine = MagicMock()
        mock_engine.config = MagicMock()
        mock_engine.config.n_simulations = 1000
        mock_engine.config.n_years = 10
        mock_engine.config.n_workers = 4
        mock_engine.config.progress_bar = False
        mock_engine.run.return_value = MagicMock()

        config = BenchmarkConfig(scales=[1000], repetitions=1, warmup_runs=0)

        with patch.object(suite.runner, "run_with_warmup") as mock_warmup:
            metrics = BenchmarkMetrics(
                execution_time=1.0,
                simulations_per_second=1000.0,
                memory_peak_mb=256.0,
                memory_average_mb=200.0,
            )
            mock_warmup.return_value = [metrics]

            with patch("builtins.print"):  # Suppress print statements
                result = suite.benchmark_scale(mock_engine, 1000, config)
            assert result.scale == 1000
            assert result.metrics.execution_time == 1.0

    def test_benchmark_scale_with_warmup(self):
        """Test benchmark with warmup runs."""
        suite = BenchmarkSuite()
        mock_engine = MagicMock()
        mock_engine.config = MagicMock()
        mock_engine.run.return_value = MagicMock()
        config = BenchmarkConfig(scales=[1000], repetitions=3, warmup_runs=2)

        with patch.object(suite.runner, "run_with_warmup") as mock_warmup:
            metrics = BenchmarkMetrics(
                execution_time=1.0,
                simulations_per_second=1000.0,
                memory_peak_mb=256.0,
                memory_average_mb=200.0,
            )
            mock_warmup.return_value = [metrics, metrics, metrics]

            with patch("builtins.print"):  # Suppress print statements
                result = suite.benchmark_scale(mock_engine, 1000, config)

            # Check that run_with_warmup was called with correct params
            mock_warmup.assert_called_once()
            assert result.scale == 1000

    def test_runner_single_benchmark(self):
        """Test single benchmark execution via runner."""
        runner = BenchmarkRunner()
        mock_func = MagicMock(return_value=MagicMock(n_simulations=1000))

        with patch.object(runner.profiler, "start"):
            with patch.object(runner.profiler, "sample"):
                with patch.object(
                    runner.profiler, "get_metrics", return_value=(75.0, 300.0, 250.0)
                ):
                    metrics = runner.run_single_benchmark(mock_func, kwargs={"n_simulations": 1000})

                    assert metrics.simulations_per_second > 0
                    assert metrics.cpu_utilization == 75.0
                    assert metrics.memory_peak_mb == 300.0
                    assert metrics.memory_average_mb == 250.0

    def test_runner_with_warmup(self):
        """Test runner with warmup runs."""
        runner = BenchmarkRunner()
        mock_func = MagicMock(return_value=MagicMock(n_simulations=1000))

        with patch.object(runner, "run_single_benchmark") as mock_single:
            mock_single.return_value = BenchmarkMetrics(
                execution_time=1.0,
                simulations_per_second=1000.0,
                memory_peak_mb=256.0,
                memory_average_mb=200.0,
            )

            with patch("builtins.print"):  # Suppress print statements
                metrics_list = runner.run_with_warmup(mock_func, warmup_runs=2, benchmark_runs=3)

            assert len(metrics_list) == 3
            assert all(m.execution_time == 1.0 for m in metrics_list)

    def test_compare_configurations(self):
        """Test comparing multiple configurations."""
        suite = BenchmarkSuite()
        mock_factory = MagicMock()

        configs = [{"name": "baseline", "n_workers": 1}, {"name": "parallel", "n_workers": 4}]

        with patch.object(suite, "benchmark_scale") as mock_benchmark:
            # Mock different results for different configs
            metrics1 = BenchmarkMetrics(
                execution_time=10.0,
                simulations_per_second=1000.0,
                memory_peak_mb=256.0,
                memory_average_mb=200.0,
            )
            metrics2 = BenchmarkMetrics(
                execution_time=5.0,
                simulations_per_second=2000.0,
                memory_peak_mb=512.0,
                memory_average_mb=400.0,
            )
            result1 = BenchmarkResult(
                scale=10000, metrics=metrics1, configuration=configs[0], timestamp=datetime.now()
            )
            result2 = BenchmarkResult(
                scale=10000, metrics=metrics2, configuration=configs[1], timestamp=datetime.now()
            )
            mock_benchmark.side_effect = [result1, result2]

            with patch("builtins.print"):  # Suppress print statements
                comparison = suite.compare_configurations(mock_factory, configs)

            assert len(comparison.results) == 2
            assert comparison.results[0]["configuration"]["name"] == "baseline"
            assert comparison.results[1]["configuration"]["name"] == "parallel"

    def test_run_comprehensive_benchmark(self):
        """Test comprehensive benchmark suite."""
        suite = BenchmarkSuite()
        mock_engine = MagicMock()
        mock_engine.config = MagicMock()
        config = BenchmarkConfig(scales=[1000])

        with patch.object(suite, "benchmark_scale") as mock_benchmark:
            metrics = BenchmarkMetrics(
                execution_time=1.0,
                simulations_per_second=1000.0,
                memory_peak_mb=256.0,
                memory_average_mb=200.0,
            )
            result = BenchmarkResult(
                scale=1000, metrics=metrics, configuration={}, timestamp=datetime.now()
            )
            mock_benchmark.return_value = result

            with patch("builtins.print"):  # Suppress print statements
                # Mock that engine doesn't have enable_optimizations
                if hasattr(mock_engine, "enable_optimizations"):
                    delattr(mock_engine, "enable_optimizations")
                results = suite.run_comprehensive_benchmark(mock_engine, config)

            # Should have 1 result if no optimizations, 2 if has optimizations
            assert len(results.results) >= 1
            assert results.results[0].scale == 1000
            assert isinstance(results, ComprehensiveBenchmarkResult)

    def test_comprehensive_result_meets_requirements(self):
        """Test comprehensive result requirements checking."""
        metrics = BenchmarkMetrics(
            execution_time=50.0,  # Less than 60s
            simulations_per_second=2000.0,
            memory_peak_mb=3500.0,  # Less than 4000MB
            memory_average_mb=3000.0,
            accuracy_score=0.9999,  # Greater than 0.9999
        )
        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        config = BenchmarkConfig()
        comp_result = ComprehensiveBenchmarkResult([result], config, {})
        assert comp_result.meets_requirements() is True

    def test_comprehensive_result_meets_requirements_failure(self):
        """Test comprehensive result requirements failure."""
        metrics = BenchmarkMetrics(
            execution_time=65.0,  # More than 60s - FAIL
            simulations_per_second=1500.0,
            memory_peak_mb=3500.0,
            memory_average_mb=3000.0,
            accuracy_score=0.9999,
        )
        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        config = BenchmarkConfig()
        comp_result = ComprehensiveBenchmarkResult([result], config, {})
        assert comp_result.meets_requirements() is False

    def test_configuration_comparison_best(self):
        """Test finding best configuration."""
        metrics1 = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        metrics2 = BenchmarkMetrics(
            execution_time=5.0,  # Best time
            simulations_per_second=2000.0,
            memory_peak_mb=512.0,
            memory_average_mb=400.0,
        )
        result1 = BenchmarkResult(
            scale=10000, metrics=metrics1, configuration={}, timestamp=datetime.now()
        )
        result2 = BenchmarkResult(
            scale=10000, metrics=metrics2, configuration={}, timestamp=datetime.now()
        )

        comparison = ConfigurationComparison(
            [
                {"configuration": {"name": "baseline"}, "result": result1},
                {"configuration": {"name": "parallel"}, "result": result2},
            ]
        )

        best = comparison.best_configuration()
        assert best["name"] == "parallel"

    def test_comprehensive_result_summary(self):
        """Test comprehensive result summary generation."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=10000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            cpu_utilization=85.0,
            cache_hit_rate=90.0,
            accuracy_score=0.9999,
        )
        result = BenchmarkResult(
            scale=100000,
            metrics=metrics,
            configuration={},
            timestamp=datetime.now(),
            optimizations=["vectorization", "caching"],
        )
        config = BenchmarkConfig()
        comp_result = ComprehensiveBenchmarkResult([result], config, {})

        summary = comp_result.summary()
        assert "100,000" in summary
        assert "10.00s" in summary
        assert "vectorization, caching" in summary

    def test_save_report(self, temp_report_dir):
        """Test report saving."""
        metrics = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            cpu_utilization=75.0,
        )
        result = BenchmarkResult(
            scale=10000,
            metrics=metrics,
            configuration={"n_workers": 4},
            timestamp=datetime.now(),
            system_info={"cpu_count": 4},
        )
        config = BenchmarkConfig()
        comp_result = ComprehensiveBenchmarkResult([result], config, {"test": "info"})

        report_path = temp_report_dir / "test_report.json"

        with patch("builtins.print"):  # Suppress print statement
            comp_result.save_report(str(report_path))

        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Verify report content
        with open(report_path, "r") as f:
            data = json.load(f)
        assert "system_info" in data
        assert "results" in data
        assert len(data["results"]) == 1

    def test_configuration_comparison_summary(self):
        """Test configuration comparison summary."""
        metrics1 = BenchmarkMetrics(
            execution_time=10.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
            cpu_utilization=50.0,
        )
        metrics2 = BenchmarkMetrics(
            execution_time=5.0,
            simulations_per_second=2000.0,
            memory_peak_mb=512.0,
            memory_average_mb=400.0,
            cpu_utilization=80.0,
        )
        result1 = BenchmarkResult(
            scale=10000, metrics=metrics1, configuration={}, timestamp=datetime.now()
        )
        result2 = BenchmarkResult(
            scale=10000, metrics=metrics2, configuration={}, timestamp=datetime.now()
        )

        comparison = ConfigurationComparison(
            [
                {"configuration": {"name": "baseline"}, "result": result1},
                {"configuration": {"name": "parallel"}, "result": result2},
            ]
        )

        summary = comparison.summary()
        assert "baseline" in summary
        assert "parallel" in summary
        assert "10.00s" in summary
        assert "5.00s" in summary

    def test_quick_benchmark(self):
        """Test quick benchmark function."""
        mock_engine = MagicMock()
        mock_engine.config = MagicMock()
        mock_engine.run.return_value = MagicMock(n_simulations=10000)

        with patch("ergodic_insurance.benchmarking.BenchmarkRunner") as MockRunner:
            mock_runner = MockRunner.return_value
            mock_runner.run_single_benchmark.return_value = BenchmarkMetrics(
                execution_time=5.0,
                simulations_per_second=2000.0,
                memory_peak_mb=256.0,
                memory_average_mb=200.0,
            )

            metrics = run_quick_benchmark(mock_engine, n_simulations=10000)

            assert metrics.execution_time == 5.0
            assert metrics.simulations_per_second == 2000.0

    def test_system_profiler_get_system_info(self):
        """Test system profiler get_system_info."""
        # Just test that it returns the expected structure
        info = SystemProfiler.get_system_info()

        assert "cpu" in info
        assert "memory" in info
        assert "platform" in info
        assert "timestamp" in info
        assert "python_version" in info
        assert "processor" in info

        # Check cpu structure
        assert "cores_physical" in info["cpu"]
        assert "cores_logical" in info["cpu"]
        assert "cpu_freq_mhz" in info["cpu"]

        # Check memory structure
        assert "total_gb" in info["memory"]
        assert "available_gb" in info["memory"]

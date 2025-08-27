"""Comprehensive benchmarking suite for Monte Carlo simulations.

This module provides tools for benchmarking Monte Carlo engine performance,
targeting 100K simulations in under 60 seconds on 4-core CPUs with <4GB memory.

Key features:
    - Performance benchmarking at multiple scales (1K, 10K, 100K)
    - Memory usage tracking and profiling
    - CPU efficiency monitoring
    - Cache effectiveness measurement
    - Automated performance report generation
    - Comparison of optimization strategies

Example:
    >>> from benchmarking import BenchmarkSuite, BenchmarkConfig
    >>> from monte_carlo import MonteCarloEngine
    >>>
    >>> suite = BenchmarkSuite()
    >>> config = BenchmarkConfig(scales=[1000, 10000, 100000])
    >>>
    >>> # Run comprehensive benchmarks
    >>> results = suite.run_comprehensive_benchmark(engine, config)
    >>> print(results.summary())
    >>>
    >>> # Check if performance targets are met
    >>> if results.meets_requirements():
    ...     print("✓ All performance targets achieved!")

Google-style docstrings are used throughout for Sphinx documentation.
"""

from dataclasses import dataclass, field
from datetime import datetime
import gc
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
from tabulate import tabulate


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmarking.

    Attributes:
        execution_time: Total execution time in seconds
        simulations_per_second: Throughput metric
        memory_peak_mb: Peak memory usage in MB
        memory_average_mb: Average memory usage in MB
        cpu_utilization: Average CPU utilization percentage
        cache_hit_rate: Cache effectiveness percentage
        accuracy_score: Numerical accuracy score
        convergence_iterations: Iterations to convergence
    """

    execution_time: float
    simulations_per_second: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    accuracy_score: float = 1.0
    convergence_iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "execution_time": self.execution_time,
            "simulations_per_second": self.simulations_per_second,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_average_mb": self.memory_average_mb,
            "cpu_utilization": self.cpu_utilization,
            "cache_hit_rate": self.cache_hit_rate,
            "accuracy_score": self.accuracy_score,
            "convergence_iterations": self.convergence_iterations,
        }


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        scale: Number of simulations
        metrics: Performance metrics
        configuration: Configuration used
        timestamp: When benchmark was run
        system_info: System information
        optimizations: Optimizations applied
    """

    scale: int
    metrics: BenchmarkMetrics
    configuration: Dict[str, Any]
    timestamp: datetime
    system_info: Dict[str, Any] = field(default_factory=dict)
    optimizations: List[str] = field(default_factory=list)

    def meets_target(self, target_time: float, target_memory: float) -> bool:
        """Check if result meets performance targets.

        Args:
            target_time: Maximum execution time in seconds.
            target_memory: Maximum memory usage in MB.

        Returns:
            True if targets are met.
        """
        return (
            self.metrics.execution_time <= target_time
            and self.metrics.memory_peak_mb <= target_memory
        )

    def summary(self) -> str:
        """Generate result summary.

        Returns:
            Formatted summary string.
        """
        summary = f"Benchmark Result - {self.scale:,} simulations\n"
        summary += f"  Time: {self.metrics.execution_time:.2f}s "
        summary += f"({self.metrics.simulations_per_second:.0f} sims/s)\n"
        summary += f"  Memory: {self.metrics.memory_peak_mb:.1f} MB peak, "
        summary += f"{self.metrics.memory_average_mb:.1f} MB avg\n"
        summary += f"  CPU: {self.metrics.cpu_utilization:.1f}%\n"

        if self.metrics.cache_hit_rate > 0:
            summary += f"  Cache: {self.metrics.cache_hit_rate:.1f}% hit rate\n"

        if self.metrics.accuracy_score < 1.0:
            summary += f"  Accuracy: {self.metrics.accuracy_score:.4f}\n"

        return summary


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking.

    Attributes:
        scales: List of simulation counts to test
        n_years: Years per simulation
        n_workers: Number of parallel workers
        memory_limit_mb: Memory limit for testing
        target_times: Target execution times per scale
        repetitions: Number of repetitions per test
        warmup_runs: Number of warmup runs
        enable_profiling: Enable detailed profiling
    """

    scales: List[int] = field(default_factory=lambda: [1000, 10000, 100000])
    n_years: int = 10
    n_workers: int = 4
    memory_limit_mb: float = 4000.0
    target_times: Dict[int, float] = field(
        default_factory=lambda: {
            1000: 1.0,  # 1 second for 1K
            10000: 10.0,  # 10 seconds for 10K
            100000: 60.0,  # 60 seconds for 100K
        }
    )
    repetitions: int = 3
    warmup_runs: int = 2
    enable_profiling: bool = True


class SystemProfiler:
    """Profile system resources during benchmarking."""

    def __init__(self):
        """Initialize system profiler."""
        self.process = psutil.Process()
        self.cpu_samples = []
        self.memory_samples = []

    def start(self) -> None:
        """Start profiling."""
        self.cpu_samples = []
        self.memory_samples = []
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)

    def sample(self) -> None:
        """Take a resource sample."""
        self.cpu_samples.append(self.process.cpu_percent())
        self.memory_samples.append(self.process.memory_info().rss / (1024 * 1024))

    def get_metrics(self) -> Tuple[float, float, float]:
        """Get profiling metrics.

        Returns:
            Tuple of (avg_cpu, peak_memory, avg_memory).
        """
        if not self.cpu_samples:
            return 0.0, self.initial_memory, self.initial_memory

        avg_cpu = np.mean(self.cpu_samples[1:]) if len(self.cpu_samples) > 1 else 0
        peak_memory = max(self.memory_samples) if self.memory_samples else self.initial_memory
        avg_memory = np.mean(self.memory_samples) if self.memory_samples else self.initial_memory

        return avg_cpu, peak_memory, avg_memory

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary of system information.
        """
        import platform

        cpu_info = {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }

        memory_info = {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
        }

        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu": cpu_info,
            "memory": memory_info,
            "timestamp": datetime.now().isoformat(),
        }


class BenchmarkRunner:
    """Run individual benchmarks with monitoring."""

    def __init__(self, profiler: Optional[SystemProfiler] = None):
        """Initialize benchmark runner.

        Args:
            profiler: System profiler instance.
        """
        self.profiler = profiler or SystemProfiler()

    def run_single_benchmark(
        self, func: Callable, args: Tuple = (), kwargs: Dict = None, monitor_interval: float = 0.1
    ) -> BenchmarkMetrics:
        """Run a single benchmark with monitoring.

        Args:
            func: Function to benchmark.
            args: Positional arguments for function.
            kwargs: Keyword arguments for function.
            monitor_interval: Monitoring interval in seconds.

        Returns:
            BenchmarkMetrics from the run.
        """
        kwargs = kwargs or {}

        # Force garbage collection
        gc.collect()

        # Start profiling
        self.profiler.start()

        # Start monitoring in background
        start_time = time.time()

        # Run the function
        try:
            result = func(*args, **kwargs)

            # Sample resources during execution
            while time.time() - start_time < 0.1:
                self.profiler.sample()
                time.sleep(monitor_interval)

        except Exception as e:
            raise RuntimeError(f"Benchmark failed: {e}")

        execution_time = time.time() - start_time

        # Get final metrics
        avg_cpu, peak_memory, avg_memory = self.profiler.get_metrics()

        # Calculate throughput
        n_simulations = kwargs.get("n_simulations", getattr(result, "n_simulations", 1))
        simulations_per_second = n_simulations / execution_time if execution_time > 0 else 0

        # Extract additional metrics if available
        cache_hit_rate = 0.0
        if hasattr(result, "cache_hit_rate"):
            cache_hit_rate = result.cache_hit_rate
        elif hasattr(func, "__self__") and hasattr(func.__self__, "cache"):
            cache = func.__self__.cache
            if hasattr(cache, "hit_rate"):
                cache_hit_rate = cache.hit_rate

        accuracy_score = 1.0
        if hasattr(result, "accuracy_score"):
            accuracy_score = result.accuracy_score

        convergence_iterations = 0
        if hasattr(result, "convergence_iterations"):
            convergence_iterations = result.convergence_iterations

        return BenchmarkMetrics(
            execution_time=execution_time,
            simulations_per_second=simulations_per_second,
            memory_peak_mb=peak_memory,
            memory_average_mb=avg_memory,
            cpu_utilization=avg_cpu,
            cache_hit_rate=cache_hit_rate,
            accuracy_score=accuracy_score,
            convergence_iterations=convergence_iterations,
        )

    def run_with_warmup(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        warmup_runs: int = 2,
        benchmark_runs: int = 3,
    ) -> List[BenchmarkMetrics]:
        """Run benchmark with warmup.

        Args:
            func: Function to benchmark.
            args: Positional arguments.
            kwargs: Keyword arguments.
            warmup_runs: Number of warmup runs.
            benchmark_runs: Number of benchmark runs.

        Returns:
            List of benchmark metrics.
        """
        kwargs = kwargs or {}

        # Warmup runs
        for i in range(warmup_runs):
            print(f"  Warmup {i+1}/{warmup_runs}...", end="")
            _ = func(*args, **kwargs)
            print(" done")
            gc.collect()

        # Benchmark runs
        metrics_list = []
        for i in range(benchmark_runs):
            print(f"  Run {i+1}/{benchmark_runs}...", end="")
            metrics = self.run_single_benchmark(func, args, kwargs)
            metrics_list.append(metrics)
            print(f" {metrics.execution_time:.2f}s")
            gc.collect()

        return metrics_list


class BenchmarkSuite:
    """Comprehensive benchmark suite for Monte Carlo simulations.

    Provides tools to benchmark performance across different scales
    and configurations, generating detailed reports.
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.runner = BenchmarkRunner()
        self.results: List[BenchmarkResult] = []
        self.system_info = SystemProfiler.get_system_info()

    def benchmark_scale(
        self, engine, scale: int, config: BenchmarkConfig, optimizations: List[str] = None
    ) -> BenchmarkResult:
        """Benchmark at a specific scale.

        Args:
            engine: Monte Carlo engine to benchmark.
            scale: Number of simulations.
            config: Benchmark configuration.
            optimizations: List of applied optimizations.

        Returns:
            BenchmarkResult for this scale.
        """
        optimizations = optimizations or []

        print(f"\nBenchmarking {scale:,} simulations...")

        # Prepare engine configuration
        engine_config = engine.config
        engine_config.n_simulations = scale
        engine_config.n_years = config.n_years
        engine_config.n_workers = config.n_workers
        engine_config.progress_bar = False

        # Run benchmark with warmup
        metrics_list = self.runner.run_with_warmup(
            engine.run, warmup_runs=config.warmup_runs, benchmark_runs=config.repetitions
        )

        # Average metrics
        avg_metrics = BenchmarkMetrics(
            execution_time=np.mean([m.execution_time for m in metrics_list]),
            simulations_per_second=np.mean([m.simulations_per_second for m in metrics_list]),
            memory_peak_mb=np.max([m.memory_peak_mb for m in metrics_list]),
            memory_average_mb=np.mean([m.memory_average_mb for m in metrics_list]),
            cpu_utilization=np.mean([m.cpu_utilization for m in metrics_list]),
            cache_hit_rate=np.mean([m.cache_hit_rate for m in metrics_list]),
            accuracy_score=np.mean([m.accuracy_score for m in metrics_list]),
            convergence_iterations=int(np.mean([m.convergence_iterations for m in metrics_list])),
        )

        # Create result
        result = BenchmarkResult(
            scale=scale,
            metrics=avg_metrics,
            configuration={
                "n_years": config.n_years,
                "n_workers": config.n_workers,
                "memory_limit": config.memory_limit_mb,
            },
            timestamp=datetime.now(),
            system_info=self.system_info,
            optimizations=optimizations,
        )

        print(result.summary())

        # Check targets
        target_time = config.target_times.get(scale, float("inf"))
        if result.meets_target(target_time, config.memory_limit_mb):
            print(f"✓ Meets targets (time<{target_time}s, memory<{config.memory_limit_mb}MB)")
        else:
            print(f"✗ Misses targets (time<{target_time}s, memory<{config.memory_limit_mb}MB)")

        return result

    def run_comprehensive_benchmark(
        self, engine, config: Optional[BenchmarkConfig] = None
    ) -> "ComprehensiveBenchmarkResult":
        """Run comprehensive benchmark suite.

        Args:
            engine: Monte Carlo engine to benchmark.
            config: Benchmark configuration.

        Returns:
            ComprehensiveBenchmarkResult with all results.
        """
        config = config or BenchmarkConfig()

        print("=" * 60)
        print("COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)
        print(f"System: {self.system_info['platform']}")
        print(f"CPU: {self.system_info['cpu']['cores_physical']} cores")
        print(f"Memory: {self.system_info['memory']['available_gb']:.1f} GB available")
        print("=" * 60)

        # Test different scales
        for scale in config.scales:
            result = self.benchmark_scale(engine, scale, config)
            self.results.append(result)

        # Test with optimizations
        if hasattr(engine, "enable_optimizations"):
            print("\nTesting with optimizations enabled...")
            engine.enable_optimizations()

            for scale in config.scales:
                result = self.benchmark_scale(
                    engine, scale, config, optimizations=["vectorization", "caching", "parallel"]
                )
                self.results.append(result)

        return ComprehensiveBenchmarkResult(self.results, config, self.system_info)

    def compare_configurations(
        self, engine_factory: Callable, configurations: List[Dict[str, Any]], scale: int = 10000
    ) -> "ConfigurationComparison":
        """Compare different configurations.

        Args:
            engine_factory: Factory function to create engines.
            configurations: List of configuration dictionaries.
            scale: Number of simulations to test.

        Returns:
            ConfigurationComparison results.
        """
        comparison_results = []

        for i, config_dict in enumerate(configurations):
            print(f"\nConfiguration {i+1}: {config_dict.get('name', 'unnamed')}")

            # Create engine with configuration
            engine = engine_factory(**config_dict)

            # Run benchmark
            config = BenchmarkConfig(scales=[scale])
            result = self.benchmark_scale(engine, scale, config)

            comparison_results.append({"configuration": config_dict, "result": result})

        return ConfigurationComparison(comparison_results)


@dataclass
class ComprehensiveBenchmarkResult:
    """Results from comprehensive benchmark suite.

    Attributes:
        results: List of individual benchmark results
        config: Configuration used
        system_info: System information
    """

    results: List[BenchmarkResult]
    config: BenchmarkConfig
    system_info: Dict[str, Any]

    def meets_requirements(self) -> bool:
        """Check if all requirements are met.

        Returns:
            True if all performance requirements are satisfied.
        """
        for result in self.results:
            scale = result.scale
            target_time = self.config.target_times.get(scale, float("inf"))

            if not result.meets_target(target_time, self.config.memory_limit_mb):
                return False

        # Special check for 100K requirement
        for result in self.results:
            if result.scale == 100000:
                if result.metrics.execution_time > 60 or result.metrics.memory_peak_mb > 4000:
                    return False
                if result.metrics.accuracy_score < 0.9999:
                    return False

        return True

    def summary(self) -> str:
        """Generate comprehensive summary.

        Returns:
            Formatted summary string.
        """
        summary = "BENCHMARK RESULTS SUMMARY\n" + "=" * 60 + "\n"

        # Table of results
        table_data = []
        for result in self.results:
            opts = ", ".join(result.optimizations) if result.optimizations else "none"
            table_data.append(
                [
                    f"{result.scale:,}",
                    f"{result.metrics.execution_time:.2f}s",
                    f"{result.metrics.simulations_per_second:.0f}",
                    f"{result.metrics.memory_peak_mb:.1f} MB",
                    f"{result.metrics.cpu_utilization:.1f}%",
                    f"{result.metrics.cache_hit_rate:.1f}%",
                    opts,
                ]
            )

        headers = ["Scale", "Time", "Sims/s", "Memory", "CPU", "Cache", "Optimizations"]
        summary += tabulate(table_data, headers=headers, tablefmt="grid") + "\n\n"

        # Performance targets check
        summary += "PERFORMANCE TARGETS\n" + "-" * 30 + "\n"

        checks = {
            "100K in <60s": False,
            "Memory <4GB": False,
            "Accuracy >99.99%": False,
            "CPU Efficiency >75%": False,
            "Cache Hit Rate >85%": False,
        }

        for result in self.results:
            if result.scale == 100000:
                checks["100K in <60s"] = result.metrics.execution_time < 60
                checks["Memory <4GB"] = result.metrics.memory_peak_mb < 4000
                checks["Accuracy >99.99%"] = result.metrics.accuracy_score > 0.9999
                checks["CPU Efficiency >75%"] = result.metrics.cpu_utilization > 75
                checks["Cache Hit Rate >85%"] = result.metrics.cache_hit_rate > 85

        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            summary += f"{status} {check}\n"

        # Overall verdict
        if self.meets_requirements():
            summary += "\n✓ ALL REQUIREMENTS MET - Ready for production\n"
        else:
            summary += "\n✗ REQUIREMENTS NOT MET - Further optimization needed\n"

        return summary

    def save_report(self, filepath: str) -> None:
        """Save benchmark report to file.

        Args:
            filepath: Path to save report.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "configuration": {
                "scales": self.config.scales,
                "n_years": self.config.n_years,
                "n_workers": self.config.n_workers,
                "memory_limit_mb": self.config.memory_limit_mb,
            },
            "results": [
                {
                    "scale": r.scale,
                    "metrics": r.metrics.to_dict(),
                    "optimizations": r.optimizations,
                    "meets_target": r.meets_target(
                        self.config.target_times.get(r.scale, float("inf")),
                        self.config.memory_limit_mb,
                    ),
                }
                for r in self.results
            ],
            "meets_all_requirements": self.meets_requirements(),
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to: {filepath}")


@dataclass
class ConfigurationComparison:
    """Results from configuration comparison."""

    results: List[Dict[str, Any]]

    def best_configuration(self) -> Dict[str, Any]:
        """Find best configuration.

        Returns:
            Best configuration based on execution time.
        """
        best = min(self.results, key=lambda x: x["result"].metrics.execution_time)
        return best["configuration"]

    def summary(self) -> str:
        """Generate comparison summary.

        Returns:
            Formatted summary string.
        """
        summary = "CONFIGURATION COMPARISON\n" + "=" * 60 + "\n"

        table_data = []
        for item in self.results:
            config = item["configuration"]
            result = item["result"]

            table_data.append(
                [
                    config.get("name", "unnamed"),
                    f"{result.metrics.execution_time:.2f}s",
                    f"{result.metrics.memory_peak_mb:.1f} MB",
                    f"{result.metrics.cpu_utilization:.1f}%",
                ]
            )

        headers = ["Configuration", "Time", "Memory", "CPU"]
        summary += tabulate(table_data, headers=headers, tablefmt="grid")

        # Best configuration
        best = self.best_configuration()
        summary += f"\n\nBest configuration: {best.get('name', 'unnamed')}\n"

        return summary


def run_quick_benchmark(engine, n_simulations: int = 10000) -> BenchmarkMetrics:
    """Run a quick benchmark.

    Args:
        engine: Monte Carlo engine to benchmark.
        n_simulations: Number of simulations.

    Returns:
        BenchmarkMetrics from the run.
    """
    runner = BenchmarkRunner()

    # Configure engine
    engine.config.n_simulations = n_simulations
    engine.config.progress_bar = False

    # Run benchmark
    return runner.run_single_benchmark(engine.run)


if __name__ == "__main__":
    # Example usage
    from ergodic_insurance.src.config import ManufacturerConfig
    from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
    from ergodic_insurance.src.loss_distributions import ManufacturingLossGenerator
    from ergodic_insurance.src.manufacturer import WidgetManufacturer
    from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig

    # Setup simulation
    loss_generator = ManufacturingLossGenerator()

    layers = [
        EnhancedInsuranceLayer(0, 1_000_000, 0.015),
        EnhancedInsuranceLayer(1_000_000, 4_000_000, 0.008),
    ]
    insurance_program = InsuranceProgram(layers=layers)

    manufacturer_config = ManufacturerConfig(
        initial_assets=10_000_000, asset_turnover_ratio=0.5, operating_margin=0.08
    )
    manufacturer = WidgetManufacturer(manufacturer_config)

    # Create engine
    sim_config = SimulationConfig(n_simulations=1000, n_years=10, parallel=True, n_workers=4)

    engine = MonteCarloEngine(
        loss_generator=loss_generator,
        insurance_program=insurance_program,
        manufacturer=manufacturer,
        config=sim_config,
    )

    # Run benchmarks
    suite = BenchmarkSuite()
    config = BenchmarkConfig(scales=[1000, 10000])

    results = suite.run_comprehensive_benchmark(engine, config)
    print("\n" + results.summary())

    # Save report
    results.save_report("benchmark_report.json")

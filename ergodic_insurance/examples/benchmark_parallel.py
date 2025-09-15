"""Performance benchmark for enhanced parallel Monte Carlo execution.

This script benchmarks the performance of the enhanced parallel executor
across various configurations and workloads, demonstrating efficiency
on budget hardware (4-8 cores).

Usage:
    python benchmark_parallel.py [--simulations N] [--years Y] [--workers W]

Author:
    Alex Filiakov

Date:
    2025-08-26
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil
from tabulate import tabulate

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig
from ergodic_insurance.src.parallel_executor import CPUProfile


def get_system_info() -> Dict:
    """Get system information for benchmark context.

    Returns:
        Dict: System information including CPU, memory, and OS details
    """
    cpu_profile = CPUProfile.detect()

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_cores_physical": cpu_profile.n_cores,
        "cpu_cores_logical": cpu_profile.n_threads,
        "cpu_freq_mhz": cpu_profile.cpu_freq,
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": cpu_profile.available_memory / (1024**3),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
    }


def setup_simulation(
    n_simulations: int, n_years: int
) -> Tuple[ManufacturingLossGenerator, InsuranceProgram, WidgetManufacturer]:
    """Set up simulation components.

    Args:
        n_simulations: Number of simulations to run
        n_years: Number of years per simulation

    Returns:
        Tuple: (loss_generator, insurance_program, manufacturer)
    """
    # Create loss generator
    loss_generator = ManufacturingLossGenerator(
        frequency_params={"lambda": 3.0}, severity_params={"mu": 10, "sigma": 2}
    )

    # Create insurance program with multiple layers
    layers = [
        EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, premium_rate=0.015),
        EnhancedInsuranceLayer(attachment_point=1_000_000, limit=4_000_000, premium_rate=0.008),
        EnhancedInsuranceLayer(attachment_point=5_000_000, limit=20_000_000, premium_rate=0.004),
    ]
    insurance_program = InsuranceProgram(layers=layers)

    # Create manufacturer
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.8,
    )
    manufacturer = WidgetManufacturer(config)

    return loss_generator, insurance_program, manufacturer


def benchmark_configuration(
    loss_generator: ManufacturingLossGenerator,
    insurance_program: InsuranceProgram,
    manufacturer: WidgetManufacturer,
    config: SimulationConfig,
    label: str,
) -> Dict:
    """Run benchmark for a specific configuration.

    Args:
        loss_generator: Loss generator
        insurance_program: Insurance program
        manufacturer: Manufacturer model
        config: Simulation configuration
        label: Configuration label

    Returns:
        Dict: Benchmark results
    """
    print(f"\nBenchmarking: {label}")
    print("-" * 50)

    # Create engine
    engine = MonteCarloEngine(
        loss_generator=loss_generator,
        insurance_program=insurance_program,
        manufacturer=manufacturer,
        config=config,
    )

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**2)  # MB

    # Run simulation
    start_time = time.time()
    results = engine.run()
    total_time = time.time() - start_time

    # Get final memory
    final_memory = process.memory_info().rss / (1024**2)  # MB
    memory_used = final_memory - initial_memory

    # Extract metrics
    benchmark_results = {
        "label": label,
        "n_simulations": config.n_simulations,
        "n_years": config.n_years,
        "n_workers": config.n_workers,
        "use_enhanced": config.use_enhanced_parallel,
        "total_time": total_time,
        "simulations_per_second": config.n_simulations / total_time,
        "memory_used_mb": memory_used,
        "memory_per_simulation": memory_used / config.n_simulations * 1000,  # KB
    }

    # Add performance metrics if available
    if results.performance_metrics:
        perf = results.performance_metrics
        benchmark_results.update(
            {
                "setup_time": perf.setup_time,
                "computation_time": perf.computation_time,
                "serialization_time": perf.serialization_time,
                "serialization_overhead": perf.serialization_time / total_time * 100,
                "reduction_time": perf.reduction_time,
                "cpu_utilization": perf.cpu_utilization,
                "speedup": perf.speedup,
            }
        )

    # Add result quality metrics
    benchmark_results.update(
        {
            "ruin_probability": results.ruin_probability,
            "mean_growth_rate": np.mean(results.growth_rates),
            "convergence_r_hat": results.convergence.get(
                "growth_rate", type("", (), {"r_hat": 0})()
            ).r_hat,
        }
    )

    print(f"Completed in {total_time:.2f}s ({config.n_simulations / total_time:.0f} sims/s)")
    print(f"Memory used: {memory_used:.1f} MB")

    return benchmark_results


def run_benchmarks(n_simulations: int, n_years: int, max_workers: int = None) -> List[Dict]:
    """Run comprehensive benchmarks.

    Args:
        n_simulations: Number of simulations
        n_years: Years per simulation
        max_workers: Maximum number of workers to test

    Returns:
        List[Dict]: Benchmark results
    """
    # Setup simulation components
    loss_generator, insurance_program, manufacturer = setup_simulation(n_simulations, n_years)

    # Determine max workers
    if max_workers is None:
        max_workers = min(psutil.cpu_count(logical=False), 8)

    results = []

    # Benchmark configurations
    configs = [
        # Sequential baseline
        (
            SimulationConfig(
                n_simulations=n_simulations,
                n_years=n_years,
                parallel=False,
                use_enhanced_parallel=False,
                progress_bar=False,
                seed=42,
            ),
            "Sequential (Baseline)",
        ),
        # Standard parallel with different worker counts
        *[
            (
                SimulationConfig(
                    n_simulations=n_simulations,
                    n_years=n_years,
                    parallel=True,
                    use_enhanced_parallel=False,
                    n_workers=n_workers,
                    progress_bar=False,
                    seed=42,
                ),
                f"Standard Parallel ({n_workers} workers)",
            )
            for n_workers in [2, 4, min(8, max_workers)]
        ],
        # Enhanced parallel with different worker counts
        *[
            (
                SimulationConfig(
                    n_simulations=n_simulations,
                    n_years=n_years,
                    parallel=True,
                    use_enhanced_parallel=True,
                    monitor_performance=True,
                    adaptive_chunking=True,
                    shared_memory=True,
                    n_workers=n_workers,
                    progress_bar=False,
                    seed=42,
                ),
                f"Enhanced Parallel ({n_workers} workers)",
            )
            for n_workers in [2, 4, min(8, max_workers)]
        ],
        # Enhanced with optimizations disabled
        (
            SimulationConfig(
                n_simulations=n_simulations,
                n_years=n_years,
                parallel=True,
                use_enhanced_parallel=True,
                monitor_performance=True,
                adaptive_chunking=False,  # Disabled
                shared_memory=False,  # Disabled
                n_workers=4,
                progress_bar=False,
                seed=42,
            ),
            "Enhanced (No Optimizations)",
        ),
    ]

    # Run benchmarks
    for config, label in configs:
        try:
            result = benchmark_configuration(
                loss_generator, insurance_program, manufacturer, config, label
            )
            results.append(result)
        except Exception as e:
            print(f"Error in {label}: {e}")
            continue

    return results


def analyze_results(results: List[Dict]) -> None:
    """Analyze and display benchmark results.

    Args:
        results: List of benchmark results
    """
    if not results:
        print("No results to analyze")
        return

    # Find baseline
    baseline = next((r for r in results if "Sequential" in r["label"]), results[0])
    baseline_time = baseline["total_time"]

    # Calculate speedups
    for result in results:
        result["speedup_vs_baseline"] = baseline_time / result["total_time"]

    # Create summary table
    table_data = []
    for r in results:
        table_data.append(
            [
                r["label"],
                f"{r['total_time']:.2f}s",
                f"{r['simulations_per_second']:.0f}",
                f"{r['speedup_vs_baseline']:.2f}x",
                f"{r['memory_used_mb']:.1f} MB",
                f"{r.get('serialization_overhead', 0):.1f}%",
                f"{r.get('cpu_utilization', 0):.1f}%",
            ]
        )

    headers = ["Configuration", "Time", "Sims/s", "Speedup", "Memory", "IPC Overhead", "CPU%"]
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Find best configuration
    best = max(results, key=lambda r: r["simulations_per_second"])
    print(f"\nBest Configuration: {best['label']}")
    print(f"  - {best['simulations_per_second']:.0f} simulations/second")
    print(f"  - {best['speedup_vs_baseline']:.2f}x speedup vs baseline")
    print(f"  - {best['memory_used_mb']:.1f} MB memory used")

    # Check if we meet requirements
    print("\n" + "=" * 80)
    print("REQUIREMENTS CHECK")
    print("=" * 80)

    # Check 100K simulations efficiency
    for r in results:
        if r["n_simulations"] >= 100000 and "Enhanced" in r["label"] and "4" in r["label"]:
            print(f"✓ 100K simulations on 4 cores: {r['total_time']:.1f}s")
            break

    # Check memory usage
    max_memory = max(r["memory_used_mb"] for r in results)
    if max_memory < 4096:
        print(f"✓ Memory usage < 4GB: {max_memory:.1f} MB")
    else:
        print(f"✗ Memory usage > 4GB: {max_memory:.1f} MB")

    # Check serialization overhead
    enhanced_results = [r for r in results if "Enhanced Parallel" in r["label"]]
    if enhanced_results:
        avg_overhead = np.mean([r.get("serialization_overhead", 0) for r in enhanced_results])
        if avg_overhead < 5:
            print(f"✓ Serialization overhead < 5%: {avg_overhead:.2f}%")
        else:
            print(f"✗ Serialization overhead > 5%: {avg_overhead:.2f}%")

    # Check scaling
    worker_scaling = {}
    for r in results:
        if "Enhanced Parallel" in r["label"]:
            workers = r["n_workers"]
            worker_scaling[workers] = r["speedup_vs_baseline"]

    if worker_scaling:
        scaling_efficiency = []
        for workers, speedup in sorted(worker_scaling.items()):
            efficiency = speedup / workers * 100
            scaling_efficiency.append(efficiency)
            print(f"  {workers} workers: {speedup:.2f}x speedup ({efficiency:.1f}% efficiency)")

        avg_efficiency = np.mean(scaling_efficiency)
        if avg_efficiency > 60:
            print(f"✓ Near-linear scaling: {avg_efficiency:.1f}% average efficiency")
        else:
            print(f"⚠ Sub-linear scaling: {avg_efficiency:.1f}% average efficiency")


def save_results(results: List[Dict], system_info: Dict, output_file: str = None) -> None:
    """Save benchmark results to file.

    Args:
        results: Benchmark results
        system_info: System information
        output_file: Output file path
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"

    data = {
        "system_info": system_info,
        "results": results,
        "summary": {
            "best_configuration": max(results, key=lambda r: r["simulations_per_second"])["label"],
            "max_throughput": max(r["simulations_per_second"] for r in results),
            "min_memory": min(r["memory_used_mb"] for r in results),
            "max_speedup": max(r.get("speedup_vs_baseline", 1) for r in results),
        },
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark parallel Monte Carlo execution")
    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=100000,
        help="Number of simulations (default: 100000)",
    )
    parser.add_argument(
        "--years", "-y", type=int, default=10, help="Years per simulation (default: 10)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Maximum workers to test (default: auto)"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file for results")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer simulations"
    )

    args = parser.parse_args()

    # Adjust for quick mode
    if args.quick:
        args.simulations = min(args.simulations, 10000)

    # Print system info
    print("=" * 80)
    print("PARALLEL MONTE CARLO BENCHMARK")
    print("=" * 80)

    system_info = get_system_info()
    print(f"System: {system_info['platform']}")
    print(
        f"CPU: {system_info['cpu_cores_physical']} cores / {system_info['cpu_cores_logical']} threads"
    )
    print(f"Memory: {system_info['memory_available_gb']:.1f} GB available")
    print(f"Python: {system_info['python_version']}")
    print(f"\nBenchmarking {args.simulations:,} simulations × {args.years} years")
    print("=" * 80)

    # Run benchmarks
    results = run_benchmarks(args.simulations, args.years, args.workers)

    # Analyze results
    analyze_results(results)

    # Save results
    if args.output or not args.quick:
        save_results(results, system_info, args.output)


if __name__ == "__main__":
    main()

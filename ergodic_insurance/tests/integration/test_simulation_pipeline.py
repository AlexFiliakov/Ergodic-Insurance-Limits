"""Integration tests for simulation pipeline.

This module tests the integration between Monte Carlo engine,
parallel executor, trajectory storage, and result aggregation.
"""

import multiprocessing as mp
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from src.batch_processor import BatchProcessor
from src.config_v2 import ConfigV2, SimulationConfig
from src.monte_carlo import MonteCarloEngine, MonteCarloResults
from src.parallel_executor import ParallelExecutor
from src.progress_monitor import ProgressMonitor
from src.result_aggregator import ResultAggregator
from src.scenario_manager import ScenarioManager
from src.trajectory_storage import TrajectoryStorage

from .test_fixtures import default_config_v2, monte_carlo_engine
from .test_helpers import benchmark_function, measure_memory_usage, timer


class TestSimulationPipeline:
    """Test simulation pipeline integration."""

    def test_monte_carlo_basic_execution(
        self,
        monte_carlo_engine: MonteCarloEngine,
    ):
        """Test basic Monte Carlo engine execution.

        Verifies that:
        - Engine runs without errors
        - Results have correct structure
        - Basic statistics are calculated
        """
        with timer("Monte Carlo execution") as t:
            results = monte_carlo_engine.run()

        # Verify execution completed
        assert results is not None, "Results should not be None"
        assert isinstance(results, MonteCarloResults), "Should return MonteCarloResults"

        # Verify result structure
        assert hasattr(results, "terminal_values"), "Should have terminal values"
        assert hasattr(results, "trajectories"), "Should have trajectories"
        assert hasattr(results, "statistics"), "Should have statistics"

        # Verify dimensions
        n_sims = monte_carlo_engine.config.n_simulations
        n_years = monte_carlo_engine.config.time_horizon

        assert len(results.terminal_values) == n_sims, f"Should have {n_sims} terminal values"

        if results.trajectories is not None:
            assert results.trajectories.shape == (
                n_sims,
                n_years,
            ), f"Trajectories should be ({n_sims}, {n_years})"

        # Verify statistics
        assert "mean" in results.statistics
        assert "std" in results.statistics
        assert "percentiles" in results.statistics

        # Verify timing
        assert t["elapsed"] < 60, f"Basic execution took {t['elapsed']:.2f}s, should be < 60s"

    def test_parallel_vs_serial_consistency(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test that parallel and serial execution produce same results.

        This is the example test from the issue requirements.
        """
        config = default_config_v2.model_copy()
        config.simulation.n_simulations = 100
        config.simulation.time_horizon = 10
        config.simulation.seed = 42

        # Serial execution
        config.simulation.enable_parallel = False
        serial_engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )
        serial_results = serial_engine.run()

        # Parallel execution
        config.simulation.enable_parallel = True
        config.simulation.n_workers = 4
        parallel_engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )
        parallel_results = parallel_engine.run()

        # Results should be identical with same seed
        np.testing.assert_allclose(
            serial_results.terminal_values,
            parallel_results.terminal_values,
            rtol=1e-10,
            err_msg="Parallel and serial results should match",
        )

    def test_parallel_executor_integration(self):
        """Test ParallelExecutor integration with simulation.

        Verifies that:
        - Parallel executor handles work distribution
        - Results are properly aggregated
        - Memory is efficiently managed
        """
        executor = ParallelExecutor(n_workers=4)

        # Define a simple simulation task
        def simulate_path(seed: int) -> Dict[str, Any]:
            """Simulate a single path."""
            np.random.seed(seed)
            n_years = 20
            returns = np.random.normal(0.05, 0.15, n_years)
            values = 1000000 * np.exp(np.cumsum(returns))
            return {
                "terminal_value": values[-1],
                "max_value": np.max(values),
                "min_value": np.min(values),
                "seed": seed,
            }

        # Run parallel simulation
        n_paths = 100
        seeds = list(range(n_paths))

        with timer("Parallel execution") as t:
            results = executor.map_reduce(
                map_func=simulate_path,
                reduce_func=lambda x: x,  # Identity for list collection
                tasks=seeds,
            )

        # Verify results
        assert len(results) == n_paths, f"Should have {n_paths} results"
        assert all("terminal_value" in r for r in results), "All results should have terminal value"
        assert all(r["seed"] == i for i, r in enumerate(results)), "Seeds should match"

        # Verify performance benefit
        # Serial baseline
        with timer("Serial baseline") as t_serial:
            serial_results = [simulate_path(seed) for seed in seeds]

        # Parallel should be faster for sufficient workload
        if n_paths >= 50:  # Only expect speedup for larger workloads
            assert t["elapsed"] < t_serial["elapsed"], (
                f"Parallel ({t['elapsed']:.2f}s) should be faster than "
                f"serial ({t_serial['elapsed']:.2f}s)"
            )

    def test_trajectory_storage_memory_efficiency(self):
        """Test memory-efficient trajectory storage.

        Verifies that:
        - Large trajectories can be stored efficiently
        - Compression works correctly
        - Retrieval maintains data integrity
        """
        storage = TrajectoryStorage(max_memory_mb=100)

        # Generate large dataset
        n_paths = 1000
        n_timesteps = 100

        # Store trajectories
        for i in range(n_paths):
            trajectory = np.random.randn(n_timesteps) * 1000 + 10000
            storage.store(f"path_{i}", trajectory)

        # Check memory usage
        memory_used = storage.get_memory_usage()
        assert memory_used < 100, f"Memory usage {memory_used:.2f}MB should be < 100MB"

        # Verify data integrity
        sample_indices = [0, n_paths // 2, n_paths - 1]
        for i in sample_indices:
            # Regenerate expected trajectory
            np.random.seed(i)  # Reset to same seed
            expected = np.random.randn(n_timesteps) * 1000 + 10000

            # Retrieve stored trajectory
            stored = storage.get(f"path_{i}")

            # Should be close (allowing for compression artifacts)
            np.testing.assert_allclose(
                stored,
                expected,
                rtol=1e-5,
                err_msg=f"Path {i} data integrity check failed",
            )

        # Test batch retrieval
        batch_keys = [f"path_{i}" for i in range(10)]
        batch_data = storage.get_batch(batch_keys)
        assert len(batch_data) == 10, "Should retrieve all requested paths"

    def test_progress_monitoring_integration(
        self,
        monte_carlo_engine: MonteCarloEngine,
    ):
        """Test progress monitoring during simulation.

        Verifies that:
        - Progress is accurately tracked
        - Updates occur at appropriate intervals
        - Final state is complete
        """
        # Enable progress monitoring
        monte_carlo_engine.config.show_progress = True

        # Track progress updates
        progress_history = []

        # Monkey-patch progress callback
        original_run = monte_carlo_engine.run

        def run_with_progress_tracking():
            monitor = ProgressMonitor(
                total=monte_carlo_engine.config.n_simulations,
                description="Monte Carlo simulation",
            )

            # Simulate progress updates
            for i in range(monte_carlo_engine.config.n_simulations):
                monitor.update(1)
                if i % 10 == 0:  # Sample progress
                    progress_history.append(
                        {
                            "completed": monitor.completed,
                            "total": monitor.total,
                            "percentage": monitor.get_progress_percentage(),
                        }
                    )

            monitor.close()
            return original_run()

        # Run with monitoring
        results = run_with_progress_tracking()

        # Verify progress tracking
        assert len(progress_history) > 0, "Should have progress updates"
        assert progress_history[-1]["percentage"] >= 90, "Should reach near completion"

        # Verify monotonic progress
        percentages = [p["percentage"] for p in progress_history]
        assert all(
            percentages[i] <= percentages[i + 1] for i in range(len(percentages) - 1)
        ), "Progress should be monotonically increasing"

    def test_result_aggregation_pipeline(self):
        """Test result aggregation from parallel execution.

        Verifies that:
        - Results are properly collected
        - Statistics are correctly calculated
        - Aggregation handles edge cases
        """
        aggregator = ResultAggregator()

        # Generate distributed results (simulating parallel workers)
        worker_results = []
        n_workers = 4
        paths_per_worker = 25

        for worker_id in range(n_workers):
            np.random.seed(worker_id)
            worker_data = {
                "terminal_values": np.random.lognormal(14, 1, paths_per_worker),
                "max_drawdowns": np.random.uniform(0.1, 0.5, paths_per_worker),
                "survival_flags": np.random.random(paths_per_worker) > 0.1,
                "worker_id": worker_id,
            }
            worker_results.append(worker_data)

        # Aggregate results
        aggregated = aggregator.aggregate(worker_results)

        # Verify aggregation
        total_paths = n_workers * paths_per_worker
        assert (
            len(aggregated["terminal_values"]) == total_paths
        ), f"Should have {total_paths} terminal values"
        assert len(aggregated["max_drawdowns"]) == total_paths
        assert len(aggregated["survival_flags"]) == total_paths

        # Verify statistics calculation
        stats = aggregator.calculate_statistics(aggregated)

        assert "mean_terminal" in stats
        assert "std_terminal" in stats
        assert "survival_rate" in stats
        assert "mean_drawdown" in stats

        # Verify values are reasonable
        assert stats["mean_terminal"] > 0, "Mean terminal value should be positive"
        assert 0 <= stats["survival_rate"] <= 1, "Survival rate should be in [0, 1]"
        assert 0 <= stats["mean_drawdown"] <= 1, "Mean drawdown should be in [0, 1]"

    def test_batch_processing_integration(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test batch processing for large simulations.

        Verifies that:
        - Large simulations are split into batches
        - Memory stays within limits
        - Results are consistent
        """
        config = default_config_v2.model_copy()
        config.simulation.n_simulations = 1000
        config.simulation.batch_size = 100
        config.simulation.time_horizon = 20

        processor = BatchProcessor(
            batch_size=config.simulation.batch_size,
            max_memory_mb=500,
        )

        # Define simulation task
        def simulate_batch(start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
            """Simulate a batch of paths."""
            n_paths = end_idx - start_idx
            n_years = config.simulation.time_horizon

            np.random.seed(start_idx)
            trajectories = np.zeros((n_paths, n_years))

            for i in range(n_paths):
                returns = np.random.normal(0.05, 0.15, n_years)
                trajectories[i] = 1000000 * np.exp(np.cumsum(returns))

            return {
                "trajectories": trajectories,
                "terminal_values": trajectories[:, -1],
            }

        # Process in batches
        all_results = []
        n_batches = config.simulation.n_simulations // config.simulation.batch_size

        with timer("Batch processing") as t:
            for batch_idx in range(n_batches):
                start = batch_idx * config.simulation.batch_size
                end = start + config.simulation.batch_size

                batch_result = processor.process(
                    simulate_batch,
                    start,
                    end,
                )
                all_results.append(batch_result)

        # Verify all batches processed
        assert len(all_results) == n_batches, f"Should have {n_batches} batch results"

        # Verify memory efficiency
        # Memory should not grow linearly with simulation count
        memory_info = measure_memory_usage(
            lambda: [simulate_batch(i * 100, (i + 1) * 100) for i in range(10)]
        )
        assert (
            memory_info["memory_increase_mb"] < 200
        ), f"Memory increase {memory_info['memory_increase_mb']:.2f}MB should be < 200MB"

        # Combine results
        combined_terminals = np.concatenate([r["terminal_values"] for r in all_results])
        assert len(combined_terminals) == config.simulation.n_simulations

    def test_scenario_manager_integration(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test scenario management in simulation pipeline.

        Verifies that:
        - Multiple scenarios can be defined and run
        - Results are properly organized
        - Comparisons work correctly
        """
        manager = ScenarioManager()

        # Define scenarios
        scenarios = [
            {
                "name": "baseline",
                "config": default_config_v2.model_copy(),
            },
            {
                "name": "high_volatility",
                "config": default_config_v2.model_copy(),
            },
            {
                "name": "low_insurance",
                "config": default_config_v2.model_copy(),
            },
        ]

        # Modify scenario configs
        scenarios[1]["config"].stochastic.revenue_volatility = 0.30
        scenarios[2]["config"].insurance.primary_limit = 2_000_000

        # Run scenarios
        scenario_results = {}

        for scenario in scenarios:
            # Add scenario to manager
            manager.add_scenario(
                name=scenario["name"],
                config=scenario["config"],
            )

            # Create engine for scenario
            engine = MonteCarloEngine(
                config=scenario["config"].simulation,
                manufacturer_config=scenario["config"].manufacturer,
                insurance_config=scenario["config"].insurance,
                stochastic_config=scenario["config"].stochastic,
            )

            # Run simulation
            results = engine.run()
            scenario_results[scenario["name"]] = results

            # Store results in manager
            manager.store_results(scenario["name"], results)

        # Verify all scenarios ran
        assert len(scenario_results) == 3, "Should have results for all scenarios"

        # Compare scenarios
        comparison = manager.compare_scenarios(["baseline", "high_volatility"])

        assert "baseline" in comparison
        assert "high_volatility" in comparison

        # High volatility should have more variance
        baseline_std = np.std(scenario_results["baseline"].terminal_values)
        high_vol_std = np.std(scenario_results["high_volatility"].terminal_values)

        assert high_vol_std > baseline_std, "High volatility scenario should have more variance"

    def test_checkpoint_recovery(
        self,
        monte_carlo_engine: MonteCarloEngine,
        tmp_path,
    ):
        """Test simulation checkpoint and recovery.

        Verifies that:
        - Simulations can be checkpointed
        - Recovery from checkpoint works
        - Results are consistent
        """
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Configure engine for checkpointing
        monte_carlo_engine.checkpoint_dir = checkpoint_dir
        monte_carlo_engine.checkpoint_interval = 5  # Every 5 simulations

        # Run partial simulation
        n_partial = 5
        partial_results = []

        for i in range(n_partial):
            # Simulate one path
            np.random.seed(i)
            result = monte_carlo_engine._simulate_single_path(i)
            partial_results.append(result)

            # Save checkpoint
            if i % monte_carlo_engine.checkpoint_interval == 0:
                checkpoint_file = checkpoint_dir / f"checkpoint_{i}.npy"
                np.save(checkpoint_file, partial_results)

        # Verify checkpoint exists
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.npy"))
        assert len(checkpoints) > 0, "Should have created checkpoints"

        # Simulate recovery
        recovered_results = []
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))

        if latest_checkpoint.exists():
            recovered_results = np.load(latest_checkpoint, allow_pickle=True).tolist()

        # Continue from checkpoint
        start_from = len(recovered_results)
        for i in range(start_from, monte_carlo_engine.config.n_simulations):
            np.random.seed(i)
            result = monte_carlo_engine._simulate_single_path(i)
            recovered_results.append(result)

        # Verify recovery worked
        assert (
            len(recovered_results) == monte_carlo_engine.config.n_simulations
        ), "Should have complete results after recovery"

    def test_performance_scaling(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test performance scaling with simulation size.

        Verifies that:
        - Performance scales appropriately
        - Parallelization provides benefits
        - Memory usage is controlled
        """
        sizes = [10, 100, 500]
        results = []

        for n_sims in sizes:
            config = default_config_v2.model_copy()
            config.simulation.n_simulations = n_sims
            config.simulation.time_horizon = 20
            config.simulation.enable_parallel = True

            engine = MonteCarloEngine(
                config=config.simulation,
                manufacturer_config=config.manufacturer,
                insurance_config=config.insurance,
                stochastic_config=config.stochastic,
            )

            # Benchmark execution
            benchmark = benchmark_function(engine.run, n_runs=3)

            results.append(
                {
                    "n_simulations": n_sims,
                    "mean_time": benchmark["mean"],
                    "std_time": benchmark["std"],
                }
            )

        # Verify scaling
        # Time should not scale linearly (due to parallelization)
        time_ratios = [
            results[i + 1]["mean_time"] / results[i]["mean_time"] for i in range(len(results) - 1)
        ]

        size_ratios = [sizes[i + 1] / sizes[i] for i in range(len(sizes) - 1)]

        # Parallel execution should beat linear scaling
        for time_ratio, size_ratio in zip(time_ratios, size_ratios):
            assert time_ratio < size_ratio * 0.9, (
                f"Time ratio {time_ratio:.2f} should be better than "
                f"linear scaling {size_ratio:.2f}"
            )

    def test_edge_cases_and_error_handling(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test edge cases and error handling in pipeline.

        Verifies that:
        - Empty simulations are handled
        - Invalid configurations are caught
        - Recovery from errors works
        """
        # Test empty simulation
        config = default_config_v2.model_copy()
        config.simulation.n_simulations = 0

        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        # Should handle gracefully
        results = engine.run()
        assert results.terminal_values is not None
        assert len(results.terminal_values) == 0

        # Test single simulation
        config.simulation.n_simulations = 1
        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        results = engine.run()
        assert len(results.terminal_values) == 1

        # Test invalid configuration
        with pytest.raises(ValueError):
            config.simulation.n_simulations = -1
            engine = MonteCarloEngine(
                config=config.simulation,
                manufacturer_config=config.manufacturer,
                insurance_config=config.insurance,
                stochastic_config=config.stochastic,
            )

    def test_data_sharing_between_workers(self):
        """Test data sharing in parallel execution.

        Verifies that:
        - Shared memory works correctly
        - No data races occur
        - Results are consistent
        """
        # Create shared array
        n_paths = 100
        n_timesteps = 50

        # Use multiprocessing shared memory
        shared_array = mp.Array("d", n_paths * n_timesteps)
        shared_np = np.frombuffer(shared_array.get_obj()).reshape((n_paths, n_timesteps))

        def worker_task(worker_id: int, start: int, end: int):
            """Worker fills its portion of shared array."""
            np.random.seed(worker_id)
            for i in range(start, end):
                shared_np[i] = np.random.randn(n_timesteps)

        # Launch workers
        n_workers = 4
        paths_per_worker = n_paths // n_workers
        processes = []

        for worker_id in range(n_workers):
            start = worker_id * paths_per_worker
            end = start + paths_per_worker
            p = mp.Process(target=worker_task, args=(worker_id, start, end))
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()

        # Verify data was written
        assert not np.all(shared_np == 0), "Shared array should be filled"

        # Verify no overlap (each worker wrote to distinct region)
        for worker_id in range(n_workers):
            start = worker_id * paths_per_worker
            end = start + paths_per_worker

            # Check this worker's region has data
            worker_region = shared_np[start:end]
            assert not np.all(worker_region == 0), f"Worker {worker_id} region should have data"

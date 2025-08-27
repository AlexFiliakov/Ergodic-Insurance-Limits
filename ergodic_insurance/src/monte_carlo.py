"""High-performance Monte Carlo simulation engine for insurance optimization."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from tqdm import tqdm

from .bootstrap_analysis import BootstrapAnalyzer, bootstrap_confidence_interval
from .convergence import ConvergenceDiagnostics, ConvergenceStats
from .insurance_program import InsuranceProgram
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .parallel_executor import (
    ChunkingStrategy,
    ParallelExecutor,
    PerformanceMetrics,
    SharedMemoryConfig,
)
from .progress_monitor import ProgressMonitor
from .result_aggregator import (
    AggregationConfig,
    PercentileTracker,
    ResultAggregator,
    ResultExporter,
    TimeSeriesAggregator,
)
from .risk_metrics import RiskMetrics
from .ruin_probability import RuinProbabilityAnalyzer, RuinProbabilityConfig, RuinProbabilityResults
from .summary_statistics import SummaryReportGenerator, SummaryStatistics
from .trajectory_storage import StorageConfig, TrajectoryStorage


def _create_manufacturer(config_dict: Dict[str, Any]) -> Any:
    """Create manufacturer instance from config dictionary."""
    from ergodic_insurance.src.manufacturer import WidgetManufacturer as WM

    if "config" in config_dict and hasattr(config_dict["config"], "__dict__"):
        return WM(config_dict["config"])
    # Create from raw values
    manufacturer = WM.__new__(WM)
    for key, value in config_dict.items():
        setattr(manufacturer, key, value)
    return manufacturer


def _simulate_year_losses(sim_id: int, year: int) -> Tuple[float, float, float]:
    """Simulate losses for a single year."""
    np.random.seed(sim_id * 1000 + year)  # Ensure reproducibility
    n_events = np.random.poisson(3)  # Average 3 events per year
    event_amounts = np.random.lognormal(10, 2, n_events)  # Log-normal losses
    total_loss = np.sum(event_amounts)

    # Apply insurance (simplified)
    recovery = min(total_loss, 1_000_000) * 0.9  # Simplified recovery
    retained = total_loss - recovery

    return total_loss, recovery, retained


def _simulate_path_enhanced(sim_id: int, **shared) -> Dict[str, Any]:
    """Enhanced simulation function for parallel execution.

    Module-level function for pickle compatibility in multiprocessing.

    Args:
        sim_id: Simulation ID for seeding
        **shared: Shared data from parallel executor

    Returns:
        Dict with simulation results
    """
    # Add parent directory to path for imports in worker processes
    import sys

    module_path = Path(__file__).parent.parent.parent
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

    # Create manufacturer instance
    manufacturer = _create_manufacturer(shared["manufacturer_config"])

    # Initialize simulation arrays
    n_years = shared["n_years"]
    dtype = np.float32 if shared["use_float32"] else np.float64

    result_arrays = {
        "annual_losses": np.zeros(n_years, dtype=dtype),
        "insurance_recoveries": np.zeros(n_years, dtype=dtype),
        "retained_losses": np.zeros(n_years, dtype=dtype),
    }

    # Simulate years
    for year in range(n_years):
        # Generate and process losses
        total_loss, recovery, retained = _simulate_year_losses(sim_id, year)

        result_arrays["annual_losses"][year] = total_loss
        result_arrays["insurance_recoveries"][year] = recovery
        result_arrays["retained_losses"][year] = retained

        # Update manufacturer
        manufacturer.assets *= 1.0 - retained / manufacturer.assets

        # Check for ruin
        if manufacturer.assets <= 0:
            break

    return {"final_assets": manufacturer.assets, **result_arrays}


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation.

    Attributes:
        n_simulations: Number of simulation paths
        n_years: Number of years per simulation
        n_chains: Number of parallel chains for convergence
        parallel: Whether to use multiprocessing
        n_workers: Number of parallel workers (None for auto)
        chunk_size: Size of chunks for parallel processing
        use_float32: Use float32 for memory efficiency
        cache_results: Cache intermediate results
        checkpoint_interval: Save checkpoint every N simulations
        progress_bar: Show progress bar
        seed: Random seed for reproducibility
        use_enhanced_parallel: Use enhanced parallel executor for better performance
        monitor_performance: Track detailed performance metrics
        adaptive_chunking: Enable adaptive chunk sizing
        shared_memory: Enable shared memory for read-only data
    """

    n_simulations: int = 100_000
    n_years: int = 10
    n_chains: int = 4
    parallel: bool = True
    n_workers: Optional[int] = None
    chunk_size: int = 10_000
    use_float32: bool = True
    cache_results: bool = True
    checkpoint_interval: Optional[int] = None
    progress_bar: bool = True
    seed: Optional[int] = None
    use_enhanced_parallel: bool = True
    monitor_performance: bool = True
    adaptive_chunking: bool = True
    shared_memory: bool = True
    # Trajectory storage options
    enable_trajectory_storage: bool = False
    trajectory_storage_config: Optional[StorageConfig] = None
    # Aggregation options
    enable_advanced_aggregation: bool = True
    aggregation_config: Optional[AggregationConfig] = None
    generate_summary_report: bool = False
    summary_report_format: str = "markdown"
    # Bootstrap confidence interval options
    compute_bootstrap_ci: bool = False
    bootstrap_confidence_level: float = 0.95
    bootstrap_n_iterations: int = 10000
    bootstrap_method: str = "percentile"


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation.

    Attributes:
        final_assets: Final asset values for each simulation
        annual_losses: Annual loss amounts
        insurance_recoveries: Insurance recovery amounts
        retained_losses: Retained loss amounts
        growth_rates: Realized growth rates
        ruin_probability: Probability of ruin
        metrics: Risk metrics calculated from results
        convergence: Convergence statistics
        execution_time: Total execution time in seconds
        config: Simulation configuration used
        performance_metrics: Detailed performance metrics (if monitoring enabled)
        aggregated_results: Advanced aggregation results (if enabled)
        time_series_aggregation: Time series aggregation results (if enabled)
        statistical_summary: Complete statistical summary (if enabled)
        summary_report: Formatted summary report (if generated)
        bootstrap_confidence_intervals: Bootstrap confidence intervals for key metrics
    """

    final_assets: np.ndarray
    annual_losses: np.ndarray
    insurance_recoveries: np.ndarray
    retained_losses: np.ndarray
    growth_rates: np.ndarray
    ruin_probability: float
    metrics: Dict[str, float]
    convergence: Dict[str, ConvergenceStats]
    execution_time: float
    config: SimulationConfig
    performance_metrics: Optional[PerformanceMetrics] = None
    aggregated_results: Optional[Dict[str, Any]] = None
    time_series_aggregation: Optional[Dict[str, Any]] = None
    statistical_summary: Optional[Any] = None
    summary_report: Optional[str] = None
    bootstrap_confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

    def summary(self) -> str:
        """Generate summary of simulation results."""
        base_summary = (
            f"Simulation Results Summary\n"
            f"{'='*50}\n"
            f"Simulations: {self.config.n_simulations:,}\n"
            f"Years: {self.config.n_years}\n"
            f"Execution Time: {self.execution_time:.2f}s\n"
            f"Ruin Probability: {self.ruin_probability:.2%}\n"
            f"Mean Final Assets: ${np.mean(self.final_assets):,.0f}\n"
            f"Mean Growth Rate: {np.mean(self.growth_rates):.4f}\n"
            f"VaR(99%): ${self.metrics.get('var_99', 0):,.0f}\n"
            f"TVaR(99%): ${self.metrics.get('tvar_99', 0):,.0f}\n"
            f"Convergence R-hat: "
            f"{self.convergence.get('growth_rate', ConvergenceStats(0,0,0,False,0,0)).r_hat:.3f}\n"
        )

        # Add performance metrics if available
        if self.performance_metrics:
            base_summary += f"\n{'='*50}\n"
            base_summary += self.performance_metrics.summary()

        # Add advanced aggregation results if available
        if self.aggregated_results:
            base_summary += f"\n{'='*50}\nAdvanced Aggregation Results:\n"
            if "percentiles" in self.aggregated_results:
                for p, val in self.aggregated_results["percentiles"].items():
                    base_summary += f"  {p}: ${val:,.0f}\n"

        # Add bootstrap confidence intervals if available
        if self.bootstrap_confidence_intervals:
            base_summary += f"\n{'='*50}\nBootstrap Confidence Intervals (95%):\n"
            for metric_name, (lower, upper) in self.bootstrap_confidence_intervals.items():
                if (
                    "assets" in metric_name.lower()
                    or "var" in metric_name.lower()
                    or "tvar" in metric_name.lower()
                ):
                    # Format as currency for asset-related metrics
                    base_summary += f"  {metric_name}: [${lower:,.0f}, ${upper:,.0f}]\n"
                elif "probability" in metric_name.lower() or "rate" in metric_name.lower():
                    # Format as percentage for rates and probabilities
                    base_summary += f"  {metric_name}: [{lower:.2%}, {upper:.2%}]\n"
                else:
                    # Default formatting
                    base_summary += f"  {metric_name}: [{lower:.4f}, {upper:.4f}]\n"

        # Add summary report if available
        if self.summary_report:
            base_summary += f"\n{'='*50}\n{self.summary_report}"

        return base_summary


class MonteCarloEngine:
    """High-performance Monte Carlo simulation engine.

    Supports vectorized operations, parallel processing, and convergence monitoring.
    Now enhanced with CPU-optimized parallel execution for budget hardware.
    """

    def __init__(
        self,
        loss_generator: ManufacturingLossGenerator,
        insurance_program: InsuranceProgram,
        manufacturer: WidgetManufacturer,
        config: Optional[SimulationConfig] = None,
    ):
        """Initialize Monte Carlo engine.

        Args:
            loss_generator: Generator for loss events
            insurance_program: Insurance program structure
            manufacturer: Manufacturing company model
            config: Simulation configuration
        """
        self.loss_generator = loss_generator
        self.insurance_program = insurance_program
        self.manufacturer = manufacturer
        self.config = config or SimulationConfig()

        # Set up convergence diagnostics
        self.convergence_diagnostics = ConvergenceDiagnostics()

        # Cache directory for checkpoints
        self.cache_dir = Path("cache/monte_carlo")
        if self.config.cache_results:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Determine number of workers
        if self.config.n_workers is None and self.config.parallel:
            self.config.n_workers = min(os.cpu_count() or 4, 8)

        # Initialize enhanced parallel executor if enabled
        self.parallel_executor = None
        if self.config.use_enhanced_parallel and self.config.parallel:
            chunking_strategy = ChunkingStrategy(
                initial_chunk_size=self.config.chunk_size,
                adaptive=self.config.adaptive_chunking,
            )
            shared_memory_config = SharedMemoryConfig(
                enable_shared_arrays=self.config.shared_memory,
                enable_shared_objects=self.config.shared_memory,
            )
            self.parallel_executor = ParallelExecutor(
                n_workers=self.config.n_workers,
                chunking_strategy=chunking_strategy,
                shared_memory_config=shared_memory_config,
                monitor_performance=self.config.monitor_performance,
            )

        # Initialize trajectory storage if enabled
        self.trajectory_storage: Optional[TrajectoryStorage] = None
        if self.config.enable_trajectory_storage:
            storage_config = self.config.trajectory_storage_config or StorageConfig()
            self.trajectory_storage = TrajectoryStorage(storage_config)

        # Initialize aggregators if enabled
        self.result_aggregator: Optional[ResultAggregator] = None
        self.time_series_aggregator: Optional[TimeSeriesAggregator] = None
        self.summary_statistics: Optional[SummaryStatistics] = None

        if self.config.enable_advanced_aggregation:
            agg_config = self.config.aggregation_config or AggregationConfig()
            self.result_aggregator = ResultAggregator(agg_config)
            self.time_series_aggregator = TimeSeriesAggregator(agg_config)
            self.summary_statistics = SummaryStatistics()

    def run(self) -> SimulationResults:
        """Execute Monte Carlo simulation.

        Returns:
            SimulationResults object with all outputs
        """
        start_time = time.time()

        # Check for cached results
        cache_key = self._get_cache_key()
        if self.config.cache_results:
            cached = self._load_cache(cache_key)
            if cached is not None:
                print("Loaded cached results")
                return cached

        # Run simulation with appropriate executor
        if self.config.parallel:
            if self.config.use_enhanced_parallel and self.parallel_executor:
                results = self._run_enhanced_parallel()
            else:
                results = self._run_parallel()
        else:
            results = self._run_sequential()

        # Calculate metrics
        results.metrics = self._calculate_metrics(results)

        # Check convergence
        results.convergence = self._check_convergence(results)

        # Perform advanced aggregation if enabled
        if self.config.enable_advanced_aggregation:
            results = self._perform_advanced_aggregation(results)

        # Set execution time
        results.execution_time = time.time() - start_time

        # Add performance metrics if using enhanced parallel
        if self.config.use_enhanced_parallel and self.parallel_executor:
            results.performance_metrics = self.parallel_executor.performance_metrics

        # Compute bootstrap confidence intervals if requested
        if self.config.compute_bootstrap_ci:
            results.bootstrap_confidence_intervals = self.compute_bootstrap_confidence_intervals(
                results,
                confidence_level=self.config.bootstrap_confidence_level,
                n_bootstrap=self.config.bootstrap_n_iterations,
                method=self.config.bootstrap_method,
                show_progress=self.config.progress_bar,
            )

        # Cache results
        if self.config.cache_results:
            self._save_cache(cache_key, results)

        return results

    def _run_sequential(self) -> SimulationResults:
        """Run simulation sequentially."""
        n_sims = self.config.n_simulations
        n_years = self.config.n_years
        dtype = np.float32 if self.config.use_float32 else np.float64

        # Pre-allocate arrays
        final_assets = np.zeros(n_sims, dtype=dtype)
        annual_losses = np.zeros((n_sims, n_years), dtype=dtype)
        insurance_recoveries = np.zeros((n_sims, n_years), dtype=dtype)
        retained_losses = np.zeros((n_sims, n_years), dtype=dtype)

        # Progress bar
        iterator = range(n_sims)
        if self.config.progress_bar:
            iterator = tqdm(iterator, desc="Running simulations")

        # Run simulations
        for i in iterator:
            sim_results = self._run_single_simulation(i)
            final_assets[i] = sim_results["final_assets"]
            annual_losses[i] = sim_results["annual_losses"]
            insurance_recoveries[i] = sim_results["insurance_recoveries"]
            retained_losses[i] = sim_results["retained_losses"]

            # Checkpoint if needed
            if (
                self.config.checkpoint_interval
                and i > 0
                and i % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoint(
                    i,
                    final_assets[: i + 1],
                    annual_losses[: i + 1],
                    insurance_recoveries[: i + 1],
                    retained_losses[: i + 1],
                )

        # Calculate growth rates
        growth_rates = self._calculate_growth_rates(final_assets)

        # Calculate ruin probability
        ruin_mask = np.array(final_assets) <= 0
        ruin_probability = float(np.mean(ruin_mask))

        return SimulationResults(
            final_assets=final_assets,
            annual_losses=annual_losses,
            insurance_recoveries=insurance_recoveries,
            retained_losses=retained_losses,
            growth_rates=growth_rates,
            ruin_probability=ruin_probability,
            metrics={},
            convergence={},
            execution_time=0,
            config=self.config,
        )

    def _run_parallel(self) -> SimulationResults:
        """Run simulation in parallel using multiprocessing."""
        n_sims = self.config.n_simulations
        n_workers = self.config.n_workers
        chunk_size = self.config.chunk_size

        # Create chunks
        chunks = []
        for i in range(0, n_sims, chunk_size):
            chunk_end = min(i + chunk_size, n_sims)
            chunk_seed = None if self.config.seed is None else self.config.seed + i
            chunks.append((i, chunk_end, chunk_seed))

        # Run chunks in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self._run_chunk, chunk): chunk for chunk in chunks}

            # Process completed tasks
            if self.config.progress_bar:
                pbar = tqdm(total=len(chunks), desc="Processing chunks")

            for future in as_completed(futures):
                chunk_results = future.result()
                all_results.append(chunk_results)

                if self.config.progress_bar:
                    pbar.update(1)

            if self.config.progress_bar:
                pbar.close()

        # Combine results
        return self._combine_chunk_results(all_results)

    def _run_enhanced_parallel(self) -> SimulationResults:
        """Run simulation using enhanced parallel executor.

        Uses CPU-optimized parallel execution with shared memory and
        intelligent chunking for better performance on budget hardware.
        """
        n_sims = self.config.n_simulations

        # Ensure parallel executor is available
        assert self.parallel_executor is not None, "Enhanced parallel executor not initialized"

        # Prepare shared data (configuration that doesn't change)
        shared_data = {
            "n_years": self.config.n_years,
            "use_float32": self.config.use_float32,
            "manufacturer_config": self.manufacturer.__dict__.copy(),
            "insurance_layers": [layer.__dict__ for layer in self.insurance_program.layers],
            "loss_generator_params": {
                "frequency_params": getattr(self.loss_generator, "frequency_params", {}),
                "severity_params": getattr(self.loss_generator, "severity_params", {}),
            },
        }

        # Define reduce function
        def combine_results_enhanced(chunk_results):
            """Combine results from enhanced parallel execution."""
            # Flatten list of lists
            all_results = []
            for chunk in chunk_results:
                all_results.extend(chunk)

            # Extract arrays
            n_results = len(all_results)
            n_years = self.config.n_years
            dtype = np.float32 if self.config.use_float32 else np.float64

            final_assets = np.zeros(n_results, dtype=dtype)
            annual_losses = np.zeros((n_results, n_years), dtype=dtype)
            insurance_recoveries = np.zeros((n_results, n_years), dtype=dtype)
            retained_losses = np.zeros((n_results, n_years), dtype=dtype)

            for i, result in enumerate(all_results):
                final_assets[i] = result["final_assets"]
                annual_losses[i] = result["annual_losses"]
                insurance_recoveries[i] = result["insurance_recoveries"]
                retained_losses[i] = result["retained_losses"]

            # Calculate derived metrics
            growth_rates = self._calculate_growth_rates(final_assets)
            ruin_mask = np.array(final_assets) <= 0
            ruin_probability = float(np.mean(ruin_mask))

            return SimulationResults(
                final_assets=final_assets,
                annual_losses=annual_losses,
                insurance_recoveries=insurance_recoveries,
                retained_losses=retained_losses,
                growth_rates=growth_rates,
                ruin_probability=ruin_probability,
                metrics={},
                convergence={},
                execution_time=0,
                config=self.config,
                performance_metrics=None,
            )

        # Execute using enhanced parallel executor
        results = self.parallel_executor.map_reduce(
            work_function=_simulate_path_enhanced,
            work_items=range(n_sims),
            reduce_function=combine_results_enhanced,
            shared_data=shared_data,
            progress_bar=self.config.progress_bar,
        )

        # Add performance metrics from executor
        if self.config.monitor_performance and self.parallel_executor.performance_metrics:
            results.performance_metrics = self.parallel_executor.performance_metrics

        return results  # type: ignore[no-any-return]

    def _run_chunk(self, chunk: Tuple[int, int, Optional[int]]) -> Dict[str, np.ndarray]:
        """Run a chunk of simulations.

        Args:
            chunk: Tuple of (start_idx, end_idx, seed)

        Returns:
            Dictionary with simulation results for the chunk
        """
        start_idx, end_idx, seed = chunk
        n_sims = end_idx - start_idx
        n_years = self.config.n_years
        dtype = np.float32 if self.config.use_float32 else np.float64

        # Set seed for this chunk
        if seed is not None:
            np.random.seed(seed)

        # Pre-allocate arrays
        final_assets = np.zeros(n_sims, dtype=dtype)
        annual_losses = np.zeros((n_sims, n_years), dtype=dtype)
        insurance_recoveries = np.zeros((n_sims, n_years), dtype=dtype)
        retained_losses = np.zeros((n_sims, n_years), dtype=dtype)

        # Run simulations in chunk
        for i in range(n_sims):
            sim_results = self._run_single_simulation(start_idx + i)
            final_assets[i] = sim_results["final_assets"]
            annual_losses[i] = sim_results["annual_losses"]
            insurance_recoveries[i] = sim_results["insurance_recoveries"]
            retained_losses[i] = sim_results["retained_losses"]

        return {
            "final_assets": final_assets,
            "annual_losses": annual_losses,
            "insurance_recoveries": insurance_recoveries,
            "retained_losses": retained_losses,
        }

    def _run_single_simulation(self, sim_id: int) -> Dict[str, Any]:
        """Run a single simulation path.

        Args:
            sim_id: Simulation identifier

        Returns:
            Dictionary with simulation results
        """
        n_years = self.config.n_years
        dtype = np.float32 if self.config.use_float32 else np.float64

        # Create a copy of manufacturer for this simulation
        manufacturer = self.manufacturer.copy()

        # Arrays to store results
        annual_losses = np.zeros(n_years, dtype=dtype)
        insurance_recoveries = np.zeros(n_years, dtype=dtype)
        retained_losses = np.zeros(n_years, dtype=dtype)

        # Run simulation for each year
        for year in range(n_years):
            # Generate losses
            revenue = manufacturer.calculate_revenue()
            events, _ = self.loss_generator.generate_losses(duration=1.0, revenue=revenue)

            # Calculate total loss
            total_loss = sum(event.amount for event in events)
            annual_losses[year] = total_loss

            # Apply insurance
            claim_result = self.insurance_program.process_claim(total_loss)
            recovery = claim_result.get("total_recovery", 0)
            insurance_recoveries[year] = recovery

            # Calculate retained loss
            retained = total_loss - recovery
            retained_losses[year] = retained

            # Process insurance claims
            for event in events:
                manufacturer.process_insurance_claim(event.amount)

            # Update manufacturer state with annual step
            manufacturer.step(
                working_capital_pct=0.2,
                growth_rate=0.0,  # No growth rate for now
            )

            # Check for ruin
            if manufacturer.assets <= 0:
                ruin_occurred = True
                ruin_year = year
                break
        else:
            ruin_occurred = False
            ruin_year = None

        # Store trajectory if storage is enabled
        if self.trajectory_storage:
            self.trajectory_storage.store_simulation(
                sim_id=sim_id,
                annual_losses=annual_losses[: year + 1] if ruin_occurred else annual_losses,
                insurance_recoveries=insurance_recoveries[: year + 1]
                if ruin_occurred
                else insurance_recoveries,
                retained_losses=retained_losses[: year + 1] if ruin_occurred else retained_losses,
                final_assets=manufacturer.assets,
                initial_assets=self.manufacturer.assets,
                ruin_occurred=ruin_occurred,
                ruin_year=ruin_year,
            )

        return {
            "final_assets": manufacturer.assets,
            "annual_losses": annual_losses,
            "insurance_recoveries": insurance_recoveries,
            "retained_losses": retained_losses,
        }

    def _combine_chunk_results(
        self, chunk_results: List[Dict[str, np.ndarray]]
    ) -> SimulationResults:
        """Combine results from parallel chunks.

        Args:
            chunk_results: List of chunk result dictionaries

        Returns:
            Combined SimulationResults
        """
        # Concatenate arrays
        final_assets = np.concatenate([r["final_assets"] for r in chunk_results])
        annual_losses = np.vstack([r["annual_losses"] for r in chunk_results])
        insurance_recoveries = np.vstack([r["insurance_recoveries"] for r in chunk_results])
        retained_losses = np.vstack([r["retained_losses"] for r in chunk_results])

        # Calculate derived metrics
        growth_rates = self._calculate_growth_rates(final_assets)
        ruin_probability = np.mean(final_assets <= 0)

        return SimulationResults(
            final_assets=final_assets,
            annual_losses=annual_losses,
            insurance_recoveries=insurance_recoveries,
            retained_losses=retained_losses,
            growth_rates=growth_rates,
            ruin_probability=ruin_probability,
            metrics={},
            convergence={},
            execution_time=0,
            config=self.config,
        )

    def _calculate_growth_rates(self, final_assets: np.ndarray) -> np.ndarray:
        """Calculate annualized growth rates.

        Args:
            final_assets: Final asset values

        Returns:
            Array of growth rates
        """
        initial_assets = self.manufacturer.assets
        n_years = self.config.n_years

        # Avoid division by zero and log of negative numbers
        valid_mask = (final_assets > 0) & (initial_assets > 0)
        growth_rates = np.zeros_like(final_assets, dtype=np.float64)

        if np.any(valid_mask):
            growth_rates[valid_mask] = np.log(final_assets[valid_mask] / initial_assets) / n_years

        return growth_rates

    def _calculate_metrics(self, results: SimulationResults) -> Dict[str, float]:
        """Calculate risk metrics from simulation results.

        Args:
            results: Simulation results

        Returns:
            Dictionary of risk metrics
        """
        # Total losses across all years
        total_losses = np.sum(results.annual_losses, axis=1)

        # Initialize risk metrics calculator
        risk_metrics = RiskMetrics(total_losses)

        metrics = {
            "mean_loss": np.mean(total_losses),
            "median_loss": np.median(total_losses),
            "std_loss": np.std(total_losses),
            "var_95": risk_metrics.var(0.95),
            "var_99": risk_metrics.var(0.99),
            "var_995": risk_metrics.var(0.995),
            "tvar_95": risk_metrics.tvar(0.95),
            "tvar_99": risk_metrics.tvar(0.99),
            "tvar_995": risk_metrics.tvar(0.995),
            "expected_shortfall_99": risk_metrics.expected_shortfall(0.99),
            "max_loss": np.max(total_losses),
            "mean_recovery": np.mean(np.sum(results.insurance_recoveries, axis=1)),
            "mean_retained": np.mean(np.sum(results.retained_losses, axis=1)),
            "mean_growth_rate": np.mean(results.growth_rates),
            "std_growth_rate": np.std(results.growth_rates),
            "sharpe_ratio": np.mean(results.growth_rates) / np.std(results.growth_rates)
            if np.std(results.growth_rates) > 0
            else 0,
        }

        return metrics

    def _check_convergence(self, results: SimulationResults) -> Dict[str, ConvergenceStats]:
        """Check convergence of simulation results.

        Args:
            results: Simulation results

        Returns:
            Dictionary of convergence statistics
        """
        # Reshape data into chains
        n_chains = min(self.config.n_chains, results.final_assets.shape[0] // 100)

        if n_chains < 2:
            # Not enough data for multi-chain convergence
            return {}

        # Split data into chains
        chain_size = len(results.final_assets) // n_chains
        chains_growth = np.array(
            [results.growth_rates[i * chain_size : (i + 1) * chain_size] for i in range(n_chains)]
        )

        chains_losses = np.array(
            [
                np.sum(results.annual_losses[i * chain_size : (i + 1) * chain_size], axis=1)
                for i in range(n_chains)
            ]
        )

        # Stack chains for multiple metrics
        chains = np.stack([chains_growth, chains_losses], axis=2)

        # Check convergence
        convergence_stats = self.convergence_diagnostics.check_convergence(
            chains, metric_names=["growth_rate", "total_losses"]
        )

        return convergence_stats

    def _perform_advanced_aggregation(self, results: SimulationResults) -> SimulationResults:
        """Perform advanced aggregation on simulation results.

        Args:
            results: Initial simulation results

        Returns:
            Enhanced results with aggregation data
        """
        # Aggregate final assets
        if self.result_aggregator:
            results.aggregated_results = self.result_aggregator.aggregate(results.final_assets)

        # Time series aggregation
        if self.time_series_aggregator:
            # Aggregate annual losses
            results.time_series_aggregation = {
                "losses": self.time_series_aggregator.aggregate(
                    results.annual_losses.T  # Transpose to have time in rows
                ),
                "recoveries": self.time_series_aggregator.aggregate(results.insurance_recoveries.T),
                "retained": self.time_series_aggregator.aggregate(results.retained_losses.T),
            }

        # Statistical summary
        if self.summary_statistics:
            results.statistical_summary = self.summary_statistics.calculate_summary(
                results.final_assets
            )

            # Generate report if requested
            if self.config.generate_summary_report:
                report_generator = SummaryReportGenerator(style=self.config.summary_report_format)
                results.summary_report = report_generator.generate_report(
                    results.statistical_summary,
                    title="Monte Carlo Simulation Summary",
                    metadata={
                        "Simulations": self.config.n_simulations,
                        "Years": self.config.n_years,
                        "Ruin Probability": f"{results.ruin_probability:.2%}",
                        "Mean Growth Rate": f"{np.mean(results.growth_rates):.4f}",
                    },
                )

        return results

    def export_results(
        self, results: SimulationResults, filepath: Path, format: str = "csv"
    ) -> None:
        """Export simulation results to file.

        Args:
            results: Simulation results to export
            filepath: Output file path
            format: Export format ('csv', 'json', 'hdf5')
        """
        if not results.aggregated_results:
            # Perform aggregation if not already done
            if self.result_aggregator:
                results.aggregated_results = self.result_aggregator.aggregate(results.final_assets)
            else:
                # Use default aggregation
                agg_config = AggregationConfig()
                aggregator = ResultAggregator(agg_config)
                results.aggregated_results = aggregator.aggregate(results.final_assets)

        # Export based on format
        if format.lower() == "csv":
            ResultExporter.to_csv(results.aggregated_results, filepath)
        elif format.lower() == "json":
            ResultExporter.to_json(results.aggregated_results, filepath)
        elif format.lower() == "hdf5":
            ResultExporter.to_hdf5(results.aggregated_results, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def compute_bootstrap_confidence_intervals(
        self,
        results: SimulationResults,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = "percentile",
        show_progress: bool = False,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for key simulation metrics.

        Args:
            results: Simulation results to analyze.
            confidence_level: Confidence level for intervals (default 0.95).
            n_bootstrap: Number of bootstrap iterations (default 10000).
            method: Bootstrap method ('percentile' or 'bca').
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary mapping metric names to (lower, upper) confidence bounds.
        """
        analyzer = BootstrapAnalyzer(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=self.config.seed,
            show_progress=show_progress,
        )

        confidence_intervals = {}

        # Bootstrap CI for mean final assets
        _, ci = bootstrap_confidence_interval(
            results.final_assets,
            np.mean,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
            seed=self.config.seed,
        )
        confidence_intervals["Mean Final Assets"] = ci

        # Bootstrap CI for median final assets
        _, ci = bootstrap_confidence_interval(
            results.final_assets,
            np.median,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
            seed=self.config.seed,
        )
        confidence_intervals["Median Final Assets"] = ci

        # Bootstrap CI for mean growth rate
        _, ci = bootstrap_confidence_interval(
            results.growth_rates,
            np.mean,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
            seed=self.config.seed,
        )
        confidence_intervals["Mean Growth Rate"] = ci

        # Bootstrap CI for ruin probability
        # Create binary ruin indicator
        ruin_indicator = (results.final_assets <= 0).astype(float)
        _, ci = bootstrap_confidence_interval(
            ruin_indicator,
            np.mean,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
            seed=self.config.seed,
        )
        confidence_intervals["Ruin Probability"] = ci

        # Bootstrap CI for VaR if available
        if "var_99" in results.metrics:

            def var_99(x):
                return np.percentile(x, 99)

            _, ci = bootstrap_confidence_interval(
                results.final_assets,
                var_99,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                method=method,
                seed=self.config.seed,
            )
            confidence_intervals["VaR(99%)"] = ci

        # Bootstrap CI for mean annual losses
        mean_annual_losses = np.mean(
            results.annual_losses, axis=1
        )  # Mean across years for each simulation
        _, ci = bootstrap_confidence_interval(
            mean_annual_losses,
            np.mean,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
            seed=self.config.seed,
        )
        confidence_intervals["Mean Annual Losses"] = ci

        # Bootstrap CI for mean insurance recoveries
        mean_recoveries = np.mean(results.insurance_recoveries, axis=1)
        _, ci = bootstrap_confidence_interval(
            mean_recoveries,
            np.mean,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            method=method,
            seed=self.config.seed,
        )
        confidence_intervals["Mean Insurance Recoveries"] = ci

        return confidence_intervals

    def _get_cache_key(self) -> str:
        """Generate cache key for current configuration.

        Returns:
            Cache key string
        """
        key_parts = [
            f"n_sims_{self.config.n_simulations}",
            f"n_years_{self.config.n_years}",
            f"seed_{self.config.seed}",
            f"ins_{hash(str(self.insurance_program))}",
            f"mfg_{hash(str(self.manufacturer))}",
        ]
        return "_".join(key_parts)

    def _save_cache(self, cache_key: str, results: SimulationResults) -> None:
        """Save results to cache.

        Args:
            cache_key: Cache key
            results: Results to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
        except (IOError, OSError, pickle.PickleError) as e:
            warnings.warn(f"Failed to save cache: {e}")

    def _load_cache(self, cache_key: str) -> Optional[SimulationResults]:
        """Load results from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached results or None
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    loaded_data = pickle.load(f)
                    return loaded_data  # type: ignore
            except (IOError, OSError, pickle.PickleError, EOFError) as e:
                warnings.warn(f"Failed to load cache: {e}")
        return None

    def _save_checkpoint(self, iteration: int, *arrays) -> None:
        """Save checkpoint during simulation.

        Args:
            iteration: Current iteration number
            arrays: Arrays to save
        """
        checkpoint_file = self.cache_dir / f"checkpoint_{iteration}.npz"
        try:
            np.savez_compressed(checkpoint_file, iteration=iteration, *arrays)
        except (IOError, OSError, ValueError) as e:
            warnings.warn(f"Failed to save checkpoint: {e}")

    def _initialize_simulation_arrays(self) -> Dict[str, np.ndarray]:
        """Initialize arrays for simulation results."""
        n_sims = self.config.n_simulations
        n_years = self.config.n_years
        dtype = np.float32 if self.config.use_float32 else np.float64

        return {
            "final_assets": np.zeros(n_sims, dtype=dtype),
            "annual_losses": np.zeros((n_sims, n_years), dtype=dtype),
            "insurance_recoveries": np.zeros((n_sims, n_years), dtype=dtype),
            "retained_losses": np.zeros((n_sims, n_years), dtype=dtype),
        }

    def _check_convergence_at_interval(
        self, completed_iterations: int, final_assets: np.ndarray
    ) -> float:
        """Check convergence at specified interval."""
        partial_growth = self._calculate_growth_rates(final_assets[:completed_iterations])

        # Split into chains for convergence check
        n_chains = min(4, completed_iterations // 250)
        if n_chains >= 2:
            chain_size = completed_iterations // n_chains
            chains = np.array(
                [partial_growth[j * chain_size : (j + 1) * chain_size] for j in range(n_chains)]
            )
            r_hat = self.convergence_diagnostics.calculate_r_hat(chains)
        else:
            r_hat = float("inf")

        return r_hat

    def _run_simulation_batch(
        self, batch_range: range, arrays: Dict[str, np.ndarray], batch_config: Dict[str, Any]
    ) -> int:
        """Run a batch of simulations with monitoring."""
        monitor = batch_config["monitor"]
        check_intervals = batch_config["check_intervals"]
        early_stopping = batch_config["early_stopping"]
        show_progress = batch_config["show_progress"]
        completed_iterations = 0

        for i in batch_range:
            sim_results = self._run_single_simulation(i)
            arrays["final_assets"][i] = sim_results["final_assets"]
            arrays["annual_losses"][i] = sim_results["annual_losses"]
            arrays["insurance_recoveries"][i] = sim_results["insurance_recoveries"]
            arrays["retained_losses"][i] = sim_results["retained_losses"]
            completed_iterations = i + 1

            # Check if we should perform convergence check
            if completed_iterations in check_intervals and completed_iterations >= 1000:
                r_hat = self._check_convergence_at_interval(
                    completed_iterations, arrays["final_assets"]
                )
                should_continue = monitor.update(completed_iterations, r_hat)

                if early_stopping and not should_continue:
                    if show_progress:
                        print(
                            f"\n✓ Early stopping: Convergence achieved at {completed_iterations:,} iterations"
                        )
                    break
            else:
                monitor.update(completed_iterations)

        return completed_iterations

    def run_with_progress_monitoring(
        self,
        check_intervals: Optional[List[int]] = None,
        convergence_threshold: float = 1.1,
        early_stopping: bool = True,
        show_progress: bool = True,
    ) -> SimulationResults:
        """Run simulation with progress tracking and convergence monitoring."""
        if check_intervals is None:
            check_intervals = [10_000, 25_000, 50_000, 100_000]
        check_intervals = [i for i in check_intervals if i <= self.config.n_simulations]

        # Initialize monitor and arrays
        monitor = ProgressMonitor(
            total_iterations=self.config.n_simulations,
            check_intervals=check_intervals,
            update_frequency=max(self.config.n_simulations // 100, 100),
            show_console=show_progress,
            convergence_threshold=convergence_threshold,
        )

        start_time = time.time()
        arrays = self._initialize_simulation_arrays()

        # Run simulations in batches
        batch_size = min(1000, self.config.n_simulations // 10)
        completed_iterations = 0

        for batch_start in range(0, self.config.n_simulations, batch_size):
            batch_end = min(batch_start + batch_size, self.config.n_simulations)

            completed_iterations = self._run_simulation_batch(
                range(batch_start, batch_end),
                arrays,
                {
                    "monitor": monitor,
                    "check_intervals": check_intervals,
                    "early_stopping": early_stopping,
                    "show_progress": show_progress,
                },
            )

            if early_stopping and monitor.converged:
                break

        monitor.finalize()

        # Trim arrays if stopped early
        if completed_iterations < self.config.n_simulations:
            for key in arrays:
                if key == "final_assets":
                    arrays[key] = arrays[key][:completed_iterations]
                else:
                    arrays[key] = arrays[key][:completed_iterations]

        # Create results
        growth_rates = self._calculate_growth_rates(arrays["final_assets"])
        results = SimulationResults(
            final_assets=arrays["final_assets"],
            annual_losses=arrays["annual_losses"],
            insurance_recoveries=arrays["insurance_recoveries"],
            retained_losses=arrays["retained_losses"],
            growth_rates=growth_rates,
            ruin_probability=float(np.mean(arrays["final_assets"] <= 0)),
            metrics={},
            convergence={},
            execution_time=time.time() - start_time,
            config=self.config,
        )

        results.metrics = self._calculate_metrics(results)
        results.convergence = self._check_convergence(results)

        # Add progress summary to metrics
        monitor.get_stats()
        convergence_summary = monitor.generate_convergence_summary()

        results.metrics["actual_iterations"] = completed_iterations
        results.metrics["convergence_achieved"] = monitor.converged
        results.metrics["convergence_iteration"] = (
            monitor.converged_at if monitor.converged_at else 0
        )
        results.metrics["monitoring_overhead_pct"] = convergence_summary.get(
            "performance_overhead_pct", 0
        )

        # Store ESS information
        if results.convergence:
            for metric_name, conv_stats in results.convergence.items():
                results.metrics[f"ess_{metric_name}"] = conv_stats.ess

        return results

    def run_with_convergence_monitoring(
        self,
        target_r_hat: float = 1.05,
        check_interval: int = 10000,
        max_iterations: Optional[int] = None,
    ) -> SimulationResults:
        """Run simulation with automatic convergence monitoring.

        Args:
            target_r_hat: Target R-hat for convergence
            check_interval: Check convergence every N simulations
            max_iterations: Maximum iterations (None for no limit)

        Returns:
            Converged simulation results
        """
        if max_iterations is None:
            max_iterations = self.config.n_simulations * 10

        # Start with smaller batch
        original_n_sims = self.config.n_simulations
        self.config.n_simulations = check_interval

        all_results = []
        total_iterations = 0
        converged = False

        while not converged and total_iterations < max_iterations:
            # Run batch
            batch_results = self.run()
            all_results.append(batch_results)
            total_iterations += check_interval

            # Combine all results so far
            if len(all_results) > 1:
                combined = self._combine_multiple_results(all_results)
            else:
                combined = batch_results

            # Check convergence
            convergence = self._check_convergence(combined)

            if convergence:
                max_r_hat = max(stat.r_hat for stat in convergence.values())
                converged = max_r_hat < target_r_hat

                if self.config.progress_bar:
                    print(f"Iteration {total_iterations}: R-hat = {max_r_hat:.3f}")

            # Update seed for next batch
            if self.config.seed is not None:
                self.config.seed += check_interval

        # Restore original config
        self.config.n_simulations = original_n_sims

        return combined

    def _combine_multiple_results(self, results_list: List[SimulationResults]) -> SimulationResults:
        """Combine multiple simulation results.

        Args:
            results_list: List of SimulationResults to combine

        Returns:
            Combined SimulationResults
        """
        # Concatenate arrays
        final_assets = np.concatenate([r.final_assets for r in results_list])
        annual_losses = np.vstack([r.annual_losses for r in results_list])
        insurance_recoveries = np.vstack([r.insurance_recoveries for r in results_list])
        retained_losses = np.vstack([r.retained_losses for r in results_list])
        growth_rates = np.concatenate([r.growth_rates for r in results_list])

        # Recalculate metrics
        combined = SimulationResults(
            final_assets=final_assets,
            annual_losses=annual_losses,
            insurance_recoveries=insurance_recoveries,
            retained_losses=retained_losses,
            growth_rates=growth_rates,
            ruin_probability=np.mean(final_assets <= 0),
            metrics={},
            convergence={},
            execution_time=sum(r.execution_time for r in results_list),
            config=self.config,
        )

        combined.metrics = self._calculate_metrics(combined)
        combined.convergence = self._check_convergence(combined)

        return combined

    def estimate_ruin_probability(
        self,
        config: Optional[RuinProbabilityConfig] = None,
    ) -> RuinProbabilityResults:
        """Estimate ruin probability over multiple time horizons.

        Delegates to RuinProbabilityAnalyzer for specialized analysis.

        Args:
            config: Configuration for ruin probability estimation

        Returns:
            RuinProbabilityResults with comprehensive bankruptcy analysis
        """
        analyzer = RuinProbabilityAnalyzer(
            manufacturer=self.manufacturer,
            loss_generator=self.loss_generator,
            insurance_program=self.insurance_program,
            config=self.config,
        )
        return analyzer.analyze_ruin_probability(config)

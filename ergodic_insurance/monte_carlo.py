"""High-performance Monte Carlo simulation engine for insurance optimization."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import logging
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm

from .convergence import ConvergenceDiagnostics, ConvergenceStats
from .insurance_program import InsuranceProgram
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .monte_carlo_worker import run_chunk_standalone
from .parallel_executor import (
    ChunkingStrategy,
    ParallelExecutor,
    PerformanceMetrics,
    SharedMemoryConfig,
)
from .progress_monitor import ProgressMonitor
from .result_aggregator import (
    AggregationConfig,
    ResultAggregator,
    ResultExporter,
    TimeSeriesAggregator,
)
from .risk_metrics import RiskMetrics
from .ruin_probability import RuinProbabilityAnalyzer, RuinProbabilityConfig, RuinProbabilityResults
from .safe_pickle import safe_dump, safe_load
from .summary_statistics import SummaryReportGenerator, SummaryStatistics
from .trajectory_storage import StorageConfig, TrajectoryStorage


def _create_manufacturer(config_dict: Dict[str, Any]) -> "WidgetManufacturer":
    """Create a fresh manufacturer instance from a config dictionary.

    Uses :meth:`WidgetManufacturer.create_fresh` so that construction always
    goes through ``__init__`` validation.  The previous implementation used
    ``__new__`` with an unchecked ``setattr`` loop, which allowed arbitrary
    attribute injection (see issue #886).

    Args:
        config_dict: Dictionary that **must** contain a ``"config"`` key
            holding a :class:`ManufacturerConfig` instance.  An optional
            ``"stochastic_process"`` key is forwarded to the factory method.

    Returns:
        A validated :class:`WidgetManufacturer` in its initial state.

    Raises:
        TypeError: If ``config_dict["config"]`` is missing or is not a
            proper configuration object.
    """
    config = config_dict.get("config")
    if config is None or not hasattr(config, "__dict__"):
        raise TypeError(
            "_create_manufacturer requires config_dict['config'] to be a "
            "ManufacturerConfig instance, got "
            f"{type(config).__name__ if config is not None else 'None'}"
        )
    stochastic_process = config_dict.get("stochastic_process")
    return WidgetManufacturer.create_fresh(config, stochastic_process)


def _simulate_year_losses(sim_id: int, year: int) -> Tuple[float, float, float]:
    """Simulate losses for a single year.

    .. deprecated::
        This stub is unused. Enhanced parallel paths now use the configured
        loss generator passed via shared data. Retained only for backward
        compatibility with any external callers.
    """
    rng = np.random.default_rng(sim_id * 1000 + year)
    n_events = rng.poisson(3)

    if n_events == 0:
        total_loss = 0.0
    else:
        event_amounts = rng.lognormal(10, 2, n_events)
        total_loss = float(np.sum(event_amounts))

    recovery = min(total_loss, 1_000_000) * 0.9
    retained = total_loss - recovery

    return total_loss, recovery, retained


def _test_worker_function() -> bool:
    """Test function to check if multiprocessing works with scipy imports.

    Returns:
        True if the worker can execute successfully
    """
    try:
        # Try importing scipy to check if it causes issues
        from scipy import stats  # noqa: F401  # pylint: disable=unused-import

        # Test numpy is available (already imported at module level)
        _ = np.array([1, 2, 3])

        return True
    except ImportError:
        return False


def _simulate_path_enhanced(sim_id: int, **shared) -> Dict[str, Any]:
    """Enhanced simulation function for parallel execution.

    Module-level function for pickle compatibility in multiprocessing.
    Uses the configured loss generator and insurance program passed via
    shared data instead of a hardcoded stub (see issue #299).

    Args:
        sim_id: Simulation ID for seeding
        **shared: Shared data from parallel executor

    Returns:
        Dict with simulation results
    """
    import sys

    module_path = Path(__file__).parent.parent.parent
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

    # Create manufacturer instance
    manufacturer = _create_manufacturer(shared["manufacturer_config"])

    # Deep-copy the insurance program so each simulation gets fresh state (Issue #348)
    import copy

    loss_generator = shared["loss_generator"]
    insurance_program = copy.deepcopy(shared["insurance_program"])

    # Re-seed the loss generator for this simulation to ensure independence
    base_seed = shared.get("base_seed")
    if base_seed is not None:
        loss_generator.reseed(base_seed + sim_id)

    # Initialize simulation arrays
    n_years = shared["n_years"]
    dtype = np.float32 if shared["use_float32"] else np.float64

    result_arrays = {
        "annual_losses": np.zeros(n_years, dtype=dtype),
        "insurance_recoveries": np.zeros(n_years, dtype=dtype),
        "retained_losses": np.zeros(n_years, dtype=dtype),
    }

    # Track ruin at evaluation points if requested
    ruin_evaluation = shared.get("ruin_evaluation", None)
    ruin_at_year = {}
    if ruin_evaluation:
        for eval_year in ruin_evaluation:
            if eval_year <= n_years:
                ruin_at_year[eval_year] = False

    # CRN: look up once before the loop
    crn_base_seed = shared.get("crn_base_seed")

    # Simulate years using the configured loss generator
    for year in range(n_years):
        # Reset insurance program aggregate limits at start of each policy year (Issue #348)
        if year > 0:
            insurance_program.reset_annual()

        # CRN: reseed per (sim_id, year) for reproducible cross-scenario comparison
        if crn_base_seed is not None:
            year_ss = np.random.SeedSequence([crn_base_seed, sim_id, year])
            children = year_ss.spawn(2)
            loss_generator.reseed(int(children[0].generate_state(1)[0]))
            if manufacturer.stochastic_process is not None:
                manufacturer.stochastic_process.reset(int(children[1].generate_state(1)[0]))

        revenue = manufacturer.calculate_revenue()

        # Generate losses using configured loss generator
        if hasattr(loss_generator, "generate_losses"):
            year_losses, _ = loss_generator.generate_losses(duration=1.0, revenue=float(revenue))
        else:
            raise AttributeError(
                f"Loss generator {type(loss_generator).__name__} has no generate_losses method"
            )

        total_loss = sum(loss.amount for loss in year_losses)
        result_arrays["annual_losses"][year] = total_loss

        # Apply insurance PER OCCURRENCE (not aggregate) and create
        # ClaimLiability objects with LoC collateral (Issue #342).
        # Previously this path applied insurance to the aggregate total_loss,
        # which mis-applied per-occurrence deductibles.
        total_recovery = 0.0
        total_retained = 0.0

        for loss_event in year_losses:
            if loss_event.amount > 0:
                claim_result = insurance_program.process_claim(loss_event.amount)
                event_recovery = claim_result.insurance_recovery
                event_retained = loss_event.amount - event_recovery

                total_recovery += event_recovery
                total_retained += event_retained

                # Create ClaimLiability and post collateral for the retained
                # portion. See Issue #342 for double-counting rationale.
                if event_retained > 0:
                    manufacturer.process_insurance_claim(
                        claim_amount=loss_event.amount,
                        insurance_recovery=event_recovery,
                    )

        result_arrays["insurance_recoveries"][year] = total_recovery
        result_arrays["retained_losses"][year] = total_retained

        # Record insurance premium scaled by revenue (Issue #349)
        current_revenue = manufacturer.calculate_revenue()
        base_revenue = float(
            manufacturer.config.initial_assets * manufacturer.config.asset_turnover_ratio
        )
        revenue_scaling_factor = float(current_revenue) / base_revenue if base_revenue > 0 else 1.0
        base_annual_premium = insurance_program.calculate_premium()
        annual_premium = base_annual_premium * revenue_scaling_factor
        if annual_premium > 0:
            manufacturer.record_insurance_premium(annual_premium)

        # Step with config parameters (Issue #349)
        manufacturer.step(
            letter_of_credit_rate=shared.get("letter_of_credit_rate", 0.015),
            growth_rate=shared.get("growth_rate", 0.0),
            time_resolution=shared.get("time_resolution", "annual"),
            apply_stochastic=shared.get("apply_stochastic", False),
        )

        # Prune old ledger entries to bound memory (Issue #315)
        if shared.get("enable_ledger_pruning", False) and year > 0:
            manufacturer.ledger.prune_entries(before_date=year)

        # Check for ruin - use insolvency tolerance from shared config
        tolerance = shared.get("insolvency_tolerance", 10_000)
        if float(manufacturer.equity) <= tolerance or manufacturer.is_ruined:
            # Mark ruin for all future evaluation points
            if ruin_evaluation:
                for eval_year in ruin_at_year:
                    if year < eval_year:
                        ruin_at_year[eval_year] = True
            break

    result = {
        "final_assets": float(manufacturer.total_assets),
        "final_equity": float(manufacturer.equity),
        **result_arrays,
    }
    if ruin_evaluation:
        result["ruin_at_year"] = ruin_at_year
    return result


@dataclass
class MonteCarloConfig:
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
        letter_of_credit_rate: Annual LoC rate for collateral costs (default 1.5%)
        growth_rate: Revenue growth rate per period (default 0.0)
        time_resolution: Time step resolution, "annual" or "monthly" (default "annual")
        apply_stochastic: Whether to apply stochastic shocks (default False)
        enable_ledger_pruning: Prune old ledger entries each year to bound memory (default False)
        crn_base_seed: Common Random Numbers base seed for cross-scenario comparison.
            When set, the loss generator and stochastic process are reseeded at each
            (sim_id, year) boundary using SeedSequence([crn_base_seed, sim_id, year]).
            This ensures that compared scenarios (e.g. different deductibles) experience
            the same underlying random draws each year, dramatically reducing estimator
            variance for growth-lift metrics. (default None, disabled)
    """

    n_simulations: int = 100_000
    n_years: int = 10
    n_chains: int = 4
    parallel: bool = True
    n_workers: Optional[int] = None
    chunk_size: int = 10_000
    use_float32: bool = False
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
    # Periodic ruin evaluation options
    ruin_evaluation: Optional[List[int]] = None
    # Insolvency tolerance
    # NOTE: insolvency_tolerance is also referenced in _simulate_path_enhanced()
    # via shared["insolvency_tolerance"]. Keep defaults in sync.
    insolvency_tolerance: float = 10_000  # Company is insolvent when equity <= this threshold
    # Simulation step parameters (passed to manufacturer.step())
    # NOTE: letter_of_credit_rate and growth_rate defaults are coupled with
    # _simulate_path_enhanced() shared-data defaults and run_chunk_standalone().
    # If you change these defaults, update those functions accordingly.
    letter_of_credit_rate: float = 0.015  # Annual LoC rate for collateral costs
    growth_rate: float = 0.0  # Revenue growth rate per period
    time_resolution: str = "annual"  # "annual" or "monthly"
    apply_stochastic: bool = False  # Whether to apply stochastic shocks
    enable_ledger_pruning: bool = True  # Prune old ledger entries to bound memory (Issue #315)
    crn_base_seed: Optional[int] = None  # Common Random Numbers seed for cross-scenario comparison
    use_gpu: bool = False  # Use GPU-accelerated vectorized simulation (Issue #961)

    def __post_init__(self):
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        if self.n_simulations <= 0:
            raise ValueError(
                f"n_simulations must be positive, got {self.n_simulations}. "
                "Use at least 1000 for meaningful results."
            )
        if self.n_years <= 0:
            raise ValueError(
                f"n_years must be positive, got {self.n_years}. "
                "Set to the number of years to simulate (e.g. 10)."
            )
        if self.n_chains < 1:
            raise ValueError(
                f"n_chains must be >= 1, got {self.n_chains}. "
                "Use at least 2 chains for convergence diagnostics."
            )
        if self.chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {self.chunk_size}. "
                "Typical values are 1000-10000."
            )
        if not (0 < self.bootstrap_confidence_level < 1):
            raise ValueError(
                f"bootstrap_confidence_level must be between 0 and 1 (exclusive), "
                f"got {self.bootstrap_confidence_level}. Use e.g. 0.95 for a 95% CI."
            )


@dataclass
class MonteCarloResults:
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
    ruin_probability: Dict[str, float]
    metrics: Dict[str, float]
    convergence: Dict[str, ConvergenceStats]
    execution_time: float
    config: MonteCarloConfig
    performance_metrics: Optional[PerformanceMetrics] = None
    aggregated_results: Optional[Dict[str, Any]] = None
    time_series_aggregation: Optional[Dict[str, Any]] = None
    statistical_summary: Optional[Any] = None
    summary_report: Optional[str] = None
    bootstrap_confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

    def summary(self) -> str:
        """Generate summary of simulation results."""
        # Format ruin probability section
        # ruin_probability is always a dict now
        # Sort by year for display
        ruin_prob_lines = []
        for year_str in sorted(self.ruin_probability.keys(), key=int):
            prob = self.ruin_probability[year_str]
            ruin_prob_lines.append(f"  Year {year_str}: {prob:.2%}")
        ruin_prob_section = "Ruin Probability:\n" + "\n".join(ruin_prob_lines)

        base_summary = (
            f"Simulation Results Summary\n"
            f"{'='*50}\n"
            f"Simulations: {self.config.n_simulations:,}\n"
            f"Years: {self.config.n_years}\n"
            f"Execution Time: {self.execution_time:.2f}s\n"
            f"{ruin_prob_section}\n"
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
    """High-performance Monte Carlo simulation engine for insurance analysis.

    Provides efficient Monte Carlo simulation with support for parallel processing,
    convergence monitoring, checkpointing, and comprehensive result aggregation.
    Optimized for both high-end and budget hardware configurations.

    Examples:
        Basic Monte Carlo simulation::

            from .monte_carlo import MonteCarloEngine, MonteCarloConfig
            from .loss_distributions import ManufacturingLossGenerator
            from .insurance_program import InsuranceProgram
            from .manufacturer import WidgetManufacturer

            # Configure simulation
            config = MonteCarloConfig(
                n_simulations=10000,
                n_years=20,
                parallel=True,
                n_workers=4
            )

            # Create components
            loss_gen = ManufacturingLossGenerator()
            insurance = InsuranceProgram.create_standard_program()
            manufacturer = WidgetManufacturer.from_config()

            # Run Monte Carlo
            engine = MonteCarloEngine(
                loss_generator=loss_gen,
                insurance_program=insurance,
                manufacturer=manufacturer,
                config=config
            )
            results = engine.run()

            print(f"Survival rate: {results.survival_rate:.1%}")
            print(f"Mean ROE: {results.mean_roe:.2%}")

        Advanced simulation with convergence monitoring::

            # Enable convergence checking
            config = MonteCarloConfig(
                n_simulations=100000,
                check_convergence=True,
                convergence_tolerance=0.001,
                min_iterations=1000
            )

            engine = MonteCarloEngine(
                loss_generator=loss_gen,
                insurance_program=insurance,
                manufacturer=manufacturer,
                config=config
            )

            # Run with progress tracking
            results = engine.run(show_progress=True)

            # Check convergence
            if results.converged:
                print(f"Converged after {results.iterations} iterations")
                print(f"Standard error: {results.standard_error:.4f}")

    Attributes:
        loss_generator: Generator for manufacturing loss events
        insurance_program: Insurance coverage structure
        manufacturer: Manufacturing company financial model
        config: Simulation configuration parameters
        convergence_diagnostics: Convergence monitoring tools

    See Also:
        :class:`MonteCarloConfig`: Configuration parameters
        :class:`MonteCarloResults`: Simulation results container
        :class:`ParallelExecutor`: Enhanced parallel processing
        :class:`ConvergenceDiagnostics`: Convergence analysis tools
    """

    def __init__(
        self,
        loss_generator: ManufacturingLossGenerator,
        insurance_program: InsuranceProgram,
        manufacturer: WidgetManufacturer,
        config: Optional[MonteCarloConfig] = None,
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
        self.config = config or MonteCarloConfig()

        # Set up convergence diagnostics
        self.convergence_diagnostics = ConvergenceDiagnostics()

        # Cache directory for checkpoints
        self.cache_dir = Path("cache/monte_carlo")
        if self.config.cache_results:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: Global np.random.seed() was removed here (issue #299).
        # Setting the global seed leaks side effects to other code and does not
        # affect the loss generator's per-instance RandomState objects.
        # Per-chunk / per-simulation seeding is handled in the worker functions.

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

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MonteCarloResults:
        """Execute Monte Carlo simulation.

        Args:
            progress_callback: Optional callback invoked with
                ``(completed, total, elapsed_seconds)`` after each batch of
                simulations completes.  Useful for GUI progress bars, web
                dashboards, or any non-terminal environment.
            cancel_event: Optional :class:`threading.Event`.  When set, the
                engine will stop after the current batch and return partial
                results.

        Returns:
            MonteCarloResults object with all outputs
        """
        start_time = time.time()

        # Check for cached results
        cache_key = self._get_cache_key()
        if self.config.cache_results:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info("Loaded cached results")
                return cached

        # Reinitialize parallel executor if config has changed
        if self.config.use_enhanced_parallel and self.config.parallel:
            if self.parallel_executor is None or (
                self.parallel_executor and self.parallel_executor.n_workers != self.config.n_workers
            ):
                # Need to reinitialize with new worker count
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

        # Run simulation with appropriate executor
        if self.config.use_gpu:
            results = self._run_gpu(
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        elif self.config.parallel:
            if self.config.use_enhanced_parallel and self.parallel_executor:
                results = self._run_enhanced_parallel(
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            else:
                results = self._run_parallel(
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
        else:
            results = self._run_sequential(
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        # Calculate metrics
        results.metrics = self._calculate_metrics(results)

        # Check convergence
        results.convergence = self._check_convergence(results)

        # Perform advanced aggregation if enabled
        if self.config.enable_advanced_aggregation:
            results = self._perform_advanced_aggregation(results)

        # Set execution time
        results.execution_time = time.time() - start_time

        # Add performance metrics if monitoring is enabled
        if self.config.monitor_performance:
            if self.config.use_enhanced_parallel and self.parallel_executor:
                results.performance_metrics = self.parallel_executor.performance_metrics
            else:
                # Create basic performance metrics for non-enhanced execution
                execution_time = time.time() - start_time
                # Estimate memory usage based on simulation size (basic approximation)
                memory_estimate = (
                    self.config.n_simulations * self.config.n_years * 8 * 4
                )  # 4 arrays of 8-byte floats
                results.performance_metrics = PerformanceMetrics(
                    total_time=execution_time,
                    setup_time=0.0,
                    computation_time=execution_time,
                    serialization_time=0.0,
                    reduction_time=0.0,
                    memory_peak=memory_estimate,
                    cpu_utilization=0.0,
                    items_per_second=(
                        self.config.n_simulations / execution_time if execution_time > 0 else 0.0
                    ),
                    speedup=1.0,
                )

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

    def _run_sequential(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MonteCarloResults:
        """Run simulation sequentially."""
        n_sims = self.config.n_simulations
        n_years = self.config.n_years
        dtype = np.float32 if self.config.use_float32 else np.float64

        # Pre-allocate arrays
        final_assets = np.zeros(n_sims, dtype=dtype)
        final_equity = np.zeros(n_sims, dtype=dtype)
        annual_losses = np.zeros((n_sims, n_years), dtype=dtype)
        insurance_recoveries = np.zeros((n_sims, n_years), dtype=dtype)
        retained_losses = np.zeros((n_sims, n_years), dtype=dtype)

        # Track periodic ruin if requested
        ruin_at_year_all = []

        # Progress bar
        iterator = range(n_sims)
        if self.config.progress_bar:
            iterator = tqdm(iterator, desc="Running simulations")

        # How often to fire the progress callback / check cancellation
        callback_interval = max(1, n_sims // 100)
        seq_start = time.time()

        # Run simulations
        completed = 0
        for i in iterator:
            # Check for cancellation
            if cancel_event is not None and i % callback_interval == 0 and cancel_event.is_set():
                logger.info("Cancellation requested at simulation %d/%d", i, n_sims)
                break

            sim_results = self._run_single_simulation(i)
            final_assets[i] = sim_results["final_assets"]
            final_equity[i] = sim_results["final_equity"]
            annual_losses[i] = sim_results["annual_losses"]
            insurance_recoveries[i] = sim_results["insurance_recoveries"]
            retained_losses[i] = sim_results["retained_losses"]

            # Collect periodic ruin data
            if self.config.ruin_evaluation:
                ruin_at_year_all.append(sim_results["ruin_at_year"])

            completed = i + 1

            # Fire progress callback
            if progress_callback is not None and completed % callback_interval == 0:
                progress_callback(completed, n_sims, time.time() - seq_start)

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

        # Fire final callback so callers always see (total, total, ...)
        if progress_callback is not None and completed > 0 and completed != n_sims:
            progress_callback(completed, n_sims, time.time() - seq_start)

        # Truncate arrays if cancelled early
        if completed < n_sims:
            final_assets = final_assets[:completed]
            final_equity = final_equity[:completed]
            annual_losses = annual_losses[:completed]
            insurance_recoveries = insurance_recoveries[:completed]
            retained_losses = retained_losses[:completed]
            n_sims = completed

        # Calculate growth rates using equity (#355: total assets includes
        # liabilities, giving misleading growth when leverage changes)
        growth_rates = self._calculate_growth_rates(final_equity)

        # Calculate ruin probability
        ruin_probability = {}
        if self.config.ruin_evaluation:
            # Aggregate periodic ruin probabilities
            for eval_year in self.config.ruin_evaluation:
                if eval_year <= n_years:
                    ruin_count = sum(r.get(eval_year, False) for r in ruin_at_year_all)
                    ruin_probability[str(eval_year)] = ruin_count / n_sims

        # Always add final ruin probability (at max runtime)
        # Count both: companies with low final assets AND companies marked as ruined earlier
        if self.config.ruin_evaluation or ruin_at_year_all:
            # Count from ruin_at_year tracking (includes early bankruptcies)
            final_ruin_count = sum(r.get(n_years, False) for r in ruin_at_year_all)
        else:
            # Fallback to equity check if no tracking
            final_ruin_count = np.sum(np.less_equal(final_assets, self.config.insolvency_tolerance))
        ruin_probability[str(n_years)] = float(final_ruin_count / n_sims)

        return MonteCarloResults(
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

    def _run_parallel(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MonteCarloResults:
        """Run simulation in parallel using multiprocessing."""
        # Check if we're on Windows and have scipy import issues
        # Fall back to sequential execution in these cases
        try:
            # Test if we can import scipy successfully (needed for loss distributions)
            from scipy import stats  # noqa: F401  # pylint: disable=unused-import
        except (ImportError, TypeError) as e:
            warnings.warn(
                f"Scipy import failed in parallel mode: {e}. "
                "Falling back to sequential execution for reliability.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._run_sequential(
                progress_callback=progress_callback, cancel_event=cancel_event
            )

        n_sims = self.config.n_simulations
        n_workers = self.config.n_workers
        chunk_size = self.config.chunk_size

        # Create chunks
        chunks = []
        for i in range(0, n_sims, chunk_size):
            chunk_end = min(i + chunk_size, n_sims)
            chunk_seed = None if self.config.seed is None else self.config.seed + i
            chunks.append((i, chunk_end, chunk_seed))

        # Prepare config dictionary for the standalone function
        config_dict = {
            "n_years": self.config.n_years,
            "use_float32": self.config.use_float32,
            "ruin_evaluation": self.config.ruin_evaluation,
            "insolvency_tolerance": self.config.insolvency_tolerance,
            "letter_of_credit_rate": self.config.letter_of_credit_rate,
            "growth_rate": self.config.growth_rate,
            "time_resolution": self.config.time_resolution,
            "apply_stochastic": self.config.apply_stochastic,
        }

        # Run chunks in parallel using standalone function
        all_results = []
        par_start = time.time()
        completed_sims = 0
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks using the standalone function
                futures = {
                    executor.submit(
                        run_chunk_standalone,
                        chunk,
                        self.loss_generator,
                        self.insurance_program,
                        self.manufacturer,
                        config_dict,
                    ): chunk
                    for chunk in chunks
                }

                # Process completed tasks
                if self.config.progress_bar:
                    pbar = tqdm(total=len(chunks), desc="Processing chunks")

                for future in as_completed(futures):
                    # Check for cancellation before processing next result
                    if cancel_event is not None and cancel_event.is_set():
                        logger.info("Cancellation requested during parallel execution")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                    chunk_results = future.result()
                    all_results.append(chunk_results)

                    # Track completed simulations for callback
                    chunk_info = futures[future]
                    completed_sims += chunk_info[1] - chunk_info[0]

                    if self.config.progress_bar:
                        pbar.update(1)

                    if progress_callback is not None:
                        progress_callback(completed_sims, n_sims, time.time() - par_start)

                if self.config.progress_bar:
                    pbar.close()

        except (OSError, RuntimeError, ValueError, ImportError) as e:
            warnings.warn(
                f"Parallel execution failed: {e}. Falling back to sequential execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._run_sequential(
                progress_callback=progress_callback, cancel_event=cancel_event
            )

        # Combine results
        return self._combine_chunk_results(all_results)

    def _run_enhanced_parallel(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MonteCarloResults:
        """Run simulation using enhanced parallel executor.

        Uses CPU-optimized parallel execution with shared memory and
        intelligent chunking for better performance on budget hardware.
        """
        n_sims = self.config.n_simulations

        # Ensure parallel executor is available
        assert self.parallel_executor is not None, "Enhanced parallel executor not initialized"

        # Check if we can safely use enhanced parallel execution
        # On Windows with scipy import issues, fall back to standard parallel
        try:
            # Try to execute a test function to see if multiprocessing works
            import multiprocessing as mp

            with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context()) as executor:
                future = executor.submit(_test_worker_function)
                result = future.result(timeout=5)
                if not result:
                    raise RuntimeError("Worker test failed")
        except (ImportError, RuntimeError, TimeoutError) as e:
            warnings.warn(
                f"Enhanced parallel execution failed: {e}. "
                "Falling back to standard parallel execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._run_parallel(
                progress_callback=progress_callback, cancel_event=cancel_event
            )

        # Prepare shared data including the actual loss generator and insurance
        # program so _simulate_path_enhanced uses the configured model (issue #299).
        shared_data = {
            "n_years": self.config.n_years,
            "use_float32": self.config.use_float32,
            "ruin_evaluation": self.config.ruin_evaluation,
            "insolvency_tolerance": self.config.insolvency_tolerance,
            "enable_ledger_pruning": self.config.enable_ledger_pruning,
            "manufacturer_config": {
                "config": self.manufacturer.config,
                "stochastic_process": self.manufacturer.stochastic_process,
            },
            "loss_generator": self.loss_generator,
            "insurance_program": self.insurance_program,
            "base_seed": self.config.seed,
            "crn_base_seed": self.config.crn_base_seed,
            # Step parameters for manufacturer.step() (Issue #349)
            "letter_of_credit_rate": self.config.letter_of_credit_rate,
            "growth_rate": self.config.growth_rate,
            "time_resolution": self.config.time_resolution,
            "apply_stochastic": self.config.apply_stochastic,
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
            final_equity = np.zeros(n_results, dtype=dtype)
            annual_losses = np.zeros((n_results, n_years), dtype=dtype)
            insurance_recoveries = np.zeros((n_results, n_years), dtype=dtype)
            retained_losses = np.zeros((n_results, n_years), dtype=dtype)

            # Track periodic ruin if requested
            ruin_at_year_all = []

            valid_idx = 0
            for result in all_results:
                # Skip None results from failed simulations
                if result is None:
                    continue

                # Ensure result is a dictionary with expected keys
                if isinstance(result, dict) and "final_assets" in result:
                    final_assets[valid_idx] = result["final_assets"]
                    final_equity[valid_idx] = result.get(
                        "final_equity", result["final_assets"]
                    )  # type: ignore[assignment]
                    annual_losses[valid_idx] = result["annual_losses"]
                    insurance_recoveries[valid_idx] = result["insurance_recoveries"]
                    retained_losses[valid_idx] = result["retained_losses"]

                    # Collect periodic ruin data if present
                    if "ruin_at_year" in result:
                        ruin_at_year_all.append(result["ruin_at_year"])

                    valid_idx += 1
                else:
                    # Log warning for unexpected result format
                    # NOTE: Do NOT use ``import warnings`` here – it creates
                    # a local binding that shadows the module-level import and
                    # causes UnboundLocalError later in the function.  The
                    # module-level ``import warnings`` (line 11) is sufficient.
                    warnings.warn(f"Unexpected result format: {type(result)}")

            # Trim arrays to only valid results
            if valid_idx < n_results:
                final_assets = final_assets[:valid_idx]
                final_equity = final_equity[:valid_idx]
                annual_losses = annual_losses[:valid_idx]
                insurance_recoveries = insurance_recoveries[:valid_idx]
                retained_losses = retained_losses[:valid_idx]

            # Calculate derived metrics using equity (#355)
            growth_rates = self._calculate_growth_rates(final_equity)

            # Calculate ruin probability
            ruin_probability = {}
            total_simulations = len(final_assets)

            # Guard against division by zero when no valid results
            if total_simulations == 0:
                warnings.warn(
                    "No valid simulation results from parallel execution. "
                    "Falling back to sequential execution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._run_sequential()

            if self.config.ruin_evaluation and ruin_at_year_all:
                # Aggregate periodic ruin probabilities
                for eval_year in self.config.ruin_evaluation:
                    if eval_year <= n_years:
                        ruin_count = sum(r.get(eval_year, False) for r in ruin_at_year_all)
                        ruin_probability[str(eval_year)] = ruin_count / total_simulations

            # Always add final ruin probability (at max runtime)
            # Count both: companies with low final assets AND companies marked as ruined earlier
            if self.config.ruin_evaluation and ruin_at_year_all:
                # Count from ruin_at_year tracking (includes early bankruptcies)
                final_ruin_count = sum(r.get(n_years, False) for r in ruin_at_year_all)
            else:
                # Fallback to equity check if no tracking
                final_ruin_count = np.sum(
                    np.less_equal(final_assets, self.config.insolvency_tolerance)
                )
            ruin_probability[str(n_years)] = float(final_ruin_count / total_simulations)

            return MonteCarloResults(
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
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        # Add performance metrics from executor
        if self.config.monitor_performance and self.parallel_executor.performance_metrics:
            results.performance_metrics = self.parallel_executor.performance_metrics

        return results  # type: ignore[no-any-return]

    def _run_gpu(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MonteCarloResults:
        """Run simulation using GPU-accelerated vectorized engine.

        Falls back to parallel CPU execution if CuPy is not available.
        """
        from .gpu_backend import is_gpu_available
        from .gpu_mc_engine import extract_params, run_gpu_simulation

        if not is_gpu_available():
            logger.info("GPU not available, falling back to parallel CPU execution")
            warnings.warn(
                "use_gpu=True but CuPy is not available. Falling back to CPU parallel.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._run_parallel(
                progress_callback=progress_callback, cancel_event=cancel_event
            )

        # Extract flat parameters
        gpu_params = extract_params(
            self.manufacturer,
            self.insurance_program,
            self.loss_generator,
            self.config,
        )

        # Run vectorized simulation
        raw_results = run_gpu_simulation(
            gpu_params,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        # Calculate derived metrics
        final_assets = raw_results["final_assets"]
        final_equity = raw_results["final_equity"]
        growth_rates = self._calculate_growth_rates(final_equity)

        n_sims = len(final_assets)
        ruin_probability = {
            str(self.config.n_years): (
                float(np.mean(final_assets <= self.config.insolvency_tolerance))
                if n_sims > 0
                else 0.0
            )
        }

        return MonteCarloResults(
            final_assets=final_assets,
            annual_losses=raw_results["annual_losses"],
            insurance_recoveries=raw_results["insurance_recoveries"],
            retained_losses=raw_results["retained_losses"],
            growth_rates=growth_rates,
            ruin_probability=ruin_probability,
            metrics={},
            convergence={},
            execution_time=0,
            config=self.config,
        )

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

        # Track ruin at evaluation points
        ruin_at_year = {}
        if self.config.ruin_evaluation:
            for eval_year in self.config.ruin_evaluation:
                if eval_year <= n_years:  # Only track if within simulation period
                    ruin_at_year[eval_year] = False
        # Always track final year to ensure early bankruptcies propagate
        ruin_at_year[n_years] = False

        # Reset insurance program state for this simulation path (Issue #348)
        self.insurance_program.reset_annual()

        # Run simulation for each year
        for year in range(n_years):
            # Reset insurance program aggregate limits at start of each policy year (Issue #348)
            if year > 0:
                self.insurance_program.reset_annual()

            # CRN: reseed per (sim_id, year) for reproducible cross-scenario comparison
            if self.config.crn_base_seed is not None:
                year_ss = np.random.SeedSequence([self.config.crn_base_seed, sim_id, year])
                children = year_ss.spawn(2)
                self.loss_generator.reseed(int(children[0].generate_state(1)[0]))
                if manufacturer.stochastic_process is not None:
                    manufacturer.stochastic_process.reset(int(children[1].generate_state(1)[0]))

            # Generate losses using ManufacturingLossGenerator
            revenue = manufacturer.calculate_revenue()

            if hasattr(self.loss_generator, "generate_losses"):
                events, _ = self.loss_generator.generate_losses(
                    duration=1.0, revenue=float(revenue)
                )
            else:
                raise AttributeError(
                    f"Loss generator {type(self.loss_generator).__name__} has no generate_losses method"
                )

            # Calculate total loss
            total_loss = sum(event.amount for event in events)
            annual_losses[year] = total_loss

            # Apply insurance PER OCCURRENCE (not aggregate) and create
            # ClaimLiability objects with LoC collateral (Issue #342).
            total_recovery = 0.0
            total_retained = 0.0

            for event in events:
                # Process each event separately through insurance
                claim_result = self.insurance_program.process_claim(event.amount)
                event_recovery = claim_result.insurance_recovery
                event_retained = event.amount - event_recovery

                total_recovery += event_recovery
                total_retained += event_retained

                # Create ClaimLiability and post collateral for the retained
                # portion. The liability reduces equity via the balance sheet
                # (equity = assets - liabilities). We do NOT also call
                # record_insurance_loss() to avoid double-counting: the
                # liability already reduces equity, so reducing operating
                # income would penalise equity a second time.
                if event_retained > 0:
                    manufacturer.process_insurance_claim(
                        claim_amount=event.amount,
                        insurance_recovery=event_recovery,
                    )

            insurance_recoveries[year] = total_recovery
            retained_losses[year] = total_retained

            # Calculate insurance premium scaled by revenue
            # Premium should scale with exposure (revenue)
            current_revenue = manufacturer.calculate_revenue()
            base_revenue = (
                self.manufacturer.config.initial_assets
                * self.manufacturer.config.asset_turnover_ratio
            )
            revenue_scaling_factor = (
                float(current_revenue) / float(base_revenue) if base_revenue > 0 else 1.0
            )

            # Calculate scaled annual premium
            base_annual_premium = self.insurance_program.calculate_premium()
            annual_premium = base_annual_premium * revenue_scaling_factor

            if annual_premium > 0:
                manufacturer.record_insurance_premium(annual_premium)

            # Update manufacturer state with annual step
            # Apply stochastic if the manufacturer has a stochastic process
            apply_stochastic = (
                manufacturer.stochastic_process is not None or self.config.apply_stochastic
            )

            # Use config parameters instead of hardcoded values (Issue #349)
            manufacturer.step(
                letter_of_credit_rate=self.config.letter_of_credit_rate,
                growth_rate=self.config.growth_rate,
                time_resolution=self.config.time_resolution,
                apply_stochastic=apply_stochastic,
            )

            # Prune old ledger entries to bound memory (Issue #315)
            if self.config.enable_ledger_pruning and year > 0:
                manufacturer.ledger.prune_entries(before_date=year)

            # Check for ruin using insolvency tolerance or is_ruined flag
            if (
                float(manufacturer.equity) <= self.config.insolvency_tolerance
                or manufacturer.is_ruined
            ):
                # Mark ruin for all future evaluation points
                if self.config.ruin_evaluation:
                    for eval_year in ruin_at_year:
                        if year < eval_year:
                            ruin_at_year[eval_year] = True
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
                insurance_recoveries=(
                    insurance_recoveries[: year + 1] if ruin_occurred else insurance_recoveries
                ),
                retained_losses=retained_losses[: year + 1] if ruin_occurred else retained_losses,
                final_assets=float(manufacturer.total_assets),
                initial_assets=float(self.manufacturer.total_assets),
                ruin_occurred=ruin_occurred,
                ruin_year=ruin_year,
            )

        return {
            "final_assets": float(manufacturer.total_assets),
            "final_equity": float(manufacturer.equity),
            "annual_losses": annual_losses,
            "insurance_recoveries": insurance_recoveries,
            "retained_losses": retained_losses,
            "ruin_at_year": ruin_at_year,  # New field for periodic ruin tracking
        }

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> MonteCarloResults:
        """Combine results from parallel chunks.

        Args:
            chunk_results: List of chunk result dictionaries

        Returns:
            Combined MonteCarloResults
        """
        # Handle empty chunk results
        if not chunk_results:
            ruin_probability = {}
            if self.config.ruin_evaluation:
                for eval_year in self.config.ruin_evaluation:
                    if eval_year <= self.config.n_years:
                        ruin_probability[str(eval_year)] = 0.0
            ruin_probability[str(self.config.n_years)] = 0.0

            return MonteCarloResults(
                final_assets=np.array([]),
                annual_losses=np.array([]).reshape(0, self.config.n_years),
                insurance_recoveries=np.array([]).reshape(0, self.config.n_years),
                retained_losses=np.array([]).reshape(0, self.config.n_years),
                growth_rates=np.array([]),
                ruin_probability=ruin_probability,
                metrics={},  # Empty metrics for empty simulation
                convergence={},  # Empty convergence for empty simulation
                execution_time=0.0,
                config=self.config,
                time_series_aggregation=None,
            )

        # Concatenate arrays
        final_assets = np.concatenate([r["final_assets"] for r in chunk_results])
        final_equity = np.concatenate(
            [r["final_equity"] for r in chunk_results]
            if all("final_equity" in r for r in chunk_results)
            else [r["final_assets"] for r in chunk_results]
        )
        annual_losses = np.vstack([r["annual_losses"] for r in chunk_results])
        insurance_recoveries = np.vstack([r["insurance_recoveries"] for r in chunk_results])
        retained_losses = np.vstack([r["retained_losses"] for r in chunk_results])

        # Calculate derived metrics using equity (#355)
        growth_rates = self._calculate_growth_rates(final_equity)

        # Aggregate periodic ruin probabilities
        ruin_probability = {}
        total_simulations = len(final_assets)

        if self.config.ruin_evaluation:
            for eval_year in self.config.ruin_evaluation:
                if eval_year <= self.config.n_years:
                    # Count ruin occurrences for this evaluation year across all simulations
                    ruin_count = 0
                    for chunk in chunk_results:
                        if "ruin_at_year" in chunk:
                            # ruin_at_year is a list of dictionaries, one for each simulation in the chunk
                            for sim_ruin_data in chunk["ruin_at_year"]:
                                if sim_ruin_data.get(eval_year, False):
                                    ruin_count += 1
                    ruin_probability[str(eval_year)] = ruin_count / total_simulations

        # Always add final ruin probability
        # Count both: companies with low final assets AND companies marked as ruined earlier
        if self.config.ruin_evaluation and chunk_results:
            # Count from ruin_at_year tracking (includes early bankruptcies)
            final_ruin_count = 0
            for chunk in chunk_results:
                if "ruin_at_year" in chunk:
                    for sim_ruin_data in chunk["ruin_at_year"]:
                        if sim_ruin_data.get(self.config.n_years, False):
                            final_ruin_count += 1
        else:
            # Fallback to equity check if no tracking
            final_ruin_count = np.sum(np.less_equal(final_assets, self.config.insolvency_tolerance))
        ruin_probability[str(self.config.n_years)] = float(final_ruin_count / total_simulations)

        return MonteCarloResults(
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

    def _calculate_growth_rates(self, final_equity: np.ndarray) -> np.ndarray:
        """Calculate annualized growth rates based on equity.

        Uses equity rather than total assets so that growth reflects
        changes in owner value, consistent with how ruin detection
        uses equity (#355).

        Args:
            final_equity: Final equity values for each simulation

        Returns:
            Array of annualized log growth rates
        """
        initial_equity = float(self.manufacturer.equity)
        n_years = self.config.n_years

        # Avoid division by zero and log of negative numbers
        valid_mask = (final_equity > 0) & (initial_equity > 0)
        growth_rates = np.zeros_like(final_equity, dtype=np.float64)

        if np.any(valid_mask):
            growth_rates[valid_mask] = np.log(final_equity[valid_mask] / initial_equity) / n_years

        return growth_rates

    def _calculate_metrics(self, results: MonteCarloResults) -> Dict[str, float]:
        """Calculate risk metrics from simulation results.

        Args:
            results: Simulation results

        Returns:
            Dictionary of risk metrics
        """
        # Handle empty results gracefully
        if (
            len(results.final_assets) == 0
            or results.annual_losses is None
            or results.annual_losses.size == 0
        ):
            # Return default metrics for empty simulations
            return {
                "mean_loss": 0.0,
                "median_loss": 0.0,
                "std_loss": 0.0,
                "var_95": 0.0,
                "var_99": 0.0,
                "var_995": 0.0,
                "tvar_95": 0.0,
                "tvar_99": 0.0,
                "tvar_995": 0.0,
                "expected_shortfall_99": 0.0,
                "max_loss": 0.0,
                "mean_recovery": 0.0,
                "mean_retained": 0.0,
                "mean_growth_rate": 0.0,
                "survival_rate": 1.0,
                "ruin_probability": 0.0,
            }

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
            "sharpe_ratio": (
                np.mean(results.growth_rates) / np.std(results.growth_rates)
                if np.std(results.growth_rates) > 0
                else 0
            ),
        }

        return metrics

    def _check_convergence(self, results: MonteCarloResults) -> Dict[str, ConvergenceStats]:
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

    def _perform_advanced_aggregation(self, results: MonteCarloResults) -> MonteCarloResults:
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
                        "Ruin Probability": f"{results.ruin_probability.get(str(self.config.n_years), 0.0):.2%}",
                        "Mean Growth Rate": f"{np.mean(results.growth_rates):.4f}",
                    },
                )

        return results

    def export_results(
        self, results: MonteCarloResults, filepath: Path, file_format: str = "csv"
    ) -> None:
        """Export simulation results to file.

        Args:
            results: Simulation results to export
            filepath: Output file path
            file_format: Export format ('csv', 'json', 'hdf5')
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
        if file_format.lower() == "csv":
            ResultExporter.to_csv(results.aggregated_results, filepath)
        elif file_format.lower() == "json":
            ResultExporter.to_json(results.aggregated_results, filepath)
        elif file_format.lower() == "hdf5":
            ResultExporter.to_hdf5(results.aggregated_results, filepath)
        else:
            raise ValueError(f"Unsupported export format: {file_format}")

    def compute_bootstrap_confidence_intervals(
        self,
        results: MonteCarloResults,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = "percentile",
        show_progress: bool = False,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for key simulation metrics.

        All metrics share a single set of bootstrap resampling indices per
        iteration, reducing total work from ``7 * n_bootstrap`` to
        ``n_bootstrap`` iterations.

        Args:
            results: Simulation results to analyze.
            confidence_level: Confidence level for intervals (default 0.95).
            n_bootstrap: Number of bootstrap iterations (default 10000).
            method: Bootstrap method ('percentile' or 'bca').
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary mapping metric names to (lower, upper) confidence bounds.
        """
        from .bootstrap_analysis import BootstrapAnalyzer

        analyzer = BootstrapAnalyzer(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=self.config.seed,
            show_progress=show_progress,
        )

        # Prepare derived arrays
        ruin_indicator = (results.final_assets <= self.config.insolvency_tolerance).astype(float)
        mean_annual_losses = np.mean(results.annual_losses, axis=1)
        mean_recoveries = np.mean(results.insurance_recoveries, axis=1)

        # Build metrics dict: name -> (data, statistic)
        metrics: Dict[str, Tuple[np.ndarray, Callable[[np.ndarray], float]]] = {
            "Mean Final Assets": (results.final_assets, np.mean),
            "Median Final Assets": (results.final_assets, np.median),
            "Mean Growth Rate": (results.growth_rates, np.mean),
            "Ruin Probability": (ruin_indicator, np.mean),
            "Mean Annual Losses": (mean_annual_losses, np.mean),
            "Mean Insurance Recoveries": (mean_recoveries, np.mean),
        }

        if "var_99" in results.metrics:

            def var_99(x: np.ndarray) -> float:
                return float(np.percentile(x, 99))

            metrics["VaR(99%)"] = (results.final_assets, var_99)

        # Compute all CIs with shared bootstrap indices (single pass)
        bootstrap_results = analyzer.multi_confidence_interval(
            metrics, confidence_level=confidence_level, method=method
        )

        return {name: res.confidence_interval for name, res in bootstrap_results.items()}

    def _get_cache_key(self) -> str:
        """Generate cache key for current configuration.

        Returns:
            Cache key string
        """
        key_parts = [
            f"n_sims_{self.config.n_simulations}",
            f"n_years_{self.config.n_years}",
            f"seed_{self.config.seed}",
            f"ins_{hashlib.sha256(str(self.insurance_program).encode()).hexdigest()[:16]}",
            f"mfg_{hashlib.sha256(str(self.manufacturer).encode()).hexdigest()[:16]}",
        ]
        return "_".join(key_parts)

    def _save_cache(self, cache_key: str, results: MonteCarloResults) -> None:
        """Save results to cache.

        Args:
            cache_key: Cache key
            results: Results to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                safe_dump(results, f)
        except (IOError, OSError, pickle.PickleError) as e:
            warnings.warn(f"Failed to save cache: {e}")

    def _load_cache(self, cache_key: str) -> Optional[MonteCarloResults]:
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
                    loaded_data = safe_load(f)
                    return loaded_data  # type: ignore
            except (IOError, OSError, pickle.PickleError, EOFError, ValueError) as e:
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
                        logger.info(
                            "Early stopping: Convergence achieved at %s iterations",
                            f"{completed_iterations:,}",
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
    ) -> MonteCarloResults:
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

        # Calculate ruin probability using insolvency_tolerance for consistency
        # (issue #299: unify threshold across all execution paths)
        ruin_probability = {
            str(self.config.n_years): float(
                np.mean(arrays["final_assets"] <= self.config.insolvency_tolerance)
            )
        }

        results = MonteCarloResults(
            final_assets=arrays["final_assets"],
            annual_losses=arrays["annual_losses"],
            insurance_recoveries=arrays["insurance_recoveries"],
            retained_losses=arrays["retained_losses"],
            growth_rates=growth_rates,
            ruin_probability=ruin_probability,
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
    ) -> MonteCarloResults:
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

        batch_seed = self.config.seed

        # Work on a shallow copy so the caller's config object is never
        # mutated (issue #1018).  Only self.config is swapped; the
        # original dataclass instance stays untouched.
        original_config = self.config
        self.config = replace(original_config, n_simulations=check_interval)

        all_results = []
        total_iterations = 0
        converged = False

        try:
            while not converged and total_iterations < max_iterations:
                # Set seed for this batch (mutates only the local copy)
                self.config.seed = batch_seed

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
                        logger.debug("Iteration %d: R-hat = %.3f", total_iterations, max_r_hat)

                # Advance seed for next batch
                if batch_seed is not None:
                    batch_seed += check_interval
        finally:
            # Restore original config reference
            self.config = original_config

        return combined

    def _combine_multiple_results(self, results_list: List[MonteCarloResults]) -> MonteCarloResults:
        """Combine multiple simulation results.

        Args:
            results_list: List of MonteCarloResults to combine

        Returns:
            Combined MonteCarloResults
        """
        # Concatenate arrays
        final_assets = np.concatenate([r.final_assets for r in results_list])
        annual_losses = np.vstack([r.annual_losses for r in results_list])
        insurance_recoveries = np.vstack([r.insurance_recoveries for r in results_list])
        retained_losses = np.vstack([r.retained_losses for r in results_list])
        growth_rates = np.concatenate([r.growth_rates for r in results_list])

        # Recalculate metrics
        combined = MonteCarloResults(
            final_assets=final_assets,
            annual_losses=annual_losses,
            insurance_recoveries=insurance_recoveries,
            retained_losses=retained_losses,
            growth_rates=growth_rates,
            ruin_probability={
                str(self.config.n_years): float(
                    np.mean(final_assets <= self.config.insolvency_tolerance)
                )
            },
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


def __getattr__(name):
    if name == "SimulationConfig":
        warnings.warn(
            "SimulationConfig has been renamed to MonteCarloConfig. "
            "Please update your imports. The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return MonteCarloConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

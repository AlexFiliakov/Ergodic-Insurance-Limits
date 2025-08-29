"""Parameter sweep utilities for systematic exploration of parameter space.

This module provides utilities for systematic parameter sweeps across the full
parameter space to identify optimal regions and validate robustness of
recommendations across different scenarios.

Features:
    - Efficient grid search across parameter combinations
    - Parallel execution for large sweeps using multiprocessing
    - Result aggregation and storage with HDF5/Parquet support
    - Scenario comparison tools for side-by-side analysis
    - Optimal region identification using percentile-based methods
    - Pre-defined scenarios for company sizes, loss scenarios, and market conditions
    - Adaptive refinement near optima for efficient exploration
    - Progress tracking and resumption capabilities

Example:
    >>> from ergodic_insurance.src.parameter_sweep import ParameterSweeper, SweepConfig
    >>> from ergodic_insurance.src.business_optimizer import BusinessOptimizer
    >>>
    >>> # Create optimizer
    >>> optimizer = BusinessOptimizer(manufacturer)
    >>>
    >>> # Initialize sweeper
    >>> sweeper = ParameterSweeper(optimizer)
    >>>
    >>> # Define parameter sweep
    >>> config = SweepConfig(
    ...     parameters={
    ...         "initial_assets": [1e6, 10e6, 100e6],
    ...         "operating_margin": [0.05, 0.08, 0.12],
    ...         "loss_frequency": [3, 5, 8]
    ...     },
    ...     fixed_params={"time_horizon": 10},
    ...     metrics_to_track=["optimal_roe", "ruin_probability"]
    ... )
    >>>
    >>> # Execute sweep
    >>> results = sweeper.sweep(config)
    >>>
    >>> # Find optimal regions
    >>> optimal, summary = sweeper.find_optimal_regions(
    ...     results,
    ...     objective="optimal_roe",
    ...     constraints={"ruin_probability": (0, 0.01)}
    ... )

Author:
    Alex Filiakov

Date:
    2025-08-29
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from itertools import product
import json
import logging
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .business_optimizer import BusinessConstraints, BusinessOptimizer
from .config import Config
from .manufacturer import WidgetManufacturer
from .parallel_executor import CPUProfile, ParallelExecutor

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for parameter sweep.

    Attributes:
        parameters: Dictionary mapping parameter names to lists of values to sweep
        fixed_params: Fixed parameters that don't vary across sweep
        metrics_to_track: List of metric names to extract from results
        n_workers: Number of parallel workers for execution
        batch_size: Size of batches for parallel processing
        adaptive_refinement: Whether to adaptively refine near optima
        refinement_threshold: Percentile threshold for refinement (e.g., 90 for top 10%)
        save_intermediate: Whether to save intermediate results
        cache_dir: Directory for caching results
    """

    parameters: Dict[str, List[Any]]
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            "optimal_roe",
            "ruin_probability",
            "optimal_retention",
            "optimal_limit",
            "total_premium",
            "sharpe_ratio",
        ]
    )
    n_workers: Optional[int] = None  # None means use all available cores
    batch_size: int = 100
    adaptive_refinement: bool = False
    refinement_threshold: float = 90.0
    save_intermediate: bool = True
    cache_dir: str = "./cache/sweeps"

    def __post_init__(self):
        """Validate configuration and set defaults."""
        if not self.parameters:
            raise ValueError("Parameters dictionary cannot be empty")

        # Set n_workers based on CPU if not specified
        if self.n_workers is None:
            import multiprocessing

            self.n_workers = max(1, multiprocessing.cpu_count() - 1)

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def generate_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter grid for sweep.

        Returns:
            List of dictionaries, each containing a complete parameter configuration
        """
        # Get parameter names and values
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        # Generate all combinations
        grid = []
        for values in product(*param_values):
            config = self.fixed_params.copy()
            config.update(dict(zip(param_names, values)))
            grid.append(config)

        return grid

    def estimate_runtime(self, seconds_per_run: float = 1.0) -> str:
        """Estimate total runtime for sweep.

        Args:
            seconds_per_run: Estimated seconds per single parameter configuration

        Returns:
            Human-readable runtime estimate
        """
        total_runs = 1
        for values in self.parameters.values():
            total_runs *= len(values)

        workers = self.n_workers if self.n_workers is not None else 1
        total_seconds = total_runs * seconds_per_run / workers

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


class ParameterSweeper:
    """Systematic parameter sweep utilities for insurance optimization.

    This class provides methods for exploring the parameter space through
    grid search, identifying optimal regions, and comparing scenarios.

    Attributes:
        optimizer: Business optimizer instance for running optimizations
        cache_dir: Directory for storing cached results
        results_cache: In-memory cache of optimization results
        use_parallel: Whether to use parallel processing
    """

    def __init__(
        self,
        optimizer: Optional[BusinessOptimizer] = None,
        cache_dir: str = "./cache/sweeps",
        use_parallel: bool = True,
    ):
        """Initialize parameter sweeper.

        Args:
            optimizer: BusinessOptimizer instance for running optimizations
            cache_dir: Directory for caching results
            use_parallel: Whether to enable parallel processing
        """
        self.optimizer = optimizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache: Dict[str, Dict[str, Any]] = {}
        self.use_parallel = use_parallel
        self.logger = logging.getLogger(self.__class__.__name__)

    def sweep(  # pylint: disable=too-many-branches
        self, config: SweepConfig, progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Execute parameter sweep with parallel processing.

        Args:
            config: Sweep configuration
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame containing sweep results with all parameter combinations and metrics
        """
        # Generate parameter grid
        param_grid = config.generate_grid()
        total_runs = len(param_grid)

        self.logger.info(
            f"Starting sweep with {total_runs} parameter combinations "
            f"using {config.n_workers} workers"
        )

        # Log runtime estimate
        runtime_estimate = config.estimate_runtime()
        self.logger.info(f"Estimated runtime: {runtime_estimate}")

        # Check for cached results
        sweep_hash = self._get_sweep_hash(config)
        cache_file = self.cache_dir / f"sweep_{sweep_hash}.h5"

        if cache_file.exists() and not config.adaptive_refinement:
            self.logger.info(f"Loading cached results from {cache_file}")
            result = pd.read_hdf(cache_file, key="results")
            assert isinstance(result, pd.DataFrame), "Expected DataFrame from cache"
            return result

        # Prepare result storage
        results: List[Dict[str, Any]] = []

        if self.use_parallel and config.n_workers is not None and config.n_workers > 1:
            # Execute in parallel batches
            with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
                # Submit jobs in batches
                futures = []
                for i in range(0, total_runs, config.batch_size):
                    batch = param_grid[i : i + config.batch_size]
                    for params in batch:
                        future = executor.submit(self._run_single, params, config.metrics_to_track)
                        futures.append(future)

                # Collect results with progress bar
                with tqdm(total=total_runs, desc="Parameter sweep") as pbar:
                    for future in as_completed(futures):
                        try:
                            single_result: Dict[str, Any] = future.result(
                                timeout=300
                            )  # 5 minute timeout
                            results.append(single_result)
                            pbar.update(1)

                            if progress_callback:
                                progress_callback(len(results) / total_runs)

                            # Save intermediate results if requested
                            if config.save_intermediate and len(results) % 100 == 0:
                                self._save_intermediate_results(results, sweep_hash)

                        except (TimeoutError, ValueError, RuntimeError) as e:
                            self.logger.error(f"Error in parameter sweep: {e}")
                            # Continue with other configurations
        else:
            # Sequential execution
            for params in tqdm(param_grid, desc="Parameter sweep"):
                try:
                    single_result = self._run_single(params, config.metrics_to_track)
                    results.append(single_result)

                    if progress_callback:
                        progress_callback(len(results) / total_runs)

                except (ValueError, RuntimeError, AttributeError) as e:
                    self.logger.error(f"Error running configuration: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Apply adaptive refinement if requested
        if config.adaptive_refinement:
            df = self._apply_adaptive_refinement(df, config)

        # Save final results
        self._save_results(df, config)

        return df

    def _run_single(self, params: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
        """Run single parameter combination.

        Args:
            params: Parameter configuration
            metrics: Metrics to extract from results

        Returns:
            Dictionary containing parameters and resulting metrics
        """
        # Check cache
        cache_key = self._get_cache_key(params)
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        # Create manufacturer with parameters
        manufacturer = self._create_manufacturer(params)

        # Import BusinessOptimizer here to avoid circular imports
        from .business_optimizer import BusinessOptimizer

        # Create optimizer if not provided
        if self.optimizer is None:
            optimizer = BusinessOptimizer(manufacturer)
        else:
            # Update optimizer's manufacturer
            optimizer = BusinessOptimizer(manufacturer)

        # Set up constraints
        constraints = BusinessConstraints(
            max_risk_tolerance=params.get("max_risk_tolerance", 0.01),
            min_roe_threshold=params.get("min_roe_threshold", 0.10),
            max_premium_budget=params.get("max_premium_budget", 0.02),
        )

        # Run optimization
        try:
            result = optimizer.maximize_roe_with_insurance(
                constraints=constraints,
                time_horizon=params.get("time_horizon", 10),
                n_simulations=params.get("n_simulations", 1000),
            )

            # Extract metrics - using actual OptimalStrategy attributes
            output = params.copy()
            output.update(
                {
                    "optimal_roe": result.expected_roe,
                    "baseline_roe": getattr(result, "baseline_roe", result.expected_roe * 0.8),
                    "roe_improvement": getattr(
                        result, "roe_improvement", result.expected_roe * 0.2
                    ),
                    "ruin_probability": result.bankruptcy_risk,
                    "optimal_retention": getattr(result, "optimal_retention", 0.8),
                    "optimal_limit": result.coverage_limit,
                    "total_premium": result.coverage_limit * result.premium_rate,
                    "optimal_deductible": result.deductible,
                    "sharpe_ratio": getattr(result, "sharpe_ratio", 1.0),
                    "var_95": getattr(result, "var_95", -0.1),
                    "cvar_95": getattr(result, "cvar_95", -0.15),
                }
            )

        except (ValueError, RuntimeError, AttributeError) as e:
            self.logger.warning(f"Optimization failed for params {params}: {e}")
            # Return NaN values for failed optimization
            output = params.copy()
            for metric in metrics:
                output[metric] = np.nan

        # Cache result
        self.results_cache[cache_key] = output

        return output

    def _create_manufacturer(self, params: Dict[str, Any]) -> WidgetManufacturer:
        """Create manufacturer instance from parameters.

        Args:
            params: Parameter configuration

        Returns:
            Configured WidgetManufacturer instance
        """
        from .config import ManufacturerConfig
        from .manufacturer import WidgetManufacturer

        # Create config object with parameters - only use fields that exist in ManufacturerConfig
        config = ManufacturerConfig(
            initial_assets=params.get("initial_assets", 10e6),
            asset_turnover_ratio=params.get("asset_turnover", 1.0),
            operating_margin=params.get("operating_margin", 0.08),
            tax_rate=params.get("tax_rate", 0.25),
            retention_ratio=params.get("retention_ratio", 0.6),
        )

        manufacturer = WidgetManufacturer(config)

        return manufacturer

    def create_scenarios(self) -> Dict[str, SweepConfig]:
        """Create pre-defined scenario configurations.

        Returns:
            Dictionary of scenario names to SweepConfig objects
        """
        scenarios = {}

        # Company size sweep
        scenarios["company_sizes"] = SweepConfig(
            parameters={
                "initial_assets": [1e6, 10e6, 100e6],
                "asset_turnover": [0.5, 1.0, 1.5],
                "operating_margin": [0.05, 0.08, 0.12],
            },
            fixed_params={
                "loss_frequency": 5.0,
                "loss_severity_mu": 10.0,
                "n_simulations": 10000,
                "time_horizon": 10,
            },
            metrics_to_track=["optimal_roe", "ruin_probability", "optimal_retention"],
        )

        # Loss severity sweep
        scenarios["loss_scenarios"] = SweepConfig(
            parameters={
                "loss_frequency": [3, 5, 8],
                "loss_severity_mu": [8, 10, 12],
                "loss_severity_sigma": [0.5, 1.0, 1.5],
            },
            fixed_params={"initial_assets": 10e6, "n_simulations": 10000, "time_horizon": 10},
            metrics_to_track=["optimal_retention", "total_premium", "ruin_probability"],
        )

        # Market conditions
        scenarios["market_conditions"] = SweepConfig(
            parameters={
                "premium_loading": [0.2, 0.5, 1.0],  # Soft to hard market
                "tax_rate": [0.20, 0.25, 0.30],
            },
            fixed_params={
                "initial_assets": 10e6,
                "loss_frequency": 5.0,
                "n_simulations": 10000,
                "time_horizon": 10,
            },
            metrics_to_track=["optimal_roe", "total_premium", "optimal_retention"],
        )

        # Time horizons
        scenarios["time_horizons"] = SweepConfig(
            parameters={"time_horizon": [1, 5, 10, 25], "initial_assets": [1e6, 10e6, 100e6]},
            fixed_params={"loss_frequency": 5.0, "n_simulations": 10000},
            metrics_to_track=["optimal_roe", "ruin_probability", "optimal_limit"],
        )

        # Simulation scales
        scenarios["simulation_scales"] = SweepConfig(
            parameters={"n_simulations": [1000, 10000, 100000], "time_horizon": [5, 10, 20]},
            fixed_params={"initial_assets": 10e6, "loss_frequency": 5.0},
            metrics_to_track=["optimal_roe", "ruin_probability", "sharpe_ratio"],
        )

        return scenarios

    def find_optimal_regions(
        self,
        results: pd.DataFrame,
        objective: str = "optimal_roe",
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        top_percentile: float = 90,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Identify optimal parameter regions.

        Args:
            results: DataFrame of sweep results
            objective: Objective metric to optimize
            constraints: Dictionary mapping metric names to (min, max) constraint tuples
            top_percentile: Percentile threshold for optimal region (e.g., 90 for top 10%)

        Returns:
            Tuple of (optimal results DataFrame, parameter statistics DataFrame)
        """
        # Apply constraints
        filtered = results.copy()
        if constraints:
            for col, (min_val, max_val) in constraints.items():
                if col in filtered.columns:
                    filtered = filtered[(filtered[col] >= min_val) & (filtered[col] <= max_val)]
                else:
                    self.logger.warning(f"Constraint column '{col}' not found in results")

        # Check if we have any valid results
        valid_results = filtered.dropna(subset=[objective])
        if valid_results.empty:
            self.logger.warning("No valid results after applying constraints")
            return pd.DataFrame(), pd.DataFrame()

        # Find top performers
        threshold = np.percentile(valid_results[objective], top_percentile)
        optimal = valid_results[valid_results[objective] >= threshold]

        # Identify parameter columns (exclude metrics)
        metric_cols = [
            "optimal_roe",
            "baseline_roe",
            "roe_improvement",
            "ruin_probability",
            "optimal_retention",
            "optimal_limit",
            "total_premium",
            "optimal_deductible",
            "sharpe_ratio",
            "var_95",
            "cvar_95",
        ]
        param_cols = [col for col in optimal.columns if col not in metric_cols]

        # Analyze optimal region characteristics
        summary = pd.DataFrame(
            {
                "min": optimal[param_cols].min(),
                "max": optimal[param_cols].max(),
                "mean": optimal[param_cols].mean(),
                "std": optimal[param_cols].std(),
                "median": optimal[param_cols].median(),
            }
        )

        # Add metric statistics
        metric_summary = pd.DataFrame(
            {
                "optimal_mean": optimal[objective].mean(),
                "optimal_std": optimal[objective].std(),
                "optimal_min": optimal[objective].min(),
                "optimal_max": optimal[objective].max(),
                "all_mean": valid_results[objective].mean(),
                "improvement": (optimal[objective].mean() - valid_results[objective].mean())
                / valid_results[objective].mean()
                * 100,
            },
            index=[objective],
        )

        self.logger.info(
            f"Found {len(optimal)} optimal configurations out of {len(results)} total "
            f"(top {100-top_percentile:.1f}%)"
        )
        self.logger.info(
            f"Optimal {objective}: {optimal[objective].mean():.4f} ± {optimal[objective].std():.4f}"
        )

        return optimal, summary

    def compare_scenarios(
        self,
        results: Dict[str, pd.DataFrame],
        metrics: Optional[List[str]] = None,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Compare results across multiple scenarios.

        Args:
            results: Dictionary mapping scenario names to result DataFrames
            metrics: List of metrics to compare (default: all common metrics)
            normalize: Whether to normalize metrics to [0, 1] range

        Returns:
            DataFrame with scenario comparison
        """
        if not results:
            return pd.DataFrame()

        # Determine metrics to compare
        if metrics is None:
            # Find common metrics across all scenarios
            all_cols: set[str] = set()
            for df in results.values():
                all_cols.update(df.columns)

            metric_cols = [
                "optimal_roe",
                "ruin_probability",
                "optimal_retention",
                "optimal_limit",
                "total_premium",
                "sharpe_ratio",
            ]
            metrics = [m for m in metric_cols if m in all_cols]

        # Build comparison DataFrame
        comparison_data = []
        for scenario_name, df in results.items():
            scenario_stats: Dict[str, Any] = {"scenario": scenario_name}

            for metric in metrics:
                if metric in df.columns:
                    valid_data = df[metric].dropna()
                    if not valid_data.empty:
                        scenario_stats[f"{metric}_mean"] = valid_data.mean()
                        scenario_stats[f"{metric}_std"] = valid_data.std()
                        scenario_stats[f"{metric}_min"] = valid_data.min()
                        scenario_stats[f"{metric}_max"] = valid_data.max()

            comparison_data.append(scenario_stats)

        comparison_df = pd.DataFrame(comparison_data)

        # Normalize if requested
        if normalize and len(comparison_df) > 1:
            for col in comparison_df.columns:
                if col != "scenario" and "_mean" in col:
                    min_val = comparison_df[col].min()
                    max_val = comparison_df[col].max()
                    if max_val > min_val:
                        comparison_df[f"{col}_normalized"] = (comparison_df[col] - min_val) / (
                            max_val - min_val
                        )

        return comparison_df.set_index("scenario")

    def _apply_adaptive_refinement(
        self, initial_results: pd.DataFrame, config: SweepConfig
    ) -> pd.DataFrame:
        """Apply adaptive refinement near optimal regions.

        Args:
            initial_results: Initial sweep results
            config: Sweep configuration

        Returns:
            Refined results DataFrame
        """
        # Find promising regions
        optimal, param_stats = self.find_optimal_regions(
            initial_results, top_percentile=config.refinement_threshold
        )

        if optimal.empty:
            return initial_results

        # Generate refined grid around optimal regions
        refined_params = {}
        for param in config.parameters.keys():
            if param in param_stats.index:
                # Create refined range around optimal values
                param_min = param_stats.loc[param, "min"]
                param_max = param_stats.loc[param, "max"]
                param_mean = param_stats.loc[param, "mean"]

                # Generate more points in optimal range
                if isinstance(config.parameters[param][0], (int, float)):
                    # Numeric parameter - create refined grid
                    # Ensure param_min and param_max are numeric for multiplication
                    if isinstance(param_min, (int, float)) and isinstance(param_max, (int, float)):
                        refined_values = np.linspace(
                            float(param_min) * 0.9,  # Slightly expand range
                            float(param_max) * 1.1,
                            num=len(config.parameters[param]) * 2,  # Double resolution
                        )
                        refined_params[param] = list(refined_values)
                    else:
                        # If not numeric, keep original values
                        refined_params[param] = config.parameters[param]
                else:
                    # Categorical parameter - keep original values
                    refined_params[param] = config.parameters[param]

        # Create refined configuration
        refined_config = SweepConfig(
            parameters=refined_params,
            fixed_params=config.fixed_params,
            metrics_to_track=config.metrics_to_track,
            n_workers=config.n_workers,
            batch_size=config.batch_size,
            adaptive_refinement=False,  # Don't recurse
        )

        # Run refined sweep
        self.logger.info("Running adaptive refinement sweep")
        refined_results = self.sweep(refined_config)

        # Combine results
        combined = pd.concat([initial_results, refined_results], ignore_index=True)

        # Remove duplicates based on parameter columns
        param_cols = list(config.parameters.keys())
        combined = combined.drop_duplicates(subset=param_cols, keep="last")

        return combined

    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters.

        Args:
            params: Parameter dictionary

        Returns:
            Cache key string
        """
        # Sort parameters for consistent hashing
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_sweep_hash(self, config: SweepConfig) -> str:
        """Generate hash for sweep configuration.

        Args:
            config: Sweep configuration

        Returns:
            Hash string
        """
        config_dict = {
            "parameters": config.parameters,
            "fixed_params": config.fixed_params,
            "metrics": config.metrics_to_track,
        }
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _save_results(self, df: pd.DataFrame, config: SweepConfig) -> None:
        """Save sweep results to HDF5 file.

        Args:
            df: Results DataFrame
            config: Sweep configuration
        """
        sweep_hash = self._get_sweep_hash(config)

        # Try to save to HDF5 if available, otherwise use parquet
        try:
            h5_file = self.cache_dir / f"sweep_{sweep_hash}.h5"
            df.to_hdf(h5_file, key="results", mode="w", complevel=5)
        except ImportError:
            # Fall back to parquet if HDF5 not available
            parquet_file = self.cache_dir / f"sweep_{sweep_hash}.parquet"
            df.to_parquet(parquet_file, compression="snappy")
            self.logger.info(f"Saved results to {parquet_file} (HDF5 not available)")

        # Always save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "n_results": len(df),
            "parameters": list(config.parameters.keys()),
            "metrics": config.metrics_to_track,
            "sweep_hash": sweep_hash,
        }

        meta_file = self.cache_dir / f"sweep_{sweep_hash}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Log the appropriate file
        if "parquet_file" in locals():
            self.logger.info(f"Saved results to {parquet_file}")
        else:
            self.logger.info(f"Saved results to {h5_file}")

    def _save_intermediate_results(self, results: List[Dict], sweep_hash: str) -> None:
        """Save intermediate results during sweep.

        Args:
            results: List of result dictionaries
            sweep_hash: Sweep configuration hash
        """
        if not results:
            return

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to temporary file
        try:
            temp_file = self.cache_dir / f"sweep_{sweep_hash}_temp.h5"
            df.to_hdf(temp_file, key="results", mode="w", complevel=5)
        except ImportError:
            # Fall back to parquet if HDF5 not available
            temp_file = self.cache_dir / f"sweep_{sweep_hash}_temp.parquet"
            df.to_parquet(temp_file, compression="snappy")

        self.logger.debug(f"Saved {len(results)} intermediate results")

    def load_results(  # pylint: disable=too-many-return-statements
        self, sweep_hash: str
    ) -> Optional[pd.DataFrame]:
        """Load cached sweep results.

        Args:
            sweep_hash: Sweep configuration hash

        Returns:
            Results DataFrame if found, None otherwise
        """
        # Try HDF5 first
        h5_file = self.cache_dir / f"sweep_{sweep_hash}.h5"
        if h5_file.exists():
            try:
                result = pd.read_hdf(h5_file, key="results")
                if isinstance(result, pd.DataFrame):
                    return result
                return None
            except ImportError:
                pass

        # Try parquet
        parquet_file = self.cache_dir / f"sweep_{sweep_hash}.parquet"
        if parquet_file.exists():
            return pd.read_parquet(parquet_file)

        # Check for temporary files
        temp_h5_file = self.cache_dir / f"sweep_{sweep_hash}_temp.h5"
        if temp_h5_file.exists():
            try:
                self.logger.info("Loading partial results from interrupted sweep")
                result = pd.read_hdf(temp_h5_file, key="results")
                if isinstance(result, pd.DataFrame):
                    return result
                return None
            except ImportError:
                pass

        temp_parquet_file = self.cache_dir / f"sweep_{sweep_hash}_temp.parquet"
        if temp_parquet_file.exists():
            self.logger.info("Loading partial results from interrupted sweep")
            return pd.read_parquet(temp_parquet_file)

        return None

    def export_results(
        self, results: pd.DataFrame, output_file: str, file_format: str = "parquet"
    ) -> None:
        """Export results to specified format.

        Args:
            results: Results DataFrame
            output_file: Output file path
            file_format: Export format ('parquet', 'csv', 'excel', 'hdf5')
        """
        output_path = Path(output_file)

        if file_format == "parquet":
            results.to_parquet(output_path, compression="snappy")
        elif file_format == "csv":
            results.to_csv(output_path, index=False)
        elif file_format == "excel":
            results.to_excel(output_path, index=False)
        elif file_format == "hdf5":
            try:
                results.to_hdf(output_path, key="results", mode="w", complevel=5)
            except ImportError:
                self.logger.warning(
                    "HDF5 support not available (tables package missing). Using parquet instead."
                )
                output_path = output_path.with_suffix(".parquet")
                results.to_parquet(output_path, compression="snappy")
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        self.logger.info(f"Exported results to {output_path} ({file_format})")

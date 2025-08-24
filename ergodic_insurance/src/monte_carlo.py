"""High-performance Monte Carlo simulation engine for insurance optimization.

This module provides a vectorized, parallel Monte Carlo engine capable of running
millions of scenarios with convergence monitoring and efficient memory management.
"""

import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ergodic_insurance.src.convergence import ConvergenceDiagnostics, ConvergenceStats
from ergodic_insurance.src.insurance_program import InsuranceProgram
from ergodic_insurance.src.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.risk_metrics import RiskMetrics
from tqdm import tqdm


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

    def summary(self) -> str:
        """Generate summary of simulation results."""
        return (
            f"Simulation Results Summary\n"
            f"{'='*50}\n"
            f"Simulations: {self.config.n_simulations:,}\n"
            f"Years: {self.config.n_years}\n"
            f"Execution Time: {self.execution_time:.2f}s\n"
            f"Ruin Probability: {self.ruin_probability:.2%}\n"
            f"Mean Growth Rate: {np.mean(self.growth_rates):.4f}\n"
            f"VaR(99%): ${self.metrics.get('var_99', 0):,.0f}\n"
            f"TVaR(99%): ${self.metrics.get('tvar_99', 0):,.0f}\n"
            f"Convergence R-hat: {self.convergence.get('growth_rate', ConvergenceStats(0,0,0,False,0,0)).r_hat:.3f}\n"
        )


class MonteCarloEngine:
    """High-performance Monte Carlo simulation engine.

    Supports vectorized operations, parallel processing, and convergence monitoring.
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

        # Run simulation
        if self.config.parallel and self.config.n_simulations >= 10000:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        # Calculate metrics
        results.metrics = self._calculate_metrics(results)

        # Check convergence
        results.convergence = self._check_convergence(results)

        # Set execution time
        results.execution_time = time.time() - start_time

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
            metrics = manufacturer.step(
                working_capital_pct=0.2,
                growth_rate=0.0,  # No growth rate for now
            )

            # Check for ruin
            if manufacturer.assets <= 0:
                break

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
            "expected_shortfall_99": risk_metrics.expected_shortfall(risk_metrics.var(0.99)),
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
        except Exception as e:
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
                    return pickle.load(f)
            except Exception as e:
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
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint: {e}")

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

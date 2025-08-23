"""Memory-efficient Monte Carlo simulation engine.

This module provides a batch-processing Monte Carlo engine optimized for
limited memory environments, with support for parallel processing,
checkpointing, and streaming statistics calculation.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

from .claim_generator import ClaimGenerator
from .config import Config
from .insurance import InsurancePolicy
from .manufacturer import WidgetManufacturer

logger = logging.getLogger(__name__)


@dataclass
class StreamingStatistics:
    """Accumulator for streaming statistics calculation.

    Maintains running statistics without storing all data points,
    using Welford's online algorithm for numerical stability.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean
    min_val: float = float("inf")
    max_val: float = float("-inf")

    # Ergodic metrics
    log_sum: float = 0.0  # Sum of log returns for geometric mean
    survival_count: int = 0  # Number of survived scenarios

    # Percentile tracking (using reservoir sampling for memory efficiency)
    reservoir_size: int = 10000
    reservoir: np.ndarray = field(default_factory=lambda: np.array([]))

    def update(self, value: float, survived: bool = True) -> None:
        """Update statistics with a new value.

        Args:
            value: New value to incorporate.
            survived: Whether the scenario survived to completion.
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

        if survived:
            self.survival_count += 1
            if value > 0:
                self.log_sum += np.log(value)

        # Reservoir sampling for percentiles
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir = np.append(self.reservoir, value)
        else:
            # Random replacement
            idx = np.random.randint(0, self.count)
            if idx < self.reservoir_size:
                self.reservoir[idx] = value

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.m2 / (self.count - 1))

    @property
    def variance(self) -> float:
        """Calculate variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def geometric_mean(self) -> float:
        """Calculate geometric mean (time-average growth rate)."""
        if self.survival_count == 0:
            return 0.0
        return np.exp(self.log_sum / self.survival_count)

    @property
    def survival_rate(self) -> float:
        """Calculate survival rate."""
        if self.count == 0:
            return 0.0
        return self.survival_count / self.count

    def percentile(self, q: float) -> float:
        """Calculate percentile from reservoir.

        Args:
            q: Percentile to calculate (0-100).

        Returns:
            Estimated percentile value.
        """
        if len(self.reservoir) == 0:
            return 0.0
        return np.percentile(self.reservoir, q)

    def to_dict(self) -> Dict[str, float]:
        """Convert statistics to dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "p25": self.percentile(25),
            "p50": self.percentile(50),
            "p75": self.percentile(75),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "geometric_mean": self.geometric_mean,
            "survival_rate": self.survival_rate,
        }


@dataclass
class MonteCarloCheckpoint:
    """Checkpoint data for resuming simulations."""

    scenario_start: int
    scenario_end: int
    statistics: Dict[str, StreamingStatistics]
    timestamp: float

    def save(self, path: Path) -> None:
        """Save checkpoint to Parquet file.

        Args:
            path: Path to save checkpoint.
        """
        # Convert statistics to serializable format
        data = {
            "scenario_start": [self.scenario_start],
            "scenario_end": [self.scenario_end],
            "timestamp": [self.timestamp],
        }

        # Add statistics
        for metric_name, stats in self.statistics.items():
            for stat_name, value in stats.to_dict().items():
                data[f"{metric_name}_{stat_name}"] = [value]

        df = pd.DataFrame(data)
        df.to_parquet(path, compression="snappy")

    @classmethod
    def load(cls, path: Path) -> "MonteCarloCheckpoint":
        """Load checkpoint from Parquet file.

        Args:
            path: Path to checkpoint file.

        Returns:
            Loaded checkpoint.
        """
        df = pd.read_parquet(path)
        row = df.iloc[0]

        # Reconstruct statistics
        statistics = {}
        metric_names = set()
        for col in df.columns:
            if "_" in col and col not in ["scenario_start", "scenario_end", "timestamp"]:
                metric_name = col.rsplit("_", 1)[0]
                metric_names.add(metric_name)

        for metric_name in metric_names:
            stats = StreamingStatistics()
            stats.count = int(row.get(f"{metric_name}_count", 0))
            stats.mean = row.get(f"{metric_name}_mean", 0.0)
            stats.min_val = row.get(f"{metric_name}_min", float("inf"))
            stats.max_val = row.get(f"{metric_name}_max", float("-inf"))
            stats.survival_count = int(stats.count * row.get(f"{metric_name}_survival_rate", 0))
            statistics[metric_name] = stats

        return cls(
            scenario_start=int(row["scenario_start"]),
            scenario_end=int(row["scenario_end"]),
            statistics=statistics,
            timestamp=row["timestamp"],
        )


def run_single_scenario(
    scenario_id: int,
    config: Config,
    insurance_policy: InsurancePolicy,
    time_horizon: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single Monte Carlo scenario.

    Args:
        scenario_id: Unique scenario identifier.
        config: Configuration object.
        insurance_policy: Insurance policy to use.
        time_horizon: Number of years to simulate.
        seed: Random seed for this scenario.

    Returns:
        Dictionary of scenario results.
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed + scenario_id)

    # Initialize manufacturer
    manufacturer = WidgetManufacturer(config.manufacturer)

    # Initialize claim generator
    claim_generator = ClaimGenerator(
        frequency=0.1,
        severity_mean=5_000_000,
        severity_std=2_000_000,
        seed=seed + scenario_id if seed else None,
    )

    # Generate claims for entire horizon
    regular_claims, cat_claims = claim_generator.generate_all_claims(
        years=time_horizon,
        include_catastrophic=True,
        cat_frequency=0.01,
        cat_severity_mean=50_000_000,
        cat_severity_std=20_000_000,
    )
    all_claims = regular_claims + cat_claims

    # Group claims by year
    claims_by_year = {}
    for claim in all_claims:
        if claim.year not in claims_by_year:
            claims_by_year[claim.year] = []
        claims_by_year[claim.year].append(claim)

    # Track metrics
    final_assets = config.manufacturer.initial_assets
    final_equity = config.manufacturer.initial_assets
    total_claims = 0.0
    total_premiums = 0.0
    survived = True
    insolvency_year = None

    # Annual returns for ergodic calculations
    annual_returns = []

    # Run simulation
    for year in range(time_horizon):
        # Get claims for this year
        year_claims = claims_by_year.get(year, [])

        # Process claims through insurance
        for claim in year_claims:
            company_payment, insurance_recovery = insurance_policy.process_claim(claim.amount)
            manufacturer.assets -= company_payment
            total_claims += claim.amount

        # Pay insurance premium
        premium = insurance_policy.calculate_premium()
        manufacturer.assets -= premium
        total_premiums += premium

        # Calculate year's return before stepping
        initial_equity = manufacturer.equity

        # Step manufacturer forward
        metrics = manufacturer.step(
            working_capital_pct=config.working_capital.percent_of_sales,
            letter_of_credit_rate=config.debt.interest_rate,
            growth_rate=config.growth.annual_growth_rate,
        )

        # Calculate return
        if initial_equity > 0:
            year_return = (manufacturer.equity - initial_equity) / initial_equity
            annual_returns.append(1 + year_return)

        # Check for insolvency
        if manufacturer.equity <= 0:
            survived = False
            insolvency_year = year
            break

        final_assets = manufacturer.assets
        final_equity = manufacturer.equity

    # Calculate ergodic metrics
    if len(annual_returns) > 0:
        geometric_return = np.exp(np.mean(np.log(annual_returns))) - 1
        arithmetic_return = np.mean(annual_returns) - 1
    else:
        geometric_return = 0.0
        arithmetic_return = 0.0

    return {
        "scenario_id": scenario_id,
        "survived": survived,
        "insolvency_year": insolvency_year,
        "final_assets": final_assets,
        "final_equity": final_equity,
        "total_claims": total_claims,
        "total_premiums": total_premiums,
        "geometric_return": geometric_return,
        "arithmetic_return": arithmetic_return,
        "years_survived": insolvency_year if insolvency_year else time_horizon,
    }


class MonteCarloEngine:
    """Memory-efficient Monte Carlo simulation engine.

    Implements batch processing with parallel execution, checkpointing,
    and streaming statistics for memory-constrained environments.
    """

    def __init__(
        self,
        config: Config,
        insurance_policy: InsurancePolicy,
        n_scenarios: int = 10000,
        batch_size: int = 1000,
        n_jobs: int = 7,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_frequency: int = 5000,
        seed: Optional[int] = None,
    ):
        """Initialize Monte Carlo engine.

        Args:
            config: Configuration object.
            insurance_policy: Insurance policy to simulate.
            n_scenarios: Total number of scenarios to run.
            batch_size: Number of scenarios per batch.
            n_jobs: Number of parallel jobs.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_frequency: Save checkpoint every N scenarios.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.insurance_policy = insurance_policy
        self.n_scenarios = n_scenarios
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.checkpoint_frequency = checkpoint_frequency
        self.seed = seed

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("ergodic_insurance/checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize streaming statistics
        self.statistics = {
            "final_equity": StreamingStatistics(),
            "final_assets": StreamingStatistics(),
            "geometric_return": StreamingStatistics(),
            "arithmetic_return": StreamingStatistics(),
            "years_survived": StreamingStatistics(),
        }

        self.completed_scenarios = 0
        self.results_buffer = []

    def run_batch(self, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Run a batch of scenarios in parallel.

        Args:
            start_idx: Starting scenario index.
            end_idx: Ending scenario index (exclusive).

        Returns:
            List of scenario results.
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_single_scenario)(
                scenario_id=i,
                config=self.config,
                insurance_policy=self.insurance_policy,
                time_horizon=self.config.simulation.time_horizon_years,
                seed=self.seed,
            )
            for i in range(start_idx, end_idx)
        )

        return results

    def update_statistics(self, results: List[Dict[str, Any]]) -> None:
        """Update streaming statistics with batch results.

        Args:
            results: List of scenario results.
        """
        for result in results:
            survived = result["survived"]

            self.statistics["final_equity"].update(result["final_equity"], survived)
            self.statistics["final_assets"].update(result["final_assets"], survived)
            self.statistics["geometric_return"].update(result["geometric_return"], survived)
            self.statistics["arithmetic_return"].update(result["arithmetic_return"], survived)
            self.statistics["years_survived"].update(result["years_survived"], True)

    def save_checkpoint(self, scenario_end: int) -> Path:
        """Save current state as checkpoint.

        Args:
            scenario_end: Last completed scenario index.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint = MonteCarloCheckpoint(
            scenario_start=0,
            scenario_end=scenario_end,
            statistics=self.statistics,
            timestamp=time.time(),
        )

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{scenario_end:06d}.parquet"
        checkpoint.save(checkpoint_path)

        logger.info(f"Saved checkpoint at scenario {scenario_end}: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint and resume from saved state.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Number of completed scenarios.
        """
        checkpoint = MonteCarloCheckpoint.load(checkpoint_path)
        self.statistics = checkpoint.statistics
        self.completed_scenarios = checkpoint.scenario_end

        logger.info(f"Loaded checkpoint from scenario {checkpoint.scenario_end}")
        return checkpoint.scenario_end

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file.

        Returns:
            Path to latest checkpoint or None.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.parquet"))
        if not checkpoints:
            return None

        # Sort by scenario number in filename
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints[-1]

    def run(self, resume: bool = True) -> Dict[str, Any]:
        """Run Monte Carlo simulation.

        Args:
            resume: Whether to resume from checkpoint if available.

        Returns:
            Dictionary of simulation results and statistics.
        """
        start_time = time.time()

        # Check for existing checkpoint
        start_scenario = 0
        if resume:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                start_scenario = self.load_checkpoint(latest_checkpoint)
                logger.info(f"Resuming from scenario {start_scenario}")

        # Calculate batches
        n_remaining = self.n_scenarios - start_scenario
        n_batches = (n_remaining + self.batch_size - 1) // self.batch_size

        # Progress bar
        pbar = tqdm(
            total=n_remaining,
            initial=0,
            desc="Monte Carlo Simulation",
            unit="scenarios",
        )

        # Run batches
        for batch_idx in range(n_batches):
            batch_start = start_scenario + batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.n_scenarios)

            # Run batch
            batch_results = self.run_batch(batch_start, batch_end)

            # Update statistics
            self.update_statistics(batch_results)

            # Store results in buffer
            self.results_buffer.extend(batch_results)

            # Update progress
            pbar.update(batch_end - batch_start)
            self.completed_scenarios = batch_end

            # Save checkpoint if needed
            if batch_end % self.checkpoint_frequency == 0 or batch_end == self.n_scenarios:
                self.save_checkpoint(batch_end)

                # Save results buffer to Parquet
                if self.results_buffer:
                    results_df = pd.DataFrame(self.results_buffer)
                    results_path = self.checkpoint_dir / f"results_{batch_end:06d}.parquet"
                    results_df.to_parquet(results_path, compression="snappy")
                    logger.info(f"Saved {len(self.results_buffer)} results to {results_path}")
                    self.results_buffer = []  # Clear buffer

        pbar.close()

        # Calculate final statistics
        elapsed_time = time.time() - start_time

        results = {
            "n_scenarios": self.n_scenarios,
            "elapsed_time": elapsed_time,
            "scenarios_per_second": self.n_scenarios / elapsed_time,
            "statistics": {},
        }

        # Add all statistics
        for metric_name, stats in self.statistics.items():
            results["statistics"][metric_name] = stats.to_dict()

        # Log summary
        logger.info(f"Completed {self.n_scenarios} scenarios in {elapsed_time:.2f} seconds")
        logger.info(f"Survival rate: {self.statistics['final_equity'].survival_rate:.2%}")
        logger.info(
            f"Mean geometric return: {self.statistics['geometric_return'].geometric_mean:.4f}"
        )

        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Load all results from Parquet files into DataFrame.

        Returns:
            Combined DataFrame of all results.
        """
        result_files = sorted(self.checkpoint_dir.glob("results_*.parquet"))

        if not result_files:
            logger.warning("No result files found")
            return pd.DataFrame()

        dfs = []
        for file_path in result_files:
            df = pd.read_parquet(file_path)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

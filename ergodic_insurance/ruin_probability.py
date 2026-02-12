"""Ruin probability analysis for insurance optimization.

This module provides specialized classes and methods for analyzing bankruptcy
and ruin probabilities in insurance scenarios.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class RuinProbabilityConfig:
    """Configuration for ruin probability analysis."""

    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    n_simulations: int = 10000
    min_assets_threshold: float = 1_000_000
    min_equity_threshold: float = 0.0
    debt_service_coverage_ratio: float = 1.25
    consecutive_negative_periods: int = 3
    early_stopping: bool = True
    parallel: bool = True
    n_workers: Optional[int] = None
    seed: Optional[int] = None
    n_bootstrap: int = 1000
    bootstrap_confidence_level: float = 0.95


@dataclass
class RuinProbabilityResults:
    """Results from ruin probability analysis.

    Attributes:
        time_horizons: Array of time horizons analyzed (in years).
        ruin_probabilities: Probability of ruin at each time horizon.
        confidence_intervals: Bootstrap confidence intervals for probabilities.
        bankruptcy_causes: Distribution of bankruptcy causes by horizon.
        survival_curves: Survival probability curves over time.
        execution_time: Total execution time in seconds.
        n_simulations: Number of simulations run.
        convergence_achieved: Whether convergence criteria were met.
        mid_year_ruin_count: Number of simulations with mid-year ruin (Issue #279).
        ruin_month_distribution: Distribution of ruin events by month (0-11).
    """

    time_horizons: np.ndarray
    ruin_probabilities: np.ndarray
    confidence_intervals: np.ndarray
    bankruptcy_causes: Dict[str, np.ndarray]
    survival_curves: np.ndarray
    execution_time: float
    n_simulations: int
    convergence_achieved: bool
    # Issue #279: Track mid-year ruin events
    mid_year_ruin_count: int = 0
    ruin_month_distribution: Optional[Dict[int, int]] = None

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "Ruin Probability Analysis Results",
            "=" * 40,
            f"Simulations: {self.n_simulations:,}",
            f"Execution time: {self.execution_time:.2f} seconds",
            f"Convergence achieved: {self.convergence_achieved}",
            "",
            "Ruin Probabilities by Time Horizon:",
        ]

        for i, horizon in enumerate(self.time_horizons):
            prob = self.ruin_probabilities[i]
            ci_lower, ci_upper = self.confidence_intervals[i]
            lines.append(f"  {horizon:3d} years: {prob:6.2%} [{ci_lower:6.2%}, {ci_upper:6.2%}]")

        lines.append("")
        lines.append("Bankruptcy Causes (at 10 years):")
        if len(self.time_horizons) > 0:
            idx_10 = np.where(self.time_horizons == 10)[0]
            if len(idx_10) > 0:
                idx = idx_10[0]
                for cause, probs in self.bankruptcy_causes.items():
                    if idx < len(probs):
                        lines.append(f"  {cause:20s}: {probs[idx]:6.2%}")

        # Issue #279: Add mid-year ruin statistics
        if self.mid_year_ruin_count > 0:
            lines.append("")
            lines.append("Mid-Year Ruin Analysis:")
            mid_year_pct = self.mid_year_ruin_count / self.n_simulations * 100
            lines.append(
                f"  Mid-year ruin events: {self.mid_year_ruin_count:,} ({mid_year_pct:.1f}%)"
            )
            if self.ruin_month_distribution:
                lines.append("  Ruin by month:")
                month_names = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                for month, count in sorted(self.ruin_month_distribution.items()):
                    if count > 0:
                        lines.append(f"    {month_names[month]}: {count:,}")

        return "\n".join(lines)


class RuinProbabilityAnalyzer:
    """Analyzer for ruin probability calculations."""

    def __init__(self, manufacturer, loss_generator, insurance_program, config):
        """Initialize analyzer.

        Args:
            manufacturer: WidgetManufacturer instance
            loss_generator: ManufacturingLossGenerator instance
            insurance_program: InsuranceProgram instance
            config: MonteCarloConfig instance
        """
        self.manufacturer = manufacturer
        self.loss_generator = loss_generator
        self.insurance_program = insurance_program
        self.config = config

    def analyze_ruin_probability(
        self,
        config: Optional[RuinProbabilityConfig] = None,
    ) -> RuinProbabilityResults:
        """Analyze ruin probability across multiple time horizons.

        Args:
            config: Configuration for analysis

        Returns:
            RuinProbabilityResults with analysis results
        """
        config = config or RuinProbabilityConfig()
        start_time = time.time()

        # Run simulations
        if config.parallel and config.n_simulations > 1000:
            simulation_results = self._run_ruin_simulations_parallel(config)
        else:
            simulation_results = self._run_ruin_simulations_sequential(config)

        # Calculate results for each time horizon
        horizon_analysis = self._analyze_horizons(simulation_results, config)

        # Calculate bootstrap confidence intervals
        confidence_intervals = self._calculate_bootstrap_ci(
            simulation_results["bankruptcy_years"],
            config.time_horizons,
            config.n_bootstrap,
            config.bootstrap_confidence_level,
            seed=config.seed,
        )

        # Check convergence
        convergence_achieved = self._check_ruin_convergence(
            simulation_results["bankruptcy_years"],
            time_horizons=config.time_horizons,
        )

        # Convert survival curves to padded array
        padded_curves = self._pad_survival_curves(horizon_analysis["survival_curves"])

        return RuinProbabilityResults(
            time_horizons=np.array(config.time_horizons),
            ruin_probabilities=horizon_analysis["ruin_probs"],
            confidence_intervals=confidence_intervals,
            bankruptcy_causes=horizon_analysis["bankruptcy_causes"],
            survival_curves=padded_curves,
            execution_time=time.time() - start_time,
            n_simulations=config.n_simulations,
            convergence_achieved=convergence_achieved,
            # Issue #279: Include mid-year ruin statistics
            mid_year_ruin_count=simulation_results.get("mid_year_ruin_count", 0),
            ruin_month_distribution=simulation_results.get("ruin_month_distribution"),
        )

    def _analyze_horizons(self, simulation_results, config):
        """Analyze ruin data for each time horizon.

        Args:
            simulation_results: Results from simulations
            config: Configuration

        Returns:
            Dict with ruin_probs, bankruptcy_causes, survival_curves
        """
        ruin_probs = np.zeros(len(config.time_horizons))
        bankruptcy_causes = {
            "asset_threshold": np.zeros(len(config.time_horizons)),
            "equity_threshold": np.zeros(len(config.time_horizons)),
            "consecutive_negative": np.zeros(len(config.time_horizons)),
            "debt_service": np.zeros(len(config.time_horizons)),
        }
        survival_curves = []

        for i, horizon in enumerate(config.time_horizons):
            # Calculate ruin probability
            horizon_data = simulation_results["bankruptcy_years"] <= horizon
            ruin_probs[i] = np.mean(horizon_data)

            # Track bankruptcy causes — read each simulation's cause at its
            # actual bankruptcy year, not at the horizon.  With early stopping
            # the cause array is only populated up to the bankruptcy year, so
            # reading at ``horizon - 1`` would return the default ``False`` for
            # simulations that went bankrupt before the horizon (#355).
            bankruptcy_yrs = simulation_results["bankruptcy_years"]
            n_sims = len(bankruptcy_yrs)
            max_cause_year = simulation_results["bankruptcy_causes"]["asset_threshold"].shape[1]
            cause_year_idx = np.clip(bankruptcy_yrs - 1, 0, max_cause_year - 1)
            sim_indices = np.arange(n_sims)

            for cause, cause_data in bankruptcy_causes.items():
                cause_mask = simulation_results["bankruptcy_causes"][cause][
                    sim_indices, cause_year_idx
                ]
                bankrupted_subset = cause_mask[horizon_data]
                if len(bankrupted_subset) > 0:
                    cause_data[i] = np.mean(bankrupted_subset)
                else:
                    cause_data[i] = 0.0

            # Calculate survival curve
            bankruptcy_at_year = np.zeros(horizon)
            for y in range(1, horizon + 1):
                bankruptcy_at_year[y - 1] = np.sum(simulation_results["bankruptcy_years"] == y)
            survival_curve = 1.0 - np.cumsum(bankruptcy_at_year) / config.n_simulations
            survival_curves.append(survival_curve)

        return {
            "ruin_probs": ruin_probs,
            "bankruptcy_causes": bankruptcy_causes,
            "survival_curves": survival_curves,
        }

    def _pad_survival_curves(self, survival_curves):
        """Pad survival curves to uniform length.

        Args:
            survival_curves: List of survival curves

        Returns:
            Numpy array with padded curves
        """
        max_len = max(len(curve) for curve in survival_curves) if survival_curves else 0
        padded_curves = np.zeros((len(survival_curves), max_len))
        for i, curve in enumerate(survival_curves):
            padded_curves[i, : len(curve)] = curve
        return padded_curves

    def _run_ruin_simulations_sequential(self, config: RuinProbabilityConfig) -> Dict[str, Any]:
        """Run ruin probability simulations sequentially."""
        max_horizon = max(config.time_horizons)
        n_sims = config.n_simulations

        # Pre-allocate arrays
        bankruptcy_years = np.full(n_sims, max_horizon + 1, dtype=np.int32)
        bankruptcy_causes = {
            "asset_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "equity_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "consecutive_negative": np.zeros((n_sims, max_horizon), dtype=bool),
            "debt_service": np.zeros((n_sims, max_horizon), dtype=bool),
        }
        # Issue #279: Track mid-year ruin events
        mid_year_ruin_count = 0
        ruin_month_distribution: Dict[int, int] = {}

        iterator = range(n_sims)
        if self.config.progress_bar:
            iterator = tqdm(iterator, desc="Ruin probability simulations")

        for sim_id in iterator:
            result = self._run_single_ruin_simulation(sim_id, max_horizon, config)
            bankruptcy_years[sim_id] = result["bankruptcy_year"]
            for cause, cause_data in bankruptcy_causes.items():
                cause_data[sim_id] = result["causes"][cause]
            # Issue #279: Collect mid-year ruin statistics
            if result.get("is_mid_year_ruin", False):
                mid_year_ruin_count += 1
                ruin_month = result.get("ruin_month")
                if ruin_month is not None:
                    ruin_month_distribution[ruin_month] = (
                        ruin_month_distribution.get(ruin_month, 0) + 1
                    )

        return {
            "bankruptcy_years": bankruptcy_years,
            "bankruptcy_causes": bankruptcy_causes,
            "mid_year_ruin_count": mid_year_ruin_count,
            "ruin_month_distribution": ruin_month_distribution,
        }

    def _create_simulation_chunks(
        self, config: RuinProbabilityConfig, max_horizon: int
    ) -> List[Tuple[int, int, int, RuinProbabilityConfig, Optional[int]]]:
        """Create chunks for parallel processing."""
        n_sims = config.n_simulations
        n_workers = config.n_workers or 4
        chunk_size = max(100, n_sims // (n_workers * 10))

        chunks = []
        for i in range(0, n_sims, chunk_size):
            chunk_end = min(i + chunk_size, n_sims)
            chunk_seed = None if config.seed is None else config.seed + i
            chunks.append((i, chunk_end, max_horizon, config, chunk_seed))
        return chunks

    def _run_ruin_simulations_parallel(self, config: RuinProbabilityConfig) -> Dict[str, Any]:
        """Run ruin probability simulations in parallel."""
        max_horizon = max(config.time_horizons)
        chunks = self._create_simulation_chunks(config, max_horizon)

        # Run chunks in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=config.n_workers or 4) as executor:
            futures = {executor.submit(self._run_ruin_chunk, chunk): chunk for chunk in chunks}

            if self.config.progress_bar:
                pbar = tqdm(total=len(chunks), desc="Processing ruin chunks")

            for future in as_completed(futures):
                all_results.append(future.result())
                if self.config.progress_bar:
                    pbar.update(1)

            if self.config.progress_bar:
                pbar.close()

        # Combine results
        return self._combine_ruin_results(all_results)

    def _run_ruin_chunk(
        self,
        chunk: Tuple[int, int, int, RuinProbabilityConfig, Optional[int]],
    ) -> Dict[str, Any]:
        """Run a chunk of ruin simulations."""
        start_idx, end_idx, max_horizon, config, seed = chunk
        n_sims = end_idx - start_idx

        # Create local RNG for this chunk (loss generator has its own internal RNG)
        _rng = np.random.default_rng(seed)

        # Pre-allocate arrays
        bankruptcy_years = np.full(n_sims, max_horizon + 1, dtype=np.int32)
        bankruptcy_causes = {
            "asset_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "equity_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "consecutive_negative": np.zeros((n_sims, max_horizon), dtype=bool),
            "debt_service": np.zeros((n_sims, max_horizon), dtype=bool),
        }
        # Issue #279: Track mid-year ruin events
        mid_year_ruin_count = 0
        ruin_month_distribution: Dict[int, int] = {}

        for i in range(n_sims):
            result = self._run_single_ruin_simulation(start_idx + i, max_horizon, config)
            bankruptcy_years[i] = result["bankruptcy_year"]
            for cause, cause_data in bankruptcy_causes.items():
                cause_data[i] = result["causes"][cause]
            # Issue #279: Collect mid-year ruin statistics
            if result.get("is_mid_year_ruin", False):
                mid_year_ruin_count += 1
                ruin_month = result.get("ruin_month")
                if ruin_month is not None:
                    ruin_month_distribution[ruin_month] = (
                        ruin_month_distribution.get(ruin_month, 0) + 1
                    )

        return {
            "bankruptcy_years": bankruptcy_years,
            "bankruptcy_causes": bankruptcy_causes,
            "mid_year_ruin_count": mid_year_ruin_count,
            "ruin_month_distribution": ruin_month_distribution,
        }

    def _check_bankruptcy_conditions(
        self,
        manufacturer: Any,
        metrics: Dict[str, float],
        year: int,
        config: RuinProbabilityConfig,
        causes: Dict[str, np.ndarray],
        consecutive_negative_count: int,
    ) -> Tuple[bool, int]:
        """Check bankruptcy conditions and update causes.

        Returns:
            Tuple of (is_bankrupt, updated_consecutive_negative_count)
        """
        is_bankrupt = False
        current_assets = manufacturer.total_assets
        current_equity = metrics.get("equity", 0)

        # 0. Check manufacturer's is_ruined flag (set by check_solvency)
        # This is the authoritative insolvency status from the manufacturer
        if hasattr(manufacturer, "is_ruined") and manufacturer.is_ruined:
            # Mark equity threshold as the primary cause since that's what triggers is_ruined
            causes["equity_threshold"][year] = True
            is_bankrupt = True

        # 1. Asset threshold
        if current_assets <= config.min_assets_threshold:
            causes["asset_threshold"][year] = True
            is_bankrupt = True

        # 2. Equity threshold
        if current_equity <= config.min_equity_threshold:
            causes["equity_threshold"][year] = True
            is_bankrupt = True

        # 3. Consecutive negative equity
        if current_equity < 0:
            consecutive_negative_count += 1
            if consecutive_negative_count >= config.consecutive_negative_periods:
                causes["consecutive_negative"][year] = True
                is_bankrupt = True
        else:
            consecutive_negative_count = 0

        # 4. Debt service coverage (simplified)
        if hasattr(manufacturer, "debt") and manufacturer.debt > 0:
            debt_service = manufacturer.debt * 0.08  # Assume 8% debt service
            operating_income = metrics.get("operating_income", 0)
            if operating_income > 0 and debt_service > 0:
                coverage_ratio = operating_income / debt_service
                if coverage_ratio < config.debt_service_coverage_ratio:
                    causes["debt_service"][year] = True
                    is_bankrupt = True

        return is_bankrupt, consecutive_negative_count

    def _process_simulation_year(
        self,
        manufacturer: Any,
        year: int,
    ) -> Dict[str, float]:
        """Process a single year of simulation."""
        # Update state FIRST (process year's normal operations)
        metrics: Dict[str, float] = manufacturer.step(growth_rate=0.0)

        # Then apply losses at END of year
        # This prevents newly-created liabilities from being paid in the same year
        revenue = manufacturer.calculate_revenue()
        events, _ = self.loss_generator.generate_losses(duration=1.0, revenue=revenue)

        # Calculate and apply insurance
        total_loss = sum(event.amount for event in events)
        claim_result = self.insurance_program.process_claim(total_loss)
        recovery = claim_result.get("total_recovery", 0)
        retained = total_loss - recovery

        # Apply retained loss using proper claim processing to enforce limited liability
        # This ensures cash never goes negative and equity is properly floored at $0
        if retained > 0:
            manufacturer.process_uninsured_claim(
                claim_amount=retained,
                immediate_payment=False,  # Create liability with payment schedule starting next year
            )

        return metrics

    def _run_single_ruin_simulation(
        self,
        sim_id: int,
        max_horizon: int,
        config: RuinProbabilityConfig,
    ) -> Dict[str, Any]:
        """Run a single ruin probability simulation.

        Tracks multiple bankruptcy conditions with early stopping.
        Also tracks mid-year ruin events (Issue #279).
        """
        manufacturer = self.manufacturer.copy()
        causes = {
            "asset_threshold": np.zeros(max_horizon, dtype=bool),
            "equity_threshold": np.zeros(max_horizon, dtype=bool),
            "consecutive_negative": np.zeros(max_horizon, dtype=bool),
            "debt_service": np.zeros(max_horizon, dtype=bool),
        }

        consecutive_negative_count = 0
        bankruptcy_year = max_horizon + 1
        is_bankrupt = False
        # Issue #279: Track mid-year ruin events
        is_mid_year_ruin = False
        ruin_month: Optional[int] = None

        for year in range(max_horizon):
            metrics = self._process_simulation_year(manufacturer, year)

            # Issue #279: Check if this was a mid-year ruin event
            if manufacturer.is_ruined and manufacturer.ruin_month is not None:
                is_mid_year_ruin = True
                ruin_month = manufacturer.ruin_month

            # Check bankruptcy
            is_bankrupt, consecutive_negative_count = self._check_bankruptcy_conditions(
                manufacturer, metrics, year, config, causes, consecutive_negative_count
            )

            # Early stopping if bankrupt
            if is_bankrupt:
                bankruptcy_year = year + 1
                if config.early_stopping:
                    break

        # If no bankruptcy occurred, keep the default year beyond horizon
        if not is_bankrupt:
            bankruptcy_year = max_horizon + 1

        return {
            "bankruptcy_year": bankruptcy_year,
            "causes": causes,
            "is_mid_year_ruin": is_mid_year_ruin,
            "ruin_month": ruin_month,
        }

    def _combine_ruin_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine ruin simulation results from parallel chunks."""
        bankruptcy_years = np.concatenate([r["bankruptcy_years"] for r in chunk_results])

        # Combine bankruptcy causes
        all_causes = {}
        for cause in [
            "asset_threshold",
            "equity_threshold",
            "consecutive_negative",
            "debt_service",
        ]:
            cause_arrays = [r["bankruptcy_causes"][cause] for r in chunk_results]
            all_causes[cause] = np.vstack(cause_arrays)

        # Issue #279: Combine mid-year ruin statistics
        total_mid_year_ruin = sum(r.get("mid_year_ruin_count", 0) for r in chunk_results)
        combined_ruin_months: Dict[int, int] = {}
        for r in chunk_results:
            chunk_months = r.get("ruin_month_distribution", {})
            for month, count in chunk_months.items():
                combined_ruin_months[month] = combined_ruin_months.get(month, 0) + count

        return {
            "bankruptcy_years": bankruptcy_years,
            "bankruptcy_causes": all_causes,
            "mid_year_ruin_count": total_mid_year_ruin,
            "ruin_month_distribution": combined_ruin_months,
        }

    def _calculate_bootstrap_ci(
        self,
        bankruptcy_years: np.ndarray,
        time_horizons: List[int],
        n_bootstrap: int,
        confidence_level: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate bootstrap confidence intervals for ruin probabilities."""
        rng = np.random.default_rng(seed)
        n_sims = len(bankruptcy_years)
        confidence_intervals = np.zeros((len(time_horizons), 2))

        for i, horizon in enumerate(time_horizons):
            bootstrap_probs = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = rng.choice(bankruptcy_years, size=n_sims, replace=True)
                prob = np.mean(bootstrap_sample <= horizon)
                bootstrap_probs.append(prob)

            # Calculate percentiles
            alpha = 1 - confidence_level
            lower = np.percentile(bootstrap_probs, alpha / 2 * 100)
            upper = np.percentile(bootstrap_probs, (1 - alpha / 2) * 100)
            confidence_intervals[i] = [lower, upper]

        return confidence_intervals

    def _check_ruin_convergence(
        self,
        bankruptcy_years: np.ndarray,
        n_chains: int = 4,
        time_horizons: Optional[List[int]] = None,
    ) -> bool:
        """Check convergence using R-hat statistic."""
        if len(bankruptcy_years) < n_chains * 100:
            return False

        # Use the maximum configured horizon instead of hard-coded 10 (#355)
        max_horizon = max(time_horizons) if time_horizons else 10

        chain_size = len(bankruptcy_years) // n_chains
        chains = []
        for i in range(n_chains):
            chain_data = bankruptcy_years[i * chain_size : (i + 1) * chain_size]
            chain_binary = (chain_data <= max_horizon).astype(float)
            chains.append(chain_binary)

        chains = np.array(chains)  # type: ignore

        # Calculate R-hat statistic
        chain_means = np.mean(chains, axis=1)
        chain_vars = np.var(chains, axis=1, ddof=1)

        # Between-chain variance
        B = np.var(chain_means, ddof=1) * chain_size
        # Within-chain variance
        W = np.mean(chain_vars)

        # If W is zero (no within-chain variance), check if means differ
        if W < 1e-10:
            # If all chains have the same mean, converged
            # Chains have different means but no variance, not converged
            return bool(np.allclose(chain_means, chain_means[0]))

        var_plus = ((chain_size - 1) * W + B) / chain_size
        r_hat = np.sqrt(var_plus / W)

        return bool(r_hat < 1.05)  # Converged if R-hat < 1.05

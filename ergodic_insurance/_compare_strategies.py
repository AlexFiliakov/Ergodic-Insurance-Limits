"""Standalone Monte Carlo orchestration and strategy comparison.

This module provides standalone functions for running Monte Carlo simulations
and comparing insurance strategies. These were previously classmethods on
:class:`~ergodic_insurance.simulation.Simulation` but have been extracted
because they don't use any instance or class state — they orchestrate
:class:`~ergodic_insurance.monte_carlo.MonteCarloEngine` runs directly.

Functions:
    run_monte_carlo: Run paired insured/uninsured Monte Carlo simulation.
    compare_strategies: Compare multiple insurance strategies via Monte Carlo.

Since:
    Version 0.8.0
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ._warnings import ErgodicInsuranceDeprecationWarning
from .config import Config
from .insurance import InsurancePolicy
from .insurance_program import InsuranceProgram
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .monte_carlo import MonteCarloConfig, MonteCarloEngine

logger = logging.getLogger(__name__)


@dataclass
class StrategyComparisonResult:
    """Result of comparing multiple insurance strategies.

    Returned by :func:`compare_strategies`. Contains the shared uninsured
    baseline, per-strategy Monte Carlo results, and a summary DataFrame for
    quick comparison.

    Attributes:
        baseline: MonteCarloResults from the shared uninsured baseline run.
        strategy_results: Dict mapping strategy name to its MonteCarloResults.
        summary_df: DataFrame comparing strategies (same columns as the
            previous ``pd.DataFrame`` return type for backward compatibility).
        crn_seed: The Common Random Numbers base seed used for paired
            comparison across all runs.
    """

    baseline: Any  # MonteCarloResults (avoid top-level import cycle)
    strategy_results: Dict[str, Any]
    summary_df: pd.DataFrame
    crn_seed: Optional[int] = None

    def __repr__(self) -> str:
        n_strategies = len(self.strategy_results)
        names = ", ".join(sorted(self.strategy_results.keys()))
        return (
            f"StrategyComparisonResult("
            f"{n_strategies} strategies: [{names}], "
            f"summary_df={self.summary_df.shape[0]}x{self.summary_df.shape[1]})"
        )

    def __str__(self) -> str:
        lines = [
            f"StrategyComparisonResult — {len(self.strategy_results)} strategies",
            "",
            self.summary_df.to_string(),
        ]
        return "\n".join(lines)


def run_monte_carlo(  # pylint: disable=too-many-locals
    config: Config,
    insurance_policy: Optional[Union[InsuranceProgram, InsurancePolicy]] = None,
    n_scenarios: int = 10000,
    batch_size: int = 1000,
    n_jobs: int = 7,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_frequency: int = 5000,
    seed: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation using the MonteCarloEngine.

    Runs a paired insured/uninsured simulation and computes empirical
    ergodic analysis metrics comparing the two.

    Args:
        config: Configuration object.
        insurance_policy: Insurance program (or deprecated InsurancePolicy)
            to simulate. Accepts :class:`InsuranceProgram` (preferred) or
            :class:`InsurancePolicy` (deprecated, auto-converted).
        n_scenarios: Number of scenarios to run.
        batch_size: Scenarios per batch.
        n_jobs: Number of parallel jobs.
        checkpoint_dir: Directory for checkpoints.
        checkpoint_frequency: Save checkpoint every N scenarios.
        seed: Random seed.
        resume: Whether to resume from checkpoint.

    Returns:
        Dictionary of Monte Carlo results and statistics.

    Since:
        Version 0.8.0 (standalone; previously ``Simulation.run_monte_carlo``).
    """
    # Create loss generator
    loss_generator = ManufacturingLossGenerator(seed=seed)

    # Normalize to InsuranceProgram
    if insurance_policy is not None and isinstance(insurance_policy, InsurancePolicy):
        import warnings as _warnings

        _warnings.warn(
            "Passing InsurancePolicy to run_monte_carlo is deprecated. "
            "Use InsuranceProgram instead.",
            ErgodicInsuranceDeprecationWarning,
            stacklevel=2,
        )
        insurance_policy = insurance_policy.to_enhanced_program()

    # Create insurance program (Issue #348)
    if insurance_policy is not None:
        insurance_program = insurance_policy
    else:
        insurance_program = InsuranceProgram(layers=[])

    # Create manufacturer
    manufacturer = WidgetManufacturer(config=config.manufacturer)

    # Create simulation config
    sim_config = MonteCarloConfig(
        n_simulations=n_scenarios,
        n_years=config.simulation.time_horizon_years,
        parallel=n_jobs > 1 if n_jobs else True,
        n_workers=n_jobs,
        chunk_size=batch_size,
        checkpoint_interval=checkpoint_frequency,
        seed=seed,
    )

    # Run simulation WITH insurance
    engine_with_insurance = MonteCarloEngine(
        loss_generator=loss_generator,
        insurance_program=insurance_program,
        manufacturer=manufacturer,
        config=sim_config,
    )

    logger.info("Running Monte Carlo simulation WITH insurance...")
    results_with_insurance = engine_with_insurance.run()

    # Run simulation WITHOUT insurance using the same seed for fair comparison
    logger.info("Running Monte Carlo simulation WITHOUT insurance (same seed)...")

    # Create new loss generator with same seed for reproducibility
    loss_generator_no_insurance = ManufacturingLossGenerator(seed=seed)

    # Create new manufacturer instance (must be fresh for second run)
    manufacturer_no_insurance = WidgetManufacturer(config=config.manufacturer)

    # Create empty insurance program (no coverage)
    insurance_program_no_insurance = InsuranceProgram(layers=[])

    engine_without_insurance = MonteCarloEngine(
        loss_generator=loss_generator_no_insurance,
        insurance_program=insurance_program_no_insurance,
        manufacturer=manufacturer_no_insurance,
        config=sim_config,
    )

    results_without_insurance = engine_without_insurance.run()

    # Calculate empirical ergodic analysis by comparing the two runs
    ergodic_analysis = {}

    if hasattr(results_with_insurance, "statistics") and hasattr(
        results_without_insurance, "statistics"
    ):
        stats_with = results_with_insurance.statistics
        stats_without = results_without_insurance.statistics

        # Extract metrics from both runs
        if "geometric_return" in stats_with and "geometric_return" in stats_without:
            geo_with = stats_with["geometric_return"]
            geo_without = stats_without["geometric_return"]

            premium_cost = (
                insurance_policy.calculate_premium() if insurance_policy is not None else 0.0
            )
            initial_assets = config.manufacturer.initial_assets
            premium_rate = premium_cost / initial_assets

            # Calculate empirical insurance impact
            ergodic_analysis = {
                "premium_rate": premium_rate,
                # With insurance metrics
                "geometric_mean_return_with_insurance": geo_with.get("geometric_mean", 0.0),
                "survival_rate_with_insurance": geo_with.get("survival_rate", 0.0),
                "volatility_with_insurance": geo_with.get("std", 0.0),
                # Without insurance metrics
                "geometric_mean_return_without_insurance": geo_without.get("geometric_mean", 0.0),
                "survival_rate_without_insurance": geo_without.get("survival_rate", 0.0),
                "volatility_without_insurance": geo_without.get("std", 0.0),
                # Empirical deltas (insurance impact)
                "growth_impact": geo_with.get("geometric_mean", 0.0)
                - geo_without.get("geometric_mean", 0.0),
                "survival_benefit": geo_with.get("survival_rate", 0.0)
                - geo_without.get("survival_rate", 0.0),
                "volatility_reduction": geo_without.get("std", 0.0) - geo_with.get("std", 0.0),
            }

            logger.info(
                f"Empirical insurance impact: growth={ergodic_analysis['growth_impact']:.2%}, "
                f"survival={ergodic_analysis['survival_benefit']:.2%}, "
                f"volatility_reduction={ergodic_analysis['volatility_reduction']:.2%}"
            )

            return {
                "results_with_insurance": results_with_insurance,
                "results_without_insurance": results_without_insurance,
                "ergodic_analysis": ergodic_analysis,
            }

    # Fallback if statistics not available
    return {
        "results_with_insurance": results_with_insurance,
        "results_without_insurance": results_without_insurance,
    }


def compare_strategies(
    config: Config,
    insurance_policies: Mapping[str, Union[InsuranceProgram, InsurancePolicy]],
    n_scenarios: int = 1000,
    n_jobs: int = 7,
    seed: Optional[int] = None,
) -> StrategyComparisonResult:
    """Compare multiple insurance strategies via Monte Carlo.

    Runs a single uninsured baseline and one insured simulation per
    strategy, using Common Random Numbers (CRN) for proper paired
    comparison.  This requires N+1 total Monte Carlo runs for N
    strategies instead of the previous 2N.

    Args:
        config: Configuration object.
        insurance_policies: Dictionary of strategy name to
            :class:`InsuranceProgram` (or deprecated :class:`InsurancePolicy`).
        n_scenarios: Scenarios per policy.
        n_jobs: Number of parallel jobs.
        seed: Random seed.

    Returns:
        :class:`StrategyComparisonResult` containing per-strategy results,
        the shared uninsured baseline, and a summary DataFrame.

    Since:
        Version 0.8.0 (standalone; previously
        ``Simulation.compare_insurance_strategies``).
    """
    # Derive CRN base seed for paired comparison
    if seed is not None:
        crn_seed = seed
    else:
        crn_seed = int(np.random.SeedSequence().generate_state(1)[0])

    # Build shared simulation config with CRN enabled
    sim_config = MonteCarloConfig(
        n_simulations=n_scenarios,
        n_years=config.simulation.time_horizon_years,
        parallel=n_jobs > 1 if n_jobs else True,
        n_workers=n_jobs,
        checkpoint_interval=n_scenarios + 1,  # Don't checkpoint for comparisons
        seed=seed,
        crn_base_seed=crn_seed,
    )

    # --- Run uninsured baseline ONCE ---
    logger.info("Running shared uninsured baseline simulation...")
    baseline_results = MonteCarloEngine(
        loss_generator=ManufacturingLossGenerator(seed=seed),
        insurance_program=InsuranceProgram(layers=[]),
        manufacturer=WidgetManufacturer(config=config.manufacturer),
        config=sim_config,
    ).run()

    # --- Run each strategy (insured only) ---
    strategy_mc_results: Dict[str, Any] = {}
    results_rows: List[Dict[str, Any]] = []

    for policy_name, policy in insurance_policies.items():
        logger.info(f"Running Monte Carlo for strategy: {policy_name}")

        # Normalize InsurancePolicy -> InsuranceProgram
        if isinstance(policy, InsurancePolicy):
            import warnings as _warnings

            _warnings.warn(
                "Passing InsurancePolicy to compare_strategies "
                "is deprecated. Use InsuranceProgram instead.",
                ErgodicInsuranceDeprecationWarning,
                stacklevel=2,
            )
            converted = policy.to_enhanced_program()
            insurance_program: InsuranceProgram = (
                converted if converted is not None else InsuranceProgram(layers=[])
            )
        else:
            insurance_program = policy

        sim_results = MonteCarloEngine(
            loss_generator=ManufacturingLossGenerator(seed=seed),
            insurance_program=insurance_program,
            manufacturer=WidgetManufacturer(config=config.manufacturer),
            config=sim_config,
        ).run()

        strategy_mc_results[policy_name] = sim_results

        # Extract key metrics
        final_assets = sim_results.final_assets
        growth_rates = sim_results.growth_rates

        survival_rate = float(np.mean(final_assets > 0)) if len(final_assets) > 0 else 0.0
        mean_final = float(np.mean(final_assets)) if len(final_assets) > 0 else 0.0
        std_final = float(np.std(final_assets, ddof=1)) if len(final_assets) > 1 else 0.0
        if len(growth_rates) > 0:
            growth_factors = np.maximum(1 + growth_rates, 1e-10)
            geo_mean = float(np.exp(np.mean(np.log(growth_factors))) - 1)
        else:
            geo_mean = 0.0
        arith_mean = float(np.mean(growth_rates)) if len(growth_rates) > 0 else 0.0
        p95 = float(np.percentile(final_assets, 95)) if len(final_assets) > 0 else 0.0
        p99 = float(np.percentile(final_assets, 99)) if len(final_assets) > 0 else 0.0

        results_rows.append(
            {
                "policy": policy_name,
                "annual_premium": policy.calculate_premium(),
                "total_coverage": policy.get_total_coverage(),
                "survival_rate": survival_rate,
                "mean_final_equity": mean_final,
                "std_final_equity": std_final,
                "geometric_return": geo_mean,
                "arithmetic_return": arith_mean,
                "p95_final_equity": p95,
                "p99_final_equity": p99,
            }
        )

    comparison_df = pd.DataFrame(results_rows)

    # Add relative metrics
    if len(comparison_df) > 0:
        comparison_df["premium_to_coverage"] = (
            comparison_df["annual_premium"] / comparison_df["total_coverage"]
        )
        comparison_df["sharpe_ratio"] = (
            comparison_df["arithmetic_return"] / comparison_df["std_final_equity"]
        )

    return StrategyComparisonResult(
        baseline=baseline_results,
        strategy_results=strategy_mc_results,
        summary_df=comparison_df,
        crn_seed=crn_seed,
    )

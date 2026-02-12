"""Scenario comparison and batch analysis for ergodic insurance analysis.

Provides functions for comparing insured vs uninsured scenarios and
performing comprehensive batch analysis of simulation results.

For usage examples see the
`Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_
and the
`Advanced Scenarios tutorial <https://docs.mostlyoptimal.com/tutorials/06_advanced_scenarios.html>`_.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Union

import numpy as np

from .ergodic_types import (
    BatchAnalysisResults,
    ConvergenceStats,
    EnsembleAverageStats,
    ErgodicAdvantage,
    ScenarioComparison,
    ScenarioMetrics,
    SurvivalAnalysisStats,
    TimeAverageStats,
)
from .simulation import SimulationResults

if TYPE_CHECKING:
    from .ergodic_analyzer import ErgodicAnalyzer

logger = logging.getLogger(__name__)


def compare_scenarios(
    analyzer: ErgodicAnalyzer,
    insured_results: Union[List[SimulationResults], np.ndarray],
    uninsured_results: Union[List[SimulationResults], np.ndarray],
    metric: str = "equity",
) -> ScenarioComparison:
    """Compare insured vs uninsured scenarios using ergodic analysis.

    Performs side-by-side comparison calculating both time-average and
    ensemble-average growth rates to reveal the fundamental ergodic
    advantage of insurance.

    Args:
        analyzer: :class:`~ergodic_insurance.ergodic_analyzer.ErgodicAnalyzer`
            instance for growth rate calculations.
        insured_results: Simulation results from insured scenarios —
            list of :class:`SimulationResults`, list of arrays, or 2-D array.
        uninsured_results: Simulation results from uninsured scenarios
            (same format as *insured_results*).
        metric: Financial metric to analyze (default ``"equity"``).

    Returns:
        :class:`ScenarioComparison` with ``insured``, ``uninsured``, and
        ``ergodic_advantage`` fields containing growth statistics, survival
        rates, and significance test results.  Supports dict-style access
        for backward compatibility (with deprecation warnings).
    """
    # Extract trajectories
    if isinstance(insured_results, list) and isinstance(insured_results[0], SimulationResults):
        insured_trajectories = [np.asarray(getattr(r, metric)) for r in insured_results]
        uninsured_trajectories = [np.asarray(getattr(r, metric)) for r in uninsured_results]
    else:
        insured_trajectories = [np.asarray(traj) for traj in insured_results]
        uninsured_trajectories = [np.asarray(traj) for traj in uninsured_results]

    # Calculate time-average growth for each path
    insured_time_avg = [
        analyzer.calculate_time_average_growth(traj) for traj in insured_trajectories
    ]
    uninsured_time_avg = [
        analyzer.calculate_time_average_growth(traj) for traj in uninsured_trajectories
    ]

    # Filter out infinite values
    insured_time_avg_valid = [g for g in insured_time_avg if np.isfinite(g)]
    uninsured_time_avg_valid = [g for g in uninsured_time_avg if np.isfinite(g)]

    # Calculate ensemble averages
    insured_ensemble = analyzer.calculate_ensemble_average(
        insured_trajectories, metric="growth_rate"
    )
    uninsured_ensemble = analyzer.calculate_ensemble_average(
        uninsured_trajectories, metric="growth_rate"
    )

    # Compute per-scenario means
    insured_mean = float(np.mean(insured_time_avg_valid)) if insured_time_avg_valid else -np.inf
    uninsured_mean = (
        float(np.mean(uninsured_time_avg_valid)) if uninsured_time_avg_valid else -np.inf
    )

    # Guard against (-inf) - (-inf) = NaN when both lists are empty
    if np.isfinite(insured_mean) and np.isfinite(uninsured_mean):
        time_average_gain = insured_mean - uninsured_mean
    elif insured_mean == uninsured_mean:
        # Both -inf (both empty): no measurable gain
        time_average_gain = 0.0
    else:
        time_average_gain = float(insured_mean - uninsured_mean)

    # Significance test
    if insured_time_avg_valid and uninsured_time_avg_valid:
        t_stat, p_val = analyzer.significance_test(insured_time_avg_valid, uninsured_time_avg_valid)
        significant = bool(p_val < 0.05)
    else:
        t_stat = float(np.nan)
        p_val = float(np.nan)
        significant = False

    return ScenarioComparison(
        insured=ScenarioMetrics(
            time_average_mean=insured_mean,
            time_average_median=(
                float(np.median(insured_time_avg_valid)) if insured_time_avg_valid else -np.inf
            ),
            time_average_std=(
                float(np.std(insured_time_avg_valid, ddof=1))
                if len(insured_time_avg_valid) > 1
                else 0.0
            ),
            ensemble_average=insured_ensemble["mean"],
            survival_rate=insured_ensemble["survival_rate"],
            n_survived=int(insured_ensemble["n_survived"]),
        ),
        uninsured=ScenarioMetrics(
            time_average_mean=uninsured_mean,
            time_average_median=(
                float(np.median(uninsured_time_avg_valid)) if uninsured_time_avg_valid else -np.inf
            ),
            time_average_std=(
                float(np.std(uninsured_time_avg_valid, ddof=1))
                if len(uninsured_time_avg_valid) > 1
                else 0.0
            ),
            ensemble_average=uninsured_ensemble["mean"],
            survival_rate=uninsured_ensemble["survival_rate"],
            n_survived=int(uninsured_ensemble["n_survived"]),
        ),
        ergodic_advantage=ErgodicAdvantage(
            time_average_gain=time_average_gain,
            ensemble_average_gain=insured_ensemble["mean"] - uninsured_ensemble["mean"],
            survival_gain=insured_ensemble["survival_rate"] - uninsured_ensemble["survival_rate"],
            t_statistic=t_stat,
            p_value=p_val,
            significant=significant,
        ),
    )


def analyze_simulation_batch(
    analyzer: ErgodicAnalyzer,
    simulation_results: List[SimulationResults],
    label: str = "Scenario",
) -> BatchAnalysisResults:
    """Perform comprehensive ergodic analysis on a batch of simulation results.

    Provides time-average and ensemble statistics, convergence analysis,
    and survival metrics for a single scenario (e.g. all insured simulations).

    Args:
        analyzer: :class:`~ergodic_insurance.ergodic_analyzer.ErgodicAnalyzer`
            instance for growth rate calculations.
        simulation_results: List of :class:`SimulationResults` from
            Monte Carlo runs.
        label: Descriptive label for this batch (default ``"Scenario"``).

    Returns:
        :class:`BatchAnalysisResults` with ``label``, ``n_simulations``,
        ``time_average``, ``ensemble_average``, ``convergence``,
        ``survival_analysis``, and ``ergodic_divergence`` fields.
        Supports dict-style access for backward compatibility (with
        deprecation warnings).
    """
    # Extract equity trajectories
    equity_trajectories = np.array([r.equity for r in simulation_results])

    # Calculate time-average growth for each path
    time_avg_growth = [
        analyzer.calculate_time_average_growth(equity) for equity in equity_trajectories
    ]

    # Filter valid growth rates
    valid_growth = [g for g in time_avg_growth if np.isfinite(g)]

    # Calculate ensemble statistics
    ensemble_stats = analyzer.calculate_ensemble_average(equity_trajectories, metric="growth_rate")

    # Check convergence
    if len(valid_growth) > 0:
        converged, se = analyzer.check_convergence(np.array(valid_growth))
    else:
        converged, se = False, np.inf

    # Time-average statistics
    ta_mean = float(np.mean(valid_growth)) if valid_growth else -np.inf
    ta_median = float(np.median(valid_growth)) if valid_growth else -np.inf
    ta_std = float(np.std(valid_growth, ddof=1)) if len(valid_growth) > 1 else 0.0
    ta_min = float(np.min(valid_growth)) if valid_growth else -np.inf
    ta_max = float(np.max(valid_growth)) if valid_growth else -np.inf

    # Ergodic divergence
    if valid_growth:
        ergodic_divergence = float(ta_mean - ensemble_stats["mean"])
    else:
        ergodic_divergence = float(np.nan)

    return BatchAnalysisResults(
        label=label,
        n_simulations=len(simulation_results),
        time_average=TimeAverageStats(
            mean=ta_mean,
            median=ta_median,
            std=ta_std,
            min=ta_min,
            max=ta_max,
        ),
        ensemble_average=EnsembleAverageStats(
            mean=ensemble_stats["mean"],
            std=ensemble_stats["std"],
            median=ensemble_stats["median"],
            survival_rate=ensemble_stats["survival_rate"],
            n_survived=int(ensemble_stats["n_survived"]),
            n_total=int(ensemble_stats["n_total"]),
            mean_trajectory=ensemble_stats.get("mean_trajectory"),  # type: ignore[arg-type]
            std_trajectory=ensemble_stats.get("std_trajectory"),  # type: ignore[arg-type]
        ),
        convergence=ConvergenceStats(
            converged=converged,
            standard_error=se,
            threshold=analyzer.convergence_threshold,
        ),
        survival_analysis=SurvivalAnalysisStats(
            survival_rate=ensemble_stats["survival_rate"],
            mean_survival_time=float(
                np.mean(
                    [
                        r.insolvency_year if r.insolvency_year else len(r.years)
                        for r in simulation_results
                    ]
                )
            ),
        ),
        ergodic_divergence=ergodic_divergence,
    )

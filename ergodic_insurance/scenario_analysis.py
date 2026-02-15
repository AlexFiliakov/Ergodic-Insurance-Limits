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
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

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
    from .monte_carlo import MonteCarloResults

logger = logging.getLogger(__name__)


def _extract_from_monte_carlo(
    mc_results: "MonteCarloResults",
) -> Tuple[List[float], Dict[str, float]]:
    """Extract time-average growth rates and ensemble stats from MC results.

    Args:
        mc_results: :class:`~ergodic_insurance.monte_carlo.MonteCarloResults`
            from :meth:`MonteCarloEngine.run`.

    Returns:
        ``(time_avg_growth_list, ensemble_stats_dict)`` where
        *time_avg_growth_list* contains per-path growth rates (``-inf``
        for ruined paths, consistent with
        :meth:`~ergodic_insurance.ergodic_analyzer.ErgodicAnalyzer.calculate_time_average_growth`),
        and *ensemble_stats_dict* has keys ``mean``, ``std``, ``median``,
        ``survival_rate``, ``n_survived``, ``n_total``.
    """
    growth_rates = np.array(mc_results.growth_rates, dtype=np.float64)
    final_assets = np.array(mc_results.final_assets, dtype=np.float64)

    # The MC engine stores 0.0 for ruined paths; the ergodic analyzer
    # convention is -inf.  Convert ruined paths for consistency.
    ruined_mask = final_assets <= 0
    time_avg = growth_rates.copy()
    time_avg[ruined_mask] = -np.inf

    time_avg_list: List[float] = time_avg.tolist()

    # Ensemble stats from valid (non-ruined) paths
    valid = time_avg[~ruined_mask]
    n_total = len(growth_rates)
    n_survived = int(np.sum(~ruined_mask))

    ensemble: Dict[str, float] = {
        "mean": float(np.mean(valid)) if len(valid) > 0 else 0.0,
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "median": float(np.median(valid)) if len(valid) > 0 else 0.0,
        "survival_rate": n_survived / n_total if n_total > 0 else 0.0,
        "n_survived": n_survived,
        "n_total": n_total,
    }

    return time_avg_list, ensemble


def _extract_growth_and_ensemble(
    analyzer: "ErgodicAnalyzer",
    results: "Union[List[SimulationResults], np.ndarray, MonteCarloResults]",
    metric: str,
) -> Tuple[List[float], Dict[str, float]]:
    """Dispatch extraction of growth rates and ensemble stats by input type.

    Returns:
        ``(time_avg_growth_list, ensemble_stats_dict)`` — same contract as
        :func:`_extract_from_monte_carlo`.
    """
    # Lazy import to avoid circular dependency
    from .monte_carlo import MonteCarloResults as _MCResults

    if isinstance(results, _MCResults):
        return _extract_from_monte_carlo(results)

    # Existing paths: SimulationResults list or raw arrays
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], SimulationResults):
        trajectories = [np.asarray(getattr(r, metric)) for r in results]
    else:
        trajectories = [np.asarray(traj) for traj in results]

    time_avg = [analyzer.calculate_time_average_growth(traj) for traj in trajectories]
    ensemble = analyzer.calculate_ensemble_average(trajectories, metric="growth_rate")

    return time_avg, ensemble


def compare_scenarios(
    analyzer: "ErgodicAnalyzer",
    insured_results: "Union[List[SimulationResults], np.ndarray, MonteCarloResults]",
    uninsured_results: "Union[List[SimulationResults], np.ndarray, MonteCarloResults]",
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
            list of :class:`SimulationResults`, 2-D array, or
            :class:`~ergodic_insurance.monte_carlo.MonteCarloResults`.
        uninsured_results: Simulation results from uninsured scenarios
            (same format as *insured_results*; types may differ).
        metric: Financial metric to analyze (default ``"equity"``).
            Ignored when a :class:`MonteCarloResults` is passed (growth
            rates are pre-calculated by the MC engine).

    Returns:
        :class:`ScenarioComparison` with ``insured``, ``uninsured``, and
        ``ergodic_advantage`` fields containing growth statistics, survival
        rates, and significance test results.  Supports dict-style access
        for backward compatibility (with deprecation warnings).

    Example:
        Using :class:`MonteCarloResults` directly::

            from ergodic_insurance import ErgodicAnalyzer
            from ergodic_insurance.monte_carlo import MonteCarloEngine

            engine = MonteCarloEngine(manufacturer, config)
            insured_mc = engine.run(insurance_program=program)
            uninsured_mc = engine.run(insurance_program=None)

            analyzer = ErgodicAnalyzer()
            comparison = analyzer.compare_scenarios(insured_mc, uninsured_mc)
    """
    insured_time_avg, insured_ensemble = _extract_growth_and_ensemble(
        analyzer, insured_results, metric
    )
    uninsured_time_avg, uninsured_ensemble = _extract_growth_and_ensemble(
        analyzer, uninsured_results, metric
    )

    # Filter out infinite values
    insured_time_avg_valid = [g for g in insured_time_avg if np.isfinite(g)]
    uninsured_time_avg_valid = [g for g in uninsured_time_avg if np.isfinite(g)]

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

"""Ergodic analysis framework for comparing time-average vs ensemble-average growth.

Implements Ole Peters' ergodic economics framework for insurance decision
making.  For multiplicative processes like business growth with volatile
losses, ensemble averages and time averages diverge — insurance transforms
growth dynamics in ways that traditional expected value analysis cannot
capture.

Core class:
    :class:`ErgodicAnalyzer` — time-average growth, ensemble statistics,
    convergence analysis, and significance testing.

Data containers (re-exported from :mod:`~ergodic_insurance.ergodic_types`):
    :class:`ErgodicData`, :class:`ErgodicAnalysisResults`,
    :class:`ValidationResults`

Scenario and pipeline helpers (delegated to submodules):
    :func:`~ergodic_insurance.scenario_analysis.compare_scenarios`,
    :func:`~ergodic_insurance.scenario_analysis.analyze_simulation_batch`,
    :func:`~ergodic_insurance.integrated_analysis.integrate_loss_ergodic_analysis`,
    :func:`~ergodic_insurance.integrated_analysis.validate_insurance_ergodic_impact`

For usage examples see the
`Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

# Re-export data containers for backward compatibility — downstream code
# imports ErgodicData, ErgodicAnalysisResults, and ValidationResults from
# this module.
from .ergodic_types import (
    BatchAnalysisResults,
    ErgodicAnalysisResults,
    ErgodicData,
    ScenarioComparison,
    ValidationResults,
)
from .simulation import SimulationResults

if TYPE_CHECKING:
    from .insurance_program import InsuranceProgram
    from .loss_distributions import LossData

logger = logging.getLogger(__name__)

__all__ = [
    "ErgodicAnalyzer",
    "BatchAnalysisResults",
    "ErgodicAnalysisResults",
    "ErgodicData",
    "ScenarioComparison",
    "ValidationResults",
]


class ErgodicAnalyzer:
    """Analyzer for ergodic properties of insurance strategies.

    Computes time-average vs ensemble-average growth rates, demonstrating
    that insurance can improve time-average growth even when premiums
    exceed expected losses.

    Attributes:
        convergence_threshold: Standard error threshold for Monte Carlo
            convergence.

    For detailed examples see the
    `Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_.
    """

    def __init__(self, convergence_threshold: float = 0.01):
        """Initialize with convergence criteria.

        Args:
            convergence_threshold: SE threshold for Monte Carlo convergence
                (default 0.01).  Lower values require more simulations.
        """
        self.convergence_threshold = convergence_threshold

    # ------------------------------------------------------------------
    # Core ergodic calculations
    # ------------------------------------------------------------------

    def calculate_time_average_growth(
        self, values: np.ndarray, time_horizon: Optional[int] = None
    ) -> float:
        """Calculate time-average growth rate for a single trajectory.

        Computes ``g = (1/T) * ln(X(T)/X(0))``, the actual compound
        growth experienced by a single entity over time.

        Args:
            values: Array of values over time (equity, assets, wealth).
                Length must be >= 2.
            time_horizon: Override for time period *T*.  If *None*, uses
                ``len(values) - 1``.

        Returns:
            Time-average growth rate per period.  Returns ``-inf`` for
            bankrupt trajectories (final value <= 0) and ``0.0`` when
            ``time_horizon <= 0``.

        Note:
            Assumes uniform unit time steps.  For non-uniform steps,
            pass an explicit *time_horizon*.
        """
        if values is None or len(values) == 0:
            return -np.inf

        # Check for zero/negative values (bankruptcy)
        if values[-1] <= 0:
            return -np.inf

        # Skip any leading zero/negative values
        valid_mask = values > 0
        if not np.any(valid_mask):
            return -np.inf

        # Get first valid positive value
        first_idx = np.argmax(valid_mask)
        initial_value = values[first_idx]
        final_value = values[-1]

        # Calculate time period
        if time_horizon is None:
            time_horizon = int(len(values) - 1 - first_idx)

        # Calculate growth rate
        if final_value > 0 and initial_value > 0 and time_horizon > 0:
            growth_rate = float((1.0 / time_horizon) * np.log(final_value / initial_value))
            return growth_rate

        return 0.0 if time_horizon <= 0 else -np.inf

    def _extract_trajectory_values(
        self, trajectories: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract initial, final values and lengths from trajectories."""
        values_data = [(traj[-1], traj[0], len(traj)) for traj in trajectories if len(traj) > 0]
        if not values_data:
            return np.array([]), np.array([]), np.array([])

        finals, initials, lengths = zip(*values_data)
        return np.array(finals), np.array(initials), np.array(lengths)

    def _calculate_growth_rates(
        self, finals: np.ndarray, initials: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        """Calculate growth rates from trajectory values.

        Note:
            Assumes uniform unit time steps.  The formula
            ``log(f/i) / (t-1)`` divides by the number of
            inter-observation intervals, so the returned rate is
            per-period.  For non-uniform time steps, callers should
            use :meth:`calculate_time_average_growth` with an explicit
            *time_horizon*.
        """
        rates = [np.log(f / i) / (t - 1) for f, i, t in zip(finals, initials, lengths) if t > 1]
        return np.array(rates) if rates else np.array([])

    def _process_variable_length_trajectories(
        self, trajectories: List[np.ndarray], metric: str
    ) -> Dict[str, float]:
        """Process trajectories with variable lengths."""
        n_paths = len(trajectories)
        results: Dict[str, Any] = {}

        if metric in ["final_value", "growth_rate"]:
            # Get final and initial values from each trajectory
            final_values, initial_values, time_lengths = self._extract_trajectory_values(
                trajectories
            )

            # Filter valid paths (positive initial and final values)
            if len(final_values) > 0:
                valid_mask = (initial_values > 0) & (final_values > 0)
                valid_finals = final_values[valid_mask]
                valid_initials = initial_values[valid_mask]
                valid_lengths = time_lengths[valid_mask]
            else:
                valid_finals = valid_initials = valid_lengths = np.array([])

            if metric == "final_value":
                results["mean"] = np.mean(valid_finals) if len(valid_finals) > 0 else 0.0
                results["std"] = np.std(valid_finals, ddof=1) if len(valid_finals) > 1 else 0.0
                results["median"] = np.median(valid_finals) if len(valid_finals) > 0 else 0.0
            else:  # growth_rate
                growth_rates = self._calculate_growth_rates(
                    valid_finals, valid_initials, valid_lengths
                )
                results["mean"] = np.mean(growth_rates) if len(growth_rates) > 0 else 0.0
                results["std"] = np.std(growth_rates, ddof=1) if len(growth_rates) > 1 else 0.0
                results["median"] = np.median(growth_rates) if len(growth_rates) > 0 else 0.0
        else:  # full trajectory - not well-defined for different lengths
            results["mean_trajectory"] = None
            results["std_trajectory"] = None

        # Add survival statistics
        survived = sum(1 for traj in trajectories if len(traj) > 0 and traj[-1] > 0)
        results["survival_rate"] = survived / n_paths if n_paths > 0 else 0.0
        results["n_survived"] = survived
        results["n_total"] = n_paths

        return results

    def _process_fixed_length_trajectories(
        self, trajectories: np.ndarray, metric: str
    ) -> Dict[str, float]:
        """Process trajectories with fixed lengths."""
        n_paths, n_time = trajectories.shape
        results: Dict[str, Any] = {}

        if metric in ["final_value", "growth_rate"]:
            # Get final values (trajectories is now a numpy array)
            final_values = trajectories[:, -1]
            initial_values = trajectories[:, 0]

            # Filter valid paths (positive initial and final values)
            valid_mask = (initial_values > 0) & (final_values > 0)
            valid_finals = final_values[valid_mask]
            valid_initials = initial_values[valid_mask]

            if metric == "final_value":
                results["mean"] = np.mean(valid_finals) if len(valid_finals) > 0 else 0.0
                results["std"] = np.std(valid_finals, ddof=1) if len(valid_finals) > 1 else 0.0
                results["median"] = np.median(valid_finals) if len(valid_finals) > 0 else 0.0
            else:  # growth_rate
                growth_rates = np.log(valid_finals / valid_initials) / (n_time - 1)
                results["mean"] = np.mean(growth_rates) if len(growth_rates) > 0 else 0.0
                results["std"] = np.std(growth_rates, ddof=1) if len(growth_rates) > 1 else 0.0
                results["median"] = np.median(growth_rates) if len(growth_rates) > 0 else 0.0
        else:  # full trajectory
            results["mean_trajectory"] = np.mean(trajectories, axis=0)
            results["std_trajectory"] = np.std(trajectories, axis=0, ddof=1)

        # Add survival statistics
        survived = np.sum(trajectories[:, -1] > 0)
        results["survival_rate"] = survived / n_paths
        results["n_survived"] = survived
        results["n_total"] = n_paths

        return results

    def calculate_ensemble_average(
        self,
        trajectories: Union[List[np.ndarray], np.ndarray],
        metric: str = "final_value",
    ) -> Dict[str, float]:
        """Calculate ensemble average and statistics across multiple paths.

        Args:
            trajectories: List of 1-D arrays (variable lengths) or
                2-D array ``(n_paths, n_timesteps)``.
            metric: ``"final_value"``, ``"growth_rate"``, or ``"full"``.

        Returns:
            Dict with ``mean``, ``std``, ``median``, ``survival_rate``,
            ``n_survived``, ``n_total`` (and ``mean_trajectory`` /
            ``std_trajectory`` for metric ``"full"``).
        """
        # Handle list of arrays with potentially different lengths
        if isinstance(trajectories, list):
            # Check if all arrays have the same length
            lengths = [len(traj) for traj in trajectories if len(traj) > 0]
            if len(set(lengths)) == 1 and lengths:
                # All same length, can convert to 2D array
                trajectories_array = np.array(trajectories)
                return self._process_fixed_length_trajectories(trajectories_array, metric)
            # Different lengths, work with list
            return self._process_variable_length_trajectories(trajectories, metric)
        return self._process_fixed_length_trajectories(trajectories, metric)

    def check_convergence(self, values: np.ndarray, window_size: int = 100) -> Tuple[bool, float]:
        """Check Monte Carlo convergence using rolling standard error.

        Args:
            values: Array of metric values (e.g. time-average growth rates).
            window_size: Rolling window size (default 100).

        Returns:
            ``(converged, standard_error)`` — *converged* is ``True``
            when SE < :attr:`convergence_threshold`.
        """
        if len(values) < window_size:
            return False, np.inf

        # Standard error of the mean
        se = np.std(values[-window_size:], ddof=1) / np.sqrt(window_size)

        # Check if SE is below threshold
        converged = bool(se < self.convergence_threshold)

        return converged, se

    def significance_test(
        self,
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
        test_type: str = "two-sided",
    ) -> Tuple[float, float]:
        """Welch's t-test between two growth rate samples.

        Args:
            sample1: First sample (e.g. insured growth rates).
            sample2: Second sample (e.g. uninsured growth rates).
            test_type: ``"two-sided"``, ``"greater"``, or ``"less"``.

        Returns:
            ``(t_statistic, p_value)``.
        """
        # Perform Welch's t-test (does not assume equal variances)
        t_stat, p_value = stats.ttest_ind(
            sample1, sample2, equal_var=False, alternative=test_type, nan_policy="omit"
        )

        return t_stat, p_value

    # ------------------------------------------------------------------
    # Scenario comparison (delegated to scenario_analysis module)
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        insured_results: Union[List[SimulationResults], np.ndarray],
        uninsured_results: Union[List[SimulationResults], np.ndarray],
        metric: str = "equity",
    ) -> ScenarioComparison:
        """Compare insured vs uninsured scenarios using ergodic analysis.

        For detailed examples see the
        `Advanced Scenarios tutorial <https://docs.mostlyoptimal.com/tutorials/06_advanced_scenarios.html>`_.

        Args:
            insured_results: Simulation results from insured scenarios.
            uninsured_results: Simulation results from uninsured scenarios.
            metric: Financial metric to analyze (default ``"equity"``).

        Returns:
            :class:`ScenarioComparison` with ``insured``, ``uninsured``,
            and ``ergodic_advantage`` fields.  Supports dict-style access
            for backward compatibility (with deprecation warnings).
        """
        from . import scenario_analysis

        return scenario_analysis.compare_scenarios(self, insured_results, uninsured_results, metric)

    def analyze_simulation_batch(
        self, simulation_results: List[SimulationResults], label: str = "Scenario"
    ) -> BatchAnalysisResults:
        """Perform comprehensive ergodic analysis on a batch of simulations.

        For detailed examples see the
        `Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_.

        Args:
            simulation_results: List of :class:`SimulationResults` from
                Monte Carlo runs.
            label: Descriptive label for this batch.

        Returns:
            :class:`BatchAnalysisResults` with ``time_average``,
            ``ensemble_average``, ``convergence``, ``survival_analysis``,
            and ``ergodic_divergence`` fields.  Supports dict-style access
            for backward compatibility (with deprecation warnings).
        """
        from . import scenario_analysis

        return scenario_analysis.analyze_simulation_batch(self, simulation_results, label)

    # ------------------------------------------------------------------
    # Integrated analysis pipeline (delegated to integrated_analysis)
    # ------------------------------------------------------------------

    def integrate_loss_ergodic_analysis(
        self,
        loss_data: "LossData",
        insurance_program: Optional["InsuranceProgram"],
        manufacturer: Any,
        time_horizon: int,
        n_simulations: int = 100,
    ) -> ErgodicAnalysisResults:
        """End-to-end integrated loss modelling and ergodic analysis.

        Pipeline: validate -> apply insurance -> aggregate losses ->
        Monte Carlo -> ergodic metrics -> validate -> package results.

        For detailed examples see the
        `Optimization Workflow tutorial <https://docs.mostlyoptimal.com/tutorials/04_optimization_workflow.html>`_.

        Args:
            loss_data: Standardized loss data.
            insurance_program: Insurance program or *None* for uninsured.
            manufacturer: Manufacturer model instance.
            time_horizon: Analysis time horizon in years.
            n_simulations: Number of Monte Carlo runs (default 100).

        Returns:
            :class:`ErgodicAnalysisResults` with growth, survival,
            insurance impact, and validation status.
        """
        from . import integrated_analysis

        return integrated_analysis.integrate_loss_ergodic_analysis(
            self,
            loss_data,
            insurance_program,
            manufacturer,
            time_horizon,
            n_simulations,
        )

    def validate_insurance_ergodic_impact(
        self,
        base_scenario: SimulationResults,
        insurance_scenario: SimulationResults,
        insurance_program: Optional["InsuranceProgram"] = None,
    ) -> ValidationResults:
        """Validate insurance effects in ergodic calculations.

        Checks premium deductions, recovery credits, collateral impacts,
        and growth rate consistency.

        For detailed examples see the
        `Advanced Scenarios tutorial <https://docs.mostlyoptimal.com/tutorials/06_advanced_scenarios.html>`_.

        Args:
            base_scenario: Simulation results without insurance.
            insurance_scenario: Simulation results with insurance.
            insurance_program: Insurance program (optional, for detailed
                premium checks).

        Returns:
            :class:`ValidationResults` with individual check flags and
            diagnostics.
        """
        from . import integrated_analysis

        return integrated_analysis.validate_insurance_ergodic_impact(
            self, base_scenario, insurance_scenario, insurance_program
        )

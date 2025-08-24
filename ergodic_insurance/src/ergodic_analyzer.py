"""Ergodic analysis framework for comparing time-average vs ensemble-average growth.

This module provides the core ergodic analysis tools for evaluating insurance
decisions through the lens of time-average growth rates, demonstrating the
fundamental difference between expected value and experienced growth.

Based on Ole Peters' ergodic economics framework.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .simulation import SimulationResults

logger = logging.getLogger(__name__)


class ErgodicAnalyzer:
    """Analyzer for ergodic properties of insurance strategies.

    This class provides methods to calculate and compare time-average and
    ensemble-average growth rates, demonstrating how insurance transforms
    from a cost center (ensemble view) to a growth enabler (time view).
    """

    def __init__(self, convergence_threshold: float = 0.01):
        """Initialize the ergodic analyzer.

        Args:
            convergence_threshold: Standard error threshold for convergence check.
        """
        self.convergence_threshold = convergence_threshold

    def calculate_time_average_growth(
        self, values: np.ndarray, time_horizon: Optional[int] = None
    ) -> float:
        """Calculate time-average growth rate for a single trajectory.

        Uses the formula: g = (1/T) * ln(x(T)/x(0))

        Args:
            values: Array of values over time (e.g., equity, assets).
            time_horizon: Optional time horizon to use. If None, uses full array.

        Returns:
            Time-average growth rate.
        """
        # Handle edge cases and invalid trajectories
        if len(values) == 0:
            return -np.inf
        if values[-1] <= 0:
            return -np.inf

        if len(values) == 1:
            return 0.0

        # Filter out zero or negative values for calculating growth
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
            return float((1.0 / time_horizon) * np.log(final_value / initial_value))

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
        """Calculate growth rates from trajectory values."""
        rates = [np.log(f / i) / (t - 1) for f, i, t in zip(finals, initials, lengths) if t > 1]
        return np.array(rates) if rates else np.array([])

    def _process_variable_length_trajectories(
        self, trajectories: List[np.ndarray], metric: str
    ) -> Dict[str, float]:
        """Process trajectories with variable lengths."""
        n_paths = len(trajectories)
        results = {}

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
                results["std"] = np.std(valid_finals) if len(valid_finals) > 0 else 0.0
                results["median"] = np.median(valid_finals) if len(valid_finals) > 0 else 0.0
            else:  # growth_rate
                growth_rates = self._calculate_growth_rates(
                    valid_finals, valid_initials, valid_lengths
                )
                results["mean"] = np.mean(growth_rates) if len(growth_rates) > 0 else 0.0
                results["std"] = np.std(growth_rates) if len(growth_rates) > 0 else 0.0
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
        results = {}

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
                results["std"] = np.std(valid_finals) if len(valid_finals) > 0 else 0.0
                results["median"] = np.median(valid_finals) if len(valid_finals) > 0 else 0.0
            else:  # growth_rate
                growth_rates = np.log(valid_finals / valid_initials) / (n_time - 1)
                results["mean"] = np.mean(growth_rates) if len(growth_rates) > 0 else 0.0
                results["std"] = np.std(growth_rates) if len(growth_rates) > 0 else 0.0
                results["median"] = np.median(growth_rates) if len(growth_rates) > 0 else 0.0
        else:  # full trajectory
            results["mean_trajectory"] = np.mean(trajectories, axis=0)
            results["std_trajectory"] = np.std(trajectories, axis=0)

        # Add survival statistics
        survived = np.sum(trajectories[:, -1] > 0)
        results["survival_rate"] = survived / n_paths
        results["n_survived"] = survived
        results["n_total"] = n_paths

        return results

    def calculate_ensemble_average(
        self, trajectories: Union[List[np.ndarray], np.ndarray], metric: str = "final_value"
    ) -> Dict[str, float]:
        """Calculate ensemble average across multiple simulation paths.

        Args:
            trajectories: List of trajectory arrays or 2D array (paths x time).
            metric: Type of average to compute:
                   - "final_value": Average of final values
                   - "growth_rate": Average of growth rates
                   - "full": Average at each time step

        Returns:
            Dictionary with ensemble statistics.
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
        """Check if ensemble statistics have converged using standard error.

        Args:
            values: Array of values to check convergence.
            window_size: Size of rolling window for convergence check.

        Returns:
            Tuple of (converged, standard_error).
        """
        if len(values) < window_size:
            return False, np.inf

        # Calculate rolling mean and standard error
        rolling_means = np.convolve(values, np.ones(window_size) / window_size, mode="valid")

        # Standard error of the mean
        se = np.std(values[-window_size:]) / np.sqrt(window_size)

        # Check if SE is below threshold
        converged = bool(se < self.convergence_threshold)

        return converged, se

    def compare_scenarios(
        self,
        insured_results: Union[List[SimulationResults], np.ndarray],
        uninsured_results: Union[List[SimulationResults], np.ndarray],
        metric: str = "equity",
    ) -> Dict[str, Any]:
        """Compare insured vs uninsured scenarios using ergodic analysis.

        Args:
            insured_results: Results from insured simulations.
            uninsured_results: Results from uninsured simulations.
            metric: Which metric to analyze (equity, assets, etc.).

        Returns:
            Dictionary with comparison results including ergodic advantages.
        """
        # Extract trajectories
        if isinstance(insured_results, list) and isinstance(insured_results[0], SimulationResults):
            # Handle variable-length trajectories (e.g., due to insolvency)
            insured_trajectories = [getattr(r, metric) for r in insured_results]
            uninsured_trajectories = [getattr(r, metric) for r in uninsured_results]

            # Convert to list of arrays rather than 2D array to handle different lengths
            insured_trajectories = [np.asarray(traj) for traj in insured_trajectories]
            uninsured_trajectories = [np.asarray(traj) for traj in uninsured_trajectories]
        else:
            insured_trajectories = [np.asarray(traj) for traj in insured_results]
            uninsured_trajectories = [np.asarray(traj) for traj in uninsured_results]

        # Calculate time-average growth for each path
        insured_time_avg = [
            self.calculate_time_average_growth(traj) for traj in insured_trajectories
        ]
        uninsured_time_avg = [
            self.calculate_time_average_growth(traj) for traj in uninsured_trajectories
        ]

        # Filter out infinite values
        insured_time_avg_valid = [g for g in insured_time_avg if np.isfinite(g)]
        uninsured_time_avg_valid = [g for g in uninsured_time_avg if np.isfinite(g)]

        # Calculate ensemble averages
        insured_ensemble = self.calculate_ensemble_average(
            insured_trajectories, metric="growth_rate"
        )
        uninsured_ensemble = self.calculate_ensemble_average(
            uninsured_trajectories, metric="growth_rate"
        )

        # Compile results
        results = {
            "insured": {
                "time_average_mean": np.mean(insured_time_avg_valid)
                if insured_time_avg_valid
                else -np.inf,
                "time_average_median": np.median(insured_time_avg_valid)
                if insured_time_avg_valid
                else -np.inf,
                "time_average_std": np.std(insured_time_avg_valid)
                if insured_time_avg_valid
                else 0.0,
                "ensemble_average": insured_ensemble["mean"],
                "survival_rate": insured_ensemble["survival_rate"],
                "n_survived": insured_ensemble["n_survived"],
            },
            "uninsured": {
                "time_average_mean": np.mean(uninsured_time_avg_valid)
                if uninsured_time_avg_valid
                else -np.inf,
                "time_average_median": np.median(uninsured_time_avg_valid)
                if uninsured_time_avg_valid
                else -np.inf,
                "time_average_std": np.std(uninsured_time_avg_valid)
                if uninsured_time_avg_valid
                else 0.0,
                "ensemble_average": uninsured_ensemble["mean"],
                "survival_rate": uninsured_ensemble["survival_rate"],
                "n_survived": uninsured_ensemble["n_survived"],
            },
            "ergodic_advantage": {
                "time_average_gain": float(
                    float(np.mean(insured_time_avg_valid) if insured_time_avg_valid else -np.inf)
                    - float(
                        np.mean(uninsured_time_avg_valid) if uninsured_time_avg_valid else -np.inf
                    )
                ),
                "ensemble_average_gain": insured_ensemble["mean"] - uninsured_ensemble["mean"],
                "survival_gain": insured_ensemble["survival_rate"]
                - uninsured_ensemble["survival_rate"],
            },
        }

        # Add significance test if we have valid data
        if insured_time_avg_valid and uninsured_time_avg_valid:
            t_stat, p_value = self.significance_test(
                insured_time_avg_valid, uninsured_time_avg_valid
            )
            results["ergodic_advantage"]["t_statistic"] = t_stat  # type: ignore[index]
            results["ergodic_advantage"]["p_value"] = p_value  # type: ignore[index]
            results["ergodic_advantage"]["significant"] = p_value < 0.05  # type: ignore[index]
        else:
            results["ergodic_advantage"]["t_statistic"] = np.nan  # type: ignore[index]
            results["ergodic_advantage"]["p_value"] = np.nan  # type: ignore[index]
            results["ergodic_advantage"]["significant"] = False  # type: ignore[index]

        return results

    def significance_test(
        self,
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
        test_type: str = "two-sided",
    ) -> Tuple[float, float]:
        """Perform t-test for statistical significance between two samples.

        Args:
            sample1: First sample (e.g., insured growth rates).
            sample2: Second sample (e.g., uninsured growth rates).
            test_type: Type of test ('two-sided', 'greater', 'less').

        Returns:
            Tuple of (t_statistic, p_value).
        """
        # Perform independent samples t-test
        t_stat, p_value = stats.ttest_ind(
            sample1, sample2, alternative=test_type, nan_policy="omit"
        )

        return t_stat, p_value

    def analyze_simulation_batch(
        self, simulation_results: List[SimulationResults], label: str = "Scenario"
    ) -> Dict[str, Any]:
        """Analyze a batch of simulation results for ergodic properties.

        Args:
            simulation_results: List of SimulationResults objects.
            label: Label for this batch of simulations.

        Returns:
            Dictionary with comprehensive ergodic analysis.
        """
        # Extract equity trajectories
        equity_trajectories = np.array([r.equity for r in simulation_results])
        asset_trajectories = np.array([r.assets for r in simulation_results])

        # Calculate time-average growth for each path
        time_avg_growth = [
            self.calculate_time_average_growth(equity) for equity in equity_trajectories
        ]

        # Filter valid growth rates
        valid_growth = [g for g in time_avg_growth if np.isfinite(g)]

        # Calculate ensemble statistics
        ensemble_stats = self.calculate_ensemble_average(equity_trajectories, metric="growth_rate")

        # Check convergence
        if len(valid_growth) > 0:
            converged, se = self.check_convergence(np.array(valid_growth))
        else:
            converged, se = False, np.inf

        # Compile analysis
        analysis: Dict[str, Any] = {
            "label": label,
            "n_simulations": len(simulation_results),
            "time_average": {
                "mean": np.mean(valid_growth) if valid_growth else -np.inf,
                "median": np.median(valid_growth) if valid_growth else -np.inf,
                "std": np.std(valid_growth) if valid_growth else 0.0,
                "min": np.min(valid_growth) if valid_growth else -np.inf,
                "max": np.max(valid_growth) if valid_growth else -np.inf,
            },
            "ensemble_average": ensemble_stats,
            "convergence": {
                "converged": converged,
                "standard_error": se,
                "threshold": self.convergence_threshold,
            },
            "survival_analysis": {
                "survival_rate": ensemble_stats["survival_rate"],
                "mean_survival_time": np.mean(
                    [
                        r.insolvency_year if r.insolvency_year else len(r.years)
                        for r in simulation_results
                    ]
                ),
            },
        }

        # Calculate ergodic divergence
        if valid_growth:
            time_avg_mean = analysis["time_average"]["mean"]
            ensemble_mean = ensemble_stats["mean"]
            analysis["ergodic_divergence"] = time_avg_mean - ensemble_mean
        else:
            analysis["ergodic_divergence"] = np.nan

        return analysis

"""Data containers for ergodic analysis.

Provides the standardised data types used throughout the ergodic analysis
framework: input data containers, analysis results, and validation results.

For detailed usage examples see the
`Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import warnings

import numpy as np


@dataclass
class ErgodicData:
    """Standardized container for ergodic time series analysis.

    Attributes:
        time_series: Array of time points corresponding to *values*.
            Should be monotonically increasing.
        values: Array of observed values (e.g. equity, assets) at each
            time point.  Must have the same length as *time_series*.
        metadata: Analysis metadata such as simulation parameters,
            data source, and units.

    See Also:
        `Getting Started tutorial <https://docs.mostlyoptimal.com/tutorials/01_getting_started.html>`_
    """

    time_series: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate data consistency and integrity.

        Returns:
            ``True`` if arrays are non-empty and have matching lengths.
        """
        if len(self.time_series) == 0 or len(self.values) == 0:
            return False
        return len(self.time_series) == len(self.values)


@dataclass
class ErgodicAnalysisResults:
    """Comprehensive results from integrated ergodic analysis.

    Attributes:
        time_average_growth: Mean time-average growth rate across all
            valid simulation paths.  May be ``-inf`` if all paths ended
            in bankruptcy.
        ensemble_average_growth: Ensemble average growth rate calculated
            from the mean of initial and final values across all paths.
        survival_rate: Fraction of paths that remained solvent ``[0, 1]``.
        ergodic_divergence: ``time_average_growth - ensemble_average_growth``.
        insurance_impact: Insurance-related metrics (``premium_cost``,
            ``recovery_benefit``, ``net_benefit``, ``growth_improvement``).
        validation_passed: Whether the analysis passed internal validation.
        metadata: Additional analysis metadata (``n_simulations``,
            ``time_horizon``, ``n_survived``, ``loss_statistics``).

    Note:
        All growth rates are expressed as decimal values (0.05 = 5 %).
        Always check *validation_passed* before interpreting results.

    See Also:
        `Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_
    """

    time_average_growth: float
    ensemble_average_growth: float
    survival_rate: float
    ergodic_divergence: float
    insurance_impact: Dict[str, float]
    validation_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResults:
    """Results from insurance impact validation analysis.

    Attributes:
        premium_deductions_correct: Whether premiums are properly deducted
            from cash flows.
        recoveries_credited: Whether recoveries are properly credited.
        collateral_impacts_included: Whether collateral costs are modeled.
        time_average_reflects_benefit: Whether growth rates reflect
            insurance benefits.
        overall_valid: Master validation flag — all individual checks passed.
        details: Detailed diagnostic information from each validation
            check, useful for troubleshooting failures.

    See Also:
        `Advanced Scenarios tutorial <https://docs.mostlyoptimal.com/tutorials/06_advanced_scenarios.html>`_
    """

    premium_deductions_correct: bool
    recoveries_credited: bool
    collateral_impacts_included: bool
    time_average_reflects_benefit: bool
    overall_valid: bool
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dict-compatible mixin for backward compatibility (#713)
# ---------------------------------------------------------------------------


class _DictAccessMixin:
    """Mixin providing backward-compatible dict-style access with deprecation warnings.

    Allows ``result["key"]`` and ``"key" in result`` so that existing code
    using ``Dict[str, Any]`` return values continues to work.  Attribute
    access (``result.key``) is preferred and emits no warnings.
    """

    def __getitem__(self, key: str) -> Any:
        warnings.warn(
            f"Dict-style access result['{key}'] is deprecated. "
            f"Use attribute access result.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible ``.get()`` with deprecation warning."""
        warnings.warn(
            f"Dict-style access result.get('{key}') is deprecated. "
            f"Use attribute access result.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key, default)

    def keys(self) -> list:
        """Return field names (dict-compatible)."""
        return [f.name for f in dataclasses.fields(self)]

    def values(self) -> list:
        """Return field values (dict-compatible)."""
        return [getattr(self, f.name) for f in dataclasses.fields(self)]

    def items(self) -> list:
        """Return (name, value) pairs (dict-compatible)."""
        return [(f.name, getattr(self, f.name)) for f in dataclasses.fields(self)]


# ---------------------------------------------------------------------------
# Typed results for compare_scenarios() (#713)
# ---------------------------------------------------------------------------


@dataclass
class ScenarioMetrics(_DictAccessMixin):
    """Growth and survival metrics for a single scenario (insured or uninsured).

    Attributes:
        time_average_mean: Mean time-average growth rate across valid paths.
            ``-inf`` when all paths ended in bankruptcy.
        time_average_median: Median time-average growth rate.
        time_average_std: Standard deviation of time-average growth rates.
        ensemble_average: Ensemble-average growth rate.
        survival_rate: Fraction of paths that remained solvent ``[0, 1]``.
        n_survived: Number of paths that remained solvent.
    """

    time_average_mean: float
    time_average_median: float
    time_average_std: float
    ensemble_average: float
    survival_rate: float
    n_survived: int


@dataclass
class ErgodicAdvantage(_DictAccessMixin):
    """Ergodic advantage of insured over uninsured scenario.

    Attributes:
        time_average_gain: Difference in time-average growth rates
            (insured minus uninsured).
        ensemble_average_gain: Difference in ensemble-average growth rates.
        survival_gain: Difference in survival rates.
        t_statistic: Welch's t-test statistic.  ``NaN`` when insufficient data.
        p_value: Two-sided p-value.  ``NaN`` when insufficient data.
        significant: Whether the difference is statistically significant
            at the 5 % level.
    """

    time_average_gain: float
    ensemble_average_gain: float
    survival_gain: float
    t_statistic: float
    p_value: float
    significant: bool


@dataclass
class ScenarioComparison(_DictAccessMixin):
    """Typed result of :meth:`ErgodicAnalyzer.compare_scenarios`.

    Attributes:
        insured: Metrics for the insured scenario.
        uninsured: Metrics for the uninsured scenario.
        ergodic_advantage: Ergodic advantage comparison.
    """

    insured: ScenarioMetrics
    uninsured: ScenarioMetrics
    ergodic_advantage: ErgodicAdvantage


# ---------------------------------------------------------------------------
# Typed results for analyze_simulation_batch() (#713)
# ---------------------------------------------------------------------------


@dataclass
class TimeAverageStats(_DictAccessMixin):
    """Time-average growth rate statistics for a batch of simulations.

    Attributes:
        mean: Mean time-average growth rate.
        median: Median time-average growth rate.
        std: Standard deviation of time-average growth rates.
        min: Minimum time-average growth rate.
        max: Maximum time-average growth rate.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float


@dataclass
class EnsembleAverageStats(_DictAccessMixin):
    """Ensemble-average statistics for a batch of simulations.

    Attributes:
        mean: Ensemble mean growth rate.
        std: Standard deviation across paths.
        median: Ensemble median growth rate.
        survival_rate: Fraction of paths that remained solvent.
        n_survived: Number of solvent paths.
        n_total: Total number of paths.
        mean_trajectory: Mean trajectory across paths (only for ``"full"`` metric).
        std_trajectory: Std trajectory across paths (only for ``"full"`` metric).
    """

    mean: float
    std: float
    median: float
    survival_rate: float
    n_survived: int
    n_total: int
    mean_trajectory: Optional[np.ndarray] = None
    std_trajectory: Optional[np.ndarray] = None


@dataclass
class ConvergenceStats(_DictAccessMixin):
    """Convergence diagnostics for Monte Carlo time-average estimates.

    Attributes:
        converged: Whether the standard error is below the threshold.
        standard_error: Rolling standard error of the mean.
        threshold: Convergence threshold used.
    """

    converged: bool
    standard_error: float
    threshold: float


@dataclass
class SurvivalAnalysisStats(_DictAccessMixin):
    """Survival analysis for a batch of simulations.

    Attributes:
        survival_rate: Fraction of paths that remained solvent.
        mean_survival_time: Mean number of years before insolvency
            (or full horizon if solvent).
    """

    survival_rate: float
    mean_survival_time: float


@dataclass
class BatchAnalysisResults(_DictAccessMixin):
    """Typed result of :meth:`ErgodicAnalyzer.analyze_simulation_batch`.

    Attributes:
        label: Descriptive label for this batch.
        n_simulations: Number of simulations in the batch.
        time_average: Time-average growth rate statistics.
        ensemble_average: Ensemble-average statistics.
        convergence: Convergence diagnostics.
        survival_analysis: Survival analysis metrics.
        ergodic_divergence: ``time_average.mean - ensemble_average.mean``.
            ``NaN`` when no valid growth rates exist.
    """

    label: str
    n_simulations: int
    time_average: TimeAverageStats
    ensemble_average: EnsembleAverageStats
    convergence: ConvergenceStats
    survival_analysis: SurvivalAnalysisStats
    ergodic_divergence: float

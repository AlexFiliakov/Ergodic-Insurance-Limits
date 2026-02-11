"""Data containers for ergodic analysis.

Provides the standardised data types used throughout the ergodic analysis
framework: input data containers, analysis results, and validation results.

For detailed usage examples see the
`Analyzing Results tutorial <https://docs.mostlyoptimal.com/tutorials/05_analyzing_results.html>`_.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

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

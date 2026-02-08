"""Calibration parameters for optimization and decision engine heuristics.

Contains dataclass configurations that tune the financial scoring functions
used by ``BusinessOptimizer`` and ``InsuranceDecisionEngine``. These are
meta-parameters for optimizer behavior, not core business configuration.

Since:
    Version 0.9.0 (Issue #314, #458)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BusinessOptimizerConfig:
    """Calibration parameters for BusinessOptimizer financial heuristics.

    Issue #314 (C1): Consolidates all hardcoded financial multipliers from
    BusinessOptimizer into a single, documentable configuration object.

    These are simplified model parameters used by the optimizer's heuristic
    methods (_estimate_roe, _estimate_bankruptcy_risk, _estimate_growth_rate,
    etc.). They are NOT derived from manufacturer data—they are tuning knobs
    for the optimizer's internal scoring functions.
    """

    # _estimate_roe parameters
    base_roe: float = 0.15
    """Base return on equity (15%) before insurance adjustments."""
    protection_benefit_factor: float = 0.05
    """Coverage-to-assets ratio multiplier for protection benefit."""
    roe_noise_std: float = 0.1
    """Standard deviation of multiplicative noise applied to ROE."""

    # _estimate_bankruptcy_risk parameters
    base_bankruptcy_risk: float = 0.02
    """Base annual bankruptcy probability (2%)."""
    max_risk_reduction: float = 0.015
    """Maximum risk reduction from insurance coverage (1.5%)."""
    premium_burden_risk_factor: float = 0.5
    """Multiplier converting premium burden ratio to risk increase."""
    time_risk_constant: float = 20.0
    """Time constant (years) for exponential risk accumulation."""

    # _estimate_growth_rate parameters
    base_growth_rate: float = 0.10
    """Base growth rate (10%) before insurance adjustments."""
    growth_boost_factor: float = 0.03
    """Coverage ratio multiplier for growth boost (up to 3%)."""
    premium_drag_factor: float = 0.5
    """Multiplier for premium-to-revenue drag on growth."""
    asset_growth_factor: float = 0.8
    """Growth adjustment factor for asset metric."""
    equity_growth_factor: float = 1.1
    """Growth adjustment factor for equity metric."""

    # _calculate_capital_efficiency parameters
    risk_transfer_benefit_rate: float = 0.05
    """Fraction of coverage limit freed up by risk transfer (5%)."""

    # _estimate_insurance_return parameters
    risk_reduction_value: float = 0.03
    """Return contribution from risk reduction (3%)."""
    stability_value: float = 0.02
    """Return contribution from stability improvement (2%)."""
    growth_enablement_value: float = 0.03
    """Return contribution from growth enablement (3%)."""

    # _calculate_ergodic_growth parameters
    assumed_volatility: float = 0.20
    """Assumed base volatility for ergodic correction."""
    volatility_reduction_factor: float = 0.05
    """Coverage ratio multiplier for volatility reduction."""
    min_volatility: float = 0.05
    """Floor for adjusted volatility."""

    # RNG seed for reproducibility
    seed: int = 42
    """Seed for the random number generator used in Monte Carlo simulations.
    Ensures deterministic results for the same inputs."""


@dataclass
class DecisionEngineConfig:
    """Calibration parameters for InsuranceDecisionEngine heuristics.

    Issue #314 (C2): Consolidates hardcoded values from the decision engine's
    growth estimation and simulation methods.
    """

    # _estimate_growth_rate parameters
    base_growth_rate: float = 0.08
    """Base growth rate (8%) for decision engine growth estimation."""
    volatility_reduction_factor: float = 0.3
    """Coverage ratio multiplier for volatility reduction."""
    max_volatility_reduction: float = 0.15
    """Maximum volatility reduction (15%)."""
    growth_benefit_factor: float = 0.5
    """Simplified growth benefit multiplier."""

    loss_cv: float = 0.5
    """Default coefficient of variation for loss severity."""

    default_optimization_weights: Dict[str, float] = field(
        default_factory=lambda: {"growth": 0.4, "risk": 0.4, "cost": 0.2}
    )
    """Default objective function weights."""

    layer_attachment_thresholds: Tuple[float, float] = (5_000_000, 25_000_000)
    """Attachment thresholds: (primary_ceiling, first_excess_ceiling)."""

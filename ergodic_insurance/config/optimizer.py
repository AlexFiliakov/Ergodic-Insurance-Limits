"""Calibration parameters for optimization and decision engine heuristics.

Contains Pydantic configuration models that tune the financial scoring functions
used by ``BusinessOptimizer`` and ``InsuranceDecisionEngine``. These are
meta-parameters for optimizer behavior, not core business configuration.

Since:
    Version 0.9.0 (Issue #314, #458)
    Converted to Pydantic BaseModel in 0.9.x (Issue #471)
"""

from typing import Dict, Tuple

from pydantic import BaseModel, Field


class BusinessOptimizerConfig(BaseModel):
    """Calibration parameters for BusinessOptimizer financial heuristics.

    Issue #314 (C1): Consolidates all hardcoded financial multipliers from
    BusinessOptimizer into a single, documentable configuration object.

    These are simplified model parameters used by the optimizer's heuristic
    methods (_estimate_roe, _estimate_bankruptcy_risk, _estimate_growth_rate,
    etc.). They are NOT derived from manufacturer data—they are tuning knobs
    for the optimizer's internal scoring functions.
    """

    # _estimate_roe parameters
    base_roe: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Base return on equity (15%) before insurance adjustments.",
    )
    protection_benefit_factor: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Coverage-to-assets ratio multiplier for protection benefit.",
    )
    roe_noise_std: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Standard deviation of multiplicative noise applied to ROE.",
    )

    # _estimate_bankruptcy_risk parameters
    base_bankruptcy_risk: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Base annual bankruptcy probability (2%).",
    )
    max_risk_reduction: float = Field(
        default=0.015,
        ge=0,
        le=1,
        description="Maximum risk reduction from insurance coverage (1.5%).",
    )
    premium_burden_risk_factor: float = Field(
        default=0.5,
        ge=0,
        description="Multiplier converting premium burden ratio to risk increase.",
    )
    time_risk_constant: float = Field(
        default=20.0,
        gt=0,
        description="Time constant (years) for exponential risk accumulation.",
    )

    # _estimate_growth_rate parameters
    base_growth_rate: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Base growth rate (10%) before insurance adjustments.",
    )
    growth_boost_factor: float = Field(
        default=0.03,
        ge=0,
        le=1,
        description="Coverage ratio multiplier for growth boost (up to 3%).",
    )
    premium_drag_factor: float = Field(
        default=0.5,
        ge=0,
        description="Multiplier for premium-to-revenue drag on growth.",
    )
    asset_growth_factor: float = Field(
        default=0.8,
        ge=0,
        description="Growth adjustment factor for asset metric.",
    )
    equity_growth_factor: float = Field(
        default=1.1,
        ge=0,
        description="Growth adjustment factor for equity metric.",
    )

    # _calculate_capital_efficiency parameters
    risk_transfer_benefit_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Fraction of coverage limit freed up by risk transfer (5%).",
    )

    # _estimate_insurance_return parameters
    risk_reduction_value: float = Field(
        default=0.03,
        ge=0,
        le=1,
        description="Return contribution from risk reduction (3%).",
    )
    stability_value: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Return contribution from stability improvement (2%).",
    )
    growth_enablement_value: float = Field(
        default=0.03,
        ge=0,
        le=1,
        description="Return contribution from growth enablement (3%).",
    )

    # _calculate_ergodic_growth parameters
    assumed_volatility: float = Field(
        default=0.20,
        gt=0,
        le=1,
        description="Assumed base volatility for ergodic correction.",
    )
    volatility_reduction_factor: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Coverage ratio multiplier for volatility reduction.",
    )
    min_volatility: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="Floor for adjusted volatility.",
    )

    # RNG seed for reproducibility
    seed: int = Field(
        default=42,
        ge=0,
        description=(
            "Seed for the random number generator used in Monte Carlo simulations. "
            "Ensures deterministic results for the same inputs."
        ),
    )


class DecisionEngineConfig(BaseModel):
    """Calibration parameters for InsuranceDecisionEngine heuristics.

    Issue #314 (C2): Consolidates hardcoded values from the decision engine's
    growth estimation and simulation methods.
    """

    # _estimate_growth_rate parameters
    base_growth_rate: float = Field(
        default=0.08,
        ge=0,
        le=1,
        description="Base growth rate (8%) for decision engine growth estimation.",
    )
    volatility_reduction_factor: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Coverage ratio multiplier for volatility reduction.",
    )
    max_volatility_reduction: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Maximum volatility reduction (15%).",
    )
    growth_benefit_factor: float = Field(
        default=0.5,
        ge=0,
        description="Simplified growth benefit multiplier.",
    )

    loss_cv: float = Field(
        default=0.5,
        gt=0,
        description="Default coefficient of variation for loss severity.",
    )

    default_optimization_weights: Dict[str, float] = Field(
        default_factory=lambda: {"growth": 0.4, "risk": 0.4, "cost": 0.2},
        description="Default objective function weights.",
    )

    layer_attachment_thresholds: Tuple[float, float] = Field(
        default=(5_000_000, 25_000_000),
        description="Attachment thresholds: (primary_ceiling, first_excess_ceiling).",
    )

    # calculate_decision_metrics simulation parameters
    metrics_n_simulations: int = Field(
        default=1000,
        ge=10,
        description="Number of Monte Carlo simulations for decision metrics calculation.",
    )
    metrics_time_horizon: int = Field(
        default=10,
        ge=1,
        description="Time horizon in years for decision metrics simulation.",
    )
    use_crn: bool = Field(
        default=True,
        description=(
            "Use Common Random Numbers (CRN) for paired comparison between "
            "with-insurance and without-insurance simulations."
        ),
    )

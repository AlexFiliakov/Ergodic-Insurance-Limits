"""Insurance market dynamics, pricing scenarios, and cycle configuration.

Contains configuration classes for modeling external insurance market
conditions: individual pricing scenarios (soft/normal/hard markets),
Markov chain state transitions, and cycle duration dynamics.

Since:
    Version 0.9.0 (Issue #458)
"""

import logging
from typing import Dict, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class PricingScenario(BaseModel):
    """Individual market pricing scenario configuration.

    Represents a specific market condition (soft/normal/hard) with
    associated pricing parameters and market characteristics.
    """

    name: str = Field(description="Scenario name (e.g., 'Soft Market')")
    description: str = Field(description="Detailed scenario description")
    market_condition: Literal["soft", "normal", "hard"] = Field(description="Market condition type")

    # Layer-specific rates
    primary_layer_rate: float = Field(gt=0, le=0.05, description="Primary layer rate as % of limit")
    first_excess_rate: float = Field(gt=0, le=0.05, description="First excess rate as % of limit")
    higher_excess_rate: float = Field(gt=0, le=0.05, description="Higher excess rate as % of limit")

    # Market characteristics
    capacity_factor: float = Field(gt=0.5, le=2.0, description="Capacity relative to normal (1.0)")
    competition_level: Literal["low", "moderate", "high"] = Field(
        description="Level of market competition"
    )

    # Pricing factors
    retention_discount: float = Field(ge=0, le=0.5, description="Discount for higher retentions")
    volume_discount: float = Field(ge=0, le=0.5, description="Discount for large programs")
    loss_ratio_target: float = Field(gt=0, lt=1, description="Target loss ratio for insurers")
    expense_ratio: float = Field(gt=0, lt=1, description="Expense ratio for insurers")

    # Risk appetite
    new_business_appetite: Literal["restrictive", "selective", "aggressive"] = Field(
        description="Appetite for new business"
    )
    renewal_retention_focus: Literal["low", "balanced", "high"] = Field(
        description="Focus on retaining renewals"
    )
    coverage_enhancement_willingness: Literal["low", "moderate", "high"] = Field(
        description="Willingness to enhance coverage"
    )

    @model_validator(mode="after")
    def validate_rate_ordering(self) -> "PricingScenario":
        """Ensure premium rates follow expected ordering.

        Primary rates should be higher than excess rates, and first
        excess should be higher than higher excess layers.
        """
        if not self.primary_layer_rate >= self.first_excess_rate >= self.higher_excess_rate:
            raise ValueError(
                f"Rate ordering violation: primary ({self.primary_layer_rate:.3f}) >= "
                f"first_excess ({self.first_excess_rate:.3f}) >= "
                f"higher_excess ({self.higher_excess_rate:.3f}) must be maintained"
            )
        return self


class TransitionProbabilities(BaseModel):
    """Market state transition probabilities."""

    # From soft market
    soft_to_soft: float = Field(ge=0, le=1)
    soft_to_normal: float = Field(ge=0, le=1)
    soft_to_hard: float = Field(ge=0, le=1)

    # From normal market
    normal_to_soft: float = Field(ge=0, le=1)
    normal_to_normal: float = Field(ge=0, le=1)
    normal_to_hard: float = Field(ge=0, le=1)

    # From hard market
    hard_to_soft: float = Field(ge=0, le=1)
    hard_to_normal: float = Field(ge=0, le=1)
    hard_to_hard: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "TransitionProbabilities":
        """Ensure transition probabilities sum to 1.0 for each state."""
        soft_sum = self.soft_to_soft + self.soft_to_normal + self.soft_to_hard
        normal_sum = self.normal_to_soft + self.normal_to_normal + self.normal_to_hard
        hard_sum = self.hard_to_soft + self.hard_to_normal + self.hard_to_hard

        tolerance = 1e-6
        if abs(soft_sum - 1.0) > tolerance:
            raise ValueError(f"Soft market transitions sum to {soft_sum:.4f}, not 1.0")
        if abs(normal_sum - 1.0) > tolerance:
            raise ValueError(f"Normal market transitions sum to {normal_sum:.4f}, not 1.0")
        if abs(hard_sum - 1.0) > tolerance:
            raise ValueError(f"Hard market transitions sum to {hard_sum:.4f}, not 1.0")

        return self


class MarketCycles(BaseModel):
    """Market cycle configuration and dynamics."""

    average_duration_years: float = Field(gt=0, le=20)
    soft_market_duration: float = Field(gt=0, le=10)
    normal_market_duration: float = Field(gt=0, le=10)
    hard_market_duration: float = Field(gt=0, le=10)

    transition_probabilities: TransitionProbabilities = Field(
        description="Annual transition probabilities between market states"
    )

    @model_validator(mode="after")
    def validate_cycle_duration(self) -> "MarketCycles":
        """Validate that cycle durations are reasonable."""
        total_duration = (
            self.soft_market_duration + self.normal_market_duration + self.hard_market_duration
        )

        # Check if average duration is reasonable given components
        expected_avg = total_duration / 3
        if abs(self.average_duration_years - expected_avg) > expected_avg * 0.5:
            logger.warning(
                "Average duration (%.1f years) differs significantly from "
                "component average (%.1f years)",
                self.average_duration_years,
                expected_avg,
            )

        return self


class PricingScenarioConfig(BaseModel):
    """Complete pricing scenario configuration.

    Contains all market scenarios and cycle dynamics for
    insurance pricing sensitivity analysis.
    """

    scenarios: Dict[str, PricingScenario] = Field(
        description="Market scenarios (inexpensive/baseline/expensive)"
    )
    market_cycles: MarketCycles = Field(description="Market cycle dynamics and transitions")

    def get_scenario(self, scenario_name: str) -> PricingScenario:
        """Get a specific pricing scenario by name.

        Args:
            scenario_name: Name of the scenario to retrieve

        Returns:
            PricingScenario configuration

        Raises:
            KeyError: If scenario_name not found
        """
        if scenario_name not in self.scenarios:
            available = ", ".join(self.scenarios.keys())
            raise KeyError(
                f"Scenario '{scenario_name}' not found. " f"Available scenarios: {available}"
            )
        return self.scenarios[scenario_name]

    def get_rate_multiplier(self, from_scenario: str, to_scenario: str) -> float:
        """Calculate rate change multiplier between scenarios.

        Args:
            from_scenario: Starting scenario name
            to_scenario: Target scenario name

        Returns:
            Multiplier for premium rates when transitioning
        """
        from_rates = self.scenarios[from_scenario]
        to_rates = self.scenarios[to_scenario]

        # Average the rate changes across layers
        primary_mult = to_rates.primary_layer_rate / from_rates.primary_layer_rate
        excess_mult = to_rates.first_excess_rate / from_rates.first_excess_rate
        higher_mult = to_rates.higher_excess_rate / from_rates.higher_excess_rate

        return (primary_mult + excess_mult + higher_mult) / 3

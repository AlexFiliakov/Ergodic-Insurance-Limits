"""Insurance coverage, layer structure, and loss distribution configuration.

Contains configuration classes for modeling insurance programs: individual
layer definitions, multi-layer program structure, and stochastic loss
frequency/severity distributions.

Since:
    Version 0.9.0 (Issue #458)
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class InsuranceLayerConfig(BaseModel):
    """Configuration for a single insurance layer."""

    name: str = Field(description="Layer name")
    limit: float = Field(gt=0, description="Layer limit in dollars")
    attachment: float = Field(ge=0, description="Attachment point in dollars")
    base_premium_rate: float = Field(gt=0, le=1, description="Premium as percentage of limit")
    reinstatements: int = Field(default=0, ge=0, description="Number of reinstatements")
    aggregate_limit: Optional[float] = Field(
        default=None, gt=0, description="Aggregate limit if applicable"
    )
    limit_type: str = Field(
        default="per-occurrence",
        description="Type of limit: 'per-occurrence', 'aggregate', or 'hybrid'",
    )
    per_occurrence_limit: Optional[float] = Field(
        default=None, gt=0, description="Per-occurrence limit for hybrid type"
    )

    @model_validator(mode="after")
    def validate_layer_structure(self):
        """Ensure layer structure is valid.

        Returns:
            Validated layer config.

        Raises:
            ValueError: If layer structure is invalid.
        """
        # Validate limit type
        valid_limit_types = ["per-occurrence", "aggregate", "hybrid"]
        if self.limit_type not in valid_limit_types:
            raise ValueError(
                f"Invalid limit_type: {self.limit_type}. Must be one of {valid_limit_types}"
            )

        # Validate based on limit type
        if self.limit_type == "hybrid":
            # For hybrid, need both per-occurrence and aggregate limits
            if self.per_occurrence_limit is None and self.aggregate_limit is None:
                raise ValueError(
                    "Hybrid limit type requires both per_occurrence_limit and aggregate_limit to be set"
                )

        return self


class InsuranceConfig(BaseModel):
    """Enhanced insurance configuration."""

    enabled: bool = Field(default=True, description="Whether insurance is enabled")
    layers: List[InsuranceLayerConfig] = Field(default_factory=list, description="Insurance layers")
    deductible: float = Field(default=0, ge=0, description="Deductible amount")
    coinsurance: float = Field(default=1.0, gt=0, le=1, description="Coinsurance percentage")
    waiting_period_days: int = Field(default=0, ge=0, description="Waiting period for claims")
    claims_handling_cost: float = Field(
        default=0.05, ge=0, le=1, description="Claims handling cost as percentage"
    )

    @model_validator(mode="after")
    def validate_layers(self):
        """Ensure layers don't overlap and are properly ordered.

        Returns:
            Validated insurance config.

        Raises:
            ValueError: If layers overlap or are misordered.
        """
        if not self.layers:
            return self

        # Sort layers by attachment point
        sorted_layers = sorted(self.layers, key=lambda x: x.attachment)

        for i in range(len(sorted_layers) - 1):
            current = sorted_layers[i]
            next_layer = sorted_layers[i + 1]

            # Check for gaps or overlaps
            if current.attachment + current.limit < next_layer.attachment:
                print(f"Warning: Gap between layers {current.name} and {next_layer.name}")
            elif current.attachment + current.limit > next_layer.attachment:
                raise ValueError(f"Layers {current.name} and {next_layer.name} overlap")

        return self


class LossDistributionConfig(BaseModel):
    """Configuration for loss distributions."""

    frequency_distribution: str = Field(
        default="poisson", description="Frequency distribution type"
    )
    frequency_annual: float = Field(gt=0, description="Annual expected frequency")
    severity_distribution: str = Field(
        default="lognormal", description="Severity distribution type"
    )
    severity_mean: float = Field(gt=0, description="Mean severity")
    severity_std: float = Field(gt=0, description="Severity standard deviation")
    correlation_factor: float = Field(
        default=0.0, ge=-1, le=1, description="Correlation between frequency and severity"
    )
    tail_alpha: float = Field(default=2.0, gt=1, description="Tail heaviness parameter")

    @field_validator("frequency_distribution")
    @classmethod
    def validate_frequency_dist(cls, v: str) -> str:
        """Validate frequency distribution type.

        Args:
            v: Distribution type.

        Returns:
            Validated distribution type.

        Raises:
            ValueError: If distribution type is invalid.
        """
        valid_dists = ["poisson", "negative_binomial", "binomial"]
        if v not in valid_dists:
            raise ValueError(f"Invalid frequency distribution: {v}. Must be one of {valid_dists}")
        return v

    @field_validator("severity_distribution")
    @classmethod
    def validate_severity_dist(cls, v: str) -> str:
        """Validate severity distribution type.

        Args:
            v: Distribution type.

        Returns:
            Validated distribution type.

        Raises:
            ValueError: If distribution type is invalid.
        """
        valid_dists = ["lognormal", "gamma", "pareto", "weibull"]
        if v not in valid_dists:
            raise ValueError(f"Invalid severity distribution: {v}. Must be one of {valid_dists}")
        return v

"""Insurance policy structure and claim processing.

This module provides classes for modeling multi-layer insurance policies
with configurable attachment points, limits, and premium rates.

Note: For advanced features like reinstatements and complex multi-layer programs,
see the insurance_program module which provides EnhancedInsuranceLayer and
InsuranceProgram classes.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
import warnings

import yaml

if TYPE_CHECKING:
    from .insurance_pricing import InsurancePricer, MarketCycle
    from .insurance_program import InsuranceProgram
    from .loss_distributions import ManufacturingLossGenerator


@dataclass
class InsuranceLayer:
    """Represents a single insurance layer.

    Each layer has an attachment point (where coverage starts),
    a limit (maximum coverage), and a rate (premium percentage).
    """

    attachment_point: float  # Where this layer starts covering
    limit: float  # Maximum coverage from this layer
    rate: float  # Premium rate as percentage of limit

    def __post_init__(self):
        """Validate insurance layer parameters."""
        if self.attachment_point < 0:
            raise ValueError(f"Attachment point must be non-negative, got {self.attachment_point}")
        if self.limit <= 0:
            raise ValueError(f"Limit must be positive, got {self.limit}")
        if self.rate < 0:
            raise ValueError(f"Premium rate must be non-negative, got {self.rate}")

    def calculate_recovery(self, loss_amount: float) -> float:
        """Calculate recovery from this layer for a given loss.

        Args:
            loss_amount: Total loss amount to recover.

        Returns:
            Amount recovered from this layer.
        """
        if loss_amount <= self.attachment_point:
            return 0.0

        # Amount above attachment point, capped at layer limit
        excess_loss = loss_amount - self.attachment_point
        recovery = min(excess_loss, self.limit)

        return recovery

    def calculate_premium(self) -> float:
        """Calculate premium for this layer.

        Returns:
            Premium amount (rate × limit).
        """
        return self.limit * self.rate


class InsurancePolicy:
    """Multi-layer insurance policy with deductible.

    Manages multiple insurance layers and processes claims across them.
    """

    def __init__(
        self,
        layers: List[InsuranceLayer],
        deductible: float = 0.0,
        pricing_enabled: bool = False,
        pricer: Optional["InsurancePricer"] = None,
    ):
        """Initialize insurance policy.

        Args:
            layers: List of insurance layers in order of attachment.
            deductible: Amount paid by insured before insurance kicks in.
            pricing_enabled: Whether to use dynamic pricing.
            pricer: Optional InsurancePricer for dynamic pricing.
        """
        self.layers = sorted(layers, key=lambda x: x.attachment_point)
        self.deductible = deductible
        self.pricing_enabled = pricing_enabled
        self.pricer = pricer
        self.pricing_results: List[Any] = []

    def process_claim(self, claim_amount: float) -> Tuple[float, float]:
        """Process a claim through the insurance structure.

        Args:
            claim_amount: Total claim amount.

        Returns:
            Tuple of (company_payment, insurance_recovery).
        """
        if claim_amount <= 0:
            return 0.0, 0.0

        # Company pays the deductible
        company_payment = min(claim_amount, self.deductible)
        remaining_loss = claim_amount - company_payment

        # Process through insurance layers
        insurance_recovery = 0.0
        for layer in self.layers:
            if remaining_loss <= 0:
                break

            # Calculate recovery from this layer
            layer_recovery = layer.calculate_recovery(claim_amount)
            insurance_recovery += layer_recovery

        # If insurance doesn't cover everything, company pays the excess
        total_covered = company_payment + insurance_recovery
        if total_covered < claim_amount:
            excess = claim_amount - total_covered
            company_payment += excess

        return company_payment, insurance_recovery

    def calculate_recovery(self, claim_amount: float) -> float:
        """Calculate total insurance recovery for a claim.

        Args:
            claim_amount: Total claim amount.

        Returns:
            Total insurance recovery amount.
        """
        if claim_amount <= 0:
            return 0.0

        # Amount available for insurance after deductible
        if claim_amount <= self.deductible:
            return 0.0

        # Process through insurance layers
        insurance_recovery = 0.0
        for layer in self.layers:
            # Calculate recovery from this layer
            layer_recovery = layer.calculate_recovery(claim_amount)
            insurance_recovery += layer_recovery

        return insurance_recovery

    def calculate_premium(self) -> float:
        """Calculate total premium across all layers.

        Returns:
            Total annual premium.
        """
        return sum(layer.calculate_premium() for layer in self.layers)

    @classmethod
    def from_yaml(cls, config_path: str) -> "InsurancePolicy":
        """Load insurance policy from YAML configuration.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            InsurancePolicy configured from YAML.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract deductible
        deductible = config.get("deductible", 0.0)

        # Create layers
        layers = []
        for layer_config in config.get("layers", []):
            layer = InsuranceLayer(
                attachment_point=layer_config["attachment_point"],
                limit=layer_config["limit"],
                rate=layer_config["rate"],
            )
            layers.append(layer)

        return cls(layers=layers, deductible=deductible)

    def get_total_coverage(self) -> float:
        """Get total coverage across all layers.

        Returns:
            Maximum possible insurance coverage.
        """
        if not self.layers:
            return 0.0

        # Find the highest attachment point + limit
        max_coverage = 0.0
        for layer in self.layers:
            layer_top = layer.attachment_point + layer.limit
            max_coverage = max(max_coverage, layer_top)

        return max_coverage - self.deductible

    def to_enhanced_program(self) -> Optional["InsuranceProgram"]:
        """Convert to enhanced InsuranceProgram for advanced features.

        Returns:
            InsuranceProgram instance with same configuration.
        """
        try:
            from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram

            enhanced_layers = [
                EnhancedInsuranceLayer(
                    attachment_point=layer.attachment_point,
                    limit=layer.limit,
                    premium_rate=layer.rate,
                    reinstatements=0,  # Default no reinstatements
                )
                for layer in self.layers
            ]

            return InsuranceProgram(
                layers=enhanced_layers,
                deductible=self.deductible,
                name="Converted Insurance Policy",
            )
        except ImportError:
            warnings.warn(
                "Enhanced insurance_program module not available. "
                "Install with advanced features for reinstatement support.",
                UserWarning,
            )
            return None

    def apply_pricing(
        self,
        expected_revenue: float,
        market_cycle: Optional["MarketCycle"] = None,
        loss_generator: Optional["ManufacturingLossGenerator"] = None,
    ) -> None:
        """Apply dynamic pricing to all layers in the policy.

        Updates layer rates based on frequency/severity calculations.

        Args:
            expected_revenue: Expected annual revenue for scaling
            market_cycle: Optional market cycle state
            loss_generator: Optional loss generator (uses pricer's if not provided)

        Raises:
            ValueError: If pricing not enabled or pricer not configured
        """
        if not self.pricing_enabled:
            raise ValueError("Pricing not enabled for this policy")

        if self.pricer is None:
            if loss_generator is None:
                raise ValueError("Either pricer or loss_generator must be provided")

            # Create a default pricer
            from .insurance_pricing import InsurancePricer, MarketCycle

            self.pricer = InsurancePricer(
                loss_generator=loss_generator,
                market_cycle=market_cycle or MarketCycle.NORMAL,
            )

        # Apply pricing to the policy
        self.pricer.price_insurance_policy(
            policy=self,
            expected_revenue=expected_revenue,
            market_cycle=market_cycle,
            update_policy=True,
        )

    @classmethod
    def create_with_pricing(
        cls,
        layers: List[InsuranceLayer],
        loss_generator: "ManufacturingLossGenerator",
        expected_revenue: float,
        market_cycle: Optional["MarketCycle"] = None,
        deductible: float = 0.0,
    ) -> "InsurancePolicy":
        """Create insurance policy with dynamic pricing.

        Factory method that creates a policy with pricing already applied.

        Args:
            layers: Initial layer structure
            loss_generator: Loss generator for pricing
            expected_revenue: Expected annual revenue
            market_cycle: Market cycle state
            deductible: Self-insured retention

        Returns:
            InsurancePolicy with pricing applied
        """
        from .insurance_pricing import InsurancePricer, MarketCycle

        # Create pricer
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=market_cycle or MarketCycle.NORMAL,
        )

        # Create policy with pricing enabled
        policy = cls(
            layers=layers,
            deductible=deductible,
            pricing_enabled=True,
            pricer=pricer,
        )

        # Apply pricing
        policy.apply_pricing(expected_revenue, market_cycle)

        return policy

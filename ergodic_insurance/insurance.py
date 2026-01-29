"""Insurance policy structure and claim processing.

This module provides classes for modeling multi-layer insurance policies
with configurable attachment points, limits, and premium rates. It supports
complex insurance structures commonly used in commercial insurance including
excess layers, umbrella coverage, and multi-layer towers.

The module integrates with pricing engines for dynamic premium calculation
and supports both static and market-driven pricing models.

Key Features:
    - Multi-layer insurance towers with attachment points and limits
    - Deductible and self-insured retention handling
    - Dynamic pricing integration with market cycles
    - Claim allocation across multiple layers
    - Premium calculation with various rating methods

Examples:
    Simple single-layer policy::

        from ergodic_insurance.insurance import InsurancePolicy, InsuranceLayer

        # $5M excess $1M with 3% rate
        layer = InsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            rate=0.03
        )

        policy = InsurancePolicy(
            layers=[layer],
            deductible=500_000
        )

        # Process a $3M claim
        company_payment, insurance_recovery = policy.process_claim(3_000_000)

    Multi-layer tower::

        # Build a insurance tower
        layers = [
            InsuranceLayer(1_000_000, 4_000_000, 0.025),  # Primary
            InsuranceLayer(5_000_000, 5_000_000, 0.015),  # First excess
            InsuranceLayer(10_000_000, 10_000_000, 0.01), # Second excess
        ]

        tower = InsurancePolicy(layers, deductible=1_000_000)
        annual_premium = tower.calculate_premium()

Note:
    For advanced features like reinstatements and complex multi-layer programs,
    see the insurance_program module which provides EnhancedInsuranceLayer and
    InsuranceProgram classes.

Since:
    Version 0.1.0
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
    Insurance layers are the building blocks of complex insurance programs.

    Attributes:
        attachment_point: Dollar amount where this layer starts providing
            coverage. Also known as the retention or excess point.
        limit: Maximum coverage amount from this layer. The layer covers
            losses from attachment_point to (attachment_point + limit).
        rate: Premium rate as a percentage of the limit. For example,
            0.03 means 3% of limit as annual premium.

    Examples:
        Primary layer with $1M retention::

            primary = InsuranceLayer(
                attachment_point=1_000_000,  # $1M retention
                limit=5_000_000,             # $5M limit
                rate=0.025                   # 2.5% rate
            )

            # This covers losses from $1M to $6M
            # Annual premium = $5M × 2.5% = $125,000

        Excess layer in a tower::

            excess = InsuranceLayer(
                attachment_point=6_000_000,  # Attaches at $6M
                limit=10_000_000,            # $10M limit
                rate=0.01                    # 1% rate (lower for excess)
            )

    Note:
        Layers are typically structured in towers with each successive
        layer attaching where the previous layer exhausts.
    """

    attachment_point: float  # Where this layer starts covering
    limit: float  # Maximum coverage from this layer
    rate: float  # Premium rate as percentage of limit

    def __post_init__(self):
        """Validate insurance layer parameters.

        Raises:
            ValueError: If attachment_point is negative, limit is non-positive,
                or rate is negative.
        """
        if self.attachment_point < 0:
            raise ValueError(f"Attachment point must be non-negative, got {self.attachment_point}")
        if self.limit <= 0:
            raise ValueError(f"Limit must be positive, got {self.limit}")
        if self.rate < 0:
            raise ValueError(f"Premium rate must be non-negative, got {self.rate}")

    def calculate_recovery(self, loss_amount: float) -> float:
        """Calculate recovery from this layer for a given loss.

        Determines how much of a loss is covered by this specific layer
        based on its attachment point and limit.

        Args:
            loss_amount: Total loss amount in dollars to recover.

        Returns:
            float: Amount recovered from this layer in dollars. Returns 0
                if loss is below attachment point, partial recovery if loss
                partially penetrates layer, or full limit if loss exceeds
                layer exhaust point.

        Examples:
            Layer with $1M attachment, $5M limit::

                layer = InsuranceLayer(1_000_000, 5_000_000, 0.02)

                # Loss below attachment
                recovery = layer.calculate_recovery(500_000)  # Returns 0

                # Loss partially in layer
                recovery = layer.calculate_recovery(3_000_000)  # Returns 2M

                # Loss exceeds layer
                recovery = layer.calculate_recovery(10_000_000)  # Returns 5M (full limit)
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
            float: Annual premium amount in dollars (rate × limit).

        Examples:
            Calculate annual cost::

                layer = InsuranceLayer(1_000_000, 10_000_000, 0.015)
                premium = layer.calculate_premium()  # Returns 150,000
                print(f"Annual premium: ${premium:,.0f}")
        """
        return self.limit * self.rate


class InsurancePolicy:
    """Multi-layer insurance policy with deductible.

    Manages multiple insurance layers and processes claims across them,
    handling proper allocation of losses to each layer in sequence.
    Supports both static and dynamic pricing models.

    The policy structure follows standard commercial insurance practices:
    1. Insured pays deductible first
    2. Losses then penetrate layers in order of attachment
    3. Each layer pays up to its limit
    4. Insured bears losses exceeding all coverage

    Attributes:
        layers: List of InsuranceLayer objects sorted by attachment point.
        deductible: Self-insured retention before insurance applies.
        pricing_enabled: Whether to use dynamic pricing models.
        pricer: Optional pricing engine for market-based premiums.
        pricing_results: History of pricing calculations.

    Examples:
        Standard commercial property program::

            # Build insurance program
            policy = InsurancePolicy(
                layers=[
                    InsuranceLayer(500_000, 4_500_000, 0.03),   # Primary
                    InsuranceLayer(5_000_000, 10_000_000, 0.02), # Excess
                    InsuranceLayer(15_000_000, 25_000_000, 0.01) # Umbrella
                ],
                deductible=500_000  # $500K SIR
            )

            # Process various claims
            small_claim = policy.process_claim(100_000)  # All on deductible
            medium_claim = policy.process_claim(3_000_000)  # Hits primary
            large_claim = policy.process_claim(20_000_000)  # Multiple layers

    Note:
        Layers are automatically sorted by attachment point to ensure
        proper claim allocation regardless of input order.
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
            layers: List of InsuranceLayer objects defining the coverage tower.
                Layers will be automatically sorted by attachment point.
            deductible: Amount in dollars paid by insured before insurance
                applies. Also known as self-insured retention (SIR).
                Default is 0.
            pricing_enabled: Whether to use dynamic pricing models that
                adjust premiums based on market conditions. Default is False.
            pricer: Optional InsurancePricer instance for dynamic pricing.
                Required if pricing_enabled is True.

        Examples:
            Create policy with dynamic pricing::

                from ergodic_insurance.insurance_pricing import InsurancePricer

                pricer = InsurancePricer(base_rate=0.02)

                policy = InsurancePolicy(
                    layers=[layer1, layer2],
                    deductible=1_000_000,
                    pricing_enabled=True,
                    pricer=pricer
                )
        """
        self.layers = sorted(layers, key=lambda x: x.attachment_point)
        self.deductible = deductible
        self.pricing_enabled = pricing_enabled
        self.pricer = pricer
        self.pricing_results: List[Any] = []

    def process_claim(self, claim_amount: float) -> Tuple[float, float]:
        """Process a claim through the insurance structure.

        Allocates a loss across the deductible and insurance layers,
        calculating how much is paid by the company versus insurance.
        Total insurance recovery is capped at (claim_amount - deductible)
        to prevent over-recovery when layer configurations overlap with
        the deductible region.

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
            remaining_loss -= layer_recovery

        # Guard: total insurance recovery cannot exceed (claim - deductible)
        max_recoverable = claim_amount - min(claim_amount, self.deductible)
        insurance_recovery = min(insurance_recovery, max_recoverable)

        # If insurance doesn't cover everything, company pays the excess
        total_covered = company_payment + insurance_recovery
        if total_covered < claim_amount:
            excess = claim_amount - total_covered
            company_payment += excess

        return company_payment, insurance_recovery

    def calculate_recovery(self, claim_amount: float) -> float:
        """Calculate total insurance recovery for a claim.

        Recovery is capped at (claim_amount - deductible) to prevent
        over-recovery when layer configurations overlap with the
        deductible region.

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

        # Guard: total recovery cannot exceed (claim - deductible)
        max_recoverable = claim_amount - self.deductible
        insurance_recovery = min(insurance_recovery, max_recoverable)

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
                    base_premium_rate=layer.rate,
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

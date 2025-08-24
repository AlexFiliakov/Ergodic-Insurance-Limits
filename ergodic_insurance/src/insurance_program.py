"""Multi-layer insurance program with reinstatements and advanced features.

This module provides comprehensive insurance program management including
multi-layer structures, reinstatements, attachment points, and accurate
loss allocation for manufacturing risk transfer optimization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


class ReinstatementType(Enum):
    """Types of reinstatement provisions."""

    NONE = "none"
    PRO_RATA = "pro_rata"  # Premium based on time remaining
    FULL = "full"  # Full premium regardless of timing
    FREE = "free"  # No additional premium


@dataclass
class EnhancedInsuranceLayer:
    """Insurance layer with reinstatement support and advanced features.

    Extends basic layer functionality with reinstatements, tracking,
    and more sophisticated premium calculations.
    """

    attachment_point: float  # Where coverage starts
    limit: float  # Maximum coverage amount per occurrence
    premium_rate: float  # % of limit as base premium
    reinstatements: int = 0  # Number of reinstatements available
    reinstatement_premium: float = 1.0  # % of original premium per reinstatement
    reinstatement_type: ReinstatementType = ReinstatementType.PRO_RATA
    aggregate_limit: Optional[float] = None  # Annual aggregate limit if applicable

    def __post_init__(self):
        """Validate layer parameters."""
        if self.attachment_point < 0:
            raise ValueError(f"Attachment point must be non-negative, got {self.attachment_point}")
        if self.limit <= 0:
            raise ValueError(f"Limit must be positive, got {self.limit}")
        if self.premium_rate < 0:
            raise ValueError(f"Premium rate must be non-negative, got {self.premium_rate}")
        if self.reinstatements < 0:
            raise ValueError(f"Reinstatements must be non-negative, got {self.reinstatements}")
        if self.reinstatement_premium < 0:
            raise ValueError(
                f"Reinstatement premium must be non-negative, got {self.reinstatement_premium}"
            )
        if self.aggregate_limit is not None and self.aggregate_limit <= 0:
            raise ValueError(f"Aggregate limit must be positive if set, got {self.aggregate_limit}")

    def calculate_base_premium(self) -> float:
        """Calculate base premium for this layer.

        Returns:
            Base premium amount (rate × limit).
        """
        return self.limit * self.premium_rate

    def calculate_reinstatement_premium(self, timing_factor: float = 1.0) -> float:
        """Calculate premium for a single reinstatement.

        Args:
            timing_factor: Pro-rata factor based on policy period remaining (0-1).

        Returns:
            Reinstatement premium amount.
        """
        base_premium = self.calculate_base_premium()

        if self.reinstatement_type == ReinstatementType.FREE:
            return 0.0
        elif self.reinstatement_type == ReinstatementType.FULL:
            return base_premium * self.reinstatement_premium
        elif self.reinstatement_type == ReinstatementType.PRO_RATA:
            return base_premium * self.reinstatement_premium * timing_factor
        else:
            return 0.0

    def can_respond(self, loss_amount: float) -> bool:
        """Check if this layer can respond to a loss.

        Args:
            loss_amount: Total loss amount.

        Returns:
            True if loss exceeds attachment point.
        """
        return loss_amount > self.attachment_point

    def calculate_layer_loss(self, total_loss: float) -> float:
        """Calculate the portion of loss hitting this layer.

        Args:
            total_loss: Total loss amount.

        Returns:
            Amount of loss allocated to this layer (before applying limits).
        """
        if total_loss <= self.attachment_point:
            return 0.0

        excess_loss = total_loss - self.attachment_point
        return min(excess_loss, self.limit)


@dataclass
class LayerState:
    """Tracks the current state of an insurance layer during simulation.

    Maintains utilization, reinstatement count, and exhaustion status
    for accurate multi-claim processing.
    """

    layer: EnhancedInsuranceLayer
    current_limit: float = field(init=False)
    used_limit: float = 0.0
    reinstatements_used: int = 0
    total_claims_paid: float = 0.0
    reinstatement_premiums_paid: float = 0.0
    is_exhausted: bool = False
    aggregate_used: float = 0.0  # For annual aggregate tracking

    def __post_init__(self):
        """Initialize current limit to layer's base limit."""
        self.current_limit = self.layer.limit

    def process_claim(self, claim_amount: float, timing_factor: float = 1.0) -> Tuple[float, float]:
        """Process a claim against this layer.

        Args:
            claim_amount: Amount of loss allocated to this layer.
            timing_factor: Pro-rata factor for reinstatement premium.

        Returns:
            Tuple of (amount_paid, reinstatement_premium).
        """
        if self.is_exhausted or claim_amount <= 0:
            return 0.0, 0.0

        total_payment = 0.0
        total_reinstatement_premium = 0.0
        remaining_claim = claim_amount

        # Process claim with current limit (may trigger reinstatements)
        while remaining_claim > 0 and not self.is_exhausted:
            # Check aggregate limit
            available_limit = self.current_limit
            if self.layer.aggregate_limit is not None:
                remaining_aggregate = self.layer.aggregate_limit - self.aggregate_used
                if remaining_aggregate <= 0:
                    self.is_exhausted = True
                    break
                available_limit = min(available_limit, remaining_aggregate)

            # Calculate payment from available limit
            payment = min(remaining_claim, available_limit)
            if payment > 0:
                self.used_limit += payment
                self.current_limit -= payment
                self.total_claims_paid += payment
                total_payment += payment
                remaining_claim -= payment

                # Update aggregate if applicable
                if self.layer.aggregate_limit is not None:
                    self.aggregate_used += payment
                    if self.aggregate_used >= self.layer.aggregate_limit:
                        self.is_exhausted = True
                        self.current_limit = 0.0
                        break

            # Check if limit exhausted and can reinstate
            if self.current_limit == 0 and not self.is_exhausted:
                if self.reinstatements_used < self.layer.reinstatements:
                    # Trigger reinstatement
                    self.reinstatements_used += 1
                    self.current_limit = self.layer.limit
                    reinstatement_premium = self.layer.calculate_reinstatement_premium(
                        timing_factor
                    )
                    self.reinstatement_premiums_paid += reinstatement_premium
                    total_reinstatement_premium += reinstatement_premium
                else:
                    # No more reinstatements available
                    self.is_exhausted = True
                    break
            else:
                # Can't process more of this claim
                break

        return total_payment, total_reinstatement_premium

    def reset(self):
        """Reset layer state for new policy period."""
        self.current_limit = self.layer.limit
        self.used_limit = 0.0
        self.reinstatements_used = 0
        self.total_claims_paid = 0.0
        self.reinstatement_premiums_paid = 0.0
        self.is_exhausted = False
        self.aggregate_used = 0.0

    def get_available_limit(self) -> float:
        """Get currently available limit.

        Returns:
            Available limit for claims.
        """
        return self.current_limit if not self.is_exhausted else 0.0

    def get_utilization_rate(self) -> float:
        """Calculate layer utilization rate.

        Returns:
            Utilization as percentage of total available limit.
        """
        total_available = self.layer.limit * (1 + self.layer.reinstatements)
        if total_available == 0:
            return 0.0
        return self.total_claims_paid / total_available


class InsuranceProgram:
    """Comprehensive multi-layer insurance program manager.

    Handles complex insurance structures with multiple layers,
    reinstatements, and sophisticated claim allocation.
    """

    def __init__(
        self,
        layers: List[EnhancedInsuranceLayer],
        deductible: float = 0.0,
        name: str = "Manufacturing Insurance Program",
    ):
        """Initialize insurance program.

        Args:
            layers: List of insurance layers (will be sorted by attachment).
            deductible: Self-insured retention before insurance.
            name: Program identifier.
        """
        self.layers = sorted(layers, key=lambda x: x.attachment_point)
        self.deductible = deductible
        self.name = name
        self.layer_states = [LayerState(layer) for layer in self.layers]
        self.total_premiums_paid = 0.0
        self.total_claims = 0

    def calculate_annual_premium(self) -> float:
        """Calculate total annual premium for the program.

        Returns:
            Total base premium across all layers.
        """
        return sum(layer.calculate_base_premium() for layer in self.layers)

    def process_claim(self, claim_amount: float, timing_factor: float = 1.0) -> Dict[str, Any]:
        """Process a single claim through the insurance structure.

        Args:
            claim_amount: Total claim amount.
            timing_factor: Pro-rata factor for reinstatement premiums.

        Returns:
            Dictionary with claim allocation details.
        """
        if claim_amount <= 0:
            return {
                "total_claim": 0.0,
                "deductible_paid": 0.0,
                "insurance_recovery": 0.0,
                "uncovered_loss": 0.0,
                "reinstatement_premiums": 0.0,
                "layers_triggered": [],
            }

        self.total_claims += 1
        result = {
            "total_claim": claim_amount,
            "deductible_paid": min(claim_amount, self.deductible),
            "insurance_recovery": 0.0,
            "uncovered_loss": 0.0,
            "reinstatement_premiums": 0.0,
            "layers_triggered": [],
        }

        # Process through each layer
        for i, state in enumerate(self.layer_states):
            if not state.layer.can_respond(claim_amount):
                continue

            # Calculate loss to this layer
            layer_loss = state.layer.calculate_layer_loss(claim_amount)

            # Process the claim
            payment, reinstatement_premium = state.process_claim(layer_loss, timing_factor)

            if payment > 0:
                result["insurance_recovery"] += payment
                result["reinstatement_premiums"] += reinstatement_premium
                result["layers_triggered"].append(
                    {
                        "layer_index": i,
                        "attachment": state.layer.attachment_point,
                        "payment": payment,
                        "reinstatement_premium": reinstatement_premium,
                        "exhausted": state.is_exhausted,
                    }
                )

        # Calculate uncovered loss
        total_covered = result["deductible_paid"] + result["insurance_recovery"]
        if total_covered < claim_amount:
            result["uncovered_loss"] = claim_amount - total_covered
            # Company pays uncovered portion
            result["deductible_paid"] += result["uncovered_loss"]

        self.total_premiums_paid += result["reinstatement_premiums"]

        return result

    def process_annual_claims(
        self, claims: List[float], claim_times: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Process all claims for a policy year.

        Args:
            claims: List of claim amounts.
            claim_times: Optional list of claim times (0-1 for year fraction).

        Returns:
            Dictionary with annual summary statistics.
        """
        if claim_times is None:
            # Assume uniform distribution through year
            claim_times = np.linspace(0, 1, len(claims)) if claims else []

        results = {
            "total_claims": len(claims),
            "total_losses": sum(claims),
            "total_deductible": 0.0,
            "total_recovery": 0.0,
            "total_uncovered": 0.0,
            "total_reinstatement_premiums": 0.0,
            "base_premium": self.calculate_annual_premium(),
            "claim_details": [],
            "layer_summaries": [],
        }

        # Process each claim
        for claim, time in zip(claims, claim_times):
            timing_factor = 1.0 - time  # Remaining portion of year
            claim_result = self.process_claim(claim, timing_factor)

            results["total_deductible"] += claim_result["deductible_paid"]
            results["total_recovery"] += claim_result["insurance_recovery"]
            results["total_uncovered"] += claim_result["uncovered_loss"]
            results["total_reinstatement_premiums"] += claim_result["reinstatement_premiums"]
            results["claim_details"].append(claim_result)

        # Compile layer summaries
        for i, state in enumerate(self.layer_states):
            results["layer_summaries"].append(
                {
                    "layer_index": i,
                    "attachment_point": state.layer.attachment_point,
                    "limit": state.layer.limit,
                    "claims_paid": state.total_claims_paid,
                    "reinstatements_used": state.reinstatements_used,
                    "reinstatement_premiums": state.reinstatement_premiums_paid,
                    "utilization_rate": state.get_utilization_rate(),
                    "is_exhausted": state.is_exhausted,
                }
            )

        results["total_premium_paid"] = (
            results["base_premium"] + results["total_reinstatement_premiums"]
        )
        results["net_benefit"] = results["total_recovery"] - results["total_premium_paid"]

        return results

    def reset_annual(self):
        """Reset program state for new policy year."""
        for state in self.layer_states:
            state.reset()
        self.total_claims = 0

    def get_program_summary(self) -> Dict[str, any]:
        """Get current program state summary.

        Returns:
            Dictionary with program statistics.
        """
        return {
            "program_name": self.name,
            "deductible": self.deductible,
            "num_layers": len(self.layers),
            "total_coverage": self.get_total_coverage(),
            "annual_base_premium": self.calculate_annual_premium(),
            "total_claims_processed": self.total_claims,
            "total_premiums_paid": self.total_premiums_paid,
            "layers": [
                {
                    "attachment": layer.attachment_point,
                    "limit": layer.limit,
                    "exhaustion_point": layer.attachment_point + layer.limit,
                    "reinstatements": layer.reinstatements,
                    "base_premium": layer.calculate_base_premium(),
                }
                for layer in self.layers
            ],
        }

    def get_total_coverage(self) -> float:
        """Calculate maximum possible coverage.

        Returns:
            Maximum claim amount that can be covered.
        """
        if not self.layers:
            return 0.0

        # Find highest exhaustion point
        max_coverage = max(layer.attachment_point + layer.limit for layer in self.layers)

        return max_coverage

    @classmethod
    def from_yaml(cls, config_path: str) -> "InsuranceProgram":
        """Load insurance program from YAML configuration.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Configured InsuranceProgram instance.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract program parameters
        deductible = config.get("deductible", 0.0)
        name = config.get("program_name", "Insurance Program")

        # Create layers
        layers = []
        for layer_config in config.get("layers", []):
            # Parse reinstatement type
            reinstatement_type_str = layer_config.get("reinstatement_type", "pro_rata")
            reinstatement_type = ReinstatementType(reinstatement_type_str)

            layer = EnhancedInsuranceLayer(
                attachment_point=layer_config["attachment_point"],
                limit=layer_config["limit"],
                premium_rate=layer_config.get("premium_rate", layer_config.get("rate", 0.01)),
                reinstatements=layer_config.get("reinstatements", 0),
                reinstatement_premium=layer_config.get("reinstatement_premium", 1.0),
                reinstatement_type=reinstatement_type,
                aggregate_limit=layer_config.get("aggregate_limit"),
            )
            layers.append(layer)

        return cls(layers=layers, deductible=deductible, name=name)

    @classmethod
    def create_standard_manufacturing_program(
        cls, deductible: float = 250_000
    ) -> "InsuranceProgram":
        """Create standard manufacturing insurance program.

        Args:
            deductible: Self-insured retention amount.

        Returns:
            Standard manufacturing insurance program.
        """
        layers = [
            # Primary Layer
            EnhancedInsuranceLayer(
                attachment_point=deductible,
                limit=5_000_000 - deductible,
                premium_rate=0.015,  # 1.5% rate
                reinstatements=0,
            ),
            # First Excess
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=20_000_000,
                premium_rate=0.008,  # 0.8% rate
                reinstatements=1,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.FULL,
            ),
            # Second Excess
            EnhancedInsuranceLayer(
                attachment_point=25_000_000,
                limit=25_000_000,
                premium_rate=0.004,  # 0.4% rate
                reinstatements=2,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.PRO_RATA,
            ),
            # Third Excess
            EnhancedInsuranceLayer(
                attachment_point=50_000_000,
                limit=50_000_000,
                premium_rate=0.002,  # 0.2% rate
                reinstatements=999,  # Effectively unlimited
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.PRO_RATA,
            ),
        ]

        return cls(layers=layers, deductible=deductible, name="Standard Manufacturing Program")


@dataclass
class ProgramState:
    """Tracks multi-year insurance program state for simulations.

    Maintains historical data and statistics across multiple
    policy periods for long-term analysis.
    """

    program: InsuranceProgram
    years_simulated: int = 0
    total_claims: List[float] = field(default_factory=list)
    total_recoveries: List[float] = field(default_factory=list)
    total_premiums: List[float] = field(default_factory=list)
    annual_results: List[Dict] = field(default_factory=list)

    def simulate_year(
        self, annual_claims: List[float], claim_times: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Simulate one year of the insurance program.

        Args:
            annual_claims: List of claims for the year.
            claim_times: Optional timing of claims.

        Returns:
            Annual results dictionary.
        """
        # Reset for new year
        self.program.reset_annual()

        # Process claims
        results = self.program.process_annual_claims(annual_claims, claim_times)

        # Track statistics
        self.years_simulated += 1
        self.total_claims.append(sum(annual_claims))
        self.total_recoveries.append(results["total_recovery"])
        self.total_premiums.append(results["total_premium_paid"])
        self.annual_results.append(results)

        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all simulated years.

        Returns:
            Dictionary with multi-year statistics.
        """
        if self.years_simulated == 0:
            return {}

        return {
            "years_simulated": self.years_simulated,
            "average_annual_claims": float(np.mean(self.total_claims)),
            "average_annual_recovery": float(np.mean(self.total_recoveries)),
            "average_annual_premium": float(np.mean(self.total_premiums)),
            "total_claims": sum(self.total_claims),
            "total_recoveries": sum(self.total_recoveries),
            "total_premiums": sum(self.total_premiums),
            "net_benefit": sum(self.total_recoveries) - sum(self.total_premiums),
            "recovery_ratio": sum(self.total_recoveries) / sum(self.total_claims)
            if sum(self.total_claims) > 0
            else 0,
            "loss_ratio": sum(self.total_recoveries) / sum(self.total_premiums)
            if sum(self.total_premiums) > 0
            else 0,
        }

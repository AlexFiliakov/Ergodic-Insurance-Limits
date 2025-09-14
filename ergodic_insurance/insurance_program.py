"""Multi-layer insurance program with reinstatements and advanced features.

This module provides comprehensive insurance program management including
multi-layer structures, reinstatements, attachment points, and accurate
loss allocation for manufacturing risk transfer optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult
import yaml

if TYPE_CHECKING:
    from .insurance_pricing import InsurancePricer, MarketCycle
    from .loss_distributions import LossEvent, ManufacturingLossGenerator
    from .manufacturer import WidgetManufacturer


class ReinstatementType(Enum):
    """Types of reinstatement provisions."""

    NONE = "none"
    PRO_RATA = "pro_rata"  # Premium based on time remaining
    FULL = "full"  # Full premium regardless of timing
    FREE = "free"  # No additional premium


@dataclass
class OptimizationConstraints:
    """Constraints for insurance program optimization."""

    max_total_premium: Optional[float] = None  # Budget constraint
    min_total_coverage: Optional[float] = None  # Minimum coverage requirement
    max_layers: int = 5  # Maximum number of layers
    min_layers: int = 3  # Minimum number of layers
    max_attachment_gap: float = 0.0  # Maximum gap between layers (0 = no gaps)
    min_roe_improvement: float = 0.15  # Minimum ROE improvement target
    max_iterations: int = 1000  # Maximum optimization iterations
    convergence_tolerance: float = 1e-6  # Convergence tolerance


@dataclass
class OptimalStructure:
    """Result of insurance structure optimization."""

    layers: List[EnhancedInsuranceLayer]
    deductible: float
    total_premium: float
    total_coverage: float
    ergodic_benefit: float
    roe_improvement: float
    optimization_metrics: Dict[str, Any]
    convergence_achieved: bool
    iterations_used: int


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
    participation_rate: float = 1.0  # % of loss covered by this layer (default 100%)

    def __post_init__(self):
        """Validate layer parameters."""
        if self.attachment_point < 0:
            raise ValueError(f"Attachment point must be non-negative, got {self.attachment_point}")
        if self.limit <= 0:
            raise ValueError(f"Limit must be positive, got {self.limit}")
        # Initialize exhausted tracking
        self.exhausted = 0.0
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
        if self.reinstatement_type == ReinstatementType.FULL:
            return base_premium * self.reinstatement_premium
        if self.reinstatement_type == ReinstatementType.PRO_RATA:
            return base_premium * self.reinstatement_premium * timing_factor
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
        pricing_enabled: bool = False,
        pricer: Optional["InsurancePricer"] = None,
    ):
        """Initialize insurance program.

        Args:
            layers: List of insurance layers (will be sorted by attachment).
            deductible: Self-insured retention before insurance.
            name: Program identifier.
            pricing_enabled: Whether to use dynamic pricing.
            pricer: Optional InsurancePricer for dynamic pricing.
        """
        self.layers = sorted(layers, key=lambda x: x.attachment_point)
        self.deductible = deductible
        self.name = name
        self.layer_states = [LayerState(layer) for layer in self.layers]
        self.total_premiums_paid = 0.0
        self.total_claims = 0
        self.pricing_enabled = pricing_enabled
        self.pricer = pricer
        self.pricing_results: List[Any] = []

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
        result: Dict[str, Any] = {
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
            claim_times = list(np.linspace(0, 1, len(claims))) if claims else []

        results: Dict[str, Any] = {
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

    def get_program_summary(self) -> Dict[str, Any]:
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

    def _get_default_manufacturer_profile(self) -> Dict[str, Any]:
        """Get default manufacturer profile."""
        return {
            "initial_assets": 10_000_000,
            "annual_revenue": 15_000_000,
            "operating_margin": 0.08,
            "growth_rate": 0.05,
        }

    def _calculate_insurance_metrics(self, loss_history: List[List[float]]) -> tuple:
        """Calculate metrics with and without insurance.

        Returns:
            Tuple of (metrics_with, metrics_without) as numpy arrays.
        """
        metrics_with_insurance = []
        metrics_without_insurance = []

        for annual_losses in loss_history:
            # Without insurance: company bears all losses
            total_loss_without = sum(annual_losses)
            net_impact_without = -total_loss_without

            # With insurance: apply structure
            result = self.process_annual_claims(annual_losses)
            total_loss_with = result["total_deductible"]
            total_premium_paid = result["total_premium_paid"]
            net_impact_with = -total_loss_with - total_premium_paid

            metrics_without_insurance.append(net_impact_without)
            metrics_with_insurance.append(net_impact_with)

            # Reset for next year
            self.reset_annual()
        return np.array(metrics_with_insurance), np.array(metrics_without_insurance)

    def _calculate_time_average_growth(self, metrics: np.ndarray, initial_assets: float) -> tuple:
        """Calculate time-average growth rate.

        Returns:
            Tuple of (time_avg_growth, final_assets).
        """
        assets = initial_assets + np.cumsum(metrics)
        assets = np.maximum(assets, 1.0)  # Ensure positive for log
        if len(assets) > 1:
            time_avg = np.log(assets[-1] / initial_assets) / len(assets)
        else:
            time_avg = 0.0
        return time_avg, assets[-1]

    def calculate_ergodic_benefit(
        self,
        loss_history: List[List[float]],
        manufacturer_profile: Optional[Dict[str, Any]] = None,
        time_horizon: int = 100,
    ) -> Dict[str, float]:
        """Calculate ergodic benefit of insurance structure.

        Quantifies time-average growth improvement from insurance coverage
        versus ensemble-average cost.

        Args:
            loss_history: Historical loss data (list of annual loss lists).
            manufacturer_profile: Company profile with assets, revenue, etc.
            time_horizon: Time horizon for ergodic calculation.

        Returns:
            Dictionary with ergodic metrics.
        """
        if not loss_history:
            return {
                "time_average_benefit": 0.0,
                "ensemble_average_cost": 0.0,
                "ergodic_ratio": 0.0,
                "volatility_reduction": 0.0,
            }

        # Default manufacturer profile
        if manufacturer_profile is None:
            manufacturer_profile = self._get_default_manufacturer_profile()

        # Calculate metrics with and without insurance
        metrics_with, metrics_without = self._calculate_insurance_metrics(loss_history)

        # Time-average growth rates
        initial_assets = manufacturer_profile["initial_assets"]
        time_avg_with, _ = self._calculate_time_average_growth(metrics_with, initial_assets)
        time_avg_without, _ = self._calculate_time_average_growth(metrics_without, initial_assets)

        # Ensemble averages
        ensemble_avg_without = np.mean(metrics_without)
        ensemble_avg_with = np.mean(metrics_with)

        # Volatility metrics
        volatility_without = np.std(metrics_without)
        volatility_with = np.std(metrics_with)
        volatility_reduction = (
            (volatility_without - volatility_with) / volatility_without
            if volatility_without > 0
            else 0.0
        )

        return {
            "time_average_benefit": time_avg_with - time_avg_without,
            "ensemble_average_cost": ensemble_avg_with - ensemble_avg_without,
            "ergodic_ratio": (
                time_avg_with / time_avg_without if time_avg_without != 0 else float("inf")
            ),
            "volatility_reduction": volatility_reduction,
            "time_avg_growth_with": time_avg_with,
            "time_avg_growth_without": time_avg_without,
            "ensemble_avg_with": ensemble_avg_with,
            "ensemble_avg_without": ensemble_avg_without,
        }

    def find_optimal_attachment_points(
        self, loss_data: List[float], num_layers: int = 4, percentiles: Optional[List[float]] = None
    ) -> List[float]:
        """Find optimal attachment points based on loss frequency/severity.

        Uses data-driven approach to minimize gaps while optimizing cost.

        Args:
            loss_data: Historical loss amounts.
            num_layers: Number of layers to optimize.
            percentiles: Optional percentiles for attachment points.

        Returns:
            List of optimal attachment points.
        """
        if not loss_data or num_layers <= 0:
            return []

        loss_array = np.array(loss_data)
        loss_array = loss_array[loss_array > 0]  # Filter positive losses

        if len(loss_array) == 0:
            return []

        # Default percentiles based on typical layer structure
        if percentiles is None:
            if num_layers == 3:
                percentiles = [50, 90, 99]  # Working, excess, cat
            elif num_layers == 4:
                percentiles = [40, 80, 95, 99.5]
            elif num_layers == 5:
                percentiles = [30, 60, 85, 95, 99.5]
            else:
                # Even distribution
                percentiles = np.linspace(100 / (num_layers + 1), 99, num_layers).tolist()

        # Calculate attachment points from percentiles
        attachment_points = []
        for p in percentiles:
            attachment = float(np.percentile(loss_array, p))
            attachment_points.append(attachment)

        # Round to reasonable values
        attachment_points = [self._round_attachment_point(ap) for ap in attachment_points]

        # Ensure strictly increasing
        for i in range(1, len(attachment_points)):
            if attachment_points[i] <= attachment_points[i - 1]:
                attachment_points[i] = attachment_points[i - 1] * 1.5

        return attachment_points

    def _round_attachment_point(self, value: float) -> float:
        """Round attachment point to reasonable market value."""
        if value < 100_000:
            return float(round(value / 10_000) * 10_000)
        if value < 1_000_000:
            return float(round(value / 50_000) * 50_000)
        if value < 10_000_000:
            return float(round(value / 250_000) * 250_000)
        return float(round(value / 1_000_000) * 1_000_000)

    def _get_layer_capacity(self, attachment_point: float) -> float:
        """Get default capacity for a layer based on attachment point."""
        capacity_thresholds = [
            (1_000_000, 5_000_000),
            (10_000_000, 25_000_000),
            (50_000_000, 50_000_000),
            (float("inf"), 100_000_000),
        ]
        for threshold, capacity in capacity_thresholds:
            if attachment_point < threshold:
                return capacity
        return 100_000_000  # Default fallback

    def optimize_layer_widths(
        self,
        attachment_points: List[float],
        total_budget: float,
        capacity_constraints: Optional[Dict[str, float]] = None,
        loss_data: Optional[List[float]] = None,
    ) -> List[float]:
        """Optimize layer widths given attachment points and constraints.

        Args:
            attachment_points: Fixed attachment points for layers.
            total_budget: Total premium budget.
            capacity_constraints: Optional max capacity per layer.
            loss_data: Optional loss data for severity analysis.

        Returns:
            List of optimal layer widths.
        """
        if not attachment_points:
            return []

        num_layers = len(attachment_points)

        # Default capacity constraints
        if capacity_constraints is None:
            capacity_constraints = {
                f"layer_{i}": self._get_layer_capacity(ap) for i, ap in enumerate(attachment_points)
            }

        # Analyze loss severity at each attachment point
        severity_weights: List[float] = []
        if loss_data:
            for ap in attachment_points:
                excess_losses = [max(0, loss - ap) for loss in loss_data if loss > ap]
                avg_excess = float(np.mean(excess_losses)) if excess_losses else float(ap)
                severity_weights.append(avg_excess)
        else:
            # Default weights based on attachment points
            severity_weights = [1.0 / (i + 1) for i in range(num_layers)]

        # Normalize weights
        total_weight = sum(severity_weights)
        if total_weight > 0:
            severity_weights = [w / total_weight for w in severity_weights]
        else:
            severity_weights = [1.0 / num_layers] * num_layers

        # Calculate layer widths based on budget and weights
        layer_widths = []
        for i, (ap, weight) in enumerate(zip(attachment_points, severity_weights)):
            # Estimate premium rate (decreasing with attachment)
            if ap < 1_000_000:
                rate = 0.015
            elif ap < 5_000_000:
                rate = 0.010
            elif ap < 25_000_000:
                rate = 0.006
            else:
                rate = 0.003

            # Calculate width from budget allocation
            allocated_budget = total_budget * weight
            width = allocated_budget / rate

            # Apply capacity constraint
            max_capacity = capacity_constraints.get(f"layer_{i}", float("inf"))
            width = min(float(width), float(max_capacity))

            # Ensure minimum width
            min_width = ap * 0.5 if i == 0 else attachment_points[i - 1] * 0.3
            width = max(float(width), float(min_width))

            layer_widths.append(self._round_attachment_point(float(width)))

        return layer_widths

    def _get_premium_rate(self, attachment_point: float) -> float:
        """Get premium rate based on attachment point."""
        rate_thresholds = [
            (1_000_000, 0.015),
            (5_000_000, 0.010),
            (25_000_000, 0.006),
            (float("inf"), 0.003),
        ]
        for threshold, rate in rate_thresholds:
            if attachment_point < threshold:
                return rate
        return 0.003  # Default fallback

    def _calculate_reinstatements(self, layer_index: int, num_layers: int) -> int:
        """Calculate reinstatements for a layer."""
        if layer_index == 0:
            return 0  # Primary layer
        if layer_index == num_layers - 1:
            return 999  # Top layer - unlimited
        return max(0, 2 - layer_index // 2)  # Decreasing with height

    def _create_layer_structure(
        self, attachment_points: List[float], layer_widths: List[float]
    ) -> List[EnhancedInsuranceLayer]:
        """Create insurance layers from attachment points and widths."""
        layers = []
        num_layers = len(attachment_points)
        for i, (ap, width) in enumerate(zip(attachment_points, layer_widths)):
            layer = EnhancedInsuranceLayer(
                attachment_point=ap,
                limit=width,
                premium_rate=self._get_premium_rate(ap),
                reinstatements=self._calculate_reinstatements(i, num_layers),
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.PRO_RATA,
            )
            layers.append(layer)
        return layers

    def _calculate_roe_improvement(
        self, ergodic_metrics: Dict[str, float], company_profile: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate ROE improvement from ergodic metrics."""
        if company_profile and "initial_assets" in company_profile:
            initial_roe = 0.08  # Baseline assumption
            improved_roe = initial_roe + ergodic_metrics["time_average_benefit"]
            return improved_roe / initial_roe - 1.0
        return ergodic_metrics["time_average_benefit"] / 0.08

    def optimize_layer_structure(
        self,
        loss_data: List[List[float]],
        company_profile: Optional[Dict[str, Any]] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> OptimalStructure:
        """Optimize complete insurance layer structure.

        Main optimization method that orchestrates layer count, attachment points,
        and widths to maximize ergodic benefit.

        Args:
            loss_data: Historical loss data (list of annual loss lists).
            company_profile: Company financial profile.
            constraints: Optimization constraints.

        Returns:
            Optimal insurance structure.
        """
        if constraints is None:
            constraints = OptimizationConstraints()

        # Flatten loss data for analysis
        all_losses = [loss for annual_losses in loss_data for loss in annual_losses]

        # Determine optimal number of layers
        best_structure = None
        best_ergodic_benefit = -float("inf")

        for num_layers in range(constraints.min_layers, constraints.max_layers + 1):
            # Find attachment points
            attachment_points = self.find_optimal_attachment_points(all_losses, num_layers)

            if not attachment_points:
                continue

            # Set deductible and budget
            deductible = attachment_points[0] * 0.5
            budget = constraints.max_total_premium or (
                float(np.mean([sum(annual) for annual in loss_data])) * 0.15
            )

            # Optimize layer widths
            layer_widths = self.optimize_layer_widths(
                attachment_points, budget, loss_data=all_losses
            )

            # Create and test layer structure
            layers = self._create_layer_structure(attachment_points, layer_widths)
            test_program = InsuranceProgram(layers=layers, deductible=deductible)
            ergodic_metrics = test_program.calculate_ergodic_benefit(loss_data, company_profile)

            # Check if this is better
            if ergodic_metrics["time_average_benefit"] > best_ergodic_benefit:
                best_ergodic_benefit = ergodic_metrics["time_average_benefit"]
                roe_improvement = self._calculate_roe_improvement(ergodic_metrics, company_profile)

                best_structure = OptimalStructure(
                    layers=layers,
                    deductible=deductible,
                    total_premium=test_program.calculate_annual_premium(),
                    total_coverage=test_program.get_total_coverage(),
                    ergodic_benefit=best_ergodic_benefit,
                    roe_improvement=roe_improvement,
                    optimization_metrics=ergodic_metrics,
                    convergence_achieved=True,
                    iterations_used=num_layers - constraints.min_layers + 1,
                )

        # If no structure found, return a basic one
        if best_structure is None:
            basic_layers = [
                EnhancedInsuranceLayer(
                    attachment_point=250_000,
                    limit=4_750_000,
                    premium_rate=0.015,
                    reinstatements=0,
                )
            ]
            basic_program = InsuranceProgram(layers=basic_layers, deductible=250_000)

            best_structure = OptimalStructure(
                layers=basic_layers,
                deductible=250_000,
                total_premium=basic_program.calculate_annual_premium(),
                total_coverage=5_000_000,
                ergodic_benefit=0.0,
                roe_improvement=0.0,
                optimization_metrics={},
                convergence_achieved=False,
                iterations_used=0,
            )

        return best_structure

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

    def apply_pricing(
        self,
        expected_revenue: float,
        market_cycle: Optional["MarketCycle"] = None,
        loss_generator: Optional["ManufacturingLossGenerator"] = None,
    ) -> None:
        """Apply dynamic pricing to all layers in the program.

        Updates layer premium rates based on frequency/severity calculations.

        Args:
            expected_revenue: Expected annual revenue for scaling
            market_cycle: Optional market cycle state
            loss_generator: Optional loss generator (uses pricer's if not provided)

        Raises:
            ValueError: If pricing not enabled or pricer not configured
        """
        if not self.pricing_enabled:
            raise ValueError("Pricing not enabled for this program")

        if self.pricer is None:
            if loss_generator is None:
                raise ValueError("Either pricer or loss_generator must be provided")

            # Create a default pricer
            from .insurance_pricing import InsurancePricer, MarketCycle

            self.pricer = InsurancePricer(
                loss_generator=loss_generator,
                market_cycle=market_cycle or MarketCycle.NORMAL,
            )

        # Apply pricing to the program
        self.pricer.price_insurance_program(
            program=self,
            expected_revenue=expected_revenue,
            market_cycle=market_cycle,
            update_program=True,
        )

    def get_pricing_summary(self) -> Dict[str, Any]:
        """Get summary of current pricing.

        Returns:
            Dictionary with pricing details for each layer
        """
        summary: Dict[str, Any] = {
            "program_name": self.name,
            "pricing_enabled": self.pricing_enabled,
            "total_premium": self.calculate_annual_premium(),
            "layers": [],
        }

        if self.pricing_results:
            for i, (layer, pricing) in enumerate(zip(self.layers, self.pricing_results)):
                summary["layers"].append(
                    {
                        "index": i,
                        "attachment_point": layer.attachment_point,
                        "limit": layer.limit,
                        "premium_rate": layer.premium_rate,
                        "market_premium": pricing.market_premium
                        if pricing
                        else layer.limit * layer.premium_rate,
                        "pure_premium": pricing.pure_premium if pricing else None,
                        "expected_frequency": pricing.expected_frequency if pricing else None,
                        "expected_severity": pricing.expected_severity if pricing else None,
                    }
                )
        else:
            for i, layer in enumerate(self.layers):
                summary["layers"].append(
                    {
                        "index": i,
                        "attachment_point": layer.attachment_point,
                        "limit": layer.limit,
                        "premium_rate": layer.premium_rate,
                        "premium": layer.calculate_base_premium(),
                    }
                )

        return summary

    @classmethod
    def create_with_pricing(
        cls,
        layers: List[EnhancedInsuranceLayer],
        loss_generator: "ManufacturingLossGenerator",
        expected_revenue: float,
        market_cycle: Optional["MarketCycle"] = None,
        deductible: float = 0.0,
        name: str = "Priced Insurance Program",
    ) -> "InsuranceProgram":
        """Create insurance program with dynamic pricing.

        Factory method that creates a program with pricing already applied.

        Args:
            layers: Initial layer structure
            loss_generator: Loss generator for pricing
            expected_revenue: Expected annual revenue
            market_cycle: Market cycle state
            deductible: Self-insured retention
            name: Program name

        Returns:
            InsuranceProgram with pricing applied
        """
        from .insurance_pricing import InsurancePricer, MarketCycle

        # Create pricer
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=market_cycle or MarketCycle.NORMAL,
        )

        # Create program with pricing enabled
        program = cls(
            layers=layers,
            deductible=deductible,
            name=name,
            pricing_enabled=True,
            pricer=pricer,
        )

        # Apply pricing
        program.apply_pricing(expected_revenue, market_cycle)

        return program


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
            "recovery_ratio": (
                sum(self.total_recoveries) / sum(self.total_claims)
                if sum(self.total_claims) > 0
                else 0
            ),
            "loss_ratio": (
                sum(self.total_recoveries) / sum(self.total_premiums)
                if sum(self.total_premiums) > 0
                else 0
            ),
        }

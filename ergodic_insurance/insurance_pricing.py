"""Insurance pricing module with market cycle support.

This module implements realistic insurance premium calculation based on
frequency and severity distributions, replacing hardcoded premium rates
in simulations. It supports market cycle adjustments and integrates with
existing loss generators and insurance structures.

Example:
    Basic usage for pricing an insurance program::

        from ergodic_insurance.insurance_pricing import InsurancePricer, MarketCycle
        from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

        # Initialize loss generator and pricer
        loss_gen = ManufacturingLossGenerator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            loss_ratio=0.70,
            market_cycle=MarketCycle.NORMAL
        )

        # Price an insurance program
        program = InsuranceProgram(layers=[...])
        priced_program = pricer.price_insurance_program(
            program,
            expected_revenue=15_000_000
        )

        # Get total premium
        total_premium = priced_program.calculate_annual_premium()

Attributes:
    MarketCycle: Enum representing market conditions (HARD, NORMAL, SOFT)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .exposure_base import ExposureBase
    from .insurance import InsuranceLayer, InsurancePolicy
    from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram
    from .loss_distributions import ManufacturingLossGenerator


class MarketCycle(Enum):
    """Market cycle states affecting insurance pricing.

    Each state corresponds to a target loss ratio that insurers
    use to price coverage. Lower loss ratios (hard markets) result
    in higher premiums.

    Attributes:
        HARD: Seller's market with limited capacity (60% loss ratio)
        NORMAL: Balanced market conditions (70% loss ratio)
        SOFT: Buyer's market with excess capacity (80% loss ratio)
    """

    HARD = 0.60  # 60% loss ratio - higher premiums
    NORMAL = 0.70  # 70% loss ratio - standard premiums
    SOFT = 0.80  # 80% loss ratio - lower premiums


@dataclass
class PricingParameters:
    """Parameters for insurance pricing calculations.

    Attributes:
        loss_ratio: Target loss ratio for pricing (claims/premium)
        expense_ratio: Operating expense ratio (default 0.25)
        profit_margin: Target profit margin (default 0.05)
        risk_loading: Additional loading for uncertainty (default 0.10)
        confidence_level: Confidence level for pricing (default 0.95)
        simulation_years: Years to simulate for pricing (default 10)
        min_premium: Minimum premium floor (default 1000)
        max_rate_on_line: Maximum rate on line cap (default 0.50)
    """

    loss_ratio: float = 0.70
    expense_ratio: float = 0.25
    profit_margin: float = 0.05
    risk_loading: float = 0.10
    confidence_level: float = 0.95
    simulation_years: int = 10
    min_premium: float = 1000.0
    max_rate_on_line: float = 0.50


@dataclass
class LayerPricing:
    """Pricing details for a single insurance layer.

    Attributes:
        attachment_point: Where coverage starts
        limit: Maximum coverage amount
        expected_frequency: Expected claims per year hitting this layer
        expected_severity: Average severity of claims in this layer
        pure_premium: Expected loss cost
        technical_premium: Pure premium with expenses and profit
        market_premium: Final premium after market adjustments
        rate_on_line: Premium as percentage of limit
        confidence_interval: (lower, upper) bounds at confidence level
        lae_loading: LAE component embedded in the expense ratio (Issue #468)
    """

    attachment_point: float
    limit: float
    expected_frequency: float
    expected_severity: float
    pure_premium: float
    technical_premium: float
    market_premium: float
    rate_on_line: float
    confidence_interval: Tuple[float, float]
    lae_loading: float = 0.0


class InsurancePricer:
    """Calculate insurance premiums based on loss distributions and market conditions.

    This class provides methods to price individual layers and complete insurance
    programs using frequency/severity distributions from loss generators. It supports
    market cycle adjustments and maintains backward compatibility with fixed rates.

    Args:
        loss_generator: Manufacturing loss generator for frequency/severity data
        loss_ratio: Target loss ratio for pricing (or use market_cycle)
        market_cycle: Market cycle state (overrides loss_ratio if provided)
        parameters: Additional pricing parameters
        seed: Random seed for reproducible simulations

    Example:
        Pricing with different market conditions::

            # Hard market pricing (higher premiums)
            hard_pricer = InsurancePricer(
                loss_generator=loss_gen,
                market_cycle=MarketCycle.HARD
            )

            # Soft market pricing (lower premiums)
            soft_pricer = InsurancePricer(
                loss_generator=loss_gen,
                market_cycle=MarketCycle.SOFT
            )
    """

    def __init__(
        self,
        loss_generator: Optional["ManufacturingLossGenerator"] = None,
        loss_ratio: Optional[float] = None,
        market_cycle: Optional[MarketCycle] = None,
        parameters: Optional[PricingParameters] = None,
        exposure: Optional["ExposureBase"] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the insurance pricer.

        Args:
            loss_generator: Loss generator for frequency/severity
            loss_ratio: Target loss ratio (0-1)
            market_cycle: Market cycle state
            parameters: Pricing parameters
            exposure: Optional exposure object for dynamic revenue tracking
            seed: Random seed for simulations
        """
        self.loss_generator = loss_generator
        self.exposure = exposure
        self.parameters = parameters or PricingParameters()
        self.rng = np.random.default_rng(seed)

        # Set loss ratio based on market cycle or explicit value
        if market_cycle is not None:
            self.loss_ratio = market_cycle.value
            self.market_cycle = market_cycle
        elif loss_ratio is not None:
            self.loss_ratio = loss_ratio
            self.market_cycle = self._infer_market_cycle(loss_ratio)
        else:
            self.loss_ratio = self.parameters.loss_ratio
            self.market_cycle = MarketCycle.NORMAL

    def _infer_market_cycle(self, loss_ratio: float) -> MarketCycle:
        """Infer market cycle from loss ratio.

        Args:
            loss_ratio: Target loss ratio

        Returns:
            Closest market cycle state
        """
        if loss_ratio <= 0.65:
            return MarketCycle.HARD
        if loss_ratio >= 0.75:
            return MarketCycle.SOFT
        return MarketCycle.NORMAL

    def calculate_pure_premium(
        self,
        attachment_point: float,
        limit: float,
        expected_revenue: float,
        simulation_years: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate pure premium for a layer using frequency/severity.

        Pure premium represents the expected loss cost without expenses,
        profit, or risk loading.

        Args:
            attachment_point: Where layer coverage starts
            limit: Maximum coverage from this layer
            expected_revenue: Expected annual revenue for scaling
            simulation_years: Years to simulate (default from parameters)

        Returns:
            Tuple of (pure_premium, statistics_dict) with detailed metrics

        Raises:
            ValueError: If loss_generator is not configured
        """
        if self.loss_generator is None:
            raise ValueError("Loss generator required for pure premium calculation")

        years = simulation_years or self.parameters.simulation_years

        # Run simulations to estimate losses in this layer
        layer_losses = []
        frequencies = []
        severities = []

        for _ in range(years):
            # Generate annual losses
            losses, _stats = self.loss_generator.generate_losses(
                duration=1.0, revenue=expected_revenue, include_catastrophic=True
            )

            # Calculate losses hitting this layer
            annual_layer_losses = []
            for loss in losses:
                if loss.amount > attachment_point:
                    layer_loss = min(loss.amount - attachment_point, limit)
                    annual_layer_losses.append(layer_loss)

            # Track statistics
            if annual_layer_losses:
                layer_losses.extend(annual_layer_losses)
                frequencies.append(len(annual_layer_losses))
                severities.extend(annual_layer_losses)
            else:
                frequencies.append(0)

        # Calculate expected values
        if layer_losses:
            expected_frequency = float(np.mean(frequencies))
            expected_severity = float(np.mean(severities) if severities else 0)
            pure_premium = expected_frequency * expected_severity

            # Calculate confidence interval
            if len(layer_losses) > 1:
                lower = np.percentile(layer_losses, (1 - self.parameters.confidence_level) * 50)
                upper = np.percentile(layer_losses, 50 + self.parameters.confidence_level * 50)
                confidence_interval = (float(lower), float(upper))
            else:
                confidence_interval = (pure_premium * 0.8, pure_premium * 1.2)
        else:
            expected_frequency = 0.0
            expected_severity = 0.0
            pure_premium = 0.0
            confidence_interval = (0.0, 0.0)

        statistics = {
            "expected_frequency": expected_frequency,
            "expected_severity": expected_severity,
            "pure_premium": pure_premium,
            "confidence_interval": confidence_interval,
            "years_simulated": years,
            "total_losses_in_layer": len(layer_losses),
            "max_loss_in_layer": max(layer_losses) if layer_losses else 0,
            "attachment_point": attachment_point,
            "limit": limit,
        }

        return pure_premium, statistics

    def calculate_technical_premium(
        self,
        pure_premium: float,
        limit: float,
    ) -> float:
        """Convert pure premium to technical premium with risk loading.

        Technical premium adds a risk loading for parameter uncertainty
        to the pure premium. Expense and profit margins are applied
        separately via the loss ratio in calculate_market_premium()
        to avoid double-counting.

        Args:
            pure_premium: Expected loss cost
            limit: Layer limit for rate capping

        Returns:
            Technical premium amount
        """
        # Add risk loading for uncertainty
        risk_loading = 1 + self.parameters.risk_loading

        # Calculate technical premium (expense/profit applied via loss ratio)
        technical_premium = pure_premium * risk_loading

        # Apply minimum premium
        technical_premium = max(technical_premium, self.parameters.min_premium)

        # Cap at maximum rate on line
        max_premium = limit * self.parameters.max_rate_on_line
        technical_premium = min(technical_premium, max_premium)

        return technical_premium

    def calculate_market_premium(
        self,
        technical_premium: float,
        market_cycle: Optional[MarketCycle] = None,
    ) -> float:
        """Apply market cycle adjustment to technical premium.

        Market premium = Technical premium / Loss ratio

        Args:
            technical_premium: Premium with expenses and loadings
            market_cycle: Optional market cycle override

        Returns:
            Market-adjusted premium
        """
        cycle = market_cycle or self.market_cycle
        loss_ratio = cycle.value

        # Market premium = Technical premium / Loss ratio
        # Lower loss ratio (HARD market) means higher premiums
        # Higher loss ratio (SOFT market) means lower premiums
        # Example: HARD (0.6) -> premium/0.6 = 1.67x premium
        # Example: SOFT (0.8) -> premium/0.8 = 1.25x premium
        market_premium = technical_premium / loss_ratio

        return market_premium

    def price_layer(
        self,
        attachment_point: float,
        limit: float,
        expected_revenue: float,
        market_cycle: Optional[MarketCycle] = None,
    ) -> LayerPricing:
        """Price a single insurance layer.

        Complete pricing process from pure premium through market adjustment.

        Args:
            attachment_point: Where coverage starts
            limit: Maximum coverage amount
            expected_revenue: Expected annual revenue
            market_cycle: Optional market cycle override

        Returns:
            LayerPricing object with all pricing details
        """
        # Calculate pure premium
        pure_premium, stats = self.calculate_pure_premium(attachment_point, limit, expected_revenue)

        # Calculate technical premium with loadings
        technical_premium = self.calculate_technical_premium(pure_premium, limit)

        # Apply market cycle adjustment
        market_premium = self.calculate_market_premium(technical_premium, market_cycle=market_cycle)

        # Calculate rate on line
        rate_on_line = market_premium / limit if limit > 0 else 0.0

        # LAE loading is the portion of expenses attributable to loss adjustment (Issue #468)
        lae_loading = pure_premium * self.parameters.expense_ratio

        return LayerPricing(
            attachment_point=attachment_point,
            limit=limit,
            expected_frequency=stats["expected_frequency"],
            expected_severity=stats["expected_severity"],
            pure_premium=pure_premium,
            technical_premium=technical_premium,
            market_premium=market_premium,
            rate_on_line=rate_on_line,
            confidence_interval=stats["confidence_interval"],
            lae_loading=lae_loading,
        )

    def price_insurance_program(
        self,
        program: "InsuranceProgram",
        expected_revenue: Optional[float] = None,
        time: float = 0.0,
        market_cycle: Optional[MarketCycle] = None,
        update_program: bool = True,
    ) -> "InsuranceProgram":
        """Price a complete insurance program.

        Prices all layers in the program and optionally updates their rates.

        Args:
            program: Insurance program to price
            expected_revenue: Expected annual revenue (optional if using exposure)
            time: Time for exposure calculation (default 0.0)
            market_cycle: Optional market cycle override
            update_program: Whether to update program layer rates

        Returns:
            Program with updated pricing (original or copy based on update_program)
        """
        from .insurance_program import InsuranceProgram

        # Create a copy if not updating in place
        if not update_program:
            import copy

            program = copy.deepcopy(program)

        # Get actual revenue from exposure if available, otherwise use expected_revenue
        if self.exposure is not None:
            actual_revenue = self.exposure.get_exposure(time)
        elif expected_revenue is not None:
            actual_revenue = expected_revenue
        else:
            raise ValueError("Either expected_revenue or exposure must be provided")

        # Price each layer
        pricing_results = []
        for layer in program.layers:
            layer_pricing = self.price_layer(
                attachment_point=layer.attachment_point,
                limit=layer.limit,
                expected_revenue=actual_revenue,
                market_cycle=market_cycle,
            )
            pricing_results.append(layer_pricing)

            # Update layer premium rate if requested
            if update_program:
                layer.base_premium_rate = layer_pricing.rate_on_line

        # Store pricing results in program for reference
        if not hasattr(program, "pricing_results"):
            program.pricing_results = []
        program.pricing_results = pricing_results

        return program

    def price_insurance_policy(
        self,
        policy: "InsurancePolicy",
        expected_revenue: float,
        market_cycle: Optional[MarketCycle] = None,
        update_policy: bool = True,
    ) -> "InsurancePolicy":
        """Price a basic insurance policy.

        .. deprecated::
            Use :meth:`price_insurance_program` instead.

        Prices all layers in the policy and optionally updates their rates.

        Args:
            policy: Insurance policy to price
            expected_revenue: Expected annual revenue
            market_cycle: Optional market cycle override
            update_policy: Whether to update policy layer rates

        Returns:
            Policy with updated pricing (original or copy based on update_policy)
        """
        import warnings

        warnings.warn(
            "price_insurance_policy() is deprecated. "
            "Use price_insurance_program() with an InsuranceProgram instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .insurance import InsurancePolicy

        # Create a copy if not updating in place
        if not update_policy:
            import copy

            policy = copy.deepcopy(policy)

        # Price each layer
        pricing_results = []
        for layer in policy.layers:
            layer_pricing = self.price_layer(
                attachment_point=layer.attachment_point,
                limit=layer.limit,
                expected_revenue=expected_revenue,
                market_cycle=market_cycle,
            )
            pricing_results.append(layer_pricing)

            # Update layer rate if requested
            if update_policy:
                layer.rate = layer_pricing.rate_on_line

        # Store pricing results for reference
        if not hasattr(policy, "pricing_results"):
            policy.pricing_results = []
        policy.pricing_results = pricing_results

        return policy

    def compare_market_cycles(
        self,
        attachment_point: float,
        limit: float,
        expected_revenue: float,
    ) -> Dict[str, LayerPricing]:
        """Compare pricing across different market cycles.

        Useful for understanding market impact on premiums.

        Args:
            attachment_point: Where coverage starts
            limit: Maximum coverage amount
            expected_revenue: Expected annual revenue

        Returns:
            Dictionary mapping market cycle names to pricing results
        """
        # Calculate pure premium ONCE (it doesn't change by market cycle)
        pure_premium, stats = self.calculate_pure_premium(attachment_point, limit, expected_revenue)

        # Calculate technical premium ONCE (also doesn't change by market cycle)
        technical_premium = self.calculate_technical_premium(pure_premium, limit)

        results = {}

        # LAE loading is the portion of expenses attributable to loss adjustment (Issue #468)
        lae_loading = pure_premium * self.parameters.expense_ratio

        # Apply different market cycle adjustments to the same technical premium
        for cycle in MarketCycle:
            # Only the market adjustment changes
            market_premium = self.calculate_market_premium(technical_premium, market_cycle=cycle)
            rate_on_line = market_premium / limit if limit > 0 else 0.0

            results[cycle.name] = LayerPricing(
                attachment_point=attachment_point,
                limit=limit,
                expected_frequency=stats["expected_frequency"],
                expected_severity=stats["expected_severity"],
                pure_premium=pure_premium,
                technical_premium=technical_premium,
                market_premium=market_premium,
                rate_on_line=rate_on_line,
                confidence_interval=stats["confidence_interval"],
                lae_loading=lae_loading,
            )

        return results

    def simulate_cycle_transition(
        self,
        program: "InsuranceProgram",
        expected_revenue: float,
        years: int = 10,
        transition_probs: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Simulate insurance pricing over market cycle transitions.

        Models how premiums change as markets transition between states.

        Args:
            program: Insurance program to simulate
            expected_revenue: Expected annual revenue
            years: Number of years to simulate
            transition_probs: Market transition probabilities

        Returns:
            List of annual results with cycle states and premiums
        """
        if transition_probs is None:
            # Default transition probabilities
            transition_probs = {
                "hard_to_normal": 0.4,
                "hard_to_soft": 0.1,
                "normal_to_hard": 0.2,
                "normal_to_soft": 0.2,
                "soft_to_normal": 0.3,
                "soft_to_hard": 0.1,
            }

        results = []
        current_cycle = self.market_cycle

        for year in range(years):
            # Price program for current cycle
            priced_program = self.price_insurance_program(
                program=program,
                expected_revenue=expected_revenue,
                market_cycle=current_cycle,
                update_program=False,
            )

            # Calculate total premium from actual pricing results
            if hasattr(priced_program, "pricing_results") and priced_program.pricing_results:
                total_premium = sum(pr.market_premium for pr in priced_program.pricing_results)
            else:
                # Fallback to calculating from layers
                total_premium = priced_program.calculate_annual_premium()

            # Store results
            results.append(
                {
                    "year": year,
                    "market_cycle": current_cycle.name,
                    "loss_ratio": current_cycle.value,
                    "total_premium": total_premium,
                    "layer_premiums": [pr.market_premium for pr in priced_program.pricing_results],
                }
            )

            # Transition to next cycle
            current_cycle = self._transition_cycle(current_cycle, transition_probs)

        return results

    def _transition_cycle(
        self,
        current: MarketCycle,
        probs: Dict[str, float],
    ) -> MarketCycle:
        """Simulate market cycle transition.

        Args:
            current: Current market cycle
            probs: Transition probabilities

        Returns:
            Next market cycle state
        """
        rand = self.rng.random()

        if current == MarketCycle.HARD:
            if rand < probs.get("hard_to_normal", 0.4):
                return MarketCycle.NORMAL
            if rand < probs.get("hard_to_normal", 0.4) + probs.get("hard_to_soft", 0.1):
                return MarketCycle.SOFT
            return MarketCycle.HARD

        if current == MarketCycle.NORMAL:
            if rand < probs.get("normal_to_hard", 0.2):
                return MarketCycle.HARD
            if rand < probs.get("normal_to_hard", 0.2) + probs.get("normal_to_soft", 0.2):
                return MarketCycle.SOFT
            return MarketCycle.NORMAL

        # SOFT
        if rand < probs.get("soft_to_normal", 0.3):
            return MarketCycle.NORMAL
        if rand < probs.get("soft_to_normal", 0.3) + probs.get("soft_to_hard", 0.1):
            return MarketCycle.HARD
        return MarketCycle.SOFT

    @staticmethod
    def create_from_config(
        config: Dict[str, Any],
        loss_generator: Optional["ManufacturingLossGenerator"] = None,
    ) -> "InsurancePricer":
        """Create pricer from configuration dictionary.

        Args:
            config: Configuration dictionary
            loss_generator: Optional loss generator

        Returns:
            Configured InsurancePricer instance
        """
        # Extract pricing parameters
        params = PricingParameters(
            loss_ratio=config.get("loss_ratio", 0.70),
            expense_ratio=config.get("expense_ratio", 0.25),
            profit_margin=config.get("profit_margin", 0.05),
            risk_loading=config.get("risk_loading", 0.10),
            confidence_level=config.get("confidence_level", 0.95),
            simulation_years=config.get("simulation_years", 10),
            min_premium=config.get("min_premium", 1000.0),
            max_rate_on_line=config.get("max_rate_on_line", 0.50),
        )

        # Get market cycle
        cycle_str = config.get("market_cycle", "NORMAL")
        try:
            market_cycle = MarketCycle[cycle_str.upper()]
        except KeyError:
            market_cycle = MarketCycle.NORMAL

        return InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=market_cycle,
            parameters=params,
            seed=config.get("seed"),
        )

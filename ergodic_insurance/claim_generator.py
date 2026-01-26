"""Claim generation for insurance simulations.

This module provides classes for generating realistic insurance claims
with configurable frequency and severity distributions, supporting
both regular and catastrophic event modeling.

The module integrates with the enhanced loss_distributions module
for more sophisticated risk modeling with revenue-dependent frequencies
and parametric severity distributions. It provides backward compatibility
with legacy systems while supporting advanced features when available.

Key Features:
    - Poisson frequency and lognormal severity distributions
    - Separate handling of attritional and catastrophic losses
    - Integration with enhanced loss distributions (when available)
    - Reproducible random generation with seed support
    - Conversion utilities between ClaimEvent and LossData formats

Examples:
    Basic claim generation::

        generator = ClaimGenerator(
            base_frequency=0.1,  # 10% chance per year
            severity_mean=5_000_000,
            severity_std=2_000_000,
            seed=42
        )
        claims = generator.generate_claims(years=10)

    Including catastrophic events::

        regular, catastrophic = generator.generate_all_claims(
            years=10,
            include_catastrophic=True,
            cat_frequency=0.01  # 1% chance per year
        )

Note:
    The module automatically detects if enhanced loss distributions are
    available and can fall back to standard generation methods.

Since:
    Version 0.1.0
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import trend classes for frequency and severity adjustments
from .trends import NoTrend, Trend

# Import enhanced distributions if available (backward compatibility)
try:
    from .loss_distributions import LossData
    from .loss_distributions import LossEvent as EnhancedLossEvent
    from .loss_distributions import ManufacturingLossGenerator

    ENHANCED_DISTRIBUTIONS_AVAILABLE = True
except ModuleNotFoundError:
    # Module doesn't exist - this is expected in minimal installations
    ENHANCED_DISTRIBUTIONS_AVAILABLE = False
    LossData = None  # type: ignore
except ImportError as e:
    # Module exists but has issues (syntax error, missing dependency, etc.)
    logger.warning(
        "loss_distributions module exists but failed to import: %s. "
        "Falling back to standard distributions.",
        e,
    )
    ENHANCED_DISTRIBUTIONS_AVAILABLE = False
    LossData = None  # type: ignore

# Import exposure base for dynamic frequency scaling
try:
    from .exposure_base import ExposureBase

    EXPOSURE_BASE_AVAILABLE = True
except ModuleNotFoundError:
    # Module doesn't exist - this is expected in minimal installations
    EXPOSURE_BASE_AVAILABLE = False
    ExposureBase = None  # type: ignore
except ImportError as e:
    # Module exists but has issues (syntax error, missing dependency, etc.)
    logger.warning(
        "exposure_base module exists but failed to import: %s. "
        "Dynamic exposure scaling will be unavailable.",
        e,
    )
    EXPOSURE_BASE_AVAILABLE = False
    ExposureBase = None  # type: ignore


@dataclass
class ClaimEvent:
    """Represents a single claim event.

    A simple data structure containing the year and amount of an insurance claim.
    Used throughout the simulation framework to track individual loss events.

    Attributes:
        year: The year (0-indexed) when the claim occurred.
        amount: The monetary amount of the claim in dollars.

    Examples:
        Creating a claim event::

            claim = ClaimEvent(year=5, amount=1_000_000)
            print(f"Year {claim.year}: ${claim.amount:,.0f}")
    """

    year: int
    amount: float


class ClaimGenerator:
    """Generate insurance claims for simulations.

    This class generates realistic insurance claims using Poisson processes
    for frequency and lognormal distributions for severity, with support for
    both regular attritional losses and catastrophic events.

    The generator uses standard actuarial models:
    - Frequency: Poisson distribution (constant rate parameter)
    - Severity: Lognormal distribution (right-skewed, positive values)
    - Catastrophes: Bernoulli trials with separate severity distribution
    - Trends: Multiplicative adjustments over time for frequency and severity

    Attributes:
        base_frequency: Base expected number of claims per year (Poisson lambda).
        exposure_base: Optional exposure base for dynamic frequency scaling.
        severity_mean: Mean claim size in dollars.
        severity_std: Standard deviation of claim size.
        frequency_trend: Trend object for frequency adjustments over time.
        severity_trend: Trend object for severity adjustments over time.
        rng: Random number generator for reproducibility.

    Examples:
        Generate claims for a decade::

            generator = ClaimGenerator(
                base_frequency=0.2,  # 20% expected frequency
                severity_mean=10_000_000,
                seed=42
            )

            claims = generator.generate_claims(years=10)
            total_loss = sum(c.amount for c in claims)
            print(f"Total losses: ${total_loss:,.0f}")

        Separate regular and catastrophic::

            regular, cat = generator.generate_all_claims(
                years=100,
                include_catastrophic=True,
                cat_frequency=0.005  # 0.5% annual probability
            )

    Note:
        The lognormal parameters are calculated from the desired mean and
        standard deviation to ensure the distribution matches expectations.
    """

    def __init__(
        self,
        base_frequency: float,  # Expected claims per year at base exposure
        severity_mean: float,  # Mean claim size
        severity_std: float,  # Std dev of claim size
        exposure_base: Optional["ExposureBase"] = None,  # Dynamic exposure calculator
        frequency_trend: Optional[Trend] = None,  # Trend for frequency adjustments
        severity_trend: Optional[Trend] = None,  # Trend for severity adjustments
        seed: Optional[int] = None,
        n_simulations: int = 100_000,  # Number of simulations for statistics
    ):
        """Initialize claim generator with optional dynamic exposure.

        Args:
            base_frequency: Base expected number of claims per year at reference
                exposure level. Must be non-negative. Default is 0.1.
            exposure_base: Optional dynamic exposure calculator for frequency scaling.
                When provided, actual frequency = base_frequency * exposure_multiplier.
            severity_mean: Mean claim size in dollars (lognormal parameter).
                Must be positive. Default is $5M.
            severity_std: Standard deviation of claim size. Must be non-negative.
                Default is $2M.
            frequency_trend: Optional trend object for frequency adjustments over time.
                Defaults to NoTrend() for backward compatibility.
            severity_trend: Optional trend object for severity adjustments over time.
                Defaults to NoTrend() for backward compatibility.
            seed: Random seed for reproducibility. If None, uses random state.
            n_simulations: Number of simulations for percentile/CVaR calculations.
                Default is 100,000. Higher values improve accuracy but increase computation time.

        Raises:
            ValueError: If base_frequency is negative or severity parameters are invalid.

        Examples:
            Create generator with static frequency::

                generator = ClaimGenerator(
                    base_frequency=0.15,  # 15% base frequency
                    severity_mean=8_000_000,  # $8M mean
                    seed=12345
                )

            Create generator with dynamic exposure::

                from ergodic_insurance.exposure_base import RevenueExposure

                exposure = RevenueExposure(
                    base_revenue=10_000_000,
                    growth_rate=0.10
                )

                generator = ClaimGenerator(
                    base_frequency=0.1,  # 10% at base revenue
                    exposure_base=exposure,
                    severity_mean=5_000_000
                )

            Create generator with trends::

                from ergodic_insurance.trends import LinearTrend

                generator = ClaimGenerator(
                    base_frequency=0.1,
                    severity_mean=5_000_000,
                    frequency_trend=LinearTrend(annual_rate=0.03),  # 3% annual growth
                    severity_trend=LinearTrend(annual_rate=0.05)   # 5% severity inflation
                )
        """
        # Validate parameters
        if base_frequency < 0:
            raise ValueError(f"Base frequency must be non-negative, got {base_frequency}")
        if severity_mean <= 0:
            raise ValueError(f"Severity mean must be positive, got {severity_mean}")
        if severity_std < 0:
            raise ValueError(f"Severity std must be non-negative, got {severity_std}")

        self.base_frequency = base_frequency
        self.exposure_base = exposure_base
        self.severity_mean = severity_mean
        self.severity_std = severity_std
        self.frequency_trend = frequency_trend if frequency_trend is not None else NoTrend()
        self.severity_trend = severity_trend if severity_trend is not None else NoTrend()
        self.rng = np.random.RandomState(seed)
        self.n_simulations = n_simulations
        self._simulation_cache: Optional[Dict[str, Any]] = None

    def generate_claims(self, years: int = 1) -> List[ClaimEvent]:
        """Generate claims for a simulation period.

        Generates claims for each year using Poisson frequency and lognormal
        severity distributions. Claims are generated independently for each year.

        Args:
            years: Number of years to simulate. Must be positive.

        Returns:
            List[ClaimEvent]: List of claim events with year and amount.
                Empty list if years <= 0 or frequency <= 0.

        Examples:
            Generate and analyze claims::

                claims = generator.generate_claims(years=50)

                # Group by year
                by_year = {}
                for claim in claims:
                    by_year.setdefault(claim.year, []).append(claim)

                # Find worst year
                worst_year = max(by_year.keys(),
                               key=lambda y: sum(c.amount for c in by_year[y]))

        Note:
            The lognormal distribution parameters (mu, sigma) are calculated
            from the specified mean and standard deviation to ensure the
            generated values match the desired statistics.
        """
        claims: List[ClaimEvent] = []

        # Handle edge cases
        if years <= 0 or self.base_frequency <= 0:
            return claims

        for year in range(years):
            # Number of claims this year (Poisson distribution with exposure adjustment)
            adjusted_frequency = self.get_adjusted_frequency(year)
            n_claims = self.rng.poisson(adjusted_frequency)

            for _ in range(n_claims):
                # Get severity adjusted for trends
                adjusted_severity = self.get_adjusted_severity(year)

                # Claim severity (lognormal distribution)
                # Convert mean/std to lognormal parameters
                variance = self.severity_std**2
                mean = adjusted_severity

                # Lognormal parameters
                sigma = np.sqrt(np.log(1 + variance / mean**2))
                mu = np.log(mean) - sigma**2 / 2

                amount = self.rng.lognormal(mu, sigma)
                claims.append(ClaimEvent(year=year, amount=amount))

        return claims

    def get_adjusted_frequency(self, year: int) -> float:
        """Get frequency adjusted for exposure and trends at given year.

        Multiplicatively stacks exposure and trend adjustments:
        adjusted_freq = base_freq × exposure_multiplier × trend_multiplier

        Args:
            year: Year number (0-indexed) for frequency calculation.

        Returns:
            float: Adjusted frequency for the specified year.
                If no exposure_base is set, only applies trend.
                If no trend is set, only applies exposure.

        Examples:
            Check frequency scaling::

                # With exposure that doubles over 10 years and 2% trend
                freq_0 = generator.get_adjusted_frequency(0)  # Base frequency
                freq_10 = generator.get_adjusted_frequency(10)  # Scaled frequency
                print(f"Frequency scaling: {freq_10 / freq_0:.2f}x")
        """
        # Start with base frequency
        adjusted = self.base_frequency

        # Apply exposure multiplier if present
        if self.exposure_base is not None:
            exposure_mult = self.exposure_base.get_frequency_multiplier(float(year))
            adjusted *= exposure_mult

        # Apply trend multiplier (always present, defaults to NoTrend)
        trend_mult = self.frequency_trend.get_multiplier(float(year))
        adjusted *= trend_mult

        return adjusted

    def get_adjusted_severity(self, year: int) -> float:
        """Get severity adjusted for trends at given year.

        Args:
            year: Year number (0-indexed) for severity calculation.

        Returns:
            float: Adjusted severity mean for the specified year.
                Applies trend multiplier to base severity_mean.

        Examples:
            Check severity inflation::

                # With 5% annual inflation trend
                sev_0 = generator.get_adjusted_severity(0)  # Base severity
                sev_10 = generator.get_adjusted_severity(10)  # After 10 years
                print(f"Severity inflation: {sev_10 / sev_0:.2f}x")
        """
        # Apply trend multiplier to base severity
        trend_mult = self.severity_trend.get_multiplier(float(year))
        return self.severity_mean * trend_mult

    @property
    def mean(self) -> float:
        """Analytical expected annual loss.

        For a compound Poisson-Lognormal distribution, the expected annual loss is:
        E[Total Loss] = E[N] * E[X] = base_frequency * severity_mean

        Returns:
            float: Expected annual loss in dollars.

        Note:
            This is the analytical expectation without trends or exposure adjustments.
            For adjusted expectations, use simulation-based methods.

        Examples:
            >>> gen = ClaimGenerator(base_frequency=0.1, severity_mean=5_000_000)
            >>> print(f"Expected annual loss: ${gen.mean:,.0f}")
            Expected annual loss: $500,000
        """
        return self.base_frequency * self.severity_mean

    @property
    def variance(self) -> float:
        """Analytical variance of annual loss.

        For a compound Poisson-Lognormal distribution, the variance is:
        Var[Total Loss] = E[N] * (Var[X] + E[X]²)
                        = base_frequency * (severity_std² + severity_mean²)

        Returns:
            float: Variance of annual loss in dollars squared.

        Note:
            Falls back to simulation when trends or exposure adjustments are present,
            as analytical formulas become complex with time-varying parameters.

        Examples:
            >>> gen = ClaimGenerator(
            ...     base_frequency=0.1,
            ...     severity_mean=5_000_000,
            ...     severity_std=2_000_000
            ... )
            >>> print(f"Variance: {gen.variance:,.0f}")
        """
        # Check if we have trends or exposure adjustments
        has_trends = not isinstance(self.frequency_trend, NoTrend) or not isinstance(
            self.severity_trend, NoTrend
        )
        has_exposure = self.exposure_base is not None

        if has_trends or has_exposure:
            # Fall back to simulation for time-varying parameters
            losses = self._simulate_annual_losses()
            return float(np.var(losses))

        # Analytical formula for constant parameters
        return self.base_frequency * (self.severity_std**2 + self.severity_mean**2)

    @property
    def std(self) -> float:
        """Standard deviation of annual loss.

        Returns:
            float: Standard deviation of annual loss in dollars.

        Examples:
            >>> gen = ClaimGenerator(
            ...     base_frequency=0.1,
            ...     severity_mean=5_000_000,
            ...     severity_std=2_000_000
            ... )
            >>> print(f"Std deviation: ${gen.std:,.0f}")
        """
        return float(np.sqrt(self.variance))

    def generate_year(self, year: int = 0) -> List[ClaimEvent]:
        """Generate claims for a single year.

        Args:
            year: Year number for the claims (default 0). Used as the year
                field in generated ClaimEvent objects.

        Returns:
            List[ClaimEvent]: List of claim events for the specified year.
                Empty list if no claims occur (based on Poisson draw).

        Examples:
            Generate claims for year 5::

                year_claims = generator.generate_year(year=5)
                total = sum(c.amount for c in year_claims)
                print(f"Year 5: {len(year_claims)} claims, ${total:,.0f} total")

        Note:
            This method is useful for step-by-step simulation where claims
            are needed one year at a time, rather than pre-generating all.
        """
        # Generate number of claims for the year with exposure adjustment
        adjusted_frequency = self.get_adjusted_frequency(year)
        n_claims = self.rng.poisson(adjusted_frequency)

        claims = []
        for _ in range(n_claims):
            # Get severity adjusted for trends
            adjusted_severity = self.get_adjusted_severity(year)

            # Generate claim amount
            variance = self.severity_std**2
            mean = adjusted_severity

            if mean > 0:
                sigma = np.sqrt(np.log(1 + variance / mean**2))
                mu = np.log(mean) - sigma**2 / 2
                amount = self.rng.lognormal(mu, sigma)
            else:
                amount = 0.0

            claims.append(ClaimEvent(year=year, amount=amount))

        return claims

    def generate_catastrophic_claims(
        self,
        years: int,
        cat_frequency: float,
        cat_severity_mean: float,
        cat_severity_std: float,
        cat_frequency_trend: Optional[Trend] = None,  # Independent trend for cat frequency
        cat_severity_trend: Optional[Trend] = None,  # Independent trend for cat severity
    ) -> List[ClaimEvent]:
        """Generate catastrophic claims (separate from regular claims).

        Catastrophic events are modeled as rare, high-severity losses using
        Bernoulli trials for occurrence and lognormal for severity.
        Supports independent trends for catastrophic events.

        Args:
            years: Number of years to simulate.
            cat_frequency: Probability of catastrophic event per year.
                Default is 0.01 (1% annual probability).
            cat_severity_mean: Mean catastrophic claim size in dollars.
                Default is $50M.
            cat_severity_std: Standard deviation of catastrophic claim size.
                Default is $20M.
            cat_frequency_trend: Optional independent trend for catastrophic frequency.
                If None, uses main frequency_trend.
            cat_severity_trend: Optional independent trend for catastrophic severity.
                If None, uses main severity_trend.

        Returns:
            List[ClaimEvent]: List of catastrophic claim events. May be empty
                if no catastrophes occur during the simulation period.

        Examples:
            Model tail risk::

                cat_claims = generator.generate_catastrophic_claims(
                    years=100,
                    cat_frequency=0.02,  # 2% annual chance
                    cat_severity_mean=100_000_000  # $100M average
                )

                if cat_claims:
                    print(f"Catastrophes: {len(cat_claims)}")
                    print(f"Largest: ${max(c.amount for c in cat_claims):,.0f}")

        Note:
            Catastrophic claims are generated independently from regular claims.
            Use generate_all_claims() to get both types together.
        """
        # Use provided trends or fall back to main trends
        freq_trend = (
            cat_frequency_trend if cat_frequency_trend is not None else self.frequency_trend
        )
        sev_trend = cat_severity_trend if cat_severity_trend is not None else self.severity_trend

        claims = []

        for year in range(years):
            # Apply trend to catastrophic frequency
            freq_mult = freq_trend.get_multiplier(float(year))
            adjusted_cat_frequency = cat_frequency * freq_mult

            # Check if catastrophic event occurs (Bernoulli trial)
            if self.rng.random() < adjusted_cat_frequency:
                # Apply trend to catastrophic severity
                sev_mult = sev_trend.get_multiplier(float(year))
                adjusted_mean = cat_severity_mean * sev_mult

                # Generate catastrophic claim amount
                variance = cat_severity_std**2
                mean = adjusted_mean

                sigma = np.sqrt(np.log(1 + variance / mean**2))
                mu = np.log(mean) - sigma**2 / 2

                amount = self.rng.lognormal(mu, sigma)
                claims.append(ClaimEvent(year=year, amount=amount))

        return claims

    def generate_all_claims(
        self,
        years: int,
        include_catastrophic: bool = True,
        cat_frequency: float = 0.01,
        cat_severity_mean: float = 50_000_000,
        cat_severity_std: float = 20_000_000,
        cat_frequency_trend: Optional[Trend] = None,
        cat_severity_trend: Optional[Trend] = None,
    ) -> Tuple[List[ClaimEvent], List[ClaimEvent]]:
        """Generate both regular and catastrophic claims.

        Comprehensive claim generation combining attritional (regular) losses
        with potential catastrophic events for complete risk modeling.

        Args:
            years: Number of years to simulate.
            include_catastrophic: Whether to include catastrophic claims.
                Default is True.
            cat_frequency: Probability of catastrophic event per year.
                Default is 0.01 (1% annual probability).
            cat_severity_mean: Mean catastrophic claim size in dollars.
                Default is $50M.
            cat_severity_std: Standard deviation of catastrophic claim size.
                Default is $20M.
            cat_frequency_trend: Optional independent trend for catastrophic frequency.
                If None, uses main frequency_trend.
            cat_severity_trend: Optional independent trend for catastrophic severity.
                If None, uses main severity_trend.

        Returns:
            Tuple[List[ClaimEvent], List[ClaimEvent]]: Tuple containing:
                - regular_claims: List of regular/attritional claims
                - catastrophic_claims: List of catastrophic events

        Examples:
            Full risk assessment::

                regular, catastrophic = generator.generate_all_claims(
                    years=50,
                    include_catastrophic=True,
                    cat_frequency=0.02
                )

                print(f"Regular claims: {len(regular)}")
                print(f"Catastrophic events: {len(catastrophic)}")

                # Combine for total exposure
                all_claims = regular + catastrophic
                total_loss = sum(c.amount for c in all_claims)

        Note:
            The two claim types are generated independently, allowing for
            years with both regular claims and catastrophic events.
        """
        regular_claims = self.generate_claims(years)

        if include_catastrophic:
            catastrophic_claims = self.generate_catastrophic_claims(
                years,
                cat_frequency,
                cat_severity_mean,
                cat_severity_std,
                cat_frequency_trend,
                cat_severity_trend,
            )
        else:
            catastrophic_claims = []

        return regular_claims, catastrophic_claims

    def reset_seed(self, seed: int) -> None:
        """Reset the random seed for reproducibility.

        Allows changing the random seed after initialization for running
        multiple scenarios with different random outcomes.

        Args:
            seed: New random seed to use. Integer value for reproducibility.

        Examples:
            Run multiple scenarios::

                generator = ClaimGenerator(frequency=0.1)

                scenarios = {}
                for seed in range(10):
                    generator.reset_seed(seed)
                    claims = generator.generate_claims(years=20)
                    scenarios[seed] = sum(c.amount for c in claims)

                print(f"Mean loss: ${np.mean(list(scenarios.values())):,.0f}")

        Side Effects:
            Creates a new RandomState object, resetting the random sequence.
        """
        self.rng = np.random.RandomState(seed)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate the simulation cache when parameters change."""
        self._simulation_cache = None

    def _simulate_annual_losses(self) -> np.ndarray:
        """Run Monte Carlo simulation to generate annual loss amounts.

        Returns:
            np.ndarray: Array of simulated annual losses.

        Note:
            Results are cached to avoid recomputation for repeated calls.
            Cache is invalidated when parameters change.
        """
        # Check if we have cached results
        if self._simulation_cache is not None and "annual_losses" in self._simulation_cache:
            return np.array(self._simulation_cache["annual_losses"])

        # Run simulations
        annual_losses = np.zeros(self.n_simulations)

        # Save current RNG state to restore later
        current_state = self.rng.get_state()

        # Use a fixed seed for simulation reproducibility
        sim_rng = np.random.RandomState(42)

        for i in range(self.n_simulations):
            # Generate claims for one year
            n_claims = sim_rng.poisson(self.base_frequency)

            if n_claims > 0:
                # Generate claim amounts using lognormal distribution
                variance = self.severity_std**2
                mean = self.severity_mean

                if mean > 0:
                    sigma = np.sqrt(np.log(1 + variance / mean**2))
                    mu = np.log(mean) - sigma**2 / 2
                    amounts = sim_rng.lognormal(mu, sigma, n_claims)
                    annual_losses[i] = np.sum(amounts)

        # Restore RNG state
        self.rng.set_state(current_state)

        # Cache the results
        if self._simulation_cache is None:
            self._simulation_cache = {}
        self._simulation_cache["annual_losses"] = annual_losses

        return annual_losses

    def get_percentiles(self, percentiles: Optional[List[float]] = None) -> Dict[float, float]:
        """Calculate percentiles of annual loss distribution using Monte Carlo simulation.

        Args:
            percentiles: List of percentile values to calculate (0-100).
                Default is [50, 95, 99] if None.

        Returns:
            Dict[float, float]: Dictionary mapping percentile values to loss amounts.

        Examples:
            >>> gen = ClaimGenerator(
            ...     base_frequency=0.1,
            ...     severity_mean=5_000_000,
            ...     severity_std=2_000_000
            ... )
            >>> p = gen.get_percentiles([50, 90, 95, 99])
            >>> print(f"95th percentile: ${p[95]:,.0f}")

        Note:
            Results are cached for repeated calls with same parameters.
            Cache is invalidated when generator parameters change.
        """
        if percentiles is None:
            percentiles = [50, 95, 99]

        # Get simulated annual losses
        annual_losses = self._simulate_annual_losses()

        # Calculate percentiles
        result = {}
        for p in percentiles:
            if not 0 <= p <= 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {p}")
            result[p] = float(np.percentile(annual_losses, p))

        return result

    def get_cvar(self, percentiles: Optional[List[float]] = None) -> Dict[float, float]:
        """Calculate Conditional Value at Risk (CVaR) for given percentiles.

        CVaR represents the expected loss given that the loss exceeds the
        Value at Risk (VaR) threshold at the specified percentile.

        Args:
            percentiles: List of percentile values for CVaR calculation (0-100).
                Default is [95, 99] if None.

        Returns:
            Dict[float, float]: Dictionary mapping percentile values to CVaR amounts.

        Examples:
            >>> gen = ClaimGenerator(
            ...     base_frequency=0.1,
            ...     severity_mean=5_000_000,
            ...     severity_std=2_000_000
            ... )
            >>> cvar = gen.get_cvar([95, 99])
            >>> print(f"CVaR 95%: ${cvar[95]:,.0f}")

        Note:
            CVaR is also known as Conditional Tail Expectation (CTE) or
            Expected Shortfall (ES). It provides a more complete picture of
            tail risk than VaR alone.
        """
        if percentiles is None:
            percentiles = [95, 99]

        # Get simulated annual losses
        annual_losses = self._simulate_annual_losses()

        # Calculate CVaR for each percentile
        result = {}
        for p in percentiles:
            if not 0 <= p <= 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {p}")

            # Get the VaR threshold
            var_threshold = np.percentile(annual_losses, p)

            # Calculate mean of losses exceeding the threshold
            tail_losses = annual_losses[annual_losses >= var_threshold]

            if len(tail_losses) > 0:
                result[p] = float(np.mean(tail_losses))
            else:
                # Edge case: no losses exceed threshold
                result[p] = var_threshold

        return result

    def generate_enhanced_claims(
        self, years: int, revenue: Optional[float] = None, use_enhanced_distributions: bool = True
    ) -> Tuple[List[ClaimEvent], dict]:
        """Generate claims using enhanced loss distributions if available.

        This method provides integration with the advanced loss_distributions module
        for more sophisticated risk modeling including revenue-dependent frequencies
        and multiple loss types (attritional, large, catastrophic).

        Args:
            years: Number of years to simulate.
            revenue: Current revenue level for frequency scaling.
                     If None, uses $10M baseline.
            use_enhanced_distributions: Whether to use enhanced distributions
                                      if available.

        Returns:
            Tuple of (claim_events, statistics_dict).
            Falls back to standard generation if enhanced not available.
        """
        if not ENHANCED_DISTRIBUTIONS_AVAILABLE or not use_enhanced_distributions:
            # Fall back to standard generation
            regular, catastrophic = self.generate_all_claims(years)
            all_claims = regular + catastrophic

            stats = {
                "total_losses": len(all_claims),
                "regular_count": len(regular),
                "catastrophic_count": len(catastrophic),
                "total_amount": sum(c.amount for c in all_claims),
                "method": "standard",
            }
            return all_claims, stats

        # Use enhanced distributions
        if revenue is None:
            revenue = 10_000_000  # Default $10M

        # Create manufacturing loss generator with appropriate parameters
        seed = int(self.rng.get_state()[1][0]) if hasattr(self.rng, "get_state") else None  # type: ignore

        generator = ManufacturingLossGenerator(
            attritional_params={
                "base_frequency": self.base_frequency * 10,  # Scale to attritional
                "severity_mean": self.severity_mean / 100,  # Smaller losses
                "severity_cv": 1.5,
            },
            large_params={
                "base_frequency": self.base_frequency,
                "severity_mean": self.severity_mean,
                "severity_cv": 2.0,
            },
            seed=seed,
        )
        enhanced_losses, stats = generator.generate_losses(
            duration=years, revenue=revenue, include_catastrophic=True
        )

        # Convert enhanced losses to ClaimEvents
        claims = []
        for loss in enhanced_losses:
            # Convert continuous time to discrete year
            year = int(loss.time)
            if year < years:  # Ensure within simulation period
                claims.append(ClaimEvent(year=year, amount=loss.amount))

        stats["method"] = "enhanced"
        return claims, stats

    def to_loss_data(self, claims: List[ClaimEvent]) -> "LossData":
        """Convert ClaimEvents to standardized LossData format.

        Transforms the simple ClaimEvent format to the richer LossData
        structure used by advanced loss distribution modules.

        Args:
            claims: List of ClaimEvent objects to convert.

        Returns:
            LossData: Standardized loss data structure containing timestamps,
                amounts, and metadata about the claims.

        Raises:
            ImportError: If LossData class is not available (enhanced
                distributions not installed).

        Examples:
            Convert for advanced analysis::

                claims = generator.generate_claims(years=10)
                loss_data = generator.to_loss_data(claims)

                # Use with advanced analytics
                analyzer = LossAnalyzer(loss_data)
                stats = analyzer.compute_statistics()

        Note:
            The conversion preserves all claim information and adds metadata
            about the generator configuration for traceability.
        """
        if not ENHANCED_DISTRIBUTIONS_AVAILABLE or LossData is None:
            raise ImportError("LossData not available. Install enhanced distributions.")

        if not claims:
            return LossData()

        # Extract data from claims
        timestamps = np.array([float(c.year) for c in claims])
        amounts = np.array([c.amount for c in claims])

        # Sort by time
        sort_idx = np.argsort(timestamps)

        return LossData(
            timestamps=timestamps[sort_idx],
            loss_amounts=amounts[sort_idx],
            loss_types=["claim"] * len(claims),
            claim_ids=[f"claim_{i}" for i in range(len(claims))],
            metadata={
                "source": "claim_generator",
                "generator_type": self.__class__.__name__,
                "base_frequency": self.base_frequency,
                "severity_mean": self.severity_mean,
                "severity_std": self.severity_std,
            },
        )

    @staticmethod
    def from_loss_data(loss_data: "LossData") -> List[ClaimEvent]:
        """Convert LossData to ClaimEvent list.

        Static method to transform standardized LossData back to the simpler
        ClaimEvent format used by the simulation framework.

        Args:
            loss_data: Standardized loss data structure containing loss
                timestamps and amounts.

        Returns:
            List[ClaimEvent]: List of ClaimEvent objects with year and amount.

        Examples:
            Import external loss data::

                # Load loss data from external source
                loss_data = load_loss_data('historical_losses.pkl')

                # Convert to ClaimEvents for simulation
                claims = ClaimGenerator.from_loss_data(loss_data)

                # Use in simulation
                sim = Simulation(manufacturer)
                results = sim.run_with_claims(claims)

        Note:
            Timestamps are converted to integer years by truncation.
            Sub-year timing information is lost in the conversion.
        """
        claims = []
        for time, amount in zip(loss_data.timestamps, loss_data.loss_amounts):
            year = int(time)
            claims.append(ClaimEvent(year=year, amount=amount))
        return claims

    def generate_loss_data(self, years: int, include_catastrophic: bool = True) -> "LossData":
        """Generate claims and return as standardized LossData.

        Convenience method that generates claims and immediately converts
        them to the standardized LossData format for advanced analysis.

        Args:
            years: Number of years to simulate.
            include_catastrophic: Whether to include catastrophic events.
                Default is True.

        Returns:
            LossData: Standardized loss data structure with all generated
                claims (both regular and catastrophic if included).

        Raises:
            ImportError: If LossData class is not available.

        Examples:
            Generate for advanced analysis::

                loss_data = generator.generate_loss_data(
                    years=100,
                    include_catastrophic=True
                )

                # Use with specialized tools
                risk_metrics = calculate_var_cvar(loss_data)
                tail_analysis = perform_evt_analysis(loss_data)

        Note:
            This method combines regular and catastrophic claims into a
            single LossData object for unified analysis.
        """
        regular, catastrophic = self.generate_all_claims(
            years, include_catastrophic=include_catastrophic
        )
        all_claims = regular + catastrophic
        return self.to_loss_data(all_claims)

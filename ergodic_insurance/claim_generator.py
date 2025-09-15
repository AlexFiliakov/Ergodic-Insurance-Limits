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
            frequency=0.1,  # 10% chance per year
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import enhanced distributions if available (backward compatibility)
try:
    from .loss_distributions import LossData
    from .loss_distributions import LossEvent as EnhancedLossEvent
    from .loss_distributions import ManufacturingLossGenerator

    ENHANCED_DISTRIBUTIONS_AVAILABLE = True
except ImportError:
    ENHANCED_DISTRIBUTIONS_AVAILABLE = False
    LossData = None  # type: ignore


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

    Attributes:
        frequency: Expected number of claims per year (Poisson lambda).
        severity_mean: Mean claim size in dollars.
        severity_std: Standard deviation of claim size.
        rng: Random number generator for reproducibility.

    Examples:
        Generate claims for a decade::

            generator = ClaimGenerator(
                frequency=0.2,  # 20% expected frequency
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
        frequency: float = 0.1,  # Expected claims per year
        severity_mean: float = 5_000_000,  # Mean claim size
        severity_std: float = 2_000_000,  # Std dev of claim size
        seed: Optional[int] = None,
    ):
        """Initialize claim generator.

        Args:
            frequency: Expected number of claims per year (Poisson parameter).
                Must be non-negative. Default is 0.1 (10% expected frequency).
            severity_mean: Mean claim size in dollars (lognormal parameter).
                Must be positive. Default is $5M.
            severity_std: Standard deviation of claim size. Must be non-negative.
                Default is $2M.
            seed: Random seed for reproducibility. If None, uses random state.

        Raises:
            ValueError: If frequency is negative or severity parameters are invalid.

        Examples:
            Create generator with custom parameters::

                generator = ClaimGenerator(
                    frequency=0.15,  # 15% frequency
                    severity_mean=8_000_000,  # $8M mean
                    severity_std=4_000_000,  # $4M std dev
                    seed=12345
                )
        """
        if frequency < 0:
            raise ValueError(f"Frequency must be non-negative, got {frequency}")
        if severity_mean <= 0:
            raise ValueError(f"Severity mean must be positive, got {severity_mean}")
        if severity_std < 0:
            raise ValueError(f"Severity std must be non-negative, got {severity_std}")

        self.frequency = frequency
        self.severity_mean = severity_mean
        self.severity_std = severity_std
        self.rng = np.random.RandomState(seed)

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
        if years <= 0 or self.frequency <= 0:
            return claims

        for year in range(years):
            # Number of claims this year (Poisson distribution)
            n_claims = self.rng.poisson(self.frequency)

            for _ in range(n_claims):
                # Claim severity (lognormal distribution)
                # Convert mean/std to lognormal parameters
                variance = self.severity_std**2
                mean = self.severity_mean

                # Lognormal parameters
                sigma = np.sqrt(np.log(1 + variance / mean**2))
                mu = np.log(mean) - sigma**2 / 2

                amount = self.rng.lognormal(mu, sigma)
                claims.append(ClaimEvent(year=year, amount=amount))

        return claims

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
        # Generate number of claims for the year
        n_claims = self.rng.poisson(self.frequency)

        claims = []
        for _ in range(n_claims):
            # Generate claim amount
            variance = self.severity_std**2
            mean = self.severity_mean

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
        cat_frequency: float = 0.01,  # 1% chance per year
        cat_severity_mean: float = 50_000_000,
        cat_severity_std: float = 20_000_000,
    ) -> List[ClaimEvent]:
        """Generate catastrophic claims (separate from regular claims).

        Catastrophic events are modeled as rare, high-severity losses using
        Bernoulli trials for occurrence and lognormal for severity.

        Args:
            years: Number of years to simulate.
            cat_frequency: Probability of catastrophic event per year.
                Default is 0.01 (1% annual probability).
            cat_severity_mean: Mean catastrophic claim size in dollars.
                Default is $50M.
            cat_severity_std: Standard deviation of catastrophic claim size.
                Default is $20M.

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
        claims = []

        for year in range(years):
            # Check if catastrophic event occurs (Bernoulli trial)
            if self.rng.random() < cat_frequency:
                # Generate catastrophic claim amount
                variance = cat_severity_std**2
                mean = cat_severity_mean

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
                years, cat_frequency, cat_severity_mean, cat_severity_std
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
                "base_frequency": self.frequency * 10,  # Scale to attritional
                "severity_mean": self.severity_mean / 100,  # Smaller losses
                "severity_cv": 1.5,
            },
            large_params={
                "base_frequency": self.frequency,
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
                "frequency": self.frequency,
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

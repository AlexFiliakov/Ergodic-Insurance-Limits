"""Claim generation for insurance simulations.

This module provides classes for generating realistic insurance claims
with configurable frequency and severity distributions, supporting
both regular and catastrophic event modeling.

The module now integrates with the enhanced loss_distributions module
for more sophisticated risk modeling with revenue-dependent frequencies
and parametric severity distributions.
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
    """

    year: int
    amount: float


class ClaimGenerator:
    """Generate insurance claims for simulations.

    This class generates realistic insurance claims using Poisson processes
    for frequency and lognormal distributions for severity, with support for
    both regular attritional losses and catastrophic events.
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
            severity_mean: Mean claim size (lognormal parameter).
            severity_std: Standard deviation of claim size.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If frequency is negative or severity parameters are invalid.
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

    def generate_claims(self, years: int) -> List[ClaimEvent]:
        """Generate claims for a simulation period.

        Args:
            years: Number of years to simulate.

        Returns:
            List of claim events.
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
            year: Year number for the claims (default 0).

        Returns:
            List of claim events for the year.
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

        Args:
            years: Number of years to simulate.
            cat_frequency: Probability of catastrophic event per year.
            cat_severity_mean: Mean catastrophic claim size.
            cat_severity_std: Std dev of catastrophic claim size.

        Returns:
            List of catastrophic claim events.
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

        Args:
            years: Number of years to simulate.
            include_catastrophic: Whether to include catastrophic claims.
            cat_frequency: Probability of catastrophic event per year.
            cat_severity_mean: Mean catastrophic claim size.
            cat_severity_std: Std dev of catastrophic claim size.

        Returns:
            Tuple of (regular_claims, catastrophic_claims).
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

        Args:
            seed: New random seed to use.
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

        Args:
            claims: List of ClaimEvent objects.

        Returns:
            LossData instance with claim information.
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

        Args:
            loss_data: Standardized loss data.

        Returns:
            List of ClaimEvent objects.
        """
        claims = []
        for time, amount in zip(loss_data.timestamps, loss_data.loss_amounts):
            year = int(time)
            claims.append(ClaimEvent(year=year, amount=amount))
        return claims

    def generate_loss_data(self, years: int, include_catastrophic: bool = True) -> "LossData":
        """Generate claims and return as standardized LossData.

        Args:
            years: Number of years to simulate.
            include_catastrophic: Whether to include catastrophic events.

        Returns:
            LossData instance with generated claims.
        """
        regular, catastrophic = self.generate_all_claims(
            years, include_catastrophic=include_catastrophic
        )
        all_claims = regular + catastrophic
        return self.to_loss_data(all_claims)

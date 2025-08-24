"""Claim generation for insurance simulations.

This module provides classes for generating realistic insurance claims
with configurable frequency and severity distributions, supporting
both regular and catastrophic event modeling.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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

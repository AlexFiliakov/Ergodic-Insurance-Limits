"""Wrapper for ClaimDevelopment to provide test-compatible API."""

from typing import List, Optional

from ergodic_insurance.claim_generator import ClaimEvent


class ClaimDevelopmentWrapper:
    """Wrapper to provide test-compatible API for ClaimDevelopment."""

    def __init__(self, pattern: Optional[List[float]] = None, ultimate_factor: float = 1.0):
        """Initialize wrapper with development pattern.

        Args:
            pattern: List of payment percentages by year
            ultimate_factor: Factor for total ultimate claim amount
        """
        self.pattern = pattern or [0.6, 0.3, 0.1]
        self.ultimate_factor = ultimate_factor

    def develop_claims(self, claims: List[ClaimEvent]) -> List[List[ClaimEvent]]:
        """Develop claims over multiple years based on pattern.

        Args:
            claims: List of initial claims

        Returns:
            List of claim lists by year
        """
        developed = []

        for year_idx, pattern_value in enumerate(self.pattern):
            year_claims = []
            for claim in claims:
                if year_idx < len(self.pattern):
                    amount = claim.amount * pattern_value * self.ultimate_factor
                    if amount > 0:
                        year_claims.append(ClaimEvent(year=claim.year + year_idx, amount=amount))
            if year_claims:
                developed.append(year_claims)

        return developed

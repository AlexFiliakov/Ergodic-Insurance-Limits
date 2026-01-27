"""Wrapper for ClaimDevelopment to provide test-compatible API."""

from typing import List, Optional

from ergodic_insurance.loss_distributions import LossEvent


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

    def develop_losses(self, losses: List[LossEvent]) -> List[List[LossEvent]]:
        """Develop losses over multiple years based on pattern.

        Args:
            losses: List of initial losses

        Returns:
            List of loss lists by year
        """
        developed = []

        for year_idx, pattern_value in enumerate(self.pattern):
            year_losses = []
            for loss in losses:
                if year_idx < len(self.pattern):
                    amount = loss.amount * pattern_value * self.ultimate_factor
                    if amount > 0:
                        base_year = int(loss.time)
                        year_losses.append(
                            LossEvent(
                                time=float(base_year + year_idx),
                                amount=amount,
                                loss_type=loss.loss_type,
                            )
                        )
            if year_losses:
                developed.append(year_losses)

        return developed

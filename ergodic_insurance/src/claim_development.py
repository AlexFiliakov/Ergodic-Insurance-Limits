"""Claim development patterns for cash flow modeling.

This module provides classes for modeling realistic claim payment patterns,
including immediate and long-tail development patterns typical for
manufacturing liability claims. It supports IBNR estimation, reserve
calculations, and cash flow projections.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


class DevelopmentPatternType(Enum):
    """Standard claim development pattern types."""

    IMMEDIATE = "immediate"  # Property/equipment damage
    MEDIUM_TAIL_5YR = "medium_tail_5yr"  # Workers compensation
    LONG_TAIL_10YR = "long_tail_10yr"  # General liability
    VERY_LONG_TAIL_15YR = "very_long_tail_15yr"  # Product liability
    CUSTOM = "custom"  # User-defined pattern


@dataclass
class ClaimDevelopment:
    """Claim development pattern for payment timing.

    This class defines how claim payments develop over time, with
    development factors representing the percentage of total claim
    amount paid in each year.
    """

    pattern_name: str
    development_factors: List[float]  # Payment percentages by year
    tail_factor: float = 0.0  # For claims beyond pattern period

    def __post_init__(self):
        """Validate development pattern."""
        if not self.development_factors:
            raise ValueError("Development factors cannot be empty")

        total = sum(self.development_factors) + self.tail_factor
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Development factors must sum to 1.0 (±0.01), got {total:.4f}")

        if any(f < 0 for f in self.development_factors):
            raise ValueError("Development factors must be non-negative")

        if self.tail_factor < 0:
            raise ValueError(f"Tail factor must be non-negative, got {self.tail_factor}")

    @classmethod
    def create_immediate(cls) -> "ClaimDevelopment":
        """Create immediate payment pattern (property damage)."""
        return cls(
            pattern_name="IMMEDIATE",
            development_factors=[1.0],
            tail_factor=0.0,
        )

    @classmethod
    def create_medium_tail_5yr(cls) -> "ClaimDevelopment":
        """Create 5-year workers compensation pattern."""
        return cls(
            pattern_name="MEDIUM_TAIL_5YR",
            development_factors=[0.40, 0.25, 0.15, 0.10, 0.10],
            tail_factor=0.0,
        )

    @classmethod
    def create_long_tail_10yr(cls) -> "ClaimDevelopment":
        """Create 10-year general liability pattern."""
        return cls(
            pattern_name="LONG_TAIL_10YR",
            development_factors=[
                0.10,  # Year 1
                0.20,  # Year 2
                0.20,  # Year 3
                0.15,  # Year 4
                0.10,  # Year 5
                0.08,  # Year 6
                0.07,  # Year 7
                0.05,  # Year 8
                0.03,  # Year 9
                0.02,  # Year 10
            ],
            tail_factor=0.0,
        )

    @classmethod
    def create_very_long_tail_15yr(cls) -> "ClaimDevelopment":
        """Create 15-year product liability pattern."""
        return cls(
            pattern_name="VERY_LONG_TAIL_15YR",
            development_factors=[
                0.05,  # Year 1
                0.10,  # Year 2
                0.15,  # Year 3
                0.15,  # Year 4
                0.12,  # Year 5
                0.10,  # Year 6
                0.08,  # Year 7
                0.06,  # Year 8
                0.05,  # Year 9
                0.04,  # Year 10
                0.03,  # Year 11
                0.03,  # Year 12
                0.02,  # Year 13
                0.01,  # Year 14
                0.01,  # Year 15
            ],
            tail_factor=0.0,
        )

    def calculate_payments(
        self, claim_amount: float, accident_year: int, payment_year: int
    ) -> float:
        """Calculate payment amount for a specific year.

        Args:
            claim_amount: Total claim amount.
            accident_year: Year when claim occurred.
            payment_year: Year for which to calculate payment.

        Returns:
            Payment amount for the specified year.
        """
        if payment_year < accident_year:
            return 0.0

        development_year = payment_year - accident_year

        if development_year < len(self.development_factors):
            return claim_amount * self.development_factors[development_year]
        if self.tail_factor > 0:
            # Apply tail factor for years beyond pattern
            return claim_amount * self.tail_factor
        return 0.0

    def get_cumulative_paid(self, years_since_accident: int) -> float:
        """Get cumulative percentage paid by year.

        Args:
            years_since_accident: Number of years since accident.

        Returns:
            Cumulative percentage paid (0-1).
        """
        if years_since_accident <= 0:
            return 0.0

        cumulative = sum(
            self.development_factors[: min(years_since_accident, len(self.development_factors))]
        )

        if years_since_accident > len(self.development_factors) and self.tail_factor > 0:
            cumulative += self.tail_factor * (years_since_accident - len(self.development_factors))

        return min(cumulative, 1.0)


@dataclass
class Claim:
    """Individual claim with development tracking."""

    claim_id: str
    accident_year: int
    reported_year: int
    initial_estimate: float
    claim_type: str = "general_liability"
    development_pattern: Optional[ClaimDevelopment] = None
    payments_made: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default development pattern if not provided."""
        if self.development_pattern is None:
            # Default to general liability pattern
            self.development_pattern = ClaimDevelopment.create_long_tail_10yr()

    def record_payment(self, year: int, amount: float):
        """Record a payment made for this claim.

        Args:
            year: Year of payment.
            amount: Payment amount.
        """
        if year in self.payments_made:
            self.payments_made[year] += amount
        else:
            self.payments_made[year] = amount

    def get_total_paid(self) -> float:
        """Get total amount paid to date."""
        return sum(self.payments_made.values())

    def get_outstanding_reserve(self) -> float:
        """Calculate outstanding reserve requirement."""
        return max(0, self.initial_estimate - self.get_total_paid())


@dataclass
class ClaimCohort:
    """Cohort of claims from the same accident year."""

    accident_year: int
    claims: List[Claim] = field(default_factory=list)

    def add_claim(self, claim: Claim):
        """Add a claim to the cohort.

        Args:
            claim: Claim to add.

        Raises:
            ValueError: If claim is from different accident year.
        """
        if claim.accident_year != self.accident_year:
            raise ValueError(
                f"Claim from year {claim.accident_year} cannot be added to "
                f"cohort for year {self.accident_year}"
            )
        self.claims.append(claim)

    def calculate_payments(self, payment_year: int) -> float:
        """Calculate total payments for a specific year.

        Args:
            payment_year: Year for which to calculate payments.

        Returns:
            Total payment amount for the year.
        """
        total_payment = 0.0
        for claim in self.claims:
            if claim.development_pattern:
                payment = claim.development_pattern.calculate_payments(
                    claim.initial_estimate, claim.accident_year, payment_year
                )
                if payment > 0:
                    claim.record_payment(payment_year, payment)
                    total_payment += payment
        return total_payment

    def get_total_incurred(self) -> float:
        """Get total incurred amount for the cohort."""
        return sum(claim.initial_estimate for claim in self.claims)

    def get_total_paid(self) -> float:
        """Get total amount paid for the cohort."""
        return sum(claim.get_total_paid() for claim in self.claims)

    def get_outstanding_reserve(self) -> float:
        """Get total outstanding reserve for the cohort."""
        return sum(claim.get_outstanding_reserve() for claim in self.claims)


class CashFlowProjector:
    """Project cash flows based on claim development patterns."""

    def __init__(self, discount_rate: float = 0.03):
        """Initialize cash flow projector.

        Args:
            discount_rate: Annual discount rate for present value calculations.
        """
        self.discount_rate = discount_rate
        self.cohorts: Dict[int, ClaimCohort] = {}

    def add_cohort(self, cohort: ClaimCohort):
        """Add a claim cohort to the projector.

        Args:
            cohort: Claim cohort to add.
        """
        self.cohorts[cohort.accident_year] = cohort

    def project_payments(self, start_year: int, end_year: int) -> Dict[int, float]:
        """Project claim payments for a range of years.

        Args:
            start_year: First year of projection.
            end_year: Last year of projection.

        Returns:
            Dictionary mapping years to payment amounts.
        """
        payments = {}
        for year in range(start_year, end_year + 1):
            annual_payment = 0.0
            for cohort in self.cohorts.values():
                annual_payment += cohort.calculate_payments(year)
            payments[year] = annual_payment
        return payments

    def calculate_present_value(self, payments: Dict[int, float], base_year: int) -> float:
        """Calculate present value of future payments.

        Args:
            payments: Dictionary of year to payment amount.
            base_year: Year to discount to.

        Returns:
            Present value of all payments.
        """
        pv = 0.0
        for year, amount in payments.items():
            years_to_discount = year - base_year
            if years_to_discount >= 0:
                pv += amount / ((1 + self.discount_rate) ** years_to_discount)
        return pv

    def estimate_ibnr(self, evaluation_year: int, reporting_lag: int = 3) -> float:
        """Estimate IBNR using simplified chain-ladder method.

        Args:
            evaluation_year: Current evaluation year.
            reporting_lag: Average months for claim reporting.

        Returns:
            Estimated IBNR amount.
        """
        # Simplified IBNR estimation
        # In practice, would use full development triangles
        ibnr = 0.0

        for accident_year, cohort in self.cohorts.items():
            if accident_year <= evaluation_year:
                development_years = evaluation_year - accident_year

                # Estimate unreported claims based on reporting pattern
                if development_years < reporting_lag / 12:
                    # Recent accident year - significant IBNR
                    expected_ultimate = cohort.get_total_incurred() * 1.2
                    ibnr += expected_ultimate - cohort.get_total_incurred()
                elif development_years < 2:
                    # Some late-reported claims expected
                    expected_ultimate = cohort.get_total_incurred() * 1.05
                    ibnr += expected_ultimate - cohort.get_total_incurred()

        return ibnr

    def calculate_total_reserves(self, evaluation_year: int) -> Dict[str, float]:
        """Calculate total reserve requirements.

        Args:
            evaluation_year: Current evaluation year.

        Returns:
            Dictionary with case reserves, IBNR, and total.
        """
        case_reserves = sum(cohort.get_outstanding_reserve() for cohort in self.cohorts.values())

        ibnr = self.estimate_ibnr(evaluation_year)

        return {
            "case_reserves": case_reserves,
            "ibnr": ibnr,
            "total_reserves": case_reserves + ibnr,
        }


def load_development_patterns(file_path: str) -> Dict[str, ClaimDevelopment]:
    """Load development patterns from YAML configuration.

    Args:
        file_path: Path to YAML configuration file.

    Returns:
        Dictionary mapping pattern names to ClaimDevelopment objects.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    patterns = {}
    for name, data in config.get("development_patterns", {}).items():
        patterns[name] = ClaimDevelopment(
            pattern_name=name,
            development_factors=data["factors"],
            tail_factor=data.get("tail_factor", 0.0),
        )

    return patterns

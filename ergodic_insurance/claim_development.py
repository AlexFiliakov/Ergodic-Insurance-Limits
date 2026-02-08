"""Claim development patterns for cash flow modeling.

This module provides classes for modeling realistic claim payment patterns,
including immediate and long-tail development patterns typical for
manufacturing liability claims. It supports IBNR estimation, reserve
calculations, and cash flow projections.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional

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
        """Validate development pattern.

        Raises:
            ValueError: If development factors are invalid or don't sum to 1.0.
        """
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
        """Create immediate payment pattern (property damage).

        Returns:
            ClaimDevelopment with immediate payment pattern.
        """
        return cls(
            pattern_name="IMMEDIATE",
            development_factors=[1.0],
            tail_factor=0.0,
        )

    @classmethod
    def create_medium_tail_5yr(cls) -> "ClaimDevelopment":
        """Create 5-year workers compensation pattern.

        Returns:
            ClaimDevelopment with 5-year workers compensation pattern.
        """
        return cls(
            pattern_name="MEDIUM_TAIL_5YR",
            development_factors=[0.40, 0.25, 0.15, 0.10, 0.10],
            tail_factor=0.0,
        )

    @classmethod
    def create_long_tail_10yr(cls) -> "ClaimDevelopment":
        """Create 10-year general liability pattern.

        Returns:
            ClaimDevelopment with 10-year general liability pattern.
        """
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
        """Create 15-year product liability pattern.

        Returns:
            ClaimDevelopment with 15-year product liability pattern.
        """
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
        """Set default development pattern if not provided.

        Uses general liability pattern as default if no pattern is specified.
        """
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
        """Get total amount paid to date.

        Returns:
            Sum of all payments made on this claim.
        """
        return sum(self.payments_made.values())

    def get_outstanding_reserve(self) -> float:
        """Calculate outstanding reserve requirement.

        Returns:
            Outstanding reserve amount (initial estimate minus payments made).
        """
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
        """Get total incurred amount for the cohort.

        Returns:
            Sum of initial estimates for all claims in the cohort.
        """
        return sum(claim.initial_estimate for claim in self.claims)

    def get_total_paid(self) -> float:
        """Get total amount paid for the cohort.

        Returns:
            Sum of all payments made for claims in the cohort.
        """
        return sum(claim.get_total_paid() for claim in self.claims)

    def get_outstanding_reserve(self) -> float:
        """Get total outstanding reserve for the cohort.

        Returns:
            Sum of outstanding reserves for all claims in the cohort.
        """
        return sum(claim.get_outstanding_reserve() for claim in self.claims)


class CashFlowProjector:
    """Project cash flows based on claim development patterns."""

    def __init__(
        self,
        discount_rate: float = 0.03,
        a_priori_loss_ratio: Optional[float] = None,
        ibnr_factors: Optional[Dict[str, float]] = None,
    ):
        """Initialize cash flow projector.

        Args:
            discount_rate: Annual discount rate for present value calculations.
            a_priori_loss_ratio: User-provided expected loss ratio for
                Bornhuetter-Ferguson method (Tier 1 ELR). If None, ELR is
                derived from Cape Cod or industry benchmarks.
            ibnr_factors: Industry benchmark factors by pattern name (Tier 3
                ELR), e.g. {"long_tail_10yr": 1.20}. Loaded from YAML via
                load_ibnr_factors().
        """
        self.discount_rate = discount_rate
        self.a_priori_loss_ratio = a_priori_loss_ratio
        self.ibnr_factors = ibnr_factors or {}
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

    def _get_cohort_pct_developed(self, cohort: ClaimCohort, development_years: int) -> float:
        """Compute weighted-average cumulative development percentage for a cohort.

        Handles mixed development patterns within a cohort by weighting each
        claim's maturity by its size (initial estimate).

        Args:
            cohort: Claim cohort to evaluate.
            development_years: Number of years since accident year.

        Returns:
            Weighted-average percentage developed, clamped to [0.0, 1.0].
        """
        total_incurred = cohort.get_total_incurred()
        if total_incurred <= 0:
            return 0.0
        weighted = sum(
            claim.initial_estimate
            * claim.development_pattern.get_cumulative_paid(development_years)
            for claim in cohort.claims
            if claim.development_pattern
        )
        return min(weighted / total_incurred, 1.0)

    def _resolve_elr_for_cohort(
        self,
        cohort: ClaimCohort,
        development_years: int,
        earned_premium: Optional[Mapping[int, float]] = None,
    ) -> Optional[float]:
        """Resolve Expected Loss Ratio via tiered fallback.

        Tier 1: User-provided a_priori_loss_ratio.
        Tier 2: Cape Cod (Stanard-Buhlmann) from >=2 cohorts with per-cohort
            premium and pct_developed > 0.  Uses the standard formula:
            ``ELR = sum(Paid_i) / sum(Premium_i * pct_developed_i)``.
            Requires ``earned_premium`` to be supplied; without it Cape Cod
            cannot provide an independent exposure measure and is skipped.
        Tier 3: Industry benchmark from ibnr_factors keyed by dominant pattern.
        Tier 4: None (CL-only mode).

        Args:
            cohort: Current cohort being evaluated.
            development_years: Years since accident year for the current cohort.
            earned_premium: Per-cohort earned premium, keyed by accident year.
                Required for Tier 2 (Cape Cod).

        Returns:
            ELR value, or None if no method can produce one.
        """
        # Tier 1: user-provided
        if self.a_priori_loss_ratio is not None:
            return self.a_priori_loss_ratio

        # Tier 2: Cape Cod from multiple cohorts (requires per-cohort premium)
        if earned_premium:
            eligible: List[tuple] = []
            eval_year = cohort.accident_year + development_years
            for ay, c in self.cohorts.items():
                cohort_premium = earned_premium.get(ay)
                if cohort_premium is None or cohort_premium <= 0:
                    continue
                cohort_dy = eval_year - ay
                if cohort_dy < 0:
                    continue
                paid = c.get_total_paid()
                pct = self._get_cohort_pct_developed(c, cohort_dy)
                if pct > 0:
                    eligible.append((paid, cohort_premium, pct))

            if len(eligible) >= 2:
                total_paid = sum(p for p, _, _ in eligible)
                weighted_premium = sum(prem * pct for _, prem, pct in eligible)
                if weighted_premium > 0 and total_paid > 0:
                    return float(total_paid / weighted_premium)

        # Tier 3: industry benchmark from ibnr_factors
        if self.ibnr_factors:
            # Find the dominant pattern name in the cohort
            pattern_weights: Dict[str, float] = {}
            for claim in cohort.claims:
                if claim.development_pattern:
                    name = claim.development_pattern.pattern_name.lower()
                    pattern_weights[name] = pattern_weights.get(name, 0.0) + claim.initial_estimate
            if pattern_weights:
                dominant = max(pattern_weights, key=pattern_weights.get)  # type: ignore[arg-type]
                if dominant in self.ibnr_factors:
                    return self.ibnr_factors[dominant]

        # Tier 4: no ELR available
        return None

    def estimate_ibnr(
        self,
        evaluation_year: int,
        earned_premium: Optional[Mapping[int, float]] = None,
    ) -> float:
        """Estimate IBNR using maturity-adaptive Chain-Ladder / Bornhuetter-Ferguson blend.

        Per-cohort logic:

        - **Chain-Ladder (CL)** ultimate = paid_to_date / pct_developed.
          Uses actual cumulative payments as the numerator (not case
          estimates), which is the standard paid CL projection.
          ``project_payments()`` should be called before ``estimate_ibnr()``
          so that claims have recorded payments; otherwise CL degrades
          gracefully to ``None`` and BF (or no-method) takes over.
        - **Bornhuetter-Ferguson (BF)** IBNR = ELR * premium * (1 - pct).
          Requires both an ELR (via tiered fallback) *and* per-cohort
          ``earned_premium``.  When premium is unavailable, BF is skipped
          and the blend falls back to CL-only.
        - Blended ultimate uses maturity-adaptive credibility weights:
          CL weight = pct_developed, BF weight = 1 - pct_developed.
        - IBNR floored at 0 per cohort (E7).

        .. note:: **CL in deterministic simulation**

           In the current framework, claims follow their
           ``ClaimDevelopment`` pattern deterministically — actual payments
           match expected payments exactly.  Chain-Ladder's value comes from
           projecting *actual experience that deviates from expected*.  In a
           deterministic world, paid CL always recovers the case estimate as
           ultimate (``paid / pct = initial_estimate``), producing zero IBNR
           for known claims.  CL therefore contributes meaningful signal
           only when actual paid deviates from pattern (e.g. claim
           re-estimation, litigation delays).  If the simulation is extended
           to include stochastic development, CL becomes genuinely valuable.
           Until then, the adaptive blend effectively gives most weight to
           BF for immature years (where it matters most) and CL for mature
           years (where both methods converge anyway).

        Args:
            evaluation_year: Current evaluation year.
            earned_premium: Per-cohort earned premium keyed by accident year,
                e.g. ``{2019: 1_000_000, 2020: 1_100_000}``.  Required for
                Bornhuetter-Ferguson and Cape Cod methods.  When ``None``,
                the blend falls back to CL-only.

        Returns:
            Total estimated IBNR amount across all cohorts.
        """
        ibnr = 0.0

        for accident_year, cohort in self.cohorts.items():
            if accident_year > evaluation_year:
                continue
            if not cohort.claims:
                continue

            dev_years = evaluation_year - accident_year
            incurred = cohort.get_total_incurred()
            pct_developed = self._get_cohort_pct_developed(cohort, dev_years)

            # E5: Fully developed → IBNR = 0
            if pct_developed >= 1.0:
                continue

            # E4: Zero incurred → IBNR = 0
            if incurred <= 0:
                continue

            # Chain-Ladder ultimate (paid CL)
            paid_to_date = cohort.get_total_paid()
            if pct_developed > 0 and paid_to_date > 0:
                cl_ultimate = paid_to_date / pct_developed
            else:
                cl_ultimate = None

            # Bornhuetter-Ferguson ultimate (requires premium)
            elr = self._resolve_elr_for_cohort(cohort, dev_years, earned_premium)
            bf_ultimate: Optional[float] = None
            cohort_premium = earned_premium.get(accident_year) if earned_premium else None
            if elr is not None and cohort_premium is not None:
                bf_ibnr = elr * cohort_premium * (1 - pct_developed)
                bf_ultimate = incurred + bf_ibnr

            # Maturity-adaptive credibility blend (E2)
            if cl_ultimate is not None and bf_ultimate is not None:
                cl_weight = pct_developed
                bf_weight = 1 - pct_developed
                blended_ultimate = cl_weight * cl_ultimate + bf_weight * bf_ultimate
            elif bf_ultimate is not None:
                # BF only (immature year with pct=0, E2)
                blended_ultimate = bf_ultimate
            elif cl_ultimate is not None:
                # CL only (no ELR, Tier 4)
                blended_ultimate = cl_ultimate
            else:
                # E4: no method available
                blended_ultimate = incurred

            # E7: Floor IBNR at 0 per cohort
            cohort_ibnr = max(0.0, blended_ultimate - incurred)
            ibnr += cohort_ibnr

        return ibnr

    def calculate_total_reserves(
        self,
        evaluation_year: int,
        earned_premium: Optional[Mapping[int, float]] = None,
    ) -> Dict[str, float]:
        """Calculate total reserve requirements.

        Args:
            evaluation_year: Current evaluation year.
            earned_premium: Per-cohort earned premium keyed by accident year,
                passed through to ``estimate_ibnr`` for Bornhuetter-Ferguson
                and Cape Cod calculations.

        Returns:
            Dictionary with case reserves, IBNR, and total.
        """
        case_reserves = sum(cohort.get_outstanding_reserve() for cohort in self.cohorts.values())

        ibnr = self.estimate_ibnr(evaluation_year, earned_premium=earned_premium)

        return {
            "case_reserves": case_reserves,
            "ibnr": ibnr,
            "total_reserves": case_reserves + ibnr,
        }


def load_ibnr_factors(file_path: str) -> Dict[str, float]:
    """Load IBNR factors from YAML configuration.

    Reads the ``ibnr_factors`` section from a development-patterns YAML file
    for use as Tier 3 industry-benchmark ELRs in CashFlowProjector.

    Args:
        file_path: Path to YAML configuration file.

    Returns:
        Dictionary mapping pattern names to IBNR factor values.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    factors: Dict[str, float] = config.get("ibnr_factors", {})
    return factors


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

"""Claim liability dataclass for tracking insurance claim payment schedules.

This module contains the ClaimLiability dataclass, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305).
"""

from dataclasses import dataclass, field
from decimal import Decimal
import random
from typing import Any, Dict, List, Optional, Union

try:
    from ergodic_insurance.claim_development import ClaimDevelopment
    from ergodic_insurance.decimal_utils import ZERO, to_decimal
except ImportError:
    try:
        from .claim_development import ClaimDevelopment
        from .decimal_utils import ZERO, to_decimal
    except ImportError:
        from claim_development import ClaimDevelopment  # type: ignore[no-redef]
        from decimal_utils import ZERO, to_decimal  # type: ignore[no-redef]


@dataclass
class ClaimLiability:
    """Represents an outstanding insurance claim liability with payment schedule.

    This class tracks insurance claims that require multi-year payment
    schedules and manages the collateral required to support them. It uses
    the Strategy Pattern with ClaimDevelopment to define payment timing.

    The default development strategy follows a typical long-tail liability pattern
    where claims are paid over 10 years, with higher percentages in early years
    and gradually decreasing payments over time. This pattern is commonly observed
    in general liability and workers' compensation claims.

    Attributes:
        original_amount (Decimal): The original claim amount at inception.
        remaining_amount (Decimal): The unpaid balance of the claim.
        year_incurred (int): The year when the claim was first incurred.
        is_insured (bool): Whether this claim involves insurance coverage.
            True for insured claims (company deductible), False for uninsured claims.
        development_strategy (ClaimDevelopment): Strategy for payment development.
            Default uses ClaimDevelopment.create_long_tail_10yr() which follows:
            - Years 1-3: Front-loaded payments (10%, 20%, 20%)
            - Years 4-6: Moderate payments (15%, 10%, 8%)
            - Years 7-10: Tail payments (7%, 5%, 3%, 2%)

    Examples:
        Create and track a claim liability::

            # Create a $1M claim liability with default 10-year pattern
            claim = ClaimLiability(
                original_amount=1_000_000,
                remaining_amount=1_000_000,
                year_incurred=2023
            )

            # Get payment due in year 2 (1 year after incurred)
            payment_due = claim.get_payment(1)  # Returns 200,000 (20%)

            # Make the payment
            actual_payment = claim.make_payment(payment_due)
            print(f"Remaining: ${claim.remaining_amount:,.2f}")

        Custom development pattern::

            # Create claim with immediate payment pattern
            immediate_claim = ClaimLiability(
                original_amount=500_000,
                remaining_amount=500_000,
                year_incurred=2023,
                development_strategy=ClaimDevelopment.create_immediate()
            )

            # Create claim with custom pattern
            custom_pattern = ClaimDevelopment(
                pattern_name="CUSTOM",
                development_factors=[0.5, 0.3, 0.2]  # 50%, 30%, 20%
            )
            custom_claim = ClaimLiability(
                original_amount=500_000,
                remaining_amount=500_000,
                year_incurred=2023,
                development_strategy=custom_pattern
            )

    Note:
        The development factors should sum to 1.0 for full claim payout.
        Payments beyond the schedule length return 0.0 (unless tail_factor is set).

    See Also:
        :class:`~ergodic_insurance.claim_development.ClaimDevelopment`: For more
        sophisticated claim development patterns including factory methods.
        :class:`~ergodic_insurance.claim_development.Claim`: For comprehensive
        claim tracking with reserving.
    """

    original_amount: Decimal
    remaining_amount: Decimal
    year_incurred: int
    is_insured: bool = True  # Default to insured for backward compatibility
    development_strategy: ClaimDevelopment = field(
        default_factory=ClaimDevelopment.create_long_tail_10yr
    )

    # Reserve re-estimation fields (Issue #470, ASC 944-40-25)
    true_ultimate: Optional[Decimal] = None
    _total_paid: Decimal = field(default_factory=lambda: to_decimal(0), repr=False)
    _noise_std: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Convert amounts to mode-appropriate numeric type (Issue #1142)."""
        # Ensure amounts are compatible with current float/Decimal mode
        self.original_amount = to_decimal(self.original_amount)
        self.remaining_amount = to_decimal(self.remaining_amount)
        if self.true_ultimate is not None:
            self.true_ultimate = to_decimal(self.true_ultimate)

    @property
    def payment_schedule(self) -> List[float]:
        """Return development factors for backward compatibility.

        This property provides access to the development factors from the
        underlying ClaimDevelopment strategy, maintaining API compatibility
        with code that expects a payment_schedule list.

        Returns:
            List[float]: The development factors from the strategy.
        """
        return self.development_strategy.development_factors

    def get_payment(self, years_since_incurred: int) -> Decimal:
        """Calculate payment due for a given year after claim incurred.

        This method delegates to the ClaimDevelopment strategy to calculate
        the scheduled payment amount. It handles boundary conditions gracefully,
        returning zero for negative years or years beyond the development pattern.

        Args:
            years_since_incurred (int): Number of years since the claim was first
                incurred. Must be >= 0. Year 0 represents the year of incurrence.

        Returns:
            Decimal: Payment amount due for this year in absolute dollars. Returns
                ZERO if years_since_incurred is negative or beyond the schedule.

        Examples:
            Calculate payments over time::

                claim = ClaimLiability(1_000_000, 1_000_000, 2020)

                # Year of incurrence (2020)
                payment_0 = claim.get_payment(0)  # $100,000 (10%)

                # One year later (2021)
                payment_1 = claim.get_payment(1)  # $200,000 (20%)

                # Beyond schedule
                payment_20 = claim.get_payment(20)  # $0 (beyond 10-year schedule)

        Note:
            The method delegates to development_strategy.calculate_payments() and
            does not check against remaining balance.
        """
        if years_since_incurred < 0:
            return ZERO
        # Delegate to ClaimDevelopment strategy
        # calculate_payments uses (claim_amount, accident_year, payment_year)
        # Use true_ultimate for actual cash flows when reserve development is active
        claim_amount = float(
            self.true_ultimate if self.true_ultimate is not None else self.original_amount
        )
        payment_year = self.year_incurred + years_since_incurred
        payment = self.development_strategy.calculate_payments(
            claim_amount=claim_amount,
            accident_year=self.year_incurred,
            payment_year=payment_year,
        )
        return to_decimal(payment)

    def make_payment(self, amount: Union[Decimal, float]) -> Decimal:
        """Make a payment against the liability and update remaining balance.

        This method processes a payment against the claim liability, reducing
        the remaining amount. If the requested payment exceeds the remaining
        balance, only the remaining amount is paid.

        Args:
            amount: Requested payment amount in dollars. Should be >= 0.
                Negative amounts are treated as zero payment.

        Returns:
            Decimal: Actual payment made in dollars. May be less than requested
                if insufficient liability remains. Returns ZERO if amount <= 0
                or remaining_amount <= 0.

        Examples:
            Make payments and track remaining balance::

                claim = ClaimLiability(1_000_000, 1_000_000, 2020)

                # Make first scheduled payment
                actual = claim.make_payment(100_000)  # Returns 100_000
                assert claim.remaining_amount == 900_000

                # Try to overpay
                actual = claim.make_payment(2_000_000)  # Returns 900_000
                assert claim.remaining_amount == 0

                # No more payments possible
                actual = claim.make_payment(50_000)  # Returns 0

        Warning:
            This method modifies the claim's remaining_amount. Use with caution
            in concurrent scenarios.
        """
        amount_decimal = to_decimal(amount)
        payment = min(amount_decimal, self.remaining_amount)
        self.remaining_amount -= payment
        self._total_paid += payment
        return payment

    def re_estimate(self, current_year: int, rng: random.Random) -> Decimal:
        """Re-estimate remaining reserve based on claim maturity (ASC 944-40-25).

        As the claim matures, the estimate converges toward the true residual.
        Noise shrinks proportionally to maturity (1 - maturity).

        Args:
            current_year: Current simulation year.
            rng: Random number generator for reproducible noise.

        Returns:
            Decimal: Change in remaining amount (positive=adverse, negative=favorable).
        """
        if self.true_ultimate is None or self.remaining_amount <= ZERO:
            return to_decimal(0)

        # Calculate maturity as fraction of development pattern length
        years_elapsed = current_year - self.year_incurred
        pattern_length = len(self.development_strategy.development_factors)
        maturity = min(max(years_elapsed / pattern_length, 0.0), 1.0)

        # True residual: what's actually left to pay
        true_residual = max(self.true_ultimate - self._total_paid, to_decimal(0))

        # Noise shrinks to zero at full maturity
        noise_std = (1.0 - maturity) * self._noise_std
        if noise_std <= 0.0:
            # At maturity, snap to true residual
            old_remaining = self.remaining_amount
            self.remaining_amount = true_residual
            return self.remaining_amount - old_remaining

        # Apply noise around true residual
        noise_factor = 1.0 + rng.gauss(0.0, noise_std)
        new_remaining = to_decimal(max(float(true_residual) * noise_factor, 0.0))

        old_remaining = self.remaining_amount
        self.remaining_amount = new_remaining
        return new_remaining - old_remaining

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ClaimLiability":
        """Create a deep copy of this claim liability.

        Args:
            memo: Dictionary of already copied objects (for cycle detection)

        Returns:
            Independent copy of this ClaimLiability
        """
        import copy

        # Deep copy the strategy (supports ClaimDevelopment and subclasses
        # like StochasticClaimDevelopment)
        copied_strategy = copy.deepcopy(self.development_strategy, memo)

        result = ClaimLiability(
            original_amount=copy.deepcopy(self.original_amount, memo),
            remaining_amount=copy.deepcopy(self.remaining_amount, memo),
            year_incurred=self.year_incurred,
            is_insured=self.is_insured,
            development_strategy=copied_strategy,
            true_ultimate=copy.deepcopy(self.true_ultimate, memo),
            _noise_std=self._noise_std,
        )
        result._total_paid = copy.deepcopy(self._total_paid, memo)
        return result

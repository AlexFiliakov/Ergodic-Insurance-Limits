"""Decimal utilities for financial calculations.

This module provides utilities for precise financial calculations using Python's
decimal.Decimal type. Using Decimal instead of float prevents accumulation errors
in iterative simulations and ensures accounting identities hold exactly.

Example:
    Convert a float to decimal for financial use::

        from ergodic_insurance.decimal_utils import to_decimal, ZERO

        amount = to_decimal(1234.56)
        if amount != ZERO:
            print(f"Amount: {amount}")
"""

from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Union

# Standard precision for financial calculations (2 decimal places = cents)
CURRENCY_PLACES = Decimal("0.01")

# Common constants
ZERO = Decimal("0.00")
ONE = Decimal("1.00")
PENNY = Decimal("0.01")


def to_decimal(value: Union[float, int, str, Decimal, None]) -> Decimal:
    """Convert a numeric value to Decimal with proper handling.

    Converts floats, ints, strings, or existing Decimals to a standardized
    Decimal value. Floats are converted via string representation to avoid
    binary floating point artifacts.

    Args:
        value: Numeric value to convert. None is converted to ZERO.

    Returns:
        Decimal representation of the value.

    Example:
        >>> to_decimal(1234.56)
        Decimal('1234.56')
        >>> to_decimal(None)
        Decimal('0.00')
    """
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        # Convert via string to avoid float precision issues
        # Round to reasonable precision first to avoid artifacts like 0.1 -> 0.10000000000000001
        return Decimal(str(round(value, 10)))
    return Decimal(value)


def quantize_currency(value: Union[Decimal, float, int]) -> Decimal:
    """Quantize a value to currency precision (2 decimal places).

    Rounds using ROUND_HALF_UP (banker's rounding away from zero for .5 cases)
    which is standard for financial calculations.

    Args:
        value: Numeric value to quantize.

    Returns:
        Decimal rounded to 2 decimal places.

    Example:
        >>> quantize_currency(Decimal("1234.567"))
        Decimal('1234.57')
        >>> quantize_currency(1234.565)
        Decimal('1234.57')
    """
    if not isinstance(value, Decimal):
        value = to_decimal(value)
    return value.quantize(CURRENCY_PLACES, rounding=ROUND_HALF_UP)


def is_zero(value: Union[Decimal, float, int]) -> bool:
    """Check if a value is effectively zero after quantization.

    Useful for balance checks where we need exact equality after
    rounding to currency precision.

    Args:
        value: Numeric value to check.

    Returns:
        True if value rounds to zero at currency precision.

    Example:
        >>> is_zero(Decimal("0.001"))
        True
        >>> is_zero(Decimal("0.01"))
        False
    """
    return quantize_currency(value) == ZERO


def sum_decimals(*values: Union[Decimal, float, int]) -> Decimal:
    """Sum multiple values with Decimal precision.

    Converts all values to Decimal before summing to maintain precision.

    Args:
        *values: Numeric values to sum.

    Returns:
        Decimal sum of all values.

    Example:
        >>> sum_decimals(0.1, 0.2, 0.3)
        Decimal('0.6')
    """
    return sum((to_decimal(v) for v in values), ZERO)


def safe_divide(
    numerator: Union[Decimal, float, int],
    denominator: Union[Decimal, float, int],
    default: Union[Decimal, float, int] = ZERO,
) -> Decimal:
    """Safely divide two values, returning default if denominator is zero.

    Args:
        numerator: Value to divide.
        denominator: Value to divide by.
        default: Value to return if denominator is zero.

    Returns:
        Result of division, or default if denominator is zero.

    Example:
        >>> safe_divide(100, 4)
        Decimal('25')
        >>> safe_divide(100, 0, default=Decimal("-1"))
        Decimal('-1')
    """
    num = to_decimal(numerator)
    denom = to_decimal(denominator)

    if denom == ZERO:
        return to_decimal(default)

    return num / denom

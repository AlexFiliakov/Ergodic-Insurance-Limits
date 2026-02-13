"""Decimal utilities for financial calculations.

This module provides utilities for precise financial calculations using Python's
decimal.Decimal type. Using Decimal instead of float prevents accumulation errors
in iterative simulations and ensures accounting identities hold exactly.

A **float mode** (Issue #1142) can be activated per-thread via
:func:`enable_float_mode`.  When active, :func:`to_decimal` returns
``float`` instead of ``Decimal``, eliminating the overhead of Decimal
arithmetic in Monte Carlo hot paths while keeping the same numeric API.

Example:
    Convert a float to decimal for financial use::

        from ergodic_insurance.decimal_utils import to_decimal, ZERO

        amount = to_decimal(1234.56)
        if amount != ZERO:
            print(f"Amount: {amount}")
"""

from decimal import ROUND_HALF_UP, Decimal
import threading
from typing import Dict, Union

# ---------------------------------------------------------------------------
# Thread-local float mode (Issue #1142)
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def enable_float_mode() -> None:
    """Enable float mode for the current thread.

    When active, :func:`to_decimal` returns ``float`` and
    :func:`quantize_currency` rounds with :func:`round` instead of
    ``Decimal.quantize``.  This eliminates Decimal overhead in Monte
    Carlo hot paths (Issue #1142).
    """
    _thread_local.float_mode = True


def disable_float_mode() -> None:
    """Disable float mode for the current thread (restore Decimal behaviour)."""
    _thread_local.float_mode = False


def is_float_mode() -> bool:
    """Return ``True`` if float mode is active in the current thread."""
    return getattr(_thread_local, "float_mode", False)


# Standard precision for financial calculations (2 decimal places = cents)
CURRENCY_PLACES = Decimal("0.01")

# Common Decimal constants — used in comparison-only contexts.
# For arithmetic contexts that must adapt to float mode, use
# ``to_decimal(0)`` / ``to_decimal(1)`` instead.
ZERO = Decimal("0.00")
ONE = Decimal("1.00")
PENNY = Decimal("0.01")

# Type alias for metrics dictionaries used across the codebase.
# Prefer Decimal for monetary values; float is accepted for backward
# compatibility and is converted to Decimal at calculation boundaries.
MetricsDict = Dict[str, Union[Decimal, float, int, bool]]

# Numeric type returned by to_decimal (Decimal or float depending on mode).
Numeric = Union[Decimal, float]


def to_decimal(value: Union[float, int, str, Decimal, None]) -> Decimal:
    """Convert a numeric value to Decimal (or float in float mode).

    In normal mode, converts floats, ints, strings, or existing Decimals to a
    standardized Decimal value.  Floats are converted via string
    representation to avoid binary floating point artifacts.

    In **float mode** (Issue #1142), returns ``float`` directly, avoiding
    the cost of Decimal construction.

    Args:
        value: Numeric value to convert. None is converted to zero.

    Returns:
        Decimal (or float in float mode) representation of the value.

    Example:
        >>> to_decimal(1234.56)
        Decimal('1234.56')
        >>> to_decimal(None)
        Decimal('0.00')
    """
    if is_float_mode():
        # In float mode, return float instead of Decimal for performance.
        # Type annotation says Decimal for downstream compatibility; at
        # runtime float supports the same arithmetic operators.
        if value is None:
            return 0.0  # type: ignore[return-value]
        if isinstance(value, float):
            return value  # type: ignore[return-value]
        if isinstance(value, int):
            return float(value)  # type: ignore[return-value]
        if isinstance(value, Decimal):
            return float(value)  # type: ignore[return-value]
        # str → float
        return float(value)  # type: ignore[return-value]

    # --- Decimal mode (original) ---
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

    In float mode, uses :func:`round` for speed.

    Args:
        value: Numeric value to quantize.

    Returns:
        Decimal (or float) rounded to 2 decimal places.

    Example:
        >>> quantize_currency(Decimal("1234.567"))
        Decimal('1234.57')
        >>> quantize_currency(1234.565)
        Decimal('1234.57')
    """
    if is_float_mode():
        return round(float(value), 2)  # type: ignore[return-value]
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
    """Sum multiple values with Decimal precision (or float in float mode).

    Converts all values via :func:`to_decimal` before summing.

    Args:
        *values: Numeric values to sum.

    Returns:
        Sum of all values.

    Example:
        >>> sum_decimals(0.1, 0.2, 0.3)
        Decimal('0.6')
    """
    return sum((to_decimal(v) for v in values), to_decimal(0))


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

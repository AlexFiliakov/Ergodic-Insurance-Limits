"""Module-level financial constants for the Ergodic Insurance framework.

Centralizes hardcoded financial values used across multiple modules,
providing a single source of truth for default rates and thresholds.

Since:
    Version 0.9.0 (Issue #314, #458)
"""

# --- Module-level financial constants ---
# Issue #314: Centralized constants to eliminate hardcoded values across modules

DEFAULT_RISK_FREE_RATE: float = 0.02
"""Default risk-free rate (2%) used for Sharpe ratio and risk-adjusted calculations."""

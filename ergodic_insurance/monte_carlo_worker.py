"""Standalone worker function for multiprocessing Monte Carlo simulations."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import ManufacturerConfig
from .decimal_utils import ZERO, quantize_currency, safe_divide, to_decimal
from .insurance_program import InsuranceProgram
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer


def run_chunk_standalone(
    chunk: Tuple[int, int, Optional[int]],
    loss_generator: ManufacturingLossGenerator,
    insurance_program: InsuranceProgram,
    manufacturer: WidgetManufacturer,
    config_dict: Dict[str, Any],
) -> Dict[str, Union[np.ndarray, List[Dict[int, bool]]]]:
    """Standalone function to run a chunk of simulations for multiprocessing.

    This function is independent of the MonteCarloEngine class and can be pickled
    for multiprocessing on all platforms including Windows.

    Args:
        chunk: Tuple of (start_idx, end_idx, seed)
        loss_generator: Loss generator instance
        insurance_program: Insurance program instance
        manufacturer: Manufacturer instance
        config_dict: Configuration dictionary with necessary parameters

    Returns:
        Dictionary with simulation results for the chunk
    """
    start_idx, end_idx, seed = chunk
    n_sims = end_idx - start_idx
    n_years = config_dict["n_years"]
    dtype = np.float32 if config_dict.get("use_float32", False) else np.float64

    # Set seed for this chunk
    if seed is not None:
        np.random.seed(seed)

    # Pre-allocate arrays
    final_assets = np.zeros(n_sims, dtype=dtype)
    annual_losses = np.zeros((n_sims, n_years), dtype=dtype)
    insurance_recoveries = np.zeros((n_sims, n_years), dtype=dtype)
    retained_losses = np.zeros((n_sims, n_years), dtype=dtype)

    # Track periodic ruin if requested
    ruin_evaluation = config_dict.get("ruin_evaluation", None)
    ruin_at_year_all = []

    # Run simulations in chunk
    for i in range(n_sims):
        # Create a deep copy of the manufacturer for this simulation
        # This preserves ALL state including year, history, claims, ledger, etc.
        # See Issue #273 for details on why manual copy was insufficient
        sim_manufacturer = copy.deepcopy(manufacturer)

        # Run single simulation
        sim_annual_losses = np.zeros(n_years, dtype=dtype)
        sim_insurance_recoveries = np.zeros(n_years, dtype=dtype)
        sim_retained_losses = np.zeros(n_years, dtype=dtype)

        # Track ruin at evaluation points for this simulation
        ruin_at_year = {}
        if ruin_evaluation:
            for eval_year in ruin_evaluation:
                if eval_year <= n_years:
                    ruin_at_year[eval_year] = False

        # Extract settings from config_dict
        letter_of_credit_rate = config_dict.get("letter_of_credit_rate", 0.015)
        growth_rate = config_dict.get("growth_rate", 0)
        time_resolution = config_dict.get("time_resolution", "annual")
        apply_stochastic = config_dict.get("apply_stochastic", False)

        for year in range(n_years):
            # Generate losses for the year
            revenue = sim_manufacturer.calculate_revenue()

            # Calculate and pay insurance premiums (scaled by revenue)
            # Premium scales proportionally with revenue growth
            # Use Decimal arithmetic for precision in financial calculations
            initial_revenue_decimal = to_decimal(
                manufacturer.config.initial_assets * manufacturer.config.asset_turnover_ratio
            )
            # Calculate revenue multiplier using Decimal division for precision
            revenue_multiplier = safe_divide(
                to_decimal(revenue), initial_revenue_decimal, default=to_decimal(1)
            )
            # Calculate annual premium using Decimal arithmetic
            base_premium = to_decimal(insurance_program.calculate_annual_premium())
            annual_premium = base_premium * revenue_multiplier

            # Record the insurance premium for accounting purposes
            # The premium will be deducted from operating income in calculate_operating_income
            # Use public method to maintain encapsulation (Issue #276)
            if annual_premium > ZERO:
                sim_manufacturer.record_insurance_premium(annual_premium, is_annual=False)

            # Use ManufacturingLossGenerator to generate losses
            # Note: Loss generator interface requires float for numpy compatibility
            # This is the documented conversion boundary (R2 from issue #278)
            if hasattr(loss_generator, "generate_losses"):
                year_losses, _ = loss_generator.generate_losses(
                    duration=1.0, revenue=float(revenue)
                )
            else:
                raise AttributeError(
                    f"Loss generator {type(loss_generator).__name__} has no generate_losses method"
                )

            # Sum loss amounts using Decimal for precision, then convert to float
            # for numpy array storage (R3 from issue #278)
            total_year_loss_decimal = sum((to_decimal(loss.amount) for loss in year_losses), ZERO)
            sim_annual_losses[year] = float(total_year_loss_decimal)

            # Apply insurance
            # Note: process_claim expects float, this is the documented boundary
            if total_year_loss_decimal > ZERO:
                claim_result = insurance_program.process_claim(float(total_year_loss_decimal))
                # Use Decimal for financial accounting precision (R1 from issue #278)
                recovery_decimal = to_decimal(claim_result["insurance_recovery"])
                retained_decimal = to_decimal(claim_result["deductible_paid"])

                # Record the insurance loss for proper accounting using Decimal
                # The loss will be deducted from operating income in calculate_operating_income
                # Use public method to maintain encapsulation (Issue #276)
                sim_manufacturer.record_insurance_loss(retained_decimal)
            else:
                recovery_decimal = ZERO
                retained_decimal = ZERO

            # Convert to float only at numpy array storage boundary (R3 from issue #278)
            sim_insurance_recoveries[year] = float(recovery_decimal)
            sim_retained_losses[year] = float(retained_decimal)

            # Run business operations (growth, etc.)
            # Note: insurance premiums were already paid above, so pass 0 to avoid double-counting
            sim_manufacturer.step(
                letter_of_credit_rate, growth_rate, time_resolution, apply_stochastic
            )

            # Check for ruin
            if sim_manufacturer.total_assets <= 0:
                # Mark ruin for all future evaluation points
                if ruin_evaluation:
                    for eval_year in ruin_at_year:
                        if year < eval_year:
                            ruin_at_year[eval_year] = True
                break

        # Convert to float at numpy array storage boundary (R3 from issue #278)
        # This is the documented conversion point for simulation results
        final_assets[i] = float(quantize_currency(sim_manufacturer.total_assets))
        annual_losses[i] = sim_annual_losses
        insurance_recoveries[i] = sim_insurance_recoveries
        retained_losses[i] = sim_retained_losses

        # Store periodic ruin data
        if ruin_evaluation:
            ruin_at_year_all.append(ruin_at_year)

    result: Dict[str, Union[np.ndarray, List[Dict[int, bool]]]] = {
        "final_assets": final_assets,
        "annual_losses": annual_losses,
        "insurance_recoveries": insurance_recoveries,
        "retained_losses": retained_losses,
    }

    # Add periodic ruin data if tracked
    if ruin_evaluation:
        result["ruin_at_year"] = ruin_at_year_all

    return result

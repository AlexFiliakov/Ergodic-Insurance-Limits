"""Standalone worker function for multiprocessing Monte Carlo simulations."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import ManufacturerConfig
from .decimal_utils import to_decimal
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

        for year in range(n_years):
            # Generate losses for the year
            revenue = sim_manufacturer.calculate_revenue()

            # Calculate and pay insurance premiums (scaled by revenue)
            # Premium scales proportionally with revenue growth
            initial_revenue = (
                manufacturer.config.initial_assets * manufacturer.config.asset_turnover_ratio
            )
            # Convert revenue to float to avoid Decimal/float mismatch
            revenue_multiplier = float(revenue) / initial_revenue if initial_revenue > 0 else 1.0
            annual_premium = (
                float(insurance_program.calculate_annual_premium()) * revenue_multiplier
            )

            # Set the period insurance premium for accounting purposes
            # The premium will be deducted from operating income in calculate_operating_income
            # Do NOT deduct from cash here to avoid double-counting
            if annual_premium > 0:
                sim_manufacturer.period_insurance_premiums = to_decimal(annual_premium)

            # Use ManufacturingLossGenerator to generate losses
            # Convert revenue to float for loss generator compatibility
            if hasattr(loss_generator, "generate_losses"):
                year_losses, _ = loss_generator.generate_losses(
                    duration=1.0, revenue=float(revenue)
                )
            else:
                raise AttributeError(
                    f"Loss generator {type(loss_generator).__name__} has no generate_losses method"
                )

            # Convert to float in case loss amounts are Decimal
            total_year_loss = float(sum(loss.amount for loss in year_losses))
            sim_annual_losses[year] = total_year_loss

            # Apply insurance
            if total_year_loss > 0:
                claim_result = insurance_program.process_claim(total_year_loss)
                # Convert to float in case insurance values are Decimal
                recovery = float(claim_result["insurance_recovery"])
                # Calculate retained loss as company's payment (deductible + uncovered)
                retained = float(claim_result["deductible_paid"])

                # Record the insurance loss for proper accounting
                # The loss will be deducted from operating income in calculate_operating_income
                # Use += in case there are multiple losses in a year
                sim_manufacturer.period_insurance_losses += to_decimal(retained)
            else:
                recovery = 0.0
                retained = 0.0

            sim_insurance_recoveries[year] = recovery
            sim_retained_losses[year] = retained

            # Run business operations (growth, etc.)
            # Note: insurance premiums were already paid above, so pass 0 to avoid double-counting
            sim_manufacturer.step()

            # Check for ruin
            if sim_manufacturer.total_assets <= 0:
                # Mark ruin for all future evaluation points
                if ruin_evaluation:
                    for eval_year in ruin_at_year:
                        if year < eval_year:
                            ruin_at_year[eval_year] = True
                break

        # Convert to float in case total_assets is Decimal
        final_assets[i] = float(sim_manufacturer.total_assets)
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

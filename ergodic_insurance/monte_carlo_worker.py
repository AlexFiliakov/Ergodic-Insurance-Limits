"""Standalone worker function for multiprocessing Monte Carlo simulations."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import ManufacturerConfig
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
        # Create a copy of the manufacturer for this simulation to avoid state pollution
        sim_manufacturer = WidgetManufacturer(manufacturer.config)
        sim_manufacturer.total_assets = manufacturer.total_assets

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

            # Handle both ClaimGenerator and ManufacturingLossGenerator
            if hasattr(loss_generator, "generate_losses"):
                year_losses, _ = loss_generator.generate_losses(duration=1.0, revenue=revenue)
            elif hasattr(loss_generator, "generate_claims"):
                year_losses = loss_generator.generate_claims(years=1)
            else:
                raise AttributeError(
                    f"Loss generator {type(loss_generator).__name__} has neither generate_losses nor generate_claims method"
                )

            total_year_loss = sum(loss.amount for loss in year_losses)
            sim_annual_losses[year] = total_year_loss

            # Apply insurance
            if total_year_loss > 0:
                claim_result = insurance_program.process_claim(total_year_loss)
                recovery = claim_result["insurance_recovery"]
                # Calculate retained loss as company's payment (deductible + uncovered)
                retained = claim_result["deductible_paid"]

                # Process the claim through manufacturer's claim processing system
                # This properly handles cash flows and asset impacts
                company_payment, insurance_payment = sim_manufacturer.process_insurance_claim(
                    claim_amount=total_year_loss, insurance_recovery=recovery
                )
            else:
                recovery = 0.0
                retained = 0.0

            sim_insurance_recoveries[year] = recovery
            sim_retained_losses[year] = retained

            # Run business operations (growth, etc.)
            sim_manufacturer.step()

            # Check for ruin
            if sim_manufacturer.total_assets <= 0:
                # Mark ruin for all future evaluation points
                if ruin_evaluation:
                    for eval_year in ruin_at_year:
                        if year < eval_year:
                            ruin_at_year[eval_year] = True
                break

        final_assets[i] = sim_manufacturer.total_assets
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

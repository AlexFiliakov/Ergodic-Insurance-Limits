"""Standalone worker function for multiprocessing Monte Carlo simulations."""

from typing import Any, Dict, List, Optional, Tuple

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
) -> Dict[str, np.ndarray]:
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

    # Run simulations in chunk
    for i in range(n_sims):
        # Create a copy of the manufacturer for this simulation to avoid state pollution
        sim_manufacturer = WidgetManufacturer(manufacturer.config)
        sim_manufacturer.total_assets = manufacturer.total_assets

        # Run single simulation
        sim_annual_losses = np.zeros(n_years, dtype=dtype)
        sim_insurance_recoveries = np.zeros(n_years, dtype=dtype)
        sim_retained_losses = np.zeros(n_years, dtype=dtype)

        for year in range(n_years):
            # Generate losses for the year
            year_losses, _ = loss_generator.generate_losses(
                1.0, sim_manufacturer.total_assets * 0.8
            )  # 1 year duration, revenue based on assets
            total_year_loss = sum(loss.amount for loss in year_losses)
            sim_annual_losses[year] = total_year_loss

            # Apply insurance
            if total_year_loss > 0:
                result = insurance_program.process_claim(total_year_loss)
                recovery = result["insurance_recovery"]
            else:
                recovery = 0.0
            sim_insurance_recoveries[year] = recovery

            # Calculate retained loss
            retained = total_year_loss - recovery
            sim_retained_losses[year] = retained

            # Apply loss to manufacturer
            if retained > 0:
                sim_manufacturer.record_insurance_loss(retained)

            # Run business operations (growth, etc.)
            sim_manufacturer.step()

        final_assets[i] = sim_manufacturer.total_assets
        annual_losses[i] = sim_annual_losses
        insurance_recoveries[i] = sim_insurance_recoveries
        retained_losses[i] = sim_retained_losses

    return {
        "final_assets": final_assets,
        "annual_losses": annual_losses,
        "insurance_recoveries": insurance_recoveries,
        "retained_losses": retained_losses,
    }

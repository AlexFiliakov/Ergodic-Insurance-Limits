"""Standalone worker function for multiprocessing Monte Carlo simulations."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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

    # Re-seed the loss generator's internal RandomState objects for this chunk.
    # np.random.seed() only affects the global numpy state and has no effect on
    # the loss generator's per-instance RandomState objects that were pickled
    # from the parent process. Without this, every chunk produces identical
    # loss sequences. See issue #299.
    if seed is not None:
        loss_generator.reseed(seed)

    # Pre-allocate arrays
    final_assets = np.zeros(n_sims, dtype=dtype)
    annual_losses = np.zeros((n_sims, n_years), dtype=dtype)
    insurance_recoveries = np.zeros((n_sims, n_years), dtype=dtype)
    retained_losses = np.zeros((n_sims, n_years), dtype=dtype)

    # Track periodic ruin if requested
    ruin_evaluation = config_dict.get("ruin_evaluation", None)
    ruin_at_year_all = []

    # Pre-copy stochastic process once outside the hot loop (Issue #366).
    # Most runs have stochastic_process=None so this is a no-op.
    _has_stochastic = manufacturer.stochastic_process is not None

    # Run simulations in chunk
    for i in range(n_sims):
        # Create a fresh manufacturer from config instead of deep-copying the
        # entire object graph (Issue #366).  WidgetManufacturer.__init__ builds
        # a clean ledger and balance sheet from config alone — no traversal of
        # the existing manufacturer's ledger entries, caches, or history.
        sim_manufacturer = WidgetManufacturer.create_fresh(
            manufacturer.config,
            stochastic_process=(
                copy.deepcopy(manufacturer.stochastic_process) if _has_stochastic else None
            ),
        )

        # Deep-copy insurance program per simulation to avoid state leakage (Issue #348)
        sim_insurance_program = copy.deepcopy(insurance_program)

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
            # Reset insurance program aggregate limits at start of each policy year (Issue #348)
            if year > 0:
                sim_insurance_program.reset_annual()

            # Generate losses for the year
            revenue = sim_manufacturer.calculate_revenue()

            # Unified ordering: losses → claims → premium → step (Issue #349)

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

            # Sum loss amounts using native float (Issue #368: remove Decimal
            # from hot loop — results are stored in float64 arrays anyway)
            total_loss = sum(loss.amount for loss in year_losses)
            sim_annual_losses[year] = total_loss

            # Apply insurance PER OCCURRENCE (not aggregate) to correctly apply
            # per-occurrence deductibles and limits (Issue #348)
            total_recovery = 0.0
            total_retained = 0.0

            for loss_event in year_losses:
                if loss_event.amount > 0:
                    claim_result = sim_insurance_program.process_claim(loss_event.amount)
                    event_recovery = claim_result["insurance_recovery"]
                    event_retained = loss_event.amount - event_recovery

                    total_recovery += event_recovery
                    total_retained += event_retained

                    # Record the insurance loss for proper accounting
                    # The loss will be deducted from operating income in calculate_operating_income
                    # Use public method to maintain encapsulation (Issue #276)
                    if event_retained > 0:
                        sim_manufacturer.record_insurance_loss(event_retained)

            sim_insurance_recoveries[year] = total_recovery
            sim_retained_losses[year] = total_retained

            # Calculate and pay insurance premiums AFTER claims (Issue #349)
            # Premium scales proportionally with revenue growth
            base_revenue = float(
                manufacturer.config.initial_assets * manufacturer.config.asset_turnover_ratio
            )
            revenue_multiplier = float(revenue) / base_revenue if base_revenue > 0 else 1.0
            base_premium = sim_insurance_program.calculate_annual_premium()
            annual_premium = base_premium * revenue_multiplier

            # Record the insurance premium for accounting purposes
            # The premium will be deducted from operating income in calculate_operating_income
            # Use public method to maintain encapsulation (Issue #276)
            if annual_premium > 0:
                sim_manufacturer.record_insurance_premium(annual_premium, is_annual=False)

            # Run business operations (growth, etc.)
            sim_manufacturer.step(
                letter_of_credit_rate, growth_rate, time_resolution, apply_stochastic
            )

            # Check for ruin using insolvency tolerance consistent with engine
            # (issue #299: unify threshold between worker and engine)
            insolvency_tolerance = config_dict.get("insolvency_tolerance", 10_000)
            if float(sim_manufacturer.equity) <= insolvency_tolerance:
                # Mark ruin for all future evaluation points
                if ruin_evaluation:
                    for eval_year in ruin_at_year:
                        if year < eval_year:
                            ruin_at_year[eval_year] = True
                break

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

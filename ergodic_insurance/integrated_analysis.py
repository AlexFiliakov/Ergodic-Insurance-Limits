"""Integrated simulation pipeline for ergodic insurance analysis.

Provides end-to-end loss modelling, insurance application, and ergodic
growth analysis, as well as comprehensive validation of insurance effects.

For usage examples see the
`Optimization Workflow tutorial <https://docs.mostlyoptimal.com/tutorials/04_optimization_workflow.html>`_
and the
`Advanced Scenarios tutorial <https://docs.mostlyoptimal.com/tutorials/06_advanced_scenarios.html>`_.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from .ergodic_types import ErgodicAnalysisResults, ValidationResults
from .simulation import SimulationResults

if TYPE_CHECKING:
    from .ergodic_analyzer import ErgodicAnalyzer
    from .insurance_program import InsuranceProgram
    from .loss_distributions import LossData

logger = logging.getLogger(__name__)


def integrate_loss_ergodic_analysis(
    analyzer: "ErgodicAnalyzer",
    loss_data: "LossData",
    insurance_program: Optional["InsuranceProgram"],
    manufacturer: Any,
    time_horizon: int,
    n_simulations: int = 100,
) -> ErgodicAnalysisResults:
    """End-to-end integrated loss modelling and ergodic analysis.

    Pipeline: validate inputs -> apply insurance -> aggregate losses ->
    run Monte Carlo -> calculate ergodic metrics -> validate -> package.

    Args:
        analyzer: :class:`~ergodic_insurance.ergodic_analyzer.ErgodicAnalyzer`
            instance for growth rate calculations.
        loss_data: Standardized loss data with frequency/severity
            distributions.
        insurance_program: Insurance program to apply, or *None* for
            uninsured analysis.
        manufacturer: Manufacturer model instance for business simulations.
        time_horizon: Analysis time horizon in years.
        n_simulations: Number of Monte Carlo simulations (default 100).

    Returns:
        :class:`~ergodic_insurance.ergodic_types.ErgodicAnalysisResults`
        with growth rates, survival statistics, insurance impact, and
        validation status.
    """
    from .loss_distributions import LossEvent
    from .simulation import Simulation

    # Validate input data
    if not loss_data.validate():
        logger.warning("Loss data validation failed")
        return ErgodicAnalysisResults(
            time_average_growth=-np.inf,
            ensemble_average_growth=0.0,
            survival_rate=0.0,
            ergodic_divergence=-np.inf,
            insurance_impact={},
            validation_passed=False,
            metadata={"error": "Invalid loss data"},
        )

    # Apply insurance if provided
    if insurance_program:
        insured_loss_data = loss_data.apply_insurance(insurance_program)
        insurance_metadata = insured_loss_data.metadata
    else:
        insured_loss_data = loss_data
        insurance_metadata = {}

    # Convert to annual aggregates for simulation
    annual_losses = insured_loss_data.get_annual_aggregates(time_horizon)

    # Run Monte Carlo simulations
    simulation_results = []
    for sim_idx in range(n_simulations):
        mfg_copy = copy.deepcopy(manufacturer)
        sim = Simulation(manufacturer=mfg_copy, time_horizon=time_horizon, seed=sim_idx)

        # Initialize result storage
        sim.years = np.arange(time_horizon)
        sim.assets = np.zeros(time_horizon)
        sim.equity = np.zeros(time_horizon)
        sim.roe = np.zeros(time_horizon)
        sim.revenue = np.zeros(time_horizon)
        sim.net_income = np.zeros(time_horizon)
        sim.claim_counts = np.zeros(time_horizon, dtype=int)
        sim.claim_amounts = np.zeros(time_horizon)
        sim.insolvency_year = None

        # Initialise year before the loop to prevent UnboundLocalError
        # when time_horizon == 0.
        year = 0

        # Step through each year
        for year in range(time_horizon):
            loss_amount = annual_losses.get(year, 0.0)

            losses = []
            if loss_amount > 0:
                losses.append(
                    LossEvent(time=float(year), amount=loss_amount, loss_type="aggregate")
                )

            metrics = sim.step_annual(year, losses)

            sim.assets[year] = metrics.get("assets", 0)
            sim.equity[year] = metrics.get("equity", 0)
            sim.roe[year] = metrics.get("roe", 0)
            sim.revenue[year] = metrics.get("revenue", 0)
            sim.net_income[year] = metrics.get("net_income", 0)
            sim.claim_counts[year] = metrics.get("claim_count", 0)
            sim.claim_amounts[year] = metrics.get("claim_amount", 0)

            if metrics.get("equity", 0) <= 0:
                sim.insolvency_year = year
                sim.assets[year + 1 :] = 0
                sim.equity[year + 1 :] = 0
                sim.roe[year + 1 :] = np.nan
                sim.revenue[year + 1 :] = 0
                sim.net_income[year + 1 :] = 0
                break

        # Create results object
        result = SimulationResults(
            years=sim.years[: year + 1] if sim.insolvency_year else sim.years,
            assets=sim.assets[: year + 1] if sim.insolvency_year else sim.assets,
            equity=sim.equity[: year + 1] if sim.insolvency_year else sim.equity,
            roe=sim.roe[: year + 1] if sim.insolvency_year else sim.roe,
            revenue=sim.revenue[: year + 1] if sim.insolvency_year else sim.revenue,
            net_income=(sim.net_income[: year + 1] if sim.insolvency_year else sim.net_income),
            claim_counts=(
                sim.claim_counts[: year + 1] if sim.insolvency_year else sim.claim_counts
            ),
            claim_amounts=(
                sim.claim_amounts[: year + 1] if sim.insolvency_year else sim.claim_amounts
            ),
            insolvency_year=sim.insolvency_year,
        )
        simulation_results.append(result)

    # Calculate ergodic metrics
    equity_trajectories = [r.equity for r in simulation_results]
    time_avg_growth_rates = [
        analyzer.calculate_time_average_growth(traj) for traj in equity_trajectories
    ]
    valid_time_avg = [g for g in time_avg_growth_rates if np.isfinite(g)]
    ensemble_stats = analyzer.calculate_ensemble_average(equity_trajectories, metric="growth_rate")

    # Calculate insurance impact
    insurance_impact: Dict[str, float] = {}
    if insurance_metadata:
        insurance_impact = {
            "premium_cost": insurance_metadata.get("total_premiums", 0),
            "recovery_benefit": insurance_metadata.get("total_recoveries", 0),
            "net_benefit": insurance_metadata.get("net_benefit", 0),
            "growth_improvement": float(np.mean(valid_time_avg)) if valid_time_avg else 0.0,
        }

    # Calculate ergodic divergence
    time_avg_mean = float(np.mean(valid_time_avg)) if valid_time_avg else -np.inf
    ensemble_mean = float(ensemble_stats["mean"])
    ergodic_divergence = time_avg_mean - ensemble_mean

    # Validate results
    validation_passed = (
        len(valid_time_avg) > 0
        and ensemble_stats["survival_rate"] > 0
        and np.isfinite(ergodic_divergence)
    )

    return ErgodicAnalysisResults(
        time_average_growth=time_avg_mean,
        ensemble_average_growth=ensemble_mean,
        survival_rate=ensemble_stats["survival_rate"],
        ergodic_divergence=ergodic_divergence,
        insurance_impact=insurance_impact,
        validation_passed=validation_passed,
        metadata={
            "n_simulations": n_simulations,
            "time_horizon": time_horizon,
            "n_survived": ensemble_stats["n_survived"],
            "loss_statistics": insured_loss_data.calculate_statistics(),
        },
    )


def validate_insurance_ergodic_impact(
    analyzer: "ErgodicAnalyzer",
    base_scenario: SimulationResults,
    insurance_scenario: SimulationResults,
    insurance_program: Optional["InsuranceProgram"] = None,
) -> ValidationResults:
    """Validate insurance effects in ergodic calculations.

    Checks premium deductions, recovery credits, collateral impacts,
    and growth rate consistency between base and insured scenarios.

    Args:
        analyzer: :class:`~ergodic_insurance.ergodic_analyzer.ErgodicAnalyzer`
            instance for growth rate calculations.
        base_scenario: Simulation results without insurance.
        insurance_scenario: Simulation results with insurance.
        insurance_program: Insurance program applied (optional — enables
            detailed premium validation).

    Returns:
        :class:`~ergodic_insurance.ergodic_types.ValidationResults` with
        individual check flags and detailed diagnostics.
    """
    details: Dict[str, Any] = {}

    # Check premium deductions
    premium_deductions_correct = True
    if insurance_program and hasattr(insurance_program, "calculate_premium"):
        expected_premium = insurance_program.calculate_premium()
        actual_cost_diff = np.sum(base_scenario.net_income - insurance_scenario.net_income)
        premium_diff = abs(actual_cost_diff - expected_premium * len(base_scenario.years))
        premium_deductions_correct = premium_diff < 0.01 * expected_premium
        details["premium_check"] = {
            "expected": expected_premium,
            "actual_diff": actual_cost_diff,
            "valid": premium_deductions_correct,
        }

    # Check recoveries are credited
    recoveries_credited = True
    total_base_claims = np.sum(base_scenario.claim_amounts)
    total_insured_claims = np.sum(insurance_scenario.claim_amounts)

    base_final_equity = base_scenario.equity[-1] if len(base_scenario.equity) > 0 else 0
    insured_final_equity = (
        insurance_scenario.equity[-1] if len(insurance_scenario.equity) > 0 else 0
    )
    recoveries_credited = insured_final_equity >= base_final_equity * 0.95

    details["recovery_check"] = {
        "base_claims": total_base_claims,
        "insured_claims": total_insured_claims,
        "base_final_equity": base_final_equity,
        "insured_final_equity": insured_final_equity,
        "valid": recoveries_credited,
    }

    # Check collateral impacts
    collateral_impacts_included = True
    if insurance_program and hasattr(insurance_program, "collateral_requirement"):
        base_assets = base_scenario.assets
        insured_assets = insurance_scenario.assets
        asset_diff = np.mean(insured_assets - base_assets)
        collateral_impacts_included = abs(asset_diff) > 0
        details["collateral_check"] = {
            "asset_difference": asset_diff,
            "valid": collateral_impacts_included,
        }

    # Check time-average growth benefit
    base_growth = analyzer.calculate_time_average_growth(base_scenario.equity)
    insured_growth = analyzer.calculate_time_average_growth(insurance_scenario.equity)
    growth_improvement = insured_growth - base_growth

    time_average_reflects_benefit = True
    if total_base_claims > 0:
        time_average_reflects_benefit = growth_improvement >= 0 or np.isfinite(insured_growth)

    details["growth_check"] = {
        "base_growth": base_growth,
        "insured_growth": insured_growth,
        "improvement": growth_improvement,
        "valid": time_average_reflects_benefit,
    }

    overall_valid = (
        premium_deductions_correct
        and recoveries_credited
        and collateral_impacts_included
        and time_average_reflects_benefit
    )

    return ValidationResults(
        premium_deductions_correct=premium_deductions_correct,
        recoveries_credited=recoveries_credited,
        collateral_impacts_included=collateral_impacts_included,
        time_average_reflects_benefit=time_average_reflects_benefit,
        overall_valid=overall_valid,
        details=details,
    )

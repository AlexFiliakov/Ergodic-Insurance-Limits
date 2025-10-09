# Import required libraries
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
from ergodic_insurance.exposure_base import RevenueExposure
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationResults
from ergodic_insurance.simulation import Simulation, SimulationResults

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def run_tail_simulation(
    index,
    ia,
    atr,
    ebitabl,
    ded,
    lr,
    policy_limit,
    extreme_threshold_percentile,
    extreme_sev_shape,
    extreme_sev_scale_percentile,
    NUM_SIMULATIONS,
    SIM_YEARS,
    PRICING_SIMULATIONS,
    filepath="results",
):
    """Run a single simulation with specified parameters.

    Parameters:
    - index: Index of the parameter set (for random seed)
    - ia: Initial Assets
    - atr: Asset Turnover Ratio
    - ebitabl: EBITA Margin before claims and insurance
    - ded: Deductible
    - lr: Loss Ratio
    - policy_limit: Policy Limit
    - extreme_threshold_percentile: Threshold for in absolute terms, calculated from Catastrophic Losses
    - extreme_sev_shape: Extreme loss severity shape parameter, ξ (any value)
    - extreme_sev_scale_percentile: What fraction of the upper Catastrophic conditional percentile should the corresponding percentile represent?
    - NUM_SIMULATIONS: Number of Monte Carlo simulations to run
    - SIM_YEARS: Number of years to simulate
    - PRICING_SIMULATIONS: Number of simulations for pricing estimation
    """
    print(
        f"\nRunning simulation for Initial Assets: ${ia:,.0f}, ATR: {atr}, EBITABL: {ebitabl:.3f}, Deductible: ${ded:,.0f}, Loss Ratio: {lr:.2f}"
    )
    INITIAL_ASSETS = ia
    ASSET_TURNOVER_RATIO = atr  # Revenue = Assets × Turnover
    EBITABL = ebitabl  # EBITA after claims and insurance
    DEDUCTIBLE = ded
    LOSS_RATIO = lr

    # Set random seed for reproducibility
    base_seed = 42 + index * 1000

    ### Define the Corporate Profile #####

    # Create manufacturer configuration
    manufacturer_config = ManufacturerConfig(
        initial_assets=INITIAL_ASSETS,
        asset_turnover_ratio=ASSET_TURNOVER_RATIO,  # Revenue = Assets × Turnover
        base_operating_margin=EBITABL,  # EBITA before claims and insurance (need to calibrate)
        tax_rate=0.25,  # Current US Tax Rate
        retention_ratio=0.70,  # 30% dividends
        ppe_ratio=0.00,  # 0% of assets in PPE, so there is no depreciation expense
    )

    # Create widget manufacturer
    base_manufacturer = WidgetManufacturer(manufacturer_config)

    # Create exposure base based on revenue
    exposure = RevenueExposure(state_provider=base_manufacturer)

    ## Define Losses

    cur_revenue = base_manufacturer.total_assets * base_manufacturer.asset_turnover_ratio

    base_att_freq = 2.85
    base_large_freq = 0.20
    base_cat_freq = 0.02
    cat_sev_xm = 5_000_000
    cat_sev_alpha = 2.5

    # First, we determine the `extreme_threshold` that corresponds to the desired percentile in the overall frequency
    total_freq = base_att_freq + base_large_freq + base_cat_freq
    threshold_rel_to_cat = (
        extreme_threshold_percentile * total_freq
        if extreme_threshold_percentile is not None
        else None
    )
    # Catastrophic conditional percentile to match the threshold
    cat_percentile_to_match = (
        1 - threshold_rel_to_cat / base_cat_freq if threshold_rel_to_cat is not None else None
    )
    extreme_threshold = (
        (cat_sev_xm / (1 - cat_percentile_to_match) ** (-1 / cat_sev_alpha))
        if cat_percentile_to_match is not None
        else None
    )

    # Next, we determine the scale of our GPD to match 1/10th of the desired percentile in the Catastrophic conditional distribution
    PERCENTILE_MATCH_FRACTION_OF_CAT_FREQ = 0.10
    cat_sev_freq_to_match = (
        cat_percentile_to_match * PERCENTILE_MATCH_FRACTION_OF_CAT_FREQ
        if cat_percentile_to_match is not None
        else None
    )
    cat_sev_percentile_to_match = (
        1 - cat_sev_freq_to_match / base_cat_freq if cat_sev_freq_to_match is not None else None
    )
    cat_sev_VaR_to_match = (
        (cat_sev_xm / (1 - cat_sev_percentile_to_match) ** (-1 / cat_sev_alpha))
        if cat_sev_percentile_to_match is not None
        else None
    )
    extreme_sev_corresponding_VaR = (
        cat_sev_VaR_to_match - extreme_threshold if extreme_threshold is not None else None
    )
    extreme_sev_unscaled_VaR = (
        (PERCENTILE_MATCH_FRACTION_OF_CAT_FREQ ** (-extreme_sev_shape) - 1) / extreme_sev_shape
        if extreme_sev_shape != 0
        else np.log(PERCENTILE_MATCH_FRACTION_OF_CAT_FREQ)
    )
    extreme_sev_scale = (
        extreme_sev_corresponding_VaR / extreme_sev_unscaled_VaR * extreme_sev_scale_percentile
        if extreme_sev_corresponding_VaR is not None
        else None
    )

    generator_pricing = ManufacturingLossGenerator(
        attritional_params={
            "base_frequency": base_att_freq
            * cur_revenue
            / 10_000_000,  # Scale frequency with revenue
            "severity_mean": 40_000,
            "severity_cv": 0.8,
            "revenue_scaling_exponent": 1.0,
            "reference_revenue": cur_revenue,
        },
        large_params={
            "base_frequency": base_large_freq
            * cur_revenue
            / 10_000_000,  # Scale frequency with revenue
            "severity_mean": 500_000,
            "severity_cv": 1.5,
            "revenue_scaling_exponent": 1.0,
            "reference_revenue": cur_revenue,
        },
        catastrophic_params={
            "base_frequency": base_cat_freq
            * cur_revenue
            / 10_000_000,  # Scale frequency with revenue
            "severity_xm": cat_sev_xm,
            "severity_alpha": cat_sev_alpha,
            "revenue_scaling_exponent": 1.0,
            "reference_revenue": cur_revenue,
        },
        extreme_params={
            "threshold_value": extreme_threshold,
            "severity_shape": extreme_sev_shape,
            "severity_scale": extreme_sev_scale,
        },
        seed=base_seed,
    )

    deductible = DEDUCTIBLE

    ### Run Pricing Simulation #####
    # Assume the insurer has perfect knowledge of the loss distribution

    pricing_simulation_years = PRICING_SIMULATIONS

    total_insured_loss = 0.0
    insured_loss_list = []

    total_retained_loss = 0.0
    retained_loss_list = []

    for yr in range(pricing_simulation_years):
        loss_events, loss_meta = generator_pricing.generate_losses(
            duration=1, revenue=base_manufacturer.base_revenue
        )
        for loss_event in loss_events:
            insured_loss = max(min(loss_event.amount, policy_limit) - deductible, 0)

            total_insured_loss += insured_loss
            insured_loss_list.append(insured_loss)

            retained_loss = loss_event.amount - insured_loss
            total_retained_loss += retained_loss
            retained_loss_list.append(retained_loss)

    average_annual_insured_loss = total_insured_loss / pricing_simulation_years
    average_annual_retained_loss = total_retained_loss / pricing_simulation_years
    print(f"Average Annual Insured Loss: ${average_annual_insured_loss:,.0f}")
    print(f"Average Annual Retained Loss: ${average_annual_retained_loss:,.0f}")

    ground_up_losses = np.asarray(insured_loss_list, dtype=float) + np.asarray(
        retained_loss_list, dtype=float
    )
    EXCESS_KURTOSIS = pd.Series(ground_up_losses).kurtosis()
    print(f"Ground-Up Excess Kurtosis: {EXCESS_KURTOSIS:.2f}")

    loss_ratio = LOSS_RATIO

    annual_premium = average_annual_insured_loss / loss_ratio
    print(f"Annual Premium: ${annual_premium:,.0f}")

    total_cost_of_risk = annual_premium + average_annual_retained_loss
    print(f"Total Annual Cost of Risk: ${total_cost_of_risk:,.0f}")

    cur_operating_income = base_manufacturer.calculate_operating_income(cur_revenue)

    cur_net_income = base_manufacturer.calculate_net_income(
        operating_income=cur_operating_income,
        collateral_costs=0.0,
        insurance_premiums=annual_premium,
        insurance_losses=average_annual_retained_loss,
        use_accrual=True,
        time_resolution="annual",
    )

    # target_net_income = base_manufacturer.base_revenue * target_ebita_margin * (1 - base_manufacturer.tax_rate)
    # target_net_income

    # net_margin_diff = abs(cur_net_income - target_net_income) / cur_revenue

    # assert net_margin_diff < 0.0005, f"Net income not within 0.05% of target ({net_margin_diff:.2%} difference)"

    net_margin = cur_net_income / cur_revenue
    print(f"Net Margin after insurance: {net_margin:.2%}")
    print(f"EBITA Margin after insurance: {net_margin / (1 - base_manufacturer.tax_rate):.2%}")

    ## Define the Insurance Program

    all_layers = EnhancedInsuranceLayer(
        attachment_point=deductible,
        limit=policy_limit,
        limit_type="per-occurrence",
        base_premium_rate=annual_premium / policy_limit,
    )

    program = InsuranceProgram([all_layers])

    # total_premium = program.calculate_annual_premium()

    ### Set Up the Simulation With Insurance #######

    def setup_simulation_engine(
        n_simulations=10_000,
        n_years=10,
        parallel=False,
        insurance_program=None,
        seed=base_seed + 100,
    ):
        """Set up Monte Carlo simulation engine."""
        generator = ManufacturingLossGenerator(
            attritional_params={
                "base_frequency": 2.85 * cur_revenue / 10_000_000,  # Scale frequency with revenue
                "severity_mean": 40_000,
                "severity_cv": 0.8,
                "revenue_scaling_exponent": 1.0,
                "reference_revenue": cur_revenue,
            },
            large_params={
                "base_frequency": 0.20 * cur_revenue / 10_000_000,  # Scale frequency with revenue
                "severity_mean": 500_000,
                "severity_cv": 1.5,
                "revenue_scaling_exponent": 1.0,
                "reference_revenue": cur_revenue,
            },
            catastrophic_params={
                "base_frequency": 0.02 * cur_revenue / 10_000_000,  # Scale frequency with revenue
                "severity_xm": 5_000_000,
                "severity_alpha": 2.5,
                "revenue_scaling_exponent": 1.0,
                "reference_revenue": cur_revenue,
            },
            seed=seed,
        )

        # Create simulation config
        config = SimulationConfig(
            n_simulations=n_simulations,
            n_years=n_years,
            n_chains=4,
            parallel=parallel,
            n_workers=None,
            chunk_size=max(1000, n_simulations // 10),
            use_float32=True,
            cache_results=False,
            progress_bar=True,
            ruin_evaluation=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            working_capital_pct=0.0,  # Set working capital to 0% for full asset revenue generation
            seed=seed + 450,
        )

        if insurance_program is None:
            insurance_program = InsuranceProgram(
                layers=[],  # Empty list to define no coverage
                deductible=0.0,  # No deductible needed since all losses are retained
                pricer=None,
                name="No Insurance",
            )

        # Create engine
        engine = MonteCarloEngine(
            loss_generator=generator,
            insurance_program=insurance_program,
            manufacturer=base_manufacturer,
            config=config,
        )

        return engine

    # Create engine
    print("Setting up Monte Carlo engine with Insurance...")
    engine = setup_simulation_engine(
        n_simulations=NUM_SIMULATIONS,
        n_years=SIM_YEARS,
        parallel=False,
        insurance_program=program,
        seed=base_seed + 100,
    )

    ### Set Up the Simulation Without Insurance #######

    # Create engine without insurance
    print("Setting up Monte Carlo engine without Insurance...")
    engine_no_ins = setup_simulation_engine(
        n_simulations=NUM_SIMULATIONS,
        n_years=SIM_YEARS,
        parallel=False,
        insurance_program=None,
        seed=base_seed + 200,
    )

    ## Run the Simulation

    filename = f"{filepath}/Ded ({DEDUCTIBLE/1_000:.0f}K) - \
LR ({LOSS_RATIO}) - \
Pol_Lim ({policy_limit/1_000_000:.0f}M) - \
X_Th_%le ({extreme_threshold_percentile}) - \
X_Shape ({extreme_sev_shape}) - \
X_Scale ({extreme_sev_scale_percentile}) - \
{NUM_SIMULATIONS/1_000:.0f}K Sims - \
{SIM_YEARS} Yrs.pkl"

    filename_no_ins = f"{filepath}/X_Th_%le ({extreme_threshold_percentile}) - \
X_Shape ({extreme_sev_shape}) - \
X_Scale ({extreme_sev_scale_percentile}) - \
NOINS - \
{NUM_SIMULATIONS/1_000:.0f}K Sims - \
{SIM_YEARS} Yrs.pkl"

    results = engine.run()

    with open(filename, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_exists_no_ins = Path(filename_no_ins).exists()

    if file_exists_no_ins:
        print(f"Skipping no-insurance simulation run, already exists: {filename_no_ins}")
    else:
        results_no_ins = engine_no_ins.run()

        with open(filename_no_ins, "wb") as f:
            pickle.dump(results_no_ins, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nConvergence achieved: {'Yes' if results.convergence else 'No'}")

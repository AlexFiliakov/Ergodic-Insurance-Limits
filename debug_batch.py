"""Debug script to test batch simulations like the test."""

from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.simulation import Simulation
from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
import numpy as np

# Parameters from test
BASE_CONFIG = {
    "initial_assets": 10_000_000,
    "random_seed": 42,
}

# Create claim generator (shared, like in test)
claim_gen = ClaimGenerator(
    seed=BASE_CONFIG["random_seed"],
    frequency=5.52,  # Total from blog: 5 + 0.5 + 0.02
    severity_mean=213_000,  # Calibrated for $1.175M expected annual loss
    severity_std=800_000,  # Higher std to capture catastrophic tail
)

# Create insurance (same as test)
layer = InsuranceLayer(
    attachment_point=100_000,
    limit=10_000_000,
    rate=0.025,  # Higher than expected loss
)
insurance = InsurancePolicy(layers=[layer], deductible=100_000)

print(f"Annual premium: ${insurance.calculate_premium():,.0f}")
print(f"Expected annual loss: ${claim_gen.frequency * claim_gen.severity_mean:,.0f}")
print(f"Premium/Expected Loss ratio: {insurance.calculate_premium() / (claim_gen.frequency * claim_gen.severity_mean):.2%}")

# Run simulations
analyzer = ErgodicAnalyzer()
insured_batch = []
uninsured_batch = []

n_simulations = 5  # Start with just 5 for debugging

for i in range(n_simulations):
    print(f"\nRunning simulation {i+1}/{n_simulations}...")

    # Insured scenario
    insured_config = ManufacturerConfig(
        initial_assets=BASE_CONFIG["initial_assets"],
        asset_turnover_ratio=1.2,
        base_operating_margin=0.1,
        tax_rate=0.25,
        retention_ratio=0.70,
    )
    insured_mfg = WidgetManufacturer(insured_config)

    insured_simulation = Simulation(
        manufacturer=insured_mfg,
        claim_generator=claim_gen,
        time_horizon=50,  # Shorter for testing
        insurance_policy=insurance,
        seed=BASE_CONFIG["random_seed"] + i,
    )
    insured_result = insured_simulation.run()
    insured_batch.append(insured_result)

    print(f"  Insured: Survived={insured_result.insolvency_year is None}, Final equity=${insured_result.equity[-1]:,.0f}")

    # Uninsured scenario
    uninsured_config = ManufacturerConfig(
        initial_assets=BASE_CONFIG["initial_assets"],
        asset_turnover_ratio=1.2,
        base_operating_margin=0.1,
        tax_rate=0.25,
        retention_ratio=0.70,
    )
    uninsured_mfg = WidgetManufacturer(uninsured_config)

    uninsured_simulation = Simulation(
        manufacturer=uninsured_mfg,
        claim_generator=claim_gen,
        time_horizon=50,
        insurance_policy=None,
        seed=BASE_CONFIG["random_seed"] + 100 + i,
    )
    uninsured_result = uninsured_simulation.run()
    uninsured_batch.append(uninsured_result)

    print(f"  Uninsured: Survived={uninsured_result.insolvency_year is None}, Final equity=${uninsured_result.equity[-1]:,.0f}")

# Analyze results
comparison = analyzer.compare_scenarios(
    insured_results=insured_batch,
    uninsured_results=uninsured_batch,
    metric="equity",
)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Insured survival rate: {comparison['insured']['survival_rate']:.1%}")
print(f"Uninsured survival rate: {comparison['uninsured']['survival_rate']:.1%}")
print(f"Survival advantage: {comparison['advantage']['survival_gain']:.1%}")
print(f"Ergodic advantage: {comparison.get('ergodic_advantage', 'N/A')}")

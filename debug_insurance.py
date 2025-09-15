"""Debug script to understand insurance failure."""

from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.simulation import Simulation
import numpy as np

# Create manufacturer - match test parameters exactly
config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.2,
    base_operating_margin=0.1,
    tax_rate=0.25,
    retention_ratio=0.70,
)
manufacturer = WidgetManufacturer(config)

# Create claim generator
claim_gen = ClaimGenerator(
    seed=42,
    frequency=5.52,
    severity_mean=213_000,
    severity_std=800_000,
)

# Create insurance
layer = InsuranceLayer(
    attachment_point=100_000,
    limit=10_000_000,
    rate=0.025,
)
insurance = InsurancePolicy(layers=[layer], deductible=100_000)

# Calculate annual premium
annual_premium = insurance.calculate_premium()
print(f"Annual insurance premium: ${annual_premium:,.0f}")
print(f"Initial assets: ${manufacturer.assets:,.0f}")
print(f"Initial equity: ${manufacturer.equity:,.0f}")

# Run one year to see what happens
simulation = Simulation(
    manufacturer=manufacturer,
    claim_generator=claim_gen,
    time_horizon=5,  # Just 5 years for debugging
    insurance_policy=insurance,
    seed=42,
)

# Check expected annual loss
expected_loss = claim_gen.frequency * claim_gen.severity_mean
print(f"Expected annual loss: ${expected_loss:,.0f}")

# Run simulation
print("\nRunning simulation...")
result = simulation.run()

print(f"\nSimulation results:")
print(f"Survived: {result.insolvency_year is None}")
if result.insolvency_year:
    print(f"Insolvent in year: {result.insolvency_year}")

# Print year-by-year results
print("\nYear-by-year equity:")
for year, equity in enumerate(result.equity):
    print(f"Year {year}: ${equity:,.0f}")

print("\nYear-by-year claim amounts:")
for year, claims in enumerate(result.claim_amounts):
    print(f"Year {year}: ${claims:,.0f}")

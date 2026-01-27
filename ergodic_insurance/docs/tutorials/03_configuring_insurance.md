# Configuring Insurance

This tutorial explains how to set up insurance programs with multiple layers, deductibles, and advanced features like reinstatements and aggregate limits.

## Insurance Program Basics

An insurance program consists of:
- **Deductible (Retention)**: Amount the business pays before insurance kicks in
- **Layers**: Coverage tiers that sit above the deductible
- **Limits**: Maximum payout per occurrence or in aggregate
- **Premium**: Cost of the coverage

## Creating a Simple Insurance Program

```python
from ergodic_insurance.insurance_program import InsuranceProgram, EnhancedInsuranceLayer

# Create a simple single-layer program
program = InsuranceProgram(
    deductible=100_000  # $100K self-insured retention
)

# Add a primary layer
primary_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,    # Attaches above $100K deductible
    limit=5_000_000,             # $5M per-occurrence limit
    base_premium_rate=0.02       # 2% of limit = $100K premium
)
program.add_layer(primary_layer)

print(f"Deductible: ${program.deductible:,.0f}")
print(f"Total Coverage: ${program.total_coverage():,.0f}")
print(f"Annual Premium: ${program.total_premium():,.0f}")
```

## Multi-Layer Insurance Structure

Real insurance programs often have multiple layers (towers):

```python
# Create a layered insurance program
program = InsuranceProgram(deductible=250_000)

# Primary layer: $250K xs $250K
primary = EnhancedInsuranceLayer(
    attachment_point=250_000,
    limit=5_000_000,
    base_premium_rate=0.025  # Higher rate for primary layer
)

# First excess layer: $5M xs $5.25M
excess_1 = EnhancedInsuranceLayer(
    attachment_point=5_250_000,
    limit=5_000_000,
    base_premium_rate=0.015  # Lower rate for excess
)

# Second excess layer: $10M xs $10.25M
excess_2 = EnhancedInsuranceLayer(
    attachment_point=10_250_000,
    limit=10_000_000,
    base_premium_rate=0.008  # Even lower for high excess
)

program.add_layer(primary)
program.add_layer(excess_1)
program.add_layer(excess_2)

# Summary
print("=== Insurance Tower ===")
print(f"Deductible: ${program.deductible:,.0f}")
for i, layer in enumerate(program.layers, 1):
    print(f"Layer {i}: ${layer.limit:,.0f} xs ${layer.attachment_point:,.0f} @ {layer.base_premium_rate:.2%}")
print(f"Total Coverage: ${program.total_coverage():,.0f}")
print(f"Total Premium: ${program.total_premium():,.0f}")
```

## Layer Types

The framework supports different limit types:

### Per-Occurrence Limits

Traditional per-occurrence coverage - each claim is limited separately:

```python
per_occurrence_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.02,
    limit_type="per-occurrence"  # Default
)
```

### Aggregate Limits

Coverage limited by total annual claims:

```python
aggregate_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.015,
    limit_type="aggregate",
    aggregate_limit=10_000_000  # Annual aggregate cap
)
```

### Hybrid Limits

Both per-occurrence and aggregate limits:

```python
hybrid_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,           # Per-occurrence limit
    base_premium_rate=0.018,
    limit_type="hybrid",
    per_occurrence_limit=5_000_000,
    aggregate_limit=15_000_000  # Annual aggregate cap
)
```

## Reinstatements

Reinstatements restore coverage after a claim erodes the limit:

```python
# Layer with 2 reinstatements at 100% premium each
layer_with_reinstatements = EnhancedInsuranceLayer(
    attachment_point=500_000,
    limit=2_000_000,
    base_premium_rate=0.03,
    reinstatements=2,              # Two reinstatements available
    reinstatement_premium=1.0,     # 100% of original premium per reinstatement
    reinstatement_type=ReinstatementType.PRO_RATA  # Pro-rata based on time
)

# Import the enum
from ergodic_insurance.insurance_program import ReinstatementType

# Available reinstatement types:
# - NONE: No reinstatements
# - PRO_RATA: Premium based on time remaining in policy period
# - FULL: Full premium regardless of timing
# - FREE: No additional premium for reinstatement
```

## Participation Rates (Quota Share)

For co-insurance arrangements:

```python
# Layer covering only 80% of losses
quota_share_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.016,
    participation_rate=0.80  # Insurer covers 80%, company retains 20%
)
```

## Processing Claims Through the Program

```python
# Simulate claim processing
claim_amount = 3_000_000  # $3M gross loss

# Calculate how the claim flows through the program
deductible_retained = min(claim_amount, program.deductible)
excess_of_deductible = max(0, claim_amount - program.deductible)

print(f"Gross Loss: ${claim_amount:,.0f}")
print(f"Deductible Retained: ${deductible_retained:,.0f}")
print(f"Covered by Insurance: ${excess_of_deductible:,.0f}")

# Apply the claim to the program
recoveries = program.apply_claim(claim_amount)
print(f"Insurance Recovery: ${recoveries:,.0f}")
```

## Using ConfigManager for Insurance Setup

The `ConfigManager` provides preset insurance configurations:

```python
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()

# Load a profile with insurance configuration
config = manager.load_profile(
    "default",
    presets=["hard_market"]  # Apply hard market conditions
)

# Access insurance configuration
if config.insurance:
    print(f"Number of layers: {len(config.insurance.layers)}")
    for i, layer in enumerate(config.insurance.layers, 1):
        print(f"  Layer {i}: ${layer.limit:,.0f} xs ${layer.attachment:,.0f}")
```

## Integrating Insurance with Simulation

```python
from ergodic_insurance import ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.simulation import Simulation
from ergodic_insurance.insurance_program import InsuranceProgram, EnhancedInsuranceLayer

# Create manufacturer
mfg_config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0
)
manufacturer = WidgetManufacturer(mfg_config)

# Create insurance program
insurance = InsuranceProgram(deductible=100_000)
insurance.add_layer(EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.02
))

# Create claim generator
claims = ManufacturingLossGenerator.create_simple(
    frequency=0.2,
    severity_mean=1_000_000,
    severity_std=1_500_000,
    seed=42
)

# Run simulation with insurance
sim = Simulation(
    manufacturer=manufacturer,
    claim_generator=claims,
    insurance_program=insurance,  # Attach insurance
    time_horizon=30
)

results_insured = sim.run()
print(f"Insured Final Equity: ${results_insured.equity[-1]:,.0f}")
```

## Comparing Insured vs Uninsured

```python
# Run without insurance for comparison
manufacturer_no_ins = WidgetManufacturer(mfg_config)
sim_no_ins = Simulation(
    manufacturer=manufacturer_no_ins,
    claim_generator=ManufacturingLossGenerator.create_simple(frequency=0.2, severity_mean=1_000_000, severity_std=1_500_000, seed=42),
    time_horizon=30
)
results_uninsured = sim_no_ins.run()

# Compare outcomes
print("\n=== Insurance Impact ===")
print(f"Insured Final Equity:   ${results_insured.equity[-1]:,.0f}")
print(f"Uninsured Final Equity: ${results_uninsured.equity[-1]:,.0f}")
print(f"Insurance Benefit:      ${results_insured.equity[-1] - results_uninsured.equity[-1]:,.0f}")

# Time-weighted growth comparison
insured_growth = results_insured.calculate_time_weighted_roe()
uninsured_growth = results_uninsured.calculate_time_weighted_roe()
print(f"\nInsured Time-Weighted ROE:   {insured_growth:.2%}")
print(f"Uninsured Time-Weighted ROE: {uninsured_growth:.2%}")
print(f"Growth Improvement:          {insured_growth - uninsured_growth:.2%}")
```

## Insurance Premium Loading Analysis

```python
# Calculate the expected loss
expected_annual_claims = 0.2 * 1_000_000  # frequency × severity mean
print(f"Expected Annual Claims: ${expected_annual_claims:,.0f}")

# Premium paid
annual_premium = insurance.total_premium()
print(f"Annual Premium: ${annual_premium:,.0f}")

# Premium loading (premium / expected loss)
loading = annual_premium / expected_annual_claims - 1
print(f"Premium Loading: {loading:.0%}")

# Even with significant loading, insurance may improve time-average growth!
print(f"\nDespite {loading:.0%} loading, insurance improved growth by {insured_growth - uninsured_growth:.2%}")
```

## Next Steps

- [Tutorial 4: Optimization Workflow](04_optimization_workflow.md) - Find optimal deductibles and limits
- [Tutorial 5: Analyzing Results](05_analyzing_results.md) - Ergodic analysis
- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) - Complex configurations

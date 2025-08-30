"""Test script to verify notebook fixes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from ergodic_insurance.src.insurance_pricing import InsurancePricer, MarketCycle, PricingParameters
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.loss_distributions import ManufacturingLossGenerator

# Create loss generator with same params as notebook
loss_generator = ManufacturingLossGenerator(
    attritional_params={
        "base_frequency": 4,
        "severity_mean": 30_000,
        "severity_cv": 1.2,
    },
    large_params={
        "base_frequency": 0.2,
        "severity_mean": 1_500_000,
        "severity_cv": 1.5,
    },
    catastrophic_params={
        "base_frequency": 0.02,
        "severity_alpha": 3.0,
        "severity_xm": 5_000_000,
    },
    seed=42,
)

# Test pricing with 100 simulation years
deductible = 250_000
revenue = 15_000_000

print("Testing pricing with 100 simulation years (same seed for all cycles):")
print("=" * 60)

premiums = {}

for market_cycle in MarketCycle:
    # Create pricer with 100 simulation years and same seed
    pricer = InsurancePricer(
        loss_generator=loss_generator,
        market_cycle=market_cycle,
        parameters=PricingParameters(
            simulation_years=100,
            confidence_level=0.95,
            expense_ratio=0.25,
            profit_margin=0.15,
            risk_loading=0.10,
        ),
        seed=42,  # Same seed for consistency
    )

    # Calculate pricing for first layer
    pure_premium, stats = pricer.calculate_pure_premium(
        attachment_point=deductible,
        limit=5_000_000 - deductible,
        expected_revenue=revenue,
    )

    technical_premium = pricer.calculate_technical_premium(pure_premium, 5_000_000 - deductible)
    market_premium = pricer.calculate_market_premium(technical_premium, market_cycle)

    premiums[market_cycle] = market_premium

    print(f"\n{market_cycle.name} Market (Loss Ratio: {market_cycle.value:.0%}):")
    print(f"  Pure Premium: ${pure_premium:,.0f}")
    print(f"  Technical Premium: ${technical_premium:,.0f}")
    print(f"  Market Premium: ${market_premium:,.0f}")
    print(f"  Expected Frequency: {stats['expected_frequency']:.4f}")
    print(f"  Expected Severity: ${stats['expected_severity']:,.0f}")

# Check variation
hard = premiums[MarketCycle.HARD]
soft = premiums[MarketCycle.SOFT]
variation = (hard / soft - 1) * 100

print("\n" + "=" * 60)
print("VERIFICATION:")
print(f"Hard Market Premium: ${hard:,.0f}")
print(f"Soft Market Premium: ${soft:,.0f}")
print(f"Variation: {variation:.1f}%")

if variation > 30:
    print("✅ SUCCESS: Market cycles show appropriate premium variation")
else:
    print("⚠️ WARNING: Premium variation may be too low")

# Test expected value
print("\n" + "=" * 60)
print("TESTING INSURANCE EXPECTED VALUE:")

# Generate losses for 100 years
total_losses = []
for year in range(100):
    losses, _ = loss_generator.generate_losses(
        duration=1.0, revenue=revenue, include_catastrophic=True
    )
    annual_total = sum(l.amount for l in losses)
    total_losses.append(annual_total)

avg_annual_loss = np.mean(total_losses)
normal_premium = premiums[MarketCycle.NORMAL]

print(f"Average Annual Loss: ${avg_annual_loss:,.0f}")
print(f"Normal Market Premium: ${normal_premium:,.0f}")
print(f"Premium/Loss Ratio: {normal_premium/avg_annual_loss:.2f}x")

if normal_premium > avg_annual_loss:
    print("✅ CORRECT: Insurance has negative expected value (premium > expected loss)")
else:
    print("⚠️ ERROR: Insurance has positive expected value (premium < expected loss)")
    print("   This is unrealistic and needs fixing")

# Test affordability
gross_profit = revenue * 0.12  # 12% margin
print(f"\nAffordability Check:")
print(f"  Gross Profit: ${gross_profit:,.0f}")
print(f"  Premium as % of Profit: {normal_premium/gross_profit:.1%}")

if normal_premium > gross_profit * 0.5:
    print("  ⚠️ Premium may be unaffordable")
elif normal_premium < gross_profit * 0.1:
    print("  ⚠️ Premium may be too low")
else:
    print("  ✅ Premium appears reasonable")

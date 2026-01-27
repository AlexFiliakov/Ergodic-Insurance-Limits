#!/usr/bin/env python
"""Quick test of create_simple() migration helper."""

from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

# Test create_simple()
print("Testing ManufacturingLossGenerator.create_simple()...")
gen = ManufacturingLossGenerator.create_simple(
    frequency=0.1,
    severity_mean=5_000_000,
    severity_std=2_000_000,
    seed=42
)
print("[+] Generator created successfully")

# Test loss generation
losses, stats = gen.generate_losses(duration=10, revenue=10_000_000)
print(f"[+] Generated {len(losses)} losses")
print(f"  Total amount: ${stats['total_amount']:,.0f}")
print(f"  Attritional: {stats['attritional_count']} events, ${stats['attritional_amount']:,.0f}")
print(f"  Large: {stats['large_count']} events, ${stats['large_amount']:,.0f}")
print(f"  Catastrophic: {stats['catastrophic_count']} events, ${stats['catastrophic_amount']:,.0f}")

print("\nAll tests passed!")

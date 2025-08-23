"""Demonstration of the updated collateral management model with letters of credit."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ManufacturerConfig
from src.manufacturer import WidgetManufacturer


def main():
    """Demonstrate the collateral management system with letters of credit."""
    
    # Create configuration
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0
    )
    
    # Create manufacturer
    manufacturer = WidgetManufacturer(config)
    
    print("=" * 70)
    print("COLLATERAL MANAGEMENT DEMONSTRATION - LETTERS OF CREDIT")
    print("=" * 70)
    print()
    
    # Initial state
    print("INITIAL STATE:")
    print(f"  Total Assets:        ${manufacturer.assets:,.0f}")
    print(f"  Restricted Assets:   ${manufacturer.restricted_assets:,.0f}")
    print(f"  Available Assets:    ${manufacturer.available_assets:,.0f}")
    print(f"  Equity:             ${manufacturer.equity:,.0f}")
    print(f"  Collateral (LoC):   ${manufacturer.collateral:,.0f}")
    print(f"  Is Solvent:         {not manufacturer.is_ruined}")
    print()
    
    # Year 0: Normal operations
    print("YEAR 0: Normal Operations")
    metrics = manufacturer.step(working_capital_pct=0.2, growth_rate=0.05)
    print(f"  Revenue:            ${metrics['revenue']:,.0f}")
    print(f"  Net Income:         ${metrics['net_income']:,.0f}")
    print(f"  New Assets:         ${metrics['assets']:,.0f}")
    print(f"  Equity:             ${metrics['equity']:,.0f}")
    print()
    
    # Process large claim requiring letter of credit
    print("CLAIM EVENT: Large Insurance Claim")
    claim_amount = 8_000_000
    print(f"  Claim Amount:       ${claim_amount:,.0f}")
    manufacturer.process_insurance_claim(claim_amount)
    print(f"  Letter of Credit Posted: ${manufacturer.collateral:,.0f}")
    print(f"  Restricted Assets:  ${manufacturer.restricted_assets:,.0f}")
    print(f"  Available Assets:   ${manufacturer.available_assets:,.0f}")
    print()
    
    # Show monthly LoC costs
    print("MONTHLY LETTER OF CREDIT COSTS:")
    annual_rate = 0.015
    monthly_cost = manufacturer.calculate_collateral_costs(annual_rate, "monthly")
    annual_cost = manufacturer.calculate_collateral_costs(annual_rate, "annual")
    print(f"  Annual Rate:        {annual_rate:.1%}")
    print(f"  Monthly Cost:       ${monthly_cost:,.0f}")
    print(f"  Annual Cost:        ${annual_cost:,.0f}")
    print()
    
    # Year 1-5: Show claim payments and collateral reduction
    print("YEARS 1-5: Claim Payments and Collateral Reduction")
    print("-" * 70)
    print(f"{'Year':<6} {'Net Income':<15} {'Assets':<15} {'Collateral':<15} {'Claims Due':<15} {'Solvent':<10}")
    print("-" * 70)
    
    for year in range(1, 6):
        metrics = manufacturer.step(working_capital_pct=0.2, letter_of_credit_rate=annual_rate)
        print(f"{year:<6} ${metrics['net_income']:>12,.0f}  ${metrics['assets']:>12,.0f}  "
              f"${metrics['collateral']:>12,.0f}  ${metrics['claim_liabilities']:>12,.0f}  "
              f"{'Yes' if metrics['is_solvent'] else 'NO':<10}")
    
    print("-" * 70)
    print()
    
    # Test insolvency scenario
    print("INSOLVENCY SCENARIO: Massive Claim")
    print("=" * 70)
    
    # Reset and apply massive claim
    manufacturer.reset()
    manufacturer.step()  # One year of normal operations
    
    # Process a claim larger than equity can support
    massive_claim = 25_000_000
    print(f"  Processing massive claim: ${massive_claim:,.0f}")
    manufacturer.process_insurance_claim(massive_claim)
    print(f"  Letter of Credit:   ${manufacturer.collateral:,.0f}")
    print(f"  Annual LoC Cost:    ${manufacturer.collateral * annual_rate:,.0f}")
    print()
    
    # Run until insolvency
    print("Running simulation until insolvency...")
    year = 1
    while not manufacturer.is_ruined and year <= 20:
        metrics = manufacturer.step(working_capital_pct=0.2, letter_of_credit_rate=annual_rate)
        if year % 2 == 0 or manufacturer.is_ruined:  # Print every other year or on ruin
            print(f"  Year {year:2}: Equity = ${metrics['equity']:>12,.0f}, "
                  f"Assets = ${metrics['assets']:>12,.0f}, "
                  f"Solvent = {'Yes' if metrics['is_solvent'] else 'RUINED'}")
        year += 1
    
    if manufacturer.is_ruined:
        print(f"\n  Company became insolvent in year {year-1}")
        print(f"  Final Equity: ${manufacturer.equity:,.0f}")
    else:
        print(f"\n  Company survived {year-1} years")
    
    print()
    print("=" * 70)
    print("KEY INSIGHTS:")
    print("  1. All large claims require letter of credit collateral")
    print("  2. LoC costs are paid monthly (1.5% annual rate / 12)")
    print("  3. Collateral reduces as claims are paid down")
    print("  4. Restricted assets limit available capital for operations")
    print("  5. Company becomes insolvent when equity <= 0")


if __name__ == "__main__":
    main()
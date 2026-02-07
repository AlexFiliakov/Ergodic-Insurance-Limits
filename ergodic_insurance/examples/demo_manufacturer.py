"""Demonstration of the WidgetManufacturer financial model."""

from ergodic_insurance import ManufacturerConfig, WidgetManufacturer


def main():
    """Run a demonstration of the WidgetManufacturer model."""

    # Create configuration
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,
    )

    # Create manufacturer
    manufacturer = WidgetManufacturer(config)

    print("=" * 60)
    print("WIDGET MANUFACTURER FINANCIAL MODEL DEMONSTRATION")
    print("=" * 60)
    print()

    # Initial state
    print("INITIAL STATE:")
    print(f"  Assets:           ${manufacturer.assets:,.0f}")
    print(f"  Equity:           ${manufacturer.equity:,.0f}")
    print(f"  Collateral:       ${manufacturer.collateral:,.0f}")
    print()

    # Year 0: Normal operations
    print("YEAR 0: Normal Operations")
    metrics = manufacturer.step(growth_rate=0.05)
    print(f"  Revenue:          ${metrics['revenue']:,.0f}")
    print(f"  Operating Income: ${metrics['operating_income']:,.0f}")
    print(f"  Net Income:       ${metrics['net_income']:,.0f}")
    print(f"  New Assets:       ${metrics['assets']:,.0f}")
    print(f"  ROE:              {metrics['roe']:.1%}")
    print()

    # Year 1: Large insurance claim
    print("YEAR 1: Large Insurance Claim Event")
    claim_amount = 5_000_000
    print(f"  Processing claim: ${claim_amount:,.0f}")
    manufacturer.process_insurance_claim(
        claim_amount, deductible=100_000, insurance_limit=10_000_000
    )
    print(f"  Collateral added: ${manufacturer.collateral:,.0f}")

    metrics = manufacturer.step(letter_of_credit_rate=0.015)
    print(f"  Revenue:          ${metrics['revenue']:,.0f}")
    print(f"  Operating Income: ${metrics['operating_income']:,.0f}")
    print(f"  Collateral Costs: ${manufacturer.calculate_collateral_costs():,.0f}")
    print(f"  Net Income:       ${metrics['net_income']:,.0f}")
    print(f"  New Assets:       ${metrics['assets']:,.0f}")
    print(f"  Outstanding Claims: ${metrics['claim_liabilities']:,.0f}")
    print()

    # Years 2-5: Recovery period
    print("YEARS 2-5: Recovery Period")
    for year in range(2, 6):
        metrics = manufacturer.step(letter_of_credit_rate=0.015)
        print(f"  Year {year}:")
        print(f"    Net Income:     ${metrics['net_income']:,.0f}")
        print(f"    Assets:         ${metrics['assets']:,.0f}")
        print(f"    Collateral:     ${metrics['collateral']:,.0f}")
        print(f"    Claims Remaining: ${metrics['claim_liabilities']:,.0f}")

    print()
    print("=" * 60)
    print("SUMMARY METRICS:")
    print("=" * 60)

    # Calculate average metrics over history
    total_revenue = sum(m["revenue"] for m in manufacturer.metrics_history)
    total_net_income = sum(m["net_income"] for m in manufacturer.metrics_history)
    avg_roe = sum(m["roe"] for m in manufacturer.metrics_history) / len(
        manufacturer.metrics_history
    )

    print(f"  Total Revenue (6 years):    ${total_revenue:,.0f}")
    print(f"  Total Net Income (6 years): ${total_net_income:,.0f}")
    print(f"  Average ROE:                {avg_roe:.1%}")
    print(f"  Final Assets:               ${manufacturer.assets:,.0f}")
    print(f"  Final Equity:               ${manufacturer.equity:,.0f}")
    print(f"  Final Collateral:           ${manufacturer.collateral:,.0f}")

    # Show claim payment schedule
    print()
    print("CLAIM PAYMENT SCHEDULE:")
    print("  Year 1:  10% of claim")
    print("  Year 2:  20% of claim")
    print("  Year 3:  20% of claim")
    print("  Year 4:  15% of claim")
    print("  Year 5:  10% of claim")
    print("  Year 6:   8% of claim")
    print("  Year 7:   7% of claim")
    print("  Year 8:   5% of claim")
    print("  Year 9:   3% of claim")
    print("  Year 10:  2% of claim")


if __name__ == "__main__":
    main()

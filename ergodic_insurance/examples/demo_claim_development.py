"""Demonstration of claim development patterns for cash flow modeling.

This script shows how to use the claim development module to model
realistic claim payment patterns and project cash flows.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ergodic_insurance.src.claim_development import (
    CashFlowProjector,
    Claim,
    ClaimCohort,
    ClaimDevelopment,
    load_development_patterns,
)
from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.manufacturer import WidgetManufacturer


def demo_basic_development_patterns():
    """Demonstrate basic claim development patterns."""
    print("\n" + "=" * 60)
    print("BASIC CLAIM DEVELOPMENT PATTERNS")
    print("=" * 60)

    # Create different pattern types
    patterns = {
        "Immediate (Property)": ClaimDevelopment.create_immediate(),
        "Medium Tail (Workers Comp)": ClaimDevelopment.create_medium_tail_5yr(),
        "Long Tail (General Liability)": ClaimDevelopment.create_long_tail_10yr(),
        "Very Long Tail (Product Liability)": ClaimDevelopment.create_very_long_tail_15yr(),
    }

    claim_amount = 1_000_000
    accident_year = 2020

    for name, pattern in patterns.items():
        print(f"\n{name}:")
        print(f"  Pattern: {pattern.pattern_name}")
        print(f"  Development years: {len(pattern.development_factors)}")

        # Show first 3 years of payments
        for year in range(2020, 2023):
            payment = pattern.calculate_payments(claim_amount, accident_year, year)
            pct = (payment / claim_amount) * 100
            print(f"  Year {year}: ${payment:,.0f} ({pct:.1f}%)")


def demo_claim_cohort():
    """Demonstrate claim cohort management."""
    print("\n" + "=" * 60)
    print("CLAIM COHORT MANAGEMENT")
    print("=" * 60)

    # Create a cohort for 2020 accident year
    cohort = ClaimCohort(accident_year=2020)

    # Add various types of claims
    claims_data = [
        ("Property damage", 500_000, ClaimDevelopment.create_immediate()),
        ("Workers compensation", 250_000, ClaimDevelopment.create_medium_tail_5yr()),
        ("General liability", 1_000_000, ClaimDevelopment.create_long_tail_10yr()),
        ("Product liability", 2_000_000, ClaimDevelopment.create_very_long_tail_15yr()),
    ]

    for i, (claim_type, amount, pattern) in enumerate(claims_data):
        claim = Claim(
            claim_id=f"CL{i+1:03d}",
            accident_year=2020,
            reported_year=2020,
            initial_estimate=amount,
            claim_type=claim_type,
            development_pattern=pattern,
        )
        cohort.add_claim(claim)

    print(f"\nCohort Summary (Accident Year {cohort.accident_year}):")
    print(f"  Total claims: {len(cohort.claims)}")
    print(f"  Total incurred: ${cohort.get_total_incurred():,.0f}")

    # Calculate payments for first 5 years
    print("\nPayment Schedule:")
    for year in range(2020, 2025):
        payment = cohort.calculate_payments(year)
        print(f"  {year}: ${payment:,.0f}")

    print(f"\nAfter 5 years:")
    print(f"  Total paid: ${cohort.get_total_paid():,.0f}")
    print(f"  Outstanding reserve: ${cohort.get_outstanding_reserve():,.0f}")


def demo_cash_flow_projection():
    """Demonstrate multi-year cash flow projections."""
    print("\n" + "=" * 60)
    print("CASH FLOW PROJECTIONS")
    print("=" * 60)

    projector = CashFlowProjector(discount_rate=0.03)

    # Create multiple accident year cohorts
    for accident_year in [2018, 2019, 2020]:
        cohort = ClaimCohort(accident_year=accident_year)

        # Add random mix of claims
        base_amount = 500_000 * (1.1 ** (accident_year - 2018))  # Inflation

        claims = [
            Claim(
                f"CL_{accident_year}_001",
                accident_year,
                accident_year,
                base_amount * 0.5,
                "property",
                ClaimDevelopment.create_immediate(),
            ),
            Claim(
                f"CL_{accident_year}_002",
                accident_year,
                accident_year,
                base_amount * 1.0,
                "liability",
                ClaimDevelopment.create_long_tail_10yr(),
            ),
            Claim(
                f"CL_{accident_year}_003",
                accident_year,
                accident_year,
                base_amount * 1.5,
                "product",
                ClaimDevelopment.create_very_long_tail_15yr(),
            ),
        ]

        for claim in claims:
            cohort.add_claim(claim)

        projector.add_cohort(cohort)
        print(
            f"\n{accident_year} Cohort: {len(cohort.claims)} claims, ${cohort.get_total_incurred():,.0f} total"
        )

    # Project payments for next 10 years
    payments = projector.project_payments(2020, 2029)

    print("\n10-Year Payment Projection:")
    total_payments = 0.0
    for year, amount in payments.items():
        total_payments += amount
        print(f"  {year}: ${amount:,.0f}")

    print(f"\nTotal projected payments: ${total_payments:,.0f}")

    # Calculate present value
    pv = projector.calculate_present_value(payments, base_year=2020)
    print(f"Present value (3% discount): ${pv:,.0f}")

    # Estimate reserves
    reserves = projector.calculate_total_reserves(evaluation_year=2020)
    print(f"\n2020 Reserve Requirements:")
    print(f"  Case reserves: ${reserves['case_reserves']:,.0f}")
    print(f"  IBNR: ${reserves['ibnr']:,.0f}")
    print(f"  Total reserves: ${reserves['total_reserves']:,.0f}")


def demo_manufacturer_integration():
    """Demonstrate integration with WidgetManufacturer."""
    print("\n" + "=" * 60)
    print("MANUFACTURER INTEGRATION")
    print("=" * 60)

    # Create manufacturer
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,
    )
    manufacturer = WidgetManufacturer(config)
    print(f"\nInitial state:")
    print(f"  Assets: ${manufacturer.assets:,.0f}")
    print(f"  Equity: ${manufacturer.equity:,.0f}")

    # Process a large liability claim with development pattern
    claim_amount = 5_000_000
    deductible = 500_000
    insurance_limit = 3_000_000
    pattern = ClaimDevelopment.create_long_tail_10yr()

    print(f"\nProcessing ${claim_amount:,.0f} liability claim:")
    print(f"  Deductible: ${deductible:,.0f}")
    print(f"  Insurance limit: ${insurance_limit:,.0f}")
    print(f"  Development: {pattern.pattern_name}")

    (
        company_payment,
        insurance_payment,
        claim_obj,
    ) = manufacturer.process_insurance_claim_with_development(
        claim_amount=claim_amount,
        deductible=deductible,
        insurance_limit=insurance_limit,
        development_pattern=pattern,
        claim_type="general_liability",
    )

    print(f"\nClaim breakdown:")
    print(f"  Company pays (immediate): ${company_payment:,.0f}")
    print(f"  Insurance covers: ${insurance_payment:,.0f}")

    if claim_obj:
        print(f"\nClaim development schedule:")
        for year in range(5):
            payment = pattern.calculate_payments(insurance_payment, 0, year)
            print(f"  Year {year + 1}: ${payment:,.0f}")

    print(f"\nPost-claim state:")
    print(f"  Assets: ${manufacturer.assets:,.0f}")
    print(f"  Equity: ${manufacturer.equity:,.0f}")
    print(f"  Collateral posted: ${manufacturer.collateral:,.0f}")


def demo_yaml_configuration():
    """Demonstrate loading patterns from YAML configuration."""
    print("\n" + "=" * 60)
    print("YAML CONFIGURATION LOADING")
    print("=" * 60)

    # Load patterns from configuration file
    config_path = Path(__file__).parent.parent / "data" / "parameters" / "development_patterns.yaml"

    if config_path.exists():
        patterns = load_development_patterns(str(config_path))

        print(f"\nLoaded {len(patterns)} patterns from configuration:")
        for name, pattern in patterns.items():
            total_pct = sum(pattern.development_factors) + pattern.tail_factor
            print(
                f"  - {name}: {len(pattern.development_factors)} years, "
                f"tail={pattern.tail_factor:.1%}, total={total_pct:.1%}"
            )
    else:
        print(f"\nConfiguration file not found at: {config_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CLAIM DEVELOPMENT PATTERNS DEMONSTRATION")
    print("=" * 60)

    # Run all demonstrations
    demo_basic_development_patterns()
    demo_claim_cohort()
    demo_cash_flow_projection()
    demo_manufacturer_integration()
    demo_yaml_configuration()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

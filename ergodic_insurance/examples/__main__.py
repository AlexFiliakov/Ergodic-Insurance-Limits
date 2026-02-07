"""Run examples: python -m ergodic_insurance.examples.<name>

Available examples:
    demo_manufacturer          - Basic WidgetManufacturer financial model
    demo_collateral_management - Collateral management with letters of credit
    demo_stochastic            - Stochastic processes comparison
    demo_claim_development     - Claim development patterns and cash flow modeling
    demo_config_v2             - ConfigManager system demonstration
    demo_config_practical      - Practical configuration scenarios
    demo_excel_reports         - Excel report generation
    demo_insurance_pricing     - Insurance pricing with market cycles
    benchmark_parallel         - Parallel Monte Carlo performance benchmark

Usage:
    python -m ergodic_insurance.examples                          # Show this help
    python -m ergodic_insurance.examples.demo_manufacturer        # Run a specific example
"""

import sys


def main():
    """Print usage information for the examples subpackage."""
    print(__doc__)
    examples = [
        "demo_manufacturer",
        "demo_collateral_management",
        "demo_stochastic",
        "demo_claim_development",
        "demo_config_v2",
        "demo_config_practical",
        "demo_excel_reports",
        "demo_insurance_pricing",
        "benchmark_parallel",
    ]
    print("To run an example:")
    for name in examples:
        print(f"    python -m ergodic_insurance.examples.{name}")


if __name__ == "__main__":
    main()

"""Demonstration of Excel report generation for financial statements.

This script shows how to generate comprehensive Excel reports containing
balance sheets, income statements, cash flow statements, and reconciliation
reports from simulation results.

Example:
    Run the demonstration::

        python demo_excel_reports.py
"""

from pathlib import Path

import numpy as np

from ergodic_insurance import (
    InsuranceProgram,
    ManufacturerConfig,
    ManufacturingLossGenerator,
    WidgetManufacturer,
)
from ergodic_insurance.excel_reporter import ExcelReportConfig, ExcelReporter
from ergodic_insurance.financial_statements import (
    FinancialStatementConfig,
    FinancialStatementGenerator,
)


def run_simulation_with_claims(years: int = 10, seed: int = 42) -> WidgetManufacturer:
    """Run a simulation with insurance claims.

    Args:
        years: Number of years to simulate
        seed: Random seed for reproducibility

    Returns:
        WidgetManufacturer with simulation results
    """
    # Note: seed is passed directly to ManufacturingLossGenerator below
    # No module-level np.random.seed() needed

    # Configure manufacturer
    manufacturer_config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.5,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,  # Added missing required field
    )

    manufacturer = WidgetManufacturer(manufacturer_config)

    # Configure loss generator
    # Loss frequency scales with revenue (more activity = more risk exposure)
    base_frequency = 3.0  # Base frequency for $10M revenue company
    initial_revenue = manufacturer_config.initial_assets * manufacturer_config.asset_turnover_ratio

    loss_generator = ManufacturingLossGenerator.create_simple(
        frequency=base_frequency * (initial_revenue / 10_000_000),  # Scale with revenue
        severity_mean=500_000,  # Average severity
        severity_std=1_000_000,  # Standard deviation
        seed=seed,
    )

    # Configure insurance
    insurance = InsuranceProgram.simple(
        deductible=100_000,
        limit=5_000_000,
        rate=0.02,
    )

    # Run simulation
    for year in range(years):
        # Get current revenue for frequency scaling
        current_revenue = manufacturer.assets * manufacturer_config.asset_turnover_ratio

        # Generate losses for the year based on current revenue
        loss_events, _ = loss_generator.generate_losses(duration=1, revenue=current_revenue)
        claims = [loss.amount for loss in loss_events]

        # Process claims through insurance
        total_claims = sum(claims)
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=total_claims,
            deductible_amount=insurance.deductible,
            insurance_limit=insurance.get_total_coverage(),
        )

        # Calculate insurance premium
        insurance_premium = insurance.calculate_premium()

        # Run annual step
        metrics = manufacturer.step(letter_of_credit_rate=0.015, growth_rate=0.03)

        # Deduct insurance costs
        manufacturer.assets -= insurance_premium
        manufacturer.equity -= insurance_premium

        print(
            f"Year {year}: Revenue=${metrics['revenue']:,.0f}, "
            f"Net Income=${metrics['net_income']:,.0f}, "
            f"Claims=${total_claims:,.0f}, "
            f"Equity=${metrics['equity']:,.0f}"
        )

    return manufacturer


def generate_basic_excel_report(manufacturer: WidgetManufacturer, output_dir: Path) -> None:
    """Generate basic Excel report with pandas.

    Args:
        manufacturer: Manufacturer with simulation data
        output_dir: Output directory for reports
    """
    print("\n=== Generating Basic Excel Report ===")

    # Create statement generator
    stmt_generator = FinancialStatementGenerator(manufacturer=manufacturer)

    # Generate statements for the last year
    last_year = len(manufacturer.metrics_history) - 1

    balance_sheet = stmt_generator.generate_balance_sheet(last_year)
    income_statement = stmt_generator.generate_income_statement(last_year)
    cash_flow = stmt_generator.generate_cash_flow_statement(last_year)
    reconciliation = stmt_generator.generate_reconciliation_report(last_year)

    # Save to Excel
    output_file = output_dir / "basic_financial_statements.xlsx"
    with pd.ExcelWriter(
        output_file, engine="openpyxl" if pd.__version__ >= "1.0.0" else None
    ) as writer:
        balance_sheet.to_excel(writer, sheet_name="Balance Sheet", index=False)
        income_statement.to_excel(writer, sheet_name="Income Statement", index=False)
        cash_flow.to_excel(writer, sheet_name="Cash Flow", index=False)
        reconciliation.to_excel(writer, sheet_name="Reconciliation", index=False)

    print(f"Basic report saved to: {output_file}")


def generate_comprehensive_excel_report(manufacturer: WidgetManufacturer, output_dir: Path) -> None:
    """Generate comprehensive Excel report with formatting.

    Args:
        manufacturer: Manufacturer with simulation data
        output_dir: Output directory for reports
    """
    print("\n=== Generating Comprehensive Excel Report ===")

    # Configure Excel reporter
    config = ExcelReportConfig(
        output_path=output_dir,
        include_balance_sheet=True,
        include_income_statement=True,
        include_cash_flow=True,
        include_reconciliation=True,
        include_metrics_dashboard=True,
        include_pivot_data=True,
        engine="auto",  # Will use XlsxWriter if available
        currency_format="$#,##0",
        decimal_places=0,
    )

    # Create reporter
    reporter = ExcelReporter(config)

    # Generate report
    output_file = reporter.generate_trajectory_report(
        manufacturer,
        "comprehensive_financial_report.xlsx",
        title="Widget Manufacturing Company - Financial Analysis",
    )

    print(f"Comprehensive report saved to: {output_file}")
    print(f"Report engine used: {reporter.engine}")


def generate_multi_year_comparison(manufacturer: WidgetManufacturer, output_dir: Path) -> None:
    """Generate multi-year comparison report.

    Args:
        manufacturer: Manufacturer with simulation data
        output_dir: Output directory for reports
    """
    print("\n=== Generating Multi-Year Comparison Report ===")

    # Create statement generator
    stmt_generator = FinancialStatementGenerator(manufacturer=manufacturer)

    # Prepare data for all years
    all_balance_sheets = []
    all_income_statements = []
    metrics_summary = []

    for year in range(len(manufacturer.metrics_history)):
        # Generate statements
        bs = stmt_generator.generate_balance_sheet(year)
        inc = stmt_generator.generate_income_statement(year)

        # Extract key metrics
        metrics = manufacturer.metrics_history[year]
        summary = {
            "Year": year,
            "Assets": metrics.get("assets", 0),
            "Equity": metrics.get("equity", 0),
            "Revenue": metrics.get("revenue", 0),
            "Operating Income": metrics.get("operating_income", 0),
            "Net Income": metrics.get("net_income", 0),
            "ROE %": metrics.get("roe", 0) * 100,
            "ROA %": metrics.get("roa", 0) * 100,
            "Base Operating Margin %": metrics.get("base_operating_margin", 0) * 100,
            "Collateral": metrics.get("collateral", 0),
            "Claim Liabilities": metrics.get("claim_liabilities", 0),
        }
        metrics_summary.append(summary)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(metrics_summary)

    # Calculate growth rates
    if len(comparison_df) > 1:
        comparison_df["Revenue Growth %"] = comparison_df["Revenue"].pct_change() * 100
        comparison_df["Asset Growth %"] = comparison_df["Assets"].pct_change() * 100
        comparison_df["Equity Growth %"] = comparison_df["Equity"].pct_change() * 100

    # Save to Excel with formatting
    output_file = output_dir / "multi_year_comparison.xlsx"
    with pd.ExcelWriter(
        output_file, engine="openpyxl" if pd.__version__ >= "1.0.0" else None
    ) as writer:
        comparison_df.to_excel(writer, sheet_name="Multi-Year Summary", index=False)

        # Add conditional formatting if using openpyxl
        if "openpyxl" in str(type(writer)):
            worksheet = writer.sheets["Multi-Year Summary"]

            # Format currency columns
            for col in [
                "Assets",
                "Equity",
                "Revenue",
                "Operating Income",
                "Net Income",
                "Collateral",
                "Claim Liabilities",
            ]:
                col_idx = comparison_df.columns.get_loc(col) + 1
                for row in range(2, len(comparison_df) + 2):
                    cell = worksheet.cell(row=row, column=col_idx)
                    cell.number_format = "$#,##0"

    print(f"Multi-year comparison saved to: {output_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Initial Assets: ${comparison_df['Assets'].iloc[0]:,.0f}")
    print(f"Final Assets: ${comparison_df['Assets'].iloc[-1]:,.0f}")
    print(
        f"Total Growth: {(comparison_df['Assets'].iloc[-1] / comparison_df['Assets'].iloc[0] - 1) * 100:.1f}%"
    )
    print(f"Average ROE: {comparison_df['ROE %'].mean():.1f}%")
    print(f"Average Operating Margin: {comparison_df['Operating Margin %'].mean():.1f}%")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Excel Report Generation Demonstration")
    print("=" * 60)

    # Create output directory
    output_dir = Path("./excel_reports_demo")
    output_dir.mkdir(exist_ok=True)

    # Run simulation
    print("\nRunning 10-year simulation with insurance claims...")
    manufacturer = run_simulation_with_claims(years=10)

    # Generate different types of reports
    generate_basic_excel_report(manufacturer, output_dir)
    generate_comprehensive_excel_report(manufacturer, output_dir)
    generate_multi_year_comparison(manufacturer, output_dir)

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print(f"All reports saved to: {output_dir.absolute()}")
    print("=" * 60)

    # Display final metrics
    final_metrics = manufacturer.metrics_history[-1]
    print("\nFinal Year Metrics:")
    print(f"  Assets: ${final_metrics['assets']:,.0f}")
    print(f"  Equity: ${final_metrics['equity']:,.0f}")
    print(f"  Revenue: ${final_metrics['revenue']:,.0f}")
    print(f"  Net Income: ${final_metrics['net_income']:,.0f}")
    print(f"  ROE: {final_metrics['roe'] * 100:.1f}%")
    print(f"  Solvency: {'Yes' if final_metrics['is_solvent'] else 'No'}")


if __name__ == "__main__":
    main()

"""Demo script showing insurance pricing with market cycle support.

This script demonstrates:
1. Pricing individual layers using frequency/severity distributions
2. Comparing premiums across different market cycles
3. Dynamic pricing of complete insurance programs
4. Market cycle transitions over time
5. Comparison of fixed vs calculated premiums
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from ergodic_insurance import (
    EnhancedInsuranceLayer,
    InsurancePricer,
    InsuranceProgram,
    ManufacturingLossGenerator,
    MarketCycle,
)
from ergodic_insurance.insurance_pricing import PricingParameters
from ergodic_insurance.insurance_program import ReinstatementType

console = Console()


def demo_layer_pricing():
    """Demonstrate pricing a single insurance layer."""
    console.print("\n[bold cyan]Demo 1: Pricing a Single Insurance Layer[/bold cyan]")
    console.print("=" * 60)

    # Create loss generator
    loss_gen = ManufacturingLossGenerator(seed=42)

    # Create pricer with normal market conditions
    pricer = InsurancePricer(
        loss_generator=loss_gen,
        market_cycle=MarketCycle.NORMAL,
        seed=42,
    )

    # Price a primary layer
    console.print("\n[yellow]Pricing Primary Layer:[/yellow]")
    console.print("  Attachment: $250,000")
    console.print("  Limit: $4,750,000")
    console.print("  Expected Revenue: $15,000,000")

    pricing = pricer.price_layer(
        attachment_point=250_000,
        limit=4_750_000,
        expected_revenue=15_000_000,
    )

    # Display results
    table = Table(title="Primary Layer Pricing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Expected Frequency", f"{pricing.expected_frequency:.2f} claims/year")
    table.add_row("Expected Severity", f"${pricing.expected_severity:,.0f}")
    table.add_row("Pure Premium", f"${pricing.pure_premium:,.0f}")
    table.add_row("Technical Premium", f"${pricing.technical_premium:,.0f}")
    table.add_row("Market Premium", f"${pricing.market_premium:,.0f}")
    table.add_row("Rate on Line", f"{pricing.rate_on_line:.2%}")
    table.add_row(
        "Confidence Interval",
        f"${pricing.confidence_interval[0]:,.0f} - ${pricing.confidence_interval[1]:,.0f}",
    )

    console.print(table)


def demo_market_cycle_comparison():
    """Compare pricing across different market cycles."""
    console.print("\n[bold cyan]Demo 2: Market Cycle Comparison[/bold cyan]")
    console.print("=" * 60)

    # Create loss generator and pricer
    loss_gen = ManufacturingLossGenerator(seed=42)
    pricer = InsurancePricer(loss_generator=loss_gen, seed=42)

    # Compare pricing for excess layer
    console.print("\n[yellow]Comparing Excess Layer Pricing Across Market Cycles:[/yellow]")
    console.print("  Attachment: $5,000,000")
    console.print("  Limit: $20,000,000")
    console.print("  Expected Revenue: $15,000,000")

    results = pricer.compare_market_cycles(
        attachment_point=5_000_000,
        limit=20_000_000,
        expected_revenue=15_000_000,
    )

    # Display comparison
    table = Table(title="Market Cycle Premium Comparison")
    table.add_column("Market Cycle", style="cyan")
    table.add_column("Loss Ratio", style="yellow", justify="right")
    table.add_column("Market Premium", style="green", justify="right")
    table.add_column("Rate on Line", style="magenta", justify="right")

    for cycle_name, pricing in results.items():
        cycle = MarketCycle[cycle_name]
        table.add_row(
            cycle_name,
            f"{cycle.value:.0%}",
            f"${pricing.market_premium:,.0f}",
            f"{pricing.rate_on_line:.3%}",
        )

    console.print(table)

    # Calculate percentage differences
    hard_premium = results["HARD"].market_premium
    normal_premium = results["NORMAL"].market_premium
    soft_premium = results["SOFT"].market_premium

    console.print("\n[yellow]Premium Differences:[/yellow]")
    console.print(f"  Hard vs Normal: {(hard_premium/normal_premium - 1)*100:+.1f}%")
    console.print(f"  Normal vs Soft: {(normal_premium/soft_premium - 1)*100:+.1f}%")
    console.print(f"  Hard vs Soft: {(hard_premium/soft_premium - 1)*100:+.1f}%")


def demo_program_pricing():
    """Demonstrate pricing a complete insurance program."""
    console.print("\n[bold cyan]Demo 3: Complete Insurance Program Pricing[/bold cyan]")
    console.print("=" * 60)

    # Create loss generator
    loss_gen = ManufacturingLossGenerator(seed=42)

    # Create initial program with fixed rates
    layers = [
        EnhancedInsuranceLayer(
            attachment_point=250_000,
            limit=4_750_000,
            base_premium_rate=0.015,  # 1.5% fixed rate
            reinstatements=0,
        ),
        EnhancedInsuranceLayer(
            attachment_point=5_000_000,
            limit=20_000_000,
            base_premium_rate=0.008,  # 0.8% fixed rate
            reinstatements=1,
            reinstatement_type=ReinstatementType.FULL,
        ),
        EnhancedInsuranceLayer(
            attachment_point=25_000_000,
            limit=25_000_000,
            base_premium_rate=0.004,  # 0.4% fixed rate
            reinstatements=2,
            reinstatement_type=ReinstatementType.PRO_RATA,
        ),
    ]

    # Create program with pricing enabled
    program = InsuranceProgram.create_with_pricing(
        layers=layers,
        loss_generator=loss_gen,
        expected_revenue=15_000_000,
        market_cycle=MarketCycle.NORMAL,
        deductible=250_000,
        name="Manufacturing Insurance Program",
    )

    # Get pricing summary
    summary = program.get_pricing_summary()

    console.print(f"\n[yellow]Program: {summary['program_name']}[/yellow]")
    console.print(f"Total Premium: ${summary['total_premium']:,.0f}")

    # Display layer pricing
    table = Table(title="Layer-by-Layer Pricing")
    table.add_column("Layer", style="cyan")
    table.add_column("Attachment", style="yellow", justify="right")
    table.add_column("Limit", style="yellow", justify="right")
    table.add_column("Fixed Rate", style="red", justify="right")
    table.add_column("Calculated Rate", style="green", justify="right")
    table.add_column("Premium", style="magenta", justify="right")

    original_rates = [0.015, 0.008, 0.004]  # Original fixed rates

    for i, layer_info in enumerate(summary["layers"]):
        table.add_row(
            f"Layer {i+1}",
            f"${layer_info['attachment_point']:,.0f}",
            f"${layer_info['limit']:,.0f}",
            f"{original_rates[i]:.3%}",
            f"{layer_info['base_premium_rate']:.3%}",
            f"${layer_info['market_premium']:,.0f}",
        )

    console.print(table)


def demo_cycle_transitions():
    """Demonstrate premium changes during market cycle transitions."""
    console.print("\n[bold cyan]Demo 4: Market Cycle Transitions Over Time[/bold cyan]")
    console.print("=" * 60)

    # Create loss generator and pricer
    loss_gen = ManufacturingLossGenerator(seed=42)
    pricer = InsurancePricer(
        loss_generator=loss_gen,
        market_cycle=MarketCycle.NORMAL,
        seed=42,
    )

    # Create a simple program
    program = InsuranceProgram.create_standard_manufacturing_program()

    # Simulate 10 years of market cycles
    results = pricer.simulate_cycle_transition(
        program=program,
        expected_revenue=15_000_000,
        years=10,
    )

    # Display results
    table = Table(title="10-Year Market Cycle Simulation")
    table.add_column("Year", style="cyan", justify="center")
    table.add_column("Market Cycle", style="yellow")
    table.add_column("Loss Ratio", style="magenta", justify="right")
    table.add_column("Total Premium", style="green", justify="right")

    for result in results:
        table.add_row(
            str(result["year"] + 1),
            result["market_cycle"],
            f"{result['loss_ratio']:.0%}",
            f"${result['total_premium']:,.0f}",
        )

    console.print(table)

    # Calculate statistics
    premiums = [r["total_premium"] for r in results]
    console.print("\n[yellow]Premium Statistics Over 10 Years:[/yellow]")
    console.print(f"  Minimum: ${min(premiums):,.0f}")
    console.print(f"  Maximum: ${max(premiums):,.0f}")
    console.print(f"  Average: ${np.mean(premiums):,.0f}")
    console.print(f"  Std Dev: ${np.std(premiums):,.0f}")
    console.print(f"  Range: {(max(premiums)/min(premiums) - 1)*100:.1f}%")


def demo_fixed_vs_calculated():
    """Compare fixed premium rates vs calculated rates."""
    console.print("\n[bold cyan]Demo 5: Fixed vs Calculated Premium Comparison[/bold cyan]")
    console.print("=" * 60)

    # Create loss generator
    loss_gen = ManufacturingLossGenerator(seed=42)

    # Create program with fixed rates
    fixed_program = InsuranceProgram.create_standard_manufacturing_program()
    fixed_premium = fixed_program.calculate_annual_premium()

    # Create identical structure with calculated rates
    layers = []
    for layer in fixed_program.layers:
        layers.append(
            EnhancedInsuranceLayer(
                attachment_point=layer.attachment_point,
                limit=layer.limit,
                base_premium_rate=layer.base_premium_rate,  # Will be overridden
                reinstatements=layer.reinstatements,
                reinstatement_type=layer.reinstatement_type,
            )
        )

    # Price in different market conditions
    market_premiums = {}

    for cycle in MarketCycle:
        priced_program = InsuranceProgram.create_with_pricing(
            layers=layers,
            loss_generator=loss_gen,
            expected_revenue=15_000_000,
            market_cycle=cycle,
            deductible=fixed_program.deductible,
        )
        market_premiums[cycle.name] = priced_program.calculate_annual_premium()

    # Display comparison
    table = Table(title="Fixed vs Calculated Premium Comparison")
    table.add_column("Premium Type", style="cyan")
    table.add_column("Annual Premium", style="green", justify="right")
    table.add_column("Difference from Fixed", style="yellow", justify="right")

    table.add_row(
        "Fixed Rate",
        f"${fixed_premium:,.0f}",
        "Baseline",
    )

    for cycle_name, premium in market_premiums.items():
        diff_pct = (premium / fixed_premium - 1) * 100
        table.add_row(
            f"Calculated ({cycle_name})",
            f"${premium:,.0f}",
            f"{diff_pct:+.1f}%",
        )

    console.print(table)

    # Analysis
    console.print("\n[yellow]Analysis:[/yellow]")
    console.print("• Fixed rates may not reflect actual risk exposure")
    console.print("• Market cycles significantly impact premium requirements")
    console.print("• Dynamic pricing adapts to changing market conditions")
    console.print("• Calculated rates provide more accurate risk transfer pricing")


def visualize_premium_impact():
    """Create visualization of premium impact across market cycles."""
    console.print("\n[bold cyan]Creating Premium Impact Visualization...[/bold cyan]")

    # Create loss generator
    loss_gen = ManufacturingLossGenerator(seed=42)

    # Test different attachment points
    attachments = [100_000, 250_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    limit = 5_000_000  # Fixed limit for comparison

    # Calculate premiums for each attachment and market cycle
    results = {cycle.name: [] for cycle in MarketCycle}

    for attachment in track(attachments, description="Calculating premiums..."):
        pricer = InsurancePricer(loss_generator=loss_gen, seed=42)
        cycle_results = pricer.compare_market_cycles(
            attachment_point=attachment,
            limit=limit,
            expected_revenue=15_000_000,
        )

        for cycle_name, pricing in cycle_results.items():
            results[cycle_name].append(pricing.rate_on_line)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Rate on Line by Attachment Point
    for cycle_name, rates in results.items():
        ax1.plot(attachments, rates, marker="o", label=cycle_name, linewidth=2)

    ax1.set_xlabel("Attachment Point ($)", fontsize=11)
    ax1.set_ylabel("Rate on Line (%)", fontsize=11)
    ax1.set_title(
        "Premium Rates by Attachment Point and Market Cycle", fontsize=12, fontweight="bold"
    )
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

    # Plot 2: Premium Multiplier
    base_rates = results["NORMAL"]
    multipliers = {
        "HARD": [h / n for h, n in zip(results["HARD"], base_rates)],
        "SOFT": [s / n for s, n in zip(results["SOFT"], base_rates)],
    }

    x = np.arange(len(attachments))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, multipliers["HARD"], width, label="Hard Market", color="#e74c3c")
    bars2 = ax2.bar(x + width / 2, multipliers["SOFT"], width, label="Soft Market", color="#3498db")

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Normal Market")
    ax2.set_xlabel("Attachment Point ($)", fontsize=11)
    ax2.set_ylabel("Premium Multiplier vs Normal", fontsize=11)
    ax2.set_title("Market Cycle Premium Impact", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"${a/1e6:.1f}M" for a in attachments])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("insurance_pricing_analysis.png", dpi=150, bbox_inches="tight")
    console.print("[green]✓ Visualization saved as 'insurance_pricing_analysis.png'[/green]")
    plt.show()


def main():
    """Run all demonstrations."""
    console.print(
        Panel.fit(
            "[bold cyan]Insurance Pricing Module Demonstration[/bold cyan]\n"
            "Showcasing frequency/severity-based premium calculation with market cycle support",
            border_style="cyan",
        )
    )

    try:
        # Run demonstrations
        demo_layer_pricing()
        demo_market_cycle_comparison()
        demo_program_pricing()
        demo_cycle_transitions()
        demo_fixed_vs_calculated()
        visualize_premium_impact()

        console.print("\n[bold green]✓ All demonstrations completed successfully![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]Error during demonstration: {e}[/bold red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

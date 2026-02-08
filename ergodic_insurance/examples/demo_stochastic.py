#!/usr/bin/env python3
"""Demo script showing stochastic processes in action."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ergodic_insurance import ManufacturerConfig, WidgetManufacturer
from ergodic_insurance.stochastic_processes import (
    LognormalVolatility,
    StochasticConfig,
    create_stochastic_process,
)


def run_stochastic_comparison():
    """Compare deterministic vs stochastic simulations."""

    # Configuration
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=1.0,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,
    )

    # Time horizon
    years = 50

    # Run deterministic simulation
    print("Running deterministic simulation...")
    manufacturer_det = WidgetManufacturer(config)
    results_det = []
    for _ in range(years):
        metrics = manufacturer_det.step(growth_rate=0.05)
        results_det.append(metrics)

    # Run multiple stochastic simulations
    print("Running stochastic simulations...")
    n_sims = 10
    stochastic_results = []

    for sim in range(n_sims):
        # Create stochastic process with different seed for each simulation
        stoch_config = StochasticConfig(volatility=0.15, drift=0.0, random_seed=42 + sim)
        process = LognormalVolatility(stoch_config)

        # Create manufacturer with stochastic process
        manufacturer_stoch = WidgetManufacturer(config, stochastic_process=process)

        # Run simulation
        sim_results = []
        for _ in range(years):
            metrics = manufacturer_stoch.step(growth_rate=0.05, apply_stochastic=True)
            sim_results.append(metrics)

        stochastic_results.append(sim_results)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Assets over time
    ax = axes[0, 0]
    years_range = range(years)

    # Plot deterministic
    det_assets = [m["assets"] for m in results_det]
    ax.plot(years_range, det_assets, "k-", linewidth=2, label="Deterministic")

    # Plot stochastic paths
    for i, sim_results in enumerate(stochastic_results):
        stoch_assets = [m["assets"] for m in sim_results]
        ax.plot(years_range, stoch_assets, alpha=0.5, linewidth=0.5)

    ax.set_xlabel("Year")
    ax.set_ylabel("Assets ($)")
    ax.set_title("Asset Growth: Deterministic vs Stochastic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Revenue over time
    ax = axes[0, 1]

    det_revenue = [m["revenue"] for m in results_det]
    ax.plot(years_range, det_revenue, "k-", linewidth=2, label="Deterministic")

    for sim_results in stochastic_results:
        stoch_revenue = [m["revenue"] for m in sim_results]
        ax.plot(years_range, stoch_revenue, alpha=0.5, linewidth=0.5)

    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Revenue: Deterministic vs Stochastic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Distribution of final assets
    ax = axes[1, 0]

    final_assets_stoch = [sim[-1]["assets"] for sim in stochastic_results]
    final_assets_det = results_det[-1]["assets"]

    ax.hist(final_assets_stoch, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(
        final_assets_det,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Deterministic: ${final_assets_det/1e6:.1f}M",
    )
    ax.axvline(
        np.mean(final_assets_stoch),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Stochastic Mean: ${np.mean(final_assets_stoch)/1e6:.1f}M",
    )

    ax.set_xlabel("Final Assets ($)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Final Assets (Year {years})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Volatility impact
    ax = axes[1, 1]

    # Calculate annual returns for each simulation
    for sim_results in stochastic_results:
        assets = [m["assets"] for m in sim_results]
        returns = [(assets[i + 1] - assets[i]) / assets[i] for i in range(len(assets) - 1)]
        ax.plot(range(len(returns)), returns, alpha=0.3, linewidth=0.5)

    # Deterministic returns
    det_returns = [
        (det_assets[i + 1] - det_assets[i]) / det_assets[i] for i in range(len(det_assets) - 1)
    ]
    ax.plot(range(len(det_returns)), det_returns, "k-", linewidth=2, label="Deterministic")

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Return")
    ax.set_title("Annual Returns: Impact of Stochasticity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("stochastic_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()  # Close without showing to avoid blocking

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Time Horizon: {years} years")
    print(f"Number of Stochastic Simulations: {n_sims}")
    print(f"Volatility: 15% annual")
    print(f"Growth Rate: 5% annual")

    print("\nFINAL ASSETS:")
    print(f"  Deterministic:     ${final_assets_det/1e6:,.1f}M")
    print(f"  Stochastic Mean:   ${np.mean(final_assets_stoch)/1e6:,.1f}M")
    print(f"  Stochastic Median: ${np.median(final_assets_stoch)/1e6:,.1f}M")
    print(f"  Stochastic StdDev: ${np.std(final_assets_stoch)/1e6:,.1f}M")
    print(f"  Stochastic Min:    ${np.min(final_assets_stoch)/1e6:,.1f}M")
    print(f"  Stochastic Max:    ${np.max(final_assets_stoch)/1e6:,.1f}M")

    # Calculate growth rates
    det_growth = (final_assets_det / config.initial_assets) ** (1 / years) - 1
    stoch_growth_rates = [
        (fa / config.initial_assets) ** (1 / years) - 1 for fa in final_assets_stoch
    ]

    print("\nANNUALIZED GROWTH RATES:")
    print(f"  Deterministic:     {det_growth:.2%}")
    print(f"  Stochastic Mean:   {np.mean(stoch_growth_rates):.2%}")
    print(f"  Stochastic Median: {np.median(stoch_growth_rates):.2%}")

    # Demonstrate ergodic theory insight
    print("\nERGODIC THEORY INSIGHT:")
    print("  Time Average (single path over time) ≠ Ensemble Average (many paths at one time)")
    print(f"  Ensemble average final assets: ${np.mean(final_assets_stoch)/1e6:,.1f}M")

    # Calculate time average for one path
    single_path_assets = [m["assets"] for m in stochastic_results[0]]
    log_returns = [
        np.log(single_path_assets[i + 1] / single_path_assets[i])
        for i in range(len(single_path_assets) - 1)
    ]
    time_avg_growth = np.mean(log_returns)
    print(f"  Time average growth (single path): {time_avg_growth:.2%}")

    # Calculate ensemble average at final time
    ensemble_avg_growth = np.mean(stoch_growth_rates)
    print(f"  Ensemble average growth (all paths): {ensemble_avg_growth:.2%}")

    print("\n" + "=" * 60)


def demonstrate_different_processes():
    """Show different stochastic process types."""

    print("\n" + "=" * 60)
    print("COMPARING STOCHASTIC PROCESS TYPES")
    print("=" * 60)

    # Test parameters
    n_steps = 100
    initial_value = 100

    # Create different processes
    processes = {
        "GBM (drift=5%, vol=20%)": create_stochastic_process(
            "gbm", volatility=0.2, drift=0.05, random_seed=42
        ),
        "Lognormal (vol=15%)": create_stochastic_process(
            "lognormal", volatility=0.15, random_seed=42
        ),
        "Mean Reverting (vol=10%)": create_stochastic_process(
            "mean_reverting", volatility=0.1, random_seed=42
        ),
    }

    # Generate paths
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, process in processes.items():
        path = [initial_value]
        for _ in range(n_steps):
            shock = process.generate_shock(path[-1])
            path.append(path[-1] * shock)

        ax.plot(range(len(path)), path, label=name, linewidth=2)

        # Print statistics
        final_value = path[-1]
        total_return = (final_value / initial_value - 1) * 100
        annualized_return = ((final_value / initial_value) ** (1 / n_steps) - 1) * 100

        print(f"\n{name}:")
        print(f"  Initial Value: {initial_value:.2f}")
        print(f"  Final Value:   {final_value:.2f}")
        print(f"  Total Return:  {total_return:.1f}%")
        print(f"  Annualized:    {annualized_return:.1f}%")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Comparison of Stochastic Process Types")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("process_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()  # Close without showing to avoid blocking

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Stochastic Processes Demo")
    print("=" * 60)

    # Run comparison
    run_stochastic_comparison()

    # Show different process types
    demonstrate_different_processes()

    print("\nDemo complete! Check generated plots for visualizations.")

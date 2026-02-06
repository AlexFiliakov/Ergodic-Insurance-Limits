"""Generate missing documentation images for GitHub Pages."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Create directories if they don't exist
assets_dir = Path("assets")
theory_figures_dir = Path("theory/figures")
assets_dir.mkdir(exist_ok=True)
theory_figures_dir.mkdir(parents=True, exist_ok=True)

# Set style for professional looking plots
plt.style.use("seaborn-v0_8-darkgrid")


# Generate ergodic_distinction.png
def create_ergodic_distinction():
    """Create visual showing difference between time and ensemble averages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Simulate multiple paths
    np.random.seed(42)
    n_paths = 20
    n_steps = 100

    # Generate paths with multiplicative dynamics
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = 1.0

    for i in range(n_paths):
        for t in range(1, n_steps):
            # Multiplicative process with volatility
            if np.random.random() < 0.05:  # 5% chance of loss
                paths[i, t] = paths[i, t - 1] * 0.5
            else:
                paths[i, t] = paths[i, t - 1] * 1.06

    # Plot individual paths
    for i in range(n_paths):
        ax1.plot(paths[i], alpha=0.3, color="gray")

    # Plot ensemble average
    ensemble_avg = np.mean(paths, axis=0)
    ax1.plot(ensemble_avg, "b-", linewidth=3, label="Ensemble Average")

    # Plot typical path (geometric mean)
    typical_path = np.exp(np.mean(np.log(paths + 1e-10), axis=0))
    ax1.plot(typical_path, "r--", linewidth=3, label="Typical Path (Time Average)")

    ax1.set_xlabel("Time Steps", fontsize=12)
    ax1.set_ylabel("Wealth (Multiple of Initial)", fontsize=12)
    ax1.set_title("Ensemble Average vs Time Average", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Second plot: Distribution at final time
    final_wealth = paths[:, -1]
    ax2.hist(final_wealth, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax2.axvline(
        np.mean(final_wealth),
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Ensemble Avg: {np.mean(final_wealth):.2f}",
    )
    ax2.axvline(
        np.median(final_wealth),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(final_wealth):.2f}",
    )
    ax2.set_xlabel("Final Wealth", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Final Wealth Distribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(assets_dir / "ergodic_distinction.png", dpi=150, bbox_inches="tight")
    plt.close()


# Generate ensemble_vs_time.png
def create_ensemble_vs_time():
    """Create detailed comparison of ensemble vs time averages."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    np.random.seed(42)
    n_paths = 100
    n_years = 50

    # Simulate with and without insurance
    wealth_no_ins = np.ones((n_paths, n_years))
    wealth_with_ins = np.ones((n_paths, n_years))

    for i in range(n_paths):
        for t in range(1, n_years):
            # Without insurance - catastrophic risk
            if np.random.random() < 0.05:
                wealth_no_ins[i, t] = wealth_no_ins[i, t - 1] * 0.5
            else:
                wealth_no_ins[i, t] = wealth_no_ins[i, t - 1] * 1.08

            # With insurance - steady growth
            wealth_with_ins[i, t] = wealth_with_ins[i, t - 1] * 1.06

    # Plot 1: Individual trajectories
    ax = axes[0, 0]
    for i in range(min(20, n_paths)):
        ax.plot(wealth_no_ins[i], alpha=0.3, color="red")
        ax.plot(wealth_with_ins[i], alpha=0.3, color="blue")
    ax.set_xlabel("Years")
    ax.set_ylabel("Wealth (Multiple of Initial)")
    ax.set_title("Individual Wealth Trajectories")
    ax.set_yscale("log")
    ax.legend(["Without Insurance", "With Insurance"])
    ax.grid(True, alpha=0.3)

    # Plot 2: Ensemble vs Typical
    ax = axes[0, 1]
    ensemble_no = np.mean(wealth_no_ins, axis=0)
    ensemble_with = np.mean(wealth_with_ins, axis=0)
    typical_no = np.exp(np.mean(np.log(wealth_no_ins + 1e-10), axis=0))
    typical_with = np.exp(np.mean(np.log(wealth_with_ins + 1e-10), axis=0))

    ax.plot(ensemble_no, "r-", linewidth=2, label="Ensemble (No Ins)")
    ax.plot(ensemble_with, "b-", linewidth=2, label="Ensemble (With Ins)")
    ax.plot(typical_no, "r--", linewidth=2, label="Typical (No Ins)")
    ax.plot(typical_with, "b--", linewidth=2, label="Typical (With Ins)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Wealth (Multiple of Initial)")
    ax.set_title("Ensemble vs Typical Growth")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Growth rate distribution
    ax = axes[1, 0]
    growth_no = np.log(wealth_no_ins[:, -1] / wealth_no_ins[:, 0]) / (n_years - 1)
    growth_with = np.log(wealth_with_ins[:, -1] / wealth_with_ins[:, 0]) / (n_years - 1)

    ax.hist(growth_no, bins=30, alpha=0.5, color="red", label="Without Insurance")
    ax.hist(growth_with, bins=30, alpha=0.5, color="blue", label="With Insurance")
    ax.axvline(
        np.mean(growth_no),
        color="red",
        linestyle="--",
        label=f"Mean (No Ins): {np.mean(growth_no):.3f}",
    )
    ax.axvline(
        np.mean(growth_with),
        color="blue",
        linestyle="--",
        label=f"Mean (With Ins): {np.mean(growth_with):.3f}",
    )
    ax.set_xlabel("Realized Annual Growth Rate")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Growth Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Survival probability
    ax = axes[1, 1]
    survival_no = np.mean(wealth_no_ins > 0.1, axis=0)
    survival_with = np.mean(wealth_with_ins > 0.1, axis=0)

    ax.plot(survival_no * 100, "r-", linewidth=2, label="Without Insurance")
    ax.plot(survival_with * 100, "b-", linewidth=2, label="With Insurance")
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival Probability (%)")
    ax.set_title("Probability of Avoiding Ruin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.suptitle(
        "Ergodic Analysis: Insurance Impact on Wealth Dynamics",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(theory_figures_dir / "ensemble_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()


# Generate wealth_trajectories.png
def create_wealth_trajectories():
    """Create detailed wealth trajectory visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    np.random.seed(42)
    n_years = 30

    # Single trajectory with events marked
    ax = axes[0, 0]
    wealth = [10.0]  # Start with $10M
    events = []

    for year in range(1, n_years):
        if year in [5, 12, 18, 25]:  # Loss events
            wealth.append(wealth[-1] * 0.7)
            events.append((year, wealth[-1], "Loss"))
        else:
            wealth.append(wealth[-1] * 1.08)

    ax.plot(wealth, "b-", linewidth=2)
    for event in events:
        ax.plot(event[0], event[1], "ro", markersize=8)
        ax.annotate(
            f"Loss Event",
            xy=(event[0], event[1]),
            xytext=(event[0] + 1, event[1] * 1.1),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
            fontsize=9,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Assets ($M)")
    ax.set_title("Single Company Trajectory")
    ax.grid(True, alpha=0.3)

    # Comparison with and without insurance
    ax = axes[0, 1]
    wealth_no_ins = [10.0]
    wealth_with_ins = [10.0]
    premium = 0.1  # $100K per year

    for year in range(1, n_years):
        if year in [5, 12, 18, 25]:
            wealth_no_ins.append(wealth_no_ins[-1] * 0.7 * 1.08)
            wealth_with_ins.append((wealth_with_ins[-1] - premium) * 1.08)
        else:
            wealth_no_ins.append(wealth_no_ins[-1] * 1.08)
            wealth_with_ins.append((wealth_with_ins[-1] - premium) * 1.08)

    ax.plot(wealth_no_ins, "r-", linewidth=2, label="No Insurance")
    ax.plot(wealth_with_ins, "b-", linewidth=2, label="With Insurance")
    ax.set_xlabel("Year")
    ax.set_ylabel("Assets ($M)")
    ax.set_title("Insurance Impact Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Monte Carlo simulation
    ax = axes[0, 2]
    n_sims = 50
    for _ in range(n_sims):
        path = [10.0]
        for year in range(1, n_years):
            if np.random.random() < 0.1:  # 10% loss probability
                path.append(path[-1] * np.random.uniform(0.5, 0.9) * 1.08)
            else:
                path.append(path[-1] * 1.08)
        ax.plot(path, alpha=0.2, color="gray")

    # Add mean path
    ax.plot([10.0 * (1.05) ** t for t in range(n_years)], "b-", linewidth=3, label="Expected Path")
    ax.set_xlabel("Year")
    ax.set_ylabel("Assets ($M)")
    ax.set_title("Monte Carlo Simulation (50 paths)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss distribution
    ax = axes[1, 0]
    losses = np.random.lognormal(mean=np.log(100000), sigma=1.5, size=1000)
    ax.hist(losses / 1000, bins=50, alpha=0.7, color="red", edgecolor="black")
    ax.set_xlabel("Loss Amount ($K)")
    ax.set_ylabel("Frequency")
    ax.set_title("Loss Distribution (Lognormal)")
    ax.set_xlim([0, 2000])
    ax.grid(True, alpha=0.3)

    # Premium vs Coverage
    ax = axes[1, 1]
    coverages = np.linspace(0, 10, 50)  # $0-10M coverage
    premiums = 50 + coverages * 15  # Base + rate
    ax.plot(coverages, premiums, "b-", linewidth=2)
    ax.fill_between(coverages, 0, premiums, alpha=0.3)
    ax.set_xlabel("Coverage Limit ($M)")
    ax.set_ylabel("Annual Premium ($K)")
    ax.set_title("Premium vs Coverage Relationship")
    ax.grid(True, alpha=0.3)

    # ROI Analysis
    ax = axes[1, 2]
    retentions = np.linspace(0, 5, 50)  # $0-5M retention
    roi_no_ins = 8 - retentions * 0  # Constant without insurance
    roi_with_ins = 6 + retentions * 0.8  # Increases with retention

    ax.plot(retentions, roi_no_ins, "r-", linewidth=2, label="No Insurance")
    ax.plot(retentions, roi_with_ins, "b-", linewidth=2, label="With Insurance")
    ax.set_xlabel("Retention ($M)")
    ax.set_ylabel("Expected ROI (%)")
    ax.set_title("ROI vs Retention Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Wealth Dynamics and Insurance Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(theory_figures_dir / "wealth_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()


# Generate growth_rate_distribution.png
def create_growth_rate_distribution():
    """Create growth rate distribution visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    np.random.seed(42)
    n_sims = 1000
    n_years = 20

    # Simulate growth rates
    growth_rates_no_ins = []
    growth_rates_with_ins = []

    for _ in range(n_sims):
        # Without insurance
        wealth = 1.0
        for _ in range(n_years):
            if np.random.random() < 0.05:
                wealth *= 0.5
            else:
                wealth *= 1.08
        growth_rates_no_ins.append(np.log(wealth) / n_years)

        # With insurance
        wealth = 1.0
        for _ in range(n_years):
            wealth *= 1.06
        growth_rates_with_ins.append(np.log(wealth) / n_years)

    # Plot 1: Histogram comparison
    ax = axes[0, 0]
    ax.hist(
        growth_rates_no_ins,
        bins=50,
        alpha=0.5,
        color="red",
        label="Without Insurance",
        density=True,
    )
    ax.hist(
        growth_rates_with_ins,
        bins=50,
        alpha=0.5,
        color="blue",
        label="With Insurance",
        density=True,
    )
    ax.axvline(np.mean(growth_rates_no_ins), color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(growth_rates_with_ins), color="blue", linestyle="--", linewidth=2)
    ax.set_xlabel("Annual Growth Rate")
    ax.set_ylabel("Probability Density")
    ax.set_title("Growth Rate Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative distribution
    ax = axes[0, 1]
    sorted_no_ins = np.sort(growth_rates_no_ins)
    sorted_with_ins = np.sort(growth_rates_with_ins)
    cdf = np.arange(1, n_sims + 1) / n_sims

    ax.plot(sorted_no_ins, cdf, "r-", linewidth=2, label="Without Insurance")
    ax.plot(sorted_with_ins, cdf, "b-", linewidth=2, label="With Insurance")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Annual Growth Rate")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot comparison
    ax = axes[1, 0]
    bp = ax.boxplot(
        [growth_rates_no_ins, growth_rates_with_ins],
        labels=["Without Insurance", "With Insurance"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("red")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("blue")
    bp["boxes"][1].set_alpha(0.5)
    ax.set_ylabel("Annual Growth Rate")
    ax.set_title("Growth Rate Comparison")
    ax.grid(True, alpha=0.3)

    # Plot 4: Risk-Return scatter
    ax = axes[1, 1]
    # Calculate for different insurance levels
    insurance_levels = np.linspace(0, 1, 20)
    means = []
    stds = []

    for level in insurance_levels:
        rates = []
        for _ in range(100):
            wealth = 1.0
            for _ in range(n_years):
                if np.random.random() < 0.05 * (1 - level):
                    wealth *= 0.5 + 0.5 * level
                else:
                    wealth *= 1.08 - 0.02 * level
            rates.append(np.log(wealth) / n_years)
        means.append(np.mean(rates))
        stds.append(np.std(rates))

    ax.scatter(stds, means, c=insurance_levels, cmap="coolwarm", s=50)
    ax.set_xlabel("Risk (Std Dev of Growth Rate)")
    ax.set_ylabel("Return (Mean Growth Rate)")
    ax.set_title("Risk-Return Trade-off")
    cbar = plt.colorbar(ax.scatter(stds, means, c=insurance_levels, cmap="coolwarm", s=50), ax=ax)
    cbar.set_label("Insurance Level")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Growth Rate Analysis: Impact of Insurance", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(theory_figures_dir / "growth_rate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# Generate all images
print("Generating documentation images...")
create_ergodic_distinction()
print("[OK] Created ergodic_distinction.png")
create_ensemble_vs_time()
print("[OK] Created ensemble_vs_time.png")
create_wealth_trajectories()
print("[OK] Created wealth_trajectories.png")
create_growth_rate_distribution()
print("[OK] Created growth_rate_distribution.png")
print("\nAll images generated successfully!")

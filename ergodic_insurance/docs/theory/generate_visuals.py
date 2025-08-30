"""
Generate visual aids for theoretical documentation.

This script creates plots and diagrams that illustrate key concepts
in the theoretical foundations documentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150

# Output directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


def plot_ensemble_vs_time_average():
    """Create visualization of ensemble vs time average divergence."""

    np.random.seed(42)
    n_paths = 100
    n_steps = 100

    # Simulate multiplicative process
    returns = np.random.choice([1.5, 0.6], size=(n_paths, n_steps), p=[0.5, 0.5])
    wealth = np.zeros((n_paths, n_steps + 1))
    wealth[:, 0] = 100

    for t in range(n_steps):
        wealth[:, t + 1] = wealth[:, t] * returns[:, t]

    # Calculate averages
    ensemble_avg = np.mean(wealth, axis=0)

    # Time average (median represents typical path)
    median_path = np.median(wealth, axis=0)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot trajectories
    for i in range(min(20, n_paths)):
        ax1.plot(wealth[i, :], alpha=0.3, color="gray", linewidth=0.5)

    ax1.plot(ensemble_avg, "b-", linewidth=3, label="Ensemble Average")
    ax1.plot(median_path, "r-", linewidth=3, label="Median (Typical) Path")
    ax1.set_yscale("log")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Wealth")
    ax1.set_title("Ensemble vs Time Average")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot distribution at final time
    final_wealth = wealth[:, -1]
    ax2.hist(final_wealth[final_wealth > 0], bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(
        np.mean(final_wealth),
        color="b",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(final_wealth):.0f}",
    )
    ax2.axvline(
        np.median(final_wealth),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(final_wealth):.0f}",
    )
    ax2.set_xlabel("Final Wealth")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Final Wealth Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_vs_time.png", bbox_inches="tight")
    plt.close()

    print("Created: ensemble_vs_time.png")


def plot_volatility_drag():
    """Illustrate the concept of volatility drag."""

    volatilities = np.linspace(0, 0.5, 100)
    growth_rates = []

    mu = 0.10  # 10% arithmetic mean

    for sigma in volatilities:
        # Geometric growth rate
        g = mu - sigma**2 / 2
        growth_rates.append(g)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(volatilities * 100, np.array(growth_rates) * 100, "b-", linewidth=2)
    ax.axhline(
        y=mu * 100, color="r", linestyle="--", alpha=0.5, label=f"Arithmetic Mean: {mu*100:.0f}%"
    )
    ax.fill_between(
        volatilities * 100,
        np.array(growth_rates) * 100,
        mu * 100,
        alpha=0.3,
        color="red",
        label="Volatility Drag",
    )

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Growth Rate (%)")
    ax.set_title("Volatility Drag: Geometric vs Arithmetic Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(
        "g = μ - σ²/2",
        xy=(30, 2),
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
    )

    plt.savefig(output_dir / "volatility_drag.png", bbox_inches="tight")
    plt.close()

    print("Created: volatility_drag.png")


def plot_kelly_criterion():
    """Visualize Kelly criterion for different scenarios."""

    # Binary bet scenario
    probabilities = np.linspace(0.5, 1.0, 100)
    odds = [1.5, 2.0, 3.0]  # Different payoff ratios

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Kelly fraction vs probability
    for b in odds:
        kelly_fractions = (probabilities * b - (1 - probabilities)) / b
        kelly_fractions = np.maximum(0, kelly_fractions)  # No negative bets
        ax1.plot(probabilities * 100, kelly_fractions * 100, label=f"Odds = {b}:1", linewidth=2)

    ax1.set_xlabel("Win Probability (%)")
    ax1.set_ylabel("Kelly Fraction (%)")
    ax1.set_title("Kelly Criterion: Optimal Bet Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(50, 100)
    ax1.set_ylim(0, 100)

    # Growth rate vs fraction bet
    p = 0.6  # 60% win probability
    b = 2.0  # 2:1 odds
    fractions = np.linspace(0, 1, 100)

    growth_rates = []
    for f in fractions:
        if f == 0:
            g = 0
        elif f == 1:
            g = p * np.log(1 + b) + (1 - p) * (-np.inf)  # Total loss possible
        else:
            g = p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)
        growth_rates.append(g if g != -np.inf else -10)

    kelly_optimal = (p * b - (1 - p)) / b

    ax2.plot(fractions * 100, growth_rates, "b-", linewidth=2)
    ax2.axvline(
        x=kelly_optimal * 100,
        color="r",
        linestyle="--",
        label=f"Kelly Optimal: {kelly_optimal*100:.1f}%",
    )
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Fraction Bet (%)")
    ax2.set_ylabel("Expected Log Growth Rate")
    ax2.set_title("Growth Rate vs Bet Size (p=0.6, b=2)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "kelly_criterion.png", bbox_inches="tight")
    plt.close()

    print("Created: kelly_criterion.png")


def plot_insurance_impact():
    """Show impact of insurance on wealth trajectories."""

    np.random.seed(42)
    n_years = 50
    n_sims = 100

    # Parameters
    initial_wealth = 10_000_000
    base_growth = 0.08
    volatility = 0.15
    loss_prob = 0.05
    loss_severity = 0.3  # 30% loss when occurs

    # Simulate with and without insurance
    wealth_no_insurance = np.zeros((n_sims, n_years + 1))
    wealth_with_insurance = np.zeros((n_sims, n_years + 1))

    wealth_no_insurance[:, 0] = initial_wealth
    wealth_with_insurance[:, 0] = initial_wealth

    for sim in range(n_sims):
        for year in range(n_years):
            # Growth factor
            growth = np.exp(np.random.normal(base_growth, volatility))

            # Loss event
            has_loss = np.random.rand() < loss_prob

            # Without insurance
            if has_loss:
                wealth_no_insurance[sim, year + 1] = (
                    wealth_no_insurance[sim, year] * growth * (1 - loss_severity)
                )
            else:
                wealth_no_insurance[sim, year + 1] = wealth_no_insurance[sim, year] * growth

            # With insurance (premium 2%, deductible 5%)
            premium = 0.02
            deductible = 0.05

            if has_loss:
                retained_loss = min(loss_severity, deductible)
                wealth_with_insurance[sim, year + 1] = (
                    wealth_with_insurance[sim, year] * growth * (1 - retained_loss - premium)
                )
            else:
                wealth_with_insurance[sim, year + 1] = (
                    wealth_with_insurance[sim, year] * growth * (1 - premium)
                )

            # Check for bankruptcy
            if wealth_no_insurance[sim, year + 1] <= 0:
                wealth_no_insurance[sim, year + 1 :] = 0
                break
            if wealth_with_insurance[sim, year + 1] <= 0:
                wealth_with_insurance[sim, year + 1 :] = 0
                break

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectories
    for i in range(min(20, n_sims)):
        axes[0, 0].plot(wealth_no_insurance[i, :], alpha=0.3, color="red", linewidth=0.5)
        axes[0, 1].plot(wealth_with_insurance[i, :], alpha=0.3, color="green", linewidth=0.5)

    # Add median paths
    axes[0, 0].plot(np.median(wealth_no_insurance, axis=0), "r-", linewidth=2, label="Median")
    axes[0, 1].plot(np.median(wealth_with_insurance, axis=0), "g-", linewidth=2, label="Median")

    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 0].set_title("Without Insurance")
    axes[0, 1].set_title("With Insurance")
    axes[0, 0].set_xlabel("Years")
    axes[0, 1].set_xlabel("Years")
    axes[0, 0].set_ylabel("Wealth")
    axes[0, 1].set_ylabel("Wealth")
    axes[0, 0].legend()
    axes[0, 1].legend()

    # Final wealth distribution
    final_no_insurance = wealth_no_insurance[:, -1]
    final_with_insurance = wealth_with_insurance[:, -1]

    bins = np.logspace(0, 9, 30)
    axes[1, 0].hist(
        final_no_insurance[final_no_insurance > 0],
        bins=bins,
        alpha=0.5,
        color="red",
        label="No Insurance",
        edgecolor="black",
    )
    axes[1, 0].hist(
        final_with_insurance[final_with_insurance > 0],
        bins=bins,
        alpha=0.5,
        color="green",
        label="With Insurance",
        edgecolor="black",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Final Wealth")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Final Wealth Distribution")
    axes[1, 0].legend()

    # Statistics comparison
    stats_text = f"""
    Without Insurance:
      Survival Rate: {np.mean(final_no_insurance > 0):.1%}
      Median Final: ${np.median(final_no_insurance[final_no_insurance > 0])/1e6:.1f}M
      Growth Rate: {np.mean(np.log(final_no_insurance[final_no_insurance > 0]/initial_wealth)/n_years):.2%}

    With Insurance:
      Survival Rate: {np.mean(final_with_insurance > 0):.1%}
      Median Final: ${np.median(final_with_insurance[final_with_insurance > 0])/1e6:.1f}M
      Growth Rate: {np.mean(np.log(final_with_insurance[final_with_insurance > 0]/initial_wealth)/n_years):.2%}
    """

    axes[1, 1].text(
        0.1, 0.5, stats_text, fontsize=12, verticalalignment="center", family="monospace"
    )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Comparison Statistics")

    plt.suptitle("Insurance Impact on Long-Term Wealth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "insurance_impact.png", bbox_inches="tight")
    plt.close()

    print("Created: insurance_impact.png")


def plot_pareto_frontier():
    """Create Pareto frontier visualization."""

    np.random.seed(42)

    # Generate sample Pareto frontier
    n_points = 50
    weights = np.linspace(0, 1, n_points)

    # Two objectives: cost and risk
    costs = []
    risks = []

    for w in weights:
        # Simulate optimization with different weightings
        cost = 0.5 + 0.5 * (1 - w) + np.random.normal(0, 0.02)
        risk = 0.3 + 0.7 * w + np.random.normal(0, 0.02)
        costs.append(cost)
        risks.append(risk)

    # Sort by cost to get frontier
    sorted_indices = np.argsort(costs)
    costs_array = np.array(costs)[sorted_indices]
    risks_array = np.array(risks)[sorted_indices]

    # Generate dominated points
    n_dominated = 100
    dominated_costs = np.random.uniform(0.6, 1.2, n_dominated)
    dominated_risks = np.random.uniform(0.4, 1.2, n_dominated)

    # Filter to keep only dominated
    dominated_points = []
    for c, r in zip(dominated_costs, dominated_risks):
        is_dominated = False
        for cf, rf in zip(costs_array, risks_array):
            if c >= cf and r >= rf and (c > cf or r > rf):
                is_dominated = True
                break
        if is_dominated:
            dominated_points.append((c, r))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot dominated points
    if dominated_points:
        dom_c, dom_r = zip(*dominated_points)
        ax.scatter(dom_c, dom_r, alpha=0.3, color="gray", s=20, label="Dominated")

    # Plot Pareto frontier
    ax.plot(costs_array, risks_array, "b-", linewidth=2, label="Pareto Frontier")
    ax.scatter(costs_array, risks_array, color="blue", s=50, zorder=5)

    # Highlight specific points
    idx_low_cost = 5
    idx_balanced = 25
    idx_low_risk = 45

    ax.scatter(
        costs_array[idx_low_cost],
        risks_array[idx_low_cost],
        color="green",
        s=200,
        marker="*",
        label="Low Cost",
        zorder=10,
    )
    ax.scatter(
        costs_array[idx_balanced],
        risks_array[idx_balanced],
        color="orange",
        s=200,
        marker="*",
        label="Balanced",
        zorder=10,
    )
    ax.scatter(
        costs_array[idx_low_risk],
        risks_array[idx_low_risk],
        color="red",
        s=200,
        marker="*",
        label="Low Risk",
        zorder=10,
    )

    ax.set_xlabel("Cost (Premium as % of Assets)")
    ax.set_ylabel("Risk (Probability of Ruin)")
    ax.set_title("Pareto Frontier: Cost vs Risk Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(
        "Better",
        xy=(0.5, 0.3),
        xytext=(0.4, 0.2),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        fontsize=12,
        color="green",
        fontweight="bold",
    )
    ax.annotate(
        "Worse",
        xy=(1.0, 1.0),
        xytext=(1.1, 1.1),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=12,
        color="red",
        fontweight="bold",
    )

    plt.savefig(output_dir / "pareto_frontier.png", bbox_inches="tight")
    plt.close()

    print("Created: pareto_frontier.png")


def main():
    """Generate all visual aids."""

    print("Generating visual aids for theoretical documentation...")
    print("-" * 50)

    plot_ensemble_vs_time_average()
    plot_volatility_drag()
    plot_kelly_criterion()
    plot_insurance_impact()
    plot_pareto_frontier()

    print("-" * 50)
    print(f"All visuals saved to: {output_dir.absolute()}")
    print("\nTo use in documentation, reference as:")
    print("  .. image:: theory/figures/filename.png")
    print("     :width: 600px")
    print("     :align: center")


if __name__ == "__main__":
    main()

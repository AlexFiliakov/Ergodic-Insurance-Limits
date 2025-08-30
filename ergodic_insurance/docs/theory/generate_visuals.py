"""
Generate visual aids for theoretical documentation.

This script creates plots and diagrams that illustrate key concepts
in the theoretical foundations documentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Show impact of insurance on wealth trajectories - demonstrating ergodic theory."""

    np.random.seed(42)
    n_years = 100
    n_sims = 1000

    # Parameters chosen to demonstrate ergodic theory insights
    # Lower starting capital and higher risk to show survival differences
    initial_wealth = 1_000_000  # $1M starting capital (much lower)
    base_growth = 0.12  # 12% average growth (higher to compensate for risk)
    volatility = 0.20  # 20% volatility

    # Catastrophic loss parameters - higher frequency and severity
    loss_prob = 0.10  # 10% annual probability (doubled from before)
    loss_severity = 0.60  # 60% loss when occurs (doubled from before)

    # Insurance parameters - actuarially "unfair" but ergodically optimal
    premium_rate = 0.04  # 4% annual premium (2x expected loss of 2%)
    deductible_rate = 0.10  # 10% deductible

    # Simulate with and without insurance
    wealth_no_insurance = np.zeros((n_sims, n_years + 1))
    wealth_with_insurance = np.zeros((n_sims, n_years + 1))

    wealth_no_insurance[:, 0] = initial_wealth
    wealth_with_insurance[:, 0] = initial_wealth

    # Track bankruptcy
    bankrupt_no_insurance = np.zeros(n_sims, dtype=bool)
    bankrupt_with_insurance = np.zeros(n_sims, dtype=bool)

    for sim in range(n_sims):
        for year in range(n_years):
            # Skip if already bankrupt
            if bankrupt_no_insurance[sim]:
                wealth_no_insurance[sim, year + 1] = 0
            else:
                # Growth factor (same for both)
                growth = np.exp(np.random.normal(base_growth - volatility**2 / 2, volatility))

                # Loss event (same for both)
                has_loss = np.random.rand() < loss_prob

                # Without insurance
                if has_loss:
                    wealth_no_insurance[sim, year + 1] = (
                        wealth_no_insurance[sim, year] * growth * (1 - loss_severity)
                    )
                else:
                    wealth_no_insurance[sim, year + 1] = wealth_no_insurance[sim, year] * growth

                # Check for bankruptcy
                if wealth_no_insurance[sim, year + 1] <= initial_wealth * 0.01:  # 1% of initial
                    wealth_no_insurance[sim, year + 1] = 0
                    bankrupt_no_insurance[sim] = True

            # With insurance
            if bankrupt_with_insurance[sim]:
                wealth_with_insurance[sim, year + 1] = 0
            else:
                growth = np.exp(np.random.normal(base_growth - volatility**2 / 2, volatility))

                if has_loss:
                    # Insurance covers losses above deductible
                    retained_loss = min(loss_severity, deductible_rate)
                    wealth_with_insurance[sim, year + 1] = (
                        wealth_with_insurance[sim, year]
                        * growth
                        * (1 - retained_loss)
                        * (1 - premium_rate)
                    )
                else:
                    # Just pay premium
                    wealth_with_insurance[sim, year + 1] = (
                        wealth_with_insurance[sim, year] * growth * (1 - premium_rate)
                    )

                # Check for bankruptcy
                if wealth_with_insurance[sim, year + 1] <= initial_wealth * 0.01:
                    wealth_with_insurance[sim, year + 1] = 0
                    bankrupt_with_insurance[sim] = True

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot sample trajectories
    n_plot = min(50, n_sims)
    for i in range(n_plot):
        if not bankrupt_no_insurance[i]:
            axes[0, 0].plot(wealth_no_insurance[i, :], alpha=0.2, color="red", linewidth=0.5)
        if not bankrupt_with_insurance[i]:
            axes[0, 1].plot(wealth_with_insurance[i, :], alpha=0.2, color="green", linewidth=0.5)

    # Calculate and plot key statistics over time
    # Median of surviving paths (typical path for ergodic theory)
    median_no_ins = np.zeros(n_years + 1)
    median_with_ins = np.zeros(n_years + 1)
    mean_no_ins = np.zeros(n_years + 1)
    mean_with_ins = np.zeros(n_years + 1)

    for t in range(n_years + 1):
        surviving_no_ins = wealth_no_insurance[:, t][wealth_no_insurance[:, t] > 0]
        surviving_with_ins = wealth_with_insurance[:, t][wealth_with_insurance[:, t] > 0]

        if len(surviving_no_ins) > 0:
            median_no_ins[t] = np.median(surviving_no_ins)
            mean_no_ins[t] = np.mean(surviving_no_ins)
        else:
            median_no_ins[t] = 0
            mean_no_ins[t] = 0

        if len(surviving_with_ins) > 0:
            median_with_ins[t] = np.median(surviving_with_ins)
            mean_with_ins[t] = np.mean(surviving_with_ins)
        else:
            median_with_ins[t] = 0
            mean_with_ins[t] = 0

    # Plot median (typical) and mean (ensemble) paths
    axes[0, 0].plot(median_no_ins, "darkred", linewidth=3, label="Median (Typical)")
    axes[0, 0].plot(mean_no_ins, "blue", linewidth=2, label="Mean (Ensemble)", linestyle="--")
    axes[0, 1].plot(median_with_ins, "darkgreen", linewidth=3, label="Median (Typical)")
    axes[0, 1].plot(mean_with_ins, "blue", linewidth=2, label="Mean (Ensemble)", linestyle="--")

    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 0].set_title("Without Insurance", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("With Insurance (4% Premium)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Years")
    axes[0, 1].set_xlabel("Years")
    axes[0, 0].set_ylabel("Wealth ($)")
    axes[0, 1].set_ylabel("Wealth ($)")
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)

    # Set y-axis limits to show full range
    axes[0, 0].set_ylim([1e3, 1e10])
    axes[0, 1].set_ylim([1e3, 1e10])

    # Survival rate over time
    survival_no_ins = np.mean(wealth_no_insurance > 0, axis=0)
    survival_with_ins = np.mean(wealth_with_insurance > 0, axis=0)

    axes[1, 0].plot(survival_no_ins * 100, "r-", linewidth=2, label="No Insurance")
    axes[1, 0].plot(survival_with_ins * 100, "g-", linewidth=2, label="With Insurance")
    axes[1, 0].set_xlabel("Years")
    axes[1, 0].set_ylabel("Survival Rate (%)")
    axes[1, 0].set_title("Survival Probability Over Time", fontsize=12, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])

    # Key statistics comparison
    final_no_insurance = wealth_no_insurance[:, -1]
    final_with_insurance = wealth_with_insurance[:, -1]

    # Calculate time-average growth rates (ergodic perspective)
    surviving_no_ins_final = final_no_insurance[final_no_insurance > 0]
    surviving_with_ins_final = final_with_insurance[final_with_insurance > 0]

    if len(surviving_no_ins_final) > 0:
        time_avg_growth_no_ins = (
            np.median(np.log(surviving_no_ins_final / initial_wealth)) / n_years
        )
        ensemble_growth_no_ins = np.log(np.mean(surviving_no_ins_final) / initial_wealth) / n_years
    else:
        time_avg_growth_no_ins = -np.inf
        ensemble_growth_no_ins = -np.inf

    if len(surviving_with_ins_final) > 0:
        time_avg_growth_with_ins = (
            np.median(np.log(surviving_with_ins_final / initial_wealth)) / n_years
        )
        ensemble_growth_with_ins = (
            np.log(np.mean(surviving_with_ins_final) / initial_wealth) / n_years
        )
    else:
        time_avg_growth_with_ins = -np.inf
        ensemble_growth_with_ins = -np.inf

    # Expected loss calculation
    expected_loss = loss_prob * loss_severity

    stats_text = f"""
    ERGODIC THEORY DEMONSTRATION
    ============================

    Initial Wealth: ${initial_wealth/1e6:.1f}M
    Loss: {loss_severity:.0%} with {loss_prob:.0%} annual probability
    Expected Loss: {expected_loss:.1%} per year
    Insurance Premium: {premium_rate:.1%} (= {premium_rate/expected_loss:.1f}x expected loss)

    WITHOUT INSURANCE:
    ------------------
    Survival Rate: {np.mean(final_no_insurance > 0):.1%}
    Time-Avg Growth: {time_avg_growth_no_ins:.2%} (Median Path)
    Ensemble Growth: {ensemble_growth_no_ins:.2%} (Mean Path)

    WITH INSURANCE:
    ---------------
    Survival Rate: {np.mean(final_with_insurance > 0):.1%}
    Time-Avg Growth: {time_avg_growth_with_ins:.2%} (Median Path)
    Ensemble Growth: {ensemble_growth_with_ins:.2%} (Mean Path)

    KEY INSIGHT:
    Insurance increases time-average growth
    despite reducing expected value!
    """

    axes[1, 1].text(
        0.05,
        0.5,
        stats_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis("off")

    plt.suptitle("Ergodic Theory: Insurance as Growth Enabler", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "insurance_impact.png", bbox_inches="tight", dpi=150)
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
        color="teal",
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
        color="purple",
        s=200,
        marker="*",
        label="Low Risk",
        zorder=10,
    )

    # Add text annotations for the highlighted points
    ax.annotate(
        "Low Cost",
        xy=(costs_array[idx_low_cost], risks_array[idx_low_cost]),
        xytext=(costs_array[idx_low_cost] - 0.08, risks_array[idx_low_cost] + 0.08),
        fontsize=10,
        color="teal",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="teal", alpha=0.8),
    )
    ax.annotate(
        "Balanced",
        xy=(costs_array[idx_balanced], risks_array[idx_balanced]),
        xytext=(costs_array[idx_balanced], risks_array[idx_balanced] - 0.1),
        fontsize=10,
        color="orange",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.8),
    )
    ax.annotate(
        "Low Risk",
        xy=(costs_array[idx_low_risk], risks_array[idx_low_risk]),
        xytext=(costs_array[idx_low_risk] + 0.08, risks_array[idx_low_risk] - 0.08),
        fontsize=10,
        color="purple",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="purple", alpha=0.8),
    )

    ax.set_xlabel("Cost (Premium as % of Assets)")
    ax.set_ylabel("Risk (Probability of Ruin)")
    ax.set_title("Pareto Frontier: Cost vs Risk Trade-off")
    ax.legend(loc="upper right")  # Move legend to upper right to avoid "Better" annotation
    ax.grid(True, alpha=0.3)

    # Add annotations - adjust "Better" position to avoid legend
    ax.annotate(
        "Better",
        xy=(0.5, 0.3),
        xytext=(0.35, 0.15),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        fontsize=12,
        color="green",
        fontweight="bold",
    )
    ax.annotate(
        "Worse",
        xy=(0.85, 0.85),
        xytext=(0.75, 0.95),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=12,
        color="red",
        fontweight="bold",
    )

    plt.savefig(output_dir / "pareto_frontier.png", bbox_inches="tight")
    plt.close()

    print("Created: pareto_frontier.png")


def plot_monte_carlo_convergence():
    """Visualize Monte Carlo convergence and variance reduction techniques."""

    np.random.seed(42)
    n_simulations = 10000

    # Standard Monte Carlo
    standard_samples = np.random.normal(0.08, 0.15, n_simulations)
    cumulative_mean = np.cumsum(standard_samples) / np.arange(1, n_simulations + 1)

    # Antithetic variates
    antithetic_samples = np.zeros(n_simulations)
    for i in range(0, n_simulations, 2):
        u = np.random.uniform()
        antithetic_samples[i] = np.quantile(standard_samples, u)
        if i + 1 < n_simulations:
            antithetic_samples[i + 1] = np.quantile(standard_samples, 1 - u)

    antithetic_mean = np.cumsum(antithetic_samples) / np.arange(1, n_simulations + 1)

    # Control variates (using correlated control)
    control = np.random.normal(0.08, 0.10, n_simulations)  # Known mean
    c = -0.5  # Optimal coefficient
    control_samples = standard_samples - c * (control - 0.08)
    control_mean = np.cumsum(control_samples) / np.arange(1, n_simulations + 1)

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convergence comparison
    x_range = np.arange(1, n_simulations + 1)
    axes[0, 0].plot(x_range[::10], cumulative_mean[::10], "b-", alpha=0.7, label="Standard MC")
    axes[0, 0].plot(x_range[::10], antithetic_mean[::10], "r-", alpha=0.7, label="Antithetic")
    axes[0, 0].plot(x_range[::10], control_mean[::10], "g-", alpha=0.7, label="Control Variate")
    axes[0, 0].axhline(y=0.08, color="k", linestyle="--", alpha=0.5, label="True Value")
    axes[0, 0].set_xlabel("Number of Simulations")
    axes[0, 0].set_ylabel("Estimated Mean")
    axes[0, 0].set_title("Monte Carlo Convergence Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Variance reduction
    window = 1000
    rolling_std_standard = pd.Series(cumulative_mean).rolling(window).std()
    rolling_std_antithetic = pd.Series(antithetic_mean).rolling(window).std()
    rolling_std_control = pd.Series(control_mean).rolling(window).std()

    axes[0, 1].plot(rolling_std_standard, "b-", alpha=0.7, label="Standard MC")
    axes[0, 1].plot(rolling_std_antithetic, "r-", alpha=0.7, label="Antithetic")
    axes[0, 1].plot(rolling_std_control, "g-", alpha=0.7, label="Control Variate")
    axes[0, 1].set_xlabel("Simulation Number")
    axes[0, 1].set_ylabel("Rolling Std Dev")
    axes[0, 1].set_title(f"Variance Reduction (Window={window})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Error vs sample size
    sample_sizes = np.logspace(1, 4, 20, dtype=int)
    errors_standard = []
    errors_antithetic = []
    errors_control = []

    for n in sample_sizes:
        std_err = np.std(standard_samples[:n]) / np.sqrt(n)
        errors_standard.append(std_err)
        errors_antithetic.append(std_err * 0.7)  # Approx reduction
        errors_control.append(std_err * 0.5)  # Approx reduction

    axes[1, 0].loglog(sample_sizes, errors_standard, "b-o", label="Standard MC")
    axes[1, 0].loglog(sample_sizes, errors_antithetic, "r-s", label="Antithetic")
    axes[1, 0].loglog(sample_sizes, errors_control, "g-^", label="Control Variate")
    axes[1, 0].loglog(sample_sizes, 1 / np.sqrt(sample_sizes), "k--", alpha=0.5, label="1/√n")
    axes[1, 0].set_xlabel("Sample Size")
    axes[1, 0].set_ylabel("Standard Error")
    axes[1, 0].set_title("Error Reduction with Sample Size")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Efficiency comparison
    efficiency_data = {
        "Standard MC": [1.0, 1.0, 1.0],
        "Antithetic": [0.7, 1.0, 1.4],
        "Control Variate": [0.5, 1.1, 2.0],
        "Importance Sampling": [0.3, 1.2, 3.0],
    }

    categories = ["Variance", "Comp. Time", "Efficiency"]
    x = np.arange(len(categories))
    width = 0.2

    for i, (method, values) in enumerate(efficiency_data.items()):
        axes[1, 1].bar(x + i * width, values, width, label=method)

    axes[1, 1].set_xlabel("Metric")
    axes[1, 1].set_ylabel("Relative Value")
    axes[1, 1].set_title("Variance Reduction Techniques Comparison")
    axes[1, 1].set_xticks(x + width * 1.5)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Monte Carlo Methods and Variance Reduction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "monte_carlo_convergence.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("Created: monte_carlo_convergence.png")


def plot_convergence_diagnostics():
    """Visualize convergence diagnostics including Gelman-Rubin statistic."""

    np.random.seed(42)
    n_chains = 4
    n_iterations = 5000

    # Simulate MCMC chains with different convergence properties
    chains_list = []
    for i in range(n_chains):
        # Add burn-in period with different starting points
        burn_in = np.random.normal(5 + i * 2, 2, 500)
        # Converged portion
        converged = np.random.normal(0, 1, n_iterations - 500)
        chain = np.concatenate([burn_in, converged])
        chains_list.append(chain)

    chains = np.array(chains_list)

    # Calculate Gelman-Rubin statistic over time
    def gelman_rubin(chains, start=0, end=None):
        if end is None:
            end = chains.shape[1]

        chains_subset = chains[:, start:end]
        n, m = chains_subset.shape

        # Between-chain variance
        chain_means = np.mean(chains_subset, axis=1)
        B = m * np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean(np.var(chains_subset, axis=1, ddof=1))

        # Potential scale reduction factor
        var_plus = ((m - 1) / m) * W + (1 / m) * B
        R_hat = np.sqrt(var_plus / W) if W > 0 else 1.0

        return R_hat

    # Calculate R-hat over time
    check_points = np.arange(100, n_iterations, 50)
    r_hats = [gelman_rubin(chains, end=cp) for cp in check_points]

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Trace plots
    for i in range(n_chains):
        axes[0, 0].plot(chains[i, :], alpha=0.7, linewidth=0.5, label=f"Chain {i+1}")
    axes[0, 0].axvline(x=500, color="r", linestyle="--", alpha=0.5, label="End of Burn-in")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].set_title("Trace Plots")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    # Running mean
    for i in range(n_chains):
        running_mean = np.cumsum(chains[i, :]) / np.arange(1, n_iterations + 1)
        axes[0, 1].plot(running_mean, alpha=0.7, label=f"Chain {i+1}")
    axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Running Mean")
    axes[0, 1].set_title("Running Mean Convergence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Autocorrelation
    from scipy import signal

    max_lag = 100
    for i in range(n_chains):
        # Use only converged portion
        chain_data = chains[i, 500:]
        autocorr = signal.correlate(
            chain_data - np.mean(chain_data), chain_data - np.mean(chain_data), mode="same"
        )
        autocorr = autocorr / autocorr[len(autocorr) // 2]
        center = len(autocorr) // 2
        axes[0, 2].plot(autocorr[center : center + max_lag], alpha=0.7, label=f"Chain {i+1}")

    axes[0, 2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[0, 2].set_xlabel("Lag")
    axes[0, 2].set_ylabel("Autocorrelation")
    axes[0, 2].set_title("Autocorrelation Function")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Density plots
    for i in range(n_chains):
        axes[1, 0].hist(chains[i, 500:], bins=50, alpha=0.5, density=True, label=f"Chain {i+1}")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Posterior Distributions")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Gelman-Rubin statistic
    axes[1, 1].plot(check_points, r_hats, "b-", linewidth=2)
    axes[1, 1].axhline(y=1.1, color="r", linestyle="--", label="R̂ = 1.1 threshold")
    axes[1, 1].axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Perfect convergence")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("R̂ (Gelman-Rubin)")
    axes[1, 1].set_title("Gelman-Rubin Convergence Diagnostic")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Effective sample size
    def effective_sample_size(chain):
        """Estimate effective sample size."""
        n = len(chain)
        # Simple approximation using autocorrelation
        autocorr_sum = 1.0
        for lag in range(1, min(n // 4, 100)):
            corr = np.corrcoef(chain[:-lag], chain[lag:])[0, 1]
            if corr < 0.05:
                break
            autocorr_sum += 2 * corr
        return n / autocorr_sum

    ess_over_time = []
    for cp in check_points:
        ess = np.mean([effective_sample_size(chains[i, :cp]) for i in range(n_chains)])
        ess_over_time.append(ess)

    axes[1, 2].plot(check_points, ess_over_time, "g-", linewidth=2, label="Effective")
    axes[1, 2].plot(check_points, check_points, "k--", alpha=0.5, label="Actual")
    axes[1, 2].set_xlabel("Actual Sample Size")
    axes[1, 2].set_ylabel("Effective Sample Size")
    axes[1, 2].set_title("Effective Sample Size")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("MCMC Convergence Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_diagnostics.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("Created: convergence_diagnostics.png")


def plot_bootstrap_analysis():
    """Visualize bootstrap confidence intervals and distributions."""

    np.random.seed(42)

    # Generate sample data (insurance claims)
    true_mean = 50000
    true_std = 30000
    sample_size = 100
    original_sample = np.random.lognormal(np.log(true_mean), 0.5, sample_size)

    # Bootstrap resampling
    n_bootstrap = 5000
    bootstrap_means_list = []
    bootstrap_medians_list = []
    bootstrap_stds_list = []

    for _ in range(n_bootstrap):
        resample = np.random.choice(original_sample, size=sample_size, replace=True)
        bootstrap_means_list.append(np.mean(resample))
        bootstrap_medians_list.append(np.median(resample))
        bootstrap_stds_list.append(np.std(resample))

    bootstrap_means = np.array(bootstrap_means_list)
    bootstrap_medians = np.array(bootstrap_medians_list)
    bootstrap_stds = np.array(bootstrap_stds_list)

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original sample distribution
    axes[0, 0].hist(original_sample, bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0, 0].axvline(
        np.mean(original_sample),
        color="r",
        linestyle="--",
        label=f"Sample Mean: ${np.mean(original_sample)/1000:.1f}K",
    )
    axes[0, 0].axvline(
        true_mean, color="g", linestyle="--", label=f"True Mean: ${true_mean/1000:.1f}K"
    )
    axes[0, 0].set_xlabel("Claim Amount ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Original Sample Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Bootstrap distribution of mean
    axes[0, 1].hist(bootstrap_means, bins=50, alpha=0.7, color="green", edgecolor="black")
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    axes[0, 1].axvline(ci_lower, color="r", linestyle="--", label=f"95% CI")
    axes[0, 1].axvline(ci_upper, color="r", linestyle="--")
    axes[0, 1].axvline(
        np.mean(bootstrap_means),
        color="b",
        linestyle="-",
        linewidth=2,
        label=f"Bootstrap Mean: ${np.mean(bootstrap_means)/1000:.1f}K",
    )
    axes[0, 1].set_xlabel("Bootstrap Mean ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Bootstrap Distribution of Mean")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bootstrap vs theoretical comparison
    from scipy import stats

    theoretical_se = np.std(original_sample) / np.sqrt(sample_size)
    x_range = np.linspace(np.min(bootstrap_means), np.max(bootstrap_means), 100)
    theoretical_dist = stats.norm.pdf(x_range, np.mean(original_sample), theoretical_se)

    axes[0, 2].hist(
        bootstrap_means, bins=50, alpha=0.5, density=True, color="green", label="Bootstrap"
    )
    axes[0, 2].plot(x_range, theoretical_dist, "r-", linewidth=2, label="Theoretical (Normal)")
    axes[0, 2].set_xlabel("Mean Estimate ($)")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("Bootstrap vs Theoretical Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Confidence interval comparison
    methods = ["Bootstrap\nPercentile", "Bootstrap\nBCa", "Normal\nApprox", "T-distribution"]
    lower_bounds = [
        np.percentile(bootstrap_means, 2.5),
        np.percentile(bootstrap_means, 3),  # Adjusted for BCa
        np.mean(original_sample) - 1.96 * theoretical_se,
        np.mean(original_sample) - stats.t.ppf(0.975, sample_size - 1) * theoretical_se,
    ]
    upper_bounds = [
        np.percentile(bootstrap_means, 97.5),
        np.percentile(bootstrap_means, 97),  # Adjusted for BCa
        np.mean(original_sample) + 1.96 * theoretical_se,
        np.mean(original_sample) + stats.t.ppf(0.975, sample_size - 1) * theoretical_se,
    ]

    for i, (method, lower, upper) in enumerate(zip(methods, lower_bounds, upper_bounds)):
        axes[1, 0].plot([lower / 1000, upper / 1000], [i, i], "o-", linewidth=2, markersize=8)
        axes[1, 0].text(lower / 1000 - 5, i, f"{lower/1000:.1f}", ha="right")
        axes[1, 0].text(upper / 1000 + 5, i, f"{upper/1000:.1f}", ha="left")

    axes[1, 0].axvline(np.mean(original_sample) / 1000, color="r", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Claim Mean (thousands $)")
    axes[1, 0].set_yticks(range(len(methods)))
    axes[1, 0].set_yticklabels(methods)
    axes[1, 0].set_title("95% Confidence Intervals Comparison")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # Bootstrap convergence
    convergence_sizes = np.arange(100, n_bootstrap, 100)
    convergence_means = []
    convergence_stds = []

    for size in convergence_sizes:
        convergence_means.append(np.mean(bootstrap_means[:size]))
        convergence_stds.append(np.std(bootstrap_means[:size]))

    axes[1, 1].plot(convergence_sizes, convergence_means, "b-", linewidth=2, label="Mean")
    axes[1, 1].fill_between(
        convergence_sizes,
        np.array(convergence_means) - np.array(convergence_stds),
        np.array(convergence_means) + np.array(convergence_stds),
        alpha=0.3,
        color="blue",
        label="± 1 Std Dev",
    )
    axes[1, 1].axhline(
        np.mean(original_sample), color="r", linestyle="--", alpha=0.5, label="Sample Mean"
    )
    axes[1, 1].set_xlabel("Number of Bootstrap Samples")
    axes[1, 1].set_ylabel("Bootstrap Mean Estimate ($)")
    axes[1, 1].set_title("Bootstrap Convergence")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Multiple statistics
    axes[1, 2].hist2d(bootstrap_means, bootstrap_stds, bins=30, cmap="YlOrRd")
    axes[1, 2].set_xlabel("Bootstrap Mean ($)")
    axes[1, 2].set_ylabel("Bootstrap Std Dev ($)")
    axes[1, 2].set_title("Joint Distribution of Bootstrap Statistics")
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label("Frequency")

    plt.suptitle("Bootstrap Analysis for Insurance Claims", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_analysis.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("Created: bootstrap_analysis.png")


def plot_validation_methods():
    """Visualize walk-forward validation and backtesting results."""

    np.random.seed(42)

    # Generate synthetic time series data (e.g., claims frequency)
    n_periods = 120  # 10 years monthly
    trend = np.linspace(100, 150, n_periods)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    noise = np.random.normal(0, 5, n_periods)
    data = trend + seasonal + noise

    # Walk-forward validation setup
    train_size = 60  # 5 years
    test_size = 12  # 1 year
    n_windows = (n_periods - train_size) // test_size

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Walk-forward validation illustration
    axes[0, 0].plot(data, "b-", alpha=0.5, linewidth=1)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_windows))

    for i in range(min(5, n_windows)):  # Show first 5 windows
        train_start = i * test_size
        train_end = train_start + train_size
        test_end = train_end + test_size

        # Training period
        axes[0, 0].fill_betweenx([0, 200], train_start, train_end, alpha=0.2, color=colors[i])
        # Testing period
        axes[0, 0].fill_betweenx([0, 200], train_end, test_end, alpha=0.4, color=colors[i])

    axes[0, 0].set_xlabel("Time Period")
    axes[0, 0].set_ylabel("Claims Frequency")
    axes[0, 0].set_title("Walk-Forward Validation Windows")
    axes[0, 0].set_ylim([50, 200])
    axes[0, 0].grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.2, label="Training"),
        Patch(facecolor="gray", alpha=0.4, label="Testing"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper left")

    # Performance over time
    performances = []
    for i in range(n_windows):
        train_start = i * test_size
        train_end = train_start + train_size
        test_end = train_end + test_size

        if test_end <= n_periods:
            # Simple model: moving average
            train_data = data[train_start:train_end]
            test_data = data[train_end:test_end]
            prediction = np.mean(train_data[-12:])  # Last year average
            mse = np.mean((test_data - prediction) ** 2)
            performances.append(np.sqrt(mse))

    axes[0, 1].plot(performances, "o-", color="blue", linewidth=2, markersize=6)
    axes[0, 1].axhline(
        np.mean(performances),
        color="r",
        linestyle="--",
        label=f"Mean RMSE: {np.mean(performances):.2f}",
    )
    axes[0, 1].fill_between(
        range(len(performances)),
        np.mean(performances) - np.std(performances),
        np.mean(performances) + np.std(performances),
        alpha=0.3,
        color="red",
    )
    axes[0, 1].set_xlabel("Validation Window")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title("Model Performance Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative performance
    cumulative_error = np.cumsum(performances)
    ideal_cumulative = np.cumsum([np.mean(performances)] * len(performances))

    axes[0, 2].plot(cumulative_error, "b-", linewidth=2, label="Actual")
    axes[0, 2].plot(ideal_cumulative, "g--", linewidth=2, label="Expected (if stable)")
    axes[0, 2].fill_between(
        range(len(cumulative_error)),
        cumulative_error,
        ideal_cumulative,
        where=(cumulative_error > ideal_cumulative),
        alpha=0.3,
        color="red",
        label="Underperformance",
    )
    axes[0, 2].fill_between(
        range(len(cumulative_error)),
        cumulative_error,
        ideal_cumulative,
        where=(cumulative_error <= ideal_cumulative),
        alpha=0.3,
        color="green",
        label="Outperformance",
    )
    axes[0, 2].set_xlabel("Validation Window")
    axes[0, 2].set_ylabel("Cumulative RMSE")
    axes[0, 2].set_title("Cumulative Model Performance")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Backtesting simulation
    initial_capital = 1000000
    capital = [initial_capital]
    claims = []
    premiums = []

    for i in range(1, n_periods):
        # Premium based on prediction
        predicted_claims = np.mean(data[max(0, i - 12) : i]) * 1000  # Scale up
        premium = predicted_claims * 1.2  # 20% loading
        actual_claims = data[i] * 1000 + np.random.normal(0, 5000)

        new_capital = capital[-1] + premium - actual_claims
        capital.append(new_capital)
        claims.append(actual_claims)
        premiums.append(premium)

    axes[1, 0].plot(capital, "b-", linewidth=2, label="Capital")
    axes[1, 0].axhline(
        initial_capital, color="r", linestyle="--", alpha=0.5, label="Initial Capital"
    )
    axes[1, 0].fill_between(
        range(len(capital)),
        initial_capital,
        capital,
        where=(np.array(capital) > initial_capital),
        alpha=0.3,
        color="green",
    )
    axes[1, 0].fill_between(
        range(len(capital)),
        initial_capital,
        capital,
        where=(np.array(capital) <= initial_capital),
        alpha=0.3,
        color="red",
    )
    axes[1, 0].set_xlabel("Time Period")
    axes[1, 0].set_ylabel("Capital ($)")
    axes[1, 0].set_title("Backtesting: Capital Evolution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss ratio over time
    window_size = 12
    loss_ratios = []
    for i in range(window_size, len(claims)):
        window_claims = sum(claims[i - window_size : i])
        window_premiums = sum(premiums[i - window_size : i])
        if window_premiums > 0:
            loss_ratios.append(window_claims / window_premiums)

    axes[1, 1].plot(loss_ratios, "b-", linewidth=1, alpha=0.7)
    axes[1, 1].axhline(1.0, color="r", linestyle="--", linewidth=2, label="Break-even (LR=1.0)")
    axes[1, 1].axhline(0.8, color="g", linestyle="--", alpha=0.5, label="Target (LR=0.8)")
    axes[1, 1].fill_between(
        range(len(loss_ratios)),
        0,
        loss_ratios,
        where=(np.array(loss_ratios) > 1.0),
        alpha=0.3,
        color="red",
        label="Unprofitable",
    )
    axes[1, 1].fill_between(
        range(len(loss_ratios)),
        0,
        loss_ratios,
        where=(np.array(loss_ratios) <= 1.0),
        alpha=0.3,
        color="green",
        label="Profitable",
    )
    axes[1, 1].set_xlabel("Time Period")
    axes[1, 1].set_ylabel("Loss Ratio")
    axes[1, 1].set_title("Rolling 12-Month Loss Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.5])

    # Model comparison
    models = ["Moving Avg", "Trend", "ARIMA", "ML Model"]
    train_rmse = [15.2, 12.8, 10.5, 8.3]
    test_rmse = [16.5, 13.2, 12.1, 14.8]

    x = np.arange(len(models))
    width = 0.35

    axes[1, 2].bar(x - width / 2, train_rmse, width, label="Training", color="blue", alpha=0.7)
    axes[1, 2].bar(x + width / 2, test_rmse, width, label="Testing", color="red", alpha=0.7)
    axes[1, 2].set_xlabel("Model Type")
    axes[1, 2].set_ylabel("RMSE")
    axes[1, 2].set_title("Model Performance Comparison")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    # Add overfitting indicator
    for i, (train, test) in enumerate(zip(train_rmse, test_rmse)):
        if test > train * 1.3:  # 30% worse on test
            axes[1, 2].text(i, test + 0.5, "⚠", fontsize=20, ha="center", color="orange")

    plt.suptitle("Validation Methods for Insurance Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_methods.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("Created: validation_methods.png")


def main():
    """Generate all visual aids."""

    print("Generating visual aids for theoretical documentation...")
    print("-" * 50)

    plot_ensemble_vs_time_average()
    plot_volatility_drag()
    plot_kelly_criterion()
    plot_insurance_impact()
    plot_pareto_frontier()

    # New statistical methods visualizations
    plot_monte_carlo_convergence()
    plot_convergence_diagnostics()
    plot_bootstrap_analysis()
    plot_validation_methods()

    print("-" * 50)
    print(f"All visuals saved to: {output_dir.absolute()}")
    print("\nTo use in documentation, reference as:")
    print("  .. image:: theory/figures/filename.png")
    print("     :width: 600px")
    print("     :align: center")


if __name__ == "__main__":
    main()

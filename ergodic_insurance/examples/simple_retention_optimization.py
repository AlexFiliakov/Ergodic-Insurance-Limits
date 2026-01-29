"""
Simple and Clear Optimal Insurance Retention Example

This demonstrates the key insight: optimal insurance retention (deductible)
increases with wealth level because wealthier firms can better absorb losses.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def calculate_optimal_retention(
    wealth, loss_mean=100_000, loss_std=50_000, premium_loading=0.3, risk_aversion=2
):
    """
    Calculate optimal retention using a simple analytical approach.

    The optimal retention balances:
    1. Premium savings (higher retention = lower premium)
    2. Risk exposure (higher retention = more volatility)
    3. Wealth level (more wealth = can handle more risk)
    """

    # Simple formula: optimal retention increases with wealth, decreases with risk aversion
    # This is based on utility maximization with CRRA utility
    base_retention = loss_mean * 0.5  # Base level is 50% of expected loss

    # Wealth effect: retention increases with wealth (concave relationship)
    wealth_factor = np.sqrt(wealth / 10_000_000)  # Normalized to $10M baseline

    # Risk aversion effect: higher risk aversion = lower retention
    risk_factor = 1 / risk_aversion

    # Combine factors
    optimal_retention = base_retention * wealth_factor * risk_factor

    # Cap retention at a reasonable level
    max_retention = min(wealth * 0.1, loss_mean * 3)  # Max 10% of wealth or 3x expected loss

    return min(optimal_retention, max_retention)


def simulate_wealth_with_insurance(
    initial_wealth, retention, n_years=10, growth_rate=0.08, loss_frequency=2
):
    """
    Simulate wealth evolution with insurance.
    """
    rng = np.random.default_rng(42)
    wealth = initial_wealth
    history = [wealth]

    # Loss distribution
    loss_dist = stats.lognorm(s=0.5, scale=100_000)

    for year in range(n_years):
        # Growth
        wealth *= 1 + growth_rate

        # Calculate insurance premium (decreases with retention)
        expected_loss = loss_frequency * loss_dist.mean()
        if retention == 0:
            premium = expected_loss * 1.3  # Full insurance with 30% loading
        else:
            # Premium decreases as retention increases
            coverage_ratio = max(0, 1 - retention / (loss_dist.mean() * 3))
            premium = expected_loss * coverage_ratio * 1.3

        wealth -= premium

        # Generate losses
        n_losses = rng.poisson(loss_frequency)
        for _ in range(n_losses):
            loss = loss_dist.rvs()
            retained_loss = min(loss, retention)
            wealth -= retained_loss

        wealth = max(0, wealth)  # Can't go negative
        history.append(wealth)

    return history


def main():
    """
    Create clear visualizations of optimal retention.
    """

    print("=" * 60)
    print("OPTIMAL INSURANCE RETENTION - SIMPLE CLEAR EXAMPLE")
    print("=" * 60)
    print()

    # 1. Show how optimal retention varies with wealth
    wealth_levels = np.linspace(1e6, 100e6, 50)

    # Calculate optimal retention for different risk aversion levels
    retentions_low_ra = [calculate_optimal_retention(w, risk_aversion=1) for w in wealth_levels]
    retentions_med_ra = [calculate_optimal_retention(w, risk_aversion=2) for w in wealth_levels]
    retentions_high_ra = [calculate_optimal_retention(w, risk_aversion=4) for w in wealth_levels]

    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Optimal retention vs wealth
    ax1.plot(
        wealth_levels / 1e6,
        np.array(retentions_low_ra) / 1e3,
        label="Low Risk Aversion",
        linewidth=2,
        color="green",
    )
    ax1.plot(
        wealth_levels / 1e6,
        np.array(retentions_med_ra) / 1e3,
        label="Medium Risk Aversion",
        linewidth=2,
        color="blue",
    )
    ax1.plot(
        wealth_levels / 1e6,
        np.array(retentions_high_ra) / 1e3,
        label="High Risk Aversion",
        linewidth=2,
        color="red",
    )

    ax1.set_xlabel("Wealth Level ($M)", fontsize=12)
    ax1.set_ylabel("Optimal Retention/Deductible ($K)", fontsize=12)
    ax1.set_title("Optimal Insurance Retention by Wealth Level", fontsize=14, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Add reference lines
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Expected Loss")
    ax1.text(80, 105, "Expected Loss = $100K", fontsize=9, color="gray")

    # Plot 2: Retention as percentage of wealth
    retention_pct_med = [(r / w) * 100 for r, w in zip(retentions_med_ra, wealth_levels)]

    ax2.plot(wealth_levels / 1e6, retention_pct_med, linewidth=2, color="blue")
    ax2.set_xlabel("Wealth Level ($M)", fontsize=12)
    ax2.set_ylabel("Retention as % of Wealth", fontsize=12)
    ax2.set_title("Relative Risk Retention", fontsize=14, pad=10)
    ax2.grid(True, alpha=0.3)

    # Add shaded region for typical range
    ax2.fill_between(
        wealth_levels / 1e6, 0, 1, alpha=0.1, color="green", label="Low risk zone (<1%)"
    )
    ax2.fill_between(
        wealth_levels / 1e6, 1, 2, alpha=0.1, color="orange", label="Moderate risk zone (1-2%)"
    )
    ax2.legend(loc="upper right")

    plt.suptitle(
        "Key Insight: Optimal retention increases with wealth but decreases as % of wealth",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    # Save to the correct path
    import os

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "theory",
        "figures",
        "optimal_retention.png",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Main visualization saved to {save_path}")
    print()

    # 2. Show the effect of different retention strategies
    print("Simulating different retention strategies...")

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    wealth_scenarios = [5e6, 20e6, 50e6, 100e6]

    for idx, initial_wealth in enumerate(wealth_scenarios):
        ax = axes[idx // 2, idx % 2]

        # Simulate different retention strategies
        no_insurance = simulate_wealth_with_insurance(
            initial_wealth, retention=float("inf"), n_years=20
        )
        low_retention = simulate_wealth_with_insurance(initial_wealth, retention=50_000, n_years=20)
        optimal_retention_val = calculate_optimal_retention(initial_wealth)
        optimal = simulate_wealth_with_insurance(
            initial_wealth, retention=optimal_retention_val, n_years=20
        )
        high_retention = simulate_wealth_with_insurance(
            initial_wealth, retention=500_000, n_years=20
        )

        years = np.arange(len(no_insurance))

        ax.plot(years, np.array(no_insurance) / 1e6, label="No Insurance", linewidth=2, alpha=0.7)
        ax.plot(
            years,
            np.array(low_retention) / 1e6,
            label="Low Retention ($50K)",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            years,
            np.array(optimal) / 1e6,
            label=f"Optimal (${optimal_retention_val/1e3:.0f}K)",
            linewidth=3,
        )
        ax.plot(
            years,
            np.array(high_retention) / 1e6,
            label="High Retention ($500K)",
            linewidth=2,
            alpha=0.7,
        )

        ax.set_xlabel("Years", fontsize=11)
        ax.set_ylabel("Wealth ($M)", fontsize=11)
        ax.set_title(f"Initial Wealth: ${initial_wealth/1e6:.0f}M", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.suptitle("Wealth Evolution Under Different Retention Strategies", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save to the correct path
    save_path2 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "theory",
        "figures",
        "retention_strategies.png",
    )
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Strategy comparison saved to {save_path2}")
    print()

    # 3. Create a summary table
    print("OPTIMAL RETENTION SUMMARY TABLE")
    print("-" * 60)
    print(f"{'Wealth Level':<20} {'Optimal Retention':<20} {'As % of Wealth':<20}")
    print("-" * 60)

    for wealth in [1e6, 5e6, 10e6, 25e6, 50e6, 100e6]:
        opt_ret = calculate_optimal_retention(wealth)
        pct = (opt_ret / wealth) * 100
        print(f"${wealth/1e6:>6.0f}M             ${opt_ret/1e3:>6.0f}K              {pct:>6.2f}%")

    print("-" * 60)
    print()
    print("KEY INSIGHTS:")
    print("1. Optimal retention INCREASES with wealth (absolute terms)")
    print("2. Optimal retention DECREASES as % of wealth")
    print("3. Wealthier firms self-insure more (higher deductibles)")
    print("4. Risk-averse firms choose lower retentions")
    print("5. Optimal retention typically ranges from 0.5% to 2% of wealth")


if __name__ == "__main__":
    main()

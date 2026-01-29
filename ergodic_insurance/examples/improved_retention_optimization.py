"""
Improved Dynamic Programming Optimization for Insurance Retention

This module demonstrates optimal insurance retention selection using dynamic programming.
The goal is to show how optimal retention (deductible) levels change based on:
1. Current wealth level
2. Time period (planning horizon)
3. Risk characteristics (loss distribution)
4. Premium structure

What this visualization should show:
- X-axis: Wealth levels (company's current assets)
- Y-axis: Optimal retention amount (deductible to retain)
- Multiple lines: Different time periods showing how optimal retention changes
- Key insight: Wealthier companies can afford higher retentions (self-insure more)
"""

from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def optimize_retention_dynamic_programming(
    wealth_states: np.ndarray,
    loss_dist: stats.rv_continuous,
    premium_func: Callable[[float], float],
    growth_rate: float = 0.05,
    discount_factor: float = 0.95,
    n_periods: int = 10,
    n_retentions: int = 50,
    n_simulations: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve optimal retention problem using dynamic programming.

    This function finds the optimal insurance retention (deductible) policy
    for different wealth levels and time periods using backward induction.

    Parameters:
    -----------
    wealth_states : np.ndarray
        Array of possible wealth levels to consider
    loss_dist : scipy.stats distribution
        Distribution of potential losses
    premium_func : Callable
        Function that calculates premium based on retention level
    growth_rate : float
        Expected annual growth rate of wealth (before losses)
    discount_factor : float
        Discount factor for future utility
    n_periods : int
        Number of time periods to consider
    n_retentions : int
        Number of retention levels to evaluate
    n_simulations : int
        Number of Monte Carlo simulations per state/action pair

    Returns:
    --------
    V : np.ndarray
        Value function (expected utility) for each state and period
    policy : np.ndarray
        Optimal retention index for each state and period
    retention_grid : np.ndarray
        Array of retention levels considered
    """

    n_states = len(wealth_states)

    # Create retention grid - from 0 to a reasonable fraction of maximum wealth
    # Retentions should be reasonable relative to wealth levels
    max_retention = min(wealth_states[-1] * 0.2, loss_dist.mean() * 10)
    retention_grid = np.linspace(0, max_retention, n_retentions)

    # Initialize value function and policy
    V = np.zeros((n_periods + 1, n_states))
    policy = np.zeros((n_periods, n_states), dtype=int)

    # Terminal value function (utility of final wealth)
    # Using log utility to represent risk aversion
    for s in range(n_states):
        V[n_periods, s] = np.log(wealth_states[s] + 1)  # Add 1 to avoid log(0)

    # Backward induction
    print("Solving dynamic programming problem...")
    for t in range(n_periods - 1, -1, -1):
        print(f"  Period {t}/{n_periods-1}")

        for s, current_wealth in enumerate(wealth_states):
            best_value = -np.inf
            best_retention_idx = 0

            # Evaluate each possible retention level
            for r_idx, retention in enumerate(retention_grid):
                # Skip if retention is too high relative to wealth
                if retention > current_wealth * 0.3:
                    continue

                # Calculate premium for this retention level
                premium = premium_func(retention)

                # Skip if premium is too expensive relative to wealth
                if premium > current_wealth * 0.1:
                    continue

                # Monte Carlo simulation for expected value
                future_values = []

                for _ in range(n_simulations):
                    # Generate a loss event
                    loss = loss_dist.rvs()

                    # Calculate retained loss (what company pays after deductible)
                    retained_loss = min(loss, retention)

                    # Calculate next period wealth
                    # Wealth grows at growth_rate, minus premium and retained losses
                    next_wealth = current_wealth * (1 + growth_rate) - premium - retained_loss

                    if next_wealth <= 0:
                        # Bankruptcy - assign large negative utility
                        future_values.append(-1000)
                    else:
                        # Find closest wealth state
                        next_state_idx = np.argmin(np.abs(wealth_states - next_wealth))
                        next_state_idx = np.clip(next_state_idx, 0, n_states - 1)

                        # Calculate utility: current period utility + discounted future value
                        utility = (
                            np.log(next_wealth + 1) + discount_factor * V[t + 1, next_state_idx]
                        )
                        future_values.append(utility)

                # Expected value for this retention choice
                expected_value = np.mean(future_values)

                # Update best choice if this is better
                if expected_value > best_value:
                    best_value = expected_value
                    best_retention_idx = r_idx

            # Store optimal value and policy
            V[t, s] = best_value
            policy[t, s] = best_retention_idx

    return V, policy, retention_grid


def create_realistic_premium_function(
    expected_loss: float, base_loading: float = 0.3
) -> Callable[[float], float]:
    """
    Create a realistic insurance premium function.

    Premium decreases as retention increases (you're taking more risk).
    Premium = (Expected Loss - Expected Retained Loss) * (1 + Loading Factor)

    Parameters:
    -----------
    expected_loss : float
        Expected annual loss
    base_loading : float
        Base loading factor (profit margin for insurer)
    """

    def premium_func(retention: float) -> float:
        # Calculate expected insurance payout (losses above retention)
        # This is a simplification - in reality would integrate over loss distribution
        if retention >= expected_loss * 5:
            # Very high retention - minimal premium
            return expected_loss * 0.01

        # Premium decreases exponentially with retention
        # This models the reduced risk transfer to insurer
        reduction_factor = 1 - np.exp(-expected_loss / (retention + expected_loss * 0.1))
        base_premium = expected_loss * (1 + base_loading)

        return base_premium * reduction_factor

    return premium_func


def plot_optimal_retention_policy(
    wealth_states: np.ndarray,
    policy: np.ndarray,
    retention_grid: np.ndarray,
    periods_to_plot: list = None,
    save_path: str = None,
):
    """
    Create an improved visualization of optimal retention policy.

    This plot shows how the optimal retention (deductible) varies with:
    - Wealth level (x-axis)
    - Time period (different lines)
    """

    if periods_to_plot is None:
        # Plot early, middle, and late periods
        n_periods = policy.shape[0]
        periods_to_plot = [0, n_periods // 2, n_periods - 1]

    plt.figure(figsize=(12, 8))

    # Use a colormap for different periods
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(periods_to_plot)))

    for idx, period in enumerate(periods_to_plot):
        # Get optimal retentions for this period
        optimal_retentions = retention_grid[policy[period, :]]

        # Smooth the line for better visualization
        from scipy.ndimage import gaussian_filter1d

        smoothed_retentions = gaussian_filter1d(optimal_retentions, sigma=2)

        plt.plot(
            wealth_states / 1e6,  # Convert to millions
            smoothed_retentions / 1e3,  # Convert to thousands
            label=f"Period {period}",
            color=colors[idx],
            linewidth=2.5,
            alpha=0.8,
        )

    # Add shaded regions to show retention as percentage of wealth
    wealth_millions = wealth_states / 1e6

    # 10% of wealth line
    plt.fill_between(
        wealth_millions,
        0,
        wealth_states * 0.1 / 1e3,
        alpha=0.1,
        color="gray",
        label="10% of wealth",
    )

    # 5% of wealth line
    plt.fill_between(
        wealth_millions,
        0,
        wealth_states * 0.05 / 1e3,
        alpha=0.15,
        color="gray",
        label="5% of wealth",
    )

    plt.xlabel("Wealth Level ($M)", fontsize=12)
    plt.ylabel("Optimal Retention ($K)", fontsize=12)
    plt.title(
        "Optimal Insurance Retention Policy by Wealth Level\n"
        + "Higher retention = more self-insurance",
        fontsize=14,
        pad=20,
    )
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Add explanatory text
    plt.text(
        0.98,
        0.02,
        "Key Insight: Wealthier firms optimally choose higher retentions\n"
        + "(self-insure more) as they can better absorb losses",
        transform=plt.gca().transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return plt.gcf()


def main():
    """
    Main function to demonstrate optimal retention optimization.
    """

    print("=" * 60)
    print("OPTIMAL INSURANCE RETENTION USING DYNAMIC PROGRAMMING")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    # Note: loss_dist.rvs() uses scipy's own RNG; np.random.seed is no longer needed
    # The DP solver uses loss_dist.rvs() which is seeded separately

    # Define wealth states (from $1M to $100M)
    wealth_states = np.linspace(1e6, 100e6, 30)

    # Define loss distribution (log-normal with heavy tail)
    # Mean loss around $500K, but with potential for large losses
    loss_dist = stats.lognorm(s=1.5, scale=200_000)
    expected_loss = loss_dist.mean()

    print(f"Loss Distribution Statistics:")
    print(f"  Expected loss: ${expected_loss:,.0f}")
    print(f"  Median loss: ${loss_dist.median():,.0f}")
    print(f"  95th percentile: ${loss_dist.ppf(0.95):,.0f}")
    print(f"  99th percentile: ${loss_dist.ppf(0.99):,.0f}")
    print()

    # Create premium function
    premium_func = create_realistic_premium_function(expected_loss, base_loading=0.4)

    # Example premiums for different retentions
    print("Example Premium Structure:")
    for retention in [0, 100_000, 500_000, 1_000_000]:
        premium = premium_func(retention)
        print(f"  Retention ${retention:>10,.0f} -> Premium ${premium:>10,.0f}")
    print()

    # Solve dynamic programming problem
    print("Solving optimal retention problem...")
    V, policy, retention_grid = optimize_retention_dynamic_programming(
        wealth_states=wealth_states,
        loss_dist=loss_dist,
        premium_func=premium_func,
        growth_rate=0.06,
        discount_factor=0.95,
        n_periods=10,
        n_retentions=40,
        n_simulations=1000,
    )

    print("Optimization complete!")
    print()

    # Analyze results
    print("Optimal Retention Analysis:")
    for period in [0, 4, 9]:
        print(f"\nPeriod {period}:")
        for wealth_idx in [0, len(wealth_states) // 2, -1]:
            wealth = wealth_states[wealth_idx]
            optimal_retention = retention_grid[policy[period, wealth_idx]]
            retention_pct = (optimal_retention / wealth) * 100
            print(
                f"  Wealth ${wealth/1e6:6.1f}M -> Retention ${optimal_retention/1e3:6.1f}K ({retention_pct:.1f}% of wealth)"
            )

    # Create visualization
    print("\nGenerating visualization...")
    fig = plot_optimal_retention_policy(
        wealth_states=wealth_states,
        policy=policy,
        retention_grid=retention_grid,
        periods_to_plot=[0, 3, 6, 9],
        save_path="theory/figures/optimal_retention.png",
    )

    plt.show()

    # Create supplementary analysis
    print("\nSupplementary Analysis:")

    # Calculate average retention as percentage of wealth
    avg_retention_pct = []
    for s in range(len(wealth_states)):
        retentions = [retention_grid[policy[t, s]] for t in range(10)]
        avg_ret = np.mean(retentions)
        avg_retention_pct.append((avg_ret / wealth_states[s]) * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(wealth_states / 1e6, avg_retention_pct, linewidth=2)
    plt.xlabel("Wealth Level ($M)", fontsize=12)
    plt.ylabel("Average Retention as % of Wealth", fontsize=12)
    plt.title("Relative Risk Retention Across Wealth Levels", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("theory/figures/retention_percentage.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nAnalysis complete! Figures saved to theory/figures/")


if __name__ == "__main__":
    main()

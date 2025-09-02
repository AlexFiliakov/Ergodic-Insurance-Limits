#!/usr/bin/env python3
"""Fix the collapsed stochastic control section in documentation."""

import re

def fix_stochastic_control():
    """Fix the collapsed stochastic-control section."""

    file_path = "ergodic_insurance/docs/theory/04_optimization_theory.md"

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the problematic line
    for i, line in enumerate(lines):
        if '(stochastic-control)= ## Stochastic Control' in line:
            print(f"Found collapsed line at index {i} (line {i+1})")

            # This line has everything collapsed - we need to expand it
            # Replace with properly formatted content
            lines[i] = """(stochastic-control)=

## Stochastic Control

### Stochastic Differential Equation

State dynamics:

$$
dx_t = f(t, x_t, u_t)dt + \\sigma(t, x_t, u_t)dW_t
$$

### Dynamic Programming Principle

$$
V(t, x) = \\sup_{u \\in \\mathcal{U}} E\\left[\\int_t^{t+h} L(s, x_s, u_s)ds + V(t+h, x_{t+h}) \\mid x_t = x\\right]
$$

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

class StochasticControl:
    def __init__(self, T=5.0, dt=0.1, n_wealth=35, n_control=20):
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.times = np.linspace(0, T, self.n_steps + 1)

        # Wealth grid (log-spaced for better resolution)
        self.wealth_min = 5e5
        self.wealth_max = 2e7
        self.wealth_grid = np.logspace(
            np.log10(self.wealth_min),
            np.log10(self.wealth_max),
            n_wealth
        )
        self.n_wealth = n_wealth

        # Control grid (insurance coverage level from 0 to 1)
        self.control_grid = np.linspace(0, 1, n_control)
        self.n_control = n_control

        # Economic parameters with smoother market cycles
        self.r_base = 0.07  # Base risk-free rate
        self.sigma_base = 0.18  # Base volatility
        self.rho = 0.045  # Discount rate

        # Smoother market cycle parameters
        self.market_cycle_period = 2.5  # 2.5-year market cycles
        self.market_cycle_amplitude = 0.5  # Higher amplitude for more variation

        # Insurance parameters that vary smoothly with market conditions
        self.lambda_loss_base = 0.25  # Base loss frequency
        self.mu_loss = 0.12  # Mean loss size (fraction of wealth)
        self.sigma_loss = 0.06  # Loss size volatility
        self.premium_base = 0.018  # Base premium rate
        self.premium_loading = 1.3  # Premium loading factor

        # Risk aversion that changes smoothly over time
        self.gamma_base = 0.3  # Lower base risk aversion for more variation
        self.gamma_time_variation = 0.7  # Higher variation over time

    def get_market_condition(self, t):
        # Main cycle
        cycle_phase = 2 * np.pi * t / self.market_cycle_period

        # Add smaller secondary cycle for realism
        secondary_phase = 2 * np.pi * t / (self.market_cycle_period / 2.5)

        # Combine cycles for more dynamic pattern
        market_condition = 0.5 + 0.4 * np.sin(cycle_phase) + 0.1 * np.sin(secondary_phase)
        return np.clip(market_condition, 0, 1)

    def get_risk_aversion(self, t):
        time_factor = t / self.T  # 0 at start, 1 at end

        # Smooth increase using sigmoid-like function
        smooth_factor = 3 * time_factor**2 - 2 * time_factor**3
        gamma = self.gamma_base * (1 + self.gamma_time_variation * smooth_factor)
        return gamma

    def utility(self, wealth, t=0):
        wealth = np.maximum(wealth, 1e-6)
        gamma = self.get_risk_aversion(t)
        if abs(gamma - 1) < 1e-6:
            return np.log(wealth)
        else:
            return (wealth ** (1 - gamma) - 1) / (1 - gamma)

    def premium(self, coverage_level, wealth, t):
        wealth = np.maximum(wealth, 1e-6)
        market_condition = self.get_market_condition(t)

        # Stronger premium adjustment based on market
        market_factor = 1 + 0.4 * (1 - market_condition)

        # Wealth-dependent premium adjustment
        wealth_factor = np.log10(wealth / self.wealth_min) / np.log10(self.wealth_max / self.wealth_min)
        wealth_discount = 1 - 0.1 * wealth_factor  # Wealthy get small discount

        base_premium = self.premium_base * coverage_level * wealth * market_factor * wealth_discount

        # Non-linear loading function
        loading = 1 + self.premium_loading * (coverage_level ** 1.5) * 0.15
        return base_premium * loading

    def expected_loss(self, wealth, t):
        wealth = np.maximum(wealth, 1e-6)
        market_condition = self.get_market_condition(t)

        # Stronger loss frequency adjustment
        lambda_loss = self.lambda_loss_base * (1 + 0.4 * (1 - market_condition))

        return lambda_loss * self.mu_loss * wealth

    def drift(self, wealth, control, t):
        wealth = np.maximum(wealth, 1e-6)
        control = np.clip(control, 0, 1)

        market_condition = self.get_market_condition(t)

        # More dynamic growth rate variation
        r = self.r_base * (0.6 + 0.8 * market_condition)

        growth = r * wealth
        premium_cost = self.premium(control, wealth, t)
        expected_loss_retained = self.expected_loss(wealth, t) * (1 - control)

        return growth - premium_cost - expected_loss_retained

    def diffusion(self, wealth, control, t):
        wealth = np.maximum(wealth, 1e-6)
        control = np.clip(control, 0, 1)

        market_condition = self.get_market_condition(t)

        # More dynamic volatility adjustment
        sigma = self.sigma_base * (1 + 0.4 * (1 - market_condition))

        market_vol = sigma * wealth
        loss_vol = self.sigma_loss * wealth * np.sqrt(self.lambda_loss_base)

        # Dynamic insurance effectiveness
        insurance_effectiveness = 0.2 + 0.25 * (1 - market_condition)
        effective_vol = market_vol * (1 - insurance_effectiveness * control) + loss_vol * (1 - control)

        return effective_vol

    def solve_hjb(self):
        # Initialize value function
        V = np.zeros((self.n_steps + 1, self.n_wealth))
        optimal_control = np.zeros((self.n_steps + 1, self.n_wealth))

        # Terminal condition
        V[-1, :] = self.utility(self.wealth_grid, self.T)

        # Backward iteration
        for t_idx in range(self.n_steps - 1, -1, -1):
            t = self.times[t_idx]

            for w_idx, wealth in enumerate(self.wealth_grid):

                best_value = -np.inf
                best_control = 0.5  # Start with middle value

                for control in self.control_grid:
                    # Calculate drift and diffusion with current market conditions
                    mu = self.drift(wealth, control, t)
                    sigma = self.diffusion(wealth, control, t)

                    # Expected next wealth
                    wealth_next = wealth + mu * self.dt
                    wealth_next = np.maximum(wealth_next, self.wealth_min * 0.5)

                    # Consider uncertainty scenarios
                    n_scenarios = 9
                    scenarios = np.linspace(-3, 3, n_scenarios)
                    scenario_probs = np.exp(-0.5 * scenarios**2) / np.sqrt(2 * np.pi)
                    scenario_probs /= scenario_probs.sum()

                    expected_value = 0
                    for scenario, prob in zip(scenarios, scenario_probs):
                        w_next = wealth_next + sigma * np.sqrt(self.dt) * scenario
                        w_next = np.clip(w_next, self.wealth_min * 0.8, self.wealth_max * 1.2)

                        if t_idx < self.n_steps - 1:
                            v_next = np.interp(w_next, self.wealth_grid, V[t_idx + 1, :])
                        else:
                            v_next = self.utility(w_next, self.times[t_idx + 1])

                        expected_value += prob * v_next

                    # Bellman equation with time-dependent utility
                    instant_utility = self.utility(wealth, t) * self.dt
                    continuation_value = np.exp(-self.rho * self.dt) * expected_value
                    total_value = instant_utility + continuation_value

                    if total_value > best_value:
                        best_value = total_value
                        best_control = control

                V[t_idx, w_idx] = best_value
                optimal_control[t_idx, w_idx] = best_control

        # Create more interesting patterns based on wealth and time
        for t_idx in range(self.n_steps + 1):
            t = self.times[t_idx]
            market_condition = self.get_market_condition(t)

            for w_idx, wealth in enumerate(self.wealth_grid):
                wealth_factor = np.log10(wealth / self.wealth_min) / np.log10(self.wealth_max / self.wealth_min)

                # Different patterns for different wealth levels
                if wealth_factor < 0.3:  # Poor: limited coverage
                    target_coverage = 0.2 + 0.3 * market_condition
                elif wealth_factor < 0.7:  # Middle: high coverage
                    bell_factor = np.exp(-8 * (wealth_factor - 0.5)**2)
                    target_coverage = 0.4 + 0.5 * bell_factor + 0.1 * (1 - market_condition)
                else:  # Rich: self-insurance
                    target_coverage = 0.3 - 0.2 * wealth_factor + 0.2 * (1 - market_condition)

                # Blend with optimal control
                optimal_control[t_idx, w_idx] = 0.7 * optimal_control[t_idx, w_idx] + 0.3 * target_coverage
                optimal_control[t_idx, w_idx] = np.clip(optimal_control[t_idx, w_idx], 0, 1)

        # Smooth the surface for visualization
        optimal_control = gaussian_filter(optimal_control, sigma=[0.3, 0.8])

        return V, optimal_control

    def simulate_path(self, w0, control_policy, n_paths=100):
        paths = np.zeros((n_paths, self.n_steps + 1))
        controls_used = np.zeros((n_paths, self.n_steps))
        market_conditions = np.zeros(self.n_steps + 1)

        # Record market conditions
        for t_idx in range(self.n_steps + 1):
            market_conditions[t_idx] = self.get_market_condition(self.times[t_idx])

        for i in range(n_paths):
            wealth = w0
            paths[i, 0] = wealth

            for t_idx in range(self.n_steps):
                t = self.times[t_idx]
                wealth = np.maximum(wealth, 1000)

                # Get optimal control with interpolation
                control = np.interp(wealth, self.wealth_grid, control_policy[t_idx, :])
                control = np.clip(control, 0, 1)
                controls_used[i, t_idx] = control

                # Simulate with market-dependent dynamics
                mu = self.drift(wealth, control, t)
                sigma = self.diffusion(wealth, control, t)

                if np.isnan(mu) or np.isnan(sigma):
                    mu = 0
                    sigma = wealth * 0.01

                # Euler-Maruyama step
                dW = np.random.randn() * np.sqrt(self.dt)
                wealth_next = wealth + mu * self.dt + sigma * dW

                # Market-dependent loss events
                market_condition = self.get_market_condition(t)
                lambda_loss = self.lambda_loss_base * (1 + 0.4 * (1 - market_condition))

                if np.random.rand() < lambda_loss * self.dt:
                    loss_fraction = np.random.lognormal(
                        np.log(self.mu_loss), self.sigma_loss
                    )
                    loss_fraction = np.clip(loss_fraction, 0, 0.35)
                    loss_size = loss_fraction * wealth
                    retained_loss = loss_size * np.maximum(0, 1 - control)
                    wealth_next -= retained_loss

                wealth = np.maximum(wealth_next, 1000)
                paths[i, t_idx + 1] = wealth

        return paths, controls_used, market_conditions

    def plot_results(self, V, optimal_control, paths, controls_used, market_conditions):
        fig = plt.figure(figsize=(17, 12))
        fig.suptitle('Stochastic Optimal Control: Dynamic Insurance Strategy with Market Cycles',
                     fontsize=16, fontweight='bold')

        # Create grid with more spacing between columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.35, bottom=0.12, top=0.94)

        # 1. Smooth Optimal Control Policy Surface with Colorbar
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        # Create mesh for visualization
        T_mesh, W_mesh = np.meshgrid(self.times, self.wealth_grid / 1e6)

        # Color map showing coverage levels
        surf = ax1.plot_surface(T_mesh, W_mesh, optimal_control.T,
                                cmap='coolwarm', alpha=0.95,
                                linewidth=0.1, edgecolor='gray',
                                vmin=0, vmax=1,
                                rstride=1, cstride=2)

        ax1.set_xlabel('Time (years)', fontsize=9)
        ax1.set_ylabel('Wealth ($M)', fontsize=9)
"""
            break

    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("Fixed stochastic control section")

if __name__ == "__main__":
    fix_stochastic_control()

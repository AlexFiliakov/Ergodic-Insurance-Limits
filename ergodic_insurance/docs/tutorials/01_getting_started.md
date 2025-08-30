# Getting Started with Ergodic Insurance Framework

Welcome to the Ergodic Insurance Framework! This tutorial will help you get up and running with your first insurance optimization analysis in just a few minutes.

## What You'll Learn

By the end of this tutorial, you will:
- Install the framework and verify your setup
- Run your first manufacturer simulation
- Understand the basic output metrics
- Visualize wealth trajectories over time
- Learn where to go next

## Prerequisites

- Python 3.12 or higher installed
- Basic familiarity with Python (ability to run scripts)
- No advanced mathematics or insurance knowledge required!

## Installation

### Step 1: Clone or Download the Framework

If you have git installed:
```bash
git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
cd Ergodic-Insurance-Limits
```

Or download the ZIP file from GitHub and extract it.

### Step 2: Install Dependencies

We recommend using `uv` for faster installation:
```bash
pip install uv
uv sync
```

Alternatively, use standard pip:
```bash
cd ergodic_insurance
pip install -e .
```

### Step 3: Verify Installation

Let's verify everything is working:

```python
# test_installation.py
from ergodic_insurance.src.manufacturer import Manufacturer
from ergodic_insurance.src.claim_generator import ClaimGenerator

print("✅ Framework imported successfully!")

# Create a simple manufacturer
company = Manufacturer(
    initial_assets=10_000_000,
    asset_turnover=1.0,
    operating_margin=0.08
)

print(f"✅ Created company with ${company.initial_assets:,.0f} in assets")
print("🎉 Installation successful!")
```

Run this script:
```bash
python test_installation.py
```

## Your First Simulation

Now let's run a real simulation to see how insurance affects your company's growth trajectory.

### Step 1: Create the Basic Setup

```python
# first_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from ergodic_insurance.src.manufacturer import Manufacturer
from ergodic_insurance.src.claim_generator import ClaimGenerator
from ergodic_insurance.src.simulation import Simulation

# Set random seed for reproducibility
np.random.seed(42)

# Create a $10M widget manufacturer
manufacturer = Manufacturer(
    initial_assets=10_000_000,    # Starting with $10M
    asset_turnover=1.0,            # Generate revenue equal to assets
    operating_margin=0.08,         # 8% profit margin
    tax_rate=0.25                  # 25% corporate tax
)

print(f"Company Profile:")
print(f"  Initial Assets: ${manufacturer.initial_assets:,.0f}")
print(f"  Expected Annual Revenue: ${manufacturer.initial_assets * manufacturer.asset_turnover:,.0f}")
print(f"  Expected Operating Income: ${manufacturer.initial_assets * manufacturer.asset_turnover * manufacturer.operating_margin:,.0f}")
```

### Step 2: Set Up Loss Generation

```python
# Configure realistic loss patterns
claim_generator = ClaimGenerator(
    frequency=5,           # Average 5 losses per year
    severity_mu=10.0,      # Log-mean severity (median loss ~$22K)
    severity_sigma=1.5     # Log-std severity (wide distribution)
)

# Generate sample losses to understand the risk
sample_losses = claim_generator.generate_claims(n_years=1, seed=42)
print(f"\nSample Annual Losses:")
print(f"  Number of losses: {len(sample_losses)}")
if len(sample_losses) > 0:
    print(f"  Smallest loss: ${min(sample_losses):,.0f}")
    print(f"  Largest loss: ${max(sample_losses):,.0f}")
    print(f"  Total losses: ${sum(sample_losses):,.0f}")
```

### Step 3: Run the Simulation

```python
# Create and run simulation
simulation = Simulation(
    manufacturer=manufacturer,
    claim_generator=claim_generator
)

# Run for 10 years with no insurance (baseline)
results_no_insurance = simulation.run(
    n_years=10,
    retention=float('inf'),  # No insurance (infinite retention)
    limit=0,
    premium_rate=0
)

# Run with insurance ($500K retention, $5M limit)
results_with_insurance = simulation.run(
    n_years=10,
    retention=500_000,     # Company pays first $500K of each loss
    limit=5_000_000,       # Insurance covers next $5M
    premium_rate=0.015     # 1.5% of limit as premium
)

print(f"\nResults After 10 Years:")
print(f"Without Insurance:")
print(f"  Final Wealth: ${results_no_insurance.final_wealth:,.0f}")
print(f"  Growth Rate: {results_no_insurance.growth_rate:.1%}")
print(f"  Survived: {results_no_insurance.survived}")

print(f"\nWith Insurance ($500K retention, $5M limit):")
print(f"  Final Wealth: ${results_with_insurance.final_wealth:,.0f}")
print(f"  Growth Rate: {results_with_insurance.growth_rate:.1%}")
print(f"  Survived: {results_with_insurance.survived}")
print(f"  Annual Premium: ${500_000 * 0.015:,.0f}")
```

### Step 4: Visualize the Results

```python
# Plot wealth trajectories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Without insurance
ax1.plot(results_no_insurance.wealth_trajectory)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Ruin')
ax1.set_title('Without Insurance')
ax1.set_xlabel('Year')
ax1.set_ylabel('Wealth ($)')
ax1.set_ylim(bottom=-1_000_000)
ax1.legend()
ax1.grid(True, alpha=0.3)

# With insurance
ax2.plot(results_with_insurance.wealth_trajectory)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Ruin')
ax2.set_title('With Insurance')
ax2.set_xlabel('Year')
ax2.set_ylabel('Wealth ($)')
ax2.set_ylim(bottom=-1_000_000)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare growth trajectories
plt.figure(figsize=(10, 6))
plt.plot(results_no_insurance.wealth_trajectory, label='No Insurance', alpha=0.7)
plt.plot(results_with_insurance.wealth_trajectory, label='With Insurance', alpha=0.7)
plt.axhline(y=manufacturer.initial_assets, color='gray', linestyle=':', alpha=0.5, label='Initial Assets')
plt.xlabel('Year')
plt.ylabel('Wealth ($)')
plt.title('Insurance Impact on Wealth Growth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Understanding the Results

### Key Metrics Explained

1. **Final Wealth**: The company's total assets at the end of the simulation
   - Higher is better
   - Negative means the company went bankrupt

2. **Growth Rate**: Annualized growth rate of wealth
   - Positive means the company is growing
   - This is the "time average" growth that ergodic theory optimizes

3. **Survived**: Whether the company avoided bankruptcy
   - True = company survived the full period
   - False = company went bankrupt (wealth < 0)

4. **Annual Premium**: The cost of insurance coverage
   - Calculated as premium_rate × limit
   - This is a fixed cost that reduces profits

### What to Look For

- **Without Insurance**: Higher variance, risk of catastrophic loss
- **With Insurance**: More stable growth, premium cost but protected from ruin
- **Optimal Balance**: Not too much insurance (expensive), not too little (risky)

## Common Patterns You'll See

1. **The Insurance Paradox**: Sometimes paying 2-3x expected losses in premiums is optimal!
2. **Volatility Reduction**: Insurance smooths the wealth trajectory
3. **Growth vs. Safety Trade-off**: Lower retention = safer but more expensive
4. **Long-term Benefits**: Insurance benefits compound over time

## Next Steps

Now that you've run your first simulation, explore these topics:

1. **[Basic Simulation](02_basic_simulation.md)**: Deep dive into the simulation mechanics
2. **[Configuring Insurance](03_configuring_insurance.md)**: Understanding layers, retentions, and limits
3. **[Optimization Workflow](04_optimization_workflow.md)**: Finding the optimal insurance strategy
4. **[Analyzing Results](05_analyzing_results.md)**: Advanced metrics and decision-making

## Quick Tips

- **Start Simple**: Begin with single-layer insurance before exploring complex programs
- **Use Seeds**: Set `np.random.seed()` for reproducible results during testing
- **Experiment**: Try different retention levels to see the trade-offs
- **Think Long-term**: Run simulations for 20-50 years to see ergodic effects

## Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md) for common issues
- Review the [FAQ](../user_guide/faq.rst) for conceptual questions
- Explore the [API Documentation](../api/modules.rst) for detailed function references

## Summary

Congratulations! You've successfully:
- ✅ Installed the Ergodic Insurance Framework
- ✅ Created your first manufacturer model
- ✅ Generated realistic loss scenarios
- ✅ Run simulations with and without insurance
- ✅ Visualized and interpreted the results

You're now ready to explore more advanced features and optimize your insurance strategy!

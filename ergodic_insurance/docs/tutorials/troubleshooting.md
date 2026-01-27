# Troubleshooting Guide

This guide helps you resolve common issues when using the Ergodic Insurance Framework. Each issue includes symptoms, causes, and step-by-step solutions.

## Table of Contents

1. {ref}`installation-issues`
2. {ref}`import-errors`
3. {ref}`simulation-problems`
4. {ref}`performance-issues`
5. {ref}`memory-problems`
6. {ref}`numerical-issues`
7. {ref}`visualization-problems`
8. {ref}`configuration-errors`

(installation-issues)=
## Installation Issues

### Issue: Package won't install with pip

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement ergodic_insurance
```

**Solution:**
```bash
# Install from the local directory
cd ergodic_insurance
pip install -e .

# Or with uv
uv sync
```

### Issue: Dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution:**
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with updated pip
pip install --upgrade pip
pip install -e .
```

### Issue: NumPy/SciPy installation fails

**Symptoms:**
```
ERROR: Failed building wheel for numpy
```

**Solution:**
```bash
# Install pre-built wheels
pip install --only-binary :all: numpy scipy

# On Mac with M1/M2:
pip install numpy scipy --platform macosx_11_0_arm64
```

(import-errors)=
## Import Errors

### Issue: ModuleNotFoundError

**Symptoms:**
```python
>>> from ergodic_insurance.manufacturer import Manufacturer
ModuleNotFoundError: No module named 'ergodic_insurance'
```

**Solution:**
```python
# Check installation
import sys
print(sys.path)

# Add to path if needed
import sys
sys.path.append('/path/to/ergodic_insurance')

# Or reinstall
pip install -e /path/to/ergodic_insurance
```

### Issue: ImportError for specific modules

**Symptoms:**
```python
ImportError: cannot import name 'ClaimGenerator' from 'ergodic_insurance.src'
```

**Solution:**
```python
# Use correct import path
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

# Check available modules
import ergodic_insurance.src
print(dir(ergodic_insurance.src))
```

(simulation-problems)=
## Simulation Problems

### Issue: All simulations result in bankruptcy

**Symptoms:**
- Every simulation path shows ruin
- Final wealth is always negative
- Survival rate is 0%

**Diagnosis and Solution:**
```python
# Check loss severity relative to assets
manufacturer = Manufacturer(initial_assets=10_000_000)
claim_generator = ManufacturingLossGenerator.create_simple(
    frequency=5,
    severity_mu=10.0,
    severity_sigma=1.5
)

# Diagnose the problem
sample_losses = [claim_generator.generate_claims(1) for _ in range(100)]
avg_annual_loss = np.mean([sum(losses) for losses in sample_losses])
loss_ratio = avg_annual_loss / manufacturer.initial_assets

print(f"Average annual loss: ${avg_annual_loss:,.0f}")
print(f"Loss ratio: {loss_ratio:.1%}")

if loss_ratio > 0.1:
    print("⚠️ Losses too high relative to assets!")

    # Solutions:
    # 1. Reduce loss severity
    claim_generator = ManufacturingLossGenerator.create_simple(
        frequency=5,
        severity_mu=9.0,  # Reduced
        severity_sigma=1.0  # Reduced
    )

    # 2. Increase insurance coverage
    retention = 250_000  # Lower retention
    limit = 15_000_000   # Higher limit

    # 3. Increase initial assets
    manufacturer = Manufacturer(initial_assets=20_000_000)
```

### Issue: Results vary wildly between runs

**Symptoms:**
- Different results each time
- Can't reproduce analysis
- Inconsistent optimization results

**Solution:**
```python
# Always set random seed for reproducibility
import numpy as np
np.random.seed(42)

# Or pass seed to functions
results = mc_analyzer.run_simulations(
    n_simulations=1000,
    n_years=20,
    seed=42  # Fixed seed
)

# For multiple runs with different seeds
seeds = [42, 100, 200, 300, 400]
all_results = []
for seed in seeds:
    result = mc_analyzer.run_simulations(seed=seed)
    all_results.append(result)

# Average across seeds for stability
mean_growth = np.mean([r['mean_growth_rate'] for r in all_results])
```

### Issue: Negative growth rates for all strategies

**Symptoms:**
- All insurance strategies show negative growth
- No profitable configurations found

**Solution:**
```python
# Check base profitability without losses
manufacturer = Manufacturer(
    initial_assets=10_000_000,
    asset_turnover=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25
)

# Calculate base ROE
revenue = manufacturer.initial_assets * manufacturer.asset_turnover
operating_income = revenue * manufacturer.base_operating_margin
net_income = operating_income * (1 - manufacturer.tax_rate)
base_roe = net_income / manufacturer.initial_assets

print(f"Base ROE (no losses): {base_roe:.1%}")

if base_roe < 0.05:
    print("⚠️ Base profitability too low!")

    # Improve profitability
    manufacturer = Manufacturer(
        initial_assets=10_000_000,
        asset_turnover=1.2,      # Increased
        base_operating_margin=0.10,   # Increased
        tax_rate=0.21           # Reduced
    )
```

(performance-issues)=
## Performance Issues

### Issue: Simulations run too slowly

**Symptoms:**
- Single simulation takes minutes
- Monte Carlo takes hours
- Optimization never completes

**Solution:**
```python
# 1. Reduce simulation parameters for testing
quick_test = mc_analyzer.run_simulations(
    n_simulations=100,   # Reduced from 1000
    n_years=10,         # Reduced from 20
    seed=42
)

# 2. Use parallel processing
from ergodic_insurance.parallel_executor import ParallelExecutor

executor = ParallelExecutor(n_workers=4)
results = executor.run_parallel_monte_carlo(
    manufacturer=manufacturer,
    claim_generator=claim_generator,
    n_simulations=10000,
    n_years=20
)

# 3. Profile code to find bottlenecks
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run simulation
results = mc_analyzer.run_simulations(n_simulations=100)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 time consumers
```

### Issue: Optimization takes too long

**Symptoms:**
- Grid search never finishes
- Optimization runs for hours

**Solution:**
```python
# 1. Use coarser grid for initial search
# Instead of:
retentions = np.linspace(100_000, 3_000_000, 30)  # 30 points
limits = np.linspace(1_000_000, 20_000_000, 20)   # 20 points

# Use:
retentions = np.linspace(100_000, 3_000_000, 8)   # 8 points
limits = np.linspace(1_000_000, 20_000_000, 5)    # 5 points

# 2. Use adaptive refinement
# Start coarse, then refine around optimum
coarse_optimum = optimize_coarse_grid()
fine_optimum = optimize_fine_grid_around(coarse_optimum)

# 3. Use gradient-based optimization for continuous parameters
from scipy.optimize import minimize

result = minimize(
    objective_function,
    x0=initial_guess,
    method='L-BFGS-B',  # Fast for smooth problems
    bounds=bounds
)
```

(memory-problems)=
## Memory Problems

### Issue: Out of memory errors

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# 1. Process in batches
def run_large_simulation(n_total, batch_size=1000):
    all_results = []
    n_batches = n_total // batch_size

    for i in range(n_batches):
        batch_results = mc_analyzer.run_simulations(
            n_simulations=batch_size,
            seed=i
        )
        # Extract only needed metrics
        all_results.append({
            'mean_growth': np.mean(batch_results['growth_rates']),
            'survival': batch_results['survival_rate']
        })
        # Clear batch data
        del batch_results

    return all_results

# 2. Use memory-efficient storage
from ergodic_insurance.trajectory_storage import TrajectoryStorage

storage = TrajectoryStorage(
    storage_type='disk',  # Use disk instead of memory
    chunk_size=100
)

# 3. Reduce trajectory storage
# Instead of storing full trajectories
results = mc_analyzer.run_simulations(
    store_trajectories=False,  # Only store summary statistics
    n_simulations=10000
)
```

### Issue: Jupyter notebook kernel crashes

**Symptoms:**
- Kernel dies during large simulations
- "Kernel appears to have died" message

**Solution:**
```python
# 1. Increase Jupyter memory limits
# In jupyter_notebook_config.py:
# c.NotebookApp.max_buffer_size = 10000000000

# 2. Clear variables periodically
import gc

# After processing
del large_array
gc.collect()

# 3. Use generators instead of lists
def generate_simulations(n):
    for i in range(n):
        yield run_single_simulation(seed=i)

# Process one at a time
for sim_result in generate_simulations(10000):
    process(sim_result)
```

(numerical-issues)=
## Numerical Issues

### Issue: NaN or Inf values in results

**Symptoms:**
```python
>>> print(results['growth_rate'])
nan
```

**Solution:**
```python
# 1. Check for division by zero
def safe_growth_rate(final_wealth, initial_wealth, n_years):
    if initial_wealth <= 0 or final_wealth <= 0:
        return np.nan

    return (final_wealth / initial_wealth) ** (1/n_years) - 1

# 2. Check for extreme values
def validate_inputs(manufacturer, claim_generator):
    assert manufacturer.initial_assets > 0
    assert 0 < manufacturer.base_operating_margin < 1
    assert 0 < manufacturer.tax_rate < 1
    assert claim_generator.frequency >= 0
    assert claim_generator.severity_sigma > 0

# 3. Add numerical stability
def stable_log_return(final, initial):
    ratio = final / initial
    # Clip to prevent extreme values
    ratio = np.clip(ratio, 1e-10, 1e10)
    return np.log(ratio)

# 4. Debug NaN sources
def debug_nan(results):
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            nan_count = np.sum(np.isnan(value))
            if nan_count > 0:
                print(f"NaN found in {key}: {nan_count} values")
                # Find first NaN
                nan_idx = np.where(np.isnan(value))[0][0]
                print(f"  First NaN at index {nan_idx}")
```

### Issue: Overflow in exponential calculations

**Symptoms:**
```
RuntimeWarning: overflow encountered in exp
```

**Solution:**
```python
# Use log-space calculations
def safe_lognormal_sample(mu, sigma, size):
    # Instead of np.exp(normal)
    # Use built-in lognormal
    return np.random.lognormal(mu, sigma, size)

# Clip extreme values
def clip_losses(losses, max_loss):
    return np.minimum(losses, max_loss)

# Use appropriate data types
large_values = np.array([1e15, 1e16], dtype=np.float64)
```

(visualization-problems)=
## Visualization Problems

### Issue: Plots don't display in Jupyter

**Symptoms:**
- No plot output
- `<Figure size 640x480 with 1 Axes>` message only

**Solution:**
```python
# 1. Use inline backend
%matplotlib inline
import matplotlib.pyplot as plt

# 2. Explicitly show plots
plt.plot(data)
plt.show()  # Don't forget this!

# 3. Check backend
import matplotlib
print(matplotlib.get_backend())
# Set if needed
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Issue: Overlapping labels or text

**Symptoms:**
- X-axis labels overlap
- Legend covers data
- Title cut off

**Solution:**
```python
# 1. Adjust layout
plt.tight_layout()

# 2. Rotate labels
plt.xticks(rotation=45, ha='right')

# 3. Adjust figure size
plt.figure(figsize=(12, 6))  # Wider figure

# 4. Move legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 5. Adjust margins
plt.subplots_adjust(bottom=0.15, right=0.85)
```

(configuration-errors)=
## Configuration Errors

### Issue: YAML configuration won't load

**Symptoms:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution:**
```yaml
# Check YAML syntax
# BAD:
parameter: value
  nested: value  # Wrong indentation

# GOOD:
parameter: value
nested: value    # Correct indentation

# Or use online YAML validator
```

### Issue: Pydantic validation errors

**Symptoms:**
```
ValidationError: 1 validation error for ManufacturerConfig
```

**Solution:**
```python
# 1. Check required fields
from ergodic_insurance.config_v2 import ManufacturerConfig

# See what's required
print(ManufacturerConfig.schema())

# 2. Provide all required fields
config = ManufacturerConfig(
    initial_assets=10_000_000,  # Required
    asset_turnover=1.0,          # Has default
    base_operating_margin=0.08,       # Has default
    # ... other fields
)

# 3. Check value constraints
# E.g., base_operating_margin must be between 0 and 1
```

## Common Parameter Issues

### Issue: Unrealistic parameter combinations

**Problem:** Setting parameters that don't make business sense

**Guidelines:**
```python
# Reasonable parameter ranges
reasonable_ranges = {
    'asset_turnover': (0.5, 3.0),      # Industry dependent
    'base_operating_margin': (0.02, 0.25),   # 2-25%
    'tax_rate': (0.15, 0.35),          # 15-35%
    'retention_ratio': (0.01, 0.20),    # 1-20% of assets
    'base_premium_rate': (0.01, 0.05),       # 1-5% of limit
    'loss_frequency': (0.1, 20),        # Per year
    'loss_severity_mu': (8, 14),        # Log scale
    'loss_severity_sigma': (0.5, 2.5)   # Log scale
}

def validate_parameters(params):
    for param, value in params.items():
        if param in reasonable_ranges:
            min_val, max_val = reasonable_ranges[param]
            if not min_val <= value <= max_val:
                print(f"⚠️ {param}={value} outside typical range [{min_val}, {max_val}]")
```

## Getting Additional Help

If you encounter issues not covered here:

1. **Check the test suite**: Look at test files for usage examples
2. **Review notebooks**: The example notebooks show working code
3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **File an issue**: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues

## Performance Optimization Tips

### For Large Simulations

```python
# Optimal settings for different scales
configs = {
    'quick_test': {
        'n_simulations': 100,
        'n_years': 10,
        'n_workers': 1
    },
    'standard': {
        'n_simulations': 1000,
        'n_years': 20,
        'n_workers': 4
    },
    'comprehensive': {
        'n_simulations': 10000,
        'n_years': 50,
        'n_workers': 8
    }
}
```

### Memory-Efficient Patterns

```python
# Good: Generator pattern
def process_simulations():
    for i in range(10000):
        result = run_simulation(seed=i)
        yield extract_metrics(result)

# Bad: Loading all into memory
results = [run_simulation(seed=i) for i in range(10000)]
```

Remember: Start small, validate results, then scale up!

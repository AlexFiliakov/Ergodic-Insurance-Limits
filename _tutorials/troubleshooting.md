---
layout: default
title: Troubleshooting
---

# Troubleshooting Guide

Common issues and solutions when using the Ergodic Insurance Framework.

## Installation Issues

### Python Version Errors

**Problem**: `ERROR: This package requires Python 3.12 or higher`

**Solution**:
```bash
# Check your Python version
python --version

# Install Python 3.12+ if needed
# On macOS with Homebrew:
brew install python@3.12

# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.12

# On Windows:
# Download from python.org
```

### Package Installation Failures

**Problem**: `ModuleNotFoundError: No module named 'ergodic_insurance'`

**Solution**:
```bash
# Ensure you're in the project directory
cd Ergodic-Insurance-Limits

# Install in development mode
pip install -e .

# Or with uv:
uv sync
```

### Dependency Conflicts

**Problem**: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Solution**:
```bash
# Create a fresh virtual environment
python -m venv venv_fresh
source venv_fresh/bin/activate  # Windows: venv_fresh\Scripts\activate

# Install with clean dependencies
pip install --upgrade pip
pip install -e .
```

## Runtime Errors

### Memory Issues

**Problem**: `MemoryError` during large simulations

**Solutions**:

1. **Reduce simulation size**:
```python
# Instead of:
config = SimulationConfig(num_simulations=100_000)

# Use:
config = SimulationConfig(num_simulations=10_000)
```

2. **Use batch processing**:
```python
from ergodic_insurance.src.batch_processor import BatchProcessor

processor = BatchProcessor(batch_size=1000)
results = processor.process_simulations(
    total_simulations=100_000,
    simulation_fn=run_simulation
)
```

3. **Enable memory-efficient storage**:
```python
from ergodic_insurance.src.trajectory_storage import TrajectoryStorage

storage = TrajectoryStorage(
    use_compression=True,
    chunk_size=1000
)
```

### Slow Performance

**Problem**: Simulations taking too long

**Solutions**:

1. **Use parallel processing**:
```python
from ergodic_insurance.src.parallel_executor import ParallelExecutor

executor = ParallelExecutor(n_workers=8)  # Use multiple cores
results = executor.run_parallel_simulations(config)
```

2. **Profile your code**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your slow code here
results = run_simulation(config)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 time consumers
```

3. **Use caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param):
    # Expensive computation
    return result
```

### Numerical Instabilities

**Problem**: `RuntimeWarning: overflow encountered in exp`

**Solution**:
```python
import numpy as np

# Use log-space calculations
# Instead of:
result = np.exp(large_number)

# Use:
log_result = large_number  # Keep in log space
# Only exponentiate when necessary and safe
if log_result < 700:  # exp(700) is near float64 max
    result = np.exp(log_result)
else:
    result = np.inf
```

## Configuration Issues

### Invalid Parameters

**Problem**: `ValidationError: Operating margin must be between 0 and 1`

**Solution**:
```python
# Check parameter ranges
from ergodic_insurance.src.config_v2 import ManufacturerConfig

# Validate before use
config = ManufacturerConfig(
    starting_assets=10_000_000,
    operating_margin=0.08,  # 8%, not 8
    asset_turnover=1.0
)

# Use the validator
config.validate()
```

### File Not Found

**Problem**: `FileNotFoundError: Configuration file not found`

**Solution**:
```python
import os
from pathlib import Path

# Use absolute paths
config_path = Path(__file__).parent / "config" / "parameters.yaml"

# Check file exists
if not config_path.exists():
    print(f"Config file not found at: {config_path}")
    print(f"Current directory: {os.getcwd()}")
```

## Simulation Issues

### Convergence Problems

**Problem**: Results vary significantly between runs

**Solutions**:

1. **Increase sample size**:
```python
# Check convergence
from ergodic_insurance.src.convergence import check_convergence

results = []
for n in [1000, 5000, 10000, 50000]:
    result = run_simulation(n_simulations=n)
    results.append(result)

    if check_convergence(results):
        print(f"Converged at {n} simulations")
        break
```

2. **Use fixed random seeds**:
```python
import numpy as np

np.random.seed(42)  # For reproducibility
results = run_simulation(config)
```

### Unrealistic Results

**Problem**: Growth rates seem too high or negative

**Solutions**:

1. **Verify parameters**:
```python
# Sanity check parameters
def validate_parameters(config):
    checks = {
        "margin_reasonable": 0 < config.operating_margin < 0.3,
        "volatility_reasonable": 0 < config.volatility < 0.5,
        "premium_reasonable": config.premium < config.limit * 0.1
    }

    for check, passed in checks.items():
        if not passed:
            print(f"Warning: {check} failed")
```

2. **Check units consistency**:
```python
# Ensure all monetary values in same units
assets = 10_000_000  # $10M
losses = 2_000_000   # $2M (not 2000 for $2M!)
premium = 150_000    # $150K
```

## Optimization Issues

### Optimization Not Converging

**Problem**: Optimizer runs forever or fails to find solution

**Solutions**:

1. **Adjust tolerance**:
```python
from scipy.optimize import minimize

result = minimize(
    objective_function,
    initial_guess,
    method='L-BFGS-B',
    options={
        'ftol': 1e-6,  # Function tolerance
        'gtol': 1e-6,  # Gradient tolerance
        'maxiter': 1000  # Maximum iterations
    }
)
```

2. **Use bounded optimization**:
```python
# Define reasonable bounds
bounds = [
    (100_000, 10_000_000),  # Retention bounds
    (1_000_000, 100_000_000),  # Limit bounds
]

result = minimize(
    objective_function,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds
)
```

### Local Optima

**Problem**: Optimizer stuck in local optimum

**Solution**:
```python
from scipy.optimize import differential_evolution

# Use global optimizer
result = differential_evolution(
    objective_function,
    bounds=bounds,
    seed=42,
    workers=-1  # Use all CPU cores
)
```

## Data Issues

### Corrupted Cache

**Problem**: `PickleError` or strange cached results

**Solution**:
```bash
# Clear cache directory
rm -rf ./cache/*
# Or
rm -rf ergodic_insurance/notebooks/cache/*
```

### Large Output Files

**Problem**: Result files too large

**Solution**:
```python
# Use compression
import pandas as pd

# Save with compression
df.to_parquet('results.parquet', compression='snappy')

# Or for pickle
import pickle
import gzip

with gzip.open('results.pkl.gz', 'wb') as f:
    pickle.dump(results, f)
```

## Visualization Issues

### Plots Not Showing

**Problem**: Matplotlib plots not displaying

**Solution**:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# Ensure plot shows
plt.show(block=True)
```

### Plot Quality Issues

**Problem**: Plots look pixelated or wrong size

**Solution**:
```python
import matplotlib.pyplot as plt

# Set high DPI
plt.figure(figsize=(12, 8), dpi=100)

# Save high-quality
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

## Testing Issues

### Test Failures

**Problem**: Tests failing after changes

**Solution**:
```bash
# Run specific test with verbose output
pytest tests/test_simulation.py -v

# Run with coverage
pytest --cov=ergodic_insurance --cov-report=html

# Debug specific test
pytest tests/test_simulation.py::test_specific -vv --pdb
```

## Getting Help

If your issue isn't covered here:

1. **Check the documentation**:
   - API docs: `ergodic_insurance/docs/`
   - Examples: `ergodic_insurance/examples/`

2. **Search existing issues**:
   ```bash
   # Search GitHub issues
   https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues
   ```

3. **Create a minimal example**:
   ```python
   # Minimal reproducible example
   from ergodic_insurance import X

   # Simplest code that shows the problem
   result = X(minimal_params)
   print(f"Expected: ..., Got: {result}")
   ```

4. **File an issue** with:
   - Python version
   - Package versions (`pip freeze`)
   - Full error traceback
   - Minimal reproducible example

## Common Warnings

### Harmless Warnings

These warnings can usually be ignored:

- `FutureWarning: pandas.Int64Index is deprecated`
- `RuntimeWarning: divide by zero encountered in log` (if handled)
- `DeprecationWarning` from dependencies

### Important Warnings

These need attention:

- `RuntimeWarning: overflow encountered` - Check numerical stability
- `MemoryWarning` - Reduce problem size
- `ConvergenceWarning` - Increase iterations or adjust parameters

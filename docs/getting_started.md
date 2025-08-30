---
layout: default
title: Getting Started - Ergodic Insurance Framework
description: Installation and setup guide for the ergodic insurance framework
mathjax: true
---

# Getting Started

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.12 or higher
- **Memory**: 4GB RAM
- **Storage**: 500MB free space
- **Processor**: Dual-core 2.0GHz+

### Recommended Requirements
- **Memory**: 8GB+ RAM (for large simulations)
- **Processor**: Quad-core or better (for parallel processing)
- **Storage**: 2GB+ (for result caching)

## Installation

### Method 1: Install from PyPI (Recommended)

```bash
# Install the latest stable version
pip install ergodic-insurance

# Or install with optional dependencies
pip install ergodic-insurance[viz]  # Include visualization tools
pip install ergodic-insurance[dev]  # Include development tools
pip install ergodic-insurance[all]  # Install everything
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
cd Ergodic-Insurance-Limits

# Install in development mode
pip install -e .

# Or install with dependencies
pip install -e ".[all]"
```

### Method 3: Using UV Package Manager

```bash
# Install uv if not already installed
pip install uv

# Install using uv
uv pip install ergodic-insurance

# Or sync from lock file
uv sync
```

### Method 4: Docker Container

```bash
# Pull the Docker image
docker pull ghcr.io/alexfiliakov/ergodic-insurance:latest

# Run interactive session
docker run -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  ghcr.io/alexfiliakov/ergodic-insurance:latest

# Or run Jupyter notebook
docker run -it --rm \
  -p 8888:8888 \
  ghcr.io/alexfiliakov/ergodic-insurance:latest \
  jupyter notebook --ip=0.0.0.0 --allow-root
```

## Verify Installation

### Basic Verification

```python
# Test import
import ergodic_insurance
print(f"Version: {ergodic_insurance.__version__}")

# Test basic functionality
from ergodic_insurance.src import Manufacturer

manufacturer = Manufacturer(starting_assets=10_000_000)
print(f"Manufacturer created with ${manufacturer.starting_assets:,.0f} assets")
```

### Run Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ergodic_insurance --cov-report=html

# Run specific test module
pytest tests/test_manufacturer.py

# Run with verbose output
pytest -v
```

### Check Dependencies

```python
# Check all required packages are installed
import pkg_resources

required = [
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'scipy>=1.10.0',
    'matplotlib>=3.7.0',
    'pydantic>=2.0.0',
    'pytest>=7.0.0'
]

for package in required:
    try:
        pkg_resources.require(package)
        print(f"✓ {package}")
    except:
        print(f"✗ {package} - Please install")
```

## Project Structure

```
Ergodic-Insurance-Limits/
├── ergodic_insurance/
│   ├── src/               # Source code
│   │   ├── manufacturer.py
│   │   ├── insurance.py
│   │   ├── simulation.py
│   │   └── ...
│   ├── tests/             # Test suite
│   ├── notebooks/         # Jupyter notebooks
│   ├── examples/          # Example scripts
│   └── data/              # Configuration files
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── README.md             # Project overview
```

## Configuration

### Basic Configuration

```python
# Load default configuration
from ergodic_insurance.src.config_manager import ConfigManager

config = ConfigManager.load_config()
print(config)

# Or load custom configuration
config = ConfigManager.load_config("my_config.yaml")
```

### Configuration File Format

```yaml
# config.yaml
manufacturer:
  starting_assets: 10_000_000
  base_revenue: 15_000_000
  operating_margin: 0.08
  volatility: 0.15

insurance:
  layers:
    - name: "Primary"
      limit: 5_000_000
      attachment: 0
      premium_rate: 0.015
    - name: "Excess"
      limit: 20_000_000
      attachment: 5_000_000
      premium_rate: 0.008

simulation:
  n_trajectories: 1000
  time_horizon: 20
  random_seed: 42
```

### Environment Variables

```bash
# Set environment variables for configuration
export ERGODIC_CONFIG_PATH=/path/to/config
export ERGODIC_CACHE_DIR=/path/to/cache
export ERGODIC_LOG_LEVEL=INFO
export ERGODIC_PARALLEL_WORKERS=4
```

## Quick Start Examples

### Example 1: Basic Analysis

```python
from ergodic_insurance.src import quick_analysis

# Run a quick analysis with defaults
results = quick_analysis(
    assets=10_000_000,
    revenue=15_000_000,
    insurance_limit=5_000_000,
    premium_rate=0.015
)

print(f"Growth rate: {results['growth_rate']:.2%}")
print(f"Ruin probability: {results['ruin_probability']:.2%}")
```

### Example 2: Jupyter Notebook

```python
# In Jupyter notebook
%load_ext ergodic_insurance.magic

# Use magic commands
%%ergodic_simulate
assets: 10_000_000
revenue: 15_000_000
insurance_limit: 5_000_000
simulations: 1000
```

### Example 3: Command Line Interface

```bash
# Run analysis from command line
ergodic-insurance analyze \
  --assets 10000000 \
  --revenue 15000000 \
  --insurance-limit 5000000 \
  --output results.json

# Generate report
ergodic-insurance report results.json --format html

# Run optimization
ergodic-insurance optimize \
  --config my_business.yaml \
  --output optimal_insurance.yaml
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Error
```
ImportError: No module named 'ergodic_insurance'
```
**Solution**: Ensure the package is installed and Python path is correct:
```bash
pip show ergodic-insurance
python -c "import sys; print(sys.path)"
```

#### Issue 2: Memory Error
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce simulation size or use batch processing:
```python
# Instead of large single simulation
results = simulate(n_trajectories=100000)

# Use batch processing
results = simulate_batched(n_trajectories=100000, batch_size=10000)
```

#### Issue 3: Slow Performance
**Solution**: Enable parallel processing:
```python
from ergodic_insurance.src import ParallelMonteCarloEngine

engine = ParallelMonteCarloEngine(n_workers=4)
results = engine.run()
```

#### Issue 4: Numerical Instability
```
RuntimeWarning: overflow encountered in exp
```
**Solution**: Use log-space calculations:
```python
# Enable stable numerics
from ergodic_insurance.src import enable_stable_numerics
enable_stable_numerics()
```

## Getting Help

### Documentation
- [Full Documentation](/Ergodic-Insurance-Limits/)
- [API Reference](/Ergodic-Insurance-Limits/api/)
- [Tutorials](/Ergodic-Insurance-Limits/tutorials/getting_started)
- [FAQ](/Ergodic-Insurance-Limits/docs/user_guide/faq)

### Community
- [GitHub Issues](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues)
- [Discussions](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/ergodic-insurance)

### Professional Support
- Email: support@ergodic-insurance.com
- Consulting: [Request Consultation](/contact)

## Next Steps

Now that you have the framework installed:

1. **Follow the Tutorial**: [Getting Started Tutorial](/Ergodic-Insurance-Limits/tutorials/getting_started)
2. **Run Examples**: [Code Examples](/Ergodic-Insurance-Limits/docs/examples)
3. **Read Theory**: [Ergodic Economics Background](/Ergodic-Insurance-Limits/theory/01_ergodic_economics)
4. **Optimize Your Insurance**: [Optimization Guide](/Ergodic-Insurance-Limits/tutorials/optimization_workflow)

## Version History

### v1.0.0 (Current)
- Initial stable release
- Core simulation engine
- Basic optimization
- Documentation complete

### Roadmap
- v1.1.0: GPU acceleration support
- v1.2.0: Web-based interface
- v1.3.0: Real-time monitoring
- v2.0.0: Machine learning integration

---

[← Back to Home](/Ergodic-Insurance-Limits/) | [Continue to Tutorials →](/Ergodic-Insurance-Limits/tutorials/getting_started)

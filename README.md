# Ergodic Insurance Limits

![Repo Banner](assets/repo_banner_small.png)

This is a brief research model of a widget manufacturing company to determine what limit of insurance they need to optimize long-term profitability.

[![Documentation Status](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/actions/workflows/docs.yml/badge.svg)](https://alexfiliakov.github.io/Ergodic-Insurance-Limits/)

## Introduction - Why Do Companies Buy Insurance?

### Ergodic theory transforms insurance optimization fundamentally

The research reveals that **traditional expected value approaches systematically mislead insurance decisions**. Ole Peters' ergodic economics framework demonstrates that insurance creates win-win scenarios when analyzed through time averages rather than ensemble averages. For multiplicative wealth dynamics (which characterize most businesses), the time-average growth rate with insurance becomes:

$g = \lim_{T\to\infty}{\frac{1}{T}\ln{\frac{x(T)}{x(0)}}}$

This framework resolves the fundamental insurance puzzle: while insurance appears zero-sum in expected value terms, both parties benefit when optimizing time-average growth rates. For our widget manufacturing model with $10M starting assets, the hypothesis is that **optimal insurance premiums can exceed expected losses by 200-500%** while still enhancing long-term growth.
This framework resolves the fundamental insurance puzzle: while insurance appears zero-sum in expected value terms, both parties benefit when optimizing time-average growth rates. For our widget manufacturing model with $10M starting assets, the hypothesis is that **optimal insurance premiums can exceed expected losses by 200-500%** while still enhancing long-term growth.

### Value Proposition

![Ergodic Distinction Between Averages](assets/ergodic_distinction.png)

The framework fundamentally reframes insurance from cost center to growth enabler. By optimizing time-average growth rates rather than expected values, widget manufacturers can achieve **30-50% better long-term performance** while maintaining acceptable ruin probabilities. The key insight: **maximizing ergodic growth rates naturally balances profitability with survival**, eliminating the need for arbitrary risk preferences or utility functions.

This comprehensive framework provides the mathematical rigor, practical parameters, and implementation roadmap necessary for successful insurance optimization in widget manufacturing, with the ergodic approach offering genuinely novel insights that challenge conventional risk management wisdom.

## Key Features

### Financial Modeling
- **Widget manufacturer model** with comprehensive balance sheet management
- **Stochastic processes** including geometric Brownian motion, lognormal volatility, and mean-reversion
- **Insurance claim processing** with multi-year payment schedules
- **Collateral management** for letter of credit requirements

### Configuration Management (v2.0)
- **3-tier configuration architecture** with profiles, modules, and presets
- **ConfigManager** with profile inheritance and module composition
- **Pydantic v2 validation** with comprehensive type safety
- **Runtime overrides** for flexible parameter experimentation
- **Preset libraries** for common market conditions and risk scenarios
- **Full backward compatibility** with legacy ConfigLoader

### Documentation & Testing
- **Comprehensive Google-style docstrings** throughout the codebase
- **Sphinx documentation system** for professional API reference
- **90% test coverage** with pytest framework
- **Type safety** enforced with mypy static analysis

### Analysis Tools
- **Jupyter notebooks** for interactive exploration and visualization
- **Demo scripts** showing stochastic vs deterministic comparisons
- **Performance metrics** including ROE, risk of ruin, and time-average growth rates

## Results

- [Ergodic Insurance Part 1: From Cost Center to Growth Engine: When N=1](https://medium.com/@alexfiliakov/ergodic-insurance-part-1-from-cost-center-to-growth-engine-when-n-1-52c17b048a94)

## Exploratory Notebooks

- [Growth Dynamics and Asset Fluctuations](ergodic_insurance/notebooks/03_growth_dynamics.ipynb)
- [Ergodic Insurance Advantage Demonstration](ergodic_insurance/notebooks/04_ergodic_demo.ipynb)
- [Risk Metrics Suite for Tail Risk Analysis](ergodic_insurance/notebooks/05_risk_metrics.ipynb)

## Configuration System

The project uses a modern 3-tier configuration architecture for maximum flexibility:

### Quick Start
```python
from ergodic_insurance.src.config_manager import ConfigManager

# Load a configuration profile
manager = ConfigManager()
config = manager.load_profile("default")  # or "conservative", "aggressive"

# Access configuration values
print(f"Operating margin: {config.manufacturer.operating_margin:.1%}")
print(f"Growth rate: {config.growth.annual_growth_rate:.1%}")
```

### Advanced Usage
```python
# Override parameters at runtime
config = manager.load_profile(
    "conservative",
    manufacturer={"operating_margin": 0.12},
    growth={"annual_growth_rate": 0.08}
)

# Apply preset templates
config = manager.load_profile(
    "default",
    presets=["hard_market", "high_volatility"]
)

# Load specific modules only
config = manager.load_profile(
    "default",
    modules=["insurance", "stochastic"]
)
```

See [migration_guide.md](ergodic_insurance/docs/migration_guide.md) for migrating from the legacy system.

## Installation

### Prerequisites
- Python 3.12 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
cd Ergodic-Insurance-Limits
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

3. Install pre-commit hooks for code quality:
```bash
pre-commit install
```

## Project Structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the complete directory tree.

### Key Directories

```
Ergodic Insurance Limits/
├── ergodic_insurance/           # Main Python package
│   ├── src/                    # Core source code (26 modules)
│   │   ├── config_*.py        # Configuration system v2.0
│   │   ├── manufacturer.py     # Core financial model
│   │   ├── insurance_*.py     # Insurance modules
│   │   ├── *_optimizer.py     # Optimization algorithms
│   │   └── visualization.py   # Plotting utilities
│   ├── tests/                  # Test suite (100% coverage, 30+ test files)
│   ├── notebooks/              # Jupyter notebooks (14 analysis notebooks)
│   ├── examples/               # Example scripts and demos
│   ├── data/                   # Configuration files
│   │   ├── parameters/         # Legacy YAML parameters (deprecated)
│   │   └── config/             # New 3-tier configuration
│   │       ├── profiles/       # Complete configurations
│   │       ├── modules/        # Reusable components
│   │       └── presets/        # Quick-apply templates
│   ├── docs/                   # Sphinx documentation
│   │   ├── api/               # API reference
│   │   ├── architecture/      # Architecture diagrams
│   │   └── user_guide/        # Business user guide
│   └── checkpoints/           # Simulation checkpoints
├── simone/                     # TypeScript simulation
├── results/                    # Reports and outputs
├── assets/                     # Images and media
├── pyproject.toml             # Python configuration
├── setup.py                   # Package setup
├── README.md                  # This file
├── CLAUDE.md                  # Development instructions
└── LICENSE                    # MIT License
```

## Development

### Code Quality Tools

This project uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **mypy**: Static type checking
- **pylint**: Code linting
- **pytest-cov**: Test coverage reporting (minimum: 80%, achieved: 100%)

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest ergodic_insurance/tests/test_manufacturer.py

# Run with coverage report
pytest --cov=ergodic_insurance --cov-report=html
```

### Pre-commit Hooks

Pre-commit hooks run automatically on commit. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Type Checking

```bash
# Run mypy
mypy ergodic_insurance

# Run with specific file
mypy ergodic_insurance/src/manufacturer.py
```

### Code Formatting

```bash
# Format with black
black ergodic_insurance

# Sort imports with isort
isort ergodic_insurance

# Ergodic Insurance Limits

This is a brief research model of a widget manufacturing company to determine what limit of insurance they need to optimize long-term profitability.

## Introduction - Why Do Companies Buy Insurance?

### Ergodic theory transforms insurance optimization fundamentally

The research reveals that **traditional expected value approaches systematically mislead insurance decisions**. Ole Peters' ergodic economics framework demonstrates that insurance creates win-win scenarios when analyzed through time averages rather than ensemble averages. For multiplicative wealth dynamics (which characterize most businesses), the time-average growth rate with insurance becomes:

$g = \lim_{T\to\infty}{\frac{1}{T}\ln{\frac{x(T)}{x(0)}}}$

This framework resolves the fundamental insurance puzzle: while insurance appears zero-sum in expected value terms, both parties benefit when optimizing time-average growth rates. For your widget manufacturing model with $10M starting assets and 8% operating margin, **optimal insurance premiums can exceed expected losses by 200-500%** while still enhancing long-term growth.

### Value Proposition

The framework fundamentally reframes insurance from cost center to growth enabler. By optimizing time-average growth rates rather than expected values, widget manufacturers can achieve **30-50% better long-term performance** while maintaining acceptable ruin probabilities. The key insight: **maximizing ergodic growth rates naturally balances profitability with survival**, eliminating the need for arbitrary risk preferences or utility functions.

This comprehensive framework provides the mathematical rigor, practical parameters, and implementation roadmap necessary for successful insurance optimization in widget manufacturing, with the ergodic approach offering genuinely novel insights that challenge conventional risk management wisdom.

## Key Features

### Financial Modeling
- **Widget manufacturer model** with comprehensive balance sheet management
- **Stochastic processes** including GBM, lognormal volatility, and mean-reversion
- **Insurance claim processing** with multi-year payment schedules
- **Collateral management** for letter of credit requirements

### Configuration Management
- **Pydantic-based configuration** with full validation and type safety
- **YAML parameter files** for different scenarios (baseline, conservative, optimistic, stochastic)
- **Flexible override system** for parameter experimentation

### Documentation & Testing
- **Comprehensive Google-style docstrings** throughout the codebase
- **Sphinx documentation system** for professional API reference
- **100% test coverage** with pytest framework
- **Type safety** enforced with mypy static analysis

### Analysis Tools
- **Jupyter notebooks** for interactive exploration and visualization
- **Demo scripts** showing stochastic vs deterministic comparisons
- **Performance metrics** including ROE, risk of ruin, and time-average growth rates

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

## Development

### Code Quality Tools

This project uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **mypy**: Static type checking
- **pylint**: Code linting
- **pytest-cov**: Test coverage reporting (minimum: 80%)

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
```

## Project Structure

```
Ergodic Insurance Limits/
├── ergodic_insurance/           # Main Python package
│   ├── src/                    # Core source code
│   │   ├── __init__.py
│   │   ├── manufacturer.py     # Widget manufacturer financial model
│   │   ├── claim_generator.py  # Insurance claim generation
│   │   ├── config.py           # Configuration management with Pydantic
│   │   ├── config_loader.py    # YAML parameter loading utilities
│   │   ├── stochastic_processes.py # Stochastic modeling (GBM, lognormal, mean-reversion)
│   │   ├── simulation.py       # Main simulation engine
│   │   └── insurance.py        # Insurance optimization algorithms
│   ├── tests/                  # Comprehensive test suite
│   │   ├── test_manufacturer.py
│   │   ├── test_claim_generator.py
│   │   ├── test_config.py
│   │   ├── test_stochastic.py
│   │   └── test_*.py
│   ├── notebooks/              # Jupyter analysis notebooks
│   │   ├── 00_setup_verification.ipynb
│   │   ├── 01_basic_manufacturer.ipynb
│   │   ├── 02_long_term_simulation.ipynb
│   │   └── 03_growth_dynamics.ipynb
│   ├── examples/               # Example scripts and demos
│   │   ├── demo_manufacturer.py
│   │   ├── demo_collateral_management.py
│   │   └── demo_stochastic.py
│   ├── data/                   # Configuration parameters
│   │   └── parameters/
│   │       ├── baseline.yaml    # Standard configuration
│   │       ├── conservative.yaml
│   │       ├── optimistic.yaml
│   │       ├── stochastic.yaml # Stochastic process config
│   │       └── insurance.yaml
│   ├── docs/                   # Sphinx documentation
│   │   ├── conf.py            # Sphinx configuration
│   │   ├── index.rst          # Documentation main page
│   │   ├── api/               # Auto-generated API docs
│   │   └── *.rst              # Documentation files
│   └── pyproject.toml         # Python package configuration
├── simone/                     # TypeScript simulation components
│   ├── src/
│   │   ├── core/simulation.ts
│   │   ├── models/types.ts
│   │   └── utils/statistics.ts
│   ├── tests/
│   │   ├── simulation.test.ts
│   │   └── statistics.test.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── jest.config.js
├── results/                    # Generated reports and blog drafts
│   ├── BLOG_DRAFT_01_ERGODIC_LIMIT_SELECTION.md
│   └── BLOG_OUTLINE_01_ERGODIC_LIMIT_SELECTION.md
├── assets/                     # Images and documentation assets
│   └── debug/                 # Debug visualizations
├── pyproject.toml             # Root Python configuration
├── uv.lock                    # UV package lock file
├── mypy.ini                   # Type checking configuration
└── README.md                  # This file
```

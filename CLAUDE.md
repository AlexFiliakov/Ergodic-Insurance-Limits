# Claude Code Project Instructions - Ergodic Insurance Limits

## Project Overview
This project implements a revolutionary framework for optimizing insurance limits using ergodic (time-average) theory rather than traditional ensemble approaches. The framework demonstrates how insurance transforms from a cost center to a growth enabler when analyzed through time averages, with potential for 30-50% better long-term performance in widget manufacturing scenarios.

## Key Objectives
1. Build a complete simulation framework for ergodic insurance optimization
2. Generate compelling evidence for blog posts demonstrating ergodic advantages
3. Provide actuaries with practical Python tools for insurance decision-making
4. Validate that optimal insurance premiums can exceed expected losses by 200-500% while enhancing growth

## Directory Structure
```
Ergodic Insurance Limits/
├── ergodic_insurance/           # Main Python package
│   ├── src/                    # Core source code
│   │   ├── __init__.py         # Package initialization with comprehensive docs
│   │   ├── manufacturer.py     # Widget manufacturer financial model
│   │   ├── claim_generator.py  # Insurance claim generation with Poisson/lognormal
│   │   ├── claim_development.py # Claim development patterns for cash flow modeling
│   │   ├── config.py           # Pydantic-based configuration management
│   │   ├── config_loader.py    # YAML parameter loading utilities
│   │   ├── stochastic_processes.py # Stochastic modeling (GBM, lognormal, mean-reversion)
│   │   ├── simulation.py       # Main simulation engine
│   │   ├── insurance.py        # Basic insurance optimization algorithms
│   │   ├── insurance_program.py # Enhanced multi-layer insurance programs
│   │   ├── loss_distributions.py # Enhanced loss distributions for manufacturing risks
│   │   ├── monte_carlo.py      # Monte Carlo simulation engine
│   │   └── ergodic_analyzer.py # Ergodic analysis and optimization tools
│   ├── tests/                  # Comprehensive test suite (100% coverage)
│   │   ├── __init__.py
│   │   ├── conftest.py         # Pytest configuration and fixtures
│   │   ├── test_manufacturer.py
│   │   ├── test_claim_generator.py
│   │   ├── test_claim_development.py
│   │   ├── test_config.py
│   │   ├── test_stochastic.py
│   │   ├── test_insurance.py
│   │   ├── test_insurance_program.py
│   │   ├── test_loss_distributions.py
│   │   ├── test_simulation.py
│   │   ├── test_monte_carlo.py
│   │   ├── test_ergodic_analyzer.py
│   │   ├── test_integration.py
│   │   ├── test_manufacturer_methods.py
│   │   └── test_setup.py
│   ├── notebooks/              # Jupyter analysis notebooks
│   │   ├── 00_setup_verification.ipynb
│   │   ├── 01_basic_manufacturer.ipynb
│   │   ├── 02_long_term_simulation.ipynb
│   │   ├── 03_growth_dynamics.ipynb
│   │   └── 04_ergodic_demo.ipynb
│   ├── examples/               # Example scripts and demos
│   │   ├── demo_manufacturer.py
│   │   ├── demo_collateral_management.py
│   │   └── demo_stochastic.py  # Stochastic vs deterministic comparison
│   ├── data/                   # Configuration parameters
│   │   └── parameters/
│   │       ├── baseline.yaml    # Standard configuration
│   │       ├── conservative.yaml
│   │       ├── optimistic.yaml
│   │       ├── stochastic.yaml  # Stochastic process parameters
│   │       ├── insurance.yaml   # Insurance optimization settings
│   │       ├── insurance_market.yaml # Market parameters
│   │       ├── insurance_structures.yaml # Insurance program structures
│   │       ├── loss_distributions.yaml # Loss distribution parameters
│   │       ├── losses.yaml      # Legacy loss parameters
│   │       └── development_patterns.yaml # Claim development patterns
│   ├── docs/                   # Sphinx documentation system
│   │   ├── conf.py            # Sphinx configuration
│   │   ├── index.rst          # Documentation main page
│   │   ├── api/               # Auto-generated API documentation
│   │   │   ├── modules.rst
│   │   │   ├── manufacturer.rst
│   │   │   ├── config.rst
│   │   │   └── *.rst
│   │   ├── getting_started.rst
│   │   ├── theory.rst
│   │   ├── examples.rst
│   │   └── overview.rst
│   ├── htmlcov/                # Test coverage reports
│   ├── pyproject.toml          # Python package configuration
│   ├── pytest.ini             # Pytest configuration
│   ├── requirements.txt        # Python dependencies
│   └── uv.lock                # UV dependency lock file
├── simone/                     # TypeScript simulation components & sprint docs
│   ├── src/                    # TypeScript source
│   │   ├── core/simulation.ts
│   │   ├── models/types.ts
│   │   ├── utils/statistics.ts
│   │   └── index.ts
│   ├── tests/                  # Jest tests
│   │   ├── simulation.test.ts
│   │   └── statistics.test.ts
│   ├── 00_PLAN.md             # Overall project plan
│   ├── SPRINT_01_FOUNDATION.md # Core financial model sprint
│   ├── SPRINT_02_ERGODIC_FRAMEWORK.md # Ergodic theory implementation
│   ├── SPRINT_03_LOSS_MODELING.md # Insurance loss modeling
│   ├── SPRINT_*_*.md          # Additional sprint documentation
│   ├── package.json           # Node dependencies
│   ├── tsconfig.json          # TypeScript config
│   └── jest.config.js         # Jest test config
├── results/                    # Generated reports and blog drafts
│   ├── BLOG_DRAFT_01_ERGODIC_LIMIT_SELECTION.md
│   └── BLOG_OUTLINE_01_ERGODIC_LIMIT_SELECTION.md
├── assets/                     # Images and documentation assets
│   └── debug/                 # Debug visualizations
├── main.py                    # Root Python entry point
├── pyproject.toml             # Root Python configuration
├── uv.lock                    # UV package lock file
├── mypy.ini                   # MyPy type checking configuration
├── CLAUDE.md                  # This file - project instructions
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
└── CC Prompts.md             # Claude Code prompts and development history
```

## Sprint Plan Documents
The project follows a structured 3-sprint development plan:
- `simone/00_PLAN.md` - Overall project plan and architecture
- `simone/SPRINT_01_FOUNDATION.md` - Core financial model implementation
- `simone/SPRINT_02_ERGODIC_FRAMEWORK.md` - Ergodic calculations and theory
- `simone/SPRINT_03_LOSS_MODELING.md` - Insurance loss generation and modeling

## Technology Stack
- **Primary Language**: Python 3.12+
- **Core Libraries**: NumPy, SciPy, Pandas, Matplotlib, Seaborn
- **Testing**: pytest (Python), jest (TypeScript)
- **Package Management**: uv (Python), npm (TypeScript)
- **Type Checking**: mypy (Python), TypeScript
- **Code Formatting**: black (Python), prettier (TypeScript)
- **Linting**: pylint (Python), eslint (TypeScript)

## Essential Commands

### Python Development
```bash
# Install dependencies
uv sync  # or pip install -e .

# Run tests
pytest
pytest --cov=ergodic_insurance --cov-report=html  # with coverage

# Code quality
black ergodic_insurance  # format code
pylint ergodic_insurance  # lint
mypy ergodic_insurance   # type check

# Run notebooks
jupyter notebook
```

### TypeScript Development (in simone/ directory)
```bash
# Install dependencies
npm install

# Run tests
npm test

# Build and development
npm run build  # compile TypeScript
npm run dev    # watch mode
npm run lint   # lint code
npm run format # format code
```

## Critical Development Rules
1. **Test Coverage**: Maintain >80% test coverage for all new code ✅ **ACHIEVED: 100% coverage**
2. **Type Safety**: All Python code must pass mypy type checking ✅ **ENFORCED**
3. **Documentation**: All public APIs must have comprehensive docstrings ✅ **GOOGLE-STYLE IMPLEMENTED**
4. **Data Validation**: Use Pydantic models for all configuration and parameters ✅ **IMPLEMENTED**
5. **Reproducibility**: All simulations must be seedable for reproducible results ✅ **IMPLEMENTED**
6. **Performance**: Long simulations (100-1000 years) must complete in reasonable time ✅ **VERIFIED**
7. **Version Control**: Never commit directly to main branch - use feature branches ✅ **ENFORCED**
8. **Code Quality**: Run formatters and linters before committing ✅ **AUTOMATED**

### Recent Improvements ✨
- **Enhanced Documentation Standards**: All modules now feature comprehensive Google-style docstrings
- **Professional API Documentation**: Sphinx documentation system configured for automated generation
- **Stochastic Modeling**: Complete implementation of GBM, lognormal volatility, and mean-reversion processes
- **Configuration Management**: Full Pydantic validation with YAML parameter loading
- **Testing Excellence**: Comprehensive test suite with 100% coverage across all modules

## Key Technical Concepts

### Ergodic Theory Application
- **Time Average**: Growth rate experienced by a single entity over time
- **Ensemble Average**: Expected value across many parallel scenarios
- **Key Insight**: These diverge for multiplicative processes (like wealth dynamics)
- **Implication**: Insurance optimal from time-average perspective even when "expensive" by ensemble standards

### Financial Model Parameters
- **Starting Assets**: $10M baseline
- **Asset Turnover**: 0.5-1.5x (revenue per dollar of assets)
- **Operating Margin**: 8% baseline
- **Tax Rate**: 25%
- **Working Capital**: 15-25% of sales

### Stochastic Processes ✨ **NEW**
- **Geometric Brownian Motion (GBM)**: Euler-Maruyama discretization for growth modeling
- **Lognormal Volatility**: Simple revenue shock generation with configurable volatility
- **Mean-Reverting Process**: Ornstein-Uhlenbeck for bounded variable evolution
- **Configurable Parameters**: 15% annual volatility baseline, reproducible with fixed seeds
- **Integration**: Seamless integration with deterministic models, backward compatible

### Loss Modeling
- **Attritional Losses**: High frequency (3-8/year), low severity ($3K-$100K)
- **Large Losses**: Low frequency (0.1-0.5/year), high severity ($500K-$50M)
- **Catastrophic Events**: Separate modeling for extreme tail risks
- **Distributions**: Poisson frequency, Lognormal severity with variance control
- **Correlation**: 0.15-0.35 between operational and financial risks

### Insurance Structure
- **Multi-layer**: Primary ($0-5M), Excess ($5-25M), Higher layers ($25M+)
- **Premium Rates**: Decreasing by layer (1.5% → 0.8% → 0.4%)
- **Optimization Goal**: Maximize ROE subject to <1% ruin probability

## Current Development Status
- ✅ Project structure established and refined
- ✅ Configuration management implemented with Pydantic validation
- ✅ Basic manufacturer model created with comprehensive documentation
- ✅ Loss generation models implemented (ClaimGenerator with Poisson/lognormal)
- ✅ **Stochastic processes implemented** (GBM, lognormal volatility, mean-reversion)
- ✅ **Comprehensive Google-style documentation** across all modules
- ✅ **Sphinx documentation system** set up for professional API docs
- ✅ **100% test coverage** achieved with comprehensive test suite
- 🔄 Ergodic calculations integration in progress
- 📋 Insurance optimization algorithms pending
- 📋 Monte Carlo ensemble engine pending
- 📋 Blog post series development in progress

## Git Configuration
- **User**: Alex Filiakov
- **Email**: alexfiliakov@gmail.com
- **Repository**: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits

## Activity Logging

You have access to the `log_activity` tool. Use it to record your activities after every activity that is relevant for the project. This helps track development progress and understand what has been done.

## When Starting Work
1. Review this file and the sprint documents in `simone/`
2. Check current git status and recent commits
3. Run tests to ensure everything is working
4. Check the todo items in sprint documents for next tasks
5. Use the TodoWrite tool to track your work progress

## Performance Targets
- 1000-year simulations in <1 minute
- 100K Monte Carlo iterations in <10 minutes
- 1M iterations overnight on standard hardware

## Documentation Standards
- Use NumPy-style docstrings for Python functions
- Include type hints for all function parameters
- Write clear, concise comments for complex logic
- Update this file when making significant structural changes

## Contact & Support
- Project Owner: Alex Filiakov
- License: MIT
- For questions about project goals, refer to `simone/00_PLAN.md`
- For implementation details, check sprint documents and architecture.md
